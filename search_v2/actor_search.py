# Search V2 — Actor search flow.
#
# Owns the actor path described by Step 0's ActorFlowData. Fires when
# the entire raw query is the name of one or more actors (e.g.
# "Tom Hanks", "Tom Hanks and Meg Ryan").
#
# The flow has three stages:
#
#   1. Per-actor resolution. Each canonical_name is normalized via the
#      shared `normalize_string`, resolved to one or more term_ids via
#      `fetch_phrase_term_ids` (handles hyphen variants and aliasing
#      collisions), and looked up in `lex.inv_actor_postings`. The
#      billing rows are bucketed into four prominence buckets using
#      the sqrt-adaptive zone model in search_v2.actor_zones. Within
#      a single actor, multiple credits in the same movie reduce to
#      MIN bucket number (best billing wins — handles alias variants
#      and multi-role credits like Eddie Murphy in Nutty Professor).
#
#   2. Multi-actor intersection. Only movies where ALL named actors
#      appear are kept. The per-movie bucket assignment uses MAX
#      bucket number across actors — "weakest link" — so if Hanks is
#      a lead (bucket 1) and Harrelson is a minor credit (bucket 4)
#      in the same film, the film lands in bucket 4. This matches the
#      common reading of "tom hanks AND woody harrelson" — a film
#      where both have any presence, ranked by how prominent the
#      weakest of the named actors is.
#
#   3. Within-bucket popularity sort. Matched movie_ids are popularity-
#      sorted DESC (NULLS LAST, movie_id DESC tiebreaker) using the
#      shared sort helper.
#
# Design choice: 4 buckets, not a single ranked list. Unlike studio
# search (one bucket, pure popularity), actor prominence is a strong
# user-facing signal — "tom hanks LEAD roles" reads very differently
# from "tom hanks cameos" and the UI needs to display them as
# distinct groups. The character-franchise tiering model is the
# closest precedent.
#
# No LLM call. Resolution is purely deterministic (name → term_ids →
# billing rows → buckets). The query "tom hanks and meg ryan" needs
# no alias expansion because Step 0 has already committed to which
# spans are actor names.

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from db.postgres import (
    fetch_actor_billing_rows,
    fetch_phrase_term_ids,
    fetch_quality_popularity_signals,
)
from implementation.misc.helpers import normalize_string
from schemas.step_0_flow_routing import ActorFlowData
from search_v2.actor_zones import zone_cutoffs, zone_relative_position
from search_v2.popularity_sort import sort_movie_ids_by_popularity


# In-zone relative-position split for the minor zone. Below the
# midpoint (zp <= 0.5) lands in "has relevance" (bucket 3); above
# (zp > 0.5) lands in "minor / cameo" (bucket 4). Bucket 4 has no
# lower floor — even billing position 200/200 in a 200-cast film
# lands here. See plan witty-kindling-alpaca.md §Bucket definitions.
_MINOR_ZONE_RELEVANCE_SPLIT = 0.5

# Bucket constants — int values are the public surface (lower = more
# prominent). Used by callers to dispatch printers / formatters.
BUCKET_LEAD = 1
BUCKET_MAJOR = 2
BUCKET_RELEVANT = 3
BUCKET_MINOR = 4


@dataclass
class ActorSearchResult:
    """Output of run_actor_search.

    Four disjoint buckets in priority order. Callers concatenate in
    bucket order (bucket_1 + bucket_2 + ... + bucket_4) for a flat
    ranked list. Per-bucket sorts are popularity DESC, NULLS LAST,
    movie_id DESC.

    bucket_1_lead: movie_ids where the worst-billed of the named
        actors lands in the LEAD zone (sqrt-adaptive — positions 1
        to round(0.6 * sqrt(cast_size)), floor 2).
    bucket_2_major: movie_ids where the worst lands in the SUPPORTING
        zone (positions lead_cutoff+1 to round(sqrt(cast_size))).
    bucket_3_relevant: movie_ids where the worst lands in the MINOR
        zone at zp <= 0.5 (top half of the minor zone — still has
        billed-cast relevance, not a cameo).
    bucket_4_minor: movie_ids where the worst lands in the MINOR
        zone at zp > 0.5 (bottom half — cameos, walk-on credits,
        deep-cast appearances).
    """

    bucket_1_lead: list[int] = field(default_factory=list)
    bucket_2_major: list[int] = field(default_factory=list)
    bucket_3_relevant: list[int] = field(default_factory=list)
    bucket_4_minor: list[int] = field(default_factory=list)


async def run_actor_search(
    flow_data: ActorFlowData,
    *,
    limit: int = 100,
) -> ActorSearchResult:
    """Execute the actor flow.

    Resolves each canonical_name to billing rows, assigns a bucket
    per (actor, movie) using the sqrt-adaptive zone model, and
    intersects across actors with weakest-link bucket reduction.
    Within each bucket, movies are popularity-sorted.

    Args:
        flow_data: From Step0Response.to_actor_flow_data(). Carries
            one or more canonical_names Step 0 resolved (e.g.
            ["Tom Hanks"], ["Tom Hanks", "Meg Ryan"]).
        limit: Maximum total rows across all four buckets. Applied
            bucket-by-bucket in priority order — bucket 1 fills
            first, then 2, etc. Default 100.

    Returns:
        ActorSearchResult with four popularity-sorted bucket lists.
        All buckets empty when:
          - flow_data carries no references, or
          - any named actor fails to resolve to any term_ids
            (intersection is empty), or
          - the intersection of movies across all actors is empty.
    """
    references = flow_data.references
    if not references:
        return ActorSearchResult()

    # Stage 1: per-actor resolution → {movie_id: bucket} for each
    # named actor. Fanned out in parallel — each call is one
    # Postgres roundtrip for phrase_term_ids + one for billing rows.
    per_actor_buckets = await asyncio.gather(
        *(_fetch_actor_buckets(ref.canonical_name) for ref in references)
    )

    # Intersection demands all-or-nothing — if any actor resolved
    # to nothing, the intersection is empty.
    if any(not buckets for buckets in per_actor_buckets):
        return ActorSearchResult()

    # Stage 2: intersection across actors with weakest-link reduction.
    intersection_ids = set.intersection(
        *(set(buckets.keys()) for buckets in per_actor_buckets)
    )
    if not intersection_ids:
        return ActorSearchResult()

    # For each movie in the intersection, the assigned bucket is the
    # MAX bucket number across actors. Lower bucket numbers are more
    # prominent, so MAX picks the worst-billed of the named actors.
    movie_bucket: dict[int, int] = {}
    for mid in intersection_ids:
        movie_bucket[mid] = max(buckets[mid] for buckets in per_actor_buckets)

    # Stage 3: bulk popularity fetch + per-bucket sort. One round-trip
    # covering every movie in the intersection, then four cheap sorts.
    popularity = await fetch_quality_popularity_signals(list(intersection_ids))

    buckets_by_id: dict[int, set[int]] = {
        BUCKET_LEAD: set(),
        BUCKET_MAJOR: set(),
        BUCKET_RELEVANT: set(),
        BUCKET_MINOR: set(),
    }
    for mid, b in movie_bucket.items():
        buckets_by_id[b].add(mid)

    result = ActorSearchResult(
        bucket_1_lead=sort_movie_ids_by_popularity(
            buckets_by_id[BUCKET_LEAD], popularity
        ),
        bucket_2_major=sort_movie_ids_by_popularity(
            buckets_by_id[BUCKET_MAJOR], popularity
        ),
        bucket_3_relevant=sort_movie_ids_by_popularity(
            buckets_by_id[BUCKET_RELEVANT], popularity
        ),
        bucket_4_minor=sort_movie_ids_by_popularity(
            buckets_by_id[BUCKET_MINOR], popularity
        ),
    )

    # Cap total rows by trimming from the lowest-priority buckets
    # first. Preserves bucket ordering — a bucket-1 movie is never
    # dropped to make room for a bucket-2 movie.
    _apply_limit(result, limit)
    return result


# ---------------------------------------------------------------------------
# Stage 1 — per-actor resolution.
# ---------------------------------------------------------------------------


async def _fetch_actor_buckets(canonical_name: str) -> dict[int, int]:
    """Resolve one actor → {movie_id: bucket_num}.

    Returns an empty dict when the canonical_name fails to resolve.
    `fetch_phrase_term_ids` is exact-match per normalized phrase, so
    we get at most one term_id back. Hyphen-variant handling lives
    upstream in ingest, where multiple surface variants of the same
    actor name all map to a single canonical term_id via
    `normalize_string` (which preserves hyphens). The actor side of
    that pipeline writes one row per (actor, movie) into
    `lex.inv_actor_postings`, so a well-resolved canonical name
    typically yields one row per appearance.

    Per-movie reduction is MIN bucket number — lower bucket = more
    prominent. The reduction guards against the edge case where
    multiple rows for the same (term_id, movie_id) pair surface
    from ingest (e.g. distinct billing entries for an actor who
    plays multiple credited characters in one film); MIN preserves
    the best billing in that case.
    """
    norm = normalize_string(canonical_name)
    if not norm:
        return {}

    phrase_to_id = await fetch_phrase_term_ids([norm])
    if not phrase_to_id:
        return {}

    term_ids = list(phrase_to_id.values())
    rows = await fetch_actor_billing_rows(term_ids, None)

    # MIN bucket per movie. Lower bucket number = better billing.
    buckets: dict[int, int] = {}
    for movie_id, billing_position, cast_size in rows:
        b = _bucket_for_row(billing_position, cast_size)
        prev = buckets.get(movie_id)
        if prev is None or b < prev:
            buckets[movie_id] = b
    return buckets


# ---------------------------------------------------------------------------
# Bucket assignment.
# ---------------------------------------------------------------------------


def _bucket_for_row(billing_position: int, cast_size: int) -> int:
    """Map (billing_position, cast_size) → bucket 1/2/3/4.

    Uses the shared sqrt-adaptive zone model:
      - Bucket 1 (Lead): zone == "lead"
      - Bucket 2 (Major): zone == "supporting"
      - Bucket 3 (Relevant): minor zone AND zp <= 0.5
      - Bucket 4 (Minor): minor zone AND zp > 0.5

    Bucket 4 has no lower floor — any minor-zone position with
    zp > 0.5 lands here, all the way down to zp = 1.0.
    """
    cutoffs = zone_cutoffs(cast_size)
    if billing_position <= cutoffs.lead_cutoff:
        return BUCKET_LEAD
    if billing_position <= cutoffs.supp_cutoff:
        return BUCKET_MAJOR

    # In the minor zone — split on zp at 0.5. A single-member minor
    # zone (cast_size == supp_cutoff + 1) gets zp = 0.0 from
    # zone_relative_position's span <= 0 branch → lands in bucket 3.
    # That is deliberate: "the only minor" should not be called a
    # cameo.
    zp = zone_relative_position(billing_position, cutoffs.supp_cutoff + 1, cast_size)
    if zp <= _MINOR_ZONE_RELEVANCE_SPLIT:
        return BUCKET_RELEVANT
    return BUCKET_MINOR


# ---------------------------------------------------------------------------
# Limit application.
# ---------------------------------------------------------------------------


def _apply_limit(result: ActorSearchResult, limit: int) -> None:
    """Trim total rows across buckets in lowest-to-highest priority order.

    Mutates `result` in place. Preserves the bucket-priority invariant
    — a bucket-1 movie is never dropped to make room for a bucket-2
    movie. A nonpositive limit clears every bucket.
    """
    buckets = [
        result.bucket_1_lead,
        result.bucket_2_major,
        result.bucket_3_relevant,
        result.bucket_4_minor,
    ]
    if limit <= 0:
        for bucket in buckets:
            bucket.clear()
        return

    remaining = limit
    for bucket in buckets:
        if remaining <= 0:
            bucket.clear()
            continue
        if len(bucket) > remaining:
            del bucket[remaining:]
        remaining -= len(bucket)
