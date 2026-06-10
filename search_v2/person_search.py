# Search V2 — Person search flow.
#
# Owns the person path described by Step 0's PersonFlowData. Fires
# when the entire raw query is the name of one or more people in any
# credited filmmaking role (actor, director, writer, producer,
# composer). The executor is role-agnostic: it unions matches across
# all five role postings tables so a query like "david attenborough"
# (writer-only on most credits) surfaces just as well as "tom hanks"
# (actor-only).
#
# The flow has three stages:
#
#   1. Per-person resolution. Each canonical_name is normalized via
#      the shared `normalize_string`, resolved to a term_id via
#      `fetch_phrase_term_ids`, and looked up in parallel against all
#      five PEOPLE_POSTING_TABLES. Per-(person, movie) bucket
#      assignment is the MIN bucket across roles — actor credits use
#      the sqrt-adaptive zone model (buckets 1–4), and non-actor
#      credits (director / writer / producer / composer) carry no
#      prevalence data so they uniformly land in bucket 1 (LEAD).
#      Best credit wins.
#
#   2. Multi-person UNION. A movie counts as a match whenever any
#      named person has any credit on it. For each matched movie we
#      track:
#        - bucket: MIN bucket across the named people who appear on
#          it. "Best credit by any named person" defines the bucket.
#        - overlap_count: how many of the N named people appear on
#          this movie. Used as the primary within-bucket sort key
#          so the intersection of all named people surfaces above
#          single-person matches inside the same prominence tier.
#
#   3. Within-bucket sort. Movies in the same bucket sort by
#      (overlap_count DESC, popularity DESC, movie_id DESC). For the
#      common single-person case overlap_count is always 1 and the
#      sort collapses to pure popularity DESC.
#
# Design choice: 4 buckets, not a single ranked list. Unlike studio
# search (one bucket, pure popularity), person prominence is a
# strong user-facing signal — "tom hanks LEAD roles" reads very
# differently from "tom hanks cameos" and the UI needs to display
# them as distinct groups. The character-franchise tiering model is
# the closest precedent.
#
# No preferred role. The executor never weights actor credits above
# director credits or vice versa. The only place role matters is
# inside the actor table, where billing_position / cast_size carry
# prevalence data the other role tables don't have.
#
# No LLM call. Resolution is purely deterministic (name → term_id →
# role postings → buckets). Step 0 has already committed to which
# spans are person names.

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field

from db.postgres import (
    PEOPLE_POSTING_TABLES,
    PostingTable,
    fetch_actor_billing_rows,
    fetch_movie_ids_by_term_ids,
    fetch_phrase_term_ids,
    fetch_quality_popularity_signals,
)
from implementation.classes.schemas import MetadataFilters
from implementation.misc.helpers import normalize_string
from schemas.step_0_flow_routing import PersonFlowData
from search_v2.actor_zones import zone_cutoffs, zone_relative_position


# In-zone relative-position split for the minor zone. Below the
# midpoint (zp <= 0.5) lands in "has relevance" (bucket 3); above
# (zp > 0.5) lands in "minor / cameo" (bucket 4). Bucket 4 has no
# lower floor — even billing position 200/200 in a 200-cast film
# lands here.
_MINOR_ZONE_RELEVANCE_SPLIT = 0.5

# Bucket constants — int values are the public surface (lower = more
# prominent). Used by callers to dispatch printers / formatters.
BUCKET_LEAD = 1
BUCKET_MAJOR = 2
BUCKET_RELEVANT = 3
BUCKET_MINOR = 4

# Invariant: buckets are contiguous and strictly increasing with
# BUCKET_MINOR the largest (least-prominent) value. `search_v2/
# attribute_search.py` inverts these into additive weights via
# `(BUCKET_MINOR + 1) - bucket`, which only stays monotonic and
# positive while BUCKET_MINOR is the max. If a less-prominent bucket is
# ever added, bump BUCKET_MINOR to match (or update that weight map).
assert BUCKET_LEAD < BUCKET_MAJOR < BUCKET_RELEVANT < BUCKET_MINOR

# Non-actor role tables. Postings carry no billing_position /
# cast_size, so any credit in these tables is treated as a fully
# prominent "named creative" credit and lands in BUCKET_LEAD. Order
# doesn't matter — the per-person reducer picks MIN bucket across
# roles anyway.
_NON_ACTOR_ROLE_TABLES: list[PostingTable] = [
    t for t in PEOPLE_POSTING_TABLES if t is not PostingTable.ACTOR
]


@dataclass
class PersonSearchResult:
    """Output of run_person_search.

    Four disjoint buckets in priority order. Callers concatenate in
    bucket order (bucket_1 + bucket_2 + ... + bucket_4) for a flat
    ranked list. Per-bucket sort is (overlap_count DESC, popularity
    DESC, movie_id DESC).

    bucket_1_lead: movie_ids where the best credit by any named
        person lands in the LEAD actor zone (sqrt-adaptive cutoff)
        OR the person carries any director / writer / producer /
        composer credit on that movie (non-actor roles uniformly
        bucket 1 by design — no prevalence data exists for them).
    bucket_2_major: movie_ids where the best credit lands in the
        SUPPORTING actor zone.
    bucket_3_relevant: movie_ids where the best credit lands in the
        MINOR actor zone at zp <= 0.5 (still has billed-cast
        relevance, not a cameo).
    bucket_4_minor: movie_ids where the best credit lands in the
        MINOR actor zone at zp > 0.5 (cameos, walk-ons, deep-cast).
    """

    bucket_1_lead: list[int] = field(default_factory=list)
    bucket_2_major: list[int] = field(default_factory=list)
    bucket_3_relevant: list[int] = field(default_factory=list)
    bucket_4_minor: list[int] = field(default_factory=list)


async def run_person_search(
    flow_data: PersonFlowData,
    *,
    limit: int = 100,
    metadata_filters: MetadataFilters | None = None,
) -> PersonSearchResult:
    """Execute the person flow.

    Resolves each canonical_name to credits across all five role
    tables, assigns a bucket per (person, movie) (MIN across roles),
    unions across people, and sorts each bucket by overlap_count and
    popularity.

    Args:
        flow_data: From Step0Response.to_person_flow_data(). Carries
            one or more canonical_names Step 0 resolved (e.g.
            ["David Attenborough"], ["Spielberg", "John Williams"]).
        limit: Maximum total rows across all four buckets. Applied
            bucket-by-bucket in priority order — bucket 1 fills
            first, then 2, etc. Default 100.
        metadata_filters: Optional UI hard filters. Threaded into
            every role-table fetch so filtered movies are dropped at
            the candidate-generation layer rather than post-hoc.

    Returns:
        PersonSearchResult with four sorted bucket lists. All
        buckets empty when flow_data carries no references or when
        every named person fails to resolve to any credits.
    """
    if not flow_data.references:
        return PersonSearchResult()

    # Dedupe references on normalized canonical_name. Step 0 occasionally
    # emits the same person twice (e.g. on a query like "tom hanks, tom
    # hanks and meg ryan" or a copy-paste artifact); the previous
    # intersection mode was robust to this because set ∩ self == self,
    # but the new union + overlap_count semantics would inflate
    # overlap_count and rank single-person matches above genuine
    # multi-person intersections inside the same bucket. Dedupe on the
    # normalized form so "Tom Hanks" and "tom  hanks" collapse together
    # the same way `fetch_person_buckets` will resolve them.
    references = []
    seen_norm: set[str] = set()
    for ref in flow_data.references:
        norm = normalize_string(ref.canonical_name)
        if not norm or norm in seen_norm:
            continue
        seen_norm.add(norm)
        references.append(ref)
    if not references:
        return PersonSearchResult()

    # Stage 1: per-person resolution → {movie_id: bucket} for each
    # named person. Fanned out in parallel — each call issues one
    # phrase_term_ids lookup plus one fetch per role table.
    per_person_buckets = await asyncio.gather(
        *(
            fetch_person_buckets(
                ref.canonical_name, metadata_filters=metadata_filters,
            )
            for ref in references
        )
    )

    # Stage 2: UNION across people. A movie counts as a match when
    # any named person has any credit on it. We track per-movie:
    #   - bucket: MIN across the people who appear (best credit by
    #     any named person)
    #   - overlap_count: how many of the named people appear on this
    #     movie. Pre-computed as the primary within-bucket sort key
    #     so the intersection of all named people surfaces above
    #     single-person matches inside the same bucket.
    movie_bucket: dict[int, int] = {}
    movie_overlap: dict[int, int] = {}
    for buckets in per_person_buckets:
        for movie_id, bucket in buckets.items():
            prev = movie_bucket.get(movie_id)
            if prev is None or bucket < prev:
                movie_bucket[movie_id] = bucket
            movie_overlap[movie_id] = movie_overlap.get(movie_id, 0) + 1

    if not movie_bucket:
        return PersonSearchResult()

    # Stage 3: bulk popularity fetch + per-bucket sort. One round-trip
    # covering every matched movie, then four cheap sorts.
    popularity = await fetch_quality_popularity_signals(list(movie_bucket.keys()))

    buckets_by_id: dict[int, list[int]] = {
        BUCKET_LEAD: [],
        BUCKET_MAJOR: [],
        BUCKET_RELEVANT: [],
        BUCKET_MINOR: [],
    }
    for mid, b in movie_bucket.items():
        buckets_by_id[b].append(mid)

    result = PersonSearchResult(
        bucket_1_lead=_sort_bucket(buckets_by_id[BUCKET_LEAD], movie_overlap, popularity),
        bucket_2_major=_sort_bucket(buckets_by_id[BUCKET_MAJOR], movie_overlap, popularity),
        bucket_3_relevant=_sort_bucket(buckets_by_id[BUCKET_RELEVANT], movie_overlap, popularity),
        bucket_4_minor=_sort_bucket(buckets_by_id[BUCKET_MINOR], movie_overlap, popularity),
    )

    # Cap total rows by trimming from the lowest-priority buckets
    # first. Preserves bucket ordering — a bucket-1 movie is never
    # dropped to make room for a bucket-2 movie.
    _apply_limit(result, limit)
    return result


# ---------------------------------------------------------------------------
# Stage 1 — per-person resolution across all five role tables.
# ---------------------------------------------------------------------------


async def fetch_person_buckets(
    canonical_name: str,
    *,
    metadata_filters: MetadataFilters | None = None,
) -> dict[int, int]:
    """Resolve one person → {movie_id: bucket_num} across all roles.

    Shared with `/attribute_search`'s people path
    (`search_v2.attribute_search`) so the two flows resolve a person to
    the exact same prominence buckets — single-person attribute search
    matches the Step 0 person flow by construction. Keep this resolver
    role-agnostic; callers that need to combine people own that logic.

    Queries every PEOPLE_POSTING_TABLES entry in parallel for the
    person's resolved term_id. Returns an empty dict when the
    canonical_name fails to resolve.

    Per-movie reduction is MIN bucket — lower bucket = more
    prominent. The reduction lets a person who appears as a lead
    actor AND a producer on the same movie land at the actor's lead
    bucket rather than the producer's default bucket 1. (In practice
    both default to bucket 1 here, but the reducer makes the rule
    explicit and survives any future bucket-policy change for
    non-actor roles.)
    """
    norm = normalize_string(canonical_name)
    if not norm:
        return {}

    phrase_to_id = await fetch_phrase_term_ids([norm])
    if not phrase_to_id:
        return {}

    term_ids = list(phrase_to_id.values())

    # Fan out per-role queries. Actor uses billing-aware fetch so we
    # can compute prominence buckets; the other four role tables
    # carry no billing columns and surface via the generic movie-id
    # fetcher.
    actor_rows_task = fetch_actor_billing_rows(
        term_ids, None, metadata_filters=metadata_filters,
    )
    non_actor_id_tasks = [
        fetch_movie_ids_by_term_ids(
            table, term_ids, metadata_filters=metadata_filters,
        )
        for table in _NON_ACTOR_ROLE_TABLES
    ]
    actor_rows, *non_actor_id_sets = await asyncio.gather(
        actor_rows_task, *non_actor_id_tasks,
    )

    # Walk results. Actor rows become bucketed via the sqrt-adaptive
    # zone model; non-actor role hits all collapse to BUCKET_LEAD
    # since those tables carry no prevalence data.
    buckets: dict[int, int] = {}

    for movie_id, billing_position, cast_size in actor_rows:
        b = _bucket_for_actor_row(billing_position, cast_size)
        prev = buckets.get(movie_id)
        if prev is None or b < prev:
            buckets[movie_id] = b

    # Non-actor role hits all land at BUCKET_LEAD (== 1). MIN reduction
    # against any actor-side bucket already present must replace a
    # worse (higher-numbered) actor bucket with the better non-actor
    # one. setdefault would be wrong here — it skips when the key
    # exists, leaving a deep-cast actor bucket in place even when the
    # person also directed / wrote / produced / scored the film.
    # Concrete case: Tarantino on Reservoir Dogs (director bucket 1 +
    # minor-cast cameo) must end up in bucket 1, not the cameo bucket.
    for movie_ids in non_actor_id_sets:
        for movie_id in movie_ids:
            prev = buckets.get(movie_id)
            if prev is None or BUCKET_LEAD < prev:
                buckets[movie_id] = BUCKET_LEAD

    return buckets


# ---------------------------------------------------------------------------
# Bucket assignment for actor billing rows.
# ---------------------------------------------------------------------------


def _bucket_for_actor_row(billing_position: int, cast_size: int) -> int:
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
# Within-bucket sort.
# ---------------------------------------------------------------------------


def _sort_bucket(
    movie_ids: Iterable[int],
    overlap_count: dict[int, int],
    popularity: dict[int, tuple[float | None, float | None]],
) -> list[int]:
    """Sort movie_ids by (overlap_count DESC, popularity DESC, movie_id DESC).

    Overlap count is the primary within-bucket key so multi-person
    matches surface above single-person matches in the same
    prominence tier. NULLS-LAST under DESC is encoded the same way
    `popularity_sort.sort_movie_ids_by_popularity` does it — a
    leading (1 if popularity is set, else 0) element on each key
    tuple, sorted under `reverse=True`.
    """

    def _key(mid: int) -> tuple[int, int, float, int]:
        pop = popularity.get(mid, (None, None))[0]
        overlap = overlap_count.get(mid, 0)
        if pop is None:
            return (overlap, 0, 0.0, mid)
        return (overlap, 1, pop, mid)

    return sorted(movie_ids, key=_key, reverse=True)


# ---------------------------------------------------------------------------
# Limit application.
# ---------------------------------------------------------------------------


def _apply_limit(result: PersonSearchResult, limit: int) -> None:
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
