# Search V2 — Attribute search flow.
#
# Backs the `POST /attribute_search` endpoint: a deterministic browse
# path that takes a set of hard attributes (genres, audio languages,
# streaming services, release/runtime/maturity ranges, plus named
# people) and returns the top movies.
#
# No NLP, no LLM, no vector search. Every input is either a closed
# enum / numeric range (handled by MetadataFilters at the SQL layer)
# or a person name resolved against the lexical posting tables.
#
# Ranking model:
#
#   - No people supplied → the whole movie_card table ranked by the
#     neutral 80/20 popularity/reception prior
#     (`fetch_neutral_reranker_seed_ids`), filter-respecting. This is
#     the "browse everything" case and is unchanged.
#
#   - One or more people supplied → the Step 0 person flow's prominence
#     model, reused verbatim. Each person is resolved to a per-movie
#     prominence bucket via the SHARED `fetch_person_buckets`
#     (search_v2.person_search) — role-agnostic, 4 buckets:
#     LEAD / MAJOR / RELEVANT / MINOR (the minor actor zone is split at
#     zp=0.5 into "still relevant" vs "cameo"). The ONLY divergence from
#     Step 0 is how multiple people combine: Step 0 takes the MIN bucket
#     and breaks ties on overlap_count; here we SUM each person's bucket
#     weight so a movie crediting more (or more prominent) people ranks
#     higher.
#
# Bucket → weight: `(BUCKET_MINOR + 1) - bucket`, i.e. LEAD=4, MAJOR=3,
# RELEVANT=2, MINOR=1 (higher = more prominent). Summing strictly
# increasing weights and sorting `(weight DESC, popularity DESC,
# movie_id DESC)` reproduces Step 0's bucket-priority order EXACTLY for
# the single-person case: one term in the sum → weight is constant
# within a bucket and strictly orders the buckets, and the within-bucket
# tie-break (popularity_score then movie_id, NULLS last) is identical to
# Step 0's `_sort_bucket`. So a single person with no metadata filters
# yields the same ordering as the Step 0 person flow.
#
# `role` is intentionally NOT part of this flow — the endpoint no longer
# accepts it, matching Step 0's role-agnostic resolution.
#
# Flow when people are supplied:
#
#   1. Normalize person names; dedupe on the normalized name (the same
#      dedupe Step 0 applies to canonical_name). Empty-after-
#      normalization names are dropped (they contribute no credits)
#      rather than collapsing the whole result — union semantics.
#
#   2. Per-person bucket resolution via `fetch_person_buckets`, fanned
#      out in parallel. `metadata_filters` is threaded into every role
#      lookup so each per-person dict is already filter-respecting.
#
#   3. UNION + SUM of bucket weights across persons → `{movie_id:
#      summed_weight}`.
#
#   4. Fetch popularity signals for the pool and sort by
#      (summed_weight DESC, popularity_score DESC, movie_id DESC);
#      slice to `limit`.
#
# Wire-layer schemas (`PersonInput`, `AttributeSearchBody`) live in
# api/main.py — this module only sees the post-validation
# `PersonSpec` dataclass below so we stay decoupled from Pydantic /
# FastAPI internals.

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from db.postgres import (
    fetch_neutral_reranker_seed_ids,
    fetch_quality_popularity_signals,
)
from implementation.classes.schemas import MetadataFilters
from implementation.misc.helpers import normalize_string
# Reuse Step 0's person resolver and bucket scheme so the two flows
# can't drift — single-person attribute search matches Step 0 exactly.
from search_v2.person_search import BUCKET_MINOR, fetch_person_buckets


# Default cap on results returned to the API. Browse-style endpoint —
# 250 is enough for the UI to render a paginated grid without a
# follow-up call. (Step 0's person flow caps at 100; the ordering is
# identical for the overlapping prefix — this endpoint just surfaces a
# longer tail.)
DEFAULT_ATTRIBUTE_SEARCH_LIMIT = 250


@dataclass(frozen=True, slots=True)
class PersonSpec:
    """One named-person filter, post-validation.

    `name` is the raw name as supplied by the caller (already
    whitespace-stripped at the API boundary; normalization happens
    inside `run_attribute_search`). There is no role — the flow is
    role-agnostic, mirroring the Step 0 person search.
    """
    name: str


def _bucket_weight(bucket: int) -> int:
    """Map a Step 0 prominence bucket → an additive weight.

    Buckets are 1-based with lower = more prominent (LEAD=1 … MINOR=4).
    We invert to `(BUCKET_MINOR + 1) - bucket` so higher = more
    prominent (LEAD=4 … MINOR=1). The weights are strictly decreasing
    in bucket rank, which is what lets a single-person summed-weight
    sort reproduce Step 0's bucket-priority ordering exactly.
    """
    return (BUCKET_MINOR + 1) - bucket


async def _rank_people_pool(
    pool: dict[int, int],
    *,
    limit: int,
) -> list[int]:
    """Order a `{movie_id: summed_weight}` pool into ranked movie_ids.

    Sort key: (summed_weight DESC, popularity_score DESC, movie_id
    DESC), with popularity NULLS-last. This is byte-for-byte the same
    within-tier ordering Step 0's `_sort_bucket` uses (popularity_score
    only — reception is not consulted — and the leading
    "1 if popularity set else 0" flag for NULLS-last under reverse
    sort), with the summed weight prepended as the tier key.
    """
    movie_ids = list(pool.keys())
    signals = await fetch_quality_popularity_signals(movie_ids)

    def sort_key(movie_id: int) -> tuple[int, int, float, int]:
        # signals maps movie_id -> (popularity_score, reception_score);
        # Step 0 sorts on popularity_score only, so we take index 0.
        popularity = signals.get(movie_id, (None, None))[0]
        weight = pool[movie_id]
        if popularity is None:
            return (weight, 0, 0.0, movie_id)
        return (weight, 1, popularity, movie_id)

    movie_ids.sort(key=sort_key, reverse=True)
    return movie_ids[:limit]


async def run_attribute_search(
    *,
    people: list[PersonSpec],
    metadata_filters: Optional[MetadataFilters],
    limit: int = DEFAULT_ATTRIBUTE_SEARCH_LIMIT,
) -> list[int]:
    """Orchestrate the attribute-search flow → ranked movie_ids.

    Args:
      people: Zero or more `PersonSpec` filters. Multiple persons are
        UNIONED and their per-movie bucket weights summed — a movie
        crediting any supplied person qualifies, and crediting more of
        them (or more prominently) ranks higher. A single person (with
        no metadata filters) yields the same ordering as the Step 0
        person flow.
      metadata_filters: Optional hard filters (genres, languages,
        streaming services, release_date / runtime / maturity ranges).
        Applied at the SQL layer inside every posting lookup AND inside
        the no-people ranking query.
      limit: Cap on returned movie_ids (default 250).

    Returns:
      List of `movie_id`s ranked by summed prominence weight
      (descending), with popularity_score as the within-tier
      tie-breaker. Empty list when no supplied name resolves to any
      filter-eligible credit. With no people supplied, returns the
      catalog ranked by the neutral 80/20 prior.
    """
    # No-person fast path: rank the whole movie_card table by the
    # neutral 80/20 prior (filter-respecting). Unchanged behavior.
    if not people:
        return await fetch_neutral_reranker_seed_ids(
            limit=limit, metadata_filters=metadata_filters,
        )

    # Normalize + dedupe on the normalized name — the same dedupe Step 0
    # applies to canonical_name. A name that normalizes to empty
    # contributes no credits and is skipped (union semantics: it must
    # NOT collapse the whole result). We keep the raw name to hand to
    # `fetch_person_buckets`, which re-normalizes internally exactly as
    # Step 0 does, so resolution is identical.
    names: list[str] = []
    seen: set[str] = set()
    for spec in people:
        normalized = normalize_string(spec.name)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        names.append(spec.name)

    # Every supplied name was blank after normalization — no credits.
    if not names:
        return []

    # Per-person bucket resolution, in parallel, via the shared Step 0
    # resolver (role-agnostic across all five credit tables).
    per_person = await asyncio.gather(
        *(
            fetch_person_buckets(name, metadata_filters=metadata_filters)
            for name in names
        )
    )

    # UNION + SUM of bucket weights across persons. A movie's score is
    # the sum of each person's best-bucket weight on it; people who
    # don't credit the movie contribute 0 (absent from their dict).
    pool: dict[int, int] = {}
    for person_buckets in per_person:
        for movie_id, bucket in person_buckets.items():
            pool[movie_id] = pool.get(movie_id, 0) + _bucket_weight(bucket)

    if not pool:
        return []

    return await _rank_people_pool(pool, limit=limit)
