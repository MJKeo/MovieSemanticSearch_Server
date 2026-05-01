# Search V2 — Stage 4 Studio Endpoint: Query Execution
#
# Takes a StudioQuerySpec (one or more StudioRefs + a scoring_method)
# and produces an EndpointResult with [0, 1] scores per movie. Single
# entry point for both carvers (no restrict set — natural match set,
# scores compressed to [0.5, 1.0]) and qualifiers (restrict set
# provided — one entry per supplied ID, raw scores in [0, 1]).
#
# Resolution paths per ref (mirror of the previous schema):
#
#   Brand path — direct lookup against
#   `lex.inv_production_brand_postings` keyed by ProductionBrand.brand_id.
#   Time-bounded membership and renames are applied at ingest, so this
#   is just a posting read. Flat 1.0 per matched movie.
#
#   Freeform path — DF-filtered token intersection against
#   `lex.studio_token`, then GIN `&&` against
#   `movie_card.production_company_ids`. Per-name intersection (all
#   discriminative tokens must hit one company string) avoids the
#   cross-company token false-positive; cross-name union gives OR
#   semantics across the LLM's surface-form variants for one studio.
#
# Scoring-method logic:
#
#   ANY (any-of / OR) — single batched fetch across the whole call.
#   All brand_ids from refs that have a brand are unioned into one
#   posting query. All freeform_names from refs that have NO brand
#   (per the brand-wins edge case) are flattened into one token
#   resolution. Results are unioned; binary scoring means the union
#   is by construction the per-movie max.
#
#   ALL (all-of / AND) — per-ref parallel fetch via asyncio.gather,
#   then evenly weighted mean across all refs. Misses count as 0 in
#   the numerator, denominator is the ref count — so a movie matching
#   k of N refs scores k/N. Refs with the same brand_id collapse to
#   one ref before the gather (per-call brand dedup).
#
# Edge cases (both modes):
#   - Ref with neither brand nor freeform_names is rejected before any
#     fetch runs.
#   - Ref with both brand and freeform_names: brand wins, names are
#     dropped — no fall-through to freeform when the brand posting is
#     empty (deliberate; the LLM is expected to commit at the ref level
#     rather than rely on executor fallback).
#
# Scoring band: carver scores compress to [0.5, 1.0] via
# compress_to_dealbreaker_floor — only non-zero raw scores are lifted,
# zeros are dropped. Qualifier scores stay raw in [0, 1].

from __future__ import annotations

import asyncio

from db.postgres import (
    fetch_company_ids_for_tokens,
    fetch_movie_ids_by_brands,
    fetch_movie_ids_by_production_company_ids,
)
from implementation.misc.production_company_text import (
    normalize_company_string,
    tokenize_company_string,
)
from schemas.endpoint_result import EndpointResult
from schemas.enums import ScoringMethod
from schemas.studio_translation import StudioQuerySpec, StudioRef
from search_v2.endpoint_fetching.result_helpers import (
    build_endpoint_result,
    compress_to_dealbreaker_floor,
)


# DF-ceiling for the freeform path. Tokens whose `doc_frequency` in
# lex.studio_token_doc_frequency exceeds this count are too common to
# discriminate between companies — dropping them at query time is
# what makes "Warner Bros. Pictures" match the WB companies without
# "pictures" (DF >> ceiling) diluting the intersection. Pinned
# empirically; should be re-derived when the catalog grows materially.
DF_CEILING: int = 323


# ---------------------------------------------------------------------------
# Path executors — each returns {movie_id: 1.0} for matched movies.
# ---------------------------------------------------------------------------


async def _execute_brand_path(
    brand_ids: list[int],
    restrict_movie_ids: set[int] | None,
) -> dict[int, float]:
    """Direct posting-list lookup for one or more brand_ids. Empty input
    short-circuits without a DB round trip."""
    if not brand_ids:
        return {}
    movie_ids = await fetch_movie_ids_by_brands(brand_ids, restrict_movie_ids)
    return {mid: 1.0 for mid in movie_ids}


async def _execute_freeform_path(
    freeform_names: list[str],
    restrict_movie_ids: set[int] | None,
) -> dict[int, float]:
    """Token-intersection freeform path.

    Each name → normalize → tokenize. All tokens across all names go
    into one batched DF-filtered fetch (1 round trip regardless of how
    many names). Per-name intersection then runs in Python over the
    shared response — a name with any DF-dropped or unseen token
    contributes nothing (must NOT silently treat "missing" as
    "matches everything"). Cross-name union gives OR semantics across
    surface-form variants.

    When called from ANY aggregation, the input list may contain
    surface forms drawn from multiple refs that didn't have a brand
    set; cross-ref union is identical in semantics to cross-variant
    union, so flattening is safe — both fold into the same OR.
    """
    if not freeform_names:
        return {}

    # Phase 1: per-name tokenize, preserve grouping for intersection,
    # collect the flat token set for the batched fetch.
    per_name_tokens: list[list[str]] = []
    all_tokens: set[str] = set()
    for name in freeform_names:
        normalized = normalize_company_string(name)
        # already_normalized=True: skip the tokenizer's redundant
        # second normalize pass.
        tokens = tokenize_company_string(normalized, already_normalized=True)
        if not tokens:
            continue
        per_name_tokens.append(tokens)
        all_tokens.update(tokens)
    if not all_tokens:
        return {}

    # Phase 2: single DF-filtered token → company_ids fetch. sorted()
    # gives deterministic query text (stable plan cache + reproducible
    # logs).
    token_to_companies = await fetch_company_ids_for_tokens(
        sorted(all_tokens), DF_CEILING
    )

    # Phase 3: per-name intersection over the shared response. A
    # missing key collapses the name to empty — DO NOT treat missing
    # as wildcard.
    all_company_ids: set[int] = set()
    for tokens in per_name_tokens:
        per_token_sets = [token_to_companies.get(t) for t in tokens]
        if not all(per_token_sets):
            continue
        all_company_ids |= set.intersection(*per_token_sets)

    if not all_company_ids:
        return {}

    # Phase 4: GIN `&&` join against movie_card.production_company_ids.
    movie_ids = await fetch_movie_ids_by_production_company_ids(
        all_company_ids, restrict_movie_ids
    )
    return {mid: 1.0 for mid in movie_ids}


# ---------------------------------------------------------------------------
# Per-ref resolver — used by ALL mode (one fetch per ref, gathered).
# ---------------------------------------------------------------------------


async def _resolve_single_ref(
    ref: StudioRef,
    restrict_movie_ids: set[int] | None,
) -> dict[int, float]:
    """Resolve one StudioRef to a {movie_id: 1.0} score map.

    Brand wins when set: freeform_names on the same ref are dropped
    (no fall-through). When brand is unset, the ref's freeform_names
    runs the freeform path.
    """
    if ref.brand is not None:
        return await _execute_brand_path(
            [ref.brand.brand_id], restrict_movie_ids
        )
    if ref.freeform_names:
        return await _execute_freeform_path(
            ref.freeform_names, restrict_movie_ids
        )
    # Caller already filters out refs with neither set; reachable only
    # if the upstream filter is bypassed.
    return {}


# ---------------------------------------------------------------------------
# Scoring-method dispatchers.
# ---------------------------------------------------------------------------


async def _execute_any(
    refs: list[StudioRef],
    restrict_movie_ids: set[int] | None,
) -> dict[int, float]:
    """ANY combine — single batched fetch across all refs, per-movie max.

    Splits refs into the brand and freeform pools (brand wins per ref)
    and runs the two paths concurrently. The pools are independent
    posting-list reads, so asyncio.gather across them is a real
    wall-clock win when both are populated.

    Brand_ids dedupe via set; if two refs both routed to brand=disney,
    the SQL `ANY` predicate sees the value once. Freeform names are
    flattened across refs — cross-ref OR is the same operation as
    cross-variant OR within one ref, so a single freeform fetch
    covers them all.
    """
    brand_ids: set[int] = set()
    freeform_names: list[str] = []
    for ref in refs:
        if ref.brand is not None:
            brand_ids.add(ref.brand.brand_id)
            continue
        if ref.freeform_names:
            freeform_names.extend(ref.freeform_names)

    brand_scores, freeform_scores = await asyncio.gather(
        _execute_brand_path(sorted(brand_ids), restrict_movie_ids),
        _execute_freeform_path(freeform_names, restrict_movie_ids),
    )

    # Both paths score 1.0 per match, so a set union of their key sets
    # is equivalent to per-movie max(brand_score, freeform_score).
    matched = brand_scores.keys() | freeform_scores.keys()
    return {mid: 1.0 for mid in matched}


async def _execute_all(
    refs: list[StudioRef],
    restrict_movie_ids: set[int] | None,
) -> dict[int, float]:
    """ALL combine — per-ref parallel fetches, evenly weighted mean.

    Refs sharing a brand_id collapse to one ref before gather: under
    AND semantics, the same brand listed twice would otherwise inflate
    both numerator AND denominator without changing per-movie scores —
    correct numerically but wasteful, and makes counting refs unreliable
    for downstream observability. Freeform-only refs are not deduped
    (their content is opaque at this layer).

    Denominator is the post-dedup ref count. A movie that hits k of N
    refs scores k/N. Misses contribute 0 — the AND/all-of semantic
    means a movie has to clear every named studio for full credit.
    """
    # Brand-level dedup. Order-preserving so the gather order remains
    # deterministic for logging / debugging.
    seen_brand_ids: set[int] = set()
    deduped_refs: list[StudioRef] = []
    for ref in refs:
        if ref.brand is not None:
            if ref.brand.brand_id in seen_brand_ids:
                continue
            seen_brand_ids.add(ref.brand.brand_id)
        deduped_refs.append(ref)

    per_ref_scores = await asyncio.gather(
        *(
            _resolve_single_ref(ref, restrict_movie_ids)
            for ref in deduped_refs
        )
    )

    # Candidate set for averaging: in qualifier mode, the restrict pool
    # is authoritative (every supplied ID gets a score). In carver mode,
    # the natural set is the union of every ref's hit set — movies
    # absent from every ref score 0/N = 0 and would be dropped by the
    # zero-filter downstream anyway, so we elide them here.
    if restrict_movie_ids is not None:
        candidate_ids: set[int] = restrict_movie_ids
    else:
        candidate_ids = set().union(*(s.keys() for s in per_ref_scores))

    n = len(deduped_refs)
    return {
        mid: sum(s.get(mid, 0.0) for s in per_ref_scores) / n
        for mid in candidate_ids
    }


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


async def execute_studio_query(
    spec: StudioQuerySpec,
    *,
    restrict_to_movie_ids: set[int] | None = None,
) -> EndpointResult:
    """Execute one StudioQuerySpec.

    The restrict_to_movie_ids parameter controls output shape:
      - None (carver path) → one ScoredCandidate per naturally matched
        movie. Non-zero raw scores are lifted into [0.5, 1.0] via
        compress_to_dealbreaker_floor; zero-score movies are dropped.
      - set[int] (qualifier path) → exactly one ScoredCandidate per
        supplied ID, with raw [0, 1] scores. Movies absent from the
        match set score 0.0.

    Args:
        spec: Validated StudioQuerySpec from the Step 4 studio LLM.
        restrict_to_movie_ids: Optional candidate-pool restriction.

    Returns:
        EndpointResult with per-movie scores.
    """
    # Empty restrict pool: nothing to score, skip the DB round trip.
    if restrict_to_movie_ids is not None and not restrict_to_movie_ids:
        return EndpointResult()

    # Reject refs with neither brand nor freeform_names BEFORE any
    # fetch runs. The schema permits this combination (no model
    # validator) so the executor is the gate.
    valid_refs = [
        ref
        for ref in spec.studios
        if ref.brand is not None or ref.freeform_names
    ]
    if not valid_refs:
        return build_endpoint_result({}, restrict_to_movie_ids)

    if spec.scoring_method == ScoringMethod.ANY:
        scores_by_movie = await _execute_any(valid_refs, restrict_to_movie_ids)
    else:  # ALL
        scores_by_movie = await _execute_all(
            valid_refs, restrict_to_movie_ids
        )

    # Carver compression: raw [0, 1] → [0.5, 1.0]. Zeros drop out so a
    # carver call's match set carries only positive endorsements.
    if restrict_to_movie_ids is None:
        scores_by_movie = {
            mid: compress_to_dealbreaker_floor(score)
            for mid, score in scores_by_movie.items()
            if score > 0.0
        }

    return build_endpoint_result(scores_by_movie, restrict_to_movie_ids)
