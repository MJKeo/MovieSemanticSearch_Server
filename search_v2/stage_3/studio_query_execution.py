# Search V2 — Stage 3 Studio Endpoint: Query Execution
#
# Takes the LLM's StudioQuerySpec output and runs the appropriate
# lookup, producing an EndpointResult with [0, 1] scores per matched
# movie_id. Works uniformly for both dealbreakers (no restrict set —
# return naturally matched movies) and preferences (restrict set
# provided — return one entry per supplied ID, with 0.0 for
# non-matches).
#
# Two paths, chosen by the LLM's spec:
#
#   Brand path (spec.brand_id set) — direct lookup against
#   `lex.inv_production_brand_postings`. Time-bounded membership
#   (Lucasfilm joined DISNEY in 2012, Twentieth Century Fox's
#   windowed rows, etc.) was applied at ingest, so this path is just
#   a membership read. Returns flat 1.0 per matched movie — see the
#   flat-scoring rationale in
#   .claude/plans/ok-this-all-sounds-elegant-melody.md (IMDB ordering
#   is only prominence-meaningful for Hollywood billing-block films;
#   Japanese alphabetical conventions and European mixed orderings
#   make `first_matching_index` unreliable as a prominence signal).
#
#   Freeform path (spec.freeform_names non-empty) — token intersection
#   over `lex.studio_token`, DF-ceiling filtered against
#   `lex.studio_token_doc_frequency`, producing a set of
#   production_company_ids. Those ids join against
#   `movie_card.production_company_ids` via GIN && to get matched
#   movies. Intersection within a name (all discriminative tokens
#   must hit one company string) avoids the cross-company token
#   false-positive; union across names is OR semantics for the LLM's
#   multiple surface-form candidates.
#
# Precedence: brand path wins when set and non-empty; freeform is the
# backup when brand_id is unset OR when the brand path returned empty
# (rare — means the registry brand has no stamped movies, usually only
# during backfill edge cases).

from __future__ import annotations

from db.postgres import (
    fetch_company_ids_for_tokens,
    fetch_movie_ids_by_brand,
    fetch_movie_ids_by_production_company_ids,
)
from implementation.misc.production_company_text import (
    normalize_company_string,
    tokenize_company_string,
)
from schemas.endpoint_result import EndpointResult
from schemas.studio_translation import StudioQuerySpec
from search_v2.stage_3.result_helpers import build_endpoint_result


# DF-ceiling for the freeform path. Tokens whose `doc_frequency` in
# lex.studio_token_doc_frequency exceeds this count are too common to
# discriminate between companies — dropping them at query time is
# what makes "Warner Bros. Pictures" match the WB companies without
# "pictures" (DF >> ceiling) diluting the intersection. Pinned
# empirically; should be re-derived when the catalog grows materially
# (the v2 plan has the bucket-analysis procedure).
DF_CEILING: int = 323


async def _execute_brand_path(
    brand_id: int,
    restrict_movie_ids: set[int] | None,
) -> dict[int, float]:
    """Brand path — direct membership lookup. Flat 1.0 per match."""
    movie_ids = await fetch_movie_ids_by_brand(brand_id, restrict_movie_ids)
    return {mid: 1.0 for mid in movie_ids}


async def _execute_freeform_path(
    freeform_names: list[str],
    restrict_movie_ids: set[int] | None,
) -> dict[int, float]:
    """Freeform path — per-name token intersection, union across names.

    For each surface form: normalize → tokenize → fetch DF-filtered
    posting lists → intersect per-token company sets. A name that has
    *any* of its tokens DF-dropped or unseen contributes nothing (all
    tokens must participate for the intersection to be meaningful).
    Company sets union across names, then join against
    `movie_card.production_company_ids` produces the final movie set.
    """
    all_company_ids: set[int] = set()
    for name in freeform_names:
        normalized = normalize_company_string(name)
        # Pass already_normalized=True — we just ran the normalizer
        # above, so the tokenizer shouldn't run it again.
        tokens = tokenize_company_string(normalized, already_normalized=True)
        if not tokens:
            continue

        token_to_companies = await fetch_company_ids_for_tokens(tokens, DF_CEILING)

        # Every token must be present AND non-empty for the per-name
        # intersection to be well-defined. A missing token (DF-dropped
        # or never indexed) collapses this name's contribution to
        # empty — we skip it rather than silently treating "missing"
        # as "matches everything".
        per_token_sets = [token_to_companies.get(t) for t in tokens]
        if not all(per_token_sets):
            continue

        intersected = set.intersection(*per_token_sets)
        if intersected:
            all_company_ids |= intersected

    if not all_company_ids:
        return {}

    movie_ids = await fetch_movie_ids_by_production_company_ids(
        all_company_ids, restrict_movie_ids
    )
    return {mid: 1.0 for mid in movie_ids}


async def execute_studio_query(
    spec: StudioQuerySpec,
    *,
    restrict_to_movie_ids: set[int] | None = None,
) -> EndpointResult:
    """Execute one StudioQuerySpec against the brand + freeform paths.

    Single entry point for both dealbreakers and preferences. The
    restrict_to_movie_ids parameter controls output shape:
      - None (dealbreaker path) → one ScoredCandidate per naturally
        matched movie, non-matches omitted.
      - set[int] (preference path) → exactly one ScoredCandidate per
        supplied ID, with 0.0 for non-matches.

    Path precedence:
      1. If spec.brand_id is set, try the brand path. If it returns
         any matches, those are the result.
      2. Otherwise (brand unset or brand-path empty), if spec has
         freeform_names, run the freeform path.
      3. If neither path produces anything, return an empty result —
         "no match is a valid result" per the endpoint spec.

    Args:
        spec: Validated StudioQuerySpec from the step 3 studio LLM.
        restrict_to_movie_ids: Optional candidate-pool restriction.
            Pass the preference's candidate pool to get one entry per
            ID; omit to get the natural match set for dealbreakers.

    Returns:
        EndpointResult with scores ∈ [0, 1] per movie.
    """
    scores_by_movie: dict[int, float] = {}

    if spec.brand_id is not None:
        # ProductionBrand enum carries the int brand_id as an
        # attribute; the DB column is SMALLINT on that int.
        scores_by_movie = await _execute_brand_path(
            spec.brand_id.brand_id, restrict_to_movie_ids
        )

    # Freeform as fallback: runs either when brand_id was unset, OR
    # when the brand path returned empty but the LLM also provided
    # freeform_names. Covers the rare edge case of a registry brand
    # with no stamped movies during backfill.
    if not scores_by_movie and spec.freeform_names:
        scores_by_movie = await _execute_freeform_path(
            spec.freeform_names, restrict_to_movie_ids
        )

    return build_endpoint_result(scores_by_movie, restrict_to_movie_ids)
