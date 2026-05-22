# Search V2 — Studio search flow.
#
# Owns the studio path described by Step 0's StudioFlowData. Fires when
# the entire raw query is the name of one or more film studios /
# production companies (e.g. "Pixar", "A24 and Neon", "Warner Bros and
# Universal").
#
# The flow has three stages:
#
#   1. Query translation (LLM). The Step 0 canonical_name list is fed
#      to the existing Step 3 studio translator
#      (search_v2/endpoint_fetching/studio_query_generation.py). One LLM
#      call covers every named studio — the translator's prompt already
#      handles "which distinct studios are named" reasoning and emits
#      one StudioRef per studio (brand for registry umbrellas, freeform
#      names for sub-labels / long-tail). LLM failure soft-degrades to a
#      deterministic freeform spec using the raw canonical names.
#
#   2. Posting-list lookup. The translated StudioQuerySpec is run
#      through execute_studio_query in ANY mode — we want every movie
#      matched by ANY of the named studios, with no co-production /
#      match-coverage tiering. The studio executor handles both brand
#      posting reads and DF-filtered freeform token intersection.
#
#   3. Popularity sort. Matched movie_ids are popularity-sorted DESC
#      (NULLS LAST, movie_id DESC tiebreaker) — same convention the
#      franchise flows use within their tiers. A single popularity
#      bulk fetch covers every matched movie.
#
# Design choice: single tier, pure popularity sort. Match coverage
# (movies appearing in multiple of the named studios) is NOT used as a
# secondary tier — co-productions are rare enough that the bucketing
# overhead is not worth the relevance signal. If product evidence later
# shows co-productions should rank above single-studio matches, this is
# the place to add a count-based bucket layer.

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from db.postgres import fetch_quality_popularity_signals
from implementation.classes.schemas import MetadataFilters
from schemas.enums import ScoringMethod
from schemas.step_0_flow_routing import StudioFlowData
from schemas.studio_translation import StudioQuerySpec, StudioRef
# Reuse the Step-3 category handler's LLM config so the two callers
# of generate_studio_query stay in lockstep on model + reasoning
# settings — if the handler upgrades models, this caller follows.
from search_v2.endpoint_fetching.category_handlers.handler import (
    HANDLER_LLM_KWARGS,
    HANDLER_LLM_MODEL,
    HANDLER_LLM_PROVIDER,
)
from search_v2.endpoint_fetching.studio_query_generation import (
    generate_studio_query,
)
from search_v2.endpoint_fetching.studio_query_execution import (
    execute_studio_query,
)
from search_v2.popularity_sort import sort_movie_ids_by_popularity

logger = logging.getLogger(__name__)


@dataclass
class StudioSearchResult:
    """Output of run_studio_search.

    ranked: movie_ids matched by any of the named studios, sorted by
        popularity_score DESC (NULLS LAST, movie_id DESC tiebreaker).
        Single flat list — no tiers, no per-movie score. The
        bucket-then-popularity ordering used by the franchise flows
        reduces to "everyone in one tier, popularity-sorted" here, so
        we expose just the ranked list.
    """

    ranked: list[int] = field(default_factory=list)


async def run_studio_search(
    flow_data: StudioFlowData,
    *,
    limit: int = 100,
    metadata_filters: MetadataFilters | None = None,
) -> StudioSearchResult:
    """Execute the studio flow.

    Translates the Step 0 studio canonical_names into a StudioQuerySpec
    via the existing Step 3 studio LLM, runs the spec through the
    posting-list executor in ANY mode, and returns the matched movies
    sorted by popularity.

    Args:
        flow_data: From Step0Response.to_studio_flow_data(). Carries one
            or more canonical_names Step 0 resolved (e.g. ["Pixar"],
            ["A24", "Neon"]).
        limit: Maximum total rows. Default 100.

    Returns:
        StudioSearchResult with a single popularity-sorted ranked list.
        Empty when no canonical_name resolves to any studio.
    """
    canonical_names = [ref.canonical_name for ref in flow_data.references]
    if not canonical_names:
        return StudioSearchResult()

    # Stage 1: translate canonical_names into a StudioQuerySpec via the
    # Step 3 studio LLM. Soft-fail to a deterministic freeform spec on
    # any LLM failure — that guarantees we always have at least the raw
    # names to try as freeform tokens.
    spec = await _translate_studio_query(canonical_names)

    # Stage 2: execute the spec in ANY mode. restrict_to_movie_ids=None
    # gives us the natural match set (one entry per matched movie).
    endpoint_result = await execute_studio_query(
        spec, restrict_to_movie_ids=None,
        metadata_filters=metadata_filters,
    )
    matched_movie_ids = {sc.movie_id for sc in endpoint_result.scores}
    if not matched_movie_ids:
        return StudioSearchResult()

    # Stage 3: bulk popularity fetch + single-tier sort.
    popularity = await fetch_quality_popularity_signals(list(matched_movie_ids))
    ranked = sort_movie_ids_by_popularity(matched_movie_ids, popularity)

    # Negative or zero limit collapses to an empty list; positive limit
    # truncates. Plain slice (`ranked[:limit]`) would do the wrong thing
    # for limit < 0 — Python's negative-index slicing would drop the
    # *last* |limit| entries instead of clearing the list.
    if limit > 0:
        ranked = ranked[:limit]
    else:
        ranked = []

    return StudioSearchResult(ranked=ranked)


# ---------------------------------------------------------------------------
# Stage 1 — LLM translation wrapper.
# ---------------------------------------------------------------------------


async def _translate_studio_query(
    canonical_names: list[str],
) -> StudioQuerySpec:
    """Run the Step 3 studio translator on the canonical_name list.

    Returns a StudioQuerySpec the executor can consume. On any LLM
    failure (timeout, parse error, etc.) falls back to a deterministic
    spec built directly from the canonical names — each name becomes
    its own StudioRef with that name as the sole freeform_names entry.
    The fallback keeps the entire flow functional when the LLM is
    unavailable; brand-registry resolution is the only thing lost.

    Scoring_method is forced to ANY post-LLM. The entity-flow surface
    is bare-list semantics ("Pixar and Aardman" reads as "movies from
    either", not "co-productions"). Forcing ANY guarantees we don't
    accidentally pick up ALL semantics from the LLM's reading of our
    synthesized retrieval_intent.
    """
    intent_rewrite, description = _synthesize_studio_call_inputs(canonical_names)
    route_rationale = "studio name reference"

    try:
        spec, _, _ = await generate_studio_query(
            intent_rewrite=intent_rewrite,
            description=description,
            route_rationale=route_rationale,
            provider=HANDLER_LLM_PROVIDER,
            model=HANDLER_LLM_MODEL,
            **HANDLER_LLM_KWARGS,
        )
    except Exception:  # noqa: BLE001 — soft-fail by design
        logger.warning(
            "Studio translator LLM call failed; falling back to deterministic "
            "freeform spec for canonical_names=%r",
            canonical_names,
            exc_info=True,
        )
        spec = _build_fallback_spec(canonical_names)

    # Force ANY regardless of what the LLM committed. See docstring.
    spec.scoring_method = ScoringMethod.ANY
    return spec


def _synthesize_studio_call_inputs(
    canonical_names: list[str],
) -> tuple[str, str]:
    """Build (intent_rewrite, description) strings for the translator.

    Phrasing reads as "movies from any of these studios" so the LLM has
    no reason to pick ALL semantics — though we force ANY post-call
    regardless, a coherent intent_rewrite still helps the LLM resolve
    each studio cleanly.
    """
    if len(canonical_names) == 1:
        name = canonical_names[0]
        return (
            f"Movies produced by {name}.",
            f"produced by {name}",
        )
    # 2 entries → "A or B" (no comma); 3+ entries → Oxford-comma form
    # ("A, B, or C"). Avoids the weird "A, or B" reading for a 2-list.
    if len(canonical_names) == 2:
        joined = f"{canonical_names[0]} or {canonical_names[1]}"
    else:
        joined = ", ".join(canonical_names[:-1]) + f", or {canonical_names[-1]}"
    return (
        f"Movies produced by any of {joined}.",
        f"produced by any of {joined}",
    )


def _build_fallback_spec(canonical_names: list[str]) -> StudioQuerySpec:
    """Deterministic StudioQuerySpec used when the LLM call fails.

    One StudioRef per canonical_name with the name itself as the sole
    freeform_names entry. No brand resolution (the registry mapping is
    what the LLM provides) — the freeform path tokenizes + intersects,
    which still recovers most production-company strings the user
    typed verbatim. ANY-mode scoring is forced upstream.
    """
    studios = [
        StudioRef(
            name=name,
            studio_exploration=(
                "Fallback path: LLM unavailable; treating the typed name as "
                "a freeform IMDB credit string."
            ),
            brand=None,
            freeform_names=[name],
        )
        for name in canonical_names
    ]
    return StudioQuerySpec(
        exploration=(
            "Fallback path: LLM unavailable; one freeform ref per typed "
            "studio name."
        ),
        studios=studios,
        scoring_method=ScoringMethod.ANY,
    )


# Stage 3 popularity sort is delegated to search_v2.popularity_sort —
# the same convention is used by character_franchise_search and any
# future entity-flow executor that buckets matched movies before
# emitting them.
