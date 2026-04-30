# Search V2 — Stage 4 orchestrator.
#
# Entry point: run_stage_4().  One call per Step-1 interpretation
# branch.  Composes the full pipeline described in
# search_improvement_planning/step_4_planning.md:
#
#   0. detect flow + tag items
#   1. translate every LLM-backed item in parallel (20s each)
#   2. execute pool-independent items as soon as translation returns
#      (candidate generators + deterministic exclusions + trending)
#   3. assembly barrier — union candidate sets into the pool
#   4. subtract deterministic exclusions from the pool
#   5. execute pool-dependent items in parallel
#   6. compose final scores
#   7. sort, slice top-K, fetch display cards, shape the payload
#
# Soft-failure is load-bearing: a single endpoint's timeout or error
# never blocks the branch.  The per-LLM and per-execution 20-second
# budgets live in dispatch.TIMEOUT_SECONDS.

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

from qdrant_client import AsyncQdrantClient

from db.postgres import fetch_browse_seed_ids, fetch_movie_cards
from implementation.llms.generic_methods import LLMProvider
from schemas.endpoint_result import EndpointResult
from schemas.enums import EndpointRoute
from schemas.query_understanding import QueryUnderstandingResponse
from search_v2.stage_4.assembly import (
    apply_deterministic_exclusions,
    assemble_pool,
)
from search_v2.stage_4.dispatch import execute_item, translate_item
from search_v2.stage_4.display import build_display_payload
from search_v2.stage_4.flow_detection import detect_flow, tag_items
from search_v2.stage_4.scoring import score_pool
from search_v2.stage_4.types import (
    EndpointOutcome,
    Stage4Debug,
    Stage4Flow,
    Stage4Result,
    TaggedItem,
)


# Browse-seed size — top-N from movie_card ordered by a temporary
# popularity-first fallback until priors are redesigned.
BROWSE_SEED_SIZE = 2000


async def run_stage_4(
    qu: QueryUnderstandingResponse,
    intent_rewrite: str,
    *,
    today: date,
    provider: LLMProvider,
    model: str,
    qdrant_client: AsyncQdrantClient,
    top_k: int = 100,
) -> Stage4Result:
    """Run the full Stage-4 pipeline for one Step-1 branch."""
    # ----- Phase 0: flow detection + item tagging -----------------------------
    flow = detect_flow(qu)
    items = tag_items(qu, flow)

    # Partition by pipeline role. The set of items that fire pool-
    # independent is "everything whose outcome doesn't need the pool":
    # candidate generators in the active flow, plus every deterministic
    # exclusion (semantic exclusions need the pool for their HasId
    # filter). Everything else is pool-dependent.
    candidate_generators = [i for i in items if i.generates_candidates]
    det_exclusions = [
        i for i in items
        if i.role == "exclusion_dealbreaker"
        and i.endpoint != EndpointRoute.SEMANTIC
    ]
    sem_exclusions = [
        i for i in items
        if i.role == "exclusion_dealbreaker"
        and i.endpoint == EndpointRoute.SEMANTIC
    ]
    pool_dependent_items = (
        [i for i in items
         if i.role == "inclusion_dealbreaker" and not i.generates_candidates]
        + [i for i in items
           if i.role == "preference" and not i.generates_candidates]
        + sem_exclusions
    )

    pool_independent_items = candidate_generators + det_exclusions

    translate_kwargs = dict(
        intent_rewrite=intent_rewrite,
        today=today,
        provider=provider,
        model=model,
    )
    execute_kwargs = dict(qdrant_client=qdrant_client)

    # ----- Phases 1+2 + BROWSE seed -------------------------------------------
    # Three concurrent streams:
    #   (a) translate + execute pool-independent items
    #   (b) translate pool-dependent items (stash specs for phase 5)
    #   (c) in BROWSE flow only, fetch the temporary browse seed pool
    pool_independent_task = _run_pool_independent_items(
        pool_independent_items,
        translate_kwargs=translate_kwargs,
        execute_kwargs=execute_kwargs,
    )
    pool_dependent_translate_task = _run_pool_dependent_translations(
        pool_dependent_items, translate_kwargs=translate_kwargs
    )
    browse_seed_task = (
        _build_browse_seed()
        if flow == Stage4Flow.BROWSE
        else _no_browse_seed()
    )

    (
        pool_independent_outcomes,
        pool_dependent_translations,
        browse_seed_ids,
    ) = await asyncio.gather(
        pool_independent_task,
        pool_dependent_translate_task,
        browse_seed_task,
    )

    # ----- Phase 3: assembly barrier ------------------------------------------
    candidate_outcomes = [
        o for o in pool_independent_outcomes if o.item.generates_candidates
    ]
    det_exclusion_outcomes = [
        o for o in pool_independent_outcomes
        if o.item.role == "exclusion_dealbreaker"
    ]

    pool = assemble_pool(
        candidate_outcomes, browse_seed_ids=browse_seed_ids
    )
    pool_size_after_generation = len(pool)

    if not pool:
        return _short_circuit_empty(
            flow,
            pool_independent_outcomes,
            pool_dependent_translations,
            pool_size_after_generation=0,
            pool_size_after_exclusion=0,
        )

    # ----- Phase 4: deterministic exclusion subtraction -----------------------
    pool = apply_deterministic_exclusions(pool, det_exclusion_outcomes)
    pool_size_after_exclusion = len(pool)

    if not pool:
        return _short_circuit_empty(
            flow,
            pool_independent_outcomes,
            pool_dependent_translations,
            pool_size_after_generation=pool_size_after_generation,
            pool_size_after_exclusion=0,
        )

    # ----- Phase 5: pool-dependent execution ----------------------------------
    pool_set = set(pool)
    pool_dep_exec_task = _run_pool_dependent_executions(
        pool_dependent_translations,
        restrict_to_movie_ids=pool_set,
        execute_kwargs=execute_kwargs,
    )
    pool_dependent_outcomes = await pool_dep_exec_task

    # ----- Phase 6: score composition -----------------------------------------
    all_outcomes = list(pool_independent_outcomes) + list(
        pool_dependent_outcomes
    )
    inclusion_outcomes = [
        o for o in all_outcomes if o.item.role == "inclusion_dealbreaker"
    ]
    preference_outcomes = [
        o for o in all_outcomes if o.item.role == "preference"
    ]
    semantic_exclusion_outcomes = [
        o for o in all_outcomes
        if o.item.role == "exclusion_dealbreaker"
        and o.item.endpoint == EndpointRoute.SEMANTIC
    ]

    breakdowns = score_pool(
        pool,
        inclusion_outcomes=inclusion_outcomes,
        preference_outcomes=preference_outcomes,
        semantic_exclusion_outcomes=semantic_exclusion_outcomes,
    )

    # ----- Phase 7: sort, slice, shape ----------------------------------------
    # Python's sort is stable, so ties fall back to arrival order —
    # the order in which candidates entered the pool (assembly.py
    # preserves that).
    breakdowns.sort(key=lambda b: b.final_score, reverse=True)
    top_breakdowns = breakdowns[:top_k]
    top_ids = [b.movie_id for b in top_breakdowns]

    cards = await fetch_movie_cards(top_ids)
    cards_by_id: dict[int, dict] = {c["movie_id"]: c for c in cards}
    movies = build_display_payload(top_ids, cards_by_id)

    # Debug shape: per-item EndpointOutcome keyed by debug_key, plus
    # per-result ScoreBreakdown keyed by movie_id. Downstream tooling
    # can drop into either dimension without re-indexing.
    debug = Stage4Debug(
        flow=flow,
        outcomes={o.item.debug_key: o for o in all_outcomes},
        pool_size_after_generation=pool_size_after_generation,
        pool_size_after_exclusion=pool_size_after_exclusion,
        pool_size_after_scoring_trim=len(top_breakdowns),
        per_result={b.movie_id: b for b in top_breakdowns},
    )
    return Stage4Result(movies=movies, debug=debug)


# ===========================================================================
# Dispatch helpers
# ===========================================================================


async def _run_pool_independent_items(
    items: list[TaggedItem],
    *,
    translate_kwargs: dict[str, Any],
    execute_kwargs: dict[str, Any],
) -> list[EndpointOutcome]:
    async def one(item: TaggedItem) -> EndpointOutcome:
        spec, llm_ms, status, err = await translate_item(
            item, **translate_kwargs
        )
        if status != "ok":
            return EndpointOutcome(
                item=item,
                result=EndpointResult(),
                status=status,
                llm_ms=llm_ms,
                exec_ms=None,
                error_message=err,
            )
        result, exec_ms, exec_status, exec_err = await execute_item(
            item, spec, None, **execute_kwargs
        )
        return EndpointOutcome(
            item=item,
            result=result,
            status=exec_status,
            llm_ms=llm_ms,
            exec_ms=exec_ms,
            error_message=exec_err,
        )

    if not items:
        return []
    return list(await asyncio.gather(*(one(i) for i in items)))


# Tuple shape: (item, spec, llm_ms, status, error_message)
_PendingTranslation = tuple[
    TaggedItem, Any | None, float | None, str, str | None
]


async def _run_pool_dependent_translations(
    items: list[TaggedItem],
    *,
    translate_kwargs: dict[str, Any],
) -> list[_PendingTranslation]:
    async def one(item: TaggedItem) -> _PendingTranslation:
        spec, llm_ms, status, err = await translate_item(
            item, **translate_kwargs
        )
        return (item, spec, llm_ms, status, err)

    if not items:
        return []
    return list(await asyncio.gather(*(one(i) for i in items)))


async def _run_pool_dependent_executions(
    translations: list[_PendingTranslation],
    *,
    restrict_to_movie_ids: set[int],
    execute_kwargs: dict[str, Any],
) -> list[EndpointOutcome]:
    async def one(
        item: TaggedItem,
        spec: Any | None,
        llm_ms: float | None,
        translation_status: str,
        translation_error: str | None,
    ) -> EndpointOutcome:
        if translation_status != "ok":
            # Translation failed — skip execution, surface the upstream
            # status so debug shows why this endpoint has no scores.
            return EndpointOutcome(
                item=item,
                result=EndpointResult(),
                status=translation_status,  # type: ignore[arg-type]
                llm_ms=llm_ms,
                exec_ms=None,
                error_message=translation_error,
            )
        result, exec_ms, exec_status, exec_err = await execute_item(
            item, spec, restrict_to_movie_ids, **execute_kwargs
        )
        return EndpointOutcome(
            item=item,
            result=result,
            status=exec_status,
            llm_ms=llm_ms,
            exec_ms=exec_ms,
            error_message=exec_err,
        )

    if not translations:
        return []
    return list(
        await asyncio.gather(
            *(one(i, s, lm, st, e) for (i, s, lm, st, e) in translations)
        )
    )


# ===========================================================================
# BROWSE-flow seed
# ===========================================================================


async def _build_browse_seed(
) -> list[int]:
    """Top-N movie_ids ordered by the temporary browse fallback."""
    # TODO: Priors were intentionally removed. Revisit browse seeding
    # when a replacement ranking mechanism is designed.
    return await fetch_browse_seed_ids(limit=BROWSE_SEED_SIZE)


async def _no_browse_seed() -> None:
    # Placeholder awaitable so the main asyncio.gather stays symmetric
    # regardless of flow. Returning None is a signal to assembly.py
    # that browse seed is not in play for this branch.
    return None


# ===========================================================================
# Empty-pool fallback
# ===========================================================================


def _short_circuit_empty(
    flow: Stage4Flow,
    pool_independent_outcomes: list[EndpointOutcome],
    pool_dependent_translations: list[_PendingTranslation],
    *,
    pool_size_after_generation: int,
    pool_size_after_exclusion: int,
) -> Stage4Result:
    """Return an empty Stage4Result with full debug when the pool empties.

    Happens when every candidate generator came back empty, OR when
    deterministic exclusions pruned the pool to zero.  Pool-dependent
    translations have no execution counterpart to attach.  When the
    translation itself failed (timeout / error), preserve that status
    on the outcome so debug still shows the true failure mode — only
    translations that succeeded get stamped "skipped".
    """
    skipped_pool_dep = [
        EndpointOutcome(
            item=item,
            result=EndpointResult(),
            status=status if status != "ok" else "skipped",
            llm_ms=llm_ms,
            exec_ms=None,
            error_message=(
                "empty pool — no pool-dependent execution fired"
                if status == "ok"
                else err
            ),
        )
        for (item, _spec, llm_ms, status, err) in pool_dependent_translations
    ]

    all_outcomes = list(pool_independent_outcomes) + skipped_pool_dep

    return Stage4Result(
        movies=[],
        debug=Stage4Debug(
            flow=flow,
            outcomes={o.item.debug_key: o for o in all_outcomes},
            pool_size_after_generation=pool_size_after_generation,
            pool_size_after_exclusion=pool_size_after_exclusion,
            pool_size_after_scoring_trim=0,
            per_result={},
        ),
    )
