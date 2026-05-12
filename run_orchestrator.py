"""
run_orchestrator.py — CLI runner for the full search pipeline.

Runs `search_v2.full_pipeline_orchestrator.run_full_pipeline` with
`skip_bypass_steps_0_1=True`, so the raw query goes straight into
Step 2 as a single "original" branch, then through Stage 4 execution
+ ranking. Prints:

  1. Per-step completion events with elapsed (real-time, via INFO logs).
  2. Total wall-clock elapsed for the whole pipeline.
  3. The full list of generated endpoint calls, grouped by
     trait → category → individual endpoint call. Per-call output
     names the endpoint route and dumps only the non-null parameters
     of its EndpointParameters wrapper.
  4. Per-branch ranked results — the top-N candidates by Stage-4
     final score, with title and release year resolved from movie_card.

Usage:
    python run_orchestrator.py "your query here"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

# Project root on sys.path so absolute imports resolve when the script
# is invoked directly (matches the convention in run_search.py).
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")

from search_v2 import full_pipeline_orchestrator as _orchestrator_module  # noqa: E402
from search_v2 import stage_4_execution as _stage4_module  # noqa: E402
from search_v2.full_pipeline_orchestrator import (  # noqa: E402
    BranchRankedResults,
    CategoryCallWithEndpoints,
    FullPipelineResult,
    GeneratedEndpointSpec,
    TraitWithEndpoints,
    run_full_pipeline,
)
from db.postgres import fetch_movie_cards, pool as postgres_pool  # noqa: E402
import db.redis as _redis_module  # noqa: E402
from db.redis import init_redis  # noqa: E402


# Per-spec origin map populated by the generation-phase wrapper below.
# Keyed by `id(spec)` so the execution-phase wrapper can recover the
# (trait, category) labels for any spec it dispatches without changing
# the spec's signature. Cleared at the start of every pipeline run so
# stale ids from prior runs cannot leak across invocations.
_spec_origin: dict[int, tuple[str, str]] = {}

# Holds spec object references so their ids remain valid for the full
# lifetime of one pipeline run. Python may otherwise recycle id() of a
# garbage-collected spec, causing the origin lookup to alias an
# unrelated object. Cleared together with _spec_origin per run.
_spec_keepalive: list[GeneratedEndpointSpec] = []

# Timing logger — emits [trait][category][phase] X.XXs lines through the
# same handler/formatter as the existing pipeline INFO logs so output
# stays interleaved chronologically.
_timing_logger = logging.getLogger("run_orchestrator.timing")


# Top-N to display from each branch's ranked candidate list. Stage 4
# returns the full ranked set; capping the display keeps CLI output
# scannable for branches that produce hundreds or thousands of hits.
DEFAULT_TOP_N = 25


# ---------------------------------------------------------------------------
# Logging setup — surface per-step completion events from the
# orchestrator in real time. Filtering down to the orchestrator's own
# logger keeps unrelated INFO chatter (httpx, openai, etc.) silent.
# ---------------------------------------------------------------------------


def _configure_realtime_logging() -> None:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("[%(relativeCreated)6.0fms] %(message)s")
    )

    # Surface per-step events from the front-half orchestrator AND the
    # Stage-4 executor (per-branch ranking timing, auxiliary spec
    # decisions, per-call failure warnings) without pulling in noisy
    # INFO chatter from httpx / openai / etc. The run_orchestrator
    # timing logger reuses the same handler so the new
    # [trait][category][phase] lines interleave with existing events.
    for logger_name in (
        "search_v2.full_pipeline_orchestrator",
        "search_v2.stage_4_execution",
        "run_orchestrator.timing",
    ):
        sub_logger = logging.getLogger(logger_name)
        sub_logger.setLevel(logging.INFO)
        sub_logger.addHandler(handler)
        sub_logger.propagate = False


# ---------------------------------------------------------------------------
# Per-(trait, category, phase) timing instrumentation.
#
# We monkey-patch two pipeline seams from this CLI rather than editing
# the pipeline modules themselves: the goal is purely diagnostic output
# for this runner, so the instrumentation lives next to the printer.
#
#   * Generation phase — wrap _process_category_call in the front-half
#     orchestrator. Its inner `run_query_generation` call is the unit
#     of "category query generation"; we time the whole wrapper because
#     deterministic / no-op categories also flow through it and should
#     still surface a (near-zero) timing line for symmetry.
#
#   * Execution phase — wrap stage_4_execution._dispatch_call. That is
#     the single chokepoint every per-spec endpoint call passes through
#     during Phase B (generators) and Phase C (rerankers). We recover
#     the (trait, category) provenance from id(spec) using the map the
#     generation wrapper populates; auxiliary specs (shorts exclusion,
#     neutral seed) have no trait/category and are logged under a
#     synthetic "<auxiliary>" label so they remain visible.
# ---------------------------------------------------------------------------


def _install_timing_patches() -> None:
    """Replace the two pipeline functions with timing-aware wrappers.

    Idempotent — re-installation only attaches our wrapper once, even
    if the module is reloaded inside the same Python process. The
    wrappers delegate to the originals, so behavior is identical aside
    from the new log emissions.
    """
    original_process_category_call = _orchestrator_module._process_category_call
    original_dispatch_call = _stage4_module._dispatch_call

    async def timed_process_category_call(
        category_call,
        trait,
        *,
        branch_label,
        sibling_calls,
        combine_mode,
    ):
        # Resolve labels once up front so the log line stays consistent
        # regardless of whether generation succeeds, errors, or yields
        # an empty spec list (deterministic NO_OP categories included).
        trait_label = trait.surface_text
        category_label = category_call.category.name

        start = time.perf_counter()
        result = await original_process_category_call(
            category_call,
            trait,
            branch_label=branch_label,
            sibling_calls=sibling_calls,
            combine_mode=combine_mode,
        )
        elapsed = time.perf_counter() - start

        _timing_logger.info(
            "[%s][%s][generation] %.2fs",
            trait_label,
            category_label,
            elapsed,
        )

        # Register every emitted spec so the execution-phase wrapper can
        # recover (trait, category) by id(spec). Keepalive list pins the
        # spec objects for the duration of the run so id() reuse cannot
        # cause origin-map aliasing.
        for spec in result.generated_specs:
            _spec_origin[id(spec)] = (trait_label, category_label)
            _spec_keepalive.append(spec)

        return result

    async def timed_dispatch_call(spec, *, restrict):
        # Auxiliary specs (shorts exclusion, neutral seed) are not
        # produced by _process_category_call so they have no origin
        # entry — label them explicitly rather than dropping the line.
        origin = _spec_origin.get(id(spec))
        if origin is None:
            trait_label, category_label = "<auxiliary>", spec.route.value
        else:
            trait_label, category_label = origin

        start = time.perf_counter()
        result = await original_dispatch_call(spec, restrict=restrict)
        elapsed = time.perf_counter() - start

        _timing_logger.info(
            "[%s][%s][execution] %.2fs (route=%s)",
            trait_label,
            category_label,
            elapsed,
            spec.route.value,
        )
        return result

    _orchestrator_module._process_category_call = timed_process_category_call
    _stage4_module._dispatch_call = timed_dispatch_call


def _reset_timing_state() -> None:
    """Clear per-run origin tracking before each pipeline invocation."""
    _spec_origin.clear()
    _spec_keepalive.clear()


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------


def _params_to_dict(params: BaseModel | None) -> dict | None:
    # exclude_none=True drops every field whose value is None at any
    # level of nesting, so the printed dict only carries the fields
    # the LLM (or deterministic builder) actually populated. mode="json"
    # coerces enums and other non-primitive types to their JSON forms
    # so json.dumps below doesn't choke.
    if params is None:
        return None
    return params.model_dump(mode="json", exclude_none=True)


def _print_endpoint_spec(spec: GeneratedEndpointSpec, indent: str) -> None:
    print(f"{indent}- endpoint route: {spec.route.value}")
    if spec.params is None:
        # TRENDING is the only route with no params today; surface
        # that explicitly rather than printing an empty dict.
        print(f"{indent}  params: (none — endpoint takes no parameters)")
        return

    params_dict = _params_to_dict(spec.params)
    print(f"{indent}  wrapper: {type(spec.params).__name__}")
    print(f"{indent}  params (non-null fields only):")
    # json.dumps with indent=2 gives a clean nested view; we then
    # re-indent each line so it sits under the "params:" label.
    rendered = json.dumps(params_dict, indent=2, ensure_ascii=False)
    for line in rendered.splitlines():
        print(f"{indent}    {line}")


def _print_category_call(
    cc: CategoryCallWithEndpoints, indent: str
) -> None:
    print(f"{indent}CATEGORY: {cc.category.value}")
    if cc.handler_error is not None:
        print(f"{indent}  handler_error: {cc.handler_error}")
    if not cc.generated_specs:
        # Empty is a valid outcome (EXPLICIT_NO_OP, MEDIA_TYPE with
        # nothing matchable, or LLM judged nothing to fire).
        print(f"{indent}  (no endpoint calls generated)")
        return
    for spec in cc.generated_specs:
        _print_endpoint_spec(spec, indent + "  ")


def _print_trait(trait: TraitWithEndpoints, indent: str = "") -> None:
    print(
        f'{indent}TRAIT: "{trait.surface_text}" '
        f"[polarity={trait.polarity.value}, "
        f"commitment={trait.commitment}]"
    )
    if trait.step_3_error is not None:
        print(f"{indent}  step_3_error: {trait.step_3_error}")
    if not trait.category_calls:
        print(f"{indent}  (no category calls — Step 3 produced nothing)")
        return
    for cc in trait.category_calls:
        _print_category_call(cc, indent + "  ")


def _print_auxiliary_specs(result: FullPipelineResult) -> None:
    # Auxiliary specs are global fetches not attached to any trait.
    # Today the only one is the default shorts-exclusion MEDIA_TYPE
    # fetch injected when no trait emitted a MEDIA_TYPE call.
    if not result.auxiliary_endpoint_specs:
        print(
            "AUXILIARY ENDPOINT CALLS (not attached to a trait): "
            "(none — a trait already covers MEDIA_TYPE)"
        )
        return
    print(
        f"AUXILIARY ENDPOINT CALLS (not attached to a trait) — "
        f"{len(result.auxiliary_endpoint_specs)} call(s):"
    )
    for spec in result.auxiliary_endpoint_specs:
        _print_endpoint_spec(spec, indent="  ")


def _print_implicit_expectations(result: FullPipelineResult) -> None:
    if not result.branches:
        return
    branch = result.branches[0]
    print()
    print("=" * 72)
    print("IMPLICIT PRIOR POLICY")
    print("=" * 72)

    if branch.implicit_expectations_error is not None:
        print(f"implicit_expectations_error: {branch.implicit_expectations_error}")
        return

    policy = branch.implicit_expectations
    if policy is None:
        print("(no implicit-prior policy result)")
        return

    print(f"query_intent_summary: {policy.query_intent_summary}")
    print()
    print("explicit_signals:")
    for signal in policy.explicit_signals:
        print(
            f'  - "{signal.query_span}" -> axis={signal.explicit_axis}, '
            f"direction={signal.direction}, coverage={signal.coverage}"
        )
        print(f"    description: {signal.normalized_description}")
        print(f"    evidence: {signal.strength_evidence}")

    ordering = policy.ordering_axis_analysis
    print()
    print("ordering_axis_analysis:")
    print(f"  primary_ordering_axis: {ordering.primary_ordering_axis}")
    print(f"  suppresses_quality_prior: {ordering.suppresses_quality_prior}")
    print(f"  suppresses_popularity_prior: {ordering.suppresses_popularity_prior}")
    print(f"  axis_description: {ordering.axis_description}")

    specificity = policy.query_specificity_analysis
    print()
    print("query_specificity_analysis:")
    print(f"  explicit_trait_pressure: {specificity.explicit_trait_pressure}")
    print(f"  prior_room: {specificity.prior_room}")
    print(f"  specificity_summary: {specificity.specificity_summary}")

    print()
    print("final_prior_decisions:")
    print(
        f"  quality: direction={policy.quality_prior.direction}, "
        f"strength={policy.quality_prior.strength}"
    )
    print(f"    rationale: {policy.quality_prior.rationale}")
    print(
        f"  popularity: direction={policy.popularity_prior.direction}, "
        f"strength={policy.popularity_prior.strength}"
    )
    print(f"    rationale: {policy.popularity_prior.rationale}")


def _print_full_result(result: FullPipelineResult) -> None:
    if not result.branches:
        print("\nNo branches produced — nothing to display.")
        return
    # skip_bypass_steps_0_1=True always yields exactly one branch;
    # no need to label by branch kind in the header for this script.
    branch = result.branches[0]
    if branch.branch_error is not None:
        print(f"\nBranch failed: {branch.branch_error}")
        return
    if not branch.traits:
        print("\nBranch produced zero traits.")
        _print_auxiliary_specs(result)
        return
    print(f"\nGenerated {len(branch.traits)} trait(s):\n")
    for trait in branch.traits:
        _print_trait(trait)
        print()
    _print_auxiliary_specs(result)
    _print_implicit_expectations(result)


async def _print_ranked_results(
    branch_results: list[BranchRankedResults],
    *,
    top_n: int = DEFAULT_TOP_N,
) -> None:
    """Print each branch's top-N ranked candidates with title + year.

    Pulls movie_card metadata in one bulk fetch per branch so we never
    do per-candidate Postgres lookups (cross-codebase invariant). Cards
    that fail to resolve render with placeholder title/year so a stale
    ID never breaks the output.
    """
    if not branch_results:
        print("\nNo ranked results — Stage 4 did not run.")
        return

    print()
    print("=" * 72)
    print("STAGE 4 RANKED RESULTS")
    print("=" * 72)

    for br in branch_results:
        print()
        print("-" * 72)
        print(f"Branch: {br.kind}  ({br.ui_label})")
        print(f"  query:        {br.query}")
        print(f"  total ranked: {len(br.ranked)}")
        print("-" * 72)

        if br.branch_error is not None:
            print(f"  branch_error: {br.branch_error}")
            continue
        if not br.ranked:
            print("  (no candidates ranked)")
            continue

        top = br.ranked[:top_n]
        cards = await fetch_movie_cards([mid for mid, _ in top])
        cards_by_id = {c["movie_id"]: c for c in cards}

        # One block per result. The structure mirrors the scoring tree:
        #   result header (final / boost / pos / neg / title)
        #     trait line (surface, commitment, inner trait_score, weight, signed contribution)
        #       category line (name, combine_type, expressions, retrieval_intent, combined score)
        #         endpoint line (route, score)
        # Per-category and per-endpoint rows populate for positive traits
        # only — negative traits use a three-bin gate × fuzzy formula that
        # does not fold through per-category scores, so we emit only the
        # trait-level contribution for them.
        for rank, (movie_id, score) in enumerate(top, start=1):
            card = cards_by_id.get(movie_id)
            if card is None:
                title, year = "<missing card>", "?"
            else:
                title = card["title"] or "<untitled>"
                ts = card["release_ts"]
                year = (
                    datetime.fromtimestamp(ts, tz=timezone.utc).year
                    if ts is not None
                    else "?"
                )

            breakdown = br.score_breakdowns.get(movie_id)
            if breakdown is None:
                pos = neg = 0.0
                boost_pct = 0.0
                contributions: list = []
            else:
                pos = breakdown.positive_total
                neg = breakdown.negative_total
                boost_pct = breakdown.implicit_prior_boost * 100.0
                contributions = breakdown.trait_contributions

            print()
            print(
                f"#{rank:<2} final={score:+.4f}  boost={boost_pct:+.1f}%  "
                f"pos={pos:+.4f}  neg={neg:+.4f}  "
                f"{title} ({year})  [tmdb={movie_id}]"
            )
            if not contributions:
                print("    (no trait contributions recorded)")
                continue
            for tc in contributions:
                # Trait line: surface text, commitment tier, the
                # across-category scoring method that produced
                # trait_score (framings = MAX, facets = PRODUCT,
                # gate×fuzzy for negatives), inner trait_score in
                # [0, 1], weight (commitment × rarity for positives,
                # commitment for negatives), and the signed weighted
                # contribution that fed the §9 sum.
                print(
                    f'  trait {tc.surface_text!r} '
                    f"[{tc.commitment} | {tc.scoring_method}]  "
                    f"trait_score={tc.trait_score:.4f}  "
                    f"wt={tc.weight:.3f}  → {tc.contribution:+.4f}"
                )
                if not tc.category_scores:
                    # Negative traits and positive traits whose
                    # categories all errored or were NO_OP land here.
                    continue
                for cs in tc.category_scores:
                    expr_repr = ", ".join(repr(e) for e in cs.expressions)
                    print(
                        f"    category {cs.category_name} "
                        f"[{cs.combine_type}]  cat_score={cs.score:.4f}"
                    )
                    print(f"      expressions: [{expr_repr}]")
                    print(f"      retrieval_intent: {cs.retrieval_intent!r}")
                    if not cs.endpoint_scores:
                        # All calls under this category failed (None
                        # from the executor) and were filtered out
                        # before combine_calls — surface the absence
                        # rather than silently emitting nothing.
                        print("      (no live endpoint calls)")
                        continue
                    for ep in cs.endpoint_scores:
                        print(f"      • {ep.route}: {ep.score:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def _ensure_db_ready() -> None:
    """Open the Postgres pool and init Redis if not already done.

    Stage 4 execution fires endpoint queries that hit Postgres + Qdrant
    + Redis directly. The Qdrant client is a module-level singleton
    that connects on first use, but the Postgres pool needs an explicit
    open() and Redis needs init_redis() — both are no-ops when already
    initialized so we can re-enter safely.
    """
    if postgres_pool._closed:
        await postgres_pool.open()
    if _redis_module._redis_client is None:
        await init_redis()


async def _run(query: str) -> None:
    print(f'Query: "{query}"\n')
    print("Step completions (real-time):")

    await _ensure_db_ready()

    # Wipe any stale (trait, category) origin entries from a prior run
    # in the same process before the new pipeline starts emitting specs.
    _reset_timing_state()

    pipeline_start = time.perf_counter()
    result = await run_full_pipeline(query, skip_bypass_steps_0_1=True)
    pipeline_elapsed = time.perf_counter() - pipeline_start

    # The orchestrator already records its own total_elapsed; print
    # both for sanity (the small delta covers script overhead).
    print(f"\nOrchestrator total_elapsed: {result.total_elapsed:.2f}s")
    print(f"Wall-clock elapsed:         {pipeline_elapsed:.2f}s")

    _print_full_result(result)
    await _print_ranked_results(result.branch_results)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the unified front-half orchestrator with Steps 0/1 "
            "bypassed. Prints per-step completion timing and the "
            "generated endpoint calls grouped by trait → category → "
            "endpoint."
        )
    )
    parser.add_argument(
        "query",
        type=str,
        help="Raw user query to feed into the pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    _configure_realtime_logging()
    _install_timing_patches()
    args = _parse_args()
    asyncio.run(_run(args.query))


if __name__ == "__main__":
    main()
