# Search V2 — Step 2 → Step 3 → Query Generation diagnostic runner.
#
# Built specifically to surface the ADDITIVE-multiply risk discussed
# in rescore_overhaul.md's "Sharp edges to watch": when a category's
# combine_type is ADDITIVE and one of the firing endpoints is the
# keyword endpoint, a missing keyword tag *zeros the entire category*
# regardless of how strong the semantic gradient is. The corollary is
# the V4 trait-level FACETS combine_mode: when across-category fold
# is PRODUCT, any zeroed category zeros the trait.
#
# Existing `run_query_generation.py` already runs the same three
# stages, but its output is shaped for raw inspection of a single
# query. This script:
#   1. Runs the V4 contract correctly (siblings → run_step_3).
#   2. Surfaces only the fields that actually drive the multiply
#      problem: per-trait combine_mode, per-category combine_type +
#      firing endpoints + finalized_keywords + scoring_method, and
#      the semantic body content.
#   3. Flags every ADDITIVE category whose firing set includes
#      KEYWORD — that is the "this category will zero on a KW miss"
#      trip-wire. The flag is the lookup key for "is the LLM
#      committing brittle keywords here?"
#   4. Optionally batches across multiple queries so we can compare
#      verdicts across a suite without re-running stages individually.
#
# Usage:
#   python -m search_v2.run_specs                      # default suite
#   python -m search_v2.run_specs "your query here"   # single query
#   python -m search_v2.run_specs --suite path.txt    # one query/line
#   python -m search_v2.run_specs --json out.json     # machine-readable

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from schemas.enums import EndpointRoute
from schemas.step_2 import Trait
from schemas.step_3 import CategoryCall
from schemas.trait_category import CategoryName
from search_v2.endpoint_fetching.category_handlers.generated_endpoint_spec import (
    GeneratedEndpointSpec,
)
from search_v2.run_query_generation import _run_handler_with_full_output
from search_v2.step_2 import run_step_2
from search_v2.step_3 import run_step_3


# Default suite — varies query shape (compound aesthetic / positioning
# /  fused-compound / motif-driven / ADDITIVE-heavy categories) so a
# single batch run exercises the failure modes without manual
# enumeration.
_DEFAULT_SUITE: list[str] = [
    "movies about WWII",                      # mixed CENTRAL_TOPIC ADDITIVE
    "feel-good Christmas movies",             # two ADDITIVE traits stacked
    "elevated horror",                        # fused compound, FACETS risk
    "dark gritty marvel movies",              # FRAMINGS marvel + FACETS dark/gritty
    "shitty shark movies",                    # ELEMENT_PRESENCE ADDITIVE
    "cozy fall movies",                       # SEASONAL_HOLIDAY + EMOTIONAL_EXPERIENTIAL
    "movies with a twist ending",             # NARRATIVE_DEVICES ADDITIVE
    "biographical dramas about musicians",    # CENTRAL_TOPIC + GENRE compound
    "violent action movies",                  # GENRE single + adjective qualifier
    "movies for a rainy Sunday afternoon",    # WATCH_CONTEXT pure SEM
]


# ----------------------------------------------------------------------
# Diagnostic shapes — keep these flat dataclasses. JSON-serializable
# via asdict + the enum-value fields stamped at construction time.
# ----------------------------------------------------------------------


@dataclass
class _EndpointSummary:
    """One fired endpoint's relevant-to-multiply-risk surface.

    Most endpoints (entity, metadata, trending, awards, studio,
    franchise, media_type) carry only `route`. Keyword and semantic
    expand because they're the two endpoints whose ADDITIVE-multiply
    interaction is what the script exists to interrogate.
    """

    route: str
    # Keyword-only fields — None for non-keyword endpoints.
    keyword_finalized: list[str] | None = None
    keyword_scoring_method: str | None = None
    # Semantic-only fields — None for non-semantic endpoints.
    semantic_role: str | None = None
    semantic_spaces: list[str] | None = None


@dataclass
class _CategorySummary:
    """One CategoryCall's diagnostic record.

    `additive_kw_risk` is the trip-wire flag that names the failure
    mode rescore_overhaul.md flagged: ADDITIVE combine + KEYWORD
    endpoint fired = a KW miss zeros the category regardless of SEM
    strength. The flag is computed mechanically from the category's
    combine_type and the fired routes, not the LLM's reasoning, so it
    is reliable for filtering across a suite.
    """

    category: str
    combine_type: str
    expressions: list[str]
    retrieval_intent: str
    fired_endpoints: list[_EndpointSummary] = field(default_factory=list)
    additive_kw_risk: bool = False


@dataclass
class _TraitSummary:
    """One trait's diagnostic record.

    `combine_mode` is the V4 trait-level fold: FRAMINGS = MAX across
    categories (additive risk per category is local), FACETS = PRODUCT
    across categories (any zeroed category zeros the trait). FACETS +
    additive_kw_risk on any category is the highest-severity case.
    """

    surface_text: str
    polarity: str
    commitment: str
    combine_mode: str
    relationship_role: str
    replaces_axis: str | None
    axes_replaced_by_siblings: list[str]
    categories: list[_CategorySummary] = field(default_factory=list)


@dataclass
class _QuerySummary:
    """One query's full diagnostic record."""

    query: str
    traits: list[_TraitSummary] = field(default_factory=list)
    error: str | None = None


# ----------------------------------------------------------------------
# Spec → endpoint summary mapping.
# ----------------------------------------------------------------------


def _summarize_endpoint(spec: GeneratedEndpointSpec) -> _EndpointSummary:
    """Pull just the fields needed for the multiply-risk diagnostic.

    Keyword: finalized members + scoring_method (ANY vs ALL changes
    the meaning of a "miss" — ANY is binary, ALL is fractional).
    Semantic: role (carver vs qualifier) + the vector spaces actually
    queried, since multi-space semantic still surfaces as ONE
    orchestrator-visible call but the body content lives across spaces.

    Other endpoints are summarized by route only — they aren't the
    subject of this analysis.
    """
    route = spec.route.value
    summary = _EndpointSummary(route=route)

    if spec.params is None:
        return summary

    if spec.route is EndpointRoute.KEYWORD:
        # Both single and subintent KeywordQuerySpec shapes carry
        # finalized_keywords + scoring_method on the inner `parameters`.
        inner = getattr(spec.params, "parameters", None)
        if inner is not None:
            summary.keyword_finalized = list(
                getattr(inner, "finalized_keywords", []) or []
            )
            method = getattr(inner, "scoring_method", None)
            summary.keyword_scoring_method = (
                method.value if hasattr(method, "value") else
                (str(method) if method is not None else None)
            )
        return summary

    if spec.route is EndpointRoute.SEMANTIC:
        inner = getattr(spec.params, "parameters", None)
        if inner is not None:
            role = getattr(inner, "role", None)
            summary.semantic_role = (
                role.value if hasattr(role, "value") else
                (str(role) if role is not None else None)
            )
            space_queries = getattr(inner, "space_queries", None) or []
            spaces: list[str] = []
            for wq in space_queries:
                space = getattr(getattr(wq, "query", None), "space", None)
                if space is not None:
                    spaces.append(
                        space.value if hasattr(space, "value") else str(space)
                    )
            summary.semantic_spaces = spaces
        return summary

    return summary


def _summarize_category(
    call: CategoryCall, specs: list[GeneratedEndpointSpec]
) -> _CategorySummary:
    """Roll the per-category summary up from its CategoryCall + specs.

    `additive_kw_risk` fires when:
      - the category's combine_type is ADDITIVE (within-category
        product across firing endpoints), AND
      - at least one of the firing endpoints is the keyword endpoint.

    Both conditions are needed: ADDITIVE without KEYWORD just means
    multiple semantic-style signals multiplied, which is far less
    failure-prone than ADDITIVE with a binary KW gate.
    """
    cat: CategoryName = call.category
    combine_type = cat.combine_type.value
    fired: list[_EndpointSummary] = [_summarize_endpoint(s) for s in specs]
    has_kw = any(e.route == EndpointRoute.KEYWORD.value for e in fired)
    additive_kw_risk = (combine_type == "additive") and has_kw

    return _CategorySummary(
        category=cat.name,
        combine_type=combine_type,
        expressions=list(call.expressions),
        retrieval_intent=call.retrieval_intent,
        fired_endpoints=fired,
        additive_kw_risk=additive_kw_risk,
    )


# ----------------------------------------------------------------------
# Per-query orchestration — run Step 2, fan out Step 3 with sibling
# context, fan out the per-CategoryCall handler in parallel.
# ----------------------------------------------------------------------


async def _run_one_query(query: str) -> _QuerySummary:
    """Run the three stages and assemble a diagnostic record."""
    summary = _QuerySummary(query=query)

    try:
        analysis, *_ = await run_step_2(query)
    except Exception as exc:
        summary.error = f"Step 2 failed: {exc!r}"
        return summary

    if not analysis.traits:
        summary.error = "Step 2 emitted no traits"
        return summary

    # Step 3 — V4 contract: per-trait sibling context lets positioning
    # references drop replaced axes correctly. Empty siblings list
    # would silently regress positioning queries to V3 behavior.
    try:
        step_3_results = await asyncio.gather(
            *(
                run_step_3(
                    trait,
                    [s for s in analysis.traits if s is not trait],
                )
                for trait in analysis.traits
            )
        )
    except Exception as exc:
        summary.error = f"Step 3 failed: {exc!r}"
        return summary

    # Per-trait → fan out CategoryCalls in parallel; preserve order so
    # the diagnostic record reads in the same order as the input
    # decomposition.
    for trait, (decomposition, *_) in zip(analysis.traits, step_3_results):
        trait_summary = _TraitSummary(
            surface_text=trait.surface_text,
            polarity=trait.polarity.value,
            commitment=trait.commitment,
            combine_mode=decomposition.combine_mode.value,
            relationship_role=trait.relationship_role.value,
            replaces_axis=trait.replaces_axis,
            axes_replaced_by_siblings=list(trait.axes_replaced_by_siblings),
        )

        if not decomposition.category_calls:
            summary.traits.append(trait_summary)
            continue

        # Phase 6 — sibling-task context per call. Each handler
        # receives the OTHER CategoryCalls in this trait's
        # decomposition (self-category excluded) and the trait-level
        # combine_mode so it can coordinate against parallel siblings
        # under the trait's fold rule.
        all_calls = list(decomposition.category_calls)
        combine_mode = decomposition.combine_mode
        per_call_results = await asyncio.gather(
            *(
                _run_handler_with_full_output(
                    category_call=call,
                    trait=trait,
                    sibling_calls=[s for s in all_calls if s is not call],
                    combine_mode=combine_mode,
                )
                for call in all_calls
            ),
            return_exceptions=True,
        )

        for call, result in zip(decomposition.category_calls, per_call_results):
            if isinstance(result, Exception):
                # One handler crashing should not lose the rest of the
                # trait's diagnostic. Stamp a synthetic category record
                # so the failure surfaces in the report instead of being
                # silently dropped.
                trait_summary.categories.append(
                    _CategorySummary(
                        category=call.category.name,
                        combine_type=call.category.combine_type.value,
                        expressions=list(call.expressions),
                        retrieval_intent=call.retrieval_intent,
                        fired_endpoints=[
                            _EndpointSummary(route=f"<handler error: {result!r}>")
                        ],
                    )
                )
                continue
            _, specs = result
            trait_summary.categories.append(_summarize_category(call, specs))

        summary.traits.append(trait_summary)

    return summary


# ----------------------------------------------------------------------
# Pretty-print path. JSON path uses asdict directly.
# ----------------------------------------------------------------------


def _print_endpoint(endpoint: _EndpointSummary, indent: str) -> None:
    """One endpoint's line. Keyword and semantic carry extra detail
    inline because those are the routes the diagnostic exists for."""
    if endpoint.route == EndpointRoute.KEYWORD.value:
        method = endpoint.keyword_scoring_method or "?"
        members = endpoint.keyword_finalized or []
        rendered = ", ".join(members) if members else "(none)"
        print(f"{indent}[{endpoint.route}] {method}  → {rendered}")
        return
    if endpoint.route == EndpointRoute.SEMANTIC.value:
        role = endpoint.semantic_role or "?"
        spaces = ", ".join(endpoint.semantic_spaces or []) or "(none)"
        print(f"{indent}[{endpoint.route}] role={role}  spaces=[{spaces}]")
        return
    print(f"{indent}[{endpoint.route}]")


def _print_category(cat: _CategorySummary, indent: str) -> None:
    """One category — name, combine type, the all-important
    additive-kw-risk flag, the expressions, then per-endpoint lines."""
    flag = "  ⚠ ADDITIVE_KW_RISK" if cat.additive_kw_risk else ""
    print(f"{indent}— {cat.category}  ({cat.combine_type}){flag}")
    print(f"{indent}  expressions: {cat.expressions}")
    print(f"{indent}  retrieval_intent: {cat.retrieval_intent!r}")
    if not cat.fired_endpoints:
        print(f"{indent}  (no endpoints fired)")
        return
    for endpoint in cat.fired_endpoints:
        _print_endpoint(endpoint, indent + "  ")


def _print_trait(trait: _TraitSummary, indent: str) -> None:
    """One trait — surface text, polarity, commitment, combine_mode,
    role + axes (if positioning), then per-category."""
    role_suffix = f"  role={trait.relationship_role}"
    if trait.replaces_axis:
        role_suffix += f"  replaces_axis={trait.replaces_axis!r}"
    if trait.axes_replaced_by_siblings:
        role_suffix += (
            f"  axes_replaced_by_siblings={trait.axes_replaced_by_siblings}"
        )
    print(
        f"{indent}● trait '{trait.surface_text}'  "
        f"[{trait.polarity} | {trait.commitment} | {trait.combine_mode}]"
        f"{role_suffix}"
    )
    if not trait.categories:
        print(f"{indent}  (no categories)")
        return
    for cat in trait.categories:
        _print_category(cat, indent + "  ")


def _print_query_summary(summary: _QuerySummary) -> None:
    """One full query record. Header line includes a count of
    additive-kw-risk categories so a glance at a multi-query batch
    immediately surfaces the worrying ones."""
    risk_count = sum(
        1 for trait in summary.traits for cat in trait.categories
        if cat.additive_kw_risk
    )
    risk_suffix = f"  ⚠ {risk_count} additive_kw_risk" if risk_count else ""
    print(f"\n=== Query: {summary.query!r}{risk_suffix} ===")
    if summary.error:
        print(f"  ERROR: {summary.error}")
        return
    for trait in summary.traits:
        _print_trait(trait, "  ")


# ----------------------------------------------------------------------
# CLI plumbing.
# ----------------------------------------------------------------------


def _load_suite(path: Path) -> list[str]:
    """Read a suite file: one query per line, # comments + blanks
    skipped. Trailing whitespace stripped."""
    queries: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        queries.append(stripped)
    if not queries:
        raise SystemExit(f"suite file {path} contained no queries")
    return queries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run search_v2 Step 2 → Step 3 → query-generation across "
            "one or more queries with the V4 sibling contract, and "
            "surface the ADDITIVE-multiply / FACETS-product risk per "
            "(trait, category) pair."
        )
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "query",
        type=str,
        nargs="?",
        default=None,
        help=(
            "A single query to run. Mutually exclusive with --suite. "
            "When neither is given, the built-in default suite runs."
        ),
    )
    src.add_argument(
        "--suite",
        type=Path,
        default=None,
        help=(
            "Path to a text file with one query per line. Lines "
            "starting with '#' are skipped."
        ),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help=(
            "Optional path to write the full diagnostic batch as JSON. "
            "Pretty-printed text output still goes to stdout."
        ),
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=4,
        help=(
            "Max queries to run in parallel. Default 4. Lower if the "
            "LLM provider rate-limits."
        ),
    )
    return parser.parse_args()


async def _run_batch(queries: list[str], concurrent: int) -> list[_QuerySummary]:
    """Run a batch with bounded concurrency. The per-query call already
    fans out internally (Step 3 + handler) so we cap the outer fan-out
    to avoid stacking N queries × M traits × K categories of LLM
    calls all at once."""
    semaphore = asyncio.Semaphore(max(1, concurrent))

    async def _one(q: str) -> _QuerySummary:
        async with semaphore:
            return await _run_one_query(q)

    return await asyncio.gather(*(_one(q) for q in queries))


async def _main_async() -> None:
    args = _parse_args()
    if args.suite is not None:
        queries = _load_suite(args.suite)
    elif args.query is not None:
        queries = [args.query]
    else:
        queries = list(_DEFAULT_SUITE)

    print(f"[batch] {len(queries)} query(ies), concurrency={args.concurrent}")
    summaries = await _run_batch(queries, args.concurrent)

    for summary in summaries:
        _print_query_summary(summary)

    # Aggregate at the bottom — easier to spot patterns at a glance
    # than scrolling through per-query blocks.
    total_categories = sum(len(t.categories) for s in summaries for t in s.traits)
    risk_categories = sum(
        1 for s in summaries for t in s.traits for c in t.categories
        if c.additive_kw_risk
    )
    print(
        f"\n[summary] queries={len(summaries)}  "
        f"categories={total_categories}  "
        f"additive_kw_risk={risk_categories}"
    )

    if args.json is not None:
        payload = [asdict(s) for s in summaries]
        args.json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[wrote] {args.json}")


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
