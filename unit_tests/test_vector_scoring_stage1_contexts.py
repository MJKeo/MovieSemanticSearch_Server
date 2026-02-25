"""Unit tests for Stage 1 context construction in db.vector_scoring."""

from __future__ import annotations

from dataclasses import replace

import pytest

from implementation.classes.enums import RelevanceSize, VectorName
from implementation.classes.schemas import VectorSubqueries, VectorWeights

from db.vector_scoring import SpaceExecutionContext, build_space_execution_contexts


def make_vector_weights(**overrides: RelevanceSize) -> VectorWeights:
    """Build VectorWeights with explicit defaults for all non-anchor spaces."""
    base = {
        "plot_events_weight": RelevanceSize.NOT_RELEVANT,
        "plot_analysis_weight": RelevanceSize.NOT_RELEVANT,
        "viewer_experience_weight": RelevanceSize.NOT_RELEVANT,
        "watch_context_weight": RelevanceSize.NOT_RELEVANT,
        "narrative_techniques_weight": RelevanceSize.NOT_RELEVANT,
        "production_weight": RelevanceSize.NOT_RELEVANT,
        "reception_weight": RelevanceSize.NOT_RELEVANT,
    }
    base.update(overrides)
    return VectorWeights(**base)


def make_vector_subqueries(**overrides: str | None) -> VectorSubqueries:
    """Build VectorSubqueries with explicit defaults for all non-anchor spaces."""
    base = {
        "plot_events_subquery": None,
        "plot_analysis_subquery": None,
        "viewer_experience_subquery": None,
        "watch_context_subquery": None,
        "narrative_techniques_subquery": None,
        "production_subquery": None,
        "reception_subquery": None,
    }
    base.update(overrides)
    return VectorSubqueries(**base)


def context_by_name(contexts: list[SpaceExecutionContext]) -> dict[VectorName, SpaceExecutionContext]:
    """Map contexts by VectorName for direct assertions."""
    return {ctx.name: ctx for ctx in contexts}


def _non_anchor_names() -> list[VectorName]:
    return [name for name in VectorName if name != VectorName.ANCHOR]


def _weight_attr(space: VectorName) -> str:
    return f"{space.value}_weight"


def _subquery_attr(space: VectorName) -> str:
    return f"{space.value}_subquery"


def test_stage1_shape_and_order_contract() -> None:
    """Stage 1 returns one context per vector space in enum iteration order."""
    contexts = build_space_execution_contexts(make_vector_weights(), make_vector_subqueries())

    assert len(contexts) == len(VectorName) == 8
    assert [ctx.name for ctx in contexts] == list(VectorName)
    assert all(isinstance(ctx, SpaceExecutionContext) for ctx in contexts)
    assert all(ctx.normalized_weight == 0.0 for ctx in contexts)


def test_anchor_invariants() -> None:
    """Anchor should always be original-only with no effective relevance."""
    contexts = build_space_execution_contexts(
        make_vector_weights(
            plot_events_weight=RelevanceSize.LARGE,
            plot_analysis_weight=RelevanceSize.SMALL,
            viewer_experience_weight=RelevanceSize.MEDIUM,
            watch_context_weight=RelevanceSize.NOT_RELEVANT,
            narrative_techniques_weight=RelevanceSize.LARGE,
            production_weight=RelevanceSize.SMALL,
            reception_weight=RelevanceSize.NOT_RELEVANT,
        ),
        make_vector_subqueries(
            plot_events_subquery="a",
            plot_analysis_subquery=None,
            viewer_experience_subquery="b",
            watch_context_subquery="c",
            narrative_techniques_subquery="d",
            production_subquery=None,
            reception_subquery="e",
        ),
    )
    anchor = context_by_name(contexts)[VectorName.ANCHOR]

    assert anchor.did_run_original is True
    assert anchor.did_run_subquery is False
    assert anchor.effective_relevance is None
    assert anchor.is_active is True


@pytest.mark.parametrize("original_relevance", list(RelevanceSize))
@pytest.mark.parametrize("subquery_text", [None, "", "query text"])
def test_non_anchor_truth_table_parametrized(
    original_relevance: RelevanceSize,
    subquery_text: str | None,
) -> None:
    """Non-anchor state follows the corrected Stage 1 truth table."""
    target_space = VectorName.PLOT_EVENTS
    weights = make_vector_weights(**{_weight_attr(target_space): original_relevance})
    subqueries = make_vector_subqueries(**{_subquery_attr(target_space): subquery_text})

    ctx = context_by_name(build_space_execution_contexts(weights, subqueries))[target_space]

    expected_did_run_original = (original_relevance != RelevanceSize.NOT_RELEVANT)
    expected_did_run_subquery = (subquery_text is not None)
    if original_relevance == RelevanceSize.NOT_RELEVANT and subquery_text is not None:
        expected_effective = RelevanceSize.SMALL
    else:
        expected_effective = original_relevance

    assert ctx.did_run_original is expected_did_run_original
    assert ctx.did_run_subquery is expected_did_run_subquery
    assert ctx.effective_relevance == expected_effective
    assert ctx.is_active is (expected_did_run_original or expected_did_run_subquery)


def test_mixed_realistic_multi_space_scenario() -> None:
    """Mixed configuration should match expected per-space Stage 1 outputs."""
    weights = make_vector_weights(
        plot_events_weight=RelevanceSize.NOT_RELEVANT,        # excluded
        plot_analysis_weight=RelevanceSize.SMALL,             # both ran
        viewer_experience_weight=RelevanceSize.LARGE,         # original-only
        watch_context_weight=RelevanceSize.NOT_RELEVANT,      # promoted subquery-only
        narrative_techniques_weight=RelevanceSize.MEDIUM,     # both ran
        production_weight=RelevanceSize.NOT_RELEVANT,         # excluded
        reception_weight=RelevanceSize.LARGE,                 # both ran
    )
    subqueries = make_vector_subqueries(
        plot_events_subquery=None,
        plot_analysis_subquery="cozy atmosphere",
        viewer_experience_subquery=None,
        watch_context_subquery="rainy day comfort movie",
        narrative_techniques_subquery="gentle pacing and warmth",
        production_subquery=None,
        reception_subquery="beloved soundtrack",
    )
    contexts = context_by_name(build_space_execution_contexts(weights, subqueries))

    expected: dict[VectorName, tuple[bool, bool, RelevanceSize | None]] = {
        VectorName.ANCHOR: (True, False, None),
        VectorName.PLOT_EVENTS: (False, False, RelevanceSize.NOT_RELEVANT),
        VectorName.PLOT_ANALYSIS: (True, True, RelevanceSize.SMALL),
        VectorName.VIEWER_EXPERIENCE: (True, False, RelevanceSize.LARGE),
        VectorName.WATCH_CONTEXT: (False, True, RelevanceSize.SMALL),
        VectorName.NARRATIVE_TECHNIQUES: (True, True, RelevanceSize.MEDIUM),
        VectorName.PRODUCTION: (False, False, RelevanceSize.NOT_RELEVANT),
        VectorName.RECEPTION: (True, True, RelevanceSize.LARGE),
    }

    for space, expected_tuple in expected.items():
        ctx = contexts[space]
        observed = (ctx.did_run_original, ctx.did_run_subquery, ctx.effective_relevance)
        assert observed == expected_tuple


def test_all_non_anchor_inactive_case() -> None:
    """NOT_RELEVANT + no subqueries keeps every non-anchor inactive."""
    contexts = context_by_name(build_space_execution_contexts(make_vector_weights(), make_vector_subqueries()))

    active_spaces = [space for space, ctx in contexts.items() if ctx.is_active]
    assert active_spaces == [VectorName.ANCHOR]
    for space in _non_anchor_names():
        ctx = contexts[space]
        assert ctx.did_run_original is False
        assert ctx.did_run_subquery is False
        assert ctx.effective_relevance == RelevanceSize.NOT_RELEVANT


def test_all_non_anchor_subquery_only_promoted_case() -> None:
    """NOT_RELEVANT + subquery for every non-anchor yields promoted subquery-only contexts."""
    subqueries = make_vector_subqueries(
        plot_events_subquery="a",
        plot_analysis_subquery="b",
        viewer_experience_subquery="c",
        watch_context_subquery="d",
        narrative_techniques_subquery="e",
        production_subquery="f",
        reception_subquery="g",
    )
    contexts = context_by_name(build_space_execution_contexts(make_vector_weights(), subqueries))

    for space in _non_anchor_names():
        ctx = contexts[space]
        assert ctx.did_run_original is False
        assert ctx.did_run_subquery is True
        assert ctx.effective_relevance == RelevanceSize.SMALL


def test_all_non_anchor_original_only_case() -> None:
    """Relevant non-anchor spaces with no subqueries should be original-only."""
    weights = make_vector_weights(
        plot_events_weight=RelevanceSize.SMALL,
        plot_analysis_weight=RelevanceSize.MEDIUM,
        viewer_experience_weight=RelevanceSize.LARGE,
        watch_context_weight=RelevanceSize.SMALL,
        narrative_techniques_weight=RelevanceSize.MEDIUM,
        production_weight=RelevanceSize.LARGE,
        reception_weight=RelevanceSize.SMALL,
    )
    contexts = context_by_name(build_space_execution_contexts(weights, make_vector_subqueries()))

    for space in _non_anchor_names():
        ctx = contexts[space]
        original_relevance = getattr(weights, _weight_attr(space))
        assert original_relevance != RelevanceSize.NOT_RELEVANT
        assert ctx.did_run_original is True
        assert ctx.did_run_subquery is False
        assert ctx.effective_relevance == original_relevance


def test_all_non_anchor_both_ran_case() -> None:
    """Relevant non-anchor spaces with subqueries should run both and keep relevance unchanged."""
    weights = make_vector_weights(
        plot_events_weight=RelevanceSize.SMALL,
        plot_analysis_weight=RelevanceSize.MEDIUM,
        viewer_experience_weight=RelevanceSize.LARGE,
        watch_context_weight=RelevanceSize.SMALL,
        narrative_techniques_weight=RelevanceSize.MEDIUM,
        production_weight=RelevanceSize.LARGE,
        reception_weight=RelevanceSize.SMALL,
    )
    subqueries = make_vector_subqueries(
        plot_events_subquery="events",
        plot_analysis_subquery="analysis",
        viewer_experience_subquery="experience",
        watch_context_subquery="context",
        narrative_techniques_subquery="techniques",
        production_subquery="production",
        reception_subquery="reception",
    )
    contexts = context_by_name(build_space_execution_contexts(weights, subqueries))

    for space in _non_anchor_names():
        ctx = contexts[space]
        original_relevance = getattr(weights, _weight_attr(space))
        assert original_relevance != RelevanceSize.NOT_RELEVANT
        assert ctx.did_run_original is True
        assert ctx.did_run_subquery is True
        assert ctx.effective_relevance == original_relevance


@pytest.mark.parametrize(
    "subquery_text,expected_subquery_flag",
    [
        (None, False),
        ("", True),
        ("   ", True),
        ("line1\nline2", True),
        ("éclair cozy 雨 day", True),
        ("x" * 10000, True),
    ],
)
def test_subquery_text_edge_values(subquery_text: str | None, expected_subquery_flag: bool) -> None:
    """Only None is absent; all other subquery strings count as present."""
    weights = make_vector_weights(plot_events_weight=RelevanceSize.NOT_RELEVANT)
    subqueries = make_vector_subqueries(plot_events_subquery=subquery_text)
    ctx = context_by_name(build_space_execution_contexts(weights, subqueries))[VectorName.PLOT_EVENTS]

    assert ctx.did_run_subquery is expected_subquery_flag
    assert ctx.did_run_original is False
    if expected_subquery_flag:
        assert ctx.effective_relevance == RelevanceSize.SMALL
    else:
        assert ctx.effective_relevance == RelevanceSize.NOT_RELEVANT


def test_function_purity_inputs_not_modified() -> None:
    """Stage 1 should not mutate input dataclasses and should be deterministic."""
    weights = make_vector_weights(
        plot_events_weight=RelevanceSize.MEDIUM,
        production_weight=RelevanceSize.SMALL,
    )
    subqueries = make_vector_subqueries(
        plot_events_subquery="mystery and twists",
        production_subquery="90s practical effects",
    )
    original_weights = replace(weights)
    original_subqueries = replace(subqueries)

    first = build_space_execution_contexts(weights, subqueries)
    second = build_space_execution_contexts(weights, subqueries)

    assert weights == original_weights
    assert subqueries == original_subqueries
    assert first == second
    assert first is not second


def test_uniqueness_and_completeness_of_space_names() -> None:
    """Returned context names should be unique and exactly match VectorName."""
    contexts = build_space_execution_contexts(make_vector_weights(), make_vector_subqueries())
    names = [ctx.name for ctx in contexts]

    assert len(names) == len(set(names))
    assert set(names) == set(VectorName)
    assert set(_non_anchor_names()).issubset(set(names))


def test_effective_relevance_domain_constraints() -> None:
    """Anchor is the only context with None effective_relevance."""
    contexts = build_space_execution_contexts(
        make_vector_weights(
            plot_events_weight=RelevanceSize.NOT_RELEVANT,
            plot_analysis_weight=RelevanceSize.SMALL,
            viewer_experience_weight=RelevanceSize.MEDIUM,
            watch_context_weight=RelevanceSize.LARGE,
            narrative_techniques_weight=RelevanceSize.NOT_RELEVANT,
            production_weight=RelevanceSize.SMALL,
            reception_weight=RelevanceSize.NOT_RELEVANT,
        ),
        make_vector_subqueries(
            plot_events_subquery="promote me",
            plot_analysis_subquery=None,
            viewer_experience_subquery="match",
            watch_context_subquery=None,
            narrative_techniques_subquery=None,
            production_subquery="match",
            reception_subquery=None,
        ),
    )
    by_name = context_by_name(contexts)

    assert by_name[VectorName.ANCHOR].effective_relevance is None
    for space in _non_anchor_names():
        assert by_name[space].effective_relevance is not None
        assert isinstance(by_name[space].effective_relevance, RelevanceSize)
