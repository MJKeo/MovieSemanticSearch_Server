"""Unit tests for Stage 2 blending in db.vector_scoring."""

from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest

from implementation.classes.enums import RelevanceSize, VectorName

from db.vector_scoring import (
    SPACE_CONFIGS,
    SpaceExecutionContext,
    blend_space_scores,
)


def make_candidate(**overrides: float) -> SimpleNamespace:
    """Build a score object with every expected score attribute present."""
    base = {
        "anchor_score_original": 0.0,
        "plot_events_score_original": 0.0,
        "plot_events_score_subquery": 0.0,
        "plot_analysis_score_original": 0.0,
        "plot_analysis_score_subquery": 0.0,
        "viewer_experience_score_original": 0.0,
        "viewer_experience_score_subquery": 0.0,
        "watch_context_score_original": 0.0,
        "watch_context_score_subquery": 0.0,
        "narrative_techniques_score_original": 0.0,
        "narrative_techniques_score_subquery": 0.0,
        "production_score_original": 0.0,
        "production_score_subquery": 0.0,
        "reception_score_original": 0.0,
        "reception_score_subquery": 0.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def config_for(space: VectorName):
    """Get the immutable stage config for a given vector space."""
    return next(c for c in SPACE_CONFIGS if c.name == space)


def test_blend_both_ran_happy_path_and_appendix_d_scenarios() -> None:
    """Both-ran mode should apply 80/20 blending and omit exact zeros."""
    ctx = SpaceExecutionContext(
        name=VectorName.PLOT_EVENTS,
        did_run_original=True,
        did_run_subquery=True,
        effective_relevance=RelevanceSize.MEDIUM,
    )
    candidates = {
        1: make_candidate(plot_events_score_original=0.4, plot_events_score_subquery=0.9),
        2: make_candidate(plot_events_score_original=0.0, plot_events_score_subquery=0.5),
        3: make_candidate(plot_events_score_original=0.5, plot_events_score_subquery=0.0),
        4: make_candidate(plot_events_score_original=0.0, plot_events_score_subquery=0.0),
    }

    blended = blend_space_scores(candidates, config_for(VectorName.PLOT_EVENTS), ctx)

    assert set(blended) == {1, 2, 3}
    assert blended[1] == pytest.approx(0.8 * 0.9 + 0.2 * 0.4, abs=1e-12)
    assert blended[2] == pytest.approx(0.8 * 0.5 + 0.2 * 0.0, abs=1e-12)
    assert blended[3] == pytest.approx(0.8 * 0.0 + 0.2 * 0.5, abs=1e-12)


def test_blend_original_only_happy_path() -> None:
    ctx = SpaceExecutionContext(
        name=VectorName.PLOT_ANALYSIS,
        did_run_original=True,
        did_run_subquery=False,
        effective_relevance=RelevanceSize.SMALL,
    )
    candidates = {
        11: make_candidate(plot_analysis_score_original=0.77),
        22: make_candidate(plot_analysis_score_original=0.0),
    }

    blended = blend_space_scores(candidates, config_for(VectorName.PLOT_ANALYSIS), ctx)
    assert blended == {11: pytest.approx(0.77, abs=1e-12)}


def test_blend_subquery_only_happy_path() -> None:
    ctx = SpaceExecutionContext(
        name=VectorName.WATCH_CONTEXT,
        did_run_original=False,
        did_run_subquery=True,
        effective_relevance=RelevanceSize.SMALL,
    )
    candidates = {
        8: make_candidate(watch_context_score_subquery=0.62),
        9: make_candidate(watch_context_score_subquery=0.0),
    }

    blended = blend_space_scores(candidates, config_for(VectorName.WATCH_CONTEXT), ctx)
    assert blended == {8: pytest.approx(0.62, abs=1e-12)}


def test_blend_empty_candidates_returns_empty_dict() -> None:
    ctx = SpaceExecutionContext(
        name=VectorName.PLOT_EVENTS,
        did_run_original=True,
        did_run_subquery=True,
        effective_relevance=RelevanceSize.LARGE,
    )
    assert blend_space_scores({}, config_for(VectorName.PLOT_EVENTS), ctx) == {}


def test_blend_anchor_contract_original_only() -> None:
    ctx = SpaceExecutionContext(
        name=VectorName.ANCHOR,
        did_run_original=True,
        did_run_subquery=False,
        effective_relevance=None,
    )
    candidates = {
        100: make_candidate(anchor_score_original=0.91),
        101: make_candidate(anchor_score_original=0.0),
    }

    blended = blend_space_scores(candidates, config_for(VectorName.ANCHOR), ctx)
    assert blended == {100: pytest.approx(0.91, abs=1e-12)}


def test_blend_tiny_positive_boundary_is_retained() -> None:
    ctx = SpaceExecutionContext(
        name=VectorName.PRODUCTION,
        did_run_original=True,
        did_run_subquery=False,
        effective_relevance=RelevanceSize.SMALL,
    )
    candidates = {
        1: make_candidate(production_score_original=1e-15),
        2: make_candidate(production_score_original=0.0),
    }

    blended = blend_space_scores(candidates, config_for(VectorName.PRODUCTION), ctx)
    assert blended == {1: pytest.approx(1e-15, abs=0.0)}


def test_blend_deterministic_across_candidate_insertion_order() -> None:
    ctx = SpaceExecutionContext(
        name=VectorName.RECEPTION,
        did_run_original=True,
        did_run_subquery=True,
        effective_relevance=RelevanceSize.MEDIUM,
    )
    a = {
        1: make_candidate(reception_score_original=0.2, reception_score_subquery=0.4),
        2: make_candidate(reception_score_original=0.8, reception_score_subquery=0.6),
        3: make_candidate(reception_score_original=0.0, reception_score_subquery=0.0),
    }
    b = {
        3: make_candidate(reception_score_original=0.0, reception_score_subquery=0.0),
        2: make_candidate(reception_score_original=0.8, reception_score_subquery=0.6),
        1: make_candidate(reception_score_original=0.2, reception_score_subquery=0.4),
    }

    assert blend_space_scores(a, config_for(VectorName.RECEPTION), ctx) == blend_space_scores(
        b, config_for(VectorName.RECEPTION), ctx
    )


def test_blend_invalid_context_neither_search_ran_raises_value_error() -> None:
    ctx = SpaceExecutionContext(
        name=VectorName.PLOT_EVENTS,
        did_run_original=False,
        did_run_subquery=False,
        effective_relevance=RelevanceSize.NOT_RELEVANT,
    )
    with pytest.raises(ValueError, match="neither original nor subquery"):
        blend_space_scores({}, config_for(VectorName.PLOT_EVENTS), ctx)


def test_blend_invalid_config_context_combo_subquery_missing_attr_raises_value_error() -> None:
    ctx = SpaceExecutionContext(
        name=VectorName.ANCHOR,
        did_run_original=False,
        did_run_subquery=True,
        effective_relevance=None,
    )
    with pytest.raises(ValueError, match="requires a subquery score attribute"):
        blend_space_scores({1: make_candidate(anchor_score_original=0.9)}, config_for(VectorName.ANCHOR), ctx)


def test_blend_missing_required_candidate_attribute_raises_type_error() -> None:
    ctx = SpaceExecutionContext(
        name=VectorName.PLOT_EVENTS,
        did_run_original=True,
        did_run_subquery=False,
        effective_relevance=RelevanceSize.SMALL,
    )
    malformed = SimpleNamespace(anchor_score_original=0.5)
    with pytest.raises(TypeError, match="missing required score attribute"):
        blend_space_scores({1: malformed}, config_for(VectorName.PLOT_EVENTS), ctx)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_blend_non_finite_scores_raise_value_error(bad: float) -> None:
    ctx = SpaceExecutionContext(
        name=VectorName.PLOT_ANALYSIS,
        did_run_original=True,
        did_run_subquery=False,
        effective_relevance=RelevanceSize.SMALL,
    )
    with pytest.raises(ValueError, match="must be finite"):
        blend_space_scores(
            {1: make_candidate(plot_analysis_score_original=bad)},
            config_for(VectorName.PLOT_ANALYSIS),
            ctx,
        )


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_blend_out_of_domain_scores_raise_value_error(bad: float) -> None:
    ctx = SpaceExecutionContext(
        name=VectorName.WATCH_CONTEXT,
        did_run_original=False,
        did_run_subquery=True,
        effective_relevance=RelevanceSize.SMALL,
    )
    with pytest.raises(ValueError, match=r"within \[0.0, 1.0\]"):
        blend_space_scores(
            {1: make_candidate(watch_context_score_subquery=bad)},
            config_for(VectorName.WATCH_CONTEXT),
            ctx,
        )


def test_blend_non_dict_candidates_raise_type_error() -> None:
    ctx = SpaceExecutionContext(
        name=VectorName.PLOT_EVENTS,
        did_run_original=True,
        did_run_subquery=True,
        effective_relevance=RelevanceSize.MEDIUM,
    )
    with pytest.raises(TypeError, match="candidates must be dict"):
        blend_space_scores([], config_for(VectorName.PLOT_EVENTS), ctx)  # type: ignore[arg-type]


def test_blend_stress_random_dataset_bounds_and_finiteness() -> None:
    """Stress with long candidate dict and huge ids; outputs stay valid."""
    rng = random.Random(7)
    ctx = SpaceExecutionContext(
        name=VectorName.PRODUCTION,
        did_run_original=True,
        did_run_subquery=True,
        effective_relevance=RelevanceSize.LARGE,
    )
    candidates = {}
    for i in range(1000):
        movie_id = 10**12 + i
        candidates[movie_id] = make_candidate(
            production_score_original=rng.random(),
            production_score_subquery=rng.random(),
        )

    blended = blend_space_scores(candidates, config_for(VectorName.PRODUCTION), ctx)

    assert len(blended) <= len(candidates)
    assert all(movie_id >= 10**12 for movie_id in blended)
    assert all(math.isfinite(score) and 0.0 < score <= 1.0 for score in blended.values())
