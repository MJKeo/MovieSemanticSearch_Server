"""Unit tests for Stage 5 final weighted scoring in db.vector_scoring."""

from __future__ import annotations

import copy
import itertools
import math
import random

import pytest

from implementation.classes.enums import RelevanceSize, VectorName

from db.vector_scoring import SpaceExecutionContext, compute_final_scores


def make_ctx(
    name: VectorName,
    *,
    did_run_original: bool,
    did_run_subquery: bool,
    weight: float,
) -> SpaceExecutionContext:
    return SpaceExecutionContext(
        name=name,
        did_run_original=did_run_original,
        did_run_subquery=did_run_subquery,
        effective_relevance=None if name == VectorName.ANCHOR else RelevanceSize.SMALL,
        normalized_weight=weight,
    )


def active_contexts_from_weights(weights: dict[VectorName, float]) -> list[SpaceExecutionContext]:
    contexts: list[SpaceExecutionContext] = []
    for name, weight in weights.items():
        if name == VectorName.ANCHOR:
            contexts.append(
                make_ctx(
                    name,
                    did_run_original=True,
                    did_run_subquery=False,
                    weight=weight,
                )
            )
        else:
            contexts.append(
                make_ctx(
                    name,
                    did_run_original=True,
                    did_run_subquery=True,
                    weight=weight,
                )
            )
    return contexts


def assert_scores_close(actual: dict[int, float], expected: dict[int, float]) -> None:
    assert set(actual.keys()) == set(expected.keys())
    for movie_id, expected_score in expected.items():
        assert actual[movie_id] == pytest.approx(expected_score, abs=1e-12)


def test_t01_canonical_stage5_weighted_sum_example_from_guide() -> None:
    all_ids = frozenset({1, 2})
    weights = {
        VectorName.ANCHOR: 0.138,
        VectorName.PLOT_ANALYSIS: 0.086,
        VectorName.VIEWER_EXPERIENCE: 0.259,
        VectorName.WATCH_CONTEXT: 0.259,
        VectorName.PRODUCTION: 0.172,
        VectorName.RECEPTION: 0.086,
    }
    contexts = active_contexts_from_weights(weights)
    per_space = {
        VectorName.ANCHOR: {1: 0.85, 2: 0.05},
        VectorName.PLOT_ANALYSIS: {1: 0.47, 2: 0.08},
        VectorName.VIEWER_EXPERIENCE: {1: 0.92, 2: 0.05},
        VectorName.WATCH_CONTEXT: {1: 0.88},
        VectorName.PRODUCTION: {1: 0.71, 2: 0.52},
        VectorName.RECEPTION: {1: 0.35},
    }

    observed = compute_final_scores(all_ids, contexts, per_space)
    expected = {
        1: (
            0.138 * 0.85
            + 0.086 * 0.47
            + 0.259 * 0.92
            + 0.259 * 0.88
            + 0.172 * 0.71
            + 0.086 * 0.35
        ),
        2: (
            0.138 * 0.05
            + 0.086 * 0.08
            + 0.259 * 0.05
            + 0.172 * 0.52
        ),
    }
    assert_scores_close(observed, expected)


def test_t02_candidate_present_in_all_active_spaces_gets_full_sum() -> None:
    all_ids = frozenset({7})
    contexts = active_contexts_from_weights(
        {
            VectorName.ANCHOR: 0.2,
            VectorName.PLOT_EVENTS: 0.3,
            VectorName.RECEPTION: 0.5,
        }
    )
    per_space = {
        VectorName.ANCHOR: {7: 0.7},
        VectorName.PLOT_EVENTS: {7: 0.8},
        VectorName.RECEPTION: {7: 0.9},
    }

    observed = compute_final_scores(all_ids, contexts, per_space)
    assert_scores_close(observed, {7: 0.83})


def test_t03_candidate_present_in_no_spaces_gets_zero() -> None:
    all_ids = frozenset({1, 2})
    contexts = active_contexts_from_weights(
        {VectorName.ANCHOR: 0.5, VectorName.PRODUCTION: 0.5}
    )
    per_space = {
        VectorName.ANCHOR: {1: 1.0},
        VectorName.PRODUCTION: {1: 0.5},
    }

    observed = compute_final_scores(all_ids, contexts, per_space)
    assert_scores_close(observed, {1: 0.75, 2: 0.0})


def test_t04_candidate_present_in_only_one_space() -> None:
    all_ids = frozenset({11, 12})
    contexts = active_contexts_from_weights(
        {VectorName.ANCHOR: 0.4, VectorName.WATCH_CONTEXT: 0.6}
    )
    per_space = {
        VectorName.ANCHOR: {},
        VectorName.WATCH_CONTEXT: {11: 0.25},
    }

    observed = compute_final_scores(all_ids, contexts, per_space)
    assert_scores_close(observed, {11: 0.15, 12: 0.0})


def test_t05_anchor_only_active_case() -> None:
    all_ids = frozenset({3, 4})
    contexts = active_contexts_from_weights({VectorName.ANCHOR: 1.0})
    per_space = {VectorName.ANCHOR: {3: 0.33}}

    observed = compute_final_scores(all_ids, contexts, per_space)
    assert_scores_close(observed, {3: 0.33, 4: 0.0})


def test_t06_empty_space_dict_contributes_nothing() -> None:
    all_ids = frozenset({1, 2})
    contexts = active_contexts_from_weights(
        {VectorName.ANCHOR: 0.6, VectorName.PLOT_ANALYSIS: 0.4}
    )
    per_space = {
        VectorName.ANCHOR: {1: 1.0},
        VectorName.PLOT_ANALYSIS: {},
    }

    observed = compute_final_scores(all_ids, contexts, per_space)
    assert_scores_close(observed, {1: 0.6, 2: 0.0})


def test_t07_mixed_sparse_coverage_implicit_zero_behavior() -> None:
    all_ids = frozenset({10, 20, 30})
    contexts = active_contexts_from_weights(
        {
            VectorName.ANCHOR: 0.2,
            VectorName.PLOT_EVENTS: 0.3,
            VectorName.RECEPTION: 0.5,
        }
    )
    per_space = {
        VectorName.ANCHOR: {10: 1.0, 20: 0.5},
        VectorName.PLOT_EVENTS: {20: 1.0},
        VectorName.RECEPTION: {30: 0.8},
    }

    observed = compute_final_scores(all_ids, contexts, per_space)
    assert_scores_close(observed, {10: 0.2, 20: 0.4, 30: 0.4})


def test_t08_single_candidate_all_ones_scores_to_one() -> None:
    all_ids = frozenset({99})
    contexts = active_contexts_from_weights(
        {
            VectorName.ANCHOR: 0.1,
            VectorName.PLOT_EVENTS: 0.2,
            VectorName.PLOT_ANALYSIS: 0.3,
            VectorName.VIEWER_EXPERIENCE: 0.4,
        }
    )
    per_space = {
        VectorName.ANCHOR: {99: 1.0},
        VectorName.PLOT_EVENTS: {99: 1.0},
        VectorName.PLOT_ANALYSIS: {99: 1.0},
        VectorName.VIEWER_EXPERIENCE: {99: 1.0},
    }

    observed = compute_final_scores(all_ids, contexts, per_space)
    assert_scores_close(observed, {99: 1.0})


def test_t09_all_zero_or_no_entries_produces_all_zeroes() -> None:
    all_ids = frozenset({1, 2, 3})
    contexts = active_contexts_from_weights(
        {VectorName.ANCHOR: 0.5, VectorName.PLOT_EVENTS: 0.5}
    )
    per_space = {
        VectorName.ANCHOR: {1: 0.0, 2: 0.0},
        VectorName.PLOT_EVENTS: {},
    }

    observed = compute_final_scores(all_ids, contexts, per_space)
    assert_scores_close(observed, {1: 0.0, 2: 0.0, 3: 0.0})


def test_t10_tight_decimal_precision_stress() -> None:
    all_ids = frozenset({1})
    contexts = active_contexts_from_weights(
        {
            VectorName.ANCHOR: 0.333333333333,
            VectorName.PLOT_EVENTS: 0.333333333333,
            VectorName.RECEPTION: 0.333333333334,
        }
    )
    per_space = {
        VectorName.ANCHOR: {1: 0.123456789012},
        VectorName.PLOT_EVENTS: {1: 0.987654321098},
        VectorName.RECEPTION: {1: 0.555555555555},
    }

    observed = compute_final_scores(all_ids, contexts, per_space)
    expected = {
        1: (
            0.333333333333 * 0.123456789012
            + 0.333333333333 * 0.987654321098
            + 0.333333333334 * 0.555555555555
        )
    }
    assert_scores_close(observed, expected)


def test_t11_candidate_ids_large_negative_and_zero_supported() -> None:
    all_ids = frozenset({-7, 0, 10**15})
    contexts = active_contexts_from_weights(
        {VectorName.ANCHOR: 0.4, VectorName.PRODUCTION: 0.6}
    )
    per_space = {
        VectorName.ANCHOR: {-7: 0.5, 10**15: 1.0},
        VectorName.PRODUCTION: {0: 0.25, 10**15: 0.5},
    }

    observed = compute_final_scores(all_ids, contexts, per_space)
    assert_scores_close(
        observed,
        {
            -7: 0.2,
            0: 0.15,
            10**15: 0.7,
        },
    )


def test_t12_deterministic_across_active_context_order_permutations() -> None:
    all_ids = frozenset({1, 2})
    weights = {
        VectorName.ANCHOR: 0.2,
        VectorName.PLOT_EVENTS: 0.3,
        VectorName.RECEPTION: 0.5,
    }
    base_contexts = active_contexts_from_weights(weights)
    per_space = {
        VectorName.ANCHOR: {1: 0.9, 2: 0.1},
        VectorName.PLOT_EVENTS: {1: 0.8},
        VectorName.RECEPTION: {2: 0.4},
    }

    reference = compute_final_scores(all_ids, base_contexts, per_space)
    for perm in itertools.permutations(base_contexts):
        observed = compute_final_scores(all_ids, list(perm), per_space)
        assert observed == reference


def test_t13_deterministic_across_per_space_dict_insertion_order() -> None:
    all_ids = frozenset({1, 2, 3})
    contexts = active_contexts_from_weights(
        {VectorName.ANCHOR: 0.4, VectorName.PLOT_EVENTS: 0.6}
    )

    per_space_a = {
        VectorName.ANCHOR: {1: 0.3, 2: 0.5, 3: 0.0},
        VectorName.PLOT_EVENTS: {1: 0.7, 2: 0.0, 3: 0.9},
    }
    per_space_b = {
        VectorName.PLOT_EVENTS: {3: 0.9, 1: 0.7, 2: 0.0},
        VectorName.ANCHOR: {2: 0.5, 3: 0.0, 1: 0.3},
    }

    assert compute_final_scores(all_ids, contexts, per_space_a) == compute_final_scores(
        all_ids, contexts, per_space_b
    )


def test_t14_output_keyset_exactly_matches_all_candidate_ids() -> None:
    all_ids = frozenset({1, 2, 3, 4})
    contexts = active_contexts_from_weights({VectorName.ANCHOR: 1.0})
    per_space = {VectorName.ANCHOR: {1: 0.2, 2: 0.4}}

    observed = compute_final_scores(all_ids, contexts, per_space)
    assert set(observed.keys()) == set(all_ids)


def test_t15_output_scores_finite_and_within_unit_interval() -> None:
    rng = random.Random(17)
    all_ids = frozenset(range(200))
    contexts = active_contexts_from_weights(
        {
            VectorName.ANCHOR: 0.1,
            VectorName.PLOT_EVENTS: 0.2,
            VectorName.PLOT_ANALYSIS: 0.3,
            VectorName.RECEPTION: 0.4,
        }
    )
    per_space = {
        VectorName.ANCHOR: {},
        VectorName.PLOT_EVENTS: {},
        VectorName.PLOT_ANALYSIS: {},
        VectorName.RECEPTION: {},
    }
    for movie_id in all_ids:
        if movie_id % 2 == 0:
            per_space[VectorName.ANCHOR][movie_id] = rng.random()
        if movie_id % 3 == 0:
            per_space[VectorName.PLOT_EVENTS][movie_id] = rng.random()
        if movie_id % 5 == 0:
            per_space[VectorName.PLOT_ANALYSIS][movie_id] = rng.random()
        if movie_id % 7 == 0:
            per_space[VectorName.RECEPTION][movie_id] = rng.random()

    observed = compute_final_scores(all_ids, contexts, per_space)
    assert all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in observed.values())


def test_t16_inputs_are_not_mutated() -> None:
    all_ids = frozenset({1, 2})
    contexts = active_contexts_from_weights({VectorName.ANCHOR: 0.3, VectorName.PRODUCTION: 0.7})
    per_space = {
        VectorName.ANCHOR: {1: 0.1},
        VectorName.PRODUCTION: {2: 0.9},
    }

    contexts_before = copy.deepcopy(contexts)
    per_space_before = copy.deepcopy(per_space)
    _ = compute_final_scores(all_ids, contexts, per_space)

    assert contexts == contexts_before
    assert per_space == per_space_before


def test_t17_empty_candidate_ids_with_no_contributions_returns_empty_dict() -> None:
    assert compute_final_scores(frozenset(), [], {}) == {}


def test_t18_non_empty_candidates_with_empty_active_contexts_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        compute_final_scores(frozenset({1}), [], {})


def test_t19_inactive_context_in_active_contexts_raises() -> None:
    contexts = [
        make_ctx(
            VectorName.ANCHOR,
            did_run_original=False,
            did_run_subquery=False,
            weight=1.0,
        )
    ]
    with pytest.raises(ValueError, match="must be active"):
        compute_final_scores(frozenset({1}), contexts, {VectorName.ANCHOR: {1: 0.5}})


def test_t20_duplicate_context_names_raise() -> None:
    contexts = [
        make_ctx(VectorName.ANCHOR, did_run_original=True, did_run_subquery=False, weight=0.5),
        make_ctx(VectorName.ANCHOR, did_run_original=True, did_run_subquery=False, weight=0.5),
    ]
    with pytest.raises(ValueError, match="duplicate active context"):
        compute_final_scores(frozenset({1}), contexts, {VectorName.ANCHOR: {1: 0.5}})


def test_t21_missing_per_space_key_for_active_context_raises() -> None:
    contexts = active_contexts_from_weights(
        {VectorName.ANCHOR: 0.5, VectorName.PLOT_EVENTS: 0.5}
    )
    with pytest.raises(ValueError, match="missing spaces"):
        compute_final_scores(
            frozenset({1}),
            contexts,
            {VectorName.ANCHOR: {1: 0.5}},
        )


def test_t22_extra_per_space_key_raises() -> None:
    contexts = active_contexts_from_weights({VectorName.ANCHOR: 1.0})
    with pytest.raises(ValueError, match="extra spaces"):
        compute_final_scores(
            frozenset({1}),
            contexts,
            {
                VectorName.ANCHOR: {1: 0.5},
                VectorName.PLOT_EVENTS: {},
            },
        )


def test_t23_negative_context_weight_raises() -> None:
    contexts = active_contexts_from_weights({VectorName.ANCHOR: -0.1, VectorName.PLOT_EVENTS: 1.1})
    with pytest.raises(ValueError, match=r"within \[0.0, 1.0\]"):
        compute_final_scores(
            frozenset({1}),
            contexts,
            {VectorName.ANCHOR: {1: 0.5}, VectorName.PLOT_EVENTS: {1: 0.5}},
        )


def test_t24_context_weight_greater_than_one_raises() -> None:
    contexts = active_contexts_from_weights({VectorName.ANCHOR: 1.2})
    with pytest.raises(ValueError, match=r"within \[0.0, 1.0\]"):
        compute_final_scores(frozenset({1}), contexts, {VectorName.ANCHOR: {1: 0.5}})


@pytest.mark.parametrize("bad_weight", [math.nan, math.inf, -math.inf])
def test_t25_non_finite_context_weight_raises(bad_weight: float) -> None:
    contexts = active_contexts_from_weights({VectorName.ANCHOR: bad_weight})
    with pytest.raises(ValueError, match="must be finite"):
        compute_final_scores(frozenset({1}), contexts, {VectorName.ANCHOR: {1: 0.5}})


def test_t26_weight_sum_not_one_raises() -> None:
    contexts = active_contexts_from_weights(
        {VectorName.ANCHOR: 0.6, VectorName.PLOT_EVENTS: 0.4000001}
    )
    with pytest.raises(ValueError, match="must sum to 1.0"):
        compute_final_scores(
            frozenset({1}),
            contexts,
            {VectorName.ANCHOR: {1: 0.2}, VectorName.PLOT_EVENTS: {1: 0.3}},
        )


def test_t27_negative_per_space_normalized_score_raises() -> None:
    contexts = active_contexts_from_weights({VectorName.ANCHOR: 1.0})
    with pytest.raises(ValueError, match=r"within \[0.0, 1.0\]"):
        compute_final_scores(frozenset({1}), contexts, {VectorName.ANCHOR: {1: -0.01}})


def test_t28_per_space_normalized_score_above_one_raises() -> None:
    contexts = active_contexts_from_weights({VectorName.ANCHOR: 1.0})
    with pytest.raises(ValueError, match=r"within \[0.0, 1.0\]"):
        compute_final_scores(frozenset({1}), contexts, {VectorName.ANCHOR: {1: 1.01}})


@pytest.mark.parametrize("bad_score", [math.nan, math.inf, -math.inf])
def test_t29_non_finite_per_space_normalized_score_raises(bad_score: float) -> None:
    contexts = active_contexts_from_weights({VectorName.ANCHOR: 1.0})
    with pytest.raises(ValueError, match="must be finite"):
        compute_final_scores(frozenset({1}), contexts, {VectorName.ANCHOR: {1: bad_score}})


def test_t30_per_space_movie_id_not_in_all_candidate_ids_raises() -> None:
    contexts = active_contexts_from_weights({VectorName.ANCHOR: 1.0})
    with pytest.raises(ValueError, match="unknown movie_id"):
        compute_final_scores(frozenset({1}), contexts, {VectorName.ANCHOR: {2: 0.5}})


def test_t31_all_candidate_ids_wrong_type_raises_type_error() -> None:
    contexts = active_contexts_from_weights({VectorName.ANCHOR: 1.0})
    with pytest.raises(TypeError, match="all_candidate_ids must be a set"):
        compute_final_scores([1], contexts, {VectorName.ANCHOR: {1: 0.5}})  # type: ignore[arg-type]


def test_t32_all_candidate_ids_contains_non_int_raises_type_error() -> None:
    contexts = active_contexts_from_weights({VectorName.ANCHOR: 1.0})
    with pytest.raises(TypeError, match="must contain only int"):
        compute_final_scores(frozenset({"1"}), contexts, {VectorName.ANCHOR: {1: 0.5}})  # type: ignore[arg-type]


def test_t33_active_contexts_wrong_container_type_raises_type_error() -> None:
    with pytest.raises(TypeError, match="active_contexts must be list"):
        compute_final_scores(
            frozenset({1}),
            (),
            {VectorName.ANCHOR: {1: 0.5}},
        )  # type: ignore[arg-type]


def test_t34_per_space_normalized_wrong_container_type_raises_type_error() -> None:
    contexts = active_contexts_from_weights({VectorName.ANCHOR: 1.0})
    with pytest.raises(TypeError, match="per_space_normalized must be dict"):
        compute_final_scores(
            frozenset({1}),
            contexts,
            [],
        )  # type: ignore[arg-type]


def test_t35_high_volume_random_smoke_test() -> None:
    rng = random.Random(23)
    all_ids = frozenset(range(5000))
    contexts = active_contexts_from_weights(
        {
            VectorName.ANCHOR: 0.1,
            VectorName.PLOT_EVENTS: 0.2,
            VectorName.PLOT_ANALYSIS: 0.3,
            VectorName.RECEPTION: 0.4,
        }
    )
    per_space = {
        VectorName.ANCHOR: {},
        VectorName.PLOT_EVENTS: {},
        VectorName.PLOT_ANALYSIS: {},
        VectorName.RECEPTION: {},
    }

    for movie_id in all_ids:
        if movie_id % 2 == 0:
            per_space[VectorName.ANCHOR][movie_id] = rng.random()
        if movie_id % 3 == 0:
            per_space[VectorName.PLOT_EVENTS][movie_id] = rng.random()
        if movie_id % 5 == 0:
            per_space[VectorName.PLOT_ANALYSIS][movie_id] = rng.random()
        if movie_id % 7 == 0:
            per_space[VectorName.RECEPTION][movie_id] = rng.random()

    observed = compute_final_scores(all_ids, contexts, per_space)
    assert len(observed) == len(all_ids)
    assert all(math.isfinite(score) and 0.0 <= score <= 1.0 for score in observed.values())
