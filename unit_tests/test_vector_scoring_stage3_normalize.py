"""Unit tests for Stage 3 normalization in db.vector_scoring."""

from __future__ import annotations

import math
import random

import pytest

from db.vector_scoring import normalize_blended_scores


def expected_normalized(blended: dict[int, float], k: float) -> dict[int, float]:
    """Reference implementation from the guide spec.

    All input values must be > 0.0 (matching the production contract).
    """
    if not blended:
        return {}

    s_max = max(blended.values())
    s_min = min(blended.values())
    score_range = s_max - s_min

    if score_range == 0.0:
        return dict.fromkeys(blended, 1.0)

    result: dict[int, float] = {}
    for movie_id, score in blended.items():
        gap = (s_max - score) / score_range
        result[movie_id] = math.exp(-k * gap)
    return result


def test_normalize_empty_input_returns_empty_dict() -> None:
    assert normalize_blended_scores({}) == {}


def test_normalize_single_positive_candidate_is_one() -> None:
    assert normalize_blended_scores({7: 0.42}) == {7: 1.0}


def test_normalize_all_positive_identical_scores_are_one() -> None:
    assert normalize_blended_scores({1: 0.71, 2: 0.71, 3: 0.71}) == {1: 1.0, 2: 1.0, 3: 1.0}


def test_normalize_general_case_matches_formula() -> None:
    blended = {1: 0.81, 2: 0.79, 3: 0.64, 4: 0.10}
    observed = normalize_blended_scores(blended, decay_k=3.0)
    expected = expected_normalized(blended, k=3.0)

    assert set(observed) == set(expected)
    for movie_id in blended:
        assert observed[movie_id] == pytest.approx(expected[movie_id], abs=1e-12)


def test_normalize_best_is_one_and_worst_is_exp_neg_k() -> None:
    blended = {1: 1.0, 2: 0.5}
    normalized = normalize_blended_scores(blended, decay_k=3.0)

    assert normalized[1] == pytest.approx(1.0, abs=1e-12)
    assert normalized[2] == pytest.approx(math.exp(-3.0), abs=1e-12)


def test_normalize_monotonicity_higher_blended_not_lower_normalized() -> None:
    blended = {1: 0.97, 2: 0.8, 3: 0.55, 4: 0.2}
    normalized = normalize_blended_scores(blended, decay_k=3.0)

    ordered = sorted(blended.items(), key=lambda pair: pair[1], reverse=True)
    for (left_id, _), (right_id, _) in zip(ordered, ordered[1:]):
        assert normalized[left_id] >= normalized[right_id]


@pytest.mark.parametrize("k", [1.0, 3.0, 5.0])
def test_normalize_custom_k_spot_checks(k: float) -> None:
    # Produces exact gaps {0.0, 0.1, 0.5, 1.0} with s_max=1.0 and s_min=0.5.
    blended = {1: 1.0, 2: 0.95, 3: 0.75, 4: 0.5}
    normalized = normalize_blended_scores(blended, decay_k=k)

    assert normalized[1] == pytest.approx(1.0, abs=1e-12)
    assert normalized[2] == pytest.approx(math.exp(-k * 0.1), abs=1e-12)
    assert normalized[3] == pytest.approx(math.exp(-k * 0.5), abs=1e-12)
    assert normalized[4] == pytest.approx(math.exp(-k), abs=1e-12)


def test_normalize_zero_score_raises_value_error() -> None:
    blended = {1: 0.7, 2: 0.0, 3: 0.7}
    with pytest.raises(ValueError, match="must be > 0.0"):
        normalize_blended_scores(blended, decay_k=3.0)


def test_normalize_all_zero_scores_raises_value_error() -> None:
    blended = {10: 0.0, 20: 0.0, 30: 0.0}
    with pytest.raises(ValueError, match="must be > 0.0"):
        normalize_blended_scores(blended)


def test_normalize_very_small_range_stays_finite_and_ordered() -> None:
    blended = {1: 0.5000000000001, 2: 0.5, 3: 0.50000000000005}
    normalized = normalize_blended_scores(blended, decay_k=3.0)

    assert all(math.isfinite(v) for v in normalized.values())
    assert normalized[1] >= normalized[3] >= normalized[2]
    assert all(0.0 <= v <= 1.0 for v in normalized.values())


def test_normalize_valid_edge_magnitudes_behave_correctly() -> None:
    blended = {1: 1e-12, 2: 1.0 - 1e-12, 3: 0.5}
    normalized = normalize_blended_scores(blended, decay_k=3.0)

    assert normalized[2] == pytest.approx(1.0, abs=1e-12)
    assert all(0.0 <= v <= 1.0 for v in normalized.values())
    assert all(math.isfinite(v) for v in normalized.values())


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_normalize_non_finite_scores_raise_value_error(bad: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        normalize_blended_scores({1: bad})


@pytest.mark.parametrize("bad", [-0.001, 1.001])
def test_normalize_out_of_domain_scores_raise_value_error(bad: float) -> None:
    with pytest.raises(ValueError, match=r"within \[0.0, 1.0\]"):
        normalize_blended_scores({1: bad})


@pytest.mark.parametrize("bad_k", [0.0, -1.0, math.nan, math.inf])
def test_normalize_invalid_decay_k_raise_value_error(bad_k: float) -> None:
    with pytest.raises(ValueError, match="decay_k must be finite and > 0"):
        normalize_blended_scores({1: 0.5}, decay_k=bad_k)


def test_normalize_non_numeric_decay_k_raises_type_error() -> None:
    with pytest.raises(TypeError, match="decay_k must be a real number"):
        normalize_blended_scores({1: 0.5}, decay_k="3")  # type: ignore[arg-type]


def test_normalize_non_dict_input_raises_type_error() -> None:
    with pytest.raises(TypeError, match="blended must be dict"):
        normalize_blended_scores([(1, 0.5)])  # type: ignore[arg-type]


def test_normalize_return_keys_match_input_keys_exactly() -> None:
    blended = {10: 0.82, 20: 0.15, 30: 0.61, 40: 0.03}
    normalized = normalize_blended_scores(blended, decay_k=3.0)
    assert set(normalized.keys()) == set(blended.keys())


def test_normalize_stress_random_dataset_bounds_and_invariants() -> None:
    rng = random.Random(13)
    blended = {10**15: 0.99}
    for i in range(1, 1200):
        # All values must be > 0.0 per the contract.
        blended[10**15 + i] = rng.uniform(0.001, 1.0)

    normalized = normalize_blended_scores(blended, decay_k=3.0)

    assert set(normalized.keys()) == set(blended.keys())
    assert all(math.isfinite(v) and 0.0 < v <= 1.0 for v in normalized.values())
    best_movie = max(blended, key=blended.get)
    assert normalized[best_movie] == pytest.approx(1.0, abs=1e-12)
