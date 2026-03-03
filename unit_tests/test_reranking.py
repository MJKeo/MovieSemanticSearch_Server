"""Unit tests for db.reranking (quality prior reranking)."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from types import ModuleType

import pytest

# ── Stub heavy dependencies before importing the module under test ─────
# db.search transitively pulls in qdrant_client, asyncpg, etc. which may
# not be installed in the test environment.  We only need SearchCandidate.

_temporary_stubs: list[str] = []


@dataclass(slots=True)
class _StubSearchCandidate:
    movie_id: int
    vector_score: float = 0.0
    lexical_score: float = 0.0
    metadata_score: float = 0.0
    final_score: float = 0.0
    bucketed_final_score: float = 0.0
    quality_prior: float = 0.0


for _module_name, _attrs in (
    ("db.search", {"SearchCandidate": _StubSearchCandidate}),
):
    try:
        importlib.import_module(_module_name)
    except ModuleNotFoundError:
        _stub_module = ModuleType(_module_name)
        for _attr_name, _attr_value in _attrs.items():
            setattr(_stub_module, _attr_name, _attr_value)
        sys.modules[_module_name] = _stub_module
        _temporary_stubs.append(_module_name)

from db.reranking import (
    BUCKET_PRECISION,
    RECEPTION_CEIL,
    RECEPTION_FLOOR,
    normalize_reception,
    compute_quality_prior,
    rerank_candidates,
)
from db.search import SearchCandidate
from implementation.classes.enums import ReceptionType


# ---------------------------------------------------------------------------
# normalize_reception
# ---------------------------------------------------------------------------

class TestNormalizeReception:
    def test_none_returns_neutral_midpoint(self):
        assert normalize_reception(None) == 0.5

    def test_below_floor_clamps_to_zero(self):
        assert normalize_reception(RECEPTION_FLOOR - 10) == 0.0

    def test_above_ceil_clamps_to_one(self):
        assert normalize_reception(RECEPTION_CEIL + 10) == 1.0

    def test_midpoint_maps_correctly(self):
        midpoint = (RECEPTION_FLOOR + RECEPTION_CEIL) / 2
        assert normalize_reception(midpoint) == pytest.approx(0.5)

    def test_exactly_floor_returns_zero(self):
        assert normalize_reception(RECEPTION_FLOOR) == 0.0

    def test_exactly_ceil_returns_one(self):
        assert normalize_reception(RECEPTION_CEIL) == 1.0

    # --- New: boundary and interpolation tests ---

    def test_quarter_point(self):
        """25% of the [FLOOR, CEIL] range should map to 0.25."""
        # (45 - 30) / 60 = 0.25
        value = RECEPTION_FLOOR + 0.25 * (RECEPTION_CEIL - RECEPTION_FLOOR)
        assert normalize_reception(value) == pytest.approx(0.25)

    def test_three_quarter_point(self):
        """75% of the [FLOOR, CEIL] range should map to 0.75."""
        # (75 - 30) / 60 = 0.75
        value = RECEPTION_FLOOR + 0.75 * (RECEPTION_CEIL - RECEPTION_FLOOR)
        assert normalize_reception(value) == pytest.approx(0.75)

    def test_just_above_floor(self):
        """A score barely inside the range produces a small positive output."""
        result = normalize_reception(RECEPTION_FLOOR + 0.1)
        expected = 0.1 / (RECEPTION_CEIL - RECEPTION_FLOOR)
        assert result == pytest.approx(expected, abs=1e-6)
        assert result > 0.0

    def test_just_below_ceil(self):
        """A score barely below ceil produces output close to but not equal to 1."""
        result = normalize_reception(RECEPTION_CEIL - 0.1)
        rng = RECEPTION_CEIL - RECEPTION_FLOOR
        expected = (rng - 0.1) / rng
        assert result == pytest.approx(expected, abs=1e-6)
        assert result < 1.0

    def test_negative_value(self):
        """Negative scores (theoretically impossible) should clamp to 0.0."""
        assert normalize_reception(-10.0) == 0.0

    def test_zero_value(self):
        """Zero is below floor and should clamp to 0.0."""
        assert normalize_reception(0.0) == 0.0

    def test_very_large_value(self):
        """Extreme outlier above ceil clamps to 1.0."""
        assert normalize_reception(1000.0) == 1.0

    def test_linearity_property(self):
        """Verify the function is linear: equal input spacing yields equal output spacing."""
        out_40 = normalize_reception(40.0)
        out_50 = normalize_reception(50.0)
        out_60 = normalize_reception(60.0)
        # The gap between consecutive outputs should be identical
        assert (out_50 - out_40) == pytest.approx(out_60 - out_50, abs=1e-9)


# ---------------------------------------------------------------------------
# compute_quality_prior
# ---------------------------------------------------------------------------

class TestComputeQualityPrior:
    def test_poorly_received_always_returns_zero(self):
        """Poorly-received preference zeroes out the prior regardless of score."""
        assert compute_quality_prior(85.0, ReceptionType.POORLY_RECEIVED) == 0.0

    def test_poorly_received_with_none_still_zero(self):
        assert compute_quality_prior(None, ReceptionType.POORLY_RECEIVED) == 0.0

    def test_critically_acclaimed_delegates_to_normalize(self):
        expected = normalize_reception(75.0)
        assert compute_quality_prior(75.0, ReceptionType.CRITICALLY_ACCLAIMED) == expected

    def test_no_preference_delegates_to_normalize(self):
        expected = normalize_reception(60.0)
        assert compute_quality_prior(60.0, ReceptionType.NO_PREFERENCE) == expected

    def test_none_score_returns_neutral(self):
        assert compute_quality_prior(None, ReceptionType.NO_PREFERENCE) == 0.5

    # --- New: additional edge cases and cross-method verification ---

    def test_critically_acclaimed_with_none_returns_neutral(self):
        """Missing reception data under acclaimed pref still yields the neutral midpoint."""
        assert compute_quality_prior(None, ReceptionType.CRITICALLY_ACCLAIMED) == 0.5

    def test_critically_acclaimed_at_floor(self):
        """Floor-value score under acclaimed pref normalizes to 0.0, no special-casing."""
        assert compute_quality_prior(RECEPTION_FLOOR, ReceptionType.CRITICALLY_ACCLAIMED) == 0.0

    def test_critically_acclaimed_at_ceil(self):
        """Ceil-value score under acclaimed pref normalizes to 1.0."""
        assert compute_quality_prior(RECEPTION_CEIL, ReceptionType.CRITICALLY_ACCLAIMED) == 1.0

    def test_no_preference_at_floor(self):
        """Floor value with no preference normalizes to 0.0."""
        assert compute_quality_prior(RECEPTION_FLOOR, ReceptionType.NO_PREFERENCE) == 0.0

    def test_no_preference_at_ceil(self):
        """Ceil value with no preference normalizes to 1.0."""
        assert compute_quality_prior(RECEPTION_CEIL, ReceptionType.NO_PREFERENCE) == 1.0

    def test_acclaimed_and_no_preference_produce_same_result(self):
        """Both non-POORLY types delegate identically to normalize_reception."""
        score = 65.0
        acclaimed = compute_quality_prior(score, ReceptionType.CRITICALLY_ACCLAIMED)
        no_pref = compute_quality_prior(score, ReceptionType.NO_PREFERENCE)
        assert acclaimed == no_pref

    def test_poorly_received_with_extreme_high_score(self):
        """Even a perfect score is zeroed out under poorly-received preference."""
        assert compute_quality_prior(100.0, ReceptionType.POORLY_RECEIVED) == 0.0

    def test_poorly_received_with_zero_score(self):
        """A zero score under poorly-received still returns 0.0 (not delegated)."""
        assert compute_quality_prior(0.0, ReceptionType.POORLY_RECEIVED) == 0.0


# ---------------------------------------------------------------------------
# rerank_candidates
# ---------------------------------------------------------------------------

def _candidate(
    movie_id: int,
    final_score: float,
    vector_score: float = 0.0,
    lexical_score: float = 0.0,
    metadata_score: float = 0.0,
) -> SearchCandidate:
    """Helper to build a SearchCandidate with only the fields reranking needs."""
    return SearchCandidate(
        movie_id=movie_id,
        vector_score=vector_score,
        lexical_score=lexical_score,
        metadata_score=metadata_score,
        final_score=final_score,
    )


class TestRerankCandidates:
    def test_same_bucket_sorted_by_quality_prior(self):
        """Within the same relevance bucket, higher reception sorts first."""
        # Both round to 0.55 at BUCKET_PRECISION=2
        c1 = _candidate(movie_id=1, final_score=0.554)
        c2 = _candidate(movie_id=2, final_score=0.553)
        candidates = [c1, c2]
        reception_scores = {1: 40.0, 2: 80.0}

        rerank_candidates(candidates, reception_scores, ReceptionType.NO_PREFERENCE)

        # Movie 2 has higher reception, should come first within the same bucket
        assert candidates[0].movie_id == 2
        assert candidates[1].movie_id == 1

    def test_different_buckets_sorted_by_relevance(self):
        """Candidates in different buckets sort by relevance, not quality."""
        c1 = _candidate(movie_id=1, final_score=0.80)
        c2 = _candidate(movie_id=2, final_score=0.70)
        # Movie 2 has much higher reception, but lower relevance bucket
        candidates = [c2, c1]
        reception_scores = {1: 40.0, 2: 90.0}

        rerank_candidates(candidates, reception_scores, ReceptionType.NO_PREFERENCE)

        assert candidates[0].movie_id == 1
        assert candidates[1].movie_id == 2

    def test_fields_stored_on_candidates(self):
        """bucketed_final_score and quality_prior are written to each candidate."""
        c = _candidate(movie_id=1, final_score=0.7654)
        reception_scores = {1: 60.0}

        rerank_candidates([c], reception_scores, ReceptionType.NO_PREFERENCE)

        assert c.bucketed_final_score == round(0.7654, 2)
        assert c.quality_prior == normalize_reception(60.0)

    def test_returns_none(self):
        """rerank_candidates mutates in-place and returns None (like list.sort)."""
        candidates = [_candidate(movie_id=1, final_score=0.5)]
        result = rerank_candidates(candidates, {1: 70.0}, ReceptionType.NO_PREFERENCE)
        assert result is None

    def test_empty_list(self):
        candidates: list = []
        rerank_candidates(candidates, {}, ReceptionType.NO_PREFERENCE)
        assert candidates == []

    def test_single_candidate(self):
        candidates = [_candidate(movie_id=42, final_score=0.5)]
        rerank_candidates(candidates, {42: 70.0}, ReceptionType.NO_PREFERENCE)
        assert len(candidates) == 1
        assert candidates[0].movie_id == 42

    # --- New: missing & None reception scores ---

    def test_movie_id_missing_from_reception_dict(self):
        """Candidate whose movie_id is absent from reception_scores gets neutral prior."""
        c = _candidate(movie_id=99, final_score=0.5)
        # movie_id=99 is NOT in the dict
        rerank_candidates([c], {}, ReceptionType.NO_PREFERENCE)
        # dict.get() returns None → normalize_reception(None) → 0.5
        assert c.quality_prior == 0.5

    def test_reception_score_explicitly_none(self):
        """An explicitly None reception score should behave the same as missing."""
        c = _candidate(movie_id=1, final_score=0.5)
        rerank_candidates([c], {1: None}, ReceptionType.NO_PREFERENCE)
        assert c.quality_prior == 0.5

    def test_mix_of_present_none_and_missing_scores(self):
        """Realistic scenario: some movies have scores, some are None, some are absent."""
        c_present = _candidate(movie_id=1, final_score=0.50)
        c_none = _candidate(movie_id=2, final_score=0.50)
        c_missing = _candidate(movie_id=3, final_score=0.50)
        candidates = [c_present, c_none, c_missing]
        # movie_id=3 intentionally omitted from dict
        reception_scores = {1: 75.0, 2: None}

        rerank_candidates(candidates, reception_scores, ReceptionType.NO_PREFERENCE)

        assert c_present.quality_prior == normalize_reception(75.0)
        # Both None and missing should resolve to the same neutral midpoint
        assert c_none.quality_prior == 0.5
        assert c_missing.quality_prior == 0.5

    # --- New: bucket boundary precision ---

    def test_scores_straddle_bucket_boundary(self):
        """A 0.002 difference can place candidates in different buckets."""
        # round(0.554, 2) = 0.55, round(0.556, 2) = 0.56
        c_lower = _candidate(movie_id=1, final_score=0.554)
        c_higher = _candidate(movie_id=2, final_score=0.556)
        candidates = [c_lower, c_higher]
        # Give the lower-bucket candidate a much higher reception
        reception_scores = {1: 90.0, 2: 30.0}

        rerank_candidates(candidates, reception_scores, ReceptionType.NO_PREFERENCE)

        # Higher bucket (0.56) wins despite worse reception
        assert candidates[0].movie_id == 2
        assert candidates[0].bucketed_final_score == round(0.556, BUCKET_PRECISION)
        assert candidates[1].movie_id == 1
        assert candidates[1].bucketed_final_score == round(0.554, BUCKET_PRECISION)

    def test_scores_just_inside_same_bucket(self):
        """Scores within 0.003 of each other that round to the same bucket use quality prior."""
        # round(0.551, 2) = 0.55, round(0.554, 2) = 0.55
        c1 = _candidate(movie_id=1, final_score=0.551)
        c2 = _candidate(movie_id=2, final_score=0.554)
        candidates = [c1, c2]
        # c1 has higher reception, so it should win the within-bucket tiebreak
        reception_scores = {1: 85.0, 2: 40.0}

        rerank_candidates(candidates, reception_scores, ReceptionType.NO_PREFERENCE)

        assert candidates[0].bucketed_final_score == candidates[1].bucketed_final_score
        # Higher quality prior (movie 1) sorts first
        assert candidates[0].movie_id == 1

    def test_final_score_zero(self):
        """A candidate with final_score 0.0 gets bucketed to 0.0 and sorts last."""
        c_zero = _candidate(movie_id=1, final_score=0.0)
        c_positive = _candidate(movie_id=2, final_score=0.5)
        candidates = [c_zero, c_positive]
        reception_scores = {1: 90.0, 2: 30.0}

        rerank_candidates(candidates, reception_scores, ReceptionType.NO_PREFERENCE)

        assert c_zero.bucketed_final_score == 0.0
        # Positive bucket wins over zero bucket despite worse reception
        assert candidates[0].movie_id == 2
        assert candidates[1].movie_id == 1

    def test_final_score_one(self):
        """A candidate with final_score 1.0 gets bucketed to 1.0 and sorts first."""
        c_max = _candidate(movie_id=1, final_score=1.0)
        c_mid = _candidate(movie_id=2, final_score=0.5)
        candidates = [c_mid, c_max]
        reception_scores = {1: 30.0, 2: 90.0}

        rerank_candidates(candidates, reception_scores, ReceptionType.NO_PREFERENCE)

        assert c_max.bucketed_final_score == 1.0
        # Max-bucket candidate sorts first despite worse reception
        assert candidates[0].movie_id == 1

    def test_negative_final_score(self):
        """Negative final_scores (possible via genre exclusion penalty) bucket correctly."""
        c_neg = _candidate(movie_id=1, final_score=-0.05)
        c_zero = _candidate(movie_id=2, final_score=0.0)
        candidates = [c_neg, c_zero]
        reception_scores = {1: 90.0, 2: 30.0}

        rerank_candidates(candidates, reception_scores, ReceptionType.NO_PREFERENCE)

        assert c_neg.bucketed_final_score == round(-0.05, BUCKET_PRECISION)
        # Zero bucket is above negative bucket
        assert candidates[0].movie_id == 2
        assert candidates[1].movie_id == 1

    # --- New: multi-candidate sorting correctness ---

    def test_multiple_buckets_multiple_candidates_each(self):
        """6 candidates across 3 buckets: within-bucket quality ordering + cross-bucket relevance."""
        # Bucket 0.80: movies 1 (low reception) and 2 (high reception)
        c1 = _candidate(movie_id=1, final_score=0.801)
        c2 = _candidate(movie_id=2, final_score=0.804)
        # Bucket 0.60: movies 3 (low reception) and 4 (high reception)
        c3 = _candidate(movie_id=3, final_score=0.601)
        c4 = _candidate(movie_id=4, final_score=0.604)
        # Bucket 0.40: movies 5 (low reception) and 6 (high reception)
        c5 = _candidate(movie_id=5, final_score=0.401)
        c6 = _candidate(movie_id=6, final_score=0.404)

        # Scramble the input order
        candidates = [c5, c2, c3, c6, c1, c4]
        reception_scores = {
            1: 35.0, 2: 85.0,  # bucket 0.80
            3: 35.0, 4: 85.0,  # bucket 0.60
            5: 35.0, 6: 85.0,  # bucket 0.40
        }

        rerank_candidates(candidates, reception_scores, ReceptionType.NO_PREFERENCE)

        result_ids = [c.movie_id for c in candidates]
        # Within each bucket: higher reception first (even-numbered movies)
        # Across buckets: 0.80 > 0.60 > 0.40
        assert result_ids == [2, 1, 4, 3, 6, 5]

    def test_all_candidates_same_bucket(self):
        """When all candidates share the same bucket, sorting is entirely by quality prior."""
        # All round to 0.55
        c1 = _candidate(movie_id=1, final_score=0.551)
        c2 = _candidate(movie_id=2, final_score=0.552)
        c3 = _candidate(movie_id=3, final_score=0.553)
        c4 = _candidate(movie_id=4, final_score=0.554)
        candidates = [c1, c2, c3, c4]
        # Reception in descending order: 3 > 1 > 4 > 2
        reception_scores = {1: 70.0, 2: 35.0, 3: 85.0, 4: 50.0}

        rerank_candidates(candidates, reception_scores, ReceptionType.NO_PREFERENCE)

        result_ids = [c.movie_id for c in candidates]
        assert result_ids == [3, 1, 4, 2]

    def test_poorly_received_same_bucket_preserves_relative_order(self):
        """POORLY_RECEIVED zeroes all quality priors; Python's stable sort preserves input order."""
        c1 = _candidate(movie_id=1, final_score=0.551)
        c2 = _candidate(movie_id=2, final_score=0.552)
        c3 = _candidate(movie_id=3, final_score=0.553)
        candidates = [c1, c2, c3]
        reception_scores = {1: 90.0, 2: 50.0, 3: 30.0}

        rerank_candidates(candidates, reception_scores, ReceptionType.POORLY_RECEIVED)

        # All quality_priors are 0.0, same bucket → stable sort preserves input order
        for c in candidates:
            assert c.quality_prior == 0.0
        result_ids = [c.movie_id for c in candidates]
        assert result_ids == [1, 2, 3]

    # --- New: data integrity ---

    def test_original_scores_not_modified(self):
        """Reranking should only set bucketed_final_score and quality_prior, not alter other scores."""
        c = _candidate(
            movie_id=1,
            final_score=0.75,
            vector_score=0.6,
            lexical_score=0.3,
            metadata_score=0.1,
        )
        rerank_candidates([c], {1: 70.0}, ReceptionType.NO_PREFERENCE)

        # Original channel scores must be untouched
        assert c.vector_score == 0.6
        assert c.lexical_score == 0.3
        assert c.metadata_score == 0.1
        assert c.final_score == 0.75

    def test_candidates_initially_in_correct_order(self):
        """Pre-sorted input should remain in the same order after reranking (idempotent)."""
        # Already in optimal order: higher bucket first, higher reception first within bucket
        c1 = _candidate(movie_id=1, final_score=0.80)
        c2 = _candidate(movie_id=2, final_score=0.60)
        c3 = _candidate(movie_id=3, final_score=0.40)
        candidates = [c1, c2, c3]
        reception_scores = {1: 80.0, 2: 70.0, 3: 60.0}

        rerank_candidates(candidates, reception_scores, ReceptionType.NO_PREFERENCE)

        assert [c.movie_id for c in candidates] == [1, 2, 3]

    def test_candidates_initially_in_reverse_order(self):
        """Completely reverse-sorted input should be correctly reordered."""
        c1 = _candidate(movie_id=1, final_score=0.40)
        c2 = _candidate(movie_id=2, final_score=0.60)
        c3 = _candidate(movie_id=3, final_score=0.80)
        candidates = [c1, c2, c3]
        reception_scores = {1: 60.0, 2: 70.0, 3: 80.0}

        rerank_candidates(candidates, reception_scores, ReceptionType.NO_PREFERENCE)

        assert [c.movie_id for c in candidates] == [3, 2, 1]

    # --- New: reception type behavior in reranking context ---

    def test_critically_acclaimed_type_still_uses_quality_prior(self):
        """CRITICALLY_ACCLAIMED delegates to normalize_reception; higher reception wins in same bucket."""
        c1 = _candidate(movie_id=1, final_score=0.551)
        c2 = _candidate(movie_id=2, final_score=0.554)
        candidates = [c1, c2]
        reception_scores = {1: 85.0, 2: 40.0}

        rerank_candidates(candidates, reception_scores, ReceptionType.CRITICALLY_ACCLAIMED)

        # Same bucket (0.55), higher reception on movie 1 → movie 1 first
        assert candidates[0].movie_id == 1
        assert candidates[0].quality_prior > candidates[1].quality_prior

    def test_poorly_received_disables_within_bucket_quality_sort(self):
        """POORLY_RECEIVED zeroes all priors, so reception cannot reorder within a bucket."""
        c1 = _candidate(movie_id=1, final_score=0.551)
        c2 = _candidate(movie_id=2, final_score=0.554)
        candidates = [c1, c2]
        # Vastly different reception scores, but POORLY_RECEIVED zeroes the prior
        reception_scores = {1: 30.0, 2: 90.0}

        rerank_candidates(candidates, reception_scores, ReceptionType.POORLY_RECEIVED)

        # Both priors are 0.0; stable sort preserves the original [c1, c2] order
        assert candidates[0].quality_prior == 0.0
        assert candidates[1].quality_prior == 0.0
        assert candidates[0].movie_id == 1
        assert candidates[1].movie_id == 2
