"""Full conceptual QA suite for calculate_vector_scores."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from implementation.classes.enums import RelevanceSize, VectorName
from implementation.classes.schemas import VectorSubqueries, VectorWeights

from db.vector_scoring import VectorSearchResult, calculate_vector_scores


# ---------------------------------------------------------------------------
# Conceptual constants (from guides/vector_scoring_guide.md)
# ---------------------------------------------------------------------------

ALL_SPACES = list(VectorName)
NON_ANCHOR_SPACES = [space for space in VectorName if space != VectorName.ANCHOR]

SUBQUERY_BLEND_WEIGHT = 0.8
ORIGINAL_BLEND_WEIGHT = 0.2
DECAY_K = 3.0
ANCHOR_MEAN_FRACTION = 0.8

RELEVANCE_RAW = {
    RelevanceSize.NOT_RELEVANT: 0.0,
    RelevanceSize.SMALL: 1.0,
    RelevanceSize.MEDIUM: 2.0,
    RelevanceSize.LARGE: 3.0,
}


def _candidate_score_attrs() -> dict[str, float]:
    attrs: dict[str, float] = {"anchor_score_original": 0.0}
    for space in NON_ANCHOR_SPACES:
        attrs[f"{space.value}_score_original"] = 0.0
        attrs[f"{space.value}_score_subquery"] = 0.0
    return attrs


BASE_CANDIDATE_SCORES = _candidate_score_attrs()


def weight_attr(space: VectorName) -> str:
    return f"{space.value}_weight"


def subquery_text_attr(space: VectorName) -> str:
    return f"{space.value}_subquery"


def original_score_attr(space: VectorName) -> str:
    return f"{space.value}_score_original"


def subquery_score_attr(space: VectorName) -> str:
    if space == VectorName.ANCHOR:
        raise ValueError("anchor has no subquery score attribute")
    return f"{space.value}_score_subquery"


def make_candidate(**overrides: float) -> SimpleNamespace:
    base = dict(BASE_CANDIDATE_SCORES)
    base.update(overrides)
    return SimpleNamespace(**base)


def make_candidates(movie_ids: list[int]) -> dict[int, SimpleNamespace]:
    return {movie_id: make_candidate() for movie_id in movie_ids}


def make_vector_weights(**overrides: RelevanceSize) -> VectorWeights:
    base = {weight_attr(space): RelevanceSize.NOT_RELEVANT for space in NON_ANCHOR_SPACES}
    base.update(overrides)
    return VectorWeights(**base)


def make_vector_subqueries(**overrides: str | None) -> VectorSubqueries:
    base = {subquery_text_attr(space): None for space in NON_ANCHOR_SPACES}
    base.update(overrides)
    return VectorSubqueries(**base)


def set_space_scores(
    candidates: dict[int, SimpleNamespace],
    space: VectorName,
    *,
    originals: dict[int, float] | None = None,
    subqueries: dict[int, float] | None = None,
) -> None:
    if originals is not None:
        attr = original_score_attr(space)
        for movie_id, score in originals.items():
            setattr(candidates[movie_id], attr, score)
    if subqueries is not None:
        attr = subquery_score_attr(space)
        for movie_id, score in subqueries.items():
            setattr(candidates[movie_id], attr, score)


@dataclass(frozen=True)
class ReferenceContext:
    did_run_original: bool
    did_run_subquery: bool
    effective_relevance: RelevanceSize | None

    @property
    def is_active(self) -> bool:
        return self.did_run_original or self.did_run_subquery


@dataclass(frozen=True)
class ReferenceResult:
    contexts: dict[VectorName, ReferenceContext]
    normalized_weights: dict[VectorName, float]
    per_space_normalized: dict[VectorName, dict[int, float]]
    final_scores: dict[int, float]


def conceptual_contexts(
    vector_weights: VectorWeights,
    vector_subqueries: VectorSubqueries,
) -> dict[VectorName, ReferenceContext]:
    contexts: dict[VectorName, ReferenceContext] = {
        VectorName.ANCHOR: ReferenceContext(
            did_run_original=True,
            did_run_subquery=False,
            effective_relevance=None,
        )
    }
    for space in NON_ANCHOR_SPACES:
        relevance = getattr(vector_weights, weight_attr(space))
        subquery_text = getattr(vector_subqueries, subquery_text_attr(space))

        did_run_original = relevance != RelevanceSize.NOT_RELEVANT
        did_run_subquery = subquery_text is not None
        if relevance == RelevanceSize.NOT_RELEVANT and subquery_text is not None:
            effective_relevance = RelevanceSize.SMALL
        else:
            effective_relevance = relevance

        contexts[space] = ReferenceContext(
            did_run_original=did_run_original,
            did_run_subquery=did_run_subquery,
            effective_relevance=effective_relevance,
        )
    return contexts


def conceptual_normalized_weights(
    contexts: dict[VectorName, ReferenceContext],
) -> dict[VectorName, float]:
    raw_weights: dict[VectorName, float] = {}
    active_non_anchor_raw: list[float] = []

    for space in ALL_SPACES:
        if space == VectorName.ANCHOR:
            raw_weights[space] = 0.0
            continue

        ctx = contexts[space]
        if not ctx.is_active:
            raw = 0.0
        else:
            raw = RELEVANCE_RAW[ctx.effective_relevance]  # type: ignore[index]
        raw_weights[space] = raw
        if raw > 0.0:
            active_non_anchor_raw.append(raw)

    if not active_non_anchor_raw:
        raw_weights[VectorName.ANCHOR] = 1.0
    else:
        mean_active = sum(active_non_anchor_raw) / len(active_non_anchor_raw)
        raw_weights[VectorName.ANCHOR] = mean_active * ANCHOR_MEAN_FRACTION

    total_raw = sum(raw_weights.values())
    return {space: raw_weights[space] / total_raw for space in ALL_SPACES}


def conceptual_blended_scores(
    candidates: dict[int, SimpleNamespace],
    space: VectorName,
    ctx: ReferenceContext,
) -> dict[int, float]:
    if not ctx.is_active:
        return {}

    blended: dict[int, float] = {}
    for movie_id, scores in candidates.items():
        original = float(getattr(scores, original_score_attr(space)))
        subquery = (
            float(getattr(scores, subquery_score_attr(space)))
            if space != VectorName.ANCHOR
            else 0.0
        )

        if ctx.did_run_original and ctx.did_run_subquery:
            value = SUBQUERY_BLEND_WEIGHT * subquery + ORIGINAL_BLEND_WEIGHT * original
        elif ctx.did_run_original:
            value = original
        else:
            value = subquery

        if value > 0.0:
            blended[movie_id] = value
    return blended


def conceptual_normalize_space(blended: dict[int, float]) -> dict[int, float]:
    if not blended:
        return {}

    s_max = max(blended.values())
    s_min = min(blended.values())
    score_range = s_max - s_min

    if score_range == 0.0:
        return {movie_id: 1.0 for movie_id in blended}

    normalized: dict[int, float] = {}
    for movie_id, score in blended.items():
        gap = (s_max - score) / score_range
        normalized[movie_id] = math.exp(-DECAY_K * gap)
    return normalized


def conceptual_reference(
    candidates: dict[int, SimpleNamespace],
    vector_weights: VectorWeights,
    vector_subqueries: VectorSubqueries,
) -> ReferenceResult:
    contexts = conceptual_contexts(vector_weights, vector_subqueries)
    normalized_weights = conceptual_normalized_weights(contexts)

    per_space_normalized: dict[VectorName, dict[int, float]] = {}
    for space in ALL_SPACES:
        ctx = contexts[space]
        if not ctx.is_active:
            continue
        blended = conceptual_blended_scores(candidates, space, ctx)
        per_space_normalized[space] = conceptual_normalize_space(blended)

    final_scores = {movie_id: 0.0 for movie_id in candidates}
    for space, normalized in per_space_normalized.items():
        weight = normalized_weights[space]
        for movie_id, norm_score in normalized.items():
            final_scores[movie_id] += weight * norm_score

    return ReferenceResult(
        contexts=contexts,
        normalized_weights=normalized_weights,
        per_space_normalized=per_space_normalized,
        final_scores=final_scores,
    )


def context_map(result) -> dict[VectorName, object]:
    return {ctx.name: ctx for ctx in result.space_contexts}


def assert_global_invariants(result, candidates: dict[int, SimpleNamespace]) -> None:
    # 1. final score keyset matches candidates exactly.
    assert set(result.final_scores.keys()) == set(candidates.keys())

    # 2. final scores are bounded in [0, 1].
    assert all(0.0 <= score <= 1.0 for score in result.final_scores.values())

    # 3. exactly 8 contexts in stable enum order.
    assert len(result.space_contexts) == 8
    assert [ctx.name for ctx in result.space_contexts] == ALL_SPACES

    # 4. normalized weights sum to 1.0.
    weight_sum = sum(ctx.normalized_weight for ctx in result.space_contexts)
    assert math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-9)

    # 5/6. inactive spaces weight 0; active spaces weight > 0.
    for ctx in result.space_contexts:
        if ctx.is_active:
            assert ctx.normalized_weight > 0.0
        else:
            assert ctx.normalized_weight == pytest.approx(0.0, abs=1e-12)

    # 7/8. anchor invariants.
    anchor = context_map(result)[VectorName.ANCHOR]
    assert anchor.did_run_original is True
    assert anchor.did_run_subquery is False
    assert anchor.effective_relevance is None

    # 9. per_space_normalized keys are only active spaces.
    active_spaces = {ctx.name for ctx in result.space_contexts if ctx.is_active}
    assert set(result.per_space_normalized.keys()).issubset(active_spaces)

    # 10/11. sparse normalized values are in (0,1], and each non-empty space has a 1.0.
    for space, scores in result.per_space_normalized.items():
        assert space in active_spaces
        assert all(0.0 < value <= 1.0 for value in scores.values())
        if scores:
            assert any(value == 1.0 for value in scores.values())


def run_case(
    candidates: dict[int, SimpleNamespace],
    *,
    vector_weights: VectorWeights | None = None,
    vector_subqueries: VectorSubqueries | None = None,
    weight_overrides: dict[str, RelevanceSize] | None = None,
    subquery_overrides: dict[str, str | None] | None = None,
):
    if vector_weights is None:
        vector_weights = make_vector_weights(**(weight_overrides or {}))
    if vector_subqueries is None:
        vector_subqueries = make_vector_subqueries(**(subquery_overrides or {}))

    result = calculate_vector_scores(
        VectorSearchResult(
            candidates=candidates,
            vector_weights=vector_weights,
            vector_subqueries=vector_subqueries,
            debug=None,
        )
    )
    assert_global_invariants(result, candidates)
    return result, vector_weights, vector_subqueries


def assert_matches_reference(
    result,
    candidates: dict[int, SimpleNamespace],
    vector_weights: VectorWeights,
    vector_subqueries: VectorSubqueries,
    *,
    abs_tol: float = 1e-12,
) -> ReferenceResult:
    ref = conceptual_reference(candidates, vector_weights, vector_subqueries)
    observed_contexts = context_map(result)

    for space in ALL_SPACES:
        observed = observed_contexts[space]
        expected = ref.contexts[space]
        assert observed.did_run_original is expected.did_run_original
        assert observed.did_run_subquery is expected.did_run_subquery
        assert observed.effective_relevance == expected.effective_relevance
        assert observed.normalized_weight == pytest.approx(
            ref.normalized_weights[space], abs=abs_tol
        )

    assert set(result.per_space_normalized.keys()) == set(ref.per_space_normalized.keys())
    for space in ref.per_space_normalized:
        assert set(result.per_space_normalized[space].keys()) == set(
            ref.per_space_normalized[space].keys()
        )
        for movie_id, expected_score in ref.per_space_normalized[space].items():
            assert result.per_space_normalized[space][movie_id] == pytest.approx(
                expected_score, abs=abs_tol
            )

    assert set(result.final_scores.keys()) == set(ref.final_scores.keys())
    for movie_id, expected_score in ref.final_scores.items():
        assert result.final_scores[movie_id] == pytest.approx(expected_score, abs=abs_tol)

    return ref


def freeze_result(result) -> tuple[object, ...]:
    final_scores = tuple(sorted(result.final_scores.items()))
    contexts = tuple(
        (
            ctx.name.value,
            ctx.did_run_original,
            ctx.did_run_subquery,
            None if ctx.effective_relevance is None else ctx.effective_relevance.value,
            ctx.normalized_weight,
        )
        for ctx in result.space_contexts
    )
    per_space = tuple(
        (
            space.value,
            tuple(sorted(scores.items())),
        )
        for space, scores in sorted(
            result.per_space_normalized.items(),
            key=lambda pair: pair[0].value,
        )
    )
    return final_scores, contexts, per_space


def space_weight(result, space: VectorName) -> float:
    return context_map(result)[space].normalized_weight


# ---------------------------------------------------------------------------
# 2 — Empty and Minimal Input Boundaries
# ---------------------------------------------------------------------------


def test_case_12_empty_candidates_dict() -> None:
    result, _, _ = run_case({})
    assert result.final_scores == {}
    assert result.per_space_normalized == {}
    assert len(result.space_contexts) == 8


def test_case_13_single_candidate_positive_anchor_only() -> None:
    candidates = {101: make_candidate(anchor_score_original=0.77)}
    result, vector_weights, vector_subqueries = run_case(candidates)

    assert result.final_scores[101] == pytest.approx(1.0, abs=1e-12)
    assert result.per_space_normalized[VectorName.ANCHOR] == {101: 1.0}
    assert_matches_reference(result, candidates, vector_weights, vector_subqueries)


def test_case_14_single_candidate_positive_multiple_active_spaces() -> None:
    candidates = {
        7: make_candidate(
            anchor_score_original=0.33,
            viewer_experience_score_original=0.44,
            viewer_experience_score_subquery=0.88,
            production_score_original=0.51,
        )
    }
    result, vector_weights, vector_subqueries = run_case(
        candidates,
        weight_overrides={
            "viewer_experience_weight": RelevanceSize.LARGE,
            "production_weight": RelevanceSize.MEDIUM,
        },
        subquery_overrides={"viewer_experience_subquery": "cozy emotional texture"},
    )

    assert result.final_scores[7] == pytest.approx(1.0, abs=1e-12)
    assert_matches_reference(result, candidates, vector_weights, vector_subqueries)


def test_case_15_single_candidate_all_zero_scores_lexical_only() -> None:
    candidates = {11: make_candidate()}
    result, _, _ = run_case(candidates)

    assert result.final_scores[11] == pytest.approx(0.0, abs=1e-12)
    assert VectorName.ANCHOR in result.per_space_normalized
    assert 11 not in result.per_space_normalized[VectorName.ANCHOR]


def test_case_16_two_candidates_one_all_zero_one_positive() -> None:
    candidates = {
        1: make_candidate(),
        2: make_candidate(anchor_score_original=0.82),
    }
    result, _, _ = run_case(candidates)

    assert result.final_scores[1] == pytest.approx(0.0, abs=1e-12)
    assert result.final_scores[2] > 0.0


# ---------------------------------------------------------------------------
# 3 — Space Execution Flags (Stage 1 correctness)
# ---------------------------------------------------------------------------


def _anchor_positive_candidates() -> dict[int, SimpleNamespace]:
    return {
        1: make_candidate(anchor_score_original=0.9),
        2: make_candidate(anchor_score_original=0.6),
    }


def test_case_17_non_anchor_not_relevant_no_subquery_flags() -> None:
    candidates = _anchor_positive_candidates()
    result, _, _ = run_case(candidates)

    ctx = context_map(result)[VectorName.PLOT_EVENTS]
    assert ctx.did_run_original is False
    assert ctx.did_run_subquery is False
    assert ctx.effective_relevance == RelevanceSize.NOT_RELEVANT
    assert ctx.is_active is False


def test_case_18_non_anchor_not_relevant_with_subquery_promoted() -> None:
    candidates = _anchor_positive_candidates()
    result, _, _ = run_case(
        candidates,
        subquery_overrides={"plot_events_subquery": "character-driven turning points"},
    )

    ctx = context_map(result)[VectorName.PLOT_EVENTS]
    assert ctx.did_run_original is False
    assert ctx.did_run_subquery is True
    assert ctx.effective_relevance == RelevanceSize.SMALL
    assert ctx.is_active is True


def test_case_19_non_anchor_small_no_subquery() -> None:
    candidates = _anchor_positive_candidates()
    result, _, _ = run_case(
        candidates,
        weight_overrides={"plot_events_weight": RelevanceSize.SMALL},
    )

    ctx = context_map(result)[VectorName.PLOT_EVENTS]
    assert ctx.did_run_original is True
    assert ctx.did_run_subquery is False
    assert ctx.effective_relevance == RelevanceSize.SMALL
    assert ctx.is_active is True


def test_case_20_non_anchor_small_with_subquery() -> None:
    candidates = _anchor_positive_candidates()
    result, _, _ = run_case(
        candidates,
        weight_overrides={"plot_events_weight": RelevanceSize.SMALL},
        subquery_overrides={"plot_events_subquery": "cause and effect beats"},
    )

    ctx = context_map(result)[VectorName.PLOT_EVENTS]
    assert ctx.did_run_original is True
    assert ctx.did_run_subquery is True
    assert ctx.effective_relevance == RelevanceSize.SMALL
    assert ctx.is_active is True


@pytest.mark.parametrize("subquery_text", [None, "mid-level thematic motif"])
def test_case_21_non_anchor_medium_variants(subquery_text: str | None) -> None:
    candidates = _anchor_positive_candidates()
    overrides = {} if subquery_text is None else {"plot_events_subquery": subquery_text}
    result, _, _ = run_case(
        candidates,
        weight_overrides={"plot_events_weight": RelevanceSize.MEDIUM},
        subquery_overrides=overrides,
    )

    ctx = context_map(result)[VectorName.PLOT_EVENTS]
    assert ctx.did_run_original is True
    assert ctx.did_run_subquery is (subquery_text is not None)
    assert ctx.effective_relevance == RelevanceSize.MEDIUM
    assert ctx.is_active is True


@pytest.mark.parametrize("subquery_text", [None, "high-signal emotional framing"])
def test_case_22_non_anchor_large_variants(subquery_text: str | None) -> None:
    candidates = _anchor_positive_candidates()
    overrides = {} if subquery_text is None else {"plot_events_subquery": subquery_text}
    result, _, _ = run_case(
        candidates,
        weight_overrides={"plot_events_weight": RelevanceSize.LARGE},
        subquery_overrides=overrides,
    )

    ctx = context_map(result)[VectorName.PLOT_EVENTS]
    assert ctx.did_run_original is True
    assert ctx.did_run_subquery is (subquery_text is not None)
    assert ctx.effective_relevance == RelevanceSize.LARGE
    assert ctx.is_active is True


def test_case_23_all_non_anchor_not_relevant_no_subqueries_only_anchor_active() -> None:
    candidates = _anchor_positive_candidates()
    result, _, _ = run_case(candidates)

    active_spaces = {ctx.name for ctx in result.space_contexts if ctx.is_active}
    assert active_spaces == {VectorName.ANCHOR}
    assert space_weight(result, VectorName.ANCHOR) == pytest.approx(1.0, abs=1e-12)


def test_case_24_all_non_anchor_promoted_small_subquery_only() -> None:
    candidates = _anchor_positive_candidates()
    result, _, _ = run_case(
        candidates,
        subquery_overrides={
            "plot_events_subquery": "a",
            "plot_analysis_subquery": "b",
            "viewer_experience_subquery": "c",
            "watch_context_subquery": "d",
            "narrative_techniques_subquery": "e",
            "production_subquery": "f",
            "reception_subquery": "g",
        },
    )

    for space in NON_ANCHOR_SPACES:
        ctx = context_map(result)[space]
        assert ctx.did_run_original is False
        assert ctx.did_run_subquery is True
        assert ctx.effective_relevance == RelevanceSize.SMALL
        assert ctx.is_active is True


@pytest.mark.parametrize("target_space", NON_ANCHOR_SPACES)
def test_case_25_each_non_anchor_space_independently_active_with_anchor(
    target_space: VectorName,
) -> None:
    candidates = _anchor_positive_candidates()
    result, _, _ = run_case(
        candidates,
        weight_overrides={weight_attr(target_space): RelevanceSize.LARGE},
        subquery_overrides={subquery_text_attr(target_space): "focused subquery"},
    )

    active_spaces = {ctx.name for ctx in result.space_contexts if ctx.is_active}
    assert active_spaces == {VectorName.ANCHOR, target_space}
    for space in NON_ANCHOR_SPACES:
        if space != target_space:
            assert context_map(result)[space].is_active is False


# ---------------------------------------------------------------------------
# 4 — Blend Logic (Stage 2 correctness observed through final scores)
# ---------------------------------------------------------------------------


def _blend_case_base(space: VectorName) -> dict[int, SimpleNamespace]:
    candidates = make_candidates([1, 2])
    candidates[1].anchor_score_original = 0.8
    candidates[2].anchor_score_original = 0.8
    set_space_scores(
        candidates,
        space,
        originals={1: 0.0, 2: 0.0},
        subqueries={1: 0.0, 2: 0.0} if space != VectorName.ANCHOR else None,
    )
    return candidates


def test_case_26_both_ran_candidate_in_both_searches_blend_ordering() -> None:
    space = VectorName.PLOT_EVENTS
    candidates = _blend_case_base(space)
    set_space_scores(candidates, space, originals={1: 0.50, 2: 0.50}, subqueries={1: 0.90, 2: 0.50})

    result, vector_weights, vector_subqueries = run_case(
        candidates,
        weight_overrides={"plot_events_weight": RelevanceSize.MEDIUM},
        subquery_overrides={"plot_events_subquery": "subquery exists"},
    )

    assert result.final_scores[1] > result.final_scores[2]
    assert_matches_reference(result, candidates, vector_weights, vector_subqueries)


def test_case_27_both_ran_candidate_only_subquery_penalized_vs_both() -> None:
    space = VectorName.PLOT_EVENTS
    candidates = _blend_case_base(space)
    set_space_scores(candidates, space, originals={1: 0.0, 2: 0.8}, subqueries={1: 0.8, 2: 0.8})

    result, _, _ = run_case(
        candidates,
        weight_overrides={"plot_events_weight": RelevanceSize.MEDIUM},
        subquery_overrides={"plot_events_subquery": "subquery exists"},
    )

    assert result.final_scores[2] > result.final_scores[1]


def test_case_28_both_ran_candidate_only_original_heavily_penalized() -> None:
    space = VectorName.PLOT_EVENTS
    candidates = _blend_case_base(space)
    set_space_scores(candidates, space, originals={1: 0.8, 2: 0.4}, subqueries={1: 0.0, 2: 0.4})

    result, _, _ = run_case(
        candidates,
        weight_overrides={"plot_events_weight": RelevanceSize.MEDIUM},
        subquery_overrides={"plot_events_subquery": "subquery exists"},
    )

    assert result.final_scores[2] > result.final_scores[1]


def test_case_29_original_only_ordering_reflects_original_scores() -> None:
    space = VectorName.PLOT_EVENTS
    candidates = _blend_case_base(space)
    set_space_scores(candidates, space, originals={1: 0.9, 2: 0.2})

    result, _, _ = run_case(
        candidates,
        weight_overrides={"plot_events_weight": RelevanceSize.LARGE},
    )

    assert context_map(result)[space].did_run_subquery is False
    assert result.final_scores[1] > result.final_scores[2]


def test_case_30_subquery_only_promoted_small_ordering_reflects_subquery_scores() -> None:
    space = VectorName.PLOT_EVENTS
    candidates = _blend_case_base(space)
    set_space_scores(candidates, space, subqueries={1: 0.7, 2: 0.3})

    result, _, _ = run_case(
        candidates,
        subquery_overrides={"plot_events_subquery": "subquery exists"},
    )

    ctx = context_map(result)[space]
    assert ctx.did_run_original is False
    assert ctx.did_run_subquery is True
    assert ctx.effective_relevance == RelevanceSize.SMALL
    assert result.final_scores[1] > result.final_scores[2]


def test_case_31_both_ran_identical_blended_scores_yield_identical_finals() -> None:
    space = VectorName.PLOT_EVENTS
    candidates = _blend_case_base(space)
    # Use inputs whose blended values are bit-identical in IEEE 754:
    # A: 0.8*0.6 + 0.2*0.6 = 0.6
    # B: 0.8*0.6 + 0.2*0.6 = 0.6
    set_space_scores(candidates, space, originals={1: 0.6, 2: 0.6}, subqueries={1: 0.6, 2: 0.6})

    result, _, _ = run_case(
        candidates,
        weight_overrides={"plot_events_weight": RelevanceSize.MEDIUM},
        subquery_overrides={"plot_events_subquery": "subquery exists"},
    )

    assert result.final_scores[1] == pytest.approx(result.final_scores[2], abs=1e-12)


def test_case_32_blend_precision_verifies_80_20_ratio_numerically() -> None:
    space = VectorName.PLOT_EVENTS
    candidates = _blend_case_base(space)
    set_space_scores(candidates, space, originals={1: 0.2, 2: 0.8}, subqueries={1: 0.7, 2: 0.2})

    result, vector_weights, vector_subqueries = run_case(
        candidates,
        weight_overrides={"plot_events_weight": RelevanceSize.MEDIUM},
        subquery_overrides={"plot_events_subquery": "subquery exists"},
    )

    ref = assert_matches_reference(result, candidates, vector_weights, vector_subqueries)
    blended_1 = SUBQUERY_BLEND_WEIGHT * 0.7 + ORIGINAL_BLEND_WEIGHT * 0.2
    blended_2 = SUBQUERY_BLEND_WEIGHT * 0.2 + ORIGINAL_BLEND_WEIGHT * 0.8
    assert blended_1 == pytest.approx(0.6, abs=1e-12)
    assert blended_2 == pytest.approx(0.32, abs=1e-12)
    assert result.final_scores[1] == pytest.approx(ref.final_scores[1], abs=1e-12)
    assert result.final_scores[2] == pytest.approx(ref.final_scores[2], abs=1e-12)


# ---------------------------------------------------------------------------
# 5 — Normalization Behavior (Stage 3 correctness)
# ---------------------------------------------------------------------------


def _normalization_case(scores: dict[int, float]) -> tuple[object, dict[int, SimpleNamespace]]:
    space = VectorName.PLOT_ANALYSIS
    candidates = make_candidates(sorted(scores.keys()))
    for movie_id in scores:
        candidates[movie_id].anchor_score_original = 0.7
    set_space_scores(candidates, space, originals=scores)
    result, _, _ = run_case(
        candidates,
        weight_overrides={"plot_analysis_weight": RelevanceSize.SMALL},
    )
    return result, candidates


def test_case_33_identical_blended_scores_all_normalize_to_one() -> None:
    result, _ = _normalization_case({1: 0.71, 2: 0.71, 3: 0.71})
    normalized = result.per_space_normalized[VectorName.PLOT_ANALYSIS]
    assert normalized == {1: 1.0, 2: 1.0, 3: 1.0}


def test_case_34_two_candidates_wide_gap_worst_is_exp_neg_3() -> None:
    result, _ = _normalization_case({1: 1.0, 2: 0.5})
    normalized = result.per_space_normalized[VectorName.PLOT_ANALYSIS]
    assert normalized[1] == pytest.approx(1.0, abs=1e-12)
    assert normalized[2] == pytest.approx(math.exp(-3.0), abs=1e-12)


def test_case_35_tight_cluster_scores_follow_exponential_decay_formula() -> None:
    scores = {1: 0.80, 2: 0.79, 3: 0.78, 4: 0.77, 5: 0.76}
    result, _ = _normalization_case(scores)
    normalized = result.per_space_normalized[VectorName.PLOT_ANALYSIS]

    s_max = 0.80
    s_min = 0.76
    score_range = s_max - s_min
    for movie_id, score in scores.items():
        gap = (s_max - score) / score_range
        assert normalized[movie_id] == pytest.approx(math.exp(-3.0 * gap), abs=1e-12)


def test_case_36_wide_spread_dropoff_steep_worst_near_point_zero_five() -> None:
    scores = {1: 0.90, 2: 0.70, 3: 0.50, 4: 0.30, 5: 0.10}
    result, _ = _normalization_case(scores)
    normalized = result.per_space_normalized[VectorName.PLOT_ANALYSIS]

    assert normalized[1] == pytest.approx(1.0, abs=1e-12)
    assert normalized[5] == pytest.approx(math.exp(-3.0), abs=1e-12)
    assert normalized[5] < 0.06


@pytest.mark.parametrize("pool_size", [2, 10, 100])
def test_case_37_best_candidate_exactly_one_for_varied_pool_sizes(pool_size: int) -> None:
    scores = {i: 1.0 - i / (pool_size * 1000.0) for i in range(pool_size)}
    result, _ = _normalization_case(scores)
    normalized = result.per_space_normalized[VectorName.PLOT_ANALYSIS]
    assert max(normalized.values()) == 1.0


def test_case_38_single_positive_candidate_in_pool_gets_one_and_others_absent() -> None:
    result, _ = _normalization_case({1: 0.82, 2: 0.0, 3: 0.0})
    normalized = result.per_space_normalized[VectorName.PLOT_ANALYSIS]
    assert normalized == {1: 1.0}
    assert 2 not in normalized
    assert 3 not in normalized


def test_case_39_epsilon_apart_scores_behave_per_gap_formula() -> None:
    result, _ = _normalization_case({1: 0.500001, 2: 0.500000})
    normalized = result.per_space_normalized[VectorName.PLOT_ANALYSIS]
    assert normalized[1] == pytest.approx(1.0, abs=1e-12)
    assert normalized[2] == pytest.approx(math.exp(-3.0), abs=1e-12)


def test_case_40_small_absolute_magnitudes_normalize_by_relative_distance() -> None:
    result, _ = _normalization_case({1: 0.002, 2: 0.001})
    normalized = result.per_space_normalized[VectorName.PLOT_ANALYSIS]
    assert normalized[1] == pytest.approx(1.0, abs=1e-12)
    assert normalized[2] == pytest.approx(math.exp(-3.0), abs=1e-12)


# ---------------------------------------------------------------------------
# 6 — Weight Computation (Stage 4 correctness)
# ---------------------------------------------------------------------------


def _single_positive_candidate() -> dict[int, SimpleNamespace]:
    return {1: make_candidate(anchor_score_original=0.9)}


def test_case_41_all_non_anchor_large_weight_distribution() -> None:
    candidates = _single_positive_candidate()
    result, _, _ = run_case(
        candidates,
        weight_overrides={weight_attr(space): RelevanceSize.LARGE for space in NON_ANCHOR_SPACES},
    )

    expected_anchor = 2.4 / 23.4
    expected_non_anchor = 3.0 / 23.4
    assert space_weight(result, VectorName.ANCHOR) == pytest.approx(expected_anchor, abs=1e-12)
    for space in NON_ANCHOR_SPACES:
        assert space_weight(result, space) == pytest.approx(expected_non_anchor, abs=1e-12)


def test_case_42_all_non_anchor_small_weight_distribution() -> None:
    candidates = _single_positive_candidate()
    result, _, _ = run_case(
        candidates,
        weight_overrides={weight_attr(space): RelevanceSize.SMALL for space in NON_ANCHOR_SPACES},
    )

    expected_anchor = 0.8 / 7.8
    expected_non_anchor = 1.0 / 7.8
    assert space_weight(result, VectorName.ANCHOR) == pytest.approx(expected_anchor, abs=1e-12)
    for space in NON_ANCHOR_SPACES:
        assert space_weight(result, space) == pytest.approx(expected_non_anchor, abs=1e-12)


def test_case_43_mixed_large_medium_small_and_four_inactive_weights() -> None:
    candidates = _single_positive_candidate()
    result, _, _ = run_case(
        candidates,
        weight_overrides={
            "plot_events_weight": RelevanceSize.LARGE,
            "plot_analysis_weight": RelevanceSize.MEDIUM,
            "viewer_experience_weight": RelevanceSize.SMALL,
        },
    )

    # Raw: anchor=1.6, large=3, medium=2, small=1, others=0 => total=7.6
    assert space_weight(result, VectorName.ANCHOR) == pytest.approx(1.6 / 7.6, abs=1e-12)
    assert space_weight(result, VectorName.PLOT_EVENTS) == pytest.approx(3.0 / 7.6, abs=1e-12)
    assert space_weight(result, VectorName.PLOT_ANALYSIS) == pytest.approx(2.0 / 7.6, abs=1e-12)
    assert space_weight(result, VectorName.VIEWER_EXPERIENCE) == pytest.approx(1.0 / 7.6, abs=1e-12)
    for space in [
        VectorName.WATCH_CONTEXT,
        VectorName.NARRATIVE_TECHNIQUES,
        VectorName.PRODUCTION,
        VectorName.RECEPTION,
    ]:
        assert space_weight(result, space) == pytest.approx(0.0, abs=1e-12)


def test_case_44_single_non_anchor_large_active_anchor_vs_space_weights() -> None:
    candidates = _single_positive_candidate()
    result, _, _ = run_case(
        candidates,
        weight_overrides={"viewer_experience_weight": RelevanceSize.LARGE},
    )

    assert space_weight(result, VectorName.ANCHOR) == pytest.approx(2.4 / 5.4, abs=1e-12)
    assert space_weight(result, VectorName.VIEWER_EXPERIENCE) == pytest.approx(3.0 / 5.4, abs=1e-12)


def test_case_45_single_non_anchor_small_active_anchor_vs_space_weights() -> None:
    candidates = _single_positive_candidate()
    result, _, _ = run_case(
        candidates,
        weight_overrides={"viewer_experience_weight": RelevanceSize.SMALL},
    )

    assert space_weight(result, VectorName.ANCHOR) == pytest.approx(0.8 / 1.8, abs=1e-12)
    assert space_weight(result, VectorName.VIEWER_EXPERIENCE) == pytest.approx(1.0 / 1.8, abs=1e-12)


@pytest.mark.parametrize(
    "weights",
    [
        {"plot_events_weight": RelevanceSize.SMALL},
        {"plot_events_weight": RelevanceSize.MEDIUM, "production_weight": RelevanceSize.LARGE},
        {
            "plot_events_weight": RelevanceSize.SMALL,
            "plot_analysis_weight": RelevanceSize.MEDIUM,
            "viewer_experience_weight": RelevanceSize.LARGE,
        },
    ],
)
def test_case_46_anchor_weight_strictly_less_than_mean_active_non_anchor_weights(
    weights: dict[str, RelevanceSize],
) -> None:
    candidates = _single_positive_candidate()
    result, _, _ = run_case(candidates, weight_overrides=weights)

    non_anchor_active_weights = [
        ctx.normalized_weight
        for ctx in result.space_contexts
        if ctx.name != VectorName.ANCHOR and ctx.is_active
    ]
    assert non_anchor_active_weights
    anchor_weight = space_weight(result, VectorName.ANCHOR)
    assert anchor_weight < (sum(non_anchor_active_weights) / len(non_anchor_active_weights))


def test_case_47_promoted_spaces_contribute_to_weights() -> None:
    candidates = _single_positive_candidate()
    result, _, _ = run_case(
        candidates,
        subquery_overrides={
            "plot_events_subquery": "events",
            "viewer_experience_subquery": "experience",
            "reception_subquery": "reception",
        },
    )

    for space in [VectorName.PLOT_EVENTS, VectorName.VIEWER_EXPERIENCE, VectorName.RECEPTION]:
        assert space_weight(result, space) > 0.0
    for space in [VectorName.PLOT_ANALYSIS, VectorName.WATCH_CONTEXT, VectorName.NARRATIVE_TECHNIQUES, VectorName.PRODUCTION]:
        assert space_weight(result, space) == pytest.approx(0.0, abs=1e-12)


def test_case_48_weight_ordering_large_medium_small_anchor() -> None:
    candidates = _single_positive_candidate()
    result, _, _ = run_case(
        candidates,
        weight_overrides={
            "plot_events_weight": RelevanceSize.SMALL,
            "plot_analysis_weight": RelevanceSize.MEDIUM,
            "viewer_experience_weight": RelevanceSize.LARGE,
        },
    )

    w_small = space_weight(result, VectorName.PLOT_EVENTS)
    w_medium = space_weight(result, VectorName.PLOT_ANALYSIS)
    w_large = space_weight(result, VectorName.VIEWER_EXPERIENCE)
    w_anchor = space_weight(result, VectorName.ANCHOR)

    # With raw weights [small=1, medium=2, large=3], anchor raw is
    # 0.8 * mean([1,2,3]) = 1.6, so ordering is large > medium > anchor > small.
    assert w_large > w_medium > w_anchor > w_small


# ---------------------------------------------------------------------------
# 7 — Final Score Composition (Stage 5 correctness)
# ---------------------------------------------------------------------------


def test_case_49_candidate_top_in_all_active_spaces_reaches_one() -> None:
    candidates = make_candidates([1, 2, 3])
    for movie_id, score in [(1, 0.95), (2, 0.5), (3, 0.2)]:
        candidates[movie_id].anchor_score_original = score

    set_space_scores(candidates, VectorName.VIEWER_EXPERIENCE, originals={1: 0.9, 2: 0.4, 3: 0.1})
    set_space_scores(candidates, VectorName.PRODUCTION, originals={1: 0.8, 2: 0.5, 3: 0.2})

    result, _, _ = run_case(
        candidates,
        weight_overrides={
            "viewer_experience_weight": RelevanceSize.LARGE,
            "production_weight": RelevanceSize.MEDIUM,
        },
    )

    assert result.final_scores[1] == pytest.approx(1.0, abs=1e-12)


def test_case_50_candidate_present_in_only_one_active_space_equals_that_weight() -> None:
    candidates = make_candidates([1, 2])
    candidates[1].anchor_score_original = 0.0
    candidates[2].anchor_score_original = 0.9
    set_space_scores(candidates, VectorName.PRODUCTION, originals={1: 0.9, 2: 0.1})

    result, _, _ = run_case(
        candidates,
        weight_overrides={"production_weight": RelevanceSize.LARGE},
    )

    production_weight = space_weight(result, VectorName.PRODUCTION)
    assert result.final_scores[1] == pytest.approx(production_weight, abs=1e-12)


def test_case_51_candidate_absent_from_all_active_spaces_scores_zero() -> None:
    candidates = {
        1: make_candidate(anchor_score_original=0.9, production_score_original=0.9),
        2: make_candidate(),  # absent everywhere
    }
    result, _, _ = run_case(
        candidates,
        weight_overrides={"production_weight": RelevanceSize.MEDIUM},
    )

    assert result.final_scores[2] == pytest.approx(0.0, abs=1e-12)


def test_case_52_complementary_coverage_switches_with_weight_swap() -> None:
    candidates = make_candidates([1, 2])
    candidates[1].anchor_score_original = 0.8
    candidates[2].anchor_score_original = 0.8

    # A strong in viewer_experience, B strong in production.
    set_space_scores(candidates, VectorName.VIEWER_EXPERIENCE, originals={1: 0.9, 2: 0.2})
    set_space_scores(candidates, VectorName.PRODUCTION, originals={1: 0.2, 2: 0.9})

    result_viewer_heavy, _, _ = run_case(
        candidates,
        weight_overrides={
            "viewer_experience_weight": RelevanceSize.LARGE,
            "production_weight": RelevanceSize.SMALL,
        },
    )
    result_production_heavy, _, _ = run_case(
        candidates,
        weight_overrides={
            "viewer_experience_weight": RelevanceSize.SMALL,
            "production_weight": RelevanceSize.LARGE,
        },
    )

    assert result_viewer_heavy.final_scores[1] > result_viewer_heavy.final_scores[2]
    assert result_production_heavy.final_scores[2] > result_production_heavy.final_scores[1]


def test_case_53_score_is_additive_across_spaces() -> None:
    candidates = make_candidates([1, 2, 3])
    # No anchor contribution in this scenario.
    candidates[1].anchor_score_original = 0.0
    candidates[2].anchor_score_original = 0.0
    candidates[3].anchor_score_original = 0.0

    set_space_scores(candidates, VectorName.PLOT_EVENTS, originals={1: 0.8, 2: 0.4, 3: 0.2})
    set_space_scores(candidates, VectorName.PRODUCTION, originals={1: 0.6, 2: 0.9, 3: 0.3})

    result, _, _ = run_case(
        candidates,
        weight_overrides={
            "plot_events_weight": RelevanceSize.SMALL,
            "production_weight": RelevanceSize.MEDIUM,
        },
    )

    w_plot = space_weight(result, VectorName.PLOT_EVENTS)
    w_prod = space_weight(result, VectorName.PRODUCTION)
    norm_plot = result.per_space_normalized[VectorName.PLOT_EVENTS][1]
    norm_prod = result.per_space_normalized[VectorName.PRODUCTION][1]
    expected = w_plot * norm_plot + w_prod * norm_prod
    assert result.final_scores[1] == pytest.approx(expected, abs=1e-12)


def test_case_54_missing_high_weight_space_is_heavily_penalized() -> None:
    candidates = make_candidates([1, 2])
    # A appears in high-weighted viewer_experience. B does not.
    candidates[1].anchor_score_original = 0.9
    candidates[2].anchor_score_original = 0.1
    set_space_scores(candidates, VectorName.VIEWER_EXPERIENCE, originals={1: 0.9, 2: 0.0})
    set_space_scores(candidates, VectorName.PRODUCTION, originals={1: 0.6, 2: 1.0})

    result, _, _ = run_case(
        candidates,
        weight_overrides={
            "viewer_experience_weight": RelevanceSize.LARGE,
            "production_weight": RelevanceSize.MEDIUM,
        },
    )

    assert result.final_scores[1] > result.final_scores[2]
    assert (result.final_scores[1] - result.final_scores[2]) > 0.3


# ---------------------------------------------------------------------------
# 8 — Ordering and Ranking Correctness
# ---------------------------------------------------------------------------


def test_case_55_uniformly_higher_candidate_ranks_first() -> None:
    candidates = make_candidates([1, 2, 3])
    for movie_id, anchor in [(1, 0.9), (2, 0.6), (3, 0.3)]:
        candidates[movie_id].anchor_score_original = anchor

    set_space_scores(candidates, VectorName.VIEWER_EXPERIENCE, originals={1: 0.9, 2: 0.6, 3: 0.3})
    set_space_scores(candidates, VectorName.PRODUCTION, originals={1: 0.8, 2: 0.5, 3: 0.2})

    result, _, _ = run_case(
        candidates,
        weight_overrides={
            "viewer_experience_weight": RelevanceSize.MEDIUM,
            "production_weight": RelevanceSize.MEDIUM,
        },
    )

    assert result.final_scores[1] > result.final_scores[2] > result.final_scores[3]


def test_case_56_highest_weight_space_can_dominate_ranking() -> None:
    candidates = make_candidates([1, 2])
    candidates[1].anchor_score_original = 0.8
    candidates[2].anchor_score_original = 0.8

    # A dominates high-weight viewer_experience, B dominates low-weight production.
    set_space_scores(candidates, VectorName.VIEWER_EXPERIENCE, originals={1: 0.9, 2: 0.3})
    set_space_scores(candidates, VectorName.PRODUCTION, originals={1: 0.3, 2: 0.9})

    result, _, _ = run_case(
        candidates,
        weight_overrides={
            "viewer_experience_weight": RelevanceSize.LARGE,
            "production_weight": RelevanceSize.SMALL,
        },
    )
    assert result.final_scores[1] > result.final_scores[2]


def test_case_57_ordering_is_stable_across_repeated_calls() -> None:
    rng = random.Random(57)
    candidates = make_candidates(list(range(20)))
    for movie_id in candidates:
        candidates[movie_id].anchor_score_original = rng.random()
        for space in NON_ANCHOR_SPACES:
            setattr(candidates[movie_id], original_score_attr(space), rng.random())
            setattr(candidates[movie_id], subquery_score_attr(space), rng.random())

    weight_overrides = {
        "plot_events_weight": RelevanceSize.MEDIUM,
        "plot_analysis_weight": RelevanceSize.SMALL,
        "viewer_experience_weight": RelevanceSize.LARGE,
        "watch_context_weight": RelevanceSize.MEDIUM,
        "narrative_techniques_weight": RelevanceSize.SMALL,
        "production_weight": RelevanceSize.MEDIUM,
        "reception_weight": RelevanceSize.SMALL,
    }
    subquery_overrides = {
        "plot_events_subquery": "events",
        "plot_analysis_subquery": "analysis",
        "viewer_experience_subquery": "experience",
        "watch_context_subquery": "context",
        "narrative_techniques_subquery": "techniques",
        "production_subquery": "production",
        "reception_subquery": "reception",
    }

    baseline, _, _ = run_case(
        candidates,
        weight_overrides=weight_overrides,
        subquery_overrides=subquery_overrides,
    )
    baseline_snapshot = freeze_result(baseline)

    for _ in range(99):
        observed, _, _ = run_case(
            candidates,
            weight_overrides=weight_overrides,
            subquery_overrides=subquery_overrides,
        )
        assert freeze_result(observed) == baseline_snapshot


def test_case_58_cozy_90s_scenario_strong_decent_weak_ordering() -> None:
    # 1=strong cozy match, 2=decent match, 3=weak match.
    candidates = make_candidates([1, 2, 3])

    # Shared active-space configuration from the design document.
    weight_overrides = {
        "plot_analysis_weight": RelevanceSize.SMALL,
        "viewer_experience_weight": RelevanceSize.LARGE,
        "watch_context_weight": RelevanceSize.LARGE,
        "production_weight": RelevanceSize.MEDIUM,
    }
    subquery_overrides = {
        "plot_analysis_subquery": "cozy comforting atmosphere nostalgic warmth",
        "viewer_experience_subquery": "cozy warm comforting relaxing not intense",
        "watch_context_subquery": "rainy day movie relaxation great soundtrack",
        "production_subquery": "1990s 90s",
        "reception_subquery": "praised soundtrack beloved",
    }

    # Strong
    candidates[1].anchor_score_original = 0.72
    candidates[1].plot_analysis_score_original = 0.41
    candidates[1].plot_analysis_score_subquery = 0.68
    candidates[1].viewer_experience_score_original = 0.65
    candidates[1].viewer_experience_score_subquery = 0.81
    candidates[1].watch_context_score_original = 0.58
    candidates[1].watch_context_score_subquery = 0.79
    candidates[1].production_score_original = 0.55
    candidates[1].production_score_subquery = 0.71
    candidates[1].reception_score_subquery = 0.44

    # Decent
    candidates[2].anchor_score_original = 0.60
    candidates[2].plot_analysis_score_original = 0.30
    candidates[2].plot_analysis_score_subquery = 0.49
    candidates[2].viewer_experience_score_original = 0.48
    candidates[2].viewer_experience_score_subquery = 0.60
    candidates[2].watch_context_score_original = 0.44
    candidates[2].watch_context_score_subquery = 0.57
    candidates[2].production_score_original = 0.52
    candidates[2].production_score_subquery = 0.63
    candidates[2].reception_score_subquery = 0.30

    # Weak
    candidates[3].anchor_score_original = 0.38
    candidates[3].plot_analysis_score_subquery = 0.22
    candidates[3].viewer_experience_score_original = 0.31
    candidates[3].production_score_original = 0.47
    candidates[3].production_score_subquery = 0.60

    result, _, _ = run_case(
        candidates,
        weight_overrides=weight_overrides,
        subquery_overrides=subquery_overrides,
    )

    assert result.final_scores[1] > result.final_scores[2] > result.final_scores[3]
    assert (result.final_scores[1] - result.final_scores[3]) > 0.4


# ---------------------------------------------------------------------------
# 9 — Promotion Rule Edge Cases
# ---------------------------------------------------------------------------


def test_case_59_promotion_ignores_stale_original_scores() -> None:
    candidates_stale = {
        1: make_candidate(anchor_score_original=0.8, reception_score_original=1.0, reception_score_subquery=0.7),
        2: make_candidate(anchor_score_original=0.8, reception_score_original=0.0, reception_score_subquery=0.5),
    }
    candidates_clean = {
        1: make_candidate(anchor_score_original=0.8, reception_score_original=0.0, reception_score_subquery=0.7),
        2: make_candidate(anchor_score_original=0.8, reception_score_original=0.0, reception_score_subquery=0.5),
    }

    args = dict(subquery_overrides={"reception_subquery": "audience sentiment"})
    stale_result, _, _ = run_case(candidates_stale, **args)
    clean_result, _, _ = run_case(candidates_clean, **args)

    assert stale_result.final_scores == clean_result.final_scores
    assert stale_result.per_space_normalized == clean_result.per_space_normalized
    assert context_map(stale_result)[VectorName.RECEPTION].did_run_original is False


def test_case_60_promotion_does_not_double_promote_small_plus_subquery() -> None:
    candidates = _anchor_positive_candidates()
    result, _, _ = run_case(
        candidates,
        weight_overrides={"reception_weight": RelevanceSize.SMALL},
        subquery_overrides={"reception_subquery": "subquery exists"},
    )

    ctx = context_map(result)[VectorName.RECEPTION]
    assert ctx.effective_relevance == RelevanceSize.SMALL


def test_case_61_multiple_promotions_affect_anchor_weight_mean() -> None:
    candidates = _anchor_positive_candidates()
    result, _, _ = run_case(
        candidates,
        subquery_overrides={
            "plot_events_subquery": "events",
            "plot_analysis_subquery": "analysis",
            "reception_subquery": "reception",
        },
    )

    # Three promoted SMALL spaces -> raw non-anchor active = [1,1,1].
    # Anchor raw = 0.8. Total raw = 3.8.
    assert space_weight(result, VectorName.ANCHOR) == pytest.approx(0.8 / 3.8, abs=1e-12)
    for space in [VectorName.PLOT_EVENTS, VectorName.PLOT_ANALYSIS, VectorName.RECEPTION]:
        ctx = context_map(result)[space]
        assert ctx.effective_relevance == RelevanceSize.SMALL
        assert ctx.is_active is True
        assert space_weight(result, space) == pytest.approx(1.0 / 3.8, abs=1e-12)


def test_case_62_promotion_changes_final_scores_vs_non_promoted_request() -> None:
    candidates = {
        1: make_candidate(anchor_score_original=0.0, reception_score_subquery=0.95),
        2: make_candidate(anchor_score_original=0.9, reception_score_subquery=0.0),
    }
    promoted, _, _ = run_case(
        candidates,
        subquery_overrides={"reception_subquery": "critically beloved soundtrack"},
    )
    not_promoted, _, _ = run_case(candidates)

    assert promoted.final_scores[1] > not_promoted.final_scores[1]


# ---------------------------------------------------------------------------
# 10 — Candidates Not Returned in Searches (0.0 handling)
# ---------------------------------------------------------------------------


def test_case_63_blended_zero_excluded_from_space_pool() -> None:
    candidates = make_candidates([1, 2])
    set_space_scores(candidates, VectorName.PLOT_ANALYSIS, originals={1: 0.8, 2: 0.0})
    result, _, _ = run_case(
        candidates,
        weight_overrides={"plot_analysis_weight": RelevanceSize.SMALL},
    )

    normalized = result.per_space_normalized[VectorName.PLOT_ANALYSIS]
    assert normalized == {1: 1.0}
    assert 2 not in normalized
    assert result.final_scores[2] == pytest.approx(0.0, abs=1e-12)


def test_case_64_zero_candidate_does_not_change_nonzero_pool_normalization() -> None:
    with_zero = make_candidates([1, 2, 3])
    set_space_scores(with_zero, VectorName.PLOT_ANALYSIS, originals={1: 0.80, 2: 0.79, 3: 0.0})
    result_with_zero, _, _ = run_case(
        with_zero,
        weight_overrides={"plot_analysis_weight": RelevanceSize.SMALL},
    )

    no_zero = make_candidates([1, 2])
    set_space_scores(no_zero, VectorName.PLOT_ANALYSIS, originals={1: 0.80, 2: 0.79})
    result_no_zero, _, _ = run_case(
        no_zero,
        weight_overrides={"plot_analysis_weight": RelevanceSize.SMALL},
    )

    norm_with_zero = result_with_zero.per_space_normalized[VectorName.PLOT_ANALYSIS]
    norm_no_zero = result_no_zero.per_space_normalized[VectorName.PLOT_ANALYSIS]
    assert norm_with_zero[1] == pytest.approx(norm_no_zero[1], abs=1e-12)
    assert norm_with_zero[2] == pytest.approx(norm_no_zero[2], abs=1e-12)
    assert 3 not in norm_with_zero


def test_case_65_candidate_only_one_of_many_active_spaces_scores_lower() -> None:
    candidates = make_candidates([1, 2])
    # Candidate 1 appears in all spaces. Candidate 2 appears only in production.
    for movie_id, anchor in [(1, 0.9), (2, 0.0)]:
        candidates[movie_id].anchor_score_original = anchor

    set_space_scores(candidates, VectorName.PLOT_EVENTS, originals={1: 0.8, 2: 0.0})
    set_space_scores(candidates, VectorName.VIEWER_EXPERIENCE, originals={1: 0.8, 2: 0.0})
    set_space_scores(candidates, VectorName.PRODUCTION, originals={1: 0.5, 2: 0.9})

    result, _, _ = run_case(
        candidates,
        weight_overrides={
            "plot_events_weight": RelevanceSize.MEDIUM,
            "viewer_experience_weight": RelevanceSize.LARGE,
            "production_weight": RelevanceSize.MEDIUM,
        },
    )
    assert result.final_scores[1] > result.final_scores[2]


def test_case_66_all_candidates_zero_in_active_space_yields_empty_space_dict() -> None:
    candidates = make_candidates([1, 2, 3])
    for movie_id in candidates:
        candidates[movie_id].anchor_score_original = 0.7 - 0.1 * movie_id
    # Active viewer_experience space but all zero.
    set_space_scores(candidates, VectorName.VIEWER_EXPERIENCE, originals={1: 0.0, 2: 0.0, 3: 0.0})
    # Another active space with signal.
    set_space_scores(candidates, VectorName.PRODUCTION, originals={1: 0.9, 2: 0.6, 3: 0.3})

    result, _, _ = run_case(
        candidates,
        weight_overrides={
            "viewer_experience_weight": RelevanceSize.LARGE,
            "production_weight": RelevanceSize.MEDIUM,
        },
    )

    assert VectorName.VIEWER_EXPERIENCE in result.per_space_normalized
    assert result.per_space_normalized[VectorName.VIEWER_EXPERIENCE] == {}
    assert result.final_scores[1] > result.final_scores[2] > result.final_scores[3]


# ---------------------------------------------------------------------------
# 11 — Numerical Precision and Stability
# ---------------------------------------------------------------------------


def test_case_67_large_candidate_pool_performance_and_stability() -> None:
    rng = random.Random(67)
    candidates = make_candidates(list(range(5000)))
    for movie_id in candidates:
        candidates[movie_id].anchor_score_original = rng.random()
        candidates[movie_id].viewer_experience_score_original = rng.random()
        candidates[movie_id].production_score_original = rng.random()

    start = time.perf_counter()
    result, _, _ = run_case(
        candidates,
        weight_overrides={
            "viewer_experience_weight": RelevanceSize.LARGE,
            "production_weight": RelevanceSize.MEDIUM,
        },
    )
    elapsed_s = time.perf_counter() - start

    assert elapsed_s < 1.0
    assert all(math.isfinite(score) for score in result.final_scores.values())


def test_case_68_exact_zero_score_excluded_from_normalization_pool() -> None:
    candidates = make_candidates([1, 2])
    set_space_scores(candidates, VectorName.PLOT_EVENTS, originals={1: 0.8, 2: 0.0})
    result, _, _ = run_case(
        candidates,
        weight_overrides={"plot_events_weight": RelevanceSize.SMALL},
    )

    normalized = result.per_space_normalized[VectorName.PLOT_EVENTS]
    assert 2 not in normalized


def test_case_69_extremely_small_positive_score_is_included() -> None:
    candidates = make_candidates([1, 2])
    set_space_scores(candidates, VectorName.PLOT_EVENTS, originals={1: 0.0002, 2: 0.0001})
    result, _, _ = run_case(
        candidates,
        weight_overrides={"plot_events_weight": RelevanceSize.SMALL},
    )

    normalized = result.per_space_normalized[VectorName.PLOT_EVENTS]
    assert 1 in normalized and 2 in normalized
    assert normalized[2] > 0.0


def test_case_70_barely_above_zero_blended_score_still_participates() -> None:
    candidates = make_candidates([1, 2])
    set_space_scores(candidates, VectorName.PLOT_EVENTS, originals={1: 0.001, 2: 0.0}, subqueries={1: 0.0, 2: 0.002})

    result, _, _ = run_case(
        candidates,
        weight_overrides={"plot_events_weight": RelevanceSize.MEDIUM},
        subquery_overrides={"plot_events_subquery": "subquery exists"},
    )

    normalized = result.per_space_normalized[VectorName.PLOT_EVENTS]
    assert 1 in normalized
    assert normalized[1] > 0.0


def test_case_71_all_spaces_active_all_large_full_pipeline_stress() -> None:
    rng = random.Random(71)
    candidates = make_candidates(list(range(1, 51)))
    for movie_id in candidates:
        candidates[movie_id].anchor_score_original = rng.random()
        for space in NON_ANCHOR_SPACES:
            setattr(candidates[movie_id], original_score_attr(space), rng.random())
            setattr(candidates[movie_id], subquery_score_attr(space), rng.random())

    result, _, _ = run_case(
        candidates,
        weight_overrides={weight_attr(space): RelevanceSize.LARGE for space in NON_ANCHOR_SPACES},
        subquery_overrides={subquery_text_attr(space): f"{space.value} subquery" for space in NON_ANCHOR_SPACES},
    )

    assert set(result.per_space_normalized.keys()) == set(ALL_SPACES)
    assert all(len(space_scores) == 50 for space_scores in result.per_space_normalized.values())


def test_case_72_tiny_final_score_difference_has_no_spurious_ordering_noise() -> None:
    candidates = make_candidates([1, 2])
    candidates[1].anchor_score_original = 0.8
    candidates[2].anchor_score_original = 0.8
    set_space_scores(candidates, VectorName.PLOT_EVENTS, originals={1: 0.9, 2: 0.6})
    set_space_scores(candidates, VectorName.PRODUCTION, originals={1: 0.6, 2: 0.9})

    result, _, _ = run_case(
        candidates,
        weight_overrides={
            "plot_events_weight": RelevanceSize.SMALL,
            "production_weight": RelevanceSize.SMALL,
        },
    )

    diff = abs(result.final_scores[1] - result.final_scores[2])
    assert diff < 1e-10


# ---------------------------------------------------------------------------
# 12 — Each Space Functions Independently
# ---------------------------------------------------------------------------


def test_case_73_spaces_normalize_independently_without_cross_contamination() -> None:
    candidates = make_candidates([1, 2, 3])
    for movie_id in candidates:
        candidates[movie_id].anchor_score_original = 0.5

    set_space_scores(candidates, VectorName.PLOT_EVENTS, originals={1: 0.9, 2: 0.6, 3: 0.3})
    set_space_scores(candidates, VectorName.PRODUCTION, originals={1: 0.3, 2: 0.6, 3: 0.9})

    result, _, _ = run_case(
        candidates,
        weight_overrides={
            "plot_events_weight": RelevanceSize.MEDIUM,
            "production_weight": RelevanceSize.MEDIUM,
        },
    )

    assert result.per_space_normalized[VectorName.PLOT_EVENTS][1] == pytest.approx(1.0, abs=1e-12)
    assert result.per_space_normalized[VectorName.PRODUCTION][3] == pytest.approx(1.0, abs=1e-12)


def test_case_74_adding_candidate_in_one_space_does_not_change_other_space_norm() -> None:
    base = make_candidates([1, 2])
    expanded = make_candidates([1, 2, 3])
    for movie_id in [1, 2]:
        base[movie_id].anchor_score_original = 0.6
        expanded[movie_id].anchor_score_original = 0.6

    # Same scores for 1/2 in both spaces.
    set_space_scores(base, VectorName.PLOT_EVENTS, originals={1: 0.8, 2: 0.6})
    set_space_scores(base, VectorName.PRODUCTION, originals={1: 0.7, 2: 0.5})

    set_space_scores(expanded, VectorName.PLOT_EVENTS, originals={1: 0.8, 2: 0.6, 3: 0.9})
    set_space_scores(expanded, VectorName.PRODUCTION, originals={1: 0.7, 2: 0.5, 3: 0.0})

    result_base, _, _ = run_case(
        base,
        weight_overrides={"plot_events_weight": RelevanceSize.MEDIUM, "production_weight": RelevanceSize.MEDIUM},
    )
    result_expanded, _, _ = run_case(
        expanded,
        weight_overrides={"plot_events_weight": RelevanceSize.MEDIUM, "production_weight": RelevanceSize.MEDIUM},
    )

    base_prod = result_base.per_space_normalized[VectorName.PRODUCTION]
    expanded_prod = result_expanded.per_space_normalized[VectorName.PRODUCTION]
    assert expanded_prod[1] == pytest.approx(base_prod[1], abs=1e-12)
    assert expanded_prod[2] == pytest.approx(base_prod[2], abs=1e-12)


# ---------------------------------------------------------------------------
# 13 — Stale Data and Defensive Behavior
# ---------------------------------------------------------------------------


def test_case_75_stale_original_in_non_searched_space_is_ignored() -> None:
    stale = {
        1: make_candidate(anchor_score_original=0.8, plot_events_score_original=0.99),
        2: make_candidate(anchor_score_original=0.4, plot_events_score_original=0.10),
    }
    clean = {
        1: make_candidate(anchor_score_original=0.8, plot_events_score_original=0.0),
        2: make_candidate(anchor_score_original=0.4, plot_events_score_original=0.0),
    }

    result_stale, _, _ = run_case(stale)
    result_clean, _, _ = run_case(clean)
    assert result_stale.final_scores == result_clean.final_scores


def test_case_76_stale_subquery_score_without_subquery_text_is_ignored() -> None:
    stale = {
        1: make_candidate(anchor_score_original=0.8, production_score_original=0.7, production_score_subquery=1.0),
        2: make_candidate(anchor_score_original=0.8, production_score_original=0.3, production_score_subquery=0.9),
    }
    clean = {
        1: make_candidate(anchor_score_original=0.8, production_score_original=0.7, production_score_subquery=0.0),
        2: make_candidate(anchor_score_original=0.8, production_score_original=0.3, production_score_subquery=0.0),
    }

    args = dict(weight_overrides={"production_weight": RelevanceSize.MEDIUM})
    result_stale, _, _ = run_case(stale, **args)
    result_clean, _, _ = run_case(clean, **args)
    assert result_stale.final_scores == result_clean.final_scores


def test_case_77_stale_original_in_promoted_small_space_is_ignored() -> None:
    stale = {
        1: make_candidate(anchor_score_original=0.8, reception_score_original=0.99, reception_score_subquery=0.8),
        2: make_candidate(anchor_score_original=0.8, reception_score_original=0.01, reception_score_subquery=0.4),
    }
    clean = {
        1: make_candidate(anchor_score_original=0.8, reception_score_original=0.0, reception_score_subquery=0.8),
        2: make_candidate(anchor_score_original=0.8, reception_score_original=0.0, reception_score_subquery=0.4),
    }

    args = dict(subquery_overrides={"reception_subquery": "audience reaction"})
    result_stale, _, _ = run_case(stale, **args)
    result_clean, _, _ = run_case(clean, **args)
    assert result_stale.final_scores == result_clean.final_scores


# ---------------------------------------------------------------------------
# 14 — Per-Space Isolation Tests (one space at a time)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("space", NON_ANCHOR_SPACES)
def test_case_78_each_non_anchor_space_in_isolation(space: VectorName) -> None:
    candidates = make_candidates([1, 2, 3])
    for movie_id in candidates:
        candidates[movie_id].anchor_score_original = 0.7

    set_space_scores(
        candidates,
        space,
        originals={1: 0.9, 2: 0.6, 3: 0.3},
        subqueries={1: 0.9, 2: 0.6, 3: 0.3},
    )

    result, _, _ = run_case(
        candidates,
        weight_overrides={weight_attr(space): RelevanceSize.LARGE},
        subquery_overrides={subquery_text_attr(space): "space-specific query"},
    )

    assert result.final_scores[1] > result.final_scores[2] > result.final_scores[3]
    assert set(result.per_space_normalized.keys()) == {VectorName.ANCHOR, space}


# ---------------------------------------------------------------------------
# 15 — Regression Scenarios from the Design Document
# ---------------------------------------------------------------------------


def test_case_79_plan_example_youve_got_mail_vs_shawshank() -> None:
    # Query setup from the design plan's cozy/rainy-night scenario.
    weight_overrides = {
        "plot_analysis_weight": RelevanceSize.SMALL,
        "viewer_experience_weight": RelevanceSize.LARGE,
        "watch_context_weight": RelevanceSize.LARGE,
        "production_weight": RelevanceSize.MEDIUM,
    }
    subquery_overrides = {
        "plot_analysis_subquery": "cozy comforting atmosphere nostalgic warmth",
        "viewer_experience_subquery": "cozy warm comforting relaxing not intense",
        "watch_context_subquery": "rainy day movie relaxation great soundtrack",
        "production_subquery": "1990s 90s",
        "reception_subquery": "praised for soundtrack beloved",
    }

    # 1 = You've Got Mail, 2 = Shawshank.
    candidates = {
        1: make_candidate(
            anchor_score_original=0.72,
            plot_analysis_score_original=0.41,
            plot_analysis_score_subquery=0.68,
            viewer_experience_score_original=0.65,
            viewer_experience_score_subquery=0.81,
            watch_context_score_original=0.58,
            watch_context_score_subquery=0.79,
            production_score_original=0.55,
            production_score_subquery=0.71,
            reception_score_subquery=0.44,
        ),
        2: make_candidate(
            anchor_score_original=0.38,
            plot_analysis_score_original=0.00,
            plot_analysis_score_subquery=0.22,
            viewer_experience_score_original=0.31,
            viewer_experience_score_subquery=0.00,
            watch_context_score_original=0.00,
            watch_context_score_subquery=0.00,
            production_score_original=0.47,
            production_score_subquery=0.60,
            reception_score_subquery=0.00,
        ),
    }
    result, _, _ = run_case(
        candidates,
        weight_overrides=weight_overrides,
        subquery_overrides=subquery_overrides,
    )

    assert result.final_scores[1] > result.final_scores[2]
    assert result.final_scores[1] > (result.final_scores[2] + 0.4)

    assert space_weight(result, VectorName.ANCHOR) == pytest.approx(1.6 / 11.6, abs=1e-12)
    assert space_weight(result, VectorName.VIEWER_EXPERIENCE) == pytest.approx(3.0 / 11.6, abs=1e-12)
    assert space_weight(result, VectorName.WATCH_CONTEXT) == pytest.approx(3.0 / 11.6, abs=1e-12)
    assert space_weight(result, VectorName.PRODUCTION) == pytest.approx(2.0 / 11.6, abs=1e-12)
    assert space_weight(result, VectorName.PLOT_ANALYSIS) == pytest.approx(1.0 / 11.6, abs=1e-12)
    assert space_weight(result, VectorName.RECEPTION) == pytest.approx(1.0 / 11.6, abs=1e-12)


def test_case_80_appendix_d_scenario2_candidate_only_anchor() -> None:
    candidates = {
        1: make_candidate(anchor_score_original=0.9, viewer_experience_score_original=0.0),
        2: make_candidate(anchor_score_original=0.3, viewer_experience_score_original=0.8),
    }
    result, _, _ = run_case(
        candidates,
        weight_overrides={"viewer_experience_weight": RelevanceSize.LARGE},
    )

    assert result.final_scores[1] == pytest.approx(space_weight(result, VectorName.ANCHOR), abs=1e-12)


def test_case_81_appendix_d_scenario3_subquery_not_original_penalty() -> None:
    candidates = {
        1: make_candidate(anchor_score_original=0.8, viewer_experience_score_original=0.0, viewer_experience_score_subquery=0.8),
        2: make_candidate(anchor_score_original=0.8, viewer_experience_score_original=0.8, viewer_experience_score_subquery=0.8),
    }
    result, _, _ = run_case(
        candidates,
        weight_overrides={"viewer_experience_weight": RelevanceSize.MEDIUM},
        subquery_overrides={"viewer_experience_subquery": "cozy warmth"},
    )

    assert result.final_scores[2] > result.final_scores[1]


def test_case_82_appendix_d_scenario4_original_not_subquery_heavy_penalty() -> None:
    candidates = {
        1: make_candidate(anchor_score_original=0.8, viewer_experience_score_original=0.8, viewer_experience_score_subquery=0.0),
        2: make_candidate(anchor_score_original=0.8, viewer_experience_score_original=0.8, viewer_experience_score_subquery=0.8),
    }
    result, _, _ = run_case(
        candidates,
        weight_overrides={"viewer_experience_weight": RelevanceSize.MEDIUM},
        subquery_overrides={"viewer_experience_subquery": "cozy warmth"},
    )

    assert result.final_scores[2] > result.final_scores[1]
    assert (result.final_scores[2] - result.final_scores[1]) > 0.2


# ---------------------------------------------------------------------------
# 16 — Output Structure Completeness
# ---------------------------------------------------------------------------


def test_case_83_per_space_keys_exactly_match_active_spaces() -> None:
    candidates = {
        1: make_candidate(anchor_score_original=0.9, plot_events_score_original=0.8, reception_score_subquery=0.7),
        2: make_candidate(anchor_score_original=0.6, plot_events_score_original=0.2, reception_score_subquery=0.4),
    }
    result, _, _ = run_case(
        candidates,
        weight_overrides={"plot_events_weight": RelevanceSize.SMALL},
        subquery_overrides={"reception_subquery": "critical response"},
    )

    active_spaces = {ctx.name for ctx in result.space_contexts if ctx.is_active}
    assert set(result.per_space_normalized.keys()) == active_spaces


def test_case_84_active_space_with_no_positive_scores_present_as_empty_dict() -> None:
    candidates = {
        1: make_candidate(anchor_score_original=0.9, viewer_experience_score_original=0.0),
        2: make_candidate(anchor_score_original=0.6, viewer_experience_score_original=0.0),
    }
    result, _, _ = run_case(
        candidates,
        weight_overrides={"viewer_experience_weight": RelevanceSize.LARGE},
    )

    assert VectorName.VIEWER_EXPERIENCE in result.per_space_normalized
    assert result.per_space_normalized[VectorName.VIEWER_EXPERIENCE] == {}


def test_case_85_space_contexts_have_required_fields_populated() -> None:
    candidates = {
        1: make_candidate(anchor_score_original=0.9, plot_events_score_original=0.7, plot_events_score_subquery=0.8),
        2: make_candidate(anchor_score_original=0.4, plot_events_score_original=0.3, plot_events_score_subquery=0.2),
    }
    result, _, _ = run_case(
        candidates,
        weight_overrides={"plot_events_weight": RelevanceSize.MEDIUM},
        subquery_overrides={"plot_events_subquery": "events"},
    )

    for ctx in result.space_contexts:
        assert isinstance(ctx.did_run_original, bool)
        assert isinstance(ctx.did_run_subquery, bool)
        assert ctx.normalized_weight is not None
        if ctx.name == VectorName.ANCHOR:
            assert ctx.effective_relevance is None
        else:
            assert ctx.effective_relevance is not None


def test_case_86_final_score_keys_match_non_sequential_candidate_ids() -> None:
    movie_ids = {7, 42, 999, 100000}
    candidates = {
        7: make_candidate(anchor_score_original=0.9),
        42: make_candidate(anchor_score_original=0.6),
        999: make_candidate(anchor_score_original=0.3),
        100000: make_candidate(anchor_score_original=0.1),
    }
    result, _, _ = run_case(candidates)
    assert set(result.final_scores.keys()) == movie_ids
