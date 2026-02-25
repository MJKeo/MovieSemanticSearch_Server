"""Unit tests for Stage 4 weight normalization in db.vector_scoring."""

from __future__ import annotations

import math

import pytest

from implementation.classes.enums import RelevanceSize, VectorName

from db.vector_scoring import SpaceExecutionContext, compute_normalized_weights


def make_ctx(
    name: VectorName,
    *,
    did_run_original: bool,
    did_run_subquery: bool,
    effective_relevance: RelevanceSize | None,
) -> SpaceExecutionContext:
    return SpaceExecutionContext(
        name=name,
        did_run_original=did_run_original,
        did_run_subquery=did_run_subquery,
        effective_relevance=effective_relevance,
    )


def build_inactive_non_anchor_contexts() -> list[SpaceExecutionContext]:
    return [
        make_ctx(VectorName.ANCHOR, did_run_original=True, did_run_subquery=False, effective_relevance=None),
        make_ctx(VectorName.PLOT_EVENTS, did_run_original=False, did_run_subquery=False, effective_relevance=RelevanceSize.NOT_RELEVANT),
        make_ctx(VectorName.PLOT_ANALYSIS, did_run_original=False, did_run_subquery=False, effective_relevance=RelevanceSize.NOT_RELEVANT),
        make_ctx(VectorName.VIEWER_EXPERIENCE, did_run_original=False, did_run_subquery=False, effective_relevance=RelevanceSize.NOT_RELEVANT),
        make_ctx(VectorName.WATCH_CONTEXT, did_run_original=False, did_run_subquery=False, effective_relevance=RelevanceSize.NOT_RELEVANT),
        make_ctx(VectorName.NARRATIVE_TECHNIQUES, did_run_original=False, did_run_subquery=False, effective_relevance=RelevanceSize.NOT_RELEVANT),
        make_ctx(VectorName.PRODUCTION, did_run_original=False, did_run_subquery=False, effective_relevance=RelevanceSize.NOT_RELEVANT),
        make_ctx(VectorName.RECEPTION, did_run_original=False, did_run_subquery=False, effective_relevance=RelevanceSize.NOT_RELEVANT),
    ]


def test_weights_canonical_guide_example_exactness() -> None:
    contexts = [
        make_ctx(VectorName.ANCHOR, did_run_original=True, did_run_subquery=False, effective_relevance=None),
        make_ctx(VectorName.PLOT_EVENTS, did_run_original=False, did_run_subquery=False, effective_relevance=RelevanceSize.NOT_RELEVANT),
        make_ctx(VectorName.PLOT_ANALYSIS, did_run_original=True, did_run_subquery=True, effective_relevance=RelevanceSize.SMALL),
        make_ctx(VectorName.VIEWER_EXPERIENCE, did_run_original=True, did_run_subquery=True, effective_relevance=RelevanceSize.LARGE),
        make_ctx(VectorName.WATCH_CONTEXT, did_run_original=True, did_run_subquery=True, effective_relevance=RelevanceSize.LARGE),
        make_ctx(VectorName.NARRATIVE_TECHNIQUES, did_run_original=False, did_run_subquery=False, effective_relevance=RelevanceSize.NOT_RELEVANT),
        make_ctx(VectorName.PRODUCTION, did_run_original=True, did_run_subquery=True, effective_relevance=RelevanceSize.MEDIUM),
        make_ctx(VectorName.RECEPTION, did_run_original=False, did_run_subquery=True, effective_relevance=RelevanceSize.SMALL),
    ]

    compute_normalized_weights(contexts)
    by_name = {ctx.name: ctx for ctx in contexts}
    total_raw = 11.6

    assert by_name[VectorName.ANCHOR].normalized_weight == pytest.approx(1.6 / total_raw, abs=1e-12)
    assert by_name[VectorName.PLOT_EVENTS].normalized_weight == pytest.approx(0.0, abs=1e-12)
    assert by_name[VectorName.PLOT_ANALYSIS].normalized_weight == pytest.approx(1.0 / total_raw, abs=1e-12)
    assert by_name[VectorName.VIEWER_EXPERIENCE].normalized_weight == pytest.approx(3.0 / total_raw, abs=1e-12)
    assert by_name[VectorName.WATCH_CONTEXT].normalized_weight == pytest.approx(3.0 / total_raw, abs=1e-12)
    assert by_name[VectorName.NARRATIVE_TECHNIQUES].normalized_weight == pytest.approx(0.0, abs=1e-12)
    assert by_name[VectorName.PRODUCTION].normalized_weight == pytest.approx(2.0 / total_raw, abs=1e-12)
    assert by_name[VectorName.RECEPTION].normalized_weight == pytest.approx(1.0 / total_raw, abs=1e-12)
    assert sum(ctx.normalized_weight for ctx in contexts) == pytest.approx(1.0, abs=1e-12)


def test_weights_all_non_anchor_inactive_anchor_gets_everything() -> None:
    contexts = build_inactive_non_anchor_contexts()
    compute_normalized_weights(contexts)

    for ctx in contexts:
        if ctx.name == VectorName.ANCHOR:
            assert ctx.normalized_weight == pytest.approx(1.0, abs=1e-12)
        else:
            assert ctx.normalized_weight == pytest.approx(0.0, abs=1e-12)


def test_weights_single_non_anchor_active_large() -> None:
    contexts = build_inactive_non_anchor_contexts()
    contexts[3] = make_ctx(
        VectorName.VIEWER_EXPERIENCE,
        did_run_original=True,
        did_run_subquery=True,
        effective_relevance=RelevanceSize.LARGE,
    )

    compute_normalized_weights(contexts)
    by_name = {ctx.name: ctx for ctx in contexts}

    assert by_name[VectorName.ANCHOR].normalized_weight == pytest.approx(2.4 / 5.4, abs=1e-12)
    assert by_name[VectorName.VIEWER_EXPERIENCE].normalized_weight == pytest.approx(3.0 / 5.4, abs=1e-12)


def test_weights_all_non_anchor_active_mixed_relevance_mapping() -> None:
    contexts = [
        make_ctx(VectorName.ANCHOR, did_run_original=True, did_run_subquery=False, effective_relevance=None),
        make_ctx(VectorName.PLOT_EVENTS, did_run_original=True, did_run_subquery=True, effective_relevance=RelevanceSize.SMALL),
        make_ctx(VectorName.PLOT_ANALYSIS, did_run_original=True, did_run_subquery=True, effective_relevance=RelevanceSize.MEDIUM),
        make_ctx(VectorName.VIEWER_EXPERIENCE, did_run_original=True, did_run_subquery=True, effective_relevance=RelevanceSize.LARGE),
        make_ctx(VectorName.WATCH_CONTEXT, did_run_original=True, did_run_subquery=False, effective_relevance=RelevanceSize.SMALL),
        make_ctx(VectorName.NARRATIVE_TECHNIQUES, did_run_original=True, did_run_subquery=False, effective_relevance=RelevanceSize.MEDIUM),
        make_ctx(VectorName.PRODUCTION, did_run_original=True, did_run_subquery=True, effective_relevance=RelevanceSize.LARGE),
        make_ctx(VectorName.RECEPTION, did_run_original=False, did_run_subquery=True, effective_relevance=RelevanceSize.SMALL),
    ]

    compute_normalized_weights(contexts)
    by_name = {ctx.name: ctx for ctx in contexts}

    raw_non_anchor = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]
    anchor_raw = (sum(raw_non_anchor) / len(raw_non_anchor)) * 0.8
    total = anchor_raw + sum(raw_non_anchor)

    assert by_name[VectorName.ANCHOR].normalized_weight == pytest.approx(anchor_raw / total, abs=1e-12)
    assert by_name[VectorName.PLOT_EVENTS].normalized_weight == pytest.approx(1.0 / total, abs=1e-12)
    assert by_name[VectorName.PLOT_ANALYSIS].normalized_weight == pytest.approx(2.0 / total, abs=1e-12)
    assert by_name[VectorName.VIEWER_EXPERIENCE].normalized_weight == pytest.approx(3.0 / total, abs=1e-12)
    assert by_name[VectorName.RECEPTION].normalized_weight == pytest.approx(1.0 / total, abs=1e-12)


def test_weights_promoted_small_space_gets_raw_one() -> None:
    contexts = build_inactive_non_anchor_contexts()
    contexts[7] = make_ctx(
        VectorName.RECEPTION,
        did_run_original=False,
        did_run_subquery=True,
        effective_relevance=RelevanceSize.SMALL,
    )

    compute_normalized_weights(contexts)
    by_name = {ctx.name: ctx for ctx in contexts}
    total = 0.8 + 1.0
    assert by_name[VectorName.ANCHOR].normalized_weight == pytest.approx(0.8 / total, abs=1e-12)
    assert by_name[VectorName.RECEPTION].normalized_weight == pytest.approx(1.0 / total, abs=1e-12)


def test_weights_inactive_non_anchor_spaces_are_zero() -> None:
    contexts = build_inactive_non_anchor_contexts()
    contexts[2] = make_ctx(
        VectorName.PLOT_ANALYSIS,
        did_run_original=True,
        did_run_subquery=True,
        effective_relevance=RelevanceSize.MEDIUM,
    )

    compute_normalized_weights(contexts)
    for ctx in contexts:
        if not ctx.is_active and ctx.name != VectorName.ANCHOR:
            assert ctx.normalized_weight == pytest.approx(0.0, abs=1e-12)


def test_weights_non_negative_and_sum_to_one() -> None:
    contexts = [
        make_ctx(VectorName.ANCHOR, did_run_original=True, did_run_subquery=False, effective_relevance=None),
        make_ctx(VectorName.PLOT_EVENTS, did_run_original=True, did_run_subquery=True, effective_relevance=RelevanceSize.SMALL),
        make_ctx(VectorName.PLOT_ANALYSIS, did_run_original=False, did_run_subquery=True, effective_relevance=RelevanceSize.SMALL),
    ]
    compute_normalized_weights(contexts)

    assert all(ctx.normalized_weight >= 0.0 for ctx in contexts)
    assert sum(ctx.normalized_weight for ctx in contexts) == pytest.approx(1.0, abs=1e-12)


def test_weights_mutates_contexts_in_place_without_replacing_objects() -> None:
    contexts = build_inactive_non_anchor_contexts()
    list_id_before = id(contexts)
    object_ids_before = [id(ctx) for ctx in contexts]

    compute_normalized_weights(contexts)

    assert id(contexts) == list_id_before
    assert [id(ctx) for ctx in contexts] == object_ids_before


def test_weights_custom_anchor_mean_fraction_changes_anchor_weight() -> None:
    contexts = build_inactive_non_anchor_contexts()
    contexts[1] = make_ctx(
        VectorName.PLOT_EVENTS,
        did_run_original=True,
        did_run_subquery=True,
        effective_relevance=RelevanceSize.SMALL,
    )
    contexts[3] = make_ctx(
        VectorName.VIEWER_EXPERIENCE,
        did_run_original=True,
        did_run_subquery=True,
        effective_relevance=RelevanceSize.LARGE,
    )

    compute_normalized_weights(contexts, anchor_mean_fraction=0.5)
    by_name = {ctx.name: ctx for ctx in contexts}
    anchor_raw = ((1.0 + 3.0) / 2.0) * 0.5
    total = anchor_raw + 1.0 + 3.0

    assert by_name[VectorName.ANCHOR].normalized_weight == pytest.approx(anchor_raw / total, abs=1e-12)
    assert by_name[VectorName.PLOT_EVENTS].normalized_weight == pytest.approx(1.0 / total, abs=1e-12)
    assert by_name[VectorName.VIEWER_EXPERIENCE].normalized_weight == pytest.approx(3.0 / total, abs=1e-12)


def test_weights_empty_contexts_raise_value_error() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        compute_normalized_weights([])


def test_weights_no_anchor_raises_value_error() -> None:
    contexts = [make_ctx(VectorName.PLOT_EVENTS, did_run_original=True, did_run_subquery=True, effective_relevance=RelevanceSize.SMALL)]
    with pytest.raises(ValueError, match="exactly one anchor"):
        compute_normalized_weights(contexts)


def test_weights_multiple_anchors_raise_value_error() -> None:
    contexts = [
        make_ctx(VectorName.ANCHOR, did_run_original=True, did_run_subquery=False, effective_relevance=None),
        make_ctx(VectorName.ANCHOR, did_run_original=True, did_run_subquery=False, effective_relevance=None),
    ]
    with pytest.raises(ValueError, match="exactly one anchor"):
        compute_normalized_weights(contexts)


def test_weights_inactive_anchor_raises_value_error() -> None:
    contexts = build_inactive_non_anchor_contexts()
    contexts[0] = make_ctx(
        VectorName.ANCHOR,
        did_run_original=False,
        did_run_subquery=False,
        effective_relevance=None,
    )
    with pytest.raises(ValueError, match="anchor context must be active"):
        compute_normalized_weights(contexts)


def test_weights_active_non_anchor_with_none_effective_relevance_raises_value_error() -> None:
    contexts = build_inactive_non_anchor_contexts()
    contexts[4] = make_ctx(
        VectorName.WATCH_CONTEXT,
        did_run_original=False,
        did_run_subquery=True,
        effective_relevance=None,
    )
    with pytest.raises(ValueError, match="must define effective_relevance"):
        compute_normalized_weights(contexts)


def test_weights_active_non_anchor_with_non_enum_relevance_raises_value_error() -> None:
    contexts = build_inactive_non_anchor_contexts()
    contexts[6] = make_ctx(
        VectorName.PRODUCTION,
        did_run_original=True,
        did_run_subquery=True,
        effective_relevance=RelevanceSize.SMALL,
    )
    contexts[6].effective_relevance = "small"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="invalid effective_relevance type"):
        compute_normalized_weights(contexts)


def test_weights_active_non_anchor_not_relevant_raises_value_error() -> None:
    contexts = build_inactive_non_anchor_contexts()
    contexts[2] = make_ctx(
        VectorName.PLOT_ANALYSIS,
        did_run_original=True,
        did_run_subquery=False,
        effective_relevance=RelevanceSize.NOT_RELEVANT,
    )
    with pytest.raises(ValueError, match="cannot have effective_relevance=NOT_RELEVANT"):
        compute_normalized_weights(contexts)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_weights_invalid_anchor_mean_fraction_raises_value_error(bad: float) -> None:
    contexts = build_inactive_non_anchor_contexts()
    with pytest.raises(ValueError, match="anchor_mean_fraction must be finite and > 0"):
        compute_normalized_weights(contexts, anchor_mean_fraction=bad)


def test_weights_non_numeric_anchor_mean_fraction_raises_type_error() -> None:
    contexts = build_inactive_non_anchor_contexts()
    with pytest.raises(TypeError, match="anchor_mean_fraction must be a real number"):
        compute_normalized_weights(contexts, anchor_mean_fraction="0.8")  # type: ignore[arg-type]


def test_weights_non_list_contexts_raises_type_error() -> None:
    with pytest.raises(TypeError, match="contexts must be a list"):
        compute_normalized_weights(tuple(build_inactive_non_anchor_contexts()))  # type: ignore[arg-type]


def test_weights_non_context_item_raises_type_error() -> None:
    contexts = build_inactive_non_anchor_contexts()
    contexts[3] = "not a context"  # type: ignore[assignment]
    with pytest.raises(TypeError, match="must be SpaceExecutionContext"):
        compute_normalized_weights(contexts)
