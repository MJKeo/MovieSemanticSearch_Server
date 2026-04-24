"""Unit tests for the per-category handler output schemas built by
search_v2/stage_3/category_handlers/schema_factories.py.

These tests are pinned to the documented contract in
search_improvement_planning/category_handler_planning.md
(specifically §"Handler LLM output schema per bucket" and
§"Building output schemas dynamically"). They are deliberately
written against the doc, not the current implementation — if a test
fails, the schema factory has drifted from the spec.

Documented invariants exercised:

  • Schemas are built from (bucket, endpoint set) and cached in
    OUTPUT_SCHEMAS at module import.
  • Categories whose endpoint set resolves to no LLM-wrapper endpoint
    (currently TRENDING-only) get NO schema.
  • SINGLE  → {requirement_aspects, should_run_endpoint, endpoint_parameters}
              with per-aspect {aspect_description, relation_to_endpoint, coverage_gaps}.
  • MUTEX   → {requirement_aspects, endpoint_to_run, endpoint_parameters}
              with per-aspect {aspect_description, endpoint_coverage,
              best_endpoint, best_endpoint_gaps}; endpoint_to_run is a
              Literal over candidate endpoint values plus "None"; the
              endpoint_parameters union covers every candidate wrapper.
  • TIERED  → MUTEX shape plus a `performance_vs_bias_analysis: str`
              field placed BEFORE endpoint_to_run (the planning doc
              calls out that the bias reasoning must precede the
              commitment).
  • COMBO   → {requirement_aspects, overall_endpoint_fits,
              per_endpoint_breakdown}; per_endpoint_breakdown is a
              sub-MODEL (not a list) with one named field per
              candidate endpoint, each carrying
              {should_run_endpoint, endpoint_parameters}.
  • coverage_gaps / best_endpoint_gaps / endpoint_parameters are
    nullable with default None (so the LLM can abstain cleanly).
"""

from __future__ import annotations

from typing import Literal, Union, get_args, get_origin
import types

import pytest
from pydantic import BaseModel

from schemas.endpoint_parameters import EndpointParameters
from schemas.enums import CategoryName, EndpointRoute, HandlerBucket
from search_v2.stage_3.category_handlers.endpoint_registry import ROUTE_TO_WRAPPER
from search_v2.stage_3.category_handlers.schema_factories import (
    OUTPUT_SCHEMAS,
    get_output_schema,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _llm_wrapper_endpoints(category: CategoryName) -> tuple[EndpointRoute, ...]:
    """Endpoints in the category that map to an LLM wrapper class.

    Mirrors the doc's "TRENDING resolves to no wrapper" carve-out
    without re-implementing the factory's filter — TRENDING is the
    only entry in ROUTE_TO_WRAPPER set to None today.
    """
    return tuple(r for r in category.endpoints if ROUTE_TO_WRAPPER[r] is not None)


def _is_optional(annotation: object) -> bool:
    """True if annotation is Optional[X] (i.e. Union with NoneType)."""
    if get_origin(annotation) is not Union and not isinstance(
        annotation, types.UnionType
    ):
        return False
    return type(None) in get_args(annotation)


def _strip_optional(annotation: object) -> object:
    """Return X for Optional[X], else the original annotation."""
    args = tuple(a for a in get_args(annotation) if a is not type(None))
    if len(args) == 1:
        return args[0]
    return Union[args]  # type: ignore[valid-type]


def _list_item_type(annotation: object) -> type:
    """Return the X in list[X]."""
    assert get_origin(annotation) is list, f"expected list[...], got {annotation!r}"
    (item,) = get_args(annotation)
    return item


def _categories_in_bucket(bucket: HandlerBucket) -> list[CategoryName]:
    """All categories of a given bucket that produce an LLM schema."""
    return [c for c in CategoryName if c.bucket is bucket and c in OUTPUT_SCHEMAS]


# ── Module-level invariants ──────────────────────────────────────────


class TestOutputSchemasModule:
    """Top-level wiring: which categories get a schema, which don't."""

    def test_output_schemas_populated_at_import(self) -> None:
        # The planning doc requires eager construction at module load
        # so misconfiguration surfaces at startup, not on first request.
        assert isinstance(OUTPUT_SCHEMAS, dict)
        assert len(OUTPUT_SCHEMAS) > 0

    def test_every_category_with_llm_endpoint_has_schema(self) -> None:
        # The doc says: only categories whose endpoint set resolves to
        # NO LLM wrapper are absent. Conversely, every other category
        # MUST have a schema.
        for category in CategoryName:
            has_llm_endpoint = bool(_llm_wrapper_endpoints(category))
            in_schemas = category in OUTPUT_SCHEMAS
            assert has_llm_endpoint == in_schemas, (
                f"{category.name}: has_llm_endpoint={has_llm_endpoint} "
                f"but in_schemas={in_schemas}"
            )

    def test_trending_only_category_has_no_schema(self) -> None:
        # TRENDING is the canonical "no LLM wrapper" category and the
        # planning doc explicitly carves it out.
        assert CategoryName.TRENDING not in OUTPUT_SCHEMAS

    def test_get_output_schema_raises_for_trending(self) -> None:
        # The accessor's documented behavior for no-schema categories
        # is to raise KeyError so callers special-case upstream.
        with pytest.raises(KeyError):
            get_output_schema(CategoryName.TRENDING)

    def test_get_output_schema_returns_cached_class(self) -> None:
        # Doc: "Build once at process start and cache by category name;
        # don't rebuild per request."
        a = get_output_schema(CategoryName.CREDIT_TITLE)
        b = get_output_schema(CategoryName.CREDIT_TITLE)
        assert a is b

    def test_every_schema_subclasses_basemodel(self) -> None:
        for category, cls in OUTPUT_SCHEMAS.items():
            assert issubclass(cls, BaseModel), (
                f"{category.name} schema is not a BaseModel subclass"
            )


# ── SINGLE bucket ────────────────────────────────────────────────────


SINGLE_TOP_FIELDS = {"requirement_aspects", "should_run_endpoint", "endpoint_parameters"}
SINGLE_ASPECT_FIELDS = {"aspect_description", "relation_to_endpoint", "coverage_gaps"}


@pytest.mark.parametrize(
    "category",
    _categories_in_bucket(HandlerBucket.SINGLE),
    ids=lambda c: c.name,
)
class TestSingleBucketSchema:
    """Per-doc shape for SINGLE-endpoint handler outputs."""

    def test_top_level_fields_exact(self, category: CategoryName) -> None:
        cls = get_output_schema(category)
        assert set(cls.model_fields) == SINGLE_TOP_FIELDS

    def test_should_run_endpoint_is_required_bool(
        self, category: CategoryName
    ) -> None:
        cls = get_output_schema(category)
        field = cls.model_fields["should_run_endpoint"]
        assert field.annotation is bool
        assert field.is_required()

    def test_endpoint_parameters_optional_correct_wrapper(
        self, category: CategoryName
    ) -> None:
        # Doc: "endpoint-specific parameter shape ... Left null when
        # should_run_endpoint is false." The wrapper must be the
        # category's single LLM-endpoint wrapper.
        cls = get_output_schema(category)
        field = cls.model_fields["endpoint_parameters"]
        assert _is_optional(field.annotation), (
            "endpoint_parameters must be nullable so the LLM can abstain"
        )
        assert field.default is None

        (endpoint,) = _llm_wrapper_endpoints(category)
        expected_wrapper = ROUTE_TO_WRAPPER[endpoint]
        assert _strip_optional(field.annotation) is expected_wrapper

    def test_requirement_aspects_is_list_of_aspect_model(
        self, category: CategoryName
    ) -> None:
        cls = get_output_schema(category)
        field = cls.model_fields["requirement_aspects"]
        assert get_origin(field.annotation) is list
        aspect_cls = _list_item_type(field.annotation)
        assert issubclass(aspect_cls, BaseModel)
        assert set(aspect_cls.model_fields) == SINGLE_ASPECT_FIELDS

    def test_aspect_field_types(self, category: CategoryName) -> None:
        cls = get_output_schema(category)
        aspect_cls = _list_item_type(cls.model_fields["requirement_aspects"].annotation)

        # aspect_description and relation_to_endpoint are required strings.
        for required_field in ("aspect_description", "relation_to_endpoint"):
            f = aspect_cls.model_fields[required_field]
            assert f.annotation is str, f"{required_field} must be str"
            assert f.is_required(), f"{required_field} must be required"

        # coverage_gaps is nullable to let the LLM signal "fully covered".
        gaps = aspect_cls.model_fields["coverage_gaps"]
        assert _is_optional(gaps.annotation)
        assert _strip_optional(gaps.annotation) is str
        assert gaps.default is None


# ── MUTEX / TIERED shared shape ──────────────────────────────────────


MUTEX_ASPECT_FIELDS = {
    "aspect_description",
    "endpoint_coverage",
    "best_endpoint",
    "best_endpoint_gaps",
}


def _multi_endpoint_categories() -> list[CategoryName]:
    """Categories whose schemas use the multi-candidate aspect shape."""
    return [
        c
        for c in CategoryName
        if c.bucket in (HandlerBucket.MUTEX, HandlerBucket.TIERED)
        and c in OUTPUT_SCHEMAS
    ]


@pytest.mark.parametrize(
    "category", _multi_endpoint_categories(), ids=lambda c: c.name
)
class TestMutexAndTieredSharedShape:
    """Fields that mutex and tiered share, per the planning doc:
    'Tiered: same shape as Mutually exclusive [plus one extra field].'
    """

    def test_endpoint_to_run_literal_includes_candidates_and_none(
        self, category: CategoryName
    ) -> None:
        cls = get_output_schema(category)
        field = cls.model_fields["endpoint_to_run"]
        assert get_origin(field.annotation) is Literal
        values = set(get_args(field.annotation))
        expected = {r.value for r in _llm_wrapper_endpoints(category)} | {"None"}
        assert values == expected

    def test_endpoint_parameters_optional_union_over_wrappers(
        self, category: CategoryName
    ) -> None:
        # Doc: "endpoint_parameters: discriminated Union over the
        # endpoint param models. Left null when endpoint_to_run == None."
        cls = get_output_schema(category)
        field = cls.model_fields["endpoint_parameters"]
        assert _is_optional(field.annotation), (
            "endpoint_parameters must be nullable for the 'None' branch"
        )
        assert field.default is None
        inner = _strip_optional(field.annotation)
        union_members = set(get_args(inner)) if get_args(inner) else {inner}
        expected = {
            ROUTE_TO_WRAPPER[r] for r in _llm_wrapper_endpoints(category)
        }
        assert union_members == expected

    def test_aspect_field_set_matches_doc(self, category: CategoryName) -> None:
        cls = get_output_schema(category)
        aspect_cls = _list_item_type(cls.model_fields["requirement_aspects"].annotation)
        assert set(aspect_cls.model_fields) == MUTEX_ASPECT_FIELDS

    def test_aspect_field_types(self, category: CategoryName) -> None:
        cls = get_output_schema(category)
        aspect_cls = _list_item_type(cls.model_fields["requirement_aspects"].annotation)

        # aspect_description and endpoint_coverage are required strings.
        for required_field in ("aspect_description", "endpoint_coverage"):
            f = aspect_cls.model_fields[required_field]
            assert f.annotation is str
            assert f.is_required()

        # best_endpoint is a Literal over the candidates only — the
        # 'None' option lives on endpoint_to_run, not on per-aspect
        # picks (an aspect is only emitted when at least one endpoint
        # speaks to it).
        best = aspect_cls.model_fields["best_endpoint"]
        assert get_origin(best.annotation) is Literal
        assert set(get_args(best.annotation)) == {
            r.value for r in _llm_wrapper_endpoints(category)
        }
        assert best.is_required()

        # best_endpoint_gaps is nullable to signal "fully covered".
        gaps = aspect_cls.model_fields["best_endpoint_gaps"]
        assert _is_optional(gaps.annotation)
        assert _strip_optional(gaps.annotation) is str
        assert gaps.default is None


# ── MUTEX-specific (no extras) ───────────────────────────────────────


MUTEX_TOP_FIELDS = {"requirement_aspects", "endpoint_to_run", "endpoint_parameters"}


@pytest.mark.parametrize(
    "category",
    _categories_in_bucket(HandlerBucket.MUTEX),
    ids=lambda c: c.name,
)
class TestMutexBucketSchema:
    """Mutex must NOT carry the tiered-only bias-analysis field."""

    def test_top_level_fields_exact(self, category: CategoryName) -> None:
        cls = get_output_schema(category)
        assert set(cls.model_fields) == MUTEX_TOP_FIELDS

    def test_no_performance_vs_bias_analysis(self, category: CategoryName) -> None:
        cls = get_output_schema(category)
        assert "performance_vs_bias_analysis" not in cls.model_fields


# ── TIERED-specific (extra bias-analysis field) ──────────────────────


TIERED_TOP_FIELDS = {
    "requirement_aspects",
    "performance_vs_bias_analysis",
    "endpoint_to_run",
    "endpoint_parameters",
}


@pytest.mark.parametrize(
    "category",
    _categories_in_bucket(HandlerBucket.TIERED),
    ids=lambda c: c.name,
)
class TestTieredBucketSchema:
    """Tiered = mutex + a retrospective bias-vs-performance reasoning
    field placed BEFORE endpoint_to_run."""

    def test_top_level_fields_exact(self, category: CategoryName) -> None:
        cls = get_output_schema(category)
        assert set(cls.model_fields) == TIERED_TOP_FIELDS

    def test_performance_vs_bias_analysis_required_str(
        self, category: CategoryName
    ) -> None:
        cls = get_output_schema(category)
        f = cls.model_fields["performance_vs_bias_analysis"]
        assert f.annotation is str
        assert f.is_required()

    def test_bias_analysis_precedes_endpoint_to_run(
        self, category: CategoryName
    ) -> None:
        # Pydantic preserves field declaration order; the planning doc
        # is explicit that the model must reason about the bias
        # BEFORE committing to endpoint_to_run.
        order = list(get_output_schema(category).model_fields)
        assert order.index("performance_vs_bias_analysis") < order.index(
            "endpoint_to_run"
        )


# ── COMBO bucket ─────────────────────────────────────────────────────


COMBO_TOP_FIELDS = {
    "requirement_aspects",
    "overall_endpoint_fits",
    "per_endpoint_breakdown",
}
COMBO_ASPECT_FIELDS = {"aspect_description", "endpoint_coverage"}
COMBO_ENTRY_FIELDS = {"should_run_endpoint", "endpoint_parameters"}


@pytest.mark.parametrize(
    "category",
    _categories_in_bucket(HandlerBucket.COMBO),
    ids=lambda c: c.name,
)
class TestComboBucketSchema:
    """Combo is the only bucket with an enumerated per-endpoint
    breakdown — the planning doc calls the not-a-list shape a hard
    requirement so the LLM must address every endpoint explicitly."""

    def test_top_level_fields_exact(self, category: CategoryName) -> None:
        cls = get_output_schema(category)
        assert set(cls.model_fields) == COMBO_TOP_FIELDS

    def test_overall_endpoint_fits_required_str(
        self, category: CategoryName
    ) -> None:
        f = get_output_schema(category).model_fields["overall_endpoint_fits"]
        assert f.annotation is str
        assert f.is_required()

    def test_aspect_field_set_matches_doc(self, category: CategoryName) -> None:
        cls = get_output_schema(category)
        aspect_cls = _list_item_type(cls.model_fields["requirement_aspects"].annotation)
        assert set(aspect_cls.model_fields) == COMBO_ASPECT_FIELDS

    def test_aspect_field_types(self, category: CategoryName) -> None:
        cls = get_output_schema(category)
        aspect_cls = _list_item_type(cls.model_fields["requirement_aspects"].annotation)
        for required_field in COMBO_ASPECT_FIELDS:
            f = aspect_cls.model_fields[required_field]
            assert f.annotation is str
            assert f.is_required()

    def test_per_endpoint_breakdown_is_submodel_not_list(
        self, category: CategoryName
    ) -> None:
        # Doc: "per_endpoint_breakdown — NOT a freeform list. Every
        # candidate endpoint in the category is addressed explicitly."
        field = get_output_schema(category).model_fields["per_endpoint_breakdown"]
        breakdown_cls = field.annotation
        assert isinstance(breakdown_cls, type) and issubclass(breakdown_cls, BaseModel)
        assert get_origin(field.annotation) is not list

    def test_per_endpoint_breakdown_has_one_named_field_per_endpoint(
        self, category: CategoryName
    ) -> None:
        field = get_output_schema(category).model_fields["per_endpoint_breakdown"]
        breakdown_cls = field.annotation
        expected_keys = {r.value for r in _llm_wrapper_endpoints(category)}
        assert set(breakdown_cls.model_fields) == expected_keys

    def test_per_endpoint_entry_shape(self, category: CategoryName) -> None:
        breakdown_cls = (
            get_output_schema(category).model_fields["per_endpoint_breakdown"].annotation
        )
        for route in _llm_wrapper_endpoints(category):
            entry_field = breakdown_cls.model_fields[route.value]
            entry_cls = entry_field.annotation
            assert issubclass(entry_cls, BaseModel)
            assert set(entry_cls.model_fields) == COMBO_ENTRY_FIELDS

            # should_run_endpoint: required bool.
            srf = entry_cls.model_fields["should_run_endpoint"]
            assert srf.annotation is bool
            assert srf.is_required()

            # endpoint_parameters: Optional[wrapper for THIS endpoint].
            ep = entry_cls.model_fields["endpoint_parameters"]
            assert _is_optional(ep.annotation)
            assert ep.default is None
            assert _strip_optional(ep.annotation) is ROUTE_TO_WRAPPER[route]


# ── EndpointParameters wiring sanity check ───────────────────────────


class TestEndpointParametersWiring:
    """The wrappers slotted into every schema must inherit the shared
    EndpointParameters base — that's how action_role + polarity get
    onto every emission. This test guards against accidental swap-in
    of a non-wrapped parameter model."""

    def test_every_wrapper_inherits_endpoint_parameters(self) -> None:
        for route, wrapper in ROUTE_TO_WRAPPER.items():
            if wrapper is None:
                continue
            assert issubclass(wrapper, EndpointParameters), (
                f"{route.value} -> {wrapper.__name__} does not inherit "
                f"EndpointParameters"
            )
