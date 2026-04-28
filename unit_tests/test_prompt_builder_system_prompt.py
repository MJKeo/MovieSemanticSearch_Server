"""Unit tests for the per-category system prompt built by
search_v2/stage_3/category_handlers/prompt_builder.py.

These tests are pinned to the documented contract in
search_improvement_planning/category_handler_planning.md — specifically
§"Prompt assembly" → "Finalized section table" and §"Prompt builder
behavior" — and are deliberately written against the spec, not the
current implementation. If a test fails, the builder has drifted
from the doc.

Ground-truth chunk content is read directly from the markdown files
on disk so the builder's internal caches are not a source of truth
for the tests. A bug that routes the wrong chunk into the wrong slot,
drops a chunk entirely, or assembles sections out of order will
surface as a concrete assertion failure.

Documented invariants exercised:

  • The 8-chunk system prompt contains, in order:
        Role → Shared vocab → Endpoint context → Input spec
        → Core objective → Additional notes → Guardrails
        → Few-shot examples.
  • Shared sections (role, vocabulary, input spec) are byte-identical
    for every category.
  • Core objective + guardrails are keyed off `category.bucket`; a
    category never inherits another bucket's objective or guardrails.
  • Endpoint-context chunks appear iff the route is in
    `category.endpoints`, in the priority order declared there.
    TRENDING has no chunk and is silently skipped if ever listed.
  • Per-category additional notes and few-shot examples are keyed
    off `CategoryName.name.lower()` and differ per category.
  • CategoryName.TRENDING raises ValueError — it has no LLM codepath
    and must never fall through to the builder.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from schemas.enums import EndpointRoute, HandlerBucket
from schemas.trait_category import CategoryName
from search_v2.stage_3.category_handlers.prompt_builder import (
    build_system_prompt,
)


# ── Fixtures — independent ground truth ──────────────────────────────

# Path to the prompt chunks; resolved relative to the repo root so the
# tests don't depend on any import-time caches in prompt_builder.
_PROMPTS_DIR = (
    Path(__file__).resolve().parent.parent
    / "search_v2"
    / "stage_3"
    / "category_handlers"
    / "prompts"
)


def _read(path: Path) -> str:
    # Matches the builder's rstrip convention so substring checks are
    # byte-identical even when the file ends with a trailing newline.
    return path.read_text(encoding="utf-8").rstrip()


@pytest.fixture(scope="module")
def expected_chunks() -> dict:
    """Load every expected chunk directly from disk.

    Returned dict mirrors the taxonomy in the planning doc's section
    table: `shared`, `bucket_objective`, `bucket_guardrails`,
    `endpoint`, `notes`, `examples`.
    """

    shared = {
        "role": _read(_PROMPTS_DIR / "shared" / "role.md"),
        "vocabulary": _read(_PROMPTS_DIR / "shared" / "shared_vocabulary.md"),
        "input_spec": _read(_PROMPTS_DIR / "shared" / "input_spec.md"),
    }
    bucket_objective = {
        bucket: _read(_PROMPTS_DIR / "buckets" / f"{bucket.value}_objective.md")
        for bucket in HandlerBucket
    }
    bucket_guardrails = {
        bucket: _read(_PROMPTS_DIR / "buckets" / f"{bucket.value}_guardrails.md")
        for bucket in HandlerBucket
    }
    endpoint = {
        route: _read(_PROMPTS_DIR / "endpoints" / f"{route.value}.md")
        for route in EndpointRoute
        if route is not EndpointRoute.TRENDING
    }
    notes = {
        category: _read(
            _PROMPTS_DIR
            / "categories"
            / "additional_objective_notes"
            / f"{category.name.lower()}.md"
        )
        for category in CategoryName
        if category is not CategoryName.TRENDING
    }
    examples = {
        category: _read(
            _PROMPTS_DIR
            / "categories"
            / "few_shot_examples"
            / f"{category.name.lower()}.md"
        )
        for category in CategoryName
        if category is not CategoryName.TRENDING
    }
    return {
        "shared": shared,
        "bucket_objective": bucket_objective,
        "bucket_guardrails": bucket_guardrails,
        "endpoint": endpoint,
        "notes": notes,
        "examples": examples,
    }


@pytest.fixture(scope="module")
def built_prompts() -> dict[CategoryName, str]:
    # Build once per test session to keep the parametrized sweep fast.
    # Any import-time or build-time explosion will surface on the very
    # first test that touches the fixture.
    return {
        category: build_system_prompt(category)
        for category in CategoryName
        if category is not CategoryName.TRENDING
    }


_NON_TRENDING = [c for c in CategoryName if c is not CategoryName.TRENDING]


# ── Shared-section invariants ───────────────────────────────────────
# Every category — regardless of bucket or endpoint set — must include
# the three shared chunks verbatim. Planning doc §"Full system-prompt
# composition" table, rows 1/2/4.


@pytest.mark.parametrize("category", _NON_TRENDING, ids=lambda c: c.name)
class TestSharedSectionsPresent:
    def test_role_present(self, category, built_prompts, expected_chunks):
        assert expected_chunks["shared"]["role"] in built_prompts[category]

    def test_shared_vocabulary_present(
        self, category, built_prompts, expected_chunks
    ):
        assert (
            expected_chunks["shared"]["vocabulary"] in built_prompts[category]
        )

    def test_input_spec_present(self, category, built_prompts, expected_chunks):
        assert (
            expected_chunks["shared"]["input_spec"] in built_prompts[category]
        )


# ── Bucket-keyed section invariants ─────────────────────────────────
# Planning doc §"Finalized section table" rows 5 (objective) and 7
# (guardrails): one chunk per bucket, uniform across every category
# in the same bucket. A category MUST include its bucket's objective
# and guardrails and MUST NOT include any other bucket's.


@pytest.mark.parametrize("category", _NON_TRENDING, ids=lambda c: c.name)
class TestBucketSections:
    def test_correct_bucket_objective_present(
        self, category, built_prompts, expected_chunks
    ):
        expected = expected_chunks["bucket_objective"][category.bucket]
        assert expected in built_prompts[category]

    def test_correct_bucket_guardrails_present(
        self, category, built_prompts, expected_chunks
    ):
        expected = expected_chunks["bucket_guardrails"][category.bucket]
        assert expected in built_prompts[category]

    def test_no_other_bucket_objective_present(
        self, category, built_prompts, expected_chunks
    ):
        # A wrong-bucket objective leaking into the prompt would
        # silently misinform the LLM about its decision shape
        # (e.g. a SINGLE category told "pick which endpoint fits
        # best").
        prompt = built_prompts[category]
        for bucket, content in expected_chunks["bucket_objective"].items():
            if bucket is not category.bucket:
                assert content not in prompt, (
                    f"{category.name} ({category.bucket.value}) leaked the "
                    f"{bucket.value} objective chunk"
                )

    def test_no_other_bucket_guardrails_present(
        self, category, built_prompts, expected_chunks
    ):
        prompt = built_prompts[category]
        for bucket, content in expected_chunks["bucket_guardrails"].items():
            if bucket is not category.bucket:
                assert content not in prompt, (
                    f"{category.name} ({category.bucket.value}) leaked the "
                    f"{bucket.value} guardrails chunk"
                )


# ── Endpoint context invariants ─────────────────────────────────────
# Planning doc §"Finalized section table" row 3 plus §"Endpoint-context
# chunks are static per endpoint": one chunk per endpoint in the
# category's tuple, appended verbatim, in the tuple's priority order.
# TRENDING has no LLM chunk (and in the enum today is only listed by
# the TRENDING category which never reaches the builder).


@pytest.mark.parametrize("category", _NON_TRENDING, ids=lambda c: c.name)
class TestEndpointChunks:
    def test_each_categorys_endpoints_present(
        self, category, built_prompts, expected_chunks
    ):
        prompt = built_prompts[category]
        for route in category.endpoints:
            if route is EndpointRoute.TRENDING:
                continue
            assert expected_chunks["endpoint"][route] in prompt, (
                f"{category.name} missing endpoint chunk for "
                f"{route.value}"
            )

    def test_non_categorys_endpoints_absent(
        self, category, built_prompts, expected_chunks
    ):
        prompt = built_prompts[category]
        for route, content in expected_chunks["endpoint"].items():
            if route in category.endpoints:
                continue
            assert content not in prompt, (
                f"{category.name} leaked endpoint chunk for "
                f"{route.value} (not in category.endpoints: "
                f"{[r.value for r in category.endpoints]})"
            )


_MULTI_ENDPOINT_CATEGORIES = [
    c
    for c in _NON_TRENDING
    if len([r for r in c.endpoints if r is not EndpointRoute.TRENDING]) > 1
]


@pytest.mark.parametrize(
    "category", _MULTI_ENDPOINT_CATEGORIES, ids=lambda c: c.name
)
def test_endpoint_chunks_in_priority_order(
    category, built_prompts, expected_chunks
):
    # Planning doc §"Modular handler construction" frames
    # `category.endpoints` as priority-ordered. For tiered buckets
    # the bias reasoning only makes sense if the LLM sees the
    # endpoints in that order; for mutex and combo it keeps the
    # prompt stable across the taxonomy.
    prompt = built_prompts[category]
    positions = [
        prompt.index(expected_chunks["endpoint"][route])
        for route in category.endpoints
        if route is not EndpointRoute.TRENDING
    ]
    assert positions == sorted(positions), (
        f"{category.name} endpoints out of priority order: "
        f"declared {[r.value for r in category.endpoints]} but positions "
        f"were {positions}"
    )


# ── Per-category section invariants ─────────────────────────────────
# Planning doc §"Finalized section table" rows 6 (additional notes)
# and 8 (few-shot examples): one chunk per category, keyed by
# `CategoryName.name.lower()`. Notes and examples MUST be the ones
# belonging to the built category and MUST NOT be another category's.


@pytest.mark.parametrize("category", _NON_TRENDING, ids=lambda c: c.name)
class TestPerCategorySections:
    def test_additional_notes_present(
        self, category, built_prompts, expected_chunks
    ):
        assert (
            expected_chunks["notes"][category] in built_prompts[category]
        )

    def test_few_shot_examples_present(
        self, category, built_prompts, expected_chunks
    ):
        assert (
            expected_chunks["examples"][category] in built_prompts[category]
        )

    def test_no_other_categorys_notes_present(
        self, category, built_prompts, expected_chunks
    ):
        # Catches a bug that looked up category-keyed chunks by
        # index / value / hash instead of by CategoryName.name —
        # the wrong category's notes would still be a valid chunk
        # but would misinform the LLM about the category it's
        # handling.
        prompt = built_prompts[category]
        for other, content in expected_chunks["notes"].items():
            if other is category:
                continue
            assert content not in prompt, (
                f"{category.name} leaked notes for {other.name}"
            )

    def test_no_other_categorys_examples_present(
        self, category, built_prompts, expected_chunks
    ):
        prompt = built_prompts[category]
        for other, content in expected_chunks["examples"].items():
            if other is category:
                continue
            assert content not in prompt, (
                f"{category.name} leaked examples for {other.name}"
            )


# ── Section ordering ────────────────────────────────────────────────
# Planning doc §"Finalized section table": the eight sections must
# appear in the listed order. Order matters for small-model attention
# anchoring per §"Prompt assembly" rationale.


@pytest.mark.parametrize("category", _NON_TRENDING, ids=lambda c: c.name)
def test_sections_in_planning_doc_order(
    category, built_prompts, expected_chunks
):
    prompt = built_prompts[category]

    # Use the first non-TRENDING endpoint as the anchor for the
    # endpoint-context slot — every non-TRENDING category has at
    # least one such endpoint by construction.
    first_endpoint = next(
        r for r in category.endpoints if r is not EndpointRoute.TRENDING
    )

    positions = {
        "role": prompt.index(expected_chunks["shared"]["role"]),
        "vocabulary": prompt.index(expected_chunks["shared"]["vocabulary"]),
        "endpoint": prompt.index(expected_chunks["endpoint"][first_endpoint]),
        "input_spec": prompt.index(expected_chunks["shared"]["input_spec"]),
        "objective": prompt.index(
            expected_chunks["bucket_objective"][category.bucket]
        ),
        "notes": prompt.index(expected_chunks["notes"][category]),
        "guardrails": prompt.index(
            expected_chunks["bucket_guardrails"][category.bucket]
        ),
        "examples": prompt.index(expected_chunks["examples"][category]),
    }
    declared_order = [
        "role",
        "vocabulary",
        "endpoint",
        "input_spec",
        "objective",
        "notes",
        "guardrails",
        "examples",
    ]
    actual_order = sorted(positions, key=positions.get)
    assert actual_order == declared_order, (
        f"{category.name} assembled sections out of order: "
        f"got {actual_order}, expected {declared_order}"
    )


# ── TRENDING raises ─────────────────────────────────────────────────
# Planning doc §"Prompt builder behavior": TRENDING has no LLM handler
# and the builder must raise rather than silently producing a prompt.


def test_trending_raises_value_error():
    with pytest.raises(ValueError, match="TRENDING"):
        build_system_prompt(CategoryName.TRENDING)


# ── Coverage sanity check ───────────────────────────────────────────
# Planning doc §"Category-handlers module layout": one notes file and
# one examples file per CategoryName except TRENDING. This test is a
# guardrail against future enum additions silently drifting past the
# prompt-authoring pipeline.


def test_every_non_trending_category_has_both_chunks(expected_chunks):
    expected_members = {c for c in CategoryName if c is not CategoryName.TRENDING}
    assert set(expected_chunks["notes"]) == expected_members
    assert set(expected_chunks["examples"]) == expected_members
