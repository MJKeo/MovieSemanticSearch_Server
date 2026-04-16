"""Unit tests for keyword_query_generation._build_classification_registry_section.

Verifies the rendered classification-registry section of the step 3
keyword endpoint system prompt against the underlying enums
(OverallKeyword, SourceMaterialType, ALL_CONCEPT_TAGS). The
expected content is derived from the enum values themselves — not
from what the rendering function happens to produce — so drift
between the enums and the rendered prompt surfaces as a test
failure.

Collision handling: OverallKeyword wins over SourceMaterialType and
ConceptTag on name collisions (the registry drops the non-keyword
entry). The test reconstructs that precedence independently before
checking membership.
"""

import re

import pytest

from implementation.classes.overall_keywords import OverallKeyword
from schemas.enums import ALL_CONCEPT_TAGS, SourceMaterialType
from schemas.unified_classification import entry_for, UnifiedClassification
from search_v2.stage_3 import keyword_query_generation
from search_v2.stage_3.keyword_query_generation import (
    _FAMILIES,
    _build_classification_registry_section,
)


# ---------------------------------------------------------------------------
# Ground truth reconstructed from the source enums.
# ---------------------------------------------------------------------------


def _expected_name_to_definition() -> dict[str, str]:
    """Build the expected {member_name: definition} map from source enums.

    OverallKeyword is iterated first so its entries win name collisions
    with SourceMaterialType and ConceptTag — matching the collision
    rule implemented in schemas.unified_classification._build_registry.
    Definitions for SourceMaterialType members (which have no definition
    on the enum itself) are pulled from the registry entry, because the
    hand-authored definitions live in unified_classification.py and
    that IS the source of truth for those members.
    """
    mapping: dict[str, str] = {}

    for kw in OverallKeyword:
        mapping[kw.name] = kw.definition

    for smt in SourceMaterialType:
        if smt.name in mapping:
            continue
        # Definitions for SourceMaterialType live in the registry
        # (hand-authored in unified_classification.py). Pulling from
        # entry_for keeps this a single source of truth and still
        # independent of the rendering function under test.
        mapping[smt.name] = entry_for(UnifiedClassification(smt.name)).definition

    for tag in ALL_CONCEPT_TAGS:
        if tag.name in mapping:
            continue
        mapping[tag.name] = tag.description

    return mapping


EXPECTED_NAME_TO_DEFINITION: dict[str, str] = _expected_name_to_definition()


# ---------------------------------------------------------------------------
# Rendered section under test (built once per test module).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rendered_section() -> str:
    return _build_classification_registry_section()


# ---------------------------------------------------------------------------
# Structural checks — family headers, size, shape.
# ---------------------------------------------------------------------------


def test_header_present(rendered_section: str) -> None:
    assert rendered_section.splitlines()[0] == "CLASSIFICATION REGISTRY"


def test_total_member_count_matches_source_enums(rendered_section: str) -> None:
    # Every rendered member line has the shape `NAME — definition`.
    # NAMES are all-caps tokens with underscores/digits. Counting them
    # independently of _FAMILIES catches both duplicates and omissions.
    member_lines = [
        line for line in rendered_section.splitlines()
        if re.match(r"^[A-Z][A-Z0-9_]*\s+—\s+.+$", line)
    ]
    assert len(member_lines) == len(EXPECTED_NAME_TO_DEFINITION)


def test_all_twenty_one_family_headers_present(rendered_section: str) -> None:
    # Derived from _FAMILIES itself only for the header strings — the
    # fact that there are 21 is asserted independently so a renumbering
    # or split/merge surfaces immediately.
    family_headers = [header for header, _ in _FAMILIES]
    assert len(family_headers) == 21
    for header in family_headers:
        assert header in rendered_section, f"Missing family header: {header!r}"


# ---------------------------------------------------------------------------
# Per-member checks — every enum member is rendered with its real
# definition. Parametrized so failures point at the exact member.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,definition", sorted(EXPECTED_NAME_TO_DEFINITION.items()))
def test_every_enum_member_rendered_with_definition(
    rendered_section: str, name: str, definition: str
) -> None:
    expected_line = f"{name} — {definition}"
    assert expected_line in rendered_section, (
        f"Expected rendered line for {name!r} not found: {expected_line!r}"
    )


@pytest.mark.parametrize("name", sorted(EXPECTED_NAME_TO_DEFINITION))
def test_each_member_name_appears_exactly_once_as_a_line(
    rendered_section: str, name: str
) -> None:
    # A member's NAME could theoretically show up inside another member's
    # definition text; anchor on the `^NAME — ` line form to count only
    # rendered entries.
    pattern = re.compile(rf"^{re.escape(name)}\s+—\s+.+$", re.MULTILINE)
    matches = pattern.findall(rendered_section)
    assert len(matches) == 1, (
        f"Expected exactly one rendered line for {name!r}, got {len(matches)}"
    )


# ---------------------------------------------------------------------------
# Collision precedence — verify OverallKeyword wins over other sources.
# ---------------------------------------------------------------------------


def test_overall_keyword_wins_collision_with_source_material(
    rendered_section: str,
) -> None:
    kw_names = {kw.name for kw in OverallKeyword}
    smt_names = {smt.name for smt in SourceMaterialType}
    tag_names = {tag.name for tag in ALL_CONCEPT_TAGS}
    collisions = kw_names & (smt_names | tag_names)

    # If the test data has no collisions this test is vacuous but still
    # correct — assert at least one so silent drift in the enums does
    # not hide a regression in the precedence rule.
    assert collisions, "Expected at least one name collision; enums may have drifted"

    for name in collisions:
        expected_definition = OverallKeyword[name].definition
        expected_line = f"{name} — {expected_definition}"
        assert expected_line in rendered_section, (
            f"Collision {name!r} should resolve to OverallKeyword definition"
        )


# ---------------------------------------------------------------------------
# Import-time consistency checks — corrupting _FAMILIES must raise.
# ---------------------------------------------------------------------------


def test_missing_member_from_families_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Drop one member from a family and re-run the builder. The
    # "some registry members were not placed in any family" path must
    # fire.
    original = keyword_query_generation._FAMILIES
    mutated = [(header, list(members)) for header, members in original]
    # Remove an arbitrary member that we know is in the registry.
    removed_name = mutated[0][1].pop(0)
    monkeypatch.setattr(keyword_query_generation, "_FAMILIES", mutated)

    with pytest.raises(RuntimeError, match=removed_name):
        _build_classification_registry_section()


def test_unknown_member_in_families_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Inject a name that is not in CLASSIFICATION_ENTRIES. The
    # "referenced by _FAMILIES is not in the registry" path must fire.
    original = keyword_query_generation._FAMILIES
    mutated = [(header, list(members)) for header, members in original]
    mutated[0][1].append("TOTALLY_MADE_UP_MEMBER")
    monkeypatch.setattr(keyword_query_generation, "_FAMILIES", mutated)

    with pytest.raises(RuntimeError, match="TOTALLY_MADE_UP_MEMBER"):
        _build_classification_registry_section()


def test_duplicate_member_across_families_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Add an existing member to a second family. The "listed in more
    # than one family" path must fire before the missing-members check.
    original = keyword_query_generation._FAMILIES
    mutated = [(header, list(members)) for header, members in original]
    duplicated_name = mutated[0][1][0]
    mutated[1][1].append(duplicated_name)
    monkeypatch.setattr(keyword_query_generation, "_FAMILIES", mutated)

    with pytest.raises(RuntimeError, match=duplicated_name):
        _build_classification_registry_section()
