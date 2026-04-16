"""Unit tests for the unified classification registry (schemas/unified_classification.py).

Verifies every source-enum member surfaces correctly in the registry:
name, display, source, and source_id match the originating enum.
Collision precedence (OverallKeyword wins over SourceMaterialType
and ConceptTag) is also verified. Definition values are only checked
for presence, not content.
"""

import pytest

from implementation.classes.overall_keywords import OverallKeyword
from schemas.enums import ALL_CONCEPT_TAGS, SourceMaterialType
from schemas.unified_classification import (
    CLASSIFICATION_ENTRIES,
    ClassificationSource,
    UnifiedClassification,
    entry_for,
)


# Names that exist in more than one source enum. These must resolve
# to OverallKeyword in the registry (keyword takes precedence).
COLLISIONS: set[str] = (
    {kw.name for kw in OverallKeyword}
    & ({smt.name for smt in SourceMaterialType} | {tag.name for tag in ALL_CONCEPT_TAGS})
)


def test_total_entry_count_matches_sources_minus_collisions() -> None:
    expected = (
        len(list(OverallKeyword))
        + len(list(SourceMaterialType))
        + len(ALL_CONCEPT_TAGS)
        - len(COLLISIONS)
    )
    assert len(CLASSIFICATION_ENTRIES) == expected


def test_unified_enum_matches_registry_keys() -> None:
    registry_names = set(CLASSIFICATION_ENTRIES.keys())
    enum_names = {m.name for m in UnifiedClassification}
    enum_values = {m.value for m in UnifiedClassification}
    assert registry_names == enum_names
    assert registry_names == enum_values


def test_known_collision_resolves_to_keyword() -> None:
    # BIOGRAPHY exists in both OverallKeyword and SourceMaterialType.
    # Registry must expose the OverallKeyword version.
    assert "BIOGRAPHY" in COLLISIONS
    entry = CLASSIFICATION_ENTRIES["BIOGRAPHY"]
    assert entry.source is ClassificationSource.KEYWORD
    assert entry.source_id == OverallKeyword.BIOGRAPHY.keyword_id


# ---------------------------------------------------------------------------
# Per-source coverage: every member of each source enum must appear in the
# registry (except collisions with a higher-precedence source). For each
# present member, verify name, display, source, and source_id.
# Definition is only checked for non-empty presence.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kw", list(OverallKeyword), ids=lambda k: k.name)
def test_every_overall_keyword_registered(kw: OverallKeyword) -> None:
    assert kw.name in CLASSIFICATION_ENTRIES, f"OverallKeyword.{kw.name} missing from registry"
    entry = CLASSIFICATION_ENTRIES[kw.name]
    assert entry.name == kw.name
    assert entry.display == kw.value
    assert entry.source is ClassificationSource.KEYWORD
    assert entry.source_id == kw.keyword_id
    assert entry.backing_column == "keyword_ids"
    assert isinstance(entry.definition, str) and entry.definition


@pytest.mark.parametrize("smt", list(SourceMaterialType), ids=lambda s: s.name)
def test_every_source_material_type_registered_or_shadowed(smt: SourceMaterialType) -> None:
    entry = CLASSIFICATION_ENTRIES[smt.name]
    if smt.name in COLLISIONS:
        # Shadowed by OverallKeyword — confirmed in the dedicated collision test.
        assert entry.source is ClassificationSource.KEYWORD
        return
    assert entry.name == smt.name
    assert entry.source is ClassificationSource.SOURCE_MATERIAL
    assert entry.source_id == smt.source_material_type_id
    assert entry.backing_column == "source_material_type_ids"
    # Display must be non-empty and not equal to the raw enum member name
    # (hand-written labels preserve acronyms like "TV Adaptation").
    assert isinstance(entry.display, str) and entry.display
    assert entry.display != smt.name
    assert isinstance(entry.definition, str) and entry.definition


@pytest.mark.parametrize("tag", list(ALL_CONCEPT_TAGS), ids=lambda t: t.name)
def test_every_concept_tag_registered_or_shadowed(tag) -> None:
    entry = CLASSIFICATION_ENTRIES[tag.name]
    if tag.name in COLLISIONS:
        assert entry.source is ClassificationSource.KEYWORD
        return
    assert entry.name == tag.name
    assert entry.display == tag.name.replace("_", " ").title()
    assert entry.source is ClassificationSource.CONCEPT_TAG
    assert entry.source_id == tag.concept_tag_id
    assert entry.backing_column == "concept_tag_ids"
    assert isinstance(entry.definition, str) and entry.definition


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------


def test_entry_for_roundtrips_every_member() -> None:
    # Every UnifiedClassification member must resolve via entry_for().
    for member in UnifiedClassification:
        entry = entry_for(member)
        assert entry.name == member.value


def test_no_duplicate_source_id_within_source() -> None:
    # (source, source_id) pairs must be unique — two entries pointing to
    # the same backing ID would silently merge at query time.
    seen: set[tuple[ClassificationSource, int]] = set()
    for entry in CLASSIFICATION_ENTRIES.values():
        key = (entry.source, entry.source_id)
        assert key not in seen, f"Duplicate ({entry.source.value}, {entry.source_id}) at {entry.name}"
        seen.add(key)
