"""Unit tests for concept tag enums in schemas/enums.py.

Covers stability (persisted values/IDs), the new description attribute,
behavioral properties (str subclass, value lookup), and ALL_CONCEPT_TAGS
correctness including global ID uniqueness across all 7 enums.
"""

import pytest

from schemas.enums import (
    ALL_CONCEPT_TAGS,
    CharacterTag,
    ContentFlagTag,
    EndingTag,
    ExperientialTag,
    NarrativeStructureTag,
    PlotArchetypeTag,
    SettingTag,
)


# ---------------------------------------------------------------------------
# Expected members with their stable (value, concept_tag_id) pairs.
# These are persisted in LLM structured output and GIN-indexed in Postgres —
# any change is a breaking/data-corruption bug.
# ---------------------------------------------------------------------------

_EXPECTED_NARRATIVE_STRUCTURE_TAGS = {
    "PLOT_TWIST":           ("plot_twist", 1),
    "TWIST_VILLAIN":        ("twist_villain", 2),
    "TIME_LOOP":            ("time_loop", 3),
    "NONLINEAR_TIMELINE":   ("nonlinear_timeline", 4),
    "UNRELIABLE_NARRATOR":  ("unreliable_narrator", 5),
    "OPEN_ENDING":          ("open_ending", 6),
    "SINGLE_LOCATION":      ("single_location", 7),
    "BREAKING_FOURTH_WALL": ("breaking_fourth_wall", 8),
    "CLIFFHANGER_ENDING":   ("cliffhanger_ending", 9),
}

_EXPECTED_PLOT_ARCHETYPE_TAGS = {
    "REVENGE":    ("revenge", 11),
    "UNDERDOG":   ("underdog", 12),
    "KIDNAPPING": ("kidnapping", 13),
    "CON_ARTIST": ("con_artist", 14),
}

_EXPECTED_SETTING_TAGS = {
    "POST_APOCALYPTIC": ("post_apocalyptic", 21),
    "HAUNTED_LOCATION": ("haunted_location", 22),
    "SMALL_TOWN":       ("small_town", 23),
}

_EXPECTED_CHARACTER_TAGS = {
    "FEMALE_LEAD":   ("female_lead", 31),
    "ENSEMBLE_CAST": ("ensemble_cast", 32),
    "ANTI_HERO":     ("anti_hero", 33),
}

_EXPECTED_ENDING_TAGS = {
    "HAPPY_ENDING":       ("happy_ending", 41),
    "SAD_ENDING":         ("sad_ending", 42),
    "BITTERSWEET_ENDING": ("bittersweet_ending", 43),
    "NO_CLEAR_CHOICE":    ("no_clear_choice", -1),
}

_EXPECTED_EXPERIENTIAL_TAGS = {
    "FEEL_GOOD":  ("feel_good", 51),
    "TEARJERKER": ("tearjerker", 52),
}

_EXPECTED_CONTENT_FLAG_TAGS = {
    "ANIMAL_DEATH": ("animal_death", 61),
}


# ---------------------------------------------------------------------------
# Stability tests — one class per enum
# ---------------------------------------------------------------------------

class TestNarrativeStructureTagStability:
    def test_values_are_stable(self):
        """All 9 members exist with their exact (value, concept_tag_id) pairs."""
        for member_name, (expected_value, expected_id) in _EXPECTED_NARRATIVE_STRUCTURE_TAGS.items():
            member = NarrativeStructureTag[member_name]
            assert member.value == expected_value
            assert member.concept_tag_id == expected_id

    def test_member_count(self):
        """Exactly 9 members — catches accidental additions or removals."""
        assert len(NarrativeStructureTag) == 9


class TestPlotArchetypeTagStability:
    def test_values_are_stable(self):
        """All 4 members exist with their exact (value, concept_tag_id) pairs."""
        for member_name, (expected_value, expected_id) in _EXPECTED_PLOT_ARCHETYPE_TAGS.items():
            member = PlotArchetypeTag[member_name]
            assert member.value == expected_value
            assert member.concept_tag_id == expected_id

    def test_member_count(self):
        """Exactly 4 members — catches accidental additions or removals."""
        assert len(PlotArchetypeTag) == 4


class TestSettingTagStability:
    def test_values_are_stable(self):
        """All 3 members exist with their exact (value, concept_tag_id) pairs."""
        for member_name, (expected_value, expected_id) in _EXPECTED_SETTING_TAGS.items():
            member = SettingTag[member_name]
            assert member.value == expected_value
            assert member.concept_tag_id == expected_id

    def test_member_count(self):
        """Exactly 3 members — catches accidental additions or removals."""
        assert len(SettingTag) == 3


class TestCharacterTagStability:
    def test_values_are_stable(self):
        """All 3 members exist with their exact (value, concept_tag_id) pairs."""
        for member_name, (expected_value, expected_id) in _EXPECTED_CHARACTER_TAGS.items():
            member = CharacterTag[member_name]
            assert member.value == expected_value
            assert member.concept_tag_id == expected_id

    def test_member_count(self):
        """Exactly 3 members — catches accidental additions or removals."""
        assert len(CharacterTag) == 3


class TestEndingTagStability:
    def test_values_are_stable(self):
        """All 4 members (including NO_CLEAR_CHOICE) exist with their exact pairs."""
        for member_name, (expected_value, expected_id) in _EXPECTED_ENDING_TAGS.items():
            member = EndingTag[member_name]
            assert member.value == expected_value
            assert member.concept_tag_id == expected_id

    def test_member_count(self):
        """Exactly 4 members — catches accidental additions or removals."""
        assert len(EndingTag) == 4


class TestExperientialTagStability:
    def test_values_are_stable(self):
        """All 2 members exist with their exact (value, concept_tag_id) pairs."""
        for member_name, (expected_value, expected_id) in _EXPECTED_EXPERIENTIAL_TAGS.items():
            member = ExperientialTag[member_name]
            assert member.value == expected_value
            assert member.concept_tag_id == expected_id

    def test_member_count(self):
        """Exactly 2 members — catches accidental additions or removals."""
        assert len(ExperientialTag) == 2


class TestContentFlagTagStability:
    def test_values_are_stable(self):
        """1 member exists with its exact (value, concept_tag_id) pair."""
        for member_name, (expected_value, expected_id) in _EXPECTED_CONTENT_FLAG_TAGS.items():
            member = ContentFlagTag[member_name]
            assert member.value == expected_value
            assert member.concept_tag_id == expected_id

    def test_member_count(self):
        """Exactly 1 member — catches accidental additions or removals."""
        assert len(ContentFlagTag) == 1


# ---------------------------------------------------------------------------
# Description attribute tests
# ---------------------------------------------------------------------------

class TestConceptTagDescriptions:
    def test_all_concept_tags_have_nonempty_description(self):
        """Every tag in ALL_CONCEPT_TAGS has a non-empty description string."""
        for tag in ALL_CONCEPT_TAGS:
            assert isinstance(tag.description, str), f"{tag} description is not a str"
            assert len(tag.description) > 0, f"{tag} has empty description"

    def test_no_clear_choice_has_description(self):
        """NO_CLEAR_CHOICE (excluded from ALL_CONCEPT_TAGS) also has a description."""
        assert isinstance(EndingTag.NO_CLEAR_CHOICE.description, str)
        assert len(EndingTag.NO_CLEAR_CHOICE.description) > 0

    def test_description_is_str_type(self):
        """.description attribute is a str instance (not accidentally an int from tuple unpacking)."""
        # Check one representative from each enum
        assert isinstance(NarrativeStructureTag.PLOT_TWIST.description, str)
        assert isinstance(PlotArchetypeTag.REVENGE.description, str)
        assert isinstance(SettingTag.POST_APOCALYPTIC.description, str)
        assert isinstance(CharacterTag.FEMALE_LEAD.description, str)
        assert isinstance(EndingTag.HAPPY_ENDING.description, str)
        assert isinstance(ExperientialTag.FEEL_GOOD.description, str)
        assert isinstance(ContentFlagTag.ANIMAL_DEATH.description, str)


# ---------------------------------------------------------------------------
# Behavior tests
# ---------------------------------------------------------------------------

class TestConceptTagBehavior:
    def test_is_str_subclass(self):
        """Required for Pydantic JSON schema enum constraints in LLM structured output."""
        assert isinstance(NarrativeStructureTag.PLOT_TWIST, str)
        assert isinstance(PlotArchetypeTag.REVENGE, str)
        assert isinstance(SettingTag.POST_APOCALYPTIC, str)
        assert isinstance(CharacterTag.FEMALE_LEAD, str)
        assert isinstance(EndingTag.HAPPY_ENDING, str)
        assert isinstance(ExperientialTag.FEEL_GOOD, str)
        assert isinstance(ContentFlagTag.ANIMAL_DEATH, str)

    def test_value_lookup(self):
        """Pydantic deserializes enum values from LLM JSON responses via value lookup."""
        assert NarrativeStructureTag("plot_twist") == NarrativeStructureTag.PLOT_TWIST

    def test_invalid_value_raises(self):
        """Unknown string values must raise ValueError."""
        with pytest.raises(ValueError):
            NarrativeStructureTag("nonexistent")

    def test_concept_tag_id_accessible(self):
        """.concept_tag_id is accessible as an int attribute on each member."""
        assert isinstance(NarrativeStructureTag.PLOT_TWIST.concept_tag_id, int)
        assert NarrativeStructureTag.PLOT_TWIST.concept_tag_id == 1
        assert isinstance(ContentFlagTag.ANIMAL_DEATH.concept_tag_id, int)
        assert ContentFlagTag.ANIMAL_DEATH.concept_tag_id == 61


# ---------------------------------------------------------------------------
# ALL_CONCEPT_TAGS tests
# ---------------------------------------------------------------------------

class TestAllConceptTags:
    def test_excludes_no_clear_choice(self):
        """EndingTag.NO_CLEAR_CHOICE is not in ALL_CONCEPT_TAGS."""
        assert EndingTag.NO_CLEAR_CHOICE not in ALL_CONCEPT_TAGS

    def test_includes_all_positive_id_tags(self):
        """Every member with concept_tag_id >= 0 from all 7 enums is present."""
        all_enums = [
            NarrativeStructureTag,
            PlotArchetypeTag,
            SettingTag,
            CharacterTag,
            EndingTag,
            ExperientialTag,
            ContentFlagTag,
        ]
        for enum_cls in all_enums:
            for member in enum_cls:
                if member.concept_tag_id >= 0:
                    assert member in ALL_CONCEPT_TAGS, f"{member} missing from ALL_CONCEPT_TAGS"

    def test_count(self):
        """Expected total: 9 + 4 + 3 + 3 + 3 + 2 + 1 = 25 storable tags."""
        assert len(ALL_CONCEPT_TAGS) == 25

    def test_no_duplicate_ids_across_enums(self):
        """concept_tag_ids are globally unique across all 7 enums (critical for GIN index)."""
        ids = [tag.concept_tag_id for tag in ALL_CONCEPT_TAGS]
        assert len(ids) == len(set(ids)), f"Duplicate IDs found: {[i for i in ids if ids.count(i) > 1]}"

    def test_no_duplicate_values_across_enums(self):
        """String values are globally unique across all 7 enums."""
        values = [tag.value for tag in ALL_CONCEPT_TAGS]
        assert len(values) == len(set(values)), f"Duplicate values: {[v for v in values if values.count(v) > 1]}"
