"""Unit tests for enum conversion and string formatting behavior."""

import pytest

from implementation.classes.enums import MaturityRating, StreamingAccessType
from schemas.enums import MetadataType, SourceMaterialType


@pytest.mark.parametrize(
    ("raw_method", "expected"),
    [
        ("subscription", StreamingAccessType.SUBSCRIPTION),
        ("buy", StreamingAccessType.BUY),
        ("rent", StreamingAccessType.RENT),
        ("invalid", None),
    ],
)
def test_streaming_access_type_from_string(raw_method: str, expected: StreamingAccessType | None) -> None:
    """StreamingAccessType.from_string should normalize case/spacing and reject unknown values."""
    assert StreamingAccessType.from_string(raw_method) == expected


@pytest.mark.parametrize(
    ("method_id", "expected"),
    [
        (1, StreamingAccessType.SUBSCRIPTION),
        (2, StreamingAccessType.BUY),
        (3, StreamingAccessType.RENT),
        (999, None),
    ],
)
def test_streaming_access_type_from_type_id(method_id: int, expected: StreamingAccessType | None) -> None:
    """StreamingAccessType.from_type_id should resolve known IDs and reject unknown ones."""
    assert StreamingAccessType.from_type_id(method_id) == expected


def test_maturity_rating_enum_labels_are_stable() -> None:
    """MaturityRating labels should remain stable for parsing and display."""
    assert MaturityRating.G.value == "g"
    assert MaturityRating.PG.value == "pg"
    assert MaturityRating.PG_13.value == "pg-13"
    assert MaturityRating.R.value == "r"
    assert MaturityRating.NC_17.value == "nc-17"
    assert MaturityRating.UNRATED.value == "unrated"


def test_maturity_rating_maturity_ranks_are_stable() -> None:
    """Maturity ranks should remain stable for persisted numeric comparisons."""
    assert MaturityRating.G.maturity_rank == 1
    assert MaturityRating.PG.maturity_rank == 2
    assert MaturityRating.PG_13.maturity_rank == 3
    assert MaturityRating.R.maturity_rank == 4
    assert MaturityRating.NC_17.maturity_rank == 5
    assert MaturityRating.UNRATED.maturity_rank == 999


def test_streaming_access_type_enum_values_are_stable() -> None:
    """StreamingAccessType values should remain stable for watch-offering keys."""
    assert StreamingAccessType.SUBSCRIPTION.type_id == 1
    assert StreamingAccessType.BUY.type_id == 2
    assert StreamingAccessType.RENT.type_id == 3
    assert StreamingAccessType.SUBSCRIPTION.value == "subscription"
    assert StreamingAccessType.BUY.value == "buy"
    assert StreamingAccessType.RENT.value == "rent"


# ---------------------------------------------------------------------------
# SourceMaterialType
# ---------------------------------------------------------------------------

# Expected members with their stable (value, id) pairs.
# These are persisted in LLM structured output and will be GIN-indexed
# in Postgres — any change is a breaking/data-corruption bug.
_EXPECTED_SOURCE_MATERIAL_TYPES = {
    "NOVEL_ADAPTATION":       ("novel_adaptation", 1),
    "SHORT_STORY_ADAPTATION": ("short_story_adaptation", 2),
    "STAGE_ADAPTATION":       ("stage_adaptation", 3),
    "TRUE_STORY":             ("true_story", 4),
    "BIOGRAPHY":              ("biography", 5),
    "COMIC_ADAPTATION":       ("comic_adaptation", 6),
    "FOLKLORE_ADAPTATION":    ("folklore_adaptation", 7),
    "VIDEO_GAME_ADAPTATION":  ("video_game_adaptation", 8),
    "REMAKE":                 ("remake", 9),
    "TV_ADAPTATION":          ("tv_adaptation", 10),
}


class TestSourceMaterialTypeStability:
    def test_source_material_type_values_are_stable(self):
        """All 10 members exist with their exact string values."""
        for member_name, (expected_value, _) in _EXPECTED_SOURCE_MATERIAL_TYPES.items():
            member = SourceMaterialType[member_name]
            assert member.value == expected_value

    def test_source_material_type_ids_are_stable(self):
        """All 10 members have their exact integer IDs (1-10)."""
        for member_name, (_, expected_id) in _EXPECTED_SOURCE_MATERIAL_TYPES.items():
            member = SourceMaterialType[member_name]
            assert member.source_material_type_id == expected_id

    def test_source_material_type_member_count(self):
        """Exactly 10 members — catches accidental additions or removals."""
        assert len(SourceMaterialType) == 10


class TestSourceMaterialTypeBehavior:
    def test_is_str_subclass(self):
        """Required for Pydantic JSON schema enum constraints in LLM structured output."""
        assert isinstance(SourceMaterialType.NOVEL_ADAPTATION, str)

    def test_value_matches_expected_string(self):
        """Pydantic serialization depends on .value returning the enum string."""
        assert SourceMaterialType.NOVEL_ADAPTATION.value == "novel_adaptation"
        # str, Enum subclass — equality with the raw string works via __eq__
        assert SourceMaterialType.NOVEL_ADAPTATION == "novel_adaptation"

    def test_lookup_by_value(self):
        """Pydantic deserializes enum values from LLM JSON responses via value lookup."""
        assert SourceMaterialType("novel_adaptation") == SourceMaterialType.NOVEL_ADAPTATION
        assert SourceMaterialType("remake") == SourceMaterialType.REMAKE

    def test_invalid_value_raises(self):
        """Unknown string values must raise ValueError."""
        with pytest.raises(ValueError):
            SourceMaterialType("nonexistent")

    def test_source_material_type_id_accessible_as_attribute(self):
        """source_material_type_id is an instance attribute, not just a constructor arg."""
        assert SourceMaterialType.NOVEL_ADAPTATION.source_material_type_id == 1
        assert SourceMaterialType.TV_ADAPTATION.source_material_type_id == 10


class TestSourceMaterialTypeUniqueness:
    def test_no_duplicate_ids(self):
        """No two members share the same source_material_type_id."""
        ids = [m.source_material_type_id for m in SourceMaterialType]
        assert len(ids) == len(set(ids))

    def test_no_duplicate_values(self):
        """No two members share the same string value."""
        values = [m.value for m in SourceMaterialType]
        assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# MetadataType
# ---------------------------------------------------------------------------

class TestMetadataType:
    def test_metadata_type_production_techniques_exists(self):
        """New PRODUCTION_TECHNIQUES member exists with correct string value."""
        assert MetadataType.PRODUCTION_TECHNIQUES == "production_techniques"

    def test_metadata_type_source_material_v2_exists(self):
        """New SOURCE_MATERIAL_V2 member exists with correct string value."""
        assert MetadataType.SOURCE_MATERIAL_V2 == "source_material_v2"

    def test_metadata_type_member_count(self):
        """Exactly 12 members after later metadata additions."""
        assert len(MetadataType) == 12
