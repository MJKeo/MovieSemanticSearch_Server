"""
Shared enums used across multiple modules in the codebase.
"""

from enum import Enum, StrEnum


class MetadataType(StrEnum):
    """Canonical names for all metadata generation types.

    StrEnum so values are plain strings — compatible with SQLite column
    names, custom_id formatting, and anywhere a string is expected.
    """
    PLOT_EVENTS = "plot_events"
    RECEPTION = "reception"
    PLOT_ANALYSIS = "plot_analysis"
    VIEWER_EXPERIENCE = "viewer_experience"
    WATCH_CONTEXT = "watch_context"
    NARRATIVE_TECHNIQUES = "narrative_techniques"
    PRODUCTION_KEYWORDS = "production_keywords"
    SOURCE_OF_INSPIRATION = "source_of_inspiration"
    SOURCE_MATERIAL_V2 = "source_material_v2"
    CONCEPT_TAGS = "concept_tags"


# Source material classification for movies.
#
# Each member carries a string value (for Pydantic JSON schema enum
# constraints in LLM structured output) and a stable integer ID (for
# future movie_card.source_material_type_ids GIN-indexed storage).
#
# See search_improvement_planning/source_material_type_enum.md for full
# definitions, boundary notes, and deliberate exclusions.
class SourceMaterialType(str, Enum):
    source_material_type_id: int

    def __new__(cls, value: str, source_material_type_id: int) -> "SourceMaterialType":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.source_material_type_id = source_material_type_id
        return obj

    NOVEL_ADAPTATION       = ("novel_adaptation", 1)
    SHORT_STORY_ADAPTATION = ("short_story_adaptation", 2)
    STAGE_ADAPTATION       = ("stage_adaptation", 3)
    TRUE_STORY             = ("true_story", 4)
    BIOGRAPHY              = ("biography", 5)
    COMIC_ADAPTATION       = ("comic_adaptation", 6)
    FOLKLORE_ADAPTATION    = ("folklore_adaptation", 7)
    VIDEO_GAME_ADAPTATION  = ("video_game_adaptation", 8)
    REMAKE                 = ("remake", 9)
    TV_ADAPTATION          = ("tv_adaptation", 10)


# ---------------------------------------------------------------------------
# Binary concept tags for deterministic search retrieval.
#
# Split into 7 per-category enums so the JSON schema self-enforces
# category membership — the model cannot produce a tag in the wrong
# category field, eliminating the need for a runtime model_validator.
#
# Each member carries a string value (for Pydantic JSON schema enum
# constraints in LLM structured output) and a stable integer ID (for
# movie_card.concept_tag_ids GIN-indexed storage).
#
# IDs are gapped by category to allow future additions within a category
# without renumbering existing tags.
#
# See search_improvement_planning/concept_tags.md for full definitions,
# classification signals, and deliberate exclusions.
# ---------------------------------------------------------------------------


# Narrative structure tags (IDs 1-8): structural choices in how the
# story is told.
class NarrativeStructureTag(str, Enum):
    concept_tag_id: int

    def __new__(cls, value: str, concept_tag_id: int) -> "NarrativeStructureTag":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.concept_tag_id = concept_tag_id
        return obj

    PLOT_TWIST            = ("plot_twist", 1)
    TWIST_VILLAIN         = ("twist_villain", 2)
    TIME_LOOP             = ("time_loop", 3)
    NONLINEAR_TIMELINE    = ("nonlinear_timeline", 4)
    UNRELIABLE_NARRATOR   = ("unreliable_narrator", 5)
    OPEN_ENDING           = ("open_ending", 6)
    SINGLE_LOCATION       = ("single_location", 7)
    BREAKING_FOURTH_WALL  = ("breaking_fourth_wall", 8)


# Plot archetype tags (IDs 11-14): the central premise or driving force.
class PlotArchetypeTag(str, Enum):
    concept_tag_id: int

    def __new__(cls, value: str, concept_tag_id: int) -> "PlotArchetypeTag":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.concept_tag_id = concept_tag_id
        return obj

    REVENGE    = ("revenge", 11)
    UNDERDOG   = ("underdog", 12)
    KIDNAPPING = ("kidnapping", 13)
    CON_ARTIST = ("con_artist", 14)


# Setting tags (IDs 21-23): settings users search for as the primary filter.
class SettingTag(str, Enum):
    concept_tag_id: int

    def __new__(cls, value: str, concept_tag_id: int) -> "SettingTag":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.concept_tag_id = concept_tag_id
        return obj

    POST_APOCALYPTIC = ("post_apocalyptic", 21)
    HAUNTED_LOCATION = ("haunted_location", 22)
    SMALL_TOWN       = ("small_town", 23)


# Character tags (IDs 31-33).
class CharacterTag(str, Enum):
    concept_tag_id: int

    def __new__(cls, value: str, concept_tag_id: int) -> "CharacterTag":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.concept_tag_id = concept_tag_id
        return obj

    FEMALE_PROTAGONIST = ("female_protagonist", 31)
    ENSEMBLE_CAST      = ("ensemble_cast", 32)
    ANTI_HERO          = ("anti_hero", 33)


# Ending tags (IDs 41-42): strong deal-breakers based on how the movie ends.
class EndingTag(str, Enum):
    concept_tag_id: int

    def __new__(cls, value: str, concept_tag_id: int) -> "EndingTag":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.concept_tag_id = concept_tag_id
        return obj

    HAPPY_ENDING = ("happy_ending", 41)
    SAD_ENDING   = ("sad_ending", 42)


# Experiential tags (IDs 51-52): binary experiential qualities.
class ExperientialTag(str, Enum):
    concept_tag_id: int

    def __new__(cls, value: str, concept_tag_id: int) -> "ExperientialTag":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.concept_tag_id = concept_tag_id
        return obj

    FEEL_GOOD  = ("feel_good", 51)
    TEARJERKER = ("tearjerker", 52)


# Content flag tags (ID 61): avoidance deal-breakers.
class ContentFlagTag(str, Enum):
    concept_tag_id: int

    def __new__(cls, value: str, concept_tag_id: int) -> "ContentFlagTag":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.concept_tag_id = concept_tag_id
        return obj

    ANIMAL_DEATH = ("animal_death", 61)


# All concept tags as a flat tuple, for consumers that need to iterate
# across categories (GIN index population, query routing, etc.).
ALL_CONCEPT_TAGS: tuple = (
    *NarrativeStructureTag,
    *PlotArchetypeTag,
    *SettingTag,
    *CharacterTag,
    *EndingTag,
    *ExperientialTag,
    *ContentFlagTag,
)
