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
    PRODUCTION_TECHNIQUES = "production_techniques"
    FRANCHISE = "franchise"
    SOURCE_OF_INSPIRATION = "source_of_inspiration"
    SOURCE_MATERIAL_V2 = "source_material_v2"
    CONCEPT_TAGS = "concept_tags"


class AwardOutcome(str, Enum):
    """Award nomination outcome with a stable integer ID for Postgres storage."""
    outcome_id: int

    def __new__(cls, value: str, outcome_id: int) -> "AwardOutcome":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.outcome_id = outcome_id
        return obj

    WINNER = ("winner", 1)
    NOMINEE = ("nominee", 2)


# Twelve major award ceremonies tracked in IMDB scraping.
# Each member's value is the IMDB GraphQL `event.text` string;
# ceremony_id is a stable SMALLINT for Postgres storage.
class AwardCeremony(str, Enum):
    ceremony_id: int

    def __new__(cls, value: str, ceremony_id: int) -> "AwardCeremony":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.ceremony_id = ceremony_id
        return obj

    ACADEMY_AWARDS = ("Academy Awards, USA", 1)
    GOLDEN_GLOBES  = ("Golden Globes, USA", 2)
    BAFTA          = ("BAFTA Awards", 3)
    CANNES         = ("Cannes Film Festival", 4)
    VENICE         = ("Venice Film Festival", 5)
    BERLIN         = ("Berlin International Film Festival", 6)
    SAG            = ("Actor Awards", 7)
    CRITICS_CHOICE = ("Critics Choice Awards", 8)
    SUNDANCE       = ("Sundance Film Festival", 9)
    RAZZIE         = ("Razzie Awards", 10)
    SPIRIT_AWARDS  = ("Film Independent Spirit Awards", 11)
    GOTHAM         = ("Gotham Awards", 12)


# O(1) lookup from IMDB event.text string to AwardCeremony member.
CEREMONY_BY_EVENT_TEXT: dict[str, AwardCeremony] = {c.value: c for c in AwardCeremony}


class BoxOfficeStatus(StrEnum):
    HIT = "hit"
    FLOP = "flop"


# Narrative-position axis of franchise classification. Mutually
# exclusive: a film carries at most one value, or null. Null covers
# first-entry and standalone films. Orthogonal to the is_crossover
# and is_spinoff booleans on FranchiseOutput; may populate even when
# FranchiseOutput.lineage is null (pair-remakes like Scarface 1983).
#
# Comment above the class so it is NOT sent to the LLM as part of
# the JSON schema description — the system prompt carries the
# definitional text.
class LineagePosition(str, Enum):
    lineage_position_id: int

    def __new__(cls, value: str, lineage_position_id: int) -> "LineagePosition":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.lineage_position_id = lineage_position_id
        return obj

    SEQUEL = ("sequel", 1)
    PREQUEL = ("prequel", 2)
    # REMAKE is retained in the enum for classification fidelity but
    # is NOT consumed at search time — film-to-film retellings are
    # covered by source_of_inspiration, which handles the cross-medium
    # adaptation case more uniformly. Removing the value outright
    # would push borderline cases (Scarface 1983 / 1932) into
    # misleading alternative labels. Keep writing it; don't read it
    # in the search path. See search_improvement_planning/
    # franchise_test_iterations.md (v5).
    REMAKE = ("remake", 3)
    REBOOT = ("reboot", 4)


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


# Narrative structure tags (IDs 1-9): structural choices in how the
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
    CLIFFHANGER_ENDING    = ("cliffhanger_ending", 9)


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

    FEMALE_LEAD        = ("female_lead", 31)
    ENSEMBLE_CAST      = ("ensemble_cast", 32)
    ANTI_HERO          = ("anti_hero", 33)


# Ending tags (IDs 41-43): strong deal-breakers based on how the movie ends.
class EndingTag(str, Enum):
    concept_tag_id: int

    def __new__(cls, value: str, concept_tag_id: int) -> "EndingTag":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.concept_tag_id = concept_tag_id
        return obj

    HAPPY_ENDING       = ("happy_ending", 41)
    SAD_ENDING         = ("sad_ending", 42)
    BITTERSWEET_ENDING = ("bittersweet_ending", 43)
    # Classification-only value: none of the above ending tags apply.
    # Filtered out before storage — never appears in concept_tag_ids.
    NO_CLEAR_CHOICE    = ("no_clear_choice", -1)


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


# All concept tags as a flat tuple, excluding classification-only
# values (NO_CLEAR_CHOICE) that are never stored or searched.
ALL_CONCEPT_TAGS: tuple = tuple(
    tag for tag in (
        *NarrativeStructureTag,
        *PlotArchetypeTag,
        *SettingTag,
        *CharacterTag,
        *EndingTag,
        *ExperientialTag,
        *ContentFlagTag,
    )
    if tag.concept_tag_id >= 0
)
