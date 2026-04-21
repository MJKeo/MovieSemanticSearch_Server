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


class BudgetSize(StrEnum):
    SMALL = "small"
    LARGE = "large"


# Popularity scoring direction for the metadata endpoint.
class PopularityMode(StrEnum):
    POPULAR = "popular"
    NICHE = "niche"


# Reception scoring direction for the metadata endpoint.
class ReceptionMode(StrEnum):
    WELL_RECEIVED = "well_received"
    POORLY_RECEIVED = "poorly_received"


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
    # docs/modules/ingestion.md franchise section for rationale.
    REMAKE = ("remake", 3)
    REBOOT = ("reboot", 4)


# Step 3 franchise endpoint structural flags. Stored as plain strings
# in the structured-output schema so the LLM can emit either or both
# when a single concept combines them.
class FranchiseStructuralFlag(StrEnum):
    SPINOFF = "spinoff"
    CROSSOVER = "crossover"


# Step 3 franchise endpoint launcher scope. Mutually exclusive: a
# query can ask for movies that launched a franchise OR launched a
# subgroup, or neither.
class FranchiseLaunchScope(StrEnum):
    FRANCHISE = "franchise"
    SUBGROUP = "subgroup"


# Source material classification for movies.
#
# Each member carries a string value (for Pydantic JSON schema enum
# constraints in LLM structured output) and a stable integer ID (for
# future movie_card.source_material_type_ids GIN-indexed storage).
#
# See docs/modules/ingestion.md (SourceMaterialType section) for
# boundary notes and deliberate exclusions.
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
# See search_improvement_planning/ingestion.md for generation design
# rationale, and new_system_brainstorm.md for search routing tables.
# ---------------------------------------------------------------------------


# Narrative structure tags (IDs 1-9): structural choices in how the
# story is told.
class NarrativeStructureTag(str, Enum):
    concept_tag_id: int
    description: str

    def __new__(cls, value: str, concept_tag_id: int, description: str) -> "NarrativeStructureTag":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.concept_tag_id = concept_tag_id
        obj.description = description
        return obj

    PLOT_TWIST            = ("plot_twist", 1, "Has a significant plot twist or surprise revelation that recontextualizes part or all of the story, including mid-story reveals and identity twists.")
    TWIST_VILLAIN         = ("twist_villain", 2, "A character revealed as the villain is a surprise — the villain's identity itself is the twist.")
    TIME_LOOP             = ("time_loop", 3, "Characters relive the same time period repeatedly. Distinct from time travel.")
    NONLINEAR_TIMELINE    = ("nonlinear_timeline", 4, "Story is told out of chronological order as a deliberate, defining structural choice — not just a brief flashback.")
    UNRELIABLE_NARRATOR   = ("unreliable_narrator", 5, "The narrator or POV character's account is revealed as untrustworthy.")
    OPEN_ENDING           = ("open_ending", 6, "The story deliberately leaves its central question unresolved or ambiguous.")
    SINGLE_LOCATION       = ("single_location", 7, "Nearly all action takes place in one location (bottle movie). The constraint is a defining feature.")
    BREAKING_FOURTH_WALL  = ("breaking_fourth_wall", 8, "Characters directly address the audience or acknowledge they are in a movie as a notable, deliberate choice.")
    CLIFFHANGER_ENDING    = ("cliffhanger_ending", 9, "The story ends on a major unresolved moment designed to leave the audience in suspense for a sequel or continuation.")


# Plot archetype tags (IDs 11-14): the central premise or driving force.
class PlotArchetypeTag(str, Enum):
    concept_tag_id: int
    description: str

    def __new__(cls, value: str, concept_tag_id: int, description: str) -> "PlotArchetypeTag":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.concept_tag_id = concept_tag_id
        obj.description = description
        return obj

    REVENGE    = ("revenge", 11, "The central plot is driven by a character seeking vengeance. Revenge is the primary narrative engine, not a subplot.")
    UNDERDOG   = ("underdog", 12, "Protagonist is clearly outmatched and overcomes the odds. The power imbalance is a defining feature of the story.")
    KIDNAPPING = ("kidnapping", 13, "The plot centers on a kidnapping or abduction as a central story element, not just one event among many.")
    CON_ARTIST = ("con_artist", 14, "Protagonist is a con artist, grifter, or scammer — the movie is about deception and manipulation as a craft. Distinct from heist/robbery.")


# Setting tags (IDs 21-23): settings users search for as the primary filter.
class SettingTag(str, Enum):
    concept_tag_id: int
    description: str

    def __new__(cls, value: str, concept_tag_id: int, description: str) -> "SettingTag":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.concept_tag_id = concept_tag_id
        obj.description = description
        return obj

    POST_APOCALYPTIC = ("post_apocalyptic", 21, "Set after civilization's collapse. Distinct from dystopian (society intact but oppressive).")
    HAUNTED_LOCATION = ("haunted_location", 22, "Set in or centered around a haunted house, building, or specific location. Narrower than supernatural horror.")
    SMALL_TOWN       = ("small_town", 23, "The small-town setting is central to the story's identity and atmosphere, not just incidental.")


# Character tags (IDs 31-33).
class CharacterTag(str, Enum):
    concept_tag_id: int
    description: str

    def __new__(cls, value: str, concept_tag_id: int, description: str) -> "CharacterTag":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.concept_tag_id = concept_tag_id
        obj.description = description
        return obj

    FEMALE_LEAD        = ("female_lead", 31, "The single protagonist is female, or in a two-hander one co-lead is female. Ensemble casts never qualify.")
    ENSEMBLE_CAST      = ("ensemble_cast", 32, "No single protagonist — multiple characters share roughly equal narrative weight.")
    ANTI_HERO          = ("anti_hero", 33, "Protagonist is morally ambiguous, operates outside conventional morality, or lacks traditional heroic qualities as a defining trait.")


# Ending tags (IDs 41-43): strong deal-breakers based on how the movie ends.
class EndingTag(str, Enum):
    concept_tag_id: int
    description: str

    def __new__(cls, value: str, concept_tag_id: int, description: str) -> "EndingTag":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.concept_tag_id = concept_tag_id
        obj.description = description
        return obj

    HAPPY_ENDING       = ("happy_ending", 41, "Things work out for the protagonists. The overall resolution is positive or optimistic.")
    SAD_ENDING         = ("sad_ending", 42, "The story ends predominantly sad or negatively for the protagonists — loss, failure, or death. Not just bittersweet.")
    BITTERSWEET_ENDING = ("bittersweet_ending", 43, "The ending mixes positive and negative elements — some things work out, others don't. Neither purely happy nor purely sad.")
    # Classification-only value: none of the above ending tags apply.
    # Filtered out before storage — never appears in concept_tag_ids.
    NO_CLEAR_CHOICE    = ("no_clear_choice", -1, "None of the above ending tags apply — the evidence is ambiguous, insufficient, or the ending does not fit happy/sad/bittersweet.")


# Experiential tags (IDs 51-52): binary experiential qualities.
class ExperientialTag(str, Enum):
    concept_tag_id: int
    description: str

    def __new__(cls, value: str, concept_tag_id: int, description: str) -> "ExperientialTag":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.concept_tag_id = concept_tag_id
        obj.description = description
        return obj

    FEEL_GOOD  = ("feel_good", 51, "The overall effect of watching the movie is uplifting and positive — the trajectory and ending leave the viewer feeling good.")
    TEARJERKER = ("tearjerker", 52, "The movie is designed to make you cry and audiences report that it does. Based on emotional impact, not just sad plot events.")


# Content flag tags (ID 61): avoidance deal-breakers.
class ContentFlagTag(str, Enum):
    concept_tag_id: int
    description: str

    def __new__(cls, value: str, concept_tag_id: int, description: str) -> "ContentFlagTag":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.concept_tag_id = concept_tag_id
        obj.description = description
        return obj

    ANIMAL_DEATH = ("animal_death", 61, "An animal dies on screen or as a significant plot point.")


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


# ---------------------------------------------------------------------------
# Search V2 flow routing.
# ---------------------------------------------------------------------------

# The three major search flows that step 1 can route a query into.
# See search_improvement_planning/finalized_search_proposal.md for
# definitions and routing rules.
class SearchFlow(StrEnum):
    EXACT_TITLE = "exact_title"
    SIMILARITY = "similarity"
    STANDARD = "standard"


# Step 1 ambiguity level. This is a compact branching signal, not a
# confidence score.
class QueryAmbiguityLevel(StrEnum):
    CLEAR = "clear"
    MODERATE = "moderate"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Search V2 query understanding (step 2).
# ---------------------------------------------------------------------------

# Which retrieval endpoint handles a dealbreaker or preference.
# Each endpoint has its own step 3 LLM (or deterministic function)
# that translates the description into a query specification.
# See search_improvement_planning/finalized_search_proposal.md
# (Step 3: Query Translation) for endpoint definitions.
class EndpointRoute(StrEnum):
    ENTITY = "entity"
    STUDIO = "studio"
    METADATA = "metadata"
    AWARDS = "awards"
    FRANCHISE_STRUCTURE = "franchise_structure"
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    TRENDING = "trending"


# Whether a dealbreaker is an inclusion (generates candidates,
# contributes to tier count) or exclusion (filters/penalizes
# candidates after assembly, does not count toward tier).
class DealbreakDirection(StrEnum):
    INCLUSION = "inclusion"
    EXCLUSION = "exclusion"


# System-level prior for quality and notability dimensions.
# Shared enum with independent semantics per dimension:
#   enhanced  — explicitly important in the query
#   standard  — implicit default expectation
#   inverted  — user wants the opposite (campy/bad, hidden/obscure)
#   suppressed — a dominant primary preference pushes this prior
#                to the background (second-order inference)
class SystemPrior(StrEnum):
    ENHANCED = "enhanced"
    STANDARD = "standard"
    INVERTED = "inverted"
    SUPPRESSED = "suppressed"


# ---------------------------------------------------------------------------
# Search V2 entity endpoint (step 3).
# ---------------------------------------------------------------------------

# What kind of entity the lookup targets. Determines which posting
# table(s) are searched and which type-specific fields are populated
# in the EntityQuerySpec output.
class EntityType(StrEnum):
    PERSON = "person"
    CHARACTER = "character"
    TITLE_PATTERN = "title_pattern"


# Which role table(s) to search for a person entity. Specific roles
# search a single posting table; broad_person searches all 5 tables
# with cross-posting score consolidation via primary_category.
class PersonCategory(StrEnum):
    ACTOR = "actor"
    DIRECTOR = "director"
    WRITER = "writer"
    PRODUCER = "producer"
    COMPOSER = "composer"
    BROAD_PERSON = "broad_person"


# Specific-role subset used where broad_person would be invalid.
class SpecificPersonCategory(StrEnum):
    ACTOR = "actor"
    DIRECTOR = "director"
    WRITER = "writer"
    PRODUCER = "producer"
    COMPOSER = "composer"


# How to score actor billing position using the zone-based adaptive
# threshold system. Only meaningful when actor results are in play
# (person_category is actor or broad_person). See finalized search
# proposal "Actor Prominence Scoring" for zone definitions and
# per-mode score tables.
class ActorProminenceMode(StrEnum):
    DEFAULT = "default"
    LEAD = "lead"
    SUPPORTING = "supporting"
    MINOR = "minor"


# How to match a title pattern against movie title strings.
# contains = LIKE '%pattern%', starts_with = LIKE 'pattern%'.
class TitlePatternMatchType(StrEnum):
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"


# ---------------------------------------------------------------------------
# Search V2 awards endpoint (step 3).
# ---------------------------------------------------------------------------

# How the scoring_mark is interpreted when scoring award matches.
# FLOOR: binary — score 1.0 if has_count >= scoring_mark, else 0.0.
# THRESHOLD: gradient — min(has_count, scoring_mark) / scoring_mark.
class AwardScoringMode(StrEnum):
    FLOOR = "floor"
    THRESHOLD = "threshold"


# ---------------------------------------------------------------------------
# Search V2 metadata endpoint (step 3).
# ---------------------------------------------------------------------------

# Which single metadata attribute is the primary target for this
# dealbreaker or preference. The step 3 metadata LLM selects the
# one column that best represents the step 2 description, and
# execution code queries ONLY that column. The LLM may still
# populate other attribute fields in the output for context, but
# only the column identified here drives candidate generation
# (dealbreakers) and scoring (preferences).
#
# This simplifies the execution layer: one metadata item = one
# column query = one [0, 1] score. No within-dealbreaker multi-
# attribute combination logic needed.
class MetadataAttribute(StrEnum):
    RELEASE_DATE = "release_date"
    RUNTIME = "runtime"
    MATURITY_RATING = "maturity_rating"
    STREAMING = "streaming"
    AUDIO_LANGUAGE = "audio_language"
    COUNTRY_OF_ORIGIN = "country_of_origin"
    BUDGET_SCALE = "budget_scale"
    BOX_OFFICE = "box_office"
    POPULARITY = "popularity"
    RECEPTION = "reception"
