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


# Release format classification, mirrored from IMDB's titleType.id.
# UNKNOWN is the catch-all for IMDB title types outside the supported
# set (tvSeries, videoGame, tvEpisode, etc.) and for missing values; it
# doubles as the column default in movie_card so newly added rows are
# always flagged until a backfill or fresh ingest writes the real value.
# String values match IMDB's titleType.id verbatim so the alias dict
# below is a direct {imdb_string: member} lookup.
class ReleaseFormat(str, Enum):
    release_format_id: int

    def __new__(cls, value: str, release_format_id: int) -> "ReleaseFormat":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.release_format_id = release_format_id
        return obj

    UNKNOWN  = ("unknown", 0)
    MOVIE    = ("movie",   1)
    TV_MOVIE = ("tvMovie", 2)
    SHORT    = ("short",   3)
    VIDEO    = ("video",   4)


# O(1) lookup from IMDB titleType.id string to ReleaseFormat member.
# UNKNOWN is intentionally excluded — it isn't a real IMDB title type,
# only a sentinel for unsupported / missing values.
RELEASE_FORMAT_BY_IMDB_TYPE: dict[str, ReleaseFormat] = {
    m.value: m for m in ReleaseFormat if m is not ReleaseFormat.UNKNOWN
}


def release_format_id_for_imdb_type(imdb_title_type: str | None) -> int:
    """Map an IMDB titleType.id string to its ReleaseFormat int id.

    Returns ReleaseFormat.UNKNOWN.release_format_id (0) for None and
    for any value outside the supported set. UNKNOWN is the audit
    handle for content the search index should not be serving.
    """
    if imdb_title_type is None:
        return ReleaseFormat.UNKNOWN.release_format_id
    member = RELEASE_FORMAT_BY_IMDB_TYPE.get(imdb_title_type)
    if member is None:
        return ReleaseFormat.UNKNOWN.release_format_id
    return member.release_format_id


# ---------------------------------------------------------------------------
# Binary concept tags for deterministic search retrieval.
#
# Single source of truth: every concept tag is a member of the master
# `ConceptTag` enum, and every tag carries its category + selection
# criteria + boundary cases + (optional) long-form instructions as
# attributes. The seven per-category enums (NarrativeStructureTag,
# PlotArchetypeTag, ...) are DYNAMICALLY DERIVED below by filtering on
# the master enum's category attribute, then exposed under the
# original names so existing Pydantic structured-output schemas
# (which type their list fields against e.g. `list[NarrativeStructureTag]`)
# keep working. The dynamic per-category enums carry only
# `concept_tag_id` and `description` (attached after construction) —
# rich content (selection_criteria etc.) lives on the master.
#
# The prompt for the concept_tags generator assembles its tag-
# definitions section by iterating these enums and their attributes —
# see movie_ingestion/metadata_generation/prompts/concept_tags_assembly.py.
#
# IDs are gapped by category to allow future additions within a category
# without renumbering existing tags. NO_CLEAR_CHOICE (id=-1) is a
# classification-only value: the LLM may emit it for endings, but it
# is filtered out before storage and never appears in concept_tag_ids
# arrays in Postgres.
#
# See search_improvement_planning/ingestion.md for generation design
# rationale, and new_system_brainstorm.md for search routing tables.
# ---------------------------------------------------------------------------


# Category-level metadata for concept tags. Carries everything the
# prompt assembler needs to render a category's section: the human-
# readable header, intro subtitle, cardinality ("multi" = a list of
# zero or more, "one_of" = exactly one), the ConceptTagsOutput field
# name, the dynamic per-category enum's class name (for downstream
# imports), and optional category-level prose (cross-tag relationship
# notes shown at the top of the section, plus section-level "HOW TO
# THINK" instructions shown at the end).
class ConceptTagCategory(str, Enum):
    display_label: str
    intro_text: str | None
    cardinality: str
    field_name: str
    enum_class_name: str
    section_instructions: str | None
    cross_tag_note: str | None

    def __new__(
        cls,
        value: str,
        display_label: str,
        intro_text: str | None,
        cardinality: str,
        field_name: str,
        enum_class_name: str,
        section_instructions: str | None = None,
        cross_tag_note: str | None = None,
    ) -> "ConceptTagCategory":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.display_label = display_label
        obj.intro_text = intro_text
        obj.cardinality = cardinality
        obj.field_name = field_name
        obj.enum_class_name = enum_class_name
        obj.section_instructions = section_instructions
        obj.cross_tag_note = cross_tag_note
        return obj

    NARRATIVE_STRUCTURE = (
        "narrative_structure",
        "NARRATIVE STRUCTURE",
        "structural choices in how the story is told",
        "multi",
        "narrative_structure",
        "NarrativeStructureTag",
        None,
        (
            "OPEN_ENDING and CLIFFHANGER_ENDING describe HOW the story ends, "
            "not how it FEELS. They coexist with the emotional ENDINGS tags "
            "below. A cliffhanger that leaves the audience devastated still "
            "gets a SAD_ENDING tag. An open ending that leaves audiences "
            "feeling warm still gets a HAPPY_ENDING tag. Evaluate both this "
            "section and the ENDINGS section independently."
        ),
    )
    PLOT_ARCHETYPES = (
        "plot_archetypes",
        "PLOT ARCHETYPES",
        "the central premise or driving force. A tag applies when the "
        "concept IS the movie, not just an element in the plot",
        "multi",
        "plot_archetypes",
        "PlotArchetypeTag",
    )
    SETTINGS = (
        "settings",
        "SETTINGS",
        "the setting is a defining characteristic of the movie",
        "multi",
        "settings",
        "SettingTag",
    )
    CHARACTERS = (
        "characters",
        "CHARACTERS",
        None,
        "multi",
        "characters",
        "CharacterTag",
    )
    ENDINGS = (
        "endings",
        "ENDINGS",
        "how the audience FEELS when the credits roll. Exactly one tag "
        "per movie (including no_clear_choice)",
        "one_of",
        "endings",
        "EndingTag",
        # section_instructions: the endings HOW-TO block, paraphrased.
        (
            "HOW TO THINK THROUGH THIS CATEGORY — work through these steps "
            "internally before selecting a tag:\n"
            "\n"
            "BASE RATES (apply BEFORE looking at evidence): HAPPY_ENDING is "
            "the empirically dominant tag and the correct default for the "
            "majority of narrative film, including films with substantial "
            "cost-along-the-way. SAD_ENDING is common for tragedies and "
            "downer endings. BITTERSWEET_ENDING is UNCOMMON — most films "
            "with mixed elements still land clearly on HAPPY or SAD. "
            "NO_CLEAR_CHOICE is reserved for genuinely contemplative or "
            "existential endings that fit none of the three. Do not treat "
            "the four tags as equal-probability options.\n"
            "\n"
            "1. The CLOSING SCENE test (PRIMARY): Identify the literal "
            "final scene before credits from plot_summary. If it is a "
            "recognized celebration beat — a triumphant kiss, a family "
            "reunion hug, a threat-defeated cheer, a platform-raise, a "
            "mountaintop / sunrise / restored shot — tag HAPPY_ENDING "
            "regardless of what was lost during the runtime. If it is a "
            "beat of loss, defeat, or grief (a funeral, a destroyed home, "
            "a protagonist alone in ruin) with no upswing → SAD_ENDING. "
            "Bittersweet endings tend to close on a quiet, contemplative "
            "beat (an unspoken moment, a long look, a what-might-have-been "
            "montage) where joy and sorrow sit together unresolved.\n"
            "\n"
            "2. emotional_observations (whole-movie audience reactions): "
            "Filter for language about how audiences felt AS THE FILM "
            "ENDS, not journey-level emotion. Map:\n"
            "   - \"uplifting\", \"satisfying\", \"triumphant\", \"warm closure\", \"feel-good\", \"earned\", \"hard-won\", \"achievement at a cost\", \"sacrifice rewarded\" → HAPPY_ENDING. Cost-and-victory framing is HAPPY, not bittersweet.\n"
            "   - \"devastating\", \"tragic\", \"heartbreaking\", \"bleak finale\", \"crushing\", \"unrelenting loss\" → SAD_ENDING.\n"
            "   - explicit \"audiences leave with mixed feelings they cannot resolve\", \"a knot despite the win\", \"genuinely torn\", \"unable to celebrate fully\" → BITTERSWEET_ENDING. The bar is HIGH: audiences must affirmatively say they cannot land on HAPPY OR SAD.\n"
            "   - \"ambiguous\", \"contemplative\", \"open to interpretation\", existential / philosophical reactions without clear valence → NO_CLEAR_CHOICE.\n"
            "Filter out journey-level emotions (\"tense\", \"frightening\", \"dark\") that describe the runtime experience but not the ending.\n"
            "\n"
            "3. Final state of affairs from plot_summary: where do the "
            "characters stand at the close — what has been gained, lost, "
            "or left unresolved? Treat this as factual evidence to be "
            "reconciled with the closing-scene beat, NOT as a direct "
            "emotion verdict. A protagonist who survived a horror movie "
            "with the threat defeated is HAPPY even though they suffered.\n"
            "\n"
            "4. Ending-related plot_keywords: any keywords that directly "
            "signal ending type (\"happy ending\", \"tragic ending\", "
            "\"twist ending\").\n"
            "\n"
            "DEFAULT RULES: When the evidence is ambiguous between HAPPY "
            "and BITTERSWEET, default to HAPPY — bittersweet has a higher "
            "bar and requires that the audience genuinely cannot decide "
            "between HAPPY and SAD. When the evidence is ambiguous between "
            "SAD and BITTERSWEET, default to SAD for the same reason. Pick "
            "BITTERSWEET only when you can affirmatively defend that "
            "NEITHER HAPPY NOR SAD would be a reasonable alternative.\n"
            "\n"
            "Structural ambiguity about WHAT happened (an ending where the "
            "reality of events is left open) is NOT the same as emotional "
            "ambiguity about HOW the audience felt — if the closing scene "
            "is warm/positive, tag HAPPY_ENDING even when the plot leaves "
            "a structural question open."
        ),
        # cross_tag_note: structural-vs-emotional independence note.
        (
            "The ending tag captures the AUDIENCE'S emotional experience at the "
            "end of the film — not a factual ledger of what went right vs wrong "
            "in the plot. A protagonist who dies saving others may leave "
            "audiences feeling devastated (sad), triumphant (happy), or "
            "genuinely torn (bittersweet) — the plot outcome alone does not "
            "determine the tag. These tags are independent of OPEN_ENDING and "
            "CLIFFHANGER_ENDING above (which describe narrative structure, not "
            "emotion). Always evaluate this section even when you tagged "
            "OPEN_ENDING or CLIFFHANGER_ENDING."
        ),
    )
    EXPERIENTIAL = (
        "experiential",
        "EXPERIENTIAL",
        "binary deal-breaker qualities",
        "multi",
        "experiential",
        "ExperientialTag",
    )
    CONTENT_FLAGS = (
        "content_flags",
        "CONTENT FLAGS",
        "things users specifically search to AVOID",
        "multi",
        "content_flags",
        "ContentFlagTag",
    )


# Master concept tag enum — single source of truth for tag metadata.
# Every member carries:
#   - value: the slug string (e.g. "plot_twist") used in JSON I/O
#   - concept_tag_id: stable integer for Postgres storage (gapped by category)
#   - category: which ConceptTagCategory this tag belongs to
#   - description: one-line definition shown in the prompt
#   - selection_criteria: paraphrased "Check:" content — which inputs to
#     consult and what to look for when deciding whether to include this tag
#   - boundary_cases: paraphrased "NOT <tag>:" content — generalized
#     descriptions of patterns that look like the tag but are NOT it.
#     Deliberately written without naming specific movies to avoid
#     polluting evaluation results when the test set overlaps with examples.
#   - long_form_instructions: optional extended reasoning block, used only
#     for tags with stricter defaults that need step-by-step logic
#     (currently FEMALE_LEAD only).
#
# When editing this enum, slug values and concept_tag_ids MUST stay
# byte-identical to prior versions — Postgres rows and tracker.db JSON
# reference them. Adding new members is safe within a category's gap.
class ConceptTag(str, Enum):
    concept_tag_id: int
    category: ConceptTagCategory
    description: str
    selection_criteria: str
    boundary_cases: str
    long_form_instructions: str | None

    def __new__(
        cls,
        value: str,
        concept_tag_id: int,
        category: ConceptTagCategory,
        description: str,
        selection_criteria: str,
        boundary_cases: str,
        long_form_instructions: str | None = None,
    ) -> "ConceptTag":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.concept_tag_id = concept_tag_id
        obj.category = category
        obj.description = description
        obj.selection_criteria = selection_criteria
        obj.boundary_cases = boundary_cases
        obj.long_form_instructions = long_form_instructions
        return obj

    # -- Narrative Structure (IDs 1-9) -----------------------------------

    PLOT_TWIST = (
        "plot_twist",
        1,
        ConceptTagCategory.NARRATIVE_STRUCTURE,
        "A surprise revelation recontextualizes events the audience has "
        "already seen. The audience must have formed one understanding that "
        "the reveal overturns.",
        "Look at information_control narrative_technique_terms for twist / "
        "reversal / hidden-truth labels; plot_keywords for \"surprise ending\" "
        "or \"plot twist\"; plot_summary for explicit late reveals that reframe "
        "prior scenes; craft_observations where reviewers describe twists, "
        "rug-pulls, or third-act reveals.",
        "Any surprise or unexpected event without recontextualization of prior "
        "scenes; a late-act betrayal that the audience could see coming from "
        "setup; new information that adds to the story but does not change "
        "prior understanding; tragic irony at the ending where nothing earlier "
        "in the film is reinterpreted; sequel-setup reveals that introduce new "
        "questions rather than overturning past ones; structural ambiguity at "
        "the end of the film about what is real (that is OPEN_ENDING, not a "
        "recontextualizing twist).",
    )

    TWIST_VILLAIN = (
        "twist_villain",
        2,
        ConceptTagCategory.NARRATIVE_STRUCTURE,
        "A character is presented to the audience as good, trustworthy, "
        "neutral, or allied to the protagonist for a substantial portion of "
        "the film, and is then revealed late to be the (or a) primary "
        "antagonist. The moral category flip — good/ally to villain — is "
        "the surprise; the audience's earlier belief in the character's "
        "side is what gets overturned. Auto-implies PLOT_TWIST.",
        "Look at information_control terms describing a hidden antagonist, "
        "false ally, or betrayal reveal; plot_summary phrasings indicating "
        "a character was revealed to have been the villain all along, was "
        "secretly orchestrating events the protagonist was working against, "
        "or was a hidden mastermind behind the conflict; craft_observations "
        "where reviewers describe the antagonist reveal as the craft moment. "
        "The signal must indicate that the audience's prior read of the "
        "character was wrong about their alignment, not just about their "
        "motives.",
        "A character that the audience knew from the start was a villain, "
        "even if the depth or true scope of their evil is revealed later "
        "(motivation depth is not category flip); a villain whose plan is "
        "more elaborate or surprising than expected, but whose villainy "
        "itself was never in doubt; a character who is suspicious or "
        "sinister-coded from the first act, even if confirmation comes "
        "late; a protagonist's psychological alter-ego, split personality, "
        "or hallucination revealed late (closer to UNRELIABLE_NARRATOR than "
        "to category-flip villainy); a known antagonist whose true methods "
        "are revealed later. The audience must have genuinely believed the "
        "character was on the protagonist's side, neutral, or good — "
        "surprising-evil-motivation alone is not enough.",
    )

    TIME_LOOP = (
        "time_loop",
        3,
        ConceptTagCategory.NARRATIVE_STRUCTURE,
        "Characters relive the same time period repeatedly as the central "
        "premise. Distinct from time travel.",
        "Look at narrative_delivery terms for \"time loop\" or \"reliving\"; "
        "plot_keywords for \"time loop\" directly; plot_summary for explicit "
        "re-living language such as a character waking to the same day "
        "repeatedly.",
        "Time travel where the protagonist visits different periods (distinct "
        "concept); a single repeated scene used as a flashback; dream "
        "sequences that recur without claiming temporal repetition; cyclical "
        "themes or visual repetition without literal relived time.",
    )

    NONLINEAR_TIMELINE = (
        "nonlinear_timeline",
        4,
        ConceptTagCategory.NARRATIVE_STRUCTURE,
        "Non-chronological structure is a defining identity of the film. The "
        "audience reconstructs the timeline from deliberately scrambled "
        "pieces.",
        "Look at narrative_delivery terms for \"nonlinear\" or \"fragmented "
        "timeline\"; plot_keywords for \"nonlinear timeline\"; plot_summary "
        "structure showing deliberate chapter-shuffling; craft_observations "
        "where reviewers describe the film as \"told in chapters\", \"moves "
        "between timelines\", or \"non-linearly structured\".",
        "Occasional flashbacks within an otherwise chronological main "
        "narrative; a single framing device or prologue placed out of order; "
        "a flash-forward cold open with the remainder of the film "
        "chronological; epistolary or anthology structure that nonetheless "
        "proceeds chronologically within and across segments.",
    )

    UNRELIABLE_NARRATOR = (
        "unreliable_narrator",
        5,
        ConceptTagCategory.NARRATIVE_STRUCTURE,
        "The narrator or POV character's account is later revealed to the "
        "AUDIENCE as distorted or fabricated. Trust with the audience is "
        "broken — not just between characters.",
        "Look at pov_perspective terms for \"unreliable narrator\" or "
        "\"subjective POV\"; plot_summary for explicit revelation that prior "
        "shown material was distorted; craft_observations where reviewers "
        "flag unreliable narration as a craft choice.",
        "A character who lies to other characters but where the audience sees "
        "the truth (that is deception, not unreliable narration); a character "
        "who hallucinates unless the film presents the hallucinations as "
        "reality to the audience and then reveals the distortion; flashbacks "
        "shown from a character's biased perspective without an in-film "
        "reveal that those flashbacks were wrong.",
    )

    OPEN_ENDING = (
        "open_ending",
        6,
        ConceptTagCategory.NARRATIVE_STRUCTURE,
        "The story completes its narrative arc but deliberately leaves its "
        "CENTRAL THEMATIC QUESTION — what the movie was about — ambiguous. "
        "Discriminating question: did the film resolve the main conflict "
        "it posed about its protagonists? If YES, do not tag OPEN_ENDING "
        "regardless of any lingering side details or franchise hooks. "
        "Audiences should be debating what the film MEANS, not what comes "
        "next.",
        "Look at plot_keywords for \"ambiguous ending\"; plot_summary for a "
        "final beat that intentionally avoids resolution of the central "
        "question; emotional_observations for \"ambiguous\", \"lingering "
        "question\", \"audiences debate\"; craft_observations for "
        "reviewer descriptions of intentional ambiguity at the close. "
        "Apply the resolution test: name the central conflict the film "
        "posed in act one (protagonist's goal, central mystery, central "
        "relationship), then ask whether the film resolved it. Only tag "
        "OPEN_ENDING when the answer is genuinely ambiguous.",
        "A sequel setup with an unresolved main conflict (that is "
        "CLIFFHANGER_ENDING, not OPEN); a franchise-hook coda, post-credits "
        "tease, or universe-continues epilogue — if the central conflict "
        "of THIS film is resolved, the tag does not apply regardless of "
        "whether the franchise continues; a horror ending where the threat "
        "is contained but evil-still-exists framing remains (the central "
        "haunting/possession/menace was resolved); unanswered side "
        "questions when the central conflict has been resolved; an ending "
        "that is emotionally unsatisfying but narratively clear; an ending "
        "where the protagonist's fate is uncertain in detail but the "
        "film's main question has been answered.",
    )

    SINGLE_LOCATION = (
        "single_location",
        7,
        ConceptTagCategory.NARRATIVE_STRUCTURE,
        "Nearly all action takes place in one physical location. The spatial "
        "constraint is a defining feature of the film's identity.",
        "Look at plot_summary to count distinct locations and assess whether "
        "the spatial constraint is the film's identity; identity-level "
        "descriptors such as \"bottle movie\" or \"one-room drama\" in "
        "craft_observations or related fields.",
        "A film set mostly in one building but with significant scenes "
        "elsewhere; a film that uses one location heavily but also includes "
        "another substantial setting; a haunted-location premise where "
        "characters also travel elsewhere; episodic films that revisit the "
        "same location across acts without committing to it as the entire "
        "setting.",
    )

    BREAKING_FOURTH_WALL = (
        "breaking_fourth_wall",
        8,
        ConceptTagCategory.NARRATIVE_STRUCTURE,
        "Characters directly address the audience or acknowledge they are in "
        "a movie. A notable, deliberate, recurring stylistic choice.",
        "Look at additional_narrative_devices terms for \"fourth wall break\" "
        "or \"direct address\"; plot_keywords for \"breaking the fourth wall\"; "
        "craft_observations where reviewers single out direct camera address "
        "as a craft choice.",
        "Voiceover narration where the character tells their story without "
        "acknowledging the audience or the camera; documentary-style "
        "interviews; songs that comment on the action unless characters "
        "explicitly address viewers; a single brief in-joke aside that is "
        "not a recurring stylistic device.",
    )

    CLIFFHANGER_ENDING = (
        "cliffhanger_ending",
        9,
        ConceptTagCategory.NARRATIVE_STRUCTURE,
        "The central conflict is INTENTIONALLY unresolved at credits as "
        "deliberate setup for a planned followup film — the story stops "
        "mid-arc on purpose to leave the audience on the edge of their "
        "seat between installments. Cliffhanger is about intentionality: "
        "the film knowingly defers resolution to a next chapter that "
        "audiences are expected to wait for. Distinct from OPEN_ENDING "
        "(completed arc with thematic ambiguity) and from films that "
        "happen to be part of a series but stand as complete arcs.",
        "Look at plot_summary for a closing beat where the central "
        "conflict is left unresolved AND continuation is signaled (an "
        "antagonist mid-victory, a protagonist mid-action, an explicit "
        "to-be-continued beat); plot_keywords for explicit cliffhanger "
        "framings; emotional_observations and craft_observations / "
        "reviewer commentary for how audiences and reviewers DESCRIBED "
        "the ending — language about being left hanging, demanding the "
        "next installment, deliberate sequel setup, or being left on "
        "the edge of one's seat between films; release context where "
        "the film is known to be one of a planned series AND reviewers "
        "frame the ending as deliberate setup for that next "
        "installment. The intentionality must be evident: the film must "
        "have chosen to stop mid-arc as part of a planned continuation.",
        "A satisfying resolution where a villain happens to survive or a "
        "sequel later becomes possible (resolution is still achieved); "
        "central conflict resolved even when side threads remain open; "
        "thematic ambiguity at the end without an unresolved plot "
        "question (that is OPEN_ENDING) — an ambiguous ending where the "
        "audience debates what HAPPENED but the film itself does not "
        "promise a continuation that answers it is NOT a cliffhanger; a "
        "film that is part of a series but where the individual film "
        "stands as a complete arc — being one of a series alone does "
        "NOT qualify; a sequel that exists only because the first film "
        "was successful but where the first film was written as "
        "standalone; plot holes, structural ambiguity, or open "
        "interpretations that are not deliberate sequel setup.",
    )

    # -- Plot Archetypes (IDs 11-14) -------------------------------------

    REVENGE = (
        "revenge",
        11,
        ConceptTagCategory.PLOT_ARCHETYPES,
        "Vengeance is the primary narrative engine. The protagonist's central "
        "goal throughout the film is to get revenge.",
        "Look at plot_keywords for \"revenge\" directly; conflict_type for "
        "vengeance-driven framings; plot_summary where the protagonist's "
        "stated goal involves \"make them pay\"; elevator_pitch describing "
        "the central plot engine; thematic_concepts naming \"vengeance\" or "
        "\"retribution\" as central themes.",
        "A rescue mission motivated by anger or loss (the goal is rescue, not "
        "vengeance); a character seeking justice through legal means; a "
        "subplot of retaliation within a different main plot; a single act "
        "of revenge as the inciting incident for a story that then becomes "
        "about something else.",
    )

    UNDERDOG = (
        "underdog",
        12,
        ConceptTagCategory.PLOT_ARCHETYPES,
        "A narrative where the protagonist — or the side or group they "
        "belong to — is structurally weaker than the force they face "
        "(in resources, numbers, status, ability, social power, or "
        "perceived prospects), AND the central dramatic question the "
        "film is built around is whether they can prevail against that "
        "disadvantage. The improbable-victory question must be what the "
        "story is asking. Setting-level asymmetry qualifies when the "
        "protagonist belongs to the disadvantaged side and the film's "
        "core arc is their improbable rise; setting-level asymmetry does "
        "NOT qualify when the film is using that asymmetry as backdrop "
        "for a different central question (escape, survival, "
        "investigation, relationship, revelation).",
        "Look at narrative_archetype terms describing an outmatched "
        "protagonist rising, an against-the-odds story, or an improbable "
        "triumph; plot_keywords naming the protagonist or their faction "
        "as outclassed, outgunned, outnumbered, or improbable; "
        "plot_summary framing where the structural disadvantage is "
        "explicitly cited as the central dramatic stake (the weaker side "
        "named as such and the film's tension hinging on whether they "
        "can prevail); thematic_concepts describing rise-from-disadvantage "
        "or David-vs-Goliath dynamics; conflict_type and "
        "conflict_stakes_design framings that place asymmetric power at "
        "the center of the conflict as the driving question. The "
        "asymmetry must align with the film's central question being "
        "'can the weaker side prevail?' — power imbalance present in the "
        "world without being the film's central question does not "
        "qualify.",
        "A protagonist who faces a stronger adversary but whose competence "
        "makes the outcome merely uncertain rather than improbable; a "
        "story where the protagonist confronts a more powerful threat "
        "but the central dramatic question is escape, survival, "
        "revelation, recovery, or revenge — not 'can the weaker side "
        "rise and win?'; a power asymmetry that exists as world-building "
        "or backdrop but where the film's main story is something else "
        "(an investigation, a relationship, a horror escape, an identity "
        "reveal); a lone dissenter in an intellectual disagreement; any "
        "conflict where one side has more power, unless the film frames "
        "the improbable-victory question as the central story it is "
        "telling.",
    )

    KIDNAPPING = (
        "kidnapping",
        13,
        ConceptTagCategory.PLOT_ARCHETYPES,
        "A kidnapping or abduction IS the central plot. The movie is about "
        "the abduction event itself and its direct consequences (rescue, "
        "escape, ransom).",
        "Look at plot_keywords for \"kidnapping\" or \"abduction\"; "
        "plot_summary showing the kidnapping as both the inciting incident "
        "AND an ongoing plot driver; parental_guide_items category "
        "\"Abduction\" or \"Kidnapping\" at non-trivial severity "
        "(corroborating, but only tag when plot_summary also centers the "
        "abduction as the plot's engine).",
        "Imprisonment as backstory motivating a different main plot (e.g. "
        "revenge); captives serving as the premise for a different central "
        "plot (a chase or escape that is itself the story) — the abduction "
        "is incidental scaffolding; supernatural capture or imprisonment by "
        "non-human forces; a brief capture that is one event among many in a "
        "larger story; a kidnapping that occurs but is resolved early as "
        "setup for a different central engine.",
    )

    CON_ARTIST = (
        "con_artist",
        14,
        ConceptTagCategory.PLOT_ARCHETYPES,
        "The protagonist is a con artist, grifter, or scammer. The movie is "
        "about deception as a craft. Distinct from heist (theft/robbery).",
        "Look at plot_keywords for \"con artist\" or \"grifter\"; "
        "plot_summary showing a deception-driven plot with con artistry as "
        "the protagonist's mode; thematic_concepts naming \"deception as "
        "craft\" or \"art of the con\" as central themes.",
        "A character who lies or manipulates for personal revenge or "
        "survival; a villain who deceives others while the protagonist plays "
        "a different role; a character who runs a single con as part of a "
        "larger non-con plot; a heist or theft story where the crime is "
        "property-taking rather than identity-deception.",
    )

    # -- Settings (IDs 21-23) --------------------------------------------

    POST_APOCALYPTIC = (
        "post_apocalyptic",
        21,
        ConceptTagCategory.SETTINGS,
        "Set after civilization's collapse. Society has fallen.",
        "Look at plot_keywords for \"post apocalypse\"; plot_summary "
        "establishing a collapsed-civilization setting as the world the "
        "story operates in.",
        "Dystopian settings where society is intact but oppressive (distinct "
        "concept); science fiction set on other planets or in space that did "
        "not arise from Earth's collapse; a localized disaster that has not "
        "toppled civilization; near-future societies under stress but still "
        "functioning.",
    )

    HAUNTED_LOCATION = (
        "haunted_location",
        22,
        ConceptTagCategory.SETTINGS,
        "The story centers on a supernaturally haunted location. That place "
        "IS the anchor of the haunting.",
        "Look at plot_keywords for \"haunted house\" or named haunted places; "
        "plot_summary anchoring supernatural events to a specific location "
        "as the locus of the haunting.",
        "Broader supernatural horror involving possessions, curses, or mobile "
        "ghosts that follow characters; a scary location that is not "
        "supernaturally haunted; a place of historical suffering without "
        "supernatural haunting tied to it; entities whose haunting moves "
        "with characters rather than being tied to a place.",
    )

    SMALL_TOWN = (
        "small_town",
        23,
        ConceptTagCategory.SETTINGS,
        "The small-town setting is central to the film's identity and "
        "atmosphere. The story feels inseparable from its small-town context.",
        "Look at plot_keywords for \"small town\" directly; plot_summary "
        "explicitly naming the small-town setting and depending on it for "
        "atmosphere, community dynamics, or thematic weight.",
        "A film set in a rural area that is not a town; a film where the "
        "small-town setting is incidental backdrop and the story could happen "
        "anywhere; a city suburb; a film that mentions a town's name without "
        "the town's character mattering to the plot.",
    )

    # -- Characters (IDs 31-33) ------------------------------------------

    FEMALE_LEAD = (
        "female_lead",
        31,
        ConceptTagCategory.CHARACTERS,
        "Every lead role in the film is held by a female character. This "
        "covers both a single female sole protagonist AND an all-female "
        "lead group (two female co-leads, a female trio, a female "
        "ensemble). The disqualifier is the presence of any male "
        "character in a lead role — a male sole protagonist, any "
        "male-and-female co-lead pairing, or any ensemble that includes "
        "male leads. Female supporting characters in a male-led story do "
        "not qualify.",
        "Identify the LEAD ROLE(S) of the film from plot_summary (whose "
        "decisions drive the plot, whose arc(s) form the spine of the "
        "story). Use top_billed_cast as corroborating evidence — the top "
        "slots typically correspond to the lead role(s). Then determine "
        "the gender of EACH lead role from named characters, pronouns in "
        "plot_summary, plot_keywords, and elevator_pitch / "
        "character_arcs[].reasoning where the lead(s) are named. Tag "
        "FEMALE_LEAD when every lead role — whether one or many — is "
        "held by a female character.",
        "Any film where at least one lead role is held by a male "
        "character: a male sole protagonist; a mixed-gender duo or trio "
        "of co-leads; an ensemble that includes any male leads alongside "
        "female ones; a male-led story with a prominent female "
        "supporting character (love interest, family member, mentor, "
        "sidekick, antagonist) who is not herself a lead; a female POV "
        "character in a story whose decision-driving lead is male; any "
        "case where the top-billed actor is male and the plot does not "
        "unambiguously center a different all-female lead structure.",
        # long_form_instructions: 3-step reasoning block.
        "This tag covers TWO patterns: a single female protagonist, OR "
        "an all-female lead group (two co-leads, a trio, a larger "
        "ensemble of leads). The disqualifier is the presence of ANY "
        "male character in a lead role. Reason through this tag in "
        "three explicit steps:\n"
        "\n"
        "STEP 1 — Identify the LEAD ROLE(S) of the movie.\n"
        "Read the plot_summary and ask: whose decisions and "
        "transformation drive the movie? Is there ONE single core "
        "protagonist whose arc IS the story? Or are there TWO OR MORE "
        "co-leads / ensemble members whose storylines together form the "
        "spine of the film? Either pattern can qualify. Use "
        "top_billed_cast as corroborating evidence — the top slots "
        "typically correspond to the lead role(s), though plot_summary "
        "is the primary source. Be explicit about exactly which "
        "characters constitute the lead role(s).\n"
        "\n"
        "STEP 2 — Determine the gender of EACH lead role.\n"
        "For each lead identified in step 1, determine their gender from "
        "named characters and pronouns in plot_summary, plus "
        "plot_keywords. State each lead and their gender explicitly.\n"
        "\n"
        "STEP 3 — Apply the all-female test.\n"
        "If EVERY lead role is held by a female character (one woman "
        "alone, two women as co-leads, three or more women as an "
        "ensemble), tag FEMALE_LEAD. If ANY lead role is held by a male "
        "character — a male sole protagonist, a mixed-gender duo or "
        "trio, an ensemble that contains any male leads — do NOT tag, "
        "even if a prominent female character is among the leads. If "
        "gender cannot be determined with high confidence for any lead "
        "role, do not tag.",
    )

    ENSEMBLE_CAST = (
        "ensemble_cast",
        32,
        ConceptTagCategory.CHARACTERS,
        "An ensemble film has NO SINGLE PROTAGONIST — the story IS the "
        "group's collective arc, or an event the group all reacts to. "
        "Three or more decision-driving protagonists share roughly equal "
        "weight in screen time, plot agency, and arc development. "
        "Discriminating question (apply BEFORE counting named characters): "
        "if you removed any ONE of the candidates' storylines, would the "
        "film still be a recognizable, complete movie? If yes → it is NOT "
        "an ensemble; that character was support, not co-lead. Only tag "
        "ENSEMBLE_CAST when removing any candidate would fundamentally "
        "collapse the film.",
        "Apply the removal test FIRST — name each candidate, then ask "
        "whether the film survives their removal. Only after the removal "
        "test passes, corroborate with: pov_perspective terms for "
        "\"multiple POVs\" or \"rotating POV\"; plot_summary showing three "
        "or more characters with independent intertwined arcs of "
        "comparable importance; character_arcs[].reasoning naming multiple "
        "developed protagonists; characterization_methods terms describing "
        "ensemble work. A long top_billed_cast list is a WEAK signal — "
        "five named slots is normal for any film with a developed "
        "supporting cast; do not infer ensemble from cast size alone.",
        "The HERO-WITH-ALLIES PATTERN — a single protagonist surrounded by "
        "a developed supporting cast (mentor, romantic interest, comic "
        "relief, sidekick, rival, antagonist) — is NOT ensemble even when "
        "those supporting characters are individually beloved, "
        "well-developed, and famously named. The protagonist's journey is "
        "the film's spine; everyone else is in orbit. A protagonist with "
        "several important supporting characters; parallel plotlines where "
        "one character's arc is clearly primary; exactly two co-leads of "
        "equal weight (ensemble requires three or more decision-driving "
        "protagonists); a long character list with one clear lead. A long "
        "plot_summary naming many characters does not by itself make the "
        "film an ensemble — count whose DECISIONS drive the plot forward, "
        "not how many are named.",
    )

    ANTI_HERO = (
        "anti_hero",
        33,
        ConceptTagCategory.CHARACTERS,
        "The protagonist is presented as operating outside conventional "
        "morality as a defining character trait. Substantive moral "
        "boundary-crossing — criminal acts, violence, exploitation, or "
        "vigilantism — is or HAS BEEN their primary mode of operating for "
        "a meaningful portion of the runtime, not a single extreme choice "
        "made under duress. A redemption arc does NOT disqualify: a "
        "character originally presented as an anti-hero who later finds "
        "their way still counts — the original anti-heroism is what the "
        "film is about. The discriminating question: if you described "
        "this protagonist to someone in one sentence, would \"morally "
        "compromised\", \"criminal\", or \"outlaw\" be central to that "
        "sentence — either as their current mode, or as the starting "
        "point the story departs from?",
        "Derive ANTI_HERO from raw behavior described in plot_summary, "
        "NOT from pre-classified upstream labels. Primary sources: "
        "plot_summary (does the protagonist actually OPERATE as an "
        "anti-hero for a substantive portion of the runtime — criminal "
        "acts, exploitation, vigilantism as default mode at any point in "
        "the story? Or principled action with rough edges throughout?); "
        "character_arc_labels from plot_analysis (an arc transformation "
        "FROM a morally compromised starting point INTO a moral end-state "
        "still indicates the character WAS an anti-hero for a meaningful "
        "portion of the runtime — redemption arcs still qualify); "
        "conflict_type (is the protagonist's stance structurally outside "
        "conventional morality, at least for a meaningful portion of the "
        "story?); plot_keywords for explicit anti-hero framings. For "
        "ENSEMBLE films with multiple decision-driving protagonists, "
        "evaluate each storyline's protagonist independently — if ANY ONE "
        "of them operates substantively as an anti-hero across a "
        "meaningful portion of the runtime, the film qualifies.",
        "A flawed but fundamentally moral character who does the right "
        "thing throughout — being morally compromised must be a "
        "substantive operating mode at some point in the film, not "
        "merely a personality flaw or rough edges. A single act of "
        "extreme moral consequence under impossible duress (a parent "
        "making a tragic mercy choice, a survivor killing to save others "
        "in a life-or-death situation, a non-criminal protagonist driven "
        "to violence by an extraordinary event) does NOT make a "
        "character an anti-hero if their operating mode across the rest "
        "of the film is fundamentally moral. A parent on a rescue "
        "mission acts morally regardless of methods; a character "
        "described as \"rebellious\" or \"rule-breaking\" without "
        "substantive moral transgression; a minor-rule-breaking teen or "
        "adolescent. The test is whether substantive anti-heroic "
        "operation occupies a meaningful portion of the runtime — "
        "whether or not the arc ends in redemption.",
    )

    # -- Endings (IDs 41-43, plus classification-only NO_CLEAR_CHOICE=-1) -

    HAPPY_ENDING = (
        "happy_ending",
        41,
        ConceptTagCategory.ENDINGS,
        "Audience leaves feeling positive — satisfaction, relief, triumph, "
        "or warmth. This is the EMPIRICALLY DOMINANT ending type for "
        "narrative film: most genre cinema (romance, action, adventure, "
        "family, comedy, superhero, horror-with-survival, sports films) "
        "lands here, including films whose protagonists pay substantial "
        "costs along the way. A hard-won victory IS happy. Cost-along-the-"
        "way is the standard structure of a satisfying happy ending, not "
        "a disqualifier.",
        "Look at the literal CLOSING SCENE in plot_summary — if it is a "
        "recognized celebration beat (a triumphant kiss, a family reunion "
        "hug, a threat-defeated cheer, a protagonist-restored shot, a "
        "platform-raise / mountaintop moment, a smiling embrace), the "
        "ending is HAPPY regardless of what was lost during the runtime. "
        "Look at emotional_observations for \"uplifting\", \"satisfying\", "
        "\"triumphant\", \"warm closure\", \"earned\", \"hard-won\", "
        "\"achievement at a cost\" (this last phrase signals happy with "
        "sacrifice, NOT bittersweet); plot_summary for the final beat at "
        "credits and the characters' end-state; plot_keywords for \"happy "
        "ending\". Surviving a horror story and defeating the threat is "
        "happy when the protagonists are safe and the danger is over. "
        "When evidence is ambiguous between HAPPY and BITTERSWEET, default "
        "to HAPPY — bittersweet is the rarer, higher-bar tag.",
        "Merely surviving a horrific ordeal without positive feeling — the "
        "audience feels relief but not triumph or warmth; a victory that "
        "feels hollow or Pyrrhic to the audience where the cost has "
        "overwhelmed the win; a film with a positive plot outcome that "
        "lands flat or empty because of accumulated grief; an ending where "
        "the protagonists win on paper but the audience leaves devastated "
        "(that is SAD, not happy).",
    )

    SAD_ENDING = (
        "sad_ending",
        42,
        ConceptTagCategory.ENDINGS,
        "Audience leaves feeling predominantly sad — grief, devastation, "
        "heartbreak. The lasting emotion is loss, failure, or defeat.",
        "Look at emotional_observations for \"devastating\", \"tragic\", "
        "\"heartbreaking\", \"bleak\", \"left me sobbing\", \"crushing\"; "
        "plot_summary for an ending state defined by loss (a funeral, a "
        "destroyed home, a protagonist alone in ruin, an unsaved life); "
        "the literal closing scene as a beat of grief, defeat, or loss "
        "with no recuperative upswing. A cliffhanger where the heroes "
        "have lost and the villain remains at large IS sad — narrative "
        "closure is not required.",
        "A victory achieved at great cost where the audience still feels "
        "the victory; an emotionally intense movie with a positive outcome; "
        "a tragic journey that ends in redemption, peace, or thematic "
        "uplift; a film with grim mid-story events but a recuperative "
        "ending.",
    )

    BITTERSWEET_ENDING = (
        "bittersweet_ending",
        43,
        ConceptTagCategory.ENDINGS,
        "An uncommon ending type where the audience leaves with genuinely "
        "unresolvable mixed feelings — they cannot comfortably say \"this "
        "was a happy movie\" OR \"this was a sad movie\" and mean it. "
        "Discriminating question: would a reasonable viewer feel something "
        "is missing or wrong if you called the ending HAPPY? AND would they "
        "feel the same way if you called it SAD? Only if BOTH are true does "
        "BITTERSWEET apply. Films with mixed elements almost always still "
        "land clearly on HAPPY or SAD — bittersweet is rare and is NOT a "
        "fallback for endings that feel complicated.",
        "Look at emotional_observations for language that *explicitly* "
        "describes the AUDIENCE leaving with unresolved mixed feelings — "
        "\"mixed feelings\", \"joy AND sorrow held in tension\", \"unable "
        "to celebrate fully\", \"a knot in the stomach despite the win\", "
        "\"genuinely torn\". Treat \"earned\", \"hard-won\", \"achievement "
        "at a cost\", \"sacrifice for victory\" as HAPPY signals, NOT "
        "bittersweet signals — they describe the standard structure of "
        "satisfying happy endings. Look at the literal closing scene in "
        "plot_summary: bittersweet endings tend to close on a quiet, "
        "contemplative beat (a long look, an unspoken moment, a "
        "what-might-have-been montage, a protagonist staring into the "
        "distance with both achievement and loss visible) rather than "
        "on a celebration beat. If the closing scene is a triumph kiss, "
        "family reunion, threat-defeated cheer, or platform-raise "
        "moment, the ending is HAPPY no matter what was lost along the "
        "way.",
        "Sacrifice along the journey followed by an unambiguous celebration "
        "or triumph at the credits-roll moment — that is HAPPY, full stop. "
        "Genre films (romance, action, family, horror-with-survival, "
        "superhero) where the protagonists pay a price and then win — that "
        "is the *default structure* of a happy ending, not bittersweet. A "
        "tragic journey that ends in redemption, peace, or thematic uplift "
        "— that is SAD or HAPPY depending on the lasting emotion, not "
        "bittersweet. A film with grief mid-runtime but a recuperative "
        "ending — that is HAPPY. Structural ambiguity about WHAT happened "
        "(open or ambiguous endings) — narrative uncertainty is not "
        "emotional ambiguity; route those to OPEN_ENDING above and pick "
        "the emotional tag separately. \"This ending was complicated\" or "
        "\"the audience had a lot to process\" is NOT enough — bittersweet "
        "requires that neither HAPPY nor SAD would be a defensible "
        "alternative. If you can defend HAPPY or SAD, pick that one.",
    )

    NO_CLEAR_CHOICE = (
        "no_clear_choice",
        -1,
        ConceptTagCategory.ENDINGS,
        "The evidence is ambiguous, insufficient, or the ending's emotion "
        "does not clearly fit happy, sad, or bittersweet. Classification-"
        "only; filtered out before storage.",
        "Use when the extracted observations do not point clearly to one "
        "of the above. Specifically: emotional_observations describe the "
        "ending as \"ambiguous\", \"lingering\", \"contemplative\", "
        "\"philosophical\", or \"open to interpretation\" without clear "
        "positive/negative/mixed valence; the closing scene in "
        "plot_summary is an existential beat (a long-held expression, a "
        "cosmic-indifference image, an unresolved philosophical question) "
        "that doesn't map to celebration / loss / mixed-feelings; "
        "craft_observations describe the ending as deliberately "
        "interpretive.",
        "Endings that have some complexity but where one of the three primary "
        "tags clearly applies (do not default to NO_CLEAR_CHOICE just because "
        "the ending is not simple); endings with a clear emotional valence "
        "even when the narrative is structurally ambiguous (the structural "
        "openness goes to OPEN_ENDING above, the emotional valence to one of "
        "HAPPY/SAD/BITTERSWEET here).",
    )

    # -- Experiential (IDs 51-52) ----------------------------------------

    FEEL_GOOD = (
        "feel_good",
        51,
        ConceptTagCategory.EXPERIENTIAL,
        "The overall audience experience is warm and uplifting. The viewer "
        "leaves feeling positive, hopeful, and lifted up. About emotional "
        "WARMTH, not excitement or adrenaline.",
        "PRIMARY source: emotional_observations. Read the full emotional "
        "landscape described and apply a holistic test: is the dominant "
        "tone OVERWHELMINGLY one of feel-good emotions — warmth, joy, "
        "hope, uplift, charm, heartwarming connection? Does the "
        "description make clear that the movie is meant to leave the "
        "audience hopeful and emotionally lighter? Tag FEEL_GOOD only "
        "when the warm/hopeful signal is overwhelming across the "
        "described audience experience. A small amount of tension or "
        "stakes during the journey does not disqualify warmth at the "
        "destination — but a meaningful counterweight of heavy emotions "
        "does.",
        "A movie whose described emotional landscape contains a strong "
        "presence of NON-feel-good emotions — grief, dread, devastation, "
        "sustained dark tension, prolonged despair, heavy melancholy — "
        "alongside its warmer beats. That is a MIXED experience, not "
        "feel_good, even when the warmer beats are real. A heavy or "
        "emotionally devastating movie that nonetheless has a "
        "recuperative ending is not feel_good. A pure adrenaline "
        "experience with no emotional warmth (action thrills, horror "
        "scares); cathartic satisfaction from violent revenge; a "
        "guilty-pleasure enjoyment of trashy or gory content. Do NOT "
        "infer from genre alone. When the emotional landscape contains a "
        "meaningful counterweight of heavy feelings, the film is not "
        "feel_good — this tag is reserved for films whose warm, "
        "hopeful, lifting tone is the overwhelming signal.",
    )

    TEARJERKER = (
        "tearjerker",
        52,
        ConceptTagCategory.EXPERIENTIAL,
        "The movie makes audiences cry — and the available text directly "
        "states that it does. This is a strict, literal test: tag only "
        "when there is a direct statement that audiences cried, sobbed, "
        "wept, or shed tears.",
        "PRIMARY and AUTHORITATIVE source: emotional_observations. Apply "
        "a single, literal test — is there a DIRECT, EXPLICIT statement "
        "anywhere in the available text that the audience cried, sobbed, "
        "wept, shed tears, or was reduced to tears? If YES, tag "
        "TEARJERKER. If NO, do NOT tag. Do not derive this tag from any "
        "other signal — not from plot sadness, not from genre, not from "
        "emotional intensity, not from synonyms or near-equivalents.",
        "Movies described as \"moving\", \"touching\", \"tugs at "
        "heartstrings\", \"poignant\", \"devastating\", \"heartbreaking\", "
        "\"emotionally wrecking\", or any similar language that does NOT "
        "literally state audience crying. Plot-level sadness, tragic "
        "events, character deaths, emotional intensity, or themes of "
        "grief that the text does not explicitly connect to audience "
        "crying. The bar is strict and literal: the available text must "
        "directly describe audiences as having cried — synonyms for "
        "emotional impact alone do not qualify.",
    )

    # -- Content Flags (ID 61) -------------------------------------------

    ANIMAL_DEATH = (
        "animal_death",
        61,
        ConceptTagCategory.CONTENT_FLAGS,
        "A non-human animal (dog, cat, horse, bird, etc.) dies on screen or "
        "as a significant plot point. Exclusively about animals.",
        "TIERED EVIDENCE — apply in order. TIER 1 (PRIMARY): "
        "plot_summary and plot_keywords. Look for ANY evidence that a "
        "non-human animal dies in the film — the death does not have to "
        "be violent, severe, central, or extensively depicted. An "
        "on-screen death, an off-screen death described as part of the "
        "story, a pet euthanized, an animal killed by an antagonist, an "
        "animal killed in passing as part of a beat — all qualify. If "
        "plot_summary or plot_keywords contains any such evidence of an "
        "animal dying, tag ANIMAL_DEATH. TIER 2 (FALLBACK, used only "
        "when TIER 1 shows NO evidence of an animal death in the plot): "
        "consult parental_guide_items. If parental_guide_items lists a "
        "category like \"Violence Against Animals\" or \"Animal "
        "cruelty\" at any severity, tag ANIMAL_DEATH. The fallback "
        "exists to catch animal deaths that the plot text happens not "
        "to mention; the presence of unrelated parental_guide entries "
        "(violence against humans, profanity, frightening scenes) is "
        "NOT a signal either way for animal_death — only an "
        "animal-specific advisory counts.",
        "Human deaths of any kind; violence against humans; the word "
        "\"animal\" appearing in an unrelated context; creatures in "
        "fantasy or sci-fi that are clearly not real animals; an animal "
        "merely mentioned without dying (a pet that appears in a scene, "
        "a working animal in the background, a meat-industry backdrop "
        "where no on-screen or plot-described death occurs); a "
        "parental_guide entry for human violence or general frightening "
        "scenes with no animal-specific category.",
    )


# ---------------------------------------------------------------------------
# Dynamic per-category enum classes.
#
# Built at module load by filtering the master ConceptTag enum on category.
# Used by Pydantic-typed Assessment models (NarrativeStructureAssessment,
# etc.) so the LLM's JSON schema enforces per-category tag values for
# structured output. Each member has the same string value as the master
# enum member, plus `concept_tag_id` and `description` attributes attached
# post-construction so call sites like
# ConceptTagsOutput.apply_deterministic_fixups() and
# ConceptTagsOutput.all_concept_tag_ids() keep working without changes.
#
# Rich attributes (selection_criteria, boundary_cases,
# long_form_instructions) live only on the master ConceptTag — the prompt
# assembler reads from the master, not from these derived classes.
# ---------------------------------------------------------------------------


def _build_category_enum(category: ConceptTagCategory) -> type[Enum]:
    """Build a per-category str-Enum subset of ConceptTag.

    Each member is a real `str, Enum` value with the same `_value_` as the
    master ConceptTag member. After enum construction we attach
    `concept_tag_id` and `description` to each member instance so existing
    consumers that read `.concept_tag_id` on per-category tags keep
    working without indirection through the master enum.
    """
    members = [t for t in ConceptTag if t.category is category]
    cls = Enum(
        category.enum_class_name,
        {t.name: t.value for t in members},
        type=str,
    )
    for t in members:
        member = getattr(cls, t.name)
        member.concept_tag_id = t.concept_tag_id
        member.description = t.description
    return cls


NarrativeStructureTag = _build_category_enum(ConceptTagCategory.NARRATIVE_STRUCTURE)
PlotArchetypeTag      = _build_category_enum(ConceptTagCategory.PLOT_ARCHETYPES)
SettingTag            = _build_category_enum(ConceptTagCategory.SETTINGS)
CharacterTag          = _build_category_enum(ConceptTagCategory.CHARACTERS)
EndingTag             = _build_category_enum(ConceptTagCategory.ENDINGS)
ExperientialTag       = _build_category_enum(ConceptTagCategory.EXPERIENTIAL)
ContentFlagTag        = _build_category_enum(ConceptTagCategory.CONTENT_FLAGS)


# All concept tags as a flat tuple, excluding classification-only values
# (NO_CLEAR_CHOICE) that are never stored or searched. Derived from the
# master ConceptTag enum; consumers iterate this for .name, .description,
# and .concept_tag_id (see schemas/unified_classification.py).
ALL_CONCEPT_TAGS: tuple = tuple(
    tag for tag in ConceptTag if tag.concept_tag_id >= 0
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
    CHARACTER_FRANCHISE = "character_franchise"
    NON_CHARACTER_FRANCHISE = "non_character_franchise"
    STUDIO = "studio"
    ACTOR = "actor"
    STANDARD = "standard"


# Step 1 ambiguity level. This is a compact branching signal, not a
# confidence score.
class QueryAmbiguityLevel(StrEnum):
    CLEAR = "clear"
    MODERATE = "moderate"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Search V2 step 2 modifier discriminator.
#
# Every requirement fragment is an attribute. Polarity phrases and
# role markers are nested inside that attribute as entries in its
# modifiers list; this enum discriminates which kind of modifier each
# entry is. Kept to exactly the two kinds that actually bind to an
# adjacent attribute — standalone fragment types (attribute,
# selection_rule) are no longer part of this vocabulary.
# ---------------------------------------------------------------------------
class LanguageType(StrEnum):
    POLARITY_MODIFIER = "polarity_modifier"
    ROLE_MARKER = "role_marker"


# Fit of a coverage_evidence entry's category against its
# captured_meaning. Drives downstream dispatch:
#   clean / partial  → dispatch to the category handler
#   no_fit           → pruned before dispatch (observation turned out
#                      speculative or empty)
class FitQuality(StrEnum):
    CLEAN = "clean"
    PARTIAL = "partial"
    NO_FIT = "no_fit"


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
    MEDIA_TYPE = "media_type"
    NEUTRAL_SEED = "neutral_seed"
    CHRONOLOGICAL = "chronological"


# Whether a generated endpoint spec runs as a candidate finder
# (produces its own pool by membership / top-K) or a pool reranker
# (scores an existing candidate pool). See
# search_improvement_planning/search_method_deterministic_logic.md §2.
class OperationType(StrEnum):
    CANDIDATE_GENERATOR = "candidate_generator"
    POOL_RERANKER = "pool_reranker"


# How a CategoryName composes its orchestrator-visible call scores
# into a single per-category score during Stage 4 per-trait scoring.
# Declared per-category on the CategoryName enum; consumed by the
# Stage 4 within-category combine. See
# search_improvement_planning/rescore_overhaul.md (Within-category
# combine) for the full rationale and per-mode semantics.
#
# - SINGLE: category fires exactly one orchestrator-visible call;
#   passthrough.
# - ADDITIVE: multiple calls together complete the picture; product
#   across [0, 1] scores. Strict — any 0 zeros the category.
# - ALTERNATIVES: each call is a distinct way of finding the trait;
#   max across calls. Matching any one is sufficient evidence.
# - CONSENSUS: every committed call should weigh in. Soft geometric
#   mean over committed-call scores with an EPS floor (mirrors the
#   FACETS across-category fold). One endpoint scoring high cannot
#   over-promote the category on its own — sibling endpoints that
#   committed but scored weakly pull the result down without a single
#   zero collapsing it. Single-commit cases passthrough (geom mean
#   over one element = that element).
# - NO_OP: category never fires (e.g. BELOW_THE_LINE_CREATOR's
#   EXPLICIT_NO_OP bucket emits zero specs); the per-category combine
#   returns a sentinel that the across-category max skips entirely.
class CategoryCombineType(StrEnum):
    SINGLE = "single"
    ADDITIVE = "additive"
    ALTERNATIVES = "alternatives"
    CONSENSUS = "consensus"
    NO_OP = "no_op"


# Handler query-generation bucket. Determines the shared instruction
# shape a category handler uses before endpoint-specific schemas fill in
# concrete parameters.
# See search_improvement_planning/query_buckets.md for definitions.
class HandlerBucket(StrEnum):
    NO_LLM_PURE_CODE = "no_llm_pure_code"
    EXPLICIT_NO_OP = "explicit_no_op"
    SINGLE_NON_METADATA_ENDPOINT = "single_non_metadata_endpoint"
    SINGLE_METADATA_ENDPOINT = "single_metadata_endpoint"
    PREFERRED_REPRESENTATION_FALLBACK = "preferred_representation_fallback"
    SEMANTIC_PREFERRED_DETERMINISTIC_SUPPORT = "semantic_preferred_deterministic_support"
    CHARACTER_FRANCHISE_FANOUT = "character_franchise_fanout"
    AUDIENCE_SUITABILITY_DETERMINISTIC_FIRST = "audience_suitability_deterministic_first"


# ---------------------------------------------------------------------------
# Search V2 step 2 query categorization taxonomy.
#
# CategoryName has moved to schemas/trait_category.py to keep the 44-cat
# trait taxonomy isolated from the broader shared-enum module. Import
# from `schemas.trait_category` rather than from this file.
# ---------------------------------------------------------------------------


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
# in the per-category entity spec output (PersonQuerySpec /
# CharacterQuerySpec / TitlePatternQuerySpec).
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


# How to score billing prominence for entity lookups. Unified enum
# covering both actor-table searches (person_category is actor or
# broad_person) and character searches (entity_type is CHARACTER).
#
# Valid-mode subsets, enforced by the per-category entity spec validator:
#   actor-table  → {DEFAULT, LEAD, SUPPORTING, MINOR}
#   character    → {DEFAULT, CENTRAL}
#   everything else (director/writer/producer/composer, title_pattern)
#                → field must be null
#
# When the LLM emits an out-of-scope value for the entity being
# searched, the validator remaps rather than rejects:
#   character receives LEAD        → CENTRAL (prominence-as-subject)
#   character receives SUPPORTING  → DEFAULT
#   character receives MINOR       → DEFAULT
#   actor-table receives CENTRAL   → LEAD (top-billing equivalent)
#
# Mode semantics:
#   DEFAULT    — no explicit prominence signal; scorer picks a
#                gentle default curve.
#   LEAD       — actor-table: user wants the actor in a leading role.
#   SUPPORTING — actor-table: user wants the actor in a supporting role.
#   MINOR      — actor-table: user wants a cameo / minor appearance.
#   CENTRAL    — character: user frames the character as the subject
#                of the film ("Spider-Man movies").
#
# See finalized_search_proposal.md §Actor Prominence Scoring and
# search_improvement_planning/character_scoring_revamp.md for the
# per-mode curves and the compression-to-[0.5, 1.0] rationale.
class ProminenceMode(StrEnum):
    DEFAULT = "default"
    LEAD = "lead"
    SUPPORTING = "supporting"
    MINOR = "minor"
    CENTRAL = "central"


# Subset of ProminenceMode that is valid for actor-table scoring.
# Used by execution code to drive the actor-mode dispatch table.
_ACTOR_VALID_PROMINENCE_MODES: frozenset[ProminenceMode] = frozenset({
    ProminenceMode.DEFAULT,
    ProminenceMode.LEAD,
    ProminenceMode.SUPPORTING,
    ProminenceMode.MINOR,
})

# Subset of ProminenceMode that is valid for character scoring.
_CHARACTER_VALID_PROMINENCE_MODES: frozenset[ProminenceMode] = frozenset({
    ProminenceMode.DEFAULT,
    ProminenceMode.CENTRAL,
})


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


# How multiple executable award searches from one category call combine.
# ANY: alternatives; a movie's best search score wins.
# AVERAGE: additive partial credit; missing searches count as 0.0.
class AwardCombineMode(StrEnum):
    ANY = "any"
    AVERAGE = "average"


# Shared scoring method for endpoint specs that combine multiple
# same-call values or dimensions.
class ScoringMethod(StrEnum):
    ANY = "ANY"
    ALL = "ALL"


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


# ---------------------------------------------------------------------------
# Search V2 trait-role + polarity vocabulary.
#
# Every Trait committed by Step 2 carries a role (carver vs qualifier)
# and a polarity (positive vs negative). Those two pre-committed
# values stamp through onto every EndpointParameters wrapper produced
# by Step 4, where together they determine which of the four
# HandlerResult buckets the finding falls into:
#
#                   | POSITIVE                | NEGATIVE
#   ----------------+-------------------------+----------------------
#   CARVER          | inclusion_candidates    | exclusion_ids
#   QUALIFIER       | preference_specs        | downrank_candidates
#
# See search_improvement_planning/category_handler_planning.md
# ("From LLM output to return buckets") for the full mapping.
# ---------------------------------------------------------------------------


# Whether the trait gates eligibility (CARVER — a yes/no test that
# selects which movies pass) or scores/refines within an already-
# gated population (QUALIFIER — a descriptive attribute that colors
# the ranking).
class Role(StrEnum):
    CARVER = "carver"
    QUALIFIER = "qualifier"


# Whether the trait pushes candidates IN (positive) or OUT / DOWN
# (negative). Orthogonal to role — together they cover the full
# 2x2 of inclusion / exclusion / preference / downrank.
class Polarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


# How a trait relates to the rest of the query. Closed enum read by
# Step 3's role analysis to branch decomposition behavior:
#
#   INDEPENDENT covers parallel filters AND qualifier-on-population.
#     No cross-trait info flow needed; the trait stands on its own
#     for retrieval / scoring purposes.
#   POSITIONING_REFERENCE marks an anchor that a sibling is comparing,
#     transposing, or scoping against. The trait's identity is being
#     used as a template; specific axes of that template may be
#     replaced by siblings.
#   POSITIONING_QUALIFIER marks the modifier that names a substitute
#     for some axis on a sibling reference. The qualifier itself is
#     independently scorable, but its meaning in the query is
#     SUBSTITUTION on the reference.
#
# The role is a structural classification, not a polarity / weight
# choice. Read by what the trait DOES in the query, not by surface
# tokens — the same connective ("with", "but", "-style", "like")
# can join independent or positioning relations depending on what
# content phrases it joins.
class TraitRelationshipRole(StrEnum):
    INDEPENDENT = "independent"
    POSITIONING_REFERENCE = "positioning_reference"
    POSITIONING_QUALIFIER = "positioning_qualifier"


# How Phase D (stage-4 across-category fold) collapses a trait's
# per-category scores into a single trait_score in [0, 1]:
#
#   SOLO — exactly one surviving category cleanly covers every
#     dimension the trait calls for. Other candidates surfaced as
#     adjacency context but do not add coverage the primary doesn't
#     already provide. Phase D has nothing to fold; the single
#     category's score IS the trait_score. The orchestrator trims
#     category_calls to the first entry before retrieval, so dropped
#     categories never fan out to handler-LLM calls or endpoint
#     fetches.
#   FRAMINGS — multiple surviving categories are alternative homes
#     for the same underlying thing, AND no single category covers
#     the trait cleanly on its own; matching ANY ONE is sufficient
#     evidence of the criterion. Phase D MAX-folds them; redundant
#     categories reinforce each other as alternative routes to the
#     same signal.
#   FACETS — categories cover DIFFERENT axes of a compound concept;
#     ALL facets must fire to a degree for the criterion to be met.
#     Phase D PRODUCT-folds them; duplicating axis coverage
#     amplifies the wrong signals.
#
# Step 3 commits this AFTER its candidate analysis and BEFORE
# committing category_calls — the choice shapes what categories make
# sense to commit. The decision is hierarchical: ask the SOLO
# coverage question first; only when no single category covers the
# trait cleanly does the FRAMINGS-vs-FACETS question come into play.
# Surfacing the commit on TraitDecomposition lets stage-4 branch the
# across-category fold mechanically.
class TraitCombineMode(StrEnum):
    SOLO = "solo"
    FRAMINGS = "framings"
    FACETS = "facets"
