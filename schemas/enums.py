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
            "OPEN_ENDING and CLIFFHANGER_ENDING describe HOW the story ends, not how it FEELS — "
            "they coexist with the emotional ENDINGS tags below. A devastating cliffhanger still "
            "tags SAD_ENDING; a warm open ending still tags HAPPY_ENDING. Evaluate both sections "
            "independently."
        ),
    )
    PLOT_ARCHETYPES = (
        "plot_archetypes",
        "PLOT ARCHETYPES",
        "the central premise / driving force. Tag applies when the concept IS the movie, "
        "not just an element in the plot",
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
        "how the audience FEELS when credits roll. Exactly one tag (including no_clear_choice)",
        "one_of",
        "endings",
        "EndingTag",
        # section_instructions: the endings HOW-TO block, paraphrased.
        (
            "HOW TO THINK THROUGH THIS CATEGORY — work the steps before selecting:\n"
            "\n"
            "BASE RATES (apply FIRST): HAPPY is the empirically dominant default for most "
            "narrative film including ones with substantial cost-along-the-way. SAD is common "
            "for tragedies/downer endings. BITTERSWEET is UNCOMMON — mixed elements usually "
            "still land clearly on HAPPY or SAD. NO_CLEAR_CHOICE is for genuinely contemplative/"
            "existential endings that fit none. Not equal-probability options.\n"
            "\n"
            "1. CLOSING SCENE (PRIMARY): identify the literal final scene before credits from "
            "plot_summary. Celebration beat (triumphant kiss, reunion hug, threat-defeated cheer, "
            "platform-raise, mountaintop/sunrise/restored shot) → HAPPY regardless of runtime cost. "
            "Loss/defeat/grief beat (funeral, destroyed home, protagonist alone in ruin) with no "
            "upswing → SAD. Quiet contemplative beat (unspoken moment, long look, "
            "what-might-have-been montage) with joy+sorrow unresolved → BITTERSWEET candidate.\n"
            "\n"
            "2. emotional_observations — filter for ENDING language, not journey-level:\n"
            "   - 'uplifting' / 'satisfying' / 'triumphant' / 'warm closure' / 'feel-good' / 'earned' / 'hard-won' / 'achievement at a cost' / 'sacrifice rewarded' → HAPPY (cost+victory IS happy, not bittersweet).\n"
            "   - 'devastating' / 'tragic' / 'heartbreaking' / 'bleak finale' / 'crushing' / 'unrelenting loss' → SAD.\n"
            "   - explicit 'mixed feelings audiences cannot resolve' / 'knot despite the win' / 'genuinely torn' / 'unable to celebrate fully' → BITTERSWEET (HIGH bar; must affirm cannot land on HAPPY or SAD).\n"
            "   - 'ambiguous' / 'contemplative' / 'open to interpretation' / existential without valence → NO_CLEAR_CHOICE.\n"
            "Discard runtime emotions ('tense', 'frightening', 'dark') that don't describe the ending.\n"
            "\n"
            "3. Final state from plot_summary — what's gained/lost/unresolved? Factual evidence "
            "to reconcile with closing scene, NOT a direct verdict. Horror survived + threat "
            "defeated = HAPPY despite suffering.\n"
            "\n"
            "4. Ending plot_keywords ('happy ending', 'tragic ending', 'twist ending').\n"
            "\n"
            "DEFAULTS: ambiguous HAPPY-vs-BITTERSWEET → HAPPY. ambiguous SAD-vs-BITTERSWEET → SAD. "
            "Pick BITTERSWEET only when neither HAPPY nor SAD is a defensible alternative.\n"
            "\n"
            "Structural ambiguity about WHAT happened ≠ emotional ambiguity. Warm closing scene → "
            "HAPPY even if the plot leaves a structural question open."
        ),
        # cross_tag_note: structural-vs-emotional independence note.
        (
            "Captures the AUDIENCE'S emotional experience at credits, not a factual ledger of "
            "plot outcomes. A protagonist dying to save others may leave audiences devastated "
            "(sad), triumphant (happy), or torn (bittersweet) — outcome alone doesn't determine "
            "the tag. Independent of OPEN_ENDING / CLIFFHANGER_ENDING above (structural, not "
            "emotional); always evaluate even when you tagged those."
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
    # (Already terse — no edits.)


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
        "A surprise revelation recontextualizes events the audience has already seen. "
        "The audience must have formed an understanding the reveal overturns.",
        "information_control NT terms (twist/reversal/hidden-truth); plot_keywords "
        "'surprise ending'/'plot twist'; plot_summary explicit late reveals reframing prior "
        "scenes; craft_observations describing twists/rug-pulls/third-act reveals.",
        "Any surprise without recontextualization of prior scenes; foreshadowed late betrayal; "
        "new info that adds but doesn't change prior understanding; final-act tragic irony "
        "with nothing earlier reinterpreted; sequel-setup reveals raising new questions rather "
        "than overturning past ones; end-of-film structural ambiguity about what's real "
        "(→ OPEN_ENDING, not a recontextualizing twist).",
    )

    TWIST_VILLAIN = (
        "twist_villain",
        2,
        ConceptTagCategory.NARRATIVE_STRUCTURE,
        "A character presented as good/trustworthy/neutral/allied for a substantial portion of "
        "the film is revealed late to be a primary antagonist. The moral-category FLIP (good/ally "
        "→ villain) is the surprise. Auto-implies PLOT_TWIST.",
        "information_control terms for hidden antagonist / false ally / betrayal reveal; "
        "plot_summary phrasings like 'villain all along', 'secretly orchestrating', 'hidden "
        "mastermind'; craft_observations citing the antagonist reveal as a craft moment. The "
        "signal must overturn ALIGNMENT, not just motives.",
        "Known villain whose evil's depth/scope is later revealed (motivation depth ≠ category "
        "flip); villain whose plan is more elaborate than expected but whose villainy was never "
        "in doubt; suspicious/sinister-coded from act one even if confirmation is late; "
        "protagonist's psychological alter-ego / split-personality / hallucination revealed late "
        "(→ UNRELIABLE_NARRATOR territory); known antagonist whose methods are revealed later. "
        "Surprising-evil-motivation alone is not enough — the audience must have genuinely "
        "believed the character was on the protagonist's side, neutral, or good.",
    )

    TIME_LOOP = (
        "time_loop",
        3,
        ConceptTagCategory.NARRATIVE_STRUCTURE,
        "Characters relive the same time period repeatedly as the central premise. "
        "Distinct from time travel.",
        "narrative_delivery terms 'time loop'/'reliving'; plot_keywords 'time loop' directly; "
        "plot_summary explicit re-living language (waking to the same day repeatedly).",
        "Time travel visiting different periods (distinct concept); single repeated scene used "
        "as flashback; recurring dream sequences without claimed temporal repetition; cyclical "
        "themes / visual repetition without literally relived time.",
    )

    NONLINEAR_TIMELINE = (
        "nonlinear_timeline",
        4,
        ConceptTagCategory.NARRATIVE_STRUCTURE,
        "Non-chronological structure is a DEFINING identity of the film. The audience "
        "reconstructs the timeline from deliberately scrambled pieces.",
        "narrative_delivery terms 'nonlinear'/'fragmented timeline'; plot_keywords 'nonlinear "
        "timeline'; plot_summary structure showing deliberate chapter-shuffling; "
        "craft_observations like 'told in chapters', 'moves between timelines', "
        "'non-linearly structured'.",
        "Occasional flashbacks inside an otherwise chronological narrative; single framing "
        "device / prologue out of order; flash-forward cold open with the rest chronological; "
        "epistolary / anthology structures that proceed chronologically within and across segments.",
    )

    UNRELIABLE_NARRATOR = (
        "unreliable_narrator",
        5,
        ConceptTagCategory.NARRATIVE_STRUCTURE,
        "The narrator/POV character's account is later revealed to the AUDIENCE as distorted "
        "or fabricated. Trust with the audience is broken — not just between characters.",
        "pov_perspective terms 'unreliable narrator'/'subjective POV'; plot_summary explicit "
        "revelation that prior shown material was distorted; craft_observations flagging "
        "unreliable narration as a craft choice.",
        "Character lies to other characters but the audience sees the truth (= deception, not "
        "unreliable narration); hallucinations unless the film presents them as reality then "
        "reveals the distortion; flashbacks shown from a biased perspective without an in-film "
        "reveal that the flashbacks were wrong.",
    )

    OPEN_ENDING = (
        "open_ending",
        6,
        ConceptTagCategory.NARRATIVE_STRUCTURE,
        "Story completes its arc but deliberately leaves its CENTRAL THEMATIC QUESTION "
        "ambiguous. Discriminating test: did the film resolve the main conflict it posed about "
        "its protagonists? If YES, do not tag — regardless of side details or franchise hooks. "
        "Audiences should debate what the film MEANS, not what comes next.",
        "plot_keywords 'ambiguous ending'; plot_summary final beat that intentionally avoids "
        "central-question resolution; emotional_observations 'ambiguous'/'lingering question'/"
        "'audiences debate'; craft_observations describing intentional ambiguity at close. "
        "Resolution test: name the act-one central conflict (goal/mystery/relationship), then "
        "ask whether the film resolved it. Tag only when genuinely ambiguous.",
        "Sequel setup with unresolved main conflict (→ CLIFFHANGER_ENDING); franchise-hook coda / "
        "post-credits tease / universe-continues epilogue when THIS film's central conflict is "
        "resolved (franchise continuation is irrelevant); horror ending where the central threat "
        "is contained even if 'evil still exists' lingers (central haunting/possession resolved); "
        "unanswered side questions with the central conflict resolved; ending that is emotionally "
        "unsatisfying but narratively clear; protagonist's fate uncertain in detail but film's "
        "main question answered.",
    )

    SINGLE_LOCATION = (
        "single_location",
        7,
        ConceptTagCategory.NARRATIVE_STRUCTURE,
        "Nearly all action in one physical location. The spatial constraint is a defining "
        "feature of the film's identity.",
        "plot_summary — count distinct locations and judge whether the constraint is identity-"
        "level; craft_observations descriptors like 'bottle movie' / 'one-room drama'.",
        "Mostly one building but significant scenes elsewhere; one location used heavily plus "
        "another substantial setting; haunted-location premise where characters also travel "
        "elsewhere; episodic films revisiting the same location across acts without committing "
        "to it as the entire setting.",
    )

    BREAKING_FOURTH_WALL = (
        "breaking_fourth_wall",
        8,
        ConceptTagCategory.NARRATIVE_STRUCTURE,
        "Characters directly address the audience or acknowledge they're in a movie. A "
        "notable, deliberate, recurring stylistic choice.",
        "additional_narrative_devices terms 'fourth wall break'/'direct address'; plot_keywords "
        "'breaking the fourth wall'; craft_observations singling out direct camera address as a "
        "craft choice.",
        "Voiceover narration without acknowledging audience or camera; documentary-style "
        "interviews; songs commenting on action unless characters explicitly address viewers; "
        "a single brief in-joke aside that is not recurring.",
    )

    CLIFFHANGER_ENDING = (
        "cliffhanger_ending",
        9,
        ConceptTagCategory.NARRATIVE_STRUCTURE,
        "Central conflict INTENTIONALLY unresolved at credits as deliberate setup for a planned "
        "followup — the film stops mid-arc on purpose, deferring resolution to a next chapter "
        "audiences are expected to wait for. Intentionality is the test. Distinct from OPEN_ENDING "
        "(completed arc + thematic ambiguity) and from series films that stand as complete arcs.",
        "plot_summary closing beat where central conflict is unresolved AND continuation is "
        "signaled (antagonist mid-victory, protagonist mid-action, explicit to-be-continued); "
        "plot_keywords explicit cliffhanger framings; emotional_observations + craft_observations / "
        "reviewer commentary describing 'left hanging' / 'demanding the next installment' / "
        "'deliberate sequel setup' / 'edge of seat between films'; release context — known planned "
        "series AND reviewers framing the ending as deliberate setup. Intentionality must be evident: "
        "the film must have chosen to stop mid-arc as part of a planned continuation.",
        "Satisfying resolution where a villain happens to survive or a sequel later becomes "
        "possible (resolution still achieved); central conflict resolved with side threads open; "
        "thematic ambiguity at the end without an unresolved plot question (→ OPEN_ENDING) — "
        "audiences debating what HAPPENED without the film promising continuation isn't a "
        "cliffhanger; part of a series but the individual film stands as a complete arc (series "
        "membership alone doesn't qualify); sequel that exists only because the first film was "
        "successful but the first was written standalone; plot holes / structural ambiguity / "
        "open interpretations that are not deliberate sequel setup.",
    )

    # -- Plot Archetypes (IDs 11-14) -------------------------------------

    REVENGE = (
        "revenge",
        11,
        ConceptTagCategory.PLOT_ARCHETYPES,
        "Vengeance is the primary narrative engine; the protagonist's central goal throughout "
        "is revenge.",
        "plot_keywords 'revenge' directly; conflict_type vengeance-driven framings; plot_summary "
        "where the protagonist's stated goal is 'make them pay'; thematic_concepts naming "
        "'vengeance'/'retribution' as central.",
        "Rescue mission motivated by anger/loss (goal is rescue, not vengeance); justice via "
        "legal means; retaliation subplot inside a different main plot; one act of revenge as "
        "inciting incident for a story that becomes about something else.",
    )

    UNDERDOG = (
        "underdog",
        12,
        ConceptTagCategory.PLOT_ARCHETYPES,
        "Protagonist (or their side/group) is structurally weaker than the opposing force "
        "(resources, numbers, status, ability, social power, perceived prospects) AND the "
        "film's CENTRAL DRAMATIC QUESTION is whether they can prevail. Setting-level asymmetry "
        "qualifies only when the protagonist belongs to the weaker side AND the film's core arc "
        "is their improbable rise; it does NOT qualify when the asymmetry is backdrop for a "
        "different central question (escape, survival, investigation, relationship, revelation).",
        "narrative_archetype terms about outmatched-protagonist rising / against-the-odds / "
        "improbable triumph; plot_keywords naming protagonist or faction as outclassed/outgunned/"
        "outnumbered/improbable; plot_summary framing where the structural disadvantage is the "
        "central dramatic stake (weaker side named as such; tension hinges on whether they prevail); "
        "thematic_concepts describing rise-from-disadvantage / David-vs-Goliath; conflict_type + "
        "conflict_stakes_design placing asymmetric power at the center of the conflict. The "
        "asymmetry must align with 'can the weaker side prevail?' as the film's central question — "
        "world-level power imbalance without that being the central question does not qualify.",
        "Protagonist faces a stronger adversary but competence makes the outcome merely uncertain "
        "rather than improbable; protagonist confronts a more powerful threat but the central "
        "question is escape/survival/revelation/recovery/revenge (NOT 'can the weaker side rise?'); "
        "power asymmetry as world-building/backdrop while the main story is an investigation, "
        "relationship, horror escape, or identity reveal; lone dissenter in an intellectual "
        "disagreement; any conflict with power imbalance unless the improbable-victory question "
        "IS the central story.",
    )

    KIDNAPPING = (
        "kidnapping",
        13,
        ConceptTagCategory.PLOT_ARCHETYPES,
        "A kidnapping/abduction IS the central plot — the movie is about the abduction event "
        "itself and its direct consequences (rescue, escape, ransom).",
        "plot_keywords 'kidnapping'/'abduction'; plot_summary showing the kidnapping as BOTH the "
        "inciting incident AND an ongoing plot driver; parental_guide_items 'Abduction'/"
        "'Kidnapping' at non-trivial severity (corroborating, but only tag when plot_summary "
        "centers the abduction as the engine).",
        "Imprisonment as backstory motivating a different main plot (e.g. revenge); captives as "
        "premise for a different central plot (a chase/escape that IS the story — abduction is "
        "incidental scaffolding); supernatural capture or imprisonment by non-human forces; brief "
        "capture as one event among many; kidnapping resolved early as setup for a different engine.",
    )

    CON_ARTIST = (
        "con_artist",
        14,
        ConceptTagCategory.PLOT_ARCHETYPES,
        "Protagonist is a con artist / grifter / scammer; the movie is about deception as a "
        "craft. Distinct from heist (theft/robbery).",
        "plot_keywords 'con artist'/'grifter'; plot_summary deception-driven plot with con "
        "artistry as the protagonist's mode; thematic_concepts naming 'deception as craft' / "
        "'art of the con' as central.",
        "Character lies/manipulates for personal revenge or survival; villain deceives while "
        "the protagonist plays a different role; single con inside a larger non-con plot; "
        "heist/theft where the crime is property-taking rather than identity-deception.",
    )

    # -- Settings (IDs 21-23) --------------------------------------------

    POST_APOCALYPTIC = (
        "post_apocalyptic",
        21,
        ConceptTagCategory.SETTINGS,
        "Set after civilization's collapse — society has fallen.",
        "plot_keywords 'post apocalypse'; plot_summary establishing a collapsed-civilization "
        "setting as the world the story operates in.",
        "Dystopia where society is intact-but-oppressive (distinct); sci-fi on other planets / "
        "in space not arising from Earth's collapse; localized disaster that hasn't toppled "
        "civilization; near-future stressed-but-functioning societies.",
    )

    HAUNTED_LOCATION = (
        "haunted_location",
        22,
        ConceptTagCategory.SETTINGS,
        "Story centers on a supernaturally haunted location — that place IS the haunting's anchor.",
        "plot_keywords 'haunted house' or named haunted places; plot_summary anchoring "
        "supernatural events to a specific location as the locus of the haunting.",
        "Broader supernatural horror (possessions, curses, mobile ghosts following characters); "
        "scary non-supernaturally-haunted location; place of historical suffering without "
        "supernatural haunting tied to it; entities whose haunting moves with characters rather "
        "than the place.",
    )

    SMALL_TOWN = (
        "small_town",
        23,
        ConceptTagCategory.SETTINGS,
        "Small-town setting is central to the film's identity and atmosphere — the story feels "
        "inseparable from its small-town context.",
        "plot_keywords 'small town' directly; plot_summary explicitly naming the small-town "
        "setting and depending on it for atmosphere, community dynamics, or thematic weight.",
        "Rural area that's not a town; small-town setting as incidental backdrop where the story "
        "could happen anywhere; city suburb; mentions a town's name without the town's character "
        "mattering to the plot.",
    )

    # -- Characters (IDs 31-33) ------------------------------------------

    FEMALE_LEAD = (
        "female_lead",
        31,
        ConceptTagCategory.CHARACTERS,
        "EVERY lead role in the film is held by a female character. Covers a single female sole "
        "protagonist OR an all-female lead group (co-leads / trio / ensemble). Disqualifier: any "
        "male character in a lead role — male sole protagonist, mixed-gender co-leads, ensemble "
        "with any male leads. Female supporting characters in a male-led story do NOT qualify.",
        "Identify lead role(s) from plot_summary (whose decisions drive the plot, whose arc(s) "
        "form the spine). top_billed_cast corroborates — top slots typically = lead role(s). "
        "Determine each lead's gender from named characters + pronouns in plot_summary, plus "
        "plot_keywords. Tag only when EVERY lead — one or many — is female.",
        "Any film with at least one male lead: male sole protagonist; mixed-gender duo/trio of "
        "co-leads; ensemble including any male leads alongside female; male-led story with a "
        "prominent female SUPPORTING character (love interest, family, mentor, sidekick, "
        "antagonist) not herself a lead; female POV character whose decision-driving lead is "
        "male; top-billed actor is male and plot doesn't unambiguously center a different "
        "all-female lead structure.",
        # long_form_instructions: 3-step reasoning block.
        "Two patterns qualify: single female protagonist OR all-female lead group (co-leads / "
        "trio / ensemble). Disqualifier: ANY male lead role.\n"
        "\n"
        "STEP 1 — Identify lead role(s). From plot_summary: whose decisions and transformation "
        "drive the movie? ONE core protagonist whose arc IS the story, OR 2+ co-leads/ensemble "
        "members whose storylines together form the spine? Either qualifies. top_billed_cast "
        "corroborates (top slots typically = lead roles) but plot_summary is primary. Name the "
        "specific characters that constitute the lead role(s).\n"
        "\n"
        "STEP 2 — Determine each lead's gender from named characters, plot_summary pronouns, "
        "and plot_keywords. State each lead + gender explicitly.\n"
        "\n"
        "STEP 3 — All-female test. If every lead is female (one alone, co-leads, or 3+ "
        "ensemble), tag. If any lead is male — male sole protagonist, mixed-gender duo/trio, "
        "ensemble containing any male leads — do NOT tag, even with a prominent female lead "
        "among them. If any lead's gender can't be determined with high confidence, do not tag.",
    )

    ENSEMBLE_CAST = (
        "ensemble_cast",
        32,
        ConceptTagCategory.CHARACTERS,
        "NO single protagonist — the story IS the group's collective arc or an event the group "
        "reacts to. 3+ decision-driving protagonists share roughly equal weight in screen time, "
        "agency, and arc development. Discriminating test (apply FIRST): if you removed any ONE "
        "candidate's storyline, would the film still be a recognizable complete movie? If yes → "
        "NOT ensemble (that character was support). Only tag when removing any candidate would "
        "fundamentally collapse the film.",
        "Apply removal test FIRST — name each candidate, then ask whether the film survives their "
        "removal. After passing, corroborate with: pov_perspective terms 'multiple POVs'/"
        "'rotating POV'; plot_summary showing 3+ characters with independent intertwined arcs of "
        "comparable importance; character_arcs[].reasoning naming multiple developed protagonists; "
        "characterization_methods terms for ensemble work. Long top_billed_cast is a WEAK signal — "
        "five slots is normal for any film with a developed supporting cast; do NOT infer ensemble "
        "from cast size alone.",
        "HERO-WITH-ALLIES — single protagonist surrounded by developed supporting cast (mentor, "
        "love interest, comic relief, sidekick, rival, antagonist) — is NOT ensemble even when "
        "supports are beloved, well-developed, and famously named. The protagonist's journey is "
        "the spine; everyone else orbits. Protagonist with several important supports; parallel "
        "plotlines with one clearly primary; exactly two co-leads of equal weight (ensemble needs "
        "3+); long character list with one clear lead. A long plot_summary naming many characters "
        "does NOT make a film ensemble — count whose DECISIONS drive the plot, not how many are named.",
    )

    ANTI_HERO = (
        "anti_hero",
        33,
        ConceptTagCategory.CHARACTERS,
        "Protagonist is presented as operating outside conventional morality as a DEFINING "
        "trait. Substantive moral boundary-crossing — criminal acts, violence, exploitation, "
        "vigilantism — is or HAS BEEN their primary mode for a meaningful portion of the "
        "runtime, not a single extreme choice under duress. Redemption arcs do NOT disqualify "
        "— original anti-heroism is what the film is about. Discriminating test: in a one-"
        "sentence description, would 'morally compromised'/'criminal'/'outlaw' be central — "
        "either current mode or starting point the story departs from?",
        "Derive from raw behavior in plot_summary, NOT pre-classified upstream labels. Primary: "
        "plot_summary (does the protagonist OPERATE as an anti-hero — criminal acts, "
        "exploitation, vigilantism as default at any point — or principled action with rough "
        "edges throughout?); character_arc_labels from plot_analysis (a transformation FROM a "
        "morally compromised start INTO a moral end-state still indicates anti-hero earlier — "
        "redemption arcs qualify); conflict_type (is the protagonist's stance structurally "
        "outside conventional morality for a meaningful portion?); plot_keywords explicit "
        "anti-hero framings. For ENSEMBLE films, evaluate each decision-driving protagonist "
        "independently — if ANY ONE operates substantively as anti-hero across a meaningful "
        "portion, the film qualifies.",
        "Flawed-but-fundamentally-moral character who does the right thing throughout — moral "
        "compromise must be a substantive operating mode at some point, not just rough edges. "
        "A SINGLE act of extreme moral consequence under impossible duress (parent's tragic "
        "mercy choice, survivor killing to save others, non-criminal driven to violence by an "
        "extraordinary event) does NOT qualify if the rest of the runtime is fundamentally "
        "moral. Parent on a rescue mission acts morally regardless of methods; character "
        "described as 'rebellious'/'rule-breaking' without substantive moral transgression; "
        "minor-rule-breaking teen. Test: does substantive anti-heroic operation occupy a "
        "meaningful portion of the runtime — redemption arc or no.",
    )

    # -- Endings (IDs 41-43, plus classification-only NO_CLEAR_CHOICE=-1) -

    HAPPY_ENDING = (
        "happy_ending",
        41,
        ConceptTagCategory.ENDINGS,
        "Audience leaves feeling positive — satisfaction, relief, triumph, warmth. EMPIRICALLY "
        "DOMINANT for narrative film: most genre cinema (romance, action, adventure, family, "
        "comedy, superhero, horror-with-survival, sports) lands here, INCLUDING films whose "
        "protagonists pay substantial costs along the way. A hard-won victory IS happy. "
        "Cost-along-the-way is the standard structure of a satisfying happy ending, NOT a "
        "disqualifier.",
        "CLOSING SCENE in plot_summary — if it's a celebration beat (triumphant kiss, family "
        "reunion hug, threat-defeated cheer, protagonist-restored shot, platform-raise/"
        "mountaintop moment, smiling embrace), ending is HAPPY regardless of runtime cost. "
        "emotional_observations 'uplifting'/'satisfying'/'triumphant'/'warm closure'/'earned'/"
        "'hard-won'/'achievement at a cost' (the last = happy with sacrifice, NOT bittersweet); "
        "plot_summary final beat + characters' end-state; plot_keywords 'happy ending'. Horror "
        "survived + threat defeated = HAPPY when protagonists are safe and danger is over. "
        "Ambiguous HAPPY-vs-BITTERSWEET → default HAPPY (bittersweet has the higher bar).",
        "Merely surviving a horrific ordeal without positive feeling (relief without triumph/"
        "warmth); victory that feels hollow or Pyrrhic where cost overwhelms the win; positive "
        "plot outcome that lands flat/empty from accumulated grief; protagonists win on paper "
        "but the audience leaves devastated (→ SAD, not happy).",
    )

    SAD_ENDING = (
        "sad_ending",
        42,
        ConceptTagCategory.ENDINGS,
        "Audience leaves feeling predominantly sad — grief, devastation, heartbreak. Lasting "
        "emotion is loss, failure, defeat.",
        "emotional_observations 'devastating'/'tragic'/'heartbreaking'/'bleak'/'left me "
        "sobbing'/'crushing'; plot_summary ending state defined by loss (funeral, destroyed "
        "home, protagonist alone in ruin, unsaved life); closing scene is a grief/defeat/loss "
        "beat with no recuperative upswing. Cliffhanger where heroes have lost and villain "
        "remains at large IS sad — narrative closure not required.",
        "Victory at great cost where audience still feels the victory; emotionally intense "
        "movie with positive outcome; tragic journey ending in redemption, peace, or thematic "
        "uplift; grim mid-story but recuperative ending.",
    )

    BITTERSWEET_ENDING = (
        "bittersweet_ending",
        43,
        ConceptTagCategory.ENDINGS,
        "UNCOMMON ending where the audience leaves with genuinely unresolvable mixed feelings — "
        "cannot comfortably call it 'a happy movie' OR 'a sad movie' and mean it. Discriminating "
        "test: would a reasonable viewer feel something is missing/wrong if you called the "
        "ending HAPPY? AND the same way if you called it SAD? Tag only if BOTH are true. Films "
        "with mixed elements almost always land clearly on HAPPY or SAD — bittersweet is rare "
        "and NOT a fallback for 'complicated'.",
        "emotional_observations that EXPLICITLY describe the AUDIENCE leaving unresolved — "
        "'mixed feelings', 'joy AND sorrow held in tension', 'unable to celebrate fully', 'knot "
        "in the stomach despite the win', 'genuinely torn'. Treat 'earned'/'hard-won'/"
        "'achievement at a cost'/'sacrifice for victory' as HAPPY signals, NOT bittersweet "
        "(standard structure of satisfying happy endings). Closing scene: bittersweet endings "
        "tend toward a quiet contemplative beat (long look, unspoken moment, "
        "what-might-have-been montage, protagonist staring into distance with achievement+loss "
        "both visible) rather than celebration. Triumph kiss/family reunion/threat-defeated "
        "cheer/platform-raise → HAPPY regardless of runtime cost.",
        "Sacrifice along the journey followed by unambiguous celebration/triumph at credits → "
        "HAPPY, full stop. Genre films (romance, action, family, horror-with-survival, "
        "superhero) where protagonists pay a price then win = DEFAULT STRUCTURE of a happy "
        "ending, not bittersweet. Tragic journey ending in redemption/peace/thematic uplift → "
        "SAD or HAPPY by lasting emotion. Mid-runtime grief + recuperative ending → HAPPY. "
        "Structural ambiguity about WHAT happened (open/ambiguous endings) ≠ emotional "
        "ambiguity — route to OPEN_ENDING and pick the emotional tag separately. 'Complicated' "
        "or 'audience had a lot to process' is NOT enough — bittersweet requires that neither "
        "HAPPY nor SAD is a defensible alternative. If you can defend either, pick that one.",
    )

    NO_CLEAR_CHOICE = (
        "no_clear_choice",
        -1,
        ConceptTagCategory.ENDINGS,
        "Evidence ambiguous/insufficient or ending's emotion doesn't fit happy/sad/bittersweet. "
        "Classification-only; filtered out before storage.",
        "Use when extracted observations don't point clearly to one of the above: "
        "emotional_observations describe the ending as 'ambiguous'/'lingering'/'contemplative'/"
        "'philosophical'/'open to interpretation' without clear positive/negative/mixed "
        "valence; plot_summary closing scene is an existential beat (long-held expression, "
        "cosmic-indifference image, unresolved philosophical question) that doesn't map to "
        "celebration/loss/mixed-feelings; craft_observations describe the ending as "
        "deliberately interpretive.",
        "Endings with some complexity but where one of HAPPY/SAD/BITTERSWEET clearly applies "
        "(don't default to NO_CLEAR_CHOICE just because the ending isn't simple); endings with "
        "clear emotional valence even when narratively structurally ambiguous (structural "
        "openness → OPEN_ENDING above; emotional valence → one of HAPPY/SAD/BITTERSWEET here).",
    )

    # -- Experiential (IDs 51-52) ----------------------------------------

    FEEL_GOOD = (
        "feel_good",
        51,
        ConceptTagCategory.EXPERIENTIAL,
        "Overall experience is warm and uplifting — viewer leaves positive, hopeful, lifted up. "
        "About emotional WARMTH, not excitement or adrenaline.",
        "PRIMARY: emotional_observations. Holistic test on the full emotional landscape — is the "
        "dominant tone OVERWHELMINGLY feel-good (warmth, joy, hope, uplift, charm, heartwarming "
        "connection)? Does the description make clear the movie aims to leave audiences hopeful "
        "and lighter? Tag only when warm/hopeful is overwhelming. A small amount of tension/"
        "stakes doesn't disqualify warmth at the destination — a meaningful counterweight of "
        "heavy emotions does.",
        "Emotional landscape with strong presence of NON-feel-good emotions — grief, dread, "
        "devastation, sustained dark tension, prolonged despair, heavy melancholy — alongside "
        "warm beats = MIXED experience, not feel_good. Heavy/devastating movie with a "
        "recuperative ending ≠ feel_good. Pure adrenaline (action thrills, horror scares); "
        "cathartic satisfaction from violent revenge; guilty-pleasure trashy/gory content. Do "
        "NOT infer from genre. When the landscape contains a meaningful heavy counterweight, "
        "not feel_good — this tag is reserved for films whose warm/hopeful/lifting tone is the "
        "overwhelming signal.",
    )

    TEARJERKER = (
        "tearjerker",
        52,
        ConceptTagCategory.EXPERIENTIAL,
        "Makes audiences cry — and the available text DIRECTLY STATES so. Strict literal test: "
        "tag only on a direct statement that audiences cried, sobbed, wept, or shed tears.",
        "PRIMARY/AUTHORITATIVE: emotional_observations. Single literal test — is there a "
        "DIRECT, EXPLICIT statement that audiences cried/sobbed/wept/shed tears/were reduced to "
        "tears? YES → tag. NO → do not tag. Do NOT derive from plot sadness, genre, emotional "
        "intensity, synonyms, or near-equivalents.",
        "Movies described as 'moving'/'touching'/'tugs at heartstrings'/'poignant'/"
        "'devastating'/'heartbreaking'/'emotionally wrecking' or similar that doesn't literally "
        "state audience crying. Plot sadness, tragic events, character deaths, emotional "
        "intensity, or grief themes not explicitly connected to audience crying. The bar is "
        "strict: text must directly describe audiences as having cried — synonyms for "
        "emotional impact alone don't qualify.",
    )

    # -- Content Flags (ID 61) -------------------------------------------

    ANIMAL_DEATH = (
        "animal_death",
        61,
        ConceptTagCategory.CONTENT_FLAGS,
        "A non-human animal (dog, cat, horse, bird, etc.) dies on screen or as a significant "
        "plot point. Exclusively about animals.",
        "TIERED EVIDENCE — apply in order. TIER 1 (PRIMARY): plot_summary + plot_keywords. Any "
        "evidence that a non-human animal dies — death doesn't need to be violent, severe, "
        "central, or extensively depicted. On-screen death, off-screen death described as part "
        "of the story, pet euthanized, animal killed by antagonist, animal killed in passing as "
        "part of a beat — all qualify; tag. TIER 2 (FALLBACK, only when TIER 1 shows NO "
        "evidence): parental_guide_items. If listed at any severity as 'Violence Against "
        "Animals'/'Animal cruelty', tag. Fallback catches animal deaths the plot text doesn't "
        "mention; unrelated parental_guide entries (violence against humans, profanity, "
        "frightening scenes) are NOT a signal either way — only an animal-specific advisory.",
        "Human deaths of any kind; violence against humans; the word 'animal' in unrelated "
        "context; fantasy/sci-fi creatures clearly not real animals; animal mentioned without "
        "dying (pet appearing in scene, working animal in background, meat-industry backdrop "
        "with no on-screen or plot-described death); parental_guide entry for human violence "
        "or general frightening scenes with no animal-specific category.",
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
    PERSON = "person"
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
