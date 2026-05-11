# Query-side Body classes for the step 3 semantic endpoint.
#
# Each Body class mirrors the *embeddable* fields of an ingestion-side
# *Output schema (or the per-space vector text function in
# movie_ingestion/final_ingestion/vector_text.py when the space is
# assembled from multiple metadata sources). Each exposes an
# embedding_text() method that reproduces the ingestion-side vector
# text sequence exactly so query-side vectors and document-side
# vectors land in the same token neighborhood.
#
# These Bodies drop ingestion-side chain-of-thought fields
# (justification, reasoning, identity_note, extraction-zone fields,
# etc.) because the query-side LLM carries its own CoT on the wrapping
# SemanticDealbreakerSpec / SemanticPreferenceSpec classes in
# schemas/semantic_translation.py.
#
# Query-side bodies also intentionally avoid required content fields.
# Some spaces are prose-first on the ingestion side, but forcing prose
# at query time would pressure the model to pad with filler when the
# true signal is sparse. Step 1/2 should normally prevent fully empty
# bodies from being produced, but the schema should not force the LLM
# to invent content just to satisfy validation.
#
# Duplication against ingestion-side embedding_text() is deliberate.
# Factoring the shared formatting into one location would couple
# unrelated concerns; keeping the duplication visible in code review
# guards against silent divergence that would degrade cross-space
# cosine alignment without a hard failure.
#
# Field descriptions in this module are deliberately definitional
# rather than enumerative. Listing candidate terms inside a field
# description recruits the LLM to copy them as if they were a menu,
# regardless of whether the user's trait actually grounded those
# concepts — a failure mode observed in production (e.g. a request
# for one archetype producing 2-3 adjacent archetypes from the field's
# example list). Grounding discipline and register rules live in the
# semantic endpoint prompt (search_v2/endpoint_fetching/category_
# handlers/prompts/endpoints/semantic.md) and in per-category
# additional_objective_notes.
#
# See search_improvement_planning/finalized_search_proposal.md
# (Endpoint 6: Semantic) for the full design rationale.

from pydantic import BaseModel, ConfigDict, Field, constr

from implementation.misc.helpers import normalize_string


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------
#
# Names mirror the ingestion-side sub-models in schemas/metadata.py
# (TermsSection, TermsWithNegationsSection). Structurally identical
# minus the justification field; kept distinct so a future divergence
# on either side is local rather than cross-cutting.


class TermsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    terms: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
    )


class TermsWithNegationsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    terms: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
    )
    negations: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
    )


# ---------------------------------------------------------------------------
# Anchor
# ---------------------------------------------------------------------------
#
# Mirrors create_anchor_vector_text in
# movie_ingestion/final_ingestion/vector_text.py (lines 65–136) but
# deliberately omits title, original_title, and maturity_summary —
# ingestion-side identity/filter signals that the query-side LLM has
# no business generating. Term lists pass through normalize_string;
# prose fields are lowercased; lines are joined with "\n".
class AnchorBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identity_pitch: constr(strip_whitespace=True, min_length=1) | None = None
    identity_overview: constr(strip_whitespace=True, min_length=1) | None = None
    genre_signatures: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
    )
    themes: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
    )
    emotional_palette: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
    )
    key_draws: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
    )
    reception_summary: constr(strip_whitespace=True, min_length=1) | None = None

    def embedding_text(self) -> str:
        parts: list[str] = []

        if self.identity_pitch:
            parts.append(f"identity_pitch: {self.identity_pitch.lower()}")

        if self.identity_overview:
            parts.append(f"identity_overview: {self.identity_overview.lower()}")

        if self.genre_signatures:
            parts.append("genre_signatures: " + ", ".join(
                normalize_string(g) for g in self.genre_signatures
            ))

        if self.themes:
            parts.append("themes: " + ", ".join(
                normalize_string(t) for t in self.themes
            ))

        if self.emotional_palette:
            parts.append("emotional_palette: " + ", ".join(
                normalize_string(t) for t in self.emotional_palette
            ))

        if self.key_draws:
            parts.append("key_draws: " + ", ".join(
                normalize_string(t) for t in self.key_draws
            ))

        if self.reception_summary:
            parts.append(f"reception_summary: {self.reception_summary.lower()}")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Plot Events
# ---------------------------------------------------------------------------
#
# Mirrors PlotEventsOutput.embedding_text in schemas/metadata.py
# (lines 334–335). Ingestion-side plot_events is raw prose with no
# label prefix — just the lowercased plot summary.
class PlotEventsBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plot_summary: constr(strip_whitespace=True, min_length=1) | None = Field(
        default=None,
        description=(
            "Synopsis-register prose that mirrors the ingest-side "
            "plot_summary text (past-tense, third-person, no labels). "
            "Two valid shapes depending on what the user grounded:\n"
            "\n"
            "1) SPECIFIC EVENT or event-chain query: 1-3 dense "
            "sentences of compact synopsis prose that restate ONLY "
            "the events the user named. Generic agents are acceptable "
            "in place of specifics ('a heist crew', 'the "
            "protagonist'); never invent settings, character names, "
            "motives, side-events, or outcomes the user did not "
            "supply.\n"
            "\n"
            "2) MOTIF / element / setting query: short synopsis-prose "
            "fragments naming the motif or setting in the syntactic "
            "positions a real movie synopsis would use (the noun "
            "standing bare; the noun as subject of a generic action; "
            "the noun recurring), joined by periods. Mirrors the "
            "phrasings that appear inside real synopses so cosine "
            "alignment lands on films that contain the motif or "
            "setting WITHOUT fabricating a plot around it.\n"
            "\n"
            "Critical: a motif query is NEVER expanded into an "
            "invented plot. Inventing context around the motif "
            "shifts the retrieval target away from the films the "
            "user actually wants."
        ),
    )

    def embedding_text(self) -> str:
        return self.plot_summary.lower() if self.plot_summary else ""


# ---------------------------------------------------------------------------
# Plot Analysis
# ---------------------------------------------------------------------------
#
# Mirrors PlotAnalysisOutput.embedding_text in schemas/metadata.py
# (lines 600–652). Ingestion-side label for conflict_type is emitted
# as "conflict:" — this Body matches that label exactly. Drops the
# ElevatorPitchWithJustification / ThematicConceptWithJustification /
# CharacterArcWithReasoning wrappers; the query side carries raw
# strings since its CoT scaffolding lives on the top-level spec.
class PlotAnalysisBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    elevator_pitch: constr(strip_whitespace=True, min_length=1) | None = Field(
        default=None,
        description=(
            "Log-line capsule, ≤6 words ideal. Generic, no proper "
            "nouns (replace any name with a role — a former pilot, "
            "the protagonist, a grieving mother).\n"
            "\n"
            "CROSS-FIELD REPETITION IS DELIBERATE. The ingest-side "
            "generator deliberately repeats load-bearing thematic "
            "terms across elevator_pitch / plot_overview / "
            "thematic_concepts / character_arcs to weight the "
            "central concept in the embedded vector. Reuse the same "
            "load-bearing terms across those four fields when "
            "populating this body."
        ),
    )
    plot_overview: constr(strip_whitespace=True, min_length=1) | None = Field(
        default=None,
        description=(
            "1-3 sentence thematic plot summary. Generic, NO proper "
            "nouns (use roles in place of names; classes of place in "
            "place of specific cities). Repeat the load-bearing "
            "thematic terms from elevator_pitch and thematic_concepts "
            "deliberately — this is the cross-field repetition the "
            "ingest generator emits on purpose."
        ),
    )
    genre_signatures: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description=(
            "2-6 short genre-or-subgenre phrases (1-4 words each) "
            "describing the type of story. Prefer sharp compound "
            "labels over broad umbrella labels. Restate only genre "
            "signatures the trait grounded — do not pad with "
            "adjacent-but-unnamed genre flavor."
        ),
    )
    conflict_type: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description=(
            "0-2 generalized 'X vs Y' conflict phrases describing "
            "the dramatic tension the trait grounded. Generic terms "
            "applicable across stories, not specific to this movie's "
            "universe. Empty when the trait does not name a "
            "conflict."
        ),
    )
    thematic_concepts: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description=(
            "0-5 thematic concept labels (2-6 words each) capturing "
            "the territory the story explores plus any moral "
            "messages named in the trait. Generic, human-world "
            "terms — not movie-universe specifics. Reuse the "
            "load-bearing terms from elevator_pitch / plot_overview "
            "here — the ingest side does this on purpose."
        ),
    )
    character_arcs: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description=(
            "0-3 short generalized arc labels (1-4 words). Empty "
            "when the trait does not name a character "
            "transformation. Reuse load-bearing thematic vocabulary "
            "where the arc embodies the theme."
        ),
    )

    def embedding_text(self) -> str:
        parts: list[str] = []

        # Prose fields — lowercased only; punctuation is meaningful.
        if self.elevator_pitch:
            parts.append(f"elevator_pitch: {self.elevator_pitch.lower()}")
        if self.plot_overview:
            parts.append(f"plot_overview: {self.plot_overview.lower()}")

        # Enumerated categorical fields — each term normalized.
        if self.genre_signatures:
            parts.append("genre_signatures: " + ", ".join(
                normalize_string(g) for g in self.genre_signatures
            ))
        if self.conflict_type:
            parts.append("conflict: " + ", ".join(
                normalize_string(c) for c in self.conflict_type
            ))
        if self.thematic_concepts:
            parts.append("themes: " + ", ".join(
                normalize_string(t) for t in self.thematic_concepts
            ))
        if self.character_arcs:
            parts.append("character_arcs: " + ", ".join(
                normalize_string(arc) for arc in self.character_arcs
            ))

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Viewer Experience
# ---------------------------------------------------------------------------
#
# Mirrors ViewerExperienceOutput.embedding_text in schemas/metadata.py
# (lines 726–751). Eight sections in fixed order; each emits a
# "<label>: <terms>" line when terms are non-empty and a
# "<label>_negations: <negations>" line when negations are non-empty.
class ViewerExperienceBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Voice for ALL viewer_experience sub-fields — applies to every
    # field below (referenced from per-field descriptions to avoid
    # restating the rule eight times).
    #
    # Phrases are written like search queries, not sentences. Short:
    # 1-5 words ideal. Common everyday user wording, not academic
    # critic-prose. The ingest generator deliberately includes
    # redundant near-duplicates per section (synonyms, slang,
    # paraphrases) so query-side bodies should too. First/second-
    # person fragments are fine.
    #
    # TRUE SYNONYMS ONLY. Substitution test: "could I show this term
    # to the user instead of their original word, and would they say
    # yes, that's the same thing?" If no, drop it. Adjacent concepts
    # that drift the meaning hurt retrieval rather than help it.
    #
    # ====================================================================
    # CRITICAL: terms and negations BOTH POINT AT THE SAME RETRIEVAL
    # TARGET. They are complementary phrasings of the same concept,
    # NOT opposites:
    #
    #   - terms = "what films matching this body ARE", no "not"/"no"
    #       prefix
    #   - negations = "what films matching this body are NOT", with
    #       "not"/"no" prefix
    #
    # Both fields cluster on the SAME side of the embedding. They
    # reinforce each other — an affirmative term and its "not <opposite>"
    # negation are the same idea phrased two ways.
    #
    # Field signature is mechanical: terms NEVER carry "not"/"no"
    # prefix; negations ALWAYS do. If you find yourself writing
    # "not too X" inside a terms list, move it to negations and check
    # whether the body's direction matches. Contradictory pairings
    # (affirming a term and then negating that same term in negations)
    # are never emitted.
    #
    # NEGATIONS DEFAULT-POPULATE. The ingest text routinely emits 1-3
    # negations per active section even when no boundary was named.
    # Author 1-3 negations per active section that REINFORCE the
    # direction the terms already point — same retrieval target,
    # opposite-syntactic-form ("not"/"no") phrasing. Suppress only
    # when the section is barely populated.
    # ====================================================================

    emotional_palette: TermsWithNegationsSection = Field(
        default_factory=TermsWithNegationsSection,
        description=(
            "Dominant felt emotions while watching (5-10 terms when "
            "active). Pick the affective direction the trait calls "
            "for and keep terms+negations BOTH on that side. See "
            "class-level rules for true-synonym discipline and "
            "same-direction reinforcement."
        ),
    )
    tension_adrenaline: TermsWithNegationsSection = Field(
        default_factory=TermsWithNegationsSection,
        description=(
            "Stress / suspense pressure (3-8 terms when active). "
            "Pick the direction the trait calls for and keep "
            "terms+negations on that side."
        ),
    )
    tone_self_seriousness: TermsWithNegationsSection = Field(
        default_factory=TermsWithNegationsSection,
        description=(
            "The movie's attitude toward itself (3-8 terms when "
            "active). Pick the direction the trait calls for and "
            "keep terms+negations on that side."
        ),
    )
    cognitive_complexity: TermsWithNegationsSection = Field(
        default_factory=TermsWithNegationsSection,
        description=(
            "Mental effects and ease of follow (3-6 terms when "
            "active). Pick the direction the trait calls for and "
            "keep terms+negations on that side."
        ),
    )
    disturbance_profile: TermsWithNegationsSection = Field(
        default_factory=TermsWithNegationsSection,
        description=(
            "Unsettling / fear flavor (3-8 terms when active; empty "
            "when the movie has no significant disturbing elements). "
            "Pick the specific flavor the trait calls for (gory vs "
            "restrained vs psychological-dread, etc.) and keep "
            "terms+negations on that side."
        ),
    )
    sensory_load: TermsWithNegationsSection = Field(
        default_factory=TermsWithNegationsSection,
        description=(
            "EXTREME sensory properties only — the kind you would "
            "warn someone about. More than 90% of films have this "
            "empty. Populate ONLY for genuinely overwhelming sensory "
            "bombardment (strobe, sustained deafening volume, "
            "motion-sickness camera) or exceptional calmness "
            "(meditative pacing, ambient/ASMR). Standard action, "
            "standard horror, and standard spectacle are NOT "
            "sensory_load events — leave empty. When the trait does "
            "ground it, pick the direction and keep terms+negations "
            "on that side."
        ),
    )
    emotional_volatility: TermsWithNegationsSection = Field(
        default_factory=TermsWithNegationsSection,
        description=(
            "How the emotional tone changes over time (2-5 terms "
            "when active; empty when tone is consistent or "
            "ungrounded in the trait). Pick the direction the trait "
            "calls for and keep terms+negations on that side."
        ),
    )
    ending_aftertaste: TermsWithNegationsSection = Field(
        default_factory=TermsWithNegationsSection,
        description=(
            "Final emotion the viewer leaves with (3-6 terms when "
            "active; empty when the trait gives no signal about the "
            "ending). Pick the direction the trait calls for and "
            "keep terms+negations on that side."
        ),
    )

    def embedding_text(self) -> str:
        parts: list[str] = []

        labeled_sections = (
            ("emotional_palette", self.emotional_palette),
            ("tension_adrenaline", self.tension_adrenaline),
            ("tone_self_seriousness", self.tone_self_seriousness),
            ("cognitive_complexity", self.cognitive_complexity),
            ("disturbance_profile", self.disturbance_profile),
            ("sensory_load", self.sensory_load),
            ("emotional_volatility", self.emotional_volatility),
            ("ending_aftertaste", self.ending_aftertaste),
        )

        for label, section in labeled_sections:
            if section.terms:
                normalized_terms = ", ".join(
                    normalize_string(term) for term in section.terms
                )
                parts.append(f"{label}: {normalized_terms}")
            if section.negations:
                normalized_negations = ", ".join(
                    normalize_string(term) for term in section.negations
                )
                parts.append(f"{label}_negations: {normalized_negations}")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Watch Context
# ---------------------------------------------------------------------------
#
# Mirrors WatchContextOutput.embedding_text in schemas/metadata.py
# (lines 858–874). Four sections in fixed order; skips empty
# sections. The ingestion-side identity_note field is explicitly
# excluded from embedding text and has no query-side counterpart.
class WatchContextBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Voice for ALL watch_context sub-fields:
    # Search-query phrases, not sentences. Short: 1-6 words. Common
    # everyday user wording, intent-framed (from the viewer's
    # perspective). Redundant near-duplicates encouraged for true
    # synonyms only — the ingest side deliberately includes synonym
    # density that query-side bodies should mirror. Crude/vernacular
    # phrasing matches ingest and should not be sanitized away. NO
    # plot details, character names, or proper nouns. Same
    # substitution test as viewer_experience.

    self_experience_motivations: TermsSection = Field(
        default_factory=TermsSection,
        description=(
            "Self-focused experiential reason someone would seek "
            "out this movie — what emotional or psychological need "
            "it fulfills (4-8 terms when active). Frame from the "
            "viewer's perspective; capture the PURPOSE of the "
            "emotion, not the emotion label itself."
        ),
    )
    external_motivations: TermsSection = Field(
        default_factory=TermsSection,
        description=(
            "Value the movie provides BEYOND the viewing experience "
            "itself — cultural significance, social currency, "
            "conversation starters, relationship bonding (2-4 terms "
            "when active)."
        ),
    )
    key_movie_feature_draws: TermsSection = Field(
        default_factory=TermsSection,
        description=(
            "Standout movie attributes that function as 'watch this "
            "if you want X' draws (2-4 terms when active). "
            "Interpretations or evaluations of features, positive or "
            "negative."
        ),
    )
    watch_scenarios: TermsSection = Field(
        default_factory=TermsSection,
        description=(
            "Best real-world occasions, contexts, or social settings "
            "for this movie — who to watch with, what time of year, "
            "what setting (3-6 terms when active)."
        ),
    )

    def embedding_text(self) -> str:
        parts: list[str] = []

        sections = (
            ("self_experience_motivations", self.self_experience_motivations.terms),
            ("external_motivations", self.external_motivations.terms),
            ("key_movie_feature_draws", self.key_movie_feature_draws.terms),
            ("watch_scenarios", self.watch_scenarios.terms),
        )
        for label, terms in sections:
            if not terms:
                continue
            normalized_terms = [normalize_string(term) for term in terms]
            parts.append(f"{label}: {', '.join(normalized_terms)}")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Narrative Techniques
# ---------------------------------------------------------------------------
#
# Mirrors NarrativeTechniquesOutput.embedding_text in
# schemas/metadata.py (lines 924–950). Nine sections in fixed order;
# skips empty sections.
class NarrativeTechniquesBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Voice for ALL narrative_techniques sub-fields:
    # Short tag-shaped phrases (1-6 words). 1-3 terms per active
    # section is typical; rich queries touch 4-6 of the 9 sub-fields,
    # not just one or two.
    #
    # USE CANONICAL CRAFT LABELS VERBATIM when the trait names a
    # recognized technique. Established craft-name spellings are the
    # exact strings the ingest side uses; paraphrases of canonical
    # names cost cosine similarity. Movie-agnostic: no character
    # names, actors, places, brands, or unique proper nouns.
    #
    # NO SYNONYM PADDING. Unlike viewer_experience and watch_context,
    # narrative_techniques does NOT benefit from redundant near-
    # duplicate terms — each term in this space names a distinct
    # technique. Emit one term per technique the trait grounded; do
    # NOT add adjacent-but-different techniques as "synonyms".

    narrative_archetype: TermsSection = Field(
        default_factory=TermsSection,
        description=(
            "Macro story shape — the well-known whole-plot label "
            "(0-1 phrase). Emit only when the trait names a "
            "recognizable macro shape."
        ),
    )
    narrative_delivery: TermsSection = Field(
        default_factory=TermsSection,
        description=(
            "Temporal structure — how time is arranged or "
            "manipulated (0-2 terms). Use the canonical craft label "
            "verbatim when the technique has an established name. "
            "Emit only when the trait grounds the temporal "
            "structure."
        ),
    )
    pov_perspective: TermsSection = Field(
        default_factory=TermsSection,
        description=(
            "Audience viewpoint and lens reliability (0-2 terms). "
            "Emit only when the trait grounds a perspective choice "
            "or reliability framing. Use the canonical craft label "
            "verbatim."
        ),
    )
    characterization_methods: TermsSection = Field(
        default_factory=TermsSection,
        description=(
            "How character is conveyed (0-3 terms). Emit only when "
            "the trait grounds a characterization technique. Use the "
            "canonical craft label verbatim where one exists."
        ),
    )
    character_arcs: TermsSection = Field(
        default_factory=TermsSection,
        description=(
            "How characters change across the story (0-3 terms). "
            "Movie-agnostic technique labels. Emit only when the "
            "trait grounds a character-transformation pattern."
        ),
    )
    audience_character_perception: TermsSection = Field(
        default_factory=TermsSection,
        description=(
            "Deliberate audience positioning toward characters "
            "(0-3 terms). Emit ONLY the audience-positioning the "
            "trait grounded. Each archetype names a distinct "
            "audience-character relationship; do not enumerate "
            "adjacent archetypes the trait did not name."
        ),
    )
    information_control: TermsSection = Field(
        default_factory=TermsSection,
        description=(
            "Surprise / suspense / misdirection mechanics (0-2 "
            "terms). Use the canonical craft label verbatim. Emit "
            "only when the trait grounds the mechanic."
        ),
    )
    conflict_stakes_design: TermsSection = Field(
        default_factory=TermsSection,
        description=(
            "How stakes / pressure are built (0-2 terms). Emit only "
            "when the trait grounds the stakes-design pattern. Use "
            "the canonical craft label verbatim where one exists."
        ),
    )
    additional_narrative_devices: TermsSection = Field(
        default_factory=TermsSection,
        description=(
            "Catchall for structural / framing / meta tricks not "
            "captured above (0-4 terms). Emit only when the trait "
            "grounds a structural device. Use the canonical craft "
            "label verbatim where one exists."
        ),
    )

    def embedding_text(self) -> str:
        parts: list[str] = []

        sections = (
            ("narrative_archetype", self.narrative_archetype.terms),
            ("narrative_delivery", self.narrative_delivery.terms),
            ("pov_perspective", self.pov_perspective.terms),
            ("characterization_methods", self.characterization_methods.terms),
            ("character_arcs", self.character_arcs.terms),
            (
                "audience_character_perception",
                self.audience_character_perception.terms,
            ),
            ("information_control", self.information_control.terms),
            ("conflict_stakes_design", self.conflict_stakes_design.terms),
            (
                "additional_narrative_devices",
                self.additional_narrative_devices.terms,
            ),
        )
        for label, terms in sections:
            if not terms:
                continue
            normalized_terms = [normalize_string(term) for term in terms]
            parts.append(f"{label}: {', '.join(normalized_terms)}")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Production
# ---------------------------------------------------------------------------
#
# Mirrors create_production_vector_text in
# movie_ingestion/final_ingestion/vector_text.py (lines 258–277).
# Emits filming_locations line (comma-joined, lowercased) then
# production_techniques line (comma-joined, per-term normalized to
# match ingestion's ProductionTechniquesOutput.embedding_text).
# Skips either line if its list is empty.
#
# Does NOT enforce the is_animation() gate or the [:3] cap from
# ingestion — both are ingest-time data hygiene. At query time the
# LLM is expected to emit only relevant filming locations, and if it
# emits more than 3 that's an intentional signal worth preserving.
class ProductionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filming_locations: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description=(
            "Place names where the film was physically shot. Match "
            "the user's geographic specificity EXACTLY — do NOT add "
            "finer detail than the user supplied. Adding city or "
            "landmark detail the user did not name changes the "
            "retrieval target. The ingest side stores raw IMDB "
            "filming-location strings (city + country), but query-"
            "side specificity is capped at what the user supplied."
        ),
    )
    production_techniques: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description=(
            "Concrete making / rendering / capture method terms "
            "(0-2 typical). Empty for any conventional live-action "
            "film without a distinctive production technique. Use "
            "the canonical craft term verbatim where one exists. "
            "Emit only when the trait grounds the technique."
        ),
    )

    def embedding_text(self) -> str:
        parts: list[str] = []

        if self.filming_locations:
            locations = ", ".join(self.filming_locations).lower()
            parts.append(f"filming_locations: {locations}")

        if self.production_techniques:
            techniques_text = ", ".join(
                normalize_string(t) for t in self.production_techniques
            )
            parts.append(f"production_techniques: {techniques_text}")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Reception
# ---------------------------------------------------------------------------
#
# Mirrors ReceptionOutput.embedding_text in schemas/metadata.py
# (lines 429–439). Emits reception_summary (prose, required), then
# praised and criticized lines when those lists are non-empty.
# Does NOT emit the major_award_wins line — that's appended from
# structured award data at ingest time via _reception_award_wins_text,
# not from LLM generation.
class ReceptionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reception_summary: constr(strip_whitespace=True, min_length=1) | None = Field(
        default=None,
        description=(
            "1-2 sentence evaluative prose summary of how the film "
            "was received, in third-person. Compact and specific — "
            "no filler. Use only when the trait names a whole-work "
            "reception shape (overall critical reception, cultural "
            "status, divisiveness pattern). Aspect-level praise or "
            "criticism goes in the two term lists below."
        ),
    )
    praised_qualities: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description=(
            "Aspect-level praise tags (3-6 terms when active). "
            "Adjective+noun shape, 1-3 words each. Describe "
            "filmmaking CRAFT EXECUTION (writing, directing, "
            "performances, cinematography, score, editing, design) "
            "— NOT subject matter or premise interest. Movie-"
            "agnostic — no proper nouns. Emit only the aspects the "
            "trait grounded."
        ),
    )
    criticized_qualities: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description=(
            "Aspect-level criticism tags (3-6 terms when active). "
            "Same adjective+noun shape and craft-execution-focus "
            "rule as praised_qualities. Emit only the aspects the "
            "trait grounded."
        ),
    )

    def embedding_text(self) -> str:
        parts: list[str] = []

        if self.reception_summary:
            parts.append(f"reception_summary: {self.reception_summary.lower()}")

        if self.praised_qualities:
            praised = ", ".join(
                normalize_string(q) for q in self.praised_qualities
            )
            parts.append(f"praised: {praised}")

        if self.criticized_qualities:
            criticized = ", ".join(
                normalize_string(q) for q in self.criticized_qualities
            )
            parts.append(f"criticized: {criticized}")

        return "\n".join(parts)
