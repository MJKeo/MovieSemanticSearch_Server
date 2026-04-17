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

    plot_summary: constr(strip_whitespace=True, min_length=1) | None = None

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

    elevator_pitch: constr(strip_whitespace=True, min_length=1) | None = None
    plot_overview: constr(strip_whitespace=True, min_length=1) | None = None
    genre_signatures: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
    )
    conflict_type: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
    )
    thematic_concepts: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
    )
    character_arcs: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
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

    emotional_palette: TermsWithNegationsSection = Field(
        default_factory=TermsWithNegationsSection,
    )
    tension_adrenaline: TermsWithNegationsSection = Field(
        default_factory=TermsWithNegationsSection,
    )
    tone_self_seriousness: TermsWithNegationsSection = Field(
        default_factory=TermsWithNegationsSection,
    )
    cognitive_complexity: TermsWithNegationsSection = Field(
        default_factory=TermsWithNegationsSection,
    )
    disturbance_profile: TermsWithNegationsSection = Field(
        default_factory=TermsWithNegationsSection,
    )
    sensory_load: TermsWithNegationsSection = Field(
        default_factory=TermsWithNegationsSection,
    )
    emotional_volatility: TermsWithNegationsSection = Field(
        default_factory=TermsWithNegationsSection,
    )
    ending_aftertaste: TermsWithNegationsSection = Field(
        default_factory=TermsWithNegationsSection,
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

    self_experience_motivations: TermsSection = Field(
        default_factory=TermsSection,
    )
    external_motivations: TermsSection = Field(
        default_factory=TermsSection,
    )
    key_movie_feature_draws: TermsSection = Field(
        default_factory=TermsSection,
    )
    watch_scenarios: TermsSection = Field(
        default_factory=TermsSection,
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

    narrative_archetype: TermsSection = Field(default_factory=TermsSection)
    narrative_delivery: TermsSection = Field(default_factory=TermsSection)
    pov_perspective: TermsSection = Field(default_factory=TermsSection)
    characterization_methods: TermsSection = Field(default_factory=TermsSection)
    character_arcs: TermsSection = Field(default_factory=TermsSection)
    audience_character_perception: TermsSection = Field(default_factory=TermsSection)
    information_control: TermsSection = Field(default_factory=TermsSection)
    conflict_stakes_design: TermsSection = Field(default_factory=TermsSection)
    additional_narrative_devices: TermsSection = Field(default_factory=TermsSection)

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
    )
    production_techniques: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
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

    reception_summary: constr(strip_whitespace=True, min_length=1) | None = None
    praised_qualities: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
    )
    criticized_qualities: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
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
