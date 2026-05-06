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

    plot_summary: constr(strip_whitespace=True, min_length=1) | None = Field(
        default=None,
        description=(
            "Synopsis-register prose that mirrors the ingest-side "
            "plot_summary text (past-tense, third-person, no labels). "
            "Two valid shapes depending on what the user asked for:\n"
            "\n"
            "1) SPECIFIC EVENT or event-chain query (e.g. 'a heist "
            "that falls apart due to crew betrayal'): 1-3 dense "
            "sentences restating only the events the user named, "
            "phrased as compact synopsis prose. Do NOT invent "
            "settings, character names, motives, side-events, or "
            "outcomes the user did not supply. Generic agents are "
            "fine ('a heist crew', 'the protagonist'); specific ones "
            "fabricated by you are not.\n"
            "\n"
            "2) MOTIF / element / setting query (e.g. 'clowns as a "
            "recurring motif', 'set in 1940s Berlin'): short "
            "fragments naming the motif or setting in synopsis "
            "contexts, joined by periods. Examples for the clown "
            "motif: 'the clown. is a clown. and then the clown. "
            "encounters a clown. the clown returns.' For setting: "
            "'the story is set in 1940s berlin. takes place during "
            "wartime berlin.' These fragments mirror the actual "
            "phrasings that appear inside real movie synopses — "
            "they retrieve films that contain the motif/setting "
            "WITHOUT fabricating a plot around it.\n"
            "\n"
            "Critical: a motif query must NEVER be expanded into "
            "an invented plot ('a clown chases a woman through a "
            "carnival as her boyfriend tries to save her'). That "
            "shifts the retrieval target away from films the user "
            "actually wants."
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
            "nouns (replace any name with a role: 'a former pilot', "
            "'the protagonist', 'a grieving mother'). Examples: "
            "'parent's love fuels humanity-saving mission', "
            "'investigation reveals escalating truth', 'romance "
            "under external pressure'.\n"
            "\n"
            "CROSS-FIELD REPETITION IS DELIBERATE. The ingest-side "
            "generator prompt explicitly says it should be 'almost "
            "comical' how much the load-bearing thematic concepts "
            "are repeated across elevator_pitch, plot_overview, "
            "thematic_concepts, and character_arcs — that "
            "repetition weights the central concept in the embedded "
            "vector. When you populate this body, REUSE the same "
            "load-bearing terms across these four fields. A grief "
            "body should have 'grief' or 'mourning' appear in "
            "elevator_pitch AND plot_overview AND thematic_concepts "
            "AND (where applicable) character_arcs."
        ),
    )
    plot_overview: constr(strip_whitespace=True, min_length=1) | None = Field(
        default=None,
        description=(
            "1-3 sentence thematic plot summary. Generic, NO proper "
            "nouns (use 'a former pilot' not 'Cooper'; 'a city' not "
            "'New York'). Repeat the load-bearing thematic terms "
            "from elevator_pitch and thematic_concepts deliberately "
            "— this is the cross-field repetition the ingest "
            "generator emits on purpose."
        ),
    )
    genre_signatures: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description=(
            "2-6 short genre-or-subgenre phrases (1-4 words each) "
            "describing the type of story. Sharp compound labels "
            "preferred over broad umbrellas. Examples: 'epic space "
            "odyssey', 'biblical passion drama', 'buddy police "
            "mystery', 'workplace dramedy', 'survival thriller', "
            "'coming-of-age dramedy'."
        ),
    )
    conflict_type: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description=(
            "0-2 generalized 'X vs Y' conflict phrases describing "
            "the dramatic tension. Generic terms applicable across "
            "stories, not specific to this movie's universe. "
            "Examples: 'man vs nature', 'survival of humanity vs "
            "planetary collapse', 'career ambition vs personal "
            "relationships', 'individual truth vs institutional "
            "power'. Empty when the trait does not name a conflict."
        ),
    )
    thematic_concepts: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description=(
            "0-5 thematic concept labels (2-6 words each) capturing "
            "the territory the story explores plus any moral "
            "messages. Generic, human-world terms — not movie-"
            "universe specifics. Examples: 'love as guiding force', "
            "'redemptive sacrificial suffering', 'identity shaped "
            "by image', 'human fragility vs nature'. Reuse the "
            "load-bearing terms from elevator_pitch / plot_overview "
            "here — the ingest side does this on purpose."
        ),
    )
    character_arcs: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description=(
            "0-3 short generalized arc labels (1-4 words). "
            "Examples: 'sacrificial redemption', 'corruption arc', "
            "'coming-of-age', 'from resentment to savior', "
            "'emergent heroism'. Empty when the trait does not "
            "name a character transformation. Reuse load-bearing "
            "thematic vocabulary where the arc embodies the theme."
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
    # redundant near-duplicates per section — synonyms (e.g.
    # "uplifting", "inspiring", "hopeful"), slang ("tearjerker",
    # "gorefest", "white knuckle"), and paraphrases ("kept me
    # guessing", "unpredictable"). First/second-person fragments are
    # fine ("kept me guessing", "made me nauseous").
    #
    # TRUE SYNONYMS ONLY. Substitution test: "could I show this term
    # to the user instead of their original word, and would they say
    # yes, that's the same thing?" If no, drop it. Adjacent concepts
    # that drift the meaning ('haunting' → 'eerie / supernatural';
    # 'bittersweet' → 'tragic') hurt retrieval, not help.
    #
    # ====================================================================
    # CRITICAL: terms and negations BOTH POINT AT THE SAME RETRIEVAL
    # TARGET. They are complementary phrasings of the same concept,
    # NOT opposites:
    #
    #   - terms = "what films matching this body ARE", no "not"/"no" prefix
    #   - negations = "what films matching this body are NOT", with
    #                  "not"/"no" prefix
    #
    # Both fields cluster on the SAME side of the embedding. They
    # reinforce each other.
    #
    # CORRECT pairings (terms and negations point the same way):
    #   - feel-good body:
    #       terms = ["happy", "uplifting", "joyful"]
    #       negations = ["not sad", "not depressing", "not bleak"]
    #     ("happy" and "not sad" are the same idea phrased two ways)
    #   - gory body (looking for gore-heavy films):
    #       terms = ["gory", "bloody", "graphic violence"]
    #       negations = ["not peaceful", "not gentle", "not for kids"]
    #   - non-gory body (looking for restrained films):
    #       terms = ["light scares", "tame violence", "restrained"]
    #       negations = ["no gore", "not too gory", "not bloody"]
    #
    # CONTRADICTORY pairings (DO NOT EMIT):
    #   - terms = ["gory"] + negations = ["not too gory"]   ← contradicts
    #   - terms = ["happy"] + negations = ["not happy"]     ← contradicts
    #   - terms = ["uplifting"] + negations = ["not uplifting"] ← contradicts
    #
    # Field signature is mechanical: terms NEVER carry "not"/"no"
    # prefix; negations ALWAYS do. If you find yourself writing
    # "not too X" inside a terms list, move it to negations and check
    # whether the body's direction matches.
    #
    # NEGATIONS DEFAULT-POPULATE. The ingest text routinely emits 1-3
    # negations per active section even when no boundary was named.
    # Author 1-3 negations per active section that REINFORCE the
    # direction the terms already point — same retrieval target,
    # opposite-syntactic-form ("not"/"no") phrasing. Suppress only
    # when the section is barely populated.
    # ====================================================================
    #
    # Per-field examples below show BOTH directions a section can take
    # (e.g. emotional_palette can be feel-good OR sad-leaning), with
    # paired terms+negations examples for each direction so you can see
    # how they reinforce one another.

    emotional_palette: TermsWithNegationsSection = Field(
        default_factory=TermsWithNegationsSection,
        description=(
            "Dominant felt emotions while watching (5-10 terms "
            "when active).\n"
            "\n"
            "DIRECTION A — feel-good / uplifting:\n"
            "  terms: ['uplifting', 'feel-good', 'heartwarming', "
            "'joyful', 'cozy', 'warm', 'laugh out loud']\n"
            "  negations: ['not sad', 'not depressing', "
            "'not heartbreaking', 'not bleak']\n"
            "\n"
            "DIRECTION B — sad / heavy:\n"
            "  terms: ['heartbreaking', 'tearjerker', "
            "'devastating', 'bittersweet', 'gut-wrenching']\n"
            "  negations: ['not feel-good', 'not uplifting', "
            "'not cheerful', 'not light']\n"
            "\n"
            "Pick the direction the trait calls for and keep "
            "terms+negations BOTH on that side. See class-level "
            "rules for true-synonym discipline and the "
            "same-direction reinforcement principle."
        ),
    )
    tension_adrenaline: TermsWithNegationsSection = Field(
        default_factory=TermsWithNegationsSection,
        description=(
            "Stress / suspense pressure (3-8 terms when active).\n"
            "\n"
            "DIRECTION A — high tension:\n"
            "  terms: ['edge of your seat', 'white knuckle', "
            "'tense the whole time', 'high adrenaline', 'nail "
            "biter']\n"
            "  negations: ['not relaxed', 'not low-stakes', "
            "'not chill']\n"
            "\n"
            "DIRECTION B — low tension / relaxed:\n"
            "  terms: ['relaxed', 'chill', 'low stakes', 'easy "
            "watching']\n"
            "  negations: ['not stressful', 'not anxiety "
            "inducing', 'not edge of your seat']"
        ),
    )
    tone_self_seriousness: TermsWithNegationsSection = Field(
        default_factory=TermsWithNegationsSection,
        description=(
            "The movie's attitude toward itself (3-8 terms when "
            "active).\n"
            "\n"
            "DIRECTION A — earnest / serious:\n"
            "  terms: ['earnest', 'heartfelt', 'grounded', "
            "'sincere', 'serious']\n"
            "  negations: ['not campy', 'not cheesy', 'not "
            "ironic', 'not winking']\n"
            "\n"
            "DIRECTION B — campy / self-aware:\n"
            "  terms: ['campy', 'winking self aware', 'cheesy', "
            "'over the top', 'so bad it\\'s good']\n"
            "  negations: ['not earnest', 'not grounded', 'not "
            "serious', 'not solemn']"
        ),
    )
    cognitive_complexity: TermsWithNegationsSection = Field(
        default_factory=TermsWithNegationsSection,
        description=(
            "Mental effects, ease of follow (3-6 terms when "
            "active).\n"
            "\n"
            "DIRECTION A — cerebral / demanding:\n"
            "  terms: ['cerebral', 'thought provoking', 'mind-"
            "bending', 'requires attention', 'layered']\n"
            "  negations: ['not easygoing', 'not throwaway', "
            "'not mindless']\n"
            "\n"
            "DIRECTION B — easy / digestible:\n"
            "  terms: ['digestible', 'straightforward', 'easy "
            "to follow', 'lightweight']\n"
            "  negations: ['not confusing', 'not hard to "
            "follow', 'not draining', 'not thought provoking']"
        ),
    )
    disturbance_profile: TermsWithNegationsSection = Field(
        default_factory=TermsWithNegationsSection,
        description=(
            "Unsettling / fear flavor (3-8 terms when active; "
            "empty when the movie has no significant disturbing "
            "elements).\n"
            "\n"
            "DIRECTION A — gory / disturbing:\n"
            "  terms: ['gory', 'bloody', 'graphic violence', "
            "'splatter', 'body horror', 'gross', 'disturbing', "
            "'nightmare fuel']\n"
            "  negations: ['not peaceful', 'not gentle', 'not "
            "for kids', 'not family friendly']\n"
            "\n"
            "DIRECTION B — restrained / non-gory:\n"
            "  terms: ['light scares', 'tame violence', "
            "'restrained', 'mild']\n"
            "  negations: ['no gore', 'not too gory', 'not "
            "bloody', 'not graphic', 'not disturbing']\n"
            "\n"
            "DIRECTION C — psychological / dread (no gore):\n"
            "  terms: ['creepy', 'unsettling', 'psychological "
            "horror', 'existential dread', 'paranoia vibes']\n"
            "  negations: ['no gore', 'no jump scares', 'not "
            "splatter']"
        ),
    )
    sensory_load: TermsWithNegationsSection = Field(
        default_factory=TermsWithNegationsSection,
        description=(
            "EXTREME sensory properties only — the kind you'd warn "
            "someone about. >90% of films have this empty. Populate "
            "ONLY for genuinely overwhelming sensory bombardment "
            "(strobe, sustained deafening volume, motion-sickness "
            "camera) or exceptional calmness (meditative pacing, "
            "ambient/ASMR). Standard action / standard horror / "
            "standard spectacle is NOT a sensory_load event — leave "
            "empty.\n"
            "\n"
            "DIRECTION A — overstimulating:\n"
            "  terms: ['eye-straining', 'overstimulating', "
            "'ear-popping', 'sensory overload']\n"
            "  negations: ['not soothing', 'not quiet', "
            "'not calm']\n"
            "\n"
            "DIRECTION B — exceptionally calm:\n"
            "  terms: ['soothing', 'quiet', 'meditative', "
            "'ambient']\n"
            "  negations: ['not loud', 'not overstimulating', "
            "'not bombastic']"
        ),
    )
    emotional_volatility: TermsWithNegationsSection = Field(
        default_factory=TermsWithNegationsSection,
        description=(
            "How the emotional tone changes over time (2-5 terms "
            "when active; empty when tone is consistent).\n"
            "\n"
            "DIRECTION A — volatile / whiplash:\n"
            "  terms: ['tonal whiplash', 'laugh then cry', "
            "'gets dark fast', 'mood swings', 'emotional "
            "rollercoaster']\n"
            "  negations: ['not consistent tone', 'not steady']\n"
            "\n"
            "DIRECTION B — consistent tone:\n"
            "  terms: ['consistent tone', 'steady', 'even keel']\n"
            "  negations: ['no tonal whiplash', 'not all over "
            "the place', 'not a rollercoaster']"
        ),
    )
    ending_aftertaste: TermsWithNegationsSection = Field(
        default_factory=TermsWithNegationsSection,
        description=(
            "Final emotion you leave with (3-6 terms when active; "
            "empty when no narrative evidence about the ending).\n"
            "\n"
            "DIRECTION A — satisfying / happy:\n"
            "  terms: ['satisfying ending', 'earned payoff', "
            "'happy ending', 'feel-good payoff']\n"
            "  negations: ['not a downer ending', 'not "
            "bleak', 'not unsatisfying']\n"
            "\n"
            "DIRECTION B — bleak / haunting:\n"
            "  terms: ['bittersweet ending', 'haunting "
            "ending', 'gut punch ending', 'wrecked me', "
            "'devastating ending']\n"
            "  negations: ['not a happy ending', 'not "
            "uplifting', 'not feel-good']"
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
    # synonyms. Crude/vernacular phrasing matches ingest ("stoned
    # movie", "scared shitless", "cry your eyes out") and should not
    # be sanitized away. NO plot details, character names, or proper
    # nouns. TRUE SYNONYMS ONLY — same substitution test as
    # viewer_experience.

    self_experience_motivations: TermsSection = Field(
        default_factory=TermsSection,
        description=(
            "Self-focused experiential reason someone would seek "
            "out this movie — what emotional/psychological need it "
            "fulfills (4-8 terms when active). Frame from the "
            "viewer's perspective; capture the PURPOSE of the "
            "emotion, not the emotion label itself. Examples: "
            "'need a laugh', 'cathartic watch', 'escape from "
            "reality', 'test my nerves', 'turn my brain off', "
            "'will blow my mind', 'cry-your-eyes-out movie', "
            "'feel small in the universe'."
        ),
    )
    external_motivations: TermsSection = Field(
        default_factory=TermsSection,
        description=(
            "Value the movie provides BEYOND the viewing "
            "experience itself — cultural significance, social "
            "currency, conversation starters, relationship "
            "bonding (2-4 terms when active). Examples: 'sparks "
            "conversation', 'culturally iconic', 'impress film "
            "snobs', 'learn something new', 'good for family "
            "discussion', 'movie for gamers'."
        ),
    )
    key_movie_feature_draws: TermsSection = Field(
        default_factory=TermsSection,
        description=(
            "Standout movie attributes that function as 'watch "
            "this if you want X' draws (2-4 terms when active). "
            "These are interpretations/evaluations of features, "
            "positive or negative. Examples: 'incredible "
            "soundtrack', 'visually stunning', 'compelling "
            "characters', 'hilariously bad dialogue', 'over the "
            "top violence', 'epic orchestral score', "
            "'jack black voice', 'spectacular visuals'."
        ),
    )
    watch_scenarios: TermsSection = Field(
        default_factory=TermsSection,
        description=(
            "Best real-world occasions, contexts, social settings "
            "for this movie — who to watch with, what time of "
            "year, what setting (3-6 terms when active). "
            "Examples: 'date night movie', 'solo movie night', "
            "'cozy night in', 'halloween movie', 'stoned movie', "
            "'background at a party', 'family movie night', "
            "'long flight watch', 'rainy sunday movie'."
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
    # USE CANONICAL CRAFT LABELS VERBATIM. Established technique
    # names — "Chekhov's gun", "dramatic irony", "unreliable
    # narrator", "non-linear timeline", "ticking clock deadline",
    # "found-footage presentation" — are the exact strings the
    # ingest side uses. Do NOT paraphrase them ('non-linear
    # narrative' instead of 'non-linear timeline' is a paraphrase
    # that costs cosine similarity). Movie-agnostic: no character
    # names, actors, places, brands, or unique proper nouns.

    narrative_archetype: TermsSection = Field(
        default_factory=TermsSection,
        description=(
            "Macro story shape — the well-known whole-plot label "
            "(0-1 phrase). Examples: 'cautionary tale', 'underdog "
            "rise', 'revenge spiral', 'quest/adventure', 'tragic "
            "love', 'heist blueprint', 'whodunit mystery'."
        ),
    )
    narrative_delivery: TermsSection = Field(
        default_factory=TermsSection,
        description=(
            "Temporal structure — how time is arranged or "
            "manipulated (0-2 terms). Use canonical labels "
            "VERBATIM: 'linear chronology', 'non-linear "
            "timeline' (NOT 'non-linear narrative'), 'flashback-"
            "driven structure', 'parallel timelines', 'time loop "
            "structure', 'reverse chronology'."
        ),
    )
    pov_perspective: TermsSection = Field(
        default_factory=TermsSection,
        description=(
            "Audience viewpoint and lens reliability (0-2 terms). "
            "Examples: 'first-person pov', 'third-person limited "
            "pov', 'multiple pov switching', 'unreliable narrator' "
            "(canonical — never paraphrase)."
        ),
    )
    characterization_methods: TermsSection = Field(
        default_factory=TermsSection,
        description=(
            "How character is conveyed (0-3 terms). Examples: "
            "'show don't tell actions', 'backstory drip-feed', "
            "'character foil contrast', 'indirect characterization "
            "through dialogue', 'mask slips moments'."
        ),
    )
    character_arcs: TermsSection = Field(
        default_factory=TermsSection,
        description=(
            "How characters change across the story (0-3 terms). "
            "Movie-agnostic technique labels. Examples: "
            "'redemption arc', 'corruption arc', 'coming-of-age "
            "arc', 'disillusionment arc', 'flat arc', "
            "'tragic flaw spiral'."
        ),
    )
    audience_character_perception: TermsSection = Field(
        default_factory=TermsSection,
        description=(
            "Deliberate audience positioning toward characters "
            "(0-3 terms). Often UNDER-utilized — surface it when "
            "the trait grounds it. Examples: 'lovable rogue', "
            "'love-to-hate antagonist', 'morally gray lead', "
            "'sympathetic monster', 'misunderstood outsider'."
        ),
    )
    information_control: TermsSection = Field(
        default_factory=TermsSection,
        description=(
            "Surprise / suspense / misdirection mechanics (0-2 "
            "terms). Use canonical labels VERBATIM: 'plot twist / "
            "reversal', 'dramatic irony', 'red herrings', "
            "'Chekhov's gun', 'slow-burn reveal', 'misdirection "
            "editing'."
        ),
    )
    conflict_stakes_design: TermsSection = Field(
        default_factory=TermsSection,
        description=(
            "How stakes / pressure are built (0-2 terms). Often "
            "UNDER-utilized. Examples: 'ticking clock deadline' "
            "(canonical), 'escalation ladder', 'no-win dilemma', "
            "'forced sacrifice choice', 'Pyrrhic victory'."
        ),
    )
    additional_narrative_devices: TermsSection = Field(
        default_factory=TermsSection,
        description=(
            "Catchall for structural / framing / meta tricks not "
            "captured above (0-4 terms). Often UNDER-utilized — "
            "surface it when applicable. Examples: 'cold open', "
            "'cliffhanger ending', 'framed story', 'story-within-"
            "a-story', 'found-footage presentation', 'epistolary "
            "format', 'chaptered structure', 'anthology segments', "
            "'fourth-wall breaks', 'genre deconstruction'."
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
            "the user's specificity EXACTLY — do NOT add finer "
            "geographic detail than the user supplied. If the user "
            "said 'Iceland', emit ['Iceland'], not ['Reykjavik, "
            "Iceland'] or ['Iceland, Vatnajökull glacier']. Adding "
            "city/landmark detail the user did not name changes the "
            "retrieval target. If the user said 'Tokyo', emit "
            "['Tokyo']. If they said 'Monument Valley, Utah', emit "
            "that. The ingest side stores raw IMDB filming-location "
            "strings (city + country), but query-side specificity "
            "should be capped at what the user supplied."
        ),
    )
    production_techniques: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description=(
            "Concrete making/rendering/capture method terms (0-2 "
            "typical). Animation modalities and capture methods: "
            "'hand-drawn animation', 'computer animation', "
            "'cgi animation', 'stop-motion', 'rotoscope', "
            "'motion-capture', '3d animation', '2d animation', "
            "'black-and-white', 'single-take', 'long take', "
            "'handheld-camera', 'found-footage'. Empty for any "
            "conventional live-action film without distinctive "
            "production technique."
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
            "was received, in third-person ('audiences praise X, "
            "while some criticize Y'). Compact and specific — no "
            "filler. Use only when the trait names whole-work "
            "reception shape (cult, classic, divisive, era-"
            "defining). Aspect-level praise/criticism goes in the "
            "two term lists below."
        ),
    )
    praised_qualities: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description=(
            "Aspect-level praise tags (3-6 terms when active). "
            "ADJECTIVE+NOUN shape, 1-3 words each. Describe "
            "filmmaking EXECUTION, not subject matter ('sharp "
            "dialogue' yes, 'engaging debate' no; 'inventive "
            "structure' yes, 'intriguing premise' no). Movie-"
            "agnostic — no proper nouns. Examples: 'spectacular "
            "cinematography', 'evocative score', 'compelling "
            "performances', 'sharp dialogue', 'ambitious themes', "
            "'vibrant animation', 'expanded worldbuilding'."
        ),
    )
    criticized_qualities: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description=(
            "Aspect-level criticism tags (3-6 terms when active). "
            "Same adjective+noun shape and execution-focus rule "
            "as praised_qualities. Examples: 'convoluted plot', "
            "'slow pacing', 'thin character development', "
            "'predictable plotting', 'derivative sequel', "
            "'questionable scientific clarity'."
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
