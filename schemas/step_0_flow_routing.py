# Step 0 (flow routing) LLM structured output models.
#
# Step 0 is a narrow classifier that decides which of the entity-bearing
# search flows (specific title, similarity to titles, character franchise,
# non-character franchise, studio, actor) should execute for a given user
# query, or whether none of them apply. It runs in parallel with Step 1
# on the raw query; the merge happens in code afterward.
#
# Studio and actor are list-style entity flows: like similarity_to_titles,
# they accept one or more entities in `selected_entities` (e.g. a query
# of "tom hanks and woody harrelson" fires the actor flow with two
# EntityReference entries). The studio/actor flows currently route to
# no-op handlers — the executors are not yet implemented.
#
# The schema follows the FACTS → DECISION pattern:
#   Zone 1 — extractive observations (entity candidates, qualifiers)
#   Zone 2 — entity-flow decision (single enum + payload)
#   Zone 3 — standard-flow co-fire decision (ambiguity reasoning + bool)
#
# Standard-flow firing and primary_flow are DERIVED in code from the
# LLM's outputs (see Step0Response.fire_standard_flow / .primary_flow).
# The LLM is never asked to produce them directly.
#
# Authoring convention note: this module uses the Rich-fields variant of
# ADR-036 — the EntityKind taxonomy is taught via Field descriptions on
# `most_likely_kind` rather than only in the system prompt. This is a
# deliberate flip from the prior version of this file, which followed
# the Lean-fields convention. The rich field description is the densest
# teaching surface for the per-kind decision that drives the entire
# routing outcome.

from pydantic import BaseModel, ConfigDict, Field, constr, model_validator

from schemas.enums import SearchFlow


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

# EntityKind classifies a single observed entity span. This is the per-
# candidate classification step — the global selected_entity_flow is
# synthesized across the candidate list.
#
# A separate EntityFlow enum (below) carries the global routing choice
# because it adds two members EntityKind does not have: SIMILARITY_TO_TITLES
# (which is a frame over title candidates, not a candidate kind) and
# NONE_OF_THE_ABOVE (which is the "no entity match" terminal — only
# meaningful at the global decision layer).
from enum import StrEnum


class EntityKind(StrEnum):
    SPECIFIC_TITLE = "specific_title"
    CHARACTER_FRANCHISE = "character_franchise"
    NON_CHARACTER_FRANCHISE = "non_character_franchise"
    STUDIO = "studio"
    ACTOR = "actor"


class EntityFlow(StrEnum):
    SPECIFIC_TITLE = "specific_title"
    SIMILARITY_TO_TITLES = "similarity_to_titles"
    CHARACTER_FRANCHISE = "character_franchise"
    NON_CHARACTER_FRANCHISE = "non_character_franchise"
    STUDIO = "studio"
    ACTOR = "actor"
    NONE_OF_THE_ABOVE = "none_of_the_above"


# Mapping from the LLM-facing EntityFlow enum to the downstream-facing
# SearchFlow enum. NONE_OF_THE_ABOVE has no SearchFlow counterpart —
# primary_flow falls back to STANDARD in that case.
_ENTITY_FLOW_TO_SEARCH_FLOW: dict[EntityFlow, SearchFlow] = {
    EntityFlow.SPECIFIC_TITLE: SearchFlow.EXACT_TITLE,
    EntityFlow.SIMILARITY_TO_TITLES: SearchFlow.SIMILARITY,
    EntityFlow.CHARACTER_FRANCHISE: SearchFlow.CHARACTER_FRANCHISE,
    EntityFlow.NON_CHARACTER_FRANCHISE: SearchFlow.NON_CHARACTER_FRANCHISE,
    EntityFlow.STUDIO: SearchFlow.STUDIO,
    EntityFlow.ACTOR: SearchFlow.ACTOR,
}


# Mapping from EntityFlow to the EntityKind every `selected_entities`
# entry must carry on the matching `entity_candidates` row. NONE_OF_THE_
# ABOVE is absent (no entities are selected). SIMILARITY_TO_TITLES is
# the one flow whose name differs from the kind it carries — similarity
# references are titles, so the candidate kind is SPECIFIC_TITLE.
_EXPECTED_KIND_FOR_FLOW: dict[EntityFlow, EntityKind] = {
    EntityFlow.SPECIFIC_TITLE: EntityKind.SPECIFIC_TITLE,
    EntityFlow.SIMILARITY_TO_TITLES: EntityKind.SPECIFIC_TITLE,
    EntityFlow.CHARACTER_FRANCHISE: EntityKind.CHARACTER_FRANCHISE,
    EntityFlow.NON_CHARACTER_FRANCHISE: EntityKind.NON_CHARACTER_FRANCHISE,
    EntityFlow.STUDIO: EntityKind.STUDIO,
    EntityFlow.ACTOR: EntityKind.ACTOR,
}


# Description for the `qualifiers` field on Step0Response. The field
# carries the LLM's extracted list of phrases that change the intended
# meaning of the search relative to a bare-entity lookup. Defined here
# as a module-level string (matching the _MOST_LIKELY_KIND_DESCRIPTION
# pattern below) so the Rich-fields teaching surface stays consolidated
# with the schema rather than living only in the system prompt.
_QUALIFIERS_DESCRIPTION = """\
Phrases extracted from the query that change the intended meaning of \
the search relative to a search for the bare entity name. The \
operating test for every non-entity token is: would the search return \
a meaningfully different result set, ranking, or selection criterion \
if this token were removed? If yes, record the phrase here. If no, \
DROP it — do not record conversational packaging that does not shift \
what the search should return.

Phrases that change meaning, and therefore belong in this list, \
include any language that narrows the result set along a dimension \
the entity alone does not constrain, expresses a preference about \
ranking or quality, attaches a comparison or condition the entity \
flow cannot honor, or otherwise shifts the semantics of what counts \
as a match. The surface form (genre, mood, era, runtime, streaming, \
rating, plot or concept phrase, non-installment cast or director \
reference, etc.) is secondary — the test is whether removing the \
token would change what the search should return.

Phrases that do NOT change meaning, and therefore must be excluded, \
include politeness, speech-act framing, conversational filler and \
discourse markers, hedges that do not actually soften the criteria, \
vacuous quantifiers that do not narrow the catalog, and bare result- \
type words that merely name what kind of result is being requested. \
These tokens are present in the query but do not shift what the \
search should return relative to a bare-entity lookup.

Similarity-frame phrasing is also excluded — it is captured \
structurally by the entity-flow choice. Cast / director / year \
markers that exist specifically to disambiguate a film installment \
are excluded as well — they belong to the corresponding candidate's \
canonical_name or release_year_if_stated.

A non-empty list forces selected_entity_flow to none_of_the_above. \
That invariant is the reason this list must contain only phrases \
that genuinely change the search — fluff recorded here would \
incorrectly demote queries that are effectively pure entity lookups.
"""


# ---------------------------------------------------------------------------
# Per-candidate observation
# ---------------------------------------------------------------------------

# EntityCandidate is the per-span classification. Filling this list is
# the entity-recognition step; selected_entity_flow downstream synthesizes
# across these.
#
# The `most_likely_kind` Field carries the densest teaching for the kind
# taxonomy (Rich-fields variant of ADR-036). Definitions are framed as
# membership criteria + disqualifiers, in principle form. No specific
# franchise or title names appear here so the description stays general.
_MOST_LIKELY_KIND_DESCRIPTION = """\
Pick which kind of entity this span names. The kind drives the global \
routing decision, so commit deliberately.

specific_title — the typed span, after normalizing whitespace, equals \
the canonical title of exactly one movie record. Disqualified when: \
(a) the closest match requires adding or dropping descriptor words \
such as the plural "movies" or "films" (those plural markers signal a \
descriptor query, not a title); (b) multiple distinct films share the \
same canonical title string and resolution requires picking one of \
them; (c) the typed span only matches a film's title under partial-word \
or nickname-style overlap; (d) you would need to consult your own \
knowledge of franchise installments to decide which canonical record \
applies.

character_franchise — the typed span names a fictional protagonist \
whose identity persists across multiple films, where the franchise is \
organized around following that character (often with different actors \
playing the same role across installments). Resolution targets the \
character entity, not any single film.

non_character_franchise — the typed span names a series, property, or \
IP umbrella whose installments share a setting, premise, or branding \
rather than a single protagonist. Multiple disjoint protagonists may \
live under the umbrella. Resolution targets the franchise entity, not \
any single film.

studio — the typed span names a film studio, production company, \
distribution label, or animation house (e.g. a recognized corporate \
producer of films). Resolution targets the studio entity. Disqualified \
when the span is only a parent corporation referenced as a brand \
generality rather than a specific film-producing label.

actor — the typed span names a real human performer credited as an \
actor in films. Resolution targets the person entity. Disqualified \
when the span names a director, writer, producer, or composer rather \
than a performer; those non-actor crew roles do NOT fire the actor \
flow and should fall back to none_of_the_above.
"""


class EntityCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    span_text: constr(strip_whitespace=True, min_length=1)
    most_likely_kind: EntityKind = Field(description=_MOST_LIKELY_KIND_DESCRIPTION)
    canonical_name: constr(strip_whitespace=True, min_length=1)
    release_year_if_stated: int | None = None
    kind_reasoning: constr(strip_whitespace=True, min_length=1)


# ---------------------------------------------------------------------------
# Selected entity payload
# ---------------------------------------------------------------------------

# EntityReference is the entity payload carried into the chosen flow.
# One shape across all flows — the field's *meaning* is dispatched by
# selected_entity_flow:
#   * SPECIFIC_TITLE / SIMILARITY_TO_TITLES → canonical_name is a movie title
#   * CHARACTER_FRANCHISE / NON_CHARACTER_FRANCHISE → canonical_name is a
#     character or franchise umbrella name
#   * STUDIO → canonical_name is a studio / production-company name
#   * ACTOR → canonical_name is an actor's person-entity name
#   * NONE_OF_THE_ABOVE → list is empty
class EntityReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_name: constr(strip_whitespace=True, min_length=1)
    release_year_if_stated: int | None = None


# ---------------------------------------------------------------------------
# Existing executor input types — kept as Step0Response adapter outputs,
# not as LLM-generation targets. run_exact_title_search and
# run_similarity_search still consume these shapes.
# ---------------------------------------------------------------------------

class ExactTitleFlowData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    should_be_searched: bool = Field(...)
    exact_title_to_search: str = Field(...)
    release_year: int | None = Field(default=None)


class SimilarityReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    similar_search_title: constr(strip_whitespace=True, min_length=1) = Field(...)
    release_year: int | None = Field(default=None)


class SimilarityFlowData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    should_be_searched: bool = Field(...)
    references: list[SimilarityReference] = Field(...)


# Franchise flow-data shapes. Both wrap a single canonical_name resolved
# from selected_entities[0].canonical_name. We keep two distinct types
# (rather than a shared FranchiseFlowData) so the two flows can diverge
# without a schema migration once the character-side executor lands.
class CharacterFranchiseFlowData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_name: constr(strip_whitespace=True, min_length=1)


class NonCharacterFranchiseFlowData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_name: constr(strip_whitespace=True, min_length=1)


# Studio and actor flow-data shapes. Both wrap one or more canonical
# names — a query like "tom hanks and woody harrelson" produces two
# entries on the actor side, "warner bros and pixar" produces two on
# the studio side. Distinct types (rather than a shared
# PersonOrCompanyFlowData) so each executor can evolve its own fields
# (e.g. actor-side prominence mode, studio-side label disambiguation)
# without a schema migration.
class StudioReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_name: constr(strip_whitespace=True, min_length=1)


class StudioFlowData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    references: list[StudioReference]


class ActorReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_name: constr(strip_whitespace=True, min_length=1)


class ActorFlowData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    references: list[ActorReference]


# ---------------------------------------------------------------------------
# Top-level Step 0 response
# ---------------------------------------------------------------------------

# Field order is the LLM's generation order — every dependent field
# follows the evidence it depends on. The three zones are demarcated by
# inline comments below.
class Step0Response(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Zone 1 — Exploration (extractive)
    # Every entity-shaped span in the query, classified by kind. Every
    # non-entity phrase that changes the intended meaning of the search
    # relative to a bare-entity lookup (see _QUALIFIERS_DESCRIPTION for
    # the operating test).
    entity_candidates: list[EntityCandidate] = Field(...)
    qualifiers: list[str] = Field(description=_QUALIFIERS_DESCRIPTION)

    # Zone 2 — Entity flow decision (mutually exclusive enum)
    # The reasoning field is placed immediately before the enum it
    # scaffolds, so the LLM commits to why before what.
    selected_entity_flow_reasoning: constr(strip_whitespace=True, min_length=1)
    selected_entity_flow: EntityFlow
    selected_entities: list[EntityReference] = Field(...)

    # Zone 3 — Standard co-fire decision
    # primary_intent_ambiguity_reasoning scaffolds the boolean. The
    # boolean is the ONLY signal the model gives for "fire standard
    # alongside an entity flow." Standard firing in the no-entity case
    # is derived (see fire_standard_flow property below).
    primary_intent_ambiguity_reasoning: constr(strip_whitespace=True, min_length=1)
    also_fire_standard_due_to_ambiguity: bool

    # -----------------------------------------------------------------
    # Validators
    # -----------------------------------------------------------------

    # selected_entities cardinality must match the selected flow.
    @model_validator(mode="after")
    def _entities_match_flow_cardinality(self) -> "Step0Response":
        n = len(self.selected_entities)
        flow = self.selected_entity_flow
        if flow == EntityFlow.NONE_OF_THE_ABOVE and n != 0:
            raise ValueError(
                "selected_entities must be empty when selected_entity_flow is "
                "none_of_the_above."
            )
        if flow in {
            EntityFlow.SPECIFIC_TITLE,
            EntityFlow.CHARACTER_FRANCHISE,
            EntityFlow.NON_CHARACTER_FRANCHISE,
        } and n != 1:
            raise ValueError(
                f"selected_entity_flow={flow.value} requires exactly one "
                f"entry in selected_entities (got {n})."
            )
        # similarity, studio, and actor are the list-style flows — one
        # or more entries. The LLM may surface multiple titles, studios,
        # or actors as the entire query payload (e.g.
        # "tom hanks and woody harrelson" → two ACTOR entries).
        if (
            flow
            in {
                EntityFlow.SIMILARITY_TO_TITLES,
                EntityFlow.STUDIO,
                EntityFlow.ACTOR,
            }
            and n < 1
        ):
            raise ValueError(
                f"selected_entity_flow={flow.value} requires at least "
                "one entry in selected_entities."
            )
        return self

    # The "no qualifiers" rule becomes a schema-level invariant: any
    # qualifier present in the query forces selected_entity_flow to
    # NONE_OF_THE_ABOVE. "Qualifier" here carries the refined meaning
    # from _QUALIFIERS_DESCRIPTION — only phrases that change the
    # intended meaning of the search relative to a bare-entity lookup
    # belong in the list. Conversational packaging that does not shift
    # the search must be dropped at extraction time and does not trip
    # this gate.
    @model_validator(mode="after")
    def _qualifiers_force_no_entity(self) -> "Step0Response":
        if self.qualifiers and self.selected_entity_flow != EntityFlow.NONE_OF_THE_ABOVE:
            raise ValueError(
                "selected_entity_flow must be none_of_the_above when "
                "qualifiers are present — entity-flow routing requires "
                "the entity to be the entire query."
            )
        return self

    # Homogeneity rule: every entity in selected_entities must, when
    # looked up against entity_candidates by canonical_name, carry the
    # EntityKind that matches the selected flow. The prompt teaches
    # the LLM to route mixed-kind queries to NONE_OF_THE_ABOVE; this
    # validator is a hard backstop for the studio/actor list-shaped
    # flows where a slip-up would silently corrupt the executor's
    # downstream resolution.
    @model_validator(mode="after")
    def _selected_entities_match_flow_kind(self) -> "Step0Response":
        expected = _EXPECTED_KIND_FOR_FLOW.get(self.selected_entity_flow)
        if expected is None:
            # NONE_OF_THE_ABOVE — no entities to check.
            return self

        # Build a canonical_name → kind map from candidates. If the LLM
        # emits a selected_entity whose canonical_name has no matching
        # candidate, we skip the check for that entry: the candidate
        # list is the authoritative source of kind, and we don't want
        # to over-constrain when the LLM has legitimately collapsed an
        # alias into a different canonical form. The COVERAGE/RESOLUTION
        # principles in the prompt are the primary enforcement layer;
        # this validator catches the worst-case mismatch.
        kind_by_name = {
            cand.canonical_name: cand.most_likely_kind
            for cand in self.entity_candidates
        }
        for ref in self.selected_entities:
            kind = kind_by_name.get(ref.canonical_name)
            if kind is None:
                continue
            if kind != expected:
                raise ValueError(
                    f"selected_entity_flow={self.selected_entity_flow.value} "
                    f"requires every selected entity to be classified as "
                    f"{expected.value}, but {ref.canonical_name!r} was "
                    f"classified as {kind.value} in entity_candidates."
                )
        return self

    # -----------------------------------------------------------------
    # Derived properties — read by orchestrators / runners
    # -----------------------------------------------------------------

    @property
    def fire_standard_flow(self) -> bool:
        # Standard fires in two cases:
        #   (1) no entity flow was chosen (NONE_OF_THE_ABOVE) — standard
        #       is the fallback so every query routes somewhere;
        #   (2) an entity was chosen but the LLM flagged defensible
        #       ambiguity in primary intent — co-fire standard.
        return (
            self.selected_entity_flow == EntityFlow.NONE_OF_THE_ABOVE
            or self.also_fire_standard_due_to_ambiguity
        )

    @property
    def primary_flow(self) -> SearchFlow:
        # Entity wins primary status when present; STANDARD only when
        # NONE_OF_THE_ABOVE. Mirrors the "more specific reading wins"
        # tiebreaker the previous schema encoded explicitly.
        return _ENTITY_FLOW_TO_SEARCH_FLOW.get(
            self.selected_entity_flow, SearchFlow.STANDARD
        )

    # -----------------------------------------------------------------
    # Adapter methods — construct executor input payloads from the new
    # schema so run_exact_title_search and run_similarity_search keep
    # their existing signatures.
    # -----------------------------------------------------------------

    def to_exact_title_flow_data(self) -> ExactTitleFlowData | None:
        # Only valid when SPECIFIC_TITLE was selected. Cardinality
        # validator guarantees exactly one entry in selected_entities.
        if self.selected_entity_flow != EntityFlow.SPECIFIC_TITLE:
            return None
        ref = self.selected_entities[0]
        return ExactTitleFlowData(
            should_be_searched=True,
            exact_title_to_search=ref.canonical_name,
            release_year=ref.release_year_if_stated,
        )

    def to_similarity_flow_data(self) -> SimilarityFlowData | None:
        # Only valid when SIMILARITY_TO_TITLES was selected. Each entry
        # in selected_entities maps 1:1 to a SimilarityReference.
        if self.selected_entity_flow != EntityFlow.SIMILARITY_TO_TITLES:
            return None
        return SimilarityFlowData(
            should_be_searched=True,
            references=[
                SimilarityReference(
                    similar_search_title=ref.canonical_name,
                    release_year=ref.release_year_if_stated,
                )
                for ref in self.selected_entities
            ],
        )

    def to_character_franchise_flow_data(self) -> CharacterFranchiseFlowData | None:
        # Only valid when CHARACTER_FRANCHISE was selected. Cardinality
        # validator guarantees exactly one entry in selected_entities.
        if self.selected_entity_flow != EntityFlow.CHARACTER_FRANCHISE:
            return None
        return CharacterFranchiseFlowData(
            canonical_name=self.selected_entities[0].canonical_name,
        )

    def to_non_character_franchise_flow_data(
        self,
    ) -> NonCharacterFranchiseFlowData | None:
        # Only valid when NON_CHARACTER_FRANCHISE was selected.
        if self.selected_entity_flow != EntityFlow.NON_CHARACTER_FRANCHISE:
            return None
        return NonCharacterFranchiseFlowData(
            canonical_name=self.selected_entities[0].canonical_name,
        )

    def to_studio_flow_data(self) -> StudioFlowData | None:
        # Only valid when STUDIO was selected. Cardinality validator
        # guarantees at least one entry in selected_entities. Each
        # entry maps 1:1 to a StudioReference. The studio executor is
        # not yet wired into the orchestrator — this adapter exists so
        # downstream callers can plumb the payload through once it is.
        if self.selected_entity_flow != EntityFlow.STUDIO:
            return None
        return StudioFlowData(
            references=[
                StudioReference(canonical_name=ref.canonical_name)
                for ref in self.selected_entities
            ],
        )

    def to_actor_flow_data(self) -> ActorFlowData | None:
        # Only valid when ACTOR was selected. Cardinality validator
        # guarantees at least one entry in selected_entities. Each
        # entry maps 1:1 to an ActorReference. The actor executor is
        # not yet wired into the orchestrator — this adapter exists so
        # downstream callers can plumb the payload through once it is.
        if self.selected_entity_flow != EntityFlow.ACTOR:
            return None
        return ActorFlowData(
            references=[
                ActorReference(canonical_name=ref.canonical_name)
                for ref in self.selected_entities
            ],
        )
