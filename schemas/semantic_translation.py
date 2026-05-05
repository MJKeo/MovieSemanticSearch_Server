# Step 3 semantic endpoint structured output models.
#
# Translates one category-handler call (a single trait's
# retrieval_intent + expressions, routed to the semantic endpoint)
# into a multi-vector search plan over the 7 non-anchor Qdrant spaces.
#
# Two shapes coexist in this file:
#
# 1. SINGLE-ENDPOINT shape (SemanticParameters / SemanticEndpointParameters)
#    — used by buckets 3/4 when semantic owns the entire call. Internal
#    structure: role_exploration → role → aspects → space_candidates →
#    space_queries. Untouched by the walk-then-commit refactor.
#
# 2. MULTI-ENDPOINT shape (SemanticWalk + SemanticParametersSubintent /
#    SemanticEndpointSubintentParameters) — used by buckets 5/6/8.
#    Split across two classes that live in different positions of the
#    bucket schema:
#       a. SemanticWalk — the space-grounded analysis layer
#          (`aspects` / `space_candidates`). Lives at the bucket level,
#          emitted BEFORE the coverage_assignments commitment so the
#          LLM walks the spaces concretely before committing.
#       b. SemanticParametersSubintent — thin commitment-only spec
#          (`role_exploration` + `role` + `space_queries`). Lives inside
#          the per-endpoint semantic_parameters slot, populated only
#          when coverage_assignments delegates a slice to this endpoint.
#    `role_exploration` + `role` stay paired in the thin spec because
#    they are semantic-internal commitment, not space-grounded analysis.
#
# Single unified shape — SemanticParameters — is emitted regardless of
# whether the trait wants carver-style (strict, equal-vote) or
# qualifier-style (looser, weighted-sum) retrieval. The LLM commits the
# decision in the `role` field at the top of the schema and every
# downstream field reads from that commitment:
#
#   role == CARVER
#       Strict bar. Executor sums elbow-decayed scores evenly across
#       active spaces, divides by count, compresses [0,1]→[0.5,1] for
#       survivors. Per-entry weights are populated for honesty but
#       ignored at execution; every active space gets equal vote, so
#       marginal spaces dilute directly.
#
#   role == QUALIFIER
#       Looser bar. Executor takes Σ(w·cos)/Σw across entries with
#       CENTRAL=2.0 / SUPPORTING=1.0. SUPPORTING weight exists for
#       spaces that round out the match without being load-bearing.
#
# The schema enforces neither selectivity nor weight-vs-role
# consistency — the runtime takes what the LLM emits and dispatches on
# `role`. Inconsistent commits (e.g. CARVER + 5 spaces) are an LLM
# behavior issue, not a schema-level guard.
#
# Single-trait calls only. Bundling of multiple semantic-routed
# traits happens at the orchestrator layer, not here.
#
# Body authoring uses the existing 7 SpaceBody types in each space's
# native ingest-side vocabulary (term lists, prose, terms-with-
# negations). Space-entry wrappers are intentionally thin — space +
# content. Per-space reasoning lives on SpaceCandidate.
#
# Server-side recovery: duplicate-space entries collapse via
# _merge_bodies (list concat-dedupe, prose ". " join, nested
# BaseModel recurse). Weight collisions resolve to CENTRAL.

from __future__ import annotations

from enum import Enum
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, conlist, constr, model_validator

from schemas.endpoint_parameters import EndpointParameters
from schemas.semantic_bodies import (
    NarrativeTechniquesBody,
    PlotAnalysisBody,
    PlotEventsBody,
    ProductionBody,
    ReceptionBody,
    ViewerExperienceBody,
    WatchContextBody,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SemanticSpace(str, Enum):
    PLOT_EVENTS = "plot_events"
    PLOT_ANALYSIS = "plot_analysis"
    VIEWER_EXPERIENCE = "viewer_experience"
    WATCH_CONTEXT = "watch_context"
    NARRATIVE_TECHNIQUES = "narrative_techniques"
    PRODUCTION = "production"
    RECEPTION = "reception"


class SpaceWeight(str, Enum):
    CENTRAL = "central"
    SUPPORTING = "supporting"


# Semantic-side retrieval-shape enum. Distinct from the (now-removed)
# Step-2 Role enum: this one lives on the semantic schema and is
# emitted by the handler LLM per call rather than being inherited from
# the parent trait. The literal values (carver / qualifier) match the
# old Role enum's values intentionally so the executor's score-shape
# vocabulary stays consistent — the difference is *who decides*.
class SemanticRetrievalShape(str, Enum):
    CARVER = "carver"
    QUALIFIER = "qualifier"


# ---------------------------------------------------------------------------
# SpaceCandidate — shared exploration entry. The space_candidates
# list IS the permitted commit set; space_queries entries that name
# an unexplored space are a sign the analysis layer was skipped.
# ---------------------------------------------------------------------------


class SpaceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    space: SemanticSpace = Field(
        ...,
        description=(
            "Vector space whose ingest-side vocabulary plausibly "
            "carries >=1 entry from `aspects`. List only spaces with "
            "substantive strengths; do not enumerate all 7 for "
            "thoroughness."
        ),
    )
    strengths: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Which entries from `aspects` THIS space genuinely OWNS at "
            "retrieval time, written holistically. Compose interlocking "
            "aspects into one phrase — 'darkly funny' is one signal "
            "this space carries, not two; the embedding lands more "
            "accurately when composed signals stay composed than when "
            "split and re-intersected. Read `aspects` (primary: what "
            "needs covering) and `retrieval_intent` (discriminative "
            "anchor when an aspect could surface-match here but the "
            "intent narrows it elsewhere).\n"
            "\n"
            "TEST: if I removed this space from candidates, which "
            "aspects orphan? Name them.\n"
            "\n"
            "NEVER:\n"
            "- BACK-RATIONALIZE. 'Plausible' isn't a strength. Drop "
            "the candidate or commit what's substantively owned.\n"
            "- ENUMERATE 1:1 WITH ASPECTS. Composed aspects fold; "
            "splitting them defeats the embedding."
        ),
    )
    weaknesses: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "What this space MISSES or OVER-PULLS relative to the "
            "aspects this candidate is being considered for. Two "
            "failure modes — both belong here:\n"
            "- under-coverage: aspects this space does NOT pick up, "
            "with the destination space that catches them (occasion → "
            "watch_context; craft → narrative_techniques; scalar "
            "reception → metadata; etc.).\n"
            "- over-coverage: content this space's embedding would "
            "ALSO match beyond the slice (e.g. plot_analysis on a "
            "broad genre query pulls every adjacent thematic shape).\n"
            "\n"
            "Suggested vocabulary (not enforced): prefix lines with "
            "'under-coverage:' and 'over-coverage:'. Use 'none' only "
            "when `strengths` is a clean fit on both axes.\n"
            "\n"
            "NEVER:\n"
            "- HEDGE WITHOUT NAMING. Either 'none' or specific "
            "gap/over-pull with concrete content.\n"
            "- INVENT WEAKNESSES to look thorough.\n"
            "- RESTATE `strengths` IN NEGATIVE FORM. Weaknesses name "
            "what's NOT here or what's excessively here, not a "
            "re-listing of what's owned."
        ),
    )


# ---------------------------------------------------------------------------
# Space-entry wrappers. Thin: space + content only. Literal[…] tag +
# extra="forbid" force each body to match its declared space.
#
# Plain Union (no Field(discriminator=...)) so Pydantic emits anyOf
# rather than oneOf — OpenAI structured outputs rejects oneOf;
# Literal+forbid still guarantees one branch.
# ---------------------------------------------------------------------------


_SPACE_DESC = (
    "Vector space this entry targets. Must match the Body type on "
    "`content` AND appear in `space_candidates` — committing to a "
    "space that wasn't analyzed is a signal the exploration step "
    "was skipped."
)
_CONTENT_DESC = (
    "Structured payload in this space's native ingest-side "
    "vocabulary. Author the body to read like ingest-side text for "
    "a matching movie — match the verbosity and register each sub-"
    "field uses on the ingest side. Fold every aspect this space "
    "owns (per its SpaceCandidate.strengths) into ONE body; "
    "do not split one space's coverage across multiple entries.\n"
    "\n"
    "NEVER:\n"
    "- POPULATE A SUB-FIELD WITHOUT REAL SIGNAL. Empty sub-fields "
    "are valid and expected; padding dilutes the query vector.\n"
    "- COPY user-side phrasing verbatim if the space's ingest-side "
    "register differs. Translate.\n"
    "- INCLUDE NUMERICS. Years, runtimes, ratings route to "
    "metadata, not here."
)


class PlotEventsEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    space: Literal[SemanticSpace.PLOT_EVENTS] = Field(..., description=_SPACE_DESC)
    content: PlotEventsBody = Field(..., description=_CONTENT_DESC)


class PlotAnalysisEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    space: Literal[SemanticSpace.PLOT_ANALYSIS] = Field(..., description=_SPACE_DESC)
    content: PlotAnalysisBody = Field(..., description=_CONTENT_DESC)


class ViewerExperienceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    space: Literal[SemanticSpace.VIEWER_EXPERIENCE] = Field(..., description=_SPACE_DESC)
    content: ViewerExperienceBody = Field(..., description=_CONTENT_DESC)


class WatchContextEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    space: Literal[SemanticSpace.WATCH_CONTEXT] = Field(..., description=_SPACE_DESC)
    content: WatchContextBody = Field(..., description=_CONTENT_DESC)


class NarrativeTechniquesEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    space: Literal[SemanticSpace.NARRATIVE_TECHNIQUES] = Field(..., description=_SPACE_DESC)
    content: NarrativeTechniquesBody = Field(..., description=_CONTENT_DESC)


class ProductionEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    space: Literal[SemanticSpace.PRODUCTION] = Field(..., description=_SPACE_DESC)
    content: ProductionBody = Field(..., description=_CONTENT_DESC)


class ReceptionEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    space: Literal[SemanticSpace.RECEPTION] = Field(..., description=_SPACE_DESC)
    content: ReceptionBody = Field(..., description=_CONTENT_DESC)


SemanticSpaceEntry = Union[
    PlotEventsEntry,
    PlotAnalysisEntry,
    ViewerExperienceEntry,
    WatchContextEntry,
    NarrativeTechniquesEntry,
    ProductionEntry,
    ReceptionEntry,
]


# ---------------------------------------------------------------------------
# Body merge helper. Generic walk, list concat-dedupe, prose ". " join,
# nested BaseModel recurse. Same-space duplicate recovery uses this.
# ---------------------------------------------------------------------------


def _merge_bodies(bodies: list[BaseModel]) -> BaseModel:
    merged = bodies[0].model_copy()
    for field_name in type(merged).model_fields:
        values = [getattr(b, field_name) for b in bodies]
        sample = next((v for v in values if v is not None), None)

        if isinstance(sample, list):
            combined: list = []
            seen: set = set()
            for v in values:
                for item in v:
                    if item not in seen:
                        seen.add(item)
                        combined.append(item)
            setattr(merged, field_name, combined)
        elif isinstance(sample, BaseModel):
            non_none = [v for v in values if v is not None]
            setattr(merged, field_name, _merge_bodies(non_none))
        else:
            non_empty = [v for v in values if v]
            setattr(
                merged,
                field_name,
                ". ".join(non_empty) if non_empty else None,
            )
    return merged


# ---------------------------------------------------------------------------
# WeightedSpaceQuery — every space entry carries a weight on the
# unified schema. Executor uses weights on the qualifier path and
# ignores them on the carver path; the weight is populated honestly
# regardless of which path the LLM committed to.
# ---------------------------------------------------------------------------


_WEIGHT_DESC = (
    "How load-bearing this space's signal is for the trait. Read "
    "`retrieval_intent` (primary source: framing of what the search "
    "positions against) IN LIGHT OF the matching "
    "`SpaceCandidate.strengths` for `query.space` above (i.e., "
    "what this specific space actually carries) — mechanical mapping; "
    "do not re-derive from `aspects` wholesale.\n"
    "- CENTRAL: retrieval_intent treats the aspects this space carries "
    "(per its SpaceCandidate.strengths) as defining whether the "
    "match holds. Missing this signal = trait broken. Multiple central "
    "spaces are fine when several axes are equally load-bearing.\n"
    "- SUPPORTING: the aspects this space carries round out the "
    "experience but aren't load-bearing on their own. Trait stays "
    "recognizable without it.\n"
    "\n"
    "ALWAYS POPULATED. Weights are read by the executor on the "
    "qualifier path (Σ(w·cos)/Σw) and IGNORED on the carver path "
    "(equal-vote scoring). Populate honestly regardless of `role` so "
    "the data stays interpretable for evaluation and any future shape "
    "change.\n"
    "\n"
    "All-supporting is acceptable for broad-and-balanced traits — a "
    "truthful signal, not a cop-out. If a space would be below "
    "SUPPORTING (barely-there), drop the entry instead of including "
    "it.\n"
    "\n"
    "TEST: look up the `strengths` on the SpaceCandidate "
    "matching `query.space`. Does `retrieval_intent` name those "
    "aspects as defining the match, or as one contributor among "
    "several? Defining → CENTRAL. Contributor → SUPPORTING."
)


class WeightedSpaceQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weight: SpaceWeight = Field(..., description=_WEIGHT_DESC)
    query: SemanticSpaceEntry = Field(
        ...,
        description=(
            "Space-targeted body — {space + content}. `space` MUST "
            "appear in `space_candidates` (any commit to a space "
            "absent from candidates means exploration was skipped). "
            "`content` is authored against the rules on the matching "
            "entry-wrapper class (PlotEventsEntry / "
            "ViewerExperienceEntry / etc.) — fold every aspect this "
            "space owns per its SpaceCandidate.strengths into "
            "ONE body in the space's ingest-side vocabulary; do not "
            "split coverage across multiple WeightedSpaceQuery "
            "entries with the same `query.space`."
        ),
    )


# ---------------------------------------------------------------------------
# Shared field descriptions for the unified parameters class.
# ---------------------------------------------------------------------------


_ROLE_EXPLORATION_DESC = (
    "Decide whether this call should retrieve carver-style (strict, "
    "equal-vote across spaces) or qualifier-style (looser, weighted "
    "sum). Read `retrieval_intent` (primary) and `expressions`.\n"
    "\n"
    "CARVER applies when the trait NAMES THE POPULATION whose "
    "semantic profile must match — the call describes what to find "
    "directly ('movies about war', 'feel-good Christmas movies', "
    "'filmed in Iceland'). Adding a marginal space would let "
    "clearly-wrong movies past the gate, so the bar is strict and "
    "every active space gets equal vote at execution.\n"
    "\n"
    "QUALIFIER applies when the trait POSITIONS A POPULATION AGAINST "
    "A REFERENCE / threshold / archetype — the call describes how a "
    "found movie should compare ('more atmospheric than typical "
    "period pieces', 'darker than usual rom-com', 'leans more on "
    "visual symbolism than dialogue'). Adding a space rounds out the "
    "match instead of diluting it; SUPPORTING-weight signals are "
    "permitted at execution.\n"
    "\n"
    "TEST: 'if I dropped a marginal space from the commit, would "
    "that leak wrong movies past the gate (carver) or just thin the "
    "match a little (qualifier)?'\n"
    "\n"
    "NEVER:\n"
    "- HEDGE. Pick one; the choice steers selectivity and weighting "
    "below.\n"
    "- RE-DERIVE FROM `aspects`. Aspects are downstream of the same "
    "intent — they don't add signal for this decision."
)


_ROLE_DESC = (
    "Mechanical commit of the choice argued in `role_exploration`. "
    "CARVER if the exploration concluded population-naming; "
    "QUALIFIER if positioning-against-a-reference. No new reasoning "
    "at this step — the commit must match the conclusion above."
)


_ASPECTS_DESC = (
    "Holistic decomposition of the trait into atoms — distinct axes "
    "the trait calls for. Read `expressions` (primary: the dimensions "
    "Step 3 already owned for this trait — atoms are derived from "
    "these, not re-derived from scratch) plus `retrieval_intent` "
    "(framing: what the search positions against). NOT 1:1 with "
    "expressions: one expression may carry several aspects, several "
    "expressions may collapse into one. Compose interlocking signals "
    "into one aspect when they perform better embedded together — "
    "'darkly funny' is one aspect, not two; the dark and the funny "
    "are inseparable in the felt experience and a combined cosine "
    "outperforms two intersected ones.\n"
    "\n"
    "TEST per aspect: would removing it change which space ends up "
    "carrying the trait? Yes → distinct aspect. No → fold.\n"
    "\n"
    "NEVER:\n"
    "- COPY EXPRESSIONS VERBATIM. Decompose first.\n"
    "- SPLIT A COMPOSED SIGNAL that performs better embedded "
    "together.\n"
    "- INVENT axes the inputs don't signal."
)

_SPACE_CANDIDATES_DESC = (
    "Per-space exploration. One entry per space whose ingest-side "
    "vocabulary plausibly carries >=1 entry from `aspects`. Read "
    "`aspects` (primary: what needs covering) and `retrieval_intent` "
    "(tiebreaker on near-adjacencies). Skip spaces with no real "
    "coverage — do not enumerate all 7.\n"
    "\n"
    "This list IS the permitted commit set: `space_queries` below "
    "pulls exclusively from spaces that appear here.\n"
    "\n"
    "TEST per candidate: 'if I dropped this candidate, would the "
    "commit step lose a real routing option?' Yes → keep. No → drop.\n"
    "\n"
    "NEVER:\n"
    "- DUPLICATE A SPACE across candidates.\n"
    "- LIST A SPACE whose strengths would be hand-waving "
    "rather than substantively named."
)

_SPACE_QUERIES_DESC = (
    "Committed set of spaces to search, each carrying a weight. Pull "
    "exclusively from `space_candidates` above. Selectivity bar "
    "depends on `role`:\n"
    "\n"
    "If role == CARVER: STRICT BAR. Carver execution sums elbow-"
    "decayed scores evenly across active spaces and divides by the "
    "count — every marginal space directly dilutes the gate. Typical "
    "1–2 spaces, occasionally 3. Commit only to spaces whose signal "
    "is genuinely load-bearing for the trait. A third space is "
    "justified only when each of the three would clearly let a wrong "
    "movie pass if dropped. Prefer one well-targeted space over two "
    "thin ones. Per-entry `weight` is populated for honesty but "
    "ignored at execution.\n"
    "\n"
    "If role == QUALIFIER: LOOSER BAR. Qualifier execution takes a "
    "normalized weighted sum — Σ(w·cos)/Σw across entries. Typical "
    "2–4 spaces, sometimes more. Additional spaces round out the "
    "match without diluting load-bearing ones. Drop entries whose "
    "signal is below SUPPORTING (barely-there) — those just add "
    "noise.\n"
    "\n"
    "Fold every aspect a space owns (per its "
    "SpaceCandidate.strengths) into ONE entry's `query.content`; "
    "do not split coverage across multiple entries with the same "
    "`query.space`. Same-space duplicates merge server-side rather "
    "than fail; emit cleanly when you can.\n"
    "\n"
    "NEVER:\n"
    "- COMMIT to a space absent from `space_candidates`.\n"
    "- SPLIT one space's coverage across multiple entries.\n"
    "- IGNORE the role-keyed selectivity bar — `role` above dictates "
    "how strict to be."
)


# ---------------------------------------------------------------------------
# SemanticParameters — unified shape; LLM commits role at the top.
# ---------------------------------------------------------------------------


class SemanticParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role_exploration: constr(strip_whitespace=True, min_length=1) = Field(
        ..., description=_ROLE_EXPLORATION_DESC
    )
    role: SemanticRetrievalShape = Field(..., description=_ROLE_DESC)
    aspects: conlist(
        constr(strip_whitespace=True, min_length=1), min_length=1
    ) = Field(..., description=_ASPECTS_DESC)
    space_candidates: conlist(SpaceCandidate, min_length=1) = Field(
        ..., description=_SPACE_CANDIDATES_DESC
    )
    space_queries: conlist(WeightedSpaceQuery, min_length=1) = Field(
        ..., description=_SPACE_QUERIES_DESC
    )

    @model_validator(mode="after")
    def _drop_empty_and_merge_duplicates(self) -> "SemanticParameters":
        # Drop entries whose body produces no embedding text — a space
        # with all-empty sub-fields contributes nothing but noise to
        # both the carver elbow-normalize step and the qualifier
        # weighted sum.
        kept = [
            wq for wq in self.space_queries
            if wq.query.content.embedding_text().strip()
        ]
        if not kept:
            raise ValueError(
                "space_queries empty after dropping entries with no embedding text"
            )

        grouped: dict = {}
        order: list = []
        for wq in kept:
            space = wq.query.space
            if space not in grouped:
                grouped[space] = [wq]
                order.append(space)
            else:
                grouped[space].append(wq)

        if len(order) == len(kept):
            self.space_queries = kept
            return self

        # Stronger weight wins on collision; bodies merge. (Even on the
        # carver path the weight is preserved for inspection; execution
        # ignores it.)
        merged: list = []
        for space in order:
            wqs = grouped[space]
            if len(wqs) == 1:
                merged.append(wqs[0])
                continue
            stronger = (
                SpaceWeight.CENTRAL
                if any(wq.weight == SpaceWeight.CENTRAL for wq in wqs)
                else SpaceWeight.SUPPORTING
            )
            base_query = wqs[0].query.model_copy()
            base_query.content = _merge_bodies([wq.query.content for wq in wqs])
            merged.append(WeightedSpaceQuery(weight=stronger, query=base_query))
        self.space_queries = merged
        return self


class SemanticEndpointParameters(EndpointParameters):
    parameters: SemanticParameters = Field(
        ...,
        description=(
            "Semantic payload. Decide retrieval shape (carver vs "
            "qualifier) in `role_exploration` and commit it in `role`, "
            "then decompose the trait into aspects, explore which "
            "spaces plausibly cover them, and commit the weighted "
            "space set per the role-keyed selectivity bar. Describe "
            "the target concept directly regardless of polarity; "
            "polarity is applied downstream by the handler when "
            "bucketing the finding."
        ),
    )


# ---------------------------------------------------------------------------
# Multi-endpoint walk + thin commitment
#
# Used when the semantic endpoint is one of several contending for the
# call's intent (buckets 5/6/8). Two classes that live in DIFFERENT
# positions of the bucket schema:
#
#   - SemanticWalk lives at the bucket level, emitted BEFORE the
#     coverage_assignments commitment phase. It carries the
#     space-grounded analysis of how the semantic endpoint could
#     cover the call's retrieval_intent. Forces concrete vector-space
#     awareness before any commitment.
#   - SemanticParametersSubintent lives inside the per-endpoint
#     semantic_parameters slot, populated only when coverage_assignments
#     delegates a slice to this endpoint. Carries just the commitment
#     layer — role_exploration + role + space_queries — plus the
#     same-space-merge validator.
#
# `role_exploration` and `role` stay paired in the thin spec because
# they are semantic-internal commitment (carver vs qualifier
# selectivity, not space-grounded analysis).
#
# Sub-types whose descriptors don't reference the raw inputs
# (SemanticSpaceEntry and its body types) are reused as-is.
# ---------------------------------------------------------------------------


class SemanticWalk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Reuses the shared SpaceCandidate class: its descriptor reads
    # `aspects` (sibling field on this walk) and `retrieval_intent`
    # (in the user message). Both inputs are exactly what the walk
    # needs to ground itself.
    aspects: conlist(
        constr(strip_whitespace=True, min_length=1), min_length=1
    ) = Field(
        ...,
        description=(
            "Holistic decomposition of the call's intent into atoms — "
            "distinct axes the trait calls for that the semantic "
            "endpoint could potentially carry. Read the call's "
            "`retrieval_intent` + `expressions` (in the user message). "
            "NOT 1:1 with phrases in the intent: one phrase may carry "
            "several aspects, several phrases may collapse into one. "
            "Compose interlocking signals into one aspect when they "
            "perform better embedded together — 'darkly funny' is one "
            "aspect, not two; the dark and the funny are inseparable "
            "in the felt experience and a combined cosine outperforms "
            "two intersected ones.\n"
            "\n"
            "TEST per aspect: would removing it change which space "
            "ends up carrying the trait? Yes → distinct aspect. "
            "No → fold.\n"
            "\n"
            "NEVER:\n"
            "- COPY `retrieval_intent` PHRASING VERBATIM. Decompose "
            "first.\n"
            "- SPLIT A COMPOSED SIGNAL that performs better embedded "
            "together.\n"
            "- INVENT axes the call doesn't signal."
        ),
    )
    space_candidates: conlist(SpaceCandidate, min_length=1) = Field(
        ...,
        description=(
            "Per-space exploration of how the semantic endpoint could "
            "cover the aspects above. One entry per space whose "
            "ingest-side vocabulary plausibly carries >=1 entry from "
            "`aspects`. Read `aspects` (primary: what needs covering) "
            "and the call's `retrieval_intent` (tiebreaker on "
            "near-adjacencies). Skip spaces with no real coverage — do "
            "not enumerate all 7.\n"
            "\n"
            "This is the GROUNDED walk that precedes the bucket-level "
            "coverage_assignments commitment. Surface every plausibly "
            "useful space with concrete coverage prose so the "
            "commitment phase can read off real candidates rather than "
            "abstract optimism. An empty space_candidates with all "
            "aspects orphaned is a valid signal that the semantic "
            "endpoint has nothing useful — the commitment phase is "
            "allowed to leave the call unowned by semantic.\n"
            "\n"
            "TEST per candidate: 'if I dropped this candidate, would "
            "the commit step lose a real routing option?' Yes → keep. "
            "No → drop.\n"
            "\n"
            "NEVER:\n"
            "- DUPLICATE A SPACE across candidates.\n"
            "- LIST A SPACE whose strengths would be hand-waving "
            "rather than substantively named."
        ),
    )


_WEIGHT_DESC_SUBINTENT = (
    "How load-bearing this space's signal is for the slice. Read "
    "`semantic_retrieval_intent` (primary source: framing of what "
    "the search positions against) IN LIGHT OF the matching "
    "`SpaceCandidate.strengths` from the bucket-level "
    "`semantic_walk` above (i.e., what this specific space actually "
    "carries) — mechanical mapping; do not re-derive from "
    "`semantic_walk.aspects` wholesale.\n"
    "- CENTRAL: semantic_retrieval_intent treats the aspects this "
    "space carries (per its SpaceCandidate.strengths) as "
    "defining whether the match holds. Missing this signal = slice "
    "broken. Multiple central spaces are fine when several axes are "
    "equally load-bearing.\n"
    "- SUPPORTING: the aspects this space carries round out the "
    "experience but aren't load-bearing on their own. Slice stays "
    "recognizable without it.\n"
    "\n"
    "ALWAYS POPULATED. Weights are read by the executor on the "
    "qualifier path (Σ(w·cos)/Σw) and IGNORED on the carver path "
    "(equal-vote scoring). Populate honestly regardless of `role` so "
    "the data stays interpretable for evaluation and any future "
    "shape change.\n"
    "\n"
    "All-supporting is acceptable for broad-and-balanced slices — a "
    "truthful signal, not a cop-out. If a space would be below "
    "SUPPORTING (barely-there), drop the entry instead of including "
    "it."
)


class WeightedSpaceQuerySubintent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weight: SpaceWeight = Field(..., description=_WEIGHT_DESC_SUBINTENT)
    query: SemanticSpaceEntry = Field(
        ...,
        description=(
            "Space-targeted body — {space + content}. `space` MUST "
            "appear in the bucket-level `semantic_walk.space_candidates` "
            "above (any commit to a space absent from candidates means "
            "the walk was skipped). `content` is authored against the "
            "rules on the matching entry-wrapper class (PlotEventsEntry "
            "/ ViewerExperienceEntry / etc.) — fold every aspect this "
            "space owns per the matching SpaceCandidate.strengths "
            "into ONE body in the space's ingest-side vocabulary; do "
            "not split coverage across multiple "
            "WeightedSpaceQuerySubintent entries with the same "
            "`query.space`."
        ),
    )


_ROLE_EXPLORATION_DESC_SUBINTENT = (
    "Decide whether this call should retrieve carver-style (strict, "
    "equal-vote across spaces) or qualifier-style (looser, weighted "
    "sum). Read `semantic_retrieval_intent` — the slice of intent "
    "this endpoint was assigned by the bucket-level "
    "coverage_assignments above.\n"
    "\n"
    "CARVER applies when the slice NAMES THE POPULATION whose "
    "semantic profile must match. Adding a marginal space would let "
    "clearly-wrong movies past the gate, so the bar is strict and "
    "every active space gets equal vote at execution.\n"
    "\n"
    "QUALIFIER applies when the slice POSITIONS A POPULATION "
    "AGAINST A REFERENCE / threshold / archetype. Adding a space "
    "rounds out the match instead of diluting it; SUPPORTING-weight "
    "signals are permitted at execution.\n"
    "\n"
    "TEST: 'if I dropped a marginal space from the commit, would "
    "that leak wrong movies past the gate (carver) or just thin the "
    "match a little (qualifier)?'\n"
    "\n"
    "NEVER:\n"
    "- HEDGE. Pick one; the choice steers selectivity and weighting "
    "below.\n"
    "- RE-DERIVE FROM `semantic_walk.aspects`. Aspects are downstream "
    "of the same intent — they don't add signal for this decision."
)


_SPACE_QUERIES_DESC_SUBINTENT = (
    "Committed set of spaces to search, each carrying a weight. Pull "
    "exclusively from the bucket-level `semantic_walk.space_candidates` "
    "above. Selectivity bar depends on `role`:\n"
    "\n"
    "If role == CARVER: STRICT BAR. Carver execution sums elbow-"
    "decayed scores evenly across active spaces and divides by the "
    "count — every marginal space directly dilutes the gate. Typical "
    "1–2 spaces, occasionally 3. Commit only to spaces whose signal "
    "is genuinely load-bearing for the slice. Per-entry `weight` is "
    "populated for honesty but ignored at execution.\n"
    "\n"
    "If role == QUALIFIER: LOOSER BAR. Qualifier execution takes a "
    "normalized weighted sum — Σ(w·cos)/Σw across entries. Typical "
    "2–4 spaces, sometimes more. Drop entries whose signal is below "
    "SUPPORTING.\n"
    "\n"
    "Fold every aspect a space owns (per its "
    "SpaceCandidate.strengths for that space, in the walk above) "
    "into ONE entry's `query.content`; do not split coverage across "
    "multiple entries with the same `query.space`. Same-space "
    "duplicates merge server-side rather than fail; emit cleanly "
    "when you can.\n"
    "\n"
    "NEVER:\n"
    "- COMMIT to a space absent from `semantic_walk.space_candidates`.\n"
    "- SPLIT one space's coverage across multiple entries.\n"
    "- IGNORE the role-keyed selectivity bar — `role` above dictates "
    "how strict to be."
)


_SEMANTIC_RETRIEVAL_INTENT_DESC = (
    "The slice of the call's intent this endpoint owns, committed by "
    "the bucket-level coverage_assignments above (the entry whose "
    "endpoint_kind matches semantic). Restate the assigned "
    "slice_description here so the commitment below has a single, "
    "self-contained pointer.\n"
    "\n"
    "Scope: free-form thematic, tonal, experiential, narrative-craft, "
    "production, or specific-aspect reception qualifiers — anything "
    "that lands in the 7 vector spaces rather than a structured "
    "column or a registry member. Leave categorical classification, "
    "structured attributes, named entities, and awards to their "
    "respective endpoints. Every field on this endpoint's "
    "`parameters` reads from this intent (and from the upstream "
    "semantic_walk) rather than from any other input."
)


class SemanticParametersSubintent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Thin commitment-only spec for multi-endpoint contexts. The
    # analysis layer that previously lived here (`aspects` and
    # `space_candidates`) has been lifted up to SemanticWalk at the
    # bucket level — populated BEFORE coverage_assignments so the
    # commitment phase is grounded in concrete space candidates.
    # `role_exploration` + `role` stay paired here because they are
    # semantic-internal commitment, not space-grounded analysis.

    role_exploration: constr(strip_whitespace=True, min_length=1) = Field(
        ..., description=_ROLE_EXPLORATION_DESC_SUBINTENT
    )
    role: SemanticRetrievalShape = Field(..., description=_ROLE_DESC)
    space_queries: conlist(WeightedSpaceQuerySubintent, min_length=1) = Field(
        ..., description=_SPACE_QUERIES_DESC_SUBINTENT
    )

    @model_validator(mode="after")
    def _drop_empty_and_merge_duplicates(self) -> "SemanticParametersSubintent":
        kept = [
            wq for wq in self.space_queries
            if wq.query.content.embedding_text().strip()
        ]
        if not kept:
            raise ValueError(
                "space_queries empty after dropping entries with no embedding text"
            )

        grouped: dict = {}
        order: list = []
        for wq in kept:
            space = wq.query.space
            if space not in grouped:
                grouped[space] = [wq]
                order.append(space)
            else:
                grouped[space].append(wq)

        if len(order) == len(kept):
            self.space_queries = kept
            return self

        merged: list = []
        for space in order:
            wqs = grouped[space]
            if len(wqs) == 1:
                merged.append(wqs[0])
                continue
            stronger = (
                SpaceWeight.CENTRAL
                if any(wq.weight == SpaceWeight.CENTRAL for wq in wqs)
                else SpaceWeight.SUPPORTING
            )
            base_query = wqs[0].query.model_copy()
            base_query.content = _merge_bodies([wq.query.content for wq in wqs])
            merged.append(WeightedSpaceQuerySubintent(weight=stronger, query=base_query))
        self.space_queries = merged
        return self


class SemanticEndpointSubintentParameters(EndpointParameters):
    semantic_retrieval_intent: constr(
        strip_whitespace=True, min_length=1
    ) = Field(..., description=_SEMANTIC_RETRIEVAL_INTENT_DESC)
    parameters: SemanticParametersSubintent = Field(
        ...,
        description=(
            "Semantic endpoint thin commitment payload. Reads off "
            "`semantic_retrieval_intent` (the assigned slice) and "
            "the bucket-level `semantic_walk.space_candidates` (the "
            "grounded analysis above) to commit role + weighted "
            "space set per the role-keyed selectivity bar."
        ),
    )
