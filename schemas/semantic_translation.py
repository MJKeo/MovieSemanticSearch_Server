# Step 3 semantic endpoint structured output models.
#
# Translates one category-handler call (a single trait's
# retrieval_intent + expressions, routed to the semantic endpoint)
# into a multi-vector search plan over the 7 non-anchor Qdrant spaces.
#
# Two top-level shapes — CarverSemanticParameters and
# QualifierSemanticParameters — share the exploration layer
# (aspects + space_candidates) and diverge only at commitment.
#
#   role == CARVER
#       space_queries: list[SemanticSpaceEntry]; no weights. Executor
#       elbow-normalizes each space, applies linear decay over the
#       10% window below the elbow, sums across active spaces /
#       count, then compresses [0,1]→[0.5,1] for survivors. Strict
#       bar: equal-vote scoring means marginal spaces dilute directly.
#
#   role == QUALIFIER
#       space_queries: list[WeightedSpaceQuery]; per-space weight.
#       Executor takes Σ(w·cos)/Σw across entries. Looser bar:
#       SUPPORTING weight exists for spaces that round out the match.
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
# BaseModel recurse). Weight collisions on the qualifier path
# resolve to CENTRAL.

from __future__ import annotations

from enum import Enum
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, conlist, constr, model_validator

from schemas.endpoint_parameters import (
    POLARITY_DESCRIPTION,
    ROLE_DESCRIPTION,
    EndpointParameters,
)
from schemas.enums import Polarity, Role
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
            "substantive coverage; do not enumerate all 7 for "
            "thoroughness."
        ),
    )
    aspects_covered: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Which entries from `aspects` THIS space owns, written "
            "holistically. Compose interlocking aspects into one "
            "phrase — 'darkly funny' is one signal this space "
            "carries, not two; the embedding lands more accurately "
            "when composed signals stay composed than when split and "
            "re-intersected. Read `aspects` (primary: what needs "
            "covering) and `retrieval_intent` (discriminative anchor "
            "when an aspect could surface-match here but the intent "
            "narrows it elsewhere).\n"
            "\n"
            "TEST: if I removed this space from candidates, which "
            "aspects orphan? Name them.\n"
            "\n"
            "NEVER:\n"
            "- BACK-RATIONALIZE. 'Plausible' isn't coverage. Drop "
            "the candidate or commit what's substantively owned.\n"
            "- ENUMERATE 1:1 WITH ASPECTS. Composed aspects fold; "
            "splitting them defeats the embedding."
        ),
    )
    gap: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "What this space MISSES, defined relative to "
            "`aspects_covered` above and the full `aspects` list — "
            "specifically: entries in `aspects` that this space does "
            "NOT pick up (and which other space catches them), or "
            "sub-shades inside an aspect that this space's boundary "
            "redirects elsewhere (occasion → watch_context; craft → "
            "narrative_techniques; scalar reception → metadata, not "
            "here). 'nothing' when `aspects_covered` is a clean fit "
            "for everything in `aspects` this space could plausibly "
            "carry, with no competing adjacency.\n"
            "\n"
            "NEVER:\n"
            "- HEDGE WITHOUT NAMING. Either 'nothing' or specific gap "
            "with destination — no middle.\n"
            "- INVENT GAPS to look thorough.\n"
            "- RESTATE `aspects_covered` IN NEGATIVE FORM. The gap "
            "names what's NOT here, not a re-listing of what is."
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
    "owns (per its SpaceCandidate.aspects_covered) into ONE body; "
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
# Body merge helper. Copied verbatim from the prior implementation —
# generic walk, list concat-dedupe, prose ". " join, nested BaseModel
# recurse. Same-space duplicate recovery uses this.
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
# WeightedSpaceQuery — qualifier-side wrapper around a space entry.
# ---------------------------------------------------------------------------


class WeightedSpaceQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weight: SpaceWeight = Field(
        ...,
        description=(
            "How load-bearing this space's signal is for the trait. "
            "Read `retrieval_intent` (primary source: framing of what "
            "the search positions against) IN LIGHT OF the matching "
            "`SpaceCandidate.aspects_covered` for `query.space` "
            "above (i.e., what this specific space actually carries) "
            "— mechanical mapping; do not re-derive from `aspects` "
            "wholesale.\n"
            "- CENTRAL: retrieval_intent treats the aspects this "
            "space carries (per its SpaceCandidate.aspects_covered) "
            "as defining whether the match holds. Missing this signal "
            "= trait broken. Multiple central spaces are fine when "
            "several axes are equally load-bearing.\n"
            "- SUPPORTING: the aspects this space carries round out "
            "the experience but aren't load-bearing on their own. "
            "Trait stays recognizable without it.\n"
            "\n"
            "All-supporting is acceptable for broad-and-balanced "
            "traits — a truthful signal, not a cop-out. If a space "
            "would be below SUPPORTING (barely-there), drop the entry "
            "instead of including it.\n"
            "\n"
            "TEST: look up the `aspects_covered` on the SpaceCandidate "
            "matching `query.space`. Does `retrieval_intent` name "
            "those aspects as defining the match, or as one "
            "contributor among several? Defining → CENTRAL. "
            "Contributor → SUPPORTING."
        ),
    )
    query: SemanticSpaceEntry = Field(
        ...,
        description=(
            "Space-targeted body — {space + content}. `space` MUST "
            "appear in `space_candidates` (any commit to a space "
            "absent from candidates means exploration was skipped). "
            "`content` is authored against the rules on the matching "
            "entry-wrapper class (PlotEventsEntry / "
            "ViewerExperienceEntry / etc.) — fold every aspect this "
            "space owns per its SpaceCandidate.aspects_covered into "
            "ONE body in the space's ingest-side vocabulary; do not "
            "split coverage across multiple WeightedSpaceQuery "
            "entries with the same `query.space`."
        ),
    )


# ---------------------------------------------------------------------------
# Shared field descriptions for the two top-level parameters classes.
# ---------------------------------------------------------------------------


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
    "- LIST A SPACE whose aspects_covered would be hand-waving "
    "rather than substantively named."
)


# ---------------------------------------------------------------------------
# CarverSemanticParameters — role == CARVER, no weights.
# ---------------------------------------------------------------------------


class CarverSemanticParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aspects: conlist(
        constr(strip_whitespace=True, min_length=1), min_length=1
    ) = Field(..., description=_ASPECTS_DESC)
    space_candidates: conlist(SpaceCandidate, min_length=1) = Field(
        ..., description=_SPACE_CANDIDATES_DESC
    )
    space_queries: conlist(SemanticSpaceEntry, min_length=1) = Field(
        ...,
        description=(
            "Committed minimum set of LOAD-BEARING spaces to search. "
            "Pull exclusively from `space_candidates`. Carver "
            "execution sums elbow-normalized scores across active "
            "spaces and divides by their count — every active space "
            "gets equal vote, so adding a marginal space directly "
            "dilutes the gate. Strict bar: only spaces whose signal "
            "genuinely decides the trait.\n"
            "\n"
            "Fold every aspect a space owns (per its "
            "SpaceCandidate.aspects_covered) into ONE entry's "
            "content. Same-space duplicates merge server-side rather "
            "than fail; emit cleanly when you can.\n"
            "\n"
            "TEST per entry: 'would dropping this space let a "
            "clearly-wrong movie pass the trait?' Yes → keep. No → "
            "drop.\n"
            "\n"
            "NEVER:\n"
            "- COMMIT to a space absent from `space_candidates`.\n"
            "- SPLIT one space's coverage across multiple entries.\n"
            "- PAD with marginal spaces. Equal-vote scoring means "
            "every extra space dilutes proportionally."
        ),
    )

    @model_validator(mode="after")
    def _drop_empty_and_merge_duplicates(self) -> "CarverSemanticParameters":
        # Drop entries whose body produces no embedding text — a
        # space with all-empty sub-fields contributes nothing but
        # noise to the elbow-normalize step downstream.
        kept = [e for e in self.space_queries if e.content.embedding_text().strip()]
        if not kept:
            raise ValueError(
                "space_queries empty after dropping entries with no embedding text"
            )

        grouped: dict = {}
        order: list = []
        for entry in kept:
            if entry.space not in grouped:
                grouped[entry.space] = [entry]
                order.append(entry.space)
            else:
                grouped[entry.space].append(entry)

        if len(order) == len(kept):
            self.space_queries = kept
            return self

        merged: list = []
        for space in order:
            entries = grouped[space]
            if len(entries) == 1:
                merged.append(entries[0])
                continue
            base = entries[0].model_copy()
            base.content = _merge_bodies([e.content for e in entries])
            merged.append(base)
        self.space_queries = merged
        return self


# ---------------------------------------------------------------------------
# QualifierSemanticParameters — role == QUALIFIER, weighted.
# ---------------------------------------------------------------------------


class QualifierSemanticParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aspects: conlist(
        constr(strip_whitespace=True, min_length=1), min_length=1
    ) = Field(..., description=_ASPECTS_DESC)
    space_candidates: conlist(SpaceCandidate, min_length=1) = Field(
        ..., description=_SPACE_CANDIDATES_DESC
    )
    space_queries: conlist(WeightedSpaceQuery, min_length=1) = Field(
        ...,
        description=(
            "Committed set of spaces to search, each with a per-space "
            "weight. Each entry's `query.space` MUST appear in "
            "`space_candidates` above; pull exclusively from that "
            "list. Looser bar than the carver path: SUPPORTING weight "
            "exists for spaces that round out the match without being "
            "load-bearing. Drop the entry if signal is below "
            "SUPPORTING.\n"
            "\n"
            "Fold every aspect a space owns (per its "
            "SpaceCandidate.aspects_covered for that space) into ONE "
            "entry's `query.content`; let the merge validator catch "
            "accidental duplication. Executor takes a normalized "
            "weighted sum — Σ(w·cos)/Σw — across these entries.\n"
            "\n"
            "Weight assignment is read off `retrieval_intent` per the "
            "rule on `WeightedSpaceQuery.weight` — see that field's "
            "description for the CENTRAL vs SUPPORTING test.\n"
            "\n"
            "TEST per entry: 'is the signal real enough that omitting "
            "this space would noticeably degrade the match?' Yes "
            "(decisive) → CENTRAL. Yes (rounds out) → SUPPORTING. "
            "No → drop.\n"
            "\n"
            "NEVER:\n"
            "- COMMIT to a space absent from `space_candidates`.\n"
            "- SPLIT one space's coverage across multiple entries.\n"
            "- INCLUDE a space whose signal is below SUPPORTING."
        ),
    )

    @model_validator(mode="after")
    def _drop_empty_and_merge_duplicates(self) -> "QualifierSemanticParameters":
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
            # Stronger weight wins on collision; bodies merge.
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


# ---------------------------------------------------------------------------
# Endpoint wrappers. role narrowed by Literal so the schema choice
# carries the role unambiguously. Polarity stays on the wrapper —
# bodies describe the target concept regardless of polarity, the
# executor flips at merge time.
# ---------------------------------------------------------------------------


class CarverSemanticEndpointParameters(EndpointParameters):
    role: Literal[Role.CARVER] = Field(..., description=ROLE_DESCRIPTION)
    parameters: CarverSemanticParameters = Field(
        ...,
        description=(
            "Carver semantic payload. Decompose the trait into "
            "aspects, explore which spaces plausibly cover them, then "
            "commit only to spaces whose signal is genuinely load-"
            "bearing — every active space gets equal vote at scoring, "
            "so the bar is strict. Describe the target concept "
            "directly regardless of polarity; the wrapper's polarity "
            "field handles negation."
        ),
    )
    polarity: Polarity = Field(..., description=POLARITY_DESCRIPTION)


class QualifierSemanticEndpointParameters(EndpointParameters):
    role: Literal[Role.QUALIFIER] = Field(..., description=ROLE_DESCRIPTION)
    parameters: QualifierSemanticParameters = Field(
        ...,
        description=(
            "Qualifier semantic payload. Decompose the trait into "
            "aspects, explore which spaces plausibly cover them, then "
            "commit to a weighted set — CENTRAL where retrieval_intent "
            "names a space's signal as decisive, SUPPORTING where it "
            "rounds out without being load-bearing. Describe the "
            "target concept directly regardless of polarity; the "
            "wrapper's polarity field handles negation."
        ),
    )
    polarity: Polarity = Field(..., description=POLARITY_DESCRIPTION)
