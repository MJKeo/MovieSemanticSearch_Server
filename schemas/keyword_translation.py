# Step 3 keyword endpoint structured output model.
#
# Translates a category-handler call (retrieval_intent + expressions)
# routed to the keyword endpoint into a finalized list of
# UnifiedClassification members + a scoring method. Executor fetches
# per-movie hit counts against the backing movie_card columns
# (keyword_ids / source_material_type_ids / concept_tag_ids) and
# combines per `scoring_method`:
#   ANY → 1.0 if movie matches ≥1 finalized member
#   ALL → matched_count / len(finalized_keywords)
#
# Two shapes coexist in this file:
#
# 1. SINGLE-ENDPOINT shape (KeywordQuerySpec / KeywordEndpointParameters)
#    — used by buckets 3/4 when the keyword endpoint owns the entire
#    call. Two-layer internal structure:
#       a. Analysis — `attributes` (decomposition + shortlisting).
#       b. Commitment — `finalized_keywords` + `scoring_method`.
#    Untouched by the walk-then-commit refactor.
#
# 2. MULTI-ENDPOINT shape (KeywordWalk + KeywordQuerySpecSubintent /
#    KeywordEndpointSubintentParameters) — used by buckets 5/6/8.
#    Split across two classes that live in different positions of the
#    bucket schema:
#       a. KeywordWalk — the registry-grounded analysis layer
#          (`attributes` / `potential_keywords`). Lives at the bucket
#          level, emitted BEFORE the coverage_assignments commitment so
#          the LLM walks the registry concretely before committing.
#       b. KeywordQuerySpecSubintent — thin commitment-only spec
#          (`finalized_keywords` + `scoring_method`). Lives inside the
#          per-endpoint `keyword_parameters` slot, populated only when
#          coverage_assignments delegates a slice to this endpoint.
#
# Validator dedupes finalized_keywords server-side. The LLM is told
# to emit duplicates freely if a member is the best fit for multiple
# attributes — asking it to pre-dedupe risks dropping genuine signals.
#
# Direction-agnostic: polarity is stamped on the wrapper post-hoc.
#
# No class-level docstrings or Field descriptions on the inner LLM
# response models — per the "No docstrings on Pydantic classes used
# as LLM response_format" convention, both propagate into the JSON
# schema sent on every API call.

from pydantic import BaseModel, ConfigDict, Field, constr, field_validator

from schemas.endpoint_parameters import EndpointParameters
from schemas.enums import ScoringMethod
from schemas.unified_classification import UnifiedClassification


class PotentialKeyword(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    keyword: UnifiedClassification = Field(
        ...,
        description=(
            "One candidate registry member for the parent `attribute`. "
            "Pick by definitional fit against the attribute, not label "
            "echo from the query text."
        ),
    )

    coverage: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "How `keyword` matches the parent `attribute`: what aspect "
            "it owns + what specifically it misses (or 'fully covered' "
            "when no real gap). One short sentence, concrete to THIS "
            "attribute.\n"
            "\n"
            "NEVER:\n"
            "- HEDGE WITHOUT NAMING. Commit 'fully covered' or name "
            "the specific gap (a sub-form, a cross-family neighbor, a "
            "tonal sub-shade the boundary redirects elsewhere).\n"
            "- BACK-RATIONALIZE. If the only thing you can say is "
            "'plausible', drop the candidate.\n"
            "- GENERALIZE. Coverage is about THIS attribute, not the "
            "keyword's general scope."
        ),
    )


class AttributeAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attribute: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "One distinguishable facet of the call's intent. Read "
            "retrieval_intent + expressions HOLISTICALLY — NOT 1:1 "
            "with expressions. One expression may carry multiple "
            "facets; several expressions may collapse into one. Short "
            "noun-phrase, in user/database vocabulary.\n"
            "\n"
            "Test: would an independent retrieval against this facet "
            "alone hit a meaningful slice of the user's intent? Yes → "
            "distinct attribute. No → fold it into the attribute it "
            "actually belongs to.\n"
            "\n"
            "NEVER:\n"
            "- COPY an expression verbatim. Decompose first.\n"
            "- INVENT facets the inputs don't signal.\n"
            "- SPLIT one facet across multiple attributes."
        ),
    )

    potential_keywords: list[PotentialKeyword] = Field(
        ...,
        min_length=1,
        description=(
            "Registry members that could plausibly answer THIS "
            "`attribute`, each with its coverage analysis. One when "
            "fit is unambiguous; two or three when adjacency is real "
            "(broader vs narrower, cross-family neighbors); more when "
            "the attribute genuinely sits between several.\n"
            "\n"
            "Test per candidate: 'if I dropped this, would the commit "
            "step lose a real routing option?' Yes → keep. No → drop.\n"
            "\n"
            "NEVER:\n"
            "- LIST ONLY ONE when a definitional adjacency competes — "
            "surface it so finalized_keywords is grounded.\n"
            "- PAD with members whose coverage you can't substantively "
            "name.\n"
            "- DUPLICATE a member within one attribute."
        ),
    )


class KeywordQuerySpec(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    # Field order is cognitive scaffolding: analysis grounds the
    # commitment, the commitment grounds the scoring mode. Do not
    # reorder.

    attributes: list[AttributeAnalysis] = Field(
        ...,
        min_length=1,
        description=(
            "Analysis layer. One AttributeAnalysis per distinct facet "
            "of the call's intent, derived holistically from "
            "retrieval_intent + expressions.\n"
            "\n"
            "Coverage: every aspect the inputs signal is owned by "
            "some attribute; every attribute traces back to something "
            "explicit in the inputs. Cardinality follows the inputs — "
            "concrete single-facet calls resolve to one entry; "
            "compound calls resolve to several.\n"
            "\n"
            "NEVER:\n"
            "- MIRROR the expression list 1:1. Decompose first.\n"
            "- DROP a facet silently. If a facet resists candidate "
            "shortlisting, the facet was wrong (too abstract / not "
            "actually independent) — revise it, don't skip it."
        ),
    )

    finalized_keywords: list[UnifiedClassification] = Field(
        ...,
        min_length=1,
        description=(
            "Commitment layer. The MINIMUM set of registry members "
            "whose union covers the span of `attributes`. Pull from "
            "members surfaced in `attributes[*].potential_keywords`. "
            "(A member not previously shortlisted is allowed but "
            "signals the attribute analysis was incomplete.)\n"
            "\n"
            "The attributes do not survive past this layer — only the "
            "deduped union of finalized members reaches execution. "
            "Do NOT think of finalized_keywords as a flattening of "
            "every potential_keyword; it is a fresh minimum-cover "
            "commitment.\n"
            "\n"
            "Test per member: 'if I dropped this, would the remaining "
            "set still cover every attribute this member was the best "
            "fit for?' Yes → drop. No → keep.\n"
            "\n"
            "Validator dedupes server-side. Emit duplicates freely "
            "when the same member is the best fit for multiple "
            "attributes; do NOT pre-dedupe — it risks dropping "
            "genuine multi-attribute signal.\n"
            "\n"
            "NEVER:\n"
            "- INVENT members not in the registry.\n"
            "- PAD past the minimum covering set.\n"
            "- LEAVE EMPTY. Routing committed this call to the "
            "keyword endpoint; if no member fits cleanly, pick the "
            "closest definitionally supported one anyway."
        ),
    )

    scoring_method: ScoringMethod = Field(
        ...,
        description=(
            "Aggregation across `finalized_keywords`, read off "
            "`retrieval_intent`:\n"
            "- ANY → we only care if the movie has at least one of "
            "the finalized members. Movies score equally high for "
            "matching 1+ values. Pick when retrieval_intent treats "
            "the members as substitutable alternatives (paraphrases, "
            "surface-form variants, OR-style framing).\n"
            "- ALL → we care how many finalized members the movie "
            "matches. Movies score higher depending on how many "
            "values they match. Pick when retrieval_intent treats "
            "each member as a distinct facet that matters (AND-style "
            "framing, compound coverage requirement).\n"
            "\n"
            "Test: does retrieval_intent treat the finalized members "
            "as substitutable, or as each-matters? Substitutable → "
            "ANY. Each-matters → ALL.\n"
            "\n"
            "When N=1 the two modes are mathematically identical — "
            "default to ANY and move on. Do NOT re-derive the mode "
            "from the attributes count; cardinality is not the signal."
        ),
    )

    # Server-side dedupe so the LLM can list the same registry member
    # under multiple attributes without penalty (different attributes
    # may genuinely point to the same canonical concept). Order is
    # preserved; the deduped count is what ALL divides by.
    @field_validator("finalized_keywords", mode="after")
    @classmethod
    def _dedupe_finalized_keywords(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for member in value:
            if member in seen:
                continue
            seen.add(member)
            out.append(member)
        return out


class KeywordEndpointParameters(EndpointParameters):
    parameters: KeywordQuerySpec = Field(
        ...,
        description=(
            "Keyword endpoint payload. Decompose the call's "
            "retrieval_intent + expressions into distinct attributes, "
            "shortlist candidates per attribute with coverage prose, "
            "then commit to the minimum set of registry members whose "
            "union covers the attribute span. Read retrieval_intent "
            "for AND/OR framing → scoring. Describe target concepts "
            "directly regardless of polarity — negation is handled on "
            "the wrapper's polarity field."
        ),
    )


# ── Multi-endpoint walk + thin commitment ─────────────────────────
#
# Used when the keyword endpoint is one of several contending for the
# call's intent (buckets 5/6/8). Two classes that live in DIFFERENT
# positions of the bucket schema:
#
#   - KeywordWalk lives at the bucket level, emitted BEFORE the
#     coverage_assignments commitment phase. It carries the
#     registry-grounded analysis of how the keyword endpoint could
#     cover the call's retrieval_intent. Forces concrete registry
#     awareness before any commitment.
#   - KeywordQuerySpecSubintent lives inside the per-endpoint
#     keyword_parameters slot, populated only when the bucket-level
#     coverage_assignments delegates a slice to this endpoint. Carries
#     just the commitment layer (finalized_keywords + scoring_method).
#
# Both reference `keyword_walk.attributes[*].potential_keywords`
# (analysis) and the wrapper's `keyword_retrieval_intent` (the slice
# this endpoint was assigned by coverage_assignments). The thin spec
# does NOT carry its own analysis — that lives upstream in
# KeywordWalk so it grounds the commitment phase that decides whether
# to fire keyword at all.


class KeywordWalk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Reuses the shared AttributeAnalysis class: its descriptor reads
    # off the call's `retrieval_intent + expressions` (in the user
    # message), which is exactly the input the walk grounds itself in.
    # The walk surfaces what the keyword endpoint COULD cover; the
    # commitment that follows it decides whether to fire and on what
    # slice.
    attributes: list[AttributeAnalysis] = Field(
        ...,
        min_length=1,
        description=(
            "Registry-grounded analysis of how the keyword endpoint "
            "could cover this call's intent. One AttributeAnalysis "
            "per distinct facet of the call's retrieval_intent + "
            "expressions, derived holistically from them.\n"
            "\n"
            "This is the GROUNDED walk that precedes the bucket-level "
            "coverage_assignments commitment. Surface every plausible "
            "registry member with concrete coverage prose so the "
            "commitment phase can read off real candidates rather "
            "than abstract optimism. Empty potential_keywords on a "
            "facet is a valid signal that the registry has nothing "
            "useful — the commitment phase is allowed to leave the "
            "facet unowned by keyword (delegating to a sibling "
            "endpoint or naming it as intentionally_uncovered).\n"
            "\n"
            "Coverage: every aspect the call's intent signals is "
            "owned by some attribute; every attribute traces back to "
            "something explicit in the call. Cardinality follows the "
            "intent — concrete single-facet intents resolve to one "
            "entry; compound intents resolve to several.\n"
            "\n"
            "NEVER:\n"
            "- MIRROR the call's phrasing 1:1. Decompose first.\n"
            "- INVENT facets the call doesn't signal.\n"
            "- SPLIT one facet across multiple attributes.\n"
            "- HEDGE in coverage prose. Either name the specific gap "
            "or commit 'fully covered'."
        ),
    )


class KeywordQuerySpecSubintent(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    # Thin commitment-only spec for multi-endpoint contexts. The
    # analysis layer that previously lived here (`attributes`) has
    # been lifted up to KeywordWalk at the bucket level — populated
    # BEFORE coverage_assignments so the commitment phase is grounded
    # in concrete registry candidates. This spec carries only the
    # commitment that fires when coverage_assignments delegated a
    # slice to keyword.

    finalized_keywords: list[UnifiedClassification] = Field(
        ...,
        min_length=1,
        description=(
            "Commitment layer. The MINIMUM set of registry members "
            "whose union covers the slice this endpoint was assigned "
            "by coverage_assignments. Pull from members surfaced in "
            "`keyword_walk.attributes[*].potential_keywords` at the "
            "bucket level above. (A member not previously shortlisted "
            "is allowed but signals the walk was incomplete.)\n"
            "\n"
            "If the walk surfaced no clean fit, the bucket-level "
            "coverage_assignments should NOT have delegated a slice "
            "to keyword in the first place — keyword_parameters would "
            "be null and this spec never instantiates. By the time "
            "this field is being populated, commitment has already "
            "decided keyword fires; commit at least one grounded "
            "member.\n"
            "\n"
            "Test per member: 'if I dropped this, would the remaining "
            "set still cover every attribute this member was the best "
            "fit for?' Yes → drop. No → keep.\n"
            "\n"
            "Validator dedupes server-side. Emit duplicates freely "
            "when the same member is the best fit for multiple "
            "attributes; do NOT pre-dedupe — it risks dropping "
            "genuine multi-attribute signal.\n"
            "\n"
            "NEVER:\n"
            "- INVENT members not in the registry.\n"
            "- PAD past the minimum covering set."
        ),
    )

    scoring_method: ScoringMethod = Field(
        ...,
        description=(
            "Aggregation across `finalized_keywords`, read off the "
            "wrapper's `keyword_retrieval_intent`:\n"
            "- ANY → we only care if the movie has at least one of "
            "the finalized members. Movies score equally high for "
            "matching 1+ values. Pick when keyword_retrieval_intent "
            "treats the members as substitutable alternatives "
            "(paraphrases, surface-form variants, OR-style framing).\n"
            "- ALL → we care how many finalized members the movie "
            "matches. Movies score higher depending on how many "
            "values they match. Pick when keyword_retrieval_intent "
            "treats each member as a distinct facet that matters "
            "(AND-style framing, compound coverage requirement).\n"
            "\n"
            "Test: does keyword_retrieval_intent treat the finalized "
            "members as substitutable, or as each-matters? "
            "Substitutable → ANY. Each-matters → ALL.\n"
            "\n"
            "When N=1 the two modes are mathematically identical — "
            "default to ANY and move on. Do NOT re-derive the mode "
            "from the finalized count; cardinality is not the signal."
        ),
    )

    @field_validator("finalized_keywords", mode="after")
    @classmethod
    def _dedupe_finalized_keywords(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for member in value:
            if member in seen:
                continue
            seen.add(member)
            out.append(member)
        return out


class KeywordEndpointSubintentParameters(EndpointParameters):
    keyword_retrieval_intent: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "The slice of the call's intent this endpoint owns, "
            "committed by the bucket-level coverage_assignments above "
            "(the entry whose endpoint_kind matches keyword). Restate "
            "the assigned slice_description here so the commitment "
            "below has a single, self-contained pointer.\n"
            "\n"
            "Scope: canonical UnifiedClassification registry members "
            "across genre, sub-genre, source material, narrative "
            "mechanics, etc. Structured attributes, named entities, "
            "free-form thematic qualifiers, and awards belong to "
            "their respective endpoints. Every field on this "
            "endpoint's `parameters` reads from this intent (and from "
            "the upstream keyword_walk) rather than from any other "
            "input."
        ),
    )
    parameters: KeywordQuerySpecSubintent = Field(
        ...,
        description=(
            "Keyword endpoint thin commitment payload. Reads off "
            "`keyword_retrieval_intent` (the assigned slice) and "
            "`keyword_walk.attributes[*].potential_keywords` (the "
            "bucket-level grounded analysis above) to commit "
            "finalized_keywords + scoring_method."
        ),
    )
