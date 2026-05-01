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
# Two-layer shape:
#   1. Analysis — `attributes`. Holistic decomposition of the call's
#      intent into distinct facets, each with shortlisted candidates.
#      Describes; does not commit.
#   2. Commitment — `finalized_keywords` + `scoring_method`. Minimum covering
#      set of registry members + AND/OR aggregation mode read off
#      retrieval_intent. The attribute facets do not survive past
#      this layer; only the deduped union reaches execution.
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

from schemas.endpoint_parameters import (
    POLARITY_DESCRIPTION,
    ROLE_DESCRIPTION,
    EndpointParameters,
)
from schemas.enums import Polarity, Role, ScoringMethod
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
    role: Role = Field(..., description=ROLE_DESCRIPTION)
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
    polarity: Polarity = Field(..., description=POLARITY_DESCRIPTION)
