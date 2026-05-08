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

from typing import Literal

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

    strengths: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "What this keyword genuinely OWNS of the parent `attribute` "
            "at retrieval time. Frame operationally — would using only "
            "this member retrieve the slice the attribute names?\n"
            "\n"
            "NEVER:\n"
            "- BACK-RATIONALIZE. If the only thing you can say is "
            "'plausible', drop the candidate.\n"
            "- GENERALIZE the keyword's broad scope. Strengths name what "
            "this attribute asks for and this member supplies."
        ),
    )

    weaknesses: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "What this keyword MISSES or OVER-PULLS relative to the "
            "parent `attribute`. Two failure modes — both belong here:\n"
            "- under-coverage: aspects of the attribute this member "
            "doesn't reach (a sub-form, a cross-family neighbor, a "
            "tonal sub-shade redirected elsewhere).\n"
            "- over-coverage: content this member ALSO retrieves beyond "
            "the slice (e.g., SPORT covers running but also football, "
            "basketball, hockey).\n"
            "\n"
            "Suggested vocabulary (not enforced): prefix lines with "
            "'under-coverage:' and/or 'over-coverage:'. Use 'none' only "
            "when the member is a clean fit on both axes — rare.\n"
            "\n"
            "NEVER:\n"
            "- HEDGE WITHOUT NAMING. Either 'none' or specific "
            "gap/over-pull with concrete content.\n"
            "- RESTATE STRENGTHS IN NEGATIVE. Weaknesses name failures, "
            "not the inverse of what was owned.\n"
            "- INVENT WEAKNESSES to look thorough."
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
            "`attribute`, each with strengths + weaknesses. One when "
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
            "- PAD with members whose strengths you can't substantively "
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
            "Aggregation across `finalized_keywords`. Defaults to "
            "ANY; ALL is reserved for genuine plural intent.\n"
            "\n"
            "- ANY (singular intent) → the user's expression names "
            "ONE attribute. The keyword commit may include multiple "
            "registry members because you have converted that one "
            "attribute into several registry surface forms — "
            "paraphrases, alternative routes, sub-form alternatives. "
            "Matching any one is sufficient evidence the user's one "
            "thing is present.\n"
            "- ALL (plural intent) → the user's expression names "
            "MULTIPLE distinct attributes the user wants present "
            "together — separate things, each independently demanded, "
            "compoundable. Each must be matched for the call's "
            "intent to be satisfied.\n"
            "\n"
            "Operational test: read the call's expressions. One "
            "expression with multiple keywords commits to ANY. "
            "Multiple expressions naming genuinely distinct "
            "attributes that the user conjoined may commit to ALL.\n"
            "\n"
            "When N=1 (one finalized keyword) the two modes are "
            "mathematically identical — default to ANY and move on. "
            "Do NOT re-derive the mode from the keyword count; "
            "cardinality is not the signal."
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
            "shortlist candidates per attribute with strengths + "
            "weaknesses, then commit to the minimum set of registry "
            "members whose "
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
#     coverage_commitments commitment phase. It carries the
#     registry-grounded analysis of how the keyword endpoint could
#     cover the call's retrieval_intent. Each candidate carries an
#     explicit verdict (commit/abstain) plus a single-claim reason.
#   - KeywordQuerySpecSubintent lives inside the per-endpoint
#     keyword_parameters slot, populated only when the bucket-level
#     coverage_commitments delegates a slice to this endpoint. Its
#     `finalized_keywords` is server-derived from the walk's verdict
#     commits — the LLM no longer emits the commit list directly.
#
# The Phase 5 schema enforcement (verdict_reason → verdict on every
# PotentialKeyword in the multi-endpoint walk) makes the abstention
# pathway a hard structural choice between two valid outputs, instead
# of soft prose preference. `finalized_keywords` becomes a derivation
# of those verdicts, eliminating the LLM's ability to commit a member
# whose verdict is "abstain" or to commit a member not in the walk.


class PotentialKeywordWithVerdict(PotentialKeyword):
    """Multi-endpoint walk variant of PotentialKeyword.

    Adds a verdict_reason → verdict pair AFTER the strengths /
    weaknesses fields so the LLM's prose reasoning lands first and
    the structural commit reads off it. The bucket-level model
    validator collects every potential_keyword whose verdict is
    "commit" and overwrites the downstream KeywordQuerySpecSubintent's
    finalized_keywords with that derived list.
    """

    # Field order: keyword → strengths → weaknesses (inherited) →
    # verdict_reason → verdict. Reasoning before the commit so prose
    # serves as evidence, not post-hoc justification.

    verdict_reason: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "One short sentence justifying the verdict below, citing "
            "the strengths or weaknesses text already written above. "
            "Single-claim only — do NOT write OR-disjunctions ('clean "
            "superset OR near-clean'); pick one claim and commit.\n"
            "\n"
            "For 'commit': name the single superset condition the "
            "candidate satisfies, citing the strengths text. Example "
            "shape (paraphrase the framing, do not echo): 'covers the "
            "named attribute with no gap; over-pull is acceptable.'\n"
            "\n"
            "For 'abstain': name exactly ONE of the three failure "
            "modes:\n"
            "- gaps: some attribute-satisfying movies carry none of "
            "the tags this candidate represents.\n"
            "- stretching: the tag names something semantically "
            "adjacent to the attribute rather than the attribute "
            "itself.\n"
            "- dominated-by-sibling: another candidate in this same "
            "walk covers strictly more of the attribute with no "
            "additional weakness, making this one redundant.\n"
            "\n"
            "Cite the weaknesses text or the dominating sibling. Do "
            "NOT generate fresh reasoning at the verdict step — read "
            "what you already wrote and pin it to one claim."
        ),
    )

    verdict: Literal["commit", "abstain"] = Field(
        ...,
        description=(
            "Active commit choice for this candidate, read off the "
            "verdict_reason just written.\n"
            "\n"
            "- 'commit' → this candidate (alone or in ANY-mode union "
            "with the other commits in this walk) passes the keyword "
            "endpoint's superset test. Every movie that genuinely "
            "satisfies the parent attribute carries at least one of "
            "the committed tags. Over-pull is acceptable.\n"
            "- 'abstain' → this candidate fails the superset bar via "
            "one of the three failure modes named in verdict_reason "
            "(gaps / stretching / dominated-by-sibling). The "
            "candidate stays in the walk for inspection but is "
            "EXCLUDED from finalized_keywords.\n"
            "\n"
            "Default to 'abstain' when the analysis is ambiguous. "
            "'commit' requires a single named superset condition. "
            "Abstention is not a fallback for difficulty; it is the "
            "principled outcome when committing would harm retrieval. "
            "finalized_keywords is server-derived from the verdicts "
            "across this walk's potential_keywords — the LLM does NOT "
            "populate finalized_keywords directly."
        ),
    )


class AttributeAnalysisWithVerdict(AttributeAnalysis):
    """Multi-endpoint walk variant of AttributeAnalysis.

    Overrides potential_keywords to use PotentialKeywordWithVerdict
    so each candidate carries an explicit verdict. Single-endpoint
    keyword buckets continue to use the base AttributeAnalysis class
    with no verdict fields — the verdict pathway is multi-endpoint-
    only per the Phase 5 scope.
    """

    potential_keywords: list[PotentialKeywordWithVerdict] = Field(
        ...,
        min_length=1,
        description=(
            "Registry members that could plausibly answer THIS "
            "`attribute`, each with strengths + weaknesses + a verdict "
            "(commit/abstain). One when fit is unambiguous; two or "
            "three when adjacency is real (broader vs narrower, "
            "cross-family neighbors); more when the attribute genuinely "
            "sits between several.\n"
            "\n"
            "Surface every plausibly useful candidate. The verdict "
            "field on each candidate is the active commit choice; "
            "abstain is sanctioned and frequent — finalized_keywords "
            "is derived from verdict commits server-side, so a "
            "candidate that abstains is correctly excluded.\n"
            "\n"
            "NEVER:\n"
            "- LIST ONLY ONE when a definitional adjacency competes — "
            "surface it so the verdict choice is grounded.\n"
            "- PAD with members whose strengths you can't substantively "
            "name.\n"
            "- DUPLICATE a member within one attribute."
        ),
    )


class KeywordWalk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Uses AttributeAnalysisWithVerdict (multi-endpoint variant). Each
    # candidate carries an explicit verdict_reason → verdict commitment
    # that the bucket-level model validator harvests to derive
    # finalized_keywords on the downstream subintent.
    attributes: list[AttributeAnalysisWithVerdict] = Field(
        ...,
        min_length=1,
        description=(
            "Registry-grounded analysis of how the keyword endpoint "
            "could cover this call's intent. One AttributeAnalysis "
            "per distinct facet of the call's retrieval_intent + "
            "expressions, derived holistically from them.\n"
            "\n"
            "This is the GROUNDED walk that precedes the bucket-level "
            "coverage_exploration / coverage_assignments commitment. "
            "Surface every plausible registry member with concrete "
            "strengths + weaknesses so the commitment phase can read "
            "off real candidates rather than abstract optimism. Empty "
            "potential_keywords on a facet is a valid signal that the "
            "registry has nothing useful — the commitment phase is "
            "allowed to leave the facet unowned by keyword (delegating "
            "to a sibling endpoint).\n"
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
            "- HEDGE in strengths or weaknesses. Either name the "
            "specific content or commit 'none'."
        ),
    )


class KeywordQuerySpecSubintent(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    # Thin commitment-only spec for multi-endpoint contexts. The
    # analysis layer that previously lived here (`attributes`) has
    # been lifted up to KeywordWalk at the bucket level — populated
    # BEFORE coverage_commitments so the commitment phase is grounded
    # in concrete registry candidates. This spec carries only the
    # commitment that fires when coverage_commitments delegated a
    # slice to keyword.
    #
    # Phase 5: finalized_keywords is server-DERIVED from the upstream
    # walk's verdict commits. The LLM should emit an empty list `[]`
    # for this field; a bucket-level model validator overwrites it
    # post-parse with `[pk.keyword for pk in walk.attributes[*].
    # potential_keywords if pk.verdict == "commit"]` (deduped). This
    # eliminates the LLM's chance to commit a non-walked member or a
    # verdict-abstained one.

    finalized_keywords: list[UnifiedClassification] = Field(
        default_factory=list,
        description=(
            "Server-DERIVED commitment layer — the LLM should emit "
            "an empty list `[]` for this field. It is overwritten "
            "post-parse with the union of `keyword_walk.attributes[*]"
            ".potential_keywords` whose `verdict == \"commit\"`. The "
            "verdicts you wrote on the walk above ARE the commit; "
            "this field exists only to carry the derived list to "
            "the executor.\n"
            "\n"
            "If you find yourself wanting to populate this directly, "
            "go back to the walk and adjust the verdicts there — "
            "verdicts are the single source of truth. Do not "
            "duplicate the commit logic here."
        ),
    )

    scoring_method: ScoringMethod = Field(
        ...,
        description=(
            "Aggregation across `finalized_keywords`, read off the "
            "wrapper's `keyword_retrieval_intent`. Defaults to ANY; "
            "ALL is reserved for genuine plural intent.\n"
            "\n"
            "- ANY (singular intent) → keyword_retrieval_intent names "
            "ONE attribute. The keyword commit may include multiple "
            "registry members because you have converted that one "
            "attribute into several registry surface forms — "
            "paraphrases, alternative routes, sub-form alternatives. "
            "Matching any one is sufficient evidence the user's one "
            "thing is present.\n"
            "- ALL (plural intent) → keyword_retrieval_intent names "
            "MULTIPLE distinct attributes the user wants present "
            "together — separate things, each independently demanded, "
            "compoundable. Each must be matched for the call's "
            "intent to be satisfied.\n"
            "\n"
            "Operational test: read keyword_retrieval_intent. One "
            "expression with multiple keywords commits to ANY. "
            "Multiple expressions naming genuinely distinct "
            "attributes that the user conjoined may commit to ALL.\n"
            "\n"
            "When N=1 (one finalized keyword) the two modes are "
            "mathematically identical — default to ANY and move on. "
            "Do NOT re-derive the mode from the finalized count; "
            "cardinality is not the signal."
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
