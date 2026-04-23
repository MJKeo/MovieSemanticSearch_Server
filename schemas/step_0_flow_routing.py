# Step 0 (flow routing) LLM structured output models.
#
# Step 0 is a narrow classifier that decides which of the three major
# search flows (exact_title, similarity, standard) should execute for
# a given user query, and carries the title payload each non-standard
# flow needs. It runs in parallel with Step 1 on the raw query; the
# merge happens in code afterward.
#
# The schema follows the observations-first pattern: extractive
# observation fields come first (titles seen, with inline ambiguity
# reasoning per title; qualifiers), then three per-flow decision
# fields with the payload needed to execute, then a primary_flow
# enum identifying the most likely user intent for result-list
# ordering.
#
# See search_improvement_planning/steps_1_2_improving.md (Step 0: Flow
# Routing) for the full design rationale.
#
# No class-level docstrings or Field descriptions — all LLM-facing
# guidance lives in the system prompt (search_v2/step_0.py).
# Developer notes live in comments above the class.

from pydantic import BaseModel, ConfigDict, Field, constr, model_validator

from schemas.enums import SearchFlow


# A single title span observed in the query.
# - span_text: the span as it appeared in the query, preserving the
#   user's wording and any typos.
# - most_likely_canonical_title: the resolved real movie title the LLM
#   judges the span is most likely referring to. Typo correction
#   happens only when the typed form is not itself a plausible English
#   word — the rule lives in the system prompt.
# - ambiguity_potential: brief LLM reasoning about whether the span
#   could plausibly also represent a non-title query (genre, mood,
#   descriptor, etc.) or whether the LLM is uncertain the user was
#   actually searching for this title. Empty ambiguity is acceptable
#   — when the title is unambiguous, the LLM states that directly.
#   This field scaffolds the flow decision: a high-ambiguity title
#   paired with real standard-search value fires both flows; a
#   low-ambiguity title (even if not 100% certain) does not.
class TitleObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    span_text: constr(strip_whitespace=True, min_length=1) = Field(...)
    most_likely_canonical_title: constr(strip_whitespace=True, min_length=1) = Field(...)
    ambiguity_potential: constr(strip_whitespace=True, min_length=1) = Field(...)


# Exact-title flow decision payload.
# - should_be_searched: whether the exact-title search should execute.
# - exact_title_to_search: the canonical title the search would use.
#   Always populated with the LLM's best candidate even when
#   should_be_searched is False (can be empty string only when there
#   is genuinely no candidate title in the query). When
#   should_be_searched is True, the string must be non-empty.
class ExactTitleFlowData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    should_be_searched: bool = Field(...)
    exact_title_to_search: str = Field(...)


# Similarity flow decision payload.
# - should_be_searched: whether the similarity search should execute.
#   Hard rule: must be False whenever `qualifiers` is non-empty.
#   Similarity is an "only mode" — once the query carries
#   descriptive qualifiers, it belongs in the standard flow where
#   those qualifiers can be applied.
# - similar_search_title: the canonical reference title the search
#   would use. Always populated with the LLM's best candidate even
#   when should_be_searched is False (can be empty string only when
#   there is genuinely no reference title in the query). When
#   should_be_searched is True, the string must be non-empty.
class SimilarityFlowData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    should_be_searched: bool = Field(...)
    similar_search_title: str = Field(...)


# Top-level Step 0 response.
#
# Field order follows the FACTS → DECISION pattern so earlier reasoning
# is frozen by the time later fields are generated:
# 1. titles_observed — every title span the LLM recognizes in the
#    query, each with inline ambiguity reasoning. Titles that appear
#    inside a similarity frame are still recorded here (so the title
#    extraction is always visible); whether similarity executes is
#    decided separately in similarity_flow_data.
# 2. qualifiers — every non-title, non-similarity-framing qualifier
#    phrase quoted from the query (genre words, mood adjectives, year
#    references, streaming, ratings, entities used as filters).
# 3. exact_title_flow_data — structured decision for the exact-title
#    flow (should_be_searched + exact_title_to_search).
# 4. similarity_flow_data — structured decision for the similarity
#    flow (should_be_searched + similar_search_title). Blocked by the
#    presence of any qualifier.
# 5. enable_primary_flow — boolean controlling the standard (default)
#    search flow. Intentionally asymmetric with the other two flow
#    fields because the standard flow has no title payload.
# 6. primary_flow — the most likely user intent. Used by downstream
#    assembly to order rendered result lists (primary flow's results
#    first). Must correspond to a firing flow; enforced by validator.
class Step0Response(BaseModel):
    model_config = ConfigDict(extra="forbid")

    titles_observed: list[TitleObservation] = Field(...)
    qualifiers: list[str] = Field(...)
    exact_title_flow_data: ExactTitleFlowData = Field(...)
    similarity_flow_data: SimilarityFlowData = Field(...)
    enable_primary_flow: bool = Field(...)
    primary_flow: SearchFlow = Field(...)

    # At least one flow must fire. An all-false output is invalid
    # because every query must route somewhere — the standard flow is
    # the fallback when nothing else fits.
    @model_validator(mode="after")
    def validate_at_least_one_flow(self) -> "Step0Response":
        if (
            not self.exact_title_flow_data.should_be_searched
            and not self.similarity_flow_data.should_be_searched
            and not self.enable_primary_flow
        ):
            raise ValueError(
                "at least one flow must fire (exact_title_flow_data, "
                "similarity_flow_data, or enable_primary_flow)."
            )
        return self

    # primary_flow must correspond to a flow that is actually firing.
    # This catches the LLM nominating a flow as primary while leaving
    # its should_be_searched/enable bit off, which would break
    # downstream ordering.
    @model_validator(mode="after")
    def validate_primary_matches_firing_flow(self) -> "Step0Response":
        if (
            self.primary_flow == SearchFlow.EXACT_TITLE
            and not self.exact_title_flow_data.should_be_searched
        ):
            raise ValueError(
                "primary_flow is exact_title but "
                "exact_title_flow_data.should_be_searched is false."
            )
        if (
            self.primary_flow == SearchFlow.SIMILARITY
            and not self.similarity_flow_data.should_be_searched
        ):
            raise ValueError(
                "primary_flow is similarity but "
                "similarity_flow_data.should_be_searched is false."
            )
        if (
            self.primary_flow == SearchFlow.STANDARD
            and not self.enable_primary_flow
        ):
            raise ValueError(
                "primary_flow is standard but enable_primary_flow is false."
            )
        return self

    # Similarity is an "only mode": it cannot execute when the query
    # carries descriptive qualifiers. Those belong in the standard
    # flow.
    @model_validator(mode="after")
    def validate_similarity_requires_no_qualifiers(self) -> "Step0Response":
        if self.similarity_flow_data.should_be_searched and self.qualifiers:
            raise ValueError(
                "similarity_flow_data.should_be_searched must be false "
                "when qualifiers are present."
            )
        return self

    # When a flow is marked should_be_searched, its title payload must
    # be a non-empty string — otherwise the downstream lookup has
    # nothing to search on. When should_be_searched is False, the
    # string may be empty (no candidate available) or populated (the
    # LLM's best guess, preserved for debugging).
    @model_validator(mode="after")
    def validate_flow_titles_when_searched(self) -> "Step0Response":
        if (
            self.exact_title_flow_data.should_be_searched
            and not self.exact_title_flow_data.exact_title_to_search.strip()
        ):
            raise ValueError(
                "exact_title_to_search must be a non-empty title string "
                "when exact_title_flow_data.should_be_searched is true."
            )
        if (
            self.similarity_flow_data.should_be_searched
            and not self.similarity_flow_data.similar_search_title.strip()
        ):
            raise ValueError(
                "similar_search_title must be a non-empty reference title "
                "when similarity_flow_data.should_be_searched is true."
            )
        return self
