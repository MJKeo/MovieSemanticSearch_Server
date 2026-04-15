# Step 1 (flow routing) LLM structured output models.
#
# Classifies a user query into one of three search flows and may
# produce multiple interpretations when the query is genuinely
# ambiguous. See search_improvement_planning/finalized_search_proposal.md
# (Step 1: Flow Routing) for the full design rationale.
#
# No class-level docstrings — Field(description=...) carries the
# compact LLM-facing guidance. Developer notes live in comments above
# the class (per prompt authoring conventions).

from pydantic import BaseModel, ConfigDict, Field, conlist, constr

from schemas.enums import SearchFlow


# Per-interpretation output. Field order follows the model's decision
# chain: cite routing evidence → rewrite intent → classify flow →
# generate display label → extract title.
class QueryInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    routing_signals: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "One short sentence citing the specific query words or "
            "patterns that determined this flow classification."
        ),
    )
    intent_rewrite: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "The user's query rewritten as a complete, concrete "
            "statement of what they are looking for under this "
            "interpretation. Make implicit expectations explicit."
        ),
    )
    flow: SearchFlow = Field(
        ...,
        description=(
            "Which search flow handles this interpretation."
        ),
    )
    display_phrase: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Short phrase (2-8 words) labeling this interpretation "
            "for display in the app. For exact_title: the movie "
            "title. For similarity: 'Movies like [title]'. For "
            "standard: a brief summary of the search intent."
        ),
    )
    title: str | None = Field(
        default=None,
        description=(
            "The movie title extracted from the query, using the "
            "most common fully expanded English-language title form. "
            "Required when flow is exact_title or similarity. Null "
            "for standard."
        ),
    )


# Top-level response from the step 1 flow routing LLM.
class FlowRoutingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    interpretation_analysis: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "One concise sentence: is there a single clear reading "
            "of this query, or are there multiple equally reasonable "
            "interpretations? If multiple, name what makes the query "
            "ambiguous. Do not manufacture ambiguity."
        ),
    )
    interpretations: conlist(
        QueryInterpretation,
        min_length=1,
        max_length=3,
    ) = Field(
        ...,
        description=(
            "1-3 interpretations of the query. Most queries produce "
            "exactly 1. Only produce multiple when the query text "
            "genuinely supports multiple equally reasonable readings. "
            "The first interpretation is the default."
        ),
    )
