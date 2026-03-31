"""
Watch Context generator (Wave 2).

Async method that generates watch context metadata for a single movie.
Extracts WHY and WHEN someone would choose to watch this movie. Powers
queries like "date night movie" or "something to watch high."

CRITICAL DESIGN DECISION: Watch context receives ZERO plot information.
No overview, no plot_synopsis. It answers "watch this if you want X
attributes" -- not "watch this if you want these specific events."
Plot detail anchors the model on narrative events rather than
experiential attributes.

Inputs:
    - movie (MovieInputData): raw fields for title, overall_keywords, maturity
    - genre_signatures: LLM-refined genre phrases from plot_analysis (Wave 2).
      Falls back to raw movie.genres when unavailable.
    - emotional_observations: from Wave 1 reception extraction zone (may be None)
    - craft_observations: from Wave 1 reception extraction zone (may be None)
    - thematic_observations: from Wave 1 reception extraction zone (may be None)

Removed inputs (vs current system):
    - overview: no plot info in watch context
    - plot_keywords: not relevant to viewing occasions
    - merged_keywords: replaced by overall_keywords only
    - review_insights_brief: replaced by individual observation fields
    - reception_summary / audience_reception_attributes / featured_reviews:
      subsumed by individual observation fields

Skip condition: genre_signatures OR genres >= 1. Enforced by
    pre_consolidation. All other inputs are enrichments.

Response schema: WatchContextOutput (no justifications) by default.
    WatchContextWithJustificationsOutput available for evaluation.

Provider/model defaults: OpenAI gpt-5-mini, reasoning_effort: medium
    (matching current system; will be re-evaluated).

See docs/llm_metadata_generation_new_flow.md Section 5.3.
See evaluation_data/watch_context_eval_guide.md for input contract rationale.
"""

from typing import Tuple

from movie_ingestion.metadata_generation.inputs import (
    MetadataType,
    MovieInputData,
    build_user_prompt,
)
from movie_ingestion.metadata_generation.schemas import WatchContextOutput
from movie_ingestion.metadata_generation.prompts.watch_context import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_WITH_JUSTIFICATIONS,  # noqa: F401 — exported for evaluation pipeline
)
from movie_ingestion.metadata_generation.errors import (
    MetadataGenerationError,
    MetadataGenerationEmptyResponseError,
)
from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from implementation.llms.vector_metadata_generation_methods import TokenUsage

GENERATION_TYPE = MetadataType.WATCH_CONTEXT

# Production defaults — matching current system (gpt-5-mini with medium reasoning).
# Will be re-evaluated via the evaluation pipeline and updated to the winner.
_DEFAULT_PROVIDER = LLMProvider.OPENAI
_DEFAULT_MODEL = "gpt-5-mini"


def build_watch_context_user_prompt(
    movie: MovieInputData,
    genre_signatures: list[str] | None = None,
    emotional_observations: str | None = None,
    craft_observations: str | None = None,
    thematic_observations: str | None = None,
) -> str:
    """Build the user prompt for watch_context generation from a movie's fields.

    Shared by the production generator and the evaluation pipeline so the
    prompt construction logic stays in one place.

    CRITICAL: No plot information is included. Watch context operates on
    experiential signals only (observations, genres, keywords, maturity).

    Genre input uses genre_signatures (LLM-refined from plot_analysis) when
    available, falling back to raw genres. Primary observation fields use
    "not available" when absent so the model sees explicit absence signals.

    Inputs are labeled for the LLM to match the SYSTEM_PROMPT's INPUTS
    section. Secondary inputs (overall_keywords, maturity_summary) are
    omitted when empty.
    """
    # Genre input: prefer genre_signatures, fall back to raw genres
    genre_input = genre_signatures if genre_signatures else (movie.genres or None)

    return build_user_prompt(
        title=movie.title_with_year(),
        genre_signatures=genre_input,
        overall_keywords=movie.overall_keywords or None,
        maturity_summary=movie.maturity_summary(),
        emotional_observations=emotional_observations or "not available",
        craft_observations=craft_observations or "not available",
        thematic_observations=thematic_observations or "not available",
    )


async def generate_watch_context(
    movie: MovieInputData,
    genre_signatures: list[str] | None = None,
    emotional_observations: str | None = None,
    craft_observations: str | None = None,
    thematic_observations: str | None = None,
    provider: LLMProvider = _DEFAULT_PROVIDER,
    model: str = _DEFAULT_MODEL,
    system_prompt: str = SYSTEM_PROMPT,
    response_format: type = WatchContextOutput,
    **kwargs,
) -> Tuple[WatchContextOutput, TokenUsage]:
    """Generate watch context metadata for a single movie.

    Builds the user prompt from the movie's experiential fields plus
    Wave 1 observation fields and Wave 2 genre_signatures, calls the
    specified LLM provider with structured output, and returns the
    parsed result alongside token usage.

    No plot_synopsis parameter — watch context deliberately receives zero
    plot information (Decision 2 in the redesigned flow spec).

    Defaults to OpenAI gpt-5-mini with reasoning_effort: medium (matching
    the current system). Callers can override provider/model/kwargs to
    test different configurations during evaluation.

    Args:
        movie: Raw movie input data loaded from the ingestion pipeline.
        genre_signatures: LLM-refined genre phrases from plot_analysis.
            Falls back to raw movie.genres when None.
        emotional_observations: Emotional tone/mood observations from
            Wave 1 reception extraction zone. May be None.
        craft_observations: Performance/cinematography/craft observations
            from Wave 1 reception extraction zone. May be None.
        thematic_observations: Theme/meaning observations from Wave 1
            reception extraction zone. May be None.
        provider: Which LLM backend to use. Defaults to OPENAI.
        model: Model identifier. Defaults to "gpt-5-mini".
        **kwargs: Provider-specific params (e.g. reasoning_effort, temperature).

    Returns:
        Tuple of (WatchContextOutput, TokenUsage).

    Raises:
        MetadataGenerationError: If the LLM call raises an exception.
        MetadataGenerationEmptyResponseError: If the LLM returns None.
    """
    user_prompt = build_watch_context_user_prompt(
        movie, genre_signatures, emotional_observations,
        craft_observations, thematic_observations,
    )
    title_with_year = movie.title_with_year()

    try:
        parsed, input_tokens, output_tokens = await generate_llm_response_async(
            provider=provider,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            response_format=response_format,
            model=model,
            **kwargs,
        )
    except Exception as e:
        print(f"{GENERATION_TYPE} generation failed for '{title_with_year}': {e}")
        raise MetadataGenerationError(GENERATION_TYPE, title_with_year, e) from e

    if parsed is None:
        print(f"{GENERATION_TYPE} generation returned None for '{title_with_year}'")
        raise MetadataGenerationEmptyResponseError(GENERATION_TYPE, title_with_year)

    return parsed, TokenUsage(input_tokens, output_tokens, model)
