"""
Viewer Experience generator (Wave 2).

Async method that generates viewer experience metadata for a single movie.
Extracts what it FEELS LIKE to watch the movie -- emotional, sensory,
cognitive experience. Powers queries like "edge of your seat thriller"
or "cozy feel-good movie."

Inputs:
    - movie (MovieInputData): raw fields for title, genres, keywords, maturity
    - plot_synopsis: from Wave 1 plot_events output (may be None)
    - review_insights_brief: from Wave 1 reception output (may be None)

Removed inputs (vs current system):
    - overview: superseded by plot_synopsis
    - plot_keywords / overall_keywords as separate inputs: merged via
      movie.merged_keywords()
    - maturity_rating / maturity_reasoning / parental_guide_items:
      consolidated via movie.maturity_summary()
    - reception_summary / audience_reception_attributes / featured_reviews:
      subsumed by review_insights_brief

Skip condition: requires plot_synopsis OR review_insights_brief.
    Can run without plot data if review data exists -- reviews carry
    strong emotional/tonal signal independently.

Response schema: ViewerExperienceOutput (no justifications) by default.
    ViewerExperienceWithJustificationsOutput available for evaluation.

Provider/model defaults: OpenAI gpt-5-mini, reasoning_effort: low
    (matching current system; will be re-evaluated).

See docs/llm_metadata_generation_new_flow.md Section 5.2.
"""

from typing import Tuple

from movie_ingestion.metadata_generation.inputs import (
    MetadataType,
    MovieInputData,
    build_user_prompt,
)
from movie_ingestion.metadata_generation.schemas import ViewerExperienceOutput
from movie_ingestion.metadata_generation.prompts.viewer_experience import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_WITH_JUSTIFICATIONS,
)
from movie_ingestion.metadata_generation.errors import (
    MetadataGenerationError,
    MetadataGenerationEmptyResponseError,
)
from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from implementation.llms.vector_metadata_generation_methods import TokenUsage

GENERATION_TYPE = MetadataType.VIEWER_EXPERIENCE

# Production defaults — matching current system (gpt-5-mini with low reasoning).
# Will be re-evaluated via the evaluation pipeline and updated to the winner.
_DEFAULT_PROVIDER = LLMProvider.OPENAI
_DEFAULT_MODEL = "gpt-5-mini"


def build_viewer_experience_user_prompt(
    movie: MovieInputData,
    plot_synopsis: str | None,
    review_insights_brief: str | None,
) -> str:
    """Build the user prompt for viewer_experience generation from a movie's fields.

    Shared by the production generator and the evaluation pipeline so the
    prompt construction logic stays in one place.

    Inputs are labeled for the LLM to match the SYSTEM_PROMPT's INPUTS
    section. None values and empty lists are skipped by build_user_prompt.
    """
    return build_user_prompt(
        title=movie.title_with_year(),
        genres=movie.genres or None,
        plot_synopsis=plot_synopsis,
        merged_keywords=movie.merged_keywords() or None,
        maturity_summary=movie.maturity_summary(),
        review_insights_brief=review_insights_brief,
    )


async def generate_viewer_experience(
    movie: MovieInputData,
    plot_synopsis: str | None = None,
    review_insights_brief: str | None = None,
    provider: LLMProvider = _DEFAULT_PROVIDER,
    model: str = _DEFAULT_MODEL,
    system_prompt: str = SYSTEM_PROMPT,
    response_format: type = ViewerExperienceOutput,
    **kwargs,
) -> Tuple[ViewerExperienceOutput, TokenUsage]:
    """Generate viewer experience metadata for a single movie.

    Builds the user prompt from the movie's fields plus Wave 1 outputs,
    calls the specified LLM provider with structured output, and returns
    the parsed result alongside token usage.

    Defaults to OpenAI gpt-5-mini with reasoning_effort: low (matching
    the current system). Callers can override provider/model/kwargs to
    test different configurations during evaluation.

    Args:
        movie: Raw movie input data loaded from the ingestion pipeline.
        plot_synopsis: Plot summary from Wave 1 plot_events. May be None
            if plot_events failed or was skipped.
        review_insights_brief: Dense observation paragraph from Wave 1
            reception. May be None if reception failed or was skipped.
        provider: Which LLM backend to use. Defaults to OPENAI.
        model: Model identifier. Defaults to "gpt-5-mini".
        **kwargs: Provider-specific params (e.g. reasoning_effort, temperature).

    Returns:
        Tuple of (ViewerExperienceOutput, TokenUsage).

    Raises:
        MetadataGenerationError: If the LLM call raises an exception.
        MetadataGenerationEmptyResponseError: If the LLM returns None.
    """
    user_prompt = build_viewer_experience_user_prompt(movie, plot_synopsis, review_insights_brief)
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
