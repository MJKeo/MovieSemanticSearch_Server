"""
Plot Events generator (Wave 1).

Async method that generates plot events metadata for a single movie.
Accepts MovieInputData, builds a user prompt, calls the async OpenAI
structured output API, and returns the parsed PlotEventsOutput with
token usage.

Inputs (from MovieInputData):
    - title_with_year(): "Title (Year)" format
    - overview: TMDB marketing summary
    - plot_summaries: shorter IMDB user-written summaries
    - plot_synopses: first entry only (longest/most detailed plot recount)
    - plot_keywords: plot-specific keywords only (not overall)

Response schema: PlotEventsOutput
Model: gpt-5-mini, reasoning_effort: minimal

See docs/llm_metadata_generation_new_flow.md Section 4.1.
"""

from typing import Tuple

from movie_ingestion.metadata_generation.inputs import (
    MovieInputData,
    MultiLineList,
    build_user_prompt,
)
from movie_ingestion.metadata_generation.schemas import PlotEventsOutput
from movie_ingestion.metadata_generation.prompts.plot_events import SYSTEM_PROMPT
from movie_ingestion.metadata_generation.errors import (
    MetadataGenerationError,
    MetadataGenerationEmptyResponseError,
)
from implementation.llms.generic_methods import generate_openai_response_async
from implementation.llms.vector_metadata_generation_methods import TokenUsage

GENERATION_TYPE = "plot_events"


async def generate_plot_events(
    movie: MovieInputData,
) -> Tuple[PlotEventsOutput, TokenUsage]:
    """Generate plot events metadata for a single movie.

    Builds the user prompt from the movie's plot-related fields, calls
    gpt-5-mini via the async OpenAI client with structured output, and
    returns the parsed result alongside token usage.

    Args:
        movie: Raw movie input data loaded from the ingestion pipeline.

    Returns:
        Tuple of (PlotEventsOutput, TokenUsage).

    Raises:
        MetadataGenerationError: If the LLM call raises an exception.
        MetadataGenerationEmptyResponseError: If the LLM returns None.
    """
    title_with_year = movie.title_with_year()

    # Only use the first synopsis — it's the longest/most detailed recount
    # and additional entries add redundant token cost.
    first_synopsis = movie.plot_synopses[0] if movie.plot_synopses else None

    # Build user prompt — plot_summaries uses MultiLineList because each
    # entry is a multi-paragraph text block, not a short keyword.
    user_prompt = build_user_prompt(
        title=title_with_year,
        overview=movie.overview or None,
        plot_summaries=MultiLineList(movie.plot_summaries) if movie.plot_summaries else None,
        plot_synopsis=first_synopsis,
        plot_keywords=movie.plot_keywords or None,
    )

    try:
        parsed, input_tokens, output_tokens = await generate_openai_response_async(
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            response_format=PlotEventsOutput,
            model="gpt-5-mini",
            reasoning_effort="minimal",
            verbosity="low",
        )
    except Exception as e:
        print(f"{GENERATION_TYPE} generation failed for '{title_with_year}': {e}")
        raise MetadataGenerationError(GENERATION_TYPE, title_with_year, e) from e

    if parsed is None:
        print(f"{GENERATION_TYPE} generation returned None for '{title_with_year}'")
        raise MetadataGenerationEmptyResponseError(GENERATION_TYPE, title_with_year)

    return parsed, TokenUsage(input_tokens, output_tokens, "gpt-5-mini")
