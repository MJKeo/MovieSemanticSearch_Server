"""
Plot Events generator (Wave 1).

Async method that generates plot events metadata for a single movie.
Accepts MovieInputData, builds a user prompt, calls the specified LLM
provider via the unified routing method, and returns the parsed
PlotEventsOutput with token usage.

Inputs (from MovieInputData):
    - title_with_year(): "Title (Year)" format
    - overview: TMDB marketing summary
    - plot_summaries: shorter IMDB user-written summaries
    - plot_synopses: first entry only (longest/most detailed plot recount)
    - plot_keywords: plot-specific keywords only (not overall)

Response schema: PlotEventsOutput
Provider and model: caller must specify (no defaults).

See docs/llm_metadata_generation_new_flow.md Section 4.1.
"""

import re
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
from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from implementation.llms.vector_metadata_generation_methods import TokenUsage

GENERATION_TYPE = "plot_events"


def build_plot_events_user_prompt(movie: MovieInputData) -> str:
    """Build the user prompt for plot_events generation from a movie's fields.

    Shared by the production generator and the evaluation pipeline so the
    prompt construction logic stays in one place.

    Uses only the first synopsis (longest/most detailed), caps summaries
    at 3 entries, and collapses embedded newlines in synopses.
    """
    # Only use the first synopsis — it's the longest/most detailed recount
    # and additional entries add redundant token cost.
    first_synopsis = movie.plot_synopses[0] if movie.plot_synopses else None

    # Synopses sometimes contain embedded newlines that waste tokens and
    # can confuse the LLM — collapse them into single spaces.
    if first_synopsis:
        first_synopsis = re.sub(r"\n+", " ", first_synopsis)

    # Cap at 3 summaries — additional entries add diminishing value
    # relative to the token cost.
    plot_summaries = movie.plot_summaries[:3] if movie.plot_summaries else None

    # plot_summaries uses MultiLineList because each entry is a
    # multi-paragraph text block, not a short keyword.
    return build_user_prompt(
        title=movie.title_with_year(),
        overview=movie.overview or None,
        plot_summaries=MultiLineList(plot_summaries) if plot_summaries else None,
        plot_synopsis=first_synopsis,
        plot_keywords=movie.plot_keywords or None,
    )


async def generate_plot_events(
    movie: MovieInputData,
    provider: LLMProvider,
    model: str,
    **kwargs,
) -> Tuple[PlotEventsOutput, TokenUsage]:
    """Generate plot events metadata for a single movie.

    Builds the user prompt from the movie's plot-related fields, calls
    the specified LLM provider with structured output, and returns the
    parsed result alongside token usage.

    Args:
        movie: Raw movie input data loaded from the ingestion pipeline.
        provider: Which LLM backend to use (e.g. LLMProvider.OPENAI).
        model: Model identifier (e.g. "gpt-5-mini", "gemini-2.5-flash").
        **kwargs: Provider-specific params (reasoning_effort, temperature, etc.).

    Returns:
        Tuple of (PlotEventsOutput, TokenUsage).

    Raises:
        MetadataGenerationError: If the LLM call raises an exception.
        MetadataGenerationEmptyResponseError: If the LLM returns None.
    """
    user_prompt = build_plot_events_user_prompt(movie)
    title_with_year = movie.title_with_year()

    try:
        parsed, input_tokens, output_tokens = await generate_llm_response_async(
            provider=provider,
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            response_format=PlotEventsOutput,
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
