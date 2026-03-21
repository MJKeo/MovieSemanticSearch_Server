"""
Plot Events generator (Wave 1).

Two-branch design (ADR-033): build_plot_events_prompts() selects
both the user prompt inputs and the system prompt based on whether
the movie has a quality synopsis:

- Synopsis branch (condensation): Synopsis is the primary source.
  LLM condenses it into a shorter summary. Uses SYSTEM_PROMPT_SYNOPSIS.
  Only selected when the first synopsis meets MIN_SYNOPSIS_CHARS.
- Synthesis branch: No synopsis, or synopsis too short to condense
  reliably. LLM unifies summaries, overview, and keywords into a
  coherent plot picture. Uses SYSTEM_PROMPT_SYNTHESIS. Short synopses
  are demoted into the summaries list as supplementary input.

All branching is contained in build_plot_events_prompts(), which
returns (user_prompt, system_prompt). Callers just unpack both.

Response schema: PlotEventsOutput
No provider/model defaults — caller must specify.
"""

import re
from typing import Tuple

from movie_ingestion.metadata_generation.inputs import (
    MovieInputData,
    MultiLineList,
    build_user_prompt,
)
from movie_ingestion.metadata_generation.schemas import PlotEventsOutput
from movie_ingestion.metadata_generation.prompts.plot_events import (
    SYSTEM_PROMPT_SYNOPSIS,
    SYSTEM_PROMPT_SYNTHESIS,
)
from movie_ingestion.metadata_generation.errors import (
    MetadataGenerationError,
    MetadataGenerationEmptyResponseError,
)
from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from implementation.llms.vector_metadata_generation_methods import TokenUsage

GENERATION_TYPE = "plot_events"

# Synopses below this length are too thin to condense reliably (often
# review blurbs or marketing copy rather than chronological plot recounts).
# Derived from the real corpus: 30K synopsis movies show that entries under
# ~1,000 chars are consistently non-plot text, while 1,000+ consistently
# contain named characters performing plot actions. The IMDB parser itself
# expects synopses to be 500-2,000 words (~2,500-15,000 chars), so 1,000
# chars is a conservative floor. ~21% of synopsis movies fall below this.
MIN_SYNOPSIS_CHARS = 1000


def build_plot_events_prompts(movie: MovieInputData) -> Tuple[str, str]:
    """Build both user and system prompts for plot_events generation.

    Branches on whether the movie has a quality synopsis, selecting
    different inputs and system prompts for each case:

    - Synopsis branch (condensation): The synopsis is a comprehensive
      plot recount (>= MIN_SYNOPSIS_CHARS). The LLM condenses it into
      a shorter summary.
      Inputs: title, overview, plot_synopsis, plot_keywords.

    - Synthesis branch: No synopsis, or synopsis too short to condense
      reliably. The LLM unifies multiple partial sources into a coherent
      plot picture. Short synopses are demoted into the summaries list
      as supplementary input rather than discarded.
      Inputs: title, overview, plot_summaries, plot_keywords.

    Returns (user_prompt, system_prompt) so all branching logic is
    contained here. Callers just unpack both prompts.

    Shared by the production generator and the evaluation pipeline so the
    prompt construction logic stays in one place.
    """
    first_synopsis = None
    if movie.plot_synopses:
        first_synopsis = re.sub(r"\n+", " ", movie.plot_synopses[0])

    has_quality_synopsis = first_synopsis is not None and len(first_synopsis) >= MIN_SYNOPSIS_CHARS

    if has_quality_synopsis:
        # Synopsis branch — condense the detailed recount.
        user_prompt = build_user_prompt(
            title=movie.title_with_year(),
            overview=movie.overview or None,
            plot_synopsis=first_synopsis,
            plot_keywords=movie.plot_keywords or None,
        )
        return user_prompt, SYSTEM_PROMPT_SYNOPSIS
    else:
        # Synthesis branch — unify partial sources.
        # Cap at 3 summaries — additional entries add diminishing value
        # relative to the token cost.
        plot_summaries = list(movie.plot_summaries[:3]) if movie.plot_summaries else []

        # If a synopsis exists but was too short to condense, demote it
        # into the summaries list as supplementary input. Prepend so it
        # appears first (it's still the most plot-like text available).
        if first_synopsis is not None and not has_quality_synopsis:
            plot_summaries.insert(0, first_synopsis)

        # plot_summaries uses MultiLineList because each entry is a
        # multi-paragraph text block, not a short keyword.
        user_prompt = build_user_prompt(
            title=movie.title_with_year(),
            overview=movie.overview or None,
            plot_summaries=MultiLineList(plot_summaries) if plot_summaries else None,
            plot_keywords=movie.plot_keywords or None,
        )
        return user_prompt, SYSTEM_PROMPT_SYNTHESIS


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
        provider: Which LLM backend to use.
        model: Model identifier.
        **kwargs: Provider-specific params passed through to the LLM call.

    Returns:
        Tuple of (PlotEventsOutput, TokenUsage).

    Raises:
        MetadataGenerationError: If the LLM call raises an exception.
        MetadataGenerationEmptyResponseError: If the LLM returns None.
    """
    user_prompt, system_prompt = build_plot_events_prompts(movie)
    title_with_year = movie.title_with_year()

    try:
        parsed, input_tokens, output_tokens = await generate_llm_response_async(
            provider=provider,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
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
