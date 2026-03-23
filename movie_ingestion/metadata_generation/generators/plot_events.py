"""
Plot Events generator (Wave 1).

Two-branch design (ADR-033): build_plot_events_prompts() selects
both the user prompt inputs and the system prompt based on whether
the movie has a quality synopsis:

- Synopsis branch (condensation): Synopsis is the primary source.
  LLM condenses it into a shorter summary. Uses SYSTEM_PROMPT_SYNOPSIS.
  Only selected when the first synopsis meets MIN_SYNOPSIS_CHARS.
- Synthesis branch: No synopsis, or synopsis too short to condense
  reliably. LLM unifies summaries and overview into a coherent plot
  picture. Uses SYSTEM_PROMPT_SYNTHESIS. Short synopses are demoted
  into the summaries list as supplementary input.

Inputs are limited to title, overview, and plot text (synopsis or
summaries). Plot keywords are deliberately excluded — evaluation
showed they act as hallucination springboards and get incorrectly
treated as plot events.

Output is a single plot_summary field. Setting and character fields
were removed after evaluation showed they add redundancy (setting
already in summary) and analytical burden (motivations/roles better
handled by downstream plot_analysis).

All branching is contained in build_plot_events_prompts(), which
returns (user_prompt, system_prompt). Callers just unpack both.

Response schema: PlotEventsOutput
Provider/model are fixed: OpenAI gpt-5-mini with minimal reasoning
and low verbosity. Chosen via 21-movie evaluation across 6 candidates
— gpt-5-mini had near-zero errors (4.93/5.0 overall, 4.86 groundedness)
while the next-best candidate (qwen3.5-flash) had small but consistent
inference leaps (4.56 overall). The $17 batch-pricing premium over
cheaper models is justified by the plot_summary's critical downstream
role (feeds 4/5 Wave 2 generators and plot_events embedding).
"""

import re
from typing import Tuple

from movie_ingestion.metadata_generation.inputs import (
    MetadataType,
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

GENERATION_TYPE = MetadataType.PLOT_EVENTS

# Fixed provider/model for production generation. Evaluation across 6
# candidates showed gpt-5-mini is the most reliable for this task.
_PROVIDER = LLMProvider.OPENAI
_MODEL = "gpt-5-mini"
_MODEL_KWARGS = {"reasoning_effort": "minimal", "verbosity": "low"}

# Synopses below this length are too thin to condense reliably. Evaluation
# of plot_events output showed that synopses at ~1K chars trigger a 67%
# hallucination rate in the condensation path — the model fills gaps from
# training knowledge when the synopsis is incomplete or thematic. At 4K+
# chars, hallucination drops to 0%. 2,500 chars is the threshold where
# synopses are consistently detailed enough that the condensation path
# produces faithful output without fabrication. Synopses below this are
# demoted into the summaries list and routed through the safer synthesis
# path, which uses a "text consolidation tool" framing that constrains
# hallucination even with sparse input.
MIN_SYNOPSIS_CHARS = 2500


def build_plot_events_prompts(movie: MovieInputData) -> Tuple[str, str]:
    """Build both user and system prompts for plot_events generation.

    Branches on whether the movie has a quality synopsis, selecting
    different inputs and system prompts for each case:

    - Synopsis branch (condensation): The synopsis is a comprehensive
      plot recount (>= MIN_SYNOPSIS_CHARS). The LLM condenses it into
      a shorter summary.
      Inputs: title, overview, plot_synopsis.

    - Synthesis branch: No synopsis, or synopsis too short to condense
      reliably. The LLM unifies multiple partial sources into a coherent
      plot picture. Short synopses are demoted into the summaries list
      as supplementary input rather than discarded.
      Inputs: title, overview, plot_summaries.

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
        )
        return user_prompt, SYSTEM_PROMPT_SYNTHESIS


async def generate_plot_events(
    movie: MovieInputData,
) -> Tuple[PlotEventsOutput, TokenUsage]:
    """Generate plot events metadata for a single movie.

    Builds the user prompt from the movie's plot-related fields, calls
    gpt-5-mini via OpenAI with structured output, and returns the
    parsed result alongside token usage.

    Provider/model are fixed — see module docstring for rationale.

    Args:
        movie: Raw movie input data loaded from the ingestion pipeline.

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
            provider=_PROVIDER,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            response_format=PlotEventsOutput,
            model=_MODEL,
            **_MODEL_KWARGS,
        )
    except Exception as e:
        print(f"{GENERATION_TYPE} generation failed for '{title_with_year}': {e}")
        raise MetadataGenerationError(GENERATION_TYPE, title_with_year, e) from e

    if parsed is None:
        print(f"{GENERATION_TYPE} generation returned None for '{title_with_year}'")
        raise MetadataGenerationEmptyResponseError(GENERATION_TYPE, title_with_year)

    return parsed, TokenUsage(input_tokens, output_tokens, _MODEL)
