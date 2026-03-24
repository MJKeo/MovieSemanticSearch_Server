"""
Plot Analysis generator (Wave 2).

Async method that generates plot analysis metadata for a single movie.
Extracts WHAT TYPE OF STORY this is -- themes, lessons, elevator pitch,
genre signatures, character arcs, conflict type, and a generalized
plot overview.

Inputs:
    - movie (MovieInputData): raw fields for title, genres, plot_keywords,
      plus raw plot sources used as fallback
    - plot_synopsis: from Wave 1 plot_events output (may be None)
    - thematic_observations: from Wave 1 reception extraction zone (may be None)

When plot_synopsis is unavailable, the prompt builder automatically
selects the best available raw plot text (longest of first synopsis,
longest plot_summary, or overview) and passes it with a distinct
'plot_text' label so the LLM knows the quality tier.

When key inputs (plot_synopsis/plot_text, thematic_observations) are
absent, they are explicitly passed as "not available" in the user
message so the LLM can calibrate confidence and output empty lists
for fields it cannot confidently populate.

Skip condition (enforced by pre_consolidation):
    Tier 1: plot_synopsis (from Wave 1) → always eligible
    Tier 2: plot fallback >= 400 chars → always eligible
    Tier 3: plot fallback 250-399 chars + thematic_observations >= 300 chars → eligible
    Otherwise: skipped

Response schema: PlotAnalysisWithJustificationsOutput (justification
    fields scaffold better labels via chain-of-thought; only the labels
    are embedded).

Provider/model: OpenAI gpt-5-mini, reasoning_effort: minimal,
    verbosity: low. Finalized via evaluation pipeline.

See docs/llm_metadata_generation_new_flow.md Section 5.1.
"""

from typing import Tuple

from movie_ingestion.metadata_generation.inputs import (
    MetadataType,
    MovieInputData,
    build_user_prompt,
)
from movie_ingestion.metadata_generation.schemas import PlotAnalysisWithJustificationsOutput
from movie_ingestion.metadata_generation.prompts.plot_analysis import SYSTEM_PROMPT
from movie_ingestion.metadata_generation.errors import (
    MetadataGenerationError,
    MetadataGenerationEmptyResponseError,
)
from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from implementation.llms.vector_metadata_generation_methods import TokenUsage

GENERATION_TYPE = MetadataType.PLOT_ANALYSIS

# Finalized production config — gpt-5-mini with minimal reasoning, low
# verbosity, and justification schema. Determined via evaluation pipeline.
_PROVIDER = LLMProvider.OPENAI
_MODEL = "gpt-5-mini"
_RESPONSE_FORMAT = PlotAnalysisWithJustificationsOutput


def build_plot_analysis_user_prompt(
    movie: MovieInputData,
    plot_synopsis: str | None,
    thematic_observations: str | None,
) -> str:
    """Build the user prompt for plot_analysis generation from a movie's fields.

    Shared by the production generator and the evaluation pipeline so the
    prompt construction logic stays in one place.

    When plot_synopsis (from Wave 1 plot_events) is available, it's passed
    with the 'plot_synopsis' label. When unavailable, the best raw plot
    text is computed from the movie's synopses/summaries/overview and
    passed with the 'plot_text' label — the distinct label signals to
    the LLM that this is lower-quality fallback text.

    Inputs are labeled for the LLM to match the SYSTEM_PROMPT's INPUTS
    section. None values and empty lists are skipped by build_user_prompt.
    """
    # Determine plot input: use Wave 1 output if available, else fallback.
    # When neither is available, explicitly signal "not available" so the
    # LLM knows the data is absent (not just missing from the message).
    plot_synopsis_label = None
    plot_text_label = None
    if plot_synopsis:
        plot_synopsis_label = plot_synopsis
    else:
        fallback = movie.best_plot_fallback()
        if fallback:
            plot_text_label = fallback
        else:
            # No plot data at all — explicitly signal absence
            plot_synopsis_label = "not available"

    # Explicitly signal absent thematic_observations so the LLM
    # can calibrate confidence (it's the strongest thematic source).
    effective_thematic = thematic_observations if thematic_observations else "not available"

    return build_user_prompt(
        title=movie.title_with_year(),
        genres=movie.genres or None,
        plot_synopsis=plot_synopsis_label,
        plot_text=plot_text_label,
        merged_keywords=movie.merged_keywords() or None,
        thematic_observations=effective_thematic,
    )


async def generate_plot_analysis(
    movie: MovieInputData,
    plot_synopsis: str | None = None,
    thematic_observations: str | None = None,
) -> Tuple[PlotAnalysisWithJustificationsOutput, TokenUsage]:
    """Generate plot analysis metadata for a single movie.

    Builds the user prompt from the movie's fields plus Wave 1 outputs,
    calls OpenAI gpt-5-mini with justification schema, and returns the
    parsed result alongside token usage.

    Model configuration is finalized (gpt-5-mini, minimal reasoning,
    low verbosity, justification schema) and not caller-configurable.

    Args:
        movie: Raw movie input data loaded from the ingestion pipeline.
        plot_synopsis: Plot summary from Wave 1 plot_events. May be None
            if plot_events failed or was skipped. When None, the prompt
            builder falls back to the best available raw plot text.
        thematic_observations: Reviewer observations about themes, meaning,
            and messages from Wave 1 reception extraction zone. May be None
            if reception failed or was skipped.

    Returns:
        Tuple of (PlotAnalysisWithJustificationsOutput, TokenUsage).

    Raises:
        MetadataGenerationError: If the LLM call raises an exception.
        MetadataGenerationEmptyResponseError: If the LLM returns None.
    """
    user_prompt = build_plot_analysis_user_prompt(
        movie, plot_synopsis, thematic_observations,
    )
    title_with_year = movie.title_with_year()

    try:
        parsed, input_tokens, output_tokens = await generate_llm_response_async(
            provider=_PROVIDER,
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            response_format=_RESPONSE_FORMAT,
            model=_MODEL,
            reasoning_effort="minimal",
            verbosity="low",
        )
    except Exception as e:
        print(f"{GENERATION_TYPE} generation failed for '{title_with_year}': {e}")
        raise MetadataGenerationError(GENERATION_TYPE, title_with_year, e) from e

    if parsed is None:
        print(f"{GENERATION_TYPE} generation returned None for '{title_with_year}'")
        raise MetadataGenerationEmptyResponseError(GENERATION_TYPE, title_with_year)

    return parsed, TokenUsage(input_tokens, output_tokens, _MODEL)
