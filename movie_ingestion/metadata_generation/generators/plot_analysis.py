"""
Plot Analysis generator (Wave 2).

Async method that generates plot analysis metadata for a single movie.
Extracts WHAT TYPE OF STORY this is -- themes, lessons, core concepts,
genre signatures, character arcs, conflict type, and a generalized
plot overview.

Inputs:
    - movie (MovieInputData): raw fields for title, genres, plot_keywords,
      plus raw plot sources used as fallback
    - plot_synopsis: from Wave 1 plot_events output (may be None)
    - thematic_observations: from Wave 1 reception extraction zone (may be None)
    - emotional_observations: from Wave 1 reception extraction zone (may be None)

When plot_synopsis is unavailable, the prompt builder automatically
selects the best available raw plot text (longest of first synopsis,
longest plot_summary, or overview) and passes it with a distinct
'plot_text' label so the LLM knows the quality tier.

Skip condition: requires plot_synopsis OR at least one of
    thematic_observations / emotional_observations
    (enforced by pre_consolidation).

Response schema: PlotAnalysisOutput (no justifications) by default.
    PlotAnalysisWithJustificationsOutput available for evaluation.

Provider/model defaults: OpenAI gpt-5-mini, reasoning_effort: low
    (matching current system; will be re-evaluated).

See docs/llm_metadata_generation_new_flow.md Section 5.1.
"""

from typing import Tuple

from movie_ingestion.metadata_generation.inputs import (
    MetadataType,
    MovieInputData,
    build_user_prompt,
)
from movie_ingestion.metadata_generation.schemas import PlotAnalysisOutput
from movie_ingestion.metadata_generation.prompts.plot_analysis import SYSTEM_PROMPT
from movie_ingestion.metadata_generation.errors import (
    MetadataGenerationError,
    MetadataGenerationEmptyResponseError,
)
from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from implementation.llms.vector_metadata_generation_methods import TokenUsage

GENERATION_TYPE = MetadataType.PLOT_ANALYSIS

# Production defaults — matching current system (gpt-5-mini with low reasoning).
# Will be re-evaluated via the evaluation pipeline and updated to the winner.
_DEFAULT_PROVIDER = LLMProvider.OPENAI
_DEFAULT_MODEL = "gpt-5-mini"


def _best_plot_fallback(movie: MovieInputData) -> str | None:
    """Find the longest available plot text from raw movie sources.

    Used when Wave 1 plot_events did not produce a plot_synopsis.
    Returns the longest of:
        - First synopsis entry (plot_synopses[0])
        - Longest plot_summary entry
        - Overview text

    Returns None if no plot text is available at all.
    """
    candidates: list[str] = []
    if movie.plot_synopses:
        candidates.append(movie.plot_synopses[0])
    if movie.plot_summaries:
        candidates.append(max(movie.plot_summaries, key=len))
    if movie.overview:
        candidates.append(movie.overview)
    if not candidates:
        return None
    return max(candidates, key=len)


def build_plot_analysis_user_prompt(
    movie: MovieInputData,
    plot_synopsis: str | None,
    thematic_observations: str | None,
    emotional_observations: str | None,
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
    # Determine plot input: use Wave 1 output if available, else fallback
    plot_synopsis_label = None
    plot_text_label = None
    if plot_synopsis:
        plot_synopsis_label = plot_synopsis
    else:
        fallback = _best_plot_fallback(movie)
        if fallback:
            plot_text_label = fallback

    return build_user_prompt(
        title=movie.title_with_year(),
        genres=movie.genres or None,
        plot_synopsis=plot_synopsis_label,
        plot_text=plot_text_label,
        merged_keywords=movie.merged_keywords() or None,
        thematic_observations=thematic_observations,
        emotional_observations=emotional_observations,
    )


async def generate_plot_analysis(
    movie: MovieInputData,
    plot_synopsis: str | None = None,
    thematic_observations: str | None = None,
    emotional_observations: str | None = None,
    provider: LLMProvider = _DEFAULT_PROVIDER,
    model: str = _DEFAULT_MODEL,
    system_prompt: str = SYSTEM_PROMPT,
    response_format: type = PlotAnalysisOutput,
    **kwargs,
) -> Tuple[PlotAnalysisOutput, TokenUsage]:
    """Generate plot analysis metadata for a single movie.

    Builds the user prompt from the movie's fields plus Wave 1 outputs,
    calls the specified LLM provider with structured output, and returns
    the parsed result alongside token usage.

    Defaults to OpenAI gpt-5-mini with reasoning_effort: low (matching
    the current system). Callers can override provider/model/kwargs to
    test different configurations during evaluation.

    Args:
        movie: Raw movie input data loaded from the ingestion pipeline.
        plot_synopsis: Plot summary from Wave 1 plot_events. May be None
            if plot_events failed or was skipped. When None, the prompt
            builder falls back to the best available raw plot text.
        thematic_observations: Reviewer observations about themes, meaning,
            and messages from Wave 1 reception extraction zone. May be None
            if reception failed or was skipped.
        emotional_observations: Reviewer observations about emotional tone,
            mood, and atmosphere from Wave 1 reception extraction zone.
            May be None if reception failed or was skipped. Experimental —
            included to test impact on output quality.
        provider: Which LLM backend to use. Defaults to OPENAI.
        model: Model identifier. Defaults to "gpt-5-mini".
        system_prompt: System prompt to use. Defaults to SYSTEM_PROMPT.
        response_format: Pydantic schema for structured output. Defaults
            to PlotAnalysisOutput.
        **kwargs: Provider-specific params (e.g. reasoning_effort, temperature).

    Returns:
        Tuple of (PlotAnalysisOutput, TokenUsage).

    Raises:
        MetadataGenerationError: If the LLM call raises an exception.
        MetadataGenerationEmptyResponseError: If the LLM returns None.
    """
    user_prompt = build_plot_analysis_user_prompt(
        movie, plot_synopsis, thematic_observations, emotional_observations,
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
