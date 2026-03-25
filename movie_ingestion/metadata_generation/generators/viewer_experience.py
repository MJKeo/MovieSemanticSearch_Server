"""
Viewer Experience generator (Wave 2).

Async method that generates viewer experience metadata for a single movie.
Extracts what it FEELS LIKE to watch the movie -- emotional, sensory,
cognitive experience. Powers queries like "edge of your seat thriller"
or "cozy feel-good movie."

Inputs:
    - movie (MovieInputData): raw fields for title, keywords, maturity, and
      raw plot fallback text
    - plot_synopsis: from finalized plot_events metadata (may be None)
    - generalized_plot_overview: from finalized plot_analysis metadata
      (backup only; may be None)
    - emotional_observations: from finalized reception metadata (may be None)
    - craft_observations: from finalized reception metadata (may be None)
    - thematic_observations: from finalized reception metadata (may be None)
    - genre_signatures: from finalized plot_analysis metadata (may be None)
    - character_arcs: finalized arc labels from plot_analysis (may be None)

Removed inputs (vs current system):
    - overlapping raw plot/review inputs once finalized upstream metadata
      exists for the same signal
    - raw genres when genre_signatures are available

Narrative fallback order (shared with pre_consolidation eligibility):
    1. plot_synopsis if >= 400 chars
    2. movie.best_plot_fallback() if >= 500 chars
    3. generalized_plot_overview if >= 200 chars

Skip condition: enforced by pre_consolidation using the finalized
    viewer_experience narrative/observation thresholds. Combined path
    threshold varies by narrative source quality.

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
from movie_ingestion.metadata_generation.pre_consolidation import (
    resolve_viewer_experience_narrative,
    filter_viewer_experience_observations,
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

def _resolve_genre_context(
    movie: MovieInputData,
    genre_signatures: list[str] | None,
) -> list[str] | None:
    """Use plot_analysis genre signatures when available, else raw genres."""
    if genre_signatures and len(genre_signatures) >= 2:
        return genre_signatures
    return movie.genres or None


def build_viewer_experience_user_prompt(
    movie: MovieInputData,
    plot_synopsis: str | None,
    generalized_plot_overview: str | None = None,
    emotional_observations: str | None = None,
    craft_observations: str | None = None,
    thematic_observations: str | None = None,
    genre_signatures: list[str] | None = None,
    character_arcs: list[str] | None = None,
) -> str:
    """Build the user prompt for viewer_experience generation from a movie's fields.

    Shared by the production generator and the evaluation pipeline so the
    prompt construction logic stays in one place.

    Inputs are labeled for the LLM to match the SYSTEM_PROMPT's INPUTS
    section. None values and empty lists are skipped by build_user_prompt.
    Only one narrative input is passed, selected by the finalized
    fallback order. Genre signatures win over raw genres when available.
    """
    narrative_input, narrative_input_source = resolve_viewer_experience_narrative(
        movie,
        plot_synopsis,
        generalized_plot_overview,
    )
    filtered_emotional, filtered_craft, filtered_thematic = (
        filter_viewer_experience_observations(
            emotional_observations,
            craft_observations,
            thematic_observations,
        )
    )

    return build_user_prompt(
        title=movie.title_with_year(),
        genre_context=_resolve_genre_context(movie, genre_signatures),
        narrative_input_source=narrative_input_source,
        narrative_input=narrative_input,
        emotional_observations=filtered_emotional,
        craft_observations=filtered_craft,
        thematic_observations=filtered_thematic,
        character_arcs=character_arcs or None,
        merged_keywords=movie.merged_keywords() or None,
        maturity_summary=movie.maturity_summary(),
    )


async def generate_viewer_experience(
    movie: MovieInputData,
    plot_synopsis: str | None = None,
    generalized_plot_overview: str | None = None,
    emotional_observations: str | None = None,
    craft_observations: str | None = None,
    thematic_observations: str | None = None,
    genre_signatures: list[str] | None = None,
    character_arcs: list[str] | None = None,
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
        plot_synopsis: Finalized plot_events plot summary. Primary
            narrative source when available.
        generalized_plot_overview: Finalized plot_analysis generalized
            overview. Only used when stronger narrative inputs are absent.
        emotional_observations: Finalized reception emotional observations.
        craft_observations: Finalized reception craft observations.
        thematic_observations: Finalized reception thematic observations.
        genre_signatures: Finalized plot_analysis genre signatures.
        character_arcs: Finalized plot_analysis character arc labels.
        provider: Which LLM backend to use. Defaults to OPENAI.
        model: Model identifier. Defaults to "gpt-5-mini".
        **kwargs: Provider-specific params (e.g. reasoning_effort, temperature).

    Returns:
        Tuple of (ViewerExperienceOutput, TokenUsage).

    Raises:
        MetadataGenerationError: If the LLM call raises an exception.
        MetadataGenerationEmptyResponseError: If the LLM returns None.
    """
    user_prompt = build_viewer_experience_user_prompt(
        movie,
        plot_synopsis,
        generalized_plot_overview=generalized_plot_overview,
        emotional_observations=emotional_observations,
        craft_observations=craft_observations,
        thematic_observations=thematic_observations,
        genre_signatures=genre_signatures,
        character_arcs=character_arcs,
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
