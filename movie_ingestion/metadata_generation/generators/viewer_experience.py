"""
Viewer Experience generator (Wave 2).

Async method that generates viewer experience metadata for a single movie.
Extracts what it FEELS LIKE to watch the movie -- emotional, sensory,
cognitive experience. Powers queries like "edge of your seat thriller"
or "cozy feel-good movie."

Inputs:
    - movie (MovieInputData): raw fields for title, maturity, and genres
    - generalized_plot_overview: from finalized plot_analysis metadata
      (sole narrative source; may be None)
    - emotional_observations: from finalized reception metadata (may be None)
    - craft_observations: from finalized reception metadata (may be None)
    - thematic_observations: from finalized reception metadata (may be None)
    - genre_signatures: from finalized plot_analysis metadata (may be None)

Removed inputs (validated by Round 3 evaluation):
    - plot_summary: GPO-only outperformed the full fallback chain
    - raw plot fallback (synopses/summaries/overview): same reason
    - merged_keywords: 1.8% citation rate, pure noise
    - character_arcs: 1.5% citation rate, too compressed to ground terms

Narrative source: generalized_plot_overview only (GPO). Round 3 evaluation
    showed GPO matches or slightly exceeds the full fallback chain
    (plot_summary → raw synopsis → overview) across all input quality
    buckets. GPO strips noise while preserving the thematic/emotional
    core that this generation needs.

Skip condition: enforced by pre_consolidation. Eligible when GPO >= 350
    chars standalone, observations meet standalone thresholds, or GPO
    >= 200 chars + any usable observation.

Response schema: ViewerExperienceWithJustificationsOutput (production).
    Justifications provide chain-of-thought that improves specificity
    and term diversity (+0.33 overall uplift in Round 2). Justifications
    are discarded before embedding — no retrieval impact.

Provider/model defaults: OpenAI gpt-5-mini, reasoning_effort: minimal
    (production winner from Round 2/3 evaluation).
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
from movie_ingestion.metadata_generation.schemas import (
    ViewerExperienceWithJustificationsOutput,
)
from movie_ingestion.metadata_generation.prompts.viewer_experience import (
    SYSTEM_PROMPT_WITH_JUSTIFICATIONS,
)
from movie_ingestion.metadata_generation.errors import (
    MetadataGenerationError,
    MetadataGenerationEmptyResponseError,
)
from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from implementation.llms.vector_metadata_generation_methods import TokenUsage

GENERATION_TYPE = MetadataType.VIEWER_EXPERIENCE

# Production defaults — gpt-5-mini with minimal reasoning and justification
# schema (Round 2/3 evaluation winner: 4.37 overall, zero section_discipline
# failures, uniform quality across all input profiles).
_PROVIDER = LLMProvider.OPENAI
_MODEL = "gpt-5-mini"

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
    generalized_plot_overview: str | None = None,
    emotional_observations: str | None = None,
    craft_observations: str | None = None,
    thematic_observations: str | None = None,
    genre_signatures: list[str] | None = None,
) -> str:
    """Build the user prompt for viewer_experience generation from a movie's fields.

    Shared by the production generator and the evaluation pipeline so the
    prompt construction logic stays in one place.

    Inputs are labeled for the LLM to match the SYSTEM_PROMPT's input
    interpretation section. Genre signatures win over raw genres when
    available. Narrative input is GPO-only (no fallback chain).

    Removed inputs (Round 3 evaluation, tier 1 pruning):
    - merged_keywords: 1.8% citation rate, pure noise
    - character_arcs: 1.5% citation rate, too compressed

    When key inputs are absent, they are explicitly included as
    "not available" so the LLM can calibrate confidence and produce
    empty sections rather than hallucinating from genre stereotypes.
    """
    narrative_input, narrative_input_source = resolve_viewer_experience_narrative(
        generalized_plot_overview,
    )
    filtered_emotional, filtered_craft, filtered_thematic = (
        filter_viewer_experience_observations(
            emotional_observations,
            craft_observations,
            thematic_observations,
        )
    )

    # Explicitly signal absent inputs so the LLM knows data is missing
    # (not just omitted from the message). This helps the model calibrate
    # confidence and produce empty sections rather than hallucinating.
    effective_narrative = narrative_input if narrative_input else "not available"
    effective_narrative_source = narrative_input_source if narrative_input_source else "not available"
    effective_emotional = filtered_emotional if filtered_emotional else "not available"
    effective_craft = filtered_craft if filtered_craft else "not available"
    effective_thematic = filtered_thematic if filtered_thematic else "not available"

    return build_user_prompt(
        title=movie.title_with_year(),
        genre_context=_resolve_genre_context(movie, genre_signatures),
        narrative_input_source=effective_narrative_source,
        narrative_input=effective_narrative,
        emotional_observations=effective_emotional,
        craft_observations=effective_craft,
        thematic_observations=effective_thematic,
        maturity_summary=movie.maturity_summary(),
    )


async def generate_viewer_experience(
    movie: MovieInputData,
    generalized_plot_overview: str | None = None,
    emotional_observations: str | None = None,
    craft_observations: str | None = None,
    thematic_observations: str | None = None,
    genre_signatures: list[str] | None = None,
) -> Tuple[ViewerExperienceWithJustificationsOutput, TokenUsage]:
    """Generate viewer experience metadata for a single movie.

    Builds the user prompt from the movie's fields plus upstream outputs,
    calls the specified LLM provider with structured output, and returns
    the parsed result alongside token usage.

    Locked production config: gpt-5-mini, reasoning_effort: minimal, with
    justification schema. Justifications are discarded before embedding.

    Args:
        movie: Raw movie input data loaded from the ingestion pipeline.
        generalized_plot_overview: Finalized plot_analysis generalized
            overview. Sole narrative source (GPO-only path).
        emotional_observations: Finalized reception emotional observations.
        craft_observations: Finalized reception craft observations.
        thematic_observations: Finalized reception thematic observations.
        genre_signatures: Finalized plot_analysis genre signatures.

    Returns:
        Tuple of (ViewerExperienceWithJustificationsOutput, TokenUsage).

    Raises:
        MetadataGenerationError: If the LLM call raises an exception.
        MetadataGenerationEmptyResponseError: If the LLM returns None.
    """
    user_prompt = build_viewer_experience_user_prompt(
        movie,
        generalized_plot_overview=generalized_plot_overview,
        emotional_observations=emotional_observations,
        craft_observations=craft_observations,
        thematic_observations=thematic_observations,
        genre_signatures=genre_signatures,
    )
    title_with_year = movie.title_with_year()

    try:
        parsed, input_tokens, output_tokens = await generate_llm_response_async(
            provider=_PROVIDER,
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT_WITH_JUSTIFICATIONS,
            response_format=ViewerExperienceWithJustificationsOutput,
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
