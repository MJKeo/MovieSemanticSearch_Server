"""
Production Techniques generator.

Async method that generates production_techniques metadata for a single movie.
Classification task: filters plot_keywords and overall_keywords to keep only
production-technique terms. This includes concrete making/rendering/capture
methods such as animation modalities, animation sub-techniques, visual
capture techniques, and the explicit found-footage carveout. The LLM
classifies existing keywords only.

Inputs (from MovieInputData):
    - title_with_year(): "Title (Year)" format
    - overall_keywords: curated keyword taxonomy
    - plot_keywords: free-form community keywords

Skip condition: plot_keywords > 0 OR overall_keywords >= 3
    (enforced by pre_consolidation).

Response schema: ProductionTechniquesOutput.

Default provider/model: OpenAI gpt-5.4-mini with reasoning_effort: low
and verbosity: low.
"""

from typing import Tuple

from schemas.enums import MetadataType
from schemas.movie_input import MovieInputData
from schemas.metadata import ProductionTechniquesOutput
from movie_ingestion.metadata_generation.inputs import build_user_prompt
from movie_ingestion.metadata_generation.prompts.production_techniques import (
    SYSTEM_PROMPT,
)
from movie_ingestion.metadata_generation.errors import (
    MetadataGenerationError,
    MetadataGenerationEmptyResponseError,
)
from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from implementation.llms.vector_metadata_generation_methods import TokenUsage

GENERATION_TYPE = MetadataType.PRODUCTION_TECHNIQUES

_DEFAULT_PROVIDER = LLMProvider.OPENAI
_DEFAULT_MODEL = "gpt-5.4-mini"
_DEFAULT_KWARGS = {"reasoning_effort": "low", "verbosity": "low"}


def build_production_techniques_user_prompt(movie: MovieInputData) -> str:
    """Build the user prompt for production_techniques generation."""
    return build_user_prompt(
        title=movie.title_with_year(),
        overall_keywords=movie.overall_keywords or None,
        plot_keywords=movie.plot_keywords or None,
    )


async def generate_production_techniques(
    movie: MovieInputData,
) -> Tuple[ProductionTechniquesOutput, TokenUsage]:
    """Generate production_techniques metadata for a single movie.

    Uses the hardcoded default model (gpt-5.4-mini with low reasoning
    effort and low verbosity).

    Args:
        movie: Raw movie input data from the ingestion pipeline.

    Returns:
        Tuple of (ProductionTechniquesOutput, TokenUsage).

    Raises:
        MetadataGenerationError: If the LLM call raises an exception.
        MetadataGenerationEmptyResponseError: If the LLM returns None.
    """
    user_prompt = build_production_techniques_user_prompt(movie)
    title_with_year = movie.title_with_year()

    try:
        parsed, input_tokens, output_tokens = await generate_llm_response_async(
            provider=_DEFAULT_PROVIDER,
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            response_format=ProductionTechniquesOutput,
            model=_DEFAULT_MODEL,
            **_DEFAULT_KWARGS,
        )
    except Exception as e:
        print(f"{GENERATION_TYPE} generation failed for '{title_with_year}': {e}")
        raise MetadataGenerationError(GENERATION_TYPE, title_with_year, e) from e

    if parsed is None:
        print(f"{GENERATION_TYPE} generation returned None for '{title_with_year}'")
        raise MetadataGenerationEmptyResponseError(GENERATION_TYPE, title_with_year)

    return parsed, TokenUsage(input_tokens, output_tokens, _DEFAULT_MODEL)
