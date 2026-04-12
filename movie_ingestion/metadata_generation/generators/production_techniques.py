"""
Production Techniques generator.

Async method that generates production_techniques metadata for a single movie.
Classification task: filters plot_keywords and overall_keywords to keep only
production-technique terms. The LLM classifies existing keywords only.

Inputs (from MovieInputData):
    - title_with_year(): "Title (Year)" format
    - overall_keywords: curated keyword taxonomy
    - plot_keywords: free-form community keywords

Skip condition: plot_keywords > 0 OR overall_keywords >= 3
    (enforced by pre_consolidation).

Response schema: ProductionTechniquesOutput.

Default provider/model: OpenAI gpt-5-mini with reasoning_effort: low.
The defaults are used in production, but provider/model/kwargs may be
overridden by evaluation tooling such as notebooks.
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
_DEFAULT_MODEL = "gpt-5-mini"


def build_production_techniques_user_prompt(movie: MovieInputData) -> str:
    """Build the user prompt for production_techniques generation."""
    return build_user_prompt(
        title=movie.title_with_year(),
        overall_keywords=movie.overall_keywords or None,
        plot_keywords=movie.plot_keywords or None,
    )


async def generate_production_techniques(
    movie: MovieInputData,
    provider: LLMProvider | None = None,
    model: str | None = None,
    **kwargs,
) -> Tuple[ProductionTechniquesOutput, TokenUsage]:
    """Generate production_techniques metadata for a single movie.

    Args:
        movie: Raw movie input data from the ingestion pipeline.
        provider: LLM provider override. Defaults to _DEFAULT_PROVIDER.
        model: Model name override. Defaults to _DEFAULT_MODEL.
        **kwargs: Provider-specific kwargs passed through to the LLM call.
            When not provided and the provider is OpenAI, defaults to
            reasoning_effort "low" and verbosity "low".

    Returns:
        Tuple of (ProductionTechniquesOutput, TokenUsage).

    Raises:
        MetadataGenerationError: If the LLM call raises an exception.
        MetadataGenerationEmptyResponseError: If the LLM returns None.
    """
    user_prompt = build_production_techniques_user_prompt(movie)
    title_with_year = movie.title_with_year()
    effective_provider = provider or _DEFAULT_PROVIDER
    effective_model = model or _DEFAULT_MODEL

    if not kwargs and effective_provider == LLMProvider.OPENAI:
        kwargs = {"reasoning_effort": "low", "verbosity": "low"}

    try:
        parsed, input_tokens, output_tokens = await generate_llm_response_async(
            provider=effective_provider,
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            response_format=ProductionTechniquesOutput,
            model=effective_model,
            **kwargs,
        )
    except Exception as e:
        print(f"{GENERATION_TYPE} generation failed for '{title_with_year}': {e}")
        raise MetadataGenerationError(GENERATION_TYPE, title_with_year, e) from e

    if parsed is None:
        print(f"{GENERATION_TYPE} generation returned None for '{title_with_year}'")
        raise MetadataGenerationEmptyResponseError(GENERATION_TYPE, title_with_year)

    return parsed, TokenUsage(input_tokens, output_tokens, effective_model)
