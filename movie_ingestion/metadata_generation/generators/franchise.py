"""
Franchise generator.

Identifies whether a movie belongs to a recognizable IP or brand, classifies
its role in that franchise, and optionally captures an established subgroup
name when one exists.
"""

from __future__ import annotations

from typing import Tuple

from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from implementation.llms.vector_metadata_generation_methods import TokenUsage
from movie_ingestion.metadata_generation.errors import (
    MetadataGenerationEmptyResponseError,
    MetadataGenerationError,
)
from movie_ingestion.metadata_generation.inputs import build_user_prompt
from movie_ingestion.metadata_generation.prompts.franchise import SYSTEM_PROMPT
from schemas.enums import MetadataType
from schemas.metadata import FranchiseOutput
from schemas.movie_input import MovieInputData

GENERATION_TYPE = MetadataType.FRANCHISE

_DEFAULT_PROVIDER = LLMProvider.OPENAI
_DEFAULT_MODEL = "gpt-5-mini"


def build_franchise_user_prompt(movie: MovieInputData) -> str:
    """Build the user prompt for franchise generation.

    Inputs are chosen to give the model the strongest possible signals
    for franchise identification, chronology (is_prequel), and director-
    era subgroup detection (culturally_recognized_groups):
    - release_year is labeled separately so the model can anchor
      "prior-released at this film's release" reasoning.
    - directors (explicitly labeled) support director-era subgroup
      detection ("raimi trilogy", "bay era", "daniel craig era"
      indirectly via the main cast, etc.).
    - top_billed_cast pairs actors with characters, which strengthens
      actor-run subgroup detection (e.g. "daniel craig era").
    """
    return build_user_prompt(
        title_with_year=movie.title_with_year(),
        release_year=str(movie.release_year) if movie.release_year is not None else "not available",
        overview=movie.overview or "not available",
        collection_name=movie.collection_name or "not available",
        production_companies=movie.production_companies or "not available",
        directors=movie.directors or "not available",
        overall_keywords=movie.overall_keywords or "not available",
        characters=movie.characters[:5] or "not available",
        top_billed_cast=movie.top_billed_cast(5) or "not available",
    )


async def generate_franchise(
    movie: MovieInputData,
    provider: LLMProvider | None = None,
    model: str | None = None,
    **kwargs,
) -> Tuple[FranchiseOutput, TokenUsage]:
    """Generate franchise metadata for a single movie.

    Args:
        movie: Raw movie input data from the ingestion pipeline.
        provider: LLM provider override. Defaults to _DEFAULT_PROVIDER.
        model: Model name override. Defaults to _DEFAULT_MODEL.
        **kwargs: Provider-specific kwargs passed through to the LLM call
            (e.g. reasoning_effort, verbosity for OpenAI; thinking_config
            for Gemini). When not provided, defaults to reasoning_effort
            "low" and verbosity "low" for OpenAI.

    Returns:
        Tuple of (FranchiseOutput, TokenUsage).

    Raises:
        MetadataGenerationError: If the LLM call raises an exception.
        MetadataGenerationEmptyResponseError: If the LLM returns None.
    """
    effective_provider = provider or _DEFAULT_PROVIDER
    effective_model = model or _DEFAULT_MODEL

    # Apply default OpenAI kwargs when no overrides are provided and the
    # provider is OpenAI (or defaulting to OpenAI).
    if not kwargs and effective_provider == LLMProvider.OPENAI:
        kwargs = {"reasoning_effort": "low", "verbosity": "low"}

    user_prompt = build_franchise_user_prompt(movie)
    title_with_year = movie.title_with_year()

    try:
        parsed, input_tokens, output_tokens = await generate_llm_response_async(
            provider=effective_provider,
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            response_format=FranchiseOutput,
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
