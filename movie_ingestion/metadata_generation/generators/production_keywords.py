"""
Production Keywords generator (Wave 2).

Async method that generates production keywords metadata for a single movie.
Classification task: filters the merged keyword list to keep only
production-relevant keywords. The LLM classifies (not generates).

Inputs (from MovieInputData):
    - title_with_year(): "Title (Year)" format
    - merged_keywords(): deduplicated union of plot + overall keywords

Skip condition: requires merged_keywords >= 1 entry
    (enforced by pre_consolidation).

Response schema: ProductionKeywordsOutput (no justifications).

Provider/model: OpenAI gpt-5-mini with reasoning_effort: low.
    Selected as the production candidate based on evaluation results
    (r2-5-mini-low): perfect precision (5.00), near-perfect recall
    (4.92), zero failures across 48 test movies.

See docs/llm_metadata_generation_new_flow.md Section 5.5.
"""

from typing import Tuple

from schemas.enums import MetadataType
from schemas.movie_input import MovieInputData
from schemas.metadata import ProductionKeywordsOutput
from movie_ingestion.metadata_generation.inputs import build_user_prompt
from movie_ingestion.metadata_generation.prompts.production_keywords import SYSTEM_PROMPT
from movie_ingestion.metadata_generation.errors import (
    MetadataGenerationError,
    MetadataGenerationEmptyResponseError,
)
from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from implementation.llms.vector_metadata_generation_methods import TokenUsage

GENERATION_TYPE = MetadataType.PRODUCTION_KEYWORDS

# Finalized production config — r2-5-mini-low evaluation winner.
_PROVIDER = LLMProvider.OPENAI
_MODEL = "gpt-5-mini"
_KWARGS = {"reasoning_effort": "low"}


def build_production_keywords_user_prompt(movie: MovieInputData) -> str:
    """Build the user prompt for production_keywords generation from a movie's fields.

    Shared by the production generator and the evaluation pipeline so the
    prompt construction logic stays in one place.

    This is the simplest generator — only title and merged_keywords are
    needed. The LLM's job is to filter the keyword list, not generate
    new content.
    """
    return build_user_prompt(
        title=movie.title_with_year(),
        merged_keywords=movie.merged_keywords() or None,
    )


async def generate_production_keywords(
    movie: MovieInputData,
) -> Tuple[ProductionKeywordsOutput, TokenUsage]:
    """Generate production keywords metadata for a single movie.

    Builds the user prompt from the movie's title and merged keywords,
    calls OpenAI gpt-5-mini with structured output, and returns the
    parsed result alongside token usage.

    This is a classification task — the LLM filters keywords from the
    input list, it does not generate new ones. No Wave 1 outputs needed.

    Uses the finalized r2-5-mini-low configuration: OpenAI gpt-5-mini,
    ProductionKeywordsOutput schema (no justifications), reasoning_effort
    low. Selected based on evaluation results across 48 movies: perfect
    precision, near-perfect recall, zero hard failures.

    Args:
        movie: Raw movie input data loaded from the ingestion pipeline.

    Returns:
        Tuple of (ProductionKeywordsOutput, TokenUsage).

    Raises:
        MetadataGenerationError: If the LLM call raises an exception.
        MetadataGenerationEmptyResponseError: If the LLM returns None.
    """
    user_prompt = build_production_keywords_user_prompt(movie)
    title_with_year = movie.title_with_year()

    try:
        parsed, input_tokens, output_tokens = await generate_llm_response_async(
            provider=_PROVIDER,
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            response_format=ProductionKeywordsOutput,
            model=_MODEL,
            **_KWARGS,
        )
    except Exception as e:
        print(f"{GENERATION_TYPE} generation failed for '{title_with_year}': {e}")
        raise MetadataGenerationError(GENERATION_TYPE, title_with_year, e) from e

    if parsed is None:
        print(f"{GENERATION_TYPE} generation returned None for '{title_with_year}'")
        raise MetadataGenerationEmptyResponseError(GENERATION_TYPE, title_with_year)

    return parsed, TokenUsage(input_tokens, output_tokens, _MODEL)
