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

Response schema: ProductionKeywordsOutput (no justifications) by default.
    ProductionKeywordsWithJustificationsOutput available for evaluation.

Provider/model defaults: OpenAI gpt-5-mini, reasoning_effort: low,
    verbosity: low.

See docs/llm_metadata_generation_new_flow.md Section 5.5.
"""

from typing import Tuple

from movie_ingestion.metadata_generation.inputs import (
    MovieInputData,
    build_user_prompt,
)
from movie_ingestion.metadata_generation.schemas import ProductionKeywordsOutput
from movie_ingestion.metadata_generation.prompts.production_keywords import SYSTEM_PROMPT
from movie_ingestion.metadata_generation.errors import (
    MetadataGenerationError,
    MetadataGenerationEmptyResponseError,
)
from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from implementation.llms.vector_metadata_generation_methods import TokenUsage

GENERATION_TYPE = "production_keywords"

# Production defaults — matching legacy system (gpt-5-mini with low reasoning).
# Will be re-evaluated via the evaluation pipeline and updated to the winner.
_DEFAULT_PROVIDER = LLMProvider.OPENAI
_DEFAULT_MODEL = "gpt-5-mini"


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
    provider: LLMProvider = _DEFAULT_PROVIDER,
    model: str = _DEFAULT_MODEL,
    system_prompt: str = SYSTEM_PROMPT,
    response_format: type = ProductionKeywordsOutput,
    **kwargs,
) -> Tuple[ProductionKeywordsOutput, TokenUsage]:
    """Generate production keywords metadata for a single movie.

    Builds the user prompt from the movie's title and merged keywords,
    calls the specified LLM provider with structured output, and returns
    the parsed result alongside token usage.

    This is a classification task — the LLM filters keywords from the
    input list, it does not generate new ones. No Wave 1 outputs needed.

    Defaults to OpenAI gpt-5-mini with reasoning_effort: low (matching
    the legacy system). Callers can override provider/model/kwargs to
    test different configurations during evaluation.

    Args:
        movie: Raw movie input data loaded from the ingestion pipeline.
        provider: Which LLM backend to use. Defaults to OPENAI.
        model: Model identifier. Defaults to "gpt-5-mini".
        **kwargs: Provider-specific params (e.g. reasoning_effort, temperature).

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
