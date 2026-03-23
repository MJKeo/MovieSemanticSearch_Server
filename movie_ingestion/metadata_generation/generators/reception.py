"""
Reception generator (Wave 1).

Async method that generates reception metadata for a single movie.
Accepts MovieInputData, builds a user prompt, calls the specified LLM
provider via the unified routing method, and returns the parsed
ReceptionOutput with token usage.

Dual purpose: produces both evaluative reception metadata (summary,
praise/complaint attributes) AND a review_insights_brief intermediate
output consumed by all Wave 2 generators.

Inputs (from MovieInputData):
    - title_with_year(): "Title (Year)" format
    - reception_summary: externally generated audience opinion summary
    - audience_reception_attributes: key attributes with sentiment labels
    - featured_reviews: up to 5 full review texts (this is the ONLY call
      that receives raw reviews — Wave 2 calls get the brief instead)

Response schema: ReceptionOutput
Provider/model defaults: OpenAI gpt-5-mini, reasoning_effort: low.

See docs/llm_metadata_generation_new_flow.md Section 4.2.
"""

import re
from typing import Tuple

from movie_ingestion.metadata_generation.inputs import (
    MetadataType,
    MovieInputData,
    MultiLineList,
    build_user_prompt,
)
from movie_ingestion.metadata_generation.schemas import ReceptionOutput
from movie_ingestion.metadata_generation.prompts.reception import SYSTEM_PROMPT
from movie_ingestion.metadata_generation.errors import (
    MetadataGenerationError,
    MetadataGenerationEmptyResponseError,
)
from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from implementation.llms.vector_metadata_generation_methods import TokenUsage

GENERATION_TYPE = MetadataType.RECEPTION

# Production defaults — gpt-5-mini with low reasoning effort
_DEFAULT_PROVIDER = LLMProvider.OPENAI
_DEFAULT_MODEL = "gpt-5-mini"

# Review truncation limits — balance token cost vs signal quality
_MAX_REVIEW_COUNT = 5
_MAX_REVIEW_CHARS = 5000


def _truncate_reviews(
    reviews: list[dict],
    max_count: int = _MAX_REVIEW_COUNT,
    max_chars: int = _MAX_REVIEW_CHARS,
) -> list[dict]:
    """Select reviews up to count and character limits.

    Adds reviews one at a time. Stops after the review that causes
    either limit to be hit — so the review that crosses the char
    threshold IS included, but no further reviews are added.
    """
    kept: list[dict] = []
    accumulated_chars = 0
    for review in reviews:
        kept.append(review)
        accumulated_chars += len(review.get("text", ""))
        if len(kept) >= max_count or accumulated_chars >= max_chars:
            break
    return kept


def _format_attributes(attributes: list[dict]) -> str:
    """Format audience_reception_attributes as 'name (sentiment)' pairs."""
    return ", ".join(
        f"{attr.get('name', '')} ({attr.get('sentiment', '')})"
        for attr in attributes
    )


def build_reception_user_prompt(movie: MovieInputData) -> str:
    """Build the user prompt for reception generation from a movie's fields.

    Shared by the production generator and the evaluation pipeline so the
    prompt construction logic stays in one place.

    Reviews are truncated to at most 5 entries or 5000 combined chars of
    review text, whichever limit is hit first. Each review is formatted
    as "summary: text" and passed as a MultiLineList.
    """
    # Format audience_reception_attributes as "name (sentiment)" pairs
    formatted_attributes = (
        _format_attributes(movie.audience_reception_attributes)
        if movie.audience_reception_attributes
        else None
    )

    # Truncate reviews, collapse embedded newlines that waste tokens and
    # can confuse the LLM, then format each as "summary: text".
    truncated = _truncate_reviews(movie.featured_reviews)
    formatted_reviews = (
        MultiLineList([
            f"{r.get('summary', '')}: {re.sub(r'\n+', ' ', r.get('text', ''))}"
            for r in truncated
        ])
        if truncated
        else None
    )

    return build_user_prompt(
        title=movie.title_with_year(),
        reception_summary=movie.reception_summary or None,
        audience_reception_attributes=formatted_attributes,
        featured_reviews=formatted_reviews,
    )


async def generate_reception(
    movie: MovieInputData,
    provider: LLMProvider = _DEFAULT_PROVIDER,
    model: str = _DEFAULT_MODEL,
    **kwargs,
) -> Tuple[ReceptionOutput, TokenUsage]:
    """Generate reception metadata for a single movie.

    Builds the user prompt from the movie's reception-related fields,
    calls the specified LLM provider with structured output, and returns
    the parsed result alongside token usage.

    Defaults to OpenAI gpt-5-mini with low reasoning effort. Callers
    can override provider/model/kwargs to use a different configuration
    (e.g., for evaluation).

    Args:
        movie: Raw movie input data loaded from the ingestion pipeline.
        provider: Which LLM backend to use. Defaults to OPENAI.
        model: Model identifier. Defaults to "gpt-5-mini".
        **kwargs: Provider-specific params (e.g. reasoning_effort, temperature,
            thinking_config). Passed directly to generate_llm_response_async.

    Returns:
        Tuple of (ReceptionOutput, TokenUsage).

    Raises:
        MetadataGenerationError: If the LLM call raises an exception.
        MetadataGenerationEmptyResponseError: If the LLM returns None.
    """
    user_prompt = build_reception_user_prompt(movie)
    title_with_year = movie.title_with_year()

    try:
        parsed, input_tokens, output_tokens = await generate_llm_response_async(
            provider=provider,
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            response_format=ReceptionOutput,
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
