"""
Reception generator (Wave 1).

Async method that generates reception metadata for a single movie.
Accepts MovieInputData, builds a user prompt, calls the specified LLM
provider via the unified routing method, and returns the parsed
ReceptionOutput with token usage.

Dual-zone output:

Extraction zone (NOT embedded, consumed by Wave 2 generators):
    - source_material_hint: classifying phrase for adaptations/remakes
    - thematic_observations: themes, meaning, messages
    - emotional_observations: emotional tone, mood, atmosphere
    - craft_observations: narrative structure, pacing, craft

Synthesis zone (embedded into reception_vectors):
    - reception_summary: evaluative summary of audience opinion
    - praised_qualities: 0-6 tag phrases for what audiences liked
    - criticized_qualities: 0-6 tag phrases for what audiences disliked

Inputs (from MovieInputData):
    - title_with_year(): "Title (Year)" format
    - genres: genre labels for contextual grounding
    - reception_summary: externally generated audience opinion summary
    - audience_reception_attributes: key attributes with sentiment labels
    - featured_reviews: raw review texts truncated to 6K combined chars
      (this is the ONLY call that receives raw reviews — Wave 2 calls
      get the brief instead)

Response schema: ReceptionOutput
Provider/model: OpenAI gpt-5-mini, reasoning_effort: minimal, verbosity: low.
Provider/model are fixed — evaluated across 36 movies in 6 input-richness
buckets against multiple candidates and reasoning efforts. Minimal reasoning
with the revised prompt matched or exceeded low-reasoning quality on the
old prompt while halving output token cost.

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

# Fixed LLM configuration — see module docstring for rationale.
_PROVIDER = LLMProvider.OPENAI
_MODEL = "gpt-5-mini"
_MODEL_KWARGS = {"reasoning_effort": "minimal", "verbosity": "low"}

# Review truncation limit — balance token cost vs signal quality.
# No count cap: the character budget naturally limits bloat while
# allowing more diverse perspectives when individual reviews are short.
# 6K chars keeps ~2x headroom above the observed output saturation
# point (~3K chars of review content) while trimming the long tail.
_MAX_REVIEW_CHARS = 6000


def _truncate_reviews(
    reviews: list[dict],
    max_chars: int = _MAX_REVIEW_CHARS,
) -> list[dict]:
    """Select reviews up to the character budget, prioritising diversity.

    Reviews are sorted shortest-first so that many short perspectives
    are included before a single long review can consume the budget.
    Adds reviews one at a time, stopping after the review that causes
    the character threshold to be crossed (that review IS included).

    Edge case: if every review individually exceeds the budget, the
    shortest review is truncated to fit — some signal is always better
    than none.
    """
    if not reviews:
        return []

    # Sort ascending by text length so shorter (more diverse) reviews
    # are packed in first.
    sorted_reviews = sorted(reviews, key=lambda r: len(r.get("text", "")))

    # Edge case: even the shortest review exceeds the budget.
    # Truncate its text to fit rather than returning nothing.
    shortest_text = sorted_reviews[0].get("text", "")
    if len(shortest_text) >= max_chars:
        truncated = {**sorted_reviews[0], "text": shortest_text[:max_chars]}
        return [truncated]

    kept: list[dict] = []
    accumulated_chars = 0
    for review in sorted_reviews:
        kept.append(review)
        accumulated_chars += len(review.get("text", ""))
        if accumulated_chars >= max_chars:
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

    Includes genres for contextual grounding (placed before
    reception-specific fields to prime the model with movie context).
    Reviews are sorted shortest-first for perspective diversity, then
    truncated to 6K combined chars of review text. Each review is
    formatted as "summary: text" and passed as a MultiLineList.
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
        genres=movie.genres or None,
        reception_summary=movie.reception_summary or None,
        audience_reception_attributes=formatted_attributes,
        featured_reviews=formatted_reviews,
    )


async def generate_reception(
    movie: MovieInputData,
) -> Tuple[ReceptionOutput, TokenUsage]:
    """Generate reception metadata for a single movie.

    Builds the user prompt from the movie's reception-related fields,
    calls gpt-5-mini via OpenAI with structured output, and returns
    the parsed result alongside token usage.

    Provider/model are fixed — see module docstring for rationale.

    Args:
        movie: Raw movie input data loaded from the ingestion pipeline.

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
            provider=_PROVIDER,
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            response_format=ReceptionOutput,
            model=_MODEL,
            **_MODEL_KWARGS,
        )
    except Exception as e:
        print(f"{GENERATION_TYPE} generation failed for '{title_with_year}': {e}")
        raise MetadataGenerationError(GENERATION_TYPE, title_with_year, e) from e

    if parsed is None:
        print(f"{GENERATION_TYPE} generation returned None for '{title_with_year}'")
        raise MetadataGenerationEmptyResponseError(GENERATION_TYPE, title_with_year)

    return parsed, TokenUsage(input_tokens, output_tokens, _MODEL)
