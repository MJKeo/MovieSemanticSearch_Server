"""
Source of Inspiration generator (Wave 2).

Async method that generates source of inspiration metadata for a single
movie. Determines source material (novel, true story, etc.) and production
medium (live-action, animation, etc.).

This is the ONLY generation that allows parametric knowledge -- the LLM
may use its training data for source material facts. This is safe because
source material facts are categorical and verifiable, and this is a
leaf-node classification that doesn't cascade to other generations.

Inputs:
    - movie (MovieInputData): raw fields for title, merged_keywords
    - source_material_hint: from Wave 1 reception extraction zone
      (may be None). Short classifying phrase like "based on autobiography",
      "remake", "based on book, sequel". Highest-confidence grounding
      signal when present — reviewer-extracted evidence.

Removed inputs (vs current system):
    - plot_synopsis: removed per ADR-033 (barely used, saves ~83.6M tokens)
    - featured_reviews: replaced first by review_insights_brief, then by
      the more targeted source_material_hint
    - review_insights_brief: replaced by source_material_hint (the source
      signal was in this dedicated field, not in the observation blobs)
    - plot_keywords / overall_keywords as separate inputs: merged via
      movie.merged_keywords()

Skip condition: eligible when merged_keywords >= 1 OR source_material_hint
    is present. Near-zero skip rate (~21 movies lack all keywords).

Response schema: SourceOfInspirationOutput (no justifications) by default.
    SourceOfInspirationWithJustificationsOutput available for evaluation.

Provider/model defaults: OpenAI gpt-5-mini, reasoning_effort: low,
    verbosity: low.

See docs/llm_metadata_generation_new_flow.md Section 5.5.
"""

from typing import Tuple

from movie_ingestion.metadata_generation.inputs import (
    MetadataType,
    MovieInputData,
    build_user_prompt,
)
from movie_ingestion.metadata_generation.schemas import SourceOfInspirationOutput
from movie_ingestion.metadata_generation.prompts.source_of_inspiration import SYSTEM_PROMPT
from movie_ingestion.metadata_generation.errors import (
    MetadataGenerationError,
    MetadataGenerationEmptyResponseError,
)
from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from implementation.llms.vector_metadata_generation_methods import TokenUsage

GENERATION_TYPE = MetadataType.SOURCE_OF_INSPIRATION

# Production defaults — matching legacy system (gpt-5-mini with low reasoning).
# Will be re-evaluated via the evaluation pipeline and updated to the winner.
_DEFAULT_PROVIDER = LLMProvider.OPENAI
_DEFAULT_MODEL = "gpt-5-mini"


def build_source_of_inspiration_user_prompt(
    movie: MovieInputData,
    source_material_hint: str | None,
) -> str:
    """Build the user prompt for source_of_inspiration generation.

    Shared by the production generator and the evaluation pipeline so the
    prompt construction logic stays in one place.

    Inputs are labeled for the LLM to match the SYSTEM_PROMPT's INPUTS
    section. None values and empty lists are skipped by build_user_prompt.

    source_material_hint is the highest-confidence grounding signal when
    present — a short reviewer-extracted classification phrase from the
    Wave 1 reception generator (e.g., "based on autobiography", "remake").

    plot_synopsis was removed per ADR-033 — this generator barely uses
    plot data, and removing it saves ~83.6M input tokens across the corpus.
    """
    return build_user_prompt(
        title=movie.title_with_year(),
        merged_keywords=movie.merged_keywords() or None,
        source_material_hint=source_material_hint,
    )


async def generate_source_of_inspiration(
    movie: MovieInputData,
    source_material_hint: str | None = None,
    provider: LLMProvider = _DEFAULT_PROVIDER,
    model: str = _DEFAULT_MODEL,
    system_prompt: str = SYSTEM_PROMPT,
    response_format: type = SourceOfInspirationOutput,
    **kwargs,
) -> Tuple[SourceOfInspirationOutput, TokenUsage]:
    """Generate source of inspiration metadata for a single movie.

    Builds the user prompt from the movie's fields plus Wave 1 outputs,
    calls the specified LLM provider with structured output, and returns
    the parsed result alongside token usage.

    This is the ONLY generation where parametric knowledge is allowed --
    the LLM may contribute source material facts from its training data
    for well-known films.

    Defaults to OpenAI gpt-5-mini with reasoning_effort: low (matching
    the legacy system). Callers can override provider/model/kwargs to
    test different configurations during evaluation.

    Args:
        movie: Raw movie input data loaded from the ingestion pipeline.
        source_material_hint: Short classifying phrase from Wave 1
            reception extraction zone (e.g., "based on autobiography",
            "remake"). May be None if reception failed, was skipped, or
            reviewers didn't mention source material.
        provider: Which LLM backend to use. Defaults to OPENAI.
        model: Model identifier. Defaults to "gpt-5-mini".
        **kwargs: Provider-specific params (e.g. reasoning_effort, temperature).

    Returns:
        Tuple of (SourceOfInspirationOutput, TokenUsage).

    Raises:
        MetadataGenerationError: If the LLM call raises an exception.
        MetadataGenerationEmptyResponseError: If the LLM returns None.
    """
    user_prompt = build_source_of_inspiration_user_prompt(
        movie, source_material_hint,
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
