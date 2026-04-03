"""
Source of Inspiration generator.

Two classification decisions per movie:
1. source_material — what existing media does the film draw from?
   (adaptations, remakes, reboots, reimaginings, spinoffs)
2. franchise_lineage — where does the film sit in a franchise timeline?
   (sequel, prequel, trilogy position, franchise starter)

Parametric knowledge allowed at 95%+ confidence. Leaf-node classification.

Inputs: title (with year), merged_keywords, source_material_hint (from
Wave 1 reception). The hint is the highest-confidence signal when present
and may contain evidence for either field.

Eligible when merged_keywords >= 1 OR source_material_hint is present.
Near-zero skip rate (~21 movies lack all inputs).

Provider/model defaults: OpenAI gpt-5-mini, reasoning_effort: low.
"""

from typing import Tuple

from schemas.enums import MetadataType
from schemas.movie_input import MovieInputData
from schemas.metadata import SourceOfInspirationOutput
from movie_ingestion.metadata_generation.inputs import build_user_prompt
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
    section. Primary inputs are always included; missing values are
    rendered as "not available" so the model sees explicit absence.

    source_material_hint is the highest-confidence grounding signal when
    present — a short reviewer-extracted classification phrase from the
    Wave 1 reception generator (e.g., "based on autobiography", "remake",
    "based on book, sequel"). It may contain evidence for EITHER
    source_material or franchise_lineage — the prompt instructs the model
    to parse it for both.

    plot_synopsis was removed per ADR-033 — this generator barely uses
    plot data, and removing it saves ~83.6M input tokens across the corpus.
    """
    return build_user_prompt(
        title=movie.title_with_year(),
        merged_keywords=movie.merged_keywords() or "not available",
        source_material_hint=source_material_hint or "not available",
    )


async def generate_source_of_inspiration(
    movie: MovieInputData,
    source_material_hint: str | None = None,
) -> Tuple[SourceOfInspirationOutput, TokenUsage]:
    """Generate source of inspiration metadata for a single movie.

    Builds the user prompt from the movie's fields plus Wave 1 outputs,
    calls gpt-5-mini (low reasoning, low verbosity) with the base
    non-justification prompt, and returns the parsed result alongside
    token usage.

    This is the ONLY generation where parametric knowledge is allowed --
    the LLM may contribute source material and franchise facts from its
    training data when at least 95% confident.

    Args:
        movie: Raw movie input data loaded from the ingestion pipeline.
        source_material_hint: Short classifying phrase from Wave 1
            reception extraction zone (e.g., "based on autobiography",
            "remake", "sequel"). May be None if reception failed, was
            skipped, or reviewers didn't mention source material.
            May contain evidence for either source_material or
            franchise_lineage.

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
            provider=_DEFAULT_PROVIDER,
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            response_format=SourceOfInspirationOutput,
            model=_DEFAULT_MODEL,
            reasoning_effort="low",
            verbosity="low",
        )
    except Exception as e:
        print(f"{GENERATION_TYPE} generation failed for '{title_with_year}': {e}")
        raise MetadataGenerationError(GENERATION_TYPE, title_with_year, e) from e

    if parsed is None:
        print(f"{GENERATION_TYPE} generation returned None for '{title_with_year}'")
        raise MetadataGenerationEmptyResponseError(GENERATION_TYPE, title_with_year)

    return parsed, TokenUsage(input_tokens, output_tokens, _DEFAULT_MODEL)
