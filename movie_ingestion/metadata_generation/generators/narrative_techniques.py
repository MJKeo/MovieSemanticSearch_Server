"""
Narrative Techniques generator (Wave 2).

Async method that generates narrative techniques metadata for a single movie.
Extracts HOW the story is told -- cinematic narrative craft, structure,
storytelling mechanics. Powers queries like "movie with an unreliable narrator"
or "non-linear timeline."

Inputs:
    - movie (MovieInputData): raw fields for title, genres, merged_keywords
    - plot_summary: from Wave 1 plot_events output (may be None)
    - craft_observations: from Wave 1 reception extraction zone (may be None)

Narrative fallback order (shared with pre_consolidation eligibility):
    1. plot_summary (Wave 1 output) -- always preferred
    2. best_plot_fallback() if >= 500 chars standalone or >= 300 combined

Uses merged_keywords (plot + overall deduplicated) rather than
overall_keywords only. Structural technique tags can appear in either
keyword list, and the prompt handles noise by instructing the LLM to
use keywords only when consistent with primary evidence.

Skip condition: enforced by pre_consolidation using tiered eligibility
    (plot_summary OR fallback >= 500 OR craft >= 450 OR combined).

Response schema: NarrativeTechniquesWithJustificationsOutput (production).
    Justifications improve section grounding and are discarded before
    embedding.

Provider/model: OpenAI gpt-5-mini, reasoning_effort: minimal,
    verbosity: low. Finalized via evaluation pipeline.

See docs/llm_metadata_generation_new_flow.md Section 5.4.
"""

from typing import Tuple

from movie_ingestion.metadata_generation.inputs import (
    MetadataType,
    MovieInputData,
    build_user_prompt,
)
from movie_ingestion.metadata_generation.pre_consolidation import (
    resolve_narrative_techniques_narrative,
    _MIN_NT_CRAFT_OBSERVATIONS_COMBINED_CHARS,
)
from movie_ingestion.metadata_generation.schemas import (
    NarrativeTechniquesWithJustificationsOutput,
)
from movie_ingestion.metadata_generation.prompts.narrative_techniques import (
    SYSTEM_PROMPT_WITH_JUSTIFICATIONS,
)
from movie_ingestion.metadata_generation.errors import (
    MetadataGenerationError,
    MetadataGenerationEmptyResponseError,
)
from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from implementation.llms.vector_metadata_generation_methods import TokenUsage

GENERATION_TYPE = MetadataType.NARRATIVE_TECHNIQUES

# Finalized production config — gpt-5-mini with minimal reasoning, low
# verbosity, and justification schema. Determined via evaluation pipeline.
_PROVIDER = LLMProvider.OPENAI
_MODEL = "gpt-5-mini"
_RESPONSE_FORMAT = NarrativeTechniquesWithJustificationsOutput


def _filter_craft_observations(craft_observations: str | None) -> str | None:
    """Filter craft_observations by the combined-path inclusion threshold.

    Returns the observations if they meet the minimum length for inclusion,
    or None if below threshold. The standalone threshold (450 chars) is
    checked during eligibility; here we use the lower combined threshold
    (300 chars) since eligible movies may have craft_observations in the
    300-449 range when combined with plot fallback.
    """
    if craft_observations and len(craft_observations) >= _MIN_NT_CRAFT_OBSERVATIONS_COMBINED_CHARS:
        return craft_observations
    return None


def build_narrative_techniques_user_prompt(
    movie: MovieInputData,
    plot_summary: str | None = None,
    craft_observations: str | None = None,
) -> str:
    """Build the user prompt for narrative_techniques generation.

    Shared by the production generator and the evaluation pipeline so the
    prompt construction logic stays in one place.

    Resolves narrative input via the shared fallback ladder in
    pre_consolidation, and filters craft_observations by minimum length.
    The prompt labels distinguish data quality tiers so the LLM can
    calibrate confidence: "plot_synopsis" (LLM-condensed) vs "plot_text"
    (raw human-written).

    Absent primary inputs (plot data, craft observations) are explicitly
    included as "not available" rather than silently omitted. This helps
    the LLM calibrate confidence — seeing "plot_synopsis: not available"
    is a stronger signal than a missing field.

    Uses merged_keywords (plot + overall deduplicated) -- structural tags
    like "nonlinear timeline" and "unreliable narrator" can appear in
    either keyword list. The prompt handles noise by instructing the LLM
    to use keywords only when consistent with primary evidence.
    """
    # Resolve narrative input via shared fallback ladder
    narrative_text, narrative_label = resolve_narrative_techniques_narrative(
        movie, plot_summary,
    )

    # Filter craft observations by inclusion threshold
    filtered_craft = _filter_craft_observations(craft_observations)

    # Build prompt with labeled fields. The narrative label becomes the
    # field name (plot_synopsis or plot_text) so the LLM knows the
    # quality tier. Absent primary inputs are explicitly signaled as
    # "not available" so the LLM can calibrate confidence rather than
    # having to infer absence from a missing field.
    prompt_fields: dict[str, str | list | None] = {
        "title": movie.title_with_year(),
        "genres": movie.genres or None,
    }

    # Narrative input uses the resolved label as the field key.
    # When absent, explicitly signal so the LLM knows plot data is missing.
    if narrative_text and narrative_label:
        prompt_fields[narrative_label] = narrative_text
    else:
        prompt_fields["plot_synopsis"] = "not available"

    # Craft observations: explicit absence signal when filtered out
    prompt_fields["craft_observations"] = filtered_craft or "not available"
    prompt_fields["keywords"] = movie.merged_keywords() or None

    return build_user_prompt(**prompt_fields)


async def generate_narrative_techniques(
    movie: MovieInputData,
    plot_summary: str | None = None,
    craft_observations: str | None = None,
) -> Tuple[NarrativeTechniquesWithJustificationsOutput, TokenUsage]:
    """Generate narrative techniques metadata for a single movie.

    Builds the user prompt from the movie's fields plus Wave 1 outputs,
    calls OpenAI gpt-5-mini with justification schema, and returns the
    parsed result alongside token usage.

    Model configuration is finalized (gpt-5-mini, minimal reasoning,
    low verbosity, justification schema) and not caller-configurable.

    Args:
        movie: Raw movie input data loaded from the ingestion pipeline.
        plot_summary: Plot summary from Wave 1 plot_events. May be None
            if plot_events failed or was skipped.
        craft_observations: Craft/structural observations from Wave 1
            reception extraction zone. May be None if reception failed
            or had no craft observations.

    Returns:
        Tuple of (NarrativeTechniquesWithJustificationsOutput, TokenUsage).

    Raises:
        MetadataGenerationError: If the LLM call raises an exception.
        MetadataGenerationEmptyResponseError: If the LLM returns None.
    """
    user_prompt = build_narrative_techniques_user_prompt(
        movie, plot_summary, craft_observations,
    )
    title_with_year = movie.title_with_year()

    try:
        parsed, input_tokens, output_tokens = await generate_llm_response_async(
            provider=_PROVIDER,
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT_WITH_JUSTIFICATIONS,
            response_format=_RESPONSE_FORMAT,
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
