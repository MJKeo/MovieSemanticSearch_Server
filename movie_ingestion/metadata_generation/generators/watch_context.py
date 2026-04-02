"""
Watch Context generator (Wave 2).

Async method that generates watch context metadata for a single movie.
Extracts WHY and WHEN someone would choose to watch this movie. Powers
queries like "date night movie" or "something to watch high."

CRITICAL DESIGN DECISION: Watch context receives ZERO plot information.
No overview, no plot_synopsis. It answers "watch this if you want X
attributes" -- not "watch this if you want these specific events."
Plot detail anchors the model on narrative events rather than
experiential attributes.

Inputs:
    - movie (MovieInputData): raw fields for title, overall_keywords, maturity
    - genre_signatures: LLM-refined genre phrases from plot_analysis (Wave 2).
      Falls back to raw movie.genres when unavailable.
    - emotional_observations: from Wave 1 reception extraction zone (may be None)
    - craft_observations: from Wave 1 reception extraction zone (may be None)
    - thematic_observations: from Wave 1 reception extraction zone (may be None)

Removed inputs (vs current system):
    - overview: no plot info in watch context
    - plot_keywords: not relevant to viewing occasions
    - merged_keywords: replaced by overall_keywords only
    - review_insights_brief: replaced by individual observation fields
    - reception_summary / audience_reception_attributes / featured_reviews:
      subsumed by individual observation fields

Skip condition: (genre_signatures OR genres >= 1) AND at least one
    observation field (emotional, craft, or thematic). Enforced by
    pre_consolidation. Genre-only inputs without observations produce
    generic, undifferentiated terms — evaluation Phase 1 showed all
    candidates score 1.6-2.5 on observation-absent movies.

Response schema: WatchContextWithIdentityNoteOutput — includes a brief
    identity_note pre-classification (2-8 words) plus evidence_basis
    per section. Evolved from evaluation rounds 1-4: identity_note
    provides priming for ambiguous-identity movies without the
    over-constraining regression seen with full-sentence
    viewing_appeal_summary (Round 3 H12).

Provider/model: OpenAI gpt-5-mini, reasoning_effort: minimal.
    Finalized after 4 evaluation rounds (50 movies, 6 buckets).

See docs/llm_metadata_generation_new_flow.md Section 5.3.
See evaluation_data/watch_context_eval_guide.md for input contract rationale.
"""

from typing import Tuple

from movie_ingestion.metadata_generation.inputs import (
    MetadataType,
    MovieInputData,
    build_user_prompt,
)
from movie_ingestion.metadata_generation.schemas import WatchContextWithIdentityNoteOutput
from movie_ingestion.metadata_generation.prompts.watch_context import (
    SYSTEM_PROMPT,  # noqa: F401 — exported for evaluation pipeline
    SYSTEM_PROMPT_WITH_JUSTIFICATIONS,
    SYSTEM_PROMPT_WITH_IDENTITY_NOTE,
)
from movie_ingestion.metadata_generation.errors import (
    MetadataGenerationError,
    MetadataGenerationEmptyResponseError,
)
from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from implementation.llms.vector_metadata_generation_methods import TokenUsage

GENERATION_TYPE = MetadataType.WATCH_CONTEXT

# Production config — finalized after 4 evaluation rounds (r4-identity-note-minimal).
_PROVIDER = LLMProvider.OPENAI
_MODEL = "gpt-5-mini"
_SYSTEM_PROMPT = SYSTEM_PROMPT_WITH_IDENTITY_NOTE
_RESPONSE_FORMAT = WatchContextWithIdentityNoteOutput
_LLM_KWARGS = {"reasoning_effort": "minimal", "verbosity": "low"}


def _classify_input_richness(
    emotional_observations: str | None,
    craft_observations: str | None,
    thematic_observations: str | None,
) -> str | None:
    """Classify observation richness and return a guidance line for the prompt.

    Deterministic Python-side classification so the LLM doesn't spend
    reasoning tokens deciding what input tier it's in. Returns None for
    rich inputs (no extra guidance needed).

    Thresholds:
        - "present" = non-None and > 20 chars (excludes trivial stubs)
        - "rich" = present and > 200 chars (substantial observation text)
    """
    observations = [emotional_observations, craft_observations, thematic_observations]
    present = [o for o in observations if o and len(o.strip()) > 20]
    rich = [o for o in present if len(o.strip()) > 200]

    if len(rich) >= 2:
        # Rich inputs — no extra guidance needed, full coverage expected
        return None
    elif len(present) <= 1:
        # Sparse: 0-1 short observations available
        return (
            "INPUT CALIBRATION: Your inputs are sparse. Favor sections "
            "directly informed by available observations and keep other "
            "sections empty or minimal."
        )
    else:
        # Middle ground: 2-3 observations present but mostly short
        return (
            "INPUT CALIBRATION: Your observation inputs are present but "
            "thin. Generate terms proportional to the depth of evidence — "
            "short observations warrant fewer terms than detailed ones."
        )


def build_watch_context_user_prompt(
    movie: MovieInputData,
    genre_signatures: list[str] | None = None,
    emotional_observations: str | None = None,
    craft_observations: str | None = None,
    thematic_observations: str | None = None,
) -> str:
    """Build the user prompt for watch_context generation from a movie's fields.

    Shared by the production generator and the evaluation pipeline so the
    prompt construction logic stays in one place.

    CRITICAL: No plot information is included. Watch context operates on
    experiential signals only (observations, genres, keywords, maturity).

    Genre input uses genre_signatures (LLM-refined from plot_analysis) when
    available, falling back to raw genres. Primary observation fields use
    "not available" when absent so the model sees explicit absence signals.

    Inputs are labeled for the LLM to match the SYSTEM_PROMPT's INPUTS
    section. Secondary inputs (overall_keywords, maturity_summary) are
    omitted when empty.

    A dynamic INPUT CALIBRATION line is prepended when inputs are sparse
    or thin, so the model receives richness guidance without needing to
    self-assess. Rich inputs get no extra guidance.
    """
    # Genre input: prefer genre_signatures, fall back to raw genres
    genre_input = genre_signatures if genre_signatures else (movie.genres or None)

    # Dynamic richness guidance — classified in Python, not by the LLM
    richness_guidance = _classify_input_richness(
        emotional_observations, craft_observations, thematic_observations,
    )

    prompt = build_user_prompt(
        title=movie.title_with_year(),
        genre_signatures=genre_input,
        overall_keywords=movie.overall_keywords or None,
        maturity_summary=movie.maturity_summary(),
        emotional_observations=emotional_observations or "not available",
        craft_observations=craft_observations or "not available",
        thematic_observations=thematic_observations or "not available",
    )

    # Prepend richness guidance when inputs are sparse or thin
    if richness_guidance:
        prompt = f"{richness_guidance}\n\n{prompt}"

    return prompt


async def generate_watch_context(
    movie: MovieInputData,
    genre_signatures: list[str] | None = None,
    emotional_observations: str | None = None,
    craft_observations: str | None = None,
    thematic_observations: str | None = None,
) -> Tuple[WatchContextWithIdentityNoteOutput, TokenUsage]:
    """Generate watch context metadata for a single movie.

    Builds the user prompt from the movie's experiential fields plus
    Wave 1 observation fields and Wave 2 genre_signatures, calls
    gpt-5-mini with structured output, and returns the parsed result
    alongside token usage.

    No plot_synopsis parameter — watch context deliberately receives zero
    plot information (Decision 2 in the redesigned flow spec).

    Uses the finalized production config (r4-identity-note-minimal):
    OpenAI gpt-5-mini, minimal reasoning, identity_note schema with
    evidence_basis per section.

    Args:
        movie: Raw movie input data loaded from the ingestion pipeline.
        genre_signatures: LLM-refined genre phrases from plot_analysis.
            Falls back to raw movie.genres when None.
        emotional_observations: Emotional tone/mood observations from
            Wave 1 reception extraction zone. May be None.
        craft_observations: Performance/cinematography/craft observations
            from Wave 1 reception extraction zone. May be None.
        thematic_observations: Theme/meaning observations from Wave 1
            reception extraction zone. May be None.

    Returns:
        Tuple of (WatchContextWithIdentityNoteOutput, TokenUsage).

    Raises:
        MetadataGenerationError: If the LLM call raises an exception.
        MetadataGenerationEmptyResponseError: If the LLM returns None.
    """
    user_prompt = build_watch_context_user_prompt(
        movie, genre_signatures, emotional_observations,
        craft_observations, thematic_observations,
    )
    title_with_year = movie.title_with_year()

    try:
        parsed, input_tokens, output_tokens = await generate_llm_response_async(
            provider=_PROVIDER,
            user_prompt=user_prompt,
            system_prompt=_SYSTEM_PROMPT,
            response_format=_RESPONSE_FORMAT,
            model=_MODEL,
            **_LLM_KWARGS,
        )
    except Exception as e:
        print(f"{GENERATION_TYPE} generation failed for '{title_with_year}': {e}")
        raise MetadataGenerationError(GENERATION_TYPE, title_with_year, e) from e

    if parsed is None:
        print(f"{GENERATION_TYPE} generation returned None for '{title_with_year}'")
        raise MetadataGenerationEmptyResponseError(GENERATION_TYPE, title_with_year)

    return parsed, TokenUsage(input_tokens, output_tokens, _MODEL)
