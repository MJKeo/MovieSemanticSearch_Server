"""
Franchise generator (v9 — adds exact-match franchise reference
anchors plus new worked examples for selective continuity, same-
source adaptation boundaries, recast-without-reset sequels, and
trunk-prequel backstory cases).

Produces FranchiseOutput for a movie along three orthogonal blocks:

    IDENTITY — lineage (narrowest recognizable line), shared_universe
               (broader entity: formal shared cosmos OR parent
               franchise of a spinoff sub-lineage),
               recognized_subgroups (named sub-phases), and
               launched_subgroup (earliest-released entry in one of
               those subgroups).

    NARRATIVE POSITION — lineage_position (mutually exclusive enum:
                         sequel / prequel / remake / reboot / null),
                         is_crossover (independent boolean driven by
                         a single identity question), and is_spinoff
                         (independent boolean driven by structural
                         trunk-vs-branch situating with a parametric-
                         knowledge supplement).

    FRANCHISE LAUNCH — launched_franchise (boolean, four-part test:
                       first cinematic entry, not a spinoff, source-
                       material recognition test, relevant follow-ups
                       test).

See schemas/metadata.py::FranchiseOutput for the schema contract and
movie_ingestion/metadata_generation/prompts/franchise.py for the
system prompt that drives generation.
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
_DEFAULT_MODEL = "gpt-5.4-mini"
_DEFAULT_KWARGS = {"reasoning_effort": "low", "verbosity": "low"}


def build_franchise_user_prompt(movie: MovieInputData) -> str:
    """Build the user prompt for franchise generation.

    Inputs are chosen to give the model the strongest possible
    signals for lineage/shared_universe identification,
    launched_subgroup chronology, lineage_position (sequel / prequel
    / remake / reboot) reasoning, and director/actor-era subgroup
    detection (recognized_subgroups):
    - release_year is labeled separately so the model can anchor
      "is this film the earliest-released entry of its subgroup?"
      and prequel/sequel ordering reasoning.
    - directors (explicitly labeled) support director-era subgroup
      detection ("raimi trilogy", "nolan batman era", "snyderverse").
    - top_billed_cast pairs actors with characters, which strengthens
      actor-era subgroup detection ("daniel craig era") and the
      PRIOR-ROLE PROMINENCE TEST used in the spinoff constraint
      check.
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
) -> Tuple[FranchiseOutput, TokenUsage]:
    """Generate franchise metadata for a single movie.

    Uses the hardcoded default model (gpt-5.4-mini with low reasoning
    effort and low verbosity).

    Args:
        movie: Raw movie input data from the ingestion pipeline.

    Returns:
        Tuple of (FranchiseOutput, TokenUsage).

    Raises:
        MetadataGenerationError: If the LLM call raises an exception.
        MetadataGenerationEmptyResponseError: If the LLM returns None.
    """
    user_prompt = build_franchise_user_prompt(movie)
    title_with_year = movie.title_with_year()

    try:
        parsed, input_tokens, output_tokens = await generate_llm_response_async(
            provider=_DEFAULT_PROVIDER,
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            response_format=FranchiseOutput,
            model=_DEFAULT_MODEL,
            **_DEFAULT_KWARGS,
        )
    except Exception as e:
        print(f"{GENERATION_TYPE} generation failed for '{title_with_year}': {e}")
        raise MetadataGenerationError(GENERATION_TYPE, title_with_year, e) from e

    if parsed is None:
        print(f"{GENERATION_TYPE} generation returned None for '{title_with_year}'")
        raise MetadataGenerationEmptyResponseError(GENERATION_TYPE, title_with_year)

    return parsed, TokenUsage(input_tokens, output_tokens, _DEFAULT_MODEL)
