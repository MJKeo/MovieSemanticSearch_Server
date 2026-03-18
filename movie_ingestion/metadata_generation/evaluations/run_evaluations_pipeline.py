"""
CLI entry point for the LLM metadata evaluation pipeline.

Loads test corpus movies, filters out movies that lack sufficient data,
then runs reference generation (Phase 0) and candidate evaluation (Phase 1)
for each metadata type.

Currently supports:
    - plot_events

Usage:
    python -m movie_ingestion.metadata_generation.evaluations.run_evaluations_pipeline
"""

import asyncio

from movie_ingestion.metadata_generation.evaluations.shared import (
    EVALUATION_TEST_SET_TMDB_IDS,
    ORIGINAL_SET_TMDB_IDS,
    MEDIUM_SPARSITY_TMDB_IDS,
    HIGH_SPARSITY_TMDB_IDS,
    load_movie_input_data,
)
from movie_ingestion.metadata_generation.evaluations.plot_events import (
    PLOT_EVENTS_CANDIDATES,
    generate_reference_responses,
    run_evaluation,
)
from movie_ingestion.metadata_generation.inputs import MovieInputData
from movie_ingestion.metadata_generation.pre_consolidation import check_plot_events


def _filter_plot_events_eligible(
    movie_inputs: dict[int, MovieInputData],
) -> dict[int, MovieInputData]:
    """Return only movies that have sufficient data to generate plot_events metadata.

    Calls check_plot_events() directly to test whether each movie has enough
    text data (overview, synopses, summaries) to produce meaningful output.
    Skipped movies are printed with their reason so the caller can see what
    was excluded.
    """
    eligible: dict[int, MovieInputData] = {}
    for tmdb_id, movie_input in movie_inputs.items():
        skip_reason = check_plot_events(movie_input)
        if skip_reason is None:
            eligible[tmdb_id] = movie_input
        else:
            print(f"  SKIPPED {tmdb_id} ({movie_input.title_with_year()}): {skip_reason}")
    return eligible


async def main() -> None:
    """Run the full evaluation pipeline for all supported metadata types."""

    print(f"Loading movie input data for {len(EVALUATION_TEST_SET_TMDB_IDS)} movie(s)...")
    temp_evaluation_set = ORIGINAL_SET_TMDB_IDS[:5] + MEDIUM_SPARSITY_TMDB_IDS[:3] + HIGH_SPARSITY_TMDB_IDS[:3]
    movie_inputs = load_movie_input_data(temp_evaluation_set)

    if not movie_inputs:
        print("No movies loaded — check that the ingestion pipeline has run.")
        return

    # Filter out movies that lack sufficient data for plot_events generation
    # before committing any LLM spend on them.
    print(f"\nChecking plot_events eligibility for {len(movie_inputs)} loaded movies...")
    eligible_inputs = _filter_plot_events_eligible(movie_inputs)

    skipped_count = len(movie_inputs) - len(eligible_inputs)
    print(f"Eligible: {len(eligible_inputs)} | Skipped: {skipped_count}")

    if not eligible_inputs:
        print("No eligible movies — nothing to evaluate.")
        return

    # Phase 0: generate reference responses using GPT-5.4
    print("\n--- Phase 0: Reference Generation (plot_events) ---")
    await generate_reference_responses(eligible_inputs)

    # Phase 1: generate candidate outputs and score them with a judge
    print("\n--- Phase 1: Candidate Evaluation (plot_events) ---")
    await run_evaluation(
        candidates=PLOT_EVENTS_CANDIDATES,
        movie_inputs=eligible_inputs,
        concurrency=10,
    )

    print("\nPipeline complete.")


if __name__ == "__main__":
    asyncio.run(main())
