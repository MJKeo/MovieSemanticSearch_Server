"""
CLI entry point for the LLM metadata evaluation pipeline.

Loads test corpus movies, filters out movies that lack sufficient data,
then runs candidate evaluation for each metadata type. Each candidate
output is scored by a rubric-based LLM judge (no reference outputs).

Currently supports:
    - plot_events

Usage:
    python -m movie_ingestion.metadata_generation.evaluations.run_evaluations_pipeline
    python -m movie_ingestion.metadata_generation.evaluations.run_evaluations_pipeline --branch synopsis
    python -m movie_ingestion.metadata_generation.evaluations.run_evaluations_pipeline --branch synthesis
"""

import argparse
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
    run_evaluation,
)
from movie_ingestion.metadata_generation.inputs import MovieInputData
from movie_ingestion.metadata_generation.pre_consolidation import check_plot_events


def _filter_plot_events_eligible(
    movie_inputs: dict[int, MovieInputData],
    branch: str | None = None,
) -> dict[int, MovieInputData]:
    """Return only movies that have sufficient data to generate plot_events metadata.

    Calls check_plot_events() directly to test whether each movie has enough
    text data (overview, synopses, summaries) to produce meaningful output.

    When branch is specified, additionally filters to only movies matching
    the requested branch:
    - "synopsis": only movies with at least one synopsis
    - "synthesis": only movies without any synopsis

    Skipped movies are printed with their reason so the caller can see what
    was excluded.
    """
    eligible: dict[int, MovieInputData] = {}
    for tmdb_id, movie_input in movie_inputs.items():
        skip_reason = check_plot_events(movie_input)
        if skip_reason is not None:
            print(f"  SKIPPED {tmdb_id} ({movie_input.title_with_year()}): {skip_reason}")
            continue

        # Branch filtering — only include movies matching the requested branch
        if branch == "synopsis" and not movie_input.plot_synopses:
            print(f"  SKIPPED {tmdb_id} ({movie_input.title_with_year()}): no synopsis (branch=synopsis)")
            continue
        if branch == "synthesis" and movie_input.plot_synopses:
            print(f"  SKIPPED {tmdb_id} ({movie_input.title_with_year()}): has synopsis (branch=synthesis)")
            continue

        eligible[tmdb_id] = movie_input
    return eligible


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the LLM metadata evaluation pipeline.",
    )
    parser.add_argument(
        "--branch",
        choices=["synopsis", "synthesis"],
        default=None,
        help=(
            "Filter test corpus to a specific plot_events branch. "
            "'synopsis' = only movies with a synopsis (condensation branch). "
            "'synthesis' = only movies without a synopsis (synthesis branch). "
            "Omit to evaluate all eligible movies."
        ),
    )
    return parser.parse_args()


async def main(branch: str | None = None) -> None:
    """Run the full evaluation pipeline for all supported metadata types.

    Args:
        branch: Optional branch filter ("synopsis" or "synthesis").
    """

    print(f"Loading movie input data for {len(EVALUATION_TEST_SET_TMDB_IDS)} movie(s)...")
    # temp_evaluation_set = ORIGINAL_SET_TMDB_IDS[:5] + MEDIUM_SPARSITY_TMDB_IDS[:3] + HIGH_SPARSITY_TMDB_IDS[:3]
    movie_inputs = load_movie_input_data(EVALUATION_TEST_SET_TMDB_IDS[:1])

    if not movie_inputs:
        print("No movies loaded — check that the ingestion pipeline has run.")
        return

    # Filter out movies that lack sufficient data for plot_events generation
    # before committing any LLM spend on them.
    branch_label = f" (branch={branch})" if branch else ""
    print(f"\nChecking plot_events eligibility{branch_label} for {len(movie_inputs)} loaded movies...")
    eligible_inputs = _filter_plot_events_eligible(movie_inputs, branch=branch)

    skipped_count = len(movie_inputs) - len(eligible_inputs)
    print(f"Eligible: {len(eligible_inputs)} | Skipped: {skipped_count}")

    if not eligible_inputs:
        print("No eligible movies — nothing to evaluate.")
        return

    # Candidate evaluation: generate outputs and score with rubric-based judge
    print(f"\n--- Candidate Evaluation (plot_events{branch_label}) ---")
    await run_evaluation(
        candidates=PLOT_EVENTS_CANDIDATES,
        movie_inputs=eligible_inputs,
        concurrency=4,
    )

    print("\nPipeline complete.")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(branch=args.branch))
