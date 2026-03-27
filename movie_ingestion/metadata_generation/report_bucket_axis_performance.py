"""
Print average evaluation performance per candidate per axis, split by bucket.

This reads the metadata-generation evaluation JSON files for a generation type,
maps each movie to its evaluation bucket, and prints one table per bucket with
the average score for every candidate on every evaluation axis.

Usage:
    python -m movie_ingestion.metadata_generation.report_bucket_axis_performance <generation_type>

Example:
    python -m movie_ingestion.metadata_generation.report_bucket_axis_performance narrative_techniques
"""

import argparse
import glob
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DATA_DIR = PROJECT_ROOT / "movie_ingestion" / "metadata_generation" / "evaluation_data"
INGESTION_DATA_DIR = PROJECT_ROOT / "ingestion_data"


@dataclass
class CandidateAxisStats:
    candidate_name: str
    score_values: dict[str, list[float]] = field(default_factory=dict)
    movie_count: int = 0


def load_bucket_definitions(generation_type: str) -> dict:
    """Load the bucket definition file for the generation type."""
    bucket_path = INGESTION_DATA_DIR / f"{generation_type}_eval_buckets.json"
    if not bucket_path.exists():
        raise FileNotFoundError(
            f"No bucket file found at {bucket_path}. "
            f"Available: {list(INGESTION_DATA_DIR.glob('*_eval_buckets.json'))}"
        )
    with open(bucket_path) as f:
        return json.load(f)


def _extract_tmdb_ids(bucket_payload: dict) -> list[str]:
    """Extract tmdb_ids from one bucket payload across known bucket formats."""
    if isinstance(bucket_payload.get("tmdb_ids"), list):
        return [str(tmdb_id) for tmdb_id in bucket_payload["tmdb_ids"]]

    if isinstance(bucket_payload.get("movies"), list):
        return [
            str(movie["tmdb_id"])
            for movie in bucket_payload["movies"]
            if isinstance(movie, dict) and movie.get("tmdb_id") is not None
        ]

    if isinstance(bucket_payload.get("samples"), list):
        return [
            str(sample["tmdb_id"])
            for sample in bucket_payload["samples"]
            if isinstance(sample, dict) and sample.get("tmdb_id") is not None
        ]

    return []


def build_id_to_bucket_map(buckets_data: dict) -> dict[str, str]:
    """Map tmdb_id (as string) -> bucket name for supported bucket file shapes."""
    mapping: dict[str, str] = {}

    if isinstance(buckets_data.get("buckets"), dict):
        bucket_items = buckets_data["buckets"].items()
    else:
        bucket_items = [
            (bucket_name, bucket_payload)
            for bucket_name, bucket_payload in buckets_data.items()
            if isinstance(bucket_payload, dict)
        ]

    for bucket_name, bucket_payload in bucket_items:
        for tmdb_id in _extract_tmdb_ids(bucket_payload):
            mapping[tmdb_id] = bucket_name

    return mapping


def collect_bucket_axis_scores(
    generation_type: str,
    id_to_bucket: dict[str, str],
) -> tuple[dict[str, dict[str, CandidateAxisStats]], list[str]]:
    """
    Collect per-bucket axis scores from evaluation JSON files.

    Returns:
        (
            {
                bucket_name: {
                    candidate_name: CandidateAxisStats(...)
                }
            },
            sorted_axis_names,
        )
    """
    pattern = str(EVAL_DATA_DIR / f"{generation_type}_*_evaluation.json")
    files = sorted(glob.glob(pattern))

    bucket_results: dict[str, dict[str, CandidateAxisStats]] = defaultdict(dict)
    all_axes: set[str] = set()

    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)

        tmdb_id = str(data.get("tmdb_id"))
        bucket_name = id_to_bucket.get(tmdb_id, "unknown")

        for candidate_name, candidate_eval in data.get("candidate_evaluations", {}).items():
            if not isinstance(candidate_eval, dict):
                continue

            axis_scores = {
                axis_name: score
                for axis_name, score in candidate_eval.items()
                if axis_name.endswith("_score") and isinstance(score, (int, float))
            }
            if not axis_scores:
                continue

            candidate_stats = bucket_results[bucket_name].get(candidate_name)
            if candidate_stats is None:
                candidate_stats = CandidateAxisStats(
                    candidate_name=candidate_name,
                    score_values=defaultdict(list),
                )
                bucket_results[bucket_name][candidate_name] = candidate_stats

            candidate_stats.movie_count += 1
            for axis_name, score in axis_scores.items():
                candidate_stats.score_values[axis_name].append(float(score))
                all_axes.add(axis_name)

    return bucket_results, sorted(all_axes)


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _column_widths(
    axes: list[str],
    rows: list[tuple[str, list[str]]],
) -> tuple[int, dict[str, int]]:
    candidate_width = max(
        len("Candidate"),
        max((len(candidate_name) for candidate_name, _ in rows), default=0),
    )
    axis_widths = {}
    for axis_index, axis_name in enumerate(axes):
        display_name = axis_name.removesuffix("_score")
        axis_widths[axis_name] = max(
            len(display_name),
            max((len(row_values[axis_index]) for _, row_values in rows), default=0),
        )
    return candidate_width, axis_widths


def print_bucket_table(
    bucket_name: str,
    candidate_stats_by_name: dict[str, CandidateAxisStats],
    axes: list[str],
) -> None:
    """Print one table for a single bucket."""
    print(f"\nBucket: {bucket_name}")
    if not candidate_stats_by_name:
        print("No evaluation data.")
        return

    rows: list[tuple[str, list[str]]] = []
    for candidate_name in sorted(candidate_stats_by_name):
        stats = candidate_stats_by_name[candidate_name]
        row_values = []
        for axis_name in axes:
            avg_score = _avg(stats.score_values.get(axis_name, []))
            row_values.append(f"{avg_score:.2f}" if avg_score is not None else "-")
        rows.append((candidate_name, row_values))

    candidate_width, axis_widths = _column_widths(axes, rows)

    header = [
        f"{'Candidate':<{candidate_width}}",
    ]
    header.extend(
        f"{axis_name.removesuffix('_score'):>{axis_widths[axis_name]}}"
        for axis_name in axes
    )
    print("  " + "  ".join(header))

    separator_parts = [
        "─" * candidate_width,
    ]
    separator_parts.extend("─" * axis_widths[axis_name] for axis_name in axes)
    print("  " + "  ".join(separator_parts))

    for candidate_name, row_values in rows:
        cells = [
            f"{candidate_name:<{candidate_width}}",
        ]
        for axis_name, value in zip(axes, row_values, strict=True):
            cells.append(f"{value:>{axis_widths[axis_name]}}")
        print("  " + "  ".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate average evaluation performance per candidate per axis, "
            "with a separate table for each bucket."
        )
    )
    parser.add_argument(
        "generation_type",
        help="Metadata type (e.g. narrative_techniques, reception, viewer_experience)",
    )
    args = parser.parse_args()

    buckets_data = load_bucket_definitions(args.generation_type)
    id_to_bucket = build_id_to_bucket_map(buckets_data)
    bucket_results, axes = collect_bucket_axis_scores(args.generation_type, id_to_bucket)

    if not bucket_results:
        print(f"No evaluation data found for '{args.generation_type}'.")
        print(f"Looked in: {EVAL_DATA_DIR}/{args.generation_type}_*_evaluation.json")
        return

    print(f"Generation type: {args.generation_type}")
    print(f"Evaluation axes: {', '.join(axes)}")
    print(f"Buckets with data: {len(bucket_results)}")

    for bucket_name in sorted(bucket_results):
        print_bucket_table(bucket_name, bucket_results[bucket_name], axes)


if __name__ == "__main__":
    main()
