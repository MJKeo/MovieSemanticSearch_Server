"""
Estimate the cost of running a metadata generation type across the full corpus.

Uses token usage from evaluation data and movie counts from the tracker DB
to project per-candidate costs at both standard and batch API pricing.

Usage:
    python -m movie_ingestion.metadata_generation.estimate_generation_cost <generation_type>

Example:
    python -m movie_ingestion.metadata_generation.estimate_generation_cost viewer_experience
"""

import argparse
import glob
import json
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DATA_DIR = PROJECT_ROOT / "movie_ingestion" / "metadata_generation" / "evaluation_data"
INGESTION_DATA_DIR = PROJECT_ROOT / "ingestion_data"
TRACKER_DB_PATH = INGESTION_DATA_DIR / "tracker.db"


MODEL_PRICING: dict[str, tuple[float, float]] = {
    # (input_price_per_M, output_price_per_M)
    "qwen3.5-flash":                                   (0.10, 0.40),
    "gemini-2.5-flash":                                (0.30, 2.50),
    "gemini-2.5-flash-lite":                           (0.10, 0.40),
    "gemini-3.1-flash-lite":                           (0.10, 0.40),
    "gpt-5-mini":                                      (0.25, 2.00),
    "gpt-5-nano":                                      (0.05, 0.40),
    "gpt-5.4-nano":                                    (0.20, 1.25),
    "openai/gpt-oss-120b":                             (0.15, 0.60),
    "meta-llama/llama-4-scout-17b-16e-instruct":       (0.11, 0.34),
    "kimi-k2.5":                                       (0.60, 3.00),
    "gemini-3.1-flash-lite-preview":                   (0.25, 1.50),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class MovieTokenUsage:
    input_tokens: int
    output_tokens: int
    cost_usd: float  # as reported in eval data (may be 0 if not tracked)


@dataclass
class BucketStats:
    bucket_name: str
    sample_count: int
    avg_input_tokens: float
    avg_output_tokens: float
    avg_cost_usd: float


@dataclass
class CandidateStats:
    candidate_name: str
    model: str
    bucket_stats: list[BucketStats] = field(default_factory=list)
    avg_quality: float | None = None
    quality_sample_count: int = 0


# ---------------------------------------------------------------------------
# Step 1: Gather token usage from evaluation data
# ---------------------------------------------------------------------------
def load_bucket_definitions(generation_type: str) -> dict:
    """Load bucket definitions and build tmdb_id -> bucket_name mapping."""
    bucket_path = INGESTION_DATA_DIR / f"{generation_type}_eval_buckets.json"
    if not bucket_path.exists():
        raise FileNotFoundError(
            f"No bucket file found at {bucket_path}. "
            f"Available: {list(INGESTION_DATA_DIR.glob('*_eval_buckets.json'))}"
        )
    with open(bucket_path) as f:
        return json.load(f)


def build_id_to_bucket_map(buckets_data: dict) -> dict[str, str]:
    """Map tmdb_id (as string) -> bucket name."""
    mapping = {}
    for bucket_name, bucket_info in buckets_data.get("buckets", {}).items():
        for movie in bucket_info.get("movies", []):
            mapping[str(movie.get("tmdb_id"))] = bucket_name
    return mapping


def collect_token_usage(
    generation_type: str, id_to_bucket: dict[str, str]
) -> dict[str, dict[str, list[MovieTokenUsage]]]:
    """
    Parse all non-evaluation JSON files for the generation type.

    Returns: {candidate_name: {bucket_name: [MovieTokenUsage, ...]}}
    """
    pattern = str(EVAL_DATA_DIR / f"{generation_type}_*.json")
    files = [f for f in glob.glob(pattern) if "_evaluation.json" not in f]

    # {candidate: {bucket: [usages]}}
    results: dict[str, dict[str, list[MovieTokenUsage]]] = {}
    # Track model per candidate
    models: dict[str, str] = {}

    for fpath in files:
        tmdb_id = (
            os.path.basename(fpath)
            .replace(f"{generation_type}_", "")
            .replace(".json", "")
        )
        with open(fpath) as f:
            data = json.load(f)

        bucket = id_to_bucket.get(tmdb_id, "unknown")

        for cand_name, cand_data in data.get("candidate_results", {}).items():
            if cand_name not in results:
                results[cand_name] = {}
            if cand_name not in models:
                models[cand_name] = cand_data.get("model", "unknown")
            if bucket not in results[cand_name]:
                results[cand_name][bucket] = []
            results[cand_name][bucket].append(
                MovieTokenUsage(
                    input_tokens=cand_data.get("input_tokens") or 0,
                    output_tokens=cand_data.get("output_tokens") or 0,
                    cost_usd=cand_data.get("cost_usd") or 0.0,
                )
            )

    return results, models


def collect_quality_scores(generation_type: str) -> dict[str, list[float]]:
    """
    Parse evaluation files and compute average score per candidate.

    Averages all numeric *_score fields in each candidate evaluation.
    Returns: {candidate_name: [per-movie average scores]}
    """
    pattern = str(EVAL_DATA_DIR / f"{generation_type}_*_evaluation.json")
    files = glob.glob(pattern)

    scores: dict[str, list[float]] = {}
    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        for cand_name, cand_data in data.get("candidate_evaluations", {}).items():
            if not isinstance(cand_data, dict):
                continue
            # Average all fields ending in _score
            score_values = [
                v for k, v in cand_data.items()
                if k.endswith("_score") and isinstance(v, (int, float))
            ]
            if score_values:
                scores.setdefault(cand_name, []).append(
                    sum(score_values) / len(score_values)
                )
    return scores


def _compute_cost(usage: MovieTokenUsage, model: str) -> float:
    """Return the recorded cost_usd, or estimate from MODEL_PRICING as fallback."""
    if usage.cost_usd > 0:
        return usage.cost_usd
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return 0.0
    input_price_per_m, output_price_per_m = pricing
    return (
        usage.input_tokens * input_price_per_m / 1_000_000
        + usage.output_tokens * output_price_per_m / 1_000_000
    )


def build_candidate_stats(
    token_usage: dict,
    models: dict[str, str],
    quality_scores: dict[str, list[float]],
) -> list[CandidateStats]:
    """Aggregate per-bucket stats and quality scores into CandidateStats."""
    candidates = []
    for cand_name in sorted(token_usage.keys()):
        model = models.get(cand_name, "unknown")
        buckets = token_usage[cand_name]
        bucket_stats = []
        for bucket_name in sorted(buckets.keys()):
            usages = buckets[bucket_name]
            n = len(usages)
            costs = [_compute_cost(u, model) for u in usages]
            bucket_stats.append(
                BucketStats(
                    bucket_name=bucket_name,
                    sample_count=n,
                    avg_input_tokens=sum(u.input_tokens for u in usages) / n,
                    avg_output_tokens=sum(u.output_tokens for u in usages) / n,
                    avg_cost_usd=sum(costs) / n,
                )
            )

        q_scores = quality_scores.get(cand_name, [])
        candidates.append(
            CandidateStats(
                candidate_name=cand_name,
                model=models.get(cand_name, "unknown"),
                bucket_stats=bucket_stats,
                avg_quality=(sum(q_scores) / len(q_scores)) if q_scores else None,
                quality_sample_count=len(q_scores),
            )
        )
    return candidates


# ---------------------------------------------------------------------------
# Step 2: Get corpus movie count
# ---------------------------------------------------------------------------
ELIGIBLE_STATUSES = (
    "imdb_quality_passed",
    "metadata_generated",
    "embedded",
    "ingested",
)


def get_eligible_movie_count(generation_type: str) -> int:
    """Count movies eligible for a specific metadata generation type.

    Uses the eligible_for_{type} flag in the generated_metadata table when
    available (most accurate — reflects actual input-quality checks). Falls
    back to counting movies at imdb_quality_passed or later in movie_progress
    if eligibility hasn't been evaluated yet for this type.
    """
    conn = sqlite3.connect(str(TRACKER_DB_PATH))
    try:
        eligibility_col = f"eligible_for_{generation_type}"

        # Count movies explicitly marked eligible for this type
        row = conn.execute(
            f"SELECT COUNT(*) FROM generated_metadata WHERE {eligibility_col} = 1",
        ).fetchone()
        eligible_count = row[0]

        if eligible_count > 0:
            return eligible_count

        # Fallback: eligibility not yet evaluated — use pipeline status count
        placeholders = ",".join("?" for _ in ELIGIBLE_STATUSES)
        row = conn.execute(
            f"SELECT COUNT(*) FROM movie_progress WHERE status IN ({placeholders})",
            ELIGIBLE_STATUSES,
        ).fetchone()
        return row[0]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Step 3 & 5: Display results
# ---------------------------------------------------------------------------
def print_detailed_table(candidate: CandidateStats, total_movies: int) -> None:
    """Print per-bucket projection for a single candidate."""
    print(f"\n{'─' * 100}")
    quality_str = f"{candidate.avg_quality:.2f}" if candidate.avg_quality else "N/A"
    print(
        f"  {candidate.candidate_name}  │  model: {candidate.model}  │  "
        f"avg quality: {quality_str} (n={candidate.quality_sample_count})"
    )
    print(f"{'─' * 100}")
    header = (
        f"  {'Bucket':<35} {'Sample':>6} {'Avg In':>8} {'Avg Out':>8} "
        f"{'Avg $/movie':>12} {'Proj Cost':>12}"
    )
    print(header)
    print(f"  {'─' * 93}")

    # Use uniform movie count (we don't have per-bucket corpus counts for
    # all types, so we project using the overall average cost per movie).
    overall_avg_cost = _weighted_avg_cost(candidate)
    for bs in candidate.bucket_stats:
        # Per-bucket projected cost assumes equal distribution — shown for
        # per-bucket cost visibility, but the total uses weighted average.
        print(
            f"  {bs.bucket_name:<35} {bs.sample_count:>6} {bs.avg_input_tokens:>8.0f} "
            f"{bs.avg_output_tokens:>8.0f} {bs.avg_cost_usd:>11.6f}  "
        )

    projected_standard = overall_avg_cost * total_movies
    projected_batch = projected_standard * 0.5
    print(f"  {'─' * 93}")
    print(f"  {'Weighted avg cost/movie:':<55} ${overall_avg_cost:.6f}")
    print(f"  {f'Projected STANDARD cost ({total_movies:,} movies):':<55} ${projected_standard:,.2f}")
    print(f"  {f'Projected BATCH cost (50% discount):':<55} ${projected_batch:,.2f}")


def _weighted_avg_cost(candidate: CandidateStats) -> float:
    """Weighted average cost across all sampled movies."""
    total_cost = 0.0
    total_n = 0
    for bs in candidate.bucket_stats:
        total_cost += bs.avg_cost_usd * bs.sample_count
        total_n += bs.sample_count
    return total_cost / total_n if total_n else 0.0


def _weighted_avg_tokens(candidate: CandidateStats) -> tuple[float, float]:
    """Weighted average input/output tokens across all sampled movies."""
    total_in = total_out = 0.0
    total_n = 0
    for bs in candidate.bucket_stats:
        total_in += bs.avg_input_tokens * bs.sample_count
        total_out += bs.avg_output_tokens * bs.sample_count
        total_n += bs.sample_count
    if total_n == 0:
        return 0.0, 0.0
    return total_in / total_n, total_out / total_n


def print_summary_table(candidates: list[CandidateStats], total_movies: int) -> None:
    """Print the final comparison table across all candidates."""
    print(f"\n{'═' * 110}")
    print(f"  SUMMARY — {total_movies:,} eligible movies")
    print(f"{'═' * 110}")
    header = (
        f"  {'Candidate':<45} {'Avg In':>7} {'Avg Out':>8} "
        f"{'Batch Cost':>12} {'Std Cost':>12} {'Quality':>8}"
    )
    print(header)
    print(f"  {'─' * 104}")

    # Sort by quality descending (candidates without scores go last)
    sorted_candidates = sorted(
        candidates,
        key=lambda c: (c.avg_quality is not None, c.avg_quality or 0),
        reverse=True,
    )

    for c in sorted_candidates:
        avg_cost = _weighted_avg_cost(c)
        avg_in, avg_out = _weighted_avg_tokens(c)
        batch = avg_cost * total_movies * 0.5
        standard = avg_cost * total_movies
        q = f"{c.avg_quality:.2f}" if c.avg_quality else "N/A"
        print(
            f"  {c.candidate_name:<45} {avg_in:>7.0f} {avg_out:>8.0f} "
            f"${batch:>10,.2f} ${standard:>10,.2f} {q:>8}"
        )

    print(f"{'═' * 110}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Estimate metadata generation cost across the full corpus."
    )
    parser.add_argument(
        "generation_type",
        help="Metadata type (e.g. viewer_experience, reception, narrative_techniques)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show per-bucket breakdown for each candidate",
    )
    args = parser.parse_args()

    generation_type = args.generation_type

    # Load bucket definitions and build ID mapping
    buckets_data = load_bucket_definitions(generation_type)
    id_to_bucket = build_id_to_bucket_map(buckets_data)

    # Collect token usage and quality scores
    token_usage, models = collect_token_usage(generation_type, id_to_bucket)
    quality_scores = collect_quality_scores(generation_type)

    if not token_usage:
        print(f"No evaluation data found for '{generation_type}'.")
        print(f"Looked in: {EVAL_DATA_DIR}/{generation_type}_*.json")
        return

    # Build stats
    candidates = build_candidate_stats(token_usage, models, quality_scores)

    # Get corpus size for this specific metadata type
    total_movies = get_eligible_movie_count(generation_type)
    print(f"\nGeneration type: {generation_type}")
    print(f"Eligible movies for {generation_type}: {total_movies:,}")
    print(f"Candidates evaluated: {len(candidates)}")
    print(f"Buckets: {len(buckets_data.get('buckets', {}))}")

    # Detailed per-candidate tables (optional)
    if args.detailed:
        for c in candidates:
            print_detailed_table(c, total_movies)

    # Summary comparison
    print_summary_table(candidates, total_movies)


if __name__ == "__main__":
    main()
