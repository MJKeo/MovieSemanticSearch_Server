"""
Sample movies near candidate quality-score thresholds for manual review.

For each candidate threshold, selects the 15 closest movies immediately
below and 15 immediately above.  Fetches full TMDB (tracker DB) and IMDB
(per-movie JSON) data for each sampled movie and writes a single JSON
file for manual inspection.

Output: ingestion_data/threshold_candidate_samples.json

Usage:
    python -m movie_ingestion.imdb_quality_scoring.sample_threshold_candidates
"""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import orjson

from movie_ingestion.scoring_utils import unpack_provider_keys
from movie_ingestion.tracker import INGESTION_DATA_DIR, init_db

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# The 4 candidate threshold scores identified from survival-curve analysis.
CANDIDATE_SCORES: list[float] = [0.27, 0.38, 0.43, 0.54]

# Number of movies to sample on each side of the candidate threshold.
SAMPLES_PER_SIDE: int = 15

# Output path for the JSON file.
OUTPUT_PATH: Path = INGESTION_DATA_DIR / "threshold_candidate_samples.json"

# Directory containing per-movie IMDB JSON files ({tmdb_id}.json).
_IMDB_DIR: Path = INGESTION_DATA_DIR / "imdb"

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


def _fetch_scored_movies(db: sqlite3.Connection) -> list[tuple[int, float]]:
    """Fetch all movies with a stage_5_quality_score, sorted by score.

    Returns a list of (tmdb_id, score) tuples in ascending score order.
    """
    rows = db.execute(
        """
        SELECT tmdb_id, stage_5_quality_score
        FROM movie_progress
        WHERE stage_5_quality_score IS NOT NULL
        ORDER BY stage_5_quality_score ASC
        """
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def _select_nearest(
    movies: list[tuple[int, float]],
    candidate: float,
    n: int,
) -> list[tuple[int, float]]:
    """Select the n closest movies below and n closest above the candidate.

    Movies are already sorted by score ascending, so we partition into
    below/above and take the n nearest from each side.  Returns a combined
    list sorted by score ascending (below first, then above).
    """
    # Partition: below includes movies with score < candidate,
    # above includes movies with score >= candidate.
    below: list[tuple[int, float]] = []
    above: list[tuple[int, float]] = []
    for tmdb_id, score in movies:
        if score < candidate:
            below.append((tmdb_id, score))
        else:
            above.append((tmdb_id, score))

    # Take the n closest from each side.  Since movies are sorted ascending:
    # - closest below = last n elements of `below`
    # - closest above = first n elements of `above`
    nearest_below = below[-n:] if len(below) >= n else below
    nearest_above = above[:n] if len(above) >= n else above

    return nearest_below + nearest_above


def _load_all_tmdb_data(
    db: sqlite3.Connection, tmdb_ids: set[int]
) -> dict[int, dict]:
    """Load all tmdb_data columns for the given tmdb_ids.

    Uses SELECT * and cursor.description to build dicts dynamically,
    so all columns are included without hardcoding the schema.  The
    watch_provider_keys BLOB is unpacked to a list of ints for JSON
    serializability.
    """
    if not tmdb_ids:
        return {}

    # SQLite doesn't support array binding — use a parameterized IN clause.
    placeholders = ",".join("?" for _ in tmdb_ids)
    cursor = db.execute(
        f"SELECT * FROM tmdb_data WHERE tmdb_id IN ({placeholders})",  # noqa: S608
        list(tmdb_ids),
    )
    columns = [desc[0] for desc in cursor.description]

    result: dict[int, dict] = {}
    for row in cursor:
        row_dict = dict(zip(columns, row))

        # Unpack the BLOB column to a JSON-serializable list of provider IDs.
        row_dict["watch_provider_keys"] = unpack_provider_keys(
            row_dict.get("watch_provider_keys")
        )

        result[row_dict["tmdb_id"]] = row_dict
    return result


def _load_one_imdb_json(path: Path) -> tuple[int, dict] | None:
    """Load a single IMDB JSON file.  Returns (tmdb_id, data) or None."""
    try:
        tmdb_id = int(path.stem)
    except ValueError:
        return None
    try:
        with open(path, "rb") as f:
            return tmdb_id, orjson.loads(f.read())
    except FileNotFoundError:
        return None


def _load_all_imdb_data(tmdb_ids: set[int]) -> dict[int, dict]:
    """Load IMDB JSON files for the given tmdb_ids using parallel I/O."""
    paths = [_IMDB_DIR / f"{tid}.json" for tid in tmdb_ids]

    result: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        for pair in pool.map(_load_one_imdb_json, paths):
            if pair is not None:
                result[pair[0]] = pair[1]
    return result


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------


def _build_output(
    candidates: dict[float, list[tuple[int, float]]],
    tmdb_data: dict[int, dict],
    imdb_data: dict[int, dict],
) -> dict:
    """Assemble the final JSON structure.

    For each candidate threshold, produces a list of single-key dicts
    mapping tmdb_id → {quality_score, tmdb_data, imdb_data}, sorted by
    quality_score ascending.
    """
    output: dict[str, list] = {}

    for candidate_score, movies in candidates.items():
        key = str(candidate_score)
        entries: list[dict] = []

        for tmdb_id, score in movies:
            entry = {
                str(tmdb_id): {
                    "quality_score": round(score, 6),
                    "tmdb_data": tmdb_data.get(tmdb_id, {}),
                    "imdb_data": imdb_data.get(tmdb_id, {}),
                }
            }
            entries.append(entry)

        output[key] = entries

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    db = init_db()

    try:
        # 1. Fetch all scored movies.
        print("Fetching scored movies from tracker database...")
        movies = _fetch_scored_movies(db)
        print(f"  {len(movies):,} movies with stage_5_quality_score")

        # 2. Select nearest movies for each candidate threshold.
        candidates: dict[float, list[tuple[int, float]]] = {}
        all_tmdb_ids: set[int] = set()

        for score in CANDIDATE_SCORES:
            nearest = _select_nearest(movies, score, SAMPLES_PER_SIDE)
            candidates[score] = nearest
            all_tmdb_ids.update(tid for tid, _ in nearest)

            # Report the score range for each candidate.
            scores = [s for _, s in nearest]
            below_count = sum(1 for s in scores if s < score)
            above_count = len(scores) - below_count
            print(
                f"  Candidate {score}: {below_count} below, {above_count} above"
                f"  | score range [{scores[0]:.4f}, {scores[-1]:.4f}]"
            )

        print(f"\n  {len(all_tmdb_ids)} unique movies to load data for")

        # 3. Bulk-load TMDB and IMDB data for all sampled movies.
        print("Loading TMDB data...")
        tmdb_data = _load_all_tmdb_data(db, all_tmdb_ids)
        print(f"  Loaded {len(tmdb_data):,} TMDB records")

    finally:
        db.close()

    print("Loading IMDB JSON files...")
    imdb_data = _load_all_imdb_data(all_tmdb_ids)
    print(f"  Loaded {len(imdb_data):,} IMDB records")

    # 4. Build and write the output JSON.
    output = _build_output(candidates, tmdb_data, imdb_data)

    # Use json (not orjson) for pretty-printed, human-readable output.
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
