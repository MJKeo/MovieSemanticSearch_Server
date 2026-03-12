"""
Sample movies near candidate quality-score thresholds for manual review.

Movies are split into three non-overlapping groups (same classification as
plot_quality_scores.py), each with its own candidate thresholds from
survival-curve analysis.  For each threshold, selects the 10 closest movies
below/equal and 10 closest above.  Fetches full TMDB and IMDB data from the
tracker DB and writes one JSON file per group.

Output:
    ingestion_data/threshold_samples_has_providers.json
    ingestion_data/threshold_samples_no_providers_new.json
    ingestion_data/threshold_samples_no_providers_old.json

Usage:
    python -m movie_ingestion.imdb_quality_scoring.sample_threshold_candidates
"""

import json
import sqlite3
from dataclasses import dataclass

from movie_ingestion.scoring_utils import (
    HAS_PROVIDERS_SQL,
    NO_PROVIDERS_SQL,
    THEATER_WINDOW_SQL_PARAM,
    unpack_provider_keys,
)
from movie_ingestion.tracker import INGESTION_DATA_DIR, deserialize_imdb_row, init_db

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupConfig:
    """Configuration for a single movie group's threshold sampling."""
    name: str
    thresholds: list[float]
    output_filename: str


GROUPS: list[GroupConfig] = [
    GroupConfig(
        name="has_providers",
        thresholds=[0.35, 0.486],
        output_filename="threshold_samples_has_providers.json",
    ),
    GroupConfig(
        name="no_providers_new",
        thresholds=[0.37, 0.44, 0.5],
        output_filename="threshold_samples_no_providers_new.json",
    ),
    GroupConfig(
        name="no_providers_old",
        thresholds=[0.6, 0.621, 0.654],
        output_filename="threshold_samples_no_providers_old.json",
    ),
]

# Number of movies to sample on each side of the candidate threshold.
SAMPLES_PER_SIDE: int = 10

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


def _fetch_grouped_movies(
    db: sqlite3.Connection,
) -> dict[str, list[tuple[int, float]]]:
    """Fetch all scored movies in one query, classified into groups.

    Returns a dict mapping group name → list of (tmdb_id, score) tuples,
    each sorted by score ascending.  Group classification uses the same
    logic as plot_quality_scores.py (provider status + release date).
    """
    # Single query with CASE expression to classify movies into groups.
    rows = db.execute(
        f"""
        SELECT
            mp.tmdb_id,
            mp.stage_5_quality_score,
            CASE
                WHEN {HAS_PROVIDERS_SQL} THEN 'has_providers'
                WHEN {NO_PROVIDERS_SQL}
                     AND td.release_date >= date('now', ?)
                     THEN 'no_providers_new'
                ELSE 'no_providers_old'
            END AS group_name
        FROM movie_progress mp
        JOIN tmdb_data td ON td.tmdb_id = mp.tmdb_id
        WHERE mp.stage_5_quality_score IS NOT NULL
        ORDER BY mp.stage_5_quality_score ASC
        """,
        (THEATER_WINDOW_SQL_PARAM,),
    ).fetchall()

    # Partition into groups.  Already sorted by score from the query.
    groups: dict[str, list[tuple[int, float]]] = {
        "has_providers": [],
        "no_providers_new": [],
        "no_providers_old": [],
    }
    for tmdb_id, score, group_name in rows:
        groups[group_name].append((tmdb_id, score))

    return groups


def _select_nearest(
    movies: list[tuple[int, float]],
    candidate: float,
    n: int,
) -> list[tuple[int, float]]:
    """Select the n closest movies at/below and n closest above the candidate.

    Movies are already sorted by score ascending.  Partition:
    - below/equal: score <= candidate  (take last n)
    - above:       score > candidate   (take first n)

    Returns a combined list sorted by score ascending.
    """
    below: list[tuple[int, float]] = []
    above: list[tuple[int, float]] = []
    for tmdb_id, score in movies:
        if score <= candidate:
            below.append((tmdb_id, score))
        else:
            above.append((tmdb_id, score))

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

    placeholders = ",".join("?" for _ in tmdb_ids)
    cursor = db.execute(
        f"SELECT * FROM tmdb_data WHERE tmdb_id IN ({placeholders})",  # noqa: S608
        list(tmdb_ids),
    )
    columns = [desc[0] for desc in cursor.description]

    result: dict[int, dict] = {}
    for row in cursor:
        row_dict = dict(zip(columns, row))
        row_dict["watch_provider_keys"] = unpack_provider_keys(
            row_dict.get("watch_provider_keys")
        )
        result[row_dict["tmdb_id"]] = row_dict
    return result


def _load_all_imdb_data(db: sqlite3.Connection, tmdb_ids: set[int]) -> dict[int, dict]:
    """Load IMDB data from the imdb_data SQLite table for the given tmdb_ids.

    Each row has individual columns for every IMDBScrapedMovie field.
    JSON TEXT columns (lists/objects) are deserialized back to Python
    types by deserialize_imdb_row().
    """
    if not tmdb_ids:
        return {}
    placeholders = ",".join("?" for _ in tmdb_ids)
    prev_factory = db.row_factory
    db.row_factory = sqlite3.Row
    rows = db.execute(
        f"SELECT * FROM imdb_data WHERE tmdb_id IN ({placeholders})",  # noqa: S608
        tuple(tmdb_ids),
    ).fetchall()
    db.row_factory = prev_factory
    return {row["tmdb_id"]: deserialize_imdb_row(row) for row in rows}


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------


def _build_output(
    candidates: dict[float, list[tuple[int, float]]],
    tmdb_data: dict[int, dict],
    imdb_data: dict[int, dict],
) -> dict:
    """Assemble the final JSON structure for a single group.

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
        # 1. Fetch all scored movies, classified into groups (single DB pass).
        print("Fetching scored movies from tracker database...")
        grouped_movies = _fetch_grouped_movies(db)
        for name, movies in grouped_movies.items():
            print(f"  {name}: {len(movies):,} movies")

        # 2. Select nearest movies for each group's thresholds.
        # Maps group_name → {threshold → [(tmdb_id, score), ...]}
        group_candidates: dict[str, dict[float, list[tuple[int, float]]]] = {}
        all_tmdb_ids: set[int] = set()

        for group in GROUPS:
            movies = grouped_movies[group.name]
            candidates: dict[float, list[tuple[int, float]]] = {}

            for threshold in group.thresholds:
                nearest = _select_nearest(movies, threshold, SAMPLES_PER_SIDE)
                candidates[threshold] = nearest
                all_tmdb_ids.update(tid for tid, _ in nearest)

                # Report the score range for each threshold.
                scores = [s for _, s in nearest]
                below_count = sum(1 for s in scores if s <= threshold)
                above_count = len(scores) - below_count
                print(
                    f"  [{group.name}] {threshold}: "
                    f"{below_count} below/equal, {above_count} above"
                    f"  | score range [{scores[0]:.4f}, {scores[-1]:.4f}]"
                )

            group_candidates[group.name] = candidates

        print(f"\n  {len(all_tmdb_ids)} unique movies to load data for")

        # 3. Bulk-load TMDB data for all sampled movies (single DB query).
        print("Loading TMDB data...")
        tmdb_data = _load_all_tmdb_data(db, all_tmdb_ids)
        print(f"  Loaded {len(tmdb_data):,} TMDB records")

        # 4. Bulk-load IMDB data from SQLite (single query).
        print("Loading IMDB data from SQLite...")
        imdb_data = _load_all_imdb_data(db, all_tmdb_ids)
        print(f"  Loaded {len(imdb_data):,} IMDB records")

    finally:
        db.close()

    # 5. Build and write one JSON file per group.
    for group in GROUPS:
        output = _build_output(
            group_candidates[group.name], tmdb_data, imdb_data
        )
        output_path = INGESTION_DATA_DIR / group.output_filename

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
