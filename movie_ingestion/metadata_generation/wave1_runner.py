"""
Direct (non-batch) runner for Wave 1 metadata generation.

Generates plot_events and reception outputs individually for a set of
movies, storing results immediately to a dedicated SQLite table in the
tracker DB. Each generation type has its own function so they can be
run independently — plot_events is ready now, reception will be wired
in once its evaluation is complete.

The stored results serve two purposes:
    1. Wave 2 evaluation — generators like plot_analysis and
       viewer_experience need plot_synopsis and review_insights_brief
       as inputs. The fetch helper provides these from the table.
    2. Idempotent re-runs — movies that already have a result for a
       given type are skipped automatically.

Table schema (wave1_results in tracker.db):
    tmdb_id       INTEGER PRIMARY KEY
    plot_events   TEXT    -- JSON of PlotEventsOutput.model_dump(), nullable
    reception     TEXT    -- JSON of ReceptionOutput.model_dump(), nullable

Usage:
    python -m movie_ingestion.metadata_generation.wave1_runner
"""

import asyncio
import json
import sqlite3
from pathlib import Path

from movie_ingestion.metadata_generation.evaluations.shared import (
    EVALUATION_TEST_SET_TMDB_IDS,
    load_movie_input_data,
)
from movie_ingestion.metadata_generation.generators.plot_events import generate_plot_events
from movie_ingestion.metadata_generation.generators.reception import generate_reception
from movie_ingestion.metadata_generation.inputs import MovieInputData
from movie_ingestion.metadata_generation.pre_consolidation import (
    check_plot_events,
    check_reception,
)
from movie_ingestion.metadata_generation.schemas import PlotEventsOutput, ReceptionOutput

# Default tracker DB path (same as the rest of the ingestion pipeline)
_DEFAULT_DB_PATH = Path("ingestion_data/tracker.db")

# ---------------------------------------------------------------------------
# Table DDL
# ---------------------------------------------------------------------------

_CREATE_WAVE1_RESULTS_TABLE = """
    CREATE TABLE IF NOT EXISTS wave1_results (
        tmdb_id       INTEGER PRIMARY KEY,
        plot_events   TEXT,    -- JSON (PlotEventsOutput.model_dump())
        reception     TEXT     -- JSON (ReceptionOutput.model_dump())
    )
"""


def _open_connection(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode and FULL synchronous pragma."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.row_factory = sqlite3.Row
    return conn


def init_wave1_table(conn: sqlite3.Connection) -> None:
    """Create the wave1_results table if it doesn't exist."""
    conn.execute(_CREATE_WAVE1_RESULTS_TABLE)
    conn.commit()


# ---------------------------------------------------------------------------
# Generate + store: plot_events
# ---------------------------------------------------------------------------

async def generate_and_store_plot_events(
    movie_inputs: dict[int, MovieInputData],
    db_path: Path = _DEFAULT_DB_PATH,
    concurrency: int = 10,
) -> None:
    """Generate plot_events for movies that don't already have results.

    For each movie: checks eligibility via check_plot_events(), calls
    generate_plot_events(), and upserts the JSON result into the
    plot_events column of wave1_results. Movies that already have a
    non-NULL plot_events value are skipped entirely.

    Async tasks return result objects; all DB writes happen in a
    single synchronous batch after gather() completes.

    Args:
        movie_inputs: Dict of tmdb_id -> MovieInputData to process.
        db_path: Path to the tracker SQLite database.
        concurrency: Max concurrent LLM requests.
    """
    conn = _open_connection(db_path)
    init_wave1_table(conn)

    # Find movies that already have plot_events results
    existing = {
        row["tmdb_id"]
        for row in conn.execute(
            "SELECT tmdb_id FROM wave1_results WHERE plot_events IS NOT NULL"
        ).fetchall()
    }

    # Filter to movies needing generation, checking eligibility
    pending: list[tuple[int, MovieInputData]] = []
    for tmdb_id, movie in movie_inputs.items():
        if tmdb_id in existing:
            continue
        skip_reason = check_plot_events(movie)
        if skip_reason is not None:
            print(f"  SKIP plot_events for {tmdb_id} ({movie.title_with_year()}): {skip_reason}")
            continue
        pending.append((tmdb_id, movie))

    if not pending:
        print("plot_events: all movies already have results — nothing to do.")
        conn.close()
        return

    print(
        f"plot_events: generating {len(pending)} movies "
        f"(skipping {len(existing)} existing, concurrency={concurrency})..."
    )

    semaphore = asyncio.Semaphore(concurrency)

    async def _generate_one(
        tmdb_id: int, movie: MovieInputData,
    ) -> tuple[int, str, str, int, int] | tuple[int, None, str, None, None]:
        """Return (tmdb_id, result_json, title, in_tokens, out_tokens) or failure tuple."""
        async with semaphore:
            title = movie.title_with_year()
            try:
                output, token_usage = await generate_plot_events(movie)
                result_json = json.dumps(output.model_dump())
                return (tmdb_id, result_json, title, token_usage.input_tokens, token_usage.output_tokens)
            except Exception as e:
                print(f"  plot_events FAILED: {title} — {e}")
                return (tmdb_id, None, title, None, None)

    # Fan out async generation — no DB access inside tasks
    tasks = [_generate_one(tmdb_id, movie) for tmdb_id, movie in pending]
    results = await asyncio.gather(*tasks)

    # Batch-write all successful results synchronously
    completed = 0
    failed = 0
    for tmdb_id, result_json, title, in_tokens, out_tokens in results:
        if result_json is None:
            failed += 1
            continue
        conn.execute(
            """
            INSERT INTO wave1_results (tmdb_id, plot_events)
            VALUES (?, ?)
            ON CONFLICT(tmdb_id) DO UPDATE SET plot_events = excluded.plot_events
            """,
            (tmdb_id, result_json),
        )
        completed += 1
        print(f"  [{completed + failed}/{len(results)}] plot_events OK: {title} (in={in_tokens}, out={out_tokens})")

    conn.commit()
    conn.close()
    print(f"plot_events complete: {completed} succeeded, {failed} failed out of {len(results)}.")


# ---------------------------------------------------------------------------
# Generate + store: reception
# ---------------------------------------------------------------------------

async def generate_and_store_reception(
    movie_inputs: dict[int, MovieInputData],
    db_path: Path = _DEFAULT_DB_PATH,
    concurrency: int = 10,
) -> None:
    """Generate reception for movies that don't already have results.

    Same pattern as generate_and_store_plot_events but for the reception
    column. Uses check_reception() for eligibility and generate_reception()
    for LLM generation.

    Args:
        movie_inputs: Dict of tmdb_id -> MovieInputData to process.
        db_path: Path to the tracker SQLite database.
        concurrency: Max concurrent LLM requests.
    """
    conn = _open_connection(db_path)
    init_wave1_table(conn)

    # Find movies that already have reception results
    existing = {
        row["tmdb_id"]
        for row in conn.execute(
            "SELECT tmdb_id FROM wave1_results WHERE reception IS NOT NULL"
        ).fetchall()
    }

    # Filter to movies needing generation, checking eligibility
    pending: list[tuple[int, MovieInputData]] = []
    for tmdb_id, movie in movie_inputs.items():
        if tmdb_id in existing:
            continue
        skip_reason = check_reception(movie)
        if skip_reason is not None:
            print(f"  SKIP reception for {tmdb_id} ({movie.title_with_year()}): {skip_reason}")
            continue
        pending.append((tmdb_id, movie))

    if not pending:
        print("reception: all movies already have results — nothing to do.")
        conn.close()
        return

    print(
        f"reception: generating {len(pending)} movies "
        f"(skipping {len(existing)} existing, concurrency={concurrency})..."
    )

    semaphore = asyncio.Semaphore(concurrency)

    async def _generate_one(
        tmdb_id: int, movie: MovieInputData,
    ) -> tuple[int, str, str, int, int] | tuple[int, None, str, None, None]:
        """Return (tmdb_id, result_json, title, in_tokens, out_tokens) or failure tuple."""
        async with semaphore:
            title = movie.title_with_year()
            try:
                output, token_usage = await generate_reception(movie)
                result_json = json.dumps(output.model_dump())
                return (tmdb_id, result_json, title, token_usage.input_tokens, token_usage.output_tokens)
            except Exception as e:
                print(f"  reception FAILED: {title} — {e}")
                return (tmdb_id, None, title, None, None)

    # Fan out async generation — no DB access inside tasks
    tasks = [_generate_one(tmdb_id, movie) for tmdb_id, movie in pending]
    results = await asyncio.gather(*tasks)

    # Batch-write all successful results synchronously
    completed = 0
    failed = 0
    for tmdb_id, result_json, title, in_tokens, out_tokens in results:
        if result_json is None:
            failed += 1
            continue
        conn.execute(
            """
            INSERT INTO wave1_results (tmdb_id, reception)
            VALUES (?, ?)
            ON CONFLICT(tmdb_id) DO UPDATE SET reception = excluded.reception
            """,
            (tmdb_id, result_json),
        )
        completed += 1
        print(f"  [{completed + failed}/{len(results)}] reception OK: {title} (in={in_tokens}, out={out_tokens})")

    conn.commit()
    conn.close()
    print(f"reception complete: {completed} succeeded, {failed} failed out of {len(results)}.")


# ---------------------------------------------------------------------------
# Fetch helper — used by Wave 2 evaluation code
# ---------------------------------------------------------------------------

def get_wave1_results(
    tmdb_ids: list[int],
    db_path: Path = _DEFAULT_DB_PATH,
) -> dict[int, dict]:
    """Fetch stored Wave 1 results for a set of movies.

    Returns a dict mapping tmdb_id to a dict with deserialized Pydantic
    model instances (or None if the column is NULL).

    Example return value:
        {
            9377: {
                "plot_events": PlotEventsOutput(...),
                "reception": ReceptionOutput(...),
            },
            1584: {
                "plot_events": PlotEventsOutput(...),
                "reception": None,
            },
        }

    Movies not present in the table are omitted from the result dict.

    Args:
        tmdb_ids: List of TMDB IDs to fetch results for.
        db_path: Path to the tracker SQLite database.

    Returns:
        Dict of tmdb_id -> {"plot_events": ..., "reception": ...}.
    """
    conn = _open_connection(db_path)

    # Query all requested IDs in one shot
    placeholders = ",".join("?" for _ in tmdb_ids)
    rows = conn.execute(
        f"SELECT tmdb_id, plot_events, reception FROM wave1_results "
        f"WHERE tmdb_id IN ({placeholders})",
        tmdb_ids,
    ).fetchall()
    conn.close()

    results: dict[int, dict] = {}
    for row in rows:
        plot_events_output = None
        reception_output = None

        if row["plot_events"] is not None:
            plot_events_output = PlotEventsOutput.model_validate(
                json.loads(row["plot_events"])
            )
        if row["reception"] is not None:
            reception_output = ReceptionOutput.model_validate(
                json.loads(row["reception"])
            )

        results[row["tmdb_id"]] = {
            "plot_events": plot_events_output,
            "reception": reception_output,
        }

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    """Generate plot_events for all movies in the evaluation test set."""
    print(f"Loading movie input data for {len(EVALUATION_TEST_SET_TMDB_IDS)} movies...")
    movie_inputs = load_movie_input_data(EVALUATION_TEST_SET_TMDB_IDS)

    if not movie_inputs:
        print("No movies loaded — check that the ingestion pipeline has run.")
        return

    print(f"Loaded {len(movie_inputs)} movies.\n")
    await generate_and_store_plot_events(movie_inputs)


if __name__ == "__main__":
    asyncio.run(main())
