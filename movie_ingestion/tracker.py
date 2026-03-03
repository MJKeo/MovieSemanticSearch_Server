"""
SQLite checkpoint tracker for the movie ingestion pipeline.

Provides the shared infrastructure used by all 8 pipeline stages:
  - Schema initialization (movie_progress + filter_log tables)
  - The log_filter helper for recording filtered-out movies
  - JSON file I/O utilities (load_json, save_json)

The tracker database lives at ./ingestion_data/tracker.db and records
per-movie progress through the pipeline. If the process crashes at any
point, restarting picks up where it left off because each stage queries
the tracker for movies in the appropriate status.
"""

import json
import os
import sqlite3
from enum import StrEnum
from pathlib import Path

# ---------------------------------------------------------------------------
# Pipeline stages — used as the `stage` value in filter_log entries.
# StrEnum so the values are plain strings (compatible with SQLite text columns).
# ---------------------------------------------------------------------------


class PipelineStage(StrEnum):
    TMDB_EXPORT_FILTER = "tmdb_export_filter"
    TMDB_FETCH = "tmdb_fetch"
    TMDB_QUALITY_FUNNEL = "tmdb_quality_funnel"
    IMDB_SCRAPE = "imdb_scrape"
    ESSENTIAL_DATA_CHECK = "essential_data_check"
    LLM_PHASE1 = "llm_phase1"
    LLM_PHASE2 = "llm_phase2"
    EMBEDDING = "embedding"
    INGESTION = "ingestion"


# ---------------------------------------------------------------------------
# Paths — all ingestion data lives under this root, relative to the project.
# ---------------------------------------------------------------------------

INGESTION_DATA_DIR = Path("./ingestion_data")
TRACKER_DB_PATH = INGESTION_DATA_DIR / "tracker.db"
TMDB_DATA_DIR = INGESTION_DATA_DIR / "tmdb"

# ---------------------------------------------------------------------------
# Schema — exact tables and indexes from the pipeline architecture guide.
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
-- Main progress tracker: one row per movie, tracks status through the pipeline.
CREATE TABLE IF NOT EXISTS movie_progress (
    tmdb_id          INTEGER PRIMARY KEY,
    imdb_id          TEXT,
    status           TEXT NOT NULL DEFAULT 'pending',
    -- Statuses (in order):
    --   pending → tmdb_fetched → quality_passed → imdb_scraped →
    --   phase1_complete → phase2_complete → embedded → ingested
    --
    -- Terminal statuses:
    --   filtered_out        (missing essential data, scrape failures, etc.)
    --   below_quality_cutoff (didn't make the top ~100K in the TMDB quality funnel)
    quality_score    REAL,
    batch1_custom_id TEXT,
    batch2_custom_id TEXT,
    updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Every movie that gets dropped, with why.
-- title and year come from TMDB and are populated for any movie that made it
-- past Stage 1 (the initial adult/video filter). For movies filtered at
-- Stage 1, these will be NULL since we only have the tmdb_id at that point.
CREATE TABLE IF NOT EXISTS filter_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    tmdb_id    INTEGER NOT NULL,
    title      TEXT,
    year       INTEGER,
    stage      TEXT NOT NULL,
    -- Stages: 'tmdb_export_filter', 'tmdb_fetch', 'tmdb_quality_funnel',
    --         'imdb_scrape', 'essential_data_check', 'llm_phase1',
    --         'llm_phase2', 'embedding', 'ingestion'
    reason     TEXT NOT NULL,
    details    TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_progress_status ON movie_progress(status);
CREATE INDEX IF NOT EXISTS idx_filter_log_stage ON filter_log(stage);
CREATE INDEX IF NOT EXISTS idx_filter_log_tmdb  ON filter_log(tmdb_id);
"""


def init_db() -> sqlite3.Connection:
    """
    Initialize the SQLite tracker database and return an open connection.

    Creates the ingestion_data directory and tracker.db if they do not exist.
    Creates the movie_progress and filter_log tables with all indexes.
    Safe to call multiple times — all DDL uses IF NOT EXISTS.

    Returns:
        An open sqlite3.Connection with WAL mode enabled.
    """
    INGESTION_DATA_DIR.mkdir(parents=True, exist_ok=True)

    db = sqlite3.connect(str(TRACKER_DB_PATH))

    # WAL mode gives better concurrent read performance (useful when
    # inspecting the DB in another terminal while a stage is running).
    db.execute("PRAGMA journal_mode=WAL")

    db.executescript(_SCHEMA_SQL)
    db.commit()

    return db


# ---------------------------------------------------------------------------
# log_filter — the single helper for recording filtered-out movies.
# ---------------------------------------------------------------------------


def log_filter(
    db: sqlite3.Connection,
    tmdb_id: int,
    stage: str,
    reason: str,
    details: str | None = None,
) -> None:
    """
    Record a filtered-out movie in filter_log and update movie_progress status.

    Automatically attaches title and year from the TMDB detail file if it
    exists on disk (i.e., the movie made it past Stage 1 and was fetched in
    Stage 2). For Stage 1 filtering, title and year will be NULL since no
    TMDB detail file exists yet.

    The UPDATE to movie_progress is intentionally tolerant of missing rows —
    Stage 1 filtered movies are never inserted into movie_progress, so the
    UPDATE matches zero rows, which is correct behavior.

    Args:
        db:      Open SQLite connection.
        tmdb_id: TMDB movie ID being filtered.
        stage:   Pipeline stage name (e.g., 'tmdb_export_filter', 'tmdb_fetch').
        reason:  Human-readable filter reason (e.g., 'adult', 'missing_imdb_id').
        details: Optional JSON string with additional context.
    """
    title = None
    year = None

    # Pull title/year from TMDB data if available (exists for anything past Stage 1)
    tmdb_path = TMDB_DATA_DIR / f"{tmdb_id}.json"
    if tmdb_path.exists():
        tmdb_data = load_json(tmdb_path)
        title = tmdb_data.get("title")
        release_date = tmdb_data.get("release_date", "")
        if release_date and len(release_date) >= 4:
            try:
                year = int(release_date[:4])
            except ValueError:
                pass

    db.execute(
        """INSERT INTO filter_log (tmdb_id, title, year, stage, reason, details)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (tmdb_id, title, year, stage, reason, details),
    )
    db.execute(
        "UPDATE movie_progress SET status = 'filtered_out', updated_at = CURRENT_TIMESTAMP WHERE tmdb_id = ?",
        (tmdb_id,),
    )


# ---------------------------------------------------------------------------
# JSON file I/O — used by all stages for per-movie data files.
# ---------------------------------------------------------------------------


def load_json(path: str | Path) -> dict:
    """Load and parse a JSON file from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: dict | list) -> None:
    """
    Write a JSON file atomically (write to .tmp then rename).

    The write-then-rename pattern ensures that a crash mid-write only
    corrupts the .tmp file, never the real output file. This matters for
    Stages 2+ where individual movie files must survive interruptions.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    tmp_path.rename(path)
