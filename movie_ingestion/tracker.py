"""
SQLite checkpoint tracker for the movie ingestion pipeline.

Provides the shared infrastructure used by all 9 pipeline stages:
  - Schema initialization (movie_progress, filter_log, tmdb_data, imdb_data tables)
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

import orjson

# ---------------------------------------------------------------------------
# Pipeline stages — used as the `stage` value in filter_log entries.
# StrEnum so the values are plain strings (compatible with SQLite text columns).
# ---------------------------------------------------------------------------


class PipelineStage(StrEnum):
    TMDB_EXPORT_FILTER = "tmdb_export_filter"
    TMDB_FETCH = "tmdb_fetch"
    TMDB_QUALITY_FUNNEL = "tmdb_quality_funnel"
    IMDB_SCRAPE = "imdb_scrape"
    IMDB_QUALITY_FUNNEL = "imdb_quality_funnel"
    LLM_PHASE1 = "llm_phase1"
    LLM_PHASE2 = "llm_phase2"
    EMBEDDING = "embedding"
    INGESTION = "ingestion"


# ---------------------------------------------------------------------------
# Movie progress statuses — the `status` column in movie_progress.
# StrEnum so values are plain strings (compatible with SQLite text columns
# and usable directly in parameterised queries).
# ---------------------------------------------------------------------------


class MovieStatus(StrEnum):
    """Movie processing status in movie_progress.status.

    Progression:
      PENDING → TMDB_FETCHED → TMDB_QUALITY_CALCULATED →
      TMDB_QUALITY_PASSED → IMDB_SCRAPED → IMDB_QUALITY_CALCULATED →
      IMDB_QUALITY_PASSED → METADATA_GENERATED →
      EMBEDDED → INGESTED

    Terminal:
      FILTERED_OUT (with variable reason strings in filter_log)

    Retryable:
      INGESTION_FAILED (failure details in ingestion_failures table)
    """
    PENDING = "pending"
    TMDB_FETCHED = "tmdb_fetched"
    TMDB_QUALITY_CALCULATED = "tmdb_quality_calculated"
    TMDB_QUALITY_PASSED = "tmdb_quality_passed"
    IMDB_SCRAPED = "imdb_scraped"
    IMDB_QUALITY_CALCULATED = "imdb_quality_calculated"
    IMDB_QUALITY_PASSED = "imdb_quality_passed"
    METADATA_GENERATED = "metadata_generated"
    EMBEDDED = "embedded"
    INGESTED = "ingested"
    # Terminal status — reason details are in filter_log.reason
    FILTERED_OUT = "filtered_out"
    # Retryable failure — movie failed during ingestion, eligible for retry.
    # Failure details are in ingestion_failures table.
    INGESTION_FAILED = "ingestion_failed"


# ---------------------------------------------------------------------------
# Paths — all ingestion data lives under this root, relative to the project.
# ---------------------------------------------------------------------------

INGESTION_DATA_DIR = Path("./ingestion_data")
TRACKER_DB_PATH = INGESTION_DATA_DIR / "tracker.db"

# ---------------------------------------------------------------------------
# Schema — exact tables and indexes from the pipeline architecture guide.
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
-- Main progress tracker: one row per movie, tracks status through the pipeline.
CREATE TABLE IF NOT EXISTS movie_progress (
    tmdb_id          INTEGER PRIMARY KEY,
    imdb_id          TEXT,
    status           TEXT NOT NULL DEFAULT 'pending',
    -- Statuses (in order) — defined in MovieStatus enum:
    --   pending → tmdb_fetched → tmdb_quality_calculated →
    --   tmdb_quality_passed → imdb_scraped → imdb_quality_calculated →
    --   imdb_quality_passed → metadata_generated →
    --   embedded → ingested
    --
    -- Terminal statuses:
    --   filtered_out        (all filtering — reason details in filter_log.reason)
    -- Retryable statuses:
    --   ingestion_failed    (failure details in ingestion_failures table)
    stage_3_quality_score REAL,
    stage_5_quality_score REAL,
    updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Every movie that gets dropped, with why.
-- Join on tmdb_data to get title/year when needed for display.
CREATE TABLE IF NOT EXISTS filter_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    tmdb_id    INTEGER NOT NULL,
    stage      TEXT NOT NULL,
    -- Stages: 'tmdb_export_filter', 'tmdb_fetch', 'tmdb_quality_funnel',
    --         'imdb_scrape', 'imdb_quality_funnel', 'llm_phase1',
    --         'llm_phase2', 'embedding', 'ingestion'
    reason     TEXT NOT NULL,
    details    TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_progress_status ON movie_progress(status);
CREATE INDEX IF NOT EXISTS idx_filter_log_stage ON filter_log(stage);
CREATE INDEX IF NOT EXISTS idx_filter_log_tmdb  ON filter_log(tmdb_id);

-- TMDB detail data: one row per movie fetched in Stage 2.
-- Stores the subset of fields needed for Stage 3 quality filtering and
-- downstream pipeline stages. Full response data is NOT persisted — only
-- the fields actually consumed by later stages.
CREATE TABLE IF NOT EXISTS tmdb_data (
    tmdb_id             INTEGER PRIMARY KEY,
    imdb_id             TEXT,
    title               TEXT,
    release_date        TEXT,
    duration            INTEGER,
    poster_url          TEXT,
    watch_provider_keys BLOB,
    vote_count          INTEGER DEFAULT 0,
    popularity          REAL DEFAULT 0.0,
    vote_average        REAL DEFAULT 0.0,
    overview_length     INTEGER DEFAULT 0,
    genre_count         INTEGER DEFAULT 0,
    has_revenue         INTEGER DEFAULT 0,
    has_budget          INTEGER DEFAULT 0,
    has_production_companies INTEGER DEFAULT 0,
    has_production_countries INTEGER DEFAULT 0,
    has_keywords        INTEGER DEFAULT 0,
    has_cast_and_crew   INTEGER DEFAULT 0,
    budget              INTEGER DEFAULT 0,
    maturity_rating     TEXT,
    reviews             TEXT,
    collection_name     TEXT,
    revenue             INTEGER DEFAULT 0
);

-- IMDB scraped data: one row per movie scraped in Stage 4.
-- Each IMDBScrapedMovie field gets its own column. List and object fields
-- are stored as JSON TEXT arrays (SQLite has no native array type).
CREATE TABLE IF NOT EXISTS imdb_data (
    tmdb_id              INTEGER PRIMARY KEY,
    -- IMDB title type (e.g. "movie", "videoGame")
    imdb_title_type      TEXT,
    -- Scalars from main page
    original_title       TEXT,
    maturity_rating      TEXT,
    overview             TEXT,
    imdb_rating          REAL,
    imdb_vote_count      INTEGER DEFAULT 0,
    metacritic_rating    REAL,
    reception_summary    TEXT,
    budget               INTEGER,
    -- List-of-strings (JSON arrays)
    overall_keywords     TEXT,    -- JSON array of strings
    genres               TEXT,    -- JSON array of strings
    countries_of_origin  TEXT,    -- JSON array of strings
    production_companies TEXT,    -- JSON array of strings
    filming_locations    TEXT,    -- JSON array of strings
    languages            TEXT,    -- JSON array of strings
    synopses             TEXT,    -- JSON array of strings
    plot_summaries       TEXT,    -- JSON array of strings
    plot_keywords        TEXT,    -- JSON array of strings
    maturity_reasoning   TEXT,    -- JSON array of strings
    directors            TEXT,    -- JSON array of strings
    writers              TEXT,    -- JSON array of strings
    actors               TEXT,    -- JSON array of strings
    characters           TEXT,    -- JSON array of strings
    producers            TEXT,    -- JSON array of strings
    composers            TEXT,    -- JSON array of strings
    -- List-of-objects (JSON arrays of objects)
    review_themes        TEXT,    -- JSON array of {name, sentiment}
    parental_guide_items TEXT,    -- JSON array of {category, severity}
    featured_reviews     TEXT,    -- JSON array of {summary, text}
    -- Awards (JSON array of {ceremony, category, outcome, year})
    awards               TEXT,
    -- Box office (whole USD dollars, NULL when data unavailable)
    box_office_worldwide INTEGER  -- inclusive of domestic
);

-- Batch IDs per movie per metadata generation type.
-- Each column holds the OpenAI Batch API batch_id for that generation type,
-- or NULL if no batch request has been submitted for that movie/type.
CREATE TABLE IF NOT EXISTS metadata_batch_ids (
    tmdb_id                        INTEGER PRIMARY KEY,
    plot_events_batch_id           TEXT,
    reception_batch_id             TEXT,
    plot_analysis_batch_id         TEXT,
    viewer_experience_batch_id     TEXT,
    watch_context_batch_id         TEXT,
    narrative_techniques_batch_id  TEXT,
    production_keywords_batch_id   TEXT,
    source_of_inspiration_batch_id TEXT,
    source_material_v2_batch_id    TEXT
);

-- Individual request failures within a batch, for tracking and retry.
-- A movie with eligible_for_X = 1 and X IS NULL in generated_metadata
-- plus a row here clearly failed; without a row here it hasn't been attempted.
CREATE TABLE IF NOT EXISTS generation_failures (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    tmdb_id        INTEGER NOT NULL,
    metadata_type  TEXT NOT NULL,
    error_message  TEXT NOT NULL,
    batch_id       TEXT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_gen_failures_type ON generation_failures(metadata_type);
CREATE INDEX IF NOT EXISTS idx_gen_failures_batch ON generation_failures(batch_id);

-- Per-movie ingestion failures, for diagnostics and retry tracking.
-- A single movie may have multiple rows if it failed at more than one step
-- (e.g. both movie card and lexical ingestion in the same run).
-- The error_message includes the step prefix (e.g. "Postgres movie card: <error>").
CREATE TABLE IF NOT EXISTS ingestion_failures (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    tmdb_id        INTEGER NOT NULL,
    error_message  TEXT NOT NULL,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_ingest_failures_tmdb ON ingestion_failures(tmdb_id);

-- Generated metadata results and eligibility flags, one row per movie.
-- JSON columns store the full LLM output (parsed on read). Eligibility
-- flags are NULL until evaluated, then set to 1 (eligible) or 0 (ineligible)
-- based on input data quality checks in pre_consolidation.py.
CREATE TABLE IF NOT EXISTS generated_metadata (
    tmdb_id                              INTEGER PRIMARY KEY,
    -- Generated results (JSON text, NULL until generated)
    plot_events                          TEXT,
    reception                            TEXT,
    plot_analysis                        TEXT,
    viewer_experience                    TEXT,
    watch_context                        TEXT,
    narrative_techniques                 TEXT,
    production_keywords                  TEXT,
    source_of_inspiration                TEXT,
    source_material_v2                   TEXT,
    -- Eligibility flags (NULL = not evaluated, 1 = eligible, 0 = ineligible)
    eligible_for_plot_events             INTEGER,
    eligible_for_reception               INTEGER,
    eligible_for_plot_analysis           INTEGER,
    eligible_for_viewer_experience       INTEGER,
    eligible_for_watch_context           INTEGER,
    eligible_for_narrative_techniques    INTEGER,
    eligible_for_production_keywords     INTEGER,
    eligible_for_source_of_inspiration   INTEGER,
    eligible_for_source_material_v2      INTEGER
);
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
    # FULL sync ensures every commit is fsynced to disk, preventing
    # corruption if the process is killed or the system crashes.
    db.execute("PRAGMA synchronous=FULL")

    db.executescript(_SCHEMA_SQL)

    # Migrate existing databases: add columns introduced after the initial
    # schema.  Each ALTER TABLE is wrapped in a try/except because SQLite
    # raises OperationalError if the column already exists (i.e. the DB was
    # created with the updated CREATE TABLE statement above).
    _MIGRATIONS = [
        "ALTER TABLE tmdb_data ADD COLUMN budget INTEGER DEFAULT 0",
        "ALTER TABLE tmdb_data ADD COLUMN maturity_rating TEXT",
        "ALTER TABLE tmdb_data ADD COLUMN reviews TEXT",
        "ALTER TABLE filter_log DROP COLUMN title",
        "ALTER TABLE filter_log DROP COLUMN year",
        # Rename quality_score → stage_3_quality_score and add stage_5_quality_score
        # for databases created before the column rename.
        "ALTER TABLE movie_progress RENAME COLUMN quality_score TO stage_3_quality_score",
        "ALTER TABLE movie_progress ADD COLUMN stage_5_quality_score REAL",
        # v2 scorer removed the essential_data_passed status; reset any movies
        # at that status back to imdb_scraped so they can be re-scored.
        "UPDATE movie_progress SET status = 'imdb_scraped', stage_5_quality_score = NULL WHERE status = 'essential_data_passed'",
        # Add imdb_title_type column for existing databases.
        "ALTER TABLE imdb_data ADD COLUMN imdb_title_type TEXT",
        # Remove batch custom ID columns — replaced by metadata_batch_ids table.
        "ALTER TABLE movie_progress DROP COLUMN batch1_custom_id",
        "ALTER TABLE movie_progress DROP COLUMN batch2_custom_id",
        # Remove wave1_results table — replaced by generated_metadata table.
        "DROP TABLE IF EXISTS wave1_results",
        # Add batch_id column to generation_failures for tracing failures back to batches.
        "ALTER TABLE generation_failures ADD COLUMN batch_id TEXT",
        # Collapse phase1_complete / phase2_complete into metadata_generated.
        "UPDATE movie_progress SET status = 'metadata_generated' WHERE status IN ('phase1_complete', 'phase2_complete')",
        # Add collection_name and revenue columns to tmdb_data.
        "ALTER TABLE tmdb_data ADD COLUMN collection_name TEXT",
        "ALTER TABLE tmdb_data ADD COLUMN revenue INTEGER DEFAULT 0",
        # Add awards and box office columns to imdb_data.
        "ALTER TABLE imdb_data ADD COLUMN awards TEXT",
        "ALTER TABLE imdb_data ADD COLUMN box_office_worldwide INTEGER",
        # Add source_material_v2 columns for enum-constrained re-generation.
        "ALTER TABLE generated_metadata ADD COLUMN source_material_v2 TEXT",
        "ALTER TABLE generated_metadata ADD COLUMN eligible_for_source_material_v2 INTEGER",
        "ALTER TABLE metadata_batch_ids ADD COLUMN source_material_v2_batch_id TEXT",
        # Add concept_tags columns for binary concept tag classification.
        "ALTER TABLE generated_metadata ADD COLUMN concept_tags TEXT",
        "ALTER TABLE generated_metadata ADD COLUMN eligible_for_concept_tags INTEGER",
        "ALTER TABLE metadata_batch_ids ADD COLUMN concept_tags_batch_id TEXT",
    ]
    for stmt in _MIGRATIONS:
        try:
            db.execute(stmt)
        except sqlite3.OperationalError:
            pass  # Column already exists — nothing to do.

    # Migrate imdb_data from single JSON blob to individual columns.
    # Detects the old schema by checking for the `data` column.
    _migrate_imdb_data_blob_to_columns(db)

    db.commit()

    return db


# ---------------------------------------------------------------------------
# imdb_data column definitions — the single source of truth for column
# order, used by serialize_imdb_movie(), deserialize_imdb_row(), and the
# blob→columns migration.
# ---------------------------------------------------------------------------

# Ordered list of imdb_data columns (excluding tmdb_id primary key).
# This order MUST match the CREATE TABLE definition above.
IMDB_DATA_COLUMNS: tuple[str, ...] = (
    "imdb_title_type",
    "original_title", "maturity_rating", "overview",
    "imdb_rating", "imdb_vote_count", "metacritic_rating",
    "reception_summary", "budget",
    "overall_keywords", "genres", "countries_of_origin",
    "production_companies", "filming_locations", "languages",
    "synopses", "plot_summaries", "plot_keywords", "maturity_reasoning",
    "directors", "writers", "actors", "characters", "producers", "composers",
    "review_themes", "parental_guide_items", "featured_reviews",
    "awards", "box_office_worldwide",
)

# Columns that store JSON arrays/objects and need deserialization on read.
IMDB_JSON_COLUMNS: frozenset[str] = frozenset({
    "overall_keywords", "genres", "countries_of_origin",
    "production_companies", "filming_locations", "languages",
    "synopses", "plot_summaries", "plot_keywords", "maturity_reasoning",
    "directors", "writers", "actors", "characters", "producers", "composers",
    "review_themes", "parental_guide_items", "featured_reviews",
    "awards",
})

# Pre-built SQL fragments for INSERT, derived from IMDB_DATA_COLUMNS.
_IMDB_INSERT_COLS = ", ".join(["tmdb_id"] + list(IMDB_DATA_COLUMNS))
_IMDB_INSERT_PLACEHOLDERS = ", ".join("?" * (1 + len(IMDB_DATA_COLUMNS)))
IMDB_INSERT_SQL = (
    f"INSERT OR REPLACE INTO imdb_data ({_IMDB_INSERT_COLS}) "
    f"VALUES ({_IMDB_INSERT_PLACEHOLDERS})"
)


def serialize_imdb_movie(tmdb_id: int, data: dict) -> tuple:
    """Convert an IMDBScrapedMovie dict to a tuple for SQL INSERT.

    Scalar fields are passed through directly. List/object fields are
    serialized to compact JSON strings (or None if the list is empty).
    Empty lists become NULL so IS NOT NULL works for presence checks.

    Args:
        tmdb_id: The TMDB movie ID (primary key).
        data:    The model_dump(mode="json") dict from IMDBScrapedMovie.

    Returns:
        A tuple of values in IMDB_INSERT_SQL column order.
    """
    values: list = [tmdb_id]
    for col in IMDB_DATA_COLUMNS:
        val = data.get(col)
        if col in IMDB_JSON_COLUMNS:
            # Serialize lists/objects to JSON TEXT, or NULL if empty.
            if val:
                values.append(orjson.dumps(val).decode())
            else:
                values.append(None)
        else:
            values.append(val)
    return tuple(values)


def deserialize_imdb_row(row: sqlite3.Row) -> dict:
    """Convert a raw imdb_data SQLite row to a dict with JSON columns parsed.

    Scalar columns are returned as-is. JSON TEXT columns are deserialized
    back to Python lists/dicts so consumers see the same dict shape as
    the original IMDBScrapedMovie model_dump().

    Args:
        row: A sqlite3.Row from a SELECT on imdb_data.

    Returns:
        A dict with all fields, JSON columns deserialized to lists/dicts.
    """
    d = dict(row)
    for col in IMDB_JSON_COLUMNS:
        val = d.get(col)
        d[col] = orjson.loads(val) if val is not None else []
    return d


# ---------------------------------------------------------------------------
# imdb_data migration: single JSON blob → individual columns
# ---------------------------------------------------------------------------


def _migrate_imdb_data_blob_to_columns(db: sqlite3.Connection) -> None:
    """Migrate imdb_data from the old single-blob schema to individual columns.

    Detects the old schema by checking for a `data` column via PRAGMA
    table_info. If found, uses json_extract() to expand the blob into the
    new column layout. Safe to call multiple times — no-ops if the migration
    has already been applied.
    """
    columns_info = db.execute("PRAGMA table_info(imdb_data)").fetchall()
    col_names = {row[1] for row in columns_info}

    # If there's no `data` column, the table already uses the new schema
    # (or was just created fresh with individual columns).
    if "data" not in col_names:
        return

    print("  Migrating imdb_data from JSON blob to individual columns...")

    # Build the new table with the expanded schema, then copy data over
    # using json_extract() to pull each field from the blob.
    # Using a temp table + rename avoids ALTER TABLE limitations in SQLite.
    col_defs = """
        tmdb_id              INTEGER PRIMARY KEY,
        original_title       TEXT,
        maturity_rating      TEXT,
        overview             TEXT,
        imdb_rating          REAL,
        imdb_vote_count      INTEGER DEFAULT 0,
        metacritic_rating    REAL,
        reception_summary    TEXT,
        budget               INTEGER,
        overall_keywords     TEXT,
        genres               TEXT,
        countries_of_origin  TEXT,
        production_companies TEXT,
        filming_locations    TEXT,
        languages            TEXT,
        synopses             TEXT,
        plot_summaries       TEXT,
        plot_keywords        TEXT,
        maturity_reasoning   TEXT,
        directors            TEXT,
        writers              TEXT,
        actors               TEXT,
        characters           TEXT,
        producers            TEXT,
        composers            TEXT,
        review_themes        TEXT,
        parental_guide_items TEXT,
        featured_reviews     TEXT
    """

    db.execute(f"CREATE TABLE imdb_data_new ({col_defs})")

    # Build json_extract expressions for each column.
    # Scalar fields: json_extract returns the native value directly.
    # Array/object fields: json_extract returns the JSON sub-tree as TEXT,
    # which is exactly what we want to store.
    extract_exprs = ["tmdb_id"]
    for col in IMDB_DATA_COLUMNS:
        extract_exprs.append(f"json_extract(data, '$.{col}')")

    insert_cols = ", ".join(["tmdb_id"] + list(IMDB_DATA_COLUMNS))
    select_exprs = ", ".join(extract_exprs)

    db.execute(
        f"INSERT INTO imdb_data_new ({insert_cols}) "
        f"SELECT {select_exprs} FROM imdb_data"
    )

    row_count = db.execute("SELECT COUNT(*) FROM imdb_data_new").fetchone()[0]

    db.execute("DROP TABLE imdb_data")
    db.execute("ALTER TABLE imdb_data_new RENAME TO imdb_data")

    print(f"  Migration complete: {row_count:,} rows expanded to individual columns.")


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

    The UPDATE to movie_progress is intentionally tolerant of missing rows —
    Stage 1 filtered movies are never inserted into movie_progress, so the
    UPDATE matches zero rows, which is correct behavior.

    Does NOT commit — the caller is responsible for batching commits.

    Args:
        db:      Open SQLite connection.
        tmdb_id: TMDB movie ID being filtered.
        stage:   Pipeline stage name (e.g., 'tmdb_export_filter', 'tmdb_fetch').
        reason:  Human-readable filter reason (e.g., 'adult', 'missing_imdb_id').
        details: Optional JSON string with additional context.
    """
    db.execute(
        """INSERT INTO filter_log (tmdb_id, stage, reason, details)
           VALUES (?, ?, ?, ?)""",
        (tmdb_id, stage, reason, details),
    )
    db.execute(
        "UPDATE movie_progress SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE tmdb_id = ?",
        (MovieStatus.FILTERED_OUT, tmdb_id),
    )


def batch_log_filter(
    db: sqlite3.Connection,
    entries: list[tuple[int, str, str, str | None]],
) -> None:
    """
    Batch version of log_filter — bulk-insert into filter_log and bulk-update
    movie_progress for multiple filtered movies at once.

    Each entry is a tuple of (tmdb_id, stage, reason, details).

    Does NOT commit — the caller is responsible for batching commits.
    """
    if not entries:
        return

    db.executemany(
        """INSERT INTO filter_log (tmdb_id, stage, reason, details)
           VALUES (?, ?, ?, ?)""",
        entries,
    )
    db.executemany(
        "UPDATE movie_progress SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE tmdb_id = ?",
        [(MovieStatus.FILTERED_OUT, e[0]) for e in entries],
    )


# ---------------------------------------------------------------------------
# log_ingestion_failures — bulk-insert ingestion errors and mark status.
# ---------------------------------------------------------------------------


def log_ingestion_failures(
    db: sqlite3.Connection,
    failures: list[tuple[int, str]],
) -> None:
    """
    Bulk-insert ingestion failure rows and mark movies as ingestion_failed.

    Each tuple is (tmdb_id, error_message). The error_message should include
    a human-readable step prefix (e.g. "Postgres movie card: <error>").
    A single movie may appear multiple times if it failed at more than one
    step (e.g. both movie card and lexical ingestion).

    Does NOT commit — the caller is responsible for batching commits.

    Args:
        db:       Open SQLite connection.
        failures: List of (tmdb_id, error_message) tuples.
    """
    if not failures:
        return

    db.executemany(
        "INSERT INTO ingestion_failures (tmdb_id, error_message) VALUES (?, ?)",
        failures,
    )
    # Deduplicate tmdb_ids for the status update — a movie with two failure
    # rows still gets a single status update.
    failed_tmdb_ids = list({f[0] for f in failures})
    db.executemany(
        "UPDATE movie_progress SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE tmdb_id = ?",
        [(MovieStatus.INGESTION_FAILED, tid) for tid in failed_tmdb_ids],
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
