"""
Shared infrastructure for the LLM metadata evaluation pipeline.

This module provides:

EvaluationCandidate (dataclass):
    Fully specifies a candidate LLM configuration to evaluate:
    candidate_id, provider, model, system_prompt, response_format, kwargs.

DB utilities:
    EVAL_DB_PATH — path to the shared evaluation SQLite database.
    get_eval_connection(db_path) — open a connection to the eval DB.
    create_candidates_table(conn) — create the shared candidates table.
    store_candidate(conn, candidate, metadata_type) — upsert a candidate record.

Data loading:
    load_movie_input_data(tmdb_ids, tracker_db_path, imdb_dir) — load
    MovieInputData for the 70 test-corpus movies from the ingestion tracker.

Score analysis:
    compute_score_summary(conn, table, score_columns, ...) — return a pandas
    DataFrame with mean per candidate per scoring dimension. Supports optional
    score_weights for weighted overall_mean computation.

Test corpus:
    ORIGINAL_SET_TMDB_IDS, MEDIUM_SPARSITY_TMDB_IDS, HIGH_SPARSITY_TMDB_IDS,
    ALL_TMDB_IDS — the 70 pre-selected evaluation movies, stratified by data
    sparsity to ensure coverage of sparse-data failure modes.
"""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from implementation.llms.generic_methods import LLMProvider
from movie_ingestion.metadata_generation.inputs import MovieInputData

# ---------------------------------------------------------------------------
# Test corpus — 70 movies stratified by data sparsity
# ---------------------------------------------------------------------------

ORIGINAL_SET_TMDB_IDS = [
    9377, 269149, 1584, 109445, 2493, 354912, 508965, 14160, 10674, 808,
    13397, 76341, 85, 155, 245891, 1771, 569094, 299534, 11, 671, 120, 98,
    27205, 603, 157336, 335984, 329, 329865, 493922, 694, 49018, 1034541,
    176, 807, 496243, 419430, 1359, 550, 597, 13, 666277, 423, 11036, 1824,
    25195, 216015, 392044, 545611, 22538, 37136,
]
MEDIUM_SPARSITY_TMDB_IDS = [
    329974, 1498832, 821937, 92, 160, 45739, 576560, 1383243, 1642210, 1639488,
]
HIGH_SPARSITY_TMDB_IDS = [
    270909, 493103, 64262, 1611977, 706910, 1297426, 35952, 158227, 215782, 1642486,
]
EVALUATION_TEST_SET_TMDB_IDS = ORIGINAL_SET_TMDB_IDS + MEDIUM_SPARSITY_TMDB_IDS + HIGH_SPARSITY_TMDB_IDS

# ---------------------------------------------------------------------------
# EvaluationCandidate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvaluationCandidate:
    """Fully specifies a candidate LLM configuration to evaluate.

    All 5 attributes that define a candidate — provider, model, model
    parameters, system prompt, and structured output schema — are captured
    here alongside a user-assigned unique identifier.

    frozen=True makes instances hashable, safe to use as dict keys, and
    prevents accidental mutation after creation.
    """
    candidate_id: str                       # user-assigned unique identifier
    provider: LLMProvider
    model: str
    system_prompt: str
    response_format: type[BaseModel]        # the structured output schema class
    kwargs: dict = field(default_factory=dict)  # model params (temperature, etc.)

# ---------------------------------------------------------------------------
# DB utilities
# ---------------------------------------------------------------------------

# Shared evaluation database — gitignored via evaluation_data/ in .gitignore
EVAL_DB_PATH = Path("evaluation_data/eval.db")


def get_eval_connection(db_path: Path | str = EVAL_DB_PATH) -> sqlite3.Connection:
    """Open a connection to the evaluation SQLite database.

    Creates the evaluation_data/ directory and the DB file if they don't
    exist. Sets WAL journal mode for crash-safety (each row costs API spend)
    and row_factory = sqlite3.Row so columns are accessible by name.

    Accepts str or Path for convenience — coerces to Path internally.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def create_candidates_table(conn: sqlite3.Connection) -> None:
    """Create the shared candidates table if it doesn't exist.

    One row per (candidate_id, metadata_type) pair — the same candidate_id
    may be reused across multiple metadata evaluations (e.g., plot_events
    and plot_analysis) if the configuration is identical.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            candidate_id  TEXT NOT NULL,
            metadata_type TEXT NOT NULL,
            provider      TEXT NOT NULL,
            model         TEXT NOT NULL,
            system_prompt TEXT NOT NULL,
            parameters    TEXT NOT NULL,   -- JSON of kwargs
            schema_class  TEXT NOT NULL,   -- response_format.__name__
            created_at    TEXT NOT NULL,
            PRIMARY KEY (candidate_id, metadata_type)
        )
    """)
    conn.commit()


def store_candidate(
    conn: sqlite3.Connection,
    candidate: EvaluationCandidate,
    metadata_type: str,
) -> None:
    """Upsert a candidate record into the candidates table.

    Uses INSERT OR REPLACE so re-running with the same candidate_id
    updates the record if any attribute changed.

    Args:
        conn: Open connection to the eval DB (candidates table must exist).
        candidate: The candidate configuration to store.
        metadata_type: Which evaluation this candidate belongs to,
            e.g. 'plot_events' or 'plot_analysis'.
    """
    conn.execute(
        """
        INSERT OR REPLACE INTO candidates
            (candidate_id, metadata_type, provider, model, system_prompt,
             parameters, schema_class, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            candidate.candidate_id,
            metadata_type,
            candidate.provider.value,
            candidate.model,
            candidate.system_prompt,
            json.dumps(candidate.kwargs),
            candidate.response_format.__name__,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Default paths match the ingestion pipeline conventions
_DEFAULT_TRACKER_DB = Path("ingestion_data/tracker.db")
_DEFAULT_IMDB_DIR = Path("ingestion_data/imdb")


def load_movie_input_data(
    tmdb_ids: list[int],
    tracker_db_path: Path = _DEFAULT_TRACKER_DB,
    imdb_dir: Path = _DEFAULT_IMDB_DIR,
) -> dict[int, MovieInputData]:
    """Load MovieInputData for the given tmdb_ids from the ingestion tracker.

    Queries the tracker SQLite for title/release_date from tmdb_data, and
    the full per-movie IMDB data from imdb_data. Both tables must be populated
    (i.e., the movie must have completed at least through Stage 4 of the
    ingestion pipeline).

    Movies missing from either table are logged and skipped — they will not
    appear in the returned dict.

    Args:
        tmdb_ids: List of TMDB movie IDs to load.
        tracker_db_path: Path to ingestion_data/tracker.db.
        imdb_dir: Directory containing per-movie IMDB JSON files.

    Returns:
        Dict mapping tmdb_id → MovieInputData for all successfully loaded movies.
    """
    if not tracker_db_path.exists():
        raise FileNotFoundError(
            f"Tracker DB not found at {tracker_db_path}. "
            "Ensure the ingestion pipeline has run through at least Stage 4."
        )

    result: dict[int, MovieInputData] = {}
    placeholders = ", ".join("?" * len(tmdb_ids))

    with sqlite3.connect(str(tracker_db_path)) as tracker:
        tracker.row_factory = sqlite3.Row

        # Join tmdb_data (title, release_date) with imdb_data (everything else).
        # imdb_data.overview is preferred over tmdb_data (more detailed);
        # release_year is extracted from tmdb_data.release_date ("YYYY-MM-DD").
        rows = tracker.execute(
            f"""
            SELECT
                t.tmdb_id,
                t.title,
                CAST(SUBSTR(t.release_date, 1, 4) AS INTEGER) AS release_year,
                i.overview,
                i.maturity_rating,
                i.reception_summary,
                i.genres,
                i.synopses,
                i.plot_summaries,
                i.plot_keywords,
                i.overall_keywords,
                i.featured_reviews,
                i.review_themes,
                i.maturity_reasoning,
                i.parental_guide_items
            FROM tmdb_data t
            JOIN imdb_data i ON t.tmdb_id = i.tmdb_id
            WHERE t.tmdb_id IN ({placeholders})
            """,
            tmdb_ids,
        ).fetchall()

    loaded_ids: set[int] = set()
    for row in rows:
        tmdb_id = row["tmdb_id"]
        loaded_ids.add(tmdb_id)

        # JSON columns are stored as TEXT arrays; parse each one
        def _parse_json_list(raw: str | None) -> list:
            if not raw:
                return []
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return []

        result[tmdb_id] = MovieInputData(
            tmdb_id=tmdb_id,
            title=row["title"] or "",
            release_year=row["release_year"],
            overview=row["overview"] or "",
            genres=_parse_json_list(row["genres"]),
            plot_synopses=_parse_json_list(row["synopses"]),
            plot_summaries=_parse_json_list(row["plot_summaries"]),
            plot_keywords=_parse_json_list(row["plot_keywords"]),
            overall_keywords=_parse_json_list(row["overall_keywords"]),
            featured_reviews=_parse_json_list(row["featured_reviews"]),
            reception_summary=row["reception_summary"],
            # review_themes maps to audience_reception_attributes: [{name, sentiment}]
            audience_reception_attributes=_parse_json_list(row["review_themes"]),
            maturity_rating=row["maturity_rating"] or "",
            maturity_reasoning=_parse_json_list(row["maturity_reasoning"]),
            parental_guide_items=_parse_json_list(row["parental_guide_items"]),
        )

    # Warn for any requested IDs that were not found in both tables
    missing = set(tmdb_ids) - loaded_ids
    if missing:
        print(
            f"  Warning: {len(missing)} movie(s) not found in tracker "
            f"(missing from tmdb_data or imdb_data): {sorted(missing)}"
        )

    print(f"  Loaded {len(result)}/{len(tmdb_ids)} movies from tracker.")
    return result

# ---------------------------------------------------------------------------
# Score analysis
# ---------------------------------------------------------------------------

def compute_score_summary(
    conn: sqlite3.Connection,
    table: str,
    score_columns: list[str],
    candidate_ids: list[str] | None = None,
    movie_ids: list[int] | None = None,
    score_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute mean scores per candidate per dimension.

    Queries the given evaluation results table and aggregates scores by
    candidate_id. Returns a DataFrame indexed by candidate_id with columns
    for each dimension's mean score (e.g., 'plot_summary_mean'), plus an
    'overall_mean' column.

    Args:
        conn: Open connection to the eval DB.
        table: Name of the evaluation results table (e.g., 'plot_events_evaluations').
        score_columns: List of score column names to aggregate
            (e.g., ['plot_summary_score', 'setting_score']).
        candidate_ids: If provided, filter to only these candidate IDs.
        movie_ids: If provided, filter to only these movie IDs.
        score_weights: If provided, a mapping from score column name to its
            weight for computing overall_mean as a weighted average. Columns
            not in the dict get weight 1.0. If None, all columns are equally
            weighted (simple average).

    Returns:
        DataFrame indexed by candidate_id with mean columns per dimension
        plus an overall_mean column.
    """
    score_cols_sql = ", ".join(score_columns)
    conditions: list[str] = []
    params: list = []

    if candidate_ids:
        placeholders = ", ".join("?" * len(candidate_ids))
        conditions.append(f"candidate_id IN ({placeholders})")
        params.extend(candidate_ids)

    if movie_ids:
        placeholders = ", ".join("?" * len(movie_ids))
        conditions.append(f"movie_id IN ({placeholders})")
        params.extend(movie_ids)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    rows = conn.execute(
        f"SELECT candidate_id, {score_cols_sql} FROM {table} {where_clause}",
        params,
    ).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(row) for row in rows])

    agg: dict[str, pd.Series] = {}
    for col in score_columns:
        # Strip '_score' suffix to form the human-readable dimension label
        dim = col.removesuffix("_score")
        agg[f"{dim}_mean"] = df.groupby("candidate_id")[col].mean()

    summary = pd.DataFrame(agg)

    # overall_mean: weighted average across dimension means (equal weights if none given)
    mean_cols = [c for c in summary.columns if c.endswith("_mean")]
    if score_weights:
        # Build weight series aligned to mean_cols. Each mean_col is "{dim}_mean",
        # and score_columns are "{dim}_score", so we map back to look up weights.
        weights = []
        for mc in mean_cols:
            # Reverse the dim_mean → dim_score mapping to find the weight
            dim = mc.removesuffix("_mean")
            score_col = f"{dim}_score"
            weights.append(score_weights.get(score_col, 1.0))
        weight_series = pd.Series(weights, index=mean_cols)
        summary["overall_mean"] = (
            summary[mean_cols].mul(weight_series).sum(axis=1) / weight_series.sum()
        )
    else:
        summary["overall_mean"] = summary[mean_cols].mean(axis=1)

    return summary.sort_values("overall_mean", ascending=False)
