"""
SQLite state management for batch tracking and generation results.

Manages two tables in tracker.db (alongside existing movie_progress,
tmdb_data, imdb_data tables):

metadata_batches — one row per OpenAI batch submission:
    batch_id        TEXT PK     OpenAI's batch ID (e.g., "batch_abc123")
    wave            INTEGER     1 or 2
    status          TEXT        pending / completed / failed / expired
    input_file_id   TEXT        OpenAI file ID for uploaded JSONL
    output_file_id  TEXT        OpenAI file ID for results (set on completion)
    error_file_id   TEXT        OpenAI file ID for errors (set on completion)
    created_at      TEXT        ISO timestamp
    completed_at    TEXT        ISO timestamp (set on completion)
    request_count   INTEGER     Number of requests in this batch

metadata_results — one row per (tmdb_id, generation_type):
    tmdb_id                 INTEGER     Movie ID
    generation_type         TEXT        "plot_events", "reception", "plot_analysis", etc.
    wave                    INTEGER     1 or 2
    status                  TEXT        pending / completed / failed / skipped
    batch_id                TEXT        FK to metadata_batches
    result_json             TEXT        Full LLM response JSON (parsed on read)
    plot_synopsis           TEXT        Wave 1 only: extracted from plot_events output
    review_insights_brief   TEXT        Wave 1 only: extracted from reception output
    input_tokens            INTEGER     Token usage
    output_tokens           INTEGER     Token usage
    error                   TEXT        Error message if failed

Key design decisions:
    - plot_synopsis and review_insights_brief are scalar columns (not buried
      in result_json) because they're queried directly during Wave 2 request
      building via SELECT ... WHERE tmdb_id = ?.
    - result_json stores the full LLM response so we can re-parse with
      updated schemas if needed without re-running the batch.
    - status tracks per-generation state independently of movie_progress
      status, since a movie can have partial results (e.g., plot_events
      succeeded but reception failed).

Provides helper functions:
    - init_metadata_tables(db)
    - insert_batch(db, batch_id, wave, ...)
    - update_batch_status(db, batch_id, status, ...)
    - upsert_result(db, tmdb_id, generation_type, ...)
    - get_wave1_outputs(db, tmdb_id) -> (plot_synopsis, review_insights_brief)
    - get_active_batch(db) -> batch row or None
    - get_generation_status(db, tmdb_id) -> dict of generation_type -> status
"""
