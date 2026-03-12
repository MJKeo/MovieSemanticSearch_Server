"""
Builds JSONL request files for OpenAI Batch API submission.

Two main functions, one per wave:

build_wave1_requests(db_path) -> Path:
    1. Queries tracker for all movies at 'imdb_quality_passed' status
       that don't already have Wave 1 results in metadata_results
    2. Loads each movie's data from tmdb_data + imdb_data tables
    3. Constructs MovieInputData for each movie
    4. Runs pre-consolidation (keyword routing, maturity consolidation,
       skip assessment) per movie
    5. For each movie, calls plot_events and reception generators to
       build request body dicts
    6. Wraps each body in the JSONL batch format:
       {"custom_id": "12345-plot_events", "method": "POST",
        "url": "/v1/chat/completions", "body": {...}}
    7. Writes all requests to a JSONL file, returns the file path

build_wave2_requests(db_path) -> Path:
    1. Queries metadata_results for all movies with completed Wave 1
       that don't yet have Wave 2 results
    2. Loads Wave 1 intermediate outputs (plot_synopsis,
       review_insights_brief) from metadata_results
    3. Re-evaluates skip conditions using actual Wave 1 outputs:
       - plot_synopsis length determines narrative_techniques skip
       - review_insights_brief availability affects viewer_experience skip
       - Partial pipeline logic if plot_events failed
    4. For each movie x eligible generation, calls the appropriate
       generator to build request body dicts
    5. Writes JSONL file, returns path

Also handles:
    - Loading and assembling MovieInputData from tmdb_data + imdb_data
    - Inserting 'pending' rows into metadata_results for each request
    - Logging skip reasons for skipped generations
    - Progress reporting (movie count, request count, skip count)

The JSONL files are written to ingestion_data/metadata_batches/ with
timestamped filenames (e.g., wave1_20260312_143022.jsonl).
"""
