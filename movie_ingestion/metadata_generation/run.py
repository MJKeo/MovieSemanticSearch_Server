"""
CLI entry point for the metadata generation pipeline.

Commands:
    submit --wave 1    Build Wave 1 JSONL (plot_events + reception for all
                       eligible movies), upload to OpenAI Files API, create
                       batch. Stores batch_id in metadata_batches table.

    submit --wave 2    Build Wave 2 JSONL using stored Wave 1 intermediate
                       outputs (plot_synopsis, review_insights_brief).
                       Evaluates per-movie skip conditions before building
                       requests. Uploads and creates batch.

    status             Check the status of the most recent active batch.
                       Reports: batch_id, wave, status, request counts
                       (total/completed/failed), elapsed time.

    process            Download results from the most recently completed batch.
                       For Wave 1: parses results, stores intermediate outputs
                       (plot_synopsis, review_insights_brief) to metadata_results,
                       then auto-submits Wave 2 batch.
                       For Wave 2: parses results, stores final metadata,
                       updates tracker movie_progress status.

Data flow:
    1. Queries tracker.db for movies at 'imdb_quality_passed' status
    2. Loads movie data from tmdb_data + imdb_data tables
    3. Delegates to request_builder, batch_manager, result_processor

Usage:
    python -m movie_ingestion.metadata_generation.run submit --wave 1
    python -m movie_ingestion.metadata_generation.run status
    python -m movie_ingestion.metadata_generation.run process
"""
