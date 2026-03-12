"""
Parses batch result JSONL files and stores results to SQLite.

Two main functions:

process_results(results_path: Path, wave: int, db_path: Path) -> ProcessingSummary:
    1. Reads the downloaded results JSONL line by line
    2. For each line:
       a. Decodes custom_id -> (tmdb_id, generation_type)
       b. Extracts the response body (choices[0].message.content)
       c. Validates against the appropriate Pydantic schema for that
          generation_type (PlotEventsMetadata, ReceptionMetadata, etc.)
       d. For Wave 1: extracts intermediate outputs:
          - plot_events -> plot_synopsis (from plot_summary field)
          - reception -> review_insights_brief (from the new field)
       e. Extracts token usage (prompt_tokens, completion_tokens)
       f. Upserts result into metadata_results table
    3. Returns ProcessingSummary with counts: succeeded, failed,
       total tokens, per-generation breakdowns

process_errors(errors_path: Path, db_path: Path) -> int:
    1. Reads the error JSONL line by line
    2. For each line: decodes custom_id, extracts error message
    3. Updates metadata_results status to 'failed' with error message
    4. Returns count of errors processed

Schema routing:
    Maps generation_type string -> Pydantic schema class:
    "plot_events" -> PlotEventsMetadata
    "reception" -> ReceptionMetadata
    "plot_analysis" -> PlotAnalysisMetadata
    ... etc.

    Uses Pydantic's model_validate_json() for parsing. If validation
    fails, the result is marked as failed with the validation error
    stored in the error column.

ProcessingSummary is a dataclass with:
    total, succeeded, failed, skipped counts
    total_input_tokens, total_output_tokens
    per_generation: dict[str, GenerationCounts]
"""
