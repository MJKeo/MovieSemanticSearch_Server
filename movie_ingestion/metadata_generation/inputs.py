"""
Input data structures and shared utilities for the generation pipeline.

Data structures:

MovieInputData (dataclass):
    Raw fields loaded from tmdb_data + imdb_data tables. This is the
    contract between "load data from SQLite" and "run pre-consolidation."
    Fields include: tmdb_id, title, release_year, overview, genres,
    plot_synopses, plot_summaries, plot_keywords, overall_keywords,
    featured_reviews, reception_summary, audience_reception_attributes,
    maturity_rating, maturity_reasoning, parental_guide_items.

ConsolidatedInputs (dataclass):
    Post-pre-consolidation data passed to generators. Contains:
    - title_with_year (str): "Title (Year)" formatted string
    - All original MovieInputData fields
    - Routed keywords: plot_keywords, overall_keywords, merged_keywords
    - maturity_summary (str | None)
    - generations_to_run (set[str])
    - skip_reasons (dict[str, str])

Wave1Outputs (dataclass):
    Intermediate outputs from Wave 1, loaded from metadata_results:
    - plot_synopsis (str | None): from plot_events
    - review_insights_brief (str | None): from reception
    - plot_events_succeeded (bool)
    - reception_succeeded (bool)

Utility functions:

encode_custom_id(tmdb_id: int, generation_type: str) -> str:
    Produces batch request custom_id, e.g., "12345-plot_events".
    Must be unique per request in a batch.

decode_custom_id(custom_id: str) -> tuple[int, str]:
    Reverse mapping: "12345-plot_events" -> (12345, "plot_events").

format_title(title: str, year: int | None) -> str:
    Produces "Title (Year)" or just "Title" if year is None.

build_user_prompt(**labeled_fields) -> str:
    Assembles a user prompt from named fields. Skips None values.
    Example: build_user_prompt(title="The Matrix (1999)", genres="Action, Sci-Fi")
    -> "title: The Matrix (1999)\ngenres: Action, Sci-Fi"
"""
