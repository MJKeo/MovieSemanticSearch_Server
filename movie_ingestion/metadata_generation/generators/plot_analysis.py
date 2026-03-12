"""
Plot Analysis request builder (Wave 2).

Purpose: Extract WHAT TYPE OF STORY this is -- themes, lessons, core
concepts. Powers queries like "movie about the cost of revenge" or
"redemption story."

Inputs:
    - title_with_year: "Title (Year)" format
    - genres: genre list
    - plot_synopsis: from Wave 1 plot_events output (required)
    - plot_keywords: plot-specific keywords only
    - review_insights_brief: from Wave 1 reception output (may be None)

Removed inputs (vs current system):
    - overview: superseded by plot_synopsis
    - reception_summary: subsumed by review_insights_brief
    - featured_reviews: subsumed by review_insights_brief

Skip condition: requires plot_synopsis (plot_events must have succeeded).

Response schema: PlotAnalysisMetadata
    - core_concept: single dominant story concept (6 words or less)
    - genre_signatures: 2-6 search-query-like genre phrases
    - conflict_scale: scale of consequences
    - character_arcs: 1-3 key character transformations
    - themes_primary: 1-3 core thematic concepts
    - lessons_learned: 0-3 key takeaways
    - generalized_plot_overview: 1-3 sentence thematic overview
    (All justification fields removed)

Model: gpt-5-mini, reasoning_effort: low

See docs/llm_metadata_generation_new_flow.md Section 5.1.
"""
