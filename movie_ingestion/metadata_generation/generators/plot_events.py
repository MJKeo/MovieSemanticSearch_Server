"""
Plot Events request builder (Wave 1).

Purpose: Extract WHAT HAPPENS -- concrete events, characters, settings.
Produces plot_summary which feeds Wave 2 as plot_synopsis.

Inputs (from ConsolidatedInputs):
    - title_with_year: "Title (Year)" format
    - overview: TMDB marketing summary
    - plot_summaries: shorter IMDB user-written summaries
    - plot_synopses: longest/most detailed plot recounts
    - plot_keywords: plot-specific keywords only (not overall)

Skip condition: Skips if ALL text sources (overview, synopses, summaries)
are either missing or too sparse. See _check_plot_events() in
pre_consolidation.py for exact thresholds.

Response schema: PlotEventsMetadata
    - plot_summary (str): detailed chronological spoiler-containing summary
    - setting (str): 10-word where/when phrase
    - major_characters (list[MajorCharacter]): essential characters

Model: gpt-5-mini, reasoning_effort: minimal

Critical prompt rule: "Only describe what is evident from the provided
data. Do not supplement with your own knowledge of this film. If data
is limited, produce a shorter summary rather than inventing details."

See docs/llm_metadata_generation_new_flow.md Section 4.1.
"""
