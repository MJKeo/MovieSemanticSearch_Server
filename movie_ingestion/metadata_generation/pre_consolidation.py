"""
Pre-consolidation phase: pure data processing before any LLM calls.

Three functions that transform raw movie data into optimized LLM inputs.
No LLM cost — all logic is deterministic and testable without mocking.

1. route_keywords(plot_keywords, overall_keywords) -> KeywordRouting:
    Produces three keyword lists per the routing table:
    - plot_keywords (list[str]): passed to plot_events, plot_analysis
    - overall_keywords (list[str]): passed to watch_context, narrative_techniques
    - merged_keywords (list[str]): deduplicated union for viewer_experience,
      production_keywords, source_of_inspiration
    Implementation: merged = list(dict.fromkeys(plot + overall))

2. consolidate_maturity(rating, reasoning, parental_guide_items) -> str | None:
    Produces a single maturity_summary string:
    - If maturity_reasoning exists (>=1 item):
      "R -- violence, strong language, brief nudity"
    - If absent, falls back to parental_guide_items:
      "R -- severe violence, moderate language, mild nudity"
    - If no maturity data at all: returns None
    Sent to: viewer_experience, watch_context

3. assess_skip_conditions(movie_data, wave1_outputs=None) -> SkipAssessment:
    Evaluates per-generation minimum data thresholds and returns:
    - generations_to_run (set[str]): which generation types to include
    - skip_reasons (dict[str, str]): why each skipped generation was skipped

    Called twice in the pipeline:
    - Before Wave 1: determines plot_events (always runs) and reception skips
    - Before Wave 2: uses actual Wave 1 outputs to determine:
      * plot_analysis: requires plot_synopsis
      * viewer_experience: requires plot_synopsis OR review_insights_brief
      * watch_context: requires genres (>=1)
      * narrative_techniques: requires plot_synopsis >100 words
      * production_keywords: requires merged_keywords >=5
      * source_of_inspiration: always runs (title + year always available)

    Also handles partial pipeline logic: if plot_events failed,
    skips plot_analysis and narrative_techniques but keeps others.

See docs/llm_metadata_generation_new_flow.md Section 3 (Pre-Consolidation)
and Section 6 (Sparse Movie Skip Conditions) for the full specification.
"""
