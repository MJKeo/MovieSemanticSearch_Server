"""
Watch Context request builder (Wave 2 -- moved from Wave 1 in current system).

Purpose: Extract WHY and WHEN someone would choose to watch this movie.
Powers queries like "date night movie" or "something to watch high."

CRITICAL DESIGN DECISION: Watch context receives ZERO plot information.
No overview, no plot_synopsis. It answers "watch this if you want X
attributes" -- not "watch this if you want these specific events."
Plot detail anchors the model on narrative events rather than
experiential attributes.

Inputs:
    - title_with_year: "Title (Year)" format
    - genres: strong signal for occasion matching (horror -> halloween)
    - overall_keywords: categorical tags only (not plot_keywords)
    - maturity_summary: content advisory ("don't watch with kids")
    - review_insights_brief: from Wave 1 -- primary value driver,
      tells you how the movie *feels*

Removed inputs (vs current system):
    - overview: no plot info in watch context
    - plot_keywords: limited signal for viewing occasion decisions
    - reception_summary / audience_reception_attributes / featured_reviews:
      subsumed by review_insights_brief

Skip condition: requires genres (>=1 entry). Nearly always available.

Response schema: WatchContextMetadata
    4 sections (self_experience_motivations, external_motivations,
    key_movie_feature_draws, watch_scenarios), each with terms.
    (All justification fields removed)

Model: gpt-5-mini, reasoning_effort: medium

See docs/llm_metadata_generation_new_flow.md Section 5.3.
"""
