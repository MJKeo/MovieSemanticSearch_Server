"""
Viewer Experience request builder (Wave 2).

Purpose: Extract what it FEELS LIKE to watch the movie -- emotional,
sensory, cognitive experience. Powers queries like "edge of your seat
thriller" or "cozy feel-good movie."

Inputs:
    - title_with_year: "Title (Year)" format
    - genres: genre list
    - plot_synopsis: from Wave 1 (may be None if plot_events failed)
    - merged_keywords: deduplicated union of plot + overall keywords
    - maturity_summary: consolidated content advisory string
    - review_insights_brief: from Wave 1 reception (may be None)

Removed inputs (vs current system):
    - overview: superseded by plot_synopsis
    - plot_keywords / overall_keywords as separate inputs: merged
    - maturity_rating / maturity_reasoning / parental_guide_items:
      consolidated into maturity_summary
    - reception_summary / audience_reception_attributes / featured_reviews:
      subsumed by review_insights_brief

Skip condition: requires plot_synopsis OR review_insights_brief.
Can run without plot data if review data exists -- reviews carry
strong emotional/tonal signal independently.

Response schema: ViewerExperienceMetadata
    8 sections (emotional_palette, tension_adrenaline, tone_self_seriousness,
    cognitive_complexity, disturbance_profile, sensory_load,
    emotional_volatility, ending_aftertaste), each with terms + negations.
    3 sections are optional (should_skip flag).
    (All justification fields removed)

Model: gpt-5-mini, reasoning_effort: low

See docs/llm_metadata_generation_new_flow.md Section 5.2.
"""
