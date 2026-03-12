"""
Narrative Techniques request builder (Wave 2).

Purpose: Extract HOW the story is told -- cinematic narrative craft,
structure, storytelling mechanics. Powers queries like "movie with an
unreliable narrator" or "non-linear timeline."

Inputs:
    - title_with_year: "Title (Year)" format
    - genres: ADDED (not in current system) -- helps ground structural
      analysis ("mystery" implies information control, "documentary"
      implies specific POV structures)
    - plot_synopsis: from Wave 1 (required, must be >100 words)
    - overall_keywords: structural tags ("nonlinear timeline",
      "unreliable narrator") tend to live in overall keywords
    - review_insights_brief: from Wave 1 -- reviews often reveal
      structural observations ("the twist was predictable")

Removed inputs (vs current system):
    - plot_keywords: rarely carry structural narrative signal
    - reception_summary / featured_reviews: subsumed by brief

Skip condition: requires plot_synopsis with >100 words. Structural
analysis needs sufficient plot detail -- a thin 2-sentence summary
doesn't provide enough material to identify techniques.

Response schema: NarrativeTechniquesMetadata
    11 sections (pov_perspective, narrative_delivery, narrative_archetype,
    information_control, characterization_methods, character_arcs,
    audience_character_perception, conflict_stakes_design,
    thematic_delivery, meta_techniques, additional_plot_devices),
    each with terms.
    (All justification fields removed)

Model: gpt-5-mini, reasoning_effort: medium

See docs/llm_metadata_generation_new_flow.md Section 5.4.
"""
