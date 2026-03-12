"""
System prompt for Narrative Techniques generation.

Instructs the LLM to extract storytelling craft attributes: POV, narrative
delivery, archetype, information control, characterization, and more.

Receives genres as a NEW input (not in current system). Genres help ground
structural analysis -- "mystery" implies information control techniques,
"documentary" implies specific POV structures.

Receives overall_keywords only (not plot_keywords). Structural tags
like "nonlinear timeline" and "unreliable narrator" tend to live in
overall keywords; plot keywords add noise without structural signal.

Based on existing prompt at:
implementation/prompts/vector_metadata_generation_prompts.py (NARRATIVE_TECHNIQUES section)

Key modifications:
    - Title input described as "Title (Year)" format
    - genres added as input with structural analysis guidance
    - overall_keywords only (not plot_keywords)
    - review_insights_brief replaces reception_summary + featured_reviews
    - No justification fields in output spec (11 total removed)
"""

SYSTEM_PROMPT = ""
