"""
System prompt for Plot Analysis generation.

Instructs the LLM to extract thematic content: core concept, genre
signatures, character arcs, themes, lessons, and generalized plot overview.

Receives review_insights_brief instead of raw reviews -- the brief
provides thematic observations at ~150-250 tokens instead of ~550-2600.

Based on existing prompt at:
implementation/prompts/vector_metadata_generation_prompts.py (PLOT_ANALYSIS section)

Key modifications:
    - Title input described as "Title (Year)" format
    - overview input removed (superseded by plot_synopsis)
    - review_insights_brief replaces reception_summary + featured_reviews
    - No justification fields in output spec (explanation_and_justification
      removed from core_concept, themes, lessons)
    - arc_transformation_description kept on CharacterArc (not a justification --
      it's a longer description that helps produce accurate arc labels)
"""

SYSTEM_PROMPT = ""
