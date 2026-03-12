"""
System prompt for Viewer Experience generation.

Instructs the LLM to extract emotional, sensory, and cognitive experience
attributes. Each section produces terms (what the movie IS like) and
negations (what it is NOT like).

Receives merged_keywords (deduplicated union of plot + overall) and
maturity_summary (consolidated content advisory string).

Can run without plot_synopsis if review_insights_brief exists -- reviews
carry strong emotional/tonal signal independently.

Based on existing prompt at:
implementation/prompts/vector_metadata_generation_prompts.py (VIEWER_EXPERIENCE section)

Key modifications:
    - Title input described as "Title (Year)" format
    - maturity_summary replaces maturity_rating + maturity_reasoning +
      parental_guide_items as three separate inputs
    - merged_keywords replaces separate plot/overall keyword inputs
    - review_insights_brief replaces reception_summary +
      audience_reception_attributes + featured_reviews
    - No justification fields in output spec (one per section, 8 total removed)
"""

SYSTEM_PROMPT = ""
