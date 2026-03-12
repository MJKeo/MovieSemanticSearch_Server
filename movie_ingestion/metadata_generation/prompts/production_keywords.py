"""
System prompt for Production Keywords generation (Production sub-call A).

Instructs the LLM to FILTER (not generate) the provided keyword list
to only production-relevant keywords. The model classifies existing
keywords, it doesn't create new ones.

Receives merged_keywords (deduplicated union of plot + overall). The
full merged list gives the classifier more material -- some plot_keywords
may have production relevance (e.g., "shot on location"). Extra keywords
just mean more to filter, not more noise in output.

Based on existing prompt at:
implementation/prompts/vector_metadata_generation_prompts.py (PRODUCTION section)

Key modifications:
    - Title input described as "Title (Year)" format
    - merged_keywords replaces overall_keywords only
    - No justification field in output spec
"""

SYSTEM_PROMPT = ""
