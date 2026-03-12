"""
System prompt for Watch Context generation.

Instructs the LLM to extract viewing occasion and motivation attributes.
Answers "when/why would someone choose to watch this?"

CRITICAL: This prompt must contain NO plot-related instructions or
references. Watch context is purely experiential -- it receives no
overview, no plot_synopsis, no plot_keywords. Plot detail anchors
the model on narrative events rather than experiential attributes.

Receives overall_keywords only (not plot_keywords). Categorical tags
like "family-friendly", "cult classic" inform viewing occasions;
plot-specific tags like "murder investigation" don't.

Based on existing prompt at:
implementation/prompts/vector_metadata_generation_prompts.py (WATCH_CONTEXT section)

Key modifications:
    - Title input described as "Title (Year)" format
    - overview input removed entirely
    - overall_keywords only (not plot_keywords)
    - maturity_summary added as input
    - review_insights_brief replaces all reception inputs
    - No justification fields in output spec (4 total removed)
    - All plot-related language stripped from prompt
"""

SYSTEM_PROMPT = ""
