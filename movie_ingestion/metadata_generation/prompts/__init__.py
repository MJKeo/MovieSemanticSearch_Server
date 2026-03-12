"""
System prompts for each LLM generation type.

Each module exports a single SYSTEM_PROMPT string constant containing
the full system prompt for that generation. Prompts are large (100-600+
lines each) and kept separate from generator code for clarity.

Prompts follow the established patterns from the existing system
(implementation/prompts/vector_metadata_generation_prompts.py):
    1. Context / Core Goal
    2. INPUTS description
    3. GENERAL RULES (data integrity, handling guidelines)
    4. OUTPUT (JSON schema hint)
    5. FIELD-BY-FIELD INSTRUCTIONS with word limits, examples,
       transformation rules, redundancy guidance

Key changes from existing prompts:
    - All prompts reference "Title (Year)" input format
    - Justification field instructions removed
    - plot_events prompt includes no-hallucination rule
    - reception prompt includes review_insights_brief instructions
      with explicit source material extraction directive
    - source_of_inspiration prompt includes parametric knowledge
      allowance: "If you are highly confident, include it"
    - watch_context prompt has no plot-related instructions
    - narrative_techniques prompt references genres input

8 prompt files for 7 generation types because production has 2
separate LLM calls (production_keywords + source_of_inspiration)
with different prompts.
"""
