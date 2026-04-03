"""
System prompts for each LLM generation type.

Each module exports a SYSTEM_PROMPT string constant (plot_events exports
two branch-specific variants: SYSTEM_PROMPT_SYNOPSIS and SYSTEM_PROMPT_SYNTHESIS).
Prompts are large (100-600+ lines each) and kept separate from generator
code for clarity.

8 prompt files for 8 generation types: plot_events, reception,
plot_analysis, viewer_experience, watch_context, narrative_techniques,
production_keywords, source_of_inspiration.
"""
