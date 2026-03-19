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
    - Justification removed from base variant output spec

Two prompt variants exported:
    - SYSTEM_PROMPT: for ProductionKeywordsOutput (no justification field)
    - SYSTEM_PROMPT_WITH_JUSTIFICATIONS: for ProductionKeywordsWithJustificationsOutput
      (adds justification string before terms)
"""

# ---------------------------------------------------------------------------
# Shared prompt sections (identical between variants)
# ---------------------------------------------------------------------------

_PREAMBLE = """\
You are an expert film analyst. Your task is to take a list of keywords and \
return every keyword that relates to the production of the movie (how it was produced in the real world).

INPUTS
- title: the title of the movie, formatted as "Title (Year)" for temporal context and disambiguation
- merged_keywords: a deduplicated list of keywords representing plot elements and high-level movie attributes. Not all are production-relevant -- your job is to filter.

PRODUCTION RELEVANCY
- In what way was the movie produced? (ex. "live action", "animation")
- When/where was the movie created?
- Who created the movie and why did they create it?
- What was the production process like?
- Was there a source of inspiration for the movie?
- Any keyword that doesn't relate to one of the above questions must not be included.

GUIDELINES
- ONLY include keywords from the provided list. Adding new keywords is a catastrophic failure.
- It is NOT related to plot events, thematic analysis, genres, or any other contents of the final movie product. ONLY how it was produced in the first place.
- DO NOT use any information other than what's present in the input.
- You may leave terms as an empty list if no production keywords are present."""

# ---------------------------------------------------------------------------
# Variant-specific output sections
# ---------------------------------------------------------------------------

# No-justifications variant: output contains only terms
_OUTPUT_NO_JUSTIFICATIONS = """

OUTPUT
- JSON schema.
- terms: all keywords from the provided list that relate to HOW the movie was produced in the real world."""

# With-justifications variant: output also contains a justification field
_OUTPUT_WITH_JUSTIFICATIONS = """

OUTPUT
- JSON schema.
- justification: a concise justification (1 sentence) referencing the evidence used. Written BEFORE terms to guide your thinking.
- terms: all keywords from the provided list that relate to HOW the movie was produced in the real world."""


# ---------------------------------------------------------------------------
# Assembled prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = _PREAMBLE + _OUTPUT_NO_JUSTIFICATIONS

SYSTEM_PROMPT_WITH_JUSTIFICATIONS = _PREAMBLE + _OUTPUT_WITH_JUSTIFICATIONS
