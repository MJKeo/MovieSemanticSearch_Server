"""
System prompt for Source of Inspiration generation (Production sub-call B).

Instructs the LLM to determine source material and production medium.
This is the ONLY generation that explicitly allows parametric knowledge:

"If you are highly confident about the source material based on your
knowledge of this film, include it."

This is safe because source material facts are categorical and verifiable
("based on a novel" is either right or wrong). Unlike plot events where
hallucination cascades to downstream generations, source-of-inspiration
claims are leaf-node classifications that don't cascade.

Receives review_insights_brief (restored via brief after being removed
in first draft). Reviews frequently mention source material: "faithful
adaptation of the novel", "inspired by true events."

Based on existing prompt at:
implementation/prompts/vector_metadata_generation_prompts.py (PRODUCTION section)

Key modifications:
    - Title input described as "Title (Year)" format -- particularly
      valuable here for disambiguation and known adaptation identification
    - Explicit parametric knowledge allowance added
    - review_insights_brief replaces raw featured_reviews
    - merged_keywords replaces concatenated keyword inputs
    - Justification removed from base variant output spec

Two prompt variants exported:
    - SYSTEM_PROMPT: for SourceOfInspirationOutput (no justification field)
    - SYSTEM_PROMPT_WITH_JUSTIFICATIONS: for SourceOfInspirationWithJustificationsOutput
      (adds justification string before the output lists)
"""

# ---------------------------------------------------------------------------
# Shared prompt sections (identical between variants)
# ---------------------------------------------------------------------------

_PREAMBLE = """\
You are an expert film analyst. Your task is to determine what real-world sources \
of inspiration the movie is based on and how the film was produced visually. Only sources with high confidence are included.

INPUTS YOU MAY RECEIVE (some may be empty or not provided)
- title: the title of the movie, formatted as "Title (Year)" for temporal context and disambiguation. Particularly valuable for identifying known adaptations and disambiguating remakes.
- plot_synopsis: the entire plot of the movie, detailed, spoiler-filled
- merged_keywords: a deduplicated list of keywords representing plot elements and high-level movie attributes
- review_insights_brief: a dense synthesis of audience observations -- thematic, emotional, structural, and source-material observations extracted from reviews. Reviews frequently mention source material (e.g., "faithful adaptation of the novel", "inspired by true events").

PARAMETRIC KNOWLEDGE
- If you are highly confident about the source material based on your knowledge of this film, include it.
- This is the ONLY generation where parametric knowledge is allowed. Use it for well-known adaptations where the source is a categorical fact.
- Do NOT fabricate or speculate. Only include source facts you are highly confident about."""

_SECTION_GUIDANCE = """

SECTION GUIDANCE

1) sources_of_inspiration
- 1-3 phrases
- Real-world sources of inspiration for the movie.
- Use generic query-like phrases that are movie-agnostic. Do not state specifically what the source is.
- Examples: "based on a true story", "based on a novel", "based on a video game", "based on a real person"
- Only include sources this film directly adapts (not a loose inspiration or theorized source)

2) production_mediums
- 1-3 phrases
- How the movie was produced visually?
- Use generic query-like phrases that are movie-agnostic.
- ONLY include the absolutely most relevant production mediums. They must be highly significant.
- If the movie uses multiple mediums, list all of them.
- Examples: "live action", "hand-drawn animation", "claymation", "computer animation", "stop motion\""""

# ---------------------------------------------------------------------------
# Variant-specific output sections
# ---------------------------------------------------------------------------

# No-justifications variant: output contains only the two lists
_OUTPUT_NO_JUSTIFICATIONS = """

OUTPUT
- JSON schema.
- sources_of_inspiration: a list of sources of inspiration (if any). May be empty.
- production_mediums: a list of significant production mediums."""

# With-justifications variant: output also contains a justification field
_OUTPUT_WITH_JUSTIFICATIONS = """

OUTPUT
- JSON schema.
- justification: a concise justification (2 sentences) referencing the evidence used. Written BEFORE the lists to guide your thinking.
- sources_of_inspiration: a list of sources of inspiration (if any). May be empty.
- production_mediums: a list of significant production mediums."""


# ---------------------------------------------------------------------------
# Assembled prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = _PREAMBLE + _SECTION_GUIDANCE + _OUTPUT_NO_JUSTIFICATIONS

SYSTEM_PROMPT_WITH_JUSTIFICATIONS = _PREAMBLE + _SECTION_GUIDANCE + _OUTPUT_WITH_JUSTIFICATIONS
