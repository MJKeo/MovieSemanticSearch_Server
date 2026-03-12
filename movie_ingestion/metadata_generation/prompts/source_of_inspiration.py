"""
System prompt for Source of Inspiration generation (Production sub-call B).

Instructs the LLM to determine source material and production medium.
This is the ONLY generation that explicitly allows parametric knowledge:

"If you are highly confident about the source material based on your
knowledge, include it."

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
    - review_insights_brief added (replaces raw featured_reviews)
    - merged_keywords replaces concatenated keyword inputs
    - No justification field in output spec
"""

SYSTEM_PROMPT = ""
