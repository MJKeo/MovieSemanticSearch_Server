"""
Production request builder (Wave 2) -- TWO separate LLM sub-calls.

Builds request bodies for both production_keywords and source_of_inspiration,
which are independent LLM calls submitted as separate batch requests.

Sub-call A: Production Keywords
    Purpose: Filter keyword list to production-relevant keywords.
    The LLM classifies (not generates).
    Inputs:
        - title_with_year: "Title (Year)" format
        - merged_keywords: full deduplicated union (both plot + overall)
    Skip condition: merged_keywords >= 5 entries
    Response schema: ProductionKeywordsResponse (terms list)
    Model: gpt-5-mini, reasoning_effort: low

Sub-call B: Source of Inspiration
    Purpose: Determine source material and production medium.
    Inputs:
        - title_with_year: "Title (Year)" -- particularly valuable for
          disambiguation and identifying known adaptations
        - plot_synopsis: from Wave 1
        - merged_keywords: deduplicated union
        - review_insights_brief: from Wave 1 -- reviews frequently
          mention source material ("faithful adaptation of the novel")
    Skip condition: NEVER skips. Title + year always available, and
    parametric knowledge is explicitly allowed for this generation.
    Response schema: SourceOfInspirationResponse
        (sources_of_inspiration list, production_mediums list)
    Model: gpt-5-mini, reasoning_effort: low

    PARAMETRIC KNOWLEDGE ALLOWED: The prompt includes: "If you are
    highly confident about the source material based on your knowledge,
    include it." This is the ONLY generation that allows this -- all
    others forbid it or don't mention it.

Public interface:
    build_production_keywords_request(inputs, wave1_outputs) -> dict
    build_source_of_inspiration_request(inputs, wave1_outputs) -> dict

Both are called by request_builder.py and submitted as separate
batch request lines with custom_ids like "12345-production_keywords"
and "12345-source_of_inspiration".

See docs/llm_metadata_generation_new_flow.md Section 5.5.
"""
