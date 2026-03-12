"""
Reception request builder (Wave 1).

Purpose: Extract what audiences and critics think about the movie.
Dual purpose: also produces review_insights_brief -- a dense ~150-250
token paragraph consolidating review observations for Wave 2 consumption.

Inputs (from ConsolidatedInputs):
    - title_with_year: "Title (Year)" format
    - reception_summary: externally generated audience opinion summary
    - audience_reception_attributes: key attributes with sentiment labels
    - featured_reviews: up to 5 full review texts -- THIS IS THE ONLY
      CALL that receives raw reviews (all Wave 2 calls get the brief instead)

Skip condition: requires at least ONE of:
    - featured_reviews (>=1 review)
    - reception_summary
    - audience_reception_attributes (>=2 attributes)

Response schema: ReceptionMetadata
    - new_reception_summary (str): 2-3 sentence evaluative summary
    - praise_attributes (list[str]): 0-4 tag-like phrases
    - complaint_attributes (list[str]): 0-4 tag-like phrases
    - review_insights_brief (str): NEW -- dense paragraph of thematic,
      emotional, structural, and source-material observations from reviews.
      Not embedded. Purely an intermediate input for Wave 2.

Model: gpt-5-mini, reasoning_effort: low

The prompt must instruct the model to include source material observations
in the brief (e.g., "reviewers described it as a faithful adaptation").

See docs/llm_metadata_generation_new_flow.md Section 4.2.
"""
