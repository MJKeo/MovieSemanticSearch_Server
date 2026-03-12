"""
System prompt for Reception generation.

Dual-purpose prompt: produces both evaluative reception metadata AND a
descriptive review_insights_brief for downstream consumption.

The review_insights_brief is distinct from new_reception_summary:
    - new_reception_summary is evaluative: "was it good/bad and why"
    - review_insights_brief is descriptive: "what did reviewers observe?"
      covering themes, emotions, structural elements, source material

Must explicitly instruct source material extraction:
"Include any source material observations from reviews in the brief
(e.g., 'reviewers described it as a faithful adaptation of the novel',
'noted it was inspired by real events')."

Based on existing prompt at:
implementation/prompts/vector_metadata_generation_prompts.py (RECEPTION section)

Key modifications:
    - Title input described as "Title (Year)" format
    - review_insights_brief field added with detailed instructions
    - Source material extraction directive added
    - No justification fields in output spec
"""

SYSTEM_PROMPT = ""
