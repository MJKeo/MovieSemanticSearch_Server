"""
System prompt for Plot Events generation.

Instructs the LLM to extract concrete plot events, characters, and settings
from provided data. Must include the no-hallucination rule:

"Only describe what is evident from the provided data. Do not supplement
with your own knowledge of this film. If data is limited, produce a
shorter summary rather than inventing details."

The plot_summary output is the most critical field in the entire pipeline --
it feeds 4 of 5 Wave 2 generations as plot_synopsis.

Based on existing prompt at:
implementation/prompts/vector_metadata_generation_prompts.py (PLOT_EVENTS section)

Key modifications:
    - Title input described as "Title (Year)" format
    - Explicit no-hallucination instruction added
    - No justification fields in output spec
"""

SYSTEM_PROMPT = ""
