"""
LLM metadata generation pipeline using OpenAI's Batch API.

Generates 8 types of LLM metadata for ~109K movies. Each metadata type
is handled individually via the CLI — no wave grouping.

Workflow (per metadata type, currently plot_events):
    python -m movie_ingestion.metadata_generation.run eligibility
    python -m movie_ingestion.metadata_generation.run submit
    python -m movie_ingestion.metadata_generation.run status
    python -m movie_ingestion.metadata_generation.run process

See docs/llm_metadata_generation_new_flow.md for the full design document
covering input routing, skip conditions, and design decisions.
"""
