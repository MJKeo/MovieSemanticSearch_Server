"""
LLM metadata generation pipeline using OpenAI's Batch API.

Generates 7 types of LLM metadata across 8 vector spaces for ~112K movies.
Uses a two-wave batch architecture: Wave 1 (plot_events + reception) produces
intermediate outputs that feed Wave 2 (plot_analysis, viewer_experience,
watch_context, narrative_techniques, production).

Workflow:
    python -m movie_ingestion.metadata_generation.run submit --wave 1
    python -m movie_ingestion.metadata_generation.run status
    python -m movie_ingestion.metadata_generation.run process
    python -m movie_ingestion.metadata_generation.run status
    python -m movie_ingestion.metadata_generation.run process

See docs/llm_metadata_generation_new_flow.md for the full design document
covering input routing, skip conditions, and design decisions.
"""
