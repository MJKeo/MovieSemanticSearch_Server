---
name: "new-metadata-evaluation"
description: "Use when the user wants to add a new metadata evaluation type under movie_ingestion/metadata_generation/evaluations/."
---

# New Metadata Evaluation

Scaffold a new metadata evaluation module and then open the rubric-design conversation.

## Workflow
1. Treat `references/original-command.md` as the source of truth for the scaffold details.
2. Read the exact evaluation, schema, generator, prompt, and pipeline files listed there before editing.
3. Generate the mechanical pieces completely, but leave rubric-dimension TODOs where the legacy workflow requires design discussion first.
4. Keep candidate IDs, table names, and prompt builders aligned with the target metadata type.

## References
- `references/original-command.md` -> full scaffold recipe and TODO placeholders
- `movie_ingestion/metadata_generation/evaluations/plot_events.py` -> reference implementation
- `movie_ingestion/metadata_generation/schemas.py` -> source of embeddable fields
