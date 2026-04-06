# [063] — Vector Search Prompt Realignment: Systematic Audit-and-Rewrite Process

## Status
Active

## Context

As the metadata generation schemas evolved (schema field renames, section
merges, schema simplifications), the search-time subquery and weight prompts
in `implementation/prompts/` became misaligned with the actual embedded content.
Specific problems discovered across multiple vector spaces:

- Prompts described fields that no longer exist (e.g. `core_concept`,
  `themes_primary`, `lessons_learned` in plot_analysis)
- Prompts fabricated content categories (e.g. "aesthetic vibes" in
  viewer_experience, person names in production)
- Prompts had wrong format awareness (e.g. describing structured labeled fields
  when the embedded content is a flat unlabeled comma-separated list)
- Prompts had no full embedded examples, causing the LLM to guess format
- Critical semantic boundaries were missing or wrong (e.g. experience vs theme,
  motivation vs emotion, technique vs experience, evaluation vs description)
- Example outputs leaked terms from adjacent vector spaces

These misalignments cause LLM subqueries to generate terms with zero or
near-zero cosine similarity against actual embedded content.

## Decision

Establish a systematic 3-phase process for realigning each vector search prompt
pair (subquery + weight), codified as the `/realign-vector-search-prompts`
command in `.claude/commands/`. The process:

**Phase 1 — Build source of truth**: Read `vector_text.py`, `embedding_text()`
methods, generation prompts/schemas, and `Movie` helper methods to enumerate
exactly what content is embedded for the target space and in what format.

**Phase 2 — Catalog misalignments**: Compare the current prompts against the
source of truth. Rate each misalignment by severity (critical/medium/low) and
stop for human confirmation before rewriting.

**Phase 3 — Rewrite from scratch**: Rewrite both prompts following established
design principles:
- Accurate field inventory with example terms
- At least one full embedded example showing realistic assembled content
- Format-aware transformation approach (flat list vs. prose vs. labeled)
- Critical boundary section distinguishing this space from adjacent spaces
- Example outputs cleaned of cross-space leakage

This process was applied to 7 of 8 vector spaces: plot_analysis, plot_events,
viewer_experience, watch_context, narrative_techniques, production, and reception.
The anchor space (`dense_anchor_vectors`) was not realigned in this pass.

## Alternatives Considered

1. **Incremental patch of existing prompts**: Each prompt had multiple
   compounding misalignments. Patching individual issues risked introducing
   new inconsistencies. A full rewrite from the source of truth is more
   reliable.

2. **Single generic subquery prompt for all spaces**: Would lose space-specific
   format awareness entirely. Different spaces have fundamentally different
   embedded formats (prose vs. flat list vs. labeled fields), requiring
   space-specific guidance.

3. **Automated prompt generation from schema**: The schema fields are necessary
   but not sufficient — the prompt also needs realistic example terms, format
   descriptions, and boundary guidance that require human curation of real
   embedded data.

## Consequences

- All 7 realigned prompts are grounded in the actual embedded content format
  and field inventory as of the current schema versions.
- Any future schema change to generation prompts or `embedding_text()` must
  trigger a corresponding review of the affected search prompts.
- The `/realign-vector-search-prompts` command provides a reproducible process
  for future realignment passes.
- Prompts reference the QU cache version convention: bump `v{N}` prefix when
  any system prompt changes (not yet implemented, but the convention is now
  established in the prompt design).
- No code logic changed — all changes are prompt text only. Evaluation against
  query understanding test cases is needed to validate subquery quality.

## References

- `implementation/prompts/vector_subquery_prompts.py`
- `implementation/prompts/vector_weights_prompts.py`
- `.claude/commands/realign-vector-search-prompts.md`
- `movie_ingestion/final_ingestion/vector_text.py` (source of truth for embedded format)
- `schemas/metadata.py` (generation schemas — defines what `embedding_text()` produces)
- ADR-058 (vector text formatting conventions — establishes the labeled-field convention)
