# [048] — Narrative Techniques: 11→9 Section Consolidation

## Status
Active

## Context

After R3 evaluation (56 movies, 7 candidates, gpt-5-mini with minimal
reasoning as production target), analysis of failure patterns identified
two sections that consistently underperformed across all candidates:

- `thematic_delivery`: the top hallucination source. Models over-inferred
  delivery mechanisms from theme topics — e.g., generating "metaphor" because
  the movie has a thematic idea about identity, not because there is textual
  evidence of a metaphorical device. Failed the FIRST-gating check at
  unusually high rates even on rich-input movies.
- `meta_techniques`: empty for most movies (fourth-wall breaks and genre
  deconstruction are rare). The section incurred section-discipline penalties
  for normal films that simply don't use self-aware devices.

Additionally, `additional_plot_devices` (section 3 in the 11-section schema)
was a broad catchall, but its early position caused the model to deposit
terms there before attempting more specific sections.

## Decision

Reduce to 9 sections by:

1. **Removing `thematic_delivery`** entirely — the section cannot be reliably
   grounded from the available inputs and its retrieval signal is already
   covered by `plot_analysis` vectors.
2. **Merging `meta_techniques` into `additional_narrative_devices`** — the
   merged section captures the rare meta/self-aware devices as "additional
   narrative devices" alongside other plot-level mechanics.
3. **Renaming `additional_plot_devices` → `additional_narrative_devices`**
   to reflect the expanded scope.
4. **Moving the catchall section to last position (section 9)** so the model
   fills specific sections first and deposits remaining terms into the catchall.
5. **Trimming preamble reusability examples from 8 to 4** — the model already
   scores 4.0 on `technique_abstraction`, so example reduction reduces prompt
   tokens without quality cost.

The ingestion-side schemas (`NarrativeTechniquesOutput`,
`NarrativeTechniquesWithJustificationsOutput` in
`movie_ingestion/metadata_generation/schemas.py`) and the search-side
schema (`NarrativeTechniquesMetadata` in `implementation/classes/schemas.py`)
were both updated to 9 sections with matching field names and order.

## Alternatives Considered

1. **Keep `thematic_delivery` but add a stricter gating instruction**: Tried
   in prior evaluation rounds with tighter prompt language. Models continued
   over-generating for this section — the issue is the section's conceptual
   ambiguity with theme topics, not prompt clarity.

2. **Keep `meta_techniques` as a separate empty-allowed section**: The section
   was nearly always empty, adding schema tokens without retrieval benefit.
   Merging into a catchall preserves the rare valid uses without a dedicated
   section tax.

3. **Reorder `additional_narrative_devices` to earlier position**: Rejected.
   Early catchall placement is the failure mode — models use it as a default
   rather than filling specific sections first.

## Consequences

- Existing evaluation data (R1-R3) was collected on the 11-section schema
  and cannot be directly compared to 9-section output; a fresh evaluation
  pass is needed.
- Test files referencing the old field names (`thematic_delivery`,
  `meta_techniques`, `additional_plot_devices`) need updating.
- The search-side schema (`NarrativeTechniquesMetadata`) has been updated
  to match the 9-section generation-side schema with identical field
  names and ordering.

## References

- ADR-045 (Wave 2 finalization pattern)
- `movie_ingestion/metadata_generation/schemas.py`
- `movie_ingestion/metadata_generation/prompts/narrative_techniques.py`
- `implementation/classes/schemas.py`
