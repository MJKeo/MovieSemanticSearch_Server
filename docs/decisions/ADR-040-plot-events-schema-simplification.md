# [040] — plot_events Schema Simplification After 42-Movie Evaluation

## Status
Active

## Context

ADR-033 designed `PlotEventsOutput` with three fields: `plot_summary`,
`setting`, and `major_characters` (each `MajorCharacter` having `name`,
`role`, and `motivation`). A 42-movie evaluation across 7 size-based
buckets revealed two independent quality problems:

**Problem 1 — Hallucination in Branch A (synopsis condensation):**
The `MIN_SYNOPSIS_CHARS` gate was set to 1,000. At ~1K char synopses,
hallucination rate was 67% — the model fills gaps from training knowledge
when input is too thin. At 4K+ char synopses: 0% hallucination. The
2,500 char threshold is where synopses are consistently detailed enough
for faithful condensation.

**Problem 2 — `plot_keywords` acting as hallucination springboards:**
Keywords like "intersex" (in Bol) triggered fabrication of entire character
arcs not present in the text. Keywords like "Oreos" (in Drunks) were
treated as narrative events. The overview already provides high-level
framing; keywords added noise and hallucination surface area.

**Problem 3 — Structural output fields failing consistently:**
- `setting` exceeded the ≤10 word constraint 83% of the time, producing
  verbose location descriptions that competed with `plot_summary` content.
- `major_characters` with motivation/role fields added analytical burden
  to what is fundamentally a text condensation task — the model had to
  simultaneously condense plot and extract structured character analysis,
  which degraded both tasks. Character names appear naturally in
  `plot_summary`; motivation analysis belongs in downstream `plot_analysis`.

## Decision

Three coordinated changes to `generate_plot_events()` and its schema:

1. **Raise `MIN_SYNOPSIS_CHARS` from 1,000 to 2,500.** Synopses below
   this threshold are demoted into the summaries list and routed to
   Branch B (synthesis), which has 0% hallucination at all input sizes.

2. **Remove `plot_keywords` from both branch inputs.** Neither branch
   prompt receives keywords. The overview provides sufficient high-level
   framing without introducing hallucination risk.

3. **Remove `setting` and `major_characters` from `PlotEventsOutput`.**
   Output is now `plot_summary` only. `MajorCharacter` schema class is
   no longer used by `plot_events` (retained if needed by other generators).

The `_MIN_PLOT_TEXT_CHARS = 600` eligibility gate in `pre_consolidation.py`
is unchanged — it governs whether any substantial plot text exists, not
which branch to use.

## Alternatives Considered

1. **Keep `setting` with a stricter prompt constraint**: 83% failure rate
   suggests the field is fundamentally misaligned with the condensation
   task, not just under-constrained. Dropping it entirely is cleaner.

2. **Keep `major_characters` without motivation/role fields (name only)**:
   Character names already appear in `plot_summary` naturally. A redundant
   name list adds schema complexity without distinct value. Dropped.

3. **Keep `plot_keywords` with a denylist for problematic keywords**:
   The hallucination pattern is general (any evocative keyword can
   trigger knowledge recall), not limited to specific keywords. A denylist
   would need ongoing maintenance. Dropping keywords is simpler and
   eliminates the risk class.

4. **Lower `MIN_SYNOPSIS_CHARS` to 1,500 instead of 2,500**: Insufficient.
   The 67% hallucination rate extends through the 1K–2K range; the clean
   transition is at ~4K, with 2,500 as a practical middle point that captures
   the quality improvement without over-restricting the synopsis branch.

5. **Update `implementation/classes/schemas.py`**: Left unchanged — these
   are search-side schemas. The decision to diverge generation-side from
   search-side schemas is established in ADR-025. Alignment deferred to
   deployment.

## Consequences

- `PlotEventsOutput` contains only `plot_summary`. Unit tests referencing
  `setting`, `major_characters`, or `MajorCharacter` on `PlotEventsOutput`
  will fail and must be updated.
- Branch A now routes ~1K–2.5K char synopses to Branch B. The synopsis
  branch population shrinks; synthesis branch grows. Quality improves
  across both branches.
- `plot_keywords` is no longer passed to `build_plot_events_prompts()`.
  The keyword routing table in `pre_consolidation.py` is updated to
  reflect this (plot_keywords → other Wave 2 generators only).
- The 4-dimension judge rubric used during evaluation (groundedness,
  plot_summary, character_quality, setting) is now historical — the
  `character_quality` and `setting` dimensions are no longer applicable
  to the production schema.

## References

- ADR-033 (plot_events two-branch design) — original schema and branch gate
- ADR-038 (600-char eligibility gate) — separate eligibility gate unchanged
- ADR-039 (gpt-5-mini model selection) — evaluated alongside these schema changes
- `movie_ingestion/metadata_generation/generators/plot_events.py`
- `movie_ingestion/metadata_generation/schemas.py`
- `movie_ingestion/metadata_generation/prompts/plot_events.py`
