# [038] — plot_events Eligibility: 600-Char Minimum on Longest Text Source

## Status
Active

## Context

The original `check_plot_events()` eligibility check allowed generation
to proceed as long as at least one text source existed (synopsis, summaries,
or overview) and was not sparse (overview >= 10 chars, synopsis >= 50 chars,
combined summaries >= 50 chars). In practice this permitted movies with
very short plot text to enter generation, producing low-quality plot_events
output — sparse inputs yielded truncated or unreliable outputs from both
the synopsis-condensation and synthesis branches.

The overview source was included in the old check, contributing a path
to eligibility despite being consistently too brief (~53 tokens avg per
ADR-033) to anchor meaningful plot event extraction.

## Decision

Replace the multi-source existence + sparseness check with a single rule:
eligible if and only if the longest text among the first synopsis entry
(if any) and all `plot_summaries` entries is >= 600 chars
(`_MIN_PLOT_TEXT_CHARS = 600`).

Overview is explicitly excluded from the measurement — it is too short
to support plot event extraction on its own and its inclusion in the
old check created a false eligibility path.

The old multi-constant approach (`_MIN_SYNOPSIS_CHARS_FOR_SPARSE`,
`_MIN_SUMMARIES_COMBINED_CHARS`, `_MIN_OVERVIEW_CHARS`) and the
`_all_text_sources_sparse()` helper are removed entirely.

## Alternatives Considered

1. **Keep overview as a candidate source with a higher threshold**: Overview
   averages ~53 tokens; even at 600 chars it would admit very few additional
   movies, and the overview content is insufficient for plot event extraction
   (marketing blurbs, not plot recounts). Excluded.

2. **Lower the threshold (e.g., 300 chars)**: Would admit more movies but
   contradicts the observed quality problem — short text produces poor
   output. 600 chars is a meaningful lower bound for a plot-bearing text.

3. **Per-branch thresholds**: Branch A (synopsis condensation) already has
   its own `MIN_SYNOPSIS_CHARS = 1000` gate in the generator itself.
   A separate eligibility-level threshold using a single rule is simpler
   and avoids duplicating branching logic in the eligibility layer.

## Consequences

- Movies that have only an overview or only very short summaries/synopsis
  are now filtered out before any LLM cost is incurred.
- The eligibility check is simpler: one constant, one comparison, no
  per-source logic.
- Coverage is reduced for movies with sparse text, but these movies
  produced low-quality output anyway — the trade-off favors precision
  over recall in plot_events generation.
- Evaluation runners and `wave1_runner` that call `check_plot_events()`
  to pre-filter the corpus will now apply the stricter gate automatically.

## References

- ADR-033 (plot_events two-branch design) — covers `MIN_SYNOPSIS_CHARS`
  branch gate and the overview-exclusion rationale
- `movie_ingestion/metadata_generation/batch_generation/pre_consolidation.py` —
  `check_plot_events()`, `_MIN_PLOT_TEXT_CHARS`
