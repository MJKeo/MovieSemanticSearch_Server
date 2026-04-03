# [046] — Viewer Experience: GPO-Only Narrative Input with Tier-1 Input Pruning

## Status
Active

## Context

After viewer_experience production config selection (Round 2: gpt-5-mini with
justifications, $0.00246/movie), Round 3 tested two independent simplification
hypotheses:

1. **GPO-only narrative**: Replace the full fallback chain
   (plot_summary → raw synopsis fallback → generalized_plot_overview) with
   GPO alone. Hypothesis: two layers of LLM abstraction strip noise while
   preserving the thematic/emotional core viewer_experience needs, making
   raw synopsis text actively harmful by grounding the model in concrete
   plot details rather than felt-experience vocabulary.

2. **Tier 1 input pruning**: Remove merged_keywords (~57 tokens) and
   character_arcs (~14 tokens). Both had <2% citation rate in justification
   analysis of 400 section-level justifications from the Round 2 production
   candidate.

Round 3 ran 50 movies × 6 candidates (baseline + 4 ablations + caveman variant)
using the full 7-axis rubric.

## Decision

**GPO-only narrative**: Replace the fallback chain with `generalized_plot_overview`
as the sole narrative input. Round 3 results: GPO-only scored +0.046 over
baseline across all buckets. This simplifies eligibility (removes plot_summary
and raw_fallback paths) and reduces dependency on Wave 1 plot_events outputs
in the viewer_experience pipeline.

**Tier 1 pruning (merged_keywords + character_arcs removed)**: tier1-pruned
scored +0.031 over baseline with the tightest consistency (stdev 0.160).

**Tier 2 inputs kept (thematic_observations + genre_context)**: tier1-tier2-pruned
dropped -0.069 overall, concentrated in the floor_plot_summary bucket (-0.268).
Thematic observations compensate for thin narrative in that population — the
citation rate understated their actual contribution.

**Caveman justifications rejected**: Terse justifications caused -0.44
specificity and -0.38 term diversity drops. Full-sentence justifications
generate richer vocabulary during reasoning.

## Alternatives Considered

1. **Full fallback chain**: Retains raw synopsis text. Results showed raw
   fallback (B4: 4.21) performed worse than observation-standalone (B6/B7:
   4.60-4.62). Concrete plot text anchors the model to events rather than
   felt experience.

2. **Remove tier 2 inputs (thematic + genre context)**: Marginally better
   citation rate data, but -0.268 on floor_plot_summary bucket is a
   meaningful quality regression for that population.

3. **No narrative input (observations only)**: Not tested directly, but
   B6/B7 results (obs-standalone with minimal context) scored higher than
   B4 (raw fallback with obs), suggesting the GPO is providing genuine lift
   beyond observations alone.

## Consequences

- `viewer_experience` eligibility is simplified to 3 paths:
  GPO >= 350 standalone / obs standalone / GPO >= 200 + obs.
- `plot_summary` and `character_arcs` params removed from
  `generate_viewer_experience()` — callers updated.
- Viewer_experience no longer depends on raw plot fallback quality; its
  quality is determined entirely by GPO and observation richness.
- watch_context uses a different input contract and is NOT affected by
  this decision.

## References

- ADR-045 (finalization pattern) — general Wave 2 finalization approach
- `movie_ingestion/metadata_generation/generators/viewer_experience.py`
- `movie_ingestion/metadata_generation/batch_generation/pre_consolidation.py`
- `ingestion_data/viewer_experience_eval_guide.md`
