# [028] — LLM Evaluation Pipeline Design for Metadata Generation

## Status
Active

## Context

Before committing a model/provider for each Stage 6 generation type
(~112K movies × estimated ~$150 total run cost), output quality must be
validated across multiple candidate models. ADR-027 established that
generators are real-time async callers; this decision governs how those
callers are evaluated systematically.

Key requirements: (1) a fixed reference quality bar to score against,
(2) idempotent reruns so partial runs can resume cheaply, (3) per-movie
per-dimension scores that can be queried/aggregated in SQL, (4) support
for testing different model params (reasoning depth, temperature) as the
primary differentiation axis.

## Decision

A **two-phase pointwise evaluation pipeline** in
`movie_ingestion/metadata_generation/evaluations/`.

**Phase 0** generates one gold-standard reference output per movie using
GPT-5.4 via the ChatGPT WHAM backend (see ADR-030). References are stored
once and reused across all candidate evaluations. *(Originally Claude Opus
via Anthropic OAuth; switched to GPT-5.4/WHAM for cost — see ADR-030.)*

**Phase 1** generates each candidate's output for every movie, then calls
a GPT-5.4 judge (via WHAM) that scores the candidate against the reference
on per-dimension rubrics. Scores are stored as individual INTEGER columns —
not JSON blobs — so SQL aggregation works per dimension.

**`EvaluationCandidate` frozen dataclass** (in `shared.py`) is the config
unit: `candidate_id`, `provider`, `model`, `system_prompt`,
`response_format`, `kwargs`. Candidates are defined per-metadata-type
(e.g., `PLOT_EVENTS_CANDIDATES` in `plot_events.py`) so the list stays
co-located with the rubric and table DDL it belongs to.

**Separate eval DB** at `evaluation_data/eval.db` (gitignored), not
co-located with tracker.db. WAL journal mode; each row is an API spend.

**Phase 1 calls `generate_llm_response_async` directly** (bypassing the
higher-level generator function) so candidate-specific system prompts are
honoured faithfully.

**Judge rubric is aligned to the generation prompt** — evaluates adherence
to what the generator was instructed to do (factual extraction, compactness,
character minimalism), not generic narrative quality.

## Alternatives Considered

1. **No reference model; pairwise comparisons between candidates**: Would
   require O(n²) judge calls and makes it harder to add candidates
   incrementally. A fixed reference anchors all scores to a single scale.

2. **Score dimensions as a JSON column**: Simpler schema, but prevents
   `SELECT AVG(groundedness_score) FROM ...` style queries. Per-column
   layout costs negligible schema complexity.

3. **Re-use the generator's system prompt for the judge**: A judge that
   grades against its own understanding of "good" may not reflect what
   the generation prompt asked for — leading to scores that reward
   narrative quality over factual groundedness. Rubric explicitly mirrors
   generation prompt instructions.

4. **Store candidates list in `shared.py`**: Rejected in favour of
   per-type files — shared.py has no knowledge of rubrics or schemas,
   so `PLOT_EVENTS_CANDIDATES` living in `plot_events.py` keeps the
   configuration next to the evaluation logic it configures.

## Consequences

- Every new metadata type evaluation requires: a candidates list,
  per-type table DDL, a judge schema, and Phase 0/1 implementations in
  a new `evaluations/<type>.py` file.
- `evaluation_data/eval.db` must be gitignored — it accumulates API spend
  and contains evaluation results not needed in source control.
- `check_plot_events()` and `check_reception()` are public in
  `pre_consolidation.py` so evaluation runners and the wave1_runner
  can call them directly to pre-filter ineligible movies before spend.
- Model selection for production commits should reference the
  `analyze_results.py` output (quality score + per-movie cost).

## References

- ADR-026 (multi-provider routing) — `generate_llm_response_async`
- ADR-027 (real-time generator contract) — why generators are real-time callers
- ADR-012 (LLM generation cost optimization) — model selection rationale
- movie_ingestion/metadata_generation/evaluations/ — full implementation
- docs/modules/ingestion.md (Model Evaluation section)
