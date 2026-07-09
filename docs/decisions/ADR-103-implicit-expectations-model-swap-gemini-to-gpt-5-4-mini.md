# [103] — Extend gpt-5.4-mini swap to implicit_expectations

## Status
Active (drafted by docs-maintainer from DIFF_CONTEXT — pending human review)

## Context
ADR-100 swapped Step 3 (trait decomposition) from `gemini-3.5-flash` to
OpenAI `gpt-5.4-mini` (`reasoning_effort="low"`, `verbosity="low"`) after
empirical validation showed more decisive minimum-call-set consolidation.
That left `search_v2/implicit_expectations.py` (the implicit quality/
popularity prior policy LLM, run in parallel with Step 3 inside the
`decomposition` span) as the last per-branch executor still on Gemini
(`gemini-3.5-flash`, `thinking_budget=0`, temperature 0.35) — a third
distinct provider/config combination alongside Steps 0-2 (Gemini) and
Step 3 / entity-query-generation (OpenAI gpt-5.4-mini).

## Decision
Swap `implicit_expectations.py`'s executor to the same OpenAI
`gpt-5.4-mini` / `reasoning_effort="low"` / `verbosity="low"` configuration
used by Step 3 and the entity/query-generation callsites. The full
orchestrator calls `run_implicit_expectations` without a model override,
so this is a single-module constant edit with no caller impact.
`gpt-5.4-mini` was already priced in `pricing.py`.

## Alternatives Considered
- **Stay on Gemini 3.5 Flash.** Rejected on consistency grounds — no
  quality regression was reported on Gemini, but running a fourth
  distinct model config for one small per-branch call added tuning
  surface for no measured benefit. Unlike ADR-100, this swap was not
  driven by a measured quality gap; it's a consistency-motivated
  follow-on.

## Consequences
- `implicit_expectations` now shares OpenAI's rate-limit/outage exposure
  with Step 3 and entity/query-generation, deepening (not newly
  introducing) the dual-provider dependency ADR-100 already flagged.
- One fewer distinct model configuration to tune across the
  standard-branch pipeline.

## References
- ADR-100 (the precedent this extends — Step 3's Gemini → gpt-5.4-mini swap)
- `search_v2/implicit_expectations.py`, `search_v2/step_3.py`
- `docs/modules/search_v2.md` (Key Files: `implicit_expectations.py`)
