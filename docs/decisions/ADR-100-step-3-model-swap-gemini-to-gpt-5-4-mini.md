# [100] — Step 3 model swap: Gemini 3.5 Flash → OpenAI gpt-5.4-mini

## Status
Active

## Context
ADR-090 set Step 3 (trait decomposition) on `gemini-3.5-flash` with
`thinking_level="low"`. ADR-096 then redesigned Step 3's prompt
architecture (aspects-as-parts, `CandidateFit`, `consolidation_analysis`)
to fix over-fragmentation of single-concept traits. While validating that
redesign in `search_v2/category_candidates_experiment/CONSOLIDATION_EXPERIMENT.md`,
we ran the full redesigned prompt/schema against two models on a held-fixed
Step 2 output: `fix_gemini` (Gemini 3.5 Flash) and `fix_gpt`
(gpt-5.4-mini, reasoning low / verbosity low).

The consolidation goal is decisive minimum-call-set selection — keep a
coherent trait as one SOLO call rather than splitting it into brittle
FACETS. On that axis gpt-5.4-mini at low/low consolidated single concepts
more decisively than Gemini, which more often left a residual facet split.
The subsequent category-definition audit round (`audit_gpt`) and Step-2
source-fix round (`s2fix_gpt`) were both run on gpt-5.4-mini and carried
the suitability-redundancy, audio-language-trap, and story-consolidation
wins, confirming the model choice held up as the prompt evolved.

## Decision
Swap Step 3's model from `gemini-3.5-flash` (`thinking_level="low"`) to
OpenAI `gpt-5.4-mini` with `reasoning_effort="low"`, `verbosity="low"`.
The model is locked as a module constant in `search_v2/step_3.py`
(`_PROVIDER` / `_MODEL` / `_MODEL_KWARGS`) — callers cannot override, which
keeps the step reproducible and cost/latency predictable end-to-end.

This supersedes the model-selection portion of ADR-090 (item 3). The other
two ADR-090 changes — the `category_candidates` schema floor (`min_length=5`)
and the prune-ruthlessly framing — remain in force.

## Alternatives Considered
- **Stay on Gemini 3.5 Flash with the redesigned prompt (`fix_gemini`)**:
  cleaner category hygiene on some cases, but consolidated less decisively —
  more residual FACET splits on traits that should land as one SOLO call,
  which is the exact failure mode ADR-096 set out to fix.
- **gpt-5.4-mini at higher reasoning effort**: rejected — low/low already met
  the consolidation bar; raising effort adds cost/latency for no measured gain
  on this step.

## Consequences
- Step 3 is now an OpenAI dependency in the V2 search-time critical path.
  Previously V2 search-time query understanding was described as Gemini-only
  (Steps 0/1/2); Step 3 breaks that — reflected in docs/PROJECT.md constraints.
- Provider/API surface differs from Steps 0/1/2: tuning Step 3 means OpenAI
  `reasoning_effort` / `verbosity`, not Gemini `thinking_level` / temperature.
- ADR-090's model decision is now historical; readers should treat ADR-090 as
  superseded on model selection only (schema floor + framing still active).
- Each provider has independent rate-limit / outage exposure, so a V2 search
  now depends on both Gemini and OpenAI availability.

## References
- `search_v2/step_3.py` (executor comment + `_PROVIDER`/`_MODEL`/`_MODEL_KWARGS`)
- `search_v2/category_candidates_experiment/CONSOLIDATION_EXPERIMENT.md` (validation: `fix_gpt`, `audit_gpt`, `s2fix_gpt`)
- ADR-090 (superseded on model selection), ADR-096 (Step 3 consolidation redesign)
- docs/PROJECT.md (LLM provider constraints), docs/modules/search_v2.md (Step 3)
