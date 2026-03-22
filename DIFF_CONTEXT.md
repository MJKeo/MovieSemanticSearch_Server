# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Fix stale documentation from audit

Files: `docs/modules/ingestion.md`, `docs/conventions.md`, `docs/decisions/ADR-028-llm-evaluation-pipeline-design.md`, `docs/decisions/ADR-016-combined-imdb-quality-scorer.md`, `docs/decisions/ADR-033-plot-events-cost-optimization.md`, `docs/decisions/ADR-034-reference-free-evaluation-with-opus-judge.md`, `CLAUDE.md`, `docs/PROJECT.md`

### Intent
Fix all STALE findings from a full docs audit.

### Key Fixes
- Renamed `build_plot_events_user_prompt` → `build_plot_events_prompts` in 4 docs (ingestion.md, ADR-033, ADR-034, conventions.md)
- Corrected ingestion.md claim that eval winner is "set as production default" — it is not; callers must pass provider/model explicitly
- Updated test file count from 27 → 57 in CLAUDE.md and PROJECT.md
- Replaced v1 Stage 5 weight table with v4 weights in CLAUDE.md
- Marked ADR-028 as Superseded by ADR-034 (for plot_events eval), noting structural design remains active
- Added model evolution note to ADR-016 pointing to ADR-019/ADR-021 and current code

## Tighten check_plot_events eligibility to 600-char minimum
Files: `movie_ingestion/metadata_generation/pre_consolidation.py`
Why: Previous check was too lenient — movies with very short plot text produced low-quality plot_events output.
Approach: Replaced the three-source existence + sparseness check with a single rule: longest text among first synopsis and all summaries must be >= 600 chars. Removed unused `_all_text_sources_sparse()` helper and old per-source threshold constants.
