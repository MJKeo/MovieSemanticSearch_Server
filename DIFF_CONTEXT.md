# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Docs-auditor staleness fixes (17 findings)
Files: docs/modules/search_v2.md, docs/modules/schemas.md, docs/modules/ingestion.md, CLAUDE.md, docs/conventions.md, docs/PROJECT.md, docs/decisions/ADR-100-step-3-model-swap-gemini-to-gpt-5-4-mini.md (new), docs/decisions/ADR-090-step-3-loose5-coordinated-changes.md

### Intent
Ran /run-docs-auditor-agent (docs-auditor subagent), then applied every finding it surfaced. Goal: realign permanent docs with current code so future debugging/decisions aren't misdirected.

### Key Decisions
- **Step 3 model swap was the throughline behind most HIGH findings.** Code (`search_v2/step_3.py`) finalized Step 3 to OpenAI `gpt-5.4-mini` (reasoning low / verbosity low), swapped from `gemini-3.5-flash` after the consolidation experiments — but no doc reflected it. Fixed search_v2.md (2 spots), added the dependency to PROJECT.md constraints, authored ADR-100 to record the swap, and marked ADR-090's model item (3) superseded (schema floor + framing in ADR-090 remain active). Verified against `CONSOLIDATION_EXPERIMENT.md` (`fix_gpt`/`audit_gpt`/`s2fix_gpt`).
- **schemas.md**: corrected nonexistent `Atom.candidate_internal_split` → real fields `split_exploration` + `standalone_check`; fixed the `QueryAnalysis` gotcha (top-level fields are `intent_exploration`+`atoms`+`traits`, not `holistic_read`+`atoms`); added missing rows for `step_3.py`, `implicit_expectations.py`, `consolidate_award_categories.py`.
- **ingestion.md**: removed deleted-file row `backfill_awards_boxoffice.py`; fixed concept_tags reasoning effort (minimal, not medium); added `rebuild_character_postings.py`, `backfill_release_format.py`, `backfill_keyword_ids_to_qdrant.py`.
- **conventions.md dealbreaker floor**: the [0.5,1.0] floor invariant is retired. Verified only `trending_query_execution.py` still calls `compress_to_dealbreaker_floor`; semantic/metadata/award executors emit raw [0,1]. Country-of-origin pos 2 now emits raw 0.33 (no compression). Rewrote the bullet to match.
- **`embedded` status**: defined in `MovieStatus` enum + canonical chain but never set (ingest goes metadata_generated → ingested directly). Added clarifying "defined but currently unused" notes in CLAUDE.md and conventions.md rather than implying movies transit it.
- **CLAUDE.md**: test count 76 → 77.

### Process Note
PROJECT.md, conventions.md, and docs/decisions/ are normally never modified autonomously (per docs-awareness rule); the user explicitly authorized direct edits to these three for this audit-fix pass.

### Testing Notes
Docs-only changes — no code touched. Unit tests referencing `consolidation_analysis`/`CandidateFit` are unaffected (those names were already correct in code).

## Log CF-Connecting-IP on /health + enable INFO logging
Files: api/main.py, docs/modules/api.md
Why: Cloudflare now fronts the API; the socket peer is Cloudflare's edge, so the real client IP only lives in the CF-Connecting-IP header. Wanted /health to log it for observability.
Approach: Added a `request: Request` param to `health_check` and log `request.headers.get("CF-Connecting-IP")` (None when not behind CF, so safe locally). Also added `logging.basicConfig(level=logging.INFO)` at import time — uvicorn only configures its own `uvicorn.*` loggers, leaving root at WARNING with no handler, so app-module INFO logs were being dropped entirely. This fixes visibility for the whole API, not just this line. Output goes to container stderr (`docker compose logs -f api`).
Testing notes: No unit coverage added (logging side effect). Verify manually by hitting /health through Cloudflare and tailing the api container logs.
