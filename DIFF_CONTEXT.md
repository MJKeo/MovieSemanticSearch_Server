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

## OTel smoke-test script (observability Phase 1a validation)
Files: scripts/otel_smoke_test.py | Standalone script that emits one hand-made parent+child span via the OTLP gRPC exporter to prove the local `grafana/otel-lgtm` backend (OTLP :4317, Grafana :3000) is reachable and ingesting before instrumenting the real app. Loads `.env` for OTEL_EXPORTER_OTLP_ENDPOINT. Verified: trace landed in Tempo (service.name=otel-smoke-test, 2 spans). Next: OTel SDK bootstrap + auto-instrumentation in api/main.py (observability_context/observability_todos.md Phase 1b).

## OTel tracing bootstrap module (observability Phase 1b)
Files: observability/__init__.py (new), observability/tracing.py (new)
Why: Central, one-time OTel setup so the API emits traces; auto-instruments the four network clients actually used by this stack.
Approach: `setup_tracing(app)` builds a TracerProvider (service.name/deployment.environment via env, defaults cinemind-api/local), a BatchSpanProcessor + env-configured OTLP gRPC exporter, then enables FastAPI (instrument_app on the instance), httpx, psycopg (v3), and redis auto-instrumentation. Corrected the plan's "asyncpg" to psycopg-v3 instrumentor (db uses AsyncConnectionPool). Idempotent via a module guard (reload/re-import safe). Calls load_dotenv() (non-overriding) so a bare uvicorn run resolves OTEL_EXPORTER_OTLP_ENDPOINT from .env. Added observability/__init__.py to match the majority package convention (implementation/schemas/search_v2 have one; db does not). Endpoint is pure config so local→Grafana Cloud is env-only.
Design context: observability_context/initial_implementation_context.md (config-not-code; one trace/one store). Qdrant gRPC deliberately not auto-instrumented — covered later by a manual vector-search span.
Testing notes: Module imports cleanly (verified). Not yet wired into api/main.py (Step 5) — no runtime effect until setup_tracing(app) is called. Container path deferred: needs the OTel packages in api/requirements.txt and observability/ mounted in docker-compose api service.

## Wire OTel tracing into API startup (observability Phase 1b, Step 5)
Files: api/main.py | Import `setup_tracing` and call `setup_tracing(app)` immediately after `app = FastAPI(lifespan=lifespan)` — runs at import time, before serving, so every request is a root span. py_compile passes.
Caveat (host vs container): this is a HOST-run path (`uv run uvicorn`). The docker-compose `api` service will now crash on reload — `observability/` is not volume-mounted and api/requirements.txt lacks the OTel packages (confirmed via container log: `ModuleNotFoundError: No module named 'observability'`). Making the container tracing-ready is deferred Phase 5: add OTel deps to api/requirements.txt + rebuild, add `./observability:/app/observability` mount, set endpoint to host.docker.internal:4317 (or add otel-lgtm to the compose network). Also note redis container is currently down (exited) — graceful-degradation means requests still succeed (cold cache), but no redis spans will appear until it's started.
