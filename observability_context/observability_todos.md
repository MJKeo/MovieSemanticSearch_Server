# Observability TODOs

Implementation tracker for adding full observability to the movie search
backend. Derived from `observability_logs_plan.md` — read that first for the
*why* behind each decision, and `initial_implementation_context.md` for the
finalized decisions and standing guidelines. This file tracks the *what* and
*in what order*.

**Decisions locked 2026-07-03** (see `initial_implementation_context.md`):
OTel instrumentation + Grafana stack (local `otel-lgtm`, prod Grafana Cloud);
one trace / one store; LLM `gen_ai.*` attributes fold into the traces step
(not a separate phase); Honeycomb evaluated and rejected; TTFT dropped.

**Status legend:** `[ ]` not started · `[~]` in progress · `[x]` done ·
`[?]` blocked on an open question (see bottom).

**Guiding constraints (don't lose these):**
- Single EC2 t3.large (2 vCPU, 8GB RAM) already runs Postgres + Redis +
  Qdrant + API. Any self-hosted telemetry component competes for memory —
  prefer shipping telemetry off-box (Grafana Cloud / Langfuse Cloud free tiers).
- Instrument once with OpenTelemetry; keep all backend choices swappable via OTLP.
- LLM spans and pipeline spans must live in **one trace** (single trace ID).
- Cost is priority #3: free tiers + open source; no per-seat/per-host pricing.

---

## Prerequisites / Decisions to lock before building

- [ ] **Resolve Open Question #2** — app logs land in Loki vs PostHog logs
      (don't double-ship). Blocks Phase 4.
- [ ] Confirm Grafana Cloud free-tier account is provisioned (10k metric
      series, 50GB logs/traces) and OTLP endpoint + API token obtained.
- Open Question #1 (Phoenix vs Langfuse) is **no longer a build blocker** —
  the LLM-native tool is deferred and decoupled (see Phase 2). LLM
  `gen_ai.*` attributes now land in Phase 1 with the rest of the traces.
- [ ] Confirm the exact set of search-time LLM providers in use
      (`implementation/llms/generic_methods.py`: OpenAI, Kimi/Moonshot, Gemini,
      Groq, Alibaba/Qwen, Anthropic, WHAM) so each gets consistent span
      attributes.

---

## Phase 1 — Traces first (highest-value for latency goal)

Goal: a trace waterfall of one search request, viewable locally in
`grafana/otel-lgtm`, showing every network hop and every pipeline stage —
**including LLM calls carrying `gen_ai.*` attributes** (tokens, model, cost),
all resolving to one trace. The LLM attributes are part of this phase, not a
separate one (see `initial_implementation_context.md`, decision #4).

**Status (2026-07-03):** 1a + 1b are **complete and verified** — auto-instrumentation
(FastAPI, httpx, psycopg v3, redis) produces nested child spans in a real trace, viewed
locally in Tempo. **1c (manual pipeline spans + `gen_ai.*` attributes) is the remaining
Phase 1 work** — and the high-value part for the latency goal. Implementation entries are
in `../DIFF_CONTEXT.md`.

### 1a. Local telemetry backend
- [x] Stand up the `grafana/otel-lgtm` all-in-one container locally (Loki + Grafana +
      Tempo + Prometheus), kept out of the prod compose. *(Currently a bare `docker run`
      on :3000 UI / :4317 gRPC / :4318 HTTP; folding it into a committed dev-only compose
      file is optional leftover.)*
- [x] Document the local OTLP endpoint (gRPC `:4317` / HTTP `:4318`) and the Grafana UI
      port (`:3000`, admin/admin). *(Recorded in `../DIFF_CONTEXT.md`.)*
- [x] Verify Tempo receives a hand-emitted test span before instrumenting real code.
      *(`scripts/otel_smoke_test.py` → trace confirmed in Tempo via the Grafana proxy.)*

### 1b. OTel SDK + auto-instrumentation
- [x] Add dependencies via UV (`uv add`): `opentelemetry-sdk`,
      `opentelemetry-exporter-otlp-proto-grpc`, and instrumentation packages for
      FastAPI, httpx, **psycopg v3**, and redis. *(Corrected from the plan's "asyncpg" —
      the DB layer is `psycopg_pool.AsyncConnectionPool`, not asyncpg.)*
- [x] Create a single tracing bootstrap module (`observability/tracing.py`):
      configures `TracerProvider`, resource attributes (service.name via
      `OTEL_SERVICE_NAME`, deployment.environment), OTLP gRPC exporter, and a
      BatchSpanProcessor. Idempotent guard for `--reload` re-imports.
- [x] Wire the bootstrap into API startup (`api/main.py`) — `setup_tracing(app)` runs at
      import, right after the app is built, before it serves traffic.
- [x] Enable auto-instrumentation for FastAPI (request spans), httpx (LLM +
      TMDB + proxy calls), **psycopg v3** (Postgres), redis (cache). Confirmed via Tempo:
      all four fire as nested child spans of the request span (redis verified on `/health`
      + `/movie_details`). Qdrant deliberately excluded — its async client is gRPC; that
      timing comes with the 1c manual vector-search span.
- [x] Set OTLP endpoint via env var (`OTEL_EXPORTER_OTLP_ENDPOINT`) so
      local vs EC2 is a config change, not a code change. *(Set in `.env`.)*

### 1c. Manual spans around pipeline stages
Add explicit spans (child of the FastAPI request span) around each meaningful
internal stage. Reference the pipeline in CLAUDE.md / `db/` search orchestration.
- [ ] Query Understanding parent span, with a child span per parallel LLM call
      (so per-provider tail latency is visible — the expected first finding).
- [ ] On each LLM call span, add `gen_ai.*` semantic-convention attributes:
      `gen_ai.system` (provider), `gen_ai.request.model`, input/output token
      counts, and computed dollar cost. (Full moved-up detail in Phase 2's
      attribute list — do the attributes here; skip the standalone LLM tool.)
      Skip TTFT — we don't stream; use output_tokens vs. span duration instead.
- [ ] Gate prompt/response payload capture behind a config flag; capture 100%
      now but make it a dial-able sample rate, not an on/off boolean. Put large
      payloads on span **events**, not attributes.
- [ ] Lexical search (Postgres entity matching).
- [ ] Vector search (Qdrant, across the 8 vector spaces) — consider a span per
      vector space or at least the 5-stage scoring pipeline boundaries.
- [ ] Metadata scoring (in-memory structured attribute scoring).
- [ ] Score merging.
- [ ] Quality reranking.
- [ ] Display-metadata fetch.
- [ ] Confirm parallel stages actually render as overlapping spans in the
      Tempo waterfall (validates that asyncio fan-out is truly concurrent).

### 1d. Validation
- [x] Run a representative search locally; open the trace in Grafana/Tempo.
      *(`POST /query_search` trace viewed end to end.)*
- [x] Sanity-check that a single trace ID spans the whole request end to end.
- [~] Capture the first latency finding (which stage / which provider dominates).
      *(Auto-level finding: a `query_search` ran ~47s with httpx/LLM spans dominating the
      visible time. Full per-stage / per-provider attribution awaits the 1c manual spans +
      `gen_ai.*` attributes.)*
- [x] Confirm redis auto-instrumentation nests under a real request. Verified on `/health`
      and `/movie_details` (two-run cache miss→hit test showed nested redis GET/SET spans).
      `query_search` currently does **not** hit redis (no caching on that path), so its
      trace has no redis span — this is **expected, not an instrumentation gap**. Revisit
      if/when query_search caching is added.

---

## Phase 2 — LLM-native tool (DEFERRED / decoupled — optional)

The `gen_ai.*` attributes themselves moved to Phase 1. This phase is now
**only** about adding a dedicated LLM-observability *viewer* (Phoenix or
Langfuse) for LLM-native UX: chat-style prompt/response rendering,
token/cost dashboards, and eval scoring. It is **not on the critical path**
— defer until we actually want that UX. Adding it later is a Collector
fan-out (the same trace goes to a second exporter), i.e. a config change,
not a rewrite.

**Trigger to revisit:** when Grafana's generic attribute view stops being
enough for LLM debugging/evals. Until then, LLM spans live in Tempo with
everything else.

- [?] **Decide Phoenix vs Langfuse Cloud** (Open Question #1) — only when this
      phase is triggered; no longer blocks earlier work.
- [ ] Stand up the chosen backend:
  - [ ] *If Phoenix:* run the single Phoenix container (fits hardware / self-host instinct).
  - [ ] *If Langfuse Cloud:* create project, obtain keys, point OTLP exporter at it.
- [ ] Add GenAI semantic-convention attributes to each LLM call span:
  - [ ] `gen_ai.system` (provider), `gen_ai.request.model`.
  - [ ] Input/output token counts (`gen_ai.usage.input_tokens` / `output_tokens`).
  - [ ] Dollar cost (computed per provider/model pricing).
  - [ ] Prompt version identifier.
  - [ ] Time-to-first-token.
  - [ ] Full prompt + response payloads (respect the coding-standards rule:
        never log secrets/PII — confirm prompts/responses are safe to store).
- [ ] Ensure LLM spans and pipeline spans resolve to **one trace** in the
      chosen viewer (the payoff of everything speaking OTel).
- [ ] Add a per-provider cost/latency view or dashboard.
- [ ] (Later) reserve span attributes / structure for eval scores.

---

## Phase 3 — Metrics (RED + USE)

Goal: dashboards + a small number of alerts. Percentiles, never averages.

### 3a. RED metrics per endpoint
- [ ] Rate (req/s), Errors (failure %), Duration (p50/p95/p99) per API endpoint.
- [ ] Confirm these derive from FastAPI auto-instrumentation metrics or add a
      metrics exporter (`opentelemetry` metrics + Prometheus scrape / OTLP).

### 3b. USE metrics for the EC2 box
- [ ] Utilization / Saturation / Errors for CPU, memory, disk.
- [ ] Connection-pool saturation (asyncpg pool, redis pool).
- [ ] **Memory saturation is the most likely silent killer on 8GB** — make sure
      host memory is captured (node/host metrics exporter on the box).

### 3c. Dashboards + alerts
- [ ] Grafana dashboard: RED per endpoint + USE for the box.
- [ ] Alert: p95 search latency threshold.
- [ ] Alert: error rate threshold.
- [ ] Alert: memory saturation threshold.

---

## Phase 4 — Structured logs

Goal: JSON logs correlated to traces via trace ID.

- [?] **Decide log destination** (Open Question #2): Loki (Grafana Cloud) vs
      PostHog logs — pick one, don't double-ship.
- [ ] Switch application logging to structured JSON.
- [ ] Inject the active trace ID (and span ID) into every log record so you can
      jump from a slow trace to its logs.
- [ ] Ship logs to the chosen destination.
- [ ] Verify trace→logs navigation works end to end in the UI.

---

## Phase 5 — Production rollout (EC2)

Goal: same instrumentation, telemetry shipped off-box to free-tier cloud.

- [ ] Point the OTLP exporter at Grafana Cloud (traces + metrics) via env vars —
      no application code changes from Phase 1–4.
- [ ] Point the LLM layer at its production target (Phoenix on-box or Langfuse Cloud).
- [ ] Confirm telemetry egress adds negligible memory/CPU on the t3.large
      (batch span processor, sane export intervals).
- [ ] Add host-metrics exporter on the EC2 box for USE metrics.
- [ ] Re-verify a real production search produces a complete single-trace waterfall.
- [ ] Set head-sampling config (fine at current volume — see Open Question #3).

---

## Phase 6 — Frontend / PostHog (deferred until frontend work begins)

Not yet designed — no frontend context in the plan. When frontend work starts:
- [ ] PostHog for user events (searches issued, result clicks, zero-result queries).
- [ ] Session replay, web vitals, feature flags.
- [ ] Backend product events → PostHog for funnel analysis (search events/outcomes).
- [ ] Decide whether app logs live in PostHog or Loki (tie back to Open Question #2).

---

## Open Questions (carry forward — decide when relevant)

1. [ ] Phoenix (self-hosted single container) vs Langfuse Cloud (zero footprint,
       bigger community) for the LLM-native viewer. *(No longer blocks the
       build — deferred/decoupled; decide only when Phase 2 is triggered.)*
2. [ ] App logs land in Loki (Grafana Cloud) vs PostHog logs — pick one.
       *(Blocks Phase 4.)*
3. [ ] Sampling strategy as traffic grows — head sampling fine now; revisit
       tail-based sampling only if telemetry volume becomes a cost concern.
4. [ ] Trace ingestion-pipeline runs (`movie_ingestion/`) too, or only the
       search-time path? Search-time first; ingestion is batch with tracker.db
       already giving progress visibility.
5. [ ] Sentry for error tracking — complementary; decide if/when error volume
       justifies it.
6. [ ] LLM payload capture rate at scale — capture 100% now, but agree on the
       sample rate to dial down to (leaning always-on-error + 1–5% random)
       once traffic/storage makes 100% a concern. Related to #3.
7. [ ] Datadog purely for the resume line — the one tool with more job-listing
       presence than Grafana, but can't be learned free. Decide if it's ever
       worth paying for a Pro account just to claim it. *(Current lean: no —
       OTel's vendor-neutrality already signals Datadog-readiness.)*
