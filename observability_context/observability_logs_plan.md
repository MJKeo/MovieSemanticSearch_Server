# Observability & Logs Plan

Planning context for adding observability to the movie search backend
(and eventually the frontend).

**Status (updated 2026-07-03):** the **initial approach and tooling are now
finalized** (marked ✅ below). Downstream layers (LLM-native viewer, log
destination, frontend) remain proposals. Companion docs:
`initial_implementation_context.md` (locked decisions + standing guidelines),
`observability_todos.md` (the ordered build checklist + per-item status), and
`observability_architecture.md` (what has actually shipped). **Phase 1 is
partially built:** OTel bootstrap + auto-instrumentation and manual spans for
`/title_search`, `/movie_details`, `/movie_credits` are done; the NLP-pipeline
endpoints (`/query_search` et al.) and everything after Phase 1 are not.

## Goals

1. **Debug latency** — the search pipeline is multi-stage and heavily
   parallel (query understanding fans out LLM calls; retrieval fans out
   lexical/vector/metadata channels). We need per-stage, per-call
   timing visibility to find where time actually goes. Latency is
   priority #2 in docs/PROJECT.md.
2. **Learn resume-grade, industry-standard tooling** — tool choices must
   teach transferable skills credible on a job application. Concretely:
   **OpenTelemetry** (the CNCF-graduated, vendor-neutral instrumentation
   standard — the marquee skill) plus **Grafana + Prometheus** (the most
   marketable observability backend learnable free and to depth). Not
   bespoke one-offs.
3. **LLM-specific observability** — token usage, cost, per-provider
   latency, prompt/response payloads for search-time LLM calls.

## Conceptual Model (the industry framing)

Observability is organized around three signal types:

- **Traces** — one request's journey through the system, broken into
  timed spans. The primary tool for latency debugging: turns "search
  took 2.3s" into a waterfall showing which stage ate the time and
  whether parallel calls actually ran in parallel.
- **Metrics** — cheap aggregated numbers over time (request rate,
  error rate, latency percentiles, CPU/RAM). For dashboards and
  alerting; cannot explain why one specific request was slow.
- **Logs** — discrete structured events (JSON), stamped with the
  trace ID so you can jump from a slow trace to its logs.

**OpenTelemetry (OTel)** is the vendor-neutral standard that ties this
together. Critical clarification: OTel *captures and emits* data (SDK in
our code + OTLP wire protocol + optional Collector sidecar) but **stores
nothing** — a separate backend stores, queries, and visualizes. This
decoupling is deliberate (anti vendor-lock-in): instrument once with OTel,
swap backends freely without touching application code.

## What We Want to Capture

### Backend (general)
- **RED metrics per endpoint**: Rate (req/s), Errors (failure %),
  Duration (p50/p95/p99 — percentiles, never averages; averages hide
  the slow tail users actually feel).
- **USE metrics for resources**: Utilization / Saturation / Errors for
  CPU, memory, disk, connection pools. On a single 8GB box running
  Postgres + Redis + Qdrant + API, memory saturation is the most
  likely silent killer.
- **Traces** with spans on:
  - Every network hop (FastAPI request, psycopg v3, redis, httpx — all
    covered by OTel auto-instrumentation)
  - Every meaningful internal pipeline stage (manual spans): query
    understanding, each parallel LLM call, lexical search, vector
    search, metadata scoring, score merging, quality reranking,
    display metadata fetch.
- **Structured logs** correlated to traces via trace ID.

### LLM-specific
Per-call, on the same LLM spans that already live in the trace, using
OTel's **GenAI semantic conventions** (`gen_ai.*` — don't invent our own
names): provider + model, token counts (in/out), computed dollar cost,
prompt version, and prompt/response payloads. Payload capture notes:
- **No TTFT** — we don't stream (structured-output calls resolve as one
  blob), so it isn't measurable. Use `gen_ai.usage.output_tokens` vs.
  total span duration as the thinking-vs-writing proxy instead.
- **Payloads:** capture 100% now, but behind a config flag and as a
  dial-able sample rate (not an on/off boolean). Large payloads go on span
  **events**, not attributes. Standard at scale = always-on-error + 1–5%
  random sample. (Prompts are movie queries, not PII.)

Expected first actionable finding: query understanding fans out parallel
LLM calls across providers and the slowest call gates the whole stage — a
trace waterfall will expose per-provider tail latency immediately.

## Tool Decisions

### ✅ Instrumentation: OpenTelemetry SDK — FINALIZED
The one-time, transferable, resume-grade investment. Auto-instrumentation
for FastAPI / httpx / psycopg v3 / redis, plus manual spans around pipeline
stages, plus `gen_ai.*` attributes on LLM spans. All downstream backend
choices become cheap to reverse because everything speaks OTLP.

### ✅ General backend: Grafana stack — FINALIZED
- **Local (learning + development):** the all-in-one `grafana/otel-lgtm`
  Docker image — OTel Collector + Tempo (traces) + Prometheus (metrics) +
  Loki (logs) + Pyroscope + Grafana, one container, OTLP on 4317/4318, UI
  on 3000. Dev/demo/test only, not production-grade.
- **Production (EC2):** Grafana Cloud free tier (10k metric series, 50GB
  logs/traces, 14-day retention). Ships telemetry off-box rather than
  self-hosting, so it doesn't compete for the t3.large's 8GB.
- **Why this over the alternatives:** `otel-lgtm` *is* the
  Grafana/Prometheus/Tempo/Loki stack, so learning locally builds the exact
  prod skill (dev/prod parity, one skill, all three signals in one place),
  and Grafana + Prometheus is the most marketable backend learnable for
  free. PromQL and Grafana dashboards are literal job requirements.

### ✅ One trace, one store — FINALIZED
Pipeline spans **and** LLM-call spans (with `gen_ai.*` attributes) resolve
to a **single trace** in a **single store** (Tempo). No separate LLM tool
in the first build — Grafana renders the attributes fine.

### LLM-native viewer: DEFERRED / decoupled (Phoenix vs Langfuse Cloud)
A dedicated LLM tool buys LLM-native UX (chat-style prompt/response
rendering, token/cost dashboards, eval scoring) — a real *later* want, not a
first-build need. Deferred and off the critical path; add via an OTel
Collector fan-out (same trace → second exporter) when Grafana's generic
attribute view stops sufficing.
- **Arize Phoenix** — OTel-native, strong open-source evals, single
  container. **Langfuse Cloud** — bigger community, zero footprint (v3
  self-host needs ClickHouse + Redis + S3, too heavy for the t3.large).
- Decision between them is Open Question #1 — decide only when triggered.

### Evaluated and rejected (don't re-open)
- **Honeycomb** — best-in-class latency debugging (BubbleUp), generous free
  tier, but loses on dev/prod parity (a *second* tool alongside local
  Grafana), on the resume goal (Grafana is more transferable), and on
  all-signals coverage. Would only win if latency-debugging were the sole
  goal with no learning objective.
- **Datadog / New Relic** — priced for companies; Datadog is #1 in job
  listings but can't be learned free (no self-host, thin free tier). Know
  the name; not a learning target. (Revisit only if we'd pay purely for the
  resume line — Open Question #7 in the todos; current lean: no.)
- **Jaeger / SigNoz / AWS X-Ray** — Jaeger is traces-only; SigNoz teaches a
  less-ubiquitous skill with no local parity; X-Ray is AWS lock-in and weak
  on the vendor-neutral learning goal.
- **LangSmith** (LangChain-oriented; we don't use it), **Braintrust**
  (premium evals — revisit only if eval-driven CI gating becomes a priority).

### Product analytics: PostHog — product-side, deferred to frontend
Complementary to (not a substitute for) the tracing stack — PostHog does
*not* do distributed tracing, the signal our latency goal lives in. Natural
home for frontend user events (searches issued, result clicks, zero-result
queries), session replay, web vitals, feature flags; backend search
events/outcomes can also flow here for funnel analysis. Flesh out when
frontend work begins. (Whether app logs live in PostHog vs Loki is Open
Question #2.)

## Constraints

- Single EC2 t3.large (2 vCPU, 8GB RAM) running Postgres, Redis,
  Qdrant, and the API via Docker Compose — any self-hosted
  observability component competes with these for memory.
- Cost is priority #3: prefer free tiers and open source; avoid
  per-seat/per-host commercial pricing.
- Solo developer: prefer fewer moving parts (priority #4, code
  simplicity) — one instrumentation layer, minimal backends.

## Implementation Order

1. **Traces + LLM attributes** — OTel SDK + auto-instrumentation + manual
   spans around search pipeline stages, **including `gen_ai.*` attributes on
   LLM-call spans** (tokens, model, cost). View in local `otel-lgtm`. (LLM
   attributes are folded in here, not a separate step — they're just extra
   key-values on spans we're already creating.)
2. **LLM-native viewer** *(deferred/optional)* — stand up Phoenix or
   Langfuse via a Collector fan-out only when we want LLM-native UX.
3. **Metrics** — RED per endpoint + USE for the EC2 box; Grafana
   dashboards + a small number of alerts (p95 latency, error rate,
   memory saturation).
4. **Structured logs** — JSON logs with trace ID correlation, shipped
   to Loki (or PostHog logs — see Open Question #2).
5. **Frontend / PostHog** — when frontend work begins. Cheap to add: W3C
   Trace Context lets the frontend mint traces the backend continues with no
   code change, provided we use standard OTel propagation and don't strip
   `traceparent` (CORS + Cloudflare tunnel).

## Open Questions

Tracked with status in `observability_todos.md`. Summary:

1. Phoenix vs Langfuse Cloud for the LLM-native viewer — *decoupled/deferred,
   no longer a build blocker.*
2. App logs land in Loki (Grafana Cloud) vs PostHog logs — pick one.
3. Sampling strategy as traffic grows (head sampling fine now; revisit
   tail-based only if telemetry volume becomes a cost concern).
4. Whether ingestion-pipeline runs (`movie_ingestion/`) get traced too, or
   only the search-time path. Search-time first; ingestion is batch with
   tracker.db already giving progress visibility.
5. Sentry for error tracking — complementary; decide if/when error volume
   justifies it.
6. LLM payload capture rate at scale (100% now → sampled later).
7. Whether Datadog is ever worth paying for purely as a resume line.
