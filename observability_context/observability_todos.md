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
locally in Tempo. Within **1c**, the three read endpoints are **done and tested**:
`/title_search` (1c-5), `/movie_details` (1c-6), and `/movie_credits` (1c-7) — manual
spans + semantic attributes built and runtime-verified, including error recording. Two
cross-cutting refactors have since landed on those three endpoints (both done):
**centralized naming** (`observability/names.py`) and a **universal request-outcome**
attribute pair (`outcome.success` / `outcome.failure_reason`, replacing the old
per-endpoint `*.not_found_reason`) — see the naming/outcome note under 1c below and
`observability_architecture.md` §5–§7 for the current catalog. **The remaining Phase 1
work is 1c-1 through 1c-4** (`/query_search`, `/rerun_query_search`, `/similarity_search`,
`/attribute_search`) — the high-value part for the latency goal, carrying the `gen_ai.*`
attributes. Implementation entries are in `../DIFF_CONTEXT.md`.

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

### 1c. Manual spans around pipeline stages — per endpoint
Add explicit spans (child of the FastAPI request span) around each meaningful
internal stage. Reference the pipeline in CLAUDE.md / `db/` search orchestration.

**Work the endpoints one at a time, in this order** — each later endpoint
reuses spans introduced by an earlier one, so this minimizes rework. All 7 are
in `api/main.py`. `/health` is intentionally excluded — it's three connectivity
checks already fully covered by auto-instrumentation, with no meaningful
internal stages to span.

Instrumentation weight per endpoint (see the plan for detail): #1 large, #2–#4
medium, #5–#7 tiny (mostly already auto-traced; a manual wrapper span only where
it names a real unit of work or covers Qdrant's gRPC gap).

> **Naming / outcome updates since 1c-5/6/7 shipped (both refactors done).** Two
> cross-cutting changes landed after these items and are now the current source of
> truth — see `observability_architecture.md` §5–§7. The per-item attribute names
> in 1c-5/6/7 below record the **original** build; read them as historical.
> 1. **Centralized naming.** All manual span names + attribute keys are now `Name`
>    constants derived from a namespace root in `observability/names.py` (no inline
>    literals). Renames: `movie_details.source` / `movie_credits.source` →
>    **`movie.payload_source`** (one shared key on both endpoints); `credits.*` →
>    `movie_credits.*`; the over-nested `movie.payload.source` flattened →
>    `movie.payload_source`.
> 2. **Universal request outcome (`api/outcome.py`).** Every endpoint's server span
>    now carries **`outcome.success`** (bool, every path) + **`outcome.failure_reason`**
>    (only when false), written **once** by the `@record_outcome` decorator; each
>    failure site raises `EndpointFailure(failure_reason=…)` and the reason bubbles
>    up. This **replaced** the per-endpoint `*.not_found_reason` attribute: the two
>    404 reasons are now `outcome.failure_reason` = `not_indexed` / `tmdb_removed`,
>    joined by `invalid_parameters` (422), `tmdb_fetch_failed` (502), and
>    `internal_error` (500). "Failure" is broader than "span error" — a 404 is a
>    failure but not a span error.

#### 1c-1. `POST /query_search` — the full NLP pipeline (do first) [~]
The marquee endpoint (`api/main.py:427`): Steps 0/1/2/3 → Stage 4, streamed as
SSE, with the parallel LLM fan-out the latency goal targets. Everything below
reuses these spans. This is the highest-value item in Phase 1.

> **Superseded breakdown (2026-07-06):** the detailed per-phase plan, locked
> decisions, open questions, and bite-sized implementation checklist for this
> item now live in **`query_search_planning.md`** (this folder) — treat that
> doc as the source of truth for 1c-1's work breakdown. The sub-checklist
> below predates it and describes the V1 pipeline vocabulary (the live path is
> the V2 streaming, branch-parallel architecture); read it as historical
> intent only. Check 1c-1 off here when `query_search_planning.md` §5 is
> fully landed.
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

#### 1c-2. `POST /rerun_query_search` — replay with new filters [ ]
Re-runs prior branches with a new filter set (`api/main.py:826`), bypassing
Steps 0/1 and re-entering at Step 2 (entity flows re-enter at their executor).
Same SSE shape as /query_search. Mostly reuses the Step 2 → Stage 4 spans from
1c-1.
- [ ] Span around the rerun-plan / filter translation boundary
      (`_to_rerun_plan`, `_to_metadata_filters`).
- [ ] Confirm the shared Step 2 → Stage 4 spans attach under this endpoint's
      request span (not orphaned), and that entity-flow re-entry is spanned.

#### 1c-3. `POST /similarity_search` — anchor-set vector flow, no LLM [ ]
Ranked "similar to" from a caller-supplied TMDB-ID anchor set
(`api/main.py:870`). No NLP/LLM. Introduces the Qdrant span that
auto-instrumentation can't provide (its client is gRPC).
- [ ] Vector search span(s) over the anchor set (Qdrant — the gRPC gap).
- [ ] Candidate-generation lane(s) + scoring / merge span.
- [ ] Hard-filter application + MovieCard hydration span.

#### 1c-4. `POST /attribute_search` — hard-attribute browse + person ranking [ ]
Filters + person-prominence ranking, no LLM/vector (`api/main.py:959`). Mostly
Postgres (auto-traced) + in-memory ranking.
- [ ] Person resolution span (reuses the Step 0 person model / resolver).
- [ ] Attribute query + prominence ranking span.
- [ ] MovieCard hydration span (`fetch_movie_card_summaries`).

#### 1c-5. `GET /title_search` — trigram typeahead [x]
Single trigram Postgres query (`api/main.py:1029`), already auto-traced. Both
psycopg queries (`run_title_search`, `fetch_movie_card_summaries`) get auto
spans, so timing is free — the work here is semantic attributes only.
- [x] Four attributes on the **FastAPI request (server) span** (via
      `trace.get_current_span()`), NOT on the auto psycopg child spans and NOT
      in a new wrapper span (title_search is one unit of work): `title_search.query`
      (raw text — attribute-only, never a metric label, unbounded cardinality),
      `.limit`, `.result_count` (hydrated card count = what the client got),
      `.fuzzy_result_count`. Implemented `api/main.py:1086`.
- [x] Surface the fuzzy count WITHOUT the discarded tier data: `run_title_search`
      now returns a `TitleSearchResult` NamedTuple (movie_ids, fuzzy_count)
      instead of `list[int]` (`search_v2/title_search.py`), keeping all
      attribute-setting in the endpoint and the search module OTel-free.
      `fuzzy_result_count` > 0 = fuzzy fallback fired = likely typo / catalog gap.
- [ ] (DEFERRED, optional) Per-result tier-1 vs tier-2 match mode: MODERATE
      change — the tier is computed in the SQL `CASE` but discarded (`SELECT
      movie_id` only, `db/postgres.py:3652`). Would require adding `CASE … AS
      tier` to the SELECT and threading it through. Skip unless per-result
      tiering is wanted; the fuzzy-fired signal covers the product question.
- [x] **Verify errors are recorded** — 422 rejections (empty `q`, out-of-range
      `limit`) surface on the request span as HTTP 4xx *without* marking
      the span ERROR (client-side, expected). Confirmed the FastAPI instrumentor
      does this and nothing raises an unrecorded 500.

#### Shared: `MoviePayloadSource` enum (1c-6 + 1c-7)
- [x] Define `class MoviePayloadSource(str, Enum): CACHE="cache"; TMDB="tmdb"`
      in `api/main.py` (both consumers live there — no new module, per YAGNI;
      promote to `observability/` only if a third cached endpoint adopts it,
      e.g. /similarity_search). Set on spans via `.value`. Referenced by both
      `movie_details.source` and `movie_credits.source` for consistency.

#### 1c-6. `GET /movie_details/{tmdb_id}` — detail payload [x]
Redis cache → TMDB fetch → build (`api/main.py:1733`). Redis + httpx both
auto-traced. Cold-path stages: (1) redis GET, (2) `fetch_movie_card_row`
Postgres existence-gate + source of local `reception_score`, (3) TMDB fetch
(httpx, append_to_response), (4) `_build_movie_details` CPU recombine
(reception_score folded into TMDB payload — **no auto span**), (5) encode +
cache SET, (6) cross-populate credits cache. Warm path = stage 1 only.
Span tree (cold path): request span → `movie_details.payload_creation`
(card fetch + TMDB fetch + `_build_movie_details` + encode; auto psycopg/httpx
nest inside) + `movie_details.cache_write` (the SET) + `movie_credits.build_and_cache`
(cross-populate, lives in the shared helper — nests free). Warm path = request
span + one auto redis GET.
- [x] Request-span attributes: `movie.tmdb_id` (int — the id is buried in the
      URL path, not cleanly queryable; add a real attribute. High-card →
      attribute only), `movie_details.source` (`MoviePayloadSource` value —
      the where-from / hit-miss signal, metric-label-safe).
      **`source` is set ONLY at the two success points** — `CACHE` right before
      the warm-path return, `TMDB` only AFTER a successful build (not on
      entering the cold path — a later 502 must not leave a misleading
      `source=tmdb`). On 404/502 the attribute is **absent** (no payload = no
      source); failure is captured by span status + HTTP code + not_found_reason.
      Keeps hit-rate = cache/(cache+tmdb) over successful requests only.
- [x] Subspan `movie_details.payload_creation` — wraps card fetch + TMDB fetch
      + build + encode. The build lives inside it (no standalone build span —
      only add one later if the recombine itself gets slow). Handle the
      not-indexed 404 so an expected miss does NOT mark this span ERROR
      (`record_exception=False, set_status_on_exception=False`).
- [x] Subspan `movie_details.cache_write` — wraps `cache_movie_details` SET
      (the DETAILS cache); attr `cache.write_ok` (bool) → False + event on the
      swallowed write failure, making the silent degradation queryable. This is
      the ONLY write_ok movie_details sets. Note there are two distinct cache
      writes in the cold path: the details cache (here) and the credits cache
      (inside the shared helper, below) — the credits `write_ok` is owned by
      `movie_credits.build_and_cache` (1c-7), not set by movie_details.
- [x] `movie_details.not_found_reason` attr on 404: `not_indexed` (card row
      None) vs `tmdb_removed` (TMDB 404). Does NOT mark the span ERROR.
- [x] Cross-populate credits (`_encode_and_cache_credits`):
      **NOT an HTTP call to /movie_credits** — the shared build+cache helper.
      Instrumented the helper ONCE (see 1c-7); its `movie_credits.build_and_cache` span
      nests automatically under this request span (same in-process trace).
      Nothing extra here.
- [x] DROPPED (folded above): the `movie_details.resolved` success event
      (`source` attr is the queryable where-from signal, an event would be
      redundant) and the standalone `_build_movie_details` span (build is timed
      inside payload_creation).
- [x] **Verify errors are recorded** — 502 (TMDB fetch failed after retries)
      marks the span ERROR **and** `record_exception`s the `TMDBFetchError`;
      404s do NOT mark ERROR (expected outcome). Confirmed both in Tempo;
      swallowed redis failures appear as events, not span errors.

#### 1c-7. `GET /movie_credits/{tmdb_id}` — full cast & crew [x]
Same shape as 1c-6 (`api/main.py:1817`), one fewer stage. Cold-path stages:
(1) redis GET, (2) `fetch_movie_card_row` existence-gate ONLY (reception_score
unused), (3) lean credits-only TMDB fetch (httpx), (4) `_encode_and_cache_credits`
build (crew grouped by dept) + encode + redis SET. No reception fold-in, no
cross-populate. Warm path = stage 1 only.
Span tree (cold path): request span → `movie_credits.payload_creation` (index
+ lean TMDB fetch) + `movie_credits.build_and_cache` (the shared helper: build +
encode + SET). No cross-populate, no reception fold-in. Warm path = request
span + one auto redis GET. Note the asymmetry vs 1c-6: credits fuses build +
cache into the helper, so it's ONE `build_and_cache` span (vs details' separate
build-in-payload_creation + cache_write) — and that same span is the
cross-populate bar in 1c-6.
- [x] Request-span attributes: `movie.tmdb_id`, `movie_credits.source`
      (`MoviePayloadSource` value). **`source` is the distinctive one:** this
      cache is normally pre-warmed by /movie_details' cross-populate, so
      `source` here is a direct end-to-end test of whether that strategy works
      — "is every 'See all' click landing warm, or silently paying a TMDB round
      trip?" The cross-populate is best-effort and swallows failures
      (`api/main.py:1546`), so a silently-broken warm-up would ONLY show up as a
      high `source=tmdb` rate here. Expect near-100% cache on the details→credits
      flow; anything less is actionable (broken cross-populate, direct deep-link
      hits, or >24h TTL expiry).
- [x] Index check gets **NO dedicated span or event** — it's the single
      `fetch_movie_card_row` call, already the only auto psycopg span on this
      path (unambiguous). A manual span = duplicate bar; an event = redundant
      with that span + request status. Capture only the failure outcome via
      `movie_credits.not_found_reason` (404 only; not ERROR).
- [x] Subspan `movie_credits.payload_creation` — index + lean credits-only TMDB
      fetch (auto psycopg/httpx nest inside).
- [x] Instrument the shared `_encode_and_cache_credits` helper (`movie_credits.build_and_cache`
      span: build + encode + SET; attr `cache.write_ok`). Doing it in the helper
      — not the handler — is what makes the 1c-6 cross-populate traced for free.
- [x] `movie_credits.cast_count` / `movie_credits.crew_count` on the `movie_credits.build_and_cache`
      span (NOT the request span — the build happens in the shared helper, so the
      counts live where the data is; applies to both endpoints, filter by parent
      for the credits view). `crew_count` sums members across CrewGroup depts
      (not `len(crew)`, which is the department count).
- [x] **Verify errors are recorded** — same contract as 1c-6: 502 → span ERROR
      + `record_exception`; 404 → not ERROR. Confirmed in Tempo.

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
- [ ] Connection-pool saturation (psycopg v3 pool, redis pool).
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
