# Initial Implementation Context

Durable record of the decisions and guidelines settled during the initial
observability research/design conversation (2026-07-03). Read this before
re-opening any tooling debate — several things below were evaluated and
deliberately closed, and re-litigating them wastes time. For the *why*
behind the framing, see `observability_logs_plan.md`; for open items still
needing alignment, see `observability_todos.md`.

---

## Finalized decisions

### 1. Instrumentation = OpenTelemetry (the crown jewel, backend-agnostic)
OTel is the transferable, resume-grade skill here — CNCF *graduated* (May
2026), 2nd most active CNCF project after Kubernetes, the de facto
vendor-neutral standard. The instrumentation layer is what we invest in;
the storage backend is swappable behind OTLP without touching app code.
This is the highest-leverage piece of the whole project.

### 2. Trace store = Grafana stack
- **Local dev:** `grafana/otel-lgtm` (single container: OTel Collector +
  Tempo + Prometheus + Loki + Pyroscope + Grafana, pre-wired, OTLP on
  4317/4318, UI on 3000). Dev/demo/test only — not production-grade.
- **Production (EC2):** Grafana Cloud free tier (Tempo). Ships telemetry
  off-box so it doesn't compete for the t3.large's 8GB.
- Chosen because `otel-lgtm` *is* the Grafana/Prometheus/Tempo/Loki stack,
  so learning locally builds the exact prod skill — one skill, dev/prod
  parity, all three signals in one place.

### 3. One trace, one store — LLM spans live with pipeline spans
Pipeline spans **and** LLM-call spans (with `gen_ai.*` attributes) go into
the **same trace** in the **same store** (Tempo). Do **not** stand up a
separate LLM tool in the first build. Grafana renders the `gen_ai.*`
attributes fine. This keeps the first step to a single backend.

### 4. LLM span attributes go in step 1, not a later phase
The `gen_ai.*` GenAI-semantic-convention attributes are just extra
key-values on spans we're already creating around LLM calls. Splitting
them into a separate phase risks instrumenting the LLM calls twice, and
they're the highest-value data given latency is the #1 goal. Do them
alongside the general traces. (This changed the original plan's ordering.)

### 5. Goal reframed: resume-grade, industry-standard skills
The tooling criterion is "something credible on a job application," not the
plan's original inertia toward Grafana. Conclusion held anyway, for a
stronger reason:
- **OTel** = the marquee, backend-neutral skill.
- **Grafana + Prometheus** = the most marketable backend you can learn
  *free and to depth* (PromQL + Grafana dashboards are literal job
  requirements).
- **Datadog** = #1 in job listings but effectively un-learnable on a free
  hobby project (commercial, no self-host, thin free tier). Not a learning
  target; know the name.
- OTel's vendor-neutrality lets us honestly claim Datadog-readiness without
  paying — "instrument once, point anywhere."

### 6. Honeycomb evaluated and rejected (don't re-open)
Best-in-class pure latency debugging (BubbleUp auto-diagnoses which
dimension slow traces share). Rejected anyway: loses on dev/prod parity
(would be a *second* tool alongside local Grafana), on the resume goal
(Grafana is the more transferable/recognized skill), and on all-signals
coverage. It would only win if latency-debugging were the *sole* goal with
no learning objective.

### 7. Time-to-first-token (TTFT) dropped for now
We don't stream (structured-output LLM calls resolve as one blob), so TTFT
isn't measurable and buys nothing decision-wise. Use
`gen_ai.usage.output_tokens` vs. total span duration as the
thinking-vs-writing proxy instead (large output + slow → generation-bound;
small output + slow → prompt/queue/provider-bound). Revisit TTFT only if we
ever stream results to a frontend, where it becomes a real UX metric.

---

## Guidelines to keep in mind (apply when the relevant phase arrives)

- **Cardinality rule — attributes vs. metric labels.** High-cardinality
  values (query text, `movie_id`) are fine as **span attributes** (viewed
  per-trace) but must **never** become **metric labels** — each distinct
  value mints a new time series and explodes storage. Rule of thumb: query
  text = attribute only, never a metric dimension. Matters most in the
  metrics phase.

- **LLM payload capture strategy.** Capture 100% of prompt/response
  payloads *now* (low traffic, best debugging signal), but build it as a
  **sample rate you can dial down, not a boolean you flip off**. Industry
  standard at scale = always-capture-on-error + a small random sample
  (1–5%, often fed to an LLM-as-judge eval). Put large payloads on span
  **events, not attributes** (backends choke on huge attribute values), and
  keep capture behind a config flag (the GenAI convention already gates
  it). Our prompts are movie queries, not user PII, so redaction is lower
  priority — but know the reflex exists.

- **Frontend trace propagation is cheap to add later.** Designing for
  frontend-minted traces costs ~nothing because W3C Trace Context / OTel
  handles both cases with the *same* backend code (header present → continue
  the trace; absent → mint a new one). The one requirement: use standard
  OTel propagation, **don't roll our own request-ID scheme**. When frontend
  work starts, the only backend hygiene check is that CORS allows the
  `traceparent` header and the Cloudflare tunnel doesn't strip it.

- **Span granularity heuristic.** Create a span around a unit of work you'd
  want to see as its own bar in the waterfall — something that takes
  meaningful time *or* can fail independently. One method may hold several
  spans; one span may cover many internal helper calls. A bar you'd never
  look at shouldn't exist.

- **Spans measure, logs annotate.** Don't hand-roll timing logs ("stage 1
  done in 340ms") — make a span instead; the backend derives percentiles
  from span data. The one useful summary log is a single structured
  completion line (`{trace_id, total_ms, per-stage ms}`) as a cheap,
  greppable index *into* the traces — not the primary latency tool.

- **Span status is coarse by design** (Unset / Ok / Error). "What happened"
  detail for a successful op lives in **attributes and events**, not in
  status. Status answers "did it work?"; attributes answer "what did it do?"
