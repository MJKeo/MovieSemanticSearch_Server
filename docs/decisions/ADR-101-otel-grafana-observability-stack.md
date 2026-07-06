# [101] — Observability instrumentation: OpenTelemetry + Grafana stack

## Status
Active

## Context
The system had no tracing, metrics, or structured logging — debugging
latency (PROJECT.md priority #2) meant reasoning about the multi-channel
search pipeline blind. A 2026-07-03 research pass (recorded in
`observability_context/initial_implementation_context.md` and
`observability_logs_plan.md`) evaluated instrumentation/backend options
against: not competing for the single EC2 t3.large's 8GB RAM
(PROJECT.md constraint), near-zero cost (priority #3 — free tiers/
open-source only), and building an industry-recognized, transferable
skill rather than a throwaway internal tool.

## Decision
Instrument with **OpenTelemetry** (vendor-neutral, CNCF-graduated) as the
one instrumentation layer for all telemetry — traces first (this phase),
metrics and structured logs in later phases — exported via OTLP to a
**Grafana stack**: the all-in-one `grafana/otel-lgtm` container locally
(Collector + Tempo + Prometheus + Loki + Grafana), Grafana Cloud free
tier in production (not yet wired). Pipeline spans and LLM-call spans
(carrying `gen_ai.*` attributes, once added) share **one trace in one
store** rather than routing LLM calls to a separate LLM-native tool.
Implemented in `observability/tracing.py`; as-built state tracked in
`observability_context/observability_architecture.md`.

## Alternatives Considered
- **Honeycomb** — best-in-class latency debugging (BubbleUp
  auto-diagnoses which dimension slow traces share). Rejected: would be
  a second tool alongside local Grafana (breaks dev/prod parity), no
  self-hosted free tier fitting the single-box constraint, and less
  transferable than Grafana/Prometheus as a resume-credible skill.
- **Datadog** — highest job-listing recognition. Rejected: no free
  self-host tier, so it can't be learned to depth on a hobby project;
  OTel's vendor-neutrality already signals Datadog-readiness without
  paying for it.
- **Phoenix / Langfuse (dedicated LLM-observability viewer)** —
  deferred, not rejected: folding `gen_ai.*` attributes into the same
  OTel traces (Tempo) removed it as a build blocker for Phase 1;
  revisit only if Grafana's generic attribute view stops being enough
  for LLM debugging (`observability_todos.md` Phase 2 trigger).

## Consequences
- Every future telemetry decision (traces, metrics, logs) is now an OTel
  SDK / OTLP decision; the backend is swappable via env var, but the
  `Name`-registry / span conventions (see the "Observability Conventions"
  section of `docs/conventions.md`) are now a standing dependency across
  `api/` and, eventually, `search_v2/`/`db/`.
- Telemetry is local-only today (dev `otel-lgtm`); production export
  (Grafana Cloud) is unbuilt (Phase 5) — no production visibility yet.
- Locks in specific auto-instrumentors (FastAPI, httpx, psycopg v3,
  redis); Qdrant's gRPC client is a known, deliberate gap pending a
  manual span.

## References
- `observability_context/initial_implementation_context.md` (full
  rationale and rejected alternatives)
- `observability_context/observability_architecture.md` §1–§4 (as-built
  bootstrap, backends)
- `observability_context/observability_todos.md` (phased rollout plan
  and status)
- `docs/modules/observability.md`
- `docs/conventions.md` — "Observability Conventions" (the manual
  instrumentation naming registry + per-request outcome conventions this
  stack adoption builds on; formerly the ADR-102 draft, promoted to a
  convention)
- `docs/PROJECT.md` (priority #2 latency, priority #3 cost; single-EC2
  constraint)
