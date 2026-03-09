# ADR-015: IMDB Proxy Strategy — Datacenter vs. Residential IPs

**Status:** Active

## Context

Stage 4 (IMDB scraping) routes requests through DataImpulse proxies.
During a bulk scraping run, mass failures occurred — ConnectTimeouts,
HTTP 502s, ReadErrors, and RemoteProtocolErrors — despite the pipeline
working correctly days earlier. Investigation showed DataImpulse proxy
connections were succeeding, but IMDB was returning tiny error responses
(~1.3 MB/min throughput vs ~18 MB/min when healthy).

The root cause was IMDB blocking datacenter IP ranges. DataImpulse
offers both datacenter and residential proxies; datacenter IPs are
easier to identify and block without affecting legitimate users.

## Decision

For the current bulk scrape (now ~complete), continue using datacenter
proxies with tuned constants. For the future daily-update pipeline,
switch to **residential proxies** (real ISP addresses that IMDB cannot
block without impacting legitimate users).

### Operational tuning for datacenter proxies

When datacenter IPs are not blocked, these constants maximize
throughput:
- **Request timeout: 2s** — successful fetches complete in <1s;
  anything slower means a flagged IP. Fail fast to trigger rotation.
- **Semaphore ceiling: ~35** — beyond 35, timeout rates increase
  without throughput gain. Bottleneck is IP quality, not parallelism.
- **Retry backoff: 0.3–0.8s** — each retry gets a fresh IP via
  rotation; exponential backoff wastes time on already-flagged IPs.
- **Expect throughput degradation over time** within a session as the
  datacenter subnet's usable IP pool is progressively exhausted.

Tuning details and session-by-session history are maintained in
`memory/imdb-scraping.md`.

## Alternatives Considered

1. **Residential proxies for the bulk scrape**: Would likely avoid
   blocking entirely, but DataImpulse residential bandwidth is more
   expensive and the bulk scrape was already ~complete when the
   datacenter blocking was diagnosed.
2. **Different proxy provider**: Not evaluated; DataImpulse was
   already in use and the plan supports 2000 threads / 500 concurrent
   connections, so the plan is not the limiting factor.
3. **IMDB official API**: Requires a commercial license not available
   at this project's scale.

## Consequences

- Datacenter proxies remain viable for bulk scraping provided tuning
  constants are applied — but performance degrades if IMDB starts
  blocking the subnet mid-run.
- Residential proxies are required for any long-running or
  recurring pipeline (daily updates) to avoid cumulative IP exhaustion.
- The `DATA_IMPULSE_HOST`/`DATA_IMPULSE_PORT` env vars in `http_client.py`
  should point to the residential proxy endpoint when running the
  daily-update pipeline.
- `memory/imdb-scraping.md` is the authoritative record of proxy
  tuning history across sessions.

## References

- docs/decisions/ADR-009-imdb-graphql-migration.md (scraping architecture)
- docs/modules/ingestion.md (Stage 4 proxy tuning section)
- memory/imdb-scraping.md (proxy findings and tuning history)
- movie_ingestion/imdb_scraping/http_client.py
