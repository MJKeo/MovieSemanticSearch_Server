# ADR-018 — IMDB HTTP Client Tuning for Residential Proxies

## Status
Active

## Context

ADR-015 decided to switch from datacenter to residential proxies for IMDB
scraping. Once the switch was made, the http_client.py constants — originally
calibrated for datacenter proxy behavior — needed re-tuning. The key
behavioral difference: with datacenter proxies, retrying the same IP is
often fruitless (it stays blocked); with residential proxies, each new
connection routes through a different IP, so fast failure + immediate retry
is strictly better than waiting.

The original constants were: request timeout 30s, exponential backoff
(2^attempt + rand(0,1)s), semaphore 60, max connections 80.

## Decision

Retune `http_client.py` for residential proxy characteristics:

- **Request timeout: 5s** (down from 30s) — successful fetches under
  residential proxies complete in <1s; a 5s timeout fails fast on
  degraded IPs without over-rotating on transient slowness.
- **Retry delay: flat 0.2–0.3s** (down from exponential 2^n + rand) —
  each retry gets a fresh residential IP, so backoff delay only wastes
  time; a short flat delay is sufficient to avoid burst detection.
- **Semaphore: 60** (unchanged) — tested at 100; higher concurrency
  increased timeout rates without improving throughput, confirming the
  bottleneck is IP quality, not parallelism.
- **Max connections: 80** (unchanged).

## Alternatives Considered

1. **Keep exponential backoff**: Rejected. With IP rotation on every
   retry, backoff delays provide no benefit and only slow recovery.
   Exponential backoff is appropriate when retrying the same endpoint;
   here each retry is effectively a fresh request from a new IP.

2. **Increase semaphore to 100**: Tested and rejected. Higher concurrency
   produced more timeouts, not more successful fetches — the proxy pool's
   IP quality is the binding constraint, not the number of in-flight requests.

3. **Reduce timeout further (e.g., 2s)**: Not adopted. 5s provides a
   small buffer for genuine latency variance without meaningfully increasing
   wait time on blocked IPs. The former datacenter value of 2s was
   appropriate for a tighter IP block failure mode; residential IPs can
   have slightly higher natural variance.

## Consequences

- Retry behavior is now tuned for residential proxies. If the pipeline
  is ever run against datacenter proxies again, consider restoring
  exponential backoff and a lower timeout (see ADR-015 for datacenter
  tuning parameters).
- The flat retry delay means three retries complete in under 1s of
  sleep total, significantly reducing wall-clock time for failed movies.
- `memory/imdb-scraping.md` remains the authoritative record of
  session-by-session proxy tuning history.

## References

- ADR-015 (IMDB proxy strategy — datacenter vs. residential)
- ADR-009 (IMDB GraphQL migration)
- docs/modules/ingestion.md (Stage 4 proxy tuning section)
- movie_ingestion/imdb_scraping/http_client.py
- memory/imdb-scraping.md
