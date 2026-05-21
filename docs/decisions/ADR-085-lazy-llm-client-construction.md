# [085] — Lazy LLM client construction via _LazyClient proxy

## Status
Active

## Context
LLM provider clients (OpenAI, Gemini, Kimi, etc.) are initialized at module import
time in `generic_methods.py`. Each client performs network or credential validation
at construction. On the single EC2 instance, all services share the same process, so
importing `generic_methods` eagerly constructs every provider client — even those not
used in the current request. This adds startup latency and causes failures when
credentials for unused providers are absent from the environment.

## Decision
Wrap each provider client in a `_LazyClient` proxy. The proxy defers the actual
client construction until the first attribute access (i.e., the first real call).
Module-level names like `openai_client` remain but hold a `_LazyClient` instance
rather than the real client. Callers use them identically — the proxy is transparent.

The proxy is explicitly documented as not thread-safe. Under the current single-process
async deployment this is acceptable; parallel requests share the event loop and
construction is triggered at most once per provider before any concurrent access.

## Alternatives Considered
- **Always-eager construction with try/except per provider**: Each missing credential
  logs a warning but doesn't fail. More explicit but adds ~10 lines per provider and
  still pays construction cost for unused providers.
- **Explicit provider selection at startup from env config**: Requires a startup config
  step and complicates local dev where only one provider is configured.
- **Thread-safe lazy init with a lock**: Over-engineering for a single-threaded async
  deployment; adds complexity for zero benefit in the current architecture.

## Consequences
- Startup is faster and tolerates missing credentials for unused providers.
- First call to a new provider has a one-time construction cost (negligible in practice).
- Not thread-safe — if the deployment ever moves to multi-threaded (e.g., uvicorn
  workers > 1), this must be revisited.

## References
- docs/modules/search_v2.md — Interactions section (lazy LLM clients)
- implementation/llms/generic_methods.py
- ADR-026: multi-provider LLM routing
