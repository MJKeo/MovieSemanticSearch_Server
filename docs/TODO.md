# TODO

Tracks actionable items discovered during development sessions.
Items here are things to address when the relevant work begins,
not urgent fixes.

## Switch to residential proxies for database refresh pipeline
**Context:** Datacenter proxy IPs get flagged by IMDB, causing
mass timeouts and 502s. Residential IPs (real ISP addresses) are
much harder to block. DataImpulse offers residential on the same
platform — just change the proxy port/host in `build_proxy_url()`.
**When:** Building the daily update / database refresh pipeline.
**See:** memory/imdb-scraping.md for full tuning findings.
