# Conventions Draft

Observed patterns staged for review. Remove entries you disagree
with, then run /solidify-draft-conventions to merge the rest into
docs/conventions.md.

Entries are added automatically during /safe-clear based on
patterns observed in the session.

## Flat retry delay with rotating proxies
**Observed:** Exponential backoff is counterproductive when each retry
gets a fresh proxy IP. User asked for flat ~0.25s delay since there's
no reason to wait longer on later attempts — the new IP has no memory
of previous failures.
**Proposed convention:** When retrying through a rotating proxy pool
(where each attempt uses a different IP), use flat short delays
(~0.2-0.3s) instead of exponential backoff.
**Sessions observed:** 1
