# [091] — CategoryCombineType.CONSENSUS for SENSITIVE_CONTENT

## Status
Active

## Context
`SENSITIVE_CONTENT` was using `CategoryCombineType.ALTERNATIVES` (max
across committed calls). A single endpoint scoring high — e.g. SEMANTIC
matching "gory" in descriptive plot prose while KEYWORD disagreed and
META was silent — could over-promote sensitive-content movies without
genuine multi-source consensus. The goal is that movies matching across
committed endpoints should rank higher than movies spiking on one endpoint.

## Decision
Added `CategoryCombineType.CONSENSUS` to `schemas/enums.py`: geometric
mean over committed-call scores with `_CONSENSUS_FOLD_FLOOR = 0.1`,
mirroring the FACETS across-category fold at the within-category level.
`SENSITIVE_CONTENT` switched from `ALTERNATIVES` to `CONSENSUS`.

The fold only sees committed endpoints (walk-then-commit handler gating),
so single-endpoint commits degenerate to passthrough; multi-endpoint commits
are pulled toward agreement. Score impact is intentionally aggressive:
a 3-spec [0.95, 0.05, 0.05] commit drops from 0.95 (max) to ~0.21
(consensus). Reversible by flipping the `CategoryCombineType` back to
`ALTERNATIVES` on the `SENSITIVE_CONTENT` row.

## Alternatives Considered
- **Leave ALTERNATIVES**: single SEMANTIC spike continues to over-promote
  based on plot-prose vocabulary overlap without KW/META corroboration.
- **ADDITIVE (product)**: strictly punishes any zero; too aggressive — a
  single weak endpoint would zero the category even with two strong ones.
- **Raise SEMANTIC threshold only**: narrower fix, doesn't address the
  structural single-endpoint-wins problem.

## Consequences
- SENSITIVE_CONTENT now requires at least partial corroboration across
  endpoints. "Famous for gratuitous gore" (SEMANTIC-only commit) degenerates
  to passthrough; "no gore, not too bloody" (multi-endpoint) gets pulled
  toward agreement.
- `TARGET_AUDIENCE` remains on `ALTERNATIVES` (also bucket 8); revisit
  if the same single-endpoint over-promotion pattern appears there.
- 43-member combine_type assignment is now: 27 SINGLE, 11 ADDITIVE,
  3 ALTERNATIVES, 1 CONSENSUS, 1 NO_OP.

## References
- `schemas/enums.py` (`CategoryCombineType`), `schemas/trait_category.py`,
  `search_v2/stage_4_execution.py`
- `search_improvement_planning/rescore_overhaul.md` (ALTERNATIVES rationale)
- ADR-088 (FACETS soft fold geometric mean — same floor constant)
