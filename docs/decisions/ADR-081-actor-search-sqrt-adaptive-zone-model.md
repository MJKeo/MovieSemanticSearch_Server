# [081] — Actor search: sqrt-adaptive zone model for prominence buckets

## Status
Active

## Context
Actor search needs to bucket results by how prominent an actor's role is in each film
(lead vs. supporting vs. minor). Billing position is the raw signal, but a fixed
threshold (e.g., billing ≤ 3 = lead) doesn't generalize: a cast of 4 and a cast of 40
have very different billing distributions. A fixed cutoff would over-bucket large-cast
ensemble films and under-bucket small films.

## Decision
Use a sqrt-adaptive zone model implemented in `actor_zones.py`:
- Zone cutoffs scale with sqrt(cast_size) rather than being fixed integers.
- Constants: LEAD_FLOOR=2 (minimum lead zone size), LEAD_SCALE=0.6, SUPP_SCALE=1.0.
- `zone_cutoffs(cast_size)` returns the billing-position boundaries for 4 buckets:
  LEAD, SUPPORT, MINOR, BACKGROUND.
- `zone_relative_position(billing, cast_size)` returns the normalized [0,1] position
  within the actor's zone, used for within-bucket tie-breaking.
- Within a single actor's aliases: MIN bucket (best billing across alias variants wins).
- Across multiple actors: MAX bucket (weakest-link — film must have all actors at that
  prominence level or better).

## Alternatives Considered
- **Fixed billing threshold**: Fails for large ensembles and tiny casts.
- **Percentile rank of billing within cast**: Continuously scored but loses the
  semantic bucket structure; harder to explain and tune per zone.
- **LLM-assisted role classification**: High latency and cost for what is a
  deterministic structural property; billing position is sufficient.

## Consequences
- Deterministic, no LLM call; sub-millisecond per film.
- Works correctly across both small films (3-person cast) and large ensembles (100+ cast).
- LEAD_FLOOR ensures very small casts never produce an empty lead bucket.
- Zone model is shared between actor_search.py and entity_query_execution.py; changes
  to actor_zones.py affect both paths.

## References
- docs/modules/search_v2.md — Actor executor section
- search_v2/actor_zones.py, search_v2/actor_search.py
