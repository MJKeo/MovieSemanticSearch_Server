# [095] — Step 0 ACTOR flow generalized to role-agnostic PERSON flow

## Status
Active

## Context
The Step 0 ACTOR flow resolved person names only against `lex.inv_actor_postings`.
Queries like "david attenborough" — where the person is credited as writer or
narrator on most films but rarely as actor — silently fell back to the standard
NLP flow and returned zero or wrong results. The fix was to expand the executor
to union across all five role posting tables (actor, director, writer, producer,
composer).

## Decision
Rename the entire flow (schema, prompt, executor, orchestrators, runners, docs)
from ACTOR to PERSON:
- `EntityKind.ACTOR` → `EntityKind.PERSON`, `EntityFlow.ACTOR` → `EntityFlow.PERSON`,
  `SearchFlow.ACTOR` → `SearchFlow.PERSON`, `ActorReference` → `PersonReference`,
  `ActorFlowData` → `PersonFlowData`, `actor_search.py` → `person_search.py`.
- The executor fans across ALL five posting tables in parallel with no preferred role.
- Per-(person, movie) bucket = MIN across roles: actor table → zone bucket
  (LEAD/SUPPORTING/MINOR top-half/MINOR bottom-half via `actor_zones.py`);
  non-actor tables → uniformly LEAD (no billing data available).
- Multi-person queries use UNION semantics (was intersection): any movie where
  any named person has any credit qualifies. Per-movie overlap_count = number
  of named people who appear; within-bucket sort is (overlap_count DESC,
  popularity DESC, movie_id DESC) so full-cast films surface above
  single-person matches in the same bucket.
- `actor_zones.py` kept as-is (shared with the entity endpoint's billing scorer).

## Alternatives Considered
- **Keep ACTOR, add PERSON as a parallel flow**: rejected — PERSON is a strict
  superset; no value in dual maintenance. The enum rename is the honest model.
- **Check the "most known for" role and route to that table only**: rejected —
  role popularity is not in the data at query time, and cross-role credits
  (Attenborough: writer on most films, occasionally actor) are the exact case
  that broke the old flow.
- **Intersection for multi-person**: the old ACTOR flow used MAX bucket reduction
  + intersection. Switched to UNION + overlap_count because intersection was too
  restrictive ("Spielberg and Williams" would only return their collaborations).

## Consequences
- Queries for non-actor film personnel now work (documentary narrators, composers,
  non-acting directors).
- Self credits (documentary subjects, concert performers) require a separate fix:
  IMDB classifies these under `category.id = "self"` not `"actor"`, so the scraper
  filter must include `"self"` for `lex.inv_actor_postings` to contain them. Tracked
  in the `self`-credit DIFF entry (same session).
- SSE wire contract: `branch_stage` fetch type `"actor"` → `"person"`,
  stage label `"resolving_actor"` → `"resolving_person"`. Frontend listeners
  keyed on old strings need updating.
- Unit tests on the old `actor_search.py` module name / `ActorFlowData` class need
  updating in the testing phase.

## References
- `search_v2/person_search.py` (executor), `search_v2/actor_zones.py` (shared zones)
- `schemas/enums.py` (EntityKind, EntityFlow, SearchFlow, PersonReference, PersonFlowData)
- `search_v2/step_0.py`, `search_v2/streaming_orchestrator.py`
- `docs/modules/search_v2.md` (Person search executor section)
