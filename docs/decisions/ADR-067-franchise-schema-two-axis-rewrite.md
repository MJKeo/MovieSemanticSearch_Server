# [067] — Franchise Schema Two-Axis Rewrite (v4→v8)

## Status
Active

## Context
Franchise metadata went through three major structural redesigns (v4, v5,
v8) as evaluation-driven failures exposed fundamental schema limitations.
The v3 `franchise_role` closed enum conflated two independent questions —
what brands/groups the film belongs to (identity) and how it relates to
prior films (narrative position). This produced three structural problems:
(1) pair-remakes with no continuing franchise (Scarface 1983) could not be
represented, (2) the Iron Man / X-Men 2000 null-role case was internally
contradictory, (3) the reboot/remake overlap had no clean tiebreaker.

## Decision

**v4 (identity + narrative position axes):** Replaced `franchise_role` with
two independent axes. Identity axis: `lineage` (narrowest recognizable line),
`shared_universe` (broader cosmos above lineage), `recognized_subgroups`
(named sub-phases), `launched_subgroup`. Narrative position axis:
`lineage_position` (mutually exclusive enum: sequel/prequel/remake/reboot),
`special_attributes` (spinoff, crossover). `lineage_position` can populate
even when `lineage` is null, enabling pair-remakes. Reasoning fields scoped
per decision block (not one top-level field) per the Jason Liu / Instructor
"just-in-time reasoning" pattern.

**v5 additions:** `launched_franchise` flag (four-part test: first cinematic
entry, not a spinoff, audience recognizes the film over prior source
material, spawned recognized follow-ups). Global normalization rule for all
named entities. Shape-B `shared_universe` for spinoff-parent relationships
(Puss in Boots → Shrek). Field rename `launches_subgroup` → `launched_subgroup`.

**v8 (boolean replacement of special_attributes):** Replaced
`special_attributes: list[SpecialAttribute]` enum array with two independent
boolean fields (`is_spinoff`, `is_crossover`), each with a dedicated
reasoning field. Key semantic change: `is_crossover` now includes
shared-universe team-up films (Avengers, Justice League). Spinoff procedure
rebuilt around structural situating (trunk-vs-branch analysis) rather than
the character-prominence test, solving the Solo/Creed classification problem.
`SpecialAttribute` enum deleted from `schemas/enums.py`.

`validate_and_fix()` enforces three deterministic coherence checks after
parsing: partial null-propagation (lineage null clears shared_universe,
recognized_subgroups, launched_subgroup — but deliberately preserves
lineage_position and is_spinoff for pair-remakes and standalone spinoffs),
launched_subgroup coupling to recognized_subgroups, launched_franchise
precondition check.

## Alternatives Considered
**Keep franchise_role closed enum.** Rejected after v3 evaluation showed
the conflation of identity and narrative position produced unfixable
failure classes (pair-remakes, Iron Man null-role contradiction, reboot/remake
overlap) that no amount of prompt tuning could resolve.

**Add `is_prequel` / `launches_subgroup` booleans to v3.** Tried in v2/v3.
The fundamental problem was the enum's inability to represent films that
are both sequel and spinoff, or pair-remakes without a brand entity. Patching
booleans onto a structurally flawed base could not fix the root issue.

**Trunk-vs-branch as a retrieval field.** Considered adding it as a
first-class field for downstream search. Rejected — the user declined to
promote it beyond a reasoning artifact.

## Consequences
- All franchise generation rows in `generated_metadata.franchise` (v3 format)
  became stale after v4 and must be regenerated.
- `FranchiseRole` enum deleted from `schemas/enums.py`; no production code
  referenced its stable integer IDs.
- Schema is additive to the Postgres `movie_franchise_metadata` table design
  — any future Postgres projection needs updating when franchise fields change.
- `is_crossover=true` now fires on Avengers/Justice League team-up films —
  a deliberate semantic expansion that affects downstream search behavior.

## References
- `schemas/metadata.py` — `FranchiseOutput` class
- `schemas/enums.py` — `LineagePosition`, (deleted `SpecialAttribute`, `FranchiseRole`)
- `movie_ingestion/metadata_generation/prompts/franchise.py` — v4→v9 prompt
- `search_improvement_planning/franchise_test_iterations.md` — evaluation history
- `docs/modules/ingestion.md` — Franchise Metadata section
