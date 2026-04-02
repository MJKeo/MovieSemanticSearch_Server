# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Documentation audit fixes

Files: `implementation/classes/schemas.py`, `movie_ingestion/metadata_generation/generator_registry.py`, `movie_ingestion/metadata_generation/result_processor.py`, `movie_ingestion/metadata_generation/generators/viewer_experience.py`, `docs/decisions/ADR-048-narrative-techniques-11-to-9-sections.md`, `docs/decisions/ADR-049-watch-context-phase1-phase2-evolution.md`, `docs/modules/ingestion.md`, `CLAUDE.md`

### Intent
Fix 8 staleness issues found by docs-auditor across code, decision records, and module docs.

### Key Decisions
- **NarrativeTechniquesMetadata 11→9**: Updated search-side schema to match generation-side (9 sections, same field names/order). ADR-048 said this was done but it wasn't. Now it is.
- **source_of_inspiration registered**: Added eligibility checker, prompt builder, and live generator adapters to generator_registry.py. Added to SCHEMA_BY_TYPE in result_processor.py. Generator was fully implemented but unreachable via batch pipeline CLI.
- **viewer_experience naming**: Renamed `_DEFAULT_PROVIDER`/`_DEFAULT_MODEL` → `_PROVIDER`/`_MODEL` to match ADR-045 locked-generator convention. Fixed stale docstring claiming callers can override.
- **ADR-049**: Corrected production prompt claim from `SYSTEM_PROMPT_WITH_JUSTIFICATIONS` to `SYSTEM_PROMPT_WITH_IDENTITY_NOTE` (matches code).
- **ingestion.md**: Removed stale `review_insights_brief` @property references (property doesn't exist). Updated generator contract to reflect all 8 registered.
- **CLAUDE.md**: Added missing `tmdb_quality_calculated` and `imdb_quality_calculated` intermediate statuses.
- **ADR-048**: Fixed stale claims — Decision section now accurately says update was done; Consequences section updated from pending to completed.
- **PROJECT.md test count**: Auditor claimed 60 files but actual count is 59 — no change needed.

### Testing Notes
- `NarrativeTechniquesMetadata` field changes affect any code deserializing narrative techniques from Qdrant — search pipeline tests may need field name updates
- `_DEFAULT_PROVIDER`/`_DEFAULT_MODEL` rename in viewer_experience may break tests referencing those names
- New source_of_inspiration registry entry should be covered by existing generator_registry tests

## Redesign source_of_inspiration: narrow scope, add franchise_lineage, remove production_mediums
Files: `movie_ingestion/metadata_generation/schemas.py`, `movie_ingestion/metadata_generation/prompts/source_of_inspiration.py`, `movie_ingestion/metadata_generation/generators/source_of_inspiration.py`, `docs/TODO.md`

### Intent
Redesign source_of_inspiration generation based on 55-movie, 6-candidate evaluation results. Three changes: (1) remove production_mediums (now derived deterministically from genres+keywords at embedding time), (2) narrow sources_of_inspiration → source_material (closed set of specific source types, no franchise lineage), (3) add franchise_lineage field (sequel/prequel/reboot/spinoff/reimagining/series entry).

### Key Decisions
- **production_mediums removed from LLM**: 100% of Animation-genre movies have medium keywords in IMDB data, so production medium can be derived deterministically (Animation genre → use specific medium keywords; no Animation → "live action"). Eliminates the empty-list abstention bug that caused 14 failures in evaluation (gpt54nano-medium-just returned empty 25% of the time).
- **source_material narrowed**: Closed set of valid source types (novel, true story, manga, etc.). Prevents loose inference like "inspired by historical events" for Gladiator. Sequel/prequel/reboot moved to franchise_lineage.
- **franchise_lineage added**: Franchise position was fragmented across keywords (~2,060 movies) and source_material_hint (~2,888 movies) with partial overlap. LLM serves as consolidation + gap-filling layer, drawing on both inputs and parametric knowledge from title cues (e.g., "Part 2", "Returns").
- **Reasoning variant redesign**: Evidence fields renamed (source_reasoning → source_evidence, production_medium_reasoning → lineage_evidence) and reframed as inventories that do NOT gate the decision. Addresses the anchoring-to-abstention bug where writing "No direct evidence" caused models (especially gpt54nano-medium-just) to produce empty lists even when parametric knowledge should have filled them.
- **Existing merge TODO superseded**: The TODO evaluating merging production_keywords and source_of_inspiration is now moot — the generators have cleanly separated responsibilities with no output overlap.

### Planning Context
- Evaluation showed gpt5mini-minimal as strongest candidate (99.1% accuracy, $39/100k). gpt5nano to be removed.
- Reasoning variants generally hurt accuracy (gpt54nano-medium-just: 86.4% vs 96.4% base), except for gpt5nano where it recovered some keyword extraction failures. Redesigned reasoning fields address the root cause (abstention anchoring) rather than removing reasoning entirely.
- User preference: abstain if unsure rather than defaulting — both fields should be empty when evidence is insufficient, with post-processing at embedding time handling gaps.

### Testing Notes
- Schema field names changed: `sources_of_inspiration` → `source_material`, `production_mediums` → removed, `franchise_lineage` → new. All unit tests referencing old field names will break.
- `__str__()` output format unchanged (comma-joined lowercase terms), but input fields differ.
- Generator signature unchanged — no callers need updating.
- Existing evaluation data uses old schema fields and is incompatible with new schema — new evaluation run needed.
