# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Award category tag taxonomy (Stage-3 awards endpoint)
Files: schemas/award_category_tags.py (new), backfill_award_category_tags.py (new, temporary), db/init/01_create_postgres_tables.sql, db/postgres.py, schemas/award_translation.py, search_v2/stage_3/award_query_execution.py, search_v2/stage_3/award_query_generation.py, search_improvement_planning/finalized_search_proposal.md, search_improvement_planning/full_search_capabilities.md

### Intent
Replace the brittle ceremony-specific category-string filter on the Stage-3 award LLM endpoint with a closed 3-level concept-tag enum. The LLM previously had to emit exact IMDB surface forms ("Best Performance by an Actor in a Motion Picture - Drama" for Globes vs "Best Actor in a Leading Role" for Oscars), produced zero matches on a single character mismatch, and could not express broad concepts ("any acting award") without enumerating dozens of strings. Now the LLM picks from a 81-member CategoryTag enum at whatever specificity the requirement implies; one GIN-indexed `&&` overlap query handles every level.

### Key Decisions
- **3 levels in one combined enum, not three Pydantic fields.** `CategoryTag` is a single `(str, Enum)` containing 62 leaves (ids 1..99) + 12 mid rollups (ids 100..199) + 7 top groups (ids 10000..10006). Each member carries `tag_id: int` + `level: int`. Single combined enum keeps the LLM's structured-output JSON schema simple (one `$ref` instead of a 3-way union) while the level attribute and `LEVEL_*_TAGS` constants preserve per-level views in code. User explicitly chose this over three separate per-level fields after AskUserQuestion in plan mode.
- **Mid rollups defined only where they earn their keep.** lead-acting (lead-actor + lead-actress), supporting-acting, screenplay-any, best-picture-any, animated, documentary, short, sound-any, music, visual-craft, worst-acting, worst-craft. Branches with no useful intermediate concept (director, foreign-film, casting, festival-section, etc.) skip the mid level entirely; their stored tag list is just [leaf, group]. Multi-parent leaves are allowed: `animated-short` rolls into both `animated` AND `short` mids.
- **Per-row `category_tag_ids INT[]` stores leaf + every ancestor.** A row tagged with `lead-actor` stores `[1, 100, 10000]`. Querying `&& ARRAY[10000]` (acting group) catches every leaf and mid under it; querying `&& ARRAY[1]` catches only the specific leaf. One indexed query, any specificity, no expansion logic at query time.
- **100^level id scheme** (1..99 / 100..199 / 10000+) — globally unique ids across the whole taxonomy with room for a future 4th level at 1_000_000+. INT[] (matching the dominant gin__int_ops pattern in the schema) rather than SMALLINT[] gives that headroom.
- **Tag derivation lives at ingest, not query.** `tags_for_category(raw_text) -> list[int]` wraps the existing `consolidate()` from `consolidate_award_categories.py` and looks up the leaf's pre-computed ancestor list. Called from `batch_upsert_movie_awards`. The 766 distinct raw category strings collapse into 62 leaves with 100% coverage (verified empirically).
- **Bulk insert switched from unnest(arrays) to a single VALUES clause** because Postgres requires 2-D arrays to be rectangular and `category_tag_ids` is variable-length per row (2..4 ids). VALUES with one tuple per row keeps the insert to a single round trip; ~50 awards/movie keeps the parameter count modest.
- **Prompt taxonomy rendering is generated programmatically** from the enum + hierarchy via `render_taxonomy_for_prompt()` so the LLM-facing CATEGORY TAG TAXONOMY section and the schema can never drift. Hand-written one-line descriptions per tag live in `_TAG_DESCRIPTIONS` next to the enum.
- **Surface-forms section retained but stripped down** — only award_name guidance remains (Palme d'Or, Golden Lion, Razzie Award, etc.), since `award_names` is still a free-text axis. The pre-existing per-ceremony category enumeration (Oscars use "in a Leading Role", Globes use "Best Performance by..." etc.) is gone — that knowledge was the brittle surface the tag system replaces.
- **Razzie handling is unchanged** — still ceremony-id-based exclusion in execution. The taxonomy includes a separate `razzie` group with worst-* leaves so user can opt in via category_tags as well, but the ceremony-id default-exclusion logic still applies.
- **Backfill script lives at project root** as a temporary one-off. Issues `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` + `CREATE INDEX IF NOT EXISTS`, then runs one `UPDATE WHERE category = $1` per distinct category (~766 statements, fast). Idempotent and safe to re-run; user deletes the file once executed.

### Planning Context
Plan file: `/Users/michaelkeohane/.claude/plans/great-i-m-aligned-we-mutable-reddy.md`. Built on top of `consolidate_award_categories.py` (which collapses 766 raw strings → 62 concept slugs at 100% coverage). User's design specification was very explicit in the kickoff message: leaves define the string-matching surface, higher layers are derivable from which leaves applied, ids unique across the whole set via the 100^level scheme.

### Razzie tag interaction fix (post-review)
Files: schemas/award_category_tags.py, search_v2/stage_3/award_query_execution.py, search_v2/stage_3/award_query_generation.py

Code review caught a logical bug introduced by the new tag axis. The default `ceremony_id <> 10` Razzie exclusion was originally gated only on the `ceremonies` axis (the only axis that could express Razzie intent when categories were free-text strings). With the new tag taxonomy, `category_tags=[WORST_PICTURE]` (or any other razzie-group tag) is also an unambiguous Razzie opt-in — but execution wasn't checking that axis, so the default exclusion AND-conjuncted with the tag overlap and silently zeroed out the result.

Fix:
- New module-level `RAZZIE_TAG_IDS: frozenset[int]` in `schemas/award_category_tags.py` enumerating every tag whose presence signals Razzie intent (RAZZIE group + 13 worst-* leaves + worst-acting/worst-craft mids = 16 ids).
- `execute_award_query` now overrides `exclude_razzie=False` when any tag in the resolved `category_tag_ids` is in `RAZZIE_TAG_IDS`. Symmetric with the existing ceremony-axis opt-in path.
- `_RAZZIE_HANDLING` prompt section updated to describe the dual-axis opt-in: either signal alone is sufficient, but emitting both is encouraged for self-documenting specs.
- Added an import-time assertion in `schemas/award_category_tags.py` that `_TAG_DESCRIPTIONS` covers every CategoryTag member, so adding a new tag without a description fails loudly at import rather than at first prompt assembly.

Verified end-to-end against the live DB after backfill: `category_tags=[WORST_PICTURE]` returns 261 movies (was 0 before fix); non-razzie queries (lead-actor: 1,646; acting group: 3,793) are unaffected; explicit dual-axis opt-in produces the same count as single-axis opt-in.

### Testing Notes
- Existing Stage-3 award tests will need to be updated to construct `category_tags=[CategoryTag.LEAD_ACTOR]` instead of `categories=["Best Actor in a Leading Role"]`. Test boundaries rule: not touched in this commit; will be handled in a separate testing pass.
- New unit tests should cover: tags_for_category() spot-checks (all 62 leaves), render_taxonomy_for_prompt() golden-snapshot, _resolve_category_tag_ids() ordering/dedup, and the new VALUES-based batch_upsert_movie_awards round-trip.
- Backfill spot-check: `SELECT category, category_tag_ids FROM movie_awards WHERE category IN ('Best Actor', 'Best Animated Short Film', 'Worst Picture', 'Best Foreign Language Film', 'Directors''s Fortnight') LIMIT 1 PER` — confirm arrays match `tags_for_category()` output.
- Verify GIN index is used: `EXPLAIN ANALYZE SELECT ... WHERE category_tag_ids && ARRAY[1]` should show Bitmap Index Scan on `idx_awards_category_tag_ids`.
- LLM end-to-end: a query like "movies that won Best Actor" should produce `category_tags=[CategoryTag.LEAD_ACTOR]`; "any acting award" should produce `category_tags=[CategoryTag.ACTING]`.

## Keyword endpoint: unified classification registry
Files: schemas/unified_classification.py (new), unit_tests/test_unified_classification.py (new)

### Intent
Provide the single type the step 3 keyword LLM selects from. Merges OverallKeyword (225), SourceMaterialType (10), and the seven ConceptTag category enums aggregated via ALL_CONCEPT_TAGS (25) into one `UnifiedClassification` StrEnum + a `CLASSIFICATION_ENTRIES` registry mapping each name to `(display, definition, source, source_id)`. Execution code calls `entry_for(member)` to get the backing movie_card array column (`keyword_ids` / `source_material_type_ids` / `concept_tag_ids`) and the source-specific ID for a GIN `&&` overlap query.

### Key Decisions
- **OverallKeyword precedence on name collisions.** Iterate OverallKeyword first; any SourceMaterialType or ConceptTag member whose name already exists in the registry is skipped. OverallKeyword has broader coverage and is the stronger retrieval signal. In the current vocabulary the only real collision is BIOGRAPHY — the step 3 keyword LLM sees BIOGRAPHY as a keyword, not a source material. Tradeoff: 1 entry dropped, no disambiguation suffixes on 259 other members. Genre enum was confirmed fully subsumed by OverallKeyword (all 27 TMDB genres present as keyword terms) and deliberately excluded to avoid a fourth redundant source. The Genre enum itself is untouched — only excluded from the step 3 LLM surface.
- **Dynamically built StrEnum.** Rather than a hand-written 4th enum duplicating 260 members from three source enums, `UnifiedClassification` is constructed at import time from the registry. Keeps OverallKeyword / SourceMaterialType / ConceptTag as the single source of truth for definitions and IDs; nothing to keep in sync by hand. Tradeoff: loses IDE jump-to-definition on members (e.g. `UnifiedClassification.ACTION`). Accepted — the alternative duplication cost is higher.
- **Hand-written display + definition for SourceMaterialType.** SourceMaterialType enum members carry no display label or definition. A `(display, definition)` tuple per member is hand-written in `_SOURCE_MATERIAL_METADATA` so the prompt can render "TV Adaptation" (not "Tv Adaptation" from naive `.title()`) and so the LLM can disambiguate semantically similar entries (TRUE_STORY vs the keyword-side BIOGRAPHY). Build-time fail-fast if SourceMaterialType grows a member without a metadata entry.
- **ConceptTag definitions come from the enum's `description` attribute.** No hand-writing needed. Display is `.name.replace("_", " ").title()` — safe because no concept tag names contain acronyms.
- **No `family` grouping metadata in this commit.** The 21 canonical concept families from finalized_search_proposal.md Endpoint 5 are useful for prompt grouping but not strictly required to make the registry functional. Can be added later when the step 3 keyword prompt is written.

### Planning Context
Session established the step 3 keyword endpoint design: single LLM call, single ID output, LLM always picks (no "none of these fit" option), no candidate pool cap. Step 3's job is pure semantic best-fit over the full vocabulary with definitions — step 2 already handled routing. See search_improvement_planning/finalized_search_proposal.md §Endpoint 5 and the open_questions entries marked DECIDED for keyword vocab mapping.

### Testing Notes
`unit_tests/test_unified_classification.py` parametrizes over every member of OverallKeyword, SourceMaterialType, and ALL_CONCEPT_TAGS. For each it asserts name, display, source, source_id, and backing_column; shadowed members (COLLISIONS) must resolve to the higher-precedence source. Also verifies total count, UnifiedClassification matches registry keys, `entry_for()` round-trips every member, and `(source, source_id)` pairs are globally unique. Definition content is not asserted — only that it is a non-empty string — per the explicit test scope.

## Award endpoint: post-review fixes (today date + Oscar scope discipline)
Files: search_v2/stage_3/award_query_generation.py
Why: Two issues surfaced in review. (1) `award_names` guidance for "Oscar-winning" queries told the model to emit both ceremony AND prize-name filters, which adds specificity the user didn't ask for and could miss Technical/Honorary Academy Awards stored under different award_name values. (2) The module had no `today` parameter, so relative year terms ("recent", "this decade") fell back to the LLM's training-time knowledge — inconsistent with the metadata module pattern.
Approach: Rewrote the Oscar note as a principle ("only emit an award name when the user is specifically distinguishing one prize object from others at the same ceremony") rather than a keyword rule. Added `today: date` parameter to `generate_award_query()` mirroring `generate_metadata_query()`, injected `today: {iso}` into the user prompt, and updated the YEARS section to resolve relative terms against the supplied date rather than intent_rewrite context. `_TASK` section now lists `today` as an explicit input.

## Award endpoint: query generation module + system prompt
Files: search_v2/stage_3/award_query_generation.py, schemas/award_translation.py

### Intent
Implements the LLM translation layer for the award endpoint. Takes a step-2 description + routing_rationale and produces an `AwardQuerySpec` by calling the shared LLM router.

### Key Decisions
- **6-section prompt structure**: task → positive-presence invariant → scoring shape (modes + five patterns) → filter axes (ceremonies, award_names, categories, outcome, years) → Razzie handling → output guidance. Parallel to franchise's 6-section structure.
- **Ceremony name mapping in prompt**: teaches the LLM to map user-facing names ("Oscar", "Cannes", "SAG") to exact stored string values ("Academy Awards, USA", "Cannes Film Festival", "Actor Awards"). Critical because `use_enum_values=True` on the schema means the LLM emits string values not Python enum names.
- **Scoring patterns as principle-based categories**: five named patterns with mode + mark guidance defined by what each pattern IS (not keyword shortcuts). "Oscar-winning" is explicitly taught as specific-filter-no-count (not generic award-winning) because the ceremony filter is present.
- **Razzie explicit-opt-in**: default exclusion taught as a named rule with explicit signal examples. "Worst movies" queries are explicitly redirected away from Razzie inference.
- **Schema comment cleanup**: stripped LLM-facing guidance from `#` comments in `award_translation.py`. All guidance now lives in the system prompt. Kept brief developer notes: data type info, null semantics, and the scoring formulas.

### Testing Notes
No tests added (per test-boundaries rule). Smoke calls to verify:
- "award-winning films" → scoring_shape_label="generic award-winning", mode=threshold, mark=3, all filters null
- "Oscar Best Picture winners" → ceremonies=["Academy Awards, USA"], award_names=["Oscar"], categories=["Best Picture"], outcome=winner, mode=floor, mark=1
- "most decorated films at Cannes" → ceremonies=["Cannes Film Festival"], scoring_shape_label="superlative", mode=threshold, mark=15
- "nominated at Sundance 2023" → ceremonies=["Sundance Film Festival"], outcome=nominee, years(2023,2023), mode=floor, mark=1
- "Razzie winners" → ceremonies=["Razzie Awards"], outcome=winner, mode=floor, mark=1
- "heavily decorated" → scoring_shape_label="qualitative plenty", mode=threshold, mark=5, all filters null
- "won at least 3 awards" → scoring_shape_label="explicit count: 3", mode=floor, mark=3, all filters null

## Franchise endpoint: query execution layer
Files: search_v2/stage_3/franchise_query_execution.py, db/postgres.py

### Intent
Implements the execution side of the franchise endpoint (step 3). Takes a `FranchiseQuerySpec` from the franchise LLM and produces binary-scored `EndpointResult` objects.

### Key Decisions
- **Sole data source**: `movie_franchise_metadata` only. `lex.inv_franchise_postings` no longer exists in the live DB.
- **Name/subgroup matching**: exact after `normalize_string()` in Python; SQL uses `LOWER()` on stored values as a safety net. Multi-variation arrays — any single variation matching counts as an axis hit.
- **Subgroup SQL**: `EXISTS (SELECT 1 FROM unnest(recognized_subgroups) AS sg WHERE LOWER(sg) = ANY($variations))` — handles the `TEXT[]` column without per-element expansion in Python.
- **AND semantics**: all axes combined in a single SQL WHERE clause; early exit is implicit (empty SQL result = empty EndpointResult).
- **No execution-side validation**: step 2 task decomposition is trusted; subgroup without a name axis is a valid search.
- **Binary scoring**: 1.0 for match, 0.0 for non-match (no gradient needed — franchise criteria are categorical).
- **Retry**: one retry on transient DB error; second failure returns empty result (soft failure, consistent with other stage 3 executors).
- **lineage_position**: `use_enum_values=True` on spec means the field is a string; resolved to SMALLINT ID via `LineagePosition(value).lineage_position_id` before SQL.

### Testing Notes
- Test normalized_name_variations against both `lineage` and `shared_universe` columns.
- Test subgroup match with variations that partially overlap the stored array.
- Test retry path: mock fetch to raise on first call, succeed on second; and raise on both.
- Test preference path: verify 0.0 entries for non-matching restrict_to_movie_ids.

## Award endpoint: added scoring_shape_label reasoning field
Files: schemas/award_translation.py, search_improvement_planning/finalized_search_proposal.md
Why: `concept_analysis` was doing double duty — inventorying both filter-axis signals (ceremony, category, outcome, year) and count/intensity signals (scoring shape). These are different evidence types scaffolding different decisions. The scoring shape decision (mode + mark) is the hardest in the schema: small models default to FLOOR/1 for everything, missing gradient intent. Separating the two forces the model to explicitly classify the intensity pattern before committing to numeric values.
Approach: Added `scoring_shape_label: str` between `concept_analysis` and `scoring_mode`. Brief classification from five fixed labels — follows the `value_intent_label` pattern in MetadataTranslationOutput (brief label, no consistency coupling instruction, primes via attention not by explicit constraint). Tightened `concept_analysis` comment to clarify it inventories filter axes only, not count/intensity language. Updated proposal with a reasoning-fields table documenting both fields, their positions, what they scaffold, and the rationale for the two-field split.

## Award endpoint: output schema + planning doc update
Files: schemas/award_translation.py, schemas/enums.py, search_improvement_planning/finalized_search_proposal.md
Why: Endpoint 3 (Awards) had a high-level prose spec but no output schema and several open design questions around scoring, data source dispatch, and Razzie handling.
Approach: Designed `AwardQuerySpec` with a unified flat shape — no mode discriminator at the outer level, scoring controlled by `scoring_mode` (FLOOR | THRESHOLD) + `scoring_mark`. Count unit is distinct prize rows in `movie_awards` (different ceremony, category, name, or year each count separately). Filters are Cartesian-ORed within an array, ANDed across arrays. `AwardYearFilter` sub-model handles single years (year_from == year_to) and ranges; gracefully swaps transposed values rather than erroring. `outcome` is a single nullable enum (None = both winners and nominees count). Added `AwardScoringMode` StrEnum to schemas/enums.py in the awards section alongside existing AwardCeremony / AwardOutcome. Razzie exclusion is a hardcoded execution concern — stripped from any count whose `ceremonies` field is null/empty; present in `ceremonies` = user explicitly asked for it. Fast path (award_ceremony_win_ids presence check) only when all filters null, outcome WINNER or null, FLOOR, mark=1. Updated finalized_search_proposal.md Endpoint 3 with the full scoring model, data source dispatch rules, and filter semantics.
Key decisions: count = prize rows not distinct ceremonies (prize rows match natural language like "won 11 Oscars"); ceiling mode dropped (no real use case, don't let schema shape hold back design); superlatives → THRESHOLD with high mark rather than a fourth uncapped mode; categories can stand alone without ceremonies. See conversation trail for rationale on each.
Testing notes: No tests added (per test-boundaries rule). Schema validation edge cases to watch: transposed year values (validator swaps), scoring_mark=0 (Field ge=1 rejects), stray False values are not coerced on this model (no boolean axes). Execution smoke: (1) fast path triggers only on the exact combination above; (2) Razzie present in ceremonies → included in COUNT; (3) categories-only query hits movie_awards across all non-Razzie ceremonies.

## Step 3 trending endpoint: execution module
Files: search_v2/stage_3/trending_query_execution.py
Why: Endpoint 7 (Trending) was the last step-3 endpoint without an execution module. It has no LLM-translation counterpart because step 2 flags the intent and execution is pure pass-through over precomputed Redis scores (concave-decay curve lives in `db/trending_movies.py`'s refresh job, not in the search path).
Approach: `execute_trending_query(*, restrict_to_movie_ids: set[int] | None = None) -> EndpointResult`. Single entry point mirroring the sibling endpoints' dual-mode signature. Dealbreaker path (restrict is None) emits one ScoredCandidate per movie in the Redis `trending:current` hash — the natural match set also doubles as Phase 4a candidate contribution (includes every entry Redis returns, currently up to 500 from the TMDB weekly trending API). Preference path emits exactly one ScoredCandidate per supplied id, 0.0 for ids absent from the hash. Reused `read_trending_scores()` which already handles the missing-key case by returning `{}` with a warning (graceful degradation per open_questions.md §Pipeline failure points). No rank recomputation at query time — score shape lives in the refresh job so there's a single curve definition. No runtime score clamp: refresh writes in [0,1] by construction; ScoredCandidate Pydantic validation surfaces corrupted data rather than silently truncating. `_build_endpoint_result` helper kept (instead of inlined) to match the entity/metadata pattern and keep the dealbreaker/preference branching explicit.
Design context: See finalized_search_proposal.md §Endpoint 7 for the scoring contract, §Endpoint Return Shape for the uniform [EndpointResult, ScoredCandidate] shape, and open_questions.md:733 for the graceful-degradation decision on missing trending data. Planning trail at /Users/michaelkeohane/.claude/plans/include-all-500-or-humble-ember.md.
Testing notes: No unit tests added (test-boundaries rule). Smoke check: (1) dealbreaker mode returns ~500 entries with concave distribution (rank 1 ≈ 1.0, rank 500 = 0.0); (2) preference mode returns exactly `len(restrict_to_movie_ids)` entries with 0.0 for non-trending ids; (3) empty Redis hash → empty EndpointResult in dealbreaker mode, all-zeros in preference mode; (4) empty set restrict → empty EndpointResult (preference-mode contract). Watch for: (a) the documented key-prefix discrepancy between `full_search_capabilities.md` (`{ENV}:trending:current`) and `db/redis.py` (`trending:current`) — if `redis_key()` adds an env prefix this endpoint inherits it transparently, but worth verifying when wiring the orchestrator; (b) orchestrator must tolerate 0.0-score entries in preference-mode output same as the metadata endpoint; (c) trending score 0.0 for rank-N (tail of the list) is indistinguishable from "not trending" in preference mode — acceptable because both mean "don't rank up for trending" but note for debugging.

## Step 3 franchise endpoint: stage 3 query-generation module + system prompt
Files: search_v2/stage_3/franchise_query_generation.py
Why: Franchise endpoint had an output schema with two scoped reasoning fields but no stage-3 translation module. Built the module from scratch mirroring the entity and metadata endpoint patterns so the seven-axis FranchiseQuerySpec can be produced from step 2's `description` + `routing_rationale` + step 1's `intent_rewrite`.
Approach: Six modular `_SECTION` constants concatenated into `SYSTEM_PROMPT` (task → positive-presence invariant → seven searchable axes → canonical naming → AND semantics / scope discipline → output field guidance). Axis definitions are principle-based: each axis is defined by what it IS and the signal phrase class that populates it, not by enumerated bad triggers — follows the "principle-based constraints, not failure catalogs" convention. The `launched_franchise` vs. `launched_subgroup` ambiguity (e.g., "started the MCU") is taught by surfacing the ambiguity in concept_analysis and committing to the reading that best fits intent_rewrite, not by a keyword-shortcut rule. Canonical naming section inherits the ingest-side generator's convention verbatim (lowercase everything, spell digits as words, expand "&" → "and", expand abbreviations only when the expanded form is also in common use, drop first names on director-era labels) so search-side and ingest-side strings converge on the same canonical form. AND semantics section explicitly warns against populating speculative axes to "describe the franchise as a whole" — every populated axis silently tightens the query. Output section carries the reasoning-field framing: `concept_analysis` as evidence inventory with explicit-absence discipline per axis, `name_resolution_notes` as telegraphic semicolon-separated canonical-form inventory with the "not applicable — purely structural" sentinel path for structural-only queries, spelling/punctuation variants explicitly excluded (trigram handles them). `generate_franchise_query()` takes intent_rewrite / description / routing_rationale / provider / model (all required, no defaults) and returns the standard `(output, input_tokens, output_tokens)` tuple mirroring the entity module's signature. No Field descriptions were stripped from `schemas/franchise_translation.py` — a grep confirmed the schema already has zero Field descriptions (guidance has always lived in developer comments + system prompt), so that part of the task was a no-op.
Design context: Follows conventions.md "Prompt Authoring Conventions" (cognitive-scaffolding field ordering, evidence-inventory reasoning, brief pre-generation fields, principle-based constraints, evaluation guidance over outcome shortcuts, example-eval separation, no schema details leaked to the LLM) and the "exact-match convergence for LLM-generated strings" invariant that drives the inherited canonical-naming rules. Parallels the entity-module pattern at search_v2/stage_3/entity_query_generation.py and the metadata-module pattern at search_v2/stage_3/metadata_query_generation.py. See finalized_search_proposal.md Endpoint 4 and the ingest-side franchise prompt at movie_ingestion/metadata_generation/prompts/franchise.py for the shared definition set.
Testing notes: No tests added (per test-boundaries rule). Manual smoke calls planned for: "is a Marvel movie" → lineage_or_universe_names=["marvel cinematic universe", "marvel"]; "Marvel spinoffs" → name + is_spinoff=True; "sequels" → lineage_position=SEQUEL with name null; "movies that started a franchise" → launched_franchise=True with name null; "started the MCU" → launched_franchise=True (franchise birth reading); "started Phase Three" → name + recognized_subgroups + launched_subgroup=True; "Star Wars prequels" → name + lineage_position=PREQUEL. Watch for: (a) small models collapsing `concept_analysis` into rationalization instead of evidence inventory (quotes missing), (b) `name_resolution_notes` emitted as prose instead of telegraphic form, (c) overgenerating spelling variants on `lineage_or_universe_names` despite the explicit exclusion (trigram handles them), (d) populating speculative axes to "be thorough" and over-constraining the query, (e) emitting False on boolean axes (schema validator coerces to null, but indicates prompt drift), (f) launched_franchise vs. launched_subgroup errors on ambiguous queries — concept_analysis ambiguity note should grow load-bearing or the OUTPUT guidance needs tightening.

## Step 3 franchise endpoint: reasoning fields on output schema
Files: schemas/franchise_translation.py, search_improvement_planning/finalized_search_proposal.md
Why: The franchise output schema had a single reasoning field (`concept_analysis`) bundling axis-signal detection and name expansion into one trace. Small step-3 LLMs benefit from scoped reasoning fields placed immediately before the decisions they ground — splitting into two fields targets the two distinct failure modes independently: (a) wrong axis presence/absence (pattern-matching on the franchise word), and (b) name expansion underfill/overfill (defaulting to 1 entry when alternate canonical forms exist, or padding with spelling variants that trigram already handles).
Approach: Tightened `concept_analysis` (required, first) to a pure axis-signal evidence inventory — quote signal phrases from `description` / `intent_rewrite` and pair each with its target axis; explicit-absence paths required; surface ambiguity for cases like "started the MCU" (launcher of franchise vs. subgroup). Added `name_resolution_notes` (nullable, placed immediately before `lineage_or_universe_names`) — brief telegraphic parametric-knowledge inventory of alternate canonical forms for the IP, or the sentinel "not applicable — purely structural" for structural-only queries. Short-label form per the "brief pre-generation fields, no consistency coupling" convention. Scaffolds both the lineage-name list length and subgroup list length with one field since the parametric-knowledge pattern is identical for both. Developer comments carry the full reasoning procedure (system prompt will implement it later); no `Field(description=...)` on reasoning fields per the "no LLM-facing guidance in schemas" pattern shared with entity_translation.py and metadata_translation.py.
Design context: Follows conventions.md "Cognitive-scaffolding field ordering," "Evidence inventory, not rationalization for reasoning fields," "Brief pre-generation fields, no consistency coupling." Parallels the entity endpoint's two-reasoning-field pattern (`entity_analysis` scaffolds entity identity; `prominence_evidence` scaffolds prominence mode). Explicitly chose NOT to mirror the ingest-side `FranchiseOutput` six-reasoning-field pattern — search-side LLM translates an already-classified query rather than classifying a movie from source data, so per-axis traces would inflate output tokens without proportional accuracy gain.
Testing notes: No unit tests added (per test-boundaries rule). Empirical evaluation should watch for: (a) `concept_analysis` collapsing into rationalization rather than evidence inventory (model justifies output instead of citing input), (b) `name_resolution_notes` overgenerating spelling variants despite the explicit exclusion (system prompt needs tight examples), (c) `name_resolution_notes` emitted as a full sentence instead of telegraphic form (templating risk), (d) ambiguity-surfacing in `concept_analysis` for launched_franchise vs. launched_subgroup actually changing the boolean choice downstream vs. being cosmetic. Updated finalized_search_proposal.md Endpoint 4 with a reasoning-fields subsection documenting both fields and the rationale for rejecting per-axis traces.

## Step 3 franchise endpoint: output schema + proposal update
Files: schemas/franchise_translation.py, search_improvement_planning/finalized_search_proposal.md
Why: Franchise endpoint (Step 3, Endpoint 4) had no output schema. Planning conversation aligned on the seven searchable axes, AND semantics for multi-axis concepts, up-to-3 name variations for both franchise name and subgroup, and inheriting canonical-naming guidance from the ingest-side generator.
Approach: New `schemas/franchise_translation.py` — `FranchiseQuerySpec` with flat nullable per-axis fields. Axes: `lineage_or_universe_names` (list, max 3, searched against both `lineage` and `shared_universe` via `lex.inv_franchise_postings` — always together because the ingest LLM flip-flops slots), `recognized_subgroups` (list, max 3, trigram post-lookup on 3-30 movies, only valid when franchise name is populated), `lineage_position` enum, and four structural booleans (`is_spinoff`, `is_crossover`, `launched_franchise`, `launched_subgroup`). Leading `concept_analysis` reasoning field for evidence-inventory scaffolding per conventions. Validators: subgroups require franchise name; stray `False` booleans coerced to None (direction-agnostic framing — only True or None are meaningful); at least one axis must be populated. Field ordering: reasoning → name → subgroup → lineage_position → structural booleans. Developer-only comments; all LLM-facing guidance will live in the system prompt (deferred). Updated finalized_search_proposal.md Endpoint 4 section with the schema reference, axis list, prompt-reuse requirement (inherit from `movie_ingestion/metadata_generation/prompts/franchise.py`), AND semantics, no-fallback policy for zero-result names, no pool cap, and binary-only preference scoring (franchise-recency gradient explicitly dropped). Updated Decisions Deferred list to mark franchise schema complete.
Design context: See finalized_search_proposal.md Endpoint 4 and ADR-067 (two-axis franchise schema). Canonical naming convention shared with `schemas/metadata.py::FranchiseOutput` and the ingest prompt. Searchable axes and alignment decisions captured during the planning conversation.
Testing notes: No unit tests added (per test-boundaries rule). System prompt not yet authored — will follow the entity/metadata pattern. Watch for: (a) LLM populating `recognized_subgroups` without `lineage_or_universe_names` (validator catches), (b) obscure franchises missing all 3 name variations (accepted — no further fallback), (c) step-2 routing drift sending "generic remake" queries here instead of the keyword endpoint (REMAKE value retained in the enum but not typically consumed at search time).

## Stage 3 metadata endpoint: execution module
Files: search_v2/stage_3/metadata_query_execution.py
Why: Translation module (`metadata_query_generation.py`) produces the spec; this module runs it. One function per the agreed design, dual-mode on a single entry point.
Approach: `execute_metadata_query(output, restrict_to_movie_ids=None) -> EndpointResult`. Dispatches on `target_attribute` to ten handlers mirroring the proposal's per-attribute specs. Dealbreaker mode (restrict is None) applies the widened gate as SQL WHERE; preference mode fetches every supplied id and emits a ScoredCandidate per id (0.0 for null data or out-of-range). Retry once on any exception; second failure returns an empty EndpointResult per the graceful-degradation rule. Gradient shapes mirror `db/metadata_scoring.py` (date grace: 1yr min / 5yr max / 3yr unbounded / 2yr exact; runtime grace: 30min; maturity ordinal distance with 0.5 at dist=1; reception linear ramp with centers 55/50, width 40). New decisions from this session: (1) country position decay uses `exp(-(pos-1)/1.3)`; (2) popularity/reception dealbreakers cap at 2000 rows sorted by the scoring dimension (only attributes without a natural WHERE gate); (3) UNRATED is excluded from the dealbreaker gate for non-EXACT-UNRATED queries and scores 0.0 in preference mode; (4) access-type-only streaming dealbreakers use an EXISTS-on-unnested-keys scan (full-table, rare path — flagged). DB access uses the `db.postgres.pool` connection pattern directly via a local `_fetch` helper; no per-candidate queries (always `WHERE movie_id = ANY($1)` in preference mode).
Design context: See finalized_search_proposal.md Endpoint 2 per-attribute specs + Step 3.5 Endpoint Return Shape. Gradient parity with `db/metadata_scoring.py` per CLAUDE.md guidance. User-answered open items recorded in this session: pool caps only when sort-only (no filter); single function with `restrict_to_movie_ids: set[int] | None`; UNRATED → 0.0 in preference mode; country gradient exponential.
Testing notes: No unit tests added (test-boundaries rule). Dispatch-coverage smoke checked — all 10 MetadataAttribute values resolve to a handler via both enum-member and raw-string keys. Watch for: (a) access-type-only dealbreaker latency (full-table EXISTS scan); (b) country exponential at pos 2 yielding 0.46 — steeper than the proposal's 0.7-0.8 anchor, revisit if results feel too punishing; (c) degenerate rating windows (e.g., LESS_THAN G → empty range) returning score 0 for everyone — correct but silent; (d) preference mode returns one entry per supplied id including 0.0s — orchestrator needs to tolerate zero-score entries.

## Stage 3 endpoint return shape
Files: schemas/endpoint_result.py, search_improvement_planning/finalized_search_proposal.md
Why: Every stage 3 endpoint (dealbreaker or preference, any endpoint type) needs a uniform return shape so orchestrator-side assembly/reranking code can consume them without per-endpoint branching.
Approach: New `schemas/endpoint_result.py` defines `ScoredCandidate` (movie_id + score in [0,1]) and `EndpointResult` (list of ScoredCandidate). Intentionally minimal — orchestrator owns direction (inclusion/exclusion), exclusion mode (hard-filter vs E_MULT penalty), preference weighting (regular/primary/prior), and scoring mode by wrapping results with step-2 metadata. Considered separate DealbreakerResult/PreferenceResult classes but collapsed to one since fields are identical and role lives with the orchestrator. Added "Endpoint Return Shape" subsection to finalized_search_proposal.md Step 3.5.
Design context: See finalized_search_proposal.md Step 3.5 + Phase 4a–4c.

## Step 3 metadata endpoint: query-translation module + system prompt
Files: search_v2/stage_3/metadata_query_generation.py, schemas/metadata_translation.py
Why: Stage 3 metadata endpoint had a finalized output schema but no query-translation module. Built the module from scratch mirroring the entity endpoint's pattern, and relocated the reasoning-field guidance out of `Field(description=...)` into the system prompt so the schema matches the sibling entity_translation.py style (no LLM-facing text in the schema).
Approach: New module `search_v2/stage_3/metadata_query_generation.py` — eight modular `_SECTION` constants concatenated into `SYSTEM_PROMPT` (task → direction-agnostic → literal-translation separation → 10 target attributes → per-attribute sub-object rules → one-sub-object discipline → per-field output guidance). Direction-agnostic and literal-translation sections match the entity-module invariants. Target-attribute section orders attributes by frequency and calls out the three collision pairs step 2 can still produce (audio_language vs country_of_origin, popularity vs reception, reception vs stray award references). Sub-object section teaches principle-based boundaries for every match_operation and enum pole, including today-date-anchored resolution for relative temporal terms. Output section carries the reasoning-field framing: `constraint_phrases` as evidence inventory with empty-evidence-does-not-mandate-empty-output clause, `value_intent_label` as brief label that commits to direction and boundary with no consistency-coupling language. `generate_metadata_query()` takes intent_rewrite / description / routing_rationale / today / provider / model (all required, no defaults) and returns the standard `(output, input_tokens, output_tokens)` tuple. Schema edit: dropped `description=...` kwargs on `constraint_phrases` and `value_intent_label` only; kept `default=[]`, `max_length=80`, field ordering, and all developer comments intact. Verified with a clean import and `openai.lib._pydantic.to_strict_json_schema` — neither reasoning field emits a `description` into the JSON schema.
Design context: Follows the entity-module pattern at [search_v2/stage_3/entity_query_generation.py](search_v2/stage_3/entity_query_generation.py) and the stage_2 section-triad style for attribute boundaries. See conventions.md "Prompt Authoring Conventions" (cognitive-scaffolding field ordering, evidence-inventory reasoning, brief pre-generation fields, principle-based constraints, example-eval separation, no schema details leaked to the LLM). See finalized_search_proposal.md Step 3 Endpoint 2 and full_search_capabilities.md §1/§6 for the attribute surface.
Testing notes: No tests added (per test-boundaries rule). Manual smoke calls planned for: "80s movies" → release_date BETWEEN; "French films" vs "French audio" → country vs language routing with different constraint_phrases; "hidden gems" popularity half → NICHE; "under 90 minutes" → runtime LESS_THAN 90; "PG-13 or lower" → maturity_rating LTE; "available on Netflix" → streaming with services=[Netflix]. Watch for: (a) small models collapsing `value_intent_label` into full sentences or restating cited phrases; (b) unnecessary sub-object population when only target_attribute's sub-object should be filled; (c) audio/country routing errors if the disambiguating token isn't cited in constraint_phrases.

## Step 3 metadata endpoint: reasoning fields on output schema
Files: schemas/metadata_translation.py
Why: Small step-3 LLMs need chain-of-thought scaffolding to avoid two failure modes — wrong `target_attribute` routing (e.g., "French films" → country vs "French audio" → language) and wrong sub-object population (match_operation direction, country-list expansion, temporal resolution of "recent"). Added two brief reasoning fields placed per the cognitive-scaffolding field-ordering convention.
Approach: Added `constraint_phrases: list[str]` as the FIRST field (evidence inventory — verbatim phrases from description/intent_rewrite that signal the constraint; grounds `target_attribute` routing in text rather than pattern-matching; follows "evidence inventory, not rationalization" convention; empty allowed). Added `value_intent_label: str` (max_length=80) BETWEEN `target_attribute` and the sub-objects as a brief label (~3-8 words) stating the literal intended value to prime match_operation direction and boundary selection for the populated sub-object. Field descriptions carry compact guidance per conventions; worked examples stay in system prompt. No consistency-coupling language — labels prime without templating.
Design context: See conventions.md "Cognitive-scaffolding field ordering," "Evidence inventory, not rationalization for reasoning fields," and "Brief pre-generation fields, no consistency coupling." See finalized_search_proposal.md Endpoint 2 (Movie Attributes) for the target_attribute → sub-object execution model.
Testing notes: Empirical evaluation should watch for (a) routing errors still occurring despite constraint_phrases (if so, system prompt needs the routing-distinction examples), (b) value_intent_label collapsing into full sentences or restating sub-object values (tighten description if seen), (c) whether value_intent_label adds value on simple-enum targets (budget/box_office/popularity/reception) — may be dropped there later if it proves redundant.

## Full search capabilities catalog
Files: search_improvement_planning/full_search_capabilities.md | Comprehensive inventory of all data sources available for search (Postgres tables/columns, Qdrant vector spaces/payload, Redis, lexical schema, tracker DB unpromoted fields), organized by storage location with search utility notes for each. Cross-referenced from v2_data_architecture.md, codebase schemas, and other planning docs.

## V2 finalized search proposal and planning doc updates
Files: search_improvement_planning/finalized_search_proposal.md, search_improvement_planning/open_questions.md, search_improvement_planning/types_of_searches.md
Why: Captured all finalized decisions from design conversation into the official V2 proposal document.
Approach: finalized_search_proposal.md contains the full three-step pipeline architecture (query understanding → per-source search planning → execution & assembly), including semantic dealbreaker demotion, exclusion handling via elbow-threshold penalties, pure-vibe flow, quality prior as separate dimension, and gradient metadata scoring. open_questions.md updated with 4 new V2 pipeline questions (elbow detection method, multi-interpretation triggers, semantic demotion display, exclusion query formulation). types_of_searches.md updated with 3 new V2 edge case categories (#15 pure-vibe, #16 semantic exclusion on non-tagged attributes, #17 dealbreaker demotion).

## Restructured V2 pipeline to 4 steps with 7 named endpoints
Files: search_improvement_planning/finalized_search_proposal.md
Why: Fleshed out the search execution layer with concrete, individually-addressable data endpoints. Each endpoint represents a single conceptual data domain with its own LLM (or deterministic function) for translating abstract intent into executable queries.
Approach: Renumbered the pipeline from 3 steps to 4: (1) flow routing, (2) query understanding/decomposition, (3) search execution across 7 endpoints, (4) assembly & reranking. The 7 endpoints are: Entity Lookup (lex.* posting tables), Movie Attributes (movie_card columns + denormalized award wins), Awards (movie_awards table), Franchise Structure (movie_franchise_metadata structural columns), Keywords & Concept Tags (keyword_ids + concept_tag_ids), Semantic (8 Qdrant vector spaces), Trending (Redis). Each endpoint section documents its data sources, what its LLM knows, how it handles candidate generation vs. preference scoring, and example queries. Step 2 routing enum updated from {lexical, metadata, keyword, semantic} to {entity, metadata, awards, franchise_structure, keyword, semantic, trending}. Added a routing distinction table showing how step 2 distinguishes overlapping endpoints using surface-level signals only.
Design context: Endpoints drawn at boundaries where the step 2 LLM can distinguish between them without schema knowledge. Step 2 routes directly to the specific endpoint (not to a broad category with sub-routing).

## Pipeline review: step 3/3.5 split, gap closures, and clarifications
Files: search_improvement_planning/finalized_search_proposal.md
Why: Critical review of step transitions identified gaps in preference scoring timing, pure-vibe flow detection, title substring coverage, reference movie handling, sort order expression, and routing failure handling.
Approach: Split step 3 into Query Translation (3) and Search Execution (3.5) — LLMs generate query specs in parallel, dealbreaker searches execute immediately as each responds, preference queries await candidate IDs then fire instantly. Added explicit pure-vibe flow checkpoint (triggers when no non-semantic inclusion dealbreakers exist, separate codepath). Expanded Entity Lookup to include title substring matching (ILIKE). Added reference movie parametric knowledge note (no tmdb_id resolution in standard flow). Added sort-order-as-preference via is_primary_preference. Added dual dealbreaker+preference implementation note for keyword concepts with centrality spectrums. Added routing failure acceptance note (prompt design concern, no retry mechanism). Expanded pure-vibe exclusions to include deterministic exclusions + pre-filter investigation note.
Design context: All changes from discussion — no new product decisions introduced.

## Search planning doc reversals and rationale alignment
Files: search_improvement_planning/finalized_search_proposal.md, search_improvement_planning/open_questions.md, search_improvement_planning/new_system_brainstorm.md, search_improvement_planning/types_of_searches.md
Why: The search design discussion reversed several earlier assumptions, and the planning docs needed to be brought back into alignment without introducing any new product decisions.
Approach: Updated the finalized proposal to add major-flow routing before standard decomposition, defend per-source step-2 LLMs as schema translators rather than re-interpreters, add `is_primary_preference` as the only preference-strength mechanism, split quality from notability/mainstreamness conceptually, and change semantic exclusions from effective removal to calibrated penalty-only behavior. Updated older planning docs to remove contradictions on boolean/group logic, preference weighting, similarity-flow routing, trending candidate injection, and quality-vs-discovery framing. Moved unresolved details that emerged from these reversals into open_questions.md instead of finalizing them prematurely.
Design context: Based on the current V2 planning set in search_improvement_planning/ and the latest design conversation clarifying that V1 should favor simpler tiering behavior over richer clause logic, and that "hidden gems"/"underrated" are not the same as inverted quality.
Testing notes: Verified by diff/grep that the finalized proposal no longer claims hidden gems/underrated are inverted quality, no longer frames semantic exclusions as effective removal, now documents major-flow routing and `is_primary_preference`, and that the supporting brainstorming/query-type docs no longer contradict those decisions.

## Step 1 flow routing output schema design and implementation
Files: schemas/flow_routing.py (new), schemas/enums.py, search_improvement_planning/finalized_search_proposal.md

### Intent
Define the structured output schema for the step 1 flow routing LLM and document the design rationale in the finalized proposal.

### Key Decisions
- **Top-level `interpretation_analysis` field** — one sentence assessing ambiguity before generating interpretations. Follows evidence-inventory pattern to prevent the model from manufacturing branching.
- **Per-interpretation `routing_signals`** — one short sentence per interpretation citing concrete query words that determined flow classification. Originally proposed at top level; moved to per-interpretation because each interpretation may route to a different flow with different evidence.
- **`intent_rewrite` always required** — applies to all flows (not just standard) for simplicity. Serves as the primary scaffolding field and feeds into step 2 for standard-flow branches.
- **Field ordering: routing_signals → intent_rewrite → flow → display_phrase → title** — evidence before classification, open-ended generation before constrained enums. Follows cognitive-scaffolding convention from metadata generation.
- **`display_phrase` always required** — even single-interpretation queries benefit from a display header in the app UI.
- **`SearchFlow` enum** added to schemas/enums.py alongside other shared enums.

### Planning Context
Schema design informed by prompt authoring conventions codified during metadata generation work (evidence-inventory fields, brief pre-generation fields, cognitive scaffolding ordering, abstention-first framing). See finalized_search_proposal.md Step 1 Output Structure section for full rationale per field.

## Step 1 flow routing: full decision resolution
Files: search_improvement_planning/finalized_search_proposal.md, search_improvement_planning/open_questions.md, search_improvement_planning/types_of_searches.md
Why: Resolved the two remaining open questions for step 1 (exact routing triggers and multi-interpretation branching criteria) and corrected outdated assumptions in supporting docs.
Approach: Rewrote the Step 1 section in finalized_search_proposal.md with tight flow definitions: exact title flow restricted to literal titles only (misspellings/partials/alternate titles OK, descriptions never), similarity flow requires zero qualifiers and a single named title, standard flow is everything else. Interpretation branching is now cross-flow (a query can branch into different major flows, e.g. "Scary Movie" → exact title OR standard flow). Branching bar: an intelligent person would agree interpretations are reasonably similar in likelihood. Corrected two outdated assumptions: (1) known-movie flow no longer includes fragmentary recall (descriptions go to standard flow), (2) similarity search is now a first-class step 1 route (not deferred). Updated Step 2 multi-interpretation subsection to remove the deferred open question and clarify step 2 only runs for standard-flow branches. Updated Reference Movies subsection for multi-reference queries and title ambiguity handling.
Design context: All changes from discussion — user explicitly directed that descriptions always go to standard flow regardless of identifiability, that title search with no DB matches returns "not found" with no fallback, and that any qualifier on similarity queries routes to standard flow.

## Search V2 stage 1 LLM call scaffold
Files: search_v2/__init__.py (new), search_v2/stage_1.py (new)
Why: Wire up the flow routing LLM call so we can test different provider/model combinations.
Approach: `route_query()` accepts provider, model, and kwargs with no defaults, validates the query, and delegates to `generate_llm_response_async` with `FlowRoutingResponse` as the structured output schema. System prompt is a TODO placeholder — will be implemented separately.

## Step 1 flow routing system prompt
Files: search_v2/stage_1.py
Why: Implement the system prompt that guides the step 1 flow routing LLM.
Approach: Modular 4-section prompt (`_TASK + _FLOWS + _BRANCHING + _OUTPUT`) following the pattern from concept_tags.py. Flow definitions are purpose-driven (explain what downstream pipeline each feeds and WHY boundaries exist) rather than rule-list style. Branching uses abstention-first framing (default is single interpretation, model must justify ambiguity). Output guidance encodes the cognitive chain: routing_signals (evidence inventory) → intent_rewrite (scaffolding commitment) → flow (classification follows naturally). No keyword-matching shortcuts — the model evaluates what the user intends using cited query text.
Design context: Prompt authoring conventions from metadata generation (evidence-inventory fields, brief pre-generation fields, evaluation guidance over outcome shortcuts, principle-based constraints). See finalized_search_proposal.md Step 1 for full design rationale.

## Step 1 prompt refinement: anti-inference, rewrite boundaries, and title-collision clarity
Files: search_v2/stage_1.py
Why: Tighten the routing prompt after review so small models understand how to resolve intent without inventing unsupported constraints, and so duplicate-title cases do not get misrouted away from exact-title flow.
Approach: Added an upfront rule that query text is the primary evidence and movie knowledge may recognize typed titles but not invent unsupported interpretations. Replaced the generic "when in doubt, route to standard" shortcut with evidence-based fallback wording. Tightened branching so only materially different downstream searches justify multiple interpretations. Made `routing_signals` more literal by asking for exact spans/patterns and decisive boundary cues. Clarified `intent_rewrite` to allow resolving strongly entailed latent intent (including trait-style rewrites for similarity queries) while forbidding added constraints, preferences, or quality assumptions. Expanded `title` guidance to state that same-title collisions/remakes still stay in `exact_title`, and that uniqueness is handled downstream rather than by rerouting.
Design context: Follows the repo's prompt design preference for evaluation guidance over outcome shortcuts and preserves the step 1 contract from finalized_search_proposal.md: route by supported interpretation, do not guess titles from descriptions, and keep title-search flow independent from DB-level uniqueness.

## Step 2 design decisions: preferences, priors, and semantic grouping
Files: search_improvement_planning/finalized_search_proposal.md, search_improvement_planning/open_questions.md

### Intent
Resolve open design questions for step 2 (query understanding) and update
planning docs with all finalized decisions before implementation begins.

### Key Decisions
- **Quality and notability priors** — 4-value enums (`enhanced`/`standard`/`inverted`/`suppressed`) for each dimension independently. `suppressed` is a second-order inference: depends on whether a dominant primary preference exists, so these fields must come after dealbreakers/preferences in the output schema.
- **Preference direction eliminated** — all preferences are framed as traits to promote. Negative user intent reframed as positive opposite ("not recent" → "prefer older films"). Hard exclusions ("not zombie") are dealbreakers, not preferences. Removes the `direction` field from preferences entirely.
- **Semantic preference grouping** — all semantic preferences (qualifiers on the desired experience) consolidated into a single rich description. Step 3 semantic endpoint decomposes into per-space queries. Exception: disjunctive qualifiers ("funny or intense") remain separate preferences.
- **Semantic dealbreaker vs preference distinction** — dealbreakers are distinct binary-ish traits the user treats as defining requirements where no deterministic source can evaluate them ("zombie," "female empowerment," "car chase"). Preferences are qualifiers on the experience ("funny," "dark," "slow-burn"). Dealbreakers define what kind of movie; preferences describe what it should feel like.
- **Multi-primary preference handling** — treat all marked preferences as co-primary (elevated equally, no single axis dominating).
- **Temporal bias** — not a separate field. Handled as a metadata preference routed to `metadata`, step 3 translates to concrete date parameters with grace periods.

## Keyword endpoint: reduce shortlist bias and clarify family reasoning
Files: search_v2/stage_3/keyword_query_generation.py
Why: Review of the new step 3 keyword endpoint found a few prompt-level risks for small models: `candidate_shortlist` was acting like a hard gate on the final enum choice, `concept_analysis` mixed the 21-family taxonomy with an unrelated coarse label set, and the routing context was framed strongly enough to risk biasing classification.
Approach: Reframed the step 2 context as a lightweight `routing_hint` on the prompt surface while keeping the external function contract unchanged. Changed the registry rendering to `ENUM_NAME: definition` so the selection surface matches the required enum output exactly. Rewrote `concept_analysis` to point back to the family list above instead of forcing coarse labels. Relaxed `candidate_shortlist` so it can contain one clear winner or a small set of genuine competitors, and removed the rule that the final classification must be one of the shortlisted entries. Added a compact alias section (`Bollywood`, `biopic`, `does the dog die?`, `shorts`, `twist ending`) plus broader boundary examples (`French` identity vs audio, `short films` vs runtime) to improve semantic mapping without expanding the schema.
Design context: Follows the finalized proposal's step 3 principle that endpoint LLMs are schema translators, not re-interpreters, and the repo's evidence-inventory lesson that reasoning fields should guide decisions without becoming brittle decision gates.
Testing notes: No tests run or modified per repo test-boundary rules. Main expected behavior changes are prompt-only: fewer invented shortlist competitors on obvious matches, less over-weighting of routing metadata, and cleaner fallback to the closest definitionally supported enum.
- **Thematic centrality principle** — keyword/concept tag dealbreakers for thematic concepts (zombie, Christmas, heist) should also include centrality in the grouped semantic preference. Structural concepts (sequel, award-winning) don't need this.
- **Keyword vocab in step 2** — trait descriptions covering what the vocabulary can match, not the full 225-term list. Step 3 resolves to specific IDs. The exact trait description list needs development before implementation.

### Planning Context
Decisions emerged from discussion analyzing the step 2 output structure, the current metadata scoring patterns (grace periods, linear decay), and the semantic search grouping tradeoffs. Resolved 4 open questions in open_questions.md (quality/notability wire shape, temporal bias representation, multi-primary handling, preference interaction with dealbreakers).

## Added key design principle: single requirement → dealbreaker + preference pattern
Files: search_improvement_planning/finalized_search_proposal.md
Why: Analysis of whether single user requirements could need multi-endpoint dealbreakers (tier inflation risk). Concluded that compound dealbreakers aren't needed for V1 — the genuine cases are too narrow (essentially just "remakes" straddling keyword and franchise_structure). The real cross-endpoint pattern is dealbreaker + preference: keyword dealbreaker for candidate generation (+1 tier) paired with semantic preference for within-tier centrality/specificity ranking. Added as design principle #5 with examples (scary movies, revenge on a bully, Christmas movies). Renumbered subsequent principles 6-8.

## Solidified all 7 endpoint definitions for step 2 LLM routing
Files: search_improvement_planning/finalized_search_proposal.md

### Intent
Define precise endpoint definitions, routing criteria, and boundary cases
for each of the 7 step 2 routing targets. These definitions will be fed
into the step 2 LLM prompt to enable accurate routing decisions.

### Key Decisions
- **Endpoint 1 (Entity):** Plain-English descriptions preserving all user
  qualifiers (role type, prominence, match scope). Includes character
  substring matching for generic roles. Clear boundary: no franchise, award,
  metadata, or semantic routing.
- **Endpoint 2 (Metadata):** Scoped to quantitative/logistical attributes
  only. Genre, source material type, and award_ceremony_win_ids moved OUT.
- **Endpoint 3 (Awards):** ALL award routing consolidated here, including
  generic "award-winning" (previously in metadata). Single entry point for
  anything award-related.
- **Endpoint 4 (Franchise Structure):** Sole source for franchise names AND
  structural roles. Clear boundary vs. entity (studios ≠ franchises) and vs.
  keyword (generic "remakes" = source material type keyword; franchise-specific
  remakes = franchise_structure).
- **Endpoint 5 (Keywords & Concept Tags):** Expanded to include genre_ids and
  source_material_type_ids (moved from metadata — categorical classification,
  not quantitative attributes). Step 2 LLM receives the full list of 11
  classification dimensions with all individual keywords/tags enumerated, not
  just trait descriptions. This prevents misrouting to keywords that don't
  exist. Full categorization: genre & sub-genre (~192 keywords organized by
  family), culture (~30 language keywords, renamed from "language"), animation
  technique (3), source material type (10), plus 7 concept tag categories
  (narrative structure, plot archetype, setting, character type, ending type,
  viewer experience, content warning).
- **Endpoint 6 (Semantic):** Explicitly documented as last resort for
  dealbreakers — whenever a deterministic endpoint can handle the concept, it
  must be used over semantic. Semantic is freely used for preferences even when
  other endpoints handle the same concept as a dealbreaker.
- **Endpoint 7 (Trending):** Temporal "right now" signal is the key
  distinguisher. "Popular" without temporal language routes to metadata
  (popularity_score), not trending.
- **Routing enum table and signal-to-route table** updated to reflect all
  moves (genre/source material → keyword, all awards → awards).
- **10 tricky boundary cases** documented for keyword endpoint, 5 for semantic
  endpoint, covering the most confusable routing decisions.

### Planning Context
Endpoint-by-endpoint discussion with user. Each endpoint was presented,
discussed, refined, and then written to the doc before moving to the next.
The keyword endpoint required the deepest analysis — full categorization of
all 225 keywords + 27 genres + 10 source material types + 25 concept tags
into 11 semantically distinct dimensions.

## Step 2 query understanding output schema and enums
Files: schemas/query_understanding.py (new), schemas/enums.py, search_improvement_planning/finalized_search_proposal.md

### Intent
Define the structured output schema for the step 2 query understanding LLM
and document the design rationale in the finalized proposal. This is the
schema that decomposes a standard-flow query into dealbreakers, preferences,
and system-level priors for consumption by step 3 endpoint LLMs.

### Key Decisions
- **`decomposition_analysis` replaces `query_rewrite` and `dealbreaker_summary`.**
  Step 1's `intent_rewrite` already captures full concrete intent (no need for
  a second rewrite) and `display_phrase` already serves the UI (no need for a
  second display label). The decomposition analysis is a brief evidence inventory
  (two to three sentences) that inventories the distinct requirements/qualities
  in the query and classifies each as a hard requirement or soft quality. This
  directly scaffolds the model's hardest judgment call — the dealbreaker/preference
  boundary — by forcing explicit classification before structured item emission.
- **Per-item `routing_rationale` field** on both Dealbreaker and Preference.
  Brief concept-type classification label (e.g., "named person (actor)", "genre
  classification") placed before the `route` enum. Misrouting is the #1 prompt
  design concern identified in the proposal; this field forces the model to
  categorize what kind of thing the concept is before selecting the endpoint enum.
- **`prior_assessment` field** scaffolds quality/notability prior enums. One
  sentence citing quality/notability signals and checking whether a dominant
  primary preference should suppress priors. Prevents defaulting to `standard`
  without considering the suppression inference.
- **Field ordering: analysis → dealbreakers → preferences → assessment → priors.**
  Follows cognitive-scaffolding convention. Dealbreakers before preferences
  because thematic centrality in preferences depends on knowing which keyword
  dealbreakers were emitted. Priors last because `suppressed` is a second-order
  inference depending on the decomposition.
- **Per-dealbreaker ordering: description → direction → routing_rationale → route.**
  Extractive fields first (what and which way), then evidence-inventory for
  routing (concept-type label), then constrained enum last.
- **Three new enums** added to schemas/enums.py: `EndpointRoute` (7 values),
  `DealbreakDirection` (inclusion/exclusion), `SystemPrior` (4 values shared
  by quality and notability).
- **Step 3 input updated** — endpoint LLMs receive step 1's `intent_rewrite`
  as query context (not a step 2 rewrite, since none exists).

### Planning Context
Schema design informed by prompt authoring conventions (evidence-inventory,
cognitive scaffolding, brief pre-generation fields, abstention-first) and the
concrete Step 1 precedent in schemas/flow_routing.py. The `query_rewrite` and
`dealbreaker_summary` fields were initially proposed, then replaced after
discussion identified that step 1 already covers both purposes and a
decomposition analysis field directly scaffolds the harder classification task.

## Search V2 stage 2 LLM call scaffold
Files: search_v2/stage_2.py (new)
Why: Wire up the query understanding LLM call so we can test different provider/model combinations.
Approach: `understand_query()` accepts provider, model, and kwargs with no defaults, validates the query, and delegates to `generate_llm_response_async` with `QueryUnderstandingResponse` as the structured output schema. System prompt is a TODO placeholder — will be implemented separately. The function parameter is named `query` (not `intent_rewrite`) so the interface doesn't leak step 1 internals; the docstring frames it as the user's query with no mention of upstream preprocessing.

## Step 2 query understanding system prompt
Files: search_v2/stage_2.py

### Intent
Implement the system prompt that guides the step 2 query understanding LLM to decompose standard-flow queries into dealbreakers, preferences, and system-level priors.

### Key Decisions
- **5-section modular prompt** (`_TASK + _DECOMPOSITION + _ENDPOINTS + _PRIORS + _OUTPUT`) following the stage_1.py pattern. Sections ordered for comprehension: the model needs to understand the conceptual framework before seeing endpoints, and needs endpoints before interpreting output field instructions.
- **Decomposition guidance** covers: dealbreaker vs preference distinction (what kind of movie vs what it should feel like), direction semantics (inclusion/exclusion), preference reframing (all positive), semantic preference grouping (consolidate into single rich description, exception for disjunctive intent), dual dealbreaker+preference pattern (thematic centrality for keyword dealbreakers), and reference movie trait extraction.
- **7 endpoint definitions** each follow: description → route-here-when → do-not-route-here → tricky boundaries → description format examples. Each endpoint was individually reviewed and approved during planning. Keyword endpoint includes the full enumerated vocabulary across 11 classification dimensions so the LLM can make informed routing decisions about what it covers vs what must go to semantic.
- **No implementation details leaked** — no table names, column types, index types, or ID systems. Endpoints described in terms of what they can evaluate, not how they execute.
- **Prior guidance** explains the 4-value enum with suppressed as a second-order inference, and the superlative interaction pattern.
- **Output field guidance** follows cognitive-scaffolding ordering matching the schema: decomposition_analysis (evidence inventory) → dealbreakers → preferences → prior_assessment → quality_prior → notability_prior. Per-item fields follow the cognitive chain: description → direction → routing_rationale → route.

### Planning Context
Prompt authoring conventions from metadata generation (evidence-inventory, cognitive scaffolding, brief pre-generation, abstention-first, evaluation guidance over outcome shortcuts, principle-based constraints). All endpoint definitions drawn from finalized_search_proposal.md and full_search_capabilities.md. ~8,550 tokens total.

## Step 2 prompt refinement: endpoint boundaries and concept splitting
Files: search_v2/stage_2.py
Why: Tighten the stage 2 routing prompt so small models split distinct concepts reliably, use deterministic endpoints only when they genuinely fit, and avoid vague keyword matches.
Approach: Rewrote grouping guidance into explicit merge-vs-separate rules, adding a direct rule that shared route is never a reason to merge (`Brad Pitt and Tom Hanks`, `award-winning comedy`). Added a reference-movie guardrail limiting expansion to broad high-confidence traits. Clarified metadata vs keyword culture/audio boundaries, including explicit `Bollywood` → Hindi-culture keyword mapping and `Hindi audio` → metadata. Strengthened franchise-vs-keyword rules (`sequel`/`prequel` always franchise, generic remakes/source-material in keyword, franchise-specific remakes in franchise). Clarified entity-vs-keyword character boundaries (`doctor` as character/entity lookup vs `female lead` as character-type keyword). Tightened keyword instructions so the model should only use keyword when it can point to a specific listed vocabulary fit, and aligned `routing_rationale` guidance with that requirement. Restated semantic as the fallback only when no deterministic endpoint genuinely and cleanly fits, and expanded trending examples to additional "right now" language.
Design context: Follows the repo’s evidence-inventory prompt conventions while avoiding consistency-coupling language. Also corrects a prompt-level mismatch by removing the old source-material `Sequel` label from the keyword vocabulary so the prompt matches the intended deterministic coverage.
Testing notes: Did not run tests per repo instruction boundary for this task. Changes are prompt-only; verification should use representative step 2 query cases and inspect structured outputs for route correctness and item splitting.

## Step 2 prompt refinement: enum-backed metadata and source-material coverage
Files: search_v2/stage_2.py
Why: A follow-up prompt audit found that some small deterministic vocabularies were still underspecified or mismatched with the real enums, especially source material and metadata access modes.
Approach: Replaced the source-material list with the exact enum-backed value set (`Novel Adaptation`, `Short Story Adaptation`, `Stage Adaptation`, `True Story`, `Biography`, `Comic Adaptation`, `Folklore Adaptation`, `Video Game Adaptation`, `Remake`, `TV Adaptation`). Expanded metadata guidance to include `Unrated` in maturity coverage, explicitly listed the three access-type values (`subscription`, `buy`, `rent`), and enumerated the tracked streaming services. Also reframed "free to stream" as provider-level free-service availability (for example Tubi / Pluto / Plex / Roku Channel) rather than a separate access-type enum value that does not exist.
Design context: Keeps the stage 2 prompt aligned with the actual enum-backed deterministic search surface instead of relying on lossy shorthand that can drift from implementation.
Testing notes: Prompt-only follow-up; did not run tests.

## Step 3 input specification and continuous scoring model
Files: search_improvement_planning/finalized_search_proposal.md

### Intent
Define step 3 endpoint inputs and replace strict tier-based reranking with a
continuous scoring model where dealbreakers produce [0,1] scores and preferences
are capped below one full dealbreaker.

### Key Decisions
- **Per-item endpoint calls** — each dealbreaker/preference gets its own
  independent LLM call (not one call per endpoint type). Inputs: `intent_rewrite`
  + one item's `description`, `routing_rationale`, and `direction` (dealbreakers
  only). Excluded: `route`, `is_primary_preference`, priors, other items.
- **Gradient logic is deterministic code** — step 3 LLMs produce literal
  translations ("1980-1989"). Execution code wraps with gradient decay functions.
  Same for semantic: LLM picks vector spaces and queries, code applies
  elbow-calibrated scoring.
- **Continuous scoring replaces strict tiers** — `final_score = dealbreaker_sum +
  preference_contribution - exclusion_penalties`. Each dealbreaker scored [0,1];
  preferences capped at P_CAP=0.9. Preferences can overcome partial matches but
  never a full dealbreaker miss.
- **Semantic dealbreakers score, don't demote** — contribute to dealbreaker_sum
  via elbow-calibrated cosine similarity (1.0 above elbow, decay below, 0.0
  below floor). Still cannot generate candidates.
- **Preference weighting formula** — weighted average scaled by P_CAP. Weights:
  regular=1.0, primary=3.0, quality/notability priors weighted by enum value
  (enhanced=1.5, standard=0.75, inverted=1.5 with flipped score, suppressed=0).
- **Actor billing-position gradient** — default 1.0 for top 15% billing, floor
  of 0.8 for cameos. Steepens when user specifies prominence.
- **Elbow fallback** — percentage-of-max threshold when elbow detection fails.

### Planning Context
Tier system was challenged: metadata gradients and semantic similarity don't
naturally produce binary pass/fail, forcing arbitrary cliff edges. The user
proposed continuous scoring where `P_CAP < 1.0` preserves the guarantee that
full dealbreaker matches dominate while allowing preferences to separate
near-matches. Existing `db/metadata_scoring.py` gradient patterns inform the
per-attribute decay functions.

## Scoring refinements: inverted priors and semantic exclusion penalties
Files: search_improvement_planning/finalized_search_proposal.md
Why: Two corrections to the scoring model from review.
Approach: (1) Inverted quality/notability priors are handled at the endpoint
query/scoring level — the endpoint queries for poor reception or obscurity
directly, producing a high score for niche/bad movies. No `1.0 - score`
inversion in the formula. (2) Semantic exclusion penalties use a
match-then-subtract model: score the excluded concept the same way as a
semantic inclusion dealbreaker (elbow-calibrated [0,1]), then subtract
`E_MULT × match_score` from the final score. E_MULT starts at 2.0 (tunable).
A 0.5 match costs a full dealbreaker's worth of score; a 0.9 match is
devastating; a 0.0 match costs nothing.

## Actor prominence scoring: zone-based adaptive thresholds
Files: search_improvement_planning/finalized_search_proposal.md
Why: Resolved actor prominence scoring — the last open entity endpoint question.
Approach: Zone-based system using `max(floor, round(scale * sqrt(cast_size)))`
to define LEAD/SUPPORTING/MINOR zones that adapt to cast size. Four scoring
modes (DEFAULT, LEAD, SUPPORTING, MINOR) each assign zone-based scores with
within-zone gradients. DEFAULT gives leads 1.0 with floor 0.5 for lowest
minor. LEAD mode is harsh (0.2 floor) for "starring" queries. SUPPORTING
peaks at the supporting zone. MINOR inverts — deeper billing scores higher.
Updated "Decisions Deferred" to remove old actor billing reference, added
character/title scoring details as deferred items.

## Entity endpoint step 3/4 design decisions
Files: search_improvement_planning/finalized_search_proposal.md
Why: Resolved entity endpoint design questions from planning discussion.
Approach:
- **Direction-agnostic framing (new design principle #10):** Step 3 LLMs
  always search for positive presence of an attribute. `direction` field
  moved from step 3 inputs to excluded inputs — consumed only by step 4
  code. Step 2 `description` field now always uses positive-presence form
  ("involves clowns" not "does not involve clowns"). Added as a dedicated
  subsection in Step 3 and as design principle #10. This is architecturally
  critical — prevents double-negation confusion and keeps each LLM's task
  clean.
- **No pool size limit** for entity candidates (~7K worst case is fine).
- **No-match = valid empty result** — no fallback to closest fuzzy match.
- **No re-routing responsibility** — step 3 trusts upstream routing.
- **Cross-posting table search:** Single-table when role is confident.
  Multi-table with primary anchor (nullable): primary gets full credit,
  non-primary gets 0.5 × match_score, max across tables (no summing).
  Without primary, all tables get full credit, still max-based.
- **Non-binary scoring** for character lookups (fuzzy similarity) and title
  pattern lookups (match coverage) — details to be finalized before
  implementation.
- **Actor prominence scoring** — modes and formulas still under discussion.
Testing notes: Doc-only changes, no code modified.

## Entity endpoint step 3 output schema and per-sub-type specifications
Files: schemas/entity_translation.py (new), schemas/enums.py, search_improvement_planning/finalized_search_proposal.md

### Intent
Define the structured output schema for the step 3 entity endpoint LLM and
document the per-sub-type search mechanics, scoring, and execution logic in
the finalized proposal.

### Key Decisions
- **Flat model with nullable type-specific fields.** `EntityQuerySpec` has
  `entity_name` (always required — the corrected/normalized search key) and
  `entity_type` (enum discriminator), plus nullable fields per type. Matches
  the metadata endpoint pattern of one flat object rather than discriminated
  unions.
- **4 new enums** added to `schemas/enums.py`: `EntityType` (person/character/
  studio/title_pattern), `PersonCategory` (actor/director/writer/producer/
  composer/broad_person), `ActorProminenceMode` (default/lead/supporting/
  minor), `TitlePatternMatchType` (contains/starts_with).
- **`broad_person` replaces multi-table array.** Instead of `search_categories`
  as a list, the LLM outputs a single `person_category` enum. `broad_person`
  means search all 5 role tables; any specific value means single-table search.
  `primary_category` (renamed from primary_anchor) controls cross-posting
  score consolidation for broad_person searches.
- **Character search is exact-only.** No fuzzy/token matching. The LLM
  generates the standard credited form(s) of the character name (`entity_name`
  + `character_alternative_names`). Each variation is exact-matched. Score is
  binary 1.0. Generic character type queries ("movies with a cop") are routed
  to keyword/semantic instead — character posting tables contain credited names,
  not role descriptions.
- **Studio search is exact-only.** Same normalization rules as person names.
  If too brittle in practice, LIKE substring or alias table deferred to
  implementation.
- **Title pattern search uses LIKE, no fuzziness.** `contains` → `LIKE
  '%pattern%'`, `starts_with` → `LIKE 'pattern%'`. Binary 1.0 scoring.
- **All non-actor sub-types use binary scoring.** Only actors have
  prominence-based gradients. Characters, studios, directors, writers,
  producers, composers, and title patterns are all 1.0 or 0.0.
- **Name normalization follows V1 lexical prompt rules.** Fix typos, complete
  unambiguous partial names, capitalize — but never add suffixes or infer
  names not typed.

### Planning Context
Per-sub-type specifications drawn from existing lexical search code (exact
matching for people/studios via `lex.lexical_dictionary`, LIKE substring for
characters/titles via trigram GIN), IMDB character data format (credited
character names, not role descriptions), and V1 `lexical_prompts.py`
normalization rules. Character fuzzy matching and title pattern coverage
scoring (previously deferred) resolved as unnecessary — binary scoring with
good LLM-generated search terms is sufficient.

## Entity endpoint review fixes: validator, list constraint, prompt alignment
Files: schemas/entity_translation.py, search_v2/stage_2.py
Why: Code review found three issues — no enforcement of primary_category != broad_person, unvalidated character_alternative_names list items, and step 2 prompt still routing generic character types to entity.
Approach: (1) Added model_validator that coerces primary_category=broad_person to null. (2) Changed character_alternative_names from `list[str]` to `conlist(constr(..., min_length=1), min_length=0)` to reject empty strings. (3) Updated entity endpoint definition in stage_2.py: character types narrowed to specific named characters only, generic character types ("doctor", "police officer") explicitly listed in Do NOT route section, description examples updated, keyword boundary section reversed doctor routing from entity to semantic with explanation of why character posting tables can't serve generic role lookups.

## Reorganized keyword classification dimensions for step 2 routing
Files: search_v2/stage_2.py, search_improvement_planning/finalized_search_proposal.md

### Intent
Fix misrepresentations and improve conceptual coherence of keyword
categories that the step 2 LLM uses for routing decisions.

### Key Decisions
- **5 misrepresentations fixed:** Adult Animation removed from Teen &
  Coming-of-Age; Slice of Life moved from Other to Anime Genres; Swashbuckler
  moved from War/Western/Historical to Adventure; News and Short moved from
  Other to Format & Presentation.
- **"Anime & East Asian Traditions" renamed to "Anime Genres"** and narrowed
  to only anime-specific classifications. Samurai and Wuxia moved to Action &
  Combat (live-action martial arts traditions). Kaiju and Mecha moved to
  Fantasy & Science Fiction (speculative fiction genres spanning anime and
  live-action).
- **"Other" catch-all dissolved:** History and Tragedy moved to Drama; Animation
  and Family moved to new Audience & Medium dimension; News and Short moved to
  Format & Presentation.
- **"War, Western & Historical" split** into War (2) and Western (5) as
  separate genre families.
- **2 new classification dimensions added:** Audience & Medium (Adult Animation,
  Animation, Family — cross-cut genres by who/what medium) and Format &
  Presentation (Mockumentary, Sketch Comedy, Stand-Up, News, Short, plus
  existing reality/talk/game show keywords — describe how content is delivered).
- **Dimension count: 11 → 13.** Genre sub-categories: 18 → 17 (fewer
  catch-alls, cleaner splits).
- Comedy trimmed: Mockumentary, Sketch Comedy, Stand-Up moved to Format &
  Presentation (describe presentation format, not comedy narrative type).
- Documentary renamed to Documentary & Nonfiction. Fantasy & Sci-Fi renamed to
  Fantasy & Science Fiction.

### Testing Notes
Prompt-only changes. Verify with representative step 2 queries that routing
still correctly assigns keywords to the keyword endpoint and doesn't misroute
format/audience keywords to semantic.

## Metadata endpoint planning decisions
Files: search_improvement_planning/finalized_search_proposal.md, search_improvement_planning/open_questions.md
Why: Resolved open questions and clarified important considerations for the metadata endpoint before step 3/4 implementation.
Approach: Resolved 4 open questions: (1) LLM translator does NOT soften constraints — faithful literal translation only, code applies softening using existing `db/metadata_scoring.py` patterns. (2) "Best" maps to `quality_prior: enhanced`, not a separate mechanism. (3) Always include buffer in candidate generation, trust the pipeline to narrow down. (4) Endpoint failure returns empty candidate set (retry once for transient issues). Added country-of-origin position-based gradient scoring (position 1 = 1.0, position 2 = ~0.7-0.8, position 3+ = rapid decay) to the metadata endpoint spec and deferred decisions list, with a note to verify TMDB/IMDB array ordering empirically. Added pipeline failure handling note to the endpoint spec.

## Metadata endpoint step 3 output schema and per-attribute specifications
Files: schemas/metadata_translation.py (new), schemas/enums.py, search_improvement_planning/finalized_search_proposal.md, search_improvement_planning/open_questions.md, search_improvement_planning/v2_data_architecture.md

### Intent
Finalize all 10 metadata attributes with concrete LLM output parameters,
scoring functions, candidate generation strategies, and edge case handling.
Implement the step 3 output schema and update planning docs.

### Key Decisions
- **Inclusion-only framing across all attributes.** No exclusion lists anywhere
  in the step 3 output. Exclusion dealbreakers from step 2 are handled by step 4
  scoring code. This aligns with the direction-agnostic step 3 principle.
- **New enums:** `PopularityMode` (POPULAR/NICHE) and `ReceptionMode`
  (WELL_RECEIVED/POORLY_RECEIVED) added to `schemas/enums.py`. `BudgetSize`
  moved from `implementation/classes/enums.py` to `schemas/enums.py` with
  NO_PREFERENCE removed (null = no preference).
- **Popularity supports inverse scoring.** NICHE mode scores
  `1.0 - popularity_score`, enabling "hidden gems" as NICHE + WELL_RECEIVED.
  Replaces the old boolean `prefers_popular_movies`.
- **Reception simplified to directional enum.** WELL_RECEIVED/POORLY_RECEIVED
  with null for no preference. Replaces the ternary `ReceptionType` with its
  NO_PREFERENCE value.
- **Audio language is explicit-mention-only.** Never inferred. "French films" →
  country of origin. "Foreign films" → country of origin (broad non-US set).
  Only "movies with French audio" or "dubbed in Spanish" triggers this attribute.
- **Country of origin supports multi-country lists.** LLM uses parametric
  knowledge to expand region terms ("European movies"). Score = max across all
  requested countries (no summing). IMDB array ordering confirmed as order of
  relevance — position gradient constants still need tuning.
- **UNRATED exclusion rule:** Any maturity query targeting a rated value
  (anything other than EXACT UNRATED) excludes UNRATED movies from both scoring
  and candidate generation.
- **Null data handling:** Movies with null data score 0.0 for that attribute
  (no boost) but are NOT excluded from the candidate set. For exclusion
  dealbreakers, null = did not match exclusion, so no penalty.
- **Vague terms left to LLM judgment.** No special defaults for "epic length",
  "long movie", etc. The LLM infers reasonable concrete values. Only "recent"
  gets a guideline (≈ last 3 years) since the LLM needs today's date injected.
- **Budget and box office remain binary.** Match = 1.0, no match = 0.0. No
  gradient — already bucketed classifications.

### Planning Context
Per-attribute specifications drawn from existing `db/metadata_scoring.py`
V1 patterns, `v2_data_architecture.md` data inventory, and discussion
resolving 10 attribute-level questions plus 3 cross-cutting questions.
Constraint strictness table in `open_questions.md` updated to reflect
country-of-origin position-graded scoring (was previously marked "Hard").

## Canonical keyword taxonomy rewrite across prompt and planning docs
Files: search_v2/stage_2.py, search_improvement_planning/finalized_search_proposal.md, search_improvement_planning/open_questions.md, search_improvement_planning/new_system_brainstorm.md, search_improvement_planning/full_search_capabilities.md, search_improvement_planning/v2_data_architecture.md
Why: The repo had drift between the new `OverallKeyword` definitions, the step 2 routing prompt, and the search-planning docs. The old presentation still described the keyword endpoint as 13 overlapping dimensions, still referenced obsolete source-material concepts, and still mixed cultural identity, country, runtime, and short-form classification in inconsistent ways.
Approach: Replaced the old keyword-endpoint framing with a canonical concept-family taxonomy backed by multiple deterministic stores (`genre_ids`, `keyword_ids`, `source_material_type_ids`, `concept_tag_ids`). The prompt and finalized proposal now use the same concept-first taxonomy, explicit overlap rule, and aligned boundary examples for `Short`, `Biography`, `French` vs audio, `Bollywood` vs Hindi audio, `Remakes`, `Scary movies`, `Feel-good`, and `Coming-of-Age`. Supporting planning docs were updated to remove stale “trait descriptions only” wording, replace the obsolete source-material list with the actual enum values, stop routing broad genres as a separate conceptual `genre_ids` surface, and rename `overall_keywords` as a broader keyword taxonomy rather than just a genre/sub-genre taxonomy.
Design context: Intent and category placements follow the new `OverallKeyword` definitions in `implementation/classes/overall_keywords.py`, the shared enums in `schemas/enums.py`, and the user-approved concept-family taxonomy. `Biography` is canonical under Source Material / Adaptation / Real-World Basis; `Short` is canonical as a short-form classification while pure runtime remains metadata; `News` is canonical under Nonfiction / Documentary / Real-World Media; `Adult Animation` is canonical under Animation / Anime Form / Technique.
Testing notes: Verified by grep that the stale phrases (`13 classification dimensions`, obsolete source-material values, old trait-description wording, `short movies` as runtime shorthand, and `genre/sub-genre taxonomy`) were removed from the active prompt/docs set. Also ran a local normalization check against `OverallKeyword`, `Genre`, `SourceMaterialType`, and stored concept tags: the new family taxonomy covers the full keyword-endpoint concept surface exactly once with 259 concepts, 0 missing, and 0 duplicate assignments.

## Metadata endpoint: single-column targeting via target_attribute
Files: schemas/enums.py, schemas/metadata_translation.py, search_improvement_planning/finalized_search_proposal.md
Why: Step 2 already decomposes multi-attribute concepts into separate items (e.g., "hidden gems" → niche popularity + well-received reception). The metadata endpoint should honor that by querying exactly one column per item, not combining scores across columns within a single dealbreaker/preference.
Approach: Added `MetadataAttribute` enum (10 values matching the 10 attribute fields) to `schemas/enums.py`. Added `target_attribute` as the first field in `MetadataTranslationOutput` — the LLM identifies which column best represents the step 2 description before populating attribute fields. Execution code queries ONLY that column. Updated the finalized proposal's Endpoint 2 output schema section to document the single-column targeting design, the rationale (step 2 handles decomposition, step 3 handles single-column translation), and the guarantee (one metadata item = one column query = one [0,1] score).

## Entity endpoint reasoning fields + stage 3 entity LLM module
Files: schemas/entity_translation.py, search_v2/stage_3/__init__.py, search_v2/stage_3/entity_query_generation.py
Why: Endpoint 1 needed two things before it can be exercised — scoped reasoning fields on the output schema to scaffold the high-stakes decisions (exact-match name canonicalization and the four-mode actor prominence pick), and a stage 3 module that actually drives the LLM call with a purpose-built system prompt.
Approach: Added `entity_analysis` as the first field (evidence inventory scaffolding entity_name, entity_type, and person_category) and `prominence_evidence` immediately before `actor_prominence_mode` (abstention-first — emits "not applicable" when entity is not a person or person_category is not actor/broad_person, otherwise quotes input language or says "no prominence signal"). Built `search_v2/stage_3/entity_query_generation.py` with a modular prompt (task → positive-presence invariant → entity types → person role selection → actor prominence modes → name canonicalization → output field guidance) and an async `generate_entity_query(intent_rewrite, description, routing_rationale, provider, model, **kwargs)` entry point that mirrors stage_2's `understand_query` signature and routes through `generate_llm_response_async`. After the prompt was in place, stripped Field descriptions from both reasoning fields so all LLM-facing guidance lives in the system prompt, matching the convention and the pattern used on `EntityQuerySpec`'s other fields.
Design context: Scoped-reasoning-before-decision pattern from FranchiseOutput (schemas/metadata.py) and the "evidence inventory, not rationalization", "brief pre-generation fields", "abstention-first", and "exact-match convergence for LLM-generated strings" conventions in docs/conventions.md. Endpoint spec in search_improvement_planning/finalized_search_proposal.md §Step 3 → Endpoint 1 and posting-table surface in search_improvement_planning/full_search_capabilities.md §3.
Testing notes: Smoke-tested module imports. Real coverage needs a stage 3 eval harness driving representative entity items (actor with/without prominence language, broad_person persons, characters with alternative credited forms, studios, title patterns, typo correction, partial-name expansion) through the LLM against EntityQuerySpec.

## Entity endpoint execution layer + validator default for actor prominence
Files: schemas/entity_translation.py, db/postgres.py, search_v2/stage_3/entity_query_execution.py
Why: The step 3 entity LLM already emits EntityQuerySpec; step 3.5 needs the deterministic companion that turns that spec into an EndpointResult by running the lexical-schema queries and applying the binary / prominence scoring defined in the proposal. This change lands that execution layer, plus the minor schema default (actor_prominence_mode auto-DEFAULT when the actor table participates) that was agreed during design.

### Key Decisions
- **Single entry point, restrict-driven shape.** `execute_entity_query(spec, *, restrict_to_movie_ids=None)` handles both dealbreaker (no restrict → return naturally matched) and preference (restrict provided → return one entry per ID with 0.0 for non-matches) paths, per the proposal's Endpoint Return Shape contract. Keeps orchestrator code simple and avoids duplicating scoring in two entry points.
- **Title patterns query movie_card.title, not title_token_strings.** The proposal text said title tokens but the user pointed out tokens-only cannot support multi-word patterns or full-title prefix semantics. Execution now does ILIKE against movie_card.title after normalize_string → escape_like → %wrap. Diacritic-insensitive matching is a known limitation (title column is stored in display form; no unaccent index).
- **Character matching is exact, not substring.** Uses new fetch_character_strings_exact helper. Honors the proposal's "exact-matched against lex.character_strings" text; character_alternative_names is the designed escape hatch for credited-form variation.
- **Actor prominence lives in pure functions.** `_zone_cutoffs`, `_zone_relative_position`, and per-mode scorers are all side-effect-free, enabling direct unit-test coverage without DB. Cutoffs verified against the proposal's reference table (cast 5→20→200 produces the exact lead_cutoff/supp_cutoff values).
- **broad_person cross-posting fans out in parallel.** 5 role-table fetches (1 actor-with-billing + 4 binary role queries) via asyncio.gather, then max-merged with BROAD_PERSON_NON_PRIMARY_WEIGHT=0.5 applied to non-primary tables. No summing — max preserves the 1.0 ceiling per the spec.
- **Validator default for actor_prominence_mode.** `_normalize_person_fields` renamed from `coerce_broad_person_primary` (existing broad_person → null primary coercion preserved) and now also auto-sets actor_prominence_mode to DEFAULT whenever entity_type=person and person_category ∈ {actor, broad_person} and the LLM left the field null. Lets execution code drop the null branch and assert non-null with confidence.
- **Restrict pushed to DB where cardinality matters.** Actor billing rows and title LIKE matches both accept restrict_movie_ids server-side (narrows the scan for preference queries against large match sets like "the" titles). Binary role/studio/character fetches post-filter in Python — match sets there are small enough that extending every helper for server-side restrict is not worth the API surface.
- **Missing billing data → skip row, don't crash.** The actor SQL gates on billing_position/cast_size IS NOT NULL and cast_size > 0 so any future schema drift degrades to missing-score rather than division-by-zero. Per decision #6 during design.

### Ambiguities Resolved in Design
1. Title pattern target column → movie_card.title (not title_token_strings).
2. Normalization order → normalize_string → escape_like → %wrap, always.
3. broad_person with null primary_category → all tables full credit, max.
4. Null actor_prominence_mode → schema validator coerces to DEFAULT.
5. Null billing_position/cast_size → skip row.
6. Execution shape → single entry point with restrict_to_movie_ids parameter.
7. Multi-word title patterns → handled by single ILIKE on full title.

### Testing Notes
Unit coverage should exercise: zone cutoffs at small/medium/large casts, each of the 4 prominence modes at zone boundaries and bottom positions, broad_person merge with and without primary_category, character with no alternatives vs multiple, title pattern contains vs starts_with with wildcard-containing inputs, restrict path returning exact supplied IDs with 0.0 non-match fill, empty entity_name and empty dictionary lookups both returning empty EndpointResult. Integration coverage needs a Postgres fixture with seeded lex.* tables since all real paths hit the DB. Smoke-tested pure-function paths at build time — zone cutoffs match the proposal's reference table exactly, and validator coercions fire on actor/broad_person persons.

## Award endpoint prompt: canonical surface-form guidance
Files: search_v2/stage_3/award_query_generation.py
Why: award_names and categories are matched as exact, un-normalized strings against stored `movie_awards` rows. The existing prompt only showed a few category exemplars and told the LLM to "use the most common form," leaving it to guess between diverging ceremony-specific surface forms (Oscars `Best Actor in a Leading Role` vs. Globes `Best Performance by an Actor in a Motion Picture - Drama` vs. generic `Best Actor`). Data inspection of the live `movie_awards` table also showed only `Palme d'Or` was listed for Cannes while `Grand Jury Prize`, `Un Certain Regard Award`, `Jury Prize`, `FIPRESCI Prize` are common and distinct strings the LLM had no anchor for.
Approach: Added a new `_SURFACE_FORMS` module-level string inserted between `_FILTER_AXES` and `_RAZZIE_HANDLING` in the SYSTEM_PROMPT concatenation. The section instructs the LLM to emit the official IMDB surface form for the specific ceremony in play (ceremony-specific, case/punctuation/word-order sensitive), anchored by a compact per-ceremony table and explicitly gated by three rules: (1) use parametric IMDB knowledge for anything not in the table — do not fall back to generic labels; (2) do not restrict output to table entries; (3) do not pattern-match a user's phrase onto a similar-looking row when the user clearly named a different award (e.g., "Cannes Jury Prize" → `Jury Prize`, not `Palme d'Or`). Reinforced at `_OUTPUT` for the `award_names` and `categories` field guidance with one sentence each. Also updated the module header's Structure comment to list the new section.
Design context: Table is principle-based (3-6 exemplars per ceremony chosen to teach surface-form conventions) rather than exhaustive, keeping with the prompt's existing "principle-based constraints, not failure catalogs" authoring convention called out at the top of the file. Plan file at ~/.claude/plans/sounds-good-let-s-just-happy-hamster.md.
Testing notes: Behavioral — spot-check `generate_award_query` on "Oscar Best Actor winners" (expect `Best Actor in a Leading Role`), "Golden Globe Best Director films" (expect `Best Director - Motion Picture`), "Cannes Jury Prize winners" (expect `Jury Prize`, NOT `Palme d'Or`), "Razzie Worst Picture" (expect `Worst Picture`), and "BAFTA-winning films" with no category (expect ceremony-only, null category/award_names). Regression check that the five-pattern scoring classification and filter-axis inventory reasoning remain unchanged — the new section is additive. Import-time sanity check confirmed SYSTEM_PROMPT builds and section ordering is task → direction → scoring → filter axes → surface forms → razzie → output.

## Award endpoint execution layer
Files: search_v2/stage_3/award_query_execution.py, db/postgres.py
Why: AwardQuerySpec is produced by the stage-3 award LLM; step 3.5 needs the deterministic companion that turns that spec into an EndpointResult. This change lands that execution layer, matching the dual-mode (dealbreaker / preference) restrict_to_movie_ids contract used by the franchise, entity, and trending executors.
Approach: New execute_award_query in search_v2/stage_3/award_query_execution.py dispatches between two data-source paths. Fast path hits movie_card.award_ceremony_win_ids via the GIN `&&` operator (non-Razzie ceremony id set) — triggers only when the spec reduces to a "has any non-Razzie win" presence check. Standard path runs COUNT(*) GROUP BY movie_id on public.movie_awards with whichever axes the spec populated, then applies the FLOOR or THRESHOLD scoring formula. DB helpers (fetch_award_fast_path_movie_ids, fetch_award_row_counts) added to db/postgres.py under a new AWARD ENDPOINT HELPERS section, matching the franchise-helpers pattern.

### Key Decisions
- **Fast path excludes outcome=null.** The proposal permits fast path for outcome ∈ {WINNER, null}, but award_ceremony_win_ids stores wins only, so firing the fast path on null would silently drop nomination-only movies (null semantic is "wins OR nominations"). Option 2 chosen: fast path fires only when outcome=WINNER; null is routed through movie_awards. Perf cost is negligible because the reachable fast-path surface is already small (LLM five-pattern table rarely emits FLOOR/1 + outcome=null + no filters). Divergence from the literal proposal text is called out in the module header.
- **No normalization on award_name / category.** Per user instruction and ingestion convention, these columns preserve raw IMDB surface form. _dedupe_nonempty strips wrapping whitespace and drops empties/duplicates but never calls normalize_string; fetch_award_row_counts uses exact `= ANY(...)` equality.
- **Razzie exclusion policy split by ceremony presence.** When spec.ceremonies is null/empty, the DB helper adds `ceremony_id <> 10` as a default guard. When the spec names ceremonies explicitly, no default exclusion is added — whatever the caller put in the list is respected verbatim (so `[ACADEMY_AWARDS, RAZZIE]` includes Razzie, `[ACADEMY_AWARDS]` excludes it naturally).
- **GIN-indexable overlap for the fast path.** `award_ceremony_win_ids && ARRAY[1..9,11,12]::smallint[]` is used instead of `cardinality(array_remove(...)) > 0` because only `&&` / `@>` / `<@` are GIN-indexable on an int-array column — the remove-then-cardinality form would force a seq scan.
- **FLOOR 0.0 scores dropped before build_endpoint_result.** Movies that fall below the scoring_mark on FLOOR yield 0.0; these are filtered out so the dealbreaker path omits them cleanly and the preference path falls back to the 0.0 default that build_endpoint_result already provides. Avoids emitting duplicate zero entries.
- **Retry-once contract matches franchise/entity executors.** Transient DB errors retry once; second failure returns an empty EndpointResult so the orchestrator can continue rather than hard-fail. Consistent with the soft-failure policy used across stage 3.

### Testing Notes
- Pure-function coverage was smoke-tested at build time: _dedupe_nonempty (None/empty/whitespace/dupe), _resolve_ceremony_ids (null/empty/single/with-Razzie), _resolve_outcome_id, _score_from_count (FLOOR and THRESHOLD at boundaries including saturation), and _qualifies_for_fast_path (confirmed outcome=null is NOT eligible — this is the option-2 behavior).
- Real coverage needs a stage-3 eval harness driving representative award items through the LLM + execution against seeded movie_awards data: generic "award-winning" (THRESHOLD/3 standard path), "Oscar Best Picture winners" (FLOOR/1 standard path), "won 5 Oscars" (FLOOR/5 standard path), "Razzie winners" (explicit-include path), "most decorated" (THRESHOLD/15), and a nomination-only sanity check to verify outcome=null routes to movie_awards rather than fast path.
- Integration coverage should seed movie_awards with a movie holding only nominations (outcome_id=2) and confirm: (a) outcome=null FLOOR/1 returns the movie, (b) same spec with outcome=WINNER does not, (c) empty award_ceremony_win_ids + non-empty movie_awards is handled correctly.

## Keyword endpoint step 3 output schema
Files: schemas/keyword_translation.py
Why: Step 3 keyword endpoint needs a structured output model for the 259-way UnifiedClassification selection. Entity, metadata, franchise, and award endpoints all have their translation schemas landed; keyword was the remaining gap called out in the finalized_search_proposal.md implementation checklist.
Approach: New `KeywordQuerySpec` Pydantic model with two scoped reasoning fields preceding the single enum selection. `concept_analysis` (first) is a telegraphic evidence-inventory that quotes signal phrases from description/intent_rewrite and pairs each with a concept-type angle (genre-like, cultural tradition, narrative device, etc.) — the angles map 1-1 to the 21 canonical concept families so no separate family_shortlist field is needed (would be near-mechanical copy-forward, per franchise endpoint precedent). `candidate_shortlist` (placed immediately before `classification`) is a comparative evaluation of 2-3 near-collision registry entries with the discriminating test cited for each — the anti-first-strong-match-wins mechanism that addresses the documented failure mode from personal_preferences.md. `classification` is a single UnifiedClassification member with no abstention (routing already committed).
Design context: Follows the cognitive-scaffolding and evidence-inventory conventions in docs/conventions.md. Reuses the UnifiedClassification StrEnum from schemas/unified_classification.py so the schema emits a finite JSON-schema enum with all 259 valid choices. Field placement follows "reasoning field immediately before the decision it scaffolds" — proximity matters for autoregressive attention on the final selection. No class-level docstrings or Field descriptions, consistent with the other step 3 translation schemas.
Testing notes: Unit test should parametrize construction with each UnifiedClassification member to confirm round-trip parses, and verify extra=forbid rejects unknown fields. Behavioral eval should cover near-collision disambiguation: FEEL_GOOD_ROMANCE vs FEEL_GOOD, TRUE_STORY vs BIOGRAPHY, HORROR vs PSYCHOLOGICAL_HORROR, COMING_OF_AGE vs TEEN_DRAMA. Prompt authoring is a separate task.

## Keyword endpoint step 3 query generation
Files: search_v2/stage_3/keyword_query_generation.py, schemas/keyword_translation.py
Why: Step 3 keyword endpoint needs the LLM-driving module that takes a step 2 keyword item and produces a KeywordQuerySpec. Matches the structure of the existing entity/metadata/award/franchise query_generation modules so the step 3 dispatcher can treat keyword uniformly.
Approach: New `generate_keyword_query` async function with the same (intent_rewrite, description, routing_rationale, provider, model, **kwargs) signature as the other stage 3 generators; returns Tuple[KeywordQuerySpec, int, int] per the unified LLM-function contract. System prompt is assembled from six modular sections: task, positive-presence invariant, classification families (21 canonical families summarized — the schema enum enumerates the 259 members), near-collision disambiguation (breadth-vs-specificity, explicit premise signal, cross-family proximity, mutually exclusive ending/tag pairs), scope discipline (one pick, no abstention, no invention), and output field guidance. Reasoning fields are taught as telegraphic evidence inventories with concrete format examples rather than prose rationalization. Per the "all guidance in system prompt" convention, verbose per-field comments were stripped from schemas/keyword_translation.py — the schema now carries only field order, developer notes, and structural constraints.

### Key Decisions
- **Teach families, not members.** The 259 registry members are already enumerated by the finite JSON-schema enum the Pydantic response_format emits; enumerating them in the prompt would duplicate the enum and blow up token count. The prompt teaches the 21 canonical families and the routing angle that maps into each, which is the actual narrowing step the model has to perform.
- **Disambiguation principles, not lookup tables.** Near-collision cases (HORROR vs sub-forms, TRUE_STORY vs BIOGRAPHY, FEEL_GOOD_ROMANCE vs FEEL_GOOD, COMING_OF_AGE vs TEEN_DRAMA) are handled by four comparison principles (breadth vs specificity, explicit premise signal, cross-family proximity, mutually exclusive pairs). Per personal_preferences.md "Principle-based prompt instructions, not reactive failure lists" and conventions.md "Evaluation guidance over outcome shortcuts" — teach the model how to evaluate rather than giving it a catalog of observed failures.
- **Explicit no-abstention framing.** Separate SCOPE AND ABSTENTION section directly addresses the failure mode where a small LLM might refuse or emit null on an imperfect fit. Instructs the model to fall back to the broader candidate rather than abstain, since routing has already committed the item to this endpoint and an empty output breaks the pipeline.
- **Routing rationale marked as context, not evidence.** The prompt explicitly tells the model to ignore routing_rationale when extracting signal phrases in concept_analysis, because anchoring on an already-interpreted label re-introduces routing bias. Signal phrases come from description and intent_rewrite only.
- **Shortlist must include the committed member.** Final instruction on classification names that the selection must appear in candidate_shortlist — makes the shortlist load-bearing rather than ornamental, which enforces the comparative-evaluation mechanism instead of letting the model emit a shortlist then pick something else.

### Testing Notes
- Unit test: parametrize KeywordQuerySpec construction across every UnifiedClassification member to confirm round-trip parses; verify extra=forbid rejects unknown fields and that both reasoning fields reject empty strings.
- Prompt behavior test: run generate_keyword_query against a curated evaluation set covering each of the four disambiguation principles — "scary movies" (breadth vs specificity → HORROR), "Bollywood movies" (cultural tradition → HINDI), "a movie that leaves you uplifted" (viewer-response → FEEL_GOOD vs FEEL_GOOD_ROMANCE), "biopic of Lincoln" (cross-family → BIOGRAPHY vs TRUE_STORY), "zombie movies" (explicit premise → ZOMBIE_HORROR), "coming-of-age story" vs "teen drama" (cross-family pair). Verify classification appears in candidate_shortlist.
- Format verification: confirm concept_analysis produces telegraphic phrase → angle pairs, not prose; confirm candidate_shortlist uses the `MEMBER: discriminator — present/absent` bar-separated form.

## Keyword endpoint prompt: embed full classification registry
Files: search_v2/stage_3/keyword_query_generation.py
Why: The initial prompt only summarized the 21 concept families in prose, leaving the 259 member definitions unseen by the LLM. With only family names and the enum's JSON-schema surface, the model had to rely on parametric knowledge to distinguish near-collision pairs (TRUE_STORY vs BIOGRAPHY, FEEL_GOOD_ROMANCE vs FEEL_GOOD). The definitions live in code; the prompt needs to carry them.
Approach: Added a module-level `_FAMILIES` list mapping each of the 21 family headers to its ordered list of UnifiedClassification member names, and a `_build_classification_registry_section()` function that renders the grouped listing at import time. Each entry shows `NAME — definition`, pulled from `entry_for(member).definition` so keyword/source-material/concept-tag definition edits flow through automatically. Replaces the hand-written prose family summary.
Design context: Two invariants enforced at import time so schema drift fails loudly: (1) every _FAMILIES member name must resolve to a CLASSIFICATION_ENTRIES registry entry; (2) every registry member must appear in exactly one family. Either violation raises RuntimeError at module load — adding a new OverallKeyword/ConceptTag/SourceMaterialType member forces the author to place it in a family in this file, which is where the prompt's taxonomic grouping lives.
Testing notes: Smoke-tested at import — SYSTEM_PROMPT builds to ~45K chars, registry section has 327 lines, all 259 members present. Behavioral coverage should confirm the LLM references the rendered definition text in candidate_shortlist discriminators (e.g., "BIOGRAPHY: dramatizes the life of a real person — no person named"), which is only possible when definitions are in-prompt.

## Keyword endpoint: query execution
Files: search_v2/stage_3/keyword_query_execution.py (new), db/postgres.py, search_improvement_planning/finalized_search_proposal.md, search_improvement_planning/open_questions.md
Why: Stage 3 keyword endpoint needs an executor that runs the LLM's chosen UnifiedClassification member against movie_card and returns an EndpointResult matching the dual-mode (dealbreaker / preference) contract the other stage 3 executors use.
Approach: `execute_keyword_query(spec, *, restrict_to_movie_ids=None)` resolves `spec.classification` through `entry_for(...)` to a single (backing_column, source_id) pair and issues one GIN `&&` overlap via the new `fetch_keyword_matched_movie_ids` helper in db/postgres.py. Binary scoring (1.0 / 0.0). One retry on DB error, empty EndpointResult on second failure — mirrors franchise/award/metadata executors. The DB helper whitelists the three legal columns because column names cannot be parameterized.
Design context: Simplified per user direction — no dual-backing into `genre_ids`, no cross-column unions. Updated finalized_search_proposal.md §Endpoint 5 (Data sources, Overlap rule, Execution Details) and the two relevant open_questions.md entries to match: Genre/`genre_ids` is excluded from this endpoint entirely (all 27 TMDB genres already live inside OverallKeyword), and each UnifiedClassification member resolves to exactly one column/ID pair.
Testing notes: needs an async integration test with a seeded movie_card row per source type (keyword_ids, source_material_type_ids, concept_tag_ids) verifying both dealbreaker and preference modes, plus the soft-failure path.

## Franchise endpoint contract realignment
Files: schemas/enums.py, schemas/franchise_translation.py, search_v2/stage_2.py, search_v2/stage_3/franchise_query_generation.py, search_v2/stage_3/franchise_query_execution.py, search_improvement_planning/finalized_search_proposal.md
Why: The franchise step-3 implementation had drifted from the actual execution contract in three ways: it still described fuzzy/trigram franchise matching even though execution uses exact equality on normalized stored strings; it treated `intent_rewrite` and `routing_rationale` too much like co-equal evidence rather than contextual hints around `description`; and it modeled structural/launch decisions as four booleans even though the user wanted `spinoff`/`crossover` grouped as flags and franchise-vs-subgroup launch represented as one mutually exclusive choice.
Approach: Replaced the old boolean-heavy `FranchiseQuerySpec` with a cleaner exact-match schema: `concept_analysis`, `lineage_or_universe_names`, `recognized_subgroups`, `lineage_position`, `structural_flags`, and `launch_scope`. Added `FranchiseStructuralFlag` and `FranchiseLaunchScope` enums, removed `name_resolution_notes`, and kept subgroup dependency only on `lineage_or_universe_names` — not on launch scope. Rewrote the franchise step-3 prompt around the clarified input hierarchy: `description` is authoritative for axis selection, `intent_rewrite` only contextualizes vague references, and `routing_rationale` is a hint that must not override evidence. The prompt now explicitly teaches exact stored-form matching after shared normalization, explains why the name fields are lists, carries subgroup naming guidance forward from the ingest-side franchise prompt, and keeps all step-3 descriptions in positive-identification form. Execution now derives the legacy DB booleans from `structural_flags` and `launch_scope`. Stage 2’s franchise examples were tightened to reinforce the same positive-form rule, and the finalized proposal’s Endpoint 4 section plus the shared step-3 input contract were updated to document the new schema and the exact-match design.
Design context: This keeps step 2’s current split behavior for compounds like “Marvel spinoffs” while still allowing step 3 to handle an unsplit item gracefully if one slips through. It also keeps the ingest/search alignment promise: the query-time LLM now uses the same canonical naming and subgroup heuristics as the ingest-side franchise classifier, which matters more now that matching is exact instead of fuzzy.
Testing notes: No repo tests were run per AGENTS.md. Validation consisted of `python -m py_compile` on the changed Python modules plus a direct `FranchiseQuerySpec` construction smoke test confirming enum serialization and structural-flag deduplication. Full behavioral coverage still needs prompt-level evals on named-franchise, subgroup, structural-only, and launcher queries, especially vague cases where `intent_rewrite` should clarify but not add axes.

## Metadata endpoint prompt + runtime operator cleanup
Files: implementation/classes/enums.py, implementation/classes/watch_providers.py, db/metadata_scoring.py, schemas/metadata_translation.py, search_v2/stage_2.py, search_v2/stage_3/metadata_query_generation.py, search_v2/stage_3/metadata_query_execution.py, search_improvement_planning/finalized_search_proposal.md
Why: The metadata step-3 review surfaced a finalized set of prompt and contract refinements: treat `routing_rationale` as a contextual hint instead of evidence, allow extra populated fields while keeping focus on the chosen `target_attribute`, add inclusive runtime operators so prompt translations do not need awkward threshold hacks, remove the awards-to-reception fallback, and stop duplicating streaming-service names in prompt text.
Approach: Extended `NumericalMatchOperation` with `GREATER_THAN_OR_EQUAL` / `LESS_THAN_OR_EQUAL`, then taught and executed those operators in both the legacy metadata scoring helper and the V2 metadata executor. Updated `metadata_query_generation.py` so the prompt receives `routing_hint` (derived from step 2's `routing_rationale`) and explicitly treats it as background context rather than evidence; `constraint_phrases` now draws only from `description` / `intent_rewrite`. Replaced the old "ONE SUB-OBJECT, NOT MANY" rule with a softer target-field-focus rule: prioritize the field matching `target_attribute`, but tolerate extras because execution ignores them. Removed the awards fallback from the reception attribute section. Reused `StreamingService` display names for both the step-2 metadata endpoint description and the step-3 metadata prompt so the tracked-service list comes from one enum-backed source of truth. Softened `schemas/metadata_translation.py` comments to match the same loose single-target contract. Updated the finalized proposal's runtime operator list to reflect the new inclusive options.
Design context: This keeps the user-approved architecture intact: step 3 still trusts upstream routing and makes the best metadata-space query it can, but it no longer over-anchors on the routing label or artificially forces single-field purity when execution does not require it. Inclusive runtime operators preserve the "translate intent into executable parameters" principle without introducing hidden numeric hacks like `> 89`.
Testing notes: No tests were run per AGENTS.md. Validation was limited to `python -m py_compile` over the changed Python modules to catch prompt-construction and enum/branching errors. Follow-up behavioral evals should specifically watch whether `routing_hint` reduces anchoring without hurting disambiguation, and whether the inclusive runtime operators materially improve translations like "at least 90 minutes" / "90 minutes or less."

## Entity endpoint contract alignment + schema/prompt cleanup
Files: search_v2/stage_2.py, schemas/enums.py, schemas/entity_translation.py, search_v2/stage_3/entity_query_generation.py, search_v2/stage_3/entity_query_execution.py, search_improvement_planning/finalized_search_proposal.md
Why: The entity step-3 review found three real issues: step 2 still showed a negative-form entity description example even though step 3 assumes positive-presence phrasing; the entity output schema had drifted from the finalized plan; and the current `EntityQuerySpec` / prompt bundled too much into `entity_analysis` while using the misleading shared field name `entity_name` even for literal title-pattern lookups.
Approach: Tightened the step-2 prompt so dealbreaker descriptions are always written in positive-presence form and direction alone carries inclusion vs exclusion, including fixing the stale entity example (`includes Adam Sandler in actors`, not `not starring Adam Sandler`). On the entity schema/prompt side, replaced `entity_analysis` with two narrower pre-generation fields: `entity_type_evidence` (lookup type + role signal inventory) and `name_resolution_notes` (brief canonicalization / literal-pattern note), and renamed the primary search-key field to `lookup_text`. Added `SpecificPersonCategory` so `primary_category` can no longer express `broad_person`. Updated the step-3 entity prompt to teach explicit evidence precedence (`description` > `routing_rationale` > `intent_rewrite` > parametric knowledge), clarified that title patterns are literal substring/prefix matches rather than exact dictionary lookups, and cleaned person-name wording so it no longer talks about "corporate suffixes." In the schema validator, normalized the old `prominence_evidence="not applicable"` sentinel back to null and defaulted actor-applicable null evidence to `"no prominence signal"` so actor lookups do not silently lose the field. Execution was updated for the `lookup_text` rename and the narrower `primary_category` enum. The finalized proposal's Endpoint 1 section now documents the new field layout, the literal title-pattern behavior, and the renamed search-text field so plan and implementation match again.
Design context: This keeps the user-approved entity architecture intact: flat schema, no extra invalid-state enforcement beyond what runtime already tolerates, no cap on character alternatives, and no new candidate-hint input. The change is mainly about making the model's reasoning fields smaller, clearer, and more aligned with the prompt-authoring conventions in docs/conventions.md while reducing accidental title-pattern/name conflation.
Testing notes: No tests were run per AGENTS.md. Validation was limited to `python -m py_compile` on the touched Python modules and a manual diff pass to confirm the step-2/step-3 contract, entity schema field names, execution references, and finalized plan wording all moved together. A future eval pass should specifically watch actor-applicable cases to confirm `prominence_evidence` is never left null when the actor table is in play unless the prompt/parser truly failed.

## Award endpoint prompt realignment to literal prize representation
Files: schemas/award_surface_forms.py, search_v2/stage_3/award_query_generation.py, schemas/award_translation.py, search_improvement_planning/finalized_search_proposal.md
Why: The award step-3 review uncovered a contract mismatch. The prompt still taught prize phrases like "Oscar-winning" and "Palme d'Or winners" as ceremony-only signals, which broadened the output away from the user's literal wording. It also treated step 2's `routing_rationale` too much like evidence, and the canonical prize-name examples were hand-maintained inside the prompt body.
Approach: Added a new programmatic surface-form registry in `schemas/award_surface_forms.py` that renders both the ceremony mapping table and the canonical prize-name table for the prompt. Rewrote the award prompt so `award_names` now represents named prize objects directly ("Oscar", "Golden Globe", "Palme d'Or", "Golden Lion") and does not auto-add the related ceremony; `ceremonies` is reserved for event/festival/awards-body wording like "at Cannes" or "nominated at Sundance". The prompt now exposes step 2 context as `routing_hint`, explicitly says `description` is the primary evidence, and forbids citing `routing_hint` inside `concept_analysis`. Updated `award_names` output guidance to reinforce the literal-representation rule, refreshed stale schema comments in `schemas/award_translation.py` to match the current THRESHOLD/3 generic-award-winning contract and the execution-layer fast-path restriction, and aligned the finalized proposal's Endpoint 3 section with the new prize-vs-ceremony boundary and generated prompt sections.
Design context: This keeps the existing schema structure and scoring fields intact per user direction, but shifts the prompt's reasoning toward "represent what was asked at the most direct level" instead of broadening to a parent ceremony. The generated surface-form registry follows the same no-drift pattern already used by the category-tag taxonomy. Prompt guidance was tightened around evidence precedence without changing the freeform `concept_analysis` field shape.
Testing notes: No tests were run per AGENTS.md. Validation was limited to `python -m py_compile` on the touched Python modules plus import-time prompt assembly checks. `SYSTEM_PROMPT` builds successfully with the generated tables. Follow-up behavioral evals should verify: "Oscar-winning" -> `award_names=["Oscar"]`, ceremonies null; "won at Cannes" -> `ceremonies=["Cannes Film Festival"]`, award_names null; "Cannes Palme d'Or winners" -> both axes populated; and Razzie phrasing still opts in only when explicitly named.

## Semantic endpoint planning decisions captured in proposal + open questions
Files: search_improvement_planning/finalized_search_proposal.md, search_improvement_planning/open_questions.md
Why: Planning session for the step 3 semantic endpoint resolved several open questions and reshaped parts of the proposal that were still V1-flavored. Update sweeps both docs so implementation can start from a consistent spec.
Approach:
- Endpoint 6 (Semantic): clarified that dealbreakers draw only from the 7 non-anchor spaces while preferences may use all 8; removed V1 "80/20 subquery+original blend" language — every vector search (including anchor) now uses a single LLM-generated query per selected space, no merging with the original user query.
- Added the finalized space-taxonomy convention: each `create_*_vector_text` function in `movie_ingestion/final_ingestion/vector_text.py` carries a docstring with Purpose / What's Embedded (exact structured labels) / Boundary / 2-3 example queries, imported verbatim into the step 3 prompt (convention, not code-gen).
- Finalized preference space-weight model: two-level categorical, `primary`=2 / `contributing`=1. No `minor` option — if a space's signal isn't meaningfully contributing, the LLM shouldn't select it. Drops the old t-shirt sizing (large/medium/small/not_relevant) and the stale max-across-spaces combining rule.
- Pure-vibe flow: retired the `(concept, space, subquery, role)` tuple output. New shape is one query per selected space, absorbing all concepts routed to that space and phrased in that space's native vocabulary (e.g., "scary but funny" in viewer_experience becomes `emotional_palette: darkly funny, gallows humor` + `tension_adrenaline: unsettling, creeping dread`). Rewrote "Why Individual Searches, Not Combined" into "Why Per-Space, Not Per-Concept" to reflect this.
- Added explicit Exclusion-Only Edge Case rule to the pure-vibe section and Phase 4a checkpoint: if only exclusions exist (no inclusion dealbreakers), fall through with a two-step rule — preferences take the candidate-generation role if any exist, otherwise top-K by the default quality composite (0.6*reception + 0.4*popularity).
- Rewrote Scoring Function Modes: removed stale "preserved similarity for primary" and "diminishing returns for regular" entries (both contradicted the finalized raw-weighted-cosine preference math); consolidated into Threshold+flatten / Raw weighted-sum cosine / Pass-through / Sort-by.
- Added new "Semantic Endpoint — Finalized Implementation Decisions" section summarizing all of the above plus the Pydantic-per-multi-source-space convention (anchor, plot_events, production, reception get dedicated models for ingest+query shape parity) and the retry-once-then-empty transient error policy.
- Decisions Deferred section: removed items that moved to Finalized; kept elbow/floor algorithm, cache backend, cross-space cosine comparability test, zero-dealbreaker quality floor test, and P_CAP/E_MULT empirical tuning. Deprioritized semantic-exclusion prompt tightness given match-then-penalize (not hard-filter).
- open_questions.md: marked "pure vibes dealbreakers", "t-shirt sizing", "efficient metadata generation", "multi-vector combining (max)", "step 2 exclusion query formulation" as DECIDED/SUPERSEDED/DEPRIORITIZED with cross-references back to the proposal; added new DECIDED entry for exclusion-only queries.
Design context: No code yet; this is planning doc alignment for step 3 semantic endpoint implementation. All V1 retrieval assumptions (80/20 blend, per-concept subqueries, anchor for dealbreakers, t-shirt weights) have been excised from the semantic endpoint spec. Empirical-tuning questions remain flagged as deferred to implementation.
Testing notes: N/A — planning-only changes. First implementation commit will introduce the step 3 semantic endpoint module under `search_v2/stage_3/` alongside the existing entity/metadata/awards/franchise/keyword/trending endpoints.

## Semantic endpoint: four-scenario execution model, Option B for no-dealbreaker preferences
Files: search_improvement_planning/finalized_search_proposal.md, search_improvement_planning/open_questions.md
Why: Follow-up planning session tightened the flow control across the semantic endpoint. The prior proposal conflated "pure-vibe" (dealbreaker-driven) with "zero-dealbreaker" (preference-only and exclusion-only) behind one vague codepath and left the no-dealbreaker-but-has-preferences case unresolved. Documented four explicit scenarios (D1/D2 for dealbreakers, P1/P2 for preferences) with a table covering trigger, candidate generation, similarity source, score transform, and final-reranking contribution for each.
Approach:
- Endpoint 6: added an "Execution Scenarios" subsection with two side-by-side tables (dealbreaker-side D1/D2, preference-side P1/P2), plus exclusion-only as a named edge case. Included explicit "why" paragraphs for global-elbow-calibration-in-both-D1-and-D2 (scoring must be invariant to candidate-generation role) and preference-keeps-preference-semantics-in-P2 (step 2's classification is binding regardless of mechanism).
- Finalized Option B for the zero-dealbreaker + preference-exists case: semantic preference drives candidate generation (top-N per selected space, union), but scores as a preference — raw weighted-sum cosine normalized by Σw, contributing to preference_contribution. dealbreaker_sum = 0. Rationale: candidate-generation mechanism is orthogonal to final-ranking role.
- Reworked the old "Zero-Dealbreakers Edge Case" into "Zero-Dealbreakers, Preference-Driven Retrieval (Scenario P2)" with explicit step-by-step execution and a note on P_CAP-bounded final scores being acceptable (within-query ranking only).
- Tightened "Exclusion-Only Edge Case" so it explicitly requires zero inclusion dealbreakers AND zero preferences; cross-linked to P2 for the "exclusions + preferences" case ("not clowns, something cozy" → P2, not browse fallback).
- Updated Phase 4a checkpoints to distinguish three cases: D2 (semantic-only inclusion dealbreakers), P2 (zero inclusion dealbreakers but preferences exist), exclusion-only (neither). Updated Pure-Vibe Flow intro to be explicit that it documents D2, and cross-reference the P2 section for the related variant.
- Updated "Semantic Endpoint — Finalized Implementation Decisions" section: replaced the single "exclusion-only edge case" bullet with a consolidated "Four execution scenarios" bullet summarizing D1/D2/P1/P2 plus the global-elbow-calibration and preference-semantics-in-P2 rationale; kept the narrower exclusion-only bullet beneath it.
- open_questions.md: updated the "pure vibes-based dealbreakers" entry to reference the four-scenario model; reworded the "exclusion-only" entry to clarify it only applies when preferences are also absent; added new DECIDED entry "No-dealbreaker preference-only queries: what generates candidates?" documenting Option B with rationale against Option A (browse fallback).
Design context: Clarified a load-bearing architectural rule — step 2's dealbreaker-vs-preference classification is binding for final-ranking role regardless of who generates candidate IDs. Keeps semantic dealbreaker scoring invariant across D1/D2 (global calibration probe in both cases), so cache hit rates stay high and scoring is consistent query-to-query.
Testing notes: N/A — planning-only. Evaluation items for when the endpoint is implemented: (1) verify P2 final-score ordering matches user expectations even though dealbreaker_sum is always 0; (2) confirm global calibration cache hit rates are comparable between D1 and D2 (same hash key); (3) spot-check that the P2 vs browse-fallback split on "exclusions + preferences" queries lands on P2 as intended.

## Semantic endpoint: finalized elbow/floor detection algorithm
Files: search_improvement_planning/finalized_search_proposal.md
Why: Closed the deferred decision on exact elbow/floor detection for semantic dealbreaker scoring (D1 + D2) and semantic exclusion scoring. Prior doc listed a "working direction" (two-knee Kneedle with fixed-gap fallback) but left the elbow selection rule, floor-gap constant, and pathology detector unspecified. Empirical distribution shapes across four test concepts (twist ending, dark and gritty, christmas, funny) plus a web review of Kneedle literature grounded the resolution.
Approach:
- Added a detailed 7-step algorithm bullet under "Semantic Endpoint — Finalized Implementation Decisions": corpus top-N probe (N=2000), EWMA smoothing (span=max(5, N/100)), Kneedle with curve='convex', direction='decreasing', S=1, online=True, all_knees=True.
- Pathology check: `max(y_diff) < 0.05` on the normalized difference curve means no real elbow exists → fall back to percentage-of-max (elbow = max × 0.85, floor = max × 0.50) and log for audit. Single diagnostic — linearity R², dynamic range, and concave-down shape all collapse into this one signal.
- Elbow selection rule: first knee by rank, with a rank-10 safeguard — if the first knee lands at rank < 10 AND another knee exists, skip forward to the next; if the first knee is the only one, use it as-is rather than invent a later elbow. Never pick the largest-bulge knee; first qualifying knee is always the target. Rationale: outlier-driven early knees can pinch the 1.0 boundary too tightly, but a solo early knee is still the real signal.
- Floor selection: second knee if Kneedle detected two or more (natural bimodal signal — e.g., Christmas); otherwise `floor = max(elbow_sim − 2 × (max_sim − elbow_sim), 0.0)`, gap-proportional so the decay zone widens for sharp elbows and compresses for narrow-band distributions.
- Scoring transform unchanged from the proposal direction: 1.0 above elbow, linear decay between floor and elbow, 0.0 at/below floor.
- Narrowed the matching "Decisions Deferred to Implementation" entry: struck through as resolved, listed the residual tuning items (γ non-linear decay, N, EWMA span, rank-10 threshold, 0.05 pathology cutoff) that stay deferred to evaluation.
Design context: The rank-10 safeguard is deliberately "skip forward only if a later knee exists" rather than "skip forward unconditionally to an invented floor" — preserves data-driven behavior when a concept legitimately has very few true matches ("only 3 movies are genuinely about X"). Floor-gap scaling addresses the empirical finding that elbow-percentage varies 75–90% across concepts; elbow-gap is the more invariant quantity. See turn-by-turn discussion with web-sourced Kneedle references (Satopää et al., kneed library, Kneeliverse multi-knee extension).
Testing notes: Evaluation items for when the endpoint is implemented: (1) confirm first-knee-with-rank-10-safeguard selects sensibly across the four empirical test concepts (twist ending, dark and gritty, christmas, funny) and on a handful of new concepts that stress the edge cases (very narrow top cluster, very flat distribution, strongly bimodal); (2) measure what fraction of queries hit the pathology fallback (`max(y_diff) < 0.05`) — high rate signals either a bad N, a weak embedding space, or concepts that shouldn't have been routed to semantic; (3) spot-check that the second-knee floor for bimodal distributions matches intuition (Christmas tail movies at floor boundary should feel "Christmas-adjacent but not Christmas"); (4) tune γ only if linear decay shows systematic mid-range scoring issues.

## Semantic endpoint: step 3 output schemas (bodies + translation specs)
Files: schemas/semantic_bodies.py, schemas/semantic_translation.py
Why: Step 3's semantic endpoint LLM needs to emit query specs that embed into the same structured-label format the ingestion pipeline produces. The prior proposal used free-form `query_text` strings — making the LLM the only thing enforcing format parity, a silent drift risk with no hard-failure signal. The fix is concrete per-space objects, each with an `embedding_text()` method that reproduces the ingestion-side vector text sequence exactly, wrapped in a discriminated union keyed on `space`.
Approach:
- `schemas/semantic_bodies.py`: 8 Body classes (one per vector space), each with an `embedding_text()` method that mirrors the ingestion-side format verbatim.
  - `AnchorBody` mirrors `create_anchor_vector_text` but deliberately omits `title`, `original_title`, and `maturity_summary` — ingestion identity/filter signals the query LLM shouldn't generate.
  - `PlotEventsBody` mirrors `PlotEventsOutput.embedding_text` — raw prose, no label.
  - `PlotAnalysisBody`, `ViewerExperienceBody`, `WatchContextBody`, `NarrativeTechniquesBody`, `ReceptionBody` mirror their ingestion-side counterparts' `embedding_text` methods exactly, dropping ingest-only fields (`identity_note`, Reception extraction zone, `major_award_wins`, and all justification/reasoning wrappers).
  - `ProductionBody` mirrors `create_production_vector_text` sans the `is_animation()` gate (ingest-time data hygiene, not a query concern).
  - Two shared sub-models (`TermsSection`, `TermsWithNegationsSection`) declared locally rather than imported from `schemas/metadata.py` — names match intentionally, but keeping them distinct prevents unrelated coupling.
  - Duplicating `embedding_text()` logic against ingestion-side is deliberate; factoring into one helper would hide divergence in refactors, whereas duplication makes it visible in code review. Noted in-file.
- `schemas/semantic_translation.py`:
  - Two enums: `DealbreakerSpace` (7 non-anchor) and `VectorSpace` (all 8). Declared separately rather than subclassed because OpenAI structured output needs concrete JSON-schema enum restrictions per field.
  - `PreferenceSpaceWeight` enum (`primary` / `contributing`).
  - 7 `*Dealbreaker` wrappers and 8 `*PreferenceEntry` wrappers, each pinning a `Literal[...]` on its space enum member. Discriminated-union dispatch (`Field(discriminator="space")`) makes space/body mismatch a schema-level error.
  - `SemanticDealbreakerSpec` (covers D1 + D2 + semantic exclusions): `space_selection_evidence` → `query_design_note` → `body`. Dealbreakers always pick exactly one non-anchor space.
  - `SemanticPreferenceSpec` (covers P1 + P2): `concept_inventory` → `space_plan` → `space_queries` (1..8 entries, at most one per space). Each preference entry carries `space_rationale` adjacent to its space decision for local CoT.
  - `_no_duplicate_spaces` validator on `SemanticPreferenceSpec` guards the `Σ(w × cos) / Σw` downstream sum — duplicate spaces would double-count.
  - No class-level docstrings, no `Field(description=...)` — all LLM-facing guidance lives in the system prompt per `schemas/keyword_translation.py` / `schemas/award_translation.py` convention.
Design context: Resolves the `query_text` → concrete-object decision recorded in `search_improvement_planning/finalized_search_proposal.md` (Endpoint 6: Semantic). Single-call concrete generation was chosen over a two-call "generate text then shape it" pipeline — the latency/cost savings from skipping a second LLM call outweigh the modest input-side schema verbosity, which is further mitigated by prompt caching. Deferred follow-up: factor the ingestion-side `*Output` classes to embed the new `*Body` as a sub-model so both sides share one source of truth for `embedding_text()` long-term.
Testing notes: Smoke tests passed locally — all 8 `embedding_text()` methods produce the expected labeled sequences; `SemanticDealbreakerSpec` + `SemanticPreferenceSpec` construct via dict input with discriminator dispatch; duplicate-space validator fires on reuse; JSON schemas round-trip (dealbreaker ~7K chars, preference ~10K chars — well within prompt-cache territory). Follow-ups for when the endpoint runs end-to-end: (1) unit-test parity between each Body's `embedding_text()` and the ingestion-side `embedding_text()` for a representative movie to catch label drift; (2) confirm OpenAI structured output handles the discriminated-union schema without needing the "strict mode" workaround some frameworks require; (3) evaluate LLM fidelity per-space — whether the model fills the concrete shape well for each of 8 spaces, especially narrative_techniques (9 sections, highest section count).

## Semantic endpoint schemas: code-review polish
Files: schemas/semantic_bodies.py, schemas/semantic_translation.py
Why: Post-commit review on the step 3 semantic schemas surfaced four small-but-real issues — awkward `spec.body.body.embedding_text()` access path, an over-defensive duplicate-space validator, a no-op `use_enum_values=True` on top-level specs, and an inherited `[:3]` filming-locations cap that shouldn't apply on the query side.
Approach:
- Renamed the inner `body` field on all 7 dealbreaker wrappers and all 8 preference entry wrappers to `content`. Access path is now `spec.body.content.embedding_text()` / `entry.content.embedding_text()` — reads as "the discriminated union's content is this Body" rather than the previous confusing `body.body` chain.
- Dropped `[:3]` truncation from `ProductionBody.embedding_text()`. The cap exists on the ingestion side as data hygiene against scraped noise; on the query side the LLM emits only intentional locations, so silently truncating a 4th entry would discard real signal. Updated the in-file comment to document the intentional divergence.
- Simplified `_no_duplicate_spaces` on `SemanticPreferenceSpec`: `VectorSpace(str, Enum)` members already compare equal to their string values and share a hash, so the previous `isinstance` branching was unnecessary. One set handles both forms.
- Removed `use_enum_values=True` from `SemanticDealbreakerSpec` and `SemanticPreferenceSpec` — those specs have no direct enum fields (all enums live on nested wrappers which don't carry that config), so the flag was a no-op that implied more behavior than it delivered.
Design context: No behavioral change in the semantic endpoint beyond letting filming_locations pass through uncapped. The other three are purely ergonomic/clarity wins identified during code review.
Testing notes: Smoke test re-run after changes confirms: filming_locations with 5 entries preserves all 5 in embedding_text; `spec.body.content.embedding_text()` dispatch works from dict input; preference entry `entry.content` access works; duplicate-space validator still rejects repeats. All `embedding_text()` outputs still match the ingestion-side format.

## Semantic endpoint reasoning fields: rename + drop `space_plan`
Files: schemas/semantic_translation.py, search_improvement_planning/finalized_search_proposal.md
Why: The draft semantic step-3 schema carried reasoning fields named for what they looked like (`space_selection_evidence`, `query_design_note`, `concept_inventory`, `space_plan`, `space_rationale`) rather than for the specific downstream decision each one scaffolds. Re-audited against the `Cognitive-scaffolding field ordering`, `Evidence inventory, not rationalization`, and `Brief pre-generation fields, no consistency coupling` conventions; renamed each field to match its scaffolding role and dropped one redundant field.
Approach:
- `SemanticDealbreakerSpec`:
  - `space_selection_evidence` → `signal_inventory`. Frames the field unambiguously as an evidence inventory (cite phrases → implicated spaces, explicit empty-evidence path) rather than a space-selection justification, which was read as "argue for a space" and invited over-inference.
  - `query_design_note` → `target_fields_label`. Names what the field actually primes (which sub-fields inside the chosen-space body will carry signal) and reinforces the brief-label form required by the "no consistency coupling" convention.
- `SemanticPreferenceSpec`:
  - `concept_inventory` → `qualifier_inventory`. "Qualifier" matches the proposal's vocabulary for the grouped preference description's constituent parts and makes the decomposition step explicit.
  - Dropped top-level `space_plan`. Redundant with `qualifier_inventory` (already names qualifier→space mappings) plus per-entry priming; its sentence-form holistic plan was the textbook consistency-coupling risk — once committed, subsequent entries would become "write the body that matches the plan" rather than independent per-space decisions.
  - Per-entry `space_rationale` → `carries_qualifiers` across all 8 `*PreferenceEntry` wrappers. One brief label ("carries: dark, slow-burn") primes all three downstream decisions for that entry — space commit, weight enum, body content — and the rename forbids the sentence form that would have templated the body.
- Planning doc: Added a `Reasoning fields` subsection inside `Endpoint 6: Semantic`, mirroring the award endpoint's pattern. Documents each field, its position, what it scaffolds, and the rationale. Explicit "no top-level `space_plan`" block preserves the why-we-dropped-it decision for future readers.
Design context: Decision recorded in the main planning doc (Endpoint 6 → Reasoning fields). Rename was motivated by the principle that reasoning-field *names* must match the downstream decision they scaffold — vague names invite the LLM to drift toward rationalization-after-the-fact, which the evidence-inventory framing convention was written to prevent.
Testing notes: Smoke test passes — `SemanticDealbreakerSpec` with `signal_inventory` + `target_fields_label` + body discriminator dispatch constructs from dict input; `SemanticPreferenceSpec` with `qualifier_inventory` + `space_queries` (each entry carrying `carries_qualifiers`) constructs and enforces `_no_duplicate_spaces`. When the step-3 semantic system prompt is authored, it must instruct the model to produce `target_fields_label` and `carries_qualifiers` in brief label form (not sentences) — the schema does not enforce this alone, and the conventions make it the system prompt's responsibility.

## Semantic dealbreaker spec: reasoning field renames (user edit)
Files: schemas/semantic_translation.py
Why: After the code-review polish pass, renamed `space_selection_evidence` → `signal_inventory` and `query_design_note` → `target_fields_label` on `SemanticDealbreakerSpec`. Expanded the in-file field-order comment to specify each field's expected form (evidence inventory with explicit empty-evidence path; brief 2-6 word label form for target fields, not sentence form — avoids consistency-coupling failure mode).
Approach: Field-shape intent matches what the ingestion schemas already codify: evidence-inventory framing ("cite concrete phrases") per conventions.md, and brief-label pre-generation form (not sentence) to prime without constraining.
Design context: Naming now reflects the schema's role — `signal_inventory` names the evidence-gathering role directly; `target_fields_label` names what the field produces (a label of target structured sub-fields) rather than a generic "query design note." Downstream callers: system prompt authoring for `search_v2/stage_3/semantic_query_generation.py` will thread through these names.
Testing notes: Smoke tests from the prior entries exercise construction via dict, which is resilient to the rename since it keys on field names; if any test hard-codes the old field names they'll need updating when tests are authored.

## Step 3 semantic generator + prompts
Files: search_v2/stage_3/semantic_query_generation.py, schemas/semantic_translation.py

### Intent
Wire the last missing stage-3 translator. The sixth endpoint (semantic) now has a generator module exposing two public async functions — one per spec — with prompts authored from scratch and structured to apply the small-LLM conventions we codified in prior metadata-generation work. Moves the LLM-facing "why this field exists" guidance off the Pydantic classes into the system prompts where the `No docstrings on Pydantic response_format classes` convention says it belongs.

### Key Decisions
- **Two public functions, two prompts, one module.** `generate_semantic_dealbreaker_query` and `generate_semantic_preference_query` share the direction-agnostic framing, body-authoring principles, and space-taxonomy entries as module-level constants, but each has its own task / reasoning / output section tuned to the distinct decisions it drives. Keeping both in one file makes the shared pieces visibly shared — splitting across two modules would hide drift the way duplicated prompt text in evals used to.
- **Inline per-space taxonomy (with a TODO).** The finalized proposal says the taxonomy should live on each `create_*_vector_text` function in `movie_ingestion/final_ingestion/vector_text.py` and be imported verbatim. Current docstrings don't match that shape (half are missing, others dev-only), so the canonical Purpose / What's Embedded / Boundary / Example Queries entries are authored as module-level constants in the generator file. Flagged as a follow-up to propagate back to `vector_text.py`.
- **Dealbreaker vs preference space set.** Dealbreaker taxonomy excludes anchor (7 spaces); preference taxonomy includes all 8 with an explicit note that anchor is only available for preferences. Matches the proposal's rule that anchor is too diffuse for single pass/fail dealbreakers.
- **Small-LLM conventions baked into each block.** Evidence-inventory framing on `signal_inventory` / `qualifier_inventory` with explicit "no phrase implicates X" / "no clear space" empty-evidence paths; brief-label enforcement on `target_fields_label` / `carries_qualifiers` (label form, never sentences); principle-based guidance throughout (no failure catalogs); no nonzero floors or cross-section term targets; evaluation guidance over outcome shortcuts (space taxonomy teaches boundaries, no "if word X → pick space Y" rules); example-eval separation (worked examples drawn from concepts outside the proposal's canonical examples — heist-unraveling, post-colonial identity, slow-burn-rainy-day, family-movie-night).
- **Worked examples placed inside `_*_OUTPUT` blocks.** Two per prompt, each threading through every reasoning field and the body sub-fields so the model sees the full pipeline. Both preference examples include an anchor entry to model the "broad vibe" case alongside targeted spaces.
- **Schema guidance stripped.** The long `signal_inventory` / `target_fields_label` / `qualifier_inventory` / `carries_qualifiers` explanation blocks are removed from `schemas/semantic_translation.py`; replaced with short ordering comments that point at the generator module. Verified via `openai.lib._pydantic.to_strict_json_schema()` — zero `description` fields in either spec's emitted JSON schema.
- **Stage-3 signature parity.** Both generator functions match the exact shape used by entity / franchise / keyword / award / metadata: `(intent_rewrite, description, routing_rationale, provider, model, **kwargs) -> tuple[Spec, int, int]`. No defaults on `provider` or `model` per the personal-preferences convention; identical manual-strip + non-empty validation with the same TODO pointing at the not-yet-built shared request model.

### Testing Notes
- Smoke test (run): imports succeed; prompt assembly produces ~14K / ~19K chars for dealbreaker / preference; taxonomy coverage verified (all 7 / 8 space names present in the correct prompts, anchor absent from dealbreaker); both specs construct from dict input with discriminator dispatch; `_no_duplicate_spaces` fires on repeated spaces; zero `description` fields leak into `to_strict_json_schema()` output for either spec.
- Deferred for end-to-end: a notebook or one-off script that drives the two functions against a live provider across the two worked examples (heist / post-colonial for dealbreaker; slow-burn-rainy-day / family-movie-night for preference) plus an anchor-heavy broad-vibe request. Will be added alongside the semantic evaluation harness in the next work item.
- Watch items for the eval pass: whether small models keep `target_fields_label` and `carries_qualifiers` in brief label form (the prompt forbids sentence form but does not enforce it structurally); whether two-to-three-word compact preference phrases get decomposed instead of blob-handled; whether anchor is chosen appropriately for broad-vibe qualifiers without being over-selected in multi-space requests.

## Semantic endpoint follow-up: central/supporting weights, hint-only context, and optional bodies
Files: schemas/semantic_bodies.py, schemas/semantic_translation.py, search_v2/stage_3/semantic_query_generation.py, search_improvement_planning/finalized_search_proposal.md
Why: Follow-up review of the new semantic step-3 endpoint surfaced three prompt/schema issues worth fixing now: (1) `primary` / `contributing` subtly implied a single-winner space hierarchy even though multi-primary behavior is valid and common; (2) the prompt still treated `intent_rewrite` / routing context a little too much like evidence rather than disambiguation hints; and (3) query-side prose-first bodies (`PlotEventsBody`, `ReceptionBody`) still had required fields, which could pressure the LLM into filler when the real signal was sparse.
Approach:
- Renamed the semantic preference space-weight enum values from `primary` / `contributing` to `central` / `supporting` in `PreferenceSpaceWeight`, the semantic generator prompt text, and the finalized semantic-endpoint planning sections. Kept the numeric mapping unchanged (`central` = 2, `supporting` = 1); this is a naming/behavior-framing fix, not a scoring redesign.
- Reframed prompt inputs so `description` is the primary evidence and both `intent_rewrite` and prompt-surface `routing_hint` are only hints for understanding what `description` refers to. In both dealbreaker and preference prompts, the reasoning-field guidance now explicitly forbids citing `routing_hint` in `signal_inventory` / `qualifier_inventory`.
- Tightened the preference prompt away from "2-3 qualifiers should almost always become 2-3 spaces." New guidance says to choose the smallest set of spaces that each provide genuinely strong signal and not to add a space just because it can be weakly justified. This preserves multi-space behavior without biasing toward over-splitting.
- Added space-sensitive body-authoring guidance: translate into each space's native format, allow modest schema-native expansion where it sharpens the same supported signal (especially viewer_experience), keep plot_events close to the described situation in compact prose, and keep plot_analysis tighter than viewer_experience so it does not become an inference license.
- Relaxed query-side body schemas so no individual space object forces content. `PlotEventsBody.plot_summary` and `ReceptionBody.reception_summary` are now optional, and their `embedding_text()` methods return empty output cleanly when absent. Added an in-file comment documenting the intent: step 1/2 should normally prevent empty bodies, but the schema should not force invention just to satisfy validation.
- Updated the finalized proposal's semantic-endpoint sections so the public design record matches the implemented naming (`central` / `supporting`) and weight semantics.
Design context: This preserves the existing semantic endpoint architecture: same discriminated-union bodies, same two-level preference weighting, same direction-agnostic step-3 contract, same inline taxonomy/TODO, and the same brief-label reasoning fields. Deliberately did NOT implement the larger schema redesign or move the taxonomy source-of-truth into `vector_text.py` yet — those were explicitly deferred.
Testing notes: Ran `python -m py_compile schemas/semantic_bodies.py schemas/semantic_translation.py search_v2/stage_3/semantic_query_generation.py` successfully. Did not run unit tests per repo test-boundary rules.
