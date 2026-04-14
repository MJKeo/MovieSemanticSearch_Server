# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Rename country_ids → country_of_origin_ids
Files: db/init/01_create_postgres_tables.sql, db/postgres.py, schemas/movie.py, movie_ingestion/final_ingestion/ingest_movie.py, search_improvement_planning/v2_data_needs.md | Disambiguated column/variable name to match planning docs — "country_ids" was ambiguous (could mean filming location, setting, or distribution territory). DB was recently wiped so no migration needed.

## Plot analysis vector — V2 structured-label embedding format
Files: schemas/metadata.py, movie_ingestion/final_ingestion/vector_text.py, search_improvement_planning/v2_data_architecture.md

Why: Align the plot_analysis vector with the V2 structured-label embedding
format, which is a prerequisite for cross-space rescoring (see
search_improvement_planning/new_system_brainstorm.md "Embedding Format:
Structured Labels"). The flat/partially-labeled format caused per-attribute
signal dilution that the new architecture cannot tolerate.

Approach: Rewrote `PlotAnalysisOutput.embedding_text()` so every field is
emitted with an explicit snake_case label matching the Pydantic field name
(`elevator_pitch:`, `plot_overview:`, `genre_signatures:`, `conflict:`,
`themes:`, `character_arcs:`). Field order puts the shortest/highest-signal
capsule first, then prose, then enumerated categorical slots. `character_arcs`
is placed adjacent to `themes` because plot_analysis character arcs are
*thematic* arcs (e.g. "mentor's sacrificial legacy"), semantically closest
to thematic_concepts — distinct from narrative_techniques' film-language arc
labels. `__str__()` now delegates to `embedding_text()` to keep them in
lockstep (previously two parallel implementations drifted).

Also removed the TMDB-genre merge from `create_plot_analysis_vector_text()`:
under V2, genres are a deterministic hard filter via `movie_card.genre_ids`,
and the LLM-generated `genre_signatures` already carry the compound thematic
phrasing this space owns. Appending bare enum labels diluted the structured
field.

Design context: search_improvement_planning/v2_data_architecture.md §8.3
(updated in this changeset) and §8 header for the V2 format rationale.
Decision to keep character_arcs in plot_analysis (rather than delegating
fully to narrative_techniques) was discussed with the user: the two spaces
host semantically different arc concepts and should not be collapsed.

Testing notes: Existing plot_analysis unit tests assert the old embedding
format and will need updates in a separate testing phase (per
.claude/rules/test-boundaries.md, not touched here). No re-ingestion yet —
that happens as part of the broader V2 rollout.

## Franchise generator v3 — prompt + schema rewrite from test analysis
Files: movie_ingestion/metadata_generation/prompts/franchise.py, schemas/metadata.py

### Intent
Full rewrite of the franchise generator prompt and FranchiseOutput schema
descriptions, driven by per-candidate analysis of the v2 test run (79
movies × 4 models × 3 samples). The v2 evaluation surfaced three failure
classes the v2 prompt could not fix without a structural redesign:
(1) spinoff was being over-applied to shared-universe pillar films
(Wonder Woman, Black Panther) and legacy sequels (Creed) — every wrong
trace followed the same shortcut "character appeared before + now has
own film = spinoff"; (2) "prefer missing over wrong" framing caused
medium/low tiers to correctly punt to role=None on Transformers,
Guardians of the Galaxy, Dragon Ball Z, Rurouni Kenshin, fighting the
SOT — the prompt was biasing toward an outcome rather than critical
thinking; (3) culturally_recognized_groups failures dominated the raw
row count because the prompt used brainstorm-then-filter framing and
the Raiders-of-the-Lost-Ark → ["original trilogy"] example trained
models to pad every trilogy with "original trilogy".

### Key Decisions
- Prompt architecture rewritten around concept_tags.py patterns:
  evidence hierarchy (direct / concrete inference / parametric) stated
  up front; per-field numbered procedural walkthroughs replace
  free-form "Step 1 Evidence / Step 2 Analysis" bullets; comparative
  evaluation for franchise_role ("build case for each candidate role,
  commit to strongest") replaces first-match top-to-bottom scan;
  EMPTY-THEN-ADD framing for culturally_recognized_groups replaces
  brainstorm-then-filter; IS NOT clauses added to every field.
- Spinoff operationalized with three constraints that must ALL hold:
  (a) MINOR IN SOURCE — the character/element/subplot was minor, not
  a lead, co-lead, or planned shared-universe pillar; (b) GOES
  SOMEWHERE NEW — new protagonist/story spine/setting focus;
  (c) LEAVES THE SOURCE BEHIND — source film's main characters and
  plot are not the focus here (this is what distinguishes spinoffs
  from legacy sequels like Creed, where Rocky is a returning major
  character). Explicit "Examples where constraints fail" block lists
  Wonder Woman, Black Panther, Creed, Ghostbusters: Afterlife as
  mainline not spinoff, with the specific constraint each fails.
- NULL role is now a first-class answer with two documented valid
  cases, not a hedge: (i) first theatrical entry in a franchise
  whose prior existence is limited to non-cinematic media (toys,
  games, TV, anime, manga, comics-only) — Transformers 2007, Sonic
  2020, Detective Pikachu, Iron Man 2008, Rurouni Kenshin Origins,
  X-Men 2000 (Fox continuity); (ii) documentaries. "Null is NOT
  correct when choosing between two plausible roles — commit to the
  stronger" is stated explicitly. SOT needs realignment on the null
  cases in a separate pass before re-running the test set.
- "Prefer missing over wrong" language removed throughout. Replaced
  with positive commitment framing at the top of the prompt and in
  every field's procedure: "when evidence points to a franchise,
  commit; null is reserved for genuine non-fit, not uncertainty".
- culturally_recognized_groups is rewritten with three named evidence
  tiers (studio/official usage, mainstream critical convention,
  widely-used fan terminology), and the IS NOT filter now explicitly
  blocks bare sub-series restatements ("minions", "creed", "guardians
  of the galaxy", "the avengers", "puss in boots", "hobbs and shaw",
  "fantastic beasts", "iron man trilogy") which were the single
  biggest noise source in the v2 test. The Raiders → ["original
  trilogy"] example booby trap is removed; "original trilogy" is now
  explicitly scoped to Star Wars only. The normalization rule to
  drop first names on director-era labels is kept, with the Sherlock
  Holmes case added as an explicit example.
- launches_subgroup is coupled to culturally_recognized_groups via
  an internal consistency check ("if true, the launched subgroup's
  label MUST appear in the groups list"). Addresses action item #8
  from the v2 iterations report. Mad Max 1979 is given as an explicit
  FALSE example because "the original Mad Max trilogy" is not a
  culturally-used label despite being first of a trilogy.
- is_prequel is tightened with IS NOT clauses covering reboots set
  early (Batman Begins), flashback-heavy mainline films, and films
  set in the same time period as an earlier entry.
- Top-level entity rule is elevated into the franchise_name procedure
  as a forced step (not just stated in a separate section) — this is
  the minimal-tier failure mode where models say name='batman' /
  'x-men' / 'spider-man' because they don't re-apply the rule at
  decision time.
- INPUTS section now annotates every input with its reasoning role
  (primary / confirming / cross-reference signal) and which fields
  it serves. top_billed_cast is explicitly called out as the primary
  signal for the spinoff "prior-role prominence test".
- FranchiseOutput schema descriptions encode the same procedural
  walkthroughs in compact numbered form so the structured-output
  JSON schema itself carries the procedure, not just the system
  prompt. FranchiseRole enum unchanged (no roles added or removed;
  null covers the new non-cinematic-IP-first-theatrical case).

### Planning Context
Evaluation report lives at
search_improvement_planning/franchise_test_iterations.md (analyses
v1 and v2 runs). Per-candidate analysis in this session's chat
history identified which failures were genuinely harmful vs.
debatable vs. inconsequential and mapped each recurring failure
pattern to a specific prompt-language root cause. The rewrite
targets those root causes rather than patching individual failures.

### Testing Notes
- Next step is to re-run the v2 test set (franchise_test_results.json)
  against the new prompt and compare per-candidate H/D/I counts. Expect
  large drops in culturally_recognized_groups noise and in spinoff
  over-application; expect Transformers / Guardians / Dragon Ball Z /
  Rurouni Kenshin role=None to now be correct (not D) provided SOT is
  also updated.
- SOT needs pre-run realignment for the explicit null-role cases
  documented in the new prompt. Without SOT updates, re-run numbers
  will still show Transformers-class failures as wrong even though
  they are now the prompt's intended answer.
- Verify launches_subgroup / culturally_recognized_groups co-occurrence
  after re-run (action item #8 from iterations report).

## Franchise generator v2 — definitions, schema, prompt, SOT
Files: schemas/enums.py, schemas/metadata.py, movie_ingestion/metadata_generation/prompts/franchise.py, movie_ingestion/metadata_generation/generators/test_franchise.ipynb

### Intent
Re-ground the franchise generator on a tightened, internally consistent set
of definitions. Analysis of the last test-set run (76 movies × 3 tiers × 3
samples) showed the majority of harmful failures traced back to four
definitional holes: loose role definitions, umbrella-vs-lineage tension in
`franchise_name`, no way to express prequels or subgroup-launches, and no
`crossover` role. This change locks in resolutions for all four, updates the
Pydantic schema + enum to match, rewrites the system prompt, and updates the
test notebook SOT and evaluation cells. No DB/ingestion code touches franchise
fields yet so the change is contained to the generator + tests.

### Key Decisions
- `franchise_name` now always carries the TOP-LEVEL brand entity (Marvel,
  DC Comics, Harry Potter, Godzilla). Sub-groupings (MCU, DCEU, Dark Knight
  Trilogy, Raimi Trilogy, MonsterVerse, Wizarding World, Michael Bay era,
  actor runs, phase N, saga N) all move to `culturally_recognized_groups`.
  This kills the umbrella-vs-lineage flip-flop and makes "marvel movies"
  vs "MCU movies" a group-filter question rather than a naming question.
- Role definitions tightened: `starter` now requires that the franchise
  ENTITY did not exist before the film in any form. Books / comics / games
  do NOT block starter (so Jurassic Park 1993, LOTR 2001 are still
  starters); toys / products / TV / prior theatrical entries DO block it
  (so Barbie 2023, Transformers 2007, Super Mario 2023 are not).
  `mainline` is explicitly demoted to the fallback-of-last-resort — only
  assign it after ruling out all other roles, AND null-role with a
  populated `franchise_name` is a valid preferred output when no role
  cleanly fits (Iron Man 2008, Barbie 2023, Sonic 2020).
- `spinoff` redefined to cover minor character OR story element OR subplot
  focus. Creed is no longer a spinoff (new lead character not in prior
  Rocky films + Rocky is a major co-lead) — it's now mainline. Rogue One
  is a spinoff via the "Death Star plans" subplot hook, plus `is_prequel`.
- `remake` vs `reboot` split via story-spine-preservation: Total Recall
  2012 is a remake (same "artificial memories → spy awakening" spine);
  Jungle Book 2016 is a remake; Super Mario 2023 is a reboot.
- Two new orthogonal boolean fields: `is_prequel` (set chronologically
  before a prior-released entry in the same franchise; orthogonal to
  role so Rogue One can be spinoff+prequel and Episode I can be
  mainline+prequel) and `launches_subgroup` (first entry of a notable
  culturally-recognized subgroup, e.g. Iron Man 2008 → MCU, Spider-Man
  2002 → Raimi trilogy, Godzilla 2014 → MonsterVerse).
- New `crossover` role for Freddy vs Jason / AvP / Godzilla vs Kong.
  Dominant parent franchise goes in `franchise_name`; the secondary is
  recoverable via character search.
- Group label normalization is handled PROMPT-FIRST rather than via an
  ingest-time normalizer. Added explicit rules to the prompt: lowercase,
  digits spelled as words ("phase three" not "phase 3"), "&" expanded to
  "and", most-common canonical phrasing ("raimi trilogy" not "sam raimi
  trilogy"), never restate the franchise name ("marvel film series" /
  "godzilla films" forbidden). If the next test run shows the model
  can't hold these rules, escalate to a programmatic normalizer.
- FranchiseRole enum gains `CROSSOVER = 6`. FranchiseOutput gains
  `is_prequel_reasoning` + `is_prequel` and `launches_subgroup_reasoning`
  + `launches_subgroup` (reasoning-before-decision pattern matches the
  existing fields). All existing field descriptions updated to reference
  the top-level-entity rule and the null-role-valid carve-out.
- Test notebook SOT updated for ~27 entries: Creed → mainline; all
  MCU/DCEU/Batman/Spider-Man/X-Men/Logan films re-pointed at
  Marvel/DC Comics as franchise_name with cinematic-universe labels
  moved to groups; Iron Man → role=null + launches_subgroup=True;
  Super Mario 2023 / Transformers 2007 → reboot; Barbie / Sonic /
  Resident Evil / Scooby-Doo → null role; Sherlock Holmes 2009 →
  reboot; Detective Pikachu → reboot (name="Pokemon"); Rogue One,
  Solo, Fantastic Beasts, Phantom Menace → is_prequel=True;
  Minions → is_prequel=True; Batman Begins, Casino Royale,
  Spider-Man 2002, X-Men 2000, Rise of Apes, Godzilla 2014,
  Transformers 2007, Sherlock Holmes 2009, Phantom Menace →
  launches_subgroup=True; all group labels normalized to words.

## Overall keyword enum — source-backed LLM-facing definitions
Files: implementation/classes/overall_keywords.py

Why: `OverallKeyword` previously carried only the stable IMDb ID and
display label. For query understanding, the LLM also needs a compact,
unambiguous definition of what each IMDb overall keyword means so it
can map natural-language requests onto the right tag set without having
to infer the taxonomy from the label alone.

Approach: Extended `OverallKeyword` with a third stored string,
`definition`, and populated all 225 members with one-sentence
descriptions tailored for prompt use. The wording was gathered from a
web-research pass split across multiple subagents, using IMDb interest
pages where they were definitional and otherwise stable reference
sources such as Britannica, Merriam-Webster, and Wikipedia. After the
research pass, definitions were normalized into a consistent house
style: short, declarative, and focused on what the keyword represents
or when it should apply to a movie.

## Project franchise metadata into Postgres and add a franchise backfill path
Files: db/init/01_create_postgres_tables.sql, db/postgres.py, schemas/movie.py, movie_ingestion/final_ingestion/ingest_movie.py, unit_tests/test_postgres.py, unit_tests/test_schemas_movie.py, unit_tests/test_ingest_movie.py, docs/modules/db.md, docs/modules/ingestion.md, docs/modules/schemas.md, docs/TODO.md, search_improvement_planning/franchise_metadata_planning.md, search_improvement_planning/v2_data_architecture.md, search_improvement_planning/v2_data_needs.md, search_improvement_planning/current_search_flaws.md, search_improvement_planning/types_of_searches.md, search_improvement_planning/open_questions.md, search_improvement_planning/new_system_brainstorm.md, search_improvement_planning/source_material_type_enum.md, search_improvement_planning/concept_tags.md

Why: `FranchiseOutput` had been finalized in `schemas/metadata.py` and written
to tracker SQLite, but Stage 8 still ignored it and multiple planning/docs
surfaces still described the abandoned `franchise_membership` draft. The search
redesign now needs a queryable Postgres projection that stores full franchise
metadata per movie and a shared lexical posting table built from both `lineage`
and `shared_universe`, plus a safe way to backfill existing `movie_card` rows.

Approach: Added `public.movie_franchise_metadata` and
`lex.inv_franchise_postings` to the bootstrap SQL and introduced
`ensure_movie_franchise_schema()` in `db/postgres.py` to create the final schema
and clean up obsolete draft franchise tables/indexes only when the new table is
not already present. Added Postgres helpers to upsert/delete franchise metadata
rows and atomically replace per-movie franchise postings. Extended
`schemas/movie.py` so `Movie.from_tmdb_id()` / `from_tmdb_ids()` parse
`generated_metadata.franchise` into a new `franchise_metadata` field. In Stage 8
ingestion, added a dedicated `ingest_movie_franchise_metadata()` slice that
upserts franchise rows whenever metadata exists, deletes stale PG franchise
state when it no longer exists in tracker data, and indexes the deduped union of
`lineage` and `shared_universe`. Added a Postgres-only
`--backfill-franchise-metadata` CLI path that selects tracker rows with
franchise metadata but missing `movie_franchise_metadata`, batch-loads `Movie`
objects from SQLite, and reuses the normal Postgres ingestion path with vectors
disabled. Updated module docs, TODOs, and search planning docs so they now
describe `movie_franchise_metadata` rather than the retired role/group draft.

Design context: Driven by the franchise storage plan discussed in
`search_improvement_planning/` and the repo preference for queryable structured
storage over opaque blobs. The user explicitly decided to keep subgroup data
inline as `TEXT[]`, over-count subgroup launches rather than under-count, ingest
any movie with any franchise metadata signal (including `lineage = NULL`
remake-style rows), and search both `lineage` and `shared_universe` through one
lexical channel.

Testing notes: Added targeted unit coverage for the new Postgres helpers,
tracker-backed `Movie` loading, franchise Stage 8 ingestion behavior, and the
franchise backfill selector. Verified with:
`pytest unit_tests/test_postgres.py unit_tests/test_ingest_movie.py unit_tests/test_schemas_movie.py::TestTrackerFranchiseLoading -q`
which passed (`93 passed`). A broader combined run exposed pre-existing unrelated
drift in older `test_schemas_movie.py` expectations outside this franchise work,
so only the franchise-focused subset was treated as the validation target here.

## Document franchise lexical-resolution caveat
Files: search_improvement_planning/franchise_metadata_planning.md, search_improvement_planning/v2_data_architecture.md

Why: After implementing `lex.inv_franchise_postings`, we clarified that the
integer `term_id` is only the dictionary ID of a normalized franchise string.
That storage choice is fine, but exact-only query-time resolution would be too
brittle for franchise names because small LLM wording shifts can miss entirely.

Approach: Updated the planning docs to state explicitly that `lineage` and
`shared_universe` should resolve fuzzily against stored franchise strings
before posting lookup, and that `recognized_subgroups` should remain out of
`lex.inv_franchise_postings` because subgroup labels have much higher string
variability and are better matched only after franchise lookup narrows the
candidate set.

## Remove runtime franchise schema rollout and backfill path
Files: db/postgres.py, movie_ingestion/final_ingestion/ingest_movie.py, unit_tests/test_postgres.py, unit_tests/test_ingest_movie.py, docs/modules/db.md, docs/modules/ingestion.md

Why: After reviewing the workflow, the user confirmed they plan to wipe
Postgres and rebuild from scratch rather than migrate an existing database.
That makes the runtime `ensure_movie_franchise_schema()` helper and the
Postgres-only `--backfill-franchise-metadata` command unnecessary complexity.

Approach: Removed `ensure_movie_franchise_schema()` from `db/postgres.py`,
deleted the franchise backfill selector/CLI path from Stage 8 ingestion, and
stopped calling franchise schema rollout from normal ingest code. Stage 8 now
assumes the database was initialized from `db/init/01_create_postgres_tables.sql`
before ingestion starts. Updated the affected docs and removed the now-obsolete
unit tests that only existed for rollout/backfill behavior.

Testing notes: Re-ran the focused franchise suite after the removal:
`pytest unit_tests/test_postgres.py unit_tests/test_ingest_movie.py unit_tests/test_schemas_movie.py::TestTrackerFranchiseLoading -q`
and it passed (`91 passed`).

## Put franchise lineage-position IDs on the enum itself
Files: schemas/enums.py, movie_ingestion/final_ingestion/ingest_movie.py, unit_tests/test_postgres.py, unit_tests/test_ingest_movie.py

Why: The franchise ingest path had a local `_LINEAGE_POSITION_IDS` dict to
convert `LineagePosition` enum members into Postgres SMALLINT values. The repo
already uses the stronger pattern of storing stable IDs directly on enums
(`AwardOutcome`, `AwardCeremony`, `SourceMaterialType`), and the user asked to
make `LineagePosition` consistent with that design.

Approach: Extended `LineagePosition` to carry `lineage_position_id` while
keeping its string enum values unchanged for Pydantic/LLM structured output.
Removed the local mapping from Stage 8 ingestion and now read the Postgres ID
via `franchise_metadata.lineage_position.lineage_position_id`. Updated the
franchise helper tests to assert through the enum attribute rather than magic
integers.

Testing notes: Re-ran the focused franchise suite:
`pytest unit_tests/test_postgres.py unit_tests/test_ingest_movie.py unit_tests/test_schemas_movie.py::TestTrackerFranchiseLoading -q`
and it passed (`91 passed`).

## Move franchise lineage-position conversion into the Postgres helper
Files: db/postgres.py, movie_ingestion/final_ingestion/ingest_movie.py, unit_tests/test_postgres.py, unit_tests/test_ingest_movie.py

Why: `ingest_movie_franchise_metadata()` was extracting
`lineage_position.lineage_position_id` even though the Postgres helper already
received the full `FranchiseOutput` object and was the only place that needed
the integer. That made the ingest layer carry a storage-specific conversion it
didn't otherwise use.

Approach: Simplified `upsert_movie_franchise_metadata()` so it computes the
SMALLINT value internally from `franchise_metadata.lineage_position`. The
ingest layer now just passes `movie_id`, `franchise_metadata`, and `conn`,
which keeps the DB-shaping logic inside `db/postgres.py` and makes the call
site cleaner.

Testing notes: Re-ran the focused franchise suite:
`pytest unit_tests/test_postgres.py unit_tests/test_ingest_movie.py unit_tests/test_schemas_movie.py::TestTrackerFranchiseLoading -q`
and it passed (`91 passed`).

Design context: This change lives in `implementation/classes/` because
the enum is already the canonical local catalog for IMDb's curated
overall-keyword taxonomy. The added field is intended for downstream
LLM prompts rather than end-user UI copy, so clarity and disambiguation
were prioritized over marketing language.

Testing notes: `uv run python -m py_compile implementation/classes/overall_keywords.py`
passes, and importing the enum confirms there are still 225 members and
that `definition` is available on each one. No tests were run per the
repo's default test-boundary rules.

## Add production_techniques metadata type
Files: schemas/enums.py, schemas/metadata.py, movie_ingestion/tracker.py, movie_ingestion/metadata_generation/inputs.py, movie_ingestion/metadata_generation/batch_generation/pre_consolidation.py, movie_ingestion/metadata_generation/batch_generation/generator_registry.py, movie_ingestion/metadata_generation/batch_generation/result_processor.py, movie_ingestion/metadata_generation/prompts/production_techniques.py, movie_ingestion/metadata_generation/generators/production_techniques.py, unit_tests/test_production_techniques_generator.py, unit_tests/test_pre_consolidation.py, unit_tests/test_generator_registry.py, unit_tests/test_result_processor.py, unit_tests/test_tracker.py, unit_tests/test_enums.py, unit_tests/test_metadata_inputs.py, unit_tests/test_metadata_type_consistency.py

Why: Introduce a new narrowed production metadata type alongside legacy
`production_keywords`, matching the V2 search-system direction without
changing any embedding or search-time code yet.

Approach: Added `MetadataType.PRODUCTION_TECHNIQUES`, a new
`ProductionTechniquesOutput` schema, a dedicated prompt and generator,
and tracker DB support for result, eligibility, and batch-id columns.
The new generator takes `title`, `overall_keywords`, and `plot_keywords`
as separate prompt inputs and uses the locked `gpt-5-mini` + low
reasoning config. Eligibility is intentionally narrower than legacy
`production_keywords`: eligible when `plot_keywords` is non-empty or
`overall_keywords` has at least 3 entries. The batch pipeline remained
generic; wiring was limited to the enum/registry/result-processor/tracker
surfaces needed for the new type to flow through `eligibility`, `submit`,
`process`, and `autopilot`.

Design context: This is additive only. Legacy `production_keywords`
remains untouched for backwards compatibility, and filming locations stay
out of this new metadata type for now. Search-time readers, vector text,
and `schemas/movie.py` were deliberately not updated in this pass.

Testing notes: Focused validation ran through the `uv` environment to pick
up repo-managed dependencies. Verified generator behavior, eligibility,
registry wiring, result storage, tracker schema, enum consistency, and
metadata-type/db-column consistency:
`uv run python -m pytest unit_tests/test_production_techniques_generator.py unit_tests/test_pre_consolidation.py unit_tests/test_generator_registry.py unit_tests/test_result_processor.py unit_tests/test_tracker.py unit_tests/test_enums.py unit_tests/test_metadata_type_consistency.py -q`
  New counts: 57 franchise / 19 standalone / 5 null-role-with-name
  / 5 prequels / 10 subgroup-launchers.
- Test notebook evaluation helpers (cell 8) extended with two new
  RunResult values (`PREQUEL_WRONG`, `LAUNCHES_WRONG`), new
  check_null_pairing semantics (name-without-role no longer a
  violation; is_prequel=True without name IS a violation), and new
  classify_run signature with optional `expected_is_prequel` /
  `expected_launches_subgroup` defaulted to False. Cells 9-13 updated
  to carry the new expected fields through per-candidate summary,
  per-bucket breakdown, inspector, majority-vote, and null-pairing
  report.

### Planning Context
Decision trace lives in /Users/michaelkeohane/.claude/plans/elegant-meandering-sundae.md
and in search_improvement_planning/franchise_test_iterations.md. The locked
decisions listed above were all validated against specific failure cases
from the prior run (Creed, Barbie, Iron Man, Super Mario 2023, Total Recall
2012, Jungle Book 2016, Rogue One, Detective Pikachu, Fantastic Beasts,
Godzilla 2014) before the prompt was rewritten.

### Testing Notes
End-to-end: re-run test_franchise.ipynb cells 3→6 to regenerate
franchise_test_results.json under the new prompt/schema, then 9/10/11/12/13
for the updated metrics. Spot-check: Creed majority → mainline; Iron Man
majority → franchise=Marvel, role=null, launches_subgroup=True; Super Mario
2023 → reboot; Barbie → null role; Fantastic Beasts → spinoff+prequel with
franchise=Harry Potter; Rogue One is_prequel=True; any MCU film groups use
"phase one/two/three" in words; Spider-Man 2002 groups use "raimi trilogy".
Read 2–3 franchise_name_reasoning traces to confirm the top-level rule is
actually being applied. If normalization inconsistencies persist (digit
phase labels etc.) escalate to a shared ingest-time normalizer applied at
both SOT-load and generation-write time.

Known gaps left for follow-up: no crossover movies in the test set (new
`crossover` role is untested); no documentary entries; Jurassic Park is
present but LOTR / Harry Potter 1 are not, so the "books don't block
starter" rule is only lightly tested.

## Add per-field evidence→analysis reasoning to FranchiseOutput
Files: schemas/metadata.py, movie_ingestion/metadata_generation/prompts/franchise.py
Why: Franchise classification is a multi-step judgment (identify the IP, classify the role, enumerate established sub-groupings) and the model was committing to answers without visibly weighing the signals. Per-field CoT scaffolding has been effective elsewhere in the metadata schema (CharacterArcWithReasoning, ElevatorPitchWithJustification) so adopt the same pattern here.
Approach: Added three required reasoning fields to `FranchiseOutput` — `franchise_name_reasoning`, `franchise_role_reasoning`, `culturally_recognized_groups_reasoning` — each placed immediately BEFORE its decision field so OpenAI structured-output field order forces the model to produce the reasoning first. Each reasoning field's description mandates a two-step process: Step 1 gather evidence (list relevant signals), Step 2 analyze that evidence and commit to the decision. For the role and groups fields, when `franchise_name` is null the reasoning should be "N/A — standalone" (kept required rather than optional to avoid an extra conditional branch in the schema). Rewrote the prompt's FIELDS section to mirror the schema: each numbered field now opens with an explicit Step 1 / Step 2 breakdown of what evidence to gather and how to analyze it, with a leading paragraph establishing the "evidence → analysis → decision, never skip straight to the answer" rule.
Design context: Pattern matches existing `*WithReasoning` / `*WithJustification` sub-models in schemas/metadata.py — reasoning fields are generated first, are never embedded (franchise output is non-embeddable anyway), and exist purely to scaffold higher-quality labels. Chose required strings over Optional so the model cannot silently skip the reasoning step.
Testing notes: Re-run franchise generation on the existing test set (movie_ingestion/metadata_generation/generators/test_franchise.ipynb, franchise_test_results.json) and spot-check that (a) the reasoning fields show genuine evidence enumeration rather than restating the decision, (b) franchise_name / role / groups accuracy is equal or better than the prior run, (c) the "N/A — standalone" convention is followed when franchise_name is null. Also verify the batch pipeline still parses FranchiseOutput cleanly — extra_forbid is on so any drift in the response schema will fail loudly.

## Fix female_lead over-inclusion via top_billed_cast + stricter reasoning flow
Files: schemas/movie_input.py, movie_ingestion/metadata_generation/generators/concept_tags.py, movie_ingestion/metadata_generation/prompts/concept_tags.py
Why: `female_lead` was firing on nearly every movie. Root cause was twofold: (1) the concept_tags prompt has a global "when debatable, include it — a missing tag is worse than an extra tag" bias that conflicts with a binary-gender question that should default to no, and (2) the LLM had no reliable prominence signal — it had to infer protagonist gender from plot_summary alone, which skews toward any named female character regardless of actual lead status.
Approach:
  - Added `actors: list[str]` to MovieInputData and fetched `i.actors` in `load_movie_input_data`. The column already exists in `imdb_data`.
  - Added `MovieInputData.top_billed_cast(n=5)` which pairs actors[:n] with characters[:n] and renders "Character (Actor), ..." in billing order. Pairing is positional — mostly correct, may drift slightly past the top billed few for rare multi-role actors, which is acceptable for a prominence signal.
  - `build_concept_tags_user_prompt` now passes `top_billed_cast` as a new labeled input (fallback "not available" when no actors).
  - Rewrote the FEMALE_LEAD tag definition as an explicit 3-step reasoning flow: (1) does the story have a single core protagonist at all? If not (two-hander, ensemble, trio), do not tag; (2) if yes, name them — use plot_summary as primary, cross-reference top_billed_cast for prominence, treat top-billed-man + no clearly centered woman as strong negative signal; (3) is that single core character female? Tag only if yes with high confidence.
  - Added an explicit override: "The 'when debatable, include it' rule does NOT apply to FEMALE_LEAD — a false positive here is worse than a miss." Also made two-handers ineligible (the previous version tagged two-handers if one co-lead was female, which was a major over-inclusion vector).
  - Added a new INPUTS entry for top_billed_cast explaining it's a cross-reference prominence signal, not a determinant.
  - Updated the module docstring failure-mode note to describe the new FEMALE_LEAD reasoning flow.
Design context: Chose n=5 (not n=3) for top_billed_cast so the model can see the drop-off past position 3 — that drop-off is what distinguishes a genuine ensemble from "single lead + two notable supports." 5 also amortizes across future ENSEMBLE_CAST tightening.
Testing notes: Need to re-run concept_tags over the validation set to confirm female_lead recall drops to a sane rate without collapsing legitimate female-lead classifications (e.g. Erin Brockovich, Lady Bird, Promising Young Woman). Watch for two-handers (e.g. Thelma & Louise) which are now deliberately excluded — if the user wants those tagged, the rule needs to loosen back up.

## Collapse franchise prompt title/year into single title_with_year line
Files: movie_ingestion/metadata_generation/generators/franchise.py, movie_ingestion/metadata_generation/prompts/franchise.py | Pass `movie.title_with_year()` as a single `title_with_year` field in the user prompt instead of separate `title` and `release_year` lines; updated the INPUTS section of the system prompt to match.

## Add concept_tags_run_2 column to generated_metadata
Files: movie_ingestion/tracker.py | Added migration `ALTER TABLE generated_metadata ADD COLUMN concept_tags_run_2 TEXT` to store results of a second concept tags generation run; unioning both runs improves recall. Not written to yet. Ran `init_db()` to apply.

## Remove reasoning fields from ConceptTagsOutput
Files: schemas/metadata.py, movie_ingestion/metadata_generation/prompts/concept_tags.py | Dropped the `reasoning` field from all 7 per-category assessment classes and reworded the system prompt (module docstring, endings "HOW TO THINK THROUGH" block, OUTPUT FORMAT) so it still instructs the model how to evaluate each tag internally without claiming there is a reasoning field to emit. Empirically produced better tag quality on gpt-5-mini in test_concept_tags.ipynb runs; plan is to generate twice and union the results per run to improve recall. Stale field descriptions on `ConceptTagsOutput.narrative_structure` updated to drop the "evaluate each tag" phrasing.

## Capture per-run token usage in concept tags test notebook
Files: movie_ingestion/metadata_generation/generators/test_concept_tags.ipynb | Generation loop now stores the `TokenUsage` returned by `generate_concept_tags` in a new `all_usage` dict, and the save-to-JSON cell persists it as a `"usage"` list aligned positionally with `"runs"` for cost analysis and candidate comparison.

## ConceptTagsOutput schema refinements
Files: schemas/metadata.py
Why: Endings field allowed multiple tags but the prompt and spec define it as zero-or-one. Field descriptions were stale (listing enum values redundantly).
Approach: Changed `endings` to `conlist(EndingEvidence, max_length=1)` enforcing the single-tag constraint at the schema level. Rewrote all 7 category descriptions to describe the category's purpose rather than listing enum members, since the enum already documents the options.

## Concept tag generation pipeline (new metadata type)

Files: schemas/enums.py, schemas/metadata.py, movie_ingestion/metadata_generation/inputs.py, movie_ingestion/metadata_generation/batch_generation/pre_consolidation.py, movie_ingestion/metadata_generation/prompts/concept_tags.py (new), movie_ingestion/metadata_generation/generators/concept_tags.py (new), movie_ingestion/metadata_generation/batch_generation/generator_registry.py, movie_ingestion/metadata_generation/batch_generation/result_processor.py, movie_ingestion/tracker.py

### Intent
Adds CONCEPT_TAGS as a new Wave 2 metadata generation type. Classifies 23 binary concept tags across 7 categories (narrative structure, plot archetypes, settings, characters, endings, experiential, content flags) via LLM multi-label classification. Tags enable deterministic Phase 1 search retrieval via Postgres INT[] array containment queries.

### Key Decisions
- **7 per-category enums replace single ConceptTag enum** (`NarrativeStructureTag`, `PlotArchetypeTag`, `SettingTag`, `CharacterTag`, `EndingTag`, `ExperientialTag`, `ContentFlagTag`). JSON schema self-enforces category membership — eliminates runtime model_validator. `ALL_CONCEPT_TAGS` flat tuple for codebase consumers.
- **7 per-category evidence classes** (`NarrativeStructureEvidence`, etc.) each with typed tag field. Replaces single `TagEvidence` class.
- **`validate_and_fix(content)` classmethod on `EmbeddableOutput`** — single entry point for batch result processor to validate raw LLM JSON and apply deterministic fixups. Default: pure validation. `ConceptTagsOutput` (inherits `BaseModel`, not `EmbeddableOutput` — first non-embeddable output) overrides with its own `validate_and_fix` that calls `apply_deterministic_fixups()` after validation. Result processor calls `schema_class.validate_and_fix()` uniformly for all types.
- **Category-level arrays with evidence-before-tag ordering** chosen over boolean grid (false-negative bias) and flat array (category-skipping). Evidence field forces chain-of-thought before tag commitment.
- **Post-generation fixup**: TWIST_VILLAIN implies PLOT_TWIST — handled deterministically via `ConceptTagsOutput.apply_deterministic_fixups()` instance method, called by `validate_and_fix()` override and the live generator. Fixup logic lives on the schema class, not in generators or processors.
- **Prompt section ordering**: task → inputs → tag definitions → evidence discipline → output. Evidence discipline placed after tag definitions for recency advantage during generation.
- **Parametric knowledge reframed** as "high-confidence fallback" (95%+ confidence, culturally unambiguous) instead of "tiebreaker". Input evidence trusted over parametric knowledge on conflict.
- **No tag count anchoring** — removed "2-6 tags total" from prompt. Classification is purely evidence-based.
- **Single prompt path** — no separate keyword-only prompt. Investigated the 20.7K keyword-only-eligible movies and found 99.7% have emotional_observations, making a distinct prompt unnecessary.
- **Eligibility gate**: plot_summary exists OR best_plot_fallback >= 250 chars OR plot_keywords >= 3.
- **Six LLM inputs** (~310-1140 tokens): title_with_year, plot_keywords, plot_summary/plot_text (quality-tiered), emotional_observations, narrative_technique_terms (6 of 9 sections, terms only), plot_analysis fields (arc labels + conflict_type).

### Planning Context
See search_improvement_planning/concept_tags.md for full tag definitions and design rationale. See search_improvement_planning/concept_tags_notes.md for prompt evaluation session notes.

### Testing Notes
- Schema validation: verify tags in wrong categories are rejected by typed enum constraints (no model_validator needed)
- Fixup: verify TWIST_VILLAIN without PLOT_TWIST inserts PLOT_TWIST; no duplicate when both present; no-op when neither present
- Eligibility: test all 3 eligible paths + skip case
- Prompt builder: test with all-present, all-absent, and partial inputs
- End-to-end: run eligibility evaluation, then live generation on small batch to inspect output quality

## Documentation staleness audit fixes
Files: schemas/metadata.py, docs/modules/ingestion.md, docs/modules/schemas.md, docs/modules/llms.md, docs/conventions.md, docs/decisions/ADR-065-schema-docstrings-as-comments-not-python-docstrings.md, docs/decisions/ADR-058-vector-text-formatting-conventions.md, movie_ingestion/metadata_generation/batch_generation/pre_consolidation.py

### Intent
Full docs-auditor pass found 15 stale references — all traced to the concept_tags type being added without a doc sweep. Also fixed two pre-existing convention/ADR issues.

### Key Changes
- **schemas/metadata.py**: Converted `TagEvidence` and `ConceptTagsOutput` class docstrings to `#` comment blocks per ADR-065 (these were leaking into LLM JSON schema payload)
- **ingestion.md**: Updated all "8 generators"→10, "9 batch_id columns"→10, "9 JSON result columns"→10, Wave 2 eligibility count 6→7
- **schemas.md**: MetadataType count 9→10, added `ConceptTagsOutput`/`ConceptTag`/`TagEvidence`/`CONCEPT_TAG_CATEGORIES` to Key Types, updated boundary list with `extract_narrative_technique_terms` and `load_plot_analysis_output`
- **conventions.md**: Added missing `embedded` status to pipeline chain; updated `__str__()` normalization rule to reference `embedding_text()` per ADR-057
- **ADR-065**: Updated reference count from "8 `*Output` classes" to include 9 EmbeddableOutput subclasses + ConceptTagsOutput/TagEvidence
- **ADR-058**: Struck stale open consequence about unwired fallback (now integrated)
- **llms.md**: Documented `verbosity` kwarg; updated TokenUsage import count 8→10
- **pre_consolidation.py**: Fixed docstring count from "Eight" to "Nine" eligibility methods

## Concept tags deterministic test suite

Files: movie_ingestion/metadata_generation/generators/concept_tags.py (modified), movie_ingestion/metadata_generation/generators/test_concept_tags.ipynb (new)

### Intent
Built a notebook-based evaluation harness for concept tag generation with 38 test movies across 4 buckets (core coverage, dense/messy, sparse, targeted challenges). Tests multiple LLM candidates (OpenAI gpt-5-mini, Gemini flash-lite) with 3 runs each for consistency measurement.

### Key Changes
- **Generator signature extended**: `generate_concept_tags()` now accepts keyword-only `provider`, `model`, and `llm_kwargs` overrides. Defaults unchanged — production callers are unaffected.
- **Test notebook**: 38 movies with expected tag sets, 4 LLM candidates, sequential generation loop with error capture, JSON serialization of full ConceptTagsOutput per run.
- **Notebook import fix**: Kernel cwd is the notebook's directory (not project root), and a third-party `schemas` namespace package in site-packages shadows the local `schemas/` package. Fix: `sys.path.insert(0, project_root)` + `importlib.invalidate_caches()` + purge stale `sys.modules["schemas*"]` entries.
- **Gemini model version**: Changed from `gemini-2.5-flash-lite-preview-06-17` to `gemini-3.1-flash-lite-preview`.

## Concept tags expected_tags overhaul from evaluation analysis

Files: movie_ingestion/metadata_generation/generators/test_concept_tags.ipynb (cell 2)

### Intent
Comprehensive review of all 38 test movies against gpt-5-mini candidate outputs and user prompts (input evidence). Updated expected_tags to reflect what the model should actually produce given its inputs.

### Key Changes
- **2 removals**: The Skin I Live In and Light of My Life both lost `sad_ending` (protagonist escapes/input lacks ending evidence respectively)
- **51 additions across 27 movies**: Most common additions were `anti_hero` (11 movies), `happy_ending` (11), `plot_twist` (7), `feel_good` (6), `revenge` (5)
- **Oldboy**: did NOT add `kidnapping` despite 6/9 candidate consensus — user clarified imprisonment ≠ kidnapping
- **Machete**: did NOT add `plot_twist` despite 5/9 — user clarified early betrayal is plot setup, not a recontextualizing twist
- **Tags NOT added despite high consensus**: Alien `happy_ending` (survival ≠ happy), Ferris Bueller `anti_hero` (trickster ≠ morally ambiguous), Taken `revenge` (rescue mission not vengeance), Mad Max `anti_hero` (reluctant ≠ morally ambiguous), Little Miss Sunshine `female_protagonist` (contradicts ensemble_cast)
- Total expected tag instances: 85 → 138. All 23/23 tags still represented.

## New tags + definition refinements from eval analysis

Files: schemas/enums.py, schemas/metadata.py, movie_ingestion/metadata_generation/prompts/concept_tags.py, movie_ingestion/metadata_generation/generators/concept_tags.py, movie_ingestion/metadata_generation/generators/test_concept_tags.ipynb

### Intent
Added 2 new concept tags and refined 5 existing definitions based on systematic model misclassification patterns from the gpt-5-mini evaluation.

### New Tags (23 → 25)
- **`cliffhanger_ending`** (NarrativeStructureTag, ID 9): Story ends with central conflict clearly unresolved and setup for continuation. Distinct from `open_ending` (artistic ambiguity vs unfinished story).
- **`bittersweet_ending`** (EndingTag, ID 43): Resolution contains significant positive AND negative elements. Fills the gap between `happy_ending` and `sad_ending` that was causing forced misclassification.

### Definition Refinements
- **`anti_hero`**: Added "tension or discomfort" threshold — reluctant heroes, flawed-but-good characters, and fully-redeeming protagonists don't qualify. Addresses over-application to any protagonist who does illegal things.
- **`happy_ending`**: Added bittersweet boundary — survival alone isn't sufficient, success at significant cost is bittersweet. Addresses tagging horror survival movies as happy.
- **`sad_ending`**: Added bittersweet boundary — loss tempered by hope/meaning is bittersweet.
- **`plot_twist`**: Added "must change understanding of events already shown" — early betrayals/deceptions before audience has formed expectations are plot setups, not twists.
- **`feel_good`**: Added subject matter compatibility requirement — cathartic satisfaction from violent/dark material is not feel_good.
- **`open_ending`**: Narrowed to artistic ambiguity with complete narrative arc. Sequel cliffhangers directed to `cliffhanger_ending`.

### Test Set Changes
- Terrifier 3: `open_ending` → `cliffhanger_ending`
- Schindler's List: `sad_ending` → `bittersweet_ending`
- Kill Bill Vol. 1: added `cliffhanger_ending` (daughter alive reveal), kept `anti_hero`
- Added Star Wars (1977): `happy_ending`, `feel_good`, `underdog` — resolved central conflict, NOT cliffhanger
- Added Empire Strikes Back (1980): `plot_twist`, `cliffhanger_ending`, `sad_ending` — opposite emotional valence
- Total: 38 → 40 movies, 25/25 tags represented, 145 expected tag instances

## Franchise metadata generation — full planning doc
Files: search_improvement_planning/franchise_metadata_planning.md (new), search_improvement_planning/new_system_brainstorm.md, search_improvement_planning/v2_data_needs.md, search_improvement_planning/v2_data_architecture.md, search_improvement_planning/open_questions.md

### Intent
Created comprehensive planning document for franchise metadata generation and updated all related V2 planning docs with finalized design decisions.

### Key Decisions
- **Franchise = real-world IP, not just film series:** Any recognizable IP or brand from any medium (video games, toys, books, comics, TV, etc.). Franchise name is the IP name (e.g., "mario"), not the film series name.
- **Drop `franchise_name` column, keep only `franchise_name_normalized`:** No display-name column. Same for `culturally_recognized_group` — stored normalized.
- **Finalized LLM input set (7 fields):** title, release_year, overview (identification aid only), TMDB collection_name, production_companies, overall_keywords, characters. No generated-metadata dependencies.
- **`culturally_recognized_group` is globally scoped:** Any market, not just US. American-market term takes precedence in rare conflicts.
- **No eligibility gate — all movies eligible:** Franchise generation is identification/classification (not content analysis), so sparse input causes conservative nulls rather than hallucination. False negatives are worse than false positives.
- **Wave-independent:** No dependency on Wave 1 or Wave 2 outputs. Can run in parallel with or independently from existing generation waves.

### Planning Context
See search_improvement_planning/franchise_metadata_planning.md for full specification covering output fields, LLM inputs with justifications, eligibility rationale, prompt strategy, pipeline placement, storage schema, and downstream search usage.

## Concept tags: 4-tag prompt refinement from gpt-5-mini eval analysis
Files: movie_ingestion/metadata_generation/prompts/concept_tags.py, schemas/metadata.py

### Intent
Targeted fixes for 4 tags identified via 40-movie x 3-run evaluation (120 classifications, 35% exact-match rate). Addresses systematic false positives in underdog (53.6% precision), bittersweet_ending (14.3%), ensemble_cast (55.6%), and systematic false negatives in sad_ending when cliffhanger is present (Terrifier 3 0/3, Empire Strikes Back 1/3).

### Key Changes
- **underdog**: Replaced definition with official ("expected to lose or fail due to lack of resources, talent, or status"). Added emotional engine test ("audience roots for them BECAUSE they are disadvantaged"). Explicitly demoted conflict_type as evidence source via NOTE. Added generic NOT examples (skilled grifters, structurally weaker faction, lone dissenter in debate). Removed conflict_type from Check line.
- **ensemble_cast**: Adopted official definition (multiple main characters share roughly equal importance, screen time, and storyline focus). Added group/event-as-protagonist framing. Added removal test ("if removing one character's arc collapses the film, that character is the lead"). Added SIGNAL CHECK warning about long plot_summaries with many named characters.
- **bittersweet_ending**: Added official definition anchor ("protagonist achieves main goal but suffers significant, concrete loss or cost"). Added NOT boundary: "structural ambiguity (open/ambiguous ending) is NOT emotional ambiguity — narrative uncertainty is not a loss."
- **endings section**: Reframed category header from "how the resolution feels" to "how the audience FEELS when the credits roll." Replaced "resolution" with "film's final moments" / "ending" throughout all 3 tag definitions. Added "Not all movies have resolutions, but all movies have endings." Added cliffhanger+sad example to SAD_ENDING. Changed 3-step to 4-step decision process: Steps 2-3 evaluate happy/sad with critical evidence assessment (not case-building), Step 4 evaluates bittersweet only as last resort with explicit "select NONE" for ambiguous/insufficient evidence. Step 2 explicitly instructs model to dismiss weak options upfront and to treat emotional_observations as primary over plot event inference.
- **EndingAssessment schema**: Updated reasoning description to reference 4-step process and "how the audience feels when credits roll." Updated tags description to frame empty list as correct when evidence is ambiguous.
- **ConceptTagsOutput.endings field**: Updated description to frame as audience emotion independent of structural tags.

## Endings decision process: sequential → parallel → evidence distillation
Files: movie_ingestion/metadata_generation/prompts/concept_tags.py, schemas/metadata.py, schemas/enums.py

### Intent
Post-evaluation analysis of 2-phase parallel approach showed bittersweet_ending at 16.7% precision (2 TP, 10 FP). Two root causes: (1) multi-way competition frame made NONE the hardest option to select — it needed to "win" against affirmative options rather than being the default; (2) schema used `conlist(EndingTag, max_length=1)` where empty list `[]` created completion pressure to fill the field after writing reasoning.

### Key Changes
- **Decision process → evidence distillation**: Replaced 2-phase argue-for-each-option approach with 3-step factual extraction: (1) extract ending-specific emotional_observations, filtering out journey-level emotions; (2) summarize final state of affairs (factual, no interpretation); (3) note ending-related plot_keywords. Tag selection follows from extracted evidence rather than per-option argumentation.
- **NO_CLEAR_CHOICE enum value**: Added to `EndingTag` with `concept_tag_id=-1`. Explicit "none of the above" option that the model selects as an affirmative classification rather than leaving a list empty.
- **EndingAssessment schema**: Changed `tags: conlist(EndingTag, max_length=1)` → `tag: EndingTag` (single required field). Eliminates empty-list completion pressure — choosing NO_CLEAR_CHOICE feels identical to choosing any other tag.
- **ALL_CONCEPT_TAGS**: Filters out tags with `concept_tag_id < 0` so NO_CLEAR_CHOICE never reaches storage or search.
- **all_concept_tag_ids()**: Updated to handle single `endings.tag` field and skip classification-only values.
- **_deduplicate_tags()**: Skips endings (no longer a list).
- **Bittersweet NOT list**: Added "losses that occur mid-film but not at the ending do not make an ending bittersweet" — addresses failure mode 1 (mid-film losses treated as ending evidence).
- **Removed per-tag Evidence lines**: Tag definitions now rely on the shared reasoning steps for evidence sourcing rather than repeating evidence instructions per tag.

## Franchise generator v2 follow-ups — SOT consistency + richer inputs
Files: schemas/movie_input.py, movie_ingestion/metadata_generation/generators/franchise.py, movie_ingestion/metadata_generation/prompts/franchise.py, movie_ingestion/metadata_generation/generators/test_franchise.ipynb, search_improvement_planning/franchise_test_iterations.md

### Intent
Post-v2 verification surfaced (a) three SOT rows that violated the locked Marvel-predates-starter rule or had `launches_subgroup=True` paired with an empty `culturally_recognized_groups` list (logically inconsistent under the v2 definitions), and (b) three cheap, high-signal inputs the franchise user prompt wasn't passing. This entry captures both fixes so the next full-run numbers are trustworthy.

### Key Changes
- **SOT — Spider-Man (2002) & X-Men (2000)**: `expected_role: "starter" → None`. Marvel is the top-level brand and predates both films, which blocks starter per the v2 definition. Both now match the Iron Man (1726) pattern: `franchise_name="Marvel"`, `role=None`, `launches_subgroup=True`, `groups=["raimi trilogy"] / ["x-men"]`. Null-role-with-name count rises from 5 to 7.
- **SOT — launches_subgroup ↔ groups invariant**: Every SOT row with `launches_subgroup=True` now carries a label in `expected_groups`. Filled in: Rise of the Planet of the Apes 2011 → `["caesar trilogy"]`, Transformers 2007 → `["bay era"]`, Sherlock Holmes 2009 → `["ritchie sherlock holmes films"]`. Verified no remaining `launches=True & groups=[]` rows.
- **MovieInputData.directors**: Added `directors: list[str]` field. `load_movie_input_data` now SELECTs and parses `i.directors` from the tracker's `imdb_data` table (already populated by the IMDB scraper). No migration needed.
- **Franchise user prompt — new inputs**: `build_franchise_user_prompt` now passes `release_year` as a separate labeled line, `directors` (explicitly labeled), and `top_billed_cast(5)` (actor+character pairings) alongside the existing `characters[:5]`. Release year supports chronology reasoning for `is_prequel`; directors + cast pairings are high-signal for director-era / actor-run subgroup detection ("raimi trilogy", "bay era", "daniel craig era").
- **Prompt INPUTS section**: Extended to document the three new fields and what signal they carry, so the model explicitly knows to lean on them.
- **franchise_test_iterations.md**: Added action-item #8 to check in the next run how often `launches_subgroup=True` co-occurs with an empty `culturally_recognized_groups`. If frequent, we'll tighten the prompt to require the launching label be present in the groups list.

### Verification
- Schema import + prompt build smoke test run with a populated Spider-Man (2002) fixture and a sparse fixture — both render cleanly, missing fields show `"not available"`.
- Cell 3 reparse: 76 entries, null-role-with-name count 7, zero `launches=True & groups=[]` rows.
- No consumer outside the generator + test notebook reads `movie_input.directors` yet, so the schema change is isolated.

## remove stale implementation/ files (chromadb-era dead code)
Files: implementation/vectorize.py, implementation/searching/, implementation/visualize.py, implementation/chroma_db/, AGENTS.md, docs/modules/ingestion.md, docs/modules/classes.md, docs/PROJECT.md, .claude/commands/analyze-metadata-inputs.md, search_improvement_planning/v2_data_needs.md, memory/project_embedding_model_decision.md
Why: User pruned implementation/scraping, implementation/generated_data, implementation/notebooks. Verified vectorize.py/searching/visualize.py/chroma_db/ were also stale (chromadb imports, BaseMovie deps, zero external importers; live replacements live at movie_ingestion/final_ingestion/vector_text.py and movie_ingestion/final_ingestion/ingest_movie.py) and removed them.
Approach: (1) confirmed no external importers via grep; (2) confirmed unit_tests/test_ingest_movie.py already has defensive ModuleNotFoundError fallback stubs for implementation.vectorize so deletion is safe; (3) test_vector_text.py imports from movie_ingestion.final_ingestion.vector_text, not the deleted module; (4) deleted files; (5) rewrote stale doc references to point at the live movie_ingestion paths. Also saved embedding-model decision (upgrade to text-embedding-3-large, Voyage-3-large as fallback) to search_improvement_planning/v2_data_needs.md #12 and updated the stale memory record that previously said "stay with 3-small".
Testing notes: test_ingest_movie.py fallback path is now the only path (module no longer importable). Worth a pytest run to confirm. ADR-060 still references implementation/vectorize.py as a historical note — left intact since ADRs are point-in-time decision records.

## upgrade embedding model to text-embedding-3-large
Files: implementation/llms/generic_methods.py, movie_ingestion/final_ingestion/ingest_movie.py, movie_ingestion/final_ingestion/vector_text.py, db/vector_search.py, db/init/02_qdrant_init.sh
Why: Per the 2026-04-10 decision (project memory / search_improvement_planning/v2_data_needs.md #12), upgrade from text-embedding-3-small (1536 dims) to text-embedding-3-large as part of the structured-label re-embed. Same provider/SDK/batch API, modest MTEB lift, trivial cost delta.
Approach: Code-only flip — user will handle the Qdrant wipe + re-ingest separately. Both ingestion and search already route through a single shared helper (generate_vector_embedding in implementation/llms/generic_methods.py), so changing the default model + the EMBEDDING_MODEL constant in ingest_movie.py covers every call site automatically. Chose native 3072 dims over Matryoshka truncation because the user is re-embedding everything anyway, so there is no infra-compat reason to sacrifice quality. Updated Qdrant init script (all 8 named vectors: anchor, plot_events, plot_analysis, viewer_experience, watch_context, narrative_techniques, production, reception) from size 1536 → 3072. Swapped tiktoken.encoding_for_model to "text-embedding-3-large" (both models share cl100k_base and the 8191 limit, so no functional change but keeps the string accurate). Fixed stale "1536 floats / 1536 dims" comments in vector_search.py and ingest_movie.py. Left CLAUDE.md / AGENTS.md / docs/PROJECT.md doc references alone — those belong in the follow-up cleanup once the actual re-embed lands.
Design context: project_embedding_model_decision.md memory; docs/decisions/ADR-011 embedding cache key format (emb:{model}:{hash}) means old 3-small and new 3-large cache entries coexist harmlessly — no manual Redis flush needed.
Testing notes: Verified via grep that no .py/.sh files still reference text-embedding-3-small or 1536 in the qdrant init. End-to-end validation (Qdrant drop/recreate, re-embed ~100K movies, assert `len(embedding) == 3072`) is explicitly out of scope for this code change and will happen in the follow-up task.

## reception vector awards summary + doc definition realignment
Files: schemas/metadata.py, movie_ingestion/final_ingestion/vector_text.py, search_improvement_planning/v2_data_needs.md, search_improvement_planning/v2_data_architecture.md, search_improvement_planning/new_system_brainstorm.md, search_improvement_planning/types_of_searches.md, docs/modules/schemas.md, docs/modules/ingestion.md, docs/decisions/ADR-001-eight-vector-spaces.md, docs/decisions/ADR-058-vector-text-formatting-conventions.md, docs/llm_metadata_generation_report.md, docs/llm_metadata_generation_efficiency_analysis.md
Why: We decided the reception vector should represent semantic reception/prestige language, not duplicate deterministic score buckets or award-query logic. `reception_tier` was a coarse derived label already available from structured scoring, while full nomination text would bloat the embedding with low-value detail better handled by `movie_awards`.
Approach: Changed `ReceptionOutput.embedding_text()` to emit a fully labeled shape (`reception_summary:`, `praised:`, `criticized:`). Removed `reception_tier` from `create_reception_vector_text()` entirely. Added a deterministic `_reception_award_wins_text()` helper in vector assembly that reads `Movie.imdb_data.awards`, keeps winner rows only, collapses to distinct ceremony names, emits them in fixed priority order, and intentionally excludes Razzie so the vector's `major_award_wins:` line stays prestige-oriented. Left `ReceptionOutput`'s schema fields unchanged because awards are external scraped data, not LLM-generated reception metadata. Updated the planning docs, module docs, ADRs, and reference docs so they consistently define the reception vector as labeled summary/qualities plus deterministic major-award wins, with nominations delegated to structured Postgres lookup.
Design context: search_improvement_planning/v2_data_architecture.md now documents the final embedding shape and the "no nominations in vectors" boundary; ADR-058 now reflects that vector_text supplies deterministic award wins rather than reception tier.
Testing notes: Ran `python3 -m py_compile schemas/metadata.py movie_ingestion/final_ingestion/vector_text.py`. Did not run pytest or edit tests per repo instructions.

## award outcome enum + helper for tracker-safe award handling
Files: movie_ingestion/imdb_scraping/models.py, movie_ingestion/imdb_scraping/parsers.py, movie_ingestion/final_ingestion/vector_text.py
Why: `AwardNomination.outcome` was still a raw string, which made winner checks stringly-typed at call sites. We wanted a typed `WINNER`/`NOMINEE` contract plus a small helper so award consumers can ask intentfully whether a nomination won.
Approach: Added `AwardOutcome(StrEnum)` with `WINNER` and `NOMINEE`, changed `AwardNomination.outcome` to that enum, and added `AwardNomination.did_win()` returning `self.outcome == AwardOutcome.WINNER`. Updated the GraphQL parser to emit enum values instead of raw strings and changed `_reception_award_wins_text()` to call `award.did_win()`. Left tracker serialization logic unchanged because it already persists IMDB scrape payloads from `IMDBScrapedMovie.model_dump(mode="json")`, which is the correct Pydantic boundary for serializing enums to JSON-compatible string values; `AwardNomination.model_validate(...)` on load reconstructs the enum on the way back out.
Testing notes: Ran `python3 -m py_compile movie_ingestion/imdb_scraping/models.py movie_ingestion/imdb_scraping/parsers.py movie_ingestion/final_ingestion/vector_text.py schemas/movie.py`. Attempted a runtime dump/validate roundtrip smoke test, but this shell lacks `pydantic`, so import-time execution could not run here.

## move AwardOutcome into shared schemas enums
Files: schemas/enums.py, movie_ingestion/imdb_scraping/models.py, movie_ingestion/imdb_scraping/parsers.py
Why: `AwardOutcome` is a shared schema-level concept rather than scraper-local state, so it belongs in `schemas/enums.py` alongside other cross-module enums. This keeps enum ownership consistent and avoids hiding a reusable type inside the IMDB scraper models module.
Approach: Moved `AwardOutcome(StrEnum)` into `schemas/enums.py`, removed the local definition from `movie_ingestion/imdb_scraping/models.py`, and updated both the scraper models and parser to import it from the shared enums module. JSON representation is unchanged (`"winner"` / `"nominee"`), so tracker persistence behavior remains the same.

## Franchise schema v4 — two-axis rewrite
Files: schemas/enums.py, schemas/metadata.py, movie_ingestion/metadata_generation/prompts/franchise.py, movie_ingestion/metadata_generation/generators/franchise.py

### Intent
Replace the v3 closed `franchise_role` enum with a decomposed, two-axis schema that separates IDENTITY (what brands/groups the film belongs to) from NARRATIVE POSITION (how it relates to prior films). The v3 design conflated these axes and produced three structural problems documented in search_improvement_planning/franchise_test_iterations.md that no amount of prompt tuning could fix: pair-remakes with no franchise (Scarface 1983) couldn't be represented, the Iron Man / X-Men 2000 null-case was internally contradictory, and the reboot ↔ remake overlap had no clean tiebreaker. v4 dissolves all three. The final field set was co-designed across ~8 conversation turns starting from a downstream-query inventory.

### Key Decisions
- **lineage vs shared_universe split.** `lineage` is the NARROWEST recognizable line of films (Batman, Spider-Man, Harry Potter, Godzilla), a SEMANTIC FLIP from v3 where those names were forbidden in favor of broader brands. `shared_universe` is the broader cinematic universe above the lineage (MCU, DCEU, Wizarding World, MonsterVerse) when one exists. Null when the lineage is itself top-level (Star Wars, James Bond). Matches the query shapes users actually type ("Batman movies" vs "Marvel movies").
- **`lineage_position` as a single nullable enum, not four booleans.** Values: sequel / prequel / remake / reboot / null. Schema-level mutual exclusivity means `sequel=true, prequel=true` is physically unreachable — the v3 overlap problems (Jungle Book 2016 satisfying both reboot and remake definitions) are dissolved by the enum shape, not by tiebreaker rules. Remake vs reboot tiebreaker language still in the prompt: remake = retells a specific prior story spine; reboot = new story with same IP.
- **`lineage_position` can populate with `lineage=null`.** Pair-remakes like Scarface 1983 retelling Scarface 1932 have a clean remake relationship without forming a multi-entry brand. v3's hard null-propagation rule destroyed this signal; v4 relaxes it so the inter-film relationship survives.
- **`special_attributes` as a small enum array, not separate booleans.** Spinoff and crossover are independent orthogonal predicates that combine freely with any `lineage_position` and with each other. The array form invites enumeration as a single decision and gives a cleaner empty default than two false booleans. v3's three-constraint spinoff test (MINOR IN SOURCE / GOES SOMEWHERE NEW / LEAVES THE SOURCE BEHIND) is preserved verbatim in the prompt because it was working correctly in v3.
- **`launches_subgroup` coupled to `recognized_subgroups`.** Defined as: true iff the film is the earliest-released entry in at least one of its `recognized_subgroups`. Replaces v3's vague "culturally-recognized subgroup" language. Silently enforced in `validate_and_fix` — if the model emits `launches_subgroup=true` with an empty groups list, the boolean is cleared. See the narrative-era critique in the conversation history for why this tight coupling beats the "new era trigger" framework (which undercounts MCU phase transitions and overcounts time-jumped sequels).
- **Reasoning fields before decision fields, scoped per block.** Three reasoning fields (identity, subgroups, position) instead of v3's five. Scoping keeps reasoning adjacent to the commitment it informs (Jason Liu / Instructor's "just-in-time reasoning" pattern) rather than going stale between a single top-level reasoning block and the structured answer at the bottom. Special_attributes has no dedicated reasoning field because the three-constraint spinoff test is operationalized sufficiently in the field description itself; revisit if v4 eval shows drift.
- **`FranchiseRole` deleted from `schemas/enums.py`.** No production code referenced the stable integer IDs (1–6) that enum carried — confirmed via grep. The aspirational Postgres column it was reserved for was never built.
- **`validate_and_fix` applies three deterministic fixups silently.** (1) Partial null-propagation: `lineage=null` clears `shared_universe` / `recognized_subgroups` / `launches_subgroup`, but deliberately leaves `lineage_position` and `special_attributes` alone so pair-remakes and Joker-2019-style standalone spinoffs work. (2) `launches_subgroup` coupling to `recognized_subgroups`. (3) `special_attributes` dedup. Silent correction keeps the batch pipeline flowing on single-row inconsistencies instead of hard-failing.

### Planning Context
Full field-set derivation is in the conversation history starting with the downstream-query inventory (~45 query shapes organized by tier) and progressing through iterative naming and mutual-exclusivity discussions. Key moments: (a) deciding that direct film-to-film relationship pointers are NOT needed because lineage-membership + relational flags compose for the same queries, (b) confirming `launches_subgroup` should be tightly coupled to `recognized_subgroups` rather than derived from narrative triggers (Narrative Era framework was critiqued and rejected), (c) the schema-design decision to use enums over boolean clusters driven by the concept_tags precedent and general LLM structured-output practice (Instructor, BAML, OpenAI structured outputs guide all converge on enums-over-booleans for mutually exclusive choices).

### Files modified
- `schemas/enums.py`: deleted `FranchiseRole`, added `LineagePosition` and `SpecialAttribute` enums.
- `schemas/metadata.py`: full rewrite of `FranchiseOutput` class body; updated enum imports; added `validate_and_fix` override.
- `movie_ingestion/metadata_generation/prompts/franchise.py`: full rewrite of `SYSTEM_PROMPT` (640 → ~720 lines) targeting the new schema. Preserved every v3 element that was working (EMPTY-THEN-ADD framing, three-constraint spinoff test, IS NOT filters, sub-series restatement blocks, Star-Wars-only "original trilogy" scoping, positive-commitment framing). Changed only what the new schema demands: lineage-vs-universe split section with worked examples, new lineage_position procedure with remake-vs-reboot tiebreaker, new special_attributes section, new launches_subgroup definition via earliest-released-in-subgroup rule, legal `lineage=null + lineage_position` combinations.
- `movie_ingestion/metadata_generation/generators/franchise.py`: module docstring and `build_franchise_user_prompt` docstring updated to reflect two-axis framing. No functional code changes — the generator wires the schema + prompt + LLM call generically, and the batch pipeline (`generator_registry.py`, `request_builder.py`, `result_processor.py`, `pre_consolidation.py`) required zero changes because everything references `FranchiseOutput` by class name (preserved) or `config.schema_class` generically.

### Verification performed
- Import + JSON-schema smoke test: `FranchiseOutput` imports cleanly, field order in the strict JSON schema matches reasoning-before-answer intent, registry resolves franchise → `FranchiseOutput`, prompt loads.
- `validate_and_fix` round-trip tests on 8 edge cases, all passing: Scarface pair-remake (position stays with lineage=null), null-propagation clears shared_universe, null-propagation clears recognized_subgroups + launches_subgroup, launches_subgroup forced false when groups empty, special_attributes dedup, Joker-2019-style standalone spinoff (lineage=null + special_attributes=["spinoff"] preserved), healthy Iron Man case untouched, enum rejection of invalid value "mainline" raises ValidationError as expected.

### Follow-up (not in scope for this change)
- Existing rows in the SQLite `generated_metadata.franchise` column are in v3 format and will be overwritten on the next batch run. Recommendation: re-run franchise generation for all movies after this change lands.
- `movie_ingestion/metadata_generation/generators/test_franchise.ipynb` and `franchise_test_results.json` (both in git status, uncommitted) will become stale against the new schema. Deferred to a separate task per user decision.
- Downstream integration: `FranchiseOutput` is NOT currently consumed by `vector_text.py` (which reads `SourceOfInspirationOutput.franchise_lineage` instead) or by `ingest_movie.py` (no franchise references). When FranchiseOutput is eventually wired into vector text or Postgres columns, that will be a separate change.

## Franchise v4 — schema compaction and planning-doc write-through
Files: schemas/metadata.py, schemas/enums.py, search_improvement_planning/franchise_test_iterations.md
Why: After the initial v4 rewrite landed, the per-field `Field(description=...)` text and the class-level comments inside `FranchiseOutput` were verbose enough to duplicate most of what the system prompt already carries — wasting LLM context. Separately, `LineagePosition` and `SpecialAttribute` used Python class docstrings for their documentation, which Pydantic ships into the generated JSON schema under `$defs` `description`, leaking guidance to the model through a second uncontrolled channel. And the full v4 decision set was still only in conversation form; it needed to be saved to the permanent planning doc.
Approach: Compacted every `FranchiseOutput` field description to a single short definitional sentence plus the "must be written BEFORE X" ordering note where relevant. Total per-property description chars dropped from ~9,400 to 1,875 (~80% reduction). Kept reasoning-before-answer field order, `validate_and_fix` behavior, and all validator round-trip test cases unchanged. Replaced the block-header comments inside the class (`# IDENTITY BLOCK`, etc.) with short single-line section comments and moved the architecture rationale into a single comment block ABOVE the class. Moved enum documentation from class docstrings into `#`-comments above each class definition in `schemas/enums.py`, matching the existing `SourceMaterialType` pattern — verified via `to_strict_json_schema` that `LineagePosition.description` and `SpecialAttribute.description` are both `None` in the generated JSON schema. Separately, appended a ~460-line v4 section to `search_improvement_planning/franchise_test_iterations.md` documenting the query-inventory-driven design, the final field list, the per-field definitions, the validator fixups, a 19-film worked-examples acceptance table, the v3→v4 resolution table, seven load-bearing schema-design decisions, and the test acceptance criteria for the next eval.
Testing notes: Re-ran the strict JSON schema generator — field order preserved, enum `$defs` descriptions are null. Re-ran all six validator round-trip edge cases (pair-remake, null-prop clears SU, null-prop clears groups, launches_subgroup coupling, special_attributes dedup, standalone spinoff) — all still pass. No functional regressions.
Testing notes: Ran `python3 -m py_compile schemas/enums.py movie_ingestion/imdb_scraping/models.py movie_ingestion/imdb_scraping/parsers.py schemas/movie.py`.

## Viewer experience embedding text — structured labels with explicit negations
Files: schemas/metadata.py, implementation/prompts/vector_subquery_prompts.py, unit_tests/test_metadata_embedding_text.py, unit_tests/test_vector_text.py, search_improvement_planning/v2_data_architecture.md, docs/modules/schemas.md, docs/modules/ingestion.md, docs/decisions/ADR-058-vector-text-formatting-conventions.md
Why: The V2 search planning work identified flat term bags as a retrieval weakness for multi-dimensional movies. Viewer experience needed the same structured-label treatment already planned for other spaces, but with one additional requirement: negations had to remain first-class and polarity-safe rather than being mixed into the positive term stream. The docs also needed an explicit boundary statement so new deterministic V2 data does not drift back into this vector space.
Approach: Reworked `ViewerExperienceOutput.embedding_text()` to emit fixed-order labeled multiline text, with one positive line and one optional `*_negations:` line per section. Kept the existing 8 schema fields and upstream generation inputs unchanged. Left `create_viewer_experience_vector_text()` as a thin wrapper over the schema method. Updated viewer-experience embedding tests to assert labeled output, explicit negation lines, omission of empty negation lines, and stable section ordering. Realigned the viewer-experience search subquery prompt away from the old "flat, unlabeled comma-separated list" model and rewrote its examples into the new labeled multiline shape so query embeddings and document embeddings stay format-aligned. Updated planning docs and persistent docs to describe the new embedding contract, the explicit-negation convention, and the rule that awards, franchise, source-material types, countries, box-office buckets, keyword IDs, production techniques, and concept tags remain outside this vector space.
Testing notes: `python -m py_compile schemas/metadata.py implementation/prompts/vector_subquery_prompts.py movie_ingestion/final_ingestion/vector_text.py` passed. `pytest unit_tests/test_metadata_embedding_text.py -q -k "ViewerExperienceEmbeddingText"` passed (5 tests). Broader `unit_tests/test_metadata_embedding_text.py` still has pre-existing `plot_analysis` label expectation failures unrelated to this change, and `unit_tests/test_vector_text.py` could not be imported in this environment because `orjson` is missing locally.

## Reduced anchor vector refresh
Files: movie_ingestion/final_ingestion/vector_text.py, schemas/movie.py, docs/decisions/ADR-001-eight-vector-spaces.md, docs/decisions/ADR-058-vector-text-formatting-conventions.md, docs/llm_metadata_generation_report.md, docs/modules/schemas.md, docs/decisions/ADR-056-movie-tracker-backed-schema-loader.md, search_improvement_planning/v2_data_architecture.md, search_improvement_planning/v2_data_needs.md, search_improvement_planning/open_questions.md, search_improvement_planning/new_system_brainstorm.md, search_improvement_planning/keyword_vocabulary_audit.md
Why: We decided to keep `anchor` in V2, but only as a lean holistic fingerprint rather than the old broad catch-all movie card. The embedded text needed to drop structured/filterable facts and stabilize around labeled movie-wide summary fields. The planning docs and active reference docs also needed to stop describing anchor as either broad or removed.
Approach: Rewrote `create_anchor_vector_text()` to emit labeled multiline text in a fixed order: `title`, `original_title`, `identity_pitch`, `identity_overview`, `genre_signatures`, `themes`, `emotional_palette`, `key_draws`, `maturity_summary`, `reception_summary`. Removed keywords, source/franchise signals, languages, decade, budget, awards, reception tier, and other non-holistic content from the anchor text. Switched genre content to plot-analysis `genre_signatures` only, with no IMDB-genre merge. Removed the now-unused `Movie.title_with_original()` helper. Updated the vector-space ADR, vector-formatting ADR, anchor reference section in the metadata-generation report, and the V2 planning docs so they consistently define anchor as retained in reduced form and update the embed-count language back to 8 spaces / 800K embeddings.
Design context: The reduced anchor intentionally preserves only high-level "movie as a whole" semantics while leaving structured/filterable facts in Postgres or specialized vectors. This keeps anchor useful for holistic similarity without recreating the dilution problem from the broad V1 shape.
Testing notes: Ran `python3 -m py_compile movie_ingestion/final_ingestion/vector_text.py schemas/movie.py` successfully. Ran targeted grep/consistency checks to confirm the touched planning docs no longer say anchor is dropped from V2 or assume a 7-space embedding rebuild. Did not run pytest or edit tests in this pass.

## Franchise v5 — launched_franchise flag, normalization rule, shared_universe loosening, field rename
Files: schemas/metadata.py, schemas/enums.py, movie_ingestion/metadata_generation/prompts/franchise.py, movie_ingestion/metadata_generation/generators/franchise.py, movie_ingestion/metadata_generation/generators/test_franchise.ipynb, movie_ingestion/metadata_generation/generators/franchise_test_results.json

### Intent
Address four structural gaps surfaced by the v4 79-movie eval: (1) "movies that launched a franchise" queries had no retrieval signal because launches_subgroup is structurally locked to false for franchise openers without named subgroups (Shrek, Matrix, Jurassic Park); (2) no global normalization rule for named entities allowed MCU/DCEU to leak into lineage/shared_universe fields; (3) shared_universe was too strict to handle spinoff-parent relationships (Puss in Boots→Shrek, Minions→Despicable Me, Logan→X-Men); (4) the anti-restatement filter stripped disambiguating qualifiers like "connery bond era". Also renamed launches_subgroup → launched_subgroup for tense consistency with the new flag.

### Key Decisions
- **New FIELD 7 `launched_franchise`** with a four-part test: first cinematic entry (lineage_position null), not a spinoff, source-material recognition test (film franchise dominates over any prior book/game/toy/show), and relevant follow-ups test (audience recognizes a continuing film franchise). Independent from launched_subgroup — a film can fire one, both, or neither. Iron Man 2008 fires launched_subgroup=true (opens phase one inside Marvel) and launched_franchise=false (Marvel franchise already existed). Shrek 2001 fires only launched_franchise=true.
- **Universal normalization** for every named entity (lineage, shared_universe, every subgroup label): lowercase, digits spelled out, "&" → "and", abbreviations and first+last names expanded only when the expansion is in common use (MCU → marvel cinematic universe ✓; monsterverse stays). Applied as a GLOBAL OUTPUT RULES block near the top of the prompt and restated inside FIELD 3+4 for emphasis (user chose duplication over single source of truth).
- **shared_universe now accepts two shapes**: (A) formal shared cinematic universe hosting multiple lineages (marvel cinematic universe, dc extended universe, wizarding world, monsterverse, conjuring universe); (B) parent franchise of a spinoff sub-lineage (puss in boots → shrek; minions → despicable me; logan → x-men). Shape B is new — v4 rejected it outright.
- **Hobbs & Shaw stays under lineage "fast and furious"** with no shared_universe — single spinoff, insufficient volume to promote to its own lineage. Will revisit if a second film ships.
- **REMAKE enum value retained** for classification fidelity but NOT consumed at search time — source_of_inspiration covers film-to-film retellings. Documented via a code comment above the enum member.
- **Anti-restatement carveout**: a label that differs from the lineage by a meaningful disambiguating qualifier (era, director, actor, timeline) is NOT a bare restatement. "connery bond era" is now valid.
- **validate_and_fix() coherence block for launched_franchise**: forcibly false when any precondition fails (lineage null, lineage_position populated, or spinoff in special_attributes). Keeps the flag from drifting out of sync with the rest of the record.
- **SOT updates**: normalized all expected_lineage / expected_shared_universe / expected_recognized_subgroups to lowercase canonical forms; promoted Minions to its own lineage with shared_universe="despicable me"; added shared_universe="shrek" to Puss in Boots; added shared_universe="x-men" to Logan; gave Rogue One a "star wars anthology films" subgroup with launched_subgroup=true; added expected_launched_franchise=None (placeholder pending joint user review) to all 79 rows.
- **Eval scorer extended** with a new LAUNCHED_FRANCHISE_WRONG RunResult and a launched_franchise_ok field comparison. When expected_launched_franchise is None the check is skipped (treated as OK) so the scorer remains valid until the user finalizes per-row rulings.

### Planning Context
Plan file: /Users/michaelkeohane/.claude/plans/nested-jumping-quokka.md. User rulings were iterated over four conversation turns; key changes from my initial "either-axis" framing: Iron Man / Batman Begins / Casino Royale are launched_subgroup only (NOT launched_franchise), Jaws is FALSE because sequels are culturally forgotten, spinoffs can never be launched_franchise=true by definition, and the normalization rule favors the most common form (so compact forms stay when the expansion isn't in common use). Spinoff redefinition was deferred to a future message — FIELD 6 spinoff text is unchanged in this pass.

### Testing Notes
- All modified files compile (ast.parse on .py; compile() on notebook cell sources).
- Notebook field rename was done via Python to preserve JSON structure.
- launched_franchise ruling table for all 79 movies is the next step — I have a proposal ready but will present it for joint walk-through before populating the SOT (user ruling F3).
- After SOT is locked: rerun the eval notebook against all four candidate models (gpt5-mini-medium, gpt5-mini-low, gpt5-mini-minimal, gpt54-mini-low) and compare v5 accuracy per field against the v4 baseline already in franchise_test_results.json.
- Watch for regressions specifically on: (a) lineage normalization casing, (b) launched_franchise four-part test accuracy on Group C cases (Iron Man, Black Panther, Wonder Woman, X-Men, Spider-Man 2002), (c) shape B shared_universe on spinoff parents, (d) Sherlock Holmes 2009 firing launched_subgroup for "ritchie sherlock holmes films".

## Spinoff redefinition — protagonist-shift + legacy-centrality test
Files: movie_ingestion/metadata_generation/prompts/franchise.py, movie_ingestion/metadata_generation/generators/test_franchise.ipynb, search_improvement_planning/franchise_test_iterations.md

Why: The v4 three-constraint spinoff test keyed on "MINOR IN SOURCE" as constraint (a), which bailed out on Creed (Apollo Creed was a co-lead of Rocky 1976, so Adonis technically descends from a non-minor source character). The v4 fallback was constraint (c)'s "legacy sequel with prior protagonist present = NOT a spinoff" rule, which forced Creed into a pure-sequel slot and erased its genuinely spinoff-like nature (new protagonist, Rocky as trainer not fighter, Rocky's arc already complete). User agreed via side-conversation analysis that a protagonist-shift framing is structurally cleaner.

Approach: Replaced constraints (a)(b)(c) in FIELD 6 spinoff block with a revised three-constraint test:
- (a) NEW-TO-THE-SOURCE PROTAGONIST — measured against the SOURCE at the top of the lineage tree, not the immediate predecessor. Puss in Boots LW still qualifies because Puss was a side character in Shrek 2, even though he's been the lead of his sub-lineage since 2011.
- (b) PRIOR LEAD NOT IN THE DRIVER'S SEAT — mentors, allies, passengers are fine; prior lead still calling the shots is not.
- (c) PRIOR LEGACY NOT CENTRAL TO THIS PLOT — the key new test. Distinguishes Creed (Rocky's arc is complete) from legacy sequels whose spine is the prior hero's legacy (Ghostbusters: Afterlife → Egon's redemption, TFA → Skywalker family saga, Blade Runner 2049 → Deckard's paternity, Halloween 2018 → Laurie's trauma).

Planned-pillar carve-out preserved verbatim from v4 (Wonder Woman, Black Panther, Captain Marvel, Doctor Strange, Thor are never spinoffs regardless of the constraint test).

Example lists rewritten: Creed moved from "fails" to "fires" with full constraint-by-constraint reasoning. Added TFA, Blade Runner 2049, Halloween 2018, Tron: Legacy as new negative examples exercising constraint (c) — these hold the line against legacy-sequel drift that the earlier two-constraint version of this redefinition would have mis-classified. IS NOT summary block rewritten to reorganize failures by which constraint they fail.

Test notebook SOT: single entry updated — tmdb_id 312221 (Creed) `expected_special_attributes` flipped from `[]` to `["spinoff"]`, blurb updated with new reasoning. Verified against the notebook that no other test case needed updating: Rogue One / Solo / Puss in Boots LW already `["spinoff"]`, Wonder Woman / Black Panther already `[]` via pillar carve-out. TFA / Blade Runner 2049 / Halloween 2018 / Top Gun: Maverick / Ghostbusters: Afterlife / Joker / Prometheus are not in the SOT at all so no update needed there.

Worked-examples table in search_improvement_planning/franchise_test_iterations.md updated so the Creed row shows `[spinoff]` in the special_attributes column to match the new SOT.

No schema changes, no validator changes, no generator code changes. Enum vocabulary and field structure unchanged.

Design context: Plan file /Users/michaelkeohane/.claude/plans/piped-weaving-tiger.md; in-line analysis in the conversation weighing protagonist-shift vs. minor-in-source framings and resolving the TFA / Rogue One edge cases that a pure two-constraint formulation would have gotten wrong.

Testing Notes:
- Prompt is LLM-consumed text; no compile check applies beyond reading the rewritten block and confirming internal consistency between (a)(b)(c), the example lists, and the IS NOT summary.
- Notebook change was a targeted JSON-level two-line edit; verified exactly two lines of cell 3 source differ (blurb + expected_special_attributes).
- Full eval re-run is deferred (harness re-run is already its own deferred task per the v4 plan). Acceptance targets for the next run: Creed fires spinoff with reasoning traces citing the new-protagonist + Rocky-as-trainer + arc-complete logic; no regressions on Rogue One/Solo/PIB LW/Venom/Joker/Prometheus/Maleficent; pillar disqualifier still holds Wonder Woman/Black Panther/Captain Marvel/Doctor Strange/Thor off; TFA/Blade Runner 2049/Halloween 2018/Top Gun: Maverick/Ghostbusters: Afterlife do not fire spinoff; Creed's lineage_position remains "sequel".

## Watch context embedding text — labeled multiline format
Files: schemas/metadata.py, movie_ingestion/final_ingestion/vector_text.py, implementation/prompts/vector_subquery_prompts.py, search_improvement_planning/v2_data_architecture.md, docs/modules/schemas.md, docs/modules/ingestion.md, docs/decisions/ADR-058-vector-text-formatting-conventions.md, unit_tests/test_metadata_embedding_text.py, unit_tests/test_vector_text.py, unit_tests/test_vector_subquery_prompts.py
Why: We decided to keep watch-context content untouched and change only its embedding representation. The old flat comma-separated term bag needed to become structured labeled text so document embeddings, query embeddings, and source-of-truth docs all describe the same shape.
Approach: Reworked `WatchContextOutput.embedding_text()` to emit fixed-order labeled lines for `self_experience_motivations`, `external_motivations`, `key_movie_feature_draws`, and `watch_scenarios`, omitting empty sections and still excluding `identity_note` and `evidence_basis`. Left `__str__()` and the underlying generated data unchanged. Kept `create_watch_context_vector_text()` as a thin wrapper, but documented it as returning labeled multiline schema output. Rewrote the watch-context subquery prompt from the old flat-list description to the new labeled-line format, including updated examples and output contract so query text is generated in the same structure. Updated the planning/docs sources of truth to explicitly call this a formatting-only change.
Design context: The user explicitly chose an all-or-nothing data policy for watch_context, so no terms or sections were removed. The only behavior change is formatting and alignment between ingestion-time embeddings and search-time prompt guidance.
Testing notes: `pytest unit_tests/test_metadata_embedding_text.py -q -k WatchContext` passed (6 tests). `pytest unit_tests/test_vector_subquery_prompts.py -q` passed (2 tests). `uv run python -m pytest unit_tests/test_vector_text.py -q -k WatchContext` passed (2 tests). A broader direct run of `unit_tests/test_vector_text.py` outside `uv` is still blocked locally because `orjson` is not installed in the base interpreter, and the broader `unit_tests/test_metadata_embedding_text.py` file still contains unrelated pre-existing plot-analysis expectation failures.

## Narrative techniques embedding text — structured labels on ingestion side
Files: schemas/metadata.py, unit_tests/test_metadata_embedding_text.py, unit_tests/test_vector_text.py, search_improvement_planning/v2_data_architecture.md, docs/modules/schemas.md, docs/modules/ingestion.md, docs/llm_metadata_generation_new_flow.md, docs/decisions/ADR-058-vector-text-formatting-conventions.md
Why: The `narrative_techniques` vector was still embedding as a flat comma-separated bag of terms even though the V2 search planning work depends on section-preserving structured labels for cross-space rescoring. The source-of-truth docs for this vector had also drifted: some still described old 11-section outputs and obsolete input assumptions.
Approach: Reworked `NarrativeTechniquesOutput.embedding_text()` to emit fixed-order labeled multiline text, one line per populated section, using the real schema field names (`narrative_archetype`, `narrative_delivery`, `pov_perspective`, `characterization_methods`, `character_arcs`, `audience_character_perception`, `information_control`, `conflict_stakes_design`, `additional_narrative_devices`). Empty sections are omitted, per-term `normalize_string()` behavior is preserved, and justification/evidence fields remain excluded. Left `__str__()` unchanged and kept `create_narrative_techniques_vector_text()` as a thin wrapper. Updated the V2 data architecture doc, schema/module docs, ingestion module docs, the metadata-generation flow doc, and ADR-058 so they all describe the same 9-section labeled embedding contract and current merged-keyword / plot-or-craft input model.
Design context: Scope was intentionally limited to ingestion-side embedding text and documentation. No new data sources were added to `narrative_techniques`, and search-side schema/prompt realignment plus full re-embedding remain separate follow-up work.
Testing notes: `uv run python -m pytest unit_tests/test_metadata_embedding_text.py::TestNarrativeTechniquesEmbeddingText unit_tests/test_vector_text.py::TestNarrativeTechniquesReturnsNone -q` passed (6 tests). A broader `uv run python -m pytest unit_tests/test_metadata_embedding_text.py unit_tests/test_vector_text.py -q` run still reports unrelated pre-existing failures in plot-analysis, anchor, and reception expectations that do not touch the narrative-techniques formatter.

## Franchise metadata v8 — split is_crossover / is_spinoff into independent boolean tests
Files: schemas/enums.py, schemas/metadata.py, movie_ingestion/metadata_generation/prompts/franchise.py, movie_ingestion/metadata_generation/generators/franchise.py

### Intent
Replace the `special_attributes: list[SpecialAttribute]` enum array (with a single shared `special_attributes_reasoning` field) with two independent boolean tests: `is_crossover` / `crossover_reasoning` and `is_spinoff` / `spinoff_reasoning`. Rebuild both procedures from scratch. The old v7 scaffold was character-first ("was the lead a major or minor character in the source?") and systematically misclassified origin-story branches like Solo: A Star Wars Story — Han was labeled a "lead" of the 1977 original, constraint (a) mechanically failed, and the model emitted `special_attributes = []` despite Solo being unambiguously a spinoff by its anthology sub-banner framing.

### Key Decisions
- **Split the reasoning fields.** Spinoff and crossover are two different tests with different inputs and different failure modes. Sharing one reasoning budget let the longer spinoff analysis crowd out crossover and forced the model to juggle both tests at once. Each now has its own reasoning trace, and field ordering (crossover_reasoning → is_crossover → spinoff_reasoning → is_spinoff) enforces "reason before verdict" independently for each test.
- **Crossover is now a single identity question.** "Is this film's identity the fact that multiple known entities or characters that normally live in separate stories are now interacting?" The old DEFINING-TRAIT TEST asked the model to enumerate parent franchises first, which biased toward hallucinated pairings and false positives. Starting from identity and short-circuiting on the single question is both simpler and less hallucination-prone.
- **DELIBERATE SEMANTIC CHANGE on crossover.** Shared-universe ensemble films now fire `is_crossover=true`: Avengers (2012), Age of Ultron, Infinity War, Endgame, Civil War, Justice League. The old "same top-level brand disqualifies crossover" rule is removed. User explicitly endorsed this reinterpretation ("Avengers is a crossover movie because it's about all these characters that are normally kept to their own stories have all come together"). Downstream retrieval behavior for "crossover" queries will shift to include team-up films.
- **Spinoff is rebuilt around structural situating.** New four-step procedure: (1) parametric knowledge supplement — specific named labels only (sub-banner names, studio slates), 95%+ confidence required, no invented framings, supplements rather than overrides the provided inputs; (2) structural situating — carries-forward / leaves-behind analysis plus trunk-vs-branch placement; (3) conditional character disambiguation — only runs when Step 2 is ambiguous, and reframed as lead-character / lead-plotline / lead-events (NOT major vs. minor in source); (4) verdict. Under the new scaffold Solo resolves via Step 1 parametric recall of "A Star Wars Story" before character questions ever run.
- **Parametric knowledge supplements, never overrides.** If provided inputs clearly contradict a recalled framing, trust the inputs. This is a deliberate failure mode accepted in exchange for hallucination safety — the user explicitly chose this posture over "override-allowed for specific labels".
- **Trunk-vs-branch stays as a reasoning artifact, not a first-class field.** User declined to promote it. Considered but rejected adding it as a retrieval signal.
- **Planned-pillar carve-out shrinks to one sentence.** Under structural situating, Wonder Woman / Black Panther / Captain Marvel / Doctor Strange / Thor resolve as trunk entries of their shared cinematic universe from Step 2 alone. No dedicated carve-out block needed.
- **Schema field descriptions stay minimal.** Per the earlier decision in this session, reasoning-field descriptions only say "follow the procedure defined in the system prompt" and "must be emitted BEFORE X". All definitional content lives in the system prompt to avoid two competing procedures drifting out of sync.
- **`validate_and_fix` updated.** The launched_franchise coherence check now reads `instance.is_spinoff` instead of `SpecialAttribute.SPINOFF in instance.special_attributes`. The special_attributes dedup block is removed (no longer a list). Partial null-propagation comments updated to reference `is_crossover` and `is_spinoff`.
- **FIELD 7 Test 2 updated** to read `is_spinoff` directly instead of checking membership in the retired enum list. All stale `special_attributes` references in FIELD 7 facts, decision gate, IS NOT block, and hard-constraints block were swept.

### Planning Context
The full plan lives at `/Users/michaelkeohane/.claude/plans/radiant-riding-pebble.md`. Rationale for the character-first → structure-first shift, the crossover identity-question framing, the parametric-knowledge posture, and the non-goals (tests out of scope, human planning docs untouched, other reasoning fields unchanged) are all captured there.

### Testing Notes
Did NOT touch `test_franchise.ipynb` or `franchise_test_results.json` per the test-boundaries rule — both reference `expected_special_attributes` / `special_attributes` and will need a separate pass once the user explicitly asks. Also deliberately untouched: `search_improvement_planning/franchise_metadata_planning.md`, `search_improvement_planning/franchise_test_iterations.md`, `docs/conventions_draft.md`.

Verified end-to-end via an inline Python check: `FranchiseOutput.model_json_schema()` reports the expected 14 fields in the correct order with `special_attributes` / `special_attributes_reasoning` removed and the four new fields present; `validate_and_fix` coerces `launched_franchise=False` when `is_spinoff=True`; `schemas.enums` no longer exposes `SpecialAttribute`. A grep of all `.py` files for `special_attributes` / `SpecialAttribute` returns only historic docstring references in `prompts/franchise.py` (v7 version history), `generators/franchise.py` (v8 migration note), and `schemas/metadata.py` (v8 migration note) — no runtime code references remain.

Wet-run evaluation on Solo / Avengers / a main-trunk sequel / a planned-pillar film is the expected next step and will probably want to happen in the same notebook pass that updates the test fixtures. Not run in this changeset.

## Franchise prompt — Scarface→Cape Fear test-leak replacement
Files: movie_ingestion/metadata_generation/prompts/franchise.py | Replaced all six Scarface (1983) references (intro v-notes, axes note, Field 5 intro, remake definition, worked examples, RULES section) with Cape Fear (1991) to remove answer leakage for the deterministic franchise test set; de-duplicated the remake examples list since Cape Fear was already present, and preserved the canonical "pair-remake with lineage=null" pedagogy verbatim.

## Batch status/process - tolerate older tracker schemas
Files: movie_ingestion/metadata_generation/batch_generation/run.py
Why: The review pass found that `status`/`process` could crash on an older tracker DB if the code knew about a newer metadata type but `metadata_batch_ids` had not been migrated with that `{type}_batch_id` column yet.
Approach: Made `_get_active_batch_ids()` inspect `PRAGMA table_info(metadata_batch_ids)` and build its UNION query only from batch-id columns that actually exist in that specific DB. Missing columns now also emit a warning print naming the expected `metadata_batch_ids.{type}_batch_id` column before that metadata type is skipped for polling. This keeps batch polling resilient across mixed code/schema states without changing the command entrypoints or widening migration behavior.
Design context: This is a targeted hardening fix for the batch-processing path only. It preserves the existing registry-driven CLI behavior while avoiding `OperationalError` on legacy tracker snapshots.
Testing notes: Not run. The change is a small deterministic SQL-schema guard and the repo's AGENTS instructions say not to run tests unless explicitly asked.

## Production techniques evaluation notebook scaffold
Files: movie_ingestion/metadata_generation/generators/production_techniques.py, movie_ingestion/metadata_generation/generators/test_production_techniques.ipynb
Why: Add a reusable evaluation notebook for `production_techniques` so candidate model configs can be compared deterministically on a fixed movie set, with raw outputs and token usage persisted to JSON for later analysis.
Approach: Updated `generate_production_techniques()` to accept optional provider/model/kwargs overrides while preserving the existing default OpenAI `gpt-5-mini` low/low behavior when no overrides are passed. Added `test_production_techniques.ipynb` with dynamic `find_project_root()` bootstrapping, candidate definitions, an empty `TEST_MOVIES` scaffold, movie loading, 3x-per-candidate async generation, deterministic JSON result persistence, exact normalized-set evaluation helpers, per-candidate summary reporting, per-bucket breakdown, and an optional movie inspector.
Design context: The notebook follows the franchise evaluation workflow shape but stays scoped to `production_techniques` semantics: exact set matching on `terms`, aggregate missing/extra term reporting, and no batch/search/embedding changes. The empty-test-set path deliberately avoids touching `tracker.db` so the scaffold no-ops cleanly before any fixtures are added.
Testing notes: Did not run tests. Non-test sanity checks passed: notebook JSON parses, all notebook code cells compile with top-level await enabled, and `python -m py_compile movie_ingestion/metadata_generation/generators/production_techniques.py` succeeds.

## Franchise prompt v9 anchors and reasoning examples
Files: movie_ingestion/metadata_generation/prompts/franchise.py, movie_ingestion/metadata_generation/generators/franchise.py, schemas/metadata.py
Why: The franchise prompt had a cluster of repeat failures on a small set of high-value entities plus a few under-taught reasoning patterns around selective continuity, same-source adaptations, recasts, and source-dominance launch checks. We wanted prompt-side bias correction without changing the schema or runtime contract.
Approach: Bumped the franchise prompt from v8 to v9 and added four prompt-only changes. First, inserted a short FIELD 1+2 clarification that independent adaptations of the same bounded source do not automatically form one lineage. Second, expanded FIELD 5 worked examples with Terminator: Dark Fate, Dune (2021), Mad Max: Fury Road, and Jurassic World. Third, expanded FIELD 6B / FIELD 7 examples with The Ballad of Songbirds and Snakes plus launched_franchise=false examples for Scooby-Doo (2002) and The Hunger Games (2012). Fourth, added a new FRANCHISE REFERENCE section after FIELD 7 with exact-match anchors for the known failure cases and normalization traps: Creed, Logan, Transformers (2007), Space Jam (1996), Hobbs & Shaw, Detective Pikachu, Bond era labels, Sherlock Holmes (2009), Iron Man (2008), Guardians of the Galaxy (2014), Scream (2022), Venom (2018), Ocean's Eight (2018), Split, Glass, and the canonical "the dark knight trilogy" label. Updated the generator and schema header comments to describe the prompt as v9 and call out the new reference-anchor behavior.
Design context: This deliberately keeps the FIELD procedures as the default reasoning path for novel titles. The new reference section is an override layer only for exact title/year matches where the model repeatedly drifted away from the project's established source-of-truth rulings. Per the user's later instruction, notebook / eval-fixture work was explicitly deferred and not touched in this pass.
Testing notes: No tests or notebook executions were run. This pass was limited to prompt/comment changes only, and it intentionally avoided `test_franchise.ipynb`, `franchise_test_results*.json`, and unit tests after the user asked not to involve the Python notebook or tests.

## Franchise generator: hardcode gpt-5.4-mini model candidate
Files: movie_ingestion/metadata_generation/generators/franchise.py | Removed provider/model/kwargs params from generate_franchise(); hardcoded gpt-5.4-mini with reasoning_effort=low, verbosity=low as the sole model candidate.

## Production techniques definition — include animation methods and exclude format labels
Files: movie_ingestion/metadata_generation/prompts/production_techniques.py, movie_ingestion/metadata_generation/generators/production_techniques.py, schemas/metadata.py
Why: The earlier prompt treated some format/category labels as production techniques while excluding animation modalities the user explicitly wants in-scope. We clarified the boundary so this metadata captures concrete making/rendering/capture methods instead.
Approach: Rewrote the production-techniques prompt to include animation modalities and sub-techniques (`hand-drawn animation`, `computer animation`, `2d/3d animation`, `traditional animation`, `stop motion`, `rotoscope`, `motion capture`, hybrid/partial labels), keep both broad and specific technique labels when both appear, exclude `imax` as viewing-only, exclude `documentary` / `mockumentary` / `pseudo documentary` / `anthology` / `vignette` / `nonlinear timeline` as movie-type or structural labels, and keep `found footage` as an explicit carveout rather than a general rule. Updated adjacent generator/schema descriptions to match so comments do not drift from prompt behavior.
Design context: This follows the user's clarified rule that animation is absolutely a production technique, `3d` can count when it reflects design/production, `imax` is out as viewing experience, and format/category labels should not be promoted into production-technique metadata except for the explicit `found footage` exception.
Testing notes: Not run. This pass only updates prompt/schema/comment definitions; notebook fixtures and unit tests were intentionally left untouched.

## Production techniques notebook — populate deterministic fixture set
Files: movie_ingestion/metadata_generation/generators/test_production_techniques.ipynb
Why: The notebook scaffold still had an empty `TEST_MOVIES` dict, so it could not exercise the revised production-techniques definition or measure candidate consistency on known edge cases.
Approach: Replaced the placeholder cell with a populated fixture set spanning four intent buckets: rich inputs with mixed signal, sparse inputs with mixed signal, hard true positives, and false-positive traps. The selected titles are all present in `tracker.db` and were chosen to stress the newly clarified boundary: animation modalities/sub-techniques in, `imax` out, movie-type/structural labels out, and `found footage` in only as an explicit exception. Each entry now includes `title_year`, a short rationale blurb, a bucket label, and lowercase-normalized expected outputs matching the notebook evaluator.
Design context: The fixture mix intentionally combines obvious positives (`stop motion animation`, `motion capture`, `rotoscoping`) with boundary cases (`computer animation`, `3d`, `part stop motion animation`, `found footage`) and strong false-positive traps (`mockumentary`, `filmmaking`, `behind the scenes`, `film within a film`). This should make evaluation deltas easier to interpret when comparing prompt/model variants.
Testing notes: Verified that the notebook JSON is valid and that the populated cell renders as expected when parsed. Did not execute the notebook or run tests.

## Production techniques notebook — persist hydrated inputs and add candidate miss inspector
Files: movie_ingestion/metadata_generation/generators/test_production_techniques.ipynb
Why: After the fixtures were populated, the notebook still did not surface the actual input keyword lists in saved results or provide a fast way to inspect all misses for one candidate across the full movie set.
Approach: Updated the movie-loading cell to hydrate each `TEST_MOVIES` entry with `plot_keywords`, `overall_keywords`, and a deduped `combined_keywords` list from `tracker.db`. Extended the saved `results_payload` so those hydrated inputs persist into the JSON results file. Added a new bottom cell where the user can set `CANDIDATE_MISMATCH_LABEL` and print every movie the candidate failed on, including the plot keywords, overall keywords, combined input, expected output, and per-run received outputs with missing/extra deltas.
Design context: Keeping the keyword hydration in the load cell avoids bloating the hand-maintained fixture definitions while still making the notebook self-contained after generation runs. The mismatch inspector is intentionally candidate-centric and uses the existing `evaluate_movie()` / `classify_run()` helpers so it reflects the same exact-match logic as the summary tables.
Testing notes: Verified the notebook JSON remains valid after the new cell and payload changes. Did not execute the notebook or run tests.

## Production techniques notebook — fix Gemini thinking kwargs shape
Files: movie_ingestion/metadata_generation/generators/test_production_techniques.ipynb
Why: The Gemini candidate configs were passing `thinking_budget` as a flat string kwarg plus an unsupported `verbosity` field, which does not match either the official Gemini request shape or the installed `google-genai` SDK validation rules.
Approach: Updated the Gemini notebook candidates to pass nested `thinking_config` kwargs with integer `thinking_budget` values (`0` for off, `1024` for low-budget thinking) and removed the invalid Gemini `verbosity` kwarg. The OpenAI candidates were left unchanged.
Design context: The repo's Gemini adapter forwards `kwargs` directly into the Gemini SDK config dict, so the notebook must use the SDK's actual field structure. Official Gemini docs show thinking controls under `thinkingConfig`/`thinking_config`, and the local SDK accepts `thinking_config={"thinking_budget": ...}` but rejects flat `thinking_budget` and `verbosity`.
Testing notes: Verified the notebook JSON remains valid. Also checked locally with `google.genai.types.GenerateContentConfig` that nested `thinking_config` works, while flat `thinking_budget` and Gemini `verbosity` raise validation errors. Did not execute the notebook itself.

## Production techniques notebook — switch Gemini 3 candidates to thinking_level
Files: movie_ingestion/metadata_generation/generators/test_production_techniques.ipynb
Why: For Gemini 3 models, `thinking_level` is the more appropriate control than the legacy budget knob, and the notebook labels should match the actual control being tested.
Approach: Replaced the Gemini notebook candidate configs from nested `thinking_budget` values to nested `thinking_level` values (`minimal` and `low`) and renamed the former `gemini-3-none` label to `gemini-3-minimal` so the candidate name matches the configured level.
Design context: This keeps the Gemini candidate setup aligned with Gemini 3 terminology while preserving the same low-vs-lower-reasoning comparison shape in the notebook. The OpenAI candidates and the rest of the notebook logic were left untouched.
Testing notes: Verified the notebook JSON remains valid after the change. Did not execute the notebook or run tests.

## Production techniques generator: hardcode gpt-5.4-mini model candidate
Files: movie_ingestion/metadata_generation/generators/production_techniques.py, movie_ingestion/metadata_generation/batch_generation/generator_registry.py | Removed provider/model/kwargs params from generate_production_techniques(); hardcoded gpt-5.4-mini with reasoning_effort=low, verbosity=low as the sole model candidate (matching franchise generator pattern). Updated batch registry model string from gpt-5-mini to gpt-5.4-mini.

## Narrow production vector to filming locations + production techniques
Files: schemas/movie.py, movie_ingestion/final_ingestion/vector_text.py, implementation/prompts/vector_subquery_prompts.py, implementation/prompts/vector_weights_prompts.py, docs/decisions/ADR-001-eight-vector-spaces.md, docs/decisions/ADR-058-vector-text-formatting-conventions.md, docs/modules/classes.md, docs/modules/ingestion.md, docs/modules/schemas.md, search_improvement_planning/v2_data_needs.md, search_improvement_planning/v2_data_architecture.md, search_improvement_planning/new_system_brainstorm.md, docs/TODO.md
Why: The finalized production-techniques schema made the intended production vector much narrower than several code and doc surfaces still implied. The live builder, `Movie` helper surface, search-time prompts, and source-of-truth docs needed to converge on the same lean where/how-made definition.
Approach: Rewired `Movie` to load `production_techniques`, rewrote production vector text generation to emit only `filming_locations:` and `production_techniques:` in stable order, and updated `Movie.production_text()` to match that same shape. Realigned the production subquery/weight prompts to the new two-line contract, including removing negation-style production outputs in favor of positive technique cues only. Updated ADRs, module docs, planning docs, and the resolved TODO note so the active definition consistently excludes country/studio/language/decade/budget/source/franchise/broad-medium facts from the production vector.
Design context: This follows the user-approved production vector narrowing plan and the finalized `ProductionTechniquesOutput` prompt/schema boundaries: concrete making/rendering/capture methods plus scraped filming locations, with broader deterministic facts handled by Postgres/lexical paths instead.
Testing notes: Not run. Per repo instructions for this task, tests were neither read nor executed. Existing tests that still assert the older `production_text()` or production-vector shape will need a separate update pass if the user wants test work.

## Remove duplicated Movie production formatter
Files: schemas/movie.py, docs/modules/schemas.md, docs/decisions/ADR-056-movie-tracker-backed-schema-loader.md
Why: `Movie.production_text()` had become a line-for-line duplicate of `create_production_vector_text()`, which created an unnecessary second implementation of the same production-vector contract.
Approach: Deleted `Movie.production_text()` from `schemas/movie.py` and updated the remaining docs to point to `create_production_vector_text()` in `movie_ingestion/final_ingestion/vector_text.py` as the single production-vector formatter. A repo-wide non-test search confirmed there were no live runtime callers left to migrate.
Design context: This keeps production-vector formatting logic in the vector-text module, consistent with the current ingestion architecture and with the recent production-vector narrowing work.
Testing notes: Not run. Test files were not inspected or updated in this pass; any tests still referencing `Movie.production_text()` will need a separate test update when requested.

## Add source_material_type_ids, keyword_ids, concept_tag_ids columns to movie_card
Files: db/init/01_create_postgres_tables.sql, schemas/movie.py, movie_ingestion/final_ingestion/ingest_movie.py, db/postgres.py

### Intent
Add three new INT[] columns with GIN indexes to `movie_card` for deterministic filtering on source material types, overall keywords, and concept tags.

### Key Decisions
- All three columns use `INT[] NOT NULL DEFAULT '{}'` with `gin__int_ops` GIN indexes, matching the existing pattern for genre_ids/country_of_origin_ids/etc.
- `keyword_ids` maps from `imdb_data.overall_keywords` via the `OverallKeyword` enum (225 curated terms with stable IDs).
- `source_material_type_ids` extracts from `source_material_v2_metadata.source_material_types` (10 types with stable IDs in `SourceMaterialType` enum).
- `concept_tag_ids` merges both `concept_tags_metadata` and `concept_tags_run_2_metadata` generation runs via set union, then returns sorted deduplicated IDs. Uses `ConceptTagsOutput.all_concept_tag_ids()` which already filters out classification-only values (id=-1).
- Added `concept_tags_metadata` and `concept_tags_run_2_metadata` fields to Movie model, plus corresponding entries in `_METADATA_COLUMNS` and `_METADATA_FIELD_TO_MODEL` so the tracker loader populates them automatically.

### Testing Notes
Not run. Existing tests for `upsert_movie_card` and `fetch_movie_cards` will need parameter updates. New helper methods (`source_material_type_ids()`, `keyword_ids()`, `concept_tag_ids()`) on Movie need unit test coverage.

## Add box office status helper to Movie
Files: schemas/enums.py, schemas/movie.py, docs/modules/schemas.md, unit_tests/test_enums.py, unit_tests/test_schemas_movie.py

Why: We needed a deterministic helper on the tracker-backed `Movie` object to expose only clear commercial outcomes from the available budget and worldwide gross data, without introducing new persisted fields or overclaiming on ambiguous cases.

Approach: Added `BoxOfficeStatus` as a shared `StrEnum` with stable lowercase values (`hit`, `flop`) and implemented `Movie.box_office_status()` as a pure computed helper that reuses `resolved_budget()` and `resolved_box_office_revenue()` instead of duplicating source-selection logic. The helper returns `None` for missing/unparseable release dates, release years before 1980, missing/non-positive budget or gross, and the intentionally ambiguous 1.0x-3.0x ratio band. `HIT` requires both `ratio >= 3.0` and `budget >= 1_000_000`; `FLOP` uses `ratio <= 1.0` with no budget floor. Updated the schemas module doc to list both the enum and the new helper, and added focused unit coverage for enum stability plus the helper's threshold/source-precedence behavior.

Design context: The 1980 cutoff avoids known pre-1980 gross/re-release comparability problems. The `$1M` floor applies only to `HIT` so very small indie budgets do not produce misleading "hit" tags from trivial absolute grosses, while obvious financial failures still classify as `FLOP`.

Testing notes: `uv run python -m pytest unit_tests/test_enums.py unit_tests/test_schemas_movie.py::TestBoxOfficeStatus -q` passed (`39 passed`). A broader run of `unit_tests/test_schemas_movie.py` still has pre-existing unrelated failures for missing `Movie.title_with_original()` and `Movie.production_text()` methods.

## Persist box_office_bucket on movie_card
Files: db/init/01_create_postgres_tables.sql, db/postgres.py, movie_ingestion/final_ingestion/ingest_movie.py, docs/modules/db.md, docs/modules/ingestion.md, unit_tests/test_ingest_movie.py, unit_tests/test_postgres.py

Why: `Movie.box_office_status()` already computed a clear commercial outcome bucket, but `movie_card` only persisted `budget_bucket`. To make box-office status available to downstream Postgres consumers with the same shape as budget, the canonical card row needed a nullable `box_office_bucket` text column and the Stage 8 upsert path needed to populate it.

Approach: Added `box_office_bucket TEXT` to the bootstrap `movie_card` schema immediately next to `budget_bucket`. Extended `db.postgres.upsert_movie_card()` to accept and persist the new nullable text field in both insert and conflict-update paths, and extended `fetch_movie_cards()` to select and expose it in returned card dicts. In Stage 8 ingestion, mirrored the `budget_bucket` pattern exactly: call `movie.box_office_status()`, convert the enum to `.value` when present, and pass `None` otherwise. Updated module docs to reflect that `movie_card` now stores both budget and box-office buckets. For existing Postgres databases, the rollout remains manual SQL rather than app-managed schema mutation: `ALTER TABLE public.movie_card ADD COLUMN IF NOT EXISTS box_office_bucket TEXT;`

Design context: This intentionally persists the already-finalized helper output as readable text (`hit` / `flop` / `NULL`) rather than introducing a new ID mapping or any query-understanding/search-time behavior. The goal here is storage and ingestion parity with `budget_bucket`, not search semantics.

Testing notes: `uv run python -m pytest unit_tests/test_ingest_movie.py unit_tests/test_postgres.py -q -k 'upsert_movie_card or fetch_movie_cards or movie_card_init_sql_contains_popularity_columns or ingest_movie_card'` passed (`10 passed, 63 deselected`).

## Add movie_awards Postgres table and award_ceremony_win_ids column
Files: schemas/enums.py, movie_ingestion/imdb_scraping/models.py, movie_ingestion/imdb_scraping/parsers.py, movie_ingestion/final_ingestion/vector_text.py, db/init/01_create_postgres_tables.sql, db/postgres.py, movie_ingestion/final_ingestion/ingest_movie.py, unit_tests/test_enums.py, unit_tests/test_schemas_movie.py, unit_tests/test_ingest_movie.py, unit_tests/test_postgres.py

### Intent
Structured Postgres storage for award data that already flows through the pipeline (scraped from IMDB in Stage 4). Enables deterministic award lookups ("Oscar winners", "Best Picture nominees after 2000") via a new `movie_awards` table, and broad "award-winning" metadata filtering via a GIN-indexed `award_ceremony_win_ids` array column on `movie_card`.

### Key Decisions
- **AwardCeremony enum** (12 members) consolidates ceremony strings previously hardcoded in parsers.py and vector_text.py into a single source of truth in schemas/enums.py. Each member carries the IMDB event.text string as its value and a stable integer ceremony_id for Postgres.
- **AwardOutcome upgraded** from plain StrEnum to carry outcome_id (1=winner, 2=nominee). Breaking change to the enum definition but limited blast radius (models.py, parsers.py).
- **movie_awards table** uses integer IDs for ceremony and outcome but keeps `category` as TEXT. Category will be fuzzy-matched at search time via cached distinct values + rapidfuzz (small candidate set of ~150 categories). See search_improvement_planning discussion for rationale.
- **award_ceremony_win_ids SMALLINT[]** on movie_card with GIN index for broad metadata filtering (same pattern as genre_ids, keyword_ids). Cannot use gin__int_ops (requires INT[]) but column has at most 12 values per row so negligible impact.
- **batch_upsert_movie_awards** uses delete+insert (not per-row upsert) since awards for a movie are always ingested as a complete set.
- **Batched ingestion path** gets a separate inner SAVEPOINT for awards (between movie_card and lexical), matching the existing error isolation pattern.

### Planning Context
Design evolved through conversation: started with raw TEXT columns, moved to enum IDs after analyzing codebase patterns (posting tables, GIN arrays). Decided against a separate posting table since awards don't need fuzzy text resolution — the LLM outputs structured fields that map directly to enum IDs. The movie_awards table with integer columns and composite index serves both detail and search roles.

### Testing Notes
Tests added for: AwardCeremony enum stability/uniqueness, AwardOutcome ID stability, CEREMONY_BY_EVENT_TEXT lookup, AwardNomination.ceremony_id property, create_award_ceremony_win_ids helper, ingest_movie_awards flow, batch_upsert_movie_awards DB method, updated upsert_movie_card params. Run: `pytest unit_tests/test_enums.py unit_tests/test_schemas_movie.py unit_tests/test_ingest_movie.py unit_tests/test_postgres.py -v`

## Refinements to movie_awards implementation
Files: movie_ingestion/imdb_scraping/models.py, movie_ingestion/final_ingestion/vector_text.py, schemas/movie.py, db/postgres.py, movie_ingestion/final_ingestion/ingest_movie.py

### Intent
Four follow-up cleanups to the movie_awards implementation for robustness and readability.

### Key Changes
- **`AwardNomination.ceremony_id`** now returns `int | None` instead of raising `KeyError` for unknown ceremonies. Callers filter out `None` values before ingestion.
- **`_RECEPTION_AWARD_CEREMONY_DISPLAY`** now maps `AwardCeremony` enum members (not `.value` strings) to display names. Also serves as the ordering source, replacing the deleted `_RECEPTION_AWARD_CEREMONY_ORDER` that was causing a `NameError` in `_reception_award_wins_text()`.
- **`Movie.award_ceremony_win_ids()`** added as an instance method on `Movie`, following the pattern of `concept_tag_ids()`, `keyword_ids()`, etc. The `create_award_ceremony_win_ids()` helper in ingest_movie.py now delegates to it.
- **`batch_upsert_movie_awards()`** now accepts `list[AwardNomination]` directly and extracts fields internally, instead of requiring pre-extracted tuples.

## Split inv_person_postings into role-specific posting tables
Files: db/init/01_create_postgres_tables.sql, db/postgres.py, movie_ingestion/final_ingestion/ingest_movie.py, db/lexical_search.py, docs/decisions/ADR-011-data-store-architecture.md, docs/modules/db.md, docs/modules/ingestion.md

### Intent
Replace the monolithic `lex.inv_person_postings` table with 4 role-specific tables: `inv_actor_postings` (with billing_position + cast_size for prominence scoring), `inv_director_postings`, `inv_writer_postings`, `inv_producer_postings`. This is the first step toward V2 role-aware lexical search per `search_improvement_planning/v2_data_architecture.md`.

### Key Decisions
- **Only 4 tables, not 5**: Composers intentionally excluded — no posting table needed for search.
- **Actor billing_position derived from IMDB scraper list order**: `movie.imdb_data.actors` already preserves billing order (parsers.py:491). `billing_position` = 1-based index, `cast_size` = len(deduplicated actors).
- **Search compatibility preserved**: New `_build_people_union_cte()` unions all 4 tables for compound lexical search, maintaining the existing "people as one bucket" behavior. Role-specific routing is a future V2 change.
- **`PEOPLE_POSTING_TABLES` constant**: Module-level list in postgres.py used by both compound search and exclusion resolution, so new role tables can be added in one place.
- **`_normalize_name_list()` replaces `create_people_list()`**: Returns ordered list (not set) to preserve actor billing order. Deduplicates via seen-set.

### Testing Notes
- Unit tests in test_postgres.py and test_ingest_movie.py reference old function/enum names and will need updating.
- Key areas: `batch_insert_actor_postings` parallel unnest with billing_positions array, `_build_people_union_cte` SQL generation, `_normalize_name_list` order preservation.

## Add lex.inv_composer_postings support
Files: db/init/01_create_postgres_tables.sql, db/postgres.py, movie_ingestion/final_ingestion/ingest_movie.py
Why: Composer data was already scraped/stored in models but excluded from posting tables. Adding enables lexical search for composers.
Approach: Followed the identical pattern of director/writer/producer postings — same DDL, same batch insert function, same ingestion wiring. Added COMPOSER to PostingTable enum and PEOPLE_POSTING_TABLES so lexical search picks it up automatically.
