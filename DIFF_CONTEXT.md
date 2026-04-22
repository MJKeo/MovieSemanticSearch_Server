# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Ingestion-side updates for Stage 3 entity improvements: title_normalized + hyphen variants
Files: db/init/01_create_postgres_tables.sql, implementation/misc/helpers.py, db/postgres.py, db/lexical_search.py, movie_ingestion/final_ingestion/ingest_movie.py, search_v2/stage_3/entity_query_execution.py, movie_ingestion/backfill/rebuild_lexical_postings.py, movie_ingestion/backfill/__init__.py

### Intent
Land the ingestion side of the entity-endpoint improvements agreed in
[search_improvement_planning/entity_improvement.md](search_improvement_planning/entity_improvement.md).
Two concrete wins: (1) symmetric title matching via a new
`movie_card.title_normalized` column (Stage 3 title_pattern now ILIKEs
a normalized column with both sides normalized at ingest/query
identically — closes the diacritic/case gap); (2) deterministic
hyphen-variant resolution so `"spider-man"` / `"spider man"` /
`"spiderman"` all resolve to the same credit without any LLM reasoning.
Retired the v1 title-token SQL objects that were no longer read by
search_v2.

### Key Decisions
- **`title_normalized` column on movie_card, backed by trigram GIN
  (contains) + text_pattern_ops btree (starts_with).** Both indexes
  live on the normalized column so Postgres can pick the right one per
  match mode. Stage 3's `_execute_title_pattern` now calls plain LIKE
  (not ILIKE): since both sides are already normalized, ILIKE is
  redundant and the indexes become usable.
- **Hyphen expansion helper lives in `implementation/misc/helpers.py`
  as `expand_hyphen_variants(normalized) -> list[str]`.** Takes an
  already-normalized string and returns `{with-hyphen, hyphen→space,
  hyphen→empty}` (deduped). Symmetric between ingest and query so
  callers on either side can compose with `normalize_string`.
- **Ingest expands variants, posting tables keep prominence math
  honest.** `ingest_lexical_data` now computes cast_size /
  character_cast_size from distinct normalized names BEFORE expansion,
  freezing the denominators. Each distinct name then contributes every
  hyphen variant as its own `term_id` in `lex.lexical_dictionary`
  (and `lex.character_strings` for characters). Posting rows share the
  origin name's billing_position so variants can't create phantom
  cast members or shift prominence.
- **`batch_insert_actor_postings` /
  `batch_insert_character_postings` now take parallel `term_ids` +
  `billing_positions` lists** instead of auto-generating positions
  from list index. Required for variant expansion to preserve billing.
  Binary role postings (director/writer/producer/composer) stay on
  `(term_id, movie_id)` and don't carry billing metadata.
- **V1 title-token infrastructure dropped.** Removed
  `lex.title_token_strings`, `lex.inv_title_token_postings`,
  `lex.title_token_doc_frequency`, `movie_card.title_token_count`
  from the init SQL, along with every helper in `db/postgres.py` that
  read/wrote them (`batch_upsert_title_token_strings`,
  `batch_insert_title_token_postings`, `fetch_title_token_ids`,
  `fetch_title_token_ids_exact`, `refresh_title_token_doc_frequency`,
  `_build_compound_title_ctes`, `_get_title_token_tier_config`,
  `_build_title_token_query`, the tier-config dataclasses and base
  query constant, and `PostingTable.TITLE_TOKEN`). V1
  `db/lexical_search.py` still imports — the deleted helpers are now
  module-local stubs returning `{}` so the module loads; title-token
  exclusion in v1 is a no-op. Full v1 retirement remains a separate
  PR per the plan.
- **Backfill script structured as: schema migration (ALTER + DROP +
  indexes) → TRUNCATE role posting tables → per-movie rebuild from
  `movie_card` rows.** Shared registries (`lex.lexical_dictionary`,
  `lex.character_strings`) intentionally not truncated — they're
  shared with franchises/studios/awards and their UNIQUE constraints
  make re-upserts safe; orphaned term_ids are harmless. Per-movie
  work reuses `ingest_lexical_data(movie, conn)` verbatim, so the
  backfill and main ingest paths can't drift in posting shape. CLI
  has `--schema-only`, `--dry-run`, `--max-movies N`.

### Design context
Follows the companion plan at
[search_improvement_planning/entity_improvement.md](search_improvement_planning/entity_improvement.md).
Convention for normalized-at-ingest / normalized-at-query symmetry
matches
[docs/conventions.md](docs/conventions.md) "String normalization runs
identically at ingest and query time." Backfill structure mirrors
[db/migrate_split_franchise_columns.py](db/migrate_split_franchise_columns.py):
schema migration inside one transaction, per-movie fan-out bounded
by a semaphore below the pool's max_size, counters surfaced at the
end.

### Testing notes
- Not run against the live DB. Backfill must run after this
  changeset lands and before search_v2 Stage 3 title_pattern is
  exercised in production.
- Unit tests likely to require updates (deferred per test-boundaries
  rule): `test_ingest_movie.py`, `test_postgres.py`,
  `test_lexical_search.py` — they reference the removed
  title-token helpers and the old `batch_insert_actor_postings` /
  `batch_insert_character_postings` signatures.
- Verification steps in the plan file
  (`.claude/plans/mighty-sauteeing-cookie.md`) cover: helper unit
  behavior in a REPL, smoke slice with `--max-movies 100`, spot-check
  that a Spider-Man credit resolves via `"spiderman"` against the
  rebuilt character postings, and Amélie/normalized-title sanity for
  title_pattern.
- Smoke-tested module imports for `db.postgres`, `db.lexical_search`,
  `db.search`, `movie_ingestion.final_ingestion.ingest_movie`, and
  the new `movie_ingestion.backfill.rebuild_lexical_postings`.

## Stage 2A prompt content: decompose-first flow + `interpret` verdict + ranking-based fusion
Files: search_v2/stage_2a.py

### Intent
Empirical probing against feedback queries ("Mindless action", "Tarantino boomers love", "Main character energy", "Indiana Jones runs from the boulder", "Popcorn movies", "Movies with soul") surfaced four prompt-content failure modes that the schema alone could not fix: cross-family fusion (semantic + keyword collapsed into one slot), hallucinated capabilities (invented "demographic popularity metrics" against the metadata family), idiom under-expansion (slang passed through as a single literal unit), and compound best_guess strings (one interpretation mashing multiple families). Rewrote the prompt content — verdict taxonomy, reasoning flow, endpoint grounding, fusion criterion, examples, output micro-format — to mechanically prevent each failure mode. No schema changes.

### Key Decisions
- **Replaced `best_guess` with `interpret` verdict.** `interpret` emits 1+ retrievable atoms, each tagged with exactly one family. Cardinality reveals itself: one broad atom if the phrase resolves to one concept, multiple atoms if it genuinely spans several. Each atom is single-family by construction, eliminating compound-string failures. Descriptive language (plot events, tonal words, archetypes, named genres) explicitly stays `literal`.
- **Decompose-first-then-group reasoning flow.** Pass 1 commits per-phrase verdicts (emitting atoms into inventory). Pass 2 operates on atoms, asking fuse-vs-split. This separation forces decomposition before grouping, so a phrase like "popcorn movies" that warrants multi-atom decomposition cannot be swallowed into a single slot before its sub-structure is exposed.
- **New fusion criterion: same-family AND ranking-style qualification.** Replaced the "stacked adjectives fuse" heuristic. Two atoms fuse only when (a) they sit in the same retrieval family AND (b) they jointly define a ranking gradient where "more of both" means "better match" — the preference-vs-dealbreaker distinction Step 2B already uses. Cross-family atoms never fuse, even when both are ranking-style. "If in doubt, split."
- **Endpoint descriptions rewritten in user-facing capability language.** Per-attribute bullets for metadata (10 attributes, each with an explicit limit — popularity/reception marked global-only, no per-demographic slicing). Per-category bullets for keyword (closed taxonomy, no value enumeration). Per-vector-space bullets for semantic (8 spaces). Each family names what it CANNOT do; directly antidotes the "demographic popularity metrics" hallucination.
- **Boundary examples illustrate principles, not reproduce test queries.** Seven worked examples cover descriptive-stays-literal, multi-atom interpret decomposition, cross-family split, same-family ranking fusion, same-family independent-filter split, fold-into, and evaluative breadth preservation. None of the example phrases come from any probed query — avoids teaching the model to pattern-match on test-query wording.
- **Output micro-format for `unit_analysis` updated.** Interpret emits its atoms as indented sub-lines under the verdict, with a single-family tag per atom. Sub-dimension (e.g., `metadata, popularity` or `semantic, plot_events`) is optionally cited after the family name.

### Design context
Design rationale mirrors the Step 1 retrofit patterns (see search_improvement_planning/steps_1_2_improving.md): per-item verdicts with first-class skip/translate/fold, field-order scaffolding, brevity caps with explicit word counts, principle-based constraints rather than failure catalogs. Schema (Step2AResponse, PlanningSlot, validate_partition_completeness) is unchanged — the discipline lives inside the free-form `unit_analysis` and `slot_analysis` traces.

### Testing notes
- Verified import smoke: all three branch-dynamic system prompts assemble (~21k chars each after review fixes).
- Reran feedback probe (/tmp/probe_step_2a.py) against all 6 queries. All prior failure modes resolved: Mindless+action split cleanly cross-family; Tarantino-boomers interprets into global metadata atom + semantic demographic-appeal atom (no invented per-demographic metadata); Main-character-energy fires interpret with cross-family atoms; Indiana Jones boulder stays literal (no title-pinning); Popcorn/Soul produce single-family atoms.
- Known expected breakage unchanged: search_v2.stage_2::run_stage_2 still raises, unit_tests/test_search_v2_stage_2.py still fails, notebook cell 4B still raises on .concepts — all pending the Step 2B rework.

### Review fixes applied
- **Fusion rule tightened with a same-attribute-within-multi-attribute-family clause.** Prior rule was "same family + ranking-style". Observed inconsistency in the probe (Big-budget + blockbuster fusing in one run, splitting in another) showed the model treating metadata as one fusion unit despite the downstream constraint that each metadata expression targets exactly one column. Added: fusion additionally requires same sub-dimension within metadata (10 attributes) and keyword (7 categories). semantic remains a single fusion unit since its vector spaces can be queried together in one slot. Post-fix probe confirms Big-budget + blockbuster now consistently split.
- **Module-header comment refreshed** to reference the interpret verdict and principle-illustrating example framing (prior text still said "best-guess" and "paired failure → fix").
- **Example 1 sub-dimension corrected:** changed "heist" → "con-artist" so the `plot_archetype` sub-dimension maps to a real PlotArchetypeTag enum member (REVENGE/UNDERDOG/KIDNAPPING/CON_ARTIST).
- **Example 6 completed:** added the fold-target's own interpret verdict line with two cross-family atoms, so the example covers the full fold→interpret→inventory pipeline instead of stopping at the fold_into lines.
- **Placeholder name unified:** `<extra markers>` in the PRIMARY output-markers section renamed to `<markers>` to match the format template in `_OUTPUT_GUIDANCE_HEAD`.

## One-off: tag Joker (2019) with lineage='joker'
Files: scripts/fix_joker_lineage.py | UPSERTs public.movie_franchise_metadata for movie_id=475557 with lineage='joker'; character is credited as Arthur Fleck and the franchise generator had left the lineage null, so the franchise path couldn't retrieve it on "joker movies" queries. Only the lineage column is written — lex.franchise_entry/token and movie_card.lineage_entry_ids will be refreshed by the upcoming franchise-path rebuild.

## Backfill script: rebuild lex.inv_production_brand_postings from tracker
Files: movie_ingestion/final_ingestion/rebuild_production_brand_postings.py
Why: Brand registry retune (see entry below) invalidated every posting in `lex.inv_production_brand_postings` but raw inputs (IMDB `production_companies` + TMDB `release_date`) still live in the tracker — so no full re-ingestion is needed to refresh.
Approach: TRUNCATE the table, SELECT `(tmdb_id, production_companies, release_date)` from tracker for status=INGESTED via LEFT JOINs, derive release_year from the YYYY prefix (mirrors `ingest_production_data`), run `resolve_brands_for_movie`, hand results to `batch_insert_brand_postings`. Commit every 500 movies for progress visibility and crash safety. Per-movie `try/except` swallows individual failures so one bad row doesn't sink the run. Structure mirrors the sibling `rebuild_character_postings.py` script for consistency.
Testing notes: not run yet — user runs ingestion scripts themselves. No unit tests for this script (one-shot backfill).

## Production-brand registry: trim umbrellas to casual-viewer brand identity
Files: schemas/production_brands.py

### Intent
User flagged that "Disney movies" was returning films like "No Country for Old Men" because the registry included every corporate subsidiary a brand had owned. Retuned all 31 brand rosters around a different principle: a label belongs in a brand's roster only if a casual viewer typing "<brand> movies" would expect its films — not "does the parent corporately own this?"

### Key Decisions
- **Brand-identity test over corporate-ownership test.** The prior spec's "catalog recall over label purity" principle conflated Disney-owned-Miramax with Disney-branded films. Replaced with: KEEP labels the parent actively brand-promotes (Pixar "Disney/Pixar", Marvel Studios, Lucasfilm/Star Wars); DROP autonomous-identity acquisitions (Miramax, Searchlight, Touchstone, Hollywood Pictures, Blue Sky, New Line, HBO, DC-under-WB, Focus, DreamWorks Animation, Working Title, Nickelodeon, MTV, Sony Pictures Classics); DROP home-entertainment, foreign-region, and distribution-only credits.
- **Dropped labels remain findable via their own standalone brands.** Miramax, Searchlight, Touchstone, New Line, DC, etc. are already (or stay) in the registry as their own brand entries, so "Miramax movies" still works — the change only stops those films from leaking into their former parent's umbrella.
- **Multi-brand tagging preserved where legitimately co-branded.** Pixar films still tag DISNEY + PIXAR (Disney actively markets "Disney/Pixar"); post-2022 MGM films still tag MGM + AMAZON_MGM. Removed only spurious multi-tags from autonomous sub-labels.
- **Per-brand decisions delegated to 31 parallel research subagents** (one per brand) to apply the principle with web research. Accepted their KEEP/DROP recommendations wholesale. Accepted ADD suggestions for clear IMDB surface-string variants (`Lucasfilm Ltd.`, `Lucasfilm Animation`, `A24 Films`, `A24 Films LLC`, `Neon Rated`, `Studio Ghibli, Inc.`, `Miramax Films`, `DreamWorks Animation SKG`, `PDI/DreamWorks`, `Illumination` post-2018 rename).
- **Module docstring rewritten** to document the new curation principle and update the "Miramax under both MIRAMAX and DISNEY" example (now only MIRAMAX).

### Planning Context
Brand registry background is in `search_improvement_planning/production_company_tiers.md`. The earlier registry was built by 48 parallel research subagents using the inverted principle; this retune replaces their member-list decisions but preserves the overall tier structure (24 + 7 brands), slug/ID assignments, and year-window mechanics.

### Testing Notes
- `unit_tests/production_brand_spec_dates.py` — this file is described as the "authoritative copy" in the registry docstring and is heavily out of sync now. It will need to be regenerated to match the new rosters. **Not touched per test-boundaries rule.**
- `unit_tests/test_brand_resolver.py` and `unit_tests/test_production_brands.py` — likely to fail against the new registry. Flag for the user to regenerate fixtures.
- `schemas/production_brand_surface_forms.py` reads dynamically from the enum, so it picks up changes automatically.
- Import-time assertions in `_build_and_validate_registry()` pass. Spot-check confirmed: `Miramax` surface → MIRAMAX only (was previously [MIRAMAX, DISNEY]); `Pixar` → DISNEY + PIXAR (correctly preserved co-branding); `Searchlight Pictures` → SEARCHLIGHT only.

## Planning doc: capture session learnings on Stage 1 + transfer to Step 2A/2B
Files: search_improvement_planning/steps_1_2_improving.md | Saved the session's prompt-engineering learnings into the working planning doc. Updates: extended `Step 1's True Role` with creative-spin sub-job; added new bullets to `Working With Small LLMs` Schema-Side (trace micro-format requirement, field-order vs validators, separate-class-for-distinct-semantics) and Prompt-Side (per-item verdicts beat global summaries, skip verdict prevents over/under emission, per-field grounding requirements, worked positive-shape examples, named conceptual distinctions with behavioral tests, hedging-connective tell, evaluative-word substitution as hidden enrichment, explicit brevity word caps); added new top-level section `What We Learned From The Step 1 Prompt Iteration` mirroring the existing schema-changes section; added new top-level section `Implications For Step 2A and 2B` translating the patterns into concrete hypotheses (per-item verdict for ingredient inventory, concrete grounding per ingredient, vagueness-vs-ambiguity at concept level, skip verdict for incoherent concepts, intent-anchoring rule for expressions); updated `Working Hypothesis Going Forward` to reflect Stage 1's current good state and name the highest-leverage candidate interventions for Stage 2A/2B. No code changes.

## Stage 1 prompt: brevity rule on creative_spin_analysis parentheticals
Files: search_v2/stage_1.py | User flagged that creative_spin_analysis was getting verbose — parentheticals were running 15-20 words as full explanatory sentences (e.g. "(allows the user to browse the rest of the franchise if they were looking for more than just the specific scene)"). The readings trace stayed tight on its own (~7-10 word parentheticals modeled by examples) but spins didn't pick up the same discipline. Added one explicit "Brevity:" bullet to the creative_spin_analysis Rules section: parentheticals are brief labels at most ~8 words, not explanatory sentences. Spot-check on 4 queries confirmed parentheticals tightened to 6-10 words. Bonus: Indiana Jones description-identification query now correctly emits spin_potential: none rather than a stretched franchise spin (a separate edge case I noticed — description-based ID targets one specific movie, not a broad set).

## Stage 1 prompt: preserve user evaluative wording in intent_rewrite
Files: search_v2/stage_1.py | Follow-up tweak to the creative_alternatives change. Debug runs showed the model substituting `"Best Christmas movies for families"` → `"Highly rated Christmas movies suitable for families"` — `"Best"` is deliberately broad (covers rating, popularity, critical acclaim, family-tested favorites) and `"highly rated"` picks one specific interpretation. The existing `_INTENT_REWRITE_DISCIPLINE` had "keep vague terms vague" as a positive rule but the model was treating substitution as clarification. Added one explicit "Do NOT" bullet covering evaluative-word substitution (best/top/great/good/favorite/classic) plus a worked Best-Christmas-Christmas example. Reran 21-query debug script: primary now rewrites as `"Best Christmas movies for families"` (verbatim), spins preserve `"Best"` too (`"Best animated..."`, `"Best modern..."`); same pattern for `"good horror movies"` → `"Good horror movies"`. No regressions observed in other buckets.

## Stage 1: creative_alternatives — productive sub-angle spins on broad primary intents
Files: schemas/flow_routing.py, search_v2/stage_1.py, search_improvement_planning/debug_stage_1.py, search_improvement_planning/stage_1_debug_output.json

### Intent
User testing surfaced that broad single-intent queries like `"Best Christmas movies for families"` produced a clean primary but no useful exploratory branches — the prior change made `alternative_intents` strictly about competing readings of the user's words, which correctly excluded "spin on a broad set" cases. Adding a separate `creative_alternatives` field surfaces productive sub-angles (animated / modern-streaming / heartwarming-non-traditional) within the primary's broad intent, without polluting the disciplined alternative_intents semantics.

### Key Decisions
- **Done in Stage 1, not as a separate LLM call.** Considered a parallel/serial second call. Rejected: a parallel call on the raw query duplicates routing work and risks the spin call interpreting the query differently from Stage 1; a serial call adds latency. Doing it in Stage 1 with a dedicated trace field at the END of the schema means structured-output generation has already committed primary + true alts before the spin field is generated, decoupling the reasoning by construction.
- **Separate field (`creative_alternatives`), not folded into `alternative_intents`.** Semantic-clarity reason: alternatives are different *readings* of what the user asked for; spins are productive *narrowings* of a single intent. Mixing them in one list means downstream consumers can't render `"Did you mean..."` differently from `"You might also like..."`, and the prompt discipline for each diverges.
- **Separate `CreativeSpin` class, not reusing `AlternativeIntent`.** Same shape (intent_rewrite/flow/display_phrase/title) but with `spin_angle` replacing `difference_rationale`. This keeps the type signal at the edge for downstream code and preserves the per-class semantic.
- **No new validators per user direction.** The reused `_validate_title_for_flow` helper applies to `CreativeSpin` (same flow/title invariant as the other intent classes), but no new validation logic is introduced. The "spins always emit standard flow" rule is enforced by prompt only.
- **Existing `alternative_intents` semantics tightened.** Removed "or open-endedness" from `_TASK_AND_OUTCOME` and dropped the "broad request where an adjacent exploratory branch would clearly add value" bullet from `_BRANCHING_POLICY` — both cases now belong to spins. Added a one-line clarification to the readings-trace rules: readings are competing interpretations, productive narrowings belong to spins.
- **Cap of 2 spins, matching `alternative_intents`.** Soft "be more conservative when alternative_intents already exist" rule in the prompt (no validator). Empirically this rule is currently weak — see Testing Notes.
- **Trace field (`creative_spin_analysis`) mirrors the readings-enumeration pattern.** `spin_potential: <high|low|none>` line plus per-candidate verdicts (emit / skip). Same per-item commitment pattern that worked for the readings trace in the prior change.

### Planning Context
Decision history is in conversation context; relevant background in `search_improvement_planning/steps_1_2_improving.md`. User explicitly chose: in-Stage-1 placement, separate field, separate class, max 2 spins, soft "be more conservative when alts exist" rather than hard cap, no validators.

### Testing Notes
Ran `python -m search_improvement_planning.debug_stage_1` against 21 queries (added two new buckets: `spin_candidates` and `spin_negative_controls`).

**Working as designed:**
- All 5 `spin_candidates` queries (Christmas family, Disney classics, good horror, date night, feel-good comedies) produced 0 true alts + 2 spins. The user's original failure case (`"Best Christmas movies for families"`) now emits animated + modern-era spins.
- All 3 negative controls produced 0 spins with the correct skip rationale: narrow filters, exact_title flow, similarity flow.
- Reading-ambiguous queries that coexist with spins (Disney millennial, gen z horror, Y2K vibes) still produce true alts AND now also spins — coexistence working.

**Concerns worth follow-up (not blockers):**
1. **Spin emission is aggressive when alts already exist.** Most reading-ambiguous queries now produce 1 primary + 1 alt + 2 spins = 4 branches. The "be more conservative when alternative_intents are populated" prompt rule is being interpreted as guidance, not constraint — the model still emits 2 spins. May want to tighten this if 4-branch outputs feel cluttered in production.
2. **Semantic-vagueness queries get spins.** `"cozy movie for tonight"` and `"something feel-good"` — explicitly classified as semantic vagueness (no alt branching) — still get 2 spins because the underlying set is broad. This isn't wrong (the spins are useful sub-angles) but it does mean spin_potential and the vagueness/ambiguity axis are independent. Worth confirming this matches user expectations.
3. **Mild `intent_rewrite` enrichment regression.** `"Best Christmas movies for families"` rewrote primary as `"Highly rated Christmas movies suitable for families"` — `"Highly rated"` is a proxy-trait enrichment of `"Best"` that the prompt forbids. The hedging-ban bullet from the prior change is intact but didn't catch this. Watch for similar drift.

Stage 1 prompt grew from ~16KB to ~21KB. Latency on the 21-query batch run: 19.9s (vs. 11.0s on the 13-query pre-spin run) — proportional to query count, no obvious per-call latency bloat.

## Stage 1 prompt: readings-enumeration trace + vagueness-vs-ambiguity split
Files: search_v2/stage_1.py, search_improvement_planning/debug_stage_1.py, search_improvement_planning/stage_1_debug_output.json
Why: Debug runs against `"Disney millennial favorites"` and similar "generation/demographic + preference" queries reproduced a systemic Stage 1 failure: the model would cite ambiguity in the `ambiguity_analysis` trace but emit zero `alternative_intents`, instead hedging the competing readings into the primary `intent_rewrite` with "or"/"and" connectives. Four queries showed the exact same shape (see `search_improvement_planning/stage_1_debug_output.json` for the pre-change run). Root cause was a definitional gap: the prompt conflated semantic vagueness ("cozy", "prestige" — soft edges on one reading) with reading ambiguity (phrases with multiple distinct retrieval targets), and the `ambiguity_analysis` format accepted category labels ("vibe/preference", "broad nostalgia") instead of forcing concrete reading enumeration.
Approach: Four prompt-only changes, no schema changes and no validators (user explicitly rejected adding pydantic validators). (1) Added new `_VAGUENESS_VS_AMBIGUITY` section right after `_CORE_PRINCIPLES` that names the two failure modes and gives a behavioral test ("would the readings produce different movies?"). (2) Replaced the `ambiguity_analysis` trace format from `main=...; ambiguity=...; alt=...` with a `readings:` enumeration where each line has a concrete retrieval target and a per-reading verdict (primary / emit as alt / skip). The "skip" verdict preserves correct behavior on `"Up"`-style queries where a fragment reading exists but isn't worth emitting. (3) Added `"Do not hedge with or/and to join meaningfully different retrieval targets"` to `_INTENT_REWRITE_DISCIPLINE` with a worked Disney-millennial example. (4) Reworked the `"Disney live action movies millennials would love"` boundary example to show concrete primary + alt rewrites (previously told the model alternatives "may" vary a phrase without showing what they look like), and added a new `"Up"` boundary example that explicitly demonstrates the identified-but-skipped case.
Design context: See `search_improvement_planning/steps_1_2_improving.md` for the broader Stage 1/2 iteration context. Decision against schema changes traces back to the doc's explicit learning: fields without downstream consumers are "expensive prompt baggage" (`ambiguity_level`, `hard_constraints`, `ambiguity_sources` were removed for the same reason). The per-reading-verdict structure inside the single `ambiguity_analysis` field gives the model a commitment point without adding new output fields. Decision against mechanical coupling between reading count and alt count traces back to the user's feedback that the `"Up"` case (single-word known title, no useful alt) is the correct behavior despite the model technically identifying two readings — judgment must stay in the model, not the schema.
Testing notes: Ran `python -m search_improvement_planning.debug_stage_1` pre- and post-change against 13 queries in 6 buckets. Post-change: all four original Pattern A failures (Disney millennial favorites, gen z horror favorites, 80s kids would love, movies dads like) now emit alternatives with concrete reading enumerations. Y2K vibes no longer hedges in the primary rewrite. Clean controls (Inception, Tom Cruise 90s, cozy movie for tonight, something feel-good, arthouse thriller) preserved zero-alt behavior. `"Up"` correctly enumerates and skips the fragment reading. `"Scary Movie"` and `"prestige drama"` still branch correctly on title-vs-collection ambiguity. Prompt grew from ~12KB to ~16KB — modest given the scope of the fix. Stage 1 is pinned to Gemini 3 Flash with thinking disabled; tested under that config.

## Award query-side review fixes
Files: implementation/misc/award_name_text.py, search_v2/stage_3/award_query_generation.py | Two doc-staleness fixes surfaced by `/review-code`: (1) section-header comment in the prompt file still claimed "exact stored strings are still required" while the rewritten body said the opposite — rewritten to describe the shared tokenizer + posting index; (2) module docstring in `award_name_text.py` still said "Applied symmetrically at ingest and query time" and "DF-ceiling decision will be made empirically" — both stale since ingest writes every token and query now applies a hand-curated droplist. Tightened `tokenize_award_string`'s docstring so its "no stoplist here" rationale points at the query-side helper. No behavior changes.

## Search V2 concept-revamp follow-up: prevent browse fallback on total Step 2B loss
Files: search_v2/stage_2.py, unit_tests/test_search_v2_stage_2.py, search_v2/test_stage_1_to_4.ipynb | Review surfaced a wrong-results regression: if Step 2A extracted concepts but every Step 2B call failed or was dropped, Stage 2 returned an empty concept list and Stage 4 silently treated the branch as `BROWSE`. Tightened `run_stage_2()` so that "Step 2A found concepts, but all Step 2B concepts were dropped" now raises instead of degrading into generic browse results. Added a focused unit test for that failure mode and updated the V2 notebook’s browse-flow message to reflect the current popularity-based fallback wording rather than the removed prior-based seed path. Intentionally did not add duplicate-concept dedupe logic yet per user direction; that remains an observed-risk item rather than implemented behavior.

## Search V2 Step 2 concept revamp + Stage 4 concept-aware scoring
Files: schemas/query_understanding.py, search_v2/stage_2.py, search_v2/stage_4/__init__.py, search_v2/stage_4/flow_detection.py, search_v2/stage_4/orchestrator.py, search_v2/stage_4/scoring.py, search_v2/stage_4/types.py, search_v2/stage_4/priors.py, db/postgres.py, search_v2/test_stage_1_to_4.ipynb, unit_tests/test_search_v2_stage_2.py, unit_tests/test_search_v2_stage_4.py
Why: Implement the Step 2 revamp plan so V2 query understanding reasons in concepts with multiple expressions, remove the flawed quality/notability priors from the V2 runtime, and make Stage 4 aggregate sibling dealbreakers at the concept level without changing Stage 3’s per-expression translation/execution API.
Approach: Replaced the flat Step 2 schema with `Step2AResponse`, `QueryConcept`, and `RetrievalExpression`; rewrote `run_stage_2()` into a two-step orchestration flow (single Step 2A call + parallel Step 2B per concept) with branch-failing Step 2A and concept-dropping Step 2B failures. Stage 4 now flattens expressions into tagged runtime items that preserve concept identity, uses expression-level candidate generation as before, sums per-concept maxima for inclusion-dealbreaker scoring, applies one max-based penalty per semantic-exclusion concept, and keeps preferences independent even when they originate from the same concept. Removed prior inputs from the Stage 4 score path, deleted the prior helper module, delayed movie-card fetch to the final display-shaping step, and simplified exclusion-only browse seeding to `popularity_score DESC NULLS LAST, movie_id DESC`.
Design context: Follows `search_improvement_planning/step_2_revamp.md` decisions: concepts are the stable unit, multiple expressions under one concept do not become multiple full hits, mixed dealbreaker/preference concepts score on separate sides, priors are removed for now, and Step 2B failures drop only that concept. The lazy `search_v2.stage_4.__getattr__` change keeps package imports lightweight enough for the new unit tests without changing the public `run_stage_4` surface.
Testing notes: Added focused unit coverage for schema validation, Step 2A/2B orchestration, Stage 4 flow detection/tagging, concept-level inclusion/exclusion aggregation, and browse-seed ordering. Verified with `pytest unit_tests/test_search_v2_stage_2.py unit_tests/test_search_v2_stage_4.py -q` (16 passed) plus `python -m compileall search_v2 schemas db`. Cleared stale outputs and updated the Stage 1→4 notebook scoring cell so the manual QA harness matches the new concept-based Stage 2 contract and prior-free Stage 4 scoring path.

## Award name resolution: query-side cutover to the token index
Files: implementation/misc/award_name_text.py, db/postgres.py, search_v2/stage_3/award_query_execution.py, search_v2/stage_3/award_query_generation.py, schemas/award_translation.py, docs/TODO.md

### Intent
Land the query-side half of the Award Name Resolution plan (v2_search_data_improvements.md §Award Name Resolution), following the ingest-side landing from a prior session. Stage-3 award retrieval now normalizes + tokenizes LLM prize names, resolves them through `lex.award_name_token` → `award_name_entry_id` sets, and filters `public.movie_awards` on that pre-resolved id list. Exact TEXT equality on `movie_awards.award_name` is retired.

### Key Decisions
- **Stoplist applied at query time only, NOT bilaterally like franchise.** New `AWARD_QUERY_STOPLIST` constant + `tokenize_award_string_for_query` wrapper live alongside the unchanged ingest tokenizer. Ingest writes every token to `lex.award_name_token`; query drops the droplist before posting-list fetch. Rationale (planning-doc Principle #5): the ~600-entry award corpus is small enough that keeping every token in the index is free, and the asymmetry lets the droplist be revised from the DF MV without re-ingesting. Finalized droplist from the post-backfill top-25 DF scan: `{award, awards, prize, prizes, film, films, best, a, an, and, for, of, the}`. `special` / `mention` / `jury` deliberately excluded — they're domain-meaningful (Special Jury Prize, Honorable Mention, Grand Jury Prize).
- **`fetch_award_row_counts` signature changed: `award_names: list[str]` → `award_name_entry_ids: set[int]`.** WHERE predicate swapped from `award_name = ANY(%s::text[])` to `award_name_entry_id = ANY(%s::int[])`. Only caller is `execute_award_query`, confirmed via grep — no orchestrator-level fanout to update. Docstring rewritten; the old paragraph about "exact, un-normalized equality" / "case-folding here would silently zero out valid matches" is deleted (it was a statement of the pre-cutover invariant).
- **New `fetch_award_name_entry_ids_for_tokens` is a structural copy of `fetch_franchise_entry_ids_for_tokens`.** Single batched posting-list lookup; missing tokens omitted from the returned dict. Placed immediately before `fetch_award_fast_path_movie_ids` so the award helpers stay contiguous.
- **Executor early-exit protects against silent broadening.** If `spec.award_names` was populated but token intersection resolved to zero entry ids, `execute_award_query` returns an empty `EndpointResult` rather than dropping the predicate — matches the franchise executor's rule. The empty-set → `None` collapse on the DB-helper argument (`award_name_entry_ids or None`) keeps the "axis inactive" and "axis unpopulated" codepaths identical at the SQL layer.
- **Fast path untouched.** `_qualifies_for_fast_path` still gates on `spec.award_names` being null; the fast path is orthogonal to entry resolution.
- **Ceremonies section of the prompt intentionally left alone.** `CEREMONIES` uses `AwardCeremony(c).ceremony_id` enum lookup at the executor level — still exact-match — so "a one-character difference produces zero matches" remains accurate there. Only the `AWARD NAME SURFACE FORMS` section (now renamed `AWARD NAME BASE FORMS`) was rewritten to drop exact-match framing and explain the shared-tokenizer collapse behavior.
- **Registry file `schemas/award_surface_forms.py` left in place.** Every entry's `prize_names` tuple produces correct or desirably-umbrella behavior under token intersection (verified against the 13-entry audit). `render_award_name_surface_forms_for_prompt` still feeds the prompt as a ceremony-to-base-form anchor table, not as a closed vocabulary.
- **`ceremonies` null + `award_names` populated runs unchanged.** SQL drops the ceremony clause naturally; existing `scoring_mode`/`scoring_mark` formula produces the right answer (floor/1 → 1.0 on any match). No per-case override added.

### Planning Context
Plan file: `/Users/michaelkeohane/.claude/plans/lovely-whistling-nova.md` (overwritten — prior contents were the ingest-side plan). User pre-approved during plan review: keep `schemas/award_surface_forms.py` untouched (it's prompt-only, not used at runtime); do not rename the registry file or class; do not override `scoring_mode` when `ceremonies` is null; prompt should frame award_names as "official base form" rather than exact stored strings. Audited all 13 registry entries through the query tokenizer — every entry either resolves tightly (e.g., `"Golden Lion"` → `{golden, lion}`) or produces desirable umbrella behavior (e.g., `"BAFTA Film Award"` → `{bafta}` → sweeps the BAFTA family, narrowed by the ceremony filter).

### Testing Notes
- Per .claude/rules/test-boundaries.md, no test files touched. New/changed signatures that will need test updates:
  - `AWARD_QUERY_STOPLIST` + `tokenize_award_string_for_query` in `implementation/misc/award_name_text.py`.
  - `db.postgres.fetch_award_name_entry_ids_for_tokens` is new.
  - `db.postgres.fetch_award_row_counts` parameter rename `award_names` → `award_name_entry_ids` with corresponding type change.
  - `_dedupe_nonempty` deleted from `search_v2/stage_3/award_query_execution.py`; any test that imported it will break.
  - Early-exit branch in `execute_award_query` when `spec.award_names` is populated but entry resolution is empty.
- End-to-end validation to run: (a) `tokenize_award_string_for_query("BAFTA Film Award")` → `["bafta"]`; (b) straight- vs curly-apostrophe `Palme d'Or` both → `["palme", "dor"]`; (c) `Critics Week Grand Prize` vs `Critics' Week Grand Prize` specs return identical movie sets; (d) nonsense prize name with no ceremonies → empty `EndpointResult` (not silently broadened).

## Franchise query-side review fixes
Files: db/postgres.py, search_v2/stage_3/franchise_query_execution.py, search_v2/stage_3/franchise_query_generation.py | Three post-review refinements on top of the franchise query-side cutover below: (1) collapsed the two-step early-exit block in `execute_franchise_query` into per-axis guards, dropping the unused `has_structural_axis`/`*_requested`/`*_resolved` locals; (2) fixed two prompt wording issues in `_SEARCHABLE_AXES` (`"subgroup- only"` → `"subgroup-only"`, missing article fixed to `"a named-franchise spec"`); (3) branched `fetch_franchise_movie_ids` to pick its driving table per spec shape — structural-only → `movie_franchise_metadata` directly (no `mc` join); array-only → `movie_card`; mixed → `mc LEFT JOIN mfm`. `restrict_movie_ids` now binds to the driving table's `movie_id` column in each branch. Supersedes the "base table swapped to `movie_card mc`" note in the prior entry — the helper now picks between three shapes.

## Franchise resolution: query-side cutover to the token index
Files: schemas/franchise_translation.py, implementation/misc/franchise_text.py, db/postgres.py, search_v2/stage_3/franchise_query_execution.py, search_v2/stage_3/franchise_query_generation.py

### Intent
Land the query-side half of the Franchise Resolution plan (v2_search_data_improvements.md §Franchise Resolution), following the ingest-side landing from the prior session. Stage-3 franchise retrieval now normalizes + tokenizes LLM surface forms symmetrically with ingest, resolves them through `lex.franchise_token` → `franchise_entry_id` sets, and runs a GIN `&&` overlap against `movie_card.franchise_name_entry_ids` / `subgroup_entry_ids`. Exact TEXT equality on `movie_franchise_metadata.lineage` / `shared_universe` / `recognized_subgroups` is retired.

### Key Decisions
- **Field rename: `lineage_or_universe_names` → `franchise_or_universe_names`.** User chose the more user-facing "franchise" prefix over the original planning-doc proposal `franchise_names` — the explicit "or universe" keeps the combined-search-space signal visible in the schema so the prompt does not have to constantly reiterate it. The planning-doc reference to `franchise_names` is now superseded (noted in planning doc §Query-Time Resolution but not re-edited here — docs TODO).
- **Validator is now minimal.** Dropped the cross-field rule that `recognized_subgroups` requires `franchise_or_universe_names`. Subgroup-only specs like "trilogies" → `recognized_subgroups=["trilogy"]` are now valid, per user directive. `_validate()` retains only (a) structural-flag dedupe, (b) the "at least one axis populated" guard.
- **Stopword droplist lives inside the tokenizer, not the executor.** `FRANCHISE_STOPLIST = {the, of, and, a, in, to, on, my, i, for, at, by, with}` added to `implementation/misc/franchise_text.py`, applied on the single return line of `tokenize_franchise_string`. Mirrors `award_name_text.py`'s `AWARD_STOPLIST` shape exactly — symmetric ingest/query, one source of truth. Executor never sees a stopword, so no DF ceiling at query time (the DF materialized view stays only as a diagnostic for curating the list as the corpus grows).
- **Per-name intersection + cross-name union happens in Python over one batched DB response.** New `fetch_franchise_entry_ids_for_tokens(tokens)` mirrors `fetch_company_ids_for_tokens` but *without* the DF join (stopwords are already dropped upstream, so a DF filter would be redundant dead weight). Two Postgres round trips per spec regardless of name count. Missing token → name contributes empty (matches studio Phase-3 behavior at studio_query_execution.py:119-122).
- **`fetch_franchise_movie_ids` rewritten with entry-id-set inputs.** Base table swapped from `movie_franchise_metadata` to `movie_card mc`; `LEFT JOIN movie_franchise_metadata mfm USING (movie_id)` is added only when a structural axis (lineage_position / is_spinoff / is_crossover / launched_franchise / launched_subgroup) is active, so the common name-only query stays a single-table GIN scan. Array predicates: `mc.franchise_name_entry_ids && :A` and `mc.subgroup_entry_ids && :B`. `is_spinoff` / `is_crossover` / `launched_*` arg types changed `bool | None` → `bool` (the old "None = inactive" wiring was already collapsing to a boolean at the call site).
- **Executor early-exit protects against silent broadening.** If a textual axis was *populated* in the spec but token resolution produced an empty entry-id set, the DB helper would otherwise treat it as "axis inactive" and drop the predicate — silently broadening the result. Explicit early-exit in the executor: if any requested textual axis resolves empty, return an empty EndpointResult (per user directive #3: "if all tokens drop, treat as no results found"). A no-textual-axis spec with only structural predicates still passes through.
- **Prompt rewrite commits to query scope up front.** `concept_analysis` must now state scope (umbrella vs specific lineage vs sub-phase vs pure subgroup vs pure structural) before emitting field values. Mitigates the "Doctor Strange in the MCU" OR-broadening case (planning doc §Franchise-names OR semantics when user wanted AND) without a resolver-side subset-elimination rule — which is deferred until eval data shows the prompt-only mitigation misfires (planning doc Open Decision #2). Canonical Naming section also rewritten to drop the "alternate exact stored-form attempts" framing and replace it with the umbrella-vs-specific rule (broad form for umbrella queries; narrow form alone for specific-lineage queries).

### Planning Context
Plan file: `/Users/michaelkeohane/.claude/plans/cheeky-dancing-hamming.md` (overwritten — prior contents were the ingest-side plan, explicitly marked "query-side out of scope"). User pre-approved six scope points during plan review: field rename (`franchise_or_universe_names`), validator minimalism, empty-token-set → empty-result semantics, stopword-in-tokenizer placement, studios-pattern posting-list fetch, and multi-variant emission as the umbrella-sweep mechanism. Normalization symmetry invariant (docs/conventions.md) is what makes this cutover safe: the same tokenizer runs at ingest and query, so re-running the franchise backfill under the stopword-dropping tokenizer is a precondition for shipping (noted in plan Verification #1).

### Testing Notes
- Per .claude/rules/test-boundaries.md, no test files touched. New/changed signatures that will need test updates:
  - `FranchiseQuerySpec.franchise_or_universe_names` (renamed from `lineage_or_universe_names`).
  - `FranchiseQuerySpec._validate()` no longer raises when `recognized_subgroups` is populated without `franchise_or_universe_names`.
  - `tokenize_franchise_string` now drops `FRANCHISE_STOPLIST` entries — any fixture that depended on `the` / `of` / etc. surviving will need update.
  - New `db.postgres.fetch_franchise_entry_ids_for_tokens`.
  - `db.postgres.fetch_franchise_movie_ids` signature changed from `normalized_name_variations` / `normalized_subgroup_variations` (list[str]) to `franchise_name_entry_ids` / `subgroup_entry_ids` (set[int]); structural-flag args changed `bool | None` → `bool`.
- Manual verification: ran the tokenizer on the planning doc's worked-example strings — `The Lord of the Rings` → `{lord, rings}` (the/of dropped); `Phase 1` → `{phase, one}` (cardinal); `Marvel Cinematic Universe` → `{marvel, cinematic, universe}` (scaffolding kept); `Spider-Man` → `{spider-man, spider, man}` (hyphen expansion); `the` → `[]`; `Conjuring Universe` → `{conjuring, universe}`.
- **Precondition before shipping:** re-run the franchise backfill (from the prior session's plan) under the updated `tokenize_franchise_string` so stopwords are absent from `lex.franchise_token`. Sanity check after: `SELECT COUNT(*) FROM lex.franchise_token WHERE token IN ('the', 'of', 'and', ...)` should return 0.
- End-to-end regression matrix from the plan's Verification §3 (7 worked-example queries exercising umbrella sweep, narrow lineage, phase-specific, LOTR stopword drift, Jackson trilogy subgroup, subgroup-only "phase one", Conjuring umbrella) — to be run once the backfill precondition is met.

## Award name resolution: ingest-side landing of the token-index plan
Files: db/init/01_create_postgres_tables.sql, implementation/misc/award_name_text.py, db/postgres.py, movie_ingestion/final_ingestion/ingest_movie.py, backfill_award_name_entries.py, docs/TODO.md

### Intent
Land the ingest-side half of the Award Name Resolution plan committed to search_improvement_planning/v2_search_data_improvements.md. After this change, every `public.movie_awards` row carries a resolved `award_name_entry_id` pointing at a normalized-form registry in `lex.award_name_entry`, and a token inverted index (`lex.award_name_token`) plus DF materialized view are in place ready for the stage-3 query-side cutover. The exact-string comparison in `search_v2/stage_3/award_query_execution.py` is intentionally unchanged — it continues to serve queries until the follow-up PR flips it over, and the "do not normalize" comment stays in place for now (TODO tracked).

### Key Decisions
- **Schema placement in `lex.`** (not `public.`). Aligns with `lex.production_company` / `lex.franchise_entry`; the plan doc's unqualified SQL was a doc-level omission, corrected per user confirmation.
- **Entry table is `{id, normalized}` only — no `canonical_string`.** Studio and franchise entry tables duplicate the first-seen raw form for display, but `public.movie_awards.award_name` already preserves every raw form per-movie, so the extra column would be redundant. Normalized-only keeps the plan doc's intent verbatim.
- **`INT` identity, not `BIGINT`.** ~600 distinct award-name strings observed today; `INT` is sufficient and matches the column width on `movie_awards.award_name_entry_id`.
- **No FK from `movie_awards.award_name_entry_id` → `lex.award_name_entry`.** `movie_awards` is declared before the lex tables in `01_create_postgres_tables.sql`; adding the FK inline would create a forward reference. Matches the loose-reference convention already used for `ceremony_id`, `outcome_id`, and `category_tag_ids`.
- **Inline resolution inside `ingest_movie_awards`.** Entry + token writes happen in the same call that prepares the `movie_awards` upsert, and the resolved entry id is passed through to `batch_upsert_movie_awards` so it's stamped in the same INSERT — no write-then-patch UPDATE. Matches the user's "part of the main upsert" directive and mirrors how `ingest_production_data` / `ingest_franchise_data` feed ids into `upsert_movie_card`.
- **No stoplist at ingest.** `tokenize_award_string` emits every surviving token; the only ingest-time filter is lone-hyphen residue (matching studio). Stopword choice is deferred to query time, to be made after the DF bucket distribution from the first real backfill is visible — matching how studio and franchise stage the same call. The plan doc's Stage-B stoplist is therefore a query-side artifact, not an ingest-side one.
- **Curly-apostrophe fold scoped to `normalize_award_string`, not the shared helper.** `implementation/misc/helpers.normalize_string`'s apostrophe regex only covers ASCII U+0027 and modifier variants — it does NOT strip U+2018 / U+2019, so `Palme d\u2019Or` would otherwise normalize to `palme d or` (space) while `Palme d'Or` normalizes to `palme dor` (stripped). Fixed by pre-folding U+2018 / U+2019 → U+0027 inside `normalize_award_string` before delegating. Deliberately NOT fixed in the shared helper: every existing `lex.lexical_dictionary` / `lex.production_company` / `lex.franchise_entry` row was stamped under the old rule, so a broadening there would silently invalidate already-ingested keys. The fold must be mirrored on the stage-3 query side when that cutover lands (tracked in docs/TODO.md).
- **Backfill uses one bulk `UPDATE … FROM (VALUES ...)` rather than a per-movie loop.** The transform `normalize_award_string` can't be expressed in SQL (NFKD + digit-to-word are Python-side), but with only ~600 distinct raw strings the `(raw, entry_id)` mapping fits in a single statement. Server-side one round trip; no `asyncio` fan-out.
- **Wipe-before-rebuild in the backfill.** Prior runs may have used different normalization/tokenization rules (stoplist evolution, digit-to-word changes), so merging would leave stale rows. Same rationale as `backfill_production_brands_and_companies.py`.
- **DF materialized view created even though the DF ceiling is deferred** (plan doc Open Decision #1). Registered alongside studio/franchise in `cmd_ingest`'s post-ingest refresh block so bucket distribution is always available for the empirical review.

### Planning Context
Plan file: `/Users/michaelkeohane/.claude/plans/lovely-whistling-nova.md`. Design grounded in search_improvement_planning/v2_search_data_improvements.md §Award Name Resolution, which was re-edited earlier this session to drop `raw_name` and ceremony from the entry key and route cross-festival homonym disambiguation through the row-level `movie_awards.ceremony_id` filter instead. User confirmed three architectural questions (schema, column shape, backfill strategy) before plan write. Query-time cutover is explicitly scoped out and tracked in docs/TODO.md.

### Testing Notes
- Per .claude/rules/test-boundaries.md tests were not touched in this changeset. New/affected signatures that will need test updates in a dedicated pass:
  - `batch_upsert_movie_awards` now accepts an aligned `award_name_entry_ids: Sequence[int | None] | None` parameter and writes a new `award_name_entry_id` column.
  - `ingest_movie_awards` now makes three additional DB calls (entries, tokens, then the extended movie_awards upsert) rather than one.
  - Three new public functions in db/postgres.py: `batch_upsert_award_name_entries`, `batch_insert_award_name_tokens`, `refresh_award_name_token_doc_frequency`.
  - New module `implementation/misc/award_name_text.py` with `normalize_award_string` and `tokenize_award_string` (no stoplist — deferred to query time).
- Manual verification steps (in the plan file): DDL parse on a fresh docker-compose bring-up, normalization sanity checks (straight-vs-curly apostrophe, `BAFTA Film Award` → `["award", "bafta", "film"]`), backfill dry-run expects ~588 entries and zero NULL `award_name_entry_id` rows, fresh ingest on a single movie confirms the entry id writes in the same INSERT and the MV refresh runs at the end of `cmd_ingest`.
- Bucket review for the DF ceiling is the follow-up task — inspect `SELECT token, doc_frequency FROM lex.award_name_token_doc_frequency ORDER BY doc_frequency DESC LIMIT 20` after the backfill runs.

## v2 search planning: fleshed-out Award Name + Franchise resolution plans
Files: search_improvement_planning/v2_search_data_improvements.md

### Intent
Extended the v2 search data improvements doc (originally scoped to the studio resolver) with two new top-level design sections — `Award Name Resolution` and `Franchise Resolution` — porting the studio resolver's token-index approach to the two remaining stage-3 axes with the query-side-guesses-what-ingest-wrote problem.

### Key Decisions
- **No registry for either endpoint.** Studios' closed brand enum does not port. For award_name, the ceremony axis is already a closed enum and scoping posting-list lookups by ceremony substitutes for deterministic brand routing. For franchise, the open-vocabulary space (5,206 distinct lineages) is too large to maintain; ingest-side canonical-naming rules plus token intersection carry retrieval.
- **Symmetric normalization supersedes the `award_query_execution.py:62-83` "do not normalize" comment.** Apostrophe/diacritic/case drift in IMDB's award surface forms creates silent misses; the comment's stated risk of masking mismatches is smaller than the cost of the current strict-eq behavior.
- **Franchise lineage and shared_universe become one search array at retrieval time.** Per user direction: ingest-side LLM inconsistently places franchise names in one column or the other, so the query path must not require guessing which. `movie_franchise_metadata.franchise_name_entry_ids INT[]` holds the union; original TEXT columns preserved for debugging and non-search consumers.
- **Subgroups promoted to token intersection.** Per user direction — exact match currently splits `phase 1`/`phase one`/`snyderverse`/`snyder-verse` variants.
- **DF ceiling deferred for both** — decision made only after the first full ingest reveals the bucket distribution, matching how the studio design stages the same call.
- **Franchise-only cardinal number-to-word rule** on top of the shared ordinal rule (studios/titles use ordinal-only); `phase 1 → phase one` is the motivating case.

### Planning Context
Builds on the studio resolver design at the top of the same document. User explicitly rejected registry-based proposals for both endpoints in favor of the studio freeform/token-index path minus the brand-path half. Corrected a factual error from my earlier response: `Grand Prize of the Festival` at Cannes (pre-1955) is a genuinely different prize from the `Palme d'Or`, not a rename — written into the edge-case section accordingly.

### Testing Notes
Planning-only changeset; no code. Implementation will span new DB tables (`award_name_entry`, `award_name_token`, `franchise_entry`, `franchise_token`) + column additions (`movie_awards.award_name_entry_id`, `movie_franchise_metadata.franchise_name_entry_ids`, `movie_franchise_metadata.subgroup_entry_ids`) + ingestion Stages A–C per endpoint + stage-3 query execution rewrites.

## Stage-3 semantic endpoint: query execution
Files: search_v2/stage_3/semantic_query_execution.py (new), pyproject.toml

### Intent
Complete the Stage-3 semantic endpoint by adding the executor that takes the step-3 semantic LLM's `SemanticDealbreakerSpec` / `SemanticPreferenceSpec` output, runs the corresponding vector search against Qdrant, and returns an `EndpointResult`. Generation was already in place (`semantic_query_generation.py`); this closes the loop.

### Key Decisions
- **Two public entry functions, not one.** `execute_semantic_dealbreaker_query` and `execute_semantic_preference_query` because the spec types are disjoint and the scoring math (threshold-plus-flatten vs raw weighted-sum cosine) shares only low-level primitives. Forcing one entry point would require an isinstance fork at the top for no readability win.
- **`restrict_to_movie_ids` discriminates four scenarios.** `None` → D2/P2 (candidate-generating); `set[int]` → D1/P1 (score-only); empty set → short-circuit to `EndpointResult()`. Matches the sibling-executor contract and keeps the orchestrator-visible surface identical.
- **D2 uses a single Qdrant call.** The top-2000 probe doubles as both the elbow/floor calibration distribution AND the candidate pool — same query, same space, same limit. Departure from a naive reading of the proposal which described them as two distinct steps.
- **D2 does not cross-score against other dealbreakers' candidates** per user's explicit correction during planning. Each semantic dealbreaker scores only the movies it retrieved; missing dealbreakers contribute 0 to `dealbreaker_sum`, matching how deterministic dealbreakers already behave.
- **Movie_id is the Qdrant point ID.** Confirmed at `ingest_movie.py:671` (`PointStruct(id=movie_id, ...)`). Filtered-score lookups use `HasIdCondition(has_id=[...])`, not a payload `FieldCondition` — which would silently miss since no such payload field exists.
- **Pathology check uses range, not max|diff|.** The proposal's `max(y_diff) < 0.05` threshold was ambiguous against raw cosines — applying it to the EWMA-smoothed curve fires on ordinary distributions because per-step diffs are tiny (~0.003). Replaced with `max_sim − min_sim < 0.05`, which preserves the 0.05 threshold's operational meaning ("distribution is truly flat") and actually only fires when there is no signal.
- **Floor ratio raised to 0.65** per user direction (was 0.50 in the proposal's pathology fallback). Applied both in code and plan.
- **No elbow cache.** Deferred per user direction; every dealbreaker invocation pays for one unfiltered top-2000 query. Hook-point for a future Redis cache is `_detect_elbow_floor`'s callsite.
- **P2 fills cross-space cosines with targeted HasId lookups.** After unioning per-space top-N probes, each space's cosine map is re-checked for union members it missed and filled via one filtered Qdrant call per space (parallelized). Preference scoring combines every selected space for every candidate; cross-space fill is intrinsic to the preference design (unlike D2's cross-dealbreaker scoring, which is forbidden).
- **Embedding model is `text-embedding-3-large`** (matches `ingest_movie.py:86`). CLAUDE.md's reference to `text-embedding-3-small` is stale.
- **Retry contract mirrors siblings.** Scenario-level retry loop (1 retry); second failure logs at ERROR and returns empty `EndpointResult()` rather than raising.

### Planning Context
Plan file: `/Users/michaelkeohane/.claude/plans/3-whatever-it-s-such-noble-zephyr.md`. Design grounded in `search_improvement_planning/finalized_search_proposal.md` §Endpoint 6 (Semantic) and the Semantic Endpoint Finalized Implementation Decisions section. Preference weights reflect the recent rename from `primary`/`contributing` to `central`/`supporting` (weights unchanged: 2.0 / 1.0).

### Dependencies
Added `kneed>=0.8` to `pyproject.toml` for Kneedle elbow detection per the proposal's algorithm specification (`curve='convex', direction='decreasing', S=1, online=True`). Implementing Kneedle from scratch is ~60 lines with well-known pitfalls around the S-sensitivity; the library is MIT, pure-Python, and pulled `numpy` + `scipy` which were already present.

### Testing Notes
- `_detect_elbow_floor`, `_threshold_flatten`, `_weighted_cosine_score`, `_ewma` are pure and merit unit tests with fabricated distributions (sharp knee, bimodal, flat pathology, early-knee outlier triggering rank-10 safeguard, empty input).
- Scenario helpers should be tested with mocked `qdrant_client` and `generate_vector_embedding` per the pattern in `unit_tests/test_vector_search_timing.py`.
- End-to-end: run a D2 zombie dealbreaker against the live collection in `search_v2/test_stage_3.ipynb` and eyeball whether top-scored movies are zombie-centric. Then re-run as D1 with `restrict_to_movie_ids={known_zombie, known_non_zombie, ...}` and confirm scores go ~1.0 vs ~0.0.
- Failure path: pass a closed `qdrant_client` and confirm retry fires once, ERROR logs, and `EndpointResult(scores=[])` returns without raising.
- Edge: empty `restrict_to_movie_ids` short-circuits without any Qdrant or OpenAI call.

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

## Capture Step 2 revamp design direction
Files: search_improvement_planning/step_2_revamp.md
Why: Save the current redesign discussion as a durable planning artifact before implementation work begins.
Approach: Wrote a new planning doc that records the settled direction for splitting query understanding into Step 2A concept extraction and Step 2B expression planning, removes quality/notability priors from the revamped design, and proposes concrete output schemas for both substeps using concept-level scoring with multi-expression union behavior.
Design context: Grounded in search_improvement_planning/finalized_search_proposal.md, the current search_v2/stage_2.py design, and the latest brainstorming decisions in this session about concept-vs-expression modeling.
Testing notes: Planning-only docs change; no code or tests touched.

## Add positive-presence invariant to Step 2 revamp plan
Files: search_improvement_planning/step_2_revamp.md
Why: Preserve the existing direction-agnostic Step 3 contract while the Step 2 redesign is still being formed.
Approach: Updated the planning doc so dealbreaker-side and exclusion-side expressions are always phrased as the presence of an attribute, with inclusion/exclusion metadata carrying the direction. Also clarified that negative user preferences must be converted into positive ranking targets before reaching Step 3.
Design context: Aligns the Step 2A/2B redesign with the current Step 3 principle that endpoint descriptions should test for attribute presence and let direction be handled separately.
Testing notes: Planning-only docs change; no code or tests touched.

## Consolidate Step 2 revamp decisions into the working design doc
Files: search_improvement_planning/step_2_revamp.md
Why: The earlier draft still reflected several schema ideas we later rejected, including concept-level role labels, candidate-behavior hints, generic boost logic, and endpoint-specific scope fields. The design needed a full consolidation pass so later implementation work starts from the actual current decisions rather than an outdated midpoint.
Approach: Rewrote the working planning doc around the latest settled structure: Step 2A is now minimal concept extraction (`concept_inventory_analysis` + `concepts: list[str]`), Step 2B owns expression planning with expression-level `dealbreaker` / `preference` labels, exclusions are represented as dealbreakers with `exclude` polarity, and preferences explicitly emit `core` vs `supporting`. Removed generic boost/scope fields, moved prominence and lineage-vs-universe preference down to endpoint-specific Step 3 work, and added complex-query checks plus implementation notes to preserve the reasoning behind those choices.
Design context: Synthesizes the latest discussion about keeping search-system internals out of the LLM contract, preserving positive-presence phrasing, supporting concepts that contain both dealbreakers and preferences, and containing weighting/prominence behavior inside Step 3 where the data can judge it best.
Testing notes: Planning-only docs change; no code or tests touched.

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

## Align production-brand tests with tier-spec intent
Files: unit_tests/test_brand_resolver.py, unit_tests/test_production_brands.py
Why: The new brand resolver tests needed to assert the desired behavior from `search_improvement_planning/production_company_tiers.md`, not simply mirror the current registry and resolver implementation.
Approach: Tightened resolver assertions to exact ordered outputs where ordering is part of the contract, renamed misleading tests/docstrings, and added missing edge coverage for inclusive year boundaries, multi-brand overlap windows, exact-string matching, and first-index behavior when an earlier candidate is year-gated out. Added registry spot-checks for Miramax, Dimension Films, Walt Disney Productions, Searchlight, Republic Pictures, and exact-string matching semantics.
Design context: Used the per-brand member tables plus explicit per-brand notes in `search_improvement_planning/production_company_tiers.md` as the source of truth when surrounding prose conflicted. Miramax follows the PARAMOUNT section's explicit exclusion note.
Testing notes: Ran `pytest unit_tests/test_brand_resolver.py unit_tests/test_production_brands.py -v`. Result: 62 passed, 1 failed. The failing assertion is `test_walt_disney_productions_is_gated_to_1929_1986_for_both_brands`, which exposed registry drift: `WALT_DISNEY_ANIMATION` currently encodes `Walt Disney Productions` as `(None, 1986)` instead of the spec-driven `(1929, 1986)`.

## Exhaustive production-brand date-spec coverage
Files: unit_tests/production_brand_spec_dates.py (new), unit_tests/test_production_brands.py, unit_tests/test_brand_resolver.py
Why: The selective spot-checks were missing broad classes of date drift. The user asked for comprehensive date coverage where all expected windows come from the production-brand tier spec rather than the current registry implementation.
Approach: Added a shared static fixture of date-bearing production-company expectations, grouped by `(brand, start, end)` and derived from `search_improvement_planning/production_company_tiers.md`. Replaced registry spot-checks with an exhaustive parametrized reverse-index assertion over every date-bearing string in the current schema. Replaced most single-string resolver date examples with a generated boundary harness that checks `None`, boundary years, and just-outside-window years against the same shared fixture, while keeping separate hand-written tests for index-minimization and sort-order mechanics.
Design context: Primary authority is the tier doc's per-brand member tables and surface-form tables. Cross-brand additions are only encoded where the spec intends umbrella overlap. For low-volume variants where the surface-form table names a variant not repeated in the member table, the fixture follows the nearest explicit spec lineage rather than the current registry's unconditional fallback.
Testing notes: Ran `pytest unit_tests/test_brand_resolver.py unit_tests/test_production_brands.py -q`. Result: 923 passed, 194 failed. The failures are overwhelmingly spec-vs-registry drift surfaced by the new exhaustive fixture, including the previously-seen Walt Disney Animation starts plus broader unconditional/self-brand windows, missing/incorrect start dates, and multiple umbrella-overlap mismatches across Columbia, Focus Features, DreamWorks Animation, Pixar, Lucasfilm, Marvel Studios, New Line, MGM/Amazon MGM, United Artists, and Walt Disney Animation surfaces.

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

## Stage 4 orchestrator — per-branch candidate assembly, scoring, and payload shaping
Files:
- search_v2/stage_4/__init__.py (new)
- search_v2/stage_4/types.py (new)
- search_v2/stage_4/flow_detection.py (new)
- search_v2/stage_4/priors.py (new)
- search_v2/stage_4/dispatch.py (new)
- search_v2/stage_4/assembly.py (new)
- search_v2/stage_4/scoring.py (new)
- search_v2/stage_4/display.py (new)
- search_v2/stage_4/orchestrator.py (new)
- db/postgres.py (added fetch_browse_seed_prior_inputs helper)

### Intent
First implementation of Stage 4 of the V2 search pipeline. Takes one Step-1 branch's `QueryUnderstandingResponse`, fans out all Step-3 translations + executions, assembles the candidate pool, applies exclusions, composes final scores, and returns the top-K shaped rows plus debug. Single branch in, single result set out — the caller drives multi-branch presentation.

### Key Decisions
- **Nine-module package, single public entry.** `run_stage_4()` in orchestrator.py is the only public function; everything else is internal. Split the work so each file has one job: flow_detection (pick scenario + tag items), dispatch (one LLM + one exec call with timeouts and soft-fail), assembly (union + exclusion subtract), priors (quality/notability score + browse composite), scoring (dealbreaker_sum + preference_contribution − exclusion_penalties), display (payload shaping from prefetched cards), types (runtime dataclasses), orchestrator (wires phases).
- **Per-LLM and per-execution 20s timeouts, no branch budget.** dispatch.TIMEOUT_SECONDS wraps both `translate_item` and `execute_item`. TimeoutError and other exceptions degrade to empty EndpointResult with `status ∈ {timeout, error}` on the outcome; nothing raises out of dispatch. The existing executors already retry transient errors internally, so in practice most failures surface as wait_for timeouts.
- **Pre-pass item tagging.** `tag_items` walks the step-2 items once and stamps each with role (inclusion_dealbreaker / exclusion_dealbreaker / preference), endpoint, debug_key (`role[idx]`), and `generates_candidates` based on the chosen flow. Downstream code never re-derives these flags. Matches the plan's "Initial pass marks calls as candidate generating or not" instruction.
- **Four-flow dispatch table.** STANDARD (any non-semantic inclusion db exists), D2 (only semantic inclusion dbs), P2 (no inclusions, ≥1 preference), BROWSE (no inclusions, no preferences). Deterministic exclusions never affect flow choice — they are purely a post-assembly subtraction. See step_4_planning.md §"Execution ordering" Step A for the decision table.
- **Semantic preference per item = one LLM call.** The old "pure-vibe batched LLM" pattern was retired by the user during planning; every Step-2 item now generates exactly one translation call. The existing `generate_semantic_preference_query` already accepts a single description string, so no stage-3 signature changes were needed.
- **Soft-fail status is explicit on EndpointOutcome.** EndpointResult alone can't distinguish a no-match from a timed-out call — both yield `scores=[]`. Introduced `OutcomeStatus ∈ {ok, timeout, error, skipped, no_match}` so debug records carry the failure mode. Orchestrator wraps results with status="ok" even when empty; the dispatch layer stamps timeout/error on failures; `_run_pool_dependent_executions` stamps "skipped" when the upstream translation failed.
- **Inverted prior math mirrors metadata endpoint.** Per the user's direction, quality_score and notability_score mirror `_score_reception` / `_score_popularity` in metadata_query_execution.py. Quality uses a 0.7/0.3 blend of reception ramp and popularity; both ramps flip (well→poor, popular→niche) for INVERTED. SUPPRESSED returns 0 and callers drop the term from the weighted average rather than feeding a zero-weight entry.
- **Browse composite ≡ prior-scoring formulas.** Same underlying ramps used both for ordering the BROWSE-flow seed pool and for computing prior scores during scoring. Single source of truth. Browse seed falls back to STANDARD mode when a prior is SUPPRESSED — the seed still needs a deterministic ordering even when no explicit quality/notability intent exists.
- **Full-corpus browse seed.** Added `fetch_browse_seed_prior_inputs()` to db/postgres.py — returns `(movie_id, reception_score, popularity_score)` triples for every row in movie_card. Orchestrator sorts in Python by `browse_composite` and takes top 2000. Bulk-fetch keeps the pattern consistent with the rest of the codebase; ~100K rows is fine in a single query. Chose this over a SQL-expression ordering because the composite depends on the branch's quality_prior / notability_prior modes, which can't be baked into the query.
- **Default-zero rule via dict.get(mid, 0.0).** scoring.py converts each EndpointResult's scores list into a `dict[int, float]` once per outcome, then iterates the pool with `.get(mid, 0.0)`. This uniformly handles: (a) inclusion-dealbreaker misses (candidate entered the pool via another generator), (b) preference misses, (c) deterministic-exclusion misses (= not removed, not penalized), (d) semantic-exclusion misses (= not penalized). One rule, five cases.
- **Stable arrival-order ties.** assembly.assemble_pool preserves the insertion order of unioned ids in a list (not a set). Python's stable sort in Phase 7 then keeps arrival order for equal final_scores. Matches the planning doc's explicit decision that "ties are rare enough that imposing a deterministic tiebreaker isn't worth the complexity."
- **Cards prefetch serves priors AND display.** `fetch_movie_cards(pool)` runs once in Phase 5 in parallel with pool-dependent endpoint executions. Its output feeds both the prior-score inputs for scoring AND the display payload for the top-100 rows — no second DB round-trip. display.build_display_payload now accepts a `cards_by_id` dict instead of fetching itself.
- **Empty-pool short-circuit.** If Phase 3 (generation) or Phase 4 (exclusion subtract) leaves the pool empty, `_short_circuit_empty` returns a full Stage4Debug with zero scores — pool-dependent translations that already completed are surfaced as `status="skipped"` so the debug still names them.
- **Field rename at the payload boundary.** movie_card stores `title` / `release_ts`; the API contract in step_4_planning.md specifies `movie_title` / `release_date` (ISO string). display.py does the rename + UTC-ISO conversion once at the serialization boundary. Keeps the DB layer canonical without leaking its column names to the API.

### Planning Context
Implementation follows the approved plan at /Users/michaelkeohane/.claude/plans/magical-dazzling-shell.md and the finalized design in search_improvement_planning/step_4_planning.md. User clarifications during plan review resolved: per-LLM 20s timeouts (not branch-wide), pool-independent translation for deterministic exclusions so their subtract set is ready at the barrier, starter debug shape is an EndpointOutcome dict + per-result ScoreBreakdown dict, no backfill when pool < top_k, single branch returns independently (no multi-branch merge in this layer), ignore concurrency limits for now, inverted priors mirror metadata endpoint's niche / poorly-received formulas, and `fetch_movie_cards()` returns `release_ts` which gets converted to ISO at payload time.

### Testing Notes
- AST-compiled all nine new stage_4 modules successfully.
- Ran synthetic smoke tests against flow_detection, priors, assembly, and scoring with hand-constructed EndpointOutcomes — all logical invariants held (flow selection across STANDARD/D2/D2+exclusion/P2/BROWSE; candidate-generator tagging; pool union order preservation; deterministic exclusion hard-filter; default-zero rule for asymmetric pool membership; semantic exclusion penalty driving final_score negative).
- End-to-end live testing deferred to a stage_4 notebook with real step-1/2 outputs, a live LLM provider, and qdrant/postgres connectivity — same deferred-to-eval pattern the stage_3 modules follow.
- Risks to validate on the first live run:
  1. Browse-flow memory footprint — full-corpus prior fetch lands as ~100K tuples in Python; acceptable but worth watching as the corpus grows.
  2. Exact-duplicate scores across many candidates — stable arrival-order tiebreak was asserted synthetically; real-data floods (e.g., big trending tail) may still warrant a secondary tiebreaker if ranking visibly degenerates.
  3. The metadata executor's positional-or-keyword `restrict_to_movie_ids` signature vs every other executor's keyword-only form — dispatch passes it by name in all paths, but if metadata ever gets a second positional arg the dispatcher would need updating.

## Stage 4 code review fixes
Files: search_v2/stage_4/orchestrator.py, search_v2/stage_4/dispatch.py, search_v2/stage_4/types.py, search_v2/stage_4/priors.py, db/postgres.py
Why: Post-implementation review of the new Stage 4 package surfaced one small correctness issue plus several cleanups and one structural improvement worth doing before commit.
Approach:
- **Short-circuit status preservation (orchestrator.py).** `_short_circuit_empty` was collapsing every pool-dependent outcome to `status="skipped"` even when the upstream translation itself had timed out or errored, which produced a contradictory outcome (status says "skipped" but error_message carries a timeout string). Now preserves the translation's failure status; only truly successful translations get stamped "skipped".
- **Dead imports / aliases removed.** Unused `Callable` import out of dispatch.py. `_ScoreBreakdown = ScoreBreakdown` dead alias out of orchestrator.py along with the now-unused `ScoreBreakdown` import (convention forbids backward-compat aliases).
- **Inlined `ItemRole` Literal (types.py).** Moved the Literal alias into `TaggedItem.role` directly since nothing outside that field referenced it. `OutcomeStatus` stays as a standalone alias — dispatch.py uses it in function return types, so the cross-module reuse justifies keeping it separate.
- **BROWSE-flow seed moved into SQL (db/postgres.py + orchestrator.py).** The old path fetched every row of movie_card (~100K today) and sorted in Python. Replaced with `fetch_browse_seed_ids(quality_inverted, notability_inverted, limit)` that scores / sorts / LIMITs inside Postgres, so only 2000 ids cross the wire and Python does zero per-row work. Expected latency drop is 30–60 ms per BROWSE branch at current corpus size, and it also keeps the cost from scaling linearly as the corpus grows.
- **SQL mirror of `browse_composite`.** The formula in the new SQL helper mirrors `priors.browse_composite` exactly (same ramps, same weights, same ENHANCED/STANDARD/SUPPRESSED-identical-to-each-other / INVERTED branching). Added cross-reference comments in both files so the coupling is visible. The Postgres gotcha with `GREATEST`/`LEAST` silently ignoring NULL args rather than propagating them is defended against with `COALESCE(col, 0)` inside each ramp — without it a null-reception movie would score a perfect 1.0 by accident.
- **Caller simplification.** `_build_browse_seed` collapses to a one-line call into the new helper, translating the `SystemPrior` mode enum into the SUPPRESSED-as-standard inversion bits the SQL expects.
Design context: Reviewed per .claude/commands/review-code.md; the BROWSE-flow SQL push respects PROJECT.md priority #2 (latency) and the cross-codebase invariant "Push filtering to the data layer" (docs/conventions.md). The Python ↔ SQL drift risk is mitigated by mirror comments on both sides and by a mechanical parity check (six representative mode combinations, all matching to < 1e-9).
Testing notes: AST-compiled every modified file; re-ran the import smoke test for the stage_4 package; verified Python and SQL formulas produce identical outputs across STANDARD/STANDARD, ENHANCED/STANDARD, INVERTED/INVERTED, None/None, SUPPRESSED/SUPPRESSED, and STANDARD/INVERTED mode combinations. Live end-to-end still deferred to the stage_4 notebook.

## Stage 1-4 end-to-end test notebook
Files: search_v2/test_stage_1_to_4.ipynb
Why: Existing notebooks cover stages 1+2 (test_stage_1_2) and each stage-3 endpoint in isolation (test_stage_3), but nothing walks a single query through the full pipeline with per-stage visibility.
Approach: Drives stages 1-4 manually in the notebook rather than calling `run_stage_4`, because `Stage4Debug` does not preserve the stage-3 LLM specs the user wants to inspect. Reuses the building blocks `run_stage_4` uses (`detect_flow`, `tag_items`, `translate_item`, `execute_item`, `assemble_pool`, `apply_deterministic_exclusions`, `score_pool`, `build_display_payload`) so there is no behavior drift. Six cells: setup+imports, config, stage 1, stage 2 (picks first STANDARD interpretation), stage 3 (translate all items → execute pool-independent → print per-endpoint LLM spec + results + consolidated stats), stage 4 (execute pool-dependent restricted to pool → score → top-K with per-id breakdowns + optional overviews via `load_movie_input_data` from the ingestion tracker).
Design context: BROWSE flow intentionally unsupported (would need `fetch_browse_seed_ids` + prior scoring) — notebook raises with a clear message. Only the first STANDARD interpretation is exercised to keep control flow simple; selector is a one-line change.
Testing notes: JSON-validated and per-cell syntax-checked; every import resolves; function signatures match. Live execution requires Postgres/Redis/Qdrant services + `ingestion_data/tracker.db` for overviews.

## Studio brand registry — Workstream B (DB surface-form enumeration)
Files: search_improvement_planning/production_company_tiers.md

### Intent
Populate every Tier 1/2/3 brand section in `production_company_tiers.md` with the concrete distinct IMDB `production_companies` strings that refer to its members, with tag counts and flagged collisions. Follows Workstream A's ownership graph and closes the loop so the `brand_member_company` seeding table can be derived from this one document.

### Approach
Built a single shared enumeration first (`/tmp/company_strings.tsv`: 181,871 distinct strings across 361,866 movies, 710,491 tag occurrences), then dispatched 48 parallel per-brand subagents — one per brand in the registry — each reading its brand's subsection for the member list, grepping the shared index, and writing `/tmp/ws_b_results/{BRAND_ID}.json`. Aggregator script (`/tmp/normalize_v2.py`) tolerates the schema drift observed across agents (bare strings in `members`, flat one-row-per-surface dicts, field renames `canonical`/`canonical_name`/`company_name`, `string`/`s`/`name`/`value`, `count`/`tsv_count`/`credit_count`, `surface_strings`/`strings`/`variants`/`surface_forms`) and emits strict `{canonical, start_year, end_year, surface_strings:[{s,count}]}` entries. 379 verbatim DB strings enumerated; every string verified to exist in the TSV with exact count.

### Key Decisions
- **Single shared index instead of 48 DB scans.** Avoids 48× redundant SQLite reads of the 361K-row `imdb_data` table; also lets agents focus on judgment rather than DB mechanics.
- **Tolerant aggregator, not re-prompt.** When agents returned non-canonical schemas, normalized post-hoc rather than re-dispatching. Counts are recovered by TSV lookup when missing, so information loss is zero.
- **Surface-form section inserted per-brand, not as a separate appendix.** Each brand now reads as a single self-contained unit: members table → rename history → curation notes → surface forms → flagged collisions → surface-form notes. Curator will work brand-by-brand so this layout matches the consumption pattern.
- **Collisions preserved as flagged entries, not discarded.** Every agent-flagged collision (unrelated token-sharing companies, TV-only divisions, pre-acquisition eras, home-video arms) is captured in the inserted subsection so the curator has the rejections in context with the keeps.

### Open items handed to curator
The 7 open curation choices at the bottom of the file are unchanged. New evidence surfaced: HBO Max Films has zero TSV hits (curation #7 answered — drop); Canal+ has 3,949 tag count (curation #3 confirmed — exclude from brand_member); `DC Comics` pre-2009 production credits exist in the TSV (curation #4 confirmed — keep the member row). Amazon Studios vs Amazon MGM Studios boundary (curation #5) remains unresolved because surface strings alone can't disambiguate the 2022-2023 rename window.

### Testing Notes
Not code — no tests. But curator should spot-check at least one brand where a flagged collision looks wrong (e.g. verify "Castle Entertainment" really is unrelated to Castle Rock).

## Studio brand registry — MVP trim (31 brands) with Wikidata + DB verification
Files: search_improvement_planning/production_company_tiers.md

### Intent
Narrow the 48-brand registry to a hand-curated MVP cut of 31 brands, validated against Wikidata ownership timelines and IMDB DB evidence, so `brand_member_company` seeding can proceed from this single source of truth without further research rounds.

### Approach
Two verification passes, then destructive edit of the tier list:
1. **DB evidence floor** (`/tmp/db_floor_check.py`) — scored every member of every kept brand against the WS-B TSV. Classified members as OK (max count ≥ 5), WEAK (max 1–4), or DROP (zero strings present). 3 DROP members removed; 17 WEAK members kept because Wikidata confirms them as real sub-entities with thin IMDB mass (not an accuracy issue).
2. **Wikidata date validation** — delegated the 31 date-bearing member assertions to a research agent with the Wikidata `owned by` + `inception` + `dissolved` property set. 29 OK, 2 corrected inline.

Edits: the 17 cut brand sections removed (7 from old Tier 2 + all 10 from Tier 3); the Tier 3 header deleted; the surviving Tier 2 re-labelled "still in MVP (7 brands)"; cross-brand membership table pruned to 18 edges; new "Deferred from MVP" subsection added under "Excluded on purpose" listing the 17 cuts. File dropped from 160,143 → 112,580 bytes (30%).

### Key Decisions
- **IMDB's 8-studio filter is too coarse to emulate.** It's a homepage browse aid keyed on US-corporate-entity credit strings; copying it would regress intent coverage for A24/Ghibli/Netflix/Blumhouse and lose umbrella pull-through for Pixar/Marvel/Searchlight under "Disney". 31 is the middle ground — IMDB's 8 under-scopes, the research's 48 slightly over-scopes in niche enthusiast territory.
- **Drop criterion: freeform token match handles it losslessly OR near-zero DB mass.** LAIKA/AARDMAN/AMBLIN have distinctive unique tokens — freeform path covers them with no recall loss. HBO_FILMS had near-zero mass on the specific "HBO Films" credit. All 10 Tier 3 brands deferred as enthusiast-only until query logs justify.
- **Borderline TRISTAR / TOUCHSTONE / UNITED_ARTISTS kept.** Each has a distinctive token + real catalog mass + sub-brand-as-member-of-parent umbrella role. Keeping them costs 3 rows in the enum but preserves the sub-brand intent users have for '80s–'90s catalogs.
- **Weak-evidence members kept.** 17 members with max count < 5 are Wikidata-verifiable real sub-entities. Keeping them is accurate; dropping would gain only LLM-prompt-token savings while losing a long-tail recall edge. Accuracy (the user's explicit criterion) took precedence over token frugality.
- **Dropped brands' `production_company` strings still ingest.** Only the `BrandEnum` row and `brand_member_company` umbrella edges are deferred; the raw strings remain discoverable via the freeform token-match path per the v2 studio-resolver design (`v2_search_data_improvements.md`).

### Verification artifacts (uncommitted, in /tmp)
- `/tmp/ws_b_results/*.json` — 48 original per-brand agent outputs, untouched
- `/tmp/company_strings.tsv` — shared 181,871-string index with tag counts
- `/tmp/db_floor_check.py`, `/tmp/db_floor_report.json` — DB evidence floor pass
- `/tmp/date_assertions.json` — 31 date-bearing member assertions fed to Wikidata agent

### Open items still on the curator
The "Open Curation Choices" section was trimmed from 5 to 2 items (DC Comics pre-2009 publisher credit; Amazon Studios vs Amazon MGM Studios year-gating). StudioCanal/Canal+ and HBO Max Films questions resolved by dropping those brands from MVP.

### Testing Notes
Not code — no tests. Structural validation script run post-edit: 31 brand sections confirmed, Tier 3 header removed, Wikidata corrections landed, zero-evidence members purged. Curator should still spot-check collision flags on kept brands before `brand_member_company` seeding.

## Brand-tagging registry + resolver (not yet wired into ingest)
Files: schemas/production_brands.py (new), movie_ingestion/final_ingestion/brand_resolver.py (new), unit_tests/test_production_brands.py (new), unit_tests/test_brand_resolver.py (new)

### Intent
Encode the 31-brand MVP registry from `search_improvement_planning/production_company_tiers.md` as a machine-readable enum and provide a pure function that, given a movie's IMDB production_companies list + release year, returns its brand tags. Ingestion-side wiring into `ingest_movie.py` is intentionally deferred — this changeset only lands the building blocks.

### Key Decisions
- **Slug-backed str Enum with dataclass companies**, mirroring `schemas/award_category_tags.py` pattern for attribute attachment via `__new__`. `brand_id: int`, `display_name: str`, `companies: tuple[BrandCompany, ...]` are attached per member. Alternatives considered: int-backed enum (chosen against — slug values help debugging/logging). `BrandCompany` is `@dataclass(frozen=True)` so the registry is immutable post-import.
- **Year-window semantics are inclusive**, with `None` meaning "no bound on that side". `(None, None)` = "always applicable". The `year_matches` predicate enforces the user-directed rule that a `release_year=None` movie matches ONLY unconditional memberships — any windowed row is dropped.
- **Standalone self-memberships use (None, None)** per user's resolved plan-time question. Affects A24, PIXAR, NEON, STUDIO_GHIBLI, MARVEL_STUDIOS, LUCASFILM, etc. Umbrella-acquired sub-brands carry their acquisition window (e.g. Pixar under DISNEY = 2006-, Twentieth Century Fox under DISNEY = 2019-2020).
- **MGM/UA legacy self-memberships stay open-ended** per user's resolved question — post-2022 MGM films carry BOTH the MGM and AMAZON_MGM tags (additive, not migration). Implemented by adding MGM's full string set under AMAZON_MGM with `start=2022` and UA's set with `start=2024`.
- **Reverse index built at import**. `_STRING_TO_MEMBERSHIPS: dict[str, list[tuple[brand, start, end]]]`. Same surface string can appear under multiple brands with different windows; all are stored and the resolver emits every brand that passes the year check.
- **Resolver output is `list[tuple[brand_id, first_matching_index]]`** sorted by min-index asc with brand_id asc as deterministic tiebreak. `first_matching_index` is the 0-based position within the input `production_companies` list at which the brand first matched — preserves prominence signal for downstream ranking.
- **String match is exact** (case + whitespace sensitive) per the user's guidance ("we can normalize later"). Normalization layer deferred.
- **Republic Pictures is year-gated to 2023+** under PARAMOUNT — the same IMDB string covers the 1935-1967 legacy company, which must NOT tag pre-2023 films. Tested explicitly.

### Design Context
Plan file: `/Users/michaelkeohane/.claude/plans/reflective-popping-candle.md`. Tier-doc data: `search_improvement_planning/production_company_tiers.md` (surface-form tables + Cross-Brand Membership Summary).

### Testing Notes
50 tests, all passing on first run. Split:
- `test_production_brands.py` — registry invariants (31 brands, unique ids/slugs, non-empty strings, sane year ranges, no duplicate strings within a brand, reverse-index completeness) + `year_matches` behavior (open-ended, bounded, release-year-None rule, boundary inclusivity) + spot-check tier-doc intent (Miramax / MGM / UA / Pixar / Republic Pictures multi-brand cases).
- `test_brand_resolver.py` — behavioral tests using real registry strings (not mocks) so a regression in either the data or the function fails a test. Covers: degenerate inputs, year gating (umbrella pre/post acquisition, Miramax window close, MGM/UA Amazon overlap, Republic Pictures legacy rejection), dedup + min-index tracking, sort order + tiebreak, real-catalog scenarios (Marvel MCU, pre-1935 Fox, Searchlight rebrand bridge, mixed-credit lists).

## Wire brand resolver + production-company paths into ingestion
Files: db/init/01_create_postgres_tables.sql, db/postgres.py, implementation/misc/helpers.py, implementation/misc/production_company_text.py (new), movie_ingestion/final_ingestion/ingest_movie.py, backfill_production_brands_and_companies.py (new)

### Intent
Wire the orphaned `resolve_brands_for_movie` helper into Stage-8 ingestion and introduce the v2 freeform production-company path from `search_improvement_planning/v2_search_data_improvements.md`. Each ingested movie now lands brand postings (with prominence) plus a curated, tokenized company registry — the data substrate the v2 stage-3 studio resolver will consume once the query side catches up.

### Key Decisions
- **Ordinal rule scoped to production-company strings, not the global normalizer.** `normalize_company_string` wraps `normalize_string` with a `\b\d+(st|nd|rd|th)\b` → word regex (1-30 covered). Global `normalize_string` is untouched so existing `lex.lexical_dictionary` / title postings / character postings don't need a rebuild. Trade-off: a second normalizer exists, but title-tokenizer `already_normalized` flag lets us call the existing split logic without re-normalizing and undoing the ordinal conversion.
- **Three new lex tables** (`lex.production_company`, `lex.studio_token`, `lex.inv_production_brand_postings`) plus one new column (`movie_card.production_company_ids BIGINT[]` GIN-indexed). `production_company_ids` on the card prevents the cross-company token false-positive ("a b c" query matching a movie with three separate single-token companies) that the plain postings-only model would have.
- **`lex.inv_studio_postings` is kept as a frozen snapshot** — stop writing, don't drop. The v1 compound-lexical search path and v2 stage-3 entity lookup still read from it and replacing them is out of scope. Table and v1/v2 read paths remain intact; the backfill does not drop. This is a deviation from the approved plan (which called for a DROP) made to preserve query functionality until the stage-3 cutover.
- **`ingest_production_data` split from `ingest_lexical_data`.** Brand/company work is a distinct concern from the lexical dictionary + postings flow. Splitting also lets the batched ingest path wrap production writes in their own savepoint so a brand-registry mismatch can't nuke the movie's other writes.
- **`write_production_data` exposes the core as a `Movie`-free API** so the backfill can feed it raw inputs from the tracker without reconstructing full Movie objects. `ingest_production_data` is now a thin adapter that extracts inputs from `movie.tmdb_data.release_date` / `movie.imdb_data.production_companies` and delegates.
- **Backfill reads production_companies from the SQLite tracker, not Postgres.** `movie_card` never persists the raw company list, so the tracker's `imdb_data.production_companies` JSON column is the on-disk source. Batched async commits every 200 movies so idle-in-transaction timeouts can't kill long runs.
- **DF ceiling deliberately deferred.** Ingest stores every token with no stoplist. A future query-time filter will drop tokens whose DF exceeds a dataset-size percentage — out of scope here.
- **`upsert_movie_card` gained a `production_company_ids` parameter with default `()`** so other callers don't break. Ingest path uses the dedicated `update_movie_card_production_company_ids` (UPDATE-only) because production resolution runs after `ingest_movie_card` — within the same transaction, so reruns still converge.

### Testing Notes
- Smoke-tested `normalize_company_string` / `tokenize_company_string` — `20th Century Fox` and `Twentieth Century Fox` both normalize to `twentieth century fox`; `21st Century Fox` → `twenty-first century fox`; `Metro-Goldwyn-Mayer (MGM)` tokenizes to `{metro-goldwyn-mayer, metro, goldwyn, mayer, mgm}`.
- Syntax-validated all modified Python files via `ast.parse`.
- Not run: the existing unit tests that reference `batch_insert_studio_postings` (test_postgres.py, test_ingest_movie.py) will fail — expected per the test-boundaries rule. Test updates are a separate phase.
- End-to-end smoke on real movies (Star Wars 1977 vs Force Awakens 2015, A24 indie, long-tail studio, no-imdb edge case) documented in the plan file's Verification section; not run yet.

## Review pass on production-brand / production-company ingestion
Files: db/postgres.py, implementation/misc/production_company_text.py, movie_ingestion/final_ingestion/ingest_movie.py, backfill_production_brands_and_companies.py

### Intent
Apply fixes from the review of the previous entry: one critical bug (backfill pool lifecycle would have failed immediately) plus four correctness/hygiene items. Test files intentionally left alone per the test-boundaries rule.

### Fixes applied
- **Critical — backfill pool lifecycle.** Moved `await pool.open()` into `_run_backfill` and `await pool.close()` into its `finally` block, both on the same event loop. Removed the dead second `asyncio.run(_close())` attempt that would have raised because `AsyncConnectionPool` maintenance tasks bind to the creating loop.
- **Production ids resolved before movie_card.** `write_production_data` now returns `list[int]` instead of calling `update_movie_card_production_company_ids`. `ingest_movie` runs `ingest_production_data` first and threads the returned ids through to `ingest_movie_card → upsert_movie_card`, so the card row gets its final `production_company_ids` in a single upsert instead of the previous "write empty, later UPDATE" pair. Matching reorder in `ingest_movies_to_postgres_batched` — new `sp_{tmdb_id}_production` savepoint precedes `sp_{tmdb_id}_card`, with the returned ids passed into the card call. `update_movie_card_production_company_ids` is kept exported for the backfill path where `movie_card` already exists and we just want to patch the column.
- **`already_normalized` on `tokenize_company_string`.** Added a flag analogous to the one I already plumbed into `tokenize_title_phrase`, and the hot ingest loop in `write_production_data` now passes `already_normalized=True` to avoid a redundant `normalize_string + ordinal regex` pass on every (already-normalized) company string.
- **Dead `ON CONFLICT` removed.** `batch_insert_brand_postings` DELETEs for the movie_id before INSERT, and `resolve_brands_for_movie` dedups by brand_id, so `ON CONFLICT (brand_id, movie_id) DO NOTHING` could never fire. Removed for clarity; comment points to the `batch_upsert_movie_awards` precedent that also uses delete-then-insert without ON CONFLICT.
- **Rowcount check in `update_movie_card_production_company_ids`.** Inlined a direct cursor path (rather than `_execute_on_conn`, which discards `rowcount`) and raises `ValueError` when the UPDATE affects 0 rows. Prevents silent no-ops if the card wasn't upserted first.

### Deferred
- Suggestion #6 (backfill's `missing_tracker` branch erasing pre-existing production data) explicitly deferred by the user. No change to that branch.

### Testing notes
- Re-ran `ast.parse` across all modified files.
- Smoke-tested `tokenize_company_string('20th Century Fox')` vs `tokenize_company_string('twentieth century fox', already_normalized=True)` — identical output, confirming the short-circuit path is safe.
- Unit tests still unrun per the rule; the three existing test references to `batch_insert_studio_postings` remain as pending test-update work.

## Add studio_token_doc_frequency materialized view
Files: db/init/01_create_postgres_tables.sql, db/postgres.py, movie_ingestion/final_ingestion/ingest_movie.py
Why: Freeform studio path needs DF-ceiling stop-word filtering analogous to title_token_doc_frequency. Previously only the title axis had a token-frequency MV.
Approach: Added `lex.studio_token_doc_frequency` MV (COUNT per token over `lex.studio_token`, which already enforces one row per (token, production_company_id) via PK, so COUNT per token = per-canonical-string DF as specified in v2_search_data_improvements.md). Unique index on `token` enables REFRESH CONCURRENTLY. Added `refresh_studio_token_doc_frequency()` in db/postgres.py mirroring the title version, and wired it into the post-ingestion refresh block in ingest_movie.py alongside the title and popularity refreshes.
Design context: DF-per-canonical-string semantics per search_improvement_planning/v2_search_data_improvements.md §"Stage B — Compute Token Document Frequencies".
Testing notes: Unit tests unrun per test-boundaries rule; existing test_postgres.py pattern for refresh_title_token_doc_frequency is the template.

## Refresh studio_token_doc_frequency from backfill script
Files: backfill_production_brands_and_companies.py
Files: backfill_production_brands_and_companies.py | Call `refresh_studio_token_doc_frequency()` at the end of `_run_backfill` (before pool.close) so the one-shot backfill leaves the new MV populated without a manual follow-up step.

## Wipe production data before rebuild in backfill script
Files: backfill_production_brands_and_companies.py
Why: Prior data in lex.production_company, lex.studio_token, lex.inv_production_brand_postings, and movie_card.production_company_ids was written with older normalization/tokenization rules and is considered wrong. Merging (ON CONFLICT DO NOTHING + DELETE-per-movie) would leave stale rows behind.
Approach: Added `_wipe_production_data()` step that runs once at the top of `_run_backfill` (before tracker load). TRUNCATE order: studio_token → production_company (RESTART IDENTITY CASCADE) → inv_production_brand_postings; then bulk UPDATE movie_card SET production_company_ids = '{}'. All four statements share one transaction. Per-movie writes in the subsequent loop remain unchanged — the full rewrite + fresh company-id assignment gives every movie_card row clean data. Step numbering updated 1/4→1/5, with the wipe as step 2/5.
Testing notes: Destructive operation; safe because the backfill fully rebuilds from tracker data and nothing outside these four sites references the wiped rows (FK is ON DELETE CASCADE; movie_card column is rewritten by the loop).

## Cardinal number-to-word + lone-hyphen guard for studio normalization
Files: implementation/misc/production_company_text.py, search_improvement_planning/v2_search_data_improvements.md
Why: Spot-check of the production-brand backfill surfaced two disconnects between the implementation and the v2 studio spec. (1) `-` was appearing as a studio_token row because `normalize_string` preserves hyphens and names shaped like `X - Y Productions` leave a bare `-` after whitespace split. (2) Pure-numeric tokens like `8` or `20` were kept in digit form, which meant `Section 8 Productions` couldn't match a query typed as "section eight". The spec only mandated ordinal conversion (`20th` → `twentieth`) for studios, leaving the digit form of cardinal numbers unhandled.
Approach: Extended `normalize_company_string` to chain a second regex pass after the existing ordinal pass, matching `\b\d+\b` and substituting word forms for integers in [0, 99]. Leading zeros (`01`) collapse via `int()` before lookup, so `Studio 01` and `Studio 1` produce the same token set. Capped at 99 — year-like numbers (`Fox 2000`, `Studio 100`) stay in digit form because users never spell them out, and converting them would add unnatural tokens. Compound cardinals use hyphens (`twenty-one`), matching the existing ordinal convention and piggy-backing on the tokenizer's hyphen-split to contribute `{twenty-one, twenty, one}`. The lone-hyphen guard is a post-tokenize filter in `tokenize_company_string` (`[t for t in tokens if t.replace("-", "")]`); kept scoped to the studio path so it doesn't change title tokenization behavior.
Design context: Spec updated to describe cardinal conversion as a shared rule across studio, awards, and franchise Normalization Rules sections. Awards section previously only had ordinal; franchise already had cardinal but framed as franchise-specific — reframed as part of the shared rule with matching bounds and carve-outs. The franchise Design Principles bullet and Open Decisions #3 were reworded to match.
Testing notes: Existing studio data was backfilled under the old rules and still has `-` tokens and digit-form small numbers in studio_token. A rebuild of `lex.production_company` / `lex.studio_token` is required for the new rules to take effect end-to-end — the backfill script is idempotent so a re-run should be sufficient. Unit coverage needed: `normalize_company_string` for `Section 8` / `Studio 01` / `Fox 2000` cases; `tokenize_company_string` for `X - Y Productions` and `Metro-Goldwyn-Mayer`.

## Franchise ingestion: token-index rewrite (ports studio resolver)
Files: db/init/01_create_postgres_tables.sql, implementation/misc/production_company_text.py, implementation/misc/franchise_text.py, db/postgres.py, movie_ingestion/final_ingestion/ingest_movie.py, backfill_franchise_entries_and_tokens.py

### Intent
Reimplements franchise retrieval on the same token-index + DF-ceiling substrate the studio resolver already uses. Strict string equality on `lineage` / `shared_universe` / `recognized_subgroups` missed when ingest-side and query-side LLMs produced slightly different canonical forms (`phase 1`/`phase one`, `the lord of the rings`/`lord of the rings`, `spider-man`/`spider man`). The new path collapses lineage+universe into a single search array on `movie_card`, matches through normalized token intersection, and leaves raw TEXT columns intact for debug / display / structural-flag filters. See search_improvement_planning/v2_search_data_improvements.md "Franchise Resolution" for the full design.

### Key Decisions
- **BIGINT (not INT) for franchise_entry_id** — diverges from the v2 plan wording but matches `lex.production_company.production_company_id`. Keeps the two resolver schemas isomorphic; movie_card.franchise_name_entry_ids / subgroup_entry_ids are BIGINT[] with plain GIN (no gin__int_ops — that opclass is INT[]-only).
- **Legacy `lex.inv_franchise_postings` dropped outright** (not left as a stub like `lex.inv_studio_postings`). Grep confirmed the only live writers were in this codebase; no search-side reader remained. Cleaned in the init SQL + dropped by the backfill's migration step.
- **Backfill reads from `public.movie_franchise_metadata`, not the tracker.** Raw strings are already persisted on the row and re-normalizing is idempotent, so tracker SQLite isn't a prerequisite for rebuild. Simpler path than the studio backfill (which had to read tracker because movie_card doesn't persist raw production_companies).
- **Digit-word substitution factored into a shared `apply_digit_word_substitution` helper in production_company_text.py.** franchise_text imports it so the 0–99 cardinal table has a single source of truth across both resolvers — drift between them would silently break normalization symmetry.
- **Franchise resolution runs BEFORE the card upsert**, same pattern studios use. write_franchise_data returns `(franchise_name_entry_ids, subgroup_entry_ids)` which ingest_movie_card threads into upsert_movie_card's single INSERT. Avoids the "upsert with []; then UPDATE" double-write.
- **DF ceiling deferred.** View + refresh plumbing ship now; empirical ceiling value picked in a follow-up after backfill populates `lex.franchise_token` and the distribution can be inspected.

### Planning Context
Approved plan at `/Users/michaelkeohane/.claude/plans/cheeky-dancing-hamming.md`. User-approved scope decisions on ID width, legacy-table disposition, backfill source of truth, and DF-ceiling deferral captured in that file's "Scope Decisions" section. Query-side cutover (stage_3/franchise_query_execution.py rewrite, FranchiseQuerySpec rename, numeric DF ceiling selection) is explicitly out of scope — tracked as follow-ups.

### Testing Notes
- DDL: fresh `docker-compose down -v && docker-compose up` should apply the updated 01_create_postgres_tables.sql cleanly; verify `\d public.movie_card`, `\d lex.franchise_entry`, `\d lex.franchise_token`, `\d+ lex.franchise_token_doc_frequency`, and confirm `lex.inv_franchise_postings` no longer exists.
- Per-movie ingest: run the existing CLI with `--max-movies 1 --disable-vectors` against a movie whose franchise_metadata has both lineage and shared_universe; check that movie_card.franchise_name_entry_ids and subgroup_entry_ids populate and lex.franchise_entry / lex.franchise_token gain the expected rows.
- Backfill: `python backfill_franchise_entries_and_tokens.py` is idempotent; re-running should produce byte-identical state. Post-run, inspect `SELECT token, doc_frequency FROM lex.franchise_token_doc_frequency ORDER BY doc_frequency DESC LIMIT 50` — that report drives the DF-ceiling follow-up.
- Unit tests in unit_tests/test_postgres.py and test_ingest_movie.py reference the removed `batch_insert_franchise_postings` / `replace_movie_franchise_postings` functions and will need updating in the separate test phase (per .claude/rules/test-boundaries.md, not touched here).



## Query-side studio endpoint: v2 stage-3 studio translator + retire v1 studio path
Files: schemas/enums.py, schemas/studio_translation.py, schemas/production_brand_surface_forms.py, search_v2/stage_3/studio_query_generation.py, search_v2/stage_3/studio_query_execution.py, search_v2/stage_3/entity_query_generation.py, search_v2/stage_3/entity_query_execution.py, search_v2/stage_2.py, search_v2/stage_4/dispatch.py, db/postgres.py, db/lexical_search.py, implementation/classes/schemas.py, db/init/01_create_postgres_tables.sql, backfill_production_brands_and_companies.py, search_improvement_planning/v2_search_data_improvements.md

### Intent
Wires the ingest-time brand registry + freeform token index (already populated on the backfill side) into query-time search. Adds a new stage-3 `studio` endpoint with its own LLM translator and executor, parallel to entity/award/franchise. Splits studio lookups out of the entity endpoint entirely. Retires `lex.inv_studio_postings` and the v1 compound-lexical studio branch that still read from it. Approved plan: `/Users/michaelkeohane/.claude/plans/ok-this-all-sounds-elegant-melody.md`.

### Key Decisions
- **Split, not extend.** Studio became a full sibling endpoint (new `EndpointRoute.STUDIO`, new `StudioQuerySpec`, new generation + execution modules). `EntityType.STUDIO` and `_execute_studio` were removed entirely from the entity path. Rejected the "add studio fields to EntityQuerySpec" alternative because the brand_id / freeform_names output shape shares nothing with person / character / title_pattern.
- **Flat 1.0 scoring for every brand or freeform match.** No prominence decay. Rationale: IMDB production_companies ordering is only a prominence signal for Hollywood billing-block films; Japanese catalogs are alphabetically sorted (Ghibli lands at idx 3-7 on Howl's Moving Castle, Only Yesterday, Kiki's Delivery Service, etc.) and European co-productions are mixed. Users can't express studio prominence in queries either ("Disney movies" is binary intent). `first_matching_index` and `total_brand_count` columns retained on `lex.inv_production_brand_postings` as future features, read by nothing today.
- **Brand + freeform + fallback.** Schema permits either/both/neither. Executor rule: brand_id set AND non-empty → use brand; brand_id unset → use freeform; brand_id set but empty result → fall through to freeform as backup. Avoids rejecting specs while giving the LLM latitude to always emit both when uncertain.
- **DF ceiling = 323**, applied at query time via the `lex.studio_token_doc_frequency` materialized view (already populated at ingest). Empirically picked value; no stoplist — pure DF filter. Per-name intersection: every token must survive DF filtering for the name to contribute; missing tokens collapse the name's contribution to empty (not to "matches everything").
- **Streamer disambiguation lives in stage_2, not stage_3.** NETFLIX / AMAZON_MGM / APPLE_STUDIOS are registered as producer brands. The studio prompt assumes anything it receives means production; stage_2's prompt carries the rule that "on Netflix" / "streaming on Apple TV+" routes to metadata's watch_providers path and only "Netflix originals" / "produced by" / "made by" routes to studio. Simpler per-prompt contract.
- **`lex.inv_studio_postings` dropped entirely.** `PostingTable.STUDIO`, `CompoundLexicalResult.studio_scores`, `execute_compound_lexical_search`'s studio branch, `db.lexical_search`'s include_studios / exclude_studios / studio_term_ids / studio_scores, and `LexicalCandidate.matched_studio_count` all removed. The DDL block is gone from init SQL; the backfill script's `_apply_schema_migrations` adds a one-shot `DROP TABLE IF EXISTS lex.inv_studio_postings CASCADE` for existing databases. `db.lexical_search` now drops `EntityCategory.STUDIO` entities silently with an explanatory comment — callers wanting studio coverage go through the v2 pipeline.
- **Single `thinking` field on StudioQuerySpec**, targeted. Guidance: state umbrella-vs-specific scope first, then either name the registry brand or list the surface forms. Avoids the broader multi-field evidence-inventory style of entity since the translator only has one structural decision to make.
- **Brand registry rendered into the prompt** via new `schemas/production_brand_surface_forms.py` (mirrors `schemas/award_surface_forms.py`). Caps surface-form samples at 8 per brand with a "+N more" suffix for umbrella brands that cover more strings than shown.

### Planning Context
- DF=323 value was locked in the conversation preceding implementation; bucket-analysis rederivation is a separate task. The plan's "Out of scope" section tracks it.
- Flat 1.0 scoring was chosen after a data check on Ghibli's IMDB credit ordering confirmed alphabetical sorting is the dominant convention for non-Anglophone catalogs, making `first_matching_index` systematically unreliable for those brands.
- Legacy studio table drop was approved explicitly despite breaking `api/cli_search.py`'s v1 search path — v1 is no longer the active user-facing entry point (`api/main.py` doesn't import `db.search`).

### Testing Notes
- `EntityType` is now {PERSON, CHARACTER, TITLE_PATTERN}; the entity_query executor's exhaustive dispatch ValueError message updated to "three values".
- `PostingTable` is now {ACTOR, DIRECTOR, WRITER, PRODUCER, COMPOSER, CHARACTER, TITLE_TOKEN}. Any code path that imported STUDIO will fail at import — only the expected test files (test_postgres.py, test_lexical_search.py, test_ingest_movie.py) still reference it; fixing those is deferred per the test-boundaries rule.
- `LexicalCandidate.matched_studio_count` is gone; test_lexical_search.py:236 asserts on it and will fail.
- Cross-codebase invariant preserved: the query side calls the exact same `normalize_company_string` + `tokenize_company_string` the ingest side used (reuse, not reimplementation), so tokens match by construction.
- Verification steps live in the plan's "Verification" section; no end-to-end verification has been run yet (Postgres + Qdrant would need to be up with the backfill already applied).

## Review pass on v2 studio query endpoint
Files: search_v2/stage_2.py, schemas/studio_translation.py, search_v2/stage_3/studio_query_generation.py, search_v2/stage_3/studio_query_execution.py, docs/modules/db.md
Why: /review-code surfaced one critical inconsistency, one latency warning, one readability nit, and one stale module doc. User approved four fixes; all landed in one follow-up pass.
Approach:
- **stage_2.py count refs removed.** Dropped both "one of seven" (in `_TASK`) and "one of these eight" (in `_ENDPOINTS`) — replaced with "one of the retrieval endpoints defined below" / "one of the endpoints below". Avoids stale-count maintenance when endpoints are added or removed.
- **StudioQuerySpec `brand_id` → `brand`.** The Pydantic field collided with the enum's own `brand_id` int attribute, making `spec.brand_id.brand_id` read like a typo. Renamed field to `brand`; executor now reads `spec.brand.brand_id` which clearly distinguishes enum member from its int attribute. Propagated through the LLM prompt (every `brand_id` schema-field reference became `brand`; the single remaining `brand_id` mention is the enum-attribute one in the module comment). Blast radius was zero beyond new files.
- **Freeform path batched to one DB round-trip.** `_execute_freeform_path` was doing N sequential `fetch_company_ids_for_tokens` awaits (one per freeform_name, up to 3). Rewrote into 4 phases: (1) tokenize every name + collect into a single token set, (2) one batched DF-filtered fetch, (3) per-name intersection in Python over the shared response, (4) final GIN join. Round-trip count is now 2 (tokens + movie-ids) regardless of how many freeform_names the LLM emits. Same semantics (intersection within name, union across names).
- **docs/modules/db.md refreshed.** Studio and `lex.inv_franchise_postings` were still listed in the `postgres.py` row's posting-tables parenthetical — stale after this session's studio drop and the prior franchise rewrite. Replaced with an explicit line-item for the new studio read helpers (`fetch_movie_ids_by_brand`, `fetch_company_ids_for_tokens`, `fetch_movie_ids_by_production_company_ids`) and updated the franchise mention to reference `lex.franchise_entry` / `lex.franchise_token`.
Testing notes: `python -c "import ast"` across all four files, plus a smoke-test confirming `spec.brand.brand_id` access works, `fetch_company_ids_for_tokens` is called exactly once in `_execute_freeform_path`'s source, and no `one of seven` / `one of eight` substrings remain in stage_2's SYSTEM_PROMPT. Unit-test updates still deferred per test-boundaries rule.

## Stage 1 intent branching redesign
Files: schemas/enums.py, schemas/flow_routing.py, search_v2/stage_1.py, search_improvement_planning/finalized_search_proposal.md, search_v2/test_stage_1_2.ipynb, search_v2/test_stage_1_to_4.ipynb
Why: Stage 1 previously treated branching as rare equal-likelihood ambiguity and returned a flat `interpretations` array. The new search behavior needs one authoritative primary intent plus up to two materially distinct alternatives, with branching driven by browsing value under ambiguity while preserving hard constraints.
Approach: Added `QueryAmbiguityLevel` and replaced the old Step 1 schema with top-level `ambiguity_analysis`, `ambiguity_level`, `hard_constraints`, `ambiguity_sources`, `primary_intent`, and `alternative_intents`. Split primary vs alternative intent models so alternatives can carry a scoped `difference_rationale` field that forces non-duplicate branching. Added Pydantic validators to enforce the flow/title invariant (`standard` => `title=None`; `exact_title` / `similarity` => non-empty title). Rewrote the Stage 1 system prompt from scratch around the repo's small-model prompt conventions: evidence-inventory reasoning, brief pre-generation fields, explicit empty-list paths, and schema-order generation. The prompt now uses only four dedicated boundary examples (`Scary Movie`, `Disney live action movies millennials would love`, `leonardo dicaprio boat movie from 2000`, `titties`) and keeps examples out of the general flow definitions. Updated the finalized proposal doc to match the new Step 1 contract and branching philosophy. Updated the demo notebooks to consume `primary_intent` plus `alternative_intents` instead of iterating `.interpretations`, while keeping Step 2 fan-out limited to standard-flow intents.
Design context: Based on docs/conventions.md prompt-authoring rules (cognitive-scaffolding field ordering, evidence inventory over post-hoc rationalization, brief pre-generation fields, principle-based constraints) and the user's approved plan to favor flexible branching rules over rigid slot semantics.
Testing notes: Verified schema behavior with direct `FlowRoutingResponse.model_validate(...)` calls for a valid exact-title payload and an invalid standard-flow payload carrying a title. `python -m py_compile schemas/enums.py schemas/flow_routing.py search_v2/stage_1.py` passed. Per repository test-boundary rules, no pytest files were read, edited, or run.

## Notebook cleanup after Stage 1 review
Files: search_v2/test_stage_1_2.ipynb, search_v2/test_stage_1_to_4.ipynb
Why: Code review flagged notebook regressions introduced by the Stage 1 refactor: stale saved outputs still showed the old `interpretations` schema in `test_stage_1_2.ipynb`, and `test_stage_1_to_4.ipynb` had drifted from the intended stable default demo configuration plus still used "interpretation" wording in one markdown cell.
Approach: Cleared saved outputs and execution counts in both notebooks so the committed artifacts no longer contradict the updated code. Restored `test_stage_1_to_4.ipynb`'s default query to `"Based on a book"` while keeping the stable OpenAI preset active, and updated the Stage 2 markdown cell to say "intent" instead of "interpretation."
Testing notes: Parsed both notebooks with `json.load(...)` after the cleanup and confirmed they contain zero saved outputs. No notebook cells were executed as part of this fix.

## Stage 1 display_phrase tone tweak
Files: search_v2/stage_1.py, search_improvement_planning/finalized_search_proposal.md
Why: The original `display_phrase` guidance was producing labels that were accurate but too robotic and flat for the intended browsing experience.
Approach: Adjusted the Stage 1 system prompt so `display_phrase` is still informative and semantically faithful, but explicitly encouraged to feel a little more lively, human-written, and UI-friendly. Mirrored the same expectation in the finalized Step 1 design doc so the documentation stays aligned with the prompt contract.
Testing notes: Prompt/doc-only change; no execution run.

## Pin Stage 1 to Gemini 3 Flash
Files: search_v2/stage_1.py, search_v2/test_stage_1_2.ipynb, search_v2/test_stage_1_to_4.ipynb
Why: Stage 1 flow routing runs on every query and should use a single stable low-latency config rather than letting callers drift.
Approach: Removed provider/model/**kwargs from `route_query`; hardcoded module-level `_STAGE_1_PROVIDER=GEMINI`, `_STAGE_1_MODEL="gemini-3-flash-preview"`, `_STAGE_1_KWARGS={"thinking_config": {"thinking_budget": 0}}`. Updated both notebook callers to invoke `route_query(query)` with no model params.
Testing notes: Stage 2 callers still use the notebook's shared provider/model/kwargs — only Stage 1 is pinned.

## Add LIVE_ACTION OverallKeyword as complement to ANIMATION
Files: implementation/classes/overall_keywords.py, schemas/movie.py, backfill_live_action_keyword.py, search_v2/stage_2.py, search_v2/stage_3/keyword_query_generation.py
Why: The keyword taxonomy had explicit entries for Animation and its sub-forms but no positive label for live-action titles, leaving queries like "live-action Disney remakes" with no deterministic routing target. A live-action signal was also missing from every movie card.
Approach: Added `LIVE_ACTION = (226, "Live Action", ...)` at the bottom of the enum (out of alphabetical order to preserve existing keyword_id stability). Updated `Movie.keyword_ids()` to append `LIVE_ACTION.keyword_id` whenever `ANIMATION.keyword_id` (8) is absent, so new/re-ingested movies carry the signal automatically. Wrote `backfill_live_action_keyword.py` (sync psycopg, batches of 5000 movie_ids, single UPDATE with `array_append` guarded by `NOT (keyword_ids @> ARRAY[8|226])` for idempotency) to retro-stamp existing `public.movie_card` rows. Renamed family 13 ("Animation / Anime Form / Technique" → "Animation / Live Action / Anime Form / Technique") in both the Stage 2 routing prompt and the Stage 3 `_FAMILIES` registry so the LLM can select the new member. `CLASSIFICATION_ENTRIES` and `UnifiedClassification` pick up the new member automatically from the OverallKeyword iteration in schemas/unified_classification.py — no changes needed there.
Design context: Chose ID-stability over alphabetical order per user instruction; placed LIVE_ACTION next to ANIMATION semantically (same family) rather than creating a new family to minimize prompt churn.
Testing notes: Verified enum resolves (`LIVE_ACTION.keyword_id == 226`), stage_3 import-time consistency check passes with the new _FAMILIES entry, and `keyword_ids()` stamps 226 only when 8 is absent (including the empty-keyword case). Backfill script is idempotent by construction — re-runs report zero updates.

## Split Stage 2 notebook demo into explicit Step 2A and Step 2B cells
Files: search_v2/test_stage_1_to_4.ipynb
Why: The notebook still reflected the old single-pass Stage 2 shape, which made manual QA drift from the new concept-based runtime and left the setup cell imports partially stale.
Approach: Reworked the notebook so the standard-flow demo now runs Stage 1, then a dedicated Step 2A concept-extraction cell, then a separate Step 2B per-concept planning cell that mirrors production behavior for concept drops and total Step 2B failure. Cleaned the setup cell imports so they line up with the current Step 1/Stage 2/Stage 4 module boundaries and DB helpers. Kept the later Stage 3/4 notebook orchestration operating on the assembled `QueryUnderstandingResponse`.
Testing notes: Parsed the notebook JSON after the rewrite and compiled every code cell with top-level-await support to verify syntax. No live notebook execution was run.

## Restore Step 2A boundary notes in schema and Step 2B inputs
Files: schemas/query_understanding.py, search_v2/stage_2.py, search_v2/test_stage_1_to_4.ipynb
Why: The implemented Step 2A contract had drifted to `concepts: list[str]`, which dropped the per-concept boundary rationale the design discussion intended and left Step 2B without explicit boundary guidance from Step 2A.
Approach: Added `ExtractedConcept` with `concept` plus `boundary_note`, changed `Step2AResponse` to carry `list[ExtractedConcept]`, and updated Stage 2A prompt instructions accordingly. Reworked Step 2B prompt construction so each per-concept call now receives the selected concept label, its boundary note, and a compact inventory that includes boundary notes for the rest of the concepts. Synced the end-to-end notebook so its Step 2A and Step 2B cells use the structured concept objects instead of assuming plain strings.
Testing notes: Ran `python -m py_compile schemas/query_understanding.py search_v2/stage_2.py` and recompiled every notebook code cell with top-level-await support. Per repo test-boundary rules, no pytest tests were run or edited.

## Harden Step 2A/2B around ingredient inventory and explicit coverage
Files: schemas/query_understanding.py, search_v2/stage_2.py, unit_tests/test_search_v2_stage_2.py, unit_tests/test_search_v2_stage_4.py, search_v2/test_stage_1_to_4.ipynb
Why: Recent query analysis showed Step 2B could silently drop part of a concept's meaning because the schema only carried prose analysis and the prompt did not force explicit ingredient preservation. The strongest local prompt-design pattern in this repo is to front-load compact reasoning fields that commit scope before answer fields are filled.
Approach: Reworked Step 2A so it emits a top-level `ingredient_inventory` and per-concept `required_ingredients`, with validation that concept ingredients exactly match the inventory. Reworked Step 2B so it returns a dedicated `Step2BResponse`, and each `RetrievalExpression` now declares `coverage_ingredients`. Final `QueryConcept` objects are assembled in code from Step 2A concept identity plus Step 2B planning output, with validation that every required ingredient is covered and no expression claims off-concept coverage. Rewrote the Step 2A and Step 2B prompts to follow the repo's cognitive-scaffolding conventions: inventory first, boundary/scope commitment second, emitted outputs last. Updated the manual notebook path to show ingredient inventory and to assemble final concepts the same way production now does.
Testing notes: `pytest unit_tests/test_search_v2_stage_2.py unit_tests/test_search_v2_stage_4.py -q` passed (24 tests). `python -m py_compile schemas/query_understanding.py search_v2/stage_2.py unit_tests/test_search_v2_stage_2.py unit_tests/test_search_v2_stage_4.py` passed. Recompiled all notebook code cells with top-level-await support; syntax passed. Remaining risk is live-model behavior, not schema/runtime wiring.

## Streamline Stage 1 routing schema before prompt rewrite
Files: schemas/flow_routing.py
Why: Stage 1 analysis showed the top-level routing schema had extra scaffolding fields that were not consumed downstream and were adding response bulk without improving runtime behavior. The agreed direction is to keep `ambiguity_analysis`, `primary_intent`, and `alternative_intents`, while relying on `ambiguity_analysis` to carry the ambiguity causes and the most likely reading.
Approach: Removed `ambiguity_level`, `hard_constraints`, and `ambiguity_sources` from `FlowRoutingResponse`, and updated the schema comments to describe the leaner top-level field order. Kept `routing_signals` on both intent types and left the Stage 1 prompt untouched for a separate follow-up pass so the prompt/schema contract can be reviewed deliberately.
Testing notes: `python -m py_compile schemas/flow_routing.py` passed. No Stage 1 prompt or test changes were made in this pass, so runtime structured-output calls will need the prompt updated next before Stage 1 is exercised again.

## Rewrite Stage 1 prompt around routing-plus-light-rewrite only
Files: search_v2/stage_1.py
Why: Stage 1 was overstepping by expanding queries into inferred retrieval dimensions and proxy traits, which polluted Step 2 inputs. The goal is for Stage 1 to choose flows, decide whether branching is worthwhile, and produce faithful branch rewrites without decomposing or enriching the query.
Approach: Replaced the old ambiguity-scaling / hard-constraint framing with a tighter prompt contract: `primary_intent` must be the most likely interpretation, branching must be materially useful, and `intent_rewrite` must clarify without decomposing. Added explicit anti-overreach guidance forbidding proxy traits like "iconic" / "highly-rated" unless directly supported, preserved the strong flow-boundary rules, and updated examples to target the observed failures (`Disney classics`, `iron man movies`, `Indiana Jones ... boulder`). Aligned output guidance with the leaner schema by moving the removed top-level field work into `ambiguity_analysis`.
Testing notes: `python -m py_compile search_v2/stage_1.py schemas/flow_routing.py` passed. No live Stage 1 runs or test-file changes were made in this pass.

## Compress Stage 1 ambiguity analysis into a labeled trace
Files: search_v2/stage_1.py
Why: The new single `ambiguity_analysis` field was still being described too much like prose, which would encourage token-heavy outputs. The goal is to keep the same decision information while cutting verbosity and still allowing multiple ambiguity sources when they exist.
Approach: Updated the output guidance so `ambiguity_analysis` is now framed as a compact labeled decision trace (`main=...; ambiguity=...; alt=...`) rather than paragraph prose. The prompt now explicitly allows one or more ambiguity sources inside the `ambiguity=` segment and tells the model to use short fragments instead of full sentences.
Testing notes: `python -m py_compile search_v2/stage_1.py` passed.

## Rebalance Stage 1 prompt toward broad evidence rules over narrow patterns
Files: search_v2/stage_1.py
Why: The previous prompt still risked over-matching on individual examples like `iron man movies`, which is exactly the wrong failure mode for a small model handling open-ended query space. The better approach is to teach a general evidence hierarchy for title-attempt vs collection-request readings and use only a small number of examples as boundary anchors.
Approach: Added a dedicated `READING-EVIDENCE HIERARCHY` section defining title-attempt evidence, collection-request evidence, and the tie-break rule when both appear. Tightened branching so exact-title alternatives are only allowed when there is genuine title-attempt evidence, not just a bare `X movies` shape. Reduced the prompt's dependence on specific query patterns by removing the `iron man movies` boundary example and replacing the ambiguity-analysis example with a generic collection-vs-title trace.
Testing notes: `python -m py_compile search_v2/stage_1.py` passed.

## Capture Stage 1 / Stage 2 lessons in a planning note
Files: search_improvement_planning/steps_1_2_improving.md
Why: The recent Stage 1 / 2 iteration surfaced several design lessons that are important to preserve even though the current behavior still needs more work. A dedicated note is better than leaving these conclusions buried in session history.
Approach: Wrote a new planning note summarizing the observed problems in Step 1, Step 2A, and Step 2B; what Stage 1's true role is versus overstepping; what we learned about helping small LLMs succeed on both schema and prompt design; and what the Stage 1 schema simplification taught us as reusable design principles.
Testing notes: Documentation-only change; no code execution needed.

## Character prominence scoring + [0.5, 1.0] floor compression
Files: schemas/enums.py, schemas/entity_translation.py, db/init/01_create_postgres_tables.sql, db/postgres.py, movie_ingestion/final_ingestion/ingest_movie.py, movie_ingestion/final_ingestion/rebuild_character_postings.py, search_v2/stage_2.py, search_v2/stage_3/entity_query_execution.py, search_v2/stage_3/entity_query_generation.py

### Intent
Give character-entity lookups prominence-aware scoring instead of a binary 1.0 match, and compress every actor/character prominence score into [0.5, 1.0] so real matches never fall below the dealbreaker-eligible floor. "Spider-Man movies" should now score higher on a movie where Spider-Man is top-billed than on one where he shows up as a minor role. Actor-scoring shape is unchanged; only the tail transform shifts.

### Key Decisions
- **Two character modes, not four.** CENTRAL (fixed 0.15-per-step decay, cast-size-independent) and DEFAULT (linear ramp across the character cast). Rationale and rejected alternatives in search_improvement_planning/character_scoring_revamp.md.
- **`entity_type == CHARACTER` is the gate.** Characters are a sibling `EntityType`, not a sub-type of PERSON; the validator, prompt gate, and `_execute_character` all dispatch off that single value.
- **Drop-and-rebuild the postings table.** Raw IMDB data lives in the tracker `imdb_data.characters` JSON column and in `ingestion_data/imdb/{tmdb_id}.json`, so a full rebuild is safe and simpler than an in-place ALTER + backfill. The backfill script reads the tracker's already-ordered character list rather than re-parsing raw IMDB JSON.
- **Variant max falls out of the row iteration.** Multiple term_ids for the same movie (aliases like "Peter Parker" / "Spider-Man") produce multiple rows in `fetch_character_billing_rows`; `_fetch_character_scores` takes the max per movie, so the "max across variants" behavior needs no special-casing.
- **Compression lives in one place.** A single `_compress_to_floor(raw)` helper in `search_v2/stage_3/entity_query_execution.py` is called at the tail of both `_prominence_score` (actor) and `_character_prominence_score`. Per-mode scorers stay byte-for-byte the same.
- **Step 2B carries prominence wording; Step 3 decides the mode.** Step 2B's entity-route guidance now tells the LLM to preserve actor-prominence and character-prominence language from the user's query into `RetrievalExpression.description`, so Step 3 has signal to key off. Step 3 adds a `_CHARACTER_PROMINENCE` section and two new schema fields (`character_prominence_evidence`, `character_prominence_mode`) parallel to the actor pattern.
- **Accepted quirks:** MINOR-mode 0.625 floor for a lead-role match (floor is absolute); stage 4 callers of actor scores now see a [0.5, 1.0] range (user is planning a full scoring revamp later and accepted this shift).

### Planning Context
Full design in search_improvement_planning/character_scoring_revamp.md. The in-session plan file at /Users/michaelkeohane/.claude/plans/the-2a-vs-curried-hedgehog.md resolves the planning doc's ambiguities (PERSON sub-type wording corrected; backfill re-run safety clarified; 2A/2B already-landed acknowledged). Legacy `_build_compound_character_cte` in db/postgres.py only joins on `(term_id, movie_id)`, so the schema expansion doesn't touch it.

### Testing Notes
- Planning doc's curve tables are reproduced exactly by the new scorers (verified via REPL spot-check: CENTRAL at pos 3 → 0.925, pos 8 → 0.55, pos 10+ → 0.5 floor; DEFAULT at pos 1 → 1.0, pos N → 0.5).
- Actor scores across all four modes now sit in [0.5, 1.0] for both small (cast=5) and typical (cast=25) casts; ordering within mode is preserved.
- End-to-end verification requires running the rebuild script against a dev tracker DB, then exercising the stage 3 notebook with contrasting queries ("Spider-Man movies" vs "movies with Spider-Man in it").
- Tests were not touched per test-boundaries rule; test_postgres.py already references `batch_insert_character_postings` / `lex.inv_character_postings` and will need updating in a separate phase.

## Franchise resolution: split lineage / shared_universe columns for prefer_lineage scoring
Files: db/init/01_create_postgres_tables.sql, db/postgres.py, movie_ingestion/final_ingestion/ingest_movie.py, schemas/franchise_translation.py, search_v2/stage_3/franchise_query_generation.py, search_v2/stage_3/franchise_query_execution.py, search_improvement_planning/v2_search_data_improvements.md

### Intent
Unblock queries like "shrek movies" where the user wants main-lineage entries (Shrek 1-4) slightly upranked over shared-universe-only entries (Puss in Boots) without excluding the latter. Previous design unioned both sides into `movie_card.franchise_name_entry_ids` at ingest so the retrieval path could not distinguish which slot a match came from; splitting the storage recovers that signal for scoring without changing the match set.

### Key Decisions
- **Split on the movie_card side, not the entry/token side.** The `lex.franchise_entry` and `lex.franchise_token` tables stay flat — lineage-vs-universe is a property of the movie-to-entry relationship, not of the entry itself. The same `"Shrek"` entry legitimately appears in Shrek sequels' `lineage_entry_ids` and in Puss in Boots' `shared_universe_entry_ids`. Splitting the entry space would have broken umbrella sweeps and forced the query LLM to predict which slot the ingest wrote to.
- **OR-combine for match, distinguish for scoring.** The stage-3 executor's name axis OR-combines both columns in a single SQL query, and the DB helper returns two disjoint sets: `(lineage_matched, universe_only_matched)`. A movie that matches via both sides is attributed to lineage (dominant slot wins) so the two sets never overlap.
- **New `prefer_lineage: bool` on FranchiseQuerySpec.** Default False preserves the pre-change behavior. When True, lineage matches score 1.0 and universe-only matches score 0.75. The 1.0/0.75 gap at endpoint level biases without excluding and survives downstream merging (verified in prior discussion).
- **Empty-lineage fallback.** When `prefer_lineage=true` but the lineage bucket is empty and the universe bucket is not, promote universe matches to 1.0. The flag biases ranking; it does not reject results. Without the fallback a franchise whose entry_id only ever lands in shared_universe columns in the corpus would return all matches at 0.75 for no gain.
- **Validator coerces the flag to False when it can't take effect.** Two cases: no name axis populated (nothing to apply preference to), or SPINOFF in structural_flags (user explicitly asked for spinoffs — upranking lineage would invert intent). Enforced as a hard guard even though the prompt also instructs against these patterns.
- **Prompt guidance biased toward the safe default.** Positive examples (shrek / john wick / toy story / harry potter / the conjuring movies) and negative examples (MCU / DCU / star wars / middle-earth / shrek spinoffs / harry potter and fantastic beasts) are enumerated. Guidance explicitly says "when in doubt, leave false" — misclassifying true→false only loses a ranking nudge; false→true demotes content the user wanted.
- **Pre-production so no backward-compat shim.** The old `franchise_name_entry_ids` column is removed from movie_card; the two new columns replace it. `upsert_movie_card` and `update_movie_card_franchise_ids` now take both arrays; `write_franchise_data` / `ingest_franchise_data` return a three-tuple `(lineage_ids, shared_universe_ids, subgroup_ids)`.

### Planning Context
Plan file: /Users/michaelkeohane/.claude/plans/ok-then-create-a-sleepy-goose.md. Decision thread resolved before implementation: boost magnitude = 0.75 (user's call); SPINOFF interaction = force False; multi-name umbrella queries leave flag False; entry/token tables untouched; ingest can be backfilled from movie_franchise_metadata (raw TEXT columns preserved) rather than re-running stage-6. Docs updated in search_improvement_planning/v2_search_data_improvements.md §Franchise Resolution to describe the new "one search space for matching, two for scoring" framing plus new worked examples 8 (prefer_lineage upranks Shrek main line) and 9 (empty-lineage fallback).

### Testing Notes
- DB rebuild required: the schema change drops one column and adds two with fresh GIN indexes. Run a backfill that resolves each movie's raw `lineage` / `shared_universe` / `recognized_subgroups` via `write_franchise_data`, or drop and reingest since we're pre-production.
- Spot-check after backfill: Shrek 1-4 should have the Shrek entry id in `lineage_entry_ids`; Puss in Boots should have it in `shared_universe_entry_ids` and its own lineage in `lineage_entry_ids`.
- Stage-3 end-to-end: "shrek movies" → Shrek sequels at 1.0, Puss in Boots at 0.75. "MCU movies" → all matches at 1.0 (prefer_lineage=false path unchanged). "shrek spinoffs" → validator coerces prefer_lineage=false; all matches at 1.0.
- Validator coercions: `prefer_lineage=True` with `franchise_or_universe_names=None` → coerced to False. `prefer_lineage=True` with `structural_flags=[SPINOFF]` → coerced to False.
- Tests not touched per test-boundaries rule. Any test that asserts the old `franchise_name_entry_ids` column name or binary franchise scoring will need updating in a follow-up phase — in particular tests around `fetch_franchise_movie_ids` (return type changed from `set[int]` to `tuple[set[int], set[int]]`), `upsert_movie_card` / `update_movie_card_franchise_ids` (parameter renames), and `write_franchise_data` (three-tuple return).

## Franchise prefer_lineage: review fixes + targeted migration script
Files: movie_ingestion/final_ingestion/ingest_movie.py, db/postgres.py, search_improvement_planning/v2_search_data_improvements.md, db/migrate_split_franchise_columns.py

### Intent
Address code-review findings from the prefer_lineage implementation and provide a one-shot migration path that doesn't destroy unrelated Postgres data.

### Key Decisions
- **`FranchiseEntryIds` dataclass replaces the 3-tuple return.** `write_franchise_data` and `ingest_franchise_data` now return a `@dataclass(frozen=True)` with `lineage` / `shared_universe` / `subgroup` attributes instead of a positional 3-tuple. Aligns with the "more than two related values, return a dataclass" convention and makes call sites self-documenting (`franchise_ids.lineage` vs. positional destructure). Call site in `ingest_movie` updated to attribute access.
- **Example 9 in v2_search_data_improvements.md rewritten to actually demonstrate the fallback it's named after.** Previous version walked through a Fantastic Beasts scenario that never triggered the fallback (lineage bucket non-empty) and then described the fallback in abstract prose. Replaced with a Dark Universe / The Mummy (2017) scenario where the queried name only lives in `shared_universe_entry_ids` across the corpus, so the lineage bucket is empty and the universe bucket gets promoted to 1.0 — matching the title.
- **`fetch_franchise_movie_ids` docstring tightened to describe actual behavior.** Previous text said empty set "is NOT a universal match — see the defensive guard below," but the guard only fires when ALL conditions are empty. New text documents that `None` and empty set both drop the predicate, and notes the executor's upstream normalization (`franchise_name_entry_ids or None`) as the expected contract.
- **Targeted migration over fresh Postgres rebuild.** First iteration wrote `db/rebuild_postgres.sh` that dropped the `postgres_data` Docker volume and re-ingested everything — this would destroy unrelated data (awards, lexical postings, etc.) that the schema change doesn't touch. User pushed back ("Why would I ever want that?"). Replaced with `db/migrate_split_franchise_columns.py`: idempotent `ALTER TABLE` (DROP old column+index `IF EXISTS`, ADD new columns+indexes `IF NOT EXISTS`) + loop over `movie_franchise_metadata` rows calling the existing `write_franchise_data` + `update_movie_card_franchise_ids` helpers. Preserves everything else in Postgres. `lex.franchise_entry` / `lex.franchise_token` stay intact since the token/entry space didn't change; the migration's `write_franchise_data` calls are no-op upserts against those tables. Concurrency capped at 8 via `asyncio.Semaphore` so the max-10 connection pool isn't exhausted. Supports `--schema-only` to apply only the ALTERs.
- **Non-fix from review:** The third review suggestion (validator coercion on `len(franchise_or_universe_names) > 1`) was intentionally declined. The existing two coercions (no-name-axis, SPINOFF) are structurally-meaningless combinations; a len>1 coercion would be a usage convention and would conflict with the existing "validators are for data-quality fixes + presence guards, not cross-field dependencies" convention. Left to the prompt to enforce.

### Testing Notes
- `FranchiseEntryIds` construction and attribute access sanity-checked at implementation time.
- Migration script syntax-checked; not yet executed against a real DB.
- Tests asserting the old 3-tuple return from `write_franchise_data` / `ingest_franchise_data` will fail and need updating alongside the other test updates tracked in docs/TODO.md.

## Stage 1: cap alternatives at 1, add trait extraction, free spins from strict narrowing
Files: schemas/flow_routing.py, search_v2/stage_1.py

### Intent
Two observed Stage 1 problems: (1) `alternative_intents` could reach 2 and combined with 2 spins produced up to 5 downstream branches, despite near-zero real cases where three readings are all genuinely useful; (2) `creative_alternatives` for analogical queries like "rocky but with robots" returned timid literal narrowings ("robots boxing") because the prompt mandated spins be "faithful narrowings" that stay inside the primary's intent. Addressed both with a prompt/schema change rather than an architecture split.

### Key Decisions
- **Alt cap dropped from 2 to 1.** Every worked example in the prompt already used ≤1 alt; no constructed query justified a hard max of 2. Matches authoring practice and caps total branches at 1 primary + 1 alt + 2 spins = 4. Enforced in schema (`max_length=1`) and prompt (ambiguity_analysis now allows at most one "emit as alt" line; alternative_intents guidance updated).
- **New `query_traits` field between `alternative_intents` and `creative_spin_analysis`.** Single-line `traits: X, Y, Z` format (compact; drops an earlier "load-bearing vs incidental" flag idea — per-candidate `[preserves/swaps]` annotation carries that signal better). Field placement matters: structured-output generation runs in declared order, so naming traits after routing/alt decisions but before spin reasoning forces the model to surface what the query is made of before committing to spin potential.
- **Narrowing anchor replaced with trait-preservation anchor.** Old rule: every spin must be a faithful narrowing of the primary's intent. New rule: every spin must preserve at least one trait from `query_traits` and may drop or swap the others. Both narrowings (preserve all, add specificity) and tangents (preserve one, swap another) are now valid. Soft guidance: broad primaries lean narrowing; narrow/analogical primaries lean tangential; moderately-specific can go either way.
- **`creative_spin_analysis` format now requires per-candidate `[preserves: X; swaps: Y]` annotation** referencing specific traits from the `query_traits` line. This forces the creative move to be visible in the trace and gives downstream code a structural hook for validation if wanted later.
- **exact_title / similarity default to `spin_potential: none`.** User's rationale: downstream title/similarity retrieval already expands the neighborhood, so spins there are duplicative. Framed as a default rather than a hard ban in the prompt; can be code-enforced later.
- **"Rocky but with robots" added as a worked example** in both CREATIVE SPINS and creative_spin_analysis to anchor the tangential-spin pattern. Previous examples ("Best Christmas movies for families", "Disney classics") all showed broad→narrowing, leaving the model no template for analogical queries.

### Planning Context
- Considered but deferred: moving spins to a separate LLM call with thinking enabled. Stage 1 is pinned to `gemini-3-flash-preview` with `thinking_budget: 0` ([stage_1.py:663](search_v2/stage_1.py#L663)), which caps creativity regardless of prompt quality. Kept in a single call for now — revisit if the prompt change underdelivers on real traffic.
- Considered but rejected: removing the spin/alternative distinction entirely under the looser rule. Kept them separate because alternative_intents = mutually-exclusive readings of what the user MEANT, spins = adjacent directions that preserve what the user is drawn to. Trait-preservation rule makes this sharp: "swaps every trait" = alternative reading, not a spin.
- Tests intentionally not run or read per `.claude/rules/test-boundaries.md`. Schema changes and prompt restructuring will likely require Stage 1 fixture/test updates in a later testing phase.

### Testing Notes
- Schema shape verified via `FlowRoutingResponse.model_fields` — field order is ambiguity_analysis → primary_intent → alternative_intents → query_traits → creative_spin_analysis → creative_alternatives.
- Real-query validation still needed: the two flagged failure modes (5-branch blowup, timid spins on "rocky but with robots") should be re-tested against the updated prompt.
- Stage 1 notebooks ([search_v2/test_stage_1_to_4.ipynb](search_v2/test_stage_1_to_4.ipynb), [search_v2/test_stage_3.ipynb](search_v2/test_stage_3.ipynb)) are candidates for re-running to spot-check output quality on a spread of query types (broad discovery, narrow analogical, exact_title, similarity).

### Revisions (self-review + debug run)
After the initial edits above, the user asked for a critical self-review against the codebase's prompt-authoring conventions. Eight issues were surfaced; four concrete bugs + one reordering were approved and applied in the same session:

- **Dangling "load-bearing" references removed** ([stage_1.py:420](search_v2/stage_1.py#L420), line 483). The `query_traits` format was compacted to drop the "load-bearing | incidental" flag, but two references to "load-bearing traits" still leaked into the CREATIVE SPINS rules. Rephrased to "traits that carry the user's core pull" in one place; removed from the Tom Cruise example entirely.
- **"Transformation signal" removed from `query_traits` examples.** It's a query-type property, not a trait — a spin can't meaningfully preserve or swap it. Replaced with a rule instructing the model to decompose analogical queries into their concrete trait components (archetype, genre, setting).
- **Muddled "broadens boxing to sports" third spin dropped** from the Rocky worked example. Annotation contradicted the `[preserves: X; swaps: Y]` binary the output format demands. Two clean spins now illustrate both tangent axes.
- **Restored clean "No spins."** for the Tom Cruise 90s negative-control example. Previous soft wording ("Likely no spins. If one is emitted...") could be read as permission.
- **Schema reorder: `query_traits` moved to position 2** (between `ambiguity_analysis` and `primary_intent`), not position 4. User direction: "Place before primary intent and we can see what happens." Rationale: naming traits upfront scaffolds the primary rewrite (understanding composition before writing) in addition to spin reasoning. The cost is trait enumeration runs on every query, including obvious title lookups — acceptable given the latency target.

Final schema field order: `ambiguity_analysis → query_traits → primary_intent → alternative_intents → creative_spin_analysis → creative_alternatives`.

### Debug run validation
Ran [search_improvement_planning/debug_stage_1.py](search_improvement_planning/debug_stage_1.py) across 8 buckets / 21 queries (21.6s total). Full output at [stage_1_debug_output.json](search_improvement_planning/stage_1_debug_output.json).
- **Original failure fixed.** `Disney millennial favorites` now emits 1 alt + 2 spins; previously cited ambiguity but emitted 0 alts.
- **Alt cap enforced.** No query exceeded 1 alt.
- **Negative controls clean.** `Inception` (exact_title), `Tom Cruise action movies from the 90s`, and `movies like The Matrix` (similarity) all returned 0 spins.
- **Trait extraction is concrete and on-topic** across all queries; the `[preserves: X; swaps: Y]` annotations are consistent and reference the declared `query_traits`.
- **Observations (not regressions):** (1) `Scary Movie` (exact_title) emitted 1 spin — the "default to no spins" rule is soft and the model decided a horror-parody angle was strong enough; strict enforcement would require a code gate. (2) Model sometimes uses `swaps:` for additions rather than true swaps (e.g., `[preserves: cozy; swaps: tonight]` on `cozy movie for tonight` — the spin actually adds a modifier rather than replacing "tonight"). Minor conceptual stretch; annotations remain readable. (3) "movies" occasionally appears as a trait (e.g., `traits: movies, dads like`). Noise, not broken.

## Step 2A rewritten: PlanningSlot output + own module + branch-aware prompt
Files: [schemas/query_understanding.py](schemas/query_understanding.py), [search_v2/stage_2a.py](search_v2/stage_2a.py) (new), [search_v2/stage_2.py](search_v2/stage_2.py), [search_v2/test_stage_1_to_4.ipynb](search_v2/test_stage_1_to_4.ipynb), [search_improvement_planning/debug_stage_2a.py](search_improvement_planning/debug_stage_2a.py)

### Intent
Step 2A was over-splitting compound concepts, substituting evaluative words ("best" → "highly rated"), silently dropping filler-that-matters, and emitting phantom slots with no retrieval shape. Root cause mirrored the pre-fix Step 1 failure: free-form reasoning fields with no commitment hook into structured output, and no first-class skip/fold/best-guess verdict. This change applies the per-item-verdict retrofit pattern that resolved the analogous Step 1 failures.

### Key Decisions
- **New output shape: `unit_analysis` → `inventory` → `slot_analysis` → `slots[PlanningSlot]`.** Two trace-then-structured pairs, mirroring Step 1's `ambiguity_analysis → alternative_intents` and `creative_spin_analysis → creative_alternatives` pattern. Per-phrase verdicts (literal / best_guess / filler / fold_into) committed before inventory is written; per-candidate-slot verdicts (emit / fuse_with) committed before slots are written. `PlanningSlot` carries `handle`, `scope ⊆ inventory`, `retrieval_shape` (≤8-word phantom-slot sanity check), `cohesion` (≤15 words), and `confidence: literal | inferred`. Validator: slot.scope ⊆ inventory.
- **Module split: 2A lives in `search_v2/stage_2a.py`, 2B stays in `stage_2.py`.** 2A is the most prompt-sensitive stage and iterates independently of 2B. Isolating the file makes the prompt easier to read and edit. `run_stage_2` in stage_2.py is temporarily `NotImplementedError` — the 2A→2B bridge will be rebuilt when 2B is reworked to consume `PlanningSlot` directly.
- **Branch-dynamic system prompt.** Three branch kinds (primary / alternative / spin) plug a different "YOUR INPUTS" section into an otherwise-identical base. The model is never told to consume fields that aren't in its user prompt. Primary receives `intent_rewrite` + `query_traits`; alternative adds `difference_rationale` (tagged `[shift]` on load-bearing units); spin adds `spin_angle` (tagged `[swapped_in]` on swapped content). Raw query text and `routing_signals` / `display_phrase` / `ambiguity_analysis` are deliberately excluded to close the reinterpretation vector.
- **No-reinterpretation rules explicit and numbered.** Preserve broad evaluative words verbatim (best/good/great/top/favorite/classic). Quote rewrite phrases verbatim. Never add content absent from the rewrite. Best-guess is the escape hatch when a phrase has no native retrieval shape — err broad ("late-1990s action classics"), never hyper-specific ("Die Hard, Lethal Weapon"). Coverage beats precision: a reasonable best-guess beats dropping content.
- **Retrieval families are prompt context, not output.** Eight families listed with one-line summaries for sanity filtering (is this phrase retrievable? do these two phrases go to different families?). 2A never commits a family to any slot — that's 2B's job.
- **`ExtractedConcept` kept alive in the schema.** Still referenced by `_step_2b_user_prompt` and `_run_step_2b_for_concept` inside the unchanged 2B code. It will be removed when 2B is reworked.

### Planning Context
See `~/.claude/plans/this-seems-good-rewrite-cozy-harbor.md` for full design rationale, including the analysis of what inputs 2A needs from Step 1 per branch type and the behavioral tests (fuse-vs-split, filler-vs-actionable, fold-vs-independent, literal-vs-best-guess) that govern each verdict decision.

### Testing Notes
Unit tests (`unit_tests/test_search_v2_stage_2.py`) and notebook cell 4B intentionally broken under this change — both depend on the old `step_2a_response.concepts[ExtractedConcept]` shape. They will be reworked alongside the 2B rewrite. Run `python -m search_improvement_planning.debug_stage_2a` against the Step 1 cache to validate the new 2A behavior; watch for (a) evaluative-word preservation on `"Best Christmas movies for families"`, (b) audience-binding on `"popular with millennial audiences"`, (c) filler verdict on `"good for watching tonight"`, (d) best-guess over drop on any Gen-Z-studio style phrasing.

## Step 2B rewrite: parallel-per-slot retrieval action planning
Files: schemas/query_understanding.py, search_v2/stage_2.py, search_v2/stage_4/types.py, search_v2/stage_4/flow_detection.py, search_v2/stage_4/dispatch.py

### Intent
Rebuild Step 2B from scratch around the new Stage 2A `PlanningSlot`
input shape, replacing the per-concept call that `run_stage_2` had
disabled with a parallel-per-slot fan-out. One LLM call handles one
slot at a time; each call produces either a sibling group of
`RetrievalAction`s or a slot-level skip. Concept identity at Stage 4
is now the slot itself — slot == concept by design, eliminating the
intersection-vs-union combination ambiguity that dogged the previous
per-concept shape. Full design rationale in
[search_improvement_planning/steps_1_2_improving.md](search_improvement_planning/steps_1_2_improving.md)
"Step 2B Redesign Proposal" and plan file
`~/.claude/plans/iterative-mixing-cascade.md`.

### Key Decisions
- **New schema shape.** `RetrievalExpression` → `RetrievalAction`
  with field order reordered for the small-model decision chain:
  `coverage_atoms → description → route_rationale → route → role →
  preference_strength`. Separately, `route_rationale` deliberately
  placed BEFORE `route` (reasoning before commit, same house style
  as Stage 1/2A). `description` is always positively framed — even
  for EXCLUSION actions ("movies released in the 1980s", never "not
  made in the 1980s"). Direction is carried by the `role` field, not
  the description text.
- **Role collapsed from (kind, dealbreaker_mode, preference_strength)
  to (role, preference_strength).** `ActionRole` replaces the
  two-level `ExpressionKind + DealbreakerMode` with a flat
  three-way INCLUSION / EXCLUSION / PREFERENCE enum. Cleaner to
  reason about and eliminates the strength-pairing combinatorics
  (strength only applies to PREFERENCE; enforced by a pairing
  validator).
- **`Step2BResponse` field order: `atom_analysis → skip_rationale →
  actions`.** `skip_rationale` placed BEFORE `actions` (not after)
  so structured-output generation commits the skip-or-proceed
  decision before writing the action list. `atom_analysis` is a
  per-atom verdict trace (the house-style scaffold from Stages 1 /
  2A) committing coverage+expansion, role, and route verdicts in
  one pass.
- **`CompletedSlot` as the orchestrator-assembled wire record.**
  Pairs one Stage 2A slot with its Stage 2B response. `QueryUnder-
  standingResponse` now carries `completed_slots: list[CompletedSlot]`,
  replacing the old `concepts: list[QueryConcept]`. One call per
  slot means one CompletedSlot per slot; Stage 4 iterates this list.
  Deleted: `ExtractedConcept`, `QueryConcept`, `ExpressionKind`,
  `DealbreakerMode`, `RetrievalExpression`.
- **Monolithic system prompt (not branch-dynamic).** Unlike Stage
  2A which dispatches primary/alternative/spin sections, Stage 2B
  sees one prompt that carries all eight family capabilities. A
  branch-dynamic prompt would architecturally commit to 2A's
  advisory `retrieval_shape`, contradicting the "2A is context, not
  truth" principle that permits 2B to reroute when the advisory
  shape's family cannot satisfy the atom.
- **Capability descriptions tuned with explicit CANNOT-do notes.**
  Each endpoint family lists both CAN-do bullets (the available
  sub-dimensions) and CANNOT-do bullets covering the hallucinations
  Stage 2A probing surfaced: metadata global-only popularity (no
  demographic breakdowns), keyword closed-taxonomy (unknowns reroute
  to semantic), franchise_structure title-only (character-centric
  retrieval routes to entity), semantic not-a-hard-filter for clean
  deterministic concepts. Generic one-liners are unsafe for families
  with internal structure.
- **Three named expansion motives with explicit justification
  requirements.** `ambiguity_fan_out`, `paraphrase_redundancy`,
  `defensive_retrieval` — each requires the model to name a
  specific mechanism in the ≤8-word why (which angles, which
  paraphrases, which endpoint gap). Vague thoroughness appeals are
  explicitly banned with a worked example. Default is single
  action; expansion is never a reflex.
- **Kind-layering within a slot is legitimate and called out.**
  A slot may want a keyword-horror INCLUSION + a semantic-scariest
  PREFERENCE simultaneously. The prompt gives a worked example so
  the model doesn't force one role per slot.
- **Skip as first-class, with Pydantic XOR validator.** Empty
  actions iff non-empty skip_rationale, enforced at parse time. The
  prompt treats skip as the correct answer when atoms hit capability
  mismatches — not as a cop-out.
- **Three-layer validation.** (1) Pydantic per-action strength
  pairing. (2) Pydantic response skip-XOR-actions. (3) Orchestrator
  code-side partition completeness (union of coverage_atoms ==
  focal_slot.scope, or skipped). Same math as Stage 2A's partition
  validator, one level deeper. Currently non-retrying — plan notes
  retry-once as a future refinement.
- **Stage 4 concept key uses `slot[{i}]::{handle}`.** Positional
  prefix guarantees uniqueness (handles aren't validated unique
  across slots); handle suffix keeps the key human-readable in
  debug output. Stage 4 scoring math is unchanged — MAX-within /
  additive-across just reads from the new key.
- **Stage 3 generators untouched; field rename happens at the
  dispatch boundary.** `search_v2/stage_4/dispatch.py` reads
  `item.source.route_rationale` and passes it to generators as
  `routing_rationale=` (their existing parameter name). Mild
  naming inconsistency, accepted to keep Stage 3 out of scope.
  Stage 3 internals still take `(intent_rewrite, description,
  routing_rationale)` and nothing else from the action — the
  "self-contained description" principle is preserved.

### Planning Context
Full proposal and principle catalog in
`search_improvement_planning/steps_1_2_improving.md` "Step 2B Redesign
Proposal" section. Approved implementation plan at
`~/.claude/plans/iterative-mixing-cascade.md`. The house-style prompt
scaffolds (per-item verdict trace, skip-as-first-class, ≤N-word
caps, principle-illustrating examples from a disjoint pool, worked
failure→fix pairings, explicit CANNOT-do notes) were carried forward
from the Stage 1 and Stage 2A iterations — this change is primarily
an application of those learnings to a new decision surface, plus
the concept=slot identity shift that enables parallelism at the
call level.

### Testing Notes
Existing unit tests `unit_tests/test_search_v2_stage_2.py` and
`unit_tests/test_search_v2_stage_4.py` are broken under this change
— both import `ExtractedConcept`, `QueryConcept`,
`RetrievalExpression`, `ExpressionKind`, `DealbreakerMode`, all
deleted here. Per the test-boundaries rule they were not modified
in this change; they need to be rewritten around `RetrievalAction`
/ `CompletedSlot` / `ActionRole` in a separate testing pass. The
notebook `search_v2/test_stage_1_to_4.ipynb` also needs a cell
update to call `run_stage_2b(intent_rewrite=..., stage_2a=...)`
instead of the old `run_stage_2(query=...)`. End-to-end validation:
run Stage 2A → Stage 2B → Stage 3 → Stage 4 on ~5 representative
queries spanning single-slot / multi-slot / skip / kind-layered /
defensive-retrieval cases. Confirm: (a) parallel fan-out completes,
(b) partition-completeness validator fires correctly when the model
drops an atom, (c) positive-framed descriptions flow through Stage 3
generators unchanged, (d) Stage 4 MAX-within-slot aggregation groups
sibling actions under the new slot-based concept key, (e)
kind-layered slots route INCLUSION to dealbreaker_sum and
PREFERENCE to preference_contribution without double-counting.

## Step 2B follow-ups: retry + rename cascade + atom-quote strip
Files: search_v2/stage_2.py, search_v2/stage_4/dispatch.py, search_v2/stage_3/award_query_generation.py, search_v2/stage_3/entity_query_generation.py, search_v2/stage_3/franchise_query_generation.py, search_v2/stage_3/keyword_query_generation.py, search_v2/stage_3/metadata_query_generation.py, search_v2/stage_3/semantic_query_generation.py, search_v2/stage_3/studio_query_generation.py, schemas/entity_translation.py, schemas/franchise_translation.py, schemas/studio_translation.py

### Intent
Post-review hardening of the Step 2B rewrite: per-slot retry +
error isolation so one failing slot cannot kill the whole branch;
full `routing_rationale → route_rationale` rename through every
Stage 3 generator (the earlier boundary-adapter approach was
explicitly flagged as tech debt); and quote-stripping for 2A atoms
so embedded user-quoted phrases don't collide with the prompt's
quote-delimited rendering.

### Key Decisions
- **Per-slot single retry.** `_run_step_2b_for_slot` now wraps
  `_single_slot_attempt` with one blanket retry on any exception
  (LLM transport, Pydantic validator, runtime coverage). Retries
  use the same prompt — stochastic decode variance already produces
  enough different output to recover most transient failures.
  Feedback-augmented retry was considered and deferred; not worth
  the implementation complexity until we see empirical retry
  patterns.
- **Error isolation via `gather(return_exceptions=True)`.**
  `run_stage_2b` now collects per-slot results with exceptions, logs
  failures at ERROR level with the slot handle, and drops failed
  slots from the returned `completed_slots`. Preserves tokens spent
  on successful sibling slots. The whole-stage failure path is
  explicit: if every slot failed, raise `RuntimeError` referencing
  the first underlying exception (prevents silent degradation into
  browse-flow when 2B is completely broken).
- **Full Stage 3 rename cascade.** `routing_rationale` →
  `route_rationale` across all 7 generators (award, entity,
  franchise, keyword, metadata, semantic, studio) including
  parameter names, docstrings, error messages, and the prompt-label
  strings the LLM sees in the user prompt (e.g.,
  `f"route_rationale: {route_rationale}"`). Dispatch.py no longer
  needs the boundary-adapter variable. Translation schema doc
  comments in schemas/{entity,franchise,studio}_translation.py
  updated for consistency. Convention compliance — removes the
  lingering alias debt flagged in review.
- **Atom quote-stripping in rendering + validation.**
  `_clean_atom(s) = s.replace('"', '')` applied to every atom in the
  prompt (both focal scope and sibling scopes) and to BOTH sides of
  `_validate_coverage` (so the LLM's echoed-back clean atoms match
  the cleaned scope). 2A's schema retains atoms verbatim for
  roundtrip fidelity; the clean is applied at the prompt / validation
  boundary only. A user query with an embedded quoted phrase (e.g.
  "best" films) no longer produces ambiguous prompt output.

### Testing Notes
Test files `unit_tests/test_search_v2_stage_2.py` and
`unit_tests/test_search_v2_stage_4.py` remain unchanged (both already
broken by the preceding 2B rewrite, per test-boundaries rule). The
`routing_rationale` keyword-arg calls in those test files will need
to be renamed to `route_rationale` in the separate testing pass. The
Stage 3 notebooks in `search_v2/test_stage_*.ipynb` likewise still
reference `routing_rationale=` and need a sweep. End-to-end
validation targets for this follow-up: (a) confirm one intentionally
failing slot (e.g., mock an exception on one of N parallel calls)
does not fail the others; (b) confirm a slot that fails both
attempts is dropped with an ERROR log and the remaining slots still
produce `completed_slots`; (c) confirm a 2A atom containing a
literal quote character (e.g., `best` → `"best"`-style) renders
cleanly in the prompt and validates against the LLM's echoed
`coverage_atoms`.

## Rename search_v2/stage_2.py -> stage_2b.py and rewire notebook Cell 4B
Files: search_v2/stage_2b.py (renamed from stage_2.py), schemas/query_understanding.py, search_v2/stage_3/studio_query_generation.py, search_v2/test_stage_1_to_4.ipynb

### Intent
Tidy-up pass on the Step 2B rewrite: the module file was still named
`stage_2.py` (a legacy name from when it held the full Stage 2
pipeline) despite now containing ONLY Stage 2B. Rename matches the
content; parallel with the existing `stage_2a.py` module; aligns
with the tests / docs / notebooks. Cell 4B of the end-to-end test
notebook still imported `_run_step_2b_for_concept` and manually
reassembled `QueryConcept` objects against the deleted schema, so
it's also updated to the new parallel-per-slot `run_stage_2b` API.

### Key Decisions
- **`stage_2.py` → `stage_2b.py`.** Mirrors the `stage_2a.py` naming;
  no "`run_stage_2`" wrapper remains so the old filename carried no
  semantic meaning. Stage 2 is now the pairing of two sibling modules
  callers invoke in sequence: `run_stage_2a` → `run_stage_2b`.
  Existing internal comments in the file already described it as
  Stage 2B only — no internal rewording needed.
- **Comment sweep.** Two non-test files had stale `stage_2.py`
  references in `#` comments pointing at the old path:
  [schemas/query_understanding.py:12](schemas/query_understanding.py#L12)
  and [search_v2/stage_3/studio_query_generation.py:32](search_v2/stage_3/studio_query_generation.py#L32).
  Updated both to `stage_2b.py`. Historical `DIFF_CONTEXT.md` entries
  are left alone per context-tracking rule — they describe past state
  accurately for when they were written.
- **Notebook Cell 1 imports.** Dropped the two-line "2B still lives
  in stage_2.py" note, replaced `from search_v2.stage_2 import
  _run_step_2b_for_concept` with `from search_v2.stage_2b import
  run_stage_2b`, and removed the unused `QueryConcept,
  QueryUnderstandingResponse` schema imports (only the type is
  referenced downstream, implicitly via the returned object).
- **Notebook Cell 4B full rewrite.** Replaced the 50+ line manual
  gather / QueryConcept-assembly block with a single
  `await run_stage_2b(intent_rewrite=..., stage_2a=step_2a_response,
  provider=..., model=..., **kwargs)` call. Per-slot retry + failure
  isolation + CompletedSlot packaging all happens inside the helper;
  the notebook just iterates the returned `qu.completed_slots` for
  display.
- **Notebook Cell 4B markdown.** Dropped the "Temporarily broken"
  warning (which referred to the pre-rewrite state) and replaced
  with a one-paragraph description of the new parallel-per-slot
  contract: per-slot retry, isolation, skip-as-first-class, slot-as-
  concept for Stage 4 grouping.

### Testing Notes
Both `unit_tests/test_search_v2_stage_2.py` and
`unit_tests/test_search_v2_stage_4.py` still import from
`search_v2.stage_2`; these imports now ImportError under the new
module layout. Per test-boundaries rule, those test files were not
modified in this change and remain owned by the separate testing
pass that will rewrite them around `run_stage_2b` / `CompletedSlot`
/ `ActionRole`. End-to-end manual validation: run the notebook top
to bottom on a multi-slot query (e.g. "scariest Disney animated
movies from the 90s") and confirm (a) Cell 4B produces one
`CompletedSlot` per Stage 2A slot with either actions or a skip
rationale, (b) Cell 5 still sees the new slot-based
`concept_debug_key` shape and runs Stage 3 translate/execute cleanly,
(c) Cell 6 scoring aggregates sibling actions by slot handle.

## Entity endpoint schema cleanup — unify prominence, generalize alternative forms, add alias reasoning scaffold
Files: schemas/enums.py, schemas/entity_translation.py, search_v2/stage_3/entity_query_generation.py, search_v2/stage_3/entity_query_execution.py

### Intent
Fix the entity endpoint's alias-catching gap identified this session:
Joker → Arthur Fleck / Jack Napier, Spider-Man → Peter Parker /
Miles Morales, etc. Two problems diagnosed: (1) `name_resolution_notes`
primed single-form commitment before `character_alternative_names`
was ever considered, and (2) there was no reasoning scaffold
preceding the alternatives field, so the model had no dedicated
budget for branching. Also collapses accumulated parallel actor /
character paths that had grown redundant.

### Key Decisions
- **Rename `lookup_text` → `primary_form`.** Signals "one form
  among possibly many"; pairs naturally with the generalized
  alternatives field. Title pattern case still fits (literal
  fragment is its own primary form).
- **Unify `ActorProminenceMode` + `CharacterProminenceMode` → single
  `ProminenceMode` StrEnum** with all five values (default, lead,
  supporting, minor, central). Applicability enforced by the
  `EntityQuerySpec` validator, not the enum.
- **Validator remaps out-of-scope modes rather than rejecting.**
  Character receives LEAD → CENTRAL; SUPPORTING/MINOR → DEFAULT.
  Actor-table receives CENTRAL → LEAD. Reasoning: "wrong mode for
  this entity" is a misclassification, not malformed output —
  costlier to retry than to translate. Validator also forces
  prominence fields null for non-prominence-eligible entities
  (director-only persons, title_pattern).
- **Merge `prominence_evidence` + `character_prominence_evidence`.**
  Same reasoning shape regardless of entity; one field, one scoped
  applicability rule in the prompt.
- **Rename `character_alternative_names` → `alternative_forms` and
  extend to persons.** Stage-name / legal-name variants and
  formal-vs-short credited forms are real for persons too (just
  less common than multi-incarnation characters). Title pattern
  explicitly excluded — literal substrings have no aliases.
- **Add `alternative_forms_evidence` reasoning field directly
  preceding `alternative_forms`.** Prompts the model to actively
  enumerate: re-castings under different credited names, secret-
  identity pairings, stage-name vs legal-name variants. Closes the
  structural gap where `character_alternative_names` was the only
  non-trivial output without a reasoning scaffold.
- **New field order groups identity together, scoring after.**
  entity_type_evidence → name_resolution_notes → primary_form →
  entity_type → person_category → primary_category →
  alternative_forms_evidence → alternative_forms →
  prominence_evidence → prominence_mode → title_pattern_match_type.
  Identity-shape decisions resolve before scoring knobs.
- **Execution layer extends alias handling to every person path.**
  `_execute_person_specific_role` and `_execute_person_broad` now
  resolve primary_form + alternative_forms → list of term_ids,
  not a single term_id. Max-across-variant-rows already handled in
  `_fetch_actor_scores` / `_fetch_binary_role_scores` /
  `_fetch_character_scores`, so no scoring-logic change was needed.
  Extracted `_collect_normalized_forms` + `_resolve_person_term_ids`
  helpers to avoid duplicating the normalize-dedupe-lookup flow
  across three executors.
- **`name_resolution_notes` reframed as primary-form-only.** Prompt
  language made explicit that alias reasoning belongs in
  `alternative_forms_evidence`, not here. Prevents the old
  "telegraphic note" from implicitly soaking up the model's
  aliasing budget before the alias field is reached.

### Planning Context
User diagnosed two root causes from prompt inspection:
`name_resolution_notes` exemplars ("exact user form", "typo fix",
"surname expanded") were all single-string resolutions, priming
single-name commitment; and `character_alternative_names` was the
one complex output without a preceding evidence field. Confirmed
both intuitions by reading the prompt + schema, then agreed to
collapse the parallel actor/character paths at the same time
rather than in two separate PRs. User also specified the exact
validator remap rule (character: lead→central, supporting/minor→
default; actor-table: central→lead) in place of the reject-on-
mismatch pattern I initially proposed.

### Testing Notes
- Unit tests in `unit_tests/` reference the old field names
  (`lookup_text`, `actor_prominence_mode`, `character_prominence_mode`,
  `character_alternative_names`) and the old enum names
  (`ActorProminenceMode`, `CharacterProminenceMode`). Per
  test-boundaries rule, those files were not updated in this
  changeset; they will need renaming in a separate test pass.
- `search_v2/test_stage_3.ipynb` still references the old field
  names in cells that assemble EntityQuerySpec by hand — any
  notebook-driven smoke should be re-run after the test pass
  updates the cells.
- Manual verification queries to run end-to-end:
  * "joker" → primary_form "The Joker", alternative_forms should
    include "Arthur Fleck" and "Jack Napier".
  * "spider-man" → primary_form "Spider-Man", alternatives should
    include "Peter Parker" (also "Miles Morales" if broadly scoped).
  * "indiana jones" → primary_form "Indiana Jones", alternative_forms
    should include "Indy" only if genuinely credited that way.
  * "the rock" → primary_form depends on canonical credit; aliases
    between "Dwayne Johnson" and the stage name form should appear.
  * Plus the 8 sample queries previously provided for broader
    coverage (hyphen variants, diacritic titles, cameo mode, etc.).
- Validator behavior: an LLM that emits prominence_mode=CENTRAL
  with entity_type=PERSON/person_category=ACTOR should be silently
  remapped to LEAD; an LLM that emits prominence_mode=LEAD with
  entity_type=CHARACTER should be remapped to CENTRAL. Confirm via
  unit coverage when the test pass lands.

## Entity endpoint alias-evidence — replace single-sentence verdict with three-question procedural walk
Files: search_v2/stage_3/entity_query_generation.py, schemas/entity_translation.py
Why: Initial test on obvious cases (The Rock → Dwayne Johnson, Superman → Clark Kent, Wolverine → Logan) produced empty alternative_forms every time. Diagnosed three compounding causes in the evidence-field guidance: (1) "a single short sentence" cap compressed away the enumeration step; (2) "no additional credited forms known" default sentinel acted as a tractor-beam exit ramp; (3) enumerate-and-justify shared one field, so justification won.
Approach: alternative_forms_evidence is now a three-question procedural walkthrough per entity type (secret identity / multi-incarnation / long-short variant for characters; stage-name / composite form / formal-short variant for persons), explicitly requiring each question be answered in order before a one-line `therefore alternative_forms = [...]` summary. Validator default changed from `"no additional forms considered"` (reads like a canonical answer) to `"walkthrough skipped"` (reads as a process failure, doesn't prime).

## Entity endpoint alias-evidence redesign — coherent 4-section pass to unblock alias enumeration
Files: search_v2/stage_3/entity_query_generation.py

### Intent
First The Rock test produced "Q1. Yes — he is also credited as Dwayne Johnson" followed by `alternative_forms = []` — a consistency failure between Q-answers and summary. Diagnosis surfaced three independent problems plus cross-section contradictions in the prompt. Rather than patch iteratively (which was already introducing contradictions), applied one coordinated pass across four sections.

### Key Decisions
- **Questions produce names, not yes/no.** Each Q now resolves to either a concrete credited-name string or the literal word "none". Prevents the "yes, but empty list" consistency failure by forcing the Q to commit to a name the summary can then either include or actively contradict.
- **Cost asymmetry taught explicitly in `_ALTERNATIVE_FORMS`.** Retrieval takes the max across forms → spurious alias scores zero → over-including costs ~0, under-including is a silent retrieval bug. Frames inclusion as the default stance.
- **Inclusion bar explicitly lowered.** "Demonstrably appears as a main credit string" → "plausibly appears in at least one film featuring this entity". General knowledge is now validated as the signal we want rather than treated as "unverified".
- **Scoping sentence added to `_NAME_CANONICALIZATION`.** "These rules govern primary_form only. Inclusion of additional credited forms follows the Alternative Credited Forms section, where the default stance is deliberately inclusive." Prevents primary_form's "never invent middle names" discipline from bleeding into alias enumeration.
- **Removed "if in doubt leave it out" language throughout.** That phrase directly contradicts the cost-asymmetry frame. Replaced with inclusion-biased wording everywhere it appeared.
- **`alternative_forms` field guidance rewritten** so it refers back to the walkthrough output directly: "every non-'none' name from the three questions belongs here" — no last-minute filtering, no "only add when genuinely known" hedge.
- **Single source of truth per concern.** `_ALTERNATIVE_FORMS` (tutorial) owns cost asymmetry + inclusion bar + exclusion list. Field guidance owns the procedure. No overlap, no redundant (and potentially contradictory) restatement.

### Testing Notes
Same test cases from before (The Rock, Superman, Wolverine, Darth Vader, Ice Cube, 50 Cent) should now produce non-empty alternative_forms. Watch for the opposite failure (over-production of implausible forms) on the no-alias controls (Denis Villeneuve, "midnight" title pattern) — empty list / null is still correct there. If over-production appears, the next tuning knob is the Q wording, not the cost-asymmetry frame.
