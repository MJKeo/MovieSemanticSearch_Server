# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Keep a described story as one aspect in Step 3 aspect enumeration
Files: search_v2/step_3.py
Why: Step 3 was intermittently over-splitting a single coherent story-shape trait — its characters, relationships, and events each look independently searchable, so the aspect-enumeration step would shatter them into separate aspects that then routed to distinct categories (FACETS), turning one intended search into several. Step 2 already keeps the plot whole as one trait (its ATOMICITY PLOT SHAPES rule); Step 3 was re-shattering it.
Approach: Added a `DESCRIBED STORIES ARE ONE AXIS` block to `_ASPECT_ENUMERATION`, placed at the decomposition point (where the over-split is born) rather than the downstream CONSOLIDATION step. Rationale: the desired end-state is ONE coherent premise string going to a single semantic call, and Step 3's verbatim-copy discipline (dimensions/expressions copy aspects character-for-character, no re-authoring) means a single whole-story expression can only exist if aspects keeps it whole — consolidation can cut call count but cannot re-fuse fragment expressions. The block is framed as an application of the section's existing INDEPENDENCE test (a story's pieces co-vary, so they collapse) plus Step 2's "describe-the-story vs name-a-property" discriminator, so it reinforces rather than contradicts existing guidance. No category names in the block (the section forbids translating into category vocabulary; routing follows downstream) and no query/output examples (prompt-authoring preference). Reconciled the latent contradiction with the CARDINALITY rule two ways: changed "several simultaneous conditions" → "several independently-varying conditions" (co-occurring story pieces are not several conditions), and added an explicit paragraph in the block naming the cardinality rule and resolving it.
Design context: placement decision (aspects over consolidation) discussed at length this session; mirrors Step 2's choice to handle plot shapes at ATOMICITY, not its commit phase.
Testing notes: Validated with the Step 3 batch harness (search_v2/category_candidates_experiment), 5 runs/query, fixed Step 2 input, before vs after. Story-shape variance reduced: "loser guy gets hot girl" 3×SOLO+2×FACETS → 5×SOLO; "loser guy pursues a popular girl" 4×SOLO+1×FACETS → 5×SOLO. Over-consolidation guards held (hidden gem 5×FACETS, scary horror / clean comedy 5×FACETS, japanese 5×FRAMINGS, all single-axis traits unchanged). Not improved: "washed-up boxer making a comeback" still 1/5 FACETS via Central-topic (subject-vs-story read, not the character/event split targeted; within baseline variance). Watch: "war films" produced 1/5 FACETS[Genre, Plot events] (genre query, not a story; likely noise — dominant shape unchanged). Inclusion-only absence scan 10→12 hits, all new ones false positives ("present, not absent" affirmations tripping the regex).

## Guard Step 2 against inferring spoken/audio language
Files: search_v2/step_2.py | Added a hard fidelity guardrail in `_EVALUATIVE_INTENT` so audio/spoken language enters `evaluative_intent` (and thus traits) only when the user explicitly names the spoken language — never inferred from setting, origin, or depicted culture. Phrased as a principle (no query examples, no meta-instructions) to match prompt-authoring preferences.

## Surface Step 2 `intent_exploration` on the `branch_traits` SSE event
Files: search_v2/streaming_orchestrator.py, api/main.py, run_gradio_ui.py
Why: The frontend chain-of-thought view wants the branch-level "here's how we read your request" prose. Step 2 already generates it (`QueryAnalysis.intent_exploration`) but it never crossed the wire. Separately, the per-trait `contextualized_phrase` we were streaming is degenerate (mirrors `surface_text`), so it was replaced on the wire with the richer `evaluative_intent` (the trait's evaluative substance carried from its source atom(s)).
Approach: Added `intent_exploration` as a top-level sibling of `traits` in the `branch_traits` payload built by `_handle_step2_done` — it's a required field on `QueryAnalysis` and the event only fires on the Step 2 happy path (failure paths skip the event entirely and emit `branch_results` with `branch_error`), so absence is satisfied structurally with no guard. In `_trait_to_dict` (the SSE serializer, only called by this event) swapped `contextualized_phrase` for `trait.evaluative_intent` — both are required `Trait` fields; `contextualized_phrase` stays in the schema and Step 2 generation because Step 3 endpoint translation consumes it downstream, the change is wire-only. Updated the contract-doc comment blocks (orchestrator header trait-key list, Gradio shape comment; the `/search` endpoint docstring describes the event without enumerating per-trait keys). Gradio dev UI surfaces the branch-level prose: new `intent_exploration` state map, reducer stores it from the event, `_intent_exploration_md` helper renders it above the traits line (the compact per-trait line is unchanged — it never rendered the swapped field).
Testing notes: no unit tests reference `branch_traits` / `intent_exploration` / `contextualized_phrase` / `evaluative_intent`, so no suite impact. End-to-end check: standard-flow query → each `branch_traits` event carries non-empty `intent_exploration` and each trait carries `evaluative_intent` (no `contextualized_phrase`); non-standard flows emit no `branch_traits`.

## Flatten `/movie_details` crew into a single ranked list
Files: schemas/api_responses.py, api/main.py, db/redis.py
Why: The curated details crew was three independent buckets (directors/writers/producers, each capped at 5) that only ever surfaced Directing/Writing/Production jobs — dropping cinematographer/composer/editor entirely — and ranked "top N" purely by TMDB array order. Goal: one combined, importance-ranked crew list.
Approach: `MovieDetails` now exposes a single `crew: list[CrewMember]` (the three fields are gone). Selection works on a budget of 12 DISTINCT PEOPLE: group TMDB crew rows by person `id`, then fill in priority order — all directors (capped at the global 12), top 3 writers, top 3 producers, then a `_person_fill_rank` top-up that surfaces DP/composer/editor/production-designer (leftover writers/producers re-enter below them). Per the user's call, the backend does NOT dedupe a person to one role: a selected person contributes ALL their credit rows (one `CrewMember` per job), counting as +1 toward the 12; the frontend dedupes by person for display. So the emitted list may hold >12 rows but ≤12 people. `crew_truncated` now means "more than 12 distinct crew people existed." Selection factored into `_group_crew_by_person` / `_select_crew_people` / `_person_fill_rank` helpers; `_extract_credits` / `_ExtractedCredits` updated accordingly. `_build_movie_credits` (`/movie_credits`) is untouched — still the uncapped department-grouped "See all".
Design context: backend-is-authoritative ordering convention (server fixes crew order) preserved; truncation derived from real counts, never a cap constant.
Cache: bumped `_MOVIE_DETAILS_KEY_PARTS` to `("tmdb","movie","v2")` in db/redis.py — the warm path serves cached bytes verbatim (no decode), so stale v1 entries would otherwise hand the old three-bucket JSON straight to the client for up to the 24h TTL. A fresh namespace orphans old keys to expire unused.
Testing notes: no existing tests cover `_extract_credits`; verified via throwaway smoke (co-directed films, auteur counted once w/ all rows, 6+ writers/producers → top-3 + leftover fill, below-the-line heads beating earlier noise crew, 15-director cap, nameless/missing-id rows). A later phase should add unit coverage for crew selection + the new helpers. Frontend contract changed — see writeup.

## Revise `/movie_credits` caching: cross-populate from `/movie_details` + lean cold fetch
Files: db/tmdb.py, api/main.py, docs/modules/api.md
Why: Supersedes the "independent caches / `/movie_details` untouched" decision in the entry below. Code review surfaced that the original `/movie_credits` reused the heavy `fetch_movie_details_for_endpoint` (six appended sub-resources) but consumed only `credits`, wasting upstream latency/bandwidth on the cold path. The "See all" follow-up always comes after a details view, which already fetched the full credits — so caching them there is nearly free and makes the follow-up a warm hit.
Approach:
- `db/tmdb.py`: parameterized `fetch_movie_details_for_endpoint` with an `append` kwarg (default `_API_DETAILS_APPEND` — byte-identical behavior for `/movie_details`), and added a thin `fetch_movie_credits_for_endpoint` wrapper that passes the credits-only `_API_CREDITS_APPEND`, sharing the retry/429/cooldown machinery rather than duplicating it.
- `api/main.py`: new shared helper `_encode_and_cache_credits(tmdb_id, payload)` is the single write codepath for `tmdb:credits:{id}` (honors the single-codepath-writes instinct). `/movie_details` cold path now calls it to cross-populate the credits cache from the payload it already fetched (best-effort, never blocks the details response). `/movie_credits` is now cache-first: hit → verbatim; miss → index 404 → lean credits-only fetch → build/cache/return via the shared helper.
Design context: this is the asymmetric refinement of the cross-population option originally weighed during planning; it dominates both the independent-cache design and a symmetric cross-population (common path = free warm hit, rare path = cheap fetch). Aligns with PROJECT.md priority #2 (latency) at a small, bounded cost to #4 (simplicity — one extra helper + one TMDB kwarg).
Testing notes: verified shared helper returns bytes identical to what it caches, swallows cache-write failures (graceful degradation), and produces correct ordering; `fetch_movie_credits_for_endpoint` delegates with credits-only append; details-fetch default append unchanged. A later test phase should add: `/movie_details` cross-populates the credits cache; `/movie_credits` cache hit skips TMDB entirely; cold-path uses the lean fetch.

## Add `GET /movie_credits/{tmdb_id}` — full uncapped cast & crew
Files: schemas/api_responses.py, db/redis.py, api/main.py
Why: The frontend's "See all" affordance on the details page needs the complete cast and crew, but `/movie_details` deliberately caps cast at 12 and crew at the top 5 of three job buckets to keep the common-path payload lean. This is the lazy opt-in companion: same upstream TMDB credits data, no caps, fetched only when the user asks.
Approach:
- **New structs** in `schemas/api_responses.py`: `CrewGroup { department, members }` and `MovieCredits { tmdb_id, cast?, crew? }`. Reuses the existing `CastMember` / `CrewMember` shapes unchanged — no new member types. `omit_defaults` drops empty `cast`/`crew` and absent `character`/`profile_url` on the wire.
- **New builder** `_build_movie_credits(tmdb_id, payload)` in `api/main.py` (sibling to `_extract_credits`): pure, reuses `_image_url`/`_PROFILE_SIZE`. Cast in TMDB billing order, no cap. Crew grouped by the canonical TMDB `department` field via an insertion-ordered dict (one entry per (person, job) — no merging); group order leads with `_PRIORITY_CREW_DEPARTMENTS` ("Directing","Writing","Production") then remaining departments in first-seen order. Defensive `or "Crew"` fallback for missing department so a real credit is never dropped.
- **New endpoint** structurally mirrors `/movie_details` (Redis warm path with graceful degradation → `fetch_movie_card_row` index 404 → shared-client TMDB fetch → build → encode → best-effort cache) and reuses its exact 404/502/422 error semantics so the frontend reuses error handling. Reuses the same `fetch_movie_details_for_endpoint` (its append_to_response already includes `credits`); card row's reception_score is unused (index check only).
- **Caching:** independent cache (chosen over cross-populating `/movie_details`) under new namespace `tmdb:credits:{id}`, 24h TTL. `db/redis.py` gains `get_cached_movie_credits`/`cache_movie_credits` mirroring the movie-details cache. `/movie_details` left untouched.
Testing notes: covered by ad-hoc build verification (ordering, one-entry-per-credit no-merge, omit_defaults, nameless-skip, empty-credits → `{tmdb_id}` only, route registration). A later test phase should cover `_build_movie_credits` grouping/ordering and the three endpoint error paths (404 not-in-index, 404 not-on-TMDB, 502 TMDB fail).

## Add per-section truncation flags to `GET /movie_details/{tmdb_id}`
Files: schemas/api_responses.py, api/main.py | Added optional `cast_truncated` / `crew_truncated` booleans to `MovieDetails`, each derived in `_extract_credits` by comparing the full pre-cap/pre-filter TMDB count against the returned count (crew compares against the entire crew array, so it captures both bucket-cap loss and unsurfaced departments). Purely additive; `omit_defaults` omits them when False (frontend treats absent as False), so each section can drive an independent "See all" link.

## Add `POST /attribute_search` — hard-attribute browse endpoint
Files: api/main.py, search_v2/attribute_search.py (new), db/postgres.py, docs/modules/api.md
Why: Neither `/query_search` (full NLP pipeline) nor `/similarity_search` (anchor-set similarity) covered the deterministic browse case — "show me top movies that satisfy these hard attributes." Users want to filter by genre / language / streaming service / release-runtime-maturity ranges plus named people (with optional credit-role restriction) and get back the top movies ranked by the same 80/20 popularity-vs-reception prior the V2 pipeline uses as its no-LLM neutral fallback.
Approach:
- **Extended** existing `fetch_neutral_reranker_seed_ids` (db/postgres.py) with a `restrict_movie_ids: Optional[set[int]] = None` kwarg rather than duplicating the SQL. When None, behavior is byte-identical to today; when set, the query adds `AND movie_id = ANY(%s::bigint[])` so the same 80/20 ranking runs against a pre-computed candidate pool. Empty restrict set short-circuits without a round trip. Existing V2 reranker-only fallback callers continue to work via the default.
- **New module** `search_v2/attribute_search.py` owns the orchestration: normalize + dedupe `(normalized_name, role)` tuples, one batched `fetch_phrase_term_ids` call for all distinct names, per-person posting-table lookups (specific role → one table; unset role → parallel asyncio.gather across all five `PEOPLE_POSTING_TABLES` set-only — no prominence scoring), intersect movie_id sets across people (AND), then call the extended ranker with the intersection as `restrict_movie_ids`. No-people path skips the restrict set entirely and ranks the whole catalog (filter-respecting).
- **API handler** in `api/main.py` exposes wire-level `PersonInput { name, role? }` + `AttributeSearchBody { filters?, people? }`. Role is a string Literal at the wire (one of actor/director/writer/producer/composer); a missing role maps internally to `PersonCategory.UNKNOWN`. Reuses `_to_metadata_filters` (422 on bad enum) and the shared `_json_encoder` (msgspec) for response encoding. Default limit 250; `people=None` and `people=[]` treated identically; unresolvable person names return empty list silently per user spec.
- Dispatch dict `_PERSON_CATEGORY_TO_TABLE` defined locally in the new module includes ACTOR (the entity-endpoint version excludes it because that path uses prominence scoring — distinct concern here).
Testing notes: end-to-end verification per plan — no-filter empty body → top 250 overall; genre-only filter; person with role (restrict to that role's posting table); person without role (union across all 5); intersection across multiple people; full combo (filters + multi-person); unresolved name → []; bad enum → 422; response shape parity with /similarity_search.

## Rework `/attribute_search` people ranking → prominence bands + union-sum
Files: search_v2/attribute_search.py, search_v2/actor_zones.py, search_v2/endpoint_fetching/entity_query_execution.py, api/main.py
Why: The people filter was set-membership only — a person's cameo and lead sorted identically, and multiple people AND-intersected. User wants prominence-driven ranking (mirroring the lexical entity flow's actor billing bands), a small popularity nudge that sorts *within* a tier but never across, and multiple people scored as a sum rather than an intersection.
Approach:
- **People path rewritten** in `search_v2/attribute_search.py`. Each person resolves to `{movie_id: best_band_weight}` via a new `_resolve_person_bands`, which now fans across ALL five posting tables (was: one table for a specific role) so a credit in the "wrong" role still surfaces. Band assignment: actor table → zone band (LEAD/SUPPORTING/MINOR from `zone_label`) when actor is the requested role or role is UNKNOWN, else SUPPORTING; non-actor table → LEAD when it matches the requested role or UNKNOWN, else SUPPORTING. Band weights are well-separated ints (LEAD=3, SUPPORTING=2, MINOR=1) so summed tiers stay exact. Cross-person combine is now UNION + SUM (replaces AND intersection). Final sort = (summed_band DESC, 80/20 prior DESC, movie_id DESC), prior computed in Python via `_neutral_prior` reusing `NEUTRAL_RERANKER_SEED_*_WEIGHT`; the prior ∈ [0,1] is a strictly secondary key so it can't cross a band tier. Ranking moved SQL→Python for the people branch; `fetch_neutral_reranker_seed_ids` still backs the unchanged no-people path.
- **Shared zone label**: added `zone_label(billing_position, cast_size)` to `search_v2/actor_zones.py` and refactored `_actor_prominence_score` (entity_query_execution.py) to call it — keeps attribute-search band cutoffs and the entity-flow scorer on one source of truth. Pure refactor, no entity-flow behavior change.
- **Union semantics fix**: an empty/unresolvable name now contributes an empty dict (skipped) instead of collapsing the whole result to `[]`; only an all-empty people list returns `[]`. `(name, role)` dedup unchanged (distinct roles for one name each contribute, by design).
- **api/main.py**: docstring-only update to describe union-sum + prominence-band ranking (cross-role SUPPORTING penalty); handler signature and call site unchanged.
Design context: requirements confirmed with user (sum replaces intersection; lead/supporting/minor bands; cross-role fallback at SUPPORTING; UNKNOWN MAX-merge; nudge keeps the 80/20 split; dedup unchanged). Reuses the bucket-then-tie-break pattern from db/reranking.py.
Testing notes: no `unit_tests/test_attribute_search.py` exists yet — proposed in plan (lead-vs-minor order, cross-role→SUPPORTING, two-person sum > one-person, union surfaces single-person matches, UNKNOWN MAX-merge, within-tier order follows the prior, empty/unresolvable name doesn't zero the result, no-people path still uses the neutral seed). Mock the `db.postgres` fetchers. Per-person fan-out hits all 5 tables (cost rises, bounded by `_MAX_ATTRIBUTE_SEARCH_PEOPLE=20`, run in parallel).

## Code-review follow-ups for the `/attribute_search` prominence rework
Files: search_v2/attribute_search.py, api/main.py, db/postgres.py, docs/modules/api.md
Why: Apply the actionable findings from the in-session code review.
Approach:
- **Lighter actor fetch on the wrong-role path** (`_resolve_person_bands`): the actor table now uses the heavy `fetch_actor_billing_rows` only when actor is the requested role or `role` is UNKNOWN (the only case zone prominence is used). When a *non-actor* role is requested, the actor table is a flat `fetch_movie_ids_by_term_ids` presence lookup scored at `WRONG_ROLE_BAND` — same result, smaller payload, billing fetch skipped. All lookups still run in a single `asyncio.gather`.
- **Double-count documentation**: noted in both the `/attribute_search` endpoint docstring (api/main.py) and the module docstring (attribute_search.py) that `(name, role)` pairs sum independently, so the same person under two roles double-counts — frontend should send one chip per person.
- **Dead-param cleanup**: removed the now-unused `restrict_movie_ids` kwarg from `fetch_neutral_reranker_seed_ids` (db/postgres.py) — the people path ranks in Python and no longer restricts the seed; no prod or test caller passed it. The no-restrict SQL/param order is byte-identical to before, so `unit_tests/test_neutral_reranker_seed.py` (calls with `limit=` only) is unaffected.
- **Module doc**: rewrote the stale `/attribute_search` section of docs/modules/api.md (was "intersect/AND, set membership, SQL ranking") to describe union-sum, the LEAD/SUPPORTING/MINOR bands, the cross-role SUPPORTING penalty, and the Python-side within-tier prior tie-break.
Testing notes: verified via an in-session async smoke harness (not committed) — ACTOR zone bands, DIRECTOR wrong-role SUPPORTING with billing fetch skipped (await_count 0), UNKNOWN actor-vs-director MAX-merge, and within-tier prior ordering. Formal `unit_tests/test_attribute_search.py` still pending per the plan.

Review-pass fixes (same change set):
- Bounded `AttributeSearchBody.people` at `max_length=20` (Pydantic Field). Each role=UNKNOWN person fans out to 5 parallel posting-table queries; capping at 20 prevents a single request from overwhelming the Postgres pool (max_size=10).
- `attribute_search.run_attribute_search`: copy smallest per-person set into `intersection` instead of aliasing `per_person[0]` — `&=` no longer mutates an element of the gathered list. Extracted the `term_id` dict-lookup into a small helper using `.get()` for readability.
- Inlined the `_PersonRoleLiteral` module-level alias into `PersonInput.role`; it was only consumed once.

## Docs staleness fixes from /run-docs-auditor-agent
Files: docs/conventions.md, docs/PROJECT.md, docs/modules/{classes,db,llms,schemas,search_v2}.md, search_v2/step_3.py
Why: The docs-auditor run surfaced 16 staleness issues spanning factually-wrong invariants, stale architecture descriptions, and missing Key Files entries. Fixed all in one pass.
Approach:
- conventions.md + classes.md: corrected watch-provider key encoding from `<< 2` to `<< 4` (matches helpers.py:201).
- schemas.md: removed claim that `Movie` has `concept_tags_run_2_metadata` field; clarified `concept_tag_ids()` reads only the merged `concept_tags_metadata`; updated `EndpointRoute` count from 9 to 11 (added NEUTRAL_SEED, CHRONOLOGICAL); reworded `ActionRole` from "deleted" to "superseded (retained for legacy reranking)"; marked `flow_routing.py` as dead code; added `chronological_translation.py` Key Files entry.
- search_v2.md: updated Step 3 config to `thinking_level="minimal"` + temperature 0.15 (matched the runtime dict, not the prior stale comment); removed "pending wiring" claim for `chronological_query_execution.py` (it is wired via endpoint_executors.py); added `streaming_orchestrator.py` Key Files entry as the production API entry point.
- llms.md: corrected generator count from 10 to 12; updated schemas path from `movie_ingestion/metadata_generation/schemas.py` to `schemas/metadata.py`.
- db.md: added `chronological_scoring.py` Key Files entry.
- PROJECT.md: Stage 5 description now mentions the two hard gates (title-type, missing-text); test count corrected from 76 to 77.
- search_v2/step_3.py: fixed two internal comments (lines 49 and 1350) that said `thinking_level="low"` to match the actual `_MODEL_KWARGS` value of `"minimal"`. Runtime behavior unchanged — only stale comments corrected.
Design context: ADR-090 (Step 3 loose5 changes), ADR-094 (concept_tags three-run batch pipeline). No new ADRs; all changes are doc/comment alignment, not behavioral.

## Soften Step0Response validators (truncate instead of reject)
Files: schemas/step_0_flow_routing.py
Why: The three `@model_validator` checks on `Step0Response` were rejecting LLM payloads that could be safely normalized. Failing when recovery is trivial wastes a generation and surfaces noise downstream.
Approach:
- Replaced `_entities_match_flow_cardinality` with `_truncate_selected_entities_to_flow_cardinality`: NONE_OF_THE_ABOVE → `selected_entities = []`; single-entry flows (SPECIFIC_TITLE, CHARACTER_FRANCHISE, NON_CHARACTER_FRANCHISE) → keep first 1; list-style flows (SIMILARITY_TO_TITLES, STUDIO, ACTOR) → untouched. Mutates `self` rather than raising.
- Removed `_qualifiers_force_no_entity` entirely. Qualifiers no longer force NONE_OF_THE_ABOVE at the schema layer — the prompt still teaches the convention but downstream tolerates the LLM ignoring it.
- Removed `_selected_entities_match_flow_kind` entirely. The cross-reference against `entity_candidates.most_likely_kind` is redundant — the LLM already commits to the flow + entity payload, and the `to_*_flow_data` adapters reshape from `EntityReference` (canonical_name + release_year) into whatever the executor needs without consulting candidate kinds.
- Dropped the now-unused `_EXPECTED_KIND_FOR_FLOW` mapping. `EntityKind` is still used by `EntityCandidate.most_likely_kind`.
Testing notes: smoke-tested all three former failure modes — NONE_OF_THE_ABOVE with stray entities, SPECIFIC_TITLE with N>1 entries, and ACTOR with qualifiers + mismatched candidate kind — each now constructs successfully and the adapter output is well-formed. Existing unit tests that asserted the validators raise will need to be updated in the testing phase.

## Add `branch_categories` SSE event to streaming search pipeline
Files: search_v2/streaming_orchestrator.py, api/main.py
Why: The frontend needs a progressive reveal between `branch_traits` (Step 2 done) and `branch_results` (Stage 4 done) so it can expand each trait into its (category, expressions[]) decomposition before final results land. The data already exists as `TraitWithEndpoints.category_calls` on `Step2BranchResult.traits` — it just wasn't on the wire.
Approach: New SSE event `branch_categories` fired per standard-flow branch from `_handle_finished_task`'s `_PHASE_STEP3` arm, after the branch-level error check and before `emitter("executing")` queues the next stage flip. Direct `yield` lands ahead of the queued stage event, preserving the order traits → categories → executing. Non-standard flows are skipped automatically because they never enter `_PHASE_STEP3`. Two new module-private serializers (`_category_call_to_dict`, `_trait_with_endpoints_to_dict`) keep the wire shape centralized next to `_trait_to_dict`. Per the spec, category calls with `bucket is HandlerBucket.EXPLICIT_NO_OP` or empty `expressions` are filtered out (independent conditions, both required); per-trait `step_3_error` short-circuits that trait's `category_calls` to `[]`. Generated specs / retrieval_intent / combine_mode are deliberately omitted — additive extension is cheap if the UI needs them later.
Design context: Plan at /Users/michaelkeohane/.claude/plans/here-s-a-description-of-golden-dongarra.md. EXPLICIT_NO_OP detection follows the canonical pattern in search_v2/endpoint_fetching/category_handlers/handler.py.
Testing notes: needs end-to-end SSE verification — standard-flow query should emit `branch_categories` once per branch in the expected position; non-standard flows (similarity, exact_title, franchise, studio, actor) should never emit it. Manual checks for EXPLICIT_NO_OP filtering (BELOW_THE_LINE_CREATOR is the only such category today), empty-expressions filtering, and per-trait `step_3_error` rendering.

## Decouple semantic elbow calibration from hard filters + tiered promotion loop under filter narrowing
Files: search_v2/endpoint_fetching/semantic_query_execution.py, search_v2/stage_4_execution.py, search_v2/full_pipeline_orchestrator.py, search_v2/promotion_tiers.py (new)

### Intent
Fixes two interacting issues that surfaced when diagnosing why queries like "super violent and bloody" + rated-G hard filter returned ~0 results. (1) The semantic elbow probe was passing `metadata_filters` into the Qdrant `must`, so the elbow was being calibrated against the filtered slice — small/noisy slices trigger the flat-distribution pathology or have Kneedle find a fake elbow in noise, manufacturing false positives. The elbow is meant to be an absolute "is-about-X" bar, a property of the global corpus, not the filtered slice. (2) The reranker→generator promotion fallback ran once pre-execution and only fired when no CANDIDATE_GENERATOR spec existed in the branch. When hard filters narrowed keyword/metadata generators to ∅, the branch still had structural generators present and promotion never fired — leaving the user with the "doesn't exist means doesn't exist" empty-pool policy even though a promoted semantic generator could have produced candidates.

### Key Decisions
- **Two-helper split for the semantic probe.** `_run_corpus_topn` is now unconditionally unfiltered (used for elbow calibration); a new `_run_corpus_topn_filtered` runs the filtered probe and is only invoked on the unrestricted/promoted generator paths. The no-filter case fires one probe per space (byte-identical to pre-fix behavior); the filter-active case fires two parallel probes per space (calibration + pool). Reranker paths (`_execute_carver_restricted`) get the fix for free — the probe arm is calibration-only, so dropping the filter just makes the elbow honest.
- **Per-branch tiered promotion loop in Phase B, filter-active only.** Promotion is decided in `_run_branch` after the initial union is built. When `metadata_filters.is_active` and `len(union) < CANDIDATE_FLOOR` (25), find the lowest promotable tier among the branch's rerankers that hasn't been promoted yet, flip those specs to CANDIDATE_GENERATOR (in place; `was_promoted=True`), re-dispatch just the newly promoted specs through the shared dedup-dispatch helper, merge results into the union, and loop. Within-tier dispatch is parallel; tiers run serially so the loop stops at the first tier that brings the pool over the floor. Pre-existing pre-execution fallback (`_apply_reranker_only_candidate_fallback`) is untouched — it still handles the "no generators at all" base case orthogonally.
- **Extracted `PromotionTier`, `_SEMANTIC_PROMOTION_TIERS`, `_METADATA_PROMOTION_TIERS`, `determine_promotion_tier` into `search_v2/promotion_tiers.py`** so `stage_4_execution.py` can consume them without creating an import cycle with the orchestrator. Re-exported through `full_pipeline_orchestrator.py` so existing test imports keep working.
- **Refactored shorts subtraction into `_fetch_shorts_ids` + `_apply_shorts_subtraction`** so the loop can re-apply the blocklist per iteration without paying for repeated DB round-trips.
- **Refactored `_seed_from_neutral` into a thin gate over `_seed_neutral_pool`** so the new filter-active-tier-exhausted path can seed directly when the orchestrator's pre-execution fallback hasn't added a NEUTRAL_SEED aux spec.

### Testing Notes
- End-to-end smoke: API path with "super violent and bloody" + G-only filter. Pre-fix: ~0 results. Post-fix: pool should reach ≥25 via tier promotion (or neutral-seed fallback after exhaustion), and candidates should score honestly against the unfiltered elbow (likely low — which is correct).
- No-filter invariant: any query without filters must produce a byte-identical result list to pre-fix. The semantic executor's `filter_active` short-circuit and Stage 4's filter-active gate together guarantee this.
- Verify `was_promoted=True` propagation through `ScoreBreakdown` / `TraitContribution` rendering in `run_orchestrator.py`.
- Promoted specs are removed from `pos_rerankers` after the loop so Phase C doesn't re-dispatch them as rerankers (would overwrite the generator score with a restricted rescoring).
- Unit tests in `unit_tests/test_full_pipeline_promotion_tiers.py` import `PromotionTier` and `determine_promotion_tier` from `search_v2.full_pipeline_orchestrator` — the re-export keeps those imports working without touching tests.
- Per project rule, no test files were read or modified.

## Review follow-ups for tiered-promotion + elbow decoupling
Files: search_v2/endpoint_fetching/semantic_query_execution.py, search_v2/stage_4_execution.py, docs/modules/search_v2.md
Why: Self-review surfaced one factually-wrong comment, three stale module-doc sections, a missed cross-iteration dedup opportunity, and two small style nits.
Approach:
- Corrected stale `_run_filtered_score` comment to match the new behavior (calibration probe unfiltered, filtered pool fetch lives in `_run_corpus_topn_filtered`). Also reworded the `CORPUS_PROBE_LIMIT` constant comment.
- Updated `docs/modules/search_v2.md` Phase B description, Auxiliary specs section, and Hard Filters section. The Hard Filters section previously asserted "Semantic elbow probe filtered" — directly contradicted by the fix — and is now rewritten to describe the calibration-unfiltered / pool-filtered split. Phase B docs now describe the tiered loop, dispatch_cache, and the dual-path neutral seed. CANDIDATE_FLOOR and the promotion_tiers.py extraction are also called out.
- Added cross-iteration dedup to `_dispatch_generator_specs`: optional `dispatch_cache: dict[(route, params_identity), score_map | None]` parameter. `_run_branch` creates one cache per branch and threads it through both the initial Phase B dispatch and every loop iteration. Specs whose (route, params) is already in the cache reuse the cached score map instead of re-dispatching. The within-batch dedup is preserved as the inner layer.
- Fixed `branch_kind: object` type hint on `_dispatch_generator_specs` to the proper `"BranchKind"` string forward ref.
- Removed the dead `if newly_promoted:` guard in `_run_branch`'s tiered loop. `next_tier` is selected as the min of `remaining_tiers`, which is the set of tiers in `promotable_refs` not yet promoted — so `newly_promoted` is provably non-empty when we reach the dispatch.
Testing notes: no behavioral change in the no-filter case (dispatch_cache populated but never consulted across iterations because there are no iterations). Filter-active loop now reuses initial Phase B dispatch results when promoted tiers carry structurally identical specs — verify via the new "X cross-iteration cache hits" log line in `_dispatch_generator_specs`.

## Generalize Step 0 ACTOR flow to role-agnostic PERSON flow
Files: schemas/enums.py, schemas/step_0_flow_routing.py, search_v2/step_0.py, search_v2/person_search.py (renamed from actor_search.py), search_v2/full_pipeline_orchestrator.py, search_v2/streaming_orchestrator.py, search_v2/run_step_0.py, search_v2/run_step_0_batch.py, docs/modules/search_v2.md

### Intent
The Step 0 ACTOR flow only resolved person names against `lex.inv_actor_postings`, so queries like "david attenborough" — where the person is credited as writer / narrator on most films but rarely as actor — never reached the writer/director/producer/composer posting tables and silently fell back to the standard flow. Generalized the entire flow to PERSON: schema, prompt, executor, orchestrators, runners, and module doc.

### Key Decisions
- **Enum + payload rename, not parallel addition.** `EntityKind.ACTOR` → `EntityKind.PERSON`, `EntityFlow.ACTOR` → `EntityFlow.PERSON`, `SearchFlow.ACTOR` → `SearchFlow.PERSON`, `ActorReference` → `PersonReference`, `ActorFlowData` → `PersonFlowData`, `Step0Response.to_actor_flow_data()` → `to_person_flow_data()`. There's no value in keeping the old ACTOR-only flow alongside — PERSON is a strict superset, and the LLM prompt is unambiguous about no preferred role.
- **No preferred role in the executor.** The Step 0 prompt now teaches that any of the five credited filmmaking roles (actor, director, writer, producer, composer) qualifies a span equally — no need to know "which role the person is most known for." Role markers in the query ("directed by X", "score by Y") count as content that disqualifies the flow into none_of_the_above, which is consistent with how the studio flow handles "movies by Pixar".
- **Per-(person, movie) bucket = MIN across roles.** Actor postings carry `billing_position`/`cast_size` so they bucketize via the existing sqrt-adaptive cast-zone model (lead / supporting / minor top-half / minor bottom-half). The other four role tables have no prevalence data, so any credit there uniformly lands in bucket 1 (LEAD). The MIN reducer lets a person who is both a lead actor and a producer on the same film land at bucket 1 via the actor signal — and means future prevalence-bearing data for non-actor roles drops in without restructuring.
- **Multi-person UNION (not intersection) with overlap-count as primary within-bucket sort.** The previous ACTOR flow intersected: only movies where ALL named actors appeared survived, with a weakest-link MAX bucket reduction. PERSON switches to UNION — any movie where any named person has any credit qualifies. The per-movie bucket is MIN across people-who-appear ("best credit by any named person"); the per-movie overlap_count = how many of the named people contribute. Within-bucket sort is (overlap_count DESC, popularity DESC, movie_id DESC) so the intersection of all named people surfaces above single-person matches inside the same prominence tier. Single-person queries are unchanged in practice — overlap_count is always 1 and sort collapses to pure popularity DESC.
- **Executor reuses existing helpers.** `fetch_movie_ids_by_term_ids(PostingTable, term_ids, ...)` already takes the table polymorphically, and `db.postgres.PEOPLE_POSTING_TABLES` already lists the five role tables. No new postgres helper was added.
- **`actor_zones.py` kept as-is.** The module is shared with `endpoint_fetching/entity_query_execution.py` and the zone model genuinely is actor-billing-specific. Renaming would have rippled into the non-Step-0 entity endpoint.

### Testing Notes
- Smoke: a single-person query like "david attenborough" must now return Deep Blue (2003) and similar nature docs where Attenborough is credited as writer-only, ranked among the bucket-1 results.
- Multi-person union: "Spielberg and Williams" should return films where either appears (huge result set), with films where both appear surfacing first inside each bucket. Compare to old intersection mode which returned only their collaborations.
- Unit tests on actor_search.py (test file names referencing actor flow) will need an update in the testing phase — schemas, executor module name, and payload class names all changed. Per project rule, no test files were touched.
- streaming_orchestrator `branch_stage` events: `_FETCH_ID_ACTOR` → `_FETCH_ID_PERSON`, fetch type `"actor"` → `"person"`, stage label `"resolving_actor"` → `"resolving_person"`. Any frontend listener keyed on the old strings will need to switch.

## Add `maturity_rating` to MovieCard wire payload
Files: schemas/api_responses.py, db/postgres.py
Why: Frontend needs the MPAA label ("G", "PG", "PG-13", "R", "NC-17") to render `<rating> • <year>` on each result card. The data already lives in `public.movie_card.maturity_rank` — only the wire projection was missing.
Approach: Added `maturity_rating: str | None = None` (kept default + `omit_defaults=True`, so the wire shape stays backward compatible for any consumer that doesn't read it). In `fetch_movie_card_summaries`, the existing `fetch_movie_cards` already selects `maturity_rank`; just feed it through a new `_maturity_rank_to_label` helper. The label map is *derived* from the `MaturityRating` enum (`{member.maturity_rank: member.value.upper() for member in MaturityRating if member is not UNRATED}`) so the canonical labels track the enum's source of truth instead of being hardcoded. UNRATED, NULL rank, and unknown ranks all collapse to None — per spec, the frontend omits the rating segment in that case. Ingest already normalizes UNRATED → NULL on write (movie_ingestion/final_ingestion/ingest_movie.py:255-256), but the helper still guards UNRATED defensively in case legacy rows exist.
Testing notes: verify `maturity_rating` round-trip for a PG-13 movie (rank 3 → "PG-13"), an unrated movie (NULL → field omitted from JSON due to `omit_defaults=True`), and that no other MovieCard call site regresses. All callers go through `fetch_movie_card_summaries`, which is the only constructor of `MovieCard`.

## Expand SSE wire payload for click-to-expand UI
Files: search_v2/streaming_orchestrator.py
Why: The frontend is adding click-to-expand reveals on fetch / trait / category / expression nodes of the live search tree. Each reveal needs server-side data that the SSE stream already holds but wasn't shipping. Purely additive — nothing renamed or removed, existing frontend keeps working.
Approach:
- `fetches_ready.fetches[]`: added `query` to every non-standard entity flow (standard branches already shipped it via `branch_query` from `_plan_step2_branches`). Six new module-private helpers — `_join_canonical_names`, `_similarity_query`, `_non_character_franchise_query`, `_character_franchise_query`, `_studio_query`, `_person_query` — build natural-language descriptions of what each branch is fetching (e.g. "the Star Wars franchise", "films featuring Batman", "Christopher Nolan films", "movies similar to Inception (2010), Arrival"). exact_title reuses the existing `_exact_title_label` builder since title-text IS the resolved entity expression.
- `branch_traits.traits[]`: added `contextualized_phrase` to `_trait_to_dict` — the folded user-intent restatement Step 2 already produces on every Trait.
- `branch_categories.traits[].category_calls[]`: added `definition` and `retrieval_intent` to `_category_call_to_dict`. `definition` sources from `call.category.description` on the `CategoryName` enum (same string for every emission of that category — frontend doesn't need a parallel mapping table). `retrieval_intent` was already on `CategoryCallWithEndpoints` — just needed wire-format inclusion.
Design context: Plan at /Users/michaelkeohane/.claude/plans/i-want-to-add-lovely-dijkstra.md. Per-expression `retrieval_intent` deliberately NOT added — it's category-level on the backend; frontend will look it up via parent-category state.
Testing notes: end-to-end SSE check on a query that mixes a standard branch with an entity flow (e.g. "dark gritty marvel movies" hits standard + non_character_franchise). Verify (1) every fetch entry carries `query`, (2) every trait carries `contextualized_phrase`, (3) every category_call carries `definition` + `retrieval_intent`, (4) EXPLICIT_NO_OP / empty-expressions filtering in `_trait_with_endpoints_to_dict` still suppresses the right calls.

## Order non-standard flows before standard branches in `fetches_ready`
Files: search_v2/streaming_orchestrator.py | Stable sort on `fetches` (key = `type == "standard"`) right before the yield, so non-standard flows (exact_title, similarity, non_character_franchise, character_franchise, studio, person) always occupy index 0+ in the wire payload while standard branches retain their planning order behind them. Frontend treats non-standard flows as priority categories and expects them first.

## Include IMDB `self` credit category in cast scraping
Files: movie_ingestion/imdb_scraping/http_client.py, movie_ingestion/imdb_scraping/parsers.py, docs/modules/ingestion.md
Why: IMDB classifies documentary subjects, concert performers, real-life interview subjects, and archive-footage appearances of deceased people under `category.id = "self"` — not `actor`. The previous cast filter (`["actor", "actress"]`) dropped all of those credits at the API boundary, so `lex.inv_actor_postings` (the only table the Step 0 PERSON flow's prominence scoring depends on) never received them. Result: queries like "david attenborough", "amy winehouse", "anthony bourdain documentary" returned zero or wrong-bucket results even after the PERSON-flow rename. A 14-title probe (`/tmp/imdb_expanded_credits/`) confirmed `self` is the only on-screen non-actor category IMDB's GraphQL actually returns (530 obs vs. 270 `actor` across the sample) — `narrator` / `archive_footage` / `archive_sound` / `presenter` are not real category.id values, and `miscellaneous` is too noisy to include (foreign-dub narrators + financiers + researchers all mixed together).
Approach:
- Expanded the `cast` filter at http_client.py:110 to `["actor", "actress", "self"]` and added `category { id }` to each edge so the parser can identify self credits per-edge. IMDB returns the union in canonical billing order — Attenborough at position 1 in A Life on Our Planet, Winehouse at position 1 in Amy, Tom Hanks at position 1 in Forrest Gump — so the existing sqrt-adaptive prominence model adapts automatically. `cast_size` at ingest_movie.py:488 grows naturally with the new edges.
- Added a `continue` in parsers.py's cast loop when `category.id == "self"`, gating character extraction. IMDB returns role-context labels ("Self", "Themselves", "Self - Ecologist", "Narrator") in the `characters` array on self edges; routing 321 "Self" rows across the probe sample alone into `lex.inv_character_postings` would dilute character search without adding signal. Verified on the probe response for tt11989890: "David Attenborough" lands in actors, "Self" does NOT land in characters, "Young David" (the character of Max Hughes who has an actual `actor` credit) is preserved.
Design context: Builds on the PERSON-flow rename earlier in this session. No new posting tables; no Step 0 / executor changes; no schema migration.
Testing notes:
- A re-scrape is needed for existing cached responses at `ingestion_data/imdb_graphql/*.json` — they were captured under the old filter and don't contain self edges. Re-scrape mechanics: reset `movie_progress.status` from `imdb_scraped` (and later statuses) back to `tmdb_quality_passed` for affected records, then run `python -m movie_ingestion.imdb_scraping.run`. `fetch_movie()` always re-hits the API and overwrites the cache (no `--force` flag exists). Followed by re-running Stage 8 (`ingest_movie`) to rewrite `lex.inv_actor_postings` from the updated actor lists.
- Backfill rollout is out of scope for this commit — track as a separate operations task. Could be staged: documentary/biography/music genres first (where self credits matter most), then a full sweep for narrative-film cameos.
- Outlier billing case: ~2 of 10 docs in the probe (Super Size Me, Stop Making Sense) had IMDB ordering `self` credits alphabetically rather than by prominence, so the subject lands in BUCKET_MAJOR/RELEVANT instead of BUCKET_LEAD. Still strictly better than zero results — the bucket model handles it gracefully.

## Add `/movie_details/{tmdb_id}` endpoint (TMDB + reception_score, 24h Redis cache)
Files: api/main.py, db/tmdb.py, db/redis.py, db/postgres.py, schemas/api_responses.py

### Intent
New `GET /movie_details/{tmdb_id}` endpoint that returns a curated detail payload for a single movie — overview, runtime, MPAA rating, cast/crew, watch providers, trailer, our reception_score, and outbound TMDB/IMDb links. Backs the click-through detail view from a search result tile. Cached for 24h in Redis to keep the warm path off TMDB.

### Key Decisions
- **404 on movies absent from `public.movie_card`.** The endpoint is for our index, not a generic TMDB proxy — keeps the contract single-shape and matches `/similarity_search`'s "must exist locally" stance. The Postgres lookup also supplies reception_score.
- **Curated msgspec.Struct, not pass-through.** New `MovieDetails`, `CastMember`, `CrewMember`, `WatchProvider` in `schemas/api_responses.py`. Stable wire contract decoupled from TMDB's payload churn; ~10× smaller than the raw response. `WatchProvider` field order: required fields (provider_id, name, access_type) precede optional logo_url — msgspec rejects optional-before-required at class build time.
- **Cache the encoded curated payload, not raw TMDB JSON.** Warm hits skip both the TMDB round-trip and the build/encode step. Cache key `{REDIS_ENV}:movie_details:{tmdb_id}`, value is the exact bytes returned to the client. Cold-path and warm-path responses are byte-identical.
- **New TMDB helper, not parameterized expansion of the existing one.** `fetch_movie_details_for_endpoint` in `db/tmdb.py` mirrors the ingestion-side `fetch_movie_details` but appends `credits,videos,images,external_ids,watch/providers,release_dates` (different from ingestion's `release_dates,keywords,watch/providers,credits,reviews`). Keeping the two paths separate prevents the API view from bloating ingestion payloads or vice versa.
- **Shared httpx client + AdaptiveRateLimiter on `app.state`.** Built once in the lifespan handler via the new `build_api_tmdb_client()` factory (centralizes Bearer-auth + timeout config so API and ingestion can't drift). Single instance amortizes TLS handshake across all detail requests.
- **TMDB-fresh watch providers, not Postgres `watch_offer_keys`.** Streaming availability changes frequently; the Postgres data is frozen at ingest time. The 24h Redis cache softens the freshness vs latency trade-off.
- **`fetch_movie_card_row` delegates to `fetch_movie_cards`.** Simple single-row variant — column projection wasn't worth duplicating the SQL.
- **502 (not 500) on TMDB fetch failure.** Lets the frontend distinguish upstream-down from us-broken for retry/UX.

### Testing Notes
- End-to-end: `GET /movie_details/27205` (Inception) → 200 with overview, `directors[0].name == "Christopher Nolan"`, reception_score populated, trailer_url is a YouTube link, tmdb_url is `https://www.themoviedb.org/movie/27205`.
- Cache path: second call returns same body; `redis-cli KEYS "*movie_details*"` shows the key with TTL ≈ 86400.
- 404 paths: id not in `movie_card` → "movie not found"; id present locally but deleted on TMDB → "movie not found on TMDB" (latter is rare; needs stubbed TMDB response to verify).
- 502 path: invalidate `TMDB_ACCESS_TOKEN` → endpoint returns 502, Redis cache untouched.
- Smoke-tested the builder against a synthetic payload covering certifications, multi-role crew (Nolan as director+writer), top-billed cast truncation, providers across flatrate/buy/rent, and an empty homepage string normalizing to None.

## Add `additional_images` to `/movie_details` response
Files: schemas/api_responses.py, api/main.py
Why: Detail page needs a small gallery of secondary artwork beyond the single hero poster + backdrop. The TMDB `images` sub-resource was already being fetched via `append_to_response` but discarded entirely; this surfaces it as a curated 5-item list.
Approach: New `additional_images: list[str] = []` field on `MovieDetails` (kept `omit_defaults=True`, so empty/missing stays off the wire). New `_extract_additional_images(images, exclude_paths)` helper sorts `payload.images.backdrops` by `vote_count` desc, picks URLs deduped against the primary poster/backdrop file_paths, and tops up from `posters` only if backdrops don't fill the cap of 5. Dedupe uses the raw `file_path` (not the joined CDN URL) so the comparison is size-segment-agnostic. Single shared size `w780` for both surfaces — largest size TMDB supports for both backdrops and posters, so the frontend can render a uniform grid. Helper short-circuits on missing/empty `images` block and treats missing `vote_count` as 0 for ranking.
Testing notes: smoke against a movie with rich backdrop coverage (Inception, Dune) — expect 5 backdrop URLs, none equal to `backdrop_url`. Smoke against a movie with <5 backdrops to confirm posters top up. Verify hot-path Redis cache is invalidated or keyed in a way that picks up the new field on next read (cache stores encoded bytes, so existing cached entries will lack the field until their 24h TTL expires).

## Add follow-up clarification support to /query_search
Files: api/main.py, schemas/step_1.py, search_v2/step_0.py, search_v2/step_1.py, search_v2/full_pipeline_orchestrator.py, search_v2/streaming_orchestrator.py

### Intent
Lets the user send a follow-up correction ("less violent", "more 80s", "actually I meant the comedy one") after seeing a result set, and re-run the search with both the original query and the correction in hand. Targets the case where the initial query missed the mark and the user wants to course-correct without restating the whole search from scratch.

### Key Decisions
- **Ingestion at the query-understanding layer only.** Step 0 and Step 1 see the clarification; Steps 2+ are unchanged. Step 2 still receives a single natural-language query per branch — so all existing prompt discipline (intent_exploration, trait commitment, polarity, positioning), Stage 4 scoring, and the category taxonomy stay intact. Plan at /Users/michaelkeohane/.claude/plans/1-main-plus-2-clever-manatee.md.
- **Step 0 dual prompt.** Existing `SYSTEM_PROMPT` is unchanged. New `CLARIFICATION_SYSTEM_PROMPT` swaps in a new task-and-outcome opening that teaches the model to merge original + clarification under "clarification is authoritative on conflict; original is preserved where clarification is silent; retraction drops material; flow choice is re-evaluated from the merged intent." The rest of the prompt (zones / coverage / resolution / qualifier / ambiguity / similarity / output-field) is shared. `run_step_0(query, clarification=None)` dispatches on presence.
- **Step 1 dual prompt + new schema variant.** New `Step1ClarificationResponse(exploration, main_rewrite: Spin, spins: list[Spin])` reuses the existing `Spin` class so downstream branch-plan unpacking is identical for all three slots. The clarification-mode prompt teaches TWO stances in a single LLM call — faithful translation for `main_rewrite` (no hallucinated detail, no abstractions, no resolving descriptions to specific titles) and creative-but-grounded for spins (which now operate over the rewritten intent, not the raw original). `run_step_1` returns the union type; orchestrators discriminate via `isinstance`.
- **Slot-1 branch-plan rewrite.** `_plan_step2_branches` no longer hard-codes `("original", raw_query, "Original Query")`. When Step 1 returned `Step1ClarificationResponse`, slot 1 carries `main_rewrite.query` + `main_rewrite.ui_label`. Otherwise slot 1 carries `(raw_query, raw_query)` — verbatim user text in both query and label, no static placeholder. Same fix applied to the `skip_bypass_steps_0_1` debug path. Budget priority unchanged (spin_2 dropped first when entity flow co-fires).
- **Boundary normalization at the API.** `api/main.py` strips whitespace on `clarification` and collapses empty strings to `None`, so the no-clarification fast path stays stable when the frontend sends `""` or whitespace-only input.

### Testing Notes
- No test files were read or modified per `test-boundaries.md`. Existing unit tests that fixture the literal `"Original Query"` (`unit_tests/test_full_pipeline_promotion_tiers.py`) and ones that call `run_step_0(query)` / `run_step_1(query)` positionally still compile because the new `clarification` parameter has a `None` default — but the planner-output fixtures will need expected-label updates in the testing phase.
- End-to-end verification per the plan: (1) no-clarification regression — slot-1 `label == query == raw_query` verbatim, no "Original Query" anywhere; (2) clarification path — slot-1 carries pithy `ui_label` + faithful rewrite query, spins reflect corrected intent; (3) `"that indiana jones movie where he runs from a boulder"` + `"actually one with a younger Indy"` must NOT collapse the rewrite to a specific film title; (4) empty/whitespace clarification collapses to the no-clarification path; (5) entity co-fire budget keeps the same drop-spin_2-first priority.

## Redis cache for `/similarity_search` responses
Files: db/redis.py, api/main.py
Why: Each `/similarity_search` call runs the full pipeline (Postgres signal fetch, Qdrant queries across 8 spaces, multi-anchor blending, quality reranking) plus the card-summary hydration — wholly deterministic on a 24h timescale for the same anchor set. Caching the wire response makes "More like this" repeat clicks free.
Approach:
- New helpers in `db/redis.py` (`_similar_movies_key`, `get_cached_similar_movies`, `cache_similar_movies`) mirror the shape of the existing `movie_details` cache. Key `<env>:similar:movies:<comma-joined-sorted-tmdb-ids>`; value is the msgspec-encoded `list[MovieCard]` bytes; 24h TTL matching `_MOVIE_DETAILS_TTL_SECONDS`. Caller is responsible for normalization (callees never re-sort).
- `/similarity_search` now canonicalizes `body.tmdb_ids` via `sorted(dict.fromkeys(...))` once, then runs the standard warm/cold pattern: cache GET → on hit return bytes verbatim → on miss invoke `run_similar_movies_for_ids(canonical_ids)` → hydrate → encode → cache best-effort. Cache read/write failures log + fall through (graceful degradation per docs/conventions.md) so Redis being unhappy never fails the request.
- User specified the cache key should be off "sorted IMDB IDs", but IMDB IDs are not stored in Postgres (`public.movie_card` only has `imdb_vote_count`, not the id string). Confirmed via AskUserQuestion to use sorted TMDB IDs instead — semantically equivalent (1:1 with movies in this system) and adds zero per-request lookups. If cross-system cache sharing ever needs IMDB keying, revisit by adding an `imdb_id` column.
- Cache key is computed *after* dedup+sort but *before* the in-pipeline validation that raises LookupError for unknown TMDB IDs. So order/dup-insensitive (`[1, 2, 1]` and `[2, 1]` hit the same entry) but error responses are never cached (LookupError raises before the cache write).
Design context: Plan at /Users/michaelkeohane/.claude/plans/for-similar-movie-search-crystalline-bumblebee.md.
Testing notes: end-to-end cold/warm comparison via curl (`/similarity_search` with `{"tmdb_ids": [603, 604, 605]}` twice — warm call should be visibly faster, bodies byte-identical). Verify Redis key shape + TTL via `redis-cli --scan --pattern '*similar:movies:*'` and `TTL`. Verify error-no-poisoning: bad TMDB ID → 422 and `EXISTS <key>` → 0. Order/dup invariance: `[605, 603, 603, 604]` should hit the same cache entry as `[603, 604, 605]`.

## Surgical rebuild script for `lex.inv_actor_postings`
Files: movie_ingestion/final_ingestion/rebuild_actor_postings.py (new)
Why: Operational counterpart to the `self`-credit code change above. After the re-scrape, ~109K movies sit at `status='imdb_scraped'` with fresh `actors` lists in the tracker, but `lex.inv_actor_postings` is still populated from the prior narrower scrape. Running the full Stage 8 ingest would needlessly re-embed 8 vector spaces and re-upsert movie_card / character postings / brand postings / awards — all of which are still correct. The script targets exactly the one stale table.
Approach: DROP + CREATE `lex.inv_actor_postings` with the same DDL as `db/init/01_create_postgres_tables.sql:227-237`, then process movies in batches of 500. Per batch: normalize + hyphen-expand in-memory via the existing private helpers from `ingest_movie.py` (`_normalize_name_list`, `_expand_positioned_names`, `_term_ids_and_positions`) — re-imported, not re-implemented, so byte-identical rows; one `batch_upsert_lexical_dictionary` call across the batch's variants; one `unnest()`-based INSERT for the whole batch (~218 INSERTs total instead of ~109K). After Postgres commits, advance `movie_progress.status` from `imdb_scraped` → `ingested` for the batch (guard clause keeps it from clobbering concurrent advances). Mirrors `rebuild_character_postings.py`'s overall shape but lifts the per-movie DB round-trips to per-batch.
Design context: Plan at /Users/michaelkeohane/.claude/plans/ok-so-looks-like-snug-russell.md. One-shot backfill — not part of the steady-state pipeline.
Testing notes: verify (1) tmdb 664280 has Attenborough at `billing_position=1` (the originating bug), (2) `COUNT(DISTINCT movie_id) ≈ 109,237`, (3) tracker shows `imdb_scraped=0, ingested≈109,237, filtered_out=907,548`. Re-runs require no manual reset because DROP is at the top, but the status filter means a successful run "consumes" its input — re-running on a clean tracker state would find nothing to do.

## Add `GET /title_search` — typeahead title-only lookup
Files: api/main.py, db/postgres.py, search_v2/title_search.py (new), docs/modules/api.md
Why: Frontend's "pick similar-to" picker calls on every debounced keystroke and needs a deterministic title-only lookup that bypasses the NLP pipeline entirely. None of the existing endpoints fit: `/query_search` runs the full LLM pipeline (too slow + too expensive for a keystroke), `/similarity_search` takes a tmdb_id (the user doesn't have one yet — that's what they're picking), and `/attribute_search` is for filter-driven browse, not title text. Targets p50<20ms, p95<50ms.
Approach:
- **New DB helper `fetch_title_search_movie_ids` (db/postgres.py)**: single SQL query that runs the broad substring match via the existing `idx_movie_card_title_normalized_trgm` GIN, classifies each row's tier inside the SELECT (CASE: tier 1 if starts-with or has space-prefixed token match; else tier 2), and orders by `tier ASC, 0.8*pop + 0.2*recep/100 DESC, movie_id DESC`. One round trip, LIMIT pushed to the DB. Reuses the existing `NEUTRAL_RERANKER_SEED_*_WEIGHT` constants so the within-tier sort matches `/attribute_search` exactly.
- **New module `search_v2/title_search.py`**: thin orchestrator that truncates oversized queries (100-char cap, defensive — longest catalog title is ~80 chars), normalizes via shared `normalize_string` (NFC + casefold + diacritic strip — symmetric with ingest-time `title_normalized`), escapes LIKE metacharacters via `escape_like` (`_` is part of `\w` and survives normalization, so escaping is genuinely necessary), and builds three LIKE patterns: `"<q>%"` (starts-with), `"% <q>%"` (token-prefix mid-title), `"%<q>%"` (broad substring — superset). All-punctuation queries normalize to "" → return [] without hitting Postgres.
- **Endpoint in api/main.py**: `GET /title_search?q=...&limit=...`. Trims `q` (422 on whitespace-only post-trim), validates `limit` in [1, 25] (422 out-of-range; default 10). Calls `run_title_search` → `fetch_movie_card_summaries` → msgspec-encoded `Response` with `Cache-Control: public, max-age=300`. Same `MovieCard` wire shape as `/similarity_search` and `/attribute_search` so the frontend's single result-card renderer handles all three uniformly. Uses FastAPI `Query(...)` to mark `q` required (returns FastAPI's standard 422 if absent rather than custom).
- **Tier 3 fuzzy (edit-distance ≤ 2) deliberately omitted for v1.** Spec marked it optional with a 10ms budget; adding it would require either a second pass with `levenshtein()` or a similarity threshold via `pg_trgm`, both of which trade against the p50<20ms target. Easy to layer on later behind a `fuzzy=true` query param if UX warrants.
- **`original_title` not implemented.** Spec said "if the index has them" — `public.movie_card` only stores `title`/`title_normalized`. Adding `original_title` would require a schema migration + a backfill from TMDB; flagged for a future enhancement.
- **No Redis caching at the app layer.** The `Cache-Control: public, max-age=300` header lets browsers and any CDN/reverse proxy absorb the keystroke burst. Adding a server-side Redis layer would duplicate that for negligible gain on a query that's already a single trigram-indexed Postgres hit.
Testing notes: end-to-end smoke — "matrix" → The Matrix variants in popularity order; "dark" → "The Dark Knight" tier-1 above generic "dark"-substring matches; "king kong" → 1933/1976/2005 entries ordered by popularity; "amélie" → Amélie (diacritic folding); "ark kni" → "The Dark Knight" via tier 2; whitespace-only `q` → 422; `limit=0` and `limit=26` → 422; `limit=25` clamped at hard cap; LIKE-metachar inputs ("100%", "real_thing") match literally; oversize `q` (>100 chars) silently truncates. Need to verify EXPLAIN ANALYZE confirms the trgm GIN is selected; if a short-query pathology shows up at scale, the fallback is a tsvector + `to_tsquery('q:*')` index.

Follow-up: add `char_length(title_normalized) ASC` as the second sort key (after tier, before the 80/20 popularity blend) so the shorter title wins within a tier. User asked for "less extra stuff outside of the text provided" — for query "john wi", "John Wick" (9 chars) now beats "John Wick: Chapter 2" (19 chars) regardless of popularity. Same-length titles still resolve via the popularity blend (preserving the "King Kong 1933/1976/2005 in popularity order" behavior). Length measured against the normalized column so it stays symmetric with what we matched on.

## Close-title-match tier in exact-title search
Files: db/postgres.py, search_v2/exact_title_search.py
Why: The exact-title flow previously dropped titles like "The Phantom of the Opera 25th Anniversary Edition" entirely when the user searched for "Phantom of the Opera" — they're the same movie, but the exact-LIKE pass requires byte-identical normalized titles. Added a new score band between the 1.0 exact tier and the 0.75 lineage→lineage tier to catch these "query is embedded in title" cases, penalized by how much extra text the candidate carries beyond the query.
Approach:
- **New `fetch_close_title_candidates` (db/postgres.py)**: uses pg_trgm's `<%` operator (`%s <% title_normalized` — query embedded in title) against `idx_movie_card_title_normalized_trgm`. Returns `(movie_id, title_normalized)` so the caller can do the strict gate in Python without a second round trip. Operand order matters — `query %> doc` is the commutator and means the opposite direction; I hit that bug during smoke testing and the helper now documents it inline. Literal `%` is escaped as `%%` per psycopg's parameter-parsing contract.
- **New step 6.5 in `run_exact_title_search` (search_v2/exact_title_search.py)**: pg_trgm is just a coarse recall gate; the actual semantic gate is `query_tokens.issubset(doc_tokens)`, where tokens come from a new `_title_token_set` helper that splits on whitespace *and* hyphens (so "spider-man" and "spider man" tokenize identically — necessary because `normalize_string` preserves hyphens). Strict superset only, no fuzzy per-token matching for v1.
- **Score formula**: `base = 0.80 + 0.15 * (len(query_tokens) / len(doc_tokens))` → range [0.80, 0.95]. Year handling mirrors the existing seed/title-only split — when the user supplied a year that doesn't match, multiply by 0.50 (same ratio as exact 1.0 → 0.5 title_only) to land in [0.40, 0.475]. Label is `close_title_year_match` or `close_title_year_mismatch`.
- **Close matches do NOT seed franchise fan-out** — same rule as `title_only`. Only true 1.0 seeds anchor fan-out (the "Indy boulder" rule: never let a non-exact signal upgrade to franchise inference). Close-match candidates can still be *lifted* by fan-out from real seeds via the existing `_apply` max-with-bookkeeping if their lineage/universe overlaps with a seed's.
- **Exclude already-matched IDs from the close-match scan** via the new `exclude_movie_ids` param on the helper, so the exact-match pass's seed/title_only assignments are never overwritten by a lower close-match score.
- Result-shape contract (`ExactTitleSearchResult.ranked` + `score_source`) is unchanged — callers (`streaming_orchestrator.py`, `run_step_0.py`) see new source labels but no new fields.
Design context: Plan at /Users/michaelkeohane/.claude/plans/alright-update-the-exact-crispy-bird.md. Year-handling choice (symmetric with exact) was confirmed via AskUserQuestion.
Testing notes: live smoke against the production DB — "the matrix" returns 1.000 exact + close-band 0.86–0.90 for sequels and "A Glitch in the Matrix"; "phantom of the opera" returns 1.000 exact + 0.95 for "The Phantom of the Opera" variants + 0.875 for "...at the Royal Albert Hall"; year-supplied case ("phantom of the opera" + 2004) splits cleanly into year-match 0.95, exact-year-mismatch 0.5, close-year-mismatch 0.475/0.4375. Unit tests on `exact_title_search.py` and any callers will need updates in the testing phase to cover the new tier; per project rule no test files were touched.

## Realign Step 0 prompts + schema teaching with downstream flow capabilities
Files: search_v2/step_0.py, schemas/step_0_flow_routing.py
Why: A live-traffic query ("phantom of the opera") was being routed to none_of_the_above because the Step 0 RESOLUTION PRINCIPLE forbade `specific_title` whenever multiple distinct films shared a canonical title — but `exact_title_search` actively handles that case (returns every same-title match at 1.0, then fans into lineage/universe). Audit surfaced ~10 places across the prompt + schema teaching where the LLM guidance contradicted what the downstream flows actually do, all funneling into either over-fallback to `none_of_the_above` or wrong rationalizations the model couldn't apply to edge cases.
Approach:
- **specific_title reframed as "title spec, not single film."** Both `_TASK_AND_OUTCOME` bullets (normal + clarification mode), `_RESOLUTION_PRINCIPLE`, `_OUTPUT_FIELD_GUIDANCE` for selected_entities, and the `_MOST_LIKELY_KIND_DESCRIPTION` for SPECIFIC_TITLE in the schema now teach: resolve to the most common canonical form of the typed title; the downstream search fans out to same-title films, remakes, and lineage automatically; a stated year sharpens intent. Dropped the "multiple films sharing the canonical string is a disqualifier" clause from the schema teaching.
- **RESOLUTION PRINCIPLE rescoped.** From "must be derivable from typed phrasing alone" (which chilled `non_character_franchise` on abbreviations like "MCU" that the downstream LLM expansion handles) to "the typed phrasing must name a real entity of the chosen kind." The forbidden inference is now scoped to guessing *which film* the user means when no real title was named ("the godfather one", "that pixar movie about feelings"), not to whether the surface form itself needs normalization.
- **Cast/director resolution-marker carveout removed.** Both the prompt's `_QUALIFIER_RULE` and the schema's `_QUALIFIERS_DESCRIPTION` previously claimed cast/director references used for installment disambiguation were "part of title resolution" and not qualifiers. But `ExactTitleFlowData` has no schema slot for cast/director, so those tokens would have been silently dropped. Now cast/director are always qualifiers → standard flow. Year remains a non-qualifier (captured structurally by `release_year_if_stated`).
- **Role-marker rule clarified.** Old wording said role markers ("directed by", "starring") disqualified the person flow because they "constrain the search beyond the bare-name lookup" — factually wrong (person_search unions all five role tables; role markers don't change the result set, only the ranking preference, which the flow can't honor). New wording: role-marker *tokens* force standard; the *identity* of the named person doesn't — "Christopher Nolan" still fires person flow even though he's primarily a director.
- **Similarity frame consistency.** Removed the "explicit similarity frame" wording from both task-and-outcome openings so it aligns with `_SIMILARITY_FRAME_RULE`, which has always accepted bare lists of two or more titles as an implicit frame.
- **Coverage principle consolidated.** Old "no exception" wording on the qualifier rule was directly followed by carveouts — restructured into a single paragraph with the year exclusion and similarity-frame exclusion grouped together; cast/director explicitly called out as always qualifiers.
Testing notes: "phantom of the opera" should now route to `specific_title` with canonical_name="The Phantom of the Opera" and surface all variants via exact_title_search's lineage fan-out; "phantom of the opera 2004" should route to `specific_title` with release_year=2004 and sharpen toward the Schumacher film; "the godfather one" still falls back to none_of_the_above; "MCU" routes to `non_character_franchise` (downstream alias expansion resolves it); "Christopher Nolan" still fires `person`, but "movies directed by Christopher Nolan" routes to none_of_the_above. No unit tests modified per test-boundaries.md — prompt prose changes; existing fixtures that asserted specific routing outcomes for multi-installment same-title queries will need updates in the testing phase.

## Soften Step 0 COVERAGE PRINCIPLE to defer to QUALIFIER RULE on packaging
Files: search_v2/step_0.py, search_v2/step_0_few_shot_experiment.py (new)
Why: The previous coverage principle required the entity span to equal the entire query verbatim, which conflicted with the QUALIFIER RULE's instruction to drop packaging tokens (bare result-type words like "movies"/"films", politeness, speech-act framing). The model resolved the conflict inconsistently — e.g. "tarantino's films" routed to none_of_the_above because "films" survived as leftover content even though the model itself recognized it as a result-type word.
Approach: Rewrote `_COVERAGE_PRINCIPLE` to require entity coverage AFTER packaging is dropped per the QUALIFIER RULE, and added matching "that survives the packaging-drop step" wording to the studio/person list-shape paragraph. Both `SYSTEM_PROMPT` and `CLARIFICATION_SYSTEM_PROMPT` pick up the change because both compose from the same module-level section. Validated via a 3-variant experiment (`step_0_few_shot_experiment.py`) on 16 fresh queries — baseline scored 13/16, softened coverage scored 15/16 (fixed "tarantino's films" and "shrek movies"), and adding per-route few-shot examples on top of softened coverage did not improve the score and caused a small regression on "scream", so few-shot was not adopted.
Testing notes: experiment harness is the regression surface for future prompt tweaks — rerun with `python -m search_v2.step_0_few_shot_experiment`. Remaining failure case ("scream" ambiguity) is a separate ambiguity-detection issue not addressed by this change.

## Honor UI hard filters in similarity flows + /similarity_search endpoint
Files: db/postgres.py, search_v2/similar_movies.py, search_v2/streaming_orchestrator.py, api/main.py, docs/modules/api.md
Why: The `similarity_to_titles` Step-0 flow and the standalone `/similarity_search` endpoint deliberately ignored the UI's `MetadataFilters`, justified as "anchor-based similarity is not filter-relevant." That fails real UX — a user with "streaming on Netflix" or "post-2010" filters set who then issues a similarity query had those constraints silently dropped, while every other Step-0 entity flow (exact_title, character/non-character franchise, studio, person) honored them.
Approach:
- **Postgres lanes**: added `metadata_filters: Optional[MetadataFilters] = None` to four similarity-only candidate helpers in `db/postgres.py` — `fetch_director_movie_terms` (uses `_build_inline_movie_card_filter_clause` since the FROM is `lex.inv_director_postings`); `fetch_similarity_source_candidates`, `fetch_similarity_franchise_candidates`, and `fetch_similarity_quality_candidates` all use `_build_direct_movie_card_filter_clause` since the FROM is already `public.movie_card`. The cult_garbage / prestige bucket queries each splice the same filter fragment. Existing helpers that already accept the param (`fetch_movie_ids_by_overall_keywords`, `fetch_movie_ids_by_themes_recall`, `fetch_movie_ids_by_production_company_ids`) are now called with it from the similarity path.
- **Qdrant shape search**: `_query_spaces_batch` in `similar_movies.py` now takes an optional `qdrant_filter: Filter | None` kwarg and threads it into every per-space `QueryRequest`. Both `_run_single_anchor_shape_search` and `_run_multi_anchor_shape_search` translate the incoming `metadata_filters` via the existing `build_qdrant_filter` helper from `db/vector_search.py` (returns `None` when inactive — preserving the no-filter fast path).
- **Threading**: `metadata_filters` is added as a `None`-defaulted kwarg to every entry point and helper — `run_similarity_search`, `run_similar_movies_for_ids`, `_run_single_anchor_similarity`, `_run_multi_anchor_similarity`, `_low_cohesion_fallback`. The orchestrator passes it down without transformation; the per-lane candidate fetches each pass it to their Postgres helper.
- **Streaming orchestrator**: `_run_similarity_with_hydration` now accepts `metadata_filters` and the dispatch block passes it through; removed the "intentionally NOT passed" comment and updated the `stream_full_pipeline` docstring to reflect the new behavior.
- **API endpoint**: `SimilaritySearchBody` gains `filters: Optional[MetadataFiltersInput]` (same wire shape as `/query_search`'s `filters`), translated by the existing `_to_metadata_filters` helper. The Redis cache (`get_cached_similar_movies` / `cache_similar_movies`) keys only on sorted anchor IDs, which would collide across filter configurations — rather than redesign the cache key, the endpoint skips the cache on both read and write when filters are active. Unfiltered hot path is byte-identical to before.
- **Docs**: `docs/modules/api.md` `/similarity_search` section documents the new optional `filters` body field and the cache-skip behavior.
Design context: Plan at /Users/michaelkeohane/.claude/plans/ok-update-the-similarity-to-titles-sunny-starlight.md. User confirmed: apply filters at every candidate-gen lane (matches pattern across other entity flows, more efficient than post-union filter); mirror /query_search's wire shape exactly for the API.
Testing notes: end-to-end via curl with `{"tmdb_ids":[157336], "filters":{"genres":["Comedy"], "min_release_ts":1577836800}}` — verify every returned card overlaps the Comedy genre and `release_date >= 2020-01-01`. No-filter regression check: identical wire output to pre-change baseline. Streaming flow: a query routed by Step 0 to `similarity_to_titles` with active filters should produce filter-compliant similarity cards. Per project rule no test files were touched; existing `test_similar_movies.ipynb` continues to work because every new param defaults to `None`.

Review-pass fixes (same change set):
- **2× over-fetch on filter-active candidate lanes.** Shape (Qdrant) and quality (Postgres `LIMIT`) both apply their limits before the hard filter, so a tight filter would otherwise leave thin post-filter pools. `_run_single_anchor_shape_search` / `_run_multi_anchor_shape_search` now scale `qdrant_limit` by 2 when `metadata_filters` is active; single/multi-anchor quality lanes scale `quality_limit` by 2 when filters are active. No effect on the unfiltered hot path.
- **Per-filter cache slot.** Added `metadata_filters_fingerprint(filters)` in `db/redis.py` (BLAKE2b, 16-hex-char digest, deterministic over the dataclass fields). `_similar_movies_key`, `get_cached_similar_movies`, and `cache_similar_movies` now take an optional `filter_fingerprint` kwarg; unfiltered callers pass `None` and get the legacy `<env>:similar:movies:<sorted_ids>` shape, filtered callers get `<env>:similar:movies:<sorted_ids>:f:<hex>`. Pre-existing unfiltered cache entries stay valid. Replaces the "skip cache when filters active" workaround from the initial commit.
- **build_qdrant_filter call-site simplification.** Replaced `build_qdrant_filter(metadata_filters) if metadata_filters is not None else None` with `if metadata_filters else None` — `MetadataFilters` instances are always truthy, so the two forms are equivalent and the short form reads cleaner.
- **Stale doc references corrected.** `docs/modules/search_v2.md` Hard Filters section previously said "Similarity branch excluded by design"; now describes the filter threading, the 2× over-fetch discipline, and the build_qdrant_filter wire. `docs/modules/api.md` `/similarity_search` Cache bullet now describes the fingerprint scheme instead of the cache-skip workaround.


## Constrain Step 1 spin field lengths
Files: schemas/step_1.py | Added `max_length=150` to `Spin.query` and `max_length=50` to `Spin.ui_label`, with the limits noted in field descriptions. Because `Spin` is shared, this enforces the caps across both `Step1Response` (spins) and `Step1ClarificationResponse` (main_rewrite + spins).


## Add fuzzy word_similarity fallback tier to /title_search
Files: search_v2/title_search.py, db/postgres.py
Why: The `/title_search` typeahead endpoint matched only exact substrings of `title_normalized`, so a typo ("intersteller") returned nothing and a dropped middle word ("race witch mountain" for "Race to Witch Mountain") missed the title. Fuzzy matching was deferred in v1; this adds it without regressing the latency-sensitive hot path.
Approach:
- **New cascade in `run_title_search`**: run the existing exact tiered scan first; only when it underfills (`len(exact) <= TITLE_SEARCH_EXACT_UNDERFILL_TRIGGER` = 3) AND `len(normalized) >= TITLE_SEARCH_FUZZY_MIN_QUERY_CHARS` = 3 AND `remaining = limit - len(exact) > 0`, run a second fuzzy query and append its results BELOW the exact ones. The common well-filled keystroke returns with zero extra round trips. Three new named constants live alongside the existing limit constants in `title_search.py` (trigger=3, threshold=0.45, min-chars=3).
- **`fetch_title_search_fuzzy_movie_ids` (db/postgres.py)**: DB-side `<%`-gated query, ORDER BY `word_similarity(query, title) DESC` then the same shorter-title / 80-20 popularity-reception tie-breakers as the exact tier, with a conditional `movie_id <> ALL(...)` exclude of the exact ids and a `LIMIT`. Reuses the existing trigram GIN index `idx_movie_card_title_normalized_trgm` (same operator/escaping precedent as `fetch_close_title_candidates`). The RAW normalized query is passed (NOT `escape_like`'d) because `<%` is an operator, not LIKE.
- **`_execute_read_with_word_similarity_threshold` helper (db/postgres.py)**: the 0.45 cutoff (below pg_trgm's default 0.6) is applied via `set_config('pg_trgm.word_similarity_threshold', <val>, true)` inside an explicit `conn.transaction()`. The GIN index only accelerates the `<%` operator (which reads the GUC), so a bare `word_similarity() >= x` predicate would seq-scan. `is_local=true` + the explicit transaction keep the override from leaking onto the pooled (non-autocommit) connection — protecting `fetch_close_title_candidates`'s reliance on the default 0.6.
Design context: Plan at /Users/michaelkeohane/.claude/plans/make-a-plan-to-vectorized-goblet.md. Key facts: `word_similarity` is asymmetric (query as 1st arg → length-insensitive on the title side); a single typo scores ~0.72, a dropped middle word ~0.87, messy multi-typo ~0.45 — hence the lowered threshold. 0.45 is the one knob worth retuning against real query logs.
Testing notes: end-to-end verify the edge-case matrix in the plan (empty/2-char/3-char queries; exact >3 hot path untouched; limit=3 & exact=3 → remaining 0 → no fuzzy; exact=0 → empty exclude → fuzzy only; fuzzy=0 → exact only). `EXPLAIN ANALYZE` the fuzzy SQL to confirm the trigram GIN index is used at threshold 0.45. GUC-no-leak check: a `fetch_close_title_candidates` call after a fuzzy `/title_search` must still see the default 0.6. Per project rule no test files were touched.

## Cap user query/clarification input length at 250 chars
Files: search_v2/query_input_validation.py (new), search_v2/step_0.py, search_v2/step_1.py, search_v2/streaming_orchestrator.py, search_v2/full_pipeline_orchestrator.py, api/main.py
Why: both free-text fields are concatenated into the Gemini prompts sent on every search; an unbounded field lets a caller inflate input tokens (cost/latency) and widens the prompt-injection surface. A hard char cap bounds per-request input cost for free. (Not an anti-abuse classifier — quota abuse and harmful-content filtering are separate concerns handled by auth + rate limiting and provider safety settings.)
Approach: introduced search_v2/query_input_validation.py as the single source of truth — MAX_QUERY_CHARS / MAX_CLARIFICATION_CHARS = 250, a typed QueryInputError(ValueError), and clean_query/clean_clarification helpers (strip → non-empty → length cap; blank clarification normalizes to None). Replaced the duplicated `query.strip()` + non-empty checks in all five public surfaces (Steps 0/1, both orchestrators, the API endpoint) with calls to these helpers, so the rules are defined once and enforced at every boundary that can be reached independently (CLI/notebook/batch, not just the API). REJECT over-length rather than truncate — truncation slices intent mid-clause and yields confusing results. QueryInputError subclasses ValueError so the existing "raises ValueError on empty query" contract still holds for inner layers; the API maps it to HTTP 400.
Design context: follows the existing "each public surface defends its own contract" convention (api/main.py comment) and the user's single-codepath-writes instinct. 250 is above a natural movie query (a generated spin query is capped at 150 in schemas/step_1.py); both fields capped independently so a long clarification can't bypass the query cap.
Testing notes: cover boundary matrix — empty/whitespace → error; exactly 250 chars passes; 251 rejected; blank/None clarification → None; over-length clarification rejected; API returns 400 with the message; inner layers still raise ValueError. Tune the cap from observed p99 query length if traffic warrants.

## Lower query/clarification length cap 250 -> 200
Files: search_v2/query_input_validation.py | MAX_QUERY_CHARS / MAX_CLARIFICATION_CHARS lowered to 200 (200 passes, 201 rejected); comment updated. Helpers/messages reference the constants so they track automatically.

## Add CORS middleware to the API
Files: api/main.py
Why: lock browser access to the known frontend origins so other sites' JS can't call the API in a user's browser; prerequisite for sending the planned device-ID/auth cookie.
Approach: CORSMiddleware with explicit ALLOWED_ORIGINS = ["https://www.cinemind.dev", "http://localhost:3000"], allow_credentials=True, methods GET/POST, headers Authorization/Content-Type. Origins are the bare scheme+host (no trailing slash, no path) because the browser Origin header never carries a path — the bare origin covers every sub-path (/, /similar, ...). Explicit enumeration (not "*") is mandatory with allow_credentials=True.
Design context: CORS is browser-enforced only — hygiene against cross-site browser use, NOT a defense against curl/scripted callers (see prior discussion; real anti-farming gate is a challenge-gated short-lived token, not yet built).
Testing notes: real frontend requests succeed; a cross-origin fetch from another site hits a CORS error; preflight OPTIONS on POST endpoints returns the allow headers. If the apex (non-www) cinemind.dev is ever served, add it explicitly.

## Rework `/attribute_search` people ranking → Step 0 person-flow parity; remove `role`
Files: search_v2/attribute_search.py, search_v2/person_search.py, api/main.py, docs/modules/api.md
Why: Per user direction, the people path should copy the Step 0 person-search ranking in every way — a single person (no metadata filters) must yield byte-identical ordering to `run_person_search`. Supersedes the prior band-weight (3-tier LEAD/SUPPORTING/MINOR + 80/20 prior tie-break + cross-role penalty) model and the earlier review follow-ups, which are now removed. `role` is dropped from the endpoint entirely (Step 0 is role-agnostic).
Approach:
- **Reuse, not reimplement**: promoted Step 0's `_fetch_person_buckets` → public `fetch_person_buckets` in `search_v2/person_search.py` (role-agnostic, 4 buckets LEAD/MAJOR/RELEVANT/MINOR with the minor-zone zp=0.5 split; non-actor credits → LEAD). attribute_search now calls it per person, so the two flows can't drift. Updated the one internal caller + docstring.
- **Single-person parity by construction**: each person's bucket → additive weight `(BUCKET_MINOR+1) - bucket` (LEAD=4 … MINOR=1). For one person the summed weight is constant within a bucket and strictly orders buckets, and the within-tier sort is `(popularity_score NULLS-last, movie_id DESC)` — byte-identical to Step 0's `_sort_bucket` (popularity_score only; reception NOT consulted). Verified equal output against the real `run_person_search` in a smoke harness.
- **Multi-person = SUM** (the sole intentional divergence from Step 0, which uses MIN bucket + overlap_count): weights summed across people via UNION, so more/more-prominent credits rank higher and can cross tiers (e.g. lead+minor=5 outranks a single lead=4). Verified.
- **`role` removed from the wire**: deleted the `role` field from `PersonInput`, the `_ROLE_LITERAL_TO_CATEGORY` map, and `PersonSpec.role`; simplified `_to_person_specs`; dropped the now-unused `PersonCategory` and `Literal` imports from api/main.py. Dedup is now on normalized name only (matching Step 0's canonical_name dedup).
- **Removed from attribute_search.py**: `_resolve_person_bands`, `_rank_by_prominence`, `_neutral_prior`, `_merge_max`, `_ZONE_TO_BAND`, band/wrong-role constants, `_PERSON_CATEGORY_TO_TABLE`, and the zone_label/PEOPLE_POSTING_TABLES/billing-row imports. The no-people path (neutral 80/20 seed) is unchanged.
- Note: `zone_label` (added earlier to actor_zones.py) and its use in entity_query_execution.py remain — still a valid shared helper for the entity scorer; attribute_search no longer needs it but person_search uses zone_cutoffs/zone_relative_position directly.
- Docstrings (endpoint + module) and docs/modules/api.md rewritten to the new model; the earlier double-count caveat is gone — dedup is now on normalized name, so passing the same person twice (even with different casing/spacing) collapses to a single summand, exactly like Step 0.
Testing notes: smoke-verified single-person parity (identical list vs run_person_search), multi-person sum crossing tiers, and all-unresolvable → []. Formal `unit_tests/test_attribute_search.py` still pending (write only when asked). Limit stays 250 vs Step 0's 100 — overlapping prefix identically ordered.

## Maturity-rating alias mapping (TV / legacy / foreign certs → supported ratings)
Files: implementation/classes/enums.py
Why: only 6 raw maturity strings resolved to a real rank; everything else (TV-MA, TV-14, Approved, GP, X, 16+, …) fell through to UNRATED and was stored as NULL maturity_rank, hiding ~22.7K of 109K ingested movies from rank-range filters.
Approach: added a per-member `aliases` set to MaturityRating (3-tuple `(rank, value, aliases)` via extended `__new__`), built a normalized `{alias -> member}` reverse map once at module load (`_build_maturity_alias_map`, with a collision guard that raises only on cross-member conflicts so "18+"/"18" dedup harmlessly), and resolve through the native `_missing_` hook so all `MaturityRating(x)` sites benefit (ingestion helper, the Pydantic field in schemas/metadata_translation.py, search-time construction in db/metadata_scoring.py + search_v2). Aliases are declared raw and normalized through the same `normalize_string` used at ingest/query time — single normalization source, no hand-normalized drift. `from_string_with_default` now guards None and logs a warning for non-empty unresolved values (surfaces future junk like the E10+ games rating) while staying silent for empty/NULL/"Not Rated"/"Unrated". Chose `_missing_` over patching only the helper because it's the idiomatic Enum extension point and makes the enum alias-aware everywhere for free.
Mapping (rank): G←tv-g,tv-y,approved,passed | PG←tv-pg,tv-y7,tv-y7-fv,gp,m,m/pg | PG-13←tv-14,13+,12 | R←tv-ma,16+,18+,18 | NC-17←x | UNRATED←not rated. 5 ambiguous values (E10+,T,Open,TV-13,MA-17 — ~13 movies) intentionally left to fall to UNRATED.
No change needed in movie_ingestion/final_ingestion/ingest_movie.py — it stores only maturity_rank via maturity_rating_and_rank(), so correct ranks now flow automatically. Display/label maps key on canonical members post-resolution (a TV-MA movie now displays "R", consistent with existing canonicalization).
Testing notes: verified all 30 representative inputs resolve to expected ranks incl. empty/None→UNRATED and unknowns→UNRATED+warning. Existing enum/scoring tests assert unchanged values/ranks so should stay green; new alias-coverage unit tests recommended but deferred per test-boundaries rule. User will fully re-ingest (no backfill in scope).

## Maturity alias mapping — review follow-ups (canonicalize display text + drop redundant map entries)
Files: implementation/classes/enums.py, implementation/classes/movie.py, schemas/movie.py
Why: code review of the alias change surfaced two items, both addressed.
- Display inconsistency: BaseMovie.maturity_guidance_text (implementation/classes/movie.py) and Movie.maturity_text_short (schemas/movie.py) keyed on the *raw* rating string (`== "Unrated"`, `_MATURITY_DESCRIPTIONS.get(rating.upper())`). After aliasing, a "TV-MA" movie missed the R description (fell back to "Rated TV-MA" / empty) and "Not Rated" wasn't treated as unrated. Both now resolve through MaturityRating.from_string_with_default first, branch on `rating == MaturityRating.UNRATED`, and look up / display the canonical label (`rating.value.upper()`). So TV-MA → R description, Approved → G, Not Rated/None/unknown → unrated path — consistent with the stored rank. Affects 4.9 / anchor vector text, regenerated on re-ingest.
- Redundant reverse-map entries: canonical values were added to `_MATURITY_ALIAS_TO_MEMBER` but `_missing_` never consults them (Enum resolves canonicals natively). Aliases tuples now hold only the *alternative* forms (canonical removed); the map builder iterates `member.aliases` only. Map shrank from 24 to 18 alias-only keys; `aliases` is now semantically "alternatives".
Testing notes: verified canonical still resolves natively, all aliases + unknowns/empty unchanged, no canonical leaked into the map (18 keys), and both display methods produce canonical output (TV-MA→R desc / "Rated R for ...", Not Rated→unrated path). Unit tests still deferred per test-boundaries rule.

## Backfill script: re-resolve maturity_rank for ingested movies (both stores) + add "nr" alias
Files: movie_ingestion/backfill/backfill_maturity_rank.py, implementation/classes/enums.py
Why: the maturity alias change only fixes future ingests; ~22.7K already-ingested movies still have NULL maturity_rank from the old fall-through. This one-shot backfill converges the stored value without a full re-embed.
Approach: modeled on the efficient parts of backfill_release_format.py (bucket-by-value + one bulk `UPDATE ... WHERE movie_id = ANY(%s::bigint[])` per chunk). maturity_rank is read at query time from BOTH Postgres movie_card (db/postgres.py range conds; search_v2 metadata gate) AND the Qdrant payload (db/vector_search.py range filter), so the script updates both. Resolution mirrors ingestion exactly: tracker imdb_data.maturity_rating preferred, tmdb_data.maturity_rating fallback (single LEFT JOIN query), through MaturityRating.from_string_with_default, UNRATED→None.
Efficiency/correctness decisions:
- Diffs resolved target vs current movie_card.maturity_rank and writes ONLY changed rows (alias change only moves NULL→real rank, so changed set ~22.7K of 109K). Re-running after convergence writes nothing (idempotent).
- Per chunk writes Qdrant FIRST, Postgres SECOND: the Postgres column is the diff source, so writing it last means an interrupted run leaves unfinished movies in the diff and a re-run repairs them. Assumes the two stores were consistent pre-run (true — ingester writes both from one value).
- None/unrated target uses Qdrant delete_payload (key removed) rather than a null value — filter-equivalent to Postgres NULL, no ambiguity over Qdrant null storage.
- Bucketed + chunked (default 2000/chunk) bounds statement/request size. CLI: --dry-run, --batch-size, --collection (default "movies", the ingester's write target; not imported to avoid pulling the LLM client stack).
- Self-heals drift: dry-run found 8 movies whose stored rank disagrees with fresh resolution (→ NULL bucket).
Added "nr" to MaturityRating.UNRATED aliases: the warning log surfaced "NR" (2,921 ingested movies, "Not Rated") coming from the TMDB-fallback column (not in the earlier imdb_data-only audit). Maps to UNRATED (unchanged rank, so diff count stays 22,687) and silences 2,921 warnings on the real run. Remaining unmapped non-empty values are only the 13 agreed-ambiguous ones (E10+, TV-13, T, Open, MA-17).
Testing notes: dry-run verified end-to-end (109,237 tracker ingested vs 109,270 movie_card rows; 22,687 to update — rank1=14,219, rank2=2,384, rank3=3,078, rank4=2,849, rank5=149, NULL=8). Qdrant write paths (set_payload/delete_payload) not exercised in dry-run; run without --dry-run to apply. Unit tests deferred per test-boundaries rule.

## Maturity backfill: add --export / --from-file for tracker-less propagation (e.g. EC2)
Files: movie_ingestion/backfill/backfill_maturity_rank.py
Why: the production EC2 box runs its own Postgres + Qdrant (single t3.large, all services in Docker, *_HOST=localhost) but has no tracker.db (5.1GB, gitignored — ingestion runs locally). The local backfill updated only local stores; EC2's maturity_rank is still stale in BOTH Postgres and Qdrant. Needed a tracker-independent way to push the converged ranks.
Approach: extended the existing backfill rather than adding a second script. `--export PATH` (read-only) dumps the converged local movie_card {movie_id: rank|null} to JSON (~1.5MB for 109K). `--from-file PATH` loads targets from that file instead of the tracker and runs the identical diff-and-apply over Postgres + Qdrant. Refactored the diff+apply body of run() into a shared `converge(targets, ...)` used by both the tracker and file sources (DRY). Made the tracker import lazy (inside _load_target_ranks) so --from-file never references tracker.db — safe on a box without it.
Propagation flow: local `--export ranks.json` → scp to EC2 → EC2 `git pull` (brings this code) → EC2 `--from-file ranks.json` updates EC2's own stores. converge() diffs against EC2's current movie_card, so it self-heals whatever EC2's current state is and is idempotent.
Testing notes: round-trip verified locally — export wrote 109,270 entries (rank1=16172, rank2=7883, rank3=9210, rank4=20598, rank5=282, NULL=55125); --from-file --dry-run against the already-converged local DB correctly reported 0 changes; default tracker path regression-checked (still 0, refactor intact). On EC2 the dry-run should report ~22,687 to update. Qdrant write paths not exercised in dry-run.

## Source genres & audio languages for /movie_details from Postgres, not TMDB
Files: api/main.py, implementation/classes/enums.py, implementation/classes/languages.py
Why: the detail endpoint rendered genres/spoken_languages from the live TMDB payload; user wants our own IMDB-derived data (already stored in movie_card as genre_ids/audio_language_ids INT[]) to be the source of truth instead.
Approach: _build_movie_details now maps card_row["genre_ids"]/["audio_language_ids"] back to display names via new Genre.from_id / Language.from_id reverse lookups (backed by module-level _GENRE_BY_ID / _LANGUAGE_BY_ID dicts), dropping unrecognized IDs. This fully replaces the TMDB values (no merge). card_row already carried both columns (fetch_movie_cards selects them) — no new query/schema change. Cached /movie_details payloads keep old values until 24h TTL expiry (acceptable per user).
Testing notes: from_id round-trip verified manually for genre/language hits and unknown-ID None case. Endpoint-level coverage of the new genre/language source not yet added.

## Surface curated sub-genres on /movie_details
Files: implementation/classes/overall_keywords.py, schemas/api_responses.py, api/main.py
Why: user wants sub-genres (e.g. "Splatter Horror" for Terrifier) shown on the detail view. These live in the OverallKeyword taxonomy (movie_card.keyword_ids), not the 27 top-level Genre enum, and weren't surfaced anywhere in the response.
Approach: added an is_subgenre bool as a 4th field on every OverallKeyword member (curated allow-list of 141/226 = true sub-genres; False for top-level genres, national/language cinemas, TV/format types, anime demographics, specific sports, animation techniques). Mechanical 226-line edit done via a one-off scripted rewrite with a name-resolution guard. Added OverallKeyword.from_id + _KEYWORD_BY_ID reverse index and a subgenre_names_from_ids() helper. New MovieDetails.subgenres field populated in _build_movie_details from card_row["keyword_ids"] (already selected by fetch_movie_cards). Distinct from the genres field (which now comes from genre_ids per the prior entry).
Design context: chose a curated flag over "all keywords minus genres" or family-whitelist for precision (user picked this explicitly). Borderline exclusions (anime demographics, specific sports, animation techniques, formats, Holiday/Anime/Film-Noir) flagged for user review.
Testing notes: helper verified manually (subgenre filtering, unknown-id drop, top-genre drop). No endpoint test added. The 141/85 split is the main thing to review.

## Reverse decision: show ALL keywords on /movie_details (drop is_subgenre)
Files: implementation/classes/overall_keywords.py, schemas/api_responses.py, api/main.py
Why: user decided against curating sub-genres — just surface every keyword tag.
Approach: removed the is_subgenre field added in the prior entry (reverted all 226 tuples + __new__ + annotation via scripted strip). Kept OverallKeyword.from_id + _KEYWORD_BY_ID. Renamed helper subgenre_names_from_ids -> keyword_names_from_ids (now returns every known keyword's display name, dropping only unknown IDs). Renamed MovieDetails.subgenres -> keywords; endpoint populates it from card_row["keyword_ids"]. Note: this list overlaps with `genres` (e.g. "Horror" appears in both) and includes language/format/demographic tags — accepted by user.
Testing notes: helper verified (all known ids mapped, unknown dropped); api.main imports clean. No endpoint test added.

## Exclude genre-duplicate keywords from /movie_details keywords
Files: implementation/classes/overall_keywords.py, api/main.py
Why: the keywords list repeated genres already shown (e.g. "Horror" in both).
Approach: keyword_names_from_ids now takes exclude_names; _build_movie_details computes genre_names once (extracted to a local) and passes it so any keyword duplicating a returned genre is dropped. Matching uses a hyphen-insensitive normalized key (_dedup_key = normalize_string + hyphen->space) because the Genre enum hyphenates labels the keyword taxonomy spaces (genre "Sci-Fi"/"Reality-TV" vs keyword "Sci-Fi"/"Reality TV").
Testing notes: verified Horror/Sci-Fi/Reality-TV genre dups dropped while sub-genre keywords (Slasher/Splatter Horror) kept. api.main imports clean.

## Add keyword hard-filter to /attribute_search
Files: api/main.py, implementation/classes/schemas.py, db/postgres.py
Why: let callers filter attribute search by OverallKeyword tags (sub-genres/styles) the same way genres already work.
Approach: mirrored the genre path exactly across 4 spots — (1) MetadataFiltersInput.keywords: list[str] wire field; (2) _to_metadata_filters translates via OverallKeyword(name) (exact-value match, 422 on unknown, same contract as Genre(name)); (3) MetadataFilters.keywords: list[OverallKeyword] dataclass field (is_active auto-covers it via fields()); (4) _build_movie_card_conditions emits `keyword_ids && %s::int[]`. Backed by existing GIN index idx_movie_card_keyword_ids. Semantics: OR within the keyword list, AND across filters; hard filter (excludes non-matches). All three filter-clause builders share _build_movie_card_conditions so they inherit it. No LLM/scoring changes — attribute_search is filter-only.
Testing notes: verified end-to-end manually (translation valid/invalid->422, SQL clause+params, combined genre+keyword AND). No unit tests added.

## Full vector-channel support for keyword filter (+ delete stray api/main 2.py)
Files: db/vector_search.py, movie_ingestion/final_ingestion/ingest_movie.py, movie_ingestion/backfill/backfill_keyword_ids_to_qdrant.py (new); deleted api/main 2.py
Why: code review found MetadataFiltersInput.keywords is shared by /query_search and /similarity_search, but build_qdrant_filter ignored keywords and the Qdrant payload lacked keyword_ids — so a keyword filter was applied in Postgres channels but silently dropped in the vector channel (inconsistent hard filter). User chose full support over restricting the field.
Approach: (1) build_qdrant_filter now emits a keyword_ids MatchAny condition, mirroring genres exactly. (2) _build_qdrant_payload writes payload["keyword_ids"] = movie.keyword_ids() so new points carry it. (3) New one-shot backfill backfill_keyword_ids_to_qdrant.py reads the authoritative movie_card.keyword_ids and set_payloads it onto existing points via batch_update_points (per-point SetPayloadOperation — distinct list per movie; wait=False; idempotent; Qdrant-only since Postgres already holds the value). No Qdrant payload index needed — genre_ids/audio_language_ids have none either (verified: no create_payload_index in repo). Also deleted api/main 2.py — an accidental macOS Finder duplicate committed in 0177860, 312 lines, stale (no /attribute_search etc.), referenced nowhere.
OBLIGATION: keyword filtering on the VECTOR channel (query_search/similarity_search) is INERT until the backfill runs against the live Qdrant — until then a keyword filter there excludes all already-ingested movies. The Postgres-only /attribute_search path works immediately (no Qdrant dependency).
Testing notes: verified build_qdrant_filter emits the condition (empty->None), backfill module imports + API shape (SetPayloadOperation/SetPayload) validated, _execute_read signature matches. No unit tests added; test_vector_search.py would want a keyword case (build_qdrant_filter) mirroring the genre tests.

## New /rerun_query_search endpoint — replay a search with new filters, bypassing Steps 0/1
Files: search_v2/streaming_orchestrator.py, api/main.py, docs/modules/api.md, docs/modules/search_v2.md
Why: re-running a query only to apply different hard filters re-paid for Steps 0 (flow routing) and 1 (spin generation) — two Gemini calls — yet filters never enter query understanding (they're applied at retrieval primitives only). The /query_search `fetches_ready` event already carries everything needed to replay each branch.
Approach: reorganize-over-add. Extracted `_stream_from_branch_plan()` from `stream_full_pipeline` (everything after the six `to_*_flow_data()` calls — build fetches → launch tasks → merge loop, ~lines 355-671) as a single insertion edit: that block was verified to reference only branch_plan + the six flow-data objects + metadata_filters + t0 (no step0_response/step1_response/clarification/raw query). `stream_full_pipeline` now delegates to it after Steps 0/1; new public `stream_rerun_pipeline()` stamps its own t0 and delegates directly, skipping query understanding. In api/main.py: a discriminated-union request body (`RerunSearchBody.branches`, `type` tag → 7 branch models) carrying only each flow's strictly-required data (year/label/kind nullable), plus `_to_rerun_plan` (boundary translator mirroring `_to_metadata_filters`) → `(branch_plan, six flow-data)`. Endpoint mirrors `query_search` exactly (same SSE encode loop, headers, `_to_metadata_filters`). Standard branches re-enter at Step 2 (kind is identity/label-only — confirmed not load-bearing in scoring at stage_4_execution.py:557,980 — so only `query` is required; kind assigned positionally original/spin_1/spin_2).
Design context: interpretation locked in approved plan — standard branches RUN Step 2 (branch query is the Step-2 input); not skipped, since skipping would require passing the full QueryAnalysis. At most one entity flow per type (orchestrator has one slot each) → 422 on duplicate; >3 standard → 422 (upstream never emits more).
Testing notes: validated model translation + all error paths (duplicate entity / >3 standard / blank names / kind collision / unknown type / empty branches → correct 400/422) by importing api.main and exercising _to_rerun_plan. Both files parse. NOT yet run end-to-end against live services — verification section of the plan covers the curl-based SSE check (capture /query_search fetches_ready, echo a subset to /rerun_query_search with new filters, confirm event sequence + filter-respecting results; entity-only rerun should match original when filters unchanged). No unit tests added (separate phase) — would want coverage of _to_rerun_plan demux + the stream_full_pipeline → _stream_from_branch_plan delegation equivalence.

## Review fixes for /rerun_query_search
Files: search_v2/streaming_orchestrator.py, api/main.py, docs/modules/api.md, docs/modules/search_v2.md
Why: self-review of the rerun endpoint surfaced one convention violation + three minor gaps.
Approach: (1) `_to_rerun_plan` returned a 7-element positional tuple (violates conventions.md "return a dataclass for >2 values" — positional unpack of 7 names silently misbinds on reorder). Added frozen `RerunPlan` dataclass in streaming_orchestrator.py; `stream_rerun_pipeline` now accepts `RerunPlan` (satisfies "accept domain objects, destructure internally") and `_to_rerun_plan` returns it. `_stream_from_branch_plan` keeps its 6-kwarg signature (shared by stream_full_pipeline's separate locals). (2) Entity-name fields were unbounded client free-text reaching LLM prompts (studio / character-franchise translators) + Postgres; added `_enforce_name_cap` (MAX_QUERY_CHARS) wired into `_clean_one`/`_clean_names`/similarity-refs → 422 over cap, for parity with /query_search's query bound. (3) Standard branch query used `clean_query` (200 cap), which could falsely 400 the failed-clarification slot-1 merge `f"{raw_query}. {clarification}"` (up to ~402 chars) — a branch the pipeline itself emits. Added `_clean_branch_query` capped at `MAX_QUERY_CHARS + MAX_CLARIFICATION_CHARS + 2` so every emittable branch round-trips while still bounding Gemini input. (4) Removed the optional `kind` override field from `StandardRerunBranch` (type honesty — BranchKind is Literal; overrides could deviate) — kind is now always positional, which also deleted the used_kinds collision check; replaying branches in order reproduces the original fetch ids.
Testing notes: re-validated by importing api.main — RerunPlan return shape, positional kinds, blank/over-length entity names → 422, branch query at 402 accepted / 403 → 400, duplicate-entity / >3-standard / unknown-type / empty-branches still correct. Both files parse. Still not run end-to-end against live services (same plan verification section applies). `kind`-collision error path is gone (field removed).

## Step 2 plot-shape guidance — keep a described plot in one trait
Files: search_v2/step_2.py
Why: Step 2 stochastically split story/plot-shape descriptions into peer atoms
when a query named two independently-searchable character archetypes bound by a
plot verb (e.g. "shy nerd wins over the popular cheerleader" broke apart 4/5 runs,
sometimes orphaning the bare verb "wins over" as its own trait). Searching the
participants independently and intersecting loses the relation the user described.
Approach: added generalized, example-free guidance framed as a recognition, not a
reconstruction test — "if the content describes the shape of a plot, keep the whole
description in one atom/trait." Five edits to step_2.py: (1) new PLOT SHAPES
subsection in _ATOMICITY (primary); (2) cross-link on the peer-atoms OUTCOME;
(3) a COMMON PITFALL; (4) a clause on split_exploration's FORWARD check;
(5) an ACT ON PLOT-SHAPE SPLITS backstop in _COMMIT_PHASE worded to not collide
with the existing "single-direction shaping is NOT a fuse trigger" rule. The guard
is the describe-the-story-vs-name-a-property distinction, which prevents over-firing
onto parallel filters / qualifier-on-population / positioning references.
Design context: builds on the atomicity population-test machinery; deliberately
NOT a schema change (procedural reasoning belongs in the prompt per the file's
schema=micro-prompts / prompt=procedural split).
Validation: 4-way experiment (12 queries x 5 reps) via a throwaway harness
(baseline gemini / changes gemini / changes gpt-5.4-mini none / changes gpt-5.4-mini low).
Changes-gemini fixed both breaking targets ("shy nerd wins over the popular cheerleader",
"underdog boxer beats the reigning champion") to 5/5 whole with NO control regressions
(several controls got more stable); the fix also held across gpt-5.4-mini. Note:
gpt-5.4-mini rejects reasoning_effort="minimal" (valid: none/low/medium/high/xhigh), and
its qualifier-on-population handling collapses to 1 trait vs gemini's 2 (the qualifier is
retained as a modifying_signal, not dropped) — a model difference, not caused by this change.
SHIPPED VARIANT: Gemini (gemini-3.5-flash, thinking minimal, temp 0.35) + these prompt fixes.
The experiment harness/results were temporary scaffolding and have been deleted.
Testing notes: no unit tests touched.

## Reframe SENSITIVE_CONTENT to presence-only; allow MATURITY_RATING proxy use
Files: schemas/trait_category.py
Why: Step 3 was over-decomposing positive audience traits (e.g. "family friendly")
into a SENSITIVE_CONTENT call framed as AVOIDANCE ("ensuring the absence of graphic
violence"), which (a) violates the presence-only contract and (b) represents "safe
content" poorly. Diagnosis traced the misroute to the SENSITIVE_CONTENT description,
which advertised absence framing ("no gore", "mild language only", "safe content"),
making it read as a clean fit for positive safety asks.
Approach: (1) SENSITIVE_CONTENT description reframed to name only the PRESENCE/intensity
of mature content; boundary now explicitly redirects "safe for kids"/absence asks to
TARGET_AUDIENCE and states the category never indexes absence; an edge_case preserves
Step-2 recognition that avoidance-framed content asks ("not too bloody") still route here
(the polarity is tracked separately). (2) MATURITY_RATING boundary loosened from "fires
only on an explicit rating" to allow sparing proxy use (e.g. "safe for kids" -> G/PG
ceiling) when it is the best available inclusion signal, never as a default.
Design context: part of a larger Step 3 trait-decomposition redesign (consolidation to
minimum viable call set + inclusion-only routing) still in planning; these two category
edits were authorized to land independently. Descriptions feed both the Step 2 and Step 3
taxonomy prompts via _build_full_category_taxonomy_section.
Testing notes: no unit tests touched; verify no test hardcodes the old description strings.

## Step 3 consolidation redesign — minimum-viable-call set + inclusion-only routing
Files: schemas/enums.py, schemas/step_3.py, search_v2/step_3.py,
search_v2/category_candidates_experiment/ (queries.py, run_step_3_batch.py,
summarize.py, CONSOLIDATION_EXPERIMENT.md)

### Intent
Step 3 over-fragmented single-concept traits into FACETS (PRODUCT) of
multiple category calls, and routed positive traits to avoidance-framed
categories. Move the minimum-viable-set discipline upstream so the trait
(already one concept after Step 2) is represented by the fewest calls,
with inclusion-only framing decided before a category is committed.

### Key Decisions
- aspects now enumerate distinct, non-overlapping, comprehensive PARTS
  (not the whole restated alongside its parts); reassembly is the
  consolidation step's job. No relevance enum on aspects (user decision:
  over-enumeration is fine if consolidation can reduce it).
- New `CandidateFit` enum (CLEAN_OWNERSHIP / COULD_CONSOLIDATE /
  LIKELY_DISREGARD) on each CategoryCandidate. Inclusion-eligibility folds
  into fit: a category usable here only by describing an absence caps at
  LIKELY_DISREGARD (decided at fit time, not at the call layer — user's
  #5). Candidate floor (min 5) kept; the label separates real fits from
  floor-filler without lowering it.
- `routing_exploration` renamed to `consolidation_analysis` and rewritten:
  EXPLORE options first, THEN place the trait on the breadth↔single-shape
  spectrum (continuous, not binary); minimum calls, fold finer parts into
  a broader call rather than spawning brittle separate facets. The old
  SOLO-coverage-short-circuit / FRAMINGS-vs-FACETS framing was replaced.
- combine_mode now reads off the spectrum placement (point 6); FACETS vs
  FRAMINGS logic otherwise unchanged.
- Inclusion-only stated as generalized guidance (no merge/flip/orchestrator
  mechanics — user's "no broader-system details"); merge-mechanics prose
  stripped from _MINIMUM_SET_AND_POLARITY and the dimension absence pitfall.
- SENSITIVE_CONTENT/MATURITY_RATING category edits (prior DIFF entry) are
  the companion change.
- All edits are generalized principles — no few-shot, no test-query
  leakage (removed a "hidden gem"/"feel-good"/"underrated" example block
  from the aspect section that also leaked a test query).

### Validation
3-way experiment (12 queries x 5 reps): base (gemini, category edits only,
old prompt) vs fix_gemini vs fix_gpt (gpt-5.4-mini low/low). See
search_v2/category_candidates_experiment/CONSOLIDATION_EXPERIMENT.md for
the full findings. Summary: plot-shape/single-concept fragmentation
reduced (SOLO rates up), family-friendly avoidance category dropped
(gemini), regression guards stable EXCEPT gemini regressed `thrillers`
(genre→genre+tone FACETS). Audio-language trap on nationality queries NOT
fixed by the step-3 changes — needs a category-definition lever. gpt-5.4-mini
at low/low consolidated more decisively than gemini in this run.

### Testing Notes
Step 3 unit tests (if any) that assert on TraitDecomposition shape will
need updating: field `routing_exploration`→`consolidation_analysis`, and
CategoryCandidate now requires a `fit` field. No tests were run/modified
per test-boundaries rule. Model left as Gemini (production default);
gpt-5.4-mini is an experiment candidate pending user decision.

## Category-definition audit: overlap fixes + Step-3 fit definition-adherence
Files: schemas/trait_category.py, schemas/step_3.py, search_v2/step_3.py

### Intent
Resolve category-overlap and definition-inconsistency findings the user
prioritized from the taxonomy audit (clusters A1–A4 + B1/B2/B3/B6/B9), and
wire the category definitions into Step 3's candidate `fit` ranking so a
candidate that violates its category's definition (or where another category
fits better) is ranked lower. Root cause established earlier: Step 3's
over-fragmentation/mis-routing traces to the category DEFINITIONS, not Step 3
reasoning.

### Key Decisions
- **Semantic core (A1+B2+B3):** rewrote STORY_THEMATIC_ARCHETYPE / PLOT_EVENTS
  / CHARACTER_ARCHETYPE / ELEMENT_PRESENCE / CENTRAL_TOPIC with dense,
  principle-based boundaries + generalized "decision tests" (no examples in
  the prose). Encodes the user's framing rule: STORY = whole-film
  elevator-pitch shape (even "X does Y"); PLOT = a beat WITHIN a larger story
  (framing decides, not content); CHARACTER = static type, no arc; GENRE = a
  (qualified) label; ELEMENT = explicit fallback for a present thing with no
  better home. B3: deleted CHARACTER_ARCHETYPE's "Lone female protagonist
  splits…" edge_case (it taught peeling a person-phrase into a facet).
- **Suitability (A2+B4):** MATURITY_RATING fires only on an explicit rating
  (removed the "MAY proxy for suitability" clause); SENSITIVE_CONTENT only on
  explicit mature-content; TARGET_AUDIENCE owns an audience trait ALONE (its
  handler already applies rating suitability — a separate rating/content call
  is redundant).
- **Reception (A3):** crisp generalized split across NUMERIC_RECEPTION_SCORE
  (a number) / GENERAL_APPEAL (goodness-degree, no number) / CULTURAL_STATUS
  (canonical position / reception shape) / SPECIFIC_PRAISE_CRITICISM (named
  aspect) / AWARDS (formal ceremony outcome).
- **Geography/language (A4):** aligned COUNTRY_OF_ORIGIN / CULTURAL_TRADITION
  / FILMING_LOCATION / NARRATIVE_SETTING with the already-applied
  AUDIO_LANGUAGE explicit-only gate, each with a "how to tell" test.
- **B1:** TARGET_AUDIENCE = explicitly named audience (people); VIEWING_OCCASION
  = occasion not defined by named people; a people-word inside an occasion
  phrase does not promote to audience. NO rename (kept VIEWING_OCCASION — user
  decision; WATCH_CONTEXT collides with the watch_context vector space).
- **B6:** added route-elsewhere decision tests to EMOTIONAL_EXPERIENTIAL (a
  genre/occasion/device/arc that merely evokes a feeling routes to its own
  home). GENRE untouched (A6 out of scope). **B9:** docstring 45→43.
- **Step-3 fit:** appended a definition-adherence criterion to
  `CategoryCandidate.fit` (schemas/step_3.py) and the "COMMIT A FIT LABEL"
  guidance (search_v2/step_3.py) — rank against the candidate's own taxonomy
  entry; reserve CLEAN_OWNERSHIP for the category whose definition squarely
  claims the aspect. Placed after the covers/misses prose (explore-before-commit).
- **Authoring constraint honored:** all examples curated diverse and screened
  against the 12 `QUERIES` + 25 `RESCORE_EVAL_QUERIES` (word-boundary scan of
  edited categories: zero leakage).

### Validation
`audit_gpt` 5-rep run (gpt-5.4-mini low/low, Step 2 fixed). Vs `fix_gpt`:
family-friendly redundancy fully fixed (5/5 solo Target audience); japanese
audio-language trap eliminated; washed-up-boxer consolidation improved (4/5
solo Story). Mixed: loser-guy traits now always anchor on Story (mis-anchor
gone) but still peel a Character facet ~3/5. Not fixed: serbian audio-language
(still 5/5 — Step 3 keeps Audio for a bare nationality lacking a strong
tradition alternative; AUDIO_LANGUAGE's description/good_examples still imply
nationality→language and are the Step-2-visible lever). Regression guards all
held. See search_v2/category_candidates_experiment/CONSOLIDATION_EXPERIMENT.md.

### Testing Notes
Step 3 / category unit tests that assert on category-definition text or the
`fit` field semantics may need updating; none run/modified per test-boundaries.
Model restored to Gemini (production default). Pre-existing test-adjacent terms
remain in NON-edited categories (NAMED_CHARACTER "anti-hero", GENRE "underdog
stories") — left per the edited-categories-only scope.

## Audio-language trap: Step-2 source fix (de-nationalize AUDIO, COUNTRY template, intent_exploration calibration)
Files: schemas/trait_category.py, schemas/step_2.py, search_v2/step_2.py

### Intent
Eliminate the serbian audio-language trap (bare nationality query spuriously
committing AUDIO_LANGUAGE). The prior round's boundary gate was necessary but
could not bite, because the language read was injected upstream by Step 2.

### Key Decisions
- ROOT CAUSE (corrects prior DIFF entry / Round-2 finding): the Audio commit did
  NOT originate at Step 3. Step 2's intent_exploration bundled "or in the Serbian
  language" into the trait's evaluative_intent as a fake-single read ("no competing
  interpretations"); Step 3 faithfully decomposed it into country + audio aspects.
  Controlled proof: japanese Step-2 intent was origin-only → 0 Audio; serbian
  carried the OR-language clause → Audio 5/5. Decided upstream, not by category defs.
- Three reinforcing levers: (1) de-nationalized AUDIO_LANGUAGE description +
  good_examples (removed the "<nationality>-language films" template that was
  structurally identical to "serbian movies"; now framing-forward — subtitled /
  dubbed / original-language); (2) added a bare "<nationality> films" template
  ("Mexican films") to COUNTRY_OF_ORIGIN so the dominant origin read has a home as
  clean as the one removed from AUDIO; (3) calibrated Step 2 intent_exploration —
  framing + a QueryAnalysis.intent_exploration NEVER bullet forbidding "WIDEN ONE
  READ WITH A LOW-CONFIDENCE 'OR'" (the exact serbian failure shape). Levers 1–2 are
  the only category fields Step 2 renders (name+description+good_examples).
- Examples screened against all 37 experiment queries (zero leakage); diverse per
  the "principles-over-pattern-matching" guideline.

### Validation
Experiment `s2fix_gpt` (gpt-5.4-mini low/low; Step 2 regenerated, so NOT
Step-3-isolated — old step_2_results.json saved as step_2_results.pre_s2fix.json).
serbian movies 5/5 SOLO [Country of origin] (was 3× facets + 2× framings [Audio,
Country]); family-friendly 'serbian' 5/5 SOLO [Country of origin] (was 5× framings).
Zero committed Audio calls across all 12 queries (residual Audio appears only as a
likely_disregard Step-3 candidate). Regression guards held; the spurious 2-call
Audio+Country compound also disappeared. Caveat: Step-2 resampling reshuffled
atomization on a few queries (anime/for-kids benign; standalone 'clean' routes
inconsistently — flagged for future look). See CONSOLIDATION_EXPERIMENT.md Round 3.

### Testing Notes
Step 2 / QueryAnalysis tests asserting on intent_exploration field text and any
category-vocabulary snapshot tests may need updating; none run/modified per
test-boundaries. Model restored to Gemini in both step_2.py and step_3.py.
