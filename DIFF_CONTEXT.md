# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## V3.4 Bucket-Weaver — multi-source recommendation layer
Files: search_v2/similar_movies.py, search_improvement_planning/similar_movies_test_tracker.md

### Intent
Replace V3's dominance/franchise/competitive-band caps inside
`_weave_candidates` with an explicit slot-allocation + greedy-MMR pass
over 5 named recommendation buckets (best_overall, auteur, franchise,
rare_keyword, lead_actor). Movie similarity is a recommendation
problem (rows by *why*) not a single-list search problem; the unified
ranker can't elevate Lady Bird for Barbie without distorting
mainstream/cult/franchise anchors. The weaver is a thin layer on top
of V3 scoring — preserves all V3 invariants (lanes, multipliers,
floors); only the top-section ordering changes.

### Key Decisions
- **5-bucket set** chosen by enumerating every V3 signal against
  "would a user say yes to a separate row of these?" — themes /
  source / studio / quality / format / medium / country rejected as
  style modifiers, not recommendation sources (similar_movies.md
  §V3.4 Decisions 1–5).
- **Allocation**: 5/10 floor for best-overall + 5 distributed by
  signal strength via Hamilton's largest-remainder (capped at 3 per
  bucket). Slack flows back to best-overall.
- **Greedy weave with MMR** (`λ=0.5`): per-slot picks
  `argmax((1−λ)·relevance + λ·deficit_ratio)`. Best-overall is a
  carve-out always-on (`placed[]` can exceed target via cross-
  membership; bucket-gate skips it explicitly). Without this, multi-
  bucket full credit drives best-overall past target by slot 5 and
  the algorithm short-circuits before filling the 10-slot top
  section.
- **Multi-bucket full credit** on placement (Decision 8): placing a
  candidate bumps `placed[]` for every bucket it qualifies for.
  Prevents the "best-overall surfaces 3 Nolan films, then auteur
  double-downs with 3 more" failure.
- **Format top-5 lock interaction**: layered into the per-bucket peek
  helper; bucket queues skip past format-mismatched candidates while
  slot < 5.
- **Bucket gating constants (working draft)**:
  `WEAVER_FRANCHISE_BUCKET_MIN_SCORE=0.55` (excludes tier-4 universe-
  only — Decision 6); `WEAVER_RARE_KEYWORD_BUCKET_IDF_MIN=0.55`
  (matches V3 high-tier cutoff). Post-run analysis suggests the
  rare-keyword gate should bump to ~0.70 to suppress Pixar-trio
  non-Pixar injections.

### Planning Context
Full design in similar_movies.md §V3.4 (lines 1907–2294). Hypothesis
in similar_movies_test_tracker.md §V3.4.

### Testing Notes
21-anchor + 14-cohort smoke harness re-run: canonical wins confirmed
(Lady Bird + Little Women in Barbie top 5; Frances Ha into Female-led
Gerwig top 10; The Mission #2 in Best Picture trio). Regressions in
cohorts where rare_keyword bucket fires on weak/heterogeneous
evidence (Pixar trio non-Pixar injections at #4/6/8; Slasher trio
Mamma Mia/Pennies from Heaven; Barbie documentary metas at slots
7–8). Recommended next iteration: tighten rare_keyword bucket IDF
gate before ship.

## Rescore overhaul: separate pool definition from per-trait scoring
Files: schemas/enums.py, schemas/trait_category.py, search_v2/stage_4_execution.py

### Intent
Implements the conceptual plan in
[search_improvement_planning/rescore_overhaul.md](search_improvement_planning/rescore_overhaul.md).
Fixes the trait-local reranker pool bug: in queries like "dark gritty
marvel movies", the dark/gritty SEMANTIC rerankers were scoping to
keyword-matched candidates only and never seeing the marvel STUDIO/
FRANCHISE candidates from sibling traits. Stage 4 now runs as a
5-phase pipeline (B = pool definition, C = global reranker pass,
D = per-trait scoring, E = aggregation) where every positive
reranker scores the finalized union regardless of which trait its
call lives in.

### Key Decisions
- **Per-category combine declared on `CategoryName` enum.** New
  `CategoryCombineType` enum in `schemas/enums.py` with values
  SINGLE / ADDITIVE / ALTERNATIVES / NO_OP. Populated on all 43
  members per the user's authoritative mapping (27 SINGLE, 11
  ADDITIVE, 4 ALTERNATIVES, 1 NO_OP for BELOW_THE_LINE_CREATOR).
  The `combine_calls` helper folds per-call scores into a per-
  category score; ADDITIVE multiplies, ALTERNATIVES takes max,
  SINGLE passes through, NO_OP returns None so the across-category
  max skips it. NO_OP is defense-in-depth — the EXPLICIT_NO_OP
  handler bucket already emits zero specs.
- **Across-category combine is max universally.** Step 2's
  atomization rule guarantees one trait = one criterion, so
  multiple categories within a trait are different framings of the
  same criterion (max, not noisy-OR or average).
- **Rarity weighting restricted to pure-generator traits.** Mixed
  and pure-reranker traits get rarity_factor = 1.0; only
  pure-generator traits use the corpus-rarity tier. Trait
  classification is derived from `spec.operation_type` across the
  trait's calls — no new LLM-emitted role label.
- **Negative polarity preserved verbatim.** `_score_negative_trait`
  and `_AUTHORITATIVE_NEGATION_CATEGORIES` carry over unchanged;
  the gate × fuzzy three-bin formula stays. Negative traits dispatch
  their calls against the finalized union via a thin wrapper
  (`_dispatch_negative_trait`) and route through the existing
  scoring function.
- **Empty-pool semantics tightened.** Neutral seed only fires when
  zero positive generators were attempted pipeline-wide. If
  generators ran and the union ended up empty (or shorts subtraction
  emptied it), return empty results — "if something truly doesn't
  exist, then it doesn't exist."
- **Clean rewrite of `stage_4_execution.py`**, not a feature flag.
  The recursive granularity helpers (`_run_cand_gen_trait`,
  `_run_cand_gen_category`, `_run_pure_reranker_category`,
  `_finalize_scores`, the per-execution dataclasses) are deleted.
  Public dataclasses (`BranchRankedResults`, `ScoreBreakdown`,
  `TraitContribution`) and the `execute_branches` signature are
  preserved so the orchestrator caller is unaffected.

### Planning Context
Plan file: `/Users/michaelkeohane/.claude/plans/optimized-cuddling-blossom.md`.
Per-category combine assignments and the no-op short-circuit
semantics were locked by the user before implementation began.
The 27/11/4/1 split tally was verified post-population.

### Testing Notes
End-to-end smoke tests via `run_orchestrator.py` confirm the bug fix:
"dark gritty marvel movies" → top results are Werewolf by Night,
Thunderbolts*, Black Panther: Wakanda Forever (the actual dark/gritty
Marvel films) with marvel trait_score = 1.0 × elevated 1.75 × ULTRA_RARE
1.5 = 2.625 contribution; "movies about WWII" → Dunkirk, Land of
Mine, The Longest Day rank top with maxed trait scores. Per-trait
breakdown surfaced via ScoreBreakdown.trait_contributions matches the
new per-trait scoring output.

Per `.claude/rules/test-boundaries.md` no test files were touched
or run. Existing `unit_tests/` covering stage 4 may need updates
in a follow-up testing pass.

## Per-trait score decomposition in top-25 print
Files: search_v2/stage_4_execution.py, run_orchestrator.py
Why: Debugging which trait drove a movie's rank required re-deriving per-trait contributions from raw trait_scores by hand. Surface them first-class alongside positive_total / negative_total (per "Surface computed-value decomposition alongside aggregates").
Approach: New `TraitContribution` dataclass (surface_text, commitment, contribution) and `ScoreBreakdown.trait_contributions: list[TraitContribution]` populated in `_finalize_scores` — cand-gen traits first, then pure-reranker traits, with sign already applied (negative-polarity traits surface as a non-positive contribution). Sum of contributions equals positive_total + negative_total. The top-25 table in run_orchestrator.py picks header order from the first non-empty breakdown and renders one column per trait labeled `"surface" [commitment]`. The implicit-prior post-rerank pass already mutates breakdowns in place, so trait_contributions is preserved.
Testing notes: Dataclass field is `field(default_factory=list)` so existing constructors that omitted it (none in the codebase besides the one updated here) would still work. The reranking/scoring.py `ScoreBreakdown` is a separately-named class in a different module — no collision.

## Strengths/weaknesses walks + coverage_exploration; drop intentionally_uncovered
Files: schemas/keyword_translation.py, schemas/semantic_translation.py, schemas/metadata_translation.py, search_v2/endpoint_fetching/category_handlers/schema_factories.py, search_v2/endpoint_fetching/category_handlers/output_extractor.py, search_v2/run_query_generation.py, search_v2/endpoint_fetching/category_handlers/prompts/buckets/{preferred_representation_fallback,semantic_preferred_deterministic_support,audience_suitability_deterministic_first}_{objective,guardrails}.md, search_v2/endpoint_fetching/category_handlers/prompts/endpoints/{keyword,semantic,metadata}.md, search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/{seasonal_holiday,emotional_experiential,cultural_status}.md

### Intent
Fix a structurally-adjacent bug introduced by the prior walk-then-commit refactor. End-to-end testing surfaced two failure modes the binary cover/miss walk vocabulary couldn't express: (1) `running movies` shipped `[SPORT]` alone with semantic dropped, because SPORT covers running but ALSO pulls football/basketball/hockey — over-broad, and the walk had no field for that signal; (2) `morally ambiguous protagonists` shipped `[ANTI_HERO]` alone and listed "moral gray area characterization" in `intentionally_uncovered` while semantic_walk's plot_analysis said `gap: nothing` for that exact slice — the field gave the LLM a soft-out instead of forcing it to use semantic. The bucket prompt's "don't pad fallback when preferred covers" guardrail was steering toward single-endpoint commits even when overlap was the design.

### Key Decisions
- **Strengths/weaknesses walk-candidate framing.** `PotentialKeyword`, `SpaceCandidate`, `ColumnCandidate` each now carry `strengths` (what the candidate genuinely OWNS at retrieval time) + `weaknesses` (under-coverage gaps AND over-coverage breadth — both belong in one freeform field, with suggested vocabulary `under-coverage:` / `over-coverage:` per principle 12). The four shapes are clean / under / over / both. Reframe applies to single-endpoint and multi-endpoint flows (shared classes); framing improvement helps both. Empirically verified: SPORT for "running movies" now writes `weaknesses: "over-coverage: ... pulls football, basketball, ... ; under-coverage: does not isolate running/track/marathon specifically"`.
- **`coverage_exploration: str` field.** New field between the per-endpoint walks and `coverage_assignments` in multi-endpoint bucket schemas. Argues which endpoints contribute distinct strengths or fill each other's weaknesses (puzzle-pieces composition) BEFORE the structural commit. Local tests built into the descriptor (principle 16): fire test = "does this endpoint contribute a strength the others don't, OR fill a weakness another has?"; drop test = "does another endpoint dominate this one's strengths AND weaknesses?"
- **`intentionally_uncovered` removed from the schema.** Empty `coverage_assignments` is the only abstain signal (whole-call). The field had become a soft-out the LLM used to declare slices unservable while a sibling endpoint's walk said it could be covered. Replaced with prompt-level invariant: every aspect surfaced in walks must be owned by some assignment, OR the candidate that surfaced it should have been dropped per the local tests. Routing-level mismatches surface upstream, not via this field.
- **Bucket prompts reframed around overlap-as-design.** Six prompts (objective + guardrails for buckets 5/6/8) rewritten. The old "don't pad fallback when preferred covers" guardrail replaced with the inverted pair: drop an endpoint ONLY when (a) another dominates it on the same content, or (b) its walk surfaced no useful candidate. Each guardrails file now leads with a NEVER list (principle 11). All `intentionally_uncovered` references replaced with explicit "this field doesn't exist" NEVER-list entries.
- **Endpoint prompts gained a strengths/weaknesses authoring section.** `keyword.md`, `semantic.md`, `metadata.md` each describe the four candidate shapes (clean / under / over / both) with operational framing, no example queries (principle 10).
- **Few-shot examples updated where stale.** The 3 bucket-6 example files (seasonal_holiday, emotional_experiential, cultural_status) had been rewritten in the prior refactor with intermediate vocabulary; updated again to use strengths/weaknesses + coverage_exploration. The other 47 examples and 50 additional_objective_notes files are clean.

### Verification
End-to-end via `python -m search_v2.run_query_generation`:
- **`running movies`** (regression test for over-coverage): `keyword_walk` SPORT explicitly names over-coverage; `semantic_walk` plot_analysis + plot_events surface clean strengths; `coverage_exploration` argues both fire; `coverage_assignments` has both keyword and semantic. Pre-refactor: SPORT alone, semantic dropped.
- **`morally ambiguous protagonists`** (regression test for under-coverage): Step 3 split into two category calls. Character archetype call → ANTI_HERO (weaknesses: none) alone, semantic dropped because dominated. Story/thematic archetype call → semantic plot_analysis (weaknesses: none) alone, keyword dropped because every keyword candidate showed both under- and over-coverage. Pre-refactor: ANTI_HERO alone with "moral gray area" in intentionally_uncovered.
- **`clown horror movies`** (original failure case): both keyword (HORROR) and semantic (plot_analysis with "clown imagery" thematic_concepts + viewer_experience with "eerie clown imagery") fire. The new behavior diverges from pre-refactor abstention: walks DID surface candidates with non-trivial strengths (HORROR's horror-leaning context overlap; semantic's natural-language clown vocabulary), so puzzle-pieces composition fires both. Original failure (committing COMEDY contrary to walk evidence) remains fixed.
- **`movies under 90 minutes`** (single-endpoint regression): Runtime metadata schema unchanged in shape; ColumnCandidate now uses strengths/weaknesses; commits `runtime < 90` correctly.

All 40 OUTPUT_SCHEMAS build at module import. Schema field order on multi-endpoint buckets: `[*_walk, coverage_exploration, coverage_assignments, *_parameters]`. Single-endpoint buckets unchanged.

### Testing Notes
- The clown-horror behavior shift (now fires both endpoints instead of abstaining) is by design: `coverage_exploration` is allowed to fire endpoints whose walks surface non-trivial strengths even when each candidate has weaknesses, since over-coverage in one is refined by another endpoint's specificity. Whole-call abstention is reserved for cases where NO walk surfaces useful strengths. If empirically the clown-horror retrieval pulls too many non-clown horror titles, the fix is in the semantic body authoring (more specific clown imagery vocabulary in plot_analysis.thematic_concepts), not in the commit shape.
- `unit_tests/test_schema_factories.py` is broken pre-this-refactor (stale class imports from a still-earlier change). Left broken; needs a separate test-update pass.
- The single-endpoint shape still emits `search_picture` (in `MetadataTranslationOutput`) and references it in `column_spec`/`scoring_method_reasoning` descriptors. Multi-endpoint dropped `search_picture` per the prior refactor. Cross-context descriptor language for the shared `ColumnCandidate` was already neutralized to "the call's intent" — no fresh drift.

## Walk-then-commit refactor for multi-endpoint handler buckets
Files: schemas/keyword_translation.py, schemas/semantic_translation.py, schemas/metadata_translation.py, search_v2/endpoint_fetching/category_handlers/endpoint_registry.py, search_v2/endpoint_fetching/category_handlers/schema_factories.py, search_v2/endpoint_fetching/category_handlers/prompts/buckets/*.md (6 files), search_v2/endpoint_fetching/category_handlers/prompts/endpoints/{keyword,semantic,metadata}.md

### Intent
Fix a structural failure mode in multi-endpoint handler buckets (5: PREFERRED_REPRESENTATION_FALLBACK, 6: SEMANTIC_PREFERRED_DETERMINISTIC_SUPPORT, 8: AUDIENCE_SUITABILITY_DETERMINISTIC_FIRST). Pydantic structured-output emits fields top-down; the prior schemas put prose coverage reasoning (`preferred_coverage_exploration`, `preferred_intent`, `augmentation_opportunities`, `coverage_opportunities`) BEFORE the registry/space/column-grounded analysis layer that lived inside each endpoint's subintent params. The LLM committed abstractly, then discovered contradictions in the grounded walk too late. Concrete failure: `clown horror movies` → `Element / motif presence` shipped `keyword_parameters.finalized_keywords=["COMEDY"]` while its own `potential_keywords` coverage prose said "Not covered." Inverted the order and split each multi-endpoint subintent param into a bucket-level grounded walk + thin commitment-only params; commitment now sits between them, grounded in the walks above.

### Key Decisions
- **Three-phase shape**: per-endpoint `{route}_walk` (analysis lifted from old subintent params) → `coverage_assignments: list[CoverageAssignment]` + `intentionally_uncovered: list[str]` → per-endpoint Optional `{route}_parameters` (thin commitment). One `_build_walk_then_commit` factory replaces `_build_preferred_fallback`, `_build_semantic_with_augmentation`, `_build_suitability_combo`. CoverageAssignment is a per-category dynamic class with `endpoint_kind: Literal[*declared_route_values]` (same dynamic-Literal pattern as the prior augmentation/coverage opportunity models).
- **Walk classes**: `KeywordWalk` (attributes + potential_keywords), `SemanticWalk` (aspects + space_candidates), `MetadataWalk` (column_candidates only). Registered in new `ROUTE_TO_WALK` dict in endpoint_registry; the factory raises if a multi-endpoint bucket declares a route with no walk class.
- **Thin subintent params**: `KeywordQuerySpecSubintent` keeps `finalized_keywords` + `scoring_method`; `SemanticParametersSubintent` keeps `role_exploration` + `role` + `space_queries` (role stays paired with role_exploration because they're semantic-internal commitment, not space-grounded analysis); `MetadataTranslationOutputSubintent` keeps `scoring_method_reasoning` + `column_spec` + `scoring_method`. `search_picture` is dropped from metadata's subintent shape — `coverage_assignments[kind=metadata].slice_description` plus the wrapper's `metadata_retrieval_intent` already supply the holistic restatement.
- **Class names kept**: `KeywordEndpointSubintentParameters` etc. retain their identifiers (now thinned). Avoids cascading renames across `endpoint_registry`, `endpoint_executors._WRAPPER_TO_ROUTE`, `run_search.py`, `run_search_json.py`. File-level docstrings updated to describe the walk-vs-thin contract.
- **`{route}_retrieval_intent` stays on the per-endpoint wrapper** (not lifted to bucket level). Per user direction: it represents the slice this endpoint owns, written off `coverage_assignments[endpoint_kind={route}].slice_description`. Walks read off the upstream `CategoryCall.retrieval_intent` from the user-message XML, not from any schema field.
- **`min_length=1` on inner thin-spec lists kept**: if commitment says don't fire, the thin params are null and never instantiated, so the constraint never bites. Forms a clean "if firing, must commit grounded" invariant.
- Inner-class collapse: `AttributeAnalysisSubintent` and `SpaceCandidateSubintent` deleted (replaced by reuse of shared `AttributeAnalysis` and `SpaceCandidate` whose descriptors already reference the user-message inputs the walks read). `WeightedSpaceQuerySubintent` retained because its descriptors still differ (reference `semantic_walk.space_candidates` vs `space_candidates` sibling).
- ColumnCandidate descriptor language softened from "search_picture" to "the call's intent" so the same class works in both single-endpoint (where search_picture is the prior sibling) and multi-endpoint walk (where the call's retrieval_intent plays that role) contexts. No content drift.

### Planning Context
Plan file at `~/.claude/plans/ok-this-all-makes-humming-hanrahan.md`. User clarifications during planning: (a) test_schema_factories.py was already broken at import (references nonexistent CarverSemanticEndpointSubintentParameters / QualifierSemanticEndpointSubintentParameters) — left broken, flagged for separate pass; (b) per-category prompt files in additional_objective_notes/ and few_shot_examples/ untouched per user — they don't carry old-schema JSON, output shape is conveyed by descriptors on the schemas themselves; (c) keep `{route}_retrieval_intent` on the wrapper as the slice commitment record, not at the bucket level.

### Verification
End-to-end test on the failing case: `python -m search_v2.run_query_generation "clown horror movies"` → "clown" trait → `Element / motif presence` (PREFERRED_REPRESENTATION_FALLBACK):
- Pre-refactor: `keyword_parameters.finalized_keywords=["COMEDY"]` despite coverage prose saying "Not covered."
- Post-refactor: `coverage_assignments=[]`, `intentionally_uncovered=["clowns as a notable element"]`, `keyword_parameters=null`, `semantic_parameters=null`. Handler abstains cleanly.

Buckets 3, 4 (single-endpoint), 7 (CHARACTER_FRANCHISE_FANOUT), and the no-LLM buckets verified to still build their schemas correctly. All 40 OUTPUT_SCHEMAS load at module import.

### Testing Notes
- `unit_tests/test_schema_factories.py` already broken at import pre-refactor (stale CarverSemantic/QualifierSemantic imports). Left broken per user direction; needs separate update pass.
- Worth covering eventually: a Bucket-5 case where the keyword walk surfaces a clean fit (commitment fires keyword), a Bucket-6 case where semantic + a deterministic both fire (overlap is the design), a Bucket-8 case where multiple endpoints catch distinct facets (hard ceiling on metadata + tag set on keyword + intensity on semantic).
- The walks now have `min_length=1` constraints on their internal candidate lists — same as before, but at a different schema position. If the LLM tries to abstain at the WALK level (rather than at the commitment level), validation fails. Watch for that pattern in production usage; the design intent is "always walk, abstain only at commitment."


## Stage-4 execution + ranking layer
Files: search_v2/stage_4_execution.py, search_v2/full_pipeline_orchestrator.py, search_v2/endpoint_fetching/category_handlers/generated_endpoint_spec.py

### Intent
Wire the front-half pipeline (Steps 0-3 + spec building) to actual endpoint execution and produce per-branch ranked candidate lists. Until this change, `run_full_pipeline` stopped at handing back `GeneratedEndpointSpec` lists; nothing fired them. Stage 4 closes that gap.

### Key Decisions
- New module `search_v2/stage_4_execution.py` owns all execution + scoring. The orchestrator stays focused on Steps 0-3 + auxiliary-spec planning and now calls `execute_branches(...)` at the tail of both bypass and standard paths.
- Recursive granularity per `search_improvement_planning/search_method_deterministic_logic.md` §3 / §5 / §6 / §7 / §8: category → trait → branch. Composites use **nested equal-weight averaging** (per category, then per trait) — chosen over a single flat trait-wide average because the user's design treats each level's composite as a single value at the next level. Worth flagging in §6 of the design doc as a follow-up.
- **Negative-polarity traits** use multiplicative-AND over would-be-generator calls, gated against the reranker average: `trait_score = (∏ G) × mean(R)` when both present; falls back to `∏ G` or `mean(R)` cleanly. Sign is applied at the §9 final-aggregation layer, not inside the trait.
- **Auxiliary specs** apply only at branch level, in order: NEUTRAL_SEED (additive — only when branch_pool is empty AND the upstream fallback emitted one) then MEDIA_TYPE shorts (subtractive — IDs from the SHORT-format MEDIA_TYPE generator are used as a blocklist, not a positive contribution). Both special-cased in `_apply_auxiliary` rather than going through the standard call dispatch, since neither contributes a trait_score.
- Added `was_promoted: bool = False` on `GeneratedEndpointSpec`, set True by `_apply_reranker_only_candidate_fallback` when it flips a reranker spec to CANDIDATE_GENERATOR. Stage-4 reads this so semantic-promoted traits' rarity counts only post-elbow 1.0 movies (per user clarification), not all matched candidates.
- Per-call soft-failure: `_dispatch_call` wraps `build_endpoint_coroutine` with a 25s timeout and converts exceptions to empty score maps. One bad call never tanks a whole trait or branch.

### Planning Context
Plan file at `~/.claude/plans/1-use-the-approach-melodic-cake.md`. The user explicitly chose: nested averaging at category and trait levels; multiplicative-gated for negative traits; mark `was_promoted` only at fallback time; only count post-elbow 1.0 movies for promoted traits' rarity.

### Testing Notes
- No automated tests added (per repo test-boundary rule). Manual exercise documented in plan file's Verification section.
- Worth covering eventually: negative-trait gated-multiplicative shape (a horror-comedy hits 1 of 3 G-calls vs all 3 — penalty must scale steeply); rarity tier boundaries on a small synthetic corpus; auxiliary shorts subtraction does not affect rerankers (it runs before them); branch with `branch_error` set yields empty `ranked` and propagates the error string.
- Risky edges: cand-gen trait whose pool is empty after generators returns None (drops trait silently); branch with empty cand-gen pool AND no NEUTRAL_SEED auxiliary returns empty (no automatic recovery — fallback already ran upstream).

### Self-review follow-ups
- **Negative-trait would-be-generator detection was broken.** Initial implementation partitioned negative-trait calls by `spec.operation_type`, but `handler.determine_operation_type` short-circuits every negative-polarity call to `POOL_RERANKER` regardless of route. The G partition was always empty → multiplicative-AND never fired → negative traits scored as a plain mean of rerankers (the OLD behavior). Fix: re-derive what each call WOULD have been with `determine_operation_type(category, route, Polarity.POSITIVE)`. Required carrying the originating CategoryName alongside each spec into the negative scorer.
- **`_dispatch_call` now returns `dict[int, float] | None`** so failed calls can be distinguished from legitimately empty results. Negative-trait scoring drops failed calls from both G product and R mean (option A); positive paths fold None → `{}` and absorb the zero contribution into their averages.
- Removed dead `_CallExecution.is_generator` field (read `spec.operation_type` directly). Simplified `_match_count_for_rarity` (no defensive intersection needed — `unioned ⊆ trait_pool` by construction). Dropped unused `Literal` import.
- Updated `docs/modules/search_v2.md` with the new Stage 4 design (recursive granularity, nested averaging, negative-trait gated multiplication, auxiliary-spec ordering, soft-failure semantics) and the gotcha that `operation_type` is uniformly `POOL_RERANKER` for negatives.

### CLI runner now displays ranked Stage-4 results
Files: run_orchestrator.py
Why: With Stage 4 producing per-branch ranked candidates, the existing CLI runner stopped at printing endpoint specs and didn't surface what the system actually returns. The notebook (`search_v2/test_stage_1_to_4.ipynb`) already had a "TOP N OF TOP K" display pattern; mirrored that for the CLI so a single `python run_orchestrator.py "query"` shows the full pipeline output end-to-end.
Approach: New `_print_ranked_results(branch_results)` performs one bulk `fetch_movie_cards([...])` per branch (per-candidate Postgres lookups would violate the cross-codebase invariant), formats each entry as `#rank score=±X.XXXX title (YEAR) tmdb_id=N`. Score uses `+.4f` so negative final scores (negative-polarity-dominated movies) show their sign. New `_ensure_db_ready()` opens the Postgres pool and inits Redis before pipeline execution since Stage 4 now hits live DB (was a no-op before). Added `search_v2.stage_4_execution` to the realtime-log filter so per-branch ranking timing and per-call failure warnings surface inline. Module docstring updated.
Files: search_v2/endpoint_fetching/metadata_query_execution.py
Why: Bring metadata in line with the semantic executor rework — endpoints emit raw [0, 1], cross-endpoint weighting is the merger's job. Also fix a long-standing semantic bug where popularity/reception used pure linear pass-through, so a 0.01 raw delta translated 1:1 into score, even though no user means "score 76" differently from "score 78" when they say "well-received."
Approach:
- Dropped `compress_to_dealbreaker_floor` from the dealbreaker path. Survivors now emit raw folded score in [0, 1]. Drop-on-zero rule kept (combined==0 rows still pruned from the candidate pool — they only matched the OR'd gate via one column and scored 0 raw on every populated column, so they shouldn't crowd out real candidates).
- Replaced `_score_popularity` and `_score_reception` with sigmoids anchored on real-world thresholds:
  - WELL_RECEIVED: center=70 (IMDb 7.0), k=0.22 → r=60→0.10, r=70→0.50, r=80→0.90, r=85→0.96. Saturation by ~80 reflects the user-meaning that 85 vs 90 is the same bucket.
  - POORLY_RECEIVED: center=50 (median; "average" stops being "poor" above this), k=0.22 → mirror of well.
  - POPULAR: center=0.70, k=12 → p=0.6→0.23, p=0.7→0.50, p=0.8→0.77, p=0.9→0.92. Anchors top ~30% as "popular," saturates by ~0.85.
  - NICHE: center=0.40, k=12. Slight asymmetry — "niche" requires being demonstrably outside the mainstream, not just below median.
- Left SQL `sort_signal_sql` linear. The all-unbounded path uses it only to LIMIT 5000 candidates; sigmoid is monotone in the same direction so the top-5000 set is unchanged. Avoids encoding `exp()` in SQL.
- Updated stale comment on `_score_country_position_dealbreaker` (the 0.33 raw value was previously calibrated against the floor compression, landing at 0.665 in the band; that calibration is gone but the value stays — pos 2 = 0.33 raw is still distinctly above 0 and below pos 1's 1.0).

Self-review follow-ups:
- Replaced dealbreaker drop filter `combined > 0.0` with `combined >= _DEALBREAKER_DROP_EPSILON` (=0.01). Sigmoid scorers emit tiny non-zero values in their saturated tails (e.g., reception=40 in WELL_RECEIVED → 0.0017) — those used to be exact 0 under the linear formulas and got dropped naturally. Without an epsilon, drop-on-zero degrades to a NULL-only filter and the candidate pool fills with noise. 0.01 corresponds to roughly r≈51 in WELL_RECEIVED / p≈0.32 in POPULAR — cleanly below the threshold a user means.
- Refreshed the now-stale `_make_reception_handler` sort_signal comment (no longer "mirrors `_score_reception`'s piecewise-linear shape"; it's deliberately kept linear for the SQL ORDER BY/LIMIT, with sigmoid applied post-fetch in Python — top-N is unchanged because sigmoid is monotone in reception).

Testing notes: Existing metadata unit tests will likely need updates: (a) dealbreaker scores no longer land in [0.5, 1.0] — raw [0, 1] now; (b) any test that asserts a specific popularity or reception score under the linear formulas needs new expected values from the sigmoid curves above; (c) country pos-2 dealbreaker now lands at 0.33 final (not 0.665); (d) dealbreaker drop threshold is now 0.01 not 0.0 — tests asserting "exactly 0 → dropped" still pass; tests asserting "tiny positive → kept" need new expected values.

## Semantic executor review fixes
Files: search_v2/endpoint_fetching/semantic_query_execution.py, docs/modules/search_v2.md
Why: Self-review of the semantic executor rewrite caught a negative-cosine edge case in the qualifier rescale, surfaced an executor-side empty-set short-circuit that duplicated orchestrator logic, and noted the module doc still implied universal [0.5, 1] dealbreaker compression.
Approach:
- Rewrote `_pool_relative_rescale` to be cosine-agnostic. New shape: linear-normalize cosines into [0, 1] over the pool range (min→0, max→1), then clamp anything ≥ QUALIFIER_TOP_RATIO (=0.85 in normalized space) to 1.0; everything below carries its normalized value. Old formula computed a ceiling at `top × 0.85` which broke for negative top cosines (ceiling landed above top, leaving the 1.0 band empty). Uniform-spread guard preserved unchanged.
- Replaced the qualifier+empty-restrict short-circuit with a "normalize empty restrict to None" pre-step at the top of `execute_semantic_query`. Per orchestrator contract, candidate-generator vs reranker dispatch is decided in build_endpoint_coroutine; an empty set arriving here is a leak, not a meaningful signal, so we treat it as None and run the candidate-generator path.
- Dropped redundant `int(mid)` casts in `_max_combine` / `_weighted_sum_combine` (callers already pass `set[int]`).
- Updated docs/modules/search_v2.md "Dealbreaker Score Floor" with a callout that semantic emits raw [0, 1] on every path now.
Testing notes: Inline rescale/combiner sanity checks (positive, negative, zero-straddling, uniform, cutoff boundary, missing-from-space) ran clean. Existing semantic unit tests likely need updates given (a) negative-cosine output now lands top→1.0 instead of top→<1.0; (b) qualifier rescale shape (normalized-value-as-score below cutoff vs old "rescale into [0, ceiling]" behavior) produces different intermediate scores; (c) qualifier+empty-restrict no longer short-circuits.

## Rework semantic executor: role drives both within-space norm and cross-space combine
Files: search_v2/endpoint_fetching/semantic_query_execution.py

### Intent
Realign semantic execution with the absolute-vs-relative split that `role` actually expresses. Carver names a population ("does this movie have X?") so its within-space normalization needs an absolute corpus-calibrated bar; qualifier positions against a reference ("how X are these movies relative to each other?") so its within-space normalization is purely pool-relative. The cross-space combiner also keys off role: carver max() (one-strong-signal-is-enough — ANDs across distinct questions are split into separate traits upstream), qualifier weighted sum with CENTRAL=2.0 / SUPPORTING=1.0.

### Key Decisions
- Four execution scenarios keyed on (role, restrict-presence):
  carver+restrict → corpus probe ‖ HasId per space, elbow decay, max combine.
  carver+no restrict → corpus probe = pool, elbow decay, max combine.
  qualifier+restrict → HasId per space, pool-relative rescale, weighted sum.
  qualifier+no restrict → tier-fallback promoted: corpus probe = pool, elbow decay, weighted sum.
  Both role/restrict assertion errors removed — carver+restrict and qualifier+no-restrict are now legal modes (the latter only via orchestrator-level tier-fallback promotion).
- Carver combiner switched from average-with-drop+compress to plain max(). User clarified each semantic call asks one question with possibly-uneven evidence across spaces, so OR-shaped combination is correct; AND across distinct questions happens at the trait/orchestrator level.
- Dropped [0.5, 1] dealbreaker compression on the carver path. Scores live in [0, 1] truthfully; ScoredCandidate already accepts that range.
- Dropped drop-on-zero rule. A candidate scoring 0 across every space lands at 0 naturally and contributes nothing downstream via "missing positive = opportunity cost" (search_method_deterministic_logic.md §6/§8).
- New `_pool_relative_rescale` helper: top × 0.85 → 1.0, linear decay to pool min → 0.0, with uniform-spread guard (max−min < 0.01 → all 1.0) to avoid amplifying numerical noise into apparent ranking when a space carries no differentiating signal for the pool.
- Elbow detection (Kneedle + EWMA + pathology fallback + Path B pass-through) preserved unchanged — it remains the right calibration for any path with a corpus probe.
- `compress_to_dealbreaker_floor` import removed; helper itself left in result_helpers.py for other endpoints.

### Planning Context
Design converged through conversation on search_improvement_planning/search_method_deterministic_logic.md. Key reframes during the discussion: (1) the planning doc's "always pool reranker + always elbow normalize on top × 0.85" model conflated within-space and cross-space concerns and used pool-local calibration where corpus-local was needed; (2) role isn't dead — it earns its keep when split across the two axes; (3) ANDs across distinct questions are decomposed upstream into separate traits, so within-call combination is OR-shaped (max), not AND-shaped (average-with-drop).

### Testing Notes
Per project test-boundary instruction, no tests run. Behavior changes that affect callers/tests: (a) executor now legally accepts carver+restrict and qualifier+no-restrict instead of asserting; (b) carver scores are no longer compressed into [0.5, 1]; (c) carver no longer drops candidates that fail every space (they appear with score 0); (d) qualifier output now lives in [0, 1] post-rescale instead of being raw weighted cosines. Downstream merge/orchestration code that assumed any of (a)–(d) needs a look.

## Respect restrict pools for temporary TRENDING execution
Files: search_v2/endpoint_fetching/category_handlers/handler.py
Why: `run_query_execution()` special-cases TRENDING because the route has no params wrapper and cannot go through `build_endpoint_coroutine()`, but it was hardcoding `restrict_to_movie_ids=None`. That bypassed the same operation-type rules the shared dispatcher applies to every other route.
Approach: Added the minimal inline gating to the TRENDING branch: pool rerankers with no candidate pool return without running, candidate generators locally reset `restrict_to_movie_ids` to `None`, and otherwise the existing pool is passed into `execute_trending_query()`.
Testing notes: No tests run per project test-boundary instruction.

## Add endpoint-spec promotion tier helper
Files: search_v2/full_pipeline_orchestrator.py, unit_tests/test_full_pipeline_promotion_tiers.py
Why: Reranker-only fallback needs a deterministic way to rank promotable endpoint specs while preserving negative polarity as never-promote.
Approach: Added `PromotionTier` as an ordered `IntEnum` with `NEVER_PROMOTE = -1` and tiers 1-7 matching the planning doc. Added `determine_promotion_tier(category, endpoint_spec, polarity)` so route-specific cases like `CULTURAL_STATUS` can split semantic prose (Tier 4) from metadata prior (Tier 7). Candidate-generator specs, negative-polarity specs, and unmapped positive rerankers return `NEVER_PROMOTE`.
Testing notes: Added table-driven unit coverage for every tier row, candidate-generator non-promotion, negative polarity, unmapped rerankers, and enum ordering. `uv run pytest unit_tests/test_full_pipeline_promotion_tiers.py -v` passed (35 tests).

## Centralize neutral reranker seed formula in Postgres helper
Files: db/postgres.py, search_v2/full_pipeline_orchestrator.py, unit_tests/test_neutral_reranker_seed.py
Why: The neutral top-2000 seed formula was duplicated between the orchestrator constants/comment and the DB helper, creating drift risk.
Approach: Made `db/postgres.py` the source of truth with `NEUTRAL_RERANKER_SEED_LIMIT`, `NEUTRAL_RERANKER_SEED_POPULARITY_WEIGHT = 0.8`, and `NEUTRAL_RERANKER_SEED_RECEPTION_WEIGHT = 0.2`. `fetch_neutral_reranker_seed_ids()` now defaults to the shared limit and parameterizes SQL with those constants. Removed the duplicate orchestrator constants; the neutral seed spec comment now points to the DB helper.
Testing notes: No tests run per user instruction; updated the existing neutral seed test expectations to assert against the DB constants.

## Integrate reranker-only fallback tiers for metadata and negative polarity
Files: search_improvement_planning/search_method_deterministic_logic.md
Why: The new endpoint-level candidate-generator vs reranker designation creates queries with no candidate-generating calls, including all-negative traits and metadata/semantic-only reranker traits.
Approach: Extended tier-fallback promotion beyond semantic-only categories. Added Tier 7 for global metadata priors/ordinals (general appeal, cultural status, chronological) so reception/popularity/date priors can seed a pool before emergency fallback. Added Tier NP ("never promote") for negative-polarity calls and unresolved metadata directions; reaching it seeds top 2,000 candidates by default popularity/reception blend, then runs rerankers.
Testing notes: Documentation/design update only; no tests run.

## Specify neutral top-2000 fallback seed formula
Files: search_improvement_planning/search_method_deterministic_logic.md
Why: The Tier NP backup needs a deterministic pool seed, not an underspecified "popularity/reception blend."
Approach: Defined the fallback fetch as top 2,000 movies by `0.8 * normalized_popularity_score + 0.2 * normalized_reception_score`, with both component scores normalized to `[0, 1]`.
Testing notes: Documentation/design update only; no tests run.

## Inject default shorts-exclusion MEDIA_TYPE fetch when none present
Files: search_v2/full_pipeline_orchestrator.py
Why: Without an explicit MEDIA_TYPE call from any branch/trait, shorts can leak into the result set. Stage-4 still needs a way to fully exclude them by default.
Approach: After per-trait endpoint generation completes, scan all branches/traits/category_calls for any CategoryName.MEDIA_TYPE call. If none exists, build a single GeneratedEndpointSpec(route=MEDIA_TYPE) with formats=[ReleaseFormat.SHORT] and surface it on FullPipelineResult.auxiliary_endpoint_specs (new field for fetches not attached to a trait). Both the bypass path and the standard path call _build_auxiliary_specs(branches). No execution wired yet — this is a placeholder for stage-4 to globally exclude shorts.
Testing notes: Verify that a query with an explicit MEDIA_TYPE category-call (e.g. "tv movies", "shorts") emits no auxiliary spec, while a query without any media-type intent surfaces exactly one MEDIA_TYPE/SHORT auxiliary spec.

## Atomic-form rule + multi-credit-per-film + unbounded character walk
Files: schemas/entity_translation.py, search_v2/endpoint_fetching/category_handlers/endpoint_registry.py, search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/{person_credit,named_character,character_franchise}.md

### Intent
Three coupled refinements to the film-walk template:
1. The forms list is matched by exact-string equality against an atomic credit dictionary, but the template was producing bundled strings (slash-combined "X / Y", embedded stage names like "Curtis '50 Cent' Jackson"). Bundled strings match nothing.
2. The character_exploration walk was capped at "3-5 notable films" — wrong for characters with many incarnations / aliases (more films may be needed) and wasteful for characters with one consistent credit (fewer suffice). The LLM should pick the count.
3. The "Credit per film" line implied one credit per film, but real cast blocks often list a character under several forms simultaneously (civilian + alter-ego, alias + legal name).

### Key Decisions
- Reframed `Credit per film` as a comma-separated list of every form the entity is billed under in that film's cast block, with each entry an atomic single identity. If a literal cast-list credit bundles identities (slash-combined or stage-name-embedded), the LLM splits it on the per-film line so each entry is queryable.
- Added an explicit "atomic = one identity per form" rule to the template body, with a generalized description of the split (no specific name examples). The forms field description was reduced — now points at the template's "Distinct forms" line as source of truth and inherits the atomic invariant from there. Removed the old "the cop / the wizard" generic-label examples per the no-examples-in-descriptors rule.
- Replaced `character_exploration`'s "3-5 notable films" cap with "the most popular / relevant films featuring this character — walk enough to surface every distinct credited form". The LLM picks N. `person_exploration` keeps the 3-5 cap because actor credits tend to be more uniform (no specific instruction to drop it).
- Mirrored every change in `CharacterFranchiseFanoutSchema.character_form_exploration` and `.character_forms` so bucket 7 (CHARACTER_FRANCHISE_FANOUT) stays in sync with bucket 3 (NAMED_CHARACTER).
- Updated the few-shot examples to demonstrate the atomic split: 50 Cent's Den of Thieves credit becomes "Curtis Jackson, 50 Cent" (was "Curtis '50 Cent' Jackson"); Indiana Jones's Last Crusade becomes "Indiana Jones, Henry Jones Jr." (was slash-combined); Iron Man's first film becomes "Tony Stark, Iron Man"; Voldemort's Deathly Hallows becomes "Lord Voldemort, Tom Riddle". Distinct forms / forms lines now show only atomic single-identity entries.
- Scrubbed all specific-name examples from field descriptions per the no-examples rule (especially anything resembling active test queries — The Rock, Joker). Field descriptions now use generalized rules; specific examples live only in the few-shot section where they belong.

### Testing Notes
Schema round-trip verified via inline import. Re-run the Joker query through the test notebook: character_forms should now contain atomic entries (Joker, The Joker, Arthur Fleck, Jack Napier, Heath Ledger's Joker incarnation if the model recalls it) rather than slash-bundled "Arthur Fleck / Joker". Re-run a single-form case (Tom Hanks / Yoda) to confirm the atomic rule doesn't cause over-splitting on credits that are already atomic.

## Redesign CharacterFranchiseFanoutSchema with split per-side walks
Files: search_v2/endpoint_fetching/category_handlers/endpoint_registry.py, search_v2/endpoint_fetching/category_handlers/handler.py, search_v2/endpoint_fetching/category_handlers/prompts/buckets/character_franchise_fanout_objective.md, search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/character_franchise.md, schemas/entity_translation.py, search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/{person_credit,named_character}.md

### Intent
The CHARACTER_FRANCHISE_FANOUT bucket (bucket 7) uses its own shared `CharacterFranchiseFanoutSchema`, NOT the `CharacterQuerySpec` whose `character_exploration` description was rewritten to mandate the film-walk template earlier in the session. Edits to `CharacterQuerySpec` only reach the LLM for the `NAMED_CHARACTER` category; `CHARACTER_FRANCHISE` queries (e.g. "The Joker") go through the fanout schema, which still asked for a single shared "referent_form_exploration" sentence. The handler then synthesized a CharacterQuerySpec by copying that sentence into both `query_exploration` and `character_exploration` — explaining why the test output for Joker showed an identical descriptive sentence in both fields and missed Jack Napier / Arthur Fleck entirely.

### Key Decisions
- Split `referent_form_exploration` into two parallel exploration fields: `character_form_exploration` (mirrors `CharacterTarget.character_exploration` template byte-for-byte — same film walk, same spartan tone, same "queried form is often not the dominant credit" warning) and `franchise_form_exploration` (its own template covering Series / Umbrella / Subgroups / Distinct forms — different shape because the franchise side queries series/universe titles, not per-film cast strings).
- Updated `_fanout_to_fired_endpoints` in handler.py to wire each side to its own exploration string instead of copying a single referent into both. Added a fixed sentinel for `query_exploration` on the synthesized CharacterQuerySpec ("single referent — fanout retrieval emits one CharacterTarget per call by design") since the fanout has no equivalent input.
- Rewrote `character_franchise_fanout_objective.md` to describe two parallel walks rather than "resolve the named referent once" — the latter was the source of the model's behavior of producing one descriptive sentence for both sides. Explicit instruction: "Each walk is filled in literally per its schema template — do not produce a single generic referent description and copy it into both fields."
- Rewrote the few-shot examples to demonstrate both templates filled in: Indiana Jones, Iron Man, James Bond, and Sherlock now show character_form_exploration and franchise_form_exploration as separate populated templates plus the resulting form lists. The Iron Man example specifically shows "Tony Stark" surfacing as the dominant cast credit while "Iron Man" is the franchise title, illustrating that the same name plays different roles on each side.
- Swapped the Unicode rightwards arrow `→` for ASCII `:` as the credit separator in all template descriptions and few-shot examples (entity_translation.py + person_credit, named_character, character_franchise few-shots). Reasons: `→` tokenizes as 2-3 tokens vs 1 for `:` on most LLM tokenizers (model copy fidelity drops with multi-token symbols), and unicode characters can mangle through some terminal/copy-paste paths. Verified the schema round-trips through the fanout adapter via inline import test.

### Testing Notes
Manual test: re-run the Joker query through `test_v3_endpoints.ipynb` and verify `character_exploration` now shows the populated film-walk template with Jack Napier / Arthur Fleck, and `request_overview` shows the franchise template with the Batman / DC umbrella context. Both should be distinct strings, not the same sentence copied. Iron Man should produce "Tony Stark" first in character_forms and "Iron Man" / "Marvel Cinematic Universe" in franchise_forms.

## Switch entity exploration fields to film-walk template; format-lock via few-shots
Files: schemas/entity_translation.py, search_v2/endpoint_fetching/category_handlers/prompts/categories/additional_objective_notes/{person_credit,named_character,character_franchise}.md, search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/{person_credit,character_franchise,named_character}.md

### Intent
Address retrieval failures observed during testing: querying "The Rock" was not surfacing Dwayne Johnson; querying "The Joker" was not surfacing Jack Napier or Arthur Fleck. Root cause: `person_exploration` and `character_exploration` framed alias discovery as an abstract form inventory ("which forms might appear in credits") rather than grounded recall. The model defaulted to the queried surface form and trimmed anything it could not directly justify. An initial freeform spartan rewrite did not move the needle — the model kept producing prose and skipping the film walk.

### Key Decisions
- Switched `person_exploration` and `character_exploration` from a freeform descriptive sentence to a literal **fill-in template** the model must complete: `Films:` line, per-film `Credit per film:` bullets, `Distinct forms:` line, and (person only) `Predominant role:` line. Per small-LLM intuitions, format-locking via templates plus few-shot examples produces more reliable structured behavior than prose instructions, especially for "do this film walk" patterns the model otherwise skips.
- Schema shape unchanged — the template lives inside the existing `constr` string field. No downstream code reads the exploration content; it exists as a reasoning scaffold for the model's own committed fields (`forms`, `person_category`, `prominence_mode`). Keeps the change purely prompt-side.
- Rewrote every exploration field description across PersonTarget, CharacterTarget, TitlePatternTarget, and the three QuerySpec containers in spartan tone (terse, imperative). Compressed prior NEVER-lists into short positive rules — small models often violate negative instructions immediately, so positive guidance plus format-locking few-shots carry more weight.
- Replaced every prose "Expected: <description>" few-shot in person_credit, character_franchise, and named_character with worked-example template fills. Single-form cases (Tom Hanks, Yoda, Hermione, Bigelow, Williams) demonstrate that consistent credit → single form (prevents over-generation). Multi-form cases (50 Cent, Indiana Jones, Iron Man, Voldemort) demonstrate the across-films walk that surfaces non-queried credits.
- Test-set hygiene: avoided The Rock, Batman, and Joker as few-shot subjects (active test queries). Used 50 Cent for stage-name-vs-legal-name, Indiana Jones for credit-string drift across a franchise, Iron Man for queried-name-not-dominant, Voldemort for civilian-name-on-non-franchise-anchor. Removed the pre-existing Batman example from named_character (it taught the wrong routing — Batman is character_franchise) and from character_franchise (overlap with the Joker test case).
- Added an inline "queried surface form is often NOT the dominant credit" line to both exploration fields so the model has explicit permission to override the input. This is the lesson the failure cases share.

### Testing Notes
Manual test: re-run "The Rock", "The Joker", "Dwayne Johnson", "50 Cent" via "Iron Man" and verify forms now include legal-name and per-incarnation cast-list strings. Watch for over-generation on simpler cases (Tom Hanks, Hermione Granger) where the queried form is genuinely the only credit string — the single-form template examples should counteract this. The model should produce literally-formatted exploration strings (Films:, Credit per film:, Distinct forms: lines); if it reverts to prose, format-lock is failing and the template needs sharper anchoring (e.g. an explicit "must start with `Films:`" rule).

## Move seasonal/holiday to semantic-always bucket
Files: search_improvement_planning/query_buckets.md, schemas/trait_category.py | Moved Seasonal / holiday from Preferred Representation With Fallback / Gap-Fill into Semantic-Always With Deterministic Augmentation, and changed the enum endpoint order to semantic first so deterministic keyword tags augment rather than replace the semantic seasonal read.

## Rewrite Preferred Representation With Fallback category prompt chunks
Files: search_v2/endpoint_fetching/category_handlers/prompts/categories/additional_objective_notes/{central_topic,element_presence,character_archetype,genre,cultural_tradition,format_visual,narrative_devices,story_thematic_archetype,specific_praise_criticism}.md; search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/{central_topic,element_presence,character_archetype,genre,cultural_tradition,format_visual,narrative_devices,story_thematic_archetype,specific_praise_criticism}.md; deleted stale {specific_subject,top_level_genre,sub_genre,kind_of_story}.md prompt chunks | Replaced the remaining Bucket 5 category prompts with compact, section-anchored notes and shape-focused examples calibrated for preferred-only, fallback-only, split, and no-fire decisions.

## Stale-reference cleanup after the EntityEndpointParameters removal
Files: schemas/entity_translation.py, schemas/enums.py, search_v2/endpoint_fetching/endpoint_executors.py, search_v2/endpoint_fetching/entity_query_generation.py, search_v2/endpoint_fetching/category_handlers/handler.py, search_v2/reranking/dispatch.py, run_search.py, run_search_json.py

### Intent
Sweep the codebase for references to symbols that no longer exist after recent refactors and the EntityEndpointParameters removal: `SemanticEndpointParameters` (split into `CarverSemanticEndpointParameters` + `QualifierSemanticEndpointParameters` long ago), `EntityQuerySpec` (replaced by per-category specs), `EntityEndpointParameters` (just deleted), and old `HandlerBucket` values (`SINGLE`, `MUTEX`, `TIERED`, `COMBO` — replaced by the 8-bucket taxonomy). Also fold in the two suggestions from the prior review.

### Key Decisions
- Replaced every `isinstance(x, SemanticEndpointParameters)` with `isinstance(x, (CarverSemanticEndpointParameters, QualifierSemanticEndpointParameters))`. Tuple isinstance is the standard pattern for "any of these types"; preferred over a Union type alias because the call sites are few and explicit naming makes the intent obvious.
- Stubbed `entity_query_generation.generate_entity_query` to raise `NotImplementedError` rather than deleting the file. The function is still imported by `search_v2/reranking/dispatch.py`; keeping the symbol intact preserves the import graph until the v2 dispatch path is removed wholesale, while making any actual call fail loudly with a pointer to the v3 replacement. The module-level prompt strings are now unused but left in place — pruning them is a separate, larger cleanup.
- Rewrote `handler._extract_fired_endpoints` for the 8-bucket taxonomy. Single-endpoint buckets (3, 4) keep the old `should_run_endpoint` + `endpoint_parameters` shape. Buckets 5/6/8 share a generic shape — Optional `<route>_parameters` field per candidate — so the dispatch walks `model_fields`, picks fields ending in `_parameters`, and parses the prefix as an `EndpointRoute`. CHARACTER_FRANCHISE_FANOUT (bucket 7) raises `NotImplementedError`: its schema emits form lists, not endpoint payloads, and translating those into `(route, EndpointParameters)` pairs requires a custom path that doesn't yet exist. Verified that every `HandlerBucket` value and every `CategoryName.bucket` lands in an explicit branch.
- Applied review suggestions: replaced the loop variable leak in `endpoint_executors._WRAPPER_TO_ROUTE` with `dict.update`, and dropped the empty line after each entity QuerySpec class declaration.
- Updated stale prose in `schemas/enums.py`, `search_v2/reranking/dispatch.py`, and `entity_query_generation.py`'s module header so the comments match the new symbol names.

### Testing Notes
Verified via inline scripts: (1) every touched module py_compile-clean and importable (excluding a pre-existing missing `search_v2.stage_4` module that breaks `search_v2.reranking.__init__`); (2) the rewritten `_extract_fired_endpoints` covers every `HandlerBucket` and every `CategoryName.bucket` value; (3) for every category in the per-route bucket family, the schema's `<route>_parameters` field names round-trip back to valid `EndpointRoute` values via the handler's parsing logic. Per the test-boundary rule, did not touch `unit_tests/test_schema_factories.py` (still references `EntityEndpointParameters` and `EndpointRoute.ENTITY` in `ROUTE_TO_WRAPPER` — a separate testing-pass update). The remaining `search_v2.stage_4` missing-module error is unrelated and unchanged.

## Replace EntityEndpointParameters union wrapper with per-category dispatch
Files: schemas/entity_translation.py, search_v2/endpoint_fetching/category_handlers/endpoint_registry.py, search_v2/endpoint_fetching/category_handlers/schema_factories.py, search_v2/endpoint_fetching/endpoint_executors.py

### Intent
Remove the EntityEndpointParameters wrapper whose `parameters` field was a Union of PersonQuerySpec / CharacterQuerySpec / TitlePatternQuerySpec, and have each entity-family category (PERSON_CREDIT, NAMED_CHARACTER, TITLE_TEXT) emit its narrowed spec directly. The previous union approach gave the LLM the same broad schema regardless of category and relied entirely on per-category prompt steering to pick the right member.

### Key Decisions
- Made PersonQuerySpec / CharacterQuerySpec / TitlePatternQuerySpec inherit from EndpointParameters directly (instead of plain BaseModel). This is the contract HandlerResult.preference_specs and the orchestrator's isinstance-routing rely on. Their target sub-models (PersonTarget / CharacterTarget / TitlePatternTarget) stay as BaseModel — they never escape their parent spec, so they don't need the marker.
- Added _ENTITY_DISPATCH in endpoint_registry as a sibling to _SEMANTIC_DISPATCH: CategoryName -> spec class. Added a `category` kwarg to get_output_wrapper, mirroring `role`. ENTITY now lives outside ROUTE_TO_WRAPPER (analogous to how SEMANTIC has been since it was first authored).
- _resolve_wrappers_for_bucket now takes the CategoryName directly (instead of a destructured endpoints tuple + name string) so it can pass the category through to get_output_wrapper. Every _build_* factory's signature collapsed to (category, bucket, role) — endpoints and name are derived from the CategoryName inside.
- ENTITY routes do not have subintent variants today and no current bucket pairs ENTITY with another entity-family endpoint requiring a slice-of-intent split, so requesting ENTITY under a multi-endpoint bucket raises rather than silently returning the wrong shape.
- In endpoint_executors.build_endpoint_coroutine, ENTITY is the one route where `wrapper` IS the spec (no `.parameters` indirection). Hoisted the ENTITY branch above `params = wrapper.parameters` to handle this cleanly.
- The reverse mapping _WRAPPER_TO_ROUTE used to be a pure inversion of ROUTE_TO_WRAPPER. Since ENTITY no longer appears in ROUTE_TO_WRAPPER but the orchestrator still needs to map deferred preference_specs back to a route, the three entity specs are explicitly registered to EndpointRoute.ENTITY after the dict comprehension.

### Planning Context
User asked first what logic fetches the correct schema for the three entity categories, then noticed that today there is no per-category schema — all three resolve to the same EntityEndpointParameters whose `parameters` is a Union. That is the gap this change closes: per-category dispatch at the schema level so each category's LLM sees only the relevant subset of fields.

### Testing Notes
Verified via inline scripts: (1) get_output_schema(cat, role) for each entity category resolves `endpoint_parameters` to the right narrowed Optional[Spec] type; (2) ENTITY is gone from ROUTE_TO_WRAPPER; (3) _ENTITY_DISPATCH covers exactly the three categories; (4) get_output_wrapper raises a clear ValueError when ENTITY is requested without a category; (5) the bucket-7 short-circuit still returns CharacterFranchiseFanoutSchema regardless of category. Per the test-boundaries rule, did not modify or run tests — but unit_tests/test_schema_factories.py imports EntityEndpointParameters and references EndpointRoute.ENTITY in ROUTE_TO_WRAPPER, both of which are now stale; that test file will need updating in a separate testing pass. Also note: endpoint_executors.py has a pre-existing import bug (it tries to import a non-existent SemanticEndpointParameters from semantic_translation) that predates this change and is unrelated; the file still imports cleanly via py_compile.

## Rewrite shared handler prompt chunks (role / vocab / input spec)
Files: search_v2/endpoint_fetching/category_handlers/prompts/shared/{role,shared_vocabulary,input_spec}.md
Why: After the bucket remap and CategoryCall-only payload, all three shared chunks were stale. role.md still framed the job as a per-call "decide which endpoints fire" routing decision (wrong for buckets 6/7/8). shared_vocabulary.md defined match_mode and fit_quality (neither exists on CategoryCall, neither is emitted) and framed polarity as something the LLM emits (it doesn't — handler.py stamps it from the parent Trait post-emit). input_spec.md contained per-endpoint-family expression rules and carver/qualifier framing that are prescriptive (charter says input spec is descriptive only) and only true for a subset of buckets.
Approach: Used category_handler_planning.md's "Full system-prompt composition" charter to scope each chunk by its single purpose — role is identity + posture, vocabulary is byte-identical definitions of terms used elsewhere, input spec is a descriptive payload field reference with a "what's intentionally absent" list. Folded the universal handler postures the user wants preserved into role (trust upstream routing, abstain only when no endpoint genuinely fits with the bar set high because dispatch already vouched). Reframed polarity in vocab as the positive-presence invariant (committed upstream, stamped post-hoc, parameters always describe what to FIND). Added a CategoryCall vocab entry so bucket prompts and endpoint prompts have a shared term to refer to. Stripped MEDIA_TYPE references and the "natural shape is one endpoint firing" claim that was true only for single-endpoint buckets.
Design context: search_improvement_planning/category_handler_planning.md §"Full system-prompt composition" defines each shared chunk's purpose; query_buckets.md and the up-to-date bucket prompts confirmed that decomposition shape, abstain mechanics, and reasoning-field naming all vary by bucket and therefore should not live in shared docs.
Testing notes: Confirmed via inline import that all three chunks load and compose. No unit tests run per test-boundary rule. Endpoint prompts still restate "trust upstream routing" and "positive-presence invariant" individually (entity.md, semantic.md) — those restatements are now redundant and could be removed in a follow-up sweep.

## Generalize prompt-builder no-LLM guard to bucket check
Files: search_v2/endpoint_fetching/category_handlers/prompt_builder.py
Why: The TRENDING-only raise in build_system_prompt no longer covers the full set of categories that have no LLM handler. After the bucket remap, MEDIA_TYPE (NO_LLM_PURE_CODE) and BELOW_THE_LINE_CREATOR (EXPLICIT_NO_OP) are also handler-less, and EXPLICIT_NO_OP has no objective/guardrails .md authored.
Approach: Introduced `_NON_LLM_BUCKETS = {NO_LLM_PURE_CODE, EXPLICIT_NO_OP}` and used it to (a) skip those buckets when eager-loading bucket objective/guardrail chunks (avoiding a startup FileNotFoundError once EXPLICIT_NO_OP exists), (b) skip those categories in `_load_category_chunks`, and (c) replace the `category is CategoryName.TRENDING` check in `build_system_prompt` with `category.bucket in _NON_LLM_BUCKETS`. Updated docstring accordingly.
Testing notes: Confirmed via inline import script that build_system_prompt now raises for TRENDING, MEDIA_TYPE, and BELOW_THE_LINE_CREATOR. Per test-boundary rule, no unit tests run.

## Category endpoint order alignment
Files: schemas/trait_category.py
Why: User requested category endpoint tuples be verified against search_improvement_planning/query_categories.md so preferred endpoints appear first, after undoing speculative bucket-name-based ordering changes.
Approach: Restored the speculative audience-suitability ordering changes and used query_categories.md as the source of truth. Updated Emotional / experiential and Specific praise / criticism to put semantic before keyword because their category docs list semantic reception/experience surfaces before KW support. Updated Below-the-line creator to an empty endpoint tuple because it is explicitly reserved/no-op and returns empty for now.
Testing notes: Ran a lightweight enum verification script checking the priority-sensitive endpoint tuples and key single-endpoint categories. No unit tests run per test-boundary rule.

## HandlerBucket enum remap to query bucket taxonomy
Files: schemas/enums.py, schemas/trait_category.py, docs/modules/schemas.md
Why: User requested the code-level HandlerBucket enum be changed to the new query-generation buckets from search_improvement_planning/query_buckets.md, and each CategoryName member remapped accordingly, while deliberately leaving downstream bucket consumers unchanged for now.
Approach: Replaced the old SINGLE/MUTEX/TIERED/COMBO HandlerBucket values with eight instruction-shape buckets: no-LLM pure code, explicit no-op, single non-metadata endpoint, single metadata endpoint, preferred representation fallback, semantic-preferred deterministic support, character-franchise fanout, and audience-suitability deterministic-first. Updated every CategoryName enum member's bucket assignment to match query_buckets.md and corrected the schemas module doc's HandlerBucket summary; no changes made to schema_factories, prompt_builder, handler extraction, or prompt chunk files.
Testing notes: Ran a lightweight import/listing script for schemas.trait_category only to confirm CategoryName loads and bucket counts match the planning doc. Did not run unit tests per test-boundary rule. Downstream consumers are expected to be broken until the next implementation step updates them for the new bucket enum.

## Query category bucket planning doc
Files: search_improvement_planning/query_buckets.md
Why: User wanted the active query categories grouped by reusable query-generation instruction shape, with the "keyword preferred" bucket generalized because not every case prefers keywords or falls back to semantics.
Approach: Added a new planning doc with eight buckets: no-LLM/pure-code, explicit no-op, single non-metadata endpoint, single metadata endpoint, preferred representation with fallback/gap-fill, semantic-preferred with deterministic support, character-franchise fan-out, and audience-suitability deterministic-first combo. Each bucket documents what it represents, why it is separate, how query generation should behave, and category names covered.
Testing notes: Docs-only planning change; checked the file for coverage of all active categories 1-42 and 44. No tests run per test-boundary rule.

## Shared ANY/ALL scoring method for keyword, metadata, and studio endpoints
Files: schemas/enums.py, schemas/{keyword_translation,metadata_translation,studio_translation}.py, search_v2/endpoint_fetching/{keyword_query_execution,metadata_query_execution,studio_query_execution}.py, search_v2/endpoint_fetching/category_handlers/prompts/endpoints/{keyword,metadata,studio}.md, docs/modules/schemas.md
Why: Keyword, metadata, and studio used separate scoring/composition names (`any`/`avg`, `max`/`average`, `MAX`/`AVERAGE`) for the same conceptual choice. User requested one shared enum with ANY/ALL semantics.
Approach: Added `ScoringMethod` with `ANY` and `ALL`; renamed the three endpoint output fields to `scoring_method`; updated schema descriptors and endpoint prompt docs with the shared definitions. Executors now branch on `ScoringMethod.ANY`: keyword maps ANY to any-hit 1.0 and ALL to matched fraction, metadata maps ANY to max across populated columns and ALL to mean across populated columns, and studio maps ANY to union/max across studio refs and ALL to per-ref matched fraction.
Design context: Follows the category-handler endpoint schema convention that the LLM commits the scoring/composition method explicitly on the endpoint payload.
Testing notes: Ran `python -m compileall -q` on touched schemas/executors and a small Pydantic smoke script confirming keyword/metadata/studio accept and dump `ANY`/`ALL` while executor comparisons against `ScoringMethod` work. Per test-boundary rule, no unit tests were run or edited.

## Documentation staleness sweep (post docs-auditor)
Files: docs/PROJECT.md, docs/modules/{llms,schemas,search_v2,db,ingestion,classes}.md, docs/decisions/ADR-060/074/075.md, docs/decisions/ADR-073-award-category-tag-taxonomy.md, CLAUDE.md
Why: docs-auditor surfaced 22 stale items across module docs, conventions, and ADRs. User requested all corrections.
Approach:
  - Embedding model corrected to text-embedding-3-large (3072 dims, ADR-066) in PROJECT.md, CLAUDE.md, llms.md.
  - schemas.md: ActionRole→MatchMode, EndpointRoute 7→9 (added STUDIO, MEDIA_TYPE), added Key Files entries for semantic_translation, semantic_bodies, studio_translation, award_surface_forms, production_brand_surface_forms; fixed implementation/notebooks reference.
  - search_v2.md: Stage 3 list updated to 9 endpoints (added studio + media_type short-circuit note), endpoint contract phrasing generalized.
  - db.md: clarified that the v1 title-token stubs live in lexical_search.py:33-38, not postgres.py.
  - ADR-074 and ADR-075 marked superseded by ADR-076; References sections updated to v3_step_2_rethinking.md and to live step_0/1.py modules.
  - ADR-060: removed claim that implementation/vectorize.py and implementation/scraping/gather_data.py still consume BaseMovie (deleted) — only base_movie_factory test fixture remains.
  - ADR-073: replaced reference to deleted finalized_search_proposal.md with v3_proposed_changes.md.
  - ingestion.md: removed inaccurate SourceOfInspirationWithJustificationsOutput alias claim (deleted classes); fixed `implementation/searching/` → `implementation/scraping/` typo.
  - classes.md: added countries.py and overall_keywords.py to Key Files.
  - CLAUDE.md: dropped non-existent implementation/notebooks/ row; added schemas/ and search_v2/ rows; updated test count to 76; noted V2 search uses Gemini in llms.md.
  - PROJECT.md: added schemas/ and search_v2/ to module map; updated test count to 76; LLM provider line updated for Gemini (V2) and dual gpt-5-mini/gpt-5.4-mini generators.
Design context: docs-auditor report from this session. ADR-066 cited for embedding model.
Testing notes: docs-only changes; no code paths affected. Code-comment references to deleted planning docs (`finalized_search_proposal.md`, `steps_1_2_improving.md`, `step_4_planning.md`) in schemas/ and search_v2/ files were left intact — outside scope of permanent-doc audit and would create wide-fan churn for low signal.

## Step 2 schema + prompt refactor: per-atom evaluative intent
Files: schemas/step_2.py, search_v2/step_2.py, search_improvement_planning/v3_step_2_rethinking.md, docs/modules/{schemas,search_v2}.md, search_v2/test_queries.md (test set expansion done earlier in session)

### Intent
Replace the prior atom-graph schema with a per-criterion evaluative-intent shape. Each atom now carries a unified `modifying_signals` list (surface_phrase + freeform effect string) plus an `evaluative_intent` prose statement. The intent is the load-bearing semantic field downstream consumes; modifying_signals carries provenance.

### Key Decisions
- **Drop `AbsorbedModifierKind` enum → freeform `effect` string per signal.** Running 34 queries surfaced systematic bucket-forcing (e.g. `or` → FLIPS POLARITY in q11). Closed enums with a misfit-hits-closest-bucket failure mode are unrecoverable downstream; freeform is recoverable.
- **Collapse `absorbed_modifiers` + `modified_by` into one `modifying_signals` list.** They were always conceptually the same thing — something-shaping-this-criterion's-meaning. The split was structural noise.
- **Drop `ModificationDepth` (SHALLOW/DEEP) and the nullable convention.** The binary collapsed 5+ distinct relationship shapes (subset filter, context reframe, axis stack, counterfactual transposition, style transfer); nullable-depth never produced a null in 34 queries — small models fill nullable enums.
- **Drop `modifier_atom_index`.** `surface_phrase` carries identity more specifically than a positional pointer, and the LLM systematically dropped forward-pointing edges in multi-atom queries (q11, q15, q24, q32 all showed zero edges).
- **Add `evaluative_intent` per atom.** This is the new bottleneck for downstream quality and the one place where light inference is permitted. Modal vocabulary (SOFTENS / HARDENS / FLIPS POLARITY / CONTRASTS) remains recommended but is no longer enum-enforced.
- **`_INTER_ATOM_MODIFICATION` prompt section replaced wholesale by `_EVALUATIVE_INTENT`.** The mental model shift is from "build a graph of edges between atoms" to "for each criterion, walk the whole query and consolidate everything that shapes its evaluation into a per-atom intent statement." Position in surface order is irrelevant.

### Planning Context
The 34-query test set in search_v2/test_queries.md (which grew earlier in this session to cover 11 gap categories: parallel filters, multi-anchor, use-case scoping, tonal/mood, negation-only, mixed pos/neg, counterfactuals, person-as-credit-vs-style, hedged, dense, loose figurative) was the empirical basis for every shape decision. Each schema change is tied to a concrete failure pattern observed in the prior implementation's outputs.

### Testing Notes
Re-ran the full 34-query test set after the refactor. Materially-broken queries from the prior run (q4, q11, q18, q28, q29, q32, q33, q34) all show meaningful improvements driven by the `evaluative_intent` field doing real semantic work — q28 now flags Anderson's role mismatch in the intent prose; q33/34 now articulate the meta-reception reading; q4's "65" atom intent clarifies it's age context not content. Two known-residual issues: q18 (multi-anchor) regressed slightly — each title gets a literal "identify and return the specific movie" intent rather than a reference reading; q32 produces duplicate `not too` entries on `long`. Both are intent-quality / prompt-tuning issues, not schema-shape issues.

No unit tests modified (test boundaries rule). Any tests importing `AbsorbedModifierKind`, `AbsorbedModifier`, `IncomingModification`, or `ModificationDepth` will fail at import — those types are deleted.

## Step 2 atomicity tightening: searchable-unit test + intent discipline
Files: schemas/step_2.py, search_v2/step_2.py

### Intent
Followup to the per-atom evaluative_intent refactor. Two new failures surfaced when running ad-hoc queries: (1) duplicate-atom emission — for `john wick in space`, the LLM recorded "in space" as a modifying_signal on JW AND emitted "space" as a peer atom even though the user's intent is one coupled unit; (2) intent-doesn't-consolidate — the JW atom's evaluative_intent paraphrased surface_text and ignored the in-space signal. Both point to the atomicity principle being operationalized syntactically rather than as a retrieval decision, plus the intent rule lacking a locally-checkable discipline.

### Key Decisions
- **Atomicity = retrieval decision, not syntactic decision.** Replaced `_ATOMICITY` in the system prompt with a rewrite organized around the searchable-unit test: imagine independent retrieval against each candidate piece + combination; if it lands on the user's intent → distinct atoms (PARALLEL CRITERIA pattern); if it misses → one atom with the dependent material absorbed as modifying_signals (DEEP RESHAPE pattern). Generalized principles, not pattern-by-pattern fixes.
- **Generation discipline made explicit.** Each phrase in the query gets exactly one role: surface_text of an atom, modifying_signal on some atom, or filler. Re-emitting absorbed material as a peer atom double-counts the user's intent.
- **Concept-claim principle.** When a phrase is recorded as a modifying_signal, both the phrase AND the concept it carries (setting, period, medium, style, named referent, mood) are claimed by that signal. The bare concept word does not become a separate atom even if it would be atomizable on its own. Added in both prompt and schema NEVER list after a single-pass run showed transposition queries still split.
- **Intent must reflect signals — operational test.** Sharpened `_EVALUATIVE_INTENT` "Building the intent" subsection with a locally-checkable rule: read each modifying_signal, ask whether the intent would noticeably change if the signal were removed or altered; if no, the intent hasn't consolidated that signal. Mirrored in the field description.
- **`_MODIFIER_VS_TRAIT` light edit.** Added a paragraph linking to ATOMICITY for the case where a content phrase (not a syntactic modifier) gets absorbed because the searchable-unit test demands it. Recording shape unchanged.
- **No new schema types.** The shape already supports the right output (one atom + absorbed signals + integrated intent); the fix is conceptual.

### Testing Notes
Re-ran the full 34-query test set plus the two ad-hoc queries that surfaced the failures (`john wick in space`, `something kinda like black mirror but not british`). Verified:
- JW + space collapses to one atom with "in space" absorbed and intent describing the consolidated unit.
- Black mirror query correctly preserved as two parallel atoms with intents reflecting their respective signals.
- BB + 1800s collapses to one atom in some runs (was two before); succession + pirates and wes anderson does horror still split in some runs — these involve a comparison frame ("like X") competing with a transposition ("but with Y") for the LLM's absorption attention. Run-to-run variance at temp 0.35 produces inconsistent atomization on this specific shape.
- All other queries either improved or held steady.
- q4's "with my mom" + "shes 65" now correctly absorb into one atom (audience-context concept claimed by the signal).

The residual inconsistency on comparison+transposition queries is a small-LLM consistency limit, not a principle gap. The principle is encoded; the LLM applies it probabilistically. No further prompt-tuning attempted in this round to avoid case-by-case prescription.

No unit test changes. No module doc updates needed (schema shape didn't change).

## Step 2 schema description compaction
Files: schemas/step_2.py | Compacted field descriptions on `ModifyingSignal`, `Atom`, and `QueryAnalysis` for token savings (~11% line reduction) without dropping any substantive instruction. Also removed three leftover test-query-flavored examples (`'than fight club'`, `'in the 1800s'`, `'but with pirates'`) from `surface_phrase` per the no-test-queries-as-examples rule. Schema shape and import surface unchanged; verified imports clean.

## v3 query-understanding pipeline rewrite (Steps 2/3/4)
Files: search_improvement_planning/v3_step_2_rethinking.md
Why: Pipeline shape decided through design discussion. Old plan had Stages 3 (reconstruction test) / 4 (literal test) / 5 (trait commitment) as separate calls; new plan collapses to three steps with substantive role changes.
Approach:
  - Step 2 stays one LLM call but gains a second output layer: `traits` (committed search-ready units) alongside `atoms` (analysis layer). Commit phase resolves `candidate_internal_split`, dedupes via new `redundancy_note` field, and assigns role / polarity / salience. Naming distinction (atoms vs traits) is deliberate — prevents the model from treating layer 2 as a copy of layer 1.
  - Step 3 (new) is per-trait parallel LLM calls generating the minimum set of category calls whose combined retrieval captures the trait's intent. Polarity-agnostic. Combines categorization + parametric resolution (same cognitive move). Aggregation: unweighted sum within trait; max-pool within category handled by category, not Step 3. Output schema deferred until Step 2 commit lands.
  - Step 4 reuses existing `search_v2/stage_3/` endpoint generators with light input-adapter revision. Per-(category-call, endpoint) call.
  - Build order: Step 2 commit phase first → Step 3 schema (informed by real traits + existing endpoint generator inputs) → Step 3 prompt + Step 4 fine-tuning together.
  - Doc adds explicit "Outstanding changes needed for Step 2" section listing concrete schema additions (`redundancy_note` on Atom, new `Trait` model, `traits` on `QueryAnalysis`, commit-phase prompt section).
Design context: Multi-turn design discussion this session. Key user calls: polarity committed (not hint), traits not committed_atoms, unweighted sum, defer Step 3 schema, reuse existing stage_3/ for Step 4. Old Stage 3/4/5 framing replaced wholesale.
Testing notes: Doc-only change. No code touched. Implementation work begins with Step 2 commit-phase additions to schemas/step_2.py + search_v2/step_2.py system prompt.

## Step 2 schema: commit-phase shape (atoms + traits)
Files: schemas/step_2.py
Why: Per v3_step_2_rethinking.md "Outstanding changes needed for Step 2" — the analysis layer (atoms) needed dedupe + reasoning fields, and the committed layer (traits) needed to be introduced as a separate Pydantic model so role/polarity/salience commitments are kept structurally distinct from descriptive recording.
Approach:
  - Renamed `Atom.candidate_internal_split` → `Atom.split_note` (parallel naming with the new `redundancy_note`).
  - Both `split_note` and `redundancy_note` now require a brief `because <reason>` clause inline so the commit phase has the LLM's reasoning, not just the structural signal.
  - Added `Atom.redundancy_note: str | None` (backward-looking only — forward redundancies caught in commit phase).
  - Added `Trait` model with `surface_text`, `evaluative_intent`, `role` (Literal carver|qualifier), `polarity` (Literal positive|negative), `salience` (Literal central|supporting).
  - Added `QueryAnalysis.traits: list[Trait]` after `atoms`. Step 3 consumes traits, not atoms.
  - Field descriptions tightened — they're the only docs the LLM gets for the output shape; commit-phase prompt section will live in the system prompt but won't restate field context. Each field's micro-prompt is self-contained: lead with what the field is, follow with operational rules / NEVER lists where downstream depends on discipline.
  - Updated module header comment from "two coupled outputs" → "three coupled outputs" (holistic_read, atoms, traits).
Design context: ADR pending for the commit-phase shape; planning doc at search_improvement_planning/v3_step_2_rethinking.md is canonical until then.
Testing notes: Imports verified clean. System prompt not yet updated — commit-phase section + Atom/Trait NEVER list reconciliation come next. Will then re-run 34-query test set against the new shape.

## Step 2 prompt + schema alignment to commit-phase shape
Files: search_v2/step_2.py, schemas/step_2.py

### Intent
The schema rewrite landed atoms (descriptive) + traits (committed) but the system prompt still described two outputs and framed carver/qualifier/polarity/salience as "background context for downstream stages." This pass aligns the prompt with the new shape and applies the LLM-handling principles surfaced in this session: generalized guidance over pattern-listing, evidence-then-decide separation, mechanical token-mapping at commit time, and locally-checkable operational tests. Smoke tests confirm all four commit-phase mechanics fire as designed.

### Key Decisions
- **Workflow framing replaces "applied vs background" framing.** Module header + section-ordering comment + `_TASK_FRAMING` now walk the prompt as ATOM PHASE (atomicity → modifier vs atom → evaluative intent) → COMMIT PHASE (commit phase wrapper → carver vs qualifier → polarity → salience) → CATEGORY VOCABULARY. The LLM sees the work as gather-evidence-then-commit-buckets rather than principles + miscellaneous downstream context.
- **New `_COMMIT_PHASE` section.** Walks atoms→traits: resolve splits via re-running the searchable-unit test, resolve redundancies (backward and forward), don't drop / don't invent, per-trait role/polarity/salience as evidence-reads, three locally-checkable operational tests, trait ordering rule. The "reuse work the prior phase did" principle is the section's load-bearing mental model — commitments are mechanical reads off `effect` tokens already on the source atom, not fresh interpretations.
- **`_MODIFIER_VS_TRAIT` → `_MODIFIER_VS_ATOM`.** The section's "trait" always meant atom-level criterion; the formal `Trait` is now reserved for the committed layer. Body unchanged substantively, just the naming-clash fix.
- **`_CARVER_VS_QUALIFIER` / `_POLARITY` / `_SALIENCE` reframed as commit-phase commitments.** Each opens with "Use this to commit `Trait.X`" + "read source atom's evidence." Polarity and salience are now mechanical token-reads (effect contains FLIPS POLARITY → negative; effect contains SOFTENS → supporting). Distribution-scope subsection dropped from `_POLARITY` (atom phase already distributes via signal recording). Priority-ordered signals list (1–6) dropped from `_SALIENCE` (atom phase already encodes prioritization in which effect token gets recorded). Examples blocks trimmed throughout per the generalized-over-pattern-listing principle — pattern templates risk the LLM fitting queries to enumerated cases rather than applying the principle.
- **`_ATOMICITY` adds an UNCERTAINTY-MARKING addendum** instructing the LLM to populate `split_note` / `redundancy_note` with brief reasoning when the searchable-unit test doesn't settle the call, rather than guessing. Substantively the section is unchanged.
- **Category taxonomy trimmed to vocabulary view.** New `_build_category_vocabulary_section()` renders name + description + good_examples only; boundary / edge_cases / bad_examples (Step 3 fitting machinery) removed. Header rewritten to make explicit that Step 2 recognizes "has-a-home" but does not pick categories. Token impact: ~50% reduction on the taxonomy section. Total prompt size dropped from ~50K chars to ~32K chars.
- **Schema description tweaks.** `Trait.role` reframed to lead with "read from source atom's evaluative_intent shape" + the filter-vs-downrank operational test, dropping example category enumerations. `Trait.polarity` and `Trait.salience` tightened to mechanical-rule framing, dropping the modal-list enumerations now redundant with the prompt sections. `Atom.evaluative_intent` NEVER list updated from "POLARITY/SALIENCE NUMBERS" → "POLARITY/SALIENCE VALUES" (trait commitments are Literals, not numbers).

### Principles Applied
1. Generalized guidance over pattern-listing (trimmed Examples blocks; dropped priority-ordered signal list and distribution-scope subsection).
2. Evidence-then-decide (atom phase = freeform evidence; commit phase = bucketed commitments from gathered evidence).
3. Reuse prior-phase work (commit phase reads `effect` tokens already on `ModifyingSignal`, doesn't re-interpret).
4. Operational tests must be locally checkable (each commitment has a one-question test at point of writing).
5. No test queries as examples (no overlap with /tmp/run_all_queries.sh queries).
6. Atoms describe; traits commit (reinforced wherever the model is tempted to cross the line).
7. Schema = micro-prompts; prompt = procedural (Trait field descriptions don't restate the prompt; prompt sections reference Trait fields directly).

### Testing Notes
Smoke tests verify both layers populate and commit-phase token-mapping fires:
- "scary" → 1 atom + 1 trait (clean carry-over, role=carver, polarity=positive, salience=central).
- "ideally a slow burn thriller, nothing too gory" → 2 atoms + 2 traits with: thriller atom's "ideally" SOFTENS effect → trait salience=supporting; gory atom's "nothing too" FLIPS POLARITY effect → trait polarity=negative + salience=supporting; carver/qualifier role assignment matches intent shape.

Token reduction is meaningful: prompt size dropped from ~50K → ~32K chars; smoke-test input tokens dropped to ~7.7K (down from ~10-12K range with the full taxonomy).

Full 34-query re-run still pending; flag any deep-reshape regressions, role mis-assignments on edge cases, or polarity / salience commits that don't match the source atom's effect tokens. No unit-test changes.

## Step 2 refinements: discipline gates + relevance_to_query reasoning + holistic salience
Files: schemas/step_2.py, search_v2/step_2.py, search_improvement_planning/v3_step_2_rethinking.md

### Intent
Round-2 fixes from the 34-query test set analysis. Three search-quality regressions identified: (1) coupled-pair atomization producing non-self-contained second traits (comedians taking on serious roles, Q26 BB+1800s, Q27 succession+pirates, Q29 wes anderson does horror); (2) redundancy_note misused for "subsumption / vibe-overlap" rather than its tighter "double-counting of a query span" meaning, dropping a real friday-night criterion in Q32; (3) SOFTENS → role mis-mapping plus the "carvers don't get salience" rule forcing role flips that didn't match user intent in Q31 / Q32 / Q30.

### Key Decisions
- **`Atom.split_note` and `Atom.redundancy_note` to non-null required `str`.** Discipline gates, not opt-in flags. Always populated with reasoning ("not split because <reason>" / "not redundant because <reason>" for negative cases). Forces the model to actually run the searchable-unit and absorption tests on every atom.
- **Tighten the redundancy definition.** Lead the field description with a NEVER list against subsumption ("conjunction of others implies this"), vibe overlap, and semantic similarity. Redundancy is specifically the double-counting of a query span: this atom's content has been recorded as a modifying_signal on another atom, and emitting it separately would have downstream score it twice. Cross-references "absorbed as modifying_signal" explicitly so the gate ties to the atomicity / concept-claim discipline.
- **Add `Trait.relevance_to_query: str` field before `salience`.** Reasoning step that walks through how the trait sits in the query holistically — modifiers, position, investment, load-bearing-ness. 1-2 sentences. Salience commits as the natural conclusion of this reasoning. Replaces the strict mechanical SOFTENS-only-influences-salience rule with locally-checkable holistic interpretation per principle 16 (each commitment gets its own local test).
- **Lift "carvers don't get salience" rule.** Salience now applies to all traits regardless of role. A non-central carver acts as a lenient filter — the trait still defines its own pool but with softer boundaries. Downstream code reads salience and adjusts (handled programmatically post-Step-2; not a Step 3 LLM input).
- **Modification definition does not need a separate update.** Atomicity + concept-claim already define modification ("when a content phrase reshapes another atom this deeply, it absorbs as a signal — it does NOT also appear as a separate atom"). The tightened redundancy definition with explicit cross-reference to "absorbed as modifying_signal" carries the modification-discipline work at the gate level.
- **Prompt updates:** `_ATOMICITY` UNCERTAINTY-MARKING → SPLIT AND REDUNDANCY GATES (always-populated framing); `_COMMIT_PHASE` RESOLVE REDUNDANCIES rewrite (verify claims before merging) + PER-TRAIT COMMITMENTS update (relevance_to_query before salience) + OPERATIONAL TESTS (salience test reads relevance_to_query reasoning); `_SALIENCE` rewrite (holistic-reasoning framing, modal tokens are one signal among several); `_TASK_FRAMING` small adjustment (role/polarity mechanical, salience via reasoning).

### Testing Notes
Re-ran full 34-query test set. Verified:
- Every atom has populated `split_note` and `redundancy_note` with reasoning (including "not split because..." / "not redundant because..." for negatives).
- Every trait has populated `relevance_to_query` with holistic reasoning.
- Q32 friday night NO LONGER incorrectly marked redundant — survives as a viewing-occasion qualifier. ✓
- Q31 (`preferably under 2 hours`) commits as carver / supporting; relevance_to_query explicitly notes the "preferably" hedge softens the requirement. ✓
- Q32 `long` (`not too long`) commits as carver / negative / supporting via "Not too X" handling. ✓
- Q9 wonder woman, Q10 joker work correctly under set-intersection (positive carver + negative carver).
- Q6 parody of the godfather: now atomizes to "the godfather" with "parody of" absorbed as signal; trait role = qualifier (reference), polarity = positive. Better than prior "parody" surface_text reading.

Coupled-pair atomization (comedians taking on serious roles, Q26 BB+1800s, Q27 succession+pirates, Q29 wes anderson + horror) still produces two atoms each. The redundancy gate now fires (visible reasoning in redundancy_note), but the model consistently rationalizes "not redundant because while it modifies the first atom, it also defines a distinct searchable population." This may be defensible under set-intersection semantics for some shapes (BB-style films ∩ 1800s-set films) but questionable for shapes where the second piece doesn't retrieve as a meaningful standalone population (comedians + serious roles). Iteration target — the discipline gates surface the rationalization for inspection but don't fully eliminate it. Not a regression vs prior behavior.

Token impact: input ~8180 chars (up from ~7755 due to expanded redundancy_note description and atomicity gate section); output up notably for many-atom queries (Q32: 968 → 1473 tokens) reflecting the always-populated reasoning fields. Reasonable.

No unit-test changes.

## Step 3 prototype: trait → category-call decomposition
Files: schemas/step_3.py (new), search_v2/step_3.py (new), search_v2/run_step_3.py (new), search_improvement_planning/v3_step_2_rethinking.md

### Intent
First runnable prototype of Step 3 — the abstraction-flip stage that turns a Step 2 trait into the minimum additive set of taxonomy-routed category calls Step 4 will build endpoint queries from. Per-trait LLM call, fanned out in parallel by the runner.

### Key Decisions
- **Two-layer schema mirrors atoms→traits.** `TraitDecomposition` carries an analysis layer (`target_population`, `dimensions`, `coverage_audit`) and a commitment layer (`category_calls`). The dimension inventory must precede category routing — same "exploration before decision" pattern Step 2 uses for split / standalone.
- **Dimensions are the smallest unit of searchability.** Concrete database-vocabulary pieces (a release-date value, tonal expression as the database captures it, a runtime range, a person credit). NEVER list forbids abstraction-up, category-naming, absence framing, and bundling.
- **`CategoryCall.category` is `CategoryName` (closed enum).** Auto-propagates when the taxonomy adds members; Pydantic v2 + Gemini structured-output constrain the LLM to enum values. Prompt renders each entry keyed by `cat.value` (the string the LLM emits) with `cat.name` shown for log readability.
- **Step 3 is polarity-agnostic.** Even when `trait.polarity == "negative"`, every call describes presence of the attribute. Polarity flips at merge time. The prompt's `_MINIMUM_SET_AND_POLARITY` section spells out why double-flipping would break the merge contract.
- **Additive composition only.** Unweighted sum across calls; no per-call weighting, no cross-call interaction model. If calls don't add up to the trait, decomposition is wrong.
- **Minimum-set discipline.** Most traits → 1 call; parametric traits → a few. Padding dilutes the trait's score sum relative to peers.
- **Full taxonomy detail (not the trimmed Step 2 view).** New `_build_full_category_taxonomy_section()` renders every category's description, **boundary** (what it does NOT cover, with redirects), edge_cases, good_examples, and bad_examples — the full disambiguation machinery routing decisions need. Step 2's taxonomy section is intentionally trimmed because its job is recognition only; Step 3's job is fitting.
- **LLM-input contract per trait:** surface_text, evaluative_intent, role, polarity (informational only), and `relevance_to_query` (signals decomposition aggressiveness — central traits earn fuller decomposition). `salience` is NOT shown — code-path only. Atom layer + sibling traits NOT shown — defer until eval shows traits stepping on each other.
- **Same model as Step 2.** Gemini 3 Flash, thinking disabled, temperature 0.35. Reproducibility wins; provider/model hard-coded in the run function.
- **Runner shape:** `run_step_3.py` runs Step 2 first, then `asyncio.gather` over traits calling `run_step_3(trait, holistic_read)`. Per-trait elapsed time and token usage printed alongside each decomposition; wall-clock for the fan-out reported as max-of-parallel-calls.

### Planning Context
Design discussion this session over multiple turns. Plan file: `~/.claude/plans/open-items-adaptive-turing.md`. v3_step_2_rethinking.md design choice #9 updated to include `relevance_to_query` in Step 3's LLM-input contract (committed earlier this turn).

### Testing Notes
Imports verified clean (`schemas.step_3`, `search_v2.step_3`, `search_v2.run_step_3` all import). System prompt size ~45.6K chars (expected — full taxonomy with boundary/edge_cases/bad_examples is the load-bearing disambiguation machinery for routing).

Smoke-test queries from the plan to run end-to-end:
- `python -m search_v2.run_step_3` (default sample query)
- `"John Wick but with kids, not too long"` — split exploration / commit-phase couplings.
- `"warm hug movie like Paddington"` — parametric figurative + comparison anchor decomposition.
- `"wes anderson does horror"` — out-of-context creator (multiple categories from one trait).

Manual checks per query: (1) concrete trait → 1 dimension + 1 call to expected category; (2) parametric trait → multiple dimensions covering tone/register/pacing-cluster with audit naming each; (3) `coverage_audit` references every dimension and `category_calls` corresponds 1:1 to categories named in the audit; (4) negative-polarity traits still emit presence-of-attribute calls.

If routing is shaky, iteration target is the taxonomy rendering / boundary prose, not the schema shape. No unit-test changes.

## Step 2 round 4: exploration-only gates (no embedded verdicts) + standalone_check semantics
Files: schemas/step_2.py, search_v2/step_2.py, search_improvement_planning/v3_step_2_rethinking.md

### Intent
Audit of round-3 redundancy_note rationalizations across the 34-query set surfaced a structural bias: the verdict-laden format ("not redundant because X" / "redundant given X because Y") biased the model toward committing the verdict first and rationalizing after. Almost every output started with "not redundant" and the "because" tail was post-hoc. This round restructures both gate fields as pure evidence-gathering exploration with no embedded verdicts; the commit phase reads the explorations and makes the structural calls. Also reframes the redundancy concept entirely: the test isn't "absorbed as signal" (structural) but "how does standalone retrieval relate to user-articulated intent" (semantic, anchored on holistic_read).

### Key Decisions
- **Rename `Atom.split_note` → `Atom.split_exploration`.** Pure evidence-gathering exploration. Walk through plausible subdivisions and what each would retrieve; whether the combined retrieval would capture user intent at this atom's granularity. NO "split" / "keep whole" verdict in the field.
- **Rename `Atom.redundancy_note` → `Atom.standalone_check`.** Reframed test: compare the atom's evaluative_intent against the holistic_read; describe HOW (not if) standalone retrieval relates to user-articulated intent. Walk through what population standalone retrieval returns, whether it matches a user-articulated standalone-able criterion or shifts the meaning, whether context the atom integrates from another atom survives standalone or falls away. NO "redundant" / "not redundant" verdict.
- **Standalone_check NEVER list targets the specific rationalization patterns observed.** "WRITE A VERDICT" closes verdict-first commitment; "SHORT-CIRCUIT WITH UNIQUENESS CHECKS" closes the Pattern A/B dismissals (primary subject / first mention / no other atom captures); "APPEAL TO INDEPENDENT RETRIEVABILITY AS A VIRTUE" closes the Pattern C dismissal; "USE 'WHILE [COUPLING ACKNOWLEDGED] BUT [STANDALONE VALUE]' PATTERNS" closes the Pattern D exit clause that appeared in 5 coupled-pair queries.
- **Commit phase becomes interpretive, not verifying.** `_COMMIT_PHASE` "RESOLVE SPLITS" / "RESOLVE REDUNDANCIES" → "ACT ON SPLIT EXPLORATIONS" / "ACT ON STANDALONE CHECKS". Commit phase reads the analyses and applies the searchable-unit and user-intent-comparison tests itself; it doesn't verify a structured claim. The structural decision (split / merge / keep) lives at commit time, not atom time. Reuses the work from the atom phase rather than re-deriving.
- **Concept-claim at merge made explicit.** Per principle 6, the commit-phase merge framing now states: "the merged trait absorbs both sources fully. Neither survives separately. The host's surface_text and evaluative_intent stand; the coupled atom's content is integrated via the host's modifying_signals already." Closes the "but the second atom also names a distinct population" loophole at the decision point.
- **Drop the "absorbed as modifying_signal" test entirely** in the schema description. That test misframed the question (Q9, Q10 had no signal absorption but the user confirmed they're correct; Q29 had signal absorption but the user confirmed it should still emit two atoms). The actual test is meaning-fidelity vs user intent, not structural-absorption-of-content.
- **Cross-cutting design choice #8 reframed.** "Discipline gates over opt-in flags" → "Exploration before decision (no embedded verdicts)". The principle is now: where a commitment depends on judgment, surface analysis as its own field with no embedded verdict; the decision is made at a separate point. Applies uniformly to salience (relevance_to_query → salience), splits (split_exploration → commit-phase split), couplings (standalone_check → commit-phase merge). #9 (separate principle about reasoning-before-commitment) was merged into #8 since they're the same idea applied at different scopes.
- **Q29 wes anderson does horror** is reframed as a role-assignment failure (wes anderson should commit role=qualifier as style reference, but commits carver), not a coupling failure. Separate fix on the role rule, deferred.

### Principles Applied
- **5 (locally-checkable operational tests):** standalone_check anchors on the holistic_read — the model can locally compare its own evaluative_intent's standalone meaning to the user's articulated phrasing.
- **6 (concept-claim explicit):** commit-phase merge framing explicitly states "both sources absorbed, neither survives separately."
- **7 (principles over patterns):** the standalone_check definition is generalized (HOW does standalone retrieval relate to user intent) rather than pattern-by-pattern.
- **9 (schema = micro-prompts; prompt = procedural):** field descriptions are self-contained including NEVER lists; prompt sections describe procedural workflow without duplicating field-shape rules.
- **11 (lead with NEVER):** standalone_check description leads with the four-item NEVER list closing the rationalization shapes observed in round 3.
- **13 (description vs interpretation):** atoms describe (with explorations also being descriptive analysis) and traits commit. Decision-making moves to the commit phase entirely.
- **15 (reuse prior-phase work):** commit phase reads explorations rather than re-running the searchable-unit / user-intent tests from scratch.

### Testing Notes
Schema imports clean. System prompt size: 35,917 chars (slight increase from expanded NEVER list and exploration framing). Smoke test + 34-query re-run not yet executed. Watch for:
- Standalone_check entries that engage the test (describe deviation/match against holistic_read) rather than dismissing with the verdict-first rationalizations from round 3.
- Commit phase merging on comedians, Q26, Q27 where standalone_check describes meaning-shift.
- Commit phase keeping both atoms on Q9 (wonder woman + new ones), Q10 (joker + phoenix), Q11 (superhero + marvel/dc) where standalone_check describes alignment with user intent.
- Per-atom output token cost: explorations are likely longer than the prior verdict-laden notes since dismissal patterns are forbidden; expect output tokens up.

No unit-test changes.

## Step 2 + Step 3 iteration: identity-vs-attribute, category-aware decomposition, contextualized phrase
Files: schemas/step_2.py, search_v2/step_2.py, search_v2/step_3.py, search_v2/run_step_3.py

### Intent
Follow-up to the prior Step 3 iteration (qualifier_relation/anchor_reference fields, multi-expression calls, per-dimension candidates). The 34-query eval against /tmp/step3_runs_v2/ surfaced four residual issues, all routing/decomposition discipline rather than schema shape:
1. Qualifier traits still routed to identity categories (q06 godfather emitted TITLE_TEXT despite trait_role_analysis saying "this trait IS the reference being satirized"). Step 3's category commitment is gospel — there's no recovery downstream — so prose-rescued differentiation in retrieval_intent doesn't save it.
2. Carver-negative multi-dim traits (q10 phoenix-Joker, q15 hallmark) over-exclude under the orchestrator's default additive scoring; they need intersection-of-calls semantics.
3. Bare surface_text invites shortcut routing — the model latches on "the godfather" before reading qualifier_relation.
4. Decomposition depth wasn't category-aware — q11's "DC" came close to per-character dimension explosion.

### Key Decisions
- **`Trait.contextualized_phrase: str` added** at the bottom of the Trait class (after salience). Step 2 emits a single short phrase that restates the trait with anchor_reference + meaning-shaping signals folded in. Step 3 reads this as the headline trait identity ahead of surface_text. Faithful restatement, no decomposition / parametric expansion / added or dropped details. Carver traits with no relevant modifier copy surface_text. Schema field description carries the construction discipline; Step 2's `_COMMIT_PHASE` adds a short bullet + operational test ("if I read this aloud out of context, can a fresh reader recover the trait?").
- **Identity-vs-attribute paragraph in Step 3's `_TRAIT_ROLE_ANALYSIS`.** For carver traits both kinds are fair game; for qualifier traits the named entity is a positioning anchor, so identity categories are off-limits — route only to attribute categories that describe what the entity is LIKE. This is a generalized structural rule, not a per-qualifier_relation patch — it follows directly from what "qualifier" means. The 9 identity-flavored categories ("Person credit", "Title text lookup", "Named character", "Studio / brand", "Franchise / universe lineage", "Character-franchise", "Adaptation source flag", "Below-the-line creator", "Named source creator") are listed inline in the prompt rather than maintained as a separate Python constant — no Python code does programmatic enforcement, so a runtime data structure was overkill for what's purely prompt-rendering.
- **Category-aware decomposition in `_DIMENSION_INVENTORY`.** Replaced the "concrete = 1 dim, parametric = several" heuristic with: decompose only as deep as the existing categories require. If one category captures the trait, that's ONE dimension and parametric expansion lives in the call's expressions list. Decompose into multiple facets only when no single category covers the trait. Operational test: "could the items I'm considering route to DIFFERENT categories?"
- **CLEAN-FIT TEST in `_CATEGORY_ROUTING`.** When a dimension's candidates list contains an entry with `what_this_misses="nothing"`, commit only that one. The other candidates were adjacency context surfaced for honesty, not parallel routes. Stops adjacent-category leaks (q34 hidden-gem FINANCIAL_SCALE leak).
- **Carver-negative intersection is purely an orchestrator concern; Step 3 does not dispatch on it.** The orchestrator reads role+polarity directly off the Step 2 Trait and chooses how to compose Step 3's calls (additive sum for positive traits / qualifiers, intersection over calls for carver+negative exclusions). Step 3 always describes presence of attributes; an earlier draft of this iteration prepended an INTERSECTION-MODE preface to the per-trait user prompt, but it was removed — the orchestrator can intersect without Step 3 reciting any signal, and Step 3's normal decomposition already produces co-holding calls (each describing a facet of the same population).
- **`_build_user_prompt` headline change.** `contextualized_phrase` becomes the first trait line; `surface_text` is demoted to a verbatim grounding line below it. Bare surface phrases stripped of query context invited shortcut routing on q06 (TITLE_TEXT for "the godfather") — the contextualized phrase makes the qualifier framing visible at the top.
- **Schema unchanged on Step 3.** No new TraitDecomposition fields; the intersection-mode commitment lives in trait_role_analysis prose, where the orchestrator reads it alongside the structured role+polarity it already has.

### Principles Applied
- **Step 3's category choice is gospel.** Stopped treating retrieval_intent prose as load-bearing for differentiating "retrieve this" from "use as positioning anchor"; the category itself must be right.
- **Generalized rules over edge-case lists.** Identity-vs-attribute is one principle covering parody / comparison / style / transposition cases (and any future qualifier_relation values), not four enumerated rules.
- **Programmatic dispatch where conditional sections only apply to a fraction of inputs.** Intersection mode dispatched in `_build_user_prompt` rather than a conditional in the system prompt.
- **Source-of-truth coupling.** IDENTITY_CATEGORIES set rendered dynamically into the prompt; no risk of drift between the prompt text and the classification.

### Testing Notes
Smoke tests on 6 priority queries verified (outputs in /tmp/step3_runs_v3/):
- **q06 "parody of the godfather"**: contextualized_phrase = "parody of the godfather"; godfather trait emits no TITLE_TEXT call — routes to story-thematic archetype + narrative setting + character archetype + emotional-experiential. RESOLVED.
- **q08 "darker than fight club but funnier than seven"**: both traits route to emotional-experiential only, no TITLE_TEXT for either film. RESOLVED.
- **q10 "joker but not the joaquin phoenix one"**: trait 2 (carver+negative) trait_role_analysis explicitly commits "intersection-exclusion mode; all calls must co-hold for the exclusion to apply to the specific 2019 Joaquin Phoenix film". Calls: Person:Phoenix + Character:Joker for the orchestrator to intersect. RESOLVED.
- **q21 "warm hug movie"**: single emotional-experiential call with 3 expressions. NO REGRESSION.
- **q29 "wes anderson does horror"**: contextualized_phrase = "wes anderson's directorial style applied to horror"; wes anderson trait emits no PERSON_CREDIT — routes to visual-craft acclaim + emotional-experiential + narrative devices. RESOLVED.
- **q34 "hidden gem"**: still emits FINANCIAL_SCALE alongside CULTURAL_STATUS + GENERAL_APPEAL. Per prior user direction, deferred — not a load-bearing fix for this iteration.

Full 34-query re-run + regression checks (q05, q12, q16, q17 concrete; q24, q25, q32 negative-polarity-presence) deferred to next session. Step 3 system prompt grew from ~52K to ~56K chars; Step 2 prompt grew from ~36K to ~38K chars — modest. No unit-test changes.

## Step 2 + Step 3 schema/prompt compaction
Files: schemas/step_2.py, schemas/step_3.py, search_v2/step_2.py, search_v2/step_3.py

### Intent
Prompt-bloat trim across all Step 2 / Step 3 LLM-facing surfaces — schema field descriptions and system-prompt sections — without dropping load-bearing content. Audited against the 16 schema/prompt design principles surfaced in earlier iterations. Per-file line trims: step_2 schema 568→482 (15%), step_3 schema 430→358 (17%), step_2 prompt 820→688 (16%), step_3 prompt 698→618 (11%). All operational tests, NEVER lists, "n/a" sentinel rules, mechanical token-mappings, and exploration→commit phrasing preserved.

### Key Decisions
- **Principle 9 violation fixed.** `_COMMIT_PHASE` in search_v2/step_2.py used to enumerate per-trait field-shape rules (role / polarity / relevance_to_query / salience / qualifier_relation / anchor_reference / contextualized_phrase) that the schema field descriptions already cover. Replaced with procedural framing only ("These commitments are mechanical reads off the source atom; see schema and the dedicated sections for each"). Removes ~50 lines of duplication and reinforces "schema = micro-prompts; prompt = procedural."
- **Stale reference fix.** `QueryAnalysis.traits` field description in schemas/step_2.py had a 7-step construction list referencing `split_note` / `redundancy_note` (the field names from rounds 2-3, renamed to `split_exploration` / `standalone_check` in round 4). Replaced with a one-line pointer to the system-prompt commit-phase section.
- **Trim targets.** Schema headers (design-principles preamble compressed); ModifyingSignal.effect (example flavors trimmed); Atom.standalone_check / split_exploration (meta-commentary cut, NEVER lists kept verbatim); Trait.qualifier_relation / anchor_reference / contextualized_phrase (example lists pruned to 3-4 representative each); QueryAnalysis.* (DO/NEVER lists kept, surrounding prose tightened); _ATOMICITY (pitfall and exploration framing tightened); _MODIFIER_VS_ATOM (redundant Examples block dropped); _EVALUATIVE_INTENT, _CARVER_VS_QUALIFIER, _POLARITY, _SALIENCE (procedural framing tightened); _TRAIT_ROLE_ANALYSIS (role/relation enumeration kept; identity-vs-attribute principle and examples preserved verbatim); _DIMENSION_INVENTORY / _PER_DIMENSION_CANDIDATES / _CATEGORY_ROUTING / _MINIMUM_SET_AND_POLARITY (procedural prose tightened, all operational tests preserved).
- **No principle-2 change.** `qualifier_relation` and `anchor_reference` retain literal `"n/a"` sentinel rather than `Optional[str]` — the explicit-string design forces commitment vs. silent skip and was deliberate.
- **Existing examples retained verbatim.** Cultural references (Godfather, Fight Club, Wes Anderson, Tom Hanks, Inception, Hitchcock, Marvel, DC, Stephen King) carried over from prior versions; flagged for the user to audit against eval set per principle 10.

### Principles Applied (vs. checked clean)
- 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16: clean — no fixes needed.
- 8: applied (compaction is the goal).
- 9: violation found and fixed (commit-phase / schema duplication).
- 10: flagged but not changed without eval-set diff.

### Testing Notes
All four files parse cleanly (`ast.parse`). No code paths or import surface changed; field names, types, ordering, and Pydantic constraints unchanged. Smoke-test + 34-query re-run not yet executed — recommend running before treating as final since temperature 0.35 + small-model variance can surface phrasing-sensitivity even when content is preserved.

## holistic_read → intent_exploration: three experiments + reusable batch runner
Files: schemas/step_2.py, schemas/step_3.py, search_v2/step_2.py, search_v2/step_3.py, search_v2/run_step_3.py, search_v2/run_test_queries.py (NEW), search_improvement_planning/steps_2_3_experimentation.md

### Intent
Audit of v7's `holistic_read` outputs across 42 test queries surfaced three failure modes: ~26/42 pure restatement, ~12/42 operator-ese mislabeling (CONTRASTS / FLIPS POLARITY etc. applied wrongly to transpositions / parallel comparisons / qualifications), and 0/42 cases where downstream load-bearingly needed the prose. Three experiments to find the right replacement, with v10 (Experiment 7) shipped as the new baseline.

### Key Decisions
- **Experiment 5 (rejected) — remove holistic_read entirely.** Tested the hypothesis that the field carried no value. Result: net regression. 4 v8 regressions traced to lost shape-(c) peer-gate reasoning (q03 sad endings, q10 joaquin phoenix, q32 depressing, q36 feudal japan). Lesson: holistic_read's value was the **act of perceiving the query at the query level before atom commits**, not the prose itself. Reverted.
- **Experiment 6 — replace with `intent_exploration` (exploratory framing).** New field surfaces plausible high-level intents in concrete terms and weighs which is more likely. Description deliberately avoids cue examples, concrete-prose examples, system-label blocklists, and effect-token ban — generic principles only. Prose quality materially better than v7 (q29 cited "as he hasn't directed a pure horror feature"; q36 wrote "satisfaction requires both"). But role_evidence was treating it as one-of-three evidence sources and abstract shape (a)/(b)/(c) reasoning often pre-empted; q31/q32/q34 still regressed.
- **Experiment 7 — promote intent_exploration to PRIMARY source.** Three coupled changes: (1) `_ATOMICITY` opens with intent_exploration as the partition frame; (2) `Trait.role_evidence` schema + `_CARVER_VS_QUALIFIER` PROCESS section reframe intent_exploration as primary, qualifier_relation/peer atoms/traits as contextual grounding; the three evidence shapes are HOW you reason within the primary frame, not free-standing tests; (3) removed intent_exploration from Step 3's user prompt — Step 3 receives ONLY per-trait commits.
- **Result: clean win.** All four v9 regressions resolved; sole-trait carver-of-last-resort emerged organically (q21 warm hug, q34 hidden gem now cite "sole structural anchor of the query" in role_evidence and commit carver — the deferred Experiment 4 code-side post-process is no longer needed); q08 atomization improved (films become atoms, comparison operators become signals); Step 3 routing decoupled cleanly with no leakage failures.
- **One residual flagged, deferred per user.** q18 inception/interstellar/tenet has internal inconsistency — intent_exploration weighs "movies like these" as more likely, role_evidence cites "boundaries of the desired population", role commits carver, Step 3 routes to attribute categories. Three pieces don't align. User: "Not too worried about that issue, it seems transient (an inevitable part of the LLM process)."
- **`search_v2/run_test_queries.py` (NEW) — reusable batch runner.** Replaces ad-hoc bash loops. Loads queries from `search_v2/test_queries.md`, runs Step 2 + Step 3 end-to-end with asyncio.Semaphore-bounded concurrency (default 5), each task writing its own StringIO buffer to its own file (no shared-stdout race). `--out` and `--concurrency` configurable. Cuts batch wallclock from ~6 minutes (sequential bash) to ~90 seconds.

### Principles Applied
- **Information value ≠ thinking-process value.** A schema field can be inert downstream but load-bearing for upstream reasoning quality. holistic_read failed the first test; passed the second.
- **Primary-source vs. parallel-evidence framings drive different LLM behavior.** Same field listed as one-of-three vs. named as primary produces very different downstream behavior. Procedural framing in the prompt determines whether the model uses the field load-bearingly.
- **Per-trait commits should fully encode what downstream needs.** Step 3 lost nothing by losing intent_exploration in its prompt; the per-trait fields (role, qualifier_relation, contextualized_phrase, evaluative_intent, polarity, relevance_to_query) were already the right level of abstraction.
- **Field descriptions: definitions over examples.** Description rewrites in this session removed cue examples, concrete-prose examples, and system-label blocklists per the user's directive to use clear definitions instead.

### Testing Notes
v10 outputs in `/tmp/step3_runs_v10/`. Compared against v7 (canonical, `/tmp/step3_runs_v7/`) and v9 (Experiment 6 baseline, `/tmp/step3_runs_v9/`). 14/42 differ from v7; 12/42 differ from v9. All differences either improvements (sole-trait recoveries, cleaner atomization, structural-anchor reasoning visible in role_evidence prose) or sampling-band drift; one residual issue documented. No unit-test changes. Run with: `python -m search_v2.run_test_queries --out /tmp/step3_runs_v11`.

## media_type endpoint: wrapper + executor + dispatch landed (steps 1–4 of 6)
Files: schemas/media_type_translation.py (NEW), search_v2/stage_3/media_type_query_execution.py (NEW), db/postgres.py, search_v2/stage_3/category_handlers/endpoint_registry.py, search_v2/stage_3/endpoint_executors.py, search_v2/stage_3/category_handlers/handler.py, docs/modules/search_v2.md
Why: `EndpointRoute.MEDIA_TYPE` and the `MEDIA_TYPE` trait category were already in place but the route was a placeholder mapped to `None` in ROUTE_TO_WRAPPER, so any category routing to it would fail. This change adds the wrapper, executor, Postgres helper, and dispatch glue. Steps 5–6 (handler `.md` prompt + removing `MEDIA_TYPE` from `_ENDPOINT_PROMPTLESS` + deleting the handler short-circuit) intentionally deferred.
Important caveat — NOT yet end-to-end executable: the new executor is currently UNREACHABLE through the standard handler path. `category_handlers/handler.py` short-circuits `CategoryName.MEDIA_TYPE` to an empty `HandlerResult` before the LLM codepath runs (its comment is updated to reflect the new state), and removing the short-circuit is blocked on the missing handler prompt because the MEDIA_TYPE category routes only to the MEDIA_TYPE endpoint, which is still in `prompt_builder._ENDPOINT_PROMPTLESS` — letting the call fall through would trip the "no LLM-wrapper endpoints" raise in `build_system_prompt`. The wrapper, executor, and dispatch are correct in isolation (smoke-tested via REPL); they only become reachable once step 5 (author `prompts/endpoints/media_type.md`) and step 6 (drop from `_ENDPOINT_PROMPTLESS` and delete the handler short-circuit) land.
Approach:
  - `MediaTypeQuerySpec` mirrors `StudioQuerySpec`'s closed-enum pattern: `thinking` + `formats` (a `conlist` over a `Literal` subset of `ReleaseFormat` excluding `UNKNOWN`). Closed-enum input avoids a fragile string-normalization layer at parse time and produces a clean JSON schema for OpenAI structured output.
  - `MediaTypeEndpointParameters` declares the canonical `match_mode → parameters → polarity` field order so the base class's `__pydantic_init_subclass__` validator passes. No description on `parameters` — the field is a single nested model and the simple `formats` enum is self-documenting.
  - `execute_media_type_query` is single-path: maps `ReleaseFormat` → `release_format_id` ints, calls a new `fetch_movie_ids_by_release_format` Postgres helper, returns flat 1.0 scores via the shared `build_endpoint_result` helper. Mirrors studio's brand-path scoring rationale (no prominence signal on a movie's media type).
  - New helper `fetch_movie_ids_by_release_format` placed alongside the other `fetch_movie_ids_by_*` helpers in `db/postgres.py`. SQL is a `WHERE release_format = ANY(%s::smallint[])` against `public.movie_card`. No index on the column; deferred unless this becomes a hot path (a partial index `WHERE release_format <> 1` is the obvious tuning).
  - Registry change in `endpoint_registry.py`: import the wrapper, map `EndpointRoute.MEDIA_TYPE` to `MediaTypeEndpointParameters`, and rewrite the comment block since TRENDING is now the only `None` entry.
  - Dispatch change in `endpoint_executors.py`: import `execute_media_type_query` and add a branch for `EndpointRoute.MEDIA_TYPE` between AWARDS and SEMANTIC. The reverse-map `_WRAPPER_TO_ROUTE` picks up the new wrapper automatically.
  - `category_handlers/handler.py` short-circuit comment rewritten to reflect that the wrapper and executor now exist and to explain why the short-circuit must remain pending the prompt.
  - `docs/modules/search_v2.md` updated to list `media_type_query_execution.py` and note the pending short-circuit.
Design context: Plan file `~/.claude/plans/do-steps-1-4-playful-parrot.md`. The 6-step endpoint-addition pattern is documented in the plan. ADR pending if/when steps 5–6 land. ReleaseFormat enum + `release_format_id` int defined in [schemas/enums.py:178-199](schemas/enums.py#L178-L199); `MEDIA_TYPE` trait category at [schemas/trait_category.py:549-567](schemas/trait_category.py#L549-L567).
Testing notes: Imports clean; `MediaTypeQuerySpec(formats=[ReleaseFormat.TV_MOVIE])` validates; `MediaTypeEndpointParameters` instantiates with the canonical field order; `route_for_wrapper(wrapper)` returns `EndpointRoute.MEDIA_TYPE`. Live-Postgres smoke test still pending. End-to-end via the handler is NOT reachable until steps 5–6 (see caveat above). Schema factories build per-category output schemas from `ROUTE_TO_WRAPPER` at import time, so once the handler short-circuit comes out the MEDIA_TYPE category's handler output schema will include the wrapper automatically. No unit-test changes per the test-boundaries rule.

## media_type endpoint: handler prompt + short-circuit removal (steps 5–6)
Files: search_v2/stage_3/category_handlers/prompts/endpoints/media_type.md (NEW), search_v2/stage_3/category_handlers/prompts/categories/additional_objective_notes/media_type.md (NEW), search_v2/stage_3/category_handlers/prompts/categories/few_shot_examples/media_type.md (NEW), search_v2/stage_3/category_handlers/handler.py, search_v2/stage_3/category_handlers/prompt_builder.py

### Intent
Complete the `media_type` endpoint promotion. With the wrapper, executor, and dispatch already in place from steps 1–4, what remained was the handler-prompt surface and the short-circuit that was protecting the handler from the `build_system_prompt` raise. Once the three .md files land, both can be removed and the endpoint flows through the standard handler path like every other endpoint.

### Key Decisions
- **Endpoint chunk modeled after `endpoints/keyword.md`.** Closest existing template — small, closed-enum, no posting-table mechanics. ~42 lines. Sections in concrete-before-abstract order: Purpose / Canonical question / Capabilities / Boundaries (one principle plus three one-question redirects, not a failure catalog) / Surface-phrase mapping / Polarity discipline / Scope discipline. No registry placeholder — the wrapper's `Literal` is the source of truth and re-listing the four enum values would only add tokens with drift risk.
- **Notes file modeled after `additional_objective_notes/format_visual.md`** — also a boundary-disambiguation surface for a single-bucket category. ~30 lines. Lead with a one-paragraph thesis, then a five-item NEVER list closing the predictable rationalizations ("documentary is not a release format" being the most important), then evidence-before-routing rule, then three one-question boundary tests against FORMAT_VISUAL, runtime in STRUCTURED_METADATA, and SUB_GENRE.
- **Examples file modeled after `few_shot_examples/structured_metadata.md`** (single-endpoint output shape — `requirement_aspects[].{aspect_description, relation_to_endpoint, coverage_gaps}`, `should_run_endpoint`, `endpoint_parameters`). NOT format_visual.md, which uses the multi-endpoint output shape with `endpoint_to_run` / `performance_vs_bias_analysis` / per-aspect `endpoint_coverage` arrays — that shape is for combo-bucket categories. MEDIA_TYPE is `HandlerBucket.SINGLE`. Five examples: 2 fires (single-value, multi-value), 1 negative-polarity fire (locks in the wrapper-polarity discipline), 2 no-fires (vs FORMAT_VISUAL on "documentary", vs runtime on the length-adjective "short"). The 60% no-fire weight is intentional — that's where the boundary discipline gets reinforced.
- **Surface phrasings checked against `search_v2/test_queries.md` for non-overlap.** "TV movie" appears once parenthetically in q9 and "made-for-TV" appears once in q15; my examples use "TV movie" / "TV movies" plural and "made for television" (no hyphens) to maximize lexical distance. "Shorts", "short films", "direct-to-video", "documentary", "anything short" all clear.
- **Removed the `CategoryName.MEDIA_TYPE` short-circuit at handler.py:103-116.** With the prompt files in place, the handler's standard path can build a system prompt and run the LLM. No special-casing remains for MEDIA_TYPE — the only category short-circuit left is TRENDING (deterministic executor, no LLM).
- **Removed `EndpointRoute.MEDIA_TYPE` from `_ENDPOINT_PROMPTLESS` in prompt_builder.py.** The eager-load loop now reads `endpoints/media_type.md` at import time. Updated the preceding comment block (lines 59-62) and the comment inside `build_system_prompt` (line 145-150) to reference TRENDING only.

### Authoring Principles Applied
Distilled from the Step 2 / Step 3 iterations and codified in `docs/conventions.md` "Prompt Authoring Conventions":
1. Generalized principles over failure catalogs — boundaries section gives a one-line principle plus three one-question tests, not enumerated patterns.
2. Lead with NEVER list to close predictable rationalizations — closes "documentary fits because it's a film format" at the top of the notes file.
3. Locally-checkable operational tests for boundary calls — every redirect carries a one-question test the model can run from the visible input.
4. Evidence-before-routing — every fire grounded in a verbatim phrase from `atomic_rewrite` or `parent_fragment`.
5. Polarity stays on the wrapper — the negative-polarity example explicitly demonstrates `formats=[tvMovie]` with `polarity=negative`, and the prose forbids inverting the enum subset to simulate negation.
6. Definitions over examples in prose — endpoint chunk and notes carry definitions; the few-shot file concentrates examples.
7. Don't restate the schema — wrapper's `Literal` is the source of truth.
8. Examples disjoint from the eval set — confirmed via grep against `test_queries.md`.

### Testing Notes
File presence verified for all three .md files (3299 / 3902 / 10392 chars). End-to-end runtime verification of `build_system_prompt(CategoryName.MEDIA_TYPE)` is BLOCKED by a pre-existing import error in prompt_builder.py:20 — the file imports `CoverageEvidence` and `RequirementFragment` from `schemas.step_2` but those types were removed in the v3 step-2 rewrite. This is unrelated to this change and out of scope. Once that pre-existing import is fixed, the eager-load loop will pick up the new endpoint chunk at import time and `build_system_prompt(CategoryName.MEDIA_TYPE)` will assemble all eight chunks. Live LLM smoke test of the assembled prompt against a query like "I want to watch a TV movie tonight" is also pending the import fix. No unit-test changes per the test-boundaries rule.

## media_type endpoint: revert to deterministic-routing direction
Files: search_v2/stage_3/category_handlers/prompts/endpoints/media_type.md (DELETED), search_v2/stage_3/category_handlers/prompts/categories/additional_objective_notes/media_type.md (DELETED), search_v2/stage_3/category_handlers/prompts/categories/few_shot_examples/media_type.md (DELETED), search_v2/stage_3/category_handlers/handler.py, search_v2/stage_3/category_handlers/prompt_builder.py, schemas/media_type_translation.py, docs/modules/search_v2.md
Why: After authoring the three handler-prompt chunks, decided MEDIA_TYPE should be routed deterministically by code (matching surface phrases against the ReleaseFormat enum) rather than through the LLM handler. The wrapper, executor, registry mapping, and dispatch all stay — only the LLM-handler surface is reverted. The deterministic routing path is not yet wired up; until it lands, the handler soft-fails MEDIA_TYPE to an empty result.
Approach:
  - Deleted the three .md files authored in the previous entry (`endpoints/media_type.md`, `additional_objective_notes/media_type.md`, `few_shot_examples/media_type.md`).
  - Re-added `EndpointRoute.MEDIA_TYPE` to `prompt_builder._ENDPOINT_PROMPTLESS`. Updated the preceding comment block to explain that MEDIA_TYPE will be routed deterministically rather than via LLM, and that both TRENDING and MEDIA_TYPE are short-circuited inside `handler.run_handler` so the LLM codepath should never be reached for either.
  - Restored the MEDIA_TYPE short-circuit in `handler.py` with framing that reflects the new direction — "MEDIA_TYPE will be routed deterministically; pending that path landing, soft-fail to an empty result" rather than "pending the prompt".
  - Kept the schema change from the prior in-progress revision: `MediaTypeQuerySpec.formats` now uses a three-value `Literal` subset `(TV_MOVIE, SHORT, VIDEO)` excluding both UNKNOWN (sentinel) and MOVIE (the default release container — emitting it would mean the trait should not have fired). The wrapper is now constructed by code rather than by an LLM, but the closed-enum invariant still represents the right design for a non-default-format-only endpoint and gives the eventual deterministic router a typed input contract.
  - Updated `docs/modules/search_v2.md` to reflect the deterministic-routing direction.
Design context: The full LLM-handler surface (endpoint chunk, NEVER-list notes, five worked examples) was overhead for a problem with a small fixed surface area — three enum values mapped from a finite set of unambiguous phrasings ("TV movies", "shorts", "direct-to-video"). A deterministic phrase-matcher reading the trait's surface text directly will be cheaper, faster, and more reliable than an LLM call.
Testing notes: Verified file deletions, prompt_builder edits, and handler short-circuit restoration. `MediaTypeQuerySpec(formats=[ReleaseFormat.TV_MOVIE])` still validates; `MediaTypeEndpointParameters` round-trips through `route_for_wrapper`. The wrapper and executor remain reachable via `build_endpoint_coroutine` for whatever code path eventually constructs them. No unit-test changes.

## Step 3 inter-attribute information flow rebalance (Experiment 8)
Files: search_v2/step_3.py, schemas/step_3.py, search_improvement_planning/steps_2_3_experimentation.md

### Intent
Rebalance Step 3's reading discipline across the per-trait decomposition layers. Audit + smoke runs surfaced five symptoms of the same pattern: each layer reads from a too-narrow slice of upstream context.

### Key Decisions
- trait_role_analysis prompt rewritten with explicit source priority — qualifier_relation as PRIMARY (its schema description already names Step 3 as the consumer), role + role_evidence as verdict + rationale, contextualized_phrase + evaluative_intent as grounding, anchor_reference as surface pointer. Old prompt led with role as the headline question; new prompt names that as a NEVER.
- role_evidence added to _build_user_prompt — was being committed by Step 2 but never surfaced to Step 3. Load-bearing on borderline traits and on carvers where qualifier_relation is "n/a".
- aspects prompt reframed: target_population is the primary enumeration source; trait_role_analysis qualifies whether each axis describes the population vs the reference. Replaced the old "walk both equally" framing.
- dimensions prompt rewritten to translate every aspect — removed the "two aspects share one searchable check, collapse them" allowance. Source list is aspects; target_population + trait_role_analysis are interpretation aids only. Pre-merging at the dimension layer is now explicitly forbidden; compression happens at category_calls.
- retrieval_intent schema description rewritten as a generic handoff field — removed qualifier-only framing, named retrieval_intent as Step 4's only context source beyond expressions, length expanded to 1-3 sentences. Carver and qualifier traits both populate it. Step 4 does not branch on role (orchestrator-side decision committed upstream).
- Universal "consider all upstream context" rule added to each layer's instructions: read the whole upstream context, do not stop early, do not quietly drop signals that resist translation.

### Result
v11 batch run on 42 test queries (/tmp/step3_runs_v11/):
- Aspect→dimension silent drops fell from 55 to 34 (-38%); traits with drops 52% → 37%.
- Avg retrieval_intent length 132 → 203 chars (+54%); often now names what to discriminate against.
- trait_role_analysis citing qualifier_relation explicitly: 0/84 → 38/86 — same dynamic as Experiment 7's intent_exploration promotion (the field was already engineered for this purpose; only the prompt framing was missing).
- Calls only +4 across 86 traits — no bloat. Simple single-axis traits (q31 "preferably under 2 hours") unchanged at 1 aspect/dim/call.
- Canonical fixes: q29 wes anderson trait now 5 aspects → 5 dimensions (was 5→4 with "meticulous production design" silently dropped); q33 underrated and q34 hidden gem now properly route quality + visibility + commercial-footprint to three independent calls (the previous baseline's "FINANCIAL_SCALE leakage" was actually correct decomposition that the over-eager CLEAN-FIT rule was suppressing).

### Testing Notes
Full experiment write-up in search_improvement_planning/steps_2_3_experimentation.md as Experiment 8 (hypothesis + changes + observations + 8 lessons learned). v11 worth shipping as the new Step 3 baseline. No unit-test changes per the test-boundaries rule.

## Deterministic MEDIA_TYPE category-call router
Files: search_v2/stage_3/category_handlers/media_type_router.py, search_v2/stage_3/category_handlers/handler.py, docs/modules/search_v2.md, docs/TODO.md
Why: `MEDIA_TYPE` category calls were still short-circuiting to an empty handler result even though Step 3 has already decided the expressions are media-type expressions. The endpoint only needs code to resolve which non-default `ReleaseFormat` values the expressions name.
Approach: Added a deterministic helper that matches MEDIA_TYPE expressions against broad phrase buckets for `TV_MOVIE`, `SHORT`, and `VIDEO`, returning a `MediaTypeQuerySpec` or `None` for unsupported default/movie phrasing like theatrical/feature-length. Wired the existing `CategoryName.MEDIA_TYPE` short-circuit to construct a `MediaTypeEndpointParameters` wrapper, stamp the parent trait's role/polarity, and pass through `_assemble_result` so inclusion/exclusion/preference classification stays identical to standard handler output. Updated search_v2 docs and converted the stale TODO into the remaining default-MOVIE/theatrical design question.
Design context: `schemas/media_type_translation.py` intentionally excludes `ReleaseFormat.MOVIE` and `UNKNOWN`; deterministic routing therefore covers only the three non-default values represented by the wrapper. The helper is called only after the category has already been verified as `MEDIA_TYPE`, so patterns can match broad terms like `short` and `video` without defending against whole-query ambiguity.
Testing notes: Ran a 12-case helper smoke matrix covering three variants each for TV movie, short, and video, plus a multi-format call and unsupported theatrical/feature-length calls. Ran a handler-level qualifier-positive smoke confirming the MEDIA_TYPE branch creates a `MediaTypeEndpointParameters` preference spec and records `fired_endpoints`. No unit-test files modified.

## Award endpoint multi-search query plan
Files: schemas/award_translation.py, search_v2/endpoint_fetching/award_query_execution.py
Why: The award category handler now receives `retrieval_intent` plus one or more `expressions`, and multiple expressions may either be fields of one structured award query or separate award searches. The old flat award spec could only represent one COUNT(*) query and forced those cases together.
Approach: Replaced the wrapper payload with `AwardQueryPlan`, containing an `AwardCombineMode` enum (`any`, `average`) and one or more `AwardSearch` entries. Each search has explicit `filters` and `scoring`; filters keep the existing ceremony/name/category/outcome/year axes and scoring keeps the existing FLOOR/THRESHOLD modes. The executor now runs each search independently, preserves requested-but-empty award-name behavior per search, applies Razzie policy per search, uses the fast path per qualifying search, and combines raw scores with max for `any` or average-with-missing-as-zero for `average`. Carver/dealbreaker output compresses positive combined scores into `[0.5, 1.0]`; qualifier/preference output keeps raw scores and fills missing restricted candidates with `0.0`.
Design context: Matches the current category-handler input contract in `schemas/step_3.py`: `retrieval_intent` carries the search shape, while `expressions` carry one or more database-vocabulary seeds. The schema keeps `AwardQuerySpec = AwardQueryPlan` as a temporary compatibility alias for older imports.
Testing notes: Ran `python -m py_compile schemas/award_translation.py search_v2/endpoint_fetching/award_query_execution.py` and a small `AwardQueryPlan.model_validate` smoke check. No unit tests run per repository test-boundary instructions.

## Award endpoint review fixes
Files: schemas/enums.py, schemas/award_translation.py, search_v2/endpoint_fetching/award_query_execution.py, search_v2/endpoint_fetching/*.py, search_v2/endpoint_fetching/category_handlers/*.py, search_v2/reranking/dispatch.py, run_search.py, run_search_json.py
Why: Follow-up from code review of the multi-search award endpoint change: active code still imported the deleted `search_v2.stage_3` package, award-name token resolution did not share the executor retry contract, searches ran sequentially, and `AwardCombineMode` belonged with the shared search enums.
Approach: Moved `AwardCombineMode` into `schemas.enums` next to `AwardScoringMode` and updated `schemas/award_translation.py` to import it. Rewrote active non-test imports from `search_v2.stage_3` to `search_v2.endpoint_fetching`. Added a retry wrapper around award-name token resolution so a double DB failure soft-fails only that search. Switched multi-search execution to `asyncio.gather` so independent award searches run concurrently before combination.
Testing notes: Ran `python -m py_compile` across `search_v2/endpoint_fetching`, `search_v2/reranking`, `schemas/enums.py`, `schemas/award_translation.py`, `run_search.py`, and `run_search_json.py`. `rg` shows remaining stale `search_v2.stage_3` imports only under `unit_tests/`; left them untouched per the repository test-boundary rule.

## Award endpoint prompt rewrite for query plans
Files: search_v2/endpoint_fetching/category_handlers/prompts/endpoints/awards.md
Why: The award endpoint prompt still described the old flat single-spec translation, while the schema/executor now consume an `AwardQueryPlan` with one or more searches and `any`/`average` combination semantics.
Approach: Rewrote the endpoint prompt around the current handler inputs and executor behavior: use `retrieval_intent` for plan shape, use `expressions` for concrete filter values, decide one search vs. multiple searches first, then choose combine mode, per-search scoring, and per-search filters. Kept the registry/taxonomy placeholders and avoided duplicating schema field shape beyond decision-critical behavior. Removed stale relative-year guidance that assumed a `today` input is available to this handler.
Testing notes: Verified the prompt retains `{{CEREMONY_MAPPINGS}}`, `{{AWARD_NAME_SURFACE_FORMS}}`, and `{{CATEGORY_TAG_TAXONOMY}}`. No unit tests run; markdown-only prompt update.

## V3 endpoint notebook: category-handler samples
Files: test_v3_endpoints.ipynb, schemas/entity_translation.py
Why: `test_v3_endpoints.ipynb` was still exercising the old direct endpoint-generator API for Entity, Award, Studio, Franchise, and Metadata. Those endpoint probes should use the new Step 3 `CategoryCall` inputs and category-handler runtime.
Approach: Updated the notebook intro/setup/helper cells and replaced the five requested endpoint cells with sample `CategoryCall` + parent `Trait` inputs routed through `category_handlers.run_handler()`. Each sample uses a representative category (`NAMED_CHARACTER`, `FRANCHISE_LINEAGE`, `NUMERIC_RECEPTION_SCORE`, `AWARDS`, `STUDIO_BRAND`) and prints fired endpoint wrappers plus handler buckets/top inclusion candidates. Added the missing `EntityEndpointParameters` wrapper to `schemas/entity_translation.py` so the handler registry can build dynamic output schemas for entity categories.
Design context: Matches `schemas/step_3.py` and `category_handlers/prompt_builder.py`: handlers receive only `retrieval_intent` and `expressions`; role/polarity are stamped from the parent `Trait`.
Testing notes: Verified `test_v3_endpoints.ipynb` is valid JSON, all code cells parse, and `uv run python` can import `search_v2.endpoint_fetching.category_handlers.handler`, `Trait`, `CategoryCall`, and `CategoryName`. Did not run live notebook cells because they perform LLM/database endpoint calls.

## Keyword endpoint rebuild + endpoint-prompt placeholder substitution
Files: schemas/keyword_translation.py, schemas/unified_classification_families.py (NEW), schemas/streaming_service_surface_forms.py (NEW), search_v2/endpoint_fetching/keyword_query_execution.py, search_v2/endpoint_fetching/category_handlers/prompts/endpoints/keyword.md, search_v2/endpoint_fetching/category_handlers/prompts/endpoints/metadata.md, search_v2/endpoint_fetching/category_handlers/prompt_builder.py, db/postgres.py

### Intent
Two coupled fixes. (1) The keyword endpoint's old single-`classification` schema couldn't represent multi-expression Step 3 calls — multi-dimensional traits like "80s slasher with a final-girl twist" silently collapsed to one of the three plausible registry members. Rebuild the endpoint around an analysis-layer (`attributes` with shortlisted candidates) + commitment-layer (`finalized_keywords` + `scoring`) split so multi-keyword spans are first-class. (2) The category-handler prompt path has been sending literal `{{...}}` placeholder text to the LLM since it was authored — `prompt_builder._read()` slurps the .md verbatim with no template substitution. Wire up a substitution dispatch so all six placeholders (KEYWORD/STUDIO/AWARDS/METADATA) render their dynamic content at module import.

### Key Decisions
- **Schema two-layer shape with server-side dedupe.** `attributes: list[AttributeAnalysis]` (analysis: facets + shortlisted candidates with coverage prose) → `finalized_keywords: list[UnifiedClassification]` (minimum covering set) → `scoring: Literal["any", "avg"]` (aggregation mode read off retrieval_intent). User directed: "In our validator dedupe this list. Don't ask the LLM to deduplicate... it may make the LLM drop genuinely useful signals" — so dedup runs in `@field_validator(mode="after")` while the LLM is told to "emit duplicates freely when the same member is the best fit for multiple attributes."
- **Multi-column hit-count helper replacing single-column overlap.** `db/postgres.py::fetch_keyword_hit_counts` runs ONE SQL statement with OR-of-overlap WHERE (BitmapOr-able by GIN) and `cardinality(ARRAY(SELECT unnest(col) INTERSECT SELECT unnest(...)))` per-column SELECT — no `intarray` extension dependency, single round trip regardless of how the finalized set distributes across columns. The executor groups source_ids by `ClassificationSource`, calls the helper once, then converts hit_counts to scores: `any` → `1.0` if hit_count ≥ 1 else `0.0`; `avg` → `hit_count / N`.
- **Schema field descriptions explicitly cite their derivation sources.** User directed: "Ensure schema fields properly reference each other when they're supposed to. Like if I said that one field should refer to another for deciding its value, make sure that's explicitly stated in the schema description for that field." Every cross-reference uses backtick'd field names (e.g., `coverage` says "How `keyword` matches the parent `attribute`"; `finalized_keywords` says "Pull from members surfaced in `attributes[*].potential_keywords`"; `scoring` says "Aggregation across `finalized_keywords`, read off `retrieval_intent`"). Each commitment field carries a locally-checkable test ("if I dropped this, would the remaining set still cover...?").
- **Keyword.md prompt rewritten from the ground up.** Drops the "exactly one member per firing / no list" framing the schema has now superseded. New progression: Purpose → out-of-scope boundaries (treat as ignore-when-decomposing rather than refuse) → registry placeholder → reading-inputs-as-keyword-facets → surface forms / aliases → near-collision disambiguation (the four principles re-noted to apply at BOTH shortlist and commit) → reading retrieval_intent for scoring mode (linguistic cues for any vs avg).
- **Lifted `_build_classification_registry_section` into `schemas/unified_classification_families.py` (NEW).** It lived in the deprecated `search_v2/endpoint_fetching/keyword_query_generation.py` (old standalone path that's not on the live category-handler codepath). The live `prompt_builder` couldn't import a renderer from a dead module without either reviving the dead module or duplicating the data. Hoisted `_FAMILIES` + render function with the same three import-time consistency checks (no duplicate, no orphan, no missing registry member).
- **Built `schemas/streaming_service_surface_forms.py` (NEW).** Renders display-name + alias mapping for `{{TRACKED_STREAMING_SERVICES}}` as `slug (display; aliases: ...)` lines, iterating `StreamingService` directly so a new enum value flows in automatically. Imports across to `implementation/classes/watch_providers.py`, matching the cross-package import that `schemas/metadata_translation.py` already takes.
- **Removed `{{FREE_STREAMING_SERVICES}}` and the "Free to stream" prompt example entirely.** Investigation found that "free to stream" is semantically distinct from any `StreamingAccessType` value (subscription/buy/rent are all paid; "free" = ad-supported FAST/AVOD with no enum representation). User: "Get rid of the free to stream section entirely that's not supported in our data." This avoided fabricating a `FREE_STREAMING_SERVICES` constant for a feature the data doesn't actually represent.
- **Substitution dispatch with paired safety nets.** `prompt_builder._ENDPOINT_PLACEHOLDER_RENDERERS: dict[EndpointRoute, dict[str, Callable[[], str]]]` maps each route to its `{{TOKEN}} → renderer` mapping. `_load_endpoint_chunk()` reads the .md, iterates the registered tokens, and applies substitutions. Two safety nets verified by import-time test: (1) registered token missing from .md → raises (map/file drift); (2) any `{{...}}` survives substitution → raises (someone added a placeholder without registering a renderer). Renderers run once at module import; handler-time prompt builds remain pure string concatenation.
- **Param ordering in fetch_keyword_hit_counts now matches SQL-string position.** Code-review fix: SELECT placeholders come first in the assembled query string, so the params list is built SELECT-first then WHERE-then-restrict, with inline comments tying each entry to its placeholder. Stays correct if the SELECT/WHERE arrays ever diverge.

### Authoring Principles Applied
The 16-principle guidance the user supplied for schema descriptions distilled to: closed Literals only for genuinely binary/ternary exhaustive commitments (scoring), freeform strings where the tail is long (coverage); no nullable enums; no positional pointers (identity by enum value, not list index); retrieval framing for the attribute test ("would an independent retrieval against this facet hit a meaningful slice?"); locally-checkable per-commitment tests; explicit concept-claim ("attributes do not survive past this layer; only the deduped union reaches execution"); principles over patterns; compact descriptions; schema = micro-prompts (no duplication of system-prompt rules); no test queries in examples; lead with NEVER lists.

### Discoveries
- The `{{CLASSIFICATION_REGISTRY}}` placeholder has never been substituted in the category-handler path. The expansion logic existed all along in the deprecated `keyword_query_generation.py`, but `prompt_builder.py` reads .md files raw with no template engine. The keyword endpoint LLM has been picking from the StrEnum schema's name list with no per-member definitions for the entire lifetime of the category-handler path. Same problem applies to AWARDS (3 placeholders) and STUDIO (1 placeholder); fully fixed in this session.
- 4 of the 7 expected renderers existed (award_surface_forms × 2, award_category_tags, production_brand_surface_forms); 3 needed new code or did not apply (CLASSIFICATION_REGISTRY needed lifting, TRACKED_STREAMING_SERVICES needed building, FREE_STREAMING_SERVICES had no supportable data).

### Testing Notes
Imports clean. Validator dedupe verified via smoke test (input `['HORROR', 'SLASHER_HORROR', 'HORROR']` → `['HORROR', 'SLASHER_HORROR']`). All seven endpoint chunks render with zero leftover placeholders (regex scan). Both safety-net branches verified by stubbed-out `_read` test. Keyword chunk grew from ~2.5k chars (literal `{{CLASSIFICATION_REGISTRY}}`) to ~41k chars (rendered registry with all 259 definitions across 21 families). No unit-test changes per the test-boundaries rule. `unit_tests/test_keyword_query_generation.py` still imports the deprecated module's `_build_classification_registry_section` and may need a follow-up update once that module is removed.


## Semantic endpoint rebuild — schema, executor, prompt
Files: schemas/semantic_translation.py, schemas/semantic_space_selectivity.py (NEW), search_v2/endpoint_fetching/semantic_query_execution.py, search_v2/endpoint_fetching/category_handlers/prompts/endpoints/semantic.md, search_v2/endpoint_fetching/category_handlers/prompt_builder.py

### Intent
Rewrite the semantic endpoint end-to-end for the Step 3 single-trait input shape (one CategoryCall = one retrieval_intent + one expressions list). The prior unified `SemanticParameters` shape was built for one rich user blob and used a `primary_vector` to single-space the carver path; with Step 3's already-decomposed expressions, that framing is wrong — multi-vector carvers are now the norm (pure-vibes traits often legitimately span 2+ spaces) and the LLM should not be re-decomposing inputs that upstream already split. Goal: two role-specific schemas sharing an exploration layer, a multi-vector executor for both paths, and a prompt that defers field-shape rules to the schema and carries only what schemas can't (per-space vocabulary, body authoring register, role-keyed selectivity bar).

### Key Decisions
- **Two top-level schemas, shared exploration layer.** `CarverSemanticParameters` and `QualifierSemanticParameters` both carry `aspects` (atomic decomposition) → `space_candidates` (per-space coverage analysis with what-covered/gap) → `space_queries` (committed minimum-load-bearing set). Carver entries are bare `{space + content}`; qualifier entries wrap with `WeightedSpaceQuery {weight + query}`. `primary_vector` is gone entirely — both paths score multi-vector. Two endpoint wrappers (`CarverSemanticEndpointParameters` / `QualifierSemanticEndpointParameters`) narrow `role` via `Literal[Role.CARVER]` / `Literal[Role.QUALIFIER]` so the schema choice itself encodes role and the orchestrator dispatches by type.
- **Multi-vector carver scoring with elbow + linear-decay window.** Per active space: detect elbow (existing EWMA + Kneedle + pathology fallback structure preserved), set floor uniformly at `elbow * 0.9` (10%-below-elbow decay window per spec), score each cosine via linear decay returning raw [0, 1]. Sum raw across active spaces / N → avg. avg ≤ 0 → DROP (failed every space's threshold); avg > 0 → compress via the shared `compress_to_dealbreaker_floor(raw)` helper from `result_helpers.py` rather than inlining `0.5 + 0.5 * raw`.
- **Path A / Path B handling for degenerate calibration.** `_detect_elbow_and_floor` returns a `SpaceCalibration` dataclass with `pass_through_raw` flag. Path A (Kneedle picked a rank with sim 0): re-route to pathology fallback so floor anchors against `max_sim` instead of collapsing. Path B (max_sim ≤ 0, no usable signal at all): return `_PASS_THROUGH` sentinel; the per-space scorer uses `max(0, sim)` raw cosine clamped to [0, 1] instead of running decay against a manufactured threshold. Path B is the user's explicit "if it happens, just give every vector its raw cosine as score" decision rather than a synthetic threshold.
- **Carver D2 no-fill, qualifier P2 fill.** Carver D2 (no restrict): pool = union of per-space top-N probes; movies absent from a space's probe get 0 for that space (clearing the elbow without being in top-N is implausible — the elbow rank sits inside top-N by construction). Qualifier P2: same pool construction but fills missing per-space cosines via HasId so the weighted sum is honest across the union.
- **Server-side merge validators rather than fail.** Both schemas' `_drop_empty_and_merge_duplicates` validator drops space_queries entries whose body produces no embedding text, then collapses same-space duplicates via the existing `_merge_bodies` (list concat-dedupe, prose ". " join, nested BaseModel recurse). Qualifier weight collisions resolve to CENTRAL (stronger wins). Avoids retry costs for benign LLM emission patterns the schema can recover deterministically.
- **Cross-field references in schema descriptions are explicit.** Every commitment field cites its source field by name in backticks: `aspects` reads `expressions` (primary) + `retrieval_intent`; `space_candidates.aspects_covered` reads `aspects` + `retrieval_intent`; `gap` reads `aspects_covered` + `aspects`; `space_queries` pulls exclusively from `space_candidates`; `WeightedSpaceQuery.weight` reads `retrieval_intent` IN LIGHT OF the matching `SpaceCandidate.aspects_covered`. Per the user's "schema fields properly reference each other" directive — same pattern as keyword endpoint.
- **Role-keyed selectivity guidance hoisted to a Python module.** `schemas/semantic_space_selectivity.py` defines `SelectivityProfile` (frozen dataclass) + `_PROFILES` tuple keyed on `Role` + `render_semantic_selectivity_for_prompt()`. Carver typical 1–2 spaces (equal-vote scoring means marginal spaces dilute), qualifier typical 2–4 (weighted sum tolerates fanning out, SUPPORTING weight exists for round-out spaces). Wired into `prompt_builder._ENDPOINT_PLACEHOLDER_RENDERERS` under `EndpointRoute.SEMANTIC` with token `{{SEMANTIC_SELECTIVITY_GUIDANCE}}`. Same pattern as `award_surface_forms.py` etc. — tunable without editing markdown.
- **Prompt rewritten from scratch.** `semantic.md` reorganized for LLM attention: canonical question → 7 vector spaces (purpose / sub-fields / boundary / examples per space, the load-bearing content) → body authoring (term-list vs prose register, translate-don't-echo, no numerics) → aspect decomposition (compose interlocking signals — "darkly funny" stays one aspect) → role-keyed selectivity bar (the {{}} slot) → boundaries (redirects to other endpoints) → polarity invariant → trust-upstream. Stripped everything the schema field descriptions now carry (cross-references, NEVER lists, operational tests, fold-not-split rules). Drops stale references to `primary_vector` / `qualifier_inventory` / `carries_qualifiers`.

### Planning Context
- User explicitly chose single-trait per call ("Let's switch back to having this handle only one trait at a time. Any merging of semantic queries will happen in the orchestrator level"); bundling of multiple semantic-routed traits is the orchestrator's job.
- User explicitly chose carver compression even when restrict_to_movie_ids is supplied (carver = always dealbreaker semantics regardless of restrict). Convention's literal "compression applies only to no-restrict" framing is loose; OLD code did the same coupling and matches user intent.
- User declined max_length caps on aspects / space_candidates / space_queries; merge validator handles excess-via-duplicates naturally.
- Migration to other call sites (endpoint_executors.py, dispatch.py) deferred per user direction ("I will edit other parts later, handle this executor and schema in isolation"). Both still import the old `SemanticParameters` and pass a `role=` kwarg — they'll break at import until cut over.

### Testing Notes
- `python -m py_compile` passes on schema and executor (verified during writes).
- Prompt builder substitution end-to-end test: `_ENDPOINT_CHUNKS[EndpointRoute.SEMANTIC]` is built at import, `{{SEMANTIC_SELECTIVITY_GUIDANCE}}` is gone post-substitution, no leftover `{{...}}` tokens, both `### Role: Carver` and `### Role: Qualifier` sections present, qualifier weight rule (`CENTRAL` / `SUPPORTING`) and carver bar phrasing (`dilutes the gate`) present. Final chunk ~13.4k chars.
- Sub-field names referenced in the prompt cross-checked against `schemas/semantic_bodies.py` — every name (plot_summary, elevator_pitch, the 8 viewer_experience TermsWithNegations sections, the 4 watch_context TermsSections, the 9 narrative_techniques TermsSections, filming_locations / production_techniques, reception_summary / praised_qualities / criticized_qualities) matches exactly.
- No unit-test changes per the test-boundaries rule. Existing `unit_tests/` references to old `SemanticParameters` shape will need cleanup whenever the broader migration happens.


## Add subintent endpoint parameter variants
Files: schemas/keyword_translation.py, schemas/metadata_translation.py, schemas/semantic_translation.py, search_v2/endpoint_fetching/category_handlers/prompts/endpoints/keyword.md, search_v2/endpoint_fetching/category_handlers/prompts/endpoints/metadata.md, search_v2/endpoint_fetching/category_handlers/prompts/endpoints/semantic.md
Why: Some categories route a single topic to multiple endpoints, so each endpoint needs to consume its own slice of intent rather than the shared retrieval_intent + expressions inputs.
Approach: Duplicate the existing endpoint output schemas to add `<endpoint>EndpointSubintentParameters` variants. Each subintent inner spec declares `<endpoint>_retrieval_intent` as its first field with a generic pointer to the upstream responsibility-splitting reasoning, and every downstream descriptor reads from that field instead of from raw `expressions` / `retrieval_intent`. Sub-types whose descriptors do not reference raw inputs are reused (PotentialKeyword, ColumnCandidate / ColumnSpec / sub-objects, SemanticSpaceEntry and its body types). Wrapper-level `parameters` descriptions are intentionally brief — the inner spec descriptors carry the load.
Generated classes: KeywordQuerySpecSubintent / KeywordEndpointSubintentParameters (plus AttributeAnalysisSubintent), MetadataTranslationOutputSubintent / MetadataEndpointSubintentParameters, CarverSemanticParametersSubintent / QualifierSemanticParametersSubintent (plus SpaceCandidateSubintent + WeightedSpaceQuerySubintent) / Carver- and QualifierSemanticEndpointSubintentParameters. Validators copied verbatim from originals.
Also generalized prompts in prompts/endpoints/{keyword,metadata,semantic}.md to remove explicit references to `expressions` / `retrieval_intent` (now "the brief", "the inputs", "phrases"), and deleted the redundant input-contract block + positive-presence section in metadata.md (already covered by shared/input_spec.md).
Testing notes: `python -c "from schemas.* import *Subintent*"` import smoke test passes; per test-boundaries rule no test updates were attempted.


## Rebuild schema_factories around the 8-bucket taxonomy
Files: search_v2/endpoint_fetching/category_handlers/schema_factories.py, search_v2/endpoint_fetching/category_handlers/endpoint_registry.py

### Intent
Replace the legacy SINGLE / MUTEX / TIERED / COMBO factory dispatch with one factory per HandlerBucket value, so per-category handler output schemas match the bucket-specific reasoning shapes designed in the prior planning round. The new factories source per-endpoint payloads from `get_output_wrapper(endpoint, bucket)` so subintent variants slot in automatically for multi-endpoint buckets, and add a special-case shared schema for the character-franchise fan-out bucket that does not split into per-endpoint payloads.

### Key Decisions
- **Bucket 7 special case lives on `get_output_wrapper`.** Defined `CharacterFranchiseFanoutSchema(BaseModel)` in `endpoint_registry.py` (referent_form_exploration + character_forms + franchise_forms). Prepended a `bucket is CHARACTER_FRANCHISE_FANOUT → return CharacterFranchiseFanoutSchema` short-circuit in `get_output_wrapper` so any caller asking for ENTITY or FRANCHISE_STRUCTURE under that bucket gets the shared schema. The bucket factory just hands back what `get_output_wrapper(endpoints[0], bucket)` returns. Decision per direct user instruction.
- **Bucket-specific reasoning replaces `requirement_aspects` in 5/6/7/8.** Single-endpoint buckets (3, 4) keep the existing `requirement_aspects` decomposition + should_run_endpoint + endpoint_parameters shape. Buckets 5–8 each have their own bucket-level reasoning (preferred_coverage_exploration / semantic_intent + augmentation_opportunities / referent_form_exploration / suitability_overview + coverage_opportunities) that serves as the decomposition. Per the user's resolved design choice — keeps schemas terser and avoids redundancy with bucket-level fields.
- **Bucket-level intent fields kept even though subintent wrappers carry their own `<endpoint>_retrieval_intent`.** Per user choice. The bucket reasoning commits to the split before the wrapper payload is filled; the duplication is intentional sequential reasoning, not noise. Preferred / fallback / semantic intents all live at the bucket level.
- **Combo buckets use top-level params + opportunities list.** Bucket 6 emits `semantic_intent` + `augmentation_opportunities` (list of {endpoint_kind ∈ Literal[deterministic_values], signal_description, worth_running}) + Optional `<endpoint>_parameters` per candidate. Bucket 8 mirrors that with `suitability_overview` + `coverage_opportunities` over every candidate. Forces the LLM to address every candidate via the Literal-bounded list, while keeping endpoint payloads at the top level for clean Lego-block plug-in. Existing `_build_combo`'s nested per_endpoint_breakdown shape was discarded.
- **Bucket 5 interleaves intent and params for Lego-block ordering.** Field order: preferred_coverage_exploration → preferred_intent → `<preferred>_parameters` → fallback_intent → `<fallback>_parameters`. Each endpoint's parameter slot sits immediately after the intent that commits to firing it. Endpoint position 0 in the category tuple is preferred; position 1 is fallback.
- **`MEDIA_TYPE` deliberately drops out of OUTPUT_SCHEMAS.** Its bucket is `NO_LLM_PURE_CODE`, so the no-schema factory returns None even though the route has a wrapper class. Old factory accidentally built a schema for it; the new dispatch is bucket-driven so MEDIA_TYPE joins TRENDING and BELOW_THE_LINE_CREATOR in the no-LLM exclusion set.

### Testing Notes
Smoke test: `python -c "import search_v2.endpoint_fetching.category_handlers.schema_factories"` builds 40 schemas (43 categories minus 3 no-LLM/no-op categories). Spot-check verified the field set per bucket: PERSON_CREDIT (3), RUNTIME (4), CENTRAL_TOPIC (5: `[preferred_coverage_exploration, preferred_intent, keyword_parameters, fallback_intent, semantic_parameters]`), EMOTIONAL_EXPERIENTIAL (6: `[semantic_intent, augmentation_opportunities, semantic_parameters, keyword_parameters]`), CHARACTER_FRANCHISE (7: `CharacterFranchiseFanoutSchema` direct), TARGET_AUDIENCE (8: `[suitability_overview, coverage_opportunities, keyword_parameters, metadata_parameters, semantic_parameters]`). `model_json_schema()` round-trips on every sample; TARGET_AUDIENCE is the largest at ~58KB (within OpenAI structured-output limits but worth watching as more bucket-8 categories land). `get_output_wrapper(ENTITY|FRANCHISE_STRUCTURE, CHARACTER_FRANCHISE_FANOUT)` returns `CharacterFranchiseFanoutSchema` in both cases. No test files touched per test-boundaries rule.


## Rewrite Single Non-Metadata category prompt chunks
Files: search_v2/endpoint_fetching/category_handlers/prompts/categories/additional_objective_notes/{person_credit,title_text,named_character,studio_brand,franchise_lineage,adaptation_source,awards,filming_location,plot_events,narrative_setting,viewing_occasion,visual_craft_acclaim,music_score_acclaim,dialogue_craft_acclaim,named_source_creator}.md; search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/{person_credit,title_text,named_character,studio_brand,franchise_lineage,adaptation_source,awards,filming_location,plot_events,narrative_setting,viewing_occasion,visual_craft_acclaim,music_score_acclaim,dialogue_craft_acclaim,named_source_creator}.md
Why: Bucket 3 categories needed compact, category-specific objective notes and few-shot calibration aligned to `query_buckets.md` and `query_categories.md`, with existing verbose/stale chunks replaced.
Approach: Deleted/replaced any existing files for the bucket, then authored one additional-objective note file and one few-shot file per category. Notes focus on decision boundaries, endpoint fit, and no-fire discipline; examples use the current `retrieval_intent` + `expressions` handler input shape and compact expected outcomes rather than full schema restatements. Applied small-LLM prompt learnings from planning docs: explicit section anchors, high-signal wording, no-fire examples, schema-not-restated, and shape calibration over exhaustive category lists.
Design context: `search_improvement_planning/query_buckets.md` Bucket 3 is the source for the category set and single-endpoint handling. `search_improvement_planning/query_categories.md` is the source for category boundaries, with the user confirming visual craft, dialogue craft, and named source creator should follow the current single-semantic endpoint bucket despite older combo wording in that doc.
Testing notes: Prompt-builder smoke validation passed for all 15 Single Non-Metadata categories via `build_system_prompt(category)` and confirmed both `# Additional objective notes` and `# Few-shot examples` sections are present. Full `schema_factories` import was not used for verification because the current dirty worktree has an unrelated `_build_single()` signature mismatch.


## Add Character-franchise prompt chunks
Files: search_v2/endpoint_fetching/category_handlers/prompts/categories/additional_objective_notes/character_franchise.md; search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/character_franchise.md
Why: Bucket 7's Character-franchise category needed its per-category additional objective notes and few-shot examples so the prompt builder can assemble the fixed fan-out handler prompt.
Approach: Added compact, section-anchored notes that ask whether the target is one fictional persona that also anchors a character-centered film franchise, then constrain character forms, franchise forms, separation of source/structural signals, and no-fire boundaries. Added shape-focused examples for positive fan-out, source-medium composition, and close no-fire cases.
Design context: `search_improvement_planning/query_buckets.md` Bucket 7 defines the fixed character + franchise fan-out shape; `search_improvement_planning/query_categories.md` Cat 6 defines the dual-nature referent boundary.
Testing notes: Verified `build_system_prompt(CategoryName.CHARACTER_FRANCHISE)` succeeds and includes both new section headers. No unit tests were read, edited, or run.


## Rewrite Single Metadata category prompt chunks
Files: search_v2/endpoint_fetching/category_handlers/prompts/categories/additional_objective_notes/{release_date,runtime,maturity_rating,audio_language,streaming,financial_scale,numeric_reception_score,country_of_origin,general_appeal,chronological}.md; search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/{release_date,runtime,maturity_rating,audio_language,streaming,financial_scale,numeric_reception_score,country_of_origin,general_appeal,chronological}.md; deleted stale structured_metadata.md and reception_quality.md category chunks
Why: Bucket 4 categories now need separate per-category prompt chunks instead of the old broad structured-metadata prompt, with compact small-LLM-friendly guidance and boundary examples.
Approach: Authored one additional-objective note file and one few-shot file for each Single Metadata Endpoint category from `query_buckets.md` / `query_categories.md`. Notes use explicit anchors, decision questions, boundaries, no-fire discipline, and avoid restating schema field semantics. Examples calibrate firing shape and close boundary failures rather than enumerating all possible values.
Design context: `search_improvement_planning/query_buckets.md` Bucket 4 and `search_improvement_planning/query_categories.md` Cats 13-20, 38, and 44 are the source of truth. Existing `structured_metadata.md` and `chronological.md` were stale against the finalized per-category split; `reception_quality.md` was an older reception/status prompt not used by the current Single Metadata bucket.
Testing notes: Prompt-builder smoke validation passed for all 10 Single Metadata categories via `build_system_prompt(category)`, confirming both `# Additional objective notes` and `# Few-shot examples` sections are present. No unit tests run per the test-boundaries rule.


## Rewrite Audience-Suitability prompt chunks
Files: search_v2/endpoint_fetching/category_handlers/prompts/categories/additional_objective_notes/{target_audience,sensitive_content}.md; search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/{target_audience,sensitive_content}.md
Why: Bucket 8's Target audience and Sensitive content categories needed compact, high-signal prompt chunks that reflect the finalized redundant-combo handling without carrying over verbose old schema-shaped examples.
Approach: Deleted/replaced the existing files in both category prompt folders. Notes now use explicit anchors, decision questions, endpoint-fit discriminators, boundary rules, and first-class no-fire guidance. Few-shots teach shape-level decisions: endpoint combinations for audience packaging, watch-context companion asks, adult packaging, kid-watch scenarios, binary content flags, gradient gore intensity, rating ceilings, and boundary no-fires.
Design context: `search_improvement_planning/query_buckets.md` Bucket 8 and `search_improvement_planning/query_categories.md` Cats 27-28 are the source of truth. Existing prompt content was used as context, then compressed and adjusted to the current `retrieval_intent` + `expressions` input shape and Bucket 8 schema direction where parent role/polarity is stamped outside endpoint payloads.
Testing notes: Prompt-builder smoke validation passed for Target audience and Sensitive content via `build_system_prompt(category)`, confirming both category sections assemble. No unit tests run per the test-boundaries rule.


## Rewrite Semantic-Always prompt chunks
Files: search_v2/endpoint_fetching/category_handlers/prompts/categories/additional_objective_notes/{seasonal_holiday,emotional_experiential,cultural_status}.md; search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/{seasonal_holiday,emotional_experiential,cultural_status}.md
Why: Bucket 6 categories needed compact current-shape prompt chunks aligned to semantic-always plus deterministic augmentation, and existing Seasonal / holiday and Cultural status chunks were stale or too loose for the new schema.
Approach: Deleted/replaced existing Seasonal / holiday and Cultural status files, then authored one additional-objective note file and one few-shot file for each Bucket 6 category, including the previously missing Emotional / experiential chunks. Notes anchor the semantic-first decision, deterministic augmentation criteria, boundary checks, and no-fire cases. Examples teach output-shape decisions through `semantic_intent`, `augmentation_opportunities`, and endpoint parameter commitments without restating full schema semantics.
Design context: `search_improvement_planning/query_buckets.md` Bucket 6 defines semantic-always with deterministic augmentation. `search_improvement_planning/query_categories.md` Cats 29, 33, and 39 define the category boundaries and endpoint surfaces.
Testing notes: Verified the six expected files exist and reviewed the diff for compactness/current field names. Prompt-builder smoke validation passed for Seasonal / holiday, Emotional / experiential, and Cultural status via `build_system_prompt(category)`, confirming the assembled prompts include additional objective notes and few-shot examples. No unit tests run per the test-boundaries rule.

## Align category handler runtime with schema/prompt revamp
Files: search_v2/endpoint_fetching/category_handlers/handler.py, search_v2/endpoint_fetching/endpoint_executors.py, search_v2/endpoint_fetching/orchestrator.py, search_v2/endpoint_fetching/category_handlers/schema_factories.py
Why: The handler runtime still assumed only TRENDING/MEDIA_TYPE were special-cased, still called `get_output_schema(category)` without role, and still stamped/read role + polarity on endpoint wrappers even though the revamped schemas keep role/polarity on the parent `Trait`.
Approach: Added explicit `EXPLICIT_NO_OP` no-op handling and a bucket-level `NO_LLM_PURE_CODE` dispatch that only accepts TRENDING and MEDIA_TYPE, raising for any future deterministic category without a registered codepath. Passed `trait.role` into `get_output_schema(category, role)`. Removed wrapper role/polarity mutation and now classify fired endpoints from the parent `Trait` sidecar. Updated media-type wrapper construction to only pass `parameters`. Made semantic endpoint execution receive role as an explicit argument, and registered subintent + semantic wrappers for deferred preference route lookup; orchestrator deferred preferences execute with `Role.QUALIFIER`, which matches the only handler path that emits `preference_specs`.
Design context: Matches the prompt/schema revamp where endpoint payloads are parameter-only and role/polarity are upstream trait commitments. Preserves prompt_builder's contract that no-LLM/no-op buckets never reach system-prompt construction.
Testing notes: Ran `python -m py_compile` on touched runtime files. Ran `uv run` smoke checks importing `run_handler`, verifying subintent route lookup for keyword and semantic wrappers, and confirming `run_handler()` returns an empty categorized result for `BELOW_THE_LINE_CREATOR`. No unit tests read or run per test-boundary rule.

## Fix endpoint orchestrator imports and semantic execution contract
Files: search_v2/endpoint_fetching/orchestrator.py, search_v2/endpoint_fetching/semantic_query_execution.py
Why: `orchestrator.py` still imported deleted Step-2 symbols (`Step2Response`, old requirements / coverage-evidence path) and therefore failed at import. Semantic execution also needed the role passed by `endpoint_executors.py` to be enforced explicitly, with subintent semantic parameter variants accepted.
Approach: Updated the endpoint-fetching orchestrator to import `QueryAnalysis`, `Trait`, `CategoryCall`, and `TraitDecomposition`, and to fan out `(Trait, CategoryCall)` pairs from Step-2 traits plus Step-3 decompositions. Removed the stale implicit-expectations fan-out import/path because that module is still tied to the deleted `Step2Response` contract. In semantic execution, added explicit `role: Role`, accepted base and subintent carver/qualifier params, and raised `AssertionError` when role/type or role/restriction contracts disagree: carver must use carver params with no `restrict_to_movie_ids`, qualifier must use qualifier params with supplied `restrict_to_movie_ids`. Empty qualifier pools still short-circuit to an empty result.
Design context: Keeps role as an upstream trait commitment and makes semantic execution's boundary contract explicit instead of inferring role from restriction mode.
Testing notes: Ran `python -m py_compile` on touched runtime files. Ran `uv run` smoke checks confirming `orchestrator` and semantic executor import, and confirming semantic boundary assertions for carver-with-ids, qualifier-without-ids, and wrong params family. No unit tests read or run per test-boundary rule.

## Route endpoint notebook probes through run_handler
Files: test_v3_endpoints.ipynb
Why: The V3 endpoint notebook still had several cells exercising direct generator/executor helpers instead of the category-handler entry point, so they no longer matched the intended `run_handler(category_call=..., trait=..., qdrant_client=...)` contract.
Approach: Converted the keyword, media-type, trending, semantic dealbreaker, and semantic preference cells to build notebook `Trait` + `CategoryCall` inputs and call `run_handler()` with the same expected parameters as the already-migrated entity/franchise/metadata/award/studio cells. Removed stale provider/model configuration and updated markdown headings to describe the handler path. Cleared outputs for edited cells so old direct-executor output is not mistaken for current behavior.
Design context: Follows `docs/modules/search_v2.md` category-handler direction: Step-3 category calls are the handler input, parent `Trait` owns role/polarity, and deterministic categories still flow through the shared handler return contract.
Testing notes: Parsed the notebook JSON successfully and verified all ten endpoint probe code cells call `run_handler()` exactly once with no remaining direct `generate_*` or `execute_*` calls. No unit tests run per test-boundaries rule.


## Implement CHARACTER_FRANCHISE_FANOUT handler dispatch
Files: search_v2/endpoint_fetching/category_handlers/handler.py
Why: `_extract_fired_endpoints` raised `NotImplementedError` for `HandlerBucket.CHARACTER_FRANCHISE_FANOUT`, so any Character-franchise category routed at runtime would tank the surrounding query branch instead of producing candidates. The schema layer (`CharacterFranchiseFanoutSchema`) was already implemented and the LLM call would succeed; only the form-list → endpoint-payload adapter was missing.
Approach: Added `_fanout_to_fired_endpoints` that translates the shared `CharacterFranchiseFanoutSchema` output into ordinary `(EndpointRoute, EndpointParameters)` pairs — `character_forms` → `CharacterQuerySpec` on `EndpointRoute.ENTITY` (single `CharacterTarget` whose `forms` is the list), `franchise_forms` → `FranchiseEndpointParameters` carrying a `FranchiseQuerySpec` with `franchise_names` populated. Either side is skipped when its form list is empty; both empty is a valid zero-fired outcome. The shared `referent_form_exploration` is reused for the per-target / per-spec exploration prose (executors do not read those fields — they exist only as LLM scaffolding) and `prominence_mode=DEFAULT` is hard-coded because the fanout schema commits no centrality reading. Rejected the alternative of a fanout-specific endpoint executor because it would require a new `EndpointRoute` and a new `EndpointParameters` subclass purely as a wrapper that fans out into the existing entity + franchise executors — same code, more layers, identical downstream consolidation behavior.
Design context: Bucket-7 design (one referent identified once, two retrievals) per `search_improvement_planning/query_buckets.md`. The adapter approach matches the multi-route consolidation contract every other multi-endpoint bucket already uses (`_classify_wrapper` + additive merge in `_land_outcome`), so no provenance tracking is required downstream.
Testing notes: Smoke-tested via `uv run python -c "from search_v2.endpoint_fetching.category_handlers import handler"`. Worth covering: both form lists populated → two fired pairs land in the trait's role/polarity bucket; one side empty → only the populated route fires; both empty → empty list, handler short-circuits to empty result. No unit tests read or run per the test-boundary rule.

## Replace Step 2 `role` + `salience` with a single `commitment` axis
Files: schemas/step_2.py, search_v2/step_2.py

### Intent
The deterministic-logic redesign in `search_improvement_planning/search_method_deterministic_logic.md` §1 removes `role` (carver/qualifier) and `salience` from the Step 2 trait payload. Orchestration mode is now deterministic from the endpoint, not from a per-trait LLM label, and salience generalises into a single importance axis that applies to every trait. This change implements the Step 2 side: schema, descriptors, and system prompt only.

### Key Decisions
- **Five-level commitment axis (not three).** §1 specifies `required` / `central` / `supporting`. We extended to five — `required`, `elevated`, `neutral`, `supporting`, `diminished` — to represent implicit structural-importance signal in queries that lack explicit hedge / must-language. Without the extra inner steps, every non-hedged trait flattens to a single bucket and the model loses the salience signal that previously discriminated headline wants from trailing refinements. Explicit signals occupy the extremes; structural signals occupy the inner half-step; neutral is a true middle. Multiplier scale is left to §7 of the deterministic doc and remains tunable per §12.
- **`commitment_evidence` is a separate evidence-gathering field, not a justification field.** It surveys both signal channels (explicit phrasing, structural prominence) without picking a level — the level commits in `commitment` as the natural conclusion. The atom-layer `split_exploration` / `standalone_check` framing is reused so the LLM does not slip into post-hoc justification.
- **Schema field order: polarity moves up.** `commitment_evidence` reads polarity for the structural-required override (explicit negative-polarity content traits commit `required` regardless of phrasing). Field order is now: surface_text → evaluative_intent → qualifier_relation → anchor_reference → polarity → commitment_evidence → commitment → contextualized_phrase. Cross-references in the schema descriptions are explicit so the dependency graph is legible.
- **Scope limited to Step 2.** Step 3 (`search_v2/step_3.py`), endpoint dispatch (`category_handlers/handler.py`, `schema_factories.py`, `endpoint_registry.py`), semantic execution (`semantic_query_execution.py`, `endpoint_executors.py`), the run scripts (`run_step_3.py`, `run_test_queries.py`), `search_v2/endpoint_fetching/orchestrator.py`, `search_v2/reranking/dispatch.py`, and `schemas/semantic_space_selectivity.py` all still read `trait.role` / `trait.role_evidence` / `trait.relevance_to_query` and will fail to import or run until separately migrated. Confirmed scope with user before executing. `schemas/enums.py:Role` is intentionally left in place — those downstream consumers still reference it.
- **Principle-based prompt prose, no example queries.** Per the user's earlier guidance, the new `_COMMITMENT` section uses generalized signal descriptions (function over surface tokens) rather than illustrative query examples, matching the existing prose style of the rest of the file.

### Planning Context
The plan file at `/Users/michaelkeohane/.claude/plans/great-all-this-makes-snappy-bunny.md` captures the design discussion (option 1 vs. option 2 single-field-with-five-levels evaluation, the renaming of central → neutral and the supporting/diminished swap, the commitment-evidence vs. justification framing, and the schema-placement reasoning).

### Testing Notes
Schema imports cleanly (`Trait.model_fields` returns the eight expected names in order; `commitment.annotation` is the five-literal Union). End-to-end Step 2 runs were exercised against three queries:
- "funny zombie movies from the 80s with bill murray" → headline traits at `elevated`, but `80s` and `bill murray` committed `required`. The model is conflating specificity / information density with required commitment despite the `NEVER COMMIT REQUIRED ON IMPLICIT IMPORTANCE ALONE` rule. Behavioral, not structural — worth a future prompt-tuning pass.
- "movies about grief and redemption set in small towns" → `grief` and `redemption` at `elevated` (coordinate primaries), `small towns` at `supporting`. Matches the principle.
- "ideally a fincher psychological thriller from the 90s, nothing too long" → `fincher` and `psychological thriller` at `elevated` (channels disagreed: hedge said diminished, structure said elevated; the prompt's holistic-call guidance pulled toward elevated), `90s` at `supporting`, `long` with `nothing too` at `diminished` + `negative` (the special case lands cleanly).

Unit tests intentionally not run — `unit_tests/test_search_v2_*` and `unit_tests/test_schema_factories.py` reference removed fields and will fail until downstream is migrated.

### Refinement pass after review
Folded the four findings from the post-implementation review:
- **Removed the structural-required category override**, replaced with a function-based recognition section. The previous version listed specific categories ("distribution channel, audio language, format gate, maturity ceiling") which leaked through `such as` framing and let the model treat release era and named entities as override-eligible. The new section describes what the override-eligible traits have in common — phrasing whose function names a watching precondition or asserts an exclusion (rather than expressing a preference within the watchable space). The recognition is by function, not by category list.
- **Explicit-dominates precedence replaces the "holistic call" disagreement-resolution paragraph.** Both surfaces (schema field descriptions and the system prompt's `_COMMITMENT` section) now state: when the explicit channel fires, the trait commits at the level it names regardless of structural prominence; the structural channel only sets the level when the explicit channel is silent. This eliminates the prior "structural can pull a hedged trait toward supporting" carve-out that the model was over-applying.
- **REQUIRED and DIMINISHED gated on explicit signal in both surfaces.** The schema `commitment` field's NEVER list now reads "COMMIT REQUIRED OR DIMINISHED WITHOUT AN EXPLICIT SIGNAL." The system-prompt `RECOGNIZING REQUIRED AND DIMINISHED` section says the same thing in prose. Strong implicit prominence commits ELEVATED; structural triviality commits SUPPORTING.
- **Class docstrings deleted** from `ModifyingSignal`, `Atom`, `Trait`, `QueryAnalysis` and converted to `# comment blocks` above each class. Verified `model_json_schema()` no longer carries any class-level descriptions to Gemini, per `docs/conventions.md` §"Pydantic schemas as LLM response_format."
- **Removed the `NEVER LET HEDGES FLIP POLARITY`** bullet from `commitment`'s NEVER list. The polarity field description already carries the same rule in its native location.

Re-ran the three verification queries against the refined surface:
- "funny zombie movies from the 80s with bill murray" → `funny` at `neutral`, `zombie movies` at `elevated`, `80s` and `bill murray` at `supporting` — Q1 calibration miss is fixed; named entities and release eras no longer auto-trigger `required`.
- "movies about grief and redemption set in small towns" → `grief` and `redemption` at `elevated`, `small towns` at `neutral`. Coordinate primaries unchanged; trailing setting drifted between supporting and neutral across runs (defensible either way at the boundary).
- "ideally a fincher psychological thriller from the 90s, nothing too long" → `fincher` and `psychological thriller` at `diminished` (explicit-dominates now pulls them all the way down despite headline structural prominence; the model explicitly cites "the explicit channel fires" in its evidence), `90s` at `supporting`, `long` at `diminished + negative`. Disagreement-resolution issue is fixed.

## Finalize commitment multipliers and rarity tiers in §7
Files: search_improvement_planning/search_method_deterministic_logic.md
Why: §7 listed commitment multipliers as approximate starting values and rarity as "log-based transform" without concrete tiers. Needed concrete numbers to implement Step C scoring.
Approach: Set commitment multipliers on a √e geometric scale (3.0 / 1.75 / 1.0 / 0.6 / 0.35) — chosen so a benchmark "1 REQUIRED + 3 NEUTRAL" query has REQUIRED-only candidates tying with miss-REQUIRED-but-match-all-others candidates (the "soft gate" pivot point). Set rarity as discrete tiers over corpus fraction (ultra-rare/rare/moderate/common/very common at 0.1%/1%/10%/30% boundaries, factors 1.5/1.2/1.0/0.75/0.5) — tiered rather than continuous because trait specificity is naturally coarse and tier boundaries are easier to tune than a continuous slope. Bounded [0.5, 1.5] gives a 3× span — wide enough to weight specific actors above broad genres, narrow enough that rarity refines but doesn't dominate commitment. Also updated §12 open items to reflect that these are set (still tunable via geometric ratio / tier boundaries, no longer TBD).

## Unify carver/qualifier semantic schemas; LLM commits retrieval shape per call
Files: schemas/semantic_translation.py, search_v2/endpoint_fetching/category_handlers/endpoint_registry.py, search_v2/endpoint_fetching/category_handlers/schema_factories.py, search_v2/endpoint_fetching/semantic_query_execution.py, search_v2/endpoint_fetching/endpoint_executors.py, search_v2/endpoint_fetching/category_handlers/handler.py, search_v2/endpoint_fetching/category_handlers/prompts/endpoints/semantic.md, run_search.py, run_search_json.py

### Intent
Eliminate the role-keyed schema fork on the semantic endpoint. Previously, `get_output_schema(category, role)` returned `Carver*` vs `Qualifier*` Pydantic classes whose `space_queries` shapes differed (bare entries vs weighted entries). The role decision was inherited from the parent Trait — driven by `qualifier_relation == "n/a"` after the Step-2 schema lost its explicit `role`. That bridge was a heuristic mapping; the LLM looking at the actual call has more information about whether the trait wants strict population-naming retrieval or looser positioning-against-a-reference retrieval. New design: one unified `SemanticParameters` schema with a leading `role_exploration` reasoning field and a committed `role` enum, followed by aspects/space_candidates/space_queries. The executor reads `params.role` at runtime to dispatch between equal-vote (carver) and weighted-sum (qualifier) scoring.

### Key Decisions
- **New `SemanticRetrievalShape` enum** in `schemas/semantic_translation.py` (CARVER/QUALIFIER). Distinct from the old Step-2 `Role` enum (already removed) — this lives on the semantic-side schema and is emitted by the handler LLM per call.
- **Per-entry weight is always populated, regardless of role.** Carver execution ignores it; qualifier execution uses CENTRAL=2.0 / SUPPORTING=1.0. Per user direction: "always populated and we choose to ignore when it's not relevant" — keeps the data interpretable for evaluation and avoids an Optional + soft-contract validator.
- **No selectivity validator.** The 1–2-spaces (carver) vs 2–4-spaces (qualifier) bar is now purely behavioral — surfaced through field descriptions and `{{SEMANTIC_SELECTIVITY_GUIDANCE}}`. Per user: "no vector quantity check that's not helpful." Inconsistent commits (e.g. CARVER + 5 spaces) are an LLM behavior issue, not schema-enforced.
- **Subintent variant collapses cleanly.** The 4-class surface (`{Carver,Qualifier}{,Subintent}`) became 2 (`SemanticEndpointParameters`, `SemanticEndpointSubintentParameters`).
- **`get_output_schema(category, role)` keeps its signature** so call sites compile, but the role arg is now ignored — every category builds one schema and stores it under both Role keys. `_output_class_name` drops the Carver/Qualifier suffix; classes are now bare `<Name>Output`.
- **`execute_semantic_query` no longer takes a `role` kwarg.** It reads `params.role` and dispatches internally. The boundary contract (`restrict_to_movie_ids` required for qualifier, forbidden for carver) is preserved against the LLM-committed role rather than a caller-passed one.
- **`build_endpoint_coroutine`'s `role` kwarg becomes vestigial for semantic** but is retained on the signature for back-compat. The old `_derive_role` bridge in `full_pipeline_orchestrator` is now functionally unused for semantic but left in place to limit blast radius.
- **Prompt framing rewritten** in `semantic.md`: replaced the carver/qualifier-schema-as-given framing with "your `role` decision dictates which selectivity profile applies." `{{SEMANTIC_SELECTIVITY_GUIDANCE}}` still renders both profiles every call (the LLM picks which to apply rather than being handed one schema).

### Testing Notes
- Unit tests in `unit_tests/test_schema_factories.py` reference deleted classes (`CarverSemanticEndpointParameters` etc.) and the dual-suffix class names. They will fail until updated. Per project test-boundary rules these were not modified in this changeset.
- Notebook imports (`test_search_v2.ipynb`, etc.) likely also reference deleted classes — left to update interactively.
- LLM-side change: handler-LLM must now commit `role` inside `SemanticParameters` rather than receiving a role-pre-keyed schema. Behavior on "obviously carver" categories (filming location, viewing occasion, etc.) is no longer structurally guaranteed — an evaluation pass on the LLM's role-decision accuracy across the 21 SEMANTIC-bearing categories is the natural follow-up.

## Add operation_type to GeneratedEndpointSpec
Files: schemas/enums.py, search_v2/full_pipeline_orchestrator.py

### Intent
Stage-4 execution and final scoring need to know whether each generated endpoint spec runs as a candidate finder (produces a pool) or a pool reranker (scores an existing pool). The distinction is deterministic from `(category, route, parent-trait polarity)` per `search_improvement_planning/search_method_deterministic_logic.md` §2 / §4 / §8. This change adds the field, the deriving helper, and wires assignment at every spec construction site.

### Key Decisions
- **New `OperationType` StrEnum** (`CANDIDATE_GENERATOR` / `POOL_RERANKER`) lives in `schemas/enums.py` next to `EndpointRoute` since they're paired at every spec.
- **Field is non-nullable, set at construction.** Initial implementation made it `OperationType | None` with a post-pass (`_assign_operation_types`) that walked every spec on the result. Replaced that with inline assignment at each of the four construction sites because the post-pass added a transient None state the type system couldn't enforce, and the determination is a pure function of inputs already available at construction time. The helper is the single source of truth; every site calls it. Trade-off: `_process_deterministic_category_call` now takes a `polarity` kwarg threaded from the parent trait. Worth it for type-correctness end to end.
- **Negative-polarity short-circuit** at the top of `_determine_operation_type` returns `POOL_RERANKER` regardless of route. Implements the §4/§8 rule that negative-polarity pool finders orchestrate as pure rerankers — they never add candidates, they only downrank existing union members. Applies even to routes that are already rerankers (semantic, etc.); same answer either way, dead-effort but matches the doc's framing of negative polarity as a global override.
- **METADATA splits by category.** `GENERAL_APPEAL` (Cat 38 reception/popularity prior), `CULTURAL_STATUS` (Cat 39 popularity prior on the META side), and `CHRONOLOGICAL` (Cat 44 sort-and-pick over `release_date`) → `POOL_RERANKER`. Every other METADATA-routing category → `CANDIDATE_GENERATOR`. The `release_date` column flipping mode between RELEASE_DATE (Cat 13, finder) and CHRONOLOGICAL (Cat 44, reranker) is the cleanest example of why the helper takes both category and route.
- **MEDIA_TYPE listed under always-finder.** It's the §2 META `media_type` column elevated into its own EndpointRoute in this codebase. Comment in the helper makes the reconciliation explicit.
- **Auxiliary shorts-exclusion spec hardcodes `CANDIDATE_GENERATOR`.** It has no parent trait, so `_determine_operation_type` doesn't apply. Per user direction, the comment on `_build_shorts_exclusion_spec` now spells out that this is the ONLY true categorical exclusion in the entire pipeline (every user-expressed negative trait is a soft downranker via negative-polarity reranking). The hardcode is intentional and the comment flags that a second auxiliary spec would force revisiting the assumption.
- **Tier-fallback promotion (§10) deferred.** Per user direction, that override is a follow-up pass; the helper returns the pre-promotion default for SEMANTIC traits. Comment in the SEMANTIC branch flags this.

### Testing Notes
Smoke checks (not via unit tests per test-boundary rule): construction without `operation_type` now raises `TypeError`; helper returns expected modes for the eight cases enumerated in the plan file (positive RELEASE_DATE/META → finder; positive CHRONOLOGICAL/META → reranker; positive GENERAL_APPEAL/META → reranker; positive PLOT_EVENTS/SEMANTIC → reranker; positive PERSON_CREDIT/ENTITY → finder; positive MEDIA_TYPE → finder; negative MEDIA_TYPE → reranker; negative NUMERIC_RECEPTION_SCORE/META → reranker via short-circuit). End-to-end run of `run_full_pipeline` against a query exercising every endpoint route is the natural follow-up; downstream stage-4 consumers should now read `spec.operation_type` directly with no `None` handling required.

## Drop vestigial Role params; remove carver/qualifier prompt guidance
Files: search_v2/endpoint_fetching/category_handlers/endpoint_registry.py, search_v2/endpoint_fetching/category_handlers/schema_factories.py, search_v2/endpoint_fetching/endpoint_executors.py, search_v2/endpoint_fetching/category_handlers/handler.py, search_v2/full_pipeline_orchestrator.py, search_v2/endpoint_fetching/category_handlers/prompt_builder.py, search_v2/endpoint_fetching/category_handlers/prompts/endpoints/semantic.md, schemas/semantic_translation.py, schemas/semantic_space_selectivity.py (deleted)

Why: After unifying the semantic schema, the `Role` parameter threaded through `get_output_wrapper` / `get_output_schema` / `build_endpoint_coroutine` / `_run_handler_llm` and the `_derive_role` bridge in the orchestrator were all dead — the LLM commits carver vs qualifier inside `SemanticParameters.role` and the executor reads it from `params`. Likewise the system-prompt guidance about carver vs qualifier (the opening framing paragraph, the "Selectivity bar by role" section, and the always-populated-weight paragraph that duplicated the field description) was redundant with the schema field descriptions and added ~2.4k chars per call.

Approach:
- `get_output_wrapper(endpoint, bucket, *, category=None)` — dropped `role` kwarg + `_SEMANTIC_DISPATCH` key collapsed from `(Role, bool)` to `bool`.
- `OUTPUT_SCHEMAS` re-keyed from `dict[(CategoryName, Role), ...]` to `dict[CategoryName, ...]`. `get_output_schema(category)` is the new signature.
- All bucket factories, `_resolve_wrappers_for_bucket`, `_output_class_name`, and `_BucketFactory` lose the `Role` parameter.
- `build_endpoint_coroutine` drops `role` kwarg.
- `_run_handler_llm` drops `role` kwarg; the two call sites in handler.py (one of which passed `role=trait.role` against the now-absent attribute) are updated to no longer pass it.
- `_derive_role` and the matching `Role` import removed from `full_pipeline_orchestrator.py`. Note: the orchestrator's `_process_llm_category_call` still takes `trait: Trait` so its caller signature stays unchanged; the param is unused inside the function but left to keep this changeset focused.
- `SemanticEndpointParameters.parameters` description fixed: replaced the misleading "the wrapper's polarity field handles negation" (no such field exists on `EndpointParameters`) with "polarity is applied downstream by the handler when bucketing the finding."
- `semantic.md`: removed the carver/qualifier framing paragraph, the duplicated always-populated-weight paragraph, and the entire "## Selectivity bar by role" section including the `{{SEMANTIC_SELECTIVITY_GUIDANCE}}` placeholder. Schema field descriptions remain the sole source of role/selectivity guidance.
- `prompt_builder.py`: removed the `render_semantic_selectivity_for_prompt` import and the `EndpointRoute.SEMANTIC` entry from `_ENDPOINT_PLACEHOLDER_RENDERERS`.
- `schemas/semantic_space_selectivity.py` deleted (no remaining consumers).

Verification: imports compile, `OUTPUT_SCHEMAS` populates with 40 entries (down from 80 — single-key storage), function signatures inspect-clean (`get_output_schema(category)`, `get_output_wrapper(endpoint, bucket, *, category)`, `build_endpoint_coroutine(route, wrapper, *, qdrant_client, restrict_to_movie_ids)`), semantic prompt clean of "carver" / "qualifier-style" / role-keyed selectivity section, length down ~17% (13.8k → 11.4k chars).

## Refactor category handler into query generation and temporary execution
Files: search_v2/endpoint_fetching/category_handlers/{handler.py,generated_endpoint_spec.py,prompt_builder.py,handler_result.py}, search_v2/full_pipeline_orchestrator.py, search_v2/endpoint_fetching/entity_query_generation.py, docs/modules/search_v2.md, docs/conventions_draft.md

Why: `category_handlers/handler.py` still implemented the stale HandlerResult bucketing/execution model and was bypassed by `full_pipeline_orchestrator.py`, which duplicated query-generation logic inline.
Approach: Added a shared `GeneratedEndpointSpec` dataclass module and moved initial operation-type derivation into `handler.py` as `determine_operation_type()`. Replaced old `run_handler()` with `run_query_generation(category_call, trait) -> list[GeneratedEndpointSpec]`, covering explicit no-op, TRENDING/MEDIA_TYPE deterministic generation, handler-LLM generation, fired-endpoint extraction via `output_extractor.py`, and soft-fail-to-empty behavior for handler LLM double failures. Added temporary `run_query_execution(spec) -> None`, executing only TRENDING and MEDIA_TYPE with logged soft failures and no-oping other routes. Simplified the orchestrator to delegate per-CategoryCall spec generation to `run_query_generation()` while keeping query-global reranker fallback/promotion logic in the orchestrator. Removed the unused category-handler `HandlerResult` file and stale comments/docs naming `run_handler`.
Design context: Keeps reranker-only fallback query-global, preserves list cardinality for category calls that emit zero/one/many endpoint specs, and leaves full stage-4 execution shape unresolved except for deterministic temporary hooks.
Testing notes: Did not read, edit, or run unit tests per project boundary. `uv run python -m py_compile search_v2/endpoint_fetching/category_handlers/handler.py search_v2/endpoint_fetching/category_handlers/generated_endpoint_spec.py search_v2/full_pipeline_orchestrator.py run_orchestrator.py` passed. Import smoke for handler, orchestrator, and run_orchestrator passed. Inline non-test checks confirmed TRENDING generation returns one no-param candidate-generator spec, MEDIA_TYPE shorts generation returns one candidate-generator spec, negative MEDIA_TYPE generation returns a pool-reranker spec, and positive SEMANTIC defaults to pool-reranker.

## Centralize operation-type-aware gating in build_endpoint_coroutine
Files: search_v2/endpoint_fetching/endpoint_executors.py, search_v2/endpoint_fetching/category_handlers/handler.py, run_search.py, run_search_json.py
Why: Pool rerankers with no candidate pool, missing-params specs, and finders that mistakenly receive a restrict pool all need uniform handling — pushing this into every executor would duplicate gating logic.
Approach: Changed `build_endpoint_coroutine` to take a `GeneratedEndpointSpec` (instead of separate `route`/`wrapper`). Added three pre-dispatch gates: (1) POOL_RERANKER + falsy `restrict_to_movie_ids` → empty `_empty_endpoint_result` coroutine; (2) `spec.params is None` → log warning + empty result; (3) CANDIDATE_GENERATOR → locally rebind `restrict_to_movie_ids = None` so finders never accidentally narrow to a pool (caller's set never mutated). Wired `run_query_execution` to delegate every non-TRENDING route through `build_endpoint_coroutine`, taking `qdrant_client` and `restrict_to_movie_ids` as new kwargs and dropping the explicit MEDIA_TYPE special case (now handled by the dispatcher). Updated diagnostic CLIs (`run_search.py`, `run_search_json.py`) to construct `GeneratedEndpointSpec(operation_type=CANDIDATE_GENERATOR)` at the call site — they always pass restrict=None so behavior is unchanged.
Testing notes: Did not modify tests per boundary. `python -m py_compile` passed for all four files; imports of `endpoint_executors` and `handler` succeed. Diagnostic CLIs hit a pre-existing `MatchMode` import error unrelated to this change (the symbol was already removed from `schemas/enums.py`).

## Promote shorts (<=40 min) to ReleaseFormat.SHORT
Files: scripts/backfill_short_release_format.py, movie_ingestion/final_ingestion/ingest_movie.py
Why: IMDB occasionally tags genuinely short content (<= 40 min, the standard short-film cutoff) as `movie` rather than `short`, leaking shorts into feature-only result sets.
Approach: (1) Added a runtime-based override at ingest in ingest_movie.py right after the IMDB title-type mapping — when `runtime_minutes <= 40`, force `release_format = ReleaseFormat.SHORT.release_format_id` regardless of what IMDB reported. (2) Wrote scripts/backfill_short_release_format.py as a one-off to apply the same rule to already-ingested rows: pulls tmdb_ids with `status='ingested'` from the SQLite tracker, then runs a single bulk UPDATE on public.movie_card filtered by `movie_id = ANY(...)` AND `runtime_minutes <= 40` AND `release_format <> SHORT`. Backfill updated 1439 rows.
Testing notes: ingest-time override is straightforward; backfill is idempotent (the `release_format <> SHORT` clause makes re-runs no-ops).

## Fix negative-trait scoring formula (gate-vs-OR failure modes)
Files: search_v2/stage_4_execution.py

### Intent
The `(∏ G_calls) × mean(R_calls)` formula in `_score_negative_trait` had a real failure mode on fuzzy negative concepts: "not scary" decomposed into KEYWORD:horror + SEMANTIC:scary penalized only horror-tagged movies, missing scary thrillers (Se7en, Black Swan) where the keyword tag has low recall. A naive switch to noisy-OR over everything would symmetrically break "not Joaquin Phoenix Joker" (PERSON_CREDIT + NAMED_CHARACTER + semantic), where pure OR over-penalizes Heath Ledger Jokers because semantic similarity to the JP-Joker vibe is high. The fix introduces the missing discriminator and corrects both failure modes.

### Key Decisions
- **Three-bin partition replacing two-bin**: classify each successful call as authoritative-G (G_a), evidential-G (G_e), or R. G_a covers categories with definitional membership and no recall gap (PERSON_CREDIT, NAMED_CHARACTER, AWARDS, structured metadata like RELEASE_DATE/RUNTIME/MEDIA_TYPE/CHRONOLOGICAL — 18 categories total). G_e covers fuzzy proxies (KEYWORD-style tags, GENRE, archetypes, TRENDING, CENTRAL_TOPIC, ELEMENT_PRESENCE). R is unchanged (would-be POOL_RERANKER per `determine_operation_type`).
- **New formula**: `gate × fuzzy` with `gate = ∏ G_a` (1.0 if absent) and `fuzzy = 1 − ∏(1 − s)` over G_e ∪ R (skipped if absent). Degenerate cases handled symmetrically: G_a-only → gate, fuzzy-only → fuzzy, all-empty → 0.0. Rationale: authoritative G gates because membership is settling; evidential G ORs with R because both are alternative evidence for the same fuzzy concept, with weak signals reinforcing.
- **Frozenset placement — consumer-co-located, not orchestrator-co-located**: original plan put `_AUTHORITATIVE_NEGATION_CATEGORIES` next to `_SEMANTIC_PROMOTION_TIERS` in full_pipeline_orchestrator.py for consistency with other classification tables. Hit a circular-import constraint — orchestrator imports `execute_branches` from stage_4_execution at load time, so importing the set the other way would cycle. Moved the frozenset into stage_4_execution.py with an inline note explaining why it lives there rather than with its peers.
- **Failed-call handling unchanged**: failed calls still drop from their bin (option A from prior review). A transient endpoint failure must not zero the gate or saturate the noisy-OR. Sign still applied at the §9 aggregation layer in `_aggregate_branch_scores`, not here.

### Planning Context
Design worked through in conversation with user covering both failure modes and validating the bin assignments cell-by-cell against (JP-Joker × {JP-as-Joker, Ledger Joker, JP in Her}) and (not-scary × {horror+scary, thriller scary, romcom}). Plan file at /Users/michaelkeohane/.claude/plans/implement-this-warm-pancake.md.

### Testing Notes
Tests under unit_tests/ are out of scope per project boundary. Verification path: replay "joker movies not joaquin phoenix" and a "not scary" query through `run_full_pipeline` in test_v3_endpoints.ipynb and confirm: (a) Ledger/Romero/Nicholson Jokers receive ≈0 violation while Phoenix Joker receives ≈1.0; (b) thriller-genre scary movies now receive non-zero violation. Single-G degenerate cases ("not Christopher Nolan", "not from 2024") should produce identical scores to before since gate alone returns the G_a value. `python -m py_compile` passes; no new imports beyond what was already in scope.

## Code-review follow-ups on negative-trait scoring fix
Files: search_v2/stage_4_execution.py, docs/modules/search_v2.md

Why: Code review surfaced two findings — both fixed in this changeset.

Approach:
1. Removed `CategoryName.CHRONOLOGICAL` from `_AUTHORITATIVE_NEGATION_CATEGORIES` (dead-code entry — `determine_operation_type` returns POOL_RERANKER for CHRONOLOGICAL+METADATA+POSITIVE, so the partition logic in `_score_negative_trait` never reaches the authoritative-set check for chronological calls). Replaced with an inline NOTE block explaining why CHRONOLOGICAL is intentionally excluded so a future reader does not re-add it. Fuzzy treatment is also semantically right for approximate phrasings ("old", "recent", "from the 80s").
2. Updated three stale sections of `docs/modules/search_v2.md` describing the old `(∏ G) × mean(R)` multiplicative-gated formula: rewrote the "Negative-trait scoring" section (renamed to "gate × noisy-OR, three-bin"), updated the `stage_4_execution.py` table description, and revised the gotcha bullet about `determine_operation_type` to mention the authoritative-vs-evidential split. Per `.claude/rules/docs-awareness.md`, module docs may be autonomously updated when stale.

Design context: Both findings came from /review-code; user accepted both ("Update all problems") and dismissed the third finding (magnitude inflation across traits with different decomposition depth) as not a concern. The dismissal preserves the deliberate "weak signals reinforce" property of noisy-OR — accepted that more decomposition produces higher trait magnitude.

Testing notes: `python -m py_compile` passes on stage_4_execution.py; grep confirms no remaining "multiplicative" / "mean(R)" / "gated multiplication" references in the changed files. Verification path described in the previous DIFF_CONTEXT entry still applies (notebook replay of "joker movies not joaquin phoenix" + "not scary").

## Surface positive/negative score breakdown in Stage-4 ranked output
Files: search_v2/stage_4_execution.py, run_orchestrator.py

Why: User wanted to see *why* a candidate ranked where it did without re-deriving from raw trait scores — the §9 final score is one number, but its provenance (how much positive contribution from cand-gen + positive rerankers vs. how much negative subtraction from negative rerankers) is the diagnostic signal that lets the user judge whether a movie should be there. Without the breakdown, debugging means walking the whole pipeline by hand.

Approach: Added `ScoreBreakdown` dataclass (`positive_total`, `negative_total`) and `score_breakdowns: dict[int, ScoreBreakdown]` field on `BranchRankedResults` parallel to `ranked`. By construction `positive_total + negative_total == score`. `_finalize_scores` was already iterating per-movie summing weighted contributions; split the accumulator into positive_total / negative_total based on `sign >= 0.0` (cand-gen contributions and positive-polarity pure-reranker contributions go positive; negative-polarity pure-reranker contributions go negative with sign already applied). Return type changed to `tuple[ranked_list, breakdowns_dict]`. `_run_branch` unpacks both and threads `score_breakdowns` into the `BranchRankedResults` it returns. `run_orchestrator.py` per-rank line now prints `[pos=±X.XXXX neg=±X.XXXX]` alongside `score=±X.XXXX`.

Design context: No other consumers of `_finalize_scores` or `BranchRankedResults.ranked` exist in the repo — return-shape change is safe. The breakdown computation is in-line in `_finalize_scores` rather than re-derived later because the same per-movie loop already had the necessary state (weights, signs, contributions) — doing it there is free; recomputing would re-walk every cand-gen execution and every pure-reranker score map.

Testing notes: `python -m py_compile` passes on both files. Smoke test verified `BranchRankedResults` and `ScoreBreakdown` instantiate with sane defaults. Live verification via `python run_orchestrator.py "<query>"` will surface the new bracket alongside each ranked title.

## ISO codes for Country/Language in metadata LLM schema
Files: implementation/classes/countries.py, implementation/classes/languages.py, schemas/metadata_translation.py, search_v2/endpoint_fetching/metadata_query_execution.py

Why: `TARGET_AUDIENCE` (and other Bucket-8 categories like `SENSITIVE_CONTENT`) was hitting OpenAI's 1000-enum-values-per-schema cap for structured outputs. Flat enum count was ~1193; the 334-member `Language` and 262-member `Country` enums (referenced as Pydantic field types in `AudioLanguageTranslation` / `CountryOfOriginTranslation`) accounted for 596 of those. Combined endpoint-trio buckets (KEYWORD + METADATA + SEMANTIC) compounded the cost.

Approach: Replaced the LLM-facing field types with `conlist(str, ...)` validated against ISO codes (3166-1 alpha-2 for countries, 639-1 for languages). Codes were added to the existing enums via a side-table dict (`_COUNTRY_NAME_TO_ISO`, `_LANGUAGE_NAME_TO_ISO`) stamped onto each member at module import — avoids editing the 596 enum-tuple lines. New `Country.from_iso()` / `Language.from_iso()` classmethods do reverse lookup (case-insensitive). Pydantic `field_validator(mode="after")` on the schema validates each code resolves and canonicalizes casing; unknown codes raise ValidationError, surfacing as a parse failure on the LLM-output retry path. Executor handlers in `metadata_query_execution.py` switched from `Language(v).language_id` to `Language.from_iso(v).language_id` (1-line change per handler). Long-tail entries without ISO codes (retired countries: Yugoslavia/USSR/Czechoslovakia/etc.; ~150 regional/sign/constructed languages) are unreachable via the LLM schema by design — they're <1% of real queries and remain accessible via the unchanged `country_from_string` / display-name lookup paths for non-LLM callers.

Design context: ISO 639-1 / 3166-1 alpha-2 chosen because LLM accuracy on these codes is essentially perfect (universal training-data familiarity), and uniform 2-letter casing gives a clean `^[a-z]{2}$` / `^[A-Z]{2}$` format. Mixed-length fallback to 639-3 / 3166-3 was rejected because the schema pattern would have to allow either length, degrading LLM accuracy. PALESTINE collides with PALESTINIAN_TERRITORIES on "PS" — the latter owns the code, the former intentionally gets `iso_code = None` (collision asserted at import). Kosovo uses "XK" (de facto IANA/EU code, not officially assigned by ISO).

Verification: TARGET_AUDIENCE flat enum count dropped 1193 → 597; SENSITIVE_CONTENT identical; every other category schema verified under 1000. Round-trip checks confirm ISO codes parse to correct enum members, case-insensitive normalization works, unknown / display-name inputs ("English", "USA") raise ValidationError, executor handlers produce the expected language_id / country_id arrays. JSON schema for both sub-objects is now enum-free.

Testing notes: This change only narrows the LLM-facing input shape. BaseMovie / DB / ingestion paths still use the full enum unchanged, so no migration needed. Existing test files (`test_languages.py`, `test_metadata_scoring.py`, `test_create_metadata_score.py`) use enum members directly and are unaffected by the schema-side change; not modified per test-boundaries rule.

### Follow-up refinements after code review
Files: schemas/metadata_translation.py, docs/modules/classes.md

Applied four review suggestions on the same change before commit. (1) Promoted the per-item format constraint into the JSON schema via `Annotated[str, StringConstraints(pattern=...)]` — `_LangISOCode` and `_CountryISOCode` ship `^[a-z]{2}$` / `^[A-Z]{2}$` patterns to OpenAI so the structured-output decoder is constrained to canonical-case 2-letter codes upstream of any Python-side validator. Tried `to_lower=True` / `to_upper=True` first; pydantic 2.12 evaluated `pattern` before case-conversion in this composition so the auto-conversion didn't actually run — dropped those knobs and kept strict canonical-case patterns as the cleaner LLM contract. (2) Extracted the duplicated validator body into a module-level `_validate_iso_codes(codes, from_iso, kind)` helper; each field validator is now a one-liner. (3) Tightened the `audio_language` description from "pick the closest covered language or omit the column" to "omit the column entirely — do not substitute a near-match" since audio language is an explicit user constraint where a wrong match is worse than no match (left the country description's closest-fallback wording alone — for retired entities like Yugoslavia, falling back to a current country is genuinely useful). (4) Updated `docs/modules/classes.md` to describe `iso_code`, `from_iso()`, and the LLM-schema contract on both `Country` and `Language`; corrected stale path reference (`search_v2/stage_3/...` → `search_v2/endpoint_fetching/...`).

Verification: TARGET_AUDIENCE flat enum count unchanged at 597; JSON schema items now carry `pattern` upstream of validation; pattern correctly rejects wrong-case and 3+ letter input; validator still rejects unknown 2-letter codes with helpful messages.

## Redesign implicit-prior policy output and orchestration
Files: schemas/implicit_expectations.py, search_v2/implicit_expectations.py, search_v2/full_pipeline_orchestrator.py

Why: The old implicit-expectations step was boolean (`should_apply_quality_prior` / `should_apply_notability_prior`) and still targeted a stale Step-2 fragment schema. The final search design needs direction + strength for both quality and popularity priors, reasoned from Step 2's committed trait layer rather than rediscovering criteria from the raw query.
Approach: Replaced the schema with observations-first policy output: one `ExplicitPriorSignal` per committed Step-2 trait, then whole-query ordering-axis analysis, query-specificity/prior-room analysis, then final `PriorDecision` records for quality and popularity. Rewrote `search_v2/implicit_expectations.py` to serialize `QueryAnalysis.intent_exploration` plus committed `Trait` fields (`surface_text`, `contextualized_phrase`, `evaluative_intent`, `qualifier_relation`, `anchor_reference`, `polarity`, `commitment`, `commitment_evidence`) and to use generalized guidance instead of examples. Hooked full-pipeline branch execution so the implicit-prior policy call starts immediately after Step 2 and runs in parallel with per-trait Step 3 decomposition / category-handler endpoint generation. Each `Step2BranchResult` now carries `implicit_expectations` or `implicit_expectations_error`; failures soft-fail per branch.
Design context: Raw query is retained only for provenance and ambiguity resolution. Step 2 traits are treated as the closed set of explicit user intent, which avoids re-litigating atom splits/merges and lets commitment density drive `prior_room`.
Testing notes: Did not read, edit, or run unit tests per boundary. `uv run python -m py_compile schemas/implicit_expectations.py search_v2/implicit_expectations.py search_v2/full_pipeline_orchestrator.py` passed.

## Apply implicit-prior multiplicative rerank and diagnostics
Files: search_v2/full_pipeline_orchestrator.py, search_v2/stage_4_execution.py, run_orchestrator.py

Why: The implicit-prior policy now emits strength/direction decisions, but Stage 4 output was still only base relevance. The final search needs quality/popularity priors to affect ranking when active, and the diagnostic runner needs to show both the policy reasoning and per-movie boost.
Approach: Added fixed boost tables in the full orchestrator (`quality: none/light/normal/strong = 0/2.5/6/10%`, `popularity = 0/5/12/20%`). After `execute_branches`, the orchestrator bulk-fetches `(popularity_score, reception_score)` for each branch's ranked movie IDs and applies `boosted_score = base_score * (1 + quality_cap * quality_signal + popularity_cap * popularity_signal)`, with inverse directions using `1 - normalized_signal`. Missing reception is neutral 0.5; missing popularity is 0.0. The reranked list is resorted, and `ScoreBreakdown.implicit_prior_boost` records the boost fraction for diagnostics. `run_orchestrator.py` now prints the full implicit policy reasoning and renders ranked results as a table with final score, boost %, positive base contribution, and negative base contribution.
Design context: This is a post-score rerank so implicit priors remain relevance-scaled tie/ordering pressure rather than additional trait scores. Popularity has exactly double the cap of quality at every active strength level.
Testing notes: Did not run unit tests per boundary. `uv run python -m py_compile search_v2/full_pipeline_orchestrator.py search_v2/stage_4_execution.py run_orchestrator.py` passed. Import smoke for `search_v2.full_pipeline_orchestrator` and `run_orchestrator` passed. `git diff --check` passed for changed files.

## Follow-up fixes from implicit-prior review
Files: search_v2/full_pipeline_orchestrator.py, db/postgres.py, run_search.py, run_search_json.py, docs/modules/search_v2.md

Why: Review found stale diagnostic consumers of the old boolean implicit-prior schema, missing popularity receiving max inverse boost, and stale module docs. User clarified missing data should have no effect and asked to defer the negative-base-score boost semantics for further design discussion.
Approach: Changed `_quality_signal()` and `_popularity_signal()` so `None` returns 0.0 before any positive/inverse direction logic; updated the Postgres helper docstring to match. Replaced old `should_apply_*` display reads in `run_search.py` and `run_search_json.py` with `direction/strength` summaries from the new `PriorDecision` fields. Updated `docs/modules/search_v2.md` to describe the implicit-prior policy step, parallel orchestration, post-Stage-4 multiplicative rerank, boost caps, inverse-signal handling, and missing-data behavior.
Design context: Did not alter how boosts interact with negative base scores; that remains an open design choice to decide deliberately.
Testing notes: Did not run unit tests per boundary. `uv run python -m py_compile search_v2/full_pipeline_orchestrator.py db/postgres.py run_search.py run_search_json.py` passed. `git diff --check` passed for changed files.

## Refine implicit-prior boost base for pure-negative queries
Files: search_v2/full_pipeline_orchestrator.py, docs/modules/search_v2.md

Why: Multiplying the full base score by `(1 + boost)` makes negative base scores more negative. Pure-negative queries such as "not scary" also need a neutral positive surface so quality/popularity can break ties among non-violators without magnifying penalties.
Approach: Changed the post-rerank formula to `final = base_score + prior_base * boost`, where `prior_base = positive_total` when a candidate has positive contribution and `1.0` otherwise. This preserves positive relevance amplification for normal queries and gives pure-negative / no-positive-contribution cases a neutral prior surface. Updated module docs with the exact formula.
Design context: Negative penalties remain additive and untouched. The implicit prior now adds only the lift amount rather than multiplying the full signed score.
Testing notes: Did not run unit tests per boundary. Verification pending in the next compile/check pass.

## Align implicit-prior signal shape with metadata endpoint
Files: search_v2/full_pipeline_orchestrator.py, search_v2/endpoint_fetching/metadata_query_execution.py, docs/modules/search_v2.md

Why: Implicit-prior scoring was using fully continuous linear normalization while explicit metadata popularity/reception scoring uses threshold-anchored sigmoid curves. User asked for implicit priors to match metadata endpoint scoring.
Approach: Added public `score_popularity_prior()` and `score_reception_prior()` wrappers around the metadata endpoint's existing sigmoid scorers, then switched full-orchestrator implicit signals to use those wrappers. Positive quality uses `ReceptionMode.WELL_RECEIVED`; inverse quality uses `ReceptionMode.POORLY_RECEIVED`; positive popularity uses `PopularityMode.POPULAR`; inverse popularity uses `PopularityMode.NICHE`. Missing data still returns 0.0 before scoring. Updated module docs to describe the shared sigmoid shape.
Design context: Public wrappers avoid importing private underscored helpers across module boundaries while preserving one source of truth for threshold centers/slopes.
Testing notes: Did not run unit tests per boundary. Verification pending in the next compile/check pass.

## Add Step 0 CLI runner
Files: search_v2/run_step_0.py | New thin CLI wrapper mirroring run_step_2.py — takes optional query arg, calls run_step_0, prints full Step0Response as indented JSON plus locally-measured elapsed time and token usage (run_step_0 returns a 3-tuple without elapsed, unlike run_step_2's 4-tuple).

## Add explicit release_year to Step 0 flow payloads
Files: schemas/step_0_flow_routing.py, search_v2/step_0.py
Why: Downstream exact-title and similarity searches benefit from a (title, year) pair when the user explicitly disambiguates a remake or franchise installment ("Dune 2021", "Halloween (2018)"). Year must never be inferred — silent over-restriction would drop valid results.
Approach: Added `release_year: int | None = None` to ExactTitleFlowData and SimilarityFlowData. Extended the system prompt's flow-eligibility section and OUTPUT field guidance with a hard rule that the year is only carried over from explicit user statement (never inferred from plot, sequel numbering, or franchise knowledge). Added boundary examples 11–15 covering: explicit year on exact-title, parenthesized year, descriptive Indy reference where year MUST NOT be inferred, explicit year inside a similarity frame, and sequel-number ("Top Gun 2") which is not a year.
Testing notes: py_compile passed for both files. New schema field defaults to None so existing callers/serialized payloads stay valid.

## Add exact-title search flow executor
Files: search_v2/exact_title_search.py, db/postgres.py

### Intent
Step 0's exact_title flow had no executor — it decided the flow fires but nothing downstream consumed the (title, optional release_year) payload. Built a standalone module that produces ranked (movie_id, score) results from the payload, decoupled from the standard search pipeline.

### Key Decisions
- Six-tier scoring scheme requested by the user, taking the max across contributions per candidate: 1.0 seed (title and year, when year given), 0.75 seed-lineage→cand-lineage, 0.625 seed-lineage→cand-universe, 0.5 title-only year-mismatch, 0.25 seed-universe→cand-universe, 0.125 seed-universe→cand-lineage. Seeds keep 1.0 unconditionally; title-only candidates can be lifted by franchise passes via the same max.
- Reused existing read helpers verbatim: fetch_movie_ids_with_title_like (exact-equality LIKE on normalize_string + escape_like output), fetch_movie_cards (release_ts → year via UTC fromtimestamp), and fetch_franchise_movie_ids called twice — once seeded with the seeds' lineage entry IDs and once with their universe entry IDs. The function's existing (lineage_matched, universe_only_matched) split is exactly the four-bucket partition the scoring scheme needs.
- Added one new helper to db/postgres.py — fetch_franchise_entries_for_movies — because fetch_movie_cards intentionally omits lineage_entry_ids and shared_universe_entry_ids from its SELECT. This is purely additive and placed adjacent to the other franchise read helpers.
- Year is taken only from flow_data.release_year — never re-derived. When a year is supplied but no title-match satisfies it, seeds is empty, the franchise fan-out is skipped, and only title-only candidates at 0.5 are returned. This mirrors Step 0's "never infer the year" rule (the Indy-boulder example).
- Output dataclass ExactTitleSearchResult.ranked mirrors the (movie_id, score) shape of search_v2.stage_4_execution.BranchRankedResults.ranked so any future integration into the orchestrator stays cheap; deliberately did not wire it in (user requested no edits to existing search code).

### Testing Notes
py_compile passed for both files. Import smoke (`from search_v2.exact_title_search import run_exact_title_search`) passed. Live verification deferred — ideal smoke is "Star Wars" (no year → all SW films at 1.0, lineage at 0.75) and "Dune" + 2021 (year-exact at 1.0, 1984 Lynch at 0.75 via lineage lift, no-match year → only title-only at 0.5). No tests added per test-boundary rule.

## Wire exact-title search into Step 0 runner
Files: search_v2/run_step_0.py | After Step 0 runs, conditionally invokes run_exact_title_search when exact_title_flow_data.should_be_searched is True; lazy-opens the Postgres pool, bulk-fetches movie cards for the top 25 ranked IDs, and prints a compact table (rank, score, source label, year, title). Added explicit load_dotenv at the top so import order doesn't break db.postgres' module-level conninfo build.

## Fix Step 0 hallucination on franchise umbrella names
Files: search_v2/step_0.py
Why: Bare franchise queries like "indiana jones" were being routed to exact_title with most_likely_canonical_title="Indiana Jones" — a string no movie in the index actually bears. My own example 13 (added in the release_year change) embedded the bad behavior by listing "Indiana Jones" as the canonical title for the boulder-description query.
Approach: Added a hard rule to _TITLE_OBSERVATION_RULES distinguishing franchise umbrellas (no individual installment shares the umbrella name) from real film titles. Umbrellas with no installment marker emit no TitleObservation; the umbrella string is recorded as a qualifier so the standard flow still gets the search signal. The rule explicitly carves out cases where the umbrella IS a real film's canonical title (Star Wars, Halloween, Spider-Man) — those still fire exact_title. Rewrote example 13 to drop the bogus title observation, and added examples 16 ("indiana jones" → standard) and 17 ("star wars" → exact_title) to lock in both sides of the boundary.
Verification: Live re-runs confirm "indiana jones" → titles_observed=[], primary_flow=standard, no exact-title search; "star wars" → exact_title fires and franchise expansion produces the expected lineage-rich top 25 (Star Wars at 1.0, sequels and prequels and lineage docs at 0.75).

## Surface per-trait + per-category score breakdown in run_orchestrator
Files: search_v2/stage_4_execution.py, run_orchestrator.py
Why: The CLI runner only showed each trait's signed weighted contribution as a single column. When a trait routes through multiple categories (positive trait_score = max over per-category combine_calls), we couldn't see which category drove it without re-running the pipeline mentally. User asked for the full per-trait + per-category view on the top 25.
Approach: Added a `CategoryScore` dataclass (category_name, combine_type, score) and extended `TraitContribution` with `trait_score`, `weight`, and `category_scores`. `_score_positive_trait` now also returns `dict[mid → list[CategoryScore]]` collected during the same loop that already computed combine_calls per category — no extra work in the hot path. Threaded the per-category map through Phase D's `_TraitPayload` 5-tuple to `_finalize_scores`, which writes it into the per-trait breakdown. Negative traits pass an empty dict because their gate × fuzzy formula partitions calls cross-category, so a per-category score is not well-defined; the trait-level contribution is still displayed. Replaced the markdown table in `run_orchestrator.py` with a per-result block: header line with final/boost/pos/neg/title, indented trait line per contribution, double-indented bullet per category.
Testing notes: ast/import smoke passed. No tests touched per test-boundary rule. Live verification deferred to user run.

## Add per-endpoint score breakdown to score tree
Files: search_v2/stage_4_execution.py, run_orchestrator.py
Why: Per-trait + per-category was helpful but cut off above the call layer. Couldn't see which endpoint inside a category drove its combine_calls fold. User wants the whole scoring tree visible, including per-endpoint scores and the human-language ask (expressions + retrieval_intent) attached to each category.
Approach: Added `EndpointScore` dataclass (route, score) and extended `CategoryScore` with `expressions`, `retrieval_intent`, and `endpoint_scores`. `_score_positive_trait` now keeps `(spec, scores_map)` pairs in `live_cats` instead of bare maps so per-mid extraction can label each call by route in the same loop that already feeds combine_calls. Added `CategoryCallWithEndpoints` to the TYPE_CHECKING block since live_cats now references it. run_orchestrator.py now prints a four-level tree: result header → trait → category (with expressions + retrieval_intent + cat_score) → endpoint (route + score). Per-endpoint and per-category rows are populated for positive traits only (negatives' gate × fuzzy is cross-category).
Testing notes: ast/import smoke passed. No tests touched per test-boundary rule.

## Standalone similar-movies search flow
Files: search_v2/similar_movies.py, search_v2/similar_studio_registry.py, db/postgres.py, search_v2/run_step_0.py, search_v2/full_pipeline_orchestrator.py, search_v2/test_similar_movies.ipynb

### Intent
Implement the separate "movies like X" flow from `search_improvement_planning/similar_movies.md` without changing the standard Stage-4 search/reranking path. Supports both direct TMDB-anchor debugging and Step-0 similarity routing.

### Key Decisions
- Added `run_similar_movies_for_ids()` and `run_similarity_search()` in a new standalone module. Candidate generation is lane-based: shape, director, franchise, studio, source, and quality.
- Shape lane queries Qdrant directly with existing named vectors, alias, and `QDRANT_SEARCH_PARAMS`. Single-anchor search uses each anchor vector; multi-anchor search builds normalized centroid vectors and cohesion-weighted space weights. Final merged shape scores are max-normalized so the documented weave thresholds operate on a true lane scale.
- Metadata lanes use new narrow bulk Postgres helpers only. No tracker fallback: source similarity uses `movie_card.source_material_type_ids`; studio similarity uses curated normalized production-company strings resolved through `lex.production_company`.
- Middle-bucket anchors set quality raw weight to zero and renormalize. Cult/prestige anchors activate quality scoring with the formulas from the planning doc.
- Step-0 similarity routing now executes the standalone flow in both `search_v2.run_step_0` and `run_full_pipeline` as a side result. Standard branch planning and Stage-4 scoring are unchanged.

### Testing Notes
Compile/import smoke passed for new and touched modules. Notebook JSON loads. Live smoke checks ran against local Postgres/Qdrant for Inception, a Nolan multi-anchor set, Star Wars/franchise, Oppenheimer/prestige, Sharknado/cult-ish, and `python -m search_v2.run_step_0 "movies like Inception"`; all returned lane-labeled results and lane weights summed to 1.0. Per repo test-boundary rule, no `unit_tests/` files were read, edited, or run.

## Similar-movies batch runner and smoke report
Files: search_v2/run_similar_movies_batch.py, search_v2/similar_movies.py, search_v2/similar_movies_batch_results.json, search_v2/similar_movies_batch_results.md

Why: User wanted a reusable Python module that can run the suggested TMDB anchor set directly and show the actual similar-movies outputs instead of only describing expected examples.
Approach: Added a CLI module with a broad default anchor list, configurable `--ids` and `--limit`, and JSON/Markdown report writers. The runner opens the Postgres pool, executes the standalone similarity flow for each anchor, bulk-fetches display cards, and prints lane-labeled compact tables. Also clamped direct Qdrant similarity scores to `1.0` before local lane normalization to tolerate small numeric overshoots without changing shared scoring helpers or the main search path.
Testing notes: `uv run python -m py_compile search_v2/similar_movies.py search_v2/run_similar_movies_batch.py` passed. Live run `uv run python -m search_v2.run_similar_movies_batch --limit 5` completed for the full default anchor set and wrote both report files. No `unit_tests/` files were read, edited, or run.

## V2 production medium registry + V2 similar-movies materialized views
Files: search_v2/production_medium_registry.py, db/init/01_create_postgres_tables.sql, db/postgres.py, movie_ingestion/final_ingestion/ingest_movie.py

### Intent
Lay down the data-layer prerequisites for the V2 similar-movies redesign in `search_improvement_planning/similar_movies.md` before touching any lane code: a shared production-medium similarity table importable by `search_v2/similar_movies.py`, plus three new materialized views that back the V2 director auteur prior, franchise confidence prior, and IDF-over-traits lanes (source / themes / medium). No V2 lane code is wired up yet — this round only populates the lookups and refreshes them on ingest.

### Key Decisions
- New module `search_v2/production_medium_registry.py` mirrors the existing `similar_studio_registry.py` convention (top-level frozenset + dicts, no class wrapping). Keys reference `OverallKeyword.<member>.keyword_id` instead of hardcoded ints so an enum rename surfaces at import time. Exposes `MEDIUM_TAG_IDS`, `MEDIUM_SIMILARITY` (8x8 symmetric table verbatim from the spec), and a single `medium_score(anchor_tags, candidate_tags)` helper that returns the max similarity across anchor x candidate pairs (per spec: parent ANIMATION absorbs sub-type differences). The multiplier formula `0.85 + 0.15 * score` lives in lane code, not in the registry.
- Three new MVs in `public.*` matching the existing `mv_popularity_percentile` schema: `mv_director_strength` (per-director percentile-rank of 0.8*pop + 0.2*recep, restricted to directors with >= 2 films), `mv_franchise_confidence` (per-lineage confidence + consistency, where consistency is `1 - clamp(2*stddev, 0, 1)` and single-film franchises get 1.0), and a unified `mv_trait_idf` covering all four trait families (overall_keyword / concept_tag / tmdb_genre / source_material) with a SMALLINT `trait_kind` discriminator. IDF normalized as `log(N/df) / log(N)` so values stay in `[0, 1]` regardless of catalog size.
- Chose a single unified trait-IDF MV over four per-source MVs (decided in plan-mode AskUserQuestion). Avoids duplicate aggregation passes over `movie_card`; lane code reads with a `trait_kind` filter. The medium-IDF retrieval gate becomes a filtered subset of trait_kind=1 (no separate medium MV).
- Three refresh helpers in `db/postgres.py`, all using `_execute_write` and `REFRESH MATERIALIZED VIEW CONCURRENTLY` with the existing pattern. Added `TRAIT_KIND_*` constants in the same module to keep Python and SQL discriminator values in lockstep.
- Refresh order in `ingest_movie.py` post-ingest block: phase 1 unchanged (lex DFs + popularity), phase 2 adds `refresh_director_strength` + `refresh_franchise_confidence` (both depend on freshly-rebuilt `mv_popularity_percentile`), phase 3 adds `refresh_trait_idf` (independent but kept serial for simple failure semantics).

### Planning Context
Design decisions come from the V2 spec finalized earlier in the same session. Plan file at `/Users/michaelkeohane/.claude/plans/dapper-dazzling-rabbit.md` records the four AskUserQuestion answers (all 5 MVs, unified trait-IDF shape, `1 - clamp(2*stddev, 0, 1)` consistency formula, new file under `search_v2/`).

### Testing Notes
No lane code consumes any of these MVs yet, so live verification is limited to:
  1. Re-running `db/init/01_create_postgres_tables.sql` against a Postgres instance (DDL is idempotent via `IF NOT EXISTS`).
  2. Calling each refresh helper from a Python REPL and confirming no errors.
  3. Spot-checking populated values (top-strength directors should be recognizable auteurs; LotR/MCU lineages should land high-confidence + high-consistency; common traits like DRAMA collapse to near-zero IDF while rare ones like FOLK_HORROR sit near 1.0).
Per the test-boundaries rule, no `unit_tests/` files were read, edited, or run.

## V2 similar-movies full lane rework
Files: db/postgres.py, search_v2/similar_movies.py, search_v2/format_registry.py, search_v2/country_language_registry.py, search_v2/award_taxonomy.py, search_v2/production_medium_registry.py, search_v2/run_similar_movies_batch.py

### Intent
Complete the V2 redesign from `search_improvement_planning/similar_movies.md` by wiring every remaining lane change into the live similar-movies flow. Single-anchor and multi-anchor paths now use V2 vector base weights, IDF source/themes lanes, multiplicative studio/medium/low-confidence-franchise multipliers, auteur-prior director lane, unified always-on quality lane, format lane with top-5 weaving constraint, country/language coherence multiplier, top-3 cast lane, specific-award taxonomy lane, and a low-cohesion fallback for chaotic anchor sets.

### Key Decisions
- New per-flow vector weight maps (`VECTOR_BASE_WEIGHTS_SINGLE` raises narrative_techniques and lowers production per V2 spec; multi keeps the V1 tiered set since cohesion does the heavy lifting). Tier groupings are gone.
- `LaneName` extended with `format`, `themes`, `cast`, `specific_award`. Studio kept in `LaneName` for debug visibility but excluded from `ADDITIVE_LANES` — V2 makes studio a multiplier on shape-qualifying candidates (`shape >= 0.60` → `*= 1 + 0.10 * studio_score`). The `studio_lineage` anchor type stays as a flag with no weight delta.
- `_apply_post_weight_multipliers` semantics live inline in `_build_results` (one call site, kept readable). Multipliers stack in order: studio → low-confidence-franchise → country/language → medium. Worst-case stack: `1.10 * 1.10 * 1.10 * 1.0 ≈ 1.33`. Documented in code comment.
- Franchise lane splits on confidence: high-confidence lineages (`confidence >= 0.65 AND consistency >= 0.6`) run additively with the V1 `franchise_dominant` adjustment; low-confidence lineages drop to a multiplicative path with shape gate 0.55 and the anchor-type adjustment is suppressed. Per-anchor `franchise_high_confidence: bool` is threaded through `_single_anchor_lane_weights`.
- Source lane (single + multi) uses `max(idf(t in shared))` against `mv_trait_idf` filtered to `trait_kind=4`. Multi-anchor extends `_score_multi_trait_count` with an optional `weight_fn` so per-anchor matches contribute `max idf of shared types` rather than a constant 1.0. Source anchor adjustment dropped from V1 +0.14 to V2 +0.08 because IDF already discounts common types.
- Quality lane is always-on with bucket-specific formulas: cult_garbage (`0.40*low_recep + 0.50*pop_match + 0.10*razzie`), prestige (`0.80*high_recep + 0.20*pop_or_award + 0.20*non_razzie`), middle (`0.80*pop + 0.20*recep`). Replaced `fetch_similarity_award_prestige_scores` with `fetch_similarity_award_signals` returning a `SimilarityAwardSignals` dataclass that carries both razzie and non-razzie signals in one query.
- Director lane uses `mv_director_strength` for the per-director auteur prior; new `director_signature` anchor type fires when any anchor director clears strength >= 0.80 (Tarantino, Nolan, Scorsese, Spielberg, Miyazaki, etc.).
- Format lane: binary same-bucket score plus a top-5 weaving lock (`enforce_format_top_lock` parameter) — top 5 entries must share the anchor format bucket; positions 6+ may include cross-format. Multi-anchor enforces this only when ≥2 anchors share a bucket. Also dropped V1's `weak_studio_source_count` weaving rule since studio is no longer dominant.
- Themes lane (multi-only): IDF-mass-share over the union of `keyword_ids - country - medium - format` ∪ `concept_tag_ids` ∪ `genre_ids`, restricted to traits repeated by ≥2 anchors. Trait-kind discriminator threaded into the IDF fetch via `_multi_anchor_themes_repeated`.
- Specific-award lane (multi-only): three-tier scoring on `movie_awards.category_tag_ids` with a specificity discount on cohesion (L0 fully specific, L1 = 0.6, L2 = 0.3) and tier-weighted candidate scoring (L0=1.00, L1=0.50, L2=0.20). New `search_v2/award_taxonomy.py` derives the level from the numeric ID range so no enum lookup is needed.
- Low-cohesion fallback: when `mean_pairwise_cosine < 0.35 AND every metadata cohesion < 1.0`, fall back to round-robin per-anchor single-anchor results with `limit = ceil(target * 1.2 / N)` per anchor. UI presents results identically; debug payload sets `low_cohesion_fallback_used = True`.
- Selective rare-medium retrieval: anchor medium tags with `idf >= 0.50` (excluding LIVE_ACTION) trigger `fetch_movie_ids_by_overall_keywords` to seed additional candidates so e.g. a stop-motion anchor surfaces other stop-motion films. Medium IDFs lazy-loaded once per process via `load_medium_idfs()` in `production_medium_registry.py`.
- New data-layer fetchers in `db/postgres.py`: `fetch_director_strengths`, `fetch_franchise_confidence`, `fetch_trait_idfs`, `fetch_movie_ids_by_overall_keywords`, `fetch_similarity_top_billed_cast`, `fetch_similarity_award_category_tags`, `fetch_similarity_award_signals`. Extended `fetch_similarity_signal_rows` to include `keyword_ids`, `concept_tag_ids`, `genre_ids`, `runtime_minutes`. Old `fetch_similarity_award_prestige_scores` deleted (sole caller updated).
- New registries: `search_v2/format_registry.py` (bucket priority: mockumentary > performance > news > tv_format > short > documentary > narrative_feature), `search_v2/country_language_registry.py` (30 nation/language `OverallKeyword` IDs + `US_DEFAULT` fallback), `search_v2/award_taxonomy.py` (level + tier weights). All registries derive keys from `OverallKeyword.<member>.keyword_id` so an enum rename surfaces at import time.
- `SimilarMoviesDebug` extended with V2 audit fields: `anchor_format_bucket`, `anchor_medium_tags`, `franchise_high_confidence`, `consensus_countries`, `low_cohesion_fallback_used`, `per_anchor_active_anchor_types`. None of these fields participate in scoring; they're for diagnosing centroid drift and similar failures.
- `LANES` constant renamed to `ALL_LANES` (added `ADDITIVE_LANES` alongside). Updated `run_similar_movies_batch.py` to import the new name.

### Planning Context
Plan at `/Users/michaelkeohane/.claude/plans/dapper-dazzling-rabbit.md`. Four AskUserQuestion answers fed into the design: studio kept as debug flag (not removed), multi-anchor batch runner uses ID-count-based dispatch (no flag needed), multi-anchor source lane upgraded to IDF-weighted, Razzie data unified into `fetch_similarity_award_signals` rather than a parallel fetcher.

### Testing Notes
Live single-anchor smoke confirmed end-to-end: Inception (director_signature active, top match Tenet via Nolan strength), LotR Fellowship (top 5 = Hobbit + LotR sequels with full lineage, studio=New Line, source=novel IDF 0.20), Pulp Fiction (Reservoir Dogs at top via Tarantino prior). Live multi-anchor smoke: LotR trilogy (shape_raw = 1.20 max from cohesion ≈ 0.97; top 4 are franchise sequels), Best Picture trio (specific_award lane firing strongly across L0 BEST_PICTURE matches), Korean trio (consensus_countries = {106 KOREAN}; +10% boost for matching candidates), Stephen King trio (source IDF 0.20 only — no longer dominating; themes lane carries the horror signal). Low-cohesion fallback gate verified to NOT fire on partially-cohesive sets (Toy Story + Godfather + Sharknado share THRILLER → themes_cohesion ≥ 1.0 → fallback correctly suppressed). No `unit_tests/` files were read, edited, or run.

## Review fixes for V2 similar-movies
Files: search_v2/similar_movies.py, db/postgres.py

### Intent
Address three findings from the post-implementation review of the V2 lane rework: one high-severity bug (themes lane conflated trait IDs across families), one efficiency issue (anchor cast/award fetched in a separate await before the main gather), and one type-correctness cleanup (`fetch_trait_idfs` parameter array type).

### Key Decisions
- Kind-namespaced themes lane traits: `_themes_traits_for_movie` now returns `set[tuple[int, int]]` of `(kind, trait_id)` pairs. The `Genre.HORROR.genre_id = 14` / `OverallKeyword.BASEBALL.keyword_id = 14` collision (and many like it across the keyword/concept/genre families) was producing wrong cohesion counts and wrong IDF lookups in the multi-anchor themes lane. `_multi_anchor_themes_repeated` simplified — no more disambiguation step needed since the kind is on the key. `themes_idfs` flatten step removed; the IDF fetch result already carries `(kind, trait_id)` tuples.
- Anchor cast / award fetches folded into the main multi-anchor `asyncio.gather`. Cast/specific_award cohesion is now computed AFTER that gather (the only consumer is `raw_lane_weights`, which doesn't need to exist before the candidate-generation tasks fire). Saves one round-trip on the critical path. Smaller candidate-lane tasks still gate on the sync cohesion derived from anchor_rows alone.
- `fetch_trait_idfs` parameter array changed from `bigint[]` to `int[]`. The MV's `trait_id` column is INT (every UNION branch yields INT[] elements), so `int[]` is the correct width. Postgres was auto-widening before, but the type pun was confusing.

### Testing Notes
Live verification: LotR trilogy still surfaces sequels at top with `cast_cohesion = 2.0` and `specific_award_cohesion = 2.0` (max amplification). Single-anchor Inception unchanged. `fetch_trait_idfs` direct call returns the expected (kind, trait_id, idf) rows across all four kinds — `bigint[]` was working via Postgres auto-widening, the change to `int[]` is type-correctness only.

## V3 categorization fixes — DOCUDRAMA / TRUE_CRIME / SKETCH_COMEDY / ADULT_ANIMATION / HOLIDAY_ANIMATION
Files: search_v2/format_registry.py, search_v2/production_medium_registry.py
Why: V2 testing surfaced systematic miscategorization in two registries. Catalog audit confirmed: of 30 prestige DOCUDRAMA samples 27 had no actual documentary tag (Schindler's List, GoodFellas, Oppenheimer, Pianist, Spotlight, Irishman, Zero Dark Thirty, etc.); same pattern for TRUE_CRIME (28/30 narrative crime dramas) and SKETCH_COMEDY (13/13 narrative features incl. Monty Python and the Holy Grail). Production medium ADULT_ANIMATION and HOLIDAY_ANIMATION are audience/theme tags, not techniques — co-occur with HD/CG/STOP across the catalog and were collapsing the medium matrix's two orthogonal axes (technique × audience) into one.
Approach: removed the three content tags from format buckets (DOCUDRAMA + TRUE_CRIME from documentary, SKETCH_COMEDY from tv_format) and the two audience tags from MEDIUM_TAG_IDS / MEDIUM_SIMILARITY (8x8 → 6x6 technique-only matrix). All five tags remain in the catalog and are now naturally available to the themes lane via the keyword pool (FORMAT_KEYWORD_IDS_ALL / MEDIUM_TAG_IDS shrunk → exclusion subtraction in _themes_traits_for_movie strips fewer tags). No code changes outside the registries.
Design context: see search_improvement_planning/similar_movies_v3_plan.md §1.1 / §1.2 for the audit data, and similar_movies_v2_results.md cross-cutting finding F1 for the original observation.
Testing notes: smoke-tested format_bucket() against the affected tags — DOCUDRAMA-only / TRUE_CRIME-only / SKETCH_COMEDY-only inputs now return narrative_feature; DOCUMENTARY-only still returns documentary; mockumentary > short > documentary priority unchanged. medium_score(LIVE, CG)=0.0 and medium_score(HD, ANIME)=0.85 unchanged. Re-ran the V2 batch on Oppenheimer / Godfather / Barbie / TDK Rises / Toy Story / Star Wars: Oppenheimer top 1 is now Fat Man and Little Boy (was buried at #5); top 5 is biopic narratives + Nolan-director adjacents instead of literal Manhattan-Project documentaries (Einstein and the Bomb, Oppenheimer Real Story, etc. demoted to #7 and #9). No regressions observed on the other anchors. Other registry references in search_v2/endpoint_fetching/keyword_query_generation.py are unchanged — that's the standard search pipeline, separate from the similar_movies flow, and uses keyword names as query predicates rather than format/medium grouping.

## V3 plan refinements — cast generic formula, rare-keyword lane, canonical V3 spec section
Files: search_improvement_planning/similar_movies_v3_plan.md, search_improvement_planning/similar_movies.md, docs/TODO.md

### Intent
Lock in the design for the remaining V3 lane reworks after a back-and-forth conversation, and promote the V3 plan from working-doc-only to a canonical V3 Planned Changes section in similar_movies.md (mirroring the existing V2 Planned Changes section). The V3 plan doc remains the working artifact with audit data and implementation status; the canonical spec now carries the durable design.

### Key Decisions

**Cast lane — generic N-anchor bucket-with-floor (§2.5)**: replaced V2's hardcoded-3 weave reservation with a formula parameterized by N anchors and per-actor M anchor counts. Lane silent in single-anchor; in multi-anchor, lane weight scales `0.05 + 0.10 * ratio` and a bucket floor `0.25 + 0.20 * ratio` activates per-candidate when shape ≥ 0.30 and they match a shared lead in their own top-3 billing. Top-3 billing required on both sides per user direction. Floor only pulls weak-shape candidates up to the floor; doesn't touch already-strong candidates (matches "kinda like director before but don't let it get too crazy").

**Rare-keyword lane — NEW lane (§2.6)**: parallel to themes, operates on the same trait pool with three rarity tiers. Low (IDF < 2.5) folds into themes lane. Moderate (2.5–4.5) adds individual contributions at IDF × 0.03. High (IDF ≥ 4.5) adds individual contributions at IDF × 0.05 AND counts toward a floor trigger (single super-rare ≥ 5.0 OR combo sum ≥ 7.0). Floor magnitude `0.40 + 0.05 * high_count` capped at 0.55, gated on shape ≥ 0.30. Pool = concept tags + non-registry overall keywords + TMDB genres (per user: "include genres for now. Worst case they don't contribute as much when they're frequent."), excluding format / medium / country / source / award tags handled by their own lanes.

**Auteur list as essential blocker**: deferred composition to a follow-up conversation per user direction ("save this for a followup conversation just to avoid context bloat"). Added a 🛑 callout to v3 plan §2.1, marked priority-table item 7 BLOCKED, and added a docs/TODO.md entry with a hard pause rule: any director-lane implementation must surface the gap before proceeding rather than stub or fall back to V2's mv_director_strength MV.

**Shorts harsh downrank in §3.1**: replaced the previous "hard exclude wrong-format from full top 10" framing with a multiplier (0.30 on combined score for short candidates against non-short anchors) plus a structural max-1 hard cap. Multi-anchor "moderate cohesion" defined as ≥50% of anchors sharing the bucket; below that threshold, apply the same multiplier + cap. Short-anchor case keeps a soft top-1 lock rather than full coherence (catalog's shorts skew franchise-tied).

**Franchise as 2D matrix (§2.2)**: replaced the linear T1-T7 tier list with a role × overlap lookup table (mainline / spinoff / crossover × same-subgroup / same-lineage / same-universe / disjoint). Boldface cells encode user's "role consistency" principle — spinoff↔spinoff same lineage = 0.85, crossover↔crossover same lineage = 0.85, mainline↔mainline same subgroup = 1.00. The franchise_confidence ≥ 0.6 gate is dropped entirely (Star Wars's "low consistency" was a measurement artifact over the sequels-and-spinoffs tail, irrelevant once we score by structure not statistical variance).

**V3 canonical spec section in similar_movies.md**: appended a "# V3 Planned Changes" section mirroring the existing V2 Planned Changes structure. Includes the design for each lane/multiplier change, an 11-row hypotheses table pairing each V3 change with its target failure case and observable verification, and a 10-criterion success rubric for the benchmark re-run. The v3 plan doc remains the implementation-status tracker; the canonical spec now carries the durable design contract.

### Planning Context
Full audit and detailed reasoning live in similar_movies_v3_plan.md. The conversation that produced these decisions covered: short-anchor handling (option C — soft top-1 lock with feature fill), franchise data verification (need to confirm Avengers carries multiple lineage_entry_ids), cast top-3 billing on both sides (not just lead), keyword categorization vs generic IDF (deferred categorization, generic IDF chosen — but rare-keyword lane added on top for high-rarity individuals + combos to update weave logic), and genre inclusion in rare-keyword pool (yes, common genres collapse to low tier naturally).

### Testing Notes
No code changes in this entry — pure documentation. Verification deferred to V3 implementation: every hypothesis in the new V3 Hypotheses table maps to a specific anchor in the existing 20 single-anchor + 12 multi-anchor benchmark, so the same batch runner that verified the categorization fixes can re-verify each subsequent change. The auteur list is explicitly out of scope for verification until the follow-up conversation locks composition.

## V4: trait-relationship typology + combine-mode branching
Files: schemas/enums.py, schemas/step_2.py, schemas/step_3.py, search_v2/step_2.py, search_v2/step_3.py, search_v2/run_step_3.py, search_v2/full_pipeline_orchestrator.py, search_v2/stage_4_execution.py

### Intent
Implements the V4 plan in [search_deepdive.md](search_deepdive.md). Three structural failure modes were converging in the V3 pipeline: (1) compound aesthetics (`bro movie`, `cottagecore`, `dark gritty`) decomposed into independent facets but Phase D's MAX rewarded single-facet matches; (2) positioning queries (`like X`, `X-style Y`, `X but Y`) had the reference trait export its full identity including axes the user explicitly asked to replace, causing direct cross-trait conflict on the same category; (3) fused compounds (`elevated horror`, `the godfather but with cowboys`) split into two cross-modifying atoms that each independently decomposed the same compound. V4 introduces a closed-set `relationship_role` commit at Step 2 plus a `combine_mode` commit at Step 3 to drive both decomposition and stage-4 fold behavior structurally.

### Key Decisions
- **`TraitRelationshipRole` enum (INDEPENDENT / POSITIONING_REFERENCE / POSITIONING_QUALIFIER) on `Trait`**, plus paired `replaces_axis: str | None` and `axes_replaced_by_siblings: list[str]` fields. The reference trait inherits replacements verbatim from sibling qualifiers; this is where the cross-trait reasoning lands so Step 3 can act per-trait without seeing sibling interpretive prose. Closed enum forces hard commitment; freeform replacement-axis prose handles the unbounded user-vocabulary surface.
- **`TraitCombineMode` enum (FRAMINGS / FACETS) on `TraitDecomposition`**, committed AFTER candidates and BEFORE category_calls so the mode shapes the choice of categories. FRAMINGS authorizes overlapping coverage (stage-4 MAX-folds; redundant categories reinforce as alternative routes to one signal); FACETS demands complementary coverage (stage-4 PRODUCT-folds; duplicate axis coverage amplifies wrong signals).
- **Fused-compound atomization rule.** Step 2's atom phase records bidirectional IDENTITY-SHAPING via a new controlled token on `ModifyingSignal.effect`; the commit phase reads bidirectional IDENTITY-SHAPING signals as the signature for fused compounds and merges to one trait. Single-direction shaping (qualifier shapes population's instance, but population doesn't reshape qualifier's meaning) keeps two atoms. Atom count ≠ trait count: splits push count up, merges push count down.
- **Step 3 reads sibling structural fields, not interpretive prose.** `_build_user_prompt` extends to `(trait, siblings)` and surfaces only `surface_text`, `relationship_role`, `replaces_axis`, `axes_replaced_by_siblings` per sibling — explicitly excluding `evaluative_intent`, `contextualized_phrase`, `commitment` to preserve V3's no-leak rationale.
- **`run_step_3` signature change** is opt-in: `siblings` defaults to an empty list so callers that haven't migrated keep V3 behavior. The orchestrator and CLI runner are updated to pass the sibling list.
- **`TraitWithEndpoints.combine_mode` defaults to `FRAMINGS`.** Failure paths (Step 3 errors) and existing test mocks that don't pass it land on V3-equivalent MAX behavior. Tests are not modified per `.claude/rules/test-boundaries.md`.
- **Phase D fold split into two helpers.** `combine_calls` (within-category) is unchanged; new `combine_categories(mode, scores)` lifts the same fold operators (MAX / PRODUCT) to the across-category level. `_score_positive_trait` reads `trait.combine_mode` and folds via the new helper. Negative-trait scoring is unchanged.

### Planning Context
The V4 design crystallized through the deepdive iteration in `search_deepdive.md`: round 1 surfaced 7 failure modes; round 2 (7 more queries) refined #1 (atom contamination is selective, not systemic) and discovered #7 (atomization-split duplication for fused compounds — `elevated horror`, `godfather but cowboys`). The user's pushback on the bleeding-vs-no-bleeding asymmetry resolved into a typology where `relationship_role` is the missing distinction; both directions of the asymmetry collapse to "what role does this trait play in the query's structure." Plan file: `~/.claude/plans/make-the-changes-memoized-yeti.md`.

### Testing Notes
End-to-end verification via `python -m search_v2.run_step_3 "<query>"` on the V4 success-criteria query set, all passing:
- `elevated horror` → 1 trait, INDEPENDENT (fuse-merged correctly).
- `the godfather but with cowboys` → 1 trait, INDEPENDENT (fuse-merged correctly).
- `like zathura with jungles` → zathura POSITIONING_REFERENCE with `axes_replaced_by_siblings=["setting"]`; jungles POSITIONING_QUALIFIER with `replaces_axis="setting"`. zathura's Step-3 decomposition explicitly drops the setting axis ("the 'setting' axis (space/sci-fi) is dropped because a sibling trait provides a substitute") and emits ZERO NARRATIVE_SETTING calls.
- `movies like inception but funnier` → inception POSITIONING_REFERENCE with `axes_replaced_by_siblings=["tone"]`; funnier POSITIONING_QUALIFIER with `replaces_axis="tone"`. inception emits zero EMOTIONAL_EXPERIENTIAL calls (the cerebral-mind-bending tone-axis call from V3 is gone); funnier emits the comedic-tone substitute correctly.
- `bro movie` → `combine_mode=facets`, three categories covering distinct axes (theme/emotional/character). Phase D will PRODUCT-fold these.
- `feel-good Christmas movies` → both INDEPENDENT, no regression. feel-good's compound aesthetic correctly commits FACETS; Christmas's single-axis trait commits FRAMINGS.
- `shitty shark movies` → fused into one trait `shitty shark movies` with `combine_mode=facets`. Behavior diverges from the V4 plan's two-trait prediction but produces equivalent-or-better scoring (a non-shark schlocky drama scores 0 on shark presence under FACETS product, properly excluded; a pristine shark movie scores 0 on shitty markers, properly downranked). Worth observing in subsequent runs.

Per `.claude/rules/test-boundaries.md`, no unit tests were modified; the existing `unit_tests/test_full_pipeline_promotion_tiers.py` mock-construction of `TraitWithEndpoints` continues to work because `combine_mode` carries a default value. Stage-4 scoring branch verified directly: `combine_categories(FRAMINGS, [0.5, 0.7, 0.9]) → 0.9` (max); `combine_categories(FACETS, [0.5, 0.7, 0.9]) → 0.315` (product); empty-list cases return 0.0 in both modes.


## V4 review fixes — atomization wording, role/axis validator, module doc
Files: search_v2/step_2.py, schemas/step_2.py, docs/modules/search_v2.md
Why: Three findings from a self-review of the V4 implementation. (1) The atomization-rule sentence "Every atom that survives the standalone and fuse checks produces at least one trait" was self-contradicting after the fuse-merge addition — under fuse, both atoms are absorbed into one trait, but a literal-reader LLM could read this as "two surviving atoms must produce two traits," which is exactly the override-the-fuse-rule failure mode the user originally flagged. (2) The relationship-role typology's structural invariants (per-trait role↔axis consistency + cross-trait positioning reciprocity + verbatim axis-bookkeeping match) were enforced only by prompt — Pydantic accepted any combination. (3) The search_v2 module doc's Step 3 description still said "reads qualifier_relation as the carver-vs-qualifier signal" with no mention of `relationship_role`, and the across-category combine section described only MAX without the FACETS PRODUCT branch.
Approach: Atomization rule rewritten to explicitly call out the absorbed-by-fuse case ("Two atoms absorbed by a fuse merge produce ONE trait together") and add the anti-default ("Do NOT default to one-trait-per-atom"). New `model_validator(mode="after")` on `QueryAnalysis` that enforces per-trait field consistency for each role, cross-trait reciprocity (orphan reference / orphan qualifier rejected), and verbatim axis-bookkeeping reciprocity (every qualifier's `replaces_axis` must be inherited by some reference, and every entry in any reference's `axes_replaced_by_siblings` must trace to a sibling qualifier). Validator is belt-and-suspenders — the LLM router's 1-retry covers transient noise; persistent failures surface as a hard branch error rather than silently corrupting Step 3's axis-drop logic. Module doc updated to surface relationship_role as the primary signal, document FRAMINGS/FACETS combine-mode branching, and reflect the new run_step_3 signature.
Testing notes: Verified validator catches orphan POSITIONING_QUALIFIER, orphan POSITIONING_REFERENCE, and axis mismatches between reference inheritance and qualifier substitution. JSON schema generation unaffected (model_validator is runtime-only). All previously-passing V4 verification queries continue to pass (their well-formed Step 2 outputs satisfy the validator's invariants).

## V5 query-generation diagnostic runner + findings doc
Files: search_v2/run_specs.py, search_improvement_planning/rescore_overhaul.md
Why: User suspected the per-CategoryCall query-generation phase was producing brittle keyword commitments that get unfairly punished by the V3/V4 ADDITIVE-multiply × FACETS-product strictness. Needed a diagnostic that runs Step 2 → Step 3 → handler-LLM end-to-end while honoring the V4 sibling contract, and surfaces only the fields that drive the multiply problem (combine_mode per trait, combine_type per category, fired endpoints, finalized_keywords + scoring_method on KEYWORD specs, role + spaces on SEMANTIC specs).
Approach: New `search_v2/run_specs.py` reuses `_run_handler_with_full_output` from `run_query_generation.py` rather than duplicating the handler-call path. Computes a mechanical `additive_kw_risk` flag per category (combine_type=ADDITIVE AND KEYWORD ∈ fired_routes) so a multi-query batch summary surfaces the worrying ones at a glance. Default suite covers 10 query shapes (compound aesthetic / positioning / fused-compound / ADDITIVE-heavy categories); refinement suite via --suite tests specific hypotheses. Optional --json captures machine-readable batch output. Findings written as a new V5 section in rescore_overhaul.md documenting five failure modes (vibe-only KW thinness, ALL chosen for paraphrase clusters, over-coverage keywords committed despite prompt's own warning, cross-trait keyword duplication, empty-spec categories zeroing FACETS traits) with five proposed fixes (empty-spec filter, prompt change to surface multiply consequences, ANY-default tightening, weakness-as-commitment-gate schema change, cross-trait dedup).
Design context: search_improvement_planning/rescore_overhaul.md V5 section.
Testing notes: 26 queries across two batch runs surfaced 30 ADDITIVE_KW_RISK categories (46% rate). Hypothesis tests for fix validation listed in V5's "Hypotheses still open" — H2 is monotonically safe by construction (filtering 0.0 from a max can only raise it); H1, H3, H4 require Phase B+C+D end-to-end instrumentation.

## V3 director lane — auteur list finalized
Files: search_improvement_planning/similar_movies_v3_plan.md, search_improvement_planning/similar_movies.md, docs/TODO.md

### Intent
Resolves the §2.1 BLOCKER from the prior planning session — the auteur list was the load-bearing artifact for the V3 director rework and Batch B couldn't ship end-to-end without it. The user committed to a curated list rather than letting V2's `mv_director_strength` percentile MV continue to drive single-anchor surfacing (Lucas surfacing American Graffiti for Star Wars; Spielberg over-firing across genre-promiscuous filmography).

### Key Decisions
- **60-entry curated list** composed from parametric knowledge + a research subagent's auteurism survey, filtered against a three-criterion bar (recognizable style across ≥3 features; general moviegoer awareness; rest of catalog likely to "feel the same" to a viewer who liked one). Generalist craftsmen (Ridley Scott, Ron Howard, Soderbergh, Eastwood, Cameron) and one-genre-lane stylists (Guy Ritchie) explicitly excluded with reasoning recorded.
- **DB-form verification.** All 60 names verified against `lex.lexical_dictionary` joined to `lex.inv_director_postings` via /tmp/verify_directors.py. 57 matched my normalization on first pass; three required corrections from credit-string variants: Bong Joon-ho → `bong joon ho` (no hyphen), Hirokazu Kore-eda → `hirokazu koreeda` (no hyphen, no space), Alejandro G. Iñárritu → `alejandro g inarritu` (period collapses to "g", diacritic stripped). Coens stored as two separate keys; lane fires for either credit independently.
- **Unified scoring rule replaces V2 popularity-MV gate.** For each director shared between candidate and anchors, contribution = 0.20 (single-anchor + auteur), `0.20 + 0.10*(M/N)` (multi-anchor + auteur, caps 0.30), or `0.10*(M/N)` (multi-anchor cohesion-only with M ≥ 2, caps 0.10). Take `max` across shared directors. Multi-director split case (4 anchors split 2-2 between curated directors) explicitly preserved: each director scores independently under its own M, never "winner-takes-all."
- **Cohesion-only floor at M/N ≥ 0.75**, magnitude 0.35, shape gate ≥ 0.30 — overrides shape regardless of curation. Threshold chosen so 2-of-3 (0.667) stays additive-only while 3-of-3, 3-of-4, 4-of-5 trigger the floor. Encodes the user's directive: "if cohesion is high enough we should floor regardless of whether they're stylistic or not."
- **Held back pending separate user call** (not blocking): Woody Allen, Mel Gibson, Roman Polanski. Stylistic signal is real but the inclusion call ties to public-discourse considerations rather than craft.

### Planning Context
This resolves V3 plan §2.1's BLOCKER callout, §6 open question #1, priority-table item 7's BLOCKED status, Batch B's blocked-end-to-end caveat, and the docs/TODO.md ESSENTIAL entry. Storage chosen as flat boolean (auteur or not) for now; tiering can be revisited if the additive 0.20 weight turns out coarse during smoke evaluation.

### Testing Notes
List composition is testable only via end-to-end smoke runs once the V3 director-lane code lands (Batch B). Pre-implementation sanity check before any director-lane code change: confirm the `search_v2`-side import of the frozenset uses the shared `implementation/misc/helpers.normalize_string` (same path the lexical pipeline used at ingest). The catalog film counts in the §2.1 table are point-in-time snapshots (2026-05-07); they're informational, not load-bearing on the lane logic itself.

## V3 similar-movies — Batches B–F implementation (11 items)
Files: search_v2/similar_movies.py, search_v2/auteur_directors.py (NEW), search_v2/run_similar_movies_batch.py

### Intent
Lands all 11 pending V3 items (priority-table items 3–13 minus the already-shipped Batch A items 1–2). Replaces V2's popularity-prior director lane with a curated auteur list, restructures the franchise lane around (lineage, subgroup, universe) overlap with the V2 consistency gate retired, extends themes + country/language to single-anchor, adds a new rare-keyword lane with tiered IDF + cohesion-floor, replaces the V2 cast lane with an N-anchor bucket-with-floor formula, harshly downranks shorts for non-short anchors, and makes medium piecewise on cross-category. Quality lane gets a bucket-conditional weight (middle 0.06 → 0.03). Smoke harness gains TDK Rises (49026) for H4 and a multi-anchor cohort runner with the V2 12-cohort set + a Tom Hanks trio for H9.

### Key Decisions
- **Franchise data structure (§2.2 Option A).** Pre-flight DB verification surfaced that crossover films (Avengers, Endgame) carry a SINGLE dedicated lineage rather than the union of constituent hero lineages the v3 plan presumed. Star Wars confirmed the same pattern (every theatrical film in lineage [3]; subgroups distinguish trilogies). Replaced the v3 plan's 14-row mainline/spinoff/crossover matrix with a simplified 5-tier rule on (lineage match, subgroup overlap, universe overlap). Loses the spinoff↔spinoff-vs-mainline↔spinoff nuance but satisfies H6 cleanly and works with the actual data structure. User explicitly approved (Option A).
- **Director lane architecture (§2.1 unified rule).** Director becomes a "passthrough" lane: `BASE_LANE_WEIGHTS["director"] = 1.0`, excluded from `_normalize_weights`'s denominator via the new `PASSTHROUGH_LANES` frozenset, and the lane's raw score IS the absolute contribution. The unified rule from §2.1 (single 0.20 / multi-curated 0.20+0.10*ratio / cohesion-only 0.10*ratio, max-over-d) maps directly. The §2.1 cohesion floor (M/N ≥ 0.75 → combined = max(combined, 0.35) at shape ≥ 0.30) lives in `_build_results`'s post-multiplier block alongside the rare-keyword and cast floors.
- **Rare-keyword lane (§2.6 NEW).** Same trait pool as themes (`_themes_traits_for_movie` excludes country/medium/format/source/award). Tiered IDF additive: low (<2.5) pooled into 0.05-cap bucket; moderate (2.5–4.5) at 0.03×idf per trait; high (≥4.5) at 0.05×idf per trait. Floor at 0.40 + 0.05*high_count (cap 0.55) when single high-tier ≥5.0 OR moderate+high sum ≥7.0. H11 verification showed catalog's TMDB genres all collapse to LOW tier (max idf ~1.0), confirming the user's "include genres for now; common ones won't contribute much" call.
- **Cast bucket-with-floor (§2.5 generic N-anchor).** Replaces V2's `_score_multi_trait_count` for the cast lane. Returns per-candidate score, per-candidate floor, AND a cohesion-driven lane weight (0.05 + 0.10 * ratio); the lane weight is finalized after cast scoring runs and before `_normalize_weights` is called (deferred from the V2 location). H9 smoke: Hanks trio surfaces Toy Story 3 (#1, cast=1.00), The Terminal (#7, cast=1.00), Toy Story 2 (#9, cast=1.00) — the floor lifts The Terminal into the top 10 despite weaker shape.
- **Shorts harsh downrank (§3.1).** Two sites: 0.30× combined-score multiplier in `_build_results` for cross-format candidates, AND a max-1 cap in `_weave_candidates` when the anchor isn't a short. Multi-anchor shorts dominance (≥50% of anchors are shorts) skips both. H3 smoke: Toy Story top 10 has 0 shorts (vs. V2's trio at 8/9/10).
- **Medium piecewise (§3.2).** Cross-category (live↔animation, where MEDIUM_SIMILARITY returns 0.0) now scales by 0.65×; within-category keeps the V2 0.85+0.15×score formula. H4: TDK Rises top 10 has zero animated Batman entries.
- **Country/language single-anchor + recalibration (§1.3, §2.4).** BOOST 1.10→1.05, PENALTY 0.85→0.75. Single-anchor adds `_country_consensus_per_candidate` helper; same `_build_results` plumbing that multi-anchor uses. H5: Barbie top 1 is Patch Town (US), V2's Telugu Swag dropped from #1 to #7.
- **Themes single-anchor (§2.3).** New `_single_anchor_themes_scores` reuses the same trait pool as the multi-anchor lane; denominator is `max(THEMES_MIN_DENOMINATOR=1.0, sum(anchor IDFs))` to prevent collapse on tag-light anchors. Pre-flight #3 confirmed Barbie's IDF mass (~2.5) gives top candidates the expected 0.4–0.7 score range.
- **Quality bucket-conditional weight (§4.1).** Middle bucket weight 0.06→0.03 (single + multi); prestige and cult_garbage keep 0.06 because their per-bucket formulas are tightly tuned. Implemented as a `quality_bucket=` parameter to `_single_anchor_lane_weights` for single-anchor and a conditional `quality_base` in the multi-anchor `raw_lane_weights` block.
- **Constants retired:** `FRANCHISE_HIGH_CONF_CONFIDENCE`, `FRANCHISE_HIGH_CONF_CONSISTENCY`, `LOW_CONF_FRANCHISE_SHAPE_GATE`, `LOW_CONF_FRANCHISE_MULTIPLIER_STRENGTH`, `DIRECTOR_SIGNATURE_STRENGTH_THRESHOLD`. Imports `fetch_director_strengths` and `fetch_franchise_confidence` removed from `similar_movies.py` (the MVs still exist for the standard search pipeline).
- **Auteur module.** New `search_v2/auteur_directors.py`: 63-entry frozenset (60 from §2.1 + Allen / Gibson / Polanski added per user direction), runtime resolution to `lex.lexical_dictionary.string_id` via the existing `fetch_phrase_term_ids` path, module-level cache after first call.

### Planning Context
Resolves V3 plan §5 priority items 3–13 except item 13 (award SPECIFICITY_FACTOR L2) which stays as observe-only — no over-firing seen in the 21+13 smoke. Pre-flight #1 (Avengers crossover) surfaced and resolved via user choice of Option A. Pre-flights #2-#4 confirmed the design assumptions held. Smoke harness's `DEFAULT_ANCHOR_IDS` adds 49026 (TDK Rises) to fill the H4 gap; new `DEFAULT_MULTI_ANCHOR_COHORTS` covers the 12 V2 cohesion sets (Pixar, Ghibli, MCU, Best Picture, Tarantino, Stephen King horror, Spielberg adventure, WW2, slasher, romcom, Ghibli+Pixar mix, Nolan) plus the V3 Hanks trio for H9. The runner now skips a cohort if any anchor ID is missing from the catalog (e.g., Polar Express isn't ingested) rather than aborting the whole batch.

### Testing Notes
All 11 V3 hypotheses verified passing in /tmp/v3_verification_report.md. All 10 success criteria pass. V2 baseline cohorts (Pixar / Ghibli / Tarantino / Nolan / Best Picture) preserved without regression. Outstanding follow-ups: (a) rare-keyword IDF tier thresholds may want lowering — catalog's actual IDF distribution is gentler than the v3 plan assumed and the floor rarely activates as designed (defer until a real failure case shows the floor *should* have fired); (b) Inception top 1 is Tenet (high shape) but director_signature only fires on The Prestige/Memento — worth confirming Nolan's auteur match is firing on all his candidates and not being filtered out by the candidate-generation lanes.

## V3 similar-movies — code-review fixes
Files: search_v2/similar_movies.py

### Intent
Five issues surfaced by `/review-code` after the V3 implementation landed. Two were real correctness bugs that the H1–H11 smoke didn't catch because the failure cases I designed happened to avoid the broken paths. Three were minor (calibration / cleanup) but worth shipping with the rest.

### Key Decisions
- **HIGH: rare-keyword multi-anchor IDF set** — `themes_idf_pairs_task` was fetching IDFs only for `themes_repeated` (traits in ≥2 anchors), but `_multi_anchor_rare_keyword_scores` scores against the **union** of all anchor traits. Traits in only one anchor were silently looking up `idf_lookup.get(trait, 0.0) = 0.0` and never reaching the moderate/high tier. Fix: fetch IDFs for the union once and feed both lanes (themes still iterates only over `themes_repeated` internally, so its scoring is unchanged). Removed the `cohesion_by_lane["themes"] > 0.0` gate from the IDF fetch — when only the rare-keyword lane is active (no themes repetition), we still need IDFs for the union. Verified on Best Picture trio: rare_keyword now fires for 10/10 candidates (vs. ~7-8 pre-fix).
- **MEDIUM: director gate misses M_d=1 + auteur in multi-anchor** — `director_task` was gated on `cohesion_by_lane["director"] > 0.0` (≥2 anchors share a director). V3 §2.1's unified rule explicitly fires the lane on multi-anchor + auteur with M_d=1 (one anchor's curated director matches a candidate, contribution = 0.20 + 0.10 * (1/N)). For mixed cohorts where each anchor has a different curated director, V2's strict cohesion gate silenced the entire lane. Fix: await `auteur_term_ids` independently before the parallel gather (the function is module-cached so it's effectively free), then loosen the gate to `cohesion > 0 OR (director_terms & auteur_term_ids)`. Verified on a Nolan + Tarantino + Kubrick cohort: 4 of 10 candidates now fire the director lane at exactly the 0.233 contribution V3 §2.1 specifies for M_d=1, N=3, curated.
- **LOW: cast lane weight 0.05 even with no shared leads** — `_multi_anchor_cast_v3` returned `CAST_LANE_WEIGHT_BASE = 0.05` when `shared_leads` was empty, which diluted other lanes' normalized weights via `_normalize_weights`'s denominator even though the cast lane had zero candidates. Fix: return `0.0` in the no-shared-leads branch. Restores the V2 "strict gating" behavior — the cast lane is silent when there's no cast cohesion.
- **LOW: shorts-dominant multi-anchor 1.10× upweight** — V3 §3.1 specifies that when ≥50% of multi-anchor inputs are shorts, the candidate-side shorts get a small positive boost. Initial implementation only disabled the harsh downrank without applying the boost. Fix: new `apply_shorts_boost: bool` parameter on `_build_results`; multi-anchor passes `apply_shorts_boost=shorts_dominant`. The two flags are mutually exclusive (boost only fires when downrank is disabled).
- **S2: `_normalize_weights` fallback cleanup** — the fallback branch (when every proportional lane is zero or negative) initialized `out` over `ALL_LANES` and immediately overwrote passthrough lanes via the second loop. Tightened to initialize over `proportional_lanes` only with shape=1.0; passthrough lanes flow through naturally via the existing second loop.

### Testing Notes
Re-ran targeted smoke after fixes:
- Mixed-curated cohort (Inception + Pulp Fiction + The Shining): director lane fires for Kill Bill Vol. 2 (#2), Memento (#5), Reservoir Dogs (#7), The Prestige (#10) at exactly 0.233 contribution. Pre-fix: 0 director hits.
- Best Picture trio: rare_keyword lane fires for all 10 candidates (was ~7 pre-fix).
- Regression check on H3 (Toy Story shorts), H6 (Star Wars franchise), H7 (Star Wars no Lucas), H9 (Hanks trio): all preserved. Hanks trio scores shifted slightly upward (Toy Story 3 0.997 → 1.016) because the cast-lane normalization is now correctly cohesion-driven without the 0.05 dilution from no-cohesion cohorts.
- Score relativities: same orderings as before in all V2 baseline cohorts.

## V3.1 similar-movies — calibration + recall expansion
Files: search_v2/similar_movies.py, db/postgres.py, search_v2/run_similar_movies_batch.py, search_improvement_planning/similar_movies.md

### Intent
The V3 smoke run with the new diagnostic harness exposed calibration
and recall gaps. V3 had the right architecture but couldn't surface
the matches it was scoring correctly: themes lane weight was
sub-perceptual (0.06), single-anchor director had no floor, the rare-
keyword tiers were sized for raw `log(N/df)` IDFs but the catalog
normalizes to ~[0,1], and there was only one broad-recall path
(Qdrant top-500 shape) so vector-distant auteur/themes matches like
Lady Bird vs. Barbie never entered the pool. V3.1 lands seven changes
in one bundle, no architectural changes — only thresholds, weights,
and recall paths.

### Key Decisions
- **Themes weight 0.06 → 0.12.** Bumps thematic contribution to a
  perceptual range. Pixar/MCU/Star-Wars cohesion stays intact (still
  shape-dominated); thematic-driven anchors (Barbie, Best Picture,
  Tarantino) recover thematic recall. Multi-anchor weight reads from
  `BASE_LANE_WEIGHTS` instead of the previous hardcoded 0.06 so the
  bump applies to both flows.
- **Rare-keyword floor shape gate 0.30 → 0.20.** The 0.30 gate
  blocked every observed high-rare-keyword candidate in smoke
  (Schindler shape=0.030, Dunkirk shape=0.160). Lowering catches
  shape-borderline candidates with strong distinctive-match evidence.
- **Single-anchor director floor (NEW).** Mirrors the multi-anchor
  floor with softer thresholds: magnitude 0.35, shape gate 0.20,
  ratio threshold 1.0 (binary — any auteur match qualifies). New
  constants `DIRECTOR_FLOOR_SINGLE_*`. `_build_results` now accepts
  per-flow floor parameters; defaults preserve V3 multi-anchor
  behavior. Single-anchor caller threads the softer values.
- **Moderate-combo bonus on rare-keyword lane (NEW).** Additive
  bonus on the passthrough lane: `0.05 + 0.05 * (sum_mod_high - 0.50)`,
  capped at 0.15, fires when shared moderate+high tier IDF sum ≥ 0.50.
  Not gated on shape — the bonus IS itself a shape signal that
  compensates for vector distance. Sized from real catalog: 324
  random pairs all had p99 = 0.000 mod+high IDF, so 0.50 sits deep in
  the long tail. Constants `RARE_KW_COMBO_*`.
- **Themes-recall candidate fetch (NEW path).** New
  `fetch_movie_ids_by_themes_recall` in db/postgres.py runs a single
  SQL aggregate joining `movie_card` array columns to `mv_trait_idf`
  per kind, GROUP BY movie_id, HAVING `SUM(idf) >= combo_sum_thr OR
  MAX(idf) >= single_idf_thr`. Defaults: `single=0.55, combo=0.50`.
  Per user direction, the SUM gate includes **all-tier** IDFs (not
  filtered to moderate+high) — "even if individual tags aren't rare,
  matching a bunch is rare." Risk monitored. Wired into single-anchor
  candidate-id union alongside rare_medium recall.
- **Multi-anchor consensus traits with cohesion-IDF tradeoff.** New
  helper `_multi_anchor_consensus_themes_traits` computes M_t/N
  cohesion per trait and applies an IDF-scaled bar (LOW < 0.30 needs
  cohesion 1.0; MOD 0.30-0.55 needs 0.67; HIGH ≥ 0.55 needs 0.50).
  The consensus pool feeds the same single-anchor SQL helper —
  multi-anchor logic collapses to a single fetch path with a tighter
  trait set. Fetched in the first parallel gather (themes IDFs added
  alongside it).
- **Qdrant `DEFAULT_QDRANT_LIMIT` 500 → 2000.** Catches vector-
  distant matches that targeted themes-recall can't help (Lady Bird
  shares zero high-IDF or combo-eligible traits with Barbie's pool).
  Per-anchor latency expected to grow ~150ms → ~400-500ms; tradeoff
  acceptable for recall gain.

### Planning Context
Documented in
[search_improvement_planning/similar_movies.md](search_improvement_planning/similar_movies.md)
"V3.1 Calibration Adjustments" section with full Decisions 1–7,
constants matrix, and "Expected outcomes" subsection enumerating
wins to confirm and regressions to watch.

### Testing Notes
Verification deferred to next user-driven harness run. Wins to
confirm: Lady Bird and The Favourite enter Barbie top 10; I Am Not an
Easy Man rises; Tenet for Inception preserves #1 with bigger themes
contribution; Pixar/MCU/Star-Wars/Best-Picture/Tarantino top 10s
preserve cohesion. Regressions to watch: floor over-firing from
softer 0.20 shape gate; combo bonus producing weak-shape coincidence-
match flooding; recall expansion pushing score >1.0 outliers further
into noise; latency degradation beyond ~500ms; themes-recall pool
exploding (>10k candidates) on tag-rich anchors — first regression to
confirm/deny since user opted to include all-tier IDFs in the SUM.

## V5 Phase 1: empty-spec filter + post-hoc generator-spec dedup
Files: search_v2/stage_4_execution.py, search_improvement_planning/search_overheaul_test_tracker.md

Why: Two pure-code changes from rescore_overhaul.md Phase 1.
Approach:
- 1.1 (D4): in `_score_positive_trait`, skip categories whose handler
  emitted zero generated_specs *before* they reach `combine_calls`.
  Prevents `combine_calls(SINGLE, []) → 0.0` from zeroing a
  FACETS-PRODUCT trait when one category abstains. Monotonic-safe in
  both FRAMINGS-MAX and FACETS-PRODUCT.
- 1.2 (D5): in `_run_branch` Phase B, group positive-polarity
  generator specs by `(route, model_dump(mode="json"))` and run one
  `_dispatch_call` per unique group. Broadcast each result map to all
  shared `_CallKey`s. Specs with `params is None` skip dedup.
  Score-side semantics unchanged — each (trait_idx, cat_idx, spec_idx)
  still reads its own per-coordinate map.

Design context: search_improvement_planning/rescore_overhaul.md
§Phase 1 (changes 1.1, 1.2); search_overheaul_test_tracker.md
Iteration 2.

Testing notes: validated by `python -m search_v2.run_specs --suite
/tmp/v5_suite.txt` against baseline. Aggregate metrics moved within
LLM noise floor (56.2 % → 57.8 % ADDITIVE_KW_RISK rate; +1.6 pp).
**run_specs.py stops before Phase B/D and therefore does not
exercise either change** — the experiment's purpose was to confirm
non-regression of upstream LLM commits, not visible scoring impact.
For real verification add a stage-4 unit test that constructs a
FACETS trait with one abstaining category and asserts trait_score
reflects the live category alone (deferred per
.claude/rules/test-boundaries.md).

## Step 2 validator self-heal for orphaned positioning roles
Files: schemas/step_2.py
Why: One Phase-1 verification run hit a hard Step-2 ValidationError
("trait[0] role=POSITIONING_REFERENCE but axes_replaced_by_siblings
is empty") on `Studio Ghibli style hand-drawn fantasies`. The LLM
occasionally commits a POSITIONING_REFERENCE without populating
its `axes_replaced_by_siblings` (or a POSITIONING_QUALIFIER without
its `replaces_axis`). Both states are semantically no-ops — a
reference with nothing to drop and a qualifier with nothing to
substitute behave identically to INDEPENDENT under Step 3. Today
the validator rejected; the whole query errored.
Approach: pre-pass in `_validate_relationship_roles` coerces
orphaned commits to INDEPENDENT (clearing `replaces_axis` /
`axes_replaced_by_siblings` to keep field consistency invariants).
After the per-trait coerce, if reciprocity collapses to refs-only
or quals-only, coerce the surviving orphans too. The strict
cross-trait axis-bookkeeping checks (`missing_on_refs` /
`invented_on_refs`) still run — those catch real LLM logic errors
where sibling commits disagree on axis names. Verified by full-
pipeline regression sweep: Studio Ghibli now completes cleanly
(3 traits, 7041 ranked, top='Ramayana: The Legend of Prince Rama')
and 24/24 other queries still succeed.

## V5 Phase 2a: TARGET_AUDIENCE + SENSITIVE_CONTENT → ALTERNATIVES
Files: schemas/trait_category.py, search_improvement_planning/rescore_overhal_queries.md
Why: V5 D1 / Phase 2a per
[search_improvement_planning/rescore_overhaul.md](search_improvement_planning/rescore_overhaul.md)
§Phase 2a. Under V3 ADDITIVE, TARGET_AUDIENCE and SENSITIVE_CONTENT
multiplied KW × META × SEM, so a movie inside the maturity range
(META=1.0) still got dragged to 0 by a thin-superset KW miss
even when CTX `watch_scenarios` carried the signal.
Approach: two single-line enum-tuple flips in
schemas/trait_category.py at L759 (TARGET_AUDIENCE) and L784
(SENSITIVE_CONTENT) — `CategoryCombineType.ADDITIVE` →
`CategoryCombineType.ALTERNATIVES`. No prompt or schema or
scoring-code change; combine_type is consumed deterministically
by `stage_4_execution.combine_calls` (now MAX(KW, META, SEM)) and
by `run_specs._summarize_category` (trip-wire requires
`combine_type=='additive'`).
Design context: rescore_overhaul.md §Phase 2a "Risk surface" —
ALTERNATIVES encodes the intended "gate-or-include" semantics:
maturity-eligible alone is sufficient, otherwise KW/SEM still
scores. Tracker entry: search_overheaul_test_tracker.md Iteration 3.
Testing notes: verified via run_specs against the V5 verification
suite. ADDITIVE_KW_RISK rate 56.2 % (baseline) → 45.9 % (-10.3 pp;
-4 pp structural attributable to TA + SC, -6.3 pp net headline
including LLM commit-shape drift). 3 / 3 fired TA + SC rows now
read `combine_type='alternatives'`; 0 / 3 trip the
`additive_kw_risk` flag. All three positive controls (Q9 / Q13 /
Q25) hold their commit shapes against baseline. Stage-4 scoring
effect (MAX vs PRODUCT-of-three) is invisible to run_specs but is
deterministic by construction; an orchestrator_batch sweep on
Q4 / Q5 / Q16 could close that loop if needed (deferred).

Also documented an operator footgun in the V5 verification suite
markdown: passing the `.md` directly to `run_specs --suite`
silently dispatches every non-`#`-non-blank line as a query
(burned ~314 prose lines + the OpenAI daily quota in one run
during this iteration). Replaced the misleading "How to use"
section with an explicit warning + canonical
`/tmp/v5_suite.txt` invocation.

## V3.3 similar-movies — shape multiplier (reach × quality identity boost)
Files: search_v2/similar_movies.py, db/postgres.py,
search_improvement_planning/similar_movies.md,
search_improvement_planning/similar_movies_test_tracker.md

### Intent
Add an identity-level "shape" multiplier alongside the existing
country / studio / medium / shorts multipliers in the V3
similar-movies scoring stack. Five named shapes derived from a
reach × quality grid: **dogshit** (LOW × cult_garbage bucket),
**cult_garbage** (HIGH/MID × cult_garbage), **prestige** (HIGH/MID
× prestige), **hidden_gem** (LOW × prestige),
**mainstream_blockbuster** (HIGH × middle). When anchor and
candidate share a shape with sufficient cohort cohesion, apply a
multiplicative boost.

### Key Decisions
- **STRONG (×1.15) for cult/dogshit/hidden_gem; MODERATE (×1.08)
  for prestige/mainstream_blockbuster.** Sharply distinctive
  identities deserve a bigger lift than overlapping ones; the
  +0.15 / +0.08 split aligns with country/studio multiplier
  magnitudes.
- **Multi-anchor uses cohesion-weighted boost** (`1 + max_strength
  * M_s/N`) gated at ≥0.5 cohesion. Mixed cohorts (Studio Ghibli
  + Pixar) self-suppress the boost when no single shape dominates.
- **Reach axis from `imdb_vote_count`** (≥100K HIGH, 10K–100K MID,
  <10K LOW). `fetch_similarity_signal_rows` extended to return
  `imdb_vote_count` so the classifier reads it without an extra
  fetch.
- **Quality axis reuses `_quality_bucket(row)`** (existing function
  returning prestige / cult_garbage / middle from reception_score
  + popularity_percentile). Shapeless cells (MID × middle, LOW ×
  middle) get None and no boost.
- **Plug-in point: existing multiplier stack.** Surfaced in
  `LaneEvidence.multipliers["shape"]` for diagnostics.

### Planning Context
Design discussion captured in tracker V3.3 entry. Two mechanisms
were considered together (score-level shape boost + slot-level
weaving with anchor-aware MMR quotas). Per user direction, only
the score-level boost shipped in V3.3; weaving was designed but
parked pending V3.3 verification. Shape labels were iterated
(prestige/cult/mainstream/default → 4 labels with unsignaled →
final data-derived 5 shapes from reach × `_quality_bucket`). Step
distance between positioning labels was rejected in favor of
binary same/different shape comparisons.

### Verification
Full 21-anchor + 14-cohort smoke harness re-run. Major wins on
cult-bad anchors (The Room top 10 = Troll 2, Birdemic, Movie 43,
Manos, Plan 9 — cult canon; Sharknado sequels +0.190).
Shape-cohesive multi cohorts (MCU, Pixar, Best Picture,
Tarantino) saw clean ×1.08 boosts on identity-aligned candidates.
Female-led / Gerwig and other mixed-shape cohorts saw no change
(quality-bucket strictness, not multiplier limitation).

### Testing Notes
- Score inflation continues — Empire 1.926, Iron Man 2 1.429.
  Same regression-to-watch as V3.1/V3.2; not addressed.
- Quality bucket thresholds (recep ≥85 prestige, recep ≤45 +
  pct ≥0.89 cult_garbage) are conservative — many "feel-prestige"
  films classify as middle and miss shape participation. Loosening
  thresholds is the next single-axis change recommended in the
  tracker.
- No regressions detected on previously-strong cohorts (Inception,
  Star Wars, Best Picture, MCU). Shape boost is purely additive
  on top of the existing pipeline.

## V3.3.1 similar-movies — loosen shape-classification thresholds
Files: search_v2/similar_movies.py,
search_improvement_planning/similar_movies_test_tracker.md
Why: V3.3 smoke run showed many "feel-prestige" indie films
(Frances Ha 78.8, 20th Century Women 79.0, Juno 78.2, Little
Women 1994 81.4) and cult/dogshit films (Mega Python at percentile
0.787, Leprechaun 4 at 0.883, Plan 9) failed `_quality_bucket`
strict gates and so missed shape participation. Female-led /
Gerwig cohort saw zero observable shape lift in V3.3 even though
its anchors were classifying as prestige.
Approach: decoupled shape classification from `_quality_bucket`
(which is also used by legacy quality lane formulas and shouldn't
be touched). New constants `SHAPE_PRESTIGE_RECEPTION_MIN=78.0`
and `SHAPE_POOR_RECEPTION_MAX=50.0`; percentile gate dropped
entirely (reach axis already filters by audience size). Rewrote
`_classify_shape(row)` to read reception_score and imdb_vote_count
directly. `_quality_bucket` and the legacy lane formulas
unchanged.
Verification: full 21-anchor + 14-cohort smoke harness. Major win
on Female-led / Gerwig — Frances Ha, 20th Century Women, Juno,
Little Women 1994 all now ride ×1.054 prestige boost; Licorice
Pizza and Rushmore newly enter top 10. Plan 9 from Outer Space
jumps from #10 to #4 for The Room (newly cult_garbage classified).
Inception itself reclassifies prestige and Tenet/Memento/
Mulholland Drive get prestige boosts.
Mild regression: MCU cohort splits between mainstream (Avengers,
Civil War) and prestige (Iron Man, Endgame at recep 79+).
Cohesion drops from 1.0 to 0.67 in mainstream; per-MCU-film score
drops ~0.03. Cluster integrity preserved — top 10 still 9/10
MCU; Brave New World swaps with Thor: The Dark World (both MCU).
Acceptable tradeoff — Female-led win materially outweighs MCU
mild cohort dilution. No top-1 lost in any cohort.

## V3.3.2 similar-movies — award-aware classification + cross-bucket boosts
Files: db/postgres.py, search_v2/similar_movies.py,
search_improvement_planning/similar_movies_test_tracker.md
Why: V3.3.1 reception-only thresholds were both too generous (78
default admitted middling films) and too strict (76.8-rated Killing
Fields with 3 Oscar wins missed prestige). Same-shape-only
multipliers also enforced arbitrary boundaries (cult_garbage and
dogshit cleanly separated by 10K reach split, even though they're
the same audience).
Approach: two refinements bundled.
(1) Award-aware classification: bumped default prestige floor
78→80; added with-award lowered floor at 65 firing only on
*picture-level* signals (Best Picture or Director nom/win at
non-Razzie ceremony — acting/craft awards explicitly excluded).
Razzie WIN side: any WIN in WORST_* leaves (excluding WORST_OTHER
to give Razzie Redeemer benefit of the doubt) raises poor ceiling
50→60. Tag IDs: BEST_PICTURE_ANY rollup (103) + DIRECTOR (9) for
prestige; WORST_PICTURE through WORST_DEBUT_OR_NEWCOMER (46-57)
for poor.
(2) Cross-bucket boost matrix: 5×5 dict mapping (anchor_shape,
candidate_shape) → strength on [0,1]. Same-shape 1.0,
boundary-arbitrary same-quality reach splits 0.7
(prestige↔hidden_gem, cult↔dogshit), quality-step crossings via
mainstream bridge 0.4/0.25/0.15/0.10. Effective cohesion sums
anchor cohesion × cross-strength across all anchor shapes.
SHAPE_COHESION_MIN stays at 0.5 — single-anchor 0.4 cross pairs
don't fire, but mixed cohorts where same-shape (1.0) + cross (0.4)
sum to ≥0.5 do fire.
Plug-in: extended SimilarityAwardSignals dataclass with two
booleans; extended SQL aggregate with two BOOL_OR conditions in
single pass. Always-fetch award_signals for both anchors and
candidates (V3.3.1 gated on cult_or_prestige bucket; V3.3.2 needs
universally for shape classification). _classify_shape now takes
optional award_signal arg. _shape_multiplier sums cross-strengths.
Verification: full 21-anchor + 14-cohort smoke harness. Major
wins: The Killing Fields (Best Picture cohort) 0.992→1.071 (now
prestige via picture-level signal); Sharknado→Mega Python
0.519→0.573 and Sharknado→Leprechaun 4 0.504→0.557 (cult↔dogshit
cross-bucket at 0.7 fires); Female-led / Gerwig cohort cohesion
unifies at 1.0 prestige (Barbie now classified prestige), all
prestige peers lift uniformly from ×1.054 to ×1.08.
Tradeoff: MCU cohort split — all 3 anchors classify as prestige
via picture-level signals, so mainstream MCU candidates (Iron Man
2, Age of Ultron, Cap: First Avenger) lose their V3.3.1 cohort
boost while prestige MCU candidates (Endgame, Cap: Winter Soldier,
Infinity War) gain ×1.08. Within-cluster reshuffling; top 10
still 9/10 MCU + The Rock. Cluster integrity preserved.
The Mission (62.6 + Palme d'Or) misses 65 with-award floor —
acceptable; can be loosened to 60 in V3.3.3 if recurring.
Razzie Redeemer Award limitation noted (no schema-level
distinction; WORST_OTHER excluded as heuristic buffer).

## V5 Phase 2b: REMOVE-KW from EMOTIONAL_EXPERIENTIAL + SEASONAL_HOLIDAY + SPECIFIC_PRAISE_CRITICISM
Files: schemas/trait_category.py, search_v2/endpoint_fetching/category_handlers/prompts/categories/additional_objective_notes/{emotional_experiential,seasonal_holiday,specific_praise_criticism}.md, search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/{emotional_experiential,seasonal_holiday,specific_praise_criticism}.md, search_improvement_planning/query_categories.md, search_improvement_planning/search_overheaul_test_tracker.md
Why: Eliminate the F1/F2 failure modes where these three categories
fired thin keyword commits (BITTERSWEET_ENDING for tone, FEEL_GOOD
for "wholesome", paraphrastic ALL on STORY_THEMATIC clusters) that
ADDITIVE-multiplied a registry miss into a category-zero. After
this phase, all three categories fire a single SEMANTIC endpoint
under SINGLE_NON_METADATA_ENDPOINT / CategoryCombineType.SINGLE —
the trip-wire formula `combine_type==additive AND keyword in
fired_routes` mechanically excludes them.
Approach: Per-category enum-tuple update (drop KEYWORD route,
re-bucket, re-combine-type) plus prompt rewrites for each category's
additional_objective_notes (delete `## Keyword Augmentation` /
`## Coverage Decision` multi-endpoint framing, keep domain scope and
boundary rules) and few_shot_examples (full rewrite to the
`<retrieval_intent>+<expressions>` input shape that
`build_user_message` actually emits, matching the existing
SINGLE_NON_METADATA_ENDPOINT siblings like NARRATIVE_SETTING).
4 examples per category, even-split fire/no-fire, all queries
disjoint from /tmp/v5_suite.txt per the example-eval separation
rule. query_categories.md Cat 29 / Cat 33 / Cat 40 endpoint
descriptions updated for doc/code alignment.
Design context: search_improvement_planning/rescore_overhaul.md
§Phase 2b; small-LLM principles from
docs/conventions.md §483-491 (principle-based constraints, not
failure catalogs) and §508-513 (example-eval separation);
search_improvement_planning/category_handler_planning.md §425-430
(3–5 calibration examples / handler).
Testing notes: V5 suite re-run via `python -m search_v2.run_specs
--suite /tmp/v5_suite.txt --json /tmp/run_specs_phase_2b.json
--concurrent 4`. Headline ADDITIVE_KW_RISK 39 / 85 (45.9%, phase
2a) → 25 / 82 (30.5%, phase 2b), -15.4 pp. All 23 target rows
across the suite show `combine_type=single` and `fired_endpoints=
[semantic]`; 0 KW commits. Positive controls Q9 (held at 2), Q13
(2 → 1, structural; genuine plural-intent ALL on STORY_THEMATIC
intact), Q25 (2 → 1, structural; ALL on NARRATIVE_DEVICES intact).
0 errors, 0 errored_cats. Stage-4 trait_score effect (no
KW-multiply gate, monotone in semantic gradient) verified
structurally only — orchestrator_batch sweep deferred per Phase 2a
precedent (when run_specs surface IS the verification surface,
stage-4 sweep is optional).

## V5 Phase 3: keyword.md superset test + singular/plural rewrite + bucket partial-abstention sanction
Files: search_v2/endpoint_fetching/category_handlers/prompts/endpoints/keyword.md, schemas/keyword_translation.py, search_v2/endpoint_fetching/category_handlers/prompts/buckets/preferred_representation_fallback_objective.md, search_v2/endpoint_fetching/category_handlers/prompts/buckets/audience_suitability_deterministic_first_objective.md, search_v2/endpoint_fetching/category_handlers/prompts/buckets/semantic_preferred_deterministic_support_objective.md, search_improvement_planning/search_overheaul_test_tracker.md
Why: Reduce F2 (ALL on paraphrase clusters) and F3 (over-coverage
despite weaknesses naming the over-pull) by replacing the keyword
endpoint prompt's prescriptive shape language with a single
principle-based superset test, replacing the cue-word ANY/ALL
discriminator with a singular-vs-plural framing read off the call's
expressions, and explicitly sanctioning partial abstention at the
multi-endpoint bucket level.
Approach: Three coupled prompt rewrites shipped as one bundle since
they share an editing surface (keyword.md + the multi-endpoint
bucket prompts). 3.1 — keyword.md `## Authoring strengths and
weaknesses per candidate` + `## Near-collision disambiguation`
collapsed into one `## Commitment: superset test` section
(verbatim from rescore_overhaul.md §3.1). The strengths/weaknesses
fields stay in the schema as walk-phase scaffolding; the gate from
candidates to finalized_keywords becomes the principle-based
superset test. 3.2 — keyword.md `## Reading the brief for
scoring_method` rewrite from cue-words ("or"/"and"/"both") to
singular-vs-plural framing read off the call's expressions
(verbatim from §3.2). Plus matching updates to BOTH
KeywordQuerySpec.scoring_method (L215) and
KeywordQuerySpecSubintent.scoring_method (L393) field descriptions
in schemas/keyword_translation.py — the schema-as-micro-prompt rule
demands both holders agree with the endpoint prompt. 3.3 — added a
fourth local test ("Superset test per endpoint") to the three
multi-endpoint bucket prompts that previously treated abstention as
all-or-nothing. character_franchise_fanout_objective.md
deliberately excluded from the audit (both paths fire by design
when a referent exists). No category-specific examples in either
new section — the spec explicitly forbids them so the LLM
evaluates the underlying property rather than pattern-matching.
Adheres to small-LLM principles in docs/conventions.md §483-491
(principle-based, not failure catalogs) and §475-482 (merge
ambiguous field boundaries — both scoring_method holders updated
in lockstep).
Design context: search_improvement_planning/rescore_overhaul.md
§Phase 3.
Testing notes: V5 suite re-run via `python -m search_v2.run_specs
--suite /tmp/v5_suite.txt --json /tmp/run_specs_phase_3.json
--concurrent 4`. The dominant signal is in the ANY/ALL
distribution, not the trip-wire headline: ALL ratio 14.6% (phase
2b) → 5.0% (phase 3); only 2 surviving ALL commits and both are
genuine plural intent on GENRE under combine=alternatives
(action+thriller, war+history). Headline trip-wire moved
modestly (30.5% → 28.8%, -1.7 pp) because the metric tracks
`additive AND kw fires` not commit strictness; Phase 3 reduces
brittleness inside trip-wire rows (ALL on paraphrase clusters →
ANY = movies score on partial matches instead of zeroing on
missing tag), not the row count. Positive controls Q9 / Q13 / Q25
held or improved with no over-correction. F3 abstention signal
weaker than predicted — the superset test landed in the prompt
but didn't drive more abstention on STORY_THEMATIC over-pulling
categories (kw_commits 41 → 40 essentially flat). Captured as a
follow-up iteration target rather than a ship blocker. 0 errors,
0 errored_cats, 0 schema violations.

## V5 Iteration 6 attempt: Phase 4 (deliberate-default at Step 3 + bucket prompt openings) — UNSHIPPED, REVERTED 2026-05-08
Files reverted via `git checkout HEAD --` (no longer modified):
search_v2/step_3.py, schemas/step_3.py,
search_v2/endpoint_fetching/category_handlers/prompts/buckets/preferred_representation_fallback_objective.md,
search_v2/endpoint_fetching/category_handlers/prompts/buckets/audience_suitability_deterministic_first_objective.md,
search_v2/endpoint_fetching/category_handlers/prompts/buckets/semantic_preferred_deterministic_support_objective.md,
search_improvement_planning/search_overheaul_test_tracker.md

### Intent
Phase 4 of rescore_overhaul.md targeted architectural pattern B
(fire-default everywhere) via prompt-only intervention: rewrite
_CATEGORY_ROUTING at Step 3 with deliberate-default opening,
thread the abstention frame through the operational tests as a
new ABSTENTION-DEFAULT (FIRST) test, mirror in
TraitDecomposition.category_calls field description (schema-as-
micro-prompt parity per Iteration 5 lesson #2), and re-open the
three multi-endpoint bucket prompts so abstention is the prior
with the per-endpoint Superset test reordered to be the first
local test in coverage_exploration.

### Outcome
Hypothesis NOT validated. Phase 4 produced:
- F3 abstention moved in WRONG direction: KW commits in 5 keep
  categories went 23 → 25 (+2). STORY_THEMATIC_ARCHETYPE
  unmoved at 13.
- Step 3 sprawl regressed: total category routes 80 → 92
  (+15%). Q21 atmospheric folk horror went 2 → 6 cats
  (worst single-query regression).
- Genuine plural-intent ALL collapsed on Q5 (`[ACTION,
  THRILLER]` → ANY) and Q11 (`[WAR, HISTORY]` → broke
  entirely: GENRE `[WAR]` ANY + new STORY_THEMATIC
  `[EPIC, HISTORICAL_EPIC]` trip-wire).
- Headline rate dropped 28.8% → 27.2% (1.6pp) — but this is
  the structural-axis metric Iteration 5 flagged as misleading;
  absolute counts went up on every behavioral axis.
- Q9 / Q12 / Q25 named positive controls held clean. 0 errors,
  0 schema violations.

Recommendation: REVERT all five code/prompt files. Keep the
tracker entry — it documents the failed experiment for
posterity and corroborates the architectural-pattern-B
diagnosis (prompt-only interventions plateau then regress).
Phase 5 (schema-level verdict fields per rescore_overhaul.md
§Phase 5) is the next move; the failure of Phase 4 as designed
is corroborating evidence that schema enforcement is the
load-bearing intervention.

### Reverted (2026-05-08)
The five implementation files (search_v2/step_3.py,
schemas/step_3.py, three bucket prompts) have been reverted via
`git checkout HEAD --`. The tracker entry under
search_improvement_planning/search_overheaul_test_tracker.md is
PRESERVED — it captures Iteration 6's results, lessons, and
revert decision, and is part of the standing test-tracker
history. DIFF_CONTEXT entries describing this aborted
iteration may be cleared at the next commit.

Run output: /tmp/run_specs_phase_4.json (102KB),
/tmp/diff_phase_4.py.

## V3.4.1 Bucket-Weaver calibration pass
Files: search_v2/similar_movies.py, search_improvement_planning/similar_movies_test_tracker.md

### Intent
Calibrate the V3.4 weaver to fix three regressions surfaced by the
post-V3.4 smoke run: Pixar trio non-Pixar injection (Lego Movie,
Madagascar), Slasher trio musical leakage (Pennies from Heaven,
Mamma Mia), and Barbie meta-doc surfacing past slot 5 (Tiny
Shoulders, Barbie Nation).

### Key Decisions
- **Change 1 — `WEAVER_BUCKET_INSTANTIATE_MIN: 0.30 → 0.50`.** Single
  cohesion knob; requires ≥half-anchor cohesion before instantiating
  signal buckets. Prevents weak-cohesion auteur firing on
  heterogeneous cohorts (Carpenter on The Thing only, McQueen on 12
  Years a Slave only).
- **Change 2 — `WEAVER_LAMBDA: 0.50 → 0.60`.** Increased MMR
  starvation pull intended to push bucket-unique candidates (Kill
  Bill, Django for Pulp Fiction) into top 10 by narrowing the
  relevance-vs-deficit margin.
- **Change 3 — Signal-bucket format lock all-slots.** Extended
  format lock from slots 0–4 to all 10 slots for auteur / franchise /
  rare_keyword / lead_actor queues. `_peek_next_eligible_for_bucket`
  now takes `bucket_name`; best_overall keeps the slot-5 cliff so
  cross-format candidates can still surface past slot 5 on relevance.

### Smoke run outcome (mixed)
- Change 1 VALIDATED — clean wins on Stephen King horror (Kubrick
  films Full Metal Jacket / 2001 dropped, replaced by The Innocents /
  Night House) and Best Picture trio (McQueen films Hunger / Widows
  dropped, replaced by Oppenheimer / The Irishman).
- Change 2 INVALIDATED — Pulp Fiction unchanged (relevance gap too
  wide for λ=0.6 to flip); Pixar trio worse (Lego Movie #4→#3,
  Madagascar #6→#5); Slasher trio worse (Mamma Mia #7→#5, Can't
  Stop the Music entered #8).
- Change 3 INVALIDATED — Barbie meta-docs at IDENTICAL ranks (#7,
  #8). Diagnosis was wrong: Tiny Shoulders / Barbie Nation enter via
  best_overall queue (high V3 base score 0.550), not the franchise
  queue. Format lock on signal buckets had zero observable effect.

### Ship recommendation
**Partial ship: keep Change 1, revert Changes 2 and 3.** The
cohesion gate is sound; the other two were misdirected. Defer
Barbie meta-doc fix to V3.5 format-mismatch multiplier (symmetric
with `MEDIUM_CROSS_CATEGORY_MULTIPLIER = 0.65`), which penalizes
cross-format candidates regardless of placement queue.

### Testing notes
- Smoke run output: search_v2/similar_movies_batch_results.md and
  search_v2/similar_movies_multi_anchor_results.md (V3.4.1).
- V3.4 baseline preserved at /tmp/v3_4_baseline_*.md for diff.
- 8 of 14 multi-anchor cohorts unchanged; 4 changed sets (2 wins,
  1 mixed regression, 1 loss); 2 reordered. 0 single-anchor sets
  changed.
- Awaiting user decision on partial-revert before any further code
  changes.

## V3.4.2 Cohesion-weighted rare_keyword + format-mismatch multiplier + mockumentary demotion
Files: search_v2/similar_movies.py, search_v2/format_registry.py, search_improvement_planning/similar_movies_test_tracker.md

### Intent
Close the V3.4 regression loop: kill the rare_keyword 1/N-cohesion
bug that was firing musical traits on Slasher trio and non-Pixar
traits on Pixar trio, and add a hard format-mismatch multiplier
that displaces Barbie meta-docs (Tiny Shoulders, Barbie Nation)
from top 10. Demote mockumentary from format taxonomy to themes
since it's a style overlay on narrative comedy, not a separate
content category.

### Key Decisions
- **Cohesion-weighted rare_keyword formula**:
  `cohesion_weight(M_t, N) = max(0, (M_t-1) / (N-1))` applied
  per-trait inside the IDF sum. Single-anchor traits contribute
  zero; weight scales linearly to 1.0 at full N/N cohesion.
  Membership bound by same M_t ≥ 2 floor. Generalizes naturally
  across cohort sizes; alternative was a hard 2/N threshold (less
  smooth) or a quadratic scaling (over-aggressive). Slasher
  pre/post numerical proof: signal 1.000 → 0.149.
- **`FORMAT_CROSS_CATEGORY_MULTIPLIER = 0.35`**: harsher than
  medium (×0.65) because format buckets are categorical content
  types, not style variations. Symmetric with medium piecewise.
  Skips short candidates so the existing `SHORTS_DOWNRANK_MULTIPLIER
  = 0.30` doesn't compound. User-approved value (rejected ×0.65
  parity as too soft, ×0.25 as needlessly aggressive).
- **Mockumentary removed from format**: Spinal Tap / What We Do
  in the Shadows are narrative comedies stylistically dressed as
  docs. Treating them as their own bucket prevented them from
  competing on a narrative-anchor's top section. Now flows into
  themes lane via `_themes_traits_for_movie` (which subtracts
  `FORMAT_KEYWORD_IDS_ALL`, no longer including mockumentary).
- **V3.4.1 changes 2 (λ=0.6) and 3 (signal-bucket format-lock
  all-slots) reverted** before this work. λ back to 0.50;
  `_peek_next_eligible_for_bucket` back to original signature.

### Smoke run outcome (full ship)
Set-diff vs. V3.4 baseline (clean comparison after V3.4.1 partial
revert):

- **Single-anchor**: 1 of 21 cohorts changed. Barbie: Tiny
  Shoulders/Barbie Nation OUT, Free Guy/I Am Not an Easy Man IN.
  20 cohorts unchanged.
- **Multi-anchor**: 7 of 14 cohorts changed, all wins. Pixar
  (Lego/Madagascar/Rescuers OUT, Bolt/Ratatouille/Wreck-It Ralph
  IN), Slasher (Pennies/Mamma Mia OUT, Fog/Pandorum IN), Stephen
  King (Kubrick films OUT, Innocents/Night House IN), Best
  Picture (McQueen films OUT, Irishman/There Will Be Blood IN),
  Female-led (Brooklyn IN), Nolan (mild upgrade), Tom Hanks
  (body-swap thematic tail traded for tighter Pixar cluster).
- **Format multiplier verification**: zero `format_mismatch`
  entries in top-10 outputs — correct behavior since penalized
  candidates fall out of top 10 entirely. Empirical proof =
  Barbie meta-docs disappearing.

### Testing notes
- `python -m search_v2.run_similar_movies_batch --multi --limit 10`
  output written to search_v2/similar_movies_*_results.{json,md}.
- V3.4 baseline preserved at /tmp/v3_4_baseline_*.md.
- I Am Not an Easy Man surfacing on Barbie closes the V3.1 themes
  hypothesis loop unresolved through V3.3.
- Trade-off acknowledged: Tom Hanks loses body-swap thematic tail
  (Heaven Can Wait, 13 Going on 30, Being John Malkovich) to
  cohesion gate. Per user principle "don't fire on 1/N cohesion
  ever". If single-anchor thematic surfacing matters for some
  cohort, the correct lever is per-anchor themes-recall (V3.1
  infrastructure), not relaxing the cohesion gate.
- 0 new regressions across 35 cohorts.

## V3.4.3 — rare_keyword: passthrough additive → multiplier-on-shape
Files: search_v2/similar_movies.py, search_improvement_planning/similar_movies_test_tracker.md

### Intent
V3.4.2 left rare_keyword as a passthrough additive lane: its raw
score (tier-summed shared IDF + combo bonus) flowed through with
weight 1.0 as an absolute contribution to the additive sum. Smoke
diagnostic on Barbie revealed this dominated the deep tail (because
proportional lanes decay with rank but a passthrough flat raw value
doesn't) AND inflated the top section (because the lane scored any
non-empty trait intersection — including the idf<0.30 low-tier
pool that isn't actually rare). Pleasantville sat at #8 for Barbie
with `dominant_lane=rare_keyword`, 39% of its score from the
passthrough alone.

### Key Decisions
- **Reshape rare_keyword as a multiplier on shape**, mirroring
  the studio lane pattern. Removed from `ADDITIVE_LANES` and
  `PASSTHROUGH_LANES`; weight set to 0.0 (kept in dict for debug).
  New constants `RARE_KEYWORD_MULTIPLIER_SHAPE_GATE=0.20` (matches
  the existing floor's shape gate) and `RARE_KEYWORD_MULTIPLIER_STRENGTH=0.30`.
  Multiplier formula: `score *= 1.0 + 0.30 * rare_keyword_raw`,
  gated on shape>=0.20.
- **Floor mechanism preserved unchanged** as the path for
  genuinely distinctive low-shape matches (`high_max>=0.85` OR
  `sum>=1.50` AND `shape>=RARE_KW_FLOOR_SHAPE_GATE`).
- **Director NOT converted** despite same architectural property
  (passthrough). Director's design is a recall mechanism for a
  tiny curated auteur set; passthrough is correct because Lady
  Bird's shape=0.033 needs the 0.20 contribution to surface for
  Barbie. Two passthrough lanes can serve fundamentally different
  purposes — one is a recall mechanism, one was an amplifier.

### Smoke run outcome
- 21/21 single-anchor cohorts changed, 13/14 multi-anchor changed
  (Tarantino reorder-only). This is the largest single-change
  shift in the V3.4 series — reflects rare_keyword's structural
  breadth (it fires on a wide population).
- Strong wins: Inception (mind-bending: Lost Highway/Fight Club/
  Vanilla Sky/The Prestige IN), Matrix (sci-fi classics: Blade
  Runner/Terminator/Tenet/Ghost in the Shell IN), Back to the
  Future (time-travel-specific: American Graffiti/Adam Project/
  Time Rewind IN), Spielberg adventure (Jurassic/Indy/Mummy IN,
  Star Wars OUT), John Wick (Kill Bill IN), Toy Story (Inside
  Out/Up/Ratatouille IN), Nolan trio (mind-bending tightening),
  Dark Knight (Memento/Prestige IN).
- Targeted fix: Barbie's Pleasantville/Free Guy OUT; Last Action
  Hero #1 (0.641, rare_keyword=32% of score) → #9 (0.452).
- Rare_keyword no longer dominant_lane on any candidate (was 1+
  per tag-rich anchor in V3.4.2).
- Multiplier fires 331 times in top-10s, range ×1.001 to ×1.231,
  mean ×1.071. 210 of those at ≥×1.05.
- Trade-offs: WW2 cohort gains thematic prestige (12 Years a
  Slave at #10) but loses literal WW2 (The Great Escape, Hacksaw
  Ridge); Romcom mid-list reshuffles (loses Sleepless in Seattle/
  Love Actually but top half stays solid); Female-led/Gerwig
  loses some V3.4.2 wins (Atonement, Juno, Jojo Rabbit). These
  are the cost of removing rare_keyword's lift on legitimate
  thematic-tag matches alongside the noise.

### Testing notes
- `python -m search_v2.run_similar_movies_batch --multi --limit 10`
- V3.4.2 baseline preserved at /tmp/v3_4_2_baseline_*.md.
- Calibration: STRENGTH=0.30 produces mean ×1.07 boost — same
  order of magnitude as V3.3 shape multiplier (×1.08–×1.15).
  Tunable upward if the trade-offs above prove regressive in
  real use; raising to 0.40–0.50 would partially restore lift
  to genuinely-high-rare-keyword candidates without re-
  introducing tail dominance.
- Future V3.4.4 candidate: drop the low-tier pool (idf<0.30)
  from rare_keyword entirely. The lane's name implied this
  filter; the math didn't enforce it. Would tighten the lane's
  meaning and reduce noise on tag-rich anchors further.

## V3.4.4 — rare_keyword lane removed; themes compounding bonus; bucket signal alignment
Files: search_v2/similar_movies.py, search_improvement_planning/similar_movies_test_tracker.md

### Intent
V3.4.3 fixed rare_keyword's tail dominance by converting it to a
multiplier, but two underlying issues remained: (1) themes and
rare_keyword lanes were double-counting tag overlap on the same trait
pool, and (2) the rare_keyword bucket's signal computation used
`idf >= 0.30` while its membership filter used `idf >= 0.55` — silent
no-op on tag-rich-but-no-truly-rare-tag anchors like Barbie (signal
1.000, membership empty). V3.4.4 collapses tag-overlap scoring into
a single lane (themes, with a compounding bonus) and aligns the bucket.

### Key Decisions
- **Removed rare_keyword lane entirely.** All scoring functions
  (`_rare_keyword_score_for_traits`, single/multi wrappers), all
  RARE_KW_TIER_*/COMBO_*/FLOOR_* constants, the V3.4.3
  RARE_KEYWORD_MULTIPLIER_* constants, the multiplier block in
  `_build_results`, and the floor mechanism. The lane is no longer
  in `LaneName`, `ALL_LANES`, `BASE_LANE_WEIGHTS`, or `ADDITIVE_LANES`.
- **Migrated combo bonus to themes lane.** New `_themes_combo_bonus`
  helper called from both single and multi-anchor themes scoring.
  Same constants as the old rare_keyword combo (threshold=0.50 over
  shared idf>=0.30 IDFs, base=0.05, rate=0.05, cap=0.15). Bonus added
  in absolute terms to the candidate's themes lane score (clamped
  to 1.0). The lane that does the linear scoring also does the
  compounding bonus on the same trait set — no disjoint computation.
- **Aligned bucket signal threshold with membership at 0.55.**
  Single-anchor: renamed `_high_tier_idf_sum` → `_rare_tier_idf_sum`,
  filter `idf >= IDF_TIER_LOW_MAX (0.30)` → `idf >=
  WEAVER_RARE_KEYWORD_BUCKET_IDF_MIN (0.55)`. Multi-anchor: same
  threshold change in `_compute_multi_anchor_bucket_data`'s
  cohesion-weighted signal sum.
- **Recalibrated bucket signal scale.** New constant
  `RARE_KW_BUCKET_SIGNAL_SCALE=1.00` (was 1.50 via the old
  RARE_KW_FLOOR_COMBO_SUM). Stricter idf threshold produces smaller
  sums, so the denominator shrinks proportionally. A single solid
  rare trait (idf ≈ 0.60) now maps to a signal that clears
  `WEAVER_BUCKET_INSTANTIATE_MIN (0.50)`.
- **Floor mechanism removed without replacement.** The bucket
  serves the role the floor used to (surface distinctive low-shape
  matches as exploration rows). Two parallel mechanisms for the
  same purpose was the V3 design; V3.4 weaver subsumed it cleanly.
- **Generic IDF tier constants** (`IDF_TIER_LOW_MAX=0.30`,
  `IDF_TIER_MODERATE_MAX=0.55`) replace the lane-specific
  RARE_KW_TIER_LOW_MAX/MODERATE_MAX in the themes-recall consensus
  logic.

### Smoke run outcome
- Diagnostic verification of bucket alignment: Barbie (0 rare
  traits) → signal=0.000, doesn't fire (correct). Oppenheimer (1
  rare trait, sum=0.576) → signal=0.576, fires. Schindler's List (2
  rare traits, sum=1.128) → signal=1.000, strong fire. Behavior is
  now honest about what the anchor carries.
- Single-anchor (21 cohorts): 5 identical, 9 reorder-only, 7
  changed. Multi-anchor (14 cohorts): 2 identical, 6 reorder-only,
  6 changed. Total shift ~1/3 the size of V3.4.3's full-cohort
  reshape.
- Wins: Barbie's Nights and Weekends IN at #8 (Gerwig auteur match,
  pulled by auteur bucket). Christmas on Cherry Lane and Last
  Action Hero (V3.4.3 vestiges) drop out. Dark Knight Rises gains
  Interstellar (Nolan auteur). WW2 cohort sheds 12 Years a Slave
  (V3.4.3 over-correction). Oppenheimer gains Schindler's List
  and Memento.
- Trade-off: Matrix lost V3.4.3 sci-fi classic wins (Tron, Ghost
  in the Shell, Blade Runner 1982, Terminator 1984) — V3.4.3
  multiplier had been carrying them on cyberpunk/AI tag overlap
  with raw rare_keyword scores. Under V3.4.4 they enter the
  candidate pool via themes-recall but get outranked by
  stronger-shape candidates. Recoverable via future themes
  weight tuning if needed; not a V3.4.4 blocker.

### Testing notes
- `python -m search_v2.run_similar_movies_batch --multi --limit 10`
- V3.4.3 baseline preserved at /tmp/v3_4_3_baseline_*.{md,json}.
- Bucket-signal alignment is a pure bug fix — pre-V3.4.4 the bucket
  silently no-op'd on Barbie-like anchors (signal fired,
  membership empty). Worth carrying forward as a check during
  future bucket-related work: if signal threshold and membership
  threshold differ, the bucket is internally inconsistent.
- The Matrix trade-off worth verifying in a future iteration: are
  the lost candidates (Tron / Blade Runner / Terminator) carrying
  any rare-tag overlap with Matrix that's just shy of the 0.55
  bar? If yes, themes weight bump (0.12 → 0.15) might recover
  them via the compounding bonus. If no, accept the calibration.

## V3.4.5 — themes compounding bonus reshaped from additive to multiplicative
Files: search_v2/similar_movies.py, search_improvement_planning/similar_movies_test_tracker.md

### Intent
V3.4.4's themes compounding bonus migrated the V3.1 rare_keyword combo
shape verbatim (additive 0–0.15, threshold + linear-up-to-cap), but
the additive cap was too modest at the themes-lane scale to recover
multi-tag-overlap candidates V3.4.3's shape multiplier had been
carrying (Matrix → Tron / Blade Runner / Terminator / Ghost in the
Shell). A multiplicative compounding shape on the themes lane score
(`themes = base * (1 + FACTOR * excess)`) produces visible compounding
behavior bounded by the lane's own weight (~0.10), avoiding the V3.4.3
tail-dominance failure mode (which multiplied the final score, not the
lane score).

### Key Decisions
- **Renamed `_themes_combo_bonus` → `_themes_combo_multiplier`.**
  Returns a multiplier (1.0 = no boost). Both single-anchor and
  multi-anchor themes scoring functions now compute
  `score = _clamp(base * mult)`.
- **Threshold lowered from 0.50 → 0.30.** The multiplicative shape
  needs more headroom to ramp; engaging at one shared moderate trait
  (idf ≥ 0.30) gives the multiplier room to grow with quantity.
- **FACTOR=0.50** picked over FACTOR=0.30 after running both. F=0.3
  produced almost no change vs V3.4.4 baseline (9 identical / 21
  cohorts on single-anchor) — too gentle to actually compound. F=0.5
  produced visible movement (3 identical / 21) with clear directional
  wins on prestige/genre cohorts.
- **Lane-internal multiplier is safe** where V3.4.3's final-score
  multiplier wasn't. Themes is proportional and weighted at ~0.10, so
  a 2× lane score → ~+0.05 final-score lift. Decays with rank
  naturally.

### Smoke run outcome (FACTOR=0.5 vs V3.4.4 baseline)
- Single-anchor: 3 identical / 8 reorder / 10 changed of 21.
- Multi-anchor: 5 identical / 4 reorder / 5 changed of 14.
- Strong wins: Matrix recovers Tron (1982) at #9; Oppenheimer recovers
  The Pianist at #10 (multi-tag prestige Holocaust overlap — exactly
  the case the bonus is designed for); WW2 cohort gets Empire of the
  Sun (Spielberg WW2 literal genre fit); Get Out gets Autopsy of Jane
  Doe (better horror match).
- Trade-offs: Barbie's Last Action Hero comes back at #10 (was V3.4.3
  noise; now earning the spot on genuine meta-comedy theme overlap,
  different mechanism than V3.4.3's rare_keyword crutch). Easy Man
  drops out of top-10. Nolan trades Fight Club for Lost Highway
  (both mind-bender adjacent).

### Testing notes
- `python -m search_v2.run_similar_movies_batch --multi --limit 10`
- V3.4.4 baseline (additive) preserved at /tmp/v3_4_4_baseline_*.{md,json}.
- F=0.3 trial preserved at /tmp/v3_4_5_factor03_*.{md,json}.
- The Matrix recovery is partial: Tron back, but Blade Runner /
  Terminator / Ghost in the Shell / TRON: Legacy still don't surface
  even at F=0.5. Their moderate-tier overlap with Matrix isn't strong
  enough to clear other candidates' shape advantages with the
  compounding amplification alone. Recovering them would require
  themes weight bump (0.12 → 0.15) or reintroducing per-trait tier
  coefficients — V3.4.6 candidate if it bothers anyone in real use.

## V3.4.6 — Franchise fatigue (single-anchor)
Files: search_v2/similar_movies.py, search_improvement_planning/similar_movies_test_tracker.md

### Intent
Strong franchise anchors (Star Wars, John Wick, MCU, DC, Sharknado,
LOTR/Hobbit) flooded the single-anchor top section with their own
sequels because shape similarity + the franchise lane stack additively.
The user wants *some* franchise siblings in results but also genuine
non-franchise alternatives. The fix is a hard ratio cap on franchise
vs. non-franchise placements in the greedy weaver.

### Key Decisions
- **Threshold = 0.34** (≈1:3 franchise-to-non ratio cap; ≤25% of top 10).
  Permits 2-3 franchise entries before forcing non-franchise picks.
  Per user direction.
- **Hard ban, not soft demotion.** No score modification — the weaver
  simply skips franchise candidates from every bucket while the gate
  is active. Clarity wins; no interaction with the multiplier/floor
  stack to reason about.
- **"Franchise" defined broadly per user spec**: candidate's
  `(lineage_entry_ids ∪ shared_universe_entry_ids)` intersects the
  anchor's `(lineage_entry_ids ∪ shared_universe_entry_ids)`. Cross-
  matching anchor lineage ↔ candidate universe and vice versa. Subgroup
  tags excluded.
- **Single-anchor only.** Multi-anchor consensus on franchise membership
  is real signal, not stacking artifact. Multi-anchor flow doesn't pass
  the new params; gate is inert there.

### Implementation
- New constant `FRANCHISE_FATIGUE_THRESHOLD = 0.34`.
- `_peek_next_eligible_for_bucket` gains `enforce_franchise_fatigue`,
  `is_franchise_by_movie`, `franchise_count`, `non_franchise_count`
  kwargs. Skip logic mirrors format-lock/shorts-cap patterns.
- `_weave_candidates` tracks counts alongside `shorts_count` and
  forwards into peek.
- `_build_results` forwards the two new params.
- `_run_single_anchor_similarity` builds `is_franchise_by_movie` once
  from `candidate_rows` data (no extra DB calls) and passes
  `enforce_franchise_fatigue=True`.

### Verification
- 21-anchor single-anchor smoke: 11 IDENT / 2 REORD / 8 DIFF
- Multi-anchor smoke: byte-identical (as expected)
- Non-franchise anchors all IDENT (Inception, Get Out, Oppenheimer,
  Pulp Fiction, etc.) — no over-fire risk
- Wins are uniformly franchise-cleanup: Star Wars (6 SW films
  displaced), Dark Knight (3 Batman films), Matrix (Resurrections +
  Animatrix), Sharknado, John Wick, LOTR
- Baselines: /tmp/v3_4_5_factor05_*.{md,json} (pre), /tmp/v3_4_6_*.{md,json} (post)

### Re-run
- `python -m search_v2.run_similar_movies_batch --multi --limit 10`

## V3.4.6.1 — Extend franchise fatigue gate to tail-append loop
Files: search_v2/similar_movies.py, search_improvement_planning/similar_movies_test_tracker.md

V3.4.6 only enforced the fatigue gate inside the greedy top-section
loop (slots 0..TOP_SECTION_SIZE-1). For `limit > 10`, the tail-append
loop walked V3-rank candidates and appended unconditionally, so
franchise siblings banned from the top section reappeared at rows 11+.

Fix: tail-append now consults the same `franchise_count` /
`non_franchise_count` counters from the top-section loop, applies the
same threshold check, and updates counters as it appends. The rule
operates over the **whole result list** — counters are shared across
both loops so the cap remains coherent end-to-end.

Verified at limit=20 against franchise-heavy anchors:
- Batman 1989: 5 franchise / 15 non (25%)
- Star Wars 1977: 3 franchise / 17 non (15%)
- John Wick: 3 franchise / 17 non (15%) — Ballerina out entirely

## V5 Iter 8 + Iter 9 — Phase 6 sibling-task context, Phase 7 soft FACETS fold, vacuous-spec extraction filter, drop per-candidate verdict pathway
Files:
- search_v2/endpoint_fetching/category_handlers/prompt_builder.py (Iter 8, Phase 6)
- search_v2/endpoint_fetching/category_handlers/handler.py (Iter 8, Phase 6)
- search_v2/full_pipeline_orchestrator.py (Iter 8, Phase 6)
- search_v2/run_query_generation.py, search_v2/run_specs.py (Iter 8, Phase 6 — diagnostic threading)
- search_v2/stage_4_execution.py (Iter 8, Phase 7)
- search_v2/endpoint_fetching/category_handlers/prompts/buckets/preferred_representation_fallback_objective.md (Iter 8 + Iter 9)
- search_v2/endpoint_fetching/category_handlers/prompts/buckets/semantic_preferred_deterministic_support_objective.md (Iter 8 + Iter 9)
- search_v2/endpoint_fetching/category_handlers/prompts/buckets/audience_suitability_deterministic_first_objective.md (Iter 8 + Iter 9)
- search_v2/endpoint_fetching/category_handlers/prompts/endpoints/keyword.md (Iter 8 + Iter 9)
- search_v2/endpoint_fetching/category_handlers/prompts/endpoints/semantic.md (Iter 8 only)
- search_v2/endpoint_fetching/category_handlers/output_extractor.py (Iter 9 — vacuous-spec filter)
- schemas/keyword_translation.py (Iter 9 — drop verdict pathway)
- search_v2/endpoint_fetching/category_handlers/schema_factories.py (Iter 9 — drop _WalkThenCommitOutputBase)
- search_improvement_planning/search_overheaul_test_tracker.md (Iter 8 + Iter 9 entries + active catalog updates)

### Intent

Two iterations bundled in one uncommitted working tree. Iter 8
shipped Phase 6 (sibling-task context as evidence-injection at the
handler) + Phase 7 (soft FACETS fold via geometric-mean-with-floor
`_FACETS_FOLD_FLOOR=0.1`). Iter 9 followed up with two scoped
fixes flagged from Iter 8 verification: a vacuous-spec extraction
filter to close the residual edge case Phase 7's floor was
masking, and a revert of Phase 5's per-candidate verdict pathway
on the multi-endpoint keyword walk because the per-candidate
abstraction could not natively express the union-level superset
reasoning that keyword.md's commit test calls for.

### Key Decisions

- **Phase 6 — sibling-task context, not sibling-result feedback.**
  The handler user message gains a `<sibling_categories
  combine_mode="...">` block listing each parallel category's
  `retrieval_intent` verbatim. Sibling tasks (instruction-time,
  parallel-safe) preserve per-call isolation; sibling results
  (would require sequencing) do not. Iter 8 confirmed this is the
  first intervention class to move the trip-wire ceiling that
  held across four prompt-edit iterations (Iter 5/6/7/7.x).
- **Phase 7 — geometric-mean-with-floor (EPS=0.1).** FACETS PRODUCT
  becomes `geomean([max(s, 0.1) for s in scores])`. Single-zero on
  a 2-cat trait scores 0.316 instead of zeroing; n-cat traits
  scale at `floor^(1/n)`. Synthetic numerical sweep across
  representative score patterns confirmed all-clean cases
  unchanged and single-zero cases survivable. Real-query EPS sweep
  deferred — `run_full_pipeline` requires the FastAPI Postgres pool
  startup hook, no orchestrator_batch CLI yet.
- **Iter 9 Change 1 — vacuous-spec extraction filter (kept).** New
  `_is_vacuous_spec(route, wrapper)` helper in
  `output_extractor.py` filters wrappers whose params are
  structurally empty (keyword `finalized_keywords=[]`, semantic
  `space_queries=[]`, all-null metadata `column_spec`). Symmetric
  with `coverage_commitments.{route}.verdict=abstain`. Closes
  the Q5 GENRE empty-commit edge case at the extraction layer as
  defense-in-depth.
- **Iter 9 Change 2 — per-candidate verdict pathway reverted (per
  user direction "Go with option 3").** Deleted
  `PotentialKeywordWithVerdict`, `AttributeAnalysisWithVerdict`,
  `_WalkThenCommitOutputBase`. `KeywordWalk.attributes` reverts to
  `list[AttributeAnalysis]`. `KeywordQuerySpecSubintent.finalized_keywords`
  reverts to LLM-emitted with `min_length=1` (was server-derived
  from verdicts). The bucket-level
  `coverage_commitments.{route}.verdict_reason→verdict` pathway is
  retained — that operates at the union level naturally and was
  the part of Phase 5 that delivered value. Prompt sections in
  keyword.md and the three bucket objectives reframed to one
  abstention level only.

### Iter 8 outcome

V5 suite (run_specs across 25 queries):
- cats 86 → 82, risk 26 → 21 (below phase_3 baseline of 23 for
  the first time across all V5 iterations).
- STORY_THEMATIC_ARCHETYPE kw 16 → 13 (matched phase_3 plateau).
- Q15 GENRE narrowed 5 paraphrastic → 1 canonical FANTASY; Q18
  STORY_THEMATIC abstained on stretching (sub-shape C wins).
- Positive controls Q9 / Q12 / Q25 held; schema validation errors
  at 0.
- One concern flagged for follow-up: Q5 GENRE empty-commit
  (`keyword_finalized=[]` paired with `coverage_commitments.keyword.verdict=commit`)
  — Phase 7 floor masked the trait-death damage but the underlying
  verdict-pathway-vs-bucket-commit inconsistency drove Iter 9.

### Iter 9 outcome — mixed; not a clean ship

V5 suite (run_specs across 25 queries):
- cats 82 → 90 (regressed +8), risk 21 → 25 (regressed +4, also
  above phase_3 baseline of 23).
- F2 ALL count 0 → 1: **Q5 plural-intent `[ACTION, THRILLER] ALL`
  restored** — the empty-commit case is structurally closed by
  the schema's `min_length=1` revert + the extraction-time
  filter. This is the unambiguous structural win.
- empty kw_finalized count 1 → 0.
- Per-query narrowing wins: Q21 folk horror 3→1 paraphrases
  collapsed; Q22 reconciliation FEEL_GOOD stretch dropped; Q23
  psychological mysteries 3→1 narrowed; Q16 brutal MMA narrowed
  + improved.
- Per-query narrowing regressions: Q12 PLOT_TWIST returned (Iter
  8 had dropped); Q18 Donnie Darko STORY_THEMATIC stretching
  re-emerged (Iter 8 sub-shape C win partially undone) plus new
  GENRE stretches; Q15 GENRE singular → paraphrase pair
  (sub-shape C win partially undone); Q14 passion projects new
  4-paraphrase NARRATIVE_DEVICES commit; Q20 dark trait commits
  the canonical-stretching `[DRAMA]`.

### Verdict

Per the brief's own stop conditions, Iter 9 is NOT auto-shippable:
trip-wire risk regressed past phase_8 AND past phase_3 baseline.
Per-candidate verdicts WERE doing real narrowing work that the
union-level commit cannot natively replicate; the brief's
hypothesis that union-level reasoning would match or improve was
falsified by the data. The Q5 plural-intent ALL restoration is a
real structural win driven by the `min_length=1` revert plus the
extraction filter (both halves of Change 2 + Change 1 were
load-bearing). Iter 8's evidence-injection win (sibling-context)
held architecturally — the regressions are downstream of removing
the per-candidate abstention mechanism that translated
sibling-context steering into commit-shape changes.

### Recommendation surfaced for user direction

The tracker entry at `### Iteration 9` documents three forward
paths: (a) ship as-is, accepting Q5 win and narrowing
regressions; (b) revert Change 2, keep Change 1 — preserves
Iter 8 narrowing wins and retains extraction-time defense-in-depth
on vacuous specs; (c) keep Change 2 architecturally and invest in
tightening the union-level prompt to recover narrowing power. The
working tree currently reflects (a); the user decides.

### Testing notes

- V5 suite (run_specs) does NOT replace per-trait trait_score
  validation on real candidates. Per Iter 7 lesson #3 (now
  confirmed for the FOURTH time on Iter 9: Q5 trait `bloody`
  flipped FACETS↔FRAMINGS, Q11 atomization fluctuated 1↔2
  traits, Q14 NARRATIVE_DEVICES newly fired), single-run-on-25-queries
  cannot distinguish intervention signal from cross-run Step 2
  noise on marginal cases. Multi-run aggregation is now a
  measurement prerequisite for any future iteration.
- Phase 7 EPS sweep on real catalog queries deferred — needs an
  orchestrator_batch CLI runner that can stand the FastAPI
  Postgres pool up standalone.
- Sibling-task context propagation tested via run_specs (the
  diagnostic runner threads `sibling_calls` + `combine_mode`
  through both Step 3 V4 and the per-call handler invocation).
  Production code path uses
  `_decompose_and_generate` → `_process_category_call` to compute
  siblings once and propagate; identity-based filter
  (`s for s in all_calls if s is not cc`) correct because each
  CategoryCall is a distinct object inside its decomposition.
- Tests not modified per .claude/rules/test-boundaries.md. The
  existing `unit_tests/test_full_pipeline_promotion_tiers.py`
  mocks remain compatible because dataclass defaults preserve
  pre-existing call sites; the keyword schema's revert to
  LLM-emitted `finalized_keywords` matches the pre-Phase-5
  shape, so any test stubbing that field continues to work.

## V3.4.7 — Director fatigue (single-anchor)
Files: search_v2/similar_movies.py, search_improvement_planning/similar_movies_test_tracker.md

### Intent
Auteur anchors (Nolan, Tarantino, Tim Burton, Miyazaki, Peter Jackson)
flood single-anchor results with same-director films because the
auteur bucket + director floor + shape similarity all align. Symmetric
with V3.4.6 franchise fatigue: hard-ban candidates sharing a director
with the anchor once the placed director-match ratio exceeds 0.34.

### Key Decisions
- **Same threshold (0.34)** as franchise. Per user direction.
- **Independent counters** from franchise — a candidate can be both a
  franchise sibling and a same-director match (TDK is both relative
  to Batman 1989), but the two gates count separately and ban
  independently. Sharing counters would conflate distinct dimensions.
- **No extra DB calls.** `is_director_match_by_movie` is built from
  `director_candidate_terms.keys()` — every entry there is already a
  candidate sharing ≥1 director with the anchor.
- **Single-anchor only.** Multi-anchor consensus on a director is real
  signal (Nolan trio agreeing on Nolan-style is genuine). Multi-anchor
  flow doesn't pass the new params; gate is inert there.
- **Tail loop also gated.** Same shared-counter pattern as V3.4.6.1 —
  the rule operates over the whole result list at any limit.

### Implementation
- New constant `DIRECTOR_FATIGUE_THRESHOLD = 0.34`.
- `_peek_next_eligible_for_bucket` gains parallel `enforce_director_fatigue`,
  `is_director_match_by_movie`, `director_match_count`,
  `non_director_match_count` kwargs. Independent skip clause.
- `_weave_candidates` tracks both pairs of counters and updates after
  every placement. Tail-append loop checks both gates and updates
  both counter pairs.
- `_build_results` forwards the two new params.
- `_run_single_anchor_similarity` builds `is_director_match_by_movie`
  from `director_candidate_terms.keys()` and passes
  `enforce_director_fatigue=True`.

### Verification
- 21-anchor smoke: 10 IDENT / 5 REORD / 6 DIFF
- Multi-anchor: byte-identical (as expected)
- 6 DIFFs are all auteur anchors (TDK, Spirited Away, Inception,
  Oppenheimer, Pulp Fiction, LOTR Fellowship); non-auteur anchors all
  IDENT or REORD
- Limit=20 Tim Burton anchor (Batman 1989): only 4 Burton films
  surface vs. ~9-10 pre-fix; tail loop confirmed gating
- Baseline saved at /tmp/v3_4_7_*.{md,json}

### Re-run
- `python -m search_v2.run_similar_movies_batch --multi --limit 10`

## V3.4.8 — Director multiplicative-on-shape (replaces V3.4.7 fatigue)
Files: search_v2/similar_movies.py, search_improvement_planning/similar_movies_test_tracker.md

### Intent
V3.4.7 director fatigue banned same-director candidates regardless of
match quality, removing genuinely-good auteur matches (Inception
losing The Prestige, Spirited Away losing Princess Mononoke).
The real problem was that the additive director lane gave the same
+0.20 boost regardless of shape similarity, so weak-shape candidates
rode the boost into prominence. Multiplicative-on-shape attacks the
cause: amplify only when shape supports the match.

### Key Decisions
- **Convert director from passthrough additive to multiplicative.**
  Same gate-and-amplify pattern as studio.
- **Strength 0.30, gate 0.30.** Lower gate than studio (0.60) because
  director is a stronger stylistic signal; higher strength than studio
  (0.10) because director is the primary auteur-sensibility lane.
- **Revert V3.4.7 fatigue entirely.** No separate counters, no peek
  gate, no tail-loop check. Cleaner architecture.
- **Normalize director scores to [0, 1].** Was [0, 0.20]/[0, 0.30].
  Relative structure preserved (auteur full = 1.0, cohesion scales).
- **Keep director floor + auteur bucket.** They provide minimum
  representation; the multiplier controls upper-bound clustering.

### Implementation
- Remove `director` from `ADDITIVE_LANES` and `PASSTHROUGH_LANES`;
  set `BASE_LANE_WEIGHTS["director"] = 0` for debug visibility.
- Add `DIRECTOR_MULTIPLIER_SHAPE_GATE` and `DIRECTOR_MULTIPLIER_STRENGTH`.
- Update `_single_anchor_director_score` and `_multi_anchor_director_score`
  to return [0, 1].
- Apply multiplier in `_build_results` next to studio.
- Strip out V3.4.7 fatigue: constant, params on
  `_peek_next_eligible_for_bucket` / `_weave_candidates` /
  `_build_results`, tail-loop gate, and `is_director_match_by_movie`
  builder in `_run_single_anchor_similarity`.

### Verification
- 21-anchor smoke: 12 IDENT / 2 REORD / 7 DIFF
- Multi-anchor: byte-identical
- Tim Burton (Batman 1989, limit=20): 9-10 Burton films → 3
- Nolan (TDK, top 10): 8 → 3
- Other auteur anchors: modest reductions (3-4 same-director kept)
- Baselines at /tmp/v3_4_8_*.{md,json}

### Open tuning question
TDK reduction may be too aggressive (8→3). Caused by TDK's
normalization weighting shape at only 0.34 (heavy franchise/prestige
allocation), so removing additive director (+0.20) outweighs the
×1.30 multiplier gain. If real use shows auteur anchors with high-
prestige weighting losing too much, bump
`DIRECTOR_MULTIPLIER_STRENGTH` from 0.30 toward 0.40-0.50.

### Re-run
- `python -m search_v2.run_similar_movies_batch --multi --limit 10`

## V3.4.9 — Director gap-boost (rubber-band on auteur silence)
Files: search_v2/similar_movies.py, search_improvement_planning/similar_movies_test_tracker.md

### Intent
Address the V3.4.8 side effect where the multiplier suppresses
auteur films too aggressively in the tail (slots 6–10) for some
anchors. TDK lost most Nolan films because the auteur bucket fills
its target in early slots and no remaining director match could
clear the multiplier-gated bar against stronger-shape non-director
candidates. The rubber-band re-amplifies director matches once the
weaver has gone several slots without placing one.

### Mechanism
- Bucket-weaver tracks `consecutive_non_director` (resets on each
  director-match placement, increments otherwise).
- Per slot: `k = max(0, gap - DIRECTOR_GAP_THRESHOLD + 1)`,
  `gap_boost_multiplier = (1 + DIRECTOR_GAP_INCREMENT) ** k`.
  Boost engages AT the threshold (gap=3 → k=1 → ×1.10), then
  compounds (gap=4 → ×1.21, gap=5 → ×1.331, …).
- For MMR comparison only: `effective_score = c.score *
  gap_boost_multiplier` for director-match candidates, else
  `c.score`. The persisted score is unchanged so evidence rows
  still match the deterministic pipeline output.
- Auteur bucket peek-eligibility is re-opened past quota whenever
  `gap_boost_multiplier > 1`, so the rubber-band has a candidate
  to amplify.
- `director_match` semantics: any candidate sharing a director
  with the anchor — no shape gate (unlike the V3.4.8 multiplier).

### Key Decisions
- **Compounding multiplicative, not linear additive.** User
  framing maps to `1.10^k` naturally. Geometric growth moves tail
  rankings (k=2 → ×1.21) without needing a large per-step rate.
  Initial implementation used `score + extra * capacity` and
  produced zero result changes because `capacity` was gated on
  V3.4.8's shape gate, leaving the rubber-band with nothing to
  amplify.
- **No shape gate on the rubber-band.** V3.4.8 multiplier gates
  on shape (suppress weak-shape clustering at source). V3.4.9
  rubber-band gates on placement history (re-amplify when result
  is silent). Different problems, different gates. If the
  rubber-band were also shape-gated, it would inherit V3.4.8's
  suppression and never engage for the weak-shape candidates the
  user wants surfaced.
- **Single-anchor only.** Multi-anchor `director` score is
  cohesion-scaled across anchors, not a single-auteur signal —
  rubber-band semantics don't carry over. Threaded
  `enforce_director_gap_boost` flag through `_build_results` and
  `_weave_candidates`; only `_run_single_anchor_similarity`
  passes `True`.
- **Score-vs-effective-score split.** Slot-local MMR adjustment
  only; persisted `c.score` left untouched. Evidence diagnostics
  still reflect the deterministic pipeline (multipliers, floors,
  lane contributions all unchanged from V3.4.8).

### Verification
- 21-anchor smoke: 1 REORD (TDK) / 20 IDENT
- Multi-anchor: byte-identical (flag off)
- TDK: Watchmen (#10, 0.402) replaced by The Prestige (Nolan,
  0.350). Trace: dir at slots 1,2; non-dir at 3,4; dir at 5; non-
  dir at 6,7,8,9; entering slot 10 gap=4 → k=2 → ×1.21; Prestige
  0.350 × 1.21 = 0.424 > Watchmen 0.402.
- Other 20 anchors stayed identical because either auteur density
  was already saturated (Spirited Away with 7 Miyazaki films),
  the pool was exhausted of remaining director matches, or
  post-boost effective scores were still below competing picks.

### Implementation
- New constants `DIRECTOR_GAP_THRESHOLD = 3`,
  `DIRECTOR_GAP_INCREMENT = 0.10`.
- `_CandidateScore.director_match` kept; the abandoned
  `dir_boost_capacity` field and corresponding `score_no_dir_mult`
  parallel tracking in `_build_results` removed.
- New `enforce_director_gap_boost` param threaded through
  `_build_results` → `_weave_candidates`. Defaults `False`.
- Single-anchor caller (`_run_single_anchor_similarity`) sets
  `True`.
- Weave loop: compute `gap_boost_multiplier`, apply to
  director-match candidates' `effective_score` for MMR
  comparison, advance `auteur_reopen` when active.

### Re-run
- `python -m search_v2.run_similar_movies_batch --limit 10`

## Step 0 multi-title similarity wiring
Files: schemas/step_0_flow_routing.py, search_v2/step_0.py, search_v2/similar_movies.py, search_v2/run_step_0.py

### Intent
Wire Step 0 to drive the multi-anchor similarity flow already implemented in
`run_similar_movies_for_ids`. Previously Step 0's `SimilarityFlowData`
carried a single `similar_search_title` + `release_year`; now it carries
a list of `SimilarityReference` entries so frames like "kung fu panda
meets jaws", "movies like inception, interstellar, or arrival", and
bare title lists "Godfather and Goodfellas" route directly to the
existing multi-anchor pipeline instead of falling back to standard.

### Key Decisions
- **List replaces single field (breaking schema change).** Adding a
  parallel `references` list alongside the old single-title fields
  would have kept the LLM emitting two parallel sources of truth; the
  cleaner shape is a single list (length 1 for single-reference
  frames, length N for multi). Empty list represents "no similarity
  reference" — no need for the prior empty-string convention. Schema
  validator now enforces "references non-empty when
  should_be_searched=True".
- **Drop-failed-resolve, not all-or-nothing.** Per the user's spec,
  silently drop references that don't match any movie and continue
  with the survivors; only return empty results when every reference
  fails. Implemented in `_resolve_similarity_anchors` (loops over
  references, dedupes anchor IDs while preserving order).
- **Installment-disambiguation markers are NOT qualifiers.** A cast/
  director marker that identifies which installment of a multi-version
  title the user means ("the batman with michael keaton" → Batman
  1989) stays inside the TitleObservation and lifts a release_year
  onto the matching SimilarityReference. Free-standing descriptors
  ("with kids in it", "from the 80s") remain qualifiers and block
  similarity. New prompt section + worked example (Example 20)
  draws this line for the LLM.
- **enable_primary_flow rule loosened for clean multi-title frames.**
  Previously `len(titles_observed) > 1` always fired the standard
  flow as a co-flow. Now multi-title is similarity-only when every
  title sits inside a single similarity frame (or the bare-title-list
  shape) and there are no qualifiers — preventing duplicate routing
  on the canonical multi-anchor query.

### Planning Context
The downstream similar-movies pipeline (`run_similar_movies_for_ids`)
already routes single-anchor vs. multi-anchor via list length, so the
change is contained in: (a) the Step 0 schema/prompt, (b) the title→
anchor-id resolution helper, (c) the executor entry point. No changes
to lanes, scoring, or the multi-anchor cohort path itself.

### Testing Notes
- Mass-eval Step 0 against the boundary-example set in `step_0.py` —
  Examples 18–21 are the new multi-anchor / installment-disambig
  cases. Existing examples 6 and 16 had their expected output flipped
  (Godfather+Goodfellas now → similarity).
- End-to-end: run `python -m search_v2.run_step_0 "kung fu panda
  meets jaws"` and `python -m search_v2.run_step_0 "the comedy of
  bug's life with the animation of klaus"` — the first should fire
  similarity with 2 anchors, the second should route to standard via
  qualifier presence.
- Resolution edge case: a query referencing one real movie and one
  fictional title ("movies like Inception and Frumblewax") should
  resolve only the real one and run a single-anchor similarity search.

## Similar-movies execution: latency-only optimizations (no scoring change)
Files: search_v2/similar_movies.py, db/postgres.py, db/qdrant.py,
       api/main.py, search_v2/run_step_0.py,
       search_v2/run_similar_movies_batch.py,
       implementation/misc/event_loop.py, pyproject.toml

### Intent
Cut similarity-search end-to-end latency. Every change preserves
scoring semantics — verified by running the full single + multi
anchor batch and diffing against the committed baseline JSON: all
rankings + scores identical (single-anchor: 0 diffs; multi-anchor:
all 14 cohort rankings identical, 5 cosmetic debug-count
differences ≤3 candidates each in the tail of the 10-25K candidate
pool, from numpy vs Python FP precision in the centroid).

### Key Decisions
- **Parallelize the 4 prefetches in `run_similar_movies_for_ids`.**
  anchor_rows, anchor_vectors, studio_entries, director_terms are
  independent — one `asyncio.gather` instead of 4 sequential awaits.
- **Qdrant `query_batch_points` for the 8-named-vector shape
  search.** New `_query_spaces_batch` helper replaces the prior
  `asyncio.gather(*[_query_space(...)])` pattern. Single round
  trip, server-side parallelism, shared HNSW cache across the
  batch. `_query_space` removed (was the only caller).
- **gRPC opt-in via env var `QDRANT_PREFER_GRPC=1`.** Docker-compose
  currently only exposes port 6333; comment in `db/qdrant.py`
  documents how to enable (add `6334:6334` to compose, restart,
  set env var). Default stays HTTP so the batch keeps working
  unchanged.
- **Module-cached `_load_studio_entries_by_company_id`.** The
  Postgres lookup it wraps is a static registry-shape query that
  never changes within a process; async-lock-guarded
  initialize-once pattern.
- **Postgres pool: min_size 2→4, max_size 10→25.** Sized for the
  similarity flow's fan-out (single-anchor lane gather fires up
  to 11 concurrent ops; multi-anchor has two waves of 9 + 6).
  Eliminates pool-acquire serialization at burst.
- **Multi-anchor cosine + centroid: numpy.** Replaces the
  `_dot(left, right)` over `zip(...)` loops and the 3072-wide
  Python list comprehension for centroid mean with `arr @ arr.T`
  (pairwise) and `arr.mean(axis=0)` (centroid). ULP-level FP
  differences only.
- **Auteur fetch folded into the multi-anchor gather.** Was a
  serial `await fetch_auteur_term_ids()` before the gather; now a
  small composite task (`_fetch_auteur_and_director`) awaits
  auteur, applies the gate, and awaits director if needed —
  running in parallel with shape/franchise/studio/etc.
- **uvloop install at entry points.** New
  `implementation/misc/event_loop.py:install_uvloop()` called from
  `api/main.py`, `search_v2/run_step_0.py`,
  `search_v2/run_similar_movies_batch.py`. Idempotent, no-ops
  cleanly if uvloop is missing. ~2x throughput on socket-heavy
  fan-outs.
- **Vectorized `_build_results` scoring loop.** The hottest
  pure-Python phase (10 dict-of-dict lookups × N candidates +
  weighted sum + 6 conditional multipliers per row over 5K-30K
  candidates) is now one `score_matrix` build, one BLAS dot for
  base scores, masked elementwise vector ops for the 7
  multipliers, vectorized floor selection, then a single Python
  pass to materialize the diagnostic-bearing `_CandidateScore`
  objects. Dominant-lane uses an 8-iteration loop over additive
  lanes to match Python's `max(..., key=...)` first-wins-on-tie
  semantics exactly. Sources-list ordering preserves the
  original's "ALL_LANES order, studio appended last".
- **`_ParsedRowSlice` for orchestrator-level row dedup.** New
  `_parse_candidate_rows` builds a per-candidate slice
  (format_bucket, medium_tags, country_tags, themes_traits,
  franchise_pool) once per request; orchestrator-level
  duplicate parses (format_bucket called 2x per candidate,
  franchise pool re-parsed for fatigue check) read from the
  slice. Scoring helpers still take row dicts — narrow scope
  keeps the diff small.

### Validation
- Existing batch baselines preserved at /tmp/perf_baseline/.
- `python -m search_v2.run_similar_movies_batch --multi` completes
  cleanly and emits identical ranked output.

## CHARACTER_FRANCHISE fanout — force CENTRAL prominence and prefer_lineage=True
Files: search_v2/endpoint_fetching/category_handlers/output_extractor.py
Why: CHARACTER_FRANCHISE referents (Batman, Bond, Spider-Man, Sherlock) by definition anchor their films and name a specific character-anchored franchise main line, so the prominence/lineage signals should never be left to the LLM's discretion in this category.
Approach: In `_fanout_to_fired_endpoints`, hard-code `prominence_mode=CharacterProminenceMode.CENTRAL` on the synthesized `CharacterTarget` (was `DEFAULT`) and `prefer_lineage=True` on the synthesized `FranchiseQuerySpec` (was unset → default False). Safe because the fanout schema itself does not surface either field to the LLM, and `FranchiseQuerySpec._validate` already soft-coerces `prefer_lineage` back to False when mechanically incompatible (multi-name list, SPINOFF flag, populated subgroup_names — none of which the fanout path emits, but the coercion remains a safety net). Updated the prominence-exploration stub string and adjacent comments to reflect the new behavior.
Testing notes: Confirm that CHARACTER_FRANCHISE queries (e.g. "Batman movies", "James Bond") rank lineage-only matches above shared-universe-only matches in the franchise path, and that character path scores no longer credit titles where the named character appears only peripherally.

## Implicit-prior rerank — popularity primary, quality only on popularity=none
Files: search_v2/full_pipeline_orchestrator.py
Why: For "superman movies" the boost was 20.7% (Man of Steel) vs 26.7% (Donner Cut) because both axes were summed. With popularity saturated for tentpole franchises (0.97 ≈ 0.96), the quality axis silently dominated reranking — Donner Cut's reception 75 vs Man of Steel's 61.4 created a ~6pp swing the user did not want. Fix is to make popularity the only implicit axis when it is active.
Approach: In `_apply_implicit_prior_rerank_for_branch`, replaced the additive `boost = qual*qual_sig + pop*pop_sig` with a single-axis selector: if `policy.popularity_prior.direction != "none"` and `popularity_cap > 0`, use only popularity; otherwise (popularity inactive — usually because explicit query coverage owns it) fall back to quality alone. Both axes guarded by direction + cap so a None strength still short-circuits the rerank. Updated module docstring to reflect the new contract. Per-category pop/quality weighting design was discussed but deferred — this is the simpler interim policy.
Testing notes: Smoke-test "superman movies" (Donner gap should collapse), "underrated 90s thrillers" (quality should still apply since popularity_prior would be inverse, but verify direction handling), and an explicit-popularity query like "popular comedies" (popularity should be policy.direction=none → quality fallback kicks in).

## CHARACTER_FRANCHISE fanout — exclude umbrella universes from franchise_forms
Files: search_v2/endpoint_fetching/category_handlers/endpoint_registry.py
Why: For "superman movies" the handler LLM emitted `franchise_forms=["Superman", "DC Extended Universe"]`, OR-unioning every DCEU film into the franchise retrieval. The CHARACTER_FRANCHISE_FANOUT bucket is by construction a single character-anchored lineage, and the output_extractor already hard-forces `prefer_lineage=True` — but the validator silently flips it back to False on multi-name lists, so the lineage bias was being defeated by the prompt itself.
Approach: Rewrote the `franchise_form_exploration` template description in `CharacterFranchiseFanoutSchema` to (1) keep `Umbrella` as a context slot the LLM can acknowledge but (2) explicitly forbid copying it into `Distinct forms`, and (3) restrict `Distinct forms` to true alternative names of the character's own franchise (acronyms, common short/long forms — e.g. "james bond"/"007", "the lord of the rings"/"lotr"). Added concrete worked examples for Superman, James Bond, Spider-Man, and LOTR so the LLM has anchored exemplars. Mirrored the constraint in the `franchise_forms` list-field description so both surfaces agree.
Design context: The standalone franchise prompt (`search_v2/endpoint_fetching/franchise_query_generation.py`) deliberately INVITES umbrella sweeps via multi-name OR ("marvel" + "marvel cinematic universe"); the fanout case is the opposite because the upstream router already committed to a specific character. Keeping the two prompts divergent is intentional.
Testing notes: Re-run "superman movies" and confirm `franchise_forms=["superman"]` (or similar single-form list) and that DCEU non-Superman titles no longer appear in the franchise channel. Also spot-check "spider-man movies" (should not pull in Avengers/MCU), "wolverine movies" (should not pull in broader X-Men umbrella unless wolverine's own series is the only natural canonical), and "frodo" / "lotr" (single franchise, alternates allowed). The validator's multi-name → prefer_lineage=False soft coercion remains as a safety net but should no longer fire on these queries.

## CENTRAL_TOPIC — re-route to semantic-preferred bucket; flip keyword policy from preferred→fallback to broader-superset-or-abstain
Files: schemas/trait_category.py, search_v2/endpoint_fetching/category_handlers/prompts/categories/additional_objective_notes/central_topic.md, search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/central_topic.md
Why: Old policy treated keyword as primary ("preferred") and semantic as fallback. For "movies about WW2" that produced keyword-only commits on narrow members (e.g., `WAR_EPIC` alone) that biased the category score toward sub-forms and silently zeroed attribute-satisfying films outside that sub-form (e.g., WW2 dramas like *Schindler's List*). Semantic was treated as gap-fill rather than the specificity-bearing channel.
Approach: (1) Re-routed CENTRAL_TOPIC's bucket from `PREFERRED_REPRESENTATION_FALLBACK` to `SEMANTIC_PREFERRED_DETERMINISTIC_SUPPORT`. The SPDS bucket prompt already encodes "semantic almost always fires; deterministic adds parallel sharpness; over-coverage is acceptable" — no bucket prompt edits needed. The keyword endpoint's superset test ("over-pull acceptable, gaps not, stretching not, narrow-only-with-gaps → abstain") was already correct — no schema edits needed. (2) Rewrote the category-specific `additional_objective_notes` to flip the policy: semantic almost always fires (must mirror the request's specificity, not generalize it); keyword fires only when its ANY-mode union is a true superset of the subject; cross-category keyword borrowing is explicitly allowed when the union still passes the superset test; narrow-only commits that would bias toward a sub-form must abstain on keyword and let semantic carry the call. (3) Rewrote the seven few-shot examples to cover the full decision matrix: dual-fire with broad+narrow union (WW2), dual-fire with cross-category union (real singers), dual-fire with single broad tag + semantic specificity (running), dual-fire with perfect-cover tag + semantic framing (biopics), dual-fire with very-broad over-pull + semantic specificity (Princess Diana), keyword-abstain when no member passes superset without stretching (chess), and whole-call no-fire on thematic essence (grief).
Design context: User policy clarifications captured in this session: (a) when only narrow members are available, abstain on keyword rather than fire-with-bias; (b) cross-family tag borrowing is allowed when the ANY-union genuinely composes the subject; (c) semantic must match the user's specificity rather than over-generalize; (d) score weighting stays equal between keyword and semantic for now — broadness is tolerated because semantic recovers specificity; (e) semantic still fires even when keyword has a perfect-cover tag (the two endpoints layer).
Testing notes: Run handler LLM on "movies about WW2" (expect WAR + WAR_EPIC ANY-union + semantic WW2 body), "biographical films about real singers" (expect cross-category union TRUE_STORY + BIOGRAPHY + MUSIC + MUSIC_DOCUMENTARY + semantic on real singer narratives), "movies about chess" (expect keyword abstain commitment-criteria-fail + semantic-only), "biopics" (expect BIOGRAPHY + semantic still firing with biographical framing). Verify the LLM does NOT generalize semantic ("WW2" → "war stories" is the failure mode to watch for) and does NOT commit narrow-only keyword unions that would bias the score.

## ELEMENT_PRESENCE — same bucket move as CENTRAL_TOPIC; tighten keyword bar to entailment (not correlation) and add concrete-perception gate
Files: schemas/trait_category.py, search_v2/endpoint_fetching/category_handlers/prompts/categories/additional_objective_notes/element_presence.md, search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/element_presence.md
Why: ELEMENT_PRESENCE shared CENTRAL_TOPIC's old "keyword-preferred, semantic-fallback" policy and the same biasing failure mode. It also accumulated mis-routed abstract asks ("twist ending", "anti-hero", "underdog arc") because the existing category-target didn't operationalize "concrete element" as a testable boundary. Two policy gaps unique to this category: (a) the CENTRAL_TOPIC "broader-is-fine" rule does NOT transfer cleanly — for elements, a genre tag that merely correlates with the element (WESTERN for "horses") is stretching, not over-pull, because correlation ≠ presence; (b) the concrete-vs-abstract boundary needs a usable test, not just a list of out-of-scope routes.
Approach: (1) Re-routed ELEMENT_PRESENCE's bucket from `PREFERRED_REPRESENTATION_FALLBACK` to `SEMANTIC_PREFERRED_DETERMINISTIC_SUPPORT`, parallel to CENTRAL_TOPIC. No bucket-prompt or schema changes. (2) Rewrote `additional_objective_notes` with three new emphases: the "see/hear/touch" concrete-perception test as the boundary gate, the entailment-vs-correlation tightening of the keyword superset test (registry tag must NAME the element, not merely co-occur with it — WESTERN ≠ horses, FANTASY ≠ dragons), and the explicit caveat that cross-family borrowing is rare here (since elements live mostly in GENRE/SUB-GENRE). Semantic-always-fires policy and narrow-only-abstain rule carry over from CENTRAL_TOPIC. (3) Rewrote six few-shot examples covering distinct decision paths: direct-tag fire on creature (zombies → ZOMBIE_HORROR), direct-tag fire on activity (heist → HEIST), correlation-not-entailment abstain (horses → WESTERN stretches), narrow-only-entailment abstain (witches → WITCH_HORROR alone has gaps for non-horror witch films), whole-call no-fire on abstract story shape (underdog stories → STORY_THEMATIC_ARCHETYPE), whole-call no-fire on narrative device (unreliable narrator → NARRATIVE_DEVICES). The two no-fire examples teach different boundary mistakes — story shape vs craft device — because these mis-route into ELEMENT_PRESENCE routinely.
Design context: User clarifications layered on top of the CENTRAL_TOPIC session: ELEMENT_PRESENCE must define "concrete" as perceivable (see/hear/touch) so abstract asks route elsewhere, and the keyword bar is HIGHER here than for CENTRAL_TOPIC because correlation between a genre and an element does not entail presence. The semantic plot_events motif syntax (already documented in semantic.md) is the natural channel — semantic always fires with motif text restating only the named element, no fabrication.
Testing notes: Run handler LLM on "zombie movies" (expect ZOMBIE_HORROR + semantic motif body), "heist movies" (expect HEIST + heist motif), "movies with horses" (expect keyword abstain stretching + semantic motif), "movies with witches" (expect keyword abstain narrow-only + semantic motif — verify it does NOT fire WITCH_HORROR alone). The two failure modes to watch for: (a) firing a correlating genre tag (WESTERN for horses, FANTASY for dragons) under the old over-pull-is-fine intuition, and (b) firing a narrow sub-form alone (WITCH_HORROR for witches) when broader entailing coverage doesn't exist.

## ELEMENT_PRESENCE — drop whole-call no-fire examples and reframe boundaries
Files: search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/element_presence.md, search_v2/endpoint_fetching/category_handlers/prompts/categories/additional_objective_notes/element_presence.md
Why: The previous rewrite included two whole-call no-fire examples (underdog stories → STORY_THEMATIC_ARCHETYPE, unreliable narrator → NARRATIVE_DEVICES). Upstream routing sends those asks to their home categories directly — they never reach the ELEMENT_PRESENCE handler. Teaching "whole-call no-fire" as a pathway here misframes what the handler actually sees and wastes prompt budget on a non-case.
Approach: Removed both no-fire examples from the few-shot file (4 examples remain: zombies, heists, horses, witches — direct-tag fire ×2, stretching abstain, narrow-only abstain). Removed the "Whole-call no-fire" section from `additional_objective_notes/element_presence.md` and rephrased the Boundaries header to make explicit that the listed siblings are expectations about routing, not fallback paths the handler should attempt to take.

## CENTRAL_TOPIC + ELEMENT_PRESENCE — revoke ANY-union policy; single-keyword (rare multi) clean-cover or abstain
Files: search_v2/endpoint_fetching/category_handlers/prompts/categories/additional_objective_notes/central_topic.md, search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/central_topic.md, search_v2/endpoint_fetching/category_handlers/prompts/categories/additional_objective_notes/element_presence.md
Why: Previous revision allowed stitching multiple registry members into an ANY-union to manufacture coverage (e.g., `TRUE_STORY + BIOGRAPHY + MUSIC + MUSIC_DOCUMENTARY` for "real singers", `WAR + WAR_EPIC` for WW2). For "movies about death" the handler then assembled a hodgepodge union (DRAMA / TRAGEDY / WAR / PSYCHOLOGICAL_HORROR) where none of the members actually name death as the subject — the union retrieved a heterogeneous bag of films, not death-focused films, and added noise that semantic had to fight against. The mistake is treating "the union together covers it" as if it were as good as "one member cleanly covers it"; in practice the joint commit dilutes the keyword signal whenever the individual members aren't aligned with the subject.
Approach: (1) Rewrote CENTRAL_TOPIC's `additional_objective_notes` to state that keyword fires only when a single registry member is a clean superset (perfect cover or just slightly broader where the subject is a meaningful slice of the tag's scope). Added the "too broad to carry useful signal" failure mode (BIOGRAPHY for "real singers") alongside the existing stretching (SPORT for chess) and narrow-only (WAR_EPIC for WW2) failure modes. Multi-keyword commits are now described as rare and only justified when two members jointly carve the subject without either being on its own a clean superset — when in doubt, abstain. (2) Rewrote the six few-shot examples to match: WW2 → `WAR` alone (drops the WAR_EPIC union); biopics → BIOGRAPHY perfect cover; running → SPORT broader-but-aligned; Princess Diana → abstain (BIOGRAPHY too broad — single person is a tiny slice); chess → abstain (stretching); death → abstain (no aligned tag, do not stitch). Dropped the "real singers" cross-category union example entirely (it was the canonical multi-keyword stitching case). Dropped the "grief" whole-call no-fire example for the same reason ELEMENT_PRESENCE dropped its no-fire examples — upstream routing handles those. (3) Updated ELEMENT_PRESENCE's `additional_objective_notes` to remove the "find a broader entailing member to ANY-union with" language, adding an explicit "do not stitch a broader-but-non-entailing genre onto a narrow entailing tag" rule (e.g., do not add DARK_FANTASY / SUPERNATURAL_FANTASY to WITCH_HORROR to plug witch-comedy gaps — that reintroduces stretching). Existing ELEMENT_PRESENCE few-shots already used single-keyword commits; no changes needed there.
Design context: User policy clarifications captured in this session: (a) revoke the ANY-union-to-cover-the-topic rule from the prior revision; (b) keyword fires only when a single (or rarely multiple) registry member perfectly or just slightly-broadly covers the subject; (c) if no single member is a tight fit, abstain on keyword and rely on semantic; (d) provide at least one new example showing "keywords insufficient → skipped" — death added to CENTRAL_TOPIC alongside the existing chess abstain. The motivating failure was "movies about death" where the handler stitched a noisy union instead of abstaining.
Testing notes: Run handler LLM on "movies about WW2" (expect `WAR` alone, NOT `WAR + WAR_EPIC` union), "movies about death" (expect keyword abstain commitment-criteria-fail, semantic-only), "about Princess Diana" (expect abstain — verify it no longer fires BIOGRAPHY under the too-broad rule), "movies about real singers" (expect abstain — was previously a 4-way union, should now be semantic-only), "biopics" (BIOGRAPHY perfect-cover commit still applies). Failure modes to watch for: (a) regression to multi-keyword stitching, (b) BIOGRAPHY firing on specific-person subjects, (c) any narrow-only commit.

## Step 3 fidelity discipline + awards few-shot ladder
Files: search_v2/step_3.py, search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/awards.md
Why: Diagnostic run of "award winning acting" surfaced three compounding bugs in the award path. (1) Step 3's `target_population` silently widened "winning" to "won or nominated" — drift seeded at the trait-role-analysis layer and carried through aspects, dimensions, and expressions. (2) Step 3's dimension expression embedded a parenthetical example list ("won or nominated for acting awards (Oscar, BAFTA, etc.)") that the downstream endpoint LLM read as filter values rather than illustrations, pinning the query to Oscar+BAFTA and missing every other tracked ceremony. (3) The endpoint LLM over-decomposed into four per-ceremony searches when one unfiltered-ceremony search would have covered all twelve tracked ceremonies for free — the existing few-shot bank only showed narrow-prize-or-ceremony cases, leaving the category-only / no-ceremony shape unmodeled.
Approach: Three targeted prompt edits, no schema or code changes.
  (a) `_TRAIT_ROLE_ANALYSIS`: added a FIDELITY DISCIPLINE section enumerating three drift modes (implicit broadening, implicit narrowing, invented detail) and a fidelity read-back test that requires every constraint in `target_population` to trace back to words in the trait and vice versa. Generalized guidance — no awards-specific examples. Added matching NEVER entry ("BROADEN, NARROW, OR INVENT against the trait's stated detail"). Placed the section between the source-priority list and IDENTITY-VS-ATTRIBUTE so all earlier reading feeds into it. Catches drift at the root layer rather than papering over it downstream.
  (b) `_DIMENSION_INVENTORY` COMMON PITFALLS: added EXAMPLE-IN-EXPRESSION rule. Expressions describe the database check in category-vocabulary; specific entities appear only when the trait named them, never as model-supplied examples of "what the category typically contains". Spelled out the downstream consequence (endpoint LLM reads parentheticals as filter values) so the rule isn't just stylistic.
  (c) Awards few-shot bank: rewrote from 4 to 12 examples spanning the full specificity range. New cases added: category-only with no ceremony filter (the bug case — explicit instruction not to per-ceremony decompose), prize+category combo, multi-search ANY (Oscar OR BAFTA), multi-search AVERAGE (acting AND directing), explicit count (won 3 Oscars), recognition with null outcome (Cannes), mid-rollup discipline (Best Actor OR Best Actress → `lead-acting`, not enumerate descendants), Razzie explicit-name. Each example shows concrete parameter values (`category_tags`, `ceremonies`, `award_names`, `outcome`, `scoring`) rather than prose only, matching the structured-output shape the LLM must emit.
Design context: Builds on existing Step 3 discipline that prompt drift compounds downstream — trait-role-analysis output is the contract for everything below. No new convention added; this is a regression fix against existing discipline.
Testing notes: Re-run "award winning acting" — expect Step 3 `target_population` to preserve "winning" (no nominee expansion), expressions to be category-only with no parenthetical entity examples, and the awards endpoint to emit ONE search with `category_tags=["acting"]`, `outcome="winner"`, no `ceremonies` filter, no `award_names` filter, `floor mark 1`. Other regression checks: "Oscar or BAFTA winner" still produces two searches with `combine: any`; "Oscar-winning Best Director" still emits one search with `award_names=["Oscar"]` and `category_tags=["director"]`; "Best Actor or Best Actress winner" emits `category_tags=["lead-acting"]` (mid-rollup, not enumerated leaves). Failure modes to watch for: (a) regression where target_population re-introduces nominee broadening, (b) Step 3 reverting to parenthetical exemplars in expressions, (c) endpoint LLM ignoring the new few-shot and falling back to per-ceremony decomposition.

## Fidelity discipline at every layer the user's truth conditions pass through
Files: search_v2/step_2.py, search_v2/step_3.py
Why: The previous Step-3 + few-shot fix only partially closed the drift. A re-test of "award winning acting" surfaced the actual origin: Step 2's `evaluative_intent` was itself broadening — emitting "performances that have received formal accolades or high critical acclaim for acting quality" when the user said only "award winning acting." Step 3 read the already-drifted evaluative_intent as authoritative GROUNDING, faithfully decomposed it into TWO aspects (`formal acting awards and nominations` + `critical praise for performances`), and committed TWO categories under FACETS combine_mode (PRODUCT-folds — actively requiring both award recognition AND critical praise). The Step-3 fidelity check we added couldn't catch this because evaluative_intent looked self-consistent with target_population; the drift had already happened upstream. A second leak: even with clean expressions, `retrieval_intent` strings still embedded named ceremonies ("like the Academy Awards, BAFTAs, or Golden Globes") because the EXAMPLE-IN-EXPRESSION rule only fired on `Dimension.expression` in `_DIMENSION_INVENTORY`, not on the `retrieval_intent` field committed at `_CATEGORY_ROUTING`. The endpoint LLM reads retrieval_intent verbatim and treats named entities as filter values, reproducing the per-ceremony pinning the dimension-layer fix was supposed to eliminate.
Approach: Three layered fidelity rules, all generalized — no domain examples in any of them.
  (a) `_EVALUATIVE_INTENT` (Step 2): added FIDELITY DISCIPLINE section between the "light inference is permitted" license and the operational test. Defines three drift modes (implicit broadening, implicit narrowing, invented detail) and a fidelity read-back test requiring every clause of evaluative_intent to trace back to a word in surface_text or an entry in modifying_signals. Added matching Hard-guardrails bullet. Highest-leverage fix because everything downstream reads evaluative_intent as the per-criterion contract; cleaning it here cascades through Step 3.
  (b) `_ASPECT_ENUMERATION` (Step 3): added TRUTH-CONDITION PRESERVATION subsection after the existing GROUNDING traceability rule, and a matching NEVER entry. Sharpens GROUNDING with width preservation — tracing back to target_population is not enough; the aspect must preserve the constraint at the same granularity. Same three drift modes (implicit broadening / narrowing / invented additional condition) applied to the aspect-enumeration boundary. Defense-in-depth in case Step 2 fidelity still slips.
  (c) `_CATEGORY_ROUTING` (Step 3): extended the EXAMPLE-IN-EXPRESSION rule that previously only covered `Dimension.expression` so it explicitly covers `retrieval_intent` too. Added a "retrieval_intent FIDELITY" procedural note stating that both expressions and retrieval_intent are read verbatim by the endpoint LLM and that named entities therefore appear only when the trait itself named them, never as model-supplied exemplars. Calls out "such as X, Y, or Z" formulations alongside parentheticals.
Design context: Per the user's directive, Step 3's existing fidelity check stays anchored on contextualized_phrase + evaluative_intent as-is — the bet is that Step 2 cleanup cascades down. Fix (a) is the load-bearing change; (b) and (c) are defense-in-depth at the layers most prone to re-introducing drift. The decision NOT to tighten FACETS commit (proposed option #5 in the prior diagnosis) was explicitly declined by the user — if Step 2 produces a clean single-criterion evaluative_intent, the spurious second category at Step 3 dissolves on its own without needing combine-mode rules to suppress it.
Testing notes: Re-run "award winning acting" — expect Step 2 evaluative_intent to NOT contain "or critical acclaim" (only the award-winning clause); Step 3 to produce a single aspect ("formal acting awards" — no "nominations") and a single category call (AWARDS — no SPECIFIC_PRAISE_CRITICISM); retrieval_intent on the award call to omit ceremony examples; endpoint to emit one minimal-filter search (`category_tags=["acting"]`, `outcome="winner"`, no ceremony filter, floor 1). Regression checks: "Oscar-winning Best Director" still emits clean prize+leaf-category search; "Oscar or BAFTA winner" still produces two searches with `combine: any` (these inputs explicitly name the entities, so retrieval_intent referencing them stays legitimate). Failure modes to watch for: (a) Step 2 re-introducing "or"-clauses to broaden a narrow constraint, (b) Step 3 aspect layer re-bundling related-but-not-stated conditions, (c) retrieval_intent reverting to enumerative example lists.

## CHARACTER_ARCHETYPE — binary one-or-other firing (drop Split clause)
Files: search_v2/endpoint_fetching/category_handlers/prompts/categories/additional_objective_notes/character_archetype.md
Why: Previous Coverage Decision listed three states (Preferred keyword, Fallback semantic, Split for uncovered subtype/qualifier). Split contradicts the category's nature: the archetype itself *is* the target, with no within-class refinement for semantic to add (unlike CENTRAL_TOPIC, where BIOGRAPHY tags the class and semantic carries the specific subject within it). Any qualifier sophisticated enough to seem to justify a split commit actually belongs to a sibling category (tragic past → STORY_THEMATIC_ARCHETYPE, modern → NARRATIVE_SETTING), so once routing strips the qualifier the residual is clean-binary.
Approach: Rewrote the notes around an Endpoint policy section stating exactly one endpoint fires. Keyword fires alone when a registry member directly defines the archetype; semantic fires alone on narrative_techniques otherwise. Explicit prohibition on adjacent/broader tag reaches. Added the "qualifiers route elsewhere" framing so the LLM can recognize when a request that looks like split-territory is actually a routing concern. Standardized Boundaries header to match CENTRAL_TOPIC / ELEMENT_PRESENCE style ("expectations about routing, not fallback paths"). No few-shot changes — existing 4 examples (anti-hero keyword-only, lovable rogue / manic pixie dream girl semantic-only, redemption arc no-fire/out-of-category) already match the binary rule.
Testing notes: Run handler LLM on "anti-hero protagonist" (expect keyword-only if ANTI_HERO tag exists, semantic-only otherwise — never both), "manic pixie dream girl" (expect semantic-only, no adjacent-tag commit), "femme fatale with a tragic past" (expect archetype handler to address only the archetype — verify it doesn't try to layer semantic for the qualifier). Failure modes to watch for: (a) regression to layered both-fire commits, (b) reaching for adjacent broad character tags when exact archetype isn't tagged.

## Semantic body authoring — strip schema example menus, scope synonym rule, add grounding rule, tighten CHARACTER_ARCHETYPE body discipline
Files: schemas/semantic_bodies.py, search_v2/endpoint_fetching/category_handlers/prompts/endpoints/semantic.md, search_v2/endpoint_fetching/category_handlers/prompts/categories/additional_objective_notes/character_archetype.md

### Intent
Diagnose and close the root cause of a body-authoring drift: searching "love to hate villain" populated `narrative_techniques.audience_character_perception` with three terms — "love to hate antagonist", "morally gray lead", "misunderstood outsider" — exactly three of the five examples listed in that field's schema description. The LLM was treating the field description's example list as a vocabulary menu and recruiting adjacent-but-different archetypes. Same EXAMPLE-IN-EXPRESSION failure mode previously fixed at Steps 2 and 3, now recurring at the semantic body-authoring layer.

### Key Decisions
- **Strip example lists from schema field descriptions entirely** (not just narrative_techniques). User directive: schema descriptions become definitional, not enumerative, across every Body in schemas/semantic_bodies.py — AnchorBody (no descriptions; no change), PlotEventsBody, PlotAnalysisBody, ViewerExperienceBody (per-field + class-level DIRECTION A/B/C example terms), WatchContextBody, NarrativeTechniquesBody (per-field + class-level canonical-label examples), ProductionBody, ReceptionBody. Replaced concrete term lists with rules describing what the field captures, what register to use, and when to leave empty. Structural pedagogy (motif-syntax shape for plot_events, terms+negations same-direction reinforcement for viewer_experience, cross-field repetition for plot_analysis) preserved as abstract rules.
- **Scope synonym/near-duplicate rule to specific spaces in semantic.md.** Previously phrasing rule #3 told the LLM to emit redundant near-duplicates "in term-list spaces" listed as viewer_experience / watch_context / narrative_techniques / reception. Per user directive, synonym density now applies ONLY to viewer_experience, watch_context, and plot_analysis (via cross-field repetition). narrative_techniques, production, and reception term lists do NOT use synonym density — each term names a distinct technique/location/aspect, so emitting "synonyms" means emitting different items the user did not name. This directly fixes the love-to-hate-villain failure where the LLM applied the synonym amplifier to a space where it does not belong.
- **Add a grounding-discipline section to semantic.md** at the top of "Body authoring". States that every term, phrase, or sentence emitted must trace back to the call's `semantic_intent` (multi-endpoint buckets) or `retrieval_intent` (single-endpoint buckets). Calls out the common failure mode (one specific ask → 2-3 adjacent items pulled from memory) explicitly. Less prescriptive than a per-term traceback test; the goal is steering, not a checklist. Mirrors the fidelity discipline already at Steps 2 and 3.
- **Tighten CHARACTER_ARCHETYPE notes with a body-authoring section.** Added between "Endpoint policy" and "Boundaries". States the one-trait-one-term rule for `audience_character_perception` and contrasts five common archetypes definitionally (love-to-hate antagonist / morally gray lead / misunderstood outsider / sympathetic monster / lovable rogue) to make the not-synonyms point concrete. Re-states the no-synonym-padding rule in-context and applies the substitution test in the narrowest form ("would the user accept this as the same thing they said?"). The contrasts function as anti-recruitment because they explicitly label the archetypes as distinct rather than as a candidate list.

### Planning Context
Diagnosis surfaced four amplifiers — user accepted the schema strip, scoped synonyms (not narrative_techniques), endorsed the grounding rule in a less-extreme form referencing `semantic_intent` / `retrieval_intent`, and added the CHARACTER_ARCHETYPE notes tighten. Explicit user note: "Not a problem if we don't add synonyms in the narrative_techniques space" — meaning the substitution test as a guard is unnecessary once the upstream synonym rule is correctly scoped. So the fix is at the synonym-rule scope layer, not at adding more downstream checks.

### Testing Notes
- Re-run "love to hate villain" — expect `audience_character_perception.terms = ["love-to-hate antagonist"]` (or the user's exact phrasing in canonical register), NOT a 3-way menu pull.
- Test other archetype phrasings ("manic pixie dream girl", "lovable rogue", "femme fatale") — expect one precise term per call.
- Test viewer_experience requests ("tearjerker", "edge of your seat") — expect synonym density preserved (the rule still applies in those spaces).
- Test narrative_techniques requests for techniques other than archetypes ("told in reverse chronological order", "found-footage style") — expect single canonical term, no synonym padding.
- Failure modes to watch: (a) regression to recruiting adjacent items from memory once schema examples are gone, (b) over-correction where viewer_experience bodies drop legitimate synonym density, (c) narrative_techniques bodies adding "near-duplicates" that are actually different techniques.

## Faithful evaluative_intent + structured upstream restatement before inference
Files: schemas/step_2.py, search_v2/step_2.py, schemas/step_3.py, search_v2/step_3.py
Why: Two prior rounds of prompt-only fidelity rules failed to stop the drift on "award winning acting". Diagnosis (web research + prompt audit) surfaced three compounding causes: (1) direct contradictions in our own prompts — Step 2 told the LLM that "a near-paraphrase of surface_text means you've underused signals" while also telling it not to elaborate beyond signals, and Step 3's aspect layer framed multi-faceted traits as "reliably encoding three or more axes" while also requiring width preservation; (2) lost-in-the-middle effect (Step 3 prompt at ~73K chars meant fidelity rules added mid-prompt were systematically under-weighted); (3) "models are pattern-followers more than rule-followers" — abstract fidelity rules lose to training-data priors on culturally loaded phrases unless they are pinned to the model's own generation stream via structured output. User selected two surgical fixes: redefine `evaluative_intent`'s purpose to be a faithful restatement (rejecting the elaboration license), and add a schema-forced upstream restatement field at the top of Step 3's output so the verbatim trait text appears in the model's own output stream before any inference field begins.
Approach: Generalized guidance throughout — no domain-specific examples in any new prompt text (explicit user directive).
  (a) `Atom.evaluative_intent` description in schemas/step_2.py: rewrote to define the field's purpose as faithful restatement bounded by surface_text + modifying_signals. Each modifying_signal effect maps mechanically to integration language (HARDENS → required-strength; SOFTENS → preference-strength; FLIPS POLARITY → avoid-direction; freeform → restate the effect text). Replaced the old "PARAPHRASE SURFACE_TEXT WHILE IGNORING SIGNALS" NEVER bullet (which implied paraphrase = smell) with "BROADEN, NARROW, OR INVENT against surface_text + modifying_signals" — and explicitly states that for concrete queries with empty modifying_signals, near-paraphrase IS the correct shape.
  (b) `_EVALUATIVE_INTENT` prompt in search_v2/step_2.py: softened the "ONE place where light inference is permitted" framing to "Inference is bounded to integrating signal effects". Replaced the "if it's just a rephrase, you've underused signals" Hard-guardrails bullet with its inverse: near-paraphrase is correct when no signals exist; the only smell is a rephrase that IGNORES signals that DO exist. Appended a PATTERNS section at the end of the prompt — three abstract patterns (concrete narrow query no signals / concrete query with HARDENS-SOFTENS modifier / cross-criterion reference) showing what faithful evaluative_intent looks like as abstract shapes. No domain-specific examples used; the patterns are framed in terms of the prompt's own vocabulary (surface_text, modifying_signals, HARDENS/SOFTENS/FLIPS POLARITY effects).
  (c) `TraitDecomposition.trait_restatement` in schemas/step_3.py: added as the FIRST field (declaration order = generation order in structured output), before `target_population`. Required content: verbatim reproduction of the trait's contextualized_phrase (in double quotes), evaluative_intent (in double quotes), relationship_role (single quotes), and any non-empty axis bookkeeping. No paraphrasing, no summarization, no commentary. The description explicitly names every downstream field that must trace to content present in the restatement.
  (d) `_TRAIT_RESTATEMENT` prompt section in search_v2/step_3.py: new top-level section walking the model through the restatement step. Placed FIRST in SYSTEM_PROMPT assembly (immediately after `_TASK_FRAMING`, before `_TRAIT_ROLE_ANALYSIS`) so it lands at the start of the prompt — mitigates lost-in-the-middle for the most important rule. Explains the auto-regressive rationale (whatever the model generates first conditions everything below) so the model treats this as load-bearing rather than ceremonial. Includes NEVER bullets specifically against paraphrasing the quoted strings, summarizing, adding commentary, and omitting fields that exist.
  (e) `target_population` description (schemas/step_3.py) + `_TRAIT_ROLE_ANALYSIS` FIDELITY TEST (search_v2/step_3.py): both rewritten to point at the quoted strings inside trait_restatement as the authoritative in-context record, instead of pointing at contextualized_phrase / evaluative_intent abstractly. The fidelity check now reads "trace each clause back to the words inside the quotes in trait_restatement" — anchored to content the model itself just generated, not to an upstream source it has to recall through the prompt.
  (f) `trait_role_analysis` description (schemas/step_3.py): added a "Source: trait_restatement and target_population" framing and a NEVER bullet against introducing content not in those upstream anchors. The role analysis is now explicitly a tightening of what the trait restatement names, not an enlargement of scope.
  (g) `_TASK_FRAMING` in search_v2/step_3.py: added a "0. TRAIT RESTATEMENT" layer at the top of the output-layer enumeration so the workflow framing matches the new schema shape. ANALYSIS PHASE description now starts with "produce the trait restatement" before population/role.
Design context: User explicitly declined two related fixes (add `intent_anchor` to Step 2 atoms; remove/move intent_exploration). Per their direction we tested the prompt-only Step 2 fix + the schema-forced Step 3 restatement together first. Rationale for skipping the Step 2 anchor field: evaluative_intent's redefinition is a smaller, lower-risk change to start with; if drift recurs at the Step 2 layer specifically, the atom-level anchor can be added in a follow-up. The contradicting "intent_exploration encourages elaboration at the top of every Step 2 run" remains untouched per user direction; bet is the field-description tightening on `evaluative_intent` is enough.
Testing notes: Re-run "award winning acting" — expect Step 2 evaluative_intent to be a near-paraphrase of "award winning acting" with no "or" clauses introducing "accolades" / "critical acclaim" / nomination alternatives; Step 3 to emit a `trait_restatement` containing the verbatim contextualized_phrase and evaluative_intent in quotes BEFORE target_population; target_population, trait_role_analysis, and aspects to trace every clause back to words inside those quotes. Regression checks: schema additions are backward-incompatible for any cached LLM outputs that lack `trait_restatement` (none in the codebase; verified no direct TraitDecomposition() instantiations outside the schema file). Token cost: ~80-200 extra output tokens per Step 3 call for the restatement field. Failure modes to watch: (a) Step 2 reverting to evaluative_intent elaboration despite the redefinition, (b) Step 3 model writing a paraphrased rather than verbatim restatement (defeats the auto-regressive anchor — quoted-string check is the test), (c) downstream fields ignoring the restatement and re-introducing drift (in which case the per-field "trace to restatement" rules need further sharpening).

## Awards FLOOR-vs-THRESHOLD self-evaluation + Step 3 verbatim aspect→expression
Files: search_v2/endpoint_fetching/category_handlers/prompts/endpoints/awards.md, search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/awards.md, schemas/step_3.py, search_v2/step_3.py
Why: Two related polish passes on the search pipeline.
  (1) Awards endpoint scoring miscalibration: broad-category asks ("award-winning acting", "won acting awards") were getting `floor mark 1` because the prior calibration table treated category-only-no-prize as a "specific filter" — but those asks are broad generalizations where a movie with 5 acting-award wins genuinely fits better than a movie with 1. The previous shape collapsed every match above the floor to the same score, losing the gradient the user is reaching for when they don't pin a specific ceremony/prize.
  (2) Step 3 dimension layer was paraphrasing aspect strings into "translated" expressions ("highly praised for performances" → "praised for performances"; "won acting awards" + "acting nominations" → merged "acting award wins and nominations"). The aspects step is where fidelity discipline lives, but the dimensions step had license to rewrite — so even when aspects were clean, expressions drifted before flowing into category_calls and the endpoint handler. The "translation into database-vocabulary" framing at the dimension layer was the explicit license that drove this.
Approach:
  (a) awards.md `Per-search scoring` section: added an explicit self-evaluation question at the top — "Would a movie with MORE matching award rows fit this search better than a movie with fewer?" YES → THRESHOLD, NO → FLOOR. The question is framed as the central decision and gets a direct answer for each canonical input shape: broad generalizations / disciplines / intensity language ⇒ THRESHOLD; specific ceremony / prize / category / explicit count ⇒ FLOOR. Reworked the floor/threshold rule lists to align with that framing. Added a "broad-category ask: threshold mark 3" calibration entry. The "Oscar-winning is not generic" guidance now explicitly anchors on the self-evaluation question (more Oscars don't make a movie "more of an Oscar winner" — the specific prize means floor 1).
  (b) awards few-shot bank: "award-winning acting" example flipped from `floor mark 1` to `threshold mark 3`, with an inline self-check explanation showing how the question lands. "Won acting and directing awards" (multi-search AVERAGE) flipped both searches from `floor mark 1` to `threshold mark 3` with the same rationale. "Oscar winner" added a contrasting inline self-check ("would 5 Oscars fit Oscar winner better than 1? NO") so the floor side of the rule is also pattern-visible. Other examples (BAFTA nominated, Oscar-winning Best Director, Oscar OR BAFTA winner, won 3 Oscars, Cannes recognition, Best Actor OR Best Actress winner, Razzie-nominated) already match the new rule and were left unchanged.
  (c) `Dimension.expression` field description (schemas/step_3.py): rewrote to require character-for-character verbatim copy of one of the aspect strings. No rewriting, no tightening, no rewording, no merging. The dimension layer becomes a routing slot (aspect + plausible categories), not a translation layer. Added explicit NEVER bullets for rewriting and merging. Documented the multi-routing case (same aspect string can appear in multiple dimensions when it routes to different category candidates).
  (d) `_DIMENSION_INVENTORY` prompt body (search_v2/step_3.py): rewrote the section header and core rule. The new framing is "emit one dimension per aspect routed, with expression as the aspect verbatim." Replaced the "translate every aspect into a database-vocabulary check" language with a "VERBATIM EXPRESSION — load-bearing rule of this layer" block. Added REWRITTEN ASPECT and PRE-MERGED ASPECT pitfalls to COMMON PITFALLS, framed as the most common failure modes. Updated OPERATIONAL TESTS to require character-for-character diff equality between expressions and the aspects list. Removed the CATEGORY-AWARE PHRASING section (there is no phrasing — only routing). Note that EXAMPLE-IN-EXPRESSION pitfall is now caught earlier: if an expression has parentheticals, either the aspect itself has them (upstream fidelity fix) or the expression has been rewritten (forbidden by the verbatim rule). Either path resolves upstream.
  (e) Consistency updates: `aspects` field description (schemas/step_3.py) — replaced "Translation into database-vocabulary … happens in the dimensions step" with "Database-side routing … happens in the dimensions step. Each dimension below carries this aspect's string VERBATIM in its expression field — the dimensions step routes; it does not re-author." The TEST framing flipped to "could each entry stand on its own as a clean expression string the routing step would copy verbatim?" — making aspects load-bearing in a way that motivates getting them right upstream. _ASPECT_ENUMERATION prompt section and _TASK_FRAMING (search_v2/step_3.py) updated with matching language so the workflow narrative is coherent end-to-end.
Design context: Both changes target the same failure surface — drift between what the user wrote and what the endpoint LLM receives. (a)/(b) close the FLOOR-bias on broad-category asks that was making "award winning acting" score-degrade after the per-search count exceeded 1. (c)/(d)/(e) close the channel through which Step 3 was reshaping clean aspects into paraphrased expressions before they reached the handler. The aspects layer is already protected by the existing FIDELITY DISCIPLINE / TRUTH-CONDITION PRESERVATION rules; making the dimension layer a verbatim pass-through means those upstream rules now actually govern what flows downstream.
Testing notes:
  - Re-run "award winning acting" through the full pipeline. Expect: Step 2 evaluative_intent near-paraphrase of "award winning acting"; Step 3 trait_restatement reproducing the upstream verbatim; aspects ≈ ["award winning acting"] or close; dimensions emitting `expression="award winning acting"` (verbatim, no rephrase); awards endpoint emitting ONE search with `category_tags=["acting"]`, `outcome="winner"`, no ceremony filter, `scoring: threshold, mark 3`.
  - Regression checks: "Oscar winner" still `floor mark 1` (specific prize triggers the NO answer to the self-question); "won 3 Oscars" still `floor mark 3` (explicit count); "heavily decorated" still `threshold mark 5` (qualitative plenty calibration unchanged); "Best Actor or Best Actress winner" still `floor mark 1` (specific categories).
  - Verbatim-rule check: pick any Step 3 output and diff each `dimensions[i].expression` against the `aspects` list — every expression must appear character-for-character in aspects.
  - Failure modes to watch: (a) endpoint LLM defaulting to floor 1 for broad-category asks despite the self-eval question (means the few-shot pattern hasn't taken — would need more pattern coverage), (b) Step 3 model "almost-verbatim" rewriting (e.g., trimming a word) — diff equality is the test, (c) aspects layer drifting so the verbatim copy downstream is itself wrong (means the upstream FIDELITY rules need re-tightening, not the verbatim rule).

## Shared vague-temporal vocabulary across Steps 2 / 3 / metadata handler
Files: search_v2/vague_temporal_vocabulary.py, search_v2/step_2.py, search_v2/step_3.py, search_v2/endpoint_fetching/metadata_query_generation.py
Why: Vague time / duration terms ("modern", "classic", "short", "long", etc.) were being silently narrowed at upstream stages — e.g., Step 2 evaluative_intent paraphrasing "modern" into "the last few years," or Step 3 routing "classic" as a pure quality / reception signal rather than a release-era one. Later stages can narrow a window but not widen it, so the upstream view has to stay generous and the routing has to land on the right category.
Approach: Single source of truth in search_v2/vague_temporal_vocabulary.py exporting two views — VAGUE_TEMPORAL_VOCABULARY_COMPACT (plain-English mappings, no parameter formats) and VAGUE_RELEASE_DATE_DETAILED / VAGUE_RUNTIME_DETAILED (mappings in the YYYY-MM-DD and minute-count formats the metadata endpoint emits). Compact view injected into the Step 2 prompt right after _EVALUATIVE_INTENT (the section where vague-term paraphrasing happens) and into the Step 3 prompt after _PER_DIMENSION_CANDIDATES (where routing decisions get committed). Detailed views interpolated into the metadata handler's release_date and runtime sub-object translation rules, replacing the prior "best-judgment" placeholder lines. "Classic" specifically called out at both layers as primarily a release-era signal with a mild popularity undertone — not a pure quality / reception filter — so it doesn't get split out into a reception atom or routed to quality categories.
Design context: Two views in one file means the upstream prose-level guidance and the downstream parameter-level guidance cannot drift. The compact view deliberately omits formats (no dates, no minute counts) so it doesn't bias upstream stages toward early commitment to values that should be the metadata handler's job. Mirrors the existing pattern of importing _TRACKED_STREAMING_SERVICE_NAMES into the metadata translation rules.
Testing notes: Verify Step 2 evaluative_intent for "modern action movies" yields broad era language (post-2000) rather than narrow "last few years"; Step 3 routing of "classic" lands on RELEASE_DATE category (with possible POPULARITY secondary) and not RECEPTION; metadata handler output for "modern" gives between 2000-01-01 and today rather than a 5-year window; metadata handler output for "classic" gives 1930-01-01 to 1979-12-31. Regression check: any concrete anchor in the query ("post-2010", "under 90 min") still wins over the vague default per the explicit override rule in both views.

## Genre category enforces strict KW/SEM mutual exclusivity
Files: search_v2/endpoint_fetching/category_handlers/prompts/categories/additional_objective_notes/genre.md, search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/genre.md
Why: The bucket-level prompt (PREFERRED_REPRESENTATION_FALLBACK) defaults to "overlap is the design" and authorizes both endpoints to fire when one fills the other's weakness. For Genre under MAX combine this is wrong: a clean KW hit already scores at ~1.0, so SEM cannot add recall and only injects scores from genre-adjacent films. The few-shot examples also led with a "split fire on quiet drama" case that would never reach this category intact (Step 2 splits "quiet" → EMOTIONAL_EXPERIENTIAL).
Approach: No mutex bucket exists today; authoring one would touch enum/schema_factories/two new prompt files for a shape the existing PREFERRED_REPRESENTATION_FALLBACK already structurally supports (each endpoint independently commit/abstain in coverage_commitments). Enforced strict mutex at the category-specific layer instead: rewrote additional_objective_notes/genre.md with a "this category overrides the bucket default" header and a definitional-equivalence test for the KW-vs-SEM decision; added a SEM-body authoring carve-out that overrides the semantic endpoint's synonym-density default for plot_analysis.genre_signatures (one term, user's verbatim phrase, no other sub-fields, no other spaces). Rewrote few_shot_examples/genre.md with seven examples (scary movies → KW, spaghetti westerns → KW, dark comedy → KW, neo-noir → SEM, elevated horror → SEM, cosmic horror → SEM, revenge stories → double-abstain) that all reflect calls surviving Step 2 splitting as atomic genre traits.
Design context: Decision walked in conversation — registry only lives at Step 3 (not Step 2), so the KW-vs-SEM call belongs at the handler layer with the registry in front of it. MAX combine semantics analyzed against concrete query examples; concluded firing both adds no value when tag is canonical (KW dominates the MAX) and is forbidden when not (no exact match → no KW). See search_improvement_planning/query_categories.md Cat 22 — the planning doc already names this category "Mutually exclusive"; the implementation had drifted to overlap-encouraged.
Testing notes: Run the seven few-shot queries through Step 3 in isolation; expect each to commit exactly the endpoint named. Regression watch: queries like "scary slasher movies" should still produce a single KW commit (`[SLASHER_HORROR]` since slasher is the operative concept and scary is redundant). Watch for the model rationalizing "elevated horror" → HORROR as "close enough" — that's the failure mode the definitional-equivalence test exists to catch. Sibling category CULTURAL_TRADITION uses the same (KW, METADATA) shape with the same ALTERNATIVES combine and is also marked mutually-exclusive in the planning doc but was not touched in this change; if behavior there is similarly drifting, the same category-level enforcement pattern applies.

## SOLO combine_mode — single-category traits commit explicitly, extras trimmed before retrieval
Files: schemas/enums.py, schemas/step_3.py, search_v2/step_3.py, search_v2/stage_4_execution.py, search_v2/full_pipeline_orchestrator.py, search_v2/endpoint_fetching/category_handlers/prompt_builder.py

### Intent
Step 3 was routing single-category-coverage traits to multi-category FRAMINGS commits ("good for children" → Target audience + Maturity rating, even though Target audience had `what_this_misses="Nothing"` on every dimension). The FRAMINGS-vs-FACETS binary had no expression for "one category is enough," so the LLM rationalized partial-coverage adjacents as "robust framings." Added a third value to TraitCombineMode and a hierarchical decision rule that asks the coverage question before the relationship question.

### Key Decisions
- **New enum value `SOLO`** at the head of `TraitCombineMode` (schemas/enums.py). Semantics: exactly one surviving category cleanly covers every dimension; stage-4 has nothing to fold (passthrough). Hierarchical decision: SOLO question first, FRAMINGS-vs-FACETS only when no single category covers everything cleanly. Replaces the old "single-dimensional traits default to FRAMINGS because MAX-of-one collapses to passthrough" carve-out — SOLO is now the explicit commit for any single-coverage case, single-dim or multi-dim.
- **Routing exploration reordered**: dedup → granularity → COVERAGE (SOLO short-circuit) → RELATIONSHIP & MINIMUM SET (only when no SOLO candidate). The coverage check uses the per-dimension `what_this_misses` analysis as source of truth — a candidate whose what_this_misses named no substantive gap across every dimension it could serve provides complete coverage and earns SOLO. Other candidates were adjacency context, not coverage gaps another category fills.
- **Prompt guidance generalized, not failure-specific** (user directive: "Ensure any guidance we give is generalized not just inserting examples of the exact failure mode we observed"). The new prompt text frames the rule structurally — "a candidate whose what_this_misses names a real gap does not heal the gap by being paired with another partial candidate" — without naming the specific Target-audience / Maturity-rating case.
- **Orchestrator-side SOLO trim** in `full_pipeline_orchestrator._decompose_and_generate`: if combine_mode is SOLO and the LLM emitted >1 calls, keep `all_calls[:1]` (first listed) and drop the rest BEFORE the per-CategoryCall handler fan-out. Dropped categories never reach handler-LLM endpoint generation or endpoint fetches. List ordering is the LLM's commit surface — trust the first entry as the intended primary. The schema-side prompt instructs the LLM to list the clean-fit primary first when it commits SOLO.
- **Stage-4 SOLO branch in `combine_categories`** returns the single score (passthrough). Defensive log if >1 scores arrive — that would mean the orchestrator's trim missed something. Empty scores still return 0.0 like the other modes.
- **prompt_builder `_COMBINE_MODE_LABELS`** gets a defensive `TraitCombineMode.SOLO: "single"` entry. In practice the empty-siblings branch fires for SOLO traits post-trim and emits `combine_mode="single"` regardless, but the dict mapping prevents a KeyError if SOLO is ever passed with siblings (shouldn't happen but defensive).

### Planning Context
Conversation diagnosed three reinforcing causes for the failure mode: (1) FRAMINGS/FACETS binary has no "one-category-suffices" branch, so the model treats it as a real choice and FRAMINGS naturally invites overlapping commits; (2) the old routing exploration order put the relationship question (3) before minimum-set (4), so by the time the model audited for padding, FRAMINGS had been mentally committed and "adds new signal" got rationalized; (3) no explicit fully-clean-primary gate — the prompt asked about overlap and compounding but never directly asked whether one candidate covered everything cleanly. SOLO addresses all three by adding the missing branch, reordering checks so coverage gates relationship, and making the gate explicit in both schema and prompt.

### Testing Notes
- Re-run "good for children" — expect Step 3 to emit `combine_mode="solo"` with a single `category_calls` entry (Target audience). Routing_exploration prose should explicitly invoke the coverage check before reaching the relationship question.
- Regression checks: queries that genuinely need FRAMINGS (a brand identity with multiple plausible homes) or FACETS (a compound concept with distinct compounding axes) should still commit those modes — the SOLO short-circuit only fires when a single category's coverage analysis names no real gap across every dimension. Watch for over-application of SOLO (every trait collapsed to one call) — that would suggest the LLM is treating the SOLO check as the default rather than the question.
- Verify the trim path: a trait that commits SOLO with multiple category_calls (LLM-generation noise) should produce an INFO log naming the kept and dropped categories, and the dropped categories should not appear in any handler-LLM call or endpoint fetch downstream.
- Stage-4 SOLO arithmetic: `combine_categories(SOLO, [x])` returns x; `combine_categories(SOLO, [x, y])` returns x with a warning log; `combine_categories(SOLO, [])` returns 0.0.
- Failure modes to watch: (a) LLM defaulting to SOLO when traits genuinely need FRAMINGS or FACETS (over-application); (b) LLM still emitting FRAMINGS with partial-coverage extras when SOLO is correct (the new coverage check failing to fire); (c) ordering surprises if the LLM lists a non-primary category first under SOLO (the trim would keep the wrong one — mitigated by the schema instruction to list the primary first, but not enforced beyond prompt discipline).

## Sensitive content category — multi-vector semantic guidance + positive-presence rule
Files: search_v2/endpoint_fetching/category_handlers/prompts/categories/additional_objective_notes/sensitive_content.md, search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/sensitive_content.md
Why: This category routes more "avoid X" / "no X" / "without X" phrasings than any other, which is the highest-risk surface for the LLM to invert the semantic body. The category prompt also pinned semantic to viewer_experience.disturbance_profile only — but plot_events (event-shaped content like nude scenes / animal death / torture, in motif shape) and reception (craft-execution evaluations like "gratuitous violence" / "tasteful restraint") genuinely carry sensitive-content signal, and single-space coverage was the under-recall failure mode. Verified across schemas/trait_category.py, schema_factories.py, and semantic_translation.py that no schema constraint pinned the space — the restriction was doc-only.
Approach: (1) Rewrote `## Endpoint Fit` Semantic line to enumerate three spaces with conditions for each, calling out the reception register constraint (craft execution only, not subject matter — "gratuitous gore" qualifies, "torture porn" does not). (2) Added `## Positive-presence and negation direction` section mirroring viewer_experience.md treatment, explicitly contrasting `"hot"` → `"not cold"` (correct, same retrieval target) vs `"hot"` → `"not hot"` (contradictory). (3) Rewrote the "not too bloody" few-shot example to show the actual body construction inline (terms+negations both clustering on the non-gory side). (4) Added two new examples: an exclusion ask ("avoid graphic torture") that demonstrates the body still searches affirmatively across all three spaces with polarity inverted upstream, and a multi-space inclusion ask ("famous for over-the-top, gratuitous gore") that demonstrates plot_events motif + viewer_experience + reception.criticized_qualities firing together with positive polarity. Kept examples 1, 2, 4, 5 unchanged — they already calibrate Keyword and Metadata correctly.
Design context: Per-space register rules and motif shape per search_v2/endpoint_fetching/category_handlers/prompts/endpoints/semantic.md (sections "Per-space register table", "Negations (viewer_experience only)", "Plot events — motifs and specific events", "Positive-presence invariant"). Reception register constraint per schemas/semantic_bodies.py:753-772.
Testing notes: No automated tests cover prompt text. Verify qualitatively via test_v3_endpoints.ipynb on representative queries: exclusion ("no gore" → affirmative body + polarity negative upstream), within-trait boundary ("not too bloody" → affirmative-complement body), multi-space inclusion ("famous for gore" → all three spaces fire), specific event ("no animal deaths" → keyword + plot_events motif). Spot-check that no contradictory pairings appear (terms=["gory"] with negations=["not too gory"]).


## Genre category Case A/B triage — add pseudo-genre recovery branch
Files: search_v2/endpoint_fetching/category_handlers/prompts/categories/additional_objective_notes/genre.md, search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/genre.md
Why: Step 3 occasionally routes phrases that describe a film cluster rather than name a genre identity (observed: "movies for comic book lovers" producing a Genre call as "comic book visual style"). The prior handler hard-coded one semantic shape — paste verbatim into plot_analysis.genre_signatures, abstain everywhere else — which embeds a non-genre phrase against ingest text written for genre labels and retrieves noise. Upstream routing can't be fixed here, so the handler needs a recovery branch.
Approach: Added a Case A vs Case B triage at the top of additional_objective_notes/genre.md. Case A (established genre identity — registry-clean or canonical sub-genre) keeps the existing strict mutex unchanged: exactly one of keyword/semantic fires, semantic body limited to plot_analysis.genre_signatures verbatim. Case B (pseudo-genre description — film cluster without a registry or sub-genre name) always fires semantic and never fires keyword; the semantic body authors across whichever vector spaces honestly cover the cluster using standard endpoints/semantic.md per-space rules. Eligible Case B spaces: viewer_experience, production, watch_context, narrative_techniques (tier 1) and reception (tier 2 — acclaim framings only). Explicitly excluded in Case B: plot_analysis (the whole point of the split) and plot_events (too literal, fabricates plot detail). Discriminator framed as a principle ("does the modifier restrict the genre slice, or name an orthogonal axis?") with default-to-A-on-ambiguity bias.
Approach (examples): few_shot_examples/genre.md kept all six existing Case A examples under a new "Case A" heading and removed the prior "revenge stories" double-abstain example (per the handler-tries-its-best directive — no Case C anymore). Added three Case B examples chosen to collectively touch every eligible space without overlapping shape: comic book visual style (production + viewer_experience), scratches the slasher itch (watch_context + viewer_experience), celebrated noir voiceover storytelling (reception + narrative_techniques). Three examples is the count — fewer leaves shape gaps, more risks template-priming per the principle-based prompts preference.
Design context: Triage decision walked across multiple turns with explicit user direction: (1) Case A keeps strict mutex; (2) Case B always semantic, never keyword; (3) Case C abandoned (the handler must always produce something); (4) examples must vary in shape and collectively cover all proposed-relevant spaces. Bucket framing unchanged — GENRE remains PREFERRED_REPRESENTATION_FALLBACK with CategoryCombineType.ALTERNATIVES.
Testing notes: Run test_v3_endpoints.ipynb against (a) Case A regression set ("scary movies", "neo-noir", "elevated horror", "cosmic horror") expecting unchanged endpoint commits and semantic bodies; (b) Case B activation set including "movies for comic book lovers" — confirm semantic commits on production + viewer_experience, keyword abstains, plot_analysis.genre_signatures stays empty; (c) Case B variety — at least one viewer-seeking phrasing and one acclaim phrasing to confirm watch_context, narrative_techniques, and reception are reachable in actual handler output. Failure modes to watch: (1) LLM reclassifying Case A restrictive modifiers ("elevated horror", "weird western") as Case B and abandoning the verbatim genre_signatures rule — the bias-toward-A guidance is the mitigation; (2) Case B semantic body drifting back into plot_analysis.genre_signatures — the explicit prohibition is the load-bearing rule; (3) tier-1 spaces sprawling (all four fire on every Case B call) — per-space discipline in endpoints/semantic.md is the mitigation. Empty-fire trap already covered by _is_vacuous_spec in output_extractor.py.

## Sensitive content category — single-vector decision rule + tighter keyword precision (amendment)
Files: search_v2/endpoint_fetching/category_handlers/prompts/categories/additional_objective_notes/sensitive_content.md, search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/sensitive_content.md
Why: The prior pass made all three semantic spaces available but did not constrain selection — review surfaced that multi-space firing muddies the retrieval target for this category (event-presence asks pulled atmospherically-disturbing films without the named event; reception-framed asks redundantly fired plot_events and viewer_experience). Also: keyword was over-firing on genre adjacency (horror flag from the word "torture"), elevating whole genres on precise content asks.
Approach: Rewrote `## Endpoint Fit` Semantic line as a single-vector decision rule with three "fire when…" clauses walked in order (reception → plot_events → viewer_experience) plus explicit tie-breakers ("Brutal torture scenes" → plot_events wins; "Famous for graphic torture" → reception wins; pure intensity word with no event class or evaluation framing → viewer_experience). Tightened Keyword: fires ONLY when a registry definition names exactly the axis; genre adjacency ("horror" from "torture") is explicitly prohibited; if no flag directly covers, keyword stays silent and semantic carries the whole ask. Rewrote examples 6 (avoid graphic torture → plot_events motif only, with explicit anti-pattern note about not activating a horror flag) and 7 (famous for over-the-top gratuitous gore → reception.criticized_qualities only, with explicit note that the "famous for + gratuitous" framing routes to reception even though gore is namable as an event).
Testing notes: Re-verify via test_v3_endpoints.ipynb on the same representative queries; the expected fire pattern now changes: "no graphic torture" → keyword silent (no torture flag in registry) + plot_events motif ONLY; "famous for over-the-top gratuitous gore" → reception.criticized_qualities ONLY; "not too bloody" unchanged (viewer_experience.disturbance_profile only). Watch for regression where the LLM still fans out to multiple spaces or activates a genre keyword as a content proxy.
