# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

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
