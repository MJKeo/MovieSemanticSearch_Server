# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

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
