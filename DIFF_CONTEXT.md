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
