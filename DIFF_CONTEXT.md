# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

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
