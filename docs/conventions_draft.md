# Conventions Draft

Observed patterns staged for review. Remove entries you disagree
with, then run /solidify-draft-conventions to merge the rest into
docs/conventions.md.

Entries are added automatically during /safe-clear based on
patterns observed in the session.

## Use StrEnum for domain constant sets, not bare strings
**Observed:** User corrected hardcoded `GENERATION_TYPE = "plot_events"` strings across 8 generator files, saying "We really shouldn't be hardcoding strings like 'plot_events' anywhere." Created `MetadataType(StrEnum)` as the canonical source, with all callers required to use enum members.
**Proposed convention:** When a set of related string constants is used across multiple modules (metadata types, pipeline stages, etc.), define them as a `StrEnum` in a shared location. Callers must reference the enum — never hardcode the string value. `StrEnum` keeps SQLite/JSON compatibility while preventing typos and enabling IDE support.
**Sessions observed:** 1

## Cognitive-scaffolding field ordering in structured output schemas
**Observed:** (1) During reception generator revamp, extraction-first ordering improved cheap-model quality. (2) During plot_analysis redesign, reordering from cold-start-synthesis to classify→analyze→distill→ground→synthesize dramatically improved the generation sequence for gpt-5-mini. Each field scaffolds the next: genre_signatures establishes frame, thematic_concepts provides analytical foundation, core_concept distills from established themes, character_arcs ground in the thematic framework, overview synthesizes everything. (3) During plot_analysis hardening, justification/reasoning fields placed BEFORE their corresponding labels to serve as chain-of-thought scaffolding during autoregressive generation.
**Proposed convention:** In structured output schemas, order fields so each one scaffolds the next in a natural cognitive progression. For dual-zone schemas (extraction + synthesis), extraction comes first. For single-zone analytical schemas, order from easiest/most-concrete to hardest/most-abstract. Never put a synthesis field (core concept, overview) first — it forces cold-start distillation without prior context. In justification/evaluation variants, place reasoning/explanation fields immediately before the label they scaffold. Applies to all generation-side schemas in movie_ingestion/metadata_generation/schemas.py.
**Sessions observed:** 3

## Verify provider-specific API parameter support before setting kwargs
**Observed:** Multiple invalid parameters were discovered in playground candidates during research: `reasoning_format: "hidden"` is not supported for Groq gpt-oss models (should be `include_reasoning`), `thinking_config` is not supported on Gemini Flash Lite, and temperature is not supported on gpt-5 family models. These had been set without verification and would have caused silent failures or errors at runtime.
**Proposed convention:** When configuring LLM provider kwargs in playground candidates or generator defaults, verify each parameter against the provider's actual API documentation. Parameters that work for one provider/model often don't exist for another. Include a brief comment on non-obvious constraints (e.g., "temperature only supported when reasoning_effort='none'").
**Sessions observed:** 1

## Minimize redundant API/DB calls in polling loops
**Observed:** In the autopilot loop, Claude's initial implementation called `_get_active_batch_ids()` 3 times and `check_batch_status()` redundantly per iteration. The user pushed for reusing snapshots from earlier steps and eliminating unnecessary re-queries, pointing out that stale-by-seconds data is fine for slot counting.
**Proposed convention:** In polling loops that interleave status checks with state mutations, query external APIs and DB once per logical phase. Reuse the snapshot for subsequent steps in the same iteration. If a step mutates state (e.g., clearing batch IDs), re-query once after the mutation — not per-consumer of the data. Accept that data may be seconds stale; the next iteration corrects it.
**Sessions observed:** 1

## Parameter names should match the caller's domain model, not the implementation's
**Observed:** Claude proposed `max_movies` as a parameter on `build_requests()` to limit work. User corrected to `max_batches` — "Then in this method just multiply max_batches with batch_size." The caller thinks in batches, so the parameter should be `max_batches` with the conversion happening internally.
**Proposed convention:** Function parameters exposed to callers should use the caller's domain vocabulary. Internal conversions (e.g., `max_movies = max_batches * batch_size`) belong inside the function, not at every call site. The parameter name should match what the caller is already thinking about.
**Sessions observed:** 1

## Merge ambiguous field boundaries when targeting small LLMs
**Observed:** During plot_analysis redesign, `themes_primary` and `lessons_learned` were merged into `thematic_concepts` because small LLMs (gpt-5-mini) waste cognitive budget distinguishing themes from lessons — a distinction irrelevant to vector search (embedding model doesn't know which field a label came from). The merge eliminated a common failure mode (near-duplicate labels across both fields) with zero impact on retrieval quality.
**Proposed convention:** When two structured output fields have ambiguous boundaries (the LLM frequently produces overlapping content) and the downstream consumer treats them identically (e.g., both become embedding text), merge them into a single field. This is especially important for small/cheap models where cognitive budget is limited. Evaluate by asking: "Does the embedding model or search pipeline distinguish these fields?" If no, merge.
**Sessions observed:** 1

## Remove non-embedded fields from production structured output schemas
**Observed:** During plot_analysis hardening, `character_name` and `arc_transformation_description` were removed from the production `CharacterArc` schema because neither was included in `__str__()` (embedding text). Their presence added schema complexity that mini models struggle with (conditional null logic, overlapping description/label fields) for zero search quality benefit. A separate `CharacterArcWithReasoning` schema retains a chain-of-thought field for the evaluation variant only.
**Proposed convention:** Production structured output schemas (used for batch generation at scale) should only contain fields that appear in `__str__()` (embedded) or are strictly necessary for downstream processing. Fields that exist solely to aid generation quality (reasoning, justifications, descriptions) should live in separate evaluation-variant schemas. This reduces schema complexity for cheap models and eliminates conditional-null instructions that are the highest-error instruction type for mini models.
**Sessions observed:** 1

## "FIRST: determine whether..." abstention gates for optional structured output fields
**Observed:** During plot_analysis prompt hardening, character_arcs and conflict_type instructions were restructured to lead with an explicit abstention decision ("FIRST: determine whether the movie has identifiable characters who undergo meaningful transformations") before any extraction instructions. This was driven by evaluation data showing the model almost never produced empty lists despite min_length=0 — the "empty list when..." instruction buried at the end was treated as an afterthought. Leading with the abstention gate + listing concrete content types where emptiness is expected (documentaries, concerts, shorts, anthology) dramatically changes the model's default behavior from "always produce something" to "decide first, then extract."
**Proposed convention:** When a structured output field allows empty output (min_length=0 or optional), and evaluation shows the model rarely exercises that option, restructure the field instruction to lead with the abstention decision before the extraction rules. List concrete categories where empty output is expected. The abstention gate should be the first thing the model reads for that field, not a footnote.
**Sessions observed:** 1

## Single source of truth for shared logic between eligibility and generation
**Observed:** (1) Viewer experience had duplicated narrative resolution (`_resolve_narrative_input` in generator, `_resolve_viewer_experience_narrative_input` in pre_consolidation) and duplicated observation filtering (`_filter_observations` with mirrored threshold constants in generator, `_viewer_experience_observation_lengths` in pre_consolidation). The user flagged both: "Why do we have these when we already defined thresholds in pre_consolidation?" and "Is _filter_observations() duplicating logic from _viewer_experience_observation_lengths()?" Both were unified into single public functions in pre_consolidation. (2) During narrative_techniques redesign, the same pattern was applied from the start: `resolve_narrative_techniques_narrative()` and `_filter_craft_observations()` were designed as shared functions used by both eligibility and prompt building.
**Proposed convention:** When eligibility checking (pre_consolidation) and prompt building (generators) apply the same logic (threshold filtering, input resolution, fallback ladders), that logic must live in exactly one place — a public function in pre_consolidation.py. Generators import and call it. Never mirror threshold constants or filtering logic between the two layers, even with a "mirrors the values in X" comment.
**Sessions observed:** 2

## Three-tier examples (Good/Shallow/Bad) in structured output prompts
**Observed:** During reception prompt revision, two-tier examples (Good/Bad) were insufficient to prevent GPT-5-mini from producing "shallow" output that was technically not bad (not evaluative, not vague) but missed the mark (topic-listing instead of argument-capturing). Adding a middle "Shallow" tier that shows exactly what the model currently produces — and labels it as insufficient — gave a much clearer signal of what "good" actually requires.
**Proposed convention:** When a structured output prompt has quality examples, use three tiers (Good/Shallow/Bad) rather than two (Good/Bad) when the observed failure mode is "acceptable but insufficient" rather than "clearly wrong." The Shallow tier shows the specific pattern to improve upon.
**Sessions observed:** 1

## Return dataclass objects from multi-field loaders, not positional tuples
**Observed:** During narrative_techniques redesign, a second `load_wave1_outputs_for_narrative_techniques()` was created returning `(plot_summary, craft_observations)` — a near-duplicate of `load_wave1_outputs_for_movie()` returning `(plot_synopsis, thematic_observations)`. The user immediately flagged: "Surely these can be merged into a single method that returns an object with attributes rather than a tuple where we have to remember the ordering." Replaced both with `load_wave1_outputs()` returning a `Wave1Outputs` dataclass. All fields loaded in one query; callers pick by name.
**Proposed convention:** When a function loads multiple optional fields from a data source and different callers need different subsets, return a dataclass (or similar named-field object) rather than a positional tuple. This eliminates tuple-ordering bugs, makes call sites self-documenting (`w1.craft_observations` vs `result[1]`), and allows adding fields without breaking callers. Fetch all fields at once — the marginal DB/parse cost is negligible vs the maintenance cost of per-caller loader functions.
**Sessions observed:** 1
