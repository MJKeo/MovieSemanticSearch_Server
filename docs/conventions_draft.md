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
**Observed:** (1) During reception generator revamp, extraction-first ordering improved cheap-model quality. (2) During plot_analysis redesign, reordering from cold-start-synthesis to classify→analyze→distill→ground→synthesize dramatically improved the generation sequence for gpt-5-mini. Each field scaffolds the next: genre_signatures establishes frame, thematic_concepts provides analytical foundation, core_concept distills from established themes, character_arcs ground in the thematic framework, overview synthesizes everything.
**Proposed convention:** In structured output schemas, order fields so each one scaffolds the next in a natural cognitive progression. For dual-zone schemas (extraction + synthesis), extraction comes first. For single-zone analytical schemas, order from easiest/most-concrete to hardest/most-abstract. Never put a synthesis field (core concept, overview) first — it forces cold-start distillation without prior context. Applies to all generation-side schemas in movie_ingestion/metadata_generation/schemas.py.
**Sessions observed:** 2

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

## Three-tier examples (Good/Shallow/Bad) in structured output prompts
**Observed:** During reception prompt revision, two-tier examples (Good/Bad) were insufficient to prevent GPT-5-mini from producing "shallow" output that was technically not bad (not evaluative, not vague) but missed the mark (topic-listing instead of argument-capturing). Adding a middle "Shallow" tier that shows exactly what the model currently produces — and labels it as insufficient — gave a much clearer signal of what "good" actually requires.
**Proposed convention:** When a structured output prompt has quality examples, use three tiers (Good/Shallow/Bad) rather than two (Good/Bad) when the observed failure mode is "acceptable but insufficient" rather than "clearly wrong." The Shallow tier shows the specific pattern to improve upon.
**Sessions observed:** 1
