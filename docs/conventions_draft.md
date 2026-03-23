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

## Extraction-first field ordering in structured output schemas
**Observed:** During the reception generator revamp, field ordering in the Pydantic schema was set extraction-first (observation fields before evaluative/synthesis fields) because OpenAI structured output generates fields in schema order. Placing concrete extraction before abstract synthesis improved cheap-model output quality at low reasoning effort.
**Proposed convention:** When a structured output schema serves both extraction (from source data) and synthesis (evaluative/summary content), order extraction fields first. This creates a natural cognitive flow where the model anchors on concrete observations before producing synthesized judgments. Applies to all generation-side schemas in movie_ingestion/metadata_generation/schemas.py.
**Sessions observed:** 1

## Verify provider-specific API parameter support before setting kwargs
**Observed:** Multiple invalid parameters were discovered in playground candidates during research: `reasoning_format: "hidden"` is not supported for Groq gpt-oss models (should be `include_reasoning`), `thinking_config` is not supported on Gemini Flash Lite, and temperature is not supported on gpt-5 family models. These had been set without verification and would have caused silent failures or errors at runtime.
**Proposed convention:** When configuring LLM provider kwargs in playground candidates or generator defaults, verify each parameter against the provider's actual API documentation. Parameters that work for one provider/model often don't exist for another. Include a brief comment on non-obvious constraints (e.g., "temperature only supported when reasoning_effort='none'").
**Sessions observed:** 1

## Three-tier examples (Good/Shallow/Bad) in structured output prompts
**Observed:** During reception prompt revision, two-tier examples (Good/Bad) were insufficient to prevent GPT-5-mini from producing "shallow" output that was technically not bad (not evaluative, not vague) but missed the mark (topic-listing instead of argument-capturing). Adding a middle "Shallow" tier that shows exactly what the model currently produces — and labels it as insufficient — gave a much clearer signal of what "good" actually requires.
**Proposed convention:** When a structured output prompt has quality examples, use three tiers (Good/Shallow/Bad) rather than two (Good/Bad) when the observed failure mode is "acceptable but insufficient" rather than "clearly wrong." The Shallow tier shows the specific pattern to improve upon.
**Sessions observed:** 1
