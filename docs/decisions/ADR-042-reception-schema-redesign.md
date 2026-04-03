# [042] — Reception Schema Redesign: Dual-Zone Structure with Observation Fields

## Status
Active

## Context

ADR-025 defined `ReceptionOutput` with a single `review_insights_brief`
field — a dense 150-250 token paragraph capturing thematic, emotional,
structural, and source-material observations for Wave 2 consumption.
During reception generator evaluation across 36 movies in 6 input-richness
buckets, two problems with the monolithic field became clear:

1. **Mixed-dimension signal hurt cheap-model output quality.** Wave 2
   generators consume different dimensions of reception signal: viewer
   experience needs emotional register, narrative techniques needs craft
   observations, source_of_inspiration needs source material hints.
   Forcing all dimensions into one paragraph required each Wave 2 generator
   to parse out its relevant signal from mixed text.

2. **Prompt guidance quality.** The original schema and prompt had structural
   issues hurting cheap-model output: flat hierarchy mixing global/per-field
   rules, contradictory style guidance, tag guidance duplicated across two
   fields, and critical rules buried at the end.

## Decision

Decompose `review_insights_brief` into 4 targeted observation fields plus
rename/restructure the synthesis zone fields.

**Extraction zone (Wave 2 signal, not embedded):**
- `thematic_observations` — nullable, compact phrases (1-4)
- `emotional_observations` — nullable, compact phrases (1-4)
- `craft_observations` — nullable, compact phrases (1-4)
- `source_material_hint` — nullable short phrase (adaptation/biopic hint)

Fields ordered extraction-before-synthesis so cheap models extract concrete
observations before synthesizing opinions. Grouped by downstream consumer
affinity, not 1:1 with generators.

**Synthesis zone (embedded):**
- `reception_summary` (renamed from `new_reception_summary`)
- `praised_qualities` 0-6 tags (renamed from `praise_attributes`)
- `criticized_qualities` 0-6 tags (renamed from `complaint_attributes`)

**Backward-compat `review_insights_brief` @property** concatenates the 3
observation fields. Pre-consolidation skip conditions and Wave 2 prompt
builders that access `.review_insights_brief` continue to work unchanged
until they are updated.

**Eligibility threshold raised 25→400 chars** combined review text.
Evaluation showed movies below 400 chars produce observations that merely
paraphrase the overview rather than adding genuine review signal. ~779
movies affected (0.7% of corpus); those movies retain `reception_summary`
and quality tags for Wave 2 fallback.

**Review truncation and selection changes:**
- `_MAX_REVIEW_CHARS` 5K→6K (saturation analysis showed output quality
  stops improving at ~2,000 input tokens / ~750 review tokens; 6K provides
  ~2x headroom). Previously raised to 10K during an intermediate evaluation
  pass; reverted to 6K after cost analysis.
- `_MAX_REVIEW_COUNT` removed entirely — character budget alone prevents
  prompt bloat; count cap was unnecessarily restricting perspective diversity.
- Ascending-length sort in `_truncate_reviews` — shorter reviews packed
  first to maximize perspective diversity before the char budget is hit.

## Alternatives Considered

1. **Keep `review_insights_brief` with better prompt guidance**: Improved
   prompts would still require Wave 2 generators to parse one mixed paragraph.
   Structured fields make the dimension split explicit and auditable.

2. **4 fields instead of 3 observation fields**: Source material is a
   sufficiently distinct dimension (adaptation/biopic signal) to warrant its
   own nullable field rather than being embedded in thematic_observations.

3. **Update Wave 2 consumers in the same changeset**: Deferred intentionally
   — the backward-compat property bridges the gap, and Wave 2 consumer updates
   should be validated against the new schema after the generation pipeline
   produces real outputs.

4. **Symmetric tag recall pressure** (equally penalize over- and
   under-tagging): Rejected by user as unnecessary complexity. Tag count
   should reflect breadth of distinct qualities reviewers mentioned.

## Consequences

- Wave 2 prompt builders and skip conditions continue to work via the
  `review_insights_brief` @property until explicitly updated.
- Unit tests referencing the old `review_insights_brief` field directly as
  a schema field will fail; they need updating to the @property or new fields.
- The `_MAX_REVIEW_COUNT` parameter is removed; callers that pass it will
  fail. Identified in unit test notes.
- Downstream embedding (`vectorize.py`) and search-side `ReceptionMetadata`
  schema are NOT updated in this change — deferred to follow-up.

## References

- ADR-025 (schema design) — original `review_insights_brief` as scalar
- ADR-036 (schema field description minimalism)
- ADR-043 (reception model selection) — evaluated with this schema
- `movie_ingestion/metadata_generation/schemas.py`
- `movie_ingestion/metadata_generation/generators/reception.py`
- `movie_ingestion/metadata_generation/prompts/reception.py`
- `movie_ingestion/metadata_generation/batch_generation/pre_consolidation.py`
