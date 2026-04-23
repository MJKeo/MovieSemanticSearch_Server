# [073] — Award category tag taxonomy replacing exact-string category matching

## Status
Active

## Context
The Stage 3 award LLM previously emitted exact IMDB surface forms for
award categories ("Best Performance by an Actor in a Motion Picture -
Drama" for Globes vs "Best Actor in a Leading Role" for Oscars). A
single character mismatch produced zero matches. The model could not
express broad concepts ("any acting award") without enumerating dozens
of ceremony-specific strings. 766 distinct raw category strings existed
in the corpus.

## Decision
Introduce `CategoryTag` — a single `(str, Enum)` with 81 members
across three levels:
- 62 leaves (ids 1..99): specific categories (lead-actor, animated-short…)
- 12 mid rollups (ids 100..199): only where they earn their keep
  (lead-acting, supporting-acting, screenplay-any, best-picture-any,
  animated, documentary, short, sound-any, music, visual-craft,
  worst-acting, worst-craft)
- 7 top groups (ids 10000..10006): acting, directing, screenplay,
  picture, craft, festival-section, razzie

Per row, `movie_awards.category_tag_ids INT[]` stores the leaf PLUS
every ancestor id. Querying `&& ARRAY[10000]` (acting group) catches
every leaf and mid under it in one GIN scan. The 100^level id scheme
(`1..99 / 100..199 / 10000+`) keeps ids globally unique with room for
a 4th level at 1,000,000+.

`tags_for_category(raw_text) -> list[int]` wraps the existing
`consolidate()` and ancestor lookup; called from `batch_upsert_movie_awards`.
The 766 raw strings collapse into 62 leaves at 100% coverage.
`render_taxonomy_for_prompt()` generates the LLM-facing taxonomy
section from the enum, eliminating schema/prompt drift.

## Alternatives Considered
- **Three separate per-level Pydantic fields**: Keeps levels explicit but
  makes the JSON schema a 3-way union and forces the LLM to decide level
  before picking value. Single combined enum allows the model to pick at
  any specificity.
- **Mid rollups for all branches**: Branches with no useful intermediate
  concept (director, foreign-film, casting) would add spurious mid entries.
  Mid rollups are defined only where they earn their keep.
- **Separate leaf and ancestor tables in DB**: Normalized but requires a
  JOIN per query. Storing ancestors inline in the array is the same
  tradeoff used for `concept_tag_ids` and `source_material_type_ids`.

## Consequences
- The LLM picks from 81 members at any granularity; "any acting award"
  and "Best Actor Oscar" are both expressible without enumeration.
- Existing Stage 3 award tests construct categories as strings; they
  must be updated to use `CategoryTag` enum members.
- Razzie tag interaction: the default `ceremony_id <> 10` exclusion is
  now overridden when `category_tags` contains any id in
  `RAZZIE_TAG_IDS` (16 ids). Either axis alone is sufficient for opt-in.
- Import-time assertion in `schemas/award_category_tags.py` ensures
  `_TAG_DESCRIPTIONS` covers every `CategoryTag` member — adding a tag
  without a description fails loudly at import.

## References
- schemas/award_category_tags.py
- ADR-068-award-data-postgres-storage.md
- search_improvement_planning/finalized_search_proposal.md §Endpoint 3
