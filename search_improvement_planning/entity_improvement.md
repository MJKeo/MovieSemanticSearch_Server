# Entity Endpoint Improvement Plan

Agreed change list for the Stage 3 entity endpoint
(`search_v2/stage_3/entity_query_generation.py` +
`search_v2/stage_3/entity_query_execution.py`). Target outcomes:

- Eliminate the silent-zero-match class of failures caused by name-variant
  asymmetry between LLM output and stored credited strings.
- Handle hyphen punctuation variants deterministically at both ingest
  and query time, so no LLM reasoning is required for
  "spider-man" / "spider man" / "spiderman".

## Schema — `schemas/entity_translation.py`

- Remove `character_alternative_names`.
- Add generic `alternative_names: list[str] | None` — populated for
  `person` and `character`, null for `title_pattern`.
- Update the validator and field-ordering comments to match the new
  shape.

## Prompt — `search_v2/stage_3/entity_query_generation.py`

- Lift alias guidance out of the character-specific section and
  reframe it as a **general entity-alias posture** applying to people
  and characters alike.
- Add the **categorical precision rule**: aliases must be strings a
  credits database would actually store (alternate credited names,
  stage names, birth names, canonical character aliases). NOT
  descriptive paraphrases, role types, or facts about the entity.
- Add the **95%+ confidence bar** as a second gate on top of the
  categorical rule.
- Extend examples to cover person aliases (e.g., birth name ↔ stage
  name), not just character aliases.
- Update `_OUTPUT` per-field guidance for the renamed
  `alternative_names` field.

## Execution — `search_v2/stage_3/entity_query_execution.py`

- `_execute_person_specific_role`: resolve
  `lookup_text + alternative_names` → list of `term_id`s via
  `fetch_phrase_term_ids`, pass the full list into
  `_fetch_actor_scores` / `_fetch_binary_role_scores`.
- `_execute_person_broad`: same — resolve all `term_id`s upfront,
  pass the list into both the actor task and each of the binary-role
  tasks.
- `_execute_character`: rename `character_alternative_names`
  reference to `alternative_names`; rest of the flow already handles
  multi-variant correctly.

## Hyphen variant utility — `implementation/misc/helpers.py`

- New helper that takes a normalized string and returns the set of
  hyphen variants: `{with-hyphen, hyphen→space, hyphen→empty}`.
  Strings without hyphens return as a singleton. Used symmetrically
  at ingest and query.

## Ingest — `movie_ingestion/final_ingestion/ingest_movie.py`

- Apply the hyphen-variant helper when building person/character
  dictionary entries so each variant lands in
  `lex.lexical_dictionary` / `lex.character_strings` with its own
  `term_id`.
- Posting-table writes emit one row per variant `term_id` against
  the same `(movie_id, billing_position, cast_size)` tuple, so any
  variant form can resolve to the credit.

## Query-side variant expansion (paired with the above)

- At Stage 3 query execution, wrap `normalize_string(text)` with the
  hyphen-variant helper before `fetch_phrase_term_ids` /
  `fetch_character_strings_exact`, so a user form like `"spiderman"`
  resolves against a credit stored as `"spider-man"` without
  requiring LLM alias emission.
