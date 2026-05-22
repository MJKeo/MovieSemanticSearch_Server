# [094] — concept_tags three-run batch pipeline

## Status
Active

## Context
concept_tags generation benefits from sampling diversity — running the
classifier three times and merging via majority vote improves per-tag
precision/recall versus a single run. The eval script had been doing this
interactively (3 calls per movie, `majority_merge`) but the Batch API
pipeline only stored one result per movie. Aligning the batch pipeline to
produce three independent runs requires schema changes (new column), changes
to how pending movies are counted, how results are written (first-NULL
target), and how batch IDs are tracked.

## Decision
Added `concept_tags_run_3 TEXT` column to `generated_metadata`. Batch
pipeline generalized with `runs_per_movie` + `result_columns` on
`GeneratorConfig`: concept_tags sets `runs_per_movie=3` and
`result_columns=["concept_tags", "concept_tags_run_2", "concept_tags_run_3"]`.

Key implementation choices:
- **Custom_id encodes run index for OpenAI uniqueness only** (format
  `{type}_{tmdb_id}_r{N}`). Column target is NOT derived from the run
  index — it is purely "first NULL among result_columns" at write time.
  This decouples submission-side identity from storage layout.
- **Pending-IDs query counts NULL slots per movie** (`SUM(CASE WHEN col IS
  NULL THEN 1 ELSE 0 END)`) so a movie with 2 of 3 slots filled gets
  exactly 1 new request, not 3. Single-run types collapse to the same
  query shape.
- **`metadata_batch_ids.{type}_batch_id` stays a single column.** With all
  3 requests landing in the same submission, the single-column gate works.
  If requests straddle a batch boundary, the later batch_id wins (benign).
- **`COLUMNS_BY_TYPE` duplicated in result_processor.py** (not imported
  from the registry) to keep result_processor free of generator-module
  imports. A smoke test asserts the two maps stay in sync.
- All three concept_tags columns cleared to NULL across 102,443 rows before
  the first 3-run batch submit for a clean slate.

Downstream merge logic extracted to `concept_tags_merge.py` shared module
(`majority_merge`, `LIST_CATEGORIES`, `_ASSESSMENT_BY_FIELD`) and a
`backfill_concept_tag_ids.py` backfill script for updating
`movie_card.concept_tag_ids` from the merged results.

## Alternatives Considered
- **Single run, larger model**: rejected — the accuracy ceiling for a single
  run is lower than majority vote, and the cost of 3 runs on a cheaper model
  is comparable to 1 run on a larger model.
- **Store merged result instead of raw runs**: rejected — storing raw runs
  preserves per-run reasoning for diagnosis and allows re-merge if the merge
  logic changes without regeneration.

## Consequences
- concept_tags batch runs cost 3x the previous single-run cost (~$525-575
  for 86K movies × 3 runs at the compressed prompt + gpt-5-mini-minimal config).
- `ingest_movie.py` still reads only `concept_tags`; the majority-merge
  step is applied separately via the backfill script.
- Unit tests for `parse_custom_id` (2-tuple return) need updating to handle
  the new 3-tuple `(MetadataType, tmdb_id, run_index | None)` return.

## References
- `movie_ingestion/tracker.py` (new column)
- `movie_ingestion/metadata_generation/batch_generation/generator_registry.py`
- `movie_ingestion/metadata_generation/batch_generation/request_builder.py`
- `movie_ingestion/metadata_generation/batch_generation/result_processor.py`
- `movie_ingestion/metadata_generation/concept_tags_merge.py`
- `movie_ingestion/backfill/backfill_concept_tag_ids.py`
