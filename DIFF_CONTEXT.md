# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Docs audit: fix stale documentation across 10 files

Files: CLAUDE.md, docs/conventions.md, docs/modules/ingestion.md, docs/modules/llms.md, docs/modules/classes.md, docs/modules/README.md, docs/decisions/ADR-012-llm-generation-cost-optimization.md, implementation/classes/enums.py, movie_ingestion/tracker.py

Why: Full docs-auditor scan found 16 issues — factual errors in CLAUDE.md (wrong IMDB data path, wrong Stage 3/5 descriptions, wrong signal count, wrong filename), stale conventions (missing status step, nonexistent eval.db), broken ADR cross-references (10 deleted ADRs still cited), and minor inaccuracies in module docs.

Changes:
- CLAUDE.md: Fixed IMDB output path (JSON files → imdb_data table), Stage 5 hard filters (old list → 2 gates), Stage 3 description (removed phantom hard filters), tmdb_quality_scorer signal count (10 → 4), batch_manager.py → openai_batch_manager.py
- conventions.md: Added tmdb_quality_calculated/imdb_quality_calculated to status progression; rewrote eval storage section (eval.db → per-movie JSON files)
- ingestion.md: Fixed generator lock claim (all 8 now locked), build_custom_id arg order, parse_custom_id return type, replaced analyze_evaluations.py entry with errors.py, removed references to deleted ADRs (015, 019, 028, 031, 034), removed deleted run_evaluations_pipeline.py and eval.db sections
- llms.md: Removed references to deleted ADRs (024, 029, 030)
- classes.md: Added note that ProductionMetadata is legacy and not aligned with generation-side split schemas
- enums.py: VectorCollectionName docstring ChromaDB → Qdrant
- tracker.py: Added tmdb_quality_calculated to docstring and SQL comment progressions
- README.md: Relaxed 60-line guideline to allow complex modules to exceed
- ADR-012: Moved watch_context from Wave 1 to Wave 2 to match code

## Move ingest_movie.py from db/ to movie_ingestion/final_ingestion/
Files: db/ingest_movie.py → movie_ingestion/final_ingestion/ingest_movie.py, 5 notebooks, unit_tests/test_ingest_movie.py, docs/modules/db.md, docs/modules/ingestion.md, CLAUDE.md
Why: Consolidate all ingestion pipeline logic under movie_ingestion/. The new final_ingestion/ subpackage will house all logic for taking movie data from the SQLite tracker and upserting into Postgres/Qdrant.
Approach: Moved the file, created __init__.py, updated all import paths (5 notebooks, 1 test file), updated db.md/ingestion.md module docs and CLAUDE.md architecture docs.

## Add top-level schemas/ package and refactor cross-cutting types out of metadata_generation
Files: schemas/__init__.py, schemas/metadata.py, schemas/enums.py, schemas/data_types.py, schemas/movie_input.py (all new), movie_ingestion/metadata_generation/schemas.py (deleted), movie_ingestion/metadata_generation/inputs.py (trimmed), all 8 generators, all 4 batch_generation modules, CLAUDE.md

Why: Need a home for shared Pydantic models/data classes importable by db/, api/, and movie_ingestion/. The implementation/ folder is being phased out.

Moved to schemas/:
- All Output schema classes (PlotEventsOutput, ReceptionOutput, etc.) → schemas/metadata.py
- MetadataType enum → schemas/enums.py
- MultiLineList → schemas/data_types.py
- MovieInputData + load_movie_input_data → schemas/movie_input.py

Kept in inputs.py (generation-pipeline-specific):
- build_custom_id (now overloaded: accepts int or MovieInputData), parse_custom_id
- WAVE1_TYPES, WAVE2_TYPES, ALL_GENERATION_TYPES
- ConsolidatedInputs, SkipAssessment, build_user_prompt
- Wave1Outputs, load_wave1_outputs, load_plot_analysis_output

Removed: MovieInputData.batch_id() method — callers use build_custom_id() directly.
All 12 consumer files updated to import from new canonical locations (no re-export shims).

## Extract vector text generation into movie_ingestion/final_ingestion/vector_text.py
Files: movie_ingestion/final_ingestion/vector_text.py (new), movie_ingestion/final_ingestion/ingest_movie.py, CLAUDE.md, docs/modules/ingestion.md, docs/llm_metadata_generation_report.md
Why: Vector text generation functions (8 total, one per vector space) were in implementation/vectorize.py alongside legacy ChromaDB code. Extracting them into final_ingestion/ co-locates them with the ingestion pipeline that consumes them.
Approach: Copied all 8 create_*_vector_text functions and the budget_size_to_vector_text helper verbatim into the new file. Updated ingest_movie.py import to point to the new location (linter normalized to relative import `from .vector_text import ...`). Updated all docs that referenced the old path: ingestion.md, llm_metadata_generation_report.md, CLAUDE.md, PROJECT.md, and TODO.md (6 references).

## Update test imports for schemas package split
Files: unit_tests/test_pre_consolidation.py, unit_tests/test_source_of_inspiration_generator.py, unit_tests/test_viewer_experience_generator.py, unit_tests/test_metadata_schemas.py, unit_tests/test_plot_analysis_generator.py, unit_tests/test_plot_events_generator.py, unit_tests/test_production_keywords_generator.py, unit_tests/test_narrative_techniques_generator.py, unit_tests/test_watch_context_generator.py, unit_tests/test_reception_generator.py, unit_tests/test_generator_registry.py
Why: The generation-side schema models moved from `movie_ingestion.metadata_generation.schemas` to the top-level `schemas.metadata` module, and a set of tests were still importing the deleted module path.
Approach: Repointed test imports to `schemas.metadata` without changing test intent. Re-ran collection in the `uv` environment to verify the pure import-path drift was fixed; those updated files now collect successfully.
Testing notes: `unit_tests/test_prompt_constants.py` still fails collection because `SYSTEM_PROMPT_WITH_REASONING` is missing from `movie_ingestion.metadata_generation.prompts.source_of_inspiration`, which appears to be a real API/constant change rather than a path rename.

## Add EmbeddableOutput base class with embedding_text() method
Files: schemas/metadata.py
Why: Replace the __str__()-based convention for generating embedding text with an explicit embedding_text() method on a shared base class. Makes the embedding contract explicit and applies normalize_string() consistently.
Approach: Created EmbeddableOutput(BaseModel) with abstract embedding_text(). All 8 *Output classes now inherit from it and implement embedding_text(), which assembles the same fields as __str__() but returns normalize_string()-processed text instead of manual .lower(). Existing __str__() methods retained for backward compatibility.

## Update vector text functions to accept Movie object
Files: movie_ingestion/final_ingestion/vector_text.py
Why: Standardize all vector text functions to accept a Movie object and access metadata internally. Updated create_narrative_techniques_vector_text, create_viewer_experience_vector_text, and create_watch_context_vector_text to accept Movie, access the corresponding metadata attribute, and return None when metadata is missing. Removed now-unused NarrativeTechniquesOutput, ViewerExperienceOutput, and WatchContextOutput imports.

## Switch source_of_inspiration prompt-constants test to production prompt
Files: unit_tests/test_prompt_constants.py
Why: The test still imported `SYSTEM_PROMPT_WITH_REASONING`, but the current source_of_inspiration prompt module only exports the production `SYSTEM_PROMPT`.
Approach: Updated the test to import `SYSTEM_PROMPT` only and rewrote the source_of_inspiration assertions to validate the production prompt text instead of the removed reasoning variant.
Testing notes: A follow-up collect-only run exposed a separate production_keywords prompt issue in the same file: `SYSTEM_PROMPT_WITH_JUSTIFICATIONS` is also no longer importable from `movie_ingestion.metadata_generation.prompts.production_keywords`.

## Add tracker-backed Movie schema loader
Files: schemas/movie.py
Why: Need a single object that can load full per-movie tracker data from `tmdb_data`, `imdb_data`, and `generated_metadata` while preserving source column names and exposing parsed metadata outputs.
Approach: Added `TMDBData`, `IMDBData`, and `Movie` Pydantic models plus `Movie.from_tmdb_id()`, which performs one joined SQLite query with aliased columns and then parses IMDB JSON TEXT columns, TMDB review JSON, and TMDB provider-key blobs into typed Python values. Metadata columns are parsed into the current `schemas.metadata` output models, with a narrow compatibility normalization for known legacy key drift (`justification` → `evidence_basis`, and obsolete source-of-inspiration evidence fields) so existing tracker rows still validate against the latest schema classes.
Design context: Follows the new top-level `schemas/` package split and the tracker-backed ingestion architecture documented in `docs/modules/ingestion.md`.
Testing notes: Verified syntax with `uv run python -m py_compile schemas/movie.py` and loaded a real tracker row via `Movie.from_tmdb_id(2)` to confirm source parsing, fallback helpers, and metadata validation.

## Add notebook cell for manual Movie schema inspection
Files: schemas/testing.ipynb
Why: Need a quick interactive way to manually load one tracker-backed `Movie` by `tmdb_id` and inspect its source rows and metadata in grouped sections while iterating on the new schema loader.
Approach: Replaced the empty placeholder notebook with a valid one-cell notebook that imports `Movie`, lets the user set `tmdb_id` manually, calls `Movie.from_tmdb_id()`, and pretty-prints TMDB data, IMDB data, resolved fallback fields, and each metadata object as distinct high-level groups.
Testing notes: Validated the notebook JSON with `python -m json.tool schemas/testing.ipynb`.

## Rewrite create_plot_events_vector_text with synopsis-first fallback hierarchy
Files: movie_ingestion/final_ingestion/vector_text.py, docs/TODO.md
Why: The plot_events vector should embed the richest available plot text. IMDB synopses are human-written and more detailed than LLM-generated summaries, so they should be preferred when available.
Approach: Changed `create_plot_events_vector_text` to accept `Movie` (from `schemas.movie`) instead of `PlotEventsOutput`. Fallback hierarchy: longest scraped synopsis → generated plot_summary via `plot_events_metadata.embedding_text()` → longest plot_summary entry → overview. Added `create_plot_events_vector_text_fallback` as a separate method for when the primary text exceeds the 8,191 token embedding limit — it picks the longer of longest plot_summary vs generated plot_summary, then falls back to overview. Added TODO for wiring the fallback into the embedding pipeline's error handling.
Design context: Aligns with ADR-033 two-branch strategy. The embedding model (text-embedding-3-small) errors on oversize input rather than truncating, so the fallback handles that case explicitly.

## Fix notebook Movie import path
Files: schemas/testing.ipynb
Why: The manual inspection notebook used a relative import (`from .movie import Movie`), which fails in notebook cells because they are not executed as package modules.
Approach: Replaced the relative import with `from schemas.movie import Movie` and swapped the fragile cwd-based path insertion for a `find_project_root()` helper that walks upward to `pyproject.toml` before adding the repo root to `sys.path`.
Testing notes: Re-validated notebook JSON and verified the same path-bootstrap logic successfully imports `Movie` in a `uv run python` shell.

## Restructure plot_events vector text functions: single normalize_string call
Files: movie_ingestion/final_ingestion/vector_text.py
Why: Both `create_plot_events_vector_text` and `create_plot_events_vector_text_fallback` called `normalize_string()` at multiple return sites, risking fallthrough bugs if a new branch was added without normalization.
Approach: Restructured both functions to accumulate text into a single variable using if/elif/else chains, then call `normalize_string()` once at the end. This also means `embedding_text()` results are now normalized, which they previously were not. Both functions now return `str | None` instead of `str` — returning `None` when no text source is available so callers can distinguish "no data" from empty string.

## Redesign plot_analysis vector text: labeled embedding, genre merging, remove plot_keywords
Files: schemas/metadata.py (PlotAnalysisOutput.embedding_text), movie_ingestion/final_ingestion/vector_text.py (create_plot_analysis_vector_text), schemas/testing.ipynb

### Intent
Improve plot_analysis vector embedding quality by: (1) adding semantic labels to short categorical fields for embedding disambiguation, (2) merging TMDB genres into LLM-generated genre_signatures, (3) removing noisy plot_keywords, (4) moving formatting logic into PlotAnalysisOutput.embedding_text().

### Key Decisions
- **Removed plot_keywords:** Community-voted IMDB keywords are noisy for thematic analysis (mix of incidental scenes, meta-cultural tags, and thematic terms). The LLM-generated metadata already distills thematic signal from these as an input. Raw keywords dilute the refined output.
- **Labeled short fields:** Research (Anthropic contextual retrieval study, Google Gemini docs, LlamaIndex defaults) supports adding "field: value" labels to disambiguate short values for embedding models. Prose fields (elevator_pitch, generalized_plot_overview) left unlabeled as self-contextualizing.
- **Formatting in embedding_text(), not vector_text:** User directed moving field formatting into PlotAnalysisOutput.embedding_text() so the metadata class owns its own embedding representation. vector_text function is now a thin wrapper that merges genres and delegates.
- **Genre merging:** TMDB genres appended to genre_signatures (case-insensitive dedup) so they embed as one labeled field rather than a separate unlabeled line.
- **Normalization strategy:** Prose fields lowercased directly. Label fields have each term individually normalize_string()'d. No normalize_string() on the final assembled string.
- **Return type:** Changed to `str | None` (returns None when metadata missing), matching other vector text functions.
- **Parameter type:** Changed from BaseMovie to Movie to access PlotAnalysisOutput directly.

### Testing Notes
Updated schemas/testing.ipynb to print raw metadata fields, raw TMDB genres, and final vector text output for manual verification across 10 test movies.

## Migrate create_reception_vector_text to Movie with formatting improvements
Files: schemas/movie.py, schemas/metadata.py, movie_ingestion/final_ingestion/vector_text.py, schemas/testing.ipynb

### Intent
Migrate create_reception_vector_text from BaseMovie to Movie. Improve embedding text formatting with labels and per-term normalization. Move formatting logic into ReceptionOutput.embedding_text().

### Key Decisions
- **Added reception_score() and reception_tier() to Movie:** Logic identical to BaseMovie but reads from self.imdb_data.imdb_rating/metacritic_rating.
- **Excluded IMDB review_themes:** Substantially redundant with LLM praised/criticized_qualities (the LLM already consumed review_themes as input and synthesized them).
- **Excluded IMDB reception_summary:** Contains mixed non-reception content (themes, influences, production facts) that would dilute the vector space. LLM summary is more focused on the evaluative dimension.
- **No numeric scores in embedding:** User directive — no IMDB/Metacritic numbers in vector text.
- **Labeled fields:** Added lowercase "reception:", "praised:", "criticized:" labels for embedding disambiguation. Consistent with labeled embedding convention from plot_analysis work.
- **Per-term normalization:** Each praised/criticized quality individually normalize_string()'d before joining.
- **Formatting in embedding_text():** Moved praised/criticized formatting into ReceptionOutput.embedding_text(), vector_text function is a thin wrapper that adds tier and delegates.
- **Returns None when no reception_metadata:** Early return None instead of assembling text from tier alone.

### Testing Notes
Updated schemas/testing.ipynb to print raw reception fields and final vector text for manual inspection across 10 test movies.

## Redesign anchor vector text: switch to Movie, remove dilutive content, add identity signals
Files: schemas/movie.py, movie_ingestion/final_ingestion/vector_text.py

### Intent
Redesign create_anchor_vector_text to use Movie instead of BaseMovie, concentrating on semantic identity signals and removing content better handled by specialized vector spaces, lexical search, or metadata filters.

### Key Decisions
- **Removed cast/crew/character names:** Lexical search handles entity matching; person names don't carry semantic signal in embeddings.
- **Removed production companies/filming locations:** Duplicates production vector without adding broad-recall value.
- **Removed reception praise/complaint lists:** Fully duplicated by reception vector; kept reception_summary (prose) and reception_tier (compact label) instead.
- **Removed detailed maturity text and parental guide items:** Maturity rating is a metadata filter. Replaced with maturity_text_short() — prefers IMDB maturity_reasoning prose, falls back to MPA semantic description.
- **Removed markdown section headers:** Noise tokens for the embedding model.
- **Removed plot_keywords from keyword merge:** Plot keywords over-emphasize minor content details; overall_keywords only.
- **Added elevator_pitch:** Most concise identity signal (~6 words).
- **Added IMDB overview as fallback:** Only when plot_analysis_metadata is missing.
- **Added deduplicated_genres():** Merges LLM genre_signatures with IMDB genres via substring dedup.
- **Added thematic_concepts:** From plot_analysis metadata.
- **Added source_material / franchise_lineage:** From source_of_inspiration metadata.
- **Added reception_summary:** Prose evaluative summary from reception metadata.
- **Formatting:** Prose fields use .lower(), categorical term lists use normalize_string() with semantic labels.

### New Movie methods
- `title_with_original()` — "Title (Original Title)" or just "Title"
- `maturity_text_short()` — IMDB reasoning prose or MPA description fallback
- `deduplicated_genres()` — genre_signatures + IMDB genres, substring-deduped

### Testing Notes
Needs manual inspection via testing.ipynb across diverse movies to verify output quality and token budget.

## Make Movie tracker DB default path absolute
Files: schemas/movie.py
Why: `Movie.from_tmdb_id()` defaulted to `Path("ingestion_data/tracker.db")`, which resolves from the process working directory and broke notebook usage when the kernel cwd differed from the repo root.
Approach: Changed `_DEFAULT_TRACKER_DB` to resolve from the file location (`schemas/movie.py` → repo root → `ingestion_data/tracker.db`) so the default tracker path is stable across notebooks, shells, and other entry points.
Testing notes: Verified `schemas/movie.py` compiles and that `_DEFAULT_TRACKER_DB` resolves to the real tracker DB path; `Movie.from_tmdb_id(2)` now succeeds without passing `tracker_db_path`.

## Rewrite create_production_vector_text to use Movie with new helper methods
Files: schemas/movie.py, schemas/metadata.py, movie_ingestion/final_ingestion/vector_text.py, schemas/testing.ipynb, docs/TODO.md

### Intent
Port create_production_vector_text from BaseMovie to Movie. Add deterministic helper methods on Movie for production-related data. Implement "original screenplay" default in SourceOfInspirationOutput. Add binary production medium classification.

### Key Decisions
- **New Movie methods:** `is_animation()` (binary genre check), `production_text(include_filming_locations)` (labeled format, 3-location limit), `languages_text()` (labeled primary + additional), `release_decade_bucket()` (semantic era labels), `budget_bucket_for_era()` + `_interpolated_thresholds()` (era-adjusted budget classification with linear interpolation between decade anchors).
- **`resolved_budget()` returns None for 0:** TMDB reports 0 when budget is unknown; treated as missing.
- **Filming locations excluded for animation:** Voice actors record in studios, not on location — locations are irrelevant noise.
- **Filming locations capped at 3:** Keeps vector text focused; additional locations are typically studio addresses or minor detail.
- **Binary production medium:** `is_animation()` → "animation" or "live action". Finer details (computer animation, stop motion, CGI) are already captured by production_keywords. This avoids duplicating what production_keywords already provides.
- **"original screenplay" default:** Added to `SourceOfInspirationOutput.embedding_text()` when source_material is empty AND franchise_lineage is empty or only contains "first"/"start" terms (`_is_likely_original()` helper). Ensures original films are queryable. Fallback in `create_production_vector_text` for when metadata is None entirely.
- **Production keywords unlabeled:** Intentionally broad grab-bag — a label would be misleading. Placed at end of vector text.
- **Normalization:** `.lower()` used on production_text/languages_text output (not `normalize_string()`). Source_of_inspiration uses per-term `normalize_string()`. Diacritics inconsistency reviewed and accepted — embedding models handle accented characters well, not a retrieval issue.

### Testing Notes
Updated schemas/testing.ipynb to print all production-relevant Movie data fields alongside final vector text for 14 test movies (including animation, CGI hybrid, big budget, small budget, sparse data edge cases). Manual review confirmed all fields correct.
