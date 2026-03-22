# TODO

Tracks actionable items discovered during development sessions.
Items here are things to address when the relevant work begins,
not urgent fixes.

## Include imdb_vote_count in search reranking quality boost
**Context:** The search reranking process should use `imdb_vote_count` as
a signal in its automatic quality and relevance booster. This likely
requires adding `imdb_vote_count` as a column in the ingested movie
database (Postgres) so it's available at reranking time. Movies with
higher IMDB vote counts are generally better-known and more relevant
results, so this signal can help break ties and boost well-established
films in the final ranking.
**When:** When building or refining the reranking/quality-boost stage of
the search pipeline.
**See:** db/vector_scoring.py, db/ingest_movie.py, movie_ingestion/imdb_scraping/models.py


## Verify model IDs for playground notebook providers
**Context:** The metadata generation playground notebook (Cell 2) uses several model IDs that were specified
speculatively and may not be accurate. IDs to verify against each provider's current docs:
- Alibaba/DashScope: `qwen3.5-flash`
- Gemini: `gemini-2.5-flash-lite` (lite variant may have a different canonical ID)
- OpenAI-compatible: `gpt-oss-120b` (internal/routing alias, may not be stable)
- Groq: `meta-llama/llama-4-maverick-17b-128e-instruct` (verify exact string)
**When:** Before running the notebook for real model comparisons.
**See:** movie_ingestion/metadata_generation/metadata_generation_playground.ipynb (Cell 2)


## Add release year next to title in all LLM metadata generation
**Context:** The LLM metadata generation prompts should include the
release year alongside the movie title (e.g., "The Matrix (1999)")
across all 7 metadata types. This gives the LLM better temporal context
when generating plot analysis, viewer experience, reception, and other
metadata — helping it distinguish remakes, place films in their era,
and produce more accurate descriptions.
**Status:** COMPLETE. All 8 generators (plot_events, reception, plot_analysis,
viewer_experience, watch_context, narrative_techniques, production_keywords,
source_of_inspiration) now implemented with "Title (Year)" format.
**See:** movie_ingestion/metadata_generation/generators/plot_events.py,
movie_ingestion/metadata_generation/prompts/plot_events.py,
docs/llm_metadata_generation_new_flow.md


## Iterate on plot_events evaluation rubric after initial run
**Context:** The 4-dimension rubric (groundedness, plot_summary,
character_quality, setting) was systematically improved pre-run based on
a comparison against the generation SYSTEM_PROMPT (9 identified
inconsistencies fixed). After running Phase 0 + Phase 1 on the first
movie(s), manually inspect judge reasoning in the
`plot_events_evaluations` table before running the full 70-movie corpus —
calibration may still need adjustment based on observed judge behavior.
**When:** After first small-scale evaluation run completes.
**See:** movie_ingestion/metadata_generation/evaluations/plot_events.py (JUDGE_SYSTEM_PROMPT)


## Implement request_builder.py for Batch API integration
**Context:** `movie_ingestion/metadata_generation/request_builder.py`
is currently a stub containing only a docstring. It will be needed for
Batch API request assembly during production-scale LLM generation.
**When:** When building the Batch API integration for Stage 6 production runs.
**See:** movie_ingestion/metadata_generation/request_builder.py


## Align search-side PlotAnalysis __str__() with generation-side schema
**Context:** The search-side `CoreConcept.__str__()` in
`implementation/classes/schemas.py` returns
`"{core_concept_label}: {explanation_and_justification}"`, leaking
justification text into the embedding string. The generation-side
`PlotAnalysisOutput.__str__()` correctly returns only the label. When
deploying new generation results to the production search index, the
search-side `CoreConcept.__str__()` must be updated to return only the
label, and existing Qdrant embeddings should be verified or regenerated
to use consistent embedding text.
**When:** When deploying generation pipeline results to the production
search index.
**See:** implementation/classes/schemas.py (CoreConcept, lines 98-109),
movie_ingestion/metadata_generation/schemas.py (PlotAnalysisOutput,
PlotAnalysisWithJustificationsOutput)


## Align search-side WatchContextMetadata.__str__() to lowercase terms
**Context:** The generation-side `WatchContextOutput.__str__()` in
`movie_ingestion/metadata_generation/schemas.py` lowercases all terms
before joining (`", ".join(t.lower() for t in combined_terms)`). The
search-side `WatchContextMetadata.__str__()` in
`implementation/classes/schemas.py` does NOT lowercase
(`", ".join(combined_terms)`). This means embedding text will differ
between generation and search if terms contain uppercase characters.
The search-side schema should be updated to lowercase for consistency.
**When:** When deploying generation pipeline results to the production
search index.
**See:** implementation/classes/schemas.py (WatchContextMetadata),
movie_ingestion/metadata_generation/schemas.py (WatchContextOutput)


## ~~Remove _DEFAULT_KWARGS from remaining generators~~ DONE
Completed: removed `_DEFAULT_KWARGS` and `effective_kwargs` indirection from all 6 generators
(plot_analysis, viewer_experience, watch_context, narrative_techniques, production_keywords,
source_of_inspiration). They now pass `**kwargs` directly, matching reception's pattern.
plot_events unchanged (retains its defaults).


## Retry failed Gemini plot_events generations with fallback provider
**Context:** 2 of 70 evaluation test set movies (Forrest Gump tmdb_id=13, Fifty Shades of Grey tmdb_id=216015) consistently fail plot_events generation via Gemini 2.5 Flash Lite — the model returns None, likely due to content filtering. These movies have rows in `wave1_results` but with NULL `plot_events`. Consider retrying with a fallback provider (e.g., OpenAI gpt-5-mini) for movies that fail the primary provider.
**When:** Before running Wave 2 evaluation that depends on complete plot_events coverage.
**See:** movie_ingestion/metadata_generation/wave1_runner.py, ingestion_data/tracker.db (wave1_results table)


## ~~Remove debug print statements from metadata generators~~ DONE
Removed temporary user prompt print statements from all 6 generators.


## Update test_eval_plot_events.py for reference-free evaluation
**Context:** `unit_tests/test_eval_plot_events.py` imports `generate_reference_responses`
which was removed during the evaluation pipeline restructuring. The test file will fail
at collection time. Tests need updating to remove reference-related test cases and verify
the new reference-free flow (source data in judge prompt, staggered runs, Anthropic
provider defaults).
**When:** Next time evaluation tests are being worked on.
**See:** unit_tests/test_eval_plot_events.py,
movie_ingestion/metadata_generation/evaluations/plot_events.py



## ~~Backfill plot_summaries after IMDB re-scrape~~ STALE
The premise was wrong — the `imdb_data` table already has movies with both
synopses and plot_summaries. DB query (2026-03-21) shows: of 109,277
imdb_quality_passed movies, 22,655 have synopses across all plot_summary
counts (5,215 with 0 plots, 7,881 with 1, 4,665 with 2, 4,894 with 3).


## Handle long synopses (>8K chars) before embedding
**Context:** ~2,752 synopsis movies exceed 8K chars (~2K tokens), with some
reaching 60K chars. The embedding model (text-embedding-3-small) has a hard
8,191 token limit and quality degrades with longer inputs. LLM-based
distillation via gpt-5-nano was tested and abandoned — the model compressed
too aggressively (76% reduction vs target ~30%) and introduced hallucinations.
Alternative approaches: truncation to a char/token limit, or handling in the
plot_events generator (Option A prompt can instruct the LLM to work with a
truncated version).
**When:** Before generating production embeddings for synopsis movies.
**See:** docs/decisions/ADR-033-plot-events-cost-optimization.md (Section 1)


## Replace .lower() with normalize_string() in all generation-side __str__() methods
**Context:** docs/conventions.md states that `__str__()` methods on Pydantic
schema classes feeding the embedding pipeline must use `normalize_string()`
from `implementation/misc/helpers.py` (NFC normalization, lowercase, diacritic
removal). All generation-side schemas in `movie_ingestion/metadata_generation/schemas.py`
currently use `.lower()` instead. This includes PlotEventsOutput, ReceptionOutput,
PlotAnalysisOutput, ViewerExperienceOutput, WatchContextOutput, and all their
with-justifications variants. Should be a single cross-cutting change.
**When:** Before generating production embeddings from the new pipeline.
**See:** docs/conventions.md (lines 19-23),
movie_ingestion/metadata_generation/schemas.py (all __str__ methods),
implementation/misc/helpers.py (normalize_string)


## Update unit tests for ADR-033 signature changes
**Context:** The ADR-033 implementation changed signatures in plot_events and
source_of_inspiration generators. `build_plot_events_user_prompt` now returns
`Tuple[str, str]` instead of `str`. `build_source_of_inspiration_user_prompt`
and `generate_source_of_inspiration` no longer accept `plot_synopsis`. Unit
tests for both generators will fail at import/call time until updated.
**When:** Next time generator tests are being worked on.
**See:** unit_tests/test_source_of_inspiration_generator.py,
unit_tests/test_plot_events_generator.py (if it exists),
movie_ingestion/metadata_generation/generators/plot_events.py,
movie_ingestion/metadata_generation/generators/source_of_inspiration.py


## Update plot_events embedding to use synopsis when available, generated plot_summary as fallback
**Context:** The plot_events vector space embedding process should prefer
the IMDB synopsis (human-written, detailed) as the embedding input text
when one exists for a movie. For movies without a synopsis, the
LLM-generated `plot_summary` from the plot_events metadata should be used
instead. This aligns with ADR-033's two-branch strategy where synopsis
movies (Branch A) skip LLM generation for plot_events entirely — their
synopsis is the higher-quality signal for embedding. The embedding
pipeline needs to implement this conditional logic: check for synopsis
presence, use it if available, otherwise fall back to the generated
plot_summary output. **Important:** the same `MIN_SYNOPSIS_CHARS` threshold
(1,000 chars) from the plot_events generator must be applied here — synopses
below this length are review blurbs, not plot recounts, and the generated
plot_summary will be higher quality for embedding. See
`movie_ingestion/metadata_generation/generators/plot_events.py` for the
threshold constant and rationale.
**When:** When building the production embedding pipeline for plot_events
vectors (after ADR-033 implementation is complete).
**See:** docs/decisions/ADR-033-plot-events-cost-optimization.md,
implementation/vectorize.py, movie_ingestion/metadata_generation/schemas.py (PlotEventsOutput),
movie_ingestion/metadata_generation/generators/plot_events.py (MIN_SYNOPSIS_CHARS)


