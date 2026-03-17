# TODO

Tracks actionable items discovered during development sessions.
Items here are things to address when the relevant work begins,
not urgent fixes.

## Run Stage 5 filter on scored movies
**Context:** v4 scorer, threshold analysis, and filter script are all
complete. Per-group thresholds determined (has_providers=0.486,
no_providers_new=0.55, no_providers_old=0.654). Filter script at
imdb_quality_scoring/imdb_filter.py is ready to run. Advances survivors
from imdb_quality_calculated → imdb_quality_passed.
**When:** Next session — run the filter to produce the final candidate set.
**See:** movie_ingestion/imdb_quality_scoring/imdb_filter.py,
movie_ingestion/scoring_utils.py (IMDB_QUALITY_THRESHOLDS)


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
**Status:** Plot events generator and prompt now implemented with "Title (Year)"
format. Remaining generators (reception, plot_analysis, viewer_experience,
watch_context, narrative_techniques, production) still need prompt implementation.
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


## Implement request_builder.py in evaluations package
**Context:** `movie_ingestion/metadata_generation/evaluations/request_builder.py`
is currently a stub containing only a docstring. It will be needed for
future evaluation types that require more complex prompt assembly.
**When:** When building the next metadata type evaluation after plot_events.
**See:** movie_ingestion/metadata_generation/evaluations/request_builder.py


## ~~Remove debug print statement from plot_events generator~~ DONE
Removed as a side effect of the `build_plot_events_user_prompt()` extraction refactor.
