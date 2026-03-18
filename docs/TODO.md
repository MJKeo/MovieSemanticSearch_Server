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


## Verify WHAM structured output end-to-end
**Context:** The WHAM provider (`generate_wham_response_async`) has been implemented
with `responses.stream()` + `text_format` for structured output, but hasn't been
confirmed working end-to-end with `PlotEventsOutput` and `PlotEventsJudgeOutput`.
OAuth flow and token refresh are working. The streaming parse path and
`reasoning_effort="low"` parameter need runtime verification.
**Update:** `temperature` removed from judge call (not supported with reasoning_effort != "none").
`max_tokens`/`max_output_tokens` also confirmed unsupported by WHAM endpoint. The WHAM
handler still maps `max_tokens` → `max_output_tokens` in generic_methods.py — this
remapping code is dead for WHAM and could be cleaned up, but won't cause errors since
callers no longer pass it.
**When:** Next time the evaluation pipeline is run.
**See:** `implementation/llms/generic_methods.py` (generate_wham_response_async),
`movie_ingestion/metadata_generation/evaluations/plot_events.py`

## Clean up dead WHAM parameter handling in generic_methods.py
**Context:** `generate_wham_response_async` extracts and remaps `max_tokens` →
`max_output_tokens` and `temperature` from kwargs, but WHAM rejects both when
reasoning_effort != "none" (and max_output_tokens is never supported by WHAM).
The extraction code is harmless (silently pops unused kwargs) but misleading —
it suggests these params work. Should either remove the handling or add comments
explaining the limitations.
**When:** Low priority — during next cleanup pass on the LLM provider layer.
**See:** `implementation/llms/generic_methods.py:500-513`
