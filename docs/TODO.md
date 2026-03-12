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


## Add release year next to title in all LLM metadata generation
**Context:** The LLM metadata generation prompts should include the
release year alongside the movie title (e.g., "The Matrix (1999)")
across all 7 metadata types. This gives the LLM better temporal context
when generating plot analysis, viewer experience, reception, and other
metadata — helping it distinguish remakes, place films in their era,
and produce more accurate descriptions.
**Status:** Addressed in the new metadata generation module design
(Decision 6 in docs/llm_metadata_generation_new_flow.md). All generators
in movie_ingestion/metadata_generation/generators/ use "Title (Year)" format.
Will be fully resolved when prompts are implemented.
**See:** movie_ingestion/metadata_generation/inputs.py (format_title),
docs/llm_metadata_generation_new_flow.md
