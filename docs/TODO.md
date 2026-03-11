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

## Update docs/conventions.md status progression
**Context:** Line 120 of conventions.md still references
`essential_data_passed` which was renamed to `imdb_quality_passed`
in the v2 scorer changes. Cannot be updated autonomously — needs
/solidify-draft-conventions or manual edit.
**When:** Next convention review cycle.
**See:** docs/conventions.md:120, movie_ingestion/tracker.py

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
