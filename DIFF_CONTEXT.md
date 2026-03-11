# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Round budget to int in IMDB parser
Files: movie_ingestion/imdb_scraping/parsers.py | IMDB GraphQL API occasionally returns budget as a float; added `round()` before passing to Pydantic model to prevent ValidationError.
