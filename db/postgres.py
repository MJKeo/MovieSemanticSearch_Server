"""
Database connection pool and query methods for the API service.

This module provides a psycopg v3 AsyncConnectionPool configured for production use,
along with async helper functions for executing queries and public methods for
upserting/inserting movie and lexical data.
"""

import asyncio
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence
from psycopg_pool import AsyncConnectionPool
from implementation.misc.helpers import normalize_string
from implementation.misc.sql_like import escape_like
from implementation.classes.enums import Genre
from implementation.classes.schemas import MetadataFilters


class PostingTable(Enum):
    """Supported posting tables for lexical matching and exclusion resolution."""
    PERSON = "lex.inv_person_postings"
    CHARACTER = "lex.inv_character_postings"
    STUDIO = "lex.inv_studio_postings"
    TITLE_TOKEN = "lex.inv_title_token_postings"

@dataclass(frozen=True, slots=True)
class _TitleTokenMatchConfig:
    """Configuration for one title-token matching mode."""
    use_exact_match: bool
    limit: int | None  # None = unbounded

_TITLE_TOKEN_MATCH_EXACT_ONLY = _TitleTokenMatchConfig(
    use_exact_match=True,
    limit=None,
)
_TITLE_TOKEN_MATCH_SUBSTRING = _TitleTokenMatchConfig(
    use_exact_match=False,
    limit=500,
)

# Matching boundaries:
# - Tokens with length <= 3 stay exact-only.
# - Tokens with length >= 3 use substring matching (capped at 500).
_STRING_MATCH_EXACT_ONLY_MAX_LEN = 3
# Maximum term_ids returned per query phrase during resolution.
# Phrases matching more character names than this are too vague to be
# useful and would bloat the posting join.
_CHARACTER_RESOLVE_LIMIT_PER_PHRASE: int = 500

# In-memory genre mapping cache: Genre enum -> DB genre_id.
# Loaded lazily on first request that requires genre filtering.
_GENRE_ID_CACHE: dict[Genre, int] = {}
_GENRE_BY_NORMALIZED_NAME: dict[str, Genre] = {
    genre.normalized_name: genre for genre in Genre
}
_GENRE_CACHE_LOCK = asyncio.Lock()

# Base query used for title token matching
_BASE_TITLE_TOKEN_QUERY = """\
SELECT d.string_id
FROM lex.title_token_strings d
JOIN lex.title_token_doc_frequency df
  ON df.term_id = d.string_id"""


def _build_conninfo() -> str:
    """
    Build a libpq connection string from environment variables.
    
    Returns:
        A connection string in the format expected by psycopg.
    """
    return (
        f"host={os.getenv('POSTGRES_HOST')} "
        f"dbname={os.getenv('POSTGRES_DB')} "
        f"user={os.getenv('POSTGRES_USER')} "
        f"password={os.getenv('POSTGRES_PASSWORD')}"
    )


# Create the connection pool with production-ready settings
# The pool is created inert (open=False) and will be explicitly opened
# during FastAPI startup via the lifespan handler.
pool = AsyncConnectionPool(
    conninfo=_build_conninfo(),
    min_size=2,           # Keep 2 warm connections for steady-state traffic
    max_size=10,          # Allow up to 10 connections for burst capacity
    max_lifetime=1800,    # Recycle connections after 30 minutes to prevent staleness
    max_idle=300,         # Close idle connections above min_size after 5 minutes
    timeout=5.0,           # Wait up to 5s for a connection before raising PoolTimeout
    open=False,           # Don't open connections at import time; opened in lifespan
)


# ===============================
#     PRIVATE BASE METHODS
# ===============================

async def _execute_read(query: str, params: Sequence[object] | None = None) -> list[tuple]:
    """
    Execute a read query and return all rows.
    
    Private helper method for internal use.
    
    Args:
        query: SQL query string with parameter placeholders (%s).
        params: Optional sequence of parameters to bind to the query.
    
    Returns:
        List of tuples, where each tuple represents a row.
    """
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, params)
            return await cur.fetchall()


async def _execute_read_one(query: str, params: Sequence[object] | None = None) -> tuple | None:
    """
    Execute a read query and return a single row, or None if no rows match.
    
    Private helper method for internal use.
    
    Args:
        query: SQL query string with parameter placeholders (%s).
        params: Optional sequence of parameters to bind to the query.
    
    Returns:
        A tuple representing the first row, or None if no rows were returned.
    """
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, params)
            return await cur.fetchone()


async def _execute_write(
    query: str,
    params: Sequence[object] | None = None,
    fetch_one: bool = False,
):
    """
    Execute a write query (INSERT, UPDATE, DELETE) with an explicit commit.
    
    Private helper method for internal use. The connection is used within a transaction.
    On clean exit, the transaction is explicitly committed. If an exception occurs,
    the transaction is rolled back automatically by the connection context manager.
    
    Args:
        query: SQL query string with parameter placeholders (%s).
        params: Optional sequence of parameters to bind to the query.
        fetch_one: If True, fetch and return the first row (e.g., for RETURNING clauses).
    
    Returns:
        If fetch_one is True, returns the first row as a tuple, otherwise None.
    """
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, params)
            result = await cur.fetchone() if fetch_one else None
        await conn.commit()  # Explicitly commit the transaction
        return result


# ===============================
#     PRIVATE HELPER METHODS
# ===============================


def _get_title_token_tier_config(token_len: int) -> _TitleTokenMatchConfig:
    """
    Select the title-token matching mode from token length.

    Length <= 3 remains exact-only. Longer tokens use substring matching
    with a capped candidate list.
    """
    if token_len <= _STRING_MATCH_EXACT_ONLY_MAX_LEN:
        return _TITLE_TOKEN_MATCH_EXACT_ONLY
    else:
        return _TITLE_TOKEN_MATCH_SUBSTRING

def _build_title_token_query(
    tier: _TitleTokenMatchConfig,
    token: str,
    max_df: int,
) -> tuple[str, tuple]:
    """
    Assemble the SQL query and params tuple for a given tier config.

    Params are accumulated in the same order as %s placeholders appear
    in the assembled SQL to prevent positional mismatches.
    """
    where_clauses: list[str] = []
    params: list[str | int | float] = []

    is_exact_only = tier.use_exact_match

    if is_exact_only:
        where_clauses.append("d.norm_str = %s")
        params.append(token)
    else:
        where_clauses.append(r"d.norm_str LIKE %s ESCAPE '\'")
        params.append(f"%{escape_like(token)}%")

    # Doc frequency filter — always present, always last in WHERE.
    where_clauses.append("df.doc_frequency <= %s")
    params.append(max_df)

    # ── Assemble ────────────────────────────────────────────────────
    parts = [_BASE_TITLE_TOKEN_QUERY, "WHERE\n  " + "\n  AND ".join(where_clauses)]

    if not is_exact_only:
        parts.append(
            "ORDER BY\n"
            "  (d.norm_str = %s) DESC,\n"
            "  length(d.norm_str) ASC"
        )
        params.append(token)

    if tier.limit is not None:
        parts.append(f"LIMIT {tier.limit}")

    query = "\n".join(parts)
    return query, tuple(params)


async def _build_eligible_cte(filters: MetadataFilters) -> tuple[str, list]:
    """
    Build the SQL fragment and parameter list for a MATERIALIZED eligible-set
    CTE against public.movie_card.

    Returns:
        (cte_sql, params) where cte_sql is the full
        ``eligible AS MATERIALIZED (...)`` block ready to prepend into a
        WITH chain, and params is the ordered list of bind values.
    """
    conditions: list[str] = []
    params: list = []

    if filters.min_release_ts is not None and filters.max_release_ts is not None:
        conditions.append("release_ts BETWEEN %s AND %s")
        params.extend((filters.min_release_ts, filters.max_release_ts))
    elif filters.min_release_ts is not None:
        conditions.append("release_ts >= %s")
        params.append(filters.min_release_ts)
    elif filters.max_release_ts is not None:
        conditions.append("release_ts <= %s")
        params.append(filters.max_release_ts)

    if filters.min_runtime is not None and filters.max_runtime is not None:
        conditions.append("runtime_minutes BETWEEN %s AND %s")
        params.extend((filters.min_runtime, filters.max_runtime))
    elif filters.min_runtime is not None:
        conditions.append("runtime_minutes >= %s")
        params.append(filters.min_runtime)
    elif filters.max_runtime is not None:
        conditions.append("runtime_minutes <= %s")
        params.append(filters.max_runtime)

    if filters.min_maturity_rank is not None and filters.max_maturity_rank is not None:
        conditions.append("maturity_rank BETWEEN %s AND %s")
        params.extend((filters.min_maturity_rank, filters.max_maturity_rank))
    elif filters.min_maturity_rank is not None:
        conditions.append("maturity_rank >= %s")
        params.append(filters.min_maturity_rank)
    elif filters.max_maturity_rank is not None:
        conditions.append("maturity_rank <= %s")
        params.append(filters.max_maturity_rank)

    if filters.genres is not None:
        genre_id_map = await fetch_genre_ids_by_name(filters.genres)
        genre_ids = [genre_id_map[genre] for genre in filters.genres if genre in genre_id_map]
        conditions.append("genre_ids && %s::int[]")
        params.append(genre_ids)

    if filters.watch_offer_keys is not None:
        conditions.append("watch_offer_keys && %s::int[]")
        params.append(filters.watch_offer_keys)

    where_clause = " AND ".join(conditions) if conditions else "TRUE"

    cte_sql = (
        f"eligible AS MATERIALIZED (\n"
        f"            SELECT movie_id, title_token_count\n"
        f"            FROM public.movie_card\n"
        f"            WHERE {where_clause}\n"
        f"        )"
    )
    return cte_sql, params


async def _ensure_genre_cache_loaded() -> None:
    """
    Populate the in-memory genre cache if it is currently empty.

    This performs a single bulk query to load all genre mappings from
    ``lex.genre_dictionary``. The cache is process-local and shared across
    all requests handled by that process.
    """
    if _GENRE_ID_CACHE:
        return

    async with _GENRE_CACHE_LOCK:
        if _GENRE_ID_CACHE:
            return

        rows = await _execute_read("SELECT name, genre_id FROM lex.genre_dictionary")
        for name, genre_id in rows:
            genre_enum = _GENRE_BY_NORMALIZED_NAME.get(name)
            if genre_enum is None:
                # Ignore dictionary rows that are not represented by the Genre enum.
                continue
            _GENRE_ID_CACHE[genre_enum] = int(genre_id)


async def fetch_genre_ids_by_name(genres: list[Genre]) -> dict[Genre, int]:
    """
    Resolve Genre enum values to their integer IDs in ``lex.genre_dictionary``.

    Args:
        genres: Genre enum values requested by the caller.

    Returns:
        Mapping of Genre -> genre_id for enum values present in the cache/table.
    """
    if not genres:
        return {}

    await _ensure_genre_cache_loaded()

    unique_genres = list(dict.fromkeys(genres))
    return {
        genre: _GENRE_ID_CACHE[genre]
        for genre in unique_genres
        if genre in _GENRE_ID_CACHE
    }


# ===============================
#        PUBLIC METHODS
# ===============================

async def check_postgres() -> str:
    """
    Ping Postgres via the pool to verify connectivity.
    
    This validates that the pool can successfully obtain a connection
    and execute a simple query. Used by the /health endpoint.
    
    Returns:
        'ok' if the check succeeds, otherwise an error message string.
    """
    try:
        async with pool.connection() as conn:
            await conn.execute("SELECT 1")
        return "ok"
    except Exception as e:
        return str(e)


# ===============================
#       INGESTION METHODS
# ===============================

async def upsert_lexical_dictionary(norm_str: str) -> Optional[int]:
    """
    Upsert a normalized string in lex.lexical_dictionary and return string_id.
    
    Args:
        norm_str: Normalized string to upsert.
    
    Returns:
        The string_id of the upserted or existing row.
    """
    query = """
    INSERT INTO lex.lexical_dictionary (norm_str, touched_at, created_at)
    VALUES (%s, now(), now())
    ON CONFLICT (norm_str) DO UPDATE SET
        touched_at = now()
    RETURNING string_id;
    """
    row = await _execute_write(query, (norm_str,), fetch_one=True)
    return row[0] if row else None


async def upsert_title_token_string(string_id: int, norm_str: str) -> None:
    """
    Upsert a title-token lookup row in lex.title_token_strings.
    
    Args:
        string_id: The string ID to associate with the normalized string.
        norm_str: Normalized string value.
    """
    query = """
    INSERT INTO lex.title_token_strings (string_id, norm_str)
    VALUES (%s, %s)
    ON CONFLICT (string_id) DO UPDATE SET
        norm_str = EXCLUDED.norm_str;
    """
    await _execute_write(query, (string_id, norm_str))


async def upsert_character_string(string_id: int, norm_str: str) -> None:
    """
    Upsert a character string row in lex.character_strings.
    
    Args:
        string_id: The string ID.
        norm_str: The normalized string.
    """
    query = """
    INSERT INTO lex.character_strings (string_id, norm_str)
    VALUES (%s, %s)
    ON CONFLICT (string_id) DO UPDATE SET
        norm_str = EXCLUDED.norm_str;
    """
    await _execute_write(query, (string_id, norm_str))


async def batch_insert_title_token_postings(term_ids: list[int], movie_id: int) -> None:
    """
    Insert title-token postings for one movie in a single round-trip.

    Args:
        term_ids: Title-token term IDs to insert.
        movie_id: Movie ID that owns all postings.
    """
    if not term_ids:
        return
    query = """
    INSERT INTO lex.inv_title_token_postings (term_id, movie_id)
    SELECT unnest(%s::bigint[]), %s
    ON CONFLICT (term_id, movie_id) DO NOTHING;
    """
    await _execute_write(query, (term_ids, movie_id))


async def batch_insert_person_postings(term_ids: list[int], movie_id: int) -> None:
    """
    Insert person postings for one movie in a single round-trip.

    Args:
        term_ids: Person term IDs to insert.
        movie_id: Movie ID that owns all postings.
    """
    if not term_ids:
        return
    query = """
    INSERT INTO lex.inv_person_postings (term_id, movie_id)
    SELECT unnest(%s::bigint[]), %s
    ON CONFLICT (term_id, movie_id) DO NOTHING;
    """
    await _execute_write(query, (term_ids, movie_id))


async def batch_insert_character_postings(term_ids: list[int], movie_id: int) -> None:
    """
    Insert character postings for one movie in a single round-trip.

    Args:
        term_ids: Character term IDs to insert.
        movie_id: Movie ID that owns all postings.
    """
    if not term_ids:
        return
    query = """
    INSERT INTO lex.inv_character_postings (term_id, movie_id)
    SELECT unnest(%s::bigint[]), %s
    ON CONFLICT (term_id, movie_id) DO NOTHING;
    """
    await _execute_write(query, (term_ids, movie_id))


async def batch_insert_studio_postings(term_ids: list[int], movie_id: int) -> None:
    """
    Insert studio postings for one movie in a single round-trip.

    Args:
        term_ids: Studio term IDs to insert.
        movie_id: Movie ID that owns all postings.
    """
    if not term_ids:
        return
    query = """
    INSERT INTO lex.inv_studio_postings (term_id, movie_id)
    SELECT unnest(%s::bigint[]), %s
    ON CONFLICT (term_id, movie_id) DO NOTHING;
    """
    await _execute_write(query, (term_ids, movie_id))


async def upsert_genre_dictionary(genre_id: int, name: str) -> None:
    """
    Upsert a genre lookup row in lex.genre_dictionary.
    
    Args:
        genre_id: The genre ID.
        name: The genre name.
    """
    query = """
    INSERT INTO lex.genre_dictionary (genre_id, name)
    VALUES (%s, %s)
    ON CONFLICT (genre_id) DO UPDATE SET
        name = EXCLUDED.name;
    """
    await _execute_write(query, (genre_id, name))


async def upsert_provider_dictionary(provider_id: int, name: str) -> None:
    """
    Upsert a provider lookup row in lex.provider_dictionary.
    
    Args:
        provider_id: The provider ID.
        name: The provider name.
    """
    query = """
    INSERT INTO lex.provider_dictionary (provider_id, name)
    VALUES (%s, %s)
    ON CONFLICT (provider_id) DO UPDATE SET
        name = EXCLUDED.name;
    """
    await _execute_write(query, (provider_id, name))


async def upsert_watch_method_dictionary(method_id: int, name: str) -> None:
    """
    Upsert a watch-method lookup row in lex.watch_method_dictionary.
    
    Args:
        method_id: The watch method ID.
        name: The watch method name.
    """
    query = """
    INSERT INTO lex.watch_method_dictionary (method_id, name)
    VALUES (%s, %s)
    ON CONFLICT (method_id) DO UPDATE SET
        name = EXCLUDED.name;
    """
    await _execute_write(query, (method_id, name))


async def upsert_maturity_dictionary(maturity_rank: int, label: str) -> None:
    """
    Upsert a maturity-rating lookup row in lex.maturity_dictionary.
    
    Args:
        maturity_rank: The maturity rank.
        label: The maturity label (e.g., "PG-13", "R").
    """
    query = """
    INSERT INTO lex.maturity_dictionary (maturity_rank, label)
    VALUES (%s, %s)
    ON CONFLICT (maturity_rank) DO UPDATE SET
        label = EXCLUDED.label;
    """
    await _execute_write(query, (maturity_rank, label))


async def upsert_language_dictionary(language_id: int, name: str) -> None:
    """
    Upsert a language lookup row in lex.language_dictionary.
    
    Args:
        language_id: The language ID.
        name: The language name.
    """
    query = """
    INSERT INTO lex.language_dictionary (language_id, name)
    VALUES (%s, %s)
    ON CONFLICT (language_id) DO UPDATE SET
        name = EXCLUDED.name;
    """
    await _execute_write(query, (language_id, name))


async def refresh_title_token_doc_frequency() -> None:
    """
    Refresh the lex.title_token_doc_frequency materialized view concurrently.

    This should be called after each bulk ingest so that max-df stop-word
    filtering reflects the latest posting counts.  The CONCURRENTLY option
    avoids blocking reads during the rebuild (requires the unique index
    idx_title_token_df_term_id on the view).
    """
    await _execute_write(
        "REFRESH MATERIALIZED VIEW CONCURRENTLY lex.title_token_doc_frequency;"
    )

async def upsert_movie_card(
    movie_id: int,
    title: str,
    poster_url: Optional[str],
    release_ts: Optional[int],
    runtime_minutes: Optional[int],
    maturity_rank: Optional[int],
    genre_ids: Sequence[int],
    watch_offer_keys: Sequence[int],
    audio_language_ids: Sequence[int],
    reception_score: Optional[float],
    title_token_count: int,
) -> None:
    """
    Upsert a row in public.movie_card for canonical metadata storage.
    
    Args:
        movie_id: Unique movie identifier.
        title: Movie title.
        poster_url: URL to the movie poster image.
        release_ts: Release timestamp (Unix seconds).
        runtime_minutes: Movie runtime in minutes.
        maturity_rank: Maturity rating rank.
        genre_ids: List of genre IDs.
        watch_offer_keys: List of watch offer keys (encoded provider+method).
        audio_language_ids: List of audio language IDs.
        reception_score: Precomputed reception score from IMDB/Metacritic.
        title_token_count: Number of tokens in the title.
    """
    query = """
    INSERT INTO public.movie_card (
        movie_id, title, poster_url, release_ts, runtime_minutes,
        maturity_rank, genre_ids, watch_offer_keys, audio_language_ids,
        reception_score, title_token_count, created_at, updated_at
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now(), now())
    ON CONFLICT (movie_id) DO UPDATE SET
        title = EXCLUDED.title,
        poster_url = EXCLUDED.poster_url,
        release_ts = EXCLUDED.release_ts,
        runtime_minutes = EXCLUDED.runtime_minutes,
        maturity_rank = EXCLUDED.maturity_rank,
        genre_ids = EXCLUDED.genre_ids,
        watch_offer_keys = EXCLUDED.watch_offer_keys,
        audio_language_ids = EXCLUDED.audio_language_ids,
        reception_score = EXCLUDED.reception_score,
        title_token_count = EXCLUDED.title_token_count,
        updated_at = now();
    """
    params = (
        movie_id,
        title,
        poster_url,
        release_ts,
        runtime_minutes,
        maturity_rank,
        list(genre_ids),
        list(watch_offer_keys),
        list(audio_language_ids),
        reception_score,
        title_token_count,
    )
    await _execute_write(query, params)

# ===============================
#        SEARCH METHODS
# ===============================

async def fetch_phrase_term_ids(phrases: list[str]) -> dict[str, int]:
    """
    Batch exact lookup of normalized phrases in lex.lexical_dictionary.

    Used for people and studio entity resolution at query time.
    Phrases not present in the dictionary are silently omitted (they produce
    no candidates).

    Args:
        phrases: List of already-normalized phrase strings.

    Returns:
        Mapping of norm_str → string_id for every phrase that exists.
    """
    if not phrases:
        return {}

    query = """
        SELECT norm_str, string_id
        FROM lex.lexical_dictionary
        WHERE norm_str = ANY(%s::text[])
    """
    search_results = await _execute_read(query, (phrases,))
    return {row[0]: row[1] for row in search_results}


async def fetch_phrase_postings_match_counts(
    table: PostingTable,
    term_ids: list[int],
    filters: Optional[MetadataFilters] = None,
    exclude_movie_ids: Optional[set[int]] = None,
) -> dict[int, int]:
    """
    Count distinct matched term_ids per movie from a phrase posting table.

    Builds a MATERIALIZED eligible CTE when metadata filters are active,
    otherwise queries the posting table directly.

    Args:
        table:    Posting table enum for this phrase bucket.
        term_ids: Resolved string_ids for INCLUDE phrases in this bucket.
        filters:  Optional metadata hard-filters.
        exclude_movie_ids: Optional movie IDs to exclude from results.

    Returns:
        Dict of {movie_id: matched_count}.
    """
    if not term_ids:
        return {}

    table_name = table.value

    use_eligible = filters is not None and filters.is_active

    cte_parts: list[str] = []
    params: list = []

    if use_eligible:
        eligible_cte, eligible_params = await _build_eligible_cte(filters)
        cte_parts.append(eligible_cte)
        params.extend(eligible_params)

    params.append(term_ids)
    exclusion_clause = ""
    if exclude_movie_ids:
        exclusion_clause = (
            "\n          AND NOT (p.movie_id = ANY(%s::bigint[]))"
        )
        params.append(list(exclude_movie_ids))

    eligibility_join = (
        "\n            JOIN eligible e ON e.movie_id = p.movie_id"
        if use_eligible
        else ""
    )

    with_clause = f"WITH {', '.join(cte_parts)}\n        " if cte_parts else ""

    query = f"""
        {with_clause}SELECT p.movie_id, COUNT(DISTINCT p.term_id)::int AS matched
        FROM {table_name} p{eligibility_join}
        WHERE p.term_id = ANY(%s::bigint[]){exclusion_clause}
        GROUP BY p.movie_id
    """
    search_results = await _execute_read(query, params)
    return {row[0]: row[1] for row in search_results}


async def fetch_title_token_ids(
    tokens: list[str],
    max_df: int = 10_000,
) -> dict[int, list[int]]:
    """
    Resolve title token patterns to candidate string_ids in one query.

    Matching policy is evaluated from each literal token (decoded from its
    LIKE pattern):
      - exact-only tier for short tokens
      - substring tier for longer tokens

    Args:
        tokens: Normalized title tokens in stable input order.
        max_df: Maximum document frequency for candidate terms.

    Returns:
        Dict of {query_idx: [candidate_string_id, ...]}.
    """
    if not tokens:
        return {}

    query_idxs = list(range(len(tokens)))
    like_patterns = [f"%{escape_like(token)}%" for token in tokens]

    query = r"""
        SELECT sub.query_idx, sub.string_id
        FROM (
            SELECT
                qt.query_idx,
                qt.token,
                d.string_id,
                ROW_NUMBER() OVER (
                    PARTITION BY qt.query_idx
                    ORDER BY
                        (d.norm_str = qt.token) DESC,
                        length(d.norm_str)
                ) AS rn
            FROM unnest(%s::int[], %s::text[], %s::text[]) AS qt(query_idx, token, like_pattern)
            JOIN lex.title_token_strings d
              ON (
                    (length(qt.token) <= %s AND d.norm_str = qt.token)
                 OR (length(qt.token) > %s AND d.norm_str LIKE qt.like_pattern ESCAPE '\')
                 )
            JOIN lex.title_token_doc_frequency df
              ON df.term_id = d.string_id
            WHERE df.doc_frequency <= %s
        ) sub
        WHERE length(sub.token) <= %s
           OR sub.rn <= %s
    """
    params = [
        query_idxs,
        tokens,
        like_patterns,
        _STRING_MATCH_EXACT_ONLY_MAX_LEN,
        _STRING_MATCH_EXACT_ONLY_MAX_LEN,
        max_df,
        _STRING_MATCH_EXACT_ONLY_MAX_LEN,
        _TITLE_TOKEN_MATCH_SUBSTRING.limit,
    ]
    search_results = await _execute_read(query, params)

    result: dict[int, list[int]] = {}
    for query_idx, string_id in search_results:
        result.setdefault(query_idx, []).append(string_id)
    return result


async def fetch_title_token_ids_exact(
    tokens: list[str],
    max_df: int = 10_000,
) -> dict[int, list[int]]:
    """
    Resolve exact title token matches for a list of tokens in one query.

    This path is used for EXCLUDE title tokens where expansion is disabled.

    Args:
        tokens: Normalized title tokens in stable input order.
        max_df: Maximum document frequency for candidate terms.

    Returns:
        Dict of {token_idx: [exact_match_string_id, ...]}.
    """
    if not tokens:
        return {}

    query_idxs = list(range(len(tokens)))
    query = """
        SELECT qt.query_idx, d.string_id
        FROM unnest(%s::int[], %s::text[]) AS qt(query_idx, token)
        JOIN lex.title_token_strings d
          ON d.norm_str = qt.token
        JOIN lex.title_token_doc_frequency df
          ON df.term_id = d.string_id
        WHERE df.doc_frequency <= %s
    """
    params = [query_idxs, tokens, max_df]
    search_results = await _execute_read(query, params)

    result: dict[int, list[int]] = {}
    for query_idx, string_id in search_results:
        result.setdefault(query_idx, []).append(string_id)
    return result


async def fetch_character_term_ids(
    query_idxs: list[int],
    like_patterns: list[str],
) -> dict[int, list[int]]:
    """
    Fetch character matches from lex.inv_character_postings.
    """
    query = r"""
        SELECT sub.query_idx, sub.string_id
        FROM (
            SELECT
                qc.query_idx,
                cs.string_id,
                ROW_NUMBER() OVER (
                    PARTITION BY qc.query_idx
                    ORDER BY length(cs.norm_str)
                ) AS rn
            FROM unnest(%s::int[], %s::text[]) AS qc(query_idx, like_pattern)
            JOIN lex.character_strings cs
              ON cs.norm_str LIKE qc.like_pattern ESCAPE '\'
        ) sub
        WHERE sub.rn <= %s
    """
    params = [query_idxs, like_patterns, _CHARACTER_RESOLVE_LIMIT_PER_PHRASE]
    search_results = await _execute_read(query, params)

    result: dict[int, list[int]] = {}
    for query_idx, string_id in search_results:
        result.setdefault(query_idx, []).append(string_id)

    return result


async def fetch_character_match_counts(
    query_idxs: list[int], 
    term_ids: list[int], 
    use_eligible: bool, 
    filters: Optional[MetadataFilters] = None,
    exclude_movie_ids: Optional[set[int]] = None,
) -> dict[int, int]:
    """
    Fetch character match counts from lex.inv_character_postings.

    Args:
        query_idxs: List of query indices.
        term_ids: List of term IDs.
        use_eligible: Whether to use the eligible CTE.
        filters: Optional metadata hard-filters.
        exclude_movie_ids: Optional movie IDs to exclude from results.

    Returns:
        Dict of {movie_id: matched_character_count}.
    """
    cte_parts: list[str] = []
    params: list = []

    if use_eligible:
        eligible_cte, eligible_params = await _build_eligible_cte(filters)
        cte_parts.append(eligible_cte)
        params.extend(eligible_params)

    params.extend([query_idxs, term_ids])
    cte_parts.append(
        """q_chars AS (
            SELECT unnest(%s::int[]) AS query_idx,
                   unnest(%s::bigint[]) AS term_id
        )"""
    )

    eligibility_join = (
        "\n            JOIN eligible e ON e.movie_id = p.movie_id"
        if use_eligible
        else ""
    )
    exclusion_clause = ""
    if exclude_movie_ids:
        exclusion_clause = (
            "\n            WHERE NOT (p.movie_id = ANY(%s::bigint[]))"
        )
        params.append(list(exclude_movie_ids))

    cte_parts.append(
        f"""character_matches AS (
            SELECT
                p.movie_id,
                COUNT(DISTINCT qc.query_idx)::int AS matched
            FROM q_chars qc
            JOIN lex.inv_character_postings p
              ON p.term_id = qc.term_id{eligibility_join}{exclusion_clause}
            GROUP BY p.movie_id
        )"""
    )

    with_clause = "WITH " + ",\n        ".join(cte_parts)
    query = f"""
        {with_clause}
        SELECT movie_id, matched
        FROM character_matches
    """

    search_results = await _execute_read(query, params)
    return {row[0]: row[1] for row in search_results}


async def fetch_character_match_counts_by_query(
    query_idxs: list[int],
    term_ids: list[int],
    use_eligible: bool,
    filters: Optional[MetadataFilters] = None,
    exclude_movie_ids: Optional[set[int]] = None,
) -> dict[int, dict[int, int]]:
    """
    Fetch per-query-index character match counts from inv_character_postings.

    Args:
        query_idxs: Query indices aligned with term_ids.
        term_ids: Resolved character term IDs.
        use_eligible: Whether to include the eligible CTE.
        filters: Optional metadata hard-filters.
        exclude_movie_ids: Optional movie IDs to exclude from results.

    Returns:
        Dict keyed by query_idx, each containing {movie_id: matched_count}.
    """
    cte_parts: list[str] = []
    params: list = []

    if use_eligible:
        eligible_cte, eligible_params = await _build_eligible_cte(filters)
        cte_parts.append(eligible_cte)
        params.extend(eligible_params)

    params.extend([query_idxs, term_ids])
    cte_parts.append(
        """q_chars AS (
            SELECT unnest(%s::int[]) AS query_idx,
                   unnest(%s::bigint[]) AS term_id
        )"""
    )

    eligibility_join = (
        "\n            JOIN eligible e ON e.movie_id = p.movie_id"
        if use_eligible
        else ""
    )
    exclusion_clause = ""
    if exclude_movie_ids:
        exclusion_clause = (
            "\n            WHERE NOT (p.movie_id = ANY(%s::bigint[]))"
        )
        params.append(list(exclude_movie_ids))

    cte_parts.append(
        f"""character_matches AS (
            SELECT
                qc.query_idx,
                p.movie_id,
                COUNT(DISTINCT qc.query_idx)::int AS matched
            FROM q_chars qc
            JOIN lex.inv_character_postings p
              ON p.term_id = qc.term_id{eligibility_join}{exclusion_clause}
            GROUP BY qc.query_idx, p.movie_id
        )"""
    )

    with_clause = "WITH " + ",\n        ".join(cte_parts)
    query = f"""
        {with_clause}
        SELECT query_idx, movie_id, matched
        FROM character_matches
    """

    search_results = await _execute_read(query, params)
    by_query: dict[int, dict[int, int]] = {}
    for query_idx, movie_id, matched in search_results:
        by_query.setdefault(query_idx, {})[movie_id] = matched
    return by_query


async def fetch_movie_cards(movie_ids: list[int]) -> list[dict]:
    """
    Bulk fetch canonical movie metadata from public.movie_card.

    Single query for all candidates — never per-candidate.  Results feed both
    the reranker (metadata preference scoring) and the final API response
    payload (card rendering).

    Args:
        movie_ids: List of movie IDs to fetch metadata for.

    Returns:
        List of dicts with keys: movie_id, title, poster_url,
        release_ts, runtime_minutes, maturity_rank, genre_ids,
        watch_offer_keys, audio_language_ids, reception_score.
    """
    if not movie_ids:
        return []

    query = """
        SELECT movie_id, title, poster_url, release_ts, runtime_minutes,
               maturity_rank, genre_ids, watch_offer_keys, audio_language_ids,
               reception_score
        FROM public.movie_card
        WHERE movie_id = ANY(%s::bigint[])
    """
    columns = [
        "movie_id", "title", "poster_url", "release_ts",
        "runtime_minutes", "maturity_rank", "genre_ids", "watch_offer_keys",
        "audio_language_ids", "reception_score",
    ]

    search_results = await _execute_read(query, (movie_ids,))
    return [dict(zip(columns, row)) for row in search_results]


async def fetch_title_token_match_scores(
    token_idxs: list[int],
    term_ids: list[int],
    use_eligible: bool,
    f_coeff: float,
    k: int,
    beta: float,
    title_score_threshold: float,
    title_max_candidates: int,
    filters: Optional[MetadataFilters] = None,
    exclude_movie_ids: Optional[set[int]] = None,
) -> dict[int, float]:
    """
    Compute title F-scores for one title search by joining token postings.

    Builds a MATERIALIZED eligible CTE when metadata filters are active
    (avoiding large array transfer), counts matched token positions (m)
    per movie, then applies the coverage-weighted F-score:

        title_score = (1+β²)·(coverage·specificity) / (β²·specificity + coverage)

    where coverage = m/k, specificity = m/L, β = TITLE_SCORE_BETA.

    Results are capped at TITLE_MAX_CANDIDATES (sorted by score desc) in
    the SQL query itself for wire-transfer efficiency.

    Args:
        exclude_movie_ids: Optional movie IDs to exclude from title scoring.

    Returns:
        Dict of {movie_id: title_score} for qualifying movies, capped
        at TITLE_MAX_CANDIDATES entries.
    """
    cte_prefix_parts: list[str] = []
    params: list = []

    # 1) Eligible CTE (only when filters are active)
    if use_eligible:
        eligible_cte, eligible_params = await _build_eligible_cte(filters)
        cte_prefix_parts.append(eligible_cte)
        params.extend(eligible_params)

    # 2) q_tokens CTE (always present)
    params.extend([token_idxs, term_ids])
    cte_prefix_parts.append(
        """q_tokens AS (
            SELECT unnest(%s::int[]) AS token_idx,
                   unnest(%s::bigint[]) AS term_id
        )"""
    )

    # 3) token_matches + title_matches CTEs — one match per query token (no expansion bias)
    # First get distinct (movie_id, token_idx) so each of the n query tokens contributes
    # at most 1 regardless of how many term_ids it expanded to.
    eligibility_join = (
        "\n                JOIN eligible e ON e.movie_id = p.movie_id"
        if use_eligible
        else ""
    )
    exclusion_clause = ""
    if exclude_movie_ids:
        exclusion_clause = (
            "\n            WHERE NOT (p.movie_id = ANY(%s::bigint[]))"
        )
        params.append(list(exclude_movie_ids))

    cte_prefix_parts.append(
        f"""token_matches AS (
            SELECT DISTINCT p.movie_id, qt.token_idx
            FROM q_tokens qt
            JOIN lex.inv_title_token_postings p
              ON p.term_id = qt.term_id{eligibility_join}{exclusion_clause}
        ),
        title_matches AS (
            SELECT movie_id, COUNT(*)::int AS m
            FROM token_matches
            GROUP BY movie_id
        )"""
    )

    # 4) title_scored CTE — applies F-score formula
    params.extend([f_coeff, k, beta**2, k])
    title_count_source = "eligible" if use_eligible else "public.movie_card"
    title_count_alias = "e" if use_eligible else "mc"
    cte_prefix_parts.append(
        f"""title_scored AS (
            SELECT
                tm.movie_id,
                (%s::double precision
                    * ((tm.m::double precision / %s)
                       * (tm.m::double precision / {title_count_alias}.title_token_count)))
                / (%s::double precision
                    * (tm.m::double precision / {title_count_alias}.title_token_count)
                    + (tm.m::double precision / %s))
                AS title_score
            FROM title_matches tm
            JOIN {title_count_source} {title_count_alias}
              ON {title_count_alias}.movie_id = tm.movie_id
            WHERE {title_count_alias}.title_token_count > 0
              AND %s > 0
        )"""
    )
    params.append(k)  # k > 0 guard bound into the WHERE clause

    # 5) Final SELECT with threshold filter + safety cap
    params.extend([title_score_threshold, title_max_candidates])

    with_clause = "WITH " + ",\n        ".join(cte_prefix_parts)
    query = f"""
        {with_clause}
        SELECT movie_id, title_score
        FROM title_scored
        WHERE title_score >= %s
        ORDER BY title_score DESC
        LIMIT %s
    """

    # ── Execute ───────────────────────────────────────────────────────────
    try:
        search_results = await _execute_read(query, params)
        return {row[0]: float(row[1]) for row in search_results}
    except Exception:
        # TODO - Log here
        raise


async def fetch_movie_ids_by_term_ids(
    table: PostingTable,
    term_ids: list[int],
) -> set[int]:
    """
    Resolve posting term IDs into excluded movie IDs for one posting table.

    This helper supports global lexical exclusion: each EXCLUDE entity bucket
    resolves its own term IDs to movie IDs, and the caller unions those movie
    IDs into one cross-bucket exclusion set.

    Args:
        table: Posting table to resolve against.
        term_ids: Posting term IDs for one EXCLUDE bucket.

    Returns:
        Set of movie IDs that contain at least one provided term ID.
    """
    if not term_ids:
        return set()

    query = f"""
        SELECT DISTINCT movie_id
        FROM {table.value}
        WHERE term_id = ANY(%s::bigint[])
    """
    search_results = await _execute_read(query, (term_ids,))
    return {row[0] for row in search_results}

# ===============================
#      PUBLIC HELPER METHODS
# ===============================

async def upsert_phrase_term(value: str) -> int | None:
    """Normalize a phrase and upsert it into lexical_dictionary."""
    normalized = normalize_string(value)
    if not normalized:
        return None
    return await upsert_lexical_dictionary(normalized)