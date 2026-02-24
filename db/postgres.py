"""
Database connection pool and query methods for the API service.

This module provides a psycopg v3 AsyncConnectionPool configured for production use,
along with async helper functions for executing queries and public methods for
upserting/inserting movie and lexical data.
"""

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
class TitleSearchInput:
    """
    Input payload for one title-space lexical search inside the compound query.

    Args:
        token_idxs: Positional query token indices aligned with term_ids.
        term_ids: Resolved title-token term IDs aligned with token_idxs.
        f_coeff: Precomputed title F-score coefficient (1 + beta^2).
        k: Count of non-empty query token positions.
        beta_sq: Squared beta value used by the F-score denominator.
        score_threshold: Minimum title score required for inclusion.
        max_candidates: Per-title-search cap after sorting by score desc.
    """
    token_idxs: list[int]
    term_ids: list[int]
    f_coeff: float
    k: int
    beta_sq: float
    score_threshold: float
    max_candidates: int


@dataclass(frozen=True, slots=True)
class CompoundLexicalResult:
    """
    Parsed result container from execute_compound_lexical_search.

    Args:
        people_scores: Matched people counts keyed by movie_id.
        studio_scores: Matched studio counts keyed by movie_id.
        character_by_query: Character matched counts keyed by query_idx then movie_id.
        title_scores_by_search: Title scores keyed by title search index then movie_id.
    """
    people_scores: dict[int, int]
    studio_scores: dict[int, int]
    character_by_query: dict[int, dict[int, int]]
    title_scores_by_search: dict[int, dict[int, float]]

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


async def _execute_on_conn(
    conn,
    query: str,
    params: Sequence[object] | None = None,
    fetch: bool = False,
) -> list[tuple] | None:
    """
    Execute SQL on an optional existing connection.

    Args:
        conn: Existing async psycopg connection when the caller manages transaction
            boundaries. If None, this helper acquires and commits its own connection.
        query: SQL query string with parameter placeholders (%s).
        params: Optional sequence of parameters to bind to the query.
        fetch: When True, return all rows from the cursor.

    Returns:
        Query rows when ``fetch`` is True, otherwise None.
    """
    if conn is not None:
        async with conn.cursor() as cur:
            await cur.execute(query, params)
            return await cur.fetchall() if fetch else None

    async with pool.connection() as fallback_conn:
        async with fallback_conn.cursor() as cur:
            await cur.execute(query, params)
            result = await cur.fetchall() if fetch else None
        await fallback_conn.commit()
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
        genre_ids = [genre.genre_id for genre in filters.genres]
        if genre_ids:
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

async def batch_upsert_lexical_dictionary(
    norm_strings: list[str],
    conn=None,
) -> dict[str, int]:
    """
    Batch upsert normalized strings into lex.lexical_dictionary.

    Args:
        norm_strings: Normalized strings to insert or resolve.
        conn: Optional existing async connection for caller-managed transaction scope.

    Returns:
        Mapping of ``norm_str`` to ``string_id`` for every unique input string.
    """
    if not norm_strings:
        return {}

    # Deduplicate, then sort lexicographically so that all concurrent
    # transactions acquire index locks in the same order — preventing the
    # circular-wait condition required for deadlock.
    unique_strings = sorted(dict.fromkeys(norm_strings))

    query = """
    WITH input_strings AS (
        SELECT unnest(%s::text[]) AS norm_str
    ),
    inserted AS (
        INSERT INTO lex.lexical_dictionary (norm_str, created_at)
        SELECT norm_str, now()
        FROM input_strings
        ON CONFLICT (norm_str) DO NOTHING
        RETURNING norm_str, string_id
    )
    SELECT norm_str, string_id FROM inserted
    UNION ALL
    SELECT d.norm_str, d.string_id
    FROM lex.lexical_dictionary d
    JOIN input_strings i ON i.norm_str = d.norm_str
    WHERE NOT EXISTS (
        SELECT 1 FROM inserted ins WHERE ins.norm_str = d.norm_str
    );
    """
    rows = await _execute_on_conn(conn, query, (unique_strings,), fetch=True) or []
    return {str(norm_str): int(string_id) for norm_str, string_id in rows}


async def batch_upsert_title_token_strings(
    string_ids: list[int],
    norm_strings: list[str],
    conn=None,
) -> None:
    """
    Batch upsert title token lookup rows in lex.title_token_strings.

    Args:
        string_ids: String IDs aligned one-to-one with ``norm_strings``.
        norm_strings: Normalized token strings aligned one-to-one with ``string_ids``.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    if not string_ids:
        return
    if len(string_ids) != len(norm_strings):
        raise ValueError("Title token string upsert failed: string_ids and norm_strings lengths differ.")
    unique_pairs = list(dict.fromkeys(zip(string_ids, norm_strings)))
    deduped_string_ids = [pair[0] for pair in unique_pairs]
    deduped_norm_strings = [pair[1] for pair in unique_pairs]

    query = """
    INSERT INTO lex.title_token_strings (string_id, norm_str)
    SELECT unnest(%s::bigint[]), unnest(%s::text[])
    ON CONFLICT (string_id) DO UPDATE SET
        norm_str = EXCLUDED.norm_str;
    """
    await _execute_on_conn(conn, query, (deduped_string_ids, deduped_norm_strings))


async def batch_upsert_character_strings(
    string_ids: list[int],
    norm_strings: list[str],
    conn=None,
) -> None:
    """
    Batch upsert character string rows in lex.character_strings.

    Args:
        string_ids: String IDs aligned one-to-one with ``norm_strings``.
        norm_strings: Normalized character strings aligned one-to-one with ``string_ids``.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    if not string_ids:
        return
    if len(string_ids) != len(norm_strings):
        raise ValueError("Character string upsert failed: string_ids and norm_strings lengths differ.")
    unique_pairs = list(dict.fromkeys(zip(string_ids, norm_strings)))
    deduped_string_ids = [pair[0] for pair in unique_pairs]
    deduped_norm_strings = [pair[1] for pair in unique_pairs]

    query = """
    INSERT INTO lex.character_strings (string_id, norm_str)
    SELECT unnest(%s::bigint[]), unnest(%s::text[])
    ON CONFLICT (string_id) DO UPDATE SET
        norm_str = EXCLUDED.norm_str;
    """
    await _execute_on_conn(conn, query, (deduped_string_ids, deduped_norm_strings))


async def batch_insert_title_token_postings(term_ids: list[int], movie_id: int, conn=None) -> None:
    """
    Insert title-token postings for one movie in a single round-trip.

    Args:
        term_ids: Title-token term IDs to insert.
        movie_id: Movie ID that owns all postings.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    if not term_ids:
        return
    query = """
    INSERT INTO lex.inv_title_token_postings (term_id, movie_id)
    SELECT unnest(%s::bigint[]), %s
    ON CONFLICT (term_id, movie_id) DO NOTHING;
    """
    await _execute_on_conn(conn, query, (term_ids, movie_id))


async def batch_insert_person_postings(term_ids: list[int], movie_id: int, conn=None) -> None:
    """
    Insert person postings for one movie in a single round-trip.

    Args:
        term_ids: Person term IDs to insert.
        movie_id: Movie ID that owns all postings.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    if not term_ids:
        return
    query = """
    INSERT INTO lex.inv_person_postings (term_id, movie_id)
    SELECT unnest(%s::bigint[]), %s
    ON CONFLICT (term_id, movie_id) DO NOTHING;
    """
    await _execute_on_conn(conn, query, (term_ids, movie_id))


async def batch_insert_character_postings(term_ids: list[int], movie_id: int, conn=None) -> None:
    """
    Insert character postings for one movie in a single round-trip.

    Args:
        term_ids: Character term IDs to insert.
        movie_id: Movie ID that owns all postings.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    if not term_ids:
        return
    query = """
    INSERT INTO lex.inv_character_postings (term_id, movie_id)
    SELECT unnest(%s::bigint[]), %s
    ON CONFLICT (term_id, movie_id) DO NOTHING;
    """
    await _execute_on_conn(conn, query, (term_ids, movie_id))


async def batch_insert_studio_postings(term_ids: list[int], movie_id: int, conn=None) -> None:
    """
    Insert studio postings for one movie in a single round-trip.

    Args:
        term_ids: Studio term IDs to insert.
        movie_id: Movie ID that owns all postings.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    if not term_ids:
        return
    query = """
    INSERT INTO lex.inv_studio_postings (term_id, movie_id)
    SELECT unnest(%s::bigint[]), %s
    ON CONFLICT (term_id, movie_id) DO NOTHING;
    """
    await _execute_on_conn(conn, query, (term_ids, movie_id))


async def upsert_genre_dictionary(genre_id: int, name: str, conn=None) -> None:
    """
    Upsert a genre lookup row in lex.genre_dictionary.
    
    Args:
        genre_id: The genre ID.
        name: The genre name.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    query = """
    with name_existence_check as (
        SELECT 1
        FROM lex.genre_dictionary
        WHERE name = %s
    )
    INSERT INTO lex.genre_dictionary (genre_id, name)
    SELECT %s, %s
    WHERE NOT EXISTS (SELECT 1 FROM name_existence_check)
    ON CONFLICT DO NOTHING;
    """
    await _execute_on_conn(conn, query, (name, genre_id, name))


async def upsert_provider_dictionary(provider_id: int, name: str, conn=None) -> None:
    """
    Upsert a provider lookup row in lex.provider_dictionary.
    
    Args:
        provider_id: The provider ID.
        name: The provider name.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    query = """
    INSERT INTO lex.provider_dictionary (provider_id, name)
    VALUES (%s, %s)
    ON CONFLICT (provider_id) DO NOTHING;
    """
    await _execute_on_conn(conn, query, (provider_id, name))


async def upsert_watch_method_dictionary(method_id: int, name: str, conn=None) -> None:
    """
    Upsert a watch-method lookup row in lex.watch_method_dictionary.
    
    Args:
        method_id: The watch method ID.
        name: The watch method name.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    query = """
    WITH name_existence_check AS (
        SELECT 1
        FROM lex.watch_method_dictionary
        WHERE name = %s
    )
    INSERT INTO lex.watch_method_dictionary (method_id, name)
    SELECT %s, %s
    WHERE NOT EXISTS (SELECT 1 FROM name_existence_check)
    ON CONFLICT DO NOTHING;
    """
    await _execute_on_conn(conn, query, (name, method_id, name))


async def upsert_maturity_dictionary(maturity_rank: int, label: str, conn=None) -> None:
    """
    Upsert a maturity-rating lookup row in lex.maturity_dictionary.
    
    Args:
        maturity_rank: The maturity rank.
        label: The maturity label (e.g., "PG-13", "R").
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    query = """
    WITH label_existence_check AS (
        SELECT 1
        FROM lex.maturity_dictionary
        WHERE label = %s
    )
    INSERT INTO lex.maturity_dictionary (maturity_rank, label)
    SELECT %s, %s
    WHERE NOT EXISTS (SELECT 1 FROM label_existence_check)
    ON CONFLICT DO NOTHING;
    """
    await _execute_on_conn(conn, query, (label, maturity_rank, label))


async def upsert_language_dictionary(language_id: int, name: str, conn=None) -> None:
    """
    Upsert a language lookup row in lex.language_dictionary.
    
    Args:
        language_id: The language ID.
        name: The language name.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    query = """
    WITH name_existence_check AS (
        SELECT 1
        FROM lex.language_dictionary
        WHERE name = %s
    )
    INSERT INTO lex.language_dictionary (language_id, name)
    SELECT %s, %s
    WHERE NOT EXISTS (SELECT 1 FROM name_existence_check)
    ON CONFLICT DO NOTHING;
    """
    await _execute_on_conn(conn, query, (name, language_id, name))


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
    conn=None,
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
        conn: Optional existing async connection for caller-managed transaction scope.
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
    await _execute_on_conn(conn, query, params)

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


def _build_compound_phrase_bucket_cte(
    *,
    cte_name: str,
    table_name: str,
    term_ids: list[int],
    use_eligible: bool,
    exclude_movie_ids: Optional[set[int]] = None,
) -> tuple[str, list]:
    """
    Build one phrase-bucket CTE for the compound lexical query.

    Args:
        cte_name: Name for the generated CTE block.
        table_name: Fully-qualified posting table name.
        term_ids: Resolved term IDs for this bucket.
        use_eligible: Whether the eligible CTE exists in the query.
        exclude_movie_ids: Optional IDs excluded from results.

    Returns:
        Tuple of (cte_sql, params) for this bucket.
    """
    params: list = [term_ids]
    eligibility_join = (
        "\n            JOIN eligible e ON e.movie_id = p.movie_id"
        if use_eligible
        else ""
    )
    exclusion_clause = ""
    if exclude_movie_ids:
        exclusion_clause = "\n              AND NOT (p.movie_id = ANY(%s::bigint[]))"
        params.append(list(exclude_movie_ids))

    cte_sql = f"""{cte_name} AS (
            SELECT p.movie_id, COUNT(DISTINCT p.term_id)::int AS matched
            FROM {table_name} p{eligibility_join}
            WHERE p.term_id = ANY(%s::bigint[]){exclusion_clause}
            GROUP BY p.movie_id
        )"""
    return cte_sql, params


def _build_compound_character_cte(
    *,
    query_idxs: list[int],
    term_ids: list[int],
    use_eligible: bool,
    exclude_movie_ids: Optional[set[int]] = None,
) -> tuple[list[str], list]:
    """
    Build character CTE blocks for the compound lexical query.

    The output always keeps per-query granularity by grouping on
    ``(query_idx, movie_id)``.

    Args:
        query_idxs: Query indices aligned with term_ids.
        term_ids: Resolved character term IDs aligned with query_idxs.
        use_eligible: Whether the eligible CTE exists in the query.
        exclude_movie_ids: Optional IDs excluded from results.

    Returns:
        Tuple of (cte_sql_parts, params).
    """
    params: list = [query_idxs, term_ids]
    eligibility_join = (
        "\n            JOIN eligible e ON e.movie_id = p.movie_id"
        if use_eligible
        else ""
    )
    exclusion_clause = ""
    if exclude_movie_ids:
        exclusion_clause = "\n            WHERE NOT (p.movie_id = ANY(%s::bigint[]))"
        params.append(list(exclude_movie_ids))

    cte_parts = [
        """q_chars AS (
            SELECT unnest(%s::int[]) AS query_idx,
                   unnest(%s::bigint[]) AS term_id
        )""",
        f"""character_matches AS (
            SELECT
                qc.query_idx,
                p.movie_id,
                COUNT(DISTINCT qc.query_idx)::int AS matched
            FROM q_chars qc
            JOIN lex.inv_character_postings p
              ON p.term_id = qc.term_id{eligibility_join}{exclusion_clause}
            GROUP BY qc.query_idx, p.movie_id
        )""",
    ]
    return cte_parts, params


def _build_compound_title_ctes(
    *,
    search_idx: int,
    title_search: TitleSearchInput,
    use_eligible: bool,
    exclude_movie_ids: Optional[set[int]] = None,
) -> tuple[list[str], str, list, list]:
    """
    Build CTE blocks and UNION branch for one title search.

    Args:
        search_idx: Zero-based title search index.
        title_search: One title-search payload.
        use_eligible: Whether the eligible CTE exists in the query.
        exclude_movie_ids: Optional IDs excluded from results.

    Returns:
        Tuple of (cte_sql_parts, union_sql, cte_params, union_params).
    """
    token_cte_name = f"q_tokens_{search_idx}"
    token_matches_cte_name = f"token_matches_{search_idx}"
    title_matches_cte_name = f"title_matches_{search_idx}"
    title_scored_cte_name = f"title_scored_{search_idx}"
    title_ranked_cte_name = f"title_ranked_{search_idx}"

    cte_params: list = [title_search.token_idxs, title_search.term_ids]
    eligibility_join = (
        "\n                JOIN eligible e ON e.movie_id = p.movie_id"
        if use_eligible
        else ""
    )
    exclusion_clause = ""
    if exclude_movie_ids:
        exclusion_clause = "\n            WHERE NOT (p.movie_id = ANY(%s::bigint[]))"
        cte_params.append(list(exclude_movie_ids))

    title_count_source = "eligible" if use_eligible else "public.movie_card"
    title_count_alias = "e" if use_eligible else "mc"

    cte_params.extend(
        [
            title_search.f_coeff,
            title_search.k,
            title_search.beta_sq,
            title_search.k,
            title_search.k,
            title_search.score_threshold,
        ]
    )

    cte_parts = [
        f"""{token_cte_name} AS (
            SELECT unnest(%s::int[]) AS token_idx,
                   unnest(%s::bigint[]) AS term_id
        )""",
        f"""{token_matches_cte_name} AS (
            SELECT DISTINCT p.movie_id, qt.token_idx
            FROM {token_cte_name} qt
            JOIN lex.inv_title_token_postings p
              ON p.term_id = qt.term_id{eligibility_join}{exclusion_clause}
        )""",
        f"""{title_matches_cte_name} AS (
            SELECT movie_id, COUNT(*)::int AS m
            FROM {token_matches_cte_name}
            GROUP BY movie_id
        )""",
        f"""{title_scored_cte_name} AS (
            SELECT
                tm.movie_id,
                (%s::double precision
                    * ((tm.m::double precision / %s)
                       * (tm.m::double precision / {title_count_alias}.title_token_count)))
                / (%s::double precision
                    * (tm.m::double precision / {title_count_alias}.title_token_count)
                    + (tm.m::double precision / %s))
                AS title_score
            FROM {title_matches_cte_name} tm
            JOIN {title_count_source} {title_count_alias}
              ON {title_count_alias}.movie_id = tm.movie_id
            WHERE {title_count_alias}.title_token_count > 0
              AND %s > 0
        )""",
        f"""{title_ranked_cte_name} AS (
            SELECT
                movie_id,
                title_score,
                ROW_NUMBER() OVER (ORDER BY title_score DESC) AS rn
            FROM {title_scored_cte_name}
            WHERE title_score >= %s
        )""",
    ]
    union_sql = (
        f"SELECT 'title_{search_idx}' AS bucket, -1 AS query_idx, movie_id, title_score AS score "
        f"FROM {title_ranked_cte_name} WHERE rn <= %s"
    )
    union_params = [title_search.max_candidates]
    return cte_parts, union_sql, cte_params, union_params


async def execute_compound_lexical_search(
    *,
    people_term_ids: list[int],
    studio_term_ids: list[int],
    character_query_idxs: list[int],
    character_term_ids: list[int],
    title_searches: list[TitleSearchInput],
    filters: Optional[MetadataFilters] = None,
    exclude_movie_ids: Optional[set[int]] = None,
) -> CompoundLexicalResult:
    """
    Execute one compound lexical query across all included lexical buckets.

    This query deduplicates the eligible-set materialization by building one
    SQL statement with optional bucket CTEs and a tagged UNION ALL result set.

    Args:
        people_term_ids: Resolved INCLUDE people term IDs.
        studio_term_ids: Resolved INCLUDE studio term IDs.
        character_query_idxs: Character query indices aligned with character_term_ids.
        character_term_ids: Resolved character term IDs aligned with query indices.
        title_searches: Title-space searches to score in this request.
        filters: Optional metadata hard-filters.
        exclude_movie_ids: Optional movie IDs to exclude from all buckets.

    Returns:
        CompoundLexicalResult with per-bucket parsed maps.
    """
    has_any_title = any(search.term_ids for search in title_searches)
    has_any_bucket = bool(
        people_term_ids
        or studio_term_ids
        or character_term_ids
        or has_any_title
    )
    if not has_any_bucket:
        return CompoundLexicalResult(
            people_scores={},
            studio_scores={},
            character_by_query={},
            title_scores_by_search={},
        )

    if len(character_query_idxs) != len(character_term_ids):
        raise ValueError("character_query_idxs and character_term_ids must be same length")

    use_eligible = filters is not None and filters.is_active
    cte_parts: list[str] = []
    union_parts: list[str] = []
    cte_params: list = []
    union_params: list = []

    if use_eligible:
        eligible_cte, eligible_params = await _build_eligible_cte(filters)
        cte_parts.append(eligible_cte)
        cte_params.extend(eligible_params)

    if people_term_ids:
        people_cte, people_params = _build_compound_phrase_bucket_cte(
            cte_name="people_matches",
            table_name=PostingTable.PERSON.value,
            term_ids=people_term_ids,
            use_eligible=use_eligible,
            exclude_movie_ids=exclude_movie_ids,
        )
        cte_parts.append(people_cte)
        cte_params.extend(people_params)
        union_parts.append(
            "SELECT 'people' AS bucket, -1 AS query_idx, movie_id, matched::double precision AS score "
            "FROM people_matches"
        )

    if studio_term_ids:
        studio_cte, studio_params = _build_compound_phrase_bucket_cte(
            cte_name="studio_matches",
            table_name=PostingTable.STUDIO.value,
            term_ids=studio_term_ids,
            use_eligible=use_eligible,
            exclude_movie_ids=exclude_movie_ids,
        )
        cte_parts.append(studio_cte)
        cte_params.extend(studio_params)
        union_parts.append(
            "SELECT 'studio' AS bucket, -1 AS query_idx, movie_id, matched::double precision AS score "
            "FROM studio_matches"
        )

    if character_term_ids:
        character_ctes, character_params = _build_compound_character_cte(
            query_idxs=character_query_idxs,
            term_ids=character_term_ids,
            use_eligible=use_eligible,
            exclude_movie_ids=exclude_movie_ids,
        )
        cte_parts.extend(character_ctes)
        cte_params.extend(character_params)
        union_parts.append(
            "SELECT 'character' AS bucket, query_idx, movie_id, matched::double precision AS score "
            "FROM character_matches"
        )

    for idx, title_search in enumerate(title_searches):
        if not title_search.term_ids:
            continue
        title_ctes, title_union, title_cte_params, title_union_params = _build_compound_title_ctes(
            search_idx=idx,
            title_search=title_search,
            use_eligible=use_eligible,
            exclude_movie_ids=exclude_movie_ids,
        )
        cte_parts.extend(title_ctes)
        union_parts.append(title_union)
        cte_params.extend(title_cte_params)
        union_params.extend(title_union_params)

    if not union_parts:
        return CompoundLexicalResult(
            people_scores={},
            studio_scores={},
            character_by_query={},
            title_scores_by_search={},
        )

    with_clause = "WITH " + ",\n        ".join(cte_parts)
    query = f"""
        {with_clause}
        {" UNION ALL ".join(union_parts)}
    """
    params = cte_params + union_params
    rows = await _execute_read(query, params)

    people_scores: dict[int, int] = {}
    studio_scores: dict[int, int] = {}
    character_by_query: dict[int, dict[int, int]] = {}
    title_scores_by_search: dict[int, dict[int, float]] = {}

    for bucket, query_idx, movie_id, score in rows:
        if bucket == "people":
            people_scores[movie_id] = int(score)
            continue
        if bucket == "studio":
            studio_scores[movie_id] = int(score)
            continue
        if bucket == "character":
            character_by_query.setdefault(int(query_idx), {})[movie_id] = int(score)
            continue
        if bucket.startswith("title_"):
            search_idx = int(bucket.split("_")[1])
            title_scores_by_search.setdefault(search_idx, {})[movie_id] = float(score)

    return CompoundLexicalResult(
        people_scores=people_scores,
        studio_scores=studio_scores,
        character_by_query=character_by_query,
        title_scores_by_search=title_scores_by_search,
    )


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