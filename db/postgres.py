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
from implementation.misc.sql_like import escape_like
from implementation.classes.schemas import MetadataFilters
from schemas.award_category_tags import tags_for_category
from schemas.enums import AwardCeremony
from schemas.imdb_models import AwardNomination
from schemas.metadata import FranchiseOutput


class PostingTable(Enum):
    """Supported posting tables for lexical matching and exclusion resolution."""
    ACTOR = "lex.inv_actor_postings"
    DIRECTOR = "lex.inv_director_postings"
    WRITER = "lex.inv_writer_postings"
    PRODUCER = "lex.inv_producer_postings"
    COMPOSER = "lex.inv_composer_postings"
    CHARACTER = "lex.inv_character_postings"
    TITLE_TOKEN = "lex.inv_title_token_postings"


# All role-specific people posting tables, used by search code that needs
# to union across roles (compound lexical search, exclusion resolution).
PEOPLE_POSTING_TABLES: list[PostingTable] = [
    PostingTable.ACTOR,
    PostingTable.DIRECTOR,
    PostingTable.WRITER,
    PostingTable.PRODUCER,
    PostingTable.COMPOSER,
]


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
        character_by_query: Character matched counts keyed by query_idx then movie_id.
        title_scores_by_search: Title scores keyed by title search index then movie_id.
    """
    people_scores: dict[int, int]
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

    if filters.audio_languages is not None:
        audio_language_ids = [language.language_id for language in filters.audio_languages]
        if audio_language_ids:
            conditions.append("audio_language_ids && %s::int[]")
            params.append(audio_language_ids)

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


async def batch_insert_actor_postings(
    term_ids: list[int], movie_id: int, cast_size: int, conn=None,
) -> None:
    """
    Insert actor postings with billing metadata for one movie.

    term_ids must be in billing order (lead actor first). billing_position
    is auto-generated as 1-based index from the list order.

    Args:
        term_ids: Actor term IDs in billing order.
        movie_id: Movie ID that owns all postings.
        cast_size: Total number of credited cast members in the film.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    if not term_ids:
        return
    # Parallel unnest: term_ids[i] pairs with billing_positions[i].
    # cast_size is a scalar that broadcasts across all rows.
    billing_positions = list(range(1, len(term_ids) + 1))
    query = """
    INSERT INTO lex.inv_actor_postings (term_id, movie_id, billing_position, cast_size)
    SELECT unnest(%s::bigint[]), %s, unnest(%s::int[]), %s
    ON CONFLICT (term_id, movie_id) DO NOTHING;
    """
    await _execute_on_conn(conn, query, (term_ids, movie_id, billing_positions, cast_size))


async def batch_insert_director_postings(term_ids: list[int], movie_id: int, conn=None) -> None:
    """
    Insert director postings for one movie in a single round-trip.

    Args:
        term_ids: Director term IDs to insert.
        movie_id: Movie ID that owns all postings.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    if not term_ids:
        return
    query = """
    INSERT INTO lex.inv_director_postings (term_id, movie_id)
    SELECT unnest(%s::bigint[]), %s
    ON CONFLICT (term_id, movie_id) DO NOTHING;
    """
    await _execute_on_conn(conn, query, (term_ids, movie_id))


async def batch_insert_writer_postings(term_ids: list[int], movie_id: int, conn=None) -> None:
    """
    Insert writer postings for one movie in a single round-trip.

    Args:
        term_ids: Writer term IDs to insert.
        movie_id: Movie ID that owns all postings.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    if not term_ids:
        return
    query = """
    INSERT INTO lex.inv_writer_postings (term_id, movie_id)
    SELECT unnest(%s::bigint[]), %s
    ON CONFLICT (term_id, movie_id) DO NOTHING;
    """
    await _execute_on_conn(conn, query, (term_ids, movie_id))


async def batch_insert_producer_postings(term_ids: list[int], movie_id: int, conn=None) -> None:
    """
    Insert producer postings for one movie in a single round-trip.

    Args:
        term_ids: Producer term IDs to insert.
        movie_id: Movie ID that owns all postings.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    if not term_ids:
        return
    query = """
    INSERT INTO lex.inv_producer_postings (term_id, movie_id)
    SELECT unnest(%s::bigint[]), %s
    ON CONFLICT (term_id, movie_id) DO NOTHING;
    """
    await _execute_on_conn(conn, query, (term_ids, movie_id))


async def batch_insert_composer_postings(term_ids: list[int], movie_id: int, conn=None) -> None:
    """
    Insert composer postings for one movie in a single round-trip.

    Args:
        term_ids: Composer term IDs to insert.
        movie_id: Movie ID that owns all postings.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    if not term_ids:
        return
    query = """
    INSERT INTO lex.inv_composer_postings (term_id, movie_id)
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


async def batch_upsert_production_companies(
    pairs: Sequence[tuple[str, str]],
    conn=None,
) -> dict[str, int]:
    """
    Upsert normalized production-company strings into lex.production_company.

    Each input pair is (canonical_string, normalized_string). The normalized
    form is the unique key; the canonical form is stored for display only
    and only applied on first insert (existing rows keep whatever canonical
    they had).

    Dedup-and-sort by normalized_string before issuing the query so that all
    concurrent transactions acquire the UNIQUE-index locks in the same order
    — same deadlock-avoidance pattern used by batch_upsert_lexical_dictionary.

    Returns a ``{normalized_string: production_company_id}`` mapping covering
    every unique normalized string in the input.
    """
    if not pairs:
        return {}

    # Collapse duplicates by normalized_string; the first canonical form wins
    # (insertion-order dict dedup). Then sort lexicographically by normalized
    # to keep lock-acquisition order deterministic across concurrent writers.
    unique_by_norm: dict[str, str] = {}
    for canonical, normalized in pairs:
        if normalized and normalized not in unique_by_norm:
            unique_by_norm[normalized] = canonical
    if not unique_by_norm:
        return {}
    norm_list = sorted(unique_by_norm.keys())
    canonical_list = [unique_by_norm[n] for n in norm_list]

    # Insert any new rows; then re-select all requested normalized strings
    # (whether newly inserted or pre-existing) so every caller input maps to
    # an id. Mirrors the CTE pattern in batch_upsert_lexical_dictionary.
    query = """
    WITH input_rows AS (
        SELECT unnest(%s::text[]) AS canonical_string,
               unnest(%s::text[]) AS normalized_string
    ),
    inserted AS (
        INSERT INTO lex.production_company (canonical_string, normalized_string)
        SELECT canonical_string, normalized_string FROM input_rows
        ON CONFLICT (normalized_string) DO NOTHING
        RETURNING production_company_id, normalized_string
    )
    SELECT normalized_string, production_company_id FROM inserted
    UNION ALL
    SELECT p.normalized_string, p.production_company_id
    FROM lex.production_company p
    JOIN input_rows i ON i.normalized_string = p.normalized_string
    WHERE NOT EXISTS (
        SELECT 1 FROM inserted ins
        WHERE ins.normalized_string = p.normalized_string
    );
    """
    rows = await _execute_on_conn(conn, query, (canonical_list, norm_list), fetch=True) or []
    return {str(norm): int(pid) for norm, pid in rows}


async def batch_insert_studio_tokens(
    pairs: Sequence[tuple[str, int]],
    conn=None,
) -> None:
    """
    Insert (token, production_company_id) rows into lex.studio_token.

    Idempotent via ``ON CONFLICT DO NOTHING``. Every unique pair is inserted
    once regardless of how many times it appears in the input; callers are
    free to pass duplicates.
    """
    if not pairs:
        return
    # Deduplicate in Python to keep the unnest arrays small on wide inputs.
    unique = list({(token, cid) for token, cid in pairs if token})
    if not unique:
        return
    tokens = [t for t, _ in unique]
    company_ids = [c for _, c in unique]
    query = """
    INSERT INTO lex.studio_token (token, production_company_id)
    SELECT unnest(%s::text[]), unnest(%s::bigint[])
    ON CONFLICT (token, production_company_id) DO NOTHING;
    """
    await _execute_on_conn(conn, query, (tokens, company_ids))


async def batch_insert_brand_postings(
    movie_id: int,
    rows: Sequence[tuple[int, int]],
    conn=None,
) -> None:
    """
    Replace all brand postings for one movie with the provided rows.

    Each row is ``(brand_id, first_matching_index)``; total_brand_count is
    derived as ``len(rows)`` and stamped on every row so the lexical scorer
    has a per-movie normalizer available without a second query.

    Uses delete-then-insert inside the caller's transaction, mirroring how
    batch_upsert_movie_awards handles its "always a complete set" semantics.
    This keeps brand membership consistent when a movie is re-ingested with
    a different production_companies list or when the brand registry
    changes.
    """
    # Always clear existing postings first so stale brand rows never linger
    # after a registry change.
    delete_query = "DELETE FROM lex.inv_production_brand_postings WHERE movie_id = %s"
    await _execute_on_conn(conn, delete_query, (movie_id,))

    if not rows:
        return

    brand_ids = [bid for bid, _ in rows]
    first_indices = [idx for _, idx in rows]
    total = len(rows)
    # DELETE just above cleared all rows for this movie, so no ON CONFLICT
    # clause is needed — every (brand_id, movie_id) we insert is guaranteed
    # unique within the statement because `batch_insert_brand_postings`
    # callers dedup by brand_id before passing rows (resolve_brands_for_movie
    # returns one BrandTag per brand). Matches the delete-then-insert
    # semantics of batch_upsert_movie_awards.
    query = """
    INSERT INTO lex.inv_production_brand_postings
        (brand_id, movie_id, first_matching_index, total_brand_count)
    SELECT unnest(%s::smallint[]), %s, unnest(%s::smallint[]), %s;
    """
    await _execute_on_conn(conn, query, (brand_ids, movie_id, first_indices, total))


async def update_movie_card_production_company_ids(
    movie_id: int,
    production_company_ids: Sequence[int],
    conn=None,
) -> None:
    """
    Stamp the freeform-path company-id array on movie_card.

    Exists as a standalone helper so the backfill script can set the column
    without rewriting every movie_card field. Ingest-time callers pass the
    list directly to ``upsert_movie_card`` and do not need this function.

    Raises ``ValueError`` if no movie_card row exists for ``movie_id``.
    Without this guard the UPDATE would silently no-op, hiding upstream
    ordering bugs where the card hasn't been written yet.
    """
    query = """
    UPDATE public.movie_card
    SET production_company_ids = %s,
        updated_at = now()
    WHERE movie_id = %s
    """
    params = (list(production_company_ids), movie_id)

    # _execute_on_conn doesn't surface rowcount, so run the cursor directly
    # here. The branching mirrors that helper: honor a caller-supplied conn
    # for transaction scope, or acquire+commit one locally when None.
    if conn is not None:
        async with conn.cursor() as cur:
            await cur.execute(query, params)
            rowcount = cur.rowcount
    else:
        async with pool.connection() as fallback_conn:
            async with fallback_conn.cursor() as cur:
                await cur.execute(query, params)
                rowcount = cur.rowcount
            await fallback_conn.commit()

    if rowcount == 0:
        raise ValueError(
            f"update_movie_card_production_company_ids: no movie_card row "
            f"for movie_id={movie_id}. The card must be upserted before the "
            f"production_company_ids column can be stamped."
        )


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


async def refresh_studio_token_doc_frequency() -> None:
    """
    Refresh the lex.studio_token_doc_frequency materialized view concurrently.

    Called after each bulk ingest so DF-ceiling stop-word filtering on the
    freeform studio path reflects the latest (token, production_company_id)
    rows. CONCURRENTLY avoids blocking reads during rebuild (requires the
    unique index idx_studio_token_df_token on the view).
    """
    await _execute_write(
        "REFRESH MATERIALIZED VIEW CONCURRENTLY lex.studio_token_doc_frequency;"
    )


async def refresh_franchise_token_doc_frequency() -> None:
    """
    Refresh the lex.franchise_token_doc_frequency materialized view concurrently.

    Called after each bulk ingest so DF-ceiling stop-word filtering on the
    franchise path reflects the latest (token, franchise_entry_id) rows.
    CONCURRENTLY avoids blocking reads during rebuild (requires the unique
    index idx_franchise_token_df_token on the view).
    """
    await _execute_write(
        "REFRESH MATERIALIZED VIEW CONCURRENTLY lex.franchise_token_doc_frequency;"
    )


async def refresh_award_name_token_doc_frequency() -> None:
    """
    Refresh the lex.award_name_token_doc_frequency materialized view concurrently.

    Called after each bulk ingest so DF-ceiling stop-word filtering on the
    award-name path reflects the latest (token, award_name_entry_id) rows.
    CONCURRENTLY avoids blocking reads during rebuild (requires the unique
    index idx_award_name_token_df_token on the view).
    """
    await _execute_write(
        "REFRESH MATERIALIZED VIEW CONCURRENTLY lex.award_name_token_doc_frequency;"
    )


async def batch_upsert_award_name_entries(
    normalized_strings: Sequence[str],
    conn=None,
) -> dict[str, int]:
    """
    Upsert normalized award-name strings into lex.award_name_entry.

    Unlike the studio and franchise entry tables, this one has no
    ``canonical_string`` column — raw surface forms live on
    ``public.movie_awards.award_name`` already, so the entry row carries
    only the normalized lookup key. Callers pass a flat sequence of
    normalized strings (empty strings are dropped).

    Dedup-and-sort before issuing the query so that all concurrent
    transactions acquire the UNIQUE-index locks in the same order — same
    deadlock-avoidance pattern used by batch_upsert_production_companies
    and batch_upsert_franchise_entries.

    Returns a ``{normalized: award_name_entry_id}`` mapping covering every
    unique normalized string in the input.
    """
    if not normalized_strings:
        return {}

    # Drop empties and dedup-preserving-first-occurrence via set; then sort
    # for deterministic lock ordering.
    unique = sorted({n for n in normalized_strings if n})
    if not unique:
        return {}

    # Insert any new rows; then re-select all requested normalized strings
    # (whether newly inserted or pre-existing) so every caller input maps to
    # an id. Same CTE pattern as batch_upsert_production_companies /
    # batch_upsert_franchise_entries.
    query = """
    WITH input_rows AS (
        SELECT unnest(%s::text[]) AS normalized
    ),
    inserted AS (
        INSERT INTO lex.award_name_entry (normalized)
        SELECT normalized FROM input_rows
        ON CONFLICT (normalized) DO NOTHING
        RETURNING award_name_entry_id, normalized
    )
    SELECT normalized, award_name_entry_id FROM inserted
    UNION ALL
    SELECT e.normalized, e.award_name_entry_id
    FROM lex.award_name_entry e
    JOIN input_rows i ON i.normalized = e.normalized
    WHERE NOT EXISTS (
        SELECT 1 FROM inserted ins
        WHERE ins.normalized = e.normalized
    );
    """
    rows = await _execute_on_conn(conn, query, (unique,), fetch=True) or []
    return {str(norm): int(eid) for norm, eid in rows}


async def batch_insert_award_name_tokens(
    pairs: Sequence[tuple[str, int]],
    conn=None,
) -> None:
    """
    Insert (token, award_name_entry_id) rows into lex.award_name_token.

    Idempotent via ``ON CONFLICT DO NOTHING``. Every unique pair is inserted
    once regardless of how many times it appears in the input; callers are
    free to pass duplicates.
    """
    if not pairs:
        return
    unique = list({(token, eid) for token, eid in pairs if token})
    if not unique:
        return
    tokens = [t for t, _ in unique]
    entry_ids = [e for _, e in unique]
    # award_name_entry_id is INT (see 01_create_postgres_tables.sql —
    # award volume is small enough that INT suffices), so cast to int[]
    # here rather than the bigint[] used by studio/franchise.
    query = """
    INSERT INTO lex.award_name_token (token, award_name_entry_id)
    SELECT unnest(%s::text[]), unnest(%s::int[])
    ON CONFLICT (token, award_name_entry_id) DO NOTHING;
    """
    await _execute_on_conn(conn, query, (tokens, entry_ids))


async def batch_upsert_franchise_entries(
    pairs: Sequence[tuple[str, str]],
    conn=None,
) -> dict[str, int]:
    """
    Upsert normalized franchise strings into lex.franchise_entry.

    Each input pair is (canonical_string, normalized_string), covering
    lineage, shared_universe, and every element of recognized_subgroups.
    The normalized form is the unique key; the canonical form is stored for
    display only and only applied on first insert (existing rows keep
    whatever canonical they had).

    Dedup-and-sort by normalized_string before issuing the query so that all
    concurrent transactions acquire the UNIQUE-index locks in the same order
    — same deadlock-avoidance pattern used by batch_upsert_production_companies.

    Returns a ``{normalized_string: franchise_entry_id}`` mapping covering
    every unique normalized string in the input.
    """
    if not pairs:
        return {}

    unique_by_norm: dict[str, str] = {}
    for canonical, normalized in pairs:
        if normalized and normalized not in unique_by_norm:
            unique_by_norm[normalized] = canonical
    if not unique_by_norm:
        return {}
    norm_list = sorted(unique_by_norm.keys())
    canonical_list = [unique_by_norm[n] for n in norm_list]

    # Insert any new rows; then re-select all requested normalized strings
    # so every caller input maps to an id. Mirrors the CTE pattern in
    # batch_upsert_production_companies.
    query = """
    WITH input_rows AS (
        SELECT unnest(%s::text[]) AS canonical_string,
               unnest(%s::text[]) AS normalized_string
    ),
    inserted AS (
        INSERT INTO lex.franchise_entry (canonical_string, normalized_string)
        SELECT canonical_string, normalized_string FROM input_rows
        ON CONFLICT (normalized_string) DO NOTHING
        RETURNING franchise_entry_id, normalized_string
    )
    SELECT normalized_string, franchise_entry_id FROM inserted
    UNION ALL
    SELECT f.normalized_string, f.franchise_entry_id
    FROM lex.franchise_entry f
    JOIN input_rows i ON i.normalized_string = f.normalized_string
    WHERE NOT EXISTS (
        SELECT 1 FROM inserted ins
        WHERE ins.normalized_string = f.normalized_string
    );
    """
    rows = await _execute_on_conn(conn, query, (canonical_list, norm_list), fetch=True) or []
    return {str(norm): int(fid) for norm, fid in rows}


async def batch_insert_franchise_tokens(
    pairs: Sequence[tuple[str, int]],
    conn=None,
) -> None:
    """
    Insert (token, franchise_entry_id) rows into lex.franchise_token.

    Idempotent via ``ON CONFLICT DO NOTHING``. Every unique pair is inserted
    once regardless of how many times it appears in the input; callers are
    free to pass duplicates.
    """
    if not pairs:
        return
    unique = list({(token, fid) for token, fid in pairs if token})
    if not unique:
        return
    tokens = [t for t, _ in unique]
    entry_ids = [f for _, f in unique]
    query = """
    INSERT INTO lex.franchise_token (token, franchise_entry_id)
    SELECT unnest(%s::text[]), unnest(%s::bigint[])
    ON CONFLICT (token, franchise_entry_id) DO NOTHING;
    """
    await _execute_on_conn(conn, query, (tokens, entry_ids))


async def update_movie_card_franchise_ids(
    movie_id: int,
    franchise_name_entry_ids: Sequence[int],
    subgroup_entry_ids: Sequence[int],
    conn=None,
) -> None:
    """
    Stamp the franchise entry-id arrays on movie_card.

    Exists as a standalone helper so the backfill script can set both
    columns without rewriting every movie_card field. Ingest-time callers
    pass the arrays directly to ``upsert_movie_card`` and do not need this
    function.

    Raises ``ValueError`` if no movie_card row exists for ``movie_id``.
    Without this guard the UPDATE would silently no-op, hiding upstream
    ordering bugs where the card hasn't been written yet.
    """
    query = """
    UPDATE public.movie_card
    SET franchise_name_entry_ids = %s,
        subgroup_entry_ids = %s,
        updated_at = now()
    WHERE movie_id = %s
    """
    params = (
        list(franchise_name_entry_ids),
        list(subgroup_entry_ids),
        movie_id,
    )

    # _execute_on_conn doesn't surface rowcount, so run the cursor directly
    # here. The branching mirrors update_movie_card_production_company_ids.
    if conn is not None:
        async with conn.cursor() as cur:
            await cur.execute(query, params)
            rowcount = cur.rowcount
    else:
        async with pool.connection() as fallback_conn:
            async with fallback_conn.cursor() as cur:
                await cur.execute(query, params)
                rowcount = cur.rowcount
            await fallback_conn.commit()

    if rowcount == 0:
        raise ValueError(
            f"update_movie_card_franchise_ids: no movie_card row for "
            f"movie_id={movie_id}. The card must be upserted before the "
            f"franchise_name_entry_ids / subgroup_entry_ids columns can be stamped."
        )


async def refresh_movie_popularity_scores(
    *,
    threshold: float = 0.70,
    steepness_k: float = 15.0,
    conn=None,
) -> None:
    """
    Recompute and persist ``movie_card.popularity_score`` from IMDb vote count.

    Pipeline (based on guides/popularity_metric_guide.md):
      1. Build/refresh a materialized view of global vote-count percentiles.
      2. Apply sigmoid transform and write scores to ``public.movie_card``.
      3. Zero out rows without vote count.

    Args:
        threshold: Percentile midpoint for sigmoid transition.
        steepness_k: Sigmoid steepness coefficient.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    create_view_query = """
    CREATE MATERIALIZED VIEW IF NOT EXISTS public.mv_popularity_percentile AS
    SELECT
      movie_id,
      PERCENT_RANK() OVER (ORDER BY imdb_vote_count ASC) AS percentile
    FROM public.movie_card
    WHERE imdb_vote_count IS NOT NULL;
    """
    refresh_view_query = "REFRESH MATERIALIZED VIEW public.mv_popularity_percentile;"
    update_scores_query = """
    UPDATE public.movie_card mc
    SET popularity_score = 1.0 / (1.0 + exp(-%s * (mv.percentile - %s)))
    FROM public.mv_popularity_percentile mv
    WHERE mc.movie_id = mv.movie_id;
    """
    zero_missing_query = """
    UPDATE public.movie_card
    SET popularity_score = 0.0
    WHERE imdb_vote_count IS NULL;
    """

    if conn is not None:
        await _execute_on_conn(conn, create_view_query)
        await _execute_on_conn(conn, refresh_view_query)
        await _execute_on_conn(conn, update_scores_query, (steepness_k, threshold))
        await _execute_on_conn(conn, zero_missing_query)
        return

    async with pool.connection() as local_conn:
        await _execute_on_conn(local_conn, create_view_query)
        await _execute_on_conn(local_conn, refresh_view_query)
        await _execute_on_conn(local_conn, update_scores_query, (steepness_k, threshold))
        await _execute_on_conn(local_conn, zero_missing_query)
        await local_conn.commit()


async def upsert_movie_franchise_metadata(
    movie_id: int,
    franchise_metadata: FranchiseOutput,
    conn=None,
) -> None:
    """
    Upsert the structured franchise projection for one movie.

    Args:
        movie_id: Target movie ID.
        franchise_metadata: Parsed FranchiseOutput from tracker metadata.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    query = """
    INSERT INTO public.movie_franchise_metadata (
        movie_id,
        lineage,
        shared_universe,
        recognized_subgroups,
        launched_subgroup,
        lineage_position,
        is_spinoff,
        is_crossover,
        launched_franchise
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (movie_id) DO UPDATE SET
        lineage = EXCLUDED.lineage,
        shared_universe = EXCLUDED.shared_universe,
        recognized_subgroups = EXCLUDED.recognized_subgroups,
        launched_subgroup = EXCLUDED.launched_subgroup,
        lineage_position = EXCLUDED.lineage_position,
        is_spinoff = EXCLUDED.is_spinoff,
        is_crossover = EXCLUDED.is_crossover,
        launched_franchise = EXCLUDED.launched_franchise
    """
    lineage_position = (
        franchise_metadata.lineage_position.lineage_position_id
        if franchise_metadata.lineage_position is not None
        else None
    )
    params = (
        movie_id,
        franchise_metadata.lineage,
        franchise_metadata.shared_universe,
        list(franchise_metadata.recognized_subgroups),
        franchise_metadata.launched_subgroup,
        lineage_position,
        franchise_metadata.is_spinoff,
        franchise_metadata.is_crossover,
        franchise_metadata.launched_franchise,
    )
    await _execute_on_conn(conn, query, params)


async def delete_movie_franchise_metadata(movie_id: int, conn=None) -> None:
    """
    Delete franchise metadata for one movie.

    The franchise_name_entry_ids / subgroup_entry_ids columns on movie_card
    are rewritten by upsert_movie_card on next ingest, so they need no
    separate clear here; and lex.franchise_entry / lex.franchise_token are
    registry-wide (not movie-scoped), so nothing else to delete.

    Args:
        movie_id: Target movie ID.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    delete_metadata_query = "DELETE FROM public.movie_franchise_metadata WHERE movie_id = %s"
    await _execute_on_conn(conn, delete_metadata_query, (movie_id,))


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
    country_of_origin_ids: Sequence[int],
    source_material_type_ids: Sequence[int],
    keyword_ids: Sequence[int],
    concept_tag_ids: Sequence[int],
    award_ceremony_win_ids: Sequence[int],
    imdb_vote_count: int,
    reception_score: Optional[float],
    title_token_count: int,
    budget_bucket: Optional[str] = None,
    box_office_bucket: Optional[str] = None,
    production_company_ids: Sequence[int] = (),
    franchise_name_entry_ids: Sequence[int] = (),
    subgroup_entry_ids: Sequence[int] = (),
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
        country_of_origin_ids: List of country-of-origin IDs.
        source_material_type_ids: List of source material type IDs.
        keyword_ids: List of keyword IDs.
        concept_tag_ids: List of concept tag IDs.
        award_ceremony_win_ids: List of AwardCeremony IDs where this movie won.
        imdb_vote_count: Raw IMDb vote count.
        reception_score: Precomputed reception score from IMDB/Metacritic.
        title_token_count: Number of tokens in the title.
        budget_bucket: Era-adjusted budget classification ('small', 'large', or None for mid-range/unknown).
        box_office_bucket: Box office classification ('hit', 'flop', or None for ambiguous/unknown).
        production_company_ids: lex.production_company IDs this movie credits.
            Empty sequence is allowed (movies with no IMDB production_companies).
        franchise_name_entry_ids: lex.franchise_entry IDs resolved from the
            movie's lineage + shared_universe strings (union, 0-2 elements).
            Empty sequence when the movie has no franchise metadata.
        subgroup_entry_ids: lex.franchise_entry IDs resolved from each element
            of recognized_subgroups. Empty sequence when there are no subgroups.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    query = """
    INSERT INTO public.movie_card (
        movie_id, title, poster_url, release_ts, runtime_minutes,
        maturity_rank, genre_ids, watch_offer_keys, audio_language_ids, country_of_origin_ids,
        source_material_type_ids, keyword_ids, concept_tag_ids, award_ceremony_win_ids,
        imdb_vote_count, reception_score, budget_bucket, box_office_bucket, title_token_count,
        production_company_ids, franchise_name_entry_ids, subgroup_entry_ids,
        created_at, updated_at
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now(), now())
    ON CONFLICT (movie_id) DO UPDATE SET
        title = EXCLUDED.title,
        poster_url = EXCLUDED.poster_url,
        release_ts = EXCLUDED.release_ts,
        runtime_minutes = EXCLUDED.runtime_minutes,
        maturity_rank = EXCLUDED.maturity_rank,
        genre_ids = EXCLUDED.genre_ids,
        watch_offer_keys = EXCLUDED.watch_offer_keys,
        audio_language_ids = EXCLUDED.audio_language_ids,
        country_of_origin_ids = EXCLUDED.country_of_origin_ids,
        source_material_type_ids = EXCLUDED.source_material_type_ids,
        keyword_ids = EXCLUDED.keyword_ids,
        concept_tag_ids = EXCLUDED.concept_tag_ids,
        award_ceremony_win_ids = EXCLUDED.award_ceremony_win_ids,
        imdb_vote_count = EXCLUDED.imdb_vote_count,
        reception_score = EXCLUDED.reception_score,
        budget_bucket = EXCLUDED.budget_bucket,
        box_office_bucket = EXCLUDED.box_office_bucket,
        title_token_count = EXCLUDED.title_token_count,
        production_company_ids = EXCLUDED.production_company_ids,
        franchise_name_entry_ids = EXCLUDED.franchise_name_entry_ids,
        subgroup_entry_ids = EXCLUDED.subgroup_entry_ids,
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
        list(country_of_origin_ids),
        list(source_material_type_ids),
        list(keyword_ids),
        list(concept_tag_ids),
        list(award_ceremony_win_ids),
        imdb_vote_count,
        reception_score,
        budget_bucket,
        box_office_bucket,
        title_token_count,
        list(production_company_ids),
        list(franchise_name_entry_ids),
        list(subgroup_entry_ids),
    )
    await _execute_on_conn(conn, query, params)


async def batch_upsert_movie_awards(
    movie_id: int,
    awards: list[AwardNomination],
    award_name_entry_ids: Sequence[int | None] | None = None,
    conn=None,
) -> None:
    """
    Replace all award rows for a movie via delete + bulk insert.

    Accepts AwardNomination objects directly and extracts the relevant
    fields (ceremony_id, award_name, category, outcome_id, year) internally.
    Uses delete-then-insert rather than per-row upserts because awards
    for a movie are always ingested as a complete set.

    Args:
        movie_id: The movie to upsert awards for.
        awards: AwardNomination objects with known (non-None) ceremony_id values.
        award_name_entry_ids: Optional parallel list aligned 1:1 with
            ``awards``, supplying the resolved ``lex.award_name_entry``
            id for each row. Pass ``None`` (the default) when the caller
            has not resolved the entry table yet — the column is
            nullable and the UPDATE-style stamp from the backfill script
            will fill it in later. Pass a list of the same length as
            ``awards`` (with ``None`` for any entry that normalizes to
            empty) during ingest so the id is written in the same INSERT.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    if not awards:
        return

    if award_name_entry_ids is not None and len(award_name_entry_ids) != len(awards):
        # Cheap guard — a length mismatch here means the caller built the
        # parallel list from a different filtered set than ``awards``,
        # which would silently stamp the wrong ids on rows. Fail loudly
        # rather than commit bad data.
        raise ValueError(
            "batch_upsert_movie_awards: award_name_entry_ids length "
            f"({len(award_name_entry_ids)}) does not match awards "
            f"length ({len(awards)})."
        )

    # Delete existing awards for this movie, then bulk insert the new set.
    delete_query = "DELETE FROM public.movie_awards WHERE movie_id = %s"
    await _execute_on_conn(conn, delete_query, (movie_id,))

    # Build one row tuple per award. Each row carries its own variable-
    # length category_tag_ids list (leaf + ancestor ids — see
    # schemas/award_category_tags.py). The previous unnest(text[])
    # approach can't model variable-length per-row arrays because
    # Postgres requires 2-D arrays to be rectangular; a single VALUES
    # clause with one tuple per row sidesteps that and keeps the insert
    # to a single round trip. ~50 awards per movie at most, so the
    # parameter count stays modest.
    rows: list[tuple] = []
    for idx, a in enumerate(awards):
        category = a.category or ""
        entry_id = (
            award_name_entry_ids[idx] if award_name_entry_ids is not None else None
        )
        rows.append((
            movie_id,
            a.ceremony_id,
            a.award_name,
            category,
            tags_for_category(category),
            a.outcome.outcome_id,
            a.year,
            entry_id,
        ))

    values_template = ", ".join(["(%s, %s, %s, %s, %s, %s, %s, %s)"] * len(rows))
    insert_query = f"""
    INSERT INTO public.movie_awards
        (movie_id, ceremony_id, award_name, category, category_tag_ids, outcome_id, year, award_name_entry_id)
    VALUES {values_template}
    """
    flat_params: list = []
    for row in rows:
        flat_params.extend(row)
    await _execute_on_conn(conn, insert_query, flat_params)

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


def _build_people_union_cte(
    *,
    cte_name: str,
    term_ids: list[int],
    use_eligible: bool,
    exclude_movie_ids: Optional[set[int]] = None,
) -> tuple[str, list]:
    """
    Build a CTE that unions person matches across all role-specific posting tables.

    Produces a UNION ALL across actor/director/writer/producer tables, then
    COUNT(DISTINCT term_id) to handle cross-role deduplication (same person
    credited as both actor and director counts once).

    Args:
        cte_name: Name for the generated CTE block.
        term_ids: Resolved term IDs to search for across all role tables.
        use_eligible: Whether the eligible CTE exists in the query.
        exclude_movie_ids: Optional IDs excluded from results.

    Returns:
        Tuple of (cte_sql, params) for this bucket.
    """
    sub_parts: list[str] = []
    params: list = []

    for table in PEOPLE_POSTING_TABLES:
        eligibility_join = (
            "\n                JOIN eligible e ON e.movie_id = p.movie_id"
            if use_eligible
            else ""
        )
        exclusion_clause = ""
        sub_params: list = [term_ids]
        if exclude_movie_ids:
            exclusion_clause = "\n                  AND NOT (p.movie_id = ANY(%s::bigint[]))"
            sub_params.append(list(exclude_movie_ids))

        sub_parts.append(
            f"SELECT p.movie_id, p.term_id"
            f"\n                FROM {table.value} p{eligibility_join}"
            f"\n                WHERE p.term_id = ANY(%s::bigint[]){exclusion_clause}"
        )
        params.extend(sub_params)

    inner_union = "\n                UNION ALL\n                ".join(sub_parts)
    cte_sql = f"""{cte_name} AS (
            SELECT movie_id, COUNT(DISTINCT term_id)::int AS matched
            FROM (
                {inner_union}
            ) AS combined
            GROUP BY movie_id
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
        or character_term_ids
        or has_any_title
    )
    if not has_any_bucket:
        return CompoundLexicalResult(
            people_scores={},
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
        people_cte, people_params = _build_people_union_cte(
            cte_name="people_matches",
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
    character_by_query: dict[int, dict[int, int]] = {}
    title_scores_by_search: dict[int, dict[int, float]] = {}

    for bucket, query_idx, movie_id, score in rows:
        if bucket == "people":
            people_scores[movie_id] = int(score)
            continue
        if bucket == "character":
            character_by_query.setdefault(int(query_idx), {})[movie_id] = int(score)
            continue
        if bucket.startswith("title_"):
            search_idx = int(bucket.split("_")[1])
            title_scores_by_search.setdefault(search_idx, {})[movie_id] = float(score)

    return CompoundLexicalResult(
        people_scores=people_scores,
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
        watch_offer_keys, audio_language_ids, country_of_origin_ids,
        source_material_type_ids, keyword_ids, concept_tag_ids,
        imdb_vote_count, popularity_score, reception_score, budget_bucket,
        box_office_bucket.
    """
    if not movie_ids:
        return []

    query = """
        SELECT movie_id, title, poster_url, release_ts, runtime_minutes,
               maturity_rank, genre_ids, watch_offer_keys, audio_language_ids, country_of_origin_ids,
               source_material_type_ids, keyword_ids, concept_tag_ids,
               imdb_vote_count, popularity_score, reception_score, budget_bucket,
               box_office_bucket
        FROM public.movie_card
        WHERE movie_id = ANY(%s::bigint[])
    """
    columns = [
        "movie_id", "title", "poster_url", "release_ts",
        "runtime_minutes", "maturity_rank", "genre_ids", "watch_offer_keys",
        "audio_language_ids", "country_of_origin_ids",
        "source_material_type_ids", "keyword_ids", "concept_tag_ids",
        "imdb_vote_count", "popularity_score",
        "reception_score", "budget_bucket", "box_office_bucket",
    ]

    search_results = await _execute_read(query, (movie_ids,))
    return [dict(zip(columns, row)) for row in search_results]


async def fetch_movie_ids_missing_card(candidate_ids: list[int]) -> list[int]:
    """Return the subset of candidate_ids that have no row in movie_card.

    Uses an EXCEPT query so we only return IDs that genuinely need
    ingestion.  Follows the bulk-fetch convention (single query, no
    per-candidate calls).

    Args:
        candidate_ids: Movie IDs to check against movie_card.

    Returns:
        List of movie IDs from candidate_ids that are absent from
        movie_card, preserving no particular order.
    """
    if not candidate_ids:
        return []

    query = """
        SELECT unnest(%s::bigint[])
        EXCEPT
        SELECT movie_id FROM public.movie_card
    """
    rows = await _execute_read(query, (candidate_ids,))
    return [row[0] for row in rows]


async def fetch_reception_scores(movie_ids: list[int]) -> dict[int, float | None]:
    """
    Bulk fetch reception scores for quality-prior reranking.

    Lightweight alternative to fetch_movie_cards when only the reception
    score column is needed (e.g. for within-bucket tie-breaking).

    Args:
        movie_ids: List of movie IDs to fetch scores for.

    Returns:
        Dict mapping movie_id -> reception_score (None when missing).
    """
    if not movie_ids:
        return {}

    query = """
        SELECT movie_id, reception_score
        FROM public.movie_card
        WHERE movie_id = ANY(%s::bigint[])
    """

    rows = await _execute_read(query, (movie_ids,))
    return {row[0]: row[1] for row in rows}


async def fetch_browse_seed_ids(
    quality_inverted: bool,
    notability_inverted: bool,
    *,
    limit: int,
) -> list[int]:
    """Top `limit` movie_ids ordered by the Stage-4 browse composite.

    Composite mirrors
    ``search_v2.stage_4.priors.browse_composite`` exactly — keep the
    two in sync when formulas change. Structure:

        quality    = 0.7 * reception_ramp + 0.3 * popularity_ramp
        notability = popularity_ramp
        composite  = 0.5 * quality + 0.5 * notability

    Each ramp switches on the corresponding prior mode:
      * not inverted (enhanced / standard / suppressed) — "high" ramps
          reception:  (reception_score - 55) / 40, clamped [0, 1]
          popularity: popularity_score, clamped [0, 1]
      * inverted — "low" ramps
          reception:  (50 - reception_score) / 40, clamped [0, 1]
          popularity: 1 - popularity_score, clamped [0, 1]

    NULL-handling: COALESCE(col, 0) before the arithmetic. Postgres's
    GREATEST / LEAST ignore NULL args rather than propagating them
    (``LEAST(1, NULL)`` → 1), which would otherwise let a
    null-reception movie score a perfect 1.0 by accident.

    Callers should pre-collapse SUPPRESSED to the "high" branch before
    calling — the helper does not interpret SystemPrior itself, only
    the inversion bit, because browse seeding treats suppressed-as-
    standard by design (see ``browse_composite``).
    """
    query = """
        SELECT movie_id
        FROM (
            SELECT movie_id,
                0.5 * (
                    0.7 * CASE
                        WHEN %s THEN GREATEST(0.0, LEAST(1.0, (50.0 - COALESCE(reception_score, 0.0)) / 40.0))
                        ELSE GREATEST(0.0, LEAST(1.0, (COALESCE(reception_score, 0.0) - 55.0) / 40.0))
                    END
                    + 0.3 * CASE
                        WHEN %s THEN GREATEST(0.0, LEAST(1.0, 1.0 - COALESCE(popularity_score, 0.0)))
                        ELSE GREATEST(0.0, LEAST(1.0, COALESCE(popularity_score, 0.0)))
                    END
                )
                + 0.5 * CASE
                    WHEN %s THEN GREATEST(0.0, LEAST(1.0, 1.0 - COALESCE(popularity_score, 0.0)))
                    ELSE GREATEST(0.0, LEAST(1.0, COALESCE(popularity_score, 0.0)))
                END AS composite
            FROM public.movie_card
        ) AS scored
        ORDER BY composite DESC
        LIMIT %s
    """
    # Quality composite's popularity term uses quality_mode (not
    # notability_mode); hence quality_inverted is bound twice.
    params = (quality_inverted, quality_inverted, notability_inverted, limit)
    rows = await _execute_read(query, params)
    return [row[0] for row in rows]


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
#     STUDIO ENDPOINT HELPERS
# ===============================
#
# Read helpers dedicated to the step 3 studio endpoint
# (search_v2/stage_3/studio_query_execution.py). Two paths:
#   - Brand path: direct lookup by brand_id on lex.inv_production_brand_postings.
#   - Freeform path: DF-filtered token → production_company_id intersection
#     over lex.studio_token, then GIN && join against movie_card.production_company_ids.


async def fetch_movie_ids_by_brand(
    brand_id: int,
    restrict_movie_ids: Optional[set[int]] = None,
) -> set[int]:
    """
    Resolve a ProductionBrand enum brand_id to the set of stamped movie IDs.

    Reads lex.inv_production_brand_postings. The brand-path score is flat 1.0
    per matched movie (the prominence column `first_matching_index` is stored
    but deliberately NOT used — see the studio endpoint design notes for why
    IMDB ordering is unreliable across regions).

    Args:
        brand_id: ProductionBrand.brand_id (SMALLINT). Caller passes the int,
            not the enum, so this helper stays pure SQL-layer.
        restrict_movie_ids: Optional candidate-pool restriction for the
            preference / restrict-set path. Applied server-side.

    Returns:
        Set of movie IDs stamped with the given brand at ingest. Empty when
        the brand has no postings (should be rare for registry brands).
    """
    if restrict_movie_ids is not None:
        if not restrict_movie_ids:
            return set()
        query = """
            SELECT movie_id
            FROM lex.inv_production_brand_postings
            WHERE brand_id = %s AND movie_id = ANY(%s::bigint[])
        """
        rows = await _execute_read(query, (brand_id, list(restrict_movie_ids)))
    else:
        query = """
            SELECT movie_id
            FROM lex.inv_production_brand_postings
            WHERE brand_id = %s
        """
        rows = await _execute_read(query, (brand_id,))
    return {row[0] for row in rows}


async def fetch_company_ids_for_tokens(
    tokens: list[str],
    df_ceiling: int,
) -> dict[str, set[int]]:
    """
    Resolve freeform-path tokens to production_company_ids, DF-filtered.

    One query joins lex.studio_token against the materialized DF view,
    drops tokens whose doc_frequency exceeds the ceiling, and returns the
    (token → company_ids) mapping the executor uses for per-name
    intersection. Tokens absent from the input map entirely (DF-dropped or
    never seen) are omitted — the executor must treat "missing key" as
    "name fails intersection" rather than "name matches everything".

    Args:
        tokens: Normalized + hyphen-split tokens from one freeform_name.
            May repeat across names; caller can dedupe or pass as-is.
        df_ceiling: Inclusive upper bound on `doc_frequency`. Tokens above
            this are too common to discriminate (empirically tuned to 323
            by the studio endpoint; re-derive when the catalog grows).

    Returns:
        Mapping `{token: {production_company_id, ...}}` covering only
        tokens that passed the DF filter and had at least one posting.
    """
    if not tokens:
        return {}

    query = """
        SELECT st.token, st.production_company_id
        FROM lex.studio_token st
        JOIN lex.studio_token_doc_frequency df ON df.token = st.token
        WHERE st.token = ANY(%s::text[])
          AND df.doc_frequency <= %s
    """
    rows = await _execute_read(query, (tokens, df_ceiling))
    out: dict[str, set[int]] = {}
    for token, company_id in rows:
        out.setdefault(token, set()).add(company_id)
    return out


async def fetch_movie_ids_by_production_company_ids(
    production_company_ids: set[int],
    restrict_movie_ids: Optional[set[int]] = None,
) -> set[int]:
    """
    Resolve a set of production_company_ids to the movies they appear on.

    Uses the GIN index on public.movie_card.production_company_ids with the
    `&&` (overlap) operator — a movie qualifies when any of its stamped
    company ids is in the input set. This prevents the cross-company token
    false-positive (see plan for details): three single-token unrelated
    companies won't collapse into a "matched" result.

    Args:
        production_company_ids: Company ids produced by per-name token
            intersection in the freeform path.
        restrict_movie_ids: Optional candidate-pool restriction.

    Returns:
        Set of movie IDs whose `production_company_ids` overlaps the input.
    """
    if not production_company_ids:
        return set()

    id_list = list(production_company_ids)
    if restrict_movie_ids is not None:
        if not restrict_movie_ids:
            return set()
        query = """
            SELECT movie_id
            FROM public.movie_card
            WHERE production_company_ids && %s::bigint[]
              AND movie_id = ANY(%s::bigint[])
        """
        rows = await _execute_read(query, (id_list, list(restrict_movie_ids)))
    else:
        query = """
            SELECT movie_id
            FROM public.movie_card
            WHERE production_company_ids && %s::bigint[]
        """
        rows = await _execute_read(query, (id_list,))
    return {row[0] for row in rows}


# ===============================
#     ENTITY ENDPOINT HELPERS
# ===============================
#
# Read helpers dedicated to the step 3 entity endpoint
# (search_v2/stage_3/entity_query_execution.py). Kept here so SQL for
# the lexical schema lives in one place.


async def fetch_character_strings_exact(phrases: list[str]) -> dict[str, int]:
    """Exact-match normalized character phrases against lex.character_strings.

    The entity endpoint treats each character name variation as an
    exact-string lookup (the LLM is prompted to produce the most common
    credited form). Caller must normalize phrases with normalize_string()
    before passing them in — matching policy is exact-string equality
    after that shared normalization.

    Args:
        phrases: Normalized character phrase strings.

    Returns:
        Mapping of norm_str → string_id for every phrase that exists.
        Phrases absent from the dictionary are silently omitted.
    """
    if not phrases:
        return {}

    query = """
        SELECT norm_str, string_id
        FROM lex.character_strings
        WHERE norm_str = ANY(%s::text[])
    """
    rows = await _execute_read(query, (phrases,))
    return {row[0]: row[1] for row in rows}


async def fetch_actor_billing_rows(
    term_ids: list[int],
    restrict_movie_ids: Optional[set[int]] = None,
) -> list[tuple[int, int, int]]:
    """Fetch (movie_id, billing_position, cast_size) for actor term_ids.

    Used by the entity endpoint's actor prominence scoring. Unlike
    fetch_movie_ids_by_term_ids, this returns per-row billing data so
    the caller can compute zone-based prominence scores.

    Args:
        term_ids: Resolved string IDs for actor names.
        restrict_movie_ids: Optional candidate-pool filter (used by
            preference execution to narrow the scan to the pool).

    Returns:
        List of (movie_id, billing_position, cast_size) tuples. Rows
        with NULL billing_position or cast_size are filtered out (they
        carry no prominence signal).
    """
    if not term_ids:
        return []

    params: list = [term_ids]
    restrict_clause = ""
    if restrict_movie_ids:
        restrict_clause = " AND movie_id = ANY(%s::bigint[])"
        params.append(list(restrict_movie_ids))

    # billing_position and cast_size are NOT NULL on the schema, but the
    # IS NOT NULL gates defend against any future schema drift where a
    # legacy upsert path leaves them unset — per the entity endpoint's
    # "skip rows with missing billing data" execution rule.
    query = f"""
        SELECT movie_id, billing_position, cast_size
        FROM lex.inv_actor_postings
        WHERE term_id = ANY(%s::bigint[]){restrict_clause}
          AND billing_position IS NOT NULL
          AND cast_size IS NOT NULL
          AND cast_size > 0
    """
    rows = await _execute_read(query, params)
    return [(row[0], row[1], row[2]) for row in rows]


async def fetch_movie_ids_with_title_like(
    like_pattern: str,
    restrict_movie_ids: Optional[set[int]] = None,
) -> set[int]:
    """Case-insensitive LIKE match of a title pattern against public.movie_card.title.

    The entity endpoint title_pattern sub-type routes here. The caller
    is responsible for preparing the LIKE pattern: (1) normalize the
    raw user pattern with normalize_string, (2) escape LIKE wildcards
    via escape_like, (3) wrap with '%' on both sides for contains or
    suffix '%' for starts_with.

    Args:
        like_pattern: Fully formed LIKE pattern (e.g. "%love%", "the %").
        restrict_movie_ids: Optional candidate-pool filter used by
            preference execution. Pushed to the DB to avoid pulling
            back every match for high-cardinality patterns.

    Returns:
        Set of movie IDs whose title matches the pattern.
    """
    params: list = [like_pattern]
    restrict_clause = ""
    if restrict_movie_ids:
        restrict_clause = " AND movie_id = ANY(%s::bigint[])"
        params.append(list(restrict_movie_ids))

    query = f"""
        SELECT movie_id
        FROM public.movie_card
        WHERE title ILIKE %s{restrict_clause}
    """
    rows = await _execute_read(query, params)
    return {row[0] for row in rows}


# ===============================
#   FRANCHISE ENDPOINT HELPERS
# ===============================
#
# Read helpers dedicated to the step 3 franchise endpoint
# (search_v2/stage_3/franchise_query_execution.py). Two helpers:
#
#   - fetch_franchise_entry_ids_for_tokens: posting-list fetch for a
#     batch of tokens against lex.franchise_token. No DF filter —
#     stopwords are dropped upstream in tokenize_franchise_string, so
#     every token that reaches this helper is already discriminative.
#
#   - fetch_franchise_movie_ids: final GIN `&&` overlap on
#     movie_card.franchise_name_entry_ids / subgroup_entry_ids, with
#     an optional LEFT JOIN to movie_franchise_metadata when any
#     structural axis (lineage_position / is_spinoff / is_crossover /
#     launched_franchise / launched_subgroup) is active.
#
# The two-step shape mirrors the studio endpoint
# (fetch_company_ids_for_tokens → fetch_movie_ids_by_production_company_ids)
# — same rationale: per-name intersection lives in Python, final
# movie-id resolution is a single SQL overlap.


async def fetch_franchise_entry_ids_for_tokens(
    tokens: list[str],
) -> dict[str, set[int]]:
    """
    Resolve a batch of tokens to franchise_entry_ids via lex.franchise_token.

    Mirror of fetch_company_ids_for_tokens but without a DF-ceiling filter:
    FRANCHISE_STOPLIST is applied inside tokenize_franchise_string before the
    executor ever calls this helper, so every token passed in is already
    discriminative. Re-filtering here would be redundant and would obscure
    the "ingest and query share one tokenizer" invariant.

    Tokens that have no postings (never stamped at ingest) are omitted from
    the result — the executor must treat "missing key" as "name fails
    intersection", not "name matches everything" (matches the
    Phase-3 behavior in studio_query_execution.py).

    Args:
        tokens: Tokens from one or more franchise / subgroup names, after
            normalize + tokenize + stopword drop. Caller typically passes
            `sorted(set(...))` for deterministic query text.

    Returns:
        Mapping `{token: {franchise_entry_id, ...}}` covering only tokens
        that had at least one posting.
    """
    if not tokens:
        return {}

    query = """
        SELECT token, franchise_entry_id
        FROM lex.franchise_token
        WHERE token = ANY(%s::text[])
    """
    rows = await _execute_read(query, (tokens,))
    out: dict[str, set[int]] = {}
    for token, entry_id in rows:
        out.setdefault(token, set()).add(entry_id)
    return out


async def fetch_franchise_movie_ids(
    *,
    franchise_name_entry_ids: set[int] | None,
    subgroup_entry_ids: set[int] | None,
    lineage_position_id: int | None,
    is_spinoff: bool,
    is_crossover: bool,
    launched_franchise: bool,
    launched_subgroup: bool,
    restrict_movie_ids: set[int] | None = None,
) -> set[int]:
    """Resolve pre-computed franchise entry-id sets + structural flags to movie IDs.

    Builds a single AND-composed WHERE clause from whichever axes are
    active. All populated constraints must hold simultaneously — this is
    the AND execution policy for the franchise endpoint.

    Array-axis matching uses the GIN `&&` overlap operator against the
    denormalized entry-id arrays on public.movie_card. Structural-axis
    predicates (lineage_position, is_spinoff, is_crossover,
    launched_franchise, launched_subgroup) live on
    public.movie_franchise_metadata, so we LEFT JOIN only when any of
    those is active — keeping the zero-structural-axis case as a
    single-table scan on movie_card.

    Args:
        franchise_name_entry_ids: Pre-resolved entry-id set from the
            executor's `franchise_or_universe_names` token intersection +
            cross-name union. None = axis not active. Empty set is treated
            as "no conditions" for this axis and NOT as a universal match
            — see the defensive guard below.
        subgroup_entry_ids: Same, for the `recognized_subgroups` axis.
        lineage_position_id: SMALLINT ID for the desired lineage position
            (from LineagePosition.lineage_position_id). None = not filtered.
        is_spinoff: True = require mfm.is_spinoff = TRUE. False = not filtered.
        is_crossover: True = require mfm.is_crossover = TRUE. False = not filtered.
        launched_franchise: True = require mfm.launched_franchise = TRUE.
            False = not filtered.
        launched_subgroup: True = require mfm.launched_subgroup = TRUE.
            False = not filtered.
        restrict_movie_ids: Optional candidate-pool filter. When provided,
            only movies in this set can appear in the result.

    Returns:
        Set of movie_ids that satisfy all active constraints. Empty set when
        no conditions are provided (defensive guard) or when no movies match.
    """
    # Array-axis predicates hit the denormalized entry-id arrays on
    # public.movie_card.
    array_conditions: list[str] = []
    array_params: list = []
    if franchise_name_entry_ids:
        array_conditions.append("mc.franchise_name_entry_ids && %s::bigint[]")
        array_params.append(list(franchise_name_entry_ids))
    if subgroup_entry_ids:
        array_conditions.append("mc.subgroup_entry_ids && %s::bigint[]")
        array_params.append(list(subgroup_entry_ids))

    # Structural-axis predicates hit public.movie_franchise_metadata.
    structural_conditions: list[str] = []
    structural_params: list = []
    if lineage_position_id is not None:
        structural_conditions.append("mfm.lineage_position = %s")
        structural_params.append(lineage_position_id)
    if is_spinoff:
        structural_conditions.append("mfm.is_spinoff = TRUE")
    if is_crossover:
        structural_conditions.append("mfm.is_crossover = TRUE")
    if launched_franchise:
        structural_conditions.append("mfm.launched_franchise = TRUE")
    if launched_subgroup:
        structural_conditions.append("mfm.launched_subgroup = TRUE")

    # Defensive guard: the FranchiseQuerySpec validator requires at least
    # one axis, and the executor early-exits when a requested textual axis
    # collapses to empty entry-id sets. This branch should therefore not
    # be reachable in normal operation.
    if not array_conditions and not structural_conditions:
        return set()

    # Structural-only specs (e.g. "spinoffs", "sequels with no named
    # franchise") query movie_franchise_metadata directly — mfm is far
    # smaller than movie_card and has no useful column on mc to filter by,
    # so driving from mc would force an unnecessary join across ~100K
    # rows. When at least one array axis is active, drive from movie_card
    # (GIN index) and LEFT JOIN mfm if any structural predicate is also
    # active. The LEFT JOIN is effectively an INNER join because NULL
    # mfm.* columns cannot satisfy the `= TRUE` / `= N` predicates.
    if not array_conditions:
        from_clause = "public.movie_franchise_metadata mfm"
        movie_id_expr = "mfm.movie_id"
        restrict_column = "mfm.movie_id"
        conditions = structural_conditions
        params: list = list(structural_params)
    elif structural_conditions:
        from_clause = (
            "public.movie_card mc "
            "LEFT JOIN public.movie_franchise_metadata mfm USING (movie_id)"
        )
        movie_id_expr = "mc.movie_id"
        restrict_column = "mc.movie_id"
        conditions = array_conditions + structural_conditions
        params = array_params + structural_params
    else:
        from_clause = "public.movie_card mc"
        movie_id_expr = "mc.movie_id"
        restrict_column = "mc.movie_id"
        conditions = array_conditions
        params = list(array_params)

    if restrict_movie_ids is not None:
        if not restrict_movie_ids:
            return set()
        conditions.append(f"{restrict_column} = ANY(%s::bigint[])")
        params.append(list(restrict_movie_ids))

    where_clause = " AND ".join(conditions)
    query = f"SELECT {movie_id_expr} FROM {from_clause} WHERE {where_clause}"
    rows = await _execute_read(query, params)
    return {row[0] for row in rows}


# ===============================
#    KEYWORD ENDPOINT HELPERS
# ===============================
#
# Read helper for the step 3 keyword endpoint
# (search_v2/stage_3/keyword_query_execution.py). The endpoint selects
# exactly one UnifiedClassification member, which resolves to a single
# (backing_column, source_id) pair. Execution is one GIN `&&` overlap
# against that single column — no cross-column unions, no dual-backing.


# Whitelist of columns the keyword executor is allowed to target. The
# executor must pass one of these via `backing_column`; any other value
# raises rather than being interpolated into SQL, because the column
# name goes into the query string directly (psycopg cannot parameterize
# identifiers). Source of truth is schemas/unified_classification.py.
_KEYWORD_ALLOWED_COLUMNS: frozenset[str] = frozenset(
    {"keyword_ids", "source_material_type_ids", "concept_tag_ids"}
)


async def fetch_keyword_matched_movie_ids(
    *,
    backing_column: str,
    source_id: int,
    restrict_movie_ids: set[int] | None = None,
) -> set[int]:
    """Return movie_ids whose `backing_column` array contains `source_id`.

    Single-column GIN `&&` overlap against movie_card. `backing_column`
    is validated against a whitelist because it is interpolated into
    the SQL text (column names cannot be parameterized); `source_id`
    is bound as a parameter.

    Args:
        backing_column: One of "keyword_ids", "source_material_type_ids",
            "concept_tag_ids". Resolved by the caller via
            entry_for(member).backing_column.
        source_id: The ID to look for inside that array column.
        restrict_movie_ids: Optional candidate-pool filter. When
            provided, only movies in this set can appear in the result.

    Returns:
        Set of movie_ids that have `source_id` in `backing_column`.
    """
    if backing_column not in _KEYWORD_ALLOWED_COLUMNS:
        raise ValueError(
            f"fetch_keyword_matched_movie_ids: unsupported backing_column "
            f"{backing_column!r}; allowed: {sorted(_KEYWORD_ALLOWED_COLUMNS)}"
        )

    conditions: list[str] = [f"{backing_column} && %s::int[]"]
    params: list = [[source_id]]

    if restrict_movie_ids is not None:
        conditions.append("movie_id = ANY(%s::bigint[])")
        params.append(list(restrict_movie_ids))

    where_clause = " AND ".join(conditions)
    query = f"SELECT movie_id FROM public.movie_card WHERE {where_clause}"
    rows = await _execute_read(query, params)
    return {row[0] for row in rows}


# ===============================
#     AWARD ENDPOINT HELPERS
# ===============================
#
# Read helpers dedicated to the step 3 award endpoint
# (search_v2/stage_3/award_query_execution.py). Two paths:
#
#   1) Fast path via the denormalized movie_card.award_ceremony_win_ids
#      SMALLINT[] (GIN-indexed) — a cheap "has any non-Razzie win"
#      presence check used when the spec has no filters, outcome is
#      WINNER, scoring_mode=FLOOR, scoring_mark=1.
#
#   2) Standard path via public.movie_awards — COUNT(*) grouped by
#      movie_id, with whichever filter axes the spec populated.
#
# Razzie ceremony_id = 10. The award_name axis is filtered by
# `award_name_entry_id` (INT FK into lex.award_name_entry), resolved
# upstream by the executor via token intersection against
# lex.award_name_token — see
# search_improvement_planning/v2_search_data_improvements.md § Award
# Name Resolution. Category tags remain an array-overlap match on the
# `category_tag_ids` INT[] column.


# Ceremony_id for the Razzie Awards. Stripped from the fast-path
# index check and excluded from the standard-path COUNT whenever the
# spec did not explicitly include it in `ceremonies`.
_RAZZIE_CEREMONY_ID: int = AwardCeremony.RAZZIE.ceremony_id

# All non-Razzie ceremony_ids. Passed to the GIN `&&` (array overlap)
# operator for the fast path so Postgres can use idx_movie_card_
# award_ceremony_win_ids rather than scanning. Derived from the
# AwardCeremony enum rather than hardcoded so new ceremonies land here
# automatically when the enum grows.
_NON_RAZZIE_CEREMONY_IDS: list[int] = sorted(
    c.ceremony_id for c in AwardCeremony if c is not AwardCeremony.RAZZIE
)


async def fetch_award_name_entry_ids_for_tokens(
    tokens: list[str],
) -> dict[str, set[int]]:
    """Resolve a batch of tokens to award_name_entry_ids via
    lex.award_name_token.

    Structural mirror of fetch_franchise_entry_ids_for_tokens. The
    AWARD_QUERY_STOPLIST is applied inside
    tokenize_award_string_for_query before the executor ever calls this
    helper, so every token passed in is already discriminative — no
    re-filtering happens here.

    Tokens that have no postings (never stamped at ingest) are omitted
    from the result. The executor must treat "missing key" as "this
    name fails intersection", not "this name matches everything" —
    matches the equivalent contract in the franchise and studio
    executors.

    Args:
        tokens: Tokens from one or more award names, after normalize +
            stoplist drop. Callers typically pass ``sorted(set(...))``
            for deterministic query text (stable plan cache,
            reproducible logs).

    Returns:
        Mapping ``{token: {award_name_entry_id, ...}}`` covering only
        tokens that had at least one posting.
    """
    if not tokens:
        return {}

    query = """
        SELECT token, award_name_entry_id
        FROM lex.award_name_token
        WHERE token = ANY(%s::text[])
    """
    rows = await _execute_read(query, (tokens,))
    out: dict[str, set[int]] = {}
    for token, entry_id in rows:
        out.setdefault(token, set()).add(entry_id)
    return out


async def fetch_award_fast_path_movie_ids(
    *,
    restrict_movie_ids: set[int] | None = None,
) -> set[int]:
    """Return movie_ids with at least one non-Razzie ceremony win.

    Uses the denormalized award_ceremony_win_ids SMALLINT[] on movie_card
    (GIN-indexed). Overlap with the non-Razzie ceremony set means the
    movie has won at least one prize at a non-Razzie ceremony. Razzie-
    only movies — whose only wins are at ceremony_id 10 — do not match.

    The `&&` operator is chosen over `cardinality(array_remove(...)) > 0`
    because `&&` is GIN-indexable; the remove-then-cardinality form would
    force a sequential scan.

    Args:
        restrict_movie_ids: Optional candidate-pool restriction. When
            provided, only movies in this set may appear in the result.

    Returns:
        Set of movie_ids that have at least one non-Razzie ceremony win.
    """
    conditions: list[str] = ["award_ceremony_win_ids && %s::smallint[]"]
    params: list = [_NON_RAZZIE_CEREMONY_IDS]

    if restrict_movie_ids is not None:
        conditions.append("movie_id = ANY(%s::bigint[])")
        params.append(list(restrict_movie_ids))

    where_clause = " AND ".join(conditions)
    query = f"SELECT movie_id FROM public.movie_card WHERE {where_clause}"
    rows = await _execute_read(query, params)
    return {row[0] for row in rows}


async def fetch_award_row_counts(
    *,
    ceremony_ids: list[int] | None,
    award_name_entry_ids: set[int] | None,
    category_tag_ids: list[int] | None,
    outcome_id: int | None,
    year_from: int | None,
    year_to: int | None,
    exclude_razzie: bool,
    restrict_movie_ids: set[int] | None = None,
) -> dict[int, int]:
    """Count matching movie_awards rows grouped by movie_id.

    Builds a single AND-composed WHERE clause from whichever axes are
    populated, then `GROUP BY movie_id` yields the per-movie has_count
    the scoring formula consumes.

    The award_name axis is filtered by `award_name_entry_id` — the INT
    FK into lex.award_name_entry. Entry ids come pre-resolved from the
    executor's token-intersection phase; this helper does not know (or
    need to know) how they were computed.

    Category filter uses the ingestion-derived `category_tag_ids` INT[]
    column (see schemas/award_category_tags.py) and an array-overlap
    (`&&`) test against the GIN index. The LLM emits tag ids at any
    of the three levels (leaf / mid / group); a single overlap query
    handles every specificity.

    Razzie exclusion policy:
      - When ceremony_ids is None and exclude_razzie is True, add
        `ceremony_id <> 10`. This is the default (null/empty ceremonies
        filter on the spec side) — Razzie is excluded unless the user
        explicitly asked for it.
      - When ceremony_ids is provided, the caller has already decided
        whether to include Razzie (id=10 is either in the list or not).
        No default exclusion is added — respect the user's explicit set.

    Args:
        ceremony_ids: List of AwardCeremony.ceremony_id values the row
            must match (ANY). None = no ceremony filter.
        award_name_entry_ids: Pre-resolved lex.award_name_entry ids the
            row's award_name_entry_id must match (ANY). None/empty =
            no filter on award_name.
        category_tag_ids: List of CategoryTag.tag_id values the row's
            category_tag_ids array must overlap with (ANY). None/empty
            = no filter on category.
        outcome_id: AwardOutcome.outcome_id (1=winner, 2=nominee). None =
            match either outcome.
        year_from: Inclusive lower bound of the year filter. None = no
            lower bound.
        year_to: Inclusive upper bound of the year filter. None = no
            upper bound.
        exclude_razzie: When True and ceremony_ids is None, add a
            `ceremony_id <> 10` guard. Ignored when ceremony_ids is set.
        restrict_movie_ids: Optional candidate-pool restriction pushed
            into the WHERE so the aggregation only touches relevant rows.

    Returns:
        Mapping of movie_id → row_count. Movies with zero matching rows
        are absent from the map (GROUP BY omits them).
    """
    conditions: list[str] = []
    params: list = []

    if ceremony_ids:
        conditions.append("ceremony_id = ANY(%s::smallint[])")
        params.append(ceremony_ids)
    elif exclude_razzie:
        # Default Razzie exclusion — only applied when the caller did
        # not provide an explicit ceremony_ids list.
        conditions.append("ceremony_id <> %s")
        params.append(_RAZZIE_CEREMONY_ID)

    if award_name_entry_ids:
        conditions.append("award_name_entry_id = ANY(%s::int[])")
        params.append(list(award_name_entry_ids))

    if category_tag_ids:
        # Array overlap against the GIN-indexed category_tag_ids column.
        # Hits whichever tag level(s) the LLM picked.
        conditions.append("category_tag_ids && %s::int[]")
        params.append(category_tag_ids)

    if outcome_id is not None:
        conditions.append("outcome_id = %s")
        params.append(outcome_id)

    if year_from is not None and year_to is not None:
        # Inclusive range; single-year specs arrive with year_from == year_to.
        conditions.append("year BETWEEN %s AND %s")
        params.extend([year_from, year_to])

    if restrict_movie_ids is not None:
        conditions.append("movie_id = ANY(%s::bigint[])")
        params.append(list(restrict_movie_ids))

    # Fallback WHERE clause for the degenerate "no filters, no Razzie
    # exclusion" call — would count every row in movie_awards. Execution
    # should never reach that configuration (the standard path is only
    # entered when some filter is active or exclude_razzie is True), but
    # a TRUE fallback keeps the SQL syntactically valid either way.
    where_clause = " AND ".join(conditions) if conditions else "TRUE"
    query = f"""
        SELECT movie_id, COUNT(*) AS row_count
        FROM public.movie_awards
        WHERE {where_clause}
        GROUP BY movie_id
    """
    rows = await _execute_read(query, params)
    return {row[0]: row[1] for row in rows}
