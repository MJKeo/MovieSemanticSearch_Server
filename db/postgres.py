"""
Database connection pool and query methods for the API service.

This module provides a psycopg v3 AsyncConnectionPool configured for production use,
along with async helper functions for executing queries and public methods for
upserting/inserting movie and lexical data.
"""

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, Optional, Sequence
from psycopg_pool import AsyncConnectionPool
from implementation.misc.sql_like import escape_like
from implementation.classes.enums import MaturityRating
from implementation.classes.schemas import MetadataFilters
from schemas.api_responses import MovieCard
from schemas.award_category_tags import tags_for_category
from schemas.enums import AwardCeremony, AwardOutcome
from schemas.imdb_models import AwardNomination
from schemas.metadata import FranchiseOutput


NEUTRAL_RERANKER_SEED_LIMIT = 2000
NEUTRAL_RERANKER_SEED_POPULARITY_WEIGHT = 0.8
NEUTRAL_RERANKER_SEED_RECEPTION_WEIGHT = 0.2


class PostingTable(Enum):
    """Supported posting tables for lexical matching and exclusion resolution."""
    ACTOR = "lex.inv_actor_postings"
    DIRECTOR = "lex.inv_director_postings"
    WRITER = "lex.inv_writer_postings"
    PRODUCER = "lex.inv_producer_postings"
    COMPOSER = "lex.inv_composer_postings"
    CHARACTER = "lex.inv_character_postings"


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

# Maximum term_ids returned per query phrase during resolution.
# Phrases matching more character names than this are too vague to be
# useful and would bloat the posting join.
_CHARACTER_RESOLVE_LIMIT_PER_PHRASE: int = 500


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
    # min_size raised from 4 → 10 so the steady-state burst of ~10-15
    # concurrent ops per request (Stage 4 generator/reranker fan-out
    # + per-branch hydration) does not pay connection-creation latency
    # on the hot path. The previous value left request slots 5+
    # paying ~tens-of-ms to spin up a connection mid-request.
    min_size=10,
    # Sized for the similarity flow's fan-out: the single-anchor lane gather
    # fires up to 11 concurrent ops and the multi-anchor flow has two
    # gather waves of 9 + 6. With two in-flight requests we want to avoid
    # pool-acquire serialization, so max_size is set well above the
    # per-request concurrency.
    max_size=25,
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


def _build_movie_card_conditions(
    filters: MetadataFilters,
) -> tuple[list[str], list]:
    """
    Build the per-column WHERE conditions for filtering public.movie_card by
    the user-supplied MetadataFilters. Returns ([cond_sql, ...], params).

    Pure SQL fragments — no leading ``WHERE`` or ``AND``. Each entry in the
    returned list is one self-contained condition; the caller joins them
    with ``AND``. Empty list means "no conditions" (e.g. inactive filter).

    Centralized so the WITH-chain helper (``_build_eligible_cte``) and the
    inline AND-clause helper (``_build_inline_movie_card_filter_clause``)
    share one source of truth for column-level translation.
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

    # NOTE: maturity_rank range conditions exclude rows where
    # maturity_rank IS NULL (UNRATED movies — see _build_qdrant_payload in
    # movie_ingestion/final_ingestion/ingest_movie.py:1110-1113). This
    # matches user intent — "at most PG-13" should not surface unrated
    # content — and is symmetric with the Qdrant Range filter, which also
    # excludes NULL payload values.
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

    return conditions, params


async def _build_eligible_cte(filters: MetadataFilters) -> tuple[str, list]:
    """
    Build the SQL fragment and parameter list for a MATERIALIZED eligible-set
    CTE against public.movie_card.

    Used by callers that want to inject ``WITH eligible AS MATERIALIZED (...)``
    once and reuse the resolved movie_id set across multiple unions / scans
    in the same statement (the canonical example is
    ``execute_compound_lexical_search``).

    Returns:
        (cte_sql, params) where cte_sql is the full
        ``eligible AS MATERIALIZED (...)`` block ready to prepend into a
        WITH chain, and params is the ordered list of bind values.
    """
    conditions, params = _build_movie_card_conditions(filters)
    where_clause = " AND ".join(conditions) if conditions else "TRUE"

    cte_sql = (
        f"eligible AS MATERIALIZED (\n"
        f"            SELECT movie_id\n"
        f"            FROM public.movie_card\n"
        f"            WHERE {where_clause}\n"
        f"        )"
    )
    return cte_sql, params


def _build_inline_movie_card_filter_clause(
    filters: Optional[MetadataFilters],
    *,
    movie_id_column: str = "movie_id",
) -> tuple[str, list]:
    """
    Build an inline ``AND <col> IN (SELECT movie_id FROM public.movie_card
    WHERE ...)`` fragment for primitives that issue a single-statement
    query without a WITH chain.

    The fragment includes its own leading ``AND`` so it can be concatenated
    directly onto an existing WHERE clause. Returns ("", []) when no filter
    is active so callers can splice unconditionally and the query stays
    byte-identical to today when filters are unset (no extra subquery cost).

    Args:
        filters: User filters, or None for "no filter".
        movie_id_column: Qualified column name on the outer query that holds
            the candidate's movie_id (default ``movie_id``; use
            ``<alias>.movie_id`` if the outer query is aliased).

    Returns:
        (clause_sql, params) where clause_sql is either the empty string
        (no-op) or `" AND <col> IN (SELECT movie_id FROM public.movie_card
        WHERE <conds>)"`, and params lines up with the ``%s`` placeholders
        inside.
    """
    if filters is None or not filters.is_active:
        return "", []

    conditions, params = _build_movie_card_conditions(filters)
    if not conditions:
        # is_active was True but every field collapsed to empty (e.g. an
        # empty genres list). Treat as inactive.
        return "", []

    where_clause = " AND ".join(conditions)
    clause = (
        f" AND {movie_id_column} IN ("
        f"SELECT movie_id FROM public.movie_card WHERE {where_clause})"
    )
    return clause, params


def _build_direct_movie_card_filter_clause(
    filters: Optional[MetadataFilters],
) -> tuple[str, list]:
    """
    Build a direct ``AND <conds>`` fragment for primitives whose primary
    FROM table is already ``public.movie_card`` — in that case the columns
    referenced by ``MetadataFilters`` are already in scope, so a self-IN
    subquery is wasteful. Compose conditions inline instead.

    Returns ("", []) when filters is None / inactive so the caller can
    splice unconditionally without any extra runtime cost when filters
    are unset.

    Returns:
        (clause_sql, params) where clause_sql is either the empty string
        or `" AND <cond_1> AND <cond_2> ..."` (with leading space + AND).
    """
    if filters is None or not filters.is_active:
        return "", []

    conditions, params = _build_movie_card_conditions(filters)
    if not conditions:
        return "", []

    return " AND " + " AND ".join(conditions), params



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
    # Dedupe pairs, then sort by string_id — the ON CONFLICT arbiter
    # column — so concurrent writers acquire PK locks in the same
    # order. Mirrors the deadlock-avoidance pattern in
    # batch_upsert_lexical_dictionary, which sorts by norm_str because
    # that table's arbiter is the norm_str UNIQUE index. Without this
    # sort, two movies upserting overlapping character term_ids in
    # cast-edge order can circular-wait on the PK index (observed with
    # hyphen-variant expansion, which triples per-character row count
    # and inflates cross-movie string_id overlap).
    ordered_pairs = sorted(
        dict.fromkeys(zip(string_ids, norm_strings)),
        key=lambda pair: pair[0],
    )
    deduped_string_ids = [sid for sid, _ in ordered_pairs]
    deduped_norm_strings = [ns for _, ns in ordered_pairs]

    query = """
    INSERT INTO lex.character_strings (string_id, norm_str)
    SELECT unnest(%s::bigint[]), unnest(%s::text[])
    ON CONFLICT (string_id) DO UPDATE SET
        norm_str = EXCLUDED.norm_str;
    """
    await _execute_on_conn(conn, query, (deduped_string_ids, deduped_norm_strings))



async def batch_insert_actor_postings(
    term_ids: list[int],
    billing_positions: list[int],
    movie_id: int,
    cast_size: int,
    conn=None,
) -> None:
    """
    Insert actor postings with billing metadata for one movie.

    ``term_ids`` and ``billing_positions`` must be aligned one-to-one.
    Hyphen-variant expansion emits multiple term_ids that share the same
    billing_position so any variant form can resolve to the credit
    without inflating prominence denominators (see
    [implementation/misc/helpers.py](../implementation/misc/helpers.py)
    ``expand_hyphen_variants``).

    Args:
        term_ids: Actor term IDs for every ``(name, variant)`` pair.
        billing_positions: 1-based position matching each term_id back to
            its originating distinct actor.
        movie_id: Movie ID that owns all postings.
        cast_size: Number of distinct actors (pre-variant) in the film.
            Applied uniformly to every inserted row.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    if not term_ids:
        return
    if len(term_ids) != len(billing_positions):
        raise ValueError(
            "batch_insert_actor_postings: term_ids and billing_positions "
            "lengths differ."
        )
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


async def batch_insert_character_postings(
    term_ids: list[int],
    billing_positions: list[int],
    movie_id: int,
    character_cast_size: int,
    conn=None,
) -> None:
    """
    Insert character postings with billing metadata for one movie.

    ``term_ids`` and ``billing_positions`` must be aligned one-to-one.
    Hyphen-variant expansion emits multiple term_ids sharing the same
    billing_position so any variant form resolves to the credit without
    inflating character_cast_size (see
    [implementation/misc/helpers.py](../implementation/misc/helpers.py)
    ``expand_hyphen_variants``).

    Args:
        term_ids: Character term IDs for every ``(name, variant)`` pair.
        billing_positions: 1-based cast-edge position matching each
            term_id back to its originating distinct character.
        movie_id: Movie ID that owns all postings.
        character_cast_size: Number of distinct characters (pre-variant)
            in the film. Applied uniformly to every inserted row.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    if not term_ids:
        return
    if len(term_ids) != len(billing_positions):
        raise ValueError(
            "batch_insert_character_postings: term_ids and "
            "billing_positions lengths differ."
        )
    query = """
    INSERT INTO lex.inv_character_postings
        (term_id, movie_id, billing_position, character_cast_size)
    SELECT unnest(%s::bigint[]), %s, unnest(%s::int[]), %s
    ON CONFLICT (term_id, movie_id) DO NOTHING;
    """
    await _execute_on_conn(
        conn, query, (term_ids, movie_id, billing_positions, character_cast_size)
    )


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


# ---------------------------------------------------------------------------
# V2 similar-movies materialized views.
# ---------------------------------------------------------------------------
# trait_kind discriminator values for public.mv_trait_idf. Kept in sync
# with the WITH-clause literals in db/init/01_create_postgres_tables.sql
# (the unified mv_trait_idf definition). Lane code filters this MV by
# trait_kind to read the right per-trait-family IDF table.
TRAIT_KIND_OVERALL_KEYWORD = 1
TRAIT_KIND_CONCEPT_TAG = 2
TRAIT_KIND_TMDB_GENRE = 3
TRAIT_KIND_SOURCE_MATERIAL = 4


async def refresh_director_strength() -> None:
    """
    Refresh the public.mv_director_strength materialized view concurrently.

    Backs the V2 similar-movies director-auteur lane: per-director
    percentile-rank of (0.8 * mean popularity) + (0.2 * mean reception)
    across films on lex.inv_director_postings, restricted to directors
    with >= 2 films. The director_signature anchor type triggers when
    director_strength >= 0.80.

    Must run AFTER refresh_movie_popularity_scores() — the underlying
    SQL JOINs public.mv_popularity_percentile.
    """
    await _execute_write(
        "REFRESH MATERIALIZED VIEW CONCURRENTLY public.mv_director_strength;"
    )


async def refresh_franchise_confidence() -> None:
    """
    Refresh the public.mv_franchise_confidence materialized view concurrently.

    Backs the V2 similar-movies franchise lane: per-lineage confidence
    (mean strength score) and consistency (1 - clamp(2*stddev, 0, 1)).
    Lineages with high confidence + high consistency get additive lane
    exposure; lower-confidence lineages drop to a small multiplicative
    nudge to avoid surfacing direct-to-DVD spinoffs in the top results.

    Must run AFTER refresh_movie_popularity_scores() — the underlying
    SQL JOINs public.mv_popularity_percentile.
    """
    await _execute_write(
        "REFRESH MATERIALIZED VIEW CONCURRENTLY public.mv_franchise_confidence;"
    )


async def refresh_trait_idf() -> None:
    """
    Refresh the public.mv_trait_idf materialized view concurrently.

    Unified IDF table across four trait families (overall_keyword /
    concept_tag / tmdb_genre / source_material) used by the V2 themes
    lane (multi-anchor), source lane (single + multi), and the
    medium-IDF retrieval gate. IDF is normalized log(N/df)/log(N) so
    values stay in [0, 1] regardless of catalog size.

    Independent of the popularity / director / franchise MVs — only
    requires that movie_card upserts for the current ingest batch
    have completed.
    """
    await _execute_write(
        "REFRESH MATERIALIZED VIEW CONCURRENTLY public.mv_trait_idf;"
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
    lineage_entry_ids: Sequence[int],
    shared_universe_entry_ids: Sequence[int],
    subgroup_entry_ids: Sequence[int],
    conn=None,
) -> None:
    """
    Stamp the franchise entry-id arrays on movie_card.

    Exists as a standalone helper so the backfill script can set all
    three columns without rewriting every movie_card field. Ingest-time
    callers pass the arrays directly to ``upsert_movie_card`` and do not
    need this function.

    lineage_entry_ids and shared_universe_entry_ids are stored in separate
    columns so stage-3 can score a lineage match higher than a
    universe-only match when prefer_lineage is set on the query spec.

    Raises ``ValueError`` if no movie_card row exists for ``movie_id``.
    Without this guard the UPDATE would silently no-op, hiding upstream
    ordering bugs where the card hasn't been written yet.
    """
    query = """
    UPDATE public.movie_card
    SET lineage_entry_ids = %s,
        shared_universe_entry_ids = %s,
        subgroup_entry_ids = %s,
        updated_at = now()
    WHERE movie_id = %s
    """
    params = (
        list(lineage_entry_ids),
        list(shared_universe_entry_ids),
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
            f"lineage_entry_ids / shared_universe_entry_ids / "
            f"subgroup_entry_ids columns can be stamped."
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

    The lineage_entry_ids / shared_universe_entry_ids / subgroup_entry_ids
    columns on movie_card are rewritten by upsert_movie_card on next
    ingest, so they need no separate clear here; and lex.franchise_entry /
    lex.franchise_token are registry-wide (not movie-scoped), so nothing
    else to delete.

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
    title_normalized: str,
    budget_bucket: Optional[str] = None,
    box_office_bucket: Optional[str] = None,
    release_format: int = 0,
    production_company_ids: Sequence[int] = (),
    lineage_entry_ids: Sequence[int] = (),
    shared_universe_entry_ids: Sequence[int] = (),
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
        title_normalized: Normalized form of ``title`` via
            :func:`implementation.misc.helpers.normalize_string`. Powers
            Stage 3 title_pattern ILIKE matching with symmetric
            query-time normalization.
        budget_bucket: Era-adjusted budget classification ('small', 'large', or None for mid-range/unknown).
        box_office_bucket: Box office classification ('hit', 'flop', or None for ambiguous/unknown).
        release_format: ReleaseFormat int id (schemas.enums.ReleaseFormat). 0 = UNKNOWN
            (default) for IMDB title types outside the supported set or missing data.
        production_company_ids: lex.production_company IDs this movie credits.
            Empty sequence is allowed (movies with no IMDB production_companies).
        lineage_entry_ids: lex.franchise_entry IDs resolved from the movie's
            lineage string (0 or 1 element). Empty sequence when the movie
            has no franchise metadata or no lineage.
        shared_universe_entry_ids: lex.franchise_entry IDs resolved from the
            movie's shared_universe string (0 or 1 element). Stored
            separately from lineage so stage-3 can score lineage matches
            higher than universe-only matches when prefer_lineage is set.
        subgroup_entry_ids: lex.franchise_entry IDs resolved from each element
            of recognized_subgroups. Empty sequence when there are no subgroups.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    query = """
    INSERT INTO public.movie_card (
        movie_id, title, title_normalized, poster_url, release_ts, runtime_minutes,
        maturity_rank, genre_ids, watch_offer_keys, audio_language_ids, country_of_origin_ids,
        source_material_type_ids, keyword_ids, concept_tag_ids, award_ceremony_win_ids,
        imdb_vote_count, reception_score, budget_bucket, box_office_bucket, release_format,
        production_company_ids, lineage_entry_ids, shared_universe_entry_ids, subgroup_entry_ids,
        created_at, updated_at
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now(), now())
    ON CONFLICT (movie_id) DO UPDATE SET
        title = EXCLUDED.title,
        title_normalized = EXCLUDED.title_normalized,
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
        release_format = EXCLUDED.release_format,
        production_company_ids = EXCLUDED.production_company_ids,
        lineage_entry_ids = EXCLUDED.lineage_entry_ids,
        shared_universe_entry_ids = EXCLUDED.shared_universe_entry_ids,
        subgroup_entry_ids = EXCLUDED.subgroup_entry_ids,
        updated_at = now();
    """
    params = (
        movie_id,
        title,
        title_normalized,
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
        release_format,
        list(production_company_ids),
        list(lineage_entry_ids),
        list(shared_universe_entry_ids),
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

    # Title-token scoring was removed with the v1 title-token infrastructure.
    # Stage 3 title_pattern lookups now ILIKE against movie_card.title_normalized
    # directly (fetch_movie_ids_with_title_like), and any remaining v1 callers
    # that still pass title_searches receive an empty title_scores_by_search.
    _ = title_searches

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


async def fetch_movie_card_row(movie_id: int) -> dict | None:
    """Fetch a single `movie_card` row for the `/movie_details` endpoint.

    Single-row variant of `fetch_movie_cards` used by the detail endpoint
    to (a) confirm the movie is in our index (404 fast-path) and (b)
    surface `reception_score` for the response payload. Returns the row
    dict or ``None`` when the movie is not in `movie_card`.
    """
    rows = await fetch_movie_cards([movie_id])
    return rows[0] if rows else None


async def fetch_movie_card_summaries(movie_ids: list[int]) -> list[MovieCard]:
    """Bulk-fetch `MovieCard` API response objects for the given tmdb_ids.

    Thin projection over `fetch_movie_cards` that keeps only the four
    fields the frontend needs (title, release_date, tmdb_id, poster_url)
    and converts `release_ts` (Unix seconds) into an ISO YYYY-MM-DD
    string. Preserves the caller's input order; tmdb_ids absent from
    `movie_card` are silently dropped — the API treats them as
    not-yet-ingested rather than as errors.

    Single SQL query (delegated to `fetch_movie_cards`) — never
    per-candidate, per the cross-codebase invariant.
    """
    if not movie_ids:
        return []

    rows = await fetch_movie_cards(movie_ids)
    by_id = {row["movie_id"]: row for row in rows}

    cards: list[MovieCard] = []
    for movie_id in movie_ids:
        row = by_id.get(movie_id)
        if row is None:
            continue
        cards.append(
            MovieCard(
                tmdb_id=int(row["movie_id"]),
                title=row.get("title"),
                release_date=_release_ts_to_date(row.get("release_ts")),
                poster_url=row.get("poster_url"),
                maturity_rating=_maturity_rank_to_label(row.get("maturity_rank")),
            )
        )
    return cards


def _release_ts_to_date(release_ts: int | None) -> str | None:
    """Convert a Unix-seconds timestamp to an ISO YYYY-MM-DD string."""
    if release_ts is None:
        return None
    return datetime.fromtimestamp(release_ts, tz=timezone.utc).strftime("%Y-%m-%d")


# Rank -> canonical MPAA display label, derived from the MaturityRating enum
# so the mapping stays in lockstep with the source of truth in enums.py.
# UNRATED is intentionally excluded — the frontend renders no rating segment.
_MATURITY_RANK_TO_LABEL: dict[int, str] = {
    member.maturity_rank: member.value.upper()
    for member in MaturityRating
    if member is not MaturityRating.UNRATED
}


def _maturity_rank_to_label(maturity_rank: int | None) -> str | None:
    """Convert a stored `maturity_rank` to its canonical MPAA label.

    Returns None for null ranks, the UNRATED sentinel, and any
    unknown rank value — all of which signal "no rating to show".
    Ingest normalizes UNRATED to NULL before insert, but we still
    guard here so legacy/stray rows don't surface a bogus label.
    """
    if maturity_rank is None:
        return None
    return _MATURITY_RANK_TO_LABEL.get(maturity_rank)


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


async def fetch_movie_ids_matching_filters(
    movie_ids: Iterable[int],
    metadata_filters: MetadataFilters,
) -> set[int]:
    """Return the subset of ``movie_ids`` whose movie_card row satisfies
    the active MetadataFilters.

    Helper for fetch paths that pull candidate movie_ids from a source
    other than Postgres (e.g. Redis-backed trending) and need to apply
    the user hard filter as a separate round trip. Single statement,
    no per-candidate calls.

    Returns an empty set if filters is inactive *and* the caller passed
    an empty list — callers that hold a non-empty list with an inactive
    filter should skip this helper entirely (it would return every input
    id, which is the no-op outcome).
    """
    ids_list = [int(m) for m in movie_ids]
    if not ids_list:
        return set()
    if not metadata_filters.is_active:
        return set(ids_list)
    conditions, params = _build_movie_card_conditions(metadata_filters)
    where_clause = " AND ".join(conditions) if conditions else "TRUE"
    query = f"""
        SELECT movie_id
        FROM public.movie_card
        WHERE movie_id = ANY(%s::bigint[]) AND {where_clause}
    """
    rows = await _execute_read(query, (ids_list, *params))
    return {row[0] for row in rows}


async def fetch_browse_seed_ids(
    *,
    limit: int,
    metadata_filters: Optional[MetadataFilters] = None,
) -> list[int]:
    """Top `limit` movie_ids ordered by the temporary browse fallback.

    This is the trending-endpoint fetch path (no candidate-generating
    LLM call). When the UI supplies hard filters, they're folded into
    the WHERE clause so the popularity-ordered seed respects them.
    """
    filter_clause, filter_params = _build_direct_movie_card_filter_clause(metadata_filters)
    query = f"""
        SELECT movie_id
        FROM public.movie_card
        WHERE TRUE{filter_clause}
        ORDER BY popularity_score DESC NULLS LAST, movie_id DESC
        LIMIT %s
    """
    rows = await _execute_read(query, (*filter_params, limit))
    return [row[0] for row in rows]


async def fetch_quality_popularity_seed(
    *,
    limit: int,
    metadata_filters: Optional[MetadataFilters] = None,
) -> list[int]:
    """Top `limit` movie_ids ordered by popularity * reception product.

    Used by the stage-3 orchestrator's no-inclusion fallback path:
    when only exclusion fired (no inclusion candidates and no
    preferences to use as candidate generators), we still need a
    seed pool that downstream rerankers can prune via exclusion and
    rerank via preferences / implicit priors. The product ordering
    yields a "well-known *and* well-received" pool — strict on both
    axes, unlike a popularity-only seed which leans toward famous
    but poorly-reviewed films.

    NULL handling: COALESCE both signals to 0 so movies with one
    missing signal sort to the bottom rather than out of the result
    set entirely.

    UI hard filters fold into the WHERE clause when supplied — this
    is one of the no-LLM fallback paths so the filter must apply
    here, not later via post-filtering.
    """
    filter_clause, filter_params = _build_direct_movie_card_filter_clause(metadata_filters)
    query = f"""
        SELECT movie_id
        FROM public.movie_card
        WHERE TRUE{filter_clause}
        ORDER BY (COALESCE(popularity_score, 0)
                  * COALESCE(reception_score, 0)) DESC,
                 movie_id DESC
        LIMIT %s
    """
    rows = await _execute_read(query, (*filter_params, limit))
    return [row[0] for row in rows]


async def fetch_neutral_reranker_seed_ids(
    *,
    limit: int = NEUTRAL_RERANKER_SEED_LIMIT,
    metadata_filters: Optional[MetadataFilters] = None,
) -> list[int]:
    """Top `limit` movie_ids for reranker-only fallback seeding.

    Orders by the deterministic neutral prior:

        NEUTRAL_RERANKER_SEED_POPULARITY_WEIGHT
            * normalized_popularity_score
      + NEUTRAL_RERANKER_SEED_RECEPTION_WEIGHT
            * normalized_reception_score

    `popularity_score` is already stored on a [0, 1] scale.
    `reception_score` is stored on a 0-100 scale, so normalize by
    dividing by 100 and clamp both components defensively.

    This is the **no-LLM fallback path** the user explicitly called
    out — when zero candidate generators fire, this seed is the only
    pool downstream rerankers see. The hard filter must apply here
    or filters get silently bypassed on this path.
    """
    filter_clause, filter_params = _build_direct_movie_card_filter_clause(metadata_filters)
    query = f"""
        SELECT movie_id
        FROM public.movie_card
        WHERE TRUE{filter_clause}
        ORDER BY (
            %s * COALESCE(
                LEAST(1.0, GREATEST(0.0, popularity_score)),
                0.0
            )
            +
            %s * COALESCE(
                LEAST(1.0, GREATEST(0.0, reception_score / 100.0)),
                0.0
            )
        ) DESC,
        movie_id DESC
        LIMIT %s
    """
    rows = await _execute_read(
        query,
        (
            *filter_params,
            NEUTRAL_RERANKER_SEED_POPULARITY_WEIGHT,
            NEUTRAL_RERANKER_SEED_RECEPTION_WEIGHT,
            limit,
        ),
    )
    return [row[0] for row in rows]


async def fetch_quality_popularity_signals(
    movie_ids: list[int],
) -> dict[int, tuple[float | None, float | None]]:
    """Bulk fetch (popularity_score, reception_score) for the implicit prior.

    Returns a dict keyed by movie_id with a (popularity, reception)
    tuple value. Missing movies are simply absent from the dict;
    missing column values surface as None and the implicit-prior
    reranker treats missing axis data as no effect.
    """
    if not movie_ids:
        return {}

    query = """
        SELECT movie_id, popularity_score, reception_score
        FROM public.movie_card
        WHERE movie_id = ANY(%s::bigint[])
    """

    rows = await _execute_read(query, (movie_ids,))
    return {row[0]: (row[1], row[2]) for row in rows}


async def fetch_lineage_mainline_signals(
    movie_ids: list[int],
) -> dict[int, tuple[bool, int]]:
    """Bulk fetch (is_spinoff, release_format) for the character-franchise mainline-vs-ancillary split.

    Returns a dict keyed by movie_id with `(is_spinoff, release_format)`.
    `release_format` is the SMALLINT id stored on movie_card
    (`ReleaseFormat.<member>.release_format_id`). Missing movies are
    absent from the dict.

    `is_spinoff` is LEFT-JOINed from movie_franchise_metadata and
    defaults to False when the movie has no franchise metadata row —
    a movie without any franchise structure cannot be a spinoff of
    anything, so absence reads as "not a spinoff."

    Used by the character-franchise tier construction to split the
    lineage match-set into mainline (NOT is_spinoff AND
    release_format=MOVIE) and ancillary buckets, so very-prominent
    character appearances outside the lineage can be inserted between
    them. See search_v2/character_franchise_search.py.
    """
    if not movie_ids:
        return {}

    query = """
        SELECT mc.movie_id,
               COALESCE(mfm.is_spinoff, FALSE) AS is_spinoff,
               mc.release_format
        FROM public.movie_card mc
        LEFT JOIN public.movie_franchise_metadata mfm
          ON mfm.movie_id = mc.movie_id
        WHERE mc.movie_id = ANY(%s::bigint[])
    """

    rows = await _execute_read(query, (movie_ids,))
    return {row[0]: (row[1], row[2]) for row in rows}


# =========================================
#     SIMILAR-MOVIES FLOW READ HELPERS
# =========================================
#
# These helpers are intentionally narrow and read-only. They support
# search_v2.similar_movies without changing the standard Stage-4 search
# execution path.


async def fetch_similarity_signal_rows(movie_ids: list[int]) -> dict[int, dict]:
    """Bulk fetch the Postgres signals used by the similar-movies flow.

    Returns one dict per movie_id with movie_card display fields, similarity
    lane arrays, reception/popularity signals, and the global popularity
    percentile from ``public.mv_popularity_percentile``.
    """
    if not movie_ids:
        return {}

    query = """
        SELECT
            mc.movie_id,
            mc.title,
            mc.poster_url,
            mc.release_ts,
            mc.reception_score,
            mc.popularity_score,
            mc.imdb_vote_count,
            mvp.percentile AS popularity_percentile,
            mc.source_material_type_ids,
            mc.production_company_ids,
            mc.lineage_entry_ids,
            mc.shared_universe_entry_ids,
            mc.subgroup_entry_ids,
            mc.award_ceremony_win_ids,
            mc.keyword_ids,
            mc.concept_tag_ids,
            mc.genre_ids,
            mc.runtime_minutes
        FROM public.movie_card mc
        LEFT JOIN public.mv_popularity_percentile mvp
          ON mvp.movie_id = mc.movie_id
        WHERE mc.movie_id = ANY(%s::bigint[])
    """
    columns = [
        "movie_id",
        "title",
        "poster_url",
        "release_ts",
        "reception_score",
        "popularity_score",
        "imdb_vote_count",
        "popularity_percentile",
        "source_material_type_ids",
        "production_company_ids",
        "lineage_entry_ids",
        "shared_universe_entry_ids",
        "subgroup_entry_ids",
        "award_ceremony_win_ids",
        "keyword_ids",
        "concept_tag_ids",
        "genre_ids",
        "runtime_minutes",
    ]
    rows = await _execute_read(query, (movie_ids,))
    return {row[0]: dict(zip(columns, row)) for row in rows}


async def fetch_director_term_ids_for_movies(
    movie_ids: list[int],
) -> dict[int, set[int]]:
    """Return director lexical term IDs for each supplied movie."""
    if not movie_ids:
        return {}

    query = """
        SELECT movie_id, term_id
        FROM lex.inv_director_postings
        WHERE movie_id = ANY(%s::bigint[])
    """
    rows = await _execute_read(query, (movie_ids,))
    out: dict[int, set[int]] = {}
    for movie_id, term_id in rows:
        out.setdefault(movie_id, set()).add(term_id)
    return out


async def fetch_director_movie_terms(
    term_ids: set[int],
    *,
    restrict_movie_ids: set[int] | None = None,
) -> dict[int, set[int]]:
    """Fetch movies and matching director term IDs for a director-term set."""
    if not term_ids:
        return {}

    params: list = [list(term_ids)]
    restrict_clause = ""
    if restrict_movie_ids is not None:
        if not restrict_movie_ids:
            return {}
        restrict_clause = " AND movie_id = ANY(%s::bigint[])"
        params.append(list(restrict_movie_ids))

    query = f"""
        SELECT movie_id, term_id
        FROM lex.inv_director_postings
        WHERE term_id = ANY(%s::bigint[]){restrict_clause}
    """
    rows = await _execute_read(query, params)
    out: dict[int, set[int]] = {}
    for movie_id, term_id in rows:
        out.setdefault(movie_id, set()).add(term_id)
    return out


async def fetch_production_company_ids_by_normalized_strings(
    normalized_strings: list[str],
) -> dict[str, int]:
    """Resolve normalized production-company strings to company IDs."""
    if not normalized_strings:
        return {}

    query = """
        SELECT normalized_string, production_company_id
        FROM lex.production_company
        WHERE normalized_string = ANY(%s::text[])
    """
    rows = await _execute_read(query, (normalized_strings,))
    return {row[0]: row[1] for row in rows}


async def fetch_similarity_source_candidates(
    source_material_type_ids: set[int],
) -> set[int]:
    """Movie IDs whose source-material type array overlaps the input set."""
    if not source_material_type_ids:
        return set()

    query = """
        SELECT movie_id
        FROM public.movie_card
        WHERE source_material_type_ids && %s::int[]
    """
    rows = await _execute_read(query, (list(source_material_type_ids),))
    return {row[0] for row in rows}


async def fetch_similarity_franchise_candidates(
    *,
    lineage_entry_ids: set[int],
    shared_universe_entry_ids: set[int],
    subgroup_entry_ids: set[int],
) -> set[int]:
    """Movie IDs overlapping any supplied franchise lineage/universe signal."""
    clauses: list[str] = []
    params: list = []
    if lineage_entry_ids:
        clauses.append(
            "(lineage_entry_ids && %s::bigint[] "
            "OR shared_universe_entry_ids && %s::bigint[])"
        )
        entry_ids = list(lineage_entry_ids)
        params.extend([entry_ids, entry_ids])
    if shared_universe_entry_ids:
        clauses.append(
            "(lineage_entry_ids && %s::bigint[] "
            "OR shared_universe_entry_ids && %s::bigint[])"
        )
        entry_ids = list(shared_universe_entry_ids)
        params.extend([entry_ids, entry_ids])
    if subgroup_entry_ids:
        clauses.append("subgroup_entry_ids && %s::bigint[]")
        params.append(list(subgroup_entry_ids))

    if not clauses:
        return set()

    where_clause = " OR ".join(f"({clause})" for clause in clauses)
    query = f"SELECT movie_id FROM public.movie_card WHERE {where_clause}"
    rows = await _execute_read(query, params)
    return {row[0] for row in rows}


async def fetch_similarity_quality_candidates(
    *,
    bucket: str,
    limit: int,
) -> set[int]:
    """Fetch an independent quality/reception candidate lane.

    ``bucket`` accepts ``"cult_garbage"`` or ``"prestige"``. Middle-bucket
    anchors do not use the quality lane in the similar-movies flow.
    """
    if bucket not in {"cult_garbage", "prestige"}:
        raise ValueError(f"unsupported similarity quality bucket: {bucket!r}")

    if bucket == "cult_garbage":
        query = """
            SELECT mc.movie_id
            FROM public.movie_card mc
            JOIN public.mv_popularity_percentile mvp
              ON mvp.movie_id = mc.movie_id
            WHERE mc.reception_score <= 50
              AND mvp.percentile >= 0.75
            ORDER BY
              (0.5 * LEAST(1.0, GREATEST(0.0, (50 - mc.reception_score) / 30.0))
               + 0.5 * LEAST(1.0, GREATEST(0.0, (mvp.percentile - 0.75) / 0.20))) DESC,
              mc.movie_id ASC
            LIMIT %s
        """
        rows = await _execute_read(query, (limit,))
        return {row[0] for row in rows}

    non_razzie_ids = [
        c.ceremony_id for c in AwardCeremony if c is not AwardCeremony.RAZZIE
    ]
    query = """
        SELECT mc.movie_id
        FROM public.movie_card mc
        LEFT JOIN public.mv_popularity_percentile mvp
          ON mvp.movie_id = mc.movie_id
        WHERE mc.reception_score >= 75
           OR EXISTS (
                SELECT 1
                FROM unnest(mc.award_ceremony_win_ids) AS win_id
                WHERE win_id = ANY(%s::smallint[])
           )
        ORDER BY
          (0.75 * LEAST(1.0, GREATEST(0.0, (COALESCE(mc.reception_score, 0) - 75) / 20.0))
           + 0.25 * GREATEST(
                LEAST(1.0, GREATEST(0.0, (COALESCE(mvp.percentile, 0) - 0.50) / 0.30)),
                CASE
                  WHEN EXISTS (
                    SELECT 1
                    FROM unnest(mc.award_ceremony_win_ids) AS win_id
                    WHERE win_id = ANY(%s::smallint[])
                  ) THEN 1.0
                  ELSE 0.0
                END
             )) DESC,
          mc.movie_id ASC
        LIMIT %s
    """
    rows = await _execute_read(query, (non_razzie_ids, non_razzie_ids, limit))
    return {row[0] for row in rows}


# V3.3.2 shape classification: picture-level prestige tags. A nom or
# win in any of these categories at a non-Razzie ceremony is the
# threshold that lowers the prestige reception floor from 80 to 65.
# - 103 = BEST_PICTURE_ANY (rollup that catches all best-picture
#   variants including drama/comedy-musical/action/horror-scifi/
#   crime-adventure splits)
# - 9   = DIRECTOR (excludes DEBUT_DIRECTOR=10 and ASSISTANT_DIRECTOR=11
#   on purpose — debut director is a different signal, assistant is
#   a craft credit)
SHAPE_PICTURE_LEVEL_TAG_IDS: tuple[int, ...] = (103, 9)

# V3.3.2 shape classification: bad-Razzie WIN leaves. A win in any of
# these specific WORST_* categories at the Razzie ceremony lowers the
# poorly-rated reception ceiling from 50 to 60. WORST_OTHER (id 58) is
# excluded — it's a catchall that could plausibly include the Razzie
# Redeemer Award (which is itself positive recognition for
# filmmakers who reformed). Excluding it is a heuristic; if/when the
# scraping schema starts distinguishing the redeemer, this list can
# become more inclusive.
SHAPE_BAD_RAZZIE_LEAF_IDS: tuple[int, ...] = (
    46,  # WORST_PICTURE
    47,  # WORST_LEAD_ACTOR
    48,  # WORST_LEAD_ACTRESS
    49,  # WORST_SUPPORTING_ACTOR
    50,  # WORST_SUPPORTING_ACTRESS
    51,  # WORST_DIRECTOR
    52,  # WORST_SCREENPLAY
    53,  # WORST_REMAKE_OR_SEQUEL
    54,  # WORST_CAST_OR_COUPLE
    55,  # WORST_MUSIC
    56,  # WORST_VISUAL_EFFECTS
    57,  # WORST_DEBUT_OR_NEWCOMER
)


@dataclass(frozen=True, slots=True)
class SimilarityAwardSignals:
    """Award signals consumed by the V2 quality lane and V3.3.2 shape
    classifier.

    `non_razzie_score` powers the prestige-bucket formula; `razzie_score`
    powers the cult_garbage formula. Lumped into one struct so callers
    fetch both with a single query rather than firing two parallel reads.

    V3.3.2 added two flags driving the shape classifier's award-aware
    threshold shifts:
      - `has_picture_level_signal`: a Best Picture or Director nom/win
        at any non-Razzie ceremony. Lowers prestige reception floor
        from 80 to 65.
      - `has_bad_razzie_win`: a Razzie WIN in one of the WORST_*
        categories (excluding WORST_OTHER to give the Razzie Redeemer
        Award benefit of the doubt). Raises poorly-rated reception
        ceiling from 50 to 60.
    """
    non_razzie_score: float            # 1.0 win, 0.75 nom, else 0.0
    razzie_score: float                # 1.0 if any Razzie nom/win, else 0.0
    has_picture_level_signal: bool     # V3.3.2: BP or Director nom/win at non-Razzie ceremony
    has_bad_razzie_win: bool           # V3.3.2: Razzie WIN in a WORST_* leaf (excluding WORST_OTHER)


async def fetch_similarity_award_signals(
    movie_ids: list[int],
) -> dict[int, SimilarityAwardSignals]:
    """Return non-Razzie + Razzie award signals for the supplied movies.

    Single SQL pass over `public.movie_awards` with conditional aggregates
    so cult_garbage / prestige scoring can read both sides without firing
    two queries. V3.3.2 added picture-level and bad-Razzie-win flags
    used by the shape classifier. Movies absent from the result have no
    relevant award rows.
    """
    if not movie_ids:
        return {}

    query = """
        SELECT
            movie_id,
            BOOL_OR(outcome_id = %s AND ceremony_id <> %s) AS has_non_razzie_win,
            BOOL_OR(outcome_id = %s AND ceremony_id <> %s) AS has_non_razzie_nom,
            BOOL_OR(ceremony_id = %s)                      AS has_razzie,
            BOOL_OR(
                ceremony_id <> %s
                AND category_tag_ids && %s::int[]
            ) AS has_picture_level_signal,
            BOOL_OR(
                ceremony_id = %s
                AND outcome_id = %s
                AND category_tag_ids && %s::int[]
            ) AS has_bad_razzie_win
        FROM public.movie_awards
        WHERE movie_id = ANY(%s::bigint[])
        GROUP BY movie_id
    """
    rows = await _execute_read(
        query,
        (
            AwardOutcome.WINNER.outcome_id,
            AwardCeremony.RAZZIE.ceremony_id,
            AwardOutcome.NOMINEE.outcome_id,
            AwardCeremony.RAZZIE.ceremony_id,
            AwardCeremony.RAZZIE.ceremony_id,
            AwardCeremony.RAZZIE.ceremony_id,
            list(SHAPE_PICTURE_LEVEL_TAG_IDS),
            AwardCeremony.RAZZIE.ceremony_id,
            AwardOutcome.WINNER.outcome_id,
            list(SHAPE_BAD_RAZZIE_LEAF_IDS),
            movie_ids,
        ),
    )
    out: dict[int, SimilarityAwardSignals] = {}
    for (
        movie_id,
        has_win,
        has_nom,
        has_razzie,
        has_picture_level,
        has_bad_razzie_win,
    ) in rows:
        if has_win:
            non_razzie = 1.0
        elif has_nom:
            non_razzie = 0.75
        else:
            non_razzie = 0.0
        razzie = 1.0 if has_razzie else 0.0
        if (
            non_razzie > 0.0
            or razzie > 0.0
            or has_picture_level
            or has_bad_razzie_win
        ):
            out[movie_id] = SimilarityAwardSignals(
                non_razzie_score=non_razzie,
                razzie_score=razzie,
                has_picture_level_signal=bool(has_picture_level),
                has_bad_razzie_win=bool(has_bad_razzie_win),
            )
    return out


async def fetch_director_strengths(term_ids: list[int]) -> dict[int, float]:
    """Return {term_id: director_strength in [0,1]} from mv_director_strength.

    Directors with <2 cataloged films are absent from the MV (they can't be
    "matched through" because the anchor itself would be the only film).
    """
    if not term_ids:
        return {}

    query = """
        SELECT term_id, director_strength
        FROM public.mv_director_strength
        WHERE term_id = ANY(%s::bigint[])
    """
    rows = await _execute_read(query, (term_ids,))
    return {row[0]: float(row[1]) for row in rows}


async def fetch_franchise_confidence(
    lineage_entry_ids: list[int],
) -> dict[int, tuple[float, float]]:
    """Return {lineage_entry_id: (franchise_confidence, franchise_consistency)}.

    Single-film lineages have consistency=1.0 (no spread to measure). Used
    by the V2 franchise lane to choose additive (high-confidence) vs.
    multiplicative (low-confidence) behavior.
    """
    if not lineage_entry_ids:
        return {}

    query = """
        SELECT lineage_entry_id, franchise_confidence, franchise_consistency
        FROM public.mv_franchise_confidence
        WHERE lineage_entry_id = ANY(%s::bigint[])
    """
    rows = await _execute_read(query, (lineage_entry_ids,))
    return {row[0]: (float(row[1]), float(row[2])) for row in rows}


async def fetch_trait_idfs(
    pairs: list[tuple[int, int]],
) -> dict[tuple[int, int], float]:
    """Return {(trait_kind, trait_id): idf} from mv_trait_idf.

    `pairs` are (trait_kind, trait_id) tuples; trait_kind values are the
    TRAIT_KIND_* constants in this module. Missing pairs are omitted from
    the result and treated as idf=0 by lane callers (which discards them
    naturally via the IDF formulas — log(N/df)/log(N) is 0 when df==N).
    """
    if not pairs:
        return {}

    # Split into two parallel arrays so the query stays a single round-trip
    # rather than ((kind, id) IN (...)) which Postgres can't index nicely.
    # `trait_id` in mv_trait_idf is INT (every UNION branch yields INT[]
    # column elements), so the parameter array stays INT[] for type fidelity.
    kinds = [int(p[0]) for p in pairs]
    ids = [int(p[1]) for p in pairs]
    query = """
        SELECT trait_kind, trait_id, idf
        FROM public.mv_trait_idf
        WHERE (trait_kind, trait_id) IN (
            SELECT UNNEST(%s::smallint[]), UNNEST(%s::int[])
        )
    """
    rows = await _execute_read(query, (kinds, ids))
    return {(int(row[0]), int(row[1])): float(row[2]) for row in rows}


async def fetch_movie_ids_by_overall_keywords(
    keyword_ids: list[int],
    *,
    metadata_filters: Optional[MetadataFilters] = None,
) -> set[int]:
    """Movie IDs whose `keyword_ids` array overlaps the supplied set.

    Used by the V2 single-anchor selective rare-medium retrieval lane so
    e.g. a stop-motion anchor surfaces other stop-motion films even when
    the centroid-driven shape lane misses them.

    Args:
        keyword_ids: Keyword IDs to overlap against ``movie_card.keyword_ids``.
        metadata_filters: Optional user-supplied hard filters. When active,
            the conditions are AND-folded into the same WHERE clause (no
            self-IN subquery — the FROM is already movie_card).
    """
    if not keyword_ids:
        return set()

    filter_clause, filter_params = _build_direct_movie_card_filter_clause(metadata_filters)
    query = f"""
        SELECT movie_id
        FROM public.movie_card
        WHERE keyword_ids && %s::int[]{filter_clause}
    """
    rows = await _execute_read(query, (list(keyword_ids), *filter_params))
    return {row[0] for row in rows}


async def fetch_movie_ids_by_themes_recall(
    keyword_ids: list[int],
    concept_tag_ids: list[int],
    genre_ids: list[int],
    *,
    single_idf_threshold: float = 0.55,
    combo_sum_threshold: float = 0.50,
    combo_sum_min_idf: float = 0.30,
    metadata_filters: Optional[MetadataFilters] = None,
) -> set[int]:
    """V3.2 themes-recall: movie IDs whose shared anchor traits qualify
    by single-trait rarity OR combined moderate+high tier IDF sum.

    Two recall paths in one SQL aggregate:
      - Single-trait gate: at least one shared trait with idf >=
        single_idf_threshold. Catches the "Manhattan Project" case
        where one super-rare keyword uniquely identifies the candidate.
      - Combo-sum gate: sum of shared trait IDFs filtered to
        idf >= combo_sum_min_idf (V3.2 default 0.30 — moderate+high
        tier only) >= combo_sum_threshold. V3.1 used all-tier sum;
        smoke run showed this let in low-tier-overlap noise (obscure
        foreign films sharing common keywords with Barbie). The
        moderate+high filter aligns the recall gate's tier semantics
        with the rare_keyword combo bonus inside scoring. Catches the
        "comedy + satire + female-lead" case where multiple moderate
        tags coalesce, without admitting "common-tag accumulation"
        candidates.

    Trait kinds resolved against `mv_trait_idf` use the constants
    defined at module top (TRAIT_KIND_OVERALL_KEYWORD = 1,
    TRAIT_KIND_CONCEPT_TAG = 2, TRAIT_KIND_TMDB_GENRE = 3). Trait pools
    are split by kind because the source columns on `movie_card` are
    separate arrays.

    Empty inputs short-circuit. Each kind contributes a UNION ALL leg
    only if its input list is non-empty — keeps the query plan tight
    on anchors that lack one or more trait families.
    """
    if not keyword_ids and not concept_tag_ids and not genre_ids:
        return set()

    # Build the UNION ALL dynamically so we don't scan posting lists
    # for kinds the anchor doesn't carry. Each leg follows the same
    # shape: pick movies whose array overlaps the anchor trait set,
    # join mv_trait_idf for IDFs, project (movie_id, idf) pairs.
    legs: list[str] = []
    params: list[object] = []
    if keyword_ids:
        legs.append(
            """
            SELECT m.movie_id, ti.idf
            FROM public.movie_card m, public.mv_trait_idf ti
            WHERE ti.trait_kind = 1
              AND ti.trait_id = ANY(%s::int[])
              AND m.keyword_ids && ARRAY[ti.trait_id]
            """
        )
        params.append(list(keyword_ids))
    if concept_tag_ids:
        legs.append(
            """
            SELECT m.movie_id, ti.idf
            FROM public.movie_card m, public.mv_trait_idf ti
            WHERE ti.trait_kind = 2
              AND ti.trait_id = ANY(%s::int[])
              AND m.concept_tag_ids && ARRAY[ti.trait_id]
            """
        )
        params.append(list(concept_tag_ids))
    if genre_ids:
        legs.append(
            """
            SELECT m.movie_id, ti.idf
            FROM public.movie_card m, public.mv_trait_idf ti
            WHERE ti.trait_kind = 3
              AND ti.trait_id = ANY(%s::int[])
              AND m.genre_ids && ARRAY[ti.trait_id]
            """
        )
        params.append(list(genre_ids))

    union_clause = " UNION ALL ".join(legs)
    # User hard-filter applied as a WHERE on the `shared` projection,
    # before GROUP BY — keeps the aggregation tight when the filter is
    # active, and is a no-op (empty string) when inactive so the query
    # plan is byte-identical to today on unfiltered calls.
    filter_clause, filter_params = _build_inline_movie_card_filter_clause(metadata_filters)
    # V3.2: SUM is filtered to traits at moderate+high tier
    # (idf >= combo_sum_min_idf, default 0.30) so common-tag
    # accumulation can't drag low-quality candidates into the pool.
    # MAX gate is untouched — a single rare trait still qualifies.
    query = f"""
        WITH shared AS ({union_clause})
        SELECT movie_id
        FROM shared
        WHERE TRUE{filter_clause}
        GROUP BY movie_id
        HAVING SUM(idf) FILTER (WHERE idf >= %s) >= %s
            OR MAX(idf) >= %s
    """
    params.extend(filter_params)
    params.extend([combo_sum_min_idf, combo_sum_threshold, single_idf_threshold])
    rows = await _execute_read(query, tuple(params))
    return {row[0] for row in rows}


async def fetch_similarity_top_billed_cast(
    movie_ids: list[int],
    *,
    top_k: int = 3,
) -> dict[int, set[int]]:
    """Return {movie_id: {term_id, ...}} for the top-K billed actors per movie.

    Source: lex.inv_actor_postings filtered to billing_position <= top_k.
    Drives the multi-anchor cast lane. `term_id` is the canonical actor
    identity already used by the lex layer.
    """
    if not movie_ids or top_k <= 0:
        return {}

    query = """
        SELECT movie_id, term_id
        FROM lex.inv_actor_postings
        WHERE movie_id = ANY(%s::bigint[])
          AND billing_position IS NOT NULL
          AND billing_position <= %s
    """
    rows = await _execute_read(query, (movie_ids, top_k))
    out: dict[int, set[int]] = {}
    for movie_id, term_id in rows:
        out.setdefault(int(movie_id), set()).add(int(term_id))
    return out


async def fetch_similarity_award_category_tags(
    movie_ids: list[int],
) -> dict[int, set[int]]:
    """Return {movie_id: union of category_tag_ids across that movie's awards}.

    Lumps wins and nominations together (per V2 spec V2.0). Used by the
    multi-anchor specific-award lane to compute repeated-tag cohesion + the
    tier-weighted candidate score.
    """
    if not movie_ids:
        return {}

    query = """
        SELECT movie_id, category_tag_ids
        FROM public.movie_awards
        WHERE movie_id = ANY(%s::bigint[])
          AND category_tag_ids IS NOT NULL
          AND array_length(category_tag_ids, 1) > 0
    """
    rows = await _execute_read(query, (movie_ids,))
    out: dict[int, set[int]] = {}
    for movie_id, tags in rows:
        if not tags:
            continue
        bucket = out.setdefault(int(movie_id), set())
        for tag_id in tags:
            bucket.add(int(tag_id))
    return out


async def fetch_movie_ids_by_term_ids(
    table: PostingTable,
    term_ids: list[int],
    *,
    metadata_filters: Optional[MetadataFilters] = None,
) -> set[int]:
    """
    Resolve posting term IDs into excluded movie IDs for one posting table.

    This helper supports global lexical exclusion: each EXCLUDE entity bucket
    resolves its own term IDs to movie IDs, and the caller unions those movie
    IDs into one cross-bucket exclusion set.

    Args:
        table: Posting table to resolve against.
        term_ids: Posting term IDs for one EXCLUDE bucket.
        metadata_filters: Optional user-supplied hard filters. Posting tables
            don't carry movie_card columns directly, so the filter is applied
            via an inline ``movie_id IN (SELECT movie_id FROM public.movie_card
            WHERE ...)`` subquery. No-op when filters is None / inactive.

    Returns:
        Set of movie IDs that contain at least one provided term ID.
    """
    if not term_ids:
        return set()

    filter_clause, filter_params = _build_inline_movie_card_filter_clause(metadata_filters)
    query = f"""
        SELECT DISTINCT movie_id
        FROM {table.value}
        WHERE term_id = ANY(%s::bigint[]){filter_clause}
    """
    search_results = await _execute_read(query, (term_ids, *filter_params))
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


async def fetch_movie_ids_by_brands(
    brand_ids: list[int],
    restrict_movie_ids: Optional[set[int]] = None,
    *,
    metadata_filters: Optional[MetadataFilters] = None,
) -> set[int]:
    """
    Resolve one or more ProductionBrand enum brand_ids to stamped movie IDs.

    Reads lex.inv_production_brand_postings with `brand_id = ANY(...)` so a
    single SQL round trip covers every brand the executor needs. The
    brand-path score is flat 1.0 per matched movie (the prominence column
    `first_matching_index` is stored but deliberately NOT used — see the
    studio endpoint design notes for why IMDB ordering is unreliable
    across regions).

    Args:
        brand_ids: List of ProductionBrand.brand_id values (SMALLINT).
            Caller dedupes; an empty list short-circuits to an empty set.
        restrict_movie_ids: Optional candidate-pool restriction for the
            preference / restrict-set path. Applied server-side.

    Returns:
        Union of movie IDs stamped with any of the given brands at ingest.
        Empty when none of the brands have postings or when brand_ids
        itself is empty.
    """
    if not brand_ids:
        return set()
    filter_clause, filter_params = _build_inline_movie_card_filter_clause(metadata_filters)
    if restrict_movie_ids is not None:
        if not restrict_movie_ids:
            return set()
        query = f"""
            SELECT movie_id
            FROM lex.inv_production_brand_postings
            WHERE brand_id = ANY(%s::smallint[])
              AND movie_id = ANY(%s::bigint[]){filter_clause}
        """
        rows = await _execute_read(
            query, (brand_ids, list(restrict_movie_ids), *filter_params)
        )
    else:
        query = f"""
            SELECT movie_id
            FROM lex.inv_production_brand_postings
            WHERE brand_id = ANY(%s::smallint[]){filter_clause}
        """
        rows = await _execute_read(query, (brand_ids, *filter_params))
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
    *,
    metadata_filters: Optional[MetadataFilters] = None,
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

    # FROM is movie_card already, so AND the filter conditions inline
    # rather than using a self-IN subquery.
    filter_clause, filter_params = _build_direct_movie_card_filter_clause(metadata_filters)
    id_list = list(production_company_ids)
    if restrict_movie_ids is not None:
        if not restrict_movie_ids:
            return set()
        query = f"""
            SELECT movie_id
            FROM public.movie_card
            WHERE production_company_ids && %s::bigint[]
              AND movie_id = ANY(%s::bigint[]){filter_clause}
        """
        rows = await _execute_read(query, (id_list, list(restrict_movie_ids), *filter_params))
    else:
        query = f"""
            SELECT movie_id
            FROM public.movie_card
            WHERE production_company_ids && %s::bigint[]{filter_clause}
        """
        rows = await _execute_read(query, (id_list, *filter_params))
    return {row[0] for row in rows}


# ===================================
#     MEDIA TYPE ENDPOINT HELPERS
# ===================================
#
# Read helpers dedicated to the step 3 media-type endpoint
# (search_v2/stage_3/media_type_query_execution.py). The SMALLINT
# release_format column on public.movie_card is the only data source —
# values match ReleaseFormat.release_format_id (0=UNKNOWN, 1=MOVIE,
# 2=TV_MOVIE, 3=SHORT, 4=VIDEO).


async def fetch_movie_ids_by_release_format(
    release_format_ids: list[int],
    restrict_movie_ids: Optional[set[int]] = None,
    *,
    metadata_filters: Optional[MetadataFilters] = None,
) -> set[int]:
    """
    Resolve a list of ReleaseFormat int ids to the matching movie IDs.

    Reads public.movie_card.release_format. Caller passes the int
    ReleaseFormat.release_format_id values (not enum members) so this
    helper stays pure SQL-layer; the executor handles the enum →
    int conversion.

    No index on release_format today — the LLM-facing values (TV_MOVIE,
    SHORT, VIDEO) are tiny minorities of the catalog so a sequential
    scan with a SMALLINT filter is acceptable. If this becomes a hot
    path, a partial index `WHERE release_format <> 1` is the obvious
    tuning.

    Args:
        release_format_ids: Non-empty list of ReleaseFormat.release_format_id
            ints to match. Empty list short-circuits to an empty set.
        restrict_movie_ids: Optional candidate-pool restriction for the
            preference / restrict-set path. Applied server-side.

    Returns:
        Set of movie IDs whose release_format is in the input list.
    """
    if not release_format_ids:
        return set()
    # FROM is movie_card already, so inline the filter conditions.
    filter_clause, filter_params = _build_direct_movie_card_filter_clause(metadata_filters)
    if restrict_movie_ids is not None:
        if not restrict_movie_ids:
            return set()
        query = f"""
            SELECT movie_id
            FROM public.movie_card
            WHERE release_format = ANY(%s::smallint[])
              AND movie_id = ANY(%s::bigint[]){filter_clause}
        """
        rows = await _execute_read(
            query, (release_format_ids, list(restrict_movie_ids), *filter_params)
        )
    else:
        query = f"""
            SELECT movie_id
            FROM public.movie_card
            WHERE release_format = ANY(%s::smallint[]){filter_clause}
        """
        rows = await _execute_read(query, (release_format_ids, *filter_params))
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
    *,
    metadata_filters: Optional[MetadataFilters] = None,
) -> list[tuple[int, int, int]]:
    """Fetch (movie_id, billing_position, cast_size) for actor term_ids.

    Used by the entity endpoint's actor prominence scoring. Unlike
    fetch_movie_ids_by_term_ids, this returns per-row billing data so
    the caller can compute zone-based prominence scores.

    Args:
        term_ids: Resolved string IDs for actor names.
        restrict_movie_ids: Optional candidate-pool filter (used by
            preference execution to narrow the scan to the pool).
        metadata_filters: Optional user-supplied hard filters; applied via
            an inline ``movie_id IN (SELECT ... FROM public.movie_card
            WHERE ...)`` subquery since the FROM table is a posting table.

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

    filter_clause, filter_params = _build_inline_movie_card_filter_clause(metadata_filters)
    params.extend(filter_params)

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
          AND cast_size > 0{filter_clause}
    """
    rows = await _execute_read(query, params)
    return [(row[0], row[1], row[2]) for row in rows]


async def fetch_character_billing_rows(
    term_ids: list[int],
    restrict_movie_ids: Optional[set[int]] = None,
    *,
    metadata_filters: Optional[MetadataFilters] = None,
) -> list[tuple[int, int, int]]:
    """Fetch (movie_id, billing_position, character_cast_size) for character term_ids.

    Direct analogue of fetch_actor_billing_rows used by the entity
    endpoint's character prominence scoring. Returns per-row billing
    data so the caller can compute CENTRAL / DEFAULT prominence scores.

    Multiple term_ids for the same movie (variant name lookups like
    "Spider-Man" + "Peter Parker") produce multiple rows with distinct
    billing positions; callers take the max score per movie.

    Args:
        term_ids: Resolved string IDs for character names.
        restrict_movie_ids: Optional candidate-pool filter (used by
            preference execution to narrow the scan to the pool).

    Returns:
        List of (movie_id, billing_position, character_cast_size) tuples.
        Rows missing billing metadata are filtered out.
    """
    if not term_ids:
        return []

    params: list = [term_ids]
    restrict_clause = ""
    if restrict_movie_ids:
        restrict_clause = " AND movie_id = ANY(%s::bigint[])"
        params.append(list(restrict_movie_ids))

    filter_clause, filter_params = _build_inline_movie_card_filter_clause(metadata_filters)
    params.extend(filter_params)

    query = f"""
        SELECT movie_id, billing_position, character_cast_size
        FROM lex.inv_character_postings
        WHERE term_id = ANY(%s::bigint[]){restrict_clause}
          AND billing_position IS NOT NULL
          AND character_cast_size IS NOT NULL
          AND character_cast_size > 0{filter_clause}
    """
    rows = await _execute_read(query, params)
    return [(row[0], row[1], row[2]) for row in rows]


async def fetch_movie_ids_with_title_like(
    like_pattern: str,
    restrict_movie_ids: Optional[set[int]] = None,
    *,
    metadata_filters: Optional[MetadataFilters] = None,
) -> set[int]:
    """LIKE match of a normalized title pattern against public.movie_card.title_normalized.

    The entity endpoint title_pattern sub-type routes here. The caller
    is responsible for preparing the LIKE pattern: (1) normalize the
    raw user pattern with normalize_string, (2) escape LIKE wildcards
    via escape_like, (3) wrap with '%' on both sides for contains or
    suffix '%' for starts_with.

    Because both sides of the comparison are already normalized
    (title_normalized is written at ingest via the same normalize_string
    contract), plain LIKE is symmetric — no ILIKE is needed, and the
    idx_movie_card_title_normalized_trgm / _prefix indexes get used.

    Args:
        like_pattern: Fully formed LIKE pattern (e.g. "%love%", "the %").
        restrict_movie_ids: Optional candidate-pool filter used by
            preference execution. Pushed to the DB to avoid pulling
            back every match for high-cardinality patterns.

    Returns:
        Set of movie IDs whose normalized title matches the pattern.
    """
    params: list = [like_pattern]
    restrict_clause = ""
    if restrict_movie_ids:
        restrict_clause = " AND movie_id = ANY(%s::bigint[])"
        params.append(list(restrict_movie_ids))

    # FROM is movie_card already → fold conditions inline rather than self-IN.
    filter_clause, filter_params = _build_direct_movie_card_filter_clause(metadata_filters)
    params.extend(filter_params)

    query = f"""
        SELECT movie_id
        FROM public.movie_card
        WHERE title_normalized LIKE %s{restrict_clause}{filter_clause}
    """
    rows = await _execute_read(query, params)
    return {row[0] for row in rows}


async def fetch_movie_ids_with_titles_matching_any(
    like_patterns: list[str],
    restrict_movie_ids: Optional[set[int]] = None,
    *,
    metadata_filters: Optional[MetadataFilters] = None,
) -> set[int]:
    """Single-query union of multiple LIKE patterns against title_normalized.

    Used when a TitlePatternQuerySpec carries multiple targets — each
    pattern is OR'd in one query so the executor pays one DB round trip
    instead of fanning out. Caller prepares each entry exactly as for
    `fetch_movie_ids_with_title_like` (normalize → escape_like → wrap
    with '%' for contains, suffix '%' for starts_with, or no wildcards
    for exact_match — bare LIKE without wildcards collapses to equality
    once special characters have been escaped).

    Empty `like_patterns` is a programmer error — callers should skip
    the call entirely when the target list is empty.

    Args:
        like_patterns: One fully-formed LIKE pattern per target.
        restrict_movie_ids: Optional candidate-pool filter pushed to
            the DB (preference-execution path).

    Returns:
        Set of movie IDs whose normalized title matches any pattern.
    """
    if not like_patterns:
        raise ValueError("like_patterns must be non-empty.")

    params: list = [like_patterns]
    restrict_clause = ""
    if restrict_movie_ids:
        restrict_clause = " AND movie_id = ANY(%s::bigint[])"
        params.append(list(restrict_movie_ids))

    # FROM is movie_card already → fold conditions inline rather than self-IN.
    filter_clause, filter_params = _build_direct_movie_card_filter_clause(metadata_filters)
    params.extend(filter_params)

    # `LIKE ANY (text[])` is the Postgres idiom for OR-ing a variable
    # number of patterns in a single query. Planner can still use
    # idx_movie_card_title_normalized_trgm for non-anchored patterns
    # and the _prefix index when every pattern is a starts_with.
    query = f"""
        SELECT movie_id
        FROM public.movie_card
        WHERE title_normalized LIKE ANY(%s::text[]){restrict_clause}{filter_clause}
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
#     movie_card.lineage_entry_ids / shared_universe_entry_ids /
#     subgroup_entry_ids, with an optional LEFT JOIN to
#     movie_franchise_metadata when any structural axis
#     (lineage_position / is_spinoff / is_crossover /
#     launched_franchise / launched_subgroup) is active. When a name
#     axis is active the helper returns the matched movie ids split
#     into "lineage" vs "universe-only" so stage-3 can apply
#     lineage-preference scoring when prefer_lineage is set.
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


async def fetch_franchise_entries_for_movies(
    movie_ids: list[int],
) -> dict[int, tuple[set[int], set[int]]]:
    """Return per-movie (lineage_entry_ids, shared_universe_entry_ids).

    Used by the exact-title search flow (search_v2/exact_title_search.py)
    to read a seed movie's franchise membership before fanning out to
    other movies that share those franchise entry IDs. fetch_movie_cards
    intentionally omits these arrays from its SELECT, so the exact-title
    flow needs a dedicated read rather than overloading that helper.

    Args:
        movie_ids: Movie IDs to read franchise-entry arrays for.

    Returns:
        Mapping `{movie_id: (lineage_entry_ids, shared_universe_entry_ids)}`
        as sets. Movies absent from the row set, or with NULL arrays, get
        empty sets — callers can union without nullity checks. Movies
        absent from the input list are not present in the mapping.
    """
    if not movie_ids:
        return {}

    query = """
        SELECT movie_id, lineage_entry_ids, shared_universe_entry_ids
        FROM public.movie_card
        WHERE movie_id = ANY(%s::bigint[])
    """
    rows = await _execute_read(query, (movie_ids,))
    return {
        row[0]: (set(row[1] or ()), set(row[2] or ()))
        for row in rows
    }


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
    metadata_filters: Optional[MetadataFilters] = None,
) -> tuple[set[int], set[int]]:
    """Resolve pre-computed franchise entry-id sets + structural flags to movie IDs.

    Builds a single AND-composed WHERE clause from whichever axes are
    active. All populated constraints must hold simultaneously — this is
    the AND execution policy for the franchise endpoint.

    Array-axis matching uses the GIN `&&` overlap operator against the
    denormalized entry-id arrays on public.movie_card. The name axis
    overlaps against BOTH lineage_entry_ids AND shared_universe_entry_ids
    (OR-combined for the row filter), and the SELECT returns a boolean
    flagging which side matched so stage-3 can score lineage matches
    higher than universe-only matches. Structural-axis predicates
    (lineage_position, is_spinoff, is_crossover, launched_franchise,
    launched_subgroup) live on public.movie_franchise_metadata, so we
    LEFT JOIN only when any of those is active — keeping the
    zero-structural-axis case as a single-table scan on movie_card.

    Args:
        franchise_name_entry_ids: Pre-resolved entry-id set from the
            executor's `franchise_names` token intersection + cross-name
            union. None or an empty set = axis not active (the predicate
            is dropped, not treated as a universal match). The executor
            normalizes these two forms upstream via
            `franchise_name_entry_ids or None`, so the expected contract
            is `None` for inactive and a non-empty set for active. The
            executor early-exits to an empty result before calling this
            helper if a populated `spec.franchise_names` resolves to an
            empty set, so "axis inactive" here unambiguously means "the
            spec did not populate this axis."
        subgroup_entry_ids: Same, for the `subgroup_names` axis.
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
        ``(lineage_matched, universe_only_matched)`` — disjoint sets of
        movie_ids. A movie whose lineage_entry_ids overlap the query name
        set lands in the first element regardless of whether its
        shared_universe_entry_ids also overlap (lineage-dominant). When
        the name axis is not active (`franchise_name_entry_ids is None`
        or empty), all matches land in the first element and the second
        is empty — there is no lineage-vs-universe distinction to make
        in that case. Empty sets when no conditions are provided
        (defensive guard) or when no movies match.
    """
    # The name axis overlaps both lineage_entry_ids and
    # shared_universe_entry_ids. Track it separately from the other
    # array conditions because the executor needs to know which side
    # matched per row — not just whether the row matched at all. The
    # lineage-side expression is reused in both the WHERE (OR with
    # universe side) and the SELECT list (as the flag), so we build a
    # dedicated list of lineage-side params that get bound twice.
    name_axis_active = bool(franchise_name_entry_ids)
    if name_axis_active:
        name_entry_list = list(franchise_name_entry_ids)
        lineage_hit_sql = "(mc.lineage_entry_ids && %s::bigint[])"
        universe_hit_sql = "(mc.shared_universe_entry_ids && %s::bigint[])"
        name_row_filter_sql = f"({lineage_hit_sql} OR {universe_hit_sql})"
    else:
        name_entry_list = []
        lineage_hit_sql = ""
        name_row_filter_sql = ""

    # Array-axis predicates hit the denormalized entry-id arrays on
    # public.movie_card. The name axis (if active) contributes one OR
    # clause spanning lineage + shared_universe columns; the subgroup
    # axis contributes a single && overlap on subgroup_entry_ids.
    array_conditions: list[str] = []
    array_params: list = []
    if name_axis_active:
        array_conditions.append(name_row_filter_sql)
        array_params.append(name_entry_list)  # lineage side
        array_params.append(name_entry_list)  # universe side
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
        return (set(), set())

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
            return (set(), set())
        conditions.append(f"{restrict_column} = ANY(%s::bigint[])")
        params.append(list(restrict_movie_ids))

    # Apply UI hard filters. We use the inline subquery form rather than
    # folding into a direct WHERE — the structural-only branch above
    # drives from movie_franchise_metadata (which doesn't carry movie_card
    # columns), so a self-IN against movie_card is the only uniform way
    # to apply the filter across all three FROM-clause shapes.
    filter_clause, filter_params = _build_inline_movie_card_filter_clause(
        metadata_filters,
        movie_id_column=restrict_column,
    )
    if filter_clause:
        # Strip leading " AND " — we'll splice into the conditions list.
        conditions.append(filter_clause.removeprefix(" AND "))
        params.extend(filter_params)

    where_clause = " AND ".join(conditions)

    # When the name axis is active we also SELECT a per-row boolean
    # flagging whether the lineage side matched. psycopg binds %s params
    # positionally, so the SELECT-clause param has to be prepended to
    # the WHERE params.
    if name_axis_active:
        select_clause = f"{movie_id_expr}, {lineage_hit_sql} AS lineage_hit"
        params = [name_entry_list] + params
    else:
        select_clause = movie_id_expr

    query = f"SELECT {select_clause} FROM {from_clause} WHERE {where_clause}"
    rows = await _execute_read(query, params)

    # Split rows by match source. When the name axis is inactive there's
    # no lineage-hit column; every match lands in the first bucket and
    # the caller treats the second bucket as empty.
    lineage_matched: set[int] = set()
    universe_only_matched: set[int] = set()
    if name_axis_active:
        for movie_id, is_lineage in rows:
            if is_lineage:
                lineage_matched.add(movie_id)
            else:
                universe_only_matched.add(movie_id)
    else:
        lineage_matched = {row[0] for row in rows}

    return (lineage_matched, universe_only_matched)


async def fetch_non_character_franchise_movies(
    normalized_canonical_names: list[str],
    *,
    limit: int = 100,
    metadata_filters: Optional[MetadataFilters] = None,
) -> list[tuple[int, int]]:
    """Resolve a set of franchise canonical names and bucket their movies.

    Single-round-trip lookup used by the non-character franchise executor.
    Resolves each entry in ``normalized_canonical_names`` against the
    UNIQUE index on ``lex.franchise_entry.normalized_string``, OR-unions
    the resulting ``franchise_entry_id`` set, and joins into
    ``public.movie_card`` via the GIN-indexed entry-id arrays.

    Multi-name input supports query-time alias expansion: when the
    upstream executor's LLM emits ``["marvel cinematic universe",
    "marvel"]`` for a single user reference, both forms resolve here and
    their entry-id sets union into one bucketed result. A movie that
    matches via multiple entries gets bucket=0 if ANY matching entry
    sits in its ``lineage_entry_ids`` — the ``bool_or`` aggregate over
    the JOIN expansion handles that, and ``GROUP BY mc.movie_id``
    deduplicates rows the JOIN amplified once per matched entry.

    The returned rows are pre-sorted to match the executor's append-after-
    sort algorithm: bucket 0 (primary / lineage) always precedes bucket 1
    (secondary / universe-only), and within each bucket the rows are
    sorted by ``popularity_score`` descending (NULLS LAST), with
    ``movie_id`` desc as the deterministic tiebreaker. The ``LIMIT`` is
    therefore bucket-aware — primary movies consume the budget first.

    Args:
        normalized_canonical_names: Outputs of
            ``normalize_franchise_string(name)`` — must use the same
            normalizer that populated
            ``lex.franchise_entry.normalized_string`` at ingest. Any
            divergence is a silent retrieval bug. Empty / blank entries
            are dropped; an all-empty list short-circuits to an empty
            result.
        limit: Maximum number of rows to return (primary + secondary
            combined). Defaults to 100 — enough headroom for the largest
            real franchises.

    Returns:
        ``list[(movie_id, bucket)]`` with ``bucket == 0`` for primary
        (lineage) hits and ``bucket == 1`` for secondary (universe-only)
        hits. Empty list when no name resolves.
    """
    # Drop blanks and dedupe — the same normalized form may appear
    # multiple times if the upstream LLM emitted near-duplicates that
    # collapsed under the shared normalizer. Sorting gives deterministic
    # query text so the plan cache stays warm.
    cleaned = sorted({n for n in normalized_canonical_names if n})
    if not cleaned:
        return []

    # The query already joins movie_card via the GIN-array overlap, so
    # filter conditions fold inline onto mc.* columns (no need for a
    # self-IN subquery). The helper emits unqualified column names —
    # since the join references mc.* explicitly elsewhere, we re-emit
    # via the inline-subquery form so we don't have to mess with the
    # `mc.` alias prefix.
    filter_clause, filter_params = _build_inline_movie_card_filter_clause(
        metadata_filters,
        movie_id_column="mc.movie_id",
    )

    # The JOIN amplifies one row per matched franchise_entry. GROUP BY
    # mc.movie_id collapses that back to one row per movie; bool_or over
    # the lineage-containment predicate assigns bucket=0 if ANY of the
    # matched entries lives in this movie's lineage_entry_ids (otherwise
    # it's universe-only → bucket=1). popularity_score is functionally
    # dependent on mc.movie_id (PK of movie_card), so listing it in
    # GROUP BY is just to satisfy the SQL grouping rules — there's no
    # real grouping on it. Using @> ARRAY[id]::bigint[] (rather than
    # = ANY(...)) keeps the GIN index on both array columns eligible.
    query = f"""
        SELECT
            mc.movie_id,
            CASE
                WHEN bool_or(
                    mc.lineage_entry_ids @> ARRAY[fe.franchise_entry_id]::bigint[]
                ) THEN 0
                ELSE 1
            END AS bucket
        FROM lex.franchise_entry fe
        JOIN public.movie_card mc
          ON (mc.lineage_entry_ids @> ARRAY[fe.franchise_entry_id]::bigint[]
              OR mc.shared_universe_entry_ids @> ARRAY[fe.franchise_entry_id]::bigint[])
        WHERE fe.normalized_string = ANY(%s::text[]){filter_clause}
        GROUP BY mc.movie_id, mc.popularity_score
        ORDER BY bucket ASC, mc.popularity_score DESC NULLS LAST, mc.movie_id DESC
        LIMIT %s
    """
    rows = await _execute_read(query, (cleaned, *filter_params, limit))
    return [(int(row[0]), int(row[1])) for row in rows]


# ===============================
#    KEYWORD ENDPOINT HELPERS
# ===============================
#
# Read helper for the step 3 keyword endpoint
# (search_v2/endpoint_fetching/keyword_query_execution.py). The
# endpoint commits a finalized list of UnifiedClassification members
# whose source_ids span up to three movie_card array columns
# (keyword_ids, source_material_type_ids, concept_tag_ids). Execution
# fetches a per-movie hit count across those columns; the executor
# then converts counts into scores via the spec's "any" or "avg"
# aggregation mode.


async def fetch_keyword_hit_counts(
    *,
    keyword_source_ids: list[int],
    source_material_source_ids: list[int],
    concept_tag_source_ids: list[int],
    restrict_movie_ids: set[int] | None = None,
    metadata_filters: Optional[MetadataFilters] = None,
) -> dict[int, int]:
    """Per-movie count of how many of the supplied source_ids appear
    in the movie's classification arrays on movie_card.

    The caller groups its committed UnifiedClassification members by
    backing column and passes each group's source_ids in. Counts are
    summed across the three columns into one hit count per movie.
    Movies with zero hits are omitted from the result map.

    A single SQL statement handles all three columns: an OR-of-overlap
    WHERE clause lets Postgres BitmapOr the GIN indexes, then a
    cardinality-on-array-intersect expression in the SELECT counts
    matches per column without needing the `intarray` extension. This
    keeps the round-trip count to one regardless of how the finalized
    set distributes across columns.

    Args:
        keyword_source_ids: source_ids whose backing column is
            keyword_ids. Empty list when no finalized member resolves
            to that column.
        source_material_source_ids: source_ids whose backing column
            is source_material_type_ids. Empty when none.
        concept_tag_source_ids: source_ids whose backing column is
            concept_tag_ids. Empty when none.
        restrict_movie_ids: Optional candidate-pool filter. When
            provided, only movies in this set can appear in the
            result.

    Returns:
        Mapping of movie_id → hit_count (always > 0). Movies that
        match none of the supplied source_ids are absent from the
        map.
    """
    # Caller-side bug guard: an all-empty call would emit a WHERE
    # clause that's always FALSE and waste a round-trip. The executor
    # validates `finalized_keywords` is non-empty upstream, so this
    # should be unreachable — return early rather than issue the
    # query.
    if not (
        keyword_source_ids
        or source_material_source_ids
        or concept_tag_source_ids
    ):
        return {}

    # Per-column overlap conditions are GIN-indexable. An empty
    # source_id list overlaps nothing (`col && ARRAY[]::int[]` is
    # FALSE), so we can include all three predicates unconditionally
    # and let Postgres skip the empty ones.
    where_clauses = [
        "keyword_ids && %s::int[]",
        "source_material_type_ids && %s::int[]",
        "concept_tag_ids && %s::int[]",
    ]

    # Per-column hit count via cardinality of the intersection.
    # Avoids the `intarray` extension and works on plain int[].
    select_count = (
        "cardinality(ARRAY(SELECT unnest(keyword_ids) "
        "INTERSECT SELECT unnest(%s::int[]))) "
        "+ cardinality(ARRAY(SELECT unnest(source_material_type_ids) "
        "INTERSECT SELECT unnest(%s::int[]))) "
        "+ cardinality(ARRAY(SELECT unnest(concept_tag_ids) "
        "INTERSECT SELECT unnest(%s::int[])))"
    )

    where_sql = "(" + " OR ".join(where_clauses) + ")"

    # psycopg binds %s placeholders positionally, so params must be
    # in the same order the placeholders appear in the final SQL
    # string. The SELECT clause's three placeholders come first,
    # then the WHERE clause's three, then the optional restrict.
    # Both blocks happen to need the same triple in the same order,
    # but ordering them by SQL-string position keeps the binding
    # correct even if the SELECT/WHERE arrays ever diverge.
    params: list = [
        keyword_source_ids,           # SELECT cardinality(... unnest(keyword_ids) ... unnest(%s))
        source_material_source_ids,   # SELECT ... unnest(source_material_type_ids) ... unnest(%s)
        concept_tag_source_ids,       # SELECT ... unnest(concept_tag_ids) ... unnest(%s)
        keyword_source_ids,           # WHERE keyword_ids && %s
        source_material_source_ids,   # WHERE source_material_type_ids && %s
        concept_tag_source_ids,       # WHERE concept_tag_ids && %s
    ]

    if restrict_movie_ids is not None:
        where_sql += " AND movie_id = ANY(%s::bigint[])"
        params.append(list(restrict_movie_ids))

    # FROM is movie_card — fold the user hard filter inline.
    filter_clause, filter_params = _build_direct_movie_card_filter_clause(metadata_filters)
    if filter_clause:
        where_sql += filter_clause
        params.extend(filter_params)

    query = (
        f"SELECT movie_id, {select_count} AS hits "
        f"FROM public.movie_card "
        f"WHERE {where_sql}"
    )
    rows = await _execute_read(query, params)
    # The WHERE clause guarantees ≥1 column overlap, so `hits` is
    # always ≥1 here; no zero-filtering needed.
    return {row[0]: row[1] for row in rows}


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
    metadata_filters: Optional[MetadataFilters] = None,
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

    # FROM is movie_card — fold the user hard filter inline.
    direct_filter, direct_params = _build_direct_movie_card_filter_clause(metadata_filters)
    if direct_filter:
        # _build_direct_... emits with a leading " AND " — strip it since
        # we're appending into a list joined by " AND ".
        conditions.append(direct_filter.removeprefix(" AND "))
        params.extend(direct_params)

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
    metadata_filters: Optional[MetadataFilters] = None,
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

    # User hard filter via inline subquery (movie_awards doesn't carry
    # movie_card columns). The subquery is index-friendly: Postgres
    # builds a hash semi-join on the eligible movie_id set.
    inline_filter, inline_params = _build_inline_movie_card_filter_clause(metadata_filters)
    if inline_filter:
        conditions.append(inline_filter.removeprefix(" AND "))
        params.extend(inline_params)

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
