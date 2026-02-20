"""
Database connection pool and query methods for the API service.

This module provides a psycopg v3 AsyncConnectionPool configured for production use,
along with async helper functions for executing queries and public methods for
upserting/inserting movie and lexical data.
"""

import os
from typing import Optional, Sequence
from psycopg_pool import AsyncConnectionPool
from implementation.misc.helpers import normalize_string


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


# PRIVATE BASE METHODS

async def _execute_read(query: str, params: tuple | None = None) -> list[tuple]:
    """
    Execute a read query and return all rows.
    
    Private helper method for internal use.
    
    Args:
        query: SQL query string with parameter placeholders (%s).
        params: Optional tuple of parameters to bind to the query.
    
    Returns:
        List of tuples, where each tuple represents a row.
    """
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, params)
            return await cur.fetchall()


async def _execute_read_one(query: str, params: tuple | None = None) -> tuple | None:
    """
    Execute a read query and return a single row, or None if no rows match.
    
    Private helper method for internal use.
    
    Args:
        query: SQL query string with parameter placeholders (%s).
        params: Optional tuple of parameters to bind to the query.
    
    Returns:
        A tuple representing the first row, or None if no rows were returned.
    """
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, params)
            return await cur.fetchone()


async def _execute_write(query: str, params: tuple | None = None, fetch_one: bool = False):
    """
    Execute a write query (INSERT, UPDATE, DELETE) with an explicit commit.
    
    Private helper method for internal use. The connection is used within a transaction.
    On clean exit, the transaction is explicitly committed. If an exception occurs,
    the transaction is rolled back automatically by the connection context manager.
    
    Args:
        query: SQL query string with parameter placeholders (%s).
        params: Optional tuple of parameters to bind to the query.
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


async def insert_title_token_posting(term_id: int, movie_id: int) -> None:
    """
    Insert one title-token posting row into lex.inv_title_token_postings.
    
    Args:
        term_id: The term ID.
        movie_id: The movie ID.
    """
    query = """
    INSERT INTO lex.inv_title_token_postings (term_id, movie_id)
    VALUES (%s, %s)
    ON CONFLICT (term_id, movie_id) DO NOTHING;
    """
    await _execute_write(query, (term_id, movie_id))


async def insert_person_posting(term_id: int, movie_id: int) -> None:
    """
    Insert one person posting row into lex.inv_person_postings.
    
    Args:
        term_id: The term ID.
        movie_id: The movie ID.
    """
    query = """
    INSERT INTO lex.inv_person_postings (term_id, movie_id)
    VALUES (%s, %s)
    ON CONFLICT (term_id, movie_id) DO NOTHING;
    """
    await _execute_write(query, (term_id, movie_id))


async def insert_character_posting(term_id: int, movie_id: int) -> None:
    """
    Insert one character posting row into lex.inv_character_postings.
    
    Args:
        term_id: The term ID.
        movie_id: The movie ID.
    """
    query = """
    INSERT INTO lex.inv_character_postings (term_id, movie_id)
    VALUES (%s, %s)
    ON CONFLICT (term_id, movie_id) DO NOTHING;
    """
    await _execute_write(query, (term_id, movie_id))


async def insert_studio_posting(term_id: int, movie_id: int) -> None:
    """
    Insert one studio posting row into lex.inv_studio_postings.
    
    Args:
        term_id: The term ID.
        movie_id: The movie ID.
    """
    query = """
    INSERT INTO lex.inv_studio_postings (term_id, movie_id)
    VALUES (%s, %s)
    ON CONFLICT (term_id, movie_id) DO NOTHING;
    """
    await _execute_write(query, (term_id, movie_id))


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


# ===============================
#        HELPER METHODS
# ===============================

async def upsert_phrase_term(value: str) -> int | None:
    """Normalize a phrase and upsert it into lexical_dictionary."""
    normalized = normalize_string(value)
    if not normalized:
        return None
    return await upsert_lexical_dictionary(normalized)