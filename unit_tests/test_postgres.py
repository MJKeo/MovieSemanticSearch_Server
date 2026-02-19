"""Unit tests for db.postgres methods."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from db import postgres


def _mock_pool_connection(
    mocker,
    *,
    fetchall_result=None,
    fetchone_result=None,
):
    """Mock pool.connection() -> conn.cursor() async context managers and return mocks."""
    cursor = AsyncMock()
    cursor.fetchall.return_value = fetchall_result
    cursor.fetchone.return_value = fetchone_result

    # Build async context manager for conn.cursor().
    cursor_cm = MagicMock()
    cursor_cm.__aenter__ = AsyncMock(return_value=cursor)
    cursor_cm.__aexit__ = AsyncMock(return_value=None)

    connection = MagicMock()
    connection.commit = AsyncMock()
    connection.execute = AsyncMock()
    connection.cursor.return_value = cursor_cm

    # Build async context manager for pool.connection().
    connection_cm = MagicMock()
    connection_cm.__aenter__ = AsyncMock(return_value=connection)
    connection_cm.__aexit__ = AsyncMock(return_value=None)
    mocker.patch.object(postgres.pool, "connection", return_value=connection_cm)

    return connection, cursor


def test_build_conninfo_uses_environment_variables(mocker) -> None:
    """_build_conninfo should format host/db/user/password from environment variables."""
    env_values = {
        "POSTGRES_HOST": "localhost",
        "POSTGRES_DB": "movies",
        "POSTGRES_USER": "tester",
        "POSTGRES_PASSWORD": "secret",
    }
    mocker.patch("db.postgres.os.getenv", side_effect=lambda key: env_values.get(key))
    assert postgres._build_conninfo() == "host=localhost dbname=movies user=tester password=secret"


@pytest.mark.asyncio
async def test_execute_read_fetches_all_rows(mocker) -> None:
    """_execute_read should execute SQL and fetch all rows."""
    _, cursor = _mock_pool_connection(mocker, fetchall_result=[(1,), (2,)])
    result = await postgres._execute_read("SELECT 1", (123,))
    cursor.execute.assert_awaited_once_with("SELECT 1", (123,))
    cursor.fetchall.assert_awaited_once()
    assert result == [(1,), (2,)]


@pytest.mark.asyncio
async def test_execute_read_one_fetches_single_row(mocker) -> None:
    """_execute_read_one should execute SQL and fetch one row."""
    _, cursor = _mock_pool_connection(mocker, fetchone_result=(42,))
    result = await postgres._execute_read_one("SELECT 1", (456,))
    cursor.execute.assert_awaited_once_with("SELECT 1", (456,))
    cursor.fetchone.assert_awaited_once()
    assert result == (42,)


@pytest.mark.asyncio
async def test_execute_write_without_fetch_one_commits(mocker) -> None:
    """_execute_write should commit and return None when fetch_one is False."""
    connection, cursor = _mock_pool_connection(mocker)
    result = await postgres._execute_write("INSERT INTO t VALUES (%s)", (9,), fetch_one=False)
    cursor.execute.assert_awaited_once_with("INSERT INTO t VALUES (%s)", (9,))
    cursor.fetchone.assert_not_awaited()
    connection.commit.assert_awaited_once()
    assert result is None


@pytest.mark.asyncio
async def test_execute_write_with_fetch_one_returns_row(mocker) -> None:
    """_execute_write should return fetched row when fetch_one is True."""
    connection, cursor = _mock_pool_connection(mocker, fetchone_result=(99,))
    result = await postgres._execute_write("INSERT ... RETURNING id", (1,), fetch_one=True)
    cursor.execute.assert_awaited_once_with("INSERT ... RETURNING id", (1,))
    cursor.fetchone.assert_awaited_once()
    connection.commit.assert_awaited_once()
    assert result == (99,)


@pytest.mark.asyncio
async def test_check_postgres_returns_ok_on_success(mocker) -> None:
    """check_postgres should return ok when SELECT 1 succeeds."""
    connection, _ = _mock_pool_connection(mocker)
    assert await postgres.check_postgres() == "ok"
    connection.execute.assert_awaited_once_with("SELECT 1")


@pytest.mark.asyncio
async def test_check_postgres_returns_error_string_on_failure(mocker) -> None:
    """check_postgres should return the exception message when pool usage fails."""
    connection_cm = AsyncMock()
    connection_cm.__aenter__.side_effect = RuntimeError("db unavailable")
    mocker.patch.object(postgres.pool, "connection", return_value=connection_cm)
    assert await postgres.check_postgres() == "db unavailable"


@pytest.mark.asyncio
async def test_upsert_lexical_dictionary_returns_row_id(mocker) -> None:
    """upsert_lexical_dictionary should unwrap string_id from returned tuple."""
    execute_write = mocker.patch("db.postgres._execute_write", new=AsyncMock(return_value=(555,)))
    result = await postgres.upsert_lexical_dictionary("spider-man")
    execute_write.assert_awaited_once()
    assert result == 555


@pytest.mark.asyncio
async def test_upsert_lexical_dictionary_returns_none_for_empty_row(mocker) -> None:
    """upsert_lexical_dictionary should return None when no row is returned."""
    mocker.patch("db.postgres._execute_write", new=AsyncMock(return_value=None))
    assert await postgres.upsert_lexical_dictionary("spider-man") is None


@pytest.mark.asyncio
async def test_upsert_title_token_string_calls_execute_write(mocker) -> None:
    """upsert_title_token_string should write to title token lookup table."""
    execute_write = mocker.patch("db.postgres._execute_write", new=AsyncMock())
    await postgres.upsert_title_token_string(11, "spider-man")
    query, params = execute_write.await_args.args
    assert "lex.title_token_strings" in query
    assert params == (11, "spider-man")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("function_name", "table_name"),
    [
        ("insert_title_token_posting", "lex.inv_title_token_postings"),
        ("insert_person_posting", "lex.inv_person_postings"),
        ("insert_character_posting", "lex.inv_character_postings"),
        ("insert_studio_posting", "lex.inv_studio_postings"),
    ],
)
async def test_insert_posting_functions_use_expected_tables(mocker, function_name: str, table_name: str) -> None:
    """Each insert posting function should target its table with expected params."""
    execute_write = mocker.patch("db.postgres._execute_write", new=AsyncMock())
    function = getattr(postgres, function_name)
    await function(7, 77)
    query, params = execute_write.await_args.args
    assert table_name in query
    assert params == (7, 77)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("function_name", "table_name", "params"),
    [
        ("upsert_genre_dictionary", "lex.genre_dictionary", (1, "Action")),
        ("upsert_provider_dictionary", "lex.provider_dictionary", (2, "Netflix")),
        ("upsert_watch_method_dictionary", "lex.watch_method_dictionary", (3, "rent")),
        ("upsert_maturity_dictionary", "lex.maturity_dictionary", (4, "R")),
        ("upsert_language_dictionary", "lex.language_dictionary", (5, "English")),
    ],
)
async def test_dictionary_upserts_use_expected_tables_and_params(
    mocker,
    function_name: str,
    table_name: str,
    params: tuple,
) -> None:
    """Dictionary upsert functions should execute against expected tables and params."""
    execute_write = mocker.patch("db.postgres._execute_write", new=AsyncMock())
    function = getattr(postgres, function_name)
    await function(*params)
    query, sent_params = execute_write.await_args.args
    assert table_name in query
    assert sent_params == params


@pytest.mark.asyncio
async def test_upsert_phrase_term_returns_none_when_normalized_is_empty(mocker) -> None:
    """upsert_phrase_term should stop early when normalized phrase is empty."""
    mocker.patch("db.postgres.normalize_string", return_value="")
    upsert_lexical = mocker.patch("db.postgres.upsert_lexical_dictionary", new=AsyncMock())
    assert await postgres.upsert_phrase_term("###") is None
    upsert_lexical.assert_not_awaited()


@pytest.mark.asyncio
async def test_upsert_phrase_term_normal_flow(mocker) -> None:
    """upsert_phrase_term should normalize and then upsert lexical term."""
    mocker.patch("db.postgres.normalize_string", return_value="spider man")
    upsert_lexical = mocker.patch("db.postgres.upsert_lexical_dictionary", new=AsyncMock(return_value=321))
    assert await postgres.upsert_phrase_term("Spider Man") == 321
    upsert_lexical.assert_awaited_once_with("spider man")
