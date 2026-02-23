"""Unit tests for db.postgres methods."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from db import postgres
from implementation.classes.enums import Genre


@pytest.fixture()
def fresh_genre_cache():
    """Clear the in-memory genre ID cache before and after each test that uses it.

    Without this fixture, a cache populated by one test leaks into subsequent
    tests that rely on the cache being empty at start (e.g. tests that verify
    the DB is only queried once, or tests that verify unknown names are skipped).
    """
    postgres._GENRE_ID_CACHE.clear()
    yield
    postgres._GENRE_ID_CACHE.clear()


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
async def test_execute_on_conn_with_existing_connection_fetches_rows() -> None:
    """_execute_on_conn should reuse caller connection and return fetched rows."""
    cursor = AsyncMock()
    cursor.fetchall.return_value = [(1,), (2,)]
    cursor_cm = MagicMock()
    cursor_cm.__aenter__ = AsyncMock(return_value=cursor)
    cursor_cm.__aexit__ = AsyncMock(return_value=None)
    connection = MagicMock()
    connection.cursor.return_value = cursor_cm

    result = await postgres._execute_on_conn(connection, "SELECT 1", (123,), fetch=True)

    cursor.execute.assert_awaited_once_with("SELECT 1", (123,))
    cursor.fetchall.assert_awaited_once()
    assert result == [(1,), (2,)]


@pytest.mark.asyncio
async def test_execute_on_conn_without_connection_commits_pool_connection(mocker) -> None:
    """_execute_on_conn should acquire and commit when conn is not supplied."""
    connection, cursor = _mock_pool_connection(mocker)
    result = await postgres._execute_on_conn(None, "INSERT INTO t VALUES (%s)", (9,), fetch=False)
    cursor.execute.assert_awaited_once_with("INSERT INTO t VALUES (%s)", (9,))
    connection.commit.assert_awaited_once()
    assert result is None


@pytest.mark.asyncio
async def test_batch_upsert_lexical_dictionary_empty_input_short_circuits(mocker) -> None:
    """batch_upsert_lexical_dictionary should skip DB work for empty input."""
    execute_on_conn = mocker.patch("db.postgres._execute_on_conn", new=AsyncMock())
    assert await postgres.batch_upsert_lexical_dictionary([]) == {}
    execute_on_conn.assert_not_awaited()


@pytest.mark.asyncio
async def test_batch_upsert_lexical_dictionary_dedupes_and_returns_mapping(mocker) -> None:
    """batch_upsert_lexical_dictionary should dedupe inputs and return ID map."""
    execute_on_conn = mocker.patch(
        "db.postgres._execute_on_conn",
        new=AsyncMock(return_value=[("spider-man", 11), ("hero", 22)]),
    )
    result = await postgres.batch_upsert_lexical_dictionary(["spider-man", "spider-man", "hero"])
    query, params = execute_on_conn.await_args.args[1:3]
    assert "lex.lexical_dictionary" in query
    assert params == (["spider-man", "hero"],)
    assert execute_on_conn.await_args.kwargs["fetch"] is True
    assert result == {"spider-man": 11, "hero": 22}


@pytest.mark.asyncio
async def test_batch_upsert_title_token_strings_calls_execute_on_conn(mocker) -> None:
    """batch_upsert_title_token_strings should bulk write title token lookup rows."""
    execute_on_conn = mocker.patch("db.postgres._execute_on_conn", new=AsyncMock())
    await postgres.batch_upsert_title_token_strings([11, 22], ["spider-man", "hero"])
    query, params = execute_on_conn.await_args.args[1:3]
    assert "lex.title_token_strings" in query
    assert params == ([11, 22], ["spider-man", "hero"])


@pytest.mark.asyncio
async def test_batch_upsert_title_token_strings_dedupes_duplicate_pairs(mocker) -> None:
    """batch_upsert_title_token_strings should dedupe duplicate (id, value) pairs."""
    execute_on_conn = mocker.patch("db.postgres._execute_on_conn", new=AsyncMock())
    await postgres.batch_upsert_title_token_strings(
        [11, 22, 22],
        ["spider-man", "hero", "hero"],
    )
    params = execute_on_conn.await_args.args[2]
    assert params == ([11, 22], ["spider-man", "hero"])


@pytest.mark.asyncio
async def test_batch_upsert_character_strings_calls_execute_on_conn(mocker) -> None:
    """batch_upsert_character_strings should bulk write character lookup rows."""
    execute_on_conn = mocker.patch("db.postgres._execute_on_conn", new=AsyncMock())
    await postgres.batch_upsert_character_strings([22], ["peter parker"])
    query, params = execute_on_conn.await_args.args[1:3]
    assert "lex.character_strings" in query
    assert params == ([22], ["peter parker"])


@pytest.mark.asyncio
async def test_batch_upsert_character_strings_dedupes_duplicate_pairs(mocker) -> None:
    """batch_upsert_character_strings should dedupe duplicate (id, value) pairs."""
    execute_on_conn = mocker.patch("db.postgres._execute_on_conn", new=AsyncMock())
    await postgres.batch_upsert_character_strings(
        [54, 55, 55],
        ["ferris bueller", "cameron frye", "cameron frye"],
    )
    params = execute_on_conn.await_args.args[2]
    assert params == ([54, 55], ["ferris bueller", "cameron frye"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("function_name", "table_name"),
    [
        ("batch_insert_title_token_postings", "lex.inv_title_token_postings"),
        ("batch_insert_person_postings", "lex.inv_person_postings"),
        ("batch_insert_character_postings", "lex.inv_character_postings"),
        ("batch_insert_studio_postings", "lex.inv_studio_postings"),
    ],
)
async def test_batch_insert_posting_functions_use_expected_tables(mocker, function_name: str, table_name: str) -> None:
    """Each batch posting function should target its table with expected params."""
    execute_on_conn = mocker.patch("db.postgres._execute_on_conn", new=AsyncMock())
    function = getattr(postgres, function_name)
    await function([7, 8], 77)
    conn_arg, query, params = execute_on_conn.await_args.args
    assert conn_arg is None
    assert table_name in query
    assert params == ([7, 8], 77)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "function_name",
    [
        "batch_insert_title_token_postings",
        "batch_insert_person_postings",
        "batch_insert_character_postings",
        "batch_insert_studio_postings",
    ],
)
async def test_batch_insert_posting_functions_empty_term_ids_short_circuit(
    mocker,
    function_name: str,
) -> None:
    """Batch posting insert helpers should no-op when term_ids is empty."""
    execute_on_conn = mocker.patch("db.postgres._execute_on_conn", new=AsyncMock())
    function = getattr(postgres, function_name)
    await function([], 77)
    execute_on_conn.assert_not_awaited()


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
    execute_on_conn = mocker.patch("db.postgres._execute_on_conn", new=AsyncMock())
    function = getattr(postgres, function_name)
    await function(*params)
    conn_arg, query, sent_params = execute_on_conn.await_args.args
    assert conn_arg is None
    assert table_name in query
    assert sent_params == params


@pytest.mark.parametrize(
    ("token_len", "expected_tier"),
    [
        (0, postgres._TITLE_TOKEN_MATCH_EXACT_ONLY),
        (3, postgres._TITLE_TOKEN_MATCH_EXACT_ONLY),
        (4, postgres._TITLE_TOKEN_MATCH_SUBSTRING),
        (6, postgres._TITLE_TOKEN_MATCH_SUBSTRING),
    ],
)
def test_get_title_token_tier_config_boundaries(token_len: int, expected_tier) -> None:
    """Token length should route to exact/substring config."""
    assert postgres._get_title_token_tier_config(token_len) is expected_tier


@pytest.mark.parametrize(
    ("tier", "expected_clauses", "unexpected_clauses", "expected_limit", "expected_max_df_param_position"),
    [
        (
            postgres._TITLE_TOKEN_MATCH_EXACT_ONLY,
            ["d.norm_str = %s", "df.doc_frequency <= %s"],
            ["similarity(d.norm_str, %s)", "d.norm_str %% %s", "levenshtein(d.norm_str, %s)", "ORDER BY"],
            None,
            -1,
        ),
        (
            postgres._TITLE_TOKEN_MATCH_SUBSTRING,
            ["d.norm_str LIKE %s ESCAPE '\\'", "ORDER BY", "length(d.norm_str) ASC", "df.doc_frequency <= %s"],
            ["d.norm_str %% %s", "levenshtein(d.norm_str, %s)", "similarity(d.norm_str, %s)"],
            500,
            -2,
        ),
    ],
)
def test_build_title_token_query_tier_specific_sql(
    tier,
    expected_clauses: list[str],
    unexpected_clauses: list[str],
    expected_limit: int | None,
    expected_max_df_param_position: int,
) -> None:
    """_build_title_token_query should include and exclude tier-specific SQL fragments."""
    query, params = postgres._build_title_token_query(tier, "spider", 123)
    assert query.startswith("SELECT d.string_id")
    for clause in expected_clauses:
        assert clause in query
    for clause in unexpected_clauses:
        assert clause not in query
    if expected_limit is None:
        assert "LIMIT" not in query
    else:
        assert f"LIMIT {expected_limit}" in query
    assert params[expected_max_df_param_position] == 123


@pytest.mark.asyncio
async def test_build_eligible_cte_with_no_filters() -> None:
    """_build_eligible_cte should emit TRUE where-clause when no filters are active."""
    filters = postgres.MetadataFilters()
    cte_sql, params = await postgres._build_eligible_cte(filters)
    assert "eligible AS MATERIALIZED" in cte_sql
    assert "SELECT movie_id, title_token_count" in cte_sql
    assert "WHERE TRUE" in cte_sql
    assert params == []


@pytest.mark.asyncio
async def test_build_eligible_cte_with_all_filters_preserves_param_order(mocker) -> None:
    """_build_eligible_cte should preserve deterministic SQL parameter ordering."""
    resolve_genres = mocker.patch(
        "db.postgres.fetch_genre_ids_by_name",
        new=AsyncMock(return_value={Genre.ACTION: 1, Genre.ADVENTURE: 2}),
    )
    filters = postgres.MetadataFilters(
        min_release_ts=10,
        max_release_ts=20,
        min_runtime=80,
        max_runtime=140,
        min_maturity_rank=2,
        max_maturity_rank=4,
        genres=[Genre.ACTION, Genre.ADVENTURE],
        watch_offer_keys=[99],
    )
    cte_sql, params = await postgres._build_eligible_cte(filters)
    assert "release_ts BETWEEN %s AND %s" in cte_sql
    assert "runtime_minutes BETWEEN %s AND %s" in cte_sql
    assert "maturity_rank BETWEEN %s AND %s" in cte_sql
    assert "genre_ids && %s::int[]" in cte_sql
    assert "watch_offer_keys && %s::int[]" in cte_sql
    resolve_genres.assert_awaited_once_with([Genre.ACTION, Genre.ADVENTURE])
    assert params == [10, 20, 80, 140, 2, 4, [1, 2], [99]]


@pytest.mark.asyncio
async def test_batch_upsert_title_token_strings_rejects_length_mismatch() -> None:
    """batch_upsert_title_token_strings should guard against misaligned arrays."""
    with pytest.raises(ValueError):
        await postgres.batch_upsert_title_token_strings([11], ["spider-man", "hero"])


@pytest.mark.asyncio
async def test_batch_upsert_character_strings_rejects_length_mismatch() -> None:
    """batch_upsert_character_strings should guard against misaligned arrays."""
    with pytest.raises(ValueError):
        await postgres.batch_upsert_character_strings([22], ["hero", "villain"])


@pytest.mark.asyncio
async def test_refresh_title_token_doc_frequency_calls_refresh_statement(mocker) -> None:
    """refresh_title_token_doc_frequency should issue materialized-view refresh query."""
    execute_write = mocker.patch("db.postgres._execute_write", new=AsyncMock())
    await postgres.refresh_title_token_doc_frequency()
    query = execute_write.await_args.args[0]
    assert "REFRESH MATERIALIZED VIEW CONCURRENTLY lex.title_token_doc_frequency;" in query


@pytest.mark.asyncio
async def test_upsert_movie_card_calls_execute_on_conn_with_expected_params(mocker) -> None:
    """upsert_movie_card should serialize sequences to lists and pass expected argument order."""
    execute_on_conn = mocker.patch("db.postgres._execute_on_conn", new=AsyncMock())
    await postgres.upsert_movie_card(
        movie_id=10,
        title="Movie",
        poster_url="poster",
        release_ts=1000,
        runtime_minutes=120,
        maturity_rank=3,
        genre_ids=(1, 2),
        watch_offer_keys=(100, 200),
        audio_language_ids=(7, 8),
        reception_score=72.5,
        title_token_count=4,
    )
    conn_arg, query, params = execute_on_conn.await_args.args
    assert conn_arg is None
    assert "public.movie_card" in query
    assert params == (10, "Movie", "poster", 1000, 120, 3, [1, 2], [100, 200], [7, 8], 72.5, 4)


@pytest.mark.asyncio
async def test_fetch_phrase_term_ids_empty_input_short_circuits(mocker) -> None:
    """fetch_phrase_term_ids should skip DB calls when no phrases are provided."""
    execute_read = mocker.patch("db.postgres._execute_read", new=AsyncMock())
    result = await postgres.fetch_phrase_term_ids([])
    assert result == {}
    execute_read.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_phrase_term_ids_returns_dictionary(mocker) -> None:
    """fetch_phrase_term_ids should map normalized phrase to returned string_id."""
    mocker.patch("db.postgres._execute_read", new=AsyncMock(return_value=[("tom hanks", 11), ("marvel", 22)]))
    result = await postgres.fetch_phrase_term_ids(["tom hanks", "marvel"])
    assert result == {"tom hanks": 11, "marvel": 22}


@pytest.mark.asyncio
async def test_fetch_movie_ids_by_term_ids_empty_term_ids_short_circuits(mocker) -> None:
    """fetch_movie_ids_by_term_ids should skip DB calls when term_ids are empty."""
    execute_read = mocker.patch("db.postgres._execute_read", new=AsyncMock())
    result = await postgres.fetch_movie_ids_by_term_ids(postgres.PostingTable.PERSON, [])
    assert result == set()
    execute_read.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_movie_ids_by_term_ids_returns_deduplicated_set(mocker) -> None:
    """fetch_movie_ids_by_term_ids should return distinct movie IDs as a set."""
    execute_read = mocker.patch(
        "db.postgres._execute_read",
        new=AsyncMock(return_value=[(101,), (102,), (101,)]),
    )
    result = await postgres.fetch_movie_ids_by_term_ids(postgres.PostingTable.CHARACTER, [7, 8])
    query, params = execute_read.await_args.args
    assert "FROM lex.inv_character_postings" in query
    assert params == ([7, 8],)
    assert result == {101, 102}


@pytest.mark.asyncio
async def test_fetch_title_token_ids_long_token_uses_exact_tier(mocker) -> None:
    """fetch_title_token_ids should batch by index and return grouped dict."""
    execute_read = mocker.patch(
        "db.postgres._execute_read",
        new=AsyncMock(return_value=[(0, 7), (0, 8), (1, 9)]),
    )
    result = await postgres.fetch_title_token_ids(["star", "wars"], max_df=500)
    query, params = execute_read.await_args.args
    assert "FROM unnest(%s::int[], %s::text[], %s::text[])" in query
    assert params[0] == [0, 1]
    assert params[1] == ["star", "wars"]
    assert params[2] == ["%star%", "%wars%"]
    assert params[5] == 500
    assert result == {0: [7, 8], 1: [9]}


@pytest.mark.asyncio
async def test_fetch_title_token_ids_normal_path_uses_tier_selector(mocker) -> None:
    """fetch_title_token_ids should return empty map on empty input."""
    execute_read = mocker.patch("db.postgres._execute_read", new=AsyncMock())
    result = await postgres.fetch_title_token_ids([], max_df=100)
    assert result == {}
    execute_read.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_title_token_ids_exact_uses_exact_tier(mocker) -> None:
    """fetch_title_token_ids_exact should batch exact lookups and group by idx."""
    execute_read = mocker.patch(
        "db.postgres._execute_read",
        new=AsyncMock(return_value=[(0, 13), (0, 14), (1, 99)]),
    )
    result = await postgres.fetch_title_token_ids_exact(["star", "wars"], max_df=50)
    query, params = execute_read.await_args.args
    assert "FROM unnest(%s::int[], %s::text[])" in query
    assert params == [[0, 1], ["star", "wars"], 50]
    assert result == {0: [13, 14], 1: [99]}


@pytest.mark.asyncio
async def test_fetch_character_term_ids_groups_by_query_idx(mocker) -> None:
    """fetch_character_term_ids should group returned term_ids under each query index."""
    execute_read = mocker.patch(
        "db.postgres._execute_read",
        new=AsyncMock(return_value=[(0, 11), (0, 12), (1, 22)]),
    )
    result = await postgres.fetch_character_term_ids([0, 1], ["%peter%", "%batman%"])
    query, params = execute_read.await_args.args
    assert "FROM unnest(%s::int[], %s::text[])" in query
    assert params[0] == [0, 1]
    assert params[1] == ["%peter%", "%batman%"]
    assert params[2] == postgres._CHARACTER_RESOLVE_LIMIT_PER_PHRASE
    assert result == {0: [11, 12], 1: [22]}


@pytest.mark.asyncio
async def test_fetch_movie_cards_empty_input_short_circuits(mocker) -> None:
    """fetch_movie_cards should return empty list and skip DB call for empty input."""
    execute_read = mocker.patch("db.postgres._execute_read", new=AsyncMock())
    result = await postgres.fetch_movie_cards([])
    assert result == []
    execute_read.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_movie_cards_maps_rows_to_dicts(mocker) -> None:
    """fetch_movie_cards should map positional columns to dictionary keys."""
    mocker.patch(
        "db.postgres._execute_read",
        new=AsyncMock(
            return_value=[
                (1, "A", "u", 10, 90, 2, [1], [11], [21], 80.5),
                (2, "B", None, None, None, None, [], [], [], None),
            ]
        ),
    )
    result = await postgres.fetch_movie_cards([1, 2])
    assert result[0]["movie_id"] == 1
    assert result[0]["title"] == "A"
    assert result[1]["movie_id"] == 2
    assert "audio_language_ids" in result[0]


@pytest.mark.asyncio
async def test_execute_compound_lexical_search_empty_short_circuits(mocker) -> None:
    """execute_compound_lexical_search should skip DB when all buckets are empty."""
    execute_read = mocker.patch("db.postgres._execute_read", new=AsyncMock())
    result = await postgres.execute_compound_lexical_search(
        people_term_ids=[],
        studio_term_ids=[],
        character_query_idxs=[],
        character_term_ids=[],
        title_searches=[],
    )
    execute_read.assert_not_awaited()
    assert result.people_scores == {}
    assert result.studio_scores == {}
    assert result.character_by_query == {}
    assert result.title_scores_by_search == {}


@pytest.mark.asyncio
async def test_execute_compound_lexical_search_without_filters_builds_tagged_union(mocker) -> None:
    """Compound query should omit eligible CTE and parse bucket-tagged rows."""
    execute_read = mocker.patch(
        "db.postgres._execute_read",
        new=AsyncMock(
            return_value=[
                ("people", -1, 10, 2.0),
                ("studio", -1, 10, 1.0),
                ("character", 0, 10, 1.0),
                ("title_0", -1, 10, 0.7),
            ]
        ),
    )
    result = await postgres.execute_compound_lexical_search(
        people_term_ids=[11],
        studio_term_ids=[22],
        character_query_idxs=[0],
        character_term_ids=[33],
        title_searches=[
            postgres.TitleSearchInput(
                token_idxs=[0, 1],
                term_ids=[101, 102],
                f_coeff=5.0,
                k=2,
                beta_sq=4.0,
                score_threshold=0.15,
                max_candidates=100,
            )
        ],
    )
    query, _ = execute_read.await_args.args
    assert "WITH eligible AS MATERIALIZED" not in query
    assert "SELECT 'people' AS bucket" in query
    assert "SELECT 'studio' AS bucket" in query
    assert "SELECT 'character' AS bucket" in query
    assert "SELECT 'title_0' AS bucket" in query
    assert result.people_scores == {10: 2}
    assert result.studio_scores == {10: 1}
    assert result.character_by_query == {0: {10: 1}}
    assert result.title_scores_by_search == {0: {10: 0.7}}


@pytest.mark.asyncio
async def test_execute_compound_lexical_search_with_filters_and_exclusions(mocker) -> None:
    """Compound query should include a single eligible CTE and exclusion clauses."""
    execute_read = mocker.patch(
        "db.postgres._execute_read",
        new=AsyncMock(return_value=[]),
    )
    filters = postgres.MetadataFilters(min_runtime=90)
    await postgres.execute_compound_lexical_search(
        people_term_ids=[1, 2],
        studio_term_ids=[],
        character_query_idxs=[0],
        character_term_ids=[3],
        title_searches=[],
        filters=filters,
        exclude_movie_ids={99},
    )
    query, params = execute_read.await_args.args
    assert "WITH eligible AS MATERIALIZED" in query
    assert query.count("eligible AS MATERIALIZED") == 1
    assert "NOT (p.movie_id = ANY(%s::bigint[]))" in query
    assert params[0] == 90


# ===============================
#   Genre Cache Loader Tests
# ===============================

@pytest.mark.asyncio
async def test_ensure_genre_cache_loaded_populates_from_normalized_db_rows(
    mocker, fresh_genre_cache
) -> None:
    """_ensure_genre_cache_loaded should map normalized DB names to the correct Genre members."""
    mocker.patch(
        "db.postgres._execute_read",
        new=AsyncMock(return_value=[("action", 1), ("sci-fi", 2), ("drama", 3)]),
    )
    await postgres._ensure_genre_cache_loaded()
    assert postgres._GENRE_ID_CACHE[Genre.ACTION] == 1
    assert postgres._GENRE_ID_CACHE[Genre.SCI_FI] == 2
    assert postgres._GENRE_ID_CACHE[Genre.DRAMA] == 3


@pytest.mark.asyncio
async def test_ensure_genre_cache_loaded_skips_unrecognized_names(
    mocker, fresh_genre_cache
) -> None:
    """_ensure_genre_cache_loaded should silently ignore DB rows with no matching Genre member."""
    mocker.patch(
        "db.postgres._execute_read",
        new=AsyncMock(return_value=[("action", 1), ("not_a_real_genre", 99)]),
    )
    await postgres._ensure_genre_cache_loaded()
    assert Genre.ACTION in postgres._GENRE_ID_CACHE
    assert len(postgres._GENRE_ID_CACHE) == 1


@pytest.mark.asyncio
async def test_ensure_genre_cache_loaded_is_idempotent(mocker, fresh_genre_cache) -> None:
    """_ensure_genre_cache_loaded should query the DB only once; subsequent calls are no-ops."""
    execute_read = mocker.patch(
        "db.postgres._execute_read",
        new=AsyncMock(return_value=[("action", 1)]),
    )
    await postgres._ensure_genre_cache_loaded()
    await postgres._ensure_genre_cache_loaded()
    execute_read.assert_awaited_once()


# ===============================
#   fetch_genre_ids_by_name Tests
# ===============================

@pytest.mark.asyncio
async def test_fetch_genre_ids_by_name_returns_correct_ids(mocker, fresh_genre_cache) -> None:
    """fetch_genre_ids_by_name should resolve Genre members to their DB IDs from the cache."""
    postgres._GENRE_ID_CACHE.update({Genre.ACTION: 7, Genre.DRAMA: 14})
    mocker.patch("db.postgres._ensure_genre_cache_loaded", new=AsyncMock())
    result = await postgres.fetch_genre_ids_by_name([Genre.ACTION, Genre.DRAMA])
    assert result == {Genre.ACTION: 7, Genre.DRAMA: 14}


@pytest.mark.asyncio
async def test_fetch_genre_ids_by_name_empty_input_returns_empty_without_loading_cache(
    mocker,
) -> None:
    """fetch_genre_ids_by_name should short-circuit and not touch the cache for empty input."""
    ensure_loaded = mocker.patch("db.postgres._ensure_genre_cache_loaded", new=AsyncMock())
    result = await postgres.fetch_genre_ids_by_name([])
    assert result == {}
    ensure_loaded.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_genre_ids_by_name_deduplicates_input(mocker, fresh_genre_cache) -> None:
    """fetch_genre_ids_by_name should return each genre at most once even if repeated in input."""
    postgres._GENRE_ID_CACHE[Genre.ACTION] = 5
    mocker.patch("db.postgres._ensure_genre_cache_loaded", new=AsyncMock())
    result = await postgres.fetch_genre_ids_by_name([Genre.ACTION, Genre.ACTION, Genre.ACTION])
    assert result == {Genre.ACTION: 5}


@pytest.mark.asyncio
async def test_fetch_genre_ids_by_name_omits_genres_absent_from_cache(
    mocker, fresh_genre_cache
) -> None:
    """fetch_genre_ids_by_name should silently omit genres not present in the loaded cache."""
    postgres._GENRE_ID_CACHE[Genre.ACTION] = 3
    mocker.patch("db.postgres._ensure_genre_cache_loaded", new=AsyncMock())
    result = await postgres.fetch_genre_ids_by_name([Genre.ACTION, Genre.DRAMA])
    assert Genre.ACTION in result
    assert Genre.DRAMA not in result


# ===============================
#   Cross-Boundary Invariant Tests
# ===============================

def test_genre_normalized_name_equals_value_lowercased() -> None:
    """Every Genre.normalized_name must be exactly Genre.value.lower().

    This guards against typos when adding new Genre members — the cache loader
    relies on normalized_name matching what normalize_string() produces at
    ingest time.
    """
    for genre in Genre:
        assert genre.normalized_name == genre.value.lower(), (
            f"Genre.{genre.name}: normalized_name={genre.normalized_name!r} "
            f"does not equal value.lower()={genre.value.lower()!r}"
        )


def test_genre_by_normalized_name_map_covers_all_genre_members() -> None:
    """_GENRE_BY_NORMALIZED_NAME must have an entry for every Genre member.

    A gap in this map means a genre stored in the DB will never be resolved
    into the cache, causing silent cache misses at query time.
    """
    for genre in Genre:
        assert genre.normalized_name in postgres._GENRE_BY_NORMALIZED_NAME, (
            f"Genre.{genre.name} ({genre.normalized_name!r}) is missing from "
            "_GENRE_BY_NORMALIZED_NAME"
        )
        assert postgres._GENRE_BY_NORMALIZED_NAME[genre.normalized_name] is genre


def test_ingest_and_cache_loader_use_consistent_genre_name_format() -> None:
    """Names written by ingest (normalized) must be resolvable by the cache loader's lookup map.

    This is the end-to-end contract between the write path (create_genre_ids)
    and the read path (_ensure_genre_cache_loaded). If ingest stores a name
    that the cache loader cannot look up, genre filtering silently returns no
    results.
    """
    from implementation.misc.helpers import normalize_string

    for genre in Genre:
        # Simulate what create_genre_ids writes: normalize_string(genre.value)
        simulated_ingest_name = normalize_string(genre.value)
        resolved = postgres._GENRE_BY_NORMALIZED_NAME.get(simulated_ingest_name)
        assert resolved is genre, (
            f"Genre.{genre.name}: ingest writes {simulated_ingest_name!r} but "
            f"cache loader resolves it to {resolved!r} instead of Genre.{genre.name}"
        )
