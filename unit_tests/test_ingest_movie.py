"""Unit tests for db.ingest_movie methods."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from db import ingest_movie, postgres


@pytest.mark.asyncio
async def test_upsert_movie_card_calls_execute_on_conn_with_expected_params() -> None:
    """upsert_movie_card should forward normalized params in the expected order."""
    execute_on_conn = AsyncMock()
    original = postgres._execute_on_conn
    postgres._execute_on_conn = execute_on_conn
    try:
        await ingest_movie.upsert_movie_card(
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
    finally:
        postgres._execute_on_conn = original

    conn_arg, query, params = execute_on_conn.await_args.args
    assert conn_arg is None
    assert "public.movie_card" in query
    assert params == (10, "Movie", "poster", 1000, 120, 3, [1, 2], [100, 200], [7, 8], 72.5, 4)


@pytest.mark.asyncio
async def test_ingest_movie_runs_card_and_lexical_ingestion(mocker, base_movie_factory) -> None:
    """ingest_movie should run movie-card and lexical ingestion on a single connection."""
    movie = base_movie_factory()
    ingest_card = mocker.patch("db.ingest_movie.ingest_movie_card", new=AsyncMock())
    ingest_lexical = mocker.patch("db.ingest_movie.ingest_lexical_data", new=AsyncMock())

    mock_conn = AsyncMock()
    mock_pool_cm = AsyncMock()
    mock_pool_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool_cm.__aexit__ = AsyncMock(return_value=False)
    mocker.patch("db.ingest_movie.pool.connection", return_value=mock_pool_cm)

    await ingest_movie.ingest_movie(movie)

    ingest_card.assert_awaited_once_with(movie, conn=mock_conn)
    ingest_lexical.assert_awaited_once_with(movie, conn=mock_conn)
    mock_conn.commit.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "movie_obj",
    [
        SimpleNamespace(tmdb_id=None),
        SimpleNamespace(tmdb_id=1, title=None),
        SimpleNamespace(tmdb_id=1, title="Movie", poster_url=None),
        SimpleNamespace(tmdb_id=1, title="Movie", poster_url="url", release_date=None),
        SimpleNamespace(tmdb_id=1, title="Movie", poster_url="url", release_date="2020-01-01", duration=None),
    ],
)
async def test_ingest_movie_card_missing_required_values_raise_value_error(movie_obj, mocker) -> None:
    """ingest_movie_card should raise ValueError when required fields are absent."""
    # Bypass debug file write side-effect at function start.
    mocker.patch("builtins.open", mocker.mock_open())
    with pytest.raises(ValueError):
        await ingest_movie.ingest_movie_card(movie_obj)


@pytest.mark.asyncio
async def test_ingest_movie_card_raises_for_invalid_maturity_rank(mocker) -> None:
    """ingest_movie_card should fail when maturity rating tuple is incomplete."""
    movie = SimpleNamespace(
        tmdb_id=1,
        title="Movie",
        poster_url="https://img.test/poster.png",
        release_date="2020-01-01",
        duration=120,
        maturity_rating_and_rank=lambda: (None, None),
    )
    # Bypass debug file write side-effect at function start.
    mocker.patch("builtins.open", mocker.mock_open())
    with pytest.raises(ValueError):
        await ingest_movie.ingest_movie_card(movie)


@pytest.mark.asyncio
async def test_ingest_movie_card_happy_path_calls_downstream_dependencies(mocker, base_movie_factory) -> None:
    """ingest_movie_card should compute values and call helper + movie-card upsert functions."""
    movie = base_movie_factory()
    create_genres = mocker.patch("db.ingest_movie.create_genre_ids", new=AsyncMock(return_value=[10, 11]))
    create_watch_keys = mocker.patch("db.ingest_movie.create_watch_offer_keys", new=AsyncMock(return_value=[100, 200]))
    create_langs = mocker.patch("db.ingest_movie.create_audio_language_ids", new=AsyncMock(return_value=[5]))
    upsert_card = mocker.patch("db.ingest_movie.upsert_movie_card", new=AsyncMock())

    # Bypass debug file write side-effect at function start.
    mocker.patch("builtins.open", mocker.mock_open())
    await ingest_movie.ingest_movie_card(movie)

    create_genres.assert_awaited_once_with(movie, conn=None)
    create_watch_keys.assert_awaited_once_with(movie, conn=None)
    create_langs.assert_awaited_once_with(movie, conn=None)
    kwargs = upsert_card.await_args.kwargs
    assert kwargs["movie_id"] == 1
    assert kwargs["title"] == "Spider-Man"
    assert kwargs["title_token_count"] == 3
    assert kwargs["conn"] is None


@pytest.mark.asyncio
async def test_ingest_lexical_data_requires_movie_id() -> None:
    """ingest_lexical_data should reject movie objects with missing tmdb_id."""
    with pytest.raises(ValueError):
        await ingest_movie.ingest_lexical_data(SimpleNamespace(tmdb_id=None))


@pytest.mark.asyncio
async def test_ingest_lexical_data_processes_all_entity_types(mocker) -> None:
    """ingest_lexical_data should batch-write title, people, character, and studio postings."""
    movie = SimpleNamespace(
        tmdb_id=9,
        characters=["Hero", "Villain"],
        production_companies=["Studio A"],
        normalized_title_tokens=lambda: ["spider-man", "hero"],
    )

    # Keep person iteration deterministic for assertions.
    mocker.patch("db.ingest_movie.create_people_list", return_value=["peter parker", "norman osborn"])

    # Batch dictionary resolution returns IDs for all relevant strings except villain.
    upsert_lexical = mocker.patch(
        "db.ingest_movie.batch_upsert_lexical_dictionary",
        new=AsyncMock(
            return_value={
                "spider-man": 101,
                "hero": 102,
                "peter parker": 201,
                "norman osborn": 202,
                "studio a": 401,
            }
        ),
    )
    upsert_title_string = mocker.patch("db.ingest_movie.batch_upsert_title_token_strings", new=AsyncMock())
    upsert_character_string = mocker.patch("db.ingest_movie.batch_upsert_character_strings", new=AsyncMock())
    insert_title = mocker.patch("db.ingest_movie.batch_insert_title_token_postings", new=AsyncMock())
    insert_person = mocker.patch("db.ingest_movie.batch_insert_person_postings", new=AsyncMock())
    insert_character = mocker.patch("db.ingest_movie.batch_insert_character_postings", new=AsyncMock())
    insert_studio = mocker.patch("db.ingest_movie.batch_insert_studio_postings", new=AsyncMock())

    await ingest_movie.ingest_lexical_data(movie)

    upsert_lexical.assert_awaited_once_with(
        [
            "spider-man",
            "hero",
            "peter parker",
            "norman osborn",
            "villain",
            "studio a",
        ],
        conn=None,
    )
    upsert_title_string.assert_awaited_once_with([101, 102], ["spider-man", "hero"], conn=None)
    insert_title.assert_awaited_once_with([101, 102], 9, conn=None)
    insert_person.assert_awaited_once_with([201, 202], 9, conn=None)
    upsert_character_string.assert_awaited_once_with([102], ["hero"], conn=None)
    insert_character.assert_awaited_once_with([102], 9, conn=None)
    insert_studio.assert_awaited_once_with([401], 9, conn=None)


@pytest.mark.asyncio
async def test_create_genre_ids_handles_normal_and_invalid_values(mocker) -> None:
    """create_genre_ids should delegate to movie.genre_ids()."""
    movie = SimpleNamespace(genre_ids=mocker.Mock(return_value=[1, 21]))

    result = await ingest_movie.create_genre_ids(movie)

    assert result == [1, 21]
    movie.genre_ids.assert_called_once_with()


@pytest.mark.asyncio
async def test_create_genre_ids_non_sequence_defaults_to_empty() -> None:
    """create_genre_ids should propagate attribute errors for invalid movie objects."""
    movie = SimpleNamespace(genres=123)
    with pytest.raises(AttributeError):
        await ingest_movie.create_genre_ids(movie)


@pytest.mark.asyncio
async def test_create_watch_offer_keys_deduplicates_and_sorts(mocker) -> None:
    """create_watch_offer_keys should delegate to movie.watch_offer_keys()."""
    expected = sorted({(8 << 4) | 1, (8 << 4) | 3})
    movie = SimpleNamespace(watch_offer_keys=mocker.Mock(return_value=expected))

    result = await ingest_movie.create_watch_offer_keys(movie)

    assert result == expected
    movie.watch_offer_keys.assert_called_once_with()


@pytest.mark.asyncio
async def test_create_watch_offer_keys_non_list_defaults_to_empty() -> None:
    """create_watch_offer_keys should propagate attribute errors for invalid movie objects."""
    movie = SimpleNamespace(watch_providers="invalid")
    with pytest.raises(AttributeError):
        await ingest_movie.create_watch_offer_keys(movie)


@pytest.mark.asyncio
async def test_create_audio_language_ids_paths(mocker) -> None:
    """create_audio_language_ids should delegate to movie.audio_language_ids()."""
    movie = SimpleNamespace(audio_language_ids=mocker.Mock(return_value=[80, 286]))

    result = await ingest_movie.create_audio_language_ids(movie)

    assert result == [80, 286]
    movie.audio_language_ids.assert_called_once_with()


@pytest.mark.asyncio
async def test_create_audio_language_ids_non_list_defaults_to_empty() -> None:
    """create_audio_language_ids should propagate attribute errors for invalid movie objects."""
    movie = SimpleNamespace(languages={"English"})
    with pytest.raises(AttributeError):
        await ingest_movie.create_audio_language_ids(movie)


def test_create_people_list_deduplicates_and_normalizes_names() -> None:
    """create_people_list should merge all person fields and normalize unique names."""
    movie = SimpleNamespace(
        actors=["Tom Hanks", " TOM HANKS "],
        directors=["Nora Ephron"],
        writers=["Nora Ephron", "Delia Ephron"],
        composers="not-a-list",
        producers=["Lynda Obst"],
    )
    people = ingest_movie.create_people_list(movie)
    assert people == {"tom hanks", "nora ephron", "delia ephron", "lynda obst"}
