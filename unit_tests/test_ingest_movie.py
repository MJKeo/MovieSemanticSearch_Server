"""Unit tests for db.ingest_movie methods."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from db import ingest_movie, postgres
from implementation.classes.schemas import WatchProvider


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
    """ingest_movie_card should compute values and call dictionary + movie-card upsert functions."""
    movie = base_movie_factory()
    upsert_maturity = mocker.patch("db.ingest_movie.upsert_maturity_dictionary", new=AsyncMock())
    create_genres = mocker.patch("db.ingest_movie.create_genre_ids", new=AsyncMock(return_value=[10, 11]))
    create_watch_keys = mocker.patch("db.ingest_movie.create_watch_offer_keys", new=AsyncMock(return_value=[100, 200]))
    create_langs = mocker.patch("db.ingest_movie.create_audio_language_ids", new=AsyncMock(return_value=[5]))
    upsert_card = mocker.patch("db.ingest_movie.upsert_movie_card", new=AsyncMock())

    # Bypass debug file write side-effect at function start.
    mocker.patch("builtins.open", mocker.mock_open())
    await ingest_movie.ingest_movie_card(movie)

    upsert_maturity.assert_awaited_once_with(3, "pg-13", conn=None)
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
    """create_genre_ids should normalize, skip invalids, and upsert dictionary rows."""
    movie = SimpleNamespace(genres=["Action", " ", "Sci-Fi"])

    upsert_lexical = mocker.patch(
        "db.ingest_movie.batch_upsert_lexical_dictionary",
        new=AsyncMock(side_effect=[{"action": 11}, {"sci-fi": 22}]),
    )
    upsert_genre = mocker.patch("db.ingest_movie.upsert_genre_dictionary", new=AsyncMock())

    result = await ingest_movie.create_genre_ids(movie)

    assert result == [11, 22]
    assert upsert_lexical.await_count == 2
    upsert_genre.assert_any_await(11, "action", conn=None)
    upsert_genre.assert_any_await(22, "sci-fi", conn=None)


@pytest.mark.asyncio
async def test_create_genre_ids_non_sequence_defaults_to_empty() -> None:
    """create_genre_ids should return empty list when genres is not a sequence."""
    movie = SimpleNamespace(genres=123)
    assert await ingest_movie.create_genre_ids(movie) == []


@pytest.mark.asyncio
async def test_create_watch_offer_keys_deduplicates_and_sorts(mocker) -> None:
    """create_watch_offer_keys should skip invalid providers and deduplicate generated keys."""
    providers = [
        WatchProvider(id=1, name="Netflix", logo_path="/n.png", display_priority=1, types=[1, 3, 99]),
        WatchProvider(id=2, name=" ", logo_path="/x.png", display_priority=2, types=[1]),
        WatchProvider(id=3, name="Apple TV", logo_path="/a.png", display_priority=2, types="not-a-list"),
        WatchProvider(id=4, name="Netflix", logo_path="/n2.png", display_priority=3, types=[1]),
    ]
    movie = SimpleNamespace(watch_providers=providers)

    # Map normalized provider names to deterministic lexical IDs.
    async def lexical_side_effect(norm_strings: list[str], conn=None):
        if norm_strings == ["netflix"]:
            return {"netflix": 42}
        if norm_strings == ["apple tv"]:
            return {"apple tv": 55}
        return {}

    mocker.patch("db.ingest_movie.batch_upsert_lexical_dictionary", new=AsyncMock(side_effect=lexical_side_effect))
    upsert_provider = mocker.patch("db.ingest_movie.upsert_provider_dictionary", new=AsyncMock())
    upsert_watch_method = mocker.patch("db.ingest_movie.upsert_watch_method_dictionary", new=AsyncMock())

    result = await ingest_movie.create_watch_offer_keys(movie)

    # Keys: (42,1), (42,3), and duplicate (42,1) from second Netflix provider should collapse.
    assert result == sorted({(42 << 4) | 1, (42 << 4) | 3})
    assert upsert_provider.await_count >= 2
    upsert_watch_method.assert_any_await(1, "subscription", conn=None)
    upsert_watch_method.assert_any_await(3, "rent", conn=None)


@pytest.mark.asyncio
async def test_create_watch_offer_keys_non_list_defaults_to_empty() -> None:
    """create_watch_offer_keys should return empty list when watch_providers is not a list."""
    movie = SimpleNamespace(watch_providers="invalid")
    assert await ingest_movie.create_watch_offer_keys(movie) == []


@pytest.mark.asyncio
async def test_create_audio_language_ids_paths(mocker) -> None:
    """create_audio_language_ids should normalize languages and upsert dictionary rows."""
    movie = SimpleNamespace(languages=["English", " ", "Spanish"])
    upsert_lexical = mocker.patch(
        "db.ingest_movie.batch_upsert_lexical_dictionary",
        new=AsyncMock(side_effect=[{"english": 10}, {"spanish": 20}]),
    )
    upsert_language = mocker.patch("db.ingest_movie.upsert_language_dictionary", new=AsyncMock())

    result = await ingest_movie.create_audio_language_ids(movie)

    assert result == [10, 20]
    assert upsert_lexical.await_count == 2
    upsert_language.assert_any_await(10, "English", conn=None)
    upsert_language.assert_any_await(20, "Spanish", conn=None)


@pytest.mark.asyncio
async def test_create_audio_language_ids_non_list_defaults_to_empty() -> None:
    """create_audio_language_ids should return empty list when languages is not a list."""
    movie = SimpleNamespace(languages={"English"})
    assert await ingest_movie.create_audio_language_ids(movie) == []


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
