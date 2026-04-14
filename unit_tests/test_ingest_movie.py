"""Unit tests for movie_ingestion.final_ingestion.ingest_movie methods."""

import importlib
import sys
from types import SimpleNamespace
from types import ModuleType
from unittest.mock import AsyncMock

import pytest

try:
    importlib.import_module("qdrant_client")
except ModuleNotFoundError:
    qdrant_module = ModuleType("qdrant_client")
    qdrant_models_module = ModuleType("qdrant_client.models")

    class _StubQdrantClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class _StubPointStruct:
        def __init__(self, *args, **kwargs) -> None:
            pass

    qdrant_module.QdrantClient = _StubQdrantClient
    qdrant_models_module.PointStruct = _StubPointStruct
    sys.modules["qdrant_client"] = qdrant_module
    sys.modules["qdrant_client.models"] = qdrant_models_module

try:
    importlib.import_module("implementation.llms.generic_methods")
except ModuleNotFoundError:
    generic_methods_module = ModuleType("implementation.llms.generic_methods")

    async def _stub_generate_vector_embedding(*args, **kwargs):
        return []

    generic_methods_module.generate_vector_embedding = _stub_generate_vector_embedding
    sys.modules["implementation.llms.generic_methods"] = generic_methods_module

try:
    importlib.import_module("implementation.vectorize")
except ModuleNotFoundError:
    vectorize_module = ModuleType("implementation.vectorize")

    def _stub_vectorize_text(*args, **kwargs) -> str:
        return ""

    vectorize_module.create_anchor_vector_text = _stub_vectorize_text
    vectorize_module.create_plot_events_vector_text = _stub_vectorize_text
    vectorize_module.create_plot_analysis_vector_text = _stub_vectorize_text
    vectorize_module.create_viewer_experience_vector_text = _stub_vectorize_text
    vectorize_module.create_watch_context_vector_text = _stub_vectorize_text
    vectorize_module.create_narrative_techniques_vector_text = _stub_vectorize_text
    vectorize_module.create_production_vector_text = _stub_vectorize_text
    vectorize_module.create_reception_vector_text = _stub_vectorize_text
    sys.modules["implementation.vectorize"] = vectorize_module

try:
    importlib.import_module("orjson")
except ModuleNotFoundError:
    orjson_module = ModuleType("orjson")

    def _stub_orjson_dumps(obj, *args, **kwargs):
        return json.dumps(obj).encode("utf-8")

    def _stub_orjson_loads(data, *args, **kwargs):
        if isinstance(data, (bytes, bytearray)):
            data = data.decode("utf-8")
        return json.loads(data)

    orjson_module.dumps = _stub_orjson_dumps
    orjson_module.loads = _stub_orjson_loads
    sys.modules["orjson"] = orjson_module

try:
    importlib.import_module("tiktoken")
except ModuleNotFoundError:
    tiktoken_module = ModuleType("tiktoken")

    class _StubEncoding:
        def encode(self, text):
            return list(text.encode("utf-8"))

    def _stub_encoding_for_model(*args, **kwargs):
        return _StubEncoding()

    tiktoken_module.encoding_for_model = _stub_encoding_for_model
    sys.modules["tiktoken"] = tiktoken_module

from movie_ingestion.final_ingestion import ingest_movie
from movie_ingestion.final_ingestion.ingest_movie import (
    BatchIngestionResult,
    IngestionError,
    MissingRequiredAttributeError,
    create_award_ceremony_win_ids,
)
from movie_ingestion.imdb_scraping.models import AwardNomination
from schemas.enums import AwardOutcome, LineagePosition
from schemas.metadata import FranchiseOutput
from schemas.movie import Movie, TMDBData, IMDBData
from db import postgres


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_movie(**overrides) -> Movie:
    """Build a minimal valid Movie with targeted overrides."""
    tmdb_defaults = {
        "tmdb_id": 1,
        "imdb_id": "tt0000001",
        "title": "Spider-Man",
        "release_date": "2002-05-03",
        "duration": 121,
        "budget": 139_000_000,
        "maturity_rating": "PG-13",
    }
    imdb_defaults = {
        "tmdb_id": 1,
        "original_title": None,
        "overview": "A student gets spider-like abilities.",
        "imdb_rating": 7.4,
        "metacritic_rating": 73.0,
        "budget": None,
        "genres": ["Action", "Adventure"],
        "countries_of_origin": ["USA"],
        "production_companies": ["Columbia Pictures"],
        "filming_locations": ["New York"],
        "languages": ["English"],
        "overall_keywords": ["hero", "city"],
        "maturity_rating": None,
        "maturity_reasoning": [],
        "characters": ["Peter Parker", "Mary Jane Watson", "Norman Osborn"],
        "directors": ["Sam Raimi"],
        "writers": ["David Koepp"],
        "producers": ["Laura Ziskin"],
        "composers": ["Danny Elfman"],
        "actors": ["Tobey Maguire"],
    }

    tmdb_overrides = overrides.pop("tmdb_data", {})
    imdb_overrides = overrides.pop("imdb_data", {})

    tmdb_data = TMDBData(**{**tmdb_defaults, **tmdb_overrides})
    imdb_data = IMDBData(**{**imdb_defaults, **imdb_overrides})
    return Movie(tmdb_data=tmdb_data, imdb_data=imdb_data, **overrides)


def _make_franchise_output(**overrides) -> FranchiseOutput:
    """Build a valid FranchiseOutput with targeted overrides."""
    defaults = {
        "lineage_reasoning": "Identified the franchise lineage.",
        "lineage": "shrek",
        "shared_universe": "shrek",
        "subgroups_reasoning": "This film belongs to a named subgroup.",
        "recognized_subgroups": ["puss in boots films"],
        "launched_subgroup": True,
        "position_reasoning": "This is a sequel branch.",
        "lineage_position": LineagePosition.SEQUEL,
        "crossover_reasoning": "No crossover franchises are present.",
        "is_crossover": False,
        "spinoff_reasoning": "This film is a spinoff branch.",
        "is_spinoff": True,
        "launch_reasoning": "It did not launch the broader franchise.",
        "launched_franchise": False,
    }
    return FranchiseOutput(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# upsert_movie_card (unchanged function)
# ---------------------------------------------------------------------------


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
            country_ids=(9, 10),
            source_material_type_ids=(11, 12),
            keyword_ids=(13, 14),
            concept_tag_ids=(15, 16),
            award_ceremony_win_ids=(1, 4),
            imdb_vote_count=945678,
            reception_score=72.5,
            title_token_count=4,
            box_office_bucket="hit",
        )
    finally:
        postgres._execute_on_conn = original

    conn_arg, query, params = execute_on_conn.await_args.args
    assert conn_arg is None
    assert "public.movie_card" in query
    assert params == (
        10, "Movie", "poster", 1000, 120, 3,
        [1, 2], [100, 200], [7, 8], [9, 10],
        [11, 12], [13, 14], [15, 16], [1, 4],
        945678, 72.5, None, "hit", 4,
    )


# ---------------------------------------------------------------------------
# ingest_movie (single-movie entry point)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_movie_runs_card_awards_and_lexical_ingestion(mocker) -> None:
    """ingest_movie should run card, awards, franchise, and lexical ingestion on one connection."""
    movie = _make_movie()
    ingest_card = mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.ingest_movie_card", new=AsyncMock()
    )
    ingest_awards = mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.ingest_movie_awards", new=AsyncMock()
    )
    ingest_franchise = mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.ingest_movie_franchise_metadata", new=AsyncMock()
    )
    ingest_lexical = mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.ingest_lexical_data", new=AsyncMock()
    )

    mock_conn = AsyncMock()
    mock_pool_cm = AsyncMock()
    mock_pool_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool_cm.__aexit__ = AsyncMock(return_value=False)
    mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.pool.connection",
        return_value=mock_pool_cm,
    )

    await ingest_movie.ingest_movie(movie)

    ingest_card.assert_awaited_once_with(movie, conn=mock_conn)
    ingest_awards.assert_awaited_once_with(movie, conn=mock_conn)
    ingest_franchise.assert_awaited_once_with(movie, conn=mock_conn)
    ingest_lexical.assert_awaited_once_with(movie, conn=mock_conn)
    mock_conn.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# ingest_movie_card — missing required fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_movie_card_missing_title_raises_missing_required() -> None:
    """ingest_movie_card should raise MissingRequiredAttributeError when title is None."""
    movie = _make_movie(tmdb_data={"title": None})
    with pytest.raises(MissingRequiredAttributeError):
        await ingest_movie.ingest_movie_card(movie)


@pytest.mark.asyncio
async def test_ingest_movie_card_missing_release_date_raises_missing_required() -> None:
    """ingest_movie_card should raise MissingRequiredAttributeError when release_date is None."""
    movie = _make_movie(tmdb_data={"release_date": None})
    with pytest.raises(MissingRequiredAttributeError):
        await ingest_movie.ingest_movie_card(movie)


@pytest.mark.asyncio
async def test_ingest_movie_card_missing_duration_raises() -> None:
    """ingest_movie_card should raise MissingRequiredAttributeError when duration is None."""
    movie = _make_movie(tmdb_data={"duration": None})
    with pytest.raises(MissingRequiredAttributeError):
        await ingest_movie.ingest_movie_card(movie)


# ---------------------------------------------------------------------------
# ingest_movie_card — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_movie_card_happy_path_calls_downstream_dependencies(mocker) -> None:
    """ingest_movie_card should compute values and call upsert_movie_card with correct params."""
    movie = _make_movie(
        imdb_data={"imdb_vote_count": 945678, "box_office_worldwide": 825_000_000},
    )
    upsert_card = mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.upsert_movie_card", new=AsyncMock()
    )

    await ingest_movie.ingest_movie_card(movie)

    kwargs = upsert_card.await_args.kwargs
    assert kwargs["movie_id"] == 1
    assert kwargs["title"] == "Spider-Man"
    assert kwargs["imdb_vote_count"] == 945678
    assert kwargs["budget_bucket"] == "large"
    assert kwargs["box_office_bucket"] == "hit"
    assert kwargs["title_token_count"] == 3  # spider-man, spider, man


@pytest.mark.asyncio
async def test_ingest_movie_card_ambiguous_box_office_stores_null(mocker) -> None:
    """ingest_movie_card should pass None when box office status is ambiguous or missing."""
    movie = _make_movie(imdb_data={"imdb_vote_count": 945678})
    upsert_card = mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.upsert_movie_card", new=AsyncMock()
    )

    await ingest_movie.ingest_movie_card(movie)

    kwargs = upsert_card.await_args.kwargs
    assert kwargs["box_office_bucket"] is None


# ---------------------------------------------------------------------------
# ingest_movie_franchise_metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_movie_franchise_metadata_upserts_row_and_dedupes_identical_terms(mocker) -> None:
    """Should store franchise metadata and index shared lineage/universe only once."""
    movie = _make_movie(franchise_metadata=_make_franchise_output())
    upsert_metadata = mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.upsert_movie_franchise_metadata",
        new=AsyncMock(),
    )
    upsert_lexical = mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.batch_upsert_lexical_dictionary",
        new=AsyncMock(return_value={"shrek": 11}),
    )
    replace_postings = mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.replace_movie_franchise_postings",
        new=AsyncMock(),
    )

    await ingest_movie.ingest_movie_franchise_metadata(movie)

    upsert_metadata.assert_awaited_once()
    assert upsert_metadata.await_args.args[0] == 1
    assert upsert_metadata.await_args.args[1] == movie.franchise_metadata
    assert upsert_metadata.await_args.kwargs["conn"] is None
    upsert_lexical.assert_awaited_once_with(["shrek"], conn=None)
    replace_postings.assert_awaited_once_with(1, [11], conn=None)


@pytest.mark.asyncio
async def test_ingest_movie_franchise_metadata_preserves_lineage_null_rows(mocker) -> None:
    """Should upsert franchise rows even when lineage is null."""
    movie = _make_movie(
        franchise_metadata=_make_franchise_output(
            lineage=None,
            shared_universe=None,
            recognized_subgroups=[],
            launched_subgroup=False,
            lineage_position=LineagePosition.REMAKE,
            is_spinoff=False,
        )
    )
    upsert_metadata = mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.upsert_movie_franchise_metadata",
        new=AsyncMock(),
    )
    upsert_lexical = mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.batch_upsert_lexical_dictionary",
        new=AsyncMock(return_value={}),
    )
    replace_postings = mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.replace_movie_franchise_postings",
        new=AsyncMock(),
    )

    await ingest_movie.ingest_movie_franchise_metadata(movie)

    upsert_metadata.assert_awaited_once()
    assert upsert_metadata.await_args.args[1] == movie.franchise_metadata
    assert upsert_metadata.await_args.kwargs["conn"] is None
    upsert_lexical.assert_awaited_once_with([], conn=None)
    replace_postings.assert_awaited_once_with(1, [], conn=None)


@pytest.mark.asyncio
async def test_ingest_movie_franchise_metadata_deletes_existing_state_when_metadata_missing(mocker) -> None:
    """Should delete stale Postgres franchise state when tracker franchise metadata is absent."""
    movie = _make_movie(franchise_metadata=None)
    delete_metadata = mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.delete_movie_franchise_metadata",
        new=AsyncMock(),
    )
    upsert_metadata = mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.upsert_movie_franchise_metadata",
        new=AsyncMock(),
    )

    await ingest_movie.ingest_movie_franchise_metadata(movie)

    delete_metadata.assert_awaited_once_with(1, conn=None)
    upsert_metadata.assert_not_awaited()


# ---------------------------------------------------------------------------
# ingest_lexical_data
# ---------------------------------------------------------------------------


## test_ingest_lexical_data_requires_movie_id removed — TMDBData.tmdb_id is a
## required int field (Pydantic rejects None at construction time), so the
## scenario of a Movie with tmdb_id=None is no longer constructable.


@pytest.mark.asyncio
async def test_ingest_lexical_data_processes_all_entity_types(mocker) -> None:
    """ingest_lexical_data should batch-write title, people, character, and studio postings."""
    movie = _make_movie(
        imdb_data={
            "characters": ["Hero", "Villain"],
            "production_companies": ["Studio A"],
            "directors": ["Nora Ephron"],
            "writers": [],
            "actors": [],
            "producers": [],
            "composers": [],
        },
    )

    # Keep person iteration deterministic for assertions.
    mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.create_people_list",
        return_value=["nora ephron"],
    )

    # Batch dictionary resolution returns IDs for all relevant strings.
    upsert_lexical = mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.batch_upsert_lexical_dictionary",
        new=AsyncMock(
            return_value={
                "spider-man": 101,
                "spider": 102,
                "man": 103,
                "nora ephron": 201,
                "hero": 301,
                "villain": 302,
                "studio a": 401,
            }
        ),
    )
    upsert_title_string = mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.batch_upsert_title_token_strings",
        new=AsyncMock(),
    )
    upsert_character_string = mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.batch_upsert_character_strings",
        new=AsyncMock(),
    )
    insert_title = mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.batch_insert_title_token_postings",
        new=AsyncMock(),
    )
    insert_person = mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.batch_insert_person_postings",
        new=AsyncMock(),
    )
    insert_character = mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.batch_insert_character_postings",
        new=AsyncMock(),
    )
    insert_studio = mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.batch_insert_studio_postings",
        new=AsyncMock(),
    )

    await ingest_movie.ingest_lexical_data(movie)

    # Title tokens, people, characters, and studios should all be resolved.
    upsert_lexical.assert_awaited_once()
    insert_title.assert_awaited_once()
    insert_person.assert_awaited_once()
    insert_character.assert_awaited_once()
    insert_studio.assert_awaited_once()


# ---------------------------------------------------------------------------
# Delegate helpers (genre_ids, watch_offer_keys, audio_language_ids)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_genre_ids_delegates_to_movie() -> None:
    """create_genre_ids should delegate to movie.genre_ids()."""
    movie = _make_movie(imdb_data={"genres": ["Action", "Drama"]})
    result = await ingest_movie.create_genre_ids(movie)
    # Should return integer genre IDs for known genres
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(gid, int) for gid in result)


@pytest.mark.asyncio
async def test_create_watch_offer_keys_delegates_to_movie() -> None:
    """create_watch_offer_keys should delegate to movie.watch_offer_keys()."""
    movie = _make_movie()
    result = await ingest_movie.create_watch_offer_keys(movie)
    # TMDBData.watch_provider_keys defaults to [] so result is []
    assert result == []


@pytest.mark.asyncio
async def test_create_audio_language_ids_delegates_to_movie() -> None:
    """create_audio_language_ids should delegate to movie.audio_language_ids()."""
    movie = _make_movie(imdb_data={"languages": ["English"]})
    result = await ingest_movie.create_audio_language_ids(movie)
    assert isinstance(result, list)
    assert len(result) == 1


def test_create_people_list_deduplicates_and_normalizes_names() -> None:
    """create_people_list should merge all person fields and normalize unique names."""
    movie = _make_movie(
        imdb_data={
            "actors": ["Tom Hanks", " TOM HANKS "],
            "directors": ["Nora Ephron"],
            "writers": ["Nora Ephron", "Delia Ephron"],
            "composers": [],
            "producers": ["Lynda Obst"],
        },
    )
    people = ingest_movie.create_people_list(movie)
    assert people == {"tom hanks", "nora ephron", "delia ephron", "lynda obst"}


# ---------------------------------------------------------------------------
# create_award_ceremony_win_ids
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_award_ceremony_win_ids_returns_distinct_winning_ceremony_ids() -> None:
    """Should return distinct ceremony IDs for wins only, preserving insertion order."""
    movie = _make_movie(imdb_data={
        "awards": [
            AwardNomination(ceremony="Academy Awards, USA", award_name="Oscar", category="Best Picture", outcome=AwardOutcome.WINNER, year=2024),
            AwardNomination(ceremony="Academy Awards, USA", award_name="Oscar", category="Best Director", outcome=AwardOutcome.WINNER, year=2024),
            AwardNomination(ceremony="Cannes Film Festival", award_name="Palme d'Or", category=None, outcome=AwardOutcome.WINNER, year=2023),
            AwardNomination(ceremony="Golden Globes, USA", award_name="Golden Globe", category="Best Picture - Drama", outcome=AwardOutcome.NOMINEE, year=2024),
        ],
    })
    result = await create_award_ceremony_win_ids(movie)
    # Academy Awards (1) and Cannes (4) are wins; Golden Globes (2) is nominee-only
    assert result == [1, 4]


@pytest.mark.asyncio
async def test_create_award_ceremony_win_ids_empty_when_no_awards() -> None:
    """Should return empty list when movie has no awards."""
    movie = _make_movie(imdb_data={"awards": []})
    result = await create_award_ceremony_win_ids(movie)
    assert result == []


@pytest.mark.asyncio
async def test_create_award_ceremony_win_ids_empty_when_all_nominees() -> None:
    """Should return empty list when all awards are nominations (no wins)."""
    movie = _make_movie(imdb_data={
        "awards": [
            AwardNomination(ceremony="Academy Awards, USA", award_name="Oscar", category="Best Picture", outcome=AwardOutcome.NOMINEE, year=2024),
        ],
    })
    result = await create_award_ceremony_win_ids(movie)
    assert result == []


# ---------------------------------------------------------------------------
# ingest_movie_awards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_movie_awards_calls_batch_upsert(mocker) -> None:
    """ingest_movie_awards should convert AwardNomination objects and call batch_upsert_movie_awards."""
    movie = _make_movie(imdb_data={
        "awards": [
            AwardNomination(ceremony="Academy Awards, USA", award_name="Oscar", category="Best Picture", outcome=AwardOutcome.WINNER, year=2024),
            AwardNomination(ceremony="Cannes Film Festival", award_name="Palme d'Or", category=None, outcome=AwardOutcome.WINNER, year=2023),
        ],
    })
    batch_upsert = mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.batch_upsert_movie_awards", new=AsyncMock()
    )

    await ingest_movie.ingest_movie_awards(movie)

    batch_upsert.assert_awaited_once()
    call_args = batch_upsert.await_args
    assert call_args.args[0] == 1  # movie_id
    awards = call_args.args[1]
    assert len(awards) == 2
    assert awards[0].ceremony_id == 1
    assert awards[0].category == "Best Picture"
    assert awards[0].outcome.outcome_id == 1
    assert awards[0].year == 2024
    assert awards[1].ceremony_id == 4
    assert awards[1].category is None


@pytest.mark.asyncio
async def test_ingest_movie_awards_skips_when_no_awards(mocker) -> None:
    """ingest_movie_awards should not call batch_upsert when movie has no awards."""
    movie = _make_movie(imdb_data={"awards": []})
    batch_upsert = mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.batch_upsert_movie_awards", new=AsyncMock()
    )

    await ingest_movie.ingest_movie_awards(movie)

    batch_upsert.assert_not_awaited()


# ---------------------------------------------------------------------------
# MissingRequiredAttributeError / IngestionError / BatchIngestionResult
# ---------------------------------------------------------------------------


def test_missing_required_attribute_error_is_value_error() -> None:
    """MissingRequiredAttributeError should be a subclass of ValueError."""
    assert issubclass(MissingRequiredAttributeError, ValueError)


def test_ingestion_error_stores_fields() -> None:
    """IngestionError should store tmdb_id and message."""
    err = IngestionError(tmdb_id=42, message="Postgres movie card: boom")
    assert err.tmdb_id == 42
    assert err.message == "Postgres movie card: boom"


def test_batch_ingestion_result_tracks_all_categories() -> None:
    """BatchIngestionResult should correctly store all tracking sets and lists."""
    result = BatchIngestionResult(
        succeeded_ids={1, 2},
        failed_ids={3},
        errors=[IngestionError(3, "failed")],
        filtered_ids={4},
        filter_reasons=[(4, "missing_required_attribute: no title")],
    )
    assert 1 in result.succeeded_ids
    assert 3 in result.failed_ids
    assert len(result.errors) == 1
    assert 4 in result.filtered_ids
    assert result.filter_reasons[0][0] == 4


# ---------------------------------------------------------------------------
# ingest_movies_to_postgres_batched
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_postgres_batched_happy_path(mocker) -> None:
    """ingest_movies_to_postgres_batched should return succeeded_ids for successful movies."""
    movies = [_make_movie(tmdb_data={"tmdb_id": 10}), _make_movie(tmdb_data={"tmdb_id": 20})]

    mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.ingest_movie_card", new=AsyncMock()
    )
    mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.ingest_movie_awards", new=AsyncMock()
    )
    mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.ingest_movie_franchise_metadata", new=AsyncMock()
    )
    mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.ingest_lexical_data", new=AsyncMock()
    )

    mock_conn = AsyncMock()
    mock_pool_cm = AsyncMock()
    mock_pool_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool_cm.__aexit__ = AsyncMock(return_value=False)
    mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.pool.connection",
        return_value=mock_pool_cm,
    )

    result = await ingest_movie.ingest_movies_to_postgres_batched(movies)

    assert result.succeeded_ids == {10, 20}
    assert result.failed_ids == set()
    assert result.errors == []


@pytest.mark.asyncio
async def test_postgres_batched_per_movie_isolation(mocker) -> None:
    """One movie failing should not prevent others from succeeding."""
    movies = [_make_movie(tmdb_data={"tmdb_id": 10}), _make_movie(tmdb_data={"tmdb_id": 20})]

    call_count = 0

    async def _card_side_effect(movie, conn=None):
        nonlocal call_count
        call_count += 1
        if movie.tmdb_data.tmdb_id == 10:
            raise RuntimeError("card failed")

    mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.ingest_movie_card",
        side_effect=_card_side_effect,
    )
    mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.ingest_movie_awards", new=AsyncMock()
    )
    mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.ingest_movie_franchise_metadata", new=AsyncMock()
    )
    mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.ingest_lexical_data", new=AsyncMock()
    )

    mock_conn = AsyncMock()
    mock_pool_cm = AsyncMock()
    mock_pool_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool_cm.__aexit__ = AsyncMock(return_value=False)
    mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.pool.connection",
        return_value=mock_pool_cm,
    )

    result = await ingest_movie.ingest_movies_to_postgres_batched(movies)

    assert 10 in result.failed_ids
    assert 20 in result.succeeded_ids
    assert any(e.tmdb_id == 10 for e in result.errors)


@pytest.mark.asyncio
async def test_postgres_batched_missing_required_attribute_goes_to_filtered(mocker) -> None:
    """Movies with MissingRequiredAttributeError should go to filtered_ids, not failed_ids."""
    movie = _make_movie(tmdb_data={"tmdb_id": 10, "title": None})

    mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.ingest_movie_awards", new=AsyncMock()
    )
    mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.ingest_movie_franchise_metadata", new=AsyncMock()
    )
    mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.ingest_lexical_data", new=AsyncMock()
    )

    mock_conn = AsyncMock()
    mock_pool_cm = AsyncMock()
    mock_pool_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool_cm.__aexit__ = AsyncMock(return_value=False)
    mocker.patch(
        "movie_ingestion.final_ingestion.ingest_movie.pool.connection",
        return_value=mock_pool_cm,
    )

    result = await ingest_movie.ingest_movies_to_postgres_batched([movie])

    assert 10 in result.filtered_ids
    assert 10 not in result.failed_ids
    assert len(result.filter_reasons) == 1
