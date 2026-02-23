"""
Movie ingestion methods for database operations.

This module provides methods for ingesting movie data into the database.
"""

from typing import Optional, Sequence, List
from datetime import datetime, timezone
from implementation.classes.movie import BaseMovie
from implementation.classes.enums import WatchMethodType, MaturityRating
from implementation.misc.helpers import (
    normalize_string,
    create_watch_provider_offering_key,
)
from db.postgres import (
    pool,
    batch_insert_title_token_postings,
    batch_insert_person_postings,
    batch_insert_character_postings,
    batch_insert_studio_postings,
    batch_upsert_lexical_dictionary,
    batch_upsert_title_token_strings,
    batch_upsert_character_strings,
    upsert_movie_card,
    upsert_genre_dictionary,
    upsert_maturity_dictionary,
    upsert_language_dictionary,
    upsert_provider_dictionary,
    upsert_watch_method_dictionary,
)


async def ingest_movie(movie: BaseMovie) -> None:
    """
    Ingest one BaseMovie into movie_card plus all lexical posting tables.

    Acquires a single connection for the entire movie so all writes share one
    transaction. On success the transaction is committed; on failure it is
    rolled back, guaranteeing atomicity — a crash mid-ingest cannot leave
    partial data (e.g. movie_card present but postings missing).
    """
    async with pool.connection() as conn:
        try:
            await ingest_movie_card(movie, conn=conn)
            await ingest_lexical_data(movie, conn=conn)
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise

async def ingest_movie_card(movie: BaseMovie, conn=None) -> None:
    """
    Extract fields from a BaseMovie and upsert the canonical movie_card row.

    Args:
        movie: Movie object to ingest.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    try:
        movie_id = getattr(movie, "tmdb_id") or None
        if movie_id is None:
            raise ValueError("Movie ingestion failed: ID is required but not found.")
        movie_id = int(movie_id)

        title = getattr(movie, "title") or None
        if title is None:
            raise ValueError("Movie ingestion failed: Title is required but not found.")
        title = str(title)

        poster_url = getattr(movie, "poster_url", None)
        if poster_url is None:
            raise ValueError("Movie ingestion failed: Poster URL is required but not found.")
        poster_url = str(poster_url)

        release_date = getattr(movie, "release_date", None)
        if release_date:
            parsed_release = datetime.strptime(release_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            release_ts = int(parsed_release.timestamp())
        else:
            raise ValueError("Movie ingestion failed: Release date is required but not found.")

        runtime_value = getattr(movie, "duration", None)
        if runtime_value is None:
            raise ValueError("Movie ingestion failed: Duration is required but not found.")
        runtime_minutes = int(runtime_value)

        maturity_rating, maturity_rank = movie.maturity_rating_and_rank()
        if not maturity_rating or not maturity_rank:
            raise ValueError(f"Movie ingestion failed: One or more are None. Maturity rating: {maturity_rating} and rank: {maturity_rank}.")
        if maturity_rank == MaturityRating.UNRATED.value:
            maturity_rank = None
        else:
            await upsert_maturity_dictionary(maturity_rank, maturity_rating, conn=conn)

        genre_ids = await create_genre_ids(movie, conn=conn)
        watch_offer_keys = await create_watch_offer_keys(movie, conn=conn)
        audio_language_ids = await create_audio_language_ids(movie, conn=conn)
        reception_score = movie.reception_score()
        title_token_count = len(movie.normalized_title_tokens())

        await upsert_movie_card(
            movie_id=movie_id,
            title=title,
            poster_url=poster_url,
            release_ts=release_ts,
            runtime_minutes=runtime_minutes,
            maturity_rank=maturity_rank,
            genre_ids=genre_ids,
            watch_offer_keys=watch_offer_keys,
            audio_language_ids=audio_language_ids,
            reception_score=reception_score,
            title_token_count=title_token_count,
            conn=conn,
        )
    except Exception as e:
        raise ValueError(f"Movie ingestion failed: {e}")


async def ingest_lexical_data(movie: BaseMovie, conn=None) -> None:
    """
    Ingest all lexical posting data (title tokens, people, characters, studios) for a movie.

    Args:
        movie: Movie object to ingest lexical data for.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    movie_id = getattr(movie, "tmdb_id") or None
    if movie_id is None:
        raise ValueError("Movie ingestion failed: ID is required but not found.")
    movie_id = int(movie_id)

    # Phase 1: Collect normalized strings for every lexical bucket.
    title_tokens = [
        normalized_token
        for token in movie.normalized_title_tokens()
        if (normalized_token := normalize_string(token))
    ]
    people = list(create_people_list(movie))

    raw_characters = getattr(movie, "characters", [])
    if not isinstance(raw_characters, list):
        raw_characters = []
    characters: list[str] = []
    for character in raw_characters:
        normalized_character = normalize_string(str(character))
        if not normalized_character:
            continue
        characters.append(normalized_character)

    raw_studios = getattr(movie, "production_companies", [])
    if not isinstance(raw_studios, list):
        raw_studios = []
    studios: list[str] = []
    for studio in raw_studios:
        normalized_studio = normalize_string(str(studio))
        if not normalized_studio:
            continue
        studios.append(normalized_studio)

    # Phase 2: Resolve all dictionary IDs in one round-trip.
    all_strings = list(dict.fromkeys(title_tokens + people + characters + studios))
    string_id_map = await batch_upsert_lexical_dictionary(all_strings, conn=conn)

    # Phase 3: Build aligned ID/string arrays and posting ID lists.
    title_token_string_pairs = [
        (string_id_map[token], token) for token in title_tokens if token in string_id_map
    ]
    character_string_pairs = [
        (string_id_map[character], character)
        for character in characters
        if character in string_id_map
    ]

    title_token_term_ids = [string_id for string_id, _ in title_token_string_pairs]
    person_term_ids = [string_id_map[person] for person in people if person in string_id_map]
    character_term_ids = [string_id for string_id, _ in character_string_pairs]
    studio_term_ids = [string_id_map[studio] for studio in studios if studio in string_id_map]

    # Phase 4: Sequential batch writes to lexical sub-tables and posting tables.
    # Sequential (not asyncio.gather) because a shared psycopg AsyncConnection
    # does not support concurrent in-flight queries without pipeline mode.
    await batch_upsert_title_token_strings(
        [string_id for string_id, _ in title_token_string_pairs],
        [norm_str for _, norm_str in title_token_string_pairs],
        conn=conn,
    )
    await batch_upsert_character_strings(
        [string_id for string_id, _ in character_string_pairs],
        [norm_str for _, norm_str in character_string_pairs],
        conn=conn,
    )
    await batch_insert_title_token_postings(list(dict.fromkeys(title_token_term_ids)), movie_id, conn=conn)
    await batch_insert_person_postings(list(dict.fromkeys(person_term_ids)), movie_id, conn=conn)
    await batch_insert_character_postings(list(dict.fromkeys(character_term_ids)), movie_id, conn=conn)
    await batch_insert_studio_postings(list(dict.fromkeys(studio_term_ids)), movie_id, conn=conn)


# ================================
#       HELPER METHODS
# ================================

async def _resolve_single_lexical_string_id(normalized_value: str, conn=None) -> Optional[int]:
    """
    Resolve one normalized string ID through the batch dictionary upsert API.

    Args:
        normalized_value: Already-normalized string to resolve.
        conn: Optional existing async connection for caller-managed transaction scope.

    Returns:
        Resolved lexical string_id, or None when the input does not resolve.
    """
    if not normalized_value:
        return None

    string_id_map = await batch_upsert_lexical_dictionary([normalized_value], conn=conn)
    return string_id_map.get(normalized_value)


async def create_genre_ids(movie: BaseMovie, conn=None) -> List[int]:
    """
    Build the list of genre IDs for a movie, upserting into lexical + genre dictionaries.

    Args:
        movie: Movie object containing genre data.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    raw_genres = getattr(movie, "genres", [])

    if not isinstance(raw_genres, Sequence):
        raw_genres = []
    
    genre_ids: List[int] = []
    for genre_name in raw_genres:
        normalized_genre = normalize_string(str(genre_name))
        if not normalized_genre:
            continue

        genre_id = await _resolve_single_lexical_string_id(normalized_genre, conn=conn)
        if genre_id is not None:
            genre_ids.append(genre_id)
            await upsert_genre_dictionary(genre_id, normalized_genre, conn=conn)

    return genre_ids


async def create_watch_offer_keys(movie: BaseMovie, conn=None) -> List[int]:
    """
    Build the sorted list of watch-offer keys for a movie, upserting provider/method dictionaries.

    Args:
        movie: Movie object containing watch provider data.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    raw_providers = getattr(movie, "watch_providers", [])
    if not isinstance(raw_providers, list):
        raw_providers = []

    watch_offer_key_set: set[int] = set()
    for provider in raw_providers:
        provider_name = str(getattr(provider, "name", "") or "")
        normalized_provider_name = normalize_string(provider_name)
        if not normalized_provider_name:
            continue

        provider_id = await _resolve_single_lexical_string_id(normalized_provider_name, conn=conn)
        if provider_id is None:
            continue

        await upsert_provider_dictionary(provider_id, provider_name, conn=conn)

        watch_method_types = getattr(provider, "types", [])
        if not isinstance(watch_method_types, list):
            watch_method_types = []

        for watch_method_type in watch_method_types:
            watch_method_id = int(watch_method_type)
            
            try:
                watch_method_enum = WatchMethodType(watch_method_id)
            except ValueError:
                continue
            
            await upsert_watch_method_dictionary(watch_method_id, str(watch_method_enum), conn=conn)

            watch_offer_key = create_watch_provider_offering_key(provider_id, watch_method_id)
            watch_offer_key_set.add(watch_offer_key)

    return sorted(watch_offer_key_set)


async def create_audio_language_ids(movie: BaseMovie, conn=None) -> List[int]:
    """
    Build the list of audio language IDs for a movie, upserting into lexical + language dictionaries.

    Args:
        movie: Movie object containing language data.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    raw_languages = getattr(movie, "languages", [])

    if not isinstance(raw_languages, list):
        raw_languages = []

    audio_language_ids: list[int] = []
    for language in raw_languages:
        normalized_language = normalize_string(str(language))
        if not normalized_language:
            continue

        language_id = await _resolve_single_lexical_string_id(normalized_language, conn=conn)
        if language_id is not None:
            audio_language_ids.append(language_id)
            await upsert_language_dictionary(language_id, str(language), conn=conn)

    return audio_language_ids


def create_people_list(movie: BaseMovie) -> List[str]:
    raw_people_lists = [
        getattr(movie, "actors", []),
        getattr(movie, "directors", []),
        getattr(movie, "writers", []),
        getattr(movie, "composers", []),
        getattr(movie, "producers", []),
    ]
    people_names: set[str] = set()
    for people_list in raw_people_lists:
        if not isinstance(people_list, list):
            # Default value used here because BaseMovie may not provide a person list.
            people_list = []
            
        for name in people_list:
            normalized_name = normalize_string(str(name))
            if normalized_name:
                people_names.add(normalized_name)

    return people_names