"""
Movie ingestion methods for database operations.

This module provides methods for ingesting movie data into the database.
"""

import asyncio
from typing import Optional, Sequence, List
from datetime import datetime, timezone
from implementation.classes.movie import BaseMovie
from implementation.classes.enums import WatchMethodType, MaturityRating
from implementation.misc.helpers import (
    normalize_string,
    create_watch_provider_offering_key,
)
from db.postgres import (
    batch_insert_title_token_postings,
    batch_insert_person_postings,
    batch_insert_character_postings,
    batch_insert_studio_postings,
    upsert_movie_card,
    upsert_phrase_term,
    upsert_lexical_dictionary,
    upsert_genre_dictionary,
    upsert_maturity_dictionary,
    upsert_language_dictionary,
    upsert_provider_dictionary,
    upsert_watch_method_dictionary,
    upsert_title_token_string,
    upsert_character_string,
)


async def ingest_movie(movie: BaseMovie) -> None:
    """Ingest one BaseMovie into movie_card plus all lexical posting tables in parallel."""
    await asyncio.gather(
        ingest_movie_card(movie),
        ingest_lexical_data(movie),
    )
    

async def ingest_movie_card(movie: BaseMovie) -> None:
    """Extract fields from a BaseMovie and upsert the canonical movie_card row."""
    try:
        movie_id = getattr(movie, "tmdb_id") or None
        if movie_id is None:
            raise ValueError("Movie ingestion failed: ID is required but not found.")
        movie_id = int(movie_id)

        title = getattr(movie, "title") or None
        if title is None:
            raise ValueError("Movie ingestion failed: Title is required but not found.")
        title = str(title)

        # Default value used here because BaseMovie does not define poster_url.
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
            # [DEBUG] Keep lookup table aligned with encoded maturity ranks.
            await upsert_maturity_dictionary(maturity_rank, maturity_rating)

        genre_ids = await create_genre_ids(movie)
        watch_offer_keys = await create_watch_offer_keys(movie)
        audio_language_ids = await create_audio_language_ids(movie)
        reception_score = movie.reception_score()
        title_token_count = len(movie.normalized_title_tokens())

        # Upsert canonical movie-card metadata so other systems can reference it.
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
        )
    except Exception as e:
        raise ValueError(f"Movie ingestion failed: {e}")


async def ingest_lexical_data(movie: BaseMovie) -> None:
    """Ingest all lexical posting data (title tokens, people, characters, studios) for a movie."""
    movie_id = getattr(movie, "tmdb_id") or None
    if movie_id is None:
        raise ValueError("Movie ingestion failed: ID is required but not found.")
    movie_id = int(movie_id)

    # Title token ingestion: dictionary + title_token_strings + postings.
    title_tokens = movie.normalized_title_tokens()
    title_token_term_ids: list[int] = []
    for token in title_tokens:
        normalized_token = normalize_string(token)
        if not normalized_token:
            continue
        # lex.lexical_dictionary
        token_term_id = await upsert_lexical_dictionary(normalized_token)
        if token_term_id is not None:
            title_token_term_ids.append(token_term_id)
            # [DEBUG] lex.title_token_strings
            await upsert_title_token_string(token_term_id, normalized_token)
    await batch_insert_title_token_postings(list(dict.fromkeys(title_token_term_ids)), movie_id)

    # Person name ingestion across actors/directors/writers/composers/producers.
    person_term_ids: list[int] = []
    for person in create_people_list(movie):
        normalized_person = normalize_string(person)
        if not normalized_person:
            continue
        # lex.lexical_dictionary
        person_term_id = await upsert_lexical_dictionary(normalized_person)
        if person_term_id is not None:
            person_term_ids.append(person_term_id)
    await batch_insert_person_postings(list(dict.fromkeys(person_term_ids)), movie_id)

    # Character phrase ingestion.
    raw_characters = getattr(movie, "characters", [])
    if not isinstance(raw_characters, list):
        # Default value used here because BaseMovie may not provide characters.
        raw_characters = []
    character_term_ids: list[int] = []
    for character in raw_characters:
        normalized_character = normalize_string(character)
        if not normalized_character:
            continue
        # lex.lexical_dictionary
        character_term_id = await upsert_phrase_term(normalized_character)
        if character_term_id is not None:
            character_term_ids.append(character_term_id)
            # [DEBUG] lex.character_strings
            await upsert_character_string(character_term_id, normalized_character)
    await batch_insert_character_postings(list(dict.fromkeys(character_term_ids)), movie_id)

    # Studio phrase ingestion.
    raw_studios = getattr(movie, "production_companies", [])
    if not isinstance(raw_studios, list):
        # Default value used here because BaseMovie may not provide production_companies.
        raw_studios = []
    studio_term_ids: list[int] = []
    for studio in raw_studios:
        normalized_studio = normalize_string(studio)
        if not normalized_studio:
            continue
        # lex.lexical_dictionary
        studio_term_id = await upsert_phrase_term(normalized_studio)
        if studio_term_id is not None:
            studio_term_ids.append(studio_term_id)
    await batch_insert_studio_postings(list(dict.fromkeys(studio_term_ids)), movie_id)


# ================================
#       HELPER METHODS
# ================================

async def create_genre_ids(movie: BaseMovie) -> List[int]:
    """Build the list of genre IDs for a movie, upserting into lexical + genre dictionaries."""
    raw_genres = getattr(movie, "genres", [])

    if not isinstance(raw_genres, Sequence):
        # Default value used here because BaseMovie may not provide genres.
        raw_genres = []
    
    genre_ids: List[int] = []
    for genre_name in raw_genres:
        # Normalize the string so ID fetching is consistent
        normalized_genre = normalize_string(str(genre_name))
        if not normalized_genre:
            continue

        # Derive genre ID from the shared lexical dictionary.
        genre_id = await upsert_lexical_dictionary(normalized_genre)
        if genre_id is not None:
            genre_ids.append(genre_id)
            # Keep lookup table populated for admin views, using the normalized
            # name so it is consistent with the cache-loader key format.
            await upsert_genre_dictionary(genre_id, normalized_genre)

    return genre_ids


async def create_watch_offer_keys(movie: BaseMovie) -> List[int]:
    """Build the sorted list of watch-offer keys for a movie, upserting provider/method dictionaries."""
    raw_providers = getattr(movie, "watch_providers", [])
    if not isinstance(raw_providers, list):
        # Default value used here because BaseMovie may not provide watch_providers.
        raw_providers = []

    watch_offer_key_set: set[int] = set()
    for provider in raw_providers:
        provider_name = str(getattr(provider, "name", "") or "")
        normalized_provider_name = normalize_string(provider_name)
        if not normalized_provider_name:
            continue

        # Derive provider ID from the shared lexical dictionary.
        provider_id = await upsert_lexical_dictionary(normalized_provider_name)
        if provider_id is None:
            continue

        # Keep lookup table populated for debugging/admin views.
        await upsert_provider_dictionary(provider_id, provider_name)

        watch_method_types = getattr(provider, "types", [])
        if not isinstance(watch_method_types, list):
            # Default value used here because WatchProvider may not provide types.
            watch_method_types = []

        for watch_method_type in watch_method_types:
            # provider_type is already an integer from WatchMethodType enum.
            watch_method_id = int(watch_method_type)
            
            # Get the enum member to fetch its string name for the dictionary.
            try:
                watch_method_enum = WatchMethodType(watch_method_id)
            except ValueError:
                continue
            
            await upsert_watch_method_dictionary(watch_method_id, str(watch_method_enum))

            watch_offer_key = create_watch_provider_offering_key(provider_id, watch_method_id)
            watch_offer_key_set.add(watch_offer_key)

    return sorted(watch_offer_key_set)


async def create_audio_language_ids(movie: BaseMovie) -> List[int]:
    """Build the list of audio language IDs for a movie, upserting into lexical + language dictionaries."""
    raw_languages = getattr(movie, "languages", [])

    if not isinstance(raw_languages, list):
        # Default value used here because BaseMovie may not provide languages.
        raw_languages = []

    audio_language_ids: list[int] = []
    for language in raw_languages:
        # Normalize the string so ID fetching is consistent
        normalized_language = normalize_string(str(language))
        if not normalized_language:
            continue

        # Derive language ID from the shared lexical dictionary.
        language_id = await upsert_lexical_dictionary(normalized_language)
        if language_id is not None:
            audio_language_ids.append(language_id)
            # [DEBUG] Keep lookup table populated for admin views.
            await upsert_language_dictionary(language_id, str(language))

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