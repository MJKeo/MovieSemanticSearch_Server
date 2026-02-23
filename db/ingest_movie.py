"""
Movie ingestion methods for database operations.

This module provides methods for ingesting movie data into the database.
"""


import os
import time
import json

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from typing import Optional, Sequence, List, TYPE_CHECKING
from datetime import datetime, timezone
from implementation.classes.movie import BaseMovie
from implementation.llms.generic_methods import generate_vector_embedding
from implementation.classes.enums import Genre, WatchMethodType, MaturityRating, VectorName
from implementation.vectorize import (
    create_anchor_vector_text,
    create_plot_events_vector_text,
    create_plot_analysis_vector_text,
    create_viewer_experience_vector_text,
    create_watch_context_vector_text,
    create_narrative_techniques_vector_text,
    create_production_vector_text,
    create_reception_vector_text,
)
from implementation.misc.helpers import (
    normalize_string,
    create_watch_provider_offering_key,
)
from implementation.classes.watch_providers import FILTERABLE_WATCH_PROVIDER_IDS
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

# ---------------------------------------------------------------------------
# Clients (module-level singletons – reuse across calls)
# ---------------------------------------------------------------------------

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
_qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_ALIAS = "movies"


# ================================
#       OVERALL INGESTION
# ================================

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


# ================================
#       POSTGRES INGESTION
# ================================

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

        release_ts = movie.release_ts()
        if release_ts is None:
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
#       QDRANT INGESTION
# ================================

# Map of vector name → text generation function (same order as VECTOR_NAMES)
_TEXT_GENERATORS: dict[str, callable] = {
    "anchor": create_anchor_vector_text,
    "plot_events": create_plot_events_vector_text,
    "plot_analysis": create_plot_analysis_vector_text,
    "viewer_experience": create_viewer_experience_vector_text,
    "watch_context": create_watch_context_vector_text,
    "narrative_techniques": create_narrative_techniques_vector_text,
    "production": create_production_vector_text,
    "reception": create_reception_vector_text,
}

def _build_qdrant_payload(movie: BaseMovie) -> dict:
    """
    Build the minimal hard-filter payload for the Qdrant point.

    Fields: release_ts, runtime_minutes, maturity_rank, genre_ids, watch_offer_keys
    """
    payload: dict = {}

    ts = movie.release_ts()
    if not ts:
        raise ValueError("Qdrant ingestion failed: Release date is required but not found.")
    payload["release_ts"] = ts
        

    runtime_value = getattr(movie, "duration", None)
    if runtime_value is None:
        raise ValueError("Qdrant ingestion failed: Duration is required but not found.")
    payload["runtime_minutes"] = int(runtime_value)

    _, maturity_rank = movie.maturity_rating_and_rank()
    if not maturity_rank:
        raise ValueError(f"Qdrant ingestion failed: Maturity rank is None")
    if maturity_rank == MaturityRating.UNRATED.value:
        payload["maturity_rank"] = None
    else:
        payload["maturity_rank"] = maturity_rank

    payload["genre_ids"] = movie.genre_ids()
    payload["watch_offer_keys"] = movie.watch_offer_keys()

    return payload

async def ingest_movie_to_qdrant(
    movie: BaseMovie,
    *,
    collection: str = COLLECTION_ALIAS,
    client: QdrantClient | None = None,
) -> None:
    """
    Generate all 8 vector embeddings for a movie and upsert to Qdrant.

    This function:
      1. Generates text representations for all 8 named vectors.
      2. Embeds all 8 texts in a **single** batched OpenAI API call.
      3. Builds the minimal payload for hard filtering.
      4. Upserts one point (with named vectors + payload) to Qdrant.

    Args:
        movie:          BaseMovie instance with all enriched metadata.
        collection:     Qdrant collection name/alias (default: "movies").
        client:         Optional QdrantClient override.
        openai_client:  Optional OpenAI client override.

    Raises:
        ValueError: If movie is missing required ID.
        Exception:  On embedding or Qdrant failures.
    """
    try:
        qdrant = client or _qdrant_client

        # --- Validate required fields ---
        movie_id = getattr(movie, "tmdb_id", None)
        if movie_id is None:
            raise ValueError("Movie must have a tmdb_id for Qdrant point ID.")
        movie_id = int(movie_id)

        # --- Step 1: Generate all 8 text representations ---
        texts: list[str] = []
        vector_names = [name.value for name in VectorName]
        for name in vector_names:
            text = _TEXT_GENERATORS[name](movie)
            text = text.strip() if text else None
            texts.append(text)

        # --- Step 2: Batch embed all 8 texts in a single API call ---
        filtered_texts = [text for text in texts if text]
        if not filtered_texts:
            raise ValueError("No valid text representations found for movie.")
        
        embeddings = await generate_vector_embedding(model=EMBEDDING_MODEL, text=filtered_texts)

        # --- Step 3: Build named vectors dict ---
        text_pointer, filtered_text_pointer = 0, 0
        vectors: dict[VectorName, list[float]] = {}
        while text_pointer < len(texts) and filtered_text_pointer < len(filtered_texts):
            if texts[text_pointer]:
                vectors[vector_names[text_pointer]] = embeddings[filtered_text_pointer]
                filtered_text_pointer += 1
            text_pointer += 1

        # --- Step 4: Build payload ---
        payload = _build_qdrant_payload(movie)

        # --- Step 5: Upsert single point ---
        qdrant.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=movie_id,
                    vector=vectors,
                    payload=payload,
                )
            ],
        )
    except Exception as e:
        raise ValueError(f"Qdrant ingestion failed: {e}")

async def ingest_movies_to_qdrant_batched(
    movies: list[BaseMovie],
    *,
    batch_size: int = 50,
    collection: str = COLLECTION_ALIAS,
    client: QdrantClient | None = None,
) -> dict[str, int]:
    """
    Ingest movies to Qdrant in batches, minimizing API calls.

    For each batch of N movies:
      1. Generate all N×8 text representations.
      2. Embed ALL texts in a single OpenAI API call.
      3. Build all N points (named vectors + payloads).
      4. Upsert all N points in a single Qdrant upsert.

    Args:
        movies:      List of BaseMovie instances to ingest.
        batch_size:  Number of movies per batch (default: 50).
        collection:  Qdrant collection name/alias.
        client:      Optional QdrantClient override.

    Returns:
        Summary dict with counts: {"ingested": int, "failed": int, "total": int}
    """
    qdrant = client or _qdrant_client
    vector_names = [name.value for name in VectorName]
    total = len(movies)
    ingested = 0
    failed = 0

    for batch_start in range(0, total, batch_size):
        batch = movies[batch_start : batch_start + batch_size]

        # ----------------------------------------------------------
        # Phase 1: Generate text representations for every movie
        # ----------------------------------------------------------
        # Each entry: (movie_index, vector_name, global_text_index)
        # We track which texts are non-empty so we can map embeddings back.
        batch_texts: list[str] = []                         # flat list of all non-empty texts
        text_map: list[tuple[int, str]] = []                # (movie_idx_in_batch, vector_name)
        skipped_movies: set[int] = set()                    # indices with no valid texts
        movie_ids: list[int] = []                           # validated tmdb_ids per batch slot
        payloads: list[dict] = []                           # payloads per batch slot

        for movie_idx, movie in enumerate(batch):
            # --- Validate ID ---
            movie_id = getattr(movie, "tmdb_id", None)
            if movie_id is None:
                print(f"Skipping movie without tmdb_id at batch index {batch_start + movie_idx}")
                skipped_movies.add(movie_idx)
                movie_ids.append(-1)
                payloads.append({})
                failed += 1
                continue

            movie_ids.append(int(movie_id))
            payloads.append(_build_qdrant_payload(movie))

            movie_has_text = False
            for vname in vector_names:
                text = _TEXT_GENERATORS[vname](movie)
                text = text.strip() if text else None
                if text:
                    text_map.append((movie_idx, vname))
                    batch_texts.append(text)
                    movie_has_text = True

            if not movie_has_text:
                print(f"No valid texts for movie {movie_id} — skipping")
                skipped_movies.add(movie_idx)
                failed += 1

        if not batch_texts:
            continue

        # ----------------------------------------------------------
        # Phase 2: Single batched embedding call for the whole batch
        # ----------------------------------------------------------
        try:
            all_embeddings = await generate_vector_embedding(
                model=EMBEDDING_MODEL,
                text=batch_texts,
            )
        except Exception as e:
            print(f"Embedding call failed for batch starting at {batch_start}: {e}")
            failed += len(batch) - len(skipped_movies)
            continue

        # ----------------------------------------------------------
        # Phase 3: Reassemble per-movie named vector dicts
        # ----------------------------------------------------------
        # movie_idx -> { vector_name: embedding }
        movie_vectors: dict[int, dict[str, list[float]]] = {}
        for emb_idx, (movie_idx, vname) in enumerate(text_map):
            movie_vectors.setdefault(movie_idx, {})[vname] = all_embeddings[emb_idx]

        # ----------------------------------------------------------
        # Phase 4: Build PointStructs and batch upsert
        # ----------------------------------------------------------
        points: list[PointStruct] = []
        for movie_idx in range(len(batch)):
            if movie_idx in skipped_movies:
                continue
            if movie_idx not in movie_vectors:
                continue

            points.append(
                PointStruct(
                    id=movie_ids[movie_idx],
                    vector=movie_vectors[movie_idx],
                    payload=payloads[movie_idx],
                )
            )

        if points:
            try:
                qdrant.upsert(
                    collection_name=collection,
                    points=points,
                )
                ingested += len(points)
            except Exception as e:
                print(f"Qdrant upsert failed for batch starting at {batch_start}: {e}")
                failed += len(points)

    return {"ingested": ingested, "failed": failed, "total": total}


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
    Build the list of genre IDs for a movie using the Genre enum's stable IDs.

    Each raw genre string is resolved to a Genre enum member via
    ``Genre.from_string``. Unrecognized genres are silently skipped.
    Matching the maturity-rank pattern, each resolved genre is also
    upserted into ``lex.genre_dictionary`` for human-readable lookups.

    Args:
        movie: Movie object containing genre data.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    raw_genres = getattr(movie, "genres", [])

    if not isinstance(raw_genres, Sequence):
        raw_genres = []

    genre_ids: List[int] = []
    for genre_name in raw_genres:
        genre_enum = Genre.from_string(str(genre_name))
        if genre_enum is None:
            continue

        genre_ids.append(genre_enum.genre_id)
        await upsert_genre_dictionary(genre_enum.genre_id, genre_enum.normalized_name, conn=conn)

    return genre_ids


async def create_watch_offer_keys(movie: BaseMovie, conn=None) -> List[int]:
    """
    Build the sorted list of watch-offer keys for a movie, upserting provider/method dictionaries.

    Only providers whose TMDB ID appears in FILTERABLE_WATCH_PROVIDER_IDS are
    included. The provider's TMDB ID is used directly as the provider component
    of each watch-offer key (no lexical dictionary round-trip).

    The provider_dictionary upsert is retained as a debug/sanity-check table
    but is never read at query time.

    Args:
        movie: Movie object containing watch provider data.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    raw_providers = getattr(movie, "watch_providers", [])
    if not isinstance(raw_providers, list):
        raw_providers = []

    watch_offer_key_set: set[int] = set()
    for provider in raw_providers:
        provider_id = getattr(provider, "id", None)
        if provider_id is None or provider_id not in FILTERABLE_WATCH_PROVIDER_IDS:
            continue

        provider_name = str(getattr(provider, "name", "") or "")
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