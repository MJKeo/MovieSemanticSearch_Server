"""
Movie ingestion CLI and batch ingestion methods for Postgres and Qdrant.

Provides:
  - Per-movie ingestion (ingest_movie, ingest_movie_to_qdrant)
  - Batched ingestion with per-movie error isolation
    (ingest_movies_to_postgres_batched, ingest_movies_to_qdrant_batched)
  - CLI entry point for orchestrated parallel ingestion to both databases

Usage:
    python -m movie_ingestion.final_ingestion.ingest_movie \\
        --batch-size 200 \\
        --postgres-batch-size 100 \\
        --qdrant-batch-size 50 \\
        --max-movies 1000

    # Re-run only Postgres movie_card + lexical ingestion
    python -m movie_ingestion.final_ingestion.ingest_movie \\
        --disable-vectors
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from typing import List
from schemas.movie import Movie
from implementation.llms.generic_methods import generate_vector_embedding
from implementation.classes.enums import MaturityRating, VectorName
from .vector_text import (
    create_anchor_vector_text,
    create_plot_events_vector_text,
    create_plot_analysis_vector_text,
    create_viewer_experience_vector_text,
    create_watch_context_vector_text,
    create_narrative_techniques_vector_text,
    create_production_vector_text,
    create_reception_vector_text,
)
from implementation.misc.helpers import normalize_string
from movie_ingestion.tracker import TRACKER_DB_PATH, MovieStatus, log_ingestion_failures, batch_log_filter, PipelineStage
from db.postgres import (
    pool,
    _execute_read,
    batch_insert_title_token_postings,
    batch_insert_person_postings,
    batch_insert_character_postings,
    batch_insert_studio_postings,
    batch_upsert_lexical_dictionary,
    batch_upsert_title_token_strings,
    batch_upsert_character_strings,
    upsert_movie_card,
    refresh_title_token_doc_frequency,
    refresh_movie_popularity_scores,
)

# ---------------------------------------------------------------------------
# Clients (module-level singletons – reuse across calls)
# ---------------------------------------------------------------------------

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
# Ingestion upserts are much heavier than search queries (50 movies × 8
# vectors × 3072 dims per batch), so we set a generous timeout.
_qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)

EMBEDDING_MODEL = "text-embedding-3-large"
COLLECTION_ALIAS = "movies"


class MissingRequiredAttributeError(ValueError):
    """Raised when a movie is missing a required attribute for ingestion.

    Movies with missing required attributes cannot be fixed by retrying —
    they should be filtered out of the pipeline rather than marked as
    retryable ingestion failures.
    """
    pass


@dataclass(frozen=True, slots=True)
class IngestionError:
    """Structured error record from a batch ingestion step.

    Returned by both postgres and qdrant batch functions so that
    cmd_ingest can log all failures to the ingestion_failures table.
    The message should include a step prefix (e.g. "Postgres movie card: <error>").
    """
    tmdb_id: int
    message: str


@dataclass(slots=True)
class BatchIngestionResult:
    """Result from a batched ingestion function (Postgres or Qdrant).

    ``filtered_ids`` contains movies that are permanently ineligible
    (e.g. missing a required attribute) and should be marked as
    ``filtered_out`` rather than retried. Each entry in
    ``filter_reasons`` is a (tmdb_id, reason_string) tuple.
    """
    succeeded_ids: set[int]
    failed_ids: set[int]
    errors: list[IngestionError]
    filtered_ids: set[int]
    filter_reasons: list[tuple[int, str]]


def _build_skipped_qdrant_result(movies: list[Movie]) -> BatchIngestionResult:
    """Return a synthetic all-succeeded result when Qdrant is disabled."""
    succeeded_ids = {movie.tmdb_data.tmdb_id for movie in movies}
    return BatchIngestionResult(succeeded_ids, set(), [], set(), [])


# ================================
#       OVERALL INGESTION
# ================================

async def ingest_movie(movie: Movie) -> None:
    """
    Ingest one Movie into movie_card plus all lexical posting tables.

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

async def ingest_movie_card(movie: Movie, conn=None) -> None:
    """
    Extract fields from a Movie and upsert the canonical movie_card row.

    Args:
        movie: Movie object to ingest.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    try:
        movie_id = movie.tmdb_data.tmdb_id
        if movie_id is None:
            raise MissingRequiredAttributeError("ID is required but not found.")
        movie_id = int(movie_id)

        title = movie.tmdb_data.title
        if title is None:
            raise MissingRequiredAttributeError("Title is required but not found.")
        title = str(title)

        poster_url = str(movie.tmdb_data.poster_url) if movie.tmdb_data.poster_url else None

        release_ts = movie.release_ts()
        if release_ts is None:
            raise MissingRequiredAttributeError("Release date is required but not found.")

        runtime_value = movie.tmdb_data.duration
        if runtime_value is None:
            raise MissingRequiredAttributeError("Duration is required but not found.")
        runtime_minutes = int(runtime_value)

        maturity_rating, maturity_rank = movie.maturity_rating_and_rank()
        if not maturity_rating or not maturity_rank:
            raise MissingRequiredAttributeError(f"Maturity rating ({maturity_rating}) or rank ({maturity_rank}) is missing.")
        if maturity_rank == MaturityRating.UNRATED.maturity_rank:
            maturity_rank = None

        genre_ids = await create_genre_ids(movie)
        watch_offer_keys = await create_watch_offer_keys(movie)
        audio_language_ids = await create_audio_language_ids(movie)
        country_ids = await create_country_ids(movie)
        source_material_type_ids = await create_source_material_type_ids(movie)
        keyword_ids = await create_keyword_ids(movie)
        concept_tag_ids = await create_concept_tag_ids(movie)
        imdb_vote_count = int(movie.imdb_data.imdb_vote_count or 0)
        reception_score = movie.reception_score()
        title_token_count = len(movie.normalized_title_tokens())

        # Classify the movie's budget relative to its production era.
        # Returns BudgetSize.SMALL or BudgetSize.LARGE for outliers;
        # None for mid-range budgets or when budget data is unavailable.
        budget_bucket_result = movie.budget_bucket_for_era()
        budget_bucket = budget_bucket_result.value if budget_bucket_result is not None else None

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
            country_ids=country_ids,
            source_material_type_ids=source_material_type_ids,
            keyword_ids=keyword_ids,
            concept_tag_ids=concept_tag_ids,
            imdb_vote_count=imdb_vote_count,
            reception_score=reception_score,
            title_token_count=title_token_count,
            budget_bucket=budget_bucket,
            conn=conn,
        )
    except MissingRequiredAttributeError:
        # Propagate missing-attribute errors unwrapped so callers can
        # distinguish them from transient failures.
        raise
    except Exception as e:
        raise ValueError(f"Movie ingestion failed: {e}")


async def ingest_lexical_data(movie: Movie, conn=None) -> None:
    """
    Ingest all lexical posting data (title tokens, people, characters, studios) for a movie.

    Args:
        movie: Movie object to ingest lexical data for.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    movie_id = movie.tmdb_data.tmdb_id
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

    characters: list[str] = []
    for character in movie.imdb_data.characters:
        normalized_character = normalize_string(str(character))
        if not normalized_character:
            continue
        characters.append(normalized_character)

    studios: list[str] = []
    for studio in movie.imdb_data.production_companies:
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
#    BATCHED POSTGRES INGESTION
# ================================

async def ingest_movies_to_postgres_batched(
    movies: list[Movie],
    *,
    batch_size: int = 100,
) -> BatchIngestionResult:
    """
    Ingest movies to Postgres in batches with per-movie error isolation.

    For each sub-batch of ``batch_size`` movies:
      1. Acquire one connection from the pool.
      2. For each movie, use nested SAVEPOINTs to attempt movie_card and
         lexical inserts independently — both are always attempted so we
         can collect separate error messages for each step.
      3. An outer SAVEPOINT ensures per-movie atomicity: if either step
         fails, all changes for that movie are rolled back.
      4. Commit the entire transaction once at the end of the sub-batch.

    Args:
        movies:     List of Movie instances to ingest.
        batch_size: Movies per Postgres transaction (default: 100).
    """
    succeeded_ids: set[int] = set()
    failed_ids: set[int] = set()
    errors: list[IngestionError] = []
    filtered_ids: set[int] = set()
    filter_reasons: list[tuple[int, str]] = []

    for batch_start in range(0, len(movies), batch_size):
        batch = movies[batch_start : batch_start + batch_size]

        async with pool.connection() as conn:
            try:
                for movie in batch:
                    tmdb_id = movie.tmdb_data.tmdb_id
                    savepoint_name = f"sp_{tmdb_id}"

                    # Outer SAVEPOINT for per-movie atomicity. If this
                    # fails the connection is likely dead — propagate to
                    # the outer handler.
                    await conn.execute(f"SAVEPOINT {savepoint_name}")

                    # Attempt each step in its own inner SAVEPOINT so
                    # that a failure in movie_card doesn't prevent us
                    # from also attempting (and diagnosing) lexical.
                    # PostgreSQL enters an error state after a failed
                    # query — the inner savepoint rollback restores a
                    # clean transaction state for the next step.
                    card_error: str | None = None
                    lex_error: str | None = None

                    # --- movie_card step ---
                    inner_sp_card = f"sp_{tmdb_id}_card"
                    await conn.execute(f"SAVEPOINT {inner_sp_card}")
                    try:
                        await ingest_movie_card(movie, conn=conn)
                        await conn.execute(f"RELEASE SAVEPOINT {inner_sp_card}")
                    except MissingRequiredAttributeError as e:
                        # Missing required attribute — permanent data issue.
                        # Roll back the outer savepoint (covers the inner
                        # too) and skip this movie entirely.
                        await conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                        filtered_ids.add(tmdb_id)
                        filter_reasons.append((tmdb_id, f"missing_required_attribute: {e}"))
                        print(f"Postgres movie_card filtered for {tmdb_id}: {e}")
                        continue
                    except Exception as e:
                        card_error = str(e)
                        await conn.execute(f"ROLLBACK TO SAVEPOINT {inner_sp_card}")

                    # --- lexical step ---
                    inner_sp_lex = f"sp_{tmdb_id}_lex"
                    await conn.execute(f"SAVEPOINT {inner_sp_lex}")
                    try:
                        await ingest_lexical_data(movie, conn=conn)
                        await conn.execute(f"RELEASE SAVEPOINT {inner_sp_lex}")
                    except Exception as e:
                        lex_error = str(e)
                        await conn.execute(f"ROLLBACK TO SAVEPOINT {inner_sp_lex}")

                    if card_error or lex_error:
                        # Roll back the outer savepoint so neither partial
                        # result persists for this movie.
                        await conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                        failed_ids.add(tmdb_id)
                        if card_error:
                            print(f"Postgres movie_card failed for {tmdb_id}: {card_error}")
                            errors.append(IngestionError(tmdb_id, f"Postgres movie card: {card_error}"))
                        if lex_error:
                            print(f"Postgres lexical failed for {tmdb_id}: {lex_error}")
                            errors.append(IngestionError(tmdb_id, f"Postgres lexical: {lex_error}"))
                    else:
                        await conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                        succeeded_ids.add(tmdb_id)

                await conn.commit()
            except Exception as e:
                # Connection-level failure — commit failed, nothing was
                # persisted. Move every ID in this batch to failed.
                print(f"Postgres batch commit failed at offset {batch_start}: {e}")
                await conn.rollback()
                batch_ids = {m.tmdb_data.tmdb_id for m in batch}
                succeeded_ids -= batch_ids
                failed_ids |= batch_ids
                for mid in batch_ids:
                    errors.append(IngestionError(mid, f"Postgres batch commit: {e}"))

    return BatchIngestionResult(succeeded_ids, failed_ids, errors, filtered_ids, filter_reasons)


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

def _build_qdrant_payload(movie: Movie) -> dict:
    """
    Build the minimal hard-filter payload for the Qdrant point.

    Fields: release_ts, runtime_minutes, maturity_rank, genre_ids, watch_offer_keys
    """
    payload: dict = {}

    ts = movie.release_ts()
    if not ts:
        raise MissingRequiredAttributeError("Release date is required but not found.")
    payload["release_ts"] = ts

    runtime_value = movie.tmdb_data.duration
    if runtime_value is None:
        raise MissingRequiredAttributeError("Duration is required but not found.")
    payload["runtime_minutes"] = int(runtime_value)

    _, maturity_rank = movie.maturity_rating_and_rank()
    if not maturity_rank:
        raise MissingRequiredAttributeError("Maturity rank is missing.")
    if maturity_rank == MaturityRating.UNRATED.maturity_rank:
        payload["maturity_rank"] = None
    else:
        payload["maturity_rank"] = maturity_rank

    payload["genre_ids"] = movie.genre_ids()
    payload["watch_offer_keys"] = movie.watch_offer_keys()
    payload["audio_language_ids"] = movie.audio_language_ids()

    return payload

async def ingest_movie_to_qdrant(
    movie: Movie,
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
        movie:          Movie instance with all enriched metadata.
        collection:     Qdrant collection name/alias (default: "movies").
        client:         Optional QdrantClient override.

    Raises:
        ValueError: If movie is missing required ID.
        Exception:  On embedding or Qdrant failures.
    """
    try:
        qdrant = client or _qdrant_client

        # --- Validate required fields ---
        movie_id = movie.tmdb_data.tmdb_id

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
    movies: list[Movie],
    *,
    batch_size: int = 50,
    collection: str = COLLECTION_ALIAS,
    client: QdrantClient | None = None,
) -> BatchIngestionResult:
    """
    Ingest movies to Qdrant in batches, minimizing API calls.

    For each batch of N movies:
      1. Generate all N×8 text representations.
      2. Embed ALL texts in a single OpenAI API call.
      3. Build all N points (named vectors + payloads).
      4. Upsert all N points in a single Qdrant upsert.

    The sync QdrantClient upsert is offloaded to a thread via
    asyncio.to_thread so it does not block the event loop.

    Args:
        movies:      List of Movie instances to ingest.
        batch_size:  Number of movies per batch (default: 50).
        collection:  Qdrant collection name/alias.
        client:      Optional QdrantClient override.
    """
    qdrant = client or _qdrant_client
    vector_names = [name.value for name in VectorName]
    succeeded_ids: set[int] = set()
    failed_ids: set[int] = set()
    errors: list[IngestionError] = []
    filtered_ids: set[int] = set()
    filter_reasons: list[tuple[int, str]] = []

    for batch_start in range(0, len(movies), batch_size):
        batch = movies[batch_start : batch_start + batch_size]

        # ----------------------------------------------------------
        # Phase 1: Generate text representations for every movie
        # ----------------------------------------------------------
        # We track which texts are non-empty so we can map embeddings back.
        batch_texts: list[str] = []                         # flat list of all non-empty texts
        text_map: list[tuple[int, str]] = []                # (movie_idx_in_batch, vector_name)
        skipped_movies: set[int] = set()                    # indices with no valid texts
        movie_ids: list[int] = []                           # validated tmdb_ids per batch slot
        payloads: list[dict] = []                           # payloads per batch slot

        for movie_idx, movie in enumerate(batch):
            # --- Validate ID and build payload ---
            movie_id = movie.tmdb_data.tmdb_id
            movie_ids.append(movie_id)
            try:
                payloads.append(_build_qdrant_payload(movie))
            except MissingRequiredAttributeError as e:
                # Permanent data issue — filter out, don't retry.
                print(f"Qdrant payload filtered for movie {movie_id}: {e}")
                payloads.append({})
                skipped_movies.add(movie_idx)
                filtered_ids.add(movie_id)
                filter_reasons.append((movie_id, f"missing_required_attribute: {e}"))
                continue
            except Exception as e:
                print(f"Payload build failed for movie {movie_id}: {e}")
                payloads.append({})
                skipped_movies.add(movie_idx)
                failed_ids.add(movie_id)
                errors.append(IngestionError(movie_id, f"Qdrant payload build: {e}"))
                continue

            movie_has_text = False
            for vname in vector_names:
                text = _TEXT_GENERATORS[vname](movie)
                text = text.strip() if text else None
                if text:
                    char_count = len(text)
                    # ~4 chars per token; flag anything that might hit the 8191 token limit
                    if char_count > 25000:
                        print(f"⚠ LONG TEXT movie {movie_id} | {vname} | {char_count} chars (~{char_count // 4} tokens)")
                    text_map.append((movie_idx, vname))
                    batch_texts.append(text)
                    movie_has_text = True

            if not movie_has_text:
                # No embeddable text is a data-quality outcome, not an error.
                # Mark as succeeded so the pipeline moves past this movie,
                # but also track as skipped so later errors can't flip it to failed.
                print(f"No valid texts for movie {movie_id} — skipping")
                skipped_movies.add(movie_idx)
                succeeded_ids.add(movie_id)

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
            # Dump the texts in this batch to identify the culprit
            for tm_idx, (m_idx, vn) in enumerate(text_map):
                print(f"  batch text #{tm_idx}: movie {movie_ids[m_idx]} | {vn} | {len(batch_texts[tm_idx])} chars")
            # All non-skipped movies in this batch are failed
            for idx in range(len(batch)):
                if idx not in skipped_movies:
                    failed_ids.add(movie_ids[idx])
                    errors.append(IngestionError(movie_ids[idx], f"Qdrant embedding: {e}"))
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
        point_tmdb_ids: list[int] = []
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
            point_tmdb_ids.append(movie_ids[movie_idx])

        if points:
            try:
                # Offload sync upsert to a thread so it doesn't block the event loop
                await asyncio.to_thread(
                    qdrant.upsert,
                    collection_name=collection,
                    points=points,
                )
                succeeded_ids.update(point_tmdb_ids)
            except Exception as e:
                print(f"Qdrant upsert failed for batch starting at {batch_start}: {e}")
                failed_ids.update(point_tmdb_ids)
                for tid in point_tmdb_ids:
                    errors.append(IngestionError(tid, f"Qdrant upsert: {e}"))

    return BatchIngestionResult(succeeded_ids, failed_ids, errors, filtered_ids, filter_reasons)


# ================================
#       HELPER METHODS
# ================================

async def create_genre_ids(movie: Movie) -> List[int]:
    """
    Return genre IDs by delegating to ``Movie.genre_ids()``.

    Args:
        movie: Movie object implementing ``genre_ids()``.
    """
    genre_ids = movie.genre_ids()
    return genre_ids


async def create_watch_offer_keys(movie: Movie) -> List[int]:
    """
    Return watch-offer keys by delegating to ``Movie.watch_offer_keys()``.

    Args:
        movie: Movie object implementing ``watch_offer_keys()``.
    """
    return movie.watch_offer_keys()


async def create_audio_language_ids(movie: Movie) -> List[int]:
    """
    Return audio language IDs by delegating to ``Movie.audio_language_ids()``.

    Args:
        movie: Movie object implementing ``audio_language_ids()``.
    """
    language_ids = movie.audio_language_ids()
    return language_ids


async def create_country_ids(movie: Movie) -> List[int]:
    """
    Return country-of-origin IDs by delegating to ``Movie.country_ids()``.

    Args:
        movie: Movie object implementing ``country_ids()``.
    """
    return movie.country_ids()


async def create_source_material_type_ids(movie: Movie) -> List[int]:
    """
    Return source material type IDs by delegating to ``Movie.source_material_type_ids()``.

    Args:
        movie: Movie object implementing ``source_material_type_ids()``.
    """
    return movie.source_material_type_ids()


async def create_keyword_ids(movie: Movie) -> List[int]:
    """
    Return overall keyword IDs by delegating to ``Movie.keyword_ids()``.

    Args:
        movie: Movie object implementing ``keyword_ids()``.
    """
    return movie.keyword_ids()


async def create_concept_tag_ids(movie: Movie) -> List[int]:
    """
    Return concept tag IDs by delegating to ``Movie.concept_tag_ids()``.

    Args:
        movie: Movie object implementing ``concept_tag_ids()``.
    """
    return movie.concept_tag_ids()


def create_people_list(movie: Movie) -> set[str]:
    raw_people_lists = [
        movie.imdb_data.actors,
        movie.imdb_data.directors,
        movie.imdb_data.writers,
        movie.imdb_data.composers,
        movie.imdb_data.producers,
    ]
    people_names: set[str] = set()
    for people_list in raw_people_lists:
        for name in people_list:
            normalized_name = normalize_string(str(name))
            if normalized_name:
                people_names.add(normalized_name)

    return people_names


# ================================
#       CLI ORCHESTRATION
# ================================

async def _get_eligible_tmdb_ids(
    tracker_db_path: Path,
    max_movies: int | None = None,
) -> list[int]:
    """Return movie IDs whose movie_card rows still have empty country IDs.

    The backfill target lives entirely in Postgres, so we no longer
    constrain by tracker status here.

    Args:
        tracker_db_path: Unused. Retained for call-site stability.
        max_movies: Optional cap on total movies returned.

    Returns:
        List of tmdb_ids ready for ingestion.
    """
    query = """
        SELECT movie_id
        FROM public.movie_card
        WHERE cardinality(country_ids) = 0
        ORDER BY movie_id
    """
    params: tuple[object, ...] | None = None
    if max_movies is not None:
        query += " LIMIT %s"
        params = (max_movies,)

    rows = await _execute_read(query, params)
    eligible_ids = [row[0] for row in rows]
    return eligible_ids


def _mark_ingested(
    tracker_db_path: Path,
    tmdb_ids: set[int],
) -> None:
    """Batch-update movie_progress status to 'ingested' for successfully ingested movies.

    Also clears any old ingestion_failures rows for these movies, so that
    movies that succeed on retry don't leave stale failure records behind.
    """
    if not tmdb_ids:
        return

    ids_json = json.dumps(list(tmdb_ids))
    with sqlite3.connect(str(tracker_db_path)) as db:
        # json_each() expands a JSON array into a virtual table, giving us
        # a single UPDATE with one bound parameter — no placeholder string
        # building and no SQLITE_MAX_VARIABLE_NUMBER limit.
        db.execute(
            "UPDATE movie_progress SET status = ?, updated_at = CURRENT_TIMESTAMP "
            "WHERE tmdb_id IN (SELECT value FROM json_each(?))",
            (MovieStatus.INGESTED, ids_json),
        )
        # Purge old failure rows for movies that succeeded on retry.
        db.execute(
            "DELETE FROM ingestion_failures "
            "WHERE tmdb_id IN (SELECT value FROM json_each(?))",
            (ids_json,),
        )
        db.commit()


def _mark_ingestion_failed(
    tracker_db_path: Path,
    errors: list[IngestionError],
) -> None:
    """Log ingestion failures to SQLite and set status to ingestion_failed."""
    if not errors:
        return

    with sqlite3.connect(str(tracker_db_path)) as db:
        log_ingestion_failures(
            db,
            [(e.tmdb_id, e.message) for e in errors],
        )
        db.commit()


def _mark_filtered_out(
    tracker_db_path: Path,
    filter_reasons: list[tuple[int, str]],
) -> None:
    """Mark movies as filtered_out in the tracker and log the reason.

    Each entry in filter_reasons is (tmdb_id, reason_string). Uses
    batch_log_filter so both filter_log and movie_progress are updated
    atomically.
    """
    if not filter_reasons:
        return

    entries = [
        (tmdb_id, PipelineStage.INGESTION, reason, None)
        for tmdb_id, reason in filter_reasons
    ]
    with sqlite3.connect(str(tracker_db_path)) as db:
        batch_log_filter(db, entries)
        db.commit()


async def cmd_ingest(
    batch_size: int = 200,
    postgres_batch_size: int = 100,
    qdrant_batch_size: int = 50,
    max_movies: int | None = None,
    disable_vectors: bool = False,
    tracker_db_path: Path = TRACKER_DB_PATH,
) -> None:
    """Orchestrate parallel batch ingestion to Postgres and Qdrant.

    For each super-batch of ``batch_size`` movies:
      1. Batch-load Movie objects from the tracker SQLite DB.
      2. Run Postgres ingestion and, unless disabled, Qdrant ingestion.
      3. When vectors are enabled, intersect succeeded sets so only movies
         that pass BOTH are marked ingested. When vectors are disabled,
         Postgres success alone is sufficient.
      4. Update tracker status.
      5. Print progress.

    After all movies are processed, refreshes Postgres materialized views.
    """
    # Open the Postgres connection pool (created with open=False)
    await pool.open()

    cumulative_ingested = 0
    cumulative_failed = 0
    cumulative_filtered = 0
    start_time = time.time()

    try:
        all_tmdb_ids = await _get_eligible_tmdb_ids(tracker_db_path, max_movies)
        total = len(all_tmdb_ids)
        if not total:
            print("No eligible movies (movie_card.country_ids is already populated). Nothing to ingest.")
            return

        print(
            f"Found {total:,} eligible movies "
            f"(movie_card rows with empty country_ids).\n"
            f"Config: batch_size={batch_size}, postgres_batch_size={postgres_batch_size}, "
            f"qdrant_batch_size={qdrant_batch_size}, vectors={'disabled' if disable_vectors else 'enabled'}"
        )

        for super_start in range(0, total, batch_size):
            super_ids = all_tmdb_ids[super_start : super_start + batch_size]
            batch_start_time = time.time()

            # Step 1: Batch-load Movie objects from SQLite
            movies_map = Movie.from_tmdb_ids(super_ids, tracker_db_path)
            load_failures = set(super_ids) - set(movies_map.keys())
            if load_failures:
                print(f"  Failed to load {len(load_failures)} movie(s) from tracker DB")

            movie_list = list(movies_map.values())
            if not movie_list:
                # All movies in this super-batch failed to load — log them.
                load_errors = [
                    IngestionError(tid, "Movie load: failed to load from tracker DB")
                    for tid in super_ids
                ]
                _mark_ingestion_failed(tracker_db_path, load_errors)
                cumulative_failed += len(super_ids)
                continue

            # Step 2: Run Postgres ingestion and optionally Qdrant ingestion.
            if disable_vectors:
                pg = await ingest_movies_to_postgres_batched(
                    movie_list,
                    batch_size=postgres_batch_size,
                )
                qd = _build_skipped_qdrant_result(movie_list)
            else:
                pg_result, qdrant_result = await asyncio.gather(
                    ingest_movies_to_postgres_batched(movie_list, batch_size=postgres_batch_size),
                    ingest_movies_to_qdrant_batched(movie_list, batch_size=qdrant_batch_size),
                )
                pg = pg_result
                qd = qdrant_result

            # Step 3: Separate filtered (permanent) from failed (retryable).
            # Filtered movies are missing required attributes and should
            # never be retried — they get terminal filtered_out status.
            batch_filtered = pg.filtered_ids | qd.filtered_ids
            all_filter_reasons = pg.filter_reasons + qd.filter_reasons

            # Only movies that succeeded in BOTH databases are marked ingested.
            # Filtered movies are excluded from both succeeded and failed.
            both_succeeded = (pg.succeeded_ids & qd.succeeded_ids) - batch_filtered
            batch_failed = (pg.failed_ids | qd.failed_ids | load_failures) - batch_filtered

            # Step 4: Update tracker status for successfully ingested movies
            _mark_ingested(tracker_db_path, both_succeeded)

            # Step 4b: Mark permanently ineligible movies as filtered_out.
            _mark_filtered_out(tracker_db_path, all_filter_reasons)

            # Step 4c: Log failures to ingestion_failures table and mark
            # failed movies as ingestion_failed for retry on the next run.
            all_errors = pg.errors + qd.errors
            for tid in load_failures:
                all_errors.append(IngestionError(tid, "Movie load: failed to load from tracker DB"))
            _mark_ingestion_failed(tracker_db_path, all_errors)

            cumulative_ingested += len(both_succeeded)
            cumulative_failed += len(batch_failed)
            cumulative_filtered += len(batch_filtered)
            batch_elapsed = time.time() - batch_start_time

            # Step 5: Progress reporting
            processed = min(super_start + batch_size, total)
            filtered_note = f", {len(batch_filtered)} filtered" if batch_filtered else ""
            qdrant_summary = (
                "disabled"
                if disable_vectors
                else f"{len(qd.succeeded_ids)}/{len(qd.failed_ids)}"
            )
            print(
                f"[{processed:,}/{total:,}] "
                f"Batch: {len(both_succeeded)} ingested, {len(batch_failed)} failed{filtered_note} "
                f"(PG: {len(pg.succeeded_ids)}/{len(pg.failed_ids)}, "
                f"Qdrant: {qdrant_summary}) "
                f"| {batch_elapsed:.1f}s "
                f"| Cumulative: {cumulative_ingested:,} ingested, {cumulative_failed:,} failed, {cumulative_filtered:,} filtered"
            )

        # Post-ingestion: refresh materialized views
        if cumulative_ingested > 0:
            print("\nRefreshing materialized views...")
            await refresh_title_token_doc_frequency()
            await refresh_movie_popularity_scores()
            print("Materialized views refreshed.")

    finally:
        await pool.close()

    elapsed = time.time() - start_time
    print(
        f"\nDone. {cumulative_ingested:,} ingested, {cumulative_failed:,} failed, "
        f"{cumulative_filtered:,} filtered out of {total:,} total in {elapsed:.1f}s"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch movie ingestion to Postgres and Qdrant",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Movies loaded per super-batch iteration (default: 200). "
             "Controls memory usage and progress reporting granularity.",
    )
    parser.add_argument(
        "--postgres-batch-size",
        type=int,
        default=100,
        help="Movies per Postgres transaction (default: 100).",
    )
    parser.add_argument(
        "--qdrant-batch-size",
        type=int,
        default=50,
        help="Movies per Qdrant embedding + upsert cycle (default: 50).",
    )
    parser.add_argument(
        "--max-movies",
        type=int,
        default=None,
        help="Max total movies to ingest in this run (default: all eligible).",
    )
    parser.add_argument(
        "--disable-vectors",
        action="store_true",
        help="Skip Qdrant embedding/vector ingestion and run only Postgres movie_card + lexical ingestion.",
    )

    args = parser.parse_args()
    asyncio.run(
        cmd_ingest(
            batch_size=args.batch_size,
            postgres_batch_size=args.postgres_batch_size,
            qdrant_batch_size=args.qdrant_batch_size,
            max_movies=args.max_movies,
            disable_vectors=args.disable_vectors,
        )
    )


if __name__ == "__main__":
    main()
