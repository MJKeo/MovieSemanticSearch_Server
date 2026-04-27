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
from typing import List, Sequence
from schemas.enums import LineagePosition, release_format_id_for_imdb_type
from schemas.imdb_models import AwardNomination
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
from implementation.misc.helpers import expand_hyphen_variants, normalize_string
from implementation.misc.production_company_text import (
    normalize_company_string,
    tokenize_company_string,
)
from implementation.misc.franchise_text import (
    normalize_franchise_string,
    tokenize_franchise_string,
)
from implementation.misc.award_name_text import (
    normalize_award_string,
    tokenize_award_string,
)
from movie_ingestion.tracker import TRACKER_DB_PATH, MovieStatus, log_ingestion_failures, batch_log_filter, PipelineStage
from .brand_resolver import resolve_brands_for_movie
from db.postgres import (
    pool,
    batch_insert_actor_postings,
    batch_insert_director_postings,
    batch_insert_writer_postings,
    batch_insert_producer_postings,
    batch_insert_composer_postings,
    batch_insert_character_postings,
    batch_insert_brand_postings,
    batch_insert_studio_tokens,
    batch_insert_franchise_tokens,
    batch_insert_award_name_tokens,
    batch_upsert_production_companies,
    batch_upsert_franchise_entries,
    batch_upsert_award_name_entries,
    batch_upsert_lexical_dictionary,
    batch_upsert_character_strings,
    delete_movie_franchise_metadata,
    upsert_movie_franchise_metadata,
    upsert_movie_card,
    batch_upsert_movie_awards,
    refresh_studio_token_doc_frequency,
    refresh_franchise_token_doc_frequency,
    refresh_award_name_token_doc_frequency,
    refresh_movie_popularity_scores,
    fetch_movie_ids_missing_card,
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
            # Production and franchise data both run BEFORE the card upsert
            # and return the id arrays that ingest_movie_card stamps on the
            # card row in its single INSERT. Same reason in both cases:
            # avoids a wasteful "card with [] then UPDATE restores it" pair
            # of writes. The lex tables they touch (lex.production_company,
            # lex.studio_token, lex.inv_production_brand_postings,
            # lex.franchise_entry, lex.franchise_token) have no FK into
            # movie_card, so running them before the card upsert is safe.
            production_company_ids = await ingest_production_data(movie, conn=conn)
            franchise_ids = await ingest_franchise_data(movie, conn=conn)
            await ingest_movie_card(
                movie,
                production_company_ids=production_company_ids,
                lineage_entry_ids=franchise_ids.lineage,
                shared_universe_entry_ids=franchise_ids.shared_universe,
                subgroup_entry_ids=franchise_ids.subgroup,
                conn=conn,
            )
            await ingest_movie_awards(movie, conn=conn)
            await ingest_movie_franchise_metadata(movie, conn=conn)
            await ingest_lexical_data(movie, conn=conn)
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise


# ================================
#       POSTGRES INGESTION
# ================================

async def ingest_movie_card(
    movie: Movie,
    production_company_ids: Sequence[int] = (),
    lineage_entry_ids: Sequence[int] = (),
    shared_universe_entry_ids: Sequence[int] = (),
    subgroup_entry_ids: Sequence[int] = (),
    conn=None,
) -> None:
    """
    Extract fields from a Movie and upsert the canonical movie_card row.

    Args:
        movie: Movie object to ingest.
        production_company_ids: IDs from the freeform path's production_company
            upsert. Must be resolved *before* this call so the card gets the
            final array in its single upsert (no wasteful post-write UPDATE).
            Callers that don't have companies resolved yet can pass ``()`` and
            patch the column later via update_movie_card_production_company_ids.
        lineage_entry_ids: IDs from the lineage side of the franchise resolver
            (0 or 1 element). Resolved before this call for the same
            single-upsert reason.
        shared_universe_entry_ids: IDs from the shared_universe side of the
            franchise resolver (0 or 1 element). Stored separately from
            lineage so stage-3 can score lineage matches higher than
            universe-only matches when the query sets prefer_lineage.
        subgroup_entry_ids: IDs from the recognized_subgroups side of the
            franchise resolver (one per subgroup). Resolved before this call.
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
        country_of_origin_ids = await create_country_of_origin_ids(movie)
        source_material_type_ids = await create_source_material_type_ids(movie)
        keyword_ids = await create_keyword_ids(movie)
        concept_tag_ids = await create_concept_tag_ids(movie)
        award_ceremony_win_ids = await create_award_ceremony_win_ids(movie)
        imdb_vote_count = int(movie.imdb_data.imdb_vote_count or 0)
        reception_score = movie.reception_score()
        # Symmetric with Stage 3 title_pattern query-time normalization:
        # the query-side runs normalize_string() on the user pattern and
        # LIKE-matches this column, so normalize once here at ingest.
        title_normalized = normalize_string(title)

        # Classify the movie's budget relative to its production era.
        # Returns BudgetSize.SMALL or BudgetSize.LARGE for outliers;
        # None for mid-range budgets or when budget data is unavailable.
        budget_bucket_result = movie.budget_bucket_for_era()
        budget_bucket = budget_bucket_result.value if budget_bucket_result is not None else None
        box_office_bucket_result = movie.box_office_status()
        box_office_bucket = (
            box_office_bucket_result.value
            if box_office_bucket_result is not None
            else None
        )

        # Map IMDB titleType.id ('movie' / 'tvMovie' / 'short' / 'video' / other)
        # to a stable int. Anything outside the supported set — or a missing
        # value — collapses to UNKNOWN (0), which is the column's audit handle.
        release_format = release_format_id_for_imdb_type(movie.imdb_data.imdb_title_type)

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
            country_of_origin_ids=country_of_origin_ids,
            source_material_type_ids=source_material_type_ids,
            keyword_ids=keyword_ids,
            concept_tag_ids=concept_tag_ids,
            award_ceremony_win_ids=award_ceremony_win_ids,
            imdb_vote_count=imdb_vote_count,
            reception_score=reception_score,
            title_normalized=title_normalized,
            budget_bucket=budget_bucket,
            box_office_bucket=box_office_bucket,
            release_format=release_format,
            production_company_ids=production_company_ids,
            lineage_entry_ids=lineage_entry_ids,
            shared_universe_entry_ids=shared_universe_entry_ids,
            subgroup_entry_ids=subgroup_entry_ids,
            conn=conn,
        )
    except MissingRequiredAttributeError:
        # Propagate missing-attribute errors unwrapped so callers can
        # distinguish them from transient failures.
        raise
    except Exception as e:
        raise ValueError(f"Movie ingestion failed: {e}")


async def ingest_movie_franchise_metadata(movie: Movie, conn=None) -> None:
    """
    Upsert the raw TEXT-column franchise projection for one movie.

    Token-space resolution (lex.franchise_entry / lex.franchise_token and
    the movie_card.lineage_entry_ids / shared_universe_entry_ids /
    subgroup_entry_ids arrays) now runs earlier in ``ingest_movie`` via
    ``write_franchise_data`` so the card upsert can stamp those arrays in
    a single INSERT. This function is
    responsible ONLY for the raw lineage / shared_universe /
    recognized_subgroups / structural-flag columns on
    ``public.movie_franchise_metadata``, which survive as the debug /
    display / structural-flag backing.

    When the movie has no franchise metadata, any existing row is deleted so
    reruns can remove stale records.
    """
    movie_id = movie.tmdb_data.tmdb_id
    if movie_id is None:
        raise ValueError("Movie franchise ingestion failed: ID is required but not found.")
    movie_id = int(movie_id)

    franchise_metadata = movie.franchise_metadata
    if franchise_metadata is None:
        await delete_movie_franchise_metadata(movie_id, conn=conn)
        return

    await upsert_movie_franchise_metadata(
        movie_id,
        franchise_metadata,
        conn=conn,
    )


def _expand_positioned_names(
    distinct_names: list[str],
) -> tuple[list[str], list[tuple[str, int]]]:
    """
    Expand a billing-ordered list of distinct normalized names into
    ``(variant, billing_position)`` pairs used for posting writes.

    Each distinct name contributes every hyphen variant produced by
    :func:`expand_hyphen_variants`. Every variant from the same name
    shares the same 1-based billing position, so hyphen variants do not
    inflate prominence denominators (``cast_size`` /
    ``character_cast_size``) or create phantom credits.

    Returns:
        ``(unique_variant_strings, positioned_variants)`` where the first
        list is the deduplicated set of variant forms to register in
        the lexical dictionary and the second is the per-row source for
        the posting table insert.
    """
    positioned: list[tuple[str, int]] = []
    variant_set: dict[str, None] = {}
    for position, name in enumerate(distinct_names, start=1):
        for variant in expand_hyphen_variants(name):
            positioned.append((variant, position))
            variant_set.setdefault(variant, None)
    return list(variant_set.keys()), positioned


def _term_ids_and_positions(
    positioned: list[tuple[str, int]],
    string_id_map: dict[str, int],
) -> tuple[list[int], list[int]]:
    """
    Project a ``(variant, billing_position)`` list through the
    variant→term_id map to the parallel (term_ids, billing_positions)
    arrays that billing-aware ``batch_insert_*_postings`` consumes.

    Variants missing from ``string_id_map`` are skipped — that only
    happens when a variant normalized to the empty string upstream.
    """
    term_ids: list[int] = []
    positions: list[int] = []
    for variant, position in positioned:
        term_id = string_id_map.get(variant)
        if term_id is None:
            continue
        term_ids.append(term_id)
        positions.append(position)
    return term_ids, positions


def _term_ids_only(
    positioned: list[tuple[str, int]],
    string_id_map: dict[str, int],
) -> list[int]:
    """
    Project a ``(variant, billing_position)`` list to deduplicated
    term_ids for posting tables that don't carry billing metadata
    (director / writer / producer / composer).

    Dedup preserves first-seen order so the resulting list mirrors the
    billing order of the distinct names that produced it, even though
    the binary-role posting tables don't store that order.
    """
    term_ids: list[int] = []
    seen: set[int] = set()
    for variant, _ in positioned:
        term_id = string_id_map.get(variant)
        if term_id is None or term_id in seen:
            continue
        seen.add(term_id)
        term_ids.append(term_id)
    return term_ids


async def ingest_lexical_data(movie: Movie, conn=None) -> None:
    """
    Ingest all lexical posting data (people + characters) for a movie.

    For each role list, distinct normalized names define billing
    positions 1..N. Each name is then expanded via
    :func:`expand_hyphen_variants` so that every hyphen spelling
    ("spider-man" / "spider man" / "spiderman") is stored as its own
    term_id in ``lex.lexical_dictionary`` and ``lex.character_strings``,
    and every variant gets a posting row sharing the origin name's
    billing position. Prominence denominators (``cast_size`` /
    ``character_cast_size``) count distinct names pre-expansion so
    variant bloat never changes the prominence math.

    Studios are handled separately in ``ingest_production_data``.

    Args:
        movie: Movie object to ingest lexical data for.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    movie_id = movie.tmdb_data.tmdb_id
    if movie_id is None:
        raise ValueError("Movie ingestion failed: ID is required but not found.")
    movie_id = int(movie_id)

    # Phase 1: distinct normalized names per role, in billing order.
    # ``_normalize_name_list`` dedups via dict.fromkeys so the list
    # index directly defines the billing position (actor) or cast-edge
    # position (character).
    actors = _normalize_name_list(movie.imdb_data.actors)
    directors = _normalize_name_list(movie.imdb_data.directors)
    writers = _normalize_name_list(movie.imdb_data.writers)
    producers = _normalize_name_list(movie.imdb_data.producers)
    composers = _normalize_name_list(movie.imdb_data.composers)

    characters: list[str] = []
    seen_characters: set[str] = set()
    for character in movie.imdb_data.characters:
        normalized_character = normalize_string(str(character))
        if not normalized_character or normalized_character in seen_characters:
            continue
        seen_characters.add(normalized_character)
        characters.append(normalized_character)

    # Phase 2: expand each role list into (variant, billing_position)
    # pairs plus the per-role unique variant set. Cast sizes are frozen
    # now — BEFORE expansion — so hyphen variants do not inflate the
    # prominence denominators.
    actor_cast_size = len(actors)
    character_cast_size = len(characters)

    actor_variants, actor_positioned = _expand_positioned_names(actors)
    director_variants, director_positioned = _expand_positioned_names(directors)
    writer_variants, writer_positioned = _expand_positioned_names(writers)
    producer_variants, producer_positioned = _expand_positioned_names(producers)
    composer_variants, composer_positioned = _expand_positioned_names(composers)
    character_variants, character_positioned = _expand_positioned_names(characters)

    # Phase 3: resolve every unique variant to a term_id in one round-trip.
    all_variant_strings = list(
        dict.fromkeys(
            actor_variants
            + director_variants
            + writer_variants
            + producer_variants
            + composer_variants
            + character_variants
        )
    )
    string_id_map = await batch_upsert_lexical_dictionary(all_variant_strings, conn=conn)

    # Phase 4: register character variants in the character-specific
    # subset table (used by fuzzy character matching in Stage 3).
    character_variant_pairs = [
        (string_id_map[variant], variant)
        for variant in character_variants
        if variant in string_id_map
    ]
    await batch_upsert_character_strings(
        [string_id for string_id, _ in character_variant_pairs],
        [variant for _, variant in character_variant_pairs],
        conn=conn,
    )

    # Phase 5: posting writes. Sequential (not asyncio.gather) because a
    # shared psycopg AsyncConnection does not support concurrent
    # in-flight queries without pipeline mode.
    actor_term_ids, actor_positions = _term_ids_and_positions(actor_positioned, string_id_map)
    await batch_insert_actor_postings(
        actor_term_ids,
        actor_positions,
        movie_id,
        cast_size=actor_cast_size,
        conn=conn,
    )

    await batch_insert_director_postings(
        _term_ids_only(director_positioned, string_id_map), movie_id, conn=conn,
    )
    await batch_insert_writer_postings(
        _term_ids_only(writer_positioned, string_id_map), movie_id, conn=conn,
    )
    await batch_insert_producer_postings(
        _term_ids_only(producer_positioned, string_id_map), movie_id, conn=conn,
    )
    await batch_insert_composer_postings(
        _term_ids_only(composer_positioned, string_id_map), movie_id, conn=conn,
    )

    character_term_ids, character_positions = _term_ids_and_positions(
        character_positioned, string_id_map
    )
    await batch_insert_character_postings(
        character_term_ids,
        character_positions,
        movie_id,
        character_cast_size=character_cast_size,
        conn=conn,
    )


async def write_production_data(
    movie_id: int,
    production_companies: list[str],
    release_year: int | None,
    conn=None,
) -> list[int]:
    """
    Core brand + production-company writer. Decoupled from the ``Movie``
    object so the backfill script can call it with raw tracker data.

    Writes every table tied to production data EXCEPT ``movie_card`` — the
    caller stamps ``production_company_ids`` onto the card via whichever
    path fits its flow (``upsert_movie_card`` at ingest time, or
    ``update_movie_card_production_company_ids`` at backfill time). Returns
    the sorted, deduplicated list of ``production_company_id`` values so the
    caller can pass them on directly.

    Two coordinated paths replace the old ``lex.inv_studio_postings`` write:
      1. **Brand path** — resolve_brands_for_movie maps the production-
         company list to time-bounded ProductionBrand tags. Tags are written
         to lex.inv_production_brand_postings with first_matching_index (the
         position of the brand's first matching company in the IMDB list)
         and total_brand_count so the lexical scorer can weight prominence.
      2. **Freeform path** — every distinct normalized company string is
         upserted into lex.production_company (yielding a stable
         production_company_id), tokenized via tokenize_company_string, and
         recorded in lex.studio_token so the freeform query-time path can
         intersect tokens → companies → movies. See
         search_improvement_planning/v2_search_data_improvements.md.

    Idempotent: brand postings are replaced wholesale for the movie; company
    and token tables use ON CONFLICT DO NOTHING. Safe to re-run after a
    registry change or re-ingest.
    """
    # --- Brand path -------------------------------------------------------
    brand_tags = resolve_brands_for_movie(production_companies, release_year)
    brand_rows = [(tag.brand_id, tag.first_matching_index) for tag in brand_tags]
    await batch_insert_brand_postings(movie_id, brand_rows, conn=conn)

    # --- Freeform company path -------------------------------------------
    # Build (canonical, normalized) pairs, dropping any that normalize to
    # empty (punctuation-only inputs, etc.). Preserve the first canonical
    # form we see for each unique normalized string — downstream display
    # pulls canonical_string from the table.
    pairs: list[tuple[str, str]] = []
    for raw in production_companies:
        normalized = normalize_company_string(raw)
        if not normalized:
            continue
        pairs.append((raw, normalized))

    company_id_map = await batch_upsert_production_companies(pairs, conn=conn)

    # Emit every (token, company_id) pair. Tokenize from the already-
    # normalized form via already_normalized=True to skip a redundant
    # normalize_string pass. Dedup across the movie so we don't insert
    # identical pairs multiple times when two raw variants produce the
    # same normalized form.
    token_rows: set[tuple[str, int]] = set()
    for _, normalized in pairs:
        cid = company_id_map.get(normalized)
        if cid is None:
            continue
        for token in tokenize_company_string(normalized, already_normalized=True):
            token_rows.add((token, cid))
    await batch_insert_studio_tokens(list(token_rows), conn=conn)

    # Return the company-id set for the caller to stamp on movie_card.
    # Sorted for stable diffs and deterministic array contents.
    return sorted({cid for cid in company_id_map.values()})


async def ingest_production_data(movie: Movie, conn=None) -> list[int]:
    """
    Ingest brand and production-company data for one ``Movie``.

    Thin adapter over ``write_production_data`` that extracts
    production_companies and release_year from the Movie object. Returns the
    ``production_company_ids`` for the caller to pass to ``ingest_movie_card``.
    """
    movie_id = movie.tmdb_data.tmdb_id
    if movie_id is None:
        raise ValueError("Production ingestion failed: ID is required but not found.")
    movie_id = int(movie_id)

    raw_strings: list[str] = []
    if movie.imdb_data and movie.imdb_data.production_companies:
        raw_strings = [str(s) for s in movie.imdb_data.production_companies]

    # Derive release_year for the brand resolver's window check. YYYY-MM-DD
    # is the TMDB convention; accept partial/malformed values by returning
    # None rather than raising (the resolver drops any windowed membership
    # when release_year is None).
    release_year: int | None = None
    release_date = movie.tmdb_data.release_date
    if release_date:
        try:
            release_year = int(str(release_date)[:4])
        except (ValueError, TypeError):
            release_year = None

    return await write_production_data(movie_id, raw_strings, release_year, conn=conn)


@dataclass(frozen=True)
class FranchiseEntryIds:
    """
    Franchise entry-id arrays returned by ``write_franchise_data`` /
    ``ingest_franchise_data`` for stamping on ``movie_card``.

    Lineage and shared_universe are kept in separate attributes (not
    unioned) so stage-3 can score a lineage match higher than a
    universe-only match when the query sets ``prefer_lineage``. The
    token/entry space is still flat — the same franchise_entry_id can
    appear in one movie's lineage slot and another movie's shared_universe
    slot, which is what makes lineage-vs-universe scoring work.

    Each attribute is sorted and deduplicated. ``lineage`` and
    ``shared_universe`` have 0 or 1 element each; ``subgroup`` has one
    element per unique normalized subgroup.
    """

    lineage: list[int]
    shared_universe: list[int]
    subgroup: list[int]


async def write_franchise_data(
    lineage: str | None,
    shared_universe: str | None,
    recognized_subgroups: Sequence[str],
    conn=None,
) -> FranchiseEntryIds:
    """
    Core franchise writer. Decoupled from the ``Movie`` object so the
    backfill script can call it with raw Postgres rows.

    Resolves lineage + shared_universe + recognized_subgroups to
    ``lex.franchise_entry`` rows via normalize_franchise_string, tokenizes
    every normalized string, and records ``(token, franchise_entry_id)``
    pairs in ``lex.franchise_token`` for the query-side token intersection.
    Returns a ``FranchiseEntryIds`` dataclass so the caller can stamp its
    three arrays on ``movie_card`` in its single upsert.

    See search_improvement_planning/v2_search_data_improvements.md,
    "Franchise Resolution".

    Idempotent: all writes are ON CONFLICT DO NOTHING against unique
    constraints, and the returned arrays are deterministic given identical
    input. Safe to re-run after a normalizer change or re-ingest. Takes no
    ``movie_id`` because franchise registry tables are movie-scope-free —
    the caller stamps the returned ids onto ``movie_card``.

    Args:
        lineage: Raw lineage string from movie_franchise_metadata.lineage
            (or None).
        shared_universe: Raw shared_universe string (or None).
        recognized_subgroups: Raw subgroup strings. Empty sequence when there
            are no subgroups.
        conn: Optional existing async connection for caller-managed
            transaction scope.

    Returns:
        ``FranchiseEntryIds`` with the three arrays populated. When no
        franchise data reduces to any normalized form, all three arrays
        are empty.
    """
    # Build (canonical, normalized) pairs for every string that came in.
    # Each name is tracked with its source slot so we can reconstruct the
    # two separate arrays after the shared batch upsert. A single batch
    # round-trip still covers lineage + shared_universe + all subgroups.
    lineage_pairs: list[tuple[str, str]] = []
    if lineage:
        normalized = normalize_franchise_string(lineage)
        if normalized:
            lineage_pairs.append((lineage, normalized))

    shared_universe_pairs: list[tuple[str, str]] = []
    if shared_universe:
        normalized = normalize_franchise_string(shared_universe)
        if normalized:
            shared_universe_pairs.append((shared_universe, normalized))

    subgroup_pairs: list[tuple[str, str]] = []
    for raw in recognized_subgroups:
        if not raw:
            continue
        normalized = normalize_franchise_string(raw)
        if not normalized:
            continue
        subgroup_pairs.append((raw, normalized))

    all_pairs = lineage_pairs + shared_universe_pairs + subgroup_pairs
    if not all_pairs:
        return FranchiseEntryIds(lineage=[], shared_universe=[], subgroup=[])

    entry_id_map = await batch_upsert_franchise_entries(all_pairs, conn=conn)

    # Emit (token, franchise_entry_id) for every token of every normalized
    # string. Dedup across the movie so we don't insert the same pair
    # multiple times when two inputs share tokens or when the same string
    # appears under both lineage and shared_universe.
    token_rows: set[tuple[str, int]] = set()
    for _, normalized in all_pairs:
        fid = entry_id_map.get(normalized)
        if fid is None:
            continue
        for token in tokenize_franchise_string(normalized, already_normalized=True):
            token_rows.add((token, fid))
    await batch_insert_franchise_tokens(list(token_rows), conn=conn)

    # Reconstruct the three return arrays. A name present in both lineage
    # and shared_universe (rare — same normalized string) ends up in both
    # arrays; stage-3 scoring treats a movie that matches via both slots
    # as lineage (dominant slot wins).
    lineage_entry_ids = sorted({
        entry_id_map[n] for _, n in lineage_pairs if n in entry_id_map
    })
    shared_universe_entry_ids = sorted({
        entry_id_map[n] for _, n in shared_universe_pairs if n in entry_id_map
    })
    subgroup_entry_ids = sorted({
        entry_id_map[n] for _, n in subgroup_pairs if n in entry_id_map
    })
    return FranchiseEntryIds(
        lineage=lineage_entry_ids,
        shared_universe=shared_universe_entry_ids,
        subgroup=subgroup_entry_ids,
    )


async def ingest_franchise_data(movie: Movie, conn=None) -> FranchiseEntryIds:
    """
    Ingest franchise-entry and franchise-token data for one ``Movie``.

    Thin adapter over ``write_franchise_data`` that extracts lineage /
    shared_universe / recognized_subgroups from the Movie's
    ``franchise_metadata``. When franchise_metadata is absent the movie
    contributes no rows — returns an empty ``FranchiseEntryIds`` and
    ``ingest_movie_card`` stamps empty arrays, which is the correct
    stateless default.
    """
    franchise_metadata = movie.franchise_metadata
    if franchise_metadata is None:
        return FranchiseEntryIds(lineage=[], shared_universe=[], subgroup=[])

    return await write_franchise_data(
        franchise_metadata.lineage,
        franchise_metadata.shared_universe,
        franchise_metadata.recognized_subgroups,
        conn=conn,
    )


async def ingest_movie_awards(movie: Movie, conn=None) -> None:
    """
    Upsert all award rows for a movie into public.movie_awards.

    Filters to awards with known ceremony mappings, resolves each raw
    ``award_name`` to its ``lex.award_name_entry`` id (and writes the
    corresponding token rows), then passes the AwardNomination objects
    plus the aligned entry-id list to ``batch_upsert_movie_awards`` so
    the entry id is stamped in the same INSERT as the rest of the row.
    Doing the resolution inline here — rather than via a secondary
    UPDATE — mirrors the production-company / franchise ingest pattern
    and avoids a write-then-patch round trip.

    Args:
        movie: Movie object with populated ``imdb_data.awards``.
        conn: Optional existing async connection for caller-managed transaction scope.
    """
    if not movie.imdb_data or not movie.imdb_data.awards:
        return

    movie_id = movie.tmdb_data.tmdb_id
    if movie_id is None:
        raise ValueError("Movie award ingestion failed: ID is required but not found.")
    movie_id = int(movie_id)

    # Filter out awards with unknown ceremony strings — they can't be
    # mapped to a stable integer ID so we skip them silently.
    known_awards = [a for a in movie.imdb_data.awards if a.ceremony_id is not None]

    # Deduplicate by PK fields (ceremony_id, award_name, category, year).
    # IMDB can return multiple entries for the same category when a movie
    # has multiple nominees (e.g. two actors in Best Actor). Since we don't
    # store nominee names, keep only the best outcome (lowest outcome_id,
    # i.e. winner beats nominee).
    best_by_key: dict[tuple, AwardNomination] = {}
    for a in known_awards:
        key = (a.ceremony_id, a.award_name, a.category or "", a.year)
        existing = best_by_key.get(key)
        if existing is None or a.outcome.outcome_id < existing.outcome.outcome_id:
            best_by_key[key] = a
    deduped_awards = list(best_by_key.values())
    if not deduped_awards:
        return

    # --- Award-name entry + token resolution (freeform path) ---------
    # Normalize each raw award_name once and memoize so the lookup used
    # to build token rows is the same one used to build the
    # per-award_name_entry_id list stamped on movie_awards below.
    normalized_by_raw: dict[str, str] = {}
    for a in deduped_awards:
        if a.award_name in normalized_by_raw:
            continue
        normalized_by_raw[a.award_name] = normalize_award_string(a.award_name)

    # Upsert the distinct normalized values. ``batch_upsert_award_name_entries``
    # already drops empties and dedups internally, but we only pass
    # non-empty normalized strings so the return map covers everything
    # we care about.
    unique_normalized = [n for n in normalized_by_raw.values() if n]
    entry_id_map = await batch_upsert_award_name_entries(
        unique_normalized, conn=conn,
    )

    # Emit (token, entry_id) rows for every distinct normalized string.
    # Tokenize from the already-normalized form to skip a redundant
    # normalize pass per token.
    token_rows: set[tuple[str, int]] = set()
    for normalized in set(unique_normalized):
        eid = entry_id_map.get(normalized)
        if eid is None:
            continue
        for token in tokenize_award_string(normalized, already_normalized=True):
            token_rows.add((token, eid))
    await batch_insert_award_name_tokens(list(token_rows), conn=conn)

    # Build the per-award entry-id list aligned 1:1 with deduped_awards.
    # ``None`` is a valid value for award names that normalize to an
    # empty string (rare: punctuation-only names) — the column is
    # nullable and query-side intersections simply skip those rows.
    entry_ids: list[int | None] = []
    for a in deduped_awards:
        normalized = normalized_by_raw.get(a.award_name) or ""
        entry_ids.append(entry_id_map.get(normalized) if normalized else None)

    await batch_upsert_movie_awards(
        movie_id, deduped_awards, entry_ids, conn=conn,
    )


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
                    franchise_error: str | None = None
                    lex_error: str | None = None
                    production_error: str | None = None

                    # --- production data step (runs before card) ---
                    # Upserts lex.production_company, writes brand postings
                    # + studio_token rows, and returns production_company_ids
                    # for the card step to stamp in its single upsert. If
                    # this fails we pass an empty id list to ingest_movie_card
                    # so the card step can still run independently; the
                    # outer savepoint rolls back at the end anyway when any
                    # step errors out.
                    production_company_ids: list[int] = []
                    inner_sp_production = f"sp_{tmdb_id}_production"
                    await conn.execute(f"SAVEPOINT {inner_sp_production}")
                    try:
                        production_company_ids = await ingest_production_data(
                            movie, conn=conn
                        )
                        await conn.execute(f"RELEASE SAVEPOINT {inner_sp_production}")
                    except Exception as e:
                        production_error = str(e)
                        await conn.execute(f"ROLLBACK TO SAVEPOINT {inner_sp_production}")

                    # --- movie_card step ---
                    inner_sp_card = f"sp_{tmdb_id}_card"
                    await conn.execute(f"SAVEPOINT {inner_sp_card}")
                    try:
                        await ingest_movie_card(
                            movie,
                            production_company_ids=production_company_ids,
                            conn=conn,
                        )
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

                    # --- awards step ---
                    awards_error: str | None = None
                    inner_sp_awards = f"sp_{tmdb_id}_awards"
                    await conn.execute(f"SAVEPOINT {inner_sp_awards}")
                    try:
                        await ingest_movie_awards(movie, conn=conn)
                        await conn.execute(f"RELEASE SAVEPOINT {inner_sp_awards}")
                    except Exception as e:
                        awards_error = str(e)
                        await conn.execute(f"ROLLBACK TO SAVEPOINT {inner_sp_awards}")

                    # --- franchise step ---
                    inner_sp_franchise = f"sp_{tmdb_id}_franchise"
                    await conn.execute(f"SAVEPOINT {inner_sp_franchise}")
                    try:
                        await ingest_movie_franchise_metadata(movie, conn=conn)
                        await conn.execute(f"RELEASE SAVEPOINT {inner_sp_franchise}")
                    except Exception as e:
                        franchise_error = str(e)
                        await conn.execute(f"ROLLBACK TO SAVEPOINT {inner_sp_franchise}")

                    # --- lexical step ---
                    inner_sp_lex = f"sp_{tmdb_id}_lex"
                    await conn.execute(f"SAVEPOINT {inner_sp_lex}")
                    try:
                        await ingest_lexical_data(movie, conn=conn)
                        await conn.execute(f"RELEASE SAVEPOINT {inner_sp_lex}")
                    except Exception as e:
                        lex_error = str(e)
                        await conn.execute(f"ROLLBACK TO SAVEPOINT {inner_sp_lex}")

                    if card_error or awards_error or franchise_error or lex_error or production_error:
                        # Roll back the outer savepoint so neither partial
                        # result persists for this movie.
                        await conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                        failed_ids.add(tmdb_id)
                        if card_error:
                            print(f"Postgres movie_card failed for {tmdb_id}: {card_error}")
                            errors.append(IngestionError(tmdb_id, f"Postgres movie card: {card_error}"))
                        if awards_error:
                            print(f"Postgres awards failed for {tmdb_id}: {awards_error}")
                            errors.append(IngestionError(tmdb_id, f"Postgres awards: {awards_error}"))
                        if franchise_error:
                            print(f"Postgres franchise failed for {tmdb_id}: {franchise_error}")
                            errors.append(IngestionError(tmdb_id, f"Postgres franchise: {franchise_error}"))
                        if lex_error:
                            print(f"Postgres lexical failed for {tmdb_id}: {lex_error}")
                            errors.append(IngestionError(tmdb_id, f"Postgres lexical: {lex_error}"))
                        if production_error:
                            print(f"Postgres production failed for {tmdb_id}: {production_error}")
                            errors.append(IngestionError(tmdb_id, f"Postgres production: {production_error}"))
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


async def create_country_of_origin_ids(movie: Movie) -> List[int]:
    """
    Return country-of-origin IDs by delegating to ``Movie.country_of_origin_ids()``.

    Args:
        movie: Movie object implementing ``country_of_origin_ids()``.
    """
    return movie.country_of_origin_ids()


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


async def create_award_ceremony_win_ids(movie: Movie) -> list[int]:
    """
    Return distinct AwardCeremony IDs where this movie won (excludes nominees).

    Delegates to ``Movie.award_ceremony_win_ids()`` for the core logic.

    Args:
        movie: Movie object with populated ``imdb_data.awards``.
    """
    return movie.award_ceremony_win_ids()


def _normalize_name_list(names: list[str]) -> list[str]:
    """Normalize a list of names, preserving order and removing empties/duplicates.

    Order preservation is critical for actor lists where index encodes billing position.
    """
    result: list[str] = []
    seen: set[str] = set()
    for name in names:
        normalized = normalize_string(str(name))
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


# ================================
#       CLI ORCHESTRATION
# ================================

async def _get_eligible_tmdb_ids(
    tracker_db_path: Path,
    max_movies: int | None = None,
) -> list[int]:
    """Return movie IDs that have tracker status 'ingested' but no movie_card row.

    Two-step process:
      1. Pull all tmdb_ids with status 'ingested' from the tracker SQLite DB.
      2. Filter to only those missing a row in Postgres movie_card.

    Args:
        tracker_db_path: Path to the tracker SQLite database.
        max_movies: Optional cap on total movies returned.

    Returns:
        List of tmdb_ids ready for ingestion.
    """
    # Step 1: Get all "ingested" movie IDs from the tracker
    with sqlite3.connect(str(tracker_db_path)) as db:
        rows = db.execute(
            "SELECT tmdb_id FROM movie_progress WHERE status = ?",
            (MovieStatus.INGESTED,),
        ).fetchall()
    ingested_ids = [row[0] for row in rows]

    if not ingested_ids:
        return []

    # Step 2: Keep only those that don't already have a movie_card row
    eligible_ids = await fetch_movie_ids_missing_card(ingested_ids)
    eligible_ids.sort()

    if max_movies is not None:
        eligible_ids = eligible_ids[:max_movies]

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
            print("No eligible movies (all ingested movies already have a movie_card row). Nothing to ingest.")
            return

        print(
            f"Found {total:,} eligible movies "
            f"(movie_card rows do not exist).\n"
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
            await refresh_studio_token_doc_frequency()
            await refresh_franchise_token_doc_frequency()
            await refresh_award_name_token_doc_frequency()
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
