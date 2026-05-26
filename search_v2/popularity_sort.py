# Shared popularity-sort helper for Step-0 entity-flow executors.
#
# Multiple executors (character_franchise_search, studio_search)
# bucket matched movie_ids and then sort each bucket by popularity
# DESC. The sort key convention — NULLS LAST under DESC with a
# movie_id DESC tiebreaker — is identical across them, so the helper
# lives here rather than duplicated in each search module. (The
# person flow inlines an equivalent key because it composes
# overlap_count as the primary within-bucket sort component on top
# of popularity.)
#
# The popularity dict shape mirrors db.postgres.fetch_quality_popularity_signals:
# {movie_id: (popularity_score | None, reception_score | None)}.
# Only the popularity slot is read; reception is ignored here.

from __future__ import annotations

from collections.abc import Iterable


def sort_movie_ids_by_popularity(
    movie_ids: Iterable[int],
    popularity: dict[int, tuple[float | None, float | None]],
) -> list[int]:
    """Sort movie_ids by popularity DESC, NULLS LAST, movie_id DESC.

    NULLS-LAST under DESC is encoded via a per-key tuple where the
    first element flags whether popularity is present (1 = present,
    0 = None). Python's tuple comparison under `reverse=True` then
    pushes None-popularity movies to the end of the sort.

    Args:
        movie_ids: Any iterable of movie_ids — set, list, or generator.
            Duplicates are passed through to `sorted` as-is; pre-dedupe
            at the caller if needed.
        popularity: {movie_id: (popularity_score, reception_score)} —
            the exact shape `fetch_quality_popularity_signals` returns.
            Missing movies are treated as popularity=None.

    Returns:
        A list of movie_ids ordered by the popularity sort key.
    """

    def _key(mid: int) -> tuple[int, float, int]:
        pop = popularity.get(mid, (None, None))[0]
        if pop is None:
            return (0, 0.0, mid)
        return (1, pop, mid)

    return sorted(movie_ids, key=_key, reverse=True)
