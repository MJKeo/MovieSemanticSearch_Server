"""
Trending movie score computation and refresh orchestration.

Fetches the top-N weekly trending movies from TMDB, applies concave-decay
scoring, and atomically writes the results to Redis.
"""

import logging

from db.tmdb import fetch_trending_movie_ids
from db.redis import write_trending_scores, read_trending_scores

logger = logging.getLogger(__name__)

_DEFAULT_N = 500


def compute_trending_score(rank: int, n: int) -> float:
    """
    Compute a trending score for a movie at the given 1-indexed rank.

    Formula: score = 1 - (rank / n) ** 0.5

    Concave (square root) decay gives the top of the list high scores with
    gentle differentiation, and a steeper drop-off in the tail — matching
    the intuition that ranks 1–50 are all "clearly trending" while ranks
    400–500 are barely so.

    Args:
        rank: 1-indexed position in the trending list (1 = most trending).
        n:    Total number of movies in the list (denominator).

    Returns:
        Float in [0.0, 1.0]. Rank 1 → ~1.0, rank n → 0.0.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not (1 <= rank <= n):
        raise ValueError(f"rank {rank} out of range [1, {n}]")
    return 1.0 - (rank / n) ** 0.5


async def refresh_trending_scores(n: int = _DEFAULT_N) -> int:
    """
    Fetch the top-N trending movies from TMDB and atomically update Redis.

    Steps:
        1. Fetch top-N TMDB movie IDs in rank order.
        2. Compute a concave-decay score per movie.
        3. Write scores to a Redis staging key, then atomically rename to live.

    If TMDB returns fewer than `n` movies, scores are computed against the
    actual count so the formula remains well-defined.

    Args:
        n: Number of trending movies to fetch (default 500).

    Returns:
        Number of movies written to Redis (0 on failure).
    """
    logger.info("Refreshing trending scores (n=%d)", n)

    movie_ids = await fetch_trending_movie_ids(n)
    actual_n = len(movie_ids)

    if actual_n == 0:
        logger.error("TMDB returned 0 trending movies — skipping Redis write")
        return 0

    if actual_n < n:
        logger.warning(
            "TMDB returned %d movies, fewer than requested %d — scoring against actual count",
            actual_n, n,
        )

    scores: dict[int, float] = {
        movie_id: compute_trending_score(rank=rank, n=actual_n)
        for rank, movie_id in enumerate(movie_ids, start=1)
    }

    await write_trending_scores(scores)
    logger.info("Trending scores written for %d movies", actual_n)
    return actual_n
