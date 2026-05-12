# Chronological scoring: percentile-rank curve over release_ts.
#
# Pure helper consumed by the chronological query executor. Given a
# mapping of movie_id -> release_ts (BIGINT epoch seconds, possibly
# None) and a `ChronologicalDirection`, returns a movie_id ->
# percentile-rank score map in [0, 1].
#
# Design (per the user's spec):
#   - Continuous, never saturating. The most-extreme movie in the
#     direction always wins; every distinct release date occupies
#     its own slot. A one-day difference always matters.
#   - Ranks computed over the set of UNIQUE release dates present in
#     the input pool, not over a global corpus. The endpoint is a
#     POOL_RERANKER — the curve adapts to whatever pool the sibling
#     categories produced.
#   - Same release_ts -> same score (movies sharing a release day
#     are tied).
#   - Missing release_ts (None) -> 0.0. Those movies can't
#     participate in the curve; the user accepted this floor rather
#     than dropping them from the result.
#   - Pool with fewer than 2 unique dates: no curve resolution
#     exists, so every valid-date movie gets 1.0. Missing-date
#     movies still get 0.0.

from __future__ import annotations

from schemas.chronological_translation import ChronologicalDirection


def score_chronological(
    release_ts_by_movie: dict[int, int | None],
    direction: ChronologicalDirection,
) -> dict[int, float]:
    """Score the candidate pool by recency percentile rank.

    Score = position_of_release_date_among_unique_dates / (D - 1),
    where D is the number of distinct release dates in the pool.
    The winning end of the curve is determined by `direction`:
    OLDEST_FIRST -> oldest date scores 1.0; NEWEST_FIRST -> newest
    date scores 1.0.

    Args:
        release_ts_by_movie: One entry per candidate movie. Value
            may be None for movies missing a release date.
        direction: OLDEST_FIRST (oldest wins) or NEWEST_FIRST
            (newest wins).

    Returns:
        Per-movie score in [0, 1]. Every input movie_id is present
        in the output. Missing-release_ts movies always score 0.0.
    """
    # Partition: only movies with a real release_ts participate in
    # the curve. The rest land at 0.0 directly.
    valid: dict[int, int] = {
        mid: ts for mid, ts in release_ts_by_movie.items() if ts is not None
    }
    if not valid:
        return {mid: 0.0 for mid in release_ts_by_movie}

    # Rank over UNIQUE dates so ties (movies sharing a release day)
    # collapse to one score. ascending: oldest = 0, newest = D - 1.
    unique_sorted_asc: list[int] = sorted(set(valid.values()))
    n_unique = len(unique_sorted_asc)

    if n_unique == 1:
        # Single unique date in the pool — the curve has no
        # resolution, so every valid-date movie gets credit. Missing-
        # date movies still floor at 0.
        return {
            mid: (1.0 if ts is not None else 0.0)
            for mid, ts in release_ts_by_movie.items()
        }

    rank_by_ts: dict[int, int] = {ts: i for i, ts in enumerate(unique_sorted_asc)}
    denom = n_unique - 1

    result: dict[int, float] = {}
    for mid, ts in release_ts_by_movie.items():
        if ts is None:
            result[mid] = 0.0
            continue
        asc_rank = rank_by_ts[ts]
        if direction is ChronologicalDirection.NEWEST_FIRST:
            result[mid] = asc_rank / denom
        else:
            result[mid] = (denom - asc_rank) / denom
    return result
