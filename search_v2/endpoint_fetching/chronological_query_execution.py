# Chronological endpoint: query execution.
#
# POOL_RERANKER. Receives the candidate pool (`restrict_to_movie_ids`),
# bulk-fetches each movie's `release_ts` from `movie_card`, and emits
# a per-movie percentile-rank score in [0, 1] according to the spec's
# `direction`. The percentile math lives in
# `db/chronological_scoring.py`; this file is the thin Postgres
# wrapper plus the standard error-handling shell every executor uses.
#
# Always preference-mode: chronological never carves a pool. The
# POOL_RERANKER + no-pool case is short-circuited upstream inside
# `build_endpoint_coroutine` before this function is invoked, so a
# falsy `restrict_to_movie_ids` here is treated as an empty result.
#
# Unlike most executors, the spec instance IS the parameters object
# (`ChronologicalQuerySpec` inherits directly from `EndpointParameters`
# with no nested `parameters` field — same shape as the ENTITY family).

from __future__ import annotations

import logging

from db.chronological_scoring import score_chronological
from db.postgres import pool
from schemas.chronological_translation import ChronologicalQuerySpec
from schemas.endpoint_result import EndpointResult
from search_v2.endpoint_fetching.result_helpers import build_endpoint_result


log = logging.getLogger(__name__)


async def _fetch_release_ts(movie_ids: list[int]) -> dict[int, int | None]:
    """Bulk-fetch `release_ts` for every supplied id in one round-trip.

    Returns a movie_id -> release_ts map. Ids missing from the
    movie_card table (e.g., upstream stale pool members) simply do
    not appear in the returned map — the caller fills those as
    None so the scorer floors them at 0.
    """
    sql = (
        "SELECT movie_id, release_ts FROM public.movie_card "
        "WHERE movie_id = ANY(%s)"
    )
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, [movie_ids])
            rows = await cur.fetchall()
    return {row[0]: row[1] for row in rows}


async def execute_chronological_query(
    spec: ChronologicalQuerySpec,
    restrict_to_movie_ids: set[int] | None = None,
) -> EndpointResult:
    """Score the candidate pool by recency percentile rank.

    Always preference-mode: chronological is a POOL_RERANKER and
    the caller supplies a non-empty `restrict_to_movie_ids`. Empty
    or None pools short-circuit to an empty result.

    Missing `release_ts` floors at 0; ties (movies sharing a release
    day) tie in score. See `db.chronological_scoring` for the curve
    formula.
    """
    try:
        fetched = await _fetch_release_ts(list(restrict_to_movie_ids))
    except Exception:
        log.exception(
            "chronological execution failed fetching release_ts "
            "(pool=%d, direction=%s)",
            len(restrict_to_movie_ids), spec.direction.value,
        )
        return build_endpoint_result({}, restrict_to_movie_ids)

    # Every requested movie_id must appear in the input to the
    # scorer so missing rows (no movie_card record) get floored at
    # 0 alongside the explicit-None case.
    enriched: dict[int, int | None] = {
        mid: fetched.get(mid) for mid in restrict_to_movie_ids
    }

    scored = score_chronological(enriched, spec.direction)
    return build_endpoint_result(scored, restrict_to_movie_ids)
