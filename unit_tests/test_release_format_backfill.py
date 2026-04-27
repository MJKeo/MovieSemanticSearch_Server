"""
Integration smoke tests for the ``movie_card.release_format`` backfill.

These tests read the live Postgres database (and the SQLite tracker)
to verify that the backfill produced a coherent state:

1. The ``release_format`` column exists with the right type and default.
2. The Postgres distribution matches what the tracker says we should
   have for ingested movies, bucket-by-bucket. Source of truth is the
   tracker's ``imdb_data.imdb_title_type``, run through the same
   ``release_format_id_for_imdb_type`` helper that ingestion uses.
3. The top-voted ingested movie of each non-movie title type carries
   the expected ``release_format`` int in ``movie_card``. This catches
   the case where the column is populated but for the wrong rows
   (e.g., a join went sideways).

If Postgres or the tracker is unavailable, the entire module is
skipped — these are smoke tests against running infrastructure, not
isolated unit tests.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict

import pytest

from db.postgres import _execute_read, pool
from movie_ingestion.tracker import TRACKER_DB_PATH
from schemas.enums import ReleaseFormat, release_format_id_for_imdb_type


# Tolerance for per-bucket count comparisons. The backfill and the
# tracker query are not synchronized — between when the backfill ran
# and when this test runs, more movies may have moved into 'ingested'
# status (or been filtered out). A small absolute tolerance absorbs
# that drift without making the assertion useless.
_BUCKET_COUNT_TOLERANCE = 50


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
async def open_pool():
    """Open the asyncpg pool for the duration of the module.

    The application normally opens this in the FastAPI lifespan; in a
    standalone test we manage it ourselves. If the DB isn't reachable,
    skip the whole module rather than fail — these tests assume a
    running stack and aren't meaningful otherwise.
    """
    try:
        await pool.open()
    except Exception as exc:
        pytest.skip(f"Postgres pool unavailable: {exc}")
    try:
        # Probe connection to fail fast if the host is up but the DB is
        # mid-restart or rejecting connections.
        await _execute_read("SELECT 1")
    except Exception as exc:
        await pool.close()
        pytest.skip(f"Postgres SELECT 1 failed: {exc}")
    try:
        yield
    finally:
        await pool.close()


@pytest.fixture(scope="module")
def tracker_distribution() -> dict[int, int]:
    """Count ingested-status tracker rows per ReleaseFormat int id.

    Joining on ``movie_progress.status = 'ingested'`` scopes us to the
    same population that ``movie_card`` should reflect. Buckets are
    derived through the same helper the ingest path uses, so any
    mapping drift is caught at the helper boundary.
    """
    if not TRACKER_DB_PATH.exists():
        pytest.skip(f"Tracker DB not found at {TRACKER_DB_PATH}")
    conn = sqlite3.connect(TRACKER_DB_PATH)
    try:
        cur = conn.execute(
            "SELECT i.imdb_title_type "
            "FROM imdb_data i "
            "JOIN movie_progress mp ON mp.tmdb_id = i.tmdb_id "
            "WHERE mp.status = 'ingested'"
        )
        counts: dict[int, int] = defaultdict(int)
        for (imdb_type,) in cur:
            counts[release_format_id_for_imdb_type(imdb_type)] += 1
        return dict(counts)
    finally:
        conn.close()


@pytest.fixture(scope="module")
def tracker_top_voted_per_type() -> dict[str, tuple[int, str]]:
    """Top-voted ingested movie per non-movie IMDB title type.

    Returns ``{imdb_title_type: (tmdb_id, title)}`` for ``tvMovie``,
    ``video``, and ``short``. Used by the spot-check test to assert that
    movie_card carries the right release_format for known-popular rows.
    Pulled from the tracker rather than hardcoded so it stays self-
    consistent if the corpus shifts.
    """
    if not TRACKER_DB_PATH.exists():
        pytest.skip(f"Tracker DB not found at {TRACKER_DB_PATH}")
    conn = sqlite3.connect(TRACKER_DB_PATH)
    try:
        result: dict[str, tuple[int, str]] = {}
        for imdb_type in ("tvMovie", "video", "short"):
            cur = conn.execute(
                "SELECT i.tmdb_id, t.title "
                "FROM imdb_data i "
                "JOIN movie_progress mp ON mp.tmdb_id = i.tmdb_id "
                "JOIN tmdb_data t ON t.tmdb_id = i.tmdb_id "
                "WHERE mp.status = 'ingested' "
                "  AND i.imdb_title_type = ? "
                "ORDER BY i.imdb_vote_count DESC "
                "LIMIT 1",
                (imdb_type,),
            )
            row = cur.fetchone()
            if row is None:
                pytest.skip(f"No ingested movies of type {imdb_type} in tracker")
            result[imdb_type] = (int(row[0]), str(row[1]))
        return result
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_release_format_column_exists_with_expected_shape(open_pool) -> None:
    """The column is SMALLINT NOT NULL DEFAULT 0.

    Checks the live information_schema rather than trusting the init
    SQL — a missing or mistyped ALTER on a deployed DB would slip past
    a file-only check.
    """
    rows = await _execute_read(
        "SELECT data_type, is_nullable, column_default "
        "FROM information_schema.columns "
        "WHERE table_schema = 'public' "
        "  AND table_name = 'movie_card' "
        "  AND column_name = 'release_format'"
    )
    assert len(rows) == 1, "release_format column not found on public.movie_card"
    data_type, is_nullable, column_default = rows[0]
    assert data_type == "smallint", f"expected smallint, got {data_type}"
    assert is_nullable == "NO", f"expected NOT NULL, got is_nullable={is_nullable}"
    assert column_default is not None and "0" in str(column_default), (
        f"expected DEFAULT 0, got {column_default!r}"
    )


async def test_distribution_matches_tracker(
    open_pool, tracker_distribution: dict[int, int]
) -> None:
    """Every release_format bucket in movie_card matches the tracker.

    The tracker (joined on status='ingested') is the source-of-truth
    population; movie_card should mirror it bucket-for-bucket within a
    small drift tolerance. Any bucket whose Postgres count diverges
    from the tracker count by more than the tolerance is reported with
    a per-bucket diff so the failure is diagnosable from the message
    alone.
    """
    rows = await _execute_read(
        "SELECT release_format, COUNT(*) "
        "FROM public.movie_card "
        "GROUP BY release_format"
    )
    pg_counts = {int(fmt_id): int(count) for fmt_id, count in rows}

    # Symmetric-difference comparison: any id in either side that the
    # other doesn't see is treated as count=0 on the missing side.
    all_ids = set(pg_counts) | set(tracker_distribution)
    diffs: list[str] = []
    for fmt_id in sorted(all_ids):
        pg_n = pg_counts.get(fmt_id, 0)
        trk_n = tracker_distribution.get(fmt_id, 0)
        if abs(pg_n - trk_n) > _BUCKET_COUNT_TOLERANCE:
            label = ReleaseFormat(_imdb_value_for_id(fmt_id)).name if (
                _imdb_value_for_id(fmt_id) is not None
            ) else f"id={fmt_id}"
            diffs.append(f"{label} ({fmt_id}): pg={pg_n:,} tracker={trk_n:,}")

    assert not diffs, "movie_card vs tracker bucket mismatch: " + "; ".join(diffs)


async def test_known_popular_rows_have_expected_release_format(
    open_pool, tracker_top_voted_per_type: dict[str, tuple[int, str]]
) -> None:
    """Top-voted movie of each non-movie type carries the right int.

    Iterates the (tvMovie, video, short) trio. For each, looks up the
    top-voted ingested tmdb_id in movie_card and asserts release_format
    matches the expected ReleaseFormat int. Catches the failure mode
    where the column is populated but the join wrote the wrong value
    per row.
    """
    expected_by_imdb_type = {
        "tvMovie": ReleaseFormat.TV_MOVIE.release_format_id,
        "video":   ReleaseFormat.VIDEO.release_format_id,
        "short":   ReleaseFormat.SHORT.release_format_id,
    }

    movie_ids = [tmdb_id for tmdb_id, _title in tracker_top_voted_per_type.values()]
    rows = await _execute_read(
        "SELECT movie_id, release_format "
        "FROM public.movie_card "
        "WHERE movie_id = ANY(%s::bigint[])",
        (movie_ids,),
    )
    pg_format_by_id = {int(mid): int(fmt) for mid, fmt in rows}

    failures: list[str] = []
    for imdb_type, (tmdb_id, title) in tracker_top_voted_per_type.items():
        expected_fmt = expected_by_imdb_type[imdb_type]
        actual_fmt = pg_format_by_id.get(tmdb_id)
        if actual_fmt is None:
            failures.append(
                f"{imdb_type} top-voted ({title!r}, tmdb_id={tmdb_id}) "
                "not present in movie_card"
            )
        elif actual_fmt != expected_fmt:
            failures.append(
                f"{imdb_type} top-voted ({title!r}, tmdb_id={tmdb_id}): "
                f"expected release_format={expected_fmt}, got {actual_fmt}"
            )

    assert not failures, "; ".join(failures)


async def test_unknown_bucket_is_bounded(open_pool) -> None:
    """UNKNOWN must not dominate movie_card.

    Per ADR-037, anything outside the supported title-type set is
    filtered before reaching the index, so the UNKNOWN bucket should be
    a minority. If the backfill silently failed (e.g., didn't run any
    UPDATEs), every row would still show as UNKNOWN via the column
    default — this test catches that regression.
    """
    rows = await _execute_read(
        "SELECT "
        "  SUM(CASE WHEN release_format = 0 THEN 1 ELSE 0 END), "
        "  COUNT(*) "
        "FROM public.movie_card"
    )
    unknown_n, total_n = int(rows[0][0] or 0), int(rows[0][1])
    assert total_n > 0, "movie_card is empty — nothing to verify"
    unknown_share = unknown_n / total_n
    # 10% is well above the expected near-zero share but well below the
    # 100% that a no-op backfill would produce. Any real failure mode
    # we care about lives outside this band.
    assert unknown_share < 0.10, (
        f"UNKNOWN dominates movie_card: {unknown_n:,}/{total_n:,} "
        f"({unknown_share:.1%}) — backfill likely did not run"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _imdb_value_for_id(fmt_id: int) -> str | None:
    """Return the IMDB string value for a ReleaseFormat int id, or None.

    Used only for human-readable failure messages — keeps the
    distribution-mismatch diff legible when a bucket diverges.
    """
    for member in ReleaseFormat:
        if member.release_format_id == fmt_id:
            return member.value
    return None
