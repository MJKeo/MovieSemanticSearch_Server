# Search V2 — Stage 4 display payload shaping.
#
# Phase 7 tail: take the top-K sorted ids and emit the minimal API
# payload. The planning doc names four fields for the initial
# implementation:
#
#   tmdb_id       ← movie_id (same identifier; renamed for the API)
#   movie_title   ← title
#   release_date  ← ISO "YYYY-MM-DD" string from release_ts
#   poster_url    ← poster_url
#
# Shaping is separated from fetching because the orchestrator has
# already pulled the same movie_card rows in Phase 5 (to feed the
# priors). Re-fetching would double the DB round-trips; the caller
# passes the prefetched `cards_by_id` and we rebuild the ranked list.

from __future__ import annotations

from datetime import datetime, timezone


def build_display_payload(
    ranked_movie_ids: list[int],
    cards_by_id: dict[int, dict],
) -> list[dict]:
    """Shape the top-K movie_card rows into the API payload.

    Preserves the input rank order. Rows missing from cards_by_id are
    silently dropped — per the step_4_planning.md "No backfill" rule,
    returning fewer than the requested count is acceptable.
    """
    if not ranked_movie_ids:
        return []

    payload: list[dict] = []
    for mid in ranked_movie_ids:
        row = cards_by_id.get(mid)
        if row is None:
            # No movie_card yet (extremely rare — every ingested movie
            # has one) — skip rather than emit a row with nulls.
            continue
        payload.append(
            {
                "tmdb_id": row["movie_id"],
                "movie_title": row["title"],
                "release_date": _release_ts_to_iso(row.get("release_ts")),
                "poster_url": row.get("poster_url"),
            }
        )
    return payload


def _release_ts_to_iso(release_ts: int | float | None) -> str | None:
    # release_ts is a Unix timestamp (seconds since epoch). The API
    # contract calls for a "release_date" string; ISO calendar date
    # (YYYY-MM-DD) is the stable, UI-renderable form. UTC is used so
    # the displayed date does not drift with the server's local tz.
    if release_ts is None:
        return None
    return (
        datetime.fromtimestamp(float(release_ts), tz=timezone.utc)
        .date()
        .isoformat()
    )
