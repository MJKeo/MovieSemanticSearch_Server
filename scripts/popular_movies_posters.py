"""
One-off: fetch four slates from public.movie_card and emit each movie's
poster_url (plus title / release date) to a JSON file:
  - Top 100 most-popular movies of all time
  - Top 50 most-popular movies released since 2010
  - Top 50 "hidden gems" — high reception, low reach (per the project's
    canonical definition in search_v2/similar_movies.py: imdb_vote_count
    < 10,000 AND reception_score >= 80, sorted by reception_score)
  - Top 25 "so bad it's good" — the project's `cult_garbage` shape:
    poor reception (reception_score <= SHAPE_POOR_RECEPTION_MAX = 50)
    paired with enough reach (imdb_vote_count >= SHAPE_REACH_LOW = 10K)
    to qualify as loved-for-badness rather than just unwatched-and-bad.
    Sorted by popularity DESC so the most-talked-about bad films lead.

Results are deduped by movie_id; when a movie appears in multiple slates
its `source` field lists all of them.

"Popularity" here is `movie_card.popularity_score` — the percentile-derived
sigmoid score refreshed by `refresh_movie_popularity_scores()` in
db/postgres.py. Sorting NULLS LAST so movies with no score sink.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from db.postgres import pool

load_dotenv()

# Unix seconds for 2010-01-01 UTC. release_ts is stored as Unix seconds,
# so the cutoff is computed once and bound as a parameter.
SINCE_2010_TS = int(datetime(2010, 1, 1, tzinfo=timezone.utc).timestamp())

TOP_ALL_TIME_LIMIT = 100
TOP_SINCE_2010_LIMIT = 50
HIDDEN_GEM_LIMIT = 50
SO_BAD_ITS_GOOD_LIMIT = 25

# Hidden-gem thresholds mirror SHAPE_REACH_LOW_THRESHOLD and
# SHAPE_PRESTIGE_RECEPTION_MIN in search_v2/similar_movies.py, which are
# the system's canonical "LOW reach × Acclaimed" cell.
HIDDEN_GEM_MAX_VOTE_COUNT = 10_000
HIDDEN_GEM_MIN_RECEPTION = 80.0

# "So bad it's good" mirrors the project's cult_garbage shape:
# SHAPE_POOR_RECEPTION_MAX (50) on the quality axis and
# SHAPE_REACH_LOW_THRESHOLD (10K) on the reach axis. Anything below the
# reach floor is "dogshit" (just bad) per the shape grid, so the vote
# count gate is what separates loved-for-badness from forgotten-and-bad.
SO_BAD_ITS_GOOD_MAX_RECEPTION = 50.0
SO_BAD_ITS_GOOD_MIN_VOTE_COUNT = 10_000

OUTPUT_PATH = Path("popular_movies_posters.json")


# All three queries pull the same columns so the dedupe step can merge
# rows without re-fetching. NULLS LAST keeps unscored movies out of the top.
_SELECT_COLUMNS = (
    "movie_id, title, poster_url, release_ts, popularity_score, "
    "reception_score, imdb_vote_count"
)

_TOP_ALL_TIME_QUERY = f"""
    SELECT {_SELECT_COLUMNS}
    FROM public.movie_card
    ORDER BY popularity_score DESC NULLS LAST, movie_id DESC
    LIMIT %s
"""

_TOP_SINCE_QUERY = f"""
    SELECT {_SELECT_COLUMNS}
    FROM public.movie_card
    WHERE release_ts >= %s
    ORDER BY popularity_score DESC NULLS LAST, movie_id DESC
    LIMIT %s
"""

# Hidden gems: well-received but not widely seen. Rank by reception (the
# "gem" dimension) and tie-break by popularity DESC so the most-reachable
# of the obscure-but-loved films float to the top of the slate.
_HIDDEN_GEMS_QUERY = f"""
    SELECT {_SELECT_COLUMNS}
    FROM public.movie_card
    WHERE imdb_vote_count IS NOT NULL
      AND imdb_vote_count < %s
      AND reception_score IS NOT NULL
      AND reception_score >= %s
    ORDER BY reception_score DESC,
             popularity_score DESC NULLS LAST,
             movie_id DESC
    LIMIT %s
"""

# So-bad-it's-good: poorly received films that *enough* people watched
# to develop an ironic following. Order by popularity DESC because the
# "so bad it's good" reputation is precisely what drives ongoing reach
# — surfacing the most-watched bad films, not the absolute worst.
_SO_BAD_ITS_GOOD_QUERY = f"""
    SELECT {_SELECT_COLUMNS}
    FROM public.movie_card
    WHERE reception_score IS NOT NULL
      AND reception_score <= %s
      AND imdb_vote_count IS NOT NULL
      AND imdb_vote_count >= %s
    ORDER BY popularity_score DESC NULLS LAST,
             imdb_vote_count DESC,
             movie_id DESC
    LIMIT %s
"""


async def _fetch_rows(query: str, params: tuple) -> list[tuple]:
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, params)
            return await cur.fetchall()


def _release_ts_to_date(release_ts: int | None) -> str | None:
    """Convert a Unix-seconds release_ts to an ISO YYYY-MM-DD string."""
    if release_ts is None:
        return None
    return datetime.fromtimestamp(release_ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _row_to_dict(row: tuple, source: str) -> dict:
    (
        movie_id,
        title,
        poster_url,
        release_ts,
        popularity_score,
        reception_score,
        imdb_vote_count,
    ) = row
    return {
        "movie_id": movie_id,
        "title": title,
        "poster_url": poster_url,
        "release_date": _release_ts_to_date(release_ts),
        "popularity_score": popularity_score,
        "reception_score": reception_score,
        "imdb_vote_count": imdb_vote_count,
        "sources": [source],
    }


async def collect() -> dict:
    # Pool is created inert at import time; open it for this one-shot script.
    await pool.open()
    try:
        (
            all_time_rows,
            since_2010_rows,
            hidden_gem_rows,
            so_bad_its_good_rows,
        ) = await asyncio.gather(
            _fetch_rows(_TOP_ALL_TIME_QUERY, (TOP_ALL_TIME_LIMIT,)),
            _fetch_rows(_TOP_SINCE_QUERY, (SINCE_2010_TS, TOP_SINCE_2010_LIMIT)),
            _fetch_rows(
                _HIDDEN_GEMS_QUERY,
                (HIDDEN_GEM_MAX_VOTE_COUNT, HIDDEN_GEM_MIN_RECEPTION, HIDDEN_GEM_LIMIT),
            ),
            _fetch_rows(
                _SO_BAD_ITS_GOOD_QUERY,
                (
                    SO_BAD_ITS_GOOD_MAX_RECEPTION,
                    SO_BAD_ITS_GOOD_MIN_VOTE_COUNT,
                    SO_BAD_ITS_GOOD_LIMIT,
                ),
            ),
        )
    finally:
        await pool.close()

    # Dedupe by movie_id, preserving the insertion order across slates so
    # the popular lists lead and hidden gems follow. When a movie shows up
    # in more than one slate, append the extra source rather than overwrite
    # — the JSON should faithfully report every reason it was included.
    by_id: dict[int, dict] = {}
    order: list[int] = []

    def _ingest(rows: list[tuple], source: str) -> None:
        for row in rows:
            entry = _row_to_dict(row, source)
            existing = by_id.get(entry["movie_id"])
            if existing is None:
                by_id[entry["movie_id"]] = entry
                order.append(entry["movie_id"])
            elif source not in existing["sources"]:
                existing["sources"].append(source)

    _ingest(all_time_rows, "all_time_top_100")
    _ingest(since_2010_rows, "since_2010_top_50")
    _ingest(hidden_gem_rows, "hidden_gems_top_50")
    _ingest(so_bad_its_good_rows, "so_bad_its_good_top_25")

    movies = [by_id[mid] for mid in order]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hidden_gem_criteria": {
            "imdb_vote_count_lt": HIDDEN_GEM_MAX_VOTE_COUNT,
            "reception_score_gte": HIDDEN_GEM_MIN_RECEPTION,
            "ordered_by": "reception_score DESC, popularity_score DESC",
        },
        "so_bad_its_good_criteria": {
            "reception_score_lte": SO_BAD_ITS_GOOD_MAX_RECEPTION,
            "imdb_vote_count_gte": SO_BAD_ITS_GOOD_MIN_VOTE_COUNT,
            "ordered_by": "popularity_score DESC, imdb_vote_count DESC",
        },
        "counts": {
            "all_time_top_100": len(all_time_rows),
            "since_2010_top_50": len(since_2010_rows),
            "hidden_gems_top_50": len(hidden_gem_rows),
            "so_bad_its_good_top_25": len(so_bad_its_good_rows),
            "deduped_total": len(movies),
        },
        "movies": movies,
    }


async def main() -> None:
    payload = await collect()
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(
        f"Wrote {payload['counts']['deduped_total']} movies "
        f"(all-time={payload['counts']['all_time_top_100']}, "
        f"since-2010={payload['counts']['since_2010_top_50']}, "
        f"hidden-gems={payload['counts']['hidden_gems_top_50']}, "
        f"so-bad-its-good={payload['counts']['so_bad_its_good_top_25']}) "
        f"to {OUTPUT_PATH.resolve()}"
    )


if __name__ == "__main__":
    asyncio.run(main())
