"""
Stage 5: IMDB Essential Data Hard-Filter + Combined Quality Scorer

Two entry points:

  run()       — Hard filters on imdb_scraped movies.  Survivors advance to
                'essential_data_passed'.

  score_all() — Computes the 8-signal combined TMDB+IMDB quality score for
                every 'essential_data_passed' movie and persists it to
                movie_progress.stage_5_quality_score.  Does NOT change status
                — threshold filtering is a separate step.

Hard filters (applied in priority order — first failing reason is logged):

  0. missing_imdb_json          — no JSON file on disk for this tmdb_id
  1. no_imdb_rating             — imdb_rating is None (no audience engagement)
  2. no_directors               — directors list empty
  3. no_poster_url              — poster_url missing from TMDB data
  4. no_actors                  — actors list empty
  5. no_characters              — characters list empty
  6. no_overall_keywords        — overall_keywords list empty
  7. no_languages               — languages list empty
  8. no_countries_of_origin     — countries_of_origin list empty
  9. no_release_date            — release_date missing from TMDB data

Quality scoring model — 8 signals (weights sum to 1.0):

  imdb_vote_count (0.22), watch_providers (0.20), featured_reviews_chars (0.16),
  plot_text_depth (0.12), lexical_completeness (0.10), data_completeness (0.10),
  tmdb_popularity (0.06), metacritic_rating (0.04).

  See docs/modules/ingestion.md and ADR-016 for full signal details and rationale.

Both operations are idempotent: re-running overwrites existing scores and
skips already-processed movies.

Usage:
    python -m movie_ingestion.imdb_quality_scoring.imdb_quality_scorer
"""

import datetime
import json
import math
import sqlite3
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import orjson

from movie_ingestion.scoring_utils import (
    THEATER_WINDOW_DAYS,
    VoteCountSource,
    score_popularity,
    score_vote_count,
    unpack_provider_keys,
    validate_weights,
)
from movie_ingestion.tracker import (
    INGESTION_DATA_DIR,
    MovieStatus,
    PipelineStage,
    init_db,
    log_filter,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_IMDB_DIR = INGESTION_DATA_DIR / "imdb"

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Flush to disk every N rows processed.  Bounds data loss on crash.
COMMIT_EVERY: int = 1_000

# Emit a progress line every N rows processed.
LOG_EVERY: int = 10_000

# Log cap for plot_text_depth: overview + plot_summaries + synopses total chars.
# 5001 chars places the ceiling so movies with rich synopses (~p75) saturate
# at 1.0, while overview-only movies (~150 chars) score ~0.59.
PLOT_TEXT_LOG_CAP: int = 5001

# ---------------------------------------------------------------------------
# Signal weights — must sum to 1.0.
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, float] = {
    "imdb_vote_count":        0.22,
    "watch_providers":        0.20,
    "featured_reviews_chars": 0.16,
    "plot_text_depth":        0.12,
    "lexical_completeness":   0.10,
    "data_completeness":      0.10,
    "tmdb_popularity":        0.06,
    "metacritic_rating":      0.04,
}

# Guard at module load time — see scoring_utils.validate_weights docstring.
validate_weights(WEIGHTS, label="Stage 5 WEIGHTS")

# ---------------------------------------------------------------------------
# MovieContext — unified view of both data sources for a single movie
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MovieContext:
    """Bundles TMDB and IMDB data for a single movie, providing a unified
    interface for filter predicates and scoring functions.

    tmdb: dict of column values from the tmdb_data table.
    imdb: parsed IMDB JSON dict, or None if the JSON file is missing.
    """
    tmdb_id: int
    tmdb: dict
    imdb: dict | None


# ---------------------------------------------------------------------------
# Hard filter predicates
# ---------------------------------------------------------------------------
# Each predicate receives a MovieContext and returns True when the movie FAILS
# the filter and should be eliminated.  The ordered list below defines priority:
# when a movie fails multiple filters, the first matching entry supplies the
# logged reason.

_FilterPredicate = Callable[[MovieContext], bool]


def _fails_imdb_json(ctx: MovieContext) -> bool:
    """IMDB JSON file missing from disk — cannot evaluate any IMDB fields."""
    return ctx.imdb is None


def _fails_imdb_rating(ctx: MovieContext) -> bool:
    """No IMDB rating — movie has zero recorded audience engagement and
    cannot participate in quality reranking (reception_score)."""
    if ctx.imdb is None:
        return True
    return ctx.imdb.get("imdb_rating") is None


def _fails_directors(ctx: MovieContext) -> bool:
    """No director credits — fundamental lexical search entity missing,
    signals catastrophically sparse IMDB record."""
    if ctx.imdb is None:
        return True
    return not ctx.imdb.get("directors")


def _fails_poster_url(ctx: MovieContext) -> bool:
    """No poster URL — hard pipeline requirement for display in search
    results (ingest_movie_card validates non-null)."""
    return not ctx.tmdb.get("poster_url")


def _fails_actors(ctx: MovieContext) -> bool:
    """No actor credits — critical lexical entity for 'movies with X actor'
    queries and anchor vector cast_text()."""
    if ctx.imdb is None:
        return True
    return not ctx.imdb.get("actors")


def _fails_characters(ctx: MovieContext) -> bool:
    """No character names — cannot support character-based lexical search
    or contribute to anchor vector characters_text()."""
    if ctx.imdb is None:
        return True
    return not ctx.imdb.get("characters")


def _fails_overall_keywords(ctx: MovieContext) -> bool:
    """No community keywords — severely limits both lexical matching and
    semantic signal for LLM metadata generation."""
    if ctx.imdb is None:
        return True
    return not ctx.imdb.get("overall_keywords")


def _fails_languages(ctx: MovieContext) -> bool:
    """No language metadata — cannot support language-based filtering
    in Qdrant payload or metadata scoring."""
    if ctx.imdb is None:
        return True
    return not ctx.imdb.get("languages")


def _fails_countries_of_origin(ctx: MovieContext) -> bool:
    """No country of origin — cannot support region-based queries or
    production vector text."""
    if ctx.imdb is None:
        return True
    return not ctx.imdb.get("countries_of_origin")


def _fails_release_date(ctx: MovieContext) -> bool:
    """No release date in TMDB data — cannot support date-range filtering
    in Qdrant payload or metadata scoring."""
    rd = ctx.tmdb.get("release_date")
    return rd is None or rd == ""


# Priority-ordered list of (reason_string, predicate) pairs.
# The reason string matches the values written to filter_log.reason.
_HARD_FILTERS: list[tuple[str, _FilterPredicate]] = [
    ("missing_imdb_json",      _fails_imdb_json),
    ("no_imdb_rating",         _fails_imdb_rating),
    ("no_directors",           _fails_directors),
    ("no_poster_url",          _fails_poster_url),
    ("no_actors",              _fails_actors),
    ("no_characters",          _fails_characters),
    ("no_overall_keywords",    _fails_overall_keywords),
    ("no_languages",           _fails_languages),
    ("no_countries_of_origin", _fails_countries_of_origin),
    ("no_release_date",        _fails_release_date),
]


# ---------------------------------------------------------------------------
# Filter evaluation
# ---------------------------------------------------------------------------


def _evaluate_filters(ctx: MovieContext) -> tuple[str | None, list[str]]:
    """Check all hard filters for a single movie.

    Evaluates every predicate in _HARD_FILTERS priority order.

    Returns:
        A tuple (primary_reason, all_failing_reasons) where:
        - primary_reason is the first (highest-priority) reason that fired,
          or None if the movie passes all filters.
        - all_failing_reasons is a list of every reason that fired.
    """
    failing: list[str] = [
        reason
        for reason, predicate in _HARD_FILTERS
        if predicate(ctx)
    ]
    primary = failing[0] if failing else None
    return primary, failing


# ===========================================================================
# Quality scoring — 8-signal combined TMDB+IMDB scorer
# ===========================================================================
# Each signal function receives a MovieContext (with guaranteed non-None imdb)
# and returns a normalised score.  The signal functions are private to this
# module; the public entry point is compute_imdb_quality_score().


def _score_imdb_vote_count(ctx: MovieContext, today: datetime.date) -> float:
    """IMDB vote count score in [0, 1], log-scaled with age adjustments.

    Primary notability proxy.  Uses IMDB votes (more representative for a
    US-focused app) with the IMDB log cap (12001).  Release date sourced
    from TMDB for the recency/classic multiplier.
    """
    vc = ctx.imdb.get("imdb_vote_count", 0) if ctx.imdb else 0
    release_date = ctx.tmdb.get("release_date")
    return score_vote_count(vc, release_date, today, VoteCountSource.IMDB)


def _score_watch_providers(ctx: MovieContext, today: datetime.date) -> float:
    """Binary watch-provider score in [-1, +1] with theater-window logic.

    For a US-focused app, a movie with no streaming availability past its
    theatrical window is nearly worthless as a recommendation.  Within the
    75-day theater window, absence is expected and receives +1.

    Null release_date is treated conservatively as past the theater window.
    """
    provider_count = len(unpack_provider_keys(ctx.tmdb.get("watch_provider_keys")))
    release_date = ctx.tmdb.get("release_date")

    # Determine whether the theater window has elapsed.
    past_theater: bool = True
    if release_date is not None:
        try:
            release = datetime.date.fromisoformat(release_date)
            past_theater = (today - release).days > THEATER_WINDOW_DAYS
        except ValueError:
            pass  # Non-parseable date — conservatively assume post-theater.

    if provider_count >= 1 or not past_theater:
        return 1.0
    else:
        return -1.0


def _score_featured_reviews_chars(ctx: MovieContext) -> float:
    """Tiered featured-review score in [-1, +1] based on total char count.

    IMDB featured_reviews primary (sum of .text lengths); TMDB reviews JSON
    as fallback.  Reviews feed 6 of 7 vector spaces, making absence a genuine
    red flag for LLM generation quality.

    Tiers:  0 chars → -1.0,  1-3000 → 0.0,  3001-8000 → 0.5,  8001+ → 1.0
    """
    total_chars = 0

    # IMDB primary: sum character lengths across all featured review texts.
    if ctx.imdb:
        reviews = ctx.imdb.get("featured_reviews") or []
        for review in reviews:
            text = review.get("text", "")
            total_chars += len(text)

    # TMDB fallback: only used when IMDB contributes zero review chars.
    # The reviews column stores a JSON-encoded list of review content strings.
    if total_chars == 0:
        tmdb_reviews_json = ctx.tmdb.get("reviews")
        if tmdb_reviews_json:
            try:
                tmdb_reviews = json.loads(tmdb_reviews_json)
                for text in tmdb_reviews:
                    total_chars += len(text)
            except (json.JSONDecodeError, TypeError):
                pass  # Malformed JSON — treat as no reviews.

    # Apply tiered scoring.
    if total_chars == 0:
        return -1.0
    elif total_chars <= 3_000:
        return 0.0
    elif total_chars <= 8_000:
        return 0.5
    else:
        return 1.0


def _score_plot_text_depth(ctx: MovieContext) -> float:
    """Log-scaled plot text depth score in [0, 1].

    Composite: total character count of overview + plot_summaries + synopses.
    IMDB sources primary for each component; TMDB overview_length as fallback
    for the overview component only.

    These fields are substitutes, not complements — the total text budget
    available to the LLM determines quality, regardless of which field
    contributes it.

    Log cap: 5001 chars.
    """
    total = 0

    if ctx.imdb:
        # Overview from IMDB (string).
        overview = ctx.imdb.get("overview") or ""
        total += len(overview)

        # Plot summaries: list of strings.
        for text in ctx.imdb.get("plot_summaries") or []:
            total += len(text)

        # Synopses: list of strings.
        for text in ctx.imdb.get("synopses") or []:
            total += len(text)

    # TMDB fallback for overview only — used when IMDB overview is absent.
    if ctx.imdb is None or not (ctx.imdb.get("overview") or ""):
        tmdb_overview_len = ctx.tmdb.get("overview_length") or 0
        total += tmdb_overview_len

    if total == 0:
        return 0.0

    return min(math.log10(total + 1) / math.log10(PLOT_TEXT_LOG_CAP), 1.0)


def _score_lexical_completeness(ctx: MovieContext) -> float:
    """Lexical completeness score in [-1, +1].

    Measures how well lexical search can work with this movie.  Six entity
    types, each contributing a capped sub-score of 0.0 to 1.0, then linearly
    mapped from [0, 6] to [-1, +1].

    Cap each entity type's contribution so a large cast (80 actors) cannot
    mask missing writers or producers.  Discrimination is at the extreme low
    ends — having 10 actors vs 80 is a budget difference, but having 1 actor
    vs 10 is a data quality concern.

    Entity scoring:
      actors:               <5 → 0.5,  5+ → 1.0
      characters:           <5 → 0.5,  5+ → 1.0
      writers:              0 → 0.0,   1+ → 1.0
      composers:            0 → 0.0,   1+ → 1.0
      producers:            0 → 0.0,   1+ → 1.0
      production_companies: 0 → 0.0,   1+ → 1.0  (IMDB primary, TMDB fallback)
    """
    imdb = ctx.imdb or {}

    # Actors: threshold at 5 for full score.
    actor_count = len(imdb.get("actors") or [])
    actors_sub = 1.0 if actor_count >= 5 else (0.5 if actor_count > 0 else 0.0)

    # Characters: same threshold as actors.
    char_count = len(imdb.get("characters") or [])
    chars_sub = 1.0 if char_count >= 5 else (0.5 if char_count > 0 else 0.0)

    # Binary presence for remaining entity types.
    writers_sub = 1.0 if imdb.get("writers") else 0.0
    composers_sub = 1.0 if imdb.get("composers") else 0.0
    producers_sub = 1.0 if imdb.get("producers") else 0.0

    # Production companies: IMDB primary, TMDB has_production_companies fallback.
    if imdb.get("production_companies"):
        prodco_sub = 1.0
    elif ctx.tmdb.get("has_production_companies"):
        prodco_sub = 1.0
    else:
        prodco_sub = 0.0

    total = actors_sub + chars_sub + writers_sub + composers_sub + producers_sub + prodco_sub
    # Map [0, 6] → [-1, +1]:  (total - 3) / 3
    return (total - 3.0) / 3.0


def _score_data_completeness(ctx: MovieContext) -> float:
    """Data completeness score in [-1, +1].

    Measures how likely the movie is to have a full dataset that feeds well
    into vector search.  Six fields, some with tiered sub-scoring, then
    linearly mapped from [0, 6] to [-1, +1].

    Distinct from lexical_completeness: covers supplementary fields that
    enrich LLM-generated vector metadata (plot keywords, filming locations,
    content advisories) rather than named entities for lexical matching.

    Field scoring:
      plot_keywords:        0 → 0.0,  1-4 → 0.5,  5+ → 1.0
      overall_keywords:     1 → 0.25, 2-3 → 0.5,  4+ → 1.0
      filming_locations:    0 → 0.0,  1+ → 1.0
      parental_guide_items: 0 → 0.0,  1+ → 1.0
      maturity_rating:      absent → 0.0, present → 1.0  (IMDB → TMDB fallback)
      budget:               absent → 0.0, present → 1.0  (IMDB → TMDB fallback)
    """
    imdb = ctx.imdb or {}

    # Plot keywords: tiered by count.
    pk_count = len(imdb.get("plot_keywords") or [])
    if pk_count >= 5:
        pk_sub = 1.0
    elif pk_count >= 1:
        pk_sub = 0.5
    else:
        pk_sub = 0.0

    # Overall keywords: tiered.  Post-hard-filter movies always have ≥1,
    # so in practice the minimum is 0.25.  Defensive 0.0 for missing data.
    ok_count = len(imdb.get("overall_keywords") or [])
    if ok_count >= 4:
        ok_sub = 1.0
    elif ok_count >= 2:
        ok_sub = 0.5
    elif ok_count >= 1:
        ok_sub = 0.25
    else:
        ok_sub = 0.0

    # Filming locations: binary presence.
    fl_sub = 1.0 if imdb.get("filming_locations") else 0.0

    # Parental guide items: binary presence.
    pg_sub = 1.0 if imdb.get("parental_guide_items") else 0.0

    # Maturity rating: IMDB primary, TMDB fallback.
    if imdb.get("maturity_rating"):
        mr_sub = 1.0
    elif ctx.tmdb.get("maturity_rating"):
        mr_sub = 1.0
    else:
        mr_sub = 0.0

    # Budget: IMDB primary, TMDB fallback.
    if imdb.get("budget"):
        bud_sub = 1.0
    elif ctx.tmdb.get("budget"):
        bud_sub = 1.0
    else:
        bud_sub = 0.0

    total = pk_sub + ok_sub + fl_sub + pg_sub + mr_sub + bud_sub
    # Map [0, 6] → [-1, +1]:  (total - 3) / 3
    return (total - 3.0) / 3.0


def _score_tmdb_popularity(ctx: MovieContext) -> float:
    """TMDB popularity score in [0, 1], log-scaled.

    TMDB's algorithmic activity score captures short-term buzz and current
    momentum that vote count and watch providers don't.  Delegates to the
    shared score_popularity() in scoring_utils.
    """
    popularity = ctx.tmdb.get("popularity") or 0.0
    return score_popularity(popularity)


def _score_metacritic_rating(ctx: MovieContext) -> float:
    """Metacritic rating score in [0, 1].  Binary bonus.

    Presence of a Metacritic score indicates the movie received professional
    critical coverage — a strong correlate of cultural significance.  At 15.2%
    presence, this is rare and purely a bonus.  No penalty for absence.
    """
    if ctx.imdb and ctx.imdb.get("metacritic_rating") is not None:
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Public scoring function — reusable for single-movie scoring (cron jobs)
# ---------------------------------------------------------------------------


def compute_imdb_quality_score(
    ctx: MovieContext,
    today: datetime.date,
) -> float:
    """Compute the combined TMDB+IMDB quality score for a single movie.

    Pure function: takes a MovieContext and today's date, returns the weighted
    sum of 8 normalised signals.  No database access — can be called from
    any context (batch scoring, cron jobs, unit tests) by constructing a
    MovieContext from any data source.

    Each signal is individually normalised to [-1, +1] or [0, +1] so the
    attribute weights remain meaningful.  The weighted sum produces a raw
    score (not normalised across the corpus).  A separate pass uses
    survival-curve derivative analysis to determine the threshold.

    Args:
        ctx:   MovieContext with both tmdb (dict) and imdb (dict) populated.
        today: Scoring reference date (for vote_count age adjustments and
               theater window logic).

    Returns:
        Raw quality score as a float.  Approximate range [-0.55, +0.95].
    """
    return (
        WEIGHTS["imdb_vote_count"]        * _score_imdb_vote_count(ctx, today)
        + WEIGHTS["watch_providers"]      * _score_watch_providers(ctx, today)
        + WEIGHTS["featured_reviews_chars"] * _score_featured_reviews_chars(ctx)
        + WEIGHTS["plot_text_depth"]      * _score_plot_text_depth(ctx)
        + WEIGHTS["lexical_completeness"] * _score_lexical_completeness(ctx)
        + WEIGHTS["data_completeness"]    * _score_data_completeness(ctx)
        + WEIGHTS["tmdb_popularity"]      * _score_tmdb_popularity(ctx)
        + WEIGHTS["metacritic_rating"]    * _score_metacritic_rating(ctx)
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_tmdb_data_for_filters(db: sqlite3.Connection) -> dict[int, dict]:
    """Load TMDB rows for all imdb_scraped movies (hard filter phase).

    Only selects the columns needed by the hard filter predicates (poster_url,
    release_date) plus tmdb_id for joining.  Returns dict keyed by tmdb_id.
    """
    rows = db.execute("""
        SELECT d.tmdb_id, d.poster_url, d.release_date
        FROM tmdb_data d
        JOIN movie_progress p ON d.tmdb_id = p.tmdb_id
        WHERE p.status = ?
    """, (MovieStatus.IMDB_SCRAPED,)).fetchall()

    result: dict[int, dict] = {}
    for row in rows:
        result[row[0]] = {
            "tmdb_id": row[0],
            "poster_url": row[1],
            "release_date": row[2],
        }
    return result


def _load_tmdb_data_for_scoring(db: sqlite3.Connection) -> dict[int, dict]:
    """Load TMDB rows for all essential_data_passed movies (scoring phase).

    Selects all columns needed by the 8-signal scorer: watch_provider_keys,
    popularity, release_date, overview_length, maturity_rating, budget, and
    reviews (TMDB fallback fields).  Returns dict keyed by tmdb_id.
    """
    rows = db.execute("""
        SELECT
            d.tmdb_id,
            d.release_date,
            d.watch_provider_keys,
            d.popularity,
            d.overview_length,
            d.maturity_rating,
            d.budget,
            d.reviews,
            d.has_production_companies
        FROM tmdb_data d
        JOIN movie_progress p ON d.tmdb_id = p.tmdb_id
        WHERE p.status = ?
    """, (MovieStatus.ESSENTIAL_DATA_PASSED,)).fetchall()

    result: dict[int, dict] = {}
    for row in rows:
        result[row[0]] = {
            "tmdb_id": row[0],
            "release_date": row[1],
            "watch_provider_keys": row[2],
            "popularity": row[3],
            "overview_length": row[4],
            "maturity_rating": row[5],
            "budget": row[6],
            "reviews": row[7],
            "has_production_companies": row[8],
        }
    return result


def _load_one_json(path: Path) -> tuple[int, dict] | None:
    """Load a single IMDB JSON file.  Returns (tmdb_id, data) or None.

    Returns None for missing files (FileNotFoundError) so the caller can
    skip movies whose JSON hasn't been written yet.
    """
    try:
        tmdb_id = int(path.stem)
    except ValueError:
        return None
    try:
        with open(path, "rb") as f:
            return tmdb_id, orjson.loads(f.read())
    except FileNotFoundError:
        return None


def _load_imdb_data(tmdb_ids: set[int]) -> dict[int, dict]:
    """Load IMDB JSON files for the given tmdb_ids.

    Uses orjson for fast deserialization and ThreadPoolExecutor to overlap
    file I/O across threads (I/O-bound, so threads are effective despite GIL).
    Only loads files whose stem (tmdb_id) is in the target set.
    """
    # Build the list of paths to load — construct directly from target IDs.
    # No existence check here; _load_one_json handles missing files via
    # the FileNotFoundError path in the ThreadPoolExecutor.
    json_files = [_IMDB_DIR / f"{tid}.json" for tid in tmdb_ids]

    result: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        for pair in pool.map(_load_one_json, json_files):
            if pair is not None:
                result[pair[0]] = pair[1]
    return result


# ---------------------------------------------------------------------------
# Entry point: hard filters
# ---------------------------------------------------------------------------


def run() -> None:
    """Apply hard filters to all imdb_scraped movies.

    For every movie with status='imdb_scraped':
      1. Build a MovieContext from TMDB (tracker DB) + IMDB (JSON file) data.
      2. Evaluate the hard filters in priority order.
      3. Call log_filter() for any movie that fails at least one check.

    Survivors are advanced to 'essential_data_passed' via a single bulk UPDATE.

    Progress is reported every LOG_EVERY rows.  A summary table is printed
    at the end showing the per-reason breakdown and the surviving count.
    """
    db = init_db()

    # Load both data sources upfront so the filter loop does no I/O.
    print("Loading TMDB data from tracker database...")
    tmdb_data = _load_tmdb_data_for_filters(db)
    tmdb_ids = set(tmdb_data.keys())

    print(f"Loading IMDB JSON files for {len(tmdb_ids):,} movies...")
    imdb_data = _load_imdb_data(tmdb_ids)

    total = len(tmdb_data)
    print(f"  {total:,} movies to evaluate")

    if total == 0:
        print("No imdb_scraped movies found. Has Stage 4 (IMDB scraping) completed?")
        db.close()
        return

    # Per-reason counters for the final summary table.
    reason_counts: dict[str, int] = {reason: 0 for reason, _ in _HARD_FILTERS}
    filtered_total: int = 0
    pending_commit: int = 0

    # Process movies in sorted order for deterministic output.
    sorted_ids = sorted(tmdb_ids)

    try:
        for i, tid in enumerate(sorted_ids):
            tmdb_row = tmdb_data[tid]
            imdb_row = imdb_data.get(tid)  # None if JSON file was missing

            ctx = MovieContext(tmdb_id=tid, tmdb=tmdb_row, imdb=imdb_row)

            # --- Evaluate hard filters ---
            primary_reason, all_failing = _evaluate_filters(ctx)

            if primary_reason is not None:
                # When multiple filters fire, record secondary reasons in the
                # details JSON so no diagnostic information is lost.
                details: str | None = None
                if len(all_failing) > 1:
                    details = json.dumps({"also_failed": all_failing[1:]})

                log_filter(
                    db,
                    tmdb_id=tid,
                    stage=PipelineStage.ESSENTIAL_DATA_CHECK,
                    reason=primary_reason,
                    details=details,
                )

                reason_counts[primary_reason] += 1
                filtered_total += 1

            pending_commit += 1

            # Commit periodically so progress survives a crash.
            if pending_commit >= COMMIT_EVERY:
                db.commit()
                pending_commit = 0

            if (i + 1) % LOG_EVERY == 0:
                print(
                    f"  Processed {i + 1:,}/{total:,}"
                    f" | filtered so far: {filtered_total:,}"
                )

        # Flush any remaining uncommitted writes.
        db.commit()

        # Advance all surviving movies (still at 'imdb_scraped') to the next
        # pipeline status in a single bulk UPDATE.
        db.execute(
            """UPDATE movie_progress
               SET status = ?, updated_at = CURRENT_TIMESTAMP
               WHERE status = ?""",
            (MovieStatus.ESSENTIAL_DATA_PASSED, MovieStatus.IMDB_SCRAPED),
        )
        db.commit()
    finally:
        db.close()

    # ---------------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------------
    surviving = total - filtered_total

    print(f"\nStage 5 essential data hard-filter complete")
    print(f"  Total evaluated:  {total:,}")
    print(f"  Filtered out:     {filtered_total:,}")
    print(f"  Surviving:        {surviving:,}")
    print(f"\nBreakdown by primary reason:")
    for reason, count in reason_counts.items():
        pct = (count / total * 100) if total > 0 else 0.0
        print(f"  {reason:<26}  {count:>8,}  ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Entry point: quality scoring
# ---------------------------------------------------------------------------


def score_all() -> None:
    """Compute and persist quality scores for all essential_data_passed movies.

    For every movie with status='essential_data_passed':
      1. Load TMDB data from tracker DB and IMDB data from JSON files.
      2. Build a MovieContext for each movie.
      3. Compute the 8-signal quality score via compute_imdb_quality_score().
      4. Persist the score to movie_progress.stage_5_quality_score.

    Does NOT change movie status — threshold filtering is a separate step
    that will be added after survival-curve analysis determines the cutoff.

    Idempotent: re-running overwrites existing stage_5_quality_score values.
    """
    today = datetime.date.today()
    db = init_db()

    print("Stage 5 quality scorer: loading essential_data_passed movies...")
    tmdb_data = _load_tmdb_data_for_scoring(db)
    tmdb_ids = set(tmdb_data.keys())

    print(f"Loading IMDB JSON files for {len(tmdb_ids):,} movies...")
    imdb_data = _load_imdb_data(tmdb_ids)

    total = len(tmdb_data)
    print(f"  {total:,} movies to score (reference date = {today})")

    if total == 0:
        print("No essential_data_passed movies found. Has the hard-filter step completed?")
        db.close()
        return

    scored: int = 0
    score_sum: float = 0.0
    score_min: float = float("inf")
    score_max: float = float("-inf")

    # Batch of (score, tmdb_id) tuples for bulk UPDATE via executemany.
    batch: list[tuple[float, int]] = []

    # Process movies in sorted order for deterministic output.
    sorted_ids = sorted(tmdb_ids)

    try:
        for i, tid in enumerate(sorted_ids):
            tmdb_row = tmdb_data[tid]
            imdb_row = imdb_data.get(tid)

            ctx = MovieContext(tmdb_id=tid, tmdb=tmdb_row, imdb=imdb_row)
            score = compute_imdb_quality_score(ctx, today)

            batch.append((score, tid))
            scored += 1
            score_sum += score
            score_min = min(score_min, score)
            score_max = max(score_max, score)

            # Flush batch to disk periodically to bound data loss on crash.
            if len(batch) >= COMMIT_EVERY:
                db.executemany(
                    """UPDATE movie_progress
                       SET stage_5_quality_score = ?, updated_at = CURRENT_TIMESTAMP
                       WHERE tmdb_id = ?""",
                    batch,
                )
                db.commit()
                batch.clear()

            if (i + 1) % LOG_EVERY == 0:
                print(f"  Scored {i + 1:,}/{total:,}")

        # Flush any remaining rows that didn't reach the batch threshold.
        if batch:
            db.executemany(
                """UPDATE movie_progress
                   SET stage_5_quality_score = ?, updated_at = CURRENT_TIMESTAMP
                   WHERE tmdb_id = ?""",
                batch,
            )
            db.commit()
    finally:
        db.close()

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    score_mean = score_sum / scored if scored > 0 else 0.0

    print(f"\nStage 5 quality scorer complete")
    print(f"  Movies scored:  {scored:,}")
    print(f"  Score min:      {score_min:.4f}")
    print(f"  Score mean:     {score_mean:.4f}")
    print(f"  Score max:      {score_max:.4f}")
    print(
        f"\nNext step: run survival-curve derivative analysis on stage_5_quality_score"
        f" to determine the filtering threshold."
    )


if __name__ == "__main__":
    run()
    score_all()
