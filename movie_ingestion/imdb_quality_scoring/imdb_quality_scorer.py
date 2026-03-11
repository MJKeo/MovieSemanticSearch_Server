"""
Stage 5: Combined TMDB+IMDB Quality Scorer (v2)

Computes an 8-signal quality score for every 'imdb_scraped' movie and
persists it to movie_progress.stage_5_quality_score.  Scored movies are
advanced to 'imdb_quality_calculated' status.

No hard filters — the quality score is the sole filtering mechanism.
Threshold filtering is a separate downstream step, applied per provider
group after survival-curve analysis.

Signals (weights sum to 1.0, all normalised to [0, 1]):

  Relevance (0.55):
    imdb_vote_count (0.27), critical_attention (0.12),
    community_engagement (0.08), tmdb_popularity (0.08)

  Data sufficiency (0.45):
    featured_reviews_chars (0.15), plot_text_depth (0.12),
    lexical_completeness (0.10), data_completeness (0.08)

See quality_score_design.md for full signal details and rationale.

Re-running processes any remaining unscored movies (crash recovery);
already-scored movies at 'imdb_quality_calculated' are not re-selected.

Usage:
    python -m movie_ingestion.imdb_quality_scoring.imdb_quality_scorer
"""

import datetime
import json
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import orjson

from movie_ingestion.scoring_utils import (
    VoteCountSource,
    score_popularity,
    score_vote_count,
    validate_weights,
)
from movie_ingestion.tracker import (
    INGESTION_DATA_DIR,
    MovieStatus,
    init_db,
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

# Popularity log cap for Stage 5: lowered so that ~p75 of has_providers movies
# saturate at 1.0, giving moderately popular movies full score while preventing
# viral outliers from dominating the signal.
STAGE5_POP_LOG_CAP: float = 4.0

# ---------------------------------------------------------------------------
# Signal weights — must sum to 1.0.
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, float] = {
    "imdb_vote_count":        0.27,
    "critical_attention":     0.12,
    "community_engagement":   0.08,
    "tmdb_popularity":        0.08,
    "featured_reviews_chars": 0.15,
    "plot_text_depth":        0.12,
    "lexical_completeness":   0.10,
    "data_completeness":      0.08,
}

# Guard at module load time — see scoring_utils.validate_weights docstring.
validate_weights(WEIGHTS, label="Stage 5 WEIGHTS")

# ---------------------------------------------------------------------------
# MovieContext — unified view of both data sources for a single movie
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MovieContext:
    """Bundles TMDB and IMDB data for a single movie, providing a unified
    interface for scoring functions.

    tmdb: dict of column values from the tmdb_data table.
    imdb: parsed IMDB JSON dict (guaranteed non-None for scored movies).
    """
    tmdb_id: int
    tmdb: dict
    imdb: dict


# ===========================================================================
# Quality scoring — 8-signal combined TMDB+IMDB scorer
# ===========================================================================
# Each signal function receives a MovieContext and returns a score in [0, 1].
# The public entry point is compute_imdb_quality_score().


# ---------------------------------------------------------------------------
# Relevance signals (0.55 total weight)
# ---------------------------------------------------------------------------


def _score_imdb_vote_count(ctx: MovieContext, today: datetime.date) -> float:
    """IMDB vote count score in [0, 1], log-scaled with age adjustments.

    Primary notability proxy.  Uses IMDB votes (more representative for a
    US-focused app) with the IMDB log cap (12001).  Release date sourced
    from TMDB for the recency/classic multiplier.
    """
    vc = ctx.imdb.get("imdb_vote_count", 0)
    release_date = ctx.tmdb.get("release_date")
    return score_vote_count(vc, release_date, today, VoteCountSource.IMDB)


def _score_critical_attention(ctx: MovieContext) -> float:
    """Critical attention score in [0, 1].

    Presence of professional critical coverage signals that a movie crossed
    a mainstream attention threshold.  Both fields are rare (metacritic ~16%
    of has_providers, reception_summary ~1%), so presence is highly
    discriminating.  Absence is normal and not penalised.

    Scoring: count present fields out of 2.
      0/2 → 0.0,  1/2 → 0.5,  2/2 → 1.0
    """
    count = 0
    if ctx.imdb.get("metacritic_rating") is not None:
        count += 1
    if ctx.imdb.get("reception_summary"):
        count += 1
    return count / 2.0


def _score_community_engagement(ctx: MovieContext) -> float:
    """Community engagement score in [0, 1].

    Measures whether people voluntarily contributed data to a movie's IMDB
    page.  Each contributing field type represents someone's time investment.
    Fields weighted inversely to prevalence: synopses (rare, ~4%) carry
    more weight than plot_keywords (common, ~65%).

    Sub-weights:
      plot_keywords    → 1  (most common, weakest signal)
      featured_reviews → 2  (IMDB primary, TMDB reviews fallback)
      plot_summaries   → 3
      synopses         → 4  (rarest, strongest signal)

    Score = sum of present sub-weights / 10 (max possible = 10).
    """
    total = 0

    # plot_keywords: IMDB list non-empty.
    if ctx.imdb.get("plot_keywords"):
        total += 1

    # featured_reviews: IMDB list non-empty, OR TMDB reviews non-empty.
    has_reviews = bool(ctx.imdb.get("featured_reviews"))
    if not has_reviews:
        tmdb_reviews_json = ctx.tmdb.get("reviews")
        if tmdb_reviews_json:
            try:
                has_reviews = bool(json.loads(tmdb_reviews_json))
            except (json.JSONDecodeError, TypeError):
                pass
    if has_reviews:
        total += 2

    # plot_summaries: IMDB list non-empty.
    if ctx.imdb.get("plot_summaries"):
        total += 3

    # synopses: IMDB list non-empty.
    if ctx.imdb.get("synopses"):
        total += 4

    return total / 10.0


def _score_tmdb_popularity(ctx: MovieContext) -> float:
    """TMDB popularity score in [0, 1], log-scaled.

    TMDB's algorithmic activity score captures short-term buzz and current
    momentum that vote count alone doesn't reflect.  Uses a lowered log cap
    (STAGE5_POP_LOG_CAP) so ~p75 of has_providers movies already saturate
    at 1.0, preventing viral outliers from dominating.
    """
    popularity = ctx.tmdb.get("popularity") or 0.0
    return score_popularity(popularity, log_cap=STAGE5_POP_LOG_CAP)


# ---------------------------------------------------------------------------
# Data sufficiency signals (0.45 total weight)
# ---------------------------------------------------------------------------


def _score_featured_reviews_chars(ctx: MovieContext) -> float:
    """Tiered featured-review score in [0, 1] based on total char count.

    IMDB featured_reviews primary (sum of .text lengths); TMDB reviews JSON
    as fallback when IMDB contributes zero chars.  Reviews feed 6 of 7
    vector spaces, making absence a significant data gap.

    Tiers:  0 chars → 0.0,  1–3000 → 0.33,  3001–8000 → 0.67,  8001+ → 1.0
    """
    total_chars = 0

    # IMDB primary: sum character lengths across all featured review texts.
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
        return 0.0
    elif total_chars <= 3_000:
        return 0.33
    elif total_chars <= 8_000:
        return 0.67
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
    if not overview:
        tmdb_overview_len = ctx.tmdb.get("overview_length") or 0
        total += tmdb_overview_len

    if total == 0:
        return 0.0

    return min(math.log10(total + 1) / math.log10(PLOT_TEXT_LOG_CAP), 1.0)


def _score_lexical_completeness(ctx: MovieContext) -> float:
    """Lexical completeness score in [0, 1].

    Measures how well lexical search can work with this movie.  Six entity
    types, each contributing a capped sub-score of 0.0 to 1.0, then
    averaged to [0, 1].

    Cap each entity type's contribution so a large cast (80 actors) cannot
    mask missing writers or producers.

    Entity scoring:
      actors:               0 → 0.0,  1–4 → 0.5,  5+ → 1.0
      characters:           0 → 0.0,  1–4 → 0.5,  5+ → 1.0
      writers:              0 → 0.0,  1+ → 1.0
      composers:            0 → 0.0,  1+ → 1.0
      producers:            0 → 0.0,  1+ → 1.0
      production_companies: 0 → 0.0,  1+ → 1.0  (IMDB primary, TMDB fallback)
    """
    imdb = ctx.imdb

    # Actors: threshold at 5 for full score, 3-tier with 0 handling.
    actor_count = len(imdb.get("actors") or [])
    actors_sub = 1.0 if actor_count >= 5 else (0.5 if actor_count >= 1 else 0.0)

    # Characters: same threshold as actors.
    char_count = len(imdb.get("characters") or [])
    chars_sub = 1.0 if char_count >= 5 else (0.5 if char_count >= 1 else 0.0)

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
    # Average across 6 entity types → [0, 1].
    return total / 6.0


def _score_data_completeness(ctx: MovieContext) -> float:
    """Data completeness score in [0, 1].

    Measures supplementary fields that enrich LLM-generated vector metadata
    beyond core entities and text.  Six fields, some with tiered sub-scoring,
    then averaged to [0, 1].

    Distinct from lexical_completeness: covers supplementary attributes
    (keywords depth, filming locations, content advisories) rather than
    named entities for lexical matching.

    Field scoring:
      plot_keywords:        0 → 0.0,  1–4 → 0.5,  5+ → 1.0
      overall_keywords:     0 → 0.0,  1 → 0.25,  2–3 → 0.5,  4+ → 1.0
      filming_locations:    0 → 0.0,  1+ → 1.0
      parental_guide_items: 0 → 0.0,  1+ → 1.0
      maturity_rating:      absent → 0.0, present → 1.0  (IMDB → TMDB fallback)
      budget:               absent → 0.0, present → 1.0  (IMDB → TMDB fallback)
    """
    imdb = ctx.imdb

    # Plot keywords: tiered by count.
    pk_count = len(imdb.get("plot_keywords") or [])
    if pk_count >= 5:
        pk_sub = 1.0
    elif pk_count >= 1:
        pk_sub = 0.5
    else:
        pk_sub = 0.0

    # Overall keywords: tiered.
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
    # Average across 6 fields → [0, 1].
    return total / 6.0


# ---------------------------------------------------------------------------
# Public scoring function — reusable for single-movie scoring (cron jobs)
# ---------------------------------------------------------------------------


def compute_imdb_quality_score(
    ctx: MovieContext,
    today: datetime.date,
) -> float:
    """Compute the combined TMDB+IMDB quality score for a single movie.

    Pure function: takes a MovieContext and today's date, returns the weighted
    sum of 8 normalised signals (all in [0, 1]).  No database access — can be
    called from any context (batch scoring, cron jobs, unit tests) by
    constructing a MovieContext from any data source.

    Args:
        ctx:   MovieContext with both tmdb (dict) and imdb (dict) populated.
        today: Scoring reference date (for vote_count age adjustments).

    Returns:
        Quality score in [0, 1].
    """
    return (
        WEIGHTS["imdb_vote_count"]        * _score_imdb_vote_count(ctx, today)
        + WEIGHTS["critical_attention"]   * _score_critical_attention(ctx)
        + WEIGHTS["community_engagement"] * _score_community_engagement(ctx)
        + WEIGHTS["tmdb_popularity"]      * _score_tmdb_popularity(ctx)
        + WEIGHTS["featured_reviews_chars"] * _score_featured_reviews_chars(ctx)
        + WEIGHTS["plot_text_depth"]      * _score_plot_text_depth(ctx)
        + WEIGHTS["lexical_completeness"] * _score_lexical_completeness(ctx)
        + WEIGHTS["data_completeness"]    * _score_data_completeness(ctx)
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_tmdb_data(db: sqlite3.Connection) -> dict[int, dict]:
    """Load TMDB rows for all imdb_scraped movies.

    Selects all columns needed by the 8-signal scorer: release_date,
    watch_provider_keys (for group classification), popularity, overview_length,
    maturity_rating, budget, reviews, and has_production_companies.
    Returns dict keyed by tmdb_id.
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
    """, (MovieStatus.IMDB_SCRAPED,)).fetchall()

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
# Entry point: score all imdb_scraped movies
# ---------------------------------------------------------------------------


def score_all() -> None:
    """Compute and persist quality scores for all imdb_scraped movies.

    For every movie with status='imdb_scraped':
      1. Load TMDB data from tracker DB and IMDB data from JSON files.
      2. Skip movies with missing IMDB JSON (logged, status unchanged).
      3. Build a MovieContext for each movie with IMDB data.
      4. Compute the 8-signal quality score via compute_imdb_quality_score().
      5. Persist the score to movie_progress.stage_5_quality_score.
      6. Advance scored movies to 'imdb_quality_calculated' status.

    Re-running processes any remaining unscored movies (crash recovery);
    already-scored movies at 'imdb_quality_calculated' are not re-selected.
    """
    today = datetime.date.today()
    db = init_db()

    print("Stage 5 quality scorer (v2): loading imdb_scraped movies...")
    tmdb_data = _load_tmdb_data(db)
    tmdb_ids = set(tmdb_data.keys())

    print(f"Loading IMDB JSON files for {len(tmdb_ids):,} movies...")
    imdb_data = _load_imdb_data(tmdb_ids)

    total = len(tmdb_data)
    missing_json = total - len(imdb_data)
    print(f"  {total:,} movies to score (reference date = {today})")
    if missing_json > 0:
        print(f"  WARNING: {missing_json:,} movies have no IMDB JSON — skipping them")

    if total == 0:
        print("No imdb_scraped movies found. Has Stage 4 (IMDB scraping) completed?")
        db.close()
        return

    scored: int = 0
    score_sum: float = 0.0
    score_min: float = float("inf")
    score_max: float = float("-inf")

    # Batch of (score, status, tmdb_id) tuples for bulk UPDATE via executemany.
    batch: list[tuple[float, str, int]] = []

    # Process movies in sorted order for deterministic output.
    sorted_ids = sorted(tmdb_ids)

    try:
        for i, tid in enumerate(sorted_ids):
            # Skip movies with missing IMDB JSON — don't score, don't change status.
            imdb_row = imdb_data.get(tid)
            if imdb_row is None:
                continue

            tmdb_row = tmdb_data[tid]
            ctx = MovieContext(tmdb_id=tid, tmdb=tmdb_row, imdb=imdb_row)
            score = compute_imdb_quality_score(ctx, today)

            batch.append((score, MovieStatus.IMDB_QUALITY_CALCULATED, tid))
            scored += 1
            score_sum += score
            score_min = min(score_min, score)
            score_max = max(score_max, score)

            # Flush batch to disk periodically to bound data loss on crash.
            if len(batch) >= COMMIT_EVERY:
                db.executemany(
                    """UPDATE movie_progress
                       SET stage_5_quality_score = ?,
                           status = ?,
                           updated_at = CURRENT_TIMESTAMP
                       WHERE tmdb_id = ?""",
                    batch,
                )
                db.commit()
                batch.clear()

            if (i + 1) % LOG_EVERY == 0:
                print(f"  Scored {scored:,}/{total:,}")

        # Flush any remaining rows that didn't reach the batch threshold.
        if batch:
            db.executemany(
                """UPDATE movie_progress
                   SET stage_5_quality_score = ?,
                       status = ?,
                       updated_at = CURRENT_TIMESTAMP
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

    print(f"\nStage 5 quality scorer (v2) complete")
    print(f"  Movies scored:    {scored:,}")
    if missing_json > 0:
        print(f"  Skipped (no JSON): {missing_json:,}")
    if scored > 0:
        print(f"  Score min:        {score_min:.4f}")
        print(f"  Score mean:       {score_mean:.4f}")
        print(f"  Score max:        {score_max:.4f}")
    else:
        print("  No movies scored (all missing IMDB JSON)")
    print(
        f"\nNext step: run survival-curve analysis on stage_5_quality_score"
        f" to determine per-group filtering thresholds."
    )


if __name__ == "__main__":
    score_all()
