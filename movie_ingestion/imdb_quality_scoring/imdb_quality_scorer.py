"""
Stage 5: Combined TMDB+IMDB Quality Scorer (v4)

Computes an 8-signal quality score for every 'imdb_scraped' movie and
persists it to movie_progress.stage_5_quality_score.  Scored movies are
advanced to 'imdb_quality_calculated' status.

No hard filters — the quality score is the sole filtering mechanism.
Threshold filtering is a separate downstream step, applied per provider
group after survival-curve analysis.

Signals (weights sum to 1.0, all normalised to [0, 1]):

  Relevance (0.55):
    imdb_notability (0.31), critical_attention (0.08),
    community_engagement (0.08), tmdb_popularity (0.08)

  Data sufficiency (0.45):
    featured_reviews_chars (0.15), plot_text_depth (0.12),
    lexical_completeness (0.10), data_completeness (0.08)

v4 changes from v3:
  - imdb_vote_count → imdb_notability: pure log-scaled vote count replaced
    with a vote-count × Bayesian-adjusted-rating blend.  Three confidence
    tiers (< 100 / 100–999 / ≥ 1000 votes) control how much IMDB rating
    modulates the score, based on rating-stability analysis showing that
    rating reliability varies dramatically by vote count.
  - Weights: imdb_notability 0.25→0.31, critical_attention 0.12→0.08,
    community_engagement 0.10→0.08

v3 changes from v2:
  - featured_reviews: linear chars+count blend replaces 4-tier char-only system
  - community_engagement: plot_keywords and featured_reviews use linear-to-cap
    instead of binary presence
  - lexical_completeness: composers removed, actors/characters linear to 10,
    classic-film age boost added (1.0× at 20yr → 1.5× at 50yr)
  - data_completeness: filming_locations removed, plot_keywords/overall_keywords/
    parental_guide_items use linear-to-cap instead of coarse tiers
  - Weights: imdb_vote_count 0.27→0.25, community_engagement 0.08→0.10

Re-running processes any remaining unscored movies (crash recovery);
already-scored movies at 'imdb_quality_calculated' are not re-selected.

Usage:
    python -m movie_ingestion.imdb_quality_scoring.imdb_quality_scorer
"""

import datetime
import json
import math
import sqlite3
from dataclasses import dataclass

from movie_ingestion.scoring_utils import (
    VC_CLASSIC_BOOST_CAP,
    VC_CLASSIC_RAMP_YEARS,
    VC_CLASSIC_START_YEARS,
    VC_RECENCY_BOOST_MAX,
    score_popularity,
    validate_weights,
)
from movie_ingestion.tracker import (
    MovieStatus,
    deserialize_imdb_row,
    init_db,
)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Flush to disk every N rows processed.  Bounds data loss on crash.
COMMIT_EVERY: int = 1_000

# Emit a progress line every N rows processed.
LOG_EVERY: int = 10_000

# Title types that represent actual movies/shorts — anything else (tvSeries,
# tvMiniSeries, videoGame, etc.) gets an automatic score of 0.
ALLOWED_TITLE_TYPES: set[str] = {"movie", "tvMovie", "short", "video"}

# Log cap for plot_text_depth: overview + plot_summaries + synopses total chars.
# 5001 chars places the ceiling so movies with rich synopses (~p75) saturate
# at 1.0, while overview-only movies (~150 chars) score ~0.59.
PLOT_TEXT_LOG_CAP: int = 5001

# Popularity log cap for Stage 5: lowered so that ~p75 of has_providers movies
# saturate at 1.0, giving moderately popular movies full score while preventing
# viral outliers from dominating the signal.
STAGE5_POP_LOG_CAP: float = 4.0

# ---------------------------------------------------------------------------
# imdb_notability signal: vote count × Bayesian-adjusted rating blend
# ---------------------------------------------------------------------------
# IMDB vote count log cap (same as scoring_utils._VC_LOG_CAPS[IMDB]).
NOTABILITY_VC_LOG_CAP: int = 12001

# Bayesian rating prior parameters.  m=500 sits at the midpoint of the medium
# confidence tier so the prior decays at a rate that matches the tier logic.
# C=6.0 is the observed mean IMDB rating across has_providers movies with votes.
BAYESIAN_M: int = 500
BAYESIAN_C: float = 6.0

# Vote-count tier boundaries.  Derived from rating-stability analysis:
#   < 100:  std ~1.37, 8+ ratings inflated 3-7× by self-selection — noise.
#   100–999: 8+ inflation gone, 13-15% sub-4.0 — rating has real signal.
#   ≥ 1000: std < 1.25, genuine quality drives attention — reliable.
VC_TIER_LOW_CEILING: int = 100
VC_TIER_MED_CEILING: int = 1000

# Per-tier blend weights: (vote_count_weight, rating_weight).
# Each pair sums to 1.0.  Controls how much the Bayesian-adjusted rating
# modulates the log-scaled vote count base within each confidence tier.
BLEND_LOW: tuple[float, float] = (0.95, 0.05)    # almost pure vote count
BLEND_MED: tuple[float, float] = (0.70, 0.30)    # rating has meaningful influence
BLEND_HIGH: tuple[float, float] = (0.85, 0.15)   # vote count dominates, rating modulates

# ---------------------------------------------------------------------------
# Signal weights — must sum to 1.0.
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, float] = {
    "imdb_notability":        0.31,
    "critical_attention":     0.08,
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


def _score_imdb_notability(ctx: MovieContext, today: datetime.date) -> float:
    """Notability score in [0, 1] blending vote count with Bayesian-adjusted rating.

    Replaces the pure vote-count signal with a confidence-tiered blend that
    incorporates IMDB rating, weighted by how trustworthy the rating is at
    the given vote count.

    Three confidence tiers (thresholds from rating-stability analysis):
      Low  (< 100 votes):   Rating is noise (std ~1.37, self-selection).
                             Almost pure vote count (95/5 blend).
      Med  (100–999 votes): Rating has real signal (8+ inflation gone).
                             Balanced blend (70/30 vote/rating).
      High (≥ 1000 votes):  Rating is reliable but movie is already notable.
                             Vote count dominates (85/15 blend).

    The rating component uses IMDB's Bayesian weighted average (m=500, C=6.0)
    to shrink noisy low-vote ratings toward the population mean before blending.

    Falls back to pure vote-count scoring when imdb_rating is absent.
    Age multipliers (recency and classic boosts) are applied after blending.
    """
    vc = ctx.imdb.get("imdb_vote_count", 0)
    imdb_rating = ctx.imdb.get("imdb_rating")

    # --- Vote base: log-scaled vote count in [0, 1] ---
    vote_base = min(math.log10(vc + 1) / math.log10(NOTABILITY_VC_LOG_CAP), 1.0)

    # --- Blend with Bayesian-adjusted rating (when available) ---
    if imdb_rating is not None and vc > 0:
        # Bayesian weighted rating: shrinks toward C at low vote counts.
        bayesian = (vc / (vc + BAYESIAN_M)) * imdb_rating + (BAYESIAN_M / (vc + BAYESIAN_M)) * BAYESIAN_C
        rating_normalized = bayesian / 10.0  # IMDB scale is 1–10 → [0, 1]

        # Select blend weights based on vote-count confidence tier.
        if vc < VC_TIER_LOW_CEILING:
            vote_w, rating_w = BLEND_LOW
        elif vc < VC_TIER_MED_CEILING:
            vote_w, rating_w = BLEND_MED
        else:
            vote_w, rating_w = BLEND_HIGH

        base = vote_w * vote_base + rating_w * rating_normalized
    else:
        # No rating available — degrade gracefully to pure vote count.
        base = vote_base

    # --- Age multipliers (same logic as scoring_utils.score_vote_count) ---
    release_date = ctx.tmdb.get("release_date")
    if release_date is not None:
        try:
            release = datetime.date.fromisoformat(release_date)
            # Floor at 0.5yr to avoid division instability for very new films.
            age_years = max((today - release).days / 365.0, 0.5)

            # Recency: 2× at ≤1yr, hyperbolic decay to 1× at 2yr.
            recent_boost = max(
                1.0,
                min(VC_RECENCY_BOOST_MAX, VC_RECENCY_BOOST_MAX / age_years),
            )

            # Classic: linear ramp from 1× at 20yr to 1.5× at 50yr.
            classic_boost = min(
                VC_CLASSIC_BOOST_CAP,
                1.0 + max(0.0, age_years - VC_CLASSIC_START_YEARS) / VC_CLASSIC_RAMP_YEARS,
            )

            # Apply the larger of the two adjustments.
            multiplier = max(recent_boost, classic_boost)
            base = min(base * multiplier, 1.0)
        except ValueError:
            pass  # Non-parseable date — use base score unchanged.

    return base


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

    Sub-weights (unchanged):
      plot_keywords    → 1  (linear to cap: 5+ = 1.0)
      featured_reviews → 2  (linear to cap: 5+ = 1.0; IMDB primary, TMDB fallback)
      plot_summaries   → 3  (binary)
      synopses         → 4  (binary, rarest, strongest signal)

    Score = sum of (sub-weight × sub-score) / 10 (max possible = 10).
    """
    total = 0.0

    # plot_keywords: linear growth, 5+ keywords = full credit.
    pk_count = len(ctx.imdb.get("plot_keywords") or [])
    total += 1 * min(pk_count / 5, 1.0)

    # featured_reviews: linear growth by count, 5+ reviews = full credit.
    # IMDB primary, TMDB reviews list as fallback.
    review_count = len(ctx.imdb.get("featured_reviews") or [])
    if review_count == 0:
        tmdb_reviews_json = ctx.tmdb.get("reviews")
        if tmdb_reviews_json:
            try:
                review_count = len(json.loads(tmdb_reviews_json))
            except (json.JSONDecodeError, TypeError):
                pass
    total += 2 * min(review_count / 5, 1.0)

    # plot_summaries: binary presence.
    if ctx.imdb.get("plot_summaries"):
        total += 3

    # synopses: binary presence.
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
    """Featured-review score in [0, 1] blending total chars and review count.

    IMDB featured_reviews primary; TMDB reviews JSON as fallback when IMDB
    contributes zero.  Reviews feed 6 of 7 vector spaces, making absence a
    significant data gap.

    Two sub-scores, each linear to a "good enough" cap, then averaged:
      char_score  = min(total_chars / 5000, 1.0)   — 5000+ chars saturates
      count_score = min(review_count / 5, 1.0)      — 5+ reviews saturates
    """
    total_chars = 0
    review_count = 0

    # IMDB primary: sum character lengths and count reviews.
    reviews = ctx.imdb.get("featured_reviews") or []
    for review in reviews:
        text = review.get("text", "")
        total_chars += len(text)
    review_count = len(reviews)

    # TMDB fallback: only used when IMDB contributes zero on both dimensions.
    # Gate on both to avoid overwriting a valid IMDB review_count when IMDB
    # entries happen to have empty text.
    if total_chars == 0 and review_count == 0:
        tmdb_reviews_json = ctx.tmdb.get("reviews")
        if tmdb_reviews_json:
            try:
                tmdb_reviews = json.loads(tmdb_reviews_json)
                for text in tmdb_reviews:
                    total_chars += len(text)
                review_count = len(tmdb_reviews)
            except (json.JSONDecodeError, TypeError):
                pass  # Malformed JSON — treat as no reviews.

    if total_chars == 0 and review_count == 0:
        return 0.0

    # Linear growth to "good enough" caps, then average the two dimensions.
    char_score = min(total_chars / 5000, 1.0)
    count_score = min(review_count / 5, 1.0)
    return (char_score + count_score) / 2.0


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


def _score_lexical_completeness(
    ctx: MovieContext,
    today: datetime.date,
) -> float:
    """Lexical completeness score in [0, 1] with classic-film age boost.

    Measures how well lexical search can work with this movie.  Five entity
    types (composers removed — not useful for search), each contributing a
    capped sub-score of 0.0 to 1.0, then averaged.

    A classic-film boost (same ramp as vote_count: 1.0× at 20yr → 1.5× at
    50yr) compensates for older movies having sparser IMDB entity data.

    Entity scoring:
      actors:               linear to 10 — min(count / 10, 1.0)
      characters:           linear to 10 — min(count / 10, 1.0)
      writers:              binary (most movies have 1–3)
      producers:            binary
      production_companies: binary (IMDB primary, TMDB fallback)
    """
    imdb = ctx.imdb

    # Actors: linear growth, 10+ = full credit.
    actor_count = len(imdb.get("actors") or [])
    actors_sub = min(actor_count / 10, 1.0)

    # Characters: same cap as actors.
    char_count = len(imdb.get("characters") or [])
    chars_sub = min(char_count / 10, 1.0)

    # Binary presence for remaining entity types.
    writers_sub = 1.0 if imdb.get("writers") else 0.0
    producers_sub = 1.0 if imdb.get("producers") else 0.0

    # Production companies: IMDB primary, TMDB has_production_companies fallback.
    if imdb.get("production_companies"):
        prodco_sub = 1.0
    elif ctx.tmdb.get("has_production_companies"):
        prodco_sub = 1.0
    else:
        prodco_sub = 0.0

    # Average across 5 entity types → [0, 1].
    raw = (actors_sub + chars_sub + writers_sub + producers_sub + prodco_sub) / 5.0

    # Classic-film boost: compensate for sparser entity data on older movies.
    # Same linear ramp as vote_count (1.0× at 20yr → 1.5× at 50yr).
    release_date = ctx.tmdb.get("release_date")
    if release_date is not None:
        try:
            release = datetime.date.fromisoformat(release_date)
            age_years = (today - release).days / 365.0
            classic_boost = min(
                VC_CLASSIC_BOOST_CAP,
                1.0 + max(0.0, age_years - VC_CLASSIC_START_YEARS) / VC_CLASSIC_RAMP_YEARS,
            )
            raw = min(raw * classic_boost, 1.0)
        except ValueError:
            pass  # Non-parseable date — use raw score unchanged.

    return raw


def _score_data_completeness(ctx: MovieContext) -> float:
    """Data completeness score in [0, 1].

    Measures supplementary fields that enrich LLM-generated vector metadata
    beyond core entities and text.  Five fields (filming_locations removed —
    irrelevant for animated/short films), some with linear-to-cap scoring,
    then averaged to [0, 1].

    Distinct from lexical_completeness: covers supplementary attributes
    (keywords depth, content advisories) rather than named entities for
    lexical matching.

    Field scoring:
      plot_keywords:        linear to 5 — min(count / 5, 1.0)
      overall_keywords:     linear to 6 — min(count / 6, 1.0)
      parental_guide_items: linear to 3 — min(count / 3, 1.0)
      maturity_rating:      binary (IMDB → TMDB fallback)
      budget:               binary (IMDB → TMDB fallback)
    """
    imdb = ctx.imdb

    # Plot keywords: linear growth, 5+ = full credit.
    pk_count = len(imdb.get("plot_keywords") or [])
    pk_sub = min(pk_count / 5, 1.0)

    # Overall keywords: linear growth, 6+ = full credit.
    ok_count = len(imdb.get("overall_keywords") or [])
    ok_sub = min(ok_count / 6, 1.0)

    # Parental guide items: linear growth, 3+ categories = full credit.
    pg_count = len(imdb.get("parental_guide_items") or [])
    pg_sub = min(pg_count / 3, 1.0)

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

    total = pk_sub + ok_sub + pg_sub + mr_sub + bud_sub
    # Average across 5 fields → [0, 1].
    return total / 5.0


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
    # Non-movie title types (tvSeries, videoGame, etc.) are not useful for the
    # search index.  Score them at 0 so the downstream threshold filter removes them.
    title_type = ctx.imdb.get("imdb_title_type")
    if title_type not in ALLOWED_TITLE_TYPES:
        return 0.0

    # Movies with no text sources (plot_summaries, synopses, featured_reviews) cannot
    # produce meaningful LLM-generated metadata — score them at 0.
    if (
        not ctx.imdb.get("plot_summaries")
        and not ctx.imdb.get("synopses")
        and not ctx.imdb.get("featured_reviews")
    ):
        return 0.0

    return (
        WEIGHTS["imdb_notability"]        * _score_imdb_notability(ctx, today)
        + WEIGHTS["critical_attention"]   * _score_critical_attention(ctx)
        + WEIGHTS["community_engagement"] * _score_community_engagement(ctx)
        + WEIGHTS["tmdb_popularity"]      * _score_tmdb_popularity(ctx)
        + WEIGHTS["featured_reviews_chars"] * _score_featured_reviews_chars(ctx)
        + WEIGHTS["plot_text_depth"]      * _score_plot_text_depth(ctx)
        + WEIGHTS["lexical_completeness"] * _score_lexical_completeness(ctx, today)
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


def _load_imdb_data(db: sqlite3.Connection, tmdb_ids: set[int]) -> dict[int, dict]:
    """Load IMDB data from the imdb_data SQLite table for the given tmdb_ids.

    Each row has individual columns for every IMDBScrapedMovie field.
    JSON TEXT columns (lists/objects) are deserialized back to Python
    types by deserialize_imdb_row().
    """
    if not tmdb_ids:
        return {}
    placeholders = ",".join("?" * len(tmdb_ids))
    prev_factory = db.row_factory
    db.row_factory = sqlite3.Row
    rows = db.execute(
        f"SELECT * FROM imdb_data WHERE tmdb_id IN ({placeholders})",  # noqa: S608
        tuple(tmdb_ids),
    ).fetchall()
    db.row_factory = prev_factory
    return {row["tmdb_id"]: deserialize_imdb_row(row) for row in rows}


# ---------------------------------------------------------------------------
# Entry point: score all imdb_scraped movies
# ---------------------------------------------------------------------------


def score_all() -> None:
    """Compute and persist quality scores for all imdb_quality_calculated movies.

    For every movie with status='imdb_quality_calculated':
      1. Load TMDB data and IMDB data from the tracker DB.
      2. Skip movies with missing IMDB data (logged, status unchanged).
      3. Build a MovieContext for each movie with IMDB data.
      4. Compute the 8-signal quality score via compute_imdb_quality_score().
      5. Persist the updated score to movie_progress.stage_5_quality_score.

    Re-running rescores all imdb_quality_calculated movies (for formula updates
    or crash recovery).
    """
    today = datetime.date.today()
    db = init_db()

    print("Stage 5 quality scorer (v4): loading imdb_scraped movies...")
    tmdb_data = _load_tmdb_data(db)
    tmdb_ids = set(tmdb_data.keys())

    print(f"Loading IMDB data from SQLite for {len(tmdb_ids):,} movies...")
    imdb_data = _load_imdb_data(db, tmdb_ids)

    total = len(tmdb_data)
    missing_json = total - len(imdb_data)
    print(f"  {total:,} movies to score (reference date = {today})")
    if missing_json > 0:
        print(f"  WARNING: {missing_json:,} movies have no IMDB data — skipping them")

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
            # Skip movies with missing IMDB data — don't score, don't change status.
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

    print(f"\nStage 5 quality scorer (v4) complete")
    print(f"  Movies scored:    {scored:,}")
    if missing_json > 0:
        print(f"  Skipped (no JSON): {missing_json:,}")
    if scored > 0:
        print(f"  Score min:        {score_min:.4f}")
        print(f"  Score mean:       {score_mean:.4f}")
        print(f"  Score max:        {score_max:.4f}")
    else:
        print("  No movies scored (all missing IMDB data)")
    print(
        f"\nNext step: run survival-curve analysis on stage_5_quality_score"
        f" to determine per-group filtering thresholds."
    )


if __name__ == "__main__":
    score_all()
