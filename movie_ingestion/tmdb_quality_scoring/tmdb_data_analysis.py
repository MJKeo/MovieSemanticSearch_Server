"""
TMDB data quality analysis — Stage 2.5 diagnostic script.

Reads all rows from the tmdb_data table and computes a comprehensive suite of
statistics to inform the Stage 3 quality funnel design:
  - Per-attribute distributions and missing-value rates
  - Vote count survival curve (how many movies survive at each vote threshold)
  - Cross-attribute relationships (boolean completeness vs. vote_count bands, etc.)

Results are split by US watch-provider availability (movies WITH at least one
US provider vs. movies WITHOUT) so that distributions can be compared between
the two populations. Output is written atomically to:
    ./ingestion_data/tmdb_data_analysis_with_providers.json
    ./ingestion_data/tmdb_data_analysis_no_providers.json

Usage:
    python -m movie_ingestion.tmdb_quality_scoring.tmdb_data_analysis
"""

import datetime
import struct

from movie_ingestion.tracker import INGESTION_DATA_DIR, init_db, save_json

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------

OUTPUT_PATH_WITH_PROVIDERS = INGESTION_DATA_DIR / "tmdb_data_analysis_with_providers.json"
OUTPUT_PATH_NO_PROVIDERS   = INGESTION_DATA_DIR / "tmdb_data_analysis_no_providers.json"

# ---------------------------------------------------------------------------
# Constants
#
# IMPORTANT: These constants are used only for their own attribute's analysis.
# Any cross-attribute analysis (e.g. grouping movies by vote_count to examine
# poster availability) must use percentile-derived thresholds computed from
# the actual data, never these hardcoded values.
# ---------------------------------------------------------------------------

# Survival curve thresholds — used exclusively in analyze_vote_count to
# answer "how many movies survive if we require vote_count >= X?"
VOTE_COUNT_THRESHOLDS = [10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select(lst: list, indices: list[int]) -> list:
    """Extract elements at the given indices from *lst*."""
    return [lst[i] for i in indices]


def _percentile(sorted_values: list, p: float) -> float | None:
    """
    Compute the p-th percentile from a pre-sorted list via linear interpolation.

    Args:
        sorted_values: Ascending-sorted list of numeric values.
        p:             Percentile in [0, 100].

    Returns:
        Interpolated percentile value, or None for empty input.
    """
    if not sorted_values:
        return None
    n = len(sorted_values)
    idx = (p / 100.0) * (n - 1)
    lower = int(idx)
    upper = min(lower + 1, n - 1)
    frac = idx - lower
    return sorted_values[lower] * (1.0 - frac) + sorted_values[upper] * frac


def _compute_percentile_set(sorted_values: list) -> dict:
    """Compute the standard six-percentile summary for a sorted list."""
    return {
        "p25": _percentile(sorted_values, 25),
        "p50": _percentile(sorted_values, 50),
        "p75": _percentile(sorted_values, 75),
        "p90": _percentile(sorted_values, 90),
        "p95": _percentile(sorted_values, 95),
        "p99": _percentile(sorted_values, 99),
    }


def _unpack_provider_keys(blob: bytes | None) -> list[int]:
    """
    Unpack a watch_provider_keys BLOB into a list of integer provider keys.

    The BLOB is packed as little-endian unsigned 32-bit integers ('<NI' format),
    matching the encoding written by tmdb_fetcher.py.

    Returns an empty list for None or zero-length input.
    """
    if not blob:
        return []
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}I", blob))


def _vc_band_label(vc: int, vc_percentiles: dict) -> str:
    """
    Assign a vote_count value to a named percentile band for cross-attribute
    analysis.

    Uses pre-computed vote_count percentile thresholds rather than hardcoded
    boundaries, so the bands adapt to the actual data distribution.

    Bands (left-inclusive, right-exclusive except the top):
        below_p25  → vc < p25
        p25_to_p50 → p25 <= vc < p50
        p50_to_p75 → p50 <= vc < p75
        p75_to_p90 → p75 <= vc < p90
        above_p90  → vc >= p90
    """
    p25 = vc_percentiles["p25"]
    p50 = vc_percentiles["p50"]
    p75 = vc_percentiles["p75"]
    p90 = vc_percentiles["p90"]

    if vc < p25:
        return "below_p25"
    elif vc < p50:
        return "p25_to_p50"
    elif vc < p75:
        return "p50_to_p75"
    elif vc < p90:
        return "p75_to_p90"
    else:
        return "above_p90"


# ---------------------------------------------------------------------------
# Per-attribute analysis functions
# Each function accepts only the data relevant to its attribute (or explicit
# companion arguments such as pre-computed percentiles). No function imports
# or uses constants from a sibling attribute.
# ---------------------------------------------------------------------------


def analyze_title(titles: list[str | None]) -> dict:
    """
    Analyze the title column.

    The hard-filter condition from the pipeline spec is `title IS NULL OR title = ''`.
    null_or_empty_count is the direct count of movies that fail that filter.
    """
    null_count = sum(1 for t in titles if t is None)
    empty_count = sum(1 for t in titles if t is not None and t == "")

    return {
        "null_count": null_count,
        "empty_count": empty_count,
        "null_or_empty_count": null_count + empty_count,  # hard-filter target
    }


def analyze_release_date(release_dates: list[str | None]) -> dict:
    """
    Analyze the release_date column (ISO 8601 strings or None).

    Reports:
    - null_count:    Movies with no release date — may have no IMDB data.
    - future_count:  Movies releasing after today — IMDB pages will be sparse.
    - anomaly_count: Non-null values that are not parseable YYYY dates (data
                     corruption indicator).
    - by_year_bucket: Distribution across historical eras. US consumers heavily
                      favour contemporary content; large pre-1970 volumes
                      represent a likely elimination opportunity.
    """
    today_str = datetime.date.today().isoformat()  # "YYYY-MM-DD"

    null_count = 0
    future_count = 0
    anomaly_count = 0
    year_buckets: dict[str, int] = {
        "pre_1930": 0,
        "1930_1949": 0,
        "1950_1969": 0,
        "1970_1999": 0,
        "2000_2009": 0,
        "2010_2019": 0,
        "2020_2024": 0,
        "2025_present": 0,
    }

    for rd in release_dates:
        if not rd:
            null_count += 1
            continue

        # A valid date must start with a 4-digit non-zero year
        if len(rd) < 4 or rd[:4] == "0000":
            anomaly_count += 1
            continue

        try:
            year = int(rd[:4])
        except ValueError:
            anomaly_count += 1
            continue

        # Future-release flag (movie still counted in its year bucket)
        if rd > today_str:
            future_count += 1

        if year < 1930:
            year_buckets["pre_1930"] += 1
        elif year < 1950:
            year_buckets["1930_1949"] += 1
        elif year < 1970:
            year_buckets["1950_1969"] += 1
        elif year < 2000:
            year_buckets["1970_1999"] += 1
        elif year < 2010:
            year_buckets["2000_2009"] += 1
        elif year < 2020:
            year_buckets["2010_2019"] += 1
        elif year < 2025:
            year_buckets["2020_2024"] += 1
        else:
            year_buckets["2025_present"] += 1

    return {
        "null_count": null_count,
        "future_count": future_count,
        "anomaly_count": anomaly_count,
        "by_year_bucket": year_buckets,
    }


def analyze_duration(durations: list[int | None]) -> dict:
    """
    Analyze the duration column (runtime in minutes, or None).

    null_or_zero_count is the primary hard-filter target from the pipeline
    spec. The bucket distribution reveals the volume of short-form content
    (< 40 min) that should be excluded vs. legitimate feature-length films.
    """
    buckets: dict[str, int] = {
        "null_or_0": 0,  # hard-filter target: NULL runtime or 0
        "1_39": 0,       # short films, TV pilots — below the hard-filter floor
        "40_60": 0,      # short features, some documentaries
        "61_90": 0,      # typical indie / art-house length
        "91_120": 0,     # standard mainstream feature
        "121_180": 0,    # long feature / prestige drama
        "181_plus": 0,   # epics and multi-part releases
    }

    for d in durations:
        if d is None or d == 0:
            buckets["null_or_0"] += 1
        elif d < 40:
            buckets["1_39"] += 1
        elif d <= 60:
            buckets["40_60"] += 1
        elif d <= 90:
            buckets["61_90"] += 1
        elif d <= 120:
            buckets["91_120"] += 1
        elif d <= 180:
            buckets["121_180"] += 1
        else:
            buckets["181_plus"] += 1

    return {
        "by_bucket": buckets,
    }


def analyze_poster_url(poster_urls: list[str | None]) -> dict:
    """
    Analyze the poster_url column.

    Reports only the overall null rate. A poster-vs-vote_count band breakdown
    is computed in analyze_cross_attributes, which owns all vote_count-grouped
    cross-tabulations.
    """
    total = len(poster_urls)
    null_count = sum(1 for p in poster_urls if not p)

    return {
        "null_count": null_count,
        "null_rate": round(null_count / total, 4) if total else 0.0,
    }


def analyze_watch_providers(provider_key_counts: list[int]) -> dict:
    """
    Analyze US watch provider availability.

    Accepts a pre-computed list of distinct provider key counts per movie
    (0 = no US providers) rather than raw BLOBs, to avoid re-unpacking data
    that was already decoded in run().

    Movies on US platforms are directly relevant to a US audience; this is
    the strongest geographic relevance signal in the dataset.
    """
    with_provider = sum(1 for c in provider_key_counts if c > 0)
    without_provider = len(provider_key_counts) - with_provider

    # Distribution of how many distinct provider keys a movie has.
    # Movies with many providers tend to be mainstream / widely distributed.
    count_dist: dict[str, int] = {
        "0": 0,
        "1_2": 0,
        "3_5": 0,
        "6_10": 0,
        "11_plus": 0,
    }
    for c in provider_key_counts:
        if c == 0:
            count_dist["0"] += 1
        elif c <= 2:
            count_dist["1_2"] += 1
        elif c <= 5:
            count_dist["3_5"] += 1
        elif c <= 10:
            count_dist["6_10"] += 1
        else:
            count_dist["11_plus"] += 1

    return {
        "with_any_us_provider_count": with_provider,
        "without_us_provider_count": without_provider,
        "provider_count_distribution": count_dist,
    }


def analyze_vote_count(
    vote_counts: list[int],
    percentiles: dict,
) -> dict:
    """
    Analyze the vote_count column.

    vote_count is the single strongest quality signal. The survival curve
    (how many movies remain at each hardcoded threshold) is the primary output
    for choosing the quality-funnel floor.

    The semantic bucket distribution and percentile summary provide shape context.
    """
    total = len(vote_counts)
    zero_count = sum(1 for vc in vote_counts if vc == 0)

    # Survival curve: for each threshold value, how many movies pass?
    # Uses VOTE_COUNT_THRESHOLDS — this is intentional and appropriate here
    # because the survival curve IS the vote_count analysis.
    survival_curve = []
    for threshold in VOTE_COUNT_THRESHOLDS:
        surviving = sum(1 for vc in vote_counts if vc >= threshold)
        eliminated = total - surviving
        survival_curve.append({
            "threshold": threshold,
            "surviving_count": surviving,
            "eliminated_count": eliminated,
            "elimination_pct": round(eliminated / total * 100, 2) if total else 0.0,
        })

    # Semantic buckets for shape visualization (self-contained; not reused elsewhere)
    buckets: dict[str, int] = {
        "0": 0,
        "1_10": 0,
        "11_50": 0,
        "51_100": 0,
        "101_500": 0,
        "501_1000": 0,
        "1001_5000": 0,
        "5001_10000": 0,
        "10001_plus": 0,
    }
    for vc in vote_counts:
        if vc == 0:
            buckets["0"] += 1
        elif vc <= 10:
            buckets["1_10"] += 1
        elif vc <= 50:
            buckets["11_50"] += 1
        elif vc <= 100:
            buckets["51_100"] += 1
        elif vc <= 500:
            buckets["101_500"] += 1
        elif vc <= 1000:
            buckets["501_1000"] += 1
        elif vc <= 5000:
            buckets["1001_5000"] += 1
        elif vc <= 10000:
            buckets["5001_10000"] += 1
        else:
            buckets["10001_plus"] += 1

    mean_vc = sum(vote_counts) / total if total else 0.0

    return {
        "zero_count": zero_count,
        "survival_curve": survival_curve,
        "by_bucket": buckets,
        "percentiles": percentiles,
        "mean": mean_vc,
        "median": percentiles["p50"],
    }


def analyze_popularity(
    popularities: list[float],
    vote_counts: list[int],
    vc_p50: float,
    pop_percentiles: dict,
) -> dict:
    """
    Analyze the popularity column (TMDB's composite engagement metric).

    Cumulative counts use popularity's own percentile thresholds so the
    thresholds adapt to the actual distribution rather than using hardcoded
    arbitrary values.

    The vote_count cross-tab uses both attributes' computed medians (p50) as
    the split points — no hardcoded values.
    """
    total = len(popularities)
    zero_count = sum(1 for p in popularities if p == 0.0)
    pop_p50 = pop_percentiles["p50"]

    # Cumulative counts at popularity's own percentile thresholds.
    # Replaces the naive approach of counting movies above arbitrary constants
    # (e.g., popularity >= 1.0, >= 5.0) with data-driven thresholds.
    at_percentile_thresholds: dict[str, dict] = {}
    for pct_label in ("p50", "p75", "p90", "p95", "p99"):
        threshold = pop_percentiles[pct_label]
        at_percentile_thresholds[pct_label] = {
            "threshold": threshold,
            "count_at_or_above": sum(1 for p in popularities if p >= threshold),
        }

    # Semantic distribution for shape visualization only.
    # Bucket edges have human-readable meaning but are NOT used for filtering.
    buckets: dict[str, int] = {
        "0.0": 0,
        "0.01_1.0": 0,
        "1.01_5.0": 0,
        "5.01_10.0": 0,
        "10.01_50.0": 0,
        "50.01_100.0": 0,
        "100.01_plus": 0,
    }
    for p in popularities:
        if p == 0.0:
            buckets["0.0"] += 1
        elif p <= 1.0:
            buckets["0.01_1.0"] += 1
        elif p <= 5.0:
            buckets["1.01_5.0"] += 1
        elif p <= 10.0:
            buckets["5.01_10.0"] += 1
        elif p <= 50.0:
            buckets["10.01_50.0"] += 1
        elif p <= 100.0:
            buckets["50.01_100.0"] += 1
        else:
            buckets["100.01_plus"] += 1

    # 2×2 cross-tab: are high vote_count and high popularity correlated?
    # Both medians are computed from the actual data distribution.
    hv_hp = hv_lp = lv_hp = lv_lp = 0
    for vc, pop in zip(vote_counts, popularities):
        is_high_vc = vc >= vc_p50
        is_high_pop = pop >= pop_p50
        if is_high_vc and is_high_pop:
            hv_hp += 1
        elif is_high_vc:
            hv_lp += 1
        elif is_high_pop:
            lv_hp += 1
        else:
            lv_lp += 1

    return {
        "zero_count": zero_count,
        "percentiles": pop_percentiles,
        "at_percentile_thresholds": at_percentile_thresholds,
        "by_bucket": buckets,
        "vote_count_cross_tab": {
            "note": "Split at each attribute's computed median (p50), not hardcoded values",
            "split_at_vc_median": vc_p50,
            "split_at_pop_median": pop_p50,
            "high_vc_high_pop": hv_hp,
            "high_vc_low_pop": hv_lp,
            "low_vc_high_pop": lv_hp,
            "low_vc_low_pop": lv_lp,
        },
    }


def analyze_vote_average(
    vote_averages: list[float],
    vote_counts: list[int],
    vc_percentiles: dict,
) -> dict:
    """
    Analyze the vote_average column (0–10 TMDB rating scale).

    Raw vote_average is nearly meaningless without controlling for vote_count:
    a 9.0 from 2 votes and a 9.0 from 20,000 votes are completely different.
    The primary output is the vote_average distribution at each vote_count
    percentile floor — this reveals whether ratings become a genuine signal
    once a movie has meaningful engagement.

    Note: we do NOT filter on low vote_average. Universally panned movies are
    still culturally relevant and searchable; we only filter on irrelevance.
    """
    zero_count = sum(1 for va in vote_averages if va == 0.0)

    # For each vote_count percentile floor (p50/p75/p90/p95/p99), show the
    # vote_average distribution among movies that pass that floor.
    # Floors are derived from vote_count's own percentiles — no hardcoded values.
    at_vote_count_percentiles: dict[str, dict] = {}
    for pct_label in ("p50", "p75", "p90", "p95", "p99"):
        floor = vc_percentiles[pct_label]
        eligible = [va for va, vc in zip(vote_averages, vote_counts) if vc >= floor]

        if not eligible:
            at_vote_count_percentiles[pct_label] = {"vc_floor": floor, "count": 0}
            continue

        dist: dict[str, int] = {
            "0.0": 0,       # no votes registered
            "0.1_3.0": 0,   # terrible
            "3.1_5.0": 0,   # poor
            "5.1_6.5": 0,   # mediocre / average
            "6.6_7.5": 0,   # good
            "7.6_8.5": 0,   # very good
            "8.6_10.0": 0,  # excellent
        }
        for va in eligible:
            if va == 0.0:
                dist["0.0"] += 1
            elif va <= 3.0:
                dist["0.1_3.0"] += 1
            elif va <= 5.0:
                dist["3.1_5.0"] += 1
            elif va <= 6.5:
                dist["5.1_6.5"] += 1
            elif va <= 7.5:
                dist["6.6_7.5"] += 1
            elif va <= 8.5:
                dist["7.6_8.5"] += 1
            else:
                dist["8.6_10.0"] += 1

        at_vote_count_percentiles[pct_label] = {
            "vc_floor": floor,
            "count": len(eligible),
            "mean_vote_average": sum(eligible) / len(eligible),
            "distribution": dist,
        }

    return {
        "zero_count": zero_count,
        "at_vote_count_percentiles": at_vote_count_percentiles,
    }


def analyze_overview_length(overview_lengths: list[int]) -> dict:
    """
    Analyze the overview_length column (character count of the TMDB overview).

    A zero count means the movie has no overview text at all; this is directly
    readable as by_bucket["0"]. The full distribution shows how many movies
    have a thin vs. substantive overview, informing where a minimum-length
    hard filter should be placed.
    """
    buckets: dict[str, int] = {
        "0": 0,
        "1_20": 0,      # effectively empty — likely just a title placeholder
        "21_50": 0,     # very thin; may not have meaningful IMDB data either
        "51_100": 0,    # short but present
        "101_200": 0,   # typical short overview
        "201_500": 0,   # full overview
        "501_plus": 0,  # long / detailed overview
    }
    for ol in overview_lengths:
        if ol == 0:
            buckets["0"] += 1
        elif ol <= 20:
            buckets["1_20"] += 1
        elif ol <= 50:
            buckets["21_50"] += 1
        elif ol <= 100:
            buckets["51_100"] += 1
        elif ol <= 200:
            buckets["101_200"] += 1
        elif ol <= 500:
            buckets["201_500"] += 1
        else:
            buckets["501_plus"] += 1

    return {
        "by_bucket": buckets,
    }


def analyze_genre_count(genre_counts: list[int]) -> dict:
    """
    Analyze the genre_count column.

    Only the hard-filter-relevant values are reported. Having 1 vs. 3 genres
    is not a quality signal (a pure horror film with 1 genre is no worse a
    candidate than a multi-genre blockbuster), so per-value distribution is
    deliberately excluded.
    """
    zero_count = sum(1 for gc in genre_counts if gc == 0)
    negative_count = sum(1 for gc in genre_counts if gc < 0)

    return {
        "zero_count": zero_count,
        "negative_count": negative_count,  # data corruption check; expected to be 0
    }


def analyze_boolean_fields(
    has_revenue: list[int],
    has_budget: list[int],
    has_production_companies: list[int],
    has_production_countries: list[int],
    has_keywords: list[int],
    has_cast_and_crew: list[int],
) -> dict:
    """
    Analyze all six boolean data-completeness fields.

    Per-field rates reveal how reliably each piece of metadata is present.
    The revenue×budget cross-tab examines the two most commercially meaningful
    completeness flags together.
    The completeness score distribution (0–6 sum of all booleans per movie)
    reveals how many movies are data ghosts (0) vs. fully catalogued (6).
    """
    total = len(has_revenue)

    def _field_stats(values: list[int]) -> dict:
        true_count = sum(values)
        return {
            "true_count": true_count,
            "false_count": total - true_count,
            "true_rate": round(true_count / total, 4) if total else 0.0,
        }

    # Revenue × budget 2×2 cross-tab
    neither = revenue_only = budget_only = both = 0
    for rev, bud in zip(has_revenue, has_budget):
        if rev and bud:
            both += 1
        elif rev:
            revenue_only += 1
        elif bud:
            budget_only += 1
        else:
            neither += 1

    # Completeness score: integer 0–6 summing all boolean fields per movie.
    # Score 0 = no structured metadata at all (likely a data ghost).
    # Score 6 = fully catalogued with all completeness signals present.
    score_dist: dict[str, int] = {str(i): 0 for i in range(7)}
    for rev, bud, pc, pco, kw, cc in zip(
        has_revenue, has_budget, has_production_companies,
        has_production_countries, has_keywords, has_cast_and_crew,
    ):
        score_dist[str(rev + bud + pc + pco + kw + cc)] += 1

    return {
        "has_revenue": _field_stats(has_revenue),
        "has_budget": _field_stats(has_budget),
        "has_production_companies": _field_stats(has_production_companies),
        "has_production_countries": _field_stats(has_production_countries),
        "has_keywords": _field_stats(has_keywords),
        "has_cast_and_crew": _field_stats(has_cast_and_crew),
        "revenue_budget_crosstab": {
            "neither": neither,
            "revenue_only": revenue_only,
            "budget_only": budget_only,
            "both": both,
        },
        "completeness_score_distribution": score_dist,
    }


def analyze_cross_attributes(
    vote_counts: list[int],
    provider_key_counts: list[int],
    poster_urls: list[str | None],
    has_revenue: list[int],
    has_budget: list[int],
    has_production_companies: list[int],
    has_production_countries: list[int],
    has_keywords: list[int],
    has_cast_and_crew: list[int],
    vc_percentiles: dict,
) -> dict:
    """
    Cross-attribute analysis: how do completeness signals, provider availability,
    and poster presence vary across vote_count percentile bands?

    All groupings use percentile-derived vote_count bands (not hardcoded bucket
    edges) so the analysis adapts to the actual data distribution. This answers
    whether each signal adds information beyond what vote_count alone provides,
    or is merely a downstream proxy for it.
    """
    band_labels = ["below_p25", "p25_to_p50", "p50_to_p75", "p75_to_p90", "above_p90"]

    # Accumulators: one dict per band per signal type
    band_provider: dict[str, dict] = {b: {"with": 0, "without": 0} for b in band_labels}
    band_completeness: dict[str, dict] = {b: {str(i): 0 for i in range(7)} for b in band_labels}
    band_poster: dict[str, dict] = {b: {"null": 0, "not_null": 0} for b in band_labels}

    # Single pass through all rows to populate all accumulators simultaneously
    for i, vc in enumerate(vote_counts):
        band = _vc_band_label(vc, vc_percentiles)

        # US provider availability
        if provider_key_counts[i] > 0:
            band_provider[band]["with"] += 1
        else:
            band_provider[band]["without"] += 1

        # Boolean completeness score (0–6)
        score = (has_revenue[i] + has_budget[i] + has_production_companies[i] +
                 has_production_countries[i] + has_keywords[i] + has_cast_and_crew[i])
        band_completeness[band][str(score)] += 1

        # Poster availability
        if not poster_urls[i]:
            band_poster[band]["null"] += 1
        else:
            band_poster[band]["not_null"] += 1

    # Build output dicts with derived rates

    vc_band_vs_provider: dict[str, dict] = {}
    for band in band_labels:
        s = band_provider[band]
        total_band = s["with"] + s["without"]
        vc_band_vs_provider[band] = {
            "total": total_band,
            "with_us_provider": s["with"],
            "without_us_provider": s["without"],
            "coverage_rate": round(s["with"] / total_band, 4) if total_band else 0.0,
        }

    vc_band_vs_completeness: dict[str, dict] = {}
    for band in band_labels:
        dist = band_completeness[band]
        total_band = sum(dist.values())
        total_score_sum = sum(int(k) * v for k, v in dist.items())
        vc_band_vs_completeness[band] = {
            "total": total_band,
            "mean_completeness_score": round(total_score_sum / total_band, 4) if total_band else 0.0,
            "score_distribution": dist,
        }

    vc_band_vs_poster: dict[str, dict] = {}
    for band in band_labels:
        s = band_poster[band]
        total_band = s["null"] + s["not_null"]
        vc_band_vs_poster[band] = {
            "total": total_band,
            "null_poster_count": s["null"],
            "null_poster_rate": round(s["null"] / total_band, 4) if total_band else 0.0,
        }

    # Include the actual percentile boundary values in output so the bands
    # are self-documenting without needing to cross-reference vote_count stats
    p25, p50, p75, p90 = (
        vc_percentiles["p25"], vc_percentiles["p50"],
        vc_percentiles["p75"], vc_percentiles["p90"],
    )
    band_boundaries = {
        "below_p25":  f"vote_count < {p25}",
        "p25_to_p50": f"{p25} <= vote_count < {p50}",
        "p50_to_p75": f"{p50} <= vote_count < {p75}",
        "p75_to_p90": f"{p75} <= vote_count < {p90}",
        "above_p90":  f"vote_count >= {p90}",
    }

    return {
        "band_boundaries": band_boundaries,
        "vote_count_band_vs_us_provider": vc_band_vs_provider,
        "vote_count_band_vs_completeness_score": vc_band_vs_completeness,
        "vote_count_band_vs_poster_null_rate": vc_band_vs_poster,
    }


# ---------------------------------------------------------------------------
# Analysis builder — runs the full suite on a given set of column lists
# ---------------------------------------------------------------------------


def _build_analysis(
    titles: list[str | None],
    release_dates: list[str | None],
    durations: list[int | None],
    poster_urls: list[str | None],
    provider_key_counts: list[int],
    vote_counts: list[int],
    popularities: list[float],
    vote_averages: list[float],
    overview_lengths: list[int],
    genre_counts: list[int],
    has_revenue: list[int],
    has_budget: list[int],
    has_production_companies: list[int],
    has_production_countries: list[int],
    has_keywords: list[int],
    has_cast_and_crew: list[int],
) -> dict:
    """
    Run the full analysis suite on the provided column lists and return
    the results dict.

    Computes percentiles from the group's own data so distributions are
    self-contained and comparable across splits.
    """
    total = len(vote_counts)

    # Compute percentiles for this group's vote_count and popularity
    sorted_vcs = sorted(vote_counts)
    vc_percentiles = _compute_percentile_set(sorted_vcs)

    sorted_pops = sorted(popularities)
    pop_percentiles = _compute_percentile_set(sorted_pops)

    return {
        "total_movies": total,
        # Top-level for quick reference without digging into the vote_count section
        "vote_count_percentiles": vc_percentiles,
        "title":           analyze_title(titles),
        "release_date":    analyze_release_date(release_dates),
        "duration":        analyze_duration(durations),
        "poster_url":      analyze_poster_url(poster_urls),
        "watch_providers": analyze_watch_providers(provider_key_counts),
        "vote_count":      analyze_vote_count(vote_counts, vc_percentiles),
        "popularity":      analyze_popularity(
            popularities, vote_counts, vc_percentiles["p50"], pop_percentiles,
        ),
        "vote_average":    analyze_vote_average(vote_averages, vote_counts, vc_percentiles),
        "overview_length": analyze_overview_length(overview_lengths),
        "genre_count":     analyze_genre_count(genre_counts),
        "boolean_fields":  analyze_boolean_fields(
            has_revenue, has_budget, has_production_companies,
            has_production_countries, has_keywords, has_cast_and_crew,
        ),
        "cross_attributes": analyze_cross_attributes(
            vote_counts, provider_key_counts, poster_urls,
            has_revenue, has_budget, has_production_companies,
            has_production_countries, has_keywords, has_cast_and_crew,
            vc_percentiles,
        ),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """
    Load all tmdb_data rows, split by US watch-provider availability, compute
    the full analysis suite for each group, and write results to separate files.
    """
    db = init_db()

    print("Loading tmdb_data from database (excluding filtered_out movies)...")
    rows = db.execute("""
        SELECT
            d.release_date, d.duration, d.poster_url, d.watch_provider_keys,
            d.vote_count, d.popularity, d.vote_average, d.overview_length,
            d.genre_count, d.has_revenue, d.has_budget, d.has_production_companies,
            d.has_production_countries, d.has_keywords, d.has_cast_and_crew,
            d.title
        FROM tmdb_data d
        JOIN movie_progress p ON d.tmdb_id = p.tmdb_id
        WHERE p.status != 'filtered_out'
    """).fetchall()
    db.close()

    total = len(rows)
    if total == 0:
        print("No rows found in tmdb_data. Has Stage 2 (TMDB fetching) completed?")
        return

    print(f"Loaded {total:,} rows. Extracting columns...")

    # Extract each column into its own list. Using individual list comprehensions
    # rather than zip(*rows) to avoid potential argument-count issues at large n.
    release_dates          = [r[0]  for r in rows]
    durations              = [r[1]  for r in rows]
    poster_urls            = [r[2]  for r in rows]
    provider_blobs         = [r[3]  for r in rows]
    vote_counts            = [r[4]  for r in rows]
    popularities           = [r[5]  for r in rows]
    vote_averages          = [r[6]  for r in rows]
    overview_lengths       = [r[7]  for r in rows]
    genre_counts           = [r[8]  for r in rows]
    has_revenue            = [r[9]  for r in rows]
    has_budget             = [r[10] for r in rows]
    has_production_companies  = [r[11] for r in rows]
    has_production_countries  = [r[12] for r in rows]
    has_keywords           = [r[13] for r in rows]
    has_cast_and_crew      = [r[14] for r in rows]
    titles                 = [r[15] for r in rows]

    # Decode provider BLOBs once here and share the count list with both
    # analyze_watch_providers and analyze_cross_attributes to avoid double work.
    provider_key_counts = [len(_unpack_provider_keys(b)) for b in provider_blobs]

    # All 16 column lists in a fixed order matching _build_analysis's signature.
    # Used by _select to partition each list by the same index set.
    all_columns = [
        titles, release_dates, durations, poster_urls,
        provider_key_counts, vote_counts, popularities, vote_averages,
        overview_lengths, genre_counts,
        has_revenue, has_budget, has_production_companies,
        has_production_countries, has_keywords, has_cast_and_crew,
    ]

    # Partition indices by US watch-provider availability
    with_idx = [i for i, c in enumerate(provider_key_counts) if c > 0]
    no_idx   = [i for i, c in enumerate(provider_key_counts) if c == 0]

    # --- Group 1: movies WITH at least one US watch provider ---
    print(f"\nWith US providers: {len(with_idx):,} movies")
    if with_idx:
        with_columns = [_select(col, with_idx) for col in all_columns]
        with_results = _build_analysis(*with_columns)
        save_json(OUTPUT_PATH_WITH_PROVIDERS, with_results)
        print(f"  Written to {OUTPUT_PATH_WITH_PROVIDERS}")
    else:
        print("  (empty group — skipping)")

    # --- Group 2: movies WITHOUT any US watch provider ---
    print(f"Without US providers: {len(no_idx):,} movies")
    if no_idx:
        no_columns = [_select(col, no_idx) for col in all_columns]
        no_results = _build_analysis(*no_columns)
        save_json(OUTPUT_PATH_NO_PROVIDERS, no_results)
        print(f"  Written to {OUTPUT_PATH_NO_PROVIDERS}")
    else:
        print("  (empty group — skipping)")

    print(f"\nDone. Total: {total:,} = {len(with_idx):,} (with) + {len(no_idx):,} (without)")


if __name__ == "__main__":
    run()
