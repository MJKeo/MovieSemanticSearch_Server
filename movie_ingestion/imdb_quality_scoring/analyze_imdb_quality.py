"""
Merged TMDB + IMDB search quality analysis, split by watch-provider group.

Joins TMDB data (from tracker SQLite) with IMDB scraped JSON to produce
a comprehensive report on data coverage and search-quality-relevant
metrics across both sources. Movies are classified into three groups:

  1. has_providers         — movies with at least one US watch provider
  2. recent_no_providers   — no providers, released within the last 75 days
  3. old_no_providers      — no providers, released more than 75 days ago
                             (or missing release date)

Each group gets its own analysis sections and JSON export file:
  ingestion_data/imdb_data_analysis_has_providers.json
  ingestion_data/imdb_data_analysis_recent_no_providers.json
  ingestion_data/imdb_data_analysis_old_no_providers.json

Sections per group:
  1. Merge summary — counts for each source and their overlap
  2. Dual-source coverage — per-field presence from TMDB, IMDB, and either
  3. Search quality percentiles — scale-matters fields only (IMDB primary,
     TMDB fallback), measuring the metric that determines embedding/search
     quality (e.g. entity count, text length)
  4. Composite metrics — multi-attribute scores for vector, lexical, and
     metadata search readiness
  5. Highest missing rates — fields with worst coverage

Usage:
    python -m movie_ingestion.imdb_quality_scoring.analyze_imdb_quality
"""

import datetime
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import orjson

from movie_ingestion.scoring_utils import (
    THEATER_WINDOW_DAYS,
    MovieGroup,
    classify_movie_group,
)
from movie_ingestion.tracker import INGESTION_DATA_DIR, MovieStatus, init_db


# ---------------------------------------------------------------------------
# Paths and group definitions
# ---------------------------------------------------------------------------

_IMDB_DIR = INGESTION_DATA_DIR / "imdb"

# Group keys and their human-readable labels.
_GROUP_HAS_PROVIDERS = "has_providers"
_GROUP_RECENT_NO_PROVIDERS = "recent_no_providers"
_GROUP_OLD_NO_PROVIDERS = "old_no_providers"

_GROUP_LABELS: dict[str, str] = {
    _GROUP_HAS_PROVIDERS: "Has Watch Providers",
    _GROUP_RECENT_NO_PROVIDERS: f"No Providers, Released <= {THEATER_WINDOW_DAYS} Days Ago",
    _GROUP_OLD_NO_PROVIDERS: f"No Providers, Released > {THEATER_WINDOW_DAYS} Days Ago",
}

# Output paths — one JSON per group.
_EXPORT_PATHS: dict[str, Path] = {
    _GROUP_HAS_PROVIDERS: INGESTION_DATA_DIR / "imdb_data_analysis_has_providers.json",
    _GROUP_RECENT_NO_PROVIDERS: INGESTION_DATA_DIR / "imdb_data_analysis_recent_no_providers.json",
    _GROUP_OLD_NO_PROVIDERS: INGESTION_DATA_DIR / "imdb_data_analysis_old_no_providers.json",
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_num(v: float) -> str:
    """Format a number nicely — integers stay clean, floats get 1 decimal."""
    if isinstance(v, int) or v == int(v):
        return f"{int(v):,}"
    return f"{v:,.1f}"


def _percentiles_raw(values: list[float | int]) -> tuple[float, ...] | None:
    """Compute p25, p50, p75, p90 as raw numbers. Returns None if empty."""
    if not values:
        return None
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def pct(p: float) -> float:
        k = (n - 1) * p
        f = int(k)
        c = f + 1 if f + 1 < n else f
        return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])

    return (pct(0.25), pct(0.50), pct(0.75), pct(0.90))


def _percentiles(values: list[float | int]) -> tuple[str, str, str, str]:
    """Compute p25, p50, p75, p90 and return as formatted strings."""
    raw = _percentiles_raw(values)
    if raw is None:
        return ("-", "-", "-", "-")
    return tuple(_format_num(v) for v in raw)


def _pct_str(count: int, total: int) -> str:
    """Format 'count (XX%)' right-aligned to 14 chars."""
    if total == 0:
        return f"{'0 (0%)':>14s}"
    pct = count / total * 100
    return f"{count:,} ({pct:.0f}%)"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_tmdb_data() -> dict[int, dict]:
    """
    Load TMDB rows from the tracker database for movies at any Stage 5
    status (imdb_scraped, imdb_quality_calculated, or imdb_quality_passed).
    Uses a JOIN on movie_progress to filter in SQL rather than pulling IDs
    into Python. Returns dict keyed by tmdb_id.
    """
    db = init_db()
    db.row_factory = None

    query = """
        SELECT
            td.tmdb_id, td.imdb_id, td.title, td.release_date, td.duration,
            td.poster_url, td.watch_provider_keys, td.vote_count,
            td.popularity, td.vote_average, td.overview_length,
            td.genre_count, td.has_revenue, td.has_budget,
            td.has_production_companies, td.has_production_countries,
            td.has_keywords, td.has_cast_and_crew, td.budget,
            td.maturity_rating, td.reviews
        FROM tmdb_data td
        JOIN movie_progress mp ON td.tmdb_id = mp.tmdb_id
        WHERE mp.status IN (?, ?, ?)
    """
    column_names = [
        "tmdb_id", "imdb_id", "title", "release_date", "duration",
        "poster_url", "watch_provider_keys", "vote_count",
        "popularity", "vote_average", "overview_length",
        "genre_count", "has_revenue", "has_budget",
        "has_production_companies", "has_production_countries",
        "has_keywords", "has_cast_and_crew", "budget",
        "maturity_rating", "reviews",
    ]

    rows = db.execute(query, (
        MovieStatus.IMDB_SCRAPED,
        MovieStatus.IMDB_QUALITY_CALCULATED,
        MovieStatus.IMDB_QUALITY_PASSED,
    )).fetchall()
    db.close()

    result: dict[int, dict] = {}
    for row in rows:
        d = dict(zip(column_names, row))
        result[d["tmdb_id"]] = d
    return result


def _load_one_json(path: Path) -> tuple[int, dict] | None:
    """Load a single IMDB JSON file. Returns (tmdb_id, data) or None."""
    try:
        tmdb_id = int(path.stem)
    except ValueError:
        return None
    with open(path, "rb") as f:
        return tmdb_id, orjson.loads(f.read())


def _load_imdb_data(target_ids: set[int]) -> dict[int, dict]:
    """
    Load IMDB JSON files only for movies in the target ID set.
    Files are stored as ingestion_data/imdb/<tmdb_id>.json, so we
    construct paths directly from the ID set rather than globbing the
    entire directory.

    Uses orjson for faster deserialization and ThreadPoolExecutor to
    overlap file I/O across multiple threads.
    """
    # Build the list of paths that should exist for target IDs
    json_files = [_IMDB_DIR / f"{tid}.json" for tid in target_ids]
    # Filter to files that actually exist on disk
    existing_files = [p for p in json_files if p.exists()]

    result: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        for pair in pool.map(_load_one_json, existing_files):
            if pair is not None:
                result[pair[0]] = pair[1]
    return result


def _count_watch_provider_keys(blob: bytes | None) -> int:
    """Unpack a watch_provider_keys BLOB into a count of uint32 keys."""
    if not blob:
        return 0
    return len(blob) // 4


def _decode_tmdb_reviews(raw: str | None) -> list[str]:
    """Decode the TMDB reviews column (JSON-encoded list[str] or None)."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [r for r in parsed if isinstance(r, str) and r]
        return []
    except (json.JSONDecodeError, TypeError):
        return []


# ---------------------------------------------------------------------------
# Merged record — one per movie, holding data from both sources.
# We store the raw dicts so field extractors can access either side.
# ---------------------------------------------------------------------------


@dataclass
class MergedMovie:
    tmdb_id: int
    tmdb: dict | None  # raw TMDB row dict (None if missing)
    imdb: dict | None   # raw IMDB JSON dict (None if missing)


def _merge(
    tmdb_data: dict[int, dict],
    imdb_data: dict[int, dict],
) -> tuple[list[MergedMovie], dict]:
    """
    Merge TMDB and IMDB data by tmdb_id. Returns the merged list plus
    summary stats about the merge.
    """
    all_ids = set(tmdb_data.keys()) | set(imdb_data.keys())
    merged = []
    tmdb_only = 0
    imdb_only = 0
    both = 0

    for tid in sorted(all_ids):
        t = tmdb_data.get(tid)
        i = imdb_data.get(tid)
        if t and i:
            both += 1
        elif t:
            tmdb_only += 1
        else:
            imdb_only += 1
        merged.append(MergedMovie(tmdb_id=tid, tmdb=t, imdb=i))

    stats = {
        "tmdb_total": len(tmdb_data),
        "imdb_total": len(imdb_data),
        "both": both,
        "tmdb_only": tmdb_only,
        "imdb_only": imdb_only,
        "total": len(merged),
    }
    return merged, stats


# ---------------------------------------------------------------------------
# Group classification
# ---------------------------------------------------------------------------


def _classify_movie(m: MergedMovie, today: datetime.date) -> str:
    """Classify a movie into one of the three watch-provider groups.

    Delegates to the canonical classify_movie_group() in scoring_utils, then
    maps to this module's group name strings (which use different naming for
    the no-provider groups to match analysis output filenames).

    1. has_providers — at least one US watch provider key in TMDB data.
    2. recent_no_providers — no providers, released within THEATER_WINDOW_DAYS.
    3. old_no_providers — no providers, released beyond the window (or no date).
    """
    provider_keys = m.tmdb.get("watch_provider_keys") if m.tmdb else None
    release_date = m.tmdb.get("release_date") if m.tmdb else None

    group = classify_movie_group(provider_keys, release_date, today)

    # Map canonical MovieGroup values to this module's local group name strings.
    return _MOVIE_GROUP_TO_LOCAL[group]


# Mapping from canonical MovieGroup enum to local group name strings.
_MOVIE_GROUP_TO_LOCAL: dict[MovieGroup, str] = {
    MovieGroup.HAS_PROVIDERS: _GROUP_HAS_PROVIDERS,
    MovieGroup.NO_PROVIDERS_NEW: _GROUP_RECENT_NO_PROVIDERS,
    MovieGroup.NO_PROVIDERS_OLD: _GROUP_OLD_NO_PROVIDERS,
}


# ---------------------------------------------------------------------------
# Field definitions — each field knows how to check presence and extract
# the search-quality metric from either source.
# ---------------------------------------------------------------------------


@dataclass
class FieldDef:
    """Definition of a single data point for the quality report."""
    name: str                                   # display name
    classification: str                         # "presence" or "scale"
    metric_label: str                           # e.g. "count", "chars"
    # Presence checkers — return True if the value is present in that source.
    # None means the field doesn't exist in that source.
    tmdb_present: Callable[[dict], bool] | None
    imdb_present: Callable[[dict], bool] | None
    # Metric extractor for scale-matters fields. Operates on a MergedMovie
    # and returns the numeric metric (IMDB primary, TMDB fallback), or None
    # if both sources are missing.
    extract_metric: Callable[[MergedMovie], int | float | None] | None


# --- Presence check helpers ---

def _tmdb_str(key: str) -> Callable[[dict], bool]:
    """TMDB presence check: non-empty string."""
    return lambda t: bool(t.get(key))

def _tmdb_int_pos(key: str) -> Callable[[dict], bool]:
    """TMDB presence check: integer > 0."""
    return lambda t: (t.get(key) or 0) > 0

def _tmdb_bool(key: str) -> Callable[[dict], bool]:
    """TMDB presence check: boolean flag == 1."""
    return lambda t: t.get(key) == 1

def _tmdb_blob(key: str) -> Callable[[dict], bool]:
    """TMDB presence check: non-null BLOB."""
    return lambda t: t.get(key) is not None and len(t.get(key, b"")) > 0

def _tmdb_reviews_present() -> Callable[[dict], bool]:
    """TMDB presence check: reviews JSON decodes to non-empty list."""
    return lambda t: len(_decode_tmdb_reviews(t.get("reviews"))) > 0

def _imdb_str(key: str) -> Callable[[dict], bool]:
    """IMDB presence check: non-None, non-empty string."""
    return lambda i: bool(i.get(key))

def _imdb_list(key: str) -> Callable[[dict], bool]:
    """IMDB presence check: non-empty list."""
    return lambda i: bool(i.get(key))

def _imdb_opt(key: str) -> Callable[[dict], bool]:
    """IMDB presence check: not None."""
    return lambda i: i.get(key) is not None

def _imdb_int_pos(key: str) -> Callable[[dict], bool]:
    """IMDB presence check: integer > 0."""
    return lambda i: (i.get(key) or 0) > 0


# --- Metric extractors (IMDB primary, TMDB fallback) ---

def _list_count_imdb(key: str) -> Callable[[MergedMovie], int | None]:
    """Count items in an IMDB-only list field."""
    def extract(m: MergedMovie) -> int | None:
        if m.imdb:
            val = m.imdb.get(key)
            if val:
                return len(val)
        return None
    return extract

def _overview_chars() -> Callable[[MergedMovie], int | None]:
    """Char length of overview: IMDB primary, TMDB overview_length fallback."""
    def extract(m: MergedMovie) -> int | None:
        if m.imdb:
            val = m.imdb.get("overview")
            if val:
                return len(val)
        if m.tmdb:
            val = m.tmdb.get("overview_length", 0)
            if val and val > 0:
                return val
        return None
    return extract

def _genre_count_merged() -> Callable[[MergedMovie], int | None]:
    """Genre count: IMDB list length primary, TMDB genre_count fallback."""
    def extract(m: MergedMovie) -> int | None:
        if m.imdb:
            val = m.imdb.get("genres")
            if val:
                return len(val)
        if m.tmdb:
            val = m.tmdb.get("genre_count", 0)
            if val and val > 0:
                return val
        return None
    return extract

def _synopses_chars() -> Callable[[MergedMovie], int | None]:
    """Total chars across all synopsis entries."""
    def extract(m: MergedMovie) -> int | None:
        if m.imdb:
            entries = m.imdb.get("synopses", [])
            if entries:
                return sum(len(s) for s in entries if isinstance(s, str))
        return None
    return extract

def _plot_summaries_chars() -> Callable[[MergedMovie], int | None]:
    """Total chars across all plot summary entries."""
    def extract(m: MergedMovie) -> int | None:
        if m.imdb:
            entries = m.imdb.get("plot_summaries", [])
            if entries:
                return sum(len(s) for s in entries if isinstance(s, str))
        return None
    return extract

def _reception_summary_chars() -> Callable[[MergedMovie], int | None]:
    """Char length of reception summary (IMDB only)."""
    def extract(m: MergedMovie) -> int | None:
        if m.imdb:
            val = m.imdb.get("reception_summary")
            if val:
                return len(val)
        return None
    return extract

def _featured_reviews_chars() -> Callable[[MergedMovie], int | None]:
    """
    Total text chars across all reviews. IMDB featured_reviews primary
    (sum of .text lengths), TMDB reviews JSON fallback.
    """
    def extract(m: MergedMovie) -> int | None:
        # Try IMDB first
        if m.imdb:
            reviews = m.imdb.get("featured_reviews", [])
            if reviews:
                total = sum(
                    len(r.get("text", ""))
                    for r in reviews
                    if isinstance(r, dict)
                )
                if total > 0:
                    return total
        # Fall back to TMDB reviews
        if m.tmdb:
            tmdb_reviews = _decode_tmdb_reviews(m.tmdb.get("reviews"))
            if tmdb_reviews:
                return sum(len(r) for r in tmdb_reviews)
        return None
    return extract

def _imdb_vote_count_metric() -> Callable[[MergedMovie], int | None]:
    """IMDB vote count as a raw integer metric (IMDB only)."""
    def extract(m: MergedMovie) -> int | None:
        if m.imdb:
            val = m.imdb.get("imdb_vote_count")
            if val is not None and val > 0:
                return val
        return None
    return extract


def _watch_provider_count() -> Callable[[MergedMovie], int | None]:
    """Count of watch provider offering keys (TMDB only)."""
    def extract(m: MergedMovie) -> int | None:
        if m.tmdb:
            count = _count_watch_provider_keys(m.tmdb.get("watch_provider_keys"))
            if count > 0:
                return count
        return None
    return extract


# ---------------------------------------------------------------------------
# Field registry — ordered by search channel relevance
# ---------------------------------------------------------------------------

FIELD_DEFS: list[FieldDef] = [
    # --- Lexical search fields (entity matching) ---
    FieldDef("directors",            "scale",    "count",  None,                        _imdb_list("directors"),           _list_count_imdb("directors")),
    FieldDef("writers",              "scale",    "count",  None,                        _imdb_list("writers"),             _list_count_imdb("writers")),
    FieldDef("actors",               "scale",    "count",  None,                        _imdb_list("actors"),              _list_count_imdb("actors")),
    FieldDef("characters",           "scale",    "count",  None,                        _imdb_list("characters"),          _list_count_imdb("characters")),
    FieldDef("producers",            "scale",    "count",  None,                        _imdb_list("producers"),           _list_count_imdb("producers")),
    FieldDef("composers",            "scale",    "count",  None,                        _imdb_list("composers"),           _list_count_imdb("composers")),
    FieldDef("production_companies", "scale",    "count",  _tmdb_bool("has_production_companies"), _imdb_list("production_companies"), _list_count_imdb("production_companies")),
    FieldDef("overall_keywords",     "scale",    "count",  None,                        _imdb_list("overall_keywords"),    _list_count_imdb("overall_keywords")),
    FieldDef("plot_keywords",        "scale",    "count",  None,                        _imdb_list("plot_keywords"),       _list_count_imdb("plot_keywords")),

    # --- Vector / semantic search fields (embedding quality) ---
    FieldDef("overview",             "scale",    "chars",  _tmdb_int_pos("overview_length"), _imdb_str("overview"),        _overview_chars()),
    FieldDef("synopses",             "scale",    "chars",  None,                        _imdb_list("synopses"),            _synopses_chars()),
    FieldDef("plot_summaries",       "scale",    "chars",  None,                        _imdb_list("plot_summaries"),      _plot_summaries_chars()),
    FieldDef("reception_summary",    "scale",    "chars",  None,                        _imdb_str("reception_summary"),    _reception_summary_chars()),
    FieldDef("featured_reviews",     "scale",    "chars",  _tmdb_reviews_present(),     _imdb_list("featured_reviews"),    _featured_reviews_chars()),
    FieldDef("review_themes",        "scale",    "count",  None,                        _imdb_list("review_themes"),       _list_count_imdb("review_themes")),
    FieldDef("maturity_reasoning",   "scale",    "count",  None,                        _imdb_list("maturity_reasoning"),  _list_count_imdb("maturity_reasoning")),
    FieldDef("parental_guide_items", "scale",    "count",  None,                        _imdb_list("parental_guide_items"),_list_count_imdb("parental_guide_items")),
    FieldDef("genres",               "scale",    "count",  _tmdb_int_pos("genre_count"),_imdb_list("genres"),              _genre_count_merged()),
    FieldDef("languages",            "scale",    "count",  None,                        _imdb_list("languages"),           _list_count_imdb("languages")),
    FieldDef("countries_of_origin",  "scale",    "count",  None,                        _imdb_list("countries_of_origin"), _list_count_imdb("countries_of_origin")),
    FieldDef("filming_locations",    "scale",    "count",  None,                        _imdb_list("filming_locations"),   _list_count_imdb("filming_locations")),
    FieldDef("watch_providers",      "scale",    "count",  _tmdb_blob("watch_provider_keys"), None,                       _watch_provider_count()),

    # --- Metadata search fields (presence-only, more doesn't help search) ---
    FieldDef("title",                "presence", "—",      _tmdb_str("title"),          None,                              None),
    FieldDef("original_title",       "presence", "—",      None,                        _imdb_str("original_title"),       None),
    FieldDef("release_date",         "presence", "—",      _tmdb_str("release_date"),   None,                              None),
    FieldDef("duration",             "presence", "—",      _tmdb_int_pos("duration"),   None,                              None),
    FieldDef("poster_url",           "presence", "—",      _tmdb_str("poster_url"),     None,                              None),
    FieldDef("maturity_rating",      "presence", "—",      _tmdb_str("maturity_rating"),_imdb_str("maturity_rating"),      None),
    FieldDef("budget",               "presence", "—",      _tmdb_int_pos("budget"),     _imdb_opt("budget"),               None),
    FieldDef("imdb_rating",          "presence", "—",      None,                        _imdb_opt("imdb_rating"),          None),
    FieldDef("imdb_vote_count",      "scale",    "votes",  None,                        _imdb_int_pos("imdb_vote_count"),  _imdb_vote_count_metric()),
    FieldDef("metacritic_rating",    "presence", "—",      None,                        _imdb_opt("metacritic_rating"),    None),
]


# ---------------------------------------------------------------------------
# Composite metric functions — compute each metric for a single movie.
# These are called once per movie during the single-pass accumulation.
# ---------------------------------------------------------------------------


def _composite_rich_text(m: MergedMovie) -> float | None:
    """
    Total character count of all text data useful for embeddings:
    overview + synopses + plot_summaries + reception_summary + review texts.
    IMDB primary for each component, TMDB fallback where applicable.
    """
    total = 0

    # Overview
    if m.imdb and m.imdb.get("overview"):
        total += len(m.imdb["overview"])
    elif m.tmdb and (m.tmdb.get("overview_length") or 0) > 0:
        total += m.tmdb["overview_length"]

    # Synopses
    if m.imdb:
        for s in m.imdb.get("synopses", []):
            if isinstance(s, str):
                total += len(s)

    # Plot summaries
    if m.imdb:
        for s in m.imdb.get("plot_summaries", []):
            if isinstance(s, str):
                total += len(s)

    # Reception summary
    if m.imdb and m.imdb.get("reception_summary"):
        total += len(m.imdb["reception_summary"])

    # Featured reviews — IMDB primary, TMDB fallback
    imdb_review_chars = 0
    if m.imdb:
        for r in m.imdb.get("featured_reviews", []):
            if isinstance(r, dict):
                imdb_review_chars += len(r.get("text", ""))
    if imdb_review_chars > 0:
        total += imdb_review_chars
    elif m.tmdb:
        for r in _decode_tmdb_reviews(m.tmdb.get("reviews")):
            total += len(r)

    return total if total > 0 else None


def _composite_lexical_entities(m: MergedMovie) -> int | None:
    """
    Count of all named entities useful for lexical search:
    directors + writers + actors + characters + overall_keywords + plot_keywords.
    """
    if not m.imdb:
        return None
    total = 0
    for key in ("directors", "writers", "actors", "characters",
                "overall_keywords", "plot_keywords"):
        val = m.imdb.get(key, [])
        if val:
            total += len(val)
    return total if total > 0 else None


def _composite_metadata_completeness(m: MergedMovie) -> int:
    """
    Count of metadata attributes present (0–10 scale). Each flag checks
    whether the attribute is available from either source.
    """
    score = 0

    # Genres
    if (m.imdb and m.imdb.get("genres")) or (m.tmdb and (m.tmdb.get("genre_count") or 0) > 0):
        score += 1
    # Maturity rating
    if (m.imdb and m.imdb.get("maturity_rating")) or (m.tmdb and m.tmdb.get("maturity_rating")):
        score += 1
    # Budget
    if (m.imdb and m.imdb.get("budget") is not None) or (m.tmdb and (m.tmdb.get("budget") or 0) > 0):
        score += 1
    # Duration
    if m.tmdb and (m.tmdb.get("duration") or 0) > 0:
        score += 1
    # Languages
    if m.imdb and m.imdb.get("languages"):
        score += 1
    # Countries
    if m.imdb and m.imdb.get("countries_of_origin"):
        score += 1
    # Release date
    if m.tmdb and m.tmdb.get("release_date"):
        score += 1
    # Watch providers
    if m.tmdb and _count_watch_provider_keys(m.tmdb.get("watch_provider_keys")) > 0:
        score += 1
    # IMDB rating
    if m.imdb and m.imdb.get("imdb_rating") is not None:
        score += 1
    # Metacritic rating
    if m.imdb and m.imdb.get("metacritic_rating") is not None:
        score += 1

    return score


def _composite_review_depth(m: MergedMovie) -> int | None:
    """
    Count of review/reception data items: featured_reviews count +
    review_themes count + (1 if reception_summary) + (1 if metacritic).
    TMDB reviews used as fallback for featured_reviews count.
    """
    total = 0

    # Featured reviews (IMDB primary, TMDB fallback)
    imdb_reviews = m.imdb.get("featured_reviews", []) if m.imdb else []
    if imdb_reviews:
        total += len(imdb_reviews)
    elif m.tmdb:
        tmdb_reviews = _decode_tmdb_reviews(m.tmdb.get("reviews"))
        total += len(tmdb_reviews)

    # Review themes
    if m.imdb:
        total += len(m.imdb.get("review_themes", []))

    # Reception summary
    if m.imdb and m.imdb.get("reception_summary"):
        total += 1

    # Metacritic (indicates professional critical coverage)
    if m.imdb and m.imdb.get("metacritic_rating") is not None:
        total += 1

    return total if total > 0 else None


# ---------------------------------------------------------------------------
# Single-pass stats accumulator — iterates movies once, accumulates stats
# into 3 parallel group buckets based on watch-provider classification.
# ---------------------------------------------------------------------------


@dataclass
class FieldStats:
    """Accumulated statistics for one field, built during the single pass."""
    tmdb_count: int = 0
    imdb_count: int = 0
    either_count: int = 0
    metric_values: list[float] = field(default_factory=list)


# Composite display order and labels — used by the print function.
_COMPOSITE_NAMES = [
    ("Rich Text Data",            "chars"),
    ("Lexical Entity Coverage",   "entities"),
    ("Metadata Completeness",     "/10"),
    ("Review Depth",              "items"),
    ("Lexical + Text Combined",   "entities"),
    ("Search Readiness Score",    "/30"),
]

# Type alias for the per-group stats tuple.
GroupStats = tuple[dict[str, FieldStats], dict[str, list[float]], int]


def _new_group_accumulators() -> tuple[dict[str, FieldStats], dict[str, list[float]]]:
    """Create fresh field_stats and composite_values accumulators for one group."""
    field_stats = {fd.name: FieldStats() for fd in FIELD_DEFS}
    composite_values: dict[str, list[float]] = {name: [] for name, _ in _COMPOSITE_NAMES}
    return field_stats, composite_values


def _accumulate_movie(
    m: MergedMovie,
    field_stats: dict[str, FieldStats],
    composite_values: dict[str, list[float]],
) -> None:
    """Accumulate a single movie's field presence, metrics, and composites
    into the provided group accumulators. Extracted from the loop body so
    the same logic serves all three groups without duplication."""

    # --- Field presence + metrics (one check per field per movie) ---
    for fd in FIELD_DEFS:
        fs = field_stats[fd.name]
        has_tmdb = fd.tmdb_present(m.tmdb) if fd.tmdb_present and m.tmdb else False
        has_imdb = fd.imdb_present(m.imdb) if fd.imdb_present and m.imdb else False
        if has_tmdb:
            fs.tmdb_count += 1
        if has_imdb:
            fs.imdb_count += 1
        if has_tmdb or has_imdb:
            fs.either_count += 1
        # Scale-matters fields: extract metric value
        if fd.classification == "scale" and fd.extract_metric:
            v = fd.extract_metric(m)
            if v is not None and v > 0:
                fs.metric_values.append(v)

    # --- Composites: compute base metrics once, derive the rest ---
    rich_text = _composite_rich_text(m)
    lexical = _composite_lexical_entities(m)
    metadata = _composite_metadata_completeness(m)
    review = _composite_review_depth(m)

    # Lexical + Text Combined: entity count only if text >= 500 chars
    if lexical is not None:
        lex_and_text = lexical if (rich_text is not None and rich_text >= 500) else 0
    else:
        lex_and_text = None

    # Search Readiness: metadata (0-10) + lexical (0-10) + text (0-10)
    readiness_score = metadata + min(lexical or 0, 30) / 3.0 + min(rich_text or 0, 5000) / 500.0
    readiness = round(readiness_score, 1) if readiness_score > 0 else None

    # Append non-None values to the composite lists
    if rich_text is not None:
        composite_values["Rich Text Data"].append(rich_text)
    if lexical is not None:
        composite_values["Lexical Entity Coverage"].append(lexical)
    # Metadata completeness is always an int — always appended.
    composite_values["Metadata Completeness"].append(metadata)
    if review is not None:
        composite_values["Review Depth"].append(review)
    if lex_and_text is not None:
        composite_values["Lexical + Text Combined"].append(lex_and_text)
    if readiness is not None:
        composite_values["Search Readiness Score"].append(readiness)


def _compute_all_stats(
    merged: list[MergedMovie],
    today: datetime.date,
) -> dict[str, GroupStats]:
    """
    Single pass over all movies. Classifies each movie into one of three
    watch-provider groups, then accumulates per-field presence counts,
    metric values, and per-composite value lists into that group's stats.

    Each movie is visited exactly once. Classification and accumulation
    happen in the same iteration — no secondary passes needed.

    Returns:
        dict mapping group key -> (field_stats, composite_values, count)
    """
    # Initialize accumulators for all three groups.
    accumulators: dict[str, tuple[dict[str, FieldStats], dict[str, list[float]]]] = {
        group: _new_group_accumulators()
        for group in (_GROUP_HAS_PROVIDERS, _GROUP_RECENT_NO_PROVIDERS, _GROUP_OLD_NO_PROVIDERS)
    }
    counts: dict[str, int] = {
        _GROUP_HAS_PROVIDERS: 0,
        _GROUP_RECENT_NO_PROVIDERS: 0,
        _GROUP_OLD_NO_PROVIDERS: 0,
    }

    for m in merged:
        group = _classify_movie(m, today)
        counts[group] += 1
        fs, cv = accumulators[group]
        _accumulate_movie(m, fs, cv)

    # Package into the return format: group -> (field_stats, composite_values, count).
    return {
        group: (accumulators[group][0], accumulators[group][1], counts[group])
        for group in (_GROUP_HAS_PROVIDERS, _GROUP_RECENT_NO_PROVIDERS, _GROUP_OLD_NO_PROVIDERS)
    }


# ---------------------------------------------------------------------------
# Printing helpers — now read from pre-computed stats, no iteration needed.
# ---------------------------------------------------------------------------

_LINE_WIDTH = 110


def _print_merge_summary(group: str, total: int, overall_total: int) -> None:
    """Section 1: Group summary header."""
    label = _GROUP_LABELS[group]
    pct = total / overall_total * 100 if overall_total > 0 else 0
    print(f"\n{'#' * _LINE_WIDTH}")
    print(f"  Group: {label}  —  {total:,} movies ({pct:.1f}% of {overall_total:,} total)")
    print(f"{'#' * _LINE_WIDTH}")


def _print_coverage_table(field_stats: dict[str, FieldStats], total: int) -> None:
    """Section 2: Dual-source coverage for every field."""
    print(f"\n{'=' * _LINE_WIDTH}")
    print("Dual-Source Coverage")
    print(
        f"  {'Field':<24s} "
        f"{'TMDB Present':>16s}  "
        f"{'IMDB Present':>16s}  "
        f"{'Either':>16s}"
    )
    print(f"  {'-' * (_LINE_WIDTH - 4)}")

    for fd in FIELD_DEFS:
        fs = field_stats[fd.name]

        tmdb_str = "N/A" if fd.tmdb_present is None else _pct_str(fs.tmdb_count, total)
        imdb_str = "N/A" if fd.imdb_present is None else _pct_str(fs.imdb_count, total)
        either_str = _pct_str(fs.either_count, total)

        print(
            f"  {fd.name:<24s} "
            f"{tmdb_str:>16s}  "
            f"{imdb_str:>16s}  "
            f"{either_str:>16s}"
        )


def _print_percentile_table(field_stats: dict[str, FieldStats]) -> None:
    """Section 3: Scale-matters percentiles (IMDB primary, TMDB fallback)."""
    scale_fields = [fd for fd in FIELD_DEFS if fd.classification == "scale"]

    print(f"\n{'=' * _LINE_WIDTH}")
    print("Search Quality Percentiles (IMDB primary, TMDB fallback)")
    print(
        f"  {'Field':<24s} "
        f"{'Present':>12s}  "
        f"{'p25':>8s}  {'p50':>8s}  {'p75':>8s}  {'p90':>8s}  "
        f"{'Metric'}"
    )
    print(f"  {'-' * (_LINE_WIDTH - 4)}")

    for fd in scale_fields:
        fs = field_stats[fd.name]
        p25, p50, p75, p90 = _percentiles(fs.metric_values)
        present_str = f"{len(fs.metric_values):,}"

        print(
            f"  {fd.name:<24s} "
            f"{present_str:>12s}  "
            f"{p25:>8s}  {p50:>8s}  {p75:>8s}  {p90:>8s}  "
            f"[{fd.metric_label}]"
        )


def _print_composite_table(composite_values: dict[str, list[float]]) -> None:
    """Section 4: Multi-attribute composite metrics."""
    print(f"\n{'=' * _LINE_WIDTH}")
    print("Composite Search Quality Metrics")
    print(
        f"  {'Metric':<28s} "
        f"{'Present':>12s}  "
        f"{'p25':>8s}  {'p50':>8s}  {'p75':>8s}  {'p90':>8s}  "
        f"{'Unit'}"
    )
    print(f"  {'-' * (_LINE_WIDTH - 4)}")

    for name, unit in _COMPOSITE_NAMES:
        values = composite_values[name]
        p25, p50, p75, p90 = _percentiles(values)
        present_str = f"{len(values):,}"

        print(
            f"  {name:<28s} "
            f"{present_str:>12s}  "
            f"{p25:>8s}  {p50:>8s}  {p75:>8s}  {p90:>8s}  "
            f"[{unit}]"
        )


def _print_missing_summary(field_stats: dict[str, FieldStats], total: int) -> None:
    """Section 5: Top 10 fields by missing rate (combined either-source)."""
    print(f"\n{'=' * _LINE_WIDTH}")
    print("Highest Missing Rates (either source)")
    print(f"  {'Field':<24s}  {'Missing':>16s}")
    print(f"  {'-' * 42}")

    field_missing: list[tuple[str, int, float]] = []
    for fd in FIELD_DEFS:
        fs = field_stats[fd.name]
        missing = total - fs.either_count
        pct = missing / total * 100 if total > 0 else 0
        field_missing.append((fd.name, missing, pct))

    field_missing.sort(key=lambda x: x[1], reverse=True)
    for name, missing, pct in field_missing[:10]:
        print(f"  {name:<24s}  {missing:,} ({pct:.0f}%)")

    print()


# ---------------------------------------------------------------------------
# JSON export — structured output for downstream consumption
# ---------------------------------------------------------------------------


def _build_percentile_dict(values: list[float | int]) -> dict | None:
    """Build a percentile dict from a list of values, or None if empty."""
    raw = _percentiles_raw(values)
    if raw is None:
        return None
    return {"p25": raw[0], "p50": raw[1], "p75": raw[2], "p90": raw[3]}


def _export_json(
    group: str,
    field_stats: dict[str, FieldStats],
    composite_values: dict[str, list[float]],
    total: int,
) -> None:
    """
    Export analysis data for one group as JSON to its corresponding file
    in ingestion_data/imdb_data_analysis_<group>.json.
    """
    # --- Field-level data ---
    fields = []
    for fd in FIELD_DEFS:
        fs = field_stats[fd.name]
        entry: dict = {
            "name": fd.name,
            "classification": fd.classification,
            "tmdb_present": fs.tmdb_count if fd.tmdb_present is not None else None,
            "imdb_present": fs.imdb_count if fd.imdb_present is not None else None,
            "either_present": fs.either_count,
            "missing": total - fs.either_count,
            "missing_pct": round((total - fs.either_count) / total * 100, 1) if total > 0 else 0,
        }
        # Scale-matters fields include percentiles
        if fd.classification == "scale" and fs.metric_values:
            entry["metric_label"] = fd.metric_label
            entry["metric_count"] = len(fs.metric_values)
            entry["percentiles"] = _build_percentile_dict(fs.metric_values)
        fields.append(entry)

    # --- Composite data ---
    composites = []
    for name, unit in _COMPOSITE_NAMES:
        values = composite_values[name]
        composites.append({
            "name": name,
            "unit": unit,
            "present": len(values),
            "percentiles": _build_percentile_dict(values),
        })

    report = {
        "group": group,
        "group_label": _GROUP_LABELS[group],
        "total_movies": total,
        "fields": fields,
        "composites": composites,
    }

    export_path = _EXPORT_PATHS[group]
    with open(export_path, "wb") as f:
        f.write(orjson.dumps(report, option=orjson.OPT_INDENT_2))
    print(f"  Exported: {export_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    today = datetime.date.today()

    print("Loading TMDB data from tracker database (status = imdb_scraped)...")
    tmdb_data = _load_tmdb_data()

    # Use the TMDB result keys to scope IMDB file loading, so both
    # sources cover exactly the same set of movies.
    target_ids = set(tmdb_data.keys())
    print(f"  Found {len(target_ids):,} movies")

    print("Loading IMDB scraped JSON files...")
    imdb_data = _load_imdb_data(target_ids)

    merged, stats = _merge(tmdb_data, imdb_data)
    overall_total = stats["total"]

    if overall_total == 0:
        print("No data found. Ensure the ingestion pipeline has run.")
        return

    # Single pass: classify every movie and accumulate stats per group.
    print(f"Analyzing data quality (reference date = {today})...")
    group_stats = _compute_all_stats(merged, today)

    # Print and export results for each group.
    for group in (_GROUP_HAS_PROVIDERS, _GROUP_RECENT_NO_PROVIDERS, _GROUP_OLD_NO_PROVIDERS):
        field_stats, composite_values, total = group_stats[group]

        # Section 1: Group header with count
        _print_merge_summary(group, total, overall_total)

        if total == 0:
            print("  (no movies in this group)")
            continue

        # Section 2: Dual-source coverage
        _print_coverage_table(field_stats, total)

        # Section 3: Search quality percentiles
        _print_percentile_table(field_stats)

        # Section 4: Composite metrics
        _print_composite_table(composite_values)

        # Section 5: Highest missing rates
        _print_missing_summary(field_stats, total)

    # Export structured JSON for each group.
    print(f"\nExporting JSON reports...")
    for group in (_GROUP_HAS_PROVIDERS, _GROUP_RECENT_NO_PROVIDERS, _GROUP_OLD_NO_PROVIDERS):
        field_stats, composite_values, total = group_stats[group]
        _export_json(group, field_stats, composite_values, total)


if __name__ == "__main__":
    main()
