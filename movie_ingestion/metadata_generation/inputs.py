"""
Input data structures and shared utilities for the generation pipeline.

Data structures:

MovieInputData (dataclass):
    Raw fields loaded from tmdb_data + imdb_data tables. This is the
    contract between "load data from SQLite" and "run pre-consolidation."
    Fields include: tmdb_id, title, release_year, overview, genres,
    plot_synopses, plot_summaries, plot_keywords, overall_keywords,
    featured_reviews, reception_summary, audience_reception_attributes,
    maturity_rating, maturity_reasoning, parental_guide_items.

    Methods:
    - batch_id(generation_type) -> str: batch request custom_id
    - title_with_year() -> str: "Title (Year)" formatted string
    - merged_keywords() -> list[str]: deduplicated union of plot + overall keywords
    - maturity_summary() -> str | None: delegates to pre_consolidation.consolidate_maturity()

ConsolidatedInputs (dataclass):
    Post-pre-consolidation data passed to generators. Contains:
    - movie_input (MovieInputData): original raw fields (access via
      consolidated.movie_input.genres, etc.)
    - title_with_year (str): "Title (Year)" formatted string
    - merged_keywords (list[str]): deduplicated union of plot + overall
    - maturity_summary (str | None): consolidated maturity string
    - skip_assessment (SkipAssessment): which generations to run/skip

SkipAssessment (dataclass):
    Result of assess_skip_conditions:
    - generations_to_run (set[str]): which generation types to include
    - skip_reasons (dict[str, str]): why each skipped generation was skipped

Utility functions:

build_user_prompt(**labeled_fields) -> str:
    Assembles a user prompt from named fields. Skips None values.
    Example: build_user_prompt(title="The Matrix (1999)", genres="Action, Sci-Fi")
    -> "title: The Matrix (1999)\\ngenres: Action, Sci-Fi"

Data loading:

load_movie_input_data(tmdb_ids, tracker_db_path) -> dict[int, MovieInputData]:
    Load MovieInputData for a set of movies from the ingestion tracker DB.
    Joins tmdb_data + imdb_data tables, parses JSON columns, returns a dict
    mapping tmdb_id -> MovieInputData. Movies missing from either table are
    skipped with a warning.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


# ---------------------------------------------------------------------------
# Prompt formatting helpers
# ---------------------------------------------------------------------------

class MultiLineList(list):
    """Marker for lists whose items should be formatted on separate lines.

    When passed to build_user_prompt, items are rendered as:
        key:
        - item 1
        - item 2

    Regular lists are comma-separated on a single line. Use MultiLineList
    for long-text items like plot_summaries and plot_synopses where each
    entry is a multi-sentence block.
    """
    pass


# ---------------------------------------------------------------------------
# Generation type constants
# ---------------------------------------------------------------------------

class MetadataType(StrEnum):
    """Canonical names for all 8 metadata generation types.

    StrEnum so values are plain strings — compatible with SQLite column
    names, custom_id formatting, and anywhere a string is expected.
    """
    PLOT_EVENTS = "plot_events"
    RECEPTION = "reception"
    PLOT_ANALYSIS = "plot_analysis"
    VIEWER_EXPERIENCE = "viewer_experience"
    WATCH_CONTEXT = "watch_context"
    NARRATIVE_TECHNIQUES = "narrative_techniques"
    PRODUCTION_KEYWORDS = "production_keywords"
    SOURCE_OF_INSPIRATION = "source_of_inspiration"


WAVE1_TYPES = frozenset({MetadataType.PLOT_EVENTS, MetadataType.RECEPTION})
WAVE2_TYPES = frozenset({
    MetadataType.PLOT_ANALYSIS,
    MetadataType.VIEWER_EXPERIENCE,
    MetadataType.WATCH_CONTEXT,
    MetadataType.NARRATIVE_TECHNIQUES,
    MetadataType.PRODUCTION_KEYWORDS,
    MetadataType.SOURCE_OF_INSPIRATION,
})
ALL_GENERATION_TYPES = WAVE1_TYPES | WAVE2_TYPES


# ---------------------------------------------------------------------------
# Batch custom_id helpers
# ---------------------------------------------------------------------------

def build_custom_id(tmdb_id: int, metadata_type: MetadataType) -> str:
    """Build a batch request custom_id in the format '{metadata_type}_{tmdb_id}'.

    Used by request building (to set custom_id in JSONL) and result
    processing (to map results back to movies). The format is
    deterministic so the same movie+type always produces the same ID.
    """
    return f"{metadata_type}_{tmdb_id}"


def parse_custom_id(custom_id: str) -> tuple[MetadataType, int]:
    """Parse a batch custom_id back into (metadata_type, tmdb_id).

    Inverse of build_custom_id(). Splits on the last underscore —
    safe because tmdb_id is always a pure integer (no underscores),
    so 'plot_events_12345' correctly splits into (MetadataType.PLOT_EVENTS, 12345).

    Raises ValueError if the metadata_type portion is not a valid MetadataType.
    """
    type_str, tmdb_id_str = custom_id.rsplit("_", 1)
    metadata_type = MetadataType(type_str)  # raises ValueError if invalid
    return metadata_type, int(tmdb_id_str)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MovieInputData:
    """Raw fields loaded from tmdb_data + imdb_data tables.

    This is the contract between "load data from SQLite" and
    "run pre-consolidation." All list fields default to empty lists,
    optional scalars default to None.
    """
    tmdb_id: int
    title: str
    release_year: int | None = None
    overview: str = ""
    genres: list[str] = field(default_factory=list)
    plot_synopses: list[str] = field(default_factory=list)
    plot_summaries: list[str] = field(default_factory=list)
    plot_keywords: list[str] = field(default_factory=list)
    overall_keywords: list[str] = field(default_factory=list)
    # Each dict has keys: summary (str), text (str)
    featured_reviews: list[dict] = field(default_factory=list)
    reception_summary: str | None = None
    # Each dict has keys: name (str), sentiment (str)
    audience_reception_attributes: list[dict] = field(default_factory=list)
    maturity_rating: str = ""
    maturity_reasoning: list[str] = field(default_factory=list)
    # Each dict has keys: category (str), severity (str)
    parental_guide_items: list[dict] = field(default_factory=list)

    def batch_id(self, generation_type: MetadataType) -> str:
        """Produce a batch request custom_id, e.g. 'plot_events_12345'.

        Delegates to the module-level build_custom_id() which is also
        used by result processing to parse IDs back.
        """
        return build_custom_id(self.tmdb_id, generation_type)

    def title_with_year(self) -> str:
        """Format title as 'Title (Year)' for temporal grounding and
        disambiguation. Returns just 'Title' if year is unknown."""
        if self.release_year is not None:
            return f"{self.title} ({self.release_year})"
        return self.title

    def merged_keywords(self) -> list[str]:
        """Deduplicated union of plot + overall keywords, normalized.

        Order-preserving: plot_keywords first, then unique overall_keywords
        appended. Each keyword is lowercased and stripped before dedup.
        Matches the merge logic in pre_consolidation.route_keywords().
        """
        return list(dict.fromkeys(
            kw.lower().strip()
            for kw in self.plot_keywords + self.overall_keywords
        ))

    def best_plot_fallback(self) -> str | None:
        """Find the longest available raw plot text from this movie's sources.

        Used when Wave 1 plot_events did not produce a plot_summary.
        Selects the longest of:
            - First synopsis entry (plot_synopses[0])
            - Longest plot_summary entry
            - Overview text

        Returns None if no plot text is available at all.
        """
        candidates: list[str] = []
        if self.plot_synopses:
            candidates.append(self.plot_synopses[0])
        if self.plot_summaries:
            candidates.append(max(self.plot_summaries, key=len))
        if self.overview:
            candidates.append(self.overview)
        if not candidates:
            return None
        return max(candidates, key=len)

    def maturity_summary(self) -> str | None:
        """Consolidated maturity string from available maturity data.

        Delegates to pre_consolidation.consolidate_maturity() which is
        the single source of truth for the priority chain logic.
        """
        # Import here to avoid circular import (pre_consolidation imports
        # MovieInputData from this module).
        from .pre_consolidation import consolidate_maturity

        return consolidate_maturity(
            self.maturity_rating,
            self.maturity_reasoning,
            self.parental_guide_items,
        )


@dataclass(slots=True)
class SkipAssessment:
    """Result of assess_skip_conditions — which generations to run and
    why each skipped generation was skipped."""
    generations_to_run: set[str] = field(default_factory=set)
    skip_reasons: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ConsolidatedInputs:
    """Post-pre-consolidation data passed to generators.

    Contains the original MovieInputData by composition plus derived
    fields from pre-consolidation. Generators access original fields
    via consolidated.movie_input.genres, etc.
    """
    movie_input: MovieInputData

    # Derived fields from pre-consolidation
    title_with_year: str = ""
    merged_keywords: list[str] = field(default_factory=list)
    maturity_summary: str | None = None
    skip_assessment: SkipAssessment = field(default_factory=SkipAssessment)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def build_user_prompt(**labeled_fields: str | list | None) -> str:
    """Assemble a user prompt from named fields, skipping None values.

    List values are joined with ', '. Each field becomes one line
    formatted as 'key: value'.

    Example:
        build_user_prompt(title="The Matrix (1999)", genres=["Action", "Sci-Fi"])
        -> "title: The Matrix (1999)\\ngenres: Action, Sci-Fi"
    """
    lines: list[str] = []
    for key, value in labeled_fields.items():
        if value is None:
            continue
        # Skip empty lists — avoids producing malformed lines like "key: \n- "
        if isinstance(value, list) and not value:
            continue
        if isinstance(value, MultiLineList):
            # Long-text items get one entry per line with bullet prefix
            formatted_items = "\n- ".join(str(item) for item in value)
            value = f"\n- {formatted_items}"
        elif isinstance(value, list):
            # Short items (keywords, genres) are comma-separated
            value = ", ".join(str(item) for item in value)
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Default paths match the ingestion pipeline conventions
_DEFAULT_TRACKER_DB = Path("ingestion_data/tracker.db")


def load_movie_input_data(
    tmdb_ids: list[int],
    tracker_db_path: Path = _DEFAULT_TRACKER_DB,
) -> dict[int, MovieInputData]:
    """Load MovieInputData for the given tmdb_ids from the ingestion tracker.

    Queries the tracker SQLite for title/release_date from tmdb_data, and
    the full per-movie IMDB data from imdb_data. Both tables must be populated
    (i.e., the movie must have completed at least through Stage 4 of the
    ingestion pipeline).

    Movies missing from either table are logged and skipped — they will not
    appear in the returned dict.

    Args:
        tmdb_ids: List of TMDB movie IDs to load.
        tracker_db_path: Path to ingestion_data/tracker.db.

    Returns:
        Dict mapping tmdb_id → MovieInputData for all successfully loaded movies.
    """
    if not tracker_db_path.exists():
        raise FileNotFoundError(
            f"Tracker DB not found at {tracker_db_path}. "
            "Ensure the ingestion pipeline has run through at least Stage 4."
        )

    if not tmdb_ids:
        return {}

    result: dict[int, MovieInputData] = {}
    placeholders = ", ".join("?" * len(tmdb_ids))

    with sqlite3.connect(str(tracker_db_path)) as tracker:
        tracker.row_factory = sqlite3.Row

        # Join tmdb_data (title, release_date) with imdb_data (everything else).
        # imdb_data.overview is preferred over tmdb_data (more detailed);
        # release_year is extracted from tmdb_data.release_date ("YYYY-MM-DD").
        rows = tracker.execute(
            f"""
            SELECT
                t.tmdb_id,
                t.title,
                CAST(SUBSTR(t.release_date, 1, 4) AS INTEGER) AS release_year,
                i.overview,
                i.maturity_rating,
                i.reception_summary,
                i.genres,
                i.synopses,
                i.plot_summaries,
                i.plot_keywords,
                i.overall_keywords,
                i.featured_reviews,
                i.review_themes,
                i.maturity_reasoning,
                i.parental_guide_items
            FROM tmdb_data t
            JOIN imdb_data i ON t.tmdb_id = i.tmdb_id
            WHERE t.tmdb_id IN ({placeholders})
            """,
            tmdb_ids,
        ).fetchall()

    # JSON columns are stored as TEXT arrays; parse each one
    def _parse_json_list(raw: str | None) -> list:
        if not raw:
            return []
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []

    loaded_ids: set[int] = set()
    for row in rows:
        tmdb_id = row["tmdb_id"]
        loaded_ids.add(tmdb_id)

        result[tmdb_id] = MovieInputData(
            tmdb_id=tmdb_id,
            title=row["title"] or "",
            release_year=row["release_year"],
            overview=row["overview"] or "",
            genres=_parse_json_list(row["genres"]),
            plot_synopses=_parse_json_list(row["synopses"]),
            plot_summaries=_parse_json_list(row["plot_summaries"]),
            plot_keywords=_parse_json_list(row["plot_keywords"]),
            overall_keywords=_parse_json_list(row["overall_keywords"]),
            featured_reviews=_parse_json_list(row["featured_reviews"]),
            reception_summary=row["reception_summary"],
            # review_themes maps to audience_reception_attributes: [{name, sentiment}]
            audience_reception_attributes=_parse_json_list(row["review_themes"]),
            maturity_rating=row["maturity_rating"] or "",
            maturity_reasoning=_parse_json_list(row["maturity_reasoning"]),
            parental_guide_items=_parse_json_list(row["parental_guide_items"]),
        )

    # Warn for any requested IDs that were not found in both tables
    missing = set(tmdb_ids) - loaded_ids
    if missing:
        print(
            f"  Warning: {len(missing)} movie(s) not found in tracker "
            f"(missing from tmdb_data or imdb_data): {sorted(missing)}"
        )

    print(f"  Loaded {len(result)}/{len(tmdb_ids)} movies from tracker.")
    return result


# ---------------------------------------------------------------------------
# Wave 1 output loading for Wave 2 consumers
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Wave1Outputs:
    """Parsed fields from Wave 1 generation outputs (plot_events + reception).

    All fields default to None — callers pick whichever subset they need.
    A single DB query populates all fields, so there's no cost to loading
    fields that a particular Wave 2 generator doesn't use.
    """
    # From plot_events
    plot_summary: str | None = None

    # From reception extraction zone
    thematic_observations: str | None = None
    emotional_observations: str | None = None
    craft_observations: str | None = None
    source_material_hint: str | None = None


def load_wave1_outputs(
    tmdb_id: int,
    tracker_db_path: Path = _DEFAULT_TRACKER_DB,
) -> Wave1Outputs:
    """Load all Wave 1 output fields needed by Wave 2 generators.

    Single DB query for plot_events and reception JSON, parses both,
    and extracts all fields into a Wave1Outputs object. Callers access
    whichever fields they need by name — no positional tuple ordering
    to remember.

    Returns a Wave1Outputs with all fields defaulting to None when the
    Wave 1 type wasn't generated or doesn't contain that field.
    """
    # Import here to avoid circular imports — schemas imports from this
    # module's sibling, not from inputs itself.
    from .schemas import PlotEventsOutput, ReceptionOutput

    result = Wave1Outputs()

    with sqlite3.connect(str(tracker_db_path)) as db:
        row = db.execute(
            "SELECT plot_events, reception FROM generated_metadata WHERE tmdb_id = ?",
            (tmdb_id,),
        ).fetchone()

    if row is None:
        return result

    # Extract plot_summary from plot_events output
    if row[0]:
        try:
            pe = PlotEventsOutput.model_validate_json(row[0])
            result.plot_summary = pe.plot_summary
        except Exception:
            pass

    # Extract all reception extraction-zone fields
    if row[1]:
        try:
            rec = ReceptionOutput.model_validate_json(row[1])
            result.thematic_observations = rec.thematic_observations
            result.emotional_observations = rec.emotional_observations
            result.craft_observations = rec.craft_observations
            result.source_material_hint = rec.source_material_hint
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Plot analysis output loading for downstream Wave 2 consumers
# ---------------------------------------------------------------------------

def load_plot_analysis_output(
    tmdb_id: int,
    tracker_db_path: Path = _DEFAULT_TRACKER_DB,
) -> PlotAnalysisWithJustificationsOutput | None:
    """Load the parsed plot_analysis output for a movie.

    Single DB query for the plot_analysis JSON column, parsed into the
    existing schema model. Returns None when plot_analysis wasn't
    generated or can't be parsed.
    """
    from .schemas import PlotAnalysisWithJustificationsOutput

    with sqlite3.connect(str(tracker_db_path)) as db:
        row = db.execute(
            "SELECT plot_analysis FROM generated_metadata WHERE tmdb_id = ?",
            (tmdb_id,),
        ).fetchone()

    if row is None or not row[0]:
        return None

    try:
        return PlotAnalysisWithJustificationsOutput.model_validate_json(row[0])
    except Exception:
        return None
