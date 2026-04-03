"""
Generation-pipeline-specific input structures and utilities.

Data structures:
    - ConsolidatedInputs — post-pre-consolidation data passed to generators
    - SkipAssessment — which generations to run/skip and why
    - Wave1Outputs — parsed Wave 1 output fields for Wave 2 consumers

Utilities:
    - build_user_prompt — assemble user prompts from named fields
    - build_custom_id / parse_custom_id — batch request ID formatting
    - load_wave1_outputs — load Wave 1 outputs from tracker DB
    - load_plot_analysis_output — load plot analysis output from tracker DB

Constants:
    - WAVE1_TYPES, WAVE2_TYPES, ALL_GENERATION_TYPES — generation type sets
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import overload

from schemas.enums import MetadataType
from schemas.data_types import MultiLineList
from schemas.movie_input import MovieInputData


# ---------------------------------------------------------------------------
# Generation type sets
# ---------------------------------------------------------------------------

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

@overload
def build_custom_id(tmdb_id: int, metadata_type: MetadataType) -> str: ...

@overload
def build_custom_id(movie: MovieInputData, metadata_type: MetadataType) -> str: ...

def build_custom_id(tmdb_id_or_movie: int | MovieInputData, metadata_type: MetadataType) -> str:
    """Build a batch request custom_id in the format '{metadata_type}_{tmdb_id}'.

    Accepts either a raw tmdb_id (int) or a MovieInputData instance.
    Used by request building (to set custom_id in JSONL) and result
    processing (to map results back to movies). The format is
    deterministic so the same movie+type always produces the same ID.
    """
    tmdb_id = tmdb_id_or_movie.tmdb_id if isinstance(tmdb_id_or_movie, MovieInputData) else tmdb_id_or_movie
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
# Data structures (generation-pipeline-specific)
# ---------------------------------------------------------------------------

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


# Default paths match the ingestion pipeline conventions
_DEFAULT_TRACKER_DB = Path("ingestion_data/tracker.db")


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
    from schemas.metadata import PlotEventsOutput, ReceptionOutput

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
):
    """Load the parsed plot_analysis output for a movie.

    Single DB query for the plot_analysis JSON column, parsed into the
    existing schema model. Returns None when plot_analysis wasn't
    generated or can't be parsed.
    """
    from schemas.metadata import PlotAnalysisOutput

    with sqlite3.connect(str(tracker_db_path)) as db:
        row = db.execute(
            "SELECT plot_analysis FROM generated_metadata WHERE tmdb_id = ?",
            (tmdb_id,),
        ).fetchone()

    if row is None or not row[0]:
        return None

    try:
        return PlotAnalysisOutput.model_validate_json(row[0])
    except Exception:
        return None
