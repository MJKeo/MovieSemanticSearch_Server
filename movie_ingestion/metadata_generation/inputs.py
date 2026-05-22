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
    - load_narrative_techniques_output — load NT output from tracker DB
    - load_viewer_experience_output — load VE output from tracker DB
    - extract_narrative_technique_terms — strip justifications from 7 NT
      sections used by concept_tags

Constants:
    - WAVE1_TYPES, WAVE_INDEPENDENT_TYPES, WAVE2_TYPES, ALL_GENERATION_TYPES
      — generation type sets
"""

from __future__ import annotations

import json
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
WAVE_INDEPENDENT_TYPES = frozenset({MetadataType.FRANCHISE})
WAVE2_TYPES = frozenset({
    MetadataType.PLOT_ANALYSIS,
    MetadataType.VIEWER_EXPERIENCE,
    MetadataType.WATCH_CONTEXT,
    MetadataType.NARRATIVE_TECHNIQUES,
    MetadataType.PRODUCTION_KEYWORDS,
    MetadataType.PRODUCTION_TECHNIQUES,
    MetadataType.SOURCE_OF_INSPIRATION,
    MetadataType.SOURCE_MATERIAL_V2,
    MetadataType.CONCEPT_TAGS,
})
ALL_GENERATION_TYPES = WAVE1_TYPES | WAVE_INDEPENDENT_TYPES | WAVE2_TYPES


# ---------------------------------------------------------------------------
# Batch custom_id helpers
# ---------------------------------------------------------------------------

@overload
def build_custom_id(tmdb_id: int, metadata_type: MetadataType, run_index: int | None = None) -> str: ...

@overload
def build_custom_id(movie: MovieInputData, metadata_type: MetadataType, run_index: int | None = None) -> str: ...

def build_custom_id(
    tmdb_id_or_movie: int | MovieInputData,
    metadata_type: MetadataType,
    run_index: int | None = None,
) -> str:
    """Build a batch request custom_id.

    Default format: '{metadata_type}_{tmdb_id}' (single-run types).
    Multi-run format: '{metadata_type}_{tmdb_id}_r{N}' when run_index is set
    (e.g. concept_tags emits 3 requests per movie with run_index 1..3 so
    OpenAI's per-batch custom_id uniqueness requirement is met).

    Used by request building (to set custom_id in JSONL) and result
    processing (to map results back to movies). The format is
    deterministic so the same (movie, type, run_index) always produces
    the same ID.
    """
    tmdb_id = tmdb_id_or_movie.tmdb_id if isinstance(tmdb_id_or_movie, MovieInputData) else tmdb_id_or_movie
    if run_index is None:
        return f"{metadata_type}_{tmdb_id}"
    return f"{metadata_type}_{tmdb_id}_r{run_index}"


def parse_custom_id(custom_id: str) -> tuple[MetadataType, int, int | None]:
    """Parse a batch custom_id back into (metadata_type, tmdb_id, run_index).

    Inverse of build_custom_id(). Accepts both single-run format
    ('plot_events_12345' → run_index=None) and multi-run format
    ('concept_tags_12345_r2' → run_index=2).

    The multi-run suffix is detected as a trailing token matching '_r<int>'.
    The remaining string is split on the last underscore to separate the
    metadata type from the tmdb_id — safe because tmdb_id is always a pure
    integer (no underscores).

    Raises ValueError if the metadata_type portion is not a valid MetadataType.
    """
    # Detect optional '_r<N>' suffix.
    run_index: int | None = None
    head = custom_id
    if "_r" in custom_id:
        prefix, _, maybe_run = custom_id.rpartition("_r")
        if maybe_run.isdigit():
            run_index = int(maybe_run)
            head = prefix

    type_str, tmdb_id_str = head.rsplit("_", 1)
    metadata_type = MetadataType(type_str)  # raises ValueError if invalid
    return metadata_type, int(tmdb_id_str), run_index


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

    Note: craft_observations is loaded by load_wave1_outputs and consumed
    by concept_tags (where reviewer craft language is the primary signal
    for NONLINEAR_TIMELINE, PLOT_TWIST nuance, UNRELIABLE_NARRATOR, and
    BREAKING_FOURTH_WALL).
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

    Routes the raw JSON through normalize_legacy_metadata_payload so any
    pre-existing schema migration applies to PlotAnalysisOutput too. No
    PlotAnalysisOutput-specific legacy shape exists today, so this is a
    no-op for current data — kept for parity with the NT loader and to
    stay safe against future renames.
    """
    from schemas.metadata import PlotAnalysisOutput, normalize_legacy_metadata_payload

    with sqlite3.connect(str(tracker_db_path)) as db:
        row = db.execute(
            "SELECT plot_analysis FROM generated_metadata WHERE tmdb_id = ?",
            (tmdb_id,),
        ).fetchone()

    if row is None or not row[0]:
        return None

    try:
        payload = json.loads(row[0])
        normalized = normalize_legacy_metadata_payload(payload, PlotAnalysisOutput)
        return PlotAnalysisOutput.model_validate(normalized)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Narrative techniques output loading for downstream Wave 2 consumers
# ---------------------------------------------------------------------------

def load_narrative_techniques_output(
    tmdb_id: int,
    tracker_db_path: Path = _DEFAULT_TRACKER_DB,
):
    """Load the parsed narrative_techniques output for a movie.

    Single DB query for the narrative_techniques JSON column, parsed into
    the existing schema model. Returns None when narrative_techniques wasn't
    generated or can't be parsed.

    Routes the raw JSON through normalize_legacy_metadata_payload so older
    rows that still carry the "justification" key (renamed to
    "evidence_basis" in a later schema change) are rewritten in-memory
    before validation. Without this, every legacy row silently fails to
    parse and concept_tags loses its NT input signal.
    """
    from schemas.metadata import NarrativeTechniquesOutput, normalize_legacy_metadata_payload

    with sqlite3.connect(str(tracker_db_path)) as db:
        row = db.execute(
            "SELECT narrative_techniques FROM generated_metadata WHERE tmdb_id = ?",
            (tmdb_id,),
        ).fetchone()

    if row is None or not row[0]:
        return None

    try:
        payload = json.loads(row[0])
        normalized = normalize_legacy_metadata_payload(payload, NarrativeTechniquesOutput)
        return NarrativeTechniquesOutput.model_validate(normalized)
    except Exception:
        return None


def extract_narrative_technique_terms(
    nt,
) -> dict[str, list[str]]:
    """Extract terms (no justifications) from the 5 NT sections concept tags need.

    Returns a dict mapping section name -> list of term strings. Sections
    with empty term lists are included with empty lists so the caller can
    distinguish "section had no terms" from "section wasn't available."

    The `character_arcs` and `audience_character_perception` sections
    are deliberately EXCLUDED. Three-way eval (BASE / REV / RSN) showed
    they were the primary upstream-label contamination driving ANTI_HERO
    false positives: the upstream literally emits "antihero maturation
    arc" / "sympathetic antihero" as terms, and the consumer model
    rubber-stamped those labels instead of deriving from raw behavior.
    ANTI_HERO classification is now derived from plot_summary +
    character_arc_labels (PlotAnalysis thematic transformations) +
    conflict_type, which describe what the protagonist actually does
    without using tag-shaped words.

    The 2 still-excluded sections (characterization_methods,
    conflict_stakes_design) don't carry signal for any of the 25 concept
    tags as of this baseline.

    Args:
        nt: A NarrativeTechniquesOutput instance. Type hint omitted to
            avoid circular import (schemas.metadata imports schemas.enums).
    """
    return {
        "narrative_archetype": list(nt.narrative_archetype.terms),
        "narrative_delivery": list(nt.narrative_delivery.terms),
        "pov_perspective": list(nt.pov_perspective.terms),
        "information_control": list(nt.information_control.terms),
        "additional_narrative_devices": list(nt.additional_narrative_devices.terms),
    }


# ---------------------------------------------------------------------------
# Viewer experience output loading for downstream Wave 2 consumers
# ---------------------------------------------------------------------------

def load_viewer_experience_output(
    tmdb_id: int,
    tracker_db_path: Path = _DEFAULT_TRACKER_DB,
):
    """Load the parsed viewer_experience output for a movie.

    Single DB query for the viewer_experience JSON column, parsed into
    the existing schema model. Returns None when viewer_experience wasn't
    generated or can't be parsed.

    Concept_tags previously used this loader to read `ending_aftertaste`,
    but that input has been removed: the three-way eval showed it was
    the primary driver of BITTERSWEET_ENDING over-tagging because the
    upstream generator literally emits "bittersweet" as a term for any
    ending with permanent loss, and the downstream consumer rubber-
    stamped the label. Ending classification now derives from
    emotional_observations + plot_summary closing-scene events. The
    loader is kept available for other potential consumers of the
    ViewerExperience output.
    """
    from schemas.metadata import ViewerExperienceOutput, normalize_legacy_metadata_payload

    with sqlite3.connect(str(tracker_db_path)) as db:
        row = db.execute(
            "SELECT viewer_experience FROM generated_metadata WHERE tmdb_id = ?",
            (tmdb_id,),
        ).fetchone()

    if row is None or not row[0]:
        return None

    try:
        payload = json.loads(row[0])
        normalized = normalize_legacy_metadata_payload(payload, ViewerExperienceOutput)
        return ViewerExperienceOutput.model_validate(normalized)
    except Exception:
        return None
