"""
Shared enums used across multiple modules in the codebase.
"""

from enum import StrEnum


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
