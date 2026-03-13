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
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Generation type constants
# ---------------------------------------------------------------------------

WAVE1_TYPES = frozenset({"plot_events", "reception"})
WAVE2_TYPES = frozenset({
    "plot_analysis",
    "viewer_experience",
    "watch_context",
    "narrative_techniques",
    "production_keywords",
    "source_of_inspiration",
})
ALL_GENERATION_TYPES = WAVE1_TYPES | WAVE2_TYPES


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

    def batch_id(self, generation_type: str) -> str:
        """Produce a batch request custom_id, e.g. '12345-plot_events'.

        Must be unique per request in a batch — guaranteed by the
        (tmdb_id, generation_type) pair being unique per movie.
        """
        return f"{self.tmdb_id}-{generation_type}"

    def title_with_year(self) -> str:
        """Format title as 'Title (Year)' for temporal grounding and
        disambiguation. Returns just 'Title' if year is unknown."""
        if self.release_year is not None:
            return f"{self.title} ({self.release_year})"
        return self.title


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
        # Convert lists to comma-separated strings
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        lines.append(f"{key}: {value}")
    return "\n".join(lines)
