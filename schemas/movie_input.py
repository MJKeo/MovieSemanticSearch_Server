"""
Movie input data structure and loading utilities for the generation pipeline.

MovieInputData (dataclass):
    Raw fields loaded from tmdb_data + imdb_data tables. This is the
    contract between "load data from SQLite" and "run pre-consolidation."

load_movie_input_data(tmdb_ids, tracker_db_path) -> dict[int, MovieInputData]:
    Load MovieInputData for a set of movies from the ingestion tracker DB.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


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
    collection_name: str | None = None
    overview: str = ""
    genres: list[str] = field(default_factory=list)
    plot_synopses: list[str] = field(default_factory=list)
    plot_summaries: list[str] = field(default_factory=list)
    plot_keywords: list[str] = field(default_factory=list)
    overall_keywords: list[str] = field(default_factory=list)
    production_companies: list[str] = field(default_factory=list)
    # actors and characters preserve IMDB billing order. They are
    # positionally aligned in the common case (one role per actor);
    # for the rare multi-role actor they may drift slightly past the
    # top billed few. See top_billed_cast() for the paired view.
    actors: list[str] = field(default_factory=list)
    characters: list[str] = field(default_factory=list)
    # Each dict has keys: summary (str), text (str)
    featured_reviews: list[dict] = field(default_factory=list)
    reception_summary: str | None = None
    # Each dict has keys: name (str), sentiment (str)
    audience_reception_attributes: list[dict] = field(default_factory=list)
    maturity_rating: str = ""
    maturity_reasoning: list[str] = field(default_factory=list)
    # Each dict has keys: category (str), severity (str)
    parental_guide_items: list[dict] = field(default_factory=list)

    def title_with_year(self) -> str:
        """Format title as 'Title (Year)' for temporal grounding and
        disambiguation. Returns just 'Title' if year is unknown."""
        if self.release_year is not None:
            return f"{self.title} ({self.release_year})"
        return self.title

    def top_billed_cast(self, n: int = 5) -> str | None:
        """Format the top-n billed cast as 'Character (Actor), ...'.

        Used as a prompt input signal for concept tag classification
        (especially female_lead / ensemble_cast) and any other generator
        that benefits from knowing who the most prominent on-screen
        characters are. Billing order is preserved from IMDB.

        Pairing semantics: actors[i] is zipped with characters[i]. When
        a character name is missing (short characters list or empty
        entry), the actor is rendered alone. When an actor name is
        missing, the entry is skipped entirely — a nameless actor
        carries no signal.

        Returns None when no actors are available — callers should
        render this as "not available" in the prompt so the LLM sees
        explicit absence rather than silent omission.
        """
        if not self.actors:
            return None

        top_actors = self.actors[:n]
        top_characters = self.characters[:n]

        entries: list[str] = []
        for i, actor in enumerate(top_actors):
            actor = actor.strip()
            if not actor:
                continue
            character = top_characters[i].strip() if i < len(top_characters) else ""
            if character:
                entries.append(f"{character} ({actor})")
            else:
                entries.append(actor)

        return ", ".join(entries) if entries else None

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
        from movie_ingestion.metadata_generation.batch_generation.pre_consolidation import consolidate_maturity

        return consolidate_maturity(
            self.maturity_rating,
            self.maturity_reasoning,
            self.parental_guide_items,
        )


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
                t.collection_name,
                i.overview,
                i.maturity_rating,
                i.reception_summary,
                i.genres,
                i.synopses,
                i.plot_summaries,
                i.plot_keywords,
                i.overall_keywords,
                i.production_companies,
                i.actors,
                i.characters,
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
            collection_name=row["collection_name"],
            overview=row["overview"] or "",
            genres=_parse_json_list(row["genres"]),
            plot_synopses=_parse_json_list(row["synopses"]),
            plot_summaries=_parse_json_list(row["plot_summaries"]),
            plot_keywords=_parse_json_list(row["plot_keywords"]),
            overall_keywords=_parse_json_list(row["overall_keywords"]),
            production_companies=_parse_json_list(row["production_companies"]),
            actors=_parse_json_list(row["actors"]),
            characters=_parse_json_list(row["characters"]),
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
