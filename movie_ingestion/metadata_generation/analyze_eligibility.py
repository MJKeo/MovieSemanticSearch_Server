"""
Temporary analysis script: checks eligibility and estimates input token
counts for each LLM metadata generation group across all imdb_quality_passed movies.

Outputs a JSON report to ingestion_data/eligibility_report.json with:
  - Per-group: eligible count, average input token size
  - Count of movies not eligible for ANY group

Usage:
    python -m movie_ingestion.metadata_generation.analyze_eligibility
"""

import json
import sqlite3
from pathlib import Path

import orjson
import tiktoken

from movie_ingestion.tracker import (
    TRACKER_DB_PATH,
    IMDB_DATA_COLUMNS,
    IMDB_JSON_COLUMNS,
    deserialize_imdb_row,
)
from movie_ingestion.metadata_generation.inputs import MovieInputData
from movie_ingestion.metadata_generation.schemas import (
    PlotEventsOutput,
    ReceptionOutput,
    MajorCharacter,
)
from movie_ingestion.metadata_generation.pre_consolidation import (
    assess_skip_conditions,
    route_keywords,
    consolidate_maturity,
)
from implementation.prompts.vector_metadata_generation_prompts import (
    PLOT_EVENTS_SYSTEM_PROMPT,
    PLOT_ANALYSIS_SYSTEM_PROMPT,
    VIEWER_EXPERIENCE_SYSTEM_PROMPT,
    WATCH_CONTEXT_SYSTEM_PROMPT,
    NARRATIVE_TECHNIQUES_SYSTEM_PROMPT,
    PRODUCTION_KEYWORDS_SYSTEM_PROMPT,
    SOURCE_OF_INSPIRATION_SYSTEM_PROMPT,
    RECEPTION_SYSTEM_PROMPT,
)

# tiktoken encoder — used only for the one-time system prompt token counts
ENC = tiktoken.get_encoding("o200k_base")

# Fixed system prompt token counts (computed once at import time)
SYSTEM_PROMPT_TOKENS: dict[str, int] = {
    "plot_events": len(ENC.encode(PLOT_EVENTS_SYSTEM_PROMPT)),
    "reception": len(ENC.encode(RECEPTION_SYSTEM_PROMPT)),
    "plot_analysis": len(ENC.encode(PLOT_ANALYSIS_SYSTEM_PROMPT)),
    "viewer_experience": len(ENC.encode(VIEWER_EXPERIENCE_SYSTEM_PROMPT)),
    "watch_context": len(ENC.encode(WATCH_CONTEXT_SYSTEM_PROMPT)),
    "narrative_techniques": len(ENC.encode(NARRATIVE_TECHNIQUES_SYSTEM_PROMPT)),
    "production_keywords": len(ENC.encode(PRODUCTION_KEYWORDS_SYSTEM_PROMPT)),
    "source_of_inspiration": len(ENC.encode(SOURCE_OF_INSPIRATION_SYSTEM_PROMPT)),
}


def estimate_tokens(text: str) -> int:
    """Approximate token count using character length.

    For English text with o200k_base encoding, ~3.7 characters per token
    is a reliable approximation. Avoids the per-call cost of BPE encoding,
    which was the main bottleneck when called ~800K times across all movies.
    """
    return len(text) * 1000 // 3700


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_movies(db: sqlite3.Connection) -> list[dict]:
    """Load all imdb_quality_passed movies with tmdb_data + imdb_data."""
    db.row_factory = sqlite3.Row
    imdb_cols = ", ".join(f"id.{c}" for c in IMDB_DATA_COLUMNS)
    query = f"""
        SELECT td.tmdb_id, td.title, td.release_date, td.overview,
               {imdb_cols}
        FROM movie_progress mp
        JOIN tmdb_data td ON td.tmdb_id = mp.tmdb_id
        JOIN imdb_data id ON id.tmdb_id = mp.tmdb_id
        WHERE mp.status = 'imdb_quality_passed'
    """
    rows = db.execute(query).fetchall()
    return [dict(r) for r in rows]


def row_to_movie_input(row: dict) -> MovieInputData:
    """Convert a joined DB row into MovieInputData."""
    # Deserialize JSON columns from imdb_data
    for col in IMDB_JSON_COLUMNS:
        val = row.get(col)
        if val is not None and isinstance(val, str):
            row[col] = orjson.loads(val)
        elif val is None:
            row[col] = []

    # Extract release year from tmdb release_date
    release_year = None
    rd = row.get("release_date")
    if rd and len(rd) >= 4:
        try:
            release_year = int(rd[:4])
        except ValueError:
            pass

    return MovieInputData(
        tmdb_id=row["tmdb_id"],
        title=row.get("title") or "",
        release_year=release_year,
        overview=row.get("overview") or "",
        genres=row.get("genres") or [],
        plot_synopses=row.get("synopses") or [],
        plot_summaries=row.get("plot_summaries") or [],
        plot_keywords=row.get("plot_keywords") or [],
        overall_keywords=row.get("overall_keywords") or [],
        featured_reviews=row.get("featured_reviews") or [],
        reception_summary=row.get("reception_summary"),
        audience_reception_attributes=row.get("review_themes") or [],
        maturity_rating=row.get("maturity_rating") or "",
        maturity_reasoning=row.get("maturity_reasoning") or [],
        parental_guide_items=row.get("parental_guide_items") or [],
    )


# ---------------------------------------------------------------------------
# User prompt builders (mirror the existing prompt construction logic)
# ---------------------------------------------------------------------------

def build_plot_events_prompt(mi: MovieInputData, title_with_year: str) -> str:
    """Build the user prompt for plot_events."""
    parts = []
    parts.append(f"title: {title_with_year}")
    if mi.overview:
        parts.append(f"overview: {mi.overview}")
    if mi.plot_summaries:
        parts.append(f"plot_summaries: \n-" + "\n-".join(mi.plot_summaries))
    if mi.plot_synopses:
        parts.append(f"plot_synopses: \n-" + "\n-".join(mi.plot_synopses))
    if mi.plot_keywords:
        parts.append(f"plot_keywords: {', '.join(mi.plot_keywords)}")
    return "\n".join(parts)


def build_reception_prompt(mi: MovieInputData, title_with_year: str) -> str:
    """Build the user prompt for reception."""
    parts = []
    parts.append(f"title: {title_with_year}")
    if mi.reception_summary:
        parts.append(f"reception_summary: {mi.reception_summary}")
    if mi.audience_reception_attributes:
        converted = [str(a) for a in mi.audience_reception_attributes]
        parts.append(f"audience_reception_attributes: \n -" + ". ".join(converted))
    if mi.featured_reviews:
        formatted = [
            f"{r['summary']}: {r['text']}" if isinstance(r, dict) else str(r)
            for r in mi.featured_reviews[:5]
        ]
        parts.append(f"featured_reviews: \n -" + "\n -".join(formatted))
    return "\n".join(parts)


def build_plot_analysis_prompt(
    mi: MovieInputData, title_with_year: str, plot_synopsis: str
) -> str:
    """Build the user prompt for plot_analysis."""
    parts = []
    parts.append(f"title: {title_with_year}")
    if mi.genres:
        parts.append(f"genres: {', '.join(mi.genres)}")
    if mi.overview:
        parts.append(f"overview: {mi.overview}")
    if plot_synopsis:
        parts.append(f"plot_synopsis: {plot_synopsis}")
    if mi.plot_keywords:
        parts.append(f"plot_keywords: {', '.join(mi.plot_keywords)}")
    if mi.reception_summary:
        parts.append(f"reception_summary: {mi.reception_summary}")
    if mi.featured_reviews:
        formatted = [
            f"{r['summary']}: {r['text']}" if isinstance(r, dict) else str(r)
            for r in mi.featured_reviews[:5]
        ]
        parts.append(f"featured_reviews: \n -" + "\n -".join(formatted))
    return "\n".join(parts)


def build_viewer_experience_prompt(
    mi: MovieInputData, title_with_year: str, plot_synopsis: str
) -> str:
    """Build the user prompt for viewer_experience."""
    parts = []
    parts.append(f"title: {title_with_year}")
    if mi.genres:
        parts.append(f"genres: {', '.join(mi.genres)}")
    if plot_synopsis:
        parts.append(f"plot_synopsis: {plot_synopsis}")
    if mi.plot_keywords:
        parts.append(f"plot_keywords: {', '.join(mi.plot_keywords)}")
    if mi.overall_keywords:
        parts.append(f"overall_keywords: {', '.join(mi.overall_keywords)}")
    if mi.maturity_rating:
        parts.append(f"maturity_rating: {mi.maturity_rating}")
    if mi.maturity_reasoning:
        parts.append(f"maturity_reasoning: {', '.join(mi.maturity_reasoning)}")
    if mi.parental_guide_items:
        items = [
            f"{i['category']}: {i['severity']}" if isinstance(i, dict) else str(i)
            for i in mi.parental_guide_items
        ]
        parts.append(f"parental_guide_items: {', '.join(items)}")
    if mi.reception_summary:
        parts.append(f"reception_summary: {mi.reception_summary}")
    if mi.audience_reception_attributes:
        converted = [str(a) for a in mi.audience_reception_attributes]
        parts.append(f"audience_reception_attributes: \n -" + ". ".join(converted))
    if mi.featured_reviews:
        formatted = [
            f"{r['summary']}: {r['text']}" if isinstance(r, dict) else str(r)
            for r in mi.featured_reviews[:5]
        ]
        parts.append(f"featured_reviews: \n -" + "\n -".join(formatted))
    return "\n".join(parts)


def build_watch_context_prompt(mi: MovieInputData, title_with_year: str) -> str:
    """Build the user prompt for watch_context."""
    parts = []
    parts.append(f"title: {title_with_year}")
    if mi.genres:
        parts.append(f"genres: {', '.join(mi.genres)}")
    if mi.overview:
        parts.append(f"overview: {mi.overview}")
    if mi.plot_keywords:
        parts.append(f"plot_keywords: {', '.join(mi.plot_keywords)}")
    if mi.overall_keywords:
        parts.append(f"overall_keywords: {', '.join(mi.overall_keywords)}")
    if mi.reception_summary:
        parts.append(f"reception_summary: {mi.reception_summary}")
    if mi.audience_reception_attributes:
        converted = [str(a) for a in mi.audience_reception_attributes]
        parts.append(f"audience_reception_attributes: \n -" + ". ".join(converted))
    if mi.featured_reviews:
        formatted = [
            f"{r['summary']}: {r['text']}" if isinstance(r, dict) else str(r)
            for r in mi.featured_reviews[:5]
        ]
        parts.append(f"featured_reviews: \n -" + "\n -".join(formatted))
    return "\n".join(parts)


def build_narrative_techniques_prompt(
    mi: MovieInputData, title_with_year: str, plot_synopsis: str
) -> str:
    """Build the user prompt for narrative_techniques."""
    parts = []
    parts.append(f"title: {title_with_year}")
    if plot_synopsis:
        parts.append(f"plot_synopsis: {plot_synopsis}")
    if mi.plot_keywords:
        parts.append(f"plot_keywords: {', '.join(mi.plot_keywords)}")
    if mi.overall_keywords:
        parts.append(f"overall_keywords: {', '.join(mi.overall_keywords)}")
    if mi.reception_summary:
        parts.append(f"reception_summary: {mi.reception_summary}")
    if mi.featured_reviews:
        formatted = [
            f"{r['summary']}: {r['text']}" if isinstance(r, dict) else str(r)
            for r in mi.featured_reviews[:5]
        ]
        parts.append(f"featured_reviews: \n -" + "\n -".join(formatted))
    return "\n".join(parts)


def build_production_keywords_prompt(
    mi: MovieInputData, title_with_year: str
) -> str:
    """Build the user prompt for production_keywords."""
    parts = []
    parts.append(f"title: {title_with_year}")
    if mi.overall_keywords:
        parts.append(f"keywords: {', '.join(mi.overall_keywords)}")
    return "\n".join(parts)


def build_source_of_inspiration_prompt(
    mi: MovieInputData, title_with_year: str, plot_synopsis: str,
    merged_keywords: list[str]
) -> str:
    """Build the user prompt for source_of_inspiration."""
    parts = []
    parts.append(f"title: {title_with_year}")
    if merged_keywords:
        parts.append(f"keywords: {', '.join(merged_keywords)}")
    if plot_synopsis:
        parts.append(f"plot_synopsis: {plot_synopsis}")
    if mi.featured_reviews:
        formatted = [
            f"{r['summary']}: {r['text']}" if isinstance(r, dict) else str(r)
            for r in mi.featured_reviews[:5]
        ]
        parts.append(f"featured_reviews: \n -" + "\n -".join(formatted))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Prompt builder dispatch
# ---------------------------------------------------------------------------

# Maps generation type -> (prompt_builder_fn, needs_plot_synopsis, needs_merged_kw)
PROMPT_BUILDERS = {
    "plot_events": (build_plot_events_prompt, False, False),
    "reception": (build_reception_prompt, False, False),
    "plot_analysis": (build_plot_analysis_prompt, True, False),
    "viewer_experience": (build_viewer_experience_prompt, True, False),
    "watch_context": (build_watch_context_prompt, False, False),
    "narrative_techniques": (build_narrative_techniques_prompt, True, False),
    "production_keywords": (build_production_keywords_prompt, False, False),
    "source_of_inspiration": (build_source_of_inspiration_prompt, True, True),
}


def count_input_tokens(gen_type: str, mi: MovieInputData, title_with_year: str,
                       plot_synopsis: str, merged_keywords: list[str]) -> int:
    """Count system + user prompt tokens for a given generation type."""
    builder_fn, needs_plot, needs_merged = PROMPT_BUILDERS[gen_type]

    # Build the appropriate function call
    if needs_plot and needs_merged:
        user_prompt = builder_fn(mi, title_with_year, plot_synopsis, merged_keywords)
    elif needs_plot:
        user_prompt = builder_fn(mi, title_with_year, plot_synopsis)
    else:
        user_prompt = builder_fn(mi, title_with_year)

    user_tokens = estimate_tokens(user_prompt)
    system_tokens = SYSTEM_PROMPT_TOKENS[gen_type]
    return system_tokens + user_tokens


# ---------------------------------------------------------------------------
# Synthetic plot synopsis for Wave 2 assessment
# ---------------------------------------------------------------------------

def synthesize_plot_synopsis(mi: MovieInputData) -> str | None:
    """Pick the best available text to simulate what plot_events would produce.

    Priority: longest synopsis > longest summary > overview.
    Returns None if nothing substantial is available.
    """
    candidates = []
    if mi.plot_synopses:
        candidates.append(max(mi.plot_synopses, key=len))
    if mi.plot_summaries:
        candidates.append(max(mi.plot_summaries, key=len))
    if mi.overview:
        candidates.append(mi.overview)

    if not candidates:
        return None
    return max(candidates, key=len)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    print("Connecting to tracker DB...")
    db = sqlite3.connect(str(TRACKER_DB_PATH))

    print("Loading imdb_quality_passed movies...")
    rows = load_movies(db)
    db.close()
    print(f"Loaded {len(rows):,} movies")

    # Accumulators per generation type
    ALL_TYPES = [
        "plot_events", "reception",
        "plot_analysis", "viewer_experience", "watch_context",
        "narrative_techniques", "production_keywords", "source_of_inspiration",
    ]
    eligible_counts: dict[str, int] = {t: 0 for t in ALL_TYPES}
    token_sums: dict[str, int] = {t: 0 for t in ALL_TYPES}
    ineligible_for_all = 0

    for i, row in enumerate(rows):
        if (i + 1) % 10000 == 0:
            print(f"  Processing {i + 1:,} / {len(rows):,}...")

        mi = row_to_movie_input(row)
        title_with_year = mi.title_with_year()

        # Keyword routing
        _plot_kw, _overall_kw, merged_keywords = route_keywords(
            mi.plot_keywords, mi.overall_keywords
        )

        # Maturity consolidation (needed for Wave 2 skip assessment)
        maturity_summary = consolidate_maturity(
            mi.maturity_rating, mi.maturity_reasoning, mi.parental_guide_items
        )

        # Wave 1 assessment
        wave1_assessment = assess_skip_conditions(mi)

        # For Wave 2 assessment, simulate typed outputs:
        # - Use best available text as proxy for plot_synopsis
        # - Assume reception produces review_insights_brief if it was eligible
        plot_synopsis = synthesize_plot_synopsis(mi)
        reception_eligible = "reception" in wave1_assessment.generations_to_run

        # Build synthetic typed outputs for Wave 2 assessment
        plot_events_output = None
        if plot_synopsis is not None:
            plot_events_output = PlotEventsOutput(
                plot_summary=plot_synopsis,
                setting="unknown",
                major_characters=[],
            )

        reception_output = None
        if reception_eligible:
            reception_output = ReceptionOutput(
                reception_summary="placeholder",
                review_insights_brief="placeholder",
            )

        wave2_assessment = assess_skip_conditions(
            mi,
            plot_events_output=plot_events_output,
            reception_output=reception_output,
            merged_keywords=merged_keywords,
            maturity_summary=maturity_summary,
        )

        # Combine both waves into a single eligible set
        all_eligible = wave1_assessment.generations_to_run | wave2_assessment.generations_to_run

        if not all_eligible:
            ineligible_for_all += 1
            continue

        # Count tokens for each eligible generation
        for gen_type in all_eligible:
            eligible_counts[gen_type] += 1
            tokens = count_input_tokens(
                gen_type, mi, title_with_year,
                plot_synopsis or "", merged_keywords
            )
            token_sums[gen_type] += tokens

    # Build report
    print("\nBuilding report...")
    group_results = {}
    for gen_type in ALL_TYPES:
        count = eligible_counts[gen_type]
        avg_tokens = token_sums[gen_type] / count if count > 0 else 0
        group_results[gen_type] = {
            "eligible_count": count,
            "average_input_tokens": round(avg_tokens, 1),
            "system_prompt_tokens": SYSTEM_PROMPT_TOKENS[gen_type],
        }

    report = {
        "total_movies": len(rows),
        "ineligible_for_all_groups": ineligible_for_all,
        "groups": group_results,
        "notes": {
            "wave2_assumption": (
                "Wave 2 eligibility assumes plot_events succeeds. "
                "plot_synopsis is proxied by the longest available "
                "synopsis/summary/overview for each movie."
            ),
            "token_counting": (
                "System prompt tokens use exact o200k_base encoding. "
                "User prompt tokens use character-length estimation "
                "(~3.7 chars/token). Counts include system + user prompt. "
                "JSON schema / response_format tokens are NOT included."
            ),
        },
    }

    out_path = Path("ingestion_data/eligibility_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary to console
    print(f"\nTotal movies: {len(rows):,}")
    print(f"Ineligible for ALL groups: {ineligible_for_all:,}")
    print(f"\n{'Group':<25} {'Eligible':>10} {'Avg Tokens':>12}")
    print("-" * 50)
    for gen_type in ALL_TYPES:
        g = group_results[gen_type]
        print(f"{gen_type:<25} {g['eligible_count']:>10,} {g['average_input_tokens']:>12.1f}")

    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
