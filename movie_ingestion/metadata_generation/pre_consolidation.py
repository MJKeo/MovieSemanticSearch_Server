"""
Pre-consolidation phase: pure data processing before any LLM calls.

Three functions that transform raw movie data into optimized LLM inputs.
No LLM cost — all logic is deterministic and testable without mocking.

1. route_keywords(plot_keywords, overall_keywords) -> KeywordRouting:
    Normalizes (lowercase + strip) and deduplicates all keywords, then
    produces three keyword lists per the routing table:
    - plot_keywords (list[str]): passed to plot_events, plot_analysis
    - overall_keywords (list[str]): passed to watch_context, narrative_techniques
    - merged_keywords (list[str]): deduplicated union for viewer_experience,
      production_keywords, source_of_inspiration
    Implementation: normalize each keyword, then merged = list(dict.fromkeys(plot + overall))

2. consolidate_maturity(rating, reasoning, parental_guide_items) -> str | None:
    Produces a single maturity_summary string with priority chain:
    - If maturity_reasoning exists (>=1 item): join reasoning items
      (reasoning already contains the full rating context)
    - If absent, falls back to parental_guide_items:
      "R — severe violence, moderate language, mild nudity"
    - If no reasoning/items but rating matches MPAA: "R — Restricted"
    - If nothing useful: returns None

3. Per-generation eligibility checks (_check_<type> methods):
    Eight individual methods, one per generation type, each returning
    str | None (None = eligible, str = skip reason). These are composed
    by assess_skip_conditions() into a SkipAssessment.

    Wave 1 (need only movie_input):
        check_plot_events, check_reception
    Wave 2 (need Wave 1 outputs + derived data):
        _check_plot_analysis, _check_viewer_experience, _check_watch_context,
        _check_narrative_techniques, _check_production_keywords,
        _check_source_of_inspiration

4. assess_skip_conditions(movie_input, ...) -> SkipAssessment:
    Thin orchestrator that calls the relevant per-generation checks
    and assembles a SkipAssessment. Called twice in the pipeline:
    - Before Wave 1 (all kwargs None): runs Wave 1 checks
    - Before Wave 2 (outputs provided): runs Wave 2 checks

See docs/llm_metadata_generation_new_flow.md Section 3 (Pre-Consolidation)
and Section 6 (Sparse Movie Skip Conditions) for the full specification.
"""

from __future__ import annotations

from .inputs import (
    MovieInputData,
    ConsolidatedInputs,
    SkipAssessment,
)
from .schemas import PlotEventsOutput, ReceptionOutput


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Official MPAA rating definitions used as fallback when we have a rating
# but no reasoning or parental guide items to provide more detail.
MPAA_DEFINITIONS: dict[str, str] = {
    "G": "General Audiences",
    "PG": "Parental Guidance Suggested",
    "PG-13": "Parents Strongly Cautioned",
    "R": "Restricted",
    "NC-17": "Adults Only",
}

# Minimum length (chars) for the longest plot text source to qualify
# for plot_events generation. Measured across the first synopsis entry
# and all plot_summaries entries.
_MIN_PLOT_TEXT_CHARS = 600

_MIN_REVIEWS_CHARS = 25


# ---------------------------------------------------------------------------
# 1. Keyword routing
# ---------------------------------------------------------------------------

def route_keywords(
    plot_keywords: list[str],
    overall_keywords: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Normalize and deduplicate keywords, then produce three routed lists.

    Normalization: each keyword is lowercased and stripped of whitespace
    before deduplication. This ensures consistent matching downstream.

    Returns (plot_keywords, overall_keywords, merged_keywords) where
    merged_keywords is a deduplicated union with plot_keywords first,
    then unique overall_keywords appended. Order-preserving dedup
    via dict.fromkeys().

    Routing per generation:
        plot_keywords only   -> plot_events, plot_analysis
        overall_keywords only -> watch_context, narrative_techniques
        merged_keywords      -> viewer_experience, production_keywords,
                                source_of_inspiration
    """
    normalized_plot = list(dict.fromkeys(
        kw.lower().strip() for kw in plot_keywords
    ))
    normalized_overall = list(dict.fromkeys(
        kw.lower().strip() for kw in overall_keywords
    ))
    merged = list(dict.fromkeys(normalized_plot + normalized_overall))
    return normalized_plot, normalized_overall, merged


# ---------------------------------------------------------------------------
# 2. Maturity consolidation
# ---------------------------------------------------------------------------

def consolidate_maturity(
    rating: str,
    reasoning: list[str],
    parental_guide_items: list[dict],
) -> str | None:
    """Produce a single maturity_summary string from available maturity data.

    Priority chain:
    1. reasoning exists (>=1 item): join reasoning items directly.
       Reasoning already contains the full rating context
       (e.g. "Rated R for violence and language"), so no prefix needed.
    2. parental_guide_items exist: format as "R — severe violence, ..."
       with the rating prefix since items don't include the rating.
    3. rating matches a known MPAA rating: return "R — Restricted" etc.
    4. nothing useful: return None.
    """
    # Case 1: reasoning provides the richest context
    if reasoning:
        return ", ".join(reasoning)

    # Case 2: parental guide items as fallback with severity + category
    if parental_guide_items:
        detail = ", ".join(
            f"{item['severity']} {item['category']}"
            for item in parental_guide_items
        )
        if rating:
            return f"{rating} — {detail}"
        return detail

    # Case 3: rating-only fallback using MPAA definitions
    if rating and rating in MPAA_DEFINITIONS:
        return f"{rating} — {MPAA_DEFINITIONS[rating]}"

    # Case 4: no maturity data at all
    return None


# ---------------------------------------------------------------------------
# 3. Per-generation eligibility checks
# ---------------------------------------------------------------------------
# Each _check_<type> method returns str | None:
#   None  = eligible (generation should run)
#   str   = skip reason (generation should be skipped)

def check_plot_events(movie_input: MovieInputData) -> str | None:
    """Plot events requires substantial plot text from synopses or summaries.

    Eligible only if the longest text among the first synopsis entry
    (if any) and all plot_summaries entries (if any) is >= 600 chars.
    Overview is deliberately excluded — it's too short to anchor
    plot event extraction on its own.

    Returns None if eligible, or a skip reason string if not.
    """
    # Collect candidate texts: first synopsis + all summaries
    candidates: list[str] = []
    if movie_input.plot_synopses:
        candidates.append(movie_input.plot_synopses[0])
    candidates.extend(movie_input.plot_summaries)

    longest = max((len(t) for t in candidates), default=0)

    if longest >= _MIN_PLOT_TEXT_CHARS:
        return None

    return (
        f"Longest plot text source is {longest} chars "
        f"(need >={_MIN_PLOT_TEXT_CHARS}); skipping plot_events"
    )


def check_reception(movie_input: MovieInputData) -> str | None:
    """Reception requires at least one reception signal with enough substance.

    Without any, the model would fabricate reception from parametric
    knowledge alone. Reviews must also meet a minimum combined length
    threshold to ensure there's enough text for meaningful extraction.
    """
    has_summary = movie_input.reception_summary is not None
    has_attributes = len(movie_input.audience_reception_attributes) >= 1

    # Reviews must have enough combined text to be useful
    combined_review_len = sum(
        len(r.get("text", "")) for r in movie_input.featured_reviews
    )
    has_reviews = combined_review_len >= _MIN_REVIEWS_CHARS

    if has_reviews or has_summary or has_attributes:
        return None

    return (
        "No reception summary, no audience reception attributes, "
        "and insufficient review text (combined <25 chars)"
    )


def _check_plot_analysis(
    plot_synopsis: str | None,
    review_insights_brief: str | None,
) -> str | None:
    """Plot analysis requires plot_synopsis OR review_insights_brief."""
    if plot_synopsis or review_insights_brief:
        return None
    return "Neither plot synopsis nor review insights brief available"


def _check_viewer_experience(
    plot_synopsis: str | None,
    review_insights_brief: str | None,
    genres: list[str],
    merged_keywords: list[str],
    maturity_summary: str | None,
) -> str | None:
    """Viewer experience: review_insights_brief OR plot_synopsis
    OR all of (genres, keywords, maturity_summary)."""
    if review_insights_brief or plot_synopsis:
        return None

    has_all_contextual = (
        len(genres) >= 1
        and len(merged_keywords) >= 1
        and maturity_summary is not None
    )
    if has_all_contextual:
        return None

    return (
        "Insufficient data: no plot synopsis, no review insights, "
        "and incomplete genre/keyword/maturity data"
    )


def _check_watch_context(
    review_insights_brief: str | None,
    genres: list[str],
    merged_keywords: list[str],
    maturity_summary: str | None,
) -> str | None:
    """Watch context: review_insights_brief OR all of (genres, keywords, maturity).

    Watch context deliberately receives NO plot info (Decision 2).
    """
    if review_insights_brief:
        return None

    has_all_contextual = (
        len(genres) >= 1
        and len(merged_keywords) >= 1
        and maturity_summary is not None
    )
    if has_all_contextual:
        return None

    return "No review insights and incomplete genre/keyword/maturity data"


def _check_narrative_techniques(
    plot_synopsis: str | None,
    review_insights_brief: str | None,
    genres: list[str],
    merged_keywords: list[str],
) -> str | None:
    """Narrative techniques: plot_synopsis OR review_insights_brief
    OR all of (genres, keywords)."""
    if plot_synopsis or review_insights_brief:
        return None
    if len(genres) >= 1 and len(merged_keywords) >= 1:
        return None
    return "Insufficient data for structural analysis"


def _check_production_keywords(merged_keywords: list[str]) -> str | None:
    """Production keywords requires any merged keywords."""
    if len(merged_keywords) >= 1:
        return None
    return "No keywords available"


def _check_source_of_inspiration(
    merged_keywords: list[str],
    review_insights_brief: str | None,
) -> str | None:
    """Source of inspiration: any of keywords or review insights."""
    if merged_keywords or review_insights_brief:
        return None
    return "No keywords or review insights available"



# ---------------------------------------------------------------------------
# 4. Skip condition orchestrator
# ---------------------------------------------------------------------------

def assess_skip_conditions(
    movie_input: MovieInputData,
    *,
    plot_events_output: PlotEventsOutput | None = None,
    reception_output: ReceptionOutput | None = None,
    merged_keywords: list[str] | None = None,
    maturity_summary: str | None = None,
) -> SkipAssessment:
    """Evaluate per-generation minimum data thresholds.

    Calls the relevant per-generation _check_* methods and assembles
    results into a SkipAssessment.

    Called twice in the pipeline:
    - Before Wave 1 (all kwargs None): checks plot_events + reception
    - Before Wave 2 (outputs provided): checks the 6 Wave 2 types

    Args:
        movie_input: Raw movie data from the database.
        plot_events_output: Typed output from plot_events generation.
            None for Wave 1 assessment.
        reception_output: Typed output from reception generation.
            None for Wave 1 assessment.
        merged_keywords: Pre-computed merged keyword list. If None and
            needed (Wave 2), computed from movie_input on the fly.
        maturity_summary: Pre-computed maturity summary string. Used by
            viewer_experience and watch_context eligibility checks.

    Returns:
        SkipAssessment with generations_to_run and skip_reasons.
    """
    generations_to_run: set[str] = set()
    skip_reasons: dict[str, str] = {}

    # Helper to record each check result
    def _record(gen_type: str, reason: str | None) -> None:
        if reason is None:
            generations_to_run.add(gen_type)
        else:
            skip_reasons[gen_type] = reason

    # Wave 1: no outputs provided yet — check plot_events + reception
    if plot_events_output is None and reception_output is None:
        _record("plot_events", check_plot_events(movie_input))
        _record("reception", check_reception(movie_input))
        return SkipAssessment(
            generations_to_run=generations_to_run,
            skip_reasons=skip_reasons,
        )

    # Wave 2: extract intermediates from typed Wave 1 outputs
    plot_synopsis = (
        plot_events_output.plot_summary
        if plot_events_output is not None
        else None
    )
    review_insights_brief = (
        reception_output.review_insights_brief
        if reception_output is not None
        else None
    )

    # Compute merged keywords if not pre-computed
    if merged_keywords is None:
        merged_keywords = list(
            dict.fromkeys(movie_input.plot_keywords + movie_input.overall_keywords)
        )

    _record("plot_analysis", _check_plot_analysis(
        plot_synopsis, review_insights_brief,
    ))
    _record("viewer_experience", _check_viewer_experience(
        plot_synopsis, review_insights_brief,
        movie_input.genres, merged_keywords, maturity_summary,
    ))
    _record("watch_context", _check_watch_context(
        review_insights_brief,
        movie_input.genres, merged_keywords, maturity_summary,
    ))
    _record("narrative_techniques", _check_narrative_techniques(
        plot_synopsis, review_insights_brief,
        movie_input.genres, merged_keywords,
    ))
    _record("production_keywords", _check_production_keywords(merged_keywords))
    _record("source_of_inspiration", _check_source_of_inspiration(
        merged_keywords, review_insights_brief,
    ))

    return SkipAssessment(
        generations_to_run=generations_to_run,
        skip_reasons=skip_reasons,
    )


# ---------------------------------------------------------------------------
# Orchestrator: run all pre-consolidation steps for Wave 1
# ---------------------------------------------------------------------------

def run_pre_consolidation(movie_input: MovieInputData) -> ConsolidatedInputs:
    """Run all three pre-consolidation steps and return ConsolidatedInputs.

    This is the entry point called by request_builder.build_wave1_requests()
    for each movie. It:
    1. Routes keywords into plot/overall/merged lists
    2. Consolidates maturity data into a single summary string
    3. Assesses Wave 1 skip conditions (plot_events + reception)
    4. Packages everything into ConsolidatedInputs

    For Wave 2 skip assessment, callers use assess_skip_conditions()
    directly with actual Wave 1 typed outputs.
    """
    # Step 1: keyword routing (normalizes and deduplicates)
    _plot_kw, _overall_kw, merged_keywords = route_keywords(
        movie_input.plot_keywords,
        movie_input.overall_keywords,
    )

    # Step 2: maturity consolidation
    maturity_summary = consolidate_maturity(
        movie_input.maturity_rating,
        movie_input.maturity_reasoning,
        movie_input.parental_guide_items,
    )

    # Step 3: Wave 1 skip assessment
    skip_assessment = assess_skip_conditions(movie_input)

    # Step 4: assemble ConsolidatedInputs
    return ConsolidatedInputs(
        movie_input=movie_input,
        title_with_year=movie_input.title_with_year(),
        merged_keywords=merged_keywords,
        maturity_summary=maturity_summary,
        skip_assessment=skip_assessment,
    )
