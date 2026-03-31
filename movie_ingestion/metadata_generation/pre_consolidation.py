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
from .schemas import (
    PlotAnalysisWithJustificationsOutput,
    PlotEventsOutput,
    ReceptionOutput,
)


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

_MIN_REVIEWS_CHARS = 400

# Plot analysis skip condition thresholds (tiered eligibility).
# Tier 2: plot fallback alone is enough if it's >= 400 chars.
_MIN_PLOT_FALLBACK_CHARS = 400
# Tier 3: shorter plot fallback (250-399) is eligible only when paired
# with rich thematic observations (>= 300 chars) that compensate.
_MIN_PLOT_FALLBACK_WITH_OBSERVATIONS_CHARS = 250
_MIN_THEMATIC_OBSERVATIONS_CHARS = 300

# Viewer experience narrative resolution thresholds.
# GPO-only narrative path (validated by Round 3 evaluation: GPO-only matched
# or slightly exceeded the full fallback chain across all buckets).
# Inclusion: minimum length to pass GPO to the LLM as narrative context.
# Standalone: minimum length to pass eligibility without any observations.
_MIN_VIEWER_GPO_INCLUSION_CHARS = 200
_MIN_VIEWER_GPO_STANDALONE_CHARS = 350

# Viewer experience observation thresholds.
# Inclusion: minimum per-field length to count as "usable."
_MIN_VIEWER_EMOTIONAL_OBSERVATIONS_CHARS = 120
_MIN_VIEWER_EMOTIONAL_OBSERVATIONS_STANDALONE_CHARS = 160
_MIN_VIEWER_CRAFT_OBSERVATIONS_CHARS = 120
_MIN_VIEWER_THEMATIC_OBSERVATIONS_CHARS = 140
_MIN_VIEWER_COMBINED_OBSERVATIONS_STANDALONE_CHARS = 280

# Narrative techniques eligibility thresholds (tiered).
# Tier 2: raw plot fallback standalone — needs enough detail to reveal
# temporal structure, POV, information control, and arcs.
_MIN_NT_PLOT_FALLBACK_STANDALONE_CHARS = 500
# Tier 3: craft_observations standalone — reviewers directly describe
# structural craft. At 400+ chars, evaluation showed quality holds vs the
# 450-549 range (craft_threshold_edge bucket scored 3.96-4.21 across
# candidates). Lowered from 450 to gain ~6,600 additional movies.
_MIN_NT_CRAFT_OBSERVATIONS_STANDALONE_CHARS = 400
# Tier 4: combined moderate plot + moderate craft. Neither alone is sufficient
# but together they provide enough for the LLM to triangulate technique labels.
_MIN_NT_PLOT_FALLBACK_COMBINED_CHARS = 300
_MIN_NT_CRAFT_OBSERVATIONS_COMBINED_CHARS = 300


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
        "and insufficient review text (combined <400 chars)"
    )


def _check_plot_analysis(
    plot_summary: str | None,
    thematic_observations: str | None,
    movie_input: MovieInputData,
) -> str | None:
    """Plot analysis eligibility based on tiered input quality.

    Tier 1: plot_summary from Wave 1 plot_events → always eligible.
    Tier 2: plot fallback (raw synopsis/summary/overview) >= 400 chars
            → always eligible (enough narrative to ground analysis).
    Tier 3: plot fallback 250-399 chars + thematic_observations >= 300
            chars → eligible (rich observations compensate for thin plot).
    Otherwise: skip (insufficient data for meaningful analysis).

    The plot fallback is the longest of the movie's raw plot sources
    (first synopsis, longest plot_summary, or overview), computed by
    MovieInputData.best_plot_fallback().
    """
    # Tier 1: Wave 1 plot_events produced a plot_summary
    if plot_summary:
        return None

    # Compute best available raw plot text length
    fallback = movie_input.best_plot_fallback()
    best_plot_len = len(fallback) if fallback else 0

    thematic_len = len(thematic_observations) if thematic_observations else 0

    # Tier 2: substantial plot fallback text
    if best_plot_len >= _MIN_PLOT_FALLBACK_CHARS:
        return None

    # Tier 3: moderate plot fallback + rich thematic observations
    if (best_plot_len >= _MIN_PLOT_FALLBACK_WITH_OBSERVATIONS_CHARS
            and thematic_len >= _MIN_THEMATIC_OBSERVATIONS_CHARS):
        return None

    return (
        f"Insufficient plot data for analysis: best plot fallback is "
        f"{best_plot_len} chars, thematic observations is {thematic_len} chars"
    )


def resolve_viewer_experience_narrative(
    generalized_plot_overview: str | None,
) -> tuple[str | None, str | None]:
    """Resolve the single narrative input for viewer_experience.

    Shared by both eligibility checking and prompt building so the
    resolution logic lives in one place.

    GPO-only narrative path — Round 3 evaluation validated that
    generalized_plot_overview (a cleaned LLM summary from plot_analysis)
    matches or slightly exceeds the full fallback chain (plot_summary,
    raw synopses, overview) across all input quality buckets. GPO strips
    noise while preserving the thematic/emotional core that viewer_experience
    needs. This also simplifies upstream dependencies by removing the
    plot_summary requirement from the viewer_experience pipeline.

    Returns (narrative_text, source_label), or (None, None) when GPO
    is absent or below the inclusion threshold.
    """
    if (
        generalized_plot_overview
        and len(generalized_plot_overview) >= _MIN_VIEWER_GPO_INCLUSION_CHARS
    ):
        return generalized_plot_overview, "generalized_plot_overview"

    return None, None


def filter_viewer_experience_observations(
    emotional_observations: str | None,
    craft_observations: str | None,
    thematic_observations: str | None,
) -> tuple[str | None, str | None, str | None]:
    """Filter observation inputs by their per-field inclusion thresholds.

    Observations below their threshold are returned as None. Used by
    both eligibility checking (derives lengths from non-None results)
    and prompt building (passes filtered strings directly to the LLM).
    """
    filtered_emotional = (
        emotional_observations
        if emotional_observations
        and len(emotional_observations) >= _MIN_VIEWER_EMOTIONAL_OBSERVATIONS_CHARS
        else None
    )
    filtered_craft = (
        craft_observations
        if craft_observations
        and len(craft_observations) >= _MIN_VIEWER_CRAFT_OBSERVATIONS_CHARS
        else None
    )
    filtered_thematic = (
        thematic_observations
        if thematic_observations
        and len(thematic_observations) >= _MIN_VIEWER_THEMATIC_OBSERVATIONS_CHARS
        else None
    )
    return filtered_emotional, filtered_craft, filtered_thematic


def _check_viewer_experience(
    generalized_plot_overview: str | None,
    emotional_observations: str | None,
    craft_observations: str | None,
    thematic_observations: str | None,
) -> str | None:
    """Viewer experience eligibility — GPO narrative + observation paths.

    Simplified after Round 3 evaluation validated that:
    - GPO-only narrative matches/exceeds the full fallback chain
    - Observation-standalone produces excellent output with justifications
    - All input quality buckets produce acceptable results (4.0+ avg)

    Pass when:
    - GPO is strong enough on its own (>= 350 chars), OR
    - observation input is strong enough on its own, OR
    - GPO above inclusion threshold (>= 200 chars) + any usable observation

    Genre/maturity inputs are support-only and never rescue eligibility.
    """
    narrative_input, _ = resolve_viewer_experience_narrative(
        generalized_plot_overview,
    )
    narrative_len = len(narrative_input) if narrative_input else 0

    filtered_emotional, filtered_craft, filtered_thematic = (
        filter_viewer_experience_observations(
            emotional_observations,
            craft_observations,
            thematic_observations,
        )
    )
    emotional_len = len(filtered_emotional) if filtered_emotional else 0
    craft_len = len(filtered_craft) if filtered_craft else 0
    thematic_len = len(filtered_thematic) if filtered_thematic else 0
    combined_observation_len = emotional_len + craft_len + thematic_len
    has_usable_observation = any((emotional_len, craft_len, thematic_len))

    # Path 1: Strong standalone GPO narrative.
    if narrative_len >= _MIN_VIEWER_GPO_STANDALONE_CHARS:
        return None

    # Path 2: Strong standalone observations.
    if emotional_len >= _MIN_VIEWER_EMOTIONAL_OBSERVATIONS_STANDALONE_CHARS:
        return None
    if (
        combined_observation_len >= _MIN_VIEWER_COMBINED_OBSERVATIONS_STANDALONE_CHARS
        and (emotional_len > 0 or craft_len > 0)
    ):
        return None

    # Path 3: GPO above inclusion threshold + any usable observation.
    # Looser than the old source-weighted combined thresholds — justified
    # by Round 2/3 results showing justifications handle sparse inputs well.
    if narrative_input and has_usable_observation:
        return None

    return (
        "Insufficient viewer_experience signal: GPO did not meet standalone "
        f"threshold ({narrative_len} chars, need >={_MIN_VIEWER_GPO_STANDALONE_CHARS}), "
        f"and observations were not strong enough to compensate "
        f"(combined {combined_observation_len} chars)"
    )


def _check_watch_context(
    genre_signatures: list[str] | None,
    genres: list[str],
) -> str | None:
    """Watch context: genre_signatures OR raw genres >= 1.

    Watch context deliberately receives NO plot info (Decision 2).
    Genres enable grounded categorical inference for viewing occasions
    (horror → halloween, romance → date night). All other inputs
    (observations, keywords, maturity) are enrichments that improve
    quality but never gate eligibility.

    See evaluation_data/watch_context_eval_guide.md for rationale.
    """
    if genre_signatures:
        return None
    if len(genres) >= 1:
        return None
    return "No genre_signatures and no raw genres available"


def resolve_narrative_techniques_narrative(
    movie_input: MovieInputData,
    plot_summary: str | None,
) -> tuple[str | None, str | None]:
    """Resolve the single narrative input for narrative_techniques.

    Shared by both eligibility checking and prompt building so the
    resolution logic lives in one place. Winner-takes-all: the first
    source to clear its inclusion threshold wins.

    Fallback order:
    1. plot_summary (Wave 1 plot_events output) — any length, since
       plot_events already required 600+ chars of source text.
    2. best_plot_fallback() if >= 500 chars standalone or >= 300 chars
       combined (combined path checked separately by the caller).

    Returns (narrative_text, source_label), or (None, None) when no
    narrative source clears its threshold. The source_label distinguishes
    quality tiers: "plot_synopsis" (LLM-condensed) vs "plot_text"
    (raw human-written) so the prompt can signal the quality tier.
    Labels match the field names in the system prompt's INPUTS section.
    """
    # Tier 1: LLM-condensed plot summary from Wave 1
    if plot_summary:
        return plot_summary, "plot_synopsis"

    # Tier 2+: raw plot fallback — threshold depends on whether craft
    # observations are also present (standalone vs combined). Return
    # the raw text and let the caller decide via threshold comparison.
    raw_fallback = movie_input.best_plot_fallback()
    if raw_fallback and len(raw_fallback) >= _MIN_NT_PLOT_FALLBACK_COMBINED_CHARS:
        return raw_fallback, "plot_text"

    return None, None


def _check_narrative_techniques(
    plot_summary: str | None,
    craft_observations: str | None,
    movie_input: MovieInputData,
) -> str | None:
    """Narrative techniques eligibility based on tiered input quality.

    Tier 1: plot_summary from Wave 1 plot_events -> always eligible.
    Tier 2: best_plot_fallback >= 500 chars -> eligible (substantial raw
            plot carries enough structural detail for technique analysis).
    Tier 3: craft_observations >= 450 chars -> eligible standalone
            (reviewers directly describe narrative craft).
    Tier 4: best_plot_fallback >= 300 + craft_observations >= 300 ->
            eligible (combined moderate sources compensate each other).
    Otherwise: skip.

    Genres + keywords alone are not sufficient — without plot detail or
    reviewer structural commentary, the LLM would produce genre-based
    speculation rather than grounded technique analysis.
    """
    # Tier 1: Wave 1 plot_events produced a plot_summary
    if plot_summary:
        return None

    # Compute input lengths for tiered checks
    raw_fallback = movie_input.best_plot_fallback()
    fallback_len = len(raw_fallback) if raw_fallback else 0
    craft_len = len(craft_observations) if craft_observations else 0

    # Tier 2: substantial raw plot fallback
    if fallback_len >= _MIN_NT_PLOT_FALLBACK_STANDALONE_CHARS:
        return None

    # Tier 3: craft observations standalone
    if craft_len >= _MIN_NT_CRAFT_OBSERVATIONS_STANDALONE_CHARS:
        return None

    # Tier 4: combined moderate plot + moderate craft
    if (
        fallback_len >= _MIN_NT_PLOT_FALLBACK_COMBINED_CHARS
        and craft_len >= _MIN_NT_CRAFT_OBSERVATIONS_COMBINED_CHARS
    ):
        return None

    return (
        f"Insufficient data for structural analysis: best plot fallback is "
        f"{fallback_len} chars, craft observations is {craft_len} chars"
    )


def _check_production_keywords(merged_keywords: list[str]) -> str | None:
    """Production keywords requires any merged keywords."""
    if len(merged_keywords) >= 1:
        return None
    return "No keywords available"


def _check_source_of_inspiration(
    merged_keywords: list[str],
    source_material_hint: str | None,
) -> str | None:
    """Source of inspiration: any keywords or source material hint.

    Eligible when the generator has at least some grounding data beyond
    just the title. Keywords provide structured source tags
    ("based-on-novel", "remake") and production medium signals.
    source_material_hint provides reviewer-extracted adaptation evidence.
    Either alone is sufficient.
    """
    if merged_keywords or source_material_hint:
        return None
    return "No keywords or source material hint available"



# ---------------------------------------------------------------------------
# 4. Skip condition orchestrator
# ---------------------------------------------------------------------------

def assess_skip_conditions(
    movie_input: MovieInputData,
    *,
    plot_events_output: PlotEventsOutput | None = None,
    reception_output: ReceptionOutput | None = None,
    plot_analysis_output: PlotAnalysisWithJustificationsOutput | None = None,
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
    plot_summary = (
        plot_events_output.plot_summary
        if plot_events_output is not None
        else None
    )

    # Individual fields from the reception extraction zone.
    # Used directly by Wave 2 eligibility checks and generators.
    thematic_observations = (
        reception_output.thematic_observations
        if reception_output is not None
        else None
    )
    emotional_observations = (
        reception_output.emotional_observations
        if reception_output is not None
        else None
    )
    craft_observations = (
        reception_output.craft_observations
        if reception_output is not None
        else None
    )
    source_material_hint = (
        reception_output.source_material_hint
        if reception_output is not None
        else None
    )

    generalized_plot_overview = (
        plot_analysis_output.generalized_plot_overview
        if plot_analysis_output is not None
        else None
    )

    genre_signatures = (
        plot_analysis_output.genre_signatures
        if plot_analysis_output is not None
        else None
    )

    # Compute merged keywords if not pre-computed
    if merged_keywords is None:
        merged_keywords = list(
            dict.fromkeys(movie_input.plot_keywords + movie_input.overall_keywords)
        )

    _record("plot_analysis", _check_plot_analysis(
        plot_summary, thematic_observations, movie_input,
    ))
    _record("viewer_experience", _check_viewer_experience(
        generalized_plot_overview,
        emotional_observations,
        craft_observations,
        thematic_observations,
    ))
    _record("watch_context", _check_watch_context(
        genre_signatures,
        movie_input.genres,
    ))
    _record("narrative_techniques", _check_narrative_techniques(
        plot_summary, craft_observations, movie_input,
    ))
    _record("production_keywords", _check_production_keywords(merged_keywords))
    _record("source_of_inspiration", _check_source_of_inspiration(
        merged_keywords, source_material_hint,
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
