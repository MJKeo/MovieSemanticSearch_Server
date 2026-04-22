# Search V2 — Stage 3 Entity Endpoint: Query Execution
#
# Takes the LLM's EntityQuerySpec output and runs the appropriate
# lexical-schema query, producing an EndpointResult with [0, 1] scores
# per matched movie_id. Works uniformly for both dealbreakers (no
# restrict set — return naturally matched movies) and preferences
# (restrict set provided — return one entry per supplied ID, with
# 0.0 for non-matches).
#
# The endpoint is direction-agnostic and scoring-policy-agnostic:
# exclusion framing (hard filter vs semantic penalty), preference
# weighting (regular / primary / prior), and candidate-pool assembly
# are all orchestrator concerns handled in step 4. This module's sole
# job is: spec in → raw [0, 1] per-movie scores out.
#
# Scoring criteria reference — finalized_search_proposal.md §Endpoint 1
# and search_improvement_planning/character_scoring_revamp.md:
#   - Non-actor non-character role persons (director, writer, producer,
#     composer) / title patterns → binary 1.0 match.
#   - Actor persons → zone-based prominence via billing_position +
#     cast_size, one of four prominence modes (DEFAULT / LEAD /
#     SUPPORTING / MINOR) from the unified ProminenceMode enum.
#   - Characters → billing-position prominence via CENTRAL / DEFAULT
#     from the same unified ProminenceMode enum against
#     lex.inv_character_postings.
#   - All prominence scores (actor + character) are compressed into
#     [0.5, 1.0] via `_compress_to_floor` so any real match stays at
#     or above the dealbreaker-eligible floor.
#   - broad_person → per-table scores merged with max; non-primary
#     tables discounted to 0.5× when primary_category is set, with a
#     final 0.5 floor applied so a real match can never drop below
#     the dealbreaker-eligible band after weighting.
#   - Person and character lookups both expand primary_form +
#     alternative_forms into a set of term_ids and take the max score
#     per movie across variant rows. This is the "peter parker OR
#     spider-man wins" merge.
# Studio lookups have their own stage-3 endpoint
# (search_v2/stage_3/studio_query_execution.py) and are not handled here.
#
# The scoring constants (LEAD_FLOOR, LEAD_SCALE, SUPP_SCALE, plus
# per-mode curves) are all tunable starting points from the proposal.

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass

from db.postgres import (
    PEOPLE_POSTING_TABLES,
    PostingTable,
    fetch_actor_billing_rows,
    fetch_character_billing_rows,
    fetch_character_strings_exact,
    fetch_movie_ids_by_term_ids,
    fetch_movie_ids_with_title_like,
    fetch_phrase_term_ids,
)
from implementation.misc.helpers import normalize_string
from implementation.misc.sql_like import escape_like
from schemas.endpoint_result import EndpointResult
from schemas.entity_translation import EntityQuerySpec
from search_v2.stage_3.result_helpers import build_endpoint_result
from schemas.enums import (
    EntityType,
    PersonCategory,
    ProminenceMode,
    SpecificPersonCategory,
    TitlePatternMatchType,
)

# ---------------------------------------------------------------------------
# Zone cutoff constants (proposal §Actor Prominence Scoring).
# All values are empirically tunable starting points.
# ---------------------------------------------------------------------------

LEAD_FLOOR: int = 2
LEAD_SCALE: float = 0.6
SUPP_SCALE: float = 1.0

# Weight applied to non-primary table matches in broad_person cross-
# posting consolidation. Primary gets full credit, others get this
# multiplier. Uses max across tables (not sum).
BROAD_PERSON_NON_PRIMARY_WEIGHT: float = 0.5


# ---------------------------------------------------------------------------
# Shared tail transform — compresses raw [0, 1] prominence scores into
# the global [0.5, 1.0] floor-eligible band. Every prominence scorer
# (actor + character) runs its raw output through this so a real match
# never falls below the 0.5 dealbreaker floor.
# ---------------------------------------------------------------------------


def _compress_to_floor(raw: float) -> float:
    """Affine-compress a raw [0, 1] prominence score to [0.5, 1.0].

    Formula: ``0.5 + 0.5 * raw``. The clamp defends against floating-
    point drift and against future scorers whose raw output exceeds
    [0, 1]. See character_scoring_revamp.md §"Scoring Formulas" for
    the floor-rule rationale.
    """
    return max(0.5, min(1.0, 0.5 + 0.5 * raw))


# ---------------------------------------------------------------------------
# Shared form normalization — used by every person and character path
# to build the set of normalized search strings from primary_form +
# alternative_forms.
# ---------------------------------------------------------------------------


def _collect_normalized_forms(spec: EntityQuerySpec) -> list[str]:
    """Merge primary_form + alternative_forms, normalize each, dedupe
    while preserving order. Returns the list of normalized strings
    that should be looked up in the lexical / character string
    dictionary.

    Execution-layer invariant: the max score across all variant rows
    becomes the per-movie score in the downstream fetch helpers, so
    callers can pass every normalized form without pre-ranking them.
    """
    forms = [spec.primary_form]
    if spec.alternative_forms:
        forms.extend(spec.alternative_forms)
    seen: set[str] = set()
    out: list[str] = []
    for form in forms:
        norm = normalize_string(form)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


# ---------------------------------------------------------------------------
# Actor prominence scoring (pure functions — no I/O).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ZoneCutoffs:
    """Precomputed zone boundaries for a given cast_size. Billing positions
    1..lead_cutoff are LEAD; lead_cutoff+1..supp_cutoff are SUPPORTING;
    the remainder are MINOR."""

    lead_cutoff: int
    supp_cutoff: int


def _zone_cutoffs(cast_size: int) -> _ZoneCutoffs:
    """Derive zone boundaries using sqrt-scaled growth with floors."""
    sqrt_n = math.sqrt(cast_size)
    lead = min(cast_size, max(LEAD_FLOOR, round(LEAD_SCALE * sqrt_n)))
    supp = min(cast_size, max(lead + 1, round(SUPP_SCALE * sqrt_n)))
    return _ZoneCutoffs(lead_cutoff=lead, supp_cutoff=supp)


def _zone_relative_position(
    billing_position: int,
    zone_start: int,
    zone_end: int,
) -> float:
    """Normalize an in-zone billing position to zp ∈ [0, 1].

    zp = 0.0 at the top of the zone, 1.0 at the bottom. A single-member
    zone collapses to 0.0 so the formula matches the top-of-zone score.
    """
    span = zone_end - zone_start
    if span <= 0:
        return 0.0
    return (billing_position - zone_start) / span


def _actor_score_default(zone: str, zp: float) -> float:
    """DEFAULT mode — no prominence signal, leads get full credit,
    smooth gradient through supporting and minor."""
    if zone == "lead":
        return 1.0
    if zone == "supporting":
        return 0.85 - 0.15 * zp
    return 0.7 - 0.2 * zp  # minor


def _actor_score_lead(zone: str, zp: float) -> float:
    """LEAD mode — explicit 'starring' or 'lead role' language."""
    if zone == "lead":
        return 1.0
    if zone == "supporting":
        return 0.6 - 0.2 * zp
    return 0.4 - 0.2 * zp  # minor


def _actor_score_supporting(zone: str, zp: float) -> float:
    """SUPPORTING mode — explicit 'supporting role' language."""
    if zone == "lead":
        return 0.7 - 0.1 * zp
    if zone == "supporting":
        return 1.0
    return 0.6 - 0.25 * zp  # minor


def _actor_score_minor(zone: str, zp: float) -> float:
    """MINOR mode — explicit 'cameo' or 'minor role' language. Minor
    zone ramps UP as billing gets deeper."""
    if zone == "lead":
        return 0.35 - 0.1 * zp
    if zone == "supporting":
        return 0.5 - 0.15 * zp
    return 0.7 + 0.3 * zp  # minor


# Dispatch table for actor-table prominence modes. Keyed on the
# unified ProminenceMode enum. Validator in EntityQuerySpec guarantees
# only these four values reach actor-table scoring (CENTRAL is
# remapped to LEAD before it can arrive here).
_ACTOR_MODE_SCORERS = {
    ProminenceMode.DEFAULT: _actor_score_default,
    ProminenceMode.LEAD: _actor_score_lead,
    ProminenceMode.SUPPORTING: _actor_score_supporting,
    ProminenceMode.MINOR: _actor_score_minor,
}


def _actor_prominence_score(
    billing_position: int,
    cast_size: int,
    mode: ProminenceMode,
) -> float:
    """Score a single actor credit row under the given prominence mode.

    Uses the proposal's zone-based formulas with sqrt-adaptive cutoffs
    to compute a raw [0, 1] value, then compresses into [0.5, 1.0] so
    every matched actor scores at or above the dealbreaker floor. The
    per-mode scorers are unchanged — only this tail transform shifts
    the effective range.
    """
    cutoffs = _zone_cutoffs(cast_size)

    if billing_position <= cutoffs.lead_cutoff:
        zone = "lead"
        zp = _zone_relative_position(billing_position, 1, cutoffs.lead_cutoff)
    elif billing_position <= cutoffs.supp_cutoff:
        zone = "supporting"
        zp = _zone_relative_position(
            billing_position, cutoffs.lead_cutoff + 1, cutoffs.supp_cutoff
        )
    else:
        zone = "minor"
        zp = _zone_relative_position(
            billing_position, cutoffs.supp_cutoff + 1, cast_size
        )

    raw = _ACTOR_MODE_SCORERS[mode](zone, zp)
    return _compress_to_floor(raw)


# ---------------------------------------------------------------------------
# Character prominence scoring (pure functions — no I/O).
#
# Two modes on the unified enum, picked by the stage 3 LLM from
# description wording:
#   CENTRAL — character is the subject of the query. Fixed decay
#     curve; cast_size-independent. "Spider-Man movies" scores the
#     same way whether the cast is 8 or 80.
#   DEFAULT — character is a filter with a gentle prominence
#     preference. Linear ramp from 1.0 at the top to 0.5 at the bottom
#     of the movie's character cast (pre-compression raw ramps from
#     1.0 to 0.0; compression lifts the tail to 0.5).
#
# See search_improvement_planning/character_scoring_revamp.md
# §"Scoring Formulas" for the curve tables.
# ---------------------------------------------------------------------------


def _character_score_central(billing_position: int) -> float:
    """CENTRAL raw score — fixed decay, cast_size-independent.

    Positions 1 and 2 both score 1.0 (handles alias edges like
    "Peter Parker" / "Spider-Man" where the requested character
    appears second on its cast edge). Beyond position 2, decay 0.15
    per step until the raw value reaches 0, at which point
    compression pins it to the 0.5 floor.
    """
    return max(0.0, 1.0 - 0.15 * max(0, billing_position - 2))


def _character_score_default(billing_position: int, cast_size: int) -> float:
    """DEFAULT raw score — linear ramp across the character cast.

    Pre-compression: 1.0 at position 1, smoothly descending to 0.0 at
    position == cast_size. The `max(1, size - 1)` guards against
    single-character casts where the naive denominator would be 0.
    """
    return 1.0 - (billing_position - 1) / max(1, cast_size - 1)


# Dispatch table for character prominence modes. Keyed on the unified
# ProminenceMode enum. Validator guarantees only {DEFAULT, CENTRAL}
# reach character scoring (LEAD/SUPPORTING/MINOR are remapped before
# they arrive here).
_CHARACTER_MODE_SCORERS = {
    ProminenceMode.CENTRAL: lambda pos, size: _character_score_central(pos),
    ProminenceMode.DEFAULT: _character_score_default,
}


def _character_prominence_score(
    billing_position: int,
    character_cast_size: int,
    mode: ProminenceMode,
) -> float:
    """Score a single character credit row under the given mode.

    Dispatches to the mode-specific raw scorer, then applies the
    shared [0.5, 1.0] compression so character scores sit in the same
    floor-eligible band as actor scores.
    """
    raw = _CHARACTER_MODE_SCORERS[mode](billing_position, character_cast_size)
    return _compress_to_floor(raw)


# ---------------------------------------------------------------------------
# Per-sub-type executors. Each returns a {movie_id: score} dict with
# score ∈ (0, 1] for matches only. The top-level dispatcher applies
# the restrict set and assembles the final EndpointResult.
# ---------------------------------------------------------------------------


async def _fetch_actor_scores(
    term_ids: list[int],
    mode: ProminenceMode,
    restrict_movie_ids: set[int] | None,
) -> dict[int, float]:
    """Fetch billing rows for actor term_ids and score each row. Takes
    the max score per movie across variant rows — this is how the
    'primary_form OR alias wins' merge materializes for persons."""
    rows = await fetch_actor_billing_rows(term_ids, restrict_movie_ids)
    scores: dict[int, float] = {}
    for movie_id, billing_position, cast_size in rows:
        score = _actor_prominence_score(billing_position, cast_size, mode)
        prev = scores.get(movie_id)
        if prev is None or score > prev:
            scores[movie_id] = score
    return scores


async def _fetch_binary_role_scores(
    table: PostingTable,
    term_ids: list[int],
    restrict_movie_ids: set[int] | None,
) -> dict[int, float]:
    """Binary 1.0 scoring for non-actor / non-character role tables
    (director, writer, producer, composer). Character lookups moved
    off this path and onto prominence scoring via
    `_fetch_character_scores`; actor has its own prominence path.

    Post-filters by restrict in Python — the matched set size for a
    single person is small enough that pulling it back and
    intersecting is cheaper than extending every posting helper to
    support a server-side restrict."""
    movie_ids = await fetch_movie_ids_by_term_ids(table, term_ids)
    if restrict_movie_ids is not None:
        movie_ids = movie_ids & restrict_movie_ids
    return {mid: 1.0 for mid in movie_ids}


async def _fetch_character_scores(
    term_ids: list[int],
    mode: ProminenceMode,
    restrict_movie_ids: set[int] | None,
) -> dict[int, float]:
    """Fetch billing rows for character term_ids and score each row.

    Mirrors `_fetch_actor_scores`. When the same movie has multiple
    matching rows — which happens whenever the LLM supplies variant
    names like "Spider-Man" + "Peter Parker" that each resolve to a
    distinct term_id on the same cast — the per-movie score is the
    max across rows. That naturally implements the "max across
    nicknames" rule without special-casing the caller."""
    rows = await fetch_character_billing_rows(term_ids, restrict_movie_ids)
    scores: dict[int, float] = {}
    for movie_id, billing_position, character_cast_size in rows:
        score = _character_prominence_score(
            billing_position, character_cast_size, mode
        )
        prev = scores.get(movie_id)
        if prev is None or score > prev:
            scores[movie_id] = score
    return scores


async def _resolve_person_term_ids(spec: EntityQuerySpec) -> list[int]:
    """Normalize primary_form + alternative_forms and resolve each to
    a lexical term_id via the general lexical dictionary. Returns a
    deduplicated list of term_ids (may be empty if nothing resolves).
    """
    normalized = _collect_normalized_forms(spec)
    if not normalized:
        return []
    phrase_to_id = await fetch_phrase_term_ids(normalized)
    return list({tid for tid in phrase_to_id.values()})


async def _execute_person_specific_role(
    spec: EntityQuerySpec,
    restrict_movie_ids: set[int] | None,
) -> dict[int, float]:
    """One of: actor, director, writer, producer, composer. Resolve
    primary_form + alternative_forms against the lexical dictionary,
    then query that single posting table with every resulting
    term_id. Per-movie max across variant rows is handled in the
    fetch helpers."""
    # Preconditions guaranteed by the caller (execute_entity_query):
    #   entity_type == PERSON
    #   person_category is a specific role (not None, not broad_person)
    # Assert locally so a future refactor that reroutes to this
    # function fails loudly instead of KeyError-ing inside _ROLE_TO_TABLE.
    assert spec.person_category is not None and spec.person_category != PersonCategory.BROAD_PERSON

    term_ids = await _resolve_person_term_ids(spec)
    if not term_ids:
        return {}

    if spec.person_category == PersonCategory.ACTOR:
        # prominence_mode is guaranteed non-null here by the
        # EntityQuerySpec validator (coerced to DEFAULT when the LLM
        # left it null for an actor-table search).
        assert spec.prominence_mode is not None
        return await _fetch_actor_scores(
            term_ids, spec.prominence_mode, restrict_movie_ids
        )

    table = _ROLE_TO_TABLE[spec.person_category]
    return await _fetch_binary_role_scores(table, term_ids, restrict_movie_ids)


# Specific-role person_category → posting table. Actor is excluded on
# purpose — it uses the prominence-scored path, not the binary one.
_ROLE_TO_TABLE: dict[PersonCategory, PostingTable] = {
    PersonCategory.DIRECTOR: PostingTable.DIRECTOR,
    PersonCategory.WRITER: PostingTable.WRITER,
    PersonCategory.PRODUCER: PostingTable.PRODUCER,
    PersonCategory.COMPOSER: PostingTable.COMPOSER,
}

# Map every people-posting table to the PersonCategory that anchors
# broad_person consolidation. Used to apply the primary_category
# weighting during cross-posting merge.
_TABLE_TO_ROLE: dict[PostingTable, SpecificPersonCategory] = {
    PostingTable.ACTOR: SpecificPersonCategory.ACTOR,
    PostingTable.DIRECTOR: SpecificPersonCategory.DIRECTOR,
    PostingTable.WRITER: SpecificPersonCategory.WRITER,
    PostingTable.PRODUCER: SpecificPersonCategory.PRODUCER,
    PostingTable.COMPOSER: SpecificPersonCategory.COMPOSER,
}


async def _execute_person_broad(
    spec: EntityQuerySpec,
    restrict_movie_ids: set[int] | None,
) -> dict[int, float]:
    """broad_person — resolve primary_form + alternative_forms, then
    search all five role tables in parallel with every resolved
    term_id. Merge per-table scores via max (with primary-category
    weighting)."""
    term_ids = await _resolve_person_term_ids(spec)
    if not term_ids:
        return {}

    # Kick off actor (prominence-scored) and four binary role fetches
    # concurrently. The same term_ids resolve the person in every
    # table.
    assert spec.prominence_mode is not None
    actor_task = _fetch_actor_scores(
        term_ids, spec.prominence_mode, restrict_movie_ids
    )
    binary_tasks = {
        table: _fetch_binary_role_scores(table, term_ids, restrict_movie_ids)
        for table in PEOPLE_POSTING_TABLES
        if table is not PostingTable.ACTOR
    }
    actor_scores, *binary_results = await asyncio.gather(
        actor_task, *binary_tasks.values()
    )
    per_table: dict[PostingTable, dict[int, float]] = {PostingTable.ACTOR: actor_scores}
    for table, result in zip(binary_tasks.keys(), binary_results):
        per_table[table] = result

    # Apply primary_category weighting. Tables that aren't the primary
    # get their score multiplied by BROAD_PERSON_NON_PRIMARY_WEIGHT.
    # When primary is null, every table keeps full weight.
    primary = spec.primary_category
    merged: dict[int, float] = {}
    for table, table_scores in per_table.items():
        is_primary = primary is not None and _TABLE_TO_ROLE[table] == primary
        weight = 1.0 if (primary is None or is_primary) else BROAD_PERSON_NON_PRIMARY_WEIGHT
        for movie_id, score in table_scores.items():
            weighted = score * weight
            prev = merged.get(movie_id)
            if prev is None or weighted > prev:
                merged[movie_id] = weighted

    # Apply the dealbreaker-eligible 0.5 floor to the merged score.
    # Per-row prominence scores are already in [0.5, 1.0] via
    # `_compress_to_floor`, but non-primary weighting
    # (BROAD_PERSON_NON_PRIMARY_WEIGHT = 0.5) can drop a real match
    # below the floor — e.g. a compressed actor score of 0.5 becomes
    # 0.25 under non-primary weighting. Floor here so any matched
    # movie stays at or above the same threshold as direct per-role
    # scoring. Ordering at the floor collapses (several matches all
    # land at 0.5), which is the intended behavior — the floor wins
    # over within-non-primary gradient.
    return {movie_id: max(0.5, score) for movie_id, score in merged.items()}


async def _execute_character(
    spec: EntityQuerySpec,
    restrict_movie_ids: set[int] | None,
) -> dict[int, float]:
    """Character lookup — exact-match primary_form plus every
    alternative credited form against lex.character_strings. Each
    matched term_id produces a per-row score via the CENTRAL or
    DEFAULT prominence curve, and per-movie we take the max across
    variant rows."""
    normalized = _collect_normalized_forms(spec)
    if not normalized:
        return {}

    phrase_to_id = await fetch_character_strings_exact(normalized)
    term_ids = list({tid for tid in phrase_to_id.values()})
    if not term_ids:
        return {}

    # prominence_mode is guaranteed non-null here by the
    # EntityQuerySpec validator (coerced to DEFAULT when the LLM
    # leaves it null for a character entity).
    assert spec.prominence_mode is not None
    return await _fetch_character_scores(
        term_ids, spec.prominence_mode, restrict_movie_ids
    )


async def _execute_title_pattern(
    spec: EntityQuerySpec,
    restrict_movie_ids: set[int] | None,
) -> dict[int, float]:
    """Title pattern lookup — LIKE match of a normalized pattern against
    public.movie_card.title_normalized. Both sides of the comparison are
    run through normalize_string at ingest / query time so the match is
    case- and diacritic-insensitive without ILIKE or unaccent()."""
    norm = normalize_string(spec.primary_form)
    if not norm:
        return {}

    # Escape LIKE metacharacters BEFORE wrapping with '%', so any '%'
    # or '_' the LLM emitted in the pattern text is treated as a
    # literal.
    escaped = escape_like(norm)
    match_type = spec.title_pattern_match_type or TitlePatternMatchType.CONTAINS
    if match_type == TitlePatternMatchType.STARTS_WITH:
        like_pattern = f"{escaped}%"
    else:  # CONTAINS
        like_pattern = f"%{escaped}%"

    movie_ids = await fetch_movie_ids_with_title_like(like_pattern, restrict_movie_ids)
    return {mid: 1.0 for mid in movie_ids}


# ---------------------------------------------------------------------------
# Result assembly + public entry point.
# ---------------------------------------------------------------------------


async def execute_entity_query(
    spec: EntityQuerySpec,
    *,
    restrict_to_movie_ids: set[int] | None = None,
) -> EndpointResult:
    """Execute one EntityQuerySpec against the lexical schema.

    Single entry point for both dealbreakers and preferences. The
    restrict_to_movie_ids parameter controls output shape:
      - None (dealbreaker path) → return one ScoredCandidate per
        naturally matched movie, non-matches omitted.
      - set[int] (preference path) → return exactly one ScoredCandidate
        per supplied ID, with 0.0 for non-matches.

    Returns an empty EndpointResult when the entity name normalizes
    to empty, or when exact-match dictionary resolution finds nothing —
    "no match is a valid result" per the endpoint spec.

    Args:
        spec: Validated EntityQuerySpec from the step 3 entity LLM.
        restrict_to_movie_ids: Optional candidate-pool restriction.
            Pass the preference's candidate pool to get one entry per
            ID; omit to get the natural match set for dealbreakers.

    Returns:
        EndpointResult with scores ∈ [0, 1] per movie.
    """
    if spec.entity_type == EntityType.PERSON:
        if spec.person_category is None:
            # Spec is malformed — a person entity without a category
            # has no posting table to search. Return empty rather than
            # silently defaulting, so the failure surfaces in logs.
            scores_by_movie: dict[int, float] = {}
        elif spec.person_category == PersonCategory.BROAD_PERSON:
            scores_by_movie = await _execute_person_broad(spec, restrict_to_movie_ids)
        else:
            scores_by_movie = await _execute_person_specific_role(spec, restrict_to_movie_ids)
    elif spec.entity_type == EntityType.CHARACTER:
        scores_by_movie = await _execute_character(spec, restrict_to_movie_ids)
    elif spec.entity_type == EntityType.TITLE_PATTERN:
        scores_by_movie = await _execute_title_pattern(spec, restrict_to_movie_ids)
    else:
        # Exhaustive — EntityType has three values. Any future addition
        # should fail loudly here rather than silently return empty.
        raise ValueError(f"Unhandled entity_type: {spec.entity_type}")

    return build_endpoint_result(scores_by_movie, restrict_to_movie_ids)
