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
# Scoring criteria reference — finalized_search_proposal.md §Endpoint 1:
#   - Non-actor persons / studios / characters / title patterns →
#     binary 1.0 match.
#   - Actor persons → zone-based prominence via billing_position +
#     cast_size, one of four modes (DEFAULT / LEAD / SUPPORTING /
#     MINOR) selected by the LLM.
#   - broad_person → per-table scores merged with max; non-primary
#     tables discounted to 0.5× when primary_category is set.
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
    ActorProminenceMode,
    EntityType,
    PersonCategory,
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


def _score_in_default_mode(zone: str, zp: float) -> float:
    """DEFAULT mode — no prominence signal, leads get full credit,
    smooth gradient through supporting and minor."""
    if zone == "lead":
        return 1.0
    if zone == "supporting":
        return 0.85 - 0.15 * zp
    return 0.7 - 0.2 * zp  # minor


def _score_in_lead_mode(zone: str, zp: float) -> float:
    """LEAD mode — explicit 'starring' or 'lead role' language."""
    if zone == "lead":
        return 1.0
    if zone == "supporting":
        return 0.6 - 0.2 * zp
    return 0.4 - 0.2 * zp  # minor


def _score_in_supporting_mode(zone: str, zp: float) -> float:
    """SUPPORTING mode — explicit 'supporting role' language."""
    if zone == "lead":
        return 0.7 - 0.1 * zp
    if zone == "supporting":
        return 1.0
    return 0.6 - 0.25 * zp  # minor


def _score_in_minor_mode(zone: str, zp: float) -> float:
    """MINOR mode — explicit 'cameo' or 'minor role' language. Minor
    zone ramps UP as billing gets deeper."""
    if zone == "lead":
        return 0.35 - 0.1 * zp
    if zone == "supporting":
        return 0.5 - 0.15 * zp
    return 0.7 + 0.3 * zp  # minor


# Dispatch table for the four prominence modes. Keeps _prominence_score
# readable and lets us add a mode without reshaping conditionals.
_MODE_SCORERS = {
    ActorProminenceMode.DEFAULT: _score_in_default_mode,
    ActorProminenceMode.LEAD: _score_in_lead_mode,
    ActorProminenceMode.SUPPORTING: _score_in_supporting_mode,
    ActorProminenceMode.MINOR: _score_in_minor_mode,
}


def _prominence_score(
    billing_position: int,
    cast_size: int,
    mode: ActorProminenceMode,
) -> float:
    """Score a single actor credit row under the given prominence mode.

    Uses the proposal's zone-based formulas with sqrt-adaptive cutoffs.
    Returns a value in [0, 1] — formulas are designed to stay in-range
    for realistic (billing_position, cast_size) pairs; the final clamp
    defends against floating-point drift only.
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

    raw = _MODE_SCORERS[mode](zone, zp)
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Per-sub-type executors. Each returns a {movie_id: score} dict with
# score ∈ (0, 1] for matches only. The top-level dispatcher applies
# the restrict set and assembles the final EndpointResult.
# ---------------------------------------------------------------------------


async def _fetch_actor_scores(
    term_ids: list[int],
    mode: ActorProminenceMode,
    restrict_movie_ids: set[int] | None,
) -> dict[int, float]:
    """Fetch billing rows for actor term_ids and score each row. If an
    actor is somehow credited twice for the same movie, take the max
    (defensive — PK should prevent this)."""
    rows = await fetch_actor_billing_rows(term_ids, restrict_movie_ids)
    scores: dict[int, float] = {}
    for movie_id, billing_position, cast_size in rows:
        score = _prominence_score(billing_position, cast_size, mode)
        if score <= 0.0:
            continue
        prev = scores.get(movie_id)
        if prev is None or score > prev:
            scores[movie_id] = score
    return scores


async def _fetch_binary_role_scores(
    table: PostingTable,
    term_ids: list[int],
    restrict_movie_ids: set[int] | None,
) -> dict[int, float]:
    """Binary 1.0 scoring for non-actor role tables, studios, and
    characters. Post-filters by restrict in Python — the matched set
    size for a single person/studio/character is small enough that
    pulling it back and intersecting is cheaper than extending every
    posting helper to support a server-side restrict."""
    movie_ids = await fetch_movie_ids_by_term_ids(table, term_ids)
    if restrict_movie_ids is not None:
        movie_ids = movie_ids & restrict_movie_ids
    return {mid: 1.0 for mid in movie_ids}


async def _execute_person_specific_role(
    spec: EntityQuerySpec,
    restrict_movie_ids: set[int] | None,
) -> dict[int, float]:
    """One of: actor, director, writer, producer, composer. Exact-match
    the name in the lexical dictionary, then query that single posting
    table."""
    # Preconditions guaranteed by the caller (execute_entity_query):
    #   entity_type == PERSON
    #   person_category is a specific role (not None, not broad_person)
    # Assert locally so a future refactor that reroutes to this
    # function fails loudly instead of KeyError-ing inside _ROLE_TO_TABLE.
    assert spec.person_category is not None and spec.person_category != PersonCategory.BROAD_PERSON

    norm = normalize_string(spec.lookup_text)
    if not norm:
        return {}

    phrase_to_id = await fetch_phrase_term_ids([norm])
    term_id = phrase_to_id.get(norm)
    if term_id is None:
        return {}

    if spec.person_category == PersonCategory.ACTOR:
        # actor_prominence_mode is guaranteed non-null here by the
        # EntityQuerySpec validator (coerced to DEFAULT when the LLM
        # left it null).
        assert spec.actor_prominence_mode is not None
        return await _fetch_actor_scores(
            [term_id], spec.actor_prominence_mode, restrict_movie_ids
        )

    table = _ROLE_TO_TABLE[spec.person_category]
    return await _fetch_binary_role_scores(table, [term_id], restrict_movie_ids)


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
    """broad_person — search all five role tables in parallel, then
    merge per-table scores via max (with primary-category weighting).
    """
    norm = normalize_string(spec.lookup_text)
    if not norm:
        return {}

    phrase_to_id = await fetch_phrase_term_ids([norm])
    term_id = phrase_to_id.get(norm)
    if term_id is None:
        return {}

    # Kick off actor (prominence-scored) and four binary role fetches
    # concurrently. Same term_id resolves the person in every table.
    assert spec.actor_prominence_mode is not None
    actor_task = _fetch_actor_scores(
        [term_id], spec.actor_prominence_mode, restrict_movie_ids
    )
    binary_tasks = {
        table: _fetch_binary_role_scores(table, [term_id], restrict_movie_ids)
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
    return merged


async def _execute_character(
    spec: EntityQuerySpec,
    restrict_movie_ids: set[int] | None,
) -> dict[int, float]:
    """Character lookup — exact match lookup_text plus every alternative
    credited form against lex.character_strings. A match on any variant
    scores 1.0."""
    # Build the full list of name variants, normalize each, dedupe
    # while preserving order.
    variants = [spec.lookup_text]
    if spec.character_alternative_names:
        variants.extend(spec.character_alternative_names)
    seen: set[str] = set()
    normalized: list[str] = []
    for variant in variants:
        norm = normalize_string(variant)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        normalized.append(norm)
    if not normalized:
        return {}

    phrase_to_id = await fetch_character_strings_exact(normalized)
    term_ids = list({tid for tid in phrase_to_id.values()})
    if not term_ids:
        return {}

    return await _fetch_binary_role_scores(
        PostingTable.CHARACTER, term_ids, restrict_movie_ids
    )


async def _execute_studio(
    spec: EntityQuerySpec,
    restrict_movie_ids: set[int] | None,
) -> dict[int, float]:
    """Studio lookup — exact match normalized studio name against the
    lexical dictionary, then query the studio posting table. Binary
    1.0 scoring."""
    norm = normalize_string(spec.lookup_text)
    if not norm:
        return {}

    phrase_to_id = await fetch_phrase_term_ids([norm])
    term_id = phrase_to_id.get(norm)
    if term_id is None:
        return {}

    return await _fetch_binary_role_scores(
        PostingTable.STUDIO, [term_id], restrict_movie_ids
    )


async def _execute_title_pattern(
    spec: EntityQuerySpec,
    restrict_movie_ids: set[int] | None,
) -> dict[int, float]:
    """Title pattern lookup — normalized LIKE match against the full
    title column on public.movie_card. Binary 1.0 scoring for every
    matching movie."""
    norm = normalize_string(spec.lookup_text)
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

    # Note: movie_card.title is stored in display form (preserves case
    # and diacritics). Matching via ILIKE handles the case dimension;
    # diacritic-insensitive matching would require an unaccent-indexed
    # column (tracked as a future improvement).
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
    elif spec.entity_type == EntityType.STUDIO:
        scores_by_movie = await _execute_studio(spec, restrict_to_movie_ids)
    elif spec.entity_type == EntityType.TITLE_PATTERN:
        scores_by_movie = await _execute_title_pattern(spec, restrict_to_movie_ids)
    else:
        # Exhaustive — EntityType has four values. Any future addition
        # should fail loudly here rather than silently return empty.
        raise ValueError(f"Unhandled entity_type: {spec.entity_type}")

    return build_endpoint_result(scores_by_movie, restrict_to_movie_ids)
