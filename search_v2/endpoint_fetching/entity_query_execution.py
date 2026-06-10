# Stage 3 entity-endpoint executor.
#
# Public entry point: `execute_entity_query(spec, *, restrict_to_movie_ids)`.
# Dispatches on spec type — PersonQuerySpec | CharacterQuerySpec |
# TitlePatternQuerySpec — and returns an EndpointResult with raw
# [0, 1] scores per matched movie_id. The restrict argument controls
# only output shape: None returns the natural match set;
# set[int] returns one entry per supplied ID with 0.0 for misses.
#
# The executor is direction-agnostic. Role and polarity live on the
# Trait that owns this call, are stamped onto the result by the
# handler, and are not part of this module's signature.
#
# Multi-target merge: each spec carries one or more targets. Per-target
# scoring is the same as the legacy single-entity logic. The executor
# fans across targets in parallel and takes MAX per movie — union
# semantics ("any of the targets wins on its strongest signal").
#
# Title-pattern is the one exception to fan-out: it issues a single
# OR'd LIKE query because per-pattern scoring is binary 1.0 and the
# DB does the cheap work of merging matches.
#
# Scoring criteria reference — finalized_search_proposal.md §Endpoint 1
# and search_improvement_planning/character_scoring_revamp.md:
#   - Non-actor non-character role persons (director, writer, producer,
#     composer) / title patterns → binary 1.0 match.
#   - Actor persons → zone-based prominence via billing_position +
#     cast_size; one of four PersonProminenceMode values. Raw [0, 1].
#   - Characters → billing-position prominence via CharacterProminenceMode
#     CENTRAL / DEFAULT against lex.inv_character_postings. Raw [0, 1].
#   - PersonCategory.UNKNOWN unions all five role tables with even
#     weight (no primary-table bias). MAX-merge across tables.
#
# Per-row scorers, per-target merges, and the public entry point all
# operate in raw [0, 1] space. No dealbreaker-floor compression is
# applied here — operation-type-aware band shaping lives upstream
# (or downstream of stage-3) and entity scoring stays a plain
# gradient regardless of candidate-generator vs pool-reranker mode.

from __future__ import annotations

import asyncio

from db.postgres import (
    PEOPLE_POSTING_TABLES,
    PostingTable,
    fetch_actor_billing_rows,
    fetch_character_billing_rows,
    fetch_character_strings_exact,
    fetch_movie_ids_by_term_ids,
    fetch_movie_ids_with_titles_matching_any,
    fetch_phrase_term_ids,
)
from implementation.classes.schemas import MetadataFilters
from implementation.misc.helpers import normalize_string
from implementation.misc.sql_like import escape_like
from schemas.endpoint_result import EndpointResult
from schemas.entity_translation import (
    CharacterProminenceMode,
    CharacterQuerySpec,
    CharacterTarget,
    PersonCategory,
    PersonProminenceMode,
    PersonQuerySpec,
    PersonTarget,
    TitleMatchType,
    TitlePatternQuerySpec,
    TitlePatternTarget,
)
# Zone primitives (cutoffs + in-zone relative position) live in a
# shared module so the person entity-flow bucketer in
# search_v2.person_search and the score curves here can't drift on
# tuning constants. See search_v2.actor_zones for the docstring.
from search_v2.actor_zones import (
    zone_cutoffs,
    zone_label,
    zone_relative_position,
)
from search_v2.endpoint_fetching.result_helpers import build_endpoint_result


# ---------------------------------------------------------------------------
# Form normalization — order-preserving dedupe of a target's `forms`
# list under the shared normalize_string contract. Empty / duplicate
# entries are dropped.
# ---------------------------------------------------------------------------


def _normalize_forms(forms: list[str]) -> list[str]:
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
#
# Zone cutoffs and the in-zone relative-position helper live in
# search_v2.actor_zones — they're shared with the actor entity-flow
# bucketer. The mode-specific score curves below stay here.
# ---------------------------------------------------------------------------


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


# Dispatch table for actor-table prominence modes. Keyed on
# PersonProminenceMode — every value is exhaustively covered by the
# enum so KeyError here would be a programmer error upstream.
_ACTOR_MODE_SCORERS = {
    PersonProminenceMode.DEFAULT: _actor_score_default,
    PersonProminenceMode.LEAD: _actor_score_lead,
    PersonProminenceMode.SUPPORTING: _actor_score_supporting,
    PersonProminenceMode.MINOR: _actor_score_minor,
}


def _actor_prominence_score(
    billing_position: int,
    cast_size: int,
    mode: PersonProminenceMode,
) -> float:
    """Score a single actor credit row under the given prominence mode.

    Uses the proposal's zone-based formulas with sqrt-adaptive cutoffs
    to compute a raw [0, 1] value. No band compression is applied —
    callers always receive raw scores.
    """
    # Zone boundaries and the zone label come from the shared
    # actor_zones primitives so this scorer and the attribute-search
    # band tiering can't drift on cutoffs. We still need `cutoffs` here
    # to anchor the in-zone relative position (zp) for the score curve.
    cutoffs = zone_cutoffs(cast_size)
    zone = zone_label(billing_position, cast_size)

    if zone == "lead":
        zp = zone_relative_position(billing_position, 1, cutoffs.lead_cutoff)
    elif zone == "supporting":
        zp = zone_relative_position(
            billing_position, cutoffs.lead_cutoff + 1, cutoffs.supp_cutoff
        )
    else:
        zp = zone_relative_position(
            billing_position, cutoffs.supp_cutoff + 1, cast_size
        )

    return _ACTOR_MODE_SCORERS[mode](zone, zp)


# ---------------------------------------------------------------------------
# Character prominence scoring (pure functions — no I/O).
#
# Two modes on CharacterProminenceMode:
#   CENTRAL — character is the subject of the query. Fixed decay
#     curve; cast_size-independent. "Spider-Man movies" scores the
#     same way whether the cast is 8 or 80.
#   DEFAULT — character is a filter with a gentle prominence
#     preference. Linear ramp from 1.0 at the top of the character
#     cast to 0.0 at the bottom.
#
# See search_improvement_planning/character_scoring_revamp.md
# §"Scoring Formulas" for the curve tables.
# ---------------------------------------------------------------------------


def _character_score_central(billing_position: int) -> float:
    """CENTRAL raw score — fixed decay, cast_size-independent.

    Positions 1 and 2 both score 1.0 (handles alias edges like
    "Peter Parker" / "Spider-Man" where the requested character
    appears second on its cast edge). Beyond position 2, decay 0.15
    per step until the raw value reaches 0.
    """
    return max(0.0, 1.0 - 0.15 * max(0, billing_position - 2))


def _character_score_default(billing_position: int, cast_size: int) -> float:
    """DEFAULT raw score — linear ramp across the character cast.

    1.0 at position 1, smoothly descending to 0.0 at position ==
    cast_size. The `max(1, size - 1)` guards against single-character
    casts where the naive denominator would be 0.
    """
    return 1.0 - (billing_position - 1) / max(1, cast_size - 1)


_CHARACTER_MODE_SCORERS = {
    CharacterProminenceMode.CENTRAL: lambda pos, size: _character_score_central(pos),
    CharacterProminenceMode.DEFAULT: _character_score_default,
}


def _character_prominence_score(
    billing_position: int,
    character_cast_size: int,
    mode: CharacterProminenceMode,
) -> float:
    """Score a single character credit row under the given mode.

    Dispatches to the mode-specific raw scorer and returns the raw
    [0, 1] value. No band compression is applied — callers always
    receive raw scores.
    """
    return _CHARACTER_MODE_SCORERS[mode](billing_position, character_cast_size)


# ---------------------------------------------------------------------------
# Per-table fetchers. Each returns {movie_id: score} for one set of
# resolved term_ids against one posting table. Within a single fetch,
# the per-movie score is the MAX across rows — covers same-target alias
# merging and incidental same-movie multiple-credit collisions.
# ---------------------------------------------------------------------------


async def _fetch_actor_scores(
    term_ids: list[int],
    mode: PersonProminenceMode,
    restrict_movie_ids: set[int] | None,
    *,
    metadata_filters: MetadataFilters | None = None,
) -> dict[int, float]:
    rows = await fetch_actor_billing_rows(
        term_ids, restrict_movie_ids, metadata_filters=metadata_filters,
    )
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
    *,
    metadata_filters: MetadataFilters | None = None,
) -> dict[int, float]:
    """Binary 1.0 scoring for non-actor / non-character role tables.

    Post-filters by restrict in Python — the matched set size for a
    single person is small enough that pulling it back and intersecting
    is cheaper than extending every posting helper to support a server-
    side restrict.

    The UI hard filter pushes down via fetch_movie_ids_by_term_ids'
    inline subquery (server-side).
    """
    movie_ids = await fetch_movie_ids_by_term_ids(
        table, term_ids, metadata_filters=metadata_filters,
    )
    if restrict_movie_ids is not None:
        movie_ids = movie_ids & restrict_movie_ids
    return {mid: 1.0 for mid in movie_ids}


async def _fetch_character_scores(
    term_ids: list[int],
    mode: CharacterProminenceMode,
    restrict_movie_ids: set[int] | None,
    *,
    metadata_filters: MetadataFilters | None = None,
) -> dict[int, float]:
    rows = await fetch_character_billing_rows(
        term_ids, restrict_movie_ids, metadata_filters=metadata_filters,
    )
    scores: dict[int, float] = {}
    for movie_id, billing_position, character_cast_size in rows:
        score = _character_prominence_score(
            billing_position, character_cast_size, mode
        )
        prev = scores.get(movie_id)
        if prev is None or score > prev:
            scores[movie_id] = score
    return scores


# ---------------------------------------------------------------------------
# Specific non-actor person_category → posting table. ACTOR is excluded
# (prominence path); UNKNOWN is excluded (fans across every table).
# ---------------------------------------------------------------------------

_PERSON_CATEGORY_TO_TABLE: dict[PersonCategory, PostingTable] = {
    PersonCategory.DIRECTOR: PostingTable.DIRECTOR,
    PersonCategory.WRITER: PostingTable.WRITER,
    PersonCategory.PRODUCER: PostingTable.PRODUCER,
    PersonCategory.COMPOSER: PostingTable.COMPOSER,
}


# ---------------------------------------------------------------------------
# MAX-merge helper — combines an arbitrary list of {movie_id: score}
# dicts under union semantics. Used both for UNKNOWN fan-out (across
# the five role tables) and for multi-target spec merging (across
# distinct entities in a single call).
# ---------------------------------------------------------------------------


def _max_merge(dicts: list[dict[int, float]]) -> dict[int, float]:
    merged: dict[int, float] = {}
    for d in dicts:
        for movie_id, score in d.items():
            prev = merged.get(movie_id)
            if prev is None or score > prev:
                merged[movie_id] = score
    return merged


# ---------------------------------------------------------------------------
# Per-target executors. Each runs the legacy single-entity logic for
# its target, returning {movie_id: score}.
# ---------------------------------------------------------------------------


async def _execute_person_target(
    target: PersonTarget,
    restrict_movie_ids: set[int] | None,
    *,
    metadata_filters: MetadataFilters | None = None,
) -> dict[int, float]:
    """One PersonTarget → {movie_id: score}. Branches on
    `person_category`: ACTOR uses prominence scoring; the four other
    specific roles binary 1.0; UNKNOWN unions all five tables with
    even weight and MAX-merges (no primary-table bias)."""
    normalized = _normalize_forms(target.forms)
    if not normalized:
        return {}
    phrase_to_id = await fetch_phrase_term_ids(normalized)
    term_ids = list({tid for tid in phrase_to_id.values()})
    if not term_ids:
        return {}

    if target.person_category == PersonCategory.ACTOR:
        return await _fetch_actor_scores(
            term_ids, target.prominence_mode, restrict_movie_ids,
            metadata_filters=metadata_filters,
        )

    if target.person_category == PersonCategory.UNKNOWN:
        # Fan out across every role table. Actor uses prominence
        # scoring against `target.prominence_mode`; the other four are
        # binary 1.0. MAX-merge with no per-table weighting — UNKNOWN
        # is the fallback for "we don't know what they're known for",
        # so every table contributes equally and the strongest signal
        # wins.
        actor_task = _fetch_actor_scores(
            term_ids, target.prominence_mode, restrict_movie_ids,
            metadata_filters=metadata_filters,
        )
        binary_tasks = [
            _fetch_binary_role_scores(
                table, term_ids, restrict_movie_ids,
                metadata_filters=metadata_filters,
            )
            for table in PEOPLE_POSTING_TABLES
            if table is not PostingTable.ACTOR
        ]
        per_table = await asyncio.gather(actor_task, *binary_tasks)
        return _max_merge(list(per_table))

    # Specific non-actor role: binary 1.0 against the role's table.
    # prominence_mode is read off but ignored — non-actor scoring is
    # row-presence, not billing-band, so a "wrong" mode pick by the LLM
    # has no downstream effect.
    table = _PERSON_CATEGORY_TO_TABLE.get(target.person_category)
    if table is None:
        # Exhaustive over PersonCategory — ACTOR / UNKNOWN are handled
        # above, the four remaining specific roles are in the dict.
        # A new enum value reaching here means the dispatch needs
        # updating; surface it loudly at the dispatch site rather than
        # KeyError-ing inside the dict access.
        raise ValueError(
            f"Unhandled person_category: {target.person_category!r}"
        )
    return await _fetch_binary_role_scores(
        table, term_ids, restrict_movie_ids,
        metadata_filters=metadata_filters,
    )


async def _execute_character_target(
    target: CharacterTarget,
    restrict_movie_ids: set[int] | None,
    *,
    metadata_filters: MetadataFilters | None = None,
) -> dict[int, float]:
    """One CharacterTarget → {movie_id: score}. Exact-match `forms`
    against lex.character_strings, then prominence-score each row via
    CENTRAL or DEFAULT and MAX per movie across alias rows."""
    normalized = _normalize_forms(target.forms)
    if not normalized:
        return {}
    phrase_to_id = await fetch_character_strings_exact(normalized)
    term_ids = list({tid for tid in phrase_to_id.values()})
    if not term_ids:
        return {}
    return await _fetch_character_scores(
        term_ids, target.prominence_mode, restrict_movie_ids,
        metadata_filters=metadata_filters,
    )


# ---------------------------------------------------------------------------
# Per-spec executors.
# ---------------------------------------------------------------------------


async def _execute_person_spec(
    spec: PersonQuerySpec,
    restrict_movie_ids: set[int] | None,
    *,
    metadata_filters: MetadataFilters | None = None,
) -> dict[int, float]:
    """Fan one coroutine per PersonTarget, MAX-merge per movie. Single-
    target specs collapse to a 1-element merge with no scoring drift."""
    per_target = await asyncio.gather(
        *(
            _execute_person_target(
                target, restrict_movie_ids,
                metadata_filters=metadata_filters,
            )
            for target in spec.targets
        )
    )
    return _max_merge(list(per_target))


async def _execute_character_spec(
    spec: CharacterQuerySpec,
    restrict_movie_ids: set[int] | None,
    *,
    metadata_filters: MetadataFilters | None = None,
) -> dict[int, float]:
    """Fan one coroutine per CharacterTarget, MAX-merge per movie."""
    per_target = await asyncio.gather(
        *(
            _execute_character_target(
                target, restrict_movie_ids,
                metadata_filters=metadata_filters,
            )
            for target in spec.targets
        )
    )
    return _max_merge(list(per_target))


def _build_title_like_pattern(target: TitlePatternTarget) -> str | None:
    """Normalize, escape, and wrap a TitlePatternTarget into a single
    LIKE pattern. Returns None when the pattern normalizes to empty
    so the caller can drop it from the OR list."""
    norm = normalize_string(target.pattern)
    if not norm:
        return None
    escaped = escape_like(norm)
    if target.match_type == TitleMatchType.STARTS_WITH:
        return f"{escaped}%"
    if target.match_type == TitleMatchType.EXACT_MATCH:
        # Once LIKE metacharacters are escaped, plain `LIKE 'foo'`
        # without wildcards collapses to equality.
        return escaped
    return f"%{escaped}%"  # CONTAINS


async def _execute_title_pattern_spec(
    spec: TitlePatternQuerySpec,
    restrict_movie_ids: set[int] | None,
    *,
    metadata_filters: MetadataFilters | None = None,
) -> dict[int, float]:
    """Single OR'd LIKE query across every target. Per-pattern scoring
    is binary 1.0, so union semantics collapse to set membership at
    the DB layer — no per-row identity tracking, no Python-side
    per-target merge."""
    patterns: list[str] = []
    for target in spec.targets:
        pattern = _build_title_like_pattern(target)
        if pattern is not None:
            patterns.append(pattern)
    if not patterns:
        return {}

    movie_ids = await fetch_movie_ids_with_titles_matching_any(
        patterns, restrict_movie_ids, metadata_filters=metadata_filters,
    )
    return {mid: 1.0 for mid in movie_ids}


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


# Union of every spec the entity-family executor accepts. Exported so
# callers (handler / orchestrator) can type their spec hand-off cleanly
# without re-deriving the union themselves. PEP 695 form so the alias
# is explicit to readers and type-checkers.
type EntitySpec = PersonQuerySpec | CharacterQuerySpec | TitlePatternQuerySpec


async def execute_entity_query(
    spec: EntitySpec,
    *,
    restrict_to_movie_ids: set[int] | None = None,
    metadata_filters: MetadataFilters | None = None,
) -> EndpointResult:
    """Execute one entity-family spec against the lexical schema.

    Single entry point for all three spec types. Dispatch is by
    isinstance — the LLM's category routing already picked the right
    spec family upstream, so we just match on it here.

    `restrict_to_movie_ids` controls only the output shape:
      - None → return one ScoredCandidate per naturally matched
        movie, non-matches omitted (the natural match set).
      - set[int] → return exactly one ScoredCandidate per supplied
        ID, with 0.0 for non-matches (the supplied pool, scored).

    Scoring is the same in both cases: raw [0, 1] gradient with
    no band compression. Per-row scorers (actor zone formulas,
    character billing decay, binary 1.0 for non-actor / title-
    pattern roles) emit raw values; per-target MAX-merge preserves
    them; the public entry point returns them unmodified.

    Returns an empty EndpointResult when forms / patterns normalize to
    nothing or when exact-match dictionary resolution finds nothing —
    "no match is a valid result" per the endpoint spec.

    Args:
        spec: Validated PersonQuerySpec, CharacterQuerySpec, or
            TitlePatternQuerySpec from the per-category handler LLM.
        restrict_to_movie_ids: Optional candidate-pool restriction.
            Pass a pool to get one entry per ID; omit to get the
            natural match set.
    """
    if isinstance(spec, PersonQuerySpec):
        scores_by_movie = await _execute_person_spec(
            spec, restrict_to_movie_ids,
            metadata_filters=metadata_filters,
        )
    elif isinstance(spec, CharacterQuerySpec):
        scores_by_movie = await _execute_character_spec(
            spec, restrict_to_movie_ids,
            metadata_filters=metadata_filters,
        )
    elif isinstance(spec, TitlePatternQuerySpec):
        scores_by_movie = await _execute_title_pattern_spec(
            spec, restrict_to_movie_ids,
            metadata_filters=metadata_filters,
        )
    else:
        # Exhaustive over the EntitySpec union — any new spec family
        # should fail loudly here rather than silently return empty.
        raise TypeError(
            f"Unsupported entity spec type: {type(spec).__name__}"
        )

    # Drop 0.0-score entries. Character scoring can produce organic
    # zeros (CENTRAL at billing_position >= 9, DEFAULT at the very
    # bottom of the cast); a "match" with no value contradicts the
    # natural-match-set contract. Safe in preference mode too — a
    # missing key in scores_by_movie produces ScoredCandidate(mid,
    # 0.0) via build_endpoint_result's .get(mid, 0.0), identical to
    # a 0.0-valued entry.
    scores_by_movie = {
        mid: score for mid, score in scores_by_movie.items() if score > 0.0
    }

    return build_endpoint_result(scores_by_movie, restrict_to_movie_ids)
