# Search V2 — Stage 3 Metadata Endpoint: Query Execution
#
# Runs one MetadataTranslationOutput against PostgreSQL movie_card and
# returns an EndpointResult of (movie_id, score ∈ [0, 1]) pairs. Supports
# two modes on one function:
#   - Dealbreaker mode (restrict_to_movie_ids=None): apply the widened
#     gate as a SQL WHERE clause and return every match with its score.
#     The returned list doubles as the dealbreaker's contribution to the
#     candidate pool (Phase 4a).
#   - Preference mode (restrict_to_movie_ids is a set of ids): ignore
#     the candidate gate, fetch attribute data for every supplied id,
#     and emit one ScoredCandidate per id (0.0 for null data or for
#     movies outside the grace window).
#
# All role-specific interpretation — direction, exclusion mode, weighting —
# lives in the orchestrator. This module is thin and uniform: run the
# query, compute raw per-movie scores, done.
#
# See search_improvement_planning/finalized_search_proposal.md
# (Step 3 → Endpoint 2 per-attribute specs, Step 3.5 Endpoint Return
# Shape) for the scoring contract.

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from collections.abc import Awaitable, Callable

from db.postgres import pool
from implementation.classes.countries import Country
from implementation.classes.enums import (
    DateMatchOperation,
    MaturityRating,
    NumericalMatchOperation,
    RatingMatchOperation,
    StreamingAccessType,
)
from implementation.classes.languages import Language
from implementation.classes.watch_providers import STREAMING_PROVIDER_MAP, StreamingService
from implementation.misc.helpers import create_watch_provider_offering_key
from schemas.endpoint_result import EndpointResult, ScoredCandidate
from schemas.enums import (
    BoxOfficeStatus,
    BudgetSize,
    MetadataAttribute,
    PopularityMode,
    ReceptionMode,
)
from schemas.metadata_translation import MetadataTranslationOutput
from search_v2.stage_3.result_helpers import compress_to_dealbreaker_floor

log = logging.getLogger(__name__)


# ── Gradient constants (mirrored from the proposal and db/metadata_scoring.py) ──

_SECONDS_PER_DAY = 86400

_DATE_GRACE_DAYS_MIN = 365.0
_DATE_GRACE_DAYS_MAX = 365.0 * 5
_DATE_GRACE_DAYS_UNBOUNDED = 365.0 * 3
_DATE_GRACE_DAYS_EXACT = 365.0 * 2

_RUNTIME_GRACE_MINUTES = 30.0

# Exponential decay for country-of-origin position:
#   score = exp(-(position - 1) / tau)
# tau ≈ 1.3 yields: pos1=1.00, pos2=0.46, pos3=0.21, pos4=0.10 — a steeper
# curve than the proposal's 0.7-0.8 hint at position 2, but closer to the
# "rapid decay toward 0.0" for positions 3+ that the user prioritized.
_COUNTRY_POSITION_TAU = 1.3

# Popularity and reception have no natural WHERE gate — a dealbreaker of
# "popular" or "well-received" would otherwise return every non-null movie.
# Cap the candidate pull at 5000 rows sorted by the scoring dimension, then
# drop zero-score movies in Python before returning so the candidate pool
# only contains movies the dealbreaker actually endorses. Applied ONLY in
# dealbreaker mode; other attributes filter naturally in SQL and are not
# capped. Preference mode never enters this path.
_POPULARITY_RECEPTION_DEALBREAKER_CAP = 5000

# Encoding for watch_offer_keys (see implementation/misc/helpers.py
# `create_watch_provider_offering_key`): upper 27 bits hold the provider_id,
# lower 4 bits hold the method_id — i.e. `key = (provider_id << 4) | method_id`.
# The 4-bit mask matches the encoding; narrowing it (e.g. to 2 bits) happens
# to work for the current method_id range (1-3) but silently misclassifies
# any future method with id >= 4.
_METHOD_ID_BITMASK = 0xF

_ALL_METHOD_IDS: tuple[int, ...] = tuple(m.type_id for m in StreamingAccessType)


# ── Public entry point ────────────────────────────────────────────────────

async def execute_metadata_query(
    output: MetadataTranslationOutput,
    restrict_to_movie_ids: set[int] | None = None,
) -> EndpointResult:
    """Execute one metadata query spec and return scored candidates.

    Dispatch is keyed off ``output.target_attribute``. One retry on transient
    failure; a second failure collapses to an empty result per the proposal's
    graceful-degradation rule.

    Args:
        output: The validated LLM translation output.
        restrict_to_movie_ids: None → dealbreaker mode (query the whole DB
            under the widened gate). A set → preference mode (score only
            those ids; missing data → 0.0; every supplied id gets an entry).

    Returns:
        EndpointResult whose ``scores`` list contains one ScoredCandidate
        per eligible movie. Empty list on non-recoverable failure.
    """
    handler = _DISPATCH.get(output.target_attribute)
    if handler is None:
        # Schema guarantees a valid MetadataAttribute, so this branch is
        # defensive only — log and degrade gracefully.
        log.error("metadata execution: unknown target_attribute %r", output.target_attribute)
        return EndpointResult(scores=[])

    for attempt in (1, 2):
        try:
            scored = await handler(output, restrict_to_movie_ids)
            return EndpointResult(scores=scored)
        except Exception:
            # One retry on any failure — typically transient pool/network.
            # Log both attempts; return empty if the retry also fails so the
            # assembly pipeline can keep working with contributions from other
            # endpoints.
            log.exception(
                "metadata execution failed on attempt %d for target=%s",
                attempt, output.target_attribute,
            )

    return EndpointResult(scores=[])


# ── Attribute dispatch ────────────────────────────────────────────────────

Handler = Callable[
    [MetadataTranslationOutput, "set[int] | None"],
    Awaitable[list[ScoredCandidate]],
]


# ── Shared SQL helpers ────────────────────────────────────────────────────

async def _fetch(query: str, params: list) -> list[tuple]:
    """Read-only helper — local wrapper so handlers don't reach into
    db.postgres internals directly. Centralizes the connection pattern."""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, params)
            return await cur.fetchall()


def _restrict_clause(params: list, restrict: set[int] | None) -> str:
    """Append the ``movie_id = ANY(%s)`` clause to params when in preference
    mode; return the SQL fragment (empty string in dealbreaker mode)."""
    if restrict is None:
        return ""
    params.append(list(restrict))
    return "movie_id = ANY(%s)"


def _combine_where(*fragments: str) -> str:
    """Join non-empty WHERE fragments with AND. Empty input → ``WHERE TRUE``."""
    parts = [f for f in fragments if f]
    return "WHERE " + " AND ".join(parts) if parts else "WHERE TRUE"


def _missing_ids_zero_scored(
    restrict: set[int] | None, covered: set[int]
) -> list[ScoredCandidate]:
    """In preference mode, emit score 0.0 for every supplied id not already
    covered by the handler's own output (null data or outside the gate).
    Returns empty list in dealbreaker mode."""
    if restrict is None:
        return []
    return [ScoredCandidate(movie_id=mid, score=0.0) for mid in restrict - covered]


def _date_to_unix_ts(date_str: str) -> float:
    """YYYY-MM-DD → UTC unix timestamp in seconds."""
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()


# ── Release date ──────────────────────────────────────────────────────────

async def _handle_release_date(
    output: MetadataTranslationOutput,
    restrict: set[int] | None,
) -> list[ScoredCandidate]:
    rd = output.release_date
    if rd is None:
        return []

    op = DateMatchOperation(rd.match_operation)
    first_ts = _date_to_unix_ts(rd.first_date)
    second_ts = _date_to_unix_ts(rd.second_date) if rd.second_date else None
    lo, hi, grace_days = _precompute_date_window(op, first_ts, second_ts)
    grace_seconds = grace_days * _SECONDS_PER_DAY

    params: list = []
    if restrict is None:
        # Dealbreaker mode: apply the widened gate in SQL so only movies
        # inside (literal ± grace) come back.
        fragments = ["release_ts IS NOT NULL"]
        if lo != float("-inf"):
            fragments.append("release_ts >= %s")
            params.append(int(lo - grace_seconds))
        if hi != float("inf"):
            fragments.append("release_ts <= %s")
            params.append(int(hi + grace_seconds))
    else:
        # Preference mode: fetch every supplied id; score formula handles
        # null + out-of-range cases by returning 0.0.
        fragments = [_restrict_clause(params, restrict)]

    query = f"SELECT movie_id, release_ts FROM movie_card {_combine_where(*fragments)}"
    rows = await _fetch(query, params)

    scored: list[ScoredCandidate] = []
    covered: set[int] = set()
    for movie_id, ts in rows:
        score = _score_release_date(ts, lo, hi, grace_days)
        # Dealbreaker-floor compression: the SQL gate already restricts
        # rows to the ±grace window, so every score here is ≥ 0.0 and
        # gets lifted into [0.5, 1.0]. In-window → 1.0, grace-edge → 0.5.
        # Preference path keeps the raw linear decay so ranking can
        # distinguish "just outside the window" from "dead center".
        if restrict is None:
            score = compress_to_dealbreaker_floor(score)
        scored.append(ScoredCandidate(movie_id=int(movie_id), score=score))
        covered.add(int(movie_id))

    scored.extend(_missing_ids_zero_scored(restrict, covered))
    return scored


def _precompute_date_window(
    op: DateMatchOperation, first_ts: float, second_ts: float | None,
) -> tuple[float, float, float]:
    if op == DateMatchOperation.BETWEEN and second_ts is not None:
        lo, hi = min(first_ts, second_ts), max(first_ts, second_ts)
        range_width_days = (hi - lo) / _SECONDS_PER_DAY
        grace = max(_DATE_GRACE_DAYS_MIN, min(range_width_days * 0.5, _DATE_GRACE_DAYS_MAX))
        return lo, hi, grace
    if op == DateMatchOperation.AFTER:
        return first_ts, float("inf"), _DATE_GRACE_DAYS_UNBOUNDED
    if op == DateMatchOperation.BEFORE:
        return float("-inf"), first_ts, _DATE_GRACE_DAYS_UNBOUNDED
    # EXACT — single-point range, symmetric grace on each side.
    return first_ts, first_ts, _DATE_GRACE_DAYS_EXACT


def _score_release_date(ts: int | None, lo: float, hi: float, grace_days: float) -> float:
    if ts is None:
        return 0.0
    if lo <= ts <= hi:
        return 1.0
    distance_days = (lo - ts) / _SECONDS_PER_DAY if ts < lo else (ts - hi) / _SECONDS_PER_DAY
    return max(0.0, 1.0 - distance_days / grace_days)


# ── Runtime ───────────────────────────────────────────────────────────────

async def _handle_runtime(
    output: MetadataTranslationOutput,
    restrict: set[int] | None,
) -> list[ScoredCandidate]:
    rt = output.runtime
    if rt is None:
        return []

    op = NumericalMatchOperation(rt.match_operation)
    lo, hi = _precompute_runtime_window(op, rt.first_value, rt.second_value)

    params: list = []
    if restrict is None:
        fragments = ["runtime_minutes IS NOT NULL"]
        if lo != float("-inf"):
            fragments.append("runtime_minutes >= %s")
            params.append(int(lo - _RUNTIME_GRACE_MINUTES))
        if hi != float("inf"):
            fragments.append("runtime_minutes <= %s")
            params.append(int(hi + _RUNTIME_GRACE_MINUTES))
    else:
        fragments = [_restrict_clause(params, restrict)]

    query = f"SELECT movie_id, runtime_minutes FROM movie_card {_combine_where(*fragments)}"
    rows = await _fetch(query, params)

    scored: list[ScoredCandidate] = []
    covered: set[int] = set()
    for movie_id, runtime in rows:
        score = _score_runtime(runtime, lo, hi)
        # Dealbreaker-floor compression: SQL gate enforces ±30-minute
        # grace, so every row's raw score is ≥ 0.0. In-window → 1.0,
        # ±30-min edge → 0.5. Preference path keeps the linear ramp.
        if restrict is None:
            score = compress_to_dealbreaker_floor(score)
        scored.append(ScoredCandidate(movie_id=int(movie_id), score=score))
        covered.add(int(movie_id))

    scored.extend(_missing_ids_zero_scored(restrict, covered))
    return scored


def _precompute_runtime_window(
    op: NumericalMatchOperation, first_value: float, second_value: float | None,
) -> tuple[float, float]:
    if op == NumericalMatchOperation.BETWEEN and second_value is not None:
        return min(first_value, second_value), max(first_value, second_value)
    if op == NumericalMatchOperation.GREATER_THAN:
        return first_value, float("inf")
    if op == NumericalMatchOperation.GREATER_THAN_OR_EQUAL:
        return first_value, float("inf")
    if op == NumericalMatchOperation.LESS_THAN:
        return float("-inf"), first_value
    if op == NumericalMatchOperation.LESS_THAN_OR_EQUAL:
        return float("-inf"), first_value
    # EXACT
    return first_value, first_value


def _score_runtime(runtime: int | None, lo: float, hi: float) -> float:
    if runtime is None:
        return 0.0
    if lo <= runtime <= hi:
        return 1.0
    distance = lo - runtime if runtime < lo else runtime - hi
    return max(0.0, 1.0 - distance / _RUNTIME_GRACE_MINUTES)


# ── Maturity rating ───────────────────────────────────────────────────────

async def _handle_maturity(
    output: MetadataTranslationOutput,
    restrict: set[int] | None,
) -> list[ScoredCandidate]:
    mt = output.maturity_rating
    if mt is None:
        return []

    target = MaturityRating(mt.rating)
    op = RatingMatchOperation(mt.match_operation)
    unrated_rank = MaturityRating.UNRATED.maturity_rank
    is_exact_unrated = target == MaturityRating.UNRATED and op == RatingMatchOperation.EXACT
    lo, hi = _precompute_rank_window(op, target.maturity_rank)

    params: list = []
    if restrict is None:
        if is_exact_unrated:
            # Only UNRATED movies qualify.
            fragments = ["maturity_rank = %s"]
            params.append(unrated_rank)
        else:
            # Include the 0.5 fringe (distance 1) in the candidate gate,
            # and exclude UNRATED entirely per the proposal's rule.
            gate_lo = max(1, lo - 1)
            gate_hi = min(5, hi + 1)
            fragments = [
                "maturity_rank IS NOT NULL",
                "maturity_rank <> %s",
                "maturity_rank BETWEEN %s AND %s",
            ]
            params.extend([unrated_rank, gate_lo, gate_hi])
    else:
        fragments = [_restrict_clause(params, restrict)]

    query = f"SELECT movie_id, maturity_rank FROM movie_card {_combine_where(*fragments)}"
    rows = await _fetch(query, params)

    scored: list[ScoredCandidate] = []
    covered: set[int] = set()
    for movie_id, rank in rows:
        score = _score_maturity(rank, target, lo, hi)
        scored.append(ScoredCandidate(movie_id=int(movie_id), score=score))
        covered.add(int(movie_id))

    scored.extend(_missing_ids_zero_scored(restrict, covered))
    return scored


def _precompute_rank_window(op: RatingMatchOperation, target_rank: int) -> tuple[int, int]:
    # 1..5 scale. Degenerate windows (e.g., LESS_THAN G) fall through with
    # an empty range and score 0.0 for everyone; harmless.
    if op == RatingMatchOperation.EXACT:
        return target_rank, target_rank
    if op == RatingMatchOperation.GREATER_THAN:
        return target_rank + 1, 5
    if op == RatingMatchOperation.LESS_THAN:
        return 1, target_rank - 1
    if op == RatingMatchOperation.GREATER_THAN_OR_EQUAL:
        return target_rank, 5
    if op == RatingMatchOperation.LESS_THAN_OR_EQUAL:
        return 1, target_rank
    return target_rank, target_rank


def _score_maturity(
    movie_rank: int | None, target: MaturityRating, lo: int, hi: int,
) -> float:
    if movie_rank is None:
        return 0.0
    unrated_rank = MaturityRating.UNRATED.maturity_rank
    # UNRATED matches only an EXACT-UNRATED query; otherwise score is 0.0
    # even if the orchestrator handed us an UNRATED candidate in preference mode.
    if movie_rank == unrated_rank:
        return 1.0 if target == MaturityRating.UNRATED else 0.0
    if target == MaturityRating.UNRATED:
        return 0.0
    if lo <= movie_rank <= hi:
        return 1.0
    distance = min(abs(movie_rank - lo), abs(movie_rank - hi))
    return 0.5 if distance == 1 else 0.0


# ── Streaming availability ───────────────────────────────────────────────

async def _handle_streaming(
    output: MetadataTranslationOutput,
    restrict: set[int] | None,
) -> list[ScoredCandidate]:
    sm = output.streaming
    if sm is None:
        return []

    any_method_keys, desired_method_keys, access_type_id = _precompute_streaming_keys(
        [StreamingService(s) for s in sm.services],
        StreamingAccessType(sm.preferred_access_type) if sm.preferred_access_type else None,
    )

    params: list = []
    if restrict is None:
        if any_method_keys:
            # Services specified: GIN overlap is the tight candidate gate.
            fragments = [
                "watch_offer_keys IS NOT NULL",
                "watch_offer_keys && %s",
            ]
            params.append(list(any_method_keys))
        else:
            # Access-type-only dealbreaker: no GIN-friendly gate exists
            # because every provider carries that method. Use an EXISTS on
            # the unnested array checking the method-id bits. This is a
            # full-table scan and is an accepted tradeoff given how rare
            # pure access-type dealbreakers should be in practice.
            fragments = [
                "watch_offer_keys IS NOT NULL",
                "EXISTS (SELECT 1 FROM unnest(watch_offer_keys) k WHERE (k & %s) = %s)",
            ]
            params.extend([_METHOD_ID_BITMASK, access_type_id])
    else:
        fragments = [_restrict_clause(params, restrict)]

    query = f"SELECT movie_id, watch_offer_keys FROM movie_card {_combine_where(*fragments)}"
    rows = await _fetch(query, params)

    scored: list[ScoredCandidate] = []
    covered: set[int] = set()
    for movie_id, offer_keys in rows:
        score = _score_streaming(offer_keys, any_method_keys, desired_method_keys, access_type_id)
        scored.append(ScoredCandidate(movie_id=int(movie_id), score=score))
        covered.add(int(movie_id))

    scored.extend(_missing_ids_zero_scored(restrict, covered))
    return scored


def _precompute_streaming_keys(
    services: list[StreamingService],
    preferred_access: StreamingAccessType | None,
) -> tuple[set[int], set[int], int | None]:
    """Return (any-method key set, desired-method key set, standalone access_type_id).

    Mirrors db/metadata_scoring.py._precompute_watch_providers: when services
    are provided, the preferred access type is baked into desired_method_keys
    and the standalone access_type_id is left None; when services are empty,
    the standalone id is used for the EXISTS-on-unnest scoring path.
    """
    any_keys: set[int] = set()
    desired_keys: set[int] = set()
    for svc in services:
        for pid in STREAMING_PROVIDER_MAP.get(svc, []):
            for mid in _ALL_METHOD_IDS:
                any_keys.add(create_watch_provider_offering_key(pid, mid))
            if preferred_access is not None:
                desired_keys.add(create_watch_provider_offering_key(pid, preferred_access.type_id))

    standalone_access_id = (
        preferred_access.type_id if (preferred_access and not services) else None
    )
    return any_keys, desired_keys, standalone_access_id


def _score_streaming(
    offer_keys: list[int] | None,
    any_method_keys: set[int],
    desired_method_keys: set[int],
    access_type_id: int | None,
) -> float:
    if not offer_keys:
        return 0.0
    key_set = set(offer_keys)

    # Services + access_type: desired match = 1.0, any match = 0.5.
    if any_method_keys and desired_method_keys:
        if key_set & desired_method_keys:
            return 1.0
        return 0.5 if key_set & any_method_keys else 0.0

    # Services only.
    if any_method_keys:
        return 1.0 if key_set & any_method_keys else 0.0

    # Access type only.
    if access_type_id is not None:
        methods_present = {k & _METHOD_ID_BITMASK for k in key_set}
        return 1.0 if access_type_id in methods_present else 0.0

    return 0.0


# ── Audio language ────────────────────────────────────────────────────────

async def _handle_audio_language(
    output: MetadataTranslationOutput,
    restrict: set[int] | None,
) -> list[ScoredCandidate]:
    al = output.audio_language
    if al is None or not al.languages:
        return []

    include_ids = [Language(v).language_id for v in al.languages]

    params: list = []
    if restrict is None:
        fragments = ["audio_language_ids && %s"]
        params.append(include_ids)
    else:
        fragments = [_restrict_clause(params, restrict)]

    query = f"SELECT movie_id, audio_language_ids FROM movie_card {_combine_where(*fragments)}"
    rows = await _fetch(query, params)

    include_set = set(include_ids)
    scored: list[ScoredCandidate] = []
    covered: set[int] = set()
    for movie_id, lang_ids in rows:
        score = 1.0 if lang_ids and (set(lang_ids) & include_set) else 0.0
        scored.append(ScoredCandidate(movie_id=int(movie_id), score=score))
        covered.add(int(movie_id))

    scored.extend(_missing_ids_zero_scored(restrict, covered))
    return scored


# ── Country of origin ─────────────────────────────────────────────────────

async def _handle_country_of_origin(
    output: MetadataTranslationOutput,
    restrict: set[int] | None,
) -> list[ScoredCandidate]:
    co = output.country_of_origin
    if co is None or not co.countries:
        return []

    include_ids = [Country(v).country_id for v in co.countries]
    include_set = set(include_ids)

    params: list = []
    if restrict is None:
        fragments = ["country_of_origin_ids && %s"]
        params.append(include_ids)
    else:
        fragments = [_restrict_clause(params, restrict)]

    query = f"SELECT movie_id, country_of_origin_ids FROM movie_card {_combine_where(*fragments)}"
    rows = await _fetch(query, params)

    scored: list[ScoredCandidate] = []
    covered: set[int] = set()
    for movie_id, country_ids in rows:
        # Dealbreaker path uses a discrete-position rule (pos 1 → 1.0,
        # pos 2 → 0.5, anything else → drop) so movies that only match
        # at IMDB's tail positions don't dilute the candidate pool — the
        # GIN `&&` gate above admits any overlap, but downstream we want
        # only prominently-attributed matches to land in the
        # dealbreaker band. Preference path keeps the exponential decay
        # so ranking has a continuous gradient across all positions.
        if restrict is None:
            score = _score_country_position_dealbreaker(country_ids, include_set)
            if score == 0.0:
                continue
        else:
            score = _score_country_position(country_ids, include_set)
        scored.append(ScoredCandidate(movie_id=int(movie_id), score=score))
        covered.add(int(movie_id))

    scored.extend(_missing_ids_zero_scored(restrict, covered))
    return scored


def _score_country_position(
    movie_country_ids: list[int] | None, include_set: set[int],
) -> float:
    # The stored array is ordered by IMDB's relevance. Best (lowest-index,
    # i.e. most relevant) position among the requested countries wins.
    if not movie_country_ids:
        return 0.0
    best_position: int | None = None
    for idx, cid in enumerate(movie_country_ids, start=1):
        if cid in include_set:
            if best_position is None or idx < best_position:
                best_position = idx
    if best_position is None:
        return 0.0
    return math.exp(-(best_position - 1) / _COUNTRY_POSITION_TAU)


def _score_country_position_dealbreaker(
    movie_country_ids: list[int] | None, include_set: set[int],
) -> float:
    # Discrete-position dealbreaker scorer. Positions are 1-indexed to
    # match `_score_country_position`'s enumerate(..., start=1).
    #   pos 1 → 1.0, pos 2 → 0.5, pos ≥ 3 (or no match) → 0.0 (dropped).
    # Only the first two array slots matter, so bail out of the scan as
    # soon as we either find a match or exhaust them.
    if not movie_country_ids:
        return 0.0
    for idx, cid in enumerate(movie_country_ids[:2], start=1):
        if cid in include_set:
            return 1.0 if idx == 1 else 0.5
    return 0.0


# ── Bucket-binary attributes (budget, box office) ─────────────────────────

async def _handle_budget_scale(
    output: MetadataTranslationOutput,
    restrict: set[int] | None,
) -> list[ScoredCandidate]:
    return await _handle_bucket(
        output,
        restrict,
        bucket_column="budget_bucket",
        target_value=output.budget_scale if output.budget_scale is not None else None,
        enum_cls=BudgetSize,
    )


async def _handle_box_office(
    output: MetadataTranslationOutput,
    restrict: set[int] | None,
) -> list[ScoredCandidate]:
    return await _handle_bucket(
        output,
        restrict,
        bucket_column="box_office_bucket",
        target_value=output.box_office if output.box_office is not None else None,
        enum_cls=BoxOfficeStatus,
    )


async def _handle_bucket(
    output: MetadataTranslationOutput,
    restrict: set[int] | None,
    *,
    bucket_column: str,
    target_value: str | None,
    enum_cls: type,  # BudgetSize or BoxOfficeStatus (StrEnum)
) -> list[ScoredCandidate]:
    # target_value arrives as a plain string (use_enum_values=True on the
    # schema). Validate it against the enum so we never interpolate a foreign
    # value into the WHERE clause — even though it's still parameterized.
    if target_value is None:
        return []
    target = enum_cls(target_value)

    params: list = [str(target.value)]
    if restrict is None:
        fragments = [f"{bucket_column} = %s"]
    else:
        fragments = [f"{bucket_column} = %s", _restrict_clause(params, restrict)]

    query = f"SELECT movie_id FROM movie_card {_combine_where(*fragments)}"
    rows = await _fetch(query, params)

    scored: list[ScoredCandidate] = []
    covered: set[int] = set()
    for (movie_id,) in rows:
        scored.append(ScoredCandidate(movie_id=int(movie_id), score=1.0))
        covered.add(int(movie_id))

    # Preference mode: every supplied id that didn't match the bucket gets 0.0.
    scored.extend(_missing_ids_zero_scored(restrict, covered))
    return scored


# ── Popularity ────────────────────────────────────────────────────────────

async def _handle_popularity(
    output: MetadataTranslationOutput,
    restrict: set[int] | None,
) -> list[ScoredCandidate]:
    if output.popularity is None:
        return []
    mode = PopularityMode(output.popularity)

    # Dealbreaker mode: sort by the scoring dimension, pull up to 5000 rows,
    # score them, then drop zero-score movies before returning so the pool
    # only contains movies the dealbreaker actually endorses.
    # Preference mode: score every supplied id; zero-score entries are kept
    # because the orchestrator expects one entry per supplied id.
    if restrict is None:
        direction = "DESC" if mode == PopularityMode.POPULAR else "ASC"
        query = (
            "SELECT movie_id, popularity_score FROM movie_card "
            "WHERE popularity_score IS NOT NULL "
            f"ORDER BY popularity_score {direction} LIMIT %s"
        )
        rows = await _fetch(query, [_POPULARITY_RECEPTION_DEALBREAKER_CAP])
    else:
        query = (
            "SELECT movie_id, popularity_score FROM movie_card "
            "WHERE movie_id = ANY(%s)"
        )
        rows = await _fetch(query, [list(restrict)])

    scored: list[ScoredCandidate] = []
    covered: set[int] = set()
    for movie_id, pop in rows:
        score = _score_popularity(pop, mode)
        if restrict is None and score == 0.0:
            continue
        # Dealbreaker-floor compression happens AFTER the zero-drop so
        # non-matches don't get promoted to 0.5. The surviving [>0, 1]
        # range is lifted into (0.5, 1.0] to sit in the
        # dealbreaker-eligible band. Preference path keeps raw popularity
        # so its ranking gradient spans the full [0, 1] with 0.0 for
        # movies missing popularity_score.
        if restrict is None:
            score = compress_to_dealbreaker_floor(score)
        scored.append(ScoredCandidate(movie_id=int(movie_id), score=score))
        covered.add(int(movie_id))

    scored.extend(_missing_ids_zero_scored(restrict, covered))
    return scored


def _score_popularity(pop: float | None, mode: PopularityMode) -> float:
    if pop is None:
        return 0.0
    pop = max(0.0, min(1.0, float(pop)))
    return pop if mode == PopularityMode.POPULAR else (1.0 - pop)


# ── Reception ─────────────────────────────────────────────────────────────

async def _handle_reception(
    output: MetadataTranslationOutput,
    restrict: set[int] | None,
) -> list[ScoredCandidate]:
    if output.reception is None:
        return []
    mode = ReceptionMode(output.reception)

    # Dealbreaker mode: sort by reception, pull up to 5000 rows, score, then
    # drop zero-score movies so the pool only contains movies the dealbreaker
    # actually endorses (reception ≤ 55 scores 0 for WELL_RECEIVED; reception
    # ≥ 50 scores 0 for POORLY_RECEIVED — those are the movies dropped here).
    # Preference mode keeps zero-score entries — one entry per supplied id.
    if restrict is None:
        direction = "DESC" if mode == ReceptionMode.WELL_RECEIVED else "ASC"
        query = (
            "SELECT movie_id, reception_score FROM movie_card "
            "WHERE reception_score IS NOT NULL "
            f"ORDER BY reception_score {direction} LIMIT %s"
        )
        rows = await _fetch(query, [_POPULARITY_RECEPTION_DEALBREAKER_CAP])
    else:
        query = (
            "SELECT movie_id, reception_score FROM movie_card "
            "WHERE movie_id = ANY(%s)"
        )
        rows = await _fetch(query, [list(restrict)])

    scored: list[ScoredCandidate] = []
    covered: set[int] = set()
    for movie_id, reception in rows:
        score = _score_reception(reception, mode)
        if restrict is None and score == 0.0:
            continue
        # Dealbreaker-floor compression happens AFTER the zero-drop so
        # "below the sliding cut-off" movies stay at 0.0 rather than
        # getting promoted to 0.5. Preference path keeps the raw linear
        # mapping for full ranking resolution.
        if restrict is None:
            score = compress_to_dealbreaker_floor(score)
        scored.append(ScoredCandidate(movie_id=int(movie_id), score=score))
        covered.add(int(movie_id))

    scored.extend(_missing_ids_zero_scored(restrict, covered))
    return scored


def _score_reception(reception: float | None, mode: ReceptionMode) -> float:
    if reception is None:
        return 0.0
    if mode == ReceptionMode.WELL_RECEIVED:
        return max(0.0, min(1.0, (reception - 55) / 40))
    return max(0.0, min(1.0, (50 - reception) / 40))


# ── Dispatch table (declared after handlers so names resolve) ─────────────

_DISPATCH: dict[MetadataAttribute, Handler] = {
    MetadataAttribute.RELEASE_DATE: _handle_release_date,
    MetadataAttribute.RUNTIME: _handle_runtime,
    MetadataAttribute.MATURITY_RATING: _handle_maturity,
    MetadataAttribute.STREAMING: _handle_streaming,
    MetadataAttribute.AUDIO_LANGUAGE: _handle_audio_language,
    MetadataAttribute.COUNTRY_OF_ORIGIN: _handle_country_of_origin,
    MetadataAttribute.BUDGET_SCALE: _handle_budget_scale,
    MetadataAttribute.BOX_OFFICE: _handle_box_office,
    MetadataAttribute.POPULARITY: _handle_popularity,
    MetadataAttribute.RECEPTION: _handle_reception,
}
