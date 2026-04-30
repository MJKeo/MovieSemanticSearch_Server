# Metadata Endpoint: Query Execution
#
# Runs one MetadataTranslationOutput against PostgreSQL movie_card and
# returns an EndpointResult of (movie_id, score ∈ [0, 1]) pairs. Handles
# the two operating modes uniformly:
#   - Dealbreaker mode (restrict_to_movie_ids=None): apply per-column
#     gates joined by OR (any single column qualifies a movie for the
#     candidate pool), score every returned row per populated column,
#     fold those scores into one combined score using combine_mode,
#     drop zero-combined movies, and lift survivors into the
#     dealbreaker floor band [0.5, 1.0].
#   - Preference mode (restrict_to_movie_ids is a set of ids): pull
#     every supplied id, score per populated column, fold via
#     combine_mode. Raw [0, 1] scores are preserved (no compression);
#     missing ids get 0.0 via build_endpoint_result.
#
# The new schema lets the LLM commit a single ColumnSpec where each of
# the ten attribute columns is independently null-or-populated, plus a
# combine_mode (max | average). Multi-column composition happens here:
# one SQL per call regardless of column count, single round-trip,
# per-column scoring, then folding. See schemas/metadata_translation.py
# for the schema itself and the conversation log for the full design
# rationale.

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any

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
from schemas.endpoint_result import EndpointResult
from schemas.enums import (
    BoxOfficeStatus,
    BudgetSize,
    MetadataAttribute,
    PopularityMode,
    ReceptionMode,
)
from schemas.metadata_translation import (
    AudioLanguageTranslation,
    ColumnCombineMode,
    ColumnSpec,
    CountryOfOriginTranslation,
    MaturityRatingTranslation,
    MetadataTranslationOutput,
    ReleaseDateTranslation,
    RuntimeTranslation,
    StreamingTranslation,
)
from search_v2.endpoint_fetching.result_helpers import (
    build_endpoint_result,
    compress_to_dealbreaker_floor,
)

log = logging.getLogger(__name__)


# ── Gradient / window constants ───────────────────────────────────────────

_SECONDS_PER_DAY = 86400

_DATE_GRACE_DAYS_MIN = 365.0
_DATE_GRACE_DAYS_MAX = 365.0 * 5
_DATE_GRACE_DAYS_UNBOUNDED = 365.0 * 3
_DATE_GRACE_DAYS_EXACT = 365.0 * 2

_RUNTIME_GRACE_MINUTES = 30.0

# Exponential decay for country-of-origin position (preference path):
#   score = exp(-(position - 1) / tau)
# tau ≈ 1.3 yields: pos1=1.00, pos2=0.46, pos3=0.21, pos4=0.10.
_COUNTRY_POSITION_TAU = 1.3

# Popularity / reception have no natural WHERE gate. When they're the
# ONLY populated columns in a dealbreaker call, we use ORDER BY +
# LIMIT to bound the candidate pool. When mixed with bounded columns,
# the bounded gates do the candidate-narrowing and these columns
# become free SELECT'd values for scoring (no LIMIT cap applied).
_POPULARITY_RECEPTION_DEALBREAKER_CAP = 5000

# Encoding for watch_offer_keys (see implementation/misc/helpers.py
# `create_watch_provider_offering_key`): upper 27 bits hold the
# provider_id, lower 4 bits hold the method_id. The 4-bit mask matches
# the encoding.
_METHOD_ID_BITMASK = 0xF

_ALL_METHOD_IDS: tuple[int, ...] = tuple(m.type_id for m in StreamingAccessType)


# ── Shared SQL helper ─────────────────────────────────────────────────────


async def _fetch(query: str, params: list) -> list[tuple]:
    """Read-only helper. Centralizes the connection pattern so handlers
    don't reach into db.postgres internals directly."""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, params)
            return await cur.fetchall()


def _date_to_unix_ts(date_str: str) -> float:
    """YYYY-MM-DD → UTC unix timestamp in seconds."""
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()


# ── Per-column window precomputation ──────────────────────────────────────


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


def _precompute_runtime_window(
    op: NumericalMatchOperation, first_value: float, second_value: float | None,
) -> tuple[float, float]:
    if op == NumericalMatchOperation.BETWEEN and second_value is not None:
        return min(first_value, second_value), max(first_value, second_value)
    if op in (NumericalMatchOperation.GREATER_THAN, NumericalMatchOperation.GREATER_THAN_OR_EQUAL):
        return first_value, float("inf")
    if op in (NumericalMatchOperation.LESS_THAN, NumericalMatchOperation.LESS_THAN_OR_EQUAL):
        return float("-inf"), first_value
    # EXACT
    return first_value, first_value


def _precompute_rank_window(op: RatingMatchOperation, target_rank: int) -> tuple[int, int]:
    # 1..5 scale. Degenerate windows (e.g., LESS_THAN G) fall through
    # with an empty range and score 0.0 for everyone; harmless.
    if op == RatingMatchOperation.GREATER_THAN:
        return target_rank + 1, 5
    if op == RatingMatchOperation.LESS_THAN:
        return 1, target_rank - 1
    if op == RatingMatchOperation.GREATER_THAN_OR_EQUAL:
        return target_rank, 5
    if op == RatingMatchOperation.LESS_THAN_OR_EQUAL:
        return 1, target_rank
    # EXACT
    return target_rank, target_rank


def _precompute_streaming_keys(
    services: list[StreamingService],
    preferred_access: StreamingAccessType | None,
) -> tuple[set[int], set[int], int | None]:
    """Return (any-method key set, desired-method key set, standalone access_type_id).

    When services are provided, the preferred access type is baked into
    desired_method_keys and the standalone access_type_id is left None.
    When services are empty, the standalone id is used for the
    EXISTS-on-unnest scoring path.
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


# ── Per-column scoring primitives ────────────────────────────────────────
# Each takes the row's raw column value plus precomputed scoring state
# and returns a [0, 1] score. No compression here — that's applied once
# after combine_mode folds per-column scores.


def _score_release_date(ts: int | None, lo: float, hi: float, grace_days: float) -> float:
    if ts is None:
        return 0.0
    if lo <= ts <= hi:
        return 1.0
    distance_days = (lo - ts) / _SECONDS_PER_DAY if ts < lo else (ts - hi) / _SECONDS_PER_DAY
    return max(0.0, 1.0 - distance_days / grace_days)


def _score_runtime(runtime: int | None, lo: float, hi: float) -> float:
    if runtime is None:
        return 0.0
    if lo <= runtime <= hi:
        return 1.0
    distance = lo - runtime if runtime < lo else runtime - hi
    return max(0.0, 1.0 - distance / _RUNTIME_GRACE_MINUTES)


def _score_maturity(
    movie_rank: int | None, target: MaturityRating, lo: int, hi: int,
) -> float:
    if movie_rank is None:
        return 0.0
    unrated_rank = MaturityRating.UNRATED.maturity_rank
    # UNRATED matches only an EXACT-UNRATED query.
    if movie_rank == unrated_rank:
        return 1.0 if target == MaturityRating.UNRATED else 0.0
    if target == MaturityRating.UNRATED:
        return 0.0
    if lo <= movie_rank <= hi:
        return 1.0
    distance = min(abs(movie_rank - lo), abs(movie_rank - hi))
    return 0.5 if distance == 1 else 0.0


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


def _score_country_position_preference(
    movie_country_ids: list[int] | None, include_set: set[int],
) -> float:
    # Exponential decay across all positions — preference path keeps a
    # smooth gradient so ranking distinguishes prominent vs. tail-position
    # matches.
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
    # Discrete-position dealbreaker scorer:
    #   pos 1 → 1.0, pos 2 → 0.33, pos ≥ 3 (or no match) → 0.0.
    # The raw 0.33 is calibrated against the post-combine compression
    # (compress_to_dealbreaker_floor: raw → 0.5 + 0.5 * raw), so a
    # single-column country pos-2 dealbreaker lands at ~0.665 in the
    # dealbreaker band — clearly above the floor but distinctly below
    # pos 1's 1.0. Position 3+ still scores zero so IMDB tail
    # positions don't sneak into the candidate pool.
    if not movie_country_ids:
        return 0.0
    for idx, cid in enumerate(movie_country_ids[:2], start=1):
        if cid in include_set:
            return 1.0 if idx == 1 else 0.33
    return 0.0


def _score_popularity(pop: float | None, mode: PopularityMode) -> float:
    if pop is None:
        return 0.0
    pop = max(0.0, min(1.0, float(pop)))
    return pop if mode == PopularityMode.POPULAR else (1.0 - pop)


def _score_reception(reception: float | None, mode: ReceptionMode) -> float:
    if reception is None:
        return 0.0
    if mode == ReceptionMode.WELL_RECEIVED:
        return max(0.0, min(1.0, (reception - 55) / 40))
    return max(0.0, min(1.0, (50 - reception) / 40))


# ── Column handler abstraction ────────────────────────────────────────────


@dataclass
class _ColumnHandler:
    """Per-column query and scoring contract. Built once per call from
    the LLM-supplied sub-object value; the closure captures any
    precomputed window / key state so per-row scoring is cheap.

    Fields:
        select_column: the movie_card column whose value drives both
            the gate and the scoring.
        gate_sql: dealbreaker WHERE fragment (parameterized). Joined
            into the call's overall WHERE via OR; parens wrap each
            handler's fragment when emitted by the SQL builder.
        gate_params: positional parameters for gate_sql (in order).
        score: callable that takes the row's value of select_column
            and returns a raw [0, 1] score. Scoring is uncompressed —
            compression is applied once after combine_mode folds.
        unbounded: True for popularity / reception (no natural gate;
            candidate pool needs ORDER BY + LIMIT when these are the
            only populated columns).
        sort_signal_sql: SQL expression in [0, 1] (modulo NULL → 0.0
            via COALESCE) used as the per-column contribution to
            ORDER BY in the all-unbounded dealbreaker path. None for
            bounded columns.
    """

    select_column: str
    gate_sql: str
    gate_params: list
    score: Callable[[Any], float]
    unbounded: bool
    sort_signal_sql: str | None = None


# ── Per-column handler factories ─────────────────────────────────────────


def _make_release_date_handler(value: ReleaseDateTranslation) -> _ColumnHandler:
    op = DateMatchOperation(value.match_operation)
    first_ts = _date_to_unix_ts(value.first_date)
    second_ts = _date_to_unix_ts(value.second_date) if value.second_date else None
    lo, hi, grace_days = _precompute_date_window(op, first_ts, second_ts)
    grace_seconds = grace_days * _SECONDS_PER_DAY

    fragments = ["release_ts IS NOT NULL"]
    params: list = []
    if lo != float("-inf"):
        fragments.append("release_ts >= %s")
        params.append(int(lo - grace_seconds))
    if hi != float("inf"):
        fragments.append("release_ts <= %s")
        params.append(int(hi + grace_seconds))

    return _ColumnHandler(
        select_column="release_ts",
        gate_sql=" AND ".join(fragments),
        gate_params=params,
        score=lambda ts: _score_release_date(ts, lo, hi, grace_days),
        unbounded=False,
    )


def _make_runtime_handler(value: RuntimeTranslation) -> _ColumnHandler:
    op = NumericalMatchOperation(value.match_operation)
    lo, hi = _precompute_runtime_window(op, value.first_value, value.second_value)

    fragments = ["runtime_minutes IS NOT NULL"]
    params: list = []
    if lo != float("-inf"):
        fragments.append("runtime_minutes >= %s")
        params.append(int(lo - _RUNTIME_GRACE_MINUTES))
    if hi != float("inf"):
        fragments.append("runtime_minutes <= %s")
        params.append(int(hi + _RUNTIME_GRACE_MINUTES))

    return _ColumnHandler(
        select_column="runtime_minutes",
        gate_sql=" AND ".join(fragments),
        gate_params=params,
        score=lambda rt: _score_runtime(rt, lo, hi),
        unbounded=False,
    )


def _make_maturity_handler(value: MaturityRatingTranslation) -> _ColumnHandler:
    target = MaturityRating(value.rating)
    op = RatingMatchOperation(value.match_operation)
    unrated_rank = MaturityRating.UNRATED.maturity_rank
    is_exact_unrated = target == MaturityRating.UNRATED and op == RatingMatchOperation.EXACT
    lo, hi = _precompute_rank_window(op, target.maturity_rank)

    if is_exact_unrated:
        gate_sql = "maturity_rank = %s"
        gate_params: list = [unrated_rank]
    else:
        # Include the 0.5 fringe (distance 1) in the candidate gate,
        # and exclude UNRATED entirely per the proposal's rule.
        gate_lo = max(1, lo - 1)
        gate_hi = min(5, hi + 1)
        gate_sql = (
            "maturity_rank IS NOT NULL AND maturity_rank <> %s "
            "AND maturity_rank BETWEEN %s AND %s"
        )
        gate_params = [unrated_rank, gate_lo, gate_hi]

    return _ColumnHandler(
        select_column="maturity_rank",
        gate_sql=gate_sql,
        gate_params=gate_params,
        score=lambda rank: _score_maturity(rank, target, lo, hi),
        unbounded=False,
    )


def _make_streaming_handler(value: StreamingTranslation) -> _ColumnHandler:
    services = [StreamingService(s) for s in value.services]
    preferred_access = (
        StreamingAccessType(value.preferred_access_type)
        if value.preferred_access_type is not None
        else None
    )
    any_method_keys, desired_method_keys, access_type_id = _precompute_streaming_keys(
        services, preferred_access,
    )

    if any_method_keys:
        # Services specified: GIN overlap is the tight candidate gate.
        gate_sql = "watch_offer_keys IS NOT NULL AND watch_offer_keys && %s"
        gate_params: list = [list(any_method_keys)]
    else:
        # Access-type only: full-table scan via EXISTS on unnest. Rare
        # in practice (pure access-type dealbreakers); accepted cost.
        gate_sql = (
            "watch_offer_keys IS NOT NULL AND EXISTS ("
            "SELECT 1 FROM unnest(watch_offer_keys) k WHERE (k & %s) = %s"
            ")"
        )
        gate_params = [_METHOD_ID_BITMASK, access_type_id]

    return _ColumnHandler(
        select_column="watch_offer_keys",
        gate_sql=gate_sql,
        gate_params=gate_params,
        score=lambda keys: _score_streaming(
            keys, any_method_keys, desired_method_keys, access_type_id,
        ),
        unbounded=False,
    )


def _make_audio_language_handler(value: AudioLanguageTranslation) -> _ColumnHandler:
    include_ids = [Language(v).language_id for v in value.languages]
    include_set = set(include_ids)

    return _ColumnHandler(
        select_column="audio_language_ids",
        gate_sql="audio_language_ids && %s",
        gate_params=[include_ids],
        score=lambda lang_ids: 1.0 if lang_ids and (set(lang_ids) & include_set) else 0.0,
        unbounded=False,
    )


def _make_country_handler(
    value: CountryOfOriginTranslation, dealbreaker: bool,
) -> _ColumnHandler:
    include_ids = [Country(v).country_id for v in value.countries]
    include_set = set(include_ids)

    if dealbreaker:
        score_fn = lambda cids: _score_country_position_dealbreaker(cids, include_set)
    else:
        score_fn = lambda cids: _score_country_position_preference(cids, include_set)

    return _ColumnHandler(
        select_column="country_of_origin_ids",
        gate_sql="country_of_origin_ids && %s",
        gate_params=[include_ids],
        score=score_fn,
        unbounded=False,
    )


def _make_budget_handler(value: str) -> _ColumnHandler:
    target_str = str(BudgetSize(value).value)
    return _ColumnHandler(
        select_column="budget_bucket",
        gate_sql="budget_bucket = %s",
        gate_params=[target_str],
        score=lambda b: 1.0 if b == target_str else 0.0,
        unbounded=False,
    )


def _make_box_office_handler(value: str) -> _ColumnHandler:
    target_str = str(BoxOfficeStatus(value).value)
    return _ColumnHandler(
        select_column="box_office_bucket",
        gate_sql="box_office_bucket = %s",
        gate_params=[target_str],
        score=lambda b: 1.0 if b == target_str else 0.0,
        unbounded=False,
    )


def _make_popularity_handler(value: str) -> _ColumnHandler:
    mode = PopularityMode(value)
    # Sort signal: direction-aware [0, 1] expression, NULL → 0 via
    # COALESCE so rows missing popularity_score sort last in DESC order.
    if mode == PopularityMode.POPULAR:
        sort_signal = "COALESCE(popularity_score, 0.0)"
    else:  # NICHE
        sort_signal = "COALESCE(1.0 - popularity_score, 0.0)"

    return _ColumnHandler(
        select_column="popularity_score",
        gate_sql="popularity_score IS NOT NULL",
        gate_params=[],
        score=lambda p: _score_popularity(p, mode),
        unbounded=True,
        sort_signal_sql=sort_signal,
    )


def _make_reception_handler(value: str) -> _ColumnHandler:
    mode = ReceptionMode(value)
    # Sort signal mirrors _score_reception's piecewise-linear shape;
    # clamping inside the SQL keeps the contribution in [0, 1] when
    # combined with popularity's [0, 1] signal.
    if mode == ReceptionMode.WELL_RECEIVED:
        sort_signal = (
            "COALESCE(LEAST(1.0, GREATEST(0.0, (reception_score - 55.0) / 40.0)), 0.0)"
        )
    else:  # POORLY_RECEIVED
        sort_signal = (
            "COALESCE(LEAST(1.0, GREATEST(0.0, (50.0 - reception_score) / 40.0)), 0.0)"
        )

    return _ColumnHandler(
        select_column="reception_score",
        gate_sql="reception_score IS NOT NULL",
        gate_params=[],
        score=lambda r: _score_reception(r, mode),
        unbounded=True,
        sort_signal_sql=sort_signal,
    )


def _build_handlers(
    spec: ColumnSpec, dealbreaker: bool,
) -> dict[MetadataAttribute, _ColumnHandler]:
    """One handler per populated ColumnSpec field. Null fields produce
    no handler — they don't participate in gate, SELECT, or scoring."""
    handlers: dict[MetadataAttribute, _ColumnHandler] = {}
    if spec.release_date is not None:
        handlers[MetadataAttribute.RELEASE_DATE] = _make_release_date_handler(spec.release_date)
    if spec.runtime is not None:
        handlers[MetadataAttribute.RUNTIME] = _make_runtime_handler(spec.runtime)
    if spec.maturity_rating is not None:
        handlers[MetadataAttribute.MATURITY_RATING] = _make_maturity_handler(spec.maturity_rating)
    if spec.streaming is not None:
        handlers[MetadataAttribute.STREAMING] = _make_streaming_handler(spec.streaming)
    if spec.audio_language is not None:
        handlers[MetadataAttribute.AUDIO_LANGUAGE] = _make_audio_language_handler(spec.audio_language)
    if spec.country_of_origin is not None:
        handlers[MetadataAttribute.COUNTRY_OF_ORIGIN] = _make_country_handler(
            spec.country_of_origin, dealbreaker,
        )
    if spec.budget_scale is not None:
        handlers[MetadataAttribute.BUDGET_SCALE] = _make_budget_handler(spec.budget_scale)
    if spec.box_office is not None:
        handlers[MetadataAttribute.BOX_OFFICE] = _make_box_office_handler(spec.box_office)
    if spec.popularity is not None:
        handlers[MetadataAttribute.POPULARITY] = _make_popularity_handler(spec.popularity)
    if spec.reception is not None:
        handlers[MetadataAttribute.RECEPTION] = _make_reception_handler(spec.reception)
    return handlers


# ── SQL builders ──────────────────────────────────────────────────────────


def _ordered_select_columns(
    handlers: dict[MetadataAttribute, _ColumnHandler],
) -> list[str]:
    """Stable, deduplicated SELECT-column list. Sort order is fixed so
    the per-row scoring loop can index by position."""
    return sorted({h.select_column for h in handlers.values()})


def _build_unbounded_sort_expr(
    handlers: dict[MetadataAttribute, _ColumnHandler],
    mode: ColumnCombineMode,
) -> str:
    """ORDER BY expression for the all-unbounded dealbreaker path.
    Combines per-handler sort_signal_sql via the combine_mode operator
    (sum for AVERAGE, GREATEST for MAX). Single-handler case collapses
    to that handler's signal."""
    signals = [h.sort_signal_sql for h in handlers.values()]
    if len(signals) == 1:
        return signals[0]
    if mode == ColumnCombineMode.MAX:
        return f"GREATEST({', '.join(signals)})"
    return "(" + " + ".join(signals) + ")"


def _build_dealbreaker_sql(
    handlers: dict[MetadataAttribute, _ColumnHandler],
    mode: ColumnCombineMode,
    select_cols: list[str],
) -> tuple[str, list]:
    """Compose one SELECT for the dealbreaker path.

    Two shapes:
      - All populated columns are unbounded (popularity / reception
        only): no candidate-narrowing WHERE clause exists, so we
        ORDER BY a combined sort signal and LIMIT to a fixed cap.
      - At least one bounded column: OR the per-column gates. A movie
        qualifies if any column's gate admits it; per-row scoring
        determines its actual contribution.
    """
    if all(h.unbounded for h in handlers.values()):
        non_null_clauses = [f"{col} IS NOT NULL" for col in select_cols]
        where = " OR ".join(non_null_clauses)
        sort_expr = _build_unbounded_sort_expr(handlers, mode)
        sql = (
            f"SELECT movie_id, {', '.join(select_cols)} FROM movie_card "
            f"WHERE {where} ORDER BY {sort_expr} DESC LIMIT %s"
        )
        return sql, [_POPULARITY_RECEPTION_DEALBREAKER_CAP]

    # Mixed / all-bounded: OR'd gates.
    gate_parts: list[str] = []
    params: list = []
    for h in handlers.values():
        gate_parts.append(f"({h.gate_sql})")
        params.extend(h.gate_params)
    where = " OR ".join(gate_parts)
    sql = f"SELECT movie_id, {', '.join(select_cols)} FROM movie_card WHERE {where}"
    return sql, params


def _build_preference_sql(
    restrict: set[int], select_cols: list[str],
) -> tuple[str, list]:
    """Pull every supplied id with all needed columns in one round-trip."""
    sql = (
        f"SELECT movie_id, {', '.join(select_cols)} FROM movie_card "
        "WHERE movie_id = ANY(%s)"
    )
    return sql, [list(restrict)]


# ── Per-row scoring + folding ────────────────────────────────────────────


def _score_and_combine(
    rows: list[tuple],
    handlers: dict[MetadataAttribute, _ColumnHandler],
    select_cols: list[str],
    mode: ColumnCombineMode,
) -> dict[int, float]:
    """Score each row per populated column, fold into one combined raw
    score per movie, return a {movie_id → combined raw score} map.
    Compression is NOT applied here — the caller decides based on mode
    (dealbreaker compresses; preference does not)."""
    col_to_idx = {col: i + 1 for i, col in enumerate(select_cols)}
    handler_indices = [
        (h, col_to_idx[h.select_column]) for h in handlers.values()
    ]
    is_max = mode == ColumnCombineMode.MAX

    out: dict[int, float] = {}
    for row in rows:
        movie_id = int(row[0])
        per_column = [h.score(row[idx]) for h, idx in handler_indices]
        if is_max:
            combined = max(per_column)
        else:
            combined = sum(per_column) / len(per_column)
        out[movie_id] = combined
    return out


# ── Public entry point ────────────────────────────────────────────────────


async def execute_metadata_query(
    output: MetadataTranslationOutput,
    restrict_to_movie_ids: set[int] | None = None,
) -> EndpointResult:
    """Execute one whole-call metadata translation and return scored
    candidates.

    The new schema gives us a single ColumnSpec where each of the ten
    attribute columns is independently populated-or-null, plus a
    combine_mode (max | average). One SQL per call regardless of column
    count; per-column scoring runs in Python after the fetch; the
    combine_mode folds per-column scores into one per-movie call score.

    Dealbreaker mode (restrict_to_movie_ids=None):
      - OR-joined per-column gates (any column admits the movie); the
        all-unbounded path uses ORDER BY + LIMIT instead.
      - Drop combined == 0; compress survivors into [0.5, 1.0].
    Preference mode (restrict_to_movie_ids supplied):
      - movie_id = ANY(%s) gate.
      - Raw [0, 1] combined scores preserved; missing ids zero-filled
        by build_endpoint_result.

    One retry on transient failure; a second failure collapses to an
    empty result so the assembly pipeline can keep working.
    """
    is_dealbreaker = restrict_to_movie_ids is None

    # Empty-set restriction: contract documents short-circuit (no SQL
    # round-trip). Catch this before building handlers so the path is
    # cheap and explicit.
    if restrict_to_movie_ids is not None and not restrict_to_movie_ids:
        return build_endpoint_result({}, restrict_to_movie_ids)

    handlers = _build_handlers(output.column_spec, dealbreaker=is_dealbreaker)
    if not handlers:
        # No populated columns — nothing to score against. Schema-level
        # invariants make this near-impossible, but the guard is free.
        return build_endpoint_result({}, restrict_to_movie_ids)

    mode = ColumnCombineMode(output.combine_mode)
    select_cols = _ordered_select_columns(handlers)

    if is_dealbreaker:
        sql, params = _build_dealbreaker_sql(handlers, mode, select_cols)
    else:
        sql, params = _build_preference_sql(restrict_to_movie_ids, select_cols)

    for attempt in (1, 2):
        try:
            rows = await _fetch(sql, params)
            combined_by_movie = _score_and_combine(rows, handlers, select_cols, mode)

            if is_dealbreaker:
                # Drop zero-combined rows (matched the OR'd gate via
                # one column but every populated column scored 0 raw),
                # then compress survivors into [0.5, 1.0].
                scored = {
                    mid: compress_to_dealbreaker_floor(combined)
                    for mid, combined in combined_by_movie.items()
                    if combined > 0.0
                }
            else:
                scored = combined_by_movie

            return build_endpoint_result(scored, restrict_to_movie_ids)
        except Exception:
            log.exception(
                "metadata execution failed on attempt %d (dealbreaker=%s, columns=%s)",
                attempt, is_dealbreaker, sorted(h.value for h in handlers.keys()),
            )

    return build_endpoint_result({}, restrict_to_movie_ids)
