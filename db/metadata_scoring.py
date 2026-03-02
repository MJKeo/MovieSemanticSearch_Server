"""
metadata_scoring.py — Score search candidates against inferred metadata preferences.

Post-processing step called after search, before final reranking. Computes a
weighted average of per-preference scores and assigns the result to each
candidate's metadata_score field.
"""

import asyncio
from datetime import datetime, timezone
from enum import Enum

from db.postgres import fetch_movie_cards
from db.redis import read_trending_scores
from db.search import SearchCandidate
from implementation.classes.enums import (
    BudgetSize,
    DateMatchOperation,
    Genre,
    MaturityRating,
    NumericalMatchOperation,
    RatingMatchOperation,
    ReceptionType,
    StreamingAccessType,
)
from implementation.classes.schemas import MetadataPreferencesResponse
from implementation.classes.watch_providers import StreamingService, STREAMING_PROVIDER_MAP
from implementation.misc.helpers import create_watch_provider_offering_key


# ── CONSTANTS ──────────────────────────────────────────

class ScoredPreference(Enum):
    """Metadata preference names paired with their scoring weights."""

    weight: int

    def __new__(cls, value: str, weight: int) -> "ScoredPreference":
        obj = object.__new__(cls)
        obj._value_ = value
        obj.weight = weight
        return obj

    GENRES          = ("genres",          5)
    RELEASE_DATE    = ("release_date",    4)
    WATCH_PROVIDERS = ("watch_providers", 4)
    AUDIO_LANGUAGE  = ("audio_language",  3)
    MATURITY_RATING = ("maturity_rating", 3)
    RECEPTION       = ("reception",       3)
    DURATION        = ("duration",        2)
    TRENDING        = ("trending",        2)
    POPULAR         = ("popular",         2)
    BUDGET_SIZE     = ("budget_size",     3)


_DURATION_GRACE_MINUTES = 30

_DATE_GRACE_DAYS_MINIMUM     = 365.0 * 1
_DATE_GRACE_DAYS_MAXIMUM     = 365.0 * 5
_DATE_GRACE_DAYS_UNBOUNDED   = 365.0 * 3
_DATE_GRACE_DAYS_EXACT_MATCH = 365.0 * 2

_ALL_METHOD_IDS: list[int] = [m.type_id for m in StreamingAccessType]


# ── Per-preference scorers ────────────────────────────────────────────────
# Each returns a float. Missing movie data → 0.0 (never skip an active pref).

def _score_release_date(
    release_ts: int | None,
    lo: float,
    hi: float,
    grace_days: float,
) -> float:
    """Score [0, 1] based on release_ts distance from preferred date range."""
    if release_ts is None:
        return 0.0

    if release_ts < lo:
        distance_days = (lo - release_ts) / 86400
    elif release_ts > hi:
        distance_days = (release_ts - hi) / 86400
    else:
        return 1.0

    return max(0.0, 1.0 - distance_days / grace_days)


def _score_duration(
    runtime: int | None,
    lo: float,
    hi: float,
) -> float:
    """Score [0, 1] based on runtime_minutes distance from preferred range."""
    if runtime is None:
        return 0.0

    if runtime < lo:
        distance = lo - runtime
    elif runtime > hi:
        distance = runtime - hi
    else:
        return 1.0

    return max(0.0, 1.0 - distance / _DURATION_GRACE_MINUTES)


def _score_genres(
    genre_ids: list[int] | None,
    include_ids: set[int],
    exclude_ids: set[int],
) -> float:
    """Score [-2, 1]. Exclusion hit → -2.0. Inclusion → fraction matched."""
    if genre_ids is None:
        return 0.0

    genre_id_set = set(genre_ids)

    if exclude_ids & genre_id_set:
        return -2.0

    if include_ids:
        matched = len(include_ids & genre_id_set)
        return matched / len(include_ids)

    # Only exclusions were set and none matched
    return 1.0


def _score_audio_language(
    lang_ids: list[int] | None,
    include_ids: set[int],
    exclude_ids: set[int],
) -> float:
    """Score [-2, 1]. Same exclusion pattern as genres."""
    if lang_ids is None:
        return 0.0

    lang_id_set = set(lang_ids)

    if exclude_ids & lang_id_set:
        return -2.0

    if include_ids:
        return 1.0 if (include_ids & lang_id_set) else 0.0

    return 1.0


def _score_watch_providers(
    offer_keys: list[int] | None,
    exclude_key_set: set[int],
    include_any_keys: set[int],
    include_desired_keys: set[int],
    access_type_id: int | None,
) -> float:
    """Score [0, 1]. Uses encoded watch_offer_keys."""
    if offer_keys is None:
        return 0.0

    filtered_keys = set(offer_keys) - exclude_key_set

    # Both include and access_type set
    if include_any_keys and include_desired_keys:
        if filtered_keys & include_desired_keys:
            return 1.0
        if filtered_keys & include_any_keys:
            return 0.5
        return 0.0

    # Include only
    if include_any_keys:
        return 1.0 if (filtered_keys & include_any_keys) else 0.0

    # Access type only
    if access_type_id is not None:
        type_ids = {key & 0xF for key in filtered_keys}
        return 1.0 if access_type_id in type_ids else 0.0

    # Exclude only (or no preferences)
    return 1.0 if filtered_keys else 0.0


def _score_maturity_rating(
    movie_rank: int | None,
    target: MaturityRating,
    lo: int,
    hi: int,
) -> float:
    """Score [0, 1]. Ordinal distance from valid range."""
    if movie_rank is None:
        return 0.0

    # Unrated handling
    if movie_rank == MaturityRating.UNRATED.maturity_rank:
        return 1.0 if target == MaturityRating.UNRATED else 0.0
    if target == MaturityRating.UNRATED:
        return 0.0

    if lo <= movie_rank <= hi:
        return 1.0

    distance = min(abs(movie_rank - lo), abs(movie_rank - hi))
    if distance <= 1:
        return 0.5
    return 0.0


def _score_reception(
    reception_score: float | None,
    reception_type: ReceptionType,
) -> float:
    """Score [0, 1]. Linear ramp based on reception_score."""
    if reception_score is None:
        return 0.0

    if reception_type == ReceptionType.CRITICALLY_ACCLAIMED:
        return max(0.0, min(1.0, (reception_score - 55) / 40))
    elif reception_type == ReceptionType.POORLY_RECEIVED:  # POORLY_RECEIVED (NO_PREFERENCE is never passed here)
        return max(0.0, min(1.0, (50 - reception_score) / 40))
    else:
        # We shouldn't ever hit this but if for some reason we do always return 0.0 so it doesn't impact relative rankings
        return 0.0


def _score_trending(
    movie_id: int,
    trending_scores: dict[int, float],
) -> float:
    """Score [0, 1]. Uses actual trending score from Redis."""
    return trending_scores.get(movie_id, 0.0)


def _score_popular(popularity_score: float | None) -> float:
    """Score [0, 1]. Pass-through of popularity_score."""
    if popularity_score is None:
        return 0.0
    return float(popularity_score)


def _score_budget_size(budget_bucket: str | None, preferred_size: BudgetSize) -> float:
    """
    Score [0, 1]. Binary match against the movie's stored budget bucket.

    Returns 1.0 when the stored bucket exactly matches the user's preference,
    0.0 in all other cases — including when the movie has no budget data on
    record or falls in the mid-range (both stored as NULL).
    """
    if budget_bucket is None:
        return 0.0
    return 1.0 if budget_bucket == preferred_size.value else 0.0


# ── Precomputation helpers ────────────────────────────────────────────────

def _precompute_release_date(first_date: str, match_operation: DateMatchOperation, second_date: str | None):
    """Parse dates and compute lo/hi/grace once for all candidates."""
    first_ts = datetime.strptime(first_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
    second_ts = (
        datetime.strptime(second_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
        if second_date
        else None
    )

    if match_operation == DateMatchOperation.BETWEEN and second_date is not None:
        lo, hi = min(first_ts, second_ts), max(first_ts, second_ts)
        range_width_days = (hi - lo) / 86400
        grace_days = max(_DATE_GRACE_DAYS_MINIMUM, min(range_width_days * 0.5, _DATE_GRACE_DAYS_MAXIMUM))
    elif match_operation == DateMatchOperation.AFTER:
        lo, hi = first_ts, float("inf")
        grace_days = _DATE_GRACE_DAYS_UNBOUNDED
    elif match_operation == DateMatchOperation.BEFORE:
        lo, hi = float("-inf"), first_ts
        grace_days = _DATE_GRACE_DAYS_UNBOUNDED
    else:  # EXACT
        lo, hi = first_ts, first_ts
        grace_days = _DATE_GRACE_DAYS_EXACT_MATCH

    return lo, hi, grace_days


def _precompute_duration(first_value: float, match_operation: NumericalMatchOperation, second_value: float | None):
    """Compute lo/hi once for all candidates."""
    if match_operation == NumericalMatchOperation.BETWEEN and second_value is not None:
        return min(first_value, second_value), max(first_value, second_value)
    elif match_operation == NumericalMatchOperation.GREATER_THAN:
        return first_value, float("inf")
    elif match_operation == NumericalMatchOperation.LESS_THAN:
        return float("-inf"), first_value
    else:  # EXACT
        return first_value, first_value


def _precompute_maturity(rating_value: str, match_operation: RatingMatchOperation):
    """Resolve target rating and compute lo/hi rank range once."""
    target = MaturityRating(rating_value)
    target_rank = target.maturity_rank

    if match_operation == RatingMatchOperation.EXACT:
        lo, hi = target_rank, target_rank
    elif match_operation == RatingMatchOperation.GREATER_THAN:
        lo, hi = target_rank + 1, 5
    elif match_operation == RatingMatchOperation.LESS_THAN:
        lo, hi = 1, target_rank - 1
    elif match_operation == RatingMatchOperation.GREATER_THAN_OR_EQUAL:
        lo, hi = target_rank, 5
    elif match_operation == RatingMatchOperation.LESS_THAN_OR_EQUAL:
        lo, hi = 1, target_rank
    else:  # unknown — fall back to exact
        lo, hi = target_rank, target_rank

    return target, lo, hi


def _precompute_watch_providers(
    should_include: list[str],
    should_exclude: list[str],
    preferred_access_type: StreamingAccessType | None,
):
    """Resolve StreamingService values → key sets once for all candidates.

    Returns (exclude_key_set, include_any_keys, include_desired_keys, access_type_id).
    """
    # Exclusion: one flat set of all-method keys across all excluded services
    exclude_key_set: set[int] = set()
    for value in should_exclude:
        try:
            service = StreamingService(value)
        except ValueError:
            continue
        provider_ids = STREAMING_PROVIDER_MAP.get(service, [])
        for pid in provider_ids:
            for mid in _ALL_METHOD_IDS:
                exclude_key_set.add(create_watch_provider_offering_key(pid, mid))

    # Include: any-method keys and (optionally) preferred-method keys
    include_any_keys: set[int] = set()
    include_desired_keys: set[int] = set()
    for value in should_include:
        try:
            service = StreamingService(value)
        except ValueError:
            continue
        provider_ids = STREAMING_PROVIDER_MAP.get(service, [])
        for pid in provider_ids:
            for mid in _ALL_METHOD_IDS:
                include_any_keys.add(create_watch_provider_offering_key(pid, mid))
            if preferred_access_type is not None:
                include_desired_keys.add(create_watch_provider_offering_key(pid, preferred_access_type.type_id))

    # When should_include is set, the access type is already encoded into
    # include_desired_keys (provider+method combos). access_type_id is only
    # needed for the standalone "access type only" branch in _score_watch_providers,
    # so we deliberately leave it None when should_include is present.
    access_type_id = preferred_access_type.type_id if preferred_access_type and not should_include else None

    return exclude_key_set, include_any_keys, include_desired_keys, access_type_id


# ── Main entry point ──────────────────────────────────────────────────────

async def create_metadata_scores(
    preferences: MetadataPreferencesResponse,
    candidates: list[SearchCandidate],
) -> list[SearchCandidate]:
    """
    Score each candidate against the user's inferred metadata preferences.

    Mutates each candidate's metadata_score in place and returns the same list.
    """
    if not candidates or not preferences.has_active_preferences():
        return candidates

    # ── Determine active preferences ──
    prefs = preferences
    active: set[ScoredPreference] = set()

    if (r := prefs.release_date_preference.result) is not None and r.contains_valid_data():
        active.add(ScoredPreference.RELEASE_DATE)
    if (r := prefs.duration_preference.result) is not None and r.contains_valid_data():
        active.add(ScoredPreference.DURATION)
    if (r := prefs.genres_preference.result) is not None and r.contains_valid_data():
        active.add(ScoredPreference.GENRES)
    if (r := prefs.audio_languages_preference.result) is not None and r.contains_valid_data():
        active.add(ScoredPreference.AUDIO_LANGUAGE)
    if (r := prefs.watch_providers_preference.result) is not None and r.contains_valid_data():
        active.add(ScoredPreference.WATCH_PROVIDERS)
    if (r := prefs.maturity_rating_preference.result) is not None and r.contains_valid_data():
        active.add(ScoredPreference.MATURITY_RATING)
    if (r := prefs.reception_preference) is not None and r.contains_valid_data():
        active.add(ScoredPreference.RECEPTION)
        reception_type = ReceptionType(r.reception_type)
    if (r := prefs.popular_trending_preference) is not None:
        if r.prefers_trending_movies:
            active.add(ScoredPreference.TRENDING)
        if r.prefers_popular_movies:
            active.add(ScoredPreference.POPULAR)

    # Budget size: normalize to enum if stored as a raw string, then check for a real preference.
    budget_size_pref: BudgetSize | None = None
    if (r := prefs.budget_size_preference) is not None:
        bs = BudgetSize(r.budget_size) if isinstance(r.budget_size, str) else r.budget_size
        if bs != BudgetSize.NO_PREFERENCE:
            active.add(ScoredPreference.BUDGET_SIZE)
            budget_size_pref = bs

    # ── Precompute preference constants ──
    rd_lo = rd_hi = rd_grace = 0.0
    if ScoredPreference.RELEASE_DATE in active:
        r = prefs.release_date_preference.result
        rd_lo, rd_hi, rd_grace = _precompute_release_date(
            r.first_date, DateMatchOperation(r.match_operation), r.second_date,
        )

    dur_lo = dur_hi = 0.0
    if ScoredPreference.DURATION in active:
        r = prefs.duration_preference.result
        dur_lo, dur_hi = _precompute_duration(
            r.first_value, NumericalMatchOperation(r.match_operation), r.second_value,
        )

    genre_include_ids: set[int] = set()
    genre_exclude_ids: set[int] = set()
    if ScoredPreference.GENRES in active:
        r = prefs.genres_preference.result
        genre_include_ids = {Genre(v).genre_id for v in r.should_include}
        genre_exclude_ids = {Genre(v).genre_id for v in r.should_exclude}

    lang_include_ids: set[int] = set()
    lang_exclude_ids: set[int] = set()
    if ScoredPreference.AUDIO_LANGUAGE in active:
        r = prefs.audio_languages_preference.result
        lang_include_ids = {lang.language_id for lang in r.should_include}
        lang_exclude_ids = {lang.language_id for lang in r.should_exclude}

    wp_exclude_key_set: set[int] = set()
    wp_include_any: set[int] = set()
    wp_include_desired: set[int] = set()
    wp_access_type_id: int | None = None
    if ScoredPreference.WATCH_PROVIDERS in active:
        r = prefs.watch_providers_preference.result
        wp_exclude_key_set, wp_include_any, wp_include_desired, wp_access_type_id = _precompute_watch_providers(
            r.should_include,
            r.should_exclude,
            StreamingAccessType(r.preferred_access_type) if r.preferred_access_type is not None else None,
        )

    mat_target = MaturityRating.UNRATED
    mat_lo = mat_hi = 0
    if ScoredPreference.MATURITY_RATING in active:
        r = prefs.maturity_rating_preference.result
        mat_target, mat_lo, mat_hi = _precompute_maturity(
            r.rating, RatingMatchOperation(r.match_operation),
        )

    # ── Fetch data (parallel I/O) ──
    movie_ids = [c.movie_id for c in candidates]

    if ScoredPreference.TRENDING in active:
        cards_list, trending_scores = await asyncio.gather(
            fetch_movie_cards(movie_ids),
            read_trending_scores(),
        )
    else:
        cards_list = await fetch_movie_cards(movie_ids)
        trending_scores = {}

    cards: dict[int, dict] = {card["movie_id"]: card for card in cards_list}

    # ── Precompute total weight (constant across candidates) ──
    total_weight = sum(p.weight for p in active)

    # ── Score each candidate ──
    for candidate in candidates:
        card = cards.get(candidate.movie_id)
        if card is None:
            # No movie card at all — every active preference scores 0.0
            # weighted average of all zeros = 0.0, which is the default
            continue

        weighted_sum = 0.0

        if ScoredPreference.RELEASE_DATE in active:
            weighted_sum += ScoredPreference.RELEASE_DATE.weight * _score_release_date(
                card.get("release_ts"), rd_lo, rd_hi, rd_grace,
            )

        if ScoredPreference.DURATION in active:
            weighted_sum += ScoredPreference.DURATION.weight * _score_duration(
                card.get("runtime_minutes"), dur_lo, dur_hi,
            )

        if ScoredPreference.GENRES in active:
            weighted_sum += ScoredPreference.GENRES.weight * _score_genres(
                card.get("genre_ids"), genre_include_ids, genre_exclude_ids,
            )

        if ScoredPreference.AUDIO_LANGUAGE in active:
            weighted_sum += ScoredPreference.AUDIO_LANGUAGE.weight * _score_audio_language(
                card.get("audio_language_ids"), lang_include_ids, lang_exclude_ids,
            )

        if ScoredPreference.WATCH_PROVIDERS in active:
            weighted_sum += ScoredPreference.WATCH_PROVIDERS.weight * _score_watch_providers(
                card.get("watch_offer_keys"),
                wp_exclude_key_set, wp_include_any, wp_include_desired, wp_access_type_id,
            )

        if ScoredPreference.MATURITY_RATING in active:
            weighted_sum += ScoredPreference.MATURITY_RATING.weight * _score_maturity_rating(
                card.get("maturity_rank"), mat_target, mat_lo, mat_hi,
            )

        if ScoredPreference.RECEPTION in active and reception_type is not None:
            weighted_sum += ScoredPreference.RECEPTION.weight * _score_reception(
                card.get("reception_score"), reception_type,
            )

        if ScoredPreference.TRENDING in active:
            weighted_sum += ScoredPreference.TRENDING.weight * _score_trending(
                candidate.movie_id, trending_scores,
            )

        if ScoredPreference.POPULAR in active:
            weighted_sum += ScoredPreference.POPULAR.weight * _score_popular(
                card.get("popularity_score"),
            )

        if ScoredPreference.BUDGET_SIZE in active and budget_size_pref is not None:
            weighted_sum += ScoredPreference.BUDGET_SIZE.weight * _score_budget_size(
                card.get("budget_bucket"), budget_size_pref,
            )

        candidate.metadata_score = weighted_sum / total_weight

    return candidates
