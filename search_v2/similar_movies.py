"""Standalone similar-movies search flow.

This module owns the Step-0 similarity path and the tmdb_id-facing debug
entrypoint. It is intentionally separate from the standard Stage-4 search
pipeline: no standard trait decomposition, endpoint execution, or branch
reranking code is called here.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from db.postgres import (
    SimilarityAwardSignals,
    TRAIT_KIND_CONCEPT_TAG,
    TRAIT_KIND_OVERALL_KEYWORD,
    TRAIT_KIND_SOURCE_MATERIAL,
    TRAIT_KIND_TMDB_GENRE,
    fetch_director_movie_terms,
    fetch_director_strengths,
    fetch_director_term_ids_for_movies,
    fetch_franchise_confidence,
    fetch_movie_ids_by_overall_keywords,
    fetch_movie_ids_by_production_company_ids,
    fetch_movie_ids_with_title_like,
    fetch_production_company_ids_by_normalized_strings,
    fetch_similarity_award_category_tags,
    fetch_similarity_award_signals,
    fetch_similarity_franchise_candidates,
    fetch_similarity_quality_candidates,
    fetch_similarity_signal_rows,
    fetch_similarity_source_candidates,
    fetch_similarity_top_billed_cast,
    fetch_trait_idfs,
)
from db.qdrant import qdrant_client
from db.vector_scoring import normalize_blended_scores
from db.vector_search import COLLECTION_ALIAS, QDRANT_SEARCH_PARAMS
from implementation.classes.enums import VectorName
from implementation.classes.overall_keywords import OverallKeyword
from implementation.misc.helpers import normalize_string
from implementation.misc.sql_like import escape_like
from schemas.enums import AwardCeremony
from schemas.step_0_flow_routing import SimilarityFlowData
from search_v2.award_taxonomy import (
    SPECIFICITY_FACTOR,
    TIER_WEIGHT,
    tag_level,
)
from search_v2.country_language_registry import (
    COUNTRY_LANGUAGE_KEYWORD_IDS,
    US_DEFAULT,
    country_set,
)
from search_v2.format_registry import (
    FORMAT_KEYWORD_IDS_ALL,
    FormatBucket,
    format_bucket,
)
from search_v2.production_medium_registry import (
    MEDIUM_TAG_IDS,
    load_medium_idfs,
    medium_score,
)
from search_v2.similar_studio_registry import (
    StudioSimilarityEntry,
    studio_entries_by_normalized_string,
)


LaneName = Literal[
    "shape",
    "director",
    "franchise",
    "studio",
    "source",
    "quality",
    "format",
    "themes",
    "cast",
    "specific_award",
]

# Lanes whose scores feed the additive sum. Studio is intentionally
# excluded — V2 made it a multiplier on shape-qualifying candidates so
# Blumhouse-style on-brand-but-low-shape noise no longer crowds the top
# of the list. The themes/cast/specific_award lanes are multi-anchor
# only; their weights collapse to 0 in single-anchor flow so the same
# pipeline handles both cases without branching.
ADDITIVE_LANES: tuple[LaneName, ...] = (
    "shape",
    "director",
    "franchise",
    "source",
    "quality",
    "format",
    "themes",
    "cast",
    "specific_award",
)

# All lanes carried through the debug payload, including the studio
# multiplier so debug output still shows when on-brand candidates were
# scored. Order matches LaneName.
ALL_LANES: tuple[LaneName, ...] = (
    "shape",
    "director",
    "franchise",
    "studio",
    "source",
    "quality",
    "format",
    "themes",
    "cast",
    "specific_award",
)

AnchorType = Literal[
    "standard_shape",
    "cult_garbage",
    "prestige",
    "franchise_dominant",
    "studio_lineage",
    "source_material",
    "director_signature",
]

# V2 single-anchor base vector-space weights — drops the V1 tier
# grouping. Higher narrative_techniques (storytelling-style binds
# Tarantino, Nolan), lower production (V1 locked Titanic to ship-disaster
# matches), and plot_events stays the lowest signal.
VECTOR_BASE_WEIGHTS_SINGLE: dict[VectorName, float] = {
    VectorName.PLOT_ANALYSIS: 1.00,
    VectorName.VIEWER_EXPERIENCE: 1.00,
    VectorName.WATCH_CONTEXT: 0.75,
    VectorName.NARRATIVE_TECHNIQUES: 0.55,
    VectorName.RECEPTION: 0.55,
    VectorName.ANCHOR: 0.45,
    VectorName.PRODUCTION: 0.30,
    VectorName.PLOT_EVENTS: 0.25,
}

# V2 multi-anchor base vector weights — kept at the V1 tiered set
# because cohesion does the heavy lifting and V1 results on cohesive
# anchor sets (Pixar, Ghibli, war films, Tarantino) were already strong.
VECTOR_BASE_WEIGHTS_MULTI: dict[VectorName, float] = {
    VectorName.PLOT_ANALYSIS: 1.00,
    VectorName.VIEWER_EXPERIENCE: 1.00,
    VectorName.WATCH_CONTEXT: 0.65,
    VectorName.PRODUCTION: 0.65,
    VectorName.RECEPTION: 0.65,
    VectorName.ANCHOR: 0.65,
    VectorName.NARRATIVE_TECHNIQUES: 0.35,
    VectorName.PLOT_EVENTS: 0.35,
}

BASE_LANE_WEIGHTS: dict[LaneName, float] = {
    "shape": 0.60,
    "director": 0.12,
    "franchise": 0.12,
    "studio": 0.06,        # debug-only; the multiplier doesn't read it
    "source": 0.04,
    "quality": 0.06,
    "format": 0.04,
    # Multi-only lanes; single-anchor zeros these out.
    "themes": 0.06,
    "cast": 0.03,
    "specific_award": 0.04,
}

# Single-anchor weight deltas applied additively per active anchor type.
# studio_lineage is retained as a debug flag with no weight delta — V2
# studio handling is multiplicative, not additive. director_signature is
# the new V2 anchor type for genuine top-tier auteurs (strength >= 0.80).
# source_material delta dropped from V1 +0.14 to V2 +0.08 because the
# IDF-weighted source lane no longer over-credits "novel"-tier matches.
SINGLE_ANCHOR_ADJUSTMENTS: dict[AnchorType, dict[LaneName, float]] = {
    "standard_shape": {},
    "cult_garbage": {"quality": 0.26, "shape": -0.10},
    "prestige": {"quality": 0.16, "shape": -0.06},
    "franchise_dominant": {"franchise": 0.18, "shape": -0.08},
    "studio_lineage": {},
    "source_material": {"source": 0.08, "shape": -0.04},
    "director_signature": {"director": 0.10, "shape": -0.04},
}


COMPETITIVE_BAND = 0.08
TOP_SECTION_SIZE = 10
TOP_FORMAT_LOCK = 5             # top-5 must share the anchor format bucket
MAX_TOP_DOMINANT_LANE = 4
MAX_TOP_FRANCHISE = 3

# V2 franchise confidence thresholds. Lineages that clear both gates run
# the lane additively (with raised shape gate); below either gate they
# drop to a multiplicative nudge so direct-to-DVD Barbie spinoffs can't
# dominate.
FRANCHISE_HIGH_CONF_CONFIDENCE = 0.65
FRANCHISE_HIGH_CONF_CONSISTENCY = 0.60

# V2 multipliers on combined score (applied post-additive-sum).
STUDIO_MULTIPLIER_SHAPE_GATE = 0.60
STUDIO_MULTIPLIER_STRENGTH = 0.10            # +10% per unit studio_score
LOW_CONF_FRANCHISE_SHAPE_GATE = 0.55
LOW_CONF_FRANCHISE_MULTIPLIER_STRENGTH = 0.10

# Medium multiplier: max-mismatch (live-action vs. animation) drops
# combined score by 15%; perfect medium agreement leaves it untouched.
MEDIUM_MULTIPLIER_FLOOR = 0.85
MEDIUM_MULTIPLIER_RANGE = 0.15

# Country/language coherence multiplier (multi-anchor only).
COUNTRY_CONSENSUS_BOOST = 1.10
COUNTRY_CONSENSUS_PENALTY = 0.85

# Selective rare-medium retrieval gate (V2 single-anchor): anchor medium
# tags with idf >= this threshold trigger a candidate-pool expansion via
# fetch_movie_ids_by_overall_keywords. LIVE_ACTION is always excluded
# because every catalog entry would qualify.
RARE_MEDIUM_IDF_THRESHOLD = 0.50

# director_signature anchor type triggers when the anchor's director has
# director_strength >= this percentile. Restricts the lane to genuine
# top-tier auteurs (Tarantino, Nolan, Scorsese, Spielberg, Miyazaki, etc.)
# rather than any above-median director.
DIRECTOR_SIGNATURE_STRENGTH_THRESHOLD = 0.80

# Low-cohesion fallback (multi-anchor): when both vector cohesion and
# every metadata lane's cohesion are weak, the centroid is in noise and
# we fall back to round-robin per-anchor single-anchor results.
LOW_COHESION_VECTOR_THRESHOLD = 0.35
LOW_COHESION_METADATA_MAX_THRESHOLD = 1.00

DEFAULT_QDRANT_LIMIT = 500
DEFAULT_QUALITY_LIMIT = 500

NON_RAZZIE_AWARD_IDS: frozenset[int] = frozenset(
    c.ceremony_id for c in AwardCeremony if c is not AwardCeremony.RAZZIE
)

LIVE_ACTION_TAG_ID = OverallKeyword.LIVE_ACTION.keyword_id


@dataclass(frozen=True, slots=True)
class LaneEvidence:
    lane_scores: dict[LaneName, float]
    candidate_sources: list[LaneName]
    dominant_lane: LaneName


@dataclass(frozen=True, slots=True)
class SimilarMovieResult:
    movie_id: int
    score: float
    evidence: LaneEvidence


@dataclass(frozen=True, slots=True)
class SimilarMoviesDebug:
    vector_space_weights: dict[str, float]
    vector_space_cohesion: dict[str, float] = field(default_factory=dict)
    raw_lane_weights: dict[LaneName, float] = field(default_factory=dict)
    normalized_lane_weights: dict[LaneName, float] = field(default_factory=dict)
    candidate_counts_by_lane: dict[LaneName, int] = field(default_factory=dict)
    # V2 additions — non-additive signals and audit trails. All optional so
    # a single-anchor flow can leave the multi-anchor-only fields empty
    # without forcing callers to deal with `None` checks.
    anchor_format_bucket: FormatBucket | None = None
    anchor_medium_tags: list[int] = field(default_factory=list)
    franchise_high_confidence: bool = False
    consensus_countries: list[int | str] = field(default_factory=list)
    low_cohesion_fallback_used: bool = False
    per_anchor_active_anchor_types: dict[int, list[AnchorType]] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class SimilarMoviesSearchResult:
    anchor_movie_ids: list[int]
    ranked: list[SimilarMovieResult]
    active_anchor_types: list[AnchorType]
    debug: SimilarMoviesDebug


@dataclass(frozen=True, slots=True)
class _ResolvedStudioEntry:
    company_id: int
    entry: StudioSimilarityEntry


@dataclass(slots=True)
class _CandidateScore:
    movie_id: int
    score: float
    lane_scores: dict[LaneName, float]
    candidate_sources: list[LaneName]
    dominant_lane: LaneName


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _release_year(row: dict) -> int | None:
    release_ts = row.get("release_ts")
    if release_ts is None:
        return None
    return datetime.fromtimestamp(release_ts, tz=timezone.utc).year


def _as_int_set(value: object) -> set[int]:
    if not value:
        return set()
    return {int(v) for v in value}


def _normalize_weights(raw: dict[LaneName, float]) -> dict[LaneName, float]:
    """Normalize raw lane weights so the additive lanes sum to 1.0.

    Studio is excluded — it's a multiplier in V2, not an additive lane,
    and including its raw weight in the denominator would dilute every
    other lane unnecessarily. Negative weights (possible after applying
    anchor-type deltas) get clamped to 0 before the division. If every
    lane is zero or negative, the function falls back to "all weight on
    shape" so the flow still returns ranked results.
    """
    total = sum(max(raw.get(lane, 0.0), 0.0) for lane in ADDITIVE_LANES)
    if total <= 0.0:
        return {lane: (1.0 if lane == "shape" else 0.0) for lane in ALL_LANES}
    out: dict[LaneName, float] = {
        lane: max(raw.get(lane, 0.0), 0.0) / total for lane in ADDITIVE_LANES
    }
    # Studio carries through unnormalized so debug output preserves the raw
    # studio score; the multiplier doesn't read this field.
    out["studio"] = max(raw.get("studio", 0.0), 0.0)
    return out


def _normalize_vector_weights(
    raw: dict[VectorName, float],
) -> dict[VectorName, float]:
    total = sum(raw.values())
    if total <= 0.0:
        return {}
    return {space: weight / total for space, weight in raw.items()}


def _normalize_lane_to_max(scores: dict[int, float]) -> dict[int, float]:
    """Rescale a merged lane so its strongest candidate is 1.0."""
    if not scores:
        return {}
    max_score = max(scores.values())
    if max_score <= 0.0:
        return {}
    return {movie_id: _clamp(score / max_score) for movie_id, score in scores.items()}


def _quality_bucket(row: dict) -> str:
    reception = row.get("reception_score")
    percentile = row.get("popularity_percentile") or 0.0
    if reception is not None and reception <= 45 and percentile >= 0.89:
        return "cult_garbage"
    if reception is not None and reception >= 85 and percentile >= 0.75:
        return "prestige"
    return "middle"


def _major_award_win_score(row: dict) -> float:
    wins = _as_int_set(row.get("award_ceremony_win_ids"))
    return 1.0 if wins & NON_RAZZIE_AWARD_IDS else 0.0


def _single_anchor_lane_weights(
    active_anchor_types: list[AnchorType],
    *,
    franchise_low_confidence: bool,
) -> tuple[dict[LaneName, float], dict[LaneName, float]]:
    """Build raw + normalized lane weights for single-anchor flow.

    V2 changes vs. V1: quality lane is always active (the middle bucket
    runs at base weight 0.06 as a soft notability prior); themes / cast
    / specific_award are zeroed out because they're multi-only. When the
    anchor sits on a low-confidence franchise (e.g. Barbie), the
    franchise lane is dropped from the additive sum entirely — the
    franchise multiplier handles those candidates instead — and the
    franchise_dominant adjustment is suppressed.
    """
    raw = dict(BASE_LANE_WEIGHTS)
    # Multi-only lanes contribute nothing in single-anchor flow.
    raw["themes"] = 0.0
    raw["cast"] = 0.0
    raw["specific_award"] = 0.0

    for anchor_type in active_anchor_types:
        # Suppress the franchise_dominant delta on low-confidence anchors;
        # the multiplicative path handles those without expanding the
        # franchise lane's exposure.
        if anchor_type == "franchise_dominant" and franchise_low_confidence:
            continue
        for lane, delta in SINGLE_ANCHOR_ADJUSTMENTS[anchor_type].items():
            raw[lane] = raw.get(lane, 0.0) + delta

    if franchise_low_confidence:
        raw["franchise"] = 0.0

    normalized = _normalize_weights(raw)
    return raw, normalized


def _metadata_cohesion(trait_sets: list[set[object]]) -> float:
    if not trait_sets:
        return 0.0
    counts: dict[object, int] = {}
    for traits in trait_sets:
        for trait in traits:
            counts[trait] = counts.get(trait, 0) + 1
    if not counts:
        return 0.0
    max_count = max(counts.values())
    if max_count < 2:
        return 0.0
    repetition_ratio = max_count / len(trait_sets)
    return 2.0 * math.log1p(9.0 * repetition_ratio) / math.log1p(9.0)


async def _load_studio_entries_by_company_id() -> dict[int, list[StudioSimilarityEntry]]:
    by_norm = studio_entries_by_normalized_string()
    id_by_norm = await fetch_production_company_ids_by_normalized_strings(
        sorted(by_norm)
    )
    out: dict[int, list[StudioSimilarityEntry]] = {}
    for norm, company_id in id_by_norm.items():
        out.setdefault(company_id, []).extend(by_norm.get(norm, ()))
    return out


def _active_studio_entries(
    row: dict,
    entries_by_company_id: dict[int, list[StudioSimilarityEntry]],
) -> list[_ResolvedStudioEntry]:
    release_year = _release_year(row)
    out: list[_ResolvedStudioEntry] = []
    for company_id in _as_int_set(row.get("production_company_ids")):
        for entry in entries_by_company_id.get(company_id, ()):
            if entry.era.matches(release_year):
                out.append(_ResolvedStudioEntry(company_id=company_id, entry=entry))
    return out


def _studio_score(
    anchor_row: dict,
    candidate_row: dict,
    anchor_entries: list[_ResolvedStudioEntry],
    candidate_entries: list[_ResolvedStudioEntry],
) -> float:
    if not anchor_entries or not candidate_entries:
        return 0.0

    anchor_by_company = {entry.company_id: entry for entry in anchor_entries}
    anchor_year = _release_year(anchor_row)
    candidate_year = _release_year(candidate_row)
    best = 0.0

    for candidate_entry in candidate_entries:
        anchor_entry = anchor_by_company.get(candidate_entry.company_id)
        if anchor_entry is None:
            continue
        score = min(anchor_entry.entry.base_score, candidate_entry.entry.base_score)
        if (
            (anchor_entry.entry.era_sensitive or candidate_entry.entry.era_sensitive)
            and anchor_year is not None
            and candidate_year is not None
        ):
            era_gap = abs(candidate_year - anchor_year)
            score *= 0.60 + 0.40 * math.exp(-era_gap / 18.0)
        best = max(best, score)

    return _clamp(best)


def _franchise_traits(row: dict) -> tuple[set[int], set[int], set[int]]:
    return (
        _as_int_set(row.get("lineage_entry_ids")),
        _as_int_set(row.get("shared_universe_entry_ids")),
        _as_int_set(row.get("subgroup_entry_ids")),
    )


def _franchise_score_v2(anchor_row: dict, candidate_row: dict) -> float:
    """V2 franchise lane score with subgroup gating.

    Lineage hits are full strength. Subgroup hits only score full when
    they're backed by a universe or lineage match — the V2 spec calls
    this out specifically so e.g. "Original Star Wars Trilogy" subgroup
    matches don't ride at 1.0 without the universe agreeing. Universe-
    only matches stay at 0.85; subgroup-only matches drop to a 0.40
    fallback so they still surface but can't dominate.
    """
    a_lin, a_uni, a_sub = _franchise_traits(anchor_row)
    c_lin, c_uni, c_sub = _franchise_traits(candidate_row)
    if a_lin and c_lin and (a_lin & c_lin):
        return 1.00
    if (a_sub & c_sub) and ((a_uni & c_uni) or (a_lin & c_lin)):
        return 1.00
    if a_uni and c_uni and (a_uni & c_uni):
        return 0.85
    if a_sub and c_sub and (a_sub & c_sub):
        return 0.40
    return 0.0


def _source_score_idf(
    anchor_source_ids: set[int],
    candidate_source_ids: set[int],
    source_idfs: dict[int, float],
) -> float:
    """V2 source lane: max IDF over the shared source-material types.

    Common types like ``novel`` (high df → low idf) collapse to ~0.20;
    rare types (``video_game``, ``fairy_tale``, ``stage_play``,
    ``comic_book``) stay near 1.0. ``max`` over shared traits — not sum
    — so a single rare match dominates and two common-tag overlaps
    don't compound into a full score.
    """
    if not anchor_source_ids or not candidate_source_ids:
        return 0.0
    shared = anchor_source_ids & candidate_source_ids
    if not shared:
        return 0.0
    return max(source_idfs.get(t, 0.0) for t in shared)


def _quality_score_v2(
    bucket: str,
    candidate_row: dict,
    award_signals: SimilarityAwardSignals | None,
) -> float:
    """V2 quality lane: per-bucket formulas, always-on across buckets.

    cult_garbage: weight low_reception + popularity_match heavily, with
        a 10% boost for any Razzie evidence (caught the cult-classic
        signal even when reception score isn't extreme).
    prestige: dominated by high_reception; popularity_or_award and a
        non-Razzie award bonus together carry the remaining 40%, so
        Best Picture winners outscore equally-reviewed but non-awarded
        peers.
    middle: a soft notability prior — popularity-weighted with reception
        as a tie-breaker. Always active (V1 zeroed this bucket out).
    """
    reception = candidate_row.get("reception_score")
    pop_pct = candidate_row.get("popularity_percentile") or 0.0
    non_razzie = award_signals.non_razzie_score if award_signals else 0.0
    razzie = award_signals.razzie_score if award_signals else 0.0

    if bucket == "cult_garbage":
        low_reception_match = (
            _clamp((50.0 - reception) / 30.0) if reception is not None else 0.0
        )
        pop_match = _clamp((pop_pct - 0.75) / 0.20)
        return _clamp(
            0.40 * low_reception_match + 0.50 * pop_match + 0.10 * razzie
        )

    if bucket == "prestige":
        high_reception_match = (
            _clamp((reception - 75.0) / 20.0) if reception is not None else 0.0
        )
        pop_or_award = max(_clamp((pop_pct - 0.50) / 0.30), non_razzie)
        # Award contribution is additive on top of pop_or_award (clamped
        # via the outer _clamp so a Best Picture winner with strong
        # popularity doesn't exceed 1.0).
        return _clamp(
            0.80 * high_reception_match + 0.20 * pop_or_award + 0.20 * non_razzie
        )

    # middle bucket
    reception_norm = (reception or 0.0) / 100.0
    return _clamp(0.80 * pop_pct + 0.20 * reception_norm)


def _format_score(anchor_bucket: FormatBucket, candidate_row: dict) -> float:
    """Single-anchor format lane: binary same-bucket-or-not."""
    candidate_bucket = format_bucket(candidate_row.get("keyword_ids") or ())
    return 1.0 if candidate_bucket == anchor_bucket else 0.0


def _medium_tags_for_movie(row: dict) -> set[int]:
    """Extract the medium-related keyword IDs from a movie row.

    The medium registry only knows about MEDIUM_TAG_IDS; everything else
    in keyword_ids is irrelevant to medium scoring. Returning a set lets
    callers feed it directly to ``medium_score``.
    """
    return _as_int_set(row.get("keyword_ids")) & MEDIUM_TAG_IDS


def _medium_multiplier(anchor_tags: set[int], candidate_tags: set[int]) -> float:
    """Map the medium similarity score onto the [floor, 1.0] multiplier.

    Perfect medium agreement leaves combined score unchanged; full
    mismatch (live-action anchor vs. animation candidate) drops it by
    ``1 - MEDIUM_MULTIPLIER_FLOOR`` (15% by default). Cross-medium-but-
    related (CG anchor vs. stop-motion candidate, score 0.50) gets a
    partial penalty (multiplier ≈ 0.925).

    When either side has no medium tags the score is 0.0, which here
    means "we can't tell" — applying the floor as the multiplier is a
    soft penalty consistent with treating no-signal candidates as
    cross-medium. Lane code can short-circuit by checking the tag sets
    if it wants to skip this entirely.
    """
    if not anchor_tags or not candidate_tags:
        return 1.0
    score = medium_score(anchor_tags, candidate_tags)
    return MEDIUM_MULTIPLIER_FLOOR + MEDIUM_MULTIPLIER_RANGE * score


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 0.0:
        return vector
    return [v / norm for v in vector]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _cohesion_weight(avg_pairwise_cosine: float) -> float:
    return _clamp((avg_pairwise_cosine - 0.55) / 0.30, 0.10, 1.0)


def _record_vectors(record: object) -> dict[VectorName, list[float]]:
    raw_vectors = getattr(record, "vector", None)
    if not isinstance(raw_vectors, dict):
        return {}
    out: dict[VectorName, list[float]] = {}
    for vector_name in VectorName:
        raw = raw_vectors.get(vector_name.value)
        if raw is not None:
            out[vector_name] = [float(v) for v in raw]
    return out


async def _load_anchor_vectors(
    anchor_ids: list[int],
) -> dict[int, dict[VectorName, list[float]]]:
    records = await qdrant_client.retrieve(
        collection_name=COLLECTION_ALIAS,
        ids=anchor_ids,
        with_payload=False,
        with_vectors=[space.value for space in VectorName],
    )
    return {int(getattr(record, "id")): _record_vectors(record) for record in records}


async def _query_space(
    query_vector: list[float],
    vector_name: VectorName,
    *,
    limit: int,
) -> list[tuple[int, float]]:
    response = await qdrant_client.query_points(
        collection_name=COLLECTION_ALIAS,
        query=query_vector,
        using=vector_name.value,
        limit=limit,
        with_payload=False,
        with_vectors=False,
        search_params=QDRANT_SEARCH_PARAMS,
    )
    return [(int(point.id), float(point.score)) for point in response.points]


def _normalize_space_results(
    rows: list[tuple[int, float]],
    *,
    excluded_ids: set[int],
) -> dict[int, float]:
    raw = {
        movie_id: min(score, 1.0)
        for movie_id, score in rows
        if movie_id not in excluded_ids and score > 0.0
    }
    if not raw:
        return {}
    return normalize_blended_scores(raw)


async def _run_single_anchor_shape_search(
    anchor_id: int,
    anchor_vectors: dict[VectorName, list[float]],
    *,
    qdrant_limit: int,
) -> tuple[dict[int, float], dict[str, float]]:
    active_spaces = {
        space: VECTOR_BASE_WEIGHTS_SINGLE[space]
        for space in VectorName
        if space in anchor_vectors
    }
    space_weights = _normalize_vector_weights(active_spaces)
    if not space_weights:
        return {}, {}

    rows_by_space = await asyncio.gather(
        *(
            _query_space(
                anchor_vectors[space],
                space,
                limit=qdrant_limit + 1,
            )
            for space in space_weights
        )
    )

    scores: dict[int, float] = {}
    for space, rows in zip(space_weights, rows_by_space, strict=True):
        normalized = _normalize_space_results(rows, excluded_ids={anchor_id})
        for movie_id, score in normalized.items():
            scores[movie_id] = scores.get(movie_id, 0.0) + space_weights[space] * score

    return _normalize_lane_to_max(scores), {
        space.value: weight for space, weight in space_weights.items()
    }


async def _run_multi_anchor_shape_search(
    anchor_ids: list[int],
    vectors_by_anchor: dict[int, dict[VectorName, list[float]]],
    *,
    qdrant_limit: int,
) -> tuple[dict[int, float], dict[str, float], dict[str, float], float]:
    """Run multi-anchor shape search and return scores + cohesion debug.

    Final tuple element is the mean pairwise cosine across active spaces;
    callers use it to drive the V2 shape-lane scaling and the
    low-cohesion fallback gate.
    """
    usable: dict[VectorName, list[list[float]]] = {}
    for space in VectorName:
        vectors: list[list[float]] = []
        for anchor_id in anchor_ids:
            vector = vectors_by_anchor.get(anchor_id, {}).get(space)
            if vector is None:
                break
            vectors.append(_l2_normalize(vector))
        else:
            usable[space] = vectors

    raw_space_weights: dict[VectorName, float] = {}
    cohesion_debug: dict[str, float] = {}
    avg_pairwise_by_space: dict[VectorName, float] = {}
    centroid_by_space: dict[VectorName, list[float]] = {}

    for space, vectors in usable.items():
        pairwise: list[float] = []
        for i, left in enumerate(vectors):
            for right in vectors[i + 1:]:
                pairwise.append(_dot(left, right))
        avg_pairwise = sum(pairwise) / len(pairwise) if pairwise else 1.0
        avg_pairwise_by_space[space] = avg_pairwise
        # cohesion_weight stays clamped to [0.10, 1.00] for the per-space
        # vector mix — negative weights inside the mix don't make sense.
        # The expanded shape-lane scaling clamp is applied separately by
        # the caller using mean_pairwise_cosine.
        cohesion = _cohesion_weight(avg_pairwise)
        cohesion_debug[space.value] = cohesion
        raw_space_weights[space] = (
            0.75 * cohesion + 0.25 * VECTOR_BASE_WEIGHTS_MULTI[space]
        )

        dimension = len(vectors[0])
        averaged = [
            sum(vector[idx] for vector in vectors) / len(vectors)
            for idx in range(dimension)
        ]
        centroid_by_space[space] = _l2_normalize(averaged)

    # Mean pairwise cosine across active spaces — drives V2 shape-lane
    # scaling and the low-cohesion fallback gate.
    if avg_pairwise_by_space:
        mean_pairwise_cosine = sum(avg_pairwise_by_space.values()) / len(
            avg_pairwise_by_space
        )
    else:
        mean_pairwise_cosine = 0.0

    space_weights = _normalize_vector_weights(raw_space_weights)
    if not space_weights:
        return {}, {}, cohesion_debug, mean_pairwise_cosine

    rows_by_space = await asyncio.gather(
        *(
            _query_space(
                centroid_by_space[space],
                space,
                limit=qdrant_limit + len(anchor_ids),
            )
            for space in space_weights
        )
    )

    excluded = set(anchor_ids)
    scores: dict[int, float] = {}
    for space, rows in zip(space_weights, rows_by_space, strict=True):
        normalized = _normalize_space_results(rows, excluded_ids=excluded)
        for movie_id, score in normalized.items():
            scores[movie_id] = scores.get(movie_id, 0.0) + space_weights[space] * score

    return (
        _normalize_lane_to_max(scores),
        {space.value: weight for space, weight in space_weights.items()},
        cohesion_debug,
        mean_pairwise_cosine,
    )


def _shape_raw_for_multi_anchor(mean_pairwise_cosine: float) -> float:
    """V2: scale shape-lane raw weight by overall vector cohesion.

    Maps mean_pairwise_cosine through clamp((x - 0.55)/0.30, -0.40, 1.00)
    onto a [-0.40, 1.00] signal. shape_raw = 0.60 * (1 + signal) ranges
    over [0.36, 1.20] — boosting shape on coherent sets (LotR trilogy
    sits near 1.20) and penalizing it on incoherent ones (chaotic mixed
    anchor sets sit near 0.36).
    """
    signal = _clamp((mean_pairwise_cosine - 0.55) / 0.30, -0.40, 1.00)
    return 0.60 * (1.0 + signal)


def _build_results(
    *,
    anchor_ids: list[int],
    lane_scores: dict[LaneName, dict[int, float]],
    lane_weights: dict[LaneName, float],
    limit: int,
    # V2 post-additive multipliers — each is a per-movie lookup. None means
    # "this signal isn't relevant for this flow" (e.g., country/language is
    # multi-anchor only; medium tags may be empty for live-action anchors).
    studio_score_by_movie: dict[int, float] | None = None,
    medium_multiplier_by_movie: dict[int, float] | None = None,
    low_conf_franchise_score_by_movie: dict[int, float] | None = None,
    country_consensus_match_by_movie: dict[int, bool] | None = None,
    anchor_format_bucket: FormatBucket | None = None,
    enforce_format_top_lock: bool = False,
) -> tuple[list[SimilarMovieResult], dict[LaneName, int]]:
    """Combine per-lane scores into ranked candidates with V2 multipliers.

    Pipeline:
      1. Sum lane_weights * lane_scores across ADDITIVE_LANES.
      2. Apply V2 multipliers (medium, studio, low-confidence franchise,
         multi-anchor country/language coherence) on shape-qualifying
         candidates.
      3. Weave: V1 dominance/franchise caps plus the V2 top-5 format lock.

    Studio is debug-only in lane_scores — its multiplier path is the
    studio_score_by_movie dict, not a contribution to the additive sum.
    """
    excluded = set(anchor_ids)
    candidate_ids: set[int] = set()
    for scores in lane_scores.values():
        candidate_ids.update(scores)
    candidate_ids -= excluded

    studio_score_by_movie = studio_score_by_movie or {}
    medium_multiplier_by_movie = medium_multiplier_by_movie or {}
    low_conf_franchise_score_by_movie = low_conf_franchise_score_by_movie or {}
    country_consensus_match_by_movie = country_consensus_match_by_movie or {}

    candidates: list[_CandidateScore] = []
    for movie_id in candidate_ids:
        per_lane = {
            lane: _clamp(lane_scores.get(lane, {}).get(movie_id, 0.0))
            for lane in ALL_LANES
        }
        sources = [lane for lane in ALL_LANES if per_lane[lane] > 0.0]
        if not sources:
            continue

        # Additive sum — only ADDITIVE_LANES (excludes studio).
        contributions = {
            lane: lane_weights.get(lane, 0.0) * per_lane[lane]
            for lane in ADDITIVE_LANES
        }
        # Dominance is computed on additive contributions; studio can never
        # be the dominant lane in V2 since it's not additive.
        dominant = max(
            ADDITIVE_LANES,
            key=lambda lane: (contributions[lane], per_lane[lane]),
        )
        score = sum(contributions.values())
        shape_score = per_lane["shape"]

        # Studio multiplier — applies to candidates that already cleared
        # the shape gate, so on-brand-but-low-shape noise can't ride the
        # multiplier into the top of the list.
        studio_score = studio_score_by_movie.get(movie_id, 0.0)
        if studio_score > 0.0 and shape_score >= STUDIO_MULTIPLIER_SHAPE_GATE:
            score *= 1.0 + STUDIO_MULTIPLIER_STRENGTH * studio_score

        # Low-confidence franchise multiplier — separate path from the
        # additive franchise lane; only fires when the anchor lineage was
        # below the confidence/consistency thresholds.
        low_conf_franchise = low_conf_franchise_score_by_movie.get(movie_id, 0.0)
        if (
            low_conf_franchise > 0.0
            and shape_score >= LOW_CONF_FRANCHISE_SHAPE_GATE
        ):
            score *= 1.0 + LOW_CONF_FRANCHISE_MULTIPLIER_STRENGTH * low_conf_franchise

        # Country/language coherence multiplier — multi-anchor only.
        if movie_id in country_consensus_match_by_movie:
            if country_consensus_match_by_movie[movie_id]:
                score *= COUNTRY_CONSENSUS_BOOST
            else:
                score *= COUNTRY_CONSENSUS_PENALTY

        # Medium multiplier — applied unconditionally so cross-medium
        # candidates lose ground regardless of how strong other lanes are.
        medium_mult = medium_multiplier_by_movie.get(movie_id, 1.0)
        score *= medium_mult

        # Studio score is preserved in lane_scores for debug visibility
        # even though it doesn't participate in the additive sum.
        per_lane["studio"] = _clamp(studio_score)
        if studio_score > 0.0 and "studio" not in sources:
            sources.append("studio")

        candidates.append(
            _CandidateScore(
                movie_id=movie_id,
                score=score,
                lane_scores=per_lane,
                candidate_sources=sources,
                dominant_lane=dominant,
            )
        )

    woven = _weave_candidates(
        candidates,
        limit=limit,
        anchor_format_bucket=anchor_format_bucket,
        enforce_format_top_lock=enforce_format_top_lock,
    )
    ranked = [
        SimilarMovieResult(
            movie_id=c.movie_id,
            score=c.score,
            evidence=LaneEvidence(
                lane_scores=c.lane_scores,
                candidate_sources=c.candidate_sources,
                dominant_lane=c.dominant_lane,
            ),
        )
        for c in woven
    ]
    # Debug counts cover every lane that contributed at least one candidate
    # — including studio, which doesn't appear in lane_scores when it's
    # multiplier-only, so synthesize from studio_score_by_movie.
    counts: dict[LaneName, int] = {
        lane: len(scores) for lane, scores in lane_scores.items()
    }
    if studio_score_by_movie:
        counts["studio"] = sum(
            1 for v in studio_score_by_movie.values() if v > 0.0
        )
    return ranked, counts


def _base_sort_key(candidate: _CandidateScore) -> tuple[float, int, int]:
    return (-candidate.score, -len(candidate.candidate_sources), candidate.movie_id)


def _can_enter_top_section(
    candidate: _CandidateScore,
    *,
    best_score: float,
    dominant_counts: dict[LaneName, int],
    franchise_count: int,
    top_format_lock_active: bool,
) -> bool:
    """Decide whether a candidate may enter the top section of the page.

    V2 changes vs. V1: the studio dominance / weak-studio-source guard is
    gone (studio is no longer dominant in any candidate, since it's
    multiplier-only). The format top-5 lock is enforced via the
    `top_format_lock_active` flag — candidates that don't share the
    anchor's format bucket are deferred while we're filling those first
    slots.
    """
    dominant_lane = candidate.dominant_lane
    shape_score = candidate.lane_scores["shape"]

    if top_format_lock_active and candidate.lane_scores["format"] < 1.0:
        return False
    if dominant_counts.get(dominant_lane, 0) >= MAX_TOP_DOMINANT_LANE:
        return False
    if candidate.lane_scores["franchise"] > 0.0 and franchise_count >= MAX_TOP_FRANCHISE:
        return False
    if dominant_lane != "shape" and candidate.score < best_score - COMPETITIVE_BAND:
        return False
    if dominant_lane == "source" and shape_score < 0.45:
        return False
    if dominant_lane == "franchise" and shape_score < 0.35:
        return False
    if dominant_lane == "director" and shape_score < 0.40:
        return False
    return True


def _weave_candidates(
    candidates: list[_CandidateScore],
    *,
    limit: int,
    anchor_format_bucket: FormatBucket | None = None,
    enforce_format_top_lock: bool = False,
) -> list[_CandidateScore]:
    """Order candidates with V2 lane-variety + format-lock constraints.

    Top section is built greedily from the score-sorted candidates. The
    first ``TOP_FORMAT_LOCK`` slots only accept candidates sharing the
    anchor format bucket (when ``enforce_format_top_lock`` is set);
    after that, regular dominance/franchise caps apply through slot 10.
    Anything that didn't make the top section is appended in score order
    so callers still receive a full ``limit``-sized list.
    """
    if not candidates or limit <= 0:
        return []

    base_sorted = sorted(candidates, key=_base_sort_key)
    best_score = base_sorted[0].score
    top: list[_CandidateScore] = []
    deferred: list[_CandidateScore] = []
    dominant_counts: dict[LaneName, int] = {}
    franchise_count = 0
    top_section_cap = min(TOP_SECTION_SIZE, limit)

    # Single pass: each candidate either enters top, gets deferred to the
    # remainder, or rolls past once the top section is full.
    for candidate in base_sorted:
        if len(top) >= top_section_cap:
            deferred.append(candidate)
            continue
        top_format_lock_active = (
            enforce_format_top_lock
            and anchor_format_bucket is not None
            and len(top) < TOP_FORMAT_LOCK
        )
        if _can_enter_top_section(
            candidate,
            best_score=best_score,
            dominant_counts=dominant_counts,
            franchise_count=franchise_count,
            top_format_lock_active=top_format_lock_active,
        ):
            top.append(candidate)
            dominant_counts[candidate.dominant_lane] = (
                dominant_counts.get(candidate.dominant_lane, 0) + 1
            )
            if candidate.lane_scores["franchise"] > 0.0:
                franchise_count += 1
        else:
            deferred.append(candidate)

    seen = {candidate.movie_id for candidate in top}
    remainder = [candidate for candidate in base_sorted if candidate.movie_id not in seen]
    return (top + remainder)[:limit]


async def _resolve_similarity_title_anchor(flow_data: SimilarityFlowData) -> int | None:
    title = flow_data.similar_search_title.strip()
    if not title:
        raise ValueError("similar_search_title must be non-empty.")

    pattern = escape_like(normalize_string(title))
    if not pattern:
        return None

    title_ids = await fetch_movie_ids_with_title_like(pattern)
    if not title_ids:
        return None

    rows = await fetch_similarity_signal_rows(list(title_ids))
    candidates = list(rows.values())
    if flow_data.release_year is not None:
        candidates = [
            row for row in candidates if _release_year(row) == flow_data.release_year
        ]
    if not candidates:
        return None

    candidates.sort(
        key=lambda row: (
            -(row.get("popularity_percentile") or -1.0),
            -(row.get("reception_score") or -1.0),
            -(_release_year(row) or -1),
            int(row["movie_id"]),
        )
    )
    return int(candidates[0]["movie_id"])


async def run_similarity_search(
    flow_data: SimilarityFlowData,
    *,
    limit: int = 50,
    qdrant_limit: int = DEFAULT_QDRANT_LIMIT,
) -> SimilarMoviesSearchResult:
    """Run Step-0's title-based similarity flow."""
    if not flow_data.should_be_searched:
        raise ValueError(
            "run_similarity_search called with should_be_searched=False; "
            "the caller must gate on Step 0's flow decision."
        )
    anchor_id = await _resolve_similarity_title_anchor(flow_data)
    if anchor_id is None:
        return SimilarMoviesSearchResult(
            anchor_movie_ids=[],
            ranked=[],
            active_anchor_types=[],
            debug=SimilarMoviesDebug(vector_space_weights={}),
        )
    return await run_similar_movies_for_ids(
        [anchor_id],
        limit=limit,
        qdrant_limit=qdrant_limit,
    )


async def run_similar_movies_for_ids(
    tmdb_ids: list[int],
    *,
    limit: int = 50,
    qdrant_limit: int = DEFAULT_QDRANT_LIMIT,
    quality_limit: int = DEFAULT_QUALITY_LIMIT,
) -> SimilarMoviesSearchResult:
    """Run similar-movies search for one or more anchor TMDB IDs."""
    anchor_ids = list(dict.fromkeys(int(mid) for mid in tmdb_ids))
    if not anchor_ids:
        raise ValueError("tmdb_ids must contain at least one movie ID.")

    anchor_rows = await fetch_similarity_signal_rows(anchor_ids)
    missing = [mid for mid in anchor_ids if mid not in anchor_rows]
    if missing:
        raise LookupError(f"movie_card rows not found for tmdb_ids={missing}")

    vectors_by_anchor = await _load_anchor_vectors(anchor_ids)
    studio_entries_by_company_id = await _load_studio_entries_by_company_id()
    director_terms_by_anchor = await fetch_director_term_ids_for_movies(anchor_ids)

    if len(anchor_ids) == 1:
        return await _run_single_anchor_similarity(
            anchor_ids[0],
            anchor_rows,
            vectors_by_anchor,
            studio_entries_by_company_id,
            director_terms_by_anchor,
            limit=limit,
            qdrant_limit=qdrant_limit,
            quality_limit=quality_limit,
        )

    return await _run_multi_anchor_similarity(
        anchor_ids,
        anchor_rows,
        vectors_by_anchor,
        studio_entries_by_company_id,
        director_terms_by_anchor,
        limit=limit,
        qdrant_limit=qdrant_limit,
        quality_limit=quality_limit,
    )


async def _run_single_anchor_similarity(
    anchor_id: int,
    anchor_rows: dict[int, dict],
    vectors_by_anchor: dict[int, dict[VectorName, list[float]]],
    studio_entries_by_company_id: dict[int, list[StudioSimilarityEntry]],
    director_terms_by_anchor: dict[int, set[int]],
    *,
    limit: int,
    qdrant_limit: int,
    quality_limit: int,
) -> SimilarMoviesSearchResult:
    anchor_row = anchor_rows[anchor_id]
    quality_bucket = _quality_bucket(anchor_row)
    cult_or_prestige = quality_bucket in {"cult_garbage", "prestige"}

    anchor_directors = director_terms_by_anchor.get(anchor_id, set())
    anchor_lineage, anchor_universe, anchor_subgroups = _franchise_traits(anchor_row)
    anchor_source_ids = _as_int_set(anchor_row.get("source_material_type_ids"))
    anchor_studio_entries = _active_studio_entries(
        anchor_row, studio_entries_by_company_id
    )
    anchor_studio_company_ids = {entry.company_id for entry in anchor_studio_entries}
    anchor_medium_tags = _medium_tags_for_movie(anchor_row)
    anchor_format_bucket = format_bucket(anchor_row.get("keyword_ids") or ())

    # Pre-fetches for V2 lane data. Director strengths and franchise
    # confidence are MV reads; source IDFs use the unified trait-IDF MV
    # filtered to kind=4. Medium IDFs are loaded lazily once per process
    # and cached. None of these depend on the candidate pool, so they run
    # in parallel with shape search and the candidate-generation lanes.
    director_strengths_task = fetch_director_strengths(list(anchor_directors))
    franchise_confidence_task = fetch_franchise_confidence(list(anchor_lineage))
    source_idf_task = fetch_trait_idfs(
        [(TRAIT_KIND_SOURCE_MATERIAL, t) for t in anchor_source_ids]
    )
    medium_idf_task = load_medium_idfs()

    shape_task = _run_single_anchor_shape_search(
        anchor_id,
        vectors_by_anchor.get(anchor_id, {}),
        qdrant_limit=qdrant_limit,
    )
    director_task = fetch_director_movie_terms(anchor_directors)
    franchise_task = fetch_similarity_franchise_candidates(
        lineage_entry_ids=anchor_lineage,
        shared_universe_entry_ids=anchor_universe,
        subgroup_entry_ids=anchor_subgroups,
    )
    studio_task = fetch_movie_ids_by_production_company_ids(anchor_studio_company_ids)
    source_task = fetch_similarity_source_candidates(anchor_source_ids)
    # Quality candidate lane is recall-repair only — middle-bucket flow
    # leans on candidates surfaced by other lanes plus shape.
    quality_task = (
        fetch_similarity_quality_candidates(bucket=quality_bucket, limit=quality_limit)
        if cult_or_prestige
        else _empty_set()
    )

    (
        (shape_scores, vector_space_weights),
        director_candidate_terms,
        franchise_candidate_ids,
        studio_candidate_ids,
        source_candidate_ids,
        quality_candidate_ids,
        director_strengths,
        franchise_confidence,
        source_idfs_pairs,
        medium_idfs,
    ) = await asyncio.gather(
        shape_task,
        director_task,
        franchise_task,
        studio_task,
        source_task,
        quality_task,
        director_strengths_task,
        franchise_confidence_task,
        source_idf_task,
        medium_idf_task,
    )

    # Source IDFs come back keyed by (kind, trait_id); flatten to trait_id
    # only for the lane scorer (kind is constant within this lane).
    source_idfs: dict[int, float] = {
        trait_id: idf for (_, trait_id), idf in source_idfs_pairs.items()
    }

    # V2 selective rare-medium retrieval: pull in additional candidates
    # sharing any rare medium tag with the anchor (idf >= threshold) so
    # e.g. a stop-motion anchor still surfaces other stop-motion films
    # even when the centroid-driven shape lane misses them. LIVE_ACTION
    # is excluded — it covers the bulk of the catalog so an overlap
    # query would explode the pool with no signal.
    rare_medium_tags = {
        tag
        for tag in anchor_medium_tags
        if medium_idfs.get(tag, 0.0) >= RARE_MEDIUM_IDF_THRESHOLD
        and tag != LIVE_ACTION_TAG_ID
    }
    rare_medium_candidate_ids: set[int] = (
        await fetch_movie_ids_by_overall_keywords(list(rare_medium_tags))
        if rare_medium_tags
        else set()
    )

    # V2 director-signature anchor type — fires when at least one of the
    # anchor's directors clears the auteur threshold in mv_director_strength.
    has_director_signature = any(
        strength >= DIRECTOR_SIGNATURE_STRENGTH_THRESHOLD
        for strength in director_strengths.values()
    )

    # V2 franchise confidence: a single anchor lineage clearing both gates
    # is enough to keep the additive-lane behavior. If no lineage clears
    # them, the franchise lane drops to a multiplicative nudge to prevent
    # low-quality direct-to-DVD spinoffs from dominating top results.
    franchise_high_confidence = any(
        conf >= FRANCHISE_HIGH_CONF_CONFIDENCE
        and consist >= FRANCHISE_HIGH_CONF_CONSISTENCY
        for (conf, consist) in franchise_confidence.values()
    )
    franchise_low_confidence = (
        bool(franchise_candidate_ids - {anchor_id}) and not franchise_high_confidence
    )

    active_anchor_types: list[AnchorType] = ["standard_shape"]
    if quality_bucket in {"cult_garbage", "prestige"}:
        active_anchor_types.append(quality_bucket)  # type: ignore[arg-type]
    # Suppress franchise_dominant on low-confidence anchors per V2 spec.
    if franchise_high_confidence:
        active_anchor_types.append("franchise_dominant")
    if anchor_studio_entries:
        active_anchor_types.append("studio_lineage")
    if anchor_source_ids:
        active_anchor_types.append("source_material")
    if has_director_signature:
        active_anchor_types.append("director_signature")

    raw_lane_weights, lane_weights = _single_anchor_lane_weights(
        active_anchor_types,
        franchise_low_confidence=franchise_low_confidence,
    )

    candidate_ids = set(shape_scores)
    candidate_ids.update(director_candidate_terms)
    candidate_ids.update(franchise_candidate_ids)
    candidate_ids.update(studio_candidate_ids)
    candidate_ids.update(source_candidate_ids)
    candidate_ids.update(quality_candidate_ids)
    candidate_ids.update(rare_medium_candidate_ids)
    candidate_ids.discard(anchor_id)

    candidate_rows = await fetch_similarity_signal_rows(list(candidate_ids))
    # Always-on quality lane needs award signals for prestige/cult buckets;
    # middle-bucket flow uses popularity+reception only and skips the read.
    award_signals = (
        await fetch_similarity_award_signals(list(candidate_ids))
        if cult_or_prestige
        else {}
    )

    # ----- Per-lane scoring -----
    # Director: V2 auteur prior — score by max director_strength across
    # shared term IDs. Candidates whose only shared director was filtered
    # out of the MV (single-film directors) get score 0 since the MV row
    # is absent.
    director_scores: dict[int, float] = {}
    for movie_id, terms in director_candidate_terms.items():
        if movie_id == anchor_id:
            continue
        shared = terms & anchor_directors
        if not shared:
            continue
        strength = max((director_strengths.get(t, 0.0) for t in shared), default=0.0)
        if strength > 0.0:
            director_scores[movie_id] = strength

    # Franchise: V2 subgroup-gated scoring on the candidate set. Even when
    # the lane is multiplicative-only (low-confidence), we still compute
    # the per-candidate score for the multiplier path.
    raw_franchise_scores = {
        movie_id: _franchise_score_v2(anchor_row, row)
        for movie_id, row in candidate_rows.items()
        if movie_id in franchise_candidate_ids
    }

    # Studio: scored on every studio-candidate; the multiplier path filters
    # by shape gate, so we don't need to do that here.
    studio_scores = {
        movie_id: _studio_score(
            anchor_row,
            row,
            anchor_studio_entries,
            _active_studio_entries(row, studio_entries_by_company_id),
        )
        for movie_id, row in candidate_rows.items()
        if movie_id in studio_candidate_ids
    }

    # Source: V2 IDF-weighted max over shared types.
    source_scores = {
        movie_id: _source_score_idf(
            anchor_source_ids,
            _as_int_set(row.get("source_material_type_ids")),
            source_idfs,
        )
        for movie_id, row in candidate_rows.items()
    }

    # Quality: V2 always-on per-bucket formula across every candidate row.
    quality_scores = {
        movie_id: _quality_score_v2(
            quality_bucket, row, award_signals.get(movie_id)
        )
        for movie_id, row in candidate_rows.items()
    }

    # Format: V2 binary same-bucket-or-not, every candidate.
    format_scores = {
        movie_id: _format_score(anchor_format_bucket, row)
        for movie_id, row in candidate_rows.items()
    }

    # Medium multiplier per candidate (computed once, applied in build).
    medium_multiplier_by_movie: dict[int, float] = {}
    if anchor_medium_tags:
        for movie_id, row in candidate_rows.items():
            candidate_medium = _medium_tags_for_movie(row)
            medium_multiplier_by_movie[movie_id] = _medium_multiplier(
                anchor_medium_tags, candidate_medium
            )

    # Split franchise into additive vs. multiplicative paths based on
    # confidence. Additive entries land in lane_scores; multiplicative
    # entries land in low_conf_franchise_score_by_movie.
    franchise_additive: dict[int, float] = {}
    low_conf_franchise: dict[int, float] = {}
    for movie_id, score in raw_franchise_scores.items():
        if score <= 0.0:
            continue
        if franchise_high_confidence:
            franchise_additive[movie_id] = score
        else:
            low_conf_franchise[movie_id] = score

    lane_scores: dict[LaneName, dict[int, float]] = {
        "shape": shape_scores,
        "director": {mid: s for mid, s in director_scores.items() if s > 0.0},
        "franchise": franchise_additive,
        # studio appears in lane_scores only for debug visibility — its
        # contribution to combined score is the multiplier path.
        "studio": {},
        "source": {mid: s for mid, s in source_scores.items() if s > 0.0},
        "quality": {mid: s for mid, s in quality_scores.items() if s > 0.0},
        "format": {mid: s for mid, s in format_scores.items() if s > 0.0},
        # Multi-only lanes stay empty in single-anchor flow.
        "themes": {},
        "cast": {},
        "specific_award": {},
    }

    ranked, counts = _build_results(
        anchor_ids=[anchor_id],
        lane_scores=lane_scores,
        lane_weights=lane_weights,
        limit=limit,
        studio_score_by_movie=studio_scores,
        medium_multiplier_by_movie=medium_multiplier_by_movie,
        low_conf_franchise_score_by_movie=low_conf_franchise,
        anchor_format_bucket=anchor_format_bucket,
        enforce_format_top_lock=True,
    )
    return SimilarMoviesSearchResult(
        anchor_movie_ids=[anchor_id],
        ranked=ranked,
        active_anchor_types=active_anchor_types,
        debug=SimilarMoviesDebug(
            vector_space_weights=vector_space_weights,
            raw_lane_weights=raw_lane_weights,
            normalized_lane_weights=lane_weights,
            candidate_counts_by_lane=counts,
            anchor_format_bucket=anchor_format_bucket,
            anchor_medium_tags=sorted(anchor_medium_tags),
            franchise_high_confidence=franchise_high_confidence,
        ),
    )


async def _run_multi_anchor_similarity(
    anchor_ids: list[int],
    anchor_rows: dict[int, dict],
    vectors_by_anchor: dict[int, dict[VectorName, list[float]]],
    studio_entries_by_company_id: dict[int, list[StudioSimilarityEntry]],
    director_terms_by_anchor: dict[int, set[int]],
    *,
    limit: int,
    qdrant_limit: int,
    quality_limit: int,
) -> SimilarMoviesSearchResult:
    """V2 multi-anchor similarity flow.

    Adds five new lanes/multipliers vs. V1: themes (IDF over keyword +
    concept + genre), cast (top-3-billed overlap), specific_award (3-tier
    category-tag taxonomy), country/language coherence multiplier, and
    shape-lane scaling driven by mean vector cohesion. Source lane is
    upgraded to per-anchor IDF weighting. Studio is removed from the
    additive sum and applied as a multiplier (consistent with single-
    anchor V2).
    """
    n = len(anchor_ids)

    # Per-anchor trait sets — built once and reused for cohesion +
    # candidate scoring. Order matches anchor_ids so element i is anchor
    # i's traits.
    anchor_studio_entries = {
        anchor_id: _active_studio_entries(row, studio_entries_by_company_id)
        for anchor_id, row in anchor_rows.items()
    }
    director_trait_sets = [
        set(director_terms_by_anchor.get(anchor_id, set()))
        for anchor_id in anchor_ids
    ]
    franchise_trait_sets = [
        set().union(*_franchise_traits(anchor_rows[anchor_id]))
        for anchor_id in anchor_ids
    ]
    studio_trait_sets = [
        {entry.company_id for entry in anchor_studio_entries[anchor_id]}
        for anchor_id in anchor_ids
    ]
    source_trait_sets = [
        _as_int_set(anchor_rows[anchor_id].get("source_material_type_ids"))
        for anchor_id in anchor_ids
    ]
    quality_trait_sets: list[set[str]] = [
        {bucket}
        if (bucket := _quality_bucket(anchor_rows[anchor_id]))
        in {"cult_garbage", "prestige"}
        else set()
        for anchor_id in anchor_ids
    ]
    themes_trait_sets = [
        _themes_traits_for_movie(anchor_rows[anchor_id]) for anchor_id in anchor_ids
    ]
    format_trait_sets: list[set[str]] = [
        {format_bucket(anchor_rows[anchor_id].get("keyword_ids") or ())}
        for anchor_id in anchor_ids
    ]
    country_trait_sets = [
        country_set(anchor_rows[anchor_id].get("keyword_ids") or ())
        for anchor_id in anchor_ids
    ]

    # Repeated traits — cohesion uses these across every lane.
    # Sync cohesion for every lane that doesn't need a DB read. cast and
    # specific_award are added below once their anchor-side fetches return.
    cohesion_by_lane = {
        "director": _metadata_cohesion(director_trait_sets),
        "franchise": _metadata_cohesion(franchise_trait_sets),
        "studio": _metadata_cohesion(studio_trait_sets),
        "source": _metadata_cohesion(source_trait_sets),
        "quality": _metadata_cohesion(quality_trait_sets),
        "themes": _metadata_cohesion(themes_trait_sets),
        "format": _metadata_cohesion(format_trait_sets),
    }

    repeated_quality_bucket = _repeated_quality_bucket(quality_trait_sets)
    repeated_format_bucket = _repeated_format_bucket(format_trait_sets)
    consensus_countries = _consensus_country_set(country_trait_sets)

    # Multi-anchor candidate generation. director/franchise/studio/source/
    # quality candidate fetches gate on the sync cohesion above so we don't
    # waste DB calls on lanes with no repeated traits. cast and award are
    # cheap and small (~3 rows per anchor, ≤ N total awards), so they fire
    # unconditionally — the second-round candidate-side cast/award reads
    # later are still gated on the cohesion derived from these results.
    director_terms = set().union(*director_trait_sets) if director_trait_sets else set()
    franchise_traits = (
        set().union(*franchise_trait_sets) if franchise_trait_sets else set()
    )
    studio_company_ids = (
        set().union(*studio_trait_sets) if studio_trait_sets else set()
    )
    source_ids = set().union(*source_trait_sets) if source_trait_sets else set()

    shape_task = _run_multi_anchor_shape_search(
        anchor_ids, vectors_by_anchor, qdrant_limit=qdrant_limit
    )
    director_task = (
        fetch_director_movie_terms(director_terms)
        if cohesion_by_lane["director"] > 0.0
        else _empty_dict()
    )
    franchise_task = (
        fetch_similarity_franchise_candidates(
            lineage_entry_ids=franchise_traits,
            shared_universe_entry_ids=set(),
            subgroup_entry_ids=set(),
        )
        if cohesion_by_lane["franchise"] > 0.0
        else _empty_set()
    )
    studio_task = (
        fetch_movie_ids_by_production_company_ids(studio_company_ids)
        if cohesion_by_lane["studio"] > 0.0
        else _empty_set()
    )
    source_task = (
        fetch_similarity_source_candidates(source_ids)
        if cohesion_by_lane["source"] > 0.0
        else _empty_set()
    )
    quality_task = (
        fetch_similarity_quality_candidates(
            bucket=repeated_quality_bucket, limit=quality_limit
        )
        if repeated_quality_bucket is not None and cohesion_by_lane["quality"] > 0.0
        else _empty_set()
    )
    anchor_cast_task = fetch_similarity_top_billed_cast(anchor_ids)
    anchor_award_task = fetch_similarity_award_category_tags(anchor_ids)

    (
        (shape_scores, vector_space_weights, vector_space_cohesion, mean_pairwise_cosine),
        director_candidate_terms,
        franchise_candidate_ids,
        studio_candidate_ids,
        source_candidate_ids,
        quality_candidate_ids,
        anchor_cast_by_movie,
        anchor_award_by_movie,
    ) = await asyncio.gather(
        shape_task,
        director_task,
        franchise_task,
        studio_task,
        source_task,
        quality_task,
        anchor_cast_task,
        anchor_award_task,
    )

    # Cast / specific_award cohesion needs the just-fetched anchor data.
    cast_trait_sets = [
        anchor_cast_by_movie.get(anchor_id, set()) for anchor_id in anchor_ids
    ]
    specific_award_trait_sets = [
        anchor_award_by_movie.get(anchor_id, set()) for anchor_id in anchor_ids
    ]
    cohesion_by_lane["cast"] = _metadata_cohesion(cast_trait_sets)
    cohesion_by_lane["specific_award"] = _multi_anchor_specific_award_cohesion(
        specific_award_trait_sets, anchor_count=n
    )

    # V2 low-cohesion fallback: when both vector cohesion and every
    # metadata lane's cohesion are weak, the centroid lands in noise. Bail
    # out to round-robin per-anchor single-anchor results.
    metadata_max_cohesion = max(cohesion_by_lane.values(), default=0.0)
    if (
        mean_pairwise_cosine < LOW_COHESION_VECTOR_THRESHOLD
        and metadata_max_cohesion < LOW_COHESION_METADATA_MAX_THRESHOLD
    ):
        return await _low_cohesion_fallback(
            anchor_ids,
            anchor_rows,
            vectors_by_anchor,
            studio_entries_by_company_id,
            director_terms_by_anchor,
            limit=limit,
            qdrant_limit=qdrant_limit,
            quality_limit=quality_limit,
            mean_pairwise_cosine=mean_pairwise_cosine,
        )

    # V2 raw lane weights. Shape scales with mean cohesion (range
    # [0.36, 1.20]); metadata lanes scale by their cohesion factor.
    shape_raw = _shape_raw_for_multi_anchor(mean_pairwise_cosine)
    raw_lane_weights: dict[LaneName, float] = {
        "shape": shape_raw,
        "director": 0.12 * cohesion_by_lane["director"],
        "franchise": 0.12 * cohesion_by_lane["franchise"],
        "studio": 0.06 * cohesion_by_lane["studio"],   # debug-only weight
        "source": 0.04 * cohesion_by_lane["source"],
        "quality": 0.06 * cohesion_by_lane["quality"],
        "format": 0.04 * cohesion_by_lane["format"],
        "themes": 0.06 * cohesion_by_lane["themes"],
        "cast": 0.03 * cohesion_by_lane["cast"],
        "specific_award": 0.04 * cohesion_by_lane["specific_award"],
    }
    lane_weights = _normalize_weights(raw_lane_weights)

    candidate_ids = set(shape_scores)
    candidate_ids.update(director_candidate_terms)
    candidate_ids.update(franchise_candidate_ids)
    candidate_ids.update(studio_candidate_ids)
    candidate_ids.update(source_candidate_ids)
    candidate_ids.update(quality_candidate_ids)
    candidate_ids -= set(anchor_ids)

    # Candidate-side reads.
    candidate_rows_task = fetch_similarity_signal_rows(list(candidate_ids))
    candidate_cast_task = (
        fetch_similarity_top_billed_cast(list(candidate_ids))
        if cohesion_by_lane["cast"] > 0.0
        else _empty_movie_set_dict()
    )
    candidate_award_tags_task = (
        fetch_similarity_award_category_tags(list(candidate_ids))
        if cohesion_by_lane["specific_award"] > 0.0
        else _empty_movie_set_dict()
    )
    award_signals_task = (
        fetch_similarity_award_signals(list(candidate_ids))
        if repeated_quality_bucket is not None and cohesion_by_lane["quality"] > 0.0
        else _empty_award_signals()
    )
    # Source IDFs across the union of source types touched by anchors.
    source_idf_pairs_task = (
        fetch_trait_idfs([(TRAIT_KIND_SOURCE_MATERIAL, t) for t in source_ids])
        if source_ids and cohesion_by_lane["source"] > 0.0
        else _empty_idf_dict()
    )
    # Themes IDFs only need entries for the repeated traits (the lane
    # denominator + numerator only iterate over those). Repeated traits
    # are already (kind, trait_id) tuples — feed straight into the
    # batch IDF fetch.
    themes_repeated = _multi_anchor_themes_repeated(themes_trait_sets)
    themes_idf_pairs_task = (
        fetch_trait_idfs(list(themes_repeated))
        if themes_repeated and cohesion_by_lane["themes"] > 0.0
        else _empty_idf_dict()
    )

    (
        candidate_rows,
        candidate_cast_by_movie,
        candidate_award_tags_by_movie,
        award_signals,
        source_idf_pairs,
        themes_idf_pairs,
    ) = await asyncio.gather(
        candidate_rows_task,
        candidate_cast_task,
        candidate_award_tags_task,
        award_signals_task,
        source_idf_pairs_task,
        themes_idf_pairs_task,
    )

    source_idfs: dict[int, float] = {
        trait_id: idf for (_, trait_id), idf in source_idf_pairs.items()
    }
    # Themes IDFs stay keyed by (kind, trait_id) so colliding numeric IDs
    # across the keyword/concept/genre families don't cross-contaminate.
    themes_idfs: dict[tuple[int, int], float] = themes_idf_pairs

    # ----- Per-lane scoring -----
    director_scores = _score_multi_trait_count(
        candidate_terms=director_candidate_terms,
        anchor_trait_sets=director_trait_sets,
        anchor_count=n,
    )

    franchise_candidate_traits = {
        mid: set().union(*_franchise_traits(row))
        for mid, row in candidate_rows.items()
        if mid in franchise_candidate_ids
    }
    franchise_scores = _score_multi_trait_count(
        candidate_terms=franchise_candidate_traits,
        anchor_trait_sets=franchise_trait_sets,
        anchor_count=n,
    )

    studio_candidate_traits = {
        mid: {
            entry.company_id
            for entry in _active_studio_entries(row, studio_entries_by_company_id)
        }
        for mid, row in candidate_rows.items()
        if mid in studio_candidate_ids
    }
    # Studio is multiplicative in V2 — but multi-anchor still uses
    # repetition-count scoring as input to that multiplier so a candidate
    # matching every anchor's studio gets the full +10% boost while a
    # 1-of-3 match gets ~3.3%.
    studio_scores = _score_multi_trait_count(
        candidate_terms=studio_candidate_traits,
        anchor_trait_sets=studio_trait_sets,
        anchor_count=n,
    )

    # Source: V2 IDF-weighted per-anchor match. Common types ("novel")
    # contribute ~0.20 even at full anchor coverage; rare types ride at
    # nearly 1.0.
    source_candidate_traits = {
        mid: _as_int_set(row.get("source_material_type_ids"))
        for mid, row in candidate_rows.items()
        if mid in source_candidate_ids
    }
    source_scores = _score_multi_trait_count(
        candidate_terms=source_candidate_traits,
        anchor_trait_sets=source_trait_sets,
        anchor_count=n,
        weight_fn=lambda shared: max(
            (source_idfs.get(t, 0.0) for t in shared), default=0.0
        ),
    )

    # Quality: repetition-driven scoring — a candidate in the same bucket
    # as the consensus gets full match strength; cross-bucket candidates
    # use the V2 per-bucket formula scaled by the consensus repetition.
    quality_scores: dict[int, float] = {}
    if repeated_quality_bucket is not None and cohesion_by_lane["quality"] > 0.0:
        repeated_anchor_count = sum(
            1 for traits in quality_trait_sets if repeated_quality_bucket in traits
        )
        repetition_ratio = repeated_anchor_count / n
        for mid, row in candidate_rows.items():
            if mid not in quality_candidate_ids:
                continue
            if _quality_bucket(row) == repeated_quality_bucket:
                quality_scores[mid] = repetition_ratio
            else:
                quality_scores[mid] = (
                    _quality_score_v2(
                        repeated_quality_bucket, row, award_signals.get(mid)
                    )
                    * repetition_ratio
                )

    # Format: same-bucket-or-not vs the repeated bucket, every candidate.
    format_scores: dict[int, float] = {}
    if repeated_format_bucket is not None and cohesion_by_lane["format"] > 0.0:
        for mid, row in candidate_rows.items():
            candidate_bucket = format_bucket(row.get("keyword_ids") or ())
            if candidate_bucket == repeated_format_bucket:
                format_scores[mid] = 1.0

    # Themes: per-candidate share of repeated-trait IDF mass.
    themes_scores = _multi_anchor_themes_scores(
        candidate_rows=candidate_rows,
        repeated_traits=themes_repeated,
        idf_lookup=themes_idfs,
    )

    # Cast: repetition-count over top-3 billed actors.
    cast_scores = _score_multi_trait_count(
        candidate_terms=candidate_cast_by_movie,
        anchor_trait_sets=cast_trait_sets,
        anchor_count=n,
    )

    # Specific award: tier-weighted score over repeated tags.
    specific_award_scores = _multi_anchor_specific_award_scores(
        candidate_award_tags_by_movie=candidate_award_tags_by_movie,
        anchor_trait_sets=specific_award_trait_sets,
        anchor_count=n,
    )

    # Country/language consensus multiplier per candidate.
    country_consensus_match: dict[int, bool] = {}
    if consensus_countries:
        for mid, row in candidate_rows.items():
            candidate_countries = country_set(row.get("keyword_ids") or ())
            country_consensus_match[mid] = bool(
                candidate_countries & consensus_countries
            )

    # Per the V2 spec, multi-anchor medium handling is left to a future
    # iteration — multi-anchor candidates are not scored or filtered by
    # medium in V2.0. Pass an empty multiplier dict so build_results
    # leaves combined_score untouched.

    lane_scores: dict[LaneName, dict[int, float]] = {
        "shape": shape_scores,
        "director": {mid: s for mid, s in director_scores.items() if s > 0.0},
        "franchise": {mid: s for mid, s in franchise_scores.items() if s > 0.0},
        "studio": {},   # debug-only; multiplier handles the contribution
        "source": {mid: s for mid, s in source_scores.items() if s > 0.0},
        "quality": {mid: s for mid, s in quality_scores.items() if s > 0.0},
        "format": format_scores,
        "themes": {mid: s for mid, s in themes_scores.items() if s > 0.0},
        "cast": {mid: s for mid, s in cast_scores.items() if s > 0.0},
        "specific_award": {
            mid: s for mid, s in specific_award_scores.items() if s > 0.0
        },
    }

    ranked, counts = _build_results(
        anchor_ids=anchor_ids,
        lane_scores=lane_scores,
        lane_weights=lane_weights,
        limit=limit,
        studio_score_by_movie=studio_scores,
        country_consensus_match_by_movie=country_consensus_match,
        anchor_format_bucket=repeated_format_bucket,
        enforce_format_top_lock=repeated_format_bucket is not None,
    )

    active_anchor_types: list[AnchorType] = ["standard_shape"]
    if cohesion_by_lane["franchise"] > 0.0:
        active_anchor_types.append("franchise_dominant")
    if cohesion_by_lane["studio"] > 0.0:
        active_anchor_types.append("studio_lineage")
    if cohesion_by_lane["source"] > 0.0:
        active_anchor_types.append("source_material")
    if repeated_quality_bucket in {"cult_garbage", "prestige"}:
        active_anchor_types.append(repeated_quality_bucket)  # type: ignore[arg-type]

    # Per-anchor active-anchor-types — useful for debugging centroid drift
    # cases (Best Picture trio where Schindler/12YS dominated and pushed
    # The Godfather adjacents off the page).
    per_anchor_active = _per_anchor_active_anchor_types(
        anchor_ids,
        anchor_rows,
        director_terms_by_anchor,
        anchor_studio_entries,
    )

    return SimilarMoviesSearchResult(
        anchor_movie_ids=anchor_ids,
        ranked=ranked,
        active_anchor_types=active_anchor_types,
        debug=SimilarMoviesDebug(
            vector_space_weights=vector_space_weights,
            vector_space_cohesion=vector_space_cohesion,
            raw_lane_weights=raw_lane_weights,
            normalized_lane_weights=lane_weights,
            candidate_counts_by_lane=counts,
            anchor_format_bucket=repeated_format_bucket,
            consensus_countries=sorted(consensus_countries, key=str),
            per_anchor_active_anchor_types=per_anchor_active,
        ),
    )


def _score_multi_trait_count(
    *,
    candidate_terms: dict[int, set[int] | set[str]],
    anchor_trait_sets: list[set[int] | set[str]],
    anchor_count: int,
    weight_fn=None,
) -> dict[int, float]:
    """Per-anchor-match scoring shared across the multi-anchor lanes.

    Default behavior: each anchor whose trait set intersects the candidate
    contributes ``1.0`` to the numerator; final score is averaged over
    ``anchor_count`` so a candidate matching every anchor scores 1.0. If
    ``weight_fn`` is supplied, each matching anchor contributes
    ``weight_fn(shared_traits)`` instead — used by the V2 multi-anchor
    source lane to weight per-anchor matches by the IDF of the shared
    type, so common-tag overlaps don't ride at full strength.
    """
    scores: dict[int, float] = {}
    for movie_id, traits in candidate_terms.items():
        if not traits:
            continue
        total = 0.0
        for anchor_traits in anchor_trait_sets:
            shared = traits & anchor_traits
            if not shared:
                continue
            total += weight_fn(shared) if weight_fn is not None else 1.0
        if total > 0.0:
            scores[movie_id] = _clamp(total / anchor_count)
    return scores


def _themes_traits_for_movie(row: dict) -> set[tuple[int, int]]:
    """Build the themes-lane trait pool for a movie row.

    Combines keyword_ids, concept_tag_ids, and genre_ids minus the tags
    that are already covered by other V2 lanes (country/language tags
    feed the coherence multiplier; medium tags feed the medium
    multiplier; format tags feed the format lane). Returns ``(kind,
    trait_id)`` tuples — the three families share numeric IDs (e.g.
    ``Genre.HORROR.genre_id = 14`` collides with
    ``OverallKeyword.BASEBALL.keyword_id = 14``), so the kind has to be
    on the trait key or repeated-trait counts and IDF lookups land on
    the wrong family.
    """
    keyword_ids = _as_int_set(row.get("keyword_ids"))
    excluded = (
        COUNTRY_LANGUAGE_KEYWORD_IDS
        | MEDIUM_TAG_IDS
        | FORMAT_KEYWORD_IDS_ALL
    )
    keyword_ids -= excluded
    out: set[tuple[int, int]] = set()
    for k in keyword_ids:
        out.add((TRAIT_KIND_OVERALL_KEYWORD, k))
    for c in _as_int_set(row.get("concept_tag_ids")):
        out.add((TRAIT_KIND_CONCEPT_TAG, c))
    for g in _as_int_set(row.get("genre_ids")):
        out.add((TRAIT_KIND_TMDB_GENRE, g))
    return out


def _repeated_quality_bucket(
    quality_trait_sets: list[set[str]],
) -> str | None:
    counts: dict[str, int] = {}
    for traits in quality_trait_sets:
        for trait in traits:
            counts[trait] = counts.get(trait, 0) + 1
    repeated = [(count, bucket) for bucket, count in counts.items() if count >= 2]
    if not repeated:
        return None
    repeated.sort(reverse=True)
    return repeated[0][1]


def _repeated_format_bucket(
    format_trait_sets: list[set[str]],
) -> FormatBucket | None:
    """Return the format bucket repeated by ≥2 anchors, or None.

    When more than one bucket repeats (uncommon but possible — e.g. two
    docs and two narrative features in a 4-anchor set), returns the most
    repeated. Ties broken alphabetically for stable behavior.
    """
    counts: dict[str, int] = {}
    for traits in format_trait_sets:
        for trait in traits:
            counts[trait] = counts.get(trait, 0) + 1
    repeated = sorted(
        ((count, bucket) for bucket, count in counts.items() if count >= 2),
        reverse=True,
    )
    if not repeated:
        return None
    return repeated[0][1]  # type: ignore[return-value]


def _consensus_country_set(
    country_trait_sets: list[frozenset[int | str]],
) -> set[int | str]:
    """Return the country/language buckets shared by ≥2 anchors."""
    counts: dict[int | str, int] = {}
    for traits in country_trait_sets:
        for trait in traits:
            counts[trait] = counts.get(trait, 0) + 1
    return {bucket for bucket, count in counts.items() if count >= 2}


def _multi_anchor_specific_award_cohesion(
    award_trait_sets: list[set[int]],
    *,
    anchor_count: int,
) -> float:
    """V2 specific-award lane cohesion: log-curve scaled by tier specificity.

    The lane fires only when ≥2 anchors share a category_tag_id. The
    specificity factor discounts cohesion when the only repetition is at
    a coarser level — three different acting awards repeating only at L1
    LEAD_ACTING get partial cohesion; three disjoint groups repeating at
    L2 get a mild one. A perfect L0 BEST_PICTURE three-way match maxes
    out at 2.0.
    """
    if anchor_count < 2 or not award_trait_sets:
        return 0.0
    counts: dict[int, int] = {}
    for traits in award_trait_sets:
        for tag_id in traits:
            counts[tag_id] = counts.get(tag_id, 0) + 1
    repeated = {tag_id: c for tag_id, c in counts.items() if c >= 2}
    if not repeated:
        return 0.0
    most_specific_level = min(tag_level(t) for t in repeated)
    specificity = SPECIFICITY_FACTOR[most_specific_level]
    best_ratio = max(
        c / anchor_count
        for tag_id, c in repeated.items()
        if tag_level(tag_id) == most_specific_level
    )
    return specificity * 2.0 * math.log1p(9.0 * best_ratio) / math.log1p(9.0)


def _multi_anchor_themes_repeated(
    themes_trait_sets: list[set[tuple[int, int]]],
) -> set[tuple[int, int]]:
    """Identify ``(kind, trait_id)`` pairs repeated across ≥2 anchors.

    Trait keys are already kind-namespaced by ``_themes_traits_for_movie``,
    so counting collisions across the per-anchor sets gives the right
    answer without any kind disambiguation step.
    """
    counts: dict[tuple[int, int], int] = {}
    for traits in themes_trait_sets:
        for pair in traits:
            counts[pair] = counts.get(pair, 0) + 1
    return {pair for pair, c in counts.items() if c >= 2}


def _multi_anchor_themes_scores(
    *,
    candidate_rows: dict[int, dict],
    repeated_traits: set[tuple[int, int]],
    idf_lookup: dict[tuple[int, int], float],
) -> dict[int, float]:
    """V2 themes lane: shared-IDF / total-IDF mass over repeated traits.

    A candidate matching every repeated trait scores 1.0. Common tags
    with idf ≈ 0 contribute negligibly even when matched, so anchor sets
    bound by niche shared traits (FOLK_HORROR, KAIJU, CYBERPUNK) carry
    real signal while anchor sets that share only DRAMA/COMEDY collapse
    to ~0 even at perfect overlap.

    Trait keys are kind-namespaced (``(kind, id)``) so different families
    sharing a numeric ID don't cross-contaminate.
    """
    if not repeated_traits:
        return {}
    denom = sum(idf_lookup.get(pair, 0.0) for pair in repeated_traits)
    if denom <= 0.0:
        return {}
    out: dict[int, float] = {}
    for movie_id, row in candidate_rows.items():
        candidate_traits = _themes_traits_for_movie(row)
        shared = candidate_traits & repeated_traits
        if not shared:
            continue
        numer = sum(idf_lookup.get(pair, 0.0) for pair in shared)
        score = numer / denom
        if score > 0.0:
            out[movie_id] = _clamp(score)
    return out


def _multi_anchor_specific_award_scores(
    *,
    candidate_award_tags_by_movie: dict[int, set[int]],
    anchor_trait_sets: list[set[int]],
    anchor_count: int,
) -> dict[int, float]:
    """Tier-weighted candidate scoring for the specific-award lane.

    Repeated tags across anchors form the score basis. Each level
    contributes its TIER_WEIGHT (L0=1.00, L1=0.50, L2=0.20) so a
    candidate matching the same L0 tag as the consensus dominates one
    that only shares the L2 group bucket. Unmatched candidates score 0.
    """
    if anchor_count < 2 or not anchor_trait_sets:
        return {}
    counts: dict[int, int] = {}
    for traits in anchor_trait_sets:
        for tag_id in traits:
            counts[tag_id] = counts.get(tag_id, 0) + 1
    repeated = {tag_id for tag_id, c in counts.items() if c >= 2}
    if not repeated:
        return {}
    denom = sum(TIER_WEIGHT[tag_level(t)] for t in repeated)
    if denom <= 0.0:
        return {}
    out: dict[int, float] = {}
    for movie_id, candidate_tags in candidate_award_tags_by_movie.items():
        shared = candidate_tags & repeated
        if not shared:
            continue
        numer = sum(TIER_WEIGHT[tag_level(t)] for t in shared)
        score = numer / denom
        if score > 0.0:
            out[movie_id] = _clamp(score)
    return out


def _per_anchor_active_anchor_types(
    anchor_ids: list[int],
    anchor_rows: dict[int, dict],
    director_terms_by_anchor: dict[int, set[int]],
    anchor_studio_entries: dict[int, list[_ResolvedStudioEntry]],
) -> dict[int, list[AnchorType]]:
    """Build the per-anchor active-types breakdown for the debug payload.

    Each anchor's individual flags surface on the multi-anchor result so
    failures like "Schindler's List + 12 Years a Slave dominated the
    centroid" are diagnosable from the response without having to rerun
    each anchor as a single-anchor query.
    """
    out: dict[int, list[AnchorType]] = {}
    for anchor_id in anchor_ids:
        row = anchor_rows[anchor_id]
        types: list[AnchorType] = ["standard_shape"]
        bucket = _quality_bucket(row)
        if bucket in {"cult_garbage", "prestige"}:
            types.append(bucket)  # type: ignore[arg-type]
        a_lin, a_uni, a_sub = _franchise_traits(row)
        if a_lin or a_uni or a_sub:
            types.append("franchise_dominant")
        if anchor_studio_entries.get(anchor_id):
            types.append("studio_lineage")
        if _as_int_set(row.get("source_material_type_ids")):
            types.append("source_material")
        out[anchor_id] = types
    return out


async def _low_cohesion_fallback(
    anchor_ids: list[int],
    anchor_rows: dict[int, dict],
    vectors_by_anchor: dict[int, dict[VectorName, list[float]]],
    studio_entries_by_company_id: dict[int, list[StudioSimilarityEntry]],
    director_terms_by_anchor: dict[int, set[int]],
    *,
    limit: int,
    qdrant_limit: int,
    quality_limit: int,
    mean_pairwise_cosine: float,
) -> SimilarMoviesSearchResult:
    """Fallback for chaotic anchor sets where the centroid lands in noise.

    Runs each anchor as an independent single-anchor query, then weaves
    the per-anchor lists round-robin by rank. Returns the result with a
    debug flag set so callers know the fallback fired; UI presents
    results identically to a normal multi-anchor return.
    """
    n = len(anchor_ids)
    per_anchor_limit = max(1, math.ceil(limit * 1.2 / n))
    per_anchor_results = await asyncio.gather(
        *(
            _run_single_anchor_similarity(
                anchor_id,
                anchor_rows,
                vectors_by_anchor,
                studio_entries_by_company_id,
                director_terms_by_anchor,
                limit=per_anchor_limit,
                qdrant_limit=qdrant_limit,
                quality_limit=quality_limit,
            )
            for anchor_id in anchor_ids
        )
    )

    # Round-robin interleave: take the i-th result from each anchor in
    # turn, dedupe by movie_id, stop once we hit the requested limit.
    seen: set[int] = set(anchor_ids)
    interleaved: list[SimilarMovieResult] = []
    for rank in range(per_anchor_limit):
        for result in per_anchor_results:
            if rank >= len(result.ranked):
                continue
            entry = result.ranked[rank]
            if entry.movie_id in seen:
                continue
            seen.add(entry.movie_id)
            interleaved.append(entry)
            if len(interleaved) >= limit:
                break
        if len(interleaved) >= limit:
            break

    return SimilarMoviesSearchResult(
        anchor_movie_ids=anchor_ids,
        ranked=interleaved,
        active_anchor_types=["standard_shape"],
        debug=SimilarMoviesDebug(
            vector_space_weights={},
            vector_space_cohesion={"mean_pairwise_cosine": mean_pairwise_cosine},
            low_cohesion_fallback_used=True,
        ),
    )


async def _empty_set() -> set[int]:
    return set()


async def _empty_dict() -> dict[int, set[int]]:
    return {}


async def _empty_movie_set_dict() -> dict[int, set[int]]:
    return {}


async def _empty_award_signals() -> dict[int, SimilarityAwardSignals]:
    return {}


async def _empty_idf_dict() -> dict[tuple[int, int], float]:
    return {}
