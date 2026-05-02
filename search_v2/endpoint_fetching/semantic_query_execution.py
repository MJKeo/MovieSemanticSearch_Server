# Search V2 — Stage 3 Semantic Endpoint: Query Execution
#
# Single-trait per call. The caller passes the parent trait role
# explicitly, and `params` must be one of that role's matching semantic
# parameter shapes (base or subintent).
#
# Carver scoring (per spec):
#   1) Per active space: detect elbow via EWMA + Kneedle + pathology
#      fallback. Floor is uniformly elbow * 0.9 (10%-below-elbow decay
#      window).
#   2) Per active space, per movie: linear decay sim ↦ raw ∈ [0, 1].
#      No per-space compression — that happens once at the end.
#   3) Sum raw across active spaces / N → avg ∈ [0, 1].
#      avg == 0 → DROP (movie failed every space's threshold).
#      avg  > 0 → final = 0.5 + 0.5 * avg ∈ (0.5, 1].
#
# Qualifier scoring: weighted-sum cosine — Σ(w·cos)/Σw with
# CENTRAL=2.0, SUPPORTING=1.0. No elbow calibration (the score is the
# qualifier endpoint's contribution to the global merge).
#
# Role and restriction mode are a boundary contract:
#   carver    → restrict_to_movie_ids must be None; candidate-generating
#               path uses the union of per-space top-N corpus probes.
#   qualifier → restrict_to_movie_ids must be supplied; scores exactly
#               that candidate pool. set() short-circuits without
#               Qdrant traffic because there are no candidates to score.
#
# Polarity is NOT consulted here (orchestrator concern). Retry contract
# matches sibling executors: one transient retry, then return an empty
# EndpointResult rather than raising.

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Iterable, Union

import numpy as np
from kneed import KneeLocator
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, HasIdCondition

from db.vector_search import COLLECTION_ALIAS, QDRANT_SEARCH_PARAMS
from implementation.classes.enums import VectorName
from implementation.llms.generic_methods import generate_vector_embedding
from schemas.endpoint_result import EndpointResult
from schemas.enums import Role
from schemas.semantic_bodies import (
    NarrativeTechniquesBody,
    PlotAnalysisBody,
    PlotEventsBody,
    ProductionBody,
    ReceptionBody,
    ViewerExperienceBody,
    WatchContextBody,
)
from schemas.semantic_translation import (
    CarverSemanticParameters,
    CarverSemanticParametersSubintent,
    QualifierSemanticParameters,
    QualifierSemanticParametersSubintent,
    SemanticSpaceEntry,
    SpaceWeight,
    WeightedSpaceQuery,
)
from search_v2.endpoint_fetching.result_helpers import (
    build_endpoint_result,
    compress_to_dealbreaker_floor,
)

logger = logging.getLogger(__name__)

CarverSemanticParams = CarverSemanticParameters | CarverSemanticParametersSubintent
QualifierSemanticParams = (
    QualifierSemanticParameters | QualifierSemanticParametersSubintent
)
SemanticParams = CarverSemanticParams | QualifierSemanticParams


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Top-N corpus probe. Doubles as elbow calibration sample and (for
# candidate-generating scenarios) the candidate pool.
CORPUS_PROBE_LIMIT = 2000

# Pathology detector for elbow calibration. Fires when the top-N
# probe has no discriminable structure (range below threshold).
PATHOLOGY_RANGE_THRESHOLD = 0.05
PATHOLOGY_ELBOW_RATIO = 0.85

# Below this probe length, skip Kneedle and use the pathology
# fallback — too few samples for a meaningful curve.
MIN_PROBE_SIZE = 20

# Linear-decay window: floor = elbow * FLOOR_FRACTION_OF_ELBOW.
# A movie 9.99% below the elbow gets close-to-0 raw decay; 10% below
# is the cliff. Per the carver scoring spec.
FLOOR_FRACTION_OF_ELBOW = 0.9

# EWMA smoothing for the cosine curve before Kneedle.
EWMA_SPAN_DIVISOR = 100
EWMA_SPAN_FLOOR = 5

# If the first detected knee sits earlier than this rank AND a later
# knee exists, skip forward — guards against outlier-driven early
# knees pinching the elbow too tight.
RANK_10_SAFEGUARD = 10

# Must match the ingestion-side embedding model (see
# movie_ingestion/final_ingestion/ingest_movie.py).
EMBEDDING_MODEL = "text-embedding-3-large"

# Two-level categorical weights for qualifier weighted-sum.
SPACE_WEIGHT_VALUES: dict[SpaceWeight, float] = {
    SpaceWeight.CENTRAL: 2.0,
    SpaceWeight.SUPPORTING: 1.0,
}


# Union of every query-side body type. Each variant exposes
# embedding_text() -> str. Anchor is intentionally excluded — the
# 7-space SemanticSpace enum upstream forbids AnchorBody entries.
SemanticBody = Union[
    PlotEventsBody,
    PlotAnalysisBody,
    ViewerExperienceBody,
    WatchContextBody,
    NarrativeTechniquesBody,
    ProductionBody,
    ReceptionBody,
]


# ---------------------------------------------------------------------------
# Calibration result shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpaceCalibration:
    # Per-space scoring calibration produced from a corpus probe.
    #
    # Normal mode: elbow/floor define the linear-decay window per the
    # carver scoring spec (floor = elbow * 0.9). pass_through_raw
    # signals the Path B fallback — when the probe yields max_sim <= 0
    # (no usable signal in this space), the per-space scorer uses raw
    # cosine clamped to [0, 1] instead of running elbow decay against
    # a manufactured threshold.
    elbow: float
    floor: float
    pass_through_raw: bool = False


# Sentinel for the Path B fallback. elbow/floor are unused when
# pass_through_raw is True; carry zero for shape consistency.
_PASS_THROUGH = SpaceCalibration(elbow=0.0, floor=0.0, pass_through_raw=True)


# ---------------------------------------------------------------------------
# Qdrant primitives
# ---------------------------------------------------------------------------


async def _embed_body(body: SemanticBody) -> list[float]:
    embeddings = await generate_vector_embedding(
        [body.embedding_text()], model=EMBEDDING_MODEL
    )
    return embeddings[0]


async def _run_corpus_topn(
    embedding: list[float],
    vector_name: VectorName,
    *,
    qdrant_client: AsyncQdrantClient,
    limit: int = CORPUS_PROBE_LIMIT,
) -> list[tuple[int, float]]:
    response = await qdrant_client.query_points(
        collection_name=COLLECTION_ALIAS,
        query=embedding,
        using=vector_name.value,
        limit=limit,
        with_payload=False,
        with_vectors=False,
        search_params=QDRANT_SEARCH_PARAMS,
    )
    return [(int(p.id), float(p.score)) for p in response.points]


async def _run_filtered_score(
    embedding: list[float],
    vector_name: VectorName,
    movie_ids: Iterable[int],
    *,
    qdrant_client: AsyncQdrantClient,
) -> dict[int, float]:
    # Score a specific set of movie_ids on a single named vector.
    # Movie_id IS the Qdrant point ID, so HasIdCondition is the right
    # filter (not a payload FieldCondition).
    id_list = [int(mid) for mid in movie_ids]
    if not id_list:
        return {}
    response = await qdrant_client.query_points(
        collection_name=COLLECTION_ALIAS,
        query=embedding,
        using=vector_name.value,
        query_filter=Filter(must=[HasIdCondition(has_id=id_list)]),
        limit=len(id_list),
        with_payload=False,
        with_vectors=False,
        search_params=QDRANT_SEARCH_PARAMS,
    )
    return {int(p.id): float(p.score) for p in response.points}


# ---------------------------------------------------------------------------
# Elbow detection + linear-decay scoring
# ---------------------------------------------------------------------------


def _ewma(values: list[float], span: int) -> np.ndarray:
    # Forward-pass EWMA, equivalent to pandas' ewm(span,adjust=False).
    # Probe size is ~2000 floats so the Python loop is microseconds.
    alpha = 2.0 / (span + 1.0)
    out = np.empty(len(values), dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _detect_elbow_and_floor(similarities: list[float]) -> SpaceCalibration:
    # Returns a SpaceCalibration. Input must be sorted descending
    # (Qdrant's natural order).
    #
    # Path resolution:
    #   - Empty probe / max_sim <= 0  → pass_through_raw (Path B). The
    #     pathology fallback can't manufacture a threshold from a max
    #     of 0; the per-space scorer uses raw cosine clamped to [0,1].
    #   - Short probe / flat dist / no knees / Kneedle-elbow at sim 0
    #     → pathology fallback (Path A): elbow = max_sim * 0.85,
    #     floor = elbow * 0.9.
    #   - Otherwise: Kneedle elbow at the picked rank's raw cosine,
    #     floor = elbow * 0.9.
    #
    # Floor is uniformly derived from the elbow (10%-below window)
    # per spec — the prior second-knee-floor logic is gone.

    if not similarities:
        return _PASS_THROUGH

    max_sim = float(similarities[0])

    # Path B: no usable signal in this space. Pathology can't scale
    # from max_sim <= 0; pass through raw cosine instead.
    if max_sim <= 0.0:
        logger.info(
            "Semantic calibration: max_sim=%.4f, using raw-cosine pass-through",
            max_sim,
        )
        return _PASS_THROUGH

    # Short-probe guard: Kneedle needs a curve.
    if len(similarities) < MIN_PROBE_SIZE:
        return _pathology_fallback(max_sim, reason="probe too short")

    # Flat-distribution pathology: top-to-bottom range too narrow to
    # carry discriminable structure.
    if max_sim - float(similarities[-1]) < PATHOLOGY_RANGE_THRESHOLD:
        return _pathology_fallback(max_sim, reason="flat distribution")

    span = max(EWMA_SPAN_FLOOR, len(similarities) // EWMA_SPAN_DIVISOR)
    smoothed = _ewma(similarities, span)

    locator = KneeLocator(
        x=list(range(len(smoothed))),
        y=smoothed.tolist(),
        curve="convex",
        direction="decreasing",
        S=1,
        online=True,
    )
    knees_raw = locator.all_knees if locator.all_knees else []
    knees = sorted({int(k) for k in knees_raw})
    if not knees:
        return _pathology_fallback(max_sim, reason="no knees detected")

    # Earliest knee unless it sits suspiciously early and a later
    # knee exists — outlier-driven early knees pinch the threshold
    # too tight.
    if knees[0] < RANK_10_SAFEGUARD and len(knees) >= 2:
        elbow_rank = knees[1]
    else:
        elbow_rank = knees[0]

    # Use raw cosines (not smoothed) for the threshold value —
    # smoothed values lag on a monotonic-decreasing sequence and
    # would inflate the threshold relative to what Qdrant returns
    # for downstream comparisons.
    elbow_sim = max(0.0, min(float(similarities[elbow_rank]), 1.0))

    # Path A: Kneedle picked a rank whose raw sim is 0. Route to
    # pathology fallback so the floor is anchored against max_sim
    # rather than collapsing to 0.
    if elbow_sim <= 0.0:
        return _pathology_fallback(max_sim, reason="kneedle elbow at sim=0")

    return SpaceCalibration(elbow=elbow_sim, floor=elbow_sim * FLOOR_FRACTION_OF_ELBOW)


def _pathology_fallback(max_sim: float, *, reason: str) -> SpaceCalibration:
    # Precondition: max_sim > 0 (Path B short-circuits in
    # _detect_elbow_and_floor before we ever reach here).
    logger.info(
        "Semantic calibration falling back to fixed-ratio elbow/floor "
        "(reason=%s, max_sim=%.4f)",
        reason, max_sim,
    )
    elbow = max_sim * PATHOLOGY_ELBOW_RATIO
    return SpaceCalibration(elbow=elbow, floor=elbow * FLOOR_FRACTION_OF_ELBOW)


def _per_space_raw_decay(sim: float, calib: SpaceCalibration) -> float:
    # Carver per-space raw score in [0, 1]. NO per-space [0.5, 1]
    # compression — combination across spaces happens first, then
    # compression once at the end.
    #
    # Path B (pass_through_raw): no usable threshold; clamp raw
    # cosine to [0, 1].
    # Normal mode:
    #   sim >= elbow → 1.0  (we don't reward "more above elbow" per spec)
    #   sim <= floor → 0.0  (excluded from the average)
    #   in window    → linear decay (sim - floor) / (elbow - floor)
    if calib.pass_through_raw:
        return max(0.0, min(1.0, sim))
    if sim >= calib.elbow:
        return 1.0
    if sim <= calib.floor:
        return 0.0
    # Post-calibration invariant: elbow > 0 and floor = elbow * 0.9,
    # so elbow > floor strictly. No defensive divide-by-zero branch.
    return (sim - calib.floor) / (calib.elbow - calib.floor)


# ---------------------------------------------------------------------------
# Score combiners
# ---------------------------------------------------------------------------


def _carver_combine(
    movie_ids: Iterable[int],
    per_space_raw: dict[VectorName, dict[int, float]],
) -> dict[int, float]:
    # Carver final score: sum per-space raw decays / N active spaces,
    # then compress (0, 1] → (0.5, 1] via the shared dealbreaker-band
    # helper. Movies whose sum is exactly 0 (failed every space's
    # threshold) are DROPPED, not floored.
    n = len(per_space_raw)
    if n == 0:
        return {}
    out: dict[int, float] = {}
    for mid in movie_ids:
        mid_int = int(mid)
        total = 0.0
        for cos_map in per_space_raw.values():
            total += cos_map.get(mid_int, 0.0)
        avg = total / n
        if avg <= 0.0:
            continue
        out[mid_int] = compress_to_dealbreaker_floor(avg)
    return out


def _weighted_cosine_score(
    movie_ids: Iterable[int],
    per_space_cosines: dict[VectorName, dict[int, float]],
    per_space_weights: dict[VectorName, float],
) -> dict[int, float]:
    # Qualifier final score: Σ(w · cos) / Σw across active spaces.
    # Missing per-space cosine = 0.0 (a candidate present in the
    # union pool but absent from one space's score map contributes
    # 0 for that space rather than being dropped).
    total_weight = sum(per_space_weights.values())
    if total_weight <= 0.0:
        return {int(mid): 0.0 for mid in movie_ids}
    out: dict[int, float] = {}
    for mid in movie_ids:
        mid_int = int(mid)
        numerator = 0.0
        for space, weight in per_space_weights.items():
            numerator += weight * per_space_cosines.get(space, {}).get(mid_int, 0.0)
        out[mid_int] = max(0.0, min(1.0, numerator / total_weight))
    return out


# ---------------------------------------------------------------------------
# Carver scenarios
# ---------------------------------------------------------------------------


async def _execute_carver_d2(
    params: CarverSemanticParameters,
    *,
    qdrant_client: AsyncQdrantClient,
) -> dict[int, float]:
    # No restrict — pool is the union of per-space top-N probes.
    # NO fill: a movie absent from one space's top-N is treated as 0
    # for that space. Clearing the elbow without being in top-N is
    # implausible (the elbow rank sits inside top-N by construction),
    # so the missing-fill query would mostly recover sub-floor
    # cosines that compute to 0 anyway.
    entries = list(params.space_queries)
    embeddings = await asyncio.gather(*[_embed_body(e.content) for e in entries])
    vector_names = [VectorName(e.space.value) for e in entries]

    probes = await asyncio.gather(*[
        _run_corpus_topn(emb, vn, qdrant_client=qdrant_client)
        for emb, vn in zip(embeddings, vector_names)
    ])

    per_space_raw: dict[VectorName, dict[int, float]] = {}
    candidate_pool: set[int] = set()
    for vn, probe in zip(vector_names, probes):
        calib = _detect_elbow_and_floor([cos for _, cos in probe])
        space_map: dict[int, float] = {}
        for mid, cos in probe:
            space_map[mid] = _per_space_raw_decay(cos, calib)
            candidate_pool.add(mid)
        per_space_raw[vn] = space_map

    return _carver_combine(candidate_pool, per_space_raw)


async def _execute_carver_d1(
    params: CarverSemanticParameters,
    candidate_ids: set[int],
    *,
    qdrant_client: AsyncQdrantClient,
) -> dict[int, float]:
    # Pre-built pool — corpus probe per space is calibration-only.
    # Per-space cosines for the pool come from HasId. Probe and
    # filtered fetch run in parallel per space to keep latency flat.
    entries = list(params.space_queries)
    embeddings = await asyncio.gather(*[_embed_body(e.content) for e in entries])
    vector_names = [VectorName(e.space.value) for e in entries]
    n = len(entries)

    # Single gather across 2N tasks: N probes + N filtered fetches.
    tasks = [
        _run_corpus_topn(emb, vn, qdrant_client=qdrant_client)
        for emb, vn in zip(embeddings, vector_names)
    ] + [
        _run_filtered_score(emb, vn, candidate_ids, qdrant_client=qdrant_client)
        for emb, vn in zip(embeddings, vector_names)
    ]
    results = await asyncio.gather(*tasks)
    probes, filtered_maps = results[:n], results[n:]

    per_space_raw: dict[VectorName, dict[int, float]] = {}
    for vn, probe, filtered in zip(vector_names, probes, filtered_maps):
        calib = _detect_elbow_and_floor([cos for _, cos in probe])
        per_space_raw[vn] = {
            mid: _per_space_raw_decay(cos, calib)
            for mid, cos in filtered.items()
        }

    return _carver_combine(candidate_ids, per_space_raw)


# ---------------------------------------------------------------------------
# Qualifier scenarios
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QualifierInputs:
    # Common unpack of qualifier space_queries shared by P1 and P2.
    # The merge-validator on QualifierSemanticParameters guarantees
    # unique spaces, so building a {VectorName: weight} dict is safe.
    entries: list[SemanticSpaceEntry]
    vector_names: list[VectorName]
    weights: dict[VectorName, float]


def _qualifier_inputs(
    space_queries: list[WeightedSpaceQuery],
) -> QualifierInputs:
    entries = [wq.query for wq in space_queries]
    vector_names = [VectorName(e.space.value) for e in entries]
    weights = {
        vn: SPACE_WEIGHT_VALUES[wq.weight]
        for wq, vn in zip(space_queries, vector_names)
    }
    return QualifierInputs(
        entries=entries, vector_names=vector_names, weights=weights
    )


async def _execute_qualifier_p1(
    params: QualifierSemanticParameters,
    candidate_ids: set[int],
    *,
    qdrant_client: AsyncQdrantClient,
) -> dict[int, float]:
    # Pre-built pool — one HasId per space, no corpus probe needed
    # (qualifier scoring doesn't use the elbow).
    inputs = _qualifier_inputs(list(params.space_queries))
    embeddings = await asyncio.gather(
        *[_embed_body(e.content) for e in inputs.entries]
    )
    per_space_lookups = await asyncio.gather(*[
        _run_filtered_score(emb, vn, candidate_ids, qdrant_client=qdrant_client)
        for emb, vn in zip(embeddings, inputs.vector_names)
    ])
    per_space_cosines = dict(zip(inputs.vector_names, per_space_lookups))
    return _weighted_cosine_score(
        candidate_ids, per_space_cosines, inputs.weights
    )


async def _execute_qualifier_p2(
    params: QualifierSemanticParameters,
    *,
    qdrant_client: AsyncQdrantClient,
) -> dict[int, float]:
    # Candidate-generating — top-N per space, union = pool. Missing
    # cosines (in pool via another space's probe but absent from
    # this space's) get filled via HasId so the weighted sum is
    # honest across the union.
    inputs = _qualifier_inputs(list(params.space_queries))
    embeddings = await asyncio.gather(
        *[_embed_body(e.content) for e in inputs.entries]
    )
    probes = await asyncio.gather(*[
        _run_corpus_topn(emb, vn, qdrant_client=qdrant_client)
        for emb, vn in zip(embeddings, inputs.vector_names)
    ])

    per_space_cosines: dict[VectorName, dict[int, float]] = {
        vn: dict(probe) for vn, probe in zip(inputs.vector_names, probes)
    }
    candidate_ids: set[int] = set().union(
        *(cos_map.keys() for cos_map in per_space_cosines.values())
    )
    if not candidate_ids:
        return {}

    # Fill only spaces that have at least one missing ID.
    fill_tasks: list = []
    fill_targets: list[VectorName] = []
    for emb, vn in zip(embeddings, inputs.vector_names):
        missing = candidate_ids - per_space_cosines[vn].keys()
        if missing:
            fill_tasks.append(
                _run_filtered_score(emb, vn, missing, qdrant_client=qdrant_client)
            )
            fill_targets.append(vn)
    if fill_tasks:
        fills = await asyncio.gather(*fill_tasks)
        for vn, fill in zip(fill_targets, fills):
            per_space_cosines[vn].update(fill)

    return _weighted_cosine_score(
        candidate_ids, per_space_cosines, inputs.weights
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def execute_semantic_query(
    params: SemanticParams,
    *,
    role: Role,
    restrict_to_movie_ids: set[int] | None = None,
    qdrant_client: AsyncQdrantClient,
) -> EndpointResult:
    """Execute a semantic payload and return scored candidates.

    `role` is the discriminator. It must agree with the params family:
    carver role accepts CarverSemanticParameters or its subintent
    variant, and qualifier role accepts QualifierSemanticParameters or
    its subintent variant.

    Role also determines restriction mode:
      carver    → restrict_to_movie_ids must be None.
      qualifier → restrict_to_movie_ids must be supplied; set()
                  short-circuits to an empty result.

    Polarity is NOT consulted here — both positive and negative
    findings return the same EndpointResult shape; the orchestrator
    routes IDs/scores into inclusion/exclusion or
    preference/downrank buckets per role.

    Retry contract: one transient retry, then EndpointResult() rather
    than raising. Contract violations raise AssertionError before the
    retry loop — they are programmer errors, not transient I/O.
    """
    if role is Role.CARVER:
        if not isinstance(
            params, (CarverSemanticParameters, CarverSemanticParametersSubintent)
        ):
            raise AssertionError(
                f"Semantic carver execution received unexpected params type: "
                f"{type(params).__name__}"
            )
        if restrict_to_movie_ids is not None:
            raise AssertionError(
                "Semantic carver execution must not receive "
                "restrict_to_movie_ids."
            )
        is_carver = True
        log_context = (
            f"role=carver, spaces="
            f"{[e.space.value for e in params.space_queries]}"
        )

    elif role is Role.QUALIFIER:
        if not isinstance(
            params,
            (QualifierSemanticParameters, QualifierSemanticParametersSubintent),
        ):
            raise AssertionError(
                f"Semantic qualifier execution received unexpected params type: "
                f"{type(params).__name__}"
            )
        if restrict_to_movie_ids is None:
            raise AssertionError(
                "Semantic qualifier execution requires restrict_to_movie_ids."
            )
        is_carver = False
        log_context = (
            f"role=qualifier, spaces="
            f"{[wq.query.space.value for wq in params.space_queries]}"
        )

    else:
        raise AssertionError(
            f"Unsupported semantic role: {role!r}"
        )

    # Empty qualifier pool: nothing to score, no Qdrant traffic.
    if len(restrict_to_movie_ids or set()) == 0 and role is Role.QUALIFIER:
        return EndpointResult()

    scores: dict[int, float] = {}
    for attempt in range(2):
        try:
            if is_carver:
                scores = await _execute_carver_d2(
                    params, qdrant_client=qdrant_client
                )
            else:
                scores = await _execute_qualifier_p1(
                    params,
                    restrict_to_movie_ids,
                    qdrant_client=qdrant_client,
                )
            break
        except Exception:
            if attempt == 0:
                logger.warning(
                    "Semantic query error on first attempt, retrying (%s)",
                    log_context,
                    exc_info=True,
                )
                continue
            logger.error(
                "Semantic query error on retry, returning empty (%s)",
                log_context,
                exc_info=True,
            )
            return EndpointResult()

    return build_endpoint_result(scores, restrict_to_movie_ids)
