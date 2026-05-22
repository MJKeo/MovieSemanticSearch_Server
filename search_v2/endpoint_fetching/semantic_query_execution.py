# Search V2 — Stage 3 Semantic Endpoint: Query Execution
#
# Single-trait per call. `params` is a unified SemanticParameters
# (or subintent variant) whose `role` field commits the trait's
# semantic shape: carver (population-naming, "does this movie have
# X?") or qualifier (positioning-against-reference, "how X are
# these movies relative to each other?").
#
# Role drives BOTH axes of scoring:
#
#   Within-space normalization
#     carver                → corpus-calibrated elbow decay to [0, 1].
#                             "Does this movie clear the global bar?"
#     qualifier (restrict)  → pool-relative rescale to [0, 1] (top ×
#                             0.85 → 1.0, linear to pool min → 0.0).
#                             The reference is the supplied pool itself.
#     qualifier (promoted)  → corpus-calibrated elbow decay (same as
#                             carver) because there is no upstream pool
#                             to be relative against. Promoted-qualifier
#                             only happens via tier-fallback when no
#                             candidate-generating trait exists in the
#                             query (orchestrator concern).
#
#   Cross-space combination
#     carver                → max() across active spaces. Each call
#                             asks one question that may surface in any
#                             of several spaces; one strong signal is
#                             sufficient evidence. ANDs across distinct
#                             questions are split into separate traits
#                             upstream, so the within-call combiner is
#                             OR-shaped, not AND-shaped.
#     qualifier (any mode)  → Σ(w·score)/Σw with CENTRAL=2.0 and
#                             SUPPORTING=1.0. Preserves the LLM's
#                             commitment about which spaces are
#                             load-bearing.
#
# Restriction × role dispatch:
#   carver + restrict      → probe(corpus, calibration) ‖ HasId(scoring)
#                            in parallel per space; elbow-decay HasId
#                            cosines against the probe-derived
#                            calibration; max-combine across spaces.
#   carver + no restrict   → probe per space serves as both calibration
#                            sample AND candidate pool; elbow-decay
#                            probe cosines; max-combine across spaces.
#   qualifier + restrict   → HasId per space; pool-relative rescale per
#                            space; weighted-sum combine.
#   qualifier promoted     → probe per space serves as calibration AND
#                            pool; elbow-decay; weighted-sum combine.
#   restrict=set() (any role)  → normalized to None and dispatched
#                                 as candidate generator. The
#                                 orchestrator is the source of truth
#                                 for generator-vs-reranker routing;
#                                 empty-set arrival here is treated
#                                 as a leak, not a meaningful signal.
#
# All scores live truthfully in [0, 1]. No [0.5, 1] dealbreaker
# compression on any path. No drop-on-zero — a candidate missing every
# space lands at 0 naturally and contributes nothing downstream via
# the "missing positive = opportunity cost" rule.
#
# Polarity is NOT consulted here (orchestrator concern). Retry
# contract: one transient retry, then return an empty EndpointResult
# rather than raising.

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Iterable, Union

import numpy as np
from kneed import KneeLocator
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, HasIdCondition

from db.vector_search import (
    COLLECTION_ALIAS,
    QDRANT_SEARCH_PARAMS,
    build_qdrant_filter,
)
from implementation.classes.enums import VectorName
from implementation.classes.schemas import MetadataFilters
from implementation.llms.generic_methods import generate_vector_embedding
from schemas.endpoint_result import EndpointResult
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
    SemanticParameters,
    SemanticParametersSubintent,
    SemanticRetrievalShape,
    SemanticSpaceEntry,
    SpaceWeight,
    WeightedSpaceQuery,
    WeightedSpaceQuerySubintent,
)
from search_v2.endpoint_fetching.result_helpers import build_endpoint_result

logger = logging.getLogger(__name__)

SemanticParams = SemanticParameters | SemanticParametersSubintent


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Top-N corpus probe. Doubles as elbow calibration sample and (for
# carver-no-restrict / qualifier-promoted) the candidate pool.
CORPUS_PROBE_LIMIT = 2000

# Pathology detector for elbow calibration. Fires when the top-N
# probe has no discriminable structure (range below threshold).
PATHOLOGY_RANGE_THRESHOLD = 0.05
PATHOLOGY_ELBOW_RATIO = 0.85

# Below this probe length, skip Kneedle and use the pathology
# fallback — too few samples for a meaningful curve.
MIN_PROBE_SIZE = 20

# Linear-decay window: floor = elbow * FLOOR_FRACTION_OF_ELBOW.
FLOOR_FRACTION_OF_ELBOW = 0.9

# EWMA smoothing for the cosine curve before Kneedle.
EWMA_SPAN_DIVISOR = 100
EWMA_SPAN_FLOOR = 5

# Kneedle outlier guard: skip an early knee in favor of a later one
# when the early rank looks outlier-driven.
RANK_10_SAFEGUARD = 10

# Qualifier pool-relative rescale: cosines are normalized to [0, 1]
# across the pool (min → 0, max → 1); any normalized value ≥ this
# ratio is clamped to 1.0. Operating in normalized space (rather
# than as a fraction of raw top cosine) keeps the formula
# cosine-agnostic — the top-band semantics hold even when the pool
# straddles or sits entirely below zero.
QUALIFIER_TOP_RATIO = 0.85

# Below this raw-cosine spread, the qualifier rescale would amplify
# numerical noise into apparent ranking. Emit all 1.0 for that space
# instead — the signal didn't differentiate this pool, so let
# weighted-sum and trait_weight downstream handle it.
QUALIFIER_UNIFORM_SPREAD_EPSILON = 0.01

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
    # Per-space elbow calibration produced from a corpus probe.
    #
    # Normal mode: elbow/floor define the linear-decay window
    # (floor = elbow * FLOOR_FRACTION_OF_ELBOW). pass_through_raw
    # signals the Path B fallback — when the probe yields max_sim <= 0
    # (no usable signal), the per-space scorer uses raw cosine
    # clamped to [0, 1] instead of running elbow decay against a
    # manufactured threshold.
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


async def _embed_bodies(bodies: list[SemanticBody]) -> list[list[float]]:
    """Embed multiple semantic bodies in a single OpenAI call.

    Earlier code did `asyncio.gather(*[_embed_body(b) for b in bodies])`
    which fans out N parallel single-input HTTP requests. OpenAI's
    embeddings endpoint accepts a list input, so we collapse the fan-
    out into one call that round-trips once. Returns embeddings in
    the same order as the input bodies.
    """
    if not bodies:
        return []
    return await generate_vector_embedding(
        [body.embedding_text() for body in bodies],
        model=EMBEDDING_MODEL,
    )


def _hard_filter_must(
    metadata_filters: MetadataFilters | None,
) -> list:
    """Return the ``must`` conditions for the UI hard filter, or [].

    Helper that lifts the existing ``build_qdrant_filter()`` output into
    a list of FieldConditions we can splice into another Filter's
    ``must``. Returns [] when filters is None / inactive so callers can
    concatenate unconditionally.
    """
    if metadata_filters is None or not metadata_filters.is_active:
        return []
    qd = build_qdrant_filter(metadata_filters)
    if qd is None:
        return []
    return list(qd.must or [])


async def _run_corpus_topn(
    embedding: list[float],
    vector_name: VectorName,
    *,
    qdrant_client: AsyncQdrantClient,
    limit: int = CORPUS_PROBE_LIMIT,
    metadata_filters: MetadataFilters | None = None,
) -> list[tuple[int, float]]:
    # When the UI hard filter is active, the corpus probe must run over
    # the filtered slice — otherwise the elbow-calibration sample is
    # drawn from points that will be excluded downstream, and the
    # threshold lands on the wrong distribution. With very tight
    # filters fewer than CORPUS_PROBE_LIMIT points exist; the
    # downstream pathology detector (PATHOLOGY_RANGE_THRESHOLD)
    # handles that gracefully.
    hard_must = _hard_filter_must(metadata_filters)
    query_filter = Filter(must=hard_must) if hard_must else None
    response = await qdrant_client.query_points(
        collection_name=COLLECTION_ALIAS,
        query=embedding,
        using=vector_name.value,
        query_filter=query_filter,
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
    #
    # The user hard filter is intentionally NOT applied here: this
    # primitive is the reranker scoring path (HasId pool restriction
    # only), and the supplied movie_ids come from a candidate pool
    # the upstream generators already narrowed with the filter. The
    # corpus-probe primitive (`_run_corpus_topn`) still receives the
    # filter — that one is calibration and/or pool-source, both of
    # which need filter-aware sampling.
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
# Elbow detection (used by carver paths and qualifier-promoted)
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


def _elbow_decay(sim: float, calib: SpaceCalibration) -> float:
    # Per-space score in [0, 1] via linear decay against the corpus-
    # calibrated elbow window.
    #
    # Path B (pass_through_raw): no usable threshold; clamp raw
    # cosine to [0, 1].
    # Normal mode:
    #   sim >= elbow → 1.0  (we don't reward "more above elbow")
    #   sim <= floor → 0.0
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
# Pool-relative rescale (qualifier + restrict)
# ---------------------------------------------------------------------------


def _pool_relative_rescale(cosines: dict[int, float]) -> dict[int, float]:
    # Rescale candidate-pool cosines into [0, 1] purely relative to
    # the pool itself. The reference IS the pool — qualifier asks
    # "how X are these movies among themselves," not "do they clear
    # a global bar."
    #
    # Shape:
    #   1. Linear-normalize raw cosines into [0, 1] over the pool
    #      range: pool_min → 0.0, pool_max → 1.0.
    #   2. Anything with normalized value ≥ QUALIFIER_TOP_RATIO is
    #      clamped to 1.0; everything below carries its normalized
    #      value as its score.
    # Operating in normalized space makes the rule cosine-agnostic:
    # the top-band semantics hold even when the pool straddles or
    # sits entirely below zero.
    #
    # Uniform-spread guard: if max - min < ε the space carries no
    # discriminating signal for this pool, and rescale would amplify
    # numerical noise into apparent ranking. Emit all 1.0 instead;
    # the weighted-sum combiner across spaces and the trait_weight
    # machinery downstream handle the "this signal didn't
    # differentiate" case correctly.
    if not cosines:
        return {}
    values = list(cosines.values())
    top = max(values)
    bottom = min(values)
    spread = top - bottom

    if spread < QUALIFIER_UNIFORM_SPREAD_EPSILON:
        return {mid: 1.0 for mid in cosines}

    out: dict[int, float] = {}
    for mid, c in cosines.items():
        normalized = (c - bottom) / spread
        out[mid] = 1.0 if normalized >= QUALIFIER_TOP_RATIO else normalized
    return out


# ---------------------------------------------------------------------------
# Cross-space combiners
# ---------------------------------------------------------------------------


def _max_combine(
    movie_ids: Iterable[int],
    per_space_scores: dict[VectorName, dict[int, float]],
) -> dict[int, float]:
    # Carver: each space asks the same question with different
    # evidence types. One strong signal is sufficient — take the max
    # across active spaces. Missing-from-space contributes 0 (no
    # entry → no evidence from that space, but other spaces can
    # carry the answer). Candidates that score 0 across every space
    # land at 0 naturally and contribute nothing downstream via the
    # "missing positive = opportunity cost" rule.
    if not per_space_scores:
        return {}
    out: dict[int, float] = {}
    for mid in movie_ids:
        best = 0.0
        for space_map in per_space_scores.values():
            v = space_map.get(mid, 0.0)
            if v > best:
                best = v
        out[mid] = best
    return out


def _weighted_sum_combine(
    movie_ids: Iterable[int],
    per_space_scores: dict[VectorName, dict[int, float]],
    per_space_weights: dict[VectorName, float],
) -> dict[int, float]:
    # Qualifier: Σ(w · score) / Σw across active spaces. CENTRAL
    # spaces are load-bearing and dominate; SUPPORTING spaces round
    # out the match. Missing per-space score = 0.0 (a candidate in
    # the pool but absent from one space's score map contributes 0
    # for that space rather than being dropped).
    total_weight = sum(per_space_weights.values())
    if total_weight <= 0.0:
        return {mid: 0.0 for mid in movie_ids}
    out: dict[int, float] = {}
    for mid in movie_ids:
        numerator = 0.0
        for space, weight in per_space_weights.items():
            numerator += weight * per_space_scores.get(space, {}).get(mid, 0.0)
        out[mid] = max(0.0, min(1.0, numerator / total_weight))
    return out


# ---------------------------------------------------------------------------
# Per-call inputs unpack
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CallInputs:
    # Common unpack of params.space_queries shared by all four
    # execution scenarios. The merge-validator on SemanticParameters
    # guarantees unique spaces, so building a {VectorName: weight}
    # dict is safe. weights are populated regardless of role —
    # carver paths ignore them, qualifier paths read them.
    entries: list[SemanticSpaceEntry]
    vector_names: list[VectorName]
    weights: dict[VectorName, float]


def _unpack_inputs(
    space_queries: list[WeightedSpaceQuery] | list[WeightedSpaceQuerySubintent],
) -> _CallInputs:
    entries = [wq.query for wq in space_queries]
    vector_names = [VectorName(e.space.value) for e in entries]
    weights = {
        vn: SPACE_WEIGHT_VALUES[wq.weight]
        for wq, vn in zip(space_queries, vector_names)
    }
    return _CallInputs(entries=entries, vector_names=vector_names, weights=weights)


# ---------------------------------------------------------------------------
# Execution scenarios
# ---------------------------------------------------------------------------


async def _execute_carver_restricted(
    inputs: _CallInputs,
    embeddings: list[list[float]],
    candidate_ids: set[int],
    *,
    qdrant_client: AsyncQdrantClient,
    metadata_filters: MetadataFilters | None = None,
) -> dict[int, float]:
    # Carver acting as a reranker on a supplied pool. Two parallel
    # fetches per space:
    #   - Corpus probe: sole purpose is elbow calibration. The
    #     filtered candidate pool is too small / too biased to
    #     calibrate against itself, so we always anchor the
    #     threshold against the corpus's natural distribution.
    #   - HasId on candidate_ids: produces the per-candidate
    #     cosines we score against the probe-derived calibration.
    # Per-candidate score = elbow decay of HasId cosine; combined
    # across spaces via max().
    n = len(inputs.entries)
    tasks = [
        _run_corpus_topn(
            emb, vn, qdrant_client=qdrant_client,
            metadata_filters=metadata_filters,
        )
        for emb, vn in zip(embeddings, inputs.vector_names)
    ] + [
        # No metadata_filters here — `_run_filtered_score` is the
        # reranker scoring path; `candidate_ids` already passed the
        # filter upstream at candidate-generation time.
        _run_filtered_score(emb, vn, candidate_ids, qdrant_client=qdrant_client)
        for emb, vn in zip(embeddings, inputs.vector_names)
    ]
    results = await asyncio.gather(*tasks)
    probes, filtered_maps = results[:n], results[n:]

    per_space_scores: dict[VectorName, dict[int, float]] = {}
    for vn, probe, filtered in zip(inputs.vector_names, probes, filtered_maps):
        calib = _detect_elbow_and_floor([cos for _, cos in probe])
        per_space_scores[vn] = {
            mid: _elbow_decay(cos, calib) for mid, cos in filtered.items()
        }

    return _max_combine(candidate_ids, per_space_scores)


async def _execute_carver_unrestricted(
    inputs: _CallInputs,
    embeddings: list[list[float]],
    *,
    qdrant_client: AsyncQdrantClient,
    metadata_filters: MetadataFilters | None = None,
) -> dict[int, float]:
    # Carver acting as a candidate generator. A single corpus probe
    # per space serves as both the calibration sample and the
    # candidate pool. A movie absent from one space's top-N
    # contributes 0 to that space's max input — clearing the elbow
    # while missing top-N is implausible by construction (the elbow
    # rank sits inside top-N).
    probes = await asyncio.gather(*[
        _run_corpus_topn(
            emb, vn, qdrant_client=qdrant_client,
            metadata_filters=metadata_filters,
        )
        for emb, vn in zip(embeddings, inputs.vector_names)
    ])

    per_space_scores: dict[VectorName, dict[int, float]] = {}
    candidate_pool: set[int] = set()
    for vn, probe in zip(inputs.vector_names, probes):
        calib = _detect_elbow_and_floor([cos for _, cos in probe])
        space_map: dict[int, float] = {}
        for mid, cos in probe:
            space_map[mid] = _elbow_decay(cos, calib)
            candidate_pool.add(mid)
        per_space_scores[vn] = space_map

    return _max_combine(candidate_pool, per_space_scores)


async def _execute_qualifier_restricted(
    inputs: _CallInputs,
    embeddings: list[list[float]],
    candidate_ids: set[int],
    *,
    qdrant_client: AsyncQdrantClient,
) -> dict[int, float]:
    # Qualifier acting as a reranker on a supplied pool. HasId per
    # space produces raw cosines; pool-relative rescale converts
    # them into a [0, 1] ranking purely internal to the pool. No
    # corpus probe — the question "how X are these movies?" is
    # answered relative to the supplied pool, not to the corpus.
    # Cross-space combine is weighted-sum so CENTRAL/SUPPORTING
    # structure shapes the final ranking.
    # `_run_filtered_score` is the reranker scoring path — no filter
    # needed; the supplied pool has already passed the filter upstream.
    per_space_lookups = await asyncio.gather(*[
        _run_filtered_score(emb, vn, candidate_ids, qdrant_client=qdrant_client)
        for emb, vn in zip(embeddings, inputs.vector_names)
    ])
    per_space_scores: dict[VectorName, dict[int, float]] = {
        vn: _pool_relative_rescale(cos_map)
        for vn, cos_map in zip(inputs.vector_names, per_space_lookups)
    }
    return _weighted_sum_combine(
        candidate_ids, per_space_scores, inputs.weights
    )


async def _execute_qualifier_promoted(
    inputs: _CallInputs,
    embeddings: list[list[float]],
    *,
    qdrant_client: AsyncQdrantClient,
    metadata_filters: MetadataFilters | None = None,
) -> dict[int, float]:
    # Qualifier promoted to candidate generator via tier-fallback.
    # Per-space corpus probe acts as both calibration AND pool
    # because there is no upstream pool to be relative against —
    # within-space normalization degrades to absolute (corpus
    # elbow). Cross-space combine stays weighted-sum: the trait
    # still expresses CENTRAL/SUPPORTING structure that should not
    # be flattened by the orchestration's promotion of a qualifier
    # into pool-defining duty.
    probes = await asyncio.gather(*[
        _run_corpus_topn(
            emb, vn, qdrant_client=qdrant_client,
            metadata_filters=metadata_filters,
        )
        for emb, vn in zip(embeddings, inputs.vector_names)
    ])

    per_space_scores: dict[VectorName, dict[int, float]] = {}
    candidate_pool: set[int] = set()
    for vn, probe in zip(inputs.vector_names, probes):
        calib = _detect_elbow_and_floor([cos for _, cos in probe])
        space_map: dict[int, float] = {}
        for mid, cos in probe:
            space_map[mid] = _elbow_decay(cos, calib)
            candidate_pool.add(mid)
        per_space_scores[vn] = space_map

    return _weighted_sum_combine(
        candidate_pool, per_space_scores, inputs.weights
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def execute_semantic_query(
    params: SemanticParams,
    *,
    restrict_to_movie_ids: set[int] | None = None,
    qdrant_client: AsyncQdrantClient,
    metadata_filters: MetadataFilters | None = None,
) -> EndpointResult:
    """Execute a semantic payload and return scored candidates.

    Dispatch on (role, restrict-presence):

      role=CARVER, restrict supplied   → carver-restricted (reranker):
          probe ‖ HasId per space, elbow decay, max combine.
      role=CARVER, restrict=None       → carver-unrestricted (candidate
          generator): probe per space serves as pool, elbow decay, max
          combine.
      role=QUALIFIER, restrict supplied → qualifier-restricted
          (reranker): HasId per space, pool-relative rescale,
          weighted-sum combine.
      role=QUALIFIER, restrict=None    → qualifier-promoted via
          tier-fallback (candidate generator): probe per space serves
          as pool, elbow decay, weighted-sum combine.

    Empty restrict (set()) is normalized to None — the orchestrator
    is the source of truth for candidate-generator vs reranker
    dispatch.

    Polarity is NOT consulted here — both positive and negative
    findings return the same EndpointResult shape; the orchestrator
    routes IDs/scores into inclusion/exclusion or
    preference/downrank buckets per role.

    Retry contract: one transient retry, then EndpointResult()
    rather than raising.
    """
    is_carver = params.role is SemanticRetrievalShape.CARVER

    # Candidate-generator vs reranker dispatch is decided by the
    # orchestrator (build_endpoint_coroutine) before this call. If
    # an empty restrict set leaks through, treat it as None and run
    # as candidate generator rather than re-deciding here.
    if restrict_to_movie_ids is not None and len(restrict_to_movie_ids) == 0:
        restrict_to_movie_ids = None

    inputs = _unpack_inputs(list(params.space_queries))

    log_context = (
        f"role={'carver' if is_carver else 'qualifier'}, "
        f"restrict={'yes' if restrict_to_movie_ids is not None else 'no'}, "
        f"spaces={[vn.value for vn in inputs.vector_names]}"
    )

    scores: dict[int, float] = {}
    for attempt in range(2):
        try:
            # Batch all space queries into ONE embedding call. The
            # previous gather-of-singletons paid one HTTP RTT per
            # entry; the bulk call collapses that to one round-trip.
            embeddings = await _embed_bodies(
                [e.content for e in inputs.entries]
            )
            if is_carver:
                if restrict_to_movie_ids is None:
                    scores = await _execute_carver_unrestricted(
                        inputs, embeddings,
                        qdrant_client=qdrant_client,
                        metadata_filters=metadata_filters,
                    )
                else:
                    scores = await _execute_carver_restricted(
                        inputs, embeddings, restrict_to_movie_ids,
                        qdrant_client=qdrant_client,
                        metadata_filters=metadata_filters,
                    )
            else:
                if restrict_to_movie_ids is None:
                    scores = await _execute_qualifier_promoted(
                        inputs, embeddings,
                        qdrant_client=qdrant_client,
                        metadata_filters=metadata_filters,
                    )
                else:
                    # qualifier_restricted is a pure pool reranker; no
                    # metadata_filters needed (pool already filtered).
                    scores = await _execute_qualifier_restricted(
                        inputs, embeddings, restrict_to_movie_ids,
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
