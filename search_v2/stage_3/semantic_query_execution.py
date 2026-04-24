# Search V2 — Stage 3 Semantic Endpoint: Query Execution
#
# Takes the unified SemanticParameters payload (the inner parameters
# of a SemanticEndpointParameters wrapper) plus the wrapper's
# action_role, runs the corresponding vector search against Qdrant,
# and returns the scored EndpointResult.
#
# Dispatch is a 2x2 across two orthogonal signals:
#
#                              | restrict_to_movie_ids=None | restrict_to_movie_ids=set
#   --------------------------+----------------------------+-----------------------------
#   CANDIDATE_IDENTIFICATION  | D2 (generate candidates)   | D1 (score pool)
#   CANDIDATE_RERANKING       | P2 (generate candidates)   | P1 (score pool)
#
# CANDIDATE_IDENTIFICATION paths use ONLY the space_queries entry
# whose .space matches primary_vector; weight is ignored. A single
# space, embedded, probed, and threshold+flattened. D2 reuses the
# corpus probe as both calibration sample and candidate pool; D1
# adds a filtered HasId lookup for the supplied pool.
#
# CANDIDATE_RERANKING paths use ALL space_queries entries with their
# SpaceWeight (central=2.0, supporting=1.0); primary_vector is
# ignored. Per-space cosines combine via Σ(w × cos) / Σw. P2 unions
# per-space top-N probes into a pool then fills missing cosines via
# HasId; P1 runs one filtered lookup per space against the supplied
# pool.
#
# Polarity is NOT an executor concern. Both positive and negative
# findings produce the same EndpointResult shape; the orchestrator
# routes the returned IDs/scores into inclusion_candidates vs
# exclusion_ids (for identification) or preference_specs vs
# downrank_candidates (for reranking). See
# search_improvement_planning/category_handler_planning.md
# ("From LLM output to return buckets").
#
# Retry contract matches sibling executors: transient errors retry
# once, then the second failure returns an empty EndpointResult
# (never raises). This lets the orchestrator treat "failed" and "no
# match" identically.

from __future__ import annotations

import asyncio
import logging
from typing import Iterable, Union

import numpy as np
from kneed import KneeLocator
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, HasIdCondition

from db.vector_search import COLLECTION_ALIAS, QDRANT_SEARCH_PARAMS
from implementation.classes.enums import VectorName
from implementation.llms.generic_methods import generate_vector_embedding
from schemas.endpoint_result import EndpointResult
from schemas.enums import ActionRole
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
    SemanticSpaceEntry,
    SpaceWeight,
)
from search_v2.stage_3.result_helpers import build_endpoint_result

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#
# Every numeric value here is called out as tunable in the proposal's
# "Decisions Deferred to Implementation" section. Keeping them as
# module-level constants (not inline literals) makes evaluation work
# a one-edit affair.

# Top-N for the unfiltered corpus probe. Serves both as the
# elbow/floor calibration sample and, in D2/P2, as the candidate pool.
CORPUS_PROBE_LIMIT = 2000

# Flat-distribution detector. Fires when the top-N probe shows no
# discriminable structure — the full range between the top and
# bottom cosine falls below this. In that case no real elbow exists
# and we fall back to fixed-ratio elbow/floor. (The proposal's
# "max(y_diff) < 0.05" wording was ambiguous against raw cosines;
# range is the operationally meaningful quantity — a 0.05 range means
# "everything looks the same" regardless of smoothing or length.)
PATHOLOGY_RANGE_THRESHOLD = 0.05
PATHOLOGY_ELBOW_RATIO = 0.85
PATHOLOGY_FLOOR_RATIO = 0.65

# EWMA smoothing span: max(EWMA_SPAN_FLOOR, N / EWMA_SPAN_DIVISOR).
EWMA_SPAN_DIVISOR = 100
EWMA_SPAN_FLOOR = 5

# If the first detected knee sits earlier than this rank AND another
# knee exists, skip to the next knee. Guards against outlier-driven
# early knees pinching the 1.0 boundary too tightly.
RANK_10_SAFEGUARD = 10

# Below this probe length, skip Kneedle entirely and use the
# pathology fallback — too few samples for a meaningful curve.
MIN_PROBE_SIZE = 20

# Must match the ingestion-side model in
# movie_ingestion/final_ingestion/ingest_movie.py so query embeddings
# land in the same space as document embeddings. (CLAUDE.md references
# "-small" but the code has been on "-large" for a while.)
EMBEDDING_MODEL = "text-embedding-3-large"

# Two-level categorical space weights used in the reranking
# (preference) paths. central vs supporting maps to the numeric
# multipliers the weighted-sum combiner uses.
SPACE_WEIGHT_VALUES: dict[SpaceWeight, float] = {
    SpaceWeight.CENTRAL: 2.0,
    SpaceWeight.SUPPORTING: 1.0,
}


# Union of all query-side Body types. Every variant exposes
# embedding_text() -> str producing the structured-label string that
# mirrors the ingestion-side vector text for its space. Anchor is
# intentionally excluded — the unified SemanticSpace enum covers 7
# non-anchor spaces, so no entry can carry an AnchorBody.
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
# Primitives
# ---------------------------------------------------------------------------


async def _embed_body(body: SemanticBody) -> list[float]:
    # generate_vector_embedding is batch-shaped; unwrap the single
    # embedding here so callers don't repeat the [0] indexing dance.
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
    # Unfiltered top-N on the selected named vector. Returns
    # [(movie_id, cosine), ...] in Qdrant's native descending order.
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
    # Score a specific set of movie_ids against a single embedding on
    # the chosen named vector. Movie_id is the Qdrant point ID itself
    # (ingest_movie.py writes PointStruct(id=movie_id, ...)), so we
    # filter via HasIdCondition rather than a payload FieldCondition.
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
# Elbow / floor calibration
# ---------------------------------------------------------------------------


def _ewma(values: list[float], span: int) -> np.ndarray:
    # Exponentially-weighted moving average, forward pass. Mirrors
    # pandas.Series(...).ewm(span=span, adjust=False).mean() without
    # pulling in pandas for one smoothing op. Input is ~2000 floats,
    # so the Python loop is microseconds.
    alpha = 2.0 / (span + 1.0)
    out = np.empty(len(values), dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _detect_elbow_floor(similarities: list[float]) -> tuple[float, float]:
    # Returns (elbow_sim, floor_sim) with 0 <= floor < elbow <= 1.
    # Input must be sorted descending (Qdrant's natural order).
    #
    # Algorithm per finalized_search_proposal.md's "Elbow/floor
    # detection" decisions: EWMA smoothing → pathology check → Kneedle
    # with rank-<10 safeguard → gap-proportional floor fallback.
    if not similarities:
        return (0.0, 0.0)

    max_sim = float(similarities[0])

    # Short-probe guard — Kneedle needs a curve to work with.
    if len(similarities) < MIN_PROBE_SIZE:
        return _pathology_fallback(max_sim, reason="probe too short")

    # Pathology: if top-to-bottom range is too narrow, the
    # distribution carries no discriminable structure.
    if max_sim - float(similarities[-1]) < PATHOLOGY_RANGE_THRESHOLD:
        return _pathology_fallback(max_sim, reason="flat distribution")

    span = max(EWMA_SPAN_FLOOR, len(similarities) // EWMA_SPAN_DIVISOR)
    smoothed = _ewma(similarities, span)

    # Kneedle over the smoothed curve. all_knees returns every
    # detected inflection; we pick intentionally rather than trusting
    # the library's default .knee (which is the most prominent one,
    # not necessarily the earliest).
    locator = KneeLocator(
        x=list(range(len(smoothed))),
        y=smoothed.tolist(),
        curve="convex",
        direction="decreasing",
        S=1,
        online=True,
    )
    # all_knees may come back as a set, list, or None across minor
    # versions — normalize to a sorted ascending list.
    knees_raw = locator.all_knees if locator.all_knees else []
    knees = sorted({int(k) for k in knees_raw})
    if not knees:
        return _pathology_fallback(max_sim, reason="no knees detected")

    # Elbow selection. Prefer the earliest knee, but if it sits
    # unreasonably early (outlier-driven pinch) and a later knee
    # exists, skip forward.
    if knees[0] < RANK_10_SAFEGUARD and len(knees) >= 2:
        elbow_rank = knees[1]
        used_first_knee = False
    else:
        elbow_rank = knees[0]
        used_first_knee = True
    # Translate rank → threshold using raw cosines, not the smoothed
    # curve. EWMA lags raw values on a monotonically-decreasing
    # sequence (smoothed[i] > raw[i]), so using smoothed y-values
    # inflates the threshold and shrinks the "pass" zone when
    # orchestrator compares raw Qdrant scores against it. Smoothing's
    # only job here is finding the elbow's rank.
    elbow_sim = float(similarities[elbow_rank])

    # Floor selection. Prefer a second-knee floor when Kneedle
    # exposes one (e.g., bimodal Christmas distribution). Otherwise
    # compute a gap-proportional floor that widens the decay zone for
    # sharp elbows and narrows it for compressed ones.
    floor_rank: int | None = None
    if used_first_knee and len(knees) >= 2:
        floor_rank = knees[1]
    elif not used_first_knee and len(knees) >= 3:
        floor_rank = knees[2]

    if floor_rank is not None:
        floor_sim = float(similarities[floor_rank])
    else:
        floor_sim = max(elbow_sim - 2.0 * (max_sim - elbow_sim), 0.0)

    # Clamp invariant: 0 <= floor < elbow <= 1.
    floor_sim = max(0.0, min(floor_sim, elbow_sim - 1e-9))
    elbow_sim = max(0.0, min(elbow_sim, 1.0))
    return (elbow_sim, floor_sim)


def _pathology_fallback(max_sim: float, *, reason: str) -> tuple[float, float]:
    # Shared fallback for flat / too-short / no-knee distributions.
    # Logs at INFO so calibration audits can spot concepts that
    # consistently hit this path.
    logger.info(
        "Semantic calibration falling back to fixed-ratio elbow/floor "
        "(reason=%s, max_sim=%.4f)",
        reason, max_sim,
    )
    elbow = max_sim * PATHOLOGY_ELBOW_RATIO
    floor = max_sim * PATHOLOGY_FLOOR_RATIO
    return (elbow, floor)


def _threshold_flatten(sim: float, elbow: float, floor: float) -> float:
    # Dealbreaker-only score transform (preference paths use
    # `_weighted_cosine_score` and never reach this function). Above
    # elbow → full credit (1.0); below floor → 0.0 (dropped by the
    # caller's `if s > 0.0` guard); strictly between → linear decay
    # compressed into [0.5, 1.0] so every match sits at or above the
    # stage-3 dealbreaker floor.
    if sim >= elbow:
        return 1.0
    if sim <= floor:
        return 0.0
    # Defensive: shouldn't fire after the floor-clamp in detection,
    # but avoids a divide-by-zero if it ever does.
    if elbow <= floor:
        return 1.0 if sim >= elbow else 0.0
    raw = (sim - floor) / (elbow - floor)
    # raw ∈ (0, 1) strictly because of the equality guards above, so the
    # compressed output lands in (0.5, 1.0) — no risk of colliding with
    # the "dropped" 0.0 sentinel.
    return 0.5 + 0.5 * raw


# ---------------------------------------------------------------------------
# Preference score assembly
# ---------------------------------------------------------------------------


def _weighted_cosine_score(
    movie_ids: Iterable[int],
    per_space_cosines: dict[VectorName, dict[int, float]],
    per_space_weights: dict[VectorName, float],
) -> dict[int, float]:
    # Raw weighted-sum cosine: Σ(w_space × cos_space) / Σ(w_space).
    # Missing cosines default to 0.0 — a candidate present in the
    # union but absent from a given space's fetched map contributes
    # 0 for that space rather than being dropped.
    total_weight = sum(per_space_weights.values())
    if total_weight <= 0.0:
        # Guarded upstream by SemanticParameters.space_queries'
        # min_length=1, but defense-in-depth keeps this helper pure.
        return {int(mid): 0.0 for mid in movie_ids}

    out: dict[int, float] = {}
    for mid in movie_ids:
        mid_int = int(mid)
        numerator = 0.0
        for space, weight in per_space_weights.items():
            cos = per_space_cosines.get(space, {}).get(mid_int, 0.0)
            numerator += weight * cos
        # Clamp to [0, 1] — ScoredCandidate validates the range and a
        # floating-point blip above 1.0 would otherwise reject the row.
        score = numerator / total_weight
        out[mid_int] = max(0.0, min(1.0, score))
    return out


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------


def _entry_for_primary_vector(params: SemanticParameters) -> SemanticSpaceEntry:
    # Return the space_queries entry whose .space matches
    # params.primary_vector. SemanticParameters'
    # _primary_vector_in_space_queries validator guarantees this entry
    # exists at parse time, so the loop always finds a match in
    # practice. The raise at the bottom is a safety net against a
    # validator bypass (e.g., object built programmatically) — a loud
    # failure beats silently picking the first entry.
    for entry in params.space_queries:
        if entry.space == params.primary_vector:
            return entry
    raise ValueError(
        f"primary_vector {params.primary_vector!r} not found in space_queries "
        f"(populated: {[e.space for e in params.space_queries]!r})"
    )


async def _execute_dealbreaker_d1(
    params: SemanticParameters,
    candidate_ids: set[int],
    *,
    qdrant_client: AsyncQdrantClient,
) -> dict[int, float]:
    # D1: score a pre-built candidate pool on the primary_vector
    # entry. The global probe and the filtered candidate fetch are
    # independent — gather them.
    entry = _entry_for_primary_vector(params)
    vector_name = VectorName(entry.space.value)
    embedding = await _embed_body(entry.content)

    probe, filtered = await asyncio.gather(
        _run_corpus_topn(embedding, vector_name, qdrant_client=qdrant_client),
        _run_filtered_score(
            embedding, vector_name, candidate_ids, qdrant_client=qdrant_client
        ),
    )
    elbow, floor = _detect_elbow_floor([cos for _, cos in probe])

    # Omit zeros — build_endpoint_result refills them to 0.0 when
    # restrict_to_movie_ids is the candidate pool (preference-path
    # semantics in result_helpers). This keeps the natural-match vs.
    # pool-scoring distinction centralized in the helper.
    scores: dict[int, float] = {}
    for mid in candidate_ids:
        cos = filtered.get(int(mid), 0.0)
        s = _threshold_flatten(cos, elbow, floor)
        if s > 0.0:
            scores[int(mid)] = s
    return scores


async def _execute_dealbreaker_d2(
    params: SemanticParameters,
    *,
    qdrant_client: AsyncQdrantClient,
) -> dict[int, float]:
    # D2: candidate-generating on the primary_vector entry. One Qdrant
    # call — the top-N probe doubles as both the calibration sample
    # and the candidate pool. Cross-dealbreaker scoring is explicitly
    # NOT performed here; each semantic dealbreaker scores only the
    # movies it retrieved.
    entry = _entry_for_primary_vector(params)
    vector_name = VectorName(entry.space.value)
    embedding = await _embed_body(entry.content)

    probe = await _run_corpus_topn(
        embedding, vector_name, qdrant_client=qdrant_client
    )
    elbow, floor = _detect_elbow_floor([cos for _, cos in probe])

    scores: dict[int, float] = {}
    for mid, cos in probe:
        s = _threshold_flatten(cos, elbow, floor)
        if s > 0.0:
            scores[mid] = s
    return scores


async def _execute_preference_p1(
    params: SemanticParameters,
    candidate_ids: set[int],
    *,
    qdrant_client: AsyncQdrantClient,
) -> dict[int, float]:
    # P1: score a pre-built pool via raw weighted-sum cosine across
    # every populated space. No elbow calibration. primary_vector is
    # deliberately ignored — the reranking path consumes the whole
    # space_queries list regardless.
    entries = list(params.space_queries)
    vector_names = [VectorName(e.space.value) for e in entries]
    per_space_weights = {
        vn: SPACE_WEIGHT_VALUES[e.weight]
        for vn, e in zip(vector_names, entries)
    }

    # Embed every selected space in parallel, then score each space's
    # candidate-filtered cosines in parallel.
    embeddings = await asyncio.gather(
        *[_embed_body(e.content) for e in entries]
    )
    per_space_lookups = await asyncio.gather(
        *[
            _run_filtered_score(
                emb, vn, candidate_ids, qdrant_client=qdrant_client
            )
            for emb, vn in zip(embeddings, vector_names)
        ]
    )
    per_space_cosines: dict[VectorName, dict[int, float]] = dict(
        zip(vector_names, per_space_lookups)
    )

    return _weighted_cosine_score(
        candidate_ids, per_space_cosines, per_space_weights
    )


async def _execute_preference_p2(
    params: SemanticParameters,
    *,
    qdrant_client: AsyncQdrantClient,
) -> dict[int, float]:
    # P2: candidate-generating across every populated space. Top-N per
    # space against full corpus, union = pool. Each top-N probe also
    # doubles as that space's cosine map for members it naturally
    # retrieved; missing cosines (candidate in pool via another
    # space's probe) are filled via targeted HasId lookups.
    # primary_vector is ignored here for the same reason as P1.
    entries = list(params.space_queries)
    vector_names = [VectorName(e.space.value) for e in entries]
    per_space_weights = {
        vn: SPACE_WEIGHT_VALUES[e.weight]
        for vn, e in zip(vector_names, entries)
    }

    embeddings = await asyncio.gather(
        *[_embed_body(e.content) for e in entries]
    )
    probes = await asyncio.gather(
        *[
            _run_corpus_topn(emb, vn, qdrant_client=qdrant_client)
            for emb, vn in zip(embeddings, vector_names)
        ]
    )

    # Seed per-space cosines from the probes themselves, then
    # identify which IDs in the union are missing from each space.
    per_space_cosines: dict[VectorName, dict[int, float]] = {
        vn: dict(probe) for vn, probe in zip(vector_names, probes)
    }
    candidate_ids: set[int] = set().union(
        *(cos_map.keys() for cos_map in per_space_cosines.values())
    )
    if not candidate_ids:
        return {}

    missing_by_space = {
        vn: candidate_ids - per_space_cosines[vn].keys()
        for vn in vector_names
    }

    # Fill only the spaces that have at least one missing ID — skip
    # the no-op Qdrant calls.
    fill_tasks = [
        _run_filtered_score(
            embeddings[i],
            vn,
            missing_by_space[vn],
            qdrant_client=qdrant_client,
        )
        for i, vn in enumerate(vector_names)
        if missing_by_space[vn]
    ]
    fill_targets = [vn for vn in vector_names if missing_by_space[vn]]
    fills = await asyncio.gather(*fill_tasks) if fill_tasks else []
    for vn, fill in zip(fill_targets, fills):
        per_space_cosines[vn].update(fill)

    return _weighted_cosine_score(
        candidate_ids, per_space_cosines, per_space_weights
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def execute_semantic_query(
    params: SemanticParameters,
    *,
    action_role: ActionRole,
    restrict_to_movie_ids: set[int] | None = None,
    qdrant_client: AsyncQdrantClient,
) -> EndpointResult:
    """Execute a SemanticParameters payload and return scored candidates.

    Two orthogonal signals drive the 2×2 of scenarios:

      action_role == CANDIDATE_IDENTIFICATION → dealbreaker path
        (uses only the space_queries entry whose .space matches
        primary_vector; weight is ignored).
      action_role == CANDIDATE_RERANKING → preference path
        (uses every space_queries entry with its weight; primary_vector
        is ignored).

      restrict_to_movie_ids is None → candidate-generating scenario
        (D2 or P2): run top-N probe(s) against the full corpus and
        score the retrieved movies.
      restrict_to_movie_ids is set[int] → pool-scoring scenario
        (D1 or P1): score exactly the supplied pool (one ScoredCandidate
        per supplied ID; absent matches default to 0.0 in
        build_endpoint_result).
      restrict_to_movie_ids is empty set → short-circuit with
        EndpointResult() and no Qdrant traffic.

    Polarity is NOT consulted here. Both positive and negative findings
    return the same EndpointResult shape; the orchestrator decides
    whether the IDs/scores feed inclusion_candidates vs exclusion_ids
    (identification paths) or preference_specs vs downrank_candidates
    (reranking paths).

    Retry contract: transient Qdrant or embedding errors retry once;
    a second failure returns EndpointResult() rather than raising.
    Orchestrator-side handling treats "failed" and "no match" the
    same way.
    """
    # Validate action_role before the retry loop. A bogus value is a
    # contract violation, not a transient error — wrapping it in the
    # try/except below would launder a programmer bug into a silent
    # empty result with misleading "retrying" logs. Convention
    # docs/conventions.md §"Preserve retryable exception types"
    # applies here: non-retryable exceptions must not ride the same
    # soft-fail path as transient I/O. Also builds the per-branch log
    # context while we already know which branch we're on, so retry
    # logs surface only the fields that branch actually consults.
    if action_role == ActionRole.CANDIDATE_IDENTIFICATION:
        log_context = f"primary_vector={params.primary_vector.value}"
    elif action_role == ActionRole.CANDIDATE_RERANKING:
        log_context = (
            "spaces="
            f"{[e.space.value for e in params.space_queries]}"
        )
    else:
        raise ValueError(f"Unhandled action_role: {action_role!r}")

    if restrict_to_movie_ids is not None and len(restrict_to_movie_ids) == 0:
        return EndpointResult()

    scores: dict[int, float] = {}
    for attempt in range(2):
        try:
            if action_role == ActionRole.CANDIDATE_IDENTIFICATION:
                if restrict_to_movie_ids is not None:
                    scores = await _execute_dealbreaker_d1(
                        params,
                        restrict_to_movie_ids,
                        qdrant_client=qdrant_client,
                    )
                else:
                    scores = await _execute_dealbreaker_d2(
                        params, qdrant_client=qdrant_client
                    )
            else:  # CANDIDATE_RERANKING — already validated above.
                if restrict_to_movie_ids is not None:
                    scores = await _execute_preference_p1(
                        params,
                        restrict_to_movie_ids,
                        qdrant_client=qdrant_client,
                    )
                else:
                    scores = await _execute_preference_p2(
                        params, qdrant_client=qdrant_client
                    )
            break
        except Exception:
            if attempt == 0:
                logger.warning(
                    "Semantic query error on first attempt, retrying "
                    "(action_role=%s, %s)",
                    action_role,
                    log_context,
                    exc_info=True,
                )
                continue
            logger.error(
                "Semantic query error on retry, returning empty "
                "(action_role=%s, %s)",
                action_role,
                log_context,
                exc_info=True,
            )
            return EndpointResult()

    return build_endpoint_result(scores, restrict_to_movie_ids)
