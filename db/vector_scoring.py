"""
Vector Score Calculation Module
===============================

Converts per-collection cosine similarity scores from Qdrant into a single
[0, 1] final vector score for each candidate movie.

Pipeline stages:
    1. Determine search execution flags per vector space
    2. Blend original + subquery scores per space per candidate
    3. Normalize blended scores within each space (exponential decay from best)
    4. Compute normalized weight array across all 8 spaces
    5. Weighted sum → final vector score per candidate

See vector_scoring_plan.md for the full design rationale, numerical examples,
edge case analysis, and tuning guidance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Optional, NamedTuple

from db.vector_search import CandidateVectorScores, VectorSearchResult
from implementation.classes.schemas import VectorWeights, VectorSubqueries
from implementation.classes.enums import RelevanceSize, VectorName


# ===========================================================================
# Tunable constants
# ===========================================================================
# Each constant is referenced by name in vector_scoring_plan.md Appendix C.
# Changing any of these affects scoring behavior for ALL queries — tune with
# care and measure impact on a held-out eval set before deploying.

# --- Stage 2: Original/subquery blend ratio ---
# Controls how much the purpose-built subquery dominates vs the raw user query.
# Higher SUBQUERY_BLEND_WEIGHT means the LLM-generated subquery has more say.
SUBQUERY_BLEND_WEIGHT: float = 0.8
ORIGINAL_BLEND_WEIGHT: float = 1.0 - SUBQUERY_BLEND_WEIGHT  # = 0.2

# --- Stage 3: Exponential decay steepness ---
# Higher k = steeper falloff from the best candidate in each space.
# k=3.0 means: best=1.0, 90th-percentile gap≈0.74, worst-in-pool≈0.05.
DECAY_K: float = 3.0

# --- Stage 4: Anchor weight scaling ---
# Anchor's raw weight = ANCHOR_MEAN_FRACTION × mean(active non-anchor weights).
# 0.8 keeps anchor present but never the loudest voice.
ANCHOR_MEAN_FRACTION: float = 0.8

# --- Stage 4: RelevanceSize → raw numeric weight mapping ---
# Ratios matter more than absolute values (they get normalized to sum=1).
RELEVANCE_RAW_WEIGHTS: dict[RelevanceSize, float] = {
    RelevanceSize.NOT_RELEVANT: 0.0,
    RelevanceSize.SMALL: 1.0,
    RelevanceSize.MEDIUM: 2.0,
    RelevanceSize.LARGE: 3.0,
}

# --- Stage 5: Floating-point tolerance thresholds ---
# These guard against accumulated rounding errors after normalization and
# weighted summation.  Kept separate so each can be tuned independently.
# WEIGHT_SUM_ABS_TOL: maximum acceptable deviation of the active weight sum
#   from 1.0.  Stage 4 normalizes via division, so error is bounded by a
#   few ULPs per weight — 1e-9 is conservative for up to 8 terms.
# SCORE_BOUNDARY_TOL: maximum acceptable overshoot of a final score beyond
#   [0.0, 1.0].  Larger than WEIGHT_SUM_ABS_TOL because the weighted sum
#   accumulates products of two floats across up to 8 spaces.
WEIGHT_SUM_ABS_TOL: float = 1e-9
SCORE_BOUNDARY_TOL: float = 1e-9


# ===========================================================================
# Space configuration registry
# ===========================================================================
# Maps each of the 8 vector spaces to the attribute names on the existing
# dataclasses (CandidateVectorScores, VectorWeights, VectorSubqueries).
# This avoids scattered string literals and makes adding a new vector space
# a single-line change.


class _SpaceConfig(NamedTuple):
    """
    Immutable descriptor for one vector space.

    Fields:
        name:                Human-readable space name (used as dict key everywhere).
        original_score_attr: Attribute on CandidateVectorScores for the original-query
                             cosine similarity (always present for every space).
        subquery_score_attr: Attribute on CandidateVectorScores for the subquery cosine
                             similarity. None for anchor (anchor never has a subquery).
        weight_attr:         Attribute on VectorWeights for this space's RelevanceSize.
                             None for anchor (anchor has no generated relevance).
        subquery_text_attr:  Attribute on VectorSubqueries for this space's subquery text.
                             None for anchor (anchor never generates a subquery).
    """
    name: VectorName
    original_score_attr: str
    subquery_score_attr: Optional[str]
    weight_attr: Optional[str]
    subquery_text_attr: Optional[str]


def _build_space_configs() -> tuple[_SpaceConfig, ...]:
    """Build immutable space configs directly from VectorName members."""
    configs: list[_SpaceConfig] = []
    for vector_name in VectorName:
        if vector_name == VectorName.ANCHOR:
            subquery_score_attr = None
            weight_attr = None
            subquery_text_attr = None
        else:
            subquery_score_attr = f"{vector_name.value}_score_subquery"
            weight_attr = f"{vector_name.value}_weight"
            subquery_text_attr = f"{vector_name.value}_subquery"

        configs.append(_SpaceConfig(
            name=vector_name,
            original_score_attr=f"{vector_name.value}_score_original",
            subquery_score_attr=subquery_score_attr,
            weight_attr=weight_attr,
            subquery_text_attr=subquery_text_attr,
        ))
    return tuple(configs)


# Ordered tuple of all 8 spaces, driven by VectorName enum iteration order.
SPACE_CONFIGS: tuple[_SpaceConfig, ...] = _build_space_configs()


def _validate_numeric_score(value: object, *, label: str) -> float:
    """Validate one score value is a finite float in [0.0, 1.0]."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number, got {type(value).__name__}")

    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"{label} must be finite, got {score}")
    if score < 0.0 or score > 1.0:
        raise ValueError(f"{label} must be within [0.0, 1.0], got {score}")
    return score


def _read_candidate_score(
    candidate: CandidateVectorScores,
    *,
    movie_id: int,
    attr: str,
) -> float:
    """Read and validate one score attribute from a candidate object."""
    try:
        raw = getattr(candidate, attr)
    except AttributeError as exc:
        raise TypeError(
            f"candidate {movie_id} is missing required score attribute '{attr}'"
        ) from exc
    return _validate_numeric_score(raw, label=f"{attr} for movie_id={movie_id}")


def _validate_candidate_scores(
    candidates: dict[int, CandidateVectorScores],
    original_attr: str,
    subquery_attr: Optional[str],
) -> None:
    """Validate all score attributes on every candidate before the blend loop.

    This runs once per space per request. By validating upfront, the hot
    blend loop can use bare ``getattr`` without per-iteration function-call
    or try/except overhead.
    """
    attrs = [original_attr]
    if subquery_attr is not None:
        attrs.append(subquery_attr)

    for movie_id, scores in candidates.items():
        for attr in attrs:
            _read_candidate_score(scores, movie_id=movie_id, attr=attr)


# ===========================================================================
# Intermediate and output data structures
# ===========================================================================


@dataclass(slots=True)
class SpaceExecutionContext:
    """
    Computed once per space per request during Stage 1.

    Captures whether each search variant (original / subquery) was executed,
    the effective relevance after promotion rules, and the final normalized
    weight assigned in Stage 4.

    Also stored in the output for debug logging so you can inspect exactly
    how each space was configured for a given request.
    """
    name: VectorName

    # --- Search execution flags (Stage 1) ---
    # These reflect what ACTUALLY happened at Qdrant search time, based on
    # the *original* (pre-promotion) relevance. They determine the blend
    # formula in Stage 2.
    did_run_original: bool
    did_run_subquery: bool

    # --- Weight computation inputs (Stage 1 / Stage 4) ---
    # effective_relevance incorporates the promotion rule (not_relevant +
    # subquery exists → small). It's used for weight calculation only —
    # never for determining which searches ran.
    # None for anchor (anchor's weight is computed separately).
    effective_relevance: Optional[RelevanceSize]

    # --- Computed in Stage 4 ---
    # Final normalized weight ∈ [0, 1]. All 8 weights sum to 1.0.
    # Initialized to 0.0; set by compute_normalized_weights().
    normalized_weight: float = 0.0

    @property
    def is_active(self) -> bool:
        """True if this space participates in scoring at all.

        A space is active if at least one search variant ran. Inactive spaces
        contribute nothing to any candidate's final score.
        """
        return self.did_run_original or self.did_run_subquery


@dataclass(slots=True)
class VectorScoringResult:
    """
    Complete output of the vector scoring pipeline.

    final_scores:        movie_id → final vector score ∈ [0, 1].
                         Every movie_id from the input candidates dict has an entry.
    space_contexts:      One SpaceExecutionContext per space (len=8), for debug.
    per_space_normalized: space_name → {movie_id: normalized_score} for debug.
                         Only contains entries for active spaces. Within each space,
                         only candidates with normalized_score > 0 are included
                         (sparse representation).
    """
    final_scores: dict[int, float]
    space_contexts: list[SpaceExecutionContext]
    per_space_normalized: dict[VectorName, dict[int, float]]


# ===========================================================================
# Stage 1: Determine search execution flags per vector space
# ===========================================================================


def build_space_execution_contexts(
    vector_weights: VectorWeights,
    vector_subqueries: VectorSubqueries,
) -> list[SpaceExecutionContext]:
    """
    Determine, for each of the 8 vector spaces, which searches were executed
    and what the effective relevance is after applying promotion rules.

    This is a pure function of the query-understanding output. It does NOT
    look at any candidate scores — the execution flags are system-level
    properties of the request, not per-candidate.

    Promotion rule:
        If a space's original relevance is NOT_RELEVANT but a subquery text
        was generated for it, the effective relevance is promoted to SMALL.
        This gives the space a small weight in the final score even though
        the original query was never searched against it.

    Execution flags:
        did_run_original is based on the ORIGINAL (pre-promotion) relevance,
        because search dispatch already happened before scoring runs. We
        cannot retroactively execute a search that was skipped.

    Returns:
        List of 8 SpaceExecutionContext objects (one per SPACE_CONFIGS entry),
        in the same order as SPACE_CONFIGS. normalized_weight is 0.0 on all
        of them — it gets filled in by compute_normalized_weights() in Stage 4.
    """
    contexts: list[SpaceExecutionContext] = []

    for config in SPACE_CONFIGS:

        # --- Anchor: always runs original, never runs subquery ---
        if config.name == VectorName.ANCHOR:
            contexts.append(SpaceExecutionContext(
                name=config.name,
                did_run_original=True,
                did_run_subquery=False,
                effective_relevance=None,  # anchor weight is computed separately
            ))
            continue

        # --- Non-anchor spaces ---

        # Read the original relevance from the query-understanding output.
        # This is the relevance that was active when Qdrant searches were dispatched.
        original_relevance = getattr(vector_weights, config.weight_attr)
        if not isinstance(original_relevance, RelevanceSize):
            raise TypeError(
                f"vector_weights.{config.weight_attr} must be RelevanceSize, "
                f"got {type(original_relevance).__name__}"
            )

        # Read the subquery text (may be None if the LLM didn't generate one).
        subquery_text = getattr(vector_subqueries, config.subquery_text_attr)
        if subquery_text is not None and not isinstance(subquery_text, str):
            raise TypeError(
                f"vector_subqueries.{config.subquery_text_attr} must be str or None, "
                f"got {type(subquery_text).__name__}"
            )

        # Execution flags: based on pre-promotion relevance.
        # The architecture doc says:
        #   "Always search Anchor with the original query embedding"
        #   "For channels with a generated subquery: search with the subquery embedding"
        #   "Also search every channel with relevance > not_relevant using the original
        #    query embedding"
        did_run_original = (original_relevance != RelevanceSize.NOT_RELEVANT)
        did_run_subquery = (subquery_text is not None)

        # Promotion rule: bump weight (not execution) if subquery exists despite
        # not_relevant relevance. This gives the subquery-only search results a
        # small but nonzero contribution to the final score.
        if original_relevance == RelevanceSize.NOT_RELEVANT and subquery_text is not None:
            effective_relevance = RelevanceSize.SMALL
        else:
            effective_relevance = original_relevance

        contexts.append(SpaceExecutionContext(
            name=config.name,
            did_run_original=did_run_original,
            did_run_subquery=did_run_subquery,
            effective_relevance=effective_relevance,
        ))

    return contexts


# ===========================================================================
# Stage 2: Blend original + subquery scores per space per candidate
# ===========================================================================


def blend_space_scores(
    candidates: dict[int, CandidateVectorScores],
    config: _SpaceConfig,
    ctx: SpaceExecutionContext,
) -> dict[int, float]:
    """
    For one vector space, produce a single blended cosine similarity per
    candidate by combining the original-query score and subquery score
    according to the space's execution flags.

    Blend rules (from the plan):
        - Both ran:         0.8 × subquery + 0.2 × original
        - Original only:    1.0 × original
        - Subquery only:    1.0 × subquery
        - Neither ran:      not called (space is inactive)

    The 0.0 default on CandidateVectorScores means "this candidate was not in
    the top-N for that search." When both searches ran and a candidate scored
    0.0 on one of them, the blend correctly penalizes it — that's a real signal,
    not missing data.

    Returns:
        Sparse dict of movie_id → blended score. Only entries where the
        blended score is > 0.0 are included. Candidates with blended = 0.0
        (not returned in any executed search for this space) are omitted to
        save memory and skip unnecessary work in Stage 3.
    """
    if not isinstance(candidates, dict):
        raise TypeError(f"candidates must be dict[int, CandidateVectorScores], got {type(candidates).__name__}")
    if not isinstance(config, _SpaceConfig):
        raise TypeError(f"config must be _SpaceConfig, got {type(config).__name__}")
    if not isinstance(ctx, SpaceExecutionContext):
        raise TypeError(f"ctx must be SpaceExecutionContext, got {type(ctx).__name__}")
    if not ctx.did_run_original and not ctx.did_run_subquery:
        raise ValueError("invalid blend context: neither original nor subquery search ran")
    if ctx.did_run_subquery and config.subquery_score_attr is None:
        raise ValueError("invalid blend context/config: subquery mode requires a subquery score attribute")

    # Pre-resolve attribute name(s) for this space.
    original_attr: str = config.original_score_attr
    subquery_attr: Optional[str] = config.subquery_score_attr if ctx.did_run_subquery else None

    # Determine blend mode once — it's the same for every candidate in this space.
    both_ran = ctx.did_run_original and ctx.did_run_subquery
    original_only = ctx.did_run_original and not ctx.did_run_subquery
    # subquery_only is the remaining case: not ctx.did_run_original and ctx.did_run_subquery

    # Validate all candidate scores in a single pre-pass so the hot blend
    # loop below can use bare getattr without per-iteration overhead.
    _validate_candidate_scores(candidates, original_attr, subquery_attr)

    blended: dict[int, float] = {}

    if both_ran:
        # 80/20 blend. Both original and subquery scores exist on every
        # candidate (defaulting to 0.0 if the candidate wasn't in top-N).
        sub_w = SUBQUERY_BLEND_WEIGHT
        orig_w = ORIGINAL_BLEND_WEIGHT

        for movie_id, scores in candidates.items():
            value = (
                sub_w * getattr(scores, subquery_attr)
                + orig_w * getattr(scores, original_attr)
            )

            # Only keep positive blended scores. A score of 0.0 means the
            # candidate didn't appear in either the original or subquery
            # search for this space — it's not part of this space's pool.
            if value > 0.0:
                blended[movie_id] = value

    elif original_only:
        # 100% original. Subquery was never generated / never searched.
        for movie_id, scores in candidates.items():
            value = getattr(scores, original_attr)
            if value > 0.0:
                blended[movie_id] = value

    else:
        # Subquery only. Original was never searched (relevance was
        # not_relevant at search dispatch time, but subquery existed so the
        # weight was promoted to SMALL).
        for movie_id, scores in candidates.items():
            value = getattr(scores, subquery_attr)
            if value > 0.0:
                blended[movie_id] = value

    return blended


# ===========================================================================
# Stage 3: Normalize blended scores within each space
# ===========================================================================


def normalize_blended_scores(
    blended: dict[int, float],
    decay_k: float = DECAY_K,
) -> dict[int, float]:
    """
    Transform blended cosine similarities into [0, 1] normalized scores
    using exponential decay from the best candidate in the pool.

    The normalization pool is exactly the input dict — all entries have
    blended > 0 (enforced by blend_space_scores). Candidates not in this
    dict receive an implicit normalized score of 0.0.

    Formula:
        gap(s) = (s_max - s) / (s_max - s_min)     # ∈ [0, 1]
        normalized(s) = exp(-k × gap(s))

    Properties:
        - Best candidate (gap=0):  exp(0) = 1.0
        - Worst candidate (gap=1): exp(-k) ≈ 0.05 for k=3
        - If all candidates have the same score: range=0 → all get 1.0
        - Tightly clustered candidates all score near 1.0
        - A clear gap between leader and pack creates steep dropoff

    Args:
        blended: Sparse dict of movie_id → blended cosine similarity (all > 0).
        decay_k: Steepness parameter. Higher = more winner-take-all.

    Returns:
        Sparse dict of movie_id → normalized score ∈ (0, 1].
        Same keys as input (never adds or removes entries).
    """
    if not isinstance(blended, dict):
        raise TypeError(f"blended must be dict[int, float], got {type(blended).__name__}")
    if isinstance(decay_k, bool) or not isinstance(decay_k, Real):
        raise TypeError(f"decay_k must be a real number, got {type(decay_k).__name__}")
    if not math.isfinite(decay_k) or decay_k <= 0.0:
        raise ValueError(f"decay_k must be finite and > 0, got {decay_k}")

    # --- Fast path: empty pool ---
    if not blended:
        return {}

    # Validate all scores. The contract with blend_space_scores guarantees
    # every value is > 0.0 (candidates with blended=0 are excluded upstream).
    for movie_id, raw_score in blended.items():
        score = _validate_numeric_score(raw_score, label=f"blended score for movie_id={movie_id}")
        if score <= 0.0:
            raise ValueError(
                f"blended score for movie_id={movie_id} must be > 0.0, got {score}"
            )

    # --- Fast path: single candidate or all identical scores ---
    # Find max and min in a single pass over the values.
    # For typical pool sizes (hundreds to low thousands), this is faster
    # than sorting and avoids allocating a sorted copy.
    scores = blended.values()
    s_max = max(scores)
    s_min = min(scores)
    score_range = s_max - s_min

    if score_range == 0.0:
        # Every candidate has the same blended score.
        # They're all equally "best", so they all get 1.0.
        return dict.fromkeys(blended, 1.0)

    # --- General case: compute exponential decay from best ---
    # Pre-compute the inverse range to replace division with multiplication
    # inside the loop (minor optimization for large pools).
    inv_range = 1.0 / score_range

    # Pre-compute -decay_k to avoid negation inside the loop.
    neg_k = -decay_k

    normalized: dict[int, float] = {}
    for movie_id, score in blended.items():
        # gap ∈ [0, 1]: 0 for the best candidate, 1 for the worst in pool.
        gap = (s_max - score) * inv_range

        # Exponential decay: steep dropoff for candidates far from best.
        # math.exp is implemented in C and is ~4x faster than ** for this.
        normalized[movie_id] = math.exp(neg_k * gap)

    return normalized


# ===========================================================================
# Stage 4: Compute normalized weight array
# ===========================================================================


def compute_normalized_weights(
    contexts: list[SpaceExecutionContext],
    anchor_mean_fraction: float = ANCHOR_MEAN_FRACTION,
) -> None:
    """
    Convert effective relevance sizes into a normalized float weight array
    (sums to 1.0) and write the result onto each SpaceExecutionContext's
    `normalized_weight` field.

    Weight logic:
        1. Map each non-anchor space's effective_relevance → raw numeric weight
           using RELEVANCE_RAW_WEIGHTS. Inactive spaces get 0.0.
        2. Anchor's raw weight = anchor_mean_fraction × mean(active non-anchor weights).
           If no non-anchor spaces are active, anchor gets a raw weight of 1.0.
        3. Normalize all 8 raw weights to sum to 1.0.

    This mutates the contexts list in place (sets normalized_weight on each).

    Args:
        contexts: List of 8 SpaceExecutionContext objects from Stage 1.
        anchor_mean_fraction: Scaling factor for anchor relative to the mean
                              of active non-anchor spaces. Default 0.8.
    """
    if not isinstance(contexts, list):
        raise TypeError(f"contexts must be a list, got {type(contexts).__name__}")
    if not contexts:
        raise ValueError("contexts must not be empty")
    if isinstance(anchor_mean_fraction, bool) or not isinstance(anchor_mean_fraction, Real):
        raise TypeError(
            f"anchor_mean_fraction must be a real number, got {type(anchor_mean_fraction).__name__}"
        )
    if not math.isfinite(anchor_mean_fraction) or anchor_mean_fraction <= 0.0:
        raise ValueError(f"anchor_mean_fraction must be finite and > 0, got {anchor_mean_fraction}")

    anchor_index: Optional[int] = None

    for i, ctx in enumerate(contexts):
        if not isinstance(ctx, SpaceExecutionContext):
            raise TypeError(f"contexts[{i}] must be SpaceExecutionContext, got {type(ctx).__name__}")
        if ctx.name == VectorName.ANCHOR:
            if anchor_index is not None:
                raise ValueError("contexts must contain exactly one anchor context, found >1")
            anchor_index = i
            if not ctx.is_active:
                raise ValueError("anchor context must be active")
            if ctx.effective_relevance is not None:
                raise ValueError("anchor context must have effective_relevance=None")
            continue

        if not ctx.is_active:
            continue

        if ctx.effective_relevance is None:
            raise ValueError(f"active context '{ctx.name.value}' must define effective_relevance")
        if not isinstance(ctx.effective_relevance, RelevanceSize):
            raise ValueError(
                f"active context '{ctx.name.value}' has invalid effective_relevance type: "
                f"{type(ctx.effective_relevance).__name__}"
            )
        if ctx.effective_relevance == RelevanceSize.NOT_RELEVANT:
            raise ValueError(
                f"active context '{ctx.name.value}' cannot have effective_relevance=NOT_RELEVANT"
            )

    if anchor_index is None:
        raise ValueError("contexts must contain exactly one anchor context, found 0")

    # --- Step 1: Compute raw weights for all non-anchor spaces ---
    # Collect (index, raw_weight) pairs so we can write back after normalization.
    raw_weights: list[float] = []

    for i, ctx in enumerate(contexts):
        if ctx.name == VectorName.ANCHOR:
            # Placeholder — anchor's raw weight is computed in step 2.
            raw_weights.append(0.0)
        elif not ctx.is_active:
            # Inactive space: didn't run any search, gets zero weight.
            # This covers the NOT_RELEVANT + no subquery case.
            raw_weights.append(0.0)
        else:
            # Active non-anchor space: look up the effective relevance.
            raw_weights.append(RELEVANCE_RAW_WEIGHTS[ctx.effective_relevance])

    # --- Step 2: Compute anchor's raw weight ---
    # Anchor should be "slightly below average" of active non-anchor weights.
    # This keeps it useful for broad recall without drowning out the specialized
    # spaces that directly match query intent.

    # Gather only the non-zero non-anchor weights.
    active_non_anchor = [
        raw_weights[i]
        for i in range(len(contexts))
        if i != anchor_index and raw_weights[i] > 0.0
    ]

    if not active_non_anchor:
        # Edge case: only anchor is active (every other space is not_relevant
        # with no subqueries). Anchor carries the entire weight.
        raw_weights[anchor_index] = 1.0
    else:
        # Normal case: anchor = fraction of the mean active non-anchor weight.
        mean_active = sum(active_non_anchor) / len(active_non_anchor)
        raw_weights[anchor_index] = mean_active * anchor_mean_fraction

    # --- Step 3: Normalize to sum to 1.0 ---
    total = sum(raw_weights)

    # Safety: total should never be 0 because anchor is always active and gets
    # a positive raw weight. But guard against it to avoid division by zero.
    if total == 0.0:
        # Degenerate case — shouldn't happen in practice. Give anchor everything.
        for ctx in contexts:
            ctx.normalized_weight = 1.0 if ctx.name == VectorName.ANCHOR else 0.0
        return

    for i, ctx in enumerate(contexts):
        ctx.normalized_weight = raw_weights[i] / total


# ===========================================================================
# Stage 5: Weighted sum → final vector score per candidate
# ===========================================================================


def compute_final_scores(
    all_candidate_ids: frozenset[int],
    active_contexts: list[SpaceExecutionContext],
    per_space_normalized: dict[VectorName, dict[int, float]],
) -> dict[int, float]:
    """
    Combine normalized per-space scores with normalized weights to produce
    a single [0, 1] final vector score for each candidate.

    Formula:
        final(movie) = Σ weight[space] × normalized_score[space][movie]

    The result is guaranteed ∈ [0, 1] because:
        - Each normalized_score ∈ [0, 1]  (Stage 3)
        - Weights ∈ [0, 1] and sum to 1.0  (Stage 4)
        - Weighted sum of [0,1] values with unit-sum weights ∈ [0, 1]

    Candidates that didn't appear in any space's results get 0.0.

    Args:
        all_candidate_ids: Complete set of movie_ids from the input candidates
                           dict. Every one of these gets a final score entry.
        active_contexts:   Only the SpaceExecutionContexts where is_active=True.
                           Inactive spaces are excluded so we don't iterate
                           empty dicts.
        per_space_normalized: space_name → sparse dict of movie_id → normalized
                              score. Only contains entries for active spaces.

    Returns:
        Dict of movie_id → final vector score for every candidate.
    """
    if not isinstance(all_candidate_ids, (set, frozenset)):
        raise TypeError(
            "all_candidate_ids must be a set[int] or frozenset[int], "
            f"got {type(all_candidate_ids).__name__}"
        )
    for movie_id in all_candidate_ids:
        if isinstance(movie_id, bool) or not isinstance(movie_id, int):
            raise TypeError(
                "all_candidate_ids must contain only int movie_ids, "
                f"got {type(movie_id).__name__}"
            )

    if not isinstance(active_contexts, list):
        raise TypeError(
            f"active_contexts must be list[SpaceExecutionContext], got {type(active_contexts).__name__}"
        )
    if not isinstance(per_space_normalized, dict):
        raise TypeError(
            f"per_space_normalized must be dict[VectorName, dict[int, float]], got {type(per_space_normalized).__name__}"
        )

    # Degenerate input contract: if there are no candidates, there must also
    # be no active contexts and no per-space scores.
    if not all_candidate_ids:
        if active_contexts:
            raise ValueError("active_contexts must be empty when all_candidate_ids is empty")
        if per_space_normalized:
            raise ValueError("per_space_normalized must be empty when all_candidate_ids is empty")
        return {}

    if not active_contexts:
        raise ValueError("active_contexts must not be empty when all_candidate_ids is non-empty")

    active_space_names: set[VectorName] = set()
    weight_sum = 0.0

    for i, ctx in enumerate(active_contexts):
        if not isinstance(ctx, SpaceExecutionContext):
            raise TypeError(
                f"active_contexts[{i}] must be SpaceExecutionContext, got {type(ctx).__name__}"
            )
        if not isinstance(ctx.name, VectorName):
            raise TypeError(
                f"active_contexts[{i}].name must be VectorName, got {type(ctx.name).__name__}"
            )
        if not ctx.is_active:
            raise ValueError(f"active_contexts[{i}] ('{ctx.name.value}') must be active")
        if ctx.name in active_space_names:
            raise ValueError(f"duplicate active context for space '{ctx.name.value}'")
        active_space_names.add(ctx.name)

        weight = ctx.normalized_weight
        if isinstance(weight, bool) or not isinstance(weight, Real):
            raise TypeError(
                f"normalized_weight for active space '{ctx.name.value}' must be a real number, "
                f"got {type(weight).__name__}"
            )
        weight_f = float(weight)
        if not math.isfinite(weight_f):
            raise ValueError(
                f"normalized_weight for active space '{ctx.name.value}' must be finite, got {weight_f}"
            )
        if weight_f < 0.0 or weight_f > 1.0:
            raise ValueError(
                f"normalized_weight for active space '{ctx.name.value}' must be within [0.0, 1.0], got {weight_f}"
            )
        weight_sum += weight_f

    per_space_names: set[VectorName] = set()
    for space_name, scores_by_movie in per_space_normalized.items():
        if not isinstance(space_name, VectorName):
            raise TypeError(
                f"per_space_normalized keys must be VectorName, got {type(space_name).__name__}"
            )
        if not isinstance(scores_by_movie, dict):
            raise TypeError(
                f"per_space_normalized['{space_name.value}'] must be dict[int, float], "
                f"got {type(scores_by_movie).__name__}"
            )
        per_space_names.add(space_name)

        for movie_id, norm_score in scores_by_movie.items():
            if isinstance(movie_id, bool) or not isinstance(movie_id, int):
                raise TypeError(
                    f"movie_id in per_space_normalized['{space_name.value}'] must be int, "
                    f"got {type(movie_id).__name__}"
                )
            if movie_id not in all_candidate_ids:
                raise ValueError(
                    f"per_space_normalized['{space_name.value}'] has unknown movie_id={movie_id}"
                )
            _validate_numeric_score(
                norm_score,
                label=(
                    "normalized score for "
                    f"space='{space_name.value}', movie_id={movie_id}"
                ),
            )

    if per_space_names != active_space_names:
        missing = sorted((active_space_names - per_space_names), key=lambda s: s.value)
        extra = sorted((per_space_names - active_space_names), key=lambda s: s.value)
        details: list[str] = []
        if missing:
            details.append(
                "missing spaces: " + ", ".join(space.value for space in missing)
            )
        if extra:
            details.append(
                "extra spaces: " + ", ".join(space.value for space in extra)
            )
        raise ValueError(
            "per_space_normalized keys must exactly match active context spaces ("
            + "; ".join(details)
            + ")"
        )

    if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=WEIGHT_SUM_ABS_TOL):
        raise ValueError(
            f"active context weights must sum to 1.0 within ±{WEIGHT_SUM_ABS_TOL}, got {weight_sum}"
        )

    # Initialize all candidates to 0.0. Candidates that appear in no space's
    # results will remain at 0.0, which is correct — they have no vector
    # evidence of relevance.
    final_scores: dict[int, float] = dict.fromkeys(all_candidate_ids, 0.0)

    # Accumulate weighted contributions from each active space.
    # We iterate only the sparse normalized dicts (non-zero entries), so
    # candidates with normalized=0 in a space are skipped with no work.
    for ctx in active_contexts:
        weight = float(ctx.normalized_weight)

        # Skip spaces with zero weight (shouldn't happen for active spaces,
        # but guard against floating-point edge cases).
        if weight <= 0.0:
            continue

        space_scores = per_space_normalized[ctx.name]

        for movie_id, norm_score in space_scores.items():
            # norm_score is guaranteed > 0 (sparse dict from Stage 3).
            # Accumulate the weighted contribution.
            final_scores[movie_id] += weight * float(norm_score)

    for movie_id, final_score in final_scores.items():
        if not math.isfinite(final_score):
            raise ValueError(f"final score for movie_id={movie_id} is non-finite: {final_score}")
        if final_score < -SCORE_BOUNDARY_TOL or final_score > 1.0 + SCORE_BOUNDARY_TOL:
            raise ValueError(
                f"final score for movie_id={movie_id} must be within [0.0, 1.0], got {final_score}"
            )
        # Correct tiny floating-point drift at the boundaries.
        if final_score < 0.0:
            final_scores[movie_id] = 0.0
        elif final_score > 1.0:
            final_scores[movie_id] = 1.0

    return final_scores


# ===========================================================================
# Orchestrator: ties all stages together
# ===========================================================================


def calculate_vector_scores(
    vector_search_result: VectorSearchResult,
) -> VectorScoringResult:
    """
    Main entry point. Converts raw per-collection cosine similarities into
    a single [0, 1] final vector score for every candidate movie.

    This is called once per search request, after all Qdrant searches have
    completed and candidates have been merged into the candidate map.

    Args:
        vector_search_result: VectorSearchResult containing:
            - candidates:        movie_id → CandidateVectorScores with raw cosine
                                similarities from Qdrant. A score of 0.0 means the
                                candidate was not in the top-N for that search.
            - vector_weights:    RelevanceSize per non-anchor space, from query
                                understanding.
            - vector_subqueries: Optional subquery text per non-anchor space, from
                                query understanding.

    Returns:
        VectorScoringResult containing:
            - final_scores: movie_id → [0, 1] final vector score
            - space_contexts: debug info on execution flags and weights
            - per_space_normalized: debug info on per-space normalized scores
    """
    candidates = vector_search_result.candidates
    vector_weights = vector_search_result.vector_weights
    vector_subqueries = vector_search_result.vector_subqueries
    # --- Fast path: no candidates ---
    if not candidates:
        contexts = build_space_execution_contexts(vector_weights, vector_subqueries)
        compute_normalized_weights(contexts)
        return VectorScoringResult(
            final_scores={},
            space_contexts=contexts,
            per_space_normalized={},
        )

    # --- Stage 1: Determine execution context for each space ---
    # Establishes which searches ran and what the effective relevance is.
    # This is a pure function of the query-understanding output.
    contexts = build_space_execution_contexts(vector_weights, vector_subqueries)

    # --- Stage 4: Compute normalized weights ---
    # We do this before stages 2-3 because the weights don't depend on
    # candidate scores — only on the effective relevances from Stage 1.
    # Computing them first lets us skip blend + normalize for spaces with
    # zero weight, avoiding unnecessary work in stages 2-3.
    compute_normalized_weights(contexts)

    # Build a lookup from space name → (config, context) for the loop below.
    config_by_name: dict[VectorName, _SpaceConfig] = {c.name: c for c in SPACE_CONFIGS}

    # Identify contexts that are both active (at least one search ran) and
    # have a positive weight. An active space with zero weight would produce
    # normalized scores that contribute nothing to the final sum.
    active_contexts: list[SpaceExecutionContext] = [
        ctx for ctx in contexts if ctx.is_active and ctx.normalized_weight > 0.0
    ]

    # --- Stages 2 + 3: Blend and normalize per active space ---
    per_space_normalized: dict[VectorName, dict[int, float]] = {}

    for ctx in active_contexts:
        config = config_by_name[ctx.name]

        # Stage 2: Blend original + subquery scores for this space.
        # Returns a sparse dict (only candidates with blended > 0).
        blended = blend_space_scores(candidates, config, ctx)

        # Stage 3: Normalize blended scores using exponential decay.
        # Returns a sparse dict (same keys as blended, values ∈ (0, 1]).
        normalized = normalize_blended_scores(blended)

        per_space_normalized[ctx.name] = normalized

    # --- Stage 5: Weighted sum across all active spaces ---
    all_candidate_ids = frozenset(candidates.keys())
    final_scores = compute_final_scores(
        all_candidate_ids, active_contexts, per_space_normalized,
    )

    return VectorScoringResult(
        final_scores=final_scores,
        space_contexts=contexts,
        per_space_normalized=per_space_normalized,
    )
