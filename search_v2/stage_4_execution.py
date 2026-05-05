# Search V2 — Stage 4: execute generated endpoint specs and produce
# per-branch ranked candidate lists.
#
# Driven by the recursive granularity rule (search_method_deterministic_logic.md
# §3 / §5 / §6 / §7 / §8):
#
#   query
#     └── traits             (across-trait merges at branch level)
#          └── categories    (intra-trait merges at trait level)
#               └── calls    (intra-category merges at category level)
#
# At every level a node is either "candidate-generating" (≥1 positive-
# polarity CANDIDATE_GENERATOR call somewhere in its subtree) or
# "pure-reranker." Cand-gen nodes execute generators in isolation,
# then run rerankers within their own scope; pure-reranker nodes
# defer one level up. Composites use nested equal-weight averaging.
#
# Negative-polarity traits are pure-reranker for orchestration but
# score the branch pool with a different shape (multiplicative AND
# over would-be-generator calls, gated against the reranker average).
# See _score_negative_trait.
#
# Auxiliary specs (NEUTRAL_SEED, MEDIA_TYPE shorts) are applied at
# branch level after cand-gen traits run: NEUTRAL_SEED adds IDs to a
# branch_pool that is otherwise empty (the upstream fallback only
# emits it when no cand-gen exists pipeline-wide); MEDIA_TYPE shorts
# subtracts its IDs from branch_pool unconditionally. Neither
# contributes a trait_score.

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from db.postgres import fetch_neutral_reranker_seed_ids
from db.qdrant import qdrant_client
from schemas.endpoint_result import EndpointResult
from schemas.enums import EndpointRoute, OperationType, Polarity
from schemas.trait_category import CategoryName
from search_v2.endpoint_fetching.category_handlers.generated_endpoint_spec import (
    GeneratedEndpointSpec,
)
from search_v2.endpoint_fetching.category_handlers.handler import (
    determine_operation_type,
)
from search_v2.endpoint_fetching.endpoint_executors import (
    build_endpoint_coroutine,
)

if TYPE_CHECKING:
    from search_v2.full_pipeline_orchestrator import (
        BranchKind,
        CategoryCallWithEndpoints,
        Step2BranchResult,
        TraitWithEndpoints,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Corpus size used for rarity tier lookup. The doc (§7) targets
# ~150K movies; tune here if the corpus shifts.
CORPUS_SIZE: int = 150_000

# Floor used to detect "elbow-1.0" scores for semantic-promoted rarity.
# Per-call elbow normalization clamps top-tier scores to exactly 1.0,
# so an exact-equality test would suffice — using a small epsilon
# guards against float-comparison surprises.
_ELBOW_ONE_EPSILON: float = 1e-6

# Per-attempt executor timeout. Matches the existing handler-side
# value so the executor surface stays consistent with already-tuned
# upstream behavior.
EXECUTOR_TIMEOUT_SECONDS: float = 25.0


# ---------------------------------------------------------------------------
# Commitment / rarity tables (per §7)
# ---------------------------------------------------------------------------

_COMMITMENT_MULTIPLIERS: dict[str, float] = {
    "required": 3.0,
    "elevated": 1.75,
    "neutral": 1.0,
    "supporting": 0.6,
    "diminished": 0.35,
}


def _commitment_multiplier(commitment: str) -> float:
    return _COMMITMENT_MULTIPLIERS[commitment]


def _rarity_factor(match_count: int, corpus_size: int = CORPUS_SIZE) -> float:
    """Discrete tiered rarity factor over (match_count / corpus_size).

    Empty match counts collapse to the most-common tier — a trait
    with zero matches gets `rarity = 0.5`, which keeps it from being
    accidentally amplified. (The trait will already contribute 0 to
    most movies via opportunity cost, so this is mostly defensive.)
    """
    if corpus_size <= 0:
        return 1.0
    fraction = match_count / corpus_size
    if fraction < 0.001:
        return 1.5
    if fraction < 0.01:
        return 1.2
    if fraction < 0.10:
        return 1.0
    if fraction < 0.30:
        return 0.75
    return 0.5


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class BranchRankedResults:
    """One Step-2 branch's executed + ranked output.

    `ranked` is sorted descending by final_score per §9. Ties are
    broken by the order in which candidates entered the branch_pool,
    which is deterministic given a fixed input.

    `branch_error` mirrors the upstream Step2BranchResult.branch_error;
    when set, `ranked` is empty.
    """

    kind: "BranchKind"
    query: str
    ui_label: str
    ranked: list[tuple[int, float]] = field(default_factory=list)
    branch_error: str | None = None


# ---------------------------------------------------------------------------
# Internal execution-state dataclasses (not exposed)
# ---------------------------------------------------------------------------


@dataclass
class _CallExecution:
    """One executed call. Carries both scores and the spec so callers
    can read provenance (route, operation_type, was_promoted) for
    rarity counting on semantic-promoted traits.
    """

    spec: GeneratedEndpointSpec
    scores: dict[int, float]


@dataclass
class _CategoryExecution:
    """One executed category. `pool` is the union of generator
    matches in the category (empty for pure-reranker categories,
    which defer to the trait level). `composite_scores` carries the
    equal-weight average across the category's calls, defined only
    over `pool`. `calls` is preserved for rarity bookkeeping.
    """

    is_cand_gen: bool
    pool: set[int]
    composite_scores: dict[int, float]
    calls: list[_CallExecution]


@dataclass
class _TraitExecution:
    """One executed cand-gen trait. `pool` is the union across cand-
    gen categories. `trait_scores` is the equal-weight average across
    all categories in the trait, restricted to `pool`. `match_count`
    feeds rarity (semantic-promoted vs regular finder).
    """

    trait: "TraitWithEndpoints"
    pool: set[int]
    trait_scores: dict[int, float]
    match_count: int


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def execute_branches(
    branches: list["Step2BranchResult"],
    auxiliary_specs: list[GeneratedEndpointSpec],
) -> list[BranchRankedResults]:
    """Run every branch in parallel and return ranked candidate lists.

    Each branch is independent: it executes its own cand-gen traits,
    builds its own branch_pool, applies the auxiliary specs to its
    own pool, then reranks with its own pure-reranker / negative
    traits. Errors in one branch do not affect siblings.
    """
    if not branches:
        return []
    return list(
        await asyncio.gather(
            *(_run_branch(branch, auxiliary_specs) for branch in branches)
        )
    )


# ---------------------------------------------------------------------------
# Branch-level execution
# ---------------------------------------------------------------------------


async def _run_branch(
    branch: "Step2BranchResult",
    auxiliary_specs: list[GeneratedEndpointSpec],
) -> BranchRankedResults:
    """Execute one Step-2 branch end-to-end."""
    if branch.branch_error is not None:
        # Upstream already failed for this branch; surface the error
        # and produce an empty ranked list. No execution attempted.
        return BranchRankedResults(
            kind=branch.kind,
            query=branch.query,
            ui_label=branch.ui_label,
            ranked=[],
            branch_error=branch.branch_error,
        )

    branch_start = time.perf_counter()

    # Partition traits per §4. Negative-polarity traits are always
    # pure-reranker for orchestration even when they carry would-be-
    # generator calls (handled inside _score_negative_trait).
    cand_gen_traits = [
        t for t in branch.traits if _trait_is_cand_gen(t)
    ]
    pure_rer_traits = [
        t for t in branch.traits if not _trait_is_cand_gen(t)
    ]

    # Step C — cand-gen traits in parallel, each builds its own pool
    # in isolation. Each result is a _TraitExecution (or None when
    # the trait failed upstream / produced nothing useful).
    cand_gen_executions = await asyncio.gather(
        *(_run_cand_gen_trait(t) for t in cand_gen_traits)
    )
    cand_gen_executions = [ex for ex in cand_gen_executions if ex is not None]

    # Step D — union pools.
    branch_pool: set[int] = set()
    for ex in cand_gen_executions:
        branch_pool.update(ex.pool)

    # Apply auxiliary specs (neutral seed first, shorts subtraction
    # second). Both are no-ops when their preconditions don't hold.
    branch_pool = await _apply_auxiliary(branch_pool, auxiliary_specs)

    # If the branch has nothing in its pool by now, no rerankers can
    # contribute. Return empty deterministically — no fallback
    # promotion happens at this layer; that already ran upstream.
    if not branch_pool:
        logger.info(
            "branch %s: empty pool after cand-gen + auxiliary; returning empty",
            branch.kind,
        )
        return BranchRankedResults(
            kind=branch.kind,
            query=branch.query,
            ui_label=branch.ui_label,
            ranked=[],
        )

    # Step E — pure-reranker / negative traits score the branch_pool.
    pure_rer_scores_per_trait = await asyncio.gather(
        *(
            _run_pure_reranker_trait(t, branch_pool)
            for t in pure_rer_traits
        )
    )

    # Step F — final weighted sum per §9.
    ranked = _finalize_scores(
        branch_pool=branch_pool,
        cand_gen_executions=cand_gen_executions,
        pure_rer_traits=pure_rer_traits,
        pure_rer_scores_per_trait=pure_rer_scores_per_trait,
    )

    logger.info(
        "branch %s: ranked %d candidates in %.2fs",
        branch.kind,
        len(ranked),
        time.perf_counter() - branch_start,
    )
    return BranchRankedResults(
        kind=branch.kind,
        query=branch.query,
        ui_label=branch.ui_label,
        ranked=ranked,
    )


# ---------------------------------------------------------------------------
# Trait-level classification + execution
# ---------------------------------------------------------------------------


def _trait_is_cand_gen(trait: "TraitWithEndpoints") -> bool:
    """A trait is candidate-generating iff it is positive-polarity AND
    has ≥1 CANDIDATE_GENERATOR spec in any of its category_calls.
    Negative traits never generate candidates (§4 / §8).
    """
    if trait.polarity is Polarity.NEGATIVE:
        return False
    for cc in trait.category_calls:
        for spec in cc.generated_specs:
            if spec.operation_type is OperationType.CANDIDATE_GENERATOR:
                return True
    return False


def _category_is_cand_gen(cc: "CategoryCallWithEndpoints") -> bool:
    """A category contributes generators iff ≥1 of its specs is a
    CANDIDATE_GENERATOR. (Polarity is checked at the trait level —
    cand-gen categories are reached only inside positive traits.)
    """
    return any(
        spec.operation_type is OperationType.CANDIDATE_GENERATOR
        for spec in cc.generated_specs
    )


async def _run_cand_gen_trait(
    trait: "TraitWithEndpoints",
) -> _TraitExecution | None:
    """Execute a positive cand-gen trait fully (categories then calls).

    Returns None when the trait surfaced a Step-3 error or has zero
    successfully-executed calls — both collapse to "trait contributes
    nothing." Returning None lets the caller drop it cleanly without
    needing per-trait error fields on the execution result.
    """
    if trait.step_3_error is not None:
        return None

    # Skip silently empty trait shells (e.g. all CategoryCalls had
    # handler errors and emitted nothing).
    if not trait.category_calls:
        return None

    # Run every category in parallel. Cand-gen categories build
    # their own pools; pure-reranker categories defer (we run them
    # again at the trait level once the trait pool is known).
    category_kinds = [_category_is_cand_gen(cc) for cc in trait.category_calls]
    cand_gen_categories = [
        cc for cc, is_gen in zip(trait.category_calls, category_kinds) if is_gen
    ]
    pure_rer_categories = [
        cc for cc, is_gen in zip(trait.category_calls, category_kinds) if not is_gen
    ]

    cat_executions = await asyncio.gather(
        *(_run_cand_gen_category(cc) for cc in cand_gen_categories)
    )

    # Trait pool = union of cand-gen categories' pools.
    trait_pool: set[int] = set()
    for cat_ex in cat_executions:
        trait_pool.update(cat_ex.pool)

    if not trait_pool:
        # All cand-gen categories produced nothing → trait contributes
        # nothing. Pure-reranker categories within the trait do not
        # run (§11: empty propagates upward).
        return None

    # Run pure-reranker categories restricted to the trait pool.
    rer_cat_executions = await asyncio.gather(
        *(
            _run_pure_reranker_category(cc, trait_pool)
            for cc in pure_rer_categories
        )
    )

    # Trait composite: equal-weight average across all categories
    # (cand-gen + pure-reranker), per the user's nested-averaging
    # design choice. Each category contributes its composite_scores
    # as a single [0,1] value per movie (0 if not present).
    all_cat_executions = list(cat_executions) + list(rer_cat_executions)
    num_categories = len(all_cat_executions)
    trait_scores: dict[int, float] = {}
    for movie_id in trait_pool:
        s = 0.0
        for cat_ex in all_cat_executions:
            s += cat_ex.composite_scores.get(movie_id, 0.0)
        trait_scores[movie_id] = s / num_categories if num_categories else 0.0

    match_count = _match_count_for_rarity(cat_executions)

    return _TraitExecution(
        trait=trait,
        pool=trait_pool,
        trait_scores=trait_scores,
        match_count=match_count,
    )


def _match_count_for_rarity(
    cand_gen_cat_executions: Iterable[_CategoryExecution],
) -> int:
    """Count movies for the rarity tier lookup.

    Per §7 + user clarification:
      - Promoted (semantic-promoted) generator calls: only post-elbow
        1.0 movies count. Take the union across the trait's promoted
        generator calls.
      - Regular finder generator calls: every matched candidate counts
        (i.e. every member of the call's score map).
      - When a trait mixes promoted + regular generator calls, both
        rules apply — we union both sets.

    Returns the size of the union across all generator calls in all
    cand-gen categories within the trait. By construction this set is
    a subset of the trait's pool (the pool is built from the same
    score-map keys), so no clamping is needed.
    """
    unioned: set[int] = set()
    for cat_ex in cand_gen_cat_executions:
        for call in cat_ex.calls:
            if call.spec.operation_type is not OperationType.CANDIDATE_GENERATOR:
                continue
            if call.spec.was_promoted:
                # Semantic-promoted: only 1.0-scoring movies count.
                unioned.update(
                    mid
                    for mid, sc in call.scores.items()
                    if sc >= 1.0 - _ELBOW_ONE_EPSILON
                )
            else:
                # Regular finder: all matched candidates count.
                unioned.update(call.scores.keys())
    return len(unioned)


# ---------------------------------------------------------------------------
# Category-level execution
# ---------------------------------------------------------------------------


async def _run_cand_gen_category(
    cc: "CategoryCallWithEndpoints",
) -> _CategoryExecution:
    """Execute one positive cand-gen category.

    Generators run with restrict=None and produce the category pool
    (union of their matched IDs). Rerankers in the same category run
    with restrict=category_pool — their meaning is scoped to this
    category's carve, per §3.

    Composite is the equal-weight average across all calls in the
    category (generator + reranker), defined only over the category
    pool. Movies outside the pool are not scored at this level.
    """
    if cc.handler_error is not None:
        return _CategoryExecution(
            is_cand_gen=True,
            pool=set(),
            composite_scores={},
            calls=[],
        )

    gen_specs: list[GeneratedEndpointSpec] = []
    rer_specs: list[GeneratedEndpointSpec] = []
    for spec in cc.generated_specs:
        if spec.operation_type is OperationType.CANDIDATE_GENERATOR:
            gen_specs.append(spec)
        else:
            rer_specs.append(spec)

    # Generators in parallel, full corpus. None on failure → treat
    # as empty for positive-trait paths (the call simply doesn't
    # contribute candidates).
    raw_gen_results = await asyncio.gather(
        *(_dispatch_call(spec, restrict=None) for spec in gen_specs)
    )
    gen_results = [scores or {} for scores in raw_gen_results]

    category_pool: set[int] = set()
    for scores in gen_results:
        category_pool.update(scores.keys())

    if not category_pool:
        # Empty pool → connected rerankers do not run (§11).
        return _CategoryExecution(
            is_cand_gen=True,
            pool=set(),
            composite_scores={},
            calls=[
                _CallExecution(spec=spec, scores=scores)
                for spec, scores in zip(gen_specs, gen_results)
            ],
        )

    # Rerankers within this category, restricted to its pool. None on
    # failure → treat as empty for the per-category equal-weight average.
    raw_rer_results = await asyncio.gather(
        *(_dispatch_call(spec, restrict=category_pool) for spec in rer_specs)
    )
    rer_results = [scores or {} for scores in raw_rer_results]

    all_specs = gen_specs + rer_specs
    all_scores = list(gen_results) + list(rer_results)
    num_calls = len(all_specs)

    composite: dict[int, float] = {}
    for movie_id in category_pool:
        s = 0.0
        for scores in all_scores:
            s += scores.get(movie_id, 0.0)
        composite[movie_id] = s / num_calls if num_calls else 0.0

    calls = [
        _CallExecution(spec=spec, scores=scores)
        for spec, scores in zip(all_specs, all_scores)
    ]

    return _CategoryExecution(
        is_cand_gen=True,
        pool=category_pool,
        composite_scores=composite,
        calls=calls,
    )


async def _run_pure_reranker_category(
    cc: "CategoryCallWithEndpoints",
    pool: set[int],
) -> _CategoryExecution:
    """Run a pure-reranker category against an externally-defined pool.

    Used when a category sits inside a cand-gen trait but contributes
    no generators of its own — the trait's pool (union of cand-gen
    categories) becomes its scope.
    """
    if cc.handler_error is not None or not pool:
        return _CategoryExecution(
            is_cand_gen=False,
            pool=set(),
            composite_scores={},
            calls=[],
        )

    raw_rer_results = await asyncio.gather(
        *(_dispatch_call(spec, restrict=pool) for spec in cc.generated_specs)
    )
    rer_results = [scores or {} for scores in raw_rer_results]

    num_calls = len(cc.generated_specs)
    composite: dict[int, float] = {}
    for movie_id in pool:
        s = 0.0
        for scores in rer_results:
            s += scores.get(movie_id, 0.0)
        composite[movie_id] = s / num_calls if num_calls else 0.0

    calls = [
        _CallExecution(spec=spec, scores=scores)
        for spec, scores in zip(cc.generated_specs, rer_results)
    ]
    return _CategoryExecution(
        is_cand_gen=False,
        pool=set(pool),
        composite_scores=composite,
        calls=calls,
    )


# ---------------------------------------------------------------------------
# Pure-reranker / negative traits — score the branch pool
# ---------------------------------------------------------------------------


async def _run_pure_reranker_trait(
    trait: "TraitWithEndpoints",
    branch_pool: set[int],
) -> dict[int, float]:
    """Score a pure-reranker trait against the branch_pool.

    Positive pure-reranker traits use equal-weight averaging across
    all successfully-executed calls. Negative traits use multiplicative
    AND over would-be-generator calls, gated against the reranker
    average — see _score_negative_trait.
    """
    if trait.step_3_error is not None or not branch_pool:
        return {mid: 0.0 for mid in branch_pool}

    # Flatten every call across every category in the trait, carrying
    # the originating category alongside the spec. For negative traits
    # the partition between "would-be-generator" and "would-be-reranker"
    # is not readable off `spec.operation_type` (handler.determine_operation_type
    # short-circuits every negative-polarity call to POOL_RERANKER), so
    # we re-derive what each call would have been in positive polarity
    # using the same routing function.
    paired: list[tuple[CategoryName, GeneratedEndpointSpec]] = []
    for cc in trait.category_calls:
        if cc.handler_error is not None:
            continue
        for spec in cc.generated_specs:
            paired.append((cc.category, spec))
    if not paired:
        return {mid: 0.0 for mid in branch_pool}

    # Every call runs restricted to the branch_pool. Negative-polarity
    # specs are POOL_RERANKER (per determine_operation_type), so the
    # dispatcher passes the restrict through and the executor returns
    # one score per pool ID. None on failure → call is dropped from
    # downstream aggregation rather than counted as confirmed-zero.
    call_results = await asyncio.gather(
        *(_dispatch_call(spec, restrict=branch_pool) for _, spec in paired)
    )

    if trait.polarity is Polarity.NEGATIVE:
        return _score_negative_trait(paired, call_results, branch_pool)

    # Positive pure-reranker trait: equal-weight average across calls
    # that didn't fail. A failed call is dropped from the denominator
    # rather than treated as a 0-everywhere contributor — that keeps a
    # flaky endpoint from silently dragging the trait score down.
    succeeded = [scores for scores in call_results if scores is not None]
    if not succeeded:
        return {mid: 0.0 for mid in branch_pool}
    denom = len(succeeded)
    out: dict[int, float] = {}
    for movie_id in branch_pool:
        s = sum(scores.get(movie_id, 0.0) for scores in succeeded)
        out[movie_id] = s / denom
    return out


def _score_negative_trait(
    paired: list[tuple[CategoryName, GeneratedEndpointSpec]],
    call_results: list[dict[int, float] | None],
    branch_pool: set[int],
) -> dict[int, float]:
    """Negative-trait inner score in [0, 1] (violation strength).

    Partition by what each call WOULD HAVE BEEN in positive polarity —
    not by its current operation_type, which is uniformly POOL_RERANKER
    for negatives (handler.determine_operation_type forces that). We
    re-derive the would-be type from (category, route).

      G = would-be CANDIDATE_GENERATOR (membership-shaped — discrete
          tag / posting / array-overlap signals)
      R = would-be POOL_RERANKER       (continuous similarity / prior)

    Combine:
      - G and R both present → trait_score = (∏ G) × mean(R)
        (gated multiplication: membership confirms violation, similarity
         calibrates magnitude)
      - G only              → trait_score = ∏ G
      - R only              → trait_score = mean(R)  (graceful fallback)

    Failed calls are dropped from both partitions: a transient endpoint
    failure must not zero out the multiplicative AND or skew the mean.
    Sign is applied at the §9 final-aggregation layer, not here.
    """
    g_scores: list[dict[int, float]] = []
    r_scores: list[dict[int, float]] = []
    for (category, spec), scores in zip(paired, call_results):
        if scores is None:
            # Failed call — drop entirely (option A from review).
            continue
        would_be = determine_operation_type(category, spec.route, Polarity.POSITIVE)
        if would_be is OperationType.CANDIDATE_GENERATOR:
            g_scores.append(scores)
        else:
            r_scores.append(scores)

    out: dict[int, float] = {}
    for movie_id in branch_pool:
        if g_scores:
            gen_violation: float | None = 1.0
            for scores in g_scores:
                gen_violation *= scores.get(movie_id, 0.0)
        else:
            gen_violation = None

        if r_scores:
            rer_avg: float | None = sum(
                scores.get(movie_id, 0.0) for scores in r_scores
            ) / len(r_scores)
        else:
            rer_avg = None

        if gen_violation is not None and rer_avg is not None:
            out[movie_id] = gen_violation * rer_avg
        elif gen_violation is not None:
            out[movie_id] = gen_violation
        elif rer_avg is not None:
            out[movie_id] = rer_avg
        else:
            # All calls failed; nothing to penalize on.
            out[movie_id] = 0.0
    return out


# ---------------------------------------------------------------------------
# Auxiliary spec application (NEUTRAL_SEED + MEDIA_TYPE shorts)
# ---------------------------------------------------------------------------


async def _apply_auxiliary(
    branch_pool: set[int],
    auxiliary_specs: list[GeneratedEndpointSpec],
) -> set[int]:
    """Apply auxiliary specs to the branch_pool in the documented order:

    1. NEUTRAL_SEED — additive seed when branch_pool is empty AND the
       upstream fallback already chose to emit one. (The fallback only
       emits NEUTRAL_SEED when no cand-gen exists pipeline-wide; but we
       gate again here so a branch with successful cand-gen output
       isn't accidentally diluted by the seed.)
    2. MEDIA_TYPE shorts exclusion — subtractive set-difference on
       whatever pool exists at this point. The auxiliary MEDIA_TYPE
       spec is tagged CANDIDATE_GENERATOR upstream so its executor
       returns the SHORT-format movies; we use those IDs as a
       blocklist rather than as a positive contribution.
    """
    if not auxiliary_specs:
        return branch_pool

    neutral_seed_specs = [
        s for s in auxiliary_specs if s.route is EndpointRoute.NEUTRAL_SEED
    ]
    # The shorts exclusion is the only MEDIA_TYPE entry that ever
    # lands in auxiliary_endpoint_specs (per the upstream comment in
    # _build_shorts_exclusion_spec). Filter by route alone — params
    # is wrapped in MediaTypeEndpointParameters with a SHORT format.
    shorts_specs = [
        s for s in auxiliary_specs if s.route is EndpointRoute.MEDIA_TYPE
    ]

    # Step 1: neutral seed.
    if not branch_pool and neutral_seed_specs:
        try:
            seed_ids = await fetch_neutral_reranker_seed_ids()
            branch_pool = set(seed_ids)
            logger.info(
                "auxiliary: seeded branch_pool with %d neutral-seed IDs",
                len(branch_pool),
            )
        except Exception as exc:  # noqa: BLE001 — seed fetch is best-effort
            logger.warning(
                "neutral seed fetch failed; branch will return empty (%r)", exc
            )

    # Step 2: shorts subtraction. Each spec's executor returns a score
    # map of SHORT-format movies; we take the keys as a blocklist.
    # If the call fails (None), skip it — better to let a SHORT slip
    # through than to abandon the whole branch.
    if shorts_specs and branch_pool:
        shorts_results = await asyncio.gather(
            *(_dispatch_call(spec, restrict=None) for spec in shorts_specs)
        )
        shorts_ids: set[int] = set()
        for scores in shorts_results:
            if scores is None:
                continue
            shorts_ids.update(scores.keys())
        if shorts_ids:
            before = len(branch_pool)
            branch_pool = branch_pool - shorts_ids
            logger.info(
                "auxiliary: shorts-excluded %d / %d candidates",
                before - len(branch_pool),
                before,
            )

    return branch_pool


# ---------------------------------------------------------------------------
# Final aggregation (§9)
# ---------------------------------------------------------------------------


def _finalize_scores(
    branch_pool: set[int],
    cand_gen_executions: list[_TraitExecution],
    pure_rer_traits: list["TraitWithEndpoints"],
    pure_rer_scores_per_trait: list[dict[int, float]],
) -> list[tuple[int, float]]:
    """Per-candidate final score per §9: Σ trait_score × trait_weight × sign.

    Cand-gen traits use commitment_mult × rarity_factor. Pure-reranker
    traits use commitment_mult × 1.0. Polarity sign is applied here:
    cand-gen traits are always positive (negative traits never become
    cand-gen), pure-reranker traits may be either.
    """
    # Pre-compute weights so we don't re-derive per movie.
    cand_gen_weights: list[float] = []
    for ex in cand_gen_executions:
        commit = _commitment_multiplier(ex.trait.commitment)
        rarity = _rarity_factor(ex.match_count)
        cand_gen_weights.append(commit * rarity)

    pure_rer_weights: list[float] = []
    pure_rer_signs: list[float] = []
    for trait in pure_rer_traits:
        pure_rer_weights.append(_commitment_multiplier(trait.commitment))
        pure_rer_signs.append(
            -1.0 if trait.polarity is Polarity.NEGATIVE else 1.0
        )

    final: list[tuple[int, float]] = []
    for movie_id in branch_pool:
        score = 0.0
        # Cand-gen contributions (always positive sign). A trait
        # contributes 0 for movies it didn't generate (opportunity
        # cost, §8).
        for ex, weight in zip(cand_gen_executions, cand_gen_weights):
            score += ex.trait_scores.get(movie_id, 0.0) * weight
        # Pure-reranker contributions, signed by polarity.
        for scores, weight, sign in zip(
            pure_rer_scores_per_trait, pure_rer_weights, pure_rer_signs
        ):
            score += scores.get(movie_id, 0.0) * weight * sign
        final.append((movie_id, score))

    final.sort(key=lambda mid_score: mid_score[1], reverse=True)
    return final


# ---------------------------------------------------------------------------
# Single-call dispatch wrapper
# ---------------------------------------------------------------------------


async def _dispatch_call(
    spec: GeneratedEndpointSpec,
    *,
    restrict: set[int] | None,
) -> dict[int, float] | None:
    """Run one GeneratedEndpointSpec through the existing dispatcher.

    Wraps the executor with a per-call timeout and converts the
    EndpointResult into a flat `{movie_id: score}` dict for downstream
    aggregation.

    Returns None to distinguish *failed* calls from *legitimately empty*
    results: a failed call must not contribute a confirmed-zero signal
    to negative-trait multiplicative AND, where one failure would
    otherwise zero out the entire trait's penalty. Positive-trait
    callers fold None into `{}` and absorb the zero contribution into
    their averages; the negative-trait scorer drops failed calls from
    the product entirely.

    NEUTRAL_SEED and TRENDING are not dispatched through this path.
    NEUTRAL_SEED is handled inside _apply_auxiliary; TRENDING has no
    LLM codepath in v2 and is not expected to appear in
    `generated_specs`.
    """
    try:
        coro = build_endpoint_coroutine(
            spec,
            qdrant_client=qdrant_client,
            restrict_to_movie_ids=restrict,
        )
        result: EndpointResult = await asyncio.wait_for(
            coro, timeout=EXECUTOR_TIMEOUT_SECONDS
        )
    except Exception as exc:  # noqa: BLE001 — soft-fail per call
        logger.warning(
            "endpoint call failed (route=%s, op=%s); call dropped (%r)",
            spec.route.value,
            spec.operation_type.value,
            exc,
        )
        return None

    return {sc.movie_id: sc.score for sc in result.scores}
