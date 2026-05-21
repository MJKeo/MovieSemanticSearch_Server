# Search V2 — Stage 4: execute generated endpoint specs and produce
# per-branch ranked candidate lists.
#
# Driven by the 5-phase pipeline from
# search_improvement_planning/rescore_overhaul.md:
#
#   Phase A — LLM generation (steps 0/1/2/3 + handler-LLM, upstream).
#   Phase B — Pool definition: run every positive-polarity candidate-
#             generator call across every trait in parallel; union the
#             match sets; apply shorts subtraction; neutral-seed
#             fallback only when zero generators ran pipeline-wide.
#   Phase C — Reranker pass: run every positive-polarity reranker call
#             across every trait against the finalized union (the load-
#             bearing change vs the prior trait-local-pool design — a
#             reranker now scores every candidate, regardless of which
#             trait its call lives in).
#   Phase D — Per-trait scoring: for each candidate, for each trait,
#             compose per-call scores via the category's combine type
#             (SINGLE / ADDITIVE / ALTERNATIVES / CONSENSUS / NO_OP),
#             then fold across the trait's categories by combine_mode — FRAMINGS
#             takes MAX (alternative homes for one underlying thing);
#             FACETS takes PRODUCT (compound concept whose axes must
#             compound). Trait weight is commitment × rarity (rarity =
#             1.0 unless the trait is pure-generator).
#   Phase E — Branch aggregation: Σ trait_score × weight × sign over
#             positive traits, minus the gate × fuzzy negative trait
#             contributions. Implicit-prior boost is applied later by
#             the orchestrator.
#
# Negative-polarity traits keep the existing three-bin gate × fuzzy
# formula (G_a authoritative gate × noisy-OR over G_e ∪ R). See
# `_score_negative_trait`.
#
# Auxiliary specs (NEUTRAL_SEED, MEDIA_TYPE shorts) are applied in
# Phase B: NEUTRAL_SEED seeds the pool only when no positive generator
# was attempted; MEDIA_TYPE shorts is subtracted whenever the user did
# not explicitly request a media type. Neither contributes a trait_score.

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np

from db.postgres import fetch_neutral_reranker_seed_ids
from db.qdrant import qdrant_client
from schemas.endpoint_result import EndpointResult
from schemas.enums import (
    CategoryCombineType,
    EndpointRoute,
    OperationType,
    Polarity,
    TraitCombineMode,
)
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
# Within-category combine
# ---------------------------------------------------------------------------


def combine_calls(
    combine_type: CategoryCombineType,
    scores: list[float],
) -> float | None:
    """Apply a category's within-category combine rule and return a
    per-category score in [0, 1] — or None for NO_OP, signaling that
    the category should be skipped entirely by the across-category max.

    Empty `scores` means no calls fired for the category (handler
    emitted no specs, or every call failed). For non-NO_OP modes that
    is treated as a 0.0 contribution: the category is fine to include
    in the max, it just doesn't help.

    Modes (per rescore_overhaul.md):
      SINGLE       — one orchestrator-visible call, passthrough.
      ADDITIVE     — multiple calls together complete the picture;
                     product across [0, 1] scores. Strict — any 0 zeros
                     the category.
      ALTERNATIVES — each call is an alternative way of finding the
                     trait; max across calls. Matching any one is
                     sufficient evidence.
      CONSENSUS    — every committed call should weigh in. Soft
                     geometric mean over committed-call scores with
                     `_CONSENSUS_FOLD_FLOOR` lifting any sub-floor
                     value. Prevents one endpoint from over-promoting
                     the category on its own while keeping a single
                     0 from collapsing it. Single-commit cases
                     passthrough (geom mean over one element = that
                     element, modulo the floor).
      NO_OP        — category never fires. Returns None so the across-
                     category max can skip it rather than treating it
                     as a 0.0 framing that drags down the max for
                     traits whose other categories did fire.
    """
    if combine_type is CategoryCombineType.NO_OP:
        return None
    if not scores:
        return 0.0
    if combine_type is CategoryCombineType.SINGLE:
        # Handler is responsible for ensuring SINGLE-combine categories
        # only ever fire one orchestrator-visible call. If we somehow
        # see >1, take the first deterministically and surface the
        # invariant violation in the logs — the upstream handler
        # config is the place to fix it.
        if len(scores) > 1:
            logger.warning(
                "combine_calls(SINGLE) received %d scores; expected 1. "
                "Taking the first deterministically — check the handler "
                "for the SINGLE-combine category that fired multiple "
                "orchestrator-visible calls.",
                len(scores),
            )
        return scores[0]
    if combine_type is CategoryCombineType.ADDITIVE:
        out = 1.0
        for s in scores:
            out *= s
        return out
    if combine_type is CategoryCombineType.ALTERNATIVES:
        return max(scores)
    if combine_type is CategoryCombineType.CONSENSUS:
        # Soft geometric mean over committed-call scores. Floor lifts
        # sub-EPS values so one missing/zero endpoint cannot zero the
        # category, but the lifted EPS still drags the geom_mean well
        # below the high-scorer — exactly the "no single endpoint
        # over-promotes" property the mode exists for.
        n = len(scores)
        product = 1.0
        for s in scores:
            product *= max(s, _CONSENSUS_FOLD_FLOOR)
        return product ** (1.0 / n)
    # Defensive — keep the type checker honest if a new mode lands
    # without a branch above.
    raise ValueError(f"unknown combine_type: {combine_type!r}")


# FACETS fold floor (Phase 7 / rescore_overhaul.md "Soft FACETS fold").
# Each category contribution under FACETS is lifted to at least this
# value before the geometric mean. The floor keeps a single zeroed
# category from collapsing the whole trait while preserving
# multiplicative-compounding semantics — geom_mean of [1.0, EPS] ≈
# `EPS ** 0.5` (≈ 0.316 at EPS=0.1), so a movie strong on one facet
# but missing on another still scores well above 0 but well below 1.
# Tune via the EPS sweep documented in rescore_overhaul.md Phase 7.2.
_FACETS_FOLD_FLOOR: float = 0.1

# CONSENSUS fold floor — within-category analogue of `_FACETS_FOLD_FLOOR`.
# Lifts each committed-call score to at least this value before the
# geometric mean so a single 0-scoring endpoint cannot collapse the
# category, while still dragging the result well below a clean
# multi-endpoint match. Same EPS as FACETS so the within- and across-
# category soft folds share calibration; tune together.
_CONSENSUS_FOLD_FLOOR: float = 0.1


def combine_categories(
    combine_mode: TraitCombineMode,
    category_scores: list[float],
) -> float:
    """Apply a trait's across-category combine rule and return the
    per-trait score in [0, 1].

    Empty `category_scores` means no category fired for the trait
    (every category was NO_OP, every call failed, etc.) — returns 0.0
    in any mode so an empty trait contributes nothing.

    Modes (per V4 plan in search_deepdive.md, with Phase 7 update):
      SOLO     — exactly one surviving category covers the trait
                 cleanly on its own. The single category's score IS
                 the trait_score (passthrough). Extras are trimmed
                 upstream by the orchestrator before retrieval; if
                 multiple scores arrive here the first is used
                 deterministically and the invariant breach is
                 logged.
      FRAMINGS — categories are alternative homes for one underlying
                 thing; matching any one is sufficient evidence. MAX
                 over the category scores. Redundant categories
                 reinforce as alternative routes to the same signal.
      FACETS   — categories cover different axes of a compound
                 concept; ALL facets must contribute for the
                 criterion to be met. Geometric mean over each
                 category's score lifted to a floor of
                 `_FACETS_FOLD_FLOOR`. The floor preserves
                 multiplicative-compounding signal (a movie strong
                 on every facet still scores ≈ 1.0; a movie zero on
                 one facet still scores ≈ floor^(1/n) instead of
                 collapsing the trait), while geometric-mean shape
                 normalizes the product back into [0, 1] independent
                 of category count. Replaces the strict-PRODUCT fold
                 used in the original V4 design.
    """
    if not category_scores:
        return 0.0
    if combine_mode is TraitCombineMode.SOLO:
        # SOLO commits exactly one category; the orchestrator trims
        # any extras emitted under SOLO before retrieval, so by the
        # time scores reach this fold there should be exactly one.
        # Defensive log if more arrive — surfaces an upstream trim
        # gap without corrupting the score.
        if len(category_scores) > 1:
            logger.warning(
                "combine_categories(SOLO) received %d scores; expected 1. "
                "Taking the first deterministically — the orchestrator's "
                "SOLO trim should have dropped the extras before retrieval.",
                len(category_scores),
            )
        return category_scores[0]
    if combine_mode is TraitCombineMode.FRAMINGS:
        return max(category_scores)
    if combine_mode is TraitCombineMode.FACETS:
        # Geometric mean with floor. The floor turns a single zero
        # into a heavy-but-survivable penalty; the geometric mean
        # keeps the result in [0, 1] regardless of category count.
        floored = [
            s if s > _FACETS_FOLD_FLOOR else _FACETS_FOLD_FLOOR
            for s in category_scores
        ]
        product = 1.0
        for s in floored:
            product *= s
        return product ** (1.0 / len(floored))
    # Defensive — keep the type checker honest if a new mode lands
    # without a branch above.
    raise ValueError(f"unknown combine_mode: {combine_mode!r}")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class EndpointScore:
    """One endpoint call's per-candidate score in [0, 1].

    `route` is the endpoint route value (e.g. "vector_search",
    "keyword_search") so the breakdown is human-readable without a
    round-trip through the EndpointRoute enum. `score` is the value
    that fed into the category's combine_calls fold.
    """

    route: str
    score: float


@dataclass
class CategoryScore:
    """One category's combined score under a positive trait.

    For positive traits the trait_score is `max(category_scores)` over
    the trait's non-NO_OP categories (each category's score being the
    `combine_calls` fold over its live calls). Surfacing these per-
    category scores lets callers see WHICH category drove the trait's
    contribution. `combine_type` mirrors the category's combine rule
    (SINGLE / ADDITIVE / ALTERNATIVES / CONSENSUS) so the breakdown is
    self-describing without round-tripping back to the taxonomy.

    `expressions` and `retrieval_intent` mirror the upstream Step-3
    CategoryCall so the breakdown carries the human-language ask that
    drove the category's endpoint generation. `endpoint_scores` lists
    the per-call scores that fed `combine_calls` to produce `score`,
    in the order they were dispatched.
    """

    category_name: str
    combine_type: str
    expressions: list[str]
    retrieval_intent: str
    score: float
    endpoint_scores: list[EndpointScore] = field(default_factory=list)


@dataclass
class TraitContribution:
    """One trait's signed weighted contribution to a candidate's score.

    `surface_text` and `commitment` mirror the upstream Trait so the
    output is self-describing. `contribution` is `trait_score × weight
    × sign` — the same value the §9 sum already accumulates into
    positive_total / negative_total. Negative-polarity traits surface
    as a non-positive value here.

    `trait_score` is the inner score in [0, 1] that fed the
    contribution (before weight and sign). `weight` is the multiplier
    applied (commitment × rarity for positives, commitment for
    negatives). `category_scores` decomposes the inner score into its
    per-category inputs — populated for positive traits, empty for
    negatives (whose three-bin gate × fuzzy formula does not fold
    through per-category scores).

    `scoring_method` names the across-trait fold that produced
    `trait_score` so debug output can surface which combine the
    pipeline picked: `"framings"` (MAX over categories) or
    `"facets"` (PRODUCT over categories) for positive traits;
    `"gate×fuzzy"` for negative traits (whose scoring uses the
    three-bin authoritative-gate × evidential-fuzzy formula and
    ignores combine_mode).
    """

    surface_text: str
    commitment: str
    contribution: float
    trait_score: float = 0.0
    weight: float = 0.0
    scoring_method: str = ""
    category_scores: list[CategoryScore] = field(default_factory=list)


@dataclass
class ScoreBreakdown:
    """Per-candidate decomposition of the final score.

    `positive_total` sums every contribution with a positive sign:
    every positive-polarity trait's `trait_score × trait_weight`.
    Always ≥ 0.

    `negative_total` sums negative-polarity contributions with the
    polarity sign already applied (negative traits use sign = -1), so
    the value is ≤ 0.

    Before implicit-prior post-reranking, `positive_total +
    negative_total` equals the score paired with the same movie_id in
    `BranchRankedResults.ranked`. When the full orchestrator applies
    implicit priors, `implicit_prior_boost` records the multiplicative
    boost fraction applied on top of that base relevance score.

    `trait_contributions` lists the signed weighted contribution from
    each trait that fed the §9 sum, in branch trait order. The sum of
    contributions equals positive_total + negative_total.
    """

    positive_total: float
    negative_total: float
    implicit_prior_boost: float = 0.0
    trait_contributions: list[TraitContribution] = field(default_factory=list)


@dataclass
class BranchRankedResults:
    """One Step-2 branch's executed + ranked output.

    `ranked` is sorted descending by final_score per §9. Ties are
    broken by the order in which candidates entered the union, which
    is deterministic given a fixed input.

    `score_breakdowns` carries the positive/negative decomposition of
    each ranked candidate's score (same movie_id keys as `ranked`).
    Empty when `ranked` is empty.

    `branch_error` mirrors the upstream Step2BranchResult.branch_error;
    when set, `ranked` is empty.
    """

    kind: "BranchKind"
    query: str
    ui_label: str
    ranked: list[tuple[int, float]] = field(default_factory=list)
    score_breakdowns: dict[int, ScoreBreakdown] = field(default_factory=dict)
    branch_error: str | None = None


# ---------------------------------------------------------------------------
# Internal types — call-level addressing
# ---------------------------------------------------------------------------


# Stable per-call identity inside a branch: (trait_idx, cat_idx,
# spec_idx). Drives the call_score_maps dict and lets Phase D look up
# scores by trait+category position without needing object identity
# (which would be fragile across the Phase B/C parallel dispatch).
_CallKey = tuple[int, int, int]

# Per-trait payload threaded from Phase D to Phase E. The 5th element
# carries per-candidate per-category scores so `_finalize_scores` can
# decompose each TraitContribution into the categories that produced
# its inner score. Empty for negative traits — see Phase D for the
# rationale.
_TraitPayload = tuple[
    "TraitWithEndpoints",
    dict[int, float],
    float,
    float,
    dict[int, list["CategoryScore"]],
]

# Per-branch trait-classification triplet. Determines whether rarity
# weighting fires for a positive trait.
_TraitClass = Literal["pure_generator", "mixed", "pure_reranker"]


class _TaggedSpec(NamedTuple):
    """One spec annotated with its (trait_idx, cat_idx, spec_idx) so
    Phase B/C can fan out specs in flat lists without losing the
    trait/category provenance Phase D needs.
    """

    key: _CallKey
    spec: GeneratedEndpointSpec


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def execute_branches(
    branches: list["Step2BranchResult"],
    auxiliary_specs_per_branch: list[list[GeneratedEndpointSpec]],
) -> list[BranchRankedResults]:
    """Run every branch in parallel and return ranked candidate lists.

    Each branch is independent: it runs its own Phase B/C/D/E using
    its own auxiliary spec view. Auxiliary specs are now per-branch
    (parallel list to `branches`) so each branch's shorts-exclusion /
    reranker-only fallback decisions can differ — which lets the
    streaming orchestrator dispatch Stage 4 for each branch as soon
    as that branch's Step 3 finishes, with no cross-branch gate.
    Errors in one branch do not affect siblings.
    """
    if not branches:
        return []
    if len(auxiliary_specs_per_branch) != len(branches):
        raise ValueError(
            "auxiliary_specs_per_branch must have one entry per branch "
            f"(got {len(auxiliary_specs_per_branch)} aux lists for "
            f"{len(branches)} branches)"
        )
    return list(
        await asyncio.gather(
            *(
                _run_branch(branch, aux)
                for branch, aux in zip(branches, auxiliary_specs_per_branch)
            )
        )
    )


# ---------------------------------------------------------------------------
# Branch-level execution — 5-phase flow
# ---------------------------------------------------------------------------


async def _run_branch(
    branch: "Step2BranchResult",
    auxiliary_specs: list[GeneratedEndpointSpec],
) -> BranchRankedResults:
    """Execute one Step-2 branch end-to-end via the 5-phase pipeline."""
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

    # ------------------------------------------------------------------
    # Step 1 — Classify positive specs into generators / rerankers and
    #          collect negative-polarity traits for separate handling.
    # ------------------------------------------------------------------
    pos_generators: list[_TaggedSpec] = []
    pos_rerankers: list[_TaggedSpec] = []
    negative_trait_indices: list[int] = []

    for trait_idx, trait in enumerate(branch.traits):
        if trait.step_3_error is not None:
            # Trait surfaced an upstream error — nothing to dispatch.
            # It will contribute 0.0 in Phase D regardless.
            continue
        if trait.polarity is Polarity.NEGATIVE:
            negative_trait_indices.append(trait_idx)
            continue
        for cat_idx, cc in enumerate(trait.category_calls):
            if cc.handler_error is not None:
                continue
            for spec_idx, spec in enumerate(cc.generated_specs):
                key: _CallKey = (trait_idx, cat_idx, spec_idx)
                tagged = _TaggedSpec(key=key, spec=spec)
                if spec.operation_type is OperationType.CANDIDATE_GENERATOR:
                    pos_generators.append(tagged)
                else:
                    pos_rerankers.append(tagged)

    # ------------------------------------------------------------------
    # Phase B — Pool definition.
    # Run every positive-polarity generator concurrently with no
    # restrict (each emits its own match set). Build the union from
    # successful generator score-map keys. Apply shorts subtraction.
    # Apply neutral seed fallback only when no positive generator was
    # attempted (i.e. every positive trait is structurally pure-
    # reranker AND tier-fallback promotion did not promote any).
    # ------------------------------------------------------------------
    call_score_maps: dict[_CallKey, dict[int, float] | None] = {}
    if pos_generators:
        # Phase 1.2 (rescore_overhaul.md D5): when two traits commit
        # generator specs that serialize identically, the underlying DB
        # query would otherwise execute twice. Group by
        # (route, _params_identity(params)) — the identity helper builds
        # a hashable structural fingerprint without paying for a
        # full Pydantic model_dump + json.dumps sort. Specs with
        # `params is None` skip dedup (no key to compare on).
        # Score-side semantics are unchanged: each (trait_idx, cat_idx,
        # spec_idx) coordinate still gets its own score map entry,
        # so per-trait combine + rarity logic reads identical inputs.
        dedup_groups: dict[tuple[str, object], list[_TaggedSpec]] = {}
        unkeyed: list[_TaggedSpec] = []
        for tagged in pos_generators:
            params = tagged.spec.params
            if params is None:
                unkeyed.append(tagged)
                continue
            key = (tagged.spec.route.value, _params_identity(params))
            dedup_groups.setdefault(key, []).append(tagged)

        # Run one representative per group + every unkeyed spec.
        representatives: list[_TaggedSpec] = [
            group[0] for group in dedup_groups.values()
        ] + unkeyed
        rep_results = await asyncio.gather(
            *(
                _dispatch_call(t.spec, restrict=None)
                for t in representatives
            )
        )
        rep_map: dict[_CallKey, dict[int, float] | None] = dict(
            zip((t.key for t in representatives), rep_results)
        )

        for group in dedup_groups.values():
            head_scores = rep_map[group[0].key]
            for tagged in group:
                call_score_maps[tagged.key] = head_scores
        for tagged in unkeyed:
            call_score_maps[tagged.key] = rep_map[tagged.key]

        if logger.isEnabledFor(logging.INFO):
            n_total = len(pos_generators)
            n_unique = len(representatives)
            if n_total != n_unique:
                logger.info(
                    "branch %s: generator dedup folded %d → %d specs",
                    branch.kind, n_total, n_unique,
                )

    union: set[int] = set()
    for tagged in pos_generators:
        scores = call_score_maps.get(tagged.key)
        if scores is not None:
            union.update(scores.keys())

    shorts_specs = [
        s for s in auxiliary_specs if s.route is EndpointRoute.MEDIA_TYPE
    ]
    if shorts_specs and union:
        union = await _subtract_shorts(union, shorts_specs)

    # Empty-pool semantics:
    #   * No generators were attempted → seed via NEUTRAL_SEED (if
    #     auxiliary spec is present).
    #   * Generators ran and union ended up empty → return empty
    #     results. Per rescore_overhaul.md: "if something truly
    #     doesn't exist, then it doesn't exist."
    if not union:
        if not pos_generators:
            seed_specs = [
                s for s in auxiliary_specs
                if s.route is EndpointRoute.NEUTRAL_SEED
            ]
            if seed_specs:
                union = await _seed_from_neutral(seed_specs)
        if not union:
            logger.info(
                "branch %s: empty union after Phase B; returning empty",
                branch.kind,
            )
            return BranchRankedResults(
                kind=branch.kind,
                query=branch.query,
                ui_label=branch.ui_label,
                ranked=[],
            )

    # ------------------------------------------------------------------
    # Phase C — Reranker pass against the finalized union.
    # Every positive reranker now scores every candidate in the union,
    # not just the trait-local subset. This is the load-bearing fix
    # vs the prior recursive-granularity scoring path.
    # ------------------------------------------------------------------
    if pos_rerankers:
        rer_results = await asyncio.gather(
            *(
                _dispatch_call(t.spec, restrict=union)
                for t in pos_rerankers
            )
        )
        for tagged, scores in zip(pos_rerankers, rer_results):
            call_score_maps[tagged.key] = scores

    # ------------------------------------------------------------------
    # Phase D — Per-trait scoring against the finalized union.
    # Positive traits compose per-call scores via combine_calls and
    # take max across categories; negative traits dispatch their calls
    # against the union and route through the existing gate × fuzzy
    # formula in _score_negative_trait.
    # ------------------------------------------------------------------
    pos_payloads: list[_TraitPayload] = []
    for trait_idx, trait in enumerate(branch.traits):
        if trait.polarity is Polarity.NEGATIVE:
            continue
        if trait.step_3_error is not None:
            # Trait failed upstream — contributes nothing.
            continue
        trait_scores, cat_scores = _score_positive_trait(
            trait_idx=trait_idx,
            trait=trait,
            union=union,
            call_score_maps=call_score_maps,
        )
        weight = _positive_trait_weight(
            trait_idx=trait_idx,
            trait=trait,
            call_score_maps=call_score_maps,
        )
        pos_payloads.append((trait, trait_scores, weight, +1.0, cat_scores))

    # Negative traits in parallel — each rescores the union via its
    # own three-bin formula. _score_negative_trait already handles
    # failed-call drops and the gate × fuzzy combine. Negative scoring
    # does not fold through per-category scores (the gate × fuzzy
    # bins partition calls cross-category), so the per-category map
    # is left empty for negatives.
    neg_payloads: list[_TraitPayload] = []
    if negative_trait_indices:
        negative_traits = [branch.traits[i] for i in negative_trait_indices]
        neg_score_maps = await asyncio.gather(
            *(_dispatch_negative_trait(t, union) for t in negative_traits)
        )
        for trait, neg_scores in zip(negative_traits, neg_score_maps):
            weight = _commitment_multiplier(trait.commitment)
            neg_payloads.append((trait, neg_scores, weight, -1.0, {}))

    # ------------------------------------------------------------------
    # Phase E — Branch aggregation.
    # ------------------------------------------------------------------
    ranked, score_breakdowns = _finalize_scores(
        union=union,
        branch_traits=branch.traits,
        pos_payloads=pos_payloads,
        neg_payloads=neg_payloads,
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
        score_breakdowns=score_breakdowns,
    )


# ---------------------------------------------------------------------------
# Phase B helpers — auxiliary spec application
# ---------------------------------------------------------------------------


async def _subtract_shorts(
    union: set[int],
    shorts_specs: list[GeneratedEndpointSpec],
) -> set[int]:
    """Subtract SHORT-format movies from the union.

    The auxiliary MEDIA_TYPE spec is tagged CANDIDATE_GENERATOR
    upstream so its executor returns the SHORT-format movies; we use
    those IDs as a blocklist rather than as a positive contribution.

    Failed shorts calls are skipped — better to let a SHORT slip
    through than to abandon the whole branch on a transient miss.
    """
    if not shorts_specs or not union:
        return union
    shorts_results = await asyncio.gather(
        *(_dispatch_call(spec, restrict=None) for spec in shorts_specs)
    )
    shorts_ids: set[int] = set()
    for scores in shorts_results:
        if scores is None:
            continue
        shorts_ids.update(scores.keys())
    if not shorts_ids:
        return union
    before = len(union)
    union = union - shorts_ids
    logger.info(
        "auxiliary: shorts-excluded %d / %d candidates",
        before - len(union),
        before,
    )
    return union


async def _seed_from_neutral(
    neutral_seed_specs: list[GeneratedEndpointSpec],
) -> set[int]:
    """Fetch the neutral-seed movie IDs as a fallback union.

    The seed only fires when no positive generator was attempted
    pipeline-wide — i.e., every positive trait is pure-reranker AND
    tier-fallback promotion did not promote any. Seed scores do not
    enter trait scoring (implicit priors handle quality/popularity
    contribution at branch aggregation).

    Best-effort: a fetch failure simply leaves the union empty, and
    the branch returns no results.
    """
    if not neutral_seed_specs:
        return set()
    try:
        seed_ids = await fetch_neutral_reranker_seed_ids()
    except Exception as exc:  # noqa: BLE001 — seed fetch is best-effort
        logger.warning(
            "neutral seed fetch failed; branch will return empty (%r)",
            exc,
        )
        return set()
    seeded = set(seed_ids)
    logger.info("auxiliary: seeded union with %d neutral-seed IDs", len(seeded))
    return seeded


# ---------------------------------------------------------------------------
# Phase D helpers — per-trait positive scoring
# ---------------------------------------------------------------------------


def _score_positive_trait(
    trait_idx: int,
    trait: "TraitWithEndpoints",
    union: set[int],
    call_score_maps: dict[_CallKey, dict[int, float] | None],
) -> tuple[dict[int, float], dict[int, list[CategoryScore]]]:
    """Compute per-candidate trait_score in [0, 1] for one positive trait.

    The math (combine_calls fold across specs per category, then
    combine_categories fold across categories) is identical to the
    pre-vectorized version. The per-mid Python loop is replaced with
    numpy array ops: per-spec score arrays are computed once via a
    sparse scatter, combined into per-category arrays with the
    category's fold rule (SINGLE = identity, ADDITIVE = product,
    ALTERNATIVES = max, CONSENSUS = floor + geom mean), then folded
    across categories per the trait's combine_mode (FRAMINGS = max,
    FACETS = floor + geom mean, SOLO = first category). The dataclass-construction pass
    over candidates is preserved so `score_breakdowns` stays
    semantically identical for downstream display + implicit-prior
    rerank.

    Returns a 2-tuple: `(trait_score_by_mid, category_scores_by_mid)`.
    The second element decomposes each candidate's trait_score into
    its per-category inputs (the same values the across-category
    fold consumed) so callers can render the WHY behind the score
    without re-running the combine logic.
    """
    # Resolve the live-call (spec, score-map) pairs once per (trait,
    # category) outside the per-candidate loop.
    live_cats: list[
        tuple[
            "CategoryCallWithEndpoints",
            CategoryCombineType,
            list[tuple[GeneratedEndpointSpec, dict[int, float]]],
        ]
    ] = []
    for cat_idx, cc in enumerate(trait.category_calls):
        if cc.handler_error is not None:
            continue
        combine_type = cc.category.combine_type
        if combine_type is CategoryCombineType.NO_OP:
            continue
        # Phase 1.1 (rescore_overhaul.md D4): a category whose handler
        # abstained — emitting zero specs — is semantically equivalent
        # to NO_OP at scoring time. Skipping it here prevents the
        # downstream `combine_calls(SINGLE, []) → 0.0` from entering
        # the across-category fold and zeroing a FACETS-PRODUCT trait.
        if not cc.generated_specs:
            continue
        live_pairs: list[tuple[GeneratedEndpointSpec, dict[int, float]]] = []
        for spec_idx, spec in enumerate(cc.generated_specs):
            scores = call_score_maps.get((trait_idx, cat_idx, spec_idx))
            if scores is not None:
                live_pairs.append((spec, scores))
        live_cats.append((cc, combine_type, live_pairs))

    # ----- Vectorized math path ------------------------------------
    # Build a stable mid ordering once. Every per-spec / per-category
    # array below indexes into this order. Iteration order of `set`
    # is implementation-defined but stable inside a single program
    # run, which is all we need (no callers depend on the absolute
    # order — they re-key by mid).
    mids_list = list(union)
    n = len(mids_list)
    if n == 0 or not live_cats:
        # No work to do — return zero-filled outputs with the union
        # keys still present so downstream consumers can iterate
        # over the same candidates regardless of trait outcome.
        return {mid: 0.0 for mid in mids_list}, {mid: [] for mid in mids_list}

    mid_to_idx = {mid: i for i, mid in enumerate(mids_list)}

    # Per-category arrays: one (specs_arr, cat_arr) entry per live
    # category. specs_arr[k] is the [n]-shaped score vector for the
    # k'th spec in the category; cat_arr is the [n]-shaped vector
    # after applying combine_calls.
    cat_blocks: list[
        tuple[
            "CategoryCallWithEndpoints",
            CategoryCombineType,
            list[tuple[GeneratedEndpointSpec, np.ndarray]],
            np.ndarray,
        ]
    ] = []
    for cc, combine_type, live_pairs in live_cats:
        spec_arrays: list[tuple[GeneratedEndpointSpec, np.ndarray]] = []
        for spec, scores in live_pairs:
            arr = np.zeros(n, dtype=np.float64)
            # Sparse scatter — only iterate the keys that actually
            # have a non-default score. Missing mids stay at 0.0,
            # matching the original `scores.get(mid, 0.0)` semantics.
            for mid, sc in scores.items():
                idx = mid_to_idx.get(mid)
                if idx is not None:
                    arr[idx] = sc
            spec_arrays.append((spec, arr))

        if not spec_arrays:
            # No specs survived; treat as a 0.0 category vector.
            cat_arr = np.zeros(n, dtype=np.float64)
        else:
            stacked = np.stack([arr for _, arr in spec_arrays])
            if combine_type is CategoryCombineType.SINGLE:
                if stacked.shape[0] > 1:
                    logger.warning(
                        "combine_calls(SINGLE) received %d scores; expected 1. "
                        "Taking the first deterministically.",
                        stacked.shape[0],
                    )
                cat_arr = stacked[0]
            elif combine_type is CategoryCombineType.ADDITIVE:
                cat_arr = np.prod(stacked, axis=0)
            elif combine_type is CategoryCombineType.ALTERNATIVES:
                cat_arr = np.max(stacked, axis=0)
            elif combine_type is CategoryCombineType.CONSENSUS:
                # Soft geometric mean over committed-call scores with
                # `_CONSENSUS_FOLD_FLOOR` (mirrors the FACETS across-
                # category fold, but at the within-category level).
                # See `combine_calls` for rationale.
                n_specs = stacked.shape[0]
                floored = np.maximum(stacked, _CONSENSUS_FOLD_FLOOR)
                cat_arr = np.prod(floored, axis=0) ** (1.0 / n_specs)
            else:
                raise ValueError(f"unknown combine_type: {combine_type!r}")
        cat_blocks.append((cc, combine_type, spec_arrays, cat_arr))

    # Fold across categories using the trait's combine_mode. Empty
    # cat_blocks already short-circuited above, so we know there is
    # at least one category here.
    cat_stack = np.stack([cb[3] for cb in cat_blocks])
    n_cats = cat_stack.shape[0]
    if trait.combine_mode is TraitCombineMode.SOLO:
        if n_cats > 1:
            logger.warning(
                "combine_categories(SOLO) received %d scores; expected 1. "
                "Taking the first deterministically.",
                n_cats,
            )
        trait_arr = cat_stack[0]
    elif trait.combine_mode is TraitCombineMode.FRAMINGS:
        trait_arr = np.max(cat_stack, axis=0)
    elif trait.combine_mode is TraitCombineMode.FACETS:
        floored = np.maximum(cat_stack, _FACETS_FOLD_FLOOR)
        product = np.prod(floored, axis=0)
        trait_arr = product ** (1.0 / n_cats)
    else:
        raise ValueError(f"unknown combine_mode: {trait.combine_mode!r}")

    # ----- Dataclass build pass ------------------------------------
    # Walk the union once to materialize the per-candidate
    # breakdowns. Allocating CategoryScore + EndpointScore here in
    # one tight loop is still the dominant cost of the function, but
    # the per-spec / per-category Python loops above are gone.
    out: dict[int, float] = {}
    cat_out: dict[int, list[CategoryScore]] = {}
    for i, mid in enumerate(mids_list):
        per_cat: list[CategoryScore] = []
        for cc, combine_type, spec_arrays, cat_arr in cat_blocks:
            per_cat.append(
                CategoryScore(
                    category_name=cc.category.value,
                    combine_type=combine_type.value,
                    expressions=cc.expressions,
                    retrieval_intent=cc.retrieval_intent,
                    score=float(cat_arr[i]),
                    endpoint_scores=[
                        EndpointScore(route=spec.route.value, score=float(arr[i]))
                        for spec, arr in spec_arrays
                    ],
                )
            )
        out[mid] = float(trait_arr[i])
        cat_out[mid] = per_cat
    return out, cat_out


def _classify_trait(trait: "TraitWithEndpoints") -> _TraitClass:
    """Classify a positive trait by the operation_type of its specs.

    Used for rarity bookkeeping only (per rescore_overhaul.md). The
    combine rules in Phase D are uniform across the three classes;
    this classification does not gate execution.

    Empty traits (handler errors zeroed every category) collapse to
    pure_reranker — there is no generator evidence to assess rarity
    against, and rarity defaults to 1.0 in that case anyway.
    """
    saw_generator = False
    saw_reranker = False
    for cc in trait.category_calls:
        if cc.handler_error is not None:
            continue
        for spec in cc.generated_specs:
            if spec.operation_type is OperationType.CANDIDATE_GENERATOR:
                saw_generator = True
            else:
                saw_reranker = True
            if saw_generator and saw_reranker:
                return "mixed"
    if saw_generator and not saw_reranker:
        return "pure_generator"
    return "pure_reranker"


def _positive_trait_weight(
    trait_idx: int,
    trait: "TraitWithEndpoints",
    call_score_maps: dict[_CallKey, dict[int, float] | None],
) -> float:
    """Compute the positive-trait weight = commitment × rarity.

    Rarity is only applied to pure-generator traits (rescore_overhaul
    "Trait weighting"). Mixed and pure-reranker traits get
    rarity_factor = 1.0; the trait weight is purely commitment.
    """
    commit = _commitment_multiplier(trait.commitment)
    classification = _classify_trait(trait)
    if classification != "pure_generator":
        return commit
    match_count = _match_count_for_rarity(
        trait_idx=trait_idx,
        trait=trait,
        call_score_maps=call_score_maps,
    )
    return commit * _rarity_factor(match_count)


def _match_count_for_rarity(
    trait_idx: int,
    trait: "TraitWithEndpoints",
    call_score_maps: dict[_CallKey, dict[int, float] | None],
) -> int:
    """Size of the rarity-relevant match set for one positive trait.

    Per §7 + the existing semantic-promoted rule:
      - Promoted (semantic-promoted) generator calls: only post-elbow
        1.0-scoring movies count. Take the union across the trait's
        promoted generator calls.
      - Regular finder generator calls: every matched candidate counts
        (i.e., every member of the call's score map).
      - When a trait mixes promoted + regular generator calls, both
        rules apply — we union both sets.

    Returns the size of the union. Failed (None) calls are skipped.
    Only the trait's POSITIVE generator calls feed rarity — rerankers
    do not contribute, since their meaning is "scoring" rather than
    "membership."
    """
    unioned: set[int] = set()
    for cat_idx, cc in enumerate(trait.category_calls):
        if cc.handler_error is not None:
            continue
        for spec_idx, spec in enumerate(cc.generated_specs):
            if spec.operation_type is not OperationType.CANDIDATE_GENERATOR:
                continue
            scores = call_score_maps.get((trait_idx, cat_idx, spec_idx))
            if scores is None:
                continue
            if spec.was_promoted:
                # Semantic-promoted: only 1.0-scoring movies count.
                unioned.update(
                    mid
                    for mid, sc in scores.items()
                    if sc >= 1.0 - _ELBOW_ONE_EPSILON
                )
            else:
                # Regular finder: all matched candidates count.
                unioned.update(scores.keys())
    return len(unioned)


# ---------------------------------------------------------------------------
# Negative-polarity scoring (preserved unchanged)
# ---------------------------------------------------------------------------


async def _dispatch_negative_trait(
    trait: "TraitWithEndpoints",
    union: set[int],
) -> dict[int, float]:
    """Dispatch a negative trait's calls against the finalized union
    and apply the gate × fuzzy three-bin formula.

    Negative-polarity rerank scope is unchanged from the prior design:
    every negative call runs restricted to the (now globally finalized)
    union, and `_score_negative_trait` composes them via the same
    G_a × (G_e ∪ R) shape as before. The asymmetry between positive
    and negative is intentional — see rescore_overhaul.md "Negative
    polarity (unchanged)."
    """
    if trait.step_3_error is not None or not union:
        return {mid: 0.0 for mid in union}

    paired: list[tuple[CategoryName, GeneratedEndpointSpec]] = []
    for cc in trait.category_calls:
        if cc.handler_error is not None:
            continue
        for spec in cc.generated_specs:
            paired.append((cc.category, spec))
    if not paired:
        return {mid: 0.0 for mid in union}

    call_results = await asyncio.gather(
        *(_dispatch_call(spec, restrict=union) for _, spec in paired)
    )
    return _score_negative_trait(paired, list(call_results), union)


# Categories whose membership signal is *authoritative* about a negative
# concept — when this signal says "not a member," the matter is settled
# (no recall gap). These are the specific-entity lookups (PERSON_CREDIT,
# NAMED_CHARACTER, …) and structured-metadata attributes (RELEASE_DATE,
# RUNTIME, …) where membership is definitionally answered. Negative-trait
# scoring ANDs these together and gates the fuzzy bin behind them.
#
# Every category NOT listed here that still routes as a CANDIDATE_GENERATOR
# (KEYWORD-style tags, GENRE, archetypes, TRENDING, CENTRAL_TOPIC, …) is
# *evidential* — a high-precision but low-recall proxy. Those OR with R
# rather than gating it, which is what fixes the "not scary" decomposition
# (KEYWORD:horror missing thrillers that SEMANTIC:scary catches).
#
# Lives here rather than in full_pipeline_orchestrator.py with the other
# category-classification tables because the orchestrator imports from
# this module at load time; pulling the set the other way would be a
# circular import.
_AUTHORITATIVE_NEGATION_CATEGORIES: frozenset[CategoryName] = frozenset({
    # Specific-entity lexical categories — definitional membership.
    CategoryName.PERSON_CREDIT,
    CategoryName.TITLE_TEXT,
    CategoryName.NAMED_CHARACTER,
    CategoryName.CHARACTER_FRANCHISE,
    CategoryName.STUDIO_BRAND,
    CategoryName.FRANCHISE_LINEAGE,
    CategoryName.ADAPTATION_SOURCE,
    CategoryName.NAMED_SOURCE_CREATOR,
    # Specific-award lookups — definitional yes/no.
    CategoryName.AWARDS,
    # Structured metadata — definitional attribute lookups.
    CategoryName.RELEASE_DATE,
    CategoryName.RUNTIME,
    CategoryName.MATURITY_RATING,
    CategoryName.AUDIO_LANGUAGE,
    CategoryName.STREAMING,
    CategoryName.FINANCIAL_SCALE,
    CategoryName.COUNTRY_OF_ORIGIN,
    CategoryName.MEDIA_TYPE,
    # NOTE: CHRONOLOGICAL is intentionally excluded. It lives on its
    # own EndpointRoute.CHRONOLOGICAL route and determine_operation_type
    # pins it to POOL_RERANKER, so its negative would-be-positive is
    # POOL_RERANKER and the call never enters the G_a partition
    # regardless of this set. Treating CHRONOLOGICAL as fuzzy is also
    # defensible semantically — most decomposer phrasings ("old",
    # "recent", "from the 80s") are approximate rather than precise.
})


def _score_negative_trait(
    paired: list[tuple[CategoryName, GeneratedEndpointSpec]],
    call_results: list[dict[int, float] | None],
    branch_pool: set[int],
) -> dict[int, float]:
    """Negative-trait inner score in [0, 1] (violation strength).

    Three-bin partition by what each call WOULD HAVE BEEN in positive
    polarity (operation_type is uniformly POOL_RERANKER for negatives,
    so we re-derive via determine_operation_type with POSITIVE):

      G_a = would-be CANDIDATE_GENERATOR with an *authoritative* category
            (specific-entity / structured-metadata — see
            _AUTHORITATIVE_NEGATION_CATEGORIES). Membership definitively
            answers the negative concept.
      G_e = would-be CANDIDATE_GENERATOR with an *evidential* category
            (keyword-style tags, archetypes, fuzzy descriptors). Proxy
            with high precision but low recall.
      R   = would-be POOL_RERANKER (continuous similarity / prior).

    Combine:
      fuzzy = noisy-OR over G_e ∪ R   = 1 − ∏ (1 − s_j)
      gate  = ∏ G_a

      - G_a and fuzzy both present → trait = gate × fuzzy
      - G_a only                    → trait = gate
      - fuzzy only                  → trait = fuzzy
      - all empty                   → trait = 0.0

    Why this shape: a single G call is often a *proxy* for a fuzzy
    concept (KEYWORD:horror standing in for "scary"); gating R behind
    it under-penalizes scary movies in adjacent genres. But when the
    decomposer issues *authoritative* G calls (PERSON_CREDIT:Joaquin
    Phoenix + NAMED_CHARACTER:Joker for "not Joaquin Phoenix Joker"),
    those describe required parts of a conjunctive concept and SHOULD
    gate — otherwise R's similarity to the JP-Joker vibe over-penalizes
    Heath Ledger's Joker. Authoritative vs evidential is the
    discriminator; G_e merges into the noisy-OR with R because both
    are alternative evidence, not required parts.

    Failed calls are dropped from their bin: a transient endpoint
    failure must not zero out the gate or saturate the noisy-OR.
    Sign is applied at the §9 final-aggregation layer, not here.
    """
    g_a_scores: list[dict[int, float]] = []
    fuzzy_scores: list[dict[int, float]] = []
    for (category, spec), scores in zip(paired, call_results):
        if scores is None:
            # Failed call — drop entirely (option A from review).
            continue
        would_be = determine_operation_type(category, spec.route, Polarity.POSITIVE)
        if would_be is OperationType.CANDIDATE_GENERATOR:
            if category in _AUTHORITATIVE_NEGATION_CATEGORIES:
                g_a_scores.append(scores)
            else:
                # Evidential G — high precision, low recall. Treated as
                # alternative evidence alongside R rather than as a gate.
                fuzzy_scores.append(scores)
        else:
            fuzzy_scores.append(scores)

    mids_list = list(branch_pool)
    n = len(mids_list)
    if n == 0:
        return {}
    mid_to_idx = {mid: i for i, mid in enumerate(mids_list)}

    def _scatter(score_dict: dict[int, float]) -> np.ndarray:
        arr = np.zeros(n, dtype=np.float64)
        for mid, sc in score_dict.items():
            idx = mid_to_idx.get(mid)
            if idx is not None:
                arr[idx] = sc
        return arr

    # Gate: AND across authoritative G calls = elementwise product.
    if g_a_scores:
        gate_arr: np.ndarray | None = np.ones(n, dtype=np.float64)
        for sd in g_a_scores:
            gate_arr = gate_arr * _scatter(sd)
    else:
        gate_arr = None

    # Fuzzy bin: noisy-OR across evidential G + R = 1 − ∏(1 − s).
    if fuzzy_scores:
        inv_product = np.ones(n, dtype=np.float64)
        for sd in fuzzy_scores:
            inv_product = inv_product * (1.0 - _scatter(sd))
        fuzzy_arr: np.ndarray | None = 1.0 - inv_product
    else:
        fuzzy_arr = None

    if gate_arr is not None and fuzzy_arr is not None:
        result_arr = gate_arr * fuzzy_arr
    elif gate_arr is not None:
        result_arr = gate_arr
    elif fuzzy_arr is not None:
        result_arr = fuzzy_arr
    else:
        # All calls failed; nothing to penalize on.
        result_arr = np.zeros(n, dtype=np.float64)

    return {mid: float(result_arr[i]) for i, mid in enumerate(mids_list)}


# ---------------------------------------------------------------------------
# Phase E — final aggregation
# ---------------------------------------------------------------------------


def _finalize_scores(
    union: set[int],
    branch_traits: list["TraitWithEndpoints"],
    pos_payloads: list[_TraitPayload],
    neg_payloads: list[_TraitPayload],
) -> tuple[list[tuple[int, float]], dict[int, ScoreBreakdown]]:
    """Per-candidate final score: Σ trait_score × trait_weight × sign.

    `pos_payloads` and `neg_payloads` are
    `(trait, scores_by_mid, weight, sign)` tuples produced in Phase D.
    Positive traits accumulate into `positive_total`; negative traits
    accumulate into `negative_total` with the polarity sign already
    applied (so `negative_total ≤ 0`).

    `branch_traits` is passed in to preserve a stable per-branch trait
    order in `trait_contributions`: traits are emitted in the same
    order they appear on the branch, so callers reading the breakdown
    side-by-side with the upstream trait list see consistent indexing
    even when some traits had upstream errors and contributed nothing.
    """
    # Resolve per-trait payloads once, in branch order. Traits that
    # contributed nothing (upstream errors, no positive payload, no
    # negative payload) get a None entry so the per-mid loop can skip
    # them without re-doing the lookup. id()-keyed indexing is safe
    # here because branch.traits holds every trait object live for
    # the duration of this call — no reuse of freed slots is possible.
    pos_by_id = {id(p[0]): p for p in pos_payloads}
    neg_by_id = {id(p[0]): p for p in neg_payloads}
    payloads_in_order: list[_TraitPayload | None] = [
        pos_by_id.get(id(trait)) or neg_by_id.get(id(trait))
        for trait in branch_traits
    ]

    final: list[tuple[int, float]] = []
    breakdowns: dict[int, ScoreBreakdown] = {}
    for movie_id in union:
        positive_total = 0.0
        negative_total = 0.0
        trait_contribs: list[TraitContribution] = []

        for payload in payloads_in_order:
            if payload is None:
                # Trait contributed nothing — omit from the breakdown
                # entirely so the row reflects what actually fed the
                # score, rather than padding with 0.0 contributions.
                continue
            trait, scores, weight, sign, cat_scores_by_mid = payload
            inner = scores.get(movie_id, 0.0)
            contribution = inner * weight * sign
            if sign >= 0.0:
                positive_total += contribution
            else:
                negative_total += contribution
            # Positive traits fold per-category scores via combine_mode;
            # negative traits use the three-bin gate × fuzzy formula
            # which doesn't read combine_mode at all. Surface that
            # distinction so the debug output reflects the actual
            # scoring path rather than the structurally-defaulted
            # combine_mode value on negative traits.
            scoring_method = (
                trait.combine_mode.value if sign >= 0.0 else "gate×fuzzy"
            )
            trait_contribs.append(
                TraitContribution(
                    surface_text=trait.surface_text,
                    commitment=trait.commitment,
                    contribution=contribution,
                    trait_score=inner,
                    weight=weight,
                    scoring_method=scoring_method,
                    category_scores=cat_scores_by_mid.get(movie_id, []),
                )
            )

        final.append((movie_id, positive_total + negative_total))
        breakdowns[movie_id] = ScoreBreakdown(
            positive_total=positive_total,
            negative_total=negative_total,
            trait_contributions=trait_contribs,
        )

    final.sort(key=lambda mid_score: mid_score[1], reverse=True)
    return final, breakdowns


# ---------------------------------------------------------------------------
# Params identity — cheap structural fingerprint for dedup
# ---------------------------------------------------------------------------


def _params_identity(value) -> object:
    """Build a hashable structural fingerprint for an endpoint params
    object.

    Replaces the older `json.dumps(model_dump(mode="json"))` dedup
    key. We don't need a canonical JSON serialization here — we only
    need two specs that would produce identical executor input to
    hash and compare equal. Walking the Pydantic model's
    `__dict__` directly and folding into nested tuples is roughly
    an order of magnitude faster than `model_dump` + `json.dumps`
    for the spec sizes this pipeline produces, and the resulting
    tuple is hashable / cheap to compare.
    """
    # Enum — fold to its raw value so two enum members from the
    # same class compare equal regardless of identity.
    if isinstance(value, Enum):
        return value.value
    # Pydantic BaseModel — recurse over instance fields. Using
    # `__dict__` instead of `model_dump` skips Pydantic's
    # serialization path entirely.
    if hasattr(value, "model_fields") and hasattr(value, "__dict__"):
        return tuple(
            (k, _params_identity(v))
            for k, v in sorted(value.__dict__.items())
            if not k.startswith("_")
        )
    # Sequence — recurse element-wise. Tuples and lists fold to a
    # tuple so the result is hashable.
    if isinstance(value, (list, tuple)):
        return tuple(_params_identity(v) for v in value)
    # Set — sort by repr to get a stable order, then fold.
    if isinstance(value, (set, frozenset)):
        return tuple(
            sorted((_params_identity(v) for v in value), key=repr)
        )
    # Mapping — recurse on items, sorted by key for stability.
    if isinstance(value, dict):
        return tuple(
            (k, _params_identity(value[k])) for k in sorted(value.keys())
        )
    # Primitive (str, int, float, bool, None) — return as-is.
    return value


# ---------------------------------------------------------------------------
# Single-call dispatch wrapper (preserved unchanged)
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
    to the negative-trait gate or noisy-OR, where one failure would
    otherwise zero out the gate or saturate the OR. Positive-trait
    callers fold None into "skip the call" and absorb the absence into
    their per-category combine; the negative-trait scorer drops failed
    calls from their bin entirely.

    NEUTRAL_SEED and TRENDING are not dispatched through this path.
    NEUTRAL_SEED is handled inside `_seed_from_neutral`; TRENDING has
    no LLM codepath in v2 and is not expected to appear in
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
