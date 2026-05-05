from __future__ import annotations

from dataclasses import dataclass

from schemas.endpoint_parameters import EndpointParameters
from schemas.enums import EndpointRoute, OperationType


@dataclass
class GeneratedEndpointSpec:
    """One executable endpoint spec ready for stage-4 execution.

    `was_promoted` is set True only by the reranker-only candidate
    fallback (full_pipeline_orchestrator._apply_reranker_only_candidate_fallback)
    when it flips a reranker spec's operation_type to CANDIDATE_GENERATOR.
    Stage-4 reads this to decide whether the trait's rarity should be
    counted from elbow-1.0 movies (promoted) or from all matched
    candidates (regular finder).
    """

    route: EndpointRoute
    params: EndpointParameters | None
    operation_type: OperationType
    was_promoted: bool = False
