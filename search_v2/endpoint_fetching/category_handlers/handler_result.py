# Handler return contract: the four buckets every category handler
# produces, regardless of its bucket type (Single / Mutex / Tiered /
# Combo). Pure data container — no merge logic, no scoring logic.
# Consolidation (union, set-subtract, additive scoring) happens at
# the orchestrator layer across all HandlerResults, not here.
#
# See search_improvement_planning/category_handler_planning.md
# §"Handler return contract" for the design rationale and
# §"From LLM output to return buckets" for how the handler's LLM
# output (role, polarity) maps onto these four buckets.

from __future__ import annotations

from dataclasses import dataclass, field

from schemas.endpoint_parameters import EndpointParameters
from schemas.enums import EndpointRoute
from schemas.trait_category import CategoryName


@dataclass
class HandlerResult:
    # IDs the handler wants pulled into the candidate pool, with
    # endpoint-computed scores already normalized to [0, 1]. Merged
    # as a UNION with additive score across all handlers during
    # orchestrator consolidation.
    inclusion_candidates: dict[int, float] = field(default_factory=dict)

    # Soft dealbreakers: IDs whose scores contribute negatively in
    # the final rerank. Weighting is applied deterministically at the
    # orchestrator layer — the handler only decides membership and
    # surfaces the endpoint-returned score.
    downrank_candidates: dict[int, float] = field(default_factory=dict)

    # Hard dealbreakers: removed from the final candidate pool via
    # set subtraction. No scores attached — membership alone drives
    # the removal (e.g. "PG-13 max" emits R / NC-17 IDs here).
    exclusion_ids: set[int] = field(default_factory=set)

    # Deferred preference searches. Each entry is the exact
    # `endpoint_parameters` object the handler LLM emitted with
    # role == qualifier. The orchestrator runs these against
    # the final candidate pool after inclusion / exclusion
    # consolidation and routes each to the correct endpoint by
    # inspecting its concrete EndpointParameters subclass —
    # no separate routing tag needed.
    preference_specs: list[EndpointParameters] = field(default_factory=list)

    # ---- Debug / introspection only ----
    # Not consumed by the orchestrator's scoring path — these exist
    # so notebook tooling and post-hoc analysis can see which
    # category drove this result and which (route, EndpointParameters)
    # pairs the handler's LLM elected to fire (including those
    # deferred as preferences). Populated by run_handler; default
    # values keep direct constructions of HandlerResult() — used as
    # the soft-fail fallback — backwards compatible.
    category: CategoryName | None = None
    fired_endpoints: list[tuple[EndpointRoute, EndpointParameters]] = field(
        default_factory=list
    )
