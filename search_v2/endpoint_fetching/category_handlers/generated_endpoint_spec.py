from __future__ import annotations

from dataclasses import dataclass

from schemas.endpoint_parameters import EndpointParameters
from schemas.enums import EndpointRoute, OperationType


@dataclass
class GeneratedEndpointSpec:
    """One executable endpoint spec ready for stage-4 execution."""

    route: EndpointRoute
    params: EndpointParameters | None
    operation_type: OperationType
