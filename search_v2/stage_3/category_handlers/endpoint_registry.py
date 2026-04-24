# Maps each EndpointRoute to its concrete EndpointParameters wrapper
# subclass — the Pydantic class the step-3 handler LLM emits for that
# endpoint. Used by the sibling schema_factories module to
# dynamically build per-category output schemas from a category's
# bucket + endpoint tuple.
#
# TRENDING is intentionally mapped to None: the trending endpoint
# does not run through a handler LLM (no translation wrapper exists),
# so any category that includes TRENDING in its endpoint tuple skips
# TRENDING when assembling its output schema. Trending is handled by
# a deterministic code path elsewhere.
#
# Lives in its own module (rather than on the EndpointRoute enum
# itself) because the wrapper classes already import from schemas.enums
# — attaching them to the enum would create a circular import.

from __future__ import annotations

from schemas.award_translation import AwardEndpointParameters
from schemas.endpoint_parameters import EndpointParameters
from schemas.entity_translation import EntityEndpointParameters
from schemas.enums import EndpointRoute
from schemas.franchise_translation import FranchiseEndpointParameters
from schemas.keyword_translation import KeywordEndpointParameters
from schemas.metadata_translation import MetadataEndpointParameters
from schemas.semantic_translation import SemanticEndpointParameters
from schemas.studio_translation import StudioEndpointParameters


# EndpointRoute -> wrapper class, or None for routes with no LLM
# translation step (currently just TRENDING).
ROUTE_TO_WRAPPER: dict[EndpointRoute, type[EndpointParameters] | None] = {
    EndpointRoute.ENTITY: EntityEndpointParameters,
    EndpointRoute.STUDIO: StudioEndpointParameters,
    EndpointRoute.METADATA: MetadataEndpointParameters,
    EndpointRoute.AWARDS: AwardEndpointParameters,
    EndpointRoute.FRANCHISE_STRUCTURE: FranchiseEndpointParameters,
    EndpointRoute.KEYWORD: KeywordEndpointParameters,
    EndpointRoute.SEMANTIC: SemanticEndpointParameters,
    EndpointRoute.TRENDING: None,
}
