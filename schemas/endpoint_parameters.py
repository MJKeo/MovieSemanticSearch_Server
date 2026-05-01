# Category-handler wrapper + result types.
#
# Every category handler emits its per-finding decisions as
# EndpointParameters objects (one concrete subclass per endpoint)
# and aggregates them into a HandlerResult. role + polarity are
# carried on the parent Trait (Step 3 commits them); the handler
# stamps them onto each finding post-LLM-call rather than asking
# the LLM to regenerate them. HandlerResult is the bucketed shape
# the orchestrator folds into the final rerank.
#
#               | POSITIVE               | NEGATIVE
#   ------------+------------------------+-----------------------
#   CARVER      | inclusion_candidates   | exclusion_ids
#   QUALIFIER   | preference_specs       | downrank_candidates
#
# The orchestrator routes each preference spec to its endpoint by
# isinstance-checking its concrete EndpointParameters subclass —
# there is no separate routing tag.
#
# See search_improvement_planning/category_handler_planning.md
# ("HandlerResult shape" and "EndpointParameters base class") for
# the design rationale.
#
# No class-level docstrings or Field descriptions — per the
# "No docstrings on Pydantic classes used as LLM response_format"
# convention, these classes are slotted into dynamically-built
# handler output schemas and anything here propagates into the
# JSON schema sent on every API call.

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# Abstract marker base. Concrete subclasses (one per endpoint) live
# in the per-endpoint translation modules
# (schemas/keyword_translation.py, schemas/metadata_translation.py,
# etc.) and declare a single `parameters` field. role + polarity
# are not declared here or on subclasses — the upstream Trait owns
# them, and the handler stamps them post-hoc when bucketing the
# finding into HandlerResult.


class EndpointParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")


# The four return buckets, filled by the orchestrator after fanning
# out every coverage-evidence entry to its handler. Defaults to all-
# empty so the soft-fail retry path (see planning doc "Error
# handling") can return HandlerResult() cleanly when a handler call
# fails twice in a row.
class HandlerResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # tmdb_id -> score contribution. Positive-direction candidate
    # generation findings land here.
    inclusion_candidates: dict[int, float] = Field(default_factory=dict)

    # tmdb_id -> score contribution. Reranking findings with negative
    # polarity (push candidates down but do not remove them).
    downrank_candidates: dict[int, float] = Field(default_factory=dict)

    # Hard negative: these IDs are subtracted from the assembled
    # candidate set. No score attached.
    exclusion_ids: set[int] = Field(default_factory=set)

    # Reranking findings with positive polarity. The orchestrator
    # routes each to its endpoint by isinstance-checking the concrete
    # EndpointParameters subclass.
    preference_specs: list[EndpointParameters] = Field(default_factory=list)
