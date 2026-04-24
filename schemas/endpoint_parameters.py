# Category-handler wrapper + result types.
#
# Every category handler emits its per-finding decisions as
# EndpointParameters objects (one concrete subclass per endpoint)
# and aggregates them into a HandlerResult. The action_role +
# polarity pair on every EndpointParameters tags the finding with
# its intended downstream effect; the handler does not produce
# separate bucket arrays — HandlerResult is the bucketed shape that
# the orchestrator folds into the final rerank.
#
#                              | POSITIVE               | NEGATIVE
#   --------------------------+------------------------+-----------------------
#   CANDIDATE_IDENTIFICATION  | inclusion_candidates   | exclusion_ids
#   CANDIDATE_RERANKING       | preference_specs       | downrank_candidates
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

from schemas.enums import ActionRole, Polarity


# Abstract base. Concrete subclasses (one per endpoint) live in the
# per-endpoint translation modules (schemas/keyword_translation.py,
# schemas/metadata_translation.py, etc.) and declare their own typed
# `parameters` field pointing at the corresponding *QuerySpec. We
# use concrete subclassing rather than pydantic generics because
# OpenAI structured output emits cleaner, flatter JSON schemas that
# way.
class EndpointParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_role: ActionRole = Field(
        ...,
        description=(
            "How the finding affects results. Pick "
            "'candidate_identification' for hard requirements — "
            "phrasings like 'must', 'only', 'no X', 'without'. Movies "
            "failing the condition leave the pool. Pick "
            "'candidate_reranking' for soft preferences — 'ideally', "
            "'preferably', 'not too', 'a bit', 'would like'. Matching "
            "movies get boosted or demoted; non-matching still qualify. "
            "For neutral phrasing: pick identification for categorical "
            "attributes (genre, franchise, named entity, award) and "
            "reranking for gradient attributes (tone, mood, quality, "
            "pacing). When unsure, prefer reranking — identification is "
            "the stronger commitment."
        ),
    )
    polarity: Polarity = Field(
        ...,
        description=(
            "'positive' when the user wants this characteristic "
            "('horror movies', 'starring Tom Hanks', 'with a happy "
            "ending', 'acclaimed'). 'negative' when the user wants to "
            "avoid it ('no horror', 'without gore', 'not too dark', "
            "'avoid boring'). Judge by intent, not grammar — 'movies "
            "that aren't boring' is a POSITIVE request for engaging "
            "movies, not a negative one. Gradient-downweight phrasings "
            "('not too violent', 'a bit less dark') are negative — "
            "the user wants the trait reduced."
        ),
    )


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
