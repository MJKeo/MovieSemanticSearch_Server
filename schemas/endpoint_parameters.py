# Category-handler wrapper + result types.
#
# Every category handler emits its per-finding decisions as
# EndpointParameters objects (one concrete subclass per endpoint)
# and aggregates them into a HandlerResult. The role + polarity
# pair on every EndpointParameters tags the finding with its
# intended downstream effect; the handler does not produce
# separate bucket arrays — HandlerResult is the bucketed shape that
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


# Shared Field descriptions for the two classification fields every
# EndpointParameters wrapper declares. Module-level constants so
# (a) every concrete subclass can declare the fields in its own
# preferred order with identical wording, and (b) the wording lives
# in one place — editing here propagates to all seven endpoint
# wrappers without drift.
#
# The descriptions deliberately avoid system-architecture vocabulary
# ("candidate pool", "orchestrator", "inclusion_candidates", etc.).
# The LLM that consumes this schema only has the handler prompt's
# context — nothing about how findings are consolidated downstream
# — so architecture words are either inert tokens or get mis-
# interpreted. Every sentence here should be graspable from user-
# intent reasoning alone.
ROLE_DESCRIPTION = (
    "Pick 'carver' for a binary yes/no test that decides whether a "
    "movie belongs in the results at all — a non-matching movie is "
    "dropped entirely. Triggered by 'must', 'only', 'no X', "
    "'without', or a bare attribute assertion ('starring Tom "
    "Hanks', 'horror'). "
    "Pick 'qualifier' for a descriptive attribute that colors the "
    "ranking — a non-matching movie still qualifies, just with a "
    "lower score. Triggered by 'ideally', 'preferably', 'not too', "
    "'a bit', 'would like'. "
    "Both roles combine with either polarity, so 'carver' does not "
    "imply removal: 'must star Tom Hanks' is carver+positive, 'no "
    "horror' is carver+negative, 'preferably funny' is "
    "qualifier+positive, 'not too violent' is qualifier+negative. "
    "For neutral phrasing: pick carver for categorical attributes "
    "(genre, franchise, named entity, award) and qualifier for "
    "gradient attributes (tone, mood, quality, pacing). When unsure, "
    "prefer qualifier — carver is the stronger commitment."
)

POLARITY_DESCRIPTION = (
    "'positive' when the user wants this characteristic ('horror "
    "movies', 'starring Tom Hanks', 'with a happy ending', "
    "'acclaimed'). 'negative' when the user wants to avoid it ('no "
    "horror', 'without gore', 'not too dark', 'avoid boring'). "
    "Judge by intent, not grammar — 'movies that aren't boring' is "
    "a POSITIVE request for engaging movies, not a negative one. "
    "Downweight phrasings ('not too violent', 'a bit less dark') "
    "are negative — the user wants the trait reduced. "
    "IMPORTANT: `parameters` always describes the target concept "
    "directly, regardless of polarity. When polarity is 'negative', "
    "do NOT negate the parameters — write them as you would for a "
    "positive query about the same concept, and let this field "
    "carry the negation. For 'no Tom Hanks', `parameters` "
    "describes Tom Hanks and polarity is negative; never write "
    "'not Tom Hanks' into `parameters`, never emit an anti-keyword "
    "list, never invert the search. That would double-negate."
)


# Abstract marker base. Concrete subclasses (one per endpoint) live
# in the per-endpoint translation modules
# (schemas/keyword_translation.py, schemas/metadata_translation.py,
# etc.) and declare the three LLM-facing fields directly —
# `role`, `parameters`, `polarity` in that order. The base
# is intentionally empty of those fields: Pydantic v2 preserves
# base-class field positions under inheritance even when subclasses
# redeclare, so declaring them here would force the base order on
# the LLM output. We use concrete subclassing rather than pydantic
# generics because OpenAI structured output emits cleaner, flatter
# JSON schemas that way, and the empty base still serves its load-
# bearing purpose: the orchestrator routes preference specs by
# isinstance-checking the concrete subclass
# (HandlerResult.preference_specs below).
#
# Field ordering rationale: the LLM-facing order is
# `role → parameters → polarity`. Generating `polarity`
# immediately before `parameters` risks the model pattern-matching
# on the negative token and inverting the parameter content (the
# "not Tom Hanks" double-negative failure). Placing `polarity` last
# makes it a retrospective routing judgment over already-populated
# parameters — same pattern as primary_vector in SemanticParameters.
#
# Subclasses are expected to declare all three fields;
# __pydantic_init_subclass__ below enforces this so a future endpoint
# cannot silently ship without them.

# Module-private constant used by __pydantic_init_subclass__ below.
# Kept at module scope (not on the class) because Pydantic treats
# any underscore-prefixed class attribute as a ModelPrivateAttr,
# which is not iterable at subclass-construction time.
_REQUIRED_FIELD_ORDER: tuple[str, ...] = ("role", "parameters", "polarity")


class EndpointParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Use Pydantic's own post-field-construction hook rather than
    # Python's __init_subclass__. The latter fires before Pydantic
    # has populated cls.model_fields, making it impossible to check
    # field presence or order.
    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        missing = [f for f in _REQUIRED_FIELD_ORDER if f not in cls.model_fields]
        if missing:
            raise TypeError(
                f"{cls.__name__} must declare fields {_REQUIRED_FIELD_ORDER} "
                f"(missing: {missing}). See schemas/endpoint_parameters.py "
                "for the ordering rationale."
            )
        # Order check: the LLM-facing JSON schema follows model_fields
        # declaration order. Any deviation from the canonical order
        # reintroduces the double-negative risk.
        declared_order = [f for f in cls.model_fields if f in _REQUIRED_FIELD_ORDER]
        if declared_order != list(_REQUIRED_FIELD_ORDER):
            raise TypeError(
                f"{cls.__name__} declares fields in order {declared_order}; "
                f"expected {list(_REQUIRED_FIELD_ORDER)}. Reorder the class "
                "body so `role` precedes `parameters` precedes "
                "`polarity`."
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
