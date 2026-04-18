# Search V2 — Stage 4 internal types.
#
# Dataclasses used across the stage-4 orchestrator. Not part of the
# LLM-facing schema set — these are runtime structures only.
#
# Split from the top-level orchestrator so every sibling module can
# import them without pulling in orchestration logic.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Union

from schemas.endpoint_result import EndpointResult
from schemas.enums import EndpointRoute
from schemas.query_understanding import Dealbreaker, Preference


# Four mutually-exclusive flows the orchestrator can land in. Chosen
# once per branch by flow_detection; dictates which step-2 items act
# as candidate generators and which are score-only.
class Stage4Flow(str, Enum):
    STANDARD = "standard"
    D2 = "d2_semantic_only"
    P2 = "p2_preference_driven"
    BROWSE = "exclusion_only_browse"


# A step-2 item decorated with everything the orchestrator needs to
# dispatch and score it.  `index` is a stable identifier usable as a
# debug key — preference[3] is a distinct debug slot from dealbreaker[3].
@dataclass(frozen=True)
class TaggedItem:
    source: Union[Dealbreaker, Preference]
    role: Literal[
        "inclusion_dealbreaker",
        "exclusion_dealbreaker",
        "preference",
    ]
    endpoint: EndpointRoute
    # Set by flow_detection after the flow is chosen. True means the
    # item's execution will run with restrict_to_movie_ids=None and
    # its matched-id set gets unioned into the branch pool.
    generates_candidates: bool
    # False for dealbreakers — simpler to carry the default than
    # sprinkle `isinstance(source, Preference)` checks downstream.
    is_primary_preference: bool
    # Stable debug key. Format "<role>[<idx>]" e.g. "preference[2]".
    debug_key: str


# Outcome of one end-to-end endpoint call (translation + execution).
# Soft-fail status is explicit so debug can distinguish a real no-match
# from a timed-out or errored call — the result itself is identical
# (empty EndpointResult) in both cases.
OutcomeStatus = Literal["ok", "timeout", "error", "skipped", "no_match"]


@dataclass
class EndpointOutcome:
    item: TaggedItem
    result: EndpointResult
    status: OutcomeStatus
    llm_ms: float | None
    exec_ms: float | None
    error_message: str | None


# Per-result score breakdown kept inline with each movie in debug.
# Matches the Phase 6 formula: final = dealbreaker_sum
# + preference_contribution - exclusion_penalties.
@dataclass
class ScoreBreakdown:
    movie_id: int
    dealbreaker_sum: float
    preference_contribution: float
    exclusion_penalties: float
    final_score: float
    # debug_key -> per-item score this candidate received. Useful when
    # tracking why a candidate ranked where it did. Only includes
    # items that contributed (weight > 0 for preferences, inclusion
    # dealbreakers and semantic exclusions unconditionally).
    per_item_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class Stage4Debug:
    flow: Stage4Flow
    # Debug-key → outcome, one per tagged item. Trending shows
    # llm_ms=None since there is no translation step.
    outcomes: dict[str, EndpointOutcome]
    pool_size_after_generation: int
    pool_size_after_exclusion: int
    pool_size_after_scoring_trim: int
    # Movie_id → score breakdown, one entry per returned movie.
    per_result: dict[int, ScoreBreakdown]


@dataclass
class Stage4Result:
    # Shaped rows: {"tmdb_id", "movie_title", "release_date",
    # "poster_url"}. Rank order matches sort order; length ≤ top_k.
    movies: list[dict]
    debug: Stage4Debug
