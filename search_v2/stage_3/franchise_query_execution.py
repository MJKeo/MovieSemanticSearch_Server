# Search V2 — Stage 3 Franchise Endpoint: Query Execution
#
# Takes the LLM's FranchiseQuerySpec output and runs a single AND-composed
# query against public.movie_franchise_metadata, producing binary-scored
# EndpointResult objects. movie_franchise_metadata is the sole source of
# truth for franchise data — lex.inv_franchise_postings is not used.
#
# Scoring: binary 1.0 for any movie that satisfies all populated axes, 0.0
# otherwise. AND semantics mean an empty intermediate result on the
# dealbreaker path exits with an empty EndpointResult; on the preference
# path, non-matching candidates score 0.0.
#
# Retry: transient DB errors are retried once. The second failure yields an
# empty EndpointResult rather than propagating the exception to the caller —
# a soft-failure contract consistent with the other stage 3 executors.
#
# See search_improvement_planning/finalized_search_proposal.md
# (Endpoint 4: Franchise Structure) for the scoring design rationale.

from __future__ import annotations

import logging

from db.postgres import fetch_franchise_movie_ids
from implementation.misc.helpers import normalize_string
from schemas.endpoint_result import EndpointResult
from schemas.enums import LineagePosition
from schemas.franchise_translation import FranchiseQuerySpec
from search_v2.stage_3.result_helpers import build_endpoint_result

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input preprocessing — pure functions, no I/O.
# ---------------------------------------------------------------------------


def _normalize_variations(raw: list[str] | None) -> list[str] | None:
    """Normalize a list of name or subgroup variations and deduplicate.

    Applies normalize_string to each entry, deduplicates while preserving
    first-seen order, and filters empty results. Returns None when the input
    is None or every entry normalizes to an empty string, signaling to the
    caller that this axis is not active.
    """
    if raw is None:
        return None
    seen: set[str] = set()
    result: list[str] = []
    for item in raw:
        norm = normalize_string(item)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        result.append(norm)
    return result if result else None


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


async def execute_franchise_query(
    spec: FranchiseQuerySpec,
    *,
    restrict_to_movie_ids: set[int] | None = None,
) -> EndpointResult:
    """Execute one FranchiseQuerySpec against movie_franchise_metadata.

    Single entry point for both dealbreakers and preferences. The
    restrict_to_movie_ids parameter controls output shape:
      - None (dealbreaker path) → one ScoredCandidate per naturally matched
        movie. Non-matches are omitted.
      - set[int] (preference path) → exactly one ScoredCandidate per supplied
        ID. Non-matches score 0.0.

    All populated axes are ANDed in a single SQL query. A movie must satisfy
    every active constraint to appear in the result. An empty result is a
    valid outcome — it means no movie in the table satisfied all axes jointly.

    Transient DB errors are retried once. The second failure yields an empty
    EndpointResult so the orchestrator can continue rather than hard-failing
    on a single endpoint.

    Args:
        spec: Validated FranchiseQuerySpec from the step 3 franchise LLM.
        restrict_to_movie_ids: Optional candidate-pool restriction. Pass the
            preference's candidate pool to get one entry per ID; omit for
            the natural match set (dealbreaker path).

    Returns:
        EndpointResult with scores in {0.0, 1.0} per movie.
    """
    # Normalize name and subgroup variation lists in Python so the DB
    # comparison uses the same canonical form produced by the ingest LLM.
    normalized_names = _normalize_variations(spec.lineage_or_universe_names)
    normalized_subgroups = _normalize_variations(spec.recognized_subgroups)

    # Resolve the lineage_position string value to its SMALLINT storage ID.
    # FranchiseQuerySpec uses use_enum_values=True, so spec.lineage_position
    # holds the raw string (e.g. "sequel"), not the LineagePosition member.
    lineage_position_id: int | None = None
    if spec.lineage_position is not None:
        lineage_position_id = LineagePosition(spec.lineage_position).lineage_position_id

    matched_ids: set[int] = set()
    for attempt in range(2):
        try:
            matched_ids = await fetch_franchise_movie_ids(
                normalized_name_variations=normalized_names,
                normalized_subgroup_variations=normalized_subgroups,
                lineage_position_id=lineage_position_id,
                is_spinoff=spec.is_spinoff,
                is_crossover=spec.is_crossover,
                launched_franchise=spec.launched_franchise,
                launched_subgroup=spec.launched_subgroup,
                restrict_movie_ids=restrict_to_movie_ids,
            )
            break
        except Exception:
            if attempt == 0:
                logger.warning(
                    "Franchise query DB error on first attempt, retrying",
                    exc_info=True,
                )
                continue
            # Second failure: log and return empty rather than propagating.
            # The orchestrator treats an empty result as "no match" and
            # continues; it does not see the underlying error.
            logger.error(
                "Franchise query DB error on retry attempt, returning empty result",
                exc_info=True,
            )
            return EndpointResult()

    return build_endpoint_result({mid: 1.0 for mid in matched_ids}, restrict_to_movie_ids)
