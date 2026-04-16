# Search V2 — Stage 3 Keyword Endpoint: Query Execution
#
# Takes the LLM's KeywordQuerySpec output and runs a single GIN `&&`
# overlap query against public.movie_card, producing binary-scored
# EndpointResult objects.
#
# Execution model: the LLM picks exactly one UnifiedClassification
# member. That member resolves via entry_for(member) to exactly one
# (backing_column, source_id) pair — one of:
#   - ("keyword_ids",              OverallKeyword.keyword_id)
#   - ("source_material_type_ids", SourceMaterialType.source_material_type_id)
#   - ("concept_tag_ids",          ConceptTag.concept_tag_id)
# Execution is a single-column overlap. No cross-column unions, no
# dual-backing into genre_ids. One classification, one search.
#
# Scoring: binary 1.0 for any movie whose backing column contains the
# source_id, 0.0 otherwise. The dealbreaker / preference split is the
# same dual-mode contract the other stage 3 executors expose.
#
# Retry: transient DB errors are retried once. The second failure
# yields an empty EndpointResult rather than propagating — the
# soft-failure contract shared across stage 3 executors so the
# orchestrator can continue with contributions from the rest.
#
# See search_improvement_planning/finalized_search_proposal.md
# (Endpoint 5: Keywords & Concept Tags → Execution Details) for the
# scoring rationale.

from __future__ import annotations

import logging

from db.postgres import fetch_keyword_matched_movie_ids
from schemas.endpoint_result import EndpointResult
from schemas.keyword_translation import KeywordQuerySpec
from schemas.unified_classification import UnifiedClassification, entry_for
from search_v2.stage_3.result_helpers import build_endpoint_result

logger = logging.getLogger(__name__)


async def execute_keyword_query(
    spec: KeywordQuerySpec,
    *,
    restrict_to_movie_ids: set[int] | None = None,
) -> EndpointResult:
    """Execute one KeywordQuerySpec against movie_card and return scores.

    Single entry point for both dealbreakers and preferences. The
    restrict_to_movie_ids parameter controls output shape:
      - None (dealbreaker path) → one ScoredCandidate per naturally
        matched movie at score 1.0. Non-matches are omitted.
      - set[int] (preference path) → exactly one ScoredCandidate per
        supplied ID. Non-matches score 0.0.

    Args:
        spec: Validated KeywordQuerySpec from the step 3 keyword LLM.
            spec.classification is the chosen UnifiedClassification
            member (as a raw string because the schema uses
            use_enum_values=True).
        restrict_to_movie_ids: Optional candidate-pool restriction.
            Pass the preference's candidate pool to get one entry per
            ID; omit for the natural match set (dealbreaker path).

    Returns:
        EndpointResult with scores in {0.0, 1.0} per movie.
    """
    # use_enum_values=True on KeywordQuerySpec means spec.classification
    # is a plain string (the member's value). Re-resolve to the enum
    # member so entry_for can look up the registry entry.
    member = UnifiedClassification(spec.classification)
    entry = entry_for(member)

    matched_ids: set[int] = set()
    for attempt in range(2):
        try:
            matched_ids = await fetch_keyword_matched_movie_ids(
                backing_column=entry.backing_column,
                source_id=entry.source_id,
                restrict_movie_ids=restrict_to_movie_ids,
            )
            break
        except Exception:
            if attempt == 0:
                logger.warning(
                    "Keyword query DB error on first attempt, retrying "
                    "(classification=%s, column=%s, source_id=%s)",
                    entry.name, entry.backing_column, entry.source_id,
                    exc_info=True,
                )
                continue
            # Second failure: log and return empty rather than
            # propagating. The orchestrator treats an empty result as
            # "no match" and continues; it does not see the error.
            logger.error(
                "Keyword query DB error on retry attempt, returning empty "
                "result (classification=%s, column=%s, source_id=%s)",
                entry.name, entry.backing_column, entry.source_id,
                exc_info=True,
            )
            return EndpointResult()

    return build_endpoint_result({mid: 1.0 for mid in matched_ids}, restrict_to_movie_ids)
