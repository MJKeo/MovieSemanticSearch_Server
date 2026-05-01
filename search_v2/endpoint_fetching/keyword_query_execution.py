# Search V2 — Stage 3 Keyword Endpoint: Query Execution
#
# Takes the LLM's KeywordQuerySpec output and runs a single
# multi-column overlap query against public.movie_card, producing
# binary-or-fractional EndpointResult objects per the spec's
# `scoring_method`.
#
# Execution model:
#   1. Resolve every member in spec.finalized_keywords (already
#      deduped by the schema validator) to a (backing_column,
#      source_id) pair via entry_for(member).
#   2. Group source_ids by backing column. The query runs once
#      against movie_card with an OR-of-overlap WHERE clause and a
#      per-column intersect-cardinality SELECT, returning one
#      hit_count per movie.
#   3. Convert hit counts to scores:
#        scoring_method == ANY  → 1.0 if hit_count >= 1 else 0.0
#        scoring_method == ALL  → hit_count / N  where N = len(finalized)
#      In ANY mode every movie that appears in the count map has
#      hit_count >= 1 by construction, so it scores 1.0.
#
# When N == 1 the two scoring modes collapse onto the same 0/1
# behavior; we still honor the LLM's choice rather than special-case
# it (no behavioral difference, less branching).
#
# Retry: transient DB errors are retried once. The second failure
# yields an empty EndpointResult rather than propagating — the
# soft-failure contract shared across stage 3 executors so the
# orchestrator can continue with contributions from the rest.

from __future__ import annotations

import logging

from db.postgres import fetch_keyword_hit_counts
from schemas.endpoint_result import EndpointResult
from schemas.enums import ScoringMethod
from schemas.keyword_translation import KeywordQuerySpec
from schemas.unified_classification import (
    ClassificationSource,
    UnifiedClassification,
    entry_for,
)
from search_v2.endpoint_fetching.result_helpers import build_endpoint_result

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
        matched movie. Non-matches are omitted.
      - set[int] (preference path) → exactly one ScoredCandidate per
        supplied ID. Non-matches score 0.0.

    Args:
        spec: Validated KeywordQuerySpec from the step 3 keyword LLM.
            spec.finalized_keywords is the deduped commitment list;
            spec.scoring_method decides whether the executor returns
            binary (ANY) or proportional (ALL) scores. The two
            inner-analysis fields (attributes, potential_keywords)
            are reasoning-only and unused at execution time.
        restrict_to_movie_ids: Optional candidate-pool restriction.
            Pass the preference's candidate pool to get one entry per
            ID; omit for the natural match set (dealbreaker path).

    Returns:
        EndpointResult with scores in [0.0, 1.0] per movie. ANY
        produces 0/1 only; ALL produces fractions when
        the spec commits more than one finalized keyword.
    """
    # use_enum_values=True on KeywordQuerySpec means each finalized
    # entry is a plain string (the member's value). Re-resolve to the
    # enum member so entry_for can look up the registry entry.
    members = [UnifiedClassification(name) for name in spec.finalized_keywords]

    # Group source_ids by backing column. Three groups maximum, any
    # of which may be empty — the DB helper accepts empty lists and
    # the per-column WHERE/SELECT degrade to no-ops on those columns.
    keyword_ids: list[int] = []
    source_material_ids: list[int] = []
    concept_tag_ids: list[int] = []
    for member in members:
        entry = entry_for(member)
        if entry.source == ClassificationSource.KEYWORD:
            keyword_ids.append(entry.source_id)
        elif entry.source == ClassificationSource.SOURCE_MATERIAL:
            source_material_ids.append(entry.source_id)
        elif entry.source == ClassificationSource.CONCEPT_TAG:
            concept_tag_ids.append(entry.source_id)
        else:
            # ClassificationSource is an exhaustive StrEnum — reaching
            # here means a new source was added without updating this
            # dispatch, which should fail loudly rather than silently
            # drop the member.
            raise ValueError(
                f"execute_keyword_query: unhandled ClassificationSource "
                f"{entry.source!r} for member {member.name}"
            )

    n_finalized = len(members)

    hit_counts: dict[int, int] = {}
    for attempt in range(2):
        try:
            hit_counts = await fetch_keyword_hit_counts(
                keyword_source_ids=keyword_ids,
                source_material_source_ids=source_material_ids,
                concept_tag_source_ids=concept_tag_ids,
                restrict_movie_ids=restrict_to_movie_ids,
            )
            break
        except Exception:
            if attempt == 0:
                logger.warning(
                    "Keyword query DB error on first attempt, retrying "
                    "(n_finalized=%d, scoring_method=%s)",
                    n_finalized,
                    spec.scoring_method,
                    exc_info=True,
                )
                continue
            # Second failure: log and return empty rather than
            # propagating. The orchestrator treats an empty result as
            # "no match" and continues; it does not see the error.
            logger.error(
                "Keyword query DB error on retry attempt, returning empty "
                "result (n_finalized=%d, scoring_method=%s)",
                n_finalized,
                spec.scoring_method,
                exc_info=True,
            )
            return EndpointResult()

    # Scoring conversion. ANY maps any positive hit count to 1.0;
    # ALL returns the matched fraction. The DB helper guarantees
    # every movie in the map has hit_count >= 1, so ANY doesn't
    # need a >0 guard.
    if spec.scoring_method == ScoringMethod.ANY:
        scores_by_movie = {mid: 1.0 for mid in hit_counts}
    else:  # ALL
        scores_by_movie = {
            mid: hits / n_finalized for mid, hits in hit_counts.items()
        }

    return build_endpoint_result(scores_by_movie, restrict_to_movie_ids)
