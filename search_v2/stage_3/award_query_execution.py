# Search V2 — Stage 3 Award Endpoint: Query Execution
#
# Takes an AwardQuerySpec (from the step 3 award LLM) and produces an
# EndpointResult with [0, 1] scores per movie_id. Two data-source paths:
#
#   Fast path — movie_card.award_ceremony_win_ids (GIN-indexed SMALLINT[]).
#     Triggers only when the spec reduces to "has this movie won any
#     non-Razzie prize?": all filter fields null/empty, outcome=WINNER,
#     scoring_mode=FLOOR, scoring_mark=1. Razzie id is stripped from the
#     presence check. Outcome=null does NOT qualify — `award_ceremony_win_ids`
#     only records wins, so routing nomination-inclusive queries through it
#     would silently drop nomination-only movies. These go to the standard
#     path instead.
#
#   Standard path — COUNT(*) over public.movie_awards grouped by movie_id,
#     with whichever filter axes the spec populated. The raw count is fed
#     into the FLOOR or THRESHOLD formula per the spec's scoring_mode and
#     scoring_mark.
#
# Razzie handling: when spec.ceremonies is null/empty, ceremony_id=10 is
# excluded from both paths by default. When AwardCeremony.RAZZIE appears in
# spec.ceremonies, it is included — the user explicitly asked for it.
#
# Filter axis encoding:
#   award_name is compared to stored rows with un-normalized exact equality.
#     The ingest pipeline preserves IMDB surface forms; case-folding or
#     diacritic-stripping here would silently zero out valid matches.
#   category_tags (CategoryTag enum members) are converted to integer
#     tag ids and matched against the GIN-indexed `category_tag_ids INT[]`
#     column via array overlap (`&&`). The 3-level taxonomy
#     (schemas/award_category_tags.py) lets the LLM pick at any specificity
#     and the row's tag list contains every ancestor of its leaf concept.
#
# Retry contract: transient DB errors are retried once. A second failure
# yields an empty EndpointResult so the orchestrator can continue rather
# than hard-failing on a single endpoint. Matches the franchise and entity
# executors.
#
# See search_improvement_planning/finalized_search_proposal.md (Endpoint 3:
# Awards) for the full scoring design, and full_search_capabilities.md
# (§2 movie_awards, §1.2 award_ceremony_win_ids) for the data surface.

from __future__ import annotations

import logging

from db.postgres import fetch_award_fast_path_movie_ids, fetch_award_row_counts
from schemas.award_category_tags import RAZZIE_TAG_IDS, TAG_BY_SLUG
from schemas.award_translation import AwardQuerySpec
from schemas.endpoint_result import EndpointResult
from schemas.enums import AwardCeremony, AwardOutcome, AwardScoringMode
from search_v2.stage_3.result_helpers import build_endpoint_result

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helpers — input preprocessing and scoring. No I/O.
# ---------------------------------------------------------------------------


def _dedupe_nonempty(values: list[str] | None) -> list[str] | None:
    """Remove duplicates and empty strings from a spec list, preserving order.

    Applied to award_names only. Does NOT normalize — stored values keep
    their raw IMDB surface form and the comparison must be exact. No
    case-folding, no diacritic stripping, no whitespace trimming: any of
    those could silently mask a real stored/query mismatch. Only
    duplicates and strictly-empty strings are filtered out.

    Returns None when the input is None or every entry is empty, signaling
    to the caller that this axis is not active.
    """
    if values is None:
        return None
    seen: set[str] = set()
    result: list[str] = []
    for item in values:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result if result else None


def _resolve_category_tag_ids(
    spec_category_tags: list[str] | None,
) -> list[int] | None:
    """Convert spec category_tags (raw enum string values from
    use_enum_values=True) to integer tag ids for the SQL filter.

    Deduplicates while preserving order and silently drops any unknown slugs
    (defensive — shouldn't happen because the Pydantic enum field validates
    against CategoryTag at parse time, but a stale enum/db disagreement
    shouldn't crash execution).

    Returns None when the input is None/empty so the caller skips applying
    a category filter entirely.
    """
    if not spec_category_tags:
        return None
    seen: set[int] = set()
    result: list[int] = []
    for slug in spec_category_tags:
        tag = TAG_BY_SLUG.get(slug)
        if tag is None or tag.tag_id in seen:
            continue
        seen.add(tag.tag_id)
        result.append(tag.tag_id)
    return result if result else None


def _resolve_ceremony_ids(
    spec_ceremonies: list[str] | None,
) -> tuple[list[int] | None, bool]:
    """Convert spec ceremony values to ceremony_ids and decide Razzie policy.

    AwardQuerySpec uses use_enum_values=True, so spec.ceremonies is a list
    of raw AwardCeremony string values (e.g., "Academy Awards, USA"), not
    enum members. We map each back to its SMALLINT ceremony_id for the SQL
    filter.

    Returns:
        A tuple of (ceremony_ids, exclude_razzie):
          - (None, True)  — spec did not specify ceremonies, so no ceremony
            filter is applied and the default Razzie exclusion kicks in.
          - (resolved_ids, False) — spec named specific ceremonies. Whether
            Razzie is included depends entirely on whether the caller put
            it in the list; no default exclusion is applied on top.
    """
    if not spec_ceremonies:
        return None, True
    ceremony_ids = [AwardCeremony(c).ceremony_id for c in spec_ceremonies]
    return ceremony_ids, False


def _resolve_outcome_id(outcome_value: str | None) -> int | None:
    """Map spec.outcome (raw string or None) to its SMALLINT outcome_id."""
    if outcome_value is None:
        return None
    return AwardOutcome(outcome_value).outcome_id


def _qualifies_for_fast_path(spec: AwardQuerySpec) -> bool:
    """Decide whether the spec can use the award_ceremony_win_ids fast path.

    All of the following must hold:
      - No ceremony, award_name, category, or year filter is active.
      - outcome is WINNER. Nomination-inclusive queries (outcome=null) are
        deliberately routed through movie_awards because
        award_ceremony_win_ids records wins only — firing the fast path on
        outcome=null would silently drop nomination-only movies.
      - scoring_mode=FLOOR and scoring_mark=1. Any other combination needs
        a true row count.
    """
    if spec.scoring_mode != AwardScoringMode.FLOOR.value:
        return False
    if spec.scoring_mark != 1:
        return False
    if spec.outcome != AwardOutcome.WINNER.value:
        return False
    if spec.ceremonies:
        return False
    if spec.award_names:
        return False
    if spec.category_tags:
        return False
    if spec.years is not None:
        return False
    return True


def _score_from_count(count: int, mode: str, mark: int) -> float:
    """Apply the FLOOR or THRESHOLD formula to a raw row count.

      FLOOR:     1.0 if count >= mark else 0.0
      THRESHOLD: min(count, mark) / mark

    mark is guaranteed >= 1 by the Pydantic schema, so no divide-by-zero
    guard is needed on THRESHOLD.
    """
    if mode == AwardScoringMode.FLOOR.value:
        return 1.0 if count >= mark else 0.0
    # THRESHOLD is the only other member of the enum; any unexpected value
    # indicates a schema drift we want to surface rather than silently default.
    if mode != AwardScoringMode.THRESHOLD.value:
        raise ValueError(f"Unknown scoring_mode: {mode!r}")
    return min(count, mark) / mark


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


async def execute_award_query(
    spec: AwardQuerySpec,
    *,
    restrict_to_movie_ids: set[int] | None = None,
) -> EndpointResult:
    """Execute one AwardQuerySpec against the award data sources.

    Single entry point for both dealbreakers and preferences. The
    restrict_to_movie_ids parameter controls output shape:
      - None (dealbreaker path) → one ScoredCandidate per naturally matched
        movie. Non-matches are omitted.
      - set[int] (preference path) → exactly one ScoredCandidate per supplied
        ID. Movies that do not appear in the matched set score 0.0.

    Transient DB errors are retried once. The second failure yields an
    empty EndpointResult so the orchestrator can continue rather than
    hard-failing on a single endpoint.

    Args:
        spec: Validated AwardQuerySpec from the step 3 award LLM.
        restrict_to_movie_ids: Optional candidate-pool restriction. Pass the
            preference's candidate pool to get one entry per ID; omit for
            the natural match set (dealbreaker path).

    Returns:
        EndpointResult with scores in [0, 1] per movie.
    """
    # Safety net mirroring the trending executor: an empty candidate pool
    # on the preference path means nothing can score, so skip the DB round
    # trip entirely. The orchestrator should short-circuit first, but the
    # guard costs nothing.
    if restrict_to_movie_ids is not None and not restrict_to_movie_ids:
        return EndpointResult()

    # ---------------------------------------------------------------------
    # Fast path: generic "has any non-Razzie win" presence check.
    # ---------------------------------------------------------------------
    if _qualifies_for_fast_path(spec):
        matched_ids: set[int] = set()
        for attempt in range(2):
            try:
                matched_ids = await fetch_award_fast_path_movie_ids(
                    restrict_movie_ids=restrict_to_movie_ids,
                )
                break
            except Exception:
                if attempt == 0:
                    logger.warning(
                        "Award fast-path DB error on first attempt, retrying",
                        exc_info=True,
                    )
                    continue
                logger.error(
                    "Award fast-path DB error on retry attempt, returning empty result",
                    exc_info=True,
                )
                return EndpointResult()
        # Fast path always scores matching movies 1.0 — the trigger condition
        # is FLOOR/1, so any presence means count >= 1 means score == 1.0.
        return build_endpoint_result(
            {mid: 1.0 for mid in matched_ids},
            restrict_to_movie_ids,
        )

    # ---------------------------------------------------------------------
    # Standard path: COUNT(*) on movie_awards with active filters.
    # ---------------------------------------------------------------------
    ceremony_ids, exclude_razzie = _resolve_ceremony_ids(spec.ceremonies)
    outcome_id = _resolve_outcome_id(spec.outcome)
    award_names = _dedupe_nonempty(spec.award_names)
    category_tag_ids = _resolve_category_tag_ids(spec.category_tags)
    year_from = spec.years.year_from if spec.years is not None else None
    year_to = spec.years.year_to if spec.years is not None else None

    # The default Razzie exclusion was originally gated only on the
    # ceremonies axis (the only axis that could express Razzie intent
    # back when categories were free-text strings). The category_tags
    # axis can now express Razzie intent on its own — every "worst-*"
    # category lives exclusively on ceremony_id=10, so a tag like
    # WORST_PICTURE or the RAZZIE group is an unambiguous opt-in.
    # If we left exclude_razzie=True in those cases, the
    # `ceremony_id <> 10` filter would AND-conjunct with the tag
    # overlap and silently zero out the result. Treat any Razzie tag
    # in the resolved list as equivalent to naming Razzie in the
    # ceremonies axis.
    if exclude_razzie and category_tag_ids and any(
        tid in RAZZIE_TAG_IDS for tid in category_tag_ids
    ):
        exclude_razzie = False

    counts: dict[int, int] = {}
    for attempt in range(2):
        try:
            counts = await fetch_award_row_counts(
                ceremony_ids=ceremony_ids,
                award_names=award_names,
                category_tag_ids=category_tag_ids,
                outcome_id=outcome_id,
                year_from=year_from,
                year_to=year_to,
                exclude_razzie=exclude_razzie,
                restrict_movie_ids=restrict_to_movie_ids,
            )
            break
        except Exception:
            if attempt == 0:
                logger.warning(
                    "Award standard-path DB error on first attempt, retrying",
                    exc_info=True,
                )
                continue
            logger.error(
                "Award standard-path DB error on retry attempt, returning empty result",
                exc_info=True,
            )
            return EndpointResult()

    # Apply the scoring formula to every movie with a non-zero row count.
    # FLOOR with count < mark yields 0.0; those are dropped so the
    # dealbreaker path does not emit them and the preference path falls
    # back to the 0.0 default in build_endpoint_result.
    scores_by_movie: dict[int, float] = {}
    for movie_id, row_count in counts.items():
        score = _score_from_count(row_count, spec.scoring_mode, spec.scoring_mark)
        if score > 0.0:
            scores_by_movie[movie_id] = score

    return build_endpoint_result(scores_by_movie, restrict_to_movie_ids)
