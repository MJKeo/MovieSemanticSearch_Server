# Search V2 — Stage 3 Franchise Endpoint: Query Execution
#
# Takes the LLM's FranchiseQuerySpec output and runs a single AND-composed
# query against the franchise token index + movie_card + (optionally)
# movie_franchise_metadata, producing an EndpointResult whose scores
# reflect both match presence and (when prefer_lineage is set) the
# lineage-vs-universe source of each match.
#
# Two-phase resolution, mirroring the studio freeform path:
#
#   Phase 1 — per-name token resolution. Each name in
#   `franchise_or_universe_names` and each name in `recognized_subgroups`
#   is tokenized via tokenize_franchise_string (normalize + ordinal +
#   cardinal + whitespace/hyphen split + FRANCHISE_STOPLIST drop). All
#   tokens across all names go into a single batched fetch against
#   lex.franchise_token, keeping the Postgres round trip count at 2
#   regardless of how many names the LLM emitted. Per-name intersection
#   happens in Python over the shared response. Cross-name union gives
#   OR semantics across the LLM's surface-form candidates (the umbrella
#   sweep).
#
#   Phase 2 — final resolution. The union entry-id sets (A from
#   franchise names, B from subgroups) and the structural flags feed
#   fetch_franchise_movie_ids, which runs the GIN `&&` overlap on
#   movie_card (both lineage_entry_ids and shared_universe_entry_ids
#   when a name axis is active) and (when any structural axis is
#   active) LEFT JOINs movie_franchise_metadata for the flag predicates.
#   The helper returns two disjoint sets: lineage matches and
#   universe-only matches.
#
# Scoring: when prefer_lineage is false (or cannot take effect), every
# match scores 1.0 and non-matches score 0.0 — the pre-flag behavior.
# When prefer_lineage is true, lineage matches score 1.0 and
# universe-only matches score 0.75, unless the lineage side is empty
# in which case universe matches are promoted back to 1.0 so the flag
# biases ranking without rejecting the match set. AND semantics mean
# an empty intermediate result on the dealbreaker path exits with an
# empty EndpointResult; on the preference path, non-matching
# candidates score 0.0.
#
# Early-exit rule: if every *populated* textual axis collapses to empty
# after token resolution (all tokens stopword-dropped or unmatched in
# the posting table), the channel returns empty unless a structural
# axis is active. A spec that asked for a specific franchise name but
# resolved to no entry ids should not accidentally match every spinoff
# via a lingering structural flag it never requested.
#
# Retry: transient DB errors are retried once. The second failure yields an
# empty EndpointResult rather than propagating the exception to the caller —
# a soft-failure contract consistent with the other stage 3 executors.
#
# See search_improvement_planning/v2_search_data_improvements.md
# §Franchise Resolution for the full design rationale.

from __future__ import annotations

import logging

from db.postgres import (
    fetch_franchise_entry_ids_for_tokens,
    fetch_franchise_movie_ids,
)
from implementation.misc.franchise_text import tokenize_franchise_string
from schemas.endpoint_result import EndpointResult
from schemas.enums import (
    FranchiseLaunchScope,
    FranchiseStructuralFlag,
    LineagePosition,
)
from schemas.franchise_translation import FranchiseQuerySpec
from search_v2.stage_3.result_helpers import build_endpoint_result

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token-index resolution — pure Python over a shared batched response.
# ---------------------------------------------------------------------------


async def _resolve_names_to_entry_ids(names: list[str] | None) -> set[int]:
    """Resolve a list of franchise / subgroup surface forms to entry ids.

    Pipeline per spec §Query-Time Resolution Step 1-3:
      1. Tokenize each name (normalize + ordinal + cardinal + whitespace
         / hyphen split + FRANCHISE_STOPLIST drop). A name that reduces
         to zero tokens after stopword drop contributes nothing and is
         skipped.
      2. Collect all distinct tokens across all names into one batched
         fetch against lex.franchise_token (1 round trip, not N).
      3. Per-name intersection over the shared response. A missing token
         (never stamped at ingest) collapses that name's contribution to
         empty — we must NOT silently treat "missing" as "matches
         everything", matching the studio freeform executor's Phase-3
         behavior.
      4. Union the per-name sets. Cross-name union is OR semantics for
         the LLM's multiple surface-form candidates — this is the
         umbrella-sweep mechanism (e.g. emitting
         `["marvel cinematic universe", "marvel"]` sweeps MCU PLUS every
         other `marvel`-tagged entry).

    Args:
        names: Raw surface forms from the FranchiseQuerySpec (not
            pre-normalized — tokenize_franchise_string normalizes
            internally). None or empty = axis not active.

    Returns:
        Set of franchise_entry_ids. Empty when the axis was inactive,
        every name collapsed to zero tokens, or no tokens had postings.
    """
    if not names:
        return set()

    # Phase 1: per-name tokenize. Preserve grouping for intersection, and
    # collect a flat token set for the batched fetch.
    per_name_tokens: list[list[str]] = []
    all_tokens: set[str] = set()
    for name in names:
        tokens = tokenize_franchise_string(name)
        if not tokens:
            continue
        per_name_tokens.append(tokens)
        all_tokens.update(tokens)
    if not all_tokens:
        return set()

    # Phase 2: single batched posting-list fetch. sorted() gives
    # deterministic query text (stable plan cache, reproducible logs).
    token_to_entries = await fetch_franchise_entry_ids_for_tokens(sorted(all_tokens))

    # Phase 3: per-name intersection over the shared response. A missing
    # key means the token has no postings — the name fails, it does NOT
    # match everything.
    all_entry_ids: set[int] = set()
    for tokens in per_name_tokens:
        per_token_sets = [token_to_entries.get(t) for t in tokens]
        if not all(per_token_sets):
            continue
        all_entry_ids |= set.intersection(*per_token_sets)

    return all_entry_ids


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


async def execute_franchise_query(
    spec: FranchiseQuerySpec,
    *,
    restrict_to_movie_ids: set[int] | None = None,
) -> EndpointResult:
    """Execute one FranchiseQuerySpec against the franchise token index.

    Single entry point for both dealbreakers and preferences. The
    restrict_to_movie_ids parameter controls output shape:
      - None (dealbreaker path) → one ScoredCandidate per naturally matched
        movie. Non-matches are omitted.
      - set[int] (preference path) → exactly one ScoredCandidate per supplied
        ID. Non-matches score 0.0.

    All populated axes are ANDed in a single SQL query. A movie must satisfy
    every active constraint to appear in the result. An empty result is a
    valid outcome — it means no movie satisfied all axes jointly.

    Transient DB errors are retried once. The second failure yields an empty
    EndpointResult so the orchestrator can continue rather than hard-failing
    on a single endpoint.

    Args:
        spec: Validated FranchiseQuerySpec from the step 3 franchise LLM.
        restrict_to_movie_ids: Optional candidate-pool restriction. Pass the
            preference's candidate pool to get one entry per ID; omit for
            the natural match set (dealbreaker path).

    Returns:
        EndpointResult with per-movie scores. When `spec.prefer_lineage`
        is False (or cannot take effect), every matched movie scores 1.0.
        When True, movies whose lineage_entry_ids overlap the query name
        set score 1.0, and movies that matched only via
        shared_universe_entry_ids score 0.75. When the name side has no
        lineage matches at all, universe-only matches are promoted back
        to 1.0 so the flag biases the ranking without rejecting the
        entire result set.
    """
    # Resolve the lineage_position string value to its SMALLINT storage ID.
    # FranchiseQuerySpec uses use_enum_values=True, so spec.lineage_position
    # holds the raw string (e.g. "sequel"), not the LineagePosition member.
    lineage_position_id: int | None = None
    if spec.lineage_position is not None:
        lineage_position_id = LineagePosition(spec.lineage_position).lineage_position_id

    structural_flags = set(spec.structural_flags or [])
    is_spinoff = FranchiseStructuralFlag.SPINOFF in structural_flags
    is_crossover = FranchiseStructuralFlag.CROSSOVER in structural_flags

    launched_franchise = spec.launch_scope == FranchiseLaunchScope.FRANCHISE
    launched_subgroup = spec.launch_scope == FranchiseLaunchScope.SUBGROUP

    # Phase 1 — resolve both textual axes to entry-id sets. Run
    # sequentially rather than gather()-ing because the two share the
    # same posting-list fetch pattern and the second call would benefit
    # from Postgres connection reuse; the wall-clock difference for 1-6
    # total names is negligible.
    franchise_name_entry_ids = await _resolve_names_to_entry_ids(
        spec.franchise_or_universe_names
    )
    subgroup_entry_ids = await _resolve_names_to_entry_ids(spec.recognized_subgroups)

    # Early-exit rule (user directive #3 in the plan). A textual axis
    # that was *requested* but resolved to an empty entry-id set must
    # short-circuit to an empty result — the DB helper treats an empty
    # set as "axis inactive" and skips the predicate, which would
    # silently broaden the match. "No entries resolved" is not the same
    # as "axis inactive".
    if spec.franchise_or_universe_names and not franchise_name_entry_ids:
        return build_endpoint_result({}, restrict_to_movie_ids)
    if spec.recognized_subgroups and not subgroup_entry_ids:
        return build_endpoint_result({}, restrict_to_movie_ids)

    lineage_matched: set[int] = set()
    universe_only_matched: set[int] = set()
    for attempt in range(2):
        try:
            lineage_matched, universe_only_matched = await fetch_franchise_movie_ids(
                franchise_name_entry_ids=franchise_name_entry_ids or None,
                subgroup_entry_ids=subgroup_entry_ids or None,
                lineage_position_id=lineage_position_id,
                is_spinoff=is_spinoff,
                is_crossover=is_crossover,
                launched_franchise=launched_franchise,
                launched_subgroup=launched_subgroup,
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

    # Score the match set. Three cases:
    #   1. prefer_lineage not set — every match scores 1.0. Preserves
    #      the pre-flag behavior for umbrella queries, subgroup queries,
    #      spinoff queries, etc.
    #   2. prefer_lineage set and lineage side is empty but universe
    #      side is non-empty — promote universe matches to 1.0. The
    #      flag biases ranking, so with no lineage matches to uprank
    #      it falls back to treating universe matches as the main
    #      result set rather than demoting them all.
    #   3. prefer_lineage set with a non-empty lineage side — lineage
    #      scores 1.0, universe-only scores 0.75. The helper already
    #      attributes movies that match via both sides to lineage, so
    #      the two sets are disjoint.
    if not spec.prefer_lineage:
        scores = {mid: 1.0 for mid in (lineage_matched | universe_only_matched)}
    elif not lineage_matched and universe_only_matched:
        scores = {mid: 1.0 for mid in universe_only_matched}
    else:
        scores = {mid: 1.0 for mid in lineage_matched}
        scores.update({mid: 0.75 for mid in universe_only_matched})

    return build_endpoint_result(scores, restrict_to_movie_ids)
