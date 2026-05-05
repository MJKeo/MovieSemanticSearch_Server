# Search V2 — Exact-title search flow.
#
# Owns the exact_title path described by Step 0's ExactTitleFlowData:
# (1) match title exactly via the normalized-title index;
# (2) if a release_year was supplied, partition title-matches into
#     year-exact "seeds" (1.0) and year-mismatched "title-only" (0.5);
# (3) fan out from the seeds into the same lineage and shared universe
#     at progressively lower scores, with cross-axis demotion when a
#     candidate's classification of the franchise disagrees with the
#     seed's.
#
# This module is intentionally decoupled from the standard search
# pipeline (search_v2/full_pipeline_orchestrator.py and friends). It
# only borrows read-only Postgres helpers from db/ and a couple of
# string utilities from implementation/misc/. No existing search code
# is modified.
#
# Scoring scheme (final score = max over contributions):
#   1.000  exact title (and matching year, if year was given)
#   0.750  seed's lineage entry found in candidate's own lineage
#   0.625  seed's lineage entry found in candidate's own universe
#   0.500  exact title where the user's year was given but doesn't match
#   0.250  seed's universe entry found in candidate's own universe
#   0.125  seed's universe entry found in candidate's own lineage

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from db.postgres import (
    fetch_franchise_entries_for_movies,
    fetch_franchise_movie_ids,
    fetch_movie_cards,
    fetch_movie_ids_with_title_like,
)
from implementation.misc.helpers import normalize_string
from implementation.misc.sql_like import escape_like
from schemas.step_0_flow_routing import ExactTitleFlowData


# ---------------------------------------------------------------------------
# Scoring tier constants.
#
# Named so the algorithm reads as "apply tier X" rather than embedding
# magic floats. The relative ordering between tiers is itself the
# contract — the post-pass max-with-bookkeeping relies on these being
# strictly ordered as written. Do not introduce new tiers without
# updating the merge logic.
# ---------------------------------------------------------------------------
_SCORE_SEED                 = 1.0
_SCORE_LINEAGE_TO_LINEAGE   = 0.75
_SCORE_LINEAGE_TO_UNIVERSE  = 0.625
_SCORE_TITLE_ONLY           = 0.5
_SCORE_UNIVERSE_TO_UNIVERSE = 0.25
_SCORE_UNIVERSE_TO_LINEAGE  = 0.125


# Source-label constants — surfaced in score_source for diagnostics. The
# franchise labels read "seed_axis→candidate_axis" so the cross-axis
# demotion semantics are visible at a glance.
_SRC_SEED_YEAR_MATCH    = "seed_year_match"
_SRC_SEED_TITLE_MATCH   = "seed_title_match"
_SRC_TITLE_ONLY         = "title_only"
_SRC_LINEAGE_LINEAGE    = "lineage→lineage"
_SRC_LINEAGE_UNIVERSE   = "lineage→universe"
_SRC_UNIVERSE_UNIVERSE  = "universe→universe"
_SRC_UNIVERSE_LINEAGE   = "universe→lineage"


@dataclass
class ExactTitleSearchResult:
    """Output of run_exact_title_search.

    ranked: (movie_id, score) pairs sorted descending by score, with
        ties broken by movie_id ascending for deterministic output.
        Shape mirrors search_v2.stage_4_execution.BranchRankedResults.ranked
        so any future integration into the orchestrator stays cheap.
    score_source: per-movie label naming the contribution that won the
        max — useful for debugging the franchise fan-out without
        re-running the SQL.
    """

    ranked: list[tuple[int, float]] = field(default_factory=list)
    score_source: dict[int, str] = field(default_factory=dict)


def _year_of(release_ts: int | None) -> int | None:
    """Convert a Unix-seconds release timestamp to its calendar year.

    Returns None when the timestamp is missing. Year is computed in UTC
    to match the ingest-side convention; small per-movie timezone
    differences are not interesting at the year granularity.
    """
    if release_ts is None:
        return None
    return datetime.fromtimestamp(release_ts, tz=timezone.utc).year


def _apply(
    score: dict[int, float],
    source: dict[int, str],
    movie_id: int,
    value: float,
    label: str,
) -> None:
    """Max-with-bookkeeping: write (value, label) only when value beats
    the current score. Ties don't replace the existing label so the
    earlier-applied source wins on equality (irrelevant in the current
    tier set since all tiers are strictly distinct, but a safe default).
    """
    if value > score.get(movie_id, float("-inf")):
        score[movie_id] = value
        source[movie_id] = label


async def run_exact_title_search(
    flow_data: ExactTitleFlowData,
) -> ExactTitleSearchResult:
    """Execute the exact-title search flow described by Step 0.

    See module docstring for the full scoring scheme. Inputs:
        flow_data: the ExactTitleFlowData payload Step 0 emitted. Must
            have should_be_searched=True and a non-empty title; this is
            an executor, not a router, so the caller is responsible for
            gating on Step 0's flow decisions before invoking us.

    Returns an ExactTitleSearchResult with a sorted (movie_id, score)
    list and a per-movie source label.
    """
    # ---- 1. Validate input ------------------------------------------------
    # Defensive validation at the module boundary — beyond Step 0's own
    # schema validators — so calling this function with a stale or
    # synthesized payload fails loudly rather than silently returning
    # empty results.
    if not flow_data.should_be_searched:
        raise ValueError(
            "run_exact_title_search called with should_be_searched=False; "
            "the caller must gate on Step 0's flow decision."
        )
    title = flow_data.exact_title_to_search.strip()
    if not title:
        raise ValueError("exact_title_to_search must be non-empty.")
    release_year = flow_data.release_year

    # ---- 2. Normalize + escape -------------------------------------------
    # title_normalized is written at ingest by normalize_string; running
    # the same function at query time keeps the two sides on a single
    # contract (an invariant from docs/conventions.md). escape_like
    # neutralizes %/_/\ so the bare-LIKE call below collapses to
    # equality regardless of user-typed punctuation.
    pattern = escape_like(normalize_string(title))
    if not pattern:
        # All-punctuation or all-whitespace title — nothing to search on.
        return ExactTitleSearchResult()

    # ---- 3. Title fetch ---------------------------------------------------
    title_match_ids = await fetch_movie_ids_with_title_like(pattern)
    if not title_match_ids:
        return ExactTitleSearchResult()

    # ---- 4. Bulk fetch cards for year info -------------------------------
    cards = await fetch_movie_cards(list(title_match_ids))
    id_to_year: dict[int, int | None] = {
        card["movie_id"]: _year_of(card["release_ts"]) for card in cards
    }

    # ---- 5. Partition into seeds vs title-only ---------------------------
    # When the user did not specify a year, every title hit is a seed at
    # 1.0 and there is no title-only bucket. When a year is specified,
    # seeds are the exact (title, year) pair and title-only collects
    # title-exact movies that don't satisfy the year — these are
    # downgraded to 0.5 by the user's scheme and do NOT seed franchise
    # fan-out (per the "Indy boulder" rule: never let inference upgrade
    # a non-explicit signal).
    if release_year is None:
        seeds: set[int] = set(title_match_ids)
        title_only: set[int] = set()
        seed_label = _SRC_SEED_TITLE_MATCH
    else:
        seeds = {
            mid for mid in title_match_ids
            if id_to_year.get(mid) == release_year
        }
        title_only = set(title_match_ids) - seeds
        seed_label = _SRC_SEED_YEAR_MATCH

    # ---- 6. Initial scores ------------------------------------------------
    score: dict[int, float] = {}
    source: dict[int, str] = {}
    for mid in seeds:
        score[mid] = _SCORE_SEED
        source[mid] = seed_label
    for mid in title_only:
        score[mid] = _SCORE_TITLE_ONLY
        source[mid] = _SRC_TITLE_ONLY

    # ---- 7. Franchise fan-out --------------------------------------------
    # Only seeds anchor the fan-out. Title-only candidates can still be
    # lifted by the franchise passes if they happen to share lineage or
    # universe entries with a seed (the max-with-bookkeeping handles
    # this), but they do not contribute their own franchise IDs.
    if seeds:
        seed_entries = await fetch_franchise_entries_for_movies(list(seeds))

        seed_lineage: set[int] = set()
        seed_universe: set[int] = set()
        for mid in seeds:
            lineage_ids, universe_ids = seed_entries.get(mid, (set(), set()))
            seed_lineage |= lineage_ids
            seed_universe |= universe_ids

        # Pass A: seed's lineage entries.
        # fetch_franchise_movie_ids splits hits by which side of the
        # candidate matched: lineage_hits = candidate's own lineage
        # contains the entry; universe_only_hits = candidate's universe
        # contains the entry AND the candidate's lineage does not. That
        # split aligns exactly with the 0.75 / 0.625 tiers.
        if seed_lineage:
            lineage_hits, universe_hits = await fetch_franchise_movie_ids(
                franchise_name_entry_ids=seed_lineage,
                subgroup_entry_ids=None,
                lineage_position_id=None,
                is_spinoff=False,
                is_crossover=False,
                launched_franchise=False,
                launched_subgroup=False,
            )
            for mid in lineage_hits:
                if mid in seeds:
                    continue  # Seeds keep their 1.0 unconditionally.
                _apply(
                    score, source, mid,
                    _SCORE_LINEAGE_TO_LINEAGE, _SRC_LINEAGE_LINEAGE,
                )
            for mid in universe_hits:
                if mid in seeds:
                    continue
                _apply(
                    score, source, mid,
                    _SCORE_LINEAGE_TO_UNIVERSE, _SRC_LINEAGE_UNIVERSE,
                )

        # Pass B: seed's universe entries.
        # Same call shape, lower tiers. Cross-axis demotion: an entry
        # that is part of the seed's universe but appears in a
        # candidate's lineage is the weakest signal in the scheme
        # (0.125), because it suggests the candidate predates or
        # otherwise sits in a tighter sub-lineage of a broader universe
        # the seed only loosely belongs to.
        if seed_universe:
            lineage_hits, universe_hits = await fetch_franchise_movie_ids(
                franchise_name_entry_ids=seed_universe,
                subgroup_entry_ids=None,
                lineage_position_id=None,
                is_spinoff=False,
                is_crossover=False,
                launched_franchise=False,
                launched_subgroup=False,
            )
            for mid in lineage_hits:
                if mid in seeds:
                    continue
                _apply(
                    score, source, mid,
                    _SCORE_UNIVERSE_TO_LINEAGE, _SRC_UNIVERSE_LINEAGE,
                )
            for mid in universe_hits:
                if mid in seeds:
                    continue
                _apply(
                    score, source, mid,
                    _SCORE_UNIVERSE_TO_UNIVERSE, _SRC_UNIVERSE_UNIVERSE,
                )

    # ---- 8. Sort & return -------------------------------------------------
    # Descending by score, ascending by movie_id for deterministic
    # tie-breaking. Python's sort is stable so a single key tuple is
    # enough.
    ranked = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))
    return ExactTitleSearchResult(ranked=ranked, score_source=source)
