# Search V2 — Character franchise search flow.
#
# Owns the character_franchise path described by Step 0's
# CharacterFranchiseFlowData. Fires when the entire raw query is the
# name of a character whose identity persists across multiple films
# (e.g. "Spider-Man", "James Bond", "Indiana Jones").
#
# The flow has three stages:
#
#   1. Form expansion (LLM). The Step 0 canonical_name is fed to the
#      character-franchise fanout LLM call. The fanout returns BOTH
#      cast-string aliases (character_forms — e.g. "Tony Stark",
#      "Iron Man") and franchise-name aliases (franchise_forms — e.g.
#      "Iron Man", "iron man trilogy") from one shared referent walk.
#      The fanout prompt is purpose-built for the cross-credit
#      aliasing problem and gives strictly better recall than calling
#      the franchise and character generators independently.
#
#      LLM failure (timeout, parse error, etc.) is soft: both branches
#      degrade gracefully — the runner returns an empty result with a
#      warning log rather than propagating the exception.
#
#   2. Parallel data fetches. Three lower-level Postgres helpers fire
#      concurrently:
#        - fetch_franchise_movie_ids → (lineage_matched, universe_only)
#          disjoint sets, after resolving franchise_forms to entry_ids
#          via the public resolve_franchise_names_to_entry_ids helper.
#        - fetch_character_billing_rows → raw (movie_id, billing_pos,
#          cast_size) rows for the character_forms, normalized via
#          normalize_string and resolved via fetch_character_strings_
#          exact.
#        - fetch_lineage_mainline_signals (post-franchise-resolution) →
#          (is_spinoff, release_format) per lineage-matched movie so
#          the runner can split lineage into mainline vs ancillary.
#      We bypass the standard executors (execute_franchise_query,
#      execute_entity_query) because they collapse the information we
#      need — they fold lineage vs universe into one [0, 1] score and
#      apply CENTRAL prominence mode that hides cast-size structure.
#
#   3. Seven-tier bucketed result. Movies are placed in disjoint tiers
#      by priority:
#        Tier 1 — lineage AND mainline (NOT is_spinoff AND release_
#                 format=MOVIE; crossovers still count as mainline)
#        Tier 2 — character is top-3 billed in the cast AND not in
#                 tier 1 (NEW — lets very prominent character
#                 appearances outside the lineage outrank the
#                 ancillary lineage tail without overriding the
#                 mainline franchise)
#        Tier 3 — lineage AND ancillary (is_spinoff OR not a
#                 theatrical MOVIE — animated direct-to-video,
#                 shorts, TV-movie cuts)
#        Tier 4 — universe match (excludes tier 1/2/3)
#        Tier 5 — character appearance, DEFAULT prominence >= 0.70
#                 AND billing > 3 (the previously-prominent bucket
#                 minus what now lives in tier 2)
#        Tier 6 — character appearance, 0.30 <= DEFAULT prominence < 0.70
#        Tier 7 — character appearance, DEFAULT prominence < 0.30
#      Each tier is popularity-sorted (DESC, NULLS LAST, movie_id DESC
#      tiebreaker — same convention as non_character_franchise).
#      Strict tier-priority dedupe: a movie qualifying for multiple
#      tiers appears only in its highest-priority tier.
#
# Design choice: tier 2 sits between mainline lineage and ancillary
# lineage so that a film like "Batman v Superman" — where Batman is
# a top-billed character but the film is tagged in the Superman
# lineage — outranks obscure direct-to-video Batman entries but
# never overrides "The Dark Knight" or similar mainline Batman
# theatrical releases. The mainline definition is intentionally
# loose (crossovers count as mainline; only spinoffs + non-theatrical
# formats are demoted) so we don't accidentally demote major
# theatrical entries.

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import NamedTuple

from db.postgres import (
    fetch_character_billing_rows,
    fetch_character_strings_exact,
    fetch_franchise_movie_ids,
    fetch_lineage_mainline_signals,
    fetch_quality_popularity_signals,
)
from implementation.misc.helpers import normalize_string
from schemas.enums import ReleaseFormat
from schemas.step_0_flow_routing import CharacterFranchiseFlowData
from search_v2.endpoint_fetching.category_handlers.endpoint_registry import (
    CharacterFranchiseFanoutSchema,
)
from search_v2.endpoint_fetching.character_franchise_fanout_call import (
    run_character_franchise_fanout_call,
)
from search_v2.endpoint_fetching.franchise_query_execution import (
    resolve_franchise_names_to_entry_ids,
)
from search_v2.popularity_sort import sort_movie_ids_by_popularity

logger = logging.getLogger(__name__)


# DEFAULT-mode prominence cutoffs for the character-appearance sub-
# buckets (tier 5 / 6 / 7). These bound the cast-size-relative linear
# prominence formula into Prominent / Relevant / Minor. See the runner
# docstring and the design discussion in
# plans/synthetic-sleeping-walrus.md for the rationale.
_PROMINENCE_THRESHOLD_PROMINENT = 0.70
_PROMINENCE_THRESHOLD_RELEVANT = 0.30

# Raw billing-position cutoff for the "top-billed character appearance"
# tier (tier 2). A movie where the character lands at billing position
# 1, 2, or 3 in the credits is treated as a very prominent appearance
# regardless of cast size — strong enough to outrank the ancillary
# lineage tier but never the mainline lineage tier.
_TOP_BILLED_POSITION_CUTOFF = 3

# Release-format id corresponding to a theatrical MOVIE. Used by the
# mainline-vs-ancillary lineage split. Snapshotted at module load so
# the enum lookup happens once rather than per-tier-construction.
_MOVIE_RELEASE_FORMAT_ID = ReleaseFormat.MOVIE.release_format_id


class CharacterSignals(NamedTuple):
    """Per-movie character-appearance signals after alias-row reduction.

    Two co-determined axes derived from the raw billing rows:
      * default_score — best (MAX) DEFAULT-mode prominence score
        across alias rows for the movie. Cast-size-relative,
        in [0, 1]. Drives tiers 5 / 6 / 7.
      * min_billing_position — best (MIN) raw billing position
        across alias rows. Cast-size-blind. Drives the tier 2
        top-billed cut.

    Carried as a NamedTuple so the bucketer can unpack by name —
    silent slot-order swaps would otherwise flip the tier cuts.
    """

    default_score: float
    min_billing_position: int


@dataclass
class CharacterFranchiseSearchResult:
    """Output of run_character_franchise_search.

    Seven disjoint tiers in strict priority order. Callers concatenate
    in tier order (tier_1 + tier_2 + ... + tier_7) for a flat ranked
    list. Per-tier sorts are popularity DESC, NULLS LAST, movie_id DESC.

    tier_1_lineage_mainline: movie_ids whose lineage_entry_ids overlap
        the resolved franchise entry-id set AND that are mainline
        entries (NOT is_spinoff AND release_format = MOVIE; crossovers
        count as mainline).
    tier_2_top_billed_appearance: movie_ids where any character form
        appears at billing position <= 3 in the cast AND the movie is
        not in tier_1. Lets very prominent character appearances
        outside the lineage outrank the ancillary lineage tail.
    tier_3_lineage_ancillary: movie_ids in the lineage match-set that
        are NOT mainline (spinoffs, animated direct-to-video,
        shorts, TV-movie cuts). Disjoint from tier_1.
    tier_4_universe: movie_ids whose shared_universe_entry_ids overlap
        the resolved set; disjoint from tier_1/tier_2/tier_3.
    tier_5_prominent_appearance: movie_ids where the character appears
        with DEFAULT-mode prominence >= 0.70 AND raw billing > 3 AND
        the movie is not in any higher tier.
    tier_6_relevant_appearance: same, but 0.30 <= prominence < 0.70.
    tier_7_minor_appearance: same, but prominence < 0.30.
    """

    tier_1_lineage_mainline: list[int] = field(default_factory=list)
    tier_2_top_billed_appearance: list[int] = field(default_factory=list)
    tier_3_lineage_ancillary: list[int] = field(default_factory=list)
    tier_4_universe: list[int] = field(default_factory=list)
    tier_5_prominent_appearance: list[int] = field(default_factory=list)
    tier_6_relevant_appearance: list[int] = field(default_factory=list)
    tier_7_minor_appearance: list[int] = field(default_factory=list)


async def run_character_franchise_search(
    flow_data: CharacterFranchiseFlowData,
    *,
    limit: int = 100,
) -> CharacterFranchiseSearchResult:
    """Execute the character-franchise flow.

    Calls the character-franchise fanout LLM to expand the canonical
    name into character and franchise form lists, then fetches and
    tiers movies across both signals. Falls back to an empty result on
    any LLM failure.

    Args:
        flow_data: From Step0Response.to_character_franchise_flow_data().
            Carries the canonical_name Step 0 resolved (e.g. "Spider-Man",
            "James Bond").
        limit: Maximum total rows across all seven tiers. Applied
            tier-by-tier in priority order — tier 1 fills first, then 2,
            etc. Default 100.

    Returns:
        CharacterFranchiseSearchResult with seven popularity-sorted tier
        lists. All tiers empty when the LLM fails or no franchise /
        character data resolves.
    """
    canonical_name = flow_data.canonical_name

    # Stage 1: one fanout LLM call returns both form lists.
    fanout = await _run_fanout(canonical_name)
    if fanout is None:
        return CharacterFranchiseSearchResult()

    # Stage 2: parallel data fetches. Both branches handle empty form
    # lists internally and degrade to empty results — we never need to
    # branch on "did the LLM emit forms" here.
    (lineage_matched, universe_only_matched), character_info = await asyncio.gather(
        _fetch_franchise_tiers(list(fanout.franchise_forms)),
        _fetch_character_scores(list(fanout.character_forms)),
    )

    # Stage 2b: split lineage into mainline vs ancillary using the
    # is_spinoff + release_format signals. One extra round-trip on a
    # known-small set (lineage_matched is typically <100 movies). Done
    # after the main gather rather than alongside because it depends on
    # lineage_matched as input.
    lineage_mainline, lineage_ancillary = await _split_lineage_by_mainline(
        lineage_matched
    )

    # Stage 3: bucket the character signals into the four character-
    # appearance tiers (top-billed, prominent, relevant, minor) while
    # excluding anything already in the franchise tiers (strict tier-
    # priority dedupe).
    franchise_movie_ids = lineage_matched | universe_only_matched
    (
        tier_2_top_billed,
        tier_5_prominent,
        tier_6_relevant,
        tier_7_minor,
    ) = _bucket_character_scores(character_info, exclude=franchise_movie_ids)

    # Per-tier popularity sort. Pull popularity signals in one batched
    # fetch covering every movie that lands anywhere, then apply the
    # shared sort key to each tier.
    all_movie_ids = (
        lineage_mainline
        | tier_2_top_billed
        | lineage_ancillary
        | universe_only_matched
        | tier_5_prominent
        | tier_6_relevant
        | tier_7_minor
    )
    popularity = await fetch_quality_popularity_signals(list(all_movie_ids))

    result = CharacterFranchiseSearchResult(
        tier_1_lineage_mainline=sort_movie_ids_by_popularity(lineage_mainline, popularity),
        tier_2_top_billed_appearance=sort_movie_ids_by_popularity(
            tier_2_top_billed, popularity
        ),
        tier_3_lineage_ancillary=sort_movie_ids_by_popularity(
            lineage_ancillary, popularity
        ),
        tier_4_universe=sort_movie_ids_by_popularity(universe_only_matched, popularity),
        tier_5_prominent_appearance=sort_movie_ids_by_popularity(
            tier_5_prominent, popularity
        ),
        tier_6_relevant_appearance=sort_movie_ids_by_popularity(tier_6_relevant, popularity),
        tier_7_minor_appearance=sort_movie_ids_by_popularity(tier_7_minor, popularity),
    )

    # Cap total rows by trimming from the lowest-priority tiers first.
    # Preserves tier ordering — a tier-1 movie is never dropped to make
    # room for a tier-2 movie.
    _apply_limit(result, limit)

    return result


# ---------------------------------------------------------------------------
# Stage 1 — LLM fanout call wrapper.
# ---------------------------------------------------------------------------


async def _run_fanout(
    canonical_name: str,
) -> CharacterFranchiseFanoutSchema | None:
    """Synthesize the minimal LLM inputs and invoke the fanout call.

    Returns CharacterFranchiseFanoutSchema on success or None on any
    LLM failure (already logged inside run_character_franchise_fanout_call).
    """
    # The fanout prompt expects a one-sentence positive-presence framing
    # plus the original query surface forms. Step 0 has already
    # committed that the entire query is a character name, so we use
    # the canonical_name as both the intent anchor and the sole
    # expression — matches what the prompt's character-side walk wants
    # to see as the queried referent.
    retrieval_intent = f"Movies featuring the character {canonical_name}."
    expressions = [canonical_name]
    return await run_character_franchise_fanout_call(
        retrieval_intent=retrieval_intent,
        expressions=expressions,
    )


# ---------------------------------------------------------------------------
# Stage 2 — Franchise side.
# ---------------------------------------------------------------------------


async def _fetch_franchise_tiers(
    franchise_forms: list[str],
) -> tuple[set[int], set[int]]:
    """Resolve franchise forms to (lineage_matched, universe_only).

    Mirrors the guard at franchise_query_execution.py:225-228 — if
    franchise_forms is non-empty but resolves to an empty entry-id set
    (tokenizer drift or unknown franchise), short-circuit BEFORE
    calling fetch_franchise_movie_ids. An empty franchise_name_entry_ids
    is treated by the helper as "axis inactive," which would broaden
    the match to every movie with any lineage data.
    """
    if not franchise_forms:
        return set(), set()

    entry_ids = await resolve_franchise_names_to_entry_ids(franchise_forms)
    if not entry_ids:
        # Forms were requested but resolved to nothing — return empty
        # rather than passing None and silently matching everything.
        return set(), set()

    return await fetch_franchise_movie_ids(
        franchise_name_entry_ids=entry_ids,
        subgroup_entry_ids=None,
        lineage_position_id=None,
        is_spinoff=False,
        is_crossover=False,
        launched_franchise=False,
        launched_subgroup=False,
        restrict_movie_ids=None,
    )


# ---------------------------------------------------------------------------
# Stage 2 — Character side.
# ---------------------------------------------------------------------------


async def _fetch_character_scores(
    character_forms: list[str],
) -> dict[int, CharacterSignals]:
    """Resolve character forms and compute best prominence signals per movie.

    Pipeline: normalize each form, resolve to term_ids via the
    character-strings dictionary, fetch raw billing rows, compute the
    DEFAULT-mode prominence score per row, and reduce per movie.

    Per movie we keep both the best (MAX) DEFAULT score and the best
    (MIN) raw billing position across alias rows like "Spider-Man" +
    "Peter Parker" for the same film. The two reductions are co-
    determined because all alias rows for one movie share a cast_size
    (MAX(default_score) ↔ MIN(billing_position)), but we track both
    so the downstream bucketer can apply the raw top-3 cut without
    re-deriving billing position from the score.
    """
    if not character_forms:
        return {}

    # Normalize each form. normalize_string handles unicode + casing +
    # diacritic folding consistently with the ingest-time pipeline,
    # which is what fetch_character_strings_exact expects.
    normalized = [normalize_string(form) for form in character_forms]
    normalized = [n for n in normalized if n]
    if not normalized:
        return {}

    phrase_to_id = await fetch_character_strings_exact(normalized)
    if not phrase_to_id:
        return {}

    term_ids = list(phrase_to_id.values())
    rows = await fetch_character_billing_rows(term_ids, None)

    # Per-movie reduction: MAX DEFAULT score, MIN billing position.
    # Mirrors entity_query_execution._fetch_character_scores's MAX-per-
    # movie idiom, extended to track raw billing position separately.
    info: dict[int, CharacterSignals] = {}
    for movie_id, billing_position, cast_size in rows:
        score = _default_prominence_score(billing_position, cast_size)
        prev = info.get(movie_id)
        if prev is None:
            info[movie_id] = CharacterSignals(
                default_score=score,
                min_billing_position=billing_position,
            )
            continue
        info[movie_id] = CharacterSignals(
            default_score=max(prev.default_score, score),
            min_billing_position=min(prev.min_billing_position, billing_position),
        )
    return info


def _default_prominence_score(billing_position: int, cast_size: int) -> float:
    """DEFAULT-mode prominence formula (cast-size-relative linear ramp).

    1.0 at position 1, smoothly descending to 0.0 at position ==
    cast_size. Mirrors the formula at entity_query_execution.py:248
    (_character_score_default); duplicated here so this runner has no
    runtime dependency on the entity-query module beyond its public
    surface, but the formulas must remain in lockstep — if one moves,
    update both.

    The max(1, cast_size - 1) guard handles single-character casts
    where the naive denominator would be 0. The outer max(0.0, …)
    floors the score at zero to honor the [0, 1] scoring convention
    if a row ever arrives with billing_position > cast_size (data-
    integrity edge case, not reachable from the current ingest path).
    """
    raw = 1.0 - (billing_position - 1) / max(1, cast_size - 1)
    return max(0.0, raw)


# ---------------------------------------------------------------------------
# Stage 2b — Mainline-vs-ancillary lineage split.
# ---------------------------------------------------------------------------


async def _split_lineage_by_mainline(
    lineage_matched: set[int],
) -> tuple[set[int], set[int]]:
    """Partition lineage_matched into (mainline, ancillary) using is_spinoff + release_format.

    Mainline = NOT is_spinoff AND release_format = MOVIE. Crossovers
    are intentionally NOT demoted — a major theatrical crossover
    (e.g. Batman v Superman, were it tagged in batman lineage) is
    a film a user would expect to find at the top of the franchise.

    Movies missing from the signals dict default to ancillary — a
    movie that has lineage_entry_ids set but no movie_card /
    movie_franchise_metadata row is unusual enough that we'd rather
    surface it lower than higher. The lineage_mainline_signals helper
    LEFT-JOINs both tables, so a missing-row case here would imply
    upstream data drift.
    """
    if not lineage_matched:
        return set(), set()

    signals = await fetch_lineage_mainline_signals(list(lineage_matched))
    mainline: set[int] = set()
    ancillary: set[int] = set()
    for movie_id in lineage_matched:
        is_spinoff, release_format = signals.get(movie_id, (False, 0))
        if (not is_spinoff) and release_format == _MOVIE_RELEASE_FORMAT_ID:
            mainline.add(movie_id)
        else:
            ancillary.add(movie_id)
    return mainline, ancillary


# ---------------------------------------------------------------------------
# Stage 3 — Tier bucketing and sorting.
# ---------------------------------------------------------------------------


def _bucket_character_scores(
    info: dict[int, CharacterSignals],
    *,
    exclude: set[int],
) -> tuple[set[int], set[int], set[int], set[int]]:
    """Split per-movie character signals into (tier_2, tier_5, tier_6, tier_7).

    Returns four disjoint sets in tier order:
      - tier_2_top_billed: raw billing_position <= 3 (very prominent,
        cast-size-blind); takes precedence over the DEFAULT-score cuts
      - tier_5_prominent: DEFAULT score >= 0.70 AND billing > 3
      - tier_6_relevant:  0.30 <= DEFAULT score < 0.70
      - tier_7_minor:     DEFAULT score < 0.30

    The top-3-billed check is the load-bearing addition. It catches
    "character is a top-billed actor in someone else's franchise film"
    (e.g. Batman in Batman v Superman, which Step 0 routes via "batman"
    but is tagged in the superman lineage) so those films can land
    above the obscure ancillary tail of the queried franchise without
    overriding mainline lineage entries.

    Applies the strict tier-priority dedupe — `exclude` should be
    `lineage_matched | universe_only_matched`. lineage_ancillary is
    implicitly excluded because it's a subset of lineage_matched.
    """
    tier_2: set[int] = set()
    tier_5: set[int] = set()
    tier_6: set[int] = set()
    tier_7: set[int] = set()
    for movie_id, signals in info.items():
        if movie_id in exclude:
            continue
        if signals.min_billing_position <= _TOP_BILLED_POSITION_CUTOFF:
            tier_2.add(movie_id)
        elif signals.default_score >= _PROMINENCE_THRESHOLD_PROMINENT:
            tier_5.add(movie_id)
        elif signals.default_score >= _PROMINENCE_THRESHOLD_RELEVANT:
            tier_6.add(movie_id)
        else:
            tier_7.add(movie_id)
    return tier_2, tier_5, tier_6, tier_7


# Popularity sort lives in search_v2.popularity_sort (shared with
# studio_search and any future entity-flow executor). Imported at the
# top of this module.


def _apply_limit(
    result: CharacterFranchiseSearchResult, limit: int
) -> None:
    """Trim total rows across tiers in lowest-to-highest priority order.

    Mutates `result` in place. Preserves the tier-priority invariant —
    a tier-1 movie is never dropped to make room for a tier-2 movie.
    A nonpositive limit clears every tier.
    """
    tiers = [
        result.tier_1_lineage_mainline,
        result.tier_2_top_billed_appearance,
        result.tier_3_lineage_ancillary,
        result.tier_4_universe,
        result.tier_5_prominent_appearance,
        result.tier_6_relevant_appearance,
        result.tier_7_minor_appearance,
    ]
    if limit <= 0:
        for tier in tiers:
            tier.clear()
        return

    remaining = limit
    for tier in tiers:
        if remaining <= 0:
            tier.clear()
            continue
        if len(tier) > remaining:
            del tier[remaining:]
        remaining -= len(tier)
