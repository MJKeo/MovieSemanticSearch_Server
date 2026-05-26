# Shared sqrt-adaptive cast-zone primitives.
#
# A film's cast list is rank-ordered (billing_position), but the
# meaning of "position 3" depends on the cast size: it's a co-lead in
# an 8-person indie and a supporting credit in an 80-person ensemble.
# These helpers compute the zone boundaries (lead / supporting /
# minor) and the in-zone relative position (zp ∈ [0, 1]) using a
# sqrt-scaled growth curve with floors, so the same model handles
# small and large films without per-size tuning.
#
# Two callers depend on this module:
#   - search_v2.endpoint_fetching.entity_query_execution — uses the
#     zones to produce raw [0, 1] DEFAULT/LEAD/SUPPORTING/MINOR
#     actor-prominence scores. (Stage 3 entity executor.)
#   - search_v2.person_search — uses the zones to assign each (movie,
#     actor) pair to one of four prominence buckets for the person
#     entity-flow search. Only the actor postings table carries
#     billing data; non-actor role tables don't reach this module.
#     (Step 0 PERSON flow.)
#
# Keeping the constants and helpers here ensures the two callers
# never drift on tuning parameters. If LEAD_SCALE or SUPP_SCALE
# change, both the score curves and the bucket boundaries follow.

from __future__ import annotations

import math
from dataclasses import dataclass


# Tunable starting points from the proposal §Actor Prominence Scoring.
# LEAD_FLOOR guarantees at least 2 lead slots in any film (handles
# tiny-cast edge cases where round(0.6 * sqrt(n)) would be 0 or 1).
# LEAD_SCALE / SUPP_SCALE control how fast the lead and supporting
# zones grow with cast size.
LEAD_FLOOR: int = 2
LEAD_SCALE: float = 0.6
SUPP_SCALE: float = 1.0


@dataclass(frozen=True, slots=True)
class ZoneCutoffs:
    """Precomputed zone boundaries for a given cast_size.

    Billing positions 1..lead_cutoff are LEAD;
    lead_cutoff+1..supp_cutoff are SUPPORTING; the remainder are
    MINOR. Both cutoffs are clamped to <= cast_size so a tiny cast
    never produces an empty trailing zone.
    """

    lead_cutoff: int
    supp_cutoff: int


def zone_cutoffs(cast_size: int) -> ZoneCutoffs:
    """Derive zone boundaries using sqrt-scaled growth with floors.

    Worked examples:
      n=8   → lead=2, supp=3   (lead 1-2, supp 3, minor 4-8)
      n=25  → lead=3, supp=5   (lead 1-3, supp 4-5, minor 6-25)
      n=80  → lead=5, supp=9   (lead 1-5, supp 6-9, minor 10-80)
      n=200 → lead=8, supp=14  (lead 1-8, supp 9-14, minor 15-200)
    """
    sqrt_n = math.sqrt(cast_size)
    lead = min(cast_size, max(LEAD_FLOOR, round(LEAD_SCALE * sqrt_n)))
    supp = min(cast_size, max(lead + 1, round(SUPP_SCALE * sqrt_n)))
    return ZoneCutoffs(lead_cutoff=lead, supp_cutoff=supp)


def zone_relative_position(
    billing_position: int,
    zone_start: int,
    zone_end: int,
) -> float:
    """Normalize an in-zone billing position to zp ∈ [0, 1].

    zp = 0.0 at the top of the zone, 1.0 at the bottom. A single-
    member zone (zone_end == zone_start) collapses to 0.0 so the
    formula matches the top-of-zone score and the bucketer falls into
    the relevance bucket rather than the cameo bucket.
    """
    span = zone_end - zone_start
    if span <= 0:
        return 0.0
    return (billing_position - zone_start) / span
