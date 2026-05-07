"""Tier helpers for the V2 multi-anchor specific-award lane.

`movie_awards.category_tag_ids` stores leaf + ancestor tags using a
three-level numeric encoding (per ``schemas/award_category_tags.py``):

    L0 (leaf, ids 1..99)         — specific concepts: LEAD_ACTOR,
                                   BEST_PICTURE_DRAMA, WORST_DIRECTOR
    L1 (mid,  ids 100..199)      — meaningful rollups: LEAD_ACTING covers
                                   actor + actress; BEST_PICTURE_ANY
                                   covers all picture genre splits
    L2 (group, ids 10000..10006) — seven top-level buckets: ACTING,
                                   DIRECTING, WRITING, PICTURE, CRAFT,
                                   RAZZIE, FESTIVAL_OR_TRIBUTE

The numeric range alone determines the level — no enum lookup needed
here. Keeping these constants out of ``similar_movies.py`` keeps the
main lane file scannable.
"""

from __future__ import annotations


# Per-level tier weight used by the V2 specific-award candidate score.
# Numerator/denominator both use these so candidates that match at the
# same level as the anchor consensus dominate candidates that match only
# at a coarser level. A perfect L0 BEST_PICTURE three-way match scores
# 1.0; a candidate matching only at the L2 PICTURE bucket gets 0.20.
TIER_WEIGHT: dict[int, float] = {0: 1.00, 1: 0.50, 2: 0.20}


# Specificity discount applied to lane cohesion when the anchor consensus
# only repeats at a coarser level. Three different acting awards repeating
# only at L1 LEAD_ACTING get a partial cohesion-driven boost; three
# disjoint groups repeating only at L2 get a mild one. This keeps the lane
# from over-amplifying weak generalizations like "all three are picture
# nominees in some way".
SPECIFICITY_FACTOR: dict[int, float] = {0: 1.0, 1: 0.6, 2: 0.3}


def tag_level(tag_id: int) -> int:
    """Return the taxonomy level (0, 1, or 2) for an award category tag ID.

    The numeric encoding is globally unique: 1..99 = L0 (leaves),
    100..199 = L1 (mids), 10000+ = L2 (groups). A future fourth level
    would slot in at 1_000_000+.
    """
    if tag_id >= 10000:
        return 2
    if tag_id >= 100:
        return 1
    return 0
