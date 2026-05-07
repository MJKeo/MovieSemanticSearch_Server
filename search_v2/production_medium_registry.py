"""Production-medium similarity table for the V2 similar-movies flow.

Production medium (live-action vs animation, and the specific
animation technique within that) is a shape-shaping attribute:
animated and live-action films are categorically different
watching experiences, and animation sub-types are closer to each
other than to live-action. The V2 spec in
search_improvement_planning/similar_movies.md treats this as a
multiplier on combined score using a curated similarity table
(not pure IDF), so the parent ANIMATION tag matches all animated
sub-types and obvious technique pairings (e.g., hand-drawn vs
anime) score higher than weak pairings (e.g., stop-motion vs CG).

Keys reference OverallKeyword.<member>.keyword_id rather than
hardcoded ints so a typo in the enum surfaces at import time.
"""

from __future__ import annotations

import asyncio
from typing import Iterable

from implementation.classes.overall_keywords import OverallKeyword


# Stable short aliases keep the table below readable. These are
# the only medium-related OverallKeyword IDs the registry knows
# about; any other keyword id is treated as "no medium signal."
#
# ADULT_ANIMATION and HOLIDAY_ANIMATION were intentionally removed
# from this registry: they are *audience/theme* signals, not techniques.
# ADULT_ANIMATION co-occurs with HD/CG/STOP across the catalog (e.g.,
# Persepolis is HD, Mahavatar Narsimha is CG, Sita Sings the Blues has
# neither). HOLIDAY_ANIMATION mixes stop-motion (Nightmare Before
# Christmas, Rudolph) with HD (Charlie Brown Christmas) and CG (Dragons:
# Gift of the Night Fury). Treating them as mediums collapsed two
# orthogonal axes (technique x audience) into one matrix and produced
# wrong scores like Persepolis vs Sita Sings the Blues at 1.0. They
# feed the themes lane instead via the keyword pool.
_LIVE = OverallKeyword.LIVE_ACTION.keyword_id
_ANIM = OverallKeyword.ANIMATION.keyword_id
_CG = OverallKeyword.COMPUTER_ANIMATION.keyword_id
_HD = OverallKeyword.HAND_DRAWN_ANIMATION.keyword_id
_STOP = OverallKeyword.STOP_MOTION_ANIMATION.keyword_id
_ANIME = OverallKeyword.ANIME.keyword_id


# Set of all medium-related keyword IDs. Lane code uses this to
# extract a movie's medium tags from movie_card.keyword_ids.
MEDIUM_TAG_IDS: frozenset[int] = frozenset(
    {_LIVE, _ANIM, _CG, _HD, _STOP, _ANIME}
)


# 6x6 symmetric similarity matrix from the V2 spec table (with audience
# tags removed — see comment block above). Row = anchor medium tag,
# column = candidate medium tag. Self-pairs are 1.0; LIVE_ACTION is
# fully disjoint from every animation tag. The parent ANIMATION row
# scores 0.90 against every animation sub-type so a candidate tagged
# only ANIMATION still matches a specific-technique anchor.
MEDIUM_SIMILARITY: dict[int, dict[int, float]] = {
    _LIVE:  {_LIVE: 1.00, _ANIM: 0.00, _CG: 0.00, _HD: 0.00, _STOP: 0.00, _ANIME: 0.00},
    _ANIM:  {_LIVE: 0.00, _ANIM: 1.00, _CG: 0.90, _HD: 0.90, _STOP: 0.90, _ANIME: 0.90},
    _CG:    {_LIVE: 0.00, _ANIM: 0.90, _CG: 1.00, _HD: 0.60, _STOP: 0.50, _ANIME: 0.55},
    _HD:    {_LIVE: 0.00, _ANIM: 0.90, _CG: 0.60, _HD: 1.00, _STOP: 0.65, _ANIME: 0.85},
    _STOP:  {_LIVE: 0.00, _ANIM: 0.90, _CG: 0.50, _HD: 0.65, _STOP: 1.00, _ANIME: 0.55},
    _ANIME: {_LIVE: 0.00, _ANIM: 0.90, _CG: 0.55, _HD: 0.85, _STOP: 0.55, _ANIME: 1.00},
}


def medium_score(
    anchor_tags: Iterable[int],
    candidate_tags: Iterable[int],
) -> float:
    """Return the best medium-similarity score across anchor x candidate tag pairs.

    Per the V2 spec, taking ``max`` over all anchor x candidate pairs lets
    the parent ANIMATION tag absorb sub-type differences without forcing
    callers to pick a single "most specific" tag.

    Returns 0.0 when either side has no medium tags. Tags not in
    MEDIUM_TAG_IDS are silently ignored; lane code is responsible
    for filtering to medium tags before calling this helper if it
    wants to exclude noise from non-medium keywords.
    """
    # Materialize both iterables — the inner loop iterates candidates once
    # per anchor, so a one-shot iterator (generator, zip, etc.) would be
    # exhausted after the first anchor and silently return the wrong score.
    anchors = tuple(anchor_tags)
    candidates = tuple(candidate_tags)
    best = 0.0
    found_pair = False
    for a in anchors:
        row = MEDIUM_SIMILARITY.get(a)
        if row is None:
            continue
        for c in candidates:
            score = row.get(c)
            if score is None:
                continue
            found_pair = True
            if score > best:
                best = score
    return best if found_pair else 0.0


# Module-level cache of medium-tag IDFs read from public.mv_trait_idf at first
# use. The MV is refreshed post-ingest, so the values shift only on a server
# restart cadence — caching for the process lifetime is fine. The lock guards
# the first-call race so concurrent callers don't both fire the SQL.
_MEDIUM_IDFS: dict[int, float] | None = None
_MEDIUM_IDFS_LOCK = asyncio.Lock()


async def load_medium_idfs() -> dict[int, float]:
    """Return {keyword_id: idf} for every medium tag in MEDIUM_TAG_IDS.

    Lazy-loads from public.mv_trait_idf on first call (kind=1, the
    overall_keyword family). Tags absent from the MV (df=0 cases) are
    treated as idf=0.0 so callers can use ``.get(tag, 0.0)`` safely.
    Used by the V2 single-anchor selective rare-medium retrieval gate.
    """
    global _MEDIUM_IDFS
    if _MEDIUM_IDFS is not None:
        return _MEDIUM_IDFS
    async with _MEDIUM_IDFS_LOCK:
        if _MEDIUM_IDFS is not None:
            return _MEDIUM_IDFS
        # Local import to avoid a startup cycle: db.postgres imports nothing
        # from search_v2, but downstream tooling occasionally imports this
        # registry from contexts that don't have the DB pool yet.
        from db.postgres import TRAIT_KIND_OVERALL_KEYWORD, fetch_trait_idfs

        pairs = [(TRAIT_KIND_OVERALL_KEYWORD, tag) for tag in MEDIUM_TAG_IDS]
        idfs = await fetch_trait_idfs(pairs)
        loaded: dict[int, float] = {tag: 0.0 for tag in MEDIUM_TAG_IDS}
        for (_, tag_id), idf in idfs.items():
            loaded[tag_id] = idf
        _MEDIUM_IDFS = loaded
        return _MEDIUM_IDFS
