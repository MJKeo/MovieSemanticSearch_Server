"""Format-bucket taxonomy for the V2 similar-movies flow.

Format (documentary vs short vs narrative feature vs performance vs etc.)
is its own metadata signal: viewers asking "movies like Oppenheimer"
expect narrative features in the top of the list, not historical
documentaries about the Manhattan Project. The lane treats this as a
binary same-bucket-or-not match, with a top-5 weaving constraint that
keeps the head of the result list format-coherent.

A movie is bucketed into the most specific group present in its
``keyword_ids``. Priority order (most specific first):

    mockumentary > performance > news > tv_format > short
                 > documentary > narrative_feature

Mockumentary outranks documentary because its conventions are
documentary-style but the experience is fictional. Films with no
matching keyword fall through to the implicit ``narrative_feature``
bucket — that's the bulk of the catalog.

Keys reference ``OverallKeyword.<member>.keyword_id`` so a typo in the
enum surfaces at import time.
"""

from __future__ import annotations

from typing import Iterable, Literal

from implementation.classes.overall_keywords import OverallKeyword


FormatBucket = Literal[
    "mockumentary",
    "performance",
    "news",
    "tv_format",
    "short",
    "documentary",
    "narrative_feature",
]


# Documentary group: pure DOCUMENTARY plus every sub-doc category.
#
# DOCUDRAMA and TRUE_CRIME were intentionally excluded after audit: both
# are *content* tags (based-on-real-events), not format tags. Of 30
# prestige (>=80 reception) DOCUDRAMA samples, 27 had no actual
# documentary tag — they were narrative biopics like Schindler's List,
# Goodfellas, Spotlight, The Irishman, Oppenheimer, Zero Dark Thirty,
# The Pianist, The Social Network. Treating them as documentaries broke
# the format top-lock for prestige biopic anchors. They feed the themes
# lane instead via the keyword pool in _themes_traits_for_movie.
DOCUMENTARY_KEYWORD_IDS: frozenset[int] = frozenset(
    {
        OverallKeyword.DOCUMENTARY.keyword_id,
        OverallKeyword.CRIME_DOCUMENTARY.keyword_id,
        OverallKeyword.FAITH_AND_SPIRITUALITY_DOCUMENTARY.keyword_id,
        OverallKeyword.FOOD_DOCUMENTARY.keyword_id,
        OverallKeyword.HISTORY_DOCUMENTARY.keyword_id,
        OverallKeyword.MILITARY_DOCUMENTARY.keyword_id,
        OverallKeyword.MUSIC_DOCUMENTARY.keyword_id,
        OverallKeyword.NATURE_DOCUMENTARY.keyword_id,
        OverallKeyword.POLITICAL_DOCUMENTARY.keyword_id,
        OverallKeyword.SCIENCE_AND_TECHNOLOGY_DOCUMENTARY.keyword_id,
        OverallKeyword.SPORTS_DOCUMENTARY.keyword_id,
        OverallKeyword.TRAVEL_DOCUMENTARY.keyword_id,
    }
)

MOCKUMENTARY_KEYWORD_IDS: frozenset[int] = frozenset(
    {OverallKeyword.MOCKUMENTARY.keyword_id}
)

SHORT_KEYWORD_IDS: frozenset[int] = frozenset(
    {OverallKeyword.SHORT.keyword_id}
)

PERFORMANCE_KEYWORD_IDS: frozenset[int] = frozenset(
    {
        OverallKeyword.CONCERT.keyword_id,
        OverallKeyword.STAND_UP.keyword_id,
    }
)

NEWS_KEYWORD_IDS: frozenset[int] = frozenset(
    {OverallKeyword.NEWS.keyword_id}
)

# Sparse but distinct from narrative_feature — grouping them together
# avoids surfacing reality-TV episodes alongside scripted features.
#
# SKETCH_COMEDY was intentionally excluded after audit: it's a *style*
# tag that appears on narrative features (Monty Python and the Holy
# Grail, And Now for Something Completely Different, Monty Python's The
# Meaning of Life) just as readily as on actual TV sketch shows. Putting
# it here was misclassifying feature-length sketch-style comedies as
# tv_format. It feeds the themes lane instead.
TV_FORMAT_KEYWORD_IDS: frozenset[int] = frozenset(
    {
        OverallKeyword.REALITY_TV.keyword_id,
        OverallKeyword.PARANORMAL_REALITY_TV.keyword_id,
        OverallKeyword.BUSINESS_REALITY_TV.keyword_id,
        OverallKeyword.GAME_SHOW.keyword_id,
        OverallKeyword.TALK_SHOW.keyword_id,
        OverallKeyword.SOAP_OPERA.keyword_id,
        OverallKeyword.SITCOM.keyword_id,
        OverallKeyword.COOKING_COMPETITION.keyword_id,
    }
)


# Bucket priority — most specific first. Order matters: a movie tagged
# both DOCUMENTARY and MOCKUMENTARY (uncommon, but possible) goes to
# mockumentary because the experience is fictional.
_PRIORITY: tuple[tuple[FormatBucket, frozenset[int]], ...] = (
    ("mockumentary", MOCKUMENTARY_KEYWORD_IDS),
    ("performance", PERFORMANCE_KEYWORD_IDS),
    ("news", NEWS_KEYWORD_IDS),
    ("tv_format", TV_FORMAT_KEYWORD_IDS),
    ("short", SHORT_KEYWORD_IDS),
    ("documentary", DOCUMENTARY_KEYWORD_IDS),
)


# Union of every keyword ID that participates in format bucketing.
# Themes-lane code subtracts this from a movie's keyword set so format
# tags don't double-count as theme signals (the format lane already
# handles them).
FORMAT_KEYWORD_IDS_ALL: frozenset[int] = frozenset().union(
    *(ids for _, ids in _PRIORITY)
)


def format_bucket(keyword_ids: Iterable[int] | None) -> FormatBucket:
    """Return the most specific format bucket present in ``keyword_ids``.

    Falls back to ``"narrative_feature"`` when none of the format-tag IDs
    appear — the implicit category for the bulk of the catalog.
    """
    if not keyword_ids:
        return "narrative_feature"
    kw = set(keyword_ids)
    for bucket, ids in _PRIORITY:
        if kw & ids:
            return bucket
    return "narrative_feature"
