"""Country/language coherence registry for the V2 multi-anchor flow.

Vector spaces capture nation/language differences only weakly — a
French drama and a Spanish drama with similar plot vectors look
adjacent enough that the centroid drifts. This registry pulls the
30 nation/language tags out of ``OverallKeyword`` so the multi-anchor
country/language coherence multiplier can pin results to the same
tradition as the anchor set.

A movie that carries any of these tags belongs to its nation/language
bucket(s). A movie that carries none of them is bucketed as
``US_DEFAULT`` — covering the bulk of US/UK/Canadian/Australian
English-language productions in the catalog. Treating that fallback
as a real bucket (rather than "no signal") is what gives the
"boost American films when anchors are American" behavior the V2
spec calls for.
"""

from __future__ import annotations

from typing import Iterable, Literal

from implementation.classes.overall_keywords import OverallKeyword


# Sentinel used as the bucket for any movie carrying none of the
# country/language tags. ``country_set`` returns ``{US_DEFAULT}`` for
# such movies so consensus detection treats English-language US/UK
# content as a real shared tradition rather than a missing signal.
US_DEFAULT: Literal["US_DEFAULT"] = "US_DEFAULT"


# The 30 nation/language tags from OverallKeyword (V2 spec list verbatim).
# Stored as a frozenset of keyword_ids so callers can intersect a
# movie's keyword_ids set in O(|keyword_ids|).
COUNTRY_LANGUAGE_KEYWORD_IDS: frozenset[int] = frozenset(
    {
        OverallKeyword.ARABIC.keyword_id,
        OverallKeyword.BENGALI.keyword_id,
        OverallKeyword.CANTONESE.keyword_id,
        OverallKeyword.DANISH.keyword_id,
        OverallKeyword.DUTCH.keyword_id,
        OverallKeyword.FILIPINO.keyword_id,
        OverallKeyword.FINNISH.keyword_id,
        OverallKeyword.FRENCH.keyword_id,
        OverallKeyword.GERMAN.keyword_id,
        OverallKeyword.GREEK.keyword_id,
        OverallKeyword.HINDI.keyword_id,
        OverallKeyword.ITALIAN.keyword_id,
        OverallKeyword.JAPANESE.keyword_id,
        OverallKeyword.KANNADA.keyword_id,
        OverallKeyword.KOREAN.keyword_id,
        OverallKeyword.MALAYALAM.keyword_id,
        OverallKeyword.MANDARIN.keyword_id,
        OverallKeyword.MARATHI.keyword_id,
        OverallKeyword.NORWEGIAN.keyword_id,
        OverallKeyword.PERSIAN.keyword_id,
        OverallKeyword.PORTUGUESE.keyword_id,
        OverallKeyword.PUNJABI.keyword_id,
        OverallKeyword.RUSSIAN.keyword_id,
        OverallKeyword.SPANISH.keyword_id,
        OverallKeyword.SWEDISH.keyword_id,
        OverallKeyword.TAMIL.keyword_id,
        OverallKeyword.TELUGU.keyword_id,
        OverallKeyword.THAI.keyword_id,
        OverallKeyword.TURKISH.keyword_id,
        OverallKeyword.URDU.keyword_id,
    }
)


def country_set(keyword_ids: Iterable[int] | None) -> frozenset[int | str]:
    """Return the country/language bucket set for a movie's keyword_ids.

    A film carrying multiple country/language tags (e.g., a French-Italian
    co-production) keeps both. A film with no matching tag falls back to
    ``{US_DEFAULT}`` — treated as a real bucket by the consensus rule so
    English-language US/UK content can build consensus across anchors.
    """
    if not keyword_ids:
        return frozenset({US_DEFAULT})
    matches = COUNTRY_LANGUAGE_KEYWORD_IDS.intersection(keyword_ids)
    if not matches:
        return frozenset({US_DEFAULT})
    return frozenset(matches)
