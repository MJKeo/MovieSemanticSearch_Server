"""Deterministic MEDIA_TYPE category-call router.

MEDIA_TYPE is already selected upstream by Step 3. This helper only
resolves which non-default ReleaseFormat values the media-type
expressions name, so the patterns can stay broad and phrase-level
rather than defending against arbitrary query text.
"""

from __future__ import annotations

import re

from schemas.enums import ReleaseFormat
from schemas.media_type_translation import MediaTypeQuerySpec
from schemas.step_3 import CategoryCall
from schemas.trait_category import CategoryName


_FORMAT_PATTERNS: tuple[tuple[ReleaseFormat, re.Pattern[str]], ...] = (
    (
        ReleaseFormat.TV_MOVIE,
        re.compile(
            r"\b(?:tv|television|made[-\s]?for[-\s]?(?:tv|television))\b",
            re.IGNORECASE,
        ),
    ),
    (
        ReleaseFormat.SHORT,
        re.compile(r"\b(?:shorts?|short[-\s]?form)\b", re.IGNORECASE),
    ),
    (
        ReleaseFormat.VIDEO,
        re.compile(
            r"\b(?:"
            r"video|"
            r"direct[-\s]?to[-\s]?video|"
            r"straight[-\s]?to[-\s]?video|"
            r"made[-\s]?for[-\s]?video|"
            r"home[-\s]?video|"
            r"home[-\s]?media|"
            r"dvd|"
            r"vhs|"
            r"blu[-\s]?ray|"
            r"vod"
            r")\b",
            re.IGNORECASE,
        ),
    ),
)


def build_media_type_query_spec(
    category_call: CategoryCall,
) -> MediaTypeQuerySpec | None:
    """Resolve a MEDIA_TYPE CategoryCall into endpoint parameters.

    Returns None when the call names only a default/unsupported release
    format, such as "theatrical release" / feature-length MOVIE, which
    the current MediaTypeQuerySpec intentionally cannot represent.
    """
    if category_call.category != CategoryName.MEDIA_TYPE:
        raise ValueError(
            "build_media_type_query_spec only accepts MEDIA_TYPE category calls"
        )

    formats: list[ReleaseFormat] = []
    for expression in category_call.expressions:
        for release_format, pattern in _FORMAT_PATTERNS:
            if release_format in formats:
                continue
            if pattern.search(expression):
                formats.append(release_format)

    if not formats:
        return None

    labels = ", ".join(fmt.name for fmt in formats)
    return MediaTypeQuerySpec(
        thinking=(
            "Deterministic MEDIA_TYPE phrase match resolved "
            f"{labels} from the category-call expressions."
        ),
        formats=formats,
    )
