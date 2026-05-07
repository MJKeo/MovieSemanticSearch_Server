"""Curated studio/company signals for the similar-movies flow.

This registry is deliberately narrower than the general studio endpoint.
It contains only production labels whose identity is meaningful for
"movies like X" similarity, plus the era windows from
search_improvement_planning/similar_movies.md. Broad corporate umbrellas
are kept out even when they remain valid for explicit studio search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from implementation.misc.production_company_text import normalize_company_string


StudioSimilarityStrength = Literal["high", "moderate"]

STUDIO_STRENGTH_SCORE: dict[StudioSimilarityStrength, float] = {
    "high": 1.0,
    "moderate": 0.65,
}


@dataclass(frozen=True, slots=True)
class StudioEraWindow:
    start_year: int | None
    end_year: int | None

    @property
    def is_unbounded(self) -> bool:
        return self.start_year is None and self.end_year is None

    def matches(self, release_year: int | None) -> bool:
        if self.is_unbounded:
            return True
        if release_year is None:
            return False
        if self.start_year is not None and release_year < self.start_year:
            return False
        if self.end_year is not None and release_year > self.end_year:
            return False
        return True


@dataclass(frozen=True, slots=True)
class StudioSimilarityEntry:
    label: str
    company_string: str
    normalized_string: str
    strength: StudioSimilarityStrength
    era: StudioEraWindow
    era_sensitive: bool

    @property
    def base_score(self) -> float:
        return STUDIO_STRENGTH_SCORE[self.strength]


def _window(start: int | None = None, end: int | None = None) -> StudioEraWindow:
    return StudioEraWindow(start_year=start, end_year=end)


def _entries(
    *,
    label: str,
    strength: StudioSimilarityStrength,
    strings: Iterable[str],
    eras: Iterable[StudioEraWindow],
    era_sensitive: bool,
) -> list[StudioSimilarityEntry]:
    out: list[StudioSimilarityEntry] = []
    for raw in strings:
        normalized = normalize_company_string(raw)
        if not normalized:
            continue
        for era in eras:
            out.append(
                StudioSimilarityEntry(
                    label=label,
                    company_string=raw,
                    normalized_string=normalized,
                    strength=strength,
                    era=era,
                    era_sensitive=era_sensitive,
                )
            )
    return out


_RAW_ENTRIES: list[StudioSimilarityEntry] = [
    *_entries(
        label="Pixar Animation Studios",
        strength="high",
        strings=("Pixar Animation Studios", "Pixar"),
        eras=(_window(1995, 2010), _window(2011, 2019), _window(2020, None)),
        era_sensitive=True,
    ),
    *_entries(
        label="Walt Disney Animation Studios",
        strength="high",
        strings=("Walt Disney Animation Studios", "Walt Disney Feature Animation"),
        eras=(_window(1989, 1999), _window(2000, 2008), _window(2009, 2019), _window(2020, None)),
        era_sensitive=True,
    ),
    *_entries(
        label="Studio Ghibli",
        strength="high",
        strings=("Studio Ghibli",),
        eras=(_window(),),
        era_sensitive=False,
    ),
    *_entries(
        label="DreamWorks Animation",
        strength="high",
        strings=("DreamWorks Animation", "Pacific Data Images (PDI)"),
        eras=(_window(1998, 2004), _window(2005, 2016), _window(2017, None)),
        era_sensitive=True,
    ),
    *_entries(
        label="Illumination",
        strength="high",
        strings=("Illumination Entertainment", "Illumination Studios Paris"),
        eras=(_window(),),
        era_sensitive=False,
    ),
    *_entries(
        label="Marvel Studios",
        strength="high",
        strings=("Marvel Studios",),
        eras=(_window(2008, 2019), _window(2021, None)),
        era_sensitive=True,
    ),
    *_entries(
        label="DC Studios",
        strength="high",
        strings=("DC Films", "DC Studios"),
        eras=(_window(2016, 2022), _window(2023, None)),
        era_sensitive=True,
    ),
    *_entries(
        label="Lucasfilm",
        strength="high",
        strings=("Lucasfilm", "Lucasfilm Animation"),
        eras=(_window(1977, 1989), _window(1999, 2012), _window(2015, None)),
        era_sensitive=True,
    ),
    *_entries(
        label="A24",
        strength="high",
        strings=("A24",),
        eras=(_window(),),
        era_sensitive=False,
    ),
    *_entries(
        label="Neon",
        strength="high",
        strings=("Neon",),
        eras=(_window(),),
        era_sensitive=False,
    ),
    *_entries(
        label="Blumhouse Productions",
        strength="high",
        strings=("Blumhouse Productions", "Blumhouse International"),
        eras=(_window(2009, 2014), _window(2015, None)),
        era_sensitive=True,
    ),
    *_entries(
        label="Searchlight Pictures",
        strength="high",
        strings=("Searchlight Pictures", "Fox Searchlight Pictures"),
        eras=(_window(),),
        era_sensitive=False,
    ),
    *_entries(
        label="Focus Features",
        strength="high",
        strings=("Focus Features", "Focus Features International (FFI)"),
        eras=(_window(),),
        era_sensitive=False,
    ),
    *_entries(
        label="Focus predecessors",
        strength="high",
        strings=("Good Machine", "Good Machine Films", "USA Films"),
        eras=(_window(),),
        era_sensitive=False,
    ),
    *_entries(
        label="Miramax",
        strength="high",
        strings=("Miramax",),
        eras=(_window(1989, 2005), _window(2006, None)),
        era_sensitive=True,
    ),
    *_entries(
        label="New Line Cinema",
        strength="high",
        strings=(
            "New Line Cinema",
            "New Line Productions",
            "New Line Film",
            "New Line Film Productions",
        ),
        eras=(_window(1984, 1994), _window(1995, 2008), _window(2009, None)),
        era_sensitive=True,
    ),
    *_entries(
        label="Sony Pictures Animation",
        strength="moderate",
        strings=("Sony Pictures Animation",),
        eras=(_window(2006, 2017), _window(2018, None)),
        era_sensitive=True,
    ),
    *_entries(
        label="Atomic Monster",
        strength="moderate",
        strings=("Atomic Monster",),
        eras=(_window(2024, None),),
        era_sensitive=False,
    ),
    *_entries(
        label="Touchstone Pictures",
        strength="moderate",
        strings=("Touchstone Pictures", "Touchstone Films"),
        eras=(_window(1984, 2002), _window(2003, 2016)),
        era_sensitive=True,
    ),
    *_entries(
        label="Screen Gems",
        strength="moderate",
        strings=("Screen Gems",),
        eras=(_window(1999, None),),
        era_sensitive=False,
    ),
    *_entries(
        label="Lionsgate",
        strength="moderate",
        strings=(
            "Lionsgate",
            "Lion's Gate Films",
            "Lions Gate Films",
            "Lions Gate Entertainment",
            "Lions Gate",
            "Lionsgate Premiere",
            "Lionsgate Productions",
            "Lions Gate Studios",
        ),
        eras=(_window(2000, 2011), _window(2012, None)),
        era_sensitive=True,
    ),
    *_entries(
        label="Summit Entertainment",
        strength="moderate",
        strings=("Summit Entertainment", "Summit Premiere"),
        eras=(_window(2012, None),),
        era_sensitive=False,
    ),
    *_entries(
        label="Fox 2000 Pictures",
        strength="moderate",
        strings=("Fox 2000 Pictures",),
        eras=(_window(1994, 2020),),
        era_sensitive=False,
    ),
    *_entries(
        label="Paramount specialty",
        strength="moderate",
        strings=("Paramount Vantage", "Paramount Classics"),
        eras=(_window(2006, 2013),),
        era_sensitive=False,
    ),
    *_entries(
        label="Paramount Animation",
        strength="moderate",
        strings=("Paramount Animation", "Paramount Animation Studios"),
        eras=(_window(2011, None),),
        era_sensitive=False,
    ),
    *_entries(
        label="Netflix Animation",
        strength="moderate",
        strings=("Netflix Animation",),
        eras=(_window(),),
        era_sensitive=False,
    ),
    *_entries(
        label="Apple Original Films",
        strength="moderate",
        strings=("Apple Original Films", "Apple Studios"),
        eras=(_window(),),
        era_sensitive=False,
    ),
    *_entries(
        label="Amazon Studios",
        strength="moderate",
        strings=("Amazon Studios", "Amazon MGM Studios"),
        eras=(_window(None, 2021), _window(2022, None)),
        era_sensitive=True,
    ),
    *_entries(
        label="Metro-Goldwyn-Mayer",
        strength="moderate",
        strings=(
            "Metro-Goldwyn-Mayer (MGM)",
            "Metro-Goldwyn-Mayer (MGM) Studios",
            "Metro-Goldwyn-Mayer Studios",
        ),
        eras=(_window(1924, 1959), _window(1960, 1980), _window(1981, None)),
        era_sensitive=True,
    ),
    *_entries(
        label="United Artists",
        strength="moderate",
        strings=("United Artists", "United Artists Pictures"),
        eras=(_window(1919, 1951), _window(1952, 1981), _window(1982, None)),
        era_sensitive=True,
    ),
    *_entries(
        label="Warner Bros. Animation",
        strength="moderate",
        strings=(
            "Warner Bros. Animation",
            "Warner Bros. Feature Animation",
            "Warner Bros. Cartoon Studios",
        ),
        eras=(_window(),),
        era_sensitive=False,
    ),
]


EXCLUDED_BROAD_STUDIO_STRINGS: frozenset[str] = frozenset(
    normalize_company_string(s)
    for s in (
        "Walt Disney Pictures",
        "Warner Bros. Pictures",
        "Warner Bros.",
        "Universal Pictures",
        "Universal",
        "Paramount Pictures",
        "Paramount",
        "Sony Pictures",
        "Sony Pictures Entertainment",
        "Columbia Pictures",
        "TriStar Pictures",
        "20th Century Fox",
        "20th Century Studios",
        "Netflix",
        "Netflix Studios",
    )
)


SIMILARITY_STUDIO_ENTRIES: tuple[StudioSimilarityEntry, ...] = tuple(
    entry
    for entry in _RAW_ENTRIES
    if entry.normalized_string not in EXCLUDED_BROAD_STUDIO_STRINGS
)


def studio_entries_by_normalized_string() -> dict[str, list[StudioSimilarityEntry]]:
    out: dict[str, list[StudioSimilarityEntry]] = {}
    for entry in SIMILARITY_STUDIO_ENTRIES:
        out.setdefault(entry.normalized_string, []).append(entry)
    return out
