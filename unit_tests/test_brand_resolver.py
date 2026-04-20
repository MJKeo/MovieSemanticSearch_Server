"""Behavioral tests for `resolve_brands_for_movie`.

These tests exercise the resolver against the live ProductionBrand registry
(not a mock). That way, a bug in either the helper logic OR the enum data
shows up as a test failure — which matches the user's intent: "actually test
if the code runs as intended rather than building tests based on what it
looks like the code does".
"""

import pytest

from movie_ingestion.final_ingestion.brand_resolver import resolve_brands_for_movie
from schemas.production_brands import ProductionBrand
from unit_tests.production_brand_spec_dates import EXPECTED_DATE_MEMBERSHIPS


# Shorthand lookups used across the tests. Resolving by .name lets each
# assertion express the intended brand semantically, rather than as a magic
# int. If the registry's brand_id assignment ever shifts, tests still pass
# because we compute the expected ids from the enum at test time.
def bid(name: str) -> int:
    return ProductionBrand[name].brand_id


def _spec_year_matches(
    start: int | None,
    end: int | None,
    release_year: int | None,
) -> bool:
    if release_year is None:
        return start is None and end is None
    if start is not None and release_year < start:
        return False
    if end is not None and release_year > end:
        return False
    return True


def _expected_single_string_resolution(
    expected_rows: tuple[tuple[str, int | None, int | None], ...],
    release_year: int | None,
) -> list[tuple[int, int]]:
    matching_brand_ids = {
        bid(brand_name)
        for brand_name, start, end in expected_rows
        if _spec_year_matches(start, end, release_year)
    }
    return [(brand_id, 0) for brand_id in sorted(matching_brand_ids)]


def _interesting_boundary_years(
    expected_rows: tuple[tuple[str, int | None, int | None], ...],
) -> list[int | None]:
    years: set[int | None] = {None}
    for _brand_name, start, end in expected_rows:
        if start is None and end is None:
            continue
        if start is None and end is not None:
            years.update({end, end + 1})
            continue
        if start is not None and end is None:
            years.update({start - 1, start, start + 1})
            continue
        assert start is not None and end is not None
        years.update({start - 1, start, end, end + 1})
    return [None, *sorted(year for year in years if year is not None)]


_DATE_BOUNDARY_CASES: list[tuple[str, int | None, list[tuple[int, int]]]] = []
for _surface, _expected_rows in sorted(EXPECTED_DATE_MEMBERSHIPS.items()):
    for _release_year in _interesting_boundary_years(_expected_rows):
        _DATE_BOUNDARY_CASES.append(
            (
                _surface,
                _release_year,
                _expected_single_string_resolution(_expected_rows, _release_year),
            )
        )


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty() -> None:
    assert resolve_brands_for_movie([], 2015) == []


def test_unknown_strings_return_empty() -> None:
    assert resolve_brands_for_movie(["Some Random Studio"], 2015) == []


# ---------------------------------------------------------------------------
# Basic matching
# ---------------------------------------------------------------------------

def test_single_unconditional_match() -> None:
    out = resolve_brands_for_movie(["A24"], 2015)
    assert out == [(bid("A24"), 0)]


def test_release_year_none_with_unconditional_string_matches() -> None:
    out = resolve_brands_for_movie(["A24"], None)
    assert out == [(bid("A24"), 0)]


def test_single_string_can_match_just_one_brand() -> None:
    """`Walt Disney Pictures` is in DISNEY only (no umbrella overlap for
    that specific string in the current registry). Confirms the baseline
    before the multi-brand tests below."""
    out = resolve_brands_for_movie(["Walt Disney Pictures"], 2015)
    assert out == [(bid("DISNEY"), 0)]


# ---------------------------------------------------------------------------
# Exhaustive spec alignment for single-string date behavior.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("surface", "release_year", "expected"),
    _DATE_BOUNDARY_CASES,
    ids=[
        f"{surface}-{release_year}"
        for surface, release_year, _expected in _DATE_BOUNDARY_CASES
    ],
)
def test_single_string_date_boundaries_match_spec(
    surface: str,
    release_year: int | None,
    expected: list[tuple[int, int]],
) -> None:
    out = resolve_brands_for_movie([surface], release_year)
    assert out == expected


# ---------------------------------------------------------------------------
# Dedup + index tracking
# ---------------------------------------------------------------------------

def test_same_brand_via_multiple_strings_keeps_min_index() -> None:
    """`Walt Disney Pictures` at idx 0 AND `Marvel Studios` at idx 1 both
    hit DISNEY (idx 0 via Walt Disney, idx 1 via Marvel). DISNEY's recorded
    index must be 0 (the lower), not 1."""
    out = resolve_brands_for_movie(
        ["Walt Disney Pictures", "Marvel Studios"], 2020
    )
    by_brand = dict(out)
    assert by_brand[bid("DISNEY")] == 0
    assert by_brand[bid("MARVEL_STUDIOS")] == 1


def test_repeated_string_for_same_brand_uses_earliest_index() -> None:
    """IMDB can repeat a company string (rare but possible). The resolver
    must not let a later index overwrite an earlier one."""
    out = resolve_brands_for_movie(
        ["A24", "A24", "A24"], 2020
    )
    assert out == [(bid("A24"), 0)]


def test_brand_first_appears_at_nonzero_index() -> None:
    """If an unrelated company occupies index 0, the real brand's recorded
    index should be its actual position, not 0."""
    out = resolve_brands_for_movie(
        ["Totally Fake Studio", "A24"], 2020
    )
    assert out == [(bid("A24"), 1)]


def test_year_gated_earlier_hit_does_not_steal_first_index() -> None:
    """An earlier invalid string must not suppress a later valid match for
    the same brand."""
    out = resolve_brands_for_movie(
        ["Republic Pictures", "Paramount Pictures"], 2020
    )
    assert out == [(bid("PARAMOUNT"), 1)]


# ---------------------------------------------------------------------------
# Sort order
# ---------------------------------------------------------------------------

def test_output_is_sorted_by_index_ascending() -> None:
    """Given a list that puts brand X first and brand Y second, X must come
    before Y in the output."""
    out = resolve_brands_for_movie(
        ["Neon", "Studio Ghibli", "A24"], 2020
    )
    # NEON at 0, STUDIO_GHIBLI at 1, A24 at 2 — check ascending order
    indices = [idx for _, idx in out]
    assert indices == sorted(indices)


def test_tiebreak_by_brand_id_ascending() -> None:
    """When two brands share the smallest index (e.g. both match a single
    string), the smaller brand_id sorts first."""
    out = resolve_brands_for_movie(["Pixar Animation Studios"], 2015)
    assert out == [(bid("DISNEY"), 0), (bid("PIXAR"), 0)]


def test_mixed_credits_return_all_applicable_brands() -> None:
    """Realistic multi-company credit list. Every applicable brand should
    appear in the output, each with its first matching index."""
    out = resolve_brands_for_movie(
        ["Universal Pictures", "Mac Guff Ligne"],
        2020,
    )
    by_brand = dict(out)
    # UNIVERSAL matches on both strings (Universal Pictures + Mac Guff Ligne
    # via Illumination-under-Universal membership); first
    # match is idx 0.
    assert by_brand[bid("UNIVERSAL")] == 0
    # ILLUMINATION matches via idx 1.
    assert by_brand[bid("ILLUMINATION")] == 1
