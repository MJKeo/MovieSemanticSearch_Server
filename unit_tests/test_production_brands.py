"""Invariant tests for the ProductionBrand registry.

These tests exercise the enum itself — verifying the hand-authored data in
schemas/production_brands.py is internally consistent — and check the
`year_matches` predicate. The companion resolver behavior is tested in
test_brand_resolver.py.
"""

from dataclasses import FrozenInstanceError

import pytest

from schemas.production_brands import (
    BrandCompany,
    ProductionBrand,
    memberships_for_string,
    year_matches,
)
from unit_tests.production_brand_spec_dates import EXPECTED_DATE_MEMBERSHIPS


# ---------------------------------------------------------------------------
# Enum-level invariants
# ---------------------------------------------------------------------------

def test_registry_has_31_brands() -> None:
    """The MVP registry is locked at 31 brands per production_company_tiers.md."""
    assert len(list(ProductionBrand)) == 31


def test_brand_ids_are_unique() -> None:
    ids = [b.brand_id for b in ProductionBrand]
    assert len(ids) == len(set(ids)), f"Duplicate brand_id in: {ids}"


def test_brand_ids_are_positive_ints() -> None:
    for b in ProductionBrand:
        assert isinstance(b.brand_id, int), f"{b.name}.brand_id not int"
        assert b.brand_id > 0, f"{b.name}.brand_id={b.brand_id} must be positive"


def test_slugs_are_unique() -> None:
    slugs = [b.value for b in ProductionBrand]
    assert len(slugs) == len(set(slugs)), f"Duplicate slug in: {slugs}"


def test_display_names_nonempty() -> None:
    for b in ProductionBrand:
        assert b.display_name, f"{b.name} has empty display_name"


def test_every_brand_has_at_least_one_company() -> None:
    for b in ProductionBrand:
        assert b.companies, f"{b.name} has no BrandCompany entries"


def test_company_strings_nonempty() -> None:
    for b in ProductionBrand:
        for c in b.companies:
            assert c.string, f"{b.name} has a BrandCompany with empty string"


def test_year_ranges_are_sane() -> None:
    """For any company with both bounds set, start must be <= end. Years must
    be plausible (1900 <= y <= 2050) — catches typos like swapped digits."""
    for b in ProductionBrand:
        for c in b.companies:
            if c.start_year is not None:
                assert 1900 <= c.start_year <= 2050, (
                    f"{b.name}/{c.string!r} start_year={c.start_year} out of range"
                )
            if c.end_year is not None:
                assert 1900 <= c.end_year <= 2050, (
                    f"{b.name}/{c.string!r} end_year={c.end_year} out of range"
                )
            if c.start_year is not None and c.end_year is not None:
                assert c.start_year <= c.end_year, (
                    f"{b.name}/{c.string!r}: "
                    f"start={c.start_year} > end={c.end_year}"
                )


def test_no_overlapping_windows_within_a_brand() -> None:
    """A surface string may appear multiple times within one brand ONLY if
    each entry's year window is disjoint from the others (e.g. UNITED_ARTISTS
    'United Artists' has a 1919-1981 classic era and a 2024- revival era).
    Overlapping windows would be a data-entry bug — the resolver would treat
    them as redundant, and a reader couldn't tell which era a match refers to.
    """
    for brand in ProductionBrand:
        by_string: dict[str, list[BrandCompany]] = {}
        for company in brand.companies:
            by_string.setdefault(company.string, []).append(company)
        for string, rows in by_string.items():
            if len(rows) < 2:
                continue
            rows_sorted = sorted(
                rows,
                key=lambda r: (
                    float("-inf") if r.start_year is None else r.start_year
                ),
            )
            for prev, curr in zip(rows_sorted, rows_sorted[1:]):
                prev_end = (
                    float("inf") if prev.end_year is None else prev.end_year
                )
                curr_start = (
                    float("-inf")
                    if curr.start_year is None
                    else curr.start_year
                )
                assert prev_end < curr_start, (
                    f"{brand.name}/{string!r}: overlapping windows "
                    f"{(prev.start_year, prev.end_year)} and "
                    f"{(curr.start_year, curr.end_year)}"
                )


def test_reverse_index_covers_every_brandcompany() -> None:
    """Every BrandCompany.string must resolve back to at least its own brand
    via memberships_for_string. Guards against the reverse index being stale
    (e.g. if it were built before an enum member was added)."""
    for b in ProductionBrand:
        for c in b.companies:
            hits = memberships_for_string(c.string)
            assert any(hit_brand is b for hit_brand, _, _ in hits), (
                f"memberships_for_string({c.string!r}) missing {b.name}"
            )


def test_unknown_strings_return_empty_list() -> None:
    """The reverse index must return [] (not raise, not None) for strings it
    has never seen. Callers rely on this."""
    assert memberships_for_string("Not A Real Studio") == []
    assert memberships_for_string("") == []


# ---------------------------------------------------------------------------
# year_matches predicate
# ---------------------------------------------------------------------------

def test_year_matches_both_bounds_none_matches_any_year() -> None:
    assert year_matches(None, None, 1990) is True
    assert year_matches(None, None, 2025) is True
    # Also matches when release_year is None — unconditional membership
    # is the ONE case that applies when we don't know the release date.
    assert year_matches(None, None, None) is True


def test_year_matches_release_year_none_skips_any_windowed() -> None:
    # Either bound set means the membership has an opinion about time, which
    # a release-year-less movie cannot satisfy per the registry rule.
    assert year_matches(2000, None, None) is False
    assert year_matches(None, 2010, None) is False
    assert year_matches(1980, 2005, None) is False


def test_year_matches_start_bound_inclusive() -> None:
    assert year_matches(2000, None, 1999) is False
    assert year_matches(2000, None, 2000) is True
    assert year_matches(2000, None, 2001) is True


def test_year_matches_end_bound_inclusive() -> None:
    assert year_matches(None, 2010, 2009) is True
    assert year_matches(None, 2010, 2010) is True
    assert year_matches(None, 2010, 2011) is False


def test_year_matches_closed_window() -> None:
    # A closed [start, end] window with a year inside, below, above, on edges.
    assert year_matches(2000, 2010, 1999) is False
    assert year_matches(2000, 2010, 2000) is True
    assert year_matches(2000, 2010, 2005) is True
    assert year_matches(2000, 2010, 2010) is True
    assert year_matches(2000, 2010, 2011) is False


# ---------------------------------------------------------------------------
# Exhaustive spec alignment for date-bearing strings.
# ---------------------------------------------------------------------------


def _windowed_registry_strings() -> set[str]:
    return {
        company.string
        for brand in ProductionBrand
        for company in brand.companies
        if company.start_year is not None or company.end_year is not None
    }


def test_expected_date_fixture_covers_every_windowed_registry_string() -> None:
    assert set(EXPECTED_DATE_MEMBERSHIPS) == _windowed_registry_strings()


@pytest.mark.parametrize(
    ("surface", "expected"),
    sorted(EXPECTED_DATE_MEMBERSHIPS.items()),
    ids=sorted(EXPECTED_DATE_MEMBERSHIPS),
)
def test_date_bearing_registry_rows_match_spec(
    surface: str,
    expected: tuple[tuple[str, int | None, int | None], ...],
) -> None:
    hits = [
        (brand.name, start, end)
        for brand, start, end in memberships_for_string(surface)
    ]
    assert hits == list(expected), hits


def test_memberships_for_string_requires_exact_case_and_whitespace() -> None:
    """Registry matching is exact-string only at this layer."""
    assert memberships_for_string("a24") == []
    assert memberships_for_string("A24 ") == []
    assert memberships_for_string(" A24") == []


def test_brandcompany_is_frozen() -> None:
    """BrandCompany is @dataclass(frozen=True) — enforces that the registry
    cannot be mutated after import."""
    c = BrandCompany("Test", 2000, 2010)
    with pytest.raises(FrozenInstanceError):
        c.string = "Other"  # type: ignore[misc]
