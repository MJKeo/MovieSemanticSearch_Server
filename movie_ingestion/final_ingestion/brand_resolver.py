# Pure helper that maps a movie's IMDB production-company list to a set of
# production-brand tags.
#
# Contract:
#   - Input: `production_companies` is the IMDB-scraped list of strings (from
#     `IMDBScrapedMovie.production_companies`), in the order IMDB returned
#     them. `release_year` is the integer year of theatrical release, or
#     None when unknown.
#   - Output: a list of `BrandTag(brand_id, first_matching_index)`
#     NamedTuples, one per unique brand that matched at least one string
#     after year-gating. Sorted by `first_matching_index` ascending
#     (earliest appearance first), with `brand_id` as a deterministic
#     tiebreak. `BrandTag` is tuple-compatible — it unpacks and compares
#     equal to `(brand_id, first_matching_index)`.
#
# Why first-matching-index: downstream code wants to know how prominently a
# brand features on the movie (primary producer vs. co-producer etc.). The
# index is the position in the IMDB-returned list, which mirrors how the
# movie presents its credits. Duplicate strings that map to the same brand
# collapse to the smallest index.
#
# Why pure: no DB, no I/O, no config. This keeps it trivially testable and
# safe to call inside any ingest batch. The real-world wiring (reading
# `IMDBScrapedMovie` and `TMDBData.release_date`) lives in the ingestion
# script that imports this function.

from typing import NamedTuple

from schemas.production_brands import memberships_for_string, year_matches


class BrandTag(NamedTuple):
    """A movie's brand tag output by `resolve_brands_for_movie`.

    Tuple-compatible: `BrandTag(1, 0) == (1, 0)` and destructuring
    `brand_id, idx = tag` works. Downstream code can treat the result as
    either a dataclass-ish object (named access) or a plain tuple.
    """

    brand_id: int
    first_matching_index: int


def resolve_brands_for_movie(
    production_companies: list[str],
    release_year: int | None,
) -> list[BrandTag]:
    """Return the brand tags for a movie.

    For each string in `production_companies`, look up every brand that
    claims it as a member. Drop memberships whose year window excludes
    `release_year` (None release_year drops any windowed membership per the
    registry's rule). Dedupe brands across all strings, keeping the smallest
    input index as the "first-matching index" for each brand.

    Returns a list of `BrandTag` sorted by first_matching_index ascending,
    brand_id ascending as tiebreak.
    """
    # brand_id → smallest index at which this brand first matched
    first_index: dict[int, int] = {}
    for idx, raw in enumerate(production_companies):
        for brand, start, end in memberships_for_string(raw):
            if not year_matches(start, end, release_year):
                continue
            # First-seen wins: only record on first encounter, since we
            # iterate idx ascending and want the minimum index per brand.
            if brand.brand_id not in first_index:
                first_index[brand.brand_id] = idx
    return sorted(
        (BrandTag(brand_id, idx) for brand_id, idx in first_index.items()),
        key=lambda t: (t.first_matching_index, t.brand_id),
    )
