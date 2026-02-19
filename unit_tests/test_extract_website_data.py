"""Unit tests for implementation.scraping.extract_website_data."""

import json

import pytest
from bs4 import BeautifulSoup

from implementation.scraping.extract_website_data import (
    _safe_get,
    _parse_next_data,
    extract_imdb_attributes,
    extract_summary_attributes,
    extract_plot_keywords,
    extract_parental_guide,
    extract_cast_crew,
    extract_featured_reviews,
    get_watch_providers,
)
from implementation.classes.enums import WatchMethodType


# ================================
#          TEST HELPERS
# ================================


def _wrap_next_data(payload: dict) -> str:
    """Wrap a JSON-serializable dict in minimal HTML with a __NEXT_DATA__ script tag."""
    return (
        "<html><body>"
        f'<script id="__NEXT_DATA__" type="application/json">{json.dumps(payload)}</script>'
        "</body></html>"
    )


def _imdb_page(atf_extra: dict | None = None, mcd_extra: dict | None = None) -> str:
    """
    Build minimal valid IMDB page HTML.

    Includes only the two required fields (maturity_rating and overview) in
    aboveTheFoldData by default. Callers merge extra keys into atf or mcd to
    exercise optional-field paths.
    """
    atf = {
        "certificate": {"rating": "PG-13"},
        "plot": {"plotText": {"plainText": "A great movie."}},
    }
    if atf_extra:
        atf.update(atf_extra)
    mcd: dict = {}
    if mcd_extra:
        mcd.update(mcd_extra)
    data = {"props": {"pageProps": {"aboveTheFoldData": atf, "mainColumnData": mcd}}}
    return _wrap_next_data(data)


# ================================
#       _safe_get tests
# ================================


class TestSafeGet:
    """Tests for the _safe_get nested dictionary accessor."""

    def test_returns_value_for_valid_path(self) -> None:
        """_safe_get should traverse nested dicts and return the leaf value."""
        assert _safe_get({"a": {"b": {"c": 42}}}, ["a", "b", "c"]) == 42

    def test_returns_default_for_missing_key(self) -> None:
        """_safe_get should return None (default) when a key is absent."""
        assert _safe_get({"a": {"b": 1}}, ["a", "x", "c"]) is None

    def test_returns_custom_default(self) -> None:
        """_safe_get should return the caller-specified default on missing path."""
        assert _safe_get({"a": 1}, ["a", "b"], default="fallback") == "fallback"

    def test_empty_path_returns_root(self) -> None:
        """_safe_get with an empty path should return the root object itself."""
        data = {"key": "val"}
        assert _safe_get(data, []) == data

    def test_returns_default_when_non_dict_encountered(self) -> None:
        """_safe_get should stop and return default when traversal hits a non-dict."""
        assert _safe_get({"a": "string_not_dict"}, ["a", "b"]) is None

    def test_does_not_traverse_lists_by_index(self) -> None:
        """_safe_get only handles dict traversal; list index access returns default."""
        assert _safe_get({"a": [10, 20]}, ["a", 0], default="nope") == "nope"


# ================================
#     _parse_next_data tests
# ================================


class TestParseNextData:
    """Tests for __NEXT_DATA__ JSON extraction from HTML."""

    def test_extracts_valid_json(self) -> None:
        """Should parse valid JSON from __NEXT_DATA__ script tag."""
        soup = BeautifulSoup(_wrap_next_data({"k": "v"}), "html.parser")
        assert _parse_next_data(soup) == {"k": "v"}

    def test_returns_empty_dict_when_script_tag_missing(self) -> None:
        """Should return {} when there is no __NEXT_DATA__ script tag."""
        soup = BeautifulSoup("<html><body></body></html>", "html.parser")
        assert _parse_next_data(soup) == {}

    def test_returns_empty_dict_for_malformed_json(self) -> None:
        """Should return {} when the script tag contains invalid JSON."""
        html = '<html><body><script id="__NEXT_DATA__" type="application/json">{bad</script></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        assert _parse_next_data(soup) == {}

    def test_returns_empty_dict_for_empty_script_tag(self) -> None:
        """Should return {} when the script tag has no text content."""
        html = '<html><body><script id="__NEXT_DATA__" type="application/json"></script></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        assert _parse_next_data(soup) == {}


# ================================
#  extract_imdb_attributes tests
# ================================


class TestExtractImdbAttributes:
    """Tests for extract_imdb_attributes HTML parsing."""

    def test_minimal_valid_page_extracts_required_fields(self) -> None:
        """Should extract and lowercase the two required fields."""
        result = extract_imdb_attributes(_imdb_page())
        assert result["maturity_rating"] == "pg-13"
        assert result["overview"] == "a great movie."

    def test_raises_when_maturity_rating_missing(self) -> None:
        """Should raise ValueError when certificate/rating is absent."""
        data = {
            "props": {
                "pageProps": {
                    "aboveTheFoldData": {"plot": {"plotText": {"plainText": "ok"}}},
                    "mainColumnData": {},
                }
            }
        }
        with pytest.raises(ValueError, match="maturity_rating"):
            extract_imdb_attributes(_wrap_next_data(data))

    def test_raises_when_overview_missing(self) -> None:
        """Should raise ValueError when plot/plotText/plainText is absent."""
        data = {
            "props": {
                "pageProps": {
                    "aboveTheFoldData": {"certificate": {"rating": "PG"}},
                    "mainColumnData": {},
                }
            }
        }
        with pytest.raises(ValueError, match="overview"):
            extract_imdb_attributes(_wrap_next_data(data))

    def test_extracts_optional_numeric_ratings(self) -> None:
        """Should pass through imdb and metacritic ratings when present."""
        html = _imdb_page(atf_extra={
            "ratingsSummary": {"aggregateRating": 8.5},
            "metacritic": {"metascore": {"score": 90}},
        })
        result = extract_imdb_attributes(html)
        assert result["imdb_rating"] == 8.5
        assert result["metacritic_rating"] == 90

    def test_ratings_default_to_none(self) -> None:
        """Ratings should be None when absent from the payload."""
        result = extract_imdb_attributes(_imdb_page())
        assert result["imdb_rating"] is None
        assert result["metacritic_rating"] is None

    def test_extracts_and_lowercases_genres(self) -> None:
        """Should extract genre text and convert to lowercase."""
        html = _imdb_page(atf_extra={"genres": {"genres": [{"text": "Action"}, {"text": "Drama"}]}})
        assert extract_imdb_attributes(html)["genres"] == ["action", "drama"]

    def test_extracts_keywords_limited_to_eight(self) -> None:
        """Should extract interest keywords and cap at 8."""
        edges = [{"node": {"primaryText": {"text": f"kw{i}"}}} for i in range(12)]
        html = _imdb_page(atf_extra={"interests": {"edges": edges}})
        keywords = extract_imdb_attributes(html)["keywords"]
        assert len(keywords) == 8
        assert keywords[0] == "kw0"

    def test_filters_empty_keywords(self) -> None:
        """Should skip keywords that are empty or whitespace."""
        edges = [
            {"node": {"primaryText": {"text": "valid"}}},
            {"node": {"primaryText": {"text": "  "}}},
            {"node": {"primaryText": {"text": ""}}},
        ]
        html = _imdb_page(atf_extra={"interests": {"edges": edges}})
        assert extract_imdb_attributes(html)["keywords"] == ["valid"]

    def test_extracts_original_title_lowercased(self) -> None:
        """Should extract and lowercase original_title when present."""
        html = _imdb_page(mcd_extra={"originalTitleText": {"text": "Les Intouchables"}})
        assert extract_imdb_attributes(html)["original_title"] == "les intouchables"

    def test_original_title_none_when_absent(self) -> None:
        """original_title should be None when the key is missing."""
        assert extract_imdb_attributes(_imdb_page())["original_title"] is None

    def test_extracts_production_companies(self) -> None:
        """Should extract and lowercase production company names."""
        edges = [{"node": {"company": {"companyText": {"text": "A24"}}}}]
        html = _imdb_page(atf_extra={"production": {"edges": edges}})
        assert extract_imdb_attributes(html)["production_companies"] == ["a24"]

    def test_extracts_countries_of_origin(self) -> None:
        """Should extract and lowercase country names."""
        html = _imdb_page(mcd_extra={"countriesDetails": {"countries": [{"text": "USA"}, {"text": "UK"}]}})
        assert extract_imdb_attributes(html)["countries_of_origin"] == ["usa", "uk"]

    def test_extracts_filming_locations(self) -> None:
        """Should extract and lowercase filming location text."""
        edges = [{"node": {"text": "New York"}}, {"node": {"text": "London"}}]
        html = _imdb_page(mcd_extra={"filmingLocations": {"edges": edges}})
        assert extract_imdb_attributes(html)["filming_locations"] == ["new york", "london"]

    def test_extracts_languages(self) -> None:
        """Should extract and lowercase spoken language text."""
        html = _imdb_page(mcd_extra={"spokenLanguages": {"spokenLanguages": [{"text": "English"}, {"text": "French"}]}})
        assert extract_imdb_attributes(html)["languages"] == ["english", "french"]

    def test_extracts_budget(self) -> None:
        """Should extract the numeric budget amount."""
        html = _imdb_page(mcd_extra={"productionBudget": {"budget": {"amount": 50_000_000}}})
        assert extract_imdb_attributes(html)["budget"] == 50_000_000

    def test_extracts_and_unescapes_review_summary(self) -> None:
        """Should unescape HTML entities and extract plain text for review summary."""
        review_data = {"reviewSummary": {"overall": {"medium": {"value": {"plaidHtml": "<p>Great &amp; fun</p>"}}}}}
        html = _imdb_page(mcd_extra=review_data)
        assert extract_imdb_attributes(html)["user_review_summary"] == "great & fun"

    def test_review_summary_none_when_absent(self) -> None:
        """user_review_summary should be None when not present."""
        assert extract_imdb_attributes(_imdb_page())["user_review_summary"] is None

    def test_extracts_review_themes_with_name_and_sentiment(self) -> None:
        """Should extract and lowercase review theme names and sentiments."""
        themes = [{"label": {"value": "Acting"}, "sentiment": "POSITIVE"}]
        review_data = {"reviewSummary": {"themes": themes}}
        html = _imdb_page(mcd_extra=review_data)
        result = extract_imdb_attributes(html)["review_themes"]
        assert len(result) == 1
        assert result[0].name == "acting"
        assert result[0].sentiment == "positive"

    def test_skips_review_themes_missing_name_or_sentiment(self) -> None:
        """Should filter out review themes with missing name or sentiment."""
        themes = [
            {"label": {"value": "Acting"}, "sentiment": "POSITIVE"},
            {"label": {}, "sentiment": "NEGATIVE"},
            {"label": {"value": "Plot"}},
        ]
        review_data = {"reviewSummary": {"themes": themes}}
        html = _imdb_page(mcd_extra=review_data)
        assert len(extract_imdb_attributes(html)["review_themes"]) == 1


# ================================
#  extract_summary_attributes tests
# ================================


class TestExtractSummaryAttributes:
    """Tests for extract_summary_attributes HTML parsing."""

    def _build_summary_page(self, summaries: list[str], synopses: list[str]) -> str:
        """Build summary page HTML from plain-text lists."""
        data = {
            "props": {
                "pageProps": {
                    "contentData": {
                        "data": {
                            "title": {
                                "plotSummaries": {"edges": [{"node": {"plotText": {"plaidHtml": s}}} for s in summaries]},
                                "plotSynopsis": {"edges": [{"node": {"plotText": {"plaidHtml": s}}} for s in synopses]},
                            }
                        }
                    }
                }
            }
        }
        return _wrap_next_data(data)

    def test_removes_first_summary_as_overview_duplicate(self) -> None:
        """Should drop the first plot summary (duplicate of the overview)."""
        html = self._build_summary_page(["Overview dup", "Real summary"], ["Synopsis text"])
        result = extract_summary_attributes(html)
        assert result["plot_summaries"] == ["real summary"]
        assert result["synopses"] == ["synopsis text"]

    def test_returns_empty_lists_when_no_data(self) -> None:
        """Should return empty lists when summary data is absent."""
        data = {"props": {"pageProps": {"contentData": {"data": {"title": {}}}}}}
        result = extract_summary_attributes(_wrap_next_data(data))
        assert result["plot_summaries"] == []
        assert result["synopses"] == []

    def test_filters_whitespace_only_summaries(self) -> None:
        """Should skip summaries that are empty or whitespace after extraction."""
        html = self._build_summary_page(["First", "  ", "Third"], [])
        result = extract_summary_attributes(html)
        assert result["plot_summaries"] == ["third"]


# ================================
#   extract_plot_keywords tests
# ================================


class TestExtractPlotKeywords:
    """Tests for extract_plot_keywords HTML parsing."""

    def _build_keywords_page(self, keywords: list[str]) -> str:
        """Build plot-keywords page HTML from a keyword list."""
        edges = [{"node": {"keyword": {"text": {"text": k}}}} for k in keywords]
        data = {"props": {"pageProps": {"contentData": {"data": {"title": {"keywords": {"edges": edges}}}}}}}
        return _wrap_next_data(data)

    def test_extracts_and_lowercases(self) -> None:
        """Should extract keyword text and convert to lowercase."""
        result = extract_plot_keywords(self._build_keywords_page(["Superhero", "Adventure"]))
        assert result == ["superhero", "adventure"]

    def test_limits_to_eight(self) -> None:
        """Should cap output at 8 keywords."""
        result = extract_plot_keywords(self._build_keywords_page([f"k{i}" for i in range(15)]))
        assert len(result) == 8

    def test_filters_empty_and_whitespace(self) -> None:
        """Should skip empty/whitespace keyword entries."""
        result = extract_plot_keywords(self._build_keywords_page(["valid", "  ", ""]))
        assert result == ["valid"]

    def test_returns_empty_when_data_absent(self) -> None:
        """Should return empty list when keywords section is missing."""
        result = extract_plot_keywords(_wrap_next_data({"props": {"pageProps": {}}}))
        assert result == []


# ================================
#  extract_parental_guide tests
# ================================


class TestExtractParentalGuide:
    """Tests for extract_parental_guide HTML parsing."""

    def _build_guide_page(self, reasons: list[str], categories: list[dict]) -> str:
        """Build parental-guide page HTML from reason strings and category dicts."""
        data = {
            "props": {
                "pageProps": {
                    "contentData": {
                        "data": {
                            "title": {
                                "ratingReason": {"edges": [{"node": {"ratingReason": r}} for r in reasons]},
                                "parentsGuide": {"categories": categories},
                            }
                        }
                    }
                }
            }
        }
        return _wrap_next_data(data)

    def test_extracts_reasons_and_guide_items(self) -> None:
        """Should extract rating reasons and formatted parental guide items."""
        cats = [{"category": {"text": "Violence"}, "severity": {"text": "Moderate"}}]
        result = extract_parental_guide(self._build_guide_page(["Rated PG-13 for action"], cats))
        assert result["ratingReasons"] == ["rated pg-13 for action"]
        assert result["parentsGuide"] == [{"category": "violence", "severity": "moderate"}]

    def test_filters_none_severity(self) -> None:
        """Should exclude guide items whose severity is 'None'."""
        cats = [
            {"category": {"text": "Violence"}, "severity": {"text": "Moderate"}},
            {"category": {"text": "Nudity"}, "severity": {"text": "None"}},
        ]
        result = extract_parental_guide(self._build_guide_page([], cats))
        assert len(result["parentsGuide"]) == 1

    def test_filters_empty_category_or_severity(self) -> None:
        """Should exclude guide items with empty category or severity text."""
        cats = [
            {"category": {"text": ""}, "severity": {"text": "Moderate"}},
            {"category": {"text": "Violence"}, "severity": {"text": ""}},
            {"category": {"text": "Language"}, "severity": {"text": "Mild"}},
        ]
        result = extract_parental_guide(self._build_guide_page([], cats))
        assert len(result["parentsGuide"]) == 1
        assert result["parentsGuide"][0]["category"] == "language"

    def test_returns_empty_for_missing_data(self) -> None:
        """Should return empty lists when parental guide sections are absent."""
        data = {"props": {"pageProps": {"contentData": {"data": {"title": {}}}}}}
        result = extract_parental_guide(_wrap_next_data(data))
        assert result["ratingReasons"] == []
        assert result["parentsGuide"] == []


# ================================
#    extract_cast_crew tests
# ================================


class TestExtractCastCrew:
    """Tests for extract_cast_crew HTML parsing."""

    def _build_cast_page(self, categories: list[dict]) -> str:
        """Build cast/crew page HTML from category grouping dicts."""
        data = {"props": {"pageProps": {"contentData": {"categories": categories}}}}
        return _wrap_next_data(data)

    def test_extracts_all_crew_categories(self) -> None:
        """Should extract directors, writers, cast, characters, producers, and composers."""
        cats = [
            {"name": "Director", "section": {"items": [{"rowTitle": "Christopher Nolan"}]}},
            {"name": "Writers", "section": {"items": [{"rowTitle": "Chris Nolan"}, {"rowTitle": "Jonathan Nolan"}]}},
            {
                "name": "Cast",
                "section": {
                    "items": [
                        {"rowTitle": "Leonardo DiCaprio", "characters": ["Cobb"]},
                        {"rowTitle": "Tom Hardy", "characters": ["Eames"]},
                    ],
                    "splitIndex": 1,
                },
            },
            {"name": "Producers", "section": {"items": [{"rowTitle": "Emma Thomas"}]}},
            {"name": "Composer", "section": {"items": [{"rowTitle": "Hans Zimmer"}]}},
        ]
        result = extract_cast_crew(self._build_cast_page(cats))
        assert set(result["directors"]) == {"christopher nolan"}
        assert set(result["writers"]) == {"chris nolan", "jonathan nolan"}
        assert result["cast"] == ["leonardo dicaprio", "tom hardy"]
        assert result["characters"] == ["cobb", "eames"]
        assert result["producers"] == ["emma thomas"]
        assert set(result["composers"]) == {"hans zimmer"}

    def test_split_index_limits_cast_list(self) -> None:
        """Should truncate cast to splitIndex + 1 entries."""
        cats = [
            {
                "name": "Cast",
                "section": {
                    "items": [
                        {"rowTitle": "A1", "characters": ["C1"]},
                        {"rowTitle": "A2", "characters": ["C2"]},
                        {"rowTitle": "A3", "characters": ["C3"]},
                    ],
                    "splitIndex": 1,
                },
            },
        ]
        result = extract_cast_crew(self._build_cast_page(cats))
        assert result["cast"] == ["a1", "a2"]
        assert result["characters"] == ["c1", "c2"]

    def test_deduplicates_directors_via_set(self) -> None:
        """Director set should collapse duplicate names."""
        cats = [{"name": "Director", "section": {"items": [{"rowTitle": "Joel Coen"}, {"rowTitle": "Joel Coen"}]}}]
        result = extract_cast_crew(self._build_cast_page(cats))
        assert result["directors"] == ["joel coen"]

    def test_filters_empty_names(self) -> None:
        """Should skip entries whose rowTitle is empty or whitespace."""
        cats = [{"name": "Director", "section": {"items": [{"rowTitle": "  "}, {"rowTitle": "Valid Director"}]}}]
        result = extract_cast_crew(self._build_cast_page(cats))
        assert result["directors"] == ["valid director"]

    def test_returns_empty_lists_when_no_categories(self) -> None:
        """Should return empty lists for all fields when no categories are present."""
        result = extract_cast_crew(self._build_cast_page([]))
        for key in ("directors", "writers", "cast", "characters", "producers", "composers"):
            assert result[key] == [], f"Expected empty list for {key}"


# ================================
#  extract_featured_reviews tests
# ================================


class TestExtractFeaturedReviews:
    """Tests for extract_featured_reviews HTML parsing."""

    def _build_reviews_page(self, reviews: list[dict]) -> str:
        """Build reviews page HTML from review dicts (each with summary/text keys)."""
        wrapped = [{"review": r} for r in reviews]
        data = {"props": {"pageProps": {"contentData": {"reviews": wrapped}}}}
        return _wrap_next_data(data)

    def test_extracts_reviews_with_summary_and_text(self) -> None:
        """Should include reviews that have both summary and text."""
        reviews = [
            {"reviewSummary": "Great film", "reviewText": "Really enjoyed it."},
            {"reviewSummary": "Decent", "reviewText": "It was okay."},
        ]
        result = extract_featured_reviews(self._build_reviews_page(reviews))
        assert len(result) == 2
        assert result[0]["summary"] == "Great film"

    def test_filters_reviews_missing_summary_or_text(self) -> None:
        """Should skip reviews where summary or text is None."""
        reviews = [
            {"reviewSummary": "Valid", "reviewText": "Valid text"},
            {"reviewSummary": None, "reviewText": "Missing summary"},
            {"reviewSummary": "Missing text", "reviewText": None},
        ]
        result = extract_featured_reviews(self._build_reviews_page(reviews))
        assert len(result) == 1

    def test_limits_to_ten_reviews(self) -> None:
        """Should cap output at 10 reviews."""
        reviews = [{"reviewSummary": f"S{i}", "reviewText": f"T{i}"} for i in range(15)]
        result = extract_featured_reviews(self._build_reviews_page(reviews))
        assert len(result) == 10

    def test_returns_empty_when_reviews_absent(self) -> None:
        """Should return empty list when reviews section is missing."""
        data = {"props": {"pageProps": {"contentData": {}}}}
        assert extract_featured_reviews(_wrap_next_data(data)) == []


# ================================
#   get_watch_providers tests
# ================================


class TestGetWatchProviders:
    """Tests for get_watch_providers TMDB data extraction."""

    def test_merges_types_for_same_provider_id(self) -> None:
        """Should combine watch method types when a provider appears in multiple categories."""
        provider = {"provider_id": 8, "provider_name": "Netflix", "logo_path": "/n.png", "display_priority": 1}
        data = {"results": {"US": {"flatrate": [provider], "buy": [provider], "rent": [provider]}}}
        result = get_watch_providers(data)
        assert len(result) == 1
        assert set(result[0].types) == {
            WatchMethodType.SUBSCRIPTION,
            WatchMethodType.PURCHASE,
            WatchMethodType.RENT,
        }

    def test_separates_different_provider_ids(self) -> None:
        """Should create distinct providers for different provider IDs."""
        data = {
            "results": {
                "US": {
                    "flatrate": [{"provider_id": 8, "provider_name": "Netflix", "logo_path": "/n.png", "display_priority": 1}],
                    "buy": [{"provider_id": 2, "provider_name": "Apple TV", "logo_path": "/a.png", "display_priority": 2}],
                }
            }
        }
        result = get_watch_providers(data)
        assert len(result) == 2
        assert {p.name for p in result} == {"netflix", "apple tv"}

    def test_skips_providers_with_missing_required_fields(self) -> None:
        """Should skip providers missing provider_id, provider_name, or display_priority."""
        providers = [
            {"provider_id": None, "provider_name": "Bad1", "logo_path": "/x.png", "display_priority": 1},
            {"provider_id": 1, "provider_name": None, "logo_path": "/x.png", "display_priority": 1},
            {"provider_id": 2, "provider_name": "Bad3", "logo_path": "/x.png", "display_priority": None},
            {"provider_id": 3, "provider_name": "Valid", "logo_path": "/v.png", "display_priority": 1},
        ]
        result = get_watch_providers({"results": {"US": {"flatrate": providers}}})
        assert len(result) == 1
        assert result[0].name == "valid"

    def test_returns_empty_when_us_key_missing(self) -> None:
        """Should return empty list when 'US' key is absent from results."""
        assert get_watch_providers({"results": {}}) == []

    def test_returns_empty_for_completely_empty_input(self) -> None:
        """Should return empty list when input is an empty dict."""
        assert get_watch_providers({}) == []

    def test_handles_missing_category_keys_gracefully(self) -> None:
        """Should work when some category keys (flatrate, buy, rent) are absent."""
        data = {
            "results": {
                "US": {
                    "flatrate": [{"provider_id": 5, "provider_name": "Hulu", "logo_path": "/h.png", "display_priority": 3}],
                }
            }
        }
        result = get_watch_providers(data)
        assert len(result) == 1
        assert result[0].types == [WatchMethodType.SUBSCRIPTION]

    def test_lowercases_provider_names(self) -> None:
        """Should lowercase provider names."""
        data = {
            "results": {
                "US": {
                    "flatrate": [{"provider_id": 1, "provider_name": "HBO Max", "logo_path": "/h.png", "display_priority": 1}],
                }
            }
        }
        assert get_watch_providers(data)[0].name == "hbo max"
