"""
Unit tests for movie_ingestion.imdb_scraping.parsers.

Tests the GraphQL response transformer and its helper functions (_safe_get,
_extract_plain_text, _extract_synopses_and_summaries, _score_and_filter_keywords).
All functions are pure and operate on dicts (parsed JSON from GraphQL responses).
"""

from movie_ingestion.imdb_scraping.parsers import (
    _safe_get,
    _extract_plain_text,
    _extract_synopses_and_summaries,
    _score_and_filter_keywords,
    transform_graphql_response,
)
from movie_ingestion.imdb_scraping.models import (
    ReviewTheme,
    ParentalGuideItem,
    FeaturedReview,
    IMDBScrapedMovie,
)


# ---------------------------------------------------------------------------
# GraphQL fixture builder
# ---------------------------------------------------------------------------


def _graphql_title_data(**overrides) -> dict:
    """
    Build a minimal valid GraphQL title_data dict with sensible defaults.

    All top-level fields default to values that produce non-empty output
    for the scalar fields and empty lists for the collection fields.
    Callers override specific keys to test individual extraction paths.
    """
    base = {
        "originalTitleText": {"text": "Test Movie"},
        "ratingsSummary": {"aggregateRating": 7.5, "voteCount": 1000},
        "certificate": {"rating": "R", "ratingReason": None},
        "plot": {"plotText": {"plainText": "A test plot."}},
        "metacritic": {"metascore": {"score": 72}},
        "productionBudget": {"budget": {"amount": 50_000_000}},
        "titleGenres": {"genres": []},
        "interests": {"edges": []},
        "countriesOfOrigin": {"countries": []},
        "filmingLocations": {"edges": []},
        "spokenLanguages": {"spokenLanguages": []},
        "companyCredits": {"edges": []},
        "reviewSummary": {"overall": None, "themes": []},
        "plots": {"edges": []},
        "keywords": {"edges": []},
        "parentsGuide": {"categories": []},
        "directors": {"edges": []},
        "writers": {"edges": []},
        "cast": {"edges": []},
        "producers": {"edges": []},
        "composers": {"edges": []},
        "reviews": {"edges": []},
    }
    base.update(overrides)
    return base


def _keyword_edges(keywords: list[tuple[str, int, int]]) -> list[dict]:
    """
    Build GraphQL keyword edges from (text, usersInterested, usersVoted) tuples.

    Matches the structure returned by the IMDB GraphQL API.
    """
    return [
        {
            "node": {
                "keyword": {"text": {"text": text}},
                "interestScore": {
                    "usersInterested": interested,
                    "usersVoted": voted,
                },
            }
        }
        for text, interested, voted in keywords
    ]


def _plot_edges(
    synopses: list[str] | None = None,
    summaries: list[str] | None = None,
) -> list[dict]:
    """
    Build GraphQL plot edges with plotType differentiation.

    Synopses get plotType="SYNOPSIS", summaries get plotType="SUMMARY".
    """
    edges = []
    for text in (synopses or []):
        edges.append({
            "node": {"plotText": {"plainText": text}, "plotType": "SYNOPSIS"}
        })
    for text in (summaries or []):
        edges.append({
            "node": {"plotText": {"plainText": text}, "plotType": "SUMMARY"}
        })
    return edges


# ---------------------------------------------------------------------------
# Tests: _safe_get
# ---------------------------------------------------------------------------


class TestSafeGet:
    """Tests for the _safe_get nested dict traversal helper."""

    def test_returns_value_for_valid_path(self) -> None:
        """Traverses nested dicts and returns the leaf value."""
        obj = {"a": {"b": {"c": 42}}}
        assert _safe_get(obj, ["a", "b", "c"]) == 42

    def test_returns_default_for_missing_key(self) -> None:
        """Returns None (default) when an intermediate key is absent."""
        obj = {"a": {"b": 1}}
        assert _safe_get(obj, ["a", "x", "c"]) is None

    def test_returns_custom_default(self) -> None:
        """Returns caller-specified default on missing path."""
        obj = {"a": 1}
        assert _safe_get(obj, ["missing"], default="fallback") == "fallback"

    def test_empty_path_returns_root(self) -> None:
        """Empty path returns the root object unchanged."""
        obj = {"a": 1}
        assert _safe_get(obj, []) == obj

    def test_returns_default_when_non_dict_encountered(self) -> None:
        """Stops and returns default when traversal hits a non-dict value."""
        obj = {"a": "not_a_dict"}
        assert _safe_get(obj, ["a", "b"]) is None

    def test_none_intermediate_returns_default(self) -> None:
        """Returns default when an intermediate value is None."""
        obj = {"a": None}
        assert _safe_get(obj, ["a", "b"]) is None

    def test_none_root_returns_default(self) -> None:
        """Returns default when root object is None."""
        assert _safe_get(None, ["a"]) is None


# ---------------------------------------------------------------------------
# Tests: _extract_plain_text
# ---------------------------------------------------------------------------


class TestExtractPlainText:
    """Tests for IMDB plaidHtml to plain text conversion."""

    def test_unescapes_html_entities(self) -> None:
        """HTML entities like &amp; are converted to their characters."""
        result = _extract_plain_text("Tom &amp; Jerry")
        assert result == "Tom & Jerry"

    def test_strips_html_tags(self) -> None:
        """HTML tags are removed, leaving only text."""
        result = _extract_plain_text("<p>Hello <b>World</b></p>")
        assert result == "Hello World"

    def test_returns_none_for_none_input(self) -> None:
        """None input returns None."""
        assert _extract_plain_text(None) is None

    def test_returns_none_for_empty_string(self) -> None:
        """Empty string input returns None."""
        assert _extract_plain_text("") is None

    def test_returns_none_for_whitespace_only(self) -> None:
        """Whitespace-only input returns None."""
        assert _extract_plain_text("   ") is None

    def test_returns_none_for_non_string_input(self) -> None:
        """Non-string input (e.g., int) returns None."""
        assert _extract_plain_text(123) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests: _extract_synopses_and_summaries
# ---------------------------------------------------------------------------


class TestSynopsisPriority:
    """Tests for the synopsis/summary priority extraction logic."""

    def test_synopses_present_takes_first_only(self) -> None:
        """When synopses exist, returns [first_synopsis] with plot_summaries=[]."""
        edges = _plot_edges(
            synopses=["First synopsis full text.", "Second synopsis."]
        )
        synopses, summaries = _extract_synopses_and_summaries(edges)
        assert synopses == ["First synopsis full text."]
        assert summaries == []

    def test_synopses_present_ignores_subsequent(self) -> None:
        """Multiple synopses still return only the first one."""
        edges = _plot_edges(synopses=["A", "B", "C"])
        synopses, summaries = _extract_synopses_and_summaries(edges)
        assert len(synopses) == 1
        assert synopses[0] == "A"

    def test_no_synopses_falls_back_to_summaries(self) -> None:
        """With no synopses, returns synopses=[] and populates plot_summaries."""
        edges = _plot_edges(
            summaries=["Overview dup", "Summary 1", "Summary 2", "Summary 3"]
        )
        synopses, summaries = _extract_synopses_and_summaries(edges)
        assert synopses == []
        assert summaries == ["Summary 1", "Summary 2", "Summary 3"]

    def test_summaries_skip_first_as_overview_duplicate(self) -> None:
        """First summary entry is dropped (it duplicates the main page overview)."""
        edges = _plot_edges(summaries=["Overview text", "Unique summary"])
        synopses, summaries = _extract_synopses_and_summaries(edges)
        assert summaries == ["Unique summary"]

    def test_summaries_limited_to_three(self) -> None:
        """At most 3 summaries returned (after skipping first)."""
        edges = _plot_edges(summaries=["Dup", "S1", "S2", "S3", "S4", "S5"])
        synopses, summaries = _extract_synopses_and_summaries(edges)
        assert len(summaries) == 3
        assert summaries == ["S1", "S2", "S3"]

    def test_both_empty_returns_empty_lists(self) -> None:
        """No synopses and no summaries returns both fields as empty lists."""
        synopses, summaries = _extract_synopses_and_summaries([])
        assert synopses == []
        assert summaries == []

    def test_none_input_returns_empty_lists(self) -> None:
        """None input returns both fields as empty lists."""
        synopses, summaries = _extract_synopses_and_summaries(None)
        assert synopses == []
        assert summaries == []

    def test_single_summary_only_returns_empty(self) -> None:
        """One summary entry (the overview duplicate) results in empty plot_summaries."""
        edges = _plot_edges(summaries=["Only the overview duplicate"])
        synopses, summaries = _extract_synopses_and_summaries(edges)
        assert summaries == []

    def test_whitespace_only_entries_filtered(self) -> None:
        """Whitespace-only plainText entries are skipped."""
        edges = [
            {"node": {"plotText": {"plainText": "   "}, "plotType": "SYNOPSIS"}},
            {"node": {"plotText": {"plainText": "  "}, "plotType": "SYNOPSIS"}},
        ]
        synopses, summaries = _extract_synopses_and_summaries(edges)
        assert synopses == []
        assert summaries == []

    def test_synopsis_priority_over_summaries(self) -> None:
        """When both types exist, synopses take priority and summaries are dropped."""
        edges = _plot_edges(
            synopses=["The full synopsis."],
            summaries=["Overview", "Summary 1", "Summary 2"],
        )
        synopses, summaries = _extract_synopses_and_summaries(edges)
        assert synopses == ["The full synopsis."]
        assert summaries == []


# ---------------------------------------------------------------------------
# Tests: _score_and_filter_keywords
# ---------------------------------------------------------------------------


class TestKeywordScoring:
    """Tests for the keyword vote-based scoring and filtering logic."""

    # --- Scoring formula tests ---

    def test_score_formula_basic(self) -> None:
        """score = interested - 0.75 * (voted - interested) verified with concrete values."""
        # kw1: 15 interested, 20 voted → score = 15 - 0.75*5 = 11.25
        # kw2: 5 interested, 10 voted → score = 5 - 0.75*5 = 1.25
        # N = 11.25, threshold = min(8.4375, 9.25) = 8.4375
        # Only kw1 passes, but floor of 5 pads kw2 in
        edges = _keyword_edges([
            ("keyword_a", 15, 20),
            ("keyword_b", 5, 10),
        ])
        result = _score_and_filter_keywords(edges)
        assert "keyword_a" in result
        assert "keyword_b" in result

    def test_score_fifty_fifty_split_is_positive(self) -> None:
        """50/50 split keyword (10 interested, 20 voted) scores 2.5 — still positive."""
        edges = _keyword_edges([
            ("top_keyword", 20, 20),   # score = 20
            ("contested", 10, 20),      # score = 2.5
        ])
        result = _score_and_filter_keywords(edges)
        # Floor of 5 pads "contested" in since only 2 total
        assert "contested" in result

    def test_score_all_interested(self) -> None:
        """All voters interested: score equals usersInterested."""
        edges = _keyword_edges([("loved_keyword", 10, 10)])
        result = _score_and_filter_keywords(edges)
        assert result == ["loved_keyword"]

    def test_score_all_dislikes(self) -> None:
        """All voters disliked: score is negative. N <= 0 → first 5 by position."""
        edges = _keyword_edges([
            ("disliked_1", 0, 10),
            ("disliked_2", 0, 5),
            ("disliked_3", 0, 3),
        ])
        result = _score_and_filter_keywords(edges)
        assert result == ["disliked_1", "disliked_2", "disliked_3"]

    # --- Threshold logic tests ---

    def test_threshold_popular_movie(self) -> None:
        """N=20: threshold = min(15, 18) = 15. Only high-scoring keywords pass."""
        edges = _keyword_edges([
            ("top", 20, 20),       # score = 20 → passes
            ("high", 16, 16),      # score = 16 → passes (>= 15)
            ("medium", 10, 10),    # score = 10 → fails (< 15)
            ("low", 5, 5),         # score = 5 → fails
            ("lowest", 2, 2),      # score = 2 → fails
        ])
        result = _score_and_filter_keywords(edges)
        # 2 pass threshold, floor of 5 pads the rest
        assert len(result) == 5
        assert "top" in result
        assert "high" in result

    def test_threshold_low_engagement(self) -> None:
        """N=4: threshold = min(3, 2) = 2. Wider absolute band."""
        edges = _keyword_edges([
            ("best", 4, 4),   # score = 4 → passes (>= 2)
            ("good", 3, 3),   # score = 3 → passes (>= 2)
            ("ok", 2, 2),     # score = 2 → passes (>= 2)
            ("weak", 1, 1),   # score = 1 → fails (< 2)
        ])
        result = _score_and_filter_keywords(edges)
        # 3 pass threshold, floor of 5 pads to include "weak"
        assert "best" in result
        assert "good" in result
        assert "ok" in result
        assert "weak" in result

    def test_max_score_zero_takes_first_five(self) -> None:
        """All scores = 0: returns first 5 by position (no meaningful vote signal)."""
        edges = _keyword_edges([(f"kw_{i}", 0, 0) for i in range(8)])
        result = _score_and_filter_keywords(edges)
        assert len(result) == 5
        assert result == [f"kw_{i}" for i in range(5)]

    def test_max_score_negative_takes_first_five(self) -> None:
        """All scores negative: returns first 5 by position."""
        edges = _keyword_edges([(f"neg_{i}", 0, 5) for i in range(7)])
        result = _score_and_filter_keywords(edges)
        assert len(result) == 5
        assert result == [f"neg_{i}" for i in range(5)]

    # --- Floor and cap tests ---

    def test_floor_of_five_pads_from_below_threshold(self) -> None:
        """Fewer than 5 keywords pass threshold: pads with next-highest below-threshold."""
        edges = _keyword_edges([
            ("pass_1", 20, 20),    # score=20 → passes
            ("pass_2", 16, 16),    # score=16 → passes
            ("fail_1", 14, 14),    # score=14 → fails but highest below
            ("fail_2", 10, 10),    # score=10 → fails
            ("fail_3", 5, 5),      # score=5 → fails
            ("fail_4", 1, 1),      # score=1 → fails
        ])
        result = _score_and_filter_keywords(edges)
        assert len(result) == 5
        assert "pass_1" in result
        assert "pass_2" in result
        assert "fail_1" in result  # Padded (highest below threshold)

    def test_floor_of_five_with_fewer_total(self) -> None:
        """Fewer than 5 total keywords returns all of them."""
        edges = _keyword_edges([
            ("only_1", 5, 5),
            ("only_2", 3, 3),
            ("only_3", 1, 1),
        ])
        result = _score_and_filter_keywords(edges)
        assert len(result) == 3

    def test_cap_of_fifteen(self) -> None:
        """More than 15 keywords pass threshold: capped to top 15 by score."""
        edges = _keyword_edges([(f"kw_{i}", 20, 20) for i in range(20)])
        result = _score_and_filter_keywords(edges)
        assert len(result) == 15

    # --- Edge cases ---

    def test_empty_keywords_returns_empty(self) -> None:
        """No keyword edges returns []."""
        assert _score_and_filter_keywords([]) == []

    def test_none_keywords_returns_empty(self) -> None:
        """None input returns []."""
        assert _score_and_filter_keywords(None) == []

    def test_whitespace_keywords_filtered(self) -> None:
        """Blank/whitespace keyword texts are skipped before scoring."""
        edges = _keyword_edges([
            ("valid", 5, 5),
            ("   ", 10, 10),
        ])
        result = _score_and_filter_keywords(edges)
        assert "valid" in result
        assert "   " not in result

    def test_duplicate_keywords_deduplicated(self) -> None:
        """Duplicate keyword texts are deduplicated while preserving order."""
        edges = _keyword_edges([
            ("twist ending", 10, 10),
            ("twist ending", 8, 8),
            ("revenge", 5, 5),
        ])
        result = _score_and_filter_keywords(edges)
        assert result.count("twist ending") == 1

    def test_missing_interest_score_defaults_to_zero(self) -> None:
        """Missing interestScore block defaults to 0 interested, 0 voted."""
        edges = [
            {"node": {"keyword": {"text": {"text": f"kw_{i}"}}}}
            for i in range(6)
        ]
        result = _score_and_filter_keywords(edges)
        # All scores are 0, so N=0, first 5 by position
        assert len(result) == 5

    def test_realistic_keyword_set(self) -> None:
        """Realistic keyword set with varied vote data produces a sensible subset."""
        edges = _keyword_edges([
            ("time travel", 50, 55),        # score = 46.25
            ("plot twist", 45, 50),          # score = 41.25
            ("revenge", 30, 35),             # score = 26.25
            ("hero", 20, 25),                # score = 16.25
            ("dystopia", 15, 18),            # score = 12.75
            ("explosion", 10, 15),           # score = 6.25
            ("based on novel", 8, 12),       # score = 5.0
            ("CGI", 5, 10),                  # score = 1.25
            ("surprise ending", 3, 8),       # score = -0.75
        ])
        result = _score_and_filter_keywords(edges)
        # N=46.25, threshold=min(34.6875, 44.25)=34.6875
        # Only "time travel" and "plot twist" pass → floor pads to 5
        assert len(result) == 5
        assert "time travel" in result
        assert "plot twist" in result
        assert "revenge" in result


# ---------------------------------------------------------------------------
# Tests: transform_graphql_response
# ---------------------------------------------------------------------------


class TestTransformGraphqlResponse:
    """Tests for the main GraphQL-to-IMDBScrapedMovie transformer."""

    # --- Scalar field extraction ---

    def test_extracts_original_title(self) -> None:
        """originalTitleText.text mapped to original_title."""
        data = _graphql_title_data(originalTitleText={"text": "Fight Club"})
        result = transform_graphql_response(data)
        assert result.original_title == "Fight Club"

    def test_extracts_maturity_rating(self) -> None:
        """certificate.rating mapped to maturity_rating."""
        data = _graphql_title_data(certificate={"rating": "R", "ratingReason": None})
        result = transform_graphql_response(data)
        assert result.maturity_rating == "R"

    def test_extracts_overview(self) -> None:
        """plot.plotText.plainText mapped to overview."""
        data = _graphql_title_data(plot={"plotText": {"plainText": "A plot summary."}})
        result = transform_graphql_response(data)
        assert result.overview == "A plot summary."

    def test_extracts_imdb_rating(self) -> None:
        """ratingsSummary.aggregateRating mapped to imdb_rating."""
        data = _graphql_title_data(ratingsSummary={"aggregateRating": 8.8, "voteCount": 2500000})
        result = transform_graphql_response(data)
        assert result.imdb_rating == 8.8

    def test_extracts_imdb_vote_count(self) -> None:
        """ratingsSummary.voteCount mapped to imdb_vote_count as int."""
        data = _graphql_title_data(ratingsSummary={"aggregateRating": 7.0, "voteCount": 123456})
        result = transform_graphql_response(data)
        assert result.imdb_vote_count == 123456

    def test_vote_count_defaults_to_zero(self) -> None:
        """Missing voteCount defaults to 0."""
        data = _graphql_title_data(ratingsSummary={"aggregateRating": 7.0})
        result = transform_graphql_response(data)
        assert result.imdb_vote_count == 0

    def test_extracts_metacritic_rating(self) -> None:
        """metacritic.metascore.score mapped to metacritic_rating."""
        data = _graphql_title_data(metacritic={"metascore": {"score": 66}})
        result = transform_graphql_response(data)
        assert result.metacritic_rating == 66

    def test_extracts_budget(self) -> None:
        """productionBudget.budget.amount mapped to budget."""
        data = _graphql_title_data(productionBudget={"budget": {"amount": 63_000_000}})
        result = transform_graphql_response(data)
        assert result.budget == 63_000_000

    def test_whitespace_stripping_on_string_fields(self) -> None:
        """Leading/trailing whitespace stripped from text fields."""
        data = _graphql_title_data(
            originalTitleText={"text": "  Title  "},
            certificate={"rating": "  R  ", "ratingReason": None},
            plot={"plotText": {"plainText": "  A plot.  "}},
        )
        result = transform_graphql_response(data)
        assert result.original_title == "Title"
        assert result.maturity_rating == "R"
        assert result.overview == "A plot."

    def test_none_scalar_fields_when_missing(self) -> None:
        """Missing optional scalar fields produce None."""
        data = _graphql_title_data(
            originalTitleText={},
            certificate={},
            plot={},
            ratingsSummary={},
            metacritic={},
            productionBudget={},
        )
        result = transform_graphql_response(data)
        assert result.original_title is None
        assert result.maturity_rating is None
        assert result.overview is None
        assert result.imdb_rating is None
        assert result.metacritic_rating is None
        assert result.budget is None

    # --- List field extraction ---

    def test_extracts_genres(self) -> None:
        """titleGenres.genres[].genre.text mapped to genres list."""
        data = _graphql_title_data(
            titleGenres={"genres": [
                {"genre": {"text": "Drama"}},
                {"genre": {"text": "Thriller"}},
            ]}
        )
        result = transform_graphql_response(data)
        assert result.genres == ["Drama", "Thriller"]

    def test_extracts_overall_keywords(self) -> None:
        """interests.edges[].node.primaryText.text mapped to overall_keywords (capped at 8)."""
        edges = [
            {"node": {"primaryText": {"text": f"Keyword {i}"}}}
            for i in range(10)
        ]
        data = _graphql_title_data(interests={"edges": edges})
        result = transform_graphql_response(data)
        assert len(result.overall_keywords) == 8

    def test_extracts_countries_of_origin(self) -> None:
        """countriesOfOrigin.countries[].text mapped to countries_of_origin."""
        data = _graphql_title_data(
            countriesOfOrigin={"countries": [
                {"text": "United States"},
                {"text": "Germany"},
            ]}
        )
        result = transform_graphql_response(data)
        assert result.countries_of_origin == ["United States", "Germany"]

    def test_extracts_filming_locations(self) -> None:
        """filmingLocations.edges[].node.text mapped to filming_locations."""
        data = _graphql_title_data(
            filmingLocations={"edges": [
                {"node": {"text": "Los Angeles, California"}},
                {"node": {"text": "London, England"}},
            ]}
        )
        result = transform_graphql_response(data)
        assert result.filming_locations == ["Los Angeles, California", "London, England"]

    def test_extracts_languages(self) -> None:
        """spokenLanguages.spokenLanguages[].text mapped to languages."""
        data = _graphql_title_data(
            spokenLanguages={"spokenLanguages": [
                {"text": "English"},
                {"text": "French"},
            ]}
        )
        result = transform_graphql_response(data)
        assert result.languages == ["English", "French"]

    def test_extracts_production_companies(self) -> None:
        """companyCredits.edges[].node.company.companyText.text mapped to production_companies."""
        data = _graphql_title_data(
            companyCredits={"edges": [
                {"node": {"company": {"companyText": {"text": "Fox 2000 Pictures"}}}},
                {"node": {"company": {"companyText": {"text": "Regency Enterprises"}}}},
            ]}
        )
        result = transform_graphql_response(data)
        assert result.production_companies == ["Fox 2000 Pictures", "Regency Enterprises"]

    # --- Reception summary (plaidHtml → plain text) ---

    def test_extracts_reception_summary(self) -> None:
        """reviewSummary.overall.medium.value.plaidHtml converted to plain text."""
        data = _graphql_title_data(
            reviewSummary={
                "overall": {
                    "medium": {
                        "value": {"plaidHtml": "<p>A &amp; visually stunning film.</p>"}
                    }
                },
                "themes": [],
            }
        )
        result = transform_graphql_response(data)
        assert result.reception_summary == "A & visually stunning film."

    def test_reception_summary_none_when_missing(self) -> None:
        """Missing reviewSummary.overall produces reception_summary=None."""
        data = _graphql_title_data(
            reviewSummary={"overall": None, "themes": []}
        )
        result = transform_graphql_response(data)
        assert result.reception_summary is None

    # --- Review themes ---

    def test_extracts_review_themes(self) -> None:
        """reviewSummary.themes mapped to ReviewTheme objects."""
        data = _graphql_title_data(
            reviewSummary={
                "overall": None,
                "themes": [
                    {"label": {"value": "Cinematography"}, "sentiment": "POSITIVE"},
                    {"label": {"value": "Pacing"}, "sentiment": "NEGATIVE"},
                ],
            }
        )
        result = transform_graphql_response(data)
        assert len(result.review_themes) == 2
        assert result.review_themes[0] == ReviewTheme(name="Cinematography", sentiment="POSITIVE")
        assert result.review_themes[1] == ReviewTheme(name="Pacing", sentiment="NEGATIVE")

    def test_review_themes_skips_incomplete(self) -> None:
        """Themes missing name or sentiment are filtered out."""
        data = _graphql_title_data(
            reviewSummary={
                "overall": None,
                "themes": [
                    {"label": {"value": "Valid"}, "sentiment": "POSITIVE"},
                    {"label": {"value": "MissingSentiment"}},
                    {"sentiment": "NEGATIVE"},
                ],
            }
        )
        result = transform_graphql_response(data)
        assert len(result.review_themes) == 1
        assert result.review_themes[0].name == "Valid"

    # --- Maturity reasoning ---

    def test_maturity_reasoning_wraps_in_list(self) -> None:
        """certificate.ratingReason wrapped in [str] if present."""
        data = _graphql_title_data(
            certificate={"rating": "R", "ratingReason": "Rated R for strong violence"}
        )
        result = transform_graphql_response(data)
        assert result.maturity_reasoning == ["Rated R for strong violence"]

    def test_maturity_reasoning_empty_when_no_reason(self) -> None:
        """No ratingReason produces empty list."""
        data = _graphql_title_data(
            certificate={"rating": "PG-13", "ratingReason": None}
        )
        result = transform_graphql_response(data)
        assert result.maturity_reasoning == []

    # --- Parental guide items ---

    def test_extracts_parental_guide_items(self) -> None:
        """parentsGuide.categories mapped to ParentalGuideItem objects."""
        data = _graphql_title_data(
            parentsGuide={"categories": [
                {"category": {"text": "Violence & Gore"}, "severity": {"text": "Severe"}},
                {"category": {"text": "Language"}, "severity": {"text": "Moderate"}},
            ]}
        )
        result = transform_graphql_response(data)
        assert len(result.parental_guide_items) == 2
        assert result.parental_guide_items[0] == ParentalGuideItem(
            category="Violence & Gore", severity="Severe"
        )

    def test_filters_none_severity(self) -> None:
        """Items with severity='none' (case-insensitive) are excluded."""
        data = _graphql_title_data(
            parentsGuide={"categories": [
                {"category": {"text": "Sex & Nudity"}, "severity": {"text": "None"}},
                {"category": {"text": "Cat2"}, "severity": {"text": "none"}},
                {"category": {"text": "Cat3"}, "severity": {"text": "NONE"}},
                {"category": {"text": "Violence"}, "severity": {"text": "Mild"}},
            ]}
        )
        result = transform_graphql_response(data)
        assert len(result.parental_guide_items) == 1
        assert result.parental_guide_items[0].category == "Violence"

    def test_filters_empty_category_or_severity(self) -> None:
        """Items with blank category or severity are excluded."""
        data = _graphql_title_data(
            parentsGuide={"categories": [
                {"category": {"text": ""}, "severity": {"text": "Mild"}},
                {"category": {"text": "Violence"}, "severity": {"text": ""}},
                {"category": {"text": "Language"}, "severity": {"text": "Moderate"}},
            ]}
        )
        result = transform_graphql_response(data)
        assert len(result.parental_guide_items) == 1

    # --- Synopses and plot summaries (via transform) ---

    def test_synopses_and_summaries_through_transform(self) -> None:
        """plots.edges processed through synopsis priority logic in full transform."""
        edges = _plot_edges(synopses=["The full synopsis."])
        data = _graphql_title_data(plots={"edges": edges})
        result = transform_graphql_response(data)
        assert result.synopses == ["The full synopsis."]
        assert result.plot_summaries == []

    def test_summaries_fallback_through_transform(self) -> None:
        """No synopses → summaries extracted (skipping first)."""
        edges = _plot_edges(summaries=["Overview", "Real summary 1", "Real summary 2"])
        data = _graphql_title_data(plots={"edges": edges})
        result = transform_graphql_response(data)
        assert result.synopses == []
        assert result.plot_summaries == ["Real summary 1", "Real summary 2"]

    # --- Plot keywords (via transform) ---

    def test_keywords_through_transform(self) -> None:
        """keywords.edges processed through scoring logic in full transform."""
        edges = _keyword_edges([
            ("time travel", 20, 20),
            ("revenge", 15, 15),
        ])
        data = _graphql_title_data(keywords={"edges": edges})
        result = transform_graphql_response(data)
        assert "time travel" in result.plot_keywords
        assert "revenge" in result.plot_keywords

    # --- Credits ---

    def test_extracts_directors(self) -> None:
        """directors.edges[].node.name.nameText.text mapped to directors."""
        data = _graphql_title_data(
            directors={"edges": [
                {"node": {"name": {"nameText": {"text": "David Fincher"}}}},
            ]}
        )
        result = transform_graphql_response(data)
        assert "David Fincher" in result.directors

    def test_directors_deduplicated(self) -> None:
        """Duplicate director names collapsed to one entry (via set)."""
        data = _graphql_title_data(
            directors={"edges": [
                {"node": {"name": {"nameText": {"text": "Tarantino"}}}},
                {"node": {"name": {"nameText": {"text": "Tarantino"}}}},
            ]}
        )
        result = transform_graphql_response(data)
        assert len(result.directors) == 1

    def test_extracts_writers(self) -> None:
        """writers.edges mapped to writers list (deduplicated)."""
        data = _graphql_title_data(
            writers={"edges": [
                {"node": {"name": {"nameText": {"text": "Charlie Kaufman"}}}},
                {"node": {"name": {"nameText": {"text": "Charlie Kaufman"}}}},
            ]}
        )
        result = transform_graphql_response(data)
        assert len(result.writers) == 1
        assert "Charlie Kaufman" in result.writers

    def test_extracts_actors_preserve_order(self) -> None:
        """cast.edges[].node.name.nameText.text mapped to actors (order preserved)."""
        data = _graphql_title_data(
            cast={"edges": [
                {"node": {"name": {"nameText": {"text": "Brad Pitt"}}, "characters": []}},
                {"node": {"name": {"nameText": {"text": "Edward Norton"}}, "characters": []}},
                {"node": {"name": {"nameText": {"text": "Helena Bonham Carter"}}, "characters": []}},
            ]}
        )
        result = transform_graphql_response(data)
        assert result.actors == ["Brad Pitt", "Edward Norton", "Helena Bonham Carter"]

    def test_extracts_characters(self) -> None:
        """Cast node characters[].name flattened into characters list."""
        data = _graphql_title_data(
            cast={"edges": [
                {
                    "node": {
                        "name": {"nameText": {"text": "Brad Pitt"}},
                        "characters": [{"name": "Tyler Durden"}],
                    }
                },
                {
                    "node": {
                        "name": {"nameText": {"text": "Edward Norton"}},
                        "characters": [{"name": "The Narrator"}],
                    }
                },
            ]}
        )
        result = transform_graphql_response(data)
        assert "Tyler Durden" in result.characters
        assert "The Narrator" in result.characters

    def test_extracts_producers_preserve_order(self) -> None:
        """producers.edges mapped to producers list (order preserved)."""
        data = _graphql_title_data(
            producers={"edges": [
                {"node": {"name": {"nameText": {"text": "Prod A"}}}},
                {"node": {"name": {"nameText": {"text": "Prod B"}}}},
                {"node": {"name": {"nameText": {"text": "Prod C"}}}},
            ]}
        )
        result = transform_graphql_response(data)
        assert result.producers == ["Prod A", "Prod B", "Prod C"]

    def test_extracts_composers_deduplicated(self) -> None:
        """composers.edges mapped to composers list (deduplicated via set)."""
        data = _graphql_title_data(
            composers={"edges": [
                {"node": {"name": {"nameText": {"text": "Hans Zimmer"}}}},
                {"node": {"name": {"nameText": {"text": "Hans Zimmer"}}}},
            ]}
        )
        result = transform_graphql_response(data)
        assert len(result.composers) == 1
        assert "Hans Zimmer" in result.composers

    def test_filters_empty_credit_names(self) -> None:
        """Whitespace-only credit names are skipped."""
        data = _graphql_title_data(
            directors={"edges": [
                {"node": {"name": {"nameText": {"text": "Valid"}}}},
                {"node": {"name": {"nameText": {"text": "   "}}}},
                {"node": {"name": {"nameText": {"text": ""}}}},
            ]}
        )
        result = transform_graphql_response(data)
        assert result.directors == ["Valid"]

    # --- Featured reviews ---

    def test_extracts_featured_reviews(self) -> None:
        """reviews.edges mapped to FeaturedReview objects."""
        data = _graphql_title_data(
            reviews={"edges": [
                {
                    "node": {
                        "summary": {"originalText": "Great film!"},
                        "text": {"originalText": {"plainText": "Loved every minute."}},
                    }
                },
                {
                    "node": {
                        "summary": {"originalText": "Meh"},
                        "text": {"originalText": {"plainText": "It was okay."}},
                    }
                },
            ]}
        )
        result = transform_graphql_response(data)
        assert len(result.featured_reviews) == 2
        assert result.featured_reviews[0] == FeaturedReview(
            summary="Great film!", text="Loved every minute."
        )

    def test_filters_reviews_missing_summary(self) -> None:
        """Reviews without summary are skipped."""
        data = _graphql_title_data(
            reviews={"edges": [
                {
                    "node": {
                        "summary": None,
                        "text": {"originalText": {"plainText": "Some text"}},
                    }
                },
                {
                    "node": {
                        "summary": {"originalText": "Valid"},
                        "text": {"originalText": {"plainText": "Body text"}},
                    }
                },
            ]}
        )
        result = transform_graphql_response(data)
        assert len(result.featured_reviews) == 1
        assert result.featured_reviews[0].summary == "Valid"

    def test_filters_reviews_missing_text(self) -> None:
        """Reviews without text body are skipped."""
        data = _graphql_title_data(
            reviews={"edges": [
                {
                    "node": {
                        "summary": {"originalText": "Title"},
                        "text": None,
                    }
                },
                {
                    "node": {
                        "summary": {"originalText": "Valid"},
                        "text": {"originalText": {"plainText": "Body"}},
                    }
                },
            ]}
        )
        result = transform_graphql_response(data)
        assert len(result.featured_reviews) == 1

    def test_limits_to_ten_reviews(self) -> None:
        """At most 10 reviews returned from a larger set."""
        edges = [
            {
                "node": {
                    "summary": {"originalText": f"Title {i}"},
                    "text": {"originalText": {"plainText": f"Body {i}"}},
                }
            }
            for i in range(15)
        ]
        data = _graphql_title_data(reviews={"edges": edges})
        result = transform_graphql_response(data)
        assert len(result.featured_reviews) == 10

    # --- Full integration ---

    def test_returns_imdb_scraped_movie_type(self) -> None:
        """Transform returns an IMDBScrapedMovie instance."""
        data = _graphql_title_data()
        result = transform_graphql_response(data)
        assert isinstance(result, IMDBScrapedMovie)

    def test_empty_title_data_produces_valid_model(self) -> None:
        """Completely empty title_data still produces a valid model with defaults."""
        result = transform_graphql_response({})
        assert isinstance(result, IMDBScrapedMovie)
        assert result.original_title is None
        assert result.genres == []
        assert result.directors == []
        assert result.imdb_vote_count == 0

    def test_all_empty_collections(self) -> None:
        """When all collection fields are empty, all list fields are []."""
        data = _graphql_title_data()
        result = transform_graphql_response(data)
        assert result.genres == []
        assert result.overall_keywords == []
        assert result.countries_of_origin == []
        assert result.filming_locations == []
        assert result.languages == []
        assert result.production_companies == []
        assert result.review_themes == []
        assert result.synopses == []
        assert result.plot_summaries == []
        assert result.plot_keywords == []
        assert result.maturity_reasoning == []
        assert result.parental_guide_items == []
        assert result.directors == []
        assert result.writers == []
        assert result.actors == []
        assert result.characters == []
        assert result.producers == []
        assert result.composers == []
        assert result.featured_reviews == []
