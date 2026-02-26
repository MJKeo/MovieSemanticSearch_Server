"""Unit tests for BaseMovie methods."""

import pytest

from implementation.classes.schemas import MajorCharacter, PlotEventsMetadata, WatchProvider


def test_normalize_text_removes_punctuation_and_collapses_whitespace(base_movie_factory) -> None:
    """_normalize_text should lowercase, remove punctuation, and normalize spaces."""
    movie = base_movie_factory()
    assert movie._normalize_text("  Hello,   WORLD!!  ") == "hello world"


def test_title_string_with_and_without_original_title(base_movie_factory) -> None:
    """title_string should include original title only when present."""
    with_original = base_movie_factory(original_title="Le Fabuleux Destin d'Amelie Poulain")
    without_original = base_movie_factory(original_title=None)
    assert with_original.title_string() == "Movie: Spider-Man (Le Fabuleux Destin d'Amelie Poulain)"
    assert without_original.title_string() == "Movie: Spider-Man"


def test_normalized_title_tokens_includes_hyphen_expansions(base_movie_factory) -> None:
    """normalized_title_tokens should include hyphenated token and split components once."""
    movie = base_movie_factory(title="Spider-Man Spider-Man")
    assert movie.normalized_title_tokens() == ["spider-man", "spider", "man"]


def test_maturity_rating_and_rank_known_and_unknown(base_movie_factory) -> None:
    """maturity_rating_and_rank should map known labels and fallback unknown values."""
    known = base_movie_factory(maturity_rating="PG-13")
    unknown = base_movie_factory(maturity_rating="not-a-rating")
    assert known.maturity_rating_and_rank() == ("pg-13", 3)
    assert unknown.maturity_rating_and_rank() == ("unrated", 999)


def test_genres_subset_with_limit_and_no_limit(base_movie_factory) -> None:
    """genres_subset should support slicing with an optional limit."""
    movie = base_movie_factory(genres=["Action", "Drama", "Sci-Fi"])
    assert movie.genres_subset(2) == ["Action", "Drama"]
    assert movie.genres_subset() == ["Action", "Drama", "Sci-Fi"]


def test_release_decade_bucket_era_labels(base_movie_factory) -> None:
    """release_decade_bucket should produce era-aware labels for multiple decades."""
    silent = base_movie_factory(release_date="1925-01-01")
    golden = base_movie_factory(release_date="1942-01-01")
    modern = base_movie_factory(release_date="2005-01-01")
    assert silent.release_decade_bucket() == "Release date: 1920s, silent era & early cinema"
    assert golden.release_decade_bucket() == "Release date: 1940s, golden age of hollywood"
    assert modern.release_decade_bucket() == "Release date: 2000s, 00s"


def test_release_decade_bucket_invalid_date_returns_empty(base_movie_factory) -> None:
    """release_decade_bucket should return empty string when parsing fails."""
    movie = base_movie_factory(release_date="invalid")
    assert movie.release_decade_bucket() == ""


def test_duration_bucket_boundaries(base_movie_factory) -> None:
    """duration_bucket should map durations using inclusive and exclusive thresholds."""
    assert base_movie_factory(duration=101).duration_bucket() == "short, quick watch"
    assert base_movie_factory(duration=102).duration_bucket() == "standard length"
    assert base_movie_factory(duration=117).duration_bucket() == "standard length"
    assert base_movie_factory(duration=118).duration_bucket() == "long"
    assert base_movie_factory(duration=143).duration_bucket() == "long"
    assert base_movie_factory(duration=144).duration_bucket() == "very long"


def test_budget_bucket_for_era_core_paths(base_movie_factory) -> None:
    """budget_bucket_for_era should classify low, high, and in-range budgets."""
    assert base_movie_factory(budget=None).budget_bucket_for_era() == ""
    assert base_movie_factory(release_date="2012-01-01", budget=10_000_000).budget_bucket_for_era() == "small budget"
    assert base_movie_factory(release_date="2012-01-01", budget=250_000_001).budget_bucket_for_era() == "big budget, blockbuster"
    assert base_movie_factory(release_date="2012-01-01", budget=80_000_000).budget_bucket_for_era() == ""


def test_budget_bucket_for_era_fallback_decades(base_movie_factory) -> None:
    """budget_bucket_for_era should clamp decades before 1920 and after 2030."""
    pre_1920 = base_movie_factory(release_date="1899-01-01", budget=50_000)
    post_2030 = base_movie_factory(release_date="2035-01-01", budget=300_000_000)
    assert pre_1920.budget_bucket_for_era() == "small budget"
    assert post_2030.budget_bucket_for_era() == "big budget, blockbuster"


def test_maturity_guidance_text_unrated_uses_parental_items(base_movie_factory) -> None:
    """maturity_guidance_text should return parental item text for unrated movies."""
    movie = base_movie_factory(maturity_rating="Unrated", maturity_reasoning=[])
    assert movie.maturity_guidance_text() == "moderate violence, mild language"


def test_maturity_guidance_text_rated_with_reasoning(base_movie_factory) -> None:
    """maturity_guidance_text should use semantic description plus reasons when available."""
    movie = base_movie_factory(
        maturity_rating="PG-13",
        maturity_reasoning=["Strong action violence", "Some language"],
    )
    assert movie.maturity_guidance_text() == (
        "Parents strongly cautioned. Best for teens and young adults. May contain material inappropriate for young children."
        " Strong action violence. Some language"
    )


def test_maturity_guidance_text_rated_without_reasoning_uses_parental_guide(base_movie_factory) -> None:
    """maturity_guidance_text should fall back to parental guide text when reasons are absent."""
    movie = base_movie_factory(maturity_rating="PG", maturity_reasoning=[])
    assert movie.maturity_guidance_text() == "Rated PG for moderate violence, mild language"


def test_production_text_variants(base_movie_factory) -> None:
    """production_text should include only non-empty sentence parts."""
    all_data = base_movie_factory(
        countries_of_origin=["USA", "Canada"],
        production_companies=["A24", "Plan B"],
        filming_locations=["Toronto"],
    )
    assert all_data.production_text() == (
        "Produced in USA, Canada by A24, Plan B. Filming happened in Toronto."
    )

    only_companies = base_movie_factory(countries_of_origin=[], production_companies=["A24"], filming_locations=[])
    assert only_companies.production_text() == "Produced by A24."

    empty = base_movie_factory(countries_of_origin=[], production_companies=[], filming_locations=[])
    assert empty.production_text() == ""


def test_languages_text_variants(base_movie_factory) -> None:
    """languages_text should handle zero, one, or many languages."""
    assert base_movie_factory(languages=[]).languages_text() == ""
    assert base_movie_factory(languages=["English"]).languages_text() == "Primary language: English"
    assert base_movie_factory(languages=["English", "Spanish"]).languages_text() == (
        "Primary language: English. Audio also available for Spanish"
    )


def test_cast_text_truncation_and_empty(base_movie_factory) -> None:
    """cast_text should truncate producers and actors and return empty when no data exists."""
    movie = base_movie_factory()
    cast_text = movie.cast_text()
    # Producers should be truncated to four names.
    assert "Produced by Laura Ziskin, Ian Bryce, Avi Arad, Grant Curtis" in cast_text
    assert "Extra Producer" not in cast_text
    # Actors should be truncated to five names.
    assert "Main actors: Tobey Maguire, Kirsten Dunst, Willem Dafoe, James Franco, Rosemary Harris" in cast_text
    assert "J.K. Simmons" not in cast_text

    empty = base_movie_factory(directors=[], writers=[], producers=[], composers=[], actors=[])
    assert empty.cast_text() == ""


def test_characters_text_prefers_plot_events_major_characters(base_movie_factory) -> None:
    """characters_text should prioritize major characters from plot_events_metadata when available."""
    plot_events = PlotEventsMetadata(
        plot_summary="A hero saves the city.",
        setting="New York City",
        major_characters=[
            MajorCharacter(
                name="Peter Parker",
                description="A student hero",
                role="protagonist",
                primary_motivations="Protect loved ones.",
            ),
            MajorCharacter(
                name="Norman Osborn",
                description="A powerful rival",
                role="antagonist",
                primary_motivations="Control the city.",
            ),
        ],
    )
    movie = base_movie_factory(plot_events_metadata=plot_events, characters=["Fallback Character"])
    assert movie.characters_text() == "Main characters: peter parker, norman osborn"


def test_characters_text_fallback_to_characters_list(base_movie_factory) -> None:
    """characters_text should fallback to movie.characters when plot metadata is absent."""
    plot_events = PlotEventsMetadata(
        plot_summary="A story unfolds.",
        setting="Somewhere",
        major_characters=[],
    )
    movie = base_movie_factory(plot_events_metadata=plot_events, characters=["Alice", "Bob", "Carol"])
    assert movie.characters_text() == "Main characters: Alice, Bob, Carol"


def test_characters_text_empty_when_no_sources(base_movie_factory) -> None:
    """characters_text should return empty string when no character data exists."""
    plot_events = PlotEventsMetadata(
        plot_summary="A story unfolds.",
        setting="Somewhere",
        major_characters=[],
    )
    movie = base_movie_factory(plot_events_metadata=plot_events, characters=[])
    assert movie.characters_text() == ""


def test_reception_score_paths(base_movie_factory) -> None:
    """reception_score should compute weighted score depending on available ratings."""
    both = base_movie_factory(imdb_rating=8.0, metacritic_rating=70.0)
    imdb_only = base_movie_factory(imdb_rating=8.0, metacritic_rating=None)
    metacritic_only = base_movie_factory(imdb_rating=None, metacritic_rating=70.0)
    neither = base_movie_factory(imdb_rating=None, metacritic_rating=None)
    assert both.reception_score() == 74.0
    assert imdb_only.reception_score() == 32.0
    assert metacritic_only.reception_score() == 42.0
    assert neither.reception_score() is None


def test_reception_tier_buckets(base_movie_factory) -> None:
    """reception_tier should map each score band to the expected label."""
    assert base_movie_factory(imdb_rating=9.5, metacritic_rating=90.0).reception_tier() == "Universally acclaimed"
    assert base_movie_factory(imdb_rating=7.2, metacritic_rating=65.0).reception_tier() == "Generally favorable reviews"
    assert base_movie_factory(imdb_rating=5.0, metacritic_rating=40.0).reception_tier() == "Mixed or average reviews"
    assert base_movie_factory(imdb_rating=3.0, metacritic_rating=20.0).reception_tier() == "Generally unfavorable reviews"
    assert base_movie_factory(imdb_rating=1.0, metacritic_rating=10.0).reception_tier() == "Overwhelming dislike"
    assert base_movie_factory(imdb_rating=None, metacritic_rating=None).reception_tier() is None


def test_reception_summary_text_variants(base_movie_factory) -> None:
    """reception_summary_text should include prefix only when summary is present."""
    assert base_movie_factory(reception_summary="Strong critical praise.").reception_summary_text() == (
        "Review summary: Strong critical praise."
    )
    assert base_movie_factory(reception_summary=None).reception_summary_text() == ""


def test_watch_providers_text_variants(base_movie_factory) -> None:
    """watch_providers_text should produce provider sentence when providers exist."""
    providers = [
        WatchProvider(id=1, name="Netflix", logo_path="/n.png", display_priority=1, types=["subscription"]),
        WatchProvider(id=2, name="Apple TV", logo_path="/a.png", display_priority=2, types=["buy"]),
    ]
    movie = base_movie_factory(watch_providers=providers)
    assert movie.watch_providers_text() == "Watch on Netflix, Apple TV"
    assert base_movie_factory(watch_providers=[]).watch_providers_text() == ""


# ================================
#       EDGE CASE TESTS
# ================================


def test_duration_bucket_zero_and_negative(base_movie_factory) -> None:
    """duration_bucket should classify zero and negative durations as 'short, quick watch'."""
    assert base_movie_factory(duration=0).duration_bucket() == "short, quick watch"
    assert base_movie_factory(duration=-1).duration_bucket() == "short, quick watch"


def test_budget_bucket_zero_is_small(base_movie_factory) -> None:
    """budget_bucket_for_era should classify a budget of 0 as 'small budget'."""
    movie = base_movie_factory(release_date="2020-01-01", budget=0)
    assert movie.budget_bucket_for_era() == "small budget"


def test_budget_bucket_invalid_release_date(base_movie_factory) -> None:
    """budget_bucket_for_era should return empty string when release_date is unparseable."""
    movie = base_movie_factory(release_date="not-a-date", budget=50_000_000)
    assert movie.budget_bucket_for_era() == ""


def test_normalized_title_tokens_multi_word_no_hyphens(base_movie_factory) -> None:
    """normalized_title_tokens should handle a simple multi-word title without hyphens."""
    movie = base_movie_factory(title="The Dark Knight")
    tokens = movie.normalized_title_tokens()
    assert "the" in tokens
    assert "dark" in tokens
    assert "knight" in tokens


def test_normalized_title_tokens_unicode_title(base_movie_factory) -> None:
    """normalized_title_tokens should normalize Unicode characters (diacritics removed)."""
    movie = base_movie_factory(title="Amélie")
    tokens = movie.normalized_title_tokens()
    assert "amelie" in tokens


def test_release_decade_bucket_boundary_1929_vs_1930(base_movie_factory) -> None:
    """release_decade_bucket should differentiate the silent-era / golden-age boundary."""
    silent = base_movie_factory(release_date="1929-12-31")
    golden = base_movie_factory(release_date="1930-01-01")
    assert "silent era" in silent.release_decade_bucket()
    assert "golden age" in golden.release_decade_bucket()


def test_release_decade_bucket_boundary_1949_vs_1950(base_movie_factory) -> None:
    """release_decade_bucket should differentiate golden age from the 1950s."""
    golden = base_movie_factory(release_date="1949-12-31")
    fifties = base_movie_factory(release_date="1950-01-01")
    assert "golden age" in golden.release_decade_bucket()
    assert "50s" in fifties.release_decade_bucket()


def test_characters_text_raises_when_plot_events_metadata_is_none(base_movie_factory) -> None:
    """characters_text raises AttributeError when plot_events_metadata is None (known limitation)."""
    movie = base_movie_factory(plot_events_metadata=None, characters=["Alice"])
    with pytest.raises(AttributeError):
        movie.characters_text()


def test_reception_score_treats_zero_rating_as_missing(base_movie_factory) -> None:
    """reception_score treats 0.0 ratings as missing due to truthiness check (known behavior)."""
    movie = base_movie_factory(imdb_rating=0.0, metacritic_rating=0.0)
    assert movie.reception_score() is None


def test_normalize_text_empty_string(base_movie_factory) -> None:
    """_normalize_text should return empty string for empty input."""
    movie = base_movie_factory()
    assert movie._normalize_text("") == ""


def test_normalize_text_unicode_characters(base_movie_factory) -> None:
    """_normalize_text should strip non-alphanumeric chars and collapse spaces."""
    movie = base_movie_factory()
    assert movie._normalize_text("Café — résumé!") == "café résumé"


def test_genres_subset_limit_zero(base_movie_factory) -> None:
    """genres_subset with limit=0 should return the full list (0 is falsy)."""
    movie = base_movie_factory(genres=["Action", "Drama", "Sci-Fi"])
    assert movie.genres_subset(0) == ["Action", "Drama", "Sci-Fi"]


def test_production_text_only_countries(base_movie_factory) -> None:
    """production_text should produce correct output with only countries (no companies)."""
    movie = base_movie_factory(countries_of_origin=["Japan"], production_companies=[], filming_locations=[])
    assert movie.production_text() == "Produced in Japan."


# ================================
#  RECEPTION TIER BOUNDARY TESTS
# ================================


@pytest.mark.parametrize(
    ("imdb", "meta", "expected_tier"),
    [
        # Exact boundary at 81: score = 4*9 + 0.6*75 = 36 + 45 = 81.0
        (9.0, 75.0, "Universally acclaimed"),
        # Just below 81: score = 4*9 + 0.6*74.9 = 36 + 44.94 = 80.94
        (9.0, 74.9, "Generally favorable reviews"),
        # Exact boundary at 61: score = 4*6.1 + 0.6*61 = 24.4 + 36.6 = 61.0
        (6.1, 61.0, "Generally favorable reviews"),
        # Just below 61: score = 4*7 + 0.6*54.9 = 28 + 32.94 = 60.94
        (7.0, 54.9, "Mixed or average reviews"),
        # Exact boundary at 41: score = 4*4.1 + 0.6*41 = 16.4 + 24.6 = 41.0
        (4.1, 41.0, "Mixed or average reviews"),
        # Just below 41: score = 4*5 + 0.6*34.9 = 20 + 20.94 = 40.94
        (5.0, 34.9, "Generally unfavorable reviews"),
        # Exact boundary at 21: score = 4*2.1 + 0.6*21 = 8.4 + 12.6 = 21.0
        (2.1, 21.0, "Generally unfavorable reviews"),
        # Just below 21: score = 4*3 + 0.6*14.9 = 12 + 8.94 = 20.94
        (3.0, 14.9, "Overwhelming dislike"),
    ],
)
def test_reception_tier_exact_boundaries(base_movie_factory, imdb: float, meta: float, expected_tier: str) -> None:
    """reception_tier should map scores at and just below each tier boundary correctly."""
    movie = base_movie_factory(imdb_rating=imdb, metacritic_rating=meta)
    assert movie.reception_tier() == expected_tier
