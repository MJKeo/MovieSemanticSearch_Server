"""
GraphQL response transformer for IMDB data extraction.

Transforms the raw title dict from IMDB's GraphQL API into a typed
IMDBScrapedMovie model. All field extraction is fault-tolerant: missing
data produces None or empty lists, never raises.

Preserves the keyword scoring formula, synopsis priority logic, and
plain text extraction from the original HTML parsers.
"""

import html as html_lib

from bs4 import BeautifulSoup

from .models import (
    IMDBScrapedMovie,
    AwardNomination,
    ReviewTheme,
    ParentalGuideItem,
    FeaturedReview,
)
from schemas.enums import AwardOutcome


# ---------------------------------------------------------------------------
# Award ceremony filter
# ---------------------------------------------------------------------------

# The 12 major ceremonies whose nominations we store. Values are the exact
# `award.event.text` strings returned by the IMDB GraphQL API (verified via
# live queries). Everything outside this set is silently dropped.
_IN_SCOPE_CEREMONIES: frozenset[str] = frozenset({
    "Academy Awards, USA",
    "Golden Globes, USA",
    "BAFTA Awards",
    "Cannes Film Festival",
    "Venice Film Festival",
    "Berlin International Film Festival",
    "Actor Awards",                  # SAG (rebranded in IMDB's data)
    "Critics Choice Awards",
    "Sundance Film Festival",
    "Razzie Awards",
    "Film Independent Spirit Awards",
    "Gotham Awards",
})


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _safe_get(obj: dict | list | None, path: list, default=None):
    """
    Traverse nested dicts/lists by key path.

    Returns default if any key is missing or the intermediate value
    is not a dict. Handles None intermediates gracefully.
    """
    cur = obj
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def _extract_plain_text(plaid_html: str | None) -> str | None:
    """
    Convert IMDB's plaidHtml (HTML-encoded markup) to clean plain text.

    Returns None if the input is empty or produces only whitespace.
    """
    if not plaid_html or not isinstance(plaid_html, str) or not plaid_html.strip():
        return None
    unescaped = html_lib.unescape(plaid_html)
    text = BeautifulSoup(unescaped, "html.parser").get_text(" ", strip=True)
    return text if text.strip() else None


def _strip_or_none(value) -> str | None:
    """Return stripped string or None if blank/missing."""
    if value and isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _extract_edge_texts(edges: list | None, path: list) -> list[str]:
    """
    Extract text values from a list of GraphQL edge objects.

    Each edge is navigated via _safe_get(edge, ["node", ...path]).
    Empty/blank values are filtered out, results are stripped.
    """
    if not edges:
        return []
    texts: list[str] = []
    for edge in edges:
        text = _safe_get(edge, ["node"] + path)
        if text and isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return texts


# ---------------------------------------------------------------------------
# Synopsis / summary priority logic
# ---------------------------------------------------------------------------


def _extract_synopses_and_summaries(
    plots_edges: list | None,
) -> tuple[list[str], list[str]]:
    """
    Apply synopsis priority logic to the GraphQL plots response.

    Extraction logic:
      - Synopses: take the FIRST synopsis only. A single synopsis is a
        comprehensive chronological retelling (500-2000 words).
      - Plot summaries: always extracted independently of synopses. Skip
        the first summary (duplicates the overview from the main page),
        take up to the next 3.
      - If neither exists → both fields return empty.

    GraphQL distinguishes plot types via the "plotType" field:
      - "SYNOPSIS" for synopses
      - "SUMMARY", "OUTLINE", etc. for summaries
    """
    if not plots_edges:
        return [], []

    synopses: list[str] = []
    summaries: list[str] = []

    for edge in plots_edges:
        node = (edge.get("node") or {}) if isinstance(edge, dict) else {}
        plot_type = _safe_get(node, ["plotType"]) or ""
        text = _safe_get(node, ["plotText", "plainText"])
        if not text or not isinstance(text, str) or not text.strip():
            continue

        text = text.strip()
        if plot_type == "SYNOPSIS":
            synopses.append(text)
        else:
            summaries.append(text)

    # Synopses: take the first one only (if any exist)
    result_synopses = [synopses[0]] if synopses else []

    # Summaries: always extracted independently. Skip first entry
    # (duplicates the overview from the main page), take next 10.
    summaries_excluding_overview = summaries[1:] if summaries else []
    result_summaries = summaries_excluding_overview[:3]

    return result_synopses, result_summaries


# ---------------------------------------------------------------------------
# Keyword scoring logic
# ---------------------------------------------------------------------------

# Tunable bounds for the number of plot keywords returned per movie.
# MIN is used as both the no-signal fallback count and the floor when
# vote-based scoring produces fewer results. MAX caps the output.
_MIN_PLOT_KEYWORDS = 10
_MAX_PLOT_KEYWORDS = 15


def _score_and_filter_keywords(keyword_edges: list | None) -> list[str]:
    """
    Apply community vote-based keyword scoring and filtering.

    Scoring formula:
      score = usersInterested - 0.75 * (usersVoted - usersInterested)

    The 0.75 dislike weight reflects that users are quicker to downvote
    than upvote. A keyword with a 50/50 split still scores positive,
    which is intentional — contested keywords like "twist ending" are
    still useful for semantic search.

    Threshold logic:
      N = max score across all keywords
      If N <= 0: take first _MIN_PLOT_KEYWORDS by position (no meaningful vote signal)
      If N > 0: threshold = min(0.75 * N, N - 2)
        - For popular movies (N=20): threshold=15 (tight 75% band)
        - For low-engagement (N=4): threshold=2 (wider absolute band)
      Include all keywords with score >= threshold
      Floor of _MIN_PLOT_KEYWORDS, cap of _MAX_PLOT_KEYWORDS
    """
    if not keyword_edges:
        return []

    # Build list of (keyword_text, score) tuples
    scored_keywords: list[tuple[str, float]] = []
    for edge in keyword_edges:
        node = (edge.get("node") or {}) if isinstance(edge, dict) else {}
        keyword_text = _safe_get(node, ["keyword", "text", "text"])
        if not keyword_text or not isinstance(keyword_text, str) or not keyword_text.strip():
            continue

        keyword_text = keyword_text.strip()

        # Extract vote data for scoring
        interest_score = _safe_get(node, ["interestScore"]) or {}
        users_interested = interest_score.get("usersInterested", 0) or 0
        users_voted = interest_score.get("usersVoted", 0) or 0
        dislikes = users_voted - users_interested

        score = users_interested - 0.75 * dislikes
        scored_keywords.append((keyword_text, score))

    if not scored_keywords:
        return []

    # Find the highest score (N)
    max_score = max(score for _, score in scored_keywords)

    # If N <= 0, no keyword has meaningful votes — take first batch by position
    if max_score <= 0:
        return [kw for kw, _ in scored_keywords[:_MIN_PLOT_KEYWORDS]]

    # Compute inclusion threshold
    threshold = min(0.75 * max_score, max_score - 2)

    # Build a score lookup map for sorting (handles the cap logic cleanly)
    score_map: dict[str, float] = {}
    for kw, s in scored_keywords:
        # If duplicate keyword texts exist, keep the higher score
        if kw not in score_map or s > score_map[kw]:
            score_map[kw] = s

    # Collect keywords that pass the threshold (preserving original order)
    passing: list[str] = [kw for kw, s in scored_keywords if s >= threshold]
    # Deduplicate while preserving order
    seen: set[str] = set()
    passing_deduped: list[str] = []
    for kw in passing:
        if kw not in seen:
            seen.add(kw)
            passing_deduped.append(kw)
    passing = passing_deduped

    # Enforce floor: pad with next-highest-scoring below-threshold keywords
    if len(passing) < _MIN_PLOT_KEYWORDS:
        below_threshold = sorted(
            [(kw, s) for kw, s in scored_keywords if kw not in seen],
            key=lambda x: x[1],
            reverse=True,
        )
        for kw, _ in below_threshold:
            if kw not in seen:
                seen.add(kw)
                passing.append(kw)
            if len(passing) >= _MIN_PLOT_KEYWORDS:
                break

    # Enforce cap: keep top keywords by score
    if len(passing) > _MAX_PLOT_KEYWORDS:
        passing.sort(key=lambda kw: score_map.get(kw, 0), reverse=True)
        passing = passing[:_MAX_PLOT_KEYWORDS]

    return passing


# ---------------------------------------------------------------------------
# Award nomination extraction
# ---------------------------------------------------------------------------


def _extract_awards(award_edges: list | None) -> list[AwardNomination]:
    """
    Extract and filter award nominations to in-scope ceremonies.

    Iterates the `awardNominations.edges` array from the GraphQL response
    and keeps only nominations whose `award.event.text` matches one of the
    12 tracked ceremonies in _IN_SCOPE_CEREMONIES. All other nominations
    (regional critics circles, etc.) are silently dropped.

    Returns a list of AwardNomination objects. Returns empty list when
    no in-scope nominations exist.
    """
    if not award_edges:
        return []

    awards: list[AwardNomination] = []
    for edge in award_edges:
        node = (edge.get("node") or {}) if isinstance(edge, dict) else {}

        # Extract ceremony name from award.event.text
        ceremony = _safe_get(node, ["award", "event", "text"])
        if not ceremony or not isinstance(ceremony, str):
            continue
        ceremony = ceremony.strip()

        # Filter to in-scope ceremonies only
        if ceremony not in _IN_SCOPE_CEREMONIES:
            continue

        # Extract the specific prize name (e.g., "Oscar", "Palme d'Or",
        # "Golden Lion"). Required — this is the user-facing award term.
        award_name = _safe_get(node, ["award", "text"])
        if not award_name or not isinstance(award_name, str):
            continue
        award_name = award_name.strip()

        # Extract year (required — skip if missing or non-numeric)
        year = _safe_get(node, ["award", "year"])
        if year is None:
            continue
        try:
            year = int(year)
        except (TypeError, ValueError):
            continue

        # Map isWinner boolean to outcome string
        is_winner = _safe_get(node, ["isWinner"])
        outcome = AwardOutcome.WINNER if is_winner else AwardOutcome.NOMINEE

        # Extract category (nullable — festival grand prizes like Palme d'Or
        # have no category; the award name IS the category)
        category = _strip_or_none(_safe_get(node, ["category", "text"]))

        awards.append(AwardNomination(
            ceremony=ceremony,
            award_name=award_name,
            category=category,
            outcome=outcome,
            year=year,
        ))

    return awards


# ---------------------------------------------------------------------------
# Main transformer
# ---------------------------------------------------------------------------


def transform_graphql_response(title_data: dict) -> IMDBScrapedMovie:
    """
    Transform a raw IMDB GraphQL title response into an IMDBScrapedMovie.

    This is the single transformation entry point that replaces the 6
    separate HTML parsers. The input is the "data.title" object from the
    GraphQL response (already JSON-parsed by http_client.fetch_movie).

    All field extraction is fault-tolerant: missing data produces None or
    empty lists, never raises.
    """
    # -- Scalar fields -------------------------------------------------------

    # IMDB title type — deterministic indicator of content kind (e.g. "movie",
    # "tvSeries", "videoGame"). Useful for filtering non-movie titles that
    # slip through TMDB.
    imdb_title_type = _strip_or_none(_safe_get(title_data, ["titleType", "id"]))

    original_title = _strip_or_none(_safe_get(title_data, ["originalTitleText", "text"]))
    maturity_rating = _strip_or_none(_safe_get(title_data, ["certificate", "rating"]))
    overview = _strip_or_none(_safe_get(title_data, ["plot", "plotText", "plainText"]))
    imdb_rating = _safe_get(title_data, ["ratingsSummary", "aggregateRating"])
    metacritic_rating = _safe_get(title_data, ["metacritic", "metascore", "score"])
    raw_budget = _safe_get(title_data, ["productionBudget", "budget", "amount"])
    budget = round(raw_budget) if raw_budget is not None else None

    # -- Box office data -------------------------------------------------------
    # Worldwide lifetime gross (inclusive of domestic, verified via Box Office
    # Mojo glossary). Whole USD dollars, None when data unavailable.
    raw_worldwide = _safe_get(title_data, ["lifetimeGross", "total", "amount"])
    bo_currency = _safe_get(title_data, ["lifetimeGross", "total", "currency"])
    if raw_worldwide is not None and bo_currency != "USD":
        # Flag non-USD currencies so we know if this assumption ever breaks.
        # Store 0 rather than None to distinguish "non-USD data exists" from
        # "no data at all".
        print(f"    [WARNING] Non-USD box office currency: {bo_currency} "
              f"(amount={raw_worldwide})")
        box_office_worldwide = 0
    else:
        box_office_worldwide = round(raw_worldwide) if raw_worldwide is not None else None

    # Vote count — defaults to 0, coerced to int
    imdb_vote_count_raw = _safe_get(title_data, ["ratingsSummary", "voteCount"], 0)
    try:
        imdb_vote_count = int(imdb_vote_count_raw) if imdb_vote_count_raw is not None else 0
    except (TypeError, ValueError):
        imdb_vote_count = 0

    # -- Simple list extractions ---------------------------------------------

    # Overall keywords (interest-based, from ATF — IMDB curates these)
    interest_edges = _safe_get(title_data, ["interests", "edges"]) or []
    overall_keywords = _extract_edge_texts(interest_edges, ["primaryText", "text"])

    # Genres
    genre_items = _safe_get(title_data, ["titleGenres", "genres"]) or []
    genres: list[str] = []
    for g in genre_items:
        text = _safe_get(g, ["genre", "text"])
        if text and isinstance(text, str) and text.strip():
            genres.append(text.strip())

    # Countries of origin
    country_items = _safe_get(title_data, ["countriesOfOrigin", "countries"]) or []
    countries_of_origin: list[str] = []
    for c in country_items:
        text = c.get("text") if isinstance(c, dict) else None
        if text and isinstance(text, str) and text.strip():
            countries_of_origin.append(text.strip())

    # Filming locations
    filming_edges = _safe_get(title_data, ["filmingLocations", "edges"]) or []
    filming_locations = _extract_edge_texts(filming_edges, ["text"])

    # Languages
    lang_items = _safe_get(title_data, ["spokenLanguages", "spokenLanguages"]) or []
    languages: list[str] = []
    for lang in lang_items:
        text = lang.get("text") if isinstance(lang, dict) else None
        if text and isinstance(text, str) and text.strip():
            languages.append(text.strip())

    # Production companies
    company_edges = _safe_get(title_data, ["companyCredits", "edges"]) or []
    production_companies = _extract_edge_texts(
        company_edges, ["company", "companyText", "text"]
    )

    # -- Reception summary (requires HTML-to-text conversion) ----------------

    review_html = _safe_get(
        title_data, ["reviewSummary", "overall", "medium", "value", "plaidHtml"]
    )
    reception_summary = _extract_plain_text(review_html)

    # -- Review themes -------------------------------------------------------

    raw_themes = _safe_get(title_data, ["reviewSummary", "themes"]) or []
    review_themes: list[ReviewTheme] = []
    for theme in raw_themes:
        name = _safe_get(theme, ["label", "value"])
        sentiment = _safe_get(theme, ["sentiment"])
        if name and sentiment:
            review_themes.append(ReviewTheme(name=name, sentiment=sentiment))

    # -- Maturity reasoning (from certificate.ratingReason) ------------------
    # The GraphQL API returns a single ratingReason string (e.g., "Rated R for
    # violence and language"). Wrap in a list to match the model's list[str] type.

    rating_reason = _strip_or_none(
        _safe_get(title_data, ["certificate", "ratingReason"])
    )
    maturity_reasoning = [rating_reason] if rating_reason else []

    # -- Parental guide items ------------------------------------------------

    pg_categories = _safe_get(title_data, ["parentsGuide", "categories"]) or []
    parental_guide_items: list[ParentalGuideItem] = []
    for cat in pg_categories:
        category = _safe_get(cat, ["category", "text"])
        severity = _safe_get(cat, ["severity", "text"])
        # Only include if both fields are present and severity is not "none"
        if (
            category and isinstance(category, str) and category.strip()
            and severity and isinstance(severity, str) and severity.strip()
            and severity.strip().lower() != "none"
        ):
            parental_guide_items.append(
                ParentalGuideItem(
                    category=category.strip(),
                    severity=severity.strip(),
                )
            )

    # -- Synopses and plot summaries -----------------------------------------

    plots_edges = _safe_get(title_data, ["plots", "edges"]) or []
    synopses, plot_summaries = _extract_synopses_and_summaries(plots_edges)

    # -- Plot keywords (with community vote scoring) -------------------------

    keyword_edges = _safe_get(title_data, ["keywords", "edges"]) or []
    plot_keywords = _score_and_filter_keywords(keyword_edges)

    # -- Credits -------------------------------------------------------------

    # Directors — deduplicated via set (IMDB sometimes lists same person
    # under multiple sub-roles)
    director_edges = _safe_get(title_data, ["directors", "edges"]) or []
    directors_set: set[str] = set()
    for edge in director_edges:
        name = _safe_get(edge, ["node", "name", "nameText", "text"])
        if name and isinstance(name, str) and name.strip():
            directors_set.add(name.strip())

    # Writers — deduplicated via set
    writer_edges = _safe_get(title_data, ["writers", "edges"]) or []
    writers_set: set[str] = set()
    for edge in writer_edges:
        name = _safe_get(edge, ["node", "name", "nameText", "text"])
        if name and isinstance(name, str) and name.strip():
            writers_set.add(name.strip())

    # Cast — actors preserve billing order, characters flattened
    cast_edges = _safe_get(title_data, ["cast", "edges"]) or []
    actors: list[str] = []
    characters: list[str] = []
    for edge in cast_edges:
        actor_name = _safe_get(edge, ["node", "name", "nameText", "text"])
        if actor_name and isinstance(actor_name, str) and actor_name.strip():
            actors.append(actor_name.strip())
        # Characters are a list of {"name": "..."} objects on Cast nodes
        for char in _safe_get(edge, ["node", "characters"]) or []:
            char_name = char.get("name") if isinstance(char, dict) else None
            if char_name and isinstance(char_name, str) and char_name.strip():
                characters.append(char_name.strip())

    # Producers — preserve billing order
    producer_edges = _safe_get(title_data, ["producers", "edges"]) or []
    producers: list[str] = []
    for edge in producer_edges:
        name = _safe_get(edge, ["node", "name", "nameText", "text"])
        if name and isinstance(name, str) and name.strip():
            producers.append(name.strip())

    # Composers — deduplicated via set
    composer_edges = _safe_get(title_data, ["composers", "edges"]) or []
    composers_set: set[str] = set()
    for edge in composer_edges:
        name = _safe_get(edge, ["node", "name", "nameText", "text"])
        if name and isinstance(name, str) and name.strip():
            composers_set.add(name.strip())

    # -- Featured reviews ----------------------------------------------------

    review_edges = _safe_get(title_data, ["reviews", "edges"]) or []
    featured_reviews: list[FeaturedReview] = []
    for edge in review_edges[:10]:
        node = (edge.get("node") or {}) if isinstance(edge, dict) else {}
        summary = _safe_get(node, ["summary", "originalText"])
        text = _safe_get(node, ["text", "originalText", "plainText"])
        # Both summary and text must be present for a usable review
        if summary and text:
            featured_reviews.append(FeaturedReview(summary=summary, text=text))

    # -- Awards (filtered to in-scope ceremonies) ----------------------------

    award_edges = _safe_get(title_data, ["awardNominations", "edges"]) or []
    awards = _extract_awards(award_edges)

    # -- Assemble the final model --------------------------------------------

    return IMDBScrapedMovie(
        imdb_title_type=imdb_title_type,
        original_title=original_title,
        maturity_rating=maturity_rating,
        overview=overview,
        overall_keywords=overall_keywords,
        imdb_rating=imdb_rating,
        imdb_vote_count=imdb_vote_count,
        metacritic_rating=metacritic_rating,
        reception_summary=reception_summary,
        genres=genres,
        countries_of_origin=countries_of_origin,
        production_companies=production_companies,
        filming_locations=filming_locations,
        languages=languages,
        budget=budget,
        review_themes=review_themes,
        synopses=synopses,
        plot_summaries=plot_summaries,
        plot_keywords=plot_keywords,
        maturity_reasoning=maturity_reasoning,
        parental_guide_items=parental_guide_items,
        directors=list(directors_set),
        writers=list(writers_set),
        actors=actors,
        characters=characters,
        producers=producers,
        composers=list(composers_set),
        featured_reviews=featured_reviews,
        awards=awards,
        box_office_worldwide=box_office_worldwide,
    )
