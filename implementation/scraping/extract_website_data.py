from bs4 import BeautifulSoup
import json
import re
import html as html_lib
from ..classes.schemas import WatchProvider, IMDBFeaturedReview, IMDBReviewTheme
from ..classes.enums import StreamingAccessType


# ================================
#          HELPER METHODS
# ================================


def _safe_get(obj, path, default=None):
    cur = obj
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def _parse_next_data(soup: BeautifulSoup) -> dict:
    script = soup.find("script", id="__NEXT_DATA__", type="application/json")
    if not script or not script.string:
        return {}
    try:
        return json.loads(script.string)
    except json.JSONDecodeError:
        return {}


# ================================
#    EXTRACTING DATA (IMDB)
# ================================


def extract_imdb_attributes(main_page_html: str) -> dict:
    """
    Extracts IMDb movie attributes from the main page HTML.
    
    Returns a dictionary with movie data. Required fields will raise ValueError if missing.
    Lists will only contain non-falsy values, and optional fields default to None.
    """
    soup = BeautifulSoup(main_page_html, "html.parser")
    nd = _parse_next_data(soup)

    page_props = _safe_get(nd, ["props", "pageProps"], {}) or {}
    atf = page_props.get("aboveTheFoldData", {}) or {}
    mcd = page_props.get("mainColumnData", {}) or {}

    # Core fields (pulled from __NEXT_DATA__)
    # Optional field: original_title can be None
    original_title = _safe_get(mcd, ["originalTitleText", "text"])
    if original_title:
        original_title = original_title.strip().lower()
        if len(original_title) == 0:
            original_title = None
    
    # Required field: maturity_rating must exist
    maturity_rating = _safe_get(atf, ["certificate", "rating"])
    if not maturity_rating:
        raise ValueError("maturity_rating is required but was not found")
    maturity_rating = maturity_rating.lower()
    
    # Required field: overview must exist
    overview = _safe_get(atf, ["plot", "plotText", "plainText"])
    if not overview:
        raise ValueError("overview is required but was not found")
    overview = overview.lower()

    
    # Optional field: imdb_rating may exist else defaults to None
    imdb_rating = _safe_get(atf, ["ratingsSummary", "aggregateRating"])
    
    # Optional field: metacritic_rating may exist else defaults to None
    metacritic_rating = _safe_get(atf, ["metacritic", "metascore", "score"])

    # Interests -> your "Keywords" list (Japanese, Anime, Coming-of-Age, ...)
    # Filter out falsy values from the list
    interest_edges = _safe_get(atf, ["interests", "edges"], []) or []
    keywords = []
    for e in interest_edges:
        keyword = _safe_get(e, ["node", "primaryText", "text"])
        if keyword and keyword.strip():
            keywords.append(keyword.strip().lower())
    keywords = keywords[:8] # only take the ones that are most likely to be relevant 

    # Review summary (HTML-escaped markdown-ish HTML)
    # Optional field: user_review_summary can be None
    review_html = _safe_get(mcd, ["reviewSummary", "overall", "medium", "value", "plaidHtml"])
    user_review_summary = None
    if isinstance(review_html, str) and review_html.strip():
        unescaped = html_lib.unescape(review_html)
        user_review_summary = BeautifulSoup(unescaped, "html.parser").get_text(" ", strip=True)
        if user_review_summary.strip():
            user_review_summary = user_review_summary.lower()
        else:
            user_review_summary = None

    # Genres - filter out empty strings and falsy values
    genre_items = _safe_get(atf, ["genres", "genres"], []) or []
    genres = []
    for g in genre_items:
        genre_text = _safe_get(g, ["text"])
        if genre_text and genre_text.strip():
            genres.append(genre_text.strip().lower())

    # Production companies - filter out empty strings and falsy values
    production_companies = _safe_get(atf, ["production", "edges"], []) or []
    production_companies_cleaned = []
    for c in production_companies:
        company_text = _safe_get(c, ["node", "company", "companyText", "text"])
        if company_text and company_text.strip():
            production_companies_cleaned.append(company_text.strip().lower())
    production_companies = production_companies_cleaned

    # Countries of origin - filter out empty strings and falsy values
    country_items = _safe_get(mcd, ["countriesDetails", "countries"], []) or []
    countries_of_origin = []
    for c in country_items:
        country_text = _safe_get(c, ["text"])
        if country_text and country_text.strip():
            countries_of_origin.append(country_text.strip().lower())

    # Filming locations - filter out empty strings and falsy values
    filming_locations = _safe_get(mcd, ["filmingLocations", "edges"], []) or []
    filming_locations_cleaned = []
    for fl in filming_locations:
        location_text = _safe_get(fl, ["node", "text"])
        if location_text and location_text.strip():
            filming_locations_cleaned.append(location_text.strip().lower())
    filming_locations = filming_locations_cleaned

    # Languages - filter out empty strings and falsy values
    lang_items = _safe_get(mcd, ["spokenLanguages", "spokenLanguages"], []) or []
    languages = []
    for l in lang_items:
        lang_text = _safe_get(l, ["text"])
        if lang_text and lang_text.strip():
            languages.append(lang_text.strip().lower())

    # Budget - optional field, defaults to None
    budget = _safe_get(mcd, ["productionBudget", "budget", "amount"])

    # Review Themes - optional field, defaults to []
    review_themes = _safe_get(nd, ['props', 'pageProps', 'mainColumnData', 'reviewSummary', 'themes'], []) or []
    parsed_review_themes = []
    for theme in review_themes:
        name = _safe_get(theme, ['label', 'value'], None)
        sentiment = _safe_get(theme, ['sentiment'], None)
        if name and sentiment:
            parsed_review_themes.append(IMDBReviewTheme(name=name.lower(), sentiment=sentiment.lower()))

    return {
        "original_title": original_title,  # e.g. "Sen to Chihiro no kamikakushi"
        "maturity_rating": maturity_rating,  # e.g. "PG"
        "overview": overview,
        "keywords": keywords,  # interest-based keywords list
        "imdb_rating": imdb_rating,
        "metacritic_rating": metacritic_rating,
        "user_review_summary": user_review_summary,
        "genres": genres,
        "countries_of_origin": countries_of_origin,
        "production_companies": production_companies,
        "filming_locations": filming_locations,
        "languages": languages,
        "budget": budget,
        "review_themes": parsed_review_themes,
    }
    
def extract_summary_attributes(summary_html_text: str) -> dict:
    """
    Extracts plot summaries and synopses from the summary page HTML.
    
    Returns a dictionary with plot_summaries and synopses lists, filtering out empty strings.
    """
    soup = BeautifulSoup(summary_html_text, "html.parser")
    nd = _parse_next_data(soup)

    page_props = _safe_get(nd, ["props", "pageProps"], {}) or {}
    data = _safe_get(page_props, ["contentData", "data", "title"], {}) or {}

    # Plot summaries - filter out empty strings and falsy values
    plot_summaries = _safe_get(data, ["plotSummaries", "edges"], []) or []
    plot_summaries_cleaned = []
    for s in plot_summaries:
        plot_html = _safe_get(s, ["node", "plotText", "plaidHtml"])
        if plot_html and isinstance(plot_html, str) and plot_html.strip():
            # Unescape HTML entities and extract plain text (similar to review_html processing)
            unescaped = html_lib.unescape(plot_html)
            plot_text = BeautifulSoup(unescaped, "html.parser").get_text(" ", strip=True)
            if plot_text.strip():
                plot_summaries_cleaned.append(plot_text.lower())
    plot_summaries = plot_summaries_cleaned
    # Remove the first item since it's the same as the overview
    plot_summaries.pop(0) if plot_summaries else None

    # synopses - filter out empty strings and falsy values
    synopses = _safe_get(data, ["plotSynopsis", "edges"]) or []
    synopses_cleaned = []
    for s in synopses:
        plot_html = _safe_get(s, ["node", "plotText", "plaidHtml"])
        if plot_html and isinstance(plot_html, str) and plot_html.strip():
            unescaped = html_lib.unescape(plot_html)
            plot_text = BeautifulSoup(unescaped, "html.parser").get_text(" ", strip=True)
            if plot_text.strip():
                synopses_cleaned.append(plot_text.lower())
    synopses = synopses_cleaned
    
    return {
        "plot_summaries": plot_summaries,
        "synopses": synopses
    }

def extract_plot_keywords(pk_html_text: str) -> list[str]:
    """
    Extracts plot keywords from the plot keywords page HTML.
    
    Returns a list of keywords, filtering out None and empty string values.
    """
    soup = BeautifulSoup(pk_html_text, "html.parser")
    nd = _parse_next_data(soup)

    page_props = _safe_get(nd, ["props", "pageProps"], {}) or {}
    data = _safe_get(page_props, ["contentData", "data", "title", 'keywords', 'edges'], []) or []
    
    # Filter out None and empty string values
    keywords = []
    for k in data:
        keyword = _safe_get(k, ["node", 'keyword', 'text', 'text'])
        if keyword and keyword.strip():
            keywords.append(keyword.strip().lower())
    keywords = keywords[:8] # only take the ones that are most likely to be relevant 
    
    return keywords

def extract_parental_guide(pg_html_text: str) -> dict:
    """
    Extracts parental guide information from the parental guide page HTML.
    
    Returns a dictionary with ratingReasons and parentsGuide lists, filtering out empty values.
    """
    soup = BeautifulSoup(pg_html_text, "html.parser")
    nd = _parse_next_data(soup)

    page_props = _safe_get(nd, ["props", "pageProps"], {}) or {}
    data = _safe_get(page_props, ["contentData", "data", "title"], {}) or {}

    # Rating reasons - filter out None and empty string values
    ratingReasons = _safe_get(data, ["ratingReason", "edges"], []) or []
    ratingReasons_cleaned = []
    for r in ratingReasons:
        reason = _safe_get(r, ['node', 'ratingReason'])
        if reason and reason.strip():
            ratingReasons_cleaned.append(reason.strip().lower())
    ratingReasons = ratingReasons_cleaned

    # Parents guide - filter out items with None or empty category/severity
    parentsGuide = _safe_get(data, ["parentsGuide", "categories"], []) or []
    formattedParentsGuide = []
    for p in parentsGuide:
        category = _safe_get(p, ['category', 'text'])
        severity = _safe_get(p, ['severity', 'text'])
        # Only add if both category and severity are non-empty
        if category and category.strip() and severity and severity.strip() and (severity.strip().lower() != "none"):
            formattedParentsGuide.append({
                'category': category.strip().lower(),
                'severity': severity.strip().lower()
            })

    return {
        "ratingReasons": ratingReasons,
        "parentsGuide": formattedParentsGuide
    }

def extract_cast_crew(cc_html_text: str) -> dict:
    """
    Extracts cast and crew information from the cast/crew page HTML.
    
    Returns a dictionary with directors, writers, cast, characters, producers, and composers lists.
    Filters out None and empty string values from all lists.
    """
    soup = BeautifulSoup(cc_html_text, "html.parser")
    nd = _parse_next_data(soup)

    page_props = _safe_get(nd, ["props", "pageProps"], {}) or {}
    groupings = _safe_get(page_props, ["contentData", "categories"], []) or []

    directorsData = []
    writersData = []
    castData = []
    castSplitIndex = -1
    producersData = []
    composersData = []

    for grouping in groupings:
        groupName = _safe_get(grouping, ['name'])
        groupingData = _safe_get(grouping, ['section', 'items'], []) or []
        if groupName == 'Director':
            directorsData = groupingData
        elif groupName == 'Writers':
            writersData = groupingData
        elif groupName == 'Cast':
            castData = groupingData
            castSplitIndex = _safe_get(grouping, ['section', 'splitIndex']) or -1
        elif groupName == 'Producers':
            producersData = groupingData
        elif groupName == 'Composer':
            composersData = groupingData

    # Directors - filter out None and empty string values
    directors = set()
    for d in directorsData:
        director = _safe_get(d, ['rowTitle'])
        if director and director.strip():
            directors.add(director.strip().lower())

    # Writers - filter out None and empty string values
    writers = set()
    for w in writersData:
        writer = _safe_get(w, ['rowTitle'])
        if writer and writer.strip():
            writers.add(writer.strip().lower())

    # Producers - filter out None and empty string values
    producers = []
    for p in producersData:
        producer = _safe_get(p, ['rowTitle'])
        if producer and producer.strip():
            producers.append(producer.strip().lower())

    # Composers - filter out None and empty string values
    composers = set()
    for c in composersData:
        composer = _safe_get(c, ['rowTitle'])
        if composer and composer.strip():
            composers.add(composer.strip().lower())

    # Cast is a bit more complicated - filter out None and empty string values
    cast = []
    characters = []
    if castSplitIndex > -1:
        castData = castData[:(castSplitIndex + 1)]
    for actor in castData:
        actor_name = _safe_get(actor, ['rowTitle'])
        if actor_name and actor_name.strip():
            cast.append(actor_name.strip().lower())
        # Extract characters - filter out None and empty string values
        for character in _safe_get(actor, ['characters'], []):
            if character and character.strip():
                characters.append(character.strip().lower())

    return {
        "directors": list(directors),
        "writers": list(writers),
        "cast": cast,
        "characters": characters,
        "producers": producers,
        "composers": list(composers)
    }

def extract_featured_reviews(reviews_html_text: str) -> list[IMDBFeaturedReview]:
    """
    Extracts featured reviews from the reviews page HTML.
    
    Returns a dictionary with review summaries and texts.
    """
    soup = BeautifulSoup(reviews_html_text, "html.parser")
    nd = _parse_next_data(soup)

    reviews_data = _safe_get(nd, ['props', 'pageProps', 'contentData', 'reviews'], []) or []

    featured_reviews = []

    for review_object in reviews_data[:10]:
        review_data = review_object['review']
        summary = _safe_get(review_data, ['reviewSummary'], None)
        text = _safe_get(review_data, ['reviewText'], None)

        if summary and text:
            featured_reviews.append({
                'summary': summary,
                'text': text
            })

    return featured_reviews

# ================================
#    EXTRACTING DATA (TMDB)
# ================================

def get_watch_providers(watch_providers: dict) -> list[WatchProvider]:
    """
    Extracts watch providers from TMDB watch providers data.
    
    Returns a list of WatchProvider objects, filtering out invalid providers.
    Only includes providers with valid required fields (id, name, display_priority).
    """
    try:
        us_providers = watch_providers["results"]["US"]
        
        # Dictionary to track providers by their ID (provider_id is the key)
        # Value is a WatchProvider object
        provider_dict: dict[int, WatchProvider] = {}
        
        # Process flatrate providers (subscription)
        for provider in us_providers.get("flatrate", []):
            try:
                provider_id = provider.get("provider_id")
                provider_name = provider.get("provider_name")
                display_priority = provider.get("display_priority")
                
                # Skip if required fields are missing or falsy
                if not provider_id or not provider_name or display_priority is None:
                    continue
                
                if provider_id not in provider_dict:
                    provider_dict[provider_id] = WatchProvider(
                        id=provider_id,
                        name=provider_name.lower(),
                        logo_path=provider.get("logo_path", ""),
                        display_priority=display_priority,
                        types=[StreamingAccessType.SUBSCRIPTION]
                    )
                else:
                    provider_dict[provider_id].types.append(StreamingAccessType.SUBSCRIPTION)
            except Exception as e:
                print(e)
                continue
        
        # Process buy providers (purchase)
        for provider in us_providers.get("buy", []):
            try:
                provider_id = provider.get("provider_id")
                provider_name = provider.get("provider_name")
                display_priority = provider.get("display_priority")
                
                # Skip if required fields are missing or falsy
                if not provider_id or not provider_name or display_priority is None:
                    continue
                
                if provider_id not in provider_dict:
                    provider_dict[provider_id] = WatchProvider(
                        id=provider_id,
                        name=provider_name.lower(),
                        logo_path=provider.get("logo_path", ""),
                        display_priority=display_priority,
                        types=[StreamingAccessType.BUY]
                    )
                else:
                    provider_dict[provider_id].types.append(StreamingAccessType.BUY)
            except Exception as e:
                print(e)
                continue
        
        # Process rent providers (rent)
        for provider in us_providers.get("rent", []):
            try:
                provider_id = provider.get("provider_id")
                provider_name = provider.get("provider_name")
                display_priority = provider.get("display_priority")
                
                # Skip if required fields are missing or falsy
                if not provider_id or not provider_name or display_priority is None:
                    continue
                
                if provider_id not in provider_dict:
                    provider_dict[provider_id] = WatchProvider(
                        id=provider_id,
                        name=provider_name.lower(),
                        logo_path=provider.get("logo_path", ""),
                        display_priority=display_priority,
                        types=[StreamingAccessType.RENT]
                    )
                else:
                    provider_dict[provider_id].types.append(StreamingAccessType.RENT)
            except Exception as e:
                print(e)
                continue
        
        return list(provider_dict.values())
    except Exception as e:
        print(e)
        return []
