import os
from typing import Optional, Dict, Any
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import json
import re
import html as html_lib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from extract_website_data import extract_imdb_attributes, extract_summary_attributes, extract_plot_keywords, extract_parental_guide, extract_cast_crew, get_watch_providers
from classes import IMDBMovie
from llm_generations import generate_plot_metadata, generate_vibe_metadata

# Load environment variables (for API key)
load_dotenv()

# Get API keys from environment variable
tmdb_access_token = os.getenv("TMDB_ACCESS_TOKEN")
imdb_cookie = os.getenv("IMDB_COOKIE")

# ================================
#         TMDB FETCHING
# ================================

def fetch_tmdb_movie_data(tmdb_movie_id: int) -> dict:
    # Construct the URL
    url = f"https://api.themoviedb.org/3/movie/{tmdb_movie_id}"

    # Set the headers with Bearer token authentication
    headers = {
        "Authorization": f"Bearer {tmdb_access_token}"
    }

    # Set additional query parameters
    params = {
        "append_to_response": "release_dates,keywords,watch/providers,credits"
    }

    # Make the request with access token in Authorization header
    response = requests.get(url, headers=headers, params=params)

    # Read the response as JSON
    data = response.json()
    
    return {
        "imdb_id": data["imdb_id"],
        "title": data["title"].lower(),
        "release_date": data["release_date"],
        "duration": data["runtime"],
        "watch_providers": get_watch_providers(data["watch/providers"]),
    }   


# ================================
#         IMDB FETCHING
# ================================

IMDB_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    # Safer with requests unless you know you can decode br/zstd:
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "max-age=0",
    "Priority": "u=0, i",
    "Referer": "https://www.google.com/",
    "Sec-CH-UA": '"Not;A=Brand";v="99", "Google Chrome";v="139", "Chromium";v="139"',
    "Sec-CH-UA-Mobile": "?0",
    "Sec-CH-UA-Platform": '"macOS"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
}

def perform_imdb_fetch(url: str) -> Dict[str, Any]:
    headers = dict(IMDB_HEADERS)

    # Put your real cookie string in an env var instead of hard-coding it.
    # Example: export IMDB_COOKIE='session-id=...; session-token=...; ...'
    if imdb_cookie:
        headers["Cookie"] = imdb_cookie

    with requests.Session() as session:
        resp = session.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp
        

def fetch_main_page(movie_id: str) -> Dict[str, Any]:
    return perform_imdb_fetch(f"https://www.imdb.com/title/{movie_id}/")

def fetch_summary_synopsis(movie_id: str) -> Dict[str, Any]:
    return perform_imdb_fetch(f"https://www.imdb.com/title/{movie_id}/plotsummary/")

def fetch_plot_keywords(movie_id: str) -> Dict[str, Any]:
    return perform_imdb_fetch(f"https://www.imdb.com/title/{movie_id}/keywords/")

def fetch_parent_guide(movie_id: str) -> Dict[str, Any]:
    return perform_imdb_fetch(f"https://www.imdb.com/title/{movie_id}/parentalguide/")

def fetch_cast_crew(movie_id: str) -> Dict[str, Any]:
    return perform_imdb_fetch(f"https://www.imdb.com/title/{movie_id}/fullcredits/")

def fetch_and_create_imdb_movie(tmdb_movie_id: int) -> dict:
    """
    Fetches all IMDb data for a movie in parallel and combines into an IMDBMovie object.
    
    Args:
        tmdb_movie_id: TMDB movie ID (integer)
        
    Returns:
        Dictionary with structure:
        {
            "success": bool,
            "imdb_movie": IMDBMovie | None,
            "failure_reason": str | None
        }
        
        If any fetch fails, the entire movie is invalidated and success=False.
        If an exception occurs at any point, success=False and failure_reason contains the error message.
    """
    try:
        tmdb_movie_data = fetch_tmdb_movie_data(tmdb_movie_id)
        
        imdb_movie_id = tmdb_movie_data["imdb_id"]
        # Dictionary to store parsed results as they complete
        parsed_results = {}
        
        def fetch_and_parse_main_page():
            """Fetches main page and parses IMDb attributes."""
            response = fetch_main_page(imdb_movie_id)
            return ("imdb_data", extract_imdb_attributes(response.text))
        
        def fetch_and_parse_summary():
            """Fetches summary/synopsis page and parses summary attributes."""
            response = fetch_summary_synopsis(imdb_movie_id)
            return ("summary_data", extract_summary_attributes(response.text))
        
        def fetch_and_parse_plot_keywords():
            """Fetches plot keywords page and parses keywords."""
            response = fetch_plot_keywords(imdb_movie_id)
            return ("plot_keywords_data", extract_plot_keywords(response.text))
        
        def fetch_and_parse_parental_guide():
            """Fetches parental guide page and parses parental guide data."""
            response = fetch_parent_guide(imdb_movie_id)
            return ("parental_data", extract_parental_guide(response.text))
        
        def fetch_and_parse_cast_crew():
            """Fetches cast/crew page and parses cast and crew data."""
            response = fetch_cast_crew(imdb_movie_id)
            return ("cast_crew_data", extract_cast_crew(response.text))
        
        # Execute all API calls in parallel and parse as each completes
        # If any fetch fails, the entire movie is invalidated
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tasks
            futures = {
                executor.submit(fetch_and_parse_main_page): "main_page",
                executor.submit(fetch_and_parse_summary): "summary",
                executor.submit(fetch_and_parse_plot_keywords): "plot_keywords",
                executor.submit(fetch_and_parse_parental_guide): "parental_guide",
                executor.submit(fetch_and_parse_cast_crew): "cast_crew"
            }
            
            # Process results as they complete - if any fails, raise exception
            for future in as_completed(futures):
                key, parsed_data = future.result()
                parsed_results[key] = parsed_data
        
        # Extract parsed data - all required fields must exist
        imdb_data = parsed_results["imdb_data"]
        summary_data = parsed_results["summary_data"]
        plot_keywords_data = parsed_results["plot_keywords_data"]
        parental_data = parsed_results["parental_data"]
        cast_crew_data = parsed_results["cast_crew_data"]
        
        # Generate plot summary and DenseVibe metadata using LLM in parallel - handle exceptions and empty results
        # Default values if generation fails or returns empty
        generated_plot_synopsis = None
        generated_plot_keyphrases = []
        generated_vibe_metadata = None
        
        # Execute both LLM calls in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            plot_future = executor.submit(
                generate_plot_metadata,
                overview=imdb_data["overview"],
                plot_keywords=plot_keywords_data,
                plot_summaries=summary_data["plot_summaries"],
                synopsis=summary_data["synopsis"]
            )
            vibe_future = executor.submit(
                generate_vibe_metadata,
                overview=imdb_data["overview"],
                genres=imdb_data["genres"],
                overall_keywords=imdb_data["keywords"],
                plot_keywords=plot_keywords_data,
                synopsis=summary_data["synopsis"],
                plot_summaries=summary_data["plot_summaries"],
                maturity_rating=imdb_data["maturity_rating"],
                maturity_reasoning=parental_data["ratingReasons"],
                parental_guide_items=parental_data["parentsGuide"],
                reception_summary=imdb_data.get("user_review_summary")
            )
            
            # Process results as they complete
            for future in as_completed([plot_future, vibe_future]):
                result_type, result_data = future.result()
                if result_type == "plot":
                    generated_plot_synopsis = result_data["plot_synopsis"]
                    generated_plot_keyphrases = result_data["plot_keyphrases"]
                elif result_type == "vibe":
                    generated_vibe_metadata = result_data
        
        # Combine all parsed data into IMDBMovie object
        movie = IMDBMovie(
            id=imdb_movie_id,
            tmdb_id=tmdb_movie_id,
            # Base stats
            title=tmdb_movie_data["title"],
            original_title=imdb_data.get("original_title"),  # Optional field, defaults to None
            overall_keywords=imdb_data["keywords"],
            release_date=tmdb_movie_data["release_date"],
            duration=tmdb_movie_data["duration"],
            genres=imdb_data["genres"],
            countries_of_origin=imdb_data["countries_of_origin"],
            languages=imdb_data["languages"],
            filming_locations=imdb_data["filming_locations"],
            budget=imdb_data.get("budget"),  # Optional field, defaults to None
            watch_providers=tmdb_movie_data["watch_providers"],
            # Maturity
            maturity_rating=imdb_data["maturity_rating"],
            maturity_reasoning=parental_data["ratingReasons"],
            parental_guide_items=parental_data["parentsGuide"],
            # Plot
            overview=imdb_data["overview"],
            synopsis=generated_plot_synopsis,
            plot_keywords=plot_keywords_data + generated_plot_keyphrases,
            # Cast
            directors=cast_crew_data["directors"],
            writers=cast_crew_data["writers"],
            producers=cast_crew_data["producers"],
            composers=cast_crew_data["composers"],
            actors=cast_crew_data["cast"],
            characters=cast_crew_data["characters"],
            production_companies=imdb_data["production_companies"],
            # Popularity
            imdb_rating=imdb_data["imdb_rating"],
            metacritic_rating=imdb_data["metacritic_rating"],
            reception_summary=imdb_data.get("user_review_summary"),  # Optional field, defaults to None
            # Vibe (LLM-generated viewer experience descriptors)
            vibe_summary=generated_vibe_metadata.vibe_summary,
            vibe_keywords=generated_vibe_metadata.vibe_keywords,
            watch_context_tags=generated_vibe_metadata.watch_context_tags,
            # DEBUG
            debug_synopses=summary_data["synopsis"],
            debug_plot_summaries=summary_data["plot_summaries"]
        )
        
        return {
            "success": True,
            "imdb_movie": movie,
            "failure_reason": None
        }
    except Exception as e:
        # Catch any exception and return failure result
        return {
            "success": False,
            "imdb_movie": None,
            "failure_reason": str(e)
        }

def fetch_batch_imdb_movies(tmdb_movie_ids: list[int]) -> None:
    """
    Fetches IMDb data for multiple movies in parallel and saves results to JSON files.
    
    Args:
        tmdb_movie_ids: List of TMDB movie IDs (integers) to fetch
        
    Saves successful IMDBMovie objects to saved_imdb_movies.json and failures to failed_fetches.json.
    """
    print(f"Processing {len(tmdb_movie_ids)} movies...")
    
    successful_movies = []
    failed_fetches = []
    
    # Process all movies in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all IMDb processing tasks
        futures = {executor.submit(fetch_and_create_imdb_movie, tmdb_id): tmdb_id for tmdb_id in tmdb_movie_ids}
        
        # Process results as they complete with progress bar
        for future in tqdm(as_completed(futures), total=len(tmdb_movie_ids), desc="Processing IMDb data"):
            tmdb_id = futures[future]
            result = future.result()
            
            if result["success"]:
                successful_movies.append(result["imdb_movie"])
            else:
                failed_fetches.append({
                    "tmdb_movie_id": tmdb_id,
                    "failure_reason": result["failure_reason"]
                })
    
    # Save successful movies to JSON file
    print(f"Saving {len(successful_movies)} successful movies to saved_imdb_movies.json...")
    if successful_movies:
        movies_json = [movie.model_dump(mode='json') for movie in successful_movies]
        with open("saved_imdb_movies.json", "w", encoding="utf-8") as f:
            json.dump(movies_json, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(successful_movies)} movies to saved_imdb_movies.json")
    else:
        print("No successful movies to save.")
    
    # Save failed fetches to JSON file
    print(f"Saving {len(failed_fetches)} failed fetches to failed_fetches.json...")
    if failed_fetches:
        with open("failed_fetches.json", "w", encoding="utf-8") as f:
            json.dump(failed_fetches, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(failed_fetches)} failures to failed_fetches.json")
    else:
        print("No failed fetches to save.")
    
    print(f"\nSummary: {len(successful_movies)} successful, {len(failed_fetches)} failed")

if __name__ == "__main__":
    # Override this value to test with a different movie
    tmdb_movie_ids = [129,155,545611,508965,493922,10386,8587,693134,569094,661539,940721,25195,1184918,609,1585,137106,
    346,496243,238,680,278,603,27205,289,15,62,78,348,539,419430,76341,6977,38,376867,313369,194,1417,598,120467,244786,
    274,128,372058,150540,862,1422,146,77338,582,694]

    fetch_batch_imdb_movies(tmdb_movie_ids)
    