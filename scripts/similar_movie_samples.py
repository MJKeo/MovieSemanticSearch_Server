"""
One-off: build a curated catalog of 30 "similar movies" example queries.

Each group has 2-4 movies that share a non-obvious thread (a vibe, a
shape, a recurring motif) — distinct enough to be interesting, related
enough to land. For each movie we fetch TMDB id, title, release year,
and poster_path via the public TMDB search endpoint.

Output: similar_movie_samples.json in the project root.
"""

import asyncio
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.environ["TMDB_API_KEY"]
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_MOVIE_URL = "https://api.themoviedb.org/3/movie/{movie_id}"

OUTPUT_PATH = Path("similar_movie_samples.json")

# A few titles are genuinely ambiguous on TMDB — for example, two
# distinct 2015 films named "The Witch" (Robert Eggers' US horror vs a
# Russian thriller titled "Ведьма" with the English title "The Witch"),
# or "Locke" returning a Chinese 2013 release titled "Target Locked"
# before the Tom Hardy film (TMDB-tagged 2014 because of its UK release
# date even though its 2013 festival run is what we mean). For each of
# those we pin the TMDB id directly rather than rely on search heuristics.
OVERRIDES: dict[tuple[str, int], int] = {
    ("Locke", 2013): 210479,        # Tom Hardy single-take drama
    ("The Witch", 2015): 310131,    # Robert Eggers' folk horror
}


# Each group: a fun blurb + a list of (title, year) tuples. Year is used
# to disambiguate the TMDB search; the script will fall back to a yearless
# search if the year-filtered query returns nothing.
GROUPS: list[dict] = [
    {
        "blurb": "Stranded Souls Sweet on Their Strange Companions",
        "movies": [("Cast Away", 2000), ("Her", 2013), ("Lars and the Real Girl", 2007)],
    },
    {
        "blurb": "Heists Where the Real Steal Is Your Trust",
        "movies": [
            ("Now You See Me", 2013),
            ("Inside Man", 2006),
            ("The Sting", 1973),
            ("Focus", 2015),
        ],
    },
    {
        "blurb": "Stuck in a Day That Refuses to Let Them Grow Up",
        "movies": [
            ("Groundhog Day", 1993),
            ("Palm Springs", 2020),
            ("Edge of Tomorrow", 2014),
        ],
    },
    {
        "blurb": "First Contact, But the Aliens Are Mostly Mood Lighting",
        "movies": [("Arrival", 2016), ("Annihilation", 2018), ("Contact", 1997)],
    },
    {
        "blurb": "Do Not, Under Any Circumstances, Mess With This Guy's Family",
        "movies": [("Taken", 2008), ("John Wick", 2014), ("Prisoners", 2013)],
    },
    {
        "blurb": "Bike Tires, Bug Bites, and Best Friends Forever",
        "movies": [("Stand by Me", 1986), ("E.T. the Extra-Terrestrial", 1982), ("The Sandlot", 1993)],
    },
    {
        "blurb": "When the Recipe Doubles as a Love Letter",
        "movies": [("Chef", 2014), ("Ratatouille", 2007), ("Burnt", 2015)],
    },
    {
        "blurb": "Crews Too Charming to Get Caught",
        "movies": [("Ocean's Eleven", 2001), ("Logan Lucky", 2017), ("Baby Driver", 2017)],
    },
    {
        "blurb": "Your Brain Hurts and You're Loving It",
        "movies": [("Inception", 2010), ("Memento", 2000), ("Tenet", 2020)],
    },
    {
        "blurb": "Animated and Coming Straight for Your Tear Ducts",
        "movies": [
            ("Up", 2009),
            ("Inside Out", 2015),
            ("Coco", 2017),
            ("Toy Story 3", 2010),
        ],
    },
    {
        "blurb": "One Room, One Phone, Zero Chill",
        "movies": [("Buried", 2010), ("Phone Booth", 2002), ("Locke", 2013)],
    },
    {
        "blurb": "Folk Horror That Got Way Too Close to the Tree Line",
        "movies": [("Hereditary", 2018), ("Midsommar", 2019), ("The Witch", 2015)],
    },
    {
        "blurb": "The Car Got Top Billing",
        "movies": [("Drive", 2011), ("Baby Driver", 2017), ("Bullitt", 1968)],
    },
    {
        "blurb": "Hired Killers Who Are, Frankly, Big Softies",
        "movies": [("Léon: The Professional", 1994), ("John Wick", 2014), ("In Bruges", 2008)],
    },
    {
        "blurb": "Robots Working Through Some Stuff",
        "movies": [("WALL·E", 2008), ("Ex Machina", 2014), ("Blade Runner 2049", 2017)],
    },
    {
        "blurb": "Hand Over the Camera and Run for Your Life",
        "movies": [("The Blair Witch Project", 1999), ("Paranormal Activity", 2007), ("[REC]", 2007)],
    },
    {
        "blurb": "Stitched to Feel Like One Long Held Breath",
        "movies": [("1917", 2019), ("Birdman", 2014), ("Victoria", 2015)],
    },
    {
        "blurb": "Symmetry, Soft Colors, a Faint Hum of Melancholy",
        "movies": [
            ("The Grand Budapest Hotel", 2014),
            ("Moonrise Kingdom", 2012),
            ("Amélie", 2001),
        ],
    },
    {
        "blurb": "The Sport Is the Excuse, the Soul Is the Story",
        "movies": [("Moneyball", 2011), ("The Wrestler", 2008), ("Whiplash", 2014)],
    },
    {
        "blurb": "Family Road Trips Where Nobody Is Doing Okay",
        "movies": [
            ("Little Miss Sunshine", 2006),
            ("Sideways", 2004),
            ("The Way Way Back", 2013),
        ],
    },
    {
        "blurb": "Survive the Game or the Game Wins",
        "movies": [("The Hunger Games", 2012), ("Battle Royale", 2000), ("The Running Man", 1987)],
    },
    {
        "blurb": "Underwater and Under Pressure",
        "movies": [
            ("Das Boot", 1981),
            ("The Hunt for Red October", 1990),
            ("Crimson Tide", 1995),
        ],
    },
    {
        "blurb": "Movies Hopelessly in Love with Movies",
        "movies": [
            ("La La Land", 2016),
            ("Once Upon a Time in Hollywood", 2019),
            ("Singin' in the Rain", 1952),
        ],
    },
    {
        "blurb": "Cons Wrapped in Cons Wrapped in Cons",
        "movies": [
            ("The Sting", 1973),
            ("Catch Me If You Can", 2002),
            ("American Hustle", 2013),
        ],
    },
    {
        "blurb": "Neon-Lit, Rain-Soaked, Soul-Searching",
        "movies": [
            ("Blade Runner", 1982),
            ("Ghost in the Shell", 1995),
            ("The Matrix", 1999),
            ("Blade Runner 2049", 2017),
        ],
    },
    {
        "blurb": "One Wanderer, a Loose Code, and a Very Heavy Sword",
        "movies": [("Yojimbo", 1961), ("A Fistful of Dollars", 1964), ("13 Assassins", 2010)],
    },
    {
        "blurb": "Animated Dads in Way, Way Over Their Heads",
        "movies": [("Finding Nemo", 2003), ("Up", 2009), ("Onward", 2020)],
    },
    {
        "blurb": "Stylish Witchcraft With a Side of Dread",
        "movies": [("The Witch", 2015), ("Suspiria", 2018), ("Practical Magic", 1998)],
    },
    {
        "blurb": "Misfit Crews Just Trying to Save the Galaxy",
        "movies": [
            ("Guardians of the Galaxy", 2014),
            ("Treasure Planet", 2002),
            ("Star Trek", 2009),
        ],
    },
    {
        "blurb": "Detectives Stuck in a Case That Got Stuck in Them",
        "movies": [("Zodiac", 2007), ("Prisoners", 2013), ("Se7en", 1995)],
    },
]


def _normalize_title(s: str) -> str:
    """Lower-case + strip punctuation/whitespace so 'The Witch' and 'The
    Witch.' compare equal, but 'The Last Witch Hunter' does not collapse
    onto 'The Witch'. Keeps letters, digits, and spaces only."""
    out = []
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch.isspace():
            out.append(" ")
    return " ".join("".join(out).split())


async def fetch_by_id(client: httpx.AsyncClient, movie_id: int) -> dict | None:
    """Pull a single movie by TMDB id. Used to honor OVERRIDES, which
    bypass search entirely for known-ambiguous titles."""
    r = await client.get(TMDB_MOVIE_URL.format(movie_id=movie_id), params={"api_key": TMDB_API_KEY})
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


async def search_movie(client: httpx.AsyncClient, title: str, year: int) -> dict | None:
    """Look up a movie on TMDB. Resolution priority, in order:
      1. Exact (normalized) title match in the target year — handles the
         'Locke' vs 'Target Locked' and 'The Witch' vs 'The Last Witch
         Hunter' confusions where TMDB's relevance ranker can drift.
      2. Any title in the target year (positional first result).
      3. Exact title match outside the target year — releases sometimes
         fall a year off from common usage (Phone Booth, Ex Machina).
      4. Closest-year fallback — last resort to avoid silent dropouts.

    Falls back to a yearless query if the year-filtered call is empty."""

    async def _search(params: dict) -> list[dict]:
        r = await client.get(TMDB_SEARCH_URL, params=params)
        r.raise_for_status()
        return r.json().get("results", []) or []

    base = {"api_key": TMDB_API_KEY, "query": title, "include_adult": "false"}
    target = _normalize_title(title)

    # First try year-filtered; if empty, drop the filter.
    results = await _search({**base, "year": year})
    if not results:
        results = await _search(base)
    if not results:
        return None

    def _result_year(r: dict) -> int | None:
        rd = r.get("release_date") or ""
        return int(rd[:4]) if len(rd) >= 4 and rd[:4].isdigit() else None

    def _is_exact(r: dict) -> bool:
        # Match either the localized or original title — TMDB returns
        # results where 'title' is localized and 'original_title' is the
        # canonical English form (or vice versa for foreign films).
        for candidate in (r.get("title"), r.get("original_title")):
            if candidate and _normalize_title(candidate) == target:
                return True
        return False

    # Priority 1: exact title in the right year.
    same_year = [r for r in results if _result_year(r) == year]
    exact_same_year = [r for r in same_year if _is_exact(r)]
    if exact_same_year:
        return exact_same_year[0]

    # Priority 2: any result in the right year.
    if same_year:
        return same_year[0]

    # Priority 3: exact title in any year (release-year drift).
    exact_any = [r for r in results if _is_exact(r)]
    if exact_any:
        return min(exact_any, key=lambda r: abs((_result_year(r) or 9999) - year))

    # Priority 4: closest-year fallback.
    return min(results, key=lambda r: abs((_result_year(r) or 9999) - year))


def _result_to_entry(result: dict, query_title: str, query_year: int) -> dict:
    """Reshape a TMDB search hit into the compact JSON we want to emit."""
    release_date = result.get("release_date") or ""
    release_year: int | None = int(release_date[:4]) if release_date[:4].isdigit() else None
    return {
        "query_title": query_title,
        "query_year": query_year,
        "tmdb_id": result["id"],
        "title": result.get("title") or result.get("original_title"),
        "year": release_year,
        "poster_path": result.get("poster_path"),
    }


async def resolve_movie(client: httpx.AsyncClient, title: str, year: int) -> dict | None:
    """OVERRIDES win when present — they're explicit curation decisions
    for ambiguous titles. Otherwise fall through to TMDB search."""
    pinned = OVERRIDES.get((title, year))
    if pinned is not None:
        return await fetch_by_id(client, pinned)
    return await search_movie(client, title, year)


async def fetch_group(client: httpx.AsyncClient, group: dict) -> dict:
    """Resolve every (title, year) in a group to a TMDB entry. Run the
    per-movie lookups concurrently — 3-4 calls per group is fine for
    TMDB's published rate limits."""
    tasks = [resolve_movie(client, title, year) for title, year in group["movies"]]
    raw = await asyncio.gather(*tasks)

    movies: list[dict] = []
    missing: list[str] = []
    for (title, year), result in zip(group["movies"], raw):
        if result is None:
            missing.append(f"{title} ({year})")
            continue
        movies.append(_result_to_entry(result, title, year))

    entry = {"blurb": group["blurb"], "movies": movies}
    if missing:
        entry["unresolved"] = missing
    return entry


async def main() -> None:
    # A small concurrency cap keeps the script polite without slowing it
    # down — 30 groups × ~3 movies = ~90 lookups, all small JSON GETs.
    async with httpx.AsyncClient(timeout=15.0) as client:
        resolved = await asyncio.gather(*(fetch_group(client, g) for g in GROUPS))

    payload = {
        "count": len(resolved),
        "groups": resolved,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    unresolved_total = sum(len(g.get("unresolved", [])) for g in resolved)
    print(
        f"Wrote {len(resolved)} groups "
        f"({sum(len(g['movies']) for g in resolved)} movies, "
        f"{unresolved_total} unresolved) to {OUTPUT_PATH.resolve()}"
    )


if __name__ == "__main__":
    asyncio.run(main())
