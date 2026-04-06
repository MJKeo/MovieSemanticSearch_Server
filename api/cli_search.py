"""
cli_search.py — Run a search query against the full pipeline from the terminal.

Usage examples:
    # Simple query
    python -m api.cli_search "leonardo dicaprio movies"

    # With genre filters
    python -m api.cli_search "scary movies" --genres horror thriller

    # With release date range (unix timestamps)
    python -m api.cli_search "90s comedies" --min-release-ts 631152000 --max-release-ts 946684800

    # With runtime filter and result limit
    python -m api.cli_search "short animated films" --max-runtime 90 --top 10

    # With maturity rating filter
    python -m api.cli_search "family friendly adventure" --max-maturity-rank 3

    # List available genre and language names
    python -m api.cli_search --list-genres
    python -m api.cli_search --list-languages
"""

import argparse
import asyncio
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Ensure the project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

load_dotenv(Path(_project_root) / ".env")

from db.postgres import pool, fetch_movie_cards
from db.redis import init_redis, close_redis
from db.qdrant import qdrant_client
from db.search import search
from implementation.classes.schemas import MetadataFilters
from implementation.classes.enums import Genre
from implementation.classes.languages import Language


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a search query against the movie search pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            '  python -m api.cli_search "leonardo dicaprio movies"\n'
            '  python -m api.cli_search "scary movies" --genres horror thriller\n'
            '  python -m api.cli_search "90s comedies" --min-release-ts 631152000 --max-release-ts 946684800\n'
            '  python -m api.cli_search --list-genres\n'
        ),
    )

    # Query (positional, optional so --list-* flags work alone)
    parser.add_argument("query", nargs="?", help="Natural language search query")

    # Hard filters
    parser.add_argument("--min-release-ts", type=int, default=None,
                        help="Minimum release date as unix timestamp")
    parser.add_argument("--max-release-ts", type=int, default=None,
                        help="Maximum release date as unix timestamp")
    parser.add_argument("--min-runtime", type=int, default=None,
                        help="Minimum runtime in minutes")
    parser.add_argument("--max-runtime", type=int, default=None,
                        help="Maximum runtime in minutes")
    parser.add_argument("--min-maturity-rank", type=int, default=None,
                        help="Minimum maturity rank")
    parser.add_argument("--max-maturity-rank", type=int, default=None,
                        help="Maximum maturity rank")
    parser.add_argument("--genres", nargs="+", default=None,
                        help="Genre filter(s), e.g. horror comedy sci-fi")
    parser.add_argument("--audio-languages", nargs="+", default=None,
                        help="Audio language filter(s), e.g. english spanish")
    parser.add_argument("--watch-offer-keys", nargs="+", type=int, default=None,
                        help="Watch provider key filter(s) as integers")

    # Display options
    parser.add_argument("--top", type=int, default=25,
                        help="Number of results to display (default: 25)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show timing breakdown and channel weights")

    # Info flags
    parser.add_argument("--list-genres", action="store_true",
                        help="Print all available genre names and exit")
    parser.add_argument("--list-languages", action="store_true",
                        help="Print all available language names and exit")

    return parser.parse_args()


def resolve_genres(names: list[str]) -> list[Genre]:
    """Resolve genre name strings to Genre enum members."""
    resolved = []
    for name in names:
        genre = Genre.from_string(name)
        if genre is None:
            print(f"Warning: unrecognized genre '{name}', skipping. Use --list-genres to see valid names.")
        else:
            resolved.append(genre)
    return resolved


def resolve_languages(names: list[str]) -> list[Language]:
    """Resolve language name strings to Language enum members."""
    resolved = []
    for name in names:
        # Try case-insensitive match against Language enum values
        match = None
        for lang in Language:
            if lang.value.lower() == name.lower():
                match = lang
                break
        if match is None:
            print(f"Warning: unrecognized language '{name}', skipping. Use --list-languages to see valid names.")
        else:
            resolved.append(match)
    return resolved


def build_filters(args: argparse.Namespace) -> MetadataFilters:
    """Build MetadataFilters from parsed CLI arguments."""
    genres = resolve_genres(args.genres) if args.genres else None
    languages = resolve_languages(args.audio_languages) if args.audio_languages else None

    return MetadataFilters(
        min_release_ts=args.min_release_ts,
        max_release_ts=args.max_release_ts,
        min_runtime=args.min_runtime,
        max_runtime=args.max_runtime,
        min_maturity_rank=args.min_maturity_rank,
        max_maturity_rank=args.max_maturity_rank,
        genres=genres if genres else None,
        audio_languages=languages if languages else None,
        watch_offer_keys=args.watch_offer_keys,
    )


async def run_search(query: str, filters: MetadataFilters, top_n: int, verbose: bool) -> None:
    """Initialize connections, run the search, and print results."""
    # Initialize database connections
    await pool.open()
    await init_redis()

    try:
        start = time.perf_counter()
        result = await search(
            query=query,
            metadata_filters=filters,
            qdrant_client=qdrant_client,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        candidates = result.candidates
        if not candidates:
            print("No results found.")
            return

        # Resolve movie titles for display
        display_ids = [c.movie_id for c in candidates[:top_n]]
        cards = await fetch_movie_cards(display_ids)
        title_map: dict[int, tuple[str, int | None]] = {}
        for card in cards:
            mid = card["movie_id"]
            title = card["title"] or f"[id:{mid}]"
            release_ts = card.get("release_ts")
            year = datetime.fromtimestamp(release_ts, tz=timezone.utc).year if release_ts else None
            title_map[mid] = (title, year)

        # Print results
        print(f"\n{'=' * 70}")
        print(f"TOP {min(top_n, len(candidates))} RESULTS — {query!r}")
        print(f"{'=' * 70}")

        for i, c in enumerate(candidates[:top_n], 1):
            title, year = title_map.get(c.movie_id, (f"[id:{c.movie_id}]", None))
            year_str = f" ({year})" if year else ""
            print(f"  {i:>3}. {c.final_score:.4f}  {title}{year_str}")

        print(f"\n{len(candidates)} total candidates in {elapsed_ms:.0f}ms")

        # Verbose output: timing and channel weights
        if verbose:
            dbg = result.debug
            print(f"\n{'=' * 70}")
            print("TIMING BREAKDOWN")
            print(f"{'=' * 70}")
            print(f"  Lexical search total:      {dbg.lexical_debug.latency_ms:>8.0f}ms")
            print(f"    LLM entity extraction:   {dbg.lexical_debug.llm_generation_time_ms:>8.0f}ms")
            if dbg.metadata_preferences_debug:
                print(f"  Metadata preferences LLM:  {dbg.metadata_preferences_debug.llm_generation_time_ms:>8.0f}ms")
            if dbg.channel_weights_debug:
                print(f"  Channel weights LLM:       {dbg.channel_weights_debug.llm_generation_time_ms:>8.0f}ms")
            if dbg.vector_debug:
                vd = dbg.vector_debug
                print(f"  Vector search wall clock:  {vd.wall_clock_ms:>8.0f}ms  ({vd.total_jobs_executed} jobs, {vd.total_candidates} candidates)")
            print(f"\n  Total end-to-end:          {elapsed_ms:>8.0f}ms")

            # Show channel weight details if available
            if dbg.channel_weights_debug and dbg.channel_weights_debug.channel_weights:
                cw = dbg.channel_weights_debug.channel_weights
                print(f"\n{'=' * 70}")
                print("CHANNEL WEIGHTS (LLM raw)")
                print(f"{'=' * 70}")
                print(f"  Vector:   {cw.vector_relevance}")
                print(f"  Lexical:  {cw.lexical_relevance}")
                print(f"  Metadata: {cw.metadata_relevance}")

            # Show active filters
            if filters.is_active:
                print(f"\n{'=' * 70}")
                print("ACTIVE HARD FILTERS")
                print(f"{'=' * 70}")
                if filters.min_release_ts is not None:
                    print(f"  Min release: {datetime.fromtimestamp(filters.min_release_ts, tz=timezone.utc).date()}")
                if filters.max_release_ts is not None:
                    print(f"  Max release: {datetime.fromtimestamp(filters.max_release_ts, tz=timezone.utc).date()}")
                if filters.min_runtime is not None:
                    print(f"  Min runtime: {filters.min_runtime} min")
                if filters.max_runtime is not None:
                    print(f"  Max runtime: {filters.max_runtime} min")
                if filters.min_maturity_rank is not None:
                    print(f"  Min maturity rank: {filters.min_maturity_rank}")
                if filters.max_maturity_rank is not None:
                    print(f"  Max maturity rank: {filters.max_maturity_rank}")
                if filters.genres is not None:
                    print(f"  Genres: {', '.join(g.value for g in filters.genres)}")
                if filters.audio_languages is not None:
                    print(f"  Languages: {', '.join(l.value for l in filters.audio_languages)}")
                if filters.watch_offer_keys is not None:
                    print(f"  Watch provider keys: {filters.watch_offer_keys}")

    finally:
        await close_redis()
        await pool.close()


def main() -> None:
    args = parse_args()

    # Handle info flags
    if args.list_genres:
        print("Available genres:")
        for g in Genre:
            print(f"  {g.normalized_name}")
        return

    if args.list_languages:
        print("Available languages:")
        for lang in Language:
            print(f"  {lang.value.lower()}")
        return

    if not args.query:
        print("Error: query is required (unless using --list-genres or --list-languages)")
        sys.exit(1)

    filters = build_filters(args)
    asyncio.run(run_search(args.query, filters, args.top, args.verbose))


if __name__ == "__main__":
    main()
