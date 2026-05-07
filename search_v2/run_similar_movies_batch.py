"""Batch runner for the standalone similar-movies flow.

Usage:
    python -m search_v2.run_similar_movies_batch
    python -m search_v2.run_similar_movies_batch --ids 27205 603 --limit 10

Writes JSON by default so result sets can be inspected or handed back to an
LLM without copy-pasting terminal output.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env before importing db.postgres; its module-level pool reads env vars
# when imported.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from db.postgres import fetch_movie_cards, pool as postgres_pool  # noqa: E402
from search_v2.similar_movies import (  # noqa: E402
    ALL_LANES,
    SimilarMoviesSearchResult,
    run_similar_movies_for_ids,
)


DEFAULT_ANCHOR_IDS: tuple[int, ...] = (
    27205,   # Inception
    603,     # The Matrix
    11,      # Star Wars
    862,     # Toy Story
    129,     # Spirited Away
    238,     # The Godfather
    155,     # The Dark Knight
    419430,  # Get Out
    245891,  # John Wick
    872585,  # Oppenheimer
    346698,  # Barbie
    205321,  # Sharknado
    17473,   # The Room
    497,     # The Green Mile
    680,     # Pulp Fiction
    550,     # Fight Club
    105,     # Back to the Future
    157336,  # Interstellar
    120,     # The Lord of the Rings: The Fellowship of the Ring
    597,     # Titanic
)

DEFAULT_JSON_OUT = Path("search_v2/similar_movies_batch_results.json")
DEFAULT_MARKDOWN_OUT = Path("search_v2/similar_movies_batch_results.md")


def _year_of(release_ts: int | None) -> int | None:
    if release_ts is None:
        return None
    return datetime.fromtimestamp(release_ts, tz=timezone.utc).year


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the standalone similar-movies flow for many anchors."
    )
    parser.add_argument(
        "--ids",
        type=int,
        nargs="*",
        default=list(DEFAULT_ANCHOR_IDS),
        help="TMDB IDs to run. Defaults to a broad smoke-test anchor set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of similar-movie results to keep per anchor.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_JSON_OUT,
        help="Path for structured JSON output.",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=DEFAULT_MARKDOWN_OUT,
        help="Path for Markdown table output.",
    )
    return parser.parse_args()


async def _cards_by_id(movie_ids: list[int]) -> dict[int, dict]:
    cards = await fetch_movie_cards(movie_ids)
    return {int(card["movie_id"]): card for card in cards}


async def _serialize_result(
    anchor_id: int,
    result: SimilarMoviesSearchResult,
) -> dict[str, Any]:
    ids = [anchor_id] + [item.movie_id for item in result.ranked]
    cards = await _cards_by_id(ids)
    anchor_card = cards.get(anchor_id, {})

    rows: list[dict[str, Any]] = []
    for rank, item in enumerate(result.ranked, start=1):
        card = cards.get(item.movie_id, {})
        lane_scores = item.evidence.lane_scores
        rows.append(
            {
                "rank": rank,
                "movie_id": item.movie_id,
                "title": card.get("title", "<missing card>"),
                "year": _year_of(card.get("release_ts")),
                "score": round(item.score, 6),
                "dominant_lane": item.evidence.dominant_lane,
                "lanes": list(item.evidence.candidate_sources),
                "lane_scores": {
                    lane: round(lane_scores.get(lane, 0.0), 6)
                    for lane in ALL_LANES
                },
            }
        )

    return {
        "anchor": {
            "movie_id": anchor_id,
            "title": anchor_card.get("title", "<missing card>"),
            "year": _year_of(anchor_card.get("release_ts")),
        },
        "active_anchor_types": list(result.active_anchor_types),
        "lane_weights": {
            lane: round(weight, 6)
            for lane, weight in result.debug.normalized_lane_weights.items()
        },
        "candidate_counts_by_lane": dict(result.debug.candidate_counts_by_lane),
        "results": rows,
    }


def _render_markdown(batch: list[dict[str, Any]]) -> str:
    sections: list[str] = ["# Similar Movies Batch Results", ""]
    for item in batch:
        anchor = item["anchor"]
        sections.append(
            f"## {anchor['title']} ({anchor['year']}) - `{anchor['movie_id']}`"
        )
        sections.append("")
        sections.append(
            f"Active anchor types: {', '.join(item['active_anchor_types']) or '(none)'}"
        )
        sections.append("")
        sections.append("| # | Result | Score | Dominant | Lanes |")
        sections.append("|---:|---|---:|---|---|")
        for row in item["results"]:
            title = f"{row['title']} ({row['year']})"
            lanes = ", ".join(row["lanes"])
            sections.append(
                f"| {row['rank']} | {title} `{row['movie_id']}` | "
                f"{row['score']:.3f} | {row['dominant_lane']} | {lanes} |"
            )
        sections.append("")
    return "\n".join(sections)


def _print_compact(batch: list[dict[str, Any]], *, limit: int) -> None:
    for item in batch:
        anchor = item["anchor"]
        print()
        print(
            f"{anchor['title']} ({anchor['year']}) "
            f"[tmdb={anchor['movie_id']}]"
        )
        print(f"  active: {', '.join(item['active_anchor_types'])}")
        for row in item["results"][:limit]:
            lanes = ",".join(row["lanes"])
            print(
                f"  {row['rank']:>2}. {row['score']:.3f} "
                f"{row['dominant_lane']:<10} {lanes:<32} "
                f"{row['title']} ({row['year']}) [{row['movie_id']}]"
            )


async def _main_async() -> None:
    args = _parse_args()
    if not args.ids:
        raise ValueError("at least one TMDB ID is required.")

    if postgres_pool._closed:
        await postgres_pool.open()

    batch: list[dict[str, Any]] = []
    for anchor_id in args.ids:
        result = await run_similar_movies_for_ids([anchor_id], limit=args.limit)
        batch.append(await _serialize_result(anchor_id, result))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(batch, indent=2), encoding="utf-8")
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(_render_markdown(batch), encoding="utf-8")

    _print_compact(batch, limit=args.limit)
    print()
    print(f"Wrote JSON: {args.json_out}")
    print(f"Wrote Markdown: {args.markdown_out}")


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
