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
    49026,   # The Dark Knight Rises — added for V3 H4 (medium piecewise)
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

# V3 multi-anchor cohorts. Each entry is (label, [movie_ids]). The
# first 12 mirror the V2 evaluation set (Pixar / Ghibli / Tarantino /
# war / Nolan etc.) and the last is the Tom Hanks trio added for V3
# H9 verification (cast bucket-with-floor).
DEFAULT_MULTI_ANCHOR_COHORTS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("Nolan trio", (27205, 77, 1124)),                    # Inception, Memento, The Prestige
    ("Pixar trio", (862, 12, 14160)),                     # Toy Story, Finding Nemo, Up
    ("Ghibli trio", (129, 8392, 128)),                    # Spirited Away, My Neighbor Totoro, Princess Mononoke
    ("MCU trio", (1726, 24428, 271110)),                  # Iron Man, Avengers, Civil War
    ("Stephen King horror", (694, 346364, 235)),          # Shining, IT, Misery
    ("Best Picture trio", (238, 424, 76203)),             # Godfather, Schindler's List, 12 Years a Slave
    ("Tarantino trio", (680, 24, 273248)),                # Pulp Fiction, Kill Bill 1, Hateful Eight
    ("Spielberg adventure trio", (1894, 89, 329)),        # Indiana Jones films
    ("WW2 epics", (424, 857, 562)),                       # Schindler, Saving Private Ryan, Das Boot
    ("Slasher trio", (1091, 9716, 4233)),                 # The Thing, Carrie, Halloween
    ("Romcom trio", (114, 1581, 639)),                    # Pretty Woman, Holiday, When Harry Met Sally
    ("Studio Ghibli + Pixar mix", (862, 129, 8392)),      # Cross-tradition cohesion test
    ("Tom Hanks trio (H9)", (2280, 13, 862)),             # Big, Forrest Gump, Toy Story (Hanks lead in all 3)
    ("Female-led / Gerwig", (346698, 391713, 331482)),    # Barbie, Lady Bird, Little Women — auteur cohesion
)

DEFAULT_JSON_OUT = Path("search_v2/similar_movies_batch_results.json")
DEFAULT_MARKDOWN_OUT = Path("search_v2/similar_movies_batch_results.md")
DEFAULT_MULTI_JSON_OUT = Path("search_v2/similar_movies_multi_anchor_results.json")
DEFAULT_MULTI_MARKDOWN_OUT = Path("search_v2/similar_movies_multi_anchor_results.md")


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
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Also run the multi-anchor cohort batch (V3 H9 + V2 cohesion sets).",
    )
    parser.add_argument(
        "--multi-only",
        action="store_true",
        help="Skip single-anchor batch and only run multi-anchor cohorts.",
    )
    parser.add_argument(
        "--multi-json-out",
        type=Path,
        default=DEFAULT_MULTI_JSON_OUT,
        help="Path for multi-anchor JSON output.",
    )
    parser.add_argument(
        "--multi-markdown-out",
        type=Path,
        default=DEFAULT_MULTI_MARKDOWN_OUT,
        help="Path for multi-anchor Markdown output.",
    )
    parser.add_argument(
        "--qdrant-limit",
        type=int,
        default=None,
        help=(
            "Override the per-vector-space Qdrant candidate pool size. "
            "Defaults to the module-level DEFAULT_QDRANT_LIMIT (2000). "
            "Used for sweep experiments comparing recall vs latency."
        ),
    )
    return parser.parse_args()


async def _cards_by_id(movie_ids: list[int]) -> dict[int, dict]:
    cards = await fetch_movie_cards(movie_ids)
    return {int(card["movie_id"]): card for card in cards}


def _row_for_result(
    item: Any,
    rank: int,
    card: dict,
    lane_weights: dict[str, float],
) -> dict[str, Any]:
    """Convert a SimilarMovieResult into a debug-friendly dict.

    The dict captures per-lane raw scores AND per-lane additive
    contributions (raw * weight) so a reader can see at a glance
    which lanes drove the final score. Multipliers and floor
    activations are surfaced verbatim from the V3 diagnostic fields
    on LaneEvidence.
    """
    lane_scores = item.evidence.lane_scores
    contributions = {
        lane: round(lane_weights.get(lane, 0.0) * lane_scores.get(lane, 0.0), 6)
        for lane in ALL_LANES
    }
    return {
        "rank": rank,
        "movie_id": item.movie_id,
        "title": card.get("title", "<missing card>"),
        "year": _year_of(card.get("release_ts")),
        "score": round(item.score, 6),
        "base_score": round(item.evidence.base_score, 6),
        "dominant_lane": item.evidence.dominant_lane,
        "lanes": list(item.evidence.candidate_sources),
        "lane_scores": {
            lane: round(lane_scores.get(lane, 0.0), 6)
            for lane in ALL_LANES
        },
        "lane_contributions": contributions,
        "multipliers": {
            k: round(v, 6) for k, v in item.evidence.multipliers.items()
        },
        "floor_value": round(item.evidence.floor_value, 6),
        "floor_source": item.evidence.floor_source,
    }


async def _serialize_result(
    anchor_id: int,
    result: SimilarMoviesSearchResult,
) -> dict[str, Any]:
    ids = [anchor_id] + [item.movie_id for item in result.ranked]
    cards = await _cards_by_id(ids)
    anchor_card = cards.get(anchor_id, {})

    lane_weights = result.debug.normalized_lane_weights

    rows: list[dict[str, Any]] = []
    for rank, item in enumerate(result.ranked, start=1):
        card = cards.get(item.movie_id, {})
        rows.append(_row_for_result(item, rank, card, lane_weights))

    return {
        "anchor": {
            "movie_id": anchor_id,
            "title": anchor_card.get("title", "<missing card>"),
            "year": _year_of(anchor_card.get("release_ts")),
        },
        "active_anchor_types": list(result.active_anchor_types),
        "lane_weights": {
            lane: round(weight, 6) for lane, weight in lane_weights.items()
        },
        "candidate_counts_by_lane": dict(result.debug.candidate_counts_by_lane),
        "anchor_format_bucket": result.debug.anchor_format_bucket,
        "anchor_medium_tags": list(result.debug.anchor_medium_tags),
        "consensus_countries": [str(c) for c in result.debug.consensus_countries],
        "shorts_dominant": result.debug.shorts_dominant,
        "low_cohesion_fallback_used": result.debug.low_cohesion_fallback_used,
        "results": rows,
    }


def _format_lane_breakdown(row: dict[str, Any]) -> str:
    """Format the per-lane breakdown line: lane=raw*weight=contrib for
    every lane that contributed something. Contributions are sorted
    descending so the reader sees the biggest drivers first."""
    contribs = row["lane_contributions"]
    raws = row["lane_scores"]
    nonzero = [(lane, c) for lane, c in contribs.items() if c > 0.0]
    nonzero.sort(key=lambda x: -x[1])
    if not nonzero:
        return "(no additive contributions)"
    parts = [
        f"{lane}={raws[lane]:.3f}→{contrib:.3f}"
        for lane, contrib in nonzero
    ]
    return "  ".join(parts)


def _format_multipliers_floor(row: dict[str, Any]) -> str:
    """Format the post-additive multipliers + floor activation line."""
    bits: list[str] = []
    for k, v in row["multipliers"].items():
        bits.append(f"×{v:.2f}({k})")
    if row["floor_source"]:
        bits.append(f"floor={row['floor_value']:.3f}({row['floor_source']})")
    if not bits:
        return "(no multipliers / floors)"
    return "  ".join(bits)


def _format_lane_weights_table(weights: dict[str, float]) -> list[str]:
    """One-line summary of the lane weights used for this anchor."""
    nonzero = [(k, v) for k, v in weights.items() if v > 0.0]
    nonzero.sort(key=lambda x: -x[1])
    return [
        "Lane weights: "
        + "  ".join(f"{k}={v:.3f}" for k, v in nonzero)
    ]


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
        sections.append(
            f"Anchor format bucket: {item.get('anchor_format_bucket') or '(none)'}"
        )
        if item.get("consensus_countries"):
            sections.append(
                f"Consensus countries: {', '.join(item['consensus_countries'])}"
            )
        sections.append("")
        sections.extend(_format_lane_weights_table(item["lane_weights"]))
        sections.append("")
        # Headline table.
        sections.append("| # | Result | Score | Base | Dominant | Lanes |")
        sections.append("|---:|---|---:|---:|---|---|")
        for row in item["results"]:
            title = f"{row['title']} ({row['year']})"
            lanes = ", ".join(row["lanes"])
            sections.append(
                f"| {row['rank']} | {title} `{row['movie_id']}` | "
                f"{row['score']:.3f} | {row['base_score']:.3f} | "
                f"{row['dominant_lane']} | {lanes} |"
            )
        sections.append("")
        # Per-row breakdown — what the additive sum was, what
        # multipliers fired, and whether a floor displaced it.
        sections.append("### Per-result breakdown")
        sections.append("")
        for row in item["results"]:
            sections.append(
                f"**#{row['rank']} {row['title']} ({row['year']}) "
                f"`{row['movie_id']}` — score {row['score']:.3f}**"
            )
            sections.append("")
            sections.append(f"- Lanes: {_format_lane_breakdown(row)}")
            sections.append(f"- Adjustments: {_format_multipliers_floor(row)}")
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


async def _serialize_multi_result(
    label: str,
    anchor_ids: list[int],
    result: SimilarMoviesSearchResult,
) -> dict[str, Any]:
    """Serialize a multi-anchor result with the same row schema as
    single-anchor, but with a cohort label and the full anchor list."""
    ids = list(anchor_ids) + [item.movie_id for item in result.ranked]
    cards = await _cards_by_id(ids)
    lane_weights = result.debug.normalized_lane_weights
    rows: list[dict[str, Any]] = []
    for rank, item in enumerate(result.ranked, start=1):
        card = cards.get(item.movie_id, {})
        rows.append(_row_for_result(item, rank, card, lane_weights))
    anchors_meta = [
        {
            "movie_id": aid,
            "title": cards.get(aid, {}).get("title", "<missing card>"),
            "year": _year_of(cards.get(aid, {}).get("release_ts")),
        }
        for aid in anchor_ids
    ]
    return {
        "cohort": label,
        "anchors": anchors_meta,
        "active_anchor_types": list(result.active_anchor_types),
        "lane_weights": {
            lane: round(weight, 6) for lane, weight in lane_weights.items()
        },
        "candidate_counts_by_lane": dict(result.debug.candidate_counts_by_lane),
        "anchor_format_bucket": result.debug.anchor_format_bucket,
        "consensus_countries": [str(c) for c in result.debug.consensus_countries],
        "shorts_dominant": result.debug.shorts_dominant,
        "low_cohesion_fallback_used": result.debug.low_cohesion_fallback_used,
        "vector_space_cohesion": dict(result.debug.vector_space_cohesion),
        "results": rows,
    }


def _render_multi_markdown(batch: list[dict[str, Any]]) -> str:
    sections: list[str] = ["# Similar Movies Multi-Anchor Cohort Results", ""]
    for item in batch:
        anchor_titles = ", ".join(
            f"{a['title']} ({a['year']})" for a in item["anchors"]
        )
        sections.append(f"## {item['cohort']}")
        sections.append("")
        sections.append(f"Anchors: {anchor_titles}")
        sections.append("")
        sections.append(
            f"Active anchor types: {', '.join(item['active_anchor_types']) or '(none)'}"
        )
        sections.append(
            f"Repeated format bucket: {item.get('anchor_format_bucket') or '(none)'}"
        )
        if item.get("consensus_countries"):
            sections.append(
                f"Consensus countries: {', '.join(item['consensus_countries'])}"
            )
        sections.append(
            f"Shorts-dominant cohort: {item.get('shorts_dominant', False)}"
        )
        sections.append(
            f"Low-cohesion fallback used: "
            f"{item.get('low_cohesion_fallback_used', False)}"
        )
        if item.get("vector_space_cohesion"):
            cohesion_top = sorted(
                item["vector_space_cohesion"].items(), key=lambda x: -x[1]
            )[:6]
            sections.append(
                "Top vector-space cohesion: "
                + ", ".join(f"{k}={v:.3f}" for k, v in cohesion_top)
            )
        sections.append("")
        sections.extend(_format_lane_weights_table(item["lane_weights"]))
        sections.append("")
        sections.append("| # | Result | Score | Base | Dominant | Lanes |")
        sections.append("|---:|---|---:|---:|---|---|")
        for row in item["results"]:
            title = f"{row['title']} ({row['year']})"
            lanes = ", ".join(row["lanes"])
            sections.append(
                f"| {row['rank']} | {title} `{row['movie_id']}` | "
                f"{row['score']:.3f} | {row['base_score']:.3f} | "
                f"{row['dominant_lane']} | {lanes} |"
            )
        sections.append("")
        sections.append("### Per-result breakdown")
        sections.append("")
        for row in item["results"]:
            sections.append(
                f"**#{row['rank']} {row['title']} ({row['year']}) "
                f"`{row['movie_id']}` — score {row['score']:.3f}**"
            )
            sections.append("")
            sections.append(f"- Lanes: {_format_lane_breakdown(row)}")
            sections.append(f"- Adjustments: {_format_multipliers_floor(row)}")
            sections.append("")
    return "\n".join(sections)


async def _main_async() -> None:
    args = _parse_args()

    if postgres_pool._closed:
        await postgres_pool.open()

    # Build common kwargs for the engine. Only forward qdrant_limit when
    # the caller explicitly overrode it so the engine's own default
    # (DEFAULT_QDRANT_LIMIT) remains the single source of truth otherwise.
    engine_kwargs: dict[str, Any] = {"limit": args.limit}
    if args.qdrant_limit is not None:
        engine_kwargs["qdrant_limit"] = args.qdrant_limit

    if not args.multi_only:
        if not args.ids:
            raise ValueError("at least one TMDB ID is required.")
        batch: list[dict[str, Any]] = []
        for anchor_id in args.ids:
            result = await run_similar_movies_for_ids([anchor_id], **engine_kwargs)
            batch.append(await _serialize_result(anchor_id, result))

        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(batch, indent=2), encoding="utf-8")
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(_render_markdown(batch), encoding="utf-8")

        _print_compact(batch, limit=args.limit)
        print()
        print(f"Wrote single-anchor JSON: {args.json_out}")
        print(f"Wrote single-anchor Markdown: {args.markdown_out}")

    if args.multi or args.multi_only:
        multi_batch: list[dict[str, Any]] = []
        for label, ids in DEFAULT_MULTI_ANCHOR_COHORTS:
            print(f"  ... cohort: {label}")
            try:
                result = await run_similar_movies_for_ids(list(ids), **engine_kwargs)
            except LookupError as exc:
                # A cohort may reference a movie_id that isn't in the
                # local catalog (e.g., an archival release or one
                # filtered out by the ingestion quality gate). Skip the
                # cohort and surface the gap so the smoke run still
                # produces results for the rest.
                print(f"      SKIPPED: {exc}")
                continue
            multi_batch.append(await _serialize_multi_result(label, list(ids), result))

        args.multi_json_out.parent.mkdir(parents=True, exist_ok=True)
        args.multi_json_out.write_text(
            json.dumps(multi_batch, indent=2), encoding="utf-8"
        )
        args.multi_markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.multi_markdown_out.write_text(
            _render_multi_markdown(multi_batch), encoding="utf-8"
        )
        print()
        print(f"Wrote multi-anchor JSON: {args.multi_json_out}")
        print(f"Wrote multi-anchor Markdown: {args.multi_markdown_out}")


def main() -> None:
    # Install uvloop before the event loop starts. ~2x throughput on
    # socket-heavy fan-outs (Postgres + Qdrant + Redis in parallel) —
    # see implementation/misc/event_loop.py.
    from implementation.misc.event_loop import install_uvloop
    install_uvloop()
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
