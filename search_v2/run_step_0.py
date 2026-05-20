# Search V2 — Step 0 runner
#
# Thin CLI wrapper around `search_v2.step_0.run_step_0` that prints the
# full structured response plus timing and token usage. The query is
# accepted as a positional argument with a built-in default so the
# script can be invoked with no arguments for a quick smoke test.
#
# Whichever entity flow Step 0 selects (specific_title, similarity_to_
# titles, character_franchise, non_character_franchise, studio, actor),
# the runner additionally invokes the corresponding standalone search
# flow and pretty-prints the ranked candidates. This requires Postgres
# connectivity, so the pool is opened on demand.
#
# Usage:
#   python -m search_v2.run_step_0
#   python -m search_v2.run_step_0 "your query here"

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

# Load .env BEFORE importing db.postgres — its module-level
# `pool = AsyncConnectionPool(conninfo=_build_conninfo(), ...)` reads
# POSTGRES_* env vars at import time, so we need them in os.environ first.
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from db.postgres import fetch_movie_cards, pool as postgres_pool  # noqa: E402
from schemas.step_0_flow_routing import (  # noqa: E402
    ActorFlowData,
    CharacterFranchiseFlowData,
    EntityFlow,
    ExactTitleFlowData,
    NonCharacterFranchiseFlowData,
    SimilarityFlowData,
    Step0Response,
    StudioFlowData,
)
from search_v2.character_franchise_search import (  # noqa: E402
    CharacterFranchiseSearchResult,
    run_character_franchise_search,
)
from search_v2.exact_title_search import (  # noqa: E402
    ExactTitleSearchResult,
    run_exact_title_search,
)
from search_v2.non_character_franchise_search import (  # noqa: E402
    NonCharacterFranchiseSearchResult,
    run_non_character_franchise_search,
)
from search_v2.similar_movies import (  # noqa: E402
    SimilarMoviesSearchResult,
    run_similarity_search,
)
from search_v2.step_0 import run_step_0  # noqa: E402
from search_v2.studio_search import (  # noqa: E402
    StudioSearchResult,
    run_studio_search,
)
from search_v2.actor_search import (  # noqa: E402
    ActorSearchResult,
    run_actor_search,
)


# Default query used when the user invokes the script with no
# argument. Picked to exercise a full-coverage title with genuine
# alternate reading — a classic boundary case for flow routing.
_DEFAULT_QUERY = "scary movie"

# Cap printed standalone-flow result lists. Long franchises and broad
# similarity lanes can produce 100+ rows; capping keeps CLI output
# scannable while the full result set remains available in-memory.
_MAX_PRINTED_RESULTS = 25


def _print_response(response: Step0Response) -> None:
    """Pretty-print the full Step0Response payload as indented JSON."""
    payload = response.model_dump()
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run search_v2.step_0 on a query and print the full "
            "response, elapsed time, and token usage. If the exact-"
            "title flow fires, also run the exact-title search and "
            "print its ranked output."
        )
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=_DEFAULT_QUERY,
        help=(
            "The raw user query to process. Defaults to a built-in "
            "sample query if omitted."
        ),
    )
    return parser.parse_args()


def _year_of(release_ts: int | None) -> int | None:
    """Convert Unix-seconds release_ts → calendar year (UTC), or None."""
    if release_ts is None:
        return None
    return datetime.fromtimestamp(release_ts, tz=timezone.utc).year


async def _ensure_postgres_open() -> None:
    """Lazy-open the Postgres pool — no-op when already open.

    Mirrors run_orchestrator._ensure_db_ready: the pool is created with
    open=False at import time, so a CLI invocation has to open it
    before any query helper can run. We don't need Redis or Qdrant for
    the exact-title search.
    """
    if postgres_pool._closed:
        await postgres_pool.open()


async def _print_exact_title_results(
    flow_data: ExactTitleFlowData,
    result: ExactTitleSearchResult,
) -> None:
    """Render the exact-title ranked output as a compact table.

    Fetches movie cards in a single bulk call to resolve title and year
    for the top-N rows; the underlying ranked list is unmodified.
    """
    print("\n[exact_title_search]")
    print(
        f"input: title={flow_data.exact_title_to_search!r}, "
        f"release_year={flow_data.release_year}"
    )

    if not result.ranked:
        print("results: (no matches)")
        return

    top = result.ranked[:_MAX_PRINTED_RESULTS]
    movie_ids = [mid for mid, _ in top]
    cards = await fetch_movie_cards(movie_ids)
    by_id = {card["movie_id"]: card for card in cards}

    total = len(result.ranked)
    shown = len(top)
    suffix = f" (showing top {shown} of {total})" if total > shown else ""
    print(f"results: {total} total{suffix}")

    # Layout: rank | score | source | year | title (movie_id)
    # Column widths are sized to the constants and a typical title;
    # long titles are left to wrap naturally rather than truncated so
    # diagnostic output stays faithful.
    for rank, (mid, score) in enumerate(top, start=1):
        card = by_id.get(mid)
        title = card["title"] if card else "<missing card>"
        year = _year_of(card["release_ts"]) if card else None
        year_str = str(year) if year is not None else "----"
        source = result.score_source.get(mid, "?")
        print(
            f"  {rank:>2}. {score:>5.3f}  {source:<18}  "
            f"{year_str}  {title} ({mid})"
        )


async def _print_similarity_results(
    flow_data: SimilarityFlowData,
    result: SimilarMoviesSearchResult,
) -> None:
    """Render the similar-movies ranked output as a compact table."""
    print("\n[similar_movies_search]")
    # references is a list — render it compactly so the runner stays
    # readable for both single-anchor and multi-anchor frames.
    ref_strs = [
        f"{ref.similar_search_title!r}"
        + (f"@{ref.release_year}" if ref.release_year is not None else "")
        for ref in flow_data.references
    ]
    print(f"input: references=[{', '.join(ref_strs)}]")
    if result.anchor_movie_ids:
        print(f"anchor_tmdb_ids: {result.anchor_movie_ids}")
    if result.active_anchor_types:
        print(f"active_anchor_types: {', '.join(result.active_anchor_types)}")
    if result.debug.normalized_lane_weights:
        weights = ", ".join(
            f"{lane}={weight:.3f}"
            for lane, weight in result.debug.normalized_lane_weights.items()
        )
        print(f"lane_weights: {weights}")

    if not result.ranked:
        print("results: (no matches)")
        return

    top = result.ranked[:_MAX_PRINTED_RESULTS]
    movie_ids = [item.movie_id for item in top]
    cards = await fetch_movie_cards(movie_ids)
    by_id = {card["movie_id"]: card for card in cards}

    total = len(result.ranked)
    shown = len(top)
    suffix = f" (showing top {shown} of {total})" if total > shown else ""
    print(f"results: {total} total{suffix}")

    for rank, item in enumerate(top, start=1):
        card = by_id.get(item.movie_id)
        title = card["title"] if card else "<missing card>"
        year = _year_of(card["release_ts"]) if card else None
        year_str = str(year) if year is not None else "----"
        lanes = ",".join(item.evidence.candidate_sources)
        print(
            f"  {rank:>2}. {item.score:>5.3f}  "
            f"{item.evidence.dominant_lane:<10}  {lanes:<36}  "
            f"{year_str}  {title} ({item.movie_id})"
        )


async def _print_non_character_franchise_results(
    flow_data: NonCharacterFranchiseFlowData,
    result: NonCharacterFranchiseSearchResult,
) -> None:
    """Render the non-character franchise output as two ordered tables.

    Hydrates titles + years for the union of both buckets in one bulk
    call. Primary bucket prints first, then secondary, mirroring the
    append-after-sort algorithm.
    """
    print("\n[non_character_franchise_search]")
    print(f"input: canonical_name={flow_data.canonical_name!r}")

    total_primary = len(result.primary_franchise)
    total_secondary = len(result.secondary_franchise)
    if total_primary == 0 and total_secondary == 0:
        print("results: (no matches — canonical_name may not resolve)")
        return

    # Bulk-fetch cards covering each bucket's print cap independently —
    # taking the first 2*_MAX_PRINTED_RESULTS from the concatenated list
    # would under-hydrate the secondary bucket when primary > _MAX cap.
    # The union is already deduplicated by the SQL CASE (each movie
    # lands in exactly one bucket).
    movie_ids_to_hydrate = (
        result.primary_franchise[:_MAX_PRINTED_RESULTS]
        + result.secondary_franchise[:_MAX_PRINTED_RESULTS]
    )
    cards = await fetch_movie_cards(movie_ids_to_hydrate)
    by_id = {card["movie_id"]: card for card in cards}

    def _print_bucket(label: str, movie_ids: list[int], total: int) -> None:
        if not movie_ids:
            print(f"\n{label}: (empty)")
            return
        top = movie_ids[:_MAX_PRINTED_RESULTS]
        suffix = (
            f" (showing top {len(top)} of {total})"
            if total > len(top)
            else ""
        )
        print(f"\n{label}: {total} total{suffix}")
        for rank, mid in enumerate(top, start=1):
            card = by_id.get(mid)
            title = card["title"] if card else "<missing card>"
            year = _year_of(card["release_ts"]) if card else None
            year_str = str(year) if year is not None else "----"
            print(f"  {rank:>2}. {year_str}  {title} ({mid})")

    _print_bucket("primary_franchise", result.primary_franchise, total_primary)
    _print_bucket("secondary_franchise", result.secondary_franchise, total_secondary)


async def _print_character_franchise_results(
    flow_data: CharacterFranchiseFlowData,
    result: CharacterFranchiseSearchResult,
) -> None:
    """Render the character-franchise output as seven ordered tier tables.

    Hydrates titles + years for the union of every tier (each capped
    independently at _MAX_PRINTED_RESULTS) in one bulk call, mirroring
    the non-character franchise printer. The seven tiers are strictly
    disjoint by construction so the union is already deduplicated.
    """
    print("\n[character_franchise_search]")
    print(f"input: canonical_name={flow_data.canonical_name!r}")

    tiers: list[tuple[str, list[int]]] = [
        ("tier_1_lineage_mainline",       result.tier_1_lineage_mainline),
        ("tier_2_top_billed_appearance",  result.tier_2_top_billed_appearance),
        ("tier_3_lineage_ancillary",      result.tier_3_lineage_ancillary),
        ("tier_4_universe",               result.tier_4_universe),
        ("tier_5_prominent_appearance",   result.tier_5_prominent_appearance),
        ("tier_6_relevant_appearance",    result.tier_6_relevant_appearance),
        ("tier_7_minor_appearance",       result.tier_7_minor_appearance),
    ]

    if not any(movie_ids for _, movie_ids in tiers):
        print("results: (no matches — neither character nor franchise side resolved)")
        return

    # Bulk-fetch covering each tier's print cap independently — taking
    # the first N from the concatenated list would under-hydrate the
    # lower tiers when an upper tier exceeds the cap.
    movie_ids_to_hydrate: list[int] = []
    for _, movie_ids in tiers:
        movie_ids_to_hydrate.extend(movie_ids[:_MAX_PRINTED_RESULTS])
    cards = await fetch_movie_cards(movie_ids_to_hydrate)
    by_id = {card["movie_id"]: card for card in cards}

    def _print_tier(label: str, movie_ids: list[int]) -> None:
        total = len(movie_ids)
        if total == 0:
            print(f"\n{label}: (empty)")
            return
        top = movie_ids[:_MAX_PRINTED_RESULTS]
        suffix = (
            f" (showing top {len(top)} of {total})"
            if total > len(top)
            else ""
        )
        print(f"\n{label}: {total} total{suffix}")
        for rank, mid in enumerate(top, start=1):
            card = by_id.get(mid)
            title = card["title"] if card else "<missing card>"
            year = _year_of(card["release_ts"]) if card else None
            year_str = str(year) if year is not None else "----"
            print(f"  {rank:>2}. {year_str}  {title} ({mid})")

    for label, movie_ids in tiers:
        _print_tier(label, movie_ids)


async def _print_studio_results(
    flow_data: StudioFlowData,
    result: StudioSearchResult,
) -> None:
    """Render the studio-search ranked output as a single flat table.

    Studio flow is a single-tier popularity sort — no per-movie score,
    no buckets — so the printer mirrors the within-tier renderer used
    by the franchise printers, just without the bucket labels.
    """
    print("\n[studio_search]")
    canonical = [ref.canonical_name for ref in flow_data.references]
    print(f"input: canonical_names={canonical!r}")

    total = len(result.ranked)
    if total == 0:
        print("results: (no matches — none of the named studios resolved)")
        return

    top = result.ranked[:_MAX_PRINTED_RESULTS]
    cards = await fetch_movie_cards(top)
    by_id = {card["movie_id"]: card for card in cards}

    suffix = f" (showing top {len(top)} of {total})" if total > len(top) else ""
    print(f"results: {total} total{suffix}")
    for rank, mid in enumerate(top, start=1):
        card = by_id.get(mid)
        title = card["title"] if card else "<missing card>"
        year = _year_of(card["release_ts"]) if card else None
        year_str = str(year) if year is not None else "----"
        print(f"  {rank:>2}. {year_str}  {title} ({mid})")


async def _print_actor_results(
    flow_data: ActorFlowData,
    result: ActorSearchResult,
) -> None:
    """Render the actor-search output as four bucket tables.

    Actor flow returns movies grouped by prominence: lead → major →
    has-relevance → minor/cameo. Each bucket is independently
    popularity-sorted, capped at _MAX_PRINTED_RESULTS for the CLI.
    Mirrors the franchise-tier printer with four buckets instead of
    seven.
    """
    print("\n[actor_search]")
    canonical = [ref.canonical_name for ref in flow_data.references]
    print(f"input: canonical_names={canonical!r}")

    buckets: list[tuple[str, list[int]]] = [
        ("bucket_1_lead",     result.bucket_1_lead),
        ("bucket_2_major",    result.bucket_2_major),
        ("bucket_3_relevant", result.bucket_3_relevant),
        ("bucket_4_minor",    result.bucket_4_minor),
    ]

    if not any(movie_ids for _, movie_ids in buckets):
        print(
            "results: (no matches — either an actor failed to resolve or "
            "no movie had all named actors)"
        )
        return

    # Bulk-fetch covering each bucket's print cap independently — same
    # idiom as the character-franchise tier printer.
    movie_ids_to_hydrate: list[int] = []
    for _, movie_ids in buckets:
        movie_ids_to_hydrate.extend(movie_ids[:_MAX_PRINTED_RESULTS])
    cards = await fetch_movie_cards(movie_ids_to_hydrate)
    by_id = {card["movie_id"]: card for card in cards}

    def _print_bucket(label: str, movie_ids: list[int]) -> None:
        total = len(movie_ids)
        if total == 0:
            print(f"\n{label}: (empty)")
            return
        top = movie_ids[:_MAX_PRINTED_RESULTS]
        suffix = (
            f" (showing top {len(top)} of {total})"
            if total > len(top)
            else ""
        )
        print(f"\n{label}: {total} total{suffix}")
        for rank, mid in enumerate(top, start=1):
            card = by_id.get(mid)
            title = card["title"] if card else "<missing card>"
            year = _year_of(card["release_ts"]) if card else None
            year_str = str(year) if year is not None else "----"
            print(f"  {rank:>2}. {year_str}  {title} ({mid})")

    for label, movie_ids in buckets:
        _print_bucket(label, movie_ids)


async def _main_async() -> None:
    args = _parse_args()
    print(f"[query] {args.query}\n")

    # run_step_0 returns (response, input_tokens, output_tokens) — no
    # elapsed time, so we measure it here.
    start = time.perf_counter()
    response, in_tok, out_tok = await run_step_0(args.query)
    elapsed = time.perf_counter() - start

    print("[response]")
    _print_response(response)

    print(
        f"\n[stats] elapsed={elapsed:.2f}s "
        f"input_tokens={in_tok} output_tokens={out_tok}"
    )

    # Surface the derived routing decisions so the CLI output matches
    # what an orchestrator would see (the LLM doesn't emit these
    # fields; they're computed in Step0Response).
    print(
        f"[derived] fire_standard_flow={response.fire_standard_flow} "
        f"primary_flow={response.primary_flow.value}"
    )

    # Conditionally execute the executor that matches the selected
    # entity flow. Every non-NONE entity flow now has its own executor.
    flow = response.selected_entity_flow
    if flow == EntityFlow.SPECIFIC_TITLE:
        flow_data = response.to_exact_title_flow_data()
        assert flow_data is not None  # cardinality validator guarantees this
        await _ensure_postgres_open()
        ets_start = time.perf_counter()
        ets_result = await run_exact_title_search(flow_data)
        ets_elapsed = time.perf_counter() - ets_start
        await _print_exact_title_results(flow_data, ets_result)
        print(f"[exact_title_stats] elapsed={ets_elapsed:.2f}s")
    elif flow == EntityFlow.SIMILARITY_TO_TITLES:
        flow_data = response.to_similarity_flow_data()
        assert flow_data is not None
        await _ensure_postgres_open()
        sim_start = time.perf_counter()
        sim_result = await run_similarity_search(flow_data)
        sim_elapsed = time.perf_counter() - sim_start
        await _print_similarity_results(flow_data, sim_result)
        print(f"[similar_movies_stats] elapsed={sim_elapsed:.2f}s")
    elif flow == EntityFlow.NON_CHARACTER_FRANCHISE:
        flow_data = response.to_non_character_franchise_flow_data()
        assert flow_data is not None
        await _ensure_postgres_open()
        ncf_start = time.perf_counter()
        ncf_result = await run_non_character_franchise_search(flow_data)
        ncf_elapsed = time.perf_counter() - ncf_start
        await _print_non_character_franchise_results(flow_data, ncf_result)
        print(f"[non_character_franchise_stats] elapsed={ncf_elapsed:.2f}s")
    elif flow == EntityFlow.CHARACTER_FRANCHISE:
        flow_data = response.to_character_franchise_flow_data()
        assert flow_data is not None
        await _ensure_postgres_open()
        cf_start = time.perf_counter()
        cf_result = await run_character_franchise_search(flow_data)
        cf_elapsed = time.perf_counter() - cf_start
        await _print_character_franchise_results(flow_data, cf_result)
        print(f"[character_franchise_stats] elapsed={cf_elapsed:.2f}s")
    elif flow == EntityFlow.STUDIO:
        flow_data = response.to_studio_flow_data()
        assert flow_data is not None
        await _ensure_postgres_open()
        std_start = time.perf_counter()
        std_result = await run_studio_search(flow_data)
        std_elapsed = time.perf_counter() - std_start
        await _print_studio_results(flow_data, std_result)
        print(f"[studio_stats] elapsed={std_elapsed:.2f}s")
    elif flow == EntityFlow.ACTOR:
        flow_data = response.to_actor_flow_data()
        assert flow_data is not None
        await _ensure_postgres_open()
        act_start = time.perf_counter()
        act_result = await run_actor_search(flow_data)
        act_elapsed = time.perf_counter() - act_start
        await _print_actor_results(flow_data, act_result)
        print(f"[actor_stats] elapsed={act_elapsed:.2f}s")


def main() -> None:
    from implementation.misc.event_loop import install_uvloop
    install_uvloop()
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
