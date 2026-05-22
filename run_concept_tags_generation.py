#!/usr/bin/env python3
"""
run_concept_tags_generation.py

Runs concept_tags generation 3 times in parallel for every movie in
concept_tags_test_movies.py, merges the 3 runs with majority-vote rules
(>=2/3 to include), and saves both the merged result and the raw individual
runs as JSON.

Output layout:
    concept_tags_results/<test-name>/<title_first_4_words>.json

Each JSON file contains:
    {
        "title": str,
        "year": int,
        "tmdb_id": int,
        "merged": <ConceptTagsOutput as JSON>,
        "individual_runs": [<ConceptTagsOutput as JSON>, ... ]
    }

Usage:
    python run_concept_tags_generation.py --test-name baseline_v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sqlite3
import time
from pathlib import Path

from concept_tags_test_movies import CONCEPT_TAGS_TEST_MOVIES
from movie_ingestion.metadata_generation.concept_tags_merge import majority_merge
from movie_ingestion.metadata_generation.generators.concept_tags import (
    generate_concept_tags,
)
from schemas.metadata import (
    ConceptTagsOutput,
    NarrativeTechniquesOutput,
    PlotAnalysisOutput,
    PlotEventsOutput,
    ReceptionOutput,
    normalize_legacy_metadata_payload,
)
from schemas.movie_input import MovieInputData


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRACKER_DB = Path("ingestion_data/tracker.db")
RESULTS_ROOT = Path("concept_tags_results")
NUM_RUNS_PER_MOVIE = 3

# Per-attempt LLM timeout for this eval. gpt-5.4-mini at reasoning_effort
# 'none' is much faster than gpt-5-mini at medium, but we keep a 60s
# ceiling because fanning out ~24 movies × 3 runs = ~72 concurrent calls
# inflates tail latency above the router default (25s).
LLM_TIMEOUT_SECONDS = 60.0

# ---------------------------------------------------------------------------
# Single-query data loader
# ---------------------------------------------------------------------------

def _parse_json_list(raw: str | None) -> list:
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


def fetch_all_movie_data(
    tmdb_ids: list[int],
) -> dict[
    int,
    tuple[
        MovieInputData,
        str | None,  # plot_summary
        str | None,  # emotional_observations
        NarrativeTechniquesOutput | None,
        PlotAnalysisOutput | None,
        str | None,  # craft_observations
    ],
]:
    """One SQLite query joins raw movie inputs with all upstream Wave 1/2 outputs.

    Returns a dict keyed by tmdb_id; each value is the tuple of arguments
    needed by generate_concept_tags:
        (MovieInputData, plot_summary, emotional_observations, nt_output,
         pa_output, craft_observations)

    Movies missing from the join are absent from the returned dict — callers
    must handle the missing case.
    """
    if not TRACKER_DB.exists():
        raise FileNotFoundError(
            f"Tracker DB not found at {TRACKER_DB}. "
            "Run from the project root and ensure Stage 4+ has populated tracker.db."
        )

    placeholders = ", ".join("?" * len(tmdb_ids))
    with sqlite3.connect(str(TRACKER_DB)) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            f"""
            SELECT
                t.tmdb_id,
                t.title,
                CAST(SUBSTR(t.release_date, 1, 4) AS INTEGER) AS release_year,
                t.collection_name,
                i.overview,
                i.maturity_rating,
                i.reception_summary,
                i.genres,
                i.synopses,
                i.plot_summaries,
                i.plot_keywords,
                i.overall_keywords,
                i.production_companies,
                i.actors,
                i.characters,
                i.directors,
                i.featured_reviews,
                i.review_themes,
                i.maturity_reasoning,
                i.parental_guide_items,
                g.plot_events,
                g.reception,
                g.narrative_techniques,
                g.plot_analysis
            FROM tmdb_data t
            JOIN imdb_data i ON t.tmdb_id = i.tmdb_id
            LEFT JOIN generated_metadata g ON t.tmdb_id = g.tmdb_id
            WHERE t.tmdb_id IN ({placeholders})
            """,
            tmdb_ids,
        ).fetchall()

    result: dict[int, tuple] = {}
    for row in rows:
        tmdb_id = row["tmdb_id"]

        movie = MovieInputData(
            tmdb_id=tmdb_id,
            title=row["title"] or "",
            release_year=row["release_year"],
            collection_name=row["collection_name"],
            overview=row["overview"] or "",
            genres=_parse_json_list(row["genres"]),
            plot_synopses=_parse_json_list(row["synopses"]),
            plot_summaries=_parse_json_list(row["plot_summaries"]),
            plot_keywords=_parse_json_list(row["plot_keywords"]),
            overall_keywords=_parse_json_list(row["overall_keywords"]),
            production_companies=_parse_json_list(row["production_companies"]),
            actors=_parse_json_list(row["actors"]),
            characters=_parse_json_list(row["characters"]),
            directors=_parse_json_list(row["directors"]),
            featured_reviews=_parse_json_list(row["featured_reviews"]),
            reception_summary=row["reception_summary"],
            audience_reception_attributes=_parse_json_list(row["review_themes"]),
            maturity_rating=row["maturity_rating"] or "",
            maturity_reasoning=_parse_json_list(row["maturity_reasoning"]),
            parental_guide_items=_parse_json_list(row["parental_guide_items"]),
        )

        # Wave 1 outputs — plot_summary + emotional_observations + craft_observations.
        # craft_observations is the reviewer-craft signal that drives
        # NONLINEAR_TIMELINE and disambiguates PLOT_TWIST / UNRELIABLE_NARRATOR /
        # BREAKING_FOURTH_WALL when plot_keywords are sparse.
        plot_summary: str | None = None
        emotional_observations: str | None = None
        craft_observations: str | None = None
        if row["plot_events"]:
            try:
                plot_summary = PlotEventsOutput.model_validate_json(
                    row["plot_events"]
                ).plot_summary
            except Exception:
                pass
        if row["reception"]:
            try:
                rec = ReceptionOutput.model_validate_json(row["reception"])
                emotional_observations = rec.emotional_observations
                craft_observations = rec.craft_observations
            except Exception:
                pass

        # Wave 2 dependencies. Routed through normalize_legacy_metadata_payload
        # so legacy NT rows (pre-"justification → evidence_basis" rename)
        # parse correctly. Same pattern as load_narrative_techniques_output.
        nt: NarrativeTechniquesOutput | None = None
        if row["narrative_techniques"]:
            try:
                nt_payload = json.loads(row["narrative_techniques"])
                nt_payload = normalize_legacy_metadata_payload(nt_payload, NarrativeTechniquesOutput)
                nt = NarrativeTechniquesOutput.model_validate(nt_payload)
            except Exception:
                pass

        pa: PlotAnalysisOutput | None = None
        if row["plot_analysis"]:
            try:
                pa_payload = json.loads(row["plot_analysis"])
                pa_payload = normalize_legacy_metadata_payload(pa_payload, PlotAnalysisOutput)
                pa = PlotAnalysisOutput.model_validate(pa_payload)
            except Exception:
                pass

        # ViewerExperienceOutput is no longer routed — ending_aftertaste
        # was the single contaminated upstream-label input that drove
        # the BITTERSWEET over-tag; ending classification now derives
        # from emotional_observations + plot_summary closing-scene.

        result[tmdb_id] = (
            movie, plot_summary, emotional_observations, nt, pa,
            craft_observations,
        )

    return result


# ---------------------------------------------------------------------------
# Per-movie orchestration
# ---------------------------------------------------------------------------

_SLUG_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def title_slug(title: str) -> str:
    """First 4 word-tokens of the title, lowercased and underscore-joined."""
    tokens = _SLUG_TOKEN_RE.findall(title.lower())
    return "_".join(tokens[:4]) or "untitled"


async def run_three_times(
    movie: MovieInputData,
    plot_summary: str | None,
    emotional_observations: str | None,
    nt: NarrativeTechniquesOutput | None,
    pa: PlotAnalysisOutput | None,
    craft_observations: str | None,
) -> list[ConceptTagsOutput]:
    """Fire NUM_RUNS_PER_MOVIE generation calls concurrently. Skip failures."""
    tasks = [
        generate_concept_tags(
            movie,
            plot_summary,
            emotional_observations,
            nt,
            pa,
            craft_observations=craft_observations,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        for _ in range(NUM_RUNS_PER_MOVIE)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    outputs: list[ConceptTagsOutput] = []
    for r in results:
        if isinstance(r, Exception):
            print(f"  [{movie.title_with_year()}] run failed: {r!r}")
            continue
        parsed, _token_usage = r
        outputs.append(parsed)
    return outputs


async def process_movie(
    entry: dict,
    movies_data: dict,
    results_dir: Path,
) -> None:
    """Run + merge + save for one movie. Logged best-effort; errors don't abort siblings."""
    tmdb_id = entry["tmdb_id"]
    title = entry["title"]
    year = entry["year"]

    movie_data = movies_data.get(tmdb_id)
    if movie_data is None:
        print(f"[{title} ({year})] SKIPPED — not found in tracker.db")
        return

    (
        movie, plot_summary, emotional_observations, nt, pa,
        craft_observations,
    ) = movie_data
    start = time.monotonic()
    outputs = await run_three_times(
        movie, plot_summary, emotional_observations, nt, pa,
        craft_observations,
    )
    elapsed = time.monotonic() - start

    if not outputs:
        print(f"[{title} ({year})] FAILED — all {NUM_RUNS_PER_MOVIE} runs raised; nothing saved")
        return

    merged = majority_merge(outputs)
    payload = {
        "title": title,
        "year": year,
        "tmdb_id": tmdb_id,
        "merged": json.loads(merged.model_dump_json()),
        "individual_runs": [json.loads(o.model_dump_json()) for o in outputs],
    }
    output_path = results_dir / f"{title_slug(title)}.json"
    output_path.write_text(json.dumps(payload, indent=2))
    print(
        f"[{title} ({year})] {len(outputs)}/{NUM_RUNS_PER_MOVIE} runs in {elapsed:.1f}s "
        f"→ {output_path}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main_async(test_name: str) -> None:
    results_dir = RESULTS_ROOT / test_name
    results_dir.mkdir(parents=True, exist_ok=True)

    tmdb_ids = [e["tmdb_id"] for e in CONCEPT_TAGS_TEST_MOVIES]
    print(f"Fetching {len(tmdb_ids)} movies + upstream outputs in a single query...")
    movies_data = fetch_all_movie_data(tmdb_ids)
    print(f"Loaded {len(movies_data)}/{len(tmdb_ids)} movies from tracker.db")

    missing = [e for e in CONCEPT_TAGS_TEST_MOVIES if e["tmdb_id"] not in movies_data]
    if missing:
        print(f"  Missing from tracker: {[(m['title'], m['tmdb_id']) for m in missing]}")

    print(
        f"Launching {len(CONCEPT_TAGS_TEST_MOVIES)} movies × {NUM_RUNS_PER_MOVIE} runs "
        f"= {len(CONCEPT_TAGS_TEST_MOVIES) * NUM_RUNS_PER_MOVIE} concurrent generations..."
    )
    overall_start = time.monotonic()
    await asyncio.gather(
        *(process_movie(entry, movies_data, results_dir) for entry in CONCEPT_TAGS_TEST_MOVIES)
    )
    print(f"All done in {time.monotonic() - overall_start:.1f}s → {results_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run concept_tags generation 3x per eval movie and save majority-merged results."
    )
    parser.add_argument(
        "--test-name",
        required=True,
        help="Subfolder name under concept_tags_results/ where this run's JSON files go.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args.test_name))


if __name__ == "__main__":
    main()
