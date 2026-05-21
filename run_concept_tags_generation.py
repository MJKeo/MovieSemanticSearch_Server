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
from collections import Counter
from pathlib import Path

from concept_tags_test_movies import CONCEPT_TAGS_TEST_MOVIES
from movie_ingestion.metadata_generation.generators.concept_tags import (
    generate_concept_tags,
)
from schemas.enums import EndingTag
from schemas.metadata import (
    CharacterAssessment,
    ConceptTagsOutput,
    ContentFlagAssessment,
    EndingAssessment,
    ExperientialAssessment,
    NarrativeStructureAssessment,
    NarrativeTechniquesOutput,
    PlotAnalysisOutput,
    PlotArchetypeAssessment,
    PlotEventsOutput,
    ReceptionOutput,
    SettingAssessment,
    ViewerExperienceOutput,
    normalize_legacy_metadata_payload,
)
from schemas.movie_input import MovieInputData


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRACKER_DB = Path("ingestion_data/tracker.db")
RESULTS_ROOT = Path("concept_tags_results")
NUM_RUNS_PER_MOVIE = 3

# (field_name, AssessmentCls) pairs for the six list-typed categories.
# Endings is handled separately (single tag, mode vote).
LIST_CATEGORIES: list[tuple[str, type]] = [
    ("narrative_structure", NarrativeStructureAssessment),
    ("plot_archetypes",     PlotArchetypeAssessment),
    ("settings",            SettingAssessment),
    ("characters",          CharacterAssessment),
    ("experiential",        ExperientialAssessment),
    ("content_flags",       ContentFlagAssessment),
]


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
        ViewerExperienceOutput | None,
    ],
]:
    """One SQLite query joins raw movie inputs with all upstream Wave 1/2 outputs.

    Returns a dict keyed by tmdb_id; each value is the tuple of arguments
    needed by generate_concept_tags:
        (MovieInputData, plot_summary, emotional_observations, nt_output,
         pa_output, craft_observations, ve_output)

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
                g.plot_analysis,
                g.viewer_experience
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

        # ViewerExperienceOutput — only ending_aftertaste is routed into the
        # prompt, but the whole object is parsed so future expansions can
        # route additional sections without changing the loader.
        ve: ViewerExperienceOutput | None = None
        if row["viewer_experience"]:
            try:
                ve_payload = json.loads(row["viewer_experience"])
                ve_payload = normalize_legacy_metadata_payload(ve_payload, ViewerExperienceOutput)
                ve = ViewerExperienceOutput.model_validate(ve_payload)
            except Exception:
                pass

        result[tmdb_id] = (
            movie, plot_summary, emotional_observations, nt, pa,
            craft_observations, ve,
        )

    return result


# ---------------------------------------------------------------------------
# Majority-vote merge
# ---------------------------------------------------------------------------

def majority_merge(outputs: list[ConceptTagsOutput]) -> ConceptTagsOutput:
    """Merge N ConceptTagsOutputs via majority rules.

    For each list-typed category, a tag joins the merged set iff at least
    ceil(N/2 + epsilon) runs include it — i.e. a strict majority. For N=3
    this is 2-of-3 (matches the user spec: include when 2 or 3 have it,
    exclude when 2 or 3 do not).

    For the single-value endings category, the merged tag is the mode of
    the N votes; on a tie, the first run's vote wins (deterministic).
    """
    if not outputs:
        raise ValueError("majority_merge requires at least one output")

    threshold = (len(outputs) // 2) + 1  # strict majority: 2 of 3, 2 of 2, 3 of 5

    merged_kwargs: dict = {}

    # List-typed categories
    for field_name, assessment_cls in LIST_CATEGORIES:
        counter: Counter = Counter()
        for out in outputs:
            for tag in getattr(out, field_name).tags:
                counter[tag] += 1
        majority_tags = [tag for tag, count in counter.items() if count >= threshold]
        # Sort by concept_tag_id for stable deterministic output
        majority_tags.sort(key=lambda t: t.concept_tag_id)
        merged_kwargs[field_name] = assessment_cls(tags=majority_tags)

    # Endings — mode vote, first-run tiebreaker
    ending_counter = Counter(out.endings.tag for out in outputs)
    max_count = max(ending_counter.values())
    top_candidates = [tag for tag, count in ending_counter.items() if count == max_count]
    if len(top_candidates) == 1:
        chosen_ending = top_candidates[0]
    else:
        first_run_ending = outputs[0].endings.tag
        chosen_ending = first_run_ending if first_run_ending in top_candidates else top_candidates[0]
    merged_kwargs["endings"] = EndingAssessment(tag=chosen_ending)

    return ConceptTagsOutput(**merged_kwargs)


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
    ve: ViewerExperienceOutput | None,
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
            ve_output=ve,
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
        craft_observations, ve,
    ) = movie_data
    start = time.monotonic()
    outputs = await run_three_times(
        movie, plot_summary, emotional_observations, nt, pa,
        craft_observations, ve,
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
