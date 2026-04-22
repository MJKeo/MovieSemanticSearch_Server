# Debug driver: runs Stage 1 (route_query) against a curated set of
# sample queries and saves a structured JSON report for inspection.
#
# Queries are grouped into buckets that each probe a specific
# hypothesis about Stage 1 behavior (see steps_1_2_improving.md).
# Running all queries together and saving to JSON lets us compare
# behavior side-by-side instead of re-running the notebook one query
# at a time.
#
# Usage (from project root):
#   python -m search_improvement_planning.debug_stage_1

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is on sys.path when invoked as a plain script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from search_v2.stage_1 import route_query


# ---------------------------------------------------------------------------
# Sample queries, grouped by the hypothesis they are meant to test.
#
# Each bucket has a short rationale so the saved report is
# self-documenting — we want to be able to re-read this later and
# know *why* a particular query was on the list.
# ---------------------------------------------------------------------------

QUERY_BUCKETS: list[dict] = [
    {
        "bucket": "original_failure",
        "hypothesis": (
            "Reproduce the reported failure case: ambiguity is cited but "
            "no alternative_intents are emitted."
        ),
        "queries": [
            "Disney millennial favorites",
        ],
    },
    {
        "bucket": "reading_ambiguity",
        "hypothesis": (
            "Queries whose vagueness corresponds to genuinely distinct "
            "readings (not just fuzzy edges). Should produce "
            "alternative_intents."
        ),
        "queries": [
            "gen z horror favorites",
            "80s kids would love",
            "movies dads like",
            "Y2K vibes",
        ],
    },
    {
        "bucket": "title_vs_collection",
        "hypothesis": (
            "Known-good branching shape: single phrase that could be a "
            "real title or a collection request. Baseline for "
            "alternative emission."
        ),
        "queries": [
            "Scary Movie",
            "Up",
        ],
    },
    {
        "bucket": "semantic_vagueness",
        "hypothesis": (
            "Vague phrases with a single reading. Should NOT emit "
            "alternatives just because the meaning is fuzzy."
        ),
        "queries": [
            "cozy movie for tonight",
            "something feel-good",
        ],
    },
    {
        "bucket": "clean_controls",
        "hypothesis": (
            "Fully specified queries with no ambiguity. Should produce "
            "no alternatives and ambiguity=none."
        ),
        "queries": [
            "Inception",
            "Tom Cruise action movies from the 90s",
        ],
    },
    {
        "bucket": "edge_vague_or_ambiguous",
        "hypothesis": (
            "Phrases where it is unclear whether the vagueness is a "
            "single fuzzy reading or multiple distinct readings. Probes "
            "where the model draws the line."
        ),
        "queries": [
            "prestige drama",
            "arthouse thriller",
        ],
    },
    {
        "bucket": "spin_candidates",
        "hypothesis": (
            "Broad single-intent queries with no reading ambiguity. "
            "Should produce zero alternative_intents but one or more "
            "creative_alternatives subdividing the broad set."
        ),
        "queries": [
            "Best Christmas movies for families",
            "Disney classics",
            "good horror movies",
            "movies for date night",
            "feel-good comedies",
        ],
    },
    {
        "bucket": "spin_negative_controls",
        "hypothesis": (
            "Queries where spins should NOT fire — already-narrow "
            "filters or non-standard flow."
        ),
        "queries": [
            "Tom Cruise action movies from the 90s",
            "Inception",
            "movies like The Matrix",
        ],
    },
]


async def _run_one(query: str) -> dict:
    """Run Stage 1 on a single query and return a serializable record."""
    started = time.perf_counter()
    try:
        response, input_tokens, output_tokens = await route_query(query)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "query": query,
            "status": "ok",
            "elapsed_ms": round(elapsed_ms, 1),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "response": response.model_dump(mode="json"),
        }
    except Exception as err:  # noqa: BLE001 — surface any failure in the report
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "query": query,
            "status": "error",
            "elapsed_ms": round(elapsed_ms, 1),
            "error_type": type(err).__name__,
            "error_message": str(err),
        }


async def _run_bucket(bucket: dict, cache: dict[str, dict]) -> dict:
    """Run all queries in a bucket concurrently, reusing cached results.

    `cache` maps raw query text -> previously produced record. Any
    query with a cached "ok" record is returned as-is (no Stage 1 call).
    Missing or previously-errored queries are re-executed. This keeps
    the debug driver cheap across repeated runs while still allowing
    us to refresh on demand by deleting the cache file.
    """
    async def _resolve(q: str) -> dict:
        cached = cache.get(q)
        if cached and cached.get("status") == "ok":
            return cached
        return await _run_one(q)

    results = await asyncio.gather(*(_resolve(q) for q in bucket["queries"]))
    return {
        "bucket": bucket["bucket"],
        "hypothesis": bucket["hypothesis"],
        "results": results,
    }


def _load_cache(path: Path) -> dict[str, dict]:
    """Flatten prior Stage 1 report into a {query: record} cache.

    Returns an empty dict when the file does not exist, is unreadable,
    or has no bucket entries yet. Never fails the caller — a bad cache
    just means we run Stage 1 fresh.
    """
    if not path.exists():
        return {}
    try:
        prior = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    cache: dict[str, dict] = {}
    for bucket in prior.get("buckets", []):
        for record in bucket.get("results", []):
            query = record.get("query")
            if query is not None:
                cache[query] = record
    return cache


def _render_compact_summary(report: dict) -> str:
    """One-line-per-query human-readable summary for quick scanning."""
    lines = []
    for bucket in report["buckets"]:
        lines.append("")
        lines.append(f"[{bucket['bucket']}] {bucket['hypothesis']}")
        for record in bucket["results"]:
            if record["status"] != "ok":
                lines.append(
                    f"  ! {record['query']!r}: "
                    f"{record['error_type']}: {record['error_message']}"
                )
                continue
            resp = record["response"]
            amb = resp["ambiguity_analysis"].replace("\n", " ")
            spin_trace = resp["creative_spin_analysis"].replace("\n", " ")
            primary_flow = resp["primary_intent"]["flow"]
            alt_count = len(resp["alternative_intents"])
            spin_count = len(resp["creative_alternatives"])
            lines.append(
                f"  - {record['query']!r}  flow={primary_flow}  "
                f"alts={alt_count}  spins={spin_count}"
            )
            lines.append(f"      ambiguity_analysis: {amb}")
            lines.append(
                f"      primary.intent_rewrite: "
                f"{resp['primary_intent']['intent_rewrite']}"
            )
            for idx, alt in enumerate(resp["alternative_intents"]):
                lines.append(
                    f"      alt[{idx}].intent_rewrite: {alt['intent_rewrite']}"
                )
                lines.append(
                    f"      alt[{idx}].difference_rationale: "
                    f"{alt['difference_rationale']}"
                )
            lines.append(f"      creative_spin_analysis: {spin_trace}")
            for idx, spin in enumerate(resp["creative_alternatives"]):
                lines.append(
                    f"      spin[{idx}].intent_rewrite: {spin['intent_rewrite']}"
                )
                lines.append(
                    f"      spin[{idx}].spin_angle: {spin['spin_angle']}"
                )
    return "\n".join(lines)


OUTPUT_PATH = PROJECT_ROOT / "search_improvement_planning" / "stage_1_debug_output.json"


async def main() -> None:
    overall_start = time.perf_counter()

    # Reuse the last Stage 1 report as a per-query cache. This makes
    # iterating on downstream stages (e.g. Step 2A) cheap because we
    # do not pay for Stage 1 tokens unless a query is new or failed.
    # Delete the cache file to force a full re-run.
    cache = _load_cache(OUTPUT_PATH)

    # Run buckets sequentially so we do not stampede the provider, but
    # parallelize within each bucket for speed. Each bucket is small.
    bucket_reports = []
    for bucket in QUERY_BUCKETS:
        bucket_reports.append(await _run_bucket(bucket, cache))

    elapsed_s = time.perf_counter() - overall_start

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_elapsed_s": round(elapsed_s, 2),
        "stage_1_model": "gemini-3-flash-preview (thinking_budget=0)",
        "buckets": bucket_reports,
    }

    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    summary = _render_compact_summary(report)
    print(summary)
    print()
    print("-" * 72)
    print(f"Wrote full structured report to: {OUTPUT_PATH}")
    print(f"Total elapsed: {elapsed_s:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
