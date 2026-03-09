"""
IMDB scraped data quality analysis.

Loads all JSON files from ingestion_data/imdb/ and reports per-field
coverage (present vs. missing) and distribution percentiles for the
metric most appropriate to each field type.

Usage:
    python scripts/analyze_imdb_quality.py
"""

import json
import statistics
from pathlib import Path


# ---------------------------------------------------------------------------
# Data directory
# ---------------------------------------------------------------------------

_IMDB_DIR = Path(__file__).resolve().parent.parent.parent / "ingestion_data" / "imdb"


# ---------------------------------------------------------------------------
# Field definitions — grouped by source page
#
# Each entry: (field_name, field_kind, metric_label)
#
# field_kind controls both the "missing" check and the percentile metric:
#   "opt_str"   → missing if None or ""; metric = len(value)
#   "opt_float" → missing if None; metric = raw value
#   "opt_int"   → missing if None; metric = raw value
#   "int_zero"  → missing if 0; metric = raw value
#   "list_str"  → missing if []; metric = len(list)
#   "list_obj"  → missing if []; metric = len(list)
# ---------------------------------------------------------------------------

_PAGE_GROUPS: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("Main Page", [
        ("original_title",       "opt_str",   "char len"),
        ("maturity_rating",      "opt_str",   "char len"),
        ("overview",             "opt_str",   "char len"),
        ("overall_keywords",     "list_str",  "count"),
        ("imdb_rating",          "opt_float", "value"),
        ("imdb_vote_count",      "int_zero",  "value"),
        ("metacritic_rating",    "opt_float", "value"),
        ("reception_summary",    "opt_str",   "char len"),
        ("genres",               "list_str",  "count"),
        ("countries_of_origin",  "list_str",  "count"),
        ("production_companies", "list_str",  "count"),
        ("filming_locations",    "list_str",  "count"),
        ("languages",            "list_str",  "count"),
        ("budget",               "opt_int",   "value"),
        ("review_themes",        "list_obj",  "count"),
    ]),
    ("Summary Page", [
        ("synopses",       "list_str", "count"),
        ("plot_summaries", "list_str", "count"),
    ]),
    ("Keywords Page", [
        ("plot_keywords", "list_str", "count"),
    ]),
    ("Parental Guide Page", [
        ("maturity_reasoning",    "list_str", "count"),
        ("parental_guide_items",  "list_obj", "count"),
    ]),
    ("Credits Page", [
        ("directors",  "list_str", "count"),
        ("writers",    "list_str", "count"),
        ("actors",     "list_str", "count"),
        ("characters", "list_str", "count"),
        ("producers",  "list_str", "count"),
        ("composers",  "list_str", "count"),
    ]),
    ("Reviews Page", [
        ("featured_reviews", "list_obj", "count"),
    ]),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_missing(value, kind: str) -> bool:
    """Return True if the value should be considered missing/default."""
    if kind == "opt_str":
        return value is None or value == ""
    if kind in ("opt_float", "opt_int"):
        return value is None
    if kind == "int_zero":
        return value == 0
    if kind in ("list_str", "list_obj"):
        return not value  # empty list or None
    return value is None


def _extract_metric(value, kind: str) -> float | int:
    """Extract the numeric metric from a present (non-missing) value."""
    if kind == "opt_str":
        return len(value)
    if kind in ("opt_float", "opt_int", "int_zero"):
        return value
    if kind in ("list_str", "list_obj"):
        return len(value)
    return 0


def _percentiles(values: list[float | int]) -> tuple[str, str, str, str]:
    """Compute p25, p50, p75, p90 and return as formatted strings."""
    if not values:
        return ("-", "-", "-", "-")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    # Use linear interpolation for percentile computation
    def pct(p: float) -> float:
        k = (n - 1) * p
        f = int(k)
        c = f + 1 if f + 1 < n else f
        return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])
    results = [pct(0.25), pct(0.50), pct(0.75), pct(0.90)]
    return tuple(_format_num(v) for v in results)


def _format_num(v: float) -> str:
    """Format a number nicely — integers stay clean, floats get 1 decimal."""
    if isinstance(v, int) or v == int(v):
        return f"{int(v):,}"
    return f"{v:,.1f}"


# ---------------------------------------------------------------------------
# Nested object detail analysis
# ---------------------------------------------------------------------------

# For list[obj] fields, compute avg text length of the content within items
_NESTED_TEXT_EXTRACTORS: dict[str, list[tuple[str, callable]]] = {
    "review_themes": [
        ("name length", lambda item: len(item.get("name", ""))),
    ],
    "parental_guide_items": [
        ("category length", lambda item: len(item.get("category", ""))),
    ],
    "featured_reviews": [
        ("summary length", lambda item: len(item.get("summary", ""))),
        ("text length", lambda item: len(item.get("text", ""))),
    ],
}


def _analyze_nested_field(field_name: str, all_values: list[list]) -> list[str]:
    """
    For nested object lists, return extra stat lines about inner text content.

    Only called for fields that have entries in _NESTED_TEXT_EXTRACTORS.
    Returns formatted lines to print below the main field row.
    """
    extractors = _NESTED_TEXT_EXTRACTORS.get(field_name)
    if not extractors:
        return []

    lines = []
    for label, extractor in extractors:
        # Flatten all items across all movies into a single list of metrics
        metrics = []
        for item_list in all_values:
            for item in item_list:
                metrics.append(extractor(item))

        if not metrics:
            continue

        avg = statistics.mean(metrics)
        p25, p50, p75, p90 = _percentiles(metrics)
        lines.append(
            f"  {'↳ ' + label:<24s} "
            f"{'n=' + str(len(metrics)):>12s}  "
            f"{'avg=' + _format_num(avg):>12s}  "
            f"{p25:>8s}  {p50:>8s}  {p75:>8s}  {p90:>8s}"
        )
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Load all JSON files
    json_files = sorted(_IMDB_DIR.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {_IMDB_DIR}")
        return

    movies: list[dict] = []
    for f in json_files:
        movies.append(json.loads(f.read_text(encoding="utf-8")))

    total = len(movies)
    print(f"\nIMDB Data Quality Report ({total:,} movies)")
    print("=" * 100)

    for page_name, fields in _PAGE_GROUPS:
        print(f"\n--- {page_name} ---")
        print(
            f"  {'Field':<24s} "
            f"{'Present':>12s}  {'Missing':>12s}  "
            f"{'p25':>8s}  {'p50':>8s}  {'p75':>8s}  {'p90':>8s}  "
            f"{'Metric'}"
        )
        print(f"  {'-' * 94}")

        for field_name, kind, metric_label in fields:
            present_values = []
            missing_count = 0

            # Also collect raw list values for nested analysis
            nested_lists: list[list] = []

            for movie in movies:
                value = movie.get(field_name)
                if _is_missing(value, kind):
                    missing_count += 1
                else:
                    present_values.append(_extract_metric(value, kind))
                    if kind == "list_obj":
                        nested_lists.append(value)

            present_count = total - missing_count
            pct_present = present_count / total * 100
            pct_missing = missing_count / total * 100

            p25, p50, p75, p90 = _percentiles(present_values)

            present_str = f"{present_count} ({pct_present:.0f}%)"
            missing_str = f"{missing_count} ({pct_missing:.0f}%)"

            print(
                f"  {field_name:<24s} "
                f"{present_str:>12s}  {missing_str:>12s}  "
                f"{p25:>8s}  {p50:>8s}  {p75:>8s}  {p90:>8s}  "
                f"[{metric_label}]"
            )

            # Print nested detail lines for list[obj] fields
            if kind == "list_obj" and nested_lists:
                for line in _analyze_nested_field(field_name, nested_lists):
                    print(line)

    # Summary: fields with highest missing rates
    print(f"\n{'=' * 100}")
    print("Fields with highest missing rates:")
    print(f"  {'Field':<24s}  {'Missing':>12s}")
    print(f"  {'-' * 38}")

    all_fields = []
    for _, fields in _PAGE_GROUPS:
        for field_name, kind, _ in fields:
            missing = sum(1 for m in movies if _is_missing(m.get(field_name), kind))
            pct = missing / total * 100
            all_fields.append((field_name, missing, pct))

    # Sort by missing rate descending, show top 10
    all_fields.sort(key=lambda x: x[1], reverse=True)
    for field_name, missing, pct in all_fields[:10]:
        print(f"  {field_name:<24s}  {missing} ({pct:.0f}%)")

    # Text length distribution for long-text fields (overview, synopses,
    # reception_summary, featured_reviews.text)
    print(f"\n{'=' * 100}")
    print("Long text field character length distributions:")
    long_text_fields = [
        "overview", "reception_summary", "synopses", "plot_summaries",
    ]
    for field_name in long_text_fields:
        lengths = []
        for movie in movies:
            val = movie.get(field_name)
            if val is None:
                continue
            if isinstance(val, str) and val:
                lengths.append(len(val))
            elif isinstance(val, list):
                # For list[str], compute length of each individual entry
                for entry in val:
                    if isinstance(entry, str) and entry:
                        lengths.append(len(entry))
        if not lengths:
            continue
        p25, p50, p75, p90 = _percentiles(lengths)
        avg = _format_num(statistics.mean(lengths))
        print(
            f"  {field_name:<24s}  "
            f"n={len(lengths):<6d}  avg={avg:>8s}  "
            f"p25={p25:>8s}  p50={p50:>8s}  "
            f"p75={p75:>8s}  p90={p90:>8s}  chars"
        )

    print()


if __name__ == "__main__":
    main()
