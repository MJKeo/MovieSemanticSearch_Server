"""Summarize / compare Step 3 batch result sets for the consolidation
experiment.

For each (query, trait) it aggregates the decomposition SHAPE across the
N repeat runs of each prefix and prints them side by side, so behavior
shifts and run-to-run variance are both visible. "Shape" = combine_mode +
the sorted set of committed category names (the load-bearing per-trait
measurement; individual keyword strings drift across runs by design).

It also scans every committed call for absence/avoidance language in
expressions + retrieval_intent — the inclusion-only contract says calls
must describe presence, so any hit is a candidate violation to eyeball.

Usage:
    python -m search_v2.category_candidates_experiment.summarize base fix_gemini fix_gpt
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from search_v2.category_candidates_experiment.queries import (
    QUERIES,
    slugify_first_four,
)

_RESULTS_DIR = Path(__file__).parent / "results"

# Absence / avoidance markers. The inclusion-only rule forbids framing a
# call around what should NOT be present; these surface likely violations
# for human review (some may be false positives, hence we print context).
_ABSENCE_RE = re.compile(
    r"\b(absence of|absent|free of|devoid of|lack(?:ing|s)? of|without|"
    r"avoid(?:s|ing|ance)?|exclud\w+|no (?:graphic|explicit|gore|violence|"
    r"nudity|swearing|profanity|sexual|mature)|free from|kept? out)\b",
    re.IGNORECASE,
)


def _load(prefix: str, query: str) -> dict | None:
    path = _RESULTS_DIR / f"{prefix}_{slugify_first_four(query)}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _shape(dec: dict) -> str:
    mode = dec.get("combine_mode", "?")
    cats = sorted(c["category"] for c in dec.get("category_calls", []))
    return f"{mode} [{', '.join(cats)}]"


def _trait_shapes(data: dict) -> dict[str, Counter]:
    """Map trait_surface_text -> Counter(shape) across all runs."""
    out: dict[str, Counter] = {}
    for run in data.get("runs", []):
        for tr in run.get("trait_results", []):
            surf = tr["trait_surface_text"]
            dec = tr.get("decomposition")
            shape = _shape(dec) if dec else f"ERROR:{tr.get('error','?')[:40]}"
            out.setdefault(surf, Counter())[shape] += 1
    return out


def _absence_hits(prefix: str) -> list[str]:
    hits: list[str] = []
    for q in QUERIES:
        data = _load(prefix, q)
        if not data:
            continue
        for run in data.get("runs", []):
            for tr in run.get("trait_results", []):
                dec = tr.get("decomposition")
                if not dec:
                    continue
                for call in dec.get("category_calls", []):
                    blob = " || ".join(
                        call.get("expressions", []) + [call.get("retrieval_intent", "")]
                    )
                    m = _ABSENCE_RE.search(blob)
                    if m:
                        hits.append(
                            f"[{prefix}] {q!r} / trait={tr['trait_surface_text']!r} "
                            f"/ {call['category']}: ...{_ctx(blob, m)}..."
                        )
    return hits


def _ctx(blob: str, m: re.Match) -> str:
    s = max(0, m.start() - 35)
    e = min(len(blob), m.end() + 35)
    return blob[s:e].replace("\n", " ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="+")
    args = ap.parse_args()

    # Per-query / per-trait shape distribution, prefixes side by side.
    for q in QUERIES:
        per_prefix = {p: _load(p, q) for p in args.prefixes}
        print("#" * 100)
        print("QUERY:", q)
        # union of trait surfaces (stable across prefixes since step_2 fixed)
        surfaces: list[str] = []
        for data in per_prefix.values():
            if data:
                for surf in _trait_shapes(data):
                    if surf not in surfaces:
                        surfaces.append(surf)
        for surf in surfaces:
            print(f"  TRAIT: {surf!r}")
            for p in args.prefixes:
                data = per_prefix[p]
                if not data:
                    print(f"      {p:12s}: <missing>")
                    continue
                shapes = _trait_shapes(data).get(surf, Counter())
                dist = "  ".join(
                    f"{cnt}x {sh}" for sh, cnt in shapes.most_common()
                )
                print(f"      {p:12s}: {dist}")

    # Inclusion-only scan.
    print("\n" + "=" * 100)
    print("ABSENCE / AVOIDANCE LANGUAGE IN COMMITTED CALLS (inclusion-only check)")
    for p in args.prefixes:
        hits = _absence_hits(p)
        print(f"\n--- {p}: {len(hits)} hit(s) ---")
        for h in hits[:40]:
            print("  " + h)


if __name__ == "__main__":
    main()
