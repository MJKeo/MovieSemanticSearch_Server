# Search V2 — shared vague-temporal vocabulary.
#
# Single source of truth for how vague time and duration terms map
# to concrete ranges. Two views are exported:
#
#   COMPACT — plain-English mappings, no parameter formats. Used by
#     Step 2 (so evaluative_intent prose doesn't silently shrink the
#     window) and Step 3 (so dimension/category routing reads era
#     terms as broad release-date signals, not narrow quality ones).
#     Later stages can always narrow; they cannot widen, so the
#     upstream view stays generous.
#
#   DETAILED_RELEASE_DATE / DETAILED_RUNTIME — concrete defaults in
#     the parameter formats the metadata-translation endpoint emits
#     (YYYY-MM-DD dates, minute counts). Used only by the metadata
#     query-generation prompt, which is the layer that actually
#     produces these values.
#
# Definitions live in ONE file so the compact and detailed views
# cannot drift apart. Update both together when ranges change.

# ---------------------------------------------------------------------------
# Compact view — for Steps 2 and 3.
#
# Plain English. No formats, no parameter values. The point is to
# keep upstream prose / routing analysis from collapsing a broad
# term ("modern") into a narrow one ("last few years") that the
# downstream translation layer cannot undo.
# ---------------------------------------------------------------------------

VAGUE_TEMPORAL_VOCABULARY_COMPACT = """\
VAGUE TIME / DURATION VOCABULARY

Movie queries often use vague temporal or duration terms. When you \
carry these into your prose or routing analysis, interpret them at \
the BROADER end of plausible — later stages can narrow, they \
cannot widen. Default mappings:

Release era:
- "new" / "just came out" / "brand new" — roughly the last year
- "recent" / "current" / "latest" — roughly the last 1-2 years
- "contemporary" — roughly the last 5-10 years
- "modern" — roughly 2000 to present (NOT "the last few years")
- "old-school" — roughly the 1970s through 1990s
- "old" — roughly pre-1990
- "really old" / "very old" — roughly pre-1970
- "classic" — roughly 1930s-1970s. PRIMARILY a release-era signal \
  (with a mild popularity / canonical-status undertone). Do not \
  treat "classic" as a pure quality or reception filter.
- "vintage" — roughly pre-1970
- "golden age" — roughly 1930s-1950s

Runtime length:
- "short" / "quick" — under ~95 minutes
- "standard" / "normal length" — roughly 90-120 minutes
- "long" — roughly 135-165 minutes
- "very long" — 165 minutes or more
- "epic-length" / "marathon" — 3 hours or more

When the user gives a concrete anchor (a specific year, decade, \
minute cutoff, "post-2010", "under 90 min"), prefer the concrete \
anchor over the vague default above.

---

"""


# ---------------------------------------------------------------------------
# Detailed view — for the metadata-translation endpoint.
#
# Same mappings, expressed in the parameter formats that endpoint
# actually emits. Inlined into the release_date and runtime
# sub-object translation rules at prompt build time.
# ---------------------------------------------------------------------------

VAGUE_RELEASE_DATE_DETAILED = """\
- Vague era terms — defaults when no concrete anchor is given:
  - "new", "just came out", "brand new" — between today minus ~1 year and today
  - "recent", "current", "latest" — between today minus ~2 years and today
  - "contemporary" — between today minus ~10 years and today
  - "modern" — between 2000-01-01 and today. This is broader than "contemporary" by design — do not collapse it to the last few years.
  - "old-school" — between 1970-01-01 and 1999-12-31
  - "old" — before 1990-01-01
  - "really old" / "very old" — before 1970-01-01
  - "classic" — between 1930-01-01 and 1979-12-31. Treat as primarily a release-era constraint with a mild popularity undertone — not as a pure quality / reception signal.
  - "vintage" — before 1970-01-01
  - "golden age" — between 1930-01-01 and 1959-12-31
- When the user gives a concrete anchor (year, decade, "after 2010"), use that anchor directly and ignore the defaults above.\
"""


VAGUE_RUNTIME_DETAILED = """\
- Vague duration terms — defaults when no concrete cutoff is given:
  - "short", "quick", "quick watch" — less_than 95
  - "standard", "normal length", "average length" — between 90 and 120
  - "long", "lengthy" — between 135 and 165 (use greater_than 135 when the phrasing implies "at least this long" with no upper bound)
  - "very long", "really long" — greater_than 165
  - "epic-length", "marathon" — greater_than 180
- "Epic" alone is ambiguous — it can mean runtime OR scope / genre. Only treat it as a runtime signal when co-occurring tokens ("hours", "length", "-length") make duration the clear intent; otherwise this attribute is not the right route.\
"""
