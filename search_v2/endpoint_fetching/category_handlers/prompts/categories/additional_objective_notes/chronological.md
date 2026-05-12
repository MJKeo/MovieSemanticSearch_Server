# Additional objective notes - Chronological ordinal

## Target

Fire whenever the call involves a date-direction preference — older
movies should win OR newer movies should win. The endpoint scores
the entire candidate pool on a continuous recency percentile curve;
every distinct release date gets its own slot in [0, 1], so even a
one-day difference matters.

By the time a call reaches this endpoint, upstream routing has
already decided the call concerns date direction. **The job here is
to pick the better direction, not to second-guess the routing.** Fire
broadly; abstain only when the call truly has no date signal.

## Output schema

Emit a `ChronologicalQuerySpec` with a single field:

- `direction`: `oldest_first` — the OLDEST movie in the pool scores
  1.0. Pick this for phrasings that prefer old / early movies
  ("oldest", "earliest", "first" as in earliest, "original",
  "started it", "the originals").
- `direction`: `newest_first` — the NEWEST movie in the pool scores
  1.0. Pick this for phrasings that prefer new / late movies
  ("newest", "latest", "most recent", "recent", "the new one",
  "current", "modern", "how recent").

The enum value names the WINNING END of the curve directly — match
the surface phrasing to which extreme the user wants on top. Don't
reason about "chronological order" or "sort direction" — both
readings exist in English and lead to inversion bugs.

## Conflict resolution

When the expression and the retrieval_intent disagree, lean on the
signal that is more specific about a date preference:

- "first" + intent "how recent" → intent dominates → `newest_first`.
- "oldest" + intent "sort by date" → expression dominates →
  `oldest_first`.
- Either signal alone clearly pinning a direction → use it.

Don't abstain because of conflict. Pick the direction that best
captures the bulk of the wording.

## Decision Questions

1. Does the call involve any date-direction preference (older vs.
   newer)? If yes → fire. If no → abstain (rare; almost always
   upstream misroute).
2. Which extreme does the wording prefer on top: the OLDEST movie
   (`oldest_first`) or the NEWEST movie (`newest_first`)?

## When to abstain

Only when the call has no date-direction signal at all — typically
an upstream misroute. Examples: a pure date window with no extreme
("movies from 1995 to 2005"), a quality superlative ("best ever"),
a non-date trait that landed here by accident.

When unsure between fire-with-best-guess and abstain, prefer fire —
the percentile curve degrades gracefully and a slightly mismatched
direction still produces useful signal; an abstention loses the
signal entirely.

## Boundaries

- This endpoint is a continuous curve, not a window. Cat 13
  (RELEASE_DATE) owns explicit date *ranges* with bounds; this
  endpoint scores by date *direction*. Both can fire together when
  the call carries both signals — they're complementary.
- "First Bond" selects by chronology (`oldest_first`); "best Bond"
  selects by reception — abstain on pure reception/status phrasings.
- Do not emit sibling scope (director / franchise / title) here;
  siblings define the pool that this endpoint reranks.
