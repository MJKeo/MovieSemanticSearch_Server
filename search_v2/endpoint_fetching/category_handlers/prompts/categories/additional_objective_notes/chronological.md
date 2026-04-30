# Chronological — additional notes

This category is about **release-date ordinal position** — selecting the film that sits at a particular rank when a scoped candidate set is ordered by release date. Phrasings like "the newest", "the latest", "the most recent", "the earliest", "the oldest", "the first", "the last". The requirement picks a **position** in an ordering, not a window of dates.

## Ordinal vs. range — the discriminator

This is the only boundary that matters, and it decides every firing:

- **Ordinal (yours).** The phrase selects by position within an ordering: "the newest Scorsese", "the earliest Harry Potter", "the most recent Marvel film", "the last Tarantino". One film, or top-N films, chosen by rank.
- **Range (Cat 10's — Structured metadata).** The phrase names a window of dates: "90s movies", "recent films", "before 2000", "from the last few years". A gate on the `release_date` column, not a sort-and-pick.

Useful check: "recent" is a window (last few years — range). "Most recent" is a position (the single latest — ordinal). "Newer" is a window (after some inflection — range). "Newest" is a position. When the phrase would still make sense pluralized across years, it is probably a range. When it picks out one rank, it is ordinal.

## The endpoint cannot express ordinal selection

Read this carefully. The metadata endpoint described in the endpoint-context chunk populates `release_date` with a `first_date` / `second_date` / `match_operation` shape. The `match_operation` set is `{exact, before, after, between}` — **literal date comparators only**. No sort direction, no top-N, no position-from-one-end field. The endpoint **cannot represent an ordinal request as a predicate**.

That means a clean ordinal atom ("the newest Scorsese", "the earliest Kubrick") has no valid translation into this endpoint's parameter surface. The correct response is `should_run_endpoint: false` with `coverage_gaps` recording that the atom is an ordinal-position request and the endpoint only supports range / point date predicates. Do not approximate by translating "the newest" into `after <some date>` — that would return every film after that date rather than the single latest one, flattening the ordinal into a range and silently returning the wrong answer.

This is the tightest category-specific rule: **a well-formed ordinal atom no-fires on schema grounds**, and that no-fire is the correct, faithful response for the current endpoint shape.

## Composition with other categories

Chronological almost never arrives alone. "The newest Scorsese" decomposes into a Chronological atom (position = latest) plus a Credit + title text atom (Scorsese as director). "Earliest Harry Potter" decomposes into a Chronological atom plus a Franchise lineage atom. You only see the chronological slice. Do not try to re-emit the director, franchise, or subject half of the compound in your payload — a sibling atom is handling it.

## Boundaries with nearby categories

- **Structured metadata (Cat 10).** Owns `release_date` as a **range** ("90s", "pre-WWII", "before 2000", "recent films"). Cat 10 fires with a concrete `between` / `before` / `after` predicate. You own ordinal position only. If your atomic rewrite can be restated as a date window, dispatch was wrong — it belonged in Cat 10.
- **Reception quality + superlative (Cat 25).** Owns stature and quality superlatives — "the best Scorsese", "the most iconic Bond", "the greatest horror ever". The axis is acclaim, not chronology. "Latest" is chronological; "best" is reception stature; the two never collapse. No-fire on any "best" / "greatest" / "most iconic" phrasing routed here by mistake.
- **Trending (Cat 9).** Owns live right-now buzz — "popular this week", "trending now", "what everyone's watching". That is a live-refresh signal, not an ordinal position. "Most recent release" is a position on the release-date axis and belongs here; "currently popular" is a Trending signal.

## When to no-fire

- **The atom is a range / window, not a position.** Dispatch was wrong — it belonged in Cat 10. Record the misroute in `coverage_gaps`.
- **The atom is an acclaim or stature superlative.** Dispatch was wrong — it belonged in Cat 25. Record the misroute.
- **The atom is a well-formed ordinal request.** Ordinal selection sits outside the endpoint's parameter surface (literal date comparators only). Record that in `coverage_gaps` and no-fire.
- **Parent-fragment modifiers invert the atom in a way no literal date predicate resolves.** Record the contradiction and no-fire.

No-fire is the default outcome for this category under the current endpoint shape. Do not invent a range predicate to stand in for an ordinal request.
