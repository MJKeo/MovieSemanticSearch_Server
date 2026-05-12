# Endpoint: Chronological (Recency Percentile Curve)

## What this endpoint does

Translates a CategoryCall about a release-date direction preference
(older movies should win vs. newer movies should win) into a
`ChronologicalQuerySpec` carrying one field, `direction`.

At execution time the endpoint is a POOL_RERANKER. It receives the
candidate pool the rest of the query's categories produced, looks up
each candidate's `release_ts`, and emits a per-movie percentile-rank
score in [0, 1]:

- `direction = oldest_first` — the **oldest** movie in the candidate
  pool scores 1.0; the newest scores 0.0.
- `direction = newest_first` — the **newest** movie in the candidate
  pool scores 1.0; the oldest scores 0.0.

Every distinct release date in the pool occupies its own slot in the
[0, 1] interval — a one-day difference always matters. Movies
sharing a release date receive identical scores. Movies whose
`release_ts` is missing score 0.0.

## When to fire

**Fire whenever the call involves a date-direction preference of any
kind — pick whichever direction best captures what the user wants.**
The upstream router only sends date-direction phrasings here, so by
the time a call reaches this endpoint, the answer is almost always
"yes, fire, pick a direction." Abstention is reserved for the
degenerate case where the call has *nothing* to do with dates.

The decision is which direction, not whether to fire:

- **The user wants OLD / EARLY movies to win → `oldest_first`.**
  Cues: "oldest", "earliest", "first" (as in earliest), "original",
  "the one that started it", "earliest entry", "began with", "the
  originals", "chronological order" used as a preference for the
  earliest end.
- **The user wants NEW / LATE movies to win → `newest_first`.**
  Cues: "newest", "latest", "most recent", "the new one", "current",
  "most up-to-date", "the latest entry", "recent", "how recent",
  "fresh", "modern".

Quick mapping table:

| Phrase contains... | → direction |
|---|---|
| "oldest", "earliest", "first" (as in earliest), "original", "started it", "began with" | `oldest_first` |
| "newest", "latest", "most recent", "recent", "the new one", "current", "modern", "how recent" | `newest_first` |

### Conflict resolution

When the surface expression and the retrieval_intent point in
different directions, lean on whichever signal is *more specific
about a date preference*:

- Intent says "how recent they were produced" + expression says
  "first" → intent's recency framing is the dominant signal →
  `newest_first`. ("First" here is ambiguous shorthand for "the top
  result of the recency sort.")
- Intent says "sort by date" + expression says "oldest" → expression
  pins the extreme → `oldest_first`.
- Intent and expression both ambiguous but the call clearly
  concerns date direction → pick whichever direction the bulk of
  the wording leans toward. Don't abstain just because the inputs
  aren't perfectly aligned.

## When to abstain

Abstain only when the call has no date-direction signal at all —
i.e., the call shouldn't have been routed here in the first place
(upstream misroute). Examples:

- The call is purely about a date *window* with no extreme preference
  ("movies between 1995 and 2005" with no other date language) →
  abstain; RELEASE_DATE (Cat 13) owns the window.
- The call is about quality/reception ("best ever", "most acclaimed")
  → abstain; general_appeal / cultural_status own those.
- The call has nothing to do with dates ("movies with explosions")
  → abstain.

When in doubt between fire-and-pick vs. abstain, prefer fire. The
percentile curve is graceful — a slightly mismatched direction still
produces useful rerank signal; an abstention loses the signal
entirely.

## What does NOT belong here

- **Sibling scope** (director / franchise / title / studio that
  defines the candidate pool) — those belong to their own sibling
  CategoryCalls. This endpoint never re-emits the pool boundary; it
  only ranks within the pool it is handed.

## Composition with siblings

CHRONOLOGICAL composes naturally with carving categories: "the
newest Scorsese movie" runs PERSON_CREDIT (Scorsese) to define the
pool, then CHRONOLOGICAL (`newest_first`) reranks within it. The
merged final score elevates the most-recent Scorsese to the top of
the response while keeping the rest of his filmography present in
descending recency order. No separate top-N cutoff is applied — the
score curve itself is the answer.

CHRONOLOGICAL also composes with other POOL_RERANKERs
(general_appeal, cultural_status). When both fire, both contribute
to the final rerank; an "acclaimed AND newest" framing weights each
signal at the merge layer rather than collapsing into one channel.
