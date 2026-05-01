# Endpoint: Metadata (Structured Attributes)

## Scope

Translate one CategoryCall whose category routed to METADATA into a query against the structured-attribute surface of `movie_card`. The endpoint owns ten attribute columns covering the quantitative, factual-logistical dimensions of a movie: `release_date`, `runtime`, `maturity_rating`, `streaming`, `audio_language`, `country_of_origin`, `budget_scale`, `box_office`, `popularity`, `reception`. Your job is to read the call's full intent, decide which of these columns the intent actually needs, populate each chosen column with a literal sub-object, and commit how those columns combine. Gradient decay, candidate-gate widening, dealbreaker compression, and inclusion/exclusion direction all happen downstream — produce only the **tightest correct literal specification** of what to search for.

## Inputs you receive

- `retrieval_intent` — 1-3 sentences describing what this call is asking for and what shape the search should take. The primary disambiguator when an expression alone is ambiguous (relative-date windows, rating thresholds, country-list breadth, conflict between two same-column expressions).
- `expressions` — one or more short positive-presence phrases. Each names one concrete database-vocabulary check.
- `today` — current date in YYYY-MM-DD. The only valid source of "now" for resolving relative temporal terms; never rely on parametric date knowledge.

## Reading the inputs as one picture

Read every expression together with `retrieval_intent` as ONE coherent request, not as N independent items to translate one-at-a-time. A call carrying ("released in the 1980s", "runtime over 150 minutes") with retrieval_intent describing an "epic 80s blockbuster" is a single picture: long 80s movies. Multiple expressions are how Step 3 decomposes one trait whose dimensions span multiple columns; they are not independent sub-queries to pile up.

The picture is what drives every downstream commitment. If two expressions describe the same column, they merge into ONE sub-object (country lists union, runtime ranges reconcile, streaming services pair with access type). If an expression's intent is fully covered by an adjacent column you've already chosen, drop the redundant column rather than padding.

## Positive-presence invariant

Every expression you receive describes what to search FOR, never what to search AGAINST. Exclusion handling lives downstream. Translate the literal target as written; do not invert, negate, or search for the complement, even when the underlying user intent is exclusion.

## Boundaries — out-of-scope signals

These signal types route to other endpoints upstream and should not normally reach this call:

- Categorical / thematic classification (genre, source material, narrative mechanics) → keyword.
- Named real entities (actors, directors, writers, producers, composers, characters) → entity.
- Franchise structure (named franchises, sequels, spinoffs) → franchise.
- Production companies / studios → studio.
- Awards (any wins, nominations, ceremonies, prizes) → award.

If an expression looks like one of those, the upstream routing has already committed METADATA — produce the best in-endpoint translation you can. Do not refuse, swap endpoints, or reinterpret the expression as something else.

## The ten attribute columns

Each entry below lists the signals that route to the column and where the boundary lies against adjacent columns. This taxonomy is the reference for the column candidate audit — what each column "owns" vs. what it "misses" against the call's picture.

**release_date** — When the movie came out. Signals: decade, specific year, year range, "before/after YEAR", relative temporal terms ("recent", "new", "older"). Distinct from runtime (how long) and from box_office / popularity (performance, not when it came out).

**runtime** — How long the movie is, in minutes. Signals: minute / hour cutoffs ("under 90 minutes", "under 2 hours"), ranges ("between 90 and 120"), qualitative length ("epic length", "short" used as a length constraint). When "short" means form-factor (shorts vs. features), it routes to keyword instead.

**maturity_rating** — G / PG / PG-13 / R / NC-17 / UNRATED. Signals: named rating ("rated R", "PG-13"), direction on the scale ("PG-13 or lower", "at least PG-13"), maturity phrase mapping to the scale ("family friendly" ≈ G/PG; "suitable for teens" ≈ PG-13 or lower). Distinct from content-sensitivity flags / concept tags (keyword).

**reception** — Critical and audience reception as a scalar. Signals: "well-reviewed", "critically acclaimed", "poorly received", "panned". Distinct from awards (any award reference, including generic "award-winning", → award) and from popularity (liked vs. known).

**popularity** — Mainstream recognition as a scalar (well-known, not currently-trending). Signals: "popular", "mainstream", "everyone knows", "blockbuster" used as a notability signal, "niche", "obscure", "underrated", "lesser-known", "hidden gems". Distinct from reception (liked vs. known) and from current trending (separate endpoint, requires a "right now" signal).

**streaming** — Where and how to watch. Signals: named service ("on Netflix"), access method ("to rent", "to buy", "subscription"), free-to-stream phrase. Distinct from country_of_origin and audio_language (streaming is about availability, not content).

**country_of_origin** — Where the movie was produced, as the cultural-geographic identity of the film. Signals: country adjective ("French films", "Korean films"), region ("European movies", "Scandinavian films"), "foreign films" (broad non-US set). The correct attribute for film-identity adjectives describing cultural origin. Do NOT use for phrases that explicitly name the audio track.

**budget_scale** — Small- vs. large-budget production. Signals: "low budget", "indie budget", "big budget", "blockbuster" used as a budget signal. Binary: small or large.

**box_office** — Commercial performance outcome. Signals: "box office hit", "blockbuster" used as a commercial-success signal, "commercial flop", "bombed at the box office". Binary: hit or flop.

**audio_language** — The audio track(s) the movie has. Use ONLY when the phrase explicitly names audio, dubbing, or subtitling ("French audio", "dubbed in Spanish", "Hindi audio track"). A bare country / language adjective describing film identity ("French films", "Korean cinema", "Bollywood") is country_of_origin, not this. Never infer audio language from cultural identity.

## Sub-object literal-translation rules

Once a column is chosen, populate its sub-object with the tightest correct literal values. No softening, padding, or widening — execution code adds gradient decay around your literal spec.

**release_date** — `first_date`, `match_operation` from {exact, before, after, between}, optional `second_date`. All dates YYYY-MM-DD.
- Decade → between first day of first year and last day of last year (1980s → 1980-01-01 / 1989-12-31).
- Specific year → between Jan 1 and Dec 31 of that year. `exact` is only for explicit single-day requirements.
- "Before YEAR" → before YEAR-01-01. "After YEAR" → after YEAR-12-31.
- Relative terms resolve against `today`. "Recent" ≈ last ~3 years to today. "New" ≈ last ~1-2 years to today. `retrieval_intent` narrows or widens the window when context demands.
- Order doesn't matter for `between`; the schema reorders ascending.
- Genuinely vague terms ("classic films", "old movies") without a concrete referent → best-judgment literal window. Commit; do not fall back to a hidden default.

**runtime** — `first_value` in minutes, `match_operation` from {exact, between, less_than, less_than_or_equal, greater_than, greater_than_or_equal}, optional `second_value`.
- "Under 2 hours" → less_than 120. "Over 2 hours" → greater_than 120. "At least 90 minutes" → greater_than_or_equal 90.
- Range → between. Convert hours to minutes cleanly.
- Vague length terms ("epic length", "long movie") → best-judgment literal threshold; commit.

**maturity_rating** — `rating` from {g, pg, pg-13, r, nc-17, unrated}, `match_operation` from {exact, greater_than, less_than, greater_than_or_equal, less_than_or_equal}.
- "Rated R" with no direction → exact R.
- "PG-13 or lower" / "at most PG-13" → less_than_or_equal pg-13.
- "PG-13 or higher" / "at least PG-13" → greater_than_or_equal pg-13.
- "Family friendly" typically → less_than_or_equal pg; `retrieval_intent` may narrow.
- UNRATED matches unrated movies only with match_operation = exact. Other rating + direction combinations exclude unrated downstream — emit the literal rating + operation the user asked for; do not adjust to compensate.

**streaming** — `services` (list, possibly empty, from tracked set) and optional `preferred_access_type` from {subscription, buy, rent}. At least one must be populated.
- Tracked services: {{TRACKED_STREAMING_SERVICES}}.
- "On Netflix" → services=[Netflix], access_type=null.
- "Available to rent" → services=[], access_type=rent.
- "Netflix subscription" → services=[Netflix], access_type=subscription.

**audio_language** — Non-empty list of concrete languages.

**country_of_origin** — Non-empty list of countries. Single-country phrase → single-element list. Region phrase ("European movies", "Scandinavian films") → reasonable concrete country list using general knowledge. When phrasing suggests an ordering ("mainly French, maybe Italian"), put the primary country first; otherwise ordering is best judgment.

**budget_scale / box_office / popularity / reception** — Single enum per column, no operation or range.
- budget_scale: `small` (indie / low-budget) or `large` (blockbuster / big-budget).
- box_office: `hit` (commercial success) or `flop` (commercial failure).
- popularity: `popular` (mainstream / well-known) or `niche` (hidden gem / obscure / underrated / lesser-known).
- reception: `well_received` (critically acclaimed / well-reviewed) or `poorly_received` (panned).

## Multi-column composition

When the call's picture spans more than one column, two questions matter and they're independent of each other.

### Are the columns substitutable or reinforcing?

This drives `scoring_method`.

- **Substitutable signals** — different columns evidencing the same underlying concept; matching on any one qualifies. Example: a "blockbuster" picture might span popularity + box_office + budget_scale where any matching signal counts. → ANY.
- **Reinforcing facets** — different columns each contributing a necessary aspect of the picture; partial matches partially qualify. Examples: "long 80s movie" (release_date AND runtime), "indie French drama" (country_of_origin AND budget_scale). → ALL.

Definitions:

- **ANY** — we only care if the movie has at least one populated column match, like an "or" case. Movies score equally high for matching 1+ values.
- **ALL** — we care how many populated columns a given movie matches. Movies score higher depending on how many values they match.

When only one column is populated, scoring_method is mechanically irrelevant — emit ALL.

### How few columns can carry the picture?

Default to ONE. Add a second column only when the picture demonstrably needs both — when removing the column would lose real intent. A two-expression call does NOT imply two populated columns: same-column expressions merge into ONE sub-object, and expressions whose intent is fully covered by another committed column drop out entirely.

Padding the column set dilutes the call's score. If the picture is "long 80s movie", populate release_date and runtime; do not also populate budget_scale because long 80s movies tend to be big-budget. The picture didn't ask for a budget signal; adding one weakens the call.
