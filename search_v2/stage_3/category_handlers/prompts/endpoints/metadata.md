# Endpoint: Metadata (Structured Attributes)

## Purpose

Single-column query on the structured-attribute surface of `movie_card`. Resolves a requirement to exactly one of ten attribute columns, then populates that column's sub-object with literal parameter values. Gradient decay, candidate-gate widening, and inclusion/exclusion direction are handled downstream — this endpoint only produces the **tightest correct literal specification** of the user's constraint.

## Canonical question

"Which single structured attribute does this requirement target, and what literal values / comparators pin it?"

## Capabilities

Structured, quantitative, factual-logistical dimensions of a movie. Literal date ranges, numeric comparators, enum selections, country lists, language lists, streaming-service + access-type combinations.

## Boundaries (what does NOT belong here)

- Categorical / thematic classification (genre, source material, cultural tradition, narrative mechanics) → keyword endpoint.
- Named real entities (actors, directors, writers, producers, composers, characters) → entity endpoint.
- Franchise structure (named franchises, sequels, spinoffs, subgroups) → franchise endpoint.
- Production companies / studios → studio endpoint.
- Awards (any recognition: wins, nominations, ceremonies, prizes) → award endpoint.

## Target attributes

Pick exactly one. The chosen attribute selects the single sub-object to populate.

**release_date** — When the movie came out. Signals: decade, specific year, year range, "before/after YEAR", relative temporal terms ("recent", "new", "older movies"). Distinct from runtime (how long) and box-office / popularity (performance, not release time).

**runtime** — How long the movie is. Signals: minute/hour cutoff ("under 90 minutes", "under 2 hours"), range ("between 90 and 120 minutes"), qualitative length ("epic length", "short" when it means length rather than form-factor). If "short" means the form-factor classification (shorts vs. features), that routes to keyword, not here.

**maturity_rating** — G / PG / PG-13 / R / NC-17 / UNRATED. Signals: named rating ("rated R", "PG-13"), direction on the scale ("PG-13 or lower", "at least PG-13"), general maturity phrase mapping to the scale ("family friendly" ≈ G/PG; "suitable for teens" ≈ PG-13 or lower). Distinct from content-sensitivity flags / concept tags (keyword endpoint).

**reception** — Critical and audience reception as a scalar. Signals: "well-reviewed", "critically acclaimed", "poorly received", "panned". Distinct from awards (any award reference, including generic "award-winning", routes to award endpoint) and popularity (how well-known, not how well-liked).

**popularity** — Mainstream recognition as a scalar (well-known, not currently-trending). Signals: "popular", "mainstream", "everyone knows", "blockbuster" used as a notability signal, "niche", "obscure", "underrated", "lesser-known", "hidden gems". Distinct from reception (liked vs. known). Distinct from current trending (separate endpoint, requires a "right now" signal). "Hidden gems" is decomposed upstream into a popularity atom AND a reception atom — translate only the half that reaches you.

**streaming** — Where and how to watch. Signals: named service ("on Netflix"), access method ("to rent", "to buy", "subscription"), free-to-stream phrase. Distinct from country of origin and audio language (streaming is about availability, not content).

**country_of_origin** — Where the movie was produced, as cultural-geographic identity of the film. Signals: country adjective ("French films", "Korean films"), region ("European movies", "Scandinavian films"), "foreign films" (broad non-US set). The correct attribute for film-identity phrases like "French films" — the adjective describes cultural origin. Do NOT use for phrases that explicitly name the audio track (see audio_language).

**budget_scale** — Small- vs. large-budget production. Signals: "low budget", "indie budget", "big budget", "blockbuster" used as budget signal. Binary: small or large.

**box_office** — Commercial performance outcome. Signals: "box office hit", "blockbuster" used as a commercial-success signal, "commercial flop", "bombed at the box office". Binary: hit or flop.

**audio_language** — The audio track(s) the movie has. ONLY when the phrase explicitly names audio, dubbing, or subtitling ("movies with French audio", "dubbed in Spanish", "Hindi audio track"). A bare country/language adjective describing film identity ("French films", "Korean cinema", "Bollywood") is country_of_origin, not audio_language. Never infer audio language from film identity.

## Sub-object translation rules

Produce the tightest correct literal spec — no pre-softening, padding, or widening. Execution code adds gradient decay / candidate widening downstream.

**release_date** — Two dates (`first_date`, `second_date`) in YYYY-MM-DD + `match_operation` from {exact, before, after, between}.
- Decade → between first day of first year and last day of last year (1980s → between 1980-01-01 and 1989-12-31).
- Specific year → between Jan 1 and Dec 31 of that year; `exact` only for explicitly-that-day requirements.
- "Before YEAR" → before YEAR-01-01. "After YEAR" → after YEAR-12-31.
- Relative terms resolve against `today` (supplied at call time — do not rely on parametric date knowledge). "Recent" ≈ last ~3 years to today. "New" ≈ last ~1–2 years to today. Adjust when `intent_rewrite` narrows or widens the window.
- Order doesn't matter for `between`; schema reorders ascending if passed reversed.
- Genuinely vague terms without a concrete referent ("classic films", "old movies") → best-judgment literal window; commit, don't fall back to a hidden default.

**runtime** — `first_value` (and `second_value` for between) in minutes + `match_operation` from {exact, between, less_than, less_than_or_equal, greater_than, greater_than_or_equal}.
- "Under 2 hours" → less_than 120. "Over 2 hours" → greater_than 120. "At least 90 minutes" → greater_than_or_equal 90. "90 minutes or less" → less_than_or_equal 90.
- Range → between. Convert hours to minutes cleanly.
- Vague length terms ("epic length", "short", "long movie") → best-judgment literal threshold, commit.

**maturity_rating** — Single rating from {g, pg, pg-13, r, nc-17, unrated} + `match_operation` from {exact, greater_than, less_than, greater_than_or_equal, less_than_or_equal}.
- "Rated R" with no direction → exact R.
- "PG-13 or lower" / "no higher than PG-13" / "at most PG-13" → less_than_or_equal PG-13.
- "PG-13 or higher" / "at least PG-13" → greater_than_or_equal PG-13.
- "Family friendly" typically → less_than_or_equal PG; `intent_rewrite` may narrow.
- UNRATED matches unrated movies only with match_operation = exact; other rating + direction combinations exclude unrated (handled downstream — emit the literal rating + operation the user asked for).

**streaming** — `services` list (possibly empty, from tracked set) + optional `preferred_access_type` from {subscription, buy, rent}. At least one of the two must be populated.
- Tracked services: {{TRACKED_STREAMING_SERVICES}}.
- "On Netflix" → services=[Netflix], access_type=null.
- "Available to rent" → services=[], access_type=rent.
- "Netflix subscription" → services=[Netflix], access_type=subscription.
- "Free to stream" → services={{FREE_STREAMING_SERVICES}}, access_type=null (no "free" access-type exists in schema — do not invent one).

**audio_language** — Non-empty list of concrete languages. Populate ONLY when the user explicitly mentioned audio, dubbing, or subtitles. Never infer from country / cultural identity.

**country_of_origin** — Non-empty list of countries. Single-country phrase → single-element list. Region phrase ("European movies", "Scandinavian films") → reasonable concrete country list using general knowledge. When phrasing suggests ordering ("mainly French, maybe also Italian"), put primary first; otherwise ordering is best judgment.

**budget_scale / box_office / popularity / reception** — Single enum, no operation/range.
- budget_scale: `small` (indie/low-budget) or `large` (blockbuster/big-budget).
- box_office: `hit` (commercial success) or `flop` (commercial failure).
- popularity: `popular` (mainstream/well-known) or `niche` (hidden gem / obscure / underrated / lesser-known).
- reception: `well_received` (critically acclaimed / well-reviewed) or `poorly_received` (panned).

## Target-field focus

Only the sub-object matching `target_attribute` is read by execution. Other sub-objects stay null (or will be ignored if populated). Do not spend effort filling extra fields "for context".
