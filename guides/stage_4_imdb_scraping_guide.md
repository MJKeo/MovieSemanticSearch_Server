# Stage 4: IMDB Scraping — Detailed Technical Guide

## Overview

This stage takes the ~100K movies that passed the TMDB quality funnel and scrapes 6 IMDB pages per movie to extract structured data that TMDB doesn't provide. The scraped data feeds directly into LLM generation (Stages 5–6), embedding (Stage 7), and database ingestion (Stage 8). It runs on the local laptop, routing all requests through budget residential proxies.

This is a one-time bulk operation. The daily update pipeline will have its own separate (and much cheaper) scraping strategy designed later.

**Input:** All `movie_progress` rows where `status = 'quality_passed'`

**Output:** One JSON file per movie at `./ingestion_data/imdb/{tmdb_id}.json` containing the merged extraction from all 6 pages, plus cached raw HTML at `./ingestion_data/imdb_html/{imdb_id}_{page_type}.html`

**Status transition:** `quality_passed` → `imdb_scraped` (success) or `scrape_failed` (all 6 page fetches failed / IMDB 404)

**Expected duration:** 4–8 hours at semaphore=30

**Expected cost:** $5–17 in proxy bandwidth via DataImpulse

---

## What We Scrape and Why

### Page 1 — Main Page (`/title/{imdb_id}/`)

The main page is the densest single source. Its `__NEXT_DATA__` JSON contains two major data blocks: `aboveTheFoldData` (ATF) and `mainColumnData` (MCD).

**Fields extracted:**

| Field | Source Block | Default | Downstream Use |
|---|---|---|---|
| `original_title` | MCD → `originalTitleText.text` | `None` | Production vector — identifies foreign-language films |
| `maturity_rating` | ATF → `certificate.rating` | `None` (treated as "unrated") | Maturity rank in Postgres `movie_card`, Qdrant payload filter, content-sensitivity vectors. Many foreign, indie, and older films legitimately lack MPAA certificates |
| `overview` | ATF → `plot.plotText.plainText` | `None` (TMDB overview used as fallback) | LLM prompt context for all generation phases; the TMDB overview from Stage 2 serves as the primary fallback |
| `keywords` | ATF → `interests.edges` (up to 8) | `[]` | These are IMDB's "interest" keywords (broader/genre-level like "Anime", "Coming-of-Age"). Used in the dense anchor vector alongside plot keywords |
| `imdb_rating` | ATF → `ratingsSummary.aggregateRating` | `None` | `reception_tier` calculation, `reception_score` in Postgres, reception vector |
| `imdb_vote_count` | ATF → `ratingsSummary.voteCount` | `0` | Not directly stored in final databases but useful for confidence weighting during LLM prompt construction |
| `metacritic_rating` | ATF → `metacritic.metascore.score` | `None` | `reception_tier` calculation (combined with IMDB rating), `reception_score` in Postgres |
| `user_review_summary` | MCD → `reviewSummary.overall.medium.value.plaidHtml` | `None` | LLM prompt context for reception metadata generation |
| `genres` | ATF → `genres.genres` | `[]` | `genre_ids` in Postgres and Qdrant payload; genre filtering; anchor vector text |
| `countries_of_origin` | MCD → `countriesDetails.countries` | `[]` | Production vector — "Produced in [countries]" |
| `production_companies` | ATF → `production.edges` | `[]` | Production vector, lexical `inv_studio_postings` table |
| `filming_locations` | MCD → `filmingLocations.edges` | `[]` | Production vector — real-world filming locations |
| `languages` | MCD → `spokenLanguages.spokenLanguages` | `[]` | `audio_language_ids` in Postgres; production vector — "Audio: [languages]" |
| `budget` | MCD → `productionBudget.budget.amount` | `None` | Production vector — converted to era-aware budget bucket |
| `review_themes` | MCD → `reviewSummary.themes` | `[]` | Each theme has a `name` and `sentiment`. Fed to LLM for reception metadata generation |

**No movies are filtered out during the scrape based on missing data.** Every field defaults gracefully — `None` for scalars, `[]` for lists. If `maturity_rating` is absent, it defaults to `None` and is treated as "unrated" downstream. If `overview` is absent, the TMDB overview (which was validated during the quality funnel) serves as the fallback during LLM prompt construction. The parsers must never raise on missing data. Data quality decisions belong in a separate validation step before LLM generation, not inside the scraper.

### Page 2 — Plot Summary (`/title/{imdb_id}/plotsummary/`)

Contains user-contributed plot summaries (paragraph-length) and full synopses (multi-paragraph, spoiler-containing).

**Fields extracted:**

| Field | Source Path | Default | Downstream Use |
|---|---|---|---|
| `synopses` | `contentData.data.title.plotSynopsis.edges` | `[]` | LLM Phase 1 — detailed spoiler-containing plot used for `plot_events_metadata.plot_summary` (the chronological retelling). Also feeds Phase 2 for `plot_analysis_metadata` |
| `plot_summaries` | `contentData.data.title.plotSummaries.edges` | `[]` | LLM Phase 1 — fallback input for `plot_events_metadata` generation when no synopsis exists. Multiple summaries from different users collectively approximate the depth of a single synopsis |

**Extraction logic:**

The goal is to give the LLM the richest possible plot context without redundancy.

1. Check if any synopses exist. If yes, take the **first synopsis only** and return it. A single synopsis is a comprehensive chronological retelling — often 500–2,000 words. Multiple synopses are redundant because they all retell the same story with minor wording differences. Additional synopses would burn LLM input tokens for no new information.

2. If no synopses exist, fall back to plot summaries. Take the **first 3 summaries, excluding the overview** (the first summary in IMDB's list duplicates the overview already captured from the main page, so skip it). Summaries are written by different users with different emphasis — one might focus on the protagonist's arc, another on the central conflict, a third on the setting. Three of them collectively approximate what a single synopsis would provide.

3. If neither synopses nor summaries exist (beyond the overview duplicate), both fields return empty. The LLM falls back to the TMDB overview from Stage 2.

**Why this page matters:** The TMDB overview is typically 1–3 sentences. IMDB synopses can be thousands of words. This depth is what lets the LLM generate accurate `plot_events_metadata` with settings, character arcs, and chronological events. Without it, plot vectors would be shallow.

### Page 3 — Keywords (`/title/{imdb_id}/keywords/`)

Contains IMDB's community-curated plot keywords — more specific and granular than the "interest" keywords from the main page.

**Fields extracted:**

| Field | Source Path | Default | Downstream Use |
|---|---|---|---|
| `plot_keywords` | `contentData.data.title.keywords.edges` | `[]` | Plot events vector, plot analysis vector, dense anchor vector. These are granular narrative keywords like "high school", "breaking the fourth wall", "truancy" — distinct from the broader genre-level interest keywords on the main page |

**Extraction logic:**

The goal is to fetch all high-quality keywords without using an arbitrary count cutoff that would under-fetch for keyword-rich movies and over-fetch for keyword-poor ones.

Each keyword node in the `__NEXT_DATA__` JSON contains vote data at `node.interestScore`:

- `usersInterested` — number of users who agreed this keyword applies (likes)
- `usersVoted` — total number of users who voted (likes + dislikes)
- Dislikes are derived: `dislikes = usersVoted - usersInterested`

**Scoring formula:** `score = usersInterested - 0.75 * (usersVoted - usersInterested)`

The 0.75 dislike weight reflects that users are quicker to downvote than upvote. A keyword with a 50/50 split (e.g., 10 interested, 20 voted) scores `10 - 0.75 * 10 = 2.5` — still positive, still included. This is intentional: a contested keyword like "twist ending" where half the audience disagrees is still useful for semantic search. A user searching for "movie with a twist ending" should find movies where a significant portion of the audience perceived one.

**Threshold logic:**

1. Compute the score for every keyword on the page.
2. Let N = the score of the highest-scoring keyword.
3. If N = 0 (no keyword has meaningful votes), take the first 5 keywords by position (IMDB's default ordering). No further scoring logic applies.
4. If N > 0, compute the inclusion threshold: `threshold = min(0.75 * N, N - 2)`.
5. Include all keywords with `score >= threshold`.
6. Enforce a floor of 5 and a cap of 15. If fewer than 5 keywords pass the threshold, pad with the next-highest-scoring keywords up to 5 (or however many exist). If more than 15 pass, take the top 15 by score.

**Why the threshold formula works:**

The `min(0.75 * N, N - 2)` formula adapts to the engagement level of the movie. For popular movies with high vote counts (N=20), the threshold is `min(15, 18) = 15`, taking keywords in the top 75% — a tight relative band. For lower-engagement movies (N=4), the threshold is `min(3, 2) = 2`, taking everything within 2 points of the top — a wider absolute band that's more inclusive when the vote signal is weaker. The crossover point is at N=8. Below that, the absolute band dominates, casting a wider net because the ordering itself is weaker signal. The cap of 15 prevents blockbusters with 30+ well-voted keywords from bloating LLM input tokens with increasingly generic tail keywords (ranked 20th–30th are typically things like "explosion" or "based on comic book" that add noise rather than specificity).

**Why this page matters:** Plot keywords are the primary structured signal for thematic matching. A user searching for "movie about breaking the fourth wall" benefits directly from these keywords existing in the plot vectors. TMDB keywords exist but tend to be less granular and less consistently curated than IMDB's.

### Page 4 — Parental Guide (`/title/{imdb_id}/parentalguide/`)

Contains the MPAA/content rating reasoning and category-by-category severity assessments contributed by IMDB users.

**Fields extracted:**

| Field | Source Path | Default | Downstream Use |
|---|---|---|---|
| `ratingReasons` | `contentData.data.title.ratingReason.edges` | `[]` | Combined with `maturity_rating` to produce `maturity_reasoning` text for LLM generation. Feeds the dense anchor vector's maturity guidance section |
| `parentsGuide` | `contentData.data.title.parentsGuide.categories` | `[]` | Each item has `category` (e.g., "sex & nudity", "violence & gore") and `severity` (e.g., "mild", "severe"). Stored as `parental_guide_items`. Used in LLM generation for `viewer_experience_metadata.disturbance_profile` and content-sensitivity matching |

**Note:** Parental guide entries with `severity = "none"` are filtered out during extraction — they carry no signal.

**Why this page matters:** This is the only source of detailed content-sensitivity data. The maturity rating alone (e.g., "PG-13") tells you the rating; the parental guide tells you *why* ("moderate violence, mild language, brief nudity"). This powers the disturbance profile in viewer experience vectors and lets users search for or avoid specific content types.

### Page 5 — Full Credits (`/title/{imdb_id}/fullcredits/`)

Contains the complete cast and crew listing, organized by role category.

**Fields extracted:**

| Field | Source Path | Default | Downstream Use |
|---|---|---|---|
| `directors` | Categories → "Director" items | `[]` | Production vector, lexical `inv_person_postings`, dense anchor vector |
| `writers` | Categories → "Writers" items | `[]` | Production vector, lexical `inv_person_postings` |
| `cast` | Categories → "Cast" items (up to `splitIndex`) | `[]` | Production vector (main 5), lexical `inv_person_postings`, dense anchor vector |
| `characters` | Nested within cast items | `[]` | Lexical `inv_character_postings`, production vector (main 5 character names), dense anchor vector |
| `producers` | Categories → "Producers" items | `[]` | Production vector (first 4) |
| `composers` | Categories → "Composer" items | `[]` | Production vector |

**Note:** Cast is truncated at IMDB's `splitIndex` — this is the natural dividing line between principal cast and extras/minor roles. This avoids ingesting hundreds of uncredited extras for large productions.

**Why this page matters:** While TMDB provides cast/crew via the `credits` append, the IMDB full credits page offers a more complete and differently structured view. Character names in particular are more reliably populated on IMDB than TMDB. For lexical search, character name coverage directly affects whether a query like "movie with a character named Keyser Söze" returns results.

### Page 6 — Reviews (`/title/{imdb_id}/reviews/`)

Contains user-written reviews with titles and full text.

**Fields extracted:**

| Field | Source Path | Default | Downstream Use |
|---|---|---|---|
| `featured_reviews` | `contentData.reviews` (up to 10) | `[]` | Each review has `summary` (title) and `text` (body). Fed to LLM Phase 1 for `reception_metadata` generation — praise attributes, complaint attributes, and the `new_reception_summary`. Also informs `viewer_experience_metadata` in Phase 2 |

**Why this page matters:** User reviews are the only source of authentic audience sentiment. The LLM uses them to generate grounded reception metadata — what people actually praised and complained about. Without reviews, reception vectors would be based solely on numeric ratings, losing all the qualitative signal that makes searches like "movies praised for their cinematography" work.

---

## Structural Principles

### Fully async with a single global semaphore

The parallelism model is `asyncio` + `aiohttp`. One global semaphore controls the total number of in-flight HTTP requests across all movies. Every request — regardless of which movie or which page — must acquire the semaphore before firing. This is the single source of truth for concurrency. Start at **semaphore=30** for this one-time run (conservative, to minimize wasted bandwidth on retries) and tune based on the DataImpulse dashboard's traffic and request metrics.

### All 6 pages fan out concurrently per movie

For a given movie, all 6 page fetches launch simultaneously via `asyncio.gather`. They each independently acquire the global semaphore, so total concurrency stays bounded regardless of how many movie-level coroutines are active. There is no serial dependency between pages — no gate check, no early exit.

### Cache raw HTML before parsing

Every raw HTML response is saved to disk at `./ingestion_data/imdb_html/{imdb_id}_{page_type}.html` before the parser runs. This is a one-time safety net: if a parser bug surfaces at movie 40,000, you fix the parser and reprocess locally from cached HTML instead of re-fetching through the proxy. At ~20KB compressed per page, 600K pages = ~12GB of local disk. Delete the cache after confirming all parsing is correct.

### Note on `__NEXT_DATA__` and bandwidth

The parsers extract data from the `__NEXT_DATA__` JSON blob embedded in a `<script>` tag within the full HTML page. There is no way to request only this blob from IMDB — every HTTP request returns the complete HTML document, and the proxy bills you for the full page. The `__NEXT_DATA__` extraction saves parsing complexity and memory (you work with structured JSON instead of scraping DOM elements), but it does not reduce bandwidth on the wire. What does reduce bandwidth is gzip/brotli compression, which the server applies to the full HTML before transmission. With compression active, pages that are 50–80KB uncompressed arrive as 15–25KB on the wire — this is the billable size.

### Chunk-based commit with progress reporting

Movies are processed in chunks of 500. After each chunk, results are committed to the SQLite tracker and progress is printed. This provides natural resume points and visibility into the scrape's progress without adding any throttling beyond the semaphore.

### Per-request IP rotation with US geo-targeting

Use DataImpulse's rotating port (823) which assigns a new IP on every request. Constrain to **US-only IPs** via the `__cr.us` username parameter. This ensures IMDB serves US-localized content: MPAA maturity ratings, English language defaults, and US-region metadata. Each of the 600K requests appears to come from a different US IP address.

### User-Agent rotation via `fake-useragent`

Use the `fake-useragent` Python package to generate realistic, current User-Agent strings. Initialize it once at scraper startup and call it per-request. This package pulls from real browser usage statistics, so the strings it returns are actual UAs observed in the wild — not synthetic ones that might trigger detection.

Filter to **desktop Chrome and Firefox only**. Skip Safari (low market share makes a Safari UA from a residential IP look slightly unusual) and skip all mobile UAs (IMDB's mobile pages may serve different HTML structure or a different `__NEXT_DATA__` shape, which would break your parsers). The desktop constraint ensures you always get the standard desktop page layout your parsers are built for.

### Random inter-request delays

Add a small random delay (0.05–0.3 seconds) after acquiring the semaphore and before firing each request. This breaks up burst patterns that can trigger detection even with IP rotation.

---

## Fetching Flow — Step by Step

### Step 1: Load candidate movies

Query the SQLite tracker for all movies where `status = 'quality_passed'`. This gives you approximately 100K `(tmdb_id, imdb_id)` pairs. On restart after a crash, this same query naturally returns only unprocessed movies since successful movies have already been moved to `imdb_scraped` or `scrape_failed`.

### Step 2: Initialize the aiohttp session and proxy

Create a single `aiohttp.ClientSession` with a `TCPConnector` limited to 100 concurrent TCP connections. Configure the proxy URL with your DataImpulse credentials and US country targeting (`login__cr.us:password@gw.dataimpulse.com:823`). Set the `Accept-Encoding: gzip, deflate, br` header at the session level so it applies to every request.

### Step 3: Process movies in chunks of 500

For each chunk, launch one coroutine per movie. Each coroutine runs the per-movie flow described in Step 4. Use `asyncio.gather` with `return_exceptions=True` at the chunk level to collect results without one movie's failure crashing the chunk.

### Step 4: Per-movie flow (inside each coroutine)

**4a.** Construct the 6 URLs for this movie's IMDB pages.

**4b.** Launch all 6 page fetches concurrently. Each fetch independently acquires the global semaphore, adds a random delay, sets a random User-Agent, and makes the request through the proxy.

**4c.** Each individual fetch has its own retry loop (up to 3 attempts). On HTTP 403 or 429, wait with exponential backoff (the proxy rotation gives a new IP automatically on the next attempt). On HTTP 200, return the response text. On HTTP 5xx or network timeout, retry with backoff. After exhausting retries, return an error.

**4d.** Collect results from all 6 fetches. The only scenario that prevents saving data for this movie is if the main page returned HTTP 404 (movie doesn't exist on IMDB) or if all 6 pages failed after exhausting retries (total infrastructure failure). In those cases, log to `filter_log` with `stage='imdb_scrape'` and set `status='scrape_failed'`. For partial failures (e.g., 4 of 6 pages succeeded, reviews and keywords failed), save whatever data was successfully fetched and proceed — the downstream pipeline handles missing fields gracefully.

**4e.** Cache all successfully fetched raw HTML responses to disk at `./ingestion_data/imdb_html/{imdb_id}_{page_type}.html`.

**4f.** Run the parsers on the cached HTML. Parsers never raise on missing data. Every field defaults to `None` (scalars) or `[]` (lists). If `maturity_rating` is absent, it stays `None` and is treated as "unrated" downstream. If `overview` is absent, the TMDB overview is used as fallback during LLM prompt construction.

**4g.** Merge the parser outputs into a single dictionary. Save to `./ingestion_data/imdb/{tmdb_id}.json`.

**4h.** Update `movie_progress` to `status = 'imdb_scraped'`.

### Step 5: Commit chunk and report progress

After each 500-movie chunk, commit the SQLite transaction and print progress. This is the natural resume point.

### Step 6: No data-quality filtering during scrape

The scrape stage does not filter movies based on data quality. Every movie that reaches Stage 4 already passed the TMDB quality funnel, which validated the existence of a TMDB overview, genres, runtime, IMDB ID, and sufficient vote engagement. The IMDB scrape is purely additive — it gathers richer data to layer on top. If any IMDB fields are missing (no maturity rating, no reviews, sparse credits), the movie is still saved with whatever was extracted. Data quality evaluation for vector generation fitness happens in a separate validation step before LLM generation, not here.

### Step 7: Datacenter-to-residential switchover (if needed)

If the 500-movie pilot reveals that DataImpulse datacenter proxies get blocked by IMDB (high 403 rate, CAPTCHAs), create a Residential plan on the DataImpulse dashboard. Update the proxy URL with the new residential credentials — the gateway host, port, and `__cr.us` parameter format are identical. The checkpoint system picks up exactly where you left off. Unused datacenter bandwidth stays in your account for other uses.

---

## The Merged Movie Object

After all 6 pages are parsed and merged, the JSON file for each movie at `./ingestion_data/imdb/{tmdb_id}.json` contains:

**From main page:** `original_title`, `maturity_rating`, `overview`, `keywords` (interest-based), `imdb_rating`, `imdb_vote_count`, `metacritic_rating`, `user_review_summary`, `genres`, `countries_of_origin`, `production_companies`, `filming_locations`, `languages`, `budget`, `review_themes`

**From summary page:** `synopses` (first synopsis only if any exist), `plot_summaries` (first 3 excluding overview, only if no synopses exist)

**From keywords page:** `plot_keywords` (dynamically scored with floor of 5, cap of 15)

**From parental guide page:** `maturity_reasoning` (from ratingReasons), `parental_guide_items` (from parentsGuide)

**From credits page:** `directors`, `writers`, `actors` (from cast), `characters`, `producers`, `composers`

**From reviews page:** `featured_reviews`

This merged object is what Stage 5 (LLM Phase 1) reads as input. The field names in the final object may differ from the raw parser output — the merge step handles any renaming (e.g., parser returns `cast` → merged object stores as `actors`; parser returns `ratingReasons` → merged object stores as `maturity_reasoning`; parser returns `parentsGuide` → merged object stores as `parental_guide_items`).

---

## Edge Cases

### Movie exists on TMDB but not on IMDB

Should be rare — the quality funnel requires a valid `imdb_id`. But IMDB occasionally removes titles. If the main page returns HTTP 404, log with `reason='imdb_404'` and set `status='scrape_failed'`. This is one of only two scenarios where a movie is not saved during Stage 4.

### Main page missing `maturity_rating`

Many foreign, indie, and older films legitimately lack MPAA certificates. The parser defaults `maturity_rating` to `None`. Downstream, this is treated as "unrated" — the `maturity_rank` encoding in Postgres includes an "unrated" tier, and Qdrant payload filtering handles null maturity gracefully. The movie is not filtered out.

### Main page missing `overview`

If IMDB's plot text is absent, the parser defaults `overview` to `None`. The TMDB overview (validated during the quality funnel as 20+ characters) serves as the primary fallback during LLM prompt construction. The movie is not filtered out.

### Empty reviews page

Many less-popular movies have zero user reviews on IMDB. The parser returns `featured_reviews: []`. This is fine — the LLM generation for `reception_metadata` can fall back to the `user_review_summary` and `review_themes` from the main page, or generate reception metadata from the IMDB/Metacritic ratings alone. Do not filter the movie out.

### Empty synopsis / plot summaries

Some movies have no user-contributed synopses. The extraction logic falls back to the first 3 plot summaries (excluding the overview duplicate). If neither synopses nor substantive summaries exist, both fields return empty. The LLM falls back to the TMDB overview from Stage 2 for plot generation. Plot vectors will be shallower but still functional.

### Parental guide with all "none" severities

The parser already filters these out (entries with `severity = "none"` are excluded). A movie might legitimately have no parental guide data at all. The result is `parental_guide_items: []` and `maturity_reasoning: []`. Downstream, this means the disturbance profile in viewer experience will be generated from the maturity rating alone, and `parental_guide_items` uses its default of `[]`.

### IMDB serves a CAPTCHA or challenge page instead of data

This manifests as an HTTP 200 with HTML that doesn't contain `__NEXT_DATA__`. The parser returns an empty dict from `_parse_next_data`, and all fields default to `None` or `[]`. The movie is still saved with this sparse data. If this happens systematically (>15% of requests returning empty parses), it's a signal to lower the semaphore or switch from datacenter to residential proxies on DataImpulse. Monitor the ratio of "movies with `None` maturity_rating" as a proxy for CAPTCHA rate — a sudden spike indicates detection issues.

### Cast `splitIndex` is missing or -1

If IMDB doesn't provide a split index, the parser takes the full cast list. This could be very long for ensemble films. The downstream usage only needs the main 5 actors and characters, so truncation happens at the LLM prompt construction stage, not here.

### HTML encoding issues in reviews or summaries

The parsers already handle HTML entity unescaping via `html.unescape()` and BeautifulSoup text extraction. Edge cases like malformed HTML or nested entities are handled by BeautifulSoup's lenient parser. The `.get_text(" ", strip=True)` call normalizes whitespace.

### Rate limiting mid-scrape

If the error rate spikes (visible on the DataImpulse dashboard), the semaphore-based architecture naturally slows down because retries consume semaphore slots. But you should also monitor proactively. If error rates exceed 15%, pause and lower the semaphore from 30 to 20. If you're on datacenter proxies and the error rate is high, this is your signal to switch to the residential plan.

### Resumability after crash

On restart, query `WHERE status = 'quality_passed'`. Movies already at `imdb_scraped` or `scrape_failed` are skipped. The HTML cache ensures no bandwidth is re-spent. If a crash happened mid-chunk (some movies in the chunk processed, others not), the uncommitted ones simply get reprocessed in the next run.

---

## Cost Optimization Tactics

### 1. Start with DataImpulse datacenter proxies ($0.50/GB)

DataImpulse's datacenter proxies are sourced through end users rather than traditional data centers, meaning their IPs may not appear in standard datacenter blocklists. Buy the $5 trial (5GB of datacenter traffic at $1/GB — or if datacenter is separately priced, $5 gets you 10GB at $0.50/GB). Run the 500-movie pilot through datacenter proxies. If IMDB success rates are 90%+, use datacenter for the full scrape and your total proxy cost drops to $5–9. If success rates are poor (lots of 403s or CAPTCHAs), switch to residential at $1/GB for the remainder.

### 2. Verify gzip compression is active

This determines whether you're paying for 30–50GB or 9–15GB. During the pilot batch, log the `Content-Encoding` response header. If IMDB is returning `gzip` or `br`, you're in the 9–15GB range. If not, investigate — the `Accept-Encoding` header may not be reaching IMDB through the proxy, or the proxy may be decompressing and re-serving uncompressed responses. Contact DataImpulse support if this happens — their live chat is 24/7 and responsive.

### 3. Run a 500-movie pilot first

Before the full 100K run, scrape 500 movies end-to-end. This validates all 6 parsers, confirms compression is active, establishes the actual error rate, and catches any parser bugs. Cost: <$0.50 in bandwidth. This prevents discovering issues at movie 40,000 and having to debug under bandwidth pressure.

### 4. Start the semaphore conservatively

Begin at semaphore=30. Monitor the DataImpulse dashboard for error rates over the first 1,000 movies. If errors are under 3%, increase to 40, then 50. If errors exceed 10%, drop to 20. Every failed request wastes 15–25KB of bandwidth on a response you can't use, plus the bandwidth of subsequent retries.

### 5. Don't retry empty-but-valid pages

If a page returns HTTP 200 and the parser returns empty data (no reviews, no keywords, etc.), that's valid — the movie simply doesn't have that data. Don't retry it. Only retry on HTTP errors (403, 429, 5xx) or network failures.

### 6. Time the purchase around promotions

DataImpulse runs occasional promotions (they've done Black Friday sales in the past). Since the bulk scrape isn't time-sensitive, check for active discounts before purchasing beyond the trial. At already-low rates, even a modest discount saves a few dollars.

---

## Proxy Integration — DataImpulse

DataImpulse is the recommended provider for this bulk scrape. Residential proxies are $1/GB with non-expiring traffic. Datacenter proxies are $0.50/GB. A $5 trial gets you 5GB to validate the setup before committing further. Full documentation is at `docs.dataimpulse.com`.

### Account setup

1. Register at `dataimpulse.com` and click "Try Now."
2. From the dashboard, click "+ Create a new order" and select either Datacenter ($0.50/GB) or Residential ($1/GB) proxy type. Start with Datacenter for the $5 trial — if it works on IMDB, you save 50%. If it doesn't, buy a Residential plan for the remainder.
3. After purchase, your plan page shows your **login**, **password**, and the **gateway host**. These are the credentials you'll use in code.

### Gateway and ports

DataImpulse uses a backconnect gateway at `gw.dataimpulse.com`. The port determines the connection behavior:

- **Port 823** — Rotating proxy. A new IP is assigned on every request. This is what you want for the IMDB scrape.
- **Ports 10000–20000** — Sticky proxy. The same IP is bound to a port for up to 120 minutes. Do not use this for the bulk scrape.

### US geo-targeting via inline parameters

Country targeting is configured by appending parameters to the username using the `__` delimiter. The parameter key for country is `cr` and the value is the ISO 2-letter country code.

**Format:** `login__cr.us:password@gw.dataimpulse.com:823`

This constrains all requests to US-based IPs only, ensuring IMDB serves MPAA maturity ratings, English language content, and US-region metadata. Country targeting is free — it does not change the per-GB rate. Do not use city/state/ZIP targeting as those parameters double the traffic cost.

### Full proxy URL for aiohttp

Putting it all together, the proxy URL you pass to aiohttp's `proxy` parameter is:

`http://YOUR_LOGIN__cr.us:YOUR_PASSWORD@gw.dataimpulse.com:823`

Replace `YOUR_LOGIN` and `YOUR_PASSWORD` with the credentials from your DataImpulse dashboard. The `__cr.us` suffix on the login is what activates US country targeting. This single URL is the only proxy configuration needed — DataImpulse handles IP rotation automatically on port 823.

### aiohttp session configuration

Set these headers at the **session level** so they apply to every request:

- `Accept-Encoding: gzip, deflate, br` — enables compression, critical for reducing billable bandwidth
- `Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8` — standard browser accept header
- `Accept-Language: en-US,en;q=0.9` — reinforces English-language IMDB responses alongside the US geo-targeting

Create the session with a `TCPConnector` limited to 100 concurrent TCP connections. This caps TCP-level concurrency to the proxy gateway independently of the request-level semaphore.

### Per-request configuration

- `User-Agent` — generated via `fake-useragent`, filtered to desktop Chrome and Firefox only (no Safari, no mobile). Initialize the `UserAgent` object once at scraper startup; call it per-request.
- The proxy URL stays constant across all requests — DataImpulse rotates the IP automatically on port 823.

### Switching from datacenter to residential (if needed)

If the datacenter trial shows high error rates on IMDB (>10% 403s or CAPTCHAs), switch to residential. The only change is:

1. Create a new Residential plan on the DataImpulse dashboard.
2. You'll get a new login/password pair for the residential plan.
3. Update the proxy URL with the new credentials. The gateway host, port, and parameter format are identical.

Your unused datacenter bandwidth stays in your account for future use on less-protected targets.

### Monitoring

Use the DataImpulse dashboard as the primary signal during the scrape. It shows:

- **Traffic consumed** — bandwidth in GB, broken down by time interval
- **Request count** — per-minute request volumes
- **Per-site breakdown** — which domains you're hitting and how much traffic each consumed

Check the dashboard every 30–60 minutes during the scrape. If you see traffic consumption accelerating unexpectedly (possible sign that compression isn't active) or request counts dropping (possible sign of rate limiting), investigate before continuing.

---

## Graceful Defaults

Every field extracted during the scrape defaults gracefully when absent. No movie is filtered out based on missing data. Data quality decisions happen downstream, before LLM generation.

| Field | Default | Downstream Handling |
|---|---|---|
| `maturity_rating` | `None` | Treated as "unrated" — `maturity_rank` includes an unrated tier |
| `overview` | `None` | TMDB overview (validated in quality funnel) used as fallback for LLM prompts |
| `imdb_rating` | `None` | `reception_tier` falls back to metacritic only, or "unrated" tier if both missing |
| `metacritic_rating` | `None` | `reception_tier` uses IMDB rating only |
| `filming_locations` | `[]` | Production vector omits location line |
| `budget` | `None` | Production vector omits budget bucket |
| `user_review_summary` | `None` | LLM reception prompt relies on reviews and review_themes |
| `featured_reviews` | `[]` | LLM reception prompt relies on review_themes and ratings |
| `review_themes` | `[]` | LLM reception prompt relies on reviews and ratings |
| `maturity_reasoning` | `[]` | Maturity guidance uses rating alone without reasoning detail |
| `parental_guide_items` | `[]` | Disturbance profile generated from rating and genres only |
| `plot_summaries` | `[]` | Only populated when no synopsis exists. LLM plot generation uses TMDB overview as final fallback |
| `synopses` | `[]` | First synopsis taken if any exist. LLM plot generation falls back to summaries, then TMDB overview |
| `plot_keywords` | `[]` | Dynamically scored with floor of 5 / cap of 15. Plot vectors rely on LLM-generated themes if empty |
| `genres` | `[]` | TMDB genres (from Stage 2) used as fallback |
| `languages` | `[]` | TMDB `spoken_languages` used as fallback |
| `directors`, `writers`, `cast`, etc. | `[]` | TMDB credits (from Stage 2) used as fallback |

The only scenarios where a movie is *not* saved during Stage 4 are: IMDB returns HTTP 404 on the main page (movie doesn't exist), or all 6 page fetches fail after exhausting retries (total infrastructure failure). Everything else produces a saved JSON file and advances to `imdb_scraped`.

---

## Bandwidth and Cost Estimate

| Metric | Value |
|---|---|
| Movies | 100,000 |
| Pages per movie | 6 |
| Total page fetches | 600,000 |
| Per-page compressed size (gzip expected) | 15–25KB |
| Total bandwidth (expected) | 9–15GB |
| Retry overhead (~10%) | 1–2GB |
| **Total billable bandwidth** | **10–17GB** |

| Scenario | Cost |
|---|---|
| Datacenter works, full scrape at $0.50/GB | **$5–9** |
| Datacenter fails pilot, residential at $1/GB | **$10–17** |
| Hybrid: datacenter for most, residential for retries | **$8–13** |
| **Expected outcome** | **$5–17** |

---

## Post-Scrape Cleanup

Once all 100K movies are scraped and the full pipeline has completed through database ingestion:

1. Delete the HTML cache at `./ingestion_data/imdb_html/` to reclaim ~12GB of disk space
2. The parsed JSON files at `./ingestion_data/imdb/` are worth keeping — they're only ~3–5GB total and useful for debugging or re-running the quality funnel with different parameters