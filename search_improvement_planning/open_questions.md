# Open Questions

Unresolved conversation topics, untested theories, and questions that need answers.

---

## Architecture Questions

### ~~Can the LLM reliably classify deal-breakers vs preferences?~~ DECIDED

Phase 0 will be a structured LLM call that produces the full query
decomposition in one pass:

1. Identify and articulate the user's intent in clear terms
2. Classify the query type (similarity search vs. all other searches)
3. For non-similarity searches, extract:
   - Lexical entities (actors, directors, franchises, title substrings)
   - Deterministic metadata filters (genre, country, date, runtime, etc.)
   - Semantic deal-breakers (vector-retrieved must-haves)
   - Semantic preferences (vector-retrieved nice-to-haves)
   - Reranking/sorting criteria (quality, chronological, reception, etc.)
     — null if none stated, which triggers default quality composite

This is the full decomposition from day one, not a staged rollout of
deal-breaker/preference first and scoring modes later. Each extraction step
is tractable for an LLM with good few-shot examples. Rule-based overrides
(named entities are always deal-breakers) and confidence-based fallback to
broad search serve as safety nets.

### ~~Does this replace the channel weight system or layer on top?~~ DECIDED

**The channel weight concept is eliminated entirely.** Deal-breakers are
deal-breakers regardless of which retrieval channel fulfills them — a
lexical entity match and a semantic vector match are equally non-negotiable.
The current process of having the LLM determine channel weights is unreliable
and the new decomposition structure makes it unnecessary.

Phase 0's structured output implicitly determines which channels are involved:
lexical entities route to lexical search, metadata filters route to Postgres,
semantic attributes route to vector spaces. No explicit weighting needed.

### ~~How does the anchor vector fit in the new system?~~ DECIDED

**Anchor vector is retained in V2, but only in reduced form.** The broad
jack-of-all-trades role it played in V1 is removed. Instead, anchor becomes a
lean holistic fingerprint with labeled movie-wide summary fields:

- Title and original title
- Identity pitch and identity overview
- Genre signatures and themes
- Emotional palette and key draws
- Maturity summary and reception summary

This preserves a useful "movie as a whole" semantic surface without
reintroducing the clutter that diluted V1 embeddings. Structured/filterable
facts like keywords, franchise/source material, languages, decade, budget, and
awards stay in deterministic retrieval or specialized vector spaces.

---

## Retrieval Questions

### ~~The Top-N Retrieval Gap~~ → EMPIRICALLY CONFIRMED AS SEVERE

~~Preliminary assessment was that this was less of a problem than feared.~~

**Empirical result:** The Sixth Sense does not appear in the top 1000 for
"twist ending" in narrative_techniques. It scores 0.5730 (82% of max).
Zero intersection between "funny" top-1000 and "horror" top-1000. Winter
Soldier missing entirely from "dark and gritty" vector results.

**Root cause is not retrieval depth — it's embedding quality.** Increasing
top-N from 1000 to 5000 won't fix a movie that scores 82% of max because
the embedding format dilutes its signal. See current_search_flaws.md #13.

**Implication:** Semantic deal-breakers cannot serve as reliable candidate
generators with current embeddings. Candidates must be generated from more
reliable channels (metadata, entities, keywords), with semantic concepts
applied as rescore filters on the candidate pool. See current_search_flaws.md
#14 and the revised Phase 1 architecture in new_system_brainstorm.md.

### ~~How do we handle intersection when deal-breakers span different retrieval channels?~~ ANSWERED BY TESTING

**Empirical result:** Cross-channel intersection at retrieval time fails.
"Dark gritty marvel movies" misses Winter Soldier from the vector side.
"Funny horror movies" has zero intersection between vector candidate sets.

**Answer:** Don't intersect at retrieval time. Generate candidates from
the most reliable channel (entity lookup, metadata filters, keywords),
then apply semantic deal-breakers as scoring filters via cross-space
rescoring in Phase 2. The intersection happens at scoring time, not
retrieval time. This is a fundamental revision of the Phase 1 design.

### ~~What candidate limits should we use for deal-breaker retrieval?~~ REVISED

~~500 for entity deal-breakers, 1000 for semantic deal-breakers.~~

Entity deal-breakers: 500 remains reasonable (lexical is precise).
Semantic deal-breakers as candidate generators: **no longer the primary
strategy.** Semantic concepts are applied as rescore filters on candidates
retrieved via other channels. When semantic retrieval is used (e.g.,
pure-vibe queries with no deterministic anchors), retrieval depth matters
less than embedding quality — fixing the embedding format (#13) is the
higher-priority lever.

---

## Scoring Questions

### How to set the deal-breaker threshold? — NARROWED BY EMPIRICAL DATA

**Empirical score distributions (top-2000 candidates):**

| Concept | Elbow position | Elbow % of max | Distribution shape |
|---------|---------------|----------------|-------------------|
| "twist ending" (narrative_techniques) | ~200th entry | ~75% of max | Steep initial decline, gradual tail |
| "dark and gritty" (viewer_experience) | ~200th entry | ~90% of max (0.62 vs 0.68) | Steep initial decline, gradual tail |
| "christmas" (watch_context) | ~100th entry | ~90% of max | **Bimodal**: steep decline → flat stretch → second steep decline at ~1700th |
| "funny" (viewer_experience) | ~150th entry | ~87% of max | Steep initial decline, gradual tail |

**Key findings:**
- Elbows consistently appear at 100-200 entries across concept types
- The percentage-of-max at the elbow varies significantly: 75% ("twist ending")
  to 90% ("dark and gritty," "christmas"). A single fixed percentage won't work
  well for all attribute types.
- Christmas is the only bimodal distribution, suggesting concepts with clear
  binary membership ("is or isn't a Christmas movie") produce different score
  shapes than gradient concepts ("how dark is this movie").
- The Sixth Sense scores 82% of max for "twist ending" — it would barely pass
  an 80% threshold but miss at 85%.

**Revised assessment:** Neither approach alone is sufficient.
- Relative-to-top needs to be generous (~75%) to avoid false negatives, but
  at 75% it admits ~200 candidates for most concepts — which may be too many
  for a pass/fail gate.
- Score-distribution-based (elbow detection) adapts per-query but the elbow
  percentage varies by concept type.

**Note:** Threshold selection is less critical than originally thought because
the architecture revision (see new_system_brainstorm.md) moves semantic
deal-breakers to Phase 2 rescoring rather than Phase 1 retrieval. The threshold
now operates on a candidate pool already filtered by reliable channels, which
should produce tighter, more meaningful score distributions. Revisit threshold
calibration after the embedding format improvement and architecture revision.

### ~~Where is the trim point between primary and exploratory results?~~ DECIDED

Top 25 primary results OR until score drops below 40% of the top result,
whichever is fewer. Fixed depth with a score floor. Simple, tunable, avoids
the unreliability of natural-score-gap detection.

### How should deal-breaker conformance and preference scores combine in final ranking?

Two approaches for ensuring deal-breaker conformance dominates the ranking while
preferences differentiate within conformance tiers:

**Tiered sorting:** Primary sort by deal-breaker conformance count, secondary sort
by preference + quality score within tiers. Simple, makes the guarantee explicit —
a popular 2/3 movie can never jump ahead of an obscure 3/3 movie by construction.

**Weighted formula:** Single composite score with enough weight gap that preferences
can't compensate for a missing deal-breaker. More flexible (allows near-miss
trade-offs) but requires careful tuning to prevent preferences from overriding
conformance.

The finer details — how much a preference should contribute to rescoring relative to
a deal-breaker, exact weighting, decay functions — will be decided during
implementation when we can test against real query results.

### How to handle the "spectrum deal-breaker" problem?

Deal-breakers that are spectrums create a classification problem. "Twist ending"
exists on a spectrum — does Shutter Island have a "twist ending" or just "a reveal"?

The threshold + flatten approach handles this mechanically, but the threshold
position determines where on the spectrum you draw the line. This is
acceptable — the architecture just needs to be *better than today*, not perfect.
Edge cases at position 20-25 are tolerable as long as the clear-cut examples
(Fight Club, Sixth Sense) surface in the top 5.

The threshold selection question (above) subsumes this one. Solving threshold
selection solves the spectrum problem.

### ~~How should the scoring function vary by query type?~~ DECIDED

Phase 0 outputs the full decomposition including ranking/sorting criteria.
Four scoring modes remain valid:

- **Threshold + flatten** for deal-breakers ("christmas movies")
- **Preserved similarity** for superlatives ("scariest movie ever")
- **Diminishing returns** for preferences ("dark and gritty")
- **Sort-by** for ranking axes ("critically acclaimed X")

Additionally: when Phase 0 outputs null for sorting criteria, the pipeline
applies a deterministic default quality composite (e.g., `0.6 * reception +
0.4 * popularity`). This is fixed code, not LLM-determined. Handles the
"silly comedies" problem where no explicit sorting was stated but the user
implicitly expects well-known results.

---

## Constraint Strictness

### How should NLP-extracted constraints handle user imprecision?

Users are frequently imprecise with metadata constraints. "Classic 80s action
movies" should include Terminator 2 (1991). "Leonardo DiCaprio boat movie from
the 2000s" clearly means Titanic (1997).

**Decided:** Three-tier constraint strictness model based on constraint SOURCE:

| Tier | Source | Behavior |
|------|--------|----------|
| **Hard filter** | UI controls (date picker, genre checkboxes) | Strict SQL WHERE. No exceptions. |
| **Soft constraint** | NLP-extracted metadata from query text | Generous gate + preference decay within the gate |
| **Semantic constraint** | Vector similarity | Threshold determines pass/fail |

**Soft constraint mechanics:**
1. Phase 0 extracts the constraint with a center value (e.g., `era: 1980s`)
2. Phase 1 applies a generous gate: expand range ~50% each side ("1980s" → 1975-1994)
3. Phase 2 applies preference decay: within stated range = 1.0, outside decays with distance (T2 at 1991 ≈ 0.9, a movie from 2005 ≈ 0.3)

**Attribute softness varies (for NLP-extracted constraints):**

| Attribute | Softness | Reasoning |
|-----------|----------|-----------|
| Date/era | Soft — generous ±5 years | Users imprecise about decades; cultural eras don't align to calendar decades |
| Genre | Somewhat soft — include adjacent genres | "Action movie" could include action-thriller, action-comedy |
| Certification/rating | Somewhat soft — include adjacent ratings | "PG-13" shouldn't exclude a perfect PG match |
| Country of origin | Hard | A movie is from Korea or it isn't |
| Named entities | Hard on presence | Brad Pitt is in the movie or not. Prominence is the spectrum, not presence. |
| Streaming platform | Hard | It's on Netflix or it isn't |
| Source material type | Hard | It's an adaptation or it isn't |

**Key design principle:** Strictness tier is determined by the SOURCE of the
constraint (UI vs NLP), not the attribute type. Same attribute, different
treatment based on how explicit the user was.

---

## Deal-Breaker Application

### How should deal-breakers from mixed channels be combined?

All deal-breakers are equally important regardless of channel. The system should
filter down to movies that pass all UI hard filters and attempt to conform to as
many deal-breakers as possible. Primary results = movies meeting all
deal-breakers. Fallback = movies with the highest percentage of deal-breakers met.

**Binary vs spectrum deal-breakers need separate handling:**

Some deal-breakers are binary ("has Brad Pitt" → lexical lookup, pass/fail).
Others are spectrum ("action film" → genre metadata, pass/fail; "twist ending" →
vector similarity, threshold-based). The combination works like this:

1. Binary deal-breakers: filter the candidate set (intersect)
2. Spectrum deal-breakers: threshold + flatten (pass/fail within the filtered set)
3. Count passes: rank by % of deal-breakers met
4. Within same pass-count tier: rank by preference scores

"Brad Pitt action films" → filter to movies with Brad Pitt (binary) AND action
genre (binary), then rank by preference layer (which includes Brad Pitt
prominence via billing_position scoring).

**Open sub-question:** How to handle the fallback case where no movies meet ALL
deal-breakers. If the query has 4 deal-breakers and the max any movie meets is 3,
do we show 3/4 movies? Probably yes, but the presentation should indicate which
constraint was relaxed.

---

## Presentation Questions

### ~~Append vs weave for exploratory results?~~ DECIDED

**Append.** Clearly separated "You might also like" section. Honest, sets
expectations correctly. Users can scroll past if they don't want suggestions.

Weaving creates an implicit contract that every result matches the query. When
exploratory results don't match, the experience feels broken even if the
suggestion is good.

### ~~How to signal which tier a result belongs to?~~ DECIDED

Primary results: no annotation (matching the query is the expectation).
Exploratory results: section header ("You might also like") + optional one-line
reason ("similar tone to your top results"). No per-result explanations —
expensive to generate and probably unnecessary.

---

## Retrieval & Scoring Questions

### ~~How feasible is cross-space rescoring?~~ NOW REQUIRED, NOT DEFERRED

~~Previously deferred — the union-then-rescore model was expected to handle most
cases. Empirical evidence now shows this is a real and critical gap.~~

**Empirical evidence:** "Funny horror movies" has zero intersection between
vector candidate sets. "Dark gritty Marvel movies" misses Winter Soldier from
vector results. Candidates entering via non-vector channels (entity lookup,
metadata filters) MUST be scored on semantic deal-breakers via cross-space
rescoring — this is no longer a nice-to-have.

**Implementation:** Qdrant `retrieve()` API for batched point lookups. Fetch
stored vectors for candidates in the pool, compute cosine similarity against
query embeddings for each semantic deal-breaker's target spaces. For a pool of
~500-2000 movies across 2-3 spaces, this is feasible. This is now a core part
of Phase 2, not an optional enhancement.

### ~~How should production vector / lexical overlap be handled?~~ RESOLVED

Naturally resolved by the data layer redesign. See new_system_brainstorm.md.

### ~~How should actor prominence be scored in lexical matching?~~ DECIDED

See new_system_brainstorm.md "Role-specific person posting tables."

---

## Query Understanding Questions

### ~~How should franchise entity resolution work?~~ DECIDED

See new_system_brainstorm.md "franchise_membership" table design and
"Franchise Search Flow" section for the full search strategy.

**Franchise definition:** Any recognizable intellectual property or brand from
any medium (film, video games, toys, books, comics, TV, etc.) where the movie
is an adaptation/extension of that IP. Examples: Mario, Barbie, Transformers,
Harry Potter, Marvel Cinematic Universe. The franchise name is the **IP name**,
not the film series name.

**Storage:** Only `franchise_name_normalized` is stored (no separate display-
form column). `culturally_recognized_group` is also stored normalized. Display
names can be derived at the UI layer if ever needed.

**Search approach decisions:**

- **franchise_name_normalized:** Both the ingestion LLM and search extraction
  LLM are instructed to output the most common, fully expanded form of the
  franchise/IP name — no abbreviations. Same convention as the lexical entity
  extractor for person names. This ensures both sides converge on the same
  canonical string. Searched via `inv_franchise_postings` using trigram matching
  on `franchise_name_normalized` in `lex.lexical_dictionary`. No enum or alias
  table needed.
- **franchise_role:** Stored as an integer (enum ordinal) on `franchise_membership`.
  The search extraction LLM receives the same enum definition and outputs the
  matching value. Trivially filterable with `WHERE franchise_role = $1`.
- **culturally_recognized_group:** Stored normalized. Searched via trigram
  matching within the post-franchise-lookup result set. Since franchise lookup
  narrows to 3-30 movies, simple `similarity()` or `ILIKE` on the group field
  is sufficient. No separate index or posting table needed.

**Remaining sub-questions (unchanged):**

- **LLM reliability for franchise assignment:** Expect high for mainstream
  franchises (MCU, Star Wars, HP) and well-known IPs (Mario, Barbie), spotty
  for obscure ones. Acceptable — mainstream is where franchise search matters
  most. TMDB `belongs_to_collection` covers the base case; LLM fills gaps
  best-effort.
- **"Disney" ambiguity:** Phase 0 problem, not data layer. Data stores actual
  relationships; Phase 0 interprets which level the user means from context.
- **culturally_recognized_group coverage:** Instruct LLM to leave null when no
  established term exists **globally** (any market, not just US). If multiple
  names exist across markets, prefer the American-market term. High-value for
  a small number of franchises (Star Wars trilogies, MCU phases); correctly
  absent for most.

### ~~How should the dynamic quality prior be calibrated?~~ DECIDED

**Continuous dial (0.0 to 1.0)**, not discrete modes. Phase 0 outputs a
quality_weight parameter:
- 0.0 = "hidden gems, obscurity is a feature"
- 0.5 = "neutral, quality matters but popularity doesn't"
- 1.0 = "well-known, established, canonical"

This becomes a weight on the reception/popularity signal in Phase 2 scoring.
Few-shot examples spanning the range to anchor the LLM output.

**When Phase 0 outputs null for sorting criteria:** The pipeline applies a
deterministic default composite (e.g., `0.6 * reception + 0.4 * popularity`).
This replaces the current bucket-then-sort-by-reception hack with a tunable
weighted signal.

**User-facing toggle deferred** until LLM inference is validated. If the
inference works well enough for most queries, most users never need the toggle.

### How should metadata filters handle implicit temporal signals?

"Classics," "iconic," "legendary" carry temporal-establishment implications. Now
subsumed by the three-tier constraint strictness model:

- "Disney animated classics" → Phase 0 extracts a temporal bias (prefer older) as
  a soft constraint with wide gate. Not a hard date filter. Award data from
  movie_awards can supplement as a structured "cultural establishment" signal.
- "Instant classic" → Phase 0 correctly interprets as quality + recency, outputting
  a different temporal bias.

**Remaining question:** What's the right representation for temporal bias in the
Phase 0 output? A continuous `release_year_bias` (-1.0 = older, 0.0 = neutral,
+1.0 = recent) fed into Phase 2 scoring, or a soft date range with center + decay?
Both work; the former is simpler.

### ~~When should multi-interpretation branching trigger?~~ DEFERRED TO V2

Tabled for now. The number of genuinely ambiguous movie search queries is low.
Phase 4 exploratory results partially address alternative interpretations
organically. If built later, pre-search disambiguation (clickable options before
results load) is preferred over post-search grouping.

---

## Data Layer Questions

### What's the future of the production vector space?

**Decided:** Regenerate the production vector with tightened definition — filming
locations (WHERE) + production technique keywords only (HOW). Previous definition
was too broad, causing thematic bleed. Regeneration is cheap and worth the cost.

**Remaining question:** After regeneration, is the tightened content enough to
justify a dedicated vector space? Measure whether the lean production vector
contributes meaningfully to search results. If it rarely appears in candidates'
top contributing spaces, eliminate the slot entirely (anchor is already dropped
from V2, so folding into it is no longer an option).

**Empirical evidence against current version:** For "iconic twist ending," the
routing system gave reception 23.8% weight — not ideal, but reception does carry
some twist signal in its praised_qualities. Meanwhile watch_context, which had the
most twist-related content, received zero weight. This is primarily a routing
problem (#15), but it also shows the current weight allocation system misallocates
signal across spaces generally.

### ~~How should multi-vector scores combine for a single semantic concept?~~ DECIDED

**Max.** When a concept like "twist ending" targets multiple vector spaces
(narrative_techniques, viewer_experience, reception), threshold each space
separately and take the best score. Simple, permissive — if the signal comes
through strongly in any one space, the concept is satisfied.

Alternatives considered and rejected:
- Average of above-threshold spaces (penalizes concentrated signal)
- Max for pass/fail + average for partial (unnecessary complexity)

### ~~How should keyword vocabulary mapping work?~~ DECIDED

**Answer: Pure static mapping.** The keyword vocabulary audit
([keyword_vocabulary_audit.md](keyword_vocabulary_audit.md)) revealed that
`overall_keywords` is a compact curated genre taxonomy of exactly 225 terms
— not the free-form community tagging system we assumed. 100% coverage,
near-zero long tail, trivially enumerable.

**Mapping approach:** Provide the full 225-term vocabulary as context to the
QU LLM. The LLM selects matching terms from the provided list when the user's
query implies a sub-genre or deal-breaker concept. No separate synonym table
or dynamic LLM classification needed.

**`plot_keywords` excluded:** The 114K-term `plot_keywords` vocabulary is
already consumed by the metadata generation pipeline and distilled into
structured LLM metadata. No incremental value in indexing raw plot_keywords
separately.

See [keyword_vocabulary_audit.md](keyword_vocabulary_audit.md) for the full
report including concept→keyword mappings across all deal-breaker categories.

### ~~What does source_of_inspiration re-generation look like?~~ DEFERRED

Not a priority during system redesign. Implementation details (batch pipeline
reuse, franchise_lineage removal, cost estimate) to be figured out when we're
ready to execute. Redesign the system first, then work out data pipeline changes.

### ~~How should IMDB award scraping be structured?~~ DEFERRED

Not a priority during system redesign. Implementation details to be figured out
when ready to execute. Key parameters already decided: major ceremonies only
(Academy Awards, Golden Globes, BAFTA, Cannes, Venice, Berlin, SAG, Critics
Choice, Sundance), store all nominations per movie.

### ~~Country of origin: Postgres-only or also Qdrant payload?~~ DECIDED

Postgres-only. The sequential dependency (Postgres fetch before Qdrant) only
applies to queries that filter by country — a small fraction. The round-trip
adds ~10-20ms, negligible vs vector search time. Can add to Qdrant later if
latency becomes a real problem.

---

## Completed Tests — Results Summary

Tests originally defined pre-implementation. Results captured here; detailed
analysis in current_search_flaws.md #13-15.

### ~~Test 1: Subquery quality inspection~~ COMPLETED

**Results for "iconic twist ending":**
- Channel weights: vector=large, lexical=not_relevant, metadata=small
- Metadata activated: prefers_popular_movies=True
- Space weights: narrative_techniques=0.3571, reception=0.2381,
  plot_analysis=0.1190, watch_context=0 (empty)
- watch_context had the most twist-related content but wasn't queried at all
- reception got 23.8% weight despite being less twist-specific

**Conclusion:** Vector space routing is independently broken (#15). The system
doesn't understand which spaces contain relevant signal for a given concept.

### ~~Test 2: Threshold + flatten simulation~~ COMPLETED

**Result:** Taking top-2000 candidates with no thresholding, sorting by
popularity → Fight Club and Sixth Sense both appear in the top 10.

**Conclusion:** Core architectural hypothesis CONFIRMED. Flatten + quality
re-ranking produces the expected results. The bottleneck is getting movies
into the candidate pool, not the scoring math.

**Additional finding:** Elbow-based thresholding would have excluded The Sixth
Sense. Relative-to-top at 80% would barely include it. This reinforces that
semantic deal-breakers should be applied as rescore filters (where the
candidate pool is pre-filtered by reliable channels) rather than as
candidate generators (where the threshold must be generous enough to catch
edge cases from the full collection).

### ~~Test 3: Vector score distribution analysis~~ COMPLETED

**Results:** See "How to set the deal-breaker threshold?" above for full
empirical data table. Key findings:
- Elbows at 100-200 entries, 75-90% of max depending on concept type
- Christmas is the only bimodal distribution (binary membership concept)
- Single fixed threshold percentage won't work across attribute types
- Threshold selection is less critical now that semantic deal-breakers
  move to Phase 2 rescoring rather than Phase 1 retrieval

### ~~Test 4: Cross-channel intersection sizing~~ COMPLETED

**Results:**
- "Dark gritty marvel movies": Winter Soldier missing from vector results
  entirely. Intersection misses key movies.
- "Funny horror movies": ZERO intersection between funny top-1000 and
  horror top-1000.
- "Tom Cruise 80s action": Not testable (no metadata-based retrieval yet).

**Conclusion:** Semantic concepts CANNOT reliably generate candidates for
cross-channel intersection. See current_search_flaws.md #14. This is the
most consequential finding — it forces a fundamental revision of Phase 1.

### ~~Test 5: Dynamic quality prior impact~~ COMPLETED

**Result:** "Silly comedies" vs "silly comedies everyone knows about" produces
much better results with the popularity boost (extra metadata activation of
prefers_popular_movies).

**Conclusion:** Quality prior design CONFIRMED as valuable. The default
quality composite (reception + popularity) is validated in principle.

### ~~Test 6: Soft constraint decay~~ SKIPPED

Not worth testing — behavior is predictable from first principles. The
three-tier constraint strictness model is sound; calibrating exact decay
curves is an implementation detail.

### ~~Test 7: Embedding density measurement~~ COMPLETED (reframed)

**Result:** Massive score gaps between movies in the top 10% that then
flatten out. The gap isn't simply density-saturated vs multi-dimensional —
it's that multi-dimensional movies get poorly represented by the flat-list
embedding format, muting their individual signals.

**Conclusion:** Embedding density is a symptom of the embedding format
problem, not a separate root cause. See current_search_flaws.md #13.
The structured-label embedding hypothesis is the highest-priority fix.

---

## Outstanding Tests (New)

Follow-up tests identified from the first round of empirical results.

### Test A: Structured-label embedding comparison

**The single highest-leverage test.** Compare retrieval quality between:
1. Current flat-list embedding: "plot twist / reversal, planted-foreshadowing
   clues, shocking twist, ..."
2. Structured-label embedding: "information_control: plot twist / reversal,
   planted-foreshadowing clues\nending_aftertaste: shocking twist\n..."

For the same set of movies and queries, measure:
- Where do known-relevant movies rank under each format?
- Does the structured format preserve per-attribute signal for
  multi-dimensional movies?
- Does searching with a matching structured query outperform flat text?

**Blocked on:** Deciding the exact structured format. Can be tested on a
small sample (10-20 movies) without full re-ingestion.

### Test B: Cross-space rescoring latency

Phase 2 now requires fetching stored vectors for candidates and computing
cosine similarity against query embeddings. Measure actual latency for:
- 500 candidates × 2 spaces (typical entity + semantic query)
- 1000 candidates × 3 spaces (complex multi-deal-breaker query)
- 2000 candidates × 4 spaces (worst case)

Qdrant `retrieve()` batched lookups + numpy cosine similarity. Need to
verify this stays under ~100ms to avoid dominating search latency.

### Test C: Metadata-anchored retrieval quality

For queries where deterministic channels generate candidates and semantic
concepts rescore:
- "Funny horror movies": retrieve all horror movies via genre filter,
  rescore on "funny" via cross-space rescoring. Do known funny-horror
  movies (Shaun of the Dead, Tucker and Dale, Cabin in the Woods)
  surface in the top 10?
- "Dark gritty Marvel movies": retrieve Marvel via lexical, rescore on
  "dark and gritty." Does Winter Soldier surface?

This directly validates the revised Phase 1→Phase 2 flow.
