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

### How to handle the "spectrum deal-breaker" problem? — PARTIALLY ADDRESSED

Deal-breakers that are spectrums create a classification problem. "Twist ending"
exists on a spectrum — does Shutter Island have a "twist ending" or just "a reveal"?

**Partially addressed:** Keywords and concept tags turn many spectrum questions
into simpler binary existence checks on hard data, avoiding the vector threshold
problem entirely. For example, "twist ending" can be matched via concept tags
rather than relying on vector similarity scores.

**Still open:** Purely semantic criteria not covered by deterministic searching
(e.g., "cozy date night") remain spectrum problems that need vector-based
thresholds. The threshold selection question (above) still applies for these
cases. To be revisited once search v2 is running and we can measure how many
queries fall into this purely-semantic bucket.

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
| Country of origin | Position-graded | GIN overlap is pass/fail, but scoring uses position gradient (pos 1 = 1.0, pos 2 ≈ 0.7-0.8, pos 3+ = decay) |
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

**~~Open sub-question:~~ DECIDED** Candidates are generated individually per
deal-breaker — if a movie matches at least one, it enters contention. Movies
are then scored by the number of deal-breakers they satisfy and reranked by
preference data (exact scoring logic TBD). This means the best partial matches
surface first when no movie meets all deal-breakers. Presentation of which
constraints were relaxed is handled naturally by the tiered scoring.

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

See franchise_metadata_planning.md for the implemented
`movie_franchise_metadata` design and
"Franchise Search Flow" section for the full search strategy.

**Franchise definition:** Any recognizable intellectual property or brand from
any medium (film, video games, toys, books, comics, TV, etc.) where the movie
is an adaptation/extension of that IP. Examples: Mario, Barbie, Transformers,
Harry Potter, Marvel Cinematic Universe. The franchise name is the **IP name**,
not the film series name.

**Storage:** The finalized Postgres projection is `movie_franchise_metadata`
with `lineage`, `shared_universe`, `recognized_subgroups`,
`launched_subgroup`, `lineage_position`, `is_spinoff`, `is_crossover`, and
`launched_franchise`. Named identity fields are stored normalized.

**Search approach decisions:**

- **lineage/shared_universe:** Both the ingestion LLM and search extraction
  LLM are instructed to output the most common, fully expanded franchise/IP
  names — no abbreviations. Both fields feed the same
  `inv_franchise_postings` pathway, so lookup tolerates slot swaps and
  parent-universe cases like Shrek/Puss in Boots.
- **lineage_position:** Stored as an integer ordinal on
  `movie_franchise_metadata`. Trivially filterable with `WHERE lineage_position = $1`.
- **recognized_subgroups:** Stored normalized as a text array. Searched via
  trigram matching within the post-franchise-lookup result set. Since franchise
  lookup narrows to 3-30 movies, simple `similarity()` or `ILIKE` on subgroup
  labels is sufficient.

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

### ~~How should system-level priors split quality vs notability/mainstreamness?~~ DECIDED

Both quality and notability are 4-value enums with identical value names but
independent semantics: `enhanced` (explicitly important), `standard` (implicit
default expectation), `inverted` (user wants the opposite — campy/bad for
quality, hidden/obscure for notability), `suppressed` (another preference
dominates reranking, so this prior contributes minimally).

`suppressed` is unique in that it's a second-order inference — it depends on
whether a dominant primary preference exists in the decomposition, not on the
query text alone. The LLM must assess quality/notability priors *after*
generating dealbreakers and preferences (last fields in the output).

See finalized_search_proposal.md "Output Structure: Quality / Notability
Priors" for the full specification.

### ~~How should metadata filters handle implicit temporal signals?~~ DECIDED

Temporal bias is NOT a separate system-level field. It is handled as a metadata
preference routed to the metadata endpoint, where the step 3 LLM translates
the soft intent into concrete date parameters with grace periods and decay.

- "Classic movies," "old movies" → metadata preference: "prefer older films" →
  step 3 translates to BEFORE with a date and wide grace period for boundary
  decay. Movies in the target range get a uniform boost; movies near the
  boundary get a decaying boost.
- "Recent movies" → metadata preference: "prefer recent films" → step 3
  translates to AFTER with a recent date.
- "Iconic," "legendary" → NOT temporal signals. These are quality/notability
  signals (quality: enhanced, notability: enhanced) or metadata preferences
  on reception_score.
- "Instant classic" → quality + recency combination, handled by quality prior
  (enhanced) plus a "prefer recent" metadata preference.

### ~~When should multi-interpretation branching trigger?~~ DECIDED

Branching is cross-flow — a single query can produce interpretations routed
to different major flows (e.g., exact title vs. standard flow). Triggers when
multiple interpretations are reasonably similar in likelihood to an intelligent
reader. The primary case is movie titles that double as natural language
descriptions: "Scary Movie" (the film vs. scary movies), "Date Night" (the
film vs. date night movies), "Love Story", "Not Another Teen Movie." If one
interpretation is clearly dominant ("Frozen", "Her", "Cars"), don't branch.
Cap at 3 interpretations. Within standard flow, multi-reference queries
("movies like X and Y") and qualified similarity ("movies like X but Y") are
also branching candidates when different trait extractions are equally
reasonable.

### ~~What should the exact major-flow routing triggers be in Step 1?~~ DECIDED

**Exact title flow:** User provides the literal movie title. Includes
misspellings, partial titles where the user is clearly attempting the title,
alternate official titles, and recognized single-movie abbreviations. Franchise
acronyms (LOTR, HP) go to standard flow. If the user explicitly states they're
searching by title, route here even if the title is unrecognized.
Descriptions of movies (plot, scenes, cast) always go to standard flow — even
if the movie is easily identifiable — because the standard pipeline handles
description-based identification better than a small routing LLM guessing
titles. If the title isn't found in the DB, the user sees "we don't have that
title" with no fallback.

**Reference-movie similarity flow:** User names a specific movie and asks for
similar movies with zero qualifiers. Anything beyond "similar to X" / "like X"
/ "X style movies" = standard flow. Multiple reference movies = standard flow.

**Standard flow:** Everything else.

### ~~How should Step 1 handle multiple `is_primary_preference=true` outputs?~~ DECIDED

Treat all marked preferences as **co-primary** — elevated equally in weight,
with no single axis dominating. Multiple primaries form a co-primary group that
collectively dominates within-tier ranking. This avoids arbitrary ordering-based
selection and preserves the signal that all flagged preferences are important.

### Should full boolean/group logic stay deferred unless query failures justify it?

The current V1 decision is to avoid explicit boolean clause/group logic and rely
on synonym consolidation plus strict match-count tiering. This keeps the system
simpler because many OR-style queries degrade gracefully under the current
structure.

**Open question:** After real-query evaluation, does this simplification hold up
well enough, or do OR/group-style failures become common enough to justify the
extra complexity later?

---

## Data Layer Questions

### ~~What's the future of the production vector space?~~ DECIDED

**Decided:** Regenerated with tightened definition — filming locations (WHERE) +
production technique keywords (HOW). The vector now handles queries that are too
broad or variable for keyword matching, such as "movies filmed in Ireland" or
"first person perspective filming." The tightened scope justifies keeping it as
a dedicated vector space.

### ~~How should multi-vector scores combine for a single semantic concept?~~ SUPERSEDED

Originally decided as max-across-spaces. Retired by the finalized design:
dealbreakers pick exactly 1 space (no combining needed), and preferences
use weighted-sum cosine across selected spaces with categorical weights
(`central`=2, `supporting`=1). See `finalized_search_proposal.md`
Endpoint 6 and the Semantic Endpoint Finalized Decisions section.

### ~~How should keyword vocabulary mapping work?~~ DECIDED

**Answer: Pure static mapping.** The keyword vocabulary audit
([keyword_vocabulary_audit.md](keyword_vocabulary_audit.md)) revealed that
`overall_keywords` is a compact curated taxonomy of exactly 225 terms
— not the free-form community tagging system we assumed. 100% coverage,
near-zero long tail, trivially enumerable.

**Mapping approach (current V2 design):** Step 2 receives the canonical
concept-family taxonomy for the keyword endpoint. Step 3 is an LLM call that
selects exactly one entry from a unified classification registry merging
`OverallKeyword` (225) + `SourceMaterialType` (10) + `ConceptTag` (25) into
259 members. `Genre` / `genre_ids` is excluded from this endpoint entirely
because all 27 TMDB genres are already members of `OverallKeyword` with
identical labels — the keyword column alone is sufficient, no dual-backing
into `genre_ids`. `OverallKeyword` wins any name collision (only `BIOGRAPHY`
in current vocabulary). See `schemas/unified_classification.py` and
`finalized_search_proposal.md` §Endpoint 5 "Unified Classification Registry"
for the full design.

**`plot_keywords` excluded:** The 114K-term `plot_keywords` vocabulary is
already consumed by the metadata generation pipeline and distilled into
structured LLM metadata. No incremental value in indexing raw plot_keywords
separately.

See [keyword_vocabulary_audit.md](keyword_vocabulary_audit.md) for the full
report including concept→keyword mappings across all deal-breaker categories.

### ~~Step 3 keyword endpoint: LLM call or deterministic?~~ DECIDED

**Answer: LLM call.** Step 2 does not carry full per-entry definitions for
the 259-term unified registry, which are required to disambiguate close
members (e.g., `FEEL_GOOD_ROMANCE` keyword vs. `FEEL_GOOD` concept tag,
`TRUE_STORY` vs. `BIOGRAPHY`). The step 3 LLM receives the registry grouped
by canonical family with each entry's definition and selects the single
best fit. It cannot abstain — routing already happened, so the endpoint
always picks the best available member even when the match is imperfect.

### ~~Step 3 keyword: single ID or multi-store resolution?~~ DECIDED

**Answer: Single ID, single column.** The LLM picks exactly one registry
entry. The entry's `source` determines which `movie_card` array column to
query (`keyword_ids` / `source_material_type_ids` / `concept_tag_ids`), and
execution issues a single GIN `&&` overlap against that one column with the
one `source_id`. No cross-column union, no dual-backing into `genre_ids` or
any other column — one chosen classification, one search.

### ~~Step 3 keyword: candidate pool cap?~~ DECIDED

**Answer: No limit.** Matches the entity and franchise endpoint decisions.
Downstream scoring handles broad result sets (e.g., the `DRAMA` keyword can
return 50K+ candidates — the scoring formula and preferences narrow from
there).

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

## Retrieval & Scoring Questions (New)

### ~~Should retrieval depth vary based on deal-breaker specificity?~~ DECIDED

Always include some level of buffer in candidate generation and trust the full
pipeline (scoring, preferences, priors) to narrow down to the right movies.
No explicit breadth signal per dealbreaker is needed — the pipeline's multi-stage
scoring handles broad vs. narrow dealbreakers naturally.

### ~~How should preferences interact with deal-breakers from the same attribute?~~ DECIDED

Resolved via the thematic centrality principle. When step 2 emits a keyword or
concept tag dealbreaker for a **thematic** concept (zombie, heist, Christmas,
coming-of-age), it should also include that concept's centrality in the grouped
semantic preference description. Thematic concepts have centrality spectrums —
"how central is Christmas to this movie?" matters for ranking within the passing
set. Structural concepts (sequel, award-winning, based on a true story) do not
have meaningful centrality spectrums and don't need dual emission.

This is a guiding principle for the step 2 LLM, not an automatic system
behavior. The prompt must ensure this instruction doesn't conflict with the
semantic preference grouping rules. See finalized_search_proposal.md Endpoint 5
for the full specification.

### ~~How do we handle pure vibes-based deal-breakers?~~ DECIDED

Resolved via the four-scenario execution model in
`finalized_search_proposal.md` (see Endpoint 6 → Execution Scenarios and
Pure-Vibe Flow). Summary:

- **D2 (pure-vibe dealbreaker flow):** When every inclusion dealbreaker
  routes to `semantic` (≥1 semantic inclusion, zero non-semantic
  inclusion), each semantic dealbreaker independently generates top-N
  per (dealbreaker, space); union = pool. Scored with global-elbow
  calibration; contributes to `dealbreaker_sum`.
- **P2 (preference-driven retrieval):** When zero inclusion dealbreakers
  exist but a semantic preference does, the preference generates
  candidates (top-N per selected space, union). Scored as a preference
  (raw weighted-sum cosine); contributes to `preference_contribution`.
- **Exclusion-only (neither dealbreakers nor preferences):** browse
  fallback via the default quality composite; no vector candidate
  generation at all.

Proxy-deterministic mapping was considered and rejected as unnecessary
complexity given the four-scenario model handles it cleanly.

### ~~Exclusion-only queries: what generates candidates?~~ DECIDED

When step 2 emits zero inclusion dealbreakers AND zero preferences —
only exclusion dealbreakers ("movies not starring Tom Cruise," "movies
not about clowns") — route to the browse fallback:

1. Generate candidates as top-K by the default quality composite
   (`0.6 × reception_score + 0.4 × popularity_score`).
2. Apply exclusions normally in Phase 4b.
3. Rank by the same composite plus any active priors.

**Rationale:** When the user expressed no positive intent at all, the
quality composite is the only honest candidate-generation signal. The
browse fallback mirrors what a librarian would do when asked "a movie,
but not that one" — return well-known, well-regarded movies minus the
excluded set. Options considered and rejected: inverse-generating via
the exclusion itself ("not clowns" has no positive signal to search
against), asking the user for clarification (breaks the pipeline's
always-return-something contract).

**Note:** If preferences exist alongside exclusions ("not clowns,
something cozy"), use scenario P2 instead — the semantic preference
drives candidate generation, and exclusions apply as hard filters
(deterministic) or penalties (semantic) afterward.

### ~~No-dealbreaker preference-only queries: what generates candidates?~~ DECIDED

**Option B: preferences generate candidates.** When step 2 emits zero
inclusion dealbreakers but at least one semantic preference, the
preference takes on candidate generation (scenario **P2** in the
Execution Scenarios table). Each selected space runs top-N against the
full corpus; union = candidate pool.

**Scoring stays as preferences, not dealbreakers.** Raw weighted-sum
cosine normalized by Σw (same as P1 where preferences score a
pre-built pool), contributing to `preference_contribution` scaled by
P_CAP. `dealbreaker_sum = 0`.

**Rationale:**
- Step 2 classified these items as preferences deliberately. The
  candidate-generation mechanism is orthogonal to the final-ranking
  role — who produces candidate IDs is not a reason to change what the
  items mean.
- Scoring stays consistent with P1: same function, same normalization,
  regardless of whether the preference generated candidates or scored
  a pre-built pool.
- The alternative (Option A: browse fallback on quality composite for
  any zero-dealbreaker query) throws away the user's positive intent.
  "Cozy date night movie" with every item classified as a preference
  should still do vector retrieval for "cozy," not degenerate into a
  popularity-ranked browse.
- Final scores are bounded above by P_CAP in P2 (since `dealbreaker_sum
  = 0`), but this is fine — within-query ranking is what matters, and
  cross-query score comparability is not a goal of the system.

### ~~Is there a better model than t-shirt sizing for relative vector space weighting?~~ DECIDED

**Two-level categorical scale, `minor` option removed.** Preferences select
1+ spaces and assign each one `central` (maps to 2) or `supporting` (maps
to 1). There is no `minor` / `not_relevant` weight — if a space's signal
isn't at least supporting meaningfully, don't select it.

**Reasoning:**
- Small models handle categorical Likert scales better than free-form
  numerics or point-budget allocations (they collapse to round numbers or
  uniform splits under uncertainty).
- Dropping `minor` prevents weight dilution across many marginal spaces and
  removes the model's ability to hedge by tagging everything as tangentially
  relevant.
- A `not_relevant` option is redundant with not selecting the space at all —
  creates a decision-theoretic ambiguity and adds no signal.
- With one query per selected space (not per concept) and 1–3 spaces typical
  per preference, two levels is plenty of expressiveness.

Revisit only if evaluation shows the model systematically collapsing to one
level (everything `central` or everything `supporting`).

### ~~What does "best" mean — critically acclaimed, popular, or both?~~ DECIDED

**"Best" maps to `quality_prior: enhanced`, not a separate mechanism.** When
step 2 sees "best", it sets `quality_prior: enhanced` (weight 1.5 on the
`reception_score + popularity_score` composite). This means "best" doesn't
need special handling in the metadata endpoint — it's already handled by the
prior system in step 4 scoring.

Edge case: "best and scariest horror movies" — the quality prior (enhanced)
and the superlative preference (scariest = primary) naturally coexist since
"best" is a prior signal, not a preference. Confirmed via testing that
this combination works correctly.

### Reception vector embedding: scraped vs generated summary?

When embedding the reception vector, should we default to the scraped reception
summary (from IMDB reviews/critic data) over our own LLM-generated reception
summary?

**Tradeoffs:**
- Scraped data is grounded in actual critic/audience language and may better
  match how users search for reception-related concepts
- LLM-generated summaries are more consistently structured and use our controlled
  vocabulary, which aligns better with the structured-label embedding format
- Could use both: LLM-generated for structured labels, scraped for supplementary
  natural-language signal

### ~~What's the most efficient way to generate metadata structures at search time?~~ DECIDED

**One query per selected space, absorbing all concepts routed to that
space.** Not per-concept-per-space (which would reintroduce retrieval
averaging). The LLM composes the combined query in that space's native
vocabulary — "scary but funny" routed to `viewer_experience` becomes
`emotional_palette: darkly funny, gallows humor` +
`tension_adrenaline: unsettling, creeping dread` rather than two separate
queries unioned.

**Source of truth for the per-space shape.** Any vector space whose
embedded text is assembled from more than one metadata object (anchor,
plot_events, production, reception at minimum) gets a dedicated Pydantic
model defining the embedded shape. These models serve both ingest-side
embedding text generation and search-side query generation — keeping the
two in lockstep structurally. Spaces whose embedded text maps 1:1 to an
existing generator output (`plot_analysis`, `viewer_experience`,
`watch_context`, `narrative_techniques`) reuse those models.

**Space-identification and query generation** happen inside a single step
3 semantic LLM call per item (standard flow) or one batched call for all
items (pure-vibe flow). No separate Phase 0 metadata-generation step.

### Which pipeline failure points should be fatal vs graceful?

The search pipeline has multiple stages. Need to identify:

**Fatal failures (should fail the whole query):**
- Phase 0 query understanding fails entirely (no structured output)
- Database connection failures

**Graceful failures (return empty candidate set for that endpoint):**
- One vector space query times out → score without that space
- Cross-space rescoring fails for one concept → score as if concept not evaluated
- Keyword matching returns zero results → fall back to other channels
- Trending data unavailable → skip trending injection
- Any endpoint failure → return empty candidate set, handled the same as
  finding no matches

**Decided:** On endpoint failure, return an empty candidate set so it gets
handled the same way as finding no matches. Retry once for transient issues
(network timeouts, temporary DB unavailability) before returning empty.

### What keywords are available and useful beyond current metadata filters?

**Investigation needed:** Audit the full set of available keywords across the
movie dataset to identify which ones could serve as useful deterministic signals
that aren't already covered by metadata filters, concept tags, or the
overall_keywords enum.

This may uncover additional deal-breaker concepts that can be moved from the
semantic (vector) side to the deterministic side, improving retrieval reliability
per the core architectural principle.

---

## V2 Pipeline Questions (from finalized proposal)

### How should elbow detection work for semantic exclusion thresholds?

Semantic exclusions (e.g., "not ones with clowns") use an elbow-threshold
penalty: search the full corpus for the exclusion concept, find the elbow in
the score distribution, and penalize candidates relative to it. The elbow must be
determined dynamically per concept since different concepts have different
distributions (see threshold calibration data above — "twist ending" elbows at
75% of max, "christmas" at 90%).

**Constraints:**
- Hard-coded percentage of max won't work across concept types
- Need a method that handles both tight clusters (few movies genuinely match)
  and broad distributions (many movies partially match)
- Must distinguish "above elbow" (harsh downrank) from "near elbow" (soft
  downrank) from "well below" (no penalty)
- Need to determine the penalty curve shape, not just the elbow location
- Need to determine how far below the elbow the penalty should decay to zero

**Needs empirical testing** before full implementation. Test with concepts of
varying specificity (clowns vs violence vs Christmas) to see if a single
detection method generalizes.

### What should the multi-interpretation trigger criteria be?

Step 1 can produce multiple interpretations for ambiguous queries. When should
it trigger? Candidates:
- LLM confidence score on primary interpretation falls below a threshold
- Query contains genuinely ambiguous terms ("dark knight" → the movie vs dark +
  knight themed)
- Multiple plausible dealbreaker decompositions exist

**Related opportunity:** Multi-interpretation could also handle broad-vs-narrow
tension. For "dark gritty marvel movies," one interpretation treats all three as
dealbreakers (strict) and another treats marvel as dealbreaker with dark+gritty
as preferences (relaxed). This gives users strict results AND a broader fallback
without separate fallback logic. Worth exploring post-MVP.

### Should step 1 display when a semantic dealbreaker is score-only?

When a dealbreaker is routed to `semantic` in a deterministically anchored
query, the user's stated requirement is being enforced via scoring rather than
candidate generation. If results are good, users won't notice. But if a
mid-ranked movie only partially satisfies the semantic requirement, users
might be confused.

Options:
- Don't surface it — let result quality speak for itself
- Subtly indicate in the display phrase (e.g., "horror movies, ranked by car
  chase relevance" vs "horror car chase movies")
- Show a tooltip/indicator when a requirement couldn't be hard-filtered

Likely a presentation concern to address during UI work, not an architectural
decision.

### ~~How should the step 2 vector LLM handle semantic exclusion queries?~~ DEPRIORITIZED

Exclusions are match-then-penalize, not hard-filter — a false positive
costs score weight, not removal. Originally flagged as higher-stakes than
inclusion query formulation; downgraded given the penalty-based scoring.
Per the direction-agnostic framing principle, the step 3 LLM doesn't even
know the query's direction; it always formulates "find movies WITH X" and
the orchestrator applies the exclusion multiplier. Revisit only if
evaluation shows systemically unfair exclusion penalties.

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
