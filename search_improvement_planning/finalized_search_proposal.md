# Finalized Search Proposal (V2)

This document contains only finalized, committed-to decisions for the V2 search
system. Decisions are promoted here from the broader planning docs once fully
resolved.

---

## High-Level Architecture: Three-Step Pipeline

The V2 search system follows a three-step pipeline:

1. **Query Understanding & Flow Routing** — A single LLM extracts the user's
   intentions in concrete terms and first routes the query into the correct
   major search flow before any standard decomposition happens.
2. **Search Planning** — Per-source LLMs receive relevant portions of the
   decomposition and translate them into concrete, executable search parameters
   for their respective data source.
3. **Execution & Assembly** — Candidate generation queries run first, then
   preference/reranking queries run on the deduped candidate set. Results are
   tiered by dealbreaker conformance and sorted within tiers by preferences.

---

## Step 1: Query Understanding

A single LLM call that does all interpretive work upfront. Downstream steps
receive resolved, concrete instructions — they never re-assess what the query
means.

### Major Flow Routing

Before standard dealbreaker/preference decomposition, step 1 first classifies
the query into one of the major search flows:

- **Known movie / exact title flow** — User is clearly trying to find one
  specific movie. The system should identify it directly and then optionally
  show movies like it.
- **Reference-movie similarity flow** — User is asking for "movies like X"
  with similarity as the primary task.
- **Standard flow** — Everything else. This includes the main constrained
  search pipeline described below.

**"Movies like X but qualifiers" stays in the standard flow, not the pure
similarity flow.** Once explicit qualifiers are present, the query is no
longer just "find nearest neighbors to X." It becomes a standard interpreted
search, where the LLM can use its parametric knowledge of the reference movie
as a fast way to understand the intended traits.

### Preprocessing Chain

For the standard flow, the LLM follows a structured reasoning chain to produce
its output:

1. **Rewrite the query** in its full concrete intentions. Make implicit
   expectations explicit where appropriate (e.g., "dicaprio comedies" →
   "movies starring Leonardo DiCaprio in the comedy genre"). This rewrite
   captures user intent only — system defaults like quality bias are NOT
   baked into the rewrite.
2. **Write a single brief phrase** summarizing the hard dealbreakers the user
   is asking for. This phrase is used for display in the app so users can
   easily see what interpretations were extracted.
3. **Generate a structured list** of individual dealbreakers and preferences
   with routing and inclusion/exclusion flags (see output structure below).

The LLM should consolidate synonymous or clearly related concepts into a single
dealbreaker/preference rather than splitting them into separate items. For
example, "fun and lighthearted" that both target the same vector space should
be a single preference, not two.

**V1 intentionally does NOT model full boolean/group logic.** Explicit OR-style
clause handling would improve a subset of edge cases, but the added complexity
is not justified yet because simple match-count tiering already degrades
gracefully for most such queries. The grouping rule for V1 is narrower:

- **Do group** near-synonymous concepts when separating them adds no retrieval
  or scoring value.
- **Do not group** distinct entities or constraints merely because they share a
  route. "Brad Pitt" and "Tom Hanks" remain separate dealbreakers even though
  both route to `lexical`.

### Multi-Interpretation Support

When a query is potentially ambiguous, step 1 can extract multiple
interpretations. Each interpretation is a complete, independent decomposition
that flows through steps 2 and 3 separately. Users see a brief phrase per
interpretation and can click into the one that best fits their intent.

Each interpretation contains:
- The rewritten query in concrete terms
- A brief display phrase (for UI selection)
- A list of dealbreakers (see below)
- A list of preferences (see below)
- Quality prior settings (see below)

**Open question (deferred):** Exact trigger criteria for when the LLM should
produce multiple interpretations vs one. Also worth considering: using
multi-interpretation to handle the broad-vs-narrow tension (e.g., for "dark
gritty marvel movies," one strict interpretation with all three as dealbreakers
and one relaxed interpretation with marvel as dealbreaker and dark+gritty as
preferences).

### Output Structure: Dealbreakers

Dealbreakers represent the foundational attributes around which the rest of the
query revolves. They are the criteria used for candidate generation — movies
that don't meet these are excluded or tiered down.

Each dealbreaker has:
- **Description** — A concrete string describing the requirement (e.g., "is a
  rocky movie", "does not have a fight with a russian")
- **Routing** — An enum value indicating which data source handles this
  dealbreaker. One of: `lexical`, `metadata`, `keyword`, `semantic`
- **Direction** — Whether this is an `inclusion` (must have) or `exclusion`
  (must not have)

**Routing enum definitions (surface-level, no schema details):**

| Route | What it covers | LLM needs to know |
|-------|---------------|-------------------|
| `lexical` | Named entities: actors, directors, franchises, characters, studios | Entity types available |
| `metadata` | Structured attributes: genre, year, runtime, rating, streaming, country, source material type, box office | Field names (not enum values) |
| `keyword` | Concept tags and content keywords from curated vocabulary | The full 225-term keyword vocabulary + 27 concept tags (included in prompt) |
| `semantic` | Subjective qualities, vibes, thematic concepts not covered by other sources | What the other sources DON'T cover |

**Critical:** The LLM must understand the limitations of each source. It should
know the keyword/concept tag vocabulary so it can make informed routing decisions
rather than guessing that a concept like "clowns" might be a keyword when it
isn't. When no deterministic source cleanly covers a concept, route to `semantic`.

**One dealbreaker per route, but multiple dealbreakers per query may share a
route.** For example, "Leonardo DiCaprio Rocky movies" produces two `lexical`
dealbreakers. Each dealbreaker is executed as an independent search and produces
its own candidate set.

### Output Structure: Preferences

Preferences are qualities used to evaluate and rerank candidates generated by
dealbreakers. They do not generate candidates — they only influence ordering.

Each preference has:
- **Description** — A concrete string describing the quality (e.g., "dark",
  "gritty", "funny")
- **Routing** — Same enum as dealbreakers. Determines which scoring mechanism
  evaluates this preference.
- **Direction** — `positive` (boost matches) or `negative` (downrank matches,
  e.g., "not too serious")
- **is_primary_preference** — Optional boolean. Marks that this preference is
  the dominant ranking axis rather than one equal member of a balanced set.

Negative preferences are treated as gradient reranking signals, not hard
exclusions. "Not serious" means movies that are more serious get ranked lower,
but aren't removed — the threshold is debatable and gradient rather than binary.

**Rationale for `is_primary_preference`:** This is the smallest useful addition
that lets the system distinguish between:

- **Balanced additive preferences** — "dark and gritty" where multiple
  preferences should contribute equally
- **One dominant ranking axis** — "scariest movie ever" where one preference
  should drive ordering

If no preference is marked primary, preferences are treated as equal-weighted
relative to each other aside from the separate system-level priors. V1 does not
introduce general per-preference weights.

### Output Structure: Quality / Notability Priors

A separate, explicit system-level adjustment controls how much the system
biases toward well-known, well-received movies. This is NOT a preference and
NOT baked into the query rewrite.

**Revised decision:** Conventional quality and mainstream/notability are not
the same thing and should not be conflated in the design docs. The earlier
draft treated queries like "hidden gems" and "underrated" as inverted quality,
but that collapses two distinct ideas:

- "so bad it's good" really does invert conventional quality
- "hidden gems" and "underrated" usually still want quality, but paired with
  lower popularity or lower mainstream exposure

The exact wire shape for splitting these dimensions is still open and is tracked
in `open_questions.md`. The finalized design decision here is only the
conceptual correction: **quality and notability/mainstreamness must be modeled
as separate levers.**

**Superlative interaction:** When a query has a strong superlative preference
(e.g., "scariest movie ever"), that primary preference becomes the dominant
ranking axis. System-level quality/notability priors remain secondary.

### What Step 1 Does NOT Do

- Does not know schema details (table names, column types, enum values)
- Does not determine exact search parameters (that's step 2's job)
- Does not determine vector space routing (that's step 2's vector LLM)
- Does not inject system defaults into the query rewrite

---

## Step 2: Search Planning (Per-Source LLMs)

Each data source has its own LLM that receives:
- The full rewritten query (for context)
- Only the dealbreakers and preferences routed to its domain
- Deep knowledge of its own schema

Each per-source LLM translates the abstract dealbreaker/preference descriptions
into concrete, executable search parameters. These are narrow, well-scoped tasks
suitable for small, fast models.

**Why keep per-source LLMs instead of pushing all schema-specific work into step
1?** Because exact enum values, metadata matching nuance, keyword definitions,
lexical role nuance, actor prominence cues, and vector-space behavior are too
much specialized low-level knowledge for one smaller interpretive LLM to carry
without quality loss. The split is:

- **Step 1** interprets user intent, consolidates concepts, and routes each item
  to the correct source.
- **Step 2** translates already-interpreted intent into source-specific
  executable parameters.

Step 2 LLMs are **schema translators, not re-interpreters**. They should not
decide what the user meant; they should only decide how their source should
execute the already-resolved intent.

**Lexical LLM:** Knows posting table schemas (actor, director, franchise, etc.).
Determines exact entity searches — which posting tables to query, what string
values to match. Produces one independent search per dealbreaker routed to it.

**Metadata LLM:** Knows movie_card columns, enum values, and filter types.
Determines exact SQL filter parameters — which columns, what values, what
comparison operators. Produces filter specifications per dealbreaker.

**Keyword LLM:** Knows the keyword and concept tag vocabularies. Determines
exact keyword/tag matches from the curated lists. May not be needed as a
separate LLM if step 1 already selects from the enumerated vocabulary — this
is an implementation detail.

**Vector LLM:** Knows vector space names, what each space captures, and query
formulation best practices. Determines which spaces to search, generates
expanded search queries, and handles both inclusion and exclusion semantic
operations. Also handles preference scoring setup for the execution phase.

**Trending / Redis source:** Standard flow can also inject a deterministic
candidate set from Redis for trending-oriented queries. This sits alongside
lexical, metadata, and keyword sources as another way to produce reliable
candidates before reranking.

All per-source LLMs run in parallel since they have no dependencies on each
other.

---

## Step 3: Execution & Assembly

### Phase 3a: Candidate Generation

Execute all candidate generation queries produced by step 2. Each dealbreaker
produces its own independent candidate set.

- **Deterministic dealbreakers** (lexical, metadata, keyword): Each search
  returns a set of movie IDs. These are binary — a movie is in the set or not.
- **Semantic dealbreakers are NOT used for candidate generation.** Any
  dealbreaker routed to `semantic` is demoted to a preference for scoring
  purposes (see "Semantic Dealbreaker Demotion" below).

Union and deduplicate across all inclusion dealbreaker candidate sets. Each
movie receives a count of how many inclusion dealbreaker sets it appeared in.
This count determines its tier.

**Tiering rule:** A movie that matches N inclusion dealbreakers can never rank
above a movie that matches N+1 inclusion dealbreakers, regardless of preference
scores. Tiers are strict partitions.

### Semantic Dealbreaker Demotion

**Rationale:** Empirical testing confirmed that semantic (vector) search is
unreliable as a candidate generator. The Sixth Sense scores only 82% of max
for "twist ending" in narrative_techniques — not appearing in the top 1000.
"Funny horror" has zero intersection between vector candidate sets. Semantic
search works well for scoring/ranking but poorly for generating complete
candidate sets.

**Rule:** When a dealbreaker is routed to `semantic`, it is automatically
treated as a high-weight preference rather than a candidate generator. It
does not produce a candidate set and does not count toward the tier denominator.
Instead, it influences ranking within tiers with elevated weight compared to
regular preferences.

**Pure-vibe queries:** When ALL dealbreakers are semantic (no deterministic
anchors exist), the query enters the pure-vibe flow (see below). No
dealbreakers generate candidates; instead, vector search becomes the candidate
generator as a special case.

### Phase 3b: Exclusion Handling

Exclusions are applied after candidate generation. They do NOT count toward
the tier denominator — tiers are based on inclusion match count only.

**Deterministic exclusions** (lexical, metadata, keyword): Hard filter. If a
movie matches the exclusion criteria, it is removed from the candidate set
entirely. These are binary and reliable — "not starring Arnold Schwarzenegger"
can be definitively evaluated.

**Semantic exclusions:** Cannot be hard-filtered because vector similarity
doesn't give binary results. Instead, semantic exclusions use a
global-elbow-calibrated penalty system:

1. Run a vector search against the full movie corpus for the exclusion concept
   (e.g., "clowns").
2. Analyze the score distribution to find the elbow — the point where
   similarity drops off sharply, separating movies that genuinely match the
   concept from those that are merely adjacent.
3. Apply penalties to candidates based on where they fall in the global
   distribution:
   - **Above the elbow:** High confidence this movie matches the excluded
     concept → harsh downrank
   - **Near the elbow:** Uncertain → softer downrank
   - **Well below the elbow:** No meaningful match to the concept → no penalty

**Why penalty-only instead of hard removal?** Even with global calibration,
semantic similarity is still too noisy in V1 to justify full exclusion. The
global distribution is still the right reference frame, but the action taken
should match the uncertainty of the signal: penalize heavily when confidence is
high, taper the penalty near the elbow, and apply nothing once similarity is
safely below the concept boundary.

**Why global distribution, not candidate-relative:** If none of the candidates
actually contain clowns, a candidate-relative approach would still penalize
whichever candidate is most "clown-adjacent" (maybe a circus-themed movie with
no actual clowns). The global search calibrates against what "actually has
clowns" looks like across the full corpus, preventing false penalties.

**Open question:** The exact elbow detection method and penalty curve need
empirical testing before full implementation. The elbow should be determined
dynamically per concept since different concepts have different similarity
distributions (see threshold calibration data in `open_questions.md`). A
hard-coded percentage of max score won't work across all concept types.

### Phase 3c: Preference Scoring & Final Ranking

Score all remaining candidates on preferences and system-level priors. The
combination works as:

1. **Primary sort: tier** (inclusion dealbreaker match count, descending)
2. **Secondary sort: preference + system-prior composite** (within each tier)

**Preference scoring by route type:**
- Deterministic preferences (metadata, keyword): Gradient scoring, not binary.
  "Under 100 minutes" at 101 minutes scores ~0.95, at 140 minutes scores much
  lower. Users are frequently imprecise with numeric constraints, so gradients
  prevent harsh cutoffs that miss obviously relevant results (e.g., "Leonardo
  DiCaprio boat movie from 2001" should not filter out Titanic at 1997).
- Semantic preferences: Vector similarity scores, possibly with diminishing
  returns curve.
- If a preference is marked `is_primary_preference=true`, it becomes the
  dominant within-tier ranking axis rather than one equal additive component.

**Tier assignment for metadata dealbreakers also uses gradients:** A generous
threshold determines pass/fail for tier purposes (e.g., 101 minutes "passes"
the "under 100 minutes" dealbreaker for tier credit), but the actual gradient
score is used for within-tier ranking.

**System-prior application:** Quality and notability/mainstreamness are applied
as separate within-tier signals once their final wire shape is decided. When a
strong primary preference exists, these priors remain secondary.

**Negative preferences:** Score inversely — high similarity to the negative
concept results in a lower preference score. Unlike negative dealbreakers
(exclusions), negative preferences don't remove candidates, they just push
them down in ranking.

---

## Pure-Vibe Flow (No Deterministic Anchors)

When step 1 produces no deterministic dealbreakers (all dealbreakers route to
`semantic`), the query enters a separate flow where vector search is the
candidate generator.

### How It Works

1. All semantic dealbreakers become preferences (there are no dealbreakers for
   tiering since there are no deterministic anchors).
2. The step 2 vector LLM determines relevant vector spaces and generates search
   queries. A single LLM call handles relative importance across spaces.
3. **Individual searches per concept** across relevant spaces. Synonymous
   concepts are already consolidated by step 1, so each search represents a
   distinct semantic axis. Union top-N results across all searches.
4. **Rescore each candidate** by fetching its distance in all relevant vector
   spaces (not just the space where it was initially retrieved). This catches
   movies that are near-misses in one space but strong in others.
5. Apply a **minimum similarity threshold per space** to avoid noise. A movie
   with 0.15 similarity to "lighthearted" shouldn't get credit for that score.
6. Combine scores across spaces and apply system-level priors.

### Why Individual Searches, Not Combined

"Fun" and "lighthearted" as separate searches preserve the independence of each
concept. Combining them into a single query blends the embeddings into an
average that may not match either concept well — recreating the signal dilution
problem at query time. The step 1 LLM already consolidates truly synonymous
concepts, so remaining distinct concepts should stay separate.

### Exclusions in Pure-Vibe Flow

Same mechanism as the standard flow: semantic exclusions use the
elbow-threshold penalty against the global distribution. The only difference
is that the candidate set comes from vector retrieval rather than deterministic
channels.

---

## Key Design Principles

### 1. Deterministic sources generate candidates; semantic sources score them

The most important architectural decision. Empirical testing proved semantic
search is unreliable for candidate generation but works well for ranking.
Deterministic channels (entity lookup, metadata filters, keyword matching)
produce reliable, complete candidate sets. Semantic similarity is applied
afterward to score and rank those candidates.

### 2. Each dealbreaker runs independently

A query can have multiple dealbreakers, and each produces its own candidate
set. Movies are tiered by how many dealbreaker sets they appeared in. This
means the system naturally handles partial matches — when no movie meets all
dealbreakers, the best partial matches surface first.

**Intentional simplification:** V1 does not add explicit boolean clause/group
logic on top of this. For the kinds of OR-style queries currently expected,
match-count tiering is considered good enough, and the extra logic is deferred
unless real query failures prove it necessary.

### 3. Exclusions are separate from inclusions

Inclusion dealbreakers generate candidates and determine tier placement.
Exclusion dealbreakers filter or penalize candidates after generation.
Exclusions do not count toward the tier denominator. A movie that passes 2/2
inclusions but fails an exclusion is handled differently than a movie that
passes 1/2 inclusions — the exclusion failure results in removal or harsh
penalty, not a lower tier.

### 4. Semantic dealbreakers demote to preferences

When a concept can only be evaluated via vector similarity (no keyword, entity,
or metadata coverage), it cannot reliably generate candidates. It is
automatically demoted to a high-weight preference that influences ranking
rather than candidate generation. When ALL dealbreakers are semantic, the
query enters the pure-vibe flow.

### 5. System-level priors are separate, explicit dimensions

Quality bias is not baked into the query rewrite or treated as a preference.
The design now explicitly recognizes that conventional quality and
notability/mainstreamness are distinct dimensions. The finalized decision is to
keep them separate conceptually; the exact field shape remains open.

### 6. Step 1 interprets intent; step 2 knows schemas

The interpretive LLM (step 1) needs surface-level awareness of data sources
(what each covers, keyword/concept tag vocabulary) but not schema details.
Per-source LLMs (step 2) need deep schema knowledge but receive pre-interpreted
intent. This split keeps each LLM's task tractable for smaller, faster models
without asking the step 1 model to also carry every exact enum, matching rule,
keyword definition, and low-level source-specific nuance.

### 7. Metadata constraints use gradients, not binary filters

NLP-extracted numeric and temporal constraints use gradient scoring rather than
hard cutoffs. This prevents missing obviously relevant results when users are
imprecise (which they frequently are). Tier assignment uses a generous threshold
for pass/fail; within-tier ranking uses the actual gradient score.

---

## Scoring Function Modes (from prior planning, unchanged)

Four scoring modes apply to different attribute types:

- **Threshold + flatten** — For dealbreakers. Similarity >= threshold → 1.0
  (passes), below → decay. Used for determining if a movie "has" an attribute.
- **Preserved similarity** — For superlatives. Raw similarity score is the
  ranking signal. "Scariest movie ever" needs to differentiate between
  "very scary" and "somewhat scary."
- **Diminishing returns** — For preferences. Marginal gains decrease as
  similarity increases. Being "somewhat funny" matters more than the difference
  between "very funny" and "extremely funny."
- **Sort-by** — For explicit ranking axes like "critically acclaimed" or
  "chronological." A structured signal (reception score, release date) is the
  primary sort key.

---

## Decisions Deferred to Implementation

- Exact step 1 prompt engineering (few-shot examples, chain-of-thought format)
- Exact elbow detection algorithm for semantic exclusion thresholds
- Step 2 prompt design per data source
- Specific candidate pool size limits per source type
- Exact gradient decay functions for metadata constraint scoring
- Tier assignment threshold calibration (how generous is "passing"?)
- Whether the keyword LLM is a separate call or folded into step 1
- Trim point between primary and exploratory results (currently decided as
  top 25 or 40% score floor — may need revision in V2 context)
- Multi-interpretation trigger criteria
