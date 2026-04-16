# New Search System Brainstorm

Architecture ideas, design principles, and features for the redesigned search system.

---

## Core Insight

The current system decomposes queries into **independent channel signals** and sums
them. The new system should decompose queries into a **dependency hierarchy** —
deal-breakers constrain the candidate set, preferences rank within it, and implicit
expectations are a universal prior.

**Additive scoring isn't the problem — applying it at the wrong layer is.** In the
current system, deal-breakers and preferences compete for weight in a single sum. In
the new structure, deal-breakers gate the candidate set (conjunctive), then
preferences rank within it (additive). The additive math is correct for ranking — it
just shouldn't make the pass/fail decision.

---

## Query Decomposition Framework

A user's search query is composed of three structural layers:

### 1. Deal-Breaker Attributes
Things the movie MUST have. If missing, the user sees the result as wrong.

**Characteristics:**
- High bar for inclusion — should clearly be non-negotiable from the query
- Include: named entities (actors, directors, franchises), explicit genre
  requirements, concrete structural attributes ("twist ending", "based on true story")
- The same attribute changes structural role depending on context:
  - "thriller with a twist" → thriller = deal-breaker, twist = preference
  - "iconic twist ending" → twist = deal-breaker, iconic = preference

### 2. Preferences (Qualifications)
How the user wants the deal-breaker results sorted or filtered. These are
qualifications ON the deal-breaker attributes.

**Characteristics:**
- "More of this = better" but absence isn't disqualifying
- Include: tonal qualities ("dark", "gritty"), reception signals ("iconic",
  "acclaimed"), experiential attributes ("funny", "intense")
- Additive scoring IS appropriate for combining multiple preferences

### 3. Implicit Expectations
Things the user didn't say but would be disappointed without.

**Characteristics:**
- Universal quality expectation: "comedies" really means "well-received comedies I'd enjoy"
- Mainstream accessibility bias (unless query signals otherwise)
- Language/availability assumptions
- Temporal establishment for certain language ("classics," "iconic," "legendary"
  imply cultural staying power, not just quality)

**Key insight: system-level priors should be dynamic, not constant.** Different
queries imply different strengths of quality expectation and
notability/mainstreamness expectation. These are related, but not identical,
and should not be collapsed into one concept.

Signals that push toward **strong quality / establishment expectations**:
- Superlatives ("best," "greatest," "iconic")
- Cultural reference language ("classics," "essential," "must-see")
- Social context ("something everyone's seen," "a crowd-pleaser")

Signals that push toward **lower mainstream/notability expectation**:
- Discovery language ("hidden gem," "underrated," "something I haven't seen")
- Niche descriptors ("experimental," "avant-garde," "slow cinema")
- Explicit novelty ("surprise me," "something different")

A user-facing toggle could also let users explicitly set their notability expectation
(well-known by default, discovery mode as opt-in), sidestepping the LLM inference
problem for users who know what they want.

---

## Proposed Pipeline Architecture

```
Phase 0: Query Understanding (restructured)
├── Articulate user intent in clear terms
├── Classify query type (similarity search vs all other)
├── For non-similarity: full decomposition
│   ├── Lexical entities (route to posting tables)
│   ├── Metadata filters with strictness tier (route to Postgres)
│   ├── Semantic deal-breakers with target spaces (route to Qdrant)
│   ├── Semantic preferences with target spaces
│   ├── Sorting criteria (null → default quality composite)
│   └── System-level priors (quality and notability/mainstreamness)
├── Resolve conflicts with any UI-set hard filters
│   (UI filters = hard tier, always override NLP-extracted)
└── Detect empty deal-breaker set → fall back to broad retrieval mode

Phase 1: Candidate Retrieval (deterministic channels generate, semantic rescores)
├── Apply hard metadata filters (UI-set) as strict WHERE clauses
├── Apply soft metadata filters (NLP-extracted) with generous gates
├── Lexical retrieval for entity deal-breakers (franchise, actor)
├── Keyword-based retrieval for high-coverage deal-breakers (christmas, animation)
├── Candidate pool = UNION of deterministic channel results
│   Semantic deal-breakers do NOT generate candidates — they rescore in Phase 2
│   (Empirical finding: semantic concepts cannot reliably generate candidates;
│   see current_search_flaws.md #14)
├── Exception: pure-vibe queries with no deterministic anchors still use
│   vector retrieval as the candidate generator (current system handles this OK)
└── Pool sizes of a few hundred to a few thousand candidates

Phase 2: Full Rescore (all candidates scored across all dimensions)
├── Cross-space rescoring: fetch stored vectors from Qdrant for candidates,
│   compute cosine similarity against query embeddings per semantic concept
│   (REQUIRED — not optional. Empirical testing showed candidates from
│   deterministic channels need semantic scoring in spaces where they
│   weren't retrieved. See open_questions.md cross-space rescoring.)
├── For each candidate, compute:
│   ├── Lexical match scores (entity deal-breakers met, prominence)
│   ├── Metadata proximity (soft constraint decay for NLP-extracted)
│   ├── Semantic deal-breaker scores per concept
│   │   (capped at 1.0 above threshold, decay below threshold)
│   │   Per concept: threshold each relevant vector space separately,
│   │   take best score across spaces
│   ├── Semantic preference scores (additive, with optional primary preference)
│   └── Sorting/quality score (explicit criteria or default composite)
├── Primary sort: deal-breaker conformance (% of deal-breakers met)
├── Secondary sort: preference + sorting scores (bounded — cannot
│   override deal-breaker conformance tier)
└── Deal-breakers contribute to BOTH candidate selection AND scoring;
    preferences contribute to scoring only

Phase 3: Result Assembly
├── Primary results: top of Phase 2 ranked output
├── Trim at top 25 or 40% score floor
└── If primary set is thin → trigger Phase 4

Phase 4: Exploratory Extension (conditional)
├── Run broader search (current-style expansive queries)
├── Exclude movies already in primary set
└── Append as "you might also like"
```

---

## Phase 0: Query Understanding Output Structure

Phase 0 is a single structured LLM call that produces the full query decomposition.
It **replaces** the current channel weight system entirely — deal-breakers are
deal-breakers regardless of which retrieval channel fulfills them, and the current
process of having the LLM determine channel weights (vector_relevance,
lexical_relevance, metadata_relevance) is eliminated.

### Phase 0 Processing Steps

1. **Intent articulation:** Identify and lay out the user's intentions in clear terms
2. **Query type classification:** Two top-level branches:
   - **Similarity search** ("movies like Inception") → routes to a separate
     deterministic process based on vector distance in selected spaces
   - **All other searches** → full decomposition below
3. **Full decomposition** (for non-similarity searches):
   - **Lexical entities:** Actors, directors, franchises, studios, title substrings.
     These route to lexical posting tables for retrieval.
   - **Deterministic metadata filters:** Genre, country, date/era, runtime,
     certification, streaming platform, source material type, box office bucket —
     anything that maps to a movie_card field or structured table. Each filter
     includes a **strictness tier** (see "Three-Tier Constraint Strictness" below).
   - **Semantic deal-breakers:** Concepts that must be present but require vector
     retrieval ("twist ending," "Christmas," "funny"). Threshold + flatten scoring.
   - **Semantic preferences:** Nice-to-have qualities for ranking within the
     qualifying set ("dark," "gritty," "intense"). Additive/diminishing-returns
     scoring.
   - **Reranking/sorting criteria:** Explicit ranking axes stated in the query
     ("critically acclaimed," "scariest," "top rated"). **Null if none stated** —
     triggers the default quality composite in Phase 2.
   - **System-level priors:** Separate quality and
     notability/mainstreamness adjustments. The exact wire shape is still open,
     but the conceptual split is now explicit.

### Future Direction: Expert LLMs Per Search Type

Rather than a single monolithic Phase 0 call, a future iteration could further
specialize the interpretive layer by search type. This is not because Phase 0
cannot reason about the query, but because source-specific low-level
specialization can be separated cleanly from high-level intent interpretation.
This is not planned for V2's initial implementation but is worth revisiting if
the current split still overloads the step 1 prompt.

### Phase 0 Output Example

```json
{
  "intent": "User wants well-known movies famous for their twist endings",
  "query_type": "standard",
  "lexical_entities": [],
  "metadata_filters": [],
  "semantic_deal_breakers": [
    {
      "concept": "twist ending",
      "target_spaces": ["narrative_techniques", "viewer_experience"]
    }
  ],
  "semantic_preferences": [
    {
      "concept": "iconic / culturally well-known",
      "target_spaces": ["reception"]
    }
  ],
  "sorting_criteria": null,
  "system_level_priors": "separate quality + notability settings"
}
```

### Default Quality Composite

When `sorting_criteria` is null (no explicit ranking axis stated), the pipeline
applies a deterministic default composite as the final sorting signal:

```
default_quality = 0.6 * normalized_reception + 0.4 * normalized_popularity
```

This is fixed code, not LLM-determined. It handles the "silly comedies" problem:
no explicit sorting was stated, so the default quality composite kicks in and
surfaces well-known comedies rather than obscure-but-on-topic ones. The LLM's
job is only to identify *when* an explicit ranking axis exists. When it doesn't,
deterministic code takes over.

### Ranking Strategy

Phase 0's sorting_criteria output determines the ranking mode:

- **Sort:** One attribute is the primary ranking axis. "Critically acclaimed christmas
  movies" → retrieve christmas movies, then `ORDER BY critical_acclaim DESC`. The
  ranking attribute isn't a "preference" (nice-to-have); it's the dominant output
  ordering.
- **Balance:** Multiple preferences combined additively. "Dark and gritty" → score on
  both darkness and grittiness, sum them. Standard preference scoring.
- **Superlative:** The defining attribute is simultaneously the retrieval concept AND
  the ranking signal. "Scariest movie ever" → retrieve scary movies, rank by degree
  of scariness. Raw similarity is preserved as a ranking signal, not flattened.
- **Default quality:** When sorting_criteria is null. Apply default quality composite
  after preference scoring.

The distinction matters because the current system always uses additive balance,
even when the user's intent is clearly "sort by X." This is why "critically
acclaimed christmas movies" underweights the acclaim dimension — acclaim
competes additively with christmas-ness rather than serving as the ranking axis
after christmas-ness gates the candidate set. A minimal `is_primary_preference`
style mechanism is enough to capture this distinction without adding general
preference weighting complexity.

---

## Three-Tier Constraint Strictness

Users are frequently imprecise with metadata constraints extracted from natural
language. "Classic 80s action movies" should include Terminator 2 (1991).
"Leonardo DiCaprio boat movie from the 2000s" clearly means Titanic (1997).

**Key design principle:** The strictness tier is determined by the SOURCE of the
constraint (UI vs NLP), not the attribute type. Same attribute, different
treatment based on how explicit the user was.

### The Three Tiers

| Tier | Source | Behavior | Example |
|------|--------|----------|---------|
| **Hard filter** | UI controls (date picker, genre checkboxes, platform selector) | Strict SQL WHERE. No exceptions. User explicitly set this. | User selects "1980-1989" in date range picker |
| **Soft constraint** | NLP-extracted metadata from query text | Generous gate + preference decay within the gate | "80s action movies" → gate at 1975-1994, preference peak at 1980-1989 |
| **Semantic constraint** | Vector similarity | Threshold determines pass/fail | "dark and gritty" |

### Soft Constraint Mechanics

1. Phase 0 extracts the constraint with a center value (e.g., `era: 1980s`)
2. Phase 1 applies a generous gate: expand range ~50% each side ("1980s" → 1975-1994, "2000s" → 1995-2014)
3. Phase 2 applies preference decay: within stated range = 1.0, outside decays with distance (T2 at 1991 ≈ 0.9, a 2005 movie ≈ 0.3)

The gate catches plausible candidates. The decay ensures movies actually in the
stated range rank above near-misses, but near-misses aren't excluded entirely.

### Attribute Softness (When NLP-Extracted)

| Attribute | Softness | Reasoning |
|-----------|----------|-----------|
| Date/era | Soft — generous ±5 years | Users imprecise about decades; cultural eras don't align to calendar decades |
| Genre | Somewhat soft — include adjacent genres | "Action movie" could include action-thriller, action-comedy |
| Certification/rating | Somewhat soft — include adjacent ratings | "PG-13" shouldn't exclude a perfect PG match |
| Country of origin | Hard | A movie is from Korea or it isn't |
| Named entities | Hard on presence | Brad Pitt is in the movie or not. Prominence is the spectrum, not presence. |
| Streaming platform | Hard | It's on Netflix or it isn't |
| Source material type | Hard | It's an adaptation or it isn't |

---

## Deal-Breaker Retrieval: Threshold + Flatten

For semantic deal-breakers retrieved via vector search, raw cosine similarity is a
spectrum. But deal-breakers need to be gates, not rankings. Proposed approach:

**Scoring mechanics:**

```
score(similarity):
  if similarity >= threshold: 1.0  (capped — you're fully in)
  if similarity < threshold:  decay(similarity)  (partial credit, approaching 0.0)
```

Above threshold, all candidates get equal credit — no embedding density bias.
Below threshold, a smooth decay gives partial credit rather than hard exclusion.
This means movies "just below" the threshold still contribute to deal-breaker
conformance scoring, just at reduced weight. The decay function creates graceful
degradation: if a query has 3 deal-breakers and no movie passes all 3, movies
"close on all 3" naturally rank above movies that "ace 2 but completely miss 1."

This solves the "Christmas problem": Home Alone and Die Hard both pass the christmas
threshold — both get 1.0 for christmas-ness, then you rank by critical acclaim.
Without capping, "aggressively christmas" Hallmark films inflate scores through
embedding density, and movies that are clearly christmas movies but have rich
multi-dimensional embeddings get penalized.

It also solves the "cheesy 80s comedies" problem: anything above threshold is
equally "cheesy enough" — cheesiness stops being a differentiator. The preference
layer handles ranking from there.

**Exception: Superlative queries.** When the user explicitly asks for the *most*
extreme example ("scariest movie ever," "funniest comedy"), capping destroys the
ranking signal they asked for. The scoring function must preserve raw similarity
as the ranking signal. Phase 0's sorting_criteria output determines whether to
cap or preserve. See "Scoring Function Varies by Query Type" below.

### Multi-Vector Thresholding

When a single semantic concept (e.g., "twist ending") has signal in multiple vector
spaces (narrative_techniques, viewer_experience, reception), threshold each relevant
space separately and take the **best score** across spaces as the deal-breaker score
for that concept. A movie clears the "twist ending" deal-breaker if it passes the
threshold in ANY of the target spaces — the signal might live in different spaces for
different movies.

### Threshold Options

Two approaches under consideration (LLM adjudication ruled out):

- **Score-distribution-based:** Find the natural gap/elbow in the retrieved
  candidates' similarity scores (similar to the derivative analysis already used for
  quality scoring thresholds in the ingestion pipeline)
- **Relative to top:** e.g., anything above 60-70% of the top candidate's score
  qualifies

**Empirical findings (see open_questions.md for full data):**
- Elbows at 100-200 entries across concept types
- Elbow percentage varies: 75% (twist ending) to 90% (dark and gritty, christmas)
- Christmas is the only bimodal distribution (binary membership concept)
- A single fixed percentage won't work across all attribute types

**Revised context:** Threshold selection is less critical than originally thought
because the architecture revision moves semantic deal-breakers to Phase 2
rescoring. The threshold now operates on a candidate pool already filtered by
reliable deterministic channels, which should produce tighter score distributions
than searching the full collection. Revisit threshold calibration after the
embedding format improvement and architecture revision are implemented.

---

## Deal-Breaker Combination: Deterministic Retrieval + Semantic Rescore

Deal-breakers can come from any channel — lexical (entities), metadata (structured
fields), or semantic (vector similarity). All are equally important regardless of
channel. No explicit weights between deal-breakers and preferences — the structural
distinction does the work.

### Phase 1: Deterministic Candidate Retrieval

**Revised based on empirical testing.** Candidates are generated exclusively from
deterministic channels. Semantic deal-breakers do NOT generate candidates — they
score in Phase 2 via cross-space rescoring.

Deterministic retrieval channels:
- Lexical: all movies matching entity deal-breakers (actors, directors, franchises)
- Metadata: all movies passing soft-constraint gates (genre, date, runtime, etc.)
- Keywords: all movies matching high-coverage IMDB keywords (christmas, animation)

The union of deterministic results enters Phase 2. Pool sizes of a few hundred to
a few thousand candidates.

**Why this change:** Empirical testing showed semantic concepts cannot reliably
generate candidates. "Funny horror" had zero vector intersection. "Dark gritty
Marvel" missed Winter Soldier. The Sixth Sense was outside top-1000 for "twist
ending." Semantic retrieval is unreliable as a candidate generator but works well
as a scoring signal applied to a pre-filtered pool. See current_search_flaws.md
#14.

**Exception: pure-vibe queries** with no deterministic anchors (e.g., "date night,"
"turn my brain off") still use vector retrieval as the candidate generator. The
current system handles this class reasonably well. Phase 0 detects this case when
no entity, metadata, or keyword deal-breakers are extracted.

### Phase 2: Full Rescore

Every candidate gets scored across ALL dimensions:

1. **Binary deal-breaker scores:** Entity presence (0 or 1), metadata filter pass
   (0 or 1, with soft constraints using proximity scoring for near-misses)
2. **Semantic deal-breaker scores via cross-space rescoring:** For each semantic
   deal-breaker concept, fetch the candidate's stored vectors from Qdrant for the
   concept's target spaces. Compute cosine similarity against the query embedding.
   Threshold-cap at 1.0 with decay below threshold. Take best score across spaces.
   This is now the primary mechanism for evaluating semantic deal-breakers — not
   vector retrieval, but vector scoring on a deterministically-retrieved pool.
3. **Deal-breaker conformance:** Percentage of deal-breakers met (binary +
   semantic). This is the PRIMARY sorting axis. A movie meeting 90% of deal-breakers
   ALWAYS outranks one meeting 70%, regardless of how popular the 70% movie is.
4. **Preference scores:** Additive scoring from semantic preferences (also via
   cross-space rescoring). Only differentiates within the same deal-breaker
   conformance tier.
5. **Sorting/quality score:** From explicit sorting criteria or default composite.
   Bounded influence — cannot override deal-breaker conformance.

### Deal-Breaker Types: Deterministic vs Semantic

Not all deal-breakers are the same type. The key distinction is now between
**deterministic deal-breakers** (which generate candidates in Phase 1) and
**semantic deal-breakers** (which score candidates in Phase 2).

"Funny horror movies":
- "Horror" → deterministic deal-breaker (genre metadata filter). Generates the
  candidate pool in Phase 1.
- "Funny" → semantic deal-breaker (cross-space rescoring in Phase 2). Scores
  horror candidates on funniness via vector similarity. Still thresholded since
  the query says "funny" not "funniest."

The genre filter generates candidates. The semantic scoring evaluates within that
set. They're both deal-breakers from the user's perspective but serve different
roles in the pipeline.

### Examples

**"Brad Pitt action films"**
- "Brad Pitt" → binary deal-breaker (lexical entity lookup)
- "Action" → binary deal-breaker (genre metadata filter)
- Both must pass to be a primary result
- Within primary results: rank by default quality composite (no explicit sorting)
- Brad Pitt prominence (billing_position) affects ranking as a preference signal

**"Funny horror movies"**
- "Horror" → binary deal-breaker (genre metadata filter → retrieve all horror movies)
- "Funny" → semantic deal-breaker (vector search within horror candidates, threshold
  capped at 1.0, decay below)
- Primary results: horror movies passing the "funny" threshold
- Fallback: horror movies with highest "funny" partial scores

**"Iconic twist endings"**
- "Twist ending" → semantic deal-breaker (target: narrative_techniques,
  watch_context, viewer_experience; threshold each space, take best)
- "Iconic" → semantic preference OR sorting criteria (target: reception)
- **No deterministic anchor exists** — this is a pure semantic query. Candidate
  generation falls back to vector retrieval (best available) or keyword matching
  if "twist" maps to a high-coverage IMDB keyword. Cross-space rescoring then
  evaluates the full candidate pool on both "twist ending" and "iconic."
- With improved embeddings (structured-label format), vector retrieval becomes
  more reliable for this case. Without it, keyword matching is the safety net.

**Graceful degradation:** When no movies meet ALL deal-breakers, fall back to movies
with the highest percentage met. "Dark gritty Marvel christmas movies" with 4
deal-breakers might yield zero 4/4 matches — show the best 3/4 movies instead.

**Preference weighting relative to deal-breakers:** Keep V1 simple. Preferences
remain equal-weighted relative to each other unless one is explicitly marked as
the primary preference. The structural tiering already ensures preferences
cannot override a higher deal-breaker-conformance tier, so rich preference
weighting is not required up front.

---

## Subquery Generation Changes

**Revised role:** With semantic concepts moving from candidate generation to
cross-space rescoring, the subquery's job changes. It no longer needs to maximize
retrieval recall from a massive collection. Instead, it generates the query
embedding used for cosine similarity scoring against candidates already in the pool.

For scoring, the subquery should be **precise and attribute-focused** — matching the
semantic structure of how the movie's metadata was embedded. If we adopt
structured-label embedding (see below), the search subquery should be generated in
the same structured shape so query and document embeddings occupy the same semantic
space.

**When vector retrieval is still the candidate generator** (pure-vibe queries), the
current expansive approach remains appropriate since we need broad recall.

**Critical principle:** Query expansion should rely on the *interpreted intent* from
Phase 0, not just the raw query text. If Phase 0 interprets "critically acclaimed
christmas movies" as "retrieve christmas movies, rank by acclaim," the christmas
subquery should be narrow/focused while the acclaim preference uses the intent-
informed ranking strategy.

---

## Embedding Format: Structured Labels (Prerequisite)

**This is the single highest-priority improvement identified by empirical testing.**
The embedding format problem (current_search_flaws.md #13) is a prerequisite for
the new architecture to work as designed. Without it, semantic deal-breaker scoring
in Phase 2 inherits the same signal dilution that makes Phase 1 vector retrieval
unreliable.

### Problem

Each vector space's text is embedded as a flat concatenation of metadata terms.
The embedding model (text-embedding-3-small, 1536 dims) compresses all terms into
a single vector, causing per-attribute signal dilution for multi-dimensional movies.

Evidence: The Sixth Sense doesn't appear in top-1000 for "twist ending" in
narrative_techniques despite having explicit twist language in its metadata.
Score: 0.5730 (82% of max). The flat-list format loses the twist signal among
the movie's many other narrative attributes.

### Proposed Fix: Structured-Label Embedding

Embed vector text with structured labels that preserve per-attribute context:

**Current (flat list):**
```
plot twist / reversal, planted-foreshadowing clues, slow-burn reveal,
flashback storytelling, multiple-perspective narration, ...
```

**Proposed (structured labels):**
```
information_control: plot twist / reversal, planted-foreshadowing clues
pacing_and_structure: slow-burn reveal, flashback storytelling
perspective_and_voice: multiple-perspective narration, ...
```

The structured format gives the embedding model semantic grouping context — it
knows "plot twist" belongs to information_control specifically, not just to an
undifferentiated bag of narrative terms.

### Search Query Generation

Generate search queries in the **same structured shape** as the embedded text.
The search LLM produces output matching the vector space's Pydantic model (e.g.,
NarrativeTechniquesOutput), which is templated into the same structured format
before embedding. This ensures query and document embeddings occupy the same
semantic structure.

### Validation Plan

Test on a small sample (10-20 movies) without full re-ingestion:
1. Re-embed sample movies with structured-label format
2. Generate structured search queries for known test cases
3. Compare retrieval ranks for known-relevant movies under both formats
4. If structured format significantly improves ranks → plan full re-ingestion

See open_questions.md Test A for the full test specification.

---

## Graceful Degradation

The deterministic-retrieval-then-rescore architecture handles graceful degradation
natively. Candidates enter the pool via deterministic channels, then get scored on
all deal-breakers (deterministic + semantic) with decay below threshold:

1. **Ideal case:** Movies meeting all deal-breakers rank at the top (primary results)
2. **Partial match:** Movies meeting most deal-breakers rank below, with below-threshold
   decay scores contributing partial credit. These serve as natural fallback results.
3. **Thin primary set:** If fewer than ~10 movies meet all deal-breakers, the partial
   matches fill out the result set without requiring a separate fallback mechanism.
4. **Phase 4 exploratory:** For broader "you might also like" suggestions beyond even
   partial matches, Phase 4 runs the current-style expansive search.

This keeps the graceful degradation behavior from the earlier unified-scoring
idea, but the finalized V2 design now makes the tiers explicit rather than
implicit. Movies that satisfy more inclusion dealbreakers form higher tiers,
and partial matches naturally fill lower tiers without requiring a separate
relaxation policy.

---

## Features of Exploratory Extension (Phase 4)

When primary results are thin or to provide additional value:

**Two presentation options:**
- **Append:** clearly separated "You might also like" section below primary results
- **Weave:** interleave exploratory results with primary results for variety, giving a
  mix of "exactly what you asked for" and "hey maybe consider this"

Weaving is more discovery-friendly but risks confusing users who think everything
should match their query. Appending is safer and more honest.

Exploratory results come from the current-style broad search — expansive subqueries,
additive scoring across all vector spaces. This is what the current system does well:
finding related/adjacent movies that the user might not have thought of.

---

## Handling Special Query Types

### Pure-Vibe Queries (no deterministic anchors)
When Phase 0 produces no deterministic deal-breakers (no entities, no metadata
filters, no keyword matches), vector retrieval becomes the candidate generator.
This includes "date night," "turn my brain off," "something cozy for a rainy day."
The current system handles this class reasonably well — the architecture change
primarily affects queries with mixed deterministic + semantic constraints.

With the structured-label embedding improvement, pure-vibe retrieval should also
improve since the embedding format preserves per-attribute signal better.

### Negation-Heavy Queries
Can't retrieve by what movies AREN'T. Retrieve broadly on the positive signals (genre,
vibe), then post-filter using metadata or LLM evaluation for the negated attributes.
Negations are more naturally modeled as hard filters than as retrieval queries.

### Similarity Queries ("like Inception")

**Decision:** "Movies like xyz" should get routed to a different major flow
from standard constrained search. The routing distinction is finalized even if
the exact implementation can still trail the core standard-flow work.

Could decompose the reference movie into its actual metadata attributes and determine
which are most distinctive (vs generic). Use the distinctive attributes as deal-
breakers and generic ones as preferences. A weighted mix of targeted vectors
(plot_analysis, viewer_experience, narrative_techniques, etc.) provides more precise
similarity scoring than a single generalist embedding would — each dimension's
contribution can be controlled independently.

**Franchise as a similarity attribute:** "Similar movies" also means entries to the
same franchise — if someone asks for movies like "The Dark Knight," other Batman/DC
movies are relevant beyond just vector similarity. Franchise membership
(`movie_franchise_metadata`) should be checked as an additional structured attribute
when computing similarity, alongside vector distances across spaces.

**Distinctive vs generic decomposition:** Not all of a reference movie's traits are
equally important for similarity. Inception's mind-bending nested reality structure is
distinctive; its "action movie" genre is generic. The decomposition should weight
traits by how much they differentiate the reference movie from the average movie.
Traits shared by thousands of movies (action, drama, English-language) are weak
similarity signals; traits shared by few movies (nested dream worlds, unreliable
reality) are strong ones.

**Weaving similarity into traditional results:** When any search query produces a
strong single-movie match at the top (identified by a large score gap between #1 and
#2), consider running a secondary similarity search on that movie and weaving those
results in as a "similar to your top match" tier. This combines identification with
discovery — someone searching "High School Musical" gets HSM first, then movies like
Lemonade Mouth below.

### Ambiguous Queries (Multiple Plausible Interpretations)
When Phase 0 detects that the query's intent could go in multiple valid directions,
the system should surface the competing interpretations rather than forcing a single
one.

**Example:** "I need to feel something" could mean:
- Cathartic crying (sad/emotional movies)
- Adrenaline rush (action/thriller)
- Intellectual stimulation (mind-bending/philosophical)

**Approach:** Phase 0 emits multiple interpretation branches, each with its own
deal-breaker/preference structure. Each branch runs its own retrieval pipeline in
parallel. Results are presented as clickable interpretation groups: "Did you
mean: emotional movies | intense thrillers | mind-bending films?"

This could be:
- A separate UI mode triggered only when Phase 0's confidence in a single
  interpretation falls below a threshold
- An agent-based approach where each interpretation runs its own search concurrently

**Tradeoff:** Adds latency (multiple parallel searches) and UI complexity. Worth it
for genuinely ambiguous queries where a single interpretation would be wrong most of
the time. Should NOT trigger for queries with clear intent — the system should be
confident in its single interpretation the vast majority of the time.

---

## Scoring Function Varies by Query Type

The type of scoring applied to vector results should be determined by the query type,
not applied uniformly. Phase 0's ranking strategy output drives this selection:

### Threshold + Cap (for deal-breakers)
Above threshold = 1.0 (capped), below = decay toward 0.0. Used when the attribute is
a gate: "christmas movies," "movies with twist endings," "cheesy comedies." The user
wants movies that *clear the bar*, not the single movie that maximizes the attribute.
Below-threshold decay provides graceful fallback rather than hard exclusion.

### Preserved Similarity (for superlatives)
Raw cosine similarity is the ranking signal. Used when the user explicitly asks for
the most extreme example: "scariest movie ever," "most visually stunning." Flattening
would destroy the ranking they asked for.

### Diminishing Returns (for preferences)
Additive scoring with diminishing marginal value. The difference between "very dark"
and "extremely dark" matters less than between "somewhat dark" and "very dark." Used
for tonal qualifiers, stylistic preferences, and other "more is better but with
saturation" attributes.

### Sort-by (for ranking axes)
The attribute isn't blended with other scores — it IS the sort order. Used when Phase
0 identifies a clear ranking axis: "critically acclaimed X" → sort by acclaim after
X passes the gate. Implemented as a post-gate sort rather than an additive score
component.

---

## Candidate Retrieval & Cross-Space Rescoring

### Revised Architecture: Deterministic Retrieval + Semantic Rescoring

**Empirical finding that forced this revision:** Semantic concepts cannot reliably
generate candidates with current embeddings. "Funny horror" has zero intersection
between vector candidate sets. "Dark gritty Marvel" misses Winter Soldier from
vector results. The Sixth Sense doesn't appear in the top 1000 for "twist ending."
See current_search_flaws.md #13-14 for full evidence.

**New model:** Phase 1 generates candidates exclusively from deterministic channels
(metadata filters, entity lookup, keyword matching). Phase 2 applies ALL semantic
scoring — both deal-breakers and preferences — via cross-space rescoring on the
deterministic candidate pool.

This means the intersection problem is eliminated by design: candidates are retrieved
from the most reliable channel, and semantic concepts score within that pool rather
than trying to independently generate overlapping candidate sets.

### Retrieval Depth
Entity deal-breakers: top-500 (lexical is precise).
Metadata filters: all matching rows (deterministic, exact).
Keyword filters: all matching rows (deterministic, exact).

For pure-vibe queries with no deterministic anchors, vector retrieval is still the
candidate generator. In that case, retrieval depth matters, but the primary lever
is embedding quality (see embedding format problem, current_search_flaws.md #13).

### Cross-Space Rescoring (now REQUIRED, not deferred)

Every candidate in the pool gets scored against every semantic deal-breaker and
preference via cross-space rescoring. This is the core of Phase 2.

**Implementation:** Qdrant `retrieve()` API for batched point lookups. Fetch stored
vectors for all candidates in the pool, compute cosine similarity against query
embeddings for each semantic concept's target spaces.

**Expected load:** For a pool of 500-2000 candidates across 2-4 target spaces,
that's 1000-8000 vector lookups. Qdrant batched retrieval + numpy cosine similarity
should be feasible, but latency testing is needed (see open_questions.md Test B).

**This replaces the original "union then rescore" model** where semantic deal-breakers
contributed to both candidate generation AND scoring. Now they contribute to scoring
only, with deterministic channels handling candidate generation.

---

## Trending & Discovery Candidate Injection

Trending-oriented queries ("trending now," "popular movies right now") need a
fundamentally different candidate sourcing strategy. Rather than running vector
retrieval and hoping trending movies surface through quality reranking, inject the
trending movie set directly as the candidate pool.

### Full Injection
When Phase 0 classifies the query as primarily trending/discovery (query type #7),
skip vector retrieval entirely and use the trending set from Redis as the candidate
pool. Rank by the trending signal (recency, popularity velocity, etc.) with
system-level priors applied on top.

### Partial Injection (Hybrid)
For queries that aren't explicitly trending but could benefit from trending awareness,
constrain a portion of the vector retrieval to the trending candidate pool. For
example, half the vector spaces search globally while the other half search within
trending IDs only. This surfaces trending movies that happen to match the query
without sacrificing global recall.

### Pipeline Integration
This doesn't fit cleanly into the deal-breaker/preference framework — it's a
candidate-sourcing modifier, not an attribute. It should be a Phase 0 output flag:
`candidate_source: "global" | "trending" | "hybrid"` that controls how Phase 1
populates the initial candidate pool.

---

## Data Layer Design Decisions

Decisions from the data gap analysis. These are structural changes to what data is
stored and how, needed to support the query types and pipeline architecture above.

### New Postgres Tables

#### movie_awards (inverse-lookup by award)

Stores award nominations and wins. Designed for inverse lookup — given an award, find
which movies won it.

Implementation uses SMALLINT enum IDs for `ceremony` and `outcome`
(via `AwardCeremony` and `AwardOutcome` enums in `schemas/enums.py`),
while `award_name` stays as TEXT.

```
movie_awards (
    movie_id      BIGINT REFERENCES movie_card ON DELETE CASCADE,
    ceremony_id   SMALLINT NOT NULL,  -- AwardCeremony enum ID
    award_name    TEXT NOT NULL,       -- "Oscar", "Palme d'Or", "Golden Lion"
    category      TEXT,               -- "Best Picture", etc. (nullable for grand prizes)
    outcome_id    SMALLINT NOT NULL,  -- AwardOutcome enum ID (1=winner, 2=nominee)
    year          SMALLINT NOT NULL,
    PRIMARY KEY (movie_id, ceremony_id, award_name, COALESCE(category, ''), year)
)
```

**Index:** `idx_awards_lookup (ceremony_id, award_name, category, outcome_id, year)` for "Oscar winners" queries.

**Also in vectors:** Deterministic `major_award_wins` ceremony summary included in
the reception vector for semantic queries like "award-winning thriller." Winner rows
only, distinct ceremonies only, fixed priority order, no nominations. The structured
table handles specific ceremony/category/nominee filtering; the vector handles vague
prestige language.

**Data source:** IMDB GraphQL API exposes award data (nominations + wins by ceremony).
Requires new scraping target.

#### movie_franchise_metadata

Replaces the current title-token + character-matching hack for franchise search.

**Franchise definition:** Any recognizable intellectual property or brand that
originated in any medium — film series, video games, toys, books, comics, TV
shows, board games, theme parks, etc. — where the movie is an adaptation,
extension, or product of that IP. This matches the current lexical entity
extractor's definition. Examples: "Mario" (video game IP), "Barbie" (toy IP),
"Transformers" (toy/cartoon IP), "Harry Potter" (book IP), "Marvel Cinematic
Universe" (comic/film IP). The `franchise_name_normalized` should be the **IP
name** (e.g., "mario"), not the film series name (e.g., "super mario bros movie
series"), since users search by IP name.

```
movie_franchise_metadata (
    movie_id               BIGINT PRIMARY KEY REFERENCES movie_card,
    lineage                TEXT,
    shared_universe        TEXT,
    recognized_subgroups   TEXT[] NOT NULL DEFAULT '{}',
    launched_subgroup      BOOLEAN NOT NULL DEFAULT FALSE,
    lineage_position       SMALLINT,
    is_spinoff             BOOLEAN NOT NULL DEFAULT FALSE,
    is_crossover           BOOLEAN NOT NULL DEFAULT FALSE,
    launched_franchise     BOOLEAN NOT NULL DEFAULT FALSE
)
```

**Storage note:** `lineage` and `shared_universe` are both stored normalized and
both feed `lex.inv_franchise_postings`, so franchise lookup can match either
field without caring which slot the LLM used. **These must always be searched
together** — LLMs can flip-flop which field gets which value (e.g., putting "Shrek"
in lineage vs shared_universe), so the search path must be slot-agnostic by
design.

**Data sources:**
1. TMDB `belongs_to_collection` — reliable base for ~25% of movies. Gives franchise
   name + movie membership. Doesn't cover spinoffs or brand-level groupings.
2. LLM enrichment — fills gaps: assigns spinoffs to parent franchises, infers brand-
   level groupings (MCU), emits the full `FranchiseOutput` shape, and names
   `recognized_subgroups` only when established terminology exists globally.

**LLM inputs:** title, release_year, overview (as an identification aid — helps
the LLM correctly identify which movie this is, NOT for inferring franchise from
plot similarity), TMDB collection_name (if any), production_companies,
overall_keywords, characters. Compact, high-signal set with no generated-metadata
dependencies.

**Replaces:** Current franchise search in lexical_search.py that combines title tokens
+ character matching. New approach: franchise becomes its own lexical posting table
(`inv_franchise_postings`) searched by normalized franchise name.

### Franchise Search Flow

Franchise queries are decomposed by Phase 0 into up to six components:

- **`lineage` / `shared_universe`** — canonical expanded form, same global
  normalization rule as generation. Always lowercase. Example: "dark knight
  movies" → `batman` with subgroup filter. "spinoffs of shrek" hits
  `shared_universe = shrek` (shape B: parent franchise of a spinoff).
- **`recognized_subgroups`** — free-text qualifier, trigram-matched against
  stored normalized labels. "mcu phase 3 movies" → lineage=any marvel
  lineage, subgroup=`phase three`.
- **`lineage_position`** — enum filter. "batman sequels" → lineage=batman,
  lineage_position=sequel. NOTE: `remake` is NOT consumed at search time
  — use `source_of_inspiration` for cross-film retelling queries.
- **`is_spinoff` / `is_crossover`** — boolean filters. "marvel spinoffs" →
  shared_universe=`marvel cinematic universe` AND `is_spinoff=true`.
- **`launched_subgroup`** — boolean filter. "first entry in phase three"
  combines subgroup=`phase three` with launched_subgroup=true.
- **`launched_franchise`** — boolean filter. "movies that launched a
  franchise" → launched_franchise=true.

**franchise_name resolution:** Both the ingestion LLM (franchise generation) and
the search extraction LLM are instructed to output the most common, fully expanded
form of the franchise/IP name — no abbreviations. Same convention as the lexical
entity extractor for person names. "MCU" → "Marvel Cinematic Universe", "HP" →
"Harry Potter", "Mario" → "Super Mario" (or whatever the canonical IP name is).
This ensures both sides converge on the same canonical string without needing
alias tables. After extraction, `normalize_string()` is applied and trigram
similarity against `lex.lexical_dictionary` resolves to a `term_id`.

**Phase 1 retrieval:**
1. `lineage` / `shared_universe` → `normalize_string()` → trigram similarity
   against `lex.lexical_dictionary` → `term_id` → shared franchise posting lookup
2. Boolean filters (`launched_subgroup`, `launched_franchise`, `is_spinoff`,
   `is_crossover`) and `lineage_position` applied as structured WHERE clauses
   on `movie_franchise_metadata`
3. Optional `recognized_subgroups` match: trigram similarity on the normalized
   labels. Candidate set is small enough (3-30 movies) that no index is needed.

The two-axis design means a single query like "movies that launched a franchise"
is a single structured filter (`launched_franchise=true`), while "spinoffs of
marvel" is two filters (shared_universe + `is_spinoff=true`), with no
axis-conflation ambiguity.

#### Role-specific person posting tables

Split current `inv_person_postings` into:
- `inv_actor_postings` (term_id, movie_id, billing_position, cast_size)
- `inv_director_postings` (term_id, movie_id)
- `inv_writer_postings` (term_id, movie_id)
- `inv_producer_postings` (term_id, movie_id)
- `inv_composer_postings` (term_id, movie_id)

Actor postings include billing_position and cast_size for prominence scoring.

**Entity extraction changes:** Add optional `role_hint` field to extracted entities:
`"director" | "actor" | "writer" | "producer" | "composer" | null`. When null, search
all role tables and merge results.

**Role boosting behavior:** When role isn't explicitly stated, boost the person's most
likely role rather than hard-filtering. "Spielberg films" → search all tables, but
boost director results since that's what most users mean. Still includes his producer
credits at lower weight — users may discover unexpected connections.

**Actor prominence scoring:** Three query-controlled modes:

1. **Exclude non-major:** Only include actors in top min(2-3, 10-15% of cast_size).
   Triggered by explicit "starring" language in the query.
2. **Boost by position:** Continuous prominence = `1.0 - (position / cast_size)`.
   Default mode — leads score higher than cameos without excluding anyone.
3. **Binary:** Current behavior, all actors equal. Fallback when prominence data
   is missing.
4. **Reverse:** For "minor roles" queries — boost deep credits, demote leads.

The mode is determined by Phase 0 based on query language. "Starring" triggers mode 1.
Most queries use mode 2. Mode 3 is the fallback.

### New movie_card Fields

| Field | Type | Source | Purpose |
|-------|------|--------|---------|
| `country_of_origin_ids` | INT[] | IMDB countries_of_origin | Hard filter for production-country queries like "movies from South Korea." In Postgres only, NOT Qdrant payload. Pre-filter in Postgres, then pass IDs to Qdrant. |
| `box_office_bucket` | TEXT | TMDB revenue + era adjustment | "hit" / "flop" / null. Same era-adjusted pattern as budget_bucket. |
| `source_material_types` | INT[] (enum) | LLM source_of_inspiration (re-generated with enum constraints) | Array because movies can have multiple sources (e.g., novel adaptation + based on true story). |

**Source material enum taxonomy (finalized, implemented in `schemas/enums.py`):**
See [source_material_type_enum.md](source_material_type_enum.md) for the full
definition with boundary notes and re-generation guidance.
```
NOVEL_ADAPTATION, SHORT_STORY_ADAPTATION, TRUE_STORY, BIOGRAPHY,
COMIC_ADAPTATION, FOLKLORE_ADAPTATION, STAGE_ADAPTATION,
VIDEO_GAME_ADAPTATION, REMAKE, TV_ADAPTATION
```

Array-valued because movies frequently have multiple applicable types (e.g.,
Schindler's List = NOVEL_ADAPTATION + TRUE_STORY, a live-action anime remake =
REMAKE + COMIC_ADAPTATION). Empty array = original screenplay (no explicit enum
value — original screenplays are identified by the absence of any source material
type). Queries for "original screenplays" filter for movies with an empty
`source_material_type_ids` array.

**Note on budget_bucket:** Already exists on movie_card and is already era-adjusted.
No change needed — confirmed as correctly placed.

### Production Medium via Keyword Search

Rather than a boolean `is_animation`, production medium is identified by searching
a movie's overall_keywords + production_keywords for medium-related terms. IMDB
maintains a relatively stable set of these:

```
animation, stop-motion-animation, cgi-animation, rotoscope,
claymation, live-action, motion-capture, puppet-animation, ...
```

The query understanding step maps user terms to this vocabulary:
- "claymation" → search for "stop-motion-animation" or "claymation"
- "animated movies" → search for "animation"
- "CGI" → search for "cgi-animation"

This is more flexible than a boolean — it distinguishes sub-types of animation that a
boolean would flatten. It's also a pilot case for the broader keyword-based
deal-breaker filtering concept (see below).

### Production Vector — Scope After Removals

After moving filterable concepts to structured fields, the production vector loses:
- Countries of origin → `country_of_origin_ids`
- Production companies → already in `inv_studio_postings`
- Languages → already in `audio_language_ids`
- Budget/revenue → `budget_bucket` / `box_office_bucket`
- Source material → `source_material_types` enum
- Franchise lineage → `franchise_membership` table
- Decade/era → derivable from `release_ts`
- Animation/live action → keyword search

**What remains:** Filming locations + production technique keywords.

**Production technique keywords** are tightened to terms about HOW the movie was
made (not what it's about):
- Animation modalities/sub-techniques: hand-drawn animation, 2d animation,
  3d animation, traditional animation, computer animation, cgi animation,
  stop-motion, rotoscope, motion-capture, hybrid/partial animation labels
- Visual capture/rendering: black-and-white, 3d, found-footage, single-take,
  long take, handheld-camera
- Process: practical-effects

**Explicitly excluded:** IMAX, anthology, vignette, mockumentary, nonlinear
timeline, and broad medium labels like generic "live action".

Together, filming locations (WHERE) + production technique (HOW) form a coherent
"production context" embedding. The previous definition was too broad, allowing
thematic content that bleeds into other vectors and dilutes the embedding.

**Decision: Regenerate the production vector** with the tightened definition (filming
locations + production technique keywords only). The production vector was cheap to
generate and the regeneration cost is low. The previous definition was too broad and
caused thematic bleed that diluted the embedding.

**Open question:** After regeneration, is the tightened content enough to justify a
dedicated vector space? The remaining content is coherent but thin. Options:
1. Keep as a lean, focused vector space
2. Eliminate the production slot entirely while keeping anchor as a separate
   lean holistic vector
3. Repurpose the slot for a different vector space that would add more value

Revisit after regeneration and measure whether the lean production vector contributes
meaningfully to search results.

**Production/lexical overlap (flaw #7) is naturally resolved** by these removals.
Production companies are already in studio postings; with countries and other entities
moved to structured fields, the production vector no longer contains
lexical-matchable entities.

### Keyword-Based Deal-Breaker Filtering

**Updated after vocabulary audit** — findings are summarized below and
the full `OverallKeyword` enum is in `implementation/classes/overall_keywords.py`.

IMDB's `overall_keywords` is a curated keyword taxonomy of exactly
225 terms with 100% movie coverage. It is NOT a free-form tagging system —
it's disjoint from `plot_keywords` (zero overlap). The vocabulary is compact
enough for pure static mapping; no LLM translation needed.

The primary deal-breaker value is **sub-genre precision across the entire
genre space**: 16 horror sub-types, 17 comedy sub-types, 3 western variants,
5 sci-fi sub-types, etc. These are exactly the deterministic signals that
vector search is weakest at. `plot_keywords` does not need its own search
path — its value is already absorbed by the metadata generation pipeline.

**Proposed approach:** Keywords as a **boost signal within deal-breaker retrieval, not
a hard pre-filter.** Hard filtering risks false negatives from missing tags (a
legitimate Christmas movie without the "Holiday" keyword gets silently excluded).

**How it works:**
1. Phase 0 identifies deal-breaker concepts and checks if they map to known
   `overall_keywords` terms (225-term vocabulary provided as QU LLM context)
2. Phase 1 retrieves candidates via BOTH vector search AND keyword matching (union)
3. Candidates matching the keyword get automatic pass on deal-breaker threshold
4. Candidates without the keyword can still enter via vector similarity — they just
   need to clear the vector threshold independently

**Storage:** `movie_card.keyword_ids INT[]` with GIN index (Postgres). Maps
`overall_keywords` to `lex.lexical_dictionary` string IDs.

**Query understanding:** The full 225-term vocabulary is small enough to include
as context in the QU prompt. The LLM selects matching terms when the user's
query implies a sub-genre or deal-breaker concept.

**Deal-breaker categories identified:** Production medium (6 tags), holiday
(5 tags), horror sub-genres (16), comedy sub-genres (17), thriller sub-genres
(6), drama sub-genres (12), crime sub-genres (8), western sub-genres (3),
sci-fi sub-genres (5), fantasy sub-genres (4), action styles (7), romance
sub-genres (5), thematic concepts (9), adventure sub-types (12), documentary
sub-types (11), sports (8), musical sub-types (4), language/nationality (30),
format/other (8). Full per-category mappings (user query → keyword terms) are
encoded in the `OverallKeyword` enum definitions in
`implementation/classes/overall_keywords.py`.

**Language/nationality tags complement structured fields.** The 30
language/nationality `overall_keywords` (French, Hindi, Korean, etc.) capture
"cinema tradition" rather than just production country. A query for "Korean
cinema" should check both the `Korean` keyword tag AND
`country_of_origin_ids` / `audio_language_ids`. The keyword tag captures
editorial judgment about cinema tradition membership, which doesn't always
align with production country alone.

**`plot_keywords` do not need a search path.** `plot_keywords` is a free-form
community tagging system (114K distinct terms, 53% singletons). Its value is
already absorbed by the metadata generation pipeline and distilled into
structured LLM metadata across the vector spaces.

---

## Design Principles

1. **Gate then rank.** Deal-breakers constrain the candidate set; preferences order
   within it. Never let preferences compensate for missing deal-breakers.

2. **Additive scoring is for preferences, not gates.** Weighted sums are correct for
   combining "how dark" + "how gritty." They're wrong for "is this a marvel movie."

3. **Flatten deal-breaker scores after thresholding.** A movie that "barely qualifies"
   as a christmas movie should rank the same on christmas-ness as a movie drowning in
   christmas imagery. The preference layer handles differentiation.

4. **Narrow retrieval, broad exploration.** Phase 1 uses precise queries to find what
   the user asked for. Phase 4 uses expansive queries to surface what they might also
   enjoy. Don't conflate these goals in a single retrieval step.

5. **Detect and adapt to query structure.** Different query types need different
   pipeline behavior. The system should identify whether deal-breakers exist, whether
   the query is negation-heavy, etc., and adjust accordingly rather than forcing every
   query through the same pipeline.

6. **UI filters take precedence.** When the user has explicitly set filters in the UI,
   those override any LLM-inferred deal-breakers in case of conflict.

7. **Every query type needs a fallback path.** Users can be wrong even when
   confident — someone might say "Bill Murray" and mean Bill Pullman, or name
   a franchise that doesn't match any entity in our data. The system should
   attempt to meet the query as literally as possible first, but when strict
   matching yields zero or near-zero results, every query type must have a
   defined fallback that loosens constraints. This isn't just the "thin primary
   results" case from Phase 4 — it's a design philosophy. Entity lookups fall
   back to fuzzy matching then semantic, metadata filters relax the most
   restrictive constraint first, similarity searches widen the neighborhood.
   The fallback path should be designed per-category, not as a one-size-fits-all
   broadening.

8. **Query expansion derives from interpreted intent, not raw text.** Subquery
   generation should use Phase 0's structured interpretation — deal-breakers,
   preferences, ranking strategy — as its primary input. The raw user query is
   context, but the interpreted intent determines what each subquery is trying
   to achieve. If the intent says "christmas is the deal-breaker, acclaim is
   the ranking axis," the christmas subquery should be narrow/focused while
   acclaim scoring uses the intent-informed ranking strategy.

9. **Scoring functions adapt to query structure.** The system should not apply
   the same scoring function uniformly. Threshold+flatten for deal-breakers,
   preserved similarity for superlatives, diminishing returns for preferences,
   sort-by for ranking axes. Phase 0 determines which function applies where.

10. **Query understanding / expansion LLMs should err on the side of
    over-inclusion.** The system is designed to surface the right options once
    we have all the necessary candidates — the bottleneck is getting relevant
    movies INTO the candidate pool, not filtering them out. QU and expansion
    LLMs should cast a wider net rather than a narrower one. A false positive
    in the candidate pool gets filtered by scoring; a false negative is
    permanently lost.

11. **Separating candidate generation from reordering is load-bearing.** The
    LLM must understand which parts of the query should generate candidates
    and which parts should only reorder/score results. This is the single most
    important classification Phase 0 makes. Getting this wrong means either
    (a) using a broad vibe as a candidate generator (unreliable) or (b) using
    a precise entity as merely a scoring signal (wastes its precision). Every
    other design decision flows from this split being correct.

12. **Be cautiously broad in interpretation.** When a user's query maps to a
    structured field, prefer the generous interpretation. "Based on a true
    story" likely means more than just the `TRUE_STORY` enum — it probably
    also includes `BIOGRAPHY` and possibly `FOLKLORE_ADAPTATION` depending on
    context. The QU LLM should map to multiple plausible enum values rather
    than the single most literal one. Same principle applies to genre, era,
    and other soft constraints.

---

## Concept Routing: User Concepts → Retrieval Systems

When Phase 0 decomposes a user query, each concept must be routed to the
correct retrieval system. Concept tags handle binary deal-breaker filtering
(27 tags in `concept_tag_ids`), but many common user concepts are served by
other structured fields, keywords, vectors, or query-time derivations.

### Concept tags vs. vector spaces

Concept tags and vector spaces serve different purposes for the same movie:
- Tags → Phase 1 deterministic candidate retrieval ("give me ALL revenge movies")
- Vectors → Phase 2 spectrum scoring ("among these, rank by how dark they are")

A movie tagged PLOT_TWIST still has twist-related content in its
narrative_techniques vector. The tag gets the candidate set right; the vector
scores nuance within it. This is complementary, not redundant.

### Concept tag search routing (PLOT_TWIST vs. TWIST_VILLAIN)

TWIST_VILLAIN is a subset of PLOT_TWIST where the surprise is specifically
about who the villain is. For search routing:
- "twist movies" / "movies with a plot twist" → filter by `PLOT_TWIST` tag
- "movies with a twist ending" → filter by `PLOT_TWIST` tag + rank by
  ending-specific signals (narrative_techniques vector, information_control
  terms mentioning reveals near resolution)
- "movies with a twist villain" → filter by `TWIST_VILLAIN` tag

### Route through the keyword endpoint (concept-first, multi-map)

Phase 0 should reason in terms of canonical concept families first. The keyword
endpoint is one conceptual taxonomy backed by multiple deterministic stores.
Broad concepts may resolve to `keyword_ids`, `genre_ids`,
`source_material_type_ids`, and/or `concept_tag_ids` together.

| User concept | Canonical family | Likely step-3 resolution | Notes |
|-------------|------------------|--------------------------|-------|
| "conspiracy movies" | Crime / Mystery / Suspense / Espionage | `Conspiracy Thriller`; sometimes also `Political Thriller` / `Political Drama` | One concept, possibly multi-keyword resolution. |
| "heist movies" | Crime / Mystery / Suspense / Espionage | `Heist`, optionally `Caper` | `CON_ARTIST` covers adjacent non-heist scam space. |
| "coming of age movies" | Audience / Age / Life Stage | `Coming-of-Age` | If phrased loosely ("growing up"), may need semantic fallback. |
| "superhero movies" | Action / Combat / Heroics | `Superhero` and often genre-style action backing | Multi-map is allowed. |
| "film noir" | Crime / Mystery / Suspense / Espionage | `Film Noir` plus any overlapping genre backing | Treat as one concept. |
| "time travel movies" | Fantasy / Sci-Fi / Speculative | `Time Travel` | `TIME_LOOP` stays the distinct structural sub-concept. |
| "Christmas / holiday movies" | Seasonal / Holiday | `Holiday` and holiday sub-types | Centrality still belongs in semantic ranking. |
| "true crime" | Nonfiction / Documentary / Real-World Media | `True Crime` | Canonical here, not under crime fiction. |
| "mockumentary" | Program / Presentation / Form Factor | `Mockumentary` | Presentation/form concept, not a generic documentary request. |
| "documentaries" | Nonfiction / Documentary / Real-World Media | `Documentary`, sometimes overlapping broad genre backing | One concept, potentially multi-store. |
| "biographies / biopics" | Source Material / Adaptation / Real-World Basis | `Biography`, possibly overlapping genre/keyword backing | Canonical family is real-world basis. |
| "remakes" | Source Material / Adaptation / Real-World Basis | `REMAKE` source-material type and/or overlapping keyword backing | Franchise-specific remakes route elsewhere. |
| "short films / shorts" | Program / Presentation / Form Factor | `Short` | Categorical short-form classification, not runtime. |
| "female lead" | Story Engine / Setting / Character Archetype | `female_lead` | Character classification, not entity lookup. |
| "movies with a twist ending" | Narrative Mechanics / Endings | `plot_twist` plus ending-aware downstream ranking | Structural classification with semantic/rerank nuance. |

### Route to other deterministic fields

| User concept | Route to field | Notes |
|-------------|---------------|-------|
| "award-winning movies" | `movie_awards` table | Awards are their own endpoint. |
| "movies from [country]" | `country_of_origin_ids` + `audio_language_ids` + culture keywords when appropriate | Separate production country from cultural tradition. |
| "movies on Netflix" | `watch_offer_keys` | Availability/logistics. |
| "movies from the 80s" | `release_ts` | Temporal metadata constraint. |
| "under 90 minutes" | `runtime_minutes` | Runtime metadata constraint. |
| "R-rated movies" | `maturity_rank` | Rating metadata constraint. |
| "big budget movies" | `budget_bucket` | Quantitative metadata. |
| "box office hits" | `box_office_bucket` | Quantitative metadata. |
| "[franchise] movies" | `movie_franchise_metadata` | Named franchise / lineage. |
| "[actor/director] movies" | Role-specific posting tables | Named entity lookup. |

### Route to vector similarity (spectrum concepts)

Concepts where degree matters. Phase 2 cross-space rescoring with
threshold+flatten handles the binary aspect; raw similarity handles ranking.

| User concept | Route to vector space(s) | Notes |
|-------------|-------------------------|-------|
| "scary movies" | `viewer_experience` (tension_adrenaline, disturbance_profile) | Use semantic scoring for scariness degree, but pair with keyword-compatible horror classification when the query is asking for horror movies broadly. |
| "funny [genre] movies" | `viewer_experience` (tone_self_seriousness) | "Funny horror" needs semantic scoring, not binary tagging. |
| "dark / bleak movies" | `viewer_experience` (emotional_palette, tone_self_seriousness) | Spectrum from "slightly dark" to "nihilistic." |
| "intense / tense movies" | `viewer_experience` (tension_adrenaline) | Degree of intensity is the ranking signal. |
| "mind-bending movies" | `viewer_experience` (cognitive_complexity) + `narrative_techniques` | PLOT_TWIST and NONLINEAR_TIMELINE tags cover structural aspects. |
| "slow burn movies" | `viewer_experience` (tension_adrenaline) | Pacing is fundamentally a spectrum. |
| "disturbing movies" | `viewer_experience` (disturbance_profile) | Wide spectrum from "unsettling" to "unwatchable." |
| "cozy / comforting movies" | `viewer_experience` (emotional_palette) + `watch_context` | Degree matters. |
| "visually stunning movies" | `viewer_experience` (sensory_load) + `narrative_techniques` | Degree of visual quality. |
| "great soundtrack movies" | `watch_context` (key_movie_feature_draws) | Degree and subjectivity. |
| "cheesy movies" | `viewer_experience` (tone_self_seriousness) | Degree of cheesiness is the point. |
| "gritty movies" | `viewer_experience` (tone_self_seriousness) | Spectrum — realism/rawness varies. |
| "date night movies" | `watch_context` (watch_scenarios) | Viewing context, not binary movie attribute. |
| "turn your brain off movies" | `watch_context` (self_experience_motivations) + `viewer_experience` (cognitive_complexity) | Viewing context. |

### Derived at query time (not movie attributes)

| User concept | How to handle |
|-------------|--------------|
| "critically acclaimed" | `reception_score` threshold or `movie_awards` presence |
| "underrated / hidden gems" | High `reception_score` + low `popularity_score` (anti-correlation) |
| "cult classics" | Emerges from audience engagement patterns — not classifiable at ingestion |
| "trending / popular right now" | Redis trending set + `popularity_score` |
| "classics / iconic movies" | `reception_score` + `popularity_score` + age (dynamic system-level priors in Phase 0) |

---

## V3 Ideas (Post-V2)

Ideas that are explicitly scoped to a future V3 iteration, after the core V2
architecture is validated.

### Variable Queries for Truly Variable Intent

When the user's intent is genuinely variable (ambiguous queries where multiple
interpretations are valid), V3 should support variable query execution — the
top-level LLM generates a full query understanding AND a quick human-readable
label for each interpretation. These labels get passed back to the user alongside
the results, so the user sees "Showing results for: emotional dramas" or competing
interpretation options. This extends the multi-interpretation branching concept
(currently deferred to V2+) with the added UX of transparent intent labeling.

### Document Chunking for Vector Matching

If vector matching quality remains insufficient after the structured-label
embedding improvement, V3 can explore document chunking — splitting each movie's
per-space text into multiple smaller chunks, each embedded separately. This would
allow per-attribute retrieval granularity (e.g., a chunk specifically about "twist
ending" for The Sixth Sense) rather than compressing all attributes into a single
vector. Tradeoff: significantly increases storage and retrieval complexity (8
spaces × N chunks per movie × 150K movies).

### Franchise Search Flow as Independent Search Type

"Movies like xyz" should route to a dedicated similarity search flow. That flow
would use franchise membership, vector similarity across spaces, and
distinctive-vs-generic decomposition as described in the similarity queries
section above.
