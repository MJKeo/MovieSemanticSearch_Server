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
- Universal quality prior: "comedies" really means "well-received comedies I'd enjoy"
- Mainstream accessibility bias (unless query signals otherwise)
- Language/availability assumptions
- Should be handled as a constant prior, not per-query classification

---

## Proposed Pipeline Architecture

```
Phase 0: Query Understanding (restructured)
├── Classify components into deal-breakers vs preferences
├── For each deal-breaker: which retrieval channel fulfills it?
│   (lexical for entities, vector for semantic attributes,
│    metadata for structured fields like date/genre/runtime)
├── For each preference: which scoring signal evaluates it?
├── Resolve conflicts with any UI-set hard filters
│   (UI filters take precedence in case of conflict)
└── Detect empty deal-breaker set → fall back to broad retrieval mode

Phase 1: Deal-Breaker Retrieval (conjunctive)
├── Apply hard metadata filters (date, genre, runtime, etc.)
├── Lexical retrieval for entity deal-breakers (franchise, actor)
├── Targeted vector retrieval for semantic deal-breakers
│   (narrow, focused queries — not the expansive rephrasing)
├── Candidate pool = intersection of all channels
│   (threshold + flatten: above threshold = qualifies)
└── If pool is too thin → flag for Phase 4

Phase 2: Preference Scoring (additive — now appropriate)
├── Score each candidate on preference attributes
│   (vector similarity, metadata attributes, or both)
├── Additive combination is correct here because all candidates
│   already satisfy the deal-breakers
└── Implicit quality prior (reception, mainstream accessibility)

Phase 3: Result Assembly
├── Primary results: Phase 2 ranked output
├── Trim at natural score gap or fixed depth
└── If primary set is thin → trigger Phase 4

Phase 4: Exploratory Extension (conditional)
├── Run broader search (current-style expansive queries)
├── Exclude movies already in primary set
└── Weave or append as "you might also like"
```

---

## Phase 0: Query Understanding Output Structure

The LLM query understanding step needs to produce a dependency hierarchy rather than
parallel weights:

```json
{
  "deal_breakers": [
    {
      "attribute": "franchise",
      "value": "marvel",
      "channel": "lexical"
    },
    {
      "attribute": "narrative_technique",
      "value": "twist ending",
      "channel": "vector/narrative_techniques"
    }
  ],
  "preferences": [
    {
      "attribute": "tone",
      "value": "dark and gritty",
      "channel": "vector/viewer_experience"
    },
    {
      "attribute": "status",
      "value": "iconic",
      "channel": "vector/reception"
    }
  ],
  "implicit": ["well-received", "mainstream accessibility"]
}
```

This replaces the current flat channel weight system (vector_relevance,
lexical_relevance, metadata_relevance) with structured intent classification.

---

## Deal-Breaker Retrieval: Threshold + Flatten

For semantic deal-breakers retrieved via vector search, raw cosine similarity is a
spectrum. But deal-breakers need to be gates, not rankings. Proposed approach:

**Once a candidate passes the deal-breaker retrieval threshold, its deal-breaker
score is flattened to 1.0 (pass) or 0.0 (fail).** Stop treating cosine similarity as
a ranking signal for that attribute and use it only as a gate. Then preferences rank
within the passing set.

This solves the "Christmas problem": Home Alone and Die Hard both pass the christmas
gate — then you rank by critical acclaim. Without flattening, "aggressively
christmas" Hallmark films inflate scores through embedding density, and movies that
are clearly christmas movies but have rich multi-dimensional embeddings get penalized.

### Threshold Options

- **Score-distribution-based:** Find the natural gap/elbow in the retrieved
  candidates' similarity scores (similar to the derivative analysis already used for
  quality scoring thresholds in the ingestion pipeline)
- **Relative to top:** e.g., anything above 70% of the top candidate's score
  qualifies
- **LLM-adjudicated at the boundary:** For candidates near the threshold, ask an LLM
  "is this a christmas movie?" — only needed for the fuzzy boundary, not all 500
  candidates

---

## Subquery Generation Changes

Current subquery prompts generate expansive synonym-rich rephrasing to maximize
recall. For deal-breaker retrieval, we'd want **narrow, focused queries** that
precisely match the deal-breaker attribute without dilution.

For preference scoring, the current expansive approach may still be appropriate since
we're ranking rather than gating.

This might mean two different subquery modes or prompts — one for retrieval, one for
scoring.

---

## Graceful Degradation Strategies

Three approaches considered for handling queries where strict constraint satisfaction
yields too few results:

### Option A: Tiered Results
"Top matches" (all constraints) + "You might also like" (most constraints). Honest
and transparent, but changes the API surface and pushes complexity into presentation.
Requires the system to know WHICH constraint was relaxed.

### Option B: Progressive Relaxation
Try strict first, loosen if too few results. Risk: latency from re-running searches.
And "too few" is a judgment call — 3 great results might be better than 25 mediocre.

### Option C: Weighted-AND Scoring
`score = preference_score * deal_breaker_score^k` where k is large. Movies missing
deal-breakers get nearly-zeroed but not completely, so near-misses appear at the
bottom. Preserves a single ranked list without tiers.

**Current leaning:** Tiered approach via Phase 3/4 of the pipeline. Phase 3 produces
primary results (all deal-breakers met), Phase 4 produces exploratory results (broader
search, weaved in or appended). This gives the presentation layer clear tiers without
requiring re-ranking math.

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

### Pure-Vibe Queries (no deal-breakers)
When Phase 0 produces an empty deal-breaker set, skip Phase 1 and go straight to
broad retrieval + preference scoring. Essentially the current system, which handles
this class well.

### Negation-Heavy Queries
Can't retrieve by what movies AREN'T. Retrieve broadly on the positive signals (genre,
vibe), then post-filter using metadata or LLM evaluation for the negated attributes.
Negations are more naturally modeled as hard filters than as retrieval queries.

### Similarity Queries ("like Inception")
Could decompose the reference movie into its actual metadata attributes and determine
which are most distinctive (vs generic). Use the distinctive attributes as deal-
breakers and generic ones as preferences. The anchor vector still provides a broad
similarity signal as a safety net.

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
