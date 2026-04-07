# Types of Searches

Categorizing the distinct types of queries the system needs to handle. These form a
spectrum, but defining clear buckets helps ensure each category has an appropriate
retrieval strategy.

---

## Category 1: Multi-Constraint Semantic Queries

**Pattern:** Multiple semantic attributes required simultaneously.

**Examples:**
- "iconic twist ending" — twist (deal-breaker) + iconic (preference)
- "critically acclaimed christmas movies" — christmas (deal-breaker) + acclaimed (preference)
- "dark gritty marvel movies" — marvel (deal-breaker) + dark/gritty (preference)
- "funny horror movies from the 80s" — horror + 80s (deal-breakers) + funny (preference)
- "underrated foreign thrillers" — foreign + thriller (deal-breakers) + underrated (preference)

**What makes this distinct:** Users require ALL attributes but the attributes have
different structural roles. Some are gates (the movie MUST have this), others are
ranking signals (more of this = better). The current system treats them all as
parallel ranking signals with different weights.

**Key challenge:** Determining which attributes are deal-breakers vs preferences.
The same attribute can change role depending on context:
- "thriller with a twist ending" — thriller is deal-breaker, twist is preference
- "iconic twist ending" — twist is deal-breaker, iconic is preference
- "twist ending" alone — twist is deal-breaker, no preferences

**Current system failure mode:** Additive scoring favors movies exceptional at one
attribute over movies good at all attributes. Results satisfy subsets of the query
rather than the full conjunction.

---

## Category 2: Entity-Centric Queries

**Pattern:** Named entities (people, franchises, studios) as the primary constraint.

**Examples:**
- "leonardo dicaprio movies"
- "marvel movies"
- "A24 horror films"
- "movies like The Godfather"
- "Christopher Nolan's best work"
- "Studio Ghibli films for adults"

**What makes this distinct:** The primary constraint is a concrete, verifiable entity
handled by lexical search (for people/characters) or metadata filters (for studios,
franchises). There's a clear binary pass/fail: the movie either features DiCaprio or
it doesn't.

**Current system strength:** Lexical search handles entity matching well. The main
issue is when entity results need to be cross-filtered with semantic preferences
(like "best" or "for adults") — that's where it becomes a multi-constraint query.

**Current system failure mode:** When the entity constraint should dominate but the
additive weights give too much influence to vector similarity. Or when the LLM query
understanding generates overly creative subqueries that dilute the entity focus.

---

## Category 3: Pure Vibe / Experience Queries

**Pattern:** No concrete constraints, just a desired feeling or experience.

**Examples:**
- "something cozy for a rainy day"
- "I want to feel inspired"
- "background movie while working"
- "need a good cry"
- "fun popcorn movie"
- "turn my brain off"
- "date night movie"

**What makes this distinct:** There are NO deal-breakers. The entire query is
preferences and implicit expectations. Any movie that delivers the right experience
is a valid result regardless of genre, era, or other attributes.

**Current system strength:** This is where additive scoring across viewer_experience
and watch_context vectors actually works well. The system is designed for this type of
query.

**Challenge for new system:** Must detect an empty deal-breaker set and fall back to
broad retrieval with preference-based ranking rather than trying to force the
deal-breaker pipeline.

---

## Category 4: Negation-Heavy Queries

**Pattern:** Constraints defined primarily by what the user DOESN'T want.

**Examples:**
- "not animated, not too long, not depressing"
- "horror but no jump scares and no gore"
- "romcom that isn't cheesy"
- "sci-fi that's not action-heavy"
- "something good but not too popular — I've seen all the obvious ones"

**What makes this distinct:** You can't retrieve candidates by what they aren't. The
system needs to fetch broadly and filter down, which is the reverse of the
deal-breaker → preference flow.

**Challenge:** Embedding models don't reliably handle negation in cosine similarity.
"no jump scares" actually increases similarity with vectors containing "jump scares."
The current system's subquery prompts try to translate negations into positive
alternatives ("no flashbacks" → "linear chronology"), but this is imperfect.

**Possible approach:** Positive retrieval for the genre/vibe + post-retrieval
filtering using metadata or LLM-based evaluation for the negated attributes.

---

## Category 5: Similarity / Comparison Queries

**Pattern:** A reference movie (or set of movies) defines the desired results.

**Examples:**
- "movies like Inception"
- "something similar to Parasite but funnier"
- "if you liked The Dark Knight"
- "Wes Anderson meets horror"
- "the Korean Parasite but the original Spanish version" (reference for recall,
  not actual similarity search)

**What makes this distinct:** The "deal-breaker" is "similar to X" but similar along
which dimensions is ambiguous. Users typically mean multiple dimensions simultaneously
(structure, tone, complexity, theme) making this an implicit multi-constraint query.

**Challenge for new system:** Decomposing the reference movie into specific
attributes, deciding which are deal-breakers vs preferences. "Like Inception" probably
means: complex narrative structure (deal-breaker) + mind-bending (preference) +
visually ambitious (preference). But different users might mean different aspects.

**Possible approach:** Anchor vector for broad similarity (captures overall identity),
then use the decomposed attributes from the reference movie to weight specific vector
spaces. The reference movie's actual metadata could inform which attributes matter
most (its most distinctive features vs its generic ones).

---

## Category 6: Discovery / Browsing Queries

**Pattern:** Broad exploration with loose criteria, emphasis on surfacing
interesting or unexpected results.

**Examples:**
- "what's good right now"
- "trending movies"
- "hidden gems from 2024"
- "best movies I probably haven't seen"
- "surprise me"

**What makes this distinct:** There's minimal semantic retrieval to do — the value
is in curation and personalization. "Hidden gems" implies high quality + low
popularity, which is a metadata-layer operation more than a vector search problem.

**Current system approach:** Trending/popular movies are handled via the metadata
scoring channel with explicit trending preference extraction. This works for simple
cases but breaks down for "hidden gems" which requires anti-popularity filtering.

---

## Cross-Cutting Concerns

### Implicit Expectations

Every category carries implicit expectations that the user doesn't state:
- Quality: "comedies" really means "good comedies I'd enjoy"
- Accessibility: most users implicitly want well-known or mainstream-accessible films
- Recency bias: unless specifying era, users often prefer more recent results
- Language: English-language or at least widely available with subtitles

These should be handled as a universal quality prior, not as per-query classification.

### Mixed-Category Queries

Many real queries span categories:
- "something like Inception but cozy" — similarity + vibe
- "marvel movies, surprise me" — entity + discovery
- "critically acclaimed Korean thrillers, not too gory" — multi-constraint + negation

The system needs to handle category blending, not force each query into exactly one
bucket. The deal-breaker / preference / implicit hierarchy should work across
categories — the categories above describe what the deal-breakers and preferences
typically look like, not rigid processing modes.
