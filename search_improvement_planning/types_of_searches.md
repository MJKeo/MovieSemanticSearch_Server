# Types of Searches

Categorizing the distinct types of queries the system needs to handle. Organized
into simple queries (single retrieval strategy), complex queries (require
composing multiple retrieval strategies), and V2-specific edge cases (query
patterns that require special handling under the new architecture). 17
categories total: 8 simple, 6 complex, 3 V2 edge cases. Categories carry notes
on important subtypes and edge cases identified through query analysis.

---

## Simple Queries

Each of these has a single, clear retrieval strategy.

---

### 1. Known Movie Identification

> **Finalized routing decision (updated):** The step 1 exact title flow is
> restricted to literal title matches only (exact titles, misspellings, partial
> titles, alternate official titles, single-movie abbreviations). Fragmentary
> recall queries (plot descriptions, scene descriptions, partial cast recall)
> route to the standard flow instead — the standard pipeline handles
> description-based identification better than a small routing LLM inferring
> titles from descriptions. The original broader definition below is preserved
> for reference as a query type, but its routing has changed.

User has a specific movie in mind and wants to find it. They may know the title
exactly, approximately, or not at all — providing plot fragments, scene
descriptions, or partial cast recall instead.

**Examples (title-based — routes to exact title flow):**
- "The Shawshank Redemption"
- "Lemonade Mouth"
- "Inception"

**Examples (description-based — routes to standard flow):**
- "That movie where the guy draws the woman on the ship and then it sinks"
- "The one where Bill Murray relives the same day over and over"
- "That Leonardo DiCaprio movie in the snow with the bear"

**Behavior for title matches:** DB search for all exact matches on the
extracted title string. If no matches, the user sees "we don't have that
title" with no fallback to standard flow.

**Behavior for description-based queries:** These enter the standard flow
and are handled through the normal dealbreaker/preference decomposition.
The standard pipeline's multi-channel retrieval (entity, metadata, semantic)
is better suited to identifying movies from descriptions than a single
routing LLM.

**Routing note:** This is one of the major flows Step 1 should route into before
the standard dealbreaker/preference decomposition. If the user's intent is
clearly "find this one movie" and they provide the title, the system should not
force the query through the full standard pipeline first.

---

### 2. Title Substring Search

User wants movies whose titles contain a specific word or phrase.

**Examples:**
- "Movies with 'love' in the title"
- "Movies with 'star' in the title"
- "Movies with 'death' in the name"

**Behavior:** SQL LIKE/ILIKE on the title field. Similar to title lookup but
without the similar-movies secondary results — the user is browsing titles, not
looking for a specific movie.

---

### 3. Entity Lookup

User is searching for movies associated with specific people, studios, or
franchises. No semantic or metadata qualifiers beyond the entities themselves.

**Examples:**
- "Leonardo DiCaprio movies"
- "A24 films starring Florence Pugh"
- "Warner Brothers films"
- "Marvel studios films"
- "Movies starring the Rock and John Cena"

**Behavior:** Lexical search on entity fields. When multiple entities are
specified, rank by percentage of entities matched (a movie matching both "the
Rock" and "John Cena" ranks above one matching only one of them), with quality
sorting within each match-count tier.

**Note on franchise enumeration:** Queries like "all the Fast and Furious movies"
or "every movie in the MCU" are structurally entity lookups, but the user's
expectation differs — they want completeness and chronological ordering, not
quality-ranked top-N. Both cases involve lexical filtering followed by an
ordering step, but the ordering signal (release date vs. quality) and success
metric (exhaustive recall vs. best subset) diverge. The system needs to
distinguish between "best X movies" and "all X movies" intent for franchise
entities specifically.

**Note on franchise entity resolution:** Franchise is a multi-level concept,
and the entity extraction step needs to distinguish which level the user means:
- **Studio level:** "Marvel Studios films" → production company match
- **Brand level:** "MCU movies" → specific cinematic universe
- **Character level:** "Spider-Man movies" → includes Sony Spider-Man, not MCU

"Marvel" should route to studio/brand, not match any movie with a Marvel
character in the cast. "Spider-Man movies" should include all Spider-Man films
regardless of studio. Entity extraction needs to identify the franchise level
and route accordingly.

**Note on actor prominence:** For multi-entity queries like "movies starring
the Rock and John Cena," the current system treats actor presence as binary.
But a starring role should score higher than a 30-second cameo. **Decided:**
role-specific posting tables with billing_position + cast_size stored on
actor postings. Three query modes (exclude non-major / boost by position /
binary) controlled by Phase 0 based on query language. "Starring" triggers
exclude-non-major mode. Default is boost-by-position. When person role isn't
stated ("Spielberg films"), boost the implied role (director) but still
include other roles (producer) at lower weight.

**Note on awards as entities:** Queries like "Oscar-winning films" or "Cannes
Palme d'Or winners" are structurally entity lookups where the entity is an
award. **Decided:** new `movie_awards` table with ceremony, category, and
outcome fields. Supports inverse lookup (given award → find movies). Award
win ceremony text also included in reception vector for vaguer queries like
"award-winning thriller." Nominations stay out of the vector and are handled
deterministically via `movie_awards`.

---

### 4. Pure Metadata Filter

Every attribute in the query maps to a deterministic, structured filter — genre,
era, runtime, rating, streaming availability, etc. No semantic interpretation
needed.

**Examples:**
- "Comedies"
- "80s action thrillers"
- "Movies under 90 minutes"
- "Horror movies from 2020"
- "PG-13 animated movies"

**Behavior:** SQL filters on structured metadata fields, results ranked by
quality. It's fine if few results show up — this is a browsing/filtering
operation where quality expectation is implied.

**Note:** Queries that look like pure metadata but contain concepts without clean
metadata fields (e.g., "indie horror" — there's no `is_indie` boolean) are
actually semantic queries and belong in other categories.

**Note on expanded metadata fields:** The data layer redesign adds several new
filterable fields that expand what counts as "pure metadata": country of origin
(enables "Korean movies"), box office bucket (enables "box office hits"), source
material type as enum array (enables "based on true stories," "book
adaptations"), and production medium via keyword search (enables "animated
movies," "stop motion"). Previously these required semantic retrieval; now they
can be handled deterministically.

---

### 5. Single-Concept Semantic Query

User is searching for a single concept or experience that maps to one (or at
most two) vector spaces. No named entities, no metadata filters, no
qualifications — just one idea.

**Examples:**
- "Date night" → watch_context
- "Movie where toys come to life" → plot_events
- "Movies with a great soundtrack" → production
- "Turn my brain off" → viewer_experience
- "Something cozy for a rainy day" → watch_context / viewer_experience

**Behavior:** Vector search on the 1-2 most relevant spaces, results ranked by
similarity score. This is where the current system already performs reasonably
well.

**Note on retrieval strategy:** This is the primary query type where vector
retrieval remains the candidate generator in the revised architecture. No
deterministic anchors exist (no entities, no metadata filters, no keywords),
so vector search is the only option. The structured-label embedding improvement
(see new_system_brainstorm.md) is particularly important for this query type
since retrieval quality directly determines result quality.

**Note:** Scene/moment queries ("movie with an epic training montage," "movie
with a great car chase") fit here. Even though the user is asking about a
contained element rather than the movie's overall identity, the retrieval
pattern is the same — plot_events embeds full synopses that capture scene-level
detail, reception's praised_qualities flags standout moments, and watch_context's
key_movie_feature_draws captures scenes-as-draws. Retrieval quality may vary
based on how prominently the scene features in the embedded text, but the
structural pattern is standard semantic search.

---

### 6. Reference Movie Similarity

User provides a specific movie as a reference point and wants similar movies,
without further qualifications.

**Examples:**
- "Movies like Inception"
- "Something similar to Parasite"
- "If you liked The Dark Knight"

**Routing note:** This is a separate major flow from the standard search
pipeline. Pure `"movies like <movie>"` queries should route here directly.

**Behavior:** Identify the reference movie's tmdb_id, retrieve its stored
vectors, find the N nearest neighbors across a predetermined set of vector
spaces with fixed weights (e.g., plot_analysis carries more weight for
similarity than plot_events). The hard part is correctly resolving which movie
the user means when titles are ambiguous — multiple movies may share a title
but the user almost certainly means the most well-known one.

~~**Implementation priority:** This should be routed to a different search flow
entirely and handled AFTER the initial version of search V2 is validated. The
core V2 architecture (deal-breaker/preference decomposition, deterministic
retrieval + semantic rescore) is the priority; similarity search is a distinct
enough flow to build separately.~~

> **Updated:** Similarity search is now a first-class step 1 route, not
> deferred. It is handled as part of the V2 step 1 flow routing alongside
> exact title search and the standard flow.

**Exception:** `"movies like <movie> but <qualifiers>"` belongs to the standard
flow, not this pure similarity flow. Once explicit qualifiers are present, the
query is better treated as an interpreted constrained search. Using the LLM's
parametric knowledge of mainstream reference movies is an acceptable speed and
coverage tradeoff for understanding the intended traits.

**Franchise as similarity signal:** "Similar movies" also includes entries to the
same franchise — a user asking for "movies like The Dark Knight" expects other
Batman/DC movies alongside thematically similar non-franchise films. Franchise
membership from `movie_franchise_metadata` should be checked as an additional
structured attribute alongside vector similarity.

**Note on distinctive vs generic decomposition:** Not all of a reference
movie's traits are equally important for similarity. Inception's mind-bending
nested reality structure is distinctive; its "action movie" genre is generic.
Similarity search should weight traits by how much they differentiate the
reference from the average movie. Traits shared by thousands of movies
(action, drama, English-language) are weak similarity signals; traits shared
by few movies (nested dream worlds, unreliable reality) are strong ones. This
could inform which vector spaces get higher weights for a given reference
movie — spaces where the reference is an outlier carry more similarity signal
than spaces where it's average.

**Note on weaving similarity into other searches:** When any search query
(not just explicit "movies like X") produces a strong single-movie match at
the top (large score gap between #1 and #2), consider running a secondary
similarity search on that movie and presenting those results as a "similar
to your top match" tier below. Someone searching "High School Musical" gets
HSM first, then movies like Lemonade Mouth below — combining identification
with discovery.

---

### 7. Discovery / Curation

User wants recommendations driven by popularity, trending status, or general
quality — not a specific concept or attribute.

**Examples:**
- "Trending now"
- "Popular movies right now"
- "What's good"
- "Hidden gems"
- "Best movies I probably haven't seen"
- "Surprise me"
- "Movies every film student should watch"
- "Essential sci-fi"
- "Movies that changed cinema"

**Behavior:** Primarily a metadata/curation operation — quality scores,
popularity metrics, trending data. Minimal semantic retrieval. "Hidden gems"
is a special case requiring high quality + low popularity, which is an
anti-correlation filter.

**Note on trending candidate injection:** For explicitly trending queries
("trending now," "popular movies right now"), directly fetch the list of
trending movies and consider them all to be candidates, so long as they pass
any active metadata filtering. The candidate pool comes from the trending set
in Redis rather than from vector retrieval. Vector search is unnecessary — the
user is asking about temporal popularity, not semantic content. Rank the
trending set by trending signal (recency, popularity velocity) with system-level
priors on top.

For hybrid standard-flow cases like "trending horror movies," treat trending as
its own deterministic candidate source alongside lexical, metadata, and keyword
sources. This aligns trending with the broader V2 rule that reliable candidate
sets should come from deterministic sources when possible.

**Note:** Canon/curriculum queries ("movies every film student should watch,"
"essential horror") are a subtype where the curation signal is cultural
influence and historical importance rather than popularity or general acclaim.
Reception and watch_context spaces carry partial signal for this ("culturally
iconic" is an explicit watch_context term, reception summaries capture
critical significance), but canonical influence — a movie's impact on
filmmaking itself — is entangled with general acclaim in our current
embeddings. A movie can be canon-essential without being conventionally
"good" (Triumph of the Will, The Room).

**Note on quality inversion:** "So bad it's good" queries ("a movie that's
so bad it's hilarious," "best worst movies") are a subtype where the
default quality-oriented priors actively work against correct results. Low
conventional quality IS the appeal — the user wants The Room, Troll 2,
Birdemic. The signal is a specific combination: low critic scores + high
audience engagement/cult following + reception language indicating ironic
enjoyment. The system must recognize when the quality-oriented priors should be
suppressed or inverted based on query interpretation.

This should be kept distinct from queries like "hidden gems" or "underrated,"
which usually still want good movies but with lower popularity or lower
mainstream exposure.

---

### 8. Superlative / Degree-of-Attribute

User wants the most extreme example of a specific attribute — not just movies
that have the attribute, but the ones that have it the most.

**Examples:**
- "The scariest movie ever made"
- "Funniest comedy of the 2010s"
- "Most visually stunning movie"
- "Saddest movie ever"
- "Most mind-bending movie"

**Behavior:** The defining attribute is simultaneously the retrieval concept
AND the ranking signal. Retrieve movies matching the attribute (e.g.,
viewer_experience for "scary"), then rank by how strongly they exhibit it.
System-level priors should still apply on top to avoid surfacing obscure movies
that are one-dimensionally extreme.

**Key challenge:** This conflicts with the threshold + flatten approach for
deal-breakers. If "scary" gets flattened to pass/fail, the ranking the user
is asking for disappears. But if raw cosine similarity is the ranking signal,
the embedding density problem returns — movies whose entire identity is
"scary" outrank movies that are terrifying but also have rich plot, character,
and thematic depth. The system needs a mode where the gate attribute's
similarity score is preserved as a ranking signal without letting embedding
density dominate.

**Note on banded constraints:** Queries like "a sad movie that won't completely
destroy me" or "something tense but still fun" appear to combine a superlative
with a negation/ceiling. But these are more naturally handled as a semantic
interpretation problem: "sad but not devastating" translates to a positive
concept like "cathartic" or "bittersweet," and "tense but still fun" translates
to "thrilling." The LLM interpretation step (Phase 0) should translate these
banded constraints into positive query concepts rather than the retrieval layer
trying to score with a floor and ceiling on the same axis.

---

## Complex Queries

These require composing multiple retrieval strategies. The key design challenge
is determining how the components relate to each other: does one anchor the
search while others filter/rank (sequential), must all constraints be satisfied
(intersection), or can any suffice (union)?

---

### 9. Sequential: Deal-Breaker Anchor + Preference Qualifier

One attribute is the core constraint (deal-breaker) and the other qualifies
or ranks the results. There's a clear ordering — retrieve first, rank second.

**Examples:**
- "Iconic twist endings" — retrieve movies with twist endings, rank by how
  iconic they are
- "Critically acclaimed movies about female empowerment" — retrieve movies
  about female empowerment, rank by critical acclaim
- "Underrated foreign thrillers" — retrieve foreign thrillers, rank by
  the "underrated" signal (high quality + low popularity)
- "Movies that are better than the book" — retrieve adaptations via
  `source_material_types` enum filter (NOVEL_ADAPTATION), rank by critical
  reception. Deterministic deal-breaker now, no vector matching needed.
- "Best trilogies" — retrieve franchise members via
  `movie_franchise_metadata`, filter to the target lineage/subgroup and
  non-remake continuation rows, rank by reception quality.
  Deterministic deal-breaker now.

**Key challenge:** Determining which attribute is the deal-breaker and which
is the preference. The narrower, more defining attribute should anchor
retrieval. "Iconic" applied first would give you The Godfather, Casablanca,
etc. — then you'd be fishing for twist endings among those, which misses most
twist movies.

**Note on aggregate queries:** "Best trilogies" and similar franchise-level
queries ("best movie series to binge") fit this pattern at the individual
movie level — retrieve franchise members, rank by quality. However, the user
may expect franchise-level results (the trilogy as a unit, not individual
films). Presentation-layer grouping may be needed to collapse individual
movie results into franchise-level recommendations.

---

### 10. Intersection: Multiple Deal-Breakers

Multiple attributes are all required simultaneously. The user wants movies
that satisfy ALL constraints — a movie excelling at only one is a miss.

**Examples:**
- "Family-friendly musicals" — must be both family-friendly AND a musical;
  a dark musical or a family-friendly drama both fail
- "A movie that satisfies a psych-thriller lover and their partner who likes
  artistically driven, well-crafted movies" — must partially satisfy both
  people's somewhat conflicting preferences
- "Funny horror movies" — must be genuinely funny AND genuinely horror;
  a comedy that's slightly creepy or a horror with one joke doesn't cut it
- "Something the whole family can watch but won't bore the adults" — the
  constraints are audience-defined rather than attribute-defined, but reduce
  to attribute intersection after LLM translation: "family-friendly" →
  metadata rating filter + watch_context, "won't bore adults" →
  viewer_experience (sophistication, cognitive complexity)

**Key challenge:** Syntactically identical to sequential queries — "funny
horror" looks just like "iconic twist endings" (adjective + noun). But
"funny" is a deal-breaker (a non-funny horror movie fails) while "iconic"
is a preference (a non-iconic twist movie is still acceptable). The system
needs to understand this distinction.

**Retrieval strategy (revised by empirical testing):** Cross-channel
intersection at retrieval time fails — "funny horror" had zero intersection
between vector candidate sets. The correct approach: generate candidates from
the most reliable deterministic channel (genre=horror via metadata filter),
then score on the semantic deal-breaker ("funny") via cross-space rescoring in
Phase 2. The intersection happens at scoring time, not retrieval time. See
new_system_brainstorm.md "Deterministic Retrieval + Semantic Rescore."

---

### 11. Union: Any Constraint Suffices

Multiple attributes where satisfying ANY of them qualifies a movie. Results
from each branch are merged.

**Examples:**
- "Movies that take place in medieval times or the distant future" — either
  setting independently qualifies a movie
- "Something with great action scenes or a really compelling mystery" — either
  quality makes the movie worth showing

**Behavior:** Run independent retrievals for each branch and merge results.
Movies matching both branches rank highest. Relatively rare query pattern
compared to sequential and intersection.

---

### 12. Cross-Channel Composition

Query components span different retrieval channels (lexical, metadata, vector)
and need to be combined.

**Examples:**
- "Tom Cruise 80s movies" — entity (lexical) + era (metadata)
- "Jim Carrey but only his more serious movies" — entity (lexical) +
  tone qualifier (semantic/vector)
- "Dark, gritty marvel movies" — franchise (lexical) + tone (semantic/vector)
- "Disney animated classics" — studio (lexical) + medium (metadata) +
  quality/age qualifier (semantic + metadata)
- "Christmas classics" — Christmas-ness (semantic, deal-breaker) + classic
  status (quality/reception + age, preference)

**Key challenge:** These are sequential or intersection queries where the
components happen to live in different retrieval channels. The composition
logic (sequential vs. intersection) still applies — the added complexity is
that Phase 1 must execute different retrieval mechanisms and combine their
results. "Dark gritty marvel movies" retrieves marvel movies via lexical
search, then scores them on darkness/grittiness via vector search.

**Retrieval strategy (revised by empirical testing):** This is the query type
that most directly benefits from the deterministic-retrieval-then-semantic-
rescore architecture. Empirical testing of "dark gritty marvel movies" showed
that vector results for "dark and gritty" miss key movies like Winter Soldier.
The correct approach: retrieve Marvel movies via lexical search (deterministic),
then score on "dark and gritty" via cross-space rescoring. The deterministic
channel generates reliable candidates; the semantic channel scores them.

**Note on similarity + metadata:** Queries like "something like Inception
that's on Netflix" combine reference movie similarity (vector) with platform
availability (metadata). This creates an ordering dilemma absent from entity
+ semantic cases: do you run similarity search globally then filter by
platform (better similarity matches, risk of empty results after filtering),
or filter to the platform catalog first then find the most similar within
that pool (guaranteed availability, constrained similarity neighborhood)?
The choice affects result quality because pre-filtering changes which
similarity neighbors are reachable.

**Note on entity + divergence:** Queries like "a serious Will Ferrell movie"
or "Jim Carrey but only his more serious movies" are cross-channel
(entity lexical + tone semantic), but carry a heavier interpretation burden
than other cross-channel queries. The system must understand the entity's
*typical* genre profile to recognize that "serious" is a divergence request
— the query is implicitly "Will Ferrell NOT doing comedy." This makes the
Phase 0 interpretation step load-bearing in a way it isn't for straightforward
cross-channel queries like "Tom Cruise 80s movies."

---

### 13. Negation-Modified Queries

Query includes constraints defined by what the user DOESN'T want. May be
negation-only or combined with positive attributes.

**Examples:**
- "Movies that aren't too scary" — negation only, implies the user wants
  something mildly thrilling or suspenseful
- "Movies not starring Arnold Schwarzenegger" — entity negation
- "Horror movie that doesn't have many jump scares" — positive anchor
  (horror) + negation modifier (no jump scares)
- "Movies with a great soundtrack that aren't actually about a musician" —
  positive anchor (great soundtrack) + negation modifier (not about musicians)
- "Comedies but not cheap immature jokes" — positive anchor (comedy) +
  quality negation (not lowbrow)
- "Rocky movies but not the one where he fights that Russian guy" — entity
  anchor (Rocky franchise) + specific exclusion (Rocky IV)

**Key challenge:** Embedding models don't handle negation in cosine
similarity — "no jump scares" actually increases similarity with vectors
containing "jump scares." Negations need to be handled as post-retrieval
filters rather than retrieval queries. The system retrieves on the positive
components and filters out the negated attributes afterward.

---

### 14. Intent Interpretation + Retrieval

User's query doesn't directly state movie attributes — the system must
interpret the user's intent through an LLM reasoning step before any
retrieval can happen. The LLM uses its training knowledge to translate the
query into concrete movie-attribute targets, then existing retrieval
patterns (semantic, metadata, lexical) execute on those targets.

**Examples:**
- "Movies like Inception but funnier" — reference movie similarity as
  anchor, LLM identifies "funnier" as a preference overlay
- "Wes Anderson visuals with Tarantino dialogue" — LLM decomposes two
  references into specific attribute extractions (visual style from one,
  dialogue style from the other)
- "I just went through a breakup" — LLM infers desired viewing experience
  from user's emotional state (cathartic crying? empowering recovery?
  comforting distraction?)
- "I need to feel something" — user state implies emotionally impactful
  movies, but the specific emotional direction is ambiguous
- "Spielberg meets Kubrick" — LLM must determine what each director
  represents (Spielberg's emotional warmth + Kubrick's visual precision?)
  and construct attribute targets from that

**Behavior:** Phase 0 carries heavier interpretive weight than other
categories. The LLM must translate non-attribute query language into
concrete retrieval targets, then the downstream pipeline uses whatever
pattern fits the translated intent (sequential, intersection, single-
concept semantic, etc.). The query's surface form varies widely — reference
movies, user emotional states, cultural shorthand — but the pipeline
pattern is the same: interpret first, then route to existing retrieval.

**Key challenge:** The interpretation step introduces ambiguity that
doesn't exist in other categories. "I just went through a breakup" has
multiple valid translations, and the LLM's choice determines the results
entirely. For multi-reference queries ("Wes Anderson meets horror"), the
LLM must map each reference to the right vector space — Wes Anderson's
visual style lives in production/narrative_techniques, while "horror" is
a genre/viewer_experience concept. The correctness of Phase 0's attribute
extraction is load-bearing in a way it isn't for queries that state
attributes directly.

**Note on multi-interpretation branching:** When Phase 0 detects genuinely
ambiguous intent (multiple valid translations with no strong signal
favoring one), the system should surface the competing interpretations
rather than forcing a single one. "I need to feel something" could mean
emotional catharsis, adrenaline, or intellectual stimulation — each
interpretation produces a different deal-breaker/preference structure and
different results. Present the user with clickable interpretation groups
("Did you mean: emotional movies | intense thrillers | mind-bending
films?") and let them choose. This should trigger only when Phase 0's
confidence in a single interpretation is low — most queries should resolve
to one interpretation confidently.

**Note on multi-audience queries:** "A movie that satisfies a
psych-thriller lover and their partner who likes artistically driven,
well-crafted movies" requires partially satisfying two conflicting
preference profiles simultaneously. This is structurally an intersection
query (type #10) but with the added complexity that the constraints come
from different audience profiles rather than from a single user's
requirements. The system needs to find movies in the overlap zone of two
different taste spaces — psychological intensity AND artistic craft. Phase
0 should decompose this into the two audience profiles and identify
attributes that could satisfy both.

---

### 15. All-Semantic / Pure-Vibe Queries

Every requirement in the query maps to a semantic concept with no deterministic
anchor. No entity, no metadata filter, no keyword match — the entire query is
vibes.

**Examples:**
- "Fun lighthearted movies with car chases"
- "Something cozy and heartwarming"
- "Intense edge-of-your-seat thriller vibes"
- "Movies that feel like a warm hug"

**Behavior:** Since no deterministic source can generate candidates, the query
enters the pure-vibe flow. All semantic "dealbreakers" become preferences.
Vector search generates candidates via individual searches per concept across
relevant spaces, with results unioned. Candidates are rescored by fetching
distances across all relevant spaces (not just the space where initially
retrieved). A minimum similarity threshold per space prevents noise from
weak matches.

**Key distinction from type #5 (single-concept semantic):** These queries have
multiple semantic axes that need to be evaluated independently. "Fun AND
lighthearted AND car chases" requires a movie to score well across multiple
concepts, not just one. Individual searches prevent signal dilution from
blending concepts into averaged embeddings.

**Note:** The step 1 LLM should consolidate genuinely synonymous concepts
before creating separate entries (e.g., "fun and lighthearted" might become
one preference if they target the same vector space and capture the same idea).
Distinct concepts that happen to target the same space should remain separate
(e.g., "fun" and "nostalgic" both target viewer_experience but capture
different qualities).

---

### 16. Semantic Exclusion on Non-Tagged Attributes

Query includes an exclusion requirement for a concept that isn't covered by any
deterministic data source (no keyword, no entity, no metadata field).

**Examples:**
- "Funny horror movies but not ones with clowns" — "clowns" isn't a keyword or
  tag in the system
- "Action movies without too much CGI" — CGI usage isn't a metadata field
- "Romantic comedies but nothing with cheating" — infidelity isn't a keyword
- "Thrillers without torture scenes" — torture isn't discretely tagged

**Behavior:** The exclusion cannot be hard-filtered. Instead, it's handled via
semantic elbow-threshold penalty: search the full corpus for the exclusion
concept, analyze the global score distribution to find the elbow, and penalize
candidates based on where they fall:
- Above elbow (genuinely matches concept) → harsh downrank
- Near elbow (uncertain match) → soft downrank
- Below elbow (no meaningful match) → no penalty

**Key challenge:** The penalty must calibrate against the GLOBAL distribution,
not relative to the candidate set. If no candidates actually contain clowns,
a relative approach would still penalize the most clown-adjacent candidate
(maybe a circus-themed horror with no actual clowns). The global search
establishes what "actually has clowns" looks like in absolute terms.

**Note on routing:** The step 1 LLM must know the keyword/concept tag
vocabulary to correctly identify that a concept like "clowns" isn't in the
deterministic vocabulary and must route to `semantic`. If the LLM incorrectly
routes to `keyword`, the search will silently miss all results. This is why
the full keyword vocabulary is included in the step 1 prompt.

---

### 17. Concept Without Deterministic Anchor

User states a requirement as a dealbreaker, but the concept only exists in
the semantic domain. The system does **not** demote it to a preference by
default. Instead, behavior depends on whether there are any non-semantic
inclusion dealbreakers alongside it.

**Examples:**
- "Movies with car chases" — no "car chase" keyword; becomes pure-vibe query
- "Zombie movies" (if "zombie" is not in keyword vocabulary) — no deterministic
  anchor; vector search is the only option
- "Movies about artificial intelligence" — if no AI keyword exists, semantic
  only

**Behavior:**
- If other deterministic inclusion dealbreakers exist, the semantic concept
  remains a semantic dealbreaker but scores only within the deterministically
  generated candidate pool.
- If all inclusion dealbreakers are semantic, the query becomes a pure-vibe
  query (type #15) and semantic dealbreakers generate the candidate pool.
- If there are zero inclusion dealbreakers (preferences only, or exclusions
  plus preferences), this does **not** trigger pure-vibe retrieval; candidate
  generation falls back to a browse-style top-K default-quality pool.

**Key insight:** The routing decision itself determines the confidence level.
Deterministic sources give binary, reliable results. Semantic sources give
fuzzy similarity scores. The system therefore lets semantic concepts score in
anchored flows, and only lets them generate candidates when the user's actual
inclusion dealbreakers are all semantic.

**Note:** This is a signal for where to expand the keyword vocabulary over
time. Every query where the LLM falls back to semantic for a concept that
users clearly expect to work as a filter represents a gap in the deterministic
data. Tracking these fallbacks can inform keyword taxonomy expansion.

---

## Cross-Cutting Concerns

### Implicit Expectations

Every category carries unstated expectations:
- **Quality:** "comedies" really means "good comedies I'd enjoy"
- **Accessibility:** most users implicitly want well-known or mainstream-
  accessible films unless their query signals otherwise
- **Recency bias:** unless specifying an era, users often prefer more
  recent results
- **Language/availability:** English-language or widely available with
  subtitles, watchable in the US

These should be handled as system-level priors applied across categories, not
as per-query classification.

**Caveat:** These priors are strong defaults, not universals. Specific query
types override them: "so bad it's good" can invert conventional quality, while
"hidden gems" and "underrated" should reduce mainstream/notability bias rather
than being treated as low-quality requests. The priors should be suppressible
when Phase 0 interpretation identifies these cases.

### Metadata-Semantic Boundary

Some concepts sit at the boundary between metadata and semantic:
- "Indie" — feels like metadata but has no clean database field; it's a
  concept requiring semantic retrieval
- "Critically acclaimed" — could be metadata (Metacritic > 70) or semantic
  (reception vector space)
- "Trilogy finales" — now partially structured via `movie_franchise_metadata`
  (`recognized_subgroups` + `lineage_position` + franchise identity), but
  "finale" specifically may
  still need semantic retrieval

The system needs to recognize when a surface-level-simple query actually
requires semantic retrieval because the relevant metadata field doesn't exist.

**Note:** The data layer redesign moves several concepts from the semantic side
to the metadata side: source material type (now enum), franchise membership
(now structured table), country of origin (now filterable), production medium
(now keyword-searchable), awards (now structured table). This shrinks the
semantic-only zone and enables more deal-breaker concepts to use deterministic
retrieval.

**Empirical update:** This boundary is now even more consequential than
originally understood. Testing showed that semantic concepts cannot reliably
generate candidates via vector retrieval (see current_search_flaws.md #14).
Every concept that can be moved to the deterministic/metadata side gains
reliable candidate generation instead of unreliable vector-based generation.
The more concepts that live on the deterministic side, the more query types
can use the deterministic-retrieval-then-semantic-rescore architecture that
empirically produces better results.

### Temporal-Establishment Terms

Some metadata-adjacent concepts carry implicit temporal signals that are easily
missed. "Classics," "iconic," "legendary," "timeless," and "essential" all
imply cultural staying power — quality PLUS age/establishment. The query
understanding step needs to translate these into soft date preferences (bias
toward older) rather than hard date filters or nothing at all. "Disney animated
classics" failing to bias toward older films is a concrete failure case of this
gap.

### NLP-Extracted Constraint Imprecision

Users are frequently imprecise with metadata constraints in natural language.
"Classic 80s action movies" should include Terminator 2 (1991). "Leonardo
DiCaprio boat movie from the 2000s" clearly means Titanic (1997). The system
uses a three-tier constraint strictness model to handle this — UI-set filters
are strict, NLP-extracted metadata uses generous gates with preference decay,
and semantic attributes use vector thresholds. See new_system_brainstorm.md
"Three-Tier Constraint Strictness" for the full design.
