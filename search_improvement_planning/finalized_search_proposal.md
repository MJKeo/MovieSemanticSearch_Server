# Finalized Search Proposal (V2)

This document contains only finalized, committed-to decisions for the V2 search
system. Decisions are promoted here from the broader planning docs once fully
resolved.

---

## High-Level Architecture: Pipeline Overview

The V2 search system follows a multi-step pipeline for the **standard flow**
(steps 1–4). Two other flows — known-movie and reference-movie similarity —
are routed at step 1 and bypass steps 2–4.

1. **Flow Routing** — Classify the query into the correct major search flow.
2. **Query Understanding** — A single LLM extracts structured dealbreakers,
   preferences, and quality priors from the user's query.
3. **Query Translation** — Per-endpoint LLMs translate dealbreakers and
   preferences into complete query specifications. All LLMs run in parallel.
3.5. **Search Execution** — Dealbreaker searches execute as each step 3 LLM
   responds; preference queries await candidate assembly, then execute
   immediately.
4. **Assembly & Reranking** — Assemble candidate sets, tier by dealbreaker
   conformance, score by preference results and system priors, apply
   exclusions, produce final ranked results.

---

## Step 1: Flow Routing

Classifies the query into one of three major search flows before any
decomposition happens. May also produce multiple interpretations when a
query is genuinely ambiguous.

### Flow Definitions

**Exact title flow** — The user is providing the literal title of a movie
they want to find.

Includes:
- Exact titles: "The Shawshank Redemption", "Inception"
- Misspellings: "Shawshank Redemtion", "Incpetion"
- Partial titles where the user is clearly attempting the title:
  "Shawshank Redemption" (missing "The"), "Dark Knight" (missing "The")
- Alternate official titles: "Live Die Repeat" for "Edge of Tomorrow"
- Recognized single-movie abbreviations: "T2" for Terminator 2
- Title + specification (not qualification): "Good Will Hunting with Matt
  Damon" — extract just the title, ignore the specification
- Explicit title-search intent: if the user states they are searching by
  title (e.g., "the movie called Xyzzy"), route here even if we don't
  recognize the title

Does NOT include — even if the movie is easily identifiable:
- Plot descriptions: "that movie where the ship sinks"
- Scene descriptions: "the one with the bullet dodging"
- Cast/crew descriptions: "that Leonardo DiCaprio movie in the snow"
- Franchise acronyms that reference multiple movies: "LOTR", "HP", "MCU"

These all go to the standard flow. The standard search pipeline handles
description-based identification better than a small routing LLM guessing
titles from descriptions.

**Output:** The extracted title string. The DB is searched for all exact
matches. If no matches are found, the user sees a "we don't have that
title" message — there is no fallback to standard flow.

**Reference-movie similarity flow** — The user names a specific movie
title and asks for similar movies with **zero qualifiers**.

- "Movies like Inception" → similarity flow
- "Something similar to Parasite" → similarity flow
- "Inception style movies" → similarity flow
- "Movies like Inception but funnier" → **standard flow** (qualifier)
- "Scary movies like The Conjuring" → **standard flow** (qualifier)
- "Movies like Inception set in space" → **standard flow** (qualifier)
- "Movies like Inception and Interstellar" → **standard flow** (multiple
  reference movies require trait extraction and merging, which is
  interpretive work for step 2)

The rule is strict: anything beyond "similar to X" / "like X" / "X style
movies" constitutes a qualifier and routes to standard flow. Same title-
matching rules as the exact title flow apply to the reference movie name.

**Output:** The reference movie title string.

**Standard flow** — Everything else. This is the main constrained search
pipeline described in steps 2-4 below. Includes:
- Entity lookups: "Leonardo DiCaprio movies"
- Metadata filters: "80s comedies"
- Semantic/vibe queries: "cozy date night movie"
- Description-based movie identification: "that movie where the ship sinks"
- Qualified similarity: "movies like Inception but funnier"
- Multiple-reference queries: "movies like Inception and Interstellar"
- Cross-channel composition: "dark gritty Marvel movies"
- Superlatives: "scariest movie ever"
- Discovery: "trending movies", "hidden gems"
- Franchise searches: "Rocky movies", "all MCU films"

**"Movies like X but qualifiers" stays in the standard flow.** Once
explicit qualifiers are present, the query is no longer just "find nearest
neighbors to X." It becomes a standard interpreted search, where the LLM
can use its parametric knowledge of the reference movie as a fast way to
understand the intended traits.

### Interpretation Branching

Step 1 may produce multiple interpretations when a query is genuinely
ambiguous. Branching is **cross-flow** — a single query can produce
interpretations that route to different major flows.

**Branching bar:** An intelligent person would agree that the
interpretations are reasonably similar in likelihood. If one interpretation
is clearly more correct than the others, produce only that interpretation
— including alternatives would just be confusing. The goal is not "find
all possible interpretations" but "identify when there are multiple
equally reasonable readings of the request."

**Examples where branching applies** — movie titles that double as natural
language descriptions:
- "Scary Movie" → exact title (the 2001 parody) OR standard flow (movies
  that are scary)
- "Not Another Teen Movie" → exact title (the 2001 film) OR standard flow
  (user wants something other than a teen movie)
- "Love Story" → exact title (the 1970 film) OR standard flow (movies
  with a love story)
- "Date Night" → exact title (the 2010 film) OR standard flow (movies for
  a date night)

**Examples where branching does NOT apply** — one interpretation is
clearly dominant:
- "Frozen" — clearly the Disney movie in a movie search context
- "Her" — clearly the 2013 film; no other reasonable reading
- "Cars" — clearly the Pixar film; "movies with cars" is not a similarly
  reasonable interpretation
- "La La Land" — distinctive title, no ambiguity
- "Inception (2010)" — disambiguation hint makes intent explicit
- "action movies starring Ryan Reynolds" — clearly standard flow

**Other branching candidates** (within standard flow):
- "Movies like Inception and Interstellar" — different subsets of traits
  could be extracted from the two reference movies, producing multiple
  reasonable standard-flow interpretations
- "Movies like Inception but funnier" — different key traits of the
  reference movie could be emphasized in different interpretations

Branching is capped at **3 interpretations maximum**. The default should
be a single interpretation, and the step 1 prompt should not subtly
encourage producing multiple interpretations when ambiguity is low.

### Output Structure

The step 1 LLM produces a `FlowRoutingResponse` (defined in
`schemas/flow_routing.py`). The schema is designed to follow the prompt
authoring conventions established during metadata generation — cognitive
scaffolding field ordering, evidence-inventory reasoning, brief
pre-generation fields, and abstention-first framing for rare behaviors.

#### Top-Level Fields

**`interpretation_analysis`** (string, required) — One concise sentence
stating whether the query has a single clear reading or multiple equally
reasonable interpretations. When multiple, names what makes the query
ambiguous.

*Why included:* Forces the model to assess ambiguity before generating
interpretations, following the evidence-inventory pattern. Without this
field, the model defaults to "always produce something" and is more likely
to manufacture branching. The abstention-first framing ("most queries have
a single clear reading") counteracts that tendency. Kept to one sentence
per the brief-pre-generation-fields convention — a classification, not an
essay.

**`interpretations`** (list of 1–3 `QueryInterpretation`) — The
interpretation(s) of the query. Most queries produce exactly 1. The first
interpretation in the list is the default that the system auto-executes.

#### Per-Interpretation Fields

Each `QueryInterpretation` contains, in order:

**`routing_signals`** (string, required) — One short sentence citing the
specific words or patterns in the query that determined this
interpretation's flow classification.

*Why included:* Per-interpretation evidence inventory. Forces the model to
ground each routing decision in concrete query text rather than
pattern-matching on vibes. Placed first so the cited evidence scaffolds
the downstream fields (intent rewrite, flow enum, title extraction).

**`intent_rewrite`** (string, required) — The user's query rewritten as a
complete, concrete statement of what they are looking for under this
interpretation. Makes implicit expectations explicit.

*Why included:* Serves two purposes. First, it is the primary scaffolding
field — by articulating the full concrete intention before selecting the
flow enum, the model commits to a specific reading that constrains the
remaining fields. Second, it feeds directly into step 2 for standard-flow
interpretations as the input query that gets decomposed into dealbreakers
and preferences. For exact-title and similarity flows (which skip step 2),
the rewrite still serves as a human-readable audit trail of what the model
understood.

**`flow`** (enum: `exact_title` | `similarity` | `standard`, required) —
Which major search flow handles this interpretation.

*Why included:* The core routing decision. Placed after `routing_signals`
and `intent_rewrite` so the model has already committed to evidence and a
concrete reading before selecting the enum — reducing the chance of the
enum selection driving the interpretation rather than the other way around.

**`display_phrase`** (string, required, 2–8 words) — Short label for this
interpretation as displayed in the app UI. For exact-title flows: the
movie title. For similarity: "Movies like [title]." For standard: a brief
summary of the search intent.

*Why included:* The app needs a display label for each interpretation
group, especially when multiple interpretations are presented for user
selection. Always required (not nullable) because even single-
interpretation queries benefit from a display header, and it costs the
model almost nothing. Placed after `flow` because the flow classification
informs what kind of label to generate.

**`title`** (string, nullable) — The movie title extracted from the query,
using the most common fully expanded English-language title form. Required
when flow is `exact_title` or `similarity`. Null for `standard`.

*Why included:* Both non-standard flows need a title string to look up in
the database. The "most common fully expanded" instruction follows the
exact-match convergence convention — both the LLM output and the database
entries should converge on the same canonical form to maximize match
probability without requiring fuzzy matching infrastructure. DB-side
trigram matching still serves as a safety net for residual mismatches.
Placed last because it is conditional on the flow value.

#### Design Rationale: Field Ordering

The field order within each interpretation follows the model's cognitive
chain: **evidence → intent → classification → display → extraction**. This
is deliberate:

- `routing_signals` and `intent_rewrite` are open-ended generation that
  benefits from appearing early in the token sequence (no prior commitments
  to anchor against).
- `flow` is a constrained enum that benefits from the rewrite having
  already committed the model to a direction.
- `display_phrase` and `title` are derivative fields that the model can
  generate confidently once flow is decided.

This mirrors the cognitive-scaffolding convention from metadata generation:
concrete/extractive fields before abstract/synthetic, with reasoning
immediately before the label it scaffolds.

---

## Step 2: Query Understanding

A single LLM call that does all interpretive work upfront. Downstream steps
receive resolved, concrete instructions — they never re-assess what the query
means.

### Preprocessing Chain

The LLM follows a structured reasoning chain to produce its output:

1. **Rewrite the query** in its full concrete intentions. Make implicit
   expectations explicit where appropriate (e.g., "dicaprio comedies" ->
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
  both route to `entity`.

### Multi-Interpretation Support

Interpretation branching happens in step 1, before structured decomposition.
Step 2 operates on one branch at a time: one rewritten query plus one display
phrase, producing one complete decomposition for that branch. Note that step 2
only runs for standard-flow branches — exact title and similarity branches
bypass steps 2-4 entirely.

### Output Structure: Dealbreakers

Dealbreakers represent the foundational attributes around which the rest of the
query revolves. They are the criteria used for candidate generation — movies
that don't meet these are excluded or tiered down.

Each dealbreaker has:
- **Description** — A concrete string describing the requirement (e.g., "is a
  rocky movie", "does not have a fight with a russian")
- **Routing** — An enum value indicating which endpoint handles this
  dealbreaker. One of: `entity`, `metadata`, `awards`, `franchise_structure`,
  `keyword`, `semantic`, `trending`
- **Direction** — Whether this is an `inclusion` (must have) or `exclusion`
  (must not have)

**Routing enum definitions (surface-level, no schema details):**

| Route | What it covers | Step 2 LLM needs to know |
|-------|---------------|--------------------------|
| `entity` | Named entities: actors, directors, writers, producers, composers, characters, studios, movie titles | Entity types available |
| `metadata` | Structured movie attributes: genre, year, runtime, rating, streaming, country, source material type, box office, budget, plus generic "award-winning" (denormalized win IDs) | Field names (not enum values) |
| `awards` | Specific award lookups: named ceremonies, categories, outcomes, years | The 12 ceremony names |
| `franchise_structure` | Franchise name resolution AND structural roles: sequel, prequel, remake, reboot, spinoff, crossover, launched-a-franchise | Franchise names + structural attributes available |
| `keyword` | Concept tags and content keywords from curated vocabulary | The full 225-term keyword vocabulary + 25 concept tags (included in prompt) |
| `semantic` | Subjective qualities, vibes, thematic concepts not covered by other sources | What the other sources DON'T cover |
| `trending` | Currently trending / popular right now | That trending data exists |

**Critical:** The LLM must understand the limitations of each source. It should
know the keyword/concept tag vocabulary so it can make informed routing decisions
rather than guessing that a concept like "clowns" might be a keyword when it
isn't. When no deterministic source cleanly covers a concept, route to `semantic`.

**Routing failure is accepted, not recovered from.** Endpoints overlap
conceptually enough that most misroutes still produce reasonable results. For the
narrow cases where a misroute would produce zero results (e.g., routing to
`keyword` for a term not in the vocabulary), the step 2 prompt definitions must
be precise enough to prevent this from happening. There is no retry-with-
different-route fallback. **Implementation note:** getting the endpoint
definitions and boundary descriptions right in the step 2 prompt is a critical
implementation concern — misroute prevention is a prompt design problem, not an
architectural one.

**One dealbreaker per route instance, but multiple dealbreakers per query may
share a route.** For example, "Leonardo DiCaprio Rocky movies" produces two
`entity` dealbreakers. Each dealbreaker is executed as an independent search and
produces its own candidate set.

**Distinguishing between overlapping endpoints:** Some concepts could be handled
by multiple endpoints. The step 2 LLM picks the best one per query based on
surface-level signals:

| Signal | Routes to |
|--------|-----------|
| Names a person, studio, character, or title | `entity` |
| References a structured movie attribute (genre, year, runtime, rating, streaming, country, source material) or generic "award-winning" | `metadata` |
| Names a specific ceremony, award category, or award year | `awards` |
| Names a franchise OR references franchise structural role (sequel, spinoff, remake, reboot) | `franchise_structure` |
| Matches a known keyword or concept tag from the vocabulary | `keyword` |
| Subjective quality, vibe, or thematic concept not covered above | `semantic` |
| Trending / currently popular / buzzing right now | `trending` |

### Output Structure: Preferences

Preferences are qualities used to evaluate and rerank candidates generated by
dealbreakers. They do not generate candidates — they only influence ordering.

Each preference has:
- **Description** — A concrete string describing the quality (e.g., "dark",
  "gritty", "funny")
- **Routing** — Same enum as dealbreakers. Determines which endpoint scores
  this preference.
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

**Explicit sort-order requests are preferences, not a separate mechanism.**
"In order," "chronologically," "most recent first" are expressed as metadata
preferences with `is_primary_preference=true`. Example: "all Fast and Furious
movies in order" → franchise dealbreaker + metadata preference on release_date
ascending with `is_primary_preference=true`. Without "in order," no sort
preference is emitted and default quality ranking applies. The preference
description carries the sort direction (e.g., "ordered by release date, earliest
first") and the endpoint LLM translates it into the appropriate query spec.

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

### Reference Movies in the Standard Flow

When a "movies like X but qualifiers" or multi-reference query enters the
standard flow, step 2 uses the LLM's parametric knowledge of the reference
movie(s) to extract concrete attributes — it does not resolve the movie to a
`tmdb_id`. "Movies like Inception but in space" becomes "mind-bending action
movie set in space," and the dealbreakers and preferences are extracted from
that rewrite. Multi-reference queries ("movies like Inception and
Interstellar") require the LLM to extract and merge traits from both movies,
which may produce multiple step 1 interpretations since different trait
subsets could be emphasized. If the reference movie title is ambiguous
(multiple movies share the name), step 1 extracts the title and the DB
search for exact matches handles disambiguation downstream.

### What Step 2 Does NOT Do

- Does not know schema details (table names, column types, enum values)
- Does not determine exact search parameters (that's step 3's job)
- Does not determine vector space routing (that's step 3's semantic endpoint)
- Does not inject system defaults into the query rewrite
- Does not resolve reference movies to `tmdb_id`s — uses parametric knowledge
  to extract the intended attributes instead

---

## Step 3: Query Translation

Each endpoint has its own LLM (or deterministic function) that receives:
- The full rewritten query (for context)
- Only the dealbreakers and preferences routed to it
- Deep knowledge of its own schema

Each per-endpoint LLM translates the abstract dealbreaker/preference
descriptions into **complete query specifications**. For dealbreakers, the spec
is self-contained and ready to execute. For preferences, the spec is complete
except for candidate IDs — a WHERE clause placeholder (or vector lookup target
set) that gets filled once candidate assembly provides the IDs. Execution of
these specs happens in step 3.5.

**Why keep per-endpoint LLMs instead of pushing all schema-specific work into
step 2?** Because exact enum values, metadata matching nuance, keyword
definitions, lexical role nuance, actor prominence cues, and vector-space
behavior are too much specialized low-level knowledge for one smaller
interpretive LLM to carry without quality loss. The split is:

- **Step 2** interprets user intent, consolidates concepts, and routes each
  item to the correct endpoint.
- **Step 3** translates already-interpreted intent into endpoint-specific
  query specifications.

Step 3 LLMs are **schema translators, not re-interpreters**. They should not
decide what the user meant; they should only decide how their endpoint should
execute the already-resolved intent.

All per-endpoint LLMs run in parallel since they have no dependencies on each
other.

### Endpoint 1: Entity Lookup

**Data sources:** All `lex.*` posting tables — actors, directors, writers,
producers, composers, characters, studios, titles. Fuzzy matching via trigram
GIN indexes on title and character dictionaries; exact-after-normalization
matching for all others. Also handles **title substring/pattern matching**
(ILIKE) for queries like "movies with 'love' in the title" or "starts with
'star'." Exact title identification routes to step 1's known-movie flow, so
this endpoint only handles substring and pattern-based title queries. Franchise
name resolution is handled entirely by the Franchise Structure endpoint, not
here.

**LLM knows:** Available entity types and their posting tables, role hints
(routing "Nolan" to director vs. actor), actor prominence modes (top billing
only, boost by position, binary, reverse), multi-token title intersection
logic, fuzzy matching behavior, title substring matching via ILIKE.

**Candidate generation (dealbreakers):** Each entity dealbreaker produces an
independent candidate set of movie IDs. Entity matches are binary — a movie
either appears in the posting table for that entity or it doesn't.

**Preference scoring:** Binary presence score (is the entity associated with
this movie?), with actor prominence as a gradient signal when applicable
(billing_position scoring).

**Example dealbreakers:** "starring Leonardo DiCaprio", "directed by Nolan",
"Pixar movies", "character named Tyler Durden", "movies with 'love' in the
title"

**Example preferences:** "preferably starring Brad Pitt"

### Endpoint 2: Movie Attributes

**Data sources:** `movie_card` scalar and array columns — genre, release date,
runtime, maturity rating, streaming availability, country of origin, source
material type, budget bucket, box office bucket, popularity score, reception
score, and the denormalized `award_ceremony_win_ids`.

**LLM knows:** Column names and types, all enum values (27 genre IDs, 334
language IDs, 262 country IDs, streaming provider keys + access types, 10
source material types, 5 maturity ranks), comparison operators, gradient
scoring behavior for soft constraints.

**Candidate generation (dealbreakers):** Each metadata dealbreaker produces a
candidate set via SQL WHERE clauses (hard filters) or GIN array overlap. For
NLP-extracted constraints, a generous threshold determines pass/fail for
candidate generation purposes (e.g., "80s movies" uses a generous gate of
~1975-1994).

**Preference scoring:** Gradient scoring, not binary. "Under 100 minutes" at
101 minutes scores ~0.95, at 140 minutes scores much lower. Users are
frequently imprecise with numeric constraints, so gradients prevent harsh
cutoffs that miss obviously relevant results. Different attributes have
different softness levels (see Constraint Strictness in open_questions.md).

**Example dealbreakers:** "comedy", "from the 80s", "under 2 hours", "on
Netflix", "Korean movies", "rated R", "based on a true story", "award-winning"
(simple, no specific ceremony)

**Example preferences:** "preferably recent", "family friendly", "well-reviewed",
"big budget blockbuster"

### Endpoint 3: Awards

**Data sources:** `movie_awards` table — ceremony_id, award_name, category,
outcome_id (winner/nominee), year. Indexed on
`(ceremony_id, award_name, category, outcome_id, year)`.

**LLM knows:** 12 ceremony IDs and their award structures (Academy Awards,
Golden Globes, BAFTA, Cannes, Venice, Berlin, SAG, Critics Choice, Sundance,
Razzie, Spirit Awards, Gotham), specific prize names within each ceremony,
category names, winner vs. nominee distinction.

**Candidate generation (dealbreakers):** Produces candidate sets via
deterministic SQL queries on the awards table. "Oscar Best Picture winners" ->
`ceremony=1, award_name='Oscar', category='Best Picture', outcome=1`.

**Preference scoring:** Can score by award count, ceremony prestige weighting,
or recency of awards.

**Step 2 routing distinction from Movie Attributes:** "Award-winning" (generic,
no ceremony named) -> Movie Attributes (simple `award_ceremony_win_ids` array
check). "Oscar Best Picture" (names a ceremony + category) -> Awards (needs
the full awards table schema). The surface-level signal is: does the query
name a specific ceremony, category, or year?

**Example dealbreakers:** "Oscar Best Picture winners", "2023 Cannes Palme
d'Or", "Razzie winners", "nominated at Sundance"

**Example preferences:** "preferably award-nominated"

### Endpoint 4: Franchise Structure

**Data sources:** `movie_franchise_metadata` — lineage (franchise name),
shared_universe, recognized_subgroups, lineage_position (1=sequel, 2=prequel,
3=remake, 4=reboot), is_spinoff, is_crossover, launched_franchise,
launched_subgroup. This is the sole source for all franchise queries — both
name resolution and structural filtering.

**LLM knows:** The franchise table schema, fuzzy name matching behavior,
the distinction between lineage and shared_universe, lineage_position values,
boolean flag semantics, subgroup matching via trigram similarity.

**Candidate generation (dealbreakers):** Handles two kinds of dealbreakers:
- **Franchise name** ("Marvel movies") — fuzzy-matches the lineage/
  shared_universe columns to produce a candidate set of movie IDs.
- **Structural role** ("sequels", "spinoffs") — filters on lineage_position,
  is_spinoff, is_crossover, etc.
When both appear in the same query ("Marvel spinoffs"), this endpoint handles
both as separate dealbreakers, and tiering handles the intersection.

**Preference scoring:** Binary match on structural attributes, or gradient
scoring if applicable (e.g., franchise recency).

**Example dealbreakers:** "Marvel movies", "James Bond franchise", "sequels",
"spinoff movies", "remakes", "movies that started a franchise"

**Example preferences:** "preferably not a sequel"

### Endpoint 5: Keywords & Concept Tags

**Data sources:** `movie_card.keyword_ids` (225 curated OverallKeyword terms
with definitions) + `movie_card.concept_tag_ids` (25 binary tags across 7
categories).

**LLM knows:** The full 225-term keyword vocabulary with per-keyword
definitions, all 25 concept tag definitions grouped by category (narrative
structure, plot archetype, setting, character, ending, experiential, content
flag). Maps user concepts to specific IDs.

**Candidate generation (dealbreakers):** Produces candidate sets via GIN array
overlap on keyword_ids or concept_tag_ids. These are binary — a movie either
has the keyword/tag or doesn't.

**Preference scoring:** Binary match (has the keyword/tag = 1.0, doesn't =
0.0). For dealbreaker-demoted-to-preference scenarios, this is a strong
boosting signal.

**May not require a separate LLM:** If step 2 already selects from the
enumerated vocabulary (which is included in the step 2 prompt), the endpoint
may be a deterministic ID lookup rather than an LLM call. This is an
implementation detail.

**Example dealbreakers:** "zombie movies", "heist", "coming-of-age", "movies
with a twist ending", "feel-good", "haunted house"

**Example preferences:** "preferably with a happy ending", "not a tearjerker"

**Implementation note — dual dealbreaker + preference for centrality:** Some
concepts map cleanly to a keyword/tag (binary: has it or doesn't) but also have
a meaningful spectrum above that threshold. "Christmas movies" maps to the
"Holiday" keyword for candidate generation, but Christmas-*centrality* (is
Christmas the entire premise, or just incidental backdrop?) is a useful ranking
signal. For these cases, the step 2 LLM should emit both a keyword dealbreaker
AND a semantic preference for the same concept. The pipeline already supports
this since the dealbreaker and preference target different endpoints (keyword
vs. semantic). This is an LLM prompt design concern to address when building
the step 2 prompt.

### Endpoint 6: Semantic

**Data sources:** 8 Qdrant vector spaces (OpenAI `text-embedding-3-small`,
1536 dims) — anchor, plot_events, plot_analysis, viewer_experience,
watch_context, narrative_techniques, production, reception.

**LLM knows:** What each vector space captures, subquery formulation best
practices per space, space selection logic, the 80/20 subquery/original blend
ratio.

**Candidate generation (dealbreakers — with demotion):** Semantic dealbreakers
are NOT used for candidate generation in the standard flow. Any dealbreaker
routed to `semantic` is automatically demoted to a high-weight preference for
scoring purposes (see Semantic Dealbreaker Demotion below). Exception: in the
pure-vibe flow (no non-semantic inclusion dealbreakers exist), vector search
becomes the candidate generator.

**Preference scoring:** Vector similarity scores against relevant spaces. The
LLM determines which spaces are relevant, generates expanded search queries
per space, and handles both inclusion scoring (cosine similarity, possibly
with diminishing returns curve) and exclusion penalties (global-elbow-
calibrated, see Exclusion Handling below).

**Example dealbreakers (demoted):** "dark and gritty", "visually stunning",
"slow burn"

**Example preferences:** "funny", "thought-provoking", "cozy date night vibe",
"similar vibe to Midsommar"

### Endpoint 7: Trending

**Data sources:** Redis `trending:current` hash — precomputed trending scores
[0, 1] for all trending movies, refreshed from TMDB weekly trending API.

**LLM needed?** Probably not — this is a binary/deterministic signal. Step 2
flags "trending" intent and execution reads the Redis hash directly. This
endpoint is likely a simple deterministic function rather than an LLM-backed
translator.

**Candidate generation (dealbreakers):** Returns all movie IDs with non-zero
trending scores as a candidate set.

**Preference scoring:** Pass-through of the precomputed trending score [0, 1].

**Example dealbreakers:** "trending movies", "what's popular right now"

**Example preferences:** "preferably something that's buzzing right now"

---

## Step 3.5: Search Execution

LLM generation time is the major latency bottleneck — individual database and
vector queries take only milliseconds. This step exploits that asymmetry by
overlapping execution with translation.

All step 3 LLMs fire simultaneously. As each LLM responds with a query
specification:

- **Dealbreaker specs** are self-contained — execute the search immediately,
  producing a candidate set of movie IDs. No need to wait for other endpoints
  to finish translation.
- **Preference specs** are complete except for the candidate IDs to score.
  They sit ready while dealbreaker searches run.

Once all dealbreaker searches complete, candidate assembly (step 4a) produces
the pool of movie IDs. Preference queries then fire immediately — the candidate
IDs are slotted into the prepared WHERE clauses (for SQL-based endpoints) or
used as the target set for vector lookups (for the semantic endpoint), and
execution adds only milliseconds of query time.

**Net effect:** all fetches execute in near-parallel despite the sequential
dependency between candidate generation (dealbreakers) and candidate scoring
(preferences). Wall-clock time is dominated by the slowest step 3 LLM call,
not by the sum of all searches.

**Semantic preference execution** follows the same pattern but with a different
mechanism: the step 3 semantic LLM produces expanded search queries and target
vector spaces, and execution means fetching stored vectors for the candidate
IDs and computing cosine similarity — not issuing a SQL query.

---

## Step 4: Assembly & Reranking

### Phase 4a: Candidate Generation Assembly

Collect all candidate sets produced by dealbreaker execution in step 3.5. Each
inclusion dealbreaker produces its own independent candidate set.

- **Deterministic dealbreakers** (entity, metadata, awards, franchise_structure,
  keyword, trending): Each search returns a set of movie IDs. These are
  binary — a movie is in the set or not.
- **Semantic dealbreakers are NOT used for candidate generation.** Any
  dealbreaker routed to `semantic` is demoted to a preference for scoring
  purposes (see Semantic Dealbreaker Demotion below).

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

**Minimum floor:** A demoted semantic dealbreaker still imposes a minimum
relevance floor. Candidates with essentially no semantic match to the stated
concept should not be treated as if they satisfied that user requirement. The
exact floor calibration is deferred to implementation.

**Pure-vibe queries:** When no non-semantic inclusion dealbreakers exist, the
query enters the pure-vibe flow as a separate codepath (see below). This
includes both "all dealbreakers are semantic" and "only semantic inclusions plus
deterministic exclusions" (e.g., "good date night movies not with adam sandler"
— date night is semantic, adam sandler is an entity exclusion). The flow control
checkpoint happens immediately after step 2 output is inspected: if no
deterministic inclusion dealbreaker exists, reroute to pure-vibe.

### Phase 4b: Exclusion Handling

Exclusions are applied after candidate generation. They do NOT count toward
the tier denominator — tiers are based on inclusion match count only.

**Deterministic exclusions** (entity, metadata, awards, franchise_structure,
keyword): Hard filter. If a movie matches the exclusion criteria, it is
removed from the candidate set entirely. These are binary and reliable — "not
starring Arnold Schwarzenegger" can be definitively evaluated.

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
     concept -> harsh downrank
   - **Near the elbow:** Uncertain -> softer downrank
   - **Well below the elbow:** No meaningful match to the concept -> no penalty

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

### Phase 4c: Preference Scoring & Final Ranking

Score all remaining candidates on preferences and system-level priors. The
combination works as:

1. **Primary sort: tier** (inclusion dealbreaker match count, descending)
2. **Secondary sort: preference + system-prior composite** (within each tier)

**Preference scoring by route type:**
- Deterministic preferences (metadata, keyword, awards, franchise_structure,
  entity, trending): Gradient scoring where applicable. "Under 100 minutes"
  at 101 minutes scores ~0.95, at 140 minutes scores much lower. Binary
  attributes score 1.0 or 0.0.
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

## Pure-Vibe Flow (No Deterministic Inclusion Anchors)

When step 2 output contains no non-semantic inclusion dealbreakers, the query
enters a separate codepath where vector search is the candidate generator. This
triggers when all inclusion dealbreakers route to `semantic`, regardless of
whether deterministic exclusions also exist. The detection is an explicit flow
control checkpoint immediately after step 2 completes.

### How It Works

1. All semantic dealbreakers become preferences (there are no dealbreakers for
   tiering since there are no deterministic anchors).
2. The step 3 semantic endpoint determines relevant vector spaces and generates
   search queries. A single LLM call handles relative importance across spaces.
3. **Individual searches per concept** across relevant spaces. Synonymous
   concepts are already consolidated by step 2, so each search represents a
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
problem at query time. The step 2 LLM already consolidates truly synonymous
concepts, so remaining distinct concepts should stay separate.

### Exclusions in Pure-Vibe Flow

Both exclusion types apply to the vector-generated candidate set:

- **Deterministic exclusions** (entity, metadata, keyword, etc.) hard-filter
  from the candidate set, same as the standard flow. "Not with adam sandler"
  removes all movies matching the entity exclusion from the vector results.
- **Semantic exclusions** use the elbow-threshold penalty against the global
  distribution, same as the standard flow.

**Implementation note:** investigate whether deterministic exclusion IDs can be
excluded from the vector search itself (e.g., passed as a negative filter to
Qdrant before execution) rather than filtered from results afterward. This
avoids wasting retrieval slots on movies that will be removed, which matters
more in pure-vibe flow where the candidate pool is entirely vector-generated.

---

## Key Design Principles

### 1. Deterministic sources generate candidates; semantic sources score them

The most important architectural decision. Empirical testing proved semantic
search is unreliable for candidate generation but works well for ranking.
Deterministic channels (entity lookup, metadata filters, keyword matching,
awards, franchise structure, trending) produce reliable, complete candidate
sets. Semantic similarity is applied afterward to score and rank those
candidates.

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
rather than candidate generation. When no non-semantic inclusion dealbreakers
exist, the query enters the pure-vibe flow as a separate codepath.

### 5. System-level priors are separate, explicit dimensions

Quality bias is not baked into the query rewrite or treated as a preference.
The design now explicitly recognizes that conventional quality and
notability/mainstreamness are distinct dimensions. The finalized decision is to
keep them separate conceptually; the exact field shape remains open.

### 6. Step 2 interprets intent; step 3 knows schemas

The interpretive LLM (step 2) needs surface-level awareness of endpoints
(what each covers, keyword/concept tag vocabulary) but not schema details.
Per-endpoint LLMs (step 3) need deep schema knowledge but receive
pre-interpreted intent. This split keeps each LLM's task tractable for smaller,
faster models without asking the step 2 model to also carry every exact enum,
matching rule, keyword definition, and low-level source-specific nuance.

### 7. Metadata constraints use gradients, not binary filters

NLP-extracted numeric and temporal constraints use gradient scoring rather than
hard cutoffs. This prevents missing obviously relevant results when users are
imprecise (which they frequently are). Tier assignment uses a generous threshold
for pass/fail; within-tier ranking uses the actual gradient score.

---

## Scoring Function Modes (from prior planning, unchanged)

Four scoring modes apply to different attribute types:

- **Threshold + flatten** — For dealbreakers. Similarity >= threshold -> 1.0
  (passes), below -> decay. Used for determining if a movie "has" an attribute.
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

- Exact step 2 prompt engineering (few-shot examples, chain-of-thought format)
- Exact elbow detection algorithm for semantic exclusion thresholds
- Step 3 prompt design per endpoint
- Specific candidate pool size limits per endpoint
- Exact gradient decay functions for metadata constraint scoring
- Tier assignment threshold calibration (how generous is "passing"?)
- Whether the keyword endpoint LLM is a separate call or folded into step 2
- Result pagination for long lists: return top 25 initially, cache the full
  candidate/display list in Redis, and allow fetching additional pages via
  pointer IDs or equivalent
- Multi-interpretation trigger criteria
