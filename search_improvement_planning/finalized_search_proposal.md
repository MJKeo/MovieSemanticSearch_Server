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
4. **Assembly & Reranking** — Assemble candidate sets, score each movie by
   dealbreaker match quality + preference fit + system priors, apply
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

### Input

Step 2 receives the `intent_rewrite` from step 1 as its input query. It
operates on one standard-flow interpretation at a time, producing one complete
decomposition per branch. Exact-title and similarity branches bypass steps 2-4
entirely.

### Preprocessing Chain

The LLM follows a structured reasoning chain to produce its output:

1. **Analyze the query's decomposition** — inventory the distinct requirements
   and qualities present, classifying each as a hard requirement (dealbreaker)
   or a soft quality (preference). This is a brief evidence inventory (two to
   three sentences) that guides all subsequent structured generation.
2. **Generate a structured list** of individual dealbreakers and preferences
   with routing flags (see output structure below).
3. **Assess quality and notability priors** based on the full decomposition —
   these depend on what dealbreakers and preferences were extracted, so they
   come last (see Quality / Notability Priors below).

**Grouping and consolidation rules:**

The LLM should consolidate synonymous or clearly related concepts into a single
dealbreaker/preference rather than splitting them into separate items.

- **Do group** near-synonymous concepts when separating them adds no retrieval
  or scoring value.
- **Do not group** distinct entities or constraints merely because they share a
  route. "Brad Pitt" and "Tom Hanks" remain separate dealbreakers even though
  both route to `entity`.

**Semantic preferences receive special grouping treatment.** All semantic
preferences (qualifiers on other attributes — "funny," "dark," "slow-burn")
are grouped into a single rich preference description rather than listed as
individual items. This produces better semantic matching: "dark and gritty
thriller" as a combined concept targets vector spaces more precisely than
separate "dark" and "gritty" queries unioned together. The step 3 semantic
endpoint can still decompose the combined description into per-space queries.

Exceptions to semantic preference grouping:
- **Disjunctive intent ("or"):** "funny or intense action films" → separate
  preferences, because the user would be satisfied with either qualifier
  independently.
- **Semantic dealbreakers are not grouped with semantic preferences.** They
  represent distinct defining traits (see Dealbreakers below), not qualifiers.

**V1 intentionally does NOT model full boolean/group logic.** Explicit OR-style
clause handling would improve a subset of edge cases, but the added complexity
is not justified yet because the continuous scoring model already degrades
gracefully for most such queries. Whether this simplification holds up will be
evaluated once the system is running against real queries.

### Output Structure

The step 2 LLM produces a `QueryUnderstandingResponse` (defined in
`schemas/query_understanding.py`). The schema follows the same conventions as
step 1 — cognitive-scaffolding field ordering, evidence-inventory reasoning,
and brief pre-generation fields. No class-level docstrings or field
descriptions; all LLM-facing guidance lives in the system prompt.

#### Top-Level Fields

**`decomposition_analysis`** (string, required) — Two to three sentences that
inventory the distinct requirements and qualities in the query and classify
each as a hard requirement or a soft quality.

*Thought process:* The model performs two steps in a single brief passage:
(1) **Inventory** — name each separable concept present in the query. "Dark
gritty Marvel movies" contains three concepts: dark, gritty, and Marvel.
"Scary movies like The Conjuring but set in space" contains: scary, The
Conjuring's traits, and space setting. The model extracts what is there
without adding or inferring beyond the query. (2) **Classify** — for each
concept, note whether it is a hard requirement the search revolves around or
an experiential qualifier that influences ranking. "Dark gritty Marvel
movies": Marvel is a hard requirement (defines the candidate pool), dark and
gritty are soft qualities (rank within it). "Scary movies set in space":
scary and space setting are hard requirements (both define what kind of
movie), with scary also functioning as a ranking quality within the results.

*Why included:* The model's hardest judgment call is the dealbreaker/preference
boundary. By forcing an explicit hard-vs-soft classification before the model
starts emitting structured items, it commits to a decomposition direction that
guides the rest of the output. Without this field, the model jumps straight
into generating dealbreakers and risks miscategorizing items because it never
surveyed the full picture first. The inventory step ensures nothing is missed;
the classification step ensures the dealbreaker and preference lists are
informed by a prior assessment of the whole query rather than item-by-item
ad hoc decisions. Follows the evidence-inventory pattern (cite what the input
contains and classify it, rather than explaining why downstream items are
correct) and the brief-pre-generation-field convention (a classification, not
an essay).

*Why this replaced `query_rewrite` and `dealbreaker_summary`:* Step 1's
`intent_rewrite` already captures the full concrete intent and is passed to
step 3 endpoint LLMs as context — a second rewrite in step 2 adds marginal
value while costing tokens and risking intent drift from over-expansion.
Step 1's `display_phrase` already serves the UI display need for each
interpretation. The decomposition analysis replaces both with a single field
that directly scaffolds the hardest part of step 2's job: the
dealbreaker/preference classification.

**`dealbreakers`** (list of `Dealbreaker`, min 0) — The hard requirements
extracted from the query. Each dealbreaker produces an independent candidate
set during search execution. Empty for pure-preference or pure-vibe queries.

**`preferences`** (list of `Preference`, min 0) — Qualities used to evaluate
and rerank candidates generated by dealbreakers. They do not generate
candidates — they only influence ordering. Empty for queries that are purely
constraint-based with no ranking preferences.

*Why preferences come after dealbreakers:* The thematic centrality pattern
requires the model to know what keyword dealbreakers it emitted before
deciding what centrality information to include in the grouped semantic
preference. If the model emitted a "Holiday" keyword dealbreaker, it should
include "Christmas is central to the story" in the semantic preference
description.

**`prior_assessment`** (string, required) — One sentence citing the
quality/notability signals present in the query and noting whether a dominant
primary preference should suppress the default priors.

*Thought process:* Two steps in one brief sentence: (1) **Signal citation** —
name the quality/notability signals from the query text ("'best' signals
enhanced quality", "'hidden gems' signals inverted notability", or "no
explicit quality/notability signals"). (2) **Suppression check** — note
whether the decomposition above contains a dominant primary preference that
should push system priors to the background ("the 'scariest' primary
preference should suppress both priors"). The suppression check is what makes
this a second-order inference — it depends on the decomposition, not just the
query text.

*Why included:* Without this field, the model risks defaulting to `standard`
for both priors without considering whether the decomposition implies
`suppressed`. By explicitly checking for suppression triggers (is there a
dominant primary preference?), the model is forced to reason about the
relationship between the priors and the rest of the output. Same principle as
placing `routing_signals` before `flow` in step 1: make the evidence explicit
before committing to the classification.

**`quality_prior`** (enum: `enhanced` | `standard` | `inverted` |
`suppressed`) — System-level quality bias, scaffolded by `prior_assessment`.

**`notability_prior`** (enum: `enhanced` | `standard` | `inverted` |
`suppressed`) — System-level notability bias, scaffolded by
`prior_assessment`.

#### Per-Dealbreaker Fields

Each `Dealbreaker` contains, in order:

**`description`** (string, required) — A concrete string describing the
attribute to search for, always in positive-presence form (e.g., "includes
Brad Pitt in actors", "is a horror movie", "involves clowns"). Even for
exclusion dealbreakers, the description states what to find — the `direction`
field separately controls whether the result is included or excluded.

*Why included:* The core functional output consumed by step 3 endpoint LLMs as
their task specification. Placed first because it is the most
concrete/extractive field — the model articulates what the requirement IS
before making any classifications about it. Positive-presence framing ensures
step 3 LLMs always receive a clear "find movies WITH X" instruction,
supporting the direction-agnostic framing principle (see Step 3).

**`direction`** (enum: `inclusion` | `exclusion`, required) — Whether this
dealbreaker generates candidates and contributes to the dealbreaker score sum
(inclusion) or filters/penalizes candidates after assembly (exclusion).

*Why included:* The pipeline uses this to determine how the resulting candidate
set is applied. Inclusion dealbreakers contribute a [0, 1] score to the
dealbreaker sum; exclusion dealbreakers hard-filter (deterministic) or penalize
(semantic) after candidate assembly without contributing to the dealbreaker
sum. Placed second because it is still extractive — usually obvious from query
text markers ("not", "without", "no" → exclusion; everything else → inclusion).

**`routing_rationale`** (string, required) — A brief concept-type
classification label citing why this endpoint handles this concept. Examples:
"named person (actor)", "genre classification", "thematic concept absent from
keyword taxonomy", "franchise structural role."

*Thought process:* The model identifies what KIND of thing the described
concept is — a named entity, a genre classification, an award reference, a
franchise name, a subjective experiential quality, etc. — and names that
classification in a few words. This is a label, not an explanation.

*Why included:* This is the critical misroute prevention field. The proposal
identifies misrouting as the #1 prompt design concern. By articulating the
concept's type before selecting the route enum, the model grounds the routing
decision in the concept's nature rather than pattern-matching on surface
features. Without this field, the model might route "award-winning comedy"
entirely to `awards` without separating "comedy" as a keyword, or route
"clowns" to `keyword` because it sounds like it could be a keyword even
though it isn't in the vocabulary. The rationale forces explicit
concept → endpoint matching. Follows the evidence-inventory pattern:
classify the concept before committing to the enum.

**`route`** (enum: `entity` | `metadata` | `awards` | `franchise_structure` |
`keyword` | `semantic` | `trending`, required) — Which step 3 endpoint
handles this dealbreaker.

*Why included:* Dispatches to the correct step 3 endpoint LLM. Placed last in
the per-dealbreaker chain because by this point, the model has described the
requirement, determined its direction, and classified the concept type — the
enum should follow naturally.

#### Per-Preference Fields

Each `Preference` contains, in order:

**`description`** (string, required) — A concrete string describing the
quality to promote (e.g., "dark and gritty atmosphere with a slow-burn pace",
"preferably recent", "ordered by release date, earliest first"). For semantic
preferences, this may be the consolidated grouped description per the
semantic preference grouping rules.

*Why included:* Core functional output consumed by step 3. No direction field
exists on preferences — all preferences are framed as traits to promote.
Negative user intent is reframed as a positive preference for the opposite
quality: "not recent" → "prefer older films." Anything conceptual enough to
be a hard exclusion ("not zombie", "not with clowns") is a dealbreaker, not
a preference.

**`routing_rationale`** (string, required) — Same concept-type classification
label as on dealbreakers.

**`route`** (enum, required) — Same endpoint dispatch as on dealbreakers.

**`is_primary_preference`** (bool, required) — Whether this preference is the
dominant ranking axis rather than one equal member of a balanced set. Most
preferences are false.

*Why included:* This is the smallest useful addition that lets the system
distinguish balanced additive preferences ("dark and gritty" — multiple
contribute equally) from one dominant ranking axis ("scariest movie ever" —
one drives ordering). Mark true only for superlatives ("scariest",
"funniest", "best"), explicit sort orders ("in order", "most recent first"),
or queries where one dimension overwhelmingly dominates intent. If no
preference is marked primary, preferences are treated as equal-weighted
relative to each other aside from the separate system-level priors.

**Multi-primary handling:** If the LLM marks more than one preference as
`is_primary_preference=true`, treat all marked preferences as co-primary —
elevated equally in weight, with no single axis dominating. This is the
safest degradation because picking one arbitrarily based on list ordering is
fragile, and falling back to regular equal-weight discards the signal that
these were all important.

#### Design Rationale: Field Ordering

The top-level field order follows the model's cognitive chain:
**analysis → decomposition → assessment → classification**.

- `decomposition_analysis` is open-ended evidence inventory that benefits from
  appearing first in the token sequence (no prior commitments to anchor
  against). It surveys the full query before any structured generation.
- `dealbreakers` and `preferences` are the core structured output, informed by
  the analysis. Dealbreakers come first because preferences may depend on
  knowing which dealbreakers were emitted (thematic centrality pattern).
- `prior_assessment` is a second evidence inventory that depends on the
  decomposition above — specifically, whether a dominant primary preference
  exists that implies `suppressed` priors.
- `quality_prior` and `notability_prior` are constrained enums that benefit
  from the assessment having already committed the model to a direction.

The per-item field order within dealbreakers and preferences follows its own
cognitive chain: **what → which way → why this endpoint → where**. The
description provides context for the direction; the routing rationale grounds
the route enum.

This mirrors the cognitive-scaffolding convention: concrete/extractive fields
before abstract/synthetic, with reasoning immediately before the label it
scaffolds.

#### What's NOT in the Schema

- **No `query_rewrite`** — Step 1's `intent_rewrite` already captures full
  concrete intent and is passed to step 3 endpoint LLMs as context. A second
  rewrite adds marginal value while costing tokens and risking intent drift
  from over-expansion.
- **No `dealbreaker_summary`** — Step 1's `display_phrase` already serves the
  UI display need for each interpretation.
- **No pure-vibe detection flag** — Derivable from the output by checking
  whether any dealbreaker has a non-semantic route. Don't ask the LLM to do
  what code can derive.
- **No disjunctive flag on preferences** — Handled by outputting separate
  preference items per the grouping rules.
- **No confidence scores** — Encourages over-inference in small models; the
  model generates high confidence to justify its own output.
- **No per-item dealbreaker-vs-preference rationale** — The
  `decomposition_analysis` handles this classification upstream at the survey
  level, which is where it belongs. Per-item rationalization would be post-hoc.

### Dealbreaker and Preference Semantics

The following sections describe the semantics of dealbreakers and preferences
in detail. These are behavioral specifications — how the step 2 LLM should
think about classification, routing, and grouping. The schema structure above
defines the output shape; this section defines the content.

#### Dealbreakers

Dealbreakers represent the foundational attributes around which the rest of the
query revolves. They are the criteria used for candidate generation — movies
that don't meet these are excluded or scored lower.

**Routing enum definitions (surface-level, no schema details):**

| Route | What it covers | Step 2 LLM needs to know |
|-------|---------------|--------------------------|
| `entity` | Named entities: actors, directors, writers, producers, composers, characters, studios, movie titles | Entity types available |
| `metadata` | Quantitative movie attributes: year, runtime, rating, streaming, language, country, box office, budget, popularity, reception | Field names (not enum values) |
| `awards` | All award lookups: generic "award-winning" through specific ceremony/category/year queries | The 12 ceremony names |
| `franchise_structure` | Franchise name resolution AND structural roles: sequel, prequel, remake, reboot, spinoff, crossover, launched-a-franchise | Franchise names + structural attributes available |
| `keyword` | Categorical classification: concept families backed by genres, source material types, concept tags, and curated keywords | The canonical concept-family taxonomy, overlap rules, and representative classifications it can resolve |
| `semantic` | Distinct thematic traits the user treats as defining requirements where no deterministic source can evaluate them (e.g., "zombie," "clown," "female empowerment," "car chase") | What the other sources DON'T cover |
| `trending` | Currently trending / popular right now | That trending data exists |

**Critical:** The LLM must understand the limitations of each source. It should
know what the keyword taxonomy covers via the canonical concept families and
their overlap rules (see Endpoint 5 below) so it can make informed routing
decisions rather than guessing that a concept like "clowns" might be a keyword
when it isn't. When no deterministic source cleanly covers a concept, route to
`semantic`.

**Semantic dealbreakers vs. semantic preferences are distinct concepts.** A
semantic *dealbreaker* is a concrete thematic trait the user treats as a defining
requirement — something the movie must center on or contain. These are binary-ish
in nature: "centers around zombies," "contains themes of female empowerment,"
"involves a car chase." They are NOT qualifiers or scales ("how zombie-centric?")
but rather "does this movie have this trait?" evaluated via vector thresholding.
A semantic *preference* is a qualifier on the overall experience — "funny,"
"scary," "dark and gritty," "slow-burn." These describe subjective qualities
used for ranking, not filtering. The distinction: dealbreakers define *what kind
of movie*, preferences describe *what it should feel like*.

**Routing failure is accepted, not recovered from.** Endpoints overlap
conceptually enough that most misroutes still produce reasonable results. For the
narrow cases where a misroute would produce zero results (e.g., routing to
`keyword` for a term not in the vocabulary), the step 2 prompt definitions must
be precise enough to prevent this from happening. There is no retry-with-
different-route fallback. **Implementation note:** getting the endpoint
definitions and boundary descriptions right in the step 2 prompt is a critical
implementation concern — misroute prevention is a prompt design problem, not an
architectural one. The `routing_rationale` field on each item provides structural
scaffolding for this, but the prompt must still carry precise endpoint boundary
descriptions.

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
| References a quantitative movie attribute (year, runtime, rating, streaming, language, country, budget, box office, popularity, reception) | `metadata` |
| References awards in any form, from generic "award-winning" to specific ceremony/category/year | `awards` |
| Names a franchise OR references franchise structural role (sequel, spinoff, remake, reboot) | `franchise_structure` |
| References a genre, source material type, or matches a trait covered by the keyword/concept tag vocabulary | `keyword` |
| Distinct thematic trait the user treats as a defining requirement, not covered by deterministic sources above | `semantic` |
| Trending / currently popular / buzzing right now | `trending` |

#### Preferences

Preferences are qualities used to evaluate and rerank candidates generated by
dealbreakers. They do not generate candidates — they only influence ordering.

**All preferences are framed as traits to promote.** There is no
inclusion/exclusion direction on preferences. Negative user intent is reframed
as a positive preference for the opposite quality:
- "Not recent" → metadata preference: "prefer older films"
- "Not scary" → semantic preference: "not scary" (boost movies matching "not
  scary" in vector space, rather than penalizing those matching "scary")
- "Not about zombies" → this is a dealbreaker (exclusion), not a preference,
  because it's a concrete trait used for filtering

Anything conceptual enough to be a hard exclusion ("not zombie," "not with
clowns") is a dealbreaker, not a preference. Preferences always describe
desirable qualities.

**Semantic preference grouping:** All semantic preferences (qualifiers on the
desired experience) are consolidated into a single rich preference description.
"Dark, gritty, slow-burn thriller" is one semantic preference, not three. The
step 3 semantic endpoint decomposes this into per-space queries. Exception:
disjunctive qualifiers ("funny or intense") remain separate preferences.

**Explicit sort-order requests are preferences, not a separate mechanism.**
"In order," "chronologically," "most recent first" are expressed as metadata
preferences with `is_primary_preference=true`. Example: "all Fast and Furious
movies in order" → franchise dealbreaker + metadata preference on release_date
ascending with `is_primary_preference=true`. Without "in order," no sort
preference is emitted and default quality ranking applies. The preference
description carries the sort direction (e.g., "ordered by release date, earliest
first") and the endpoint LLM translates it into the appropriate query spec.

#### Quality / Notability Priors

A separate, explicit system-level adjustment controls how much the system
biases toward well-known, well-received movies. This is NOT a preference and
NOT baked into the query rewrite.

**Quality and notability/mainstreamness are modeled as separate levers,** each
as a 4-value enum with identical value names but independent semantics:

**Quality prior** (conventional critical/audience reception):

| Value | Meaning | Example triggers |
|-------|---------|-----------------|
| `enhanced` | Quality explicitly important | "critically acclaimed," "best," "masterpiece" |
| `standard` | Implicit expectation of quality — no other signal suppresses it | Most queries without explicit quality/superlative signals |
| `inverted` | User wants conventionally bad movies | "so bad it's good," "guilty pleasures," "B-movies," "campy" |
| `suppressed` | Another preference dominates reranking, quality contributes minimally | "scariest movies ever" (scare-ranking dominates) |

**Notability prior** (mainstream popularity / how well-known):

| Value | Meaning | Example triggers |
|-------|---------|-----------------|
| `enhanced` | Notability explicitly important | "everyone knows," "mainstream," "blockbusters" |
| `standard` | Implicit expectation that popular movies bubble up | Most queries without explicit notability signals |
| `inverted` | User wants less-known movies | "hidden gems," "underrated," "obscure," "lesser known" |
| `suppressed` | Another preference dominates reranking, notability contributes minimally | "scariest movies ever" (scare-ranking dominates) |

**`suppressed` is a second-order inference.** Unlike the other values which are
determined by the query text itself, `suppressed` depends on the rest of the
decomposition — specifically, whether a dominant primary preference exists that
should push system priors to the background. The LLM must assess this *after*
generating dealbreakers and preferences, which is why `prior_assessment`
precedes the enum fields and both come last in the output.

**Superlative interaction:** When a query has a strong superlative preference
(e.g., "scariest movie ever"), that primary preference becomes the dominant
ranking axis. Both quality and notability priors should typically be `suppressed`
in this case — the user cares about the superlative axis, not about general
quality or popularity.

### Reference Movies in the Standard Flow

When a "movies like X but qualifiers" or multi-reference query enters the
standard flow, step 2 uses the LLM's parametric knowledge of the reference
movie(s) to extract concrete attributes — it does not resolve the movie to a
`tmdb_id`. "Movies like Inception but in space" becomes "mind-bending action
movie set in space," and the dealbreakers and preferences are extracted from
that expanded intent. Multi-reference queries ("movies like Inception and
Interstellar") require the LLM to extract and merge traits from both movies,
which may produce multiple step 1 interpretations since different trait
subsets could be emphasized. If the reference movie title is ambiguous
(multiple movies share the name), step 1 extracts the title and the DB
search for exact matches handles disambiguation downstream.

### What Step 2 Does NOT Do

- Does not know schema details (table names, column types, enum values)
- Does not determine exact search parameters (that's step 3's job)
- Does not determine vector space routing (that's step 3's semantic endpoint)
- Does not inject system defaults into the decomposition analysis
- Does not resolve reference movies to `tmdb_id`s — uses parametric knowledge
  to extract the intended attributes instead
- Does not rewrite the query — step 1's `intent_rewrite` serves that purpose
- Does not produce a display label — step 1's `display_phrase` serves that
  purpose

---

## Step 3: Query Translation

Each dealbreaker and preference is translated by its own independent LLM call
(or deterministic function). All calls run in parallel. Each per-endpoint LLM
translates one abstract dealbreaker or preference description into a
**complete query specification**. For dealbreakers, the spec is self-contained
and ready to execute. For preferences, the spec is complete except for
candidate IDs — a WHERE clause placeholder (or vector lookup target set) that
gets filled once candidate assembly provides the IDs. Execution of these specs
happens in step 3.5.

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

### Inputs Per Endpoint Call

Each call receives exactly:

1. **`intent_rewrite`** (from step 1) — The full concrete statement of what
   the user is looking for. Provides disambiguation context for the item being
   translated. "Preferably recent" means different things in different query
   contexts; the intent_rewrite lets the endpoint LLM calibrate.

2. **One dealbreaker or preference item:**
   - `description` — What to translate into a query specification.
   - `routing_rationale` — Concept-type classification label (e.g., "named
     person (actor)", "keyword family: horror"). Helps the endpoint LLM
     identify the correct sub-lookup type without re-interpreting the
     description from scratch.

**Excluded from endpoint inputs:** `direction` (consumed by step 4 execution
code, not by step 3 — see Direction-Agnostic Framing below), `route`
(code-level dispatch flag, not contextual for query formulation),
`is_primary_preference` (affects scoring weights in step 4, not query
formulation), system priors, decomposition analysis, items routed to other
endpoints, the original user query string (redundant with intent_rewrite).

### Direction-Agnostic Framing

**Step 3 LLMs always search for the positive presence of an attribute,
regardless of whether the result will be used for inclusion or exclusion.**
The `direction` field from step 2 is NOT passed to step 3 LLMs — it is
consumed exclusively by step 4 execution code to determine how to apply the
resulting candidate set (include or exclude).

This is a critical architectural invariant. The step 3 LLM's sole job is to
find movies that HAVE the specified attribute. Finding movies that DON'T have
an attribute is never useful — it would return ~150K results and the
inclusion/exclusion logic is a deterministic code concern, not an LLM
concern.

**How this works end-to-end:**
- "action movies not starring Arnold Schwarzenegger" → step 2 emits
  description: "includes Arnold Schwarzenegger in actors", direction:
  exclusion → step 3 receives only the description → searches for movies
  WITH Arnold → step 4 code removes those movies from the candidate set
- "80s action blockbusters except for the ones that don't feature Arnold" →
  step 2 emits description: "includes Arnold Schwarzenegger in actors",
  direction: inclusion → step 3 receives only the description → searches for
  movies WITH Arnold → step 4 code includes those movies

The step 2 `description` field is always written in positive-presence form
("includes X", "involves Y", "is a Z") regardless of the `direction` value.
The description says what to search FOR; the direction says what to do with
the results. This prevents double-negation confusion and keeps each LLM's
task clean.

### Gradient Logic Is Deterministic Code, Not LLM Output

Step 3 endpoint LLMs produce faithful, literal translations of the
requirement. "80s movies" translates to a date range of 1980-01-01 to
1989-12-31. "Under 2 hours" translates to runtime < 120 minutes.

Deterministic code in the execution layer then wraps these literal values
with gradient functions that add graceful decay for near-miss scoring. The
LLM does not know about or produce gradient parameters — it just translates
the intent into the tightest correct specification. This separation keeps
the LLM task simple and the gradient behavior consistent and tunable without
prompt changes.

The same principle applies to semantic scoring: the step 3 semantic LLM
produces vector space selections and expanded search queries. The execution
layer scores candidates using elbow-calibrated thresholds and decay functions
that the LLM has no knowledge of.

### Endpoint 1: Entity Lookup

**Definition:** Resolves named entities — real people, fictional characters,
production companies, and title patterns — to movie IDs via inverted index
posting tables. Each entity type has its own index. The step 2 LLM writes a
plain-English description of the lookup that preserves all user-specified
qualifiers (role type, prominence level, match scope) so the downstream
endpoint can construct the optimal query. Supports both inclusion (movies
matching the entity) and exclusion (movies NOT matching the entity).

**Available entity types:**
- Actors (with prominence qualifiers: lead role, supporting, cameo, minor
  role, or unspecified)
- Directors
- Writers / screenwriters
- Producers
- Composers / musicians
- Characters (specific named characters only — generic character types like
  "police officer" are better served by keyword or semantic endpoints)
- Studios / production companies
- Title patterns (substring, prefix — NOT exact title lookup, which is
  handled by flow routing)

**When to use:**
- The query names a real person in any film crew role
- The query names a specific fictional character by name ("The Joker",
  "Hannibal Lecter", "Batman")
- The query names a production company or studio
- The query describes a title pattern (contains a word, starts with a phrase)
- The query asks to exclude a specific person, character, or studio

**When NOT to use:**
- Generic character type queries ("movies with a cop", "vampire characters")
  — route to `keyword` or `semantic` (character posting tables contain
  credited character names, not role descriptions — a police officer main
  character will be credited by name, not as "cop")
- Franchise name lookup ("Marvel movies", "James Bond franchise") — route to
  `franchise_structure`
- Award lookups of any kind — route to `awards` or `metadata`
- Any structured movie attribute (genre, year, runtime, rating, streaming,
  country, source material) — route to `metadata`
- Semantic/thematic concepts ("funny", "dark", "zombie") — route to `keyword`
  or `semantic`

**Description format:** The step 2 LLM writes a natural-language description
preserving all qualifiers the user specified. Descriptions are always in
positive-presence form — they describe what to search FOR, even when the
dealbreaker direction is exclusion (see Direction-Agnostic Framing in Step 3).
Examples:
- "includes Brad Pitt in actors"
- "has Arnold Schwarzenegger in a lead role"
- "has a character named The Joker"
- "directed by Christopher Nolan"
- "title contains the word 'love'"
- "includes Adam Sandler in actors" (even when direction=exclusion — step 3
  finds movies WITH Adam Sandler, step 4 code handles the exclusion)

**No re-routing responsibility:** The step 3 entity LLM trusts that routing
was done correctly upstream. If it receives a description that seems like it
might better fit another endpoint (e.g., a franchise name), it still performs
the best possible entity lookup for that description. It does not re-route
or refuse.

**Candidate generation (dealbreakers):** Each entity dealbreaker produces an
independent candidate set of movie IDs. **No pool size limit** — the full
result set is the candidate pool. Worst case is ~7K candidates for a major
studio, which is manageable for downstream scoring.

**No-match is a valid result.** When exact matching finds no candidates (e.g.,
a character name that doesn't exist in the database), the endpoint returns an
empty candidate set. There is no fallback to the closest match — an empty set
is the honest answer and the pipeline handles it accordingly.

**Step 3 output schema:** `schemas/entity_translation.py` — `EntityQuerySpec`.
One flat object with `entity_name` (always required), `entity_type` (enum
discriminator), and nullable type-specific fields. Only fields relevant to
the entity type are populated; all others remain null. The LLM's primary job
is name/term generation — producing the correct search strings so that simple
exact or substring matching finds the right movies.

**Dealbreaker scoring:** All sub-types use binary scoring (1.0 or 0.0) except
actors, which use zone-based prominence scoring:

- **Person lookups (non-actor roles)** — Binary 1.0 if the person has a credit
  in that role. See Cross-Posting Table Search below for multi-table behavior.
- **Person lookups (actor role)** — Prominence-scored using billing position.
  See Actor Prominence Scoring below.
- **Character lookups** — Binary 1.0. The LLM generates the standard, most
  common credited form(s) of the character name. Each name variation
  (`entity_name` + `character_alternative_names`) is exact-matched against
  `lex.character_strings`. A match on any variation scores 1.0.
- **Studio lookups** — Binary 1.0. Exact match against
  `lex.lexical_dictionary`.
- **Title pattern lookups** — Binary 1.0. Substring (`LIKE '%pattern%'`) or
  prefix (`LIKE 'pattern%'`) match against movie title strings. No fuzziness.

**Preference scoring:** Same scoring behavior as dealbreakers per sub-type.

#### Per-Sub-Type Search and LLM Output

**Person lookups:** The LLM outputs `entity_name` (corrected/normalized
person name), `person_category` (which role table to search), and optionally
`primary_category` and `actor_prominence_mode`.

Name normalization follows the same rules as the current V1 lexical prompt:
fix spelling errors ("Johny Dep" → "Johnny Depp"), capitalize properly,
complete unambiguous partial names ("Scorsese" → "Martin Scorsese"), but
never add corporate suffixes or infer names not typed by the user. The
normalized name is exact-matched against `lex.lexical_dictionary` after
`normalize_string()` processing.

When `person_category` is a specific role (`actor`, `director`, `writer`,
`producer`, `composer`), only that role's posting table is searched. When
`person_category` is `broad_person`, all 5 role tables are searched with
cross-posting score consolidation (see below). The LLM uses `broad_person`
when it cannot confidently assign a single role from the description and
routing rationale.

**Character lookups:** The LLM outputs `entity_name` (the primary credited
form of the character name) and optionally `character_alternative_names`
(additional credited name variations). Only specific named characters are
routed here — generic character types ("police officer", "vampire") go to
keyword or semantic endpoints instead.

The LLM generates the standard, most common way the character name appears
in movie credits. If the character is genuinely known by multiple credited
forms, multiple variations are listed:
- "The Joker" → `entity_name="The Joker"`,
  `character_alternative_names=["Joker"]`
- "Batman" → `entity_name="Batman"`,
  `character_alternative_names=["Bruce Wayne"]`
- "Hannibal Lecter" → `entity_name="Hannibal Lecter"`,
  `character_alternative_names=[]`
- "T-800" → `entity_name="T-800"`,
  `character_alternative_names=["The Terminator", "Terminator"]`

Fix obvious misspellings only when clearly a misspelling — don't guess if the
name is ambiguous. Each name is normalized then exact-matched against
`lex.character_strings`. A match on any variation returns that movie with a
score of 1.0.

**Studio lookups:** The LLM outputs `entity_name` (corrected/normalized
studio name). Same normalization rules as person names: fix typos, capitalize,
but don't add corporate suffixes ("Disney" stays "Disney", not "Walt Disney
Pictures"). Exact match against `lex.lexical_dictionary`.

**Title pattern lookups:** The LLM outputs `entity_name` (the search pattern
text, no SQL wildcards) and `title_pattern_match_type` (`contains` or
`starts_with`). Execution code normalizes the pattern and constructs the
appropriate `LIKE` query: `LIKE '%pattern%'` for contains, `LIKE 'pattern%'`
for starts_with. Matched against `lex.title_token_strings` using the trigram
GIN index for acceleration. No fuzziness — the LIKE match is the entire
determination.

#### Cross-Posting Table Search

When `person_category` is `broad_person`, the system searches all 5 role
posting tables and deduplicates movie IDs across them. The `primary_category`
field controls score consolidation.

**When to use a specific category (single table):** When the LLM is confident
about the person's role — from an explicit role mention in the description or
a high-confidence routing_rationale. "Directed by Christopher Nolan" →
`person_category="director"`. This is the common case. No cross-posting.

**When to use `broad_person` (all tables):** When the role is ambiguous or
generic ("person" reference, no explicit role). "Christopher Nolan movies"
→ `person_category="broad_person"`. Set `primary_category` to the role the
person is predominantly known for when the LLM is confident about that; leave
null when unsure (all tables contribute equally).

**Cross-posting score consolidation (max-based, no summing):**

- **With a primary_category set:** The primary table's match gets full credit
  (1.0 for non-actor, or prominence score for actor). Non-primary table
  matches get `0.5 × match_score`. The movie's final entity score is the
  **max** across all individual table scores.
- **Without a primary_category (null):** All table matches get full credit.
  The movie's final entity score is the **max** across all tables.

This means a movie appearing in multiple tables never gets more than 1.0 for
a single entity dealbreaker. The max operation ensures no score inflation
from multi-role credits, while the 0.5 discount on non-primary matches
reflects that a non-primary role match is relevant but less on-target.

**Actor table in broad_person searches:** When `broad_person` includes the
actor table, actor results are prominence-scored using `actor_prominence_mode`
(null defaults to DEFAULT). This score flows into cross-posting consolidation
like any other table's score — if actor is the primary_category, it gets full
prominence score; if non-primary, it gets `0.5 × prominence_score`.

#### Actor Prominence Scoring

The actor posting table includes `billing_position` and `cast_size` per
movie-actor pair, enabling prominence-based scoring. The step 3 LLM
determines the appropriate mode based on the description and query context.

**Zone-based thresholds (adaptive to cast size):**

Actor positions are classified into three zones — LEAD, SUPPORTING, MINOR —
using cutoff functions that adapt to cast size via `sqrt` with minimum count
floors. This solves the cast-size adaptation problem: percentage thresholds
fail for small casts (top 10% of 5 = 0.5 actors), and fixed counts fail for
large casts (top 3 in a 200-person cast is too restrictive). `sqrt` gives
sub-linear growth: the number of leads grows with cast size, but slowly.

```
lead_cutoff    = min(cast_size, max(LEAD_FLOOR, round(LEAD_SCALE * sqrt(cast_size))))
supporting_cutoff = min(cast_size, max(lead_cutoff + 1, round(SUPP_SCALE * sqrt(cast_size))))
```

Starting constants: `LEAD_FLOOR=2, LEAD_SCALE=0.6, SUPP_SCALE=1.0`. All
subject to empirical tuning.

| cast_size | lead_cutoff | supp_cutoff | leads | supporting | minor |
|-----------|-------------|-------------|-------|------------|-------|
| 5 | 2 | 3 | 2 | 1 | 2 |
| 10 | 2 | 3 | 2 | 1 | 7 |
| 20 | 3 | 4 | 3 | 1 | 16 |
| 50 | 4 | 7 | 4 | 3 | 43 |
| 100 | 6 | 10 | 6 | 4 | 90 |
| 200 | 8 | 14 | 8 | 6 | 186 |

**Four scoring modes:**

Each mode defines a base score per zone with a within-zone gradient using
zone-relative position (`zp` = 0.0 at top of zone, 1.0 at bottom).

**DEFAULT** — No prominence signal ("Brad Pitt movies", "movies with Brad
Pitt", "movies featuring Brad Pitt", "Brad Pitt action movies"). Leads get
full credit, smooth gradient through supporting and minor.

| Zone | Score | Formula |
|------|-------|---------|
| LEAD | 1.0 (flat) | `1.0` |
| SUPPORTING | 0.85 → 0.7 | `0.85 - 0.15 * zp` |
| MINOR | 0.7 → 0.5 | `0.7 - 0.2 * zp` |

**LEAD** — Explicit "starring" or "lead role" language ("movies starring
Brad Pitt", "Brad Pitt in a lead role"). Lead zone gets full credit, steep
drop outside.

| Zone | Score | Formula |
|------|-------|---------|
| LEAD | 1.0 (flat) | `1.0` |
| SUPPORTING | 0.6 → 0.4 | `0.6 - 0.2 * zp` |
| MINOR | 0.4 → 0.2 | `0.4 - 0.2 * zp` |

**SUPPORTING** — Explicit "supporting role" language ("Brad Pitt in a
supporting role"). Supporting zone gets full credit, others reduced.

| Zone | Score | Formula |
|------|-------|---------|
| LEAD | 0.7 → 0.6 | `0.7 - 0.1 * zp` |
| SUPPORTING | 1.0 (flat) | `1.0` |
| MINOR | 0.6 → 0.35 | `0.6 - 0.25 * zp` |

**MINOR** — Explicit "cameo" or "minor role" language ("Brad Pitt cameo
movies"). Minor zone ramps up — deeper billing = higher score.

| Zone | Score | Formula |
|------|-------|---------|
| LEAD | 0.35 → 0.25 | `0.35 - 0.1 * zp` |
| SUPPORTING | 0.5 → 0.35 | `0.5 - 0.15 * zp` |
| MINOR | 0.7 → 1.0 | `0.7 + 0.3 * zp` |

**Interaction with the scoring formula:** In DEFAULT mode, the 0.5 gap
between a lead (1.0) and the lowest minor is meaningful — more than half of
P_CAP (0.9). In a single-dealbreaker query, leads always outrank minors. In
multi-dealbreaker queries, preferences CAN bridge the gap: a great action
movie where Brad Pitt has a small role can outrank a bad one where he stars.
In LEAD mode, the 0.8 gap (1.0 vs 0.2) is nearly a full dealbreaker miss —
preferences can barely compensate, matching the user's strong "starring"
intent. All zone score values are tunable starting points.

### Endpoint 2: Movie Attributes

**Definition:** Evaluates structured, quantitative movie attributes — numbers,
dates, ranges, and availability data. This endpoint handles attributes that
exist on a continuous scale or represent factual logistical information about
the movie (when it came out, how long it is, where to watch it, how it
performed). It does NOT handle categorical classification (genre, keywords,
source material type), named entity lookup, franchise structure, or award
data — each of those has a dedicated endpoint.

**Available attributes:**
- Release date (year, decade, range, relative — "80s", "recent", "before 2000")
- Runtime (minutes — "under 2 hours", "under 90 minutes", "epic length")
- Maturity rating (G / PG / PG-13 / R / NC-17 — "family friendly", "rated R")
- Streaming availability (provider + access method — "on Netflix", "free to
  stream")
- Audio language ("movies with French audio", "dubbed in Spanish")
- Country of origin ("French films", "produced in South Korea", "European
  movies")
- Budget scale ("low budget", "big budget blockbuster")
- Box office performance ("box office hit", "commercial flop")
- Popularity / mainstream recognition (for notability-driven queries)
- Critical / audience reception score (for quality-driven queries —
  "well-reviewed", "critically acclaimed")

**When to use:**
- The query specifies a numeric or temporal constraint (year, decade, runtime,
  rating level)
- The query references streaming availability or where to watch
- The query references country of origin or cultural-geographic film identity
  ("French films", "European movies", "foreign films")
- The query explicitly mentions audio language, dubbing, or audio tracks
- The query references budget scale or box office performance
- The query references general quality or reception ("well-reviewed", "best
  movies") without naming a specific award
- The query references popularity or mainstream recognition without using the
  word "trending" (trending = `trending` endpoint)

**When NOT to use:**
- Genre ("comedy", "horror", "action") — route to `keyword`
- Source material type ("based on a true story", "book adaptation") — route to
  `keyword`
- Any award reference, including generic "award-winning" — route to `awards`
- Franchise names or franchise structural roles ("sequel", "spinoff") — route
  to `franchise_structure`
- Named entities (people, characters, studios) — route to `entity`
- Thematic or experiential concepts ("funny", "dark", "cozy") — route to
  `keyword` or `semantic`

**Description format:** The step 2 LLM writes a natural-language description
preserving the user's constraint. Examples:
- "released in the 1980s"
- "runtime under 2 hours"
- "rated PG-13 or lower"
- "available on Netflix via subscription"
- "movies with French audio"
- "country of origin is France"
- "European movies"
- "big budget"
- "box office hit"
- "well-reviewed critically"
- "preferably recent" (preference)

**Step 3 LLM translation principle:** The step 3 metadata LLM translates
the user's request as faithfully as possible — it does NOT soften
constraints. "80s movies" translates to a literal date range of
1980-01-01 to 1989-12-31. Deterministic code in the execution layer then
applies softening: generous gates for candidate generation and gradient
decay for scoring. This separation keeps the LLM task simple (faithful
translation) and the softening behavior consistent, tunable, and
reusable across attributes. The existing `db/metadata_scoring.py` gradient
shapes serve as the primary reference for these decay functions.

**Step 3 output schema:** `schemas/metadata_translation.py` —
`MetadataTranslationOutput`. The first field is `target_attribute`
(`MetadataAttribute` enum) — the LLM identifies which single column best
represents the step 2 description before populating any attribute fields.
Execution code queries ONLY the column identified by `target_attribute`
for candidate generation (dealbreakers) and scoring (preferences). This
guarantees one metadata item = one column query = one [0, 1] score, with
no within-dealbreaker multi-attribute combination logic needed.

The remaining 10 nullable attribute fields follow. Complex attributes
(release date, runtime, maturity, streaming, audio language, country of
origin) have sub-objects; simple attributes (budget, box office,
popularity, reception) are direct enum values. The LLM should focus on
populating the field matching `target_attribute`; it may populate
additional fields for context, but only `target_attribute` drives the
query. Only fields matching step 2 descriptions are populated; all others
remain null. Inclusion-only framing — no exclusion lists on any attribute.
Exclusion dealbreakers from step 2 are handled by step 4 scoring code,
not by this endpoint's translation.

**Why single-column targeting:** Step 2 already decomposes multi-attribute
concepts into separate dealbreakers/preferences (e.g., "hidden gems"
becomes a NICHE popularity item and a WELL_RECEIVED reception item). Each
metadata item arriving at this endpoint represents one attribute
constraint. The `target_attribute` field makes this explicit and gives
execution code a clean dispatch key. Compound concepts that span multiple
columns (blockbuster = budget + box office + popularity) are step 2's
responsibility to decompose, not step 3's to handle in one shot.

**Candidate generation (dealbreakers):** Each metadata dealbreaker produces a
candidate set via SQL WHERE clauses (hard filters) or GIN array overlap.
For NLP-extracted constraints, deterministic code applies a generous gate
around the LLM's literal translation for candidate generation purposes
(e.g., "80s movies" → LLM produces 1980-1989, code widens to ~1975-1994).
All dealbreakers include a buffer to ensure the candidate pool is generous
enough for downstream scoring to work effectively.

**Null data handling:** Movies with null data for a dealbreaker attribute
score 0.0 for that dealbreaker — they don't get a valuable boost but are
NOT excluded from the candidate set. For exclusion dealbreakers, null data
means the movie did not match the exclusion condition, so it is not
penalized.

**Preference scoring:** Gradient scoring, not binary. "Under 100 minutes" at
101 minutes scores ~0.95, at 140 minutes scores much lower. Users are
frequently imprecise with numeric constraints, so gradients prevent harsh
cutoffs that miss obviously relevant results. Different attributes have
different softness levels (see Constraint Strictness in open_questions.md).

**Pipeline failure handling:** On endpoint failure (timeout, transient DB
error), retry once. If the retry also fails, return an empty candidate
set — handled the same way as finding no matches. No special error
surfacing needed; the scoring pipeline naturally degrades when one
endpoint contributes nothing.

#### Per-attribute specifications

**Release date** — `movie_card.release_ts` (Unix timestamp, BIGINT, nullable).
LLM outputs: `first_date` (YYYY-MM-DD), `match_operation` (EXACT / BEFORE /
AFTER / BETWEEN), `second_date` (only for BETWEEN). Scoring: linear decay
from range boundary. Grace periods: BETWEEN = max(1yr, min(range_width×0.5,
5yr)); AFTER/BEFORE = 3yr; EXACT = 2yr. Inside range → 1.0, outside →
max(0, 1 - distance_days/grace_days). Candidate generation widens the
literal range by the grace period. The LLM has today's date injected into
its prompt for resolving relative terms ("recent" ≈ last 3 years, "new" ≈
last 1-2 years). Vague terms like "classic" are left to LLM judgment if
step 2 routes them here.

**Runtime** — `movie_card.runtime_minutes` (INT, nullable). LLM outputs:
`first_value` (minutes), `match_operation` (EXACT / BETWEEN / LESS_THAN /
GREATER_THAN), `second_value` (only for BETWEEN). Scoring: linear decay
with 30-minute grace. Inside range → 1.0, outside → max(0, 1 -
distance/30). Candidate generation widens by 30 minutes. Vague terms
("epic length", "long movie") are left to LLM judgment — no special
default guidance.

**Maturity rating** — `movie_card.maturity_rank` (SMALLINT ordinal: G=1,
PG=2, PG-13=3, R=4, NC-17=5, UNRATED=999, nullable). LLM outputs:
`rating` (MaturityRating), `match_operation` (EXACT / GT / LT / GTE / LTE).
Scoring: ordinal distance on the 1-5 scale. In range → 1.0, one rank
away → 0.5, two+ ranks → 0.0. **UNRATED rule:** any query targeting a
rated value (anything other than EXACT UNRATED) excludes UNRATED movies
entirely — score 0.0 AND excluded from candidate generation gate. EXACT
UNRATED matches only UNRATED movies.

**Streaming availability** — `movie_card.watch_offer_keys` (INT[], GIN-indexed,
encoded provider+method keys). LLM outputs: `services` (list of
StreamingService — 20 tracked services), `preferred_access_type`
(SUBSCRIPTION / BUY / RENT, nullable). At least one must be populated.
Inclusion-only — no exclusion list. Scoring: both services and access_type
→ desired method match = 1.0, any method match = 0.5; services only → any
match = 1.0; access_type only → any key with that method = 1.0.
Candidate generation: GIN array overlap. "Free to stream" → services =
[TUBI, PLUTO, PLEX, ROKU], no access_type.

**Audio language** — `movie_card.audio_language_ids` (INT[], GIN-indexed,
334 Language enum values). LLM outputs: `languages` (list of Language,
non-empty). Scoring: any included language present → 1.0, none → 0.0.
Candidate generation: GIN array overlap. **Critical routing rule:** this
attribute is ONLY used when the user explicitly mentions audio, language,
or dubbing. There is no case where language is inferred. "French films" →
country of origin, NOT audio language. "Foreign films" → country of origin
(broad set of non-US countries), NOT "exclude English audio." "Bollywood
movies" → keyword endpoint (cultural tradition), NOT audio language.
"Movies with French audio" or "dubbed in Spanish" → audio language (here).

**Country of origin** — `movie_card.country_of_origin_ids` (INT[], GIN-indexed,
262 Country enum values, array ordering reflects IMDB's order of relevance).
LLM outputs: `countries` (list of Country, non-empty). Supports both single
countries and region-level expansions — the LLM uses parametric knowledge to
enumerate countries for terms like "European movies." Scoring: position-based
gradient per country — position 1 = 1.0, position 2 = ~0.7-0.8, position 3+
= rapid decay toward 0.0. When multiple countries are requested, the movie's
score is `max(score_per_country)` (best score wins, no summing). Candidate
generation: GIN array overlap across all requested country IDs (generous gate),
then scoring applies position gradient. This is the correct endpoint for
cultural-geographic film identity: "French films", "European movies", "foreign
films" all route here via country of origin, not audio language.

**Budget scale** — `movie_card.budget_bucket` (TEXT, nullable: "small" /
"large" / NULL). LLM outputs: `budget_scale` (BudgetSize: SMALL / LARGE).
Scoring: binary match → 1.0, no match → 0.0. NULL = 0.0 (mid-range or
unknown). Candidate generation: SQL WHERE budget_bucket = value.

**Box office performance** — `movie_card.box_office_bucket` (TEXT, nullable:
"hit" / "flop" / NULL, movies < 75 days old always NULL). LLM outputs:
`box_office` (BoxOfficeStatus: HIT / FLOP). Scoring: binary match → 1.0,
no match → 0.0. New movies naturally excluded from box office queries — this
is just the nature of the data, no special handling needed.

**Popularity** — `movie_card.popularity_score` (FLOAT [0,1], sigmoid-normalized
vote count percentile). LLM outputs: `popularity` (PopularityMode: POPULAR /
NICHE, nullable — null when no popularity signal). Scoring: POPULAR =
pass-through of popularity_score; NICHE = 1.0 - popularity_score (inverted).
This replaces the old PopularTrendingPreference boolean — trending is handled
by the separate trending endpoint. "Popular" without temporal signal → here.
"Trending right now" → trending endpoint. "Hidden gems" = NICHE popularity +
WELL_RECEIVED reception.

**Reception score** — `movie_card.reception_score` (FLOAT 0-100, composite:
40% IMDB scaled + 60% Metacritic, nullable). LLM outputs: `reception`
(ReceptionMode: WELL_RECEIVED / POORLY_RECEIVED, nullable — null when no
reception signal). Scoring: WELL_RECEIVED = max(0, min(1, (score-55)/40));
POORLY_RECEIVED = max(0, min(1, (50-score)/40)). "Best movies" maps to
quality_prior: enhanced (step 2 prior), NOT to reception. "Critically
acclaimed" → WELL_RECEIVED here. "Award-winning" → awards endpoint.

### Endpoint 3: Awards

**Definition:** Handles all award-related lookups — from generic "award-winning"
to specific ceremony/category/year queries. Uses two data sources depending on
specificity: the denormalized `award_ceremony_win_ids` array on `movie_card`
for fast generic checks, and the full `movie_awards` table for queries that
name a ceremony, category, outcome, or year. All award-related routing goes
through this single endpoint.

**Search capabilities:**
- Generic award-winning (`award_ceremony_win_ids` GIN overlap — binary: has
  the movie won at any of the 12 tracked ceremonies?)
- Ceremony-specific lookup (filter by ceremony_id: Academy Awards, Golden
  Globes, BAFTA, Cannes, Venice, Berlin, SAG, Critics Choice, Sundance,
  Razzie, Spirit Awards, Gotham)
- Category-specific lookup (award_name + category: "Best Picture", "Best
  Director", "Palme d'Or", "Golden Lion", etc.)
- Outcome filtering (winner vs. nominee)
- Year filtering (award year)
- Compound queries (any combination of the above)

**When to use:**
- The query mentions awards in any form — generic ("award-winning") or
  specific (ceremony, category, year, outcome)
- The query names a specific ceremony ("Oscar", "Cannes", "Sundance")
- The query references winning or being nominated

**When NOT to use:**
- General quality or reception references without mentioning awards
  ("well-reviewed", "critically acclaimed", "best movies") — route to
  `metadata`
- Named entities, even if they won awards ("Leonardo DiCaprio" means actor
  lookup, not award lookup) — route to `entity`
- Thematic or experiential concepts ("prestige film", "Oscar bait vibes") —
  route to `keyword` or `semantic`

**Description format:** Examples:
- "award-winning" (generic)
- "Oscar Best Picture winners"
- "2023 Cannes Palme d'Or"
- "Razzie winners"
- "nominated at Sundance"
- "preferably award-nominated" (preference)

**Candidate generation (dealbreakers):** Produces candidate sets via
deterministic SQL queries on the awards table, or GIN overlap on
`award_ceremony_win_ids` for generic checks.

**Preference scoring:** Can score by award count, ceremony prestige weighting,
or recency of awards.

### Endpoint 4: Franchise Structure

**Definition:** Resolves franchise names and evaluates franchise structural
roles. This is the sole source for anything franchise-related — both "which
franchise is this movie in?" (name resolution) and "what role does this movie
play in its franchise?" (structural filtering). Uses `movie_franchise_metadata`
for structured attributes and `lex.inv_franchise_postings` for fuzzy name
matching.

**Search capabilities:**
- Franchise name resolution (fuzzy-matches against `lineage` and
  `shared_universe` columns via `inv_franchise_postings`)
- Shared universe lookup (distinguishes `lineage` from `shared_universe` —
  MCU is a shared universe, Iron Man is a lineage within it)
- Subgroup matching (`recognized_subgroups` via trigram similarity — "The
  Avengers movies within the MCU")
- Lineage position filtering (sequel, prequel, remake, reboot)
- Spinoff filtering (`is_spinoff` boolean)
- Crossover filtering (`is_crossover` boolean)
- Franchise launcher filtering (`launched_franchise` boolean)
- Subgroup launcher filtering (`launched_subgroup` boolean)

**When to use:**
- The query names a franchise ("Marvel movies", "James Bond", "Star Wars")
- The query references franchise structural roles ("sequels", "prequels",
  "spinoffs", "reboots", "crossovers")
- The query asks about franchise origins ("movies that started a franchise")
- Combined: franchise name + structural role ("Marvel spinoffs") — two
  separate dealbreakers, both routed here

**When NOT to use:**
- Studio or production company names, even when closely associated with a
  franchise ("Pixar movies", "Marvel Studios films") — route to `entity`.
  The franchise is "Toy Story" or "MCU"; the studio is "Pixar" or "Marvel
  Studios."
- Generic "remakes" or "based on a true story" without naming a franchise —
  route to `keyword` (source material type). Franchise structure's
  `lineage_position=remake` only covers remakes within a tracked franchise
  lineage, not all remakes broadly.
- Named people associated with franchises ("Daniel Craig Bond movies") — the
  person routes to `entity`, the franchise routes here. Two separate
  dealbreakers.
- Thematic vibes about franchise-like concepts ("cinematic universe energy",
  "franchise fatigue") — route to `semantic`

**Description format:** Examples:
- "is a Marvel movie" (franchise name)
- "is in the James Bond franchise" (franchise name)
- "all MCU films" (shared universe)
- "Avengers movies" (subgroup)
- "is a sequel" (structural role)
- "spinoff movies" (structural role)
- "movies that started a franchise" (launcher)
- "preferably not a sequel" (preference)

**Step 3 output schema:** `schemas/franchise_translation.py` —
`FranchiseQuerySpec`. Flat model with nullable per-axis fields, preceded
by two scoped reasoning fields that scaffold the high-stakes decisions
(which axes to populate, and how many canonical name variations to
emit). Seven searchable axes, any combination of which may be populated:

1. `lineage_or_universe_names` — up to 3 canonical name variations, always
   searched against both `lineage` and `shared_universe` columns via
   `lex.inv_franchise_postings`. Both columns are searched together because
   the ingest LLM can legitimately place the same brand in either slot
   (Shrek vs. Puss in Boots). Variations cover genuinely different canonical
   names in common use (e.g., "Marvel Cinematic Universe" vs. "Marvel";
   "The Lord of the Rings" vs. "Middle-earth"); trigram fuzzy match already
   handles spelling/punctuation drift so spelling variants are not added.
2. `recognized_subgroups` — up to 3 canonical subgroup-name variations,
   applied as trigram similarity on the normalized subgroup labels of the
   post-lookup result set (3-30 movies). Only valid when
   `lineage_or_universe_names` is populated.
3. `lineage_position` — `LineagePosition` enum (SEQUEL / PREQUEL / REMAKE /
   REBOOT). REMAKE is retained in the enum for ingest fidelity but is not
   typically consumed at search time — generic remake queries route to the
   keyword endpoint via `source_material_type`.
4. `is_spinoff`, `is_crossover`, `launched_franchise`, `launched_subgroup`
   — boolean filters. Direction-agnostic: only `True` or `None` are
   meaningful; `False` is never emitted (exclusion is a step 4 concern).

**Reasoning fields (cognitive scaffolding):** Two scoped reasoning
fields precede the decisions they ground, following the entity and
metadata endpoint pattern:

1. `concept_analysis` (required, emitted FIRST) — evidence-inventory
   trace that quotes signal phrases from `description` and
   `intent_rewrite` and pairs each with the specific axis it
   implicates (franchise name phrase → `lineage_or_universe_names`,
   "sequel/prequel/reboot" → `lineage_position`, "spinoff" →
   `is_spinoff`, "crossover/team-up" → `is_crossover`, "started/
   launched a franchise" → `launched_franchise`, "started/launched a
   phase/saga" → `launched_subgroup`, named sub-series/phase label →
   `recognized_subgroups`). Explicit-absence paths are required —
   "no signal for lineage_position" is a valid trace. Grounds
   axis presence/absence in cited text rather than pattern matching
   on the franchise word. Also surfaces ambiguity ("started the
   MCU" — launcher of franchise vs. subgroup?) so the boolean
   choice is deliberate.

2. `name_resolution_notes` (nullable, emitted BEFORE
   `lineage_or_universe_names`) — brief parametric-knowledge
   inventory of alternate canonical forms of the IP identified in
   `concept_analysis`. Telegraphic form ("Marvel Cinematic Universe;
   Marvel" / "Star Wars" / sentinel "not applicable — purely
   structural"). Scaffolds list length (1 vs. 2 vs. 3) for both
   `lineage_or_universe_names` and `recognized_subgroups` by
   forcing the model to enumerate alternates before committing.
   Excludes spelling/punctuation variants (trigram on the posting
   table handles those). Sentinel-nullable when the query is
   purely structural.

No per-boolean reasoning fields — the step 3 franchise LLM translates
an already-classified query rather than classifying a movie from source
data (unlike the ingest-side `FranchiseOutput`, which has six reasoning
fields). Once `concept_analysis` has cited signal phrases, the
boolean/enum axes follow near-mechanically; adding per-axis traces would
inflate output tokens on simple queries and steal cognitive budget from
the actually-hard decision (canonical-name expansion).

**Prompt reuse:** The step 3 franchise prompt must inherit the
canonical-naming rule, subgroup definition, spinoff/crossover definitions,
and `launched_franchise` four-part test from the ingest-side generator
(`movie_ingestion/metadata_generation/prompts/franchise.py`) so the search
LLM and ingest LLM write into the same slots. Factor shared guidelines
into a common snippet if practical.

**Candidate generation (dealbreakers):** Step 2 sends one distinct concept
per call. If that concept populates multiple axes (e.g., "Marvel spinoffs"
populates both `lineage_or_universe_names` and `is_spinoff`), execution
treats them as **AND** — the movie must match every populated axis.
Compound concepts that span genuinely separate concerns (franchise name AND
a structural role described independently in the query) arrive as separate
dealbreakers from step 2; the continuous scoring model handles the
intersection via the dealbreaker sum.

**Fallback for zero-result franchise names:** No separate fallback — if
none of the 1-3 `lineage_or_universe_names` variations match the posting
table, the dealbreaker produces zero candidates and the continuous scoring
model naturally degrades. The multiple-variation strategy above is the
primary mitigation; obscure franchises that miss anyway are accepted.

**Candidate pool size limit:** None. Franchise result sets are naturally
small (3-30 movies per franchise); no cap needed.

**Scoring (dealbreakers and preferences): binary — 1.0 if the movie
matches every populated axis, 0.0 otherwise.** Every franchise axis is
categorical membership (in the posting-table result set or not; has the
`LineagePosition` enum value or not; boolean is true or not), with no
underlying spectrum. This puts the endpoint alongside awards and keyword
rather than entity or metadata, which have gradients only because their
underlying signals (billing position, release date, runtime) are
genuinely continuous. Multiple name variations under
`lineage_or_universe_names` are alternate attempts at the same concept,
not independent signals — any variation matching scores full credit. A
franchise-recency gradient was considered and dropped for V1.

### Endpoint 5: Keywords & Concept Tags

**Definition:** Evaluates categorical movie classifications through a canonical
concept-family taxonomy backed by deterministic stores. A movie either has the
classification or it doesn't. This endpoint answers "what kind of movie is
this?" through one conceptual taxonomy rather than treating `genre_ids`,
`keyword_ids`, `source_material_type_ids`, and `concept_tag_ids` as four
separate user-facing worlds.

**Data sources:** `movie_card.keyword_ids` (225 curated `OverallKeyword` terms
with definitions) + `movie_card.concept_tag_ids` (25 binary tags across 7
categories) + `movie_card.genre_ids` (27 TMDB genre IDs) +
`movie_card.source_material_type_ids` (10 source material types). All four are
GIN-indexed array columns on `movie_card`.

**Overlap rule:** Some user concepts are backed by more than one deterministic
store. Step 2 treats these as **one concept**. Step 3 may resolve that single
concept to one or more backing IDs or fields. Broad labels like `Action`,
`Horror`, `Documentary`, `Short`, `Film Noir`, `News`, `Biography`, and
`Remake` are multi-backed. Do not split them into separate dealbreakers just
because storage overlaps.

**Step 2 LLM knows:** The canonical concept-family taxonomy below, the overlap
rule, and the main boundary cases that distinguish keyword from metadata,
franchise_structure, and semantic routing.

**Step 3 LLM knows:** The full 225-term `OverallKeyword` vocabulary with
definitions, all 25 concept tag definitions, all genre IDs, and all source
material type IDs. It maps one routed concept to specific IDs and may emit
multi-store resolution when appropriate.

#### Canonical Concept Families

**1. Action / Combat / Heroics**

Action, Action Epic, B-Action, Car Action, Gun Fu, Kung Fu, Martial Arts,
One-Person Army Action, Samurai, Superhero, Sword & Sandal, Wuxia

**2. Adventure / Journey / Survival**

Adventure, Adventure Epic, Animal Adventure, Desert Adventure, Dinosaur
Adventure, Disaster, Globetrotting Adventure, Jungle Adventure, Mountain
Adventure, Quest, Road Trip, Sea Adventure, Survival, Swashbuckler, Urban
Adventure

**3. Crime / Mystery / Suspense / Espionage**

Buddy Cop, Bumbling Detective, Caper, Conspiracy Thriller, Cozy Mystery, Crime,
Cyber Thriller, Drug Crime, Erotic Thriller, Film Noir, Gangster, Hard-boiled
Detective, Heist, Legal Thriller, Mystery, Police Procedural, Political
Thriller, Psychological Thriller, Serial Killer, Spy, Suspense Mystery,
Thriller, Whodunnit

**4. Comedy / Satire / Comic Tone**

Body Swap Comedy, Buddy Comedy, Comedy, Dark Comedy, Farce, High-Concept
Comedy, Parody, Quirky Comedy, Raunchy Comedy, Romantic Comedy, Satire,
Screwball Comedy, Slapstick, Stoner Comedy

**5. Drama / History / Institutions**

Cop Drama, Costume Drama, Drama, Epic, Financial Drama, Historical Epic,
History, Legal Drama, Medical Drama, Period Drama, Political Drama, Prison
Drama, Psychological Drama, Showbiz Drama, Tragedy, Workplace Drama

**6. Horror / Macabre / Creature**

B-Horror, Body Horror, Folk Horror, Found Footage Horror, Giallo, Horror,
Monster Horror, Psychological Horror, Slasher Horror, Splatter Horror,
Supernatural Horror, Vampire Horror, Werewolf Horror, Witch Horror, Zombie
Horror

**7. Fantasy / Sci-Fi / Speculative**

Alien Invasion, Artificial Intelligence, Cyberpunk, Dark Fantasy, Dystopian
Sci-Fi, Fairy Tale, Fantasy, Fantasy Epic, Kaiju, Mecha, Sci-Fi, Sci-Fi Epic,
Space Sci-Fi, Steampunk, Supernatural Fantasy, Sword & Sorcery, Time Travel

**8. Romance / Relationship**

Dark Romance, Feel-Good Romance, Romance, Romantic Epic, Steamy Romance, Tragic
Romance

**9. War / Western / Frontier**

War, War Epic, Western, Classical Western, Contemporary Western, Spaghetti
Western, Western Epic

**10. Music / Musical / Performance**

Classic Musical, Concert, Jukebox Musical, Music, Musical, Pop Musical, Rock
Musical

**11. Sports / Competitive Activity**

Baseball, Basketball, Boxing, Extreme Sport, Football, Motorsport, Soccer,
Sport, Water Sport

**12. Audience / Age / Life Stage**

Family, Coming-of-Age, Teen Adventure, Teen Comedy, Teen Drama, Teen Fantasy,
Teen Horror, Teen Romance

**13. Animation / Anime Form / Technique**

Adult Animation, Animation, Anime, Computer Animation, Hand-Drawn Animation,
Isekai, Iyashikei, Josei, Seinen, Shojo, Shonen, Slice of Life, Stop Motion
Animation

**14. Seasonal / Holiday**

Holiday, Holiday Animation, Holiday Comedy, Holiday Family, Holiday Romance

**15. Nonfiction / Documentary / Real-World Media**

Crime Documentary, Docudrama, Documentary, Faith & Spirituality Documentary,
Food Documentary, History Documentary, Military Documentary, Music Documentary,
Nature Documentary, News, Political Documentary, Science & Technology
Documentary, Sports Documentary, Travel Documentary, True Crime

`News` is canonical here, not under presentation/form. Treat it as
real-world-media classification.

**16. Program / Presentation / Form Factor**

Business Reality TV, Cooking Competition, Game Show, Mockumentary, Paranormal
Reality TV, Reality TV, Short, Sitcom, Sketch Comedy, Soap Opera, Stand-Up,
Talk Show

`Short` is canonical here as a categorical short-film / short-form
classification. Pure runtime requests stay in metadata.

**17. Cultural / National Cinema Tradition**

Arabic, Bengali, Cantonese, Danish, Dutch, Filipino, Finnish, French, German,
Greek, Hindi, Italian, Japanese, Kannada, Korean, Malayalam, Mandarin,
Marathi, Norwegian, Persian, Portuguese, Punjabi, Russian, Spanish, Swedish,
Tamil, Telugu, Thai, Turkish, Urdu

**18. Source Material / Adaptation / Real-World Basis**

Novel Adaptation, Short Story Adaptation, Stage Adaptation, True Story,
Biography, Comic Adaptation, Folklore Adaptation, Video Game Adaptation,
Remake, TV Adaptation

`Biography` is canonical here even though it may also be backed by genre or
keyword storage. Treat "biography" / "biopic" as one real-world-basis
classification concept.

**19. Narrative Mechanics / Endings**

plot_twist, twist_villain, time_loop, nonlinear_timeline, unreliable_narrator,
open_ending, single_location, breaking_fourth_wall, cliffhanger_ending,
happy_ending, sad_ending, bittersweet_ending

**20. Story Engine / Setting / Character Archetype**

revenge, underdog, kidnapping, con_artist, post_apocalyptic,
haunted_location, small_town, female_lead, ensemble_cast, anti_hero

**21. Viewer Response / Content Sensitivity**

feel_good, tearjerker, animal_death

#### Routing Guidance

**When to use:**
- The query names a concept in one of the families above, including broad
  genres, sub-genres, form-factor labels, source material classifications,
  cultural traditions, and concept tags
- The query references a cultural-film tradition ("French cinema", "Hindi
  films", "Bollywood" via Hindi)
- The query references source material or real-world basis ("based on a true
  story", "biopics", "book adaptation", "remakes" broadly)
- The query references animation/anime form or technique ("stop motion",
  "hand-drawn", "anime", "adult animation")
- The query references short-form classification ("short films", "shorts")
- The query matches a concept tag or closely named keyword-family
  classification ("movies with a twist ending", "feel-good movies",
  "coming-of-age", "does the dog die?")

**When NOT to use:**
- Quantitative attributes (year, runtime, rating, streaming, budget, box
  office, reception) — route to `metadata`. `"Under 90 minutes"` is metadata;
  `"short films"` is keyword.
- Named entities (people, characters, studios) — route to `entity`
- Franchise names or franchise-specific structural roles — route to
  `franchise_structure`
- Awards of any kind — route to `awards`
- Subjective experiential qualifiers that describe HOW the movie feels rather
  than WHAT kind of movie it is ("funny", "dark", "cozy", "slow-burn",
  "intense") — route to `semantic`
- Thematic concepts NOT covered by any classification above ("clowns",
  "trains", "female empowerment", "capitalism") — route to `semantic`

**Tricky boundary cases:**

1. **"Zombie movies"** → keyword (Zombie Horror exists). **"Clown movies"** →
   semantic (no clown classification exists). The model must check whether the
   concept appears in the taxonomy before routing here.

2. **"Funny horror movies"** → horror is a keyword dealbreaker; funny is a
   semantic preference. Dark Comedy exists, but it is a classification label,
   not a general-purpose funny qualifier.

3. **"Scary movies"** → route the horror-compatible classification to keyword
   and also capture scariness / horror centrality in the semantic preference.
   **"Scariest movies ever"** is primarily a semantic ranking request.

4. **"Short films" / "shorts"** → keyword (form-factor classification).
   **"Under 90 minutes"** → metadata (runtime constraint).

5. **"French movies"** → keyword (cultural tradition: French). **"Movies with
   French audio"** → metadata (`audio_language_ids`). The keyword captures
   film identity; metadata captures audio-track availability.

6. **"Bollywood movies"** → keyword via Hindi cultural tradition. **"Movies
   with Hindi audio"** → metadata. Cultural identity and audio-track
   availability are different requirements.

7. **"Biographies" / "biopics"** → keyword (real-world basis) even though the
   backend may resolve Biography across multiple stores. Treat it as one
   concept.

8. **"Remakes"** (broadly) → keyword (source material / retelling
   classification). **"Batman remakes"** → `franchise_structure`
   (`lineage_position` within a franchise). Generic remakes route here;
   franchise-specific remakes route to `franchise_structure`.

9. **"Feel-good movies"** → keyword (concept tag: `feel_good`). **"Something
   uplifting and warm"** → semantic (subjective experiential description).

10. **"Coming-of-age"** → keyword (known classification). **"Movies about
    growing up"** → route to keyword only if the phrasing clearly maps to the
    known concept; otherwise semantic.

11. **"Sequel"** does **not** route here. Sequels and prequels always route to
    `franchise_structure`. Broad real-world-basis concepts such as remakes,
    biographies, and true-story movies stay here.

12. **"Critically acclaimed horror"** → horror routes here; critically
    acclaimed routes to `metadata`. Two separate items, two endpoints.

13. **"Award-winning comedy"** → comedy routes here; award-winning routes to
    `awards`. Two separate items, two endpoints.

#### Execution Details

**Candidate generation (dealbreakers):** Produces candidate sets via GIN array
overlap on `keyword_ids`, `concept_tag_ids`, `genre_ids`, or
`source_material_type_ids`. These are binary — a movie either has the
classification or doesn't.

**Preference scoring:** Binary match by default (has the classification = 1.0,
doesn't = 0.0).

**Step 3 mapping may be deterministic:** Because step 2 now receives the full
canonical taxonomy, step 3 may be implemented as deterministic ID resolution
rather than a separate LLM call. This is an implementation detail.

**Thematic centrality — dual dealbreaker + preference:** Some keyword/concept
tag dealbreakers have a meaningful centrality spectrum above the binary
threshold. "Christmas movies" maps to the `Holiday` classification for
candidate generation, but Christmas-centrality (is Christmas the whole premise
or incidental backdrop?) is a useful ranking signal within the passing set.

The guiding principle: **thematic concepts have centrality spectrums;
structural concepts don't.** "Zombie," "heist," "Christmas," and
"coming-of-age" are thematic — how central the concept is to the movie matters
for ranking. "Remake," "based on a true story," and "award-winning" are
structural — there's no meaningful spectrum. When step 2 emits a keyword /
concept-tag dealbreaker for a thematic concept, it should also include that
concept's centrality in the grouped semantic preference description. The
pipeline supports this naturally because the dealbreaker and preference target
different endpoints.

### Endpoint 6: Semantic

**Definition:** Evaluates subjective, thematic, and experiential qualities via
vector similarity across 8 embedding spaces. This endpoint covers the entire
movie across all dimensions, which means it conceptually overlaps with every
other endpoint. However, **semantic is always the last resort for
dealbreakers** — whenever a user's requirement can be evaluated
deterministically by another endpoint (entity, metadata, awards,
franchise_structure, keyword), that endpoint handles it. Deterministic sources
give binary, reliable answers; semantic gives spectrum scores that are useful
for ranking but unreliable for candidate generation. Semantic should only be
used for dealbreakers when no other endpoint can properly handle the concept.

Semantic is freely used for **preferences** (ranking/scoring) even when other
endpoints handle the same concept as a dealbreaker. For example, "horror
movies" produces a keyword dealbreaker (horror-compatible classification) AND
the "scary" qualifier can be a semantic preference for ranking within the
horror results. The dealbreaker generates candidates deterministically; the
preference scores them via vector similarity.

**Data sources:** 8 Qdrant vector spaces (OpenAI `text-embedding-3-small`,
1536 dims):
- **Anchor** (`dense_anchor_vectors`) — Holistic movie fingerprint. Broad
  "movies like X" similarity, general vibes. No subquery — always searched
  with original query. Best for queries that don't emphasize any single
  dimension.
- **Plot Events** (`plot_events_vectors`) — What literally happens.
  Chronological narrative. "Movie where a guy wakes up in a different body",
  "the one with the heist on a train."
- **Plot Analysis** (`plot_analysis_vectors`) — What type of story
  thematically. Genre signatures, themes, concepts. "Redemption stories",
  "movies about grief", "man vs nature conflict."
- **Viewer Experience** (`viewer_experience_vectors`) — What it FEELS like to
  watch. Emotional, sensory, cognitive dimensions. "Something unsettling but
  not gory", "slow burn suspense", "movies that leave you thinking."
- **Watch Context** (`watch_context_vectors`) — WHY and WHEN to watch. Viewing
  occasions and motivations. "Date night movie", "good background movie",
  "something to watch with my parents."
- **Narrative Techniques** (`narrative_techniques_vectors`) — HOW the story is
  told. Craft, structure, storytelling mechanics. "Found footage style",
  "movies that use dramatic irony well."
- **Production** (`production_vectors`) — How/where physically made. Filming
  locations + production techniques. "Movies filmed in New Zealand", "shot on
  16mm", "practical effects heavy."
- **Reception** (`reception_vectors`) — What people thought. Critical and
  audience reception. "Praised for cinematography", "controversial films
  critics hated but audiences loved."

**LLM knows:** What each vector space captures, subquery formulation best
practices per space, space selection logic, the 80/20 subquery/original blend
ratio.

**When to use:**
- **As a dealbreaker:** Only when no deterministic endpoint can evaluate the
  concept ("clowns", "trains", "capitalism", "female empowerment" — thematic
  concepts absent from the keyword taxonomy)
- **As a preference:** Freely, for subjective experiential qualifiers ("funny",
  "dark", "cozy", "intense", "slow-burn"), viewing occasion/context ("date
  night", "background movie"), thematic centrality scoring for keyword
  dealbreakers, plot description matching, production/location queries, nuanced
  reception qualifiers

**When NOT to use as a dealbreaker:**
- The concept exists as a genre, keyword, or concept tag — route dealbreaker
  to `keyword`
- The concept is a named entity — route to `entity`
- The concept is a franchise name or structural role — route to
  `franchise_structure`
- The concept is a quantitative attribute — route to `metadata`
- The concept is award-related — route to `awards`
- The concept is trending/popularity — route to `trending`

**Tricky boundary cases:**

1. **"Scary movies"** → "scary" is a subjective qualifier (semantic
   preference), but the user usually also wants a horror-compatible keyword
   classification. The likely pattern is keyword dealbreaker + semantic
   scare-intensity preference.

2. **"Movies about revenge"** → revenge is a concept tag (keyword endpoint).
   But "movies exploring the psychological toll of revenge" has thematic depth
   beyond the binary tag — the "revenge" dealbreaker routes to keyword, and
   the specificity about psychological toll can be a semantic preference.

3. **"Movies filmed in New Zealand"** → semantic (production vectors). Not
   metadata — there's no filming location column in movie_card.

4. **"Critically acclaimed"** → metadata (reception_score), not semantic. But
   "praised for its cinematography" → semantic (reception vectors), because
   the specific quality being praised can't be evaluated by a numeric score.

5. **"Dark comedy"** → keyword (Dark Comedy genre exists). **"Dark and
   funny"** → "dark" and "funny" are semantic preferences (subjective
   qualifiers). The genre label and the experiential qualifiers are different
   things — one is a classification, the other is a vibe.

**Candidate generation:** Semantic dealbreakers are NOT used for candidate
generation in the standard flow. Deterministic endpoints generate the
candidate pool; semantic dealbreakers score those candidates. Exception: in
the pure-vibe flow (no non-semantic inclusion dealbreakers exist), vector
search becomes the candidate generator.

**Dealbreaker scoring:** Semantic dealbreakers produce a continuous score in
[0, 1] for each candidate, contributing to the dealbreaker sum alongside
deterministic dealbreaker scores (see Phase 4c). The scoring uses
elbow-calibrated thresholds against the global corpus distribution:

- **Above the elbow:** Score = 1.0 (high confidence the movie genuinely has
  this trait).
- **Between elbow and floor:** Gradual decay from 1.0 toward 0.0, calibrated
  by the gap between the maximum similarity score and the elbow value.
- **Below the floor:** Score = 0.0 (no meaningful match — does not contribute
  to the dealbreaker sum).

The elbow is determined dynamically per concept via global corpus search.
If elbow detection fails, a fallback percentage-of-max threshold is used.

**Preference scoring:** Vector similarity scores against relevant spaces. The
LLM determines which spaces are relevant and generates expanded search queries
per space. All preferences (including grouped semantic preferences) are scored
via cosine similarity. For regular preferences, a diminishing-returns curve is
applied (marginal gains decrease at high similarity). For primary preferences,
raw similarity is preserved (full spectrum matters for ranking). Semantic
exclusion dealbreakers use global-elbow-calibrated penalties (see Exclusion
Handling in Phase 4b). The grouped semantic preference can be decomposed into
per-space queries by the step 3 LLM even though it arrives as a single
preference item from step 2.

**Example dealbreakers:** "centers around zombies", "involves female
empowerment themes", "contains car chases" (distinct traits that define what
kind of movie the user wants — scored via elbow-calibrated similarity)

**Example preferences (grouped):** "funny, dark, and thought-provoking with a
cozy date night vibe" (qualifiers on the desired experience, consolidated into
one rich description). Also: thematic centrality qualifiers from keyword
dealbreakers (e.g., "Christmas is central to the story, not just backdrop")
are included in the grouped semantic preference.

### Endpoint 7: Trending

**Definition:** Returns movies that are currently trending / popular right now,
based on precomputed scores from the TMDB weekly trending API stored in Redis.
This is a simple, deterministic signal — no LLM needed for translation. Step 2
flags "trending" intent and execution reads the Redis hash directly.

**Data sources:** Redis `trending:current` hash — precomputed trending scores
[0, 1] for all trending movies, refreshed from TMDB weekly trending API
(top-500 weekly, concave-decay scoring).

**When to use:**
- The query explicitly asks for what's trending, buzzing, or popular *right
  now* — the temporal "now" signal is the key distinguisher ("trending movies",
  "what's popular right now", "what's buzzing")

**When NOT to use:**
- "Popular movies" without temporal "right now" language — route to `metadata`
  (popularity_score). "Popular" alone means all-time notability, not current
  trending.
- Box office performance ("box office hits") — route to `metadata`
  (box_office_bucket)
- Award buzz ("Oscar frontrunners this year") — route to `awards`

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

### Endpoint Return Shape

Every endpoint execution function — for both dealbreakers and preferences,
across all endpoint types — returns the same shape: an `EndpointResult`
containing a list of `ScoredCandidate` objects, each a `(movie_id, score)`
pair with `score ∈ [0, 1]`. The schema lives in `schemas/endpoint_result.py`.

The endpoint function is intentionally thin: it runs its query and emits raw
per-movie scores. All role-specific interpretation — direction (inclusion vs.
exclusion), exclusion mode (hard-filter vs. semantic penalty with E_MULT),
preference weighting (regular 1.0 vs. primary 3.0 vs. prior weights), and
scoring mode (preserved similarity vs. diminishing returns vs. pass-through)
— is handled by the orchestrator when it wraps the returned scores with the
metadata from the originating step 2 item. This keeps endpoint code uniform
and keeps scoring policy in one place.

For deterministic inclusion dealbreakers, the returned `scores` list also
defines that dealbreaker's contribution to the candidate pool in Phase 4a.
For semantic dealbreakers and all preferences, the orchestrator supplies the
assembled candidate IDs as an execution input and the endpoint returns one
entry per supplied ID (non-matches at score 0.0).

---

## Step 4: Assembly & Reranking

### Phase 4a: Candidate Generation Assembly

Collect all candidate sets produced by inclusion dealbreaker execution in
step 3.5. Deterministic endpoints (entity, metadata, awards,
franchise_structure, keyword, trending) each return a set of movie IDs.
Semantic dealbreakers do NOT generate candidates — deterministic endpoints
are the sole source of candidate IDs.

Union and deduplicate across all inclusion dealbreaker candidate sets to
produce the full candidate pool.

**Pure-vibe checkpoint:** If no deterministic inclusion dealbreaker exists
(all inclusion dealbreakers route to semantic), reroute to the pure-vibe flow
(see below). This includes "only semantic inclusions plus deterministic
exclusions" (e.g., "good date night movies not with adam sandler" — date night
is semantic, adam sandler is an entity exclusion).

### Phase 4b: Exclusion Handling

Exclusions are applied after candidate generation. They do NOT contribute to
the dealbreaker sum — they filter or penalize candidates.

**Deterministic exclusions** (entity, metadata, awards, franchise_structure,
keyword): Hard filter. If a movie matches the exclusion criteria, it is
removed from the candidate set entirely. These are binary and reliable — "not
starring Arnold Schwarzenegger" can be definitively evaluated.

**Semantic exclusions:** Cannot be hard-filtered because vector similarity
doesn't give binary results. Instead, semantic exclusions use a
match-then-penalize approach:

1. Run a semantic query for the excluded concept (e.g., "clowns") — searching
   for movies where that attribute IS present and relevant, same as if it were
   an inclusion dealbreaker.
2. Score each candidate using the same elbow-calibrated scoring as semantic
   inclusion dealbreakers: 1.0 above the elbow, gradual decay between elbow
   and floor, 0.0 below the floor. This produces a `match_score` in [0, 1]
   representing how strongly the movie matches the excluded concept.
3. Subtract `E_MULT × match_score` from the movie's final score, where
   **E_MULT** (exclusion multiplier) is initially set to **2.0**, subject to
   empirical tuning.

**Example:** "80s action hits not involving clowns." A movie scoring 0.9 on
"clowns" (highly clown-centric) loses 1.8 points from its final score — a
devastating penalty that pushes it well below movies with clean scores. A
movie scoring 0.2 (tangentially related — maybe a circus scene) loses only
0.4 points — noticeable but not fatal if it otherwise matches well. A movie
scoring 0.0 (below the elbow floor) loses nothing.

**Why penalty instead of hard removal?** Semantic similarity is too noisy to
justify full exclusion. The multiplied penalty strongly discourages matches
without creating binary cliff edges. The 2.0 multiplier means even a moderate
match (0.5) costs a full dealbreaker's worth of score, making exclusions
meaningful.

**Why global corpus for elbow calibration, not candidate-relative:** If none
of the candidates actually contain clowns, a candidate-relative approach
would still penalize whichever candidate is most "clown-adjacent." The global
search calibrates against what "actually has clowns" looks like across the
full corpus, preventing false penalties.

### Phase 4c: Continuous Scoring & Final Ranking

#### Scoring Model

The V2 scoring system uses a continuous model instead of strict tier
partitioning. Every movie receives a single final score:

```
final_score = dealbreaker_sum + preference_contribution - exclusion_penalties
```

Where:
- `dealbreaker_sum` = sum of individual inclusion dealbreaker scores, each
  in [0, 1]
- `preference_contribution` = weighted average of all preference scores and
  system priors, scaled by P_CAP (a constant < 1.0)
- `exclusion_penalties` = sum of `E_MULT × match_score` for each semantic
  exclusion dealbreaker (deterministic exclusions already hard-filtered in
  Phase 4b). E_MULT is initially 2.0.

**The key guarantee:** Since `preference_contribution` is capped below 1.0 and
each dealbreaker contributes up to 1.0, preferences alone can never overcome
one full dealbreaker miss. A movie matching all N dealbreakers fully will
always outscore a movie missing one dealbreaker entirely, regardless of
preference scores. But preferences CAN overcome partial dealbreaker misses,
allowing near-matches with strong preference fit to rank above exact matches
with weak preference fit.

**P_CAP** is initially set to **0.9**, subject to empirical tuning.

**Why continuous instead of tiers:** Strict tier partitioning requires a binary
pass/fail decision for every dealbreaker. This works for naturally binary
endpoints (entity, keyword, franchise, awards) but forces arbitrary threshold
cliffs on metadata dealbreakers (what year stops counting as "the 80s"?) and
made semantic dealbreakers impossible to tier (requiring demotion to
preferences). The continuous model handles all endpoint types uniformly — binary
endpoints produce 0.0 or 1.0, gradient endpoints produce continuous scores, and
the formula handles both without special-casing.

#### Dealbreaker Scoring by Endpoint

Each inclusion dealbreaker produces a score in [0, 1]. The step 3 LLM produces
a faithful literal translation of the requirement; deterministic code in the
execution layer wraps the result with gradient functions for near-miss scoring.

| Endpoint | Scoring behavior |
|----------|-----------------|
| **Entity** | Default binary (1.0/0.0). For actors without a user-specified prominence level, a billing-position gradient: 1.0 for top 15% billing (adjusted by cast size), decaying to floor of 0.8. When prominence is specified, gradient steepens. |
| **Metadata** | Gradient. LLM produces a literal range (e.g., 1980-1989). Code wraps with attribute-specific decay: 1.0 within range, gradual decay outside. Gradient shapes follow the patterns in the existing `db/metadata_scoring.py`. |
| **Awards** | Binary: movie has the specified award/ceremony/outcome or not. |
| **Franchise** | Binary: movie matches the franchise name and/or structural role or not. |
| **Keyword** | Binary: movie has the genre/keyword/concept-tag ID or not. |
| **Semantic** | Continuous [0, 1] via elbow-calibrated cosine similarity. 1.0 above elbow, gradual decay between elbow and floor (calibrated by gap between max similarity and elbow), 0.0 below floor. Cannot generate candidates. |
| **Trending** | Pass-through of precomputed trending score [0, 1]. |

#### Preference Weighting Formula

All preferences and system priors combine into a single preference
contribution via weighted average, then scaled by P_CAP:

```
preference_contribution = P_CAP × ( Σ(w_i × score_i) / Σ(w_i) )
```

When no preferences or priors are active (denominator = 0),
`preference_contribution` = 0.

**Weight assignment:**

| Signal type | Weight | Score source |
|-------------|--------|--------------|
| Regular preference | 1.0 | Endpoint-specific score in [0, 1] |
| Primary preference | 3.0 | Same score, elevated weight dominates the average |
| Quality prior (enhanced) | 1.5 | `reception_score` + `popularity_score` composite — high quality scores high |
| Quality prior (standard) | 0.75 | Same composite, reduced influence |
| Quality prior (inverted) | 1.5 | Endpoint queries/scores for poor reception — poorly received scores high |
| Quality prior (suppressed) | 0.0 | Drops out entirely |
| Notability prior (enhanced) | 1.5 | `popularity_score` — well-known scores high |
| Notability prior (standard) | 0.75 | Same score, reduced influence |
| Notability prior (inverted) | 1.5 | Endpoint queries/scores for obscurity — niche/unknown scores high |
| Notability prior (suppressed) | 0.0 | Drops out entirely |

All weight values are initial starting points subject to empirical tuning.

**Multi-primary handling:** When multiple preferences are marked
`is_primary_preference=true`, they all receive the primary weight (3.0) and
share the elevated influence equally — co-primary, no tiebreaking by list
order.

**Behavior under different query types:**

- **Pure constraint ("80s horror"):** No explicit preferences. Only system
  priors participate (both standard by default). Preference contribution is
  modest — well-known, well-reviewed movies get a small edge within each
  dealbreaker score band.
- **Constraint + vibes ("dark gritty 80s horror"):** Semantic preference for
  "dark gritty" participates alongside standard priors. Movies that are both
  well-reviewed AND dark/gritty score highest within each dealbreaker band.
- **Superlative ("scariest movie ever"):** Primary preference dominates the
  weighted average (weight 3.0 vs regular 1.0). Both priors suppressed
  (weight 0). Ranking is driven almost entirely by the scare-factor score.
- **Multiple primary ("best and scariest"):** Co-primary preferences share
  elevated weight equally. Both contribute strongly to the average.
- **Hidden gems ("underrated horror"):** Notability prior inverted — the
  endpoint scores for obscurity (niche/unknown movies score high) rather than
  popularity. Quality prior remains standard or enhanced.
- **No preferences, both priors suppressed:** `preference_contribution` = 0.
  Ranking is entirely by dealbreaker scores.

#### Preference Scoring by Endpoint

Each preference produces a raw score in [0, 1] for each candidate:

| Endpoint | Scoring behavior |
|----------|-----------------|
| **Entity** | Binary presence (1.0/0.0), with billing-position gradient for actor preferences. |
| **Metadata** | Gradient scoring per attribute type, same decay functions as metadata dealbreakers. |
| **Awards** | Binary match, or count-based for "preferably award-nominated." |
| **Franchise** | Binary match on structural attributes. |
| **Keyword** | Binary (1.0 or 0.0). |
| **Semantic** | Cosine similarity. Diminishing-returns curve for regular preferences; raw preserved similarity for primary preferences. |
| **Trending** | Pass-through of precomputed trending score [0, 1]. |

**Scoring modes applied to raw scores:**

- **Preserved similarity** — For primary preferences. Raw score is the
  ranking signal. "Scariest ever" needs full-spectrum differentiation.
- **Diminishing returns** — For regular semantic preferences. Marginal gains
  decrease at high similarity. Being "somewhat funny" matters more than the
  gap between "very funny" and "extremely funny."
- **Pass-through** — For deterministic preferences (binary or gradient scores
  used as-is).
- **Sort-by** — For explicit sort preferences ("chronological", "most recent
  first"). Expressed as a primary preference with the structured value as the
  dominant ranking signal.

**All preferences are positive (traits to promote).** Negative user intent is
reframed: "not recent" becomes a preference for older films, "not scary"
becomes a preference matching "not scary" in vector space. Anything conceptual
enough to be a hard exclusion ("not zombie," "not with clowns") is a
dealbreaker, not a preference.

---

## Pure-Vibe Flow (No Deterministic Inclusion Anchors)

When step 2 output contains no non-semantic inclusion dealbreakers, the query
enters a separate codepath where vector search is the candidate generator. This
triggers when all inclusion dealbreakers route to `semantic`, regardless of
whether deterministic exclusions also exist. The detection is an explicit flow
control checkpoint immediately after step 2 completes.

### How It Works

1. Vector search generates the candidate pool (since no deterministic endpoints
   can). Each semantic dealbreaker gets its own step 3 LLM call as usual.
2. The step 3 semantic endpoint determines relevant vector spaces and generates
   search queries per dealbreaker/preference item.
3. **Individual searches per concept** across relevant spaces. Synonymous
   concepts are already consolidated by step 2, so each search represents a
   distinct semantic axis. Union top-N results across all searches.
4. **Rescore each candidate** by fetching its distance in all relevant vector
   spaces (not just the space where it was initially retrieved). This catches
   movies that are near-misses in one space but strong in others.
5. Apply a **minimum similarity threshold per space** to avoid noise. A movie
   with 0.15 similarity to "lighthearted" shouldn't get credit for that score.
6. Apply the standard scoring formula: `dealbreaker_sum +
   preference_contribution - exclusion_penalties`. Semantic dealbreaker scores
   contribute to `dealbreaker_sum` using elbow-calibrated scoring (same as the
   standard flow). Preferences and priors contribute to
   `preference_contribution` (same formula, same P_CAP).

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

### 2. Each dealbreaker runs and scores independently

A query can have multiple dealbreakers, and each produces its own candidate
set and its own [0, 1] score. The sum of dealbreaker scores is the primary
ranking signal. This naturally handles partial matches — when no movie meets
all dealbreakers perfectly, the best partial matches surface first based on
their total dealbreaker scores.

**Intentional simplification:** V1 does not add explicit boolean clause/group
logic. For OR-style queries, the continuous scoring model degrades gracefully
— a movie matching either branch gets partial credit. Explicit boolean logic
is deferred unless real query failures prove it necessary.

### 3. Exclusions are separate from inclusions

Inclusion dealbreakers generate candidates and contribute to the dealbreaker
sum. Exclusion dealbreakers filter or penalize candidates after generation.
Exclusions do not contribute to the dealbreaker sum. A movie that passes all
inclusions but fails an exclusion is hard-removed (deterministic) or
penalized (semantic) — a different treatment than simply missing an inclusion
dealbreaker.

### 4. Semantic dealbreakers score but don't generate candidates

When a concept can only be evaluated via vector similarity (no keyword, entity,
or metadata coverage), it cannot reliably generate candidates. Semantic
dealbreakers still produce a continuous [0, 1] score (via elbow-calibrated
cosine similarity) that contributes to the dealbreaker sum, but they do not
generate candidate sets. The operational distinction is candidate generation
only — for scoring purposes, semantic dealbreakers participate fully alongside
deterministic dealbreaker scores. When no non-semantic inclusion dealbreakers
exist, the query enters the pure-vibe flow where vector search generates
candidates.

### 5. A single requirement can produce both a dealbreaker and a preference

Some user requirements are best satisfied by querying two different endpoints
in two different roles — a deterministic dealbreaker for candidate generation
and scoring, plus a semantic preference for degree/centrality ranking. This is
not double-counting: the dealbreaker and preference serve structurally
different purposes (binary membership vs. degree scoring) and target different
endpoints.

**Examples:**
- "Scary movies" → keyword dealbreaker (horror-compatible classification,
  candidate generation +
  1.0 in dealbreaker sum) + semantic preference ("scary," ranks by how scary)
- "Revenge on a bully" → keyword dealbreaker (concept_tag: revenge, candidate
  generation) + semantic preference ("revenge on a bully," ranks by
  specificity)
- "Christmas movies" → keyword dealbreaker (Holiday keyword, candidate
  generation) + semantic preference (Christmas centrality scoring)

The general pattern: the deterministic endpoint answers "does this movie have
this trait?" (membership + dealbreaker score), and the semantic endpoint
answers "how much?" (degree, in the preference contribution). No score
inflation occurs because the dealbreaker and preference feed different parts
of the scoring formula.

**Compound dealbreakers were considered and deferred.** We evaluated whether a
single requirement should ever produce dealbreakers across multiple endpoints
(e.g., "remakes" spanning both source_material_type and
franchise_structure.lineage_position). The genuine cases are narrow enough
that V1 does not introduce a compound-dealbreaker or group_id mechanism.
If real queries expose score inflation from a single concept hitting multiple
dealbreaker endpoints, this can be revisited.

### 6. System-level priors are separate, explicit dimensions

Quality and notability biases are not baked into the query rewrite. They are
independent dimensions that participate in the preference composite with their
own weight schedules (enhanced/standard/inverted/suppressed). Quality uses a
`reception_score` + `popularity_score` composite; notability uses
`popularity_score`. Both contribute to the weighted average in the preference
contribution formula alongside explicit preferences.

### 7. Step 2 interprets intent; step 3 knows schemas

The interpretive LLM (step 2) needs surface-level awareness of endpoints
(what each covers, keyword/concept tag vocabulary) but not schema details.
Per-endpoint LLMs (step 3) need deep schema knowledge but receive
pre-interpreted intent. This split keeps each LLM's task tractable for smaller,
faster models without asking the step 2 model to also carry every exact enum,
matching rule, keyword definition, and low-level source-specific nuance.

### 8. Gradients are deterministic code, not LLM output

NLP-extracted numeric and temporal constraints use gradient scoring rather than
hard cutoffs. This prevents missing obviously relevant results when users are
imprecise (which they frequently are). The gradient logic lives in
deterministic execution code, not in the LLM output — the step 3 LLM produces
a faithful literal translation (e.g., 1980-1989), and execution code wraps it
with per-attribute decay functions. This separation keeps LLM tasks simple and
gradient behavior consistent and tunable without prompt changes.

### 9. Preferences can overcome partial matches, never full matches

The preference contribution is capped below 1.0 (P_CAP = 0.9). Since each
dealbreaker contributes up to 1.0, preferences can never overcome one full
dealbreaker miss. But they CAN overcome partial dealbreaker misses — a movie
from 1990 with great preference fit can outscore a movie from 1985 with weak
preference fit when both are evaluated against an "80s movies" dealbreaker.
This makes the system responsive to overall fit rather than rigidly
partitioning by dealbreaker match count.

### 10. Step 3 LLMs are direction-agnostic

Step 3 LLMs always search for the positive presence of an attribute. They do
not receive the `direction` field and have no knowledge of whether their
results will be used for inclusion or exclusion. The step 2 `description`
field is always written in positive-presence form. Inclusion/exclusion logic
is handled entirely by deterministic code in step 4. This prevents
double-negation confusion, keeps each LLM's task clean, and ensures the
system always searches for "movies WITH X" rather than the logically useless
"movies WITHOUT X" (~150K results). See Direction-Agnostic Framing in Step 3
for the full specification.

---

## Scoring Function Modes

Four scoring modes apply to raw scores produced by endpoints. The mode is
determined by the item's role (dealbreaker vs. preference, primary vs.
regular), not by the endpoint type:

- **Threshold + flatten** — For semantic dealbreaker scoring. Similarity
  above the elbow → 1.0, below → decay toward 0.0 calibrated by the gap
  between max similarity and elbow. Below floor → 0.0. Used for determining
  if a movie genuinely has a semantic trait.
- **Preserved similarity** — For primary preferences. Raw score is the
  ranking signal. "Scariest movie ever" needs to differentiate between
  "very scary" and "somewhat scary."
- **Diminishing returns** — For regular semantic preferences. Marginal gains
  decrease as similarity increases. Being "somewhat funny" matters more than
  the gap between "very funny" and "extremely funny."
- **Pass-through** — For deterministic endpoint scores (binary or gradient).
  Used as-is since the gradient logic already lives in the execution layer.
- **Sort-by** — For explicit sort preferences ("chronological", "most recent
  first"). Expressed as a primary preference where the structured value
  (release date, reception score) is the dominant ranking signal.

---

## Decisions Deferred to Implementation

- Exact step 2 prompt engineering (few-shot examples, chain-of-thought format)
- Exact elbow detection algorithm for semantic dealbreaker scoring and
  exclusion thresholds, with fallback percentage-of-max when elbow detection
  fails
- Step 3 prompt design per endpoint
- Step 3 output schemas per endpoint (metadata endpoint complete:
  `schemas/metadata_translation.py`; entity endpoint complete:
  `schemas/entity_translation.py`; franchise endpoint complete:
  `schemas/franchise_translation.py`; remaining endpoints pending)
- Specific candidate pool size limits per endpoint (entity endpoint: no
  limit; franchise endpoint: no limit)
- Exact gradient decay functions per metadata attribute (date ranges, runtime,
  maturity, etc.), taking heavy inspiration from existing `db/metadata_scoring.py`.
  Includes country-of-origin position gradient (position 1 = 1.0, position 2 =
  ~0.7-0.8, position 3+ = rapid decay) — IMDB array ordering confirmed as
  order of relevance; gradient constants still need empirical tuning
- P_CAP empirical tuning (starting at 0.9)
- E_MULT empirical tuning (semantic exclusion multiplier, starting at 2.0)
- Preference weight values (W_PRIMARY = 3.0, W_PRIOR values) empirical tuning
- Actor prominence zone constants empirical tuning (LEAD_FLOOR=2,
  LEAD_SCALE=0.6, SUPP_SCALE=1.0, per-mode zone score ranges)
- Studio name matching brittleness — currently exact-only after normalization;
  may need LIKE substring or alias table if too many misses in practice
- Whether the keyword endpoint LLM is a separate call or folded into step 2
- Result pagination for long lists: return top 25 initially, cache the full
  candidate/display list in Redis, and allow fetching additional pages via
  pointer IDs or equivalent
- Multi-interpretation trigger criteria
