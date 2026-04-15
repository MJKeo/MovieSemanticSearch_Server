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
is not justified yet because simple match-count tiering already degrades
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
requirement (e.g., "includes Brad Pitt in actors", "is a horror movie",
"does not involve clowns").

*Why included:* The core functional output consumed by step 3 endpoint LLMs as
their task specification. Placed first because it is the most
concrete/extractive field — the model articulates what the requirement IS
before making any classifications about it.

**`direction`** (enum: `inclusion` | `exclusion`, required) — Whether this
dealbreaker generates candidates and contributes to tier count (inclusion) or
filters/penalizes candidates after assembly (exclusion).

*Why included:* The pipeline uses this to determine how the resulting candidate
set is applied. Inclusion dealbreakers contribute +1 to tier count; exclusion
dealbreakers hard-filter (deterministic) or penalize (semantic) after
candidate assembly without counting toward the tier denominator. Placed
second because it is still extractive — usually obvious from query text
markers ("not", "without", "no" → exclusion; everything else → inclusion).

**`routing_rationale`** (string, required) — A brief concept-type
classification label citing why this endpoint handles this concept. Examples:
"named person (actor)", "genre classification", "thematic concept absent from
keyword vocabulary", "franchise structural role."

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
that don't meet these are excluded or tiered down.

**Routing enum definitions (surface-level, no schema details):**

| Route | What it covers | Step 2 LLM needs to know |
|-------|---------------|--------------------------|
| `entity` | Named entities: actors, directors, writers, producers, composers, characters, studios, movie titles | Entity types available |
| `metadata` | Quantitative movie attributes: year, runtime, rating, streaming, language, country, box office, budget, popularity, reception | Field names (not enum values) |
| `awards` | All award lookups: generic "award-winning" through specific ceremony/category/year queries | The 12 ceremony names |
| `franchise_structure` | Franchise name resolution AND structural roles: sequel, prequel, remake, reboot, spinoff, crossover, launched-a-franchise | Franchise names + structural attributes available |
| `keyword` | Categorical classification: genres, source material types, concept tags, and content keywords from curated vocabulary | Trait descriptions covering what the vocabulary can match (not the full enumerated list — see Endpoint 5) |
| `semantic` | Distinct thematic traits the user treats as defining requirements where no deterministic source can evaluate them (e.g., "zombie," "clown," "female empowerment," "car chase") | What the other sources DON'T cover |
| `trending` | Currently trending / popular right now | That trending data exists |

**Critical:** The LLM must understand the limitations of each source. It should
know what the keyword/concept tag vocabulary covers (via trait descriptions, not
the full enumerated list — see Endpoint 5 below) so it can make informed routing
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

Each endpoint has its own LLM (or deterministic function) that receives:
- Step 1's `intent_rewrite` (for full query context)
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
- Characters (exact name or substring/pattern matching for generic roles)
- Studios / production companies
- Title patterns (substring, prefix — NOT exact title lookup, which is
  handled by flow routing)

**When to use:**
- The query names a real person in any film crew role
- The query names a fictional character (specific name or generic role
  description like "police officers")
- The query names a production company or studio
- The query describes a title pattern (contains a word, starts with a phrase)
- The query asks to exclude a specific person, character, or studio

**When NOT to use:**
- Franchise name lookup ("Marvel movies", "James Bond franchise") — route to
  `franchise_structure`
- Award lookups of any kind — route to `awards` or `metadata`
- Any structured movie attribute (genre, year, runtime, rating, streaming,
  country, source material) — route to `metadata`
- Semantic/thematic concepts ("funny", "dark", "zombie") — route to `keyword`
  or `semantic`

**Description format:** The step 2 LLM writes a natural-language description
preserving all qualifiers the user specified. Examples:
- "includes Brad Pitt in actors"
- "has Arnold Schwarzenegger in a lead role"
- "has a character named The Joker"
- "movies with police officer characters"
- "directed by Christopher Nolan"
- "title contains the word 'love'"
- "not starring Adam Sandler" (exclusion)

**Candidate generation (dealbreakers):** Each entity dealbreaker produces an
independent candidate set of movie IDs. Entity matches are binary — a movie
either appears in the posting table for that entity or it doesn't.

**Preference scoring:** Binary presence score (is the entity associated with
this movie?), with actor prominence as a gradient signal when applicable
(billing_position scoring).

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
- Runtime (minutes — "under 2 hours", "short movies", "epic length")
- Maturity rating (G / PG / PG-13 / R / NC-17 — "family friendly", "rated R")
- Streaming availability (provider + access method — "on Netflix", "free to
  stream")
- Audio language ("French language films", "not in English")
- Country of origin ("Korean movies", "British films")
- Budget scale ("low budget", "big budget blockbuster")
- Box office performance ("box office hit", "commercial flop")
- Popularity / mainstream recognition (for notability-driven queries)
- Critical / audience reception score (for quality-driven queries —
  "well-reviewed", "critically acclaimed")

**When to use:**
- The query specifies a numeric or temporal constraint (year, decade, runtime,
  rating level)
- The query references streaming availability or where to watch
- The query references country of origin or audio language
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
- "Korean language films"
- "country of origin is France"
- "big budget"
- "box office hit"
- "well-reviewed critically"
- "preferably recent" (preference)

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

**Candidate generation (dealbreakers):** Handles two kinds of dealbreakers:
- **Franchise name** — fuzzy-matches the lineage/shared_universe columns to
  produce a candidate set of movie IDs.
- **Structural role** — filters on lineage_position, is_spinoff, is_crossover,
  etc.
When both appear in the same query ("Marvel spinoffs"), this endpoint handles
both as separate dealbreakers, and tiering handles the intersection.

**Preference scoring:** Binary match on structural attributes, or gradient
scoring if applicable (e.g., franchise recency).

### Endpoint 5: Keywords & Concept Tags

**Definition:** Evaluates categorical movie classifications — genres, sub-genres,
cultural/language traditions, animation techniques, source material types,
curated content keywords, and binary concept tags. These are all deterministic,
enumerated vocabularies where a movie either has the classification or it
doesn't. This endpoint answers "what kind of movie is this?" through
categorical labels.

**Data sources:** `movie_card.keyword_ids` (225 curated OverallKeyword terms
with definitions) + `movie_card.concept_tag_ids` (25 binary tags across 7
categories) + `movie_card.genre_ids` (27 TMDB genre IDs, moved here from Movie
Attributes — genre is a categorical classification, not a quantitative
attribute) + `movie_card.source_material_type_ids` (10 source material types,
moved here from Movie Attributes — source material is a categorical tag, not
a quantitative attribute). All four are GIN-indexed array columns on
`movie_card`.

**Step 2 LLM knows:** The full list of classification dimensions, category
descriptions, and all individual tags/keywords so it can make accurate routing
decisions about what this endpoint covers and — critically — what it doesn't
cover. The 11 classification dimensions are listed below.

**Step 3 LLM knows:** The full 225-term keyword vocabulary with per-keyword
definitions, all 25 concept tag definitions, all genre IDs, and all source
material type IDs. Maps user concepts to specific IDs.

#### Classification Dimensions

**1. Genre & Sub-genre** (~192 keywords + 27 TMDB genre_ids)

What type of movie/story this is. Ranges from broad genres (Action, Comedy,
Horror) through specific sub-genres (Slasher Horror, Screwball Comedy,
Spaghetti Western) to niche classifications (Giallo, Iyashikei, Gun Fu).

*Action & Combat:* Action, Action Epic, B-Action, Car Action, Gun Fu, Kung Fu,
Martial Arts, One-Person Army Action

*Adventure:* Adventure, Adventure Epic, Animal Adventure, Desert Adventure,
Dinosaur Adventure, Disaster, Globetrotting Adventure, Jungle Adventure,
Mountain Adventure, Quest, Road Trip, Sea Adventure, Survival, Urban Adventure

*Anime & East Asian Traditions:* Anime, Isekai, Iyashikei, Josei, Kaiju, Mecha,
Samurai, Seinen, Shojo, Shonen, Wuxia

*Comedy:* Body Swap Comedy, Buddy Comedy, Comedy, Dark Comedy, Farce,
High-Concept Comedy, Mockumentary, Parody, Quirky Comedy, Raunchy Comedy,
Romantic Comedy, Satire, Screwball Comedy, Sketch Comedy, Slapstick, Stand-Up,
Stoner Comedy

*Crime & Mystery:* Buddy Cop, Bumbling Detective, Caper, Cozy Mystery, Crime,
Drug Crime, Film Noir, Gangster, Hard-boiled Detective, Heist, Mystery, Police
Procedural, Serial Killer, Suspense Mystery, True Crime, Whodunnit

*Documentary:* Crime Documentary, Docudrama, Documentary, Faith & Spirituality
Documentary, Food Documentary, History Documentary, Military Documentary, Music
Documentary, Nature Documentary, Political Documentary, Science & Technology
Documentary, Sports Documentary, Travel Documentary

*Drama:* Biography, Cop Drama, Costume Drama, Drama, Epic, Financial Drama,
Historical Epic, Legal Drama, Medical Drama, Period Drama, Political Drama,
Prison Drama, Psychological Drama, Showbiz Drama, Workplace Drama

*Fantasy & Sci-Fi:* Alien Invasion, Artificial Intelligence, Cyberpunk, Dark
Fantasy, Dystopian Sci-Fi, Fairy Tale, Fantasy, Fantasy Epic, Sci-Fi, Sci-Fi
Epic, Space Sci-Fi, Steampunk, Superhero, Supernatural Fantasy, Sword &
Sorcery, Time Travel

*Holiday:* Holiday, Holiday Animation, Holiday Comedy, Holiday Family, Holiday
Romance

*Horror:* B-Horror, Body Horror, Folk Horror, Found Footage Horror, Giallo,
Horror, Monster Horror, Psychological Horror, Slasher Horror, Splatter Horror,
Supernatural Horror, Vampire Horror, Werewolf Horror, Witch Horror, Zombie
Horror

*Music & Musical:* Classic Musical, Concert, Jukebox Musical, Music, Musical,
Pop Musical, Rock Musical

*Romance:* Dark Romance, Feel-Good Romance, Romance, Romantic Epic, Steamy
Romance, Tragic Romance

*Sports:* Baseball, Basketball, Boxing, Extreme Sport, Football, Motorsport,
Soccer, Sport, Water Sport

*Teen & Coming-of-Age:* Adult Animation, Coming-of-Age, Teen Adventure, Teen
Comedy, Teen Drama, Teen Fantasy, Teen Horror, Teen Romance

*Thriller & Suspense:* Conspiracy Thriller, Cyber Thriller, Erotic Thriller,
Legal Thriller, Political Thriller, Psychological Thriller, Spy, Thriller

*War, Western & Historical:* Classical Western, Contemporary Western, Spaghetti
Western, Swashbuckler, Sword & Sandal, War, War Epic, Western, Western Epic

*Other genre-level:* Animation, Family, History, News, Short, Slice of Life,
Tragedy

*Format / Presentation:* Business Reality TV, Cooking Competition, Game Show,
Paranormal Reality TV, Reality TV, Sitcom, Soap Opera, Talk Show

**2. Culture** (~30 keywords)

The primary cultural/language tradition of the film. "French" means this is a
French-language film in the cultural sense, not merely that French audio exists.

Arabic, Bengali, Cantonese, Danish, Dutch, Filipino, Finnish, French, German,
Greek, Hindi, Italian, Japanese, Kannada, Korean, Malayalam, Mandarin, Marathi,
Norwegian, Persian, Portuguese, Punjabi, Russian, Spanish, Swedish, Tamil,
Telugu, Thai, Turkish, Urdu

**3. Animation Technique** (3 keywords)

How the animation was physically produced (distinct from "Animation" as a
genre): Computer Animation, Hand-Drawn Animation, Stop Motion Animation.

**4. Source Material Type** (10 values)

What the movie is based on: Novel, True Story, Remake, Comic, Video Game, TV
Show, Short Film, Play/Musical, Sequel (non-franchise), Other.

**5. Narrative Structure** (9 concept tags)

Storytelling techniques and structural devices: plot_twist, twist_villain,
time_loop, nonlinear_timeline, unreliable_narrator, open_ending,
single_location, breaking_fourth_wall, cliffhanger_ending.

**6. Plot Archetype** (4 concept tags)

Story pattern the movie follows: revenge, underdog, kidnapping, con_artist.

**7. Setting** (3 concept tags)

Defining setting characteristics: post_apocalyptic, haunted_location,
small_town.

**8. Character Type** (3 concept tags)

Protagonist or cast structure: female_lead, ensemble_cast, anti_hero.

**9. Ending Type** (3 concept tags)

Emotional resolution: happy_ending, sad_ending, bittersweet_ending.

**10. Viewer Experience** (2 concept tags)

How the movie makes you feel: feel_good, tearjerker.

**11. Content Warning** (1 concept tag)

Content flags: animal_death.

#### Routing Guidance

**When to use:**
- The query names a genre or sub-genre ("horror", "romantic comedy", "film
  noir", "spaghetti western")
- The query references a cultural/language film tradition ("Korean movies",
  "Bollywood", "French cinema")
- The query references source material ("based on a true story", "book
  adaptation", "remakes" broadly)
- The query references animation technique ("stop motion", "hand-drawn")
- The query matches a concept tag — narrative structure, plot archetype,
  setting type, character type, ending type, viewer experience, or content
  warning ("movies with a twist ending", "feel-good movies", "does the dog
  die?")
- The query references a sub-genre keyword that exists in the vocabulary
  ("heist movies", "kaiju", "road trip movies")

**When NOT to use:**
- Quantitative attributes (year, runtime, rating, streaming, budget, box
  office, reception) — route to `metadata`
- Named entities (people, characters, studios) — route to `entity`
- Franchise names or franchise-specific structural roles — route to
  `franchise_structure`
- Awards of any kind — route to `awards`
- Subjective experiential qualifiers that describe HOW the movie feels rather
  than WHAT kind of movie it is ("funny", "dark", "cozy", "slow-burn",
  "intense") — route to `semantic`
- Thematic concepts NOT covered by any keyword, concept tag, or genre
  ("clowns", "trains", "female empowerment", "capitalism") — route to
  `semantic`

**Tricky boundary cases:**

1. **"Zombie movies"** → keyword (Zombie Horror exists). **"Clown movies"** →
   semantic (no clown keyword). The step 2 LLM must know the vocabulary to
   make this distinction.

2. **"Funny horror movies"** → "horror" is a keyword dealbreaker, but "funny"
   is a semantic preference (subjective qualifier, not a genre). Dark Comedy
   exists but is a specific genre, not a qualifier.

3. **"French movies"** → keyword (culture: French). **"Movies with French
   audio"** → metadata (audio_language_ids). The keyword captures cultural
   identity; metadata captures audio track availability.

4. **"Remakes"** (broadly) → keyword (source material type). **"Batman
   remakes"** → franchise_structure (lineage_position within a franchise).
   Generic remakes route here; franchise-specific remakes route to
   franchise_structure.

5. **"Feel-good movies"** → keyword (concept tag: feel_good). **"Something
   uplifting and warm"** → semantic (subjective experiential description). The
   concept tag is a binary classification; the semantic query is a vibe.

6. **"Coming-of-age"** → keyword (Coming-of-Age keyword exists). **"Movies
   about growing up"** → could go either way. If the phrasing maps clearly to
   a known keyword/tag, route here. If it's a loose thematic description,
   route to semantic.

7. **"Sequel"** without franchise context → keyword (source material type).
   **"Marvel sequels"** → franchise_structure (lineage_position). The presence
   of a franchise name changes the route.

8. **"Revenge movie"** → keyword (concept tag: revenge). **"Movies about
   getting revenge on a bully"** → the "revenge" aspect routes here as a
   dealbreaker, but the specificity "on a bully" adds nothing to the
   deterministic tag and can be included in a semantic preference if relevant.

9. **"Critically acclaimed horror"** → "horror" routes here (keyword),
   "critically acclaimed" routes to metadata (reception_score). Two separate
   items, two endpoints.

10. **"Award-winning comedy"** → "comedy" routes here (keyword),
    "award-winning" routes to awards. Two separate dealbreakers, two endpoints.

#### Execution Details

**Candidate generation (dealbreakers):** Produces candidate sets via GIN array
overlap on keyword_ids, concept_tag_ids, genre_ids, or source_material_type_ids.
These are binary — a movie either has the classification or doesn't.

**Preference scoring:** Binary match (has the keyword/tag = 1.0, doesn't =
0.0). For dealbreaker-demoted-to-preference scenarios, this is a strong
boosting signal.

**May not require a separate step 3 LLM:** Since step 2 receives the full
vocabulary, the step 3 translation may be a deterministic ID lookup rather
than an LLM call. This is an implementation detail.

**Thematic centrality — dual dealbreaker + preference:** Some keyword/concept
tag dealbreakers have a meaningful centrality spectrum above the binary
threshold. "Christmas movies" maps to the "Holiday" keyword for candidate
generation, but Christmas-*centrality* (is Christmas the entire premise, or
just incidental backdrop?) is a useful ranking signal within the passing set.

The guiding principle: **thematic concepts have centrality spectrums; structural
concepts don't.** "Zombie," "heist," "Christmas," "coming-of-age" are thematic
— how central the concept is to the movie matters for ranking. "Sequel,"
"based on a true story," "award-winning" are structural — there's no meaningful
spectrum. When step 2 emits a keyword/concept tag dealbreaker for a thematic
concept, it should also include that concept's centrality in the grouped
semantic preference description. The pipeline supports this naturally since the
dealbreaker and preference target different endpoints (keyword vs. semantic).
The step 2 prompt must ensure this thematic centrality guidance doesn't conflict
with the semantic preference grouping instructions.

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
movies" produces a keyword dealbreaker (Horror genre) AND the "scary" qualifier
can be a semantic preference for ranking within the horror results. The
dealbreaker generates candidates deterministically; the preference scores them
via vector similarity.

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
  concepts absent from the keyword vocabulary)
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
   preference), but "Horror" is a keyword genre. If the user says "scary"
   they probably want Horror as a keyword dealbreaker PLUS scary as a semantic
   preference for ranking within horror results.

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

**Candidate generation (dealbreakers — with demotion):** Semantic dealbreakers
are NOT used for candidate generation in the standard flow. Any dealbreaker
routed to `semantic` is automatically demoted to a high-weight preference for
scoring purposes (see Semantic Dealbreaker Demotion below). Exception: in the
pure-vibe flow (no non-semantic inclusion dealbreakers exist), vector search
becomes the candidate generator.

**Preference scoring:** Vector similarity scores against relevant spaces. The
LLM determines which spaces are relevant and generates expanded search queries
per space. All preferences (including grouped semantic preferences) are scored
via cosine similarity, possibly with a diminishing returns curve. Semantic
exclusion *dealbreakers* use global-elbow-calibrated penalties (see Exclusion
Handling below). The grouped semantic preference can be decomposed into
per-space queries by the step 3 LLM even though it arrives as a single
preference item from step 2.

**Example dealbreakers (demoted):** "centers around zombies", "involves female
empowerment themes", "contains car chases" (distinct traits that define what
kind of movie the user wants — binary-ish, evaluated via vector thresholding)

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

**System-prior application:** Quality and notability priors are applied as
separate within-tier signals using the 4-value enum
(`enhanced`/`standard`/`inverted`/`suppressed`). When a strong primary
preference exists and the priors are `suppressed`, they contribute minimally
to within-tier ranking.

**All preferences are positive (traits to promote).** Negative user intent is
reframed: "not recent" becomes a preference for older films, "not scary"
becomes a preference matching "not scary" in vector space. Anything conceptual
enough to be a hard exclusion ("not zombie," "not with clowns") is a
dealbreaker, not a preference. This eliminates the need for a direction field
on preferences.

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

### 5. A single requirement can produce both a dealbreaker and a preference

Some user requirements are best satisfied by querying two different endpoints
in two different roles — a deterministic dealbreaker for candidate generation
(+1 tier) and a semantic preference for within-tier ranking. This is not
double-counting: the dealbreaker and preference serve structurally different
purposes (binary membership vs. degree/centrality scoring) and target
different endpoints.

**Examples:**
- "Scary movies" → keyword dealbreaker (Horror genre, candidate generation) +
  semantic preference ("scary," ranks by how scary within the horror set)
- "Revenge on a bully" → keyword dealbreaker (concept_tag: revenge, candidate
  generation) + semantic preference ("revenge on a bully," ranks by
  specificity within the revenge set)
- "Christmas movies" → keyword dealbreaker (Holiday keyword, candidate
  generation) + semantic preference (Christmas centrality, ranks by how
  central Christmas is to the movie)

The general pattern: the deterministic endpoint answers "does this movie have
this trait?" (membership), and the semantic endpoint answers "how much?"
(degree). The dealbreaker contributes +1 to tiering; the preference
influences ranking within tiers. No tier inflation occurs because only the
dealbreaker counts toward the tier denominator.

This pattern is a superset of the thematic centrality pattern described under
Endpoint 5 — thematic centrality is one instance, but the pattern also
covers cases where the preference captures specificity or nuance beyond
what the binary tag can express.

**Compound dealbreakers were considered and deferred.** We evaluated whether a
single requirement should ever produce dealbreakers across multiple endpoints
(e.g., "remakes" spanning both source_material_type and
franchise_structure.lineage_position). The genuine cases are narrow enough
that V1 does not introduce a compound-dealbreaker or group_id mechanism.
If real queries expose tier inflation from a single concept hitting multiple
dealbreaker endpoints, this can be revisited.

### 6. System-level priors are separate, explicit dimensions

Quality bias is not baked into the query rewrite or treated as a preference.
The design now explicitly recognizes that conventional quality and
notability/mainstreamness are distinct dimensions. The finalized decision is to
keep them separate conceptually; the exact field shape remains open.

### 7. Step 2 interprets intent; step 3 knows schemas

The interpretive LLM (step 2) needs surface-level awareness of endpoints
(what each covers, keyword/concept tag vocabulary) but not schema details.
Per-endpoint LLMs (step 3) need deep schema knowledge but receive
pre-interpreted intent. This split keeps each LLM's task tractable for smaller,
faster models without asking the step 2 model to also carry every exact enum,
matching rule, keyword definition, and low-level source-specific nuance.

### 8. Metadata constraints use gradients, not binary filters

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
