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
decomposition happens. Also decides whether multiple searches would
improve browsing value under the query's ambiguity or open-endedness.
Step 1 always emits one `primary_intent` and may emit up to two
`alternative_intents`.

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

### Branching Philosophy

Branching is driven by **browsing value**, not just by whether multiple
readings are theoretically possible. The question is: would running
additional searches materially improve what the user gets to browse?

Step 1 may branch **cross-flow** — a single query can produce intents
routed to different major flows. This matters most when a phrase can be
both a literal title and a natural-language request, or when a standard-
flow query contains one or more concepts that are genuinely open to
interpretation.

The model should preserve **hard constraints** across every emitted
search. Only the genuinely ambiguous or underspecified part of the query
should vary. Examples:
- `"Scary Movie"` can justify both exact-title and standard-flow intents,
  because the phrase naturally supports both searches and the results
  would be very different.
- `"Disney live action movies millennials would love"` should keep
  `Disney` and `live action` fixed while varying only what
  `"millennials would love"` could plausibly mean.

### Ambiguity Scaling

The schema includes a compact `ambiguity_level` enum:

- **`clear`** — One dominant reading. Usually emits only `primary_intent`.
  Typical for exact-title queries, zero-qualifier similarity queries, and
  standard-flow queries whose intent is already concrete.
- **`moderate`** — One main reading is strongest, but at least one useful
  alternative search would improve browsing.
- **`high`** — Multiple strong readings exist, or the query is vague
  enough that trying several distinct searches is clearly better than
  forcing one thin interpretation.

This is **not** a confidence score. It is a compact summary of branching
pressure.

### Inference Policy

Inference is allowed when the query is vague, semantically
underspecified, or uses a loose social/vibe concept that needs to be
fleshed out into something searchable. In these cases, the model may
make logical interpretive leaps about what qualities the user could
mean, as long as each emitted search remains faithful to the query's
hard constraints.

Inference is **not** allowed to guess an exact movie title from a
description. Description-based identification remains in the standard
flow even when the likely movie seems obvious.

This distinction matters because vague queries benefit from several
well-formed searches, while description-based identification should still
let the retrieval system do the matching rather than hard-committing to a
guessed title.

### Crude Language

Step 1 must preserve meaning when the user uses crude, sexual, profane,
or blunt language. The goal is semantic fidelity, not moral cleanup.

- `intent_rewrite` should stay precise and faithful.
- `display_phrase` may be lightly cleaned for UI readability only when
  that does not blur the meaning.
- `display_phrase` can be a bit more lively and human than the rewrite,
  as long as it stays informative and semantically faithful.

### Output Structure

The step 1 LLM produces a `FlowRoutingResponse` (defined in
`schemas/flow_routing.py`). The schema is designed to follow the prompt
authoring conventions established during metadata generation — cognitive
scaffolding field ordering, evidence-inventory reasoning, brief
pre-generation fields, and explicit empty paths for optional list fields.

#### Top-Level Fields

**`ambiguity_analysis`** (string, required) — One concise sentence naming
whether the query is clear, moderately ambiguous, or highly open to
multiple useful searches, and why.

*Why included:* Forces the model to assess branching value before
generating intents. It is framed as an evidence inventory rather than a
justification essay.

**`ambiguity_level`** (enum: `clear` | `moderate` | `high`, required) —
Compact ambiguity classification.

*Why included:* Gives downstream logic and debugging a stable signal
without introducing confidence scoring.

**`hard_constraints`** (list of string, required, may be empty) — Traits
that must remain fixed across every emitted search.

*Why included:* This is the core anti-drift scaffold. It tells the model
what may not vary when generating alternatives.

**`ambiguity_sources`** (list of string, required, may be empty) — The
clause(s), concept(s), or query fragments that are open to interpretation.

*Why included:* Complements `hard_constraints` by explicitly localizing the
part of the query that can vary.

**`primary_intent`** (`PrimaryIntent`, required) — The default search path.
This is the most likely or most useful main reading in movie-search
context.

**`alternative_intents`** (list of 0–2 `AlternativeIntent`) — Optional
additional searches. These may come from genuine alternate readings,
different fleshing-outs of vague semantics, or adjacent exploratory
variations that preserve hard constraints.

#### Primary Intent Fields

`PrimaryIntent` contains, in order:
- `routing_signals`
- `intent_rewrite`
- `flow`
- `display_phrase`
- `title`

This preserves the original cognitive chain: **evidence → intent →
classification → display → extraction**.

`display_phrase` should read like a short, thoughtful UI label rather
than a sterile summary. It should still be immediately understandable,
but can carry a little personality.

#### Alternative Intent Fields

`AlternativeIntent` contains, in order:
- `routing_signals`
- `difference_rationale`
- `intent_rewrite`
- `flow`
- `display_phrase`
- `title`

The extra `difference_rationale` field is deliberately placed before the
alternative's rewrite so the model must commit to what meaningfully
changes before generating the branch. This is the main guardrail against
near-duplicate alternatives.

#### Design Rationale: Field Ordering

The top-level field order follows the model's preprocessing chain:
**assess ambiguity → classify branching pressure → preserve fixed traits →
name what can vary → generate the default intent → generate optional
alternatives**.

The intent-level field ordering keeps the evidence inventory immediately
ahead of the decision it scaffolds. Alternative intents add one brief
pre-generation field (`difference_rationale`) so the model does not drift
into paraphrasing.

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
operates on one standard-flow intent at a time, producing one complete
decomposition per branch. The caller runs it for `primary_intent` and any
`alternative_intents` whose `flow` is `standard`. Exact-title and similarity
branches bypass steps 2-4 entirely.

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
   translated. It is contextual, not authoritative: the step 3 endpoint should
   use it to clarify a vague description, not to add new constraints that the
   item itself did not ask for. "Preferably recent" means different things in
   different query contexts; the intent_rewrite lets the endpoint LLM
   calibrate.

2. **One dealbreaker or preference item:**
   - `description` — What to translate into a query specification. This is the
     authoritative statement of the item being translated.
   - `routing_rationale` — Concept-type classification label (e.g., "named
     person (actor)", "keyword family: horror"). This is a hint, not
     evidence. It helps the endpoint LLM identify the intended sub-lookup type
     without re-interpreting the description from scratch, but it should not
     override the actual evidence in the description.

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
One flat object with shared fields followed by nullable type-specific fields.
The current field order is:
- `entity_type_evidence` — brief evidence inventory grounding the lookup type
  and, for person lookups, whether a specific role is explicitly named
- `name_resolution_notes` — brief note on how the search text was resolved
  (exact user form, typo fix, surname expansion, literal title fragment, etc.)
- `lookup_text` — always required; the canonical string or literal title
  fragment to search for
- `entity_type` — enum discriminator
- `person_category`, `primary_category`, `prominence_evidence`,
  `actor_prominence_mode`, `title_pattern_match_type`,
  `character_alternative_names` — nullable type-specific fields

The LLM's primary job is still search-text generation — producing the correct
literal strings so exact or literal-pattern matching finds the right movies.
The brief pre-generation fields exist to help small models reason before
committing to `lookup_text` and the role/prominence settings.

**Dealbreaker scoring:** All sub-types use binary scoring (1.0 or 0.0) except
actors, which use zone-based prominence scoring:

- **Person lookups (non-actor roles)** — Binary 1.0 if the person has a credit
  in that role. See Cross-Posting Table Search below for multi-table behavior.
- **Person lookups (actor role)** — Prominence-scored using billing position.
  See Actor Prominence Scoring below.
- **Character lookups** — Binary 1.0. The LLM generates the standard, most
  common credited form(s) of the character name. Each name variation
  (`lookup_text` + `character_alternative_names`) is exact-matched against
  `lex.character_strings`. A match on any variation scores 1.0.
- **Studio lookups** — Binary 1.0. Exact match against
  `lex.lexical_dictionary`.
- **Title pattern lookups** — Binary 1.0. Substring (`LIKE '%pattern%'`) or
  prefix (`LIKE 'pattern%'`) match against movie title strings. No fuzziness.

**Preference scoring:** Same scoring behavior as dealbreakers per sub-type.

#### Per-Sub-Type Search and LLM Output

**Person lookups:** The LLM outputs `lookup_text` (corrected/normalized
person name), `person_category` (which role table to search), and optionally
`primary_category` and `actor_prominence_mode`.

Name normalization follows the same rules as the current V1 lexical prompt:
fix spelling errors ("Johny Dep" → "Johnny Depp"), capitalize properly,
complete unambiguous partial names ("Scorsese" → "Martin Scorsese"), but
never add extra name parts not supported by the query/context or infer
entirely different names not typed by the user. The
normalized name is exact-matched against `lex.lexical_dictionary` after
`normalize_string()` processing.

When `person_category` is a specific role (`actor`, `director`, `writer`,
`producer`, `composer`), only that role's posting table is searched. When
`person_category` is `broad_person`, all 5 role tables are searched with
cross-posting score consolidation (see below). The LLM uses `broad_person`
when it cannot confidently assign a single role from the description and
routing rationale.

**Character lookups:** The LLM outputs `lookup_text` (the primary credited
form of the character name) and optionally `character_alternative_names`
(additional credited name variations). Only specific named characters are
routed here — generic character types ("police officer", "vampire") go to
keyword or semantic endpoints instead.

The LLM generates the standard, most common way the character name appears
in movie credits. If the character is genuinely known by multiple credited
forms, multiple variations are listed:
- "The Joker" → `lookup_text="The Joker"`,
  `character_alternative_names=["Joker"]`
- "Batman" → `lookup_text="Batman"`,
  `character_alternative_names=["Bruce Wayne"]`
- "Hannibal Lecter" → `lookup_text="Hannibal Lecter"`,
  `character_alternative_names=[]`
- "T-800" → `lookup_text="T-800"`,
  `character_alternative_names=["The Terminator", "Terminator"]`

Fix obvious misspellings only when clearly a misspelling — don't guess if the
name is ambiguous. Each name is normalized then exact-matched against
`lex.character_strings`. A match on any variation returns that movie with a
score of 1.0.

**Studio lookups:** The LLM outputs `lookup_text` (corrected/normalized
studio name). Same normalization rules as person names: fix typos, capitalize,
but don't add corporate suffixes ("Disney" stays "Disney", not "Walt Disney
Pictures"). Exact match against `lex.lexical_dictionary`.

**Title pattern lookups:** The LLM outputs `lookup_text` (the search pattern
text, no SQL wildcards) and `title_pattern_match_type` (`contains` or
`starts_with`). Execution code normalizes the pattern and constructs the
appropriate `ILIKE` pattern against the full title string: `'%pattern%'`
for contains, `'pattern%'` for starts_with. This is a literal pattern
match, not exact dictionary resolution. Current implementation is case-
insensitive but not diacritic-insensitive.

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
LESS_THAN_OR_EQUAL / GREATER_THAN / GREATER_THAN_OR_EQUAL), `second_value`
(only for BETWEEN). Scoring: linear decay with 30-minute grace. Inside
range → 1.0, outside → max(0, 1 -
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
to specific ceremony/category/year queries. Uses two data sources: the
denormalized `award_ceremony_win_ids` array on `movie_card` for the fast-path
case (generic "has won anything"), and `movie_awards` for all other queries.
All award-related routing goes through this single endpoint.

**Search capabilities:**
- Generic award-winning (fast path: `award_ceremony_win_ids` presence check,
  Razzie excluded)
- Ceremony-specific lookup (Academy Awards, Golden Globes, BAFTA, Cannes,
  Venice, Berlin, SAG, Critics Choice, Sundance, Razzie, Spirit Awards, Gotham)
- Category-concept lookup via the 3-level CategoryTag taxonomy
  (`schemas/award_category_tags.py`): pick at the leaf level for a specific
  concept ("Best Actor" → `lead-actor`), at the mid level for a meaningful
  rollup ("Best Actor or Best Actress" → `lead-acting`), or at the group
  level for whole-bucket queries ("any acting award" → `acting`). Backed
  by the GIN-indexed `movie_awards.category_tag_ids INT[]` column whose
  per-row entries store every ancestor of the row's leaf concept, so a
  single `&&` overlap query handles every specificity.
- Prize name filtering (e.g., "Oscar", "Palme d'Or", "BAFTA Film Award")
- Outcome filtering (winner vs. nominee)
- Year range filtering
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

**Candidate generation (dealbreakers):** Deterministic SQL on `movie_awards`
(or `award_ceremony_win_ids` fast path). Returns one `ScoredCandidate` per
matching movie with score in [0, 1] per the scoring model below.

**Step 3 output schema:** `schemas/award_translation.py` — `AwardQuerySpec`.

**Reasoning fields:**

Two scoped reasoning fields, each placed immediately before the decisions it scaffolds:

| Field | Position | Scaffolds | Purpose |
|---|---|---|---|
| `concept_analysis` | First | `ceremonies`, `award_names`, `category_tags`, `outcome`, `years` | Filter-axis evidence inventory — quotes phrases signalling ceremony, award name, category, outcome, year; explicit absence required per axis. Count/intensity language is NOT inventoried here. |
| `scoring_shape_label` | Between `concept_analysis` and `scoring_mode` | `scoring_mode`, `scoring_mark` | Brief intensity-pattern classification. Forces explicit recognition of the scoring intent before numeric commitment. One of: `"generic award-winning"`, `"specific filter, no count"`, `"explicit count: N"`, `"superlative"`, `"qualitative plenty"`. |

The two fields are deliberately scoped to different evidence types: `concept_analysis` does filter-axis detection (extractive); `scoring_shape_label` does count/intensity classification (interpretive). Mixing them into one field would make `concept_analysis` diffuse and reduce the priming effect on the scoring decisions, which are the hardest in this schema.

`scoring_shape_label` follows the `value_intent_label` pattern from `MetadataTranslationOutput` — brief label (not a sentence or explanation), no consistency coupling instruction. The label primes via autoregressive attention; the model produces `scoring_mode` and `scoring_mark` values independently.

**Scoring model:**

Count unit is distinct prize rows in `movie_awards` (different ceremony,
category, name, or year each count as a separate award).

Two scoring modes controlled by `scoring_mode` + `scoring_mark`:

| Mode | Formula | Use when |
|------|---------|----------|
| `FLOOR` | `1.0 if has_count >= scoring_mark else 0.0` | Specific filters or explicit count floors ("at least 3 wins") |
| `THRESHOLD` | `min(has_count, scoring_mark) / scoring_mark` | Generic "award-winning" or superlative language ("most decorated") |

LLM guidance for picking mode and mark:

| Query shape | Mode | Mark |
|---|---|---|
| Generic "award-winning" | `THRESHOLD` | 3 |
| Specific ceremony/category/name/year, no count language | `FLOOR` | 1 |
| Explicit count ("at least 3 wins") | `FLOOR` | user's number |
| Superlative ("most decorated", "most Oscar-winning") | `THRESHOLD` | 15 |
| Qualitative plenty ("heavily decorated at Cannes") | `THRESHOLD` | 5 |

**Data source dispatch (execution concern):**

Fast path (`award_ceremony_win_ids` presence check, Razzie id stripped):
only when all filter fields are null/empty, `outcome` is `WINNER` or null,
`scoring_mode=FLOOR`, `scoring_mark=1`.

All other cases: `COUNT(*) FROM movie_awards WHERE ...` with active filters.

**Razzie handling:** When `ceremonies` is null/empty, Razzie is excluded from
all counts and filters. When `AwardCeremony.RAZZIE` is explicitly present in
`ceremonies`, it is included — the user intentionally asked for it.

**Filter semantics:** All filter arrays use Cartesian OR — a row matches if
it satisfies any value in each populated array. Filters across arrays are
ANDed (ceremony AND category AND award_name AND outcome AND year range).
`categories` can stand alone without `ceremonies` — execution searches across
all non-Razzie ceremonies that carry that category.

**Scope discipline (prize names):** `award_names` represents named prize
objects directly. If the user says "Oscar-winning", "won an Oscar",
"Palme d'Or winners", or "Golden Lion winner", the stage-3 LLM emits the
canonical stored prize name in `award_names` and does **not** automatically
add the related ceremony. This keeps the spec aligned with the user's phrasing
instead of broadening it to a parent event. `ceremonies` is reserved for
event/festival/awards-body wording such as "at Cannes", "nominated at
Sundance", or "Academy Awards ceremony". Emit both axes only when the query
explicitly names both levels ("Cannes Palme d'Or winners"). The prompt teaches
this as a representation rule, not as a keyword shortcut list.

**Relative year resolution:** The stage-3 LLM has today's date injected
into its prompt (same pattern as the metadata endpoint). Relative terms
— "recent award winners", "this decade", "this year's Oscars" — are
resolved against the injected date, not against training-time knowledge.
Year filters always use calendar years, not award-ceremony season numbers.

**Stage-3 implementation:** `search_v2/stage_3/award_query_generation.py`.
System prompt sections: TASK, DIRECTION_AGNOSTIC (positive-presence
invariant), SCORING_SHAPE (five canonical patterns), FILTER_AXES
(representation rules for prize names vs ceremonies, years), generated
AWARD_NAME_SURFACE_FORMS, generated CATEGORY TAG TAXONOMY,
RAZZIE_HANDLING (default-exclusion with explicit opt-in), and OUTPUT.
The prompt receives `routing_hint` (renamed on the prompt surface from
step 2's `routing_rationale`) and explicitly treats it as contextual
background rather than evidence. All LLM-facing guidance lives in the prompt;
the `AwardQuerySpec` schema carries only brief developer comments (no
`Field(description=...)` entries, since the schema is passed as
`response_format` and descriptions would leak into the prompt surface without
the structured prose context).

### Endpoint 4: Franchise Structure

**Definition:** Resolves franchise names and evaluates franchise structural
roles. This is the sole source for anything franchise-related — both "which
franchise is this movie in?" (name resolution) and "what role does this movie
play in its franchise?" (structural filtering). Uses
`movie_franchise_metadata` only. Name and subgroup matching are exact on the
shared normalized stored strings, so the step 3 LLM emits 1–3 alternate
canonical stored-form attempts when more than one real canonical form is
plausible.

**Search capabilities:**
- Franchise name resolution (exact match against normalized `lineage` and
  `shared_universe` values; multiple alternate canonical forms may be emitted)
- Shared universe lookup (distinguishes `lineage` from `shared_universe` —
  MCU is a shared universe, Iron Man is a lineage within it)
- Subgroup matching (exact match against normalized `recognized_subgroups`
  labels — "The Avengers movies within the MCU")
- Lineage position filtering (sequel, prequel, remake, reboot)
- Structural-flag filtering (`structural_flags` contains `spinoff` and/or
  `crossover`)
- Launcher filtering (`launch_scope=franchise` or `launch_scope=subgroup`)

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
- "launched a subgroup" (launcher)

**Step 3 output schema:** `schemas/franchise_translation.py` —
`FranchiseQuerySpec`. Flat model with nullable per-axis fields,
preceded by one scoped reasoning field that scaffolds the high-stakes
decision of which axes to populate. Five searchable axes, any
combination of which may be populated:

1. `lineage_or_universe_names` — up to 3 canonical name variations,
   searched against both `lineage` and `shared_universe` after shared
   normalization. These are alternate exact stored-form attempts for the
   same IP, not fuzzy variants. Additional entries are added only when a
   genuinely different canonical form is in common use (e.g., "Marvel
   Cinematic Universe" vs. "Marvel"; "The Lord of the Rings" vs.
   "Middle-earth").
2. `recognized_subgroups` — up to 3 canonical subgroup-name variations,
   matched exactly after shared normalization. Only valid when
   `lineage_or_universe_names` is populated.
3. `lineage_position` — `LineagePosition` enum (SEQUEL / PREQUEL /
   REMAKE / REBOOT). REMAKE is retained in the enum for ingest fidelity
   but is not typically consumed at search time — generic remake queries
   route to the keyword endpoint via `source_material_type`.
4. `structural_flags` — optional list containing `spinoff` and/or
   `crossover`. A single concept may populate neither, one, or both.
5. `launch_scope` — nullable enum: `franchise` or `subgroup`. This
   replaces separate booleans and captures mutually exclusive launcher
   intent directly. `launch_scope=subgroup` does not require a named
   subgroup; if the user asks for "movies that launched a subgroup," the
   launcher intent is still actionable even when `recognized_subgroups`
   remains null.

**Reasoning field (cognitive scaffolding):**

1. `concept_analysis` (required, emitted FIRST) — evidence-inventory
   trace that quotes signal phrases from `description` and uses
   `intent_rewrite` only when needed to clarify a vague reference.
   `routing_rationale` is a hint, not evidence. The trace pairs each
   signal with the specific axis it implicates (franchise name phrase →
   `lineage_or_universe_names`, named sub-series/phase label →
   `recognized_subgroups`, "sequel/prequel/reboot" →
   `lineage_position`, "spinoff" / "crossover/team-up" →
   `structural_flags`, "started/launched a franchise" →
   `launch_scope=franchise`, "started/launched a subgroup/phase/saga" →
   `launch_scope=subgroup`). Explicit-absence paths are required —
   "no signal for lineage_position" is a valid trace. Grounds axis
   presence/absence in cited text rather than pattern matching on the
   franchise word.

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
none of the 1-3 `lineage_or_universe_names` variations match the stored
normalized franchise strings, the dealbreaker produces zero candidates and
the continuous scoring model naturally degrades. The multiple-variation
strategy above is the primary mitigation; obscure franchises that miss
anyway are accepted.

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
categories) + `movie_card.source_material_type_ids` (10 source material types).
All three are GIN-indexed array columns on `movie_card`. The `Genre` enum /
`genre_ids` column is intentionally **excluded** from this endpoint entirely —
all 27 TMDB genres are already represented as members of `OverallKeyword` with
identical labels, so keeping `genre_ids` in the mix would be purely redundant
and force a union across columns for no new coverage.

**Overlap rule:** Some user concepts map to multiple enums in principle (e.g.
`Biography` is both an `OverallKeyword` and a `SourceMaterialType`). Step 2
treats these as **one concept**. Step 3 picks exactly one best-fit entry from
the unified classification registry (see "Unified Classification Registry"
below); the registry entry carries its originating source, which determines
the single backing `movie_card` column at execution time. Each
`UnifiedClassification` member resolves to exactly one (column, id) pair, so
execution is always a single-column lookup. Broad labels like `Action`,
`Horror`, `Documentary`, `Short`, `Film Noir`, `News`, `Biography`, and
`Remake` resolve to a single registry entry each; do not split them into
separate dealbreakers just because multiple enums could in theory back them.

**Step 2 LLM knows:** The canonical concept-family taxonomy below, the overlap
rule, and the main boundary cases that distinguish keyword from metadata,
franchise_structure, and semantic routing.

**Step 3 LLM knows:** The full unified classification registry (259 entries
across `OverallKeyword`, `SourceMaterialType`, and `ConceptTag` — see
"Unified Classification Registry" below) with per-entry definitions, presented
grouped by the 21 canonical concept families. It picks exactly one registry
entry per dealbreaker and cannot abstain — routing already happened upstream,
so the endpoint's job is "choose the best fit" even when the match is
imperfect. Output is a single enum member (the unified registry name), not a
list and not multi-store.

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

#### Unified Classification Registry

Step 3 selects from a single `UnifiedClassification` `StrEnum` built at import
time from three source enums:

- `OverallKeyword` (225 terms, backing column `keyword_ids`)
- `SourceMaterialType` (10 terms, backing column `source_material_type_ids`)
- `ConceptTag` (25 storable tags across 7 category enums, backing column
  `concept_tag_ids`)

`Genre` (27 terms) is excluded from the LLM surface because every genre is
already present in `OverallKeyword` with an identical label — redundant on
the selection surface. Total unified members: 259 after collision dropping
(see below).

**Collision rule:** When a member name appears in more than one source enum,
the `OverallKeyword` entry wins and the others are dropped. `OverallKeyword`
has broader coverage and is the stronger retrieval signal. In the current
vocabulary the only real collision is `BIOGRAPHY` — present as both an
`OverallKeyword` term and a `SourceMaterialType`; the LLM sees it as a
keyword.

**Registry entry fields:** `name`, `display`, `definition` (LLM-facing label
used in the step 3 prompt), `source` (`keyword` | `source_material` |
`concept_tag`), `source_id` (ID within the source's backing column), and
`backing_column` (derived from `source`). Execution code calls
`entry_for(member)` to unpack these.

Implementation: `schemas/unified_classification.py`. Unit tests in
`unit_tests/test_unified_classification.py` parametrize over every source
enum member and verify registration, display, source, and source ID.

#### Execution Details

**Candidate generation (dealbreakers):** The LLM's selected registry entry
maps to exactly one `movie_card` array column and one ID. Candidate generation
issues a single GIN `&&` overlap query against that column (e.g., `keyword_ids
&& ARRAY[96]` for `HORROR`, `source_material_type_ids && ARRAY[9]` for
`REMAKE`, `concept_tag_ids && ARRAY[1]` for `PLOT_TWIST`). Binary — a movie
either has the ID or doesn't. No union across columns, no dual-backing into
`genre_ids` or any other column: the single `(backing_column, source_id)`
pair resolved from the chosen `UnifiedClassification` member is the entire
query.

**Preference scoring:** Binary match by default (has the classification = 1.0,
doesn't = 0.0).

**Step 3 is an LLM call, not deterministic.** Although step 2 already routes
the concept, the LLM is retained at step 3 because step 2 does not carry the
full per-entry definitions needed to disambiguate close-but-distinct members
of the 259-term registry (e.g., `FEEL_GOOD_ROMANCE` keyword vs. `FEEL_GOOD`
concept tag, or `TRUE_STORY` vs. `BIOGRAPHY`). The step 3 LLM receives the
registry grouped by canonical concept family, each entry with its definition,
and selects the single best fit.

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

**Data sources:** 8 Qdrant vector spaces (OpenAI `text-embedding-3-large`,
3072 dims). Dealbreakers draw from the 7 non-anchor spaces only; preferences
may use all 8 for scoring. Every space (including anchor when used for
preference scoring) is searched with a generated query tailored to that space's
structured-label shape — the original user query is never used directly for
retrieval.
- **Anchor** (`dense_anchor_vectors`) — Holistic movie fingerprint. Broad
  "movies like X" similarity, general vibes. Dealbreakers never select this
  space — it is intentionally too diffuse for a single-dimension pass/fail
  judgment. Best for preferences that don't emphasize any single dimension.
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

**LLM knows:** What each vector space captures (via the per-space taxonomy
documented in §Preliminary Step below), per-space query formulation best
practices in the matching structured-label shape, space selection logic, and
the two-level space weight scale for preferences (`central` / `supporting`).

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

#### Execution Scenarios

Four scenarios govern how the semantic endpoint executes, keyed on the
composition of step 2's output. Dealbreaker-side and preference-side logic
compose independently — a query picks one dealbreaker scenario AND one
preference scenario based on what step 2 emitted.

**Dealbreaker-side scenarios** (applies when ≥1 semantic dealbreaker exists):

| Scenario | Trigger | Candidate generation | Similarity source | Score transform | Reranking role |
|----------|---------|----------------------|-------------------|-----------------|----------------|
| **D1: Score-only** | ≥1 non-semantic inclusion dealbreaker exists | N/A — candidates come from non-semantic endpoints | Single Qdrant `query_points` call on the selected space with a `HasIdCondition` filter over the full candidate list (movie_id is the point ID) | Global calibration probe → elbow/floor → threshold-plus-flatten applied to candidate cosines | Contributes to `dealbreaker_sum` (each item ∈ [0, 1]) |
| **D2: Candidate-generating** | Zero non-semantic inclusion dealbreakers, ≥1 semantic inclusion dealbreaker | Each semantic dealbreaker independently runs top-N per (dealbreaker, space) against the full corpus; union = candidate pool | Candidate pool's own cosines from the generating search (the top-N probe serves as both calibration sample and candidate pool — one Qdrant call, not two). **No cross-dealbreaker scoring** — each dealbreaker scores only the movies it retrieved | Global calibration probe → elbow/floor → threshold-plus-flatten applied to each candidate's cosine for its own dealbreaker's space | Contributes to `dealbreaker_sum` (each item ∈ [0, 1]) |

**Preference-side scenarios** (applies when ≥1 semantic preference group exists):

| Scenario | Trigger | Candidate generation | Similarity source | Score transform | Reranking role |
|----------|---------|----------------------|-------------------|-----------------|----------------|
| **P1: Score-only** | ≥1 inclusion dealbreaker exists (semantic or non-semantic) | N/A — candidates already assembled | One `query_points` call per selected space with a `HasIdCondition` filter over the candidate list; calls run in parallel | Raw weighted-sum cosine across selected spaces, normalized by Σw (no elbow, no pool normalization) | Contributes to `preference_contribution`, scaled by P_CAP |
| **P2: Candidate-generating** | Zero inclusion dealbreakers, ≥1 semantic preference | Each selected space in the grouped preference runs top-N against the full corpus; union across spaces = candidate pool | Per-space cosines seed from each space's own top-N probe; remaining union members missing from a given space are filled via one `HasIdCondition` `query_points` call on that space (only non-empty fills fire) | Same as P1 — raw weighted-sum cosine normalized by Σw | Contributes to `preference_contribution`, scaled by P_CAP. `dealbreaker_sum = 0` |

**Edge case (exclusion-only):** zero inclusion dealbreakers AND zero
preferences (only exclusion dealbreakers). Falls back to browse-style
candidate generation (top-K by `0.6 × reception_score + 0.4 ×
popularity_score`), apply exclusions in Phase 4b, rank by the same quality
composite. See "Exclusion-Only Edge Case" below.

**Why D1 and D2 both use global elbow calibration:** The scoring function
must stay identical regardless of whether the dealbreaker also generated
candidates. Otherwise the same semantic dealbreaker would score differently
across queries depending on whether a keyword dealbreaker happened to
coexist. Global calibration against the full corpus is the only invariant
available — candidate-relative calibration would systematically underrate
dealbreakers when the candidate pool is already concept-rich, and overrate
them when the pool is thin.

**Why D2 does not cross-score across dealbreakers:** If three semantic
inclusion dealbreakers coexist, each one runs its own top-N probe and
scores only the movies that probe retrieved. Movies retrieved by dealbreaker
A but not by B contribute their A-score and 0 for B — the same asymmetry
deterministic dealbreakers already have (a movie matching "Marvel" but not
"80s" contributes full credit for Marvel and a graded score for 80s).
Cross-scoring would silently elevate D2 toward "every candidate gets every
dealbreaker's score" semantics and blur what step 2 classified as
independent requirements. If additional reranking across semantic concepts
is desired, step 2 should emit a semantic preference in addition to the
dealbreaker, and the preference carries the cross-space scoring via P1/P2.

**Why P2 keeps preference semantics (not dealbreaker semantics):** Step 2
deliberately classified these items as preferences — they are ranking
signals, not pass/fail requirements. Changing the final-ranking role based
on the candidate-generation mechanism would silently override step 2's
classification. Preferences stay preferences end-to-end; the only thing
that shifts in P2 is who produces the candidate IDs.

**Preliminary step: space identification (always first).** Every step 3
semantic call starts by identifying which vector space(s) the concept actually
lives in, using a canonical space taxonomy embedded in the prompt. The
taxonomy is authored as a module-level docstring on each `create_*_vector_text`
function in `movie_ingestion/final_ingestion/vector_text.py` and imported
verbatim into the step 3 prompt at build time. This is a convention, not code
generation. Each space's entry contains four parts:

1. **Purpose** — one sentence on what the space captures.
2. **What's embedded** — the structured labels actually present in the
   embedded text (e.g., for `narrative_techniques`: `narrative_archetype`,
   `narrative_delivery`, `pov_perspective`, `information_control`,
   `characterization_methods`, `character_arcs`, `audience_character_perception`,
   `conflict_stakes_design`, `additional_narrative_devices`). Grounds the
   model in the exact shape it should generate queries in.
3. **Boundary** — an explicit *not-this* line (e.g., "HOW the story is told,
   NOT what happens"). Misroute prevention.
4. **2–3 canonical example queries** that belong in this space.

Re-audit this convention whenever vector text generation logic changes. This
replaces V1's "ask every space to rewrite the query for itself" pattern, which
invited the LLM to fabricate signal in spaces that don't carry it (flaw #15).

Space selection rules:

- **Dealbreakers:** strictly 1 space from the 7 non-anchor spaces. If the
  LLM is genuinely split between two spaces, that's a signal the concept
  isn't a clean dealbreaker and should have gone to another endpoint. Always
  require at least one pick — zero-space output is not allowed; pick the best
  option even if imperfect.
- **Grouped preferences:** 1+ spaces chosen from all 8 (anchor allowed).
  Each selected space receives a two-level categorical weight
  (`central` → 2, `supporting` → 1). There is no `minor` option — if a
  space's signal isn't at least supporting meaningfully, don't select it.
  This prevents diluting preference scores by spreading weight across many
  marginal spaces. Anchor fits naturally for broad-vibe qualifiers ("cozy,"
  "funny and lighthearted") where anchor's thematic summary carries real
  signal.
- **Semantic exclusions:** same rule as dealbreakers — strictly 1 space from
  the 7 non-anchor spaces.

**Dealbreaker scoring:** Single-space, elbow-calibrated against the global
corpus distribution. For each dealbreaker:

1. LLM picks 1 space from the 7 non-anchor spaces and generates a query
   tailored to that space's structured-label shape. This is the only query
   — no merging with the original user query.
2. **Global calibration search:** run the query against the full corpus
   (no filters), top-N, against the selected space only. Detect the elbow
   (1.0 mark) and floor (0.0 mark) from the sorted similarity distribution.
3. **Candidate scoring:** for each candidate generated by deterministic
   endpoints, fetch cosine similarity in that space. Transform:
   - `sim ≥ elbow` → score = 1.0
   - `floor < sim < elbow` → linear decay from 1.0 to 0.0
   - `sim ≤ floor` → score = 0.0
4. Score contributes full value to `dealbreaker_sum` (not capped by P_CAP).

**Elbow caching — deferred.** The current implementation does not cache
elbow/floor; every dealbreaker invocation pays for one unfiltered top-N
probe. A future Redis cache keyed by `(query_text, space_name,
embedding_model_version, space_prompt_version, corpus_version)` with a 24h
TTL and invalidation on corpus rebuilds drops in at the call to
`_detect_elbow_floor` in `semantic_query_execution.py` without
restructuring the executor. Version components are enumerated now so an
embedding-model swap, a space-prompt revision, or a corpus regeneration
cannot silently reuse stale calibration once the cache does land.

**Preference scoring:** Multi-space weighted cosine. The LLM picks 1+ spaces,
generates one query per selected space (no per-concept decomposition — a
single query per space that absorbs all concepts routed to it and phrases
them in that space's native vocabulary), and assigns each selected space a
categorical weight of `central` (maps to 2) or `supporting` (maps to 1).
For each candidate:

```
pref_score = Σ(w_space × cosine_space) / Σ(w_space)
```

**No pool normalization, no diminishing-returns curve.** Raw weighted cosine
is the ranking signal. This is deliberate: pool normalization would falsely
promote the best-in-a-bad-pool candidate to 1.0, erasing the "nobody here
actually matches" signal. Example: if candidates score 0.20–0.25 on
"clown-ness" while true clown movies elsewhere in the corpus score 0.75+,
raw cosines correctly produce low preference contributions; normalization
would misleadingly rank a 0.25 as a "perfect clown match."

This replaces the V1 diminishing-returns curve for regular preferences and
the "preserved similarity for primary" variant — `is_primary_preference` now
only affects Phase 4c weighting (3.0 vs 1.0), not the score shape in step 3.

**Semantic exclusions** use the dealbreaker scoring path identically
(single space, global elbow calibration, [0,1] score). The resulting match
score is multiplied by E_MULT and subtracted from the final score in
Phase 4b.

**Example dealbreakers:** "centers around zombies", "involves female
empowerment themes", "contains car chases" (distinct traits that define what
kind of movie the user wants — scored via elbow-calibrated similarity)

**Example preferences (grouped):** "funny, dark, and thought-provoking with a
cozy date night vibe" (qualifiers on the desired experience, consolidated into
one rich description). Also: thematic centrality qualifiers from keyword
dealbreakers (e.g., "Christmas is central to the story, not just backdrop")
are included in the grouped semantic preference.

**Step 3 output schemas:** `schemas/semantic_translation.py` —
`SemanticDealbreakerSpec` (one call per semantic dealbreaker or exclusion;
covers D1 + D2) and `SemanticPreferenceSpec` (one call per grouped semantic
preference; covers P1 + P2). Both emit concrete per-space `*Body` objects
(`schemas/semantic_bodies.py`) via discriminated unions keyed on `space`, so
query-side vectors embed into the same structured-label format as document-
side vectors. No free-form `query_text` strings.

**Reasoning fields:**

Two reasoning fields per spec — one top-level evidence inventory, one brief
label placed immediately before the decisions it primes. Rationalization-
after-the-fact fields are deliberately avoided; every reasoning field exists
only to ground a specific downstream commit in cited evidence or prime the
body shape.

*`SemanticDealbreakerSpec`:*

| Field | Position | Scaffolds | Purpose |
|---|---|---|---|
| `signal_inventory` | First | `body.space` (discriminator commit) | Evidence inventory — cite concrete phrases from the description and, per phrase, note which of the 7 non-anchor spaces it genuinely implicates. Explicit empty-evidence path required ("no phrase implicates production" is a valid trace, not a signal to fabricate). Guards against surface-pattern misrouting — "female empowerment" looking like `viewer_experience` because the words feel emotional, when it is really `plot_analysis`. |
| `target_fields_label` | Between `signal_inventory` and `body` | Sub-fields populated inside `body.content` | Brief label form (2–6 words, not a sentence). Names which structured sub-fields within the likely-chosen space the concept will surface in (e.g., for zombies in plot_analysis: `"conflict_type, thematic_concepts"`; for "filmed in New Zealand" in production: `"filming_locations"`). Primes selective sub-field population so the body does not fabricate signal into sub-fields the concept does not cover. Brief-label form per the "brief pre-generation fields, no consistency coupling" convention — sentence-form rationales dominate attention and template the body's language. |

No separate `space` reasoning field beyond `signal_inventory`. Adding one
would be the rationalization-after-the-fact failure mode the conventions
caution against — the space commit is already grounded by cited evidence.

*`SemanticPreferenceSpec`:*

| Field | Position | Scaffolds | Purpose |
|---|---|---|---|
| `qualifier_inventory` | First, before `space_queries` | Every entry in `space_queries` | Evidence inventory — decompose the grouped description into individual qualifiers; per qualifier, cite the phrase and note which space(s) it implicates. Explicit empty-evidence path: a qualifier that doesn't clearly map to any space is flagged, not force-routed. Prevents the dominant preference failure — "blob handling" where the grouped description is treated as one undifferentiated concept and collapsed onto a single space (usually `anchor` or `viewer_experience`). Per-qualifier decomposition with cited phrases is the only scaffolding that forces separation up front. |
| Per-entry `carries_qualifiers` | First field inside each `PreferenceSpaceEntry`, before `space` / `weight` / `content` | `space`, `weight`, and `content` for that entry | Brief label form naming which qualifiers from `qualifier_inventory` land in this space (e.g., `"carries: dark, slow-burn"`, `"carries: date night, cozy"`). One label primes three downstream decisions at once: the space commit (why this entry exists), the weight enum (count and centrality of carried qualifiers informs `central` vs `supporting`), and the body content (which concepts the structured labels must express). Brief-label form is mandatory — sentence-form per-entry rationales become consistency templates for the body that follows. |

**Explicitly excluded:** no top-level `space_plan` / holistic-plan field.
`qualifier_inventory` already names qualifier→space mappings, and per-entry
`carries_qualifiers` handles weight priming per space. A sentence-form plan
would consistency-couple subsequent entries to the plan ("write the body
that matches the plan I just committed to"), which is the exact failure
mode the "brief pre-generation fields, no consistency coupling" convention
exists to prevent. Weight calibration is inherently per-entry and better
served by local scaffolding than a cross-entry narrative.

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

**Candidate generation (dealbreakers):** Returns every movie ID present in
the trending hash as a candidate set (typically the full TMDB top-500,
however many the refresh job wrote). The tail of the hash naturally sits at
score 0.0 from the concave-decay formula — those movies still enter the
candidate pool but contribute 0.0 to `dealbreaker_sum`, so they rank below
every other dealbreaker-admitted movie. No non-zero filter is applied.

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

For finer Step-4-specific implementation detail, clarified edge cases, and the
current outstanding questions for this stage, see
`search_improvement_planning/step_4_planning.md`. Treat that file as the
detailed companion for this step when this section stays high-level.

### Phase 4a: Candidate Generation Assembly

Collect all candidate sets produced by inclusion dealbreaker execution in
step 3.5. Deterministic endpoints (entity, metadata, awards,
franchise_structure, keyword, trending) each return a set of movie IDs.
Semantic dealbreakers do NOT generate candidates — deterministic endpoints
are the sole source of candidate IDs.

Union and deduplicate across all inclusion dealbreaker candidate sets to
produce the full candidate pool.

**Semantic-only inclusion checkpoint (dealbreaker scenario D2):** If no
deterministic inclusion dealbreaker exists **and at least one semantic
inclusion dealbreaker exists**, reroute to the pure-vibe dealbreaker flow
(see below). Each semantic inclusion dealbreaker independently generates
top-N per (dealbreaker, space) against the full corpus; union = candidate
pool. This covers:

- All inclusion dealbreakers route to `semantic` ("cozy date night movie")
- Only semantic inclusions plus deterministic exclusions ("good date night
  movies not with adam sandler" — date night is semantic, adam sandler is
  an entity exclusion)

**Zero-inclusion-dealbreaker, preferences-exist checkpoint (preference
scenario P2):** If no inclusion dealbreaker exists at all **but at least
one preference exists**, preferences take on candidate generation. See
"Zero-Dealbreakers, Preference-Driven Retrieval" below and
`step_4_planning.md` for the finalized Step-4 treatment. Candidates are
scored as preferences (not dealbreakers) and contribute to
`preference_contribution`.

**Zero-inclusion, zero-preference checkpoint (exclusion-only):** If no
inclusion dealbreaker AND no preferences exist (for example
"movies not starring Tom Cruise"), handle via the browse-style fallback —
top-K from `movie_card` ordered by the effective prior-based browse score,
then apply exclusions. See "Exclusion-Only Edge Case" below and
`step_4_planning.md`.

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
| **Semantic** | Raw weighted-sum cosine across LLM-selected spaces. No pool normalization, no diminishing-returns curve — preserves the "nobody matches well" signal. `is_primary_preference` affects the Phase 4c weight (3.0 vs 1.0), not the score shape. |
| **Trending** | Pass-through of precomputed trending score [0, 1]. |

**Scoring modes applied to raw scores:**

- **Raw semantic** — For all semantic preferences (primary and regular
  alike). Weighted-sum cosine across LLM-selected spaces. No pool
  normalization, no diminishing-returns curve. Primary vs. regular is
  expressed purely through the Phase 4c weight (3.0 vs. 1.0), not through
  score shape.
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

When step 2 output contains one or more inclusion dealbreakers and all of them
route to `semantic`, the query enters a separate codepath where vector search
is the candidate generator. This is **scenario D2** in the Endpoint 6
Execution Scenarios table. This includes the case where there are multiple
inclusion dealbreakers and every one is semantic. Deterministic exclusions may
still coexist. The detection is an explicit flow control checkpoint
immediately after step 2 completes.

A closely related but distinct variant — **scenario P2**, where zero
inclusion dealbreakers exist but preferences provide the only positive
signal — also uses preference-driven candidate generation, but the generated
candidates are scored as preferences (not dealbreakers). The semantic
preference version is documented in "Zero-Dealbreakers, Preference-Driven
Retrieval" below; see `step_4_planning.md` for the broader finalized Step-4
rule covering zero-inclusion preference flows more generally. The difference
is which step-2 items drive the search and how the resulting scores
contribute to `final_score`.

### How It Works

1. **Single LLM call** consumes all semantic dealbreakers AND preferences at
   once. Output shape mirrors the standard-flow per-item shape but batched:
   per-dealbreaker → 1 selected space (from the 7 non-anchor spaces) + one
   query tailored to that space's shape; per-preference → 1+ selected spaces
   (anchor allowed) each with a categorical weight (`central` / `supporting`)
   and one query per selected space. This replaces the per-item LLM calls
   used in the standard flow — the reduction in LLM wall-clock time is the
   reason pure-vibe is typically faster than a standard flow with many items,
   not slower. Per-space queries absorb all concepts routed to that space
   (one query per space, not per concept) so embedding-level combination is
   intentional and shaped by the LLM, not blended via embedding averaging
   — see "Why Per-Space, Not Per-Concept" below.
2. **In D2, candidate generation is dealbreaker-only.** Because step 2
   emitted at least one semantic inclusion dealbreaker, that dealbreaker
   is the source of positive intent. Each inclusion dealbreaker runs
   against the full corpus (no filters), top-N per (dealbreaker, space).
   Union across all dealbreaker searches = candidate pool. Preferences
   do not participate in candidate generation in D2 — letting them would
   silently override step 2's dealbreaker-vs-preference classification.
   (When step 2 emits no inclusion dealbreakers at all, the query routes
   to P2, where the semantic preference drives candidate generation
   instead. See "Zero-Dealbreakers, Preference-Driven Retrieval" below.)
3. **Anchor remains out of pure-vibe candidate generation.** Pure-vibe
   dealbreakers still draw only from the 7 non-anchor spaces. Preferences
   may select anchor for scoring, but anchor does not generate candidates.
4. **Scoring after candidate assembly:**
   - Dealbreakers: elbow-calibrated scoring (global calibration probe →
     threshold-plus-flatten) against each candidate, contributing to
     `dealbreaker_sum` (same formula as the standard flow).
   - Preferences: raw weighted-sum cosine across selected spaces with
     `central`=2 / `supporting`=1 weights, contributing to
     `preference_contribution` (same formula, same P_CAP).
5. Deterministic exclusions hard-filter the candidate pool; semantic
   exclusions use the match-then-penalize path (same as Phase 4b).

### Zero-Dealbreakers, Preference-Driven Retrieval (Scenario P2)

When step 2 emits zero inclusion dealbreakers but at least one preference
exists, preferences take on candidate generation.
This is scenario **P2** in the Execution Scenarios table.

**How it works:**

1. Step 2 produces the grouped semantic preference with its selected spaces
   and per-space queries (no per-concept decomposition — one query per
   selected space).
2. **Each selected space runs top-N against the full corpus** (no filters
   beyond any active UI filters; if UI filters are active, run a
   preliminary unfiltered probe to collect similarity distributions, then
   rerun with filters applied to produce the final candidate IDs). Union
   across spaces = candidate pool.
3. **Scoring:** raw weighted-sum cosine across the selected spaces,
   normalized by Σw (same as P1). Per-space weights: `central` = 2,
   `supporting` = 1.
4. **Final ranking role:** contributes to `preference_contribution`,
   scaled by P_CAP. `dealbreaker_sum` is zero because no inclusion
   dealbreakers were emitted.
5. Deterministic exclusions hard-filter the candidate pool; semantic
   exclusions use the match-then-penalize path (same as Phase 4b).

**Why preferences keep preference semantics here (not dealbreaker
semantics):** Step 2 classified these items as preferences deliberately.
The candidate-generation mechanism is orthogonal to the final-ranking
role — what shifts in P2 is who produces candidate IDs, not what the
items mean. Elevating them to dealbreakers here would silently override
step 2's classification and make the scoring inconsistent with P1 (where
the same preferences already run as raw weighted-sum cosine on a
pre-built candidate pool).

**Caveat:** Final scores in P2 are bounded above by P_CAP since
`dealbreaker_sum = 0`. This is fine — within-query ranking is what
matters, and all candidates share the same ceiling. Cross-query score
comparability is not a goal of the system.

### Exclusion-Only Edge Case

A query can emit zero inclusion dealbreakers AND zero preferences while
still being well-formed: "movies not starring Tom Cruise," "movies not
about clowns." Every dealbreaker in the output is an exclusion, and step
2 did not emit a preference either. The standard flow has no inclusion
signal to generate candidates from, and the P2 preference-driven path
doesn't apply because there's no preference to drive it.

**Rule:** Handle this as a browse-style fallback, not as pure-vibe
retrieval. (If a preference exists alongside exclusions — e.g., "not
clowns, something cozy" — use scenario P2 instead; the semantic
preference drives candidate generation and exclusions apply as
penalties/filters in Phase 4b.)

Generate candidates as the top-K movies from `movie_card`, ordered by the
effective browse score implied by `quality_prior` and `notability_prior`.
Apply exclusions normally in Phase 4b. `dealbreaker_sum` remains zero for all
seeded candidates; the browse ordering is only a seed heuristic, not
dealbreaker credit. Rank the surviving set by the resulting prior
contribution minus any exclusion penalties. See `step_4_planning.md` for the
Step-4-specific details and implementation considerations.

**Why this shape:** This mirrors what a librarian would do when asked
"a movie, but not that one" — return a prior-driven browse set minus the
excluded movies. The browse seed is only a candidate-generation heuristic
when the user gave no positive signal at all.
when the user expressed no positive intent at all. No new flow control
path is introduced beyond the fallback rule.

### Why Per-Space, Not Per-Concept

Concepts that route to *different* spaces always stay in separate searches —
running "fun" against `viewer_experience` and "heist" against
`plot_analysis` as independent retrievals preserves the independence of each
signal.

Concepts that route to the *same* space are absorbed into a single
LLM-authored query for that space rather than N per-concept queries unioned
together. The LLM composes the combined query in that space's native
vocabulary: "scary" + "funny" in `viewer_experience` becomes something like
`emotional_palette: darkly funny, gallows humor` plus
`tension_adrenaline: unsettling, creeping dread` — the LLM explicitly
captures the conjunction rather than hoping two independent retrievals
happen to intersect. This matches the structured-label embedding format
and lets the LLM shape the query's semantic combination instead of the
retrieval averaging out two unrelated embedded vectors.

The earlier (concept, space, subquery, role) tuple format is retired:
one query per selected space, absorbing all concepts, period.

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

- **Threshold + flatten** — For semantic dealbreaker scoring and semantic
  exclusion scoring. Similarity above the elbow → 1.0, below → linear decay
  to 0.0 at the floor. Below floor → 0.0. Used for determining if a movie
  genuinely has a semantic trait.
- **Raw weighted-sum cosine** — For all semantic preferences (primary and
  regular alike). `Σ(w × cosine) / Σ(w)` across selected spaces with
  `central`=2 / `supporting`=1 weights. No pool normalization, no
  diminishing-returns curve — preserves the "nobody matches well" signal.
  `is_primary_preference` affects only the Phase 4c weight (3.0 vs 1.0),
  not the score shape.
- **Pass-through** — For deterministic endpoint scores (binary or gradient).
  Used as-is since the gradient logic already lives in the execution layer.
- **Sort-by** — For explicit sort preferences ("chronological", "most recent
  first"). Expressed as a primary preference where the structured value
  (release date, reception score) is the dominant ranking signal.

---

## Semantic Endpoint — Finalized Implementation Decisions

Resolved during step 3 semantic endpoint planning. These are load-bearing
for the endpoint implementation and should not be re-litigated without
evaluation evidence.

- **Query model.** Every vector search (dealbreaker or preference, standard
  or pure-vibe) uses a single LLM-generated query per selected space,
  tailored to that space's structured-label shape. The original user query
  is never used directly for retrieval — the V1 "80/20 subquery/original
  blend" is retired. The anchor space is no exception: when anchor is used
  (preferences or pure-vibe candidate generation), it gets its own generated
  query like every other space.
- **Space availability.** Dealbreakers and semantic exclusions draw from
  the 7 non-anchor spaces only. Preferences may use all 8 (anchor allowed
  for scoring), but pure-vibe candidate generation remains non-anchor-only
  because it is driven by semantic dealbreakers.
- **Space selection granularity.** Two-level categorical weights for
  preferences: `central` → 2, `supporting` → 1. No `minor` option — if a
  space's signal isn't at least supporting meaningfully, don't select it.
  This prevents weight dilution across marginal spaces. Dealbreakers and
  exclusions select exactly one space.
- **Per-space query scope.** One query per selected space, absorbing all
  concepts routed to that space, phrased in that space's native vocabulary
  (e.g., "scary but funny" in `viewer_experience` becomes
  `emotional_palette: darkly funny, gallows humor` +
  `tension_adrenaline: unsettling, creeping dread`). The per-concept
  `(concept, space, subquery, role)` tuple format is retired.
- **Space taxonomy source of truth.** Each `create_*_vector_text` function
  in `movie_ingestion/final_ingestion/vector_text.py` carries a module-level
  docstring with four sections — Purpose, What's Embedded (the structured
  labels actually present), Boundary (what the space is *not* for), 2–3
  canonical example queries. Step 3 prompt imports these verbatim at build
  time. Convention, not code-gen. Re-audit when vector text generation
  changes.
- **Four execution scenarios.** Dealbreaker-side logic composes with
  preference-side logic based on step 2's output. Summary:
  - **D1 (score-only):** ≥1 non-semantic inclusion dealbreaker exists →
    semantic dealbreakers score the pre-built candidate pool via global
    elbow calibration + threshold-plus-flatten applied to candidate
    cosines. Contributes to `dealbreaker_sum`.
  - **D2 (candidate-generating):** no non-semantic inclusion dealbreaker
    but ≥1 semantic inclusion dealbreaker → each semantic dealbreaker
    generates top-N per (dealbreaker, space); union = pool. Same
    elbow-calibrated scoring. The top-N probe doubles as both
    calibration sample and candidate pool — one Qdrant call, not two.
    Dealbreakers do **not** cross-score across each other's candidate
    sets; each scores only the movies its own probe retrieved.
    Contributes to `dealbreaker_sum`.
  - **P1 (score-only):** ≥1 inclusion dealbreaker exists (semantic or
    not) → semantic preferences score the pre-built pool via raw
    weighted-sum cosine (`central`=2 / `supporting`=1), normalized by
    Σw. Contributes to `preference_contribution`.
  - **P2 (candidate-generating):** zero inclusion dealbreakers, ≥1
    semantic preference → semantic preference generates candidates
    (top-N per selected space, union). Same raw weighted-sum cosine
    scoring as P1. Contributes to `preference_contribution`.
  Global elbow calibration is used in D1 and D2 so scoring is invariant
  to whether the dealbreaker also generated candidates. P2 keeps
  preference semantics (not dealbreaker semantics) because step 2's
  dealbreaker-vs-preference classification is binding regardless of who
  produces candidate IDs.
- **Exclusion-only edge case.** When step 2 emits zero inclusion
  dealbreakers AND zero preferences (only exclusion dealbreakers),
  do not enter pure-vibe retrieval or P2. Generate candidates as top-K
  by the default quality composite (`0.6 × reception_score + 0.4 ×
  popularity_score`), then apply exclusions normally in Phase 4b.
- **New Pydantic models for multi-source vector spaces.** Any vector space
  whose embedded text is assembled from more than one metadata object
  (anchor, plot_events, production, reception at minimum) gets a dedicated
  Pydantic model that defines the embedded shape. These models become the
  templating surface for both ingest-side embedding text generation and
  search-side per-space query generation, keeping the two in lockstep.
- **Transient failure handling.** Retry once on transient errors; on
  persistent failure, return an empty result set so the orchestrator
  handles it the same as a no-match.
- **Max-across-spaces combining is retired.** The old plan to threshold
  each relevant space separately and take the max is superseded by the
  single-space-per-dealbreaker rule plus the weighted-sum preference
  formula.
- **Elbow/floor detection for dealbreaker and semantic-exclusion scoring.**
  Applies to semantic dealbreaker scoring (D1 + D2) and semantic exclusion
  scoring. Does NOT apply to preference scoring, which uses raw weighted-sum
  cosine with no elbow calibration. Implemented in
  `search_v2/stage_3/semantic_query_execution.py::_detect_elbow_floor`.
  No result caching — every invocation pays for one unfiltered top-N probe
  (a future Redis cache keyed by `(query_text, space, embedding_model,
  space_prompt_version, corpus_version)` is a known hook-point but deferred).

  Procedure:
  1. **Collect distribution.** Pull top-N cosine similarities from the full
     corpus for the LLM-generated query against the selected space, sorted
     descending. N = 2000 as a working default. In D2 this probe doubles
     as the candidate pool — the same call serves both roles.
  2. **Short-probe guard.** If fewer than 20 similarities returned, skip
     Kneedle and use the pathology fallback.
  3. **Pathology check — "is there any elbow at all?"** If the probe's
     top-to-bottom range (`max_sim − min_sim`) falls below 0.05, treat the
     distribution as flat: no discriminable structure exists. Fall back to
     `elbow = max_sim × 0.85`, `floor = max_sim × 0.65`, log at INFO for
     calibration audit, and skip the remaining steps. (The earlier
     formulation of this check as "max |diff| of the smoothed curve < 0.05"
     fired spuriously on ordinary distributions because per-step diffs of
     realistic knees sit around 0.003 — range is the operationally
     meaningful quantity.)
  4. **Smooth.** EWMA over the sorted array, span = `max(5, N // 100)`.
     Smoothing is used only to stabilize Kneedle's knee-rank detection —
     the returned `elbow_sim` and `floor_sim` are the **raw** similarities
     at the detected ranks, not smoothed values. (On a descending sequence
     EWMA lags — `smoothed[i] > raw[i]` — so reporting smoothed y-values
     would inflate the threshold and silently shrink the pass zone when
     the orchestrator compares raw Qdrant scores against it.)
  5. **Detect elbows with Kneedle.** Parameters: `curve='convex',
     direction='decreasing', S=1, online=True`, then canonicalize
     `sorted(set(locator.all_knees))` to handle version-to-version shape
     changes (set vs. list). If zero knees detected → pathology fallback.
  6. **Elbow selection — first knee, with early-rank safeguard.** Start
     with the first detected knee (smallest rank). If its rank < 10 AND at
     least one additional knee exists, skip forward to the next knee
     (guards against a handful of outliers pinching the 1.0 boundary too
     tightly). If the first knee is the only knee, use it as-is — don't
     invent a later elbow that doesn't exist in the data. Never pick the
     "largest bulge" knee; always prefer the earliest qualifying knee.
     `elbow_sim = similarities[elbow_rank]`.
  7. **Floor selection.** If two or more knees were detected AND we used
     the first knee as elbow, use the second knee as floor (natural
     bimodal signal — e.g., Christmas distributions). If we skipped to the
     second knee as elbow and a third exists, use the third as floor.
     Otherwise compute `floor_sim = max(elbow_sim − 2 × (max_sim −
     elbow_sim), 0.0)` — a gap-proportional floor that widens the decay
     zone for sharp elbows and narrows it for compressed distributions.
     Rank-based floors use `similarities[floor_rank]`.
  8. **Clamp invariant.** `floor_sim = max(0, min(floor_sim, elbow_sim −
     1e-9))`, `elbow_sim ∈ [0, 1]`. Guarantees `0 ≤ floor < elbow ≤ 1` for
     the downstream transform.
  9. **Scoring transform** applied to each candidate's cosine similarity:
     `score = 1.0` if `sim ≥ elbow_sim`; `score = (sim − floor) /
     (elbow_sim − floor)` if `floor < sim < elbow_sim`; `score = 0.0` if
     `sim ≤ floor`. Linear decay is the default; non-linear exponents
     remain deferred pending eval evidence.

  **Why first-knee instead of largest-bulge:** the first knee marks the
  earliest transition from "clearly the concept" to "less clearly the
  concept," which is exactly the semantics the 1.0 boundary needs. The
  largest bulge may sit further down the tail (e.g., the second knee in a
  bimodal distribution) and would over-admit borderline matches to full
  credit. The rank-10 safeguard exists solely to skip past outlier-driven
  early knees when a more substantive elbow is available.

  **Floor ratio 0.65, not 0.50.** The proposal originally called for
  `floor = max_sim × 0.50`; raised to 0.65 during implementation so the
  pathology fallback produces a narrower decay zone. Real matches
  typically sit above 0.5 of max; decaying all the way to that point
  awards too much score to distant candidates.

---

## Decisions Deferred to Implementation

- Exact step 2 prompt engineering (few-shot examples, chain-of-thought format)
- ~~Exact elbow and floor detection algorithm for semantic dealbreaker
  scoring and exclusion thresholds.~~ **RESOLVED AND IMPLEMENTED** — see
  "Elbow/floor detection for dealbreaker and semantic-exclusion scoring"
  above and `search_v2/stage_3/semantic_query_execution.py::_detect_elbow_floor`.
  Remaining tuning deferred to eval: non-linear decay exponent (γ), top-N
  corpus probe size (working default 2000), EWMA span constants, rank-10
  safeguard threshold, pathology-check range cutoff (0.05), and floor
  ratio (0.65).
- ~~Semantic endpoint elbow/floor cache.~~ **DEFERRED** — no cache in the
  current implementation; every dealbreaker invocation pays for one
  unfiltered top-N probe. Hook-point is the call to `_detect_elbow_floor`
  in `semantic_query_execution.py`. If latency becomes a concern, a
  Redis cache keyed by `(query_text, space, embedding_model_version,
  space_prompt_version, corpus_version)` drops in without restructuring.
- **Evaluation test — zero-dealbreaker browse fallback quality.** The
  current design routes zero-inclusion-dealbreaker queries to a browse-style
  top-K default-quality pool, then applies preference scoring on top. Open
  question: does this produce strong enough results for preference-only
  queries, or is a more semantic-first fallback needed? Measure top-K result
  quality on zero-dealbreaker queries before revisiting.
- **Evaluation test — cross-space cosine comparability in preference
  scoring.** `pref_score = Σ(w_space × cosine_space) / Σ(w_space)` assumes
  raw cosines are comparable across spaces. Different spaces may have
  different natural score bands (e.g., `production_vectors` vs
  `plot_analysis_vectors` may run at different cosine distributions),
  which would systematically over-weight hotter spaces. If evaluation
  shows skew, add a global per-space calibration layer (percentile mapping
  or floor/ceiling normalization against the corpus, computed offline)
  before the weighted sum. Within-pool normalization is still rejected —
  this is a per-space global transform, not a per-query one.
- **Semantic exclusion prompt tightness — deferred.** Originally flagged
  because exclusion false positives are more expensive than inclusion
  false positives. Deprioritized now that exclusions are match-then-penalize
  rather than hard-filter — a false positive costs score weight, not
  removal. Revisit if evaluation shows systemically unfair exclusion
  penalties.
- Step 3 prompt design per endpoint
- Step 3 output schemas per endpoint (metadata endpoint complete:
  `schemas/metadata_translation.py`; entity endpoint complete:
  `schemas/entity_translation.py`; franchise endpoint complete:
  `schemas/franchise_translation.py`; keyword endpoint classification
  registry complete: `schemas/unified_classification.py` — final
  per-call spec wrapper still pending; remaining endpoints pending)
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
- Result pagination for long lists: return top 25 initially, cache the full
  candidate/display list in Redis, and allow fetching additional pages via
  pointer IDs or equivalent
- Multi-interpretation trigger criteria
