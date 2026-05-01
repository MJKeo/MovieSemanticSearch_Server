# Query Buckets

This document groups the categories in `query_categories.md` by
query-generation shape. The goal is to reduce handler design from
many category-specific prompt patterns to a smaller set of reusable
instruction buckets.

Categories still define the conceptual boundary and the endpoint
surface. Buckets define how the query-generation layer should reason
about parameters once a category call exists.

## Principles

- Bucket by handler instruction shape, not by user-facing concept.
- Let endpoint schemas carry endpoint-specific parameter details.
- Keep category-specific nuance in category prompt notes, not in the
  shared bucket instruction.
- Prefer deterministic representations when they genuinely cover the
  expression, but do not force them when they leave meaningful gaps.
- Treat semantic vector spaces as an internal semantic-retrieval
  detail. A category can query one or more semantic spaces while still
  belonging to the same bucket if the handler instruction is the same.

## Bucket 1: No LLM / Pure Code

### What It Represents

Categories whose query generation does not require an LLM decision.
The runtime can execute them directly or parse them deterministically.

### Why Split From Other Buckets

These categories should not pay LLM latency or cost. Their behavior is
more reliable as code than as generated parameters.

### Query-Generation Handling

- Trending executes directly with no generated parameters.
- Media type is parsed from explicit expressions with deterministic
  string/regex logic.

### Categories

- Trending
- Media type

## Bucket 2: Explicit No-Op

### What It Represents

Categories that intentionally do not execute a query today.

### Why Split From Other Buckets

This is different from pure-code handling. Trending and media type are
real runtime behaviors that return results. A no-op bucket is an
explicit declaration that the category is reserved, unsupported, and
should return nothing until backing data or a real endpoint exists.

### Query-Generation Handling

- Do not build an LLM prompt.
- Do not execute an endpoint query.
- Return an empty result for the category.
- Keep the category in the taxonomy so upstream routing can stay honest
  instead of misrouting unsupported asks into approximate categories.

### Categories

- Below-the-line creator lookup

## Bucket 3: Single Non-Metadata Endpoint

### What It Represents

Categories where one non-metadata retrieval endpoint owns the category
call. This includes lexical, studio, franchise, award, keyword, and
semantic retrieval surfaces. For semantic categories, the semantic
endpoint may internally choose or query multiple vector spaces, but the
shared handler instruction does not need to reason across endpoint
families.

### Why Split From Other Buckets

There is no endpoint-selection problem at the bucket level. The generic
instruction can simply ask whether the category call should run and, if
so, fill the endpoint schema from the expression and retrieval intent.
Endpoint-specific schemas and category notes carry the detailed
interpretation.

### Query-Generation Handling

- Use the category call's expression and retrieval intent as the source
  of truth.
- Do not add routing instructions beyond the category and endpoint
  schema.
- Allow the handler to abstain if the category call does not genuinely
  fit the endpoint.
- For semantic retrieval, phrase the query to match the relevant
  ingestion text style, with category notes providing the template.

### Categories

- Person credit
- Title text lookup
- Named character lookup
- Studio / brand attribution
- Franchise / universe lineage
- Adaptation source flag
- Award records
- Filming location
- Plot events
- Narrative setting
- Viewing occasion
- Visual craft acclaim
- Music / score acclaim
- Dialogue craft acclaim
- Named source creator

## Bucket 4: Single Metadata Endpoint

### What It Represents

Categories that are answered by metadata predicates, ranges, thresholds,
priors, or sort/select operations.

### Why Split From Other Buckets

Metadata query generation is structurally different from text retrieval.
It usually emits typed values, range bounds, decay behavior, hard gates,
or numeric priors rather than search prose.

### Query-Generation Handling

- Generate typed metadata parameters from the expression and retrieval
  intent.
- Preserve the distinction between hard filters, soft preferences,
  additive priors, and ordinal selection.
- Use category notes for attribute-specific behavior such as vague-era
  ranges, runtime falloff, financial bucket choice, or reception prior
  treatment.
- Media type is excluded from this bucket even though it searches
  metadata, because it is deterministic enough to live in pure code.

### Categories

- Release date / era
- Runtime
- Maturity rating
- Audio language
- Streaming platform
- Financial scale
- Numeric reception score
- Country of origin
- General appeal / quality baseline
- Chronological ordinal

## Bucket 5: Preferred Representation With Fallback / Gap-Fill

### What It Represents

Categories where one representation is preferred when it cleanly covers
the request, but another representation should be used when the
preferred one is missing, approximate, or leaves meaningful parts of the
expression uncovered.

The preferred representation is often a canonical keyword tag, but this
bucket is intentionally broader than "keyword first." Some categories
prefer a deterministic proxy and fall back to another deterministic
surface; others use deterministic tags for exact matches and semantic
retrieval for long-tail or qualified expressions.

### Why Split From Other Buckets

These categories need a coverage judgment. Running every possible
endpoint can mix different interpretations or add noisy duplicate
signal, while always using only the preferred endpoint can miss
qualified, spectrum, or long-tail expressions.

### Query-Generation Handling

- First determine whether the preferred representation genuinely covers
  the expression.
- If it fully covers the expression, generate only that representation.
- If it partially covers the expression, generate the preferred
  representation for the covered part and fallback/gap-fill parameters
  for the remaining part.
- If it does not cover the expression, use the fallback representation.
- Do not pad with extra endpoints just because they are plausible.
  Generate the minimum set that captures the category call.
- Category-specific notes define what "preferred" means for each
  category.

### Categories

- Central topic / about-ness
- Element / motif presence
- Character archetype
- Genre
- Cultural tradition / national cinema
- Format + visual-format specifics
- Narrative devices + structural form + how-told craft
- Seasonal / holiday
- Story / thematic archetype
- Specific praise / criticism

## Bucket 6: Semantic-Always With Deterministic Augmentation

### What It Represents

Categories whose meaning is inherently semantic, so the semantic
endpoint always runs. In addition, any deterministic surface (canonical
keyword tags, numeric priors, popularity priors) that can catch signals
the semantic channel handles poorly should run alongside — not instead
of, but in parallel with, the semantic call.

### Why Split From Other Buckets

This is not a coverage-judgment bucket like Bucket 5. The semantic call
is not optional and deterministic calls are not fallbacks. Vector search
is unreliable for binary or canonical attributes (e.g. "feel-good"
expresses well as semantic prose, but a `HAPPY_ENDING` tag adds a
crisper signal that semantic blurs across many endings). Running both
is strictly stronger than running either alone, so redundancy across
channels is the explicit design.

### Query-Generation Handling

- Always generate the semantic query that captures the expression's
  core meaning.
- For each available deterministic surface, ask whether running it
  alongside the semantic query would catch a binary or canonical signal
  semantic retrieval tends to flatten. If yes, generate it.
- Skip a deterministic surface only when no clean signal is implied —
  not because semantic "already covers it." Overlap with the semantic
  read is expected and welcome.
- Do not let deterministic augmentation reshape or shrink the semantic
  query. Semantic remains the authoritative read; deterministic calls
  reinforce it.

### Categories

- Emotional / experiential
- Cultural status / canonical stature

## Bucket 7: Character-Franchise Fan-Out

### What It Represents

The unique case where a single named referent is inherently both a
character lookup and a franchise/universe lookup.

### Why Split From Other Buckets

This is not fallback, gap-fill, or additive interpretation. Both
retrieval paths are required because both are true of the referent.
Splitting the input into separate character and franchise traits would
violate the one-trait, one-category rule for dual-nature referents.

### Query-Generation Handling

- Resolve the referent once.
- Generate character-presence parameters.
- Generate franchise/universe lineage parameters.
- Run both paths as the category's fixed fan-out.
- Keep all source-medium or adaptation flags as separate category calls.

### Categories

- Character-franchise

## Bucket 8: Audience-Suitability Redundant Combo

### What It Represents

Audience-suitability categories whose intent is multi-faceted enough
that several endpoints can each carry a complementary slice of the same
suitability concept. Every endpoint with a real signal to add should
fire — overlap across deterministic gates, inclusion scoring, exclusion
scoring, and semantic intensity is welcome, not deduplicated.

One category focuses on appeal to a target audience. The other focuses
on sensitivity or content the user may want to avoid. They share the
same handler shape.

### Why Split From Other Buckets

A single suitability concept ("suitable for kids", "no gore") is
genuinely better served by multiple endpoints firing in parallel than
by any one of them alone. A maturity-rating ceiling, a wholesome-tone
semantic query, and an exclusion-keyword scoring signal all measure
"suitable for kids" from different angles, and combining them is more
robust than picking the best one. Bucket 8 makes that redundancy the
default, gated only by whether each endpoint has a real signal to
contribute.

### Query-Generation Handling

- Begin with a high-level scoping pass: read the expression and
  enumerate every angle the suitability concept exposes (hard
  ceilings, content categories, tone, watch-context, etc.).
- For each available endpoint, decide whether it carries a real
  complementary signal toward this requirement. If yes, fire it.
  Overlap with another endpoint is not a reason to skip.
- Skip an endpoint only when it has nothing distinct to contribute —
  not because another endpoint already partially covers the angle.
- Preserve polarity: endpoint parameters represent presence of an
  attribute; the wrapper's polarity field decides whether that
  presence helps or hurts.

### Categories

- Target audience
- Sensitive content

## Coverage Check

These buckets cover every active category in `query_categories.md`:
categories 1-42 and 44. Categories 43 and 45 were removed in v3.1
because parametric expansion now happens before ordinary category
query generation.
