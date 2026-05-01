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
The runtime can execute them directly, parse them deterministically,
or return a fixed empty result.

### Why Split From Other Buckets

These categories should not pay LLM latency or cost. Their behavior is
more reliable as code than as generated parameters.

### Query-Generation Handling

- Trending executes directly with no generated parameters.
- Media type is parsed from explicit expressions with deterministic
  string/regex logic.
- Below-the-line creator lookup is a reserved no-op until backing data
  exists.

### Categories

- Trending
- Media type
- Below-the-line creator lookup

## Bucket 2: Single Non-Metadata Endpoint

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

## Bucket 3: Single Metadata Endpoint

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

## Bucket 4: Preferred Representation With Fallback / Gap-Fill

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

## Bucket 5: Semantic-Preferred With Deterministic Support

### What It Represents

Categories where the primary meaning is best represented by semantic
prose, but deterministic tags or metadata priors can add useful support
when they genuinely apply.

### Why Split From Other Buckets

These should not start from deterministic tags. Their meaning is broad,
evaluative, or experiential enough that forcing a tag-first read would
flatten the request. Deterministic signals are supporting evidence, not
the main interpretation.

### Query-Generation Handling

- Start by generating the semantic query that captures the expression's
  core meaning.
- Add deterministic support only when the expression cleanly implies a
  tag, numeric prior, popularity prior, or other structured signal.
- Do not let deterministic support override the semantic read.
- Use the minimum supporting set; absence of a clean deterministic match
  is not a failure.

### Categories

- Emotional / experiential
- Cultural status / canonical stature

## Bucket 6: Character-Franchise Fan-Out

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

## Bucket 7: Audience-Suitability Deterministic-First Combo

### What It Represents

Audience-suitability categories where deterministic filters should be
used as much as possible, then softer inclusion or preference scoring
fills in the remaining intent.

One category focuses on appeal to a target audience. The other focuses
on sensitivity or content the user may want to avoid. They share the
same handler shape.

### Why Split From Other Buckets

These categories often produce both gate-like constraints and scoring
signals. They need explicit handling for deterministic filters,
inclusion candidates, and exclusion candidates rather than ordinary
single-endpoint or fallback behavior.

### Query-Generation Handling

- Use deterministic gates whenever the expression implies a clear
  maturity ceiling, content exclusion, or suitability boundary.
- Generate inclusion scoring for positive audience fit or desired
  suitability signals.
- Generate exclusion scoring or filters for content the user wants to
  avoid.
- Use semantic intensity or watch-context signals when deterministic
  filters do not fully capture the request.
- Preserve polarity: endpoint parameters should represent presence of
  an attribute; merge logic handles whether that presence helps or
  hurts.

### Categories

- Target audience
- Sensitive content

## Coverage Check

These buckets cover every active category in `query_categories.md`:
categories 1-42 and 44. Categories 43 and 45 were removed in v3.1
because parametric expansion now happens before ordinary category
query generation.
