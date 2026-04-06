# /realign-vector-search-prompts

Realign a vector space's search prompts (subquery generation + weight
assessment) with the current metadata generation and embedding pipeline.

The metadata generation system has evolved, but the search prompts still
describe the old field names, content boundaries, and data shapes. This
command walks through understanding what's ACTUALLY embedded, what the
search prompts BELIEVE is embedded, and rewrites both prompts from scratch
to maximize semantic match quality.

Do not write code until Phase 3. Phases 1-2 are research and reporting.

## The vector space is: $ARGUMENTS

Valid values: plot_events, plot_analysis, viewer_experience, watch_context,
narrative_techniques, production, reception

---

## Phase 1: Build the Source of Truth (What's Actually Embedded)

Read all of the following to understand what content actually ends up in
this vector space. Do not present findings yet — this phase is intake.

### 1a. The vector text generation function

Read `movie_ingestion/final_ingestion/vector_text.py` and find the
`create_{vector_name}_vector_text()` function. This is the FINAL
authority on what gets embedded — it assembles the text that becomes
the vector. Understand:
- What data sources it pulls from (metadata types, Movie helper methods,
  IMDB data, TMDB data)
- What labels it applies (e.g., "genre signatures:", "conflict:")
- What formatting it uses (lowercased, normalized, joined how)
- What it explicitly EXCLUDES

### 1b. The embedding_text() methods

For each metadata type that feeds this vector, read its `embedding_text()`
method in `schemas/metadata.py`. This tells you:
- Which fields from the LLM output are included vs excluded
- What labels are applied to each field
- How terms are normalized and joined
- The relative size of each embedded segment

### 1c. The metadata generation prompts and schemas

Read the generation prompt and output schema for each metadata type that
feeds this vector:
- Prompts: `movie_ingestion/metadata_generation/prompts/{type}.py`
- Schemas: `schemas/metadata.py` — find the `{Type}Output` class

Understand the SEMANTICS of each field — what it captures conceptually,
what the LLM was instructed to produce, and what examples look like.
This tells you what kind of content actually populates the embedded fields.

### 1d. Movie helper methods (if applicable)

Some vectors pull data from Movie class methods rather than (or in addition
to) LLM metadata. If the vector text function calls methods on Movie, read
their implementations in `schemas/movie.py`. Key methods used across vectors:

| Method | Used by vectors |
|--------|----------------|
| `production_text()` | production |
| `languages_text()` | production |
| `release_decade_bucket()` | production, anchor |
| `budget_bucket_for_era()` | production, anchor |
| `is_animation()` | production |
| `maturity_text_short()` | anchor |
| `reception_tier()` | reception, anchor |
| `deduplicated_genres()` | anchor |
| `title_with_original()` | anchor |

### 1e. Cross-vector data flow

Some metadata types feed multiple vectors or have extraction-zone fields
that feed other generators but aren't themselves embedded. Understand:
- Which fields are embedded vs extraction-only (e.g., reception's
  thematic_observations feed plot_analysis but aren't in the reception vector)
- Whether IMDB/TMDB data gets merged in at vector_text.py time (e.g.,
  IMDB genres merged into plot_analysis genre_signatures)

### Metadata-to-vector mapping reference

| Vector | Metadata types used | Movie methods used |
|--------|--------------------|--------------------|
| plot_events | plot_events (or raw IMDB synopses/summaries/overview as fallback) | — |
| plot_analysis | plot_analysis + IMDB genres merged into genre_signatures | — |
| viewer_experience | viewer_experience | — |
| watch_context | watch_context | — |
| narrative_techniques | narrative_techniques | — |
| production | production_keywords + source_of_inspiration | production_text(), languages_text(), release_decade_bucket(), budget_bucket_for_era(), is_animation() |
| reception | reception (synthesis zone only) | reception_tier() |

---

## Phase 2: Identify All Misalignments

Now read the current search prompts and catalog every difference from
what you learned in Phase 1.

### 2a. Read the current search prompts

- Subquery prompt: `implementation/prompts/vector_subquery_prompts.py` —
  find the `{VECTOR_NAME}_SYSTEM_PROMPT` constant
- Weight prompt: `implementation/prompts/vector_weights_prompts.py` —
  find the `{VECTOR_NAME}_WEIGHT_PROMPT` constant

### 2b. Catalog misalignments

For each prompt, check every claim it makes against your Phase 1 findings.
Produce a structured misalignment report covering:

**Field-level misalignments:**
- Stale field names (renamed, removed, merged)
- Fields listed in prompts that don't exist in the embedded content
- Fields in the embedded content that the prompts don't mention
- Incorrect descriptions of what a field contains

**Content boundary misalignments:**
- Content types the prompt claims are in this vector but aren't
  (e.g., cast/crew names claimed in production but not embedded)
- Content types actually in this vector that the prompt doesn't mention
  (e.g., IMDB genres merged into plot_analysis)

**Semantic framing misalignments:**
- Examples in the prompt that would generate subquery text better suited
  for a DIFFERENT vector space (e.g., experiential terms generated for
  a thematic vector)
- Extraction guidance that targets the wrong content shape (e.g.,
  telling the LLM to generate prose when the vector contains short labels)

**Example misalignments:**
- Example outputs that contain terms from the wrong semantic domain
- Examples that would produce poor cosine similarity against actual
  embedded content

Rate each misalignment: HIGH (wrong content will be generated / wrong
weight assigned), MEDIUM (suboptimal but partially functional), LOW
(terminology difference, minor framing issue).

### 2c. Present the report

Present your findings as a structured report with:
1. Summary of what's actually in this vector (from Phase 1)
2. The full misalignment inventory (from Phase 2b)
3. Your assessment of which misalignments matter most for retrieval quality

**STOP HERE.** Wait for confirmation before proceeding to rewrite.

---

## Phase 3: Rewrite Both Prompts

Rewrite both prompts from scratch. Do NOT try to patch the existing text —
build from the ground up based on what you now know about the actual
embedded content.

### 3a. Design principles (apply to both prompts)

**Accuracy over convention:** The prompt must describe what's ACTUALLY in the
vector, not what a reasonable person might assume. Every content claim must
trace to a specific field or method you read in Phase 1.

**Format awareness:** The subquery prompt should understand the SHAPE of the
embedded content — prose segments vs labeled term lists, relative sizes of
each segment, what produces the largest retrieval surface for cosine similarity.

**Boundary clarity:** Explicitly define what IS and ISN'T in this vector with
paired examples that highlight the boundary. The most common failure mode is
generating subquery text that belongs in an adjacent vector space. Call out
the specific adjacent spaces and the boundary between them.

**Grounded examples:** Every example embedded content string and every example
output should be realistic for the actual data. Use your Phase 1 understanding
of field semantics, labels, and formatting.

### 3b. Rewrite the subquery prompt

The subquery prompt's job is to transform a user query into text that will
produce strong cosine similarity against this vector's actual embedded content.

Structure the prompt around:
1. What's in this vector space — field-by-field with names, labels, sizes,
   and realistic examples
2. A FULL EMBEDDED EXAMPLE showing the actual format (all fields assembled as
   they appear after vector_text.py processing)
3. Transformation approach — how to maximize semantic overlap with each
   content segment
4. What to extract from user queries and how to phrase it
5. What WON'T retrieve well (content that's in other vectors, not this one)
6. Critical boundary — the most common confusion with adjacent vectors,
   with paired examples
7. Negation handling
8. When to return null
9. Examples — rewrite all examples to generate terms that match the actual
   embedded content. Fix any examples that leaked terms from adjacent vectors.

### 3c. Rewrite the weight prompt

The weight prompt's job is to assess whether a user query is relevant to this
vector space.

Structure the prompt around:
1. What this vector contains — accurate field inventory with brief descriptions
2. The key question — what kind of search intent maps to this vector
3. High relevance signals — what user query patterns strongly indicate this vector
4. Low relevance signals — what patterns do NOT indicate this vector
5. Critical boundary — the most common confusion, with paired examples
6. Comparison query guidance (how "like [Movie]" queries relate to this vector)
7. Calibration guidance for each relevance level

### 3d. Verify and update references

After writing both prompts:
- Run `python -c "from implementation.prompts.vector_subquery_prompts import VECTOR_SUBQUERY_SYSTEM_PROMPTS; from implementation.prompts.vector_weights_prompts import VECTOR_WEIGHT_SYSTEM_PROMPTS; print('OK')"` to verify both modules import cleanly
- Check that the dict mappings at the bottom of each file still reference the
  correct prompt variable names

### 3e. Update DIFF_CONTEXT.md

Append a structured entry describing what changed, why, and the key
misalignments that were fixed.

### Reference: the plot_analysis rewrite

The plot_analysis prompts have already been rewritten using this process.
Read the current `PLOT_ANALYSIS_SYSTEM_PROMPT` and `PLOT_ANALYSIS_WEIGHT_PROMPT`
as examples of the target quality and structure. Key patterns to follow:
- Full embedded example showing actual format
- Explicit field inventory with names, labels, and sizes
- Critical boundary section with paired is/isn't examples
- Examples that generate terms from the correct semantic domain
- Transformation approach oriented around cosine similarity against real content
