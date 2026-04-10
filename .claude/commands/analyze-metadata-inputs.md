Analyze the eligibility criteria and input design for a metadata
generation type. The goal is to build the optimal input contract
from scratch — which data should this generator receive, at what
quality thresholds, and through what fallback ladders.

Do not write code. This is a discussion-first process. Present
findings, analysis, and proposals as discussion points and wait
for feedback before moving on.

## The metadata type is: $ARGUMENTS

## Step 1: Read orientation files

Read all of these before any analysis:

- `docs/PROJECT.md` — priorities and constraints
- `DIFF_CONTEXT.md` — recent changes that may affect this type
- `docs/conventions.md` — cross-codebase invariants
- `docs/conventions_draft.md` — patterns under review
- The module doc in `docs/modules/` for ingestion if one exists
- `docs/llm_metadata_generation_new_flow.md` — the full generation
  pipeline spec, especially the section for this metadata type

## Step 2: Understand what this metadata type produces

Read these files for the target metadata type:

1. `movie_ingestion/metadata_generation/schemas.py`
   — find the `{TYPE}Output` class, its fields, and its `__str__()`
   method (which determines what gets embedded)
2. `movie_ingestion/metadata_generation/prompts/{type}.py`
   — the system prompt: what is the LLM being asked to do?
3. `movie_ingestion/metadata_generation/generators/{type}.py`
   — the current input contract: what data does the generator
   accept and how does it build the user prompt?

Then answer these questions (present as findings, not prose):

- **What does this metadata type represent?** One sentence.
- **What vector space does it populate?** What kinds of user
  queries should it match?
- **What are the output fields?** Which are embedded vs
  intermediate?
- **What does the current prompt ask the LLM to do?** What
  evidence does it need to do that job well?

## Step 3: Map all available upstream data

Read:

1. `movie_ingestion/metadata_generation/inputs.py`
   — `MovieInputData` fields (the raw TMDB+IMDB data available)
2. `movie_ingestion/metadata_generation/pre_consolidation.py`
   — current eligibility check for this type, observation/keyword
   routing, maturity consolidation
3. `movie_ingestion/metadata_generation/generator_registry.py`
   — Wave 1 vs Wave 2 dependencies, what finalized outputs from
   other generators are available as inputs

Build a complete inventory of every data source this generator
could potentially receive:

**Raw data** (always available if the movie passed quality gates):
- List each MovieInputData field with a note on what it contains

**Wave 1 outputs** (available for Wave 2 types):
- List each field from each Wave 1 generator's output schema
- Note which fields are extracted (grounded in source data) vs
  synthesized (LLM-generated abstractions)

**Wave 2 outputs** (available for later Wave 2 types if ordered):
- List any cross-Wave-2 dependencies if applicable

For each data source, note:
- What signal does it carry for THIS metadata type's task?
- Is it direct evidence or indirect/supporting context?
- Does it overlap with another source (redundant signal)?

## Step 4: Analyze downstream usage

Read:

1. `movie_ingestion/final_ingestion/vector_text.py` — how this
   metadata type's `__str__()` output gets composed into the
   vector text that Stage 8 embeds via `generate_vector_embedding()`
2. `implementation/prompts/` — how the search pipeline uses this
   vector space (subquery generation prompts, vector space weight
   assignment)
3. The search-side schema in `schemas/metadata.py` if one exists
   for this type

Answer:
- **What queries is this vector space designed to match?**
- **What makes a good embedding for this space?** (Specific,
  search-query-like phrases? Dense thematic labels? Narrative
  summaries?)
- **Does the current output format align with how the embedding
  is consumed?** Flag any mismatches.

## Step 5: Design the optimal input contract from scratch

Now reason about the ideal input set as if designing from zero.
For each candidate input, evaluate:

1. **Signal relevance:** Does this input carry direct evidence
   for what the LLM needs to produce? Rate as primary evidence,
   supporting context, or irrelevant.
2. **Signal quality:** Is this input clean and reliable, or noisy
   and variable-length? Does it need filtering or thresholding?
3. **Redundancy:** Does another input already carry this signal
   better? If two inputs overlap, which should win?
4. **Small-LLM cost:** Every input token competes for attention
   in small models. Is the signal-to-noise ratio good enough to
   justify the token cost? Noisy inputs can actively degrade
   output quality for small models by diluting attention from
   stronger signals.
5. **Availability:** What fraction of eligible movies will
   actually have this input populated? If it's sparse, is the
   fallback behavior correct?

Present this as a table or structured list. Group inputs into:

- **Always include** — primary evidence, high signal-to-noise
- **Include when available** — supporting context that helps but
  isn't essential
- **Include with threshold** — useful but only above a quality
  floor (specify the floor and why)
- **Exclude** — redundant, noisy, or not worth the token cost
- **Needs testing** — uncertain; include in evaluation buckets

For inputs with fallback chains (multiple sources for the same
signal), define the fallback order and justify the ranking.

## Step 6: Design eligibility criteria

Based on the input contract, propose eligibility rules:

- **What is the minimum input set for acceptable generation?**
  Which inputs are load-bearing vs nice-to-have?
- **Should there be multiple eligibility paths?** (e.g.,
  strong narrative alone vs strong observations alone vs
  combined)
- **What thresholds?** Propose specific character-length or
  item-count thresholds for each path. Justify each threshold
  with reasoning about what's "enough" for the LLM to work with.
  Frame thresholds as conservative starting points that can be
  loosened after evaluation, not optimistic ones that might let
  low-quality data through.
- **Source-quality weighting:** Should different sources of the
  same signal type have different thresholds? (e.g., LLM-refined
  text vs raw human-written text vs LLM-abstracted text)

Compare your proposal to the current eligibility check in
pre_consolidation.py. Call out what you'd change and why.

## Step 7: Identify evaluation questions

Based on steps 5-6, list the open questions that can only be
answered by running generations and evaluating output quality:

- Which inputs are you uncertain about? (the "needs testing"
  category from step 5)
- Which thresholds are you uncertain about?
- Which fallback paths might produce poor output?
- Where might small-LLM behavior differ from what you'd expect?

Frame each as a testable hypothesis:
> "If we include [input X], output quality for [section Y]
> should improve because [reasoning]. Test by comparing
> generations with and without [input X] on movies that have it."

These become the basis for evaluation bucket design.

## Output format

Present each step's findings before moving to the next. After
each step, pause and ask if I have questions or corrections
before continuing. Do not run through all 7 steps at once.

After step 7, summarize:
1. Proposed input contract (what goes in, with thresholds)
2. Proposed eligibility criteria (paths and thresholds)
3. Open questions for evaluation
4. Delta from current implementation (what changes)
