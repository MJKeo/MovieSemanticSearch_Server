# /improve-metadata-prompt

Evaluate and propose improvements to the system prompt(s) for a
metadata generation type. The goal is to assess whether the current
prompt is optimally structured for producing the best possible output
across the full spectrum of eligible inputs, and to develop a concrete
improvement plan.

Do not write code. This is a research-then-discuss process. You will
gather context, form opinions, and present them for discussion.

## The metadata type is: $ARGUMENTS

---

## Phase 1: Build Understanding

Read all of these before any analysis. Do not present findings yet —
this phase is purely intake.

### 1a. Pipeline context

- `docs/PROJECT.md` — priorities and constraints
- `docs/conventions.md` — cross-codebase invariants
- `DIFF_CONTEXT.md` — recent changes that may affect this type
- The module doc in `docs/modules/` for ingestion if one exists
- `docs/llm_metadata_generation_new_flow.md` — the full generation
  pipeline spec, especially the section for this metadata type

### 1b. What this metadata type produces and why

- `movie_ingestion/metadata_generation/schemas.py` — find the
  `{Type}Output` class, its fields, and its `__str__()` method
  (which determines what text gets embedded into the vector space)
- `implementation/prompts/` — how the search pipeline uses this
  vector space (subquery generation prompts, vector space weight
  assignment). Understand what kinds of user queries this vector
  space is designed to match.

### 1c. The current system prompt(s) and generator

- `movie_ingestion/metadata_generation/prompts/{type}.py` — the
  full system prompt text
- `movie_ingestion/metadata_generation/generators/{type}.py` — how
  the user prompt is assembled, what inputs are passed to the LLM

### 1d. Input contract and eligibility

- `movie_ingestion/metadata_generation/inputs.py` — `MovieInputData`
  fields, `build_user_prompt()` helper, Wave 1 output loading
- `movie_ingestion/metadata_generation/pre_consolidation.py` —
  eligibility checks for this type (understand what the minimum
  viable input looks like)
- `movie_ingestion/metadata_generation/generator_registry.py` —
  Wave 1 vs Wave 2 dependencies, model configuration

### 1e. Evaluation design (edge case awareness)

- `ingestion_data/{type}_eval_guide.md` — the evaluation guide for
  this type (if it exists). This describes the input quality buckets
  and what edge cases matter. Internalize the bucket definitions —
  they represent the spectrum of input quality you need to reason
  about.
- `ingestion_data/{type}_eval_buckets.json` — the specific movies
  selected for each bucket (if it exists)

### 1f. Sample inputs from real movies

Load 3-5 real movies from the tracker database to see what the
inputs actually look like in practice. Choose movies that span the
quality spectrum:

- One movie that would be a "gold standard" input (rich data)
- One movie near the eligibility floor (thin data)
- One movie in the middle

Use `movie_ingestion/metadata_generation/inputs.py`'s
`load_movie_input_data()` to understand the data shape and the
tracker DB at `ingestion_data/tracker.db` to find candidate movies.
Read the raw data to understand what the LLM will actually see.

---

## Phase 2: Present Understanding

Present what you learned, organized as:

1. **Purpose of this metadata type** — What vector space does it
   feed? What user queries should it match? What makes a strong
   embedding for this space?

2. **What the LLM is being asked to do** — Summarize the task the
   system prompt defines. What are the key instructions? What
   constraints does it impose?

3. **Input landscape** — What does the LLM receive in the user
   prompt? How does input quality vary across the eligible
   population? What do thin inputs vs rich inputs actually look
   like? (Reference the real movies you examined.)

4. **Output contract** — What fields must be populated? Which get
   embedded vs used as intermediate data? What does the `__str__()`
   method produce?

Pause here. Wait for confirmation that your understanding is correct
before proceeding to evaluation.

---

## Phase 3: Evaluate the Current Prompt

Now take a critical look at the system prompt(s). Think from the
perspective of a model roughly as intelligent as gpt-5-mini with
minimal reasoning (low reasoning effort, single-shot batch generation
with no back-and-forth). This model is capable but not deeply
reflective — clarity of instruction matters more than subtlety.

Approach this as if you have free reign to rebuild from scratch, not
as if you're limited to minor tweaks. Evaluate along these
dimensions:

### 3a. Instruction clarity
- Is the task framed clearly enough that a capable-but-not-deeply-
  reasoning model will understand what's expected on first read?
- Are there instructions that are ambiguous, could be interpreted
  multiple ways, or rely on implicit understanding?
- Are there places where the prompt tells the model *what not to
  do* when it would be clearer to tell it *what to do*?

### 3b. Information ordering
- Does the prompt lead with the most important context, or does
  the model need to read deep into the prompt before understanding
  its core task?
- Is related information grouped together, or scattered?
- Would reordering sections improve comprehension for a model that
  processes sequentially?

### 3c. Coverage across input quality spectrum
- How will this prompt perform with gold-standard inputs (rich plot
  synopses, many keywords, detailed reviews)?
- How will it perform with floor-level inputs (barely-eligible,
  sparse data)?
- Are there instructions that implicitly assume rich inputs and
  would confuse the model when inputs are thin?
- Are there edge cases in the eligible population (foreign films,
  documentaries, very old films, niche genres) where the prompt's
  assumptions might break down?

### 3d. Token efficiency
- Are there sections that are verbose without adding instructional
  value?
- Is there repetition that could be consolidated?
- Are there examples or elaborations that could be trimmed without
  losing clarity?
- Could the same instructions be conveyed more concisely?

### 3e. Output quality alignment
- Does the prompt guide the model toward producing text that will
  embed well for the intended vector space?
- Are the field-level instructions aligned with what `__str__()`
  actually outputs?
- Does the prompt's framing encourage the right level of
  specificity/abstraction for the embedding use case?

Present your evaluation as a list of observations. For each:
- State what you noticed
- Explain why it matters (what failure mode it could cause)
- Indicate severity: minor (polish), moderate (could affect output
  quality for some inputs), or significant (likely affects output
  quality across many inputs)

Pause here for discussion before proposing changes.

---

## Phase 4: Propose Improvements

Based on the discussion, present your improvement proposal:

1. **What you'd change and why** — Concrete proposed changes, each
   with the reasoning behind it and what failure mode it addresses.
   Reference specific sections of the current prompt.

2. **What you'd keep and why** — Parts of the current prompt that
   are well-designed and should be preserved. Don't just list
   everything — call out the parts that are notably effective.

3. **Open questions** — Anything you're uncertain about that would
   benefit from seeing real generation output before deciding.

This does not need to be a rigid format. Organize it in whatever
way best communicates your thinking and the reasoning behind your
recommendations.
