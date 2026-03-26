# /create-eval-rubric

Design and write a per-result scoring rubric for evaluating individual
metadata generation outputs. The rubric will be appended to the
existing eval guide for the metadata type.

**Usage:** `/create-eval-rubric <type>` — where `<type>` is the
snake_case metadata type name (e.g., `viewer_experience`,
`narrative_techniques`).

---

## The metadata type is: $ARGUMENTS

---

## Step 1: Read scoped context (do not read anything else)

Read ONLY these files. Do not explore the codebase beyond this list —
the rubric design requires understanding the metadata type's purpose,
output structure, prompt instructions, and evaluation goals, nothing
more.

1. `movie_ingestion/metadata_generation/prompts/{type}.py`
   — The system prompt. Understand what the LLM is being asked to
   produce, the output style rules, and the section definitions.

2. `movie_ingestion/metadata_generation/schemas.py`
   — Find the `{Type}Output` class. Understand the field structure
   and what gets embedded.

3. `ingestion_data/{type}_eval_guide.md`
   — The existing evaluation guide. Understand the buckets, research
   questions, analysis framework, and what quality signals are already
   called out (these inform rubric axes).

4. ONE sample evaluation result from
   `movie_ingestion/metadata_generation/evaluation_data/{type}_*.json`
   (not `*_evaluation.json`) — to see what actual output looks like.
   Pick any file; you only need one to calibrate.

5. `ingestion_data/viewer_experience_eval_guide.md` — Read ONLY the
   "Per-Result Scoring Rubric" section (starts after "Post-Evaluation
   Broader Trend Checks"). This is the reference rubric to use as a
   structural template. Match the format: axis table with weights,
   per-axis score descriptors (1-5), holistic score, and "How to
   Apply" section.

Do NOT read: PROJECT.md, conventions.md, other metadata type prompts,
generator code, the full pipeline spec, or any other files. The
generator code is irrelevant — you're evaluating outputs, not the
code that produces them.

---

## Step 2: Think through what "good" looks like

Before proposing axes, reason through:

- **What is this metadata for?** What vector space does it feed?
  What user queries should it match? What makes an output useful
  for retrieval?

- **What are the failure modes?** Based on the eval guide's "Key
  Signals to Watch For" and the prompt's instructions, what can go
  wrong? (e.g., hallucination, generic output, section overfilling,
  poor phrasing for search)

- **What are the quality dimensions?** Group failure modes into
  orthogonal axes. Each axis should measure something distinct. Aim
  for 4-7 axes — enough to be diagnostic, few enough to be usable.

- **How should axes be weighted?** The most damaging failure modes
  should get the highest weights. Weights must sum to 1.0.

Important: some level of generic language in outputs is GOOD. Broad
genre-appropriate terms ensure baseline matching for general queries.
The goal is BOTH broad terms AND specific movie-level terms that
create stronger signal for more targeted queries. Do not create an
axis that penalizes generic terms — instead, create an axis that
rewards having both layers (broad + specific).

---

## Step 3: Present your thinking

Present to the user:

1. **What "good" looks like** for this metadata type (2-3 sentences)
2. **Proposed axes** — name, weight, what it measures, what it
   catches (2-3 lines each)
3. **Questions for input** — anything you're uncertain about or
   where the user's intent matters (numbered for easy response)

Do NOT write the rubric yet. Wait for user feedback. The user may
adjust axes, weights, or framing before you write.

---

## Step 4: Write the rubric

After incorporating user feedback, write the rubric into the eval
guide at `ingestion_data/{type}_eval_guide.md`.

### Placement
Insert the rubric as a new `## Per-Result Scoring Rubric` section
BEFORE the `## After This Evaluation` section (or at the end of the
file if that section doesn't exist).

### Required structure (match the viewer_experience reference)

1. **Introductory paragraph** — what the rubric is for, how to use
   the composite score

2. **Axis weights table** — columns: Axis, Weight, Core question

3. **Per-axis sections** — for each axis:
   - Axis name, weight, and 2-3 sentence description of what it
     measures
   - Score table (1-5) with generic descriptive criteria per level
     (no specific movie examples needed)

4. **Holistic Score section** — unweighted 1-5 with descriptors

5. **How to Apply the Rubric** — evaluation procedure (read inputs
   first, score independently, holistic last, account for input
   richness in cross-bucket comparisons)

### Writing rules
- Keep axis names short (2-3 words)
- Score descriptors should be 1-2 sentences each — concrete enough
  to be actionable, generic enough to apply across movies
- Do not include movie-specific examples in the score descriptors
- Weights must sum to 1.0
- The holistic score is unweighted and separate from the composite
