# /evaluate-metadata-results

Evaluate generated metadata results for a given type by scoring each
movie's candidates against the rubric in the eval guide. Writes
per-movie evaluation JSON files.

**Usage:** `/evaluate-metadata-results <type>` — where `<type>` is the
snake_case metadata type name (e.g., `viewer_experience`).

---

## The metadata type is: $ARGUMENTS

---

## Step 1: Read scoped context (do not read anything else)

Read ONLY these files. Do not explore the codebase beyond this list.

1. `ingestion_data/{type}_eval_guide.md`
   — Read the full guide. The "Per-Result Scoring Rubric" section
   defines the axes, weights, score descriptors, and holistic score
   you will use. The "Buckets" section tells you which input profiles
   exist and what each bucket tests. Internalize both.

2. `movie_ingestion/metadata_generation/schemas.py`
   — Find the `{Type}Output` class AND its `__str__()` method. The
   `__str__()` method defines exactly which fields get embedded into
   the vector space — these are the ONLY fields you grade. Fields
   excluded from `__str__()` (justification, reasoning, explanation
   fields) exist only to help the LLM think and are NOT graded.

3. `movie_ingestion/metadata_generation/prompts/{type}.py`
   — Read the system prompt to understand what inputs the LLM had
   access to. You need this to judge groundedness — whether output
   claims are traceable to input evidence. However, do NOT grade
   based on whether the output follows the system prompt's
   instructions. The prompt may be flawed. Grade based on what
   would be optimal for downstream retrieval (vector search matching
   user queries).

4. All result files matching
   `movie_ingestion/metadata_generation/evaluation_data/{type}_*.json`
   (exclude any files ending in `_evaluation.json` — those are
   previous evaluation outputs, not generation results).
   — These are the files you will evaluate. Note each file's
   structure: tmdb_id, title, user_prompt (the inputs the LLM saw),
   and candidate_results (one or more candidates with their outputs).

Do NOT read: PROJECT.md, conventions.md, generator code, other
metadata type files, or any other files.

---

## Step 2: Establish grading principles

Before scoring anything, internalize these principles:

### Grade what gets embedded, not what gets generated
The `__str__()` method on the Output class determines what text
enters the vector space. Only grade the fields that `__str__()`
includes. Ignore justification/reasoning/explanation fields entirely
— they are scaffolding, not output.

### Grade for retrieval quality, not prompt compliance
The output will be embedded and matched against user search queries.
A "good" result is one that would cause the right user queries to
match this movie — and the wrong queries to NOT match. The system
prompt is a means to this end, not the standard itself. If the prompt
asks for something that would hurt retrieval, a result that deviates
from the prompt but serves retrieval better should score higher.

### Groundedness is judged against the LLM's inputs
The `user_prompt` field in each result file shows exactly what the
LLM saw. A term is grounded if it's traceable to evidence in that
user_prompt. A term is hallucinated if it requires knowledge not
present in the user_prompt — UNLESS the schema's docstring or system
prompt explicitly allows parametric knowledge for that field (e.g.,
`SourceOfInspirationOutput` allows training data for source material
facts). When parametric knowledge is not explicitly allowed, the LLM
should only produce claims supported by its inputs.

### Generic terms are valuable, specific terms are more valuable
Broad genre-appropriate terms (e.g., "tense", "funny", "dark") are
good — they ensure baseline matching for general queries. But the
output should ALSO have specific terms that reflect this particular
movie's experience. Both layers are needed. Grade the balance, not
the presence of either alone.

---

## Step 3: Evaluate movie by movie

Process each result file one at a time. For each movie:

### 3a. Read the inputs
Read the `user_prompt` field carefully. This is the evidence base
for groundedness judgments. Note what data was available (narrative
source, observations, keywords, genre context) and what was marked
"not available."

### 3b. Score all candidates for this movie
For each candidate in `candidate_results`, score every axis defined
in the rubric's "Per-Result Scoring Rubric" section. Apply the score
descriptors from the rubric literally.

For each axis, produce:
- `{axis_name}_score`: integer on the rubric's scale
- `{axis_name}_reasoning`: 1-3 sentences justifying the score,
  referencing specific terms or sections in the output

After all axes, produce the holistic score (if the rubric defines
one):
- `holistic_score`: integer on the rubric's scale
- `holistic_reasoning`: 1-3 sentences on overall impression

### 3c. Write the evaluation file
Write results to:
`movie_ingestion/metadata_generation/evaluation_data/{type}_{tmdb_id}_evaluation.json`

Use this structure:
```json
{
  "tmdb_id": 12345,
  "title": "Movie Title (Year)",
  "candidate_evaluations": {
    "candidate-name-1": {
      "axis_1_score": 4,
      "axis_1_reasoning": "...",
      "axis_2_score": 3,
      "axis_2_reasoning": "...",
      "holistic_score": 4,
      "holistic_reasoning": "..."
    },
    "candidate-name-2": {
      ...
    }
  }
}
```

The axis keys should use the snake_case axis names from the rubric
(e.g., `groundedness_score`, `groundedness_reasoning`,
`specificity_layering_score`, `specificity_layering_reasoning`).

---

## Step 4: Execution approach

- Evaluate ALL movies. Do not skip any result files.
- Process one movie at a time. Read its result file, evaluate all
  candidates, write the evaluation file, then move to the next.
- If you delegate, do it with a SMALL FIXED POOL of workers and give
  each worker a BATCH of movies (for example 3-8 movies each). Reuse
  the same workers for follow-on batches. Do NOT spawn one subagent
  per movie unless the eval set is tiny.
- Each candidate is evaluated independently — do not reference other
  candidates' outputs when scoring. The fact that all candidates are
  in view is for efficiency, not comparison.
- After writing each evaluation file, print a one-line summary:
  `{title} ({tmdb_id}): {candidate_name} = {holistic_score}, ...`
- After all movies are evaluated, print a summary count:
  `Evaluated {N} movies, {M} total candidate evaluations written.`

---

## Step 5: Do NOT analyze

Do not produce cross-movie analysis, aggregate statistics, or
recommendations. This command only evaluates. Analysis is a separate
step (see /understand-metadata-results).
