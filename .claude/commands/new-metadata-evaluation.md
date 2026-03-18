# /new-metadata-evaluation

Scaffold the boilerplate evaluation file for a new metadata type, then
open the spec understanding conversation to design the rubric.

**Usage:** `/new-metadata-evaluation <type>` — where `<type>` is the
snake_case metadata type name (e.g., `plot_analysis`, `viewer_experience`).

---

## The metadata type is: $ARGUMENTS

---

## Step 1: Read orientation files

Read all of these before writing any code:

1. `movie_ingestion/metadata_generation/evaluations/plot_events.py`
   — the reference implementation; scaffold everything from this
2. `movie_ingestion/metadata_generation/schemas.py`
   — find the `{TYPE}Output` class; identify its embeddable fields
   (exclude fields explicitly marked as non-embedded, like
   `review_insights_brief` in `ReceptionOutput`)
3. `movie_ingestion/metadata_generation/generators/{type}.py`
   — find the `build_{type}_user_prompt()` function
4. `movie_ingestion/metadata_generation/prompts/{type}.py`
   — find the `SYSTEM_PROMPT` constant
5. `movie_ingestion/metadata_generation/evaluations/run_evaluations_pipeline.py`
   — where to add the new type skeleton

---

## Step 2: Identify the embeddable fields

From the `{TYPE}Output` class in schemas.py:
- List all fields on the class
- Exclude any the docstring explicitly marks as intermediate or
  non-embedded (not stored in Qdrant, not consumed by downstream generators)
- The remaining fields are the **embeddable fields** — these become
  the DB table columns and the content the judge evaluates

Note that some schemas have sub-models (e.g., `MajorCharacter` in
`PlotEventsOutput`, `CharacterArc` in `PlotAnalysisOutput`). Treat
these as JSON-serialized TEXT columns, just like `major_characters`
in the plot_events tables.

---

## Step 3: Scaffold `evaluations/{type}.py`

Create `movie_ingestion/metadata_generation/evaluations/{type}.py`
by adapting `plot_events.py`. Rules:

### Names to replace throughout the file
- `plot_events` / `PlotEvents` / `PLOT_EVENTS` → `{type}` / `{Type}` / `{TYPE}`
- `PlotEventsOutput` → the correct `{TYPE}Output` class name
- `build_plot_events_user_prompt` → `build_{type}_user_prompt`
- `SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT` → import from `prompts.{type}`
- `PlotEventsJudgeOutput` → `{TYPE}JudgeOutput`
- Table names: `plot_events_references` → `{type}_references`, etc.

### `_CREATE_REFERENCES_TABLE`
Generate columns from the embeddable fields:
- `movie_id INTEGER PRIMARY KEY`
- One column per embeddable field — TEXT for strings and JSON arrays,
  INTEGER for integers; annotate JSON columns with `-- JSON array` comment
- Always include: `reference_model TEXT NOT NULL`, `input_tokens INTEGER`,
  `output_tokens INTEGER`, `created_at TEXT NOT NULL`

### `_CREATE_CANDIDATE_OUTPUTS_TABLE`
Same columns as references, but:
- `PRIMARY KEY (movie_id, candidate_id)`
- Add `candidate_id TEXT NOT NULL` after `movie_id`

### `_CREATE_EVALUATIONS_TABLE`
Leave the dimension columns as a TODO — they can't be determined until
the rubric is designed. Include this placeholder:
```python
_CREATE_EVALUATIONS_TABLE = """
    CREATE TABLE IF NOT EXISTS {type}_evaluations (
        movie_id     INTEGER NOT NULL,
        candidate_id TEXT NOT NULL,
        -- TODO: add one pair of columns per evaluation dimension:
        --   {dimension}_score     INTEGER,
        --   {dimension}_reasoning TEXT,
        judge_model            TEXT,
        judge_input_tokens     INTEGER,
        judge_output_tokens    INTEGER,
        created_at             TEXT NOT NULL,
        PRIMARY KEY (movie_id, candidate_id)
    )
"""
```

### `SCORE_COLUMNS` and `{TYPE}JudgeOutput`
```python
# TODO: fill in after rubric dimensions are decided in the spec conversation
SCORE_COLUMNS: list[str] = []

class {TYPE}JudgeOutput(BaseModel):
    """Structured output from the Claude judge for {type} evaluation.

    TODO: Add one reasoning: str field followed by one score: Literal[1, 2, 3, 4]
    field for each evaluation dimension. Reasoning must come before scores
    (spec requirement: explicit chain-of-thought before scores).
    """
    pass  # TODO: add dimension fields
```

### `JUDGE_SYSTEM_PROMPT`
```python
JUDGE_SYSTEM_PROMPT = """# TODO: Fill in the evaluation rubric.
#
# See docs/llm_evaluations_guide.md for rubric design principles:
# - Break the schema into constituent dimensions (individual fields or
#   field groups) and score each separately
# - Every score level must have observable, concrete criteria
# - Distinguish objective fields (verifiable facts) from interpretive fields
# - Explicitly encode verbosity expectations per field
# - Describe what a high-quality response looks like for each dimension
# - Semantic similarity as first-class concern for interpretive fields
#
# Structure:
# 1. Role and task
# 2. Full rubric for each dimension (score 1-4 criteria)
# 3. Scoring instructions (evaluate semantic content, not surface form;
#    reference is a calibration anchor, not ground truth; chain-of-thought
#    before scores)
# 4. Output format specification
"""
```

### `CANDIDATES` list
```python
# TODO: populate after the rubric is designed in the spec conversation.
# Candidate IDs should be prefixed with "{type}__" (e.g., "{type}__gemini-2.5-flash").
# Copy the candidate set structure from plot_events.py as a starting point.
{TYPE}_CANDIDATES: list[EvaluationCandidate] = []
```

### `_serialize_output` and `_deserialize_output`
Implement fully based on the embeddable fields you identified:
- `str` fields → stored/retrieved as-is (TEXT column)
- `list[str]` fields → `json.dumps(value)` / `json.loads(raw)`
- `list[SubModel]` fields → `json.dumps([m.model_dump() for m in value])` /
  `[SubModel.model_validate(m) for m in json.loads(raw)]`

These functions should be fully implemented — they're mechanical once
you know the field types.

### `_format_output_for_prompt`
Format each embeddable field as labelled, readable text for the judge
user prompt. Mimic the style from `_format_characters_for_prompt()` in
plot_events.py. For list fields, join with newlines or commas as appropriate
for readability.

### `_build_judge_user_prompt`
Copy directly from plot_events.py, substituting:
- The call to `_format_characters_for_prompt` → `_format_output_for_prompt`
- The field names in the formatted output block

### `generate_reference_responses` and `run_evaluation`
Copy the full function bodies from plot_events.py, substituting:
- Table names and schema class throughout
- `build_plot_events_user_prompt` → `build_{type}_user_prompt`
- `_serialize_output` / `_deserialize_output` (already renamed above)
- `create_plot_events_tables` → `create_{type}_tables`
- `PlotEventsOutput` → `{TYPE}Output`
- `PlotEventsJudgeOutput` → `{TYPE}JudgeOutput`
- The `store_candidate` call: `"plot_events"` → `"{type}"`
- The judge INSERT statement: update column list to match the TODO
  evaluations table (leave as a note that it needs updating after
  rubric is designed)
- Progress print statements: update labels

### `print_score_summary`
Copy from plot_events.py. Replace the table name and add a TODO for
the `dims` / `short_labels` lists:
```python
# TODO: fill in after SCORE_COLUMNS is finalized
dims = []        # e.g. ["groundedness", "field_quality", ...]
short_labels = []  # abbreviated labels matching dims order
```

---

## Step 4: Update `run_evaluations_pipeline.py`

Add imports at the top (after the existing plot_events imports):
```python
from movie_ingestion.metadata_generation.evaluations.{type} import (
    {TYPE}_CANDIDATES,
    generate_reference_responses as generate_{type}_references,
    run_evaluation as run_{type}_evaluation,
)
```

Add a skeleton entry in `main()` after the plot_events block:
```python
# --- {TYPE} ---
# TODO: add an eligibility filter that calls the appropriate check_{type}()
# from pre_consolidation.py (analogous to _filter_plot_events_eligible above).

print(f"\n--- Phase 0: Reference Generation ({type}) ---")
await generate_{type}_references(movie_inputs)  # TODO: pass filtered eligible_inputs

print(f"\n--- Phase 1: Candidate Evaluation ({type}) ---")
await run_{type}_evaluation(
    candidates={TYPE}_CANDIDATES,
    movie_inputs=movie_inputs,  # TODO: pass filtered eligible_inputs
    concurrency=3
)
```

Also update the module docstring's "Currently supports:" list to add the
new type.

---

## Step 5: Run the spec understanding conversation

After scaffolding, invoke the `initiate-spec-understanding-conversation`
skill, passing this as the argument:

> `docs/llm_evaluations_guide.md`, specifically for implementing the
> evaluation process for **{TYPE} metadata**, which is generated by
> `generate_{type}()`. The scaffold file is at
> `evaluations/{type}.py`. Focus on what needs to be decided to fill
> in the TODOs: the evaluation dimensions, the rubric
> (JUDGE_SYSTEM_PROMPT), the JudgeOutput fields, the evaluations table
> schema, and the SCORE_COLUMNS list. Do NOT write code yet — surface
> all decisions and conflicts that must be resolved first.
