# /build-playground-eval-cell

Write a new cell in `metadata_generation_playground.ipynb` that runs
candidate model comparisons for a metadata type and saves per-movie
evaluation JSON files.

**Usage:** `/build-playground-eval-cell <type> <candidates> <rate>`

- `<type>` — snake_case metadata type (e.g. `viewer_experience`,
  `narrative_techniques`, `watch_context`)
- `<candidates>` — comma-separated candidate specs, each formatted as
  `label:model:reasoning_effort` (e.g.
  `gpt-5-mini-low:gpt-5-mini:low,gpt-5.4-nano-medium:gpt-5.4-nano:medium`)
- `<rate>` — target generations per second (e.g. `120`). The per-movie
  rate is `rate / num_candidates`.

## The arguments are: $ARGUMENTS

---

## Step 1: Read these files (all reads, no writes)

Read every file below before writing anything. The metadata type from
the arguments determines which files to read.

1. **The generator:**
   `movie_ingestion/metadata_generation/generators/{type}.py`
   — Find `generate_{type}()` and `build_{type}_user_prompt()`.
   Note the function signatures: what upstream Wave 1 metadata fields
   does the generator accept beyond `movie: MovieInputData`? These are
   the fields you must load from the `generated_metadata` table and
   pass through.

2. **The eval buckets file:**
   `ingestion_data/{type}_eval_buckets.json`
   — Understand the bucket structure. Two formats exist:
   - **Standard** (most types): top-level `"buckets"` dict, each
     bucket has `"movies"` list with `"tmdb_id"` in each entry.
   - **Legacy** (reception only): flat dict of bucket names, each
     with a `"samples"` list containing `"tmdb_id"`.
   Adapt the loading code to whichever format this file uses.

3. **The reference cell (viewer_experience):**
   Read cell 8 of `movie_ingestion/metadata_generation/metadata_generation_playground.ipynb`
   — This is the pattern to follow. Study its structure:
   candidates → bucket loading → DB queries → MovieInputData builder →
   Wave 1 input extractor → rate-limited generator → per-movie JSON save.

4. **Existing notebook cells 0–4:**
   Read cells 0–4 of the notebook to understand what's already
   imported and defined. The cell you write can rely on:
   - `project_root`, `sqlite3`, `json`, `asyncio`, `Path`
   - `MovieInputData`, all `generate_*` imports, `LLMProvider`
   - `deserialize_imdb_row` from `movie_ingestion.tracker`
   - `PlaygroundCandidate` dataclass
   - `MODEL_PRICING` dict and `_compute_cost()` function
   - `_merge_candidate_results()` helper

5. **The schemas file (skim only):**
   `movie_ingestion/metadata_generation/schemas.py`
   — Find the `{Type}Output` class to confirm the response model name.

---

## Step 2: Determine Wave 1 input requirements

From the generator signature in step 1, classify the metadata type:

- **Wave 1 (no upstream metadata):** `plot_events`, `reception`
  — Only needs `MovieInputData`. No `_extract_wave1_inputs` needed.
- **Wave 2 (depends on Wave 1 outputs):** `plot_analysis`,
  `viewer_experience`, `watch_context`, `narrative_techniques`,
  `production_keywords`, `source_of_inspiration`
  — Needs upstream fields from `generated_metadata` table columns
  (`plot_events`, `reception`, `plot_analysis`). Build an
  `_extract_wave1_inputs()` function that reads the relevant JSON
  columns and returns a kwargs dict matching the generator's
  parameter names.

Key patterns for extracting Wave 1 fields:
- `plot_events` JSON has `plot_summary` (string)
- `reception` JSON has `emotional_observations`, `craft_observations`,
  `thematic_observations`, `source_material_hint` (all strings)
- `plot_analysis` JSON has `generalized_plot_overview` (string),
  `genre_signatures` (list[str]), `character_arcs` (list of dicts —
  extract `arc_transformation_label` strings), `thematic_concepts`
  (list of dicts)

Match the generator's parameter names exactly. Only extract fields
the generator actually accepts.

---

## Step 3: Write the cell

Add a **new cell after cell 8** in the notebook (use `edit_mode=insert`
with `cell_id` of cell 8). Follow the viewer_experience cell as the
structural template, adapting for the target type.

### Cell structure (in order):

1. **Header comment block** — type name, candidates, rate target
2. **Imports** — `build_{type}_user_prompt` and `generate_{type}`
   from `generators.{type}`
3. **Candidate definitions** — build `PlaygroundCandidate` list from
   the `<candidates>` argument
4. **Bucket loading** — read `{type}_eval_buckets.json`, extract
   tmdb_ids per bucket into a `{type_abbrev}_eval_groups` dict
5. **DB queries** — connect to `tracker.db`, fetch `tmdb_data`,
   `imdb_data`, and (if Wave 2) `generated_metadata` for all IDs
6. **`_build_{type_abbrev}_movie()` function** — constructs
   `MovieInputData` from tmdb + imdb rows (same pattern as reference)
7. **`_extract_wave1_inputs()` function** (Wave 2 only) — extracts
   upstream metadata fields into a kwargs dict
8. **Build loop** — populate `{type_abbrev}_movies` and
   `{type_abbrev}_wave1_inputs` dicts
9. **Rate limiter setup** — semaphore + token-bucket from `<rate>`
10. **`_generate_for_candidate()` async function** — calls
    `generate_{type}()` with rate limiting, returns result dict
11. **Main generation loop** — iterates buckets → movies, fires all
    candidates concurrently per movie via `asyncio.gather`
12. **Per-movie JSON save** — writes to
    `evaluation_data/{type}_{tmdb_id}.json`

### Output file format (per movie):

```json
{
  "tmdb_id": 12345,
  "title": "Movie Title (2024)",
  "user_prompt": "...",
  "candidate_results": {
    "candidate-label-1": {
      "candidate": "candidate-label-1",
      "model": "gpt-5-mini",
      "result": { ... },
      "input_tokens": 1234,
      "output_tokens": 567,
      "cost_usd": 0.001234
    },
    "candidate-label-2": { ... }
  }
}
```

The `candidate_results` value is a **dict keyed by candidate label**
(not a list).

### Naming conventions:

- All variable/function prefixes use a short abbreviation of the type
  to avoid collisions with other cells (e.g. `ve_` for
  `viewer_experience`, `nt_` for `narrative_techniques`).
- Output files: `{type}_{tmdb_id}.json`
- Eval groups dict: `{abbrev}_eval_groups`
- Movies dict: `{abbrev}_movies`

---

## Step 4: Verify

After writing the cell, re-read it from the notebook to confirm:
- Imports reference the correct generator module
- Wave 1 field extraction matches the generator's actual parameters
- Bucket loading matches the actual JSON structure
- Rate limit math is correct: semaphore = rate / num_candidates,
  interval = 1.0 / rate
- All variable prefixes are consistent and don't collide with
  existing cells
