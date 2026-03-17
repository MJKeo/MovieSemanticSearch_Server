# Conventions Draft

Observed patterns staged for review. Remove entries you disagree
with, then run /solidify-draft-conventions to merge the rest into
docs/conventions.md.

Entries are added automatically during /safe-clear based on
patterns observed in the session.

## Decompose multi-case evaluations into per-case methods
**Observed:** User directed refactoring 8 inline skip conditions (if/else blocks inside _assess_wave1/_assess_wave2) into 8 individual _check_<type>() methods, each returning str | None. The orchestrator becomes a thin loop over individual checks.
**Proposed convention:** When a function evaluates N independent cases with distinct logic (e.g., eligibility checks, validation rules), extract each case into its own method rather than inlining all cases in a single function. Compose via a thin orchestrator.
**Sessions observed:** 1

## Consistent __str__() lowercasing on all embeddable schema models
**Observed:** Code review found 5 of 8 schema `__str__()` methods missing `.lower()` calls on terms, while the other 3 lowercased. Since these strings become embedding input text, inconsistent casing could affect embedding quality. All were fixed to lowercase uniformly.
**Proposed convention:** All Pydantic schema classes whose `__str__()` output feeds the embedding pipeline must lowercase their concatenated terms. When adding a new schema, match the lowercasing pattern of existing schemas.
**Sessions observed:** 1

## Consistent return type across all LLM provider methods
**Observed:** Pre-existing bug where Kimi async/sync methods returned only the parsed model while OpenAI returned `Tuple[BaseModel, int, int]`. This mismatch caused the unified routing function to fail when dispatching to Kimi. All 5 QU callers had to be updated after fixing.
**Proposed convention:** All LLM generation functions (sync and async) must return `Tuple[BaseModel, int, int]` (parsed_response, input_tokens, output_tokens). When adding a new provider, match this signature exactly. The unified router depends on it.
**Sessions observed:** 1

## Dynamic project root resolution in notebooks
**Observed:** User replaced hardcoded `Path().resolve().parent.parent` chains in notebook Cell 0 with a `find_project_root()` helper that walks up from CWD looking for `pyproject.toml`. The hardcoded approach silently breaks when notebooks are opened from a different working directory.
**Proposed convention:** All Jupyter notebooks must resolve the project root dynamically by walking up from `__file__` or `Path.cwd()` until `pyproject.toml` is found, then add that path to `sys.path`. Never use hardcoded parent-chain traversal.
**Sessions observed:** 1

## Patch dispatch dicts directly in router tests
**Observed:** Router tests for `generate_llm_response_async` failed when patching module-level function names because `_PROVIDER_DISPATCH` captures direct function references at module load time. The fix was `patch.dict(_PROVIDER_DISPATCH, {LLMProvider.X: mock_fn})` instead of `patch("module.function_name", mock_fn)`.
**Proposed convention:** When testing code that dispatches through a module-level dict (or any eagerly-bound lookup table), patch the dict entries directly with `patch.dict()` rather than patching the module-level names that the dict was populated from.
**Sessions observed:** 1

## Shared exception classes over per-module custom errors
**Observed:** User directed replacing a `PlotEventsGenerationError` specific to plot_events.py with two shared exceptions (`MetadataGenerationError`, `MetadataGenerationEmptyResponseError`) in a central errors.py, parameterized by generation_type and title. Avoids proliferating N error classes as N generators are added.
**Proposed convention:** When multiple modules have identical failure modes differing only by context (e.g., which generation type failed), use a single shared exception class parameterized with context fields rather than creating per-module exception subclasses.
**Sessions observed:** 1

## Evaluation results in per-type SQLite tables with columnar scores
**Observed:** The plot_events evaluation pipeline stores results in `evaluation_data/eval.db` with three tables per metadata type (references, candidate_outputs, evaluations). Each scoring dimension is a separate typed column — not a JSON blob — enabling SQL queries over individual scores.
**Proposed convention:** LLM evaluation pipelines store results in `evaluation_data/eval.db`. Each metadata type gets its own table group. Scoring dimensions must be individual columns (Literal integer types), not JSON. Aggregate per-candidate summaries are computed at read time via pandas, not stored.
**Sessions observed:** 1

## EvaluationCandidate dataclass as the standard LLM-under-test config unit
**Observed:** `shared.py` defines a frozen `EvaluationCandidate` dataclass (candidate_id, provider, model, system_prompt, response_format, kwargs). All evaluation entry points take a list of these rather than ad-hoc keyword arguments. Prevents magic strings at call sites.
**Proposed convention:** Every evaluation script must accept its LLM configurations as a list of `EvaluationCandidate` instances from `evaluations/shared.py`. Do not pass provider/model/prompt as loose keyword arguments to evaluation functions.
**Sessions observed:** 1

## Two-phase idempotent evaluation: reference generation before candidate scoring
**Observed:** The plot_events evaluation has Phase 0 (generate Claude Opus reference responses) and Phase 1 (score each candidate against references). Both phases check for existing rows before inserting. Phase 1 asserts Phase 0 is complete before running.
**Proposed convention:** All metadata-type evaluations must follow the two-phase pattern: Phase 0 generates reference responses using the designated judge model; Phase 1 runs candidates and scores them. Both phases are idempotent. Never merge phases or skip Phase 0 for speed.
**Sessions observed:** 1

## Extract shared prompt builders from generators for evaluation reuse
**Observed:** The plot_events evaluation duplicated the user prompt construction logic from `generators/plot_events.py`. Code review flagged this as a maintenance risk — if the generator's prompt changed, the eval copy would silently diverge. Fixed by extracting `build_plot_events_user_prompt()` as a public function in the generator, imported by both the generator and the eval pipeline.
**Proposed convention:** When an evaluation pipeline needs the same prompt as a production generator, extract the prompt construction into a named public function in the generator module. The eval pipeline imports it rather than duplicating. This prevents silent drift between what candidates are actually asked and what the judge thinks they were asked.
**Sessions observed:** 1

## Candidate ID naming: {type}__{model}__{variant}
**Observed:** When expanding from 1 to 2-3 candidates per model, a consistent naming scheme emerged: baseline candidates use `{metadata_type}__{model}` (e.g., `plot_events__gemini-2.5-flash`), variant candidates append `__{variant-description}` (e.g., `plot_events__gemini-2.5-flash__think-1k`). Baselines keep the original ID for DB continuity with existing evaluation rows.
**Proposed convention:** Evaluation candidate IDs must follow the pattern `{metadata_type}__{model}` for the baseline configuration and `{metadata_type}__{model}__{variant}` for alternative configurations. Double-underscore separates the three semantic segments. Variant suffixes should be short and describe what changed (e.g., `think-1k`, `reason-low`, `temp-0`).
**Sessions observed:** 1

## Hard-coded pricing maps in analysis scripts, not in shared infrastructure

**Observed:** `analyze_results.py` stores `MODEL_PRICING` as a module-level dict keyed by model name. Prices are static constants with a comment citing the source and date. The dict lives in the analysis file rather than in `shared.py` or a config file.
**Proposed convention:** Per-model pricing constants belong in the analysis file that uses them, not in shared infrastructure. They are documentation-adjacent (cite source + date in a comment) and will need manual updates as prices change — keeping them local makes the update obvious and contained.
**Sessions observed:** 1

## Claude Opus via ANTHROPIC_OAUTH_KEY is the designated judge and reference model
**Observed:** Evaluation design explicitly chose Claude (via OAuth, not the standard API key) as both the reference generator and the judge for all LLM evaluations. The OAuth token is stored as `ANTHROPIC_OAUTH_KEY` in `.env`.
**Proposed convention:** All evaluation pipelines use Claude Opus (via `ANTHROPIC_OAUTH_KEY`) for Phase 0 reference generation and for judge scoring. Do not substitute a different judge model without a documented decision record, as this breaks cross-evaluation comparability.
**Sessions observed:** 1

## Candidates defined per-metadata-type, not in shared infrastructure
**Observed:** Initial design placed a global `CANDIDATES` list in `shared.py`. User directed moving it to `plot_events.py` as `PLOT_EVENTS_CANDIDATES`, stating "the best way to handle candidates is to actually define a unique list for each metadata group."
**Proposed convention:** Evaluation candidates live in the per-type evaluation file (e.g., `PLOT_EVENTS_CANDIDATES` in `evaluations/plot_events.py`), not in `shared.py`. The shared module provides the `EvaluationCandidate` dataclass and infrastructure; each type owns its candidate list. The runner imports candidates from the type-specific module.
**Sessions observed:** 1

## Keep backward-compat alias when making private functions public

**Observed:** `_check_plot_events` was made public as `check_plot_events` so it could be imported directly by `run_evaluations_pipeline.py`. The test suite already imported `_check_plot_events` by its private name, so a `_check_plot_events = check_plot_events` alias was added to prevent breaking the import without touching the test file.
**Proposed convention:** When making a private function public (removing the `_` prefix), always leave a `_old_name = new_name` alias in the same module. This preserves existing callers — especially test imports — without needing to touch files outside the module being changed.
**Sessions observed:** 1

## Judge rubric must embed the generation prompt's requirements
**Observed:** JUDGE_SYSTEM_PROMPT was found to conflict with the generation SYSTEM_PROMPT in 9 ways — penalizing "secondary threads collapsed" while the generator explicitly asks for "only 1-3 core conflicts"; rewarding "narrative arc" framing the generator doesn't use; missing signals for compactness and theme-talk avoidance. Fixed by adding an explicit "THE GENERATION PROMPT instructs the model to:" section to the rubric.
**Proposed convention:** Every evaluation judge prompt must include a concise summary of the generation prompt's instructions (what the model was asked to do and avoid). The judge cannot score faithfully if it doesn't know what the generator was optimizing for. Rubric score anchors must align with — not contradict — the generation prompt's requirements.
**Sessions observed:** 1
