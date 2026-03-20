# Conventions Draft

Observed patterns staged for review. Remove entries you disagree
with, then run /solidify-draft-conventions to merge the rest into
docs/conventions.md.

Entries are added automatically during /safe-clear based on
patterns observed in the session.

## Schema variants always get matching prompt variants
**Observed:** For plot_analysis, with-justifications and without-justifications
schema variants needed separate prompt variants (structural difference: sub-objects
vs flat labels). For viewer_experience, the structural difference was minor (just
an extra justification string per section), but the user explicitly requested
matching prompt variants anyway — wanting consistency and explicit control over
what the prompt tells the LLM across all metadata types.
**Proposed convention:** When a metadata generation type has multiple output
schema variants (e.g., with/without justification fields), always create matching
prompt variants. Extract shared sections into private constants; only the sections
that describe differing output fields are variant-specific. This applies regardless
of how structurally different the schemas are.
**Sessions observed:** 6

## No provider-specific default kwargs in multi-provider generators
**Observed:** Generators had `_DEFAULT_KWARGS = {"reasoning_effort": "low", "verbosity": "low"}` merged into every call. These OpenAI-only params caused 400 errors on Gemini, Groq, and Alibaba providers. Fixed by removing defaults — callers pass exactly the kwargs their provider needs.
**Proposed convention:** Generators that support multiple LLM providers must not have provider-specific default kwargs. Each caller is responsible for passing the complete kwargs for its target provider. Only generators locked to a single provider (like plot_events with its Gemini default) may have default kwargs.
**Sessions observed:** 2

## Defaults in parameter signature, not method body
**Observed:** Generator functions were written with `system_prompt: str | None = None` then `system_prompt = system_prompt or SYSTEM_PROMPT` in the body. User asked to put defaults directly in the parameter signature (`system_prompt: str = SYSTEM_PROMPT`) — cleaner, more self-documenting, and avoids body-level fallback logic.
**Proposed convention:** When a function parameter has a known default value available at module scope, put it in the parameter signature rather than using `None` + body-level fallback. Reserve `None` defaults for cases where the default must be computed at call time or depends on other arguments.
**Sessions observed:** 1

## No unnecessary variable indirection
**Observed:** User corrected `effective_kwargs = kwargs` followed by `**effective_kwargs` — an assignment that doesn't transform the value. Asked to remove the alias and use `**kwargs` directly.
**Proposed convention:** Don't introduce intermediate variables that merely alias another variable without transformation. If the value isn't conditionally constructed (e.g., `{**defaults, **overrides}`), use the original directly.
**Sessions observed:** 1

## Cross-reference evaluation candidates as source of truth for model params
**Observed:** Playground candidates were initially derived from stale `run_*` wrapper functions with incorrect params (wrong temperature for qwen, missing temperature for gemini/groq). The evaluation candidates file had the correct values.
**Proposed convention:** When defining model candidate configurations outside the evaluation pipeline (playgrounds, scripts, etc.), always cross-reference kwargs against the corresponding `EvaluationCandidate` definitions in `movie_ingestion/metadata_generation/evaluations/`. Those are the authoritative source for provider/model/kwargs combinations.
**Sessions observed:** 1

## Evaluation conventions — reference-free, source-data-based judging
**Observed:** The evaluation pipeline was restructured to remove reference-based
evaluation (Phase 0) entirely. Research showed rubric matters ~2.7x more than
reference for human judgment alignment on subjective metadata tasks. The judge
now receives raw SOURCE DATA (the same movie fields the candidate saw) instead
of the generation prompt's instructions, and scores against quality criteria
defined in the rubric. Storage is two tables per metadata type (candidate_outputs,
evaluations) — the references table is no longer created.
**Proposed convention:** Replace the current Evaluation Conventions section
(lines 218-251 of conventions.md) with:
- **Storage structure**: two tables per metadata type: candidate_outputs and
  evaluations. Scoring dimensions are individual typed columns.
- **Two-phase structure** bullet → replace with: Reference-free pointwise
  evaluation. For each (candidate, movie) pair, generate output, score with
  rubric-based LLM judge (Claude Opus 4.6, thinking disabled). Judge sees raw
  source data, not generation instructions or reference outputs. Multi-run
  averaging (2 sequential runs for prompt caching). Idempotent. 429 rate
  limits trigger 30s sleep + retry.
- **Judge prompt alignment** bullet → replace with: Every judge prompt must
  include the raw source data that was available to the candidate, plus the
  candidate output. Rubric score anchors define quality criteria independently
  of the generation prompt. The judge evaluates against SOURCE DATA for factual
  verification and against the rubric's quality standards for style/completeness.
**Sessions observed:** 1

## Let retryable exceptions propagate through generic error wrappers
**Observed:** `generate_anthropic_response_async` wrapped all exceptions into `ValueError`, making it impossible for callers to distinguish rate limits from parse errors. Had to add an explicit `except anthropic.RateLimitError: raise` before the catch-all so callers can implement retry logic.
**Proposed convention:** When a function wraps exceptions into a generic type (e.g., `except Exception as e: raise ValueError(...)`), always re-raise retryable exceptions (rate limits, transient network errors) before the catch-all. Callers need the original exception type to decide whether to retry or fail.
**Sessions observed:** 1

## Always use current library API conventions
**Observed:** Code accessed `model_fields` on a Pydantic v2 instance instead of the class, triggering a deprecation warning. The deprecated pattern still worked but will break in v3.
**Proposed convention:** Always code against the current, non-deprecated API patterns of whatever library version is installed. Do not rely on deprecated access patterns even if they still function. When unsure, verify the recommended usage for the installed version. This applies to all libraries, not just Pydantic.
**Sessions observed:** 1
