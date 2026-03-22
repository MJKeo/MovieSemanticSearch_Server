# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Rewrite notebook cell 4 for multi-candidate per-movie evaluation
Files: `movie_ingestion/metadata_generation/metadata_generation_playground.ipynb` (cell 4)

### Intent
Replace single-candidate, per-group JSON output with multi-candidate, per-movie JSON output for comparing plot_events generation across all non-Gemini models.

### Key Decisions
- **Halved evaluation groups** from 6 to 3 movies each (first half of each array) — 21 movies total.
- **Per-movie JSON files** at `evaluation_data/plot_events_{tmdb_id}.json` instead of per-group files. Each file stores: tmdb_id, title, user_prompt (once), and candidate_results array with model name, result, tokens, and cost.
- **Excluded Gemini candidates** by filtering on `provider != LLMProvider.GEMINI`.
- **Cost calculation** uses `MODEL_PRICING` from `analyze_results.py` — same pricing source used by the evaluation analysis.
- **Concurrent generation** — all candidates for a given movie fire simultaneously via `asyncio.gather()`, spreading rate-limit pressure across providers. Movies processed sequentially.

## Add JSON output instruction to branch-specific plot_events prompts
Files: `movie_ingestion/metadata_generation/prompts/plot_events.py` | Added `JSON with a single field: plot_summary.` to OUTPUT section of both SYSTEM_PROMPT_SYNOPSIS and SYSTEM_PROMPT_SYNTHESIS. Alibaba/Qwen requires "json" in messages when using `json_object` response format; the legacy SYSTEM_PROMPT already had this but the ADR-033 branch prompts didn't.

## Fix plot_events provider/model to gpt-5-mini
Files: `movie_ingestion/metadata_generation/generators/plot_events.py`

### Intent
Lock `generate_plot_events()` to OpenAI gpt-5-mini with `{"reasoning_effort": "minimal", "verbosity": "low"}` instead of accepting caller-specified provider/model/kwargs.

### Key Decisions
- **Model choice based on 21-movie evaluation** across 6 candidates (gpt-5-mini, gpt-5-nano, gpt-5.4-nano, qwen3.5-flash, gpt-oss-120b, llama-4-scout). gpt-5-mini scored 4.93/5.0 overall with 4.86 groundedness — near-zero errors. Next-best (qwen3.5-flash at $10.98) scored 4.56 with consistent small inference leaps. With 50% batch pricing, the premium is ~$17 total for meaningfully higher reliability on the pipeline's most critical field.
- **Removed provider/model/kwargs params** from `generate_plot_events()`. Production caller (`wave1_runner.py`) already used no-args form. Playground notebook cell 4 passes explicit args for multi-candidate evaluation — that's evaluation-specific code, not production.
- **Module-level constants** `_PROVIDER`, `_MODEL`, `_MODEL_KWARGS` keep the config visible and greppable.

## Simplify plot_events generation based on 42-movie evaluation
Files: `movie_ingestion/metadata_generation/generators/plot_events.py`, `movie_ingestion/metadata_generation/prompts/plot_events.py`, `movie_ingestion/metadata_generation/schemas.py`, `movie_ingestion/metadata_generation/evaluations/plot_events.py`, `movie_ingestion/metadata_generation/evaluations/analyze_results.py`

### Intent
Reduce hallucination and simplify the plot_events output based on evaluation of 42 movies across 7 size-based buckets.

### Key Decisions
- **Raised MIN_SYNOPSIS_CHARS from 1000 to 2500**: The condensation path hallucinated at 67% rate with ~1K char synopses (model fills gaps from training knowledge). At 4K+ synopses: 0% hallucination. 2500 is the threshold where synopses are consistently detailed enough for faithful condensation. Synopses below this are demoted into the summaries list and routed through the synthesis path, which has 0% hallucination at all input sizes.
- **Removed plot_keywords from input**: Evaluation showed keywords act as hallucination springboards (e.g., "intersex" keyword in Bol triggered fabrication of entire character arcs) and get incorrectly treated as plot events (e.g., "Oreos" in Drunks became a narrative detail). The overview already provides high-level framing.
- **Removed setting and major_characters from output**: Setting was redundant (already in plot_summary, exceeded ≤10 word constraint in 83% of results). Structured characters with motivations/roles added analytical burden to a consolidation task — character names appear naturally in plot_summary, and motivation analysis fits better in downstream plot_analysis. Output is now just plot_summary.
- **implementation/classes/schemas.py left unchanged** (reference only for future pipeline work).
- **Eligibility check unchanged**: `_MIN_PLOT_TEXT_CHARS = 600` in pre_consolidation.py stays — it gates on whether there's any substantial plot text, not on which branch to use.

### Testing Notes
- Unit tests referencing MajorCharacter, setting, or major_characters will need updating.
- Evaluation DB tables have changed schema — existing eval DBs need to be recreated (clean slate confirmed).
- Notebook cell 4 will produce simplified output (just plot_summary in result.model_dump()).
