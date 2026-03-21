# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Add Codex bootstrap file
Files: AGENTS.md
Why: Port the repo's Claude-oriented startup context into Codex's native repo instruction file so sessions can pick up the same documentation map and guardrails automatically.
Approach: Created a near-1:1 `AGENTS.md` from `CLAUDE.md`, preserving the structured docs system, update permissions, commands, architecture summary, and coding guidance while swapping Claude-specific framing for Codex-native wording.

## Correct Stage 6 and provider docs
Files: AGENTS.md, CLAUDE.md
Why: The top-level bootstrap docs still described the legacy metadata-generation location and an outdated Kimi-centric provider setup.
Approach: Updated both files to point Stage 6 at `movie_ingestion/metadata_generation/`, describe the current batch pipeline layout, and replace the old provider note with the current shared multi-provider LLM router plus provider-specific structured-output handling.

## Restructure evaluation pipeline — remove references, switch to Opus 4.6 with caching
Files: `implementation/llms/generic_methods.py`, `movie_ingestion/metadata_generation/evaluations/plot_events.py`, `movie_ingestion/metadata_generation/evaluations/run_evaluations_pipeline.py`, `docs/modules/ingestion.md`, `docs/conventions_draft.md`

### Intent
Remove reference-based evaluation (Phase 0) from the plot_events evaluation pipeline. Research showed rubric matters ~2.7x more than reference for human judgment alignment on subjective metadata tasks (Yamauchi et al., arXiv:2506.13639). Switch the judge from GPT-5.4/WHAM to Claude Opus 4.6/Anthropic with prompt caching for cost savings.

### Key Decisions
- **Reference removal**: References added anchoring bias with minimal quality benefit for this use case (subjective metadata extraction from broad movie data). Removed `generate_reference_responses()`, `_CREATE_REFERENCES_TABLE`, and all reference loading/passing in `_evaluate_one()`.
- **Source data over generation prompt**: Judge now sees raw movie fields (SOURCE DATA) instead of the generation prompt's instructions. The candidate's `build_plot_events_user_prompt()` output already contains exactly the labeled raw data fields — reused directly.
- **Rubric reframing**: "THE GENERATION PROMPT instructs:" → "A HIGH-QUALITY OUTPUT should:" throughout. Quality criteria are now self-contained, not dependent on knowing what the candidate was told.
- **Prompt caching**: Added `cache_control` kwarg to `generate_anthropic_response_async()`. When True, wraps system, user, and tool content in cache_control blocks. Staggered judge runs (run 1 alone, then runs 2-3 in parallel) ensure cache is populated before subsequent reads.
- **Judge model switch**: GPT-5.4/WHAM → Claude Opus 4.6/Anthropic. Removed WHAM auth acquisition and all WHAM-specific kwargs.

### Planning Context
Plan documented at `.claude/plans/polymorphic-mixing-gizmo.md`. All 6 implementation steps complete. Conventions update staged in `docs/conventions_draft.md` (cannot autonomously modify `docs/conventions.md`).

### Testing Notes
- `unit_tests/test_eval_plot_events.py` imports removed `generate_reference_responses` — will fail until tests are updated separately.
- Other unit tests unaffected. Run `pytest unit_tests/ --ignore=unit_tests/test_eval_plot_events.py` to verify.

## Fold Claude rule files into Codex bootstrap instructions
Files: AGENTS.md
Why: The repo's Codex bootstrap file referenced Claude-oriented rules indirectly, which made core behavior split across multiple places and less reliable to ingest at session start.
Approach: Rewrote `AGENTS.md` into a compact Codex-native instruction doc that preserves the project map while inlining the actionable rules from `.claude/rules/`, including startup docs, decision hygiene, opinion-giving, documentation permissions, context tracking, coding standards, and test boundaries.

## Convert legacy Claude commands into project-local Codex skills
Files: `.codex/skills/audit-personal-preferences/`, `.codex/skills/create-unit-test-plan/`, `.codex/skills/extract-finalized-decisions/`, `.codex/skills/force-diff-context-update/`, `.codex/skills/implement-unit-test-plan/`, `.codex/skills/ingest-spec-to-memory/`, `.codex/skills/initiate-spec-understanding-conversation/`, `.codex/skills/new-metadata-evaluation/`, `.codex/skills/review-code/`, `.codex/skills/review-workflow-suggestions/`, `.codex/skills/run-docs-auditor-agent/`, `.codex/skills/run-docs-maintainer-agent/`, `.codex/skills/run-test-planner-agent/`, `.codex/skills/safe-clear/`, `.codex/skills/save-todo/`, `.codex/skills/solidify-draft-conventions/`
Why: The repo had a mature library of Claude Code commands under `.claude/commands/`, but no project-local Codex skill equivalents. Converting them preserves those workflows in Codex's native skill system and keeps the project-specific automation discoverable in one place.
Approach: Created one skill folder per legacy command under `.codex/skills/`, each with a concise `SKILL.md`, `agents/openai.yaml` UI metadata, and a `references/original-command.md` copy of the source Claude command so the exact workflow contract remains accessible. For the three commands that delegated to Claude subagents, also copied the corresponding legacy agent prompt into `references/legacy-agent.md` so the converted skill retains the deeper analysis behavior without bloating the main skill file.

## Convert legacy Claude subagents into project-local Codex agents
Files: `.codex/agents/docs-auditor.toml`, `.codex/agents/docs-maintainer.toml`, `.codex/agents/test-planner.toml`
Why: The repo's legacy automation included three reusable Claude subagents under `.claude/agents/`, and the first Codex conversion pass used Markdown agent files based on an assumption rather than current Codex docs. Official OpenAI Codex documentation now confirms that project-scoped custom agents belong in `.codex/agents/` as standalone TOML files, not Markdown manifests.
Approach: Replaced the incorrect Markdown agent files with three standalone TOML agent configs using the documented Codex schema: required `name`, `description`, and `developer_instructions`, plus optional `model`, `model_reasoning_effort`, and `sandbox_mode`. Kept each agent narrow and role-specific, matching the official guidance that custom agents should be specialized and opinionated.

## Optimize plot_events evaluation pipeline — reduce cost and add retry
Files: `movie_ingestion/metadata_generation/evaluations/plot_events.py`, `implementation/llms/generic_methods.py`

### Intent
Reduce judge call costs and latency while adding resilience to rate limiting.

### Key Decisions
- **Disable thinking on Opus 4.6 judge**: Added `"thinking": {"type": "disabled"}` to judge kwargs. Anthropic API defaults to disabled when omitted, but explicit is clearer.
- **Caveman-speak reasoning to cut output tokens**: Judge reasoning fields were unconstrained strings producing ~2K output tokens. Added `Field(description=...)` constraints (one sentence, max 30 words, caveman-speak) to the schema and a REASONING FORMAT section to the system prompt. Caveman-speak eliminates articles and filler, compressing reasoning significantly.
- **Reduce judge runs from 3 to 2**: Statistical analysis showed the 3rd run provides only 18% SE reduction (vs 29% for run 2). On a 4-point discrete scale with a strong rubric, inter-run variance is low enough that 2 runs maintain ranking stability. Default `judge_runs` changed from 3 to 2.
- **Sequential judge execution**: Replaced stagger-then-parallel pattern with a simple sequential loop. Both runs now benefit from prompt caching (run 1 primes, run 2 reads cached).
- **429 retry with 30s sleep**: Added `anthropic.RateLimitError` re-raise in `generate_anthropic_response_async` (before the catch-all `ValueError` wrapper), then per-call retry loop in the judge execution. On 429, sleeps 30s and retries indefinitely until success.

### Testing Notes
- `unit_tests/test_eval_plot_events.py` already broken from prior session (imports removed function). Unaffected by these changes.
- Retry path testable by temporarily saturating rate limits or mocking.

## ADR-033: Plot events cost optimization design

Files: `docs/decisions/ADR-033-plot-events-cost-optimization.md`

### Intent
Design and document a cost-optimized plot_events generation pipeline that handles synopsis and non-synopsis movies differently, eliminates redundant generation for synopsis movies, and removes plot from source_of_inspiration.

### Key Decisions
- **Two-branch plot_events**: Option A (has synopsis) focuses on condensation; Option B (no synopsis) focuses on synthesis from summaries/overview/keywords.
- **Preliminary distillation dropped**: Initially planned a gpt-5-nano preprocessing pass to condense long synopses, but testing showed the model cut too aggressively and introduced hallucinations. Long synopses will be handled differently (e.g., truncation or downstream prompt limits).
- **source_of_inspiration drops plot_synopsis entirely**: Saves ~83.6M downstream input tokens. Keywords + reviews + title are sufficient.
- **Proportional output length**: No fixed token targets — prompt guidelines on what to keep/cut, length scales with story complexity.
- **5K token soft cap on Option B output**: Prompt guidance + max_tokens safety net for rare long-summary outliers.
- **No dual summary approach**: Considered generating separate broad/detailed summaries for different downstream consumers. Cost savings (~$1-2) didn't justify the complexity.
- **Downstream routing unchanged**: All Wave 2 generators still receive plot_synopsis from plot_events_output.plot_summary uniformly. All complexity lives in the generation step.

### Planning Context
Extended cost analysis session examining population data (22,894 synopsis movies, 51,328 summary-only, 37,547 overview-only). Evaluated 3 main options plus sub-variants. Hybrid approach (two summaries for synopsis movies only) was $53.30 vs $44.57 for broad-only, but added complexity for $8.73. User chose simpler two-branch design. ADR includes full implementation guide with file-by-file changes and step-by-step ordering.

## Delete synopsis distillation script
Files: `movie_ingestion/imdb_scraping/distill_long_synopses.py` (deleted), `docs/TODO.md`, `DIFF_CONTEXT.md`
Why: Testing gpt-5-nano distillation on Star Wars showed 76% compression (far beyond target), introduced hallucinations (wrong character relationships, fabricated events), and lost detail on iconic searchable scenes. User decided to skip the approach entirely rather than iterate further.
Approach: Deleted the script, removed the related TODO entry (cost estimate correction), updated the ADR-033 DIFF_CONTEXT entry to note distillation was dropped.

## Update plot_summaries and plot_keywords extraction logic
Files: `movie_ingestion/imdb_scraping/parsers.py`

### Intent
Improve data coverage for two IMDB fields that were under-populated due to overly aggressive filtering.

### Key Decisions
- **plot_summaries always extracted**: Previously discarded when synopses existed (priority logic returned empty list). Now both are extracted independently — synopses and summaries serve different downstream purposes.
- **Keyword floor raised from 5 to 10**: More keywords improve embedding quality for semantic search. Introduced `_MIN_PLOT_KEYWORDS` and `_MAX_PLOT_KEYWORDS` constants to avoid hard-coding.

### Pending
- Backfill script still needed to re-parse 425K cached GraphQL responses (`imdb_graphql.zip`) and update existing `imdb_data` rows with the new logic. Should target only `plot_summaries` and `plot_keywords` columns.

## Remove stale init_db migration that reverted imdb_quality_passed
Files: `movie_ingestion/tracker.py` | Removed the ADR-020 migration (`UPDATE ... SET status = 'imdb_quality_calculated' WHERE status = 'imdb_quality_passed'`) that fired on every `init_db()` call, reverting movies that had already passed the quality filter. Migration was one-time but ran unconditionally.

## Add debug prints and raise max_tokens in synopsis distillation
Files: `movie_ingestion/imdb_scraping/distill_long_synopses.py` | Added finish_reason, token counts, and response length debug prints to `_distill_one()`. Raised `_MAX_TOKENS` from 3,000 to 4,000 after Star Wars hit the output cap (finish_reason='length').

## Implement ADR-033: two-branch plot_events generator and source_of_inspiration plot removal
Files: `movie_ingestion/metadata_generation/prompts/plot_events.py`, `movie_ingestion/metadata_generation/generators/plot_events.py`, `movie_ingestion/metadata_generation/generators/source_of_inspiration.py`, `movie_ingestion/metadata_generation/prompts/source_of_inspiration.py`, `movie_ingestion/metadata_generation/pre_consolidation.py`, `movie_ingestion/metadata_generation/evaluations/plot_events.py`, `movie_ingestion/metadata_generation/evaluations/run_evaluations_pipeline.py`

### Intent
Implement the core generation changes from ADR-033: branch plot_events into condensation (synopsis) and synthesis (non-synopsis) modes, each with a tailored system prompt and input set, and remove plot_synopsis from source_of_inspiration to save ~83.6M input tokens.

### Key Decisions
- **Two new system prompts** (`SYSTEM_PROMPT_SYNOPSIS`, `SYSTEM_PROMPT_SYNTHESIS`): Each references only its relevant inputs. Synopsis prompt frames task as condensation; synthesis prompt frames as unification of partial sources. Both include "under 4K tokens" soft cap and proportional-length guidance.
- **`build_plot_events_user_prompt` returns `(user_prompt, system_prompt)`**: All branching logic contained in one function. Synopsis branch sends synopsis + overview + keywords; synthesis branch sends summaries + overview + keywords.
- **`max_tokens=5000` hard cap**: Added to `generate_plot_events` default kwargs as safety net alongside the prompt's 4K soft guidance. Default provider switched from Gemini to OpenAI gpt-5-mini because Gemini uses `max_output_tokens` (not `max_tokens`) and the generic router doesn't normalize this parameter across providers.
- **source_of_inspiration fully decoupled from plot**: Removed `plot_synopsis` parameter from generator, builder, prompt `_PREAMBLE`, skip check, and call site.
- **Evaluation `--branch` flag**: `run_evaluations_pipeline.py` accepts `--branch synopsis|synthesis` to filter test corpus by synopsis presence, enabling separate evaluation of each branch.

### Testing Notes
- Unit tests for `source_of_inspiration` and `plot_events` will need updates for changed signatures.
- Evaluation pipeline can be tested with `--branch synopsis` and `--branch synthesis` flags.
- Legacy prompts (`SYSTEM_PROMPT`, `SYSTEM_PROMPT_SHORT`) kept for backwards compat with existing eval candidates.

## Allow knowledge supplementation in synthesis branch prompt
Files: `movie_ingestion/metadata_generation/prompts/plot_events.py`, `movie_ingestion/metadata_generation/generators/plot_events.py`
Why: Testing showed the synthesis branch (no synopsis) was supplementing from training knowledge anyway, producing accurate but "unauthorized" details. For sparse-input movies this improves embedding quality. The previous conditional token-limit logic (`SYNTHESIS_TOKEN_LIMIT_RULE`) was unnecessary complexity.
Approach: Replaced the strict no-hallucination rule in `SYSTEM_PROMPT_SYNTHESIS` with a "may supplement" rule that allows model knowledge but prohibits contradicting provided data and warns against building entire plots from memory for very sparse inputs. Put the 4K token limit back into the base prompt unconditionally. Removed `SYNTHESIS_TOKEN_LIMIT_RULE` constant and the conditional logic in the generator that appended it.

## Add synopsis quality gate — demote thin synopses to synthesis branch
Files: `movie_ingestion/metadata_generation/generators/plot_events.py`, `docs/TODO.md`
Why: Evaluation revealed that short synopses (e.g. Tube Tales at 329 chars — a review blurb, not a plot recount) were routed to the condensation branch, which prohibits model knowledge. The model hallucinated 9 fabricated vignettes to fill structured output fields. Real corpus analysis (30K synopsis movies) showed entries under ~1,000 chars are consistently non-plot text.
Approach: Added `MIN_SYNOPSIS_CHARS = 1000` threshold. When a synopsis exists but falls below the threshold, it's demoted into the summaries list (prepended as first entry) and routed to the synthesis branch instead. The synthesis branch permits careful knowledge supplementation, producing strictly better outcomes than silent hallucination under a no-knowledge constraint. Updated the existing TODO entry about embedding pipeline synopsis usage to note that the same threshold must apply there.

## Move behavioral instructions from PlotEventsOutput schema into branch-specific prompts
Files: `movie_ingestion/metadata_generation/schemas.py`, `movie_ingestion/metadata_generation/prompts/plot_events.py`
Why: Evaluation of synthesis-branch results showed gpt-5-mini fabricating ~1000-token plots with invented character names from single-sentence overviews. Root cause: the structured output field descriptions ("Detailed chronological, spoiler-containing plot summary preserving character names and locations") created a strong competing signal that overrode the system prompt's softer length/fabrication guidelines. The model resolved the tension by satisfying the schema's demand for detail.
Approach: Stripped `PlotEventsOutput` and `MajorCharacter` field descriptions to minimal neutral labels ("Chronological plot summary.", "Character name.", etc.). Moved all behavioral instructions into the FIELDS sections of each branch-specific system prompt, where they can be tailored per task. The synopsis prompt's FIELDS section encourages preserving detail from the rich source. The synthesis prompt's FIELDS section has explicit anti-fabrication rules: never invent character names (use descriptive references), never invent plot beats, keep output proportional to input richness, and a concrete anchor ("3-5 sentence summary" for sparse input). Added a CRITICAL — OUTPUT LENGTH AND FABRICATION section to the synthesis prompt with the strongest constraints.
Design context: Continues ADR-033 iteration. Schema docstrings explain the minimal-description design decision.

## Refine setting and major_characters field instructions in plot_events prompts
Files: `movie_ingestion/metadata_generation/prompts/plot_events.py`
Why: Evaluation of 6 synopsis-branch generations revealed two patterns: (1) setting field drifted into genre labels and scene-location inventories instead of geographic/temporal context, (2) model listed characters without established goals, filling motivation with meta-narrative descriptions of plot function.
Approach: Tightened both branch prompts (SYSTEM_PROMPT_SYNOPSIS and SYSTEM_PROMPT_SYNTHESIS) with principle-based constraints rather than specific failure examples. Setting now says "Only where and when — nothing else." Major characters now requires characters to "actively drive plot decisions" with goals "clear from the input," explicitly allowing an empty list for sparse input.

## Rewrite SYSTEM_PROMPT_SYNTHESIS to eliminate hallucination from sparse input
Files: `movie_ingestion/metadata_generation/prompts/plot_events.py`

### Intent
Evaluation of 5 synthesis-branch generations (no synopsis) revealed severe hallucination when input was sparse: invented character names (Kalakalappu 2), fabricated plot beats (Meteor, Kalakalappu 2), and output length disproportionate to input richness. Root cause was three interacting prompt design issues.

### Key Decisions
- **Removed all parametric knowledge permission**: The previous "You may supplement from your own knowledge" rule was the root enabler of hallucination. The model cannot reliably distinguish recall from fabrication, so the permission undermined every subsequent guardrail. Replaced with "You have no knowledge of any film" — a fiction, but one that removes the self-assessment problem entirely.
- **Reframed task from narrative creation to text consolidation**: "Synthesize into a unified, coherent plot_summary" pressured the model toward narrative completeness, which requires gap-filling for sparse input. "Consolidate into a single organized account" frames the task as reorganizing existing text, removing the pressure to generate content.
- **Labeled input types by what they're NOT good for**: Overview is "often vague — do not treat as plot detail." Keywords are "context clues, not plot events." Prevents the model from treating every input as a source of plot events to elaborate on.
- **Traceability as internal check, not output format**: "Before including any detail, internally verify it appears in the input" with explicit "Do not cite sources or explain where details came from in your output" to prevent the model from externalizing the verification as inline parenthetical citations.
- **Proportionality as natural consequence**: Rather than a special rule to follow, proportionality falls out naturally from the consolidation framing — if there's nothing to consolidate, the output is short.

### Planning Context
Analysis followed a diagnostic-before-prescriptive pattern: first evaluate results, then identify root causes in the prompt structure, then redesign. The three root causes were identified as: (1) parametric knowledge permission undermining guardrails, (2) task framing pressuring narrative completeness, (3) anti-fabrication positioned as exception rather than primary instruction.

## Add imdb_title_type to IMDB scraping pipeline
Files: `movie_ingestion/imdb_scraping/http_client.py`, `movie_ingestion/imdb_scraping/models.py`, `movie_ingestion/imdb_scraping/parsers.py`, `movie_ingestion/tracker.py`

### Intent
IMDB's `titleType.id` field deterministically identifies content kind (e.g. "movie", "videoGame", "tvSeries"). Some TMDB entries are mislabeled as movies but are actually video games or other content types. This field enables future filtering.

### Key Decisions
- **Forward path**: Added `titleType { id }` to the existing GraphQL query, `imdb_title_type` field to `IMDBScrapedMovie`, extraction in the parser, and a new column in `imdb_data`. New scrapes get the field automatically.
- **Backfill script deleted**: A standalone backfill script was created then removed — the title type fetch was already in the main scraping flow (GraphQL query, parser, model), so the backfill will happen naturally during the next full re-scrape.
- **No filtering applied**: Title type is stored as-is for now. Filtering decisions deferred to a later step.
- **ALTER TABLE migration**: Added to `init_db()` migrations for existing databases.

## Backfill missing plot_summaries and featured_reviews
Files: `movie_ingestion/imdb_scraping/backfill_plot_and_reviews.py` (deleted)
Why: Audit of cached GraphQL responses found 41,685 `imdb_quality_passed` movies with NULL `plot_summaries` and 192,807 with empty `featured_reviews` (`"[]"`). Some of these may have data available on IMDB that wasn't captured or was null at scrape time (confirmed for 4 movies with null review text bodies).
Approach: Standalone script that queried affected movies, re-fetched only `plots` and `reviews` via a minimal GraphQL query through the proxy system. Script has been deleted after use — backfill is complete.

## Fix operator precedence bug in featured review edge parsing
Files: `movie_ingestion/imdb_scraping/parsers.py` (lines 112, 173, 430) — also was in `backfill_plot_and_reviews.py` (now deleted)
Why: `edge.get("node") or {} if isinstance(edge, dict) else {}` parsed as `edge.get("node") or ({} if isinstance(...) else {})` — the isinstance guard never fires when `edge` is not a dict because `.get()` raises AttributeError first. Fixed by adding parentheses: `(edge.get("node") or {}) if isinstance(edge, dict) else {}`. Latent bug (GraphQL edges are always dicts in practice) but now correct.

## Add title-type and missing-text-data gates to IMDB quality scorer
Files: `movie_ingestion/imdb_quality_scoring/imdb_quality_scorer.py`

### Intent
Prevent non-movie content and text-poor movies from passing the quality filter and entering the search index.

### Key Decisions
- **Title-type gate**: Added `ALLOWED_TITLE_TYPES = {"movie", "tvMovie", "short", "video"}` and an early return of 0.0 in `compute_imdb_quality_score()` for any movie whose `imdb_title_type` is not in the set (including None). This catches tvSeries, tvEpisode, videoGame, etc. that slipped through TMDB's movie classification.
- **Missing-text gate**: Added a second early return of 0.0 when a movie has no plot_summaries, no synopses, and no featured_reviews. Without any text sources, LLM metadata generation cannot produce meaningful output for most vector spaces.
- **Data cleanup**: Ran one-off scripts to retroactively filter 2,236 movies with invalid title types and 385 movies missing all text sources from `imdb_quality_passed` status, logging them to `filter_log` with stage `imdb_quality_funnel`.
