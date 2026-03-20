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
