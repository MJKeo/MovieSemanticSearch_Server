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
