# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Docs staleness fixes from /run-docs-auditor-agent
Files: docs/conventions.md, docs/PROJECT.md, docs/modules/{classes,db,llms,schemas,search_v2}.md, search_v2/step_3.py
Why: The docs-auditor run surfaced 16 staleness issues spanning factually-wrong invariants, stale architecture descriptions, and missing Key Files entries. Fixed all in one pass.
Approach:
- conventions.md + classes.md: corrected watch-provider key encoding from `<< 2` to `<< 4` (matches helpers.py:201).
- schemas.md: removed claim that `Movie` has `concept_tags_run_2_metadata` field; clarified `concept_tag_ids()` reads only the merged `concept_tags_metadata`; updated `EndpointRoute` count from 9 to 11 (added NEUTRAL_SEED, CHRONOLOGICAL); reworded `ActionRole` from "deleted" to "superseded (retained for legacy reranking)"; marked `flow_routing.py` as dead code; added `chronological_translation.py` Key Files entry.
- search_v2.md: updated Step 3 config to `thinking_level="minimal"` + temperature 0.15 (matched the runtime dict, not the prior stale comment); removed "pending wiring" claim for `chronological_query_execution.py` (it is wired via endpoint_executors.py); added `streaming_orchestrator.py` Key Files entry as the production API entry point.
- llms.md: corrected generator count from 10 to 12; updated schemas path from `movie_ingestion/metadata_generation/schemas.py` to `schemas/metadata.py`.
- db.md: added `chronological_scoring.py` Key Files entry.
- PROJECT.md: Stage 5 description now mentions the two hard gates (title-type, missing-text); test count corrected from 76 to 77.
- search_v2/step_3.py: fixed two internal comments (lines 49 and 1350) that said `thinking_level="low"` to match the actual `_MODEL_KWARGS` value of `"minimal"`. Runtime behavior unchanged — only stale comments corrected.
Design context: ADR-090 (Step 3 loose5 changes), ADR-094 (concept_tags three-run batch pipeline). No new ADRs; all changes are doc/comment alignment, not behavioral.
