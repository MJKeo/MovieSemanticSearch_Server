# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Full search capabilities catalog
Files: search_improvement_planning/full_search_capabilities.md | Comprehensive inventory of all data sources available for search (Postgres tables/columns, Qdrant vector spaces/payload, Redis, lexical schema, tracker DB unpromoted fields), organized by storage location with search utility notes for each. Cross-referenced from v2_data_architecture.md, codebase schemas, and other planning docs.

## V2 finalized search proposal and planning doc updates
Files: search_improvement_planning/finalized_search_proposal.md, search_improvement_planning/open_questions.md, search_improvement_planning/types_of_searches.md
Why: Captured all finalized decisions from design conversation into the official V2 proposal document.
Approach: finalized_search_proposal.md contains the full three-step pipeline architecture (query understanding → per-source search planning → execution & assembly), including semantic dealbreaker demotion, exclusion handling via elbow-threshold penalties, pure-vibe flow, quality prior as separate dimension, and gradient metadata scoring. open_questions.md updated with 4 new V2 pipeline questions (elbow detection method, multi-interpretation triggers, semantic demotion display, exclusion query formulation). types_of_searches.md updated with 3 new V2 edge case categories (#15 pure-vibe, #16 semantic exclusion on non-tagged attributes, #17 dealbreaker demotion).

## Search planning doc reversals and rationale alignment
Files: search_improvement_planning/finalized_search_proposal.md, search_improvement_planning/open_questions.md, search_improvement_planning/new_system_brainstorm.md, search_improvement_planning/types_of_searches.md
Why: The search design discussion reversed several earlier assumptions, and the planning docs needed to be brought back into alignment without introducing any new product decisions.
Approach: Updated the finalized proposal to add major-flow routing before standard decomposition, defend per-source step-2 LLMs as schema translators rather than re-interpreters, add `is_primary_preference` as the only preference-strength mechanism, split quality from notability/mainstreamness conceptually, and change semantic exclusions from effective removal to calibrated penalty-only behavior. Updated older planning docs to remove contradictions on boolean/group logic, preference weighting, similarity-flow routing, trending candidate injection, and quality-vs-discovery framing. Moved unresolved details that emerged from these reversals into open_questions.md instead of finalizing them prematurely.
Design context: Based on the current V2 planning set in search_improvement_planning/ and the latest design conversation clarifying that V1 should favor simpler tiering behavior over richer clause logic, and that "hidden gems"/"underrated" are not the same as inverted quality.
Testing notes: Verified by diff/grep that the finalized proposal no longer claims hidden gems/underrated are inverted quality, no longer frames semantic exclusions as effective removal, now documents major-flow routing and `is_primary_preference`, and that the supporting brainstorming/query-type docs no longer contradict those decisions.
