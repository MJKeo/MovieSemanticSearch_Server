# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Revamp reception metadata generator output structure

Files: movie_ingestion/metadata_generation/schemas.py, prompts/reception.py, generators/reception.py

### Intent
Decompose the monolithic `review_insights_brief` field into structured observation
fields so each downstream Wave 2 generator receives targeted signal per dimension
instead of a mixed-dimension paragraph.

### Key Decisions
- **Extraction-first field ordering**: observation fields come before evaluative fields
  in the schema so structured output models extract concrete observations before
  synthesizing opinions. Improves cheap-model output quality at low reasoning effort.
- **3 observation fields + source_material_hint**: thematic_observations,
  emotional_observations, craft_observations (each nullable, 1-4 sentences) plus
  source_material_hint (nullable short phrase). Grouped by downstream consumer
  affinity rather than 1:1 with generators.
- **Tag cap 4→6 with balanced precision-recall**: new prompt framing treats missing
  a well-supported tag as equally bad as fabricating an unsupported one.
- **Added genres + overview as inputs**: cheap contextual grounding (~60 extra
  input tokens) helps the model disambiguate reviewer language.
- **Backward-compat property**: review_insights_brief is now a @property that
  concatenates observation fields. Pre-consolidation and Wave 2 skip conditions
  continue to work unchanged.
- **Field renames**: new_reception_summary → reception_summary,
  praise_attributes → praised_qualities, complaint_attributes → criticized_qualities.

### Planning Context
Downstream consumption (Wave 2 generator prompt builders, pre_consolidation skip
conditions) and embedding (vectorize.py, ReceptionMetadata search-side schema) are
intentionally NOT updated in this change — deferred to a follow-up.

### Testing Notes
- test_reception_generator.py will fail on field name changes and schema structure
- Backward-compat property needs verification: instantiate ReceptionOutput with
  new fields and confirm .review_insights_brief returns expected concatenation
- pre_consolidation.py line 368 accesses .review_insights_brief — verify property works

## Rewrite reception system prompt from scratch

Files: movie_ingestion/metadata_generation/prompts/reception.py

### Intent
Improve prompt quality for gpt-5-mini at low reasoning effort. The original
prompt had structural issues that hurt cheap-model output: flat hierarchy mixing
global/per-field rules, contradictory style guidance ("not complete sentences"
vs "2-3 sentences"), tag guidance duplicated across two fields, critical global
rules buried at the end, and observation fields under-illustrated with examples.

### Key Decisions
- **Hierarchical structure**: RULES → INPUTS → EXTRACTION FIELDS → SYNTHESIS FIELDS
  replaces flat mixed-scope sections. Rules that apply everywhere are front-loaded.
- **Examples on all 3 observation fields**: each gets a good/bad pair illustrating
  the descriptive-vs-evaluative distinction, not just emotional_observations.
- **Tag guidance shared once**: stated once with shared rules, praised/criticized
  fields only specify "best" vs "worst" perspective.
- **~~Compact wording only in synthesis fields~~** SUPERSEDED: originally extraction
  fields used normal prose; now both zones use compact wording after cost analysis
  showed downstream models handle telegraphic note-form without quality loss.
  See "Reception cost analysis and optimization" entry below.
- **No cross-field deduplication rule**: user rejected "keep dimensions distinct"
  as unnecessary complexity — duplication across observation fields is fine.
- **Overview qualified as non-evidence**: "context only — not evidence for observations"
  prevents model from treating TMDB marketing text as reviewer observations.

## Reception evaluation buckets and candidate parameter research

Files: ingestion_data/reception_eval_buckets.json, movie_ingestion/metadata_generation/metadata_generation_playground.ipynb (Cell 2)

### Intent
Prepare for reception generation quality evaluation by (1) creating input-richness
buckets to identify quality thresholds and (2) researching optimal parameters for
each playground candidate model.

### Key Decisions
- **6 evaluation buckets by combined top-5 review length**: ultra-thin (0-1K),
  very-thin (1K-2.5K), thin (2.5K-5K), moderate (5K-7.5K), rich (7.5K-10.5K),
  very-rich (10.5K+). More granularity at the low end where the quality threshold
  likely lives. 6 genre-diverse movies sampled per bucket.
- **temperature 0.2 over 0.0**: developer reports of temp-0 causing incomplete
  structured output; clinical extraction study found 0.0-1.5 all comparable; 0.2
  is the safer starting point with no quality penalty.
- **reasoning_effort "low" over "minimal" (OpenAI)**: task involves cross-review
  synthesis and interpretation, not simple classification. OpenAI cookbook maps
  "minimal" to simple classification, "low" to data extraction with interpretation.
- **Gemini thinking_budget 1024 over 0**: budget 0 is analogous to OpenAI's "none",
  not "low". Variable-format review interpretation needs some reasoning room. Google
  tiered guidance: zero for standardized extraction, medium for variable-format docs.
- **Qwen enable_thinking: False is a hard requirement**: DashScope structured output
  does not work in thinking mode. Not a performance choice.
- **Groq gpt-oss-120b reasoning_format → include_reasoning**: reasoning_format is
  not supported for gpt-oss models per Groq docs. Fixed to include_reasoning: False.
- **Gemini Flash Lite thinking_config removed**: model doesn't support thinking.

## Updated playground notebook Cell 5 for reception bucket evaluation

Files: movie_ingestion/metadata_generation/metadata_generation_playground.ipynb (Cell 5)
Why: Needed to run reception generation across all 6 eval buckets (from reception_eval_buckets.json) with first 4 candidates to evaluate quality across input-richness tiers.
Approach: Followed the same pattern as the plot_events eval cell (Cell 4) — loads bucket IDs from JSON, builds MovieInputData from tracker DB, runs all candidates concurrently per movie, saves per-movie JSON as `reception_{tmdb_id}.json`.

## Updated reception review truncation and eligibility threshold

Files: movie_ingestion/metadata_generation/generators/reception.py, movie_ingestion/metadata_generation/pre_consolidation.py

### Intent
Improve review selection diversity and raise the eligibility floor based on evaluation findings across 36 movies in 6 sparsity buckets.

### Key Decisions
- **_MAX_REVIEW_CHARS 5K→10K**: moderate bucket (5-7.5K review chars) produced excellent multi-faceted output; rich bucket didn't add meaningful quality. 10K covers moderate+ comfortably.
- **_MAX_REVIEW_COUNT removed entirely**: character budget already prevents prompt bloat. Removing the count cap allows more diverse perspectives when individual reviews are short.
- **Ascending-length sort in _truncate_reviews**: shorter reviews are packed first to maximize perspective diversity before hitting the char budget. A single long review no longer crowds out 5 short ones.
- **Truncation fallback**: if all reviews exceed the budget individually, the shortest is truncated to fit — some signal is always better than skipping.
- **Eligibility threshold 25→400 chars**: evaluation showed movies with <400 combined review chars produce observations that mostly paraphrase the overview rather than adding genuine review signal. Only affects ~779 movies (0.7% of corpus), and those movies still have reception_summary/attributes available for Wave 2 fallback.

### Testing Notes
- test_reception_generator.py likely has tests for _truncate_reviews with old signature (max_count param removed)
- Ascending-sort behavior should be verified: given reviews of lengths [3000, 500, 200, 800], result should be [200, 500, 800, 3000] order with accumulation stopping at budget
- Edge case: single review of 15K chars → should return truncated version at 10K

## Clarify flat JSON output in reception system prompt
Files: movie_ingestion/metadata_generation/prompts/reception.py | Added "JSON" to output format and explicit "do NOT nest fields under zone keys" instruction to prevent Qwen 3.5 Flash from wrapping fields in EXTRACTION/SYNTHESIS wrapper objects. DashScope's structured output enforcement is weaker than OpenAI's constrained decoding, so the model was following the prompt's zone structure over the JSON schema.

## Reception cost analysis and optimization

Files: movie_ingestion/metadata_generation/evaluation_data/analyze_evaluations.py, movie_ingestion/metadata_generation/prompts/reception.py, movie_ingestion/metadata_generation/generators/reception.py

### Intent
Analyze reception generation costs across candidates and input-richness buckets,
then apply two optimizations to reduce cost without significant quality loss.

### Key Decisions
- **analyze_evaluations.py**: New script that reads all `*_evaluation.json` and
  `reception_*.json` files to produce a summary table of avg scores per axis per
  candidate, plus total cost across 36 eval movies. Used to compare candidates.
- **_MAX_REVIEW_CHARS 10K→6K**: Output token analysis showed model output saturates
  at ~2,000 input tokens (~750 tokens of review content). 6K chars provides ~2x
  headroom above saturation. Primarily compresses moderate/rich/very_rich buckets
  (~30K movies); the other ~79K are unaffected.
- **Compact observation wording in extraction fields**: Changed prompt guidance from
  "1-4 sentences" to "concise phrases, semicolon-separated" for the 3 observation
  fields. These are consumed by Wave 2 LLMs (not embedded), and telegraphic note-form
  preserves all semantic signal while reducing output tokens. Updated examples to
  match. **This reverses an earlier preference** (see personal_preferences.md
  "Compact wording for embedded output, normal prose for LLM-consumed output") —
  the user decided the downstream models can understand compact form just as well.
- **Cost estimates**: gpt-5-mini (low) drops from $108→$86 at batch pricing (20%
  reduction). gpt-5-mini (minimal) would be $62, kimi-k2.5 would be $111.

### Planning Context
- Prompt caching was initially proposed as a savings lever but is already active
  in the Batch API for GPT-5 family models — not an untapped optimization.
- Model tiering (cheaper models for thin buckets) was analyzed and rejected —
  the thin buckets represent too little cost ($9 combined) to justify the quality
  risk and implementation complexity.

## Reception evaluation round 2: 3-candidate comparison

Files: movie_ingestion/metadata_generation/evaluation_data/reception_*_evaluation.json (36 files)

### Intent
Evaluate 3 new reception candidates (gpt-5-mini-low, gpt-5-mini-minimal, kimi-k2.5-no-thinking)
across 36 movies using 5-axis grading (faithfulness, extraction_quality, synthesis_quality,
proportionality, downstream_utility). Identify winner and diagnose failure patterns.

### Key Findings
- **kimi-k2.5-no-thinking** strongest overall (4.29 avg vs 4.01/3.97 for GPT variants).
  Wins on 19/36 movies head-to-head. Never loses on extraction or proportionality.
- **GPT-5-mini variants** stronger on faithfulness (4.47-4.50 vs 4.14) but weaker on
  extraction depth, proportionality, and downstream utility by ~0.5 points.
- **Key gap: medium-richness input** — kimi leads by 0.45 overall on the middle tercile,
  where proportionality discipline matters most.
- **GPT-5-mini failure patterns**: over-extraction on thin input, generic topic-listing
  instead of capturing reviewer arguments, content-describing tags ("engaging debate")
  instead of craft-describing tags ("sharp dialogue"), source_material_hint misses.

## Created /evaluate-metadata-results and /estimate-generation-cost commands
Files: .claude/commands/evaluate-metadata-results.md, .claude/commands/estimate-generation-cost.md, docs/workflow_suggestions.md
Why: Two workflow suggestions implemented. /evaluate-metadata-results automates reading 36+ evaluation JSON files and producing a structured 7-section report (aggregate scores, per-bucket, per-axis, failure analysis, cross-candidate comparison, recommendations). /estimate-generation-cost projects per-candidate costs across the full corpus using eval token data + tracker DB movie counts per bucket, with what-if scenario modeling.

## Revised reception system prompt based on evaluation findings

Files: movie_ingestion/metadata_generation/prompts/reception.py

### Intent
Improve gpt-5-mini performance by addressing structural prompt weaknesses identified
in the 3-candidate evaluation, using principle-based improvements (not reactive fixes).

### Key Changes
- **Evidence-tracing rule**: replaced abstract "scale to input richness" with concrete
  "identify which reviewer statements support each field; if none, field must be null."
  Naturally scales — vacuously true on rich input, constraining on thin input.
- **Three-tier examples**: added "Shallow" tier between Good/Bad showing what GPT-5-mini
  currently produces (topic-listing), so the model sees it's not enough.
- **Content vs craft tag distinction**: explicit rule that tags describe execution not
  subject matter, with contrasting examples ("sharp dialogue" vs "engaging debate").
- **Tag count calibration**: replaced symmetric recall pressure with "tag count should
  reflect breadth of distinct qualities; a single review rarely supports more than 2-3."
- **source_material_hint evidence types**: listed what input evidence looks like (genre
  labels, reviewer references to originals, life stories), not just what output values
  look like.
- **Extraction vs synthesis boundary**: made explicit that extraction = technique/approach,
  synthesis = quality/reception. "Strong acting" is synthesis; "relies on physical expression"
  is extraction.

## Removed overview from reception inputs
Files: movie_ingestion/metadata_generation/generators/reception.py, movie_ingestion/metadata_generation/prompts/reception.py
Why: At minimal reasoning effort, the model treated TMDB overview as source material for extraction despite the "context only" qualifier. Evaluation across 36 movies showed overview was redundant with existing inputs (title, genres, reception_summary, attributes, reviews) and caused parametric knowledge leaking on thin-input movies.
Approach: Removed `overview` from the `build_user_prompt()` call in the generator and from the INPUTS section of the system prompt.

## Finalized reception generator LLM configuration
Files: movie_ingestion/metadata_generation/generators/reception.py
Why: Evaluation across 36 movies in 6 input-richness buckets confirmed gpt-5-mini with minimal reasoning matches or exceeds low-reasoning quality on the revised prompt while halving output token cost. No-overview variant eliminated remaining hallucination issues on thin-input movies.
Approach: Removed provider/model/kwargs parameters from `generate_reception()` signature (now accepts only `movie`). LLM config is fixed as module-level constants (`_PROVIDER`, `_MODEL`, `_MODEL_KWARGS`) matching the plot_events pattern. Downstream callers (notebooks, unit tests) that pass provider/model/kwargs will need updating.
