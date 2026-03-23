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
- **Compact wording only in synthesis fields**: extraction fields (consumed by Wave 2
  generators) use normal prose; synthesis fields (embedded) use compact wording.
  User explicitly corrected initial draft that applied compact wording everywhere.
- **No cross-field deduplication rule**: user rejected "keep dimensions distinct"
  as unnecessary complexity — duplication across observation fields is fine.
- **Overview qualified as non-evidence**: "context only — not evidence for observations"
  prevents model from treating TMDB marketing text as reviewer observations.
