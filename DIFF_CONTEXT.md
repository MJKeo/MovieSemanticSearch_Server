# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Concept tag generation pipeline (new metadata type)

Files: schemas/enums.py, schemas/metadata.py, movie_ingestion/metadata_generation/inputs.py, movie_ingestion/metadata_generation/batch_generation/pre_consolidation.py, movie_ingestion/metadata_generation/prompts/concept_tags.py (new), movie_ingestion/metadata_generation/generators/concept_tags.py (new), movie_ingestion/metadata_generation/batch_generation/generator_registry.py, movie_ingestion/metadata_generation/batch_generation/result_processor.py, movie_ingestion/tracker.py

### Intent
Adds CONCEPT_TAGS as a new Wave 2 metadata generation type. Classifies 27 binary concept tags across 7 categories (narrative structure, plot archetypes, settings, characters, endings, experiential, content flags) via LLM multi-label classification. Tags enable deterministic Phase 1 search retrieval via Postgres INT[] array containment queries.

### Key Decisions
- **ConceptTag enum uses dual-value (str, int) pattern** matching SourceMaterialType. IDs are gapped by category (1-8, 11-14, 21-23, 31-33, 41-42, 51-52, 61) for future extensibility.
- **ConceptTagsOutput does NOT subclass EmbeddableOutput** — concept tags become integer IDs in movie_card, not embedding text. This is the first non-embeddable generation output.
- **Category-level arrays with evidence-before-tag ordering** chosen over boolean grid (false-negative bias) and flat array (category-skipping). Evidence field forces chain-of-thought before tag commitment.
- **Eligibility gate**: plot_summary exists OR best_plot_fallback >= 250 chars OR plot_keywords >= 3. Lower than plot_analysis thresholds because binary classification needs less input depth than generative analysis.
- **Six LLM inputs** (~310-1140 tokens): title_with_year, plot_keywords, plot_summary/plot_text (quality-tiered), emotional_observations, narrative_technique_terms (6 of 9 sections, terms only), plot_analysis fields (arc labels + conflict_type).
- **Fallback chain**: plot_summary absent → best_plot_fallback() with "plot_text" label (any length); emotional_observations absent → "not available"; NT/PA absent → "not available".
- **New loader function** load_narrative_techniques_output() added to inputs.py, plus extract_narrative_technique_terms() helper.

### Planning Context
See search_improvement_planning/concept_tags.md for full tag definitions, routing table for concepts handled by other systems, and output schema design rationale.

### Testing Notes
- Schema validation: verify tags in wrong categories are rejected by model_validator
- Eligibility: test all 3 eligible paths + skip case
- Prompt builder: test with all-present, all-absent, and partial inputs
- End-to-end: run eligibility evaluation, then live generation on small batch to inspect output quality

## Documentation staleness audit fixes
Files: schemas/metadata.py, docs/modules/ingestion.md, docs/modules/schemas.md, docs/modules/llms.md, docs/conventions.md, docs/decisions/ADR-065-schema-docstrings-as-comments-not-python-docstrings.md, docs/decisions/ADR-058-vector-text-formatting-conventions.md, movie_ingestion/metadata_generation/batch_generation/pre_consolidation.py

### Intent
Full docs-auditor pass found 15 stale references — all traced to the concept_tags type being added without a doc sweep. Also fixed two pre-existing convention/ADR issues.

### Key Changes
- **schemas/metadata.py**: Converted `TagEvidence` and `ConceptTagsOutput` class docstrings to `#` comment blocks per ADR-065 (these were leaking into LLM JSON schema payload)
- **ingestion.md**: Updated all "8 generators"→10, "9 batch_id columns"→10, "9 JSON result columns"→10, Wave 2 eligibility count 6→7
- **schemas.md**: MetadataType count 9→10, added `ConceptTagsOutput`/`ConceptTag`/`TagEvidence`/`CONCEPT_TAG_CATEGORIES` to Key Types, updated boundary list with `extract_narrative_technique_terms` and `load_plot_analysis_output`
- **conventions.md**: Added missing `embedded` status to pipeline chain; updated `__str__()` normalization rule to reference `embedding_text()` per ADR-057
- **ADR-065**: Updated reference count from "8 `*Output` classes" to include 9 EmbeddableOutput subclasses + ConceptTagsOutput/TagEvidence
- **ADR-058**: Struck stale open consequence about unwired fallback (now integrated)
- **llms.md**: Documented `verbosity` kwarg; updated TokenUsage import count 8→10
- **pre_consolidation.py**: Fixed docstring count from "Eight" to "Nine" eligibility methods
