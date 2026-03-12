"""
Pydantic response schemas for LLM structured output.

These are the generation-side schemas used as response_format in batch
requests. They're based on the existing schemas in
implementation/classes/schemas.py but with key modifications from the
redesigned flow (docs/llm_metadata_generation_new_flow.md):

Changes from existing schemas:
    - Justification fields REMOVED from all section models (pending
      empirical validation -- easy to add back). This includes:
      * GenericTermsSection.justification
      * ViewerExperienceSection.justification
      * SourceOfInspirationSection.justification
      * CoreConcept.explanation_and_justification
      * MajorTheme.explanation_and_justification
      * MajorLessonLearned.explanation_and_justification

    - ReceptionMetadata gains review_insights_brief field:
      ~150-250 token dense paragraph capturing key thematic, emotional,
      structural, and source-material observations from reviews.
      This is an intermediate output consumed by Wave 2 generators,
      not embedded or stored in Qdrant.

    - ProductionKeywordsResponse and SourceOfInspirationResponse are
      SEPARATE schemas (they're separate LLM calls). The existing
      ProductionMetadata merges them awkwardly into one model.

The existing schemas in implementation/classes/schemas.py remain
unchanged -- they're consumed by the search pipeline for reading
metadata from Qdrant. These generation-side schemas can evolve
independently. When deploying, align the search-side schemas.

Each schema class implements __str__() for vector text generation
(lowercased, concatenated terms) matching the existing pattern.

Pydantic's type_to_response_format_param() is used in generators
to convert these schemas into the json_schema format required by
the Batch API's response_format field.
"""
