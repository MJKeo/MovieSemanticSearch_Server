# ADR-025 — Metadata Generation Schema Design: Justification Removal and Intermediate Outputs

## Status
Active

## Context

Stage 6 generates LLM vector metadata for ~112K movies via OpenAI's Batch
API. The existing search-side schemas in `implementation/classes/schemas.py`
include justification/explanation fields on section models (e.g.,
`GenericTermsSection.justification`, `CoreConcept.explanation_and_justification`)
intended to improve chain-of-thought quality. These fields are never embedded —
they exist solely to guide LLM reasoning.

Two design questions arose when creating the generation-side schemas in
`metadata_generation/schemas.py`:

1. Should the generation schemas include the same justification fields?
2. How should Wave 1 outputs pass context to Wave 2 generators without
   bloating the embedded text or duplicating raw review content in every prompt?

## Decision

**Justification fields removed from all generation-side schemas.** All
section models (`TermsSection`, `TermsWithNegationsSection`,
`OptionalTermsWithNegationsSection`, and all output schemas) omit
justification/explanation fields. This is a pending empirical validation —
the fields are easy to add back if quality suffers.

**`review_insights_brief` added to `ReceptionOutput` as a non-embedded
intermediate.** Reception (Wave 1) produces a ~150-250 token dense paragraph
capturing thematic, emotional, structural, and source-material observations
from reviews. This field is:
- Stored as a scalar column in `metadata_results` for direct SQL access
  by Wave 2 request building
- Passed to all 6 Wave 2 generator prompts as a compact review proxy
- **Excluded from `ReceptionOutput.__str__()`** so it is never embedded
  into Qdrant

**Generation schemas diverge from search schemas intentionally.** The
search-side schemas in `implementation/classes/schemas.py` remain unchanged.
The generation-side schemas in `metadata_generation/schemas.py` evolve
independently. When deploying, align the search-side schemas to match.

**`ProductionKeywordsOutput` and `SourceOfInspirationOutput` are separate
schemas** corresponding to separate LLM calls, unlike `ProductionMetadata`
in the search-side schemas which merged them into one model.

## Alternatives Considered

1. **Keep justification fields in generation schemas**: Would match the
   search-side schema structure and might improve CoT quality. Rejected
   initially — adds token cost per request across ~112K × 8 calls, and
   the empirical quality benefit is unvalidated. Easy to add back.

2. **Pass raw review text directly to Wave 2 generators**: Would give Wave 2
   the most complete reception signal. Rejected — raw reviews are verbose
   and variable in length. A dense synthesis paragraph (`review_insights_brief`)
   is more token-efficient and focuses on what matters for downstream
   generation (themes, emotional register, structural observations).

3. **Store `review_insights_brief` in a separate table or bury it in
   `result_json`**: Rejected — scalar column in `metadata_results` allows
   Wave 2 request building to SELECT it directly without JSON parsing,
   consistent with how `plot_synopsis` is handled (ADR-024, decision 3).

## Consequences

- Token cost per batch is lower without justification fields. If quality
  is insufficient, fields can be added back at the schema level and the
  batch re-run (stored `result_json` allows re-parsing without re-running).
- `review_insights_brief` requires careful handling in `result_processor.py`
  and `request_builder.py` — it must be extracted and stored as a scalar,
  not just archived in `result_json`.
- The divergence between generation-side and search-side schemas creates a
  deployment step: `implementation/classes/schemas.py` must be updated before
  the embedded metadata can be queried correctly by the search pipeline.

## References

- ADR-024 (Batch API architecture) — `review_insights_brief` as scalar column
- docs/modules/ingestion.md (Stage 6 section, Output schemas subsection)
- movie_ingestion/metadata_generation/schemas.py
- implementation/classes/schemas.py (search-side, not modified)
- docs/llm_metadata_generation_new_flow.md — Section 5 (Decision 5)
