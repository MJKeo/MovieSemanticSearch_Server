# ADR-001: Eight Named Vector Spaces

**Status:** Active

## Context

Movie search queries span fundamentally different intent types —
plot details, emotional tone, viewing context, thematic analysis,
production facts, critical reception. A single embedding space
conflates these lenses, producing mediocre results across all
of them.

## Decision

Store 8 named vectors per movie in Qdrant, each capturing a
distinct retrieval lens:

| Vector | What it captures |
|--------|-----------------|
| `dense_anchor_vectors` | Broad "movie card" — identity, content, production, cast, reception. General recall. |
| `plot_events_vectors` | Literal chronological plot (who, what, when, where). Spoiler-containing. |
| `plot_analysis_vectors` | Thematic analysis — core concept, genre signatures, character arcs, themes, lessons. |
| `narrative_techniques_vectors` | Storytelling mechanics — POV, structure, devices, archetypes. |
| `viewer_experience_vectors` | Felt experience — emotions, tone, tension, cognitive load, disturbance, aftertaste. Supports negations ("not too scary"). |
| `watch_context_vectors` | Use-case lens — motivations, scenarios, feature draws ("date night", "background movie"). |
| `production_vectors` | Making-of facts — countries, studios, locations, languages, decade, budget, cast/crew, adaptation source. |
| `reception_vectors` | Critical consensus — acclaim tier, praise/complaint attributes, reception summary. |

All vectors use OpenAI `text-embedding-3-small` (1536 dims).

## Alternatives Considered

1. **Single vector space**: Simplest but conflates all query
   intents. A "cozy date night movie" query competes with plot
   details and production facts in the same space.
2. **3-4 vector spaces** (plot, vibe, metadata, reception):
   Loses granularity between e.g. viewer experience (how it
   feels) and watch context (when to watch it).
3. **Per-field vectors** (one per attribute): Too many spaces,
   each too sparse to be useful.

## Consequences

- Each movie requires 8 × 1536 = 12,288 floats of vector storage.
  At 150K movies, this is ~7.4 GB uncompressed. Mitigated by
  scalar quantization (ADR-004).
- LLM metadata generation at ingestion time requires 8 calls per
  movie to produce the text for each space.
- Search-time query understanding must generate per-space subqueries
  and relevance weights to target the right spaces.
- Anchor vector always participates in search (general recall);
  other spaces participate based on LLM-assigned relevance.

## References

- guides/movie_vector_definitions.md (vector descriptions)
- guides/movie_vector_schemas.md (what data goes into each)
- guides/movie_vector_analysis.md (embedding space analysis)
- guides/vector_scoring_guide.md (scoring pipeline)
