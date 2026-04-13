# [058] — Vector Text Formatting Conventions: Labeled Fields, Synopsis-First Hierarchy, Formatting in embedding_text()

## Status
Active

## Context

As vector text functions were ported from `BaseMovie` to the new `Movie`
object (see ADR-056), several formatting questions arose that would set
patterns for all 8 vector spaces:
1. Should short categorical fields carry semantic labels ("genre:", "praised:") or be left bare?
2. For plot_events, should IMDB synopses or LLM-generated summaries be preferred?
3. Should field formatting logic live in the vector text function or in the metadata schema's `embedding_text()`?
4. What normalization strategy applies to assembled vector text?

## Decision

**1. Labeled short fields.** Short categorical fields (genre lists, quality
tags, keywords) are prefixed with lowercase semantic labels
("genre: drama, thriller", "praised: tight editing, performances").
Reception summary is also labeled (`reception_summary:`) so the full
reception vector has a consistently structured shape. Viewer experience
follows the same rule at section granularity: each populated section emits
`section_name:` and, when applicable, `section_name_negations:` as separate
lines so polarity is preserved explicitly rather than mixing negations into
the positive term list. Rationale: contextual retrieval research (Anthropic,
Google Gemini embedding docs, LlamaIndex defaults) supports labels for
disambiguation of short values. Labels are applied consistently across
plot_analysis, narrative_techniques, viewer_experience, watch_context,
reception, anchor, and production vector spaces. The anchor now uses this
convention for every populated field in a
stable multiline order:
`title:`, `original_title:`, `identity_pitch:`, `identity_overview:`,
`genre_signatures:`, `themes:`, `emotional_palette:`, `key_draws:`,
`maturity_summary:`, `reception_summary:`.
Narrative techniques follows the same section-label rule in a stable order:
`narrative_archetype:`, `narrative_delivery:`, `pov_perspective:`,
`characterization_methods:`, `character_arcs:`,
`audience_character_perception:`, `information_control:`,
`conflict_stakes_design:`, `additional_narrative_devices:`. Empty sections
are omitted.
Watch context follows the same section-label rule in a stable order:
`self_experience_motivations:`, `external_motivations:`,
`key_movie_feature_draws:`, `watch_scenarios:`. Empty sections are omitted.
Production follows a fixed two-line shape in stable order:
`filming_locations:` then `production_techniques:`. Empty lines are omitted,
and the function returns `None` when both signals are absent. Countries,
studios, languages, decade, budget/box office, source material, franchise,
and broad medium labels are deliberately excluded from this vector.

**2. Synopsis-first fallback hierarchy for plot_events.** The plot_events
vector embeds the richest available plot text in order:
(a) longest scraped IMDB synopsis (human-written, most detailed),
(b) LLM-generated plot_summary via `plot_events_metadata.embedding_text()`,
(c) longest plot_summary entry,
(d) TMDB overview.
A separate `create_plot_events_vector_text_fallback()` function handles
the case where the primary text exceeds the 8,191-token embedding limit
(text-embedding-3-small errors on oversize input; it does not truncate).
Rationale: aligns with ADR-033's two-branch strategy; IMDB synopses are
the highest-quality plot text in the tracker.

**3. Formatting logic lives in `embedding_text()`, not vector_text.**
For schemas with non-trivial formatting (labeled fields, per-term
normalization, genre merging), the formatting lives inside the schema's
`embedding_text()` method. The vector text function is a thin wrapper
that supplies derived signals (TMDB genres, deterministic award wins) and delegates
to `embedding_text()`. Rationale: keeps field formatting co-located with
schema field definitions; vector text functions focus on combining data
sources rather than formatting individual fields.

**4. Normalization strategy.** Prose fields use `.lower()`. Categorical
term lists apply `normalize_string()` per-term before joining. The final
assembled vector text string does NOT have `normalize_string()` applied
to it again — normalization happens at the field level, not the string
level. Exception: the plot_events functions call `normalize_string()`
once on the entire accumulated text (since the text comes from external
sources, not schema fields).

**5. Return type is `str | None`.** All vector text functions return
`None` when required metadata is absent, rather than empty string.
Callers can distinguish "no data" from empty content.

## Alternatives Considered

1. **No semantic labels**: Simpler, but short categorical values ("drama",
   "thriller") lose context when embedded in isolation. Without a label,
   the embedding model may not consistently distinguish genre terms from
   keyword terms from quality tags.

2. **Prefer LLM summary over IMDB synopsis for plot_events**: LLM summaries
   are generated by the model from the synopsis as input — they are
   secondary. When an IMDB synopsis is available, it is the ground-truth
   plot description and should be embedded directly.

3. **Keep formatting in vector_text functions**: Would centralize all
   vector text logic in one place but scatters the schema's own
   embedding representation across multiple files. When a schema field
   changes, the developer must update both the schema and the vector text
   function — easy to miss.

4. **Apply `normalize_string()` once to the full assembled string**: Would
   simplify the call sites but prevents field-level control over
   normalization (e.g., preserving label colons, handling prose differently
   from keyword lists).

## Consequences

- Labeled field convention must be applied consistently when adding new
  vector spaces or modifying existing ones. An unlabeled field in a labeled
  context creates inconsistent embedding behavior.
- The anchor is a lean holistic surface, not a catch-all metadata dump.
  Structured/filterable facts like keywords, franchise/source material,
  languages, decade, budget, and awards belong in deterministic retrieval
  or specialized spaces rather than being reintroduced into anchor text.
- For schemas with positive/negative polarity, negations should receive their
  own explicit label (for example `*_negations:`) rather than being appended
  to the positive line. This keeps retrieval aligned with how search prompts
  target the same structure.
- `embedding_text()` is now a real formatting function, not just a
  `__str__()` alias. Schema changes that affect vector text must update
  `embedding_text()`.
- ~~`create_plot_events_vector_text_fallback()` needs to be wired into the
  embedding pipeline's error handling.~~ Resolved: the token-limit fallback
  is now integrated directly into `create_plot_events_vector_text()` via
  `_plot_events_fallback_text()`. Callers do not need to handle it.
- The synopsis-first hierarchy means that if an IMDB synopsis is very
  long (>8,191 tokens), the fallback kicks in and may produce a shorter
  but less complete embedding. This is an acceptable tradeoff vs. a
  hard API error.

## References

- `movie_ingestion/final_ingestion/vector_text.py`
- `schemas/metadata.py`
- `docs/modules/schemas.md`
- ADR-033 (plot_events two-branch design)
- ADR-056 (Movie object — the parameter type for all vector text functions)
- ADR-057 (EmbeddableOutput — embedding_text() contract)
