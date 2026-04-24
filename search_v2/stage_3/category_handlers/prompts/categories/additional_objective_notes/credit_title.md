# Category notes: Credit + title text

This category covers **named credits on a film** — an actor, director, writer, producer, or composer — plus literal substring matches against movie titles. The endpoint resolves these by exact-string lookup against ingestion-time posting tables (for persons) or title-text match (for title patterns).

## Scope of what Entity actually indexes

Entity has posting tables for five credited roles only: actor, director, writer, producer, composer. Anything outside that set is invisible to this endpoint — a perfectly typed proper noun will retrieve zero movies if the role is not indexed. Treat the indexed-role list as a hard scope boundary, not a strong default.

## Using ROLE_MARKER modifiers to pick `person_category`

A `ROLE_MARKER` on the parent fragment is the strongest signal for which posting table to target. "Directed by" → `director`. "Starring" / "played by" → `actor`. "Written by" / "screenplay by" → `writer`. "Produced by" → `producer`. "Score by" / "composed by" → `composer`.

A bare person name with no role marker and no role-cued framing in `atomic_rewrite` defaults to `broad_person`. `primary_category` then carries the "predominantly known for" bias when one exists; leave it null when the person is genuinely equally credited across roles.

## Title patterns

`entity_type=title_pattern` fires when the requirement names a literal string to match inside movie titles ("titles containing 'love'", "films starting with 'The'"). It is NOT exact-title lookup — that does not reach this handler. Pick `contains` for anywhere-in-title phrasing and `starts_with` for explicit prefix phrasing.

## Boundaries with nearby categories

**Below-the-line creator (Cat 29).** If the requirement names a role Entity does not index — cinematographer, editor, production designer, costume designer, visual-effects supervisor — no-fire. Cat 29 owns that territory via semantic retrieval, and Step 2 co-emits into it. Forcing a `broad_person` lookup on a Deakins or Schoonmaker name returns nothing useful because those roles have no posting table.

**Source-material author (Cat 30).** If the named person is the author of the *book / story / work* the film adapts rather than someone credited on the film itself, no-fire. "Stephen King adaptations", "Jane Austen films", "Philip K. Dick stories" name origin-work authors, not film credits. Cat 30 handles these via semantic search over plot and reception vectors where the author's name surfaces in prose.

**Named character (Cat 2).** A character name ("Batman", "Hannibal Lecter") routes to Cat 2, not here. If a character-framed requirement somehow reaches this handler, the dispatch was wrong — the bucket guardrail on dispatch errors applies.

## When to no-fire

- The requirement names a role Entity does not index (see Cat 29 boundary above).
- The requirement names a source-material author rather than a film credit (see Cat 30 boundary).
- The requirement is too vague to pin to a specific named entity or literal title fragment ("a famous director", "a movie with a cool title").
- A `POLARITY_MODIFIER` combined with the atomic rewrite produces a contradiction Entity cannot express — record it in `coverage_gaps` rather than forcing a wrapper-polarity inversion onto a requirement that does not actually reduce to a single named entity.

Entity's posting-table retrieval is precise but narrow. When the requirement is outside its scope, no-fire with a clear `coverage_gaps` note is the correct outcome — do not degrade a mismatched requirement into a nearest-surface-form lookup that will silently return nothing.
