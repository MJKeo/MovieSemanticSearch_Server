# Below-the-line creator — additional notes

This category covers **lookups by a specific named creator in a non-indexed film-craft role** — cinematographer / director of photography, editor, production designer, costume designer, visual-effects supervisor, makeup artist. "Roger Deakins films", "Thelma Schoonmaker-edited", "Sandy Powell costumes", "Colleen Atwood designs". The ask must name a specific person by name AND identify a below-the-line role; pure axis-only acclaim with no name belongs to Craft acclaim, not here.

## Why Semantic is the only channel

Below-the-line roles are not stored in any posting table — the indexed roles are limited to actor, director, writer, producer, and composer. The creator's name can still surface in reception prose, because critics and audiences frequently name the cinematographer / editor / costume designer when praising the work. The reception space is therefore the one honest retrieval path: match films whose ingest-side reception text mentions the named creator alongside their craft.

## How to populate the reception entry

- `reception_summary` — a tight ingest-register sentence that names BOTH the creator and their role ("cinematography by Roger Deakins, widely praised for its naturalistic lighting"). Name + role together is what makes the embedding discriminative; the name alone matches too many unrelated mentions.
- `praised_qualities` — include the creator's name paired with role-naming phrases the ingest side uses ("cinematography by Roger Deakins", "Thelma Schoonmaker's editing", "costume design by Sandy Powell"). Also include the bare axis term ("cinematography", "editing", "costume design") as a secondary anchor so the vector leans on both the name and the craft.
- Leave `criticized_qualities` empty unless the query explicitly frames the ask as criticism.
- `primary_vector: "reception"`. No other space at ingest routinely names below-the-line creators by name.

Do NOT add production or narrative_techniques entries just because the craft axis touches them — those spaces do not carry person-level attribution at ingest, so the name would not actually match anything there. A second space is only honest when the query adds a distinct signal beyond the creator's name (e.g. a specific filming location), not to pad coverage of the craft axis.

## Boundaries with nearby categories

- **Credit + title text (Cat 1).** If the named role is indexed — actor, director, writer, producer, composer — that is Cat 1's territory, not this one. "Christopher Nolan films" → Cat 1. "Roger Deakins films" (cinematographer, non-indexed) → here. The named role determines routing.
- **Craft acclaim (Cat 24).** Cat 24 is acclaim attached to a craft axis WITHOUT a specific creator name. "Beautifully shot films" / "acclaimed cinematography" → Cat 24. "Roger Deakins films" → here. A name makes it Cat 29; no name keeps it Cat 24.
- **Source-material author (Cat 30).** Cat 30 also name-looks against reception prose, but for source-material authors (novelists, comic creators, screenwriters of the original book). Cat 29 is film-craft creators. "Stephen King adaptations" → Cat 30. "Thelma Schoonmaker-edited" → here.

## When to no-fire

Return `should_run_endpoint: false` when:

- The query names a creator whose role IS indexed (director, actor, writer, producer, composer) — dispatch was wrong; Cat 1 owns that.
- The query praises a craft axis without naming a specific below-the-line creator ("beautifully shot", "great costumes") — Cat 24 territory; no name means no honest reception signal to embed here.
- The query names a person who is NOT a film-craft creator (a subject depicted, a source-material author, a real-world figure) — the reception space would not carry that person as a praised quality.

No-fire with the coverage gap written plainly is always better than inventing a reception payload whose name-plus-role pairing the user did not actually provide.
