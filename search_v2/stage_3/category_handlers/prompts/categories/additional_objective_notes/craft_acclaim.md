# Craft acclaim — additional notes

This category covers **acclaim attached to a specific craft axis**: phrases that praise a named dimension of filmmaking — visuals, score, soundtrack, dialogue, production craft, technical achievement — without naming the person responsible. "Visually stunning", "iconic score", "quotable dialogue", "technical marvel", "beautifully shot", "memorable theme", "naturalistic dialogue".

## Reception is always in scope; other spaces fire per axis

The reception space's `praised_qualities` sub-field is authored at ingest with axis-naming phrases ("cinematography", "score", "production design", "dialogue", "visual effects") — so a craft-acclaim ask lands most cleanly there. Always populate a reception entry, and set `primary_vector: "reception"`.

Additional spaces fire only when the craft axis genuinely belongs to that space's ingest vocabulary:

- **Visual / technical craft** — "practical effects", "IMAX-shot", "beautifully shot" — also lands on `production.production_techniques`, where craft-technique terms are authored at ingest.
- **Dialogue or narrative craft as praise** — "quotable dialogue", "Sorkin-style dialogue" framed as acclaim — also lands on `narrative_techniques` (narrative_delivery, characterization_methods, additional_narrative_devices), where writing-craft descriptors live.
- **Musical craft** — "iconic score", "great soundtrack", "memorable theme" — lands on reception only. No other space carries music-craft vocabulary.

Populate multiple spaces when the craft axis truthfully spans them. Do not add a space when it can only weakly paraphrase the axis — a marginal entry dilutes the vector.

## How to populate the reception entry

- Put an axis-naming phrase in `praised_qualities` — "cinematography", "visual style", "musical score", "soundtrack", "dialogue", "production design", "visual effects". Match the register of ingest-side praise tags.
- Write a tight `reception_summary` describing the axis being praised (one short sentence in ingest register). Leave `criticized_qualities` empty unless the query explicitly names a criticism.
- Do NOT put tonal adjectives, plot content, or genre terms in reception — those belong to other spaces and this category does not route there.

## Boundaries with nearby categories

- **Credit + title text (Cat 1).** If the query NAMES a creator whose role is indexed (director, composer, writer), that is Cat 1's territory — no-fire here. "Christopher Nolan films" → Cat 1. "Films with stunning visuals" → here.
- **Below-the-line creator (Cat 29).** If the query names a cinematographer, editor, production designer, costume designer, or VFX supervisor, that is Cat 29's territory — no-fire here. This category is acclaim ABOUT a craft axis without naming a specific creator.
- **Reception quality (Cat 25).** Cat 25 is general reception — "acclaimed", "cult classic", "critically praised" — with no axis named. This category requires the praise to attach to a specific craft dimension. "Acclaimed films" → Cat 25. "Acclaimed cinematography" → here.
- **Viewer experience (Cat 22).** Cat 22 is how it feels to watch ("visually overwhelming as an experience"). This category is what the film was praised for. When the user frames the ask as reception — what critics or audiences singled out — Cat 24 owns it. Framing that is purely experiential with no acclaim angle belongs to Cat 22.

## When to no-fire

Return `should_run_endpoint: false` when:

- The query names a specific creator (director, cinematographer, composer, editor) — dispatch was wrong; the named-entity categories own those asks.
- The praise has no craft axis attached — "acclaimed" or "well-regarded" alone is reception-as-scalar, which belongs to metadata rather than this semantic route.
- The phrase is purely experiential or tonal with no acclaim framing — no axis-specific praised_qualities tag would honestly match.

No-fire is always better than populating reception with a fabricated axis the user did not name.
