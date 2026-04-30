# Plot events + narrative setting — additional notes

This category covers **concrete plot content and narrative time/place**: literal events that happen on screen ("a heist that unravels when a crew member betrays the others", "stranded on an island after a plane crash"), the era the story is set in ("1940s Berlin", "during the Cold War"), and the place the story takes place ("Tokyo", "a small desert town"). It is event-level and setting-level description, not theme, not tone, not a named archetype label.

## Semantic `plot_events` is the only target

Of the seven vector spaces, `plot_events` is the single space whose ingest body is raw synopsis prose — a dense, lowercased plot summary describing concrete actions, characters, and the where/when of the story. That is the only space where a specific event description or a named narrative setting lands cleanly on a matching movie's ingest side. Every other space carries labels (themes, feelings, craft terms, production terms, reception terms) rather than synopsis prose.

`primary_vector` is always `plot_events`. Do not populate any other space — thematic restatements in `plot_analysis`, tonal adjectives in `viewer_experience`, or location terms in `production` would drift off the ingest shape the user's phrasing actually matches.

## How to populate the body

- Write `plot_summary` as dense synopsis-style prose — the register of an ingest-side plot summary. One or two tight sentences describing the concrete situation: who, what happens, where, when.
- Fold narrative setting directly into the prose. "Set in 1940s Berlin" becomes part of the summary ("a story set in 1940s Berlin about…") — do not route the setting to a separate space and do not emit a bare location string.
- Stay close to what the input actually describes. You may restate the situation in natural synopsis vocabulary, but do not invent plot facts, characters, motives, or outcomes the user did not supply.
- Do not use thematic labels ("a story about trust", "a meditation on loss") or experiential adjectives ("tense", "haunting") inside the body. Those belong to other spaces and this category does not route there.

## Boundaries with nearby categories

- **Filming location (Cat 13).** Cat 13 is about where the camera physically was; this category is about where the story is set. "Filmed in Tokyo" → Cat 13. "Set in Tokyo" / "takes place in Tokyo" → here. If the target phrase names production geography rather than narrative setting, upstream misrouted — no-fire.
- **Structured metadata (Cat 10).** Cat 10 carries real-world release-date filters as a structured column. "Released in the 1940s" / "1940s movies" → Cat 10. "Set in 1940s Berlin" → here — the decade names a story-time, not a release window. Discriminator: does the year/era describe WHEN THE MOVIE CAME OUT, or WHEN THE STORY HAPPENS?
- **Kind of story / thematic archetype (Cat 21).** Cat 21 is abstract thematic patterns — grief, redemption, forgiveness, man-vs-self. This category is the concrete event-level description. "A story about redemption" → Cat 21. "A heist that falls apart when a crew member betrays the others" → here. If the phrase names a theme without a concrete event or setting, no-fire.
- **Sub-genre + story archetype (Cat 15).** Cat 15 is the named label for a story pattern (HEIST, SURVIVAL, REVENGE) — a canonical concept tag. This category is a concrete event description of the same territory. A query naming a pattern at the label level ("heist movies") lives in Cat 15; a query describing the situation ("a heist that unravels when…") lives here. Both can legitimately co-fire on the same fragment from upstream — that is expected, not a conflict.
- **Viewer experience (Cat 22).** Tone and felt experience ("tense", "unsettling", "uplifting") belong to Cat 22. Concrete events and setting belong here. A purely tonal phrase with no event or setting content is a no-fire.

## When to no-fire

Return `should_run_endpoint: false` when:

- The phrase names filming geography rather than narrative setting.
- The phrase names a real-world release era rather than a story-time setting.
- The phrase is purely thematic (a story about X) with no concrete event or setting atom.
- The phrase is purely tonal, experiential, or occasion-based with no event or setting content.
- The phrase is too vague to describe any concrete situation ("a wild plot", "crazy events") — `plot_events` has no synopsis atom to embed against.

No-fire is always better than embedding a plot_summary body assembled from themes, tone words, or filming-geography phrases the user did not ground in a concrete story event or setting.
