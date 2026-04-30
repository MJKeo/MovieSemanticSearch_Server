# Character archetype — additional notes

This category is about a **static character TYPE pattern** — lovable rogue, femme fatale, anti-hero, underdog protagonist, reluctant hero, manic pixie dream girl. The target is the class of character, not a specific named persona and not the trajectory of an arc. The endpoint choice resolves which channel best captures the type.

## How the two endpoints split the work

- **Keyword** is tier 1 and the authoritative channel when the archetype maps to a canonical registry member (the CharacterTag family — `ANTI_HERO`, `FEMALE_LEAD`, `ENSEMBLE_CAST`, and any adjacent members whose definition names the type). When one of those members defines the archetype in question, Keyword wins cleanly and Semantic adds nothing.
- **Semantic** is tier 2 and the right channel when no canonical tag covers the archetype. `narrative_techniques.characterization_methods` is the primary home for an uncanonized archetype — that sub-field exists to carry descriptor phrases for character type. `plot_events` can carry a supporting signal when the archetype is described by what the character concretely does in the synopsis.

The bias is a tiebreaker only. When no tag covers the archetype, Semantic wins decisively — picking a weakly-adjacent tag to honor the bias would misclassify the movie set.

## Boundaries with nearby categories

- **Named character (Cat 2).** A specific named persona (Batman, Wolverine, James Bond) is Cat 2. An abstract type (anti-hero, femme fatale) is Cat 7. The discriminator is whether the reference points to a single identifiable character or a class of characters. If the requirement names both a persona AND a type, the persona is Cat 2's atom and the type is this category's atom — handle only the type here.
- **Kind of story / thematic archetype (Cat 21).** Cat 21 covers character-arc trajectory ("redemption arc", "fall from grace", "coming of age") — the shape of how the character changes. Cat 7 covers the static type — who the character IS regardless of whether they change. "Anti-hero" is a static type; "redemption arc" is a trajectory. If the requirement names an arc pattern, no-fire here.
- **Specific subject / motif (Cat 6).** A motif is a thing IN the story (zombies, clowns, vampires) — a presence, not a character class. An archetype is a character type. They share no overlap when the ask is framed cleanly.

## When to no-fire

Return `endpoint_to_run: "None"` when the requirement is not actually a character-type ask:

- The phrase names an arc trajectory rather than a static type — that is Cat 21's territory.
- The phrase names a specific persona — that is Cat 2's territory.
- The phrase is too vague to point at either a canonical tag or a defensible characterization-methods body (e.g. "interesting characters", "good protagonists") — no candidate can express it cleanly.

## Picking within Semantic

When Semantic wins, the archetype's characterization-methods body is the load-bearing content. Keep the terms compact and in the register the space already uses on the ingest side — short descriptor phrases, not sentences. `primary_vector` should be `narrative_techniques` whenever that space carries the archetype; use `plot_events` as primary only if the archetype is genuinely expressible as a concrete situational pattern rather than as a character descriptor.

## The one principle

Fire **Keyword** only when a registry member's own definition clearly names the archetype. Fire **Semantic** when no member does but the archetype is a genuine character-type description. Otherwise, no-fire — fabricating a near-miss tag or padding a characterization-methods body with invented terms degrades the result set more than omission does.
