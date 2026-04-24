# Named character — additional notes

This category is about the presence of a **specific named persona** in the film — Batman, Wolverine, Harry Potter, James Bond, Hermione Granger. The target is a credited character string that appears in cast lists, not a role type and not a real person. The endpoint performs literal string lookup against the character posting table, so your job is to resolve the right credited name(s) and the right prominence mode.

## Entity type and prominence

Always emit `entity_type: "character"`. Character prominence is a **two-mode axis**, not four:

- `central` — the character is framed as the subject of the film. Triggered by possessive / about-framings ("Batman movies", "the Joker's story", "films about Wolverine", "centers on Harry Potter").
- `default` — the character is named without subject framing. "Any Wolverine appearance", "movies with Hermione", "cameo by the Joker". Use `default` whenever no language marks the character as the film's subject.

Do **not** emit `lead`, `supporting`, or `minor` for a character. Those modes describe actor billing, a different axis — a character is either the subject of the film or a named appearance within it, which is exactly the `subject` / `default` split.

## Alternative credited forms

Iconic characters are frequently credited under multiple strings across films — civilian / secret-identity pairings ("Bruce Wayne" / "Batman"), alias-plus-legal-name pairings, long-form vs. bare-name variants. Include any form that plausibly appears as a credit string in at least one film featuring the character. Cost is asymmetric: a spurious form matches nothing and costs zero, an omitted real form silently drops every film that credits it.

Do not include fan nicknames, dialogue references, or descriptive phrases — only strings that would appear in a real cast list.

## Boundaries with nearby categories

- **Character archetype (Cat 7).** A character TYPE pattern — "lovable rogue", "femme fatale", "anti-hero", "reluctant hero" — is not a specific named persona. No-fire here; it belongs to Character archetype. The discriminator is whether the reference names a single identifiable persona or a class of characters.
- **Specific subject / element / motif (Cat 6).** Fictional element categories like "movies with zombies" or "shark movies" are motif presence, not a named character. No-fire.
- **Credit + title text (Cat 1).** Named actors and other film credits belong there. "Christian Bale movies" is a person lookup; "Batman movies" is a character lookup. Even when an actor is strongly identified with a role, the surface reference picks the lane.
- **Franchise / universe lineage (Cat 4).** Franchise-anchor characters ("Batman movies", "Spider-Man movies") decompose upstream into both a Named-character atom AND a Franchise atom. Handle only the character side here — fire on the character as phrased and let the franchise atom run its own handler. Do not broaden the character lookup into franchise framing to "cover both"; that double-counts.
- **Generic role references.** "The hero", "the protagonist", "a detective" are not named characters. No-fire.
- **Real-person biopics.** "A movie about JFK", "the Princess Diana biopic" names a real subject depicted, not a credited character. That is Specific subject (Cat 6), not here — no-fire.

## When to no-fire

Apply the Single-bucket no-fire discipline strictly. The most common misroutes into this category are type patterns and generic roles — in both cases the upstream dispatch was wrong and fabricating a character name from a type phrase (inventing "The Lovable Rogue" as a primary_form) produces a zero-match lookup that also pollutes the results. Record the mismatch in `coverage_gaps` and return `should_run_endpoint: false`.

The one principle that covers most failures: **fire only when the requirement names a specific persona that would plausibly appear as a credited character string.** If you cannot point to such a string without inventing one, no-fire.
