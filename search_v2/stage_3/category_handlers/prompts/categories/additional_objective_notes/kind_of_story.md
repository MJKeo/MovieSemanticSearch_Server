# Kind of story / thematic archetype — additional notes

This category is about the **overarching thematic concept or character-arc trajectory of the whole story** — grief, redemption, forgiveness, man-vs-nature, man-vs-self, coming-of-age as a life-stage pattern, a fall-from-grace trajectory. The target is the kind of story the movie IS, not its plot events and not a static character type.

## Spectrum escape hatch (read this first)

This handler has a **mechanic no other tiered category has**: when the framing of the theme is gradient rather than binary, skip Keyword entirely — even if a registry member's definition cleanly covers the theme — and go directly to Semantic.

- **Binary framing** — "movies about redemption", "a story of forgiveness", "coming-of-age stories". The user wants the theme as a defining feature. Keyword's posting-list semantics fit: the tag is either attached to the movie or it is not.
- **Spectrum framing** — "kind of about grief", "leans redemptive", "has touches of coming-of-age", "there's a forgiveness thread", "a slight man-vs-nature flavor". The user wants the theme present as a degree, not as a defining feature. A binary posting-list membership is the wrong shape for a gradient ask — it either returns only movies where the theme is front-and-center (too narrow) or broadens to every movie the tag was attached to (not what the user asked for). Semantic similarity against `plot_analysis.thematic_concepts` / `character_arcs` naturally produces a graded match.

The spectrum escape is a **framing-level decision**, not a vocabulary-level one. Even when a perfect registry member exists, the gradient framing makes the tag the wrong instrument. State this explicitly in `performance_vs_bias_analysis` — the tier-1 bias is being bypassed on merit, not ignored.

Surface cues for spectrum framing: "kind of", "sort of", "leans", "has touches of", "a hint of", "somewhat", "there's a ___ thread", "slightly". These cues shift the ask from "the movie is about X" to "X is present in some measure." Do not match these cues mechanically — use them as evidence for a gradient intent, then judge the whole phrasing.

## How the two endpoints split the work (when framing is binary)

- **Keyword** is tier 1 and authoritative when the theme maps to a registry member whose definition names the same concept and the framing is binary. The registry covers some themes directly (e.g. `COMING_OF_AGE`, `SURVIVAL`) and leaves many uncovered. When a member fits and the framing is binary, Keyword wins cleanly.
- **Semantic** is tier 2 and the right channel for every theme the registry does not cover, and for every spectrum-framed ask regardless of registry coverage. `plot_analysis` is the home space — `thematic_concepts` carries abstract theme terms, `character_arcs` carries arc-pattern trajectories, `conflict_type` carries man-vs-X archetypes, and `elevator_pitch` carries the capsule framing when the theme is the one-sentence hook of the story.

## Boundaries with nearby categories

- **Character archetype (Cat 7).** Cat 7 covers static character TYPES (anti-hero, femme fatale). Cat 21 covers thematic concepts and character-arc trajectories (redemption arc, fall from grace). The discriminator is the conceptual axis: who the character IS versus how the character changes or what the story is about. "Anti-hero" → Cat 7. "Redemption arc" → Cat 21. A character can be an anti-hero AND follow a redemption arc; those are two separate atoms on two different axes.
- **Sub-genre / story archetype labels (Cat 15).** Cat 15 covers named story-pattern labels — recognized story-shape names like "heist", "revenge", "slasher". Cat 21 covers abstract themes and arc trajectories. The discriminator is whether the phrase names a pattern label or an abstract theme. "A revenge story" → Cat 15 (the recognized label). "A story about forgiveness" → Cat 21 (the abstract theme). "A heist movie" → Cat 15. "A story about moral compromise" → Cat 21.
- **Target audience (Cat 17).** "Coming-of-age" can read as audience framing ("teen movie", "for teenagers") or as a story pattern ("a coming-of-age story about self-acceptance"). Cat 17 is the audience lens; Cat 21 is the story-pattern lens. If upstream routed an audience-framed ask here, no-fire.

## When to no-fire

Return `endpoint_to_run: "None"` when the requirement is not a thematic / arc ask that either endpoint can act on:

- The phrase names a static character type — that is Cat 7's territory.
- The phrase names a recognized story-pattern label rather than an abstract theme — that is Cat 15's territory.
- The phrase frames the ask as audience tier — that is Cat 17's territory.
- The phrase is too vague to point at either a registry member or a defensible `plot_analysis` body ("deep themes", "meaningful stories", "stories that matter").

## The one principle

Fire **Keyword** only when a registry member's own definition clearly names the theme AND the framing is binary. Fire **Semantic** when the framing is gradient, or when the theme is genuine but uncanonized. Make the spectrum-vs-binary call explicit in `performance_vs_bias_analysis` — silently defaulting to Keyword on surface vocabulary alone is the primary failure mode for this handler.
