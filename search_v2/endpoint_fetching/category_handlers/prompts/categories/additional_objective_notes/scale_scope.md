# Scale / scope / holistic vibe — additional notes

This category is about **how large or small the movie's ambition feels** — its canvas, scope, and overall identity-level vibe. "Epic", "sprawling", "intimate", "small and personal", "operatic", "low-key" all name scale. It also covers holistic "feels like X" asks when X points at a vibe or identity rather than a plot shape.

## Plot_analysis is the home space

Scale and holistic vibe have no dedicated sub-space of their own. `plot_analysis` is the closest fit and the correct target: `elevator_pitch` carries the one-sentence identity framing that implicitly encodes scope ("a sweeping decades-spanning war saga" vs. "a quiet two-hander set in a single apartment"), and `genre_signatures` can carry scope-implying labels ("epic", "chamber drama", "character study"). `primary_vector` is `plot_analysis` for every fire in this category.

Other spaces may co-populate when the ask genuinely spans dimensions — a "feels like Lost in Translation" ask may touch `viewer_experience` for its melancholic, low-energy tonal register. Add a second space only when a concrete atom from the input actually lands there, not as reflex fan-out.

## "Feels like X" routing

"Feels like X" sits on this category only when X is vibe-shaped or identity-shaped — a specific film whose recognizable quality is its overall register ("feels like Lost in Translation", "feels like early Wes Anderson", "feels like a Lynch film"). Translate the reference into the space's native vocabulary: name the scope, tonal register, and thematic shape you would expect on the referenced film's ingest-side `elevator_pitch` and `genre_signatures`. Do not name the reference film itself in the body — ingest text describes movies, it does not cite them.

## Body-authoring for scale

`elevator_pitch` is usually the load-bearing sub-field. Write a one-sentence capsule for the kind of film the user is asking for, written the way a matching film's ingest-side pitch would read. A scale word on its own ("epic") is too thin to embed usefully — expand it into the scope-implying shape ("a sweeping, large-canvas saga spanning years and multiple fronts"). `genre_signatures` can carry compact scope-implying labels when they exist in film vocabulary ("epic", "chamber piece", "intimate character study", "small-scale drama"). Do not pad `thematic_concepts` with abstract theme words that the input does not name.

## Boundaries with nearby categories

- **Viewer experience (Cat 22).** Cat 22 is during-viewing feel — tonal aesthetic, cognitive demand, tension, sensory load. Cat 27 is scale / scope. The discriminator is the axis the qualifier lives on: "cerebral", "dark", "cozy" name the during-viewing experience and route to Cat 22; "epic", "intimate", "sprawling", "small" name the canvas and route here. If the qualifier is purely a during-viewing feel with no scope implication, no-fire.
- **Plot events (Cat 20).** Plot-shaped "feels like" asks route to Cat 20. "Feels like a heist" names the plot shape — the shape of what happens — and belongs there. Cat 27 is for vibe-shaped or identity-shaped "feels like" asks where the reference is a whole film's register, not its plot template. If the reference resolves to a specific event pattern rather than a holistic vibe, no-fire.
- **Sub-genre / story archetype (Cat 15).** Named story patterns and recognized sub-genre labels ("space opera", "neo-noir", "slasher") route to Cat 15's keyword channel. Cat 27 is for bare scope descriptors ("epic", "intimate") that are not themselves sub-genre labels. "Epic fantasy" routes the sub-genre atom to Cat 15 and leaves scale for Cat 27 on a separate atom; a bare "epic" with no story-pattern attached stays here.
- **Kind of story / thematic archetype (Cat 21).** Cat 21 covers the overarching theme or arc (grief, redemption, coming-of-age). Cat 27 covers scale and holistic identity. A thematic atom is Cat 21's; a scope atom is Cat 27's. Scale is not a theme.

## When to no-fire

Return `should_run_endpoint: false` when:

- The qualifier is a during-viewing tonal or cognitive feel with no scope dimension — that is Cat 22's territory.
- The "feels like X" reference is plot-shaped (a specific event template or sub-genre) rather than vibe-shaped — that is Cat 20 or Cat 15's territory.
- The phrase is too vague to point at any concrete scope framing or identity capsule ("good vibes", "cinematic vibe") with no scale or identity signal attached.
- The ask names a theme or arc (grief, redemption) rather than a scope — that is Cat 21's territory.

No-fire is always better than stuffing a scope word into `thematic_concepts` or stretching it onto `viewer_experience` sub-fields where it does not belong. A weak query vector underperforms no query at all.
