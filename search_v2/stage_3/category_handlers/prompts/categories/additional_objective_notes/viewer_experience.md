# Viewer experience / feel / tone — additional notes

This category is about **how the movie feels to watch while it is playing** — the during-viewing experience. Tonal aesthetic ("dark", "gritty", "whimsical", "cozy", "melancholic"), cognitive demand ("mindless", "cerebral"), realism vs. stylization, tension and disturbance intensity, emotional palette. The concept is subjective and moment-to-moment, not a thematic label or a post-credits aftertaste.

## Semantic viewer_experience is the only target

This category routes exclusively to the Semantic endpoint, and only the `viewer_experience` space carries the signal. `primary_vector` is always `viewer_experience`. Do not populate other spaces — tonal feel does not genuinely land on plot_events, plot_analysis, or watch_context in a way that strengthens the query, and spreading weak signal across spaces dilutes the match.

## Sub-field selection is load-bearing

The `viewer_experience` body has eight paired sub-fields (`terms` + `negations`). Populate only the ones the requirement actually lands on. Empty sub-fields are valid and expected — most asks hit one or two, not all eight.

- Tonal aesthetic ("dark", "whimsical", "cozy", "gritty", "melancholic", "hopeful") → `emotional_palette` and usually `tone_self_seriousness`.
- Cognitive demand ("cerebral", "mindless", "demanding", "easy to follow") → `cognitive_complexity`.
- Tension / adrenaline level ("white-knuckle", "low-stakes", "pulse-pounding") → `tension_adrenaline`.
- Disturbance intensity as a general felt axis ("unsettling", "disquieting") → `disturbance_profile`.
- Sensory density ("sensory overload", "visually loud", "quiet", "restrained") → `sensory_load`.
- Emotional turbulence across the runtime ("rollercoaster", "emotionally steady") → `emotional_volatility`.

Negations carry a qualifier's boundary — "unsettling but not gory" puts "unsettling" in `disturbance_profile.terms` and "gore" / "graphic violence" in `disturbance_profile.negations`. Use negations only when the input itself names a boundary, not to pre-empt edge cases.

## Boundaries with nearby categories

- **Post-viewing resonance (Cat 26).** This category is during-viewing feel; Cat 26 is post-viewing aftertaste. "Haunting", "stays with you", "gut-punch ending", "forgettable", and ending-type framings are all Cat 26. The `viewer_experience.ending_aftertaste` sub-field is Cat 26's primary target, not this category's — do not populate `ending_aftertaste` here. If the phrase describes what lingers after the credits, no-fire and let Cat 26 own it.
- **Scale / scope / holistic vibe (Cat 27).** Scope descriptors ("epic", "intimate", "sprawling", "small and personal") name scale, not tonal feel. "Epic feel" reads as tonal on the surface but the user is naming scope; that is Cat 27. When the qualifier is a scale word rather than a tone or cognitive-demand word, no-fire.
- **Narrative devices / pacing-as-craft (Cat 16).** Pacing named as a craft device ("slow burn as a technique", "methodical pacing", "frenetic cutting") is Cat 16. Pacing named as a felt experience ("slow", "relaxed", "rushed") can be Cat 22 via `tension_adrenaline` or `sensory_load`. The upstream dispatch should already have split these; trust its call on which framing was intended.
- **Sensitive content — disturbance (Cat 18).** Cat 18 is specific content-axis intensity for ruling out or softening ("not too bloody", "no gore"). Cat 22 uses `disturbance_profile` for the broader felt tone ("unsettling", "disquieting") rather than a content ceiling. If the requirement is about acceptable content intensity rather than overall mood, no-fire and let Cat 18 handle it.

## When to no-fire

Return `should_run_endpoint: false` when:

- The qualifier names a post-viewing aftertaste or ending type rather than a during-viewing feel.
- The qualifier names scale or scope rather than tone or cognitive demand.
- The qualifier is a craft-device pacing claim rather than a felt pacing claim.
- The qualifier sets a content-intensity ceiling on a specific axis rather than naming a tonal experience.
- The phrase is too vague to point at any viewer_experience sub-field ("a vibe", "a feel") with no concrete tonal, cognitive, or tension signal attached.

No-fire is always better than stuffing a vague tone word into every sub-field. A weak query vector underperforms no query at all.
