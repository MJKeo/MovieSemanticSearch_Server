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

## Density: 5–10 terms per active sub-field

The ingest-side `viewer_experience` text routinely emits 5–10 phrases per active section in user-search vernacular — slang, paraphrases, and TRUE synonyms. The query-side body should match. A body with 3 dictionary adjectives and zero negations sits in a thinner part of the embedding neighborhood than the documents we're trying to retrieve.

Per active sub-field, anchor on adjectives but add 1-2 vernacular synonyms and 1-2 user-search-phrase variants. "cozy, warm, gentle" is OK; "cozy, warm, gentle, comfort movie, feel good watch" is closer to the ingest neighborhood.

## True synonyms only — substitution test

When expanding a term list, every term must pass the substitution test:

> **"Could I show this term to the user instead of their original word, and would they say yes, that's the same thing?"**

- ✓ "haunting" → "haunting, lingering, sticks with you, stays with you" — true synonyms; same residue.
- ✓ "uplifting" → "uplifting, inspiring, hopeful, feel-good" — same lift.
- ✗ "haunting" → "haunting, eerie, creepy, supernatural" — eerie/creepy is a different feel; supernatural is a content claim. Drift hurts retrieval.
- ✗ "bittersweet" → "bittersweet, tragic, melancholic" — tragic is stronger; melancholic is adjacent.

If the test fails, drop the term. Drift terms degrade cosine similarity against the films the user actually wants.

## Negations default-populate — same direction as terms

`terms` and `negations` BOTH POINT AT THE SAME RETRIEVAL TARGET. They're complementary phrasings of the same concept, not opposites. `"happy"` and `"not sad"` are the same idea — putting both in the body weights that concept in the embedded vector.

The mechanical rule: `terms` never has `not`/`no` prefix; `negations` always does. Both fields cluster on the same side of the embedding.

Correct pairings:

- Feel-good body: `terms=["uplifting", "joyful", "warm"]` + `negations=["not depressing", "not bleak"]`.
- Earnest tone body: `terms=["earnest", "heartfelt", "sincere"]` + `negations=["not campy", "not ironic", "not mean spirited"]`.
- Cozy tension body: `terms=["relaxed", "chill", "low stakes"]` + `negations=["not stressful", "not edge of your seat"]`.
- Gory body: `terms=["gory", "bloody", "graphic violence"]` + `negations=["not peaceful", "not for kids", "not gentle"]`.
- Non-gory body: `terms=["light scares", "tame violence", "restrained"]` + `negations=["no gore", "not too gory", "not bloody"]`.

Contradictory pairings — these break retrieval. NEVER emit:

- `terms=["gory"]` + `negations=["not too gory"]` — same axis, opposite directions; contradicts itself.
- `terms=["happy"]` + `negations=["not happy"]` — contradicts.
- `terms=["uplifting"]` + `negations=["not uplifting"]` — contradicts.

**For each active sub-field, default-populate 1–3 negations that reinforce the direction the terms already point.** Suppress only when the section is barely populated.

When the user names a boundary ("not too gory") and it's the trait's central ask:
- If "not too gory" is its own trait (typical Step 2 split), it gets `polarity=negative` upstream and the body searches AFFIRMATIVELY for gory films (`terms=["gory", "bloody"]`, `negations=["not peaceful", "not for kids"]`). The orchestrator inverts the score.
- If "not too gory" is a within-trait boundary on a positive-finding body (e.g. "campy slasher but not too gory" as one inseparable concept), the body searches for the affirmative complement: `terms=["light scares", "tame violence"]` + `negations=["no gore", "not too gory", "not bloody"]`. Both fields cluster on the non-gory side.

The body NEVER inverts. Polarity flipping happens at the trait level, applied by the orchestrator.

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
