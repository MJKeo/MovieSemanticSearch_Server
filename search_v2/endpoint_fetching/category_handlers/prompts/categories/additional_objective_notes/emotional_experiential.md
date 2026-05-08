# Emotional / experiential - additional objective notes

## Target

Handle how the movie feels, what it does to the viewer, how it lingers, and ending-aftertaste framings: tone, mood, pacing-as-experience, cognitive demand, comfort, catharsis, feel-good, haunting, gut-punch, happy ending, sad ending, twist ending.

## Semantic Decision

This category fires a single semantic call. Pick the vector spaces whose ingest-side text actually carries the evidence the user named.

Ask:
- Is this during-viewing feel? Use `viewer_experience` for tone, mood, tension, disturbance, sensory load, cognitive demand, pacing feel, or emotional palette.
- Is this a self-experience goal or comfort/gateway pull? Use `watch_context` for "make me cry", "cheer me up", "mindless", "comfort watch", or "good first X".
- Is this a reviewer/audience label for felt impact? Use `reception` for labels like tearjerker, crowd-pleaser, comfort rewatch, unforgettable, or divisive emotional effect.
- Is this an ending type or aftertaste? Use `viewer_experience.ending_aftertaste`.

Use only spaces with concrete evidence. Empty sub-fields are better than padded vectors.

## Boundary Checks

- Concrete situations such as date night, rainy Sunday, or long flight belong to Viewing occasion.
- Genre labels belong to Genre.
- Mid-story devices such as nonlinear timeline or unreliable narrator belong to Narrative devices.
- Content ceilings such as no gore or not too graphic belong to Sensitive content.
- Broad reputation such as classic or underrated belongs to Cultural status.

## No-Fire

No-fire when the target is only a situation, genre, content rule, or structural device, or is a vague "vibe" with no concrete emotional, experiential, or ending signal.

## Endpoint coverage breadth

The category's ingest-side signal is spread across `viewer_experience` (during-viewing tone), `watch_context` (self-experience motivations like "make me cry" / "comfort watch"), and `reception` (audience labels like "tearjerker"). Most rich emotional/experiential traits land on at least two of these in parallel — they reinforce each other on the same slice rather than redundantly covering it.

Single-space coverage is the under-recall failure mode. When you populate `viewer_experience` only and skip `watch_context` / `reception`, the trait misses the films whose ingest signal lives on the sibling spaces. Default to populating every space whose ingest text genuinely carries the felt-effect signal; skip only the spaces that have nothing useful to add.

## Body authoring style

The `viewer_experience` body should follow this category's density and synonym rules — see the dedicated viewer_experience category note for the 5–10 terms / section target, the true-synonym substitution test, and the default-populate negations rule. Those apply here too whenever this category's call routes to viewer_experience.
