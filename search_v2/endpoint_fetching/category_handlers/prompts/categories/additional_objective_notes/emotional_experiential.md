# Emotional / experiential - additional objective notes

## Target

Handle how the movie feels, what it does to the viewer, how it lingers, and ending-aftertaste framings: tone, mood, pacing-as-experience, cognitive demand, comfort, catharsis, feel-good, haunting, gut-punch, happy ending, sad ending, twist ending.

## Semantic Decision

Always author the semantic read first.

Ask:
- Is this during-viewing feel? Use `viewer_experience` for tone, mood, tension, disturbance, sensory load, cognitive demand, pacing feel, or emotional palette.
- Is this a self-experience goal or comfort/gateway pull? Use `watch_context` for "make me cry", "cheer me up", "mindless", "comfort watch", or "good first X".
- Is this a reviewer/audience label for felt impact? Use `reception` for labels like tearjerker, crowd-pleaser, comfort rewatch, unforgettable, or divisive emotional effect.
- Is this an ending type or aftertaste? Use `viewer_experience.ending_aftertaste`; add keyword only for a clean ending tag.

Use only spaces with concrete evidence. Empty sub-fields are better than padded vectors.

## Keyword Augmentation

Fire keyword only for a canonical emotional or ending signal directly named or clearly entailed by the query. Examples of valid signal types: feel-good effect, tearjerker effect, happy/sad/open/cliffhanger/twist ending.

Ask:
- Does a registry member definition name this exact felt effect or ending shape?
- Would the tag catch a crisp binary signal semantic may blur?
- Can it fire without converting mood into genre or plot?

Do not add genre, audience, or content tags as emotional proxies.

## Boundary Checks

- Concrete situations such as date night, rainy Sunday, or long flight belong to Viewing occasion.
- Genre labels belong to Genre.
- Mid-story devices such as nonlinear timeline or unreliable narrator belong to Narrative devices.
- Content ceilings such as no gore or not too graphic belong to Sensitive content.
- Broad reputation such as classic or underrated belongs to Cultural status.

## No-Fire

No-fire when the target is only a situation, genre, content rule, structural device, or vague "vibe" with no concrete emotional, experiential, or ending signal.

## Endpoint coverage breadth

This category's ingest-side signal is spread across `viewer_experience` (during-viewing tone), `watch_context` (self-experience motivations like "make me cry" / "comfort watch"), `reception` (audience labels like "tearjerker"), and the keyword registry (binary effect tags like FEEL_GOOD / TEARJERKER / HAPPY_ENDING). Most rich emotional/experiential traits land on at least three of these in parallel — they reinforce each other on the same slice rather than redundantly covering it.

Single-endpoint coverage is the under-recall failure mode. When you fire `viewer_experience` only and skip `watch_context` / `reception`, the trait misses the films whose ingest signal lives on the sibling spaces. Default to firing every endpoint whose walk surfaced a real candidate; abstain only on endpoints whose walk genuinely had nothing useful to add.

## Body authoring style

When firing the semantic endpoint, the `viewer_experience` body should follow this category's density and synonym rules — see the dedicated viewer_experience category note for the 5–10 terms / section target, the true-synonym substitution test, and the default-populate negations rule. Those apply here too whenever this category's call routes to viewer_experience.
