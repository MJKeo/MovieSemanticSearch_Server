# Few-shot examples - Sensitive content

<example>
Input:
```xml
<retrieval_intent>Avoid movies where an animal dies.</retrieval_intent>
<expressions><expression>nothing where the dog dies</expression></expressions>
```
Reason first:
- Concrete content axis: animal death.
- Binary hard exclusion.
- Registry flag directly covers the axis.

Expected: fire keyword for the animal-death flag only. Do not fire metadata;
rating is not the issue. Do not soften this into semantic intensity.
</example>

<example>
Input:
```xml
<retrieval_intent>Find a sad movie where the dog actually dies.</retrieval_intent>
<expressions><expression>where the dog actually dies</expression></expressions>
```
Reason first:
- Concrete content axis: animal death.
- Binary hard inclusion.
- The emotional effect is sibling intent, not this category's payload.

Expected: fire keyword for the animal-death flag only, with parent polarity
positive. Do not fire semantic just because the result may be sad.
</example>

<example>
Input:
```xml
<retrieval_intent>Prefer horror movies that are not too bloody.</retrieval_intent>
<expressions><expression>not too bloody</expression></expressions>
```
Reason first:
- Concrete content axis: blood / gore.
- "Not too" makes it a gradient, not a hard ban.
- The trait's central ask is the affirmative complement — a horror movie
  whose violence is restrained — so the body searches FOR that complement,
  not for gore with a negative polarity flip.
- No clean binary registry flag is available from the phrasing alone.

Expected: fire semantic on `viewer_experience.disturbance_profile` with the
body pointing at restrained-violence films. Both `terms` and `negations`
cluster on the SAME side — the non-gory side — of the embedding:
`terms=["light scares", "tame violence", "restrained", "tasteful"]`
paired with `negations=["no gore", "not too gory", "not bloody",
"not graphic"]`. Polarity stays positive because the body is
already searching for the films the user wants.

Do not force a genre tag as a content flag. Do not use metadata for a
specific axis. Do NOT emit `terms=["gory"]` + `negations=["not too gory"]`
— that is a self-contradicting body on the same axis.
</example>

<example>
Input:
```xml
<retrieval_intent>Find an action movie with family-friendly intensity.</retrieval_intent>
<expressions><expression>family-friendly intensity</expression></expressions>
```
Reason first:
- The target is global content intensity, not audience packaging.
- The surface implies a rating ceiling.
- No specific content axis is named.

Expected: fire metadata for the maturity ceiling only. Keyword FAMILY belongs
to Target audience, not this content-intensity slice.
</example>

<example>
Input:
```xml
<retrieval_intent>Find something fun, nothing heavy.</retrieval_intent>
<expressions><expression>nothing heavy</expression></expressions>
```
Reason first:
- "Heavy" is vague mood/weight.
- No concrete content axis, rating ceiling, or disturbance gradient is named.

Expected: no-fire for this category. Experiential tone may own the lightness
preference.
</example>

<example>
Input:
```xml
<retrieval_intent>Avoid movies with graphic torture scenes.</retrieval_intent>
<expressions><expression>no graphic torture</expression></expressions>
```
Reason first:
- Concrete content axis: torture, framed as a SPECIFIC on-screen event
  ("graphic torture", "scenes" implied by the phrasing).
- Exclusion ask — the trait carries `polarity=negative` upstream. The
  semantic body still searches AFFIRMATIVELY for torture content; the
  orchestrator inverts the score downstream.
- Decision rule clauses walked in order: no craft-evaluation framing
  ("famous for", "gratuitous", "tasteful") → reception does NOT apply.
  A specific event class IS named (torture scenes) → plot_events fires.
  Stop there — viewer_experience does NOT also fire, because that would
  pull films that feel torture-disturbing without actually depicting
  torture, which is the wrong retrieval target for a content-presence
  ask.

Expected output:
- Keyword: fires ONLY if the registry has a flag that directly names
  torture or extreme on-screen violence. Polarity negative. Do NOT
  activate a `horror`, `slasher`, `dark`, or `extreme-cinema` flag from
  the word "torture" — that is genre adjacency, not a direct content
  match. If no flag directly covers the axis, leave keyword silent and
  let semantic carry the ask.
- Semantic: single space — `plot_events.plot_summary` in MOTIF shape:
  `"the torture. a torture scene. and then more torture. another torture
  sequence."` That is the entire body — no fabricated protagonist, no
  invented setting, no manufactured story arc. The motif fragments
  retrieve films whose plot summaries name torture as a recurring beat.
- Do NOT also fire `viewer_experience.disturbance_profile` and do NOT
  fire `reception.criticized_qualities`. Both would dilute the match
  toward films with the wrong signal (atmospheric dread without the
  named event; critical evaluation rather than event presence).
</example>

<example>
Input:
```xml
<retrieval_intent>Find horror movies famous for over-the-top, gratuitous gore.</retrieval_intent>
<expressions><expression>over-the-top, gratuitous gore</expression></expressions>
```
Reason first:
- Concrete content axis: gore. Inclusion ask, polarity positive.
- Decision rule clauses walked in order: "famous for" + "gratuitous" is a
  textbook craft-evaluation framing — the user is naming the EVALUATIVE
  REPUTATION around how the film handles gore, not just asking whether
  gore is present. The reception clause fires first.
- Tie-breaker confirms: even though gore is a namable event class (which
  would point at plot_events on its own), the "famous for + gratuitous"
  wrapper routes the entire ask to reception. Once reception fires,
  plot_events and viewer_experience do NOT also fire.

Expected output:
- Semantic: single space — `reception.criticized_qualities=["gratuitous
  gore", "excessive violence", "shock value", "over-the-top splatter"]`.
  Adjective+noun craft-execution critiques, 1-3 words each. The terms
  live in `criticized_qualities` even though the USER wants these films
  — "gratuitous" is a negative-evaluation framing on the ingest side,
  and the reception list tracks the CRITICAL framing, not the user's
  preference. Polarity stays positive — the orchestrator does NOT
  invert.
- `reception_summary` stays null unless the user names a whole-work
  reception shape (e.g. "divisive", "cult-classic gore"). The aspect-
  level critique is what the user grounded.
- Do NOT also fire `plot_events.plot_summary` and do NOT fire
  `viewer_experience.disturbance_profile`. Either would muddy the
  retrieval target — the user asked about evaluative reputation, not
  about gore presence in general or the during-viewing feel.
- Do NOT emit subject-matter nouns ("splatter film", "torture porn") in
  reception — those are sub-genre labels the schema register rejects.
  Every reception term must name an evaluative judgment of how the
  content is handled.
</example>
