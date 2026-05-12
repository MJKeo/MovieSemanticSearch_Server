# Additional objective notes - Sensitive content

## Target

Fire for concrete content axes or content-intensity ceilings: gore, blood,
graphic violence, nudity, sexual content, strong language, drug use, animal
death, or rating-level intensity.

## Decision Questions

- What content axis or rating ceiling is explicit?
- Is the user requiring presence, excluding presence, or dialing intensity?
- Is the ask binary/hard, or gradient/soft?
- Which candidate endpoints add real signal? Address each one before
  committing.

## Endpoint Fit

- Keyword: binary registry content flags. Fires ONLY when a registry definition
  names EXACTLY the content axis the user named. Genre adjacency does NOT
  count: "graphic torture" does not activate `horror`, `slasher`, `dark`, or
  `extreme-cinema` flags — only a `torture` or `extreme-violence` flag if one
  exists in the registry. "Lots of nudity" does not activate `romance` or
  `erotic-thriller`. If no registry flag directly names the axis, Keyword
  stays silent and Semantic carries the whole ask. The cost of a false-match
  keyword is high — it elevates whole genres on a precise content question.
- Metadata: global rating ceiling only. Use for "family-friendly intensity" or
  "nothing above PG-13." Do not use it for a specific axis like gore.
- Semantic: pick EXACTLY ONE vector space per trait. Multi-space firing for
  this category muddies the retrieval target — each space points at a
  different KIND of signal (event presence vs. felt intensity vs. critical
  evaluation) and combining them dilutes the match. Use the decision rule
  below to pick the one space whose ingest text most directly carries the
  user's ask.

### Semantic single-vector decision rule

Walk these three "fire when…" clauses in order. The FIRST one that fits
genuinely is the vector to populate; do not also populate the others.

1. **`reception.praised_qualities` / `reception.criticized_qualities`** —
   fire when the user invokes a CRAFT-EVALUATION framing of how the content
   is HANDLED, not just whether it appears.
   - Trigger signals: craft-judgment adjectives ("gratuitous", "exploitative",
     "tasteful", "restrained", "earned", "shock value", "responsible",
     "irresponsible", "tactful"). OR framing wrappers like "famous for /
     known for / criticized for / praised for / acclaimed for" + content
     axis.
   - Example asks: "famous for over-the-top gore", "criticized as
     exploitative", "tasteful handling of sexual violence", "praised for
     restraint around violence".
   - Body shape: `criticized_qualities` for negative-evaluation framings
     (e.g. ["gratuitous gore", "excessive violence", "shock value"]);
     `praised_qualities` for positive ones (e.g. ["tasteful handling",
     "restrained depiction", "earned weight", "discretion"]). Adjective+noun
     craft-execution terms, 1-3 words each. NEVER bare content nouns
     ("nudity", "torture porn") — those are subject-matter labels, not
     evaluative judgments; the schema register rejects them.
   - Polarity orientation: which list the term lives in tracks the CRITICAL
     framing, not the user's preference. "Famous for gratuitous gore" — the
     user wants these films but `criticized_qualities` is still the right
     bucket because "gratuitous" is a negative-evaluation framing on the
     ingest side.

2. **`plot_events.plot_summary` (motif shape)** — fire when the user names
   a SPECIFIC ON-SCREEN EVENT or scene type whose presence in the film is
   the retrieval target.
   - Trigger signals: event nouns paired with "scene(s)", "on-screen",
     "depicted", "shown", "where it happens". OR implicit event framing
     ("where the dog dies", "with nudity in it"). Specific event classes
     the user can name: nude scenes, animal death, on-screen torture, drug
     use, sex scenes, suicide depiction, sexual assault, graphic killings,
     self-harm.
   - Example asks: "no on-screen drug use", "movies where someone gets
     tortured", "no animal deaths", "avoid graphic torture scenes".
   - Body shape: motif fragments per [semantic.md plot_events motif
     rules](../../endpoints/semantic.md): `"torture. a torture scene. then
     more torture. another torture sequence."` Retrieves films whose plot
     summaries name the event as a recurring beat. DO NOT fabricate plot
     detail (no protagonists, no settings, no outcomes); the motif shape
     is the entire body.
   - Why not viewer_experience: a film can FEEL torture-disturbing without
     ever depicting torture on screen (implied off-screen violence, tense
     interrogation atmospherics). For content-PRESENCE asks, that's the
     wrong retrieval target.

3. **`viewer_experience.disturbance_profile`** — fire when the user names a
   GRADIENT FELT INTENSITY without pinning to a specific event class.
   - Trigger signals: intensity adjectives unattached to event nouns ("not
     too bloody", "disturbing", "unsettling", "intense", "brutal feel",
     "edgy", "white-knuckle"). Gradient qualifiers ("not too", "softer",
     "less graphic", "tone it down"). Whole-film intensity feel rather
     than scene presence.
   - Example asks: "not too bloody", "violent but not graphic", "disturbing
     but not brutal", "intense without crossing the line".
   - Body shape: terms+negations as a paired body where both fields cluster
     on the SAME side of the embedding (see Positive-presence section
     below). Add `sensory_load` as a secondary sub-field ONLY when the
     content axis is sensory in nature (strobing, body-horror visuals,
     sensory-overwhelm violence) — otherwise `disturbance_profile` alone
     carries the ask.
   - Why not plot_events: the user isn't naming a specific event class.
     Forcing a motif body would invent specificity the user did not
     express, and the motif fragments would have nothing concrete to
     repeat.

### Tie-breakers

- Specific event named alongside an intensity adjective → plot_events wins.
  "Brutal torture scenes" → plot_events motif. The "brutal" is a description
  of the named scenes, not a separate axis the user wants matched on.
- "Famous for / known for / criticized for" wrapper around a specific event
  → reception wins, NOT plot_events. "Famous for graphic torture" → reception
  (the user is naming the evaluative reputation, not asking whether torture
  appears).
- Pure intensity word with no event class and no evaluation framing →
  viewer_experience. "Not too intense" → viewer_experience.

## Positive-presence and negation direction

This category routes more "avoid X" / "no X" / "without X" phrasings than any
other, which makes it the highest-risk surface for inverting the semantic body.
The rule is mechanical and absolute: the body ALWAYS searches affirmatively
for the content axis named, regardless of whether the user wants to include
or exclude it. Trait polarity flips upstream; the orchestrator inverts the
score downstream. The body never inverts.

When the user says "no gore" / "avoid graphic torture" / "without nudity":

- The trait carries `polarity=negative` upstream — that piece is already done.
- The body for this category searches AFFIRMATIVELY for gory / torture / nude
  films, exactly the same way it would if the user wanted to FIND them.
- The orchestrator multiplies the score by -1 downstream so high-match films
  get penalized in the final ranking.

`terms` and `negations` both point at the SAME retrieval target. They are
complementary phrasings of the same concept, not opposites. "hot" pairs with
"not cold" — never with "not hot". The mechanical rule: `terms` never carries
`not`/`no` prefix; `negations` always does. Both fields cluster on the same
side of the embedding and reinforce each other.

Worked pairings for sensitive content:

- Gory body (whether the user wants gore OR wants to AVOID gore — both produce
  THIS body upstream):
  `terms=["gory", "bloody", "graphic violence"]` +
  `negations=["not peaceful", "not for kids", "not gentle"]`.
- Non-gory body (only when the trait's central ask is the affirmative
  complement — e.g. "campy slasher but not too gory" treated as one
  inseparable concept):
  `terms=["light scares", "tame violence", "restrained"]` +
  `negations=["no gore", "not too gory", "not bloody"]`.

Contradictory pairings — NEVER emit. They collapse the embedding to nothing:

- `terms=["gory"]` + `negations=["not too gory"]` — opposite directions on
  the same axis; self-contradicting.
- `terms=["nudity"]` + `negations=["no nudity"]` — same axis, opposite poles.
- `terms=["graphic violence"]` + `negations=["not graphic violence"]` —
  literal inversion.

If the user's words feel like they want the body to "search for the absence",
the fix is at the trait level (polarity negative), not inside the body. Build
the body that retrieves movies HAVING the target; let the orchestrator do the
inversion.

## Hard vs Soft

- "No / without / zero" + clean binary flag -> hard exclusion through Keyword.
- "With / where it has" + clean binary flag -> hard inclusion through Keyword.
- "Not too / not overly / less graphic" -> semantic intensity gradient.
- "PG-13 or lower / family-friendly intensity" -> metadata ceiling.

Parameters describe the content or rating surface itself. Parent polarity
carries whether that surface helps or hurts.

## Boundaries

- Pure audience packaging ("for kids", "family movie") belongs to Target
  audience.
- Vague mood or weight ("nothing heavy", "something light") belongs to
  experiential tone unless a concrete content axis is named.
- Do not infer content axes from outside film knowledge or genre assumptions.

## No-Fire

Return no endpoint payloads when the target names no concrete content axis, no
rating ceiling, and no disturbance gradient grounded in the words provided.
