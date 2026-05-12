# Additional objective notes — Story / thematic archetype

## Category Target

Overarching story shape, theme, or character-arc trajectory: grief,
redemption, found family, coming-of-age, man-vs-nature, revenge,
underdog, survival, moral compromise, post-apocalyptic. Ask:
"What kind of story is it?"

## Coverage Decision

- Preferred: Keyword when a registry tag directly covers a
  binary-framed story shape or theme.
- Fallback: Semantic when no tag covers the concept.
- Spectrum rule: graded wording ("kind of", "leans", "hint of",
  "thread of") uses Semantic even when a tag exists — binary tags
  are the wrong tool for degree.
- Split: Keyword for the covered binary shape AND Semantic for an
  uncovered thematic qualifier in the same category call.

## Semantic plot_analysis is the only target

When this category routes to Semantic, it routes exclusively to
the `plot_analysis` space. `primary_vector` is always
`plot_analysis`. Do NOT populate viewer_experience,
narrative_techniques, plot_events, watch_context, production, or
reception. Spreading the signal across spaces pulls in films that
*feel* like the theme or *use techniques associated with* the
theme rather than films that ARE the theme, and dilutes the match.

The two specific failure modes this rule blocks:

- **Drift to viewer_experience.** Thematic words carry emotional
  associations ("grief" → sad, "redemption" → uplifting), so it
  is tempting to translate the theme into tonal vernacular and
  emit `emotional_palette.terms = ["grief", "heavy", "melancholy"]`.
  That retrieves films that FEEL heavy — not films that are ABOUT
  grief. A grim revenge thriller and a tender mother-daughter
  grief drama share the feel but not the theme; the embedding
  must land on the theme, so the signal goes to plot_analysis.
- **Drift to narrative_techniques.** "Redemption arc" sounds like
  craft, but `character_arcs` lives on both plot_analysis (the
  arc as story content) and narrative_techniques (the arc as a
  delivery device). For this category the arc is content — what
  the story IS, not how it is told — so it goes to plot_analysis.

## Sub-field selection within plot_analysis

The `plot_analysis` body has six sub-fields. Populate the ones the
trait actually grounds; empty sub-fields are valid and expected.

- **`elevator_pitch`** — log-line capsule of the archetype, ≤6
  words ideal, generic, no proper nouns. "a redemption arc",
  "man vs nature survival", "an underdog story".
- **`plot_overview`** — 1-3 sentence thematic plot summary in
  generic terms. Restate the archetype using role-nouns
  ("a flawed protagonist", "a group of strangers", "a society")
  instead of names.
- **`genre_signatures`** — fires when the archetype carries a
  recognizable subgenre signature ("post-apocalyptic survival",
  "redemption drama", "underdog sports drama"). Stays empty for
  purely thematic asks like "about grief".
- **`conflict_type`** — fires when the archetype is naturally
  expressed as "X vs Y": "man vs nature", "man vs self", "man vs
  society", "man vs fate". Stays empty otherwise.
- **`thematic_concepts`** — fires for thematic-territory asks:
  "grief", "redemption", "moral compromise", "the price of
  ambition". Multi-word concept phrases OK.
- **`character_arcs`** — fires when the archetype is a character
  trajectory: "coming of age", "fall from grace", "redemption",
  "loss of innocence".

## Cross-field repetition is mandatory

The ingest side deliberately repeats the load-bearing thematic
term across `elevator_pitch` / `plot_overview` / `thematic_concepts`
/ `character_arcs`. The query side MUST do the same so the cosine
match lands in the same neighborhood. For a "redemption arc" query:

- `elevator_pitch`: "a redemption arc"
- `plot_overview`: "A flawed protagonist works toward redemption…"
- `thematic_concepts`: ["redemption", "atonement", "moral repair"]
- `character_arcs`: ["redemption"]

If the central concept appears only once, the embedded vector is
under-weighted on it relative to the ingest text and the match is
weaker. Reuse the same load-bearing word verbatim.

## Density and register

- Generic only — no proper nouns, no character names, no
  movie-universe specifics. Replace any named character with a
  role-noun ("the protagonist", "a grieving mother", "an aging
  hitman").
- Use everyday, human-world thematic vocabulary, not academic
  literary-criticism vocabulary.
- Term lists: 2-5 entries per active list, true paraphrases of
  the same concept — apply the substitution test ("Could I show
  this term to the user instead of their original word, and would
  they say yes, that's the same thing?"). "Redemption" → "atonement"
  / "moral repair" passes. "Redemption" → "violence" / "tragedy"
  fails as drift.

## Boundaries with nearby categories

- Concrete focal subject ("about JFK", "about the moon landing")
  → CENTRAL_TOPIC.
- Static character types ("anti-hero", "femme fatale") →
  CHARACTER_ARCHETYPE.
- Genre identity without thematic claim ("horror", "neo-noir") →
  GENRE.
- Literal event sequence ("a heist crew gets double-crossed") →
  PLOT_EVENTS.
- Tonal / felt experience ("dark", "uplifting", "haunting") →
  EMOTIONAL_EXPERIENTIAL.

## No-fire

No-fire on vague depth language ("deep movies", "meaningful
films"), static character labels, concrete subjects, genre/form
requests with no thematic content, or pure tonal asks.
