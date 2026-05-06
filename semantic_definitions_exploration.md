# Semantic Definitions Exploration

A side-by-side study of how the seven non-anchor vector spaces are
defined on the ingestion / metadata-generation side versus how the
search side instructs an LLM to author query-side bodies for them.
The end of the doc compares the two and proposes concrete fixes to
the search-side guidance so query-side text lands in the same
neighborhood as the document-side embeddings.

> The eight named vectors include `dense_anchor`, but anchor is
> intentionally not searched directly under V3 — this report covers
> the seven retrievable spaces only:
> `plot_events`, `plot_analysis`, `viewer_experience`, `watch_context`,
> `narrative_techniques`, `production`, `reception`.

---

## Part 1 — Ingestion / metadata-generation side (source of truth)

For each space below: what is generated, what the prompt actually
tells the generator to produce, what the assembled embedding text
looks like, and an excerpt from a real movie's stored metadata.

The "embedded text" examples below are reconstructed from the actual
SQLite ingestion records (`generated_metadata` table at
`ingestion_data/tracker.db`) by walking each `*Output.embedding_text()`
method in `schemas/metadata.py` and the per-space helpers in
`movie_ingestion/final_ingestion/vector_text.py`.

### 1. `plot_events` — what literally happens, in synopsis prose

**Source prompt:** `movie_ingestion/metadata_generation/prompts/plot_events.py`

Two branch-specific prompts, both producing a single
`plot_summary` field that becomes the embedded text verbatim
(lowercased, no labels).

- **Synopsis branch** (>= 2500-char IMDB synopsis available): condense
  the synopsis. Preserve "character NAMES, locations, concrete
  actions. Focus on EVENTS/FACTS, not themes. Only essential
  characters, 1-3 core conflicts." Capped at ~4000 tokens.
- **Synthesis branch** (no quality synopsis): consolidate the
  marketing overview + scattered user-written `plot_summaries` into
  a single chronological account. Heavy anti-hallucination
  scaffolding, framed explicitly as "text consolidation."

**Vector text format** (`PlotEventsOutput.embedding_text()`):

```
<plot_summary, lowercased, raw prose>
```

There is **no label**, no bulletization. The plot_summary becomes
the entire embedded string.

**Voice / register:**
- Long, dense **synopsis prose** — paragraph form, full sentences,
  past tense, third-person omniscient.
- **Names and proper nouns are present** ("Joseph Cooper", "Murphy",
  "the Endurance", "Gargantua") — generalization is *not* required.
- Spoiler-filled. Major plot beats, character arc transitions,
  thematic turning points, and the resolution.
- **Concrete events, no theme talk.** "He sacrifices himself inside
  a black hole" is fine; "the protagonist learns the value of
  sacrifice" is not.
- Token budget is large (3000-4000 tokens common).

**Real example (Interstellar, partial):**

> "In a near-future agrarian Earth ravaged by crop blight, former
> NASA pilot Joseph Cooper runs a farm with his children Murphy and
> Tom. Murphy believes a ghost is communicating with her; the
> 'ghost' turns out to be gravitational data that leads Cooper to a
> secret NASA base run by Professor John Brand. … Inside the black
> hole Cooper enters a multidimensional tesseract where time is
> navigable as spatial slices. He realizes future humans built this
> construct to transmit information…"

> The `plot_events` vector is the **only** space where named
> characters and proper-noun place names are expected on the
> ingest side. Every other space replaces them with generalized terms.

---

### 2. `plot_analysis` — what *kind* of story (genre, theme, conflict, arcs)

**Source prompt:** `movie_ingestion/metadata_generation/prompts/plot_analysis.py`

Six output fields:
- `genre_signatures` — 2-6 phrases, 1-4 words each. *Examples:*
  "epic space odyssey", "biblical passion drama", "buddy police mystery".
- `thematic_concepts` — 0-5 labels, 2-6 words. Generalized,
  human-world, thematic territory + moral message. *Examples:*
  "Love as guiding force", "Redemptive sacrificial suffering",
  "Identity shaped by image".
- `elevator_pitch` — ≤6 words, log-line style. *Examples:*
  "Parent's love fuels humanity-saving mission", "Visceral account
  of sacrificial suffering".
- `conflict_type` — 0-2 generalized "X vs Y" phrases. *Examples:*
  "survival of humanity vs planetary collapse", "career ambition
  vs personal relationships".
- `character_arcs` — 0-3 short generalized labels.  *Examples:*
  "sacrificial redemption", "from resentment to savior", "emergent
  heroism".
- `generalized_plot_overview` — 1-3 sentences. **Proper nouns
  forbidden.** Heavy thematic-term repetition encouraged ("almost
  comical how much the thematic concepts and elevator_pitch are
  emphasized").

**Vector text format** (`PlotAnalysisOutput.embedding_text()`):

```
elevator_pitch: <lowercased prose>
plot_overview: <lowercased prose, generalized, no proper nouns>
genre_signatures: <comma-joined, normalized terms>
conflict: <comma-joined>
themes: <comma-joined>
character_arcs: <comma-joined>
```

**Real example (Interstellar):**

```
elevator_pitch: parent's love fuels humanity-saving mission
plot_overview: in a near-future failing world, a former pilot accepts
  a risky space mission to secure humanity's survival, driven by
  parental love and scientific duty; the crew faces relativistic
  time loss, betrayal, and sacrifice, testing exploration as human
  identity and science for survival. the pilot sacrifices himself
  inside a black hole to transmit critical quantum data across
  time, enabling the grown daughter to complete the gravity
  solution and evacuate humanity, resolving temporal legacy
  through love as a guiding force.
genre_signatures: epic space odyssey, science driven adventure,
  emotional sci fi drama
conflict: survival of humanity vs planetary collapse, human
  curiosity exploration vs existential risk
themes: love as guiding force, science for survival, exploration
  as human identity, temporal causality and legacy
character_arcs: sacrificial redemption, from resentment to savior
```

**Voice / register:**
- Per-field labeled structure with newline separators.
- **Generic, human-world phrasing** — no "Cooper", just "a former
  pilot"; no "Endurance", just "a risky space mission".
- Mix of short terms (genre/conflict/themes/arcs) and one or two
  short prose blocks (elevator_pitch + overview).
- Heavy redundancy across fields — the prompt explicitly tells the
  generator to "really emphasize the thematic concepts throughout,
  repeating key thematic terms."

---

### 3. `viewer_experience` — what it *feels like* to watch

**Source prompt:** `movie_ingestion/metadata_generation/prompts/viewer_experience.py`

Eight paired sub-fields, every one of which is `terms` +
`negations`. Each section has its own concept guide and a list of
real-user-style example phrases. The prompt instructs the
generator to write "phrases like **search queries**, not
sentences", to keep them 1-5 words, to **deliberately include
synonym redundancy**, slang, and paraphrases, and to populate
negations users would actually type ("not too sad", "no jump
scares").

| Sub-field | Concept | Example terms (from prompt) |
|---|---|---|
| `emotional_palette` | Dominant felt emotions | "uplifting and hopeful", "tearjerker", "kept me guessing", "tears of joy", "depressing", "childhood nostalgia" |
| `tension_adrenaline` | Stress / suspense pressure | "edge of your seat", "chill", "white knuckle", "snoozefest", "slow burn suspense" |
| `tone_self_seriousness` | Movie's attitude toward itself | "earnest and heartfelt", "winking self aware", "deadpan humor", "campy", "so bad it's good" |
| `cognitive_complexity` | Mental effects, ease of follow | "thought provoking", "draining", "digestible", "straightforward" |
| `disturbance_profile` | Unsettling / fear flavor | "creepy and unsettling", "psychological horror", "gory", "body horror", "fucked up", "freaky" |
| `sensory_load` | Extreme physical-comfort sensory | "eye-straining", "ear-popping", "soothing", "quiet" — >90% empty |
| `emotional_volatility` | Tonal change over time | "tonal whiplash", "laugh then cry", "gets dark fast", "genre mash" |
| `ending_aftertaste` | The final emotion you leave with | "satisfying ending", "wrecked me", "haunting ending", "gut punch ending", "cliffhanger" |

**Vector text format** (`ViewerExperienceOutput.embedding_text()`):

```
emotional_palette: <terms>
emotional_palette_negations: <negations>
tension_adrenaline: <terms>
tension_adrenaline_negations: <negations>
... (each populated section)
```

Empty sections are **skipped entirely**, not emitted with empty
values.

**Real example (The Devil Wears Prada, tmdb_id 350):**

```
emotional_palette: feel-good, fun and light, comfort movie, uneasy
  undercurrent, bittersweet, enjoyment with frustration
emotional_palette_negations: not heartbreaking, not pure comedy,
  not uplifting all the way
tension_adrenaline: relaxed, low stakes, mild workplace tension
tension_adrenaline_negations: not edge of your seat, not high
  adrenaline
tone_self_seriousness: satirical, stylish and snarky, earnest at
  heart, performance-driven
tone_self_seriousness_negations: not campy, not mean spirited
cognitive_complexity: digestible, straightforward, lightweight
cognitive_complexity_negations: not confusing, not thought
  provoking
disturbance_profile_negations: not scary, no gore
sensory_load_negations: not overstimulating, not too loud
emotional_volatility: steady arc, gradual shift
emotional_volatility_negations: no tonal whiplash, not all over the
  place
ending_aftertaste: bittersweet ending, reflective finale,
  satisfying closure
ending_aftertaste_negations: not a bleak ending, not a cliffhanger
```

**Voice / register:**
- **Search-query language**, not movie-critic prose. Slang and
  vernacular are encouraged, including profanity ("fucked up",
  "gorefest"). The prompt is blunt: "Do NOT sanitize language —
  clean phrasing reduces recall against real user queries."
- **Synonym redundancy is the default** — multiple near-duplicates
  per section ("uplifting", "inspiring", "hopeful").
- **Negations are first-class** and populated in nearly every
  section, even when no boundary was named in the inputs — they
  describe what the movie is NOT, e.g. "no gore" on a romance.
- Compact phrases (1-5 words). Some sections have 5-10 terms.
- Per-section count grows or shrinks with input richness; sparse
  inputs → empty sections (sensory_load almost always empty).

---

### 4. `watch_context` — *why* and *when* someone would watch

**Source prompt:** `movie_ingestion/metadata_generation/prompts/watch_context.py`

The prompt forbids any plot detail in the input and frames the
task as pure experience/motivation extraction. Four sections, all
term-only (no negations):

| Sub-field | Concept | Example terms |
|---|---|---|
| `self_experience_motivations` | Self-focused emotional/psychological need | "need a laugh", "cathartic watch", "escape from reality", "test my nerves", "turn my brain off" |
| `external_motivations` | Value beyond viewing itself | "sparks conversation", "culturally iconic", "impress film snobs", "learn something new" |
| `key_movie_feature_draws` | Standout draws as "watch this if you want X" | "incredible soundtrack", "visually stunning", "hilariously bad dialogue", "over the top violence" |
| `watch_scenarios` | Real-world occasions | "date night movie", "solo movie night", "halloween movie", "stoned movie", "background at a party" |

The prompt explicitly endorses crude / explicit phrasing ("scared
shitless", "stoned movie", "cry your eyes out") to match how
real people search.

**Vector text format** (`WatchContextOutput.embedding_text()`):

```
self_experience_motivations: <terms>
external_motivations: <terms>
key_movie_feature_draws: <terms>
watch_scenarios: <terms>
```

`identity_note` is generated for chain-of-thought but is **NOT
embedded**.

**Real example (Super Mario Bros. Movie, tmdb_id 502356):**

```
self_experience_motivations: family movie night, feel-good kids
  movie, nostalgia trip, bright happy animation, laugh with the
  kids, smile-inducing movie
external_motivations: fan-service movie, show fellow fans, talk
  about callbacks, movie for gamers
key_movie_feature_draws: visually stunning animation, non-stop
  action, packed with gags, jack black voice
watch_scenarios: quick fun watch, kids birthday party movie,
  post-game hangout, cozy family night, matinee with kids
```

**Voice / register:**
- Same vernacular, search-query register as `viewer_experience`,
  but framed from the *viewer's intent* perspective rather than
  the *during-watch feel* perspective.
- 4-8 phrases per section is typical when inputs are rich.
- The motivation phrasing often uses second-person verbs ("test
  my nerves") or imperative pulls ("watch when in a bad mood").
- `key_movie_feature_draws` overlaps slightly with reception's
  praised/criticized aspects but is framed as "what you GET from
  watching" rather than "what critics liked."

---

### 5. `narrative_techniques` — *how* the story is told (craft, not content)

**Source prompt:** `movie_ingestion/metadata_generation/prompts/narrative_techniques.py`

Nine sub-fields, term-only, focused on storytelling craft:

| Sub-field | Concept | Example terms |
|---|---|---|
| `narrative_archetype` | Whole-plot label | "cautionary tale", "underdog rise", "revenge spiral", "heist blueprint", "whodunit mystery" |
| `narrative_delivery` | Temporal structure | "linear chronology", "non-linear timeline", "flashback-driven structure", "time loop structure" |
| `pov_perspective` | Audience viewpoint | "first-person pov", "third-person limited pov", "multiple pov switching", "unreliable narrator" |
| `characterization_methods` | How character is conveyed | "show don't tell actions", "backstory drip-feed", "character foil contrast" |
| `character_arcs` | How characters change | "redemption arc", "corruption arc", "coming-of-age arc", "tragic flaw spiral" |
| `audience_character_perception` | Deliberate audience positioning | "lovable rogue", "love-to-hate antagonist", "morally gray lead", "sympathetic monster" |
| `information_control` | Surprise / suspense / misdirection | "plot twist / reversal", "dramatic irony", "red herrings", "Chekhov's gun", "slow-burn reveal" |
| `conflict_stakes_design` | How stakes / pressure are built | "ticking clock deadline", "escalation ladder", "no-win dilemma", "Pyrrhic victory" |
| `additional_narrative_devices` | Catchall for structural / framing tricks | "cold open", "framed story", "found-footage presentation", "anthology segments", "fourth-wall breaks" |

Strong "evidence discipline" rules: empty sections are correct
when input doesn't ground them; **terms must be reusable across
ANY movie that uses the same device** ("ticking clock deadline" ✓,
"save-the-world final battle" ✗). Established craft labels stay
verbatim; do NOT "generalize away" technique names.

**Vector text format** (`NarrativeTechniquesOutput.embedding_text()`):

Same shape as `watch_context` — labeled per-section, comma-joined,
empty sections skipped.

**Real example (Interstellar):**

```
narrative_archetype: quest adventure
narrative_delivery: linear chronology with time jumps,
  time-manipulation climax
pov_perspective: third-person limited with ensemble shifts
characterization_methods: relationship-driven characterization,
  action-based reveal under duress, longitudinal character
  development
character_arcs: sacrifice redemption arc, corruption tragic flaw
  arc, coming-to-power arc
audience_character_perception: sympathetic protagonist, empathic
  family anchor
information_control: slow-burn reveal, plot twist reversal
conflict_stakes_design: escalation ladder, ticking-clock via time
  dilation
additional_narrative_devices: speculative multidimensional
  construct, science-as-metaphor framing, epic operatic scale
```

**Voice / register:**
- More technical / craft-vocab than `viewer_experience`, but still
  short tag phrases (1-6 words).
- Established technique names appear verbatim ("Chekhov's gun",
  "dramatic irony", "unreliable narrator") and should not be
  paraphrased away.
- 1-3 terms per active section; many sections legitimately empty
  for non-narrative content (documentaries, concert films).

---

### 6. `production` — where / how the film was physically made

**Source code:** `movie_ingestion/final_ingestion/vector_text.py:258-277`
(`create_production_vector_text`) and
`movie_ingestion/metadata_generation/prompts/production_techniques.py`.

**Two ingredients only**, both lean and concrete:

- `filming_locations` — direct from IMDB scraping. Up to **3**
  comma-joined raw place strings (lowercased), full proper-noun
  forms preserved. Skipped entirely for animation.
- `production_techniques` — LLM-FILTERED keyword list. The prompt
  is a **classifier**, not a generator: take the existing
  `plot_keywords` + `overall_keywords` and keep only animation
  modalities, sub-techniques, visual capture methods, and the
  one explicit found-footage exception. **Never invent, normalize,
  or rewrite terms** — return them exactly as written.

**Vector text format** (concatenation):

```
filming_locations: <up to 3 raw place strings, comma-joined,
                   lowercased>
production_techniques: <comma-joined, per-term normalized>
```

**Real examples:**

```
# Star Wars (tmdb_id 11)
filming_locations: tikal national park, guatemala, sidi driss
  hotel, matmata, tunisia, chott el djerid, nefta, tunisia
```

```
# An animated film
production_techniques: hand-drawn animation
```

```
# Mixed
filming_locations: monument valley, utah, usa, savannah, georgia,
  usa, beaufort, south carolina, usa
production_techniques: lens flare, cgi character in a live action
  movie
```

**Voice / register:**
- `filming_locations` are **proper nouns with full geographic
  specificity** — not abstracted to country level. "Tikal National
  Park, Guatemala" is the unit, not "Guatemala".
- `production_techniques` is a small, tag-shaped vocabulary
  derived from the IMDB keyword taxonomy.
- Empty `production_techniques` is the dominant case (default for
  any conventional live-action film).

---

### 7. `reception` — what critics / audiences said, by named aspect

**Source prompt:** `movie_ingestion/metadata_generation/prompts/reception.py`

Two zones (both prompt-internal), but **only the synthesis zone is
embedded**:

- Extraction zone (NOT embedded — used to feed Wave 2 generators):
  `source_material_hint`, `thematic_observations`,
  `emotional_observations`, `craft_observations`. Descriptive prose
  about technique / approach / subject — *what was done, not
  whether it worked*.
- **Synthesis zone (EMBEDDED):**
  - `reception_summary` — 2-3 evaluative sentences. "What did
    people think? What did they like, dislike, or find notable?"
  - `praised_qualities` — 0-6 short phrases (1-3 words). Adjective
    + attribute. Movie-agnostic. *Examples:* "compelling
    performances", "sharp dialogue", "evocative score".
  - `criticized_qualities` — 0-6 short phrases, same shape.

The prompt is strict: tags describe filmmaking **execution**, not
subject matter. "Sharp dialogue" yes; "intriguing premise" no.
Adjective+noun is the canonical shape.

**Vector text format** (`ReceptionOutput.embedding_text()` plus the
post-hoc `_reception_award_wins_text` line in
`movie_ingestion/final_ingestion/vector_text.py:280-327`):

```
reception_summary: <prose, lowercased>
praised: <comma-joined, normalized>
criticized: <comma-joined, normalized>
major_award_wins: <ceremony short names, prestige-ordered>
```

The `major_award_wins` line is appended deterministically from
structured award data — the LLM never generates it. It only emits
when the movie has prestige-tier wins (Oscars, Golden Globes,
BAFTA, Cannes, etc., **never Razzie**).

**Real example (Interstellar):**

```
reception_summary: audiences overwhelmingly praise interstellar
  for its ambitious themes, spectacular visuals, powerful emotional
  core, and hans zimmer's score, while some viewers raise concerns
  about plot convolution, pacing length, and aspects of the ending
  or scientific accuracy.
praised: spectacular cinematography, evocative score, emotional
  resonance, ambitious themes
criticized: convoluted plot, slow pacing, questionable scientific
  clarity
major_award_wins: academy awards, golden globes, bafta, critics
  choice
```

**Voice / register:**
- `reception_summary` is **evaluative prose** in the third
  person ("audiences praise X, while some criticize Y") — short,
  dense, but a real sentence.
- The two term lists are **adjective+noun aspect labels** — much
  more crystallized than `viewer_experience` terms. They sit
  closer to dictionary terms than to slang.
- `major_award_wins` is a tiny enumerated list of ceremony names,
  ordered by prestige.

---

## Part 2 — Search side

### What the search-side LLM is actually told

The semantic endpoint stage pulls together (per
`search_v2/endpoint_fetching/category_handlers/prompt_builder.py`):

1. **Shared role + vocabulary + input spec** chunks.
2. **Bucket** chunk (one of: `single_non_metadata_endpoint`,
   `preferred_representation_fallback`,
   `semantic_preferred_deterministic_support`,
   `audience_suitability_redundant_combo`,
   `character_franchise_fanout`).
3. **Endpoint** chunk — for semantic, this is
   `prompts/endpoints/semantic.md` verbatim.
4. **Category** chunk — additional notes + few-shot examples for
   the specific category that fired (e.g. `viewer_experience.md`,
   `plot_events.md`, `filming_location.md`).
5. The structured-output **schema** (`SemanticParameters` or
   `SemanticParametersSubintent`), whose Pydantic descriptors
   carry significant per-field guidance.

The output the LLM commits to is one or more `WeightedSpaceQuery`
entries — each is `{space, weight, content}` where `content` is a
`PlotEventsBody` / `PlotAnalysisBody` / `ViewerExperienceBody` /
`WatchContextBody` / `NarrativeTechniquesBody` / `ProductionBody`
/ `ReceptionBody` (defined in `schemas/semantic_bodies.py`,
mirroring the ingest-side `*Output.embedding_text()` shape).

### What the prompt says about each space

`semantic.md` defines all seven spaces in 1-2 paragraphs each.
A summary of what the search side actually tells the LLM:

- **plot_events** — "literal and concrete… chronological plot —
  actions, events, character-arc beats." `plot_summary` sub-field.
  Three example queries shown ("a heist that unravels…", "a lone
  survivor crosses…").
- **plot_analysis** — "categorical and thematic shape." Sub-field
  list given (`elevator_pitch`, `plot_overview`,
  `genre_signatures`, `conflict_type`, `thematic_concepts`,
  `character_arcs`). Three example queries.
- **viewer_experience** — "subjective and experiential."
  All eight `terms`+`negations` sub-fields enumerated. Negations
  are framed as "what it deliberately is NOT, populated only when
  the boundary actually matters."
- **watch_context** — "viewing situation, not internal content."
  Four sub-fields enumerated.
- **narrative_techniques** — "craft, not content." All nine
  sub-fields enumerated with one-shot example terms inline.
- **production** — "physical making, not storytelling."
  `filming_locations` + `production_techniques`.
- **reception** — "specific praised/criticized qualities and
  overall reception shape." `reception_summary`,
  `praised_qualities`, `criticized_qualities`.

### Body-authoring rules in `semantic.md`

The "Body authoring — match the ingest side" section gives this
guidance:

- **Term-list spaces** (viewer_experience, watch_context,
  narrative_techniques, production, reception term lists):
  "compact 2–4-word phrases."
- **Prose spaces** (plot_events `plot_summary`, plot_analysis
  `elevator_pitch`/`plot_overview`, reception `reception_summary`):
  "one or two dense sentences carrying the signal, not a paragraph."
- **Populate only sub-fields the aspects genuinely land in.** Empty
  sub-fields valid; padding forbidden.
- **Translate, don't echo.** Rewrite user-side phrasing into the
  space's ingest-side register.
- **Negations**: "only populated when the boundary actually matters
  and the input grounds it." Don't list negations to look thorough.
- **No numerics.**
- "Expansion pressure varies by space: viewer_experience often
  benefits from a few nearby tone/feeling synonyms… plot_events
  should stay close to the concrete situation described, phrased
  as compact prose; plot_analysis can translate into schema-native
  thematic/conflict language but stay tighter than
  viewer_experience."

### Per-category note files

`prompts/categories/additional_objective_notes/*.md` adds
category-specific directives. Highlights:

- `plot_events.md`: "dense synopsis register"; "Write what
  happens, not what the story means. Keep the body close to the
  given event facts. Do not invent motives, outcomes, twists, or
  character facts absent from retrieval_intent and expressions."
- `narrative_setting.md`: same `plot_events.plot_summary` space,
  but "phrase as setting description: 'set in...', 'takes place
  in...'." A setting-only synopsis fragment is valid.
- `viewer_experience.md`: "tonal feel does not genuinely land on
  plot_events, plot_analysis, or watch_context… and spreading
  weak signal across spaces dilutes the match." Explicit
  sub-field map (tonal aesthetic → emotional_palette +
  tone_self_seriousness; cognitive demand → cognitive_complexity;
  etc.). **Negations only when input names a boundary.**
- `viewing_occasion.md`: "compact `watch_context.watch_scenarios`
  short scenario phrases." Other watch_context sub-fields not
  highlighted.
- `filming_location.md`: "compact place names" only. Don't pad
  other bodies.
- `visual_craft_acclaim.md`: route to `reception` for praise,
  `production` for technique. Both only when both signals explicit.
- `music_score_acclaim.md`: "Phrase the reception body as compact
  praised qualities: 'iconic score', 'memorable soundtrack',
  'beloved theme'."

### Few-shot examples

`prompts/categories/few_shot_examples/*.md` contain concrete
expected JSON outputs. Examples I read:

- `viewer_experience.md` — three clean-fire examples:
  - "dark gritty crime movies" → `emotional_palette: [dark, bleak,
    grim]`, `tone_self_seriousness: [gritty, unglamorous, serious]`.
  - "cerebral sci-fi" → `cognitive_complexity: [cerebral,
    thought-provoking, intellectually demanding]`.
  - "whimsical cozy" → `emotional_palette: [cozy, warm, gentle]`,
    `tone_self_seriousness: [whimsical, playful, light-hearted]`.
  - All examples produce **exactly 3 short adjective-style
    terms per active sub-field. No negations. Every other
    sub-field is empty.**
- `cultural_status.md` — "modern classics" → `reception` body
  plus metadata priors. Body language is canon-stature framed.
- `emotional_experiential.md` — "make me cry" → fan-out to
  `watch_context`, `viewer_experience`, `reception` plus the
  TEARJERKER keyword tag. Confirms multi-space fan-out is
  expected for self-experience asks.

### What actually gets emitted (live `run_query_generation` runs)

I ran the production handler end-to-end on a few queries to
inspect the bodies. Highlights:

**Query: "a haunting bittersweet drama about grief that stays with
you"** — the `bittersweet` trait routed to `Emotional /
experiential` and emitted (compressed):

```json
"viewer_experience": {
  "emotional_palette": {"terms": ["bittersweet","melancholic","poignant"]},
  "tone_self_seriousness": {"terms": ["earnest","sincere"]},
  "emotional_volatility": {"terms": ["sadness to hope","pain and comfort"]},
  "ending_aftertaste": {"terms": ["bittersweet","cathartic","uplifting"]}
}
```

The `stays with you` trait emitted `viewer_experience` again with
`emotional_palette: [haunting, unforgettable, lingering]`,
`tone_self_seriousness: [profound, grave]`,
`emotional_volatility: [lasting impact]`,
`ending_aftertaste: [lingering aftertaste, haunting]`.

The `grief` trait routed to `Story / thematic archetype` and
emitted a `plot_analysis` body:

```json
{
  "elevator_pitch": "A grief-centered drama in which mourning is the core dramatic engine, …",
  "plot_overview": "The story is fundamentally organized around mourning…",
  "genre_signatures": ["grief drama","mourning narrative"],
  "conflict_type": ["man vs self","man vs grief"],
  "thematic_concepts": ["grief","mourning","bereavement","loss","absence"],
  "character_arcs": ["processing loss","working through mourning","emotional recovery after bereavement"]
}
```

**Query: "movies filmed in Iceland with non-linear timelines and
plot twists, told through unreliable narrators"** — the filming
location trait emitted:

```json
"production": {"filming_locations": ["Iceland"], "production_techniques": []}
```

The non-linear timeline trait fan-outs into a central
`narrative_techniques` body plus a SUPPORTING `plot_analysis` body.

**Query: "an over-the-top campy 80s slasher horror that is not too
gory"** — the `campy` trait emitted:

```json
"viewer_experience": {
  "emotional_palette": {"terms": ["campy","theatrical","flamboyant"]},
  "tone_self_seriousness": {"terms": ["self-aware","ironically detached","knowing","artificial"]},
  "sensory_load": {"terms": ["over-the-top","exaggerated","maximalist"]}
}
```

### Voice the search side actually produces

- **Tight, dictionary-style adjective lists.** Almost every
  emitted `viewer_experience.terms` list is 2-4 short, single-word
  to short-phrase adjectives.
- **Negations almost never fire.** Despite the schema supporting
  them, in three different queries (haunting bittersweet drama,
  campy slasher with no-gore boundary, etc.) the model emitted
  zero negations across the bodies — even when the user phrase
  ("not too gory") *does* name a boundary.
- **Plot bodies tend to be 1-2 sentence prose snippets.** The
  search side gravitates to a single elevator-pitch sentence plus
  short term lists. It does not produce dense, character-name-rich
  paragraph synopses.
- **No vernacular / slang.** Search-side output is uniformly
  professional vocabulary ("haunting", "lingering", "grave",
  "bittersweet"). Slang ("cry your eyes out", "white knuckle",
  "fucked up", "snoozefest") is **absent** from emissions in
  practice — the prompt examples don't model it.
- **No proper-noun place strings on the production side.** The
  search emits `filming_locations: ["Iceland"]` — clean but
  *coarser* than the ingest side's "Tikal National Park,
  Guatemala". Ingest-side filming_locations is comma-joined raw
  IMDB strings; search-side strings are generally the country or
  region only.

---

## Part 3 — Comparison: search-side gaps vs. ingest source of truth

The ingestion side is the source of truth — every embedded vector
in Qdrant was produced by the `embedding_text()` methods walked
above. For cosine search to work, the query-side text must land in
the same lexical / register neighborhood.

The search-side guidance is broadly correct on **structure**
(field names, fan-out logic, when not to fire) but consistently
**under-specifies the vocabulary, voice, density, and verbosity
of each space**. The result is that emitted bodies look like a
search engine's idea of "a clean adjective list" rather than a
cosine neighbor of the actual ingest text. Below, per space, what
the search side gets wrong relative to the source of truth.

### A. `viewer_experience` — biggest miss

**What ingestion produces:**
- 5-10 phrases per active section.
- Search-query-style vernacular: "edge of your seat", "kept me
  guessing", "tearjerker", "wrecked me", "snoozefest", "fucked
  up", "childhood nostalgia", "tears of joy".
- Slang and mild profanity are explicitly endorsed.
- **Synonym redundancy is the norm**, not a stretch goal — the
  ingestion prompt explicitly says "Repetition is encouraged, so
  long as the phrasing changes slightly."
- **Negations populated in nearly every section** even when no
  boundary was named — "no jump scares" appears on a romance,
  "no gore" appears on a documentary, etc. They function more as
  default complement signals than as user-side boundary markers.
- Phrases use second-person and first-person POV freely ("kept me
  guessing", "made me nauseous", "will give me nightmares").

**What search emits:**
- 2-4 short, single-word, dictionary-style adjectives per active
  section.
- Voice is critic-shaped, not user-shaped: "haunting,
  unforgettable, lingering", "profound, grave".
- Almost no slang or vernacular.
- Almost no negations (in three sample runs, zero, even on "not
  too gory").
- Almost no first-person or second-person framing.
- Few-shot examples enforce this minimalism by showing 3-term
  lists everywhere.

**Why this hurts retrieval:** the cosine distance between
"haunting, lingering" and an ingest vector that contains "wrecked
me, haunting ending, emotional hangover, gut-punch ending" is
larger than it would be if the search side emitted the same kinds
of vernacular synonyms. The query lands in a thinner adjective
neighborhood while documents are embedded in a thick
search-query phrase neighborhood.

**Specific search-side guidance that contradicts ingestion:**

- `semantic.md` says: term-list spaces want "compact 2-4-word
  phrases." Ingestion routinely uses **5-10 phrases** per active
  section — 2-4 is the **floor**, not the ceiling.
- `semantic.md` says: negations "only populated when the boundary
  actually matters and the input grounds it." Ingestion populates
  negations as the *default* for almost every section, even
  unprompted. The constraint is too restrictive — it's stripping
  a routine signal that ingest documents carry.
- `viewer_experience.md` (category note) explicitly maps eight
  sub-fields but does not show example terms. The few-shot
  examples then anchor the LLM on 3 short adjectives per
  sub-field, missing the redundancy and vernacular that ingest
  uses.
- The "translate, don't echo" rule is fine, but the *target
  register* is mis-specified — search-side translates user
  vernacular *out* of the body when ingestion explicitly *kept*
  vernacular in.

### B. `watch_context` — vernacular and 4-section pull both missed

**What ingestion produces:**
- 4-8 phrases per section.
- Crude/vernacular phrasing endorsed ("stoned movie", "scared
  shitless", "cry your eyes out", "watch when in a bad mood").
- All **four** sections populated whenever inputs allow —
  `external_motivations` and `key_movie_feature_draws` get 3-4
  terms each, not zero.
- Phrases often contain implicit second-person verbs and
  imperative pulls ("learn something new", "test my nerves").

**What search emits / is told to emit:**
- The category note for viewing occasion focuses on
  `watch_scenarios` only. The other three watch_context
  sub-fields (`self_experience_motivations`, `external_motivations`,
  `key_movie_feature_draws`) are mentioned in `semantic.md` but
  receive no category-level reinforcement, so they tend to stay
  empty.
- `key_movie_feature_draws` is a great match for "stacked cast",
  "incredible soundtrack", "Jack Black voice", but search side
  rarely emits to it.
- Crude phrasing is absent.

**Specific guidance fix:** when the category is a self-experience
goal (Cat 33), the search prompt should explicitly fan **all
four** watch_context sections, not just `watch_scenarios`. And
the few-shot examples need to model vernacular, second-person
phrasing.

### C. `narrative_techniques` — closer match, but undercounted

**What ingestion produces:**
- 1-3 terms per active section, but **multiple sections fire**
  simultaneously — Interstellar's vector touches 8 of 9 sections.
- Established craft labels appear verbatim ("Chekhov's gun",
  "ticking clock deadline", "found-footage presentation").
- `audience_character_perception` is populated with framing
  labels like "lovable rogue", "love-to-hate antagonist" — a
  whole register the search side rarely visits.

**What search emits:**
- Search bodies tend to populate 2-3 sections with 1-3 terms each.
- The handler does well on `narrative_delivery` and
  `information_control`. It rarely populates
  `audience_character_perception`,
  `conflict_stakes_design`, `characterization_methods`, or
  `additional_narrative_devices` even when the trait could.
- Established labels are sometimes paraphrased ("non-linear
  narrative" instead of the conventional "non-linear timeline" or
  "nonlinear chronology" — the ingest side uses both forms).

**Specific guidance fix:** the category note for narrative
devices should call out the under-utilized sections by name and
include examples that touch them.

### D. `plot_events` — register mismatch

**What ingestion produces:**
- Long synopsis prose (often 2000-4000 tokens).
- Named characters and proper-noun places ("Joseph Cooper",
  "Endurance", "Gargantua").
- Past-tense, third-person omniscient. Concrete scene-by-scene
  events.
- Multiple paragraphs.

**What search emits:**
- 1-2 sentence elevator-pitch-shaped plot snippets.
- Generic, archetypal phrasing ("a lone survivor crosses a
  frozen continent…"), almost always one sentence.

**Tension:** the search side is *correctly* not trying to
fabricate a movie's specific plot events from a user query (the
user can't supply names). But the *register gap* is large: a
single archetypal sentence at the query side maps poorly onto a
3000-token synopsis at the document side. Some embedding
density buffer is lost.

**Specific guidance fix:** even with no proper nouns available,
the `plot_summary` body should be 3-6 sentences of compact
synopsis prose with concrete generic actions, locations, and
turning points — not one log-line. The category notes for
`plot_events` and `narrative_setting` should ask for "compact
synopsis paragraphs in past-tense third-person", and the
few-shot examples should show paragraph-shaped outputs, not
single sentences.

### E. `plot_analysis` — generally aligned, but missing field-level redundancy

**What ingestion produces:**
- Six labeled fields, all generally populated.
- Heavy thematic-term repetition across `elevator_pitch`,
  `plot_overview`, `themes`, and `character_arcs` (the prompt
  says "almost comical how much the thematic concepts… are
  emphasized").
- `genre_signatures` always populated (2-6).

**What search emits:**
- The grief example showed a clean six-field body — pretty close
  to the ingest shape.
- However the search side rarely *cross-repeats* terms between
  fields. Ingestion explicitly does (e.g. "grief" appears in
  `themes` AND in `plot_overview` AND in `elevator_pitch`).
  Cosine similarity benefits from this redundancy because it
  weights the load-bearing concept more heavily in the embedded
  vector.

**Specific guidance fix:** in the category note for `plot_analysis`
fan-out (story/thematic archetype, central topic), instruct the
LLM to reuse the load-bearing thematic terms across
`elevator_pitch`, `plot_overview`, `thematic_concepts`, and
`character_arcs`. Ingestion does this; search should match.

### F. `production` — geographic granularity drift

**What ingestion produces:**
- Up to 3 raw IMDB filming-location strings, comma-joined,
  lowercased: "tikal national park, guatemala, sidi driss
  hotel, matmata, tunisia".
- City + country, named-landmark + region, etc.
- `production_techniques` is a tiny **filtered** list of existing
  vocabulary terms ("hand-drawn animation", "stop-motion",
  "found-footage", "computer animation").

**What search emits:**
- For "filmed in Iceland", search produces `filming_locations:
  ["Iceland"]`. That works on the cosine front because every
  Iceland-shot ingest doc has "iceland" somewhere in its text,
  but it is **strictly coarser** than what ingest carries.
- When the user names a city ("filmed in Reykjavik" or "shot in
  Taipei"), the search side has no guidance to pair city with
  country ("reykjavik, iceland") to mirror the ingest format.
- `production_techniques` outputs are usually fine when fired,
  but the search prompt does not transmit the **closed
  vocabulary** of accepted production techniques (the
  classifier's allowlist).

**Specific guidance fix:** the `filming_location.md` category note
should say: "When the user names a specific city or landmark,
emit it together with its country in the same comma-joined string,
matching the IMDB format ('reykjavik, iceland' not just
'reykjavik')." It should also explain that more-specific is
better than less-specific. For `production_techniques`, surface
the allowlist (animation modalities, capture methods, plus
found-footage) so the search side knows what vocabulary to use.

### G. `reception` — adjective+noun shape OK, but slang and award line missed

**What ingestion produces:**
- `reception_summary` is dense evaluative prose ("audiences
  praise X… while some criticize Y").
- `praised_qualities` / `criticized_qualities` are 3-6
  adjective+noun aspect labels each ("compelling lead
  performances", "convoluted plot").
- A `major_award_wins:` line is appended deterministically when
  the movie has prestige wins.

**What search emits:**
- 1-2 sentence `reception_summary`, generally well-shaped.
- Praised/criticized lists are short and adjective+noun — close
  to ingest.
- The search side has **no path to mention award wins inside the
  reception body** — it would have to query AWD separately. That
  is correct architecturally (awards have their own endpoint),
  but the `major_award_wins:` line in the embedded vector is
  carrying a real signal that *should* be picked up by reception
  cosine when the user says "Oscar-winning drama". The search
  guidance should note this — when the trait names award status,
  the reception body can include phrases like "academy awards"
  in `reception_summary` or `praised_qualities` to reach that
  line, instead of only routing to AWD.

### H. Two cross-cutting prompt-level issues

1. **The 2-4-word ceiling.** `semantic.md` line 61: term-list
   spaces want "compact 2–4-word phrases." This is a global rule
   that contradicts the per-space ingest reality:
   - `viewer_experience` ingest terms run **5-10 phrases per
     section**, of 1-5 words each.
   - `watch_context` ingest terms run **4-8 per section**.
   - `narrative_techniques` ingest terms run **1-3 per section**
     (the only space where the ceiling is right).
   - `production.production_techniques` runs 1-2 terms.
   - `reception.praised_qualities` runs 3-6 terms.
   The ceiling should be *per space*, not global, and should
   match the ingest range per space.

2. **The "translate, don't echo" rule** is one-directional. The
   prompt warns against echoing user input. It does not warn
   that translating *too professionally* loses the vernacular
   register the embedding lives in. A balanced rule:
   "Translate the user phrase into the space's ingest-side
   register — which for `viewer_experience` and `watch_context`
   includes search-query slang and synonym redundancy, and for
   `narrative_techniques` and `reception` is closer to the
   user phrasing if it already uses craft / aspect vocabulary."

---

## Part 4 — Recommended changes to search-side guidance

A focused checklist. Each item is a concrete prompt edit grounded
in the ingest source of truth.

### `prompts/endpoints/semantic.md`

1. Replace the global "compact 2–4-word phrases" rule with a
   per-space density table. Example:

   ```
   | Space                | Terms / section | Phrase length |
   |----------------------|-----------------|---------------|
   | viewer_experience    | 5–10 + 2–5 negs | 1–5 words     |
   | watch_context        | 4–8             | 1–6 words     |
   | narrative_techniques | 1–3             | 1–6 words     |
   | reception (terms)    | 3–6             | 1–3 words     |
   | production techniques| 1–2             | 1–3 words     |
   | filming_locations    | 1–3 entries     | full place    |
   ```

2. Replace the negations rule with an ingest-aligned version:

   "`viewer_experience` ingest text routinely populates negations
   in every section, even when the user input does not name a
   boundary. Author negations to mirror this — for each populated
   section, include 1-3 negations naming the closest opposites of
   the section's terms ('not a tearjerker', 'no jump scares').
   Suppress only when the section is barely populated."

3. Add a **register table** per space:

   ```
   plot_events:        synopsis prose, past tense, third-person.
                       Multi-sentence. No proper nouns when the
                       user can't supply them, but generic agents
                       and locations (e.g. "a former pilot",
                       "a frozen continent") preserved.
   plot_analysis:      labeled fields. Reuse load-bearing terms
                       across elevator_pitch, plot_overview,
                       thematic_concepts. Generic, no proper nouns.
   viewer_experience:  search-query vernacular. Slang and
                       synonym redundancy expected. First/second
                       person fragments OK ("kept me guessing").
   watch_context:      search-query vernacular, intent-framed
                       ("turn my brain off", "stoned movie",
                       "talk-about topical themes").
   narrative_techniques: established craft labels verbatim
                       ("Chekhov's gun", "ticking clock deadline",
                       "found-footage presentation"). Don't
                       paraphrase the canonical vocabulary.
   production:         filming_locations as comma-joined raw
                       place strings, city + country when both
                       are available. production_techniques drawn
                       from the closed allowlist.
   reception:          adjective+noun aspect labels for terms.
                       Reception_summary as one or two evaluative
                       prose sentences in third-person.
   ```

4. Add an explicit **slang allowlist** for `viewer_experience` and
   `watch_context`. The ingest prompt names: "tearjerker",
   "gorefest", "campy", "snoozefest", "stoned movie", "white
   knuckle", "edge of your seat", "wrecked me", "fucked up",
   "freaky", "comfort movie", "cry your eyes out". The search
   prompt should reproduce that list as the target vocabulary.

### `prompts/categories/few_shot_examples/viewer_experience.md`

5. Re-author the few-shot examples to model the actual ingest
   density. Replace the current 3-term-clean examples with one
   richer fan-out example, e.g.:

   ```
   "uplifting feel-good comedy" →
   emotional_palette: ["uplifting", "feel-good", "warm",
     "heartwarming", "joyful", "laugh out loud"]
   emotional_palette_negations: ["not depressing", "not bleak"]
   tone_self_seriousness: ["earnest", "playful", "lighthearted"]
   tone_self_seriousness_negations: ["not cynical", "not mean
     spirited"]
   tension_adrenaline_negations: ["not stressful", "not anxiety
     inducing"]
   ending_aftertaste: ["satisfying ending", "feel-good payoff"]
   ```

   The current examples train the model to be sparse — re-train
   it to match ingest density.

### `prompts/categories/additional_objective_notes/viewer_experience.md`

6. Add a "term density" line: "Each populated section should
   carry 5-10 terms and 1-3 negations to match ingest verbosity.
   Anchor on adjectives, then add 1-2 vernacular synonyms and 1-2
   user-search-phrase variants per concept."

7. Add a "phrasing variety" line: "Mix short adjectives with
   short user-search phrases in the same list — both end up in
   the same embedded vector. 'cozy, warm, gentle' is good;
   'cozy, warm, gentle, comfort movie, feel good watch' is
   better."

### `prompts/categories/additional_objective_notes/emotional_experiential.md`

8. The current note correctly fans `emotional_experiential` to
   four endpoints. Add: "Across the four endpoints, you should
   normally populate at least three of them; this category's
   ingest-side signal is spread across `viewer_experience` (tone),
   `watch_context` (motivation), `reception` (audience label), and
   keyword (binary effect tag). Single-endpoint coverage will
   under-recall."

### `prompts/categories/additional_objective_notes/plot_events.md`

9. Replace "dense synopsis register" with "compact synopsis
   prose: 3-6 sentences in past-tense third-person, naming
   generic agents, generic locations, and the major beats as
   factual events. Match the verbosity of an IMDB plot summary,
   not a one-line log-line."

10. Update the few-shot examples to show 3-6 sentence outputs.

### `prompts/categories/additional_objective_notes/filming_location.md`

11. Add: "Emit each location as a comma-joined string mirroring
    the IMDB format. When the user names a city, pair it with
    the country ('reykjavik, iceland' not just 'reykjavik').
    More-specific (city + country, region + country) ranks
    higher than country-only — match what the ingest text
    actually contains."

### `prompts/categories/additional_objective_notes/narrative_devices.md`

12. Add: "When firing `narrative_techniques`, deliberately
    populate every sub-field where the trait grounds a real
    technique. Most rich queries touch 4-6 of the 9 sub-fields.
    `audience_character_perception`,
    `conflict_stakes_design`, and `additional_narrative_devices`
    are routinely under-utilized — surface them when the trait
    grounds them. Use the canonical craft labels verbatim
    ('non-linear timeline', 'unreliable narrator',
    'Chekhov's gun', 'ticking clock deadline')."

### `prompts/categories/additional_objective_notes/visual_craft_acclaim.md` and `music_score_acclaim.md`

13. For reception bodies, anchor the term shape: "praised_qualities
    and criticized_qualities are adjective+noun aspect labels (1-3
    words each, e.g. 'evocative score', 'memorable theme',
    'spectacular cinematography'). Match ingest by emitting 3-6
    terms when the trait genuinely names that many aspects."

### Closing principle to add somewhere in `semantic.md`

14. "Your authored body must read like an ingest-side embedding
    text for a matching movie. The ingest text for each space is
    constructed by the corresponding `embedding_text()` method in
    `schemas/metadata.py`; if you don't recognize the per-space
    style on sight, look at a real movie's embedding text before
    authoring. Mismatch in vocabulary, density, or register is
    invisible at validation time but degrades retrieval silently."

---

## Summary

The ingestion side defines each vector space with a distinctive
voice — synopsis prose for `plot_events`, labeled
generic-thematic prose for `plot_analysis`, vernacular
search-query terms with default negations for `viewer_experience`,
intent-framed search-query terms for `watch_context`, established
craft labels for `narrative_techniques`, raw place strings for
`production.filming_locations`, and adjective+noun aspect labels
for `reception` — each carefully chosen to match how real users
phrase those facets.

The search side names the seven spaces and their sub-fields
correctly, fans out cleanly across categories, and avoids the
worst mistakes (no plot detail in `watch_context`, no proper
nouns in `plot_analysis`, no numerics anywhere). But its
authoring guidance is too uniform — a global "compact 2-4-word
phrases" ceiling, a "translate, don't echo" rule pointed in the
wrong direction, few-shot examples that model 3-term sparseness,
and per-category notes that under-specify the unique register of
each space.

The fix is per-space register guidance plus revised few-shot
examples that model ingest-side density and vernacular. With
those changes the cosine alignment should sharpen across all
seven spaces — particularly `viewer_experience`, where the
register gap is widest and most retrieval-relevant.
