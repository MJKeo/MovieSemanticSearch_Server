# Additional objective notes

## Category Target

Concrete focal subject the movie is about: a real person, event, case,
disaster, war, institution, place-as-subject, named public story, or a
named class of subject (real singers, war heroes, biopics). Ask:
"What subject does the film orbit around?"

## Endpoint policy

**Semantic almost always fires.** The named subject lands in
`plot_analysis` — see "Semantic plot_analysis is the only target"
below. Author the body at the request's specificity — "WW2" stays
WW2, not generalized to "war"; "real singers" stays focused on
real-life singer narratives, not generalized to "musicians";
"running" stays running, not generalized to "sport". Over-generalizing
forfeits the specificity the user gave you and is the most common
failure mode here. Abstain on semantic only when the subject is so
narrow that no plausible plot_analysis content could carry it (rare
for this category).

**Keyword fires only when a single registry member is a clean
superset of the subject — perfect cover, or just slightly broader.**
The keyword endpoint is a coarse signal; it earns its place when one
tag aligns with the subject closely enough to be useful on its own.
Do NOT stitch multiple tags together to fake coverage. If no single
member is a tight fit, abstain on keyword and let semantic carry the
call.

- A single perfect-cover or near-cover member commits.
  `BIOGRAPHY` for "biopics" (perfect cover — every biopic carries
  it). `WAR` for "WW2" (slightly broader — every WW2 film is a war
  film; WAR also retrieves WW1/Vietnam, which is acceptable over-pull
  since semantic refines on WW2 specifically). `SPORT` for "movies
  about running" (broader but still aligned — running is one slice of
  sport, and semantic recovers the running specificity).
- Members that only narrowly fit fail the test. `WAR_EPIC` alone for
  "WW2" excludes WW2 dramas like *Schindler's List* — narrow-only
  biases the score toward the sub-form and zeros valid out-of-sub-form
  matches. Abstain on keyword.
- Members that only correlate with the subject fail the test.
  `SPORT` for "movies about chess" stretches — chess is not a sport,
  and firing `SPORT` tag-matches sports films at 1.0 while
  chess-focused films that lack sport tags score 0. Abstain.
- Members that are too broad to carry useful signal fail the test.
  `BIOGRAPHY` for "movies about real singers" technically entails
  (every real-singer film is biographical), but BIOGRAPHY covers
  every biopic of every subject — the signal is too diluted to
  distinguish singer films from scientist or politician films.
  Semantic carries the singer-specificity; firing BIOGRAPHY alone
  adds noise, not signal. Abstain.
- Multi-keyword commits are rare and only justified when two members
  jointly carve the subject without either being on its own a clean
  superset (uncommon — when in doubt, abstain on keyword).

**Cross-family keyword borrowing is allowed when the rule above is
met.** A tag normally housed in ADAPTATION_SOURCE or FORMAT_VISUAL
can commit under CENTRAL_TOPIC if it cleanly covers the subject.
Family membership is not the gate; the single-member superset test is.

**Semantic fires alongside a perfect-cover keyword.** When the
registry has a member that directly names the subject class
(`BIOGRAPHY` for "biopics", `SPORT` for "running"), the keyword
commit handles class-level coverage and semantic still fires to add
the framing, qualifier, or specificity the registry can't carry on
its own. Both endpoints layer — this is the design.

## Semantic plot_analysis is the only target

When this category routes to Semantic, it routes exclusively to the
`plot_analysis` space. Do NOT populate plot_events,
viewer_experience, narrative_techniques, watch_context, production,
or reception. The reasoning is structural, not stylistic:

- **Why not plot_events.** The plot_events ingest text is a
  chronological "what happens" recount — concrete characters,
  locations, actions, sequences. A CENTRAL_TOPIC query names a
  SUBJECT, not a sequence of events. The only way to land a topic
  query in plot_events is to fabricate plot specifics around the
  subject ("dogs travel home and reunite with their owners", "the
  soldier storms the beach", "Diana attends a charity gala") —
  that pulls the embedding toward one trope rather than the
  centroid of films about the subject. The rule: if the query is
  specific enough to write plot_events legitimately, it is a
  PLOT_EVENTS-category call, not a CENTRAL_TOPIC call. Vague
  topic asks that would need padding to fill plot_events are
  exactly what plot_analysis is built for.
- **Why not the other spaces.** Topic aboutness is not a feel
  (viewer_experience), a craft device (narrative_techniques), a
  watch occasion (watch_context), a production attribute
  (production), or a critical-reception signal (reception).
  Spreading signal across these spaces retrieves films that share
  tone or craft with the subject rather than films ABOUT the
  subject, and dilutes the match.

## Sub-field selection within plot_analysis

The `plot_analysis` body has six sub-fields. Populate the ones the
subject grounds; empty sub-fields are valid and expected.

- **`elevator_pitch`** — log-line capsule naming the subject, ≤6
  words ideal. "a WW2 film", "a biopic", "about Princess Diana",
  "a film about running", "a film about a dog".
- **`plot_overview`** — 1-3 sentence generic synopsis-shaped
  sentences with the subject in SUBJECT position. Generic agents
  in place of specifics ("a runner", "a soldier", "a dog and its
  owner"). Repeat the load-bearing subject term — the ingest side
  does this on purpose. Do not invent plot beats the user did not
  give; for a pure topic ask, the overview names the subject and
  the kind of story it anchors, nothing more.
- **`genre_signatures`** — fires when the subject carries a
  recognizable subgenre or category label ("biopic", "war film",
  "sports drama", "chess film"). Stays empty when the subject is
  too narrow to form a subgenre (a specific person, a specific
  one-off event).
- **`thematic_concepts`** — fires when the subject is itself the
  thematic anchor. Use the subject term verbatim ("WW2",
  "running", "chess", "death", "Princess Diana") plus 0-2 true
  paraphrases. Do NOT drift into abstract themes the subject only
  touches ("loyalty" for dog movies, "sacrifice" for WW2) — those
  belong to STORY_THEMATIC_ARCHETYPE.
- **`conflict_type`** — usually empty. Most subjects are not "X
  vs Y" framings.
- **`character_arcs`** — usually empty. Subjects are not arc
  transformations.

## Cross-field repetition is mandatory

Reuse the subject term verbatim across the populated fields. The
ingest side repeats the load-bearing subject in `elevator_pitch` /
`plot_overview` / `thematic_concepts` deliberately so the embedded
vector weights it; the query side must match. If the central
subject appears only once in the body, the query vector is
under-weighted on it relative to the ingest text and the match is
weaker.

## Density and register

- Synopsis-register or label-shaped only. NEVER write
  meta-descriptions of the search intent — phrases like "dogs as
  central characters", "the subject is X", "X as the primary
  subject matter", "X as the focus of the story" never appear in
  any movie's stored plot_analysis text. They pull the query into
  a self-referential, critic-review neighborhood instead of the
  topical one.
- The subject must appear in SUBJECT POSITION of its sentences,
  not in object position. "A dog journeys with its owner" beats
  "movies that feature dogs". The ingest side puts the subject in
  subject position.
- Proper nouns are allowed when the user named one (a real
  person, a specific war, a named event) — the ingest side keeps
  these proper nouns for films actually about those subjects. Do
  not introduce proper nouns the user did not give.
- Term lists: 2-5 entries per active list, true paraphrases of
  the subject only. Apply the substitution test: "Could I show
  this term to the user instead of their original word, and
  would they say yes, that's the same thing?" "Biopic" →
  "biographical drama" passes. "Biopic" → "true story" /
  "drama" fails as drift.

## Boundaries

- Thematic essence (grief, redemption, found family) → STORY_THEMATIC_ARCHETYPE.
- Mere "has X" presence (movies with clowns, has horses) → ELEMENT_PRESENCE.
- Time/place setting (set during WWII, set in Tokyo) → NARRATIVE_SETTING.
- Full plot event sequence (a heist where the crew turns on each other) → PLOT_EVENTS.
