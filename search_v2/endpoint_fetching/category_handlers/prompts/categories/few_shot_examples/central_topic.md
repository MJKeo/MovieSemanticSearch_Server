# Few-shot examples

These calibrate two things on top of the keyword-vs-semantic routing
decision:

1. When Semantic fires, the body lands EXCLUSIVELY on `plot_analysis`.
   plot_events, viewer_experience, narrative_techniques, watch_context,
   production, and reception are NEVER populated for this category.
2. Cross-field repetition of the load-bearing subject term across
   `elevator_pitch` / `plot_overview` / `thematic_concepts` is
   mandatory — the ingest side does this on purpose, and the query
   side must match.

<example>
Input:
```xml
<retrieval_intent>Find films whose focal subject is World War II.</retrieval_intent>
<expressions><expression>movies about WW2</expression></expressions>
```
Expected: both endpoints commit. Keyword walk surfaces `WAR` (broad
superset — every WW2 film is a war film) and `WAR_EPIC` (narrow
sub-form — covers epic-scale combat films like *Saving Private Ryan*
but excludes WW2 dramas like *Schindler's List*). Commit `WAR` alone:
single-member cover is enough, and the over-pull (other war eras) is
acceptable because WW2 is a major slice of war films and semantic
recovers the WW2-specificity. Do NOT add `WAR_EPIC` to the commit —
the new policy avoids stitching tags together; `WAR` already covers
the subject. Semantic commits exclusively on `plot_analysis`:

```json
{
  "plot_analysis": {
    "elevator_pitch": "a WW2 film",
    "plot_overview": "A WW2 film set during the second world war, following soldiers, civilians, or resistance figures whose lives are defined by WW2.",
    "genre_signatures": ["WW2 film", "war film"],
    "thematic_concepts": ["WW2", "the second world war"]
  }
}
```

Why this works:
- "WW2" repeats verbatim across `elevator_pitch` / `plot_overview` /
  `genre_signatures` / `thematic_concepts` — cross-field weighting
  matches the ingest side.
- `plot_overview` keeps the subject in subject position and uses
  role-nouns ("soldiers, civilians, or resistance figures") instead
  of fabricating named characters or specific battles.
- `conflict_type` and `character_arcs` stay empty — the trait names
  a subject, not a transformation or a "X vs Y" framing.
- plot_events is NOT populated. Embedding "the soldier storms the
  beach" or "the unit liberates a camp" would commit the query to
  one WW2 trope and miss WW2 dramas built around other stories.
</example>

<example>
Input:
```xml
<retrieval_intent>Find biographical films as a subject class.</retrieval_intent>
<expressions><expression>biopics</expression></expressions>
```
Expected: both endpoints commit. Keyword walk surfaces `BIOGRAPHY` —
perfect cover for the subject class (the subject *is* the biographical
class). Commit `BIOGRAPHY` alone. Semantic STILL fires — the registry
covers the class label, semantic adds the framing dimension a tag
cannot carry — and routes exclusively to `plot_analysis`:

```json
{
  "plot_analysis": {
    "elevator_pitch": "a biopic",
    "plot_overview": "A biopic following the life of a real person, dramatizing real events from their biography.",
    "genre_signatures": ["biopic", "biographical drama"],
    "thematic_concepts": ["biopic", "biographical life story"]
  }
}
```

Why this works:
- "biopic" / "biographical" repeats across four sub-fields for
  cross-field weighting.
- `plot_overview` names the class and stays generic — no real person
  is invented because the user did not name one.
- plot_events is NOT populated. There are no specific life-event
  beats to embed without inventing them; padding with "the
  protagonist rises to fame and falls from grace" commits the query
  to one biopic trope.
- Do NOT abstain on semantic because keyword has perfect coverage;
  this category routes semantic by default because the two endpoints
  layer.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films whose focal subject is running.</retrieval_intent>
<expressions><expression>movies about running</expression></expressions>
```
Expected: both endpoints commit. Keyword walk surfaces `SPORT` —
running is a meaningful slice of sport films, so SPORT is a slightly
broader single-member superset. Commit `SPORT` alone; the over-pull
(football, basketball, hockey) is acceptable because semantic recovers
the running-specificity. Semantic commits exclusively on `plot_analysis`:

```json
{
  "plot_analysis": {
    "elevator_pitch": "a film about running",
    "plot_overview": "A film centered on running, following a runner or runners for whom running is the dramatic axis of the story.",
    "genre_signatures": ["running film", "sports drama"],
    "thematic_concepts": ["running"]
  }
}
```

Why this works:
- "running" / "runner" appears in all four populated fields — the
  load-bearing term is repeated for cross-field weighting.
- `plot_overview` puts running in subject position and uses a
  role-noun ("a runner") instead of fabricating a named athlete or
  a specific race.
- plot_events is NOT populated. Embedding "training arcs and race
  climaxes" commits the query to one running-film trope and biases
  retrieval away from films that center running differently (a
  recovery story, a youth-coach film, a documentary subject).
- `thematic_concepts` keeps to the subject itself; resist drifting
  to "endurance" or "perseverance" — those are STORY_THEMATIC_ARCHETYPE
  territory.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films whose focal subject is Princess Diana.</retrieval_intent>
<expressions><expression>about Princess Diana</expression></expressions>
```
Expected: semantic-only commit; keyword abstains with
`commitment-criteria-fail`. Keyword walk surfaces `BIOGRAPHY` — every
Diana film is biographical, but BIOGRAPHY covers every biopic of every
subject (scientists, politicians, athletes, musicians). Diana-focused
films are a tiny fraction of that universe — that is the
too-broad-to-be-useful pattern. No tighter registry member names Diana
or the royal family at a useful granularity. Abstain on keyword.
Semantic carries the call and commits exclusively on `plot_analysis`:

```json
{
  "plot_analysis": {
    "elevator_pitch": "about Princess Diana",
    "plot_overview": "A film about Princess Diana, dramatizing the public life of the late Princess of Wales.",
    "genre_signatures": ["biopic"],
    "thematic_concepts": ["Princess Diana", "Diana, Princess of Wales"]
  }
}
```

Why this works:
- The user named a proper noun (Princess Diana); the ingest side
  also uses the proper noun for films actually about her, so the
  query keeps it. No proper nouns are added beyond what the user
  gave.
- "Princess Diana" repeats across three sub-fields, weighting the
  central subject in the embedded vector.
- plot_events is NOT populated. Padding with "Diana visits charity
  events", "Diana speaks to the press", "Diana clashes with the
  royal family" fabricates plot specifics around the subject —
  exactly the failure mode the policy bars. The user gave a
  subject, not a plot.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films whose focal subject is chess.</retrieval_intent>
<expressions><expression>movies about chess</expression></expressions>
```
Expected: semantic-only commit; keyword abstains with
`commitment-criteria-fail`. Keyword walk surfaces `SPORT` (stretches —
chess is not a sport) and `BIOGRAPHY` (only fits chess-biopics, and
even there is too-broad). No single registry member is a clean
superset; stitching does not fix the mismatch. Abstain on keyword.
Semantic commits exclusively on `plot_analysis`:

```json
{
  "plot_analysis": {
    "elevator_pitch": "a film about chess",
    "plot_overview": "A film centered on chess, following a chess player or chess players for whom chess is the dramatic engine of the story.",
    "genre_signatures": ["chess film"],
    "thematic_concepts": ["chess"]
  }
}
```

Why this works:
- "chess" repeats across all four populated fields.
- `plot_overview` keeps chess in subject position and uses a
  role-noun ("a chess player").
- plot_events is NOT populated. Embedding "chess matches and
  training montages" would commit the query to one sub-trope
  (tournament film) and miss films that use chess differently
  (chess as metaphor, chess between two recurring characters,
  chess as a backdrop for an investigation).
</example>

<example>
Input:
```xml
<retrieval_intent>Find films whose focal subject is death.</retrieval_intent>
<expressions><expression>movies about death</expression></expressions>
```
Expected: semantic-only commit; keyword abstains with
`commitment-criteria-fail`. Keyword walk may surface tags that
*correlate* with death-focused stories (`DRAMA`, `TRAGEDY`,
`PSYCHOLOGICAL_HORROR`, `WAR`) — none *name* death as the subject.
Do not stitch a multi-tag union to manufacture coverage. Abstain on
keyword. Semantic commits exclusively on `plot_analysis`:

```json
{
  "plot_analysis": {
    "elevator_pitch": "a film about death",
    "plot_overview": "A film centered on death as the dramatic axis, where mortality, dying, and confronting death anchor the story.",
    "thematic_concepts": ["death", "mortality", "confronting death"]
  }
}
```

Why this works:
- "death" repeats across three sub-fields and survives in
  "mortality" / "confronting death" as true paraphrases.
- `genre_signatures` stays empty: death is thematic territory, not
  a subgenre signature.
- plot_events is NOT populated. Films about death sit across many
  forms (drama, comedy, fantasy, documentary) — embedding any
  specific plot beat ("a character receives a terminal diagnosis",
  "a funeral brings the family together") would commit the query
  to one sub-trope and miss the others.
- viewer_experience is NOT populated. The temptation is to emit
  `emotional_palette.terms = ["heavy", "melancholy", "somber"]` —
  but that retrieves films that FEEL heavy, not films that are
  ABOUT death. A comedy that engages mortality head-on (*The Bucket
  List*, *Departures*) matches the trait but does not read as
  "heavy"; a grim downer with no thematic engagement with death
  reads as "heavy" but is not about death. The trait is the
  SUBJECT, not the FEEL — route to plot_analysis only.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films whose focal subject is dogs.</retrieval_intent>
<expressions><expression>movies about dogs</expression></expressions>
```
Expected: both endpoints commit. Keyword walk surfaces `ANIMAL` —
dogs are a meaningful slice of animal films, so ANIMAL is a slightly
broader single-member superset (every dog film is an animal film;
the over-pull picks up cat / horse / wildlife films, which is
acceptable because semantic recovers the dog-specificity on the
same call). Commit `ANIMAL` alone. Correlation-only members fail
the test — `FAMILY` fits most dog movies but also covers Pixar /
family-comedy / holiday films with no dog at all, so it does not
earn a commit and is not stitched on. Semantic commits exclusively
on `plot_analysis`:

```json
{
  "plot_analysis": {
    "elevator_pitch": "a film about a dog",
    "plot_overview": "A film centered on a dog, where the dog and the humans around it anchor the story.",
    "genre_signatures": ["dog film", "family animal film"],
    "thematic_concepts": ["dog", "dogs"]
  }
}
```

Why this works:
- "dog" / "dogs" repeats across four sub-fields, weighting the
  central subject in the embedded vector.
- `plot_overview` puts the dog in SUBJECT position ("a dog... the
  dog and the humans around it") rather than object position
  ("movies that feature dogs"). The ingest side puts dogs in
  subject position for dog-centric films; the query must match.
- plot_events is NOT populated. Padding with "the dog gets lost
  and journeys home" or "the dog protects its owner" commits the
  query to one trope (lost-dog road movie, working-dog drama) and
  misses dog films built around different stories.
- `thematic_concepts` keeps to the subject itself. Resist drifting
  to "loyalty", "companionship", or "unconditional love" — those
  are thematic essence and belong to STORY_THEMATIC_ARCHETYPE.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films whose focal subject is Antarctica.</retrieval_intent>
<expressions><expression>movies about Antarctica</expression></expressions>
```
Expected: semantic-only commit; keyword abstains with
`commitment-criteria-fail`. Keyword walk may surface broad members
like `ADVENTURE` or `WILDERNESS` — both stretch (most adventure /
wilderness films are not about Antarctica, and most Antarctica
films are documentary or expedition stories that don't fit the
adventure-film convention). No single registry member is a clean
superset. Abstain on keyword. Semantic commits exclusively on
`plot_analysis`:

```json
{
  "plot_analysis": {
    "elevator_pitch": "a film about Antarctica",
    "plot_overview": "A film centered on Antarctica, where the continent itself — its landscape, isolation, and extreme conditions — is the focal subject of the story.",
    "thematic_concepts": ["Antarctica", "the Antarctic continent"]
  }
}
```

Why this works:
- "Antarctica" / "Antarctic" repeats across three sub-fields.
- `plot_overview` keeps Antarctica in SUBJECT position and names
  "the continent itself" to mark it as the focal subject, not just
  a backdrop. This is the distinguishing move against
  NARRATIVE_SETTING: a thriller set on an Antarctic base is
  NARRATIVE_SETTING territory; a film *about* Antarctica as a
  subject (the place itself, its exploration, its climate, the
  expeditions to it) is CENTRAL_TOPIC territory.
- `genre_signatures` stays empty — Antarctica doesn't form a
  recognizable subgenre label on its own.
- plot_events is NOT populated. Embedding "the explorers cross the
  ice", "the team gets trapped in a blizzard", "the climate
  scientist measures the ice cores" commits the query to one
  sub-trope (Shackleton-style expedition, base-isolation thriller,
  climate-doc framing) and misses the others.
</example>

**COUNTER-EXAMPLE — do NOT emit this for any CENTRAL_TOPIC trait
(shown with "movies about dogs" because it is the canonical
failure case, but the same fails for WW2, biopics, running, chess,
death, Antarctica, etc.):**

```json
{
  "plot_events": {
    "plot_summary": "dogs as central characters. dogs as primary subject matter. dogs as the focus of the story."
  }
}
```

Why this fails:
- The wording is meta-description of the search intent, not
  synopsis prose. Phrases like "as central characters", "as primary
  subject matter", "as the focus of the story" never appear in any
  movie's stored plot_events text — they pull the query into a
  self-referential critic-review neighborhood rather than near
  actual dog-movie plots.
- plot_events is the wrong space for this category. A topic ask
  gives no specific events to embed. Either the body has to
  fabricate plot specifics around the subject (which biases
  retrieval toward one sub-trope) or it has to use meta-language
  like the example above (which doesn't match the ingest register
  at all). Both paths fail. The correct route is the dogs example
  above — `plot_analysis` with the subject named in synopsis-shaped
  sentences.
