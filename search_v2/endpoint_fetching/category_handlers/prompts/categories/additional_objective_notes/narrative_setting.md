# Additional objective notes - Narrative setting

## Target

Search where or when the story takes place. Fire semantic
`plot_events.plot_summary` with a body of pure setting-anchor
fragments, joined by periods, in grammar that mirrors how real
IMDB synopses and plot summaries phrase setting.

This is narrative time/place, not release date, filming location,
production country, or national cinema tradition.

## Body shape — pure setting-anchor fragments, NEVER fabricate content

The plot_summary body is a sequence of short setting-clause
fragments joined by periods. Every fragment restates ONLY the
setting anchor the user grounded. No invented characters, no
invented actions, no invented properties, no tonal padding.

This is the form the schema's `plot_summary` field description
prescribes for motif/setting queries: "short synopsis-prose
fragments ... joined by periods. Mirrors the phrasings that
appear inside real synopses so cosine alignment lands on films
that contain the setting WITHOUT fabricating a plot around it."

Three failure modes this rule blocks:

- **Invented role-noun + action.** "In 1940s Berlin, a young
  woman discovers a secret" pulls retrieval toward films about
  young women discovering secrets and away from the broader pool
  of films set in 1940s Berlin.
- **Invented properties on the setting.** "In a magical realm
  where wizards battle ancient evil" is only valid if the user
  grounded the wizards-vs-evil property. Inventing it for the
  bare input "magical realm" shifts the target toward fantasy
  films about wizards battling evil and away from other magical
  realms.
- **Critic-blurb / metadata-caption grammar.** "Much of the film
  unfolds in X." / "Takes place in X." / "Against the backdrop
  of X." do NOT appear in real synopsis prose. They are
  marketing-blurb constructions, not synopsis constructions, so
  they don't match the ingest-side neighborhood. Stick to the
  attested forms below.

## Attested setting-clause forms — use these only

The forms below are all attested in real IMDB synopses or
plot_summary entries in the database. Every fragment in the body
must follow one of these grammars.

### Shape 1: Coordinate anchor (time / place / era / milieu / environment)

Static noun phrase naming a year, decade, era, real place, region,
milieu/institution, or environment.

**Triggers:** input is a static noun phrase with no event verb and
no fictional-world signal.

**Attested fragments:**
- `"In [coords]."` — sentence-head locative prep clause.
- `"[coords]."` — title-card / bare locative block (e.g. "1970s
  Mexico City." / "April 1917, the Western Front.").
- `"Set in [coords]."` — adjacent-to-synopsis tagline grammar
  ("An apocalyptic story set in [X]").
- `"In [year] [place]."` — stacked time+place when both grounded
  ("In 1942 Berlin.").
- `"At [milieu]."` — for institutional / point-locations
  ("At a high school.", "At the Pentagon.").

### Shape 2: Event-relative anchor

Subordinate clause built around an event verb or its aftermath.

**Triggers:** input contains "after", "before", "during", "post-",
"pre-", "in the aftermath of", "stranded", "trapped", or names an
event with built-in duration or consequence (war, pandemic,
apocalypse, revolution, disaster).

**Attested fragments by sub-family:**
- post-/after: `"After [event]."` / `"Following [event]."` /
  `"In the aftermath of [event]."` / `"Set in a post-[event]
  world."` / `"It's a post-[event] world."`
- during: `"During [event]."` / `"At the height of [event]."` /
  `"Set during [event]."` / `"In [time] during [event]."`
- stranded/trapped: `"Stranded in [place]."` / `"Trapped in
  [place]."` / `"Stranded after [event]."`

### Shape 3: World-establishing clause

Setting clause that names a non-real or abstract space.

**Triggers:** input names a non-real place ("Middle-earth",
"Pandora"), an abstract/non-physical space ("inside a dream",
"the afterlife", "cyberspace"), or describes a society by a rule
the user grounded.

**Attested fragments:**
- `"In [world]."` / `"Set in [world]."`
- `"In a [type] [world]."` only when the type paraphrases
  vocabulary the user used ("magical" → "fantasy" / "fantastical"
  / "magical world"). Do NOT add unrelated type words.
- `"Inside [space]."` / `"Within [space]."` for abstract spaces.
- If the user grounded a property: `"In [world] where
  [property]."` / `"Set in [world] where [property]."` — restate
  the property VERBATIM from the input. Do NOT invent properties.

Combos resolve via first-match-wins: `"post-apocalyptic Tokyo"`
hits Shape 2 first; emit Shape 2 fragments with Tokyo nested as
the coordinate (`"In a post-apocalyptic Tokyo."`).

## Position dimension — when the setting is reached

Real synopses describe settings that the movie opens in, reaches
via travel, flashes back to, or traverses across — each with
distinct grammar.

| Position signal in input | Position | Attested fragments |
|---|---|---|
| static noun phrase, no verb | opens-in (default) | "In [X]." / "[X]." / "Set in [X]." |
| "travels to", "journey to", "ends up in", "moves to", "relocates", "settles in" | reaches | "Travels to [X]." / "Arrives in [X]." / "Ends up in [X]." / "Eventually reaches [X]." |
| "flashback to", "remembers", "looks back on", "her past in", "memories of" | flashes-back-to | "Flashbacks to [X]." / "Through flashbacks to [X]." / "Memories of [X]." |
| "across", "from X to Y", "road trip", "odyssey", "spanning" | traverses | "Across [X]." / "Spanning [X]." / "A road trip across [X]." / "From [Y] to [Z]." (only when [Y] and [Z] are grounded) |

## Output strategy

- **Static setting expression** (no position verb): emit 3-5
  opens-in fragments from Shape 1's attested forms.
- **Position signal in input** (travel verb, flashback word,
  traversal preposition): emit 3-5 fragments from the MATCHING
  position family only.
- **Event-relative input** (Shape 2): emit 3-5 event-anchor
  fragments from the matching sub-family.

Density: 3-5 fragments. Below 3 is under-weighted; above 6 is
padding that drifts toward unattested phrasings.

## Phrasing discipline (strict)

- **Restate the anchor verbatim.** Use the exact words from the
  input ("1940s Berlin", not "wartime Berlin"; "the Arctic", not
  "the frozen North"). Shape and position framing varies; the
  anchor itself does not.
- **No role-nouns.** No "students", "soldiers", "survivors", "the
  protagonist". Real synopses include them, but adding them
  trains the embedding on the role-noun rather than the setting.
- **No invented actions.** No "the story unfolds", "the war shapes
  daily life", "a journey begins".
- **No tonal padding.** No "wartime backdrop", "ruined landscape",
  "frozen wilderness".
- **No critic-blurb phrasings.** No "Takes place in X." / "Much
  of the film unfolds in X." / "Against the backdrop of X." —
  these don't appear in synopsis prose.

## No-fire

No-fire when the expression names physical shooting geography
("filmed in Berlin"), real-world release era ("released in the
1940s"), production origin ("Japanese film", "Italian production"),
national cinema tradition ("French New Wave"), or a vague
atmosphere with no place, time, or event ("moody", "atmospheric").
