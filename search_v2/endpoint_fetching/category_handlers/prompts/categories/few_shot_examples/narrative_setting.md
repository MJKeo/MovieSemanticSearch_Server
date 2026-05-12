# Few-shot examples - Narrative setting

Each example shows the input plus the plot_summary body text the
endpoint should emit. Bodies are pure setting-anchor fragments
joined by periods, using ONLY grammar attested in real IMDB
synopses and plot_summary entries. No invented characters, no
invented actions, no invented properties, no critic-blurb
phrasings.

<example>
Input:
```xml
<retrieval_intent>Find films whose story is set in 1940s Berlin.</retrieval_intent>
<expressions><expression>1940s Berlin</expression></expressions>
```
Expected: fire semantic plot_events.
plot_summary:
> "In 1940s Berlin. Set in 1940s Berlin. 1940s Berlin."

Rationale: Shape 1 coordinate anchor, static input → opens-in
forms. Three attested fragments — sentence-head prep clause, "Set
in" tagline form, and bare title-card form. No "takes place in",
"much of the film unfolds in", or "against the backdrop of" —
those don't appear in real synopses.
</example>

<example>
Input:
```xml
<retrieval_intent>Find movies set in a high school.</retrieval_intent>
<expressions><expression>high school</expression></expressions>
```
Expected: fire semantic plot_events.
plot_summary:
> "In a high school. Set in a high school. At a high school."

Rationale: Shape 1 milieu, static input. Three opens-in forms.
No "students" role-noun even though it would match institutional
synopses naturally — the role-noun shifts retrieval toward
ensemble student films and away from teacher / single-student /
faculty films that are also set in a high school.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films set in the Arctic.</retrieval_intent>
<expressions><expression>the Arctic</expression></expressions>
```
Expected: fire semantic plot_events.
plot_summary:
> "In the Arctic. Set in the Arctic. The Arctic."

Rationale: Shape 1 environment, static input. "The Arctic"
verbatim — not "the frozen Arctic" or "the Arctic wilderness".
</example>

<example>
Input:
```xml
<retrieval_intent>Find films set after a nuclear apocalypse.</retrieval_intent>
<expressions><expression>post-apocalyptic</expression></expressions>
```
Expected: fire semantic plot_events.
plot_summary:
> "After a nuclear apocalypse. Following a nuclear apocalypse. In the aftermath of a nuclear apocalypse. Set in a post-apocalyptic world. It's a post-apocalyptic world."

Rationale: Shape 2 post-event sub-family. Five attested event-
anchor fragments. No "survivors scavenge a ruined landscape" —
fabricated plot. The "It's a post-X world" form is attested in
real plot_summaries (The Road: "It's a post-apocalyptic world").
</example>

<example>
Input:
```xml
<retrieval_intent>Find films set during the Vietnam War.</retrieval_intent>
<expressions><expression>during the Vietnam War</expression></expressions>
```
Expected: fire semantic plot_events.
plot_summary:
> "During the Vietnam War. At the height of the Vietnam War. Set during the Vietnam War."

Rationale: Shape 2 during sub-family. Three attested fragments
("At the height of X" is attested in 1917's plot_summary). No
"soldiers grapple with combat" — that is the kind of role-noun +
action fabrication this prompt rules out.
</example>

<example>
Input:
```xml
<retrieval_intent>Find movies set in a magical realm.</retrieval_intent>
<expressions><expression>magical realm</expression></expressions>
```
Expected: fire semantic plot_events.
plot_summary:
> "In a magical realm. Set in a magical realm. In a fantasy realm."

Rationale: Shape 3 fictional world. The "fantasy realm" variant
is allowed because "fantasy" is a type-paraphrase of "magical"
that real synopses use interchangeably. NO "where wizards battle
ancient evil" — the user did not ground that property.
</example>

<example>
Input:
```xml
<retrieval_intent>Find movies set in a magical realm where wizards battle ancient evil.</retrieval_intent>
<expressions><expression>magical realm where wizards battle ancient evil</expression></expressions>
```
Expected: fire semantic plot_events.
plot_summary:
> "In a magical realm where wizards battle ancient evil. Set in a magical realm where wizards battle ancient evil. In a fantasy realm where wizards battle ancient evil."

Rationale: Shape 3 with a user-grounded property. The
"where [property]" clause appears in every fragment because the
user named it. Property restated verbatim, never invented or
rephrased.
</example>

<example>
Input:
```xml
<retrieval_intent>Find movies that take place inside a shared dream.</retrieval_intent>
<expressions><expression>inside a shared dream</expression></expressions>
```
Expected: fire semantic plot_events.
plot_summary:
> "Inside a shared dream. Within a shared dream. Set inside a shared dream."

Rationale: Shape 3 abstract space. Paraphrases of the
inside/within framing the input used. No invented "where the
rules of physics bend" property.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films that flash back to a character's childhood in 1970s New York.</retrieval_intent>
<expressions><expression>flashbacks to 1970s New York</expression></expressions>
```
Expected: fire semantic plot_events.
plot_summary:
> "Flashbacks to 1970s New York. Through flashbacks to 1970s New York. Memories of 1970s New York."

Rationale: Position signal ("flashbacks to") → flashes-back-to
family only. No opens-in fragments — synopses about flashback
settings read differently from synopses that open in the setting,
and conflating the two would dilute the embedding.
</example>

<example>
Input:
```xml
<retrieval_intent>Find movies where characters travel to Paris.</retrieval_intent>
<expressions><expression>travel to Paris</expression></expressions>
```
Expected: fire semantic plot_events.
plot_summary:
> "Travels to Paris. Arrives in Paris. Ends up in Paris. Eventually reaches Paris."

Rationale: "Travel to" → reaches family only. No opens-in
fragment — the input explicitly says they TRAVEL to Paris, so
the film does not open there. Verb-of-movement fragments are
attested in real synopses ("comes to Tokyo", "returned to Saigon")
without naming a subject.
</example>

<example>
Input:
```xml
<retrieval_intent>Find road-trip movies that span the American Midwest.</retrieval_intent>
<expressions><expression>road trip across the American Midwest</expression></expressions>
```
Expected: fire semantic plot_events.
plot_summary:
> "Across the American Midwest. A road trip across the American Midwest. Spanning the American Midwest."

Rationale: Traversal input ("across") → traverses family only.
No "from small town to small town" — the user did not name towns.
The "from Y to Z" form only fires when both endpoints are
grounded.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films set in post-apocalyptic Tokyo.</retrieval_intent>
<expressions><expression>post-apocalyptic Tokyo</expression></expressions>
```
Expected: fire semantic plot_events.
plot_summary:
> "In a post-apocalyptic Tokyo. Set in a post-apocalyptic Tokyo. After an apocalypse in Tokyo."

Rationale: Combo — Shape 2 takes precedence due to "post-", with
Tokyo nested as the coordinate. Every fragment carries both the
event-relative anchor and the Tokyo coordinate.
</example>

<example>
Input:
```xml
<retrieval_intent>Find movies released in the 1940s.</retrieval_intent>
<expressions><expression>1940s</expression></expressions>
```
Expected: no-fire.

Rationale: Release-date metadata, not story setting. Belongs to
the RELEASE_DATE category.
</example>

<example>
Input:
```xml
<retrieval_intent>Find movies filmed in Berlin.</retrieval_intent>
<expressions><expression>filmed in Berlin</expression></expressions>
```
Expected: no-fire.

Rationale: Physical shooting geography, not narrative setting.
Belongs to the FILMING_LOCATION category.
</example>

**COUNTER-EXAMPLE A — do NOT emit this for "1940s Berlin" (invented plot):**

> "Much of the film unfolds in 1940s Berlin. In 1940s Berlin, a
> young woman discovers a secret that could change the course of
> the war."

Why this fails:
- "a young woman discovers a secret" — fabricated character and
  action.
- "could change the course of the war" — fabricated stakes.
- "Much of the film unfolds in" — critic-blurb / marketing
  language that doesn't appear in real synopses; embeds in a
  different neighborhood than synopsis prose.

**COUNTER-EXAMPLE B — do NOT emit this for "1940s Berlin" (unattested phrasings):**

> "Takes place in 1940s Berlin. Against the backdrop of 1940s
> Berlin. Much of the film unfolds in 1940s Berlin."

Why this fails:
- None of these three phrasings appear in real IMDB synopses or
  plot_summary entries. They are critic-blurb / metadata-caption
  grammar. The embedded vector sits in a part of the space the
  ingest-side documents don't occupy, so cosine match degrades.
- Use the attested forms: "In 1940s Berlin. Set in 1940s Berlin.
  1940s Berlin."
