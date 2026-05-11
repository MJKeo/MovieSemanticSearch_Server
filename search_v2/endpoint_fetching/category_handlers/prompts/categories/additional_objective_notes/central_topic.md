# Additional objective notes

## Category Target

Concrete focal subject the movie is about: a real person, event, case,
disaster, war, institution, place-as-subject, named public story, or a
named class of subject (real singers, war heroes, biopics). Ask:
"What subject does the film orbit around?"

## Endpoint policy

**Semantic almost always fires.** The 7 vector spaces carry graded
signal for the named subject. Author each space's body to mirror the
request's specificity — "WW2" stays WW2, not generalized to "war";
"real singers" stays focused on real-life singer narratives, not
generalized to "musicians"; "running" stays running, not generalized
to "sport". Over-generalizing forfeits the specificity the user gave
you and is the most common failure mode here. Abstain on semantic
only when no space genuinely carries the subject (rare for this
category — most named subjects land in plot_events as motif text or
in plot_analysis as a thematic anchor).

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

## Boundaries

- Thematic essence (grief, redemption, found family) → STORY_THEMATIC_ARCHETYPE.
- Mere "has X" presence (movies with clowns, has horses) → ELEMENT_PRESENCE.
- Time/place setting (set during WWII, set in Tokyo) → NARRATIVE_SETTING.
- Full plot event sequence (a heist where the crew turns on each other) → PLOT_EVENTS.
