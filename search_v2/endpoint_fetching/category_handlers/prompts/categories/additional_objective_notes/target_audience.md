# Additional objective notes - Target audience

## Target

Fire for who the movie is packaged for or suited to as a viewer group:
family, kids, teens, adults, older relatives, cross-generational viewing.

## Decision Questions

- What audience fit is explicit in the target?
- Does that fit imply a maturity ceiling?
- Is there a registry member whose definition names the audience packaging?
- Is the ask situational: who is watching, when, or with whom?
- Which candidate endpoints add real signal? Address each one before
  committing.

## Endpoint Fit

- Metadata: maturity ceiling only. Use when the audience fit rules out higher
  ratings. Parameters describe the excluded rating range; parent polarity
  carries exclusion. Do not use metadata as a positive "adult" selector.
- Keyword: audience-packaging registry members only. Use a member only when its
  definition names the audience fit. Do not use teen genre/subgenre tags as
  proxies for "for teens."
- Semantic: watch_context only for viewing scenarios and companions. Use for
  "with my kids", "with grandparents", "family movie night." Do not turn a bare
  audience label into a fake scenario.

## Boundaries

- Specific content axes ("no gore", "no nudity") belong to Sensitive content.
- Bare rating phrases ("PG-13", "rated R") belong to Maturity rating.
- Story life-stage arcs ("coming-of-age") belong to Story / thematic archetype.
- Concrete occasions without audience packaging may belong to Viewing occasion.

## No-Fire

Return no endpoint payloads when the target names no audience fit, no maturity
ceiling, and no watch scenario. Empty is better than inventing a ceiling,
registry member, or companion.
