# Interpretation-required — additional notes

This category is the last-resort fallback. You reach it only when the captured meaning is real and load-bearing but does not map cleanly to any structured category. Cats 1–30 already absorb the known patterns — the previously known members of this bucket (holistic scale, curated canon, below-the-line creators) now live in their own categories. What reaches you is the residual tail: asks whose intent is genuine but whose vocabulary has no pre-defined home.

## The interpretation task

Your job is to decode the underlying intent and construct a best-effort semantic query against whichever sub-space is most likely to carry that signal on the ingest side. Work in two steps:

1. Identify what the user actually wants. State the intent in your own words, close to the user's phrasing but sharper.
2. Ask what signal that intent would incidentally produce in a matching movie's ingest-side text. Which of the 7 spaces' sub-fields would a matching film's prose or term list naturally talk about?

Translate into that space's native vocabulary. Do not paste the user's raw phrasing into the body when the space's sub-fields use a different register — restate the intent in the terms the ingest side would use.

## Space selection for this category

With no structured discriminator to lean on, the spaces most likely to carry the signal are:

- **watch_context** — asks framed around viewing situation, personal motivation, or audience fit ("for my grandmother", "after a hard day"). Sub-fields: `self_experience_motivations`, `watch_scenarios`, `external_motivations`.
- **reception** — asks framed around how people responded to the film in a way the reception prose would naturally describe ("movies people argue about", "films that changed how critics talk about the genre"). Sub-field: `reception_summary`, occasionally `praised_qualities` when the user names the aspect.
- **plot_analysis** — asks framed around an abstract thematic or identity-level concept ("movies about mortality"). Sub-fields: `elevator_pitch`, `thematic_concepts`.
- **viewer_experience** — asks framed around a felt quality the during-viewing vocabulary can carry ("movies that feel like a warm hug"), even when no canonical tonal term fits cleanly.

Pick the single strongest space; add a second space only when a concrete atom from the input genuinely lands there. Multi-space is valid when the ask spans dimensions, but reflex fan-out dilutes the query.

## Leaning broad, flagging uncertainty

Cat 31 asks are fuzzy by construction. Narrow, overconfident queries on fuzzy intent underhit — a few nearby terms in the space's native vocabulary will usually cover the concept better than one precise-sounding phrase pulled from the user's wording. Record the interpretation openly in `requirement_aspects.aspect_description` and name what the endpoint cannot guarantee in `coverage_gaps` — that the query is interpretation-driven, not a direct match. Downstream may surface results with lower confidence; your job is to produce the best honest query, not to compensate by narrowing.

## Boundaries with every structured category

If the ask genuinely fits any structured category — even loosely — Step 2's classification was probably wrong and the right answer is no-fire. Cat 31 is not a consolation prize for near-misses: it is the bucket for asks that no structured category can hold. Before firing, check:

- Is the ask really about scale, scope, or holistic vibe? → Cat 27's territory, no-fire.
- Is it about a named curated list or canon? → Cat 28's territory, no-fire.
- Is it about a below-the-line creator? → Cat 29's territory, no-fire.
- Is it about viewing occasion or self-experience goal? → Cat 23's territory, no-fire.
- Is it about during-viewing tone or feel? → Cat 22's territory, no-fire.
- Is it about post-viewing resonance or aftertaste? → Cat 26's territory, no-fire.

If a structured category can hold the ask — even loosely — that is where it belongs, and the correct Cat 31 response is no-fire. Cat 31 only fires when the structured categories cannot carry the intent at all.

## When to no-fire

Return `should_run_endpoint: false` when:

- The intent, once decoded, fits a structured category cleanly — upstream dispatch picked Cat 31 by mistake.
- The phrase carries no resolvable intent — after reading it carefully, you cannot name what the user would be satisfied by. An empty or purely speculative observation belongs in no space.
- Every candidate space would require fabricating terms the user did not imply. A weak query vector underperforms no query at all.

No-fire is always safer than inventing ingest-side vocabulary to fill a body. The cost of a miss here is a dilute result; the cost of a silent fabrication is a misleading one.
