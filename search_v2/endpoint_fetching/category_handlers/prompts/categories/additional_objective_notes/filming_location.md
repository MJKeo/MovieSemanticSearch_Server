# Filming location — additional notes

This category is about **where the camera was** — the physical production geography of a movie. Proper-noun place names naming a shooting location: "filmed in New Zealand", "shot on location in Iceland", "Morocco shoots", "Atlanta-filmed". It is not about where the story is set, not about the country that legally financed the film, and not about a national cinema tradition.

## Semantic is the only channel

No closed-schema column carries filming geography. The structured `country_of_origin` column records the legal / financial production country — a different question. Hollywood-funded films shot abroad carry US as country_of_origin even when every frame was exposed in Jordan, Canada, or New Zealand, so a metadata branch would silently miss them.

The only place filming geography lives is the Semantic **production** space's `filming_locations` sub-field (proper-noun place names — cities, regions, countries, landscapes). That is the single target for this category. `primary_vector` is always `production`.

## How to populate the body

- Put the named place(s) into `production.filming_locations` as compact proper-noun strings. Use the form a matching movie's ingest text would carry ("New Zealand", "Iceland", "Morocco", "Atlanta") — not prose wrapping ("filmed in New Zealand").
- Leave `production.production_techniques` empty. Techniques (practical effects, 16mm, single-take) belong to a different category; this one is about place, not craft.
- Do not populate other spaces. Filming geography does not genuinely land anywhere else — adding plot_events or watch_context because you can loosely justify it dilutes the vector.

## Boundaries with nearby categories

- **Plot events + narrative setting (Cat 20).** Filming location is where the camera was; narrative setting is where the story takes place. "Filmed in Tokyo" → here. "Set in Tokyo" / "takes place in Tokyo" → Cat 20. A single film can hit both (Lost in Translation is filmed AND set in Tokyo), but each atom routes independently. If the phrase only says the story happens somewhere, this is the wrong category — no-fire.
- **Cultural tradition / national cinema (Cat 12).** A tradition is a movement or national cinematic school — Bollywood, Korean cinema, Italian neorealism. That is Cat 12. "Japanese cinema" / "Bollywood films" → Cat 12. "Filmed in Japan" / "shot in Mumbai" → here. The discriminator: does the phrase name a tradition, or does it name a shooting location?
- **Structured metadata country_of_origin (Cat 10).** Cat 10 answers "which country legally produced this?"; this category answers "where was it shot?" "American production" / "French-produced film" → Cat 10 (metadata). "Filmed in America" / "shot in France" → here. When upstream dispatch sent a country-of-origin phrase to this handler, that is a misroute — no-fire and record the mismatch.

## When to no-fire

Return `should_run_endpoint: false` when:

- The phrase names a narrative setting rather than a shooting location, and no filming-geography signal is present.
- The phrase names a production country as a legal/financial origin rather than a shooting location.
- The phrase names a national cinema tradition rather than a physical place where filming occurred.
- The phrase is too vague to identify any specific place ("filmed somewhere cool", "exotic locations") — Semantic production has no target to embed against.

No-fire is always better than embedding a filming_locations body from a phrase that was actually about story setting, legal origin, or tradition.
