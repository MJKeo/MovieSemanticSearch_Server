# Format + visual-format specifics — additional notes

This category is about **the form the movie takes**, not what it is about. Two families share the lane: (1) format — documentary, animation, anime, short, mockumentary, stop-motion — and (2) visual-format specifics — black and white, 70mm / IMAX, found-footage, handheld, widescreen, single-take long shot. Keyword owns the form-family whenever a canonical registry member names it; Semantic picks up technique-level craft descriptions that the vocabulary does not carry.

## How to apply the keyword-first bias here

The bias is mild. Canonical format tags are near-definitional flags — if a movie's ingest-time classification carries DOCUMENTARY, it *is* a documentary, and running Semantic in parallel only invites prose near-misses to dilute the hit. So when the atomic rewrite names a form the registry covers (documentary, animation, anime, short, mockumentary, stop-motion animation, and similar), Keyword wins decisively.

Equally: when the atom names a technique or visual specification the registry does **not** carry — shot on 16mm, single-take long shot, heavy practical effects, handheld cinematography, black-and-white photography, 70mm presentation, found-footage style without the horror scoping — Semantic wins cleanly. This is not over-correcting against the bias; it is the correct tier call. The registry is finite and deliberately excludes many visual-format specifics, so those route to Semantic's `production.production_techniques` sub-field without apology.

State this reasoning explicitly in `performance_vs_bias_analysis`. When no tag covers the technique, say so: "no classification member names this technique, so Keyword cannot carry it — Semantic production is the only channel with the right vocabulary."

## Semantic body shape for this category

When Semantic wins, the target is almost always the **production** space. Populate `production.production_techniques` with compact craft-term phrases in the form a matching movie's ingest text would carry ("shot on 16mm", "single-take long shot", "black and white photography", "practical creature effects", "handheld camera"). Leave `production.filming_locations` empty — locations are a different category (Cat 13). `primary_vector` is `production`. Do not add other spaces; format craft does not genuinely land in plot, viewer-experience, or watch-context vocabularies.

## Boundaries with nearby categories

- **Top-level genre (Cat 11).** Format is not genre. "Documentary" is format (here); "horror movie" is genre (Cat 11). "Documentary about climate change" composes this category with Cat 6 — emit your slice (DOCUMENTARY) and let the subject atom run separately.
- **Structured metadata (Cat 10).** Runtime ceilings like "under 2 hours" are numeric and belong to Cat 10. "Short film" asks for the form-factor classification regardless of exact runtime and belongs here (SHORT). The discriminator: is the user asserting a numeric threshold, or asking for a named form?
- **Narrative devices + structural form (Cat 16).** Format is how the movie is physically presented (black-and-white, found-footage, animation). Narrative devices are how the story is structured (plot twist, nonlinear timeline, unreliable narrator). Some phrases straddle both (mockumentary is both a format and a narrative device); defer to the canonical Keyword tag when one exists. If the atom is purely about storytelling structure with no visual-form signal, that is Cat 16's lane — no-fire here.
- **Filming location (Cat 13).** Where the camera was is Cat 13 (`production.filming_locations`). Craft and technique is this category (`production.production_techniques`). Both live in the same vector space but populate different sub-fields.

## When to no-fire

Return `endpoint_to_run: "None"` when:

- The atom is too vague to name either a canonical form or a specific technique ("interesting visuals", "stylish movies") — no tag fits, and Semantic has nothing concrete to embed against.
- The phrase is really about genre, tone, or story structure rather than physical form — upstream dispatch was wrong. Record the mismatch in the aspects and no-fire rather than reaching for a weak Keyword member.
- Modifiers invert the ask in a way no candidate can express cleanly.

The tightest failure mode for this category: picking a narrow Keyword member ("ANIMATION" for "visually striking movies") on weak evidence, or forcing Semantic to embed a filler technique phrase the atom never actually named. No-fire is always better than a weak fit in either direction.
