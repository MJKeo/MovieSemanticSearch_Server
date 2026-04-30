# Category notes: Franchise / universe lineage

This category covers **the position of a film inside a named lineage**. That includes membership in a named franchise or shared universe, membership in a named subgroup (phase, saga, trilogy, director-era), and positioning relative to other entries in that lineage — sequel, prequel, reboot, mainline-vs-offshoot, crossover, or "the original, not the remake."

The endpoint's entire value is a resolvable franchise or lineage anchor. Without one, it has nothing to point at.

## The named-franchise cue is load-bearing

Every firing case needs at least one of: a named franchise / IP / shared universe, a named subgroup, a narrative position relative to a specific franchise (even if the franchise name lives in `intent_rewrite` rather than the atom), a structural flag tied to a lineage, or a launch-scope assertion. A free-floating "sequels" or "spinoffs" with a franchise identifiable from the parent fragment or sibling fragments is fine — use those to recover the anchor. A truly unanchored structural concept routes elsewhere.

## "Remake" lives in two categories — stay in your slice

This category owns **lineage-positioning remake** phrasing: the later entry in a specific named lineage. "The Scarface remake" means the 1983 Scarface, positioned against the 1932 original inside the Scarface lineage. This is the rare case where `lineage_position=remake` applies — paired with a franchise name.

Cat 5 (Adaptation source flag) owns **origin-medium remake** phrasing: "remakes" as an independent yes/no flag on the film, with no named franchise. "Remakes of 80s movies", "foreign-language remakes", "good remakes" — no anchor, no firing here. Step 2 splits compound phrases; if the requirement on your slice has no franchise tether, no-fire and let Cat 5 handle it.

The "original, not the remake" phrasing routes here when a franchise name anchors it ("the original Scarface, not the remake" → lineage positioning for the mainline / earlier entry). Express the positive side of the user's intent in `parameters`; let wrapper polarity carry the "not the remake" exclusion when it applies to a separate sibling atom.

## Iconic characters are not your fan-out trigger

When a query names an iconic character ("Batman movies"), Step 2 emits a Cat 2 (Named character) atom for the character. Do not expand a character reference into a franchise name unless the atom on your slice actually carries franchise-lineage meaning on its own. If the only signal is the character name, no-fire and let Cat 2 handle it through the entity endpoint.

## Boundaries with nearby categories

**Adaptation source flag (Cat 5).** "Remakes" / "novel adaptations" / "video-game movies" with no named franchise → no-fire. Cat 5's keyword flag is the right channel.

**Studio / brand (Cat 3).** "Marvel Studios movies" as a production-company statement → no-fire (Cat 3). "Marvel movies" / "MCU" as the shared universe → fire here. The line is whether the user means the entity that produced the film or the franchise identity of the film. Sibling-fragment framing and `route_rationale` usually disambiguate.

**Named character (Cat 2).** Character-only queries → no-fire. Character-plus-franchise-positioning ("the original Batman movies, not the Nolan ones") splits into a Cat 2 atom for the character and a Cat 4 atom for the lineage/subgroup split — handle only your atom.

## When to no-fire

- The atom asks for a structural concept (remake, sequel, spinoff) with no named franchise anchor available from the atom, parent fragment, or sibling fragments.
- The atom names a character, actor, or studio but carries no independent franchise-lineage meaning.
- The phrase is too vague to pin to a concrete axis ("franchise movies", "movies that feel like a series").
- A `POLARITY_MODIFIER` produces a contradiction the endpoint cannot express on a single axis — record it in `coverage_gaps` rather than forcing a degraded spec.

No-fire with a clear `coverage_gaps` note is the correct outcome whenever the anchor isn't there. The endpoint cannot fabricate a franchise identity, and a nearest-name guess will either over-broaden or silently return nothing.
