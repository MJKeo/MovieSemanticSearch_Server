# Additional objective notes — Studio / brand

This category is about the **production entity** behind a film — who made it, under what brand. You are deciding between the two endpoint paths (registry `brand` vs. `freeform_names`) and writing the payload that best captures the studio the user has in mind.

## Choosing the path

Route through the curated registry when the user names an umbrella / parent-brand that the registry covers. Route through freeform surface forms when the user names a specific sub-label the umbrella brand would over-cover, or a studio the registry does not carry at all (smaller outfits, foreign studios, historical names).

The registry already encodes time-bounded ownership and rename chains. Do not try to simulate those yourself with year filters, keyword hacks, or freeform padding. Pick the brand that expresses the user's scope; the executor handles the era bounds.

## Reading the scope the user wants

The same brand name can imply different scopes depending on phrasing. "Old MGM", "classic MGM", or "golden-age MGM" still points at the `mgm` registry entry — the executor's year windows naturally pick up the pre-Amazon catalog via the release-year context on each film. "Amazon MGM" or "MGM's recent releases" points at `amazon-mgm` instead. For Disney: "Disney movies" is the broad umbrella, "Pixar" is the `pixar` sub-brand, "Walt Disney Animation specifically" is narrower than the umbrella and uses freeform surface forms because no dedicated registry slot covers it cleanly from the umbrella side.

When a named sub-label has its own registry entry (Pixar, Marvel Studios, Lucasfilm), prefer that slot over freeform. Only go freeform when neither the umbrella nor a dedicated registry entry expresses the scope.

## Boundaries with nearby categories

- **Franchise / universe lineage (Cat 4).** Marvel Studios the production company is yours; the MCU as a shared universe is Cat 4. "Disney movies" is yours; "Disney Princess films" decomposes into a studio atom here and a franchise/subgroup atom in Cat 4. Step 2 has already split those — only act on the atom routed to you.
- **Streaming platform (Cat 10, Structured metadata).** A distribution/availability claim ("streaming on Netflix", "Netflix originals", "Apple TV+ exclusives") is NOT a production-brand signal. Netflix, Amazon MGM, and Apple Studios exist as registry brands only for the producer reading — when the step-2 atom carries availability / distribution framing rather than production framing, no-fire. Upstream already disambiguated; trust its routing and decline when the atom is clearly about where to watch rather than who made it.
- **Named person (Cat 1).** "Spielberg's Amblin movies" — Amblin is yours, Spielberg is Cat 1's atom.

## When to no-fire

- **Unstructured descriptor, no concrete target.** "Indie studios", "small production companies", "arthouse studios", "major studios" do not name a brand the registry or freeform path can resolve. There is no canonical surface form for "indie" in IMDB credits. No-fire.
- **Distribution-only framing.** The atom is really about availability rather than who produced the film. No-fire; another category owns it.
- **Self-contradictory / inverted by a polarity modifier the path cannot express.** Emit the target concept straight and let the wrapper's polarity field handle negation. Never encode negation inside `freeform_names` or by picking a "close enough" brand to subtract from.

## Filling freeform_names

When you take the freeform path, emit the form(s) that would actually appear in IMDB `production_companies` strings for films associated with the studio. Prefer the credited surface form over the colloquial one. Emit fewer than three when fewer distinct forms genuinely exist — do not pad with capitalization or punctuation variants; normalization handles those. Do not emit a translated form unless IMDB itself credits the studio that way.
