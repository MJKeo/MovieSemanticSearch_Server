# Sensitive content — additional notes

This category is about **specific content axes** — violence, gore, nudity, sexual content, strong language, drug use, on-screen animal death — and whether the user wants them present, absent, or dialed to a particular intensity. The requirement names a content dimension directly, not a packaged audience and not a whole-movie mood.

## How the three endpoints split the work

Each endpoint speaks a different content-shape. Which subset fires is driven by the phrasing of the ask, not by endpoint availability.

- **KEYWORD** answers a binary presence / absence question on a registry content-flag. The cleanest member is `ANIMAL_DEATH` — canonical for "does the dog die?" / "no pet dies" phrasings. Fire KEYWORD only when the input names a binary content flag whose registry definition matches — do not force SPLATTER_HORROR, EROTIC_THRILLER, or similar genre-narrow tags as proxies for broad content axes ("gore", "nudity"), because they narrow the result pool to a genre rather than flagging the axis.
- **METADATA** answers a **rating-ceiling** question via the `maturity_rating` column. Fire when the input implies a rating ceiling as a proxy for overall content intensity ("family-friendly intensity", "nothing above PG-13 for the content"). The rating itself is the excluded value; polarity on the wrapper carries the exclusion. METADATA does not fire for specific axis asks ("no gore") — a rating ceiling covers all axes together and would remove movies the user did not object to.
- **SEMANTIC** answers a **gradient intensity** question via `viewer_experience.disturbance_profile`. Fire when the ask is spectrum-framed rather than binary — "not too bloody", "violent but not graphic", "intense but not brutal". The populated body describes the content axis; `terms` name what the disturbance IS, `negations` name the intensity boundary the user is drawing. SEMANTIC does not fire for a clean binary flag KEYWORD already covers.

## Polarity inversion — parameters describe the target concept directly

Every endpoint payload describes the content axis the user is talking about. Polarity on the wrapper carries whether the user wants it present or absent. Never write an anti-concept into `parameters`.

- "No animal death" → KEYWORD `classification: ANIMAL_DEATH` + wrapper `polarity: "negative"`. Never a hypothetical "NO_ANIMAL_DEATH" inverse.
- "Not too bloody" → SEMANTIC `disturbance_profile.terms: ["bloodshed", "gore"]` + wrapper `polarity: "negative"` + `match_mode: "trait"`. The body describes BLOOD; the wrapper flips the axis.
- "Family-friendly intensity" → METADATA `maturity_rating: {rating: "r", match_operation: "greater_than_or_equal"}` + wrapper `polarity: "negative"`. The body describes the excluded rating directly; the wrapper carries the exclusion.

## Hard vs soft — the `match_mode × polarity` framework

This category exercises the full 2×2 more than any other. Same surface word can route four ways depending on hardness and direction.

| User framing | `match_mode` | `polarity` | Endpoint | Effect |
|---|---|---|---|---|
| "No gore", "nothing with nudity" (binary presence ruled OUT) | `filter` | `negative` | KEYWORD | hard exclusion of movies tagged with the flag |
| "With nudity", "where the dog dies" (binary presence required IN) | `filter` | `positive` | KEYWORD | hard inclusion of movies tagged with the flag |
| "Not too bloody", "violent but not graphic" (intensity GRADIENT, dial down) | `trait` | `negative` | SEMANTIC | matches still qualify but score lower as the intensity rises |
| "Family-friendly intensity" (rating CEILING as content proxy) | `filter` | `negative` | METADATA | hard exclusion of movies above the ceiling |

**Do not conflate filter-negative with trait-negative.** Same axis, different hardness. "No gore" is binary rule-out — a movie with any gore is dropped entirely. "Not too bloody" is a gradient preference — a moderately bloody movie still qualifies, just scores lower. The hardness reading comes from the phrasing: "no / without / zero" = filter; "not too / not overly / avoid / a bit less" = trait.

## Boundary with Target audience (Cat 17)

Cat 17 frames the movie as a **whole-packaging** decision — who it was made for. Cat 18 frames **specific content axes** inside the movie — gore, nudity, language, violence intensity. The Semantic sub-space differs: Cat 17 populates `watch_context.watch_scenarios` (viewing occasion), Cat 18 populates `viewer_experience.disturbance_profile` (intensity gradient).

"Family-friendly" can reach both categories through upstream co-emission when the ask carries both an audience framing AND a content-intensity ceiling. Handle only the slice routed to you — stay on content axes and intensity here; if the target_entry reads as pure audience packaging with no content dimension named, the right answer is the empty combination even when `maturity_rating` is technically in your endpoint set.

## When to no-fire (every endpoint)

Return the empty combination whenever the input does not actually name a concrete content axis or a rating ceiling:

- A vague "nothing heavy" or "something light" with no specific axis named, no ceiling implied, and no framing SEMANTIC's disturbance_profile could populate without fabrication.
- A pure audience framing ("for kids", "for grown-ups") with no content-axis language — that is Cat 17's slice.
- A phrase about mood / tone / pacing with no content-dimension signal ("intense" as a tone, not as violence intensity) — Cat 22's territory.

Record the reason in `overall_endpoint_fits` and leave every endpoint's `should_run_endpoint: false`. Inventing a ceiling or a content axis the input does not support is worse than no-fire.

## The one principle

Fire exactly the endpoints whose shape matches the user's framing. Binary presence → KEYWORD. Rating ceiling → METADATA. Gradient intensity → SEMANTIC. Each one only when the input genuinely supports it; leave the rest silent.
