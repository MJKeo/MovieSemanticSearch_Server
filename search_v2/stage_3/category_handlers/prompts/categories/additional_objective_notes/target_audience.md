# Target audience — additional notes

This category is about **who the movie is packaged for** — family, kids, teens, adults, "for grown-ups", "watch with grandparents". The requirement names the audience framing, not the content axes it implies and not the character's life stage inside the story.

## How the three endpoints split the work

The candidate set carries three distinct roles. Which subset fires depends on how the user framed the ask — the empty combination is a valid outcome and firing all three "because they are available" is the primary failure mode.

- **METADATA** acts as a **maturity ceiling**, never a positive rating pick. Audience framings like "family-friendly" / "for the kids" / "safe for grandma" imply that R and NC-17 sit outside the desired pool. The wrapper encodes this as `match_mode: "filter"` + `polarity: "negative"`, and the `maturity_rating` sub-object describes the excluded rating **directly** — never write "not R-rated" into parameters. Polarity on the wrapper carries the exclusion. If the framing names no ceiling (pure "for adults", no other signal), METADATA does not fire — there is no structured attribute for "adult-packaged" in this column.
- **KEYWORD** contributes an **audience-framing tag** when the registry carries a member whose definition names the audience packaging (e.g. `FAMILY` — made to be appropriate and enjoyable for both children and adults watching together; `ADULT_ANIMATION` — animation aimed at adults; `HOLIDAY_FAMILY` for the seasonal variant). Fire additively alongside whichever other endpoints apply. If no registry member's definition names the audience framing specifically, Keyword does not fire — picking a `TEEN_*` sub-genre tag as a stand-in for "for teens" silently narrows the result to a specific genre instead of the audience framing.
- **SEMANTIC** fires on `watch_context.watch_scenarios` when the ask is **situational** — who you are watching with, the occasion, the setting of the viewing. "Watch with my kids on Saturday night", "for movie night with grandma", "something to put on with the family" all belong here. A bare audience label with no situational framing ("family movies", "for adults") does not need Semantic — `watch_context` carries occasion signal, not packaging signal.

## METADATA polarity inversion — how to encode it

For "family-friendly" / "for kids" / "safe for grandma" the excluded ratings are R and NC-17 (and often "greater than PG-13"). Emit the rating the user is ruling OUT, with wrapper polarity `negative`:

- `maturity_rating: {rating: "r", match_operation: "greater_than_or_equal"}` + wrapper `polarity: "negative"` — excludes R and NC-17.
- Pick the tightest ceiling actually implied. "Suitable for young kids" typically implies excluding `pg-13` and up; "family movie" typically implies excluding `r` and up. The wrapper flips the axis; do not pre-invert it inside `parameters`.

## Boundaries with nearby categories

- **Sensitive content (Cat 18).** Cat 17 is about the **audience framing** — who the movie is for. Cat 18 is about the **content axes** themselves — gore, nudity, language, violence intensity. The Semantic sub-space differs: Cat 17 uses `watch_context.watch_scenarios` (viewing occasion), Cat 18 uses `viewer_experience.disturbance_profile` (intensity gradient). "Family-friendly" is Cat 17 (packaging); "no gore" is Cat 18 (content axis). Upstream should split compound phrasings — handle only the atom routed to you.
- **Kind of story / thematic archetype (Cat 21).** Coming-of-age as a **story archetype** — the protagonist's arc from adolescence into adulthood — belongs to Cat 21, not here. Cat 17 is about who the film is pitched to; Cat 21 is about the story shape. A phrase like "coming-of-age movie" frames the story trajectory and routes to Cat 21. A phrase like "a teen movie" / "for teens" frames the audience and routes here. The discriminator is whether the phrase names a packaged audience or an arc pattern inside the story.
- **Structured metadata (Cat 10).** A maturity rating named **as a rating** — "PG-13 movies", "rated R", "NC-17 films" — is Cat 10's territory (bare `maturity_rating` with positive polarity). Cat 17 inherits the rating column only as a **ceiling** implied by an audience framing, always with `polarity: "negative"`. If upstream routed a bare rating phrase here, the audience framing is absent — consider whether any endpoint genuinely fits rather than manufacturing one.

## When to no-fire (every endpoint)

Return empty combination when the atom routed here is not actually an audience framing:

- Coming-of-age phrased as a story arc, not an audience — Cat 21's territory.
- A bare content axis with no audience packaging — Cat 18's territory.
- A bare rating phrase with no audience framing — Cat 10's territory.
- A phrase too vague to pin any endpoint ("movies people would like", "good stuff"). Inventing a ceiling or a registry member the input does not support is worse than no-fire.

Record the misroute in `overall_endpoint_fits` and leave each endpoint's breakdown at `should_run_endpoint: false`.

## The one principle

Fire exactly the subset the user's framing calls for. METADATA only when a maturity ceiling is implied (always `filter` + `negative`, never positive). KEYWORD only when a registry member's definition names the audience packaging. SEMANTIC only when a viewing occasion or watch-partner is named. Each firing endpoint must carry signal the others cannot — over-firing dilutes the result pool, under-firing drops signal the bucket exists to preserve.
