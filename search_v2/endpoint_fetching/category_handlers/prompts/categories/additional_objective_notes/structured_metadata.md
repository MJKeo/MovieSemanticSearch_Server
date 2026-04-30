# Structured metadata — additional notes

This category is about **closed-schema column predicates** on the structured attribute surface of `movie_card`. Your job is to pick exactly one `target_attribute` from the fixed set, then populate that one column's sub-object with the tightest correct literal spec of the requirement. The endpoint-context chunk above covers the column vocabulary and the per-attribute translation rules; the notes here focus on the discipline of picking the right column and the boundaries with the many nearby categories that get misrouted here.

## One attribute per firing

The endpoint queries exactly one column. You pick it via `target_attribute`; the sub-object for that one column is the only payload execution reads. If the requirement seems to span two columns (e.g. "French movies from the 90s" reads as both `country_of_origin` and `release_date`), that is a signal the requirement should have been split upstream — not that you should merge two columns here. Scope your response to the atom routed to you; a sibling atom is handling the other half in another call.

## Evidence before routing

Populate `constraint_phrases` with the verbatim bits of the atomic rewrite or parent fragment that pin this attribute (e.g. `["French films"]`, `["under 2 hours"]`, `["PG-13"]`). Then commit to `target_attribute`. Grounding the column choice in actual input text is what prevents the most common misroute: reading a cultural-tradition or content-sensitivity phrase as if it were a country or rating predicate.

## Polarity is on the wrapper

Describe the target concept directly. "Not on Netflix" populates `services: [Netflix]` with wrapper `polarity: negative` — never an "exclude" list inside `parameters`. "No R-rated content" populates the rating positively and flips at the wrapper. Inverting axes inside the sub-object to simulate negation is always wrong.

## Boundaries with nearby categories

Cat 10 sits adjacent to more categories than any other. The most common misroutes:

- **Chronological (Cat 32).** Release date as a **range** or **era** — "90s movies", "movies from 2015", "recent films", "older movies" — belongs here as `release_date`. Release date as an **ordinal position** within a candidate set — "the newest Scorsese", "the earliest Bond", "the most recent", "the oldest" — belongs to Chronological. The phrase is ordinal when it selects a position; range when it names a window. "Recent" is a window (last few years); "most recent" is a position (the single latest).
- **Trending (Cat 9).** "Popular right now", "trending", "what everyone's watching this week", "viral" require a live refresh signal that the static `popularity` column cannot produce. No-fire here and let the Trending dispatch carry it. `popularity` is for stable mainstream-vs-niche notability ("well-known", "obscure", "hidden gem", "underrated") — not for live buzz.
- **Target audience (Cat 17) / Sensitive content (Cat 18).** A maturity rating named **as a rating** — "PG-13 movies", "rated R", "NC-17 films" — is `maturity_rating` here. A maturity level used as a **content-sensitivity ceiling** on who the movie is for — "family-friendly", "suitable for kids", "no R-rated violence", "safe for grandma" — belongs to the audience / sensitive-content handlers, which use the keyword and semantic channels alongside maturity. Upstream should split compound phrasings like "PG-13 horror for teens" into a Cat 10 atom (the rating) and a Cat 17 atom (the audience framing) — handle only the slice routed to you.
- **Reception quality / superlative (Cat 25).** A **numeric reception score as a standalone attribute** — "rated above 8 on IMDb", "80%+ on Rotten Tomatoes", "high-scoring films" — belongs here as `reception`. **Qualitative or superlative reception** — "critically acclaimed", "cult classic", "best horror of the 80s", "era-defining", "underrated" (as stature, not notability) — belongs to Cat 25 on the semantic channel. The split: is the user naming a scalar position on the reception axis, or naming a stature / superlative framing? The latter does not reduce to a single column.
- **Cultural tradition / national cinema (Cat 12).** Country as **legal/financial production origin** — "American production", "made in France", "Mexican films" framed as where-the-movie-was-produced — belongs here as `country_of_origin`. Country as a **proxy for a cinema tradition or movement** — "Korean cinema", "Bollywood", "French New Wave", "J-horror", "Hong Kong action" — belongs to Cat 12. The tradition slot carries craft and stylistic lineage that a country-ID filter cannot; no-fire on tradition phrases even when they contain a country word.
- **Filming location (Cat 13).** `country_of_origin` is where the production was legally based (the entity that produced and financed the movie). Filming location is where shooting physically happened. A Hollywood-funded film shot in Jordan has US origin and Jordan filming location — those are different columns and different handlers. No-fire on phrases that explicitly describe shooting geography ("shot in Iceland", "filmed in New Zealand").
- **Audio language vs. country.** `audio_language` fires ONLY when the phrase explicitly names audio, dubbing, or subtitling ("French audio", "dubbed in Spanish"). Bare "French films" / "Korean films" / "foreign films" is `country_of_origin`, never `audio_language`. Never infer audio track from cultural identity.

## When to no-fire

- The requirement targets a concept that no column can represent — cultural tradition, filming location, live trending, stature, qualitative reception, content sensitivity. Record the misroute in `coverage_gaps`.
- The requirement is too vague to pin a literal value ("some era", "whatever length"). Do not invent a threshold the user did not imply.
- Parent-fragment modifiers flip the atom into a self-contradictory shape the wrapper's polarity cannot resolve. Record the contradiction in `coverage_gaps`.

The tightest failure mode: firing `release_date` on an ordinal phrase, firing `country_of_origin` on a cinema-tradition phrase, firing `maturity_rating` on a content-sensitivity phrase, or firing `popularity` on a live-trending phrase. If the atomic rewrite cannot be stated as a literal predicate on one column — a range, a comparator, an enum, or an explicit list — no-fire.
