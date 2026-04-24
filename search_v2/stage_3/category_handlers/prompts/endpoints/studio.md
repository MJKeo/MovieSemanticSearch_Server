# Endpoint: Studio (Production Company)

## Purpose

Translates a production-company requirement into one of two complementary axes:

1. **brand** — A closed enum of 31 curated ProductionBrand values for umbrella / parent-brand queries. Execution reads the ingest-time-stamped `lex.inv_production_brand_postings` keyed by the enum's brand_id, so **time-bounded ownership** (Lucasfilm under Disney only from 2012) and **rename chains** (Twentieth Century Fox → 20th Century Studios) are handled automatically.
2. **freeform_names** — Up to 3 IMDB surface forms for specific sub-labels and long-tail studios not covered by the registry at umbrella level. Execution normalizes + tokenizes + intersects discriminative tokens against `lex.studio_token`, then joins resulting production_company_ids against `movie_card.production_company_ids`.

Exactly one path is set. Both set simultaneously is allowed but only has effect as a brand→freeform fallback when brand returns empty.

## Canonical question

"Is this umbrella-level (and in the registry) or specific-sub-label / long-tail (requires freeform surface forms)?"

## Capabilities

- Umbrella / parent-brand sweep across subsidiaries and historical sub-labels, automatically time-bounded and rename-aware.
- Specific sub-label match (e.g. "Walt Disney Animation Studios" specifically, rather than the whole Disney catalog).
- Long-tail / niche / foreign studio match ("Villealfa Filmproductions", "Cannon Films", "Carolco", "Shochiku").

## Boundaries (what does NOT belong here)

- Streaming-platform availability (movies AVAILABLE on Netflix / Amazon / Apple) → metadata streaming. Streamer disambiguation (NETFLIX / AMAZON_MGM / APPLE_STUDIOS as producer vs. watch-provider) was resolved upstream in stage_2b — when an item reaches this endpoint, treat the entity as a producer. For entities that are both producers and streaming platforms, assume the user means the producer.
- Named persons → entity endpoint.
- Named franchises / shared universes → franchise endpoint.

## Brand vs. freeform_names

**Use `brand`** when the user is asking at umbrella / parent-brand level and the registry covers that brand. "Umbrella intent" = the whole catalog of this studio's productions, across subsidiaries and historical sub-labels. Examples: "Disney movies", "Warner Bros. films", "A24 indies", "MGM catalog", "Ghibli" (registry covers it), "Marvel Studios films" (its own registry entry AND a member of the DISNEY umbrella — pick `marvel-studios` for the narrow reading, `disney` for the broad reading).

The brand path respects time-bounded ownership — "Disney" does NOT match Star Wars (1977) even though Lucasfilm is now Disney-owned, because Lucasfilm joined Disney in 2012. Do not reason about this; pick the brand, the data handles the rest.

**Use `freeform_names`** when the query names:
- A specific sub-label not in the registry ("Walt Disney Animation Studios" specifically — the `disney` brand returns the whole Disney catalog; "HBO Documentary Films"; "Fox Searchlight" before its 2020 rename).
- A long-tail or niche studio absent from the registry.
- A foreign studio named in its native surface form.

**Choosing surface forms for `freeform_names`.** Emit up to 3 surface forms. The right forms are the variants most likely to appear verbatim in IMDB's `production_companies` credits for films associated with that studio. Good covering set:
- One condensed / acronym form ("MGM", "HBO", "BBC").
- One expanded / full form ("Metro-Goldwyn-Mayer", "Home Box Office").
- One alternate well-known variant when a distinct form exists (remember: if an umbrella brand exists, prefer that path).

Emit fewer than 3 when fewer distinct forms exist. Do NOT pad with spelling/capitalization variants — normalization at execution handles those. Do NOT emit semantic translations ("Japan Broadcasting Corporation" for NHK) unless that translation is a form IMDB actually uses in credits.

## Registry brands

Closed set of brand values, each followed by a sample of IMDB `production_companies` surface forms counting as that brand. The form list is a sample, not exhaustive — umbrella brands cover more strings than shown. Use this table to decide (a) which brand to pick for umbrella queries, and (b) whether a query is actually umbrella-level at all (if no registry brand covers the user's phrasing, use `freeform_names`).

{{BRAND_REGISTRY}}

## Freeform canonicalization

Each `freeform_name` is treated as a phrase that should match an IMDB `production_companies` string. Execution normalizes (lowercase, diacritic fold, punctuation strip, whitespace collapse, numeric ordinals to word form — `20th` → `twentieth`), splits on whitespace AND hyphens, and intersects the discriminative tokens against the token index. Both sides run the same normalizer — do NOT pre-normalize. Emit the natural surface form the studio is known by.

Emit each form the way IMDB would store it, not the way the user typed it:
- "WB" rarely appears standalone in IMDB credits — emit "Warner Bros." instead (and use `brand=warner-bros` anyway).
- "Mouse House" never appears — don't emit it; use `brand=disney`.
- Hyphenated and spaced variants ("Tri-Star" vs "TriStar") are handled by tokenization; emit the form you believe is most common.
- Do not include suffixes the studio doesn't use in credits. "A24" is how A24 is credited — don't expand to "A24 Films LLC". "Ghibli" expands to "Studio Ghibli" because that is the credited form.
