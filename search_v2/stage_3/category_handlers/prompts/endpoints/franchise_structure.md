# Endpoint: Franchise Structure

## Purpose

Translates a franchise requirement into parameters across six axes: five retrieval axes (`franchise_or_universe_names`, `recognized_subgroups`, `lineage_position`, `structural_flags`, `launch_scope`) and one scoring-bias flag (`prefer_lineage`). Canonical names are resolved through a shared tokenizer against an inverted index; structural flags and launch scope are closed enums matched at ingest.

## Canonical question

"Which franchise axes does this requirement signal, and what literal parameter values pin them?"

## Capabilities

- Named franchise / shared-universe lookup (specific lineages AND umbrella shared universes — one axis covers both).
- Named subgroup lookup (phases, sagas, timelines, trilogies, director-eras — independent of whether a parent franchise is also named).
- Narrative-position filter (sequel / prequel / remake / reboot).
- Structural-role filter (spinoff, crossover).
- Launch-scope filter (launched-a-franchise vs. launched-a-subgroup).
- Main-line-vs-universe-adjacent scoring bias.

## Boundaries (what does NOT belong here)

- Named persons (actors, directors, etc.) → entity endpoint.
- Named production companies / studios (Marvel Studios as producer, not as franchise) → studio endpoint.
- Awards → award endpoint.
- Generic "remake" queries with no franchise context → keyword endpoint (REMAKE classification).

## Searchable axes

Every populated axis narrows the result set (AND across axes). Within a list field, OR semantics.

**franchise_or_universe_names** — Named franchise, IP, or shared cinematic universe. Signals: proper-noun franchise references ("James Bond", "Star Wars", "Harry Potter", "Marvel", "Marvel Cinematic Universe", "MonsterVerse"). At ingest a franchise name is stored as either the lineage (specific-title slot) or the shared_universe (umbrella slot) — retrieval covers both from this single field, so do NOT predict which slot the stored value lives in. The lineage-vs-universe distinction drives `prefer_lineage` scoring, not match restriction. Populate whenever a named franchise is part of the requirement; leave null for purely structural or purely subgroup-based requirements.

**recognized_subgroups** — Named subgroup: phase, saga, trilogy, timeline, director-era slice, or other widely-used sub-lineage label. Valid examples: "phase one", "phase three", "infinity saga", "multiverse saga", "kelvin timeline", "snyderverse", "disney live-action remakes", "sequel trilogy", "skywalker saga". Rules:
- Do not restate the franchise itself as a subgroup.
- Do not invent labels on the spot — only widely-used ones.
- Populate whenever the user's requirement actually targets a recognized subgroup, INCLUDING standalone subgroup queries like "trilogies" or "phase one movies" where no parent franchise is named. Independent of `franchise_or_universe_names`.

**lineage_position** — Narrative-position enum: `sequel`, `prequel`, `remake`, `reboot`. Sequel and prequel are the common cases. Remake is rare here; it belongs in this endpoint only for a franchise-specific remake concept, not a broad remake query.

**structural_flags** — Optional list:
- `spinoff` — the requirement targets branch entries rather than the main trunk.
- `crossover` — the requirement targets films whose identity is separate known entities or characters meeting or colliding.

**launch_scope** — What the movie launched:
- `franchise` — the movie launched a franchise.
- `subgroup` — the movie launched a subgroup inside a broader franchise.

`launch_scope=subgroup` does NOT require a named subgroup. "Movies that launched a subgroup" → `launch_scope=subgroup`, `recognized_subgroups=null`. Populate `recognized_subgroups` only when the user names the subgroup itself.

**prefer_lineage** — Scoring-bias bool. When true, movies matching the query on the lineage side score higher than those matching only on the shared_universe side. Match set is unchanged — spinoffs and universe-adjacent films still appear, just ranked below main-line entries. Default false. Set true ONLY when all of the following hold:
- Exactly one specific franchise is named (not an umbrella like "Marvel" or "DC"), and that franchise has a main line plus known spinoffs or universe-adjacent entries.
- The requirement does not explicitly invite spinoffs or universe-adjacent content.
- `franchise_or_universe_names` is not a multi-name umbrella sweep.
- `recognized_subgroups` is null — a named subgroup already disambiguates intent; layering `prefer_lineage` on top adds noise.

True examples: "shrek movies" (main Shrek upranked over Puss in Boots); "john wick movies" (main John Wick upranked over The Continental); "toy story movies" (main Toy Story upranked over Forky shorts / Lightyear); "harry potter movies" (main Harry Potter upranked over Fantastic Beasts); "the conjuring movies" (main Conjuring upranked over Annabelle / The Nun).

False examples: "MCU movies" / "marvel movies" (the umbrella IS the request); "DCU movies" / "dc movies" (same); "star wars movies" (umbrella, not single lineage); "middle-earth movies" (umbrella shared universe); "shrek spinoffs" (user asked for the non-lineage side); "harry potter and fantastic beasts" (user invited the shared-universe entries); "marvel phase one movies" (subgroup disambiguates); any entry with two or more names (umbrella sweep by definition).

When in doubt, false. Misclassifying true as false loses a small ranking nudge; misclassifying false as true when the user wanted the umbrella demotes content they wanted.

## Canonical naming

Names resolve through a shared tokenizer + inverted index. Tokenizer: lowercase, diacritic fold, punctuation strip, whitespace collapse, ordinal digit-to-word ("20th" → "twentieth"), cardinal 0–99 digit-to-word ("phase 1" → "phase one"), whitespace+hyphen split, stopword drop ("the of and a in to on my i for at by with"). The same tokenizer ran at ingest — orthographic variants do NOT need enumeration; they collide automatically. Focus alternates on genuinely different canonical forms.

Follow the same canonical rules the ingest-side franchise generator uses:
- Most common, well-known canonical form.
- Lowercase.
- Spell digits as words.
- Expand "&" to "and".
- Expand abbreviations ONLY when the expanded form is also in common use:
  - "MCU" → "marvel cinematic universe"
  - "DCEU" → "dc extended universe"
  - "LOTR" → "the lord of the rings"
  - "monsterverse" stays "monsterverse"
  - "x-men" stays "x-men"
- For director-era subgroup labels, drop first names when the surname alone is the common form:
  - "Peter Jackson's Lord of the Rings Trilogy" → "jackson lotr trilogy"
  - "John Carpenter Halloween films" → "carpenter halloween films"

**Specificity — umbrella vs narrow:**
- Umbrella query ("Marvel movies", "Lord of the Rings films") → emit the broad form ("marvel cinematic universe", "the lord of the rings").
- Narrow lineage inside an umbrella ("Doctor Strange", "Captain America") → emit the narrow form ALONE. Every Doctor Strange film is already MCU; adding "marvel cinematic universe" as a second entry would OR-union the entire MCU back in and over-broaden the query.

**Count rules for `franchise_or_universe_names`:**
- 1 entry in the common case.
- 2–3 entries ONLY when genuinely different canonical names are in common use that ingest might plausibly have stored — the extra entries perform an umbrella sweep via across-name union ("marvel cinematic universe" + "marvel" sweeps the MCU proper PLUS every other "marvel"-tagged franchise entry; "the lord of the rings" + "middle-earth").
- Do NOT pad with spelling, punctuation, casing, hyphenation, diacritic, or digit-vs-word variants — the tokenizer handles those symmetrically.

**For `recognized_subgroups`:** same rules. Only emit labels that studios, mainstream film criticism, or widely-used fan terminology actually use. Do not invent a subgroup label just because one would be useful.

## Scope discipline

Populate only the axes the concept actually signals. Do not add extra axes just to describe the franchise more fully. "Sequels" is a `lineage_position` query — not an invitation to guess a franchise name. If an item seems mildly misrouted, recover only the narrowest franchise interpretation literally supported by the description plus minimal context.
