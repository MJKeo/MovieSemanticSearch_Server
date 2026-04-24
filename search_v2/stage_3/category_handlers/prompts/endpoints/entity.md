# Endpoint: Entity (Named-Entity Lookup)

## Purpose

Literal string lookup against lexical posting tables for named real entities (persons, characters) and title patterns. Execution resolves a primary credited form plus any additional credited aliases via exact string equality (after shared normalization) against an ingestion-time dictionary; title patterns are literal substring or prefix matches against movie titles.

## Canonical question

"What kind of entity does the requirement name, what is its canonical credited form (plus any other forms it might be credited under), and — for actors and characters — how prominently should it appear?"

## Capabilities

- Person lookup (actor, director, writer, producer, composer — five role-specific tables, plus a `broad_person` mode searching all five with a primary-role bias).
- Character lookup by in-story credited name, with optional `central`-to-story prominence.
- Title pattern: literal substring-anywhere or literal title-prefix match.
- Alternative credited forms: multiple surface variants OR-unioned via max-score retrieval.
- Actor-billing / character-centrality prominence scoring.

## Boundaries (what does NOT belong here)

- Production companies / studios (Pixar, A24, Marvel Studios, Disney, Ghibli) → studio endpoint. If a studio description somehow reaches here, it was misrouted — produce the closest supported lookup you can from the remaining types.
- Named franchises / shared universes → franchise endpoint.
- Role types that are NOT a specific named character ("a cop", "a vampire", "a detective") → keyword endpoint or semantic endpoint depending on framing. If one reaches here, still produce the closest character lookup from the description as written.
- Exact title lookup ("find the movie 'Heat'") — handled elsewhere; should not reach this endpoint.

## Entity types

Pick exactly one. The choice determines which sub-fields populate and which posting tables execution searches.

**person** — Any real individual in film credits: actor, director, writer / screenwriter, producer, composer / musician. A name found on a credits block. If the description identifies a real credited person connected to the film, this is a person lookup.

**character** — A specific fictional character identified by in-story name ("The Joker", "Hannibal Lecter", "Batman", "Gandalf"). Character lookups resolve to credited character-name strings from cast lists — real names like "Dr. Hannibal Lecter", not role descriptions like "a cannibalistic psychiatrist".

**title_pattern** — Substring or prefix search against movie titles, not exact title lookup. "title contains the word 'love'", "movies with 'night' in the title", "titles starting with 'The'".

## Person role selection

Five role-specific tables — `actor`, `director`, `writer`, `producer`, `composer` — plus `broad_person` searching all five.

**Specific role** when the description explicitly or nearly explicitly states it. "Directed by", "written by", "produced by", "starring", "composed the score for" → one table. Role-cued category context ("named person (director)") → one table. This is the common case.

**broad_person** only when the description does not state a role AND the person could plausibly be credited in multiple ways. "Woody Allen movies" is broad_person (directs, writes, acts — user probably wants all). "Christopher Nolan movies" is broad_person even though he is best known for directing, because the phrasing does not specify.

**primary_category** — populated only when `person_category=broad_person`. Set to the single role the person is predominantly known for when you are confident (director for Christopher Nolan, actor for Tom Cruise). Biases the cross-posting score toward their main domain without excluding the others. Leave null only when the person is genuinely equally known across multiple roles and picking one would distort the result. Leave null entirely when `person_category` is a specific role.

**Evidence precedence for person-role decisions:**
1. Description is authoritative for the requested role or prominence.
2. Parent-fragment context can break close ties.
3. Parametric knowledge is last-resort support for `primary_category` and name resolution — not a reason to override explicit phrasing.

## Alternative credited forms

Persons and characters are frequently credited under more than one string across the movie database. Each credited string is its own exact-match key in the lexical dictionary — a one-character difference means zero matches for that form. Missing a real alias silently drops every movie using it.

**Cost asymmetry — internalize this before deciding what to include.** Retrieval takes the MAXIMUM score across all forms supplied. A spurious alias that matches no credits scores zero and adds nothing. A real credited form omitted silently drops real results. Over-including costs ~0; under-including is a retrieval bug. The correct bias is toward inclusion.

**Inclusion bar — deliberately low.** Include any form you believe would plausibly appear as a credit string in at least one film featuring this entity. You do NOT need to have verified a specific film's credit list. General knowledge of how this kind of entity is typically credited is the signal — use it.

**Forms that clear the bar** (examples, not a closed list):
- A superhero / masked vigilante's civilian / secret-identity name, when films credit the civilian billing separately from the hero billing.
- A villain's legal name alongside their alias, or vice versa.
- A performer's legal name alongside their stage / rap / mononym, when both appear in real credits.
- The composite "FirstName 'StageName' LastName" form some films use for stage-name performers.
- A long-form credited name (title + full name, or legal middle name included) alongside the shorter bare form, when films vary.

**Forms that do NOT belong:**
- Descriptive phrases, scene quotes, character traits.
- Nicknames that only live in dialogue, marketing, or fan communities — not real credit strings.
- Diacritic / casing / punctuation variants — shared normalization handles these.
- Hyphenation variants — ingest expands hyphens automatically.

Title patterns have no aliases — leave alternative-form fields null for `title_pattern`.

## Prominence modes

Applies only when billing-position scoring is meaningful: `entity_type=person` with `person_category=actor` or `broad_person`, OR `entity_type=character`. For all other lookups (director-only / writer-only / producer-only / composer-only persons, title patterns), leave prominence fields null.

**Applicable modes by entity:**
- Actor-table searches (person + actor or broad_person): `default`, `lead`, `supporting`, `minor`.
- Character searches: `default`, `central`.

**Mode definitions:**

- `default` — The user names the entity without specifying prominence. "Brad Pitt movies", "movies with Spider-Man", "films featuring Hannibal Lecter". No prominence adjective. Typical case.
- `lead` (actor only) — User explicitly wants the actor in a leading role. Triggers: "starring", "in a lead role", "leading role", "main character played by". Merely listing the actor is NOT lead — the description must name the prominence.
- `supporting` (actor only) — Explicit supporting role. "Supporting role", "played a supporting part", "as a supporting character".
- `minor` (actor only) — Brief, small appearance. "Cameo", "cameos", "in a minor role", "small part".
- `central` (character only) — The character is the subject of the movie: "centers on", "is about", "the story of", "protagonist", OR when the description uses the character's name as the subject of a possessive noun phrase ("Spider-Man movies", "the Joker's story", "films about Batman").

**Principle:** `lead` / `supporting` / `minor` / `central` all require explicit language. When no such language is present, `default` is correct. Do not pick a stronger mode simply because the entity is famous or the reference feels prominent — that is what `default` covers.

## Name canonicalization

Governs `primary_form` ONLY. Alternative-form inclusion follows the Alternative Credited Forms section above — do not let primary_form's "don't invent" discipline bleed into alias enumeration; they are different decisions with different cost profiles.

`primary_form` is matched literally. For persons and characters, exact string equality after shared normalization against the ingestion-time dictionary. A one-character difference — missing initial, wrong spelling, added or dropped suffix — means zero matches for that form.

**Persons** — The full, conventional credited name.
- Correct obvious typos ("Johny Dep" → "Johnny Depp").
- Expand unambiguous partial names where surrounding context pins the referent ("Scorsese" in a query about film directors → "Martin Scorsese").
- Never add honorifics, titles, or extra name parts the user did not give unless the form is the common credited full name.
- Never invent middle names the user did not type.
- If a partial name is genuinely ambiguous and surrounding context does not pin it down, use the form the user typed.
- Stage-name / legal-name variants, when both demonstrably appear in credits, go in alternative_forms.

**Characters** — The most prominent credited form as it typically appears in cast lists. "The Joker" — not "Joker" as primary. "Hannibal Lecter" — not "Dr. Lecter" or "Hannibal the Cannibal". Fix misspellings only when clearly a misspelling; do not guess. Multiple credited incarnations and secret-identity pairings go in alternative_forms.

**Title patterns** — Literal text fragment for matching inside the title. No SQL wildcards, no quotation marks. "love" for "title contains the word love"; "The" for "titles starting with The". Pick `title_pattern_match_type=contains` for "anywhere in title"; `starts_with` for title-prefix. Literal pattern match, not canonical-name resolution — alternative_forms does not apply.
