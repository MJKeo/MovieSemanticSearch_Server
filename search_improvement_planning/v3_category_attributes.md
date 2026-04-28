# v3 Category Attributes

Per-category routing attributes for the step 2 LLM. Each category has:
- **Core definition** — what the category is.
- **Boundary** — what it owns, what it doesn't, and where rejected
  cases route instead.
- **Edge cases** — genuinely confusing situations the LLM is most
  likely to misroute.
- **Good examples** — traits that clearly belong here.
- **Bad examples** — traits that look like they belong here but
  don't, with the correct destination.

Read alongside:
- `query_categories.md` — the full taxonomy with mechanism details.
- `v3_step_2_planning.md` — schema and decision context.
- `v3_step_2_reasoning_fields.md` — reasoning-field guidance.

---

## Open flags

- **Numbering drift.** `v3_step_2_planning.md` and
  `v3_step_2_reasoning_fields.md` reference "1-43" categories, but
  this doc and `query_categories.md` enumerate 44 (Cat 18 is the
  merged financial-scale cat; the old Cat 18 number was reused, so
  the upper bound is 44). Update those references so step 2's
  prompt enum is unambiguous.
- **Inline flags** below marked `[FLAG: ...]` are points where I
  want explicit confirmation before locking the prompt language.

---

## 1. Person credit

**Core definition.** Indexed film-credit roles by name: actor,
director, writer, producer, composer.

**Boundary.** Owns the five indexed posting tables. Does not own
below-the-line credits (cinematographer, editor, production
designer, costume designer, VFX supervisor) — those route to
**Cat 40**. Does not own title-string searches — those route to
**Cat 2**. Does not own named characters — those route to **Cat 3**
(non-anchoring) or **Cat 6** (character-anchored franchise). Does
not own named source-material creators (novelist, playwright) —
those route to **Cat 41**.

**Edge cases.**
- "Movies by Christopher Nolan" — director credit, Cat 1.
- "Movies starring Tom Hanks" — actor credit with prominence
  modifier; Cat 1 with the modifier absorbed.
- "Music by John Williams" / "Hans Zimmer score" — composer is
  indexed, Cat 1 (not Cat 36, which is acclaim-prose only).
- "Roger Deakins movies" — cinematographer is **not** indexed →
  Cat 40, not Cat 1.
- "Stephen King movies" in the sense of "based on Stephen King" →
  Cat 41 (source-material creator), not Cat 1. King has zero film
  acting/directing/writing footprint that the user typically means.
  [FLAG: confirm — King occasionally writes screenplays; do we
  treat ambiguous "X movies" defaults to Cat 41 for known
  novelists, or always emit both?]

**Good examples.**
- "starring Tom Hanks" → Cat 1 (actor)
- "directed by Greta Gerwig" → Cat 1 (director)
- "written by Aaron Sorkin" → Cat 1 (writer)

**Bad examples.**
- "shot by Roger Deakins" — looks like a credit, but
  cinematographer isn't indexed → **Cat 40**.
- "any movie called Inception" — name-shaped but it's a title
  query → **Cat 2**.
- "Tolkien movies" — named referent is a source-material author,
  not a film-credit person → **Cat 41**.

---

## 2. Title text lookup

**Core definition.** Substring match against `movie_card.title_normalized`.

**Boundary.** Owns title-string lookup only. Does not own person
names (Cat 1), franchises (Cat 5/6), or characters (Cat 3/6) even
when those names happen to also appear in titles.

**Edge cases.**
- "any movie called Inception" — explicit title framing → Cat 2.
- "movies with 'Star' in the title" — explicit title framing → Cat 2.
- "Star Wars" — surface form is a title fragment, but the user
  almost always means the franchise → **Cat 5**, not Cat 2. The
  "in the title" phrasing is the disambiguator.
- "Inception" alone (no "called" / "in the title" framing) — this
  is usually a "like Inception" expansion candidate → **Cat 42** if
  comparison framing is implied, or fall through to entity-shaped
  routing if literally referring to that one movie. [FLAG: confirm
  step 2's default for a bare title with no framing.]

**Good examples.**
- "any movie called The Matrix" → Cat 2
- "movies with 'Love' in the title" → Cat 2

**Bad examples.**
- "Star Wars movies" — franchise framing → **Cat 5**.
- "Tom Hanks" — person name → **Cat 1**.
- "Batman" — character-anchored franchise → **Cat 6**.

---

## 3. Named character lookup

**Core definition.** Named-character presence/prominence in films,
where the character does NOT anchor a film franchise of their own.

**Boundary.** Owns characters that appear in films but don't define
a film franchise on their own. Routes to **Cat 6** instead when the
named character anchors their own franchise (Batman, Bond,
Spider-Man, Wolverine, Indiana Jones, John Wick, Sherlock Holmes).
Routes to **Cat 1** for actors playing the character.

**Edge cases.**
- "Yoda appearances" — Yoda doesn't anchor "Yoda films"; he
  appears in Star Wars → Cat 3 (Yoda character) + Cat 5 (Star
  Wars franchise) emitted as two traits.
- "Hermione Granger scenes" — Hermione doesn't anchor a Hermione
  film franchise (Harry Potter does) → Cat 3 + Cat 6 (Harry Potter).
- "Loki appearances" — Loki has Disney+ shows but the *film*
  franchise is the MCU → Cat 3 + Cat 5 (MCU). [FLAG: confirm —
  if Loki ever gets a film franchise this routes to Cat 6.]
- **Detection rule reminder:** "Is there a 'List of [X] films'
  Wikipedia entity with the character as the through-line?" If
  yes → Cat 6. If no → Cat 3.

**Good examples.**
- "movies featuring Yoda" → Cat 3
- "any Hermione Granger appearance" → Cat 3
- "Severus Snape scenes" → Cat 3

**Bad examples.**
- "Batman movies" — anchors a franchise → **Cat 6**, not Cat 3 + Cat 5.
- "Daniel Radcliffe" — actor name, not character → **Cat 1**.
- "movies starring Sherlock Holmes" — anchors a franchise → **Cat 6**.

---

## 4. Studio / brand attribution

**Core definition.** Production-company or curated-brand attribution
(Disney, A24, Studio Ghibli, Hammer Films, Blumhouse).

**Boundary.** Owns production company / brand identity. Does not
own franchises, even when a franchise is closely identified with a
studio (Marvel as MCU is Cat 5/6, not Cat 4). Does not own cultural
tradition (Bollywood, Korean cinema) — those route to **Cat 23**.

**Edge cases.**
- "Disney movies" — brand → Cat 4. "Disney classics" splits into
  Cat 4 + Cat 13 + Cat 39 per compound rule.
- "A24" / "Blumhouse" / "Neon" — distributor brands also indexed →
  Cat 4.
- "Marvel movies" — ambiguous: brand (Marvel Studios) vs
  franchise (MCU). Default to franchise framing → **Cat 5**.
  [FLAG: confirm default — MCU is the more common user intent.]
- "Pixar" — owned by Disney but distinct brand → Cat 4 (Pixar
  brand), not Disney.

**Good examples.**
- "A24 horror" → Cat 4 (A24) + Cat 22 (horror)
- "Studio Ghibli movies" → Cat 4
- "Hammer Films" → Cat 4

**Bad examples.**
- "Marvel movies" — franchise-shaped → **Cat 5**.
- "Bollywood" — cultural tradition → **Cat 23**.
- "Hollywood movies" — too broad; usually means
  country-of-origin → **Cat 20**, not Cat 4.

---

## 5. Franchise / universe lineage

**Core definition.** Franchise/universe membership without a single
anchoring character — MCU, Star Wars, Fast & Furious, LOTR, Star
Trek, Mission Impossible (the franchise, not Ethan Hunt).

**Boundary.** Owns character-less franchises and lineage
positioning (sequel/prequel/spinoff/crossover, mainline-vs-offshoot,
reboot/remake position). Routes to **Cat 6** when the franchise is
inherently character-anchored (Batman, Bond, Spider-Man). Routes
to **Cat 7** when "based on" / "adaptation of" framing is paired
with the franchise name (composes — both fire).

**Edge cases.**
- "Star Wars novelizations" → Cat 5 + Cat 7 (Star Wars is
  character-less).
- "Mission Impossible movies" → Cat 5 (the franchise; Ethan Hunt
  is the protagonist but the franchise identity isn't "Ethan Hunt
  films" the way "Bond films" are Bond-anchored). [FLAG: confirm —
  this is borderline. Detection rule says character-anchored if
  there's a "List of [X] films" entity centered on the character.]
- "the original Scarface, not the remake" — lineage positioning,
  Cat 5 owns this even though Scarface isn't a multi-film franchise
  in the usual sense. [FLAG: should single-remake lineage queries
  go here or to Cat 7? Right now Cat 5 owns lineage arrays.]
- "MCU phase 4" — fine-grained subgroup → Cat 5.

**Good examples.**
- "MCU movies" → Cat 5
- "the Star Wars prequels" → Cat 5
- "Fast & Furious sequels" → Cat 5

**Bad examples.**
- "Batman movies" — character-anchored → **Cat 6**.
- "based on a comic book" — adaptation flag without a named
  franchise → **Cat 7** alone.
- "Marvel comics writers' movies" — vague reference class →
  **Cat 44**.

---

## 6. Character-franchise

**Core definition.** Named characters whose identity anchors a film
franchise — Batman, James Bond, Spider-Man, Wolverine, Indiana
Jones, John Wick, Jason Bourne, Sherlock Holmes, Harry Potter,
Tarzan, Robin Hood, Jack Ryan.

**Boundary.** Owns the dual-nature referent: one trait, fans out to
ENT character postings + FRA lineage internally. Never split into
Cat 3 + Cat 5. Routes to **Cat 3** if the character appears in a
franchise but doesn't anchor one. Routes to **Cat 5** if the
franchise has no single anchoring character.

**Edge cases.**
- "Sherlock Holmes books" → Cat 6 + Cat 7 (book adaptation flag).
  Books, plural, but the named referent is character-anchored.
- "James Bond remakes" → Cat 6 + Cat 7 (remake source-flag).
- "the Christopher Nolan Batman trilogy" → Cat 6 (Batman) + Cat 1
  (Nolan as director) — Cat 6 still owns Batman; Nolan is a
  separate trait.
- **Detection rule reminder:** Wikipedia "List of [X] films"
  entity exists with the character as through-line. Bond, Batman,
  Spider-Man — yes. Yoda, Hermione, Loki — no.
- "Harry Potter movies" — Harry Potter is the title character of
  the franchise → Cat 6. Note this is different from "Hermione
  Granger" which is Cat 3 + Cat 6.

**Good examples.**
- "Batman movies from the 80s" → Cat 6 (Batman) + Cat 13 (80s)
- "James Bond" → Cat 6
- "John Wick" → Cat 6

**Bad examples.**
- "Yoda appearances" — Yoda is in someone else's franchise →
  **Cat 3 + Cat 5**, not Cat 6.
- "Star Wars movies" — franchise has no single anchoring character →
  **Cat 5**.
- "Daniel Craig as Bond" — actor + character. Cat 1 (Craig) +
  Cat 6 (Bond), not Cat 3.

---

## 7. Adaptation source flag

**Core definition.** "Based on X" surface flags for source material
type — novel adaptation, comic book movie, video-game adaptation,
based on a true story, biography, remake-as-adaptation-flag.

**Boundary.** Owns the source-type flag itself (KW SourceMaterialType
family). Composes with **Cat 5/6/41** when a specific source is
named. Does not own lineage positioning ("the original, not the
remake" → **Cat 5**).

**Edge cases.**
- "remake" alone — Cat 7 (source flag, generic).
- "the original Scarface, not the remake" — Cat 5 (lineage
  positioning).
- "based on a comic book" — Cat 7 alone (no named source).
- "Stephen King novel adaptations" → Cat 41 (King) + Cat 7 (novel).
- "based on a true story" — Cat 7 (TRUE_STORY tag).
- "biopic" — Cat 7 (BIOGRAPHY tag); but if the subject is named
  ("JFK biopic"), composes with **Cat 8**.

**Good examples.**
- "novel adaptations" → Cat 7
- "video game movies" → Cat 7
- "based on a comic book" → Cat 7

**Bad examples.**
- "the original, not the remake" — lineage, not adaptation flag →
  **Cat 5**.
- "Stephen King movies" — names the creator → **Cat 41** (with
  Cat 7 if "novel" is also said).
- "documentary" — format, not adaptation → **Cat 25**.

---

## 8. Central topic / about-ness

**Core definition.** Concrete subject the film is *about* — focal
point the film orbits around. JFK, Vietnam War, Titanic, Watergate,
Princess Diana.

**Boundary.** Owns concrete-subject-of-film. Routes to **Cat 9**
when the framing is "has X" / mere presence. Routes to **Cat 32**
when the topic is thematic essence (grief, redemption, found
family) rather than a concrete subject. Routes to **Cat 31** when
"about" really means narrative setting (place/time).

**Edge cases.**
- "Titanic movie" — could mean (a) the 1997 film, (b) any movie
  about the Titanic. Surface form leans (b) here → Cat 8. If user
  says "the Titanic movie" with a definite article and clear
  reference → ambiguous. [FLAG: confirm step-2 default for "the X
  movie" with definite article.]
- "movies about the Vietnam War" — Cat 8.
- "set during the Vietnam War" — narrative setting → **Cat 31**.
- "movies about grief" — thematic essence → **Cat 32**, not Cat 8.
- "JFK biopic" → Cat 8 (JFK) + Cat 7 (biography flag).
- "shark movies" — "movies *with* sharks" framing → **Cat 9**, not
  Cat 8. "About" vs "has" is the disambiguator.

**Good examples.**
- "movies about the moon landing" → Cat 8
- "Watergate" → Cat 8
- "Titanic movies" (about the ship) → Cat 8

**Bad examples.**
- "movies with sharks" — element presence → **Cat 9**.
- "movies about grief" — thematic → **Cat 32**.
- "set in 1940s Berlin" — narrative setting → **Cat 31**.

---

## 9. Element / motif presence

**Core definition.** Concrete element appears in the story —
"has clowns," "zombie movies," "shark movies," "robots,"
"movies with horses."

**Boundary.** Owns "has X" / element-presence framing. Routes to
**Cat 8** when the element *is* the central topic ("about the
Vietnam War" not "has Vietnam War scenes"). Routes to **Cat 10**
for character archetypes (lovable rogue, anti-hero). Routes to
**Cat 32** for thematic abstraction (loneliness, found family).

**Edge cases.**
- "shark movies" — by convention, "shark movies" usually means
  shark-as-threat-element across many films (Jaws, Deep Blue Sea,
  The Meg) → Cat 9. Borderline with Cat 8 ("about sharks" as a
  documentary subject) but the colloquial reading is element.
- "zombie movies" — Cat 9. Zombie is a creature element, not a
  thematic abstraction.
- "movies with twist endings" — structural narrative device →
  **Cat 26**, not Cat 9.
- "movies with strong female leads" — character archetype → **Cat 10**.

**Good examples.**
- "movies with clowns" → Cat 9
- "zombie movies" → Cat 9
- "has a heist" → Cat 9 [FLAG: heist could be Cat 30 plot event;
  confirm — "movies with a heist" feels element-y, but a fully
  described heist plot is Cat 30.]

**Bad examples.**
- "movies about clowns" (focal subject) — closer to **Cat 8**.
- "anti-hero protagonists" — character archetype → **Cat 10**.
- "underdog stories" — story archetype → **Cat 32**.

---

## 10. Character archetype

**Core definition.** Static character type — "lovable rogue,"
"love-to-hate villain," "underdog protagonist," "femme fatale,"
"anti-hero," "reluctant hero."

**Boundary.** Owns static character-type framings. Routes to
**Cat 32** for character-arc / trajectory framings ("redemption
arc," "coming of age"). Routes to **Cat 9** for element presence.
Routes to **Cat 27** for audience-pitch framings ("kids movie").

**Edge cases.**
- "lone female protagonist" — compound. "Female protagonist" is
  archetype (FEMALE_LEAD) → Cat 10. "Lone" is structural form
  (single-lead) → **Cat 26**. Splits into two traits per the
  atomicity test.
- "anti-hero" — Cat 10.
- "redemption arc" — character trajectory, not static type →
  **Cat 32**.
- "morally grey characters" — Cat 10 (archetype framing).
- "ensemble cast" — structural form, not archetype → **Cat 26**.

**Good examples.**
- "anti-hero protagonist" → Cat 10
- "femme fatale" → Cat 10
- "reluctant hero" → Cat 10

**Bad examples.**
- "redemption arc" — trajectory → **Cat 32**.
- "ensemble" — structural → **Cat 26**.
- "movies with kids" — element/audience-adjacent → **Cat 9** or
  **Cat 27** depending on framing.

---

## 11. Award records

**Core definition.** Formal awards — wins, nominations,
ceremony-specific filters, multi-win superlatives.

**Boundary.** Owns structured ceremony/outcome data. Composes with
**Cat 38/39** for "Oscar-winning best picture" / quality-superlative
queries. Does not own qualitative quality language without award
references ("highly rated" → Cat 38).

**Edge cases.**
- "Oscar winners" → Cat 11.
- "Oscar-winning best picture" → Cat 11 + Cat 38 (general appeal)
  per compound split rule.
- "best horror" — no award reference → Cat 38, not Cat 11.
- "Cannes winners" → Cat 11.
- "nominated for Best Director" → Cat 11.

**Good examples.**
- "Academy Award winners" → Cat 11
- "Cannes Palme d'Or" → Cat 11
- "movies with multiple Oscars" → Cat 11

**Bad examples.**
- "highly acclaimed" — no formal award reference → **Cat 39**.
- "best of all time" — quality superlative → **Cat 38 + Cat 39**.
- "great movies" — vague quality → **Cat 38**.

---

## 12. Trending

**Core definition.** "Right now" / "trending" / "what's everyone
watching" — live-refreshed signal.

**Boundary.** Owns the live-refreshed trending channel. Does not
own static popularity ("popular movies" → Cat 38, which uses
ingest-time popularity_score). Does not own "recent releases"
(release-date framing → Cat 13).

**Edge cases.**
- "what's trending" → Cat 12.
- "popular right now" — the "right now" is the disambiguator →
  Cat 12.
- "popular movies" alone — static popularity → **Cat 38**.
- "what's everyone watching" → Cat 12.
- "recent hits" — ambiguous between Cat 12 (trending) and Cat 13
  (recent) + Cat 38 (popular). [FLAG: confirm — does "recent" gate
  to Cat 13 or fold into Cat 12's "right now" semantics?]

**Good examples.**
- "trending now" → Cat 12
- "what's everyone watching" → Cat 12
- "hot right now" → Cat 12

**Bad examples.**
- "popular movies" — static → **Cat 38**.
- "recent movies" — release-date → **Cat 13**.
- "best of 2024" — chronological + quality → **Cat 13 + Cat 38**.

---

## 13. Release date / era

**Core definition.** Date ranges and eras — "90s," "old," "recent,"
"before 2000," "old-school."

**Boundary.** Owns range/decay framings on release date. Routes to
**Cat 43** for ordinal position ("newest," "first"). Composes with
quality framings ("classic" splits into Cat 13 + Cat 39).

**Edge cases.**
- "90s movies" → Cat 13.
- "modern" / "recent" / "old-school" — vague-language framings;
  Cat 13 with user-preference defaults consulted.
- "modern classic" → Cat 13 (recent narrower range) + Cat 39
  (canonical).
- "the newest Scorsese" — ordinal → **Cat 43** + Cat 1 (Scorsese).
- "from 2010 onwards" → Cat 13.
- "old movies" → Cat 13. "Classic" → Cat 13 + Cat 39 per
  compound rule.

**Good examples.**
- "from the 90s" → Cat 13
- "before 2000" → Cat 13
- "recent" → Cat 13

**Bad examples.**
- "the newest one" — ordinal → **Cat 43**.
- "classic" — compound → **Cat 13 + Cat 39**.
- "trending" — live signal → **Cat 12**.

---

## 14. Runtime

**Core definition.** Movie duration — "around 90 minutes,"
"short," "long," "under 2 hours."

**Boundary.** Owns runtime range/falloff. The 60-min floor is a
dispatcher-level default, not a Cat 14 firing. Cat 14 fires only
when the user's query explicitly mentions duration.

**Edge cases.**
- "short films" — explicit short request → Cat 14 fires AND
  unlocks the 60-min floor (also engages **Cat 21** media_type=short).
  [FLAG: confirm composition — is this Cat 14 + Cat 21, or just
  Cat 21?]
- "around 90 minutes" → Cat 14.
- "epic length" / "marathon" — Cat 14 (long-falloff).

**Good examples.**
- "under 2 hours" → Cat 14
- "90 minutes" → Cat 14
- "short movies" → Cat 14 (+ possibly Cat 21)

**Bad examples.**
- "movies" generic — no runtime mention → no Cat 14.
- "long-running franchise" — that's franchise size, not runtime →
  **Cat 5**.

---

## 15. Maturity rating

**Core definition.** MPAA-style rating ceiling/floor — "PG-13 max,"
"rated R," "G-rated."

**Boundary.** Owns maturity_rank as a hard rating filter. Composes
with **Cat 27** when audience-pitch framing implies a rating
("family movies"). Composes with **Cat 28** when content-sensitivity
framing implies a rating ("nothing too graphic").

**Edge cases.**
- "PG-13" → Cat 15.
- "family-friendly" → Cat 27 + Cat 15 (Cat 27 carries inclusion
  scoring; Cat 15 carries the rating ceiling).
- "no R-rated" → Cat 15 (negative polarity ceiling).
- "nothing too graphic" — content-sensitivity → **Cat 28** (with
  Cat 15 if a rating ceiling is implied).

**Good examples.**
- "PG-13 or below" → Cat 15
- "rated R" → Cat 15
- "G-rated" → Cat 15

**Bad examples.**
- "for kids" — audience pitch → **Cat 27** (with Cat 15 ceiling).
- "no gore" — content sensitivity → **Cat 28**.
- "violent" — content presence → **Cat 28**.

---

## 16. Audio language

**Core definition.** Spoken/audio language — "in Korean,"
"Spanish-language," "subtitled."

**Boundary.** Owns the audio_language_ids column. Does not own
country of origin (Cat 20) or cultural tradition (Cat 23). A
Korean-language film may not be Korean-cinema and vice versa.

**Edge cases.**
- "in Korean" → Cat 16.
- "subtitled" → Cat 16 (non-English audio implied).
- "Korean cinema" — tradition → **Cat 23**, not Cat 16.
- "French movies" — ambiguous between Cat 16 (French audio),
  Cat 20 (French production), Cat 23 (French cinema tradition).
  [FLAG: default routing for "French movies" — I'd lean Cat 23
  for "movies"-as-cinema but Cat 20 for "made in France." Need
  default rule.]
- "dubbed in English" → Cat 16 with English dub specification.
  [FLAG: do we model dub-vs-original audio separately?]

**Good examples.**
- "Korean-language" → Cat 16
- "Spanish-language films" → Cat 16
- "in Japanese with subtitles" → Cat 16

**Bad examples.**
- "Bollywood" — tradition → **Cat 23**.
- "made in France" — production country → **Cat 20**.
- "filmed in Spain" — filming geography → **Cat 24**.

---

## 17. Streaming platform

**Core definition.** Available on a streaming service — "on
Netflix," "on Hulu," "on Prime."

**Boundary.** Owns providers (packed uint32). Does not own
availability windows or rental/purchase distinctions [FLAG: confirm
whether step 2 should pass through any rental/buy-vs-stream nuance].

**Edge cases.**
- "on Netflix" → Cat 17.
- "available to stream" without specific service — generic → still
  Cat 17 [FLAG: confirm — does this fire Cat 17 with no specific
  provider, or does it not fire at all?].
- "leaving Netflix soon" — temporal window → not currently
  modelable in Cat 17. [FLAG: drop, or punt to Cat 44?]

**Good examples.**
- "on Netflix" → Cat 17
- "streaming on Hulu" → Cat 17
- "Prime Video" → Cat 17

**Bad examples.**
- "Disney movies" — brand/studio → **Cat 4**, not Cat 17 even
  though Disney owns Disney+.
- "in theaters" — release window, not streaming. [FLAG: do we
  model theatrical-window? Currently no obvious home.]

---

## 18. Financial scale

**Core definition.** Budget and box-office magnitude — "big-budget,"
"low-budget," "indie scale," "blockbuster," "flop," "sleeper hit,"
"made bank."

**Boundary.** Owns budget_bucket and/or box_office_bucket. Merged
from old Cat 17 (budget) + Cat 18 (box office) because compound
language ("blockbuster" = big budget ∧ big gross) routinely spans
both. Does not own quality framing — "crowd-pleaser" / "cult flop"
still composes with **Cat 38/39**.

**Edge cases.**
- "blockbuster" → Cat 18 (compound: big budget + big gross).
- "indie hit" → Cat 18 (small budget + outsized gross).
- "sleeper hit" → Cat 18.
- "indie movies" — Cat 18 (low-budget framing). Distinct from
  "indie aesthetic" which would be **Cat 22** / **Cat 33**.
- "low-budget" → Cat 18 (budget axis only).
- "flop" → Cat 18 (gross axis, low end).
- "cult classic" — composes Cat 39 (cult framing) + Cat 13
  (older era) per compound rule, NOT Cat 18.

**Good examples.**
- "big-budget" → Cat 18
- "blockbuster" → Cat 18
- "indie scale" → Cat 18

**Bad examples.**
- "cult classic" — reception, not financial → **Cat 39 + Cat 13**.
- "underrated" — reception → **Cat 39**.
- "popular" — quality baseline → **Cat 38**.

---

## 19. Numeric reception score

**Core definition.** Specific numeric thresholds — "rated above 8,"
"70%+ on RT," "5-star."

**Boundary.** Owns explicit numeric threshold filters on
reception_score. Routes to **Cat 38** for qualitative framings
("well-rated," "highly acclaimed") which use the same column as
an additive prior, not a hard threshold.

**Edge cases.**
- "above 8 on IMDB" → Cat 19.
- "well-rated" — qualitative → **Cat 38**.
- "70%+ on Rotten Tomatoes" → Cat 19.
- "Oscar-worthy" — no number → **Cat 38/39**.
- "8 or higher" → Cat 19.

**Good examples.**
- "rated above 8" → Cat 19
- "5-star movies" → Cat 19
- "above 75% on RT" → Cat 19

**Bad examples.**
- "highly rated" — no threshold → **Cat 38**.
- "best" — superlative → **Cat 38 + Cat 39**.
- "well-reviewed" — no threshold → **Cat 38**.

---

## 20. Country of origin

**Core definition.** Legal/financial production country.

**Boundary.** Owns META.country_of_origin only. Does not own
filming geography (**Cat 24**) or cultural-tradition framings
(**Cat 23**). When a tradition tag exists, country_of_origin is
misleading and Cat 23 owns the trait instead.

**Edge cases.**
- "American movies" → Cat 20.
- "Korean cinema" → Cat 23 (tradition tag exists).
- "filmed in Iceland" → Cat 24 (filming geography).
- "Hollywood-funded HK action" — production-country US, but the
  cultural tradition is HK → **Cat 23** owns this; Cat 20 would
  point at the wrong column.
- "British production" → Cat 20.

**Good examples.**
- "produced in France" → Cat 20
- "American films" → Cat 20
- "British production" → Cat 20

**Bad examples.**
- "filmed in New Zealand" — filming → **Cat 24**.
- "Bollywood" — tradition → **Cat 23**.
- "Korean-language" — audio → **Cat 16**.

---

## 21. Media type

**Core definition.** Non-default media type — TV movies, video
releases, shorts (when explicit).

**Boundary.** Owns media_type. Does NOT fire for vanilla "show me
movies" (system default media_type=movie + 60-min runtime floor).
Fires only on explicit non-default request.

**Edge cases.**
- "TV movies" → Cat 21.
- "short films" → Cat 21 (and possibly Cat 14, see Cat 14 flag).
- "direct-to-video" → Cat 21.
- "movies" alone — system default, no Cat 21 firing.

**Good examples.**
- "TV movies" → Cat 21
- "shorts" → Cat 21
- "direct-to-video" → Cat 21

**Bad examples.**
- "movies" — default, no firing.
- "documentaries" — format, not media type → **Cat 25**.
- "anime" — format → **Cat 25**.

---

## 22. Genre

**Core definition.** All genre framings — top-level (horror,
action, comedy, sci-fi, drama, romance, animation) AND sub-genre
(body horror, neo-noir, cozy mystery, space opera, slasher).

**Boundary.** Owns genre identity. Routes to **Cat 32** for story
archetype framings ("revenge stories," "underdog"). Routes to
**Cat 25** for format ("documentary," "anime"). Routes to **Cat 33**
for tonal aesthetic that isn't genre-identity ("dark," "whimsical"
alone, without genre anchoring).

**Edge cases.**
- "horror movies" → Cat 22.
- "neo-noir" → Cat 22 (sub-genre).
- "dark action" — compound: "action" is genre → Cat 22; "dark" is
  tonal → **Cat 33**. Two traits.
- "revenge story" — story shape → **Cat 32**, not Cat 22.
- "comedy with horror elements" — splits into Cat 22 (comedy) +
  some horror trait. [FLAG: "horror elements" — Cat 22 (horror as
  qualifier), Cat 9 (horror as element-presence)? I'd lean Cat 22
  with salience=supporting.]
- "anime" — format → **Cat 25**, not Cat 22.

**Good examples.**
- "horror" → Cat 22
- "neo-noir" → Cat 22
- "cozy mystery" → Cat 22

**Bad examples.**
- "underdog stories" — archetype → **Cat 32**.
- "documentary" — format → **Cat 25**.
- "dark" alone — tonal aesthetic → **Cat 33**.

---

## 23. Cultural tradition / national cinema

**Core definition.** Aesthetic/movement traditions — "Korean
cinema," "Bollywood," "Hong Kong action," "Italian neorealism,"
"French New Wave."

**Boundary.** Owns tradition-as-aesthetic. Routes to **Cat 20** for
production-country framings ("made in France"). Routes to **Cat 16**
for audio-language framings.

**Edge cases.**
- "Bollywood" → Cat 23.
- "K-drama" — TV-shaped; if asked for movies, it's Korean-cinema
  adjacent → Cat 23. [FLAG: confirm.]
- "French movies" — see Cat 16 flag; default unclear.
- "Italian neorealism" → Cat 23.
- "Hong Kong action" → Cat 23 + Cat 22 (action). [FLAG: or just
  Cat 23 since the tradition tag bundles HK + action?]

**Good examples.**
- "Bollywood" → Cat 23
- "Korean cinema" → Cat 23
- "French New Wave" → Cat 23

**Bad examples.**
- "made in France" — production country → **Cat 20**.
- "in Korean" — audio → **Cat 16**.
- "J-horror" — borderline; tradition + genre. [FLAG: confirm.]

---

## 24. Filming location

**Core definition.** Actual shooting geography — "filmed in New
Zealand," "shot on location in Iceland."

**Boundary.** Owns filming_locations prose only. Distinct from
production country (**Cat 20**) and tradition (**Cat 23**).

**Edge cases.**
- "filmed in New Zealand" → Cat 24.
- "shot in Morocco" → Cat 24.
- "set in 1940s Berlin" — narrative setting, not filming →
  **Cat 31**.
- "on location" generic — filming framing without place name. [FLAG:
  does this fire Cat 24 with no specific location?]

**Good examples.**
- "shot in Iceland" → Cat 24
- "filmed in Morocco" → Cat 24
- "on location in Vietnam" → Cat 24

**Bad examples.**
- "set in Tokyo" — narrative setting → **Cat 31**.
- "Korean movies" — country/tradition → **Cat 20/23**.
- "American films" — production country → **Cat 20**.

---

## 25. Format + visual-format specifics

**Core definition.** Format (documentary, anime, mockumentary) and
visual-format specifics (B&W, 70mm, found-footage, widescreen,
handheld).

**Boundary.** Owns format identity and visual-format technique.
Routes to **Cat 22** for genre. Routes to **Cat 35** for visual
craft acclaim ("visually stunning") — Cat 25 is descriptive
format, not acclaim.

**Edge cases.**
- "documentary" → Cat 25.
- "anime" → Cat 25.
- "found-footage" → Cat 25.
- "shot on 70mm" — format technique → Cat 25; if framed as
  acclaim ("70mm masterpiece") composes with **Cat 35**.
- "B&W" → Cat 25.
- "mockumentary" → Cat 25.

**Good examples.**
- "documentary" → Cat 25
- "anime" → Cat 25
- "found-footage horror" → Cat 25 + Cat 22

**Bad examples.**
- "visually stunning" — acclaim → **Cat 35**.
- "horror" — genre → **Cat 22**.
- "TV movies" — media type → **Cat 21**.

---

## 26. Narrative devices + structural form + how-told craft

**Core definition.** Plot twist, nonlinear timeline, unreliable
narrator, single-location, anthology, ensemble, two-hander, POV
mechanics, character-vs-plot focus, "Sorkin-style" dialogue as
craft pattern.

**Boundary.** Owns *how the story is told* at the structural/device
level. Routes to **Cat 33** for pacing-as-experience ("slow burn,"
"frenetic") — that's experiential, not structural. Routes to
**Cat 37** for dialogue acclaim that isn't a structural pattern.

**Edge cases.**
- "twist ending" — structural ending type → ambiguous between
  Cat 26 (structural device) and Cat 33 (which owns "structural
  ending types" per its definition). Per `query_categories.md`
  Cat 33 explicitly absorbs ending types → **Cat 33**, not Cat 26.
  [FLAG: confirm — Cat 26 covers "plot twist" generically; Cat 33
  covers "twist ending" specifically. Surface form decides?]
- "nonlinear timeline" → Cat 26.
- "single-location thriller" → Cat 26 (single-location) + Cat 22
  (thriller).
- "ensemble cast" → Cat 26.
- "Sorkin-style dialogue" — depends on framing. If "snappy,
  rapid-fire" structural → Cat 26. If "great dialogue" acclaim →
  **Cat 37**. Often composes both.
- "lone protagonist" — single-lead structural form → Cat 26.

**Good examples.**
- "nonlinear timeline" → Cat 26
- "anthology film" → Cat 26
- "unreliable narrator" → Cat 26

**Bad examples.**
- "slow burn" — pacing experience → **Cat 33**.
- "great dialogue" — acclaim → **Cat 37**.
- "twist ending" — structural ending → **Cat 33** per planning doc.

---

## 27. Target audience

**Core definition.** The audience the film is pitched to — "family
movies," "teen movies," "kids movie," "for adults," "watch with the
grandparents."

**Boundary.** Owns packaged-audience framing. Routes to **Cat 32**
for story archetypes like coming-of-age. Routes to **Cat 28** for
content-sensitivity ("no gore"). Routes to **Cat 34** for concrete
viewing situation ("date night").

**Edge cases.**
- "family movies" → Cat 27 + Cat 15 (G/PG ceiling).
- "kids' movies" → Cat 27 + Cat 15.
- "for adults" — Cat 27 + possibly Cat 15 floor.
- "coming-of-age" — story archetype → **Cat 32**, not Cat 27.
- "watch with kids" — viewing situation → **Cat 34**, not Cat 27.
  [FLAG: borderline — "watch with kids" is also packaged-audience.
  Surface form: imperative-mood ("watch with X") = Cat 34;
  attribute-mood ("kids movie") = Cat 27.]

**Good examples.**
- "family-friendly" → Cat 27 (+ Cat 15)
- "teen movies" → Cat 27
- "for adults" → Cat 27

**Bad examples.**
- "coming-of-age" — archetype → **Cat 32**.
- "no gore" — content sensitivity → **Cat 28**.
- "date night" — situation → **Cat 34**.

---

## 28. Sensitive content

**Core definition.** Content presence/avoidance on its own
spectrum — "no gore," "not too bloody," "with nudity," "violent
but not graphic."

**Boundary.** Owns content-as-spectrum framing. Routes to **Cat 27**
for audience-pitch framing. Composes with **Cat 15** when a rating
ceiling is implied.

**Edge cases.**
- "no gore" → Cat 28 (negative polarity).
- "violent but not graphic" → Cat 28 (with nuance — violence OK,
  graphic-violence avoided).
- "with nudity" → Cat 28 (positive presence).
- "family-friendly" — audience pitch → **Cat 27**.
- "ANIMAL_DEATH" / "no animal harm" → Cat 28 (canonical KW tag).

**Good examples.**
- "no gore" → Cat 28
- "not too bloody" → Cat 28
- "no animal harm" → Cat 28

**Bad examples.**
- "family-friendly" — audience → **Cat 27**.
- "PG-13 only" — pure rating → **Cat 15**.
- "horror" — genre → **Cat 22**.

---

## 29. Seasonal / holiday

**Core definition.** Seasonal/holiday framings — Christmas,
Halloween, Thanksgiving, summer-blockbuster.

**Boundary.** Owns seasonal framings via proxy chains (no direct
seasonal tags). Composes additively across KW + CTX + P-EVT.

**Edge cases.**
- "Christmas movies" → Cat 29.
- "Halloween viewing" → Cat 29.
- "set on Christmas Eve" — narrative setting in Cat 29's P-EVT
  channel — emits Cat 29 (the seasonal trait). [FLAG: or split
  into Cat 29 + Cat 31?]
- "summer blockbuster" — composes Cat 29 (summer) + Cat 18
  (blockbuster financial scale).
- "Thanksgiving" → Cat 29.

**Good examples.**
- "Christmas movies" → Cat 29
- "Halloween viewing" → Cat 29
- "Thanksgiving family viewing" → Cat 29

**Bad examples.**
- "winter setting" — narrative setting → **Cat 31** (unless
  framed as seasonal viewing).
- "snowed-in plot" — element/plot → **Cat 9** or **Cat 30**.
- "summer movies" alone — ambiguous; likely Cat 29 if the user
  means "movies for summer viewing."

---

## 30. Plot events

**Core definition.** Literal plot events — "a heist crew unravels
when a member betrays them," "a man wakes up with no memory and
tries to find his wife's killer."

**Boundary.** Owns transcript-style plot prose. Routes to **Cat 31**
for narrative time/place setting (same vector space, different
prompt template). Routes to **Cat 32** for thematic essence vs
literal events.

**Edge cases.**
- "a man wakes up with no memory" → Cat 30.
- "set in 1940s Berlin" → Cat 31.
- "movies about grief" — thematic → **Cat 32**.
- "heist movies" — element-shaped → **Cat 9**, not Cat 30.
  Threshold: full event description → Cat 30; bare element name →
  Cat 9.
- Detailed multi-clause plot → Cat 30.

**Good examples.**
- "a heist crew gets double-crossed" → Cat 30
- "a man hunts down his wife's killer" → Cat 30

**Bad examples.**
- "heist movies" — element → **Cat 9**.
- "set in Tokyo" — setting → **Cat 31**.
- "about grief" — thematic → **Cat 32**.

---

## 31. Narrative setting (time/place)

**Core definition.** Narrative time setting ("set in 1940s Berlin")
or place setting ("takes place in Tokyo," "on a remote island").

**Boundary.** Owns "set in X" / "takes place in Y" framings. Same
P-EVT vector space as Cat 30, different prompt template. Routes to
**Cat 13** for production-era framings ("90s movies"). Routes to
**Cat 24** for filming geography. Routes to **Cat 8** for "about
Vietnam War" (concrete subject).

**Edge cases.**
- "set in 1940s Berlin" → Cat 31 (time + place).
- "takes place in space" → Cat 31.
- "set during WWII" → Cat 31.
- "90s movies" — production era → **Cat 13**, not Cat 31.
- "movies about WWII" — focal subject → **Cat 8**, not Cat 31.
- "Vietnam War movies" — ambiguous: "set during" (Cat 31), "about"
  (Cat 8). Default to Cat 8 [FLAG: confirm].

**Good examples.**
- "set in feudal Japan" → Cat 31
- "takes place in space" → Cat 31
- "1940s Berlin setting" → Cat 31

**Bad examples.**
- "90s movies" — release era → **Cat 13**.
- "filmed in Iceland" — filming → **Cat 24**.
- "about JFK" — focal subject → **Cat 8**.

---

## 32. Story / thematic archetype

**Core definition.** Story shape and thematic essence — "movies
about grief," "redemption arcs," "man-vs-nature," "underdog
stories," "revenge stories," "post-apocalyptic," "coming-of-age,"
"found-family stories."

**Boundary.** Owns thematic essence and story-shape archetypes.
Routes to **Cat 8** for concrete focal subjects (JFK, Titanic).
Routes to **Cat 10** for static character archetypes.

**Edge cases.**
- "redemption arc" → Cat 32.
- "coming-of-age" → Cat 32.
- "underdog story" → Cat 32.
- "anti-hero" — static archetype → **Cat 10**.
- "movies about grief" → Cat 32 (thematic), distinct from
  "movies about JFK" (Cat 8 — concrete).
- "post-apocalyptic" → Cat 32 (story shape, not genre per se).
  [FLAG: borderline with Cat 22 — post-apocalyptic is genre-coded.
  Confirm.]

**Good examples.**
- "redemption arc" → Cat 32
- "found family" → Cat 32
- "underdog stories" → Cat 32

**Bad examples.**
- "anti-hero" — static type → **Cat 10**.
- "about JFK" — concrete subject → **Cat 8**.
- "horror" — genre → **Cat 22**.

---

## 33. Emotional / experiential

**Core definition.** All emotional / experiential / feel framings —
during-viewing tone, pacing-as-experience, self-experience goals,
comfort-watch, post-viewing resonance, structural ending types.

**Boundary.** Anything emotional/experiential — before, during, or
after watching. Routes to **Cat 34** for concrete named viewing
situations ("date night"). Routes to **Cat 22** for genre identity.

**Edge cases.**
- "slow burn" → Cat 33 (pacing-as-experience).
- "make me cry" → Cat 33 (self-experience goal).
- "comfort watch" → Cat 33.
- "haunting" → Cat 33 (post-viewing resonance).
- "twist ending" → Cat 33 (structural ending type — explicitly
  owned by Cat 33 per planning doc, despite feeling structural).
- "happy ending" → Cat 33.
- "dark" — tonal → Cat 33. "Dark action" → Cat 22 (action) + Cat 33
  (dark).
- "feel-good" → Cat 33.
- "date night" → **Cat 34**, not Cat 33.

**Good examples.**
- "slow burn" → Cat 33
- "make me cry" → Cat 33
- "haunting" → Cat 33

**Bad examples.**
- "date night" — concrete situation → **Cat 34**.
- "horror" — genre → **Cat 22**.
- "twist plot device" (mid-film) — structural device → **Cat 26**;
  but "twist ending" specifically → Cat 33.

---

## 34. Viewing occasion

**Core definition.** Concrete named viewing situations — "date
night," "rainy Sunday," "long flight," "with kids on Saturday,"
"background watching," "family movie night."

**Boundary.** Owns named-event surface forms. Routes to **Cat 33**
for feelings/states ("comfort watch" alone is Cat 33; "Saturday
night with kids" is Cat 34). The carve is sharp: named event vs
feeling.

**Edge cases.**
- "date night" → Cat 34.
- "rainy Sunday" → Cat 34.
- "background viewing" → Cat 34.
- "comfort watch" — feeling/state → **Cat 33**.
- "watch with kids on a Saturday" → Cat 34 (named situation) +
  possibly Cat 27 (kids audience). [FLAG: confirm — does Cat 27
  also fire here?]
- "long flight viewing" → Cat 34.

**Good examples.**
- "date night" → Cat 34
- "rainy Sunday" → Cat 34
- "long flight" → Cat 34

**Bad examples.**
- "comfort watch" — feeling → **Cat 33**.
- "family movies" — audience pitch → **Cat 27**.
- "feel-good" — feeling → **Cat 33**.

---

## 35. Visual craft acclaim

**Core definition.** "Visually stunning," "killer cinematography,"
"beautifully shot," "IMAX-shot," "practical effects," "technical
marvel."

**Boundary.** Owns visual-craft acclaim. Routes to **Cat 1** for
named composer (composer is indexed). Routes to **Cat 40** for
named cinematographer / VFX supervisor (not yet indexed). Routes
to **Cat 25** for descriptive format ("shot on 70mm" without
acclaim framing).

**Edge cases.**
- "visually stunning" → Cat 35.
- "beautifully shot" → Cat 35.
- "Roger Deakins-shot" — named cinematographer → **Cat 40**.
- "70mm masterpiece" → Cat 35 (acclaim) + Cat 25 (format).
- "great VFX" → Cat 35.
- "IMAX-shot" — technique → Cat 35. [FLAG: pure description
  without acclaim — Cat 25 vs Cat 35? Surface form "shot in IMAX"
  feels Cat 25; "an IMAX experience" feels Cat 35.]

**Good examples.**
- "visually stunning" → Cat 35
- "killer cinematography" → Cat 35
- "practical effects" → Cat 35

**Bad examples.**
- "Roger Deakins" — named DP → **Cat 40**.
- "shot in B&W" — descriptive format → **Cat 25**.
- "great score" — music acclaim → **Cat 36**.

---

## 36. Music / score acclaim

**Core definition.** "Iconic score," "great soundtrack," "memorable
theme," "amazing music."

**Boundary.** Owns music acclaim only (RCP). Routes to **Cat 1**
for named composer (composer credits are indexed).

**Edge cases.**
- "iconic score" → Cat 36.
- "John Williams score" — named composer → **Cat 1** (composer
  credit), not Cat 36. If user says "the iconic John Williams
  score" → Cat 1 + Cat 36.
- "great soundtrack" → Cat 36.
- "musical" — genre → **Cat 22**, not Cat 36.

**Good examples.**
- "iconic score" → Cat 36
- "memorable theme music" → Cat 36
- "great soundtrack" → Cat 36

**Bad examples.**
- "John Williams" — composer → **Cat 1**.
- "musicals" — genre → **Cat 22**.
- "music biopic" — biography → **Cat 7 + Cat 8**.

---

## 37. Dialogue craft acclaim

**Core definition.** "Quotable dialogue," "Sorkin-style,"
"naturalistic dialogue," "snappy banter."

**Boundary.** Owns dialogue acclaim. Routes to **Cat 26** when the
framing is structural pattern (rapid-fire walk-and-talk as a how-
told device) rather than acclaim. Often composes both.

**Edge cases.**
- "Sorkin-style" — composes Cat 26 (structural pattern) + Cat 37
  (acclaim).
- "quotable dialogue" → Cat 37.
- "naturalistic dialogue" → Cat 37.
- "great script" — broader than dialogue → **Cat 39** (specific
  praise) [FLAG: confirm — could go Cat 37 if dialogue-focused.]

**Good examples.**
- "snappy dialogue" → Cat 37
- "quotable lines" → Cat 37
- "Sorkin-style" → Cat 37 (+ Cat 26)

**Bad examples.**
- "great script" — broader → **Cat 39**.
- "Aaron Sorkin movies" — credit → **Cat 1**.
- "rapid-fire structure" — pure structural → **Cat 26**.

---

## 38. General appeal / quality baseline

**Core definition.** Qualitative quality language without specific
numeric — "well-received," "highly rated," "popular," "best,"
"great," "highly regarded."

**Boundary.** Owns numeric column priors as additive lift. Routes
to **Cat 19** for explicit numeric thresholds. Routes to **Cat 39**
for specific praise/criticism prose.

**Edge cases.**
- "well-rated" → Cat 38.
- "popular" → Cat 38 (static popularity_score).
- "trending" — live signal → **Cat 12**, not Cat 38.
- "best horror of the 80s" → Cat 38 + Cat 39 + Cat 22 (horror) +
  Cat 13 (80s) per compound rule.
- "classic" → Cat 13 + Cat 39 (canonical), not Cat 38.
- "rated above 8" — explicit number → **Cat 19**.

**Good examples.**
- "well-received" → Cat 38
- "highly regarded" → Cat 38
- "popular" → Cat 38

**Bad examples.**
- "rated above 8" — numeric → **Cat 19**.
- "cult classic" — specific reception → **Cat 39 + Cat 13**.
- "trending" — live → **Cat 12**.

---

## 39. Specific praise / criticism

**Core definition.** Reception prose for what people specifically
liked or disliked, plus canonical reception tags. "Cult,"
"underrated," "overhyped," "divisive," "praised for its tension,"
"criticized as plodding," "still holds up," "era-defining,"
"stacked cast," "thematic weight."

**Boundary.** Owns quality-as-prose. Routes to **Cat 38** for
quality-as-numeric-prior. Routes to **Cat 11** for formal awards.

**Edge cases.**
- "cult classic" → Cat 39 (cult) + Cat 13 (older era).
- "underrated" → Cat 39.
- "Oscar-winning" — formal award → **Cat 11**.
- "stacked cast" → Cat 39.
- "praised for its tension" → Cat 39.
- "still holds up" → Cat 39.
- "classic" — compound → Cat 13 + Cat 39.

**Good examples.**
- "underrated" → Cat 39
- "cult classic" → Cat 39 (+ Cat 13)
- "divisive" → Cat 39

**Bad examples.**
- "well-rated" — numeric prior → **Cat 38**.
- "Oscar-winning" — award → **Cat 11**.
- "rated above 8" — numeric → **Cat 19**.

---

## 40. Below-the-line creator lookup

**Core definition.** Cinematographer, editor, production designer,
costume designer, VFX supervisor — "Roger Deakins movies," "Thelma
Schoonmaker-edited," "Sandy Powell costumes," "Colleen Atwood
designs."

**Boundary.** Owns below-the-line credits. **Reserved — returns
empty until backing data lands.** Still routes here so step 2
doesn't misroute these names to Cat 1 (which has no posting for
these roles and would silently return zero).

**Edge cases.**
- "Roger Deakins" → Cat 40.
- "Thelma Schoonmaker" → Cat 40.
- "Christopher Nolan" — director, indexed → **Cat 1**, not Cat 40.
- "John Williams" — composer, indexed → **Cat 1**.

**Good examples.**
- "Roger Deakins" → Cat 40
- "Sandy Powell costumes" → Cat 40
- "Greig Fraser-shot" → Cat 40

**Bad examples.**
- "Christopher Nolan" — director → **Cat 1**.
- "John Williams" — composer → **Cat 1**.
- "great cinematography" — acclaim, no name → **Cat 35**.

---

## 41. Named source creator

**Core definition.** Named creator of source material — "Stephen
King novels," "Tolkien films," "based on Shakespeare plays," "Philip
K. Dick," "Neil Gaiman," "Jane Austen."

**Boundary.** Owns named creators of source material via P-EVT +
RCP semantic search. Detection rule for "based on / by / X's
<medium>":
- Character-anchored film franchise (the character's own films) →
  **Cat 6** (e.g. "Sherlock Holmes books" → Cat 6 + Cat 7).
- Character-less film franchise → **Cat 5** (e.g. "Star Wars
  novelizations" → Cat 5 + Cat 7).
- Creator of source material → **Cat 41** (e.g. "Shakespeare
  plays" → Cat 41 + Cat 7).
- Generic medium with no named referent → **Cat 7** alone.

**Edge cases.**
- "Stephen King novels" → Cat 41 + Cat 7.
- "Sherlock Holmes books" — Sherlock anchors a film franchise →
  Cat 6 + Cat 7, NOT Cat 41.
- "Star Wars novelizations" — character-less franchise → Cat 5 +
  Cat 7, NOT Cat 41.
- "Tolkien films" → Cat 41 + Cat 7. Tolkien is a creator, even
  though LOTR is a franchise — the user named the creator, not
  the franchise. [FLAG: confirm — is "Tolkien films" Cat 41 only,
  or Cat 41 + Cat 5 (LOTR franchise)? My read: Cat 41 only since
  "Tolkien" is the surface form.]
- "Stephen King movies" without "novels" — defaults to Cat 41
  (King is a known novelist, not a film-credit person).

**Good examples.**
- "Stephen King novels" → Cat 41 + Cat 7
- "Shakespeare adaptations" → Cat 41 + Cat 7
- "Neil Gaiman" → Cat 41

**Bad examples.**
- "Sherlock Holmes books" — character-anchored → **Cat 6 + Cat 7**.
- "Star Wars novelizations" — character-less franchise → **Cat 5 +
  Cat 7**.
- "Christopher Nolan movies" — film director, not source-material
  creator → **Cat 1**.

---

## 42. "Like &lt;media&gt;" reference

**Core definition.** Named-work comparison — "like Inception,"
"similar to The Office," "movies that feel like David Lynch," "in
the vein of Hitchcock thrillers," "like a Coen Brothers movie."

**Boundary.** Owns named-work expansion. Triggers on explicit
comparison surface forms only — "like X," "similar to Y," "in the
vein of Z," "X-style," "feels like Q." Routes to **Cat 44** for
vague reference classes without a named comparison target
("comedians doing drama").

**Edge cases.**
- "like Inception" → Cat 42 (named work).
- "feels like a Coen Brothers movie" → Cat 42 (named auteurs).
- "Hitchcock-style thrillers" → Cat 42 (named auteur with style
  framing).
- "comedians doing drama" — vague class → **Cat 44**.
- "in the style of A24" — composes? "A24" is a brand → Cat 4 with
  "in the style of" framing. [FLAG: confirm — Cat 4 alone, or
  Cat 42 with A24 as the named referent?]
- "auteur thrillers" — vague class → **Cat 44**.

**Good examples.**
- "like Inception" → Cat 42
- "similar to The Office" → Cat 42
- "in the vein of Hitchcock" → Cat 42

**Bad examples.**
- "comedians doing drama" — vague class → **Cat 44**.
- "Christopher Nolan movies" — direct credit, not comparison →
  **Cat 1**.
- "Inception" alone — see Cat 2 flag.

---

## 43. Chronological ordinal

**Core definition.** Release-date ordinal position — "first,"
"last," "earliest," "latest," "most recent," "the newest one,"
"the oldest one."

**Boundary.** Owns ordinal selection by release_date. Routes to
**Cat 13** for range/decay framings ("recent" without "most").
Routes to **Cat 38/39** for quality superlatives ("best," "most
acclaimed").

**Edge cases.**
- "the newest Scorsese" → Cat 43 + Cat 1.
- "the latest Marvel movie" → Cat 43 + Cat 5 (MCU).
- "most recent" → Cat 43.
- "recent" alone — range, not ordinal → **Cat 13**.
- "best Bond movie" — quality superlative → **Cat 38/39 + Cat 6**,
  not Cat 43.
- "first Indiana Jones" → Cat 43 + Cat 6.

**Good examples.**
- "the newest one" → Cat 43
- "earliest film" → Cat 43
- "most recent release" → Cat 43

**Bad examples.**
- "recent movies" — range → **Cat 13**.
- "best Bond" — quality superlative → **Cat 38/39 + Cat 6**.
- "trending" — live signal → **Cat 12**.

---

## 44. Generic parametric / catch-all

**Core definition.** Anything that needs interpretation/expansion
and doesn't fit a structured category. Vague reference classes
("comedians doing drama," "auteur directors of the 70s," "child
actors who became serious"), named lists / curated canon (Criterion
Collection, AFI Top 100, IMDb Top 250, BFI, NFR, Sight & Sound),
and anything else step 2 recognizes as real but underspecified.

**Boundary.** Last-resort routing. Routes to **Cat 42** when the
expansion is of a *named work* rather than a class. Step 2 should
exhaust all structured categories before reaching for Cat 44.

**Edge cases.**
- "Criterion Collection" → Cat 44.
- "AFI Top 100" → Cat 44.
- "comedians doing drama" → Cat 44.
- "auteur directors of the 70s" → Cat 44 + Cat 13 (era is still
  structured).
- "directors known for long takes" → Cat 44.
- "movies film school students watch" → Cat 44.
- "like Inception" — named work → **Cat 42**, not Cat 44.

**Good examples.**
- "Criterion Collection" → Cat 44
- "comedians doing drama" → Cat 44
- "AFI Top 100" → Cat 44

**Bad examples.**
- "like Inception" — named-work expansion → **Cat 42**.
- "horror movies" — clean genre fit → **Cat 22**.
- "Tom Hanks comedies" — clean structured fit → **Cat 1 + Cat 22**.

---

## Cross-cutting routing rules (recap)

These are the highest-frequency disambiguators step 2 will need:

1. **"About X" vs "has X" vs "set in X" vs "X-style".**
   - About → Cat 8 (concrete) / Cat 32 (thematic).
   - Has → Cat 9 (element) / Cat 10 (archetype).
   - Set in → Cat 31 (narrative setting).
   - Like X / X-style → Cat 42.
2. **Named referent in "based on" / "by" / "X's <medium>".**
   - Character-anchored film franchise → Cat 6 + Cat 7.
   - Character-less film franchise → Cat 5 + Cat 7.
   - Source-material creator → Cat 41 + Cat 7.
   - Generic medium → Cat 7 alone.
3. **Quality language.**
   - Numeric threshold → Cat 19.
   - Numeric prior (additive) → Cat 38.
   - Reception prose → Cat 39.
   - Formal awards → Cat 11.
4. **Era/recency framings.**
   - Range/decay → Cat 13.
   - Ordinal → Cat 43.
   - Live trending → Cat 12.
5. **Format vs genre vs media-type.**
   - Genre identity → Cat 22.
   - Format (doc, anime, B&W) → Cat 25.
   - Media type (TV-movie, short, video) → Cat 21.
6. **Tonal/experiential vs structural.**
   - Pacing-as-experience, ending-types, comfort, resonance → Cat 33.
   - Structural devices (nonlinear, anthology, ensemble) → Cat 26.
   - Concrete viewing situation → Cat 34.
   - Audience pitch → Cat 27.
   - Content sensitivity → Cat 28.
