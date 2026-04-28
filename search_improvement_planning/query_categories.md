# Query Categories (final)

Each category names a conceptual flavor of question AND the concrete
set of endpoints the dispatcher fans out to when that category fires.
Categories **compose**: a single user query routinely invokes several
(e.g. "Tom Hanks comedies from the 90s" hits Cat 1 + Cat 22 + Cat 13).
Splits were made wherever a sub-case would call a *different* set of
endpoint families, OR wherever the handler-stage prompt would have to
do internal LLM-style branching that step 2 could pre-resolve from the
trait surface.

**Endpoint-family shorthand.** ENT · FRA · STU · KW · META · AWD ·
TRENDING · P-EVT · P-ANA · VWX · CTX · NRT · PRD · RCP · PARAMETRIC.
Semantic sub-spaces (P-EVT, P-ANA, VWX, CTX, NRT, PRD, RCP) are
treated as distinct endpoint families for category-split purposes
because each one embeds a different information surface and is
phrased to mirror its own ingestion text format (per v3 #18).

**ANC dropped.** The dense_anchor capsule was previously listed as a
semantic endpoint but is no longer searched directly — it's too vague
compared to the specific component spaces, which carry sharper signal
individually. Any capsule-level vibe that previously routed to ANC
now routes to whichever specific space carries the relevant sub-axis.

---

## Global rules

- **Per-trait step-2 payload:** role (carver/qualifier), polarity
  (positive/negative), category (1-44), salience (qualifier-only:
  central/supporting). No `framing_mode` field — salience absorbs
  spectrum handling: a "kind of about grief" qualifier is just
  `supporting` salience, and downstream weighting handles the
  diminished impact correctly.
- **Categories compose:** a single query routinely fires several.
  See composition notes at the bottom.
- **Compound split rule:** if a phrase fits multiple categories, that
  is a signal it should be decomposed into separate atomic traits.
  Never invent an umbrella category to absorb a compound. The one
  exception is dual-nature *referents* (a single name that is
  inherently both a character and a franchise) — see Cat 6.
- **Derive once.** Any decision step 2 can make confidently from the
  trait surface should be made there, not re-derived inside a handler.
  This principle drives the per-attribute split of META (Cats 13-21)
  and the per-axis split of craft acclaim (Cats 35-37) from the older
  taxonomy — adding enum values to step 2 is cheaper than forcing the
  handler to re-classify on every call.
- **One trait, one category (1:1).** Each atomic trait carries
  exactly one category label. When a single referent inherently
  needs two endpoint families (e.g. character-anchored franchise),
  a dedicated category exists to fan out internally rather than
  emitting duplicate traits. See Cat 6 for the canonical case.

---

## Structured / lexical categories

### 1. Person credit
**Endpoints:** ENT.
**Handles:** actor, director, writer, producer, composer credits.
**Mechanism:** normalize → `lex.inv_<role>_postings` intersect.
**Boundaries:** indexed roles only. Below-the-line creators
(cinematographer, editor, production designer, costume designer,
VFX supervisor) → Cat 40. Title-based searches → Cat 2.

### 2. Title text lookup
**Endpoints:** ENT.
**Handles:** title substring matches ("any movie called Inception,"
"movies with 'Star' in the title").
**Mechanism:** `movie_card.title_normalized ILIKE`.
**Boundaries:** split from Cat 1 because the mechanism (column
ILIKE) and prompt shape (free string match without role typing)
differ from posting-table credit lookup. Same input shape (a
string), different SQL, different precision behavior.

### 3. Named character lookup
**Endpoints:** ENT.
**Handles:** named-character presence and prominence — characters
who appear in films but don't anchor a franchise of their own
("movies featuring Yoda," "any Hermione Granger appearance,"
"Aragorn scenes," "Severus Snape," "Loki appearances").
**Mechanism:** ENT character postings with prominence filter
(CENTRAL/DEFAULT).
**Boundaries:** when the named character *does* anchor a film
franchise (Batman, James Bond, Spider-Man, Wolverine, Indiana
Jones), route to Cat 6 instead — Cat 6 emits one trait that fans
out to ENT + FRA in parallel, rather than splitting into a Cat 3
trait + Cat 5 trait. Cat 3 is the residual case: characters that
appear in films but don't define a franchise on their own. The
prominence-mode subset differs from Cat 1 (CENTRAL/DEFAULT for
characters, LEAD/SUPPORTING/MINOR/DEFAULT for actors), which is
why this isn't folded into Cat 1.

### 4. Studio / brand attribution
**Endpoints:** STU.
**Handles:** production company and curated-brand queries (Disney,
A24, Studio Ghibli, Hammer Films).
**Mechanism:** ProductionBrand enum path (brand_id posting, with
rename-chain time-bounding) OR freeform token intersect with DF
ceiling.
**Boundaries:** brand rename handling and DF-filtered token
intersection are unique to STU; no other channel carries
production-company signal with discipline.

### 5. Franchise / universe lineage
**Endpoints:** FRA.
**Handles:** franchise/universe membership *without* a single
anchoring character — Marvel Cinematic Universe, Star Wars, Fast &
Furious, LOTR, Star Trek, Mission Impossible (the franchise name
itself, not Ethan Hunt the character) — plus lineage position
(sequel/prequel/spinoff/crossover/launched), mainline-vs-offshoot,
reboot/remake positioning.
**Mechanism:** FRA two-phase token resolution → array overlap on
lineage / shared_universe / subgroup. Remake-as-lineage-positioning
("the original Scarface, not the remake") is handled by the lineage
arrays themselves, not by a separate keyword channel.
**Boundaries:** when the named referent is anchored on a single
character (Batman, James Bond, Sherlock Holmes, Indiana Jones),
route to Cat 6 instead — those names emit one combined character +
franchise trait, never a Cat 3 + Cat 5 split. Cat 5 covers the
residual case: franchises whose identity isn't a single character.
When "based on" phrasing names a *creator of source material*, the
referent routes to Cat 41 — see Cat 41 for the detection rule.
Source-flag composition still applies: "Star Wars novelizations"
→ Cat 5 (Star Wars franchise) + Cat 7 (novel source-flag).

### 6. Character-franchise
**Endpoints:** ENT + FRA (combo).
**Handles:** named characters whose identity *anchors* a film
franchise — Batman, James Bond, Spider-Man, Wolverine, Indiana
Jones, John Wick, Jason Bourne, Sherlock Holmes, Harry Potter,
Tarzan, Robin Hood, Jack Ryan.
**Mechanism:** ENT character postings (CENTRAL/DEFAULT prominence)
AND FRA two-phase token resolution → array overlap on lineage /
shared_universe / subgroup, fired in parallel. Scores combine
additively under the standard combo orchestration shape.
**Boundaries:** **detection rule** — fire Cat 6 when the named
referent is *both* a character AND anchors a film franchise named
after or centered on them (test: does a "List of [X] films"
entity exist, with the character as the through-line across
films?). One trait, one category — never split into a Cat 3 trait
+ Cat 5 trait. Cat 3 stays for characters who appear in someone
else's franchise but don't anchor their own (Yoda, Hermione
Granger, Aragorn, Severus Snape, Loki). Cat 5 stays for franchises
without a single anchoring character (MCU, Star Wars, LOTR,
Fast & Furious, Star Trek). This is the *only* category that
exists specifically to absorb a dual-nature referent — every other
multi-endpoint case in the taxonomy is either a query-shape
compound (split per the compound split rule) or a clean
combo/tiered orchestration within a single conceptual category.
Source-flag composition still applies: "James Bond remakes" →
Cat 6 + Cat 7 (remake source-flag); "Sherlock Holmes books" →
Cat 6 + Cat 7 (novel source-flag).

### 7. Adaptation source flag
**Endpoints:** KW.
**Handles:** "novel adaptation," "comic book movie," "based on a
true story," "video-game adaptation," "biography," "remake" (as an
adaptation flag rather than lineage positioning).
**Mechanism:** KW SourceMaterialType family single-overlap.
**Boundaries:** composes with Cat 5 (when the named source is a
character-less franchise), Cat 6 (when the named source is a
character-anchored franchise), or Cat 41 (when the named source is
a creator) — any of those fire alongside Cat 7 when the query
names a specific source.

### 8. Central topic / about-ness
**Endpoints:** P-EVT + KW (tiered, keyword-first).
**Handles:** "about JFK," "Titanic movie," "Watergate," "Princess
Diana biopic," "Vietnam War." The film's central concrete subject —
the thing the movie *is about*, not just *contains*.
**Mechanism:** if the subject resolves to a canonical keyword tag
(BIOGRAPHY, TRUE_STORY, historical-event tags), query KW only and
stop. If no tag covers the request, or framing is spectrum ("loose
allegorical biography"), fall back to P-EVT prose.
**Boundaries:** concrete subjects only (JFK, Vietnam War, Titanic).
Thematic essence (grief, redemption, found family) → Cat 32. Mere
presence ("has clowns," "zombie movies") → Cat 9. Subject is the
*focal point* the film orbits around, not an element that happens
to appear.

### 9. Element / motif presence
**Endpoints:** P-EVT + KW (tiered, keyword-first).
**Handles:** "has clowns," "zombie movies," "shark movies,"
"robots," "movies with horses." Concrete element appears in the
story.
**Mechanism:** if the motif resolves to a canonical keyword tag
(ZOMBIE, CLOWN, SHARK, ROBOT), query KW only and stop. If no tag
covers the request, fall back to P-EVT prose.
**Boundaries:** "has X" framing, not "about X" (Cat 8). Character
archetype ("lovable rogue") → Cat 10. Element-presence is binary
("is this thing in the story?"); subject-of-film (Cat 8) is about
centrality.

### 10. Character archetype
**Endpoints:** KW + NRT (tiered, keyword-first).
**Handles:** "lovable rogue," "love-to-hate villain," "underdog
protagonist," "femme fatale," "anti-hero," "reluctant hero."
**Mechanism:** if the archetype matches a canonical ConceptTag
(ANTI_HERO, FEMALE_LEAD), query KW only and stop. Fallback: NRT
characterization prose.
**Boundaries:** archetype = static character type. Character arc /
trajectory → Cat 32 (story/thematic archetype). Element-presence
in story → Cat 9.

### 11. Award records
**Endpoints:** AWD.
**Handles:** formal wins, nominations, ceremony-specific filters,
multi-win superlatives.
**Mechanism:** `movie_awards` filter with COUNT thresholds; fast
path on `award_ceremony_win_ids`.
**Boundaries:** structured ceremony/outcome data lives only in AWD.
Quality-superlative queries that mention awards ("Oscar-winning
best picture") compose Cat 11 with Cats 38/39 per the compound
split rule.

### 12. Trending
**Endpoints:** TRENDING.
**Handles:** "right now," "trending," "what's everyone watching."
**Mechanism:** live-refreshed trending signal.
**Boundaries:** TRENDING is the only channel with refresh cadence.
META.popularity_score (used in Cat 38) is static ingest-time and
misses "right now" semantics entirely.

---

## Structured single-attribute (META, per attribute)

Each attribute is its own category with one column, one scoring
shape, and one prompt. Step 2 picks the attribute; the handler
dispatches deterministically without further LLM judgment. Adding
nine enum values to step 2's category list is cheaper than forcing
a unified META handler to re-classify on every call.

### 13. Release date / era
**Endpoints:** META.release_date.
**Handles:** date ranges, eras, "90s," "old," "recent," "before
2000," "in the 2000s," "old-school."
**Mechanism:** range filter with soft-falloff for vague language
(per v3 #3); user-preference defaults consulted for "modern" /
"recent" / "old-school" (per v3 #6).
**Boundaries:** range/decay framings only. Ordinal position
("newest," "earliest") → Cat 43.

### 14. Runtime
**Endpoints:** META.runtime.
**Handles:** "around 90 minutes," "short," "long," "under 2 hours."
**Mechanism:** range filter with soft-falloff.
**System default:** runtime ≥ 60 min floor applied to all queries
unless the query explicitly asks for shorts (per v3 #2). The
default is enforced at the dispatcher level, not by Cat 14 firing.

### 15. Maturity rating
**Endpoints:** META.maturity_rank.
**Handles:** "PG-13 max," "rated R," "G-rated."
**Boundaries:** when maturity is a packaged-audience framing
("family movies") or content-sensitivity framing ("nothing too
graphic"), it composes with Cat 27 or Cat 28 — Cat 15 still fires
for the rating ceiling itself, while 27/28 carry the inclusion-
scoring side.

### 16. Audio language
**Endpoints:** META.audio_language_ids.
**Handles:** "in Korean," "Spanish-language," "subtitled."

### 17. Streaming platform
**Endpoints:** META.providers (packed uint32).
**Handles:** "on Netflix," "on Hulu," "on Prime."
**Mechanism:** packed-uint32 decode against the provider key
encoding written at ingest.

### 18. Financial scale
**Endpoints:** META.budget_bucket + META.box_office_bucket.
**Handles:** budget framings ("big-budget," "low-budget," "indie
scale," "shoestring," "studio-scale"), box-office framings ("box
office hit," "blockbuster gross," "flop," "underperformer," "sleeper
hit," "made bank"), and compound framings that bundle both
("blockbuster" = big budget ∧ big gross; "indie hit" = small budget
∧ outsized gross).
**Mechanism:** META range/bucket lookup against budget_bucket and/or
box_office_bucket; the handler picks the relevant column(s) from
the trait surface (budget-only vs gross-only vs combined).
**Boundaries:** merged from the former Cat 17 (budget) and Cat 18
(box office) — one user word ("blockbuster," "indie") routinely
spans both, and the "compound split rule" specifically discourages
splitting when both halves dispatch to the same endpoint family
with the same orchestration shape. Quality framing ("crowd-pleaser,"
"cult flop") still composes with Cats 38/39 separately — only the
financial axis itself collapses.

### 19. Numeric reception score
**Endpoints:** META.reception_score (as threshold).
**Handles:** specific numeric thresholds — "rated above 8," "70%+
on RT," "5-star."
**Boundaries:** specific numeric only. Qualitative quality language
("well-rated," "best") → Cat 38 (general appeal as numeric prior).
Same column read by both, but framing differs — threshold-filter
vs additive prior.

### 20. Country of origin
**Endpoints:** META.country_of_origin.
**Handles:** "produced in," "American films," "British production."
**Boundaries:** legal/financial production country only. Filming
geography → Cat 24. Cultural-tradition framing → Cat 23.

### 21. Media type
**Endpoints:** META.media_type.
**Handles:** "TV movies," "video releases," "shorts (when
explicit)."
**Boundaries:** does not fire for vanilla "show me movies" — system
default `media_type = movie` plus Cat 14's 60-min runtime floor
cover normal cases. Fires only when a non-default media type is
explicitly requested.

---

## Structured / KW continuing

### 22. Genre
**Endpoints:** KW + P-ANA (mutually exclusive per query).
**Handles:** all genre framings, top-level (horror, action, comedy,
sci-fi, drama, romance, animation) AND sub-genre (body horror,
neo-noir, cozy mystery, space opera, slasher).
**Mechanism:** KW genre-family or sub-genre tag if canonical →
P-ANA `genre_signatures` prose for compound/qualifier-laden ("dark
action," "quiet drama") or sub-genres without tags.
**Boundaries:** merged from former top-level and sub-genre cats —
the top/sub line was too fragile for step 2 to draw reliably, and
both feed the same fallback space (`genre_signatures`). Story
archetype ("revenge," "underdog") → Cat 32 instead — that
distinction *is* sharp at step 2 because archetype names a story
shape rather than a genre.

### 23. Cultural tradition / national cinema
**Endpoints:** KW + META (mutually exclusive, keyword-first).
**Handles:** "Korean cinema," "Bollywood," "Hong Kong action,"
"Italian neorealism," "French New Wave."
**Mechanism:** KW tradition tag (BOLLYWOOD, KOREAN_CINEMA,
ITALIAN_NEOREALISM) → META country/language fallback when no tag.
**Boundaries:** tradition-as-aesthetic, not legal production
country (Cat 20) or filming geography (Cat 24). If a tradition tag
exists, the country column is misleading (Hollywood-funded HK
action isn't HK by production country); only when no tag exists
does country-of-origin become the best remaining proxy.

### 24. Filming location
**Endpoints:** PRD (`filming_locations` prose).
**Handles:** "filmed in New Zealand," "shot on location in
Iceland," "Morocco shoots."
**Mechanism:** PRD prose similarity against the filming_locations
field — the only channel carrying actual shooting-location data.
**Boundaries:** META.country_of_origin is the wrong column. It
records legal/financial production country, not filming geography
(The Revenant in Canada/Argentina, Dune in Jordan/UAE, Mission
Impossible — Fallout across Kashmir/UAE/NZ all carry US
country_of_origin). Distinct from Cat 23 (cultural tradition) and
Cat 20 (production country).

### 25. Format + visual-format specifics
**Endpoints:** KW + PRD (tiered, keyword-first).
**Handles:** format (documentary, anime, mockumentary), visual-
format specifics (B&W, 70mm, found-footage, widescreen, handheld).
**Mechanism:** canonical format/visual-format tag → KW; technique-
level long-tail without tags → PRD prose.

### 26. Narrative devices + structural form + how-told craft
**Endpoints:** KW + NRT (tiered, keyword-first).
**Handles:** plot twist, nonlinear timeline, unreliable narrator,
single-location, anthology, ensemble, two-hander, POV mechanics,
character-vs-plot focus, "Sorkin-style" dialogue as craft pattern.
**Mechanism:** canonical device tag (PLOT_TWIST, NONLINEAR_TIMELINE,
UNRELIABLE_NARRATOR, SINGLE_LOCATION, ENSEMBLE_CAST) → KW; craft-
level long-tail → NRT prose.
**Boundaries:** pacing-as-experience ("slow burn," "frenetic")
routes to Cat 33, not here — that's experiential, not structural.
This category is about *how the story is told* at the structural/
device level.

### 27. Target audience
**Endpoints:** KW + META + CTX (gate + inclusion, query-dependent).
**Handles:** "family movies," "teen movies," "kids movie," "for
adults," "watch with the grandparents." The audience being pitched
to.
**Mechanism:** META.maturity_rank as hard gate when the audience
framing implies a maturity ceiling. Within the gated pool, KW
audience-framing tags + CTX `watch_scenarios` ("watch with kids,"
"family night") contribute additive inclusion scoring.
**Boundaries:** packaged-audience framing only. Story archetype
like coming-of-age → Cat 32. Content-sensitivity ("no gore") →
Cat 28. Concrete viewing situation → Cat 34.

### 28. Sensitive content
**Endpoints:** KW + META + VWX (gate + inclusion, query-dependent).
**Handles:** "no gore," "not too bloody," "with nudity," "violent
but not graphic."
**Mechanism:** META.maturity_rank as gate when a rating ceiling is
implied. KW content tags score binary presence/absence
(ANIMAL_DEATH-style flags); VWX `disturbance_profile` scores
intensity gradient for spectrum-framed asks.
**Boundaries:** content-on-its-own-spectrum framing. Audience-
pitch framing → Cat 27.

### 29. Seasonal / holiday
**Endpoints:** KW + CTX + P-EVT (additive combo).
**Handles:** Christmas, Halloween, Thanksgiving, summer-blockbuster.
**Mechanism:** KW via **proxy chains** — the vocabulary has no
dedicated seasonal tags, so the LLM rewrites seasonal intent into
proxy tags at query-generation time (Halloween → horror +
supernatural + spooky + slasher; Christmas → family + heartwarming
+ snowed-in + winter). CTX captures seasonal viewing framing
("Christmas viewing," "Halloween movie night"). P-EVT captures
seasonal narrative settings ("set on Christmas Eve," "Halloween
night"). Scores merge additively.
**Why additive, not tiered:** no channel is authoritative — proxy
tags are inherently approximate, and the semantic spaces catch
real signal the proxy chain misses, especially for less-
canonicalized holidays.

---

## Semantic-driven categories

### 30. Plot events
**Endpoints:** P-EVT (transcript-style query phrasing).
**Handles:** literal plot events ("a heist crew unravels when a
member betrays them," "a man wakes up with no memory and tries to
find his wife's killer").
**Mechanism:** P-EVT cosine; query phrased transcript-style to
mirror ingestion text format (per v3 #18).
**Boundaries:** event prose only. Narrative time/place setting →
Cat 31 (same vector space, different prompt template).

### 31. Narrative setting (time/place)
**Endpoints:** P-EVT (descriptive query phrasing).
**Handles:** narrative time setting ("set in 1940s Berlin"),
narrative place setting ("takes place in Tokyo," "on a remote
island").
**Mechanism:** same P-EVT vector space as Cat 30 with descriptive
query phrasing ("set in X," "takes place in Y").
**Boundaries:** split from Cat 30 because the prompt template
differs — same retrieval space, different query shape. Per the
derive-once principle, step 2 labels the trait so the handler runs
the right phrasing template without re-deriving.

### 32. Story / thematic archetype
**Endpoints:** KW + P-ANA (tiered, keyword-first).
**Handles:** "movies about grief," "redemption arcs," "man-vs-
nature," "underdog stories," "revenge stories," "post-apocalyptic,"
"coming-of-age about self-acceptance," "sisterly love stories,"
"friendship stories," "found-family stories."
**Mechanism:** ConceptTag (REDEMPTION, FOUND_FAMILY, REVENGE,
CORRUPTION) → P-ANA prose fallback across `elevator_pitch`,
`conflict_type`, `thematic_concepts`, `character_arcs`.
**Boundaries:** spectrum framings ("kind of about grief," "leans
redemptive") handled via salience=supporting downstream weighting,
not via a separate routing branch. Distinguished from Cat 8 by
abstraction level — thematic essence here, concrete subject
there. Distinguished from Cat 10 by static-vs-trajectory —
character archetype is a static type, story archetype is a story
shape.

### 33. Emotional / experiential
**Endpoints:** VWX + CTX + RCP + KW (additive combo, handler-driven
field selection).
**Handles:** all emotional / experiential / feel framings —
- During-viewing feel: tone, tonal aesthetic (dark, whimsical,
  gritty), cognitive demand (mindless vs cerebral),
  realism/stylization mode, tension/disturbance intensity,
  emotional palette.
- Pacing-as-experience: "slow burn," "frenetic."
- Self-experience goals: "make me cry," "cheer me up," "challenge
  me," "something mindless."
- Comfort-watch / gateway: "go-to movie," "feel-better movie,"
  "good first anime," "accessible arthouse."
- Post-viewing resonance: "stays with you," "haunting,"
  "gut-punch ending," "forgettable."
- Structural ending types: "happy ending," "twist ending,"
  "downer ending," "ambiguous ending."

**Mechanism:** handler-driven field selection across VWX (full
surface incl. `ending_aftertaste`, `emotional_palette`, `tension`,
`disturbance`, `sensory_load`, `cognitive_complexity`,
`tone_self_seriousness`), CTX (`self_experience_motivations`),
RCP (emotional reception prose: "tearjerker," "comfort-rewatch,"
"unforgettable"), KW (TEARJERKER, FEEL_GOOD, HAPPY_ENDING,
TWIST_ENDING, OPEN_ENDING, SAD_ENDING). Tag-perfect short-circuit
when an emotional or structural-ending tag maps cleanly to the
trait.
**Boundaries:** anything emotional or experiential — before,
during, or after watching — lives here. The merger is deliberate:
step 2 is worse at fine emotional disambiguation than the handler-
stage LLM that has the trait + its full context in front of it.
Concrete viewing situations ("date night," "rainy Sunday") →
Cat 34, the sole carve-out because situations are *named events*,
sharply distinguishable at step 2 from feelings.

### 34. Viewing occasion
**Endpoints:** CTX (`watch_scenarios`).
**Handles:** concrete named situations — "date night," "rainy
Sunday," "long flight," "with kids on Saturday," "background
watching," "family movie night," "Sunday morning."
**Boundaries:** split from Cat 33 because viewing occasions are
concrete *situations*, not feelings. The carve is sharp at step 2:
named-event surface form vs feeling-or-state surface form.

### 35. Visual craft acclaim
**Endpoints:** RCP + PRD (additive combo).
**Handles:** "visually stunning," "killer cinematography,"
"beautifully shot," "IMAX-shot," "practical effects," "technical
marvel."
**Mechanism:** RCP `praised_qualities` prose for the acclaim itself
+ PRD prose for the production-craft side ("shot on 70mm,"
"practical-effects-heavy").
**Boundaries:** named cinematographer / VFX supervisor → Cat 40
(reserved for now, returns empty).

### 36. Music / score acclaim
**Endpoints:** RCP.
**Handles:** "iconic score," "great soundtrack," "memorable
theme," "amazing music."
**Mechanism:** RCP only — music isn't carried meaningfully by PRD
or NRT.
**Boundaries:** named composer → Cat 1 (composer postings).

### 37. Dialogue craft acclaim
**Endpoints:** RCP + NRT (additive combo).
**Handles:** "quotable dialogue," "Sorkin-style," "naturalistic
dialogue," "snappy banter."
**Mechanism:** RCP `praised_qualities` for the acclaim + NRT for
dialogue-as-craft-pattern.
**Boundaries:** "Sorkin-style" can also fire Cat 26 if framed as a
structural pattern (rapid-fire walk-and-talk as a how-told device)
rather than as praise.

### 38. General appeal / quality baseline
**Endpoints:** META.reception_score + META.popularity_score
(numeric priors).
**Handles:** qualitative quality language without a specific
numeric threshold — "well-received," "highly rated," "popular,"
"best," "great," "highly regarded."
**Mechanism:** numeric column priors only — additive lift, not
hard threshold.
**Boundaries:** specific numeric thresholds → Cat 19. Specific
praise/criticism prose ("praised for tension," "criticized as
overhyped") → Cat 39. Quality superlatives ("best horror of the
80s") fire Cat 38 + Cat 39 + axis cats per the compound split rule.

### 39. Specific praise / criticism
**Endpoints:** RCP + KW (additive combo).
**Handles:** reception prose for what people specifically liked or
disliked, plus canonical reception tags. "Cult," "underrated,"
"overhyped," "divisive," "praised for its tension," "criticized as
plodding," "still holds up," "era-defining," "stacked cast,"
"thematic weight" (acclaim side).
**Mechanism:** RCP `praised_qualities` / `criticized_qualities`
prose + KW (CULT_CLASSIC, UNDERRATED, DIVISIVE).
**Boundaries:** quality-as-prose, not quality-as-numeric. The
numeric prior side is Cat 38. AWD records ("Oscar-winning") →
Cat 11 by compound split rule.

---

## Trick / specialized

### 40. Below-the-line creator lookup
**Endpoints:** Reserved (returns empty for now).
**Handles:** cinematographer, editor, production designer, costume
designer, VFX supervisor — "Roger Deakins movies," "Thelma
Schoonmaker-edited," "Sandy Powell costumes," "Colleen Atwood
designs."
**Status:** category reserved as a deliberate slot. Returns empty
until postings or a directed semantic-on-credits surface lands.
Routing here keeps these queries from scattering across wrong cats
in the meantime.
**Boundaries:** future mechanism likely RCP (reception prose names
these creators when noted) + dedicated postings if/when indexed.
Distinct from Cat 1 because Cat 1 is posting-table-backed and
would fail silently for non-indexed roles.

### 41. Named source creator
**Endpoints:** P-EVT + RCP (additive combo).
**Handles:** named creator of source material — "Stephen King" (in
"Stephen King novels"), "Tolkien" (in "Tolkien films"), "Shakespeare"
(in "based on Shakespeare plays"), "Philip K. Dick," "Neil Gaiman,"
"Jane Austen."
**Mechanism:** semantic search across P-EVT (synopsis prose names
the creator: "based on Stephen King's novel") + RCP (reviews cite
the creator). The named referent stays as one trait — never split
mid-name ("Stephen King" stays one trait).
**Boundaries:** **detection rule for named-referent routing in
"based on / by / X's <medium>" phrases:**
- Is the referent a *character-anchored film franchise* (the
  character's own films)? → Cat 6 (e.g. "Sherlock Holmes books"
  → Cat 6 + Cat 7).
- Is the referent a *character-less film franchise*? → Cat 5
  (e.g. "Star Wars novelizations" → Cat 5 + Cat 7).
- Is the referent a *creator of source material*? → Cat 41 (e.g.
  "Shakespeare plays" → Cat 41 + Cat 7, "Stephen King novels" →
  Cat 41 + Cat 7, "based on a comic book" → Cat 7 alone with no
  named referent).

This category fires only for named creators, never for franchise
referents. Step 2 always decomposes the source phrase: the named-
referent half routes per the rule above; the medium half ("books,"
"plays," "novels") routes to Cat 7. Cat 1 is film credits — source-
material creators aren't film credits — so Cat 41 is the only path
for these names.

### 42. "Like &lt;media&gt;" reference
**Endpoints:** PARAMETRIC (re-routes through dispatcher).
**Handles:** named-work comparison — "like Inception," "similar to
The Office," "movies that feel like David Lynch," "in the vein of
Hitchcock thrillers," "like a Coen Brothers movie."
**Mechanism:** handler extracts 4-6 distinctive traits of the
named referent (genre, narrative devices, tone, themes, era, etc.)
and re-emits them as new sub-traits routed through the normal
dispatcher. Doesn't search corpus directly.
**Boundaries:** triggers on explicit comparison surface forms only
— "like X," "similar to Y," "in the vein of Z," "X-style," "feels
like Q." Vague reference classes without a named comparison target
("comedians doing drama," "auteur directors") → Cat 44. The
distinction: Cat 42 expands a single named work; Cat 44 expands a
class.

---

## Ordinal selection

### 43. Chronological ordinal
**Endpoints:** META.release_date (sort-and-pick).
**Handles:** release-date ordinal position within a scoped
candidate set — "first," "last," "earliest," "latest," "most
recent," "the newest one," "the oldest one."
**Mechanism:** order the candidate pool (scoped by the rest of
the query's categories) by `movie_card.release_date` and select
the top-N position indicated by the phrasing.
**Boundaries:** ordinal selection only. Range or decay framings
("90s movies," "recent," "before 2000") → Cat 13. "Most recent"
is chronology; "best" / "most acclaimed" is reception superlative
(Cats 38/39). "The latest Scorsese" is Cat 43 + Cat 1.

---

## Catch-all

### 44. Generic parametric / catch-all
**Endpoints:** PARAMETRIC (multi-mechanism handler).
**Handles:** anything that needs interpretation/expansion and
doesn't fit a structured category. Specifically:
- **Vague reference classes:** "comedians doing drama," "auteur
  directors of the 70s," "directors known for long takes," "child
  actors who became serious."
- **Named lists / curated canon:** Criterion Collection, AFI Top
  100, IMDb Top 250, BFI, National Film Registry, Sight & Sound
  greatest, "1001 Movies to See Before You Die," film-school canon.
- **Anything else step 2 recognizes as real but underspecified.**

**Mechanism:** multi-mechanism — the handler chooses what fits the
trait. Can fan out to Cat 1 (entity lookup for expanded actor/
director instances), KW (tag-resolvable expansions of list
signatures), semantic across spaces, META priors (numeric quality
lift for canonical-list queries). Doesn't default to semantic-only.
**Boundaries:** distinct from Cat 42 — Cat 42 expands a *named
work*; Cat 44 expands a *reference class or named list*. Both are
parametric; the difference is what gets expanded and how. This is
the only true catch-all in the taxonomy; the goal is to keep
shrinking it as recognizable patterns get lifted into dedicated
cats.

---

## Orchestration shapes

Every category runs under one of four shapes, which govern how many
endpoint queries can fire for that category on a given user request.

### At most one query fires

**Single endpoint** — only one endpoint is ever applicable. No
routing decision to make.
Cats 1, 2, 3, 4, 5, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
24, 30, 31, 34, 36, 38, 43.

**Mutually exclusive** — two endpoints could each individually
answer the question, but they answer different versions of it. The
handler picks whichever matches the user's framing.
Cats 22 (canonical genre → KW; qualifier-laden → P-ANA),
23 (tradition tag → KW; no tag → META fallback).

**Tiered** — an ordered preference list of endpoints. The handler
fires the first genuine fit; later tiers are fallbacks for cases
the earlier tiers can't cleanly express.
Cats 8, 9, 10, 25, 26, 32.

### More than one query may fire

**Combo** — multiple endpoints apply and each carries distinct,
complementary signal that can't be collapsed into a single call.
All applicable endpoints fire in parallel.
Cats 6, 27, 28, 29, 33, 35, 37, 39, 41.

### Special

**Parametric / re-route** — the category transforms a trait into
new sub-traits or invokes multi-mechanism dispatch.
Cats 42 (re-routes named-work expansion), 44 (multi-mechanism
catch-all).

**Reserved / no-op** — category is reserved as a routing slot but
returns empty until backing data lands.
Cat 40.

---

## Where every data gap lands

| Gap | Category | How handled |
|---|---|---|
| Below-the-line credits | Cat 40 | Reserved slot, returns empty until data lands |
| Curated canon (Criterion, AFI) | Cat 44 | Parametric expansion to canonical list signatures |
| Source-material creator | Cat 41 | Semantic (P-EVT + RCP) for name occurrences |
| Reference-work comparison ("like X") | Cat 42 | Parametric expansion → re-route |
| Vague reference class ("comedians doing drama") | Cat 44 | Parametric expansion to instances → re-route |
| Character-anchored franchise (Batman, Bond) | Cat 6 | Combo: ENT character postings + FRA lineage |
| Gateway / entry-level | Cat 33 | Folded into emotional/experiential combo |
| Cultural tradition | Cat 23 | KW tag → META fallback |
| Seasonal | Cat 29 | KW proxy chains + CTX + P-EVT additive |
| Narrative setting time/place | Cat 31 | P-EVT descriptive phrasing |
| Cast popularity ("stacked cast") | Cat 39 | RCP prose |
| Thematic weight ("has something to say") | Cat 39 (acclaim) + Cat 32 (vibe) | Framing-dependent |
| Character-vs-plot focus | Cat 26 | NRT fallback tier |
| Character archetype | Cat 10 | KW tag → NRT fallback |
| Element / motif presence | Cat 9 | KW tag → P-EVT fallback |
| Self-experience goal ("make me cry") | Cat 33 | Combo additive |
| Comfort-watch / "feel-better movie" | Cat 33 | Combo additive |
| Post-viewing resonance ("haunting") | Cat 33 | Combo additive |
| Cultural influence / still-holds-up | Cat 39 | RCP prose |
| Live trending | Cat 12 | Dedicated endpoint |
| Chronological ordinal | Cat 43 | META.release_date sort-and-pick |
| Media type (TV-movie / short / video) | Cat 21 | Dedicated META single-attribute |

---

## Composition notes

Categories are composable atoms, not mutually exclusive buckets.
The dispatcher resolves each category independently and merges
scores under the v3 trait-grouping framework (see
`v3_trait_identification.md`).

Worked examples:
- "Tom Hanks comedies from the 90s rated above 8" → Cat 1 (Hanks)
  + Cat 22 (comedy) + Cat 13 (90s) + Cat 19 (rated above 8).
- "Best horror of the 80s" → Cat 38 (general appeal) + Cat 39
  (specific reception prose) + Cat 22 (horror) + Cat 13 (80s).
- "Stephen King novels from the 90s" → Cat 41 (Stephen King) +
  Cat 7 (novel adaptation) + Cat 13 (90s).
- "Movies like Inception with a slow burn" → Cat 42 (Inception
  expansion) + Cat 33 (slow-burn pacing).
- "Batman movies from the 80s" → Cat 6 (Batman character-
  franchise) + Cat 13 (80s). Note: one trait for Batman, not
  Cat 3 + Cat 5 — Cat 6 fans out to ENT + FRA internally.
- "Yoda appearances in the prequels" → Cat 3 (Yoda character) +
  Cat 5 (Star Wars franchise, prequel lineage). Two traits here
  because Yoda doesn't anchor his own film franchise — he appears
  in the Star Wars franchise, which is character-less per Cat 5's
  rule.

---

## Compound split rule

**If a phrase or query seems to fit multiple categories, that is a
signal it should be split into separate atomic traits.** Step 2 is
expected to decompose compound phrases into their constituent
category firings rather than inventing an umbrella category.

Compound descriptors never warrant their own category. The word
"classic" means older + canonical, and the correct handling is to
fire Cat 13 (release era) + Cat 39 (canonical/acclaimed)
simultaneously. Creating a "Canonical stature" category to hold
"classic" would just duplicate endpoints already covered while
hiding the compound from dispatch.

The single exception is dual-nature *referents* — a single name
that is inherently both a character and a franchise (Cat 6). That
isn't a compound phrase; it's one referent with two natures, and
gets one trait with one category that fans out internally.

Worked examples:
- "Classic Arnold Schwarzenegger action movies" → Cat 1
  (Schwarzenegger) + Cat 22 (action) + Cat 13 (older era) +
  Cat 39 (canonical stature).
- "Disney classics" → Cat 4 (Disney) + Cat 13 (older era) +
  Cat 39 (canonical stature).
- "Lone female protagonist" → Cat 10 (female-lead archetype) +
  Cat 26 (single-lead structural form).
- "Modern classic" → Cat 13 (recent era, narrower range) +
  Cat 39 (canonical stature).
- "Stephen King novels" → Cat 41 (Stephen King) + Cat 7 (novel
  adaptation).
- "Sherlock Holmes books" → Cat 6 (Sherlock Holmes character-
  franchise) + Cat 7 (book adaptation). Note: not Cat 5 + Cat 7
  — Sherlock is character-anchored, so Cat 6 absorbs both the
  character and franchise sides as one trait.
- "Star Wars novelizations" → Cat 5 (Star Wars franchise) +
  Cat 7 (novel adaptation). Star Wars is character-less, so it
  stays in Cat 5.
- "James Bond remakes" → Cat 6 (Bond character-franchise) +
  Cat 7 (remake source-flag).
- "Batman movies" → Cat 6 (single trait, fans out to ENT + FRA).
  Counter-example to old guidance that decomposed this into
  Cat 3 + Cat 5; under the 1:1 rule, dual-nature referents like
  Batman get exactly one trait.

The only time a compound stays bound to a single category is when
the category explicitly owns the compound — e.g. a named curated
list ("Criterion Collection") in Cat 44, which *is* the compound
of "canonical recognition + specific named list," or a dual-nature
referent in Cat 6.
