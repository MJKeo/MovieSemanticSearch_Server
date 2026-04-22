# Brand Registry — Planning Doc (superseded by retune)

> **Status:** This document captures the ORIGINAL research and MVP
> member cut. A 2026-04 retune replaced the selection principle and
> trimmed every umbrella brand's roster. For the CURRENT brand
> rosters consumed by ingestion and the stage-3 studio resolver, see
> [`schemas/production_brands.py`](../schemas/production_brands.py)
> — that file is the source of truth. The per-brand tables below
> are historical research preserved for traceability and surface-
> form references; any row not marked otherwise has been retained.

## What changed in the retune

**Old principle:** "catalog recall over label purity" — include every
corporate subsidiary a brand owned during its ownership window, to
maximize recall. If the parent ever owned it, it counted.

**New principle:** a **brand-identity test, not a corporate-ownership
test**. A label belongs in a brand's roster only if a casual viewer
typing "<brand> movies" would expect its films. The test:

- **KEEP:** labels the parent actively brand-promotes — Pixar
  ("Disney/Pixar"), Marvel Studios, Lucasfilm/Star Wars, Illumination
  (Universal-promoted), Screen Gems (Sony-genre), etc.
- **DROP:** autonomous-identity acquisitions deliberately kept
  separate — Miramax, Searchlight, Touchstone, Hollywood Pictures,
  Blue Sky under Disney; New Line, Castle Rock, HBO Films, DC under
  Warner Bros.; Focus Features, DreamWorks Animation, Working Title
  under Universal; Nickelodeon, MTV, Republic under Paramount; Sony
  Pictures Classics under Sony. These remain findable via their own
  standalone brands in the registry (MIRAMAX, SEARCHLIGHT, TOUCHSTONE,
  NEW_LINE_CINEMA, DC, FOCUS_FEATURES, DREAMWORKS_ANIMATION, etc.) —
  the retune removes cross-brand leaks, not discoverability.
- **DROP:** home-entertainment-only, distribution-only,
  foreign-region, pure legal-entity, and obscure joint-venture
  credits with no user-search value.

### Motivating bug

Under the old principle, searching "Disney movies" returned *No
Country for Old Men* because Miramax was Disney-owned 1993-2010 —
even though no casual viewer thinks of that film as a Disney film.
Retune drops Miramax from DISNEY; NCFOM now tags `MIRAMAX + PARAMOUNT`
(via Paramount Vantage), which is what a casual viewer expects.

## Validation against real IMDB data

Registry validated against 361k films in `tracker.db` `imdb_data`:

**Surface-string existence.** Every retained string has non-zero film
evidence. Ten agent-proposed ADDs from the initial retune pass had
zero evidence in live IMDB data and were removed: `Lucasfilm Ltd.`,
`Illumination` (2018 rename variant), `DreamWorks Animation SKG`,
`PDI/DreamWorks`, `A24 Films`, `A24 Films LLC`, `Neon Rated`,
`Studio Ghibli, Inc.`, `Miramax Films`. IMDB consistently uses the
base forms (`A24`, `Neon`, `Studio Ghibli`, `Miramax`, `Lucasfilm`,
`Illumination Entertainment`) — agent claims about "X is also a
common IMDB surface string" were speculative and not ground-truthed.

**Date-window validation.** Three real gating bugs fixed:

| Brand | String | Old window | New window | Why |
|---|---|---|---|---|
| SEARCHLIGHT | `Searchlight Pictures` | (2020, None) | (None, None) | IMDB retroactively applies the current "Searchlight Pictures" credit to pre-2020 Fox Searchlight films (Slumdog Millionaire 2008, Black Swan 2010, 12 Years a Slave 2013 were missing their tag) |
| WARNER_BROS | `Warner Bros. Cartoon Studios` | (1980, None) | (None, None) | String actually covers the 1933-1963 classic Looney Tunes unit, not the 1980 animation relaunch (512 classic shorts were missing their tag) |
| TWENTIETH_CENTURY | `20th Century Pictures` | (1935, 2020) | (1933, 2020) | Pre-merger 20th Century Pictures (1933-35) films were missing their tag |
| UNITED_ARTISTS | `United Artists` | (1919, 1981) + (2024, None) | (1919, None) single window | Split assumption was wrong — IMDB uses bare `United Artists` on MGM-era UA films too (Rocky IV 1985, Rain Man 1988, Valkyrie 2008 were all missing their tag) |

**Date windows verified correct despite looking narrow:**
SONY's 1989 Columbia acquisition gate; DISNEY's Pixar (2006) /
Marvel (2009) / Lucasfilm (2012) acquisition gates; Summit
Entertainment 2012 gate (pre-acquisition Summit was competitor);
Atomic Monster 2024 gate (pre-merger Conjuring/Saw correctly stay
under their own identity); AMAZON_MGM 2022/2024 gates (classic
MGM/UA correctly excluded from Amazon brand).

## Lessons

1. **Corporate-ownership and brand-identity are different things.**
   The old "catalog recall" rule conflated them; the new rule
   separates them cleanly. Cross-brand leaks disappeared.
2. **Agent research about IMDB surface-string conventions is
   speculation.** Claims like "IMDB also uses variant X" or "bare
   `United Artists` is only used in the classic era" were
   confidently stated and wrong. Ground-truth against `imdb_data`
   before adding/splitting surface strings.
3. **IMDB retroactively applies current names.** Studio renames
   (Fox Searchlight → Searchlight, Illumination Entertainment →
   Illumination, Walt Disney Feature Animation → Walt Disney
   Animation Studios) cause the current name to appear on
   pre-rename films. Date windows on the new name must include
   pre-rename years unless there's specific evidence otherwise.

## How to read the historical tables below

The per-brand Members and Surface forms tables below reflect the
**original pre-retune research** (48 parallel research sub-agents,
Wikidata date cross-validation, IMDB `production_companies` TSV
evidence checks). They are preserved for their surface-form counts
and the per-member rationales. For authoritative current rosters,
read `schemas/production_brands.py`. Rows in the Members tables
below that have been DROPPED in the retune are annotated in the
brand's "Retune status" note directly under its header; all other
rows are still in the registry (potentially with an updated window
per the validation fixes above).

---

# Tier 1 — MVP must-haves (24 brands)

## DISNEY — "The Walt Disney Studios"

**Retune status:** Heavily trimmed. DROPPED from the umbrella: all 20th Century Fox / Studios variants, Searchlight Pictures, Fox Searchlight, Touchstone Pictures, Hollywood Pictures, Miramax + Dimension Films, Blue Sky Studios, and all home-entertainment / foreign-region / malformed co-credit surfaces. KEPT: Walt Disney Pictures + Animation canon, Pixar (co-branded), Marvel Studios, Lucasfilm, Disneytoon. Dropped labels remain findable via their own standalone brands.

| Member | Start | End | Rationale |
|---|---|---|---|
| Walt Disney Pictures | 1983 | — | Flagship live-action label, successor to Walt Disney Productions |
| Walt Disney Productions | 1929 | 1986 | Original corporate banner on pre-rename films |
| Walt Disney Animation Studios | 1923 | — | In-house animation, formerly Walt Disney Feature Animation |
| Pixar Animation Studios | 2006 | — | Acquired Jan 2006 |
| Marvel Studios | 2009 | — | Acquired via Marvel Entertainment |
| Lucasfilm Ltd. | 2012 | — | Acquired Oct 2012 |
| 20th Century Studios | 2019 | — | Via 21st Century Fox deal; renamed Jan 2020 |
| 20th Century Fox | 2019 | 2020 | Brief pre-rename Disney-era window |
| Searchlight Pictures | 2019 | — | Acquired via Fox; renamed from Fox Searchlight 2020 |
| Fox Searchlight Pictures | 2019 | 2020 | Pre-rename Searchlight name under Disney |
| Touchstone Pictures | 1984 | 2018 | Mature-audience label; dormant after 2018 |
| Hollywood Pictures | 1989 | 2007 | Adult-oriented sister of Touchstone |
| Miramax Films | 1993 | 2010 | Weinstein-era Disney ownership |
| Dimension Films | 1993 | 2005 | Miramax genre imprint during Disney ownership |
| DisneyToon Studios | 1988 | 2018 | DTV/theatrical animation sub-label; founded 1988 as Disney MovieToons, renamed 2003, closed 2018 |
| Blue Sky Studios | 2019 | 2021 | Fox animation studio; closed by Disney Apr 2021 |

**Rename history:** Walt Disney Productions (1929) → The Walt Disney Company (1986); Walt Disney Feature Animation → Walt Disney Animation Studios (2007); 20th Century Fox → 20th Century Studios (Jan 2020); Fox Searchlight → Searchlight Pictures (2020).

**Notes:** Excluded Disneynature (<5 notable theatricals), ESPN Films, ABC-branded film units, Jerry Bruckheimer Films (external), ILM and Skywalker Sound (Lucasfilm-internal, covered by LUCASFILM). Marvel Entertainment is a parent holding of Marvel Studios and excluded here.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| Walt Disney Pictures | `Walt Disney Pictures` | 437 |
|  | `Walt Disney Studios` | 25 |
|  | `Walt Disney Studios Motion Pictures` | 14 |
|  | `Walt Disney Home Video` | 14 |
|  | `Walt Disney Home Entertainment` | 12 |
|  | `Walt Disney British Films` | 4 |
|  | `Walt Disney Studios Home Entertainment` | 4 |
|  | `Walt Disney Pictures / Sony Pictures` | 1 |
| Walt Disney Productions | `Walt Disney Productions` | 1021 |
| Walt Disney Animation Studios | `Walt Disney Animation Studios` | 173 |
|  | `Walt Disney Feature Animation` | 37 |
|  | `Walt Disney Animation Australia` | 15 |
|  | `Walt Disney Animation Japan` | 13 |
|  | `Walt Disney Feature Animation Florida` | 7 |
|  | `Walt Disney Animation Canada` | 6 |
|  | `Walt Disney Animation France S.A.` | 3 |
|  | `Walt Disney Feature Animation Paris` | 1 |
| Pixar Animation Studios | `Pixar Animation Studios` | 120 |
|  | `Pixar` | 1 |
| Marvel Studios | `Marvel Studios` | 82 |
| Lucasfilm Ltd. | `Lucasfilm` | 86 |
|  | `Lucasfilm Animation` | 7 |
| 20th Century Studios | `20th Century Studios` | 56 |
|  | `20th Century Fox Home Entertainment` | 41 |
|  | `20th Century Fox Argentina` | 6 |
|  | `20th Century Fox Korea` | 1 |
|  | `20th Century Fox Post Production Services` | 1 |
| 20th Century Fox | `Twentieth Century Fox` | 1358 |
|  | `Twentieth Century-Fox Productions` | 42 |
|  | `Twentieth Century Fox Animation` | 37 |
|  | `20th Century Pictures` | 24 |
|  | `20th Century Fox` | 19 |
|  | `Twentieth Century Animation` | 11 |
|  | `Twentieth Century Productions` | 1 |
|  | `Twentieth Century-Fox Studios, Hollywood` | 1 |
| Searchlight Pictures | `Searchlight Pictures` | 50 |
| Fox Searchlight Pictures | `Fox Searchlight Pictures` | 104 |
| Touchstone Pictures | `Touchstone Pictures` | 230 |
|  | `Touchstone Films` | 7 |
|  | `Touchstone` | 1 |
|  | `Touchstone Pictures México` | 1 |
| Hollywood Pictures | `Hollywood Pictures` | 85 |
|  | `Hollywood Pictures Corporation (I)` | 5 |
|  | `Hollywood Pictures Corporation (II)` | 3 |
|  | `Hollywood Pictures Home Video` | 1 |
| Miramax Films | `Miramax` | 266 |
|  | `Miramax Family Films` | 2 |
|  | `Miramax International` | 2 |
|  | `Miramax Home Entertainment` | 1 |
| Dimension Films | `Dimension Films` | 118 |
|  | `Dimension Films (II)` | 3 |
| DisneyToon Studios | `Disneytoon Studios` | 54 |
| Blue Sky Studios | `Blue Sky Studios` | 29 |

**Collisions / exclusions flagged:**

- `Disney Channel` — TV channel — excluded per scope (theatrical film brands only)
- `Disney Television Animation` — TV animation arm, not theatrical
- `Walt Disney Television` — TV arm, not theatrical
- `Disneynature` — Explicitly excluded in tier doc notes (<5 notable theatricals)
- `Disney Branded Television` — TV division
- `Disney+` — Streaming platform, handled by watch_providers path not studio resolver
- `Disney Junior` — Children's TV channel
- `Disney XD` — TV channel
- `Toon Disney` — TV channel
- `Playhouse Disney` — TV programming block
- `Disney-ABC Domestic Television` — TV distribution; ABC-branded excluded per tier doc
- `Disney-ABC Television Group` — TV group; ABC-branded excluded
- `Disney Television Studios` — TV production
- `Disney Educational Productions` — Non-theatrical educational content
- `Disney Enterprises` — IP-holding entity, not a production label
- `The Walt Disney Company` — Parent holding company, not a production label
- `Walt Disney Company` — Parent holding company
- `Walt Disney Imagineering (WDI)` — Theme parks R&D, not film production
- `Walt Disney Attractions` — Theme parks, not film
- `Walt Disney World` — Theme park
- `Walt Disney Records` — Music label
- `Walt Disney Music Company` — Music publisher
- `Walt Disney Family Foundation` — Charity
- `The Walt Disney Family Museum` — Museum
- `The Walt Disney Studio Archives` — Archives, not production
- `Walt Disney Educational Media Company` — Educational non-theatrical
- `Walt Disney Industrial Training Film` — Industrial training films, not theatrical
- `Walt Disney Telecommunications and Non-Theatrical Company` — Non-theatrical by definition
- `Walt Disney Television Italia` — TV
- `Disney Television France` — TV
- `Disney Channel Italy` — TV
- `Disney Channel Latinoamérica` — TV
- `Disney HSM China Productions` — Disney Channel High School Musical China — TV
- `Disney Interactive` — Video games / digital, not film
- `Disney Interactive Japan` — Video games
- `Disney Interactive Labs` — Video games
- `Disney Interactive Studios` — Video games
- `Disney Latino` — Regional marketing unit, not production
- `Disney Online Originals` — Online content, not theatrical
- `Disney Publishing Worldwide` — Book publishing
- `Disney Research` — R&D lab, not production
- `Disney Media Distribution (DMD)` — Distribution entity, not a production label
- `Disney Broadcast Production` — Broadcast TV
- `Disney Concerts` — Concerts/music
- `Disney Theatrical Group` — Live stage productions
- `Disney Theatrical Productions (DTP)` — Live stage productions (Broadway)
- `Disney Destinations` — Travel/theme parks
- `Disney Original Documentary` — Disney+ documentary unit (streaming)
- `Walt Disney Company Italia` — Regional corporate unit, not production
- `Walt Disney Company Nordic` — Regional corporate unit
- `Walt Disney Company Russia & CIS` — Regional corporate unit
- `Roy E. Disney Productions` — Separate personal production company of Roy E. Disney, not Walt Disney Studios
- `Marvel Entertainment` — Parent holding of Marvel Studios — explicitly excluded per tier doc notes
- `Marvel Productions` — Pre-2009 Marvel TV animation arm, not the Marvel Studios theatrical label
- `Marvel Animation` — TV animation, not Marvel Studios theatrical
- `Marvel Enterprises` — Pre-2005 Marvel parent entity, not the Marvel Studios theatrical label
- `Marvel Entertainment Group` — Pre-1998 Marvel parent, not Marvel Studios
- `Marvel Knights` — Comics imprint / direct-to-video animated line, not Marvel Studios theatrical
- `Marvel Television` — TV arm
- `Marvel Films` — Pre-Marvel-Studios (1990s) film unit — predecessor but ambiguous; excluded to be conservative (Marvel Studios is the named member, tier doc includes only post-2009 acquisition)
- `Marvel Comics` — Comics publisher
- `Marvel Comics Group` — Comics publisher
- `Marvel Characters` — IP licensing entity
- `Marvel Movies` — Ambiguous small entity (count 2); not the Marvel Studios label
- `Marvel New Media` — Digital/online, not Marvel Studios theatrical
- `Lucasfilm Television` — TV arm, separate from theatrical Lucasfilm Ltd. member
- `20th Century Fox Television` — TV arm; tier doc TWENTIETH_CENTURY scope is the film label
- `Fox Television Studios` — TV
- `Fox 2000 Pictures` — Former Fox label — tier doc does not list it as a DISNEY member; note tier doc says '20th Century Studios' is the sole 20CF member surface
- `Fox STAR Studios` — Fox Star Studios India — explicitly excluded in tier doc notes
- `Fox Star Studios` — India label — explicitly excluded in tier doc notes
- `Fox International Productions (FIP)` — International label not listed as a DISNEY member in tier doc
- `Fox International Productions` — International label not listed as a DISNEY member
- `Fox Atomic` — Defunct genre label not listed in tier doc
- `Fox Animation Studios` — Earlier Fox animation unit; not same as Twentieth Century Fox Animation / 20th Century Studios
- `Fox Film Corporation` — 1915–1935 predecessor that merged with Twentieth Century Pictures to form 20th Century-Fox; pre-merger standalone entity, not a rename in tier doc scope
- `Fox Film Company` — Pre-merger predecessor variant; out of scope
- `Twentieth Century Vixen` — Unrelated adult-film entity sharing the 'Twentieth Century' token
- `20th Century Vixen` — Unrelated adult-film entity
- `20th Century Foss` — Unrelated
- `Fox West Pictures` — Unrelated Fox-prefixed label
- `Foxstar Productions` — Unrelated (not Fox Star Studios; despite name similarity, different entity)
- `Blue Sky Films` — Count 39 is inconsistent with Blue Sky Studios theatrical output (~13 films); name-collision with generic 'Blue Sky' international companies is very likely. Excluded to be conservative
- `Off Hollywood Pictures` — Unrelated indie; shares 'Hollywood Pictures' token
- `North Hollywood Pictures` — Unrelated
- `Mukta Searchlight Films` — Unrelated Indian company
- `Searchlight Films` — Predates Fox Searchlight; unrelated smaller label
- `Searchlight Productions` — Unrelated name-collision
- `Fourth Dimension Films` — Unrelated
- `Other Dimension Films` — Unrelated
- `3-Dimension Films` — Unrelated
- `Epic Dimension Films` — Unrelated
- `Marvelous Productions` — Unrelated — 'Marvelous' is a distinct English word, not Marvel
- `Marvelous Entertainment` — Unrelated (Japanese Marvelous Inc.)

**Surface-form notes:** Dominant surfaces: 'Twentieth Century Fox' (1358) dwarfs all others, reflecting that TCF is Disney's most-represented catalog after 2019 acquisition. 'Walt Disney Productions' (1021) is the pre-1986 historical name and is well-represented. Regional/divisional variants (Animation Japan/Canada/Australia/France, Feature Animation Florida/Paris) were included as they are clearly branches of Walt Disney Animation Studios. Marvel scope is intentionally narrow: only 'Marvel Studios' is included because the tier doc names Marvel Studios as the sole member and explicitly excludes Marvel Entertainment and the other pre-2009 Marvel film/TV units. Lucasfilm Animation is included (tier doc carves out ILM / Skywalker Sound as covered-by-LUCASFILM; Lucasfilm Animation isn't a theatrical-excluded unit). Fox sub-labels like Fox 2000 Pictures, Fox Atomic, Fox STAR Studios, Fox International Productions are flagged as collisions because the tier doc lists only '20th Century Studios' / '20th Century Fox' and explicitly excludes Fox Star Studios and Fox Searchlight (separate brand). 'Blue Sky Films' count 39 looked inflated relative to Blue Sky Studios' ~13 theatrical films so excluded as a likely name-collision; 'Blue Sky Studios' (29) is the canonical surface. 'Marvel Films' (count 1) is a 1990s pre-acquisition Marvel film unit — flagged but excluded; easy to re-add if downstream evaluation wants pre-MCU Marvel-branded films swept in. Potential coverage gap: no 'Miramax Films' surface exists — the canonical rename to 'Miramax' covers 266 tags which is expected given the 1990s rebrand.

## WALT_DISNEY_ANIMATION — "Walt Disney Animation Studios"

**Retune status:** Trimmed to two headline credits (`Walt Disney Animation Studios`, `Walt Disney Feature Animation`). DROPPED: satellite/facility units (Australia, Japan, Canada, France, Florida, Paris) and `Walt Disney Productions` (pre-1986 mixed animation + live-action output would dilute the animation-only brand).

| Member | Start | End | Rationale |
|---|---|---|---|
| Walt Disney Animation Studios | 2007 | — | Current name from Meet the Robinsons onward |
| Walt Disney Feature Animation | 1986 | 2007 | Prior name (Oliver & Company through Meet the Robinsons era) |
| Walt Disney Productions | 1929 | 1986 | Classic-era credit for in-house animated features pre-Feature Animation label |

**Rename history:** Founded 1923 as Disney Brothers Cartoon Studio → Walt Disney Studio (1926) → Walt Disney Productions (1929) → Walt Disney Feature Animation (1986) → Walt Disney Animation Studios (2007).

**Notes:** Scope limited to the in-house theatrical animation studio lineage. Excludes Pixar, Walt Disney Television Animation, DisneyToon Studios.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| Walt Disney Animation Studios | `Walt Disney Animation Studios` | 173 |
|  | `Walt Disney Animation Australia` | 15 |
|  | `Walt Disney Animation Japan` | 13 |
|  | `Walt Disney Animation Canada` | 6 |
|  | `Walt Disney Animation France S.A.` | 3 |
| Walt Disney Feature Animation | `Walt Disney Feature Animation` | 37 |
|  | `Walt Disney Feature Animation Florida` | 7 |
|  | `Walt Disney Feature Animation Paris` | 1 |
| Walt Disney Productions | `Walt Disney Productions` | 1021 |

**Collisions / exclusions flagged:**

- `Walt Disney Productions` — Pre-1986 parent-company credit covers all Disney output (live-action features, TV, shorts, merchandising) not only in-house animation. Including it in WALT_DISNEY_ANIMATION per the tier doc rationale (in-house animation lineage) means every Walt Disney Productions-era live-action film (Mary Poppins, Herbie, Swiss Family Robinson, etc.) also matches this brand. Downstream routing should treat this membership as overlapping with DISNEY and prefer era/genre disambiguation.

**Surface-form notes:** Regional satellite studios (Australia, Japan, Canada, France for WDAS; Florida, Paris for Feature Animation) are included as members of the in-house theatrical animation lineage -- they produced sequences/features under the parent studio's creative direction. Rejected per instructions: Walt Disney Television Animation (no matches in TSV anyway), Disneytoon Studios (count 54, belongs to DISNEY as separate DTV/sequel label), Disney Channel Animation (not present). Also excluded Walt Disney Studios Motion Pictures (distribution label, not the animation studio).

## PIXAR — "Pixar Animation Studios"

**Retune status:** No change. Single-identity studio.

| Member | Start | End | Rationale |
|---|---|---|---|
| Pixar Animation Studios | 1986 | — | Core studio, spun out from Lucasfilm's Graphics Group |

**Rename history:** Lucasfilm Computer Graphics Division / Graphics Group → Pixar (1986 Jobs acquisition).

**Notes:** Pixar is also a member of DISNEY from 2006. Pre-1986 predecessor names released no theatrical features and are excluded. Pixar Canada (2010–2013 Vancouver satellite) dropped from member list: zero DB surface-string evidence, shorts-only output.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| Pixar Animation Studios | `Pixar Animation Studios` | 120 |
|  | `Pixar` | 1 |

**Surface-form notes:** TSV case-insensitive grep for 'pixar' returned exactly two distinct strings: 'Pixar Animation Studios' (120 occurrences) and bare 'Pixar' (1 occurrence). Both map to the core Pixar Animation Studios entity. No 'Pixar Canada' surface form appears in the IMDB production_companies column -- member dropped. Pre-1986 predecessor names (Lucasfilm Computer Graphics Division / Graphics Group) are excluded per the tier list and do not appear under a Pixar surface form. No collisions: 'Pixar' token is unique to this brand.

## MARVEL_STUDIOS — "Marvel Studios"

**Retune status:** DROPPED `Marvel Films` (1993-1996 predecessor); its pre-MCU output (unreleased Fantastic Four, late-80s TV-movie orbit) isn't what casual viewers mean by "Marvel Studios."

| Member | Start | End | Rationale |
|---|---|---|---|
| Marvel Studios | 1996 | — | Primary MCU production credit |
| Marvel Films | 1993 | 1996 | Predecessor name on pre-MCU 1990s Marvel-produced films |

**Rename history:** Marvel Films (1993-1996) → Marvel Studios.

**Notes:** Marvel Enterprises / Marvel Entertainment are corporate parents, not on-screen production credits. Pre-2008 licensee productions (Sony Spider-Man, Fox X-Men) are NOT Marvel Studios catalog.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| Marvel Studios | `Marvel Studios` | 82 |
| Marvel Films | `Marvel Films` | 1 |

**Collisions / exclusions flagged:**

- `Marvel Entertainment`
- `Marvel Enterprises`
- `Marvel Entertainment Group`
- `Marvel Comics`
- `Marvel Comics Group`
- `Marvel Characters`
- `Marvel Knights`
- `Marvel Animation`
- `Marvel Television`
- `Marvel Productions`
- `Marvel New Media`
- `Marvel Worth Productions`
- `Marvel Movies`
- `Marvels Film Production`
- `Land Marvel Animation Studios`

**Surface-form notes:** Included only the two members listed in the tier file: 'Marvel Studios' (1996-, primary MCU credit, 82 rows) and 'Marvel Films' (1993-1996 predecessor, 1 row). No regional variants of 'Marvel Studios' appear in the TSV. Rejected per tier rules: corporate parents 'Marvel Entertainment' (57 rows), 'Marvel Enterprises' (12 rows), 'Marvel Entertainment Group' (7 rows); publisher 'Marvel Comics' / 'Marvel Comics Group' / 'Marvel Characters'; TV/animation arms 'Marvel Animation', 'Marvel Television', 'Marvel Knights', 'Marvel New Media'; 80s licensee 'Marvel Productions'; unrelated 'Marvel Worth Productions', 'Marvels Film Production', 'Marvel Movies', 'Land Marvel Animation Studios'. 'Marvel Entertainment' is a very common co-credit on MCU films but is explicitly excluded from this brand per tier spec. Unrelated 'Marvelous*' entries (Marvelous Productions, Marvelous Entertainment, etc.) share no lineage and are not flagged as collisions.

## LUCASFILM — "Lucasfilm"

**Retune status:** ADDED `Lucasfilm Animation` (2003-) for Clone Wars and Star Wars animated output. Proposed ADD `Lucasfilm Ltd.` was validated against IMDB and removed — zero evidence in live data (IMDB consistently uses bare `Lucasfilm`).

| Member | Start | End | Rationale |
|---|---|---|---|
| Lucasfilm Ltd. | 1971 | — | Primary production credit for Star Wars and Indiana Jones |
| Lucasfilm | 1971 | — | Common shorter credit variant on IMDB |

**Notes:** ILM and Skywalker Sound typically appear as VFX/sound credits, not production. Lucasfilm Animation is TV-only, excluded.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| Lucasfilm | `Lucasfilm` | 86 |

**Collisions / exclusions flagged:**

- `Lucasfilm Animation` — Excluded per tier list: TV-only division. Not attached to LUCASFILM brand.
- `Lucasfilm Television` — Excluded per tier list: TV division, not theatrical production credit.
- `LucasArts Entertainment Company` — Separate Lucas-family entity (video games publisher). Not a Lucasfilm production-company member; excluded to avoid cross-brand contamination.
- `Lucas Art Film` — Unrelated entity despite superficial token overlap ('Lucas' + 'Art'+'Film'); not a Lucasfilm sub-brand. Excluded.
- `Thomas Lucas Productions` — Unrelated namesake company. Excluded.
- `Otis Lucas Films` — Unrelated namesake. Excluded.
- `B. Lucas Studios` — Unrelated namesake. Excluded.
- `Lucas Camps Producties` — Unrelated (Dutch-language producer named after a person). Excluded.
- `Lucas Thorley Studios` — Unrelated namesake. Excluded.
- `Lucas Blank` — Unrelated personal/namesake entity. Excluded.
- `Lucas Dülligen Produktion` — Unrelated namesake. Excluded.
- `Lucas Ediciones S.A.` — Unrelated namesake. Excluded.
- `Lucas Martell` — Unrelated individual/namesake credit. Excluded.
- `Lucasida Entertainment` — Unrelated (different name, not a Lucasfilm variant). Excluded.
- `Lucas A. Ferrara` — Individual person credit, not a company. Excluded.

**Surface-form notes:** Case-insensitive grep over /tmp/company_strings.tsv returned three 'Lucasfilm'-prefixed surface strings: 'Lucasfilm' (86), 'Lucasfilm Animation' (7), 'Lucasfilm Television' (1). Per the tier list, Lucasfilm Animation (TV-only) and Lucasfilm Television (TV) are excluded, so only 'Lucasfilm' is retained. Notably, neither 'Lucasfilm Ltd.' nor 'Lucasfilm Ltd' appears in the TSV — the canonical long-form credit collapses to 'Lucasfilm' in this index. ILM and Skywalker Sound are excluded per the tier list (VFX/sound credits, not production credits). All other Lucas-token matches are unrelated namesakes/individuals and are listed as collisions_flagged for auditability.

## WARNER_BROS — "Warner Bros."

**Retune status:** Heavily trimmed. DROPPED: all New Line Cinema surfaces (autonomous brand — LOTR, Austin Powers, Rush Hour), Fine Line Features, Castle Rock Entertainment (Shawshank, When Harry Met Sally — own brand identity), all Turner Pictures variants, entire HBO lineage (Home Box Office, HBO Films, HBO Max, HBO Premiere Films, HBO Pictures, HBO Documentary Films — HBO is its own household brand), all DC entries (DC Comics, DC Entertainment, DC Films, DC Studios — now standalone DC brand). KEPT: core Warner Bros. live-action + animation. Window fix: `Warner Bros. Cartoon Studios` widened from (1980, None) to (None, None) to cover the 1933-1963 classic Looney Tunes unit.

| Member | Start | End | Rationale |
|---|---|---|---|
| Warner Bros. Pictures | 1923 | — | Flagship film label |
| Warner Bros. Animation | 1980 | — | In-house animation producing theatrical features |
| New Line Cinema | 1996 | — | Entered the Warner umbrella via the 1996 Time Warner-Turner merger; folded into WB ops 2008 |
| Castle Rock Entertainment | 1996 | — | Entered the Warner umbrella via the 1996 Time Warner-Turner merger |
| Turner Pictures | 1996 | 1998 | Absorbed into WB after Time Warner-Turner merger |
| HBO Films | 1999 | — | HBO's post-1999 theatrical/prestige film credit |
| HBO Pictures | 1989 | 1999 | Pre-rename HBO film credit |
| DC Entertainment | 2009 | 2022 | DC production credit through the 2022 restructure |
| DC Films | 2016 | 2022 | Dedicated DC film division 2016-2022 |
| DC Studios | 2022 | — | Gunn/Safran successor to DC Films |

**Rename history:** Time Warner (1990-2018) → WarnerMedia under AT&T (2018-2022) → Warner Bros. Discovery (2022). New Line folded into WB ops 2008.

**Notes:** Excluded Village Roadshow and Legendary (co-financiers, not WB-owned), TNT/TBS/Cartoon Network (TV-only), Warner Music.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| Warner Bros. Pictures | `Warner Bros.` | 2670 |
|  | `Warner Bros. Entertainment` | 55 |
|  | `Warner Brothers-First National Productions` | 49 |
|  | `Warner Bros./Seven Arts` | 40 |
|  | `Warner Bros. Pictures` | 28 |
|  | `Warner Bros. First National` | 3 |
|  | `Warner Bros Entertainment` | 1 |
|  | `Warner Bros. Productions` | 1 |
|  | `Warner Bros.-First National Pictures` | 1 |
|  | `Warner Brothers Entertainment` | 1 |
|  | `Warner Brothers Pictures` | 1 |
| Warner Bros. Animation | `Warner Bros. Cartoon Studios` | 512 |
|  | `Warner Bros. Animation` | 314 |
|  | `Warner Bros. Pictures Animation` | 19 |
|  | `Warner Bros. Feature Animation` | 5 |
|  | `Warner Classic Animation` | 2 |
|  | `Warner Brothers/Seven Arts Animation` | 2 |
|  | `Warner Bros. New York Animation` | 1 |
| New Line Cinema | `New Line Cinema` | 366 |
|  | `Fine Line Features` | 33 |
|  | `New Line Productions` | 7 |
|  | `New Line Film` | 1 |
|  | `New Line Film Productions` | 1 |
| Castle Rock Entertainment | `Castle Rock Entertainment` | 103 |
| Turner Pictures | `Turner Pictures (I)` | 65 |
|  | `Turner Pictures Worldwide` | 3 |
|  | `Ted Turner Pictures` | 1 |
|  | `Turner Pictures (III)` | 1 |
| HBO Films | `Home Box Office (HBO)` | 341 |
|  | `HBO Documentary Films` | 214 |
|  | `HBO Films` | 143 |
|  | `HBO Max` | 33 |
|  | `HBO Premiere Films` | 21 |
| HBO Pictures | `HBO Pictures` | 49 |
| DC Entertainment | `DC Entertainment` | 134 |
| DC Films | `DC Films` | 19 |
| DC Studios | `DC Studios` | 3 |
| DC Comics (pre-2009 publisher credit) | `DC Comics` | 25 |

**Collisions / exclusions flagged:**

- `HBO Max`
- `DC Comics`
- `Home Box Office (HBO)`
- `DC Studios`
- `Ted Turner Pictures`

**Surface-form notes:** Rule applications: Excluded all TV divisions (Warner Bros. Television, Warner Bros. Television Animation, Warner Bros. Television Productions UK, Warner Bros. International Television et al., New Line Television, HBO Sports, HBO Original Programming, HBO NYC Productions, HBO Family, HBO Studio Productions, HBO Downtown Productions, HBO Independent Productions, HBO Latin America, HBO/Cinemax Documentary, Turner Broadcasting System, TNT, Turner Original Productions, Turner Feature Animation, Turner Television, Turner Network Television, etc.), home video entities (Warner Home Video, Warner Bros. Home Entertainment, Warner Premiere, Warner Reprise Video, WarnerVision Films, HBO Home Entertainment, New Line Home Entertainment, New Line Home Video, Turner Home Entertainment, Warner Vision, Warner Music Vision), Warner Music and all subsidiaries (Warner Music Group, Warner Records, Warner Chappell, Warner Music Japan/Spain/Mexico/Nashville), corporate parents (Warner Bros. Discovery, WarnerMedia, Time Warner, Warner Communications, TimeWarner, Time Warner Entertainment Company, Time Warner Cable), foreign distribution arms (Warner Bros. Pictures Germany/Mexico, Warner Bros. Film Productions Germany, Warner Bros. Polska, Warner Bros. Korea, Warner China Film HG Corporation, Warner Española, Warner Entertainment Japan, Warner Bros. ITVP Deutschland), production-services facilities (Warner Bros. Studio Facilities, Warner Bros. Studios Leavesden, Warner Bros. De Lane Lea, Warner Bros. Post Production CreativeServices), unrelated-common-initials entries ('DC Dogs', 'DC Flav Films', 'DC Cinema', 'DC Pictures', 'DC Productions', 'DC Republic Productions', 'DC Stories', 'DC Media', 'DC Medias', 'DC Digital Content Comics', 'DC Creative' — none are associated with DC Comics/Entertainment), Warner Independent Pictures (WIP — 2003-2008 specialty division, not in tier-list member set), Picturehouse (explicitly excluded in tier doc as primarily distribution), and per tier doc: Village Roadshow/Legendary/TNT/Warner Music excluded. Warner Animation Group (tier-list member) has no surface form in the TSV — the label exists on IMDB typically as 'Warner Animation Group' but did not appear in our production_companies distinct-string index (likely films credit WB Pictures instead). HBO Max Films and Warner Max had no clear theatrical-film production string (Warner Max=4 is likely WB's HBO-Max-era film label, not in tier list member enumeration — excluded pending validation). DC Vertigo (count=1) excluded as comics-imprint only, not film production. Warner Bros./Seven Arts was WB's 1967-1969 production-era name — included under Warner Bros. Pictures. Warner Bros.-First National combinations reflect the 1929 First National acquisition era Warner production credit — included under Warner Bros. Pictures. HBO Premiere Films (21 occurrences) is the 1980s-era cable film credit — included under HBO Films as period-equivalent alongside HBO Pictures. Home Box Office (HBO) (341) is noisy but essential given how IMDB aggregates HBO theatrical film credits under this parenthesized corporate form — flagged for potential over-capture.

## NEW_LINE_CINEMA — "New Line Cinema"

**Retune status:** DROPPED the two Heron joint-venture shells (obscure coproduction SPVs casual viewers have never heard of) and `Fine Line Features` (arthouse imprint with its own identity). KEPT: four core New Line naming variants.

| Member | Start | End | Rationale |
|---|---|---|---|
| New Line Cinema | 1967 | — | Core banner; continues as a WB label after 2008 absorption |
| Fine Line Features | 1990 | 2005 | Arthouse imprint with substantial indie catalog |

**Notes:** Also member of WARNER_BROS since 1996. Picturehouse (2005-2008 WB/New Line JV) excluded as primarily distribution.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| New Line Cinema | `New Line Cinema` | 366 |
|  | `New Line Productions` | 7 |
|  | `New Line Film` | 1 |
|  | `New Line Film Productions` | 1 |
|  | `The New Line-Heron Joint Venture` | 1 |
|  | `The Fourth New Line-Heron Joint Venture` | 1 |
| Fine Line Features | `Fine Line Features` | 33 |

**Collisions / exclusions flagged:**

- `New Line Television` — TV division; tier list members cover film (New Line Cinema, Fine Line Features) only. Excluded.
- `New Line Home Entertainment` — Home video distribution label, not a production credit. Excluded.
- `New Line Home Video` — Home video distribution label, not a production credit. Excluded.
- `New Line International` — International distribution arm, not production. Excluded.
- `Red New Line` — Name collision — unrelated company (substring match only). Excluded.
- `Fine Line Media` — Ambiguous — likely unrelated small company, not the New Line arthouse imprint Fine Line Features. Excluded to avoid false positives.
- `Fine Line Productions` — Ambiguous — generic name, no evidence this is the New Line imprint (which was branded Fine Line Features). Excluded to avoid false positives.
- `The New Line-Heron Joint Venture` — INCLUDED. Genuine New Line co-production vehicle with Heron Communications from late-1980s titles (e.g., A Nightmare on Elm Street sequels).
- `The Fourth New Line-Heron Joint Venture` — INCLUDED. Fourth tranche of the same New Line/Heron joint-venture production structure.

**Surface-form notes:** Members per tier list: New Line Cinema (1967-) and Fine Line Features (1990-2005). Picturehouse (2005-2008 WB/New Line JV) explicitly excluded by tier list as primarily distribution. TV (New Line Television), home video (New Line Home Entertainment/Video), and international distribution (New Line International) arms are excluded because they are non-production. New Line-Heron Joint Venture entries are included as they are film production credits tied to New Line. Fine Line Media and Fine Line Productions are treated as name collisions (insufficient evidence they are the New Line arthouse imprint, which was consistently branded 'Fine Line Features').

## DC — "DC Studios"

**Retune status:** DROPPED pre-2009 `DC Comics` (too broad — IMDB applies it as a source credit across animated DTV and licensed tie-ins casual viewers don't mean by "DC movies"). KEPT: DC Entertainment (2009-2016), DC Films (2016-2022), DC Studios (2022-). Classic theatrical DC (Burton Batman, Donner Superman) correctly tags only WARNER_BROS via its WB credit, which matches casual expectation.

| Member | Start | End | Rationale |
|---|---|---|---|
| DC Comics | — | 2009 | Earliest production credit on pre-2009 WB DC films (Batman '89 era, Superman Returns) |
| DC Entertainment | 2009 | 2016 | Subsidiary formed 2009; credited through DCEU until DC Films |
| DC Films | 2016 | 2022 | Dedicated film division through 2022 |
| DC Studios | 2022 | — | Current Gunn/Safran division |

**Rename history:** DC Comics credit → DC Entertainment (2009) → DC Films (2016) → DC Studios (2022).

**Notes:** Also member of WARNER_BROS always (via DC Comics ownership by WB since 1969). Excludes DC Animation (TV) and DC Universe streaming.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| DC Entertainment | `DC Entertainment` | 134 |
| DC Comics | `DC Comics` | 25 |
| DC Films | `DC Films` | 19 |
| DC Studios | `DC Studios` | 3 |

**Collisions / exclusions flagged:**

- `DC Pictures` — Name collides with DC Studios brand but no evidence this is the WB/DC label; singleton credit from an unrelated indie production. Excluded.
- `DC Productions` — Generic name; no credible association with the DC Studios brand. Excluded.
- `DC Cinema` — Not a recognized WB/DC label; likely an unrelated indie shingle. Excluded.
- `DC Media` — Not a recognized WB/DC label; unrelated media outfit. Excluded.
- `DC Medias` — Unrelated indie company. Excluded.
- `DC Vertigo` — Vertigo is a DC Comics mature-readers imprint (comics-only, adjacent to DC Black Label). Per task instructions, comics-only imprints are excluded from the DC Studios film brand.
- `DC Republic Productions` — Unrelated indie production; 'Republic' suggests no tie to DC Studios. Excluded.
- `DC Film Television & Entertainment Rebate Fund` — District of Columbia (Washington, DC) production rebate fund — geographic DC, not the brand. Excluded.
- `DC Creative` — No tie to the DC Studios brand. Excluded.
- `DC Digital Content Comics` — Not a WB/DC film credit; likely unrelated digital comics entity. Excluded.
- `DC Dogs` — Unrelated indie name. Excluded.
- `DC Flav Films` — Unrelated indie. Excluded.
- `DC Stories` — Unrelated indie. Excluded.
- `ADC Films` — Substring match only; unrelated company. Excluded.
- `AMDC Films` — Substring match only; unrelated. Excluded.
- `DDC Films LLC` — Substring match only; unrelated. Excluded.
- `IDC Entertainment` — Substring match only; unrelated. Excluded.
- `Kinologic Films (DC)` — Geographic DC reference. Excluded.
- `50/50 Films DC` — Geographic DC reference. Excluded.
- `Beirut DC` — Beirut-based arts org (Lebanon); unrelated. Excluded.
- `Gather by Events DC` — DC (Washington) events company; unrelated. Excluded.
- `Residue DC` — Unrelated; likely geographic reference. Excluded.
- `Fairy Tale DC Production Committee for Theater Version` — Japanese production committee with 'DC' in its title (unrelated meaning). Excluded.

**Surface-form notes:** Four clean canonical members found in TSV: DC Entertainment (134), DC Comics (25), DC Films (19), DC Studios (3). Per tier list, excluded DC Animation (TV) and DC Universe streaming originals — neither appears in TSV as a production_companies string. DC Black Label (comics-only imprint) likewise absent; the only Vertigo-adjacent hit is 'DC Vertigo' (count 1), excluded as comics-only. Several 'DC ...' singletons (DC Pictures, DC Productions, DC Media, etc.) are name-space collisions with unrelated indie companies or geographic (Washington, DC) references and were excluded. Tier list curation flag #4 satisfied: pre-2009 'DC Comics' production credit is retained.

## UNIVERSAL — "Universal Pictures"

**Retune status:** DROPPED all Focus Features surfaces + precursors (Good Machine, USA Films), DreamWorks Animation + PDI, Working Title Films + WT2 Productions, Gramercy Pictures. KEPT: core Universal + Illumination (Universal actively co-brands Minions/Despicable Me — theme-park integration etc.). Proposed ADD `Illumination` (bare, post-2018 rename) was validated against IMDB and removed — zero evidence (IMDB retains `Illumination Entertainment` even post-rename).

| Member | Start | End | Rationale |
|---|---|---|---|
| Universal Pictures | 1912 | — | Flagship NBCUniversal film label |
| Focus Features | 2002 | — | Universal's specialty/arthouse label |
| Illumination | 2007 | — | Animation studio with Universal financing/distribution from inception |
| Illumination Mac Guff | 2011 | — | Paris CG arm producing all Illumination features |
| DreamWorks Animation | 2016 | — | Acquired Aug 2016 |
| Working Title Films | 1999 | — | Majority-owned via PolyGram acquisition |
| Gramercy Pictures | 1992 | 1999 | PolyGram/Universal specialty label (Four Weddings, Fargo) |

**Rename history:** Universal Film Manufacturing Company (1912) → Universal-International (1946-1964) → Universal Pictures (1964).

**Notes:** Working Title is ~67% Universal-owned with founders holding a minority stake. Amblin's Universal relationship is handled separately and is not treated as a UNIVERSAL brand-member edge here.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| Universal Pictures | `Universal Pictures` | 2037 |
|  | `Universal Film Manufacturing Company` | 597 |
|  | `Universal International Pictures (UI)` | 334 |
|  | `Universal Pictures International (UPI)` | 24 |
|  | `Universal` | 3 |
| Focus Features | `Focus Features` | 124 |
|  | `Focus Features International (FFI)` | 7 |
|  | `Focus World` | 3 |
|  | `Focus Features Africa First Program` | 1 |
| USA Films | `USA Films` | 11 |
| Good Machine | `Good Machine` | 45 |
|  | `Good Machine Films` | 2 |
| Illumination | `Illumination Entertainment` | 59 |
| Illumination Mac Guff | `Mac Guff Ligne` | 15 |
| DreamWorks Animation | `DreamWorks Animation` | 120 |
|  | `Pacific Data Images (PDI)` | 24 |
| Working Title Films | `Working Title Films` | 156 |
|  | `WT2 Productions` | 10 |
| Gramercy Pictures | `Gramercy Pictures (I)` | 18 |
|  | `Gramercy Pictures (II)` | 6 |

**Collisions / exclusions flagged:**

- `Universal`
- `Gramercy Pictures (I)`
- `Mac Guff Ligne`
- `Focus World`
- `Pacific Data Images (PDI)`

**Surface-form notes:** Excluded per brief: 'Universal Studios' (34) as theme-park/umbrella brand, 'NBCUniversal' (9) and 'NBC Universal Entertainment' (7) as corporate parent, 'DreamWorks Pictures' (138) as belonging to AMBLIN brand. Excluded TV/music/home-video subsidiaries (Universal Television, Universal Music, Universal Pictures Home Entertainment, Universal 1440 Entertainment, Universal Animation Studios, Universal Cartoon Studios, etc.) since they are not film production credits for the flagship brand. International Universal Pictures production arms (Universal Pictures UK, Universal Pictures Germany, Universal Pictures Spain, Universal Productions France, Universal Pictures Japan) are primarily distribution credits and excluded to avoid false positives — the UPI (International Pictures) umbrella is already captured. 'Universal-International' with hyphen does NOT appear verbatim in the TSV; 'Universal International Pictures (UI)' is its in-TSV form. 'Illumination Entertainment' is the only verbatim Illumination string (not 'Illumination' bare). Focus Features predecessors (USA Films, Good Machine) listed as separate canonical members matching tier-list granularity. All counts verified verbatim against /tmp/company_strings.tsv.

## FOCUS_FEATURES — "Focus Features"

**Retune status:** DROPPED `Focus World` (niche VOD/genre sub-label) and `Focus Features Africa First Program` (short-film residency). KEPT: core Focus + Good Machine + USA Films precursors (catalog tonally identical to Focus's prestige-indie identity).

| Member | Start | End | Rationale |
|---|---|---|---|
| Focus Features | 2002 | — | Primary credit on Universal prestige films |
| Focus World | 2010 | 2013 | Genre/VOD arm |
| Good Machine | 1991 | 2002 | Predecessor; credits appear on early Focus films (The Pianist, Far from Heaven) |
| USA Films | 1999 | 2002 | Predecessor; credits on Gosford Park, Traffic |

**Rename history:** USA Films + Good Machine + Universal Focus → Focus Features (2002).

**Notes:** Also member of UNIVERSAL always. Gramercy Pictures excluded here — users don't associate Gramercy titles with Focus.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| Focus Features | `Focus Features` | 124 |
|  | `Focus Features International (FFI)` | 7 |
|  | `Focus Features Africa First Program` | 1 |
| Focus World | `Focus World` | 3 |
| Good Machine | `Good Machine` | 45 |
|  | `Good Machine Films` | 2 |
| USA Films | `USA Films` | 11 |

**Collisions / exclusions flagged:**

- `Prime Focus World` — VFX services company (Prime Focus), unrelated to Focus Features. Explicitly excluded per instructions.
- `Fusa Films` — Substring match on 'usa Films' but unrelated company name. Excluded.
- `Lampedusa Films` — Substring match on 'usa Films' but unrelated. Excluded.
- `Aretusa Films` — Substring match on 'usa Films' but unrelated. Excluded.
- `Brooklyn USA Films.` — Contains 'USA Films' as part of a different company name ('Brooklyn USA Films'); not the USA Films label that became Focus Features. Excluded.
- `Latin USA Films` — Different company sharing 'USA Films' substring. Excluded.
- `Lusa Films S.L.` — Substring match only; Spanish company unrelated to USA Films. Excluded.
- `Syracusa Films` — Substring match only; unrelated company. Excluded.

**Surface-form notes:** Focus Features is the specialty arm of NBCUniversal, formed in 2002 from the merger of USA Films (1999-2002) and Good Machine (1991-2002); Focus World (2010-2013) was its genre/VOD label. Gramercy Pictures is explicitly excluded per tier list (users don't associate it with Focus). 'Prime Focus World' is a VFX services company (Prime Focus Limited), not affiliated with Focus Features, and was flagged as a collision. All 'USA Films' substring matches other than the exact 'USA Films' string were vetted and excluded as unrelated companies. 'Focus Features Africa First Program' is a Focus Features talent initiative and is included as a member string.

## PARAMOUNT — "Paramount Pictures"

**Retune status:** DROPPED all Nickelodeon entries (distinct kids-TV brand — SpongeBob/Rugrats/Dora not "Paramount" to casual viewers), all MTV Films entries (MTV's own brand identity — Jackass, Napoleon Dynamite), and the revived `Republic Pictures` entries (own brand identity). KEPT: core Paramount + Paramount Players + Paramount Animation + Paramount Vantage / Classics (the `Paramount` prefix signals co-branding and satisfies the casual-viewer test for Vantage's There Will Be Blood / No Country for Old Men etc.).

| Member | Start | End | Rationale |
|---|---|---|---|
| Paramount Pictures | 1912 | — | Flagship Hollywood studio |
| Paramount Players | 2017 | — | Paramount label for Viacom/Paramount-adjacent commercial properties |
| Paramount Animation | 2011 | — | In-house animation division |
| Paramount Vantage | 2006 | 2013 | Prestige imprint (No Country for Old Men, There Will Be Blood) |
| Nickelodeon Movies | 1995 | — | Theatrical films distributed as Paramount releases |
| MTV Films | 1996 | — | Youth-skewing Paramount-owned label |
| Republic Pictures | 2023 | — | 2023 Paramount revival as specialty/acquisitions imprint; requires release-year gating from the legacy Republic catalog |

**Rename history:** Famous Players (1912) → Famous Players-Lasky (1916) → Paramount Pictures (1930s); Viacom (1994) → ViacomCBS (2019) → Paramount Global (2022); Paramount + Skydance merger 2024-2025.

**Notes:** DreamWorks Pictures 2006-2008 Paramount-distribution era excluded — films catalogued under DreamWorks, not user-expected Paramount. Miramax is not treated as a PARAMOUNT member here because the 2020 deal was a minority stake plus distribution/first-look arrangement, not full brand ownership.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| Paramount Pictures | `Paramount Pictures` | 2222 |
|  | `Paramount` | 3 |
|  | `Paramount British Pictures` | 18 |
|  | `Paramount Films` | 1 |
| Paramount Players | `Paramount Players` | 15 |
| Paramount Animation | `Paramount Animation` | 18 |
|  | `Paramount Animation Studios` | 1 |
| Paramount Vantage | `Paramount Vantage` | 17 |
|  | `Paramount Classics` | 4 |
| Nickelodeon Movies | `Nickelodeon Movies` | 70 |
|  | `Nickelodeon Animation Studios` | 66 |
|  | `Nickelodeon Films` | 2 |
| MTV Films | `MTV Films` | 46 |
|  | `MTV Entertainment Studios` | 47 |
|  | `MTV Films Europe` | 4 |
| Republic Pictures | `Republic Pictures` | 884 |
|  | `Republic Pictures (III)` | 1 |

**Collisions / exclusions flagged:**

- `New Republic Pictures`
- `Fun Republic Pictures`
- `Weimaraner Republic Pictures`
- `Paramount Famous Lasky Corporation`
- `Paramount Famous Productions`
- `Paramount Pictures Cartoon Studios`
- `Paramount-Orion Filmproduktion`
- `Paramount Overseas Productions Inc.`
- `Les Studios Paramount`
- `Paramount Studios, Joinville France`
- `Nickelodeon Productions`
- `Nickelodeon Studios`
- `MTV Animation`
- `MTV Productions`
- `Paramount Television`
- `Paramount Network Television Productions`
- `Paramount Network Television`
- `Paramount Television Productions`
- `Paramount Home Entertainment`
- `Paramount Pictures Digital Entertainment`
- `Paramount+`
- `Paramount Global`
- `Paramount Global Content Distribution`
- `Paramount Italiana`
- `Paramount Channel España`
- `Paramount Pictures Spain`
- `Paramount Pictures International`
- `Paramount International`
- `Paramount Filmproduction GmbH`
- `Paramount Scope`
- `Paramount News`
- `British Paramount News`
- `Ananey - Paramount`
- `Talent Associates-Paramount`
- `CBS Paramount Network Television`
- `CBS Paramount Domestic Television`
- `United Paramount Network (UPN)`
- `Paramount Network`
- `Vikram-Paramount Studios`
- `Films Paramount`
- `Famous Players`
- `Famous Players-Lasky Corporation`
- `Famous Players Film Company`
- `Famous Players Limited`
- `Famous Players Film Corporation`
- `Famous Players Guild`
- `Famous Players International`
- `Republic Pictures Television`

**Surface-form notes:** Verbatim TSV enumeration for brand PARAMOUNT. Included surface forms per canonical member: Paramount Pictures (Paramount Pictures, Paramount, Paramount British Pictures, Paramount Films); Paramount Players (Paramount Players); Paramount Animation (Paramount Animation, Paramount Animation Studios); Paramount Vantage (Paramount Vantage, Paramount Classics); Nickelodeon Movies (Nickelodeon Movies, Nickelodeon Animation Studios, Nickelodeon Films); MTV Films (MTV Films, MTV Entertainment Studios, MTV Films Europe); Republic Pictures (Republic Pictures, Republic Pictures (III)). `Republic Pictures` is release-year-gated: the same raw string covers both the historical 1935-1967 company and Paramount's 2023 revival, so only 2023+ movies should stamp to PARAMOUNT through this member row. Key exclusions per task scope: all Paramount Television/Network/+/Global/foreign-distribution variants; pre-1930s Famous Players / Famous Players-Lasky lineage (flagged, default excluded despite 345+ combined credits — curator may re-include if scope extends); DreamWorks Pictures 2006-2008 distribution era (excluded per tier note). Collision risk `Republic Pictures` token: at least 3 unrelated companies (New Republic Pictures, Fun Republic Pictures, Weimaraner Republic Pictures) share the token and are flagged for exclusion from token-index matching. Borderline in-house theatrical credits flagged for curator: Paramount Pictures Cartoon Studios (35), Paramount Famous Lasky Corporation (32), Paramount-Orion Filmproduktion (6), MTV Animation (8), Nickelodeon Productions (75).

## SONY — "Sony Pictures"

**Retune status:** Heavily trimmed. DROPPED: all foreign-region Columbia variants (Brazil x2, Mexico, Argentina, Asia, Germany, British Productions), obscure corporate labels (`Columbia Pictures Entertainment`, `Columbia Films Productions`, `Columbia Release`), `Sony Pictures Classics` (distinct arthouse brand — Whiplash, Call Me By Your Name), low-profile imprints (Stage 6 Films + Productions, Triumph Films x2), all distribution-only credits (Sony Pictures Releasing, Sony Pictures Releasing International, Sony Pictures International, SPWA, Sony BMG Feature Films, Sony International MPPG, Sony Pictures Films India, Sony / Monumental). KEPT: Columbia core + TriStar + Screen Gems + Sony Pictures Animation + core Sony umbrella credits.

| Member | Start | End | Rationale |
|---|---|---|---|
| Columbia Pictures | 1989 | — | Flagship Sony label since Columbia Pictures Entertainment acquisition |
| TriStar Pictures | 1989 | — | Acquired alongside Columbia in 1989 |
| Screen Gems | 1999 | — | Sony's horror/genre imprint (relaunched 1999) |
| Sony Pictures Classics | 1992 | — | Arthouse/specialty distribution label |
| Sony Pictures Animation | 2002 | — | In-house animation studio |
| Stage 6 Films | 2007 | — | Low-budget / limited-theatrical label |
| Triumph Films | 1989 | — | Columbia/Gaumont JV label with small late-80s/90s library |

**Rename history:** Columbia TriStar Motion Picture Group (1998-2002) briefly unified Columbia + TriStar before separation.

**Notes:** Excludes Sony Pictures Television, Sony Music, PlayStation Productions (minimal film catalog as of 2026), Sony-Japan-only entities.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| Columbia Pictures | `Columbia Pictures` | 2188 |
|  | `Columbia Pictures Corporation` | 19 |
|  | `Columbia Pictures Entertainment` | 2 |
|  | `Columbia Pictures Industries` | 3 |
|  | `Columbia Pictures Film Production Asia` | 18 |
|  | `Deutsche Columbia Pictures Film Produktion` | 20 |
|  | `Columbia Pictures do Brasil` | 7 |
|  | `Columbia Pictures of Brasil` | 3 |
|  | `Columbia Pictures Producciones Mexico` | 4 |
|  | `Columbia Pictures of Argentina` | 1 |
|  | `Columbia Films` | 22 |
|  | `Columbia Films Productions` | 1 |
|  | `Columbia Productions` | 1 |
|  | `Columbia British Productions` | 21 |
|  | `Columbia Release` | 1 |
|  | `Columbia` | 4 |
| TriStar Pictures | `TriStar Pictures` | 131 |
|  | `Tri-Star Pictures` | 76 |
|  | `TriStar Productions` | 5 |
|  | `Tri Star Productions` | 2 |
|  | `Tri Star` | 1 |
| Screen Gems | `Screen Gems` | 180 |
| Sony Pictures Classics | `Sony Pictures Classics` | 39 |
| Sony Pictures Animation | `Sony Pictures Animation` | 54 |
| Stage 6 Films | `Stage 6 Films` | 82 |
|  | `Stage 6 Productions` | 1 |
| Triumph Films | `Triumph Films` | 14 |
|  | `Triumph Films (II)` | 4 |
| Sony Pictures (umbrella credit) | `Sony Pictures` | 1 |
|  | `Sony Pictures Entertainment` | 14 |
|  | `Sony Pictures Entertainment Company` | 2 |
|  | `Sony Pictures Releasing` | 81 |
|  | `Sony Pictures Releasing International` | 1 |
|  | `Sony Pictures International` | 8 |
|  | `Sony Pictures International Productions` | 53 |
|  | `Sony Pictures Worldwide Acquisitions (SPWA)` | 7 |
|  | `Sony Pictures Studios` | 2 |
|  | `Sony BMG Feature Films` | 3 |
|  | `Sony International Motion Picture Production Group` | 1 |
|  | `Sony Pictures Films India` | 3 |
|  | `Sony / Monumental Pictures` | 2 |

**Collisions / exclusions flagged:**

- `Columbia`
- `Sony`
- `Sony Pictures Digital`
- `Sony Pictures Digital Productions`
- `Sony Pictures Film- und Fernseh Produktion`
- `Sony Pictures Entertainment (Japan)`
- `Sony Pictures Networks`
- `Sony Pictures Networks Productions`
- `Sony Pictures High Definition Center`
- `Sony Cinematic`
- `Columbia TriStar Entertainment`
- `Columbia TriStar Film Distributors International`
- `Columbia TriStar Productions Pty. Ltd.`
- `Walt Disney Pictures / Sony Pictures`

**Surface-form notes:** Seven tier-list members resolved to their TSV surface forms. COLUMBIA, TRISTAR, and SONY_PICTURES_ANIMATION are also standalone MVP brands — their strings are listed here under SONY because they are time-bounded members of SONY since acquisition; curator can dedupe into standalone brand rows. SCREEN_GEMS and SONY_PICTURES_CLASSICS were cut from the MVP (deferred section); their strings stay reachable here via SONY and via freeform token match. Explicitly excluded per tier list: Sony Pictures Television (all variants), Sony Music (all variants), Sony-Japan-only entities (Sony Pictures Entertainment (Japan), CBS Sony, Sony Corporation of Japan), Sony Pictures Imageworks (VFX services), PlayStation Productions (no TSV matches). Home-video/distribution-only entities (Sony Pictures Home Entertainment, Columbia TriStar Home Video/Entertainment) also excluded as non-production. Added an umbrella 'Sony Pictures (umbrella credit)' grouping to capture Sony Pictures Entertainment / Sony Pictures Releasing / SPIP / SPWA / Sony Pictures Studios style credits where no specific sublabel is named; tier list does not explicitly list a bare 'Sony Pictures' credit so this is a judgment call — see collisions_flagged for the ambiguous 'Sony' (bare) entry which was excluded. No bare 'Columbia TriStar' string exists in the TSV (all Columbia TriStar entries are Television/Home Video/regional subsidiaries, excluded); the COLUMBIA brand's 'Columbia TriStar 1998-2002 combined credit' row therefore has no direct surface form in this data.

## COLUMBIA — "Columbia Pictures"

**Retune status:** DROPPED `Columbia British Productions`, `Columbia Films`, `Columbia Productions` (obscure historic B-picture variants with no casual recall), and the joint-venture shells `Columbia-Delphi Productions` and `Columbia-Thompson Venture` (casual viewers attribute those films to plain "Columbia"). KEPT: `Columbia Pictures` (1924-) and `Columbia Pictures Corporation` (1924-1968).

| Member | Start | End | Rationale |
|---|---|---|---|
| Columbia Pictures | 1924 | — | Primary credit since 1924 rename |
| Columbia Pictures Corporation | 1924 | 1968 | Classic-era legal/production credit |

**Rename history:** Cohn-Brandt-Cohn Film Sales (1918) → Columbia Pictures Corporation (1924) → Columbia Pictures Industries (1968); Coca-Cola owned 1982; Sony 1989. The 1998-2002 Columbia TriStar Motion Picture Group era sits in umbrella scope here even though no bare `Columbia TriStar` production-company string appears in the DB.

**Notes:** Also member of SONY since 1989. TriStar excluded as separate brand. Columbia Pictures Industries functions as corporate parent, not on-film credit.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| Columbia Pictures | `Columbia Pictures` | 2188 |
|  | `Columbia British Productions` | 21 |
|  | `Columbia Films` | 22 |
|  | `Columbia-Delphi Productions` | 1 |
|  | `Columbia-Thompson Venture` | 1 |
|  | `Columbia Productions` | 1 |
| Columbia Pictures Corporation | `Columbia Pictures Corporation` | 19 |

**Collisions / exclusions flagged:**

- `Province of British Columbia Film Incentive BC` — British Columbia (Canadian province) tax incentive; unrelated to Hollywood studio
- `Province of British Columbia Production Services Tax Credit` — BC provincial tax credit; unrelated
- `British Columbia Film` — Canadian provincial film agency; unrelated
- `British Columbia Film Commission` — BC government film commission; unrelated
- `British Columbia Arts Council` — BC provincial arts funding body; unrelated
- `British Columbia Television` — Canadian TV broadcaster in province of BC; unrelated
- `British Columbia Broadcasting System` — Canadian regional broadcaster; unrelated
- `British Columbia's Knowledge Network` — Canadian public broadcaster in BC; unrelated
- `British Columbia Hydro Authority` — BC utility company (location filming); unrelated
- `Directors Guild of Canada - British Columbia (DGCBC)` — Canadian guild chapter; unrelated
- `Film Development Society of British Columbia (FDBC)` — BC film funding society; unrelated
- `Government of British Columbia Film Incentive BC Program` — BC provincial tax program; unrelated
- `Province of British Columbia` — Canadian province credit; unrelated
- `Science World British Columbia` — BC science museum; unrelated
- `University of British Columbia` — Canadian university; unrelated
- `Columbia University` — Ivy League university in NYC; unrelated to Columbia Pictures studio
- `Columbia University, New York` — Columbia University variant; unrelated
- `Columbia University School of the Arts` — Columbia University film program; unrelated
- `Columbia Journalism School` — Part of Columbia University; unrelated
- `Columbia Institute for Ideas and Imagination` — Columbia University institute; unrelated
- `Columbia College, Chicago` — Separate arts college in Chicago; unrelated to Columbia Pictures
- `Columbia College Hollywood` — Film school in LA (no hits in TSV, flagged per spec); unrelated
- `Columbia State Community College, Film Crew Technology` — Tennessee community college; unrelated
- `Nippon Columbia` — Japanese record/music company (Nippon Columbia Co., Ltd.); unrelated lineage to Hollywood Columbia Pictures
- `Nippon Columbia Home Video` — Japanese music/video label; unrelated
- `Japan Columbia Corporation` — Nippon Columbia variant; unrelated
- `Columbia Records` — Record label under Sony Music (originally Columbia Phonograph); separate lineage, not the film studio
- `Sony Columbia Records` — Columbia Records variant; music label, not film studio
- `Columbia Broadcasting System (CBS)` — CBS network (originally Columbia Phonograph Broadcasting); unrelated to film studio
- `Columbia Songs` — Music publishing; unrelated to film studio
- `Columbia Music Video` — Music video label; unrelated to film studio
- `Columbia Artists Management` — Classical music talent agency; unrelated
- `Columbia Western Management` — Unrelated management entity
- `Columbia Edutainment` — Unrelated; appears unaffiliated with Columbia Pictures
- `The District of Columbia Medical Society` — Washington DC medical society; unrelated
- `District of Columbia Medical Society` — DC medical society variant; unrelated
- `Columbia` — Ambiguous bare token — could be the studio, but also the country, the university, or the record label. Excluded from members: too unsafe without context disambiguation
- `Columbia Films Productions` — Ambiguous low-count variant; not a known Columbia Pictures credit. Excluded from members pending review
- `Columbia Release` — Ambiguous low-count string; unclear lineage. Excluded from members pending review

**Surface-form notes:** Canonical members per spec are Columbia Pictures (1924-) and Columbia Pictures Corporation (1924-1968), with the 1998-2002 Columbia TriStar Motion Picture Group era handled as umbrella scope even though no isolated bare `Columbia TriStar` string appears in the DB. Included additional surface-form variants that map to Columbia Pictures: Columbia Films (short variant), Columbia British Productions (UK subsidiary with on-screen credits), Columbia-Delphi Productions and Columbia-Thompson Venture (JV on-screen production credits), and Columbia Productions (generic variant). TriStar-branded strings remain excluded because TriStar is a separate brand (TRISTAR), but Columbia TriStar-prefixed subsidiary strings should be treated as belonging under the broader COLUMBIA umbrella when they appear on movie rows rather than being discarded solely for carrying TV/home-video wording. Foreign production/distribution subsidiaries (do Brasil, of Brasil, Producciones Mexico, of Argentina, Deutsche Columbia Pictures Film Produktion, Columbia Pictures Film Production Asia) remain curator-review items; `Columbia Pictures Film Production Asia` in particular has enough theatrical presence to be a plausible inclusion. Columbia Pictures Industries and Columbia Pictures Entertainment are corporate holdings/umbrella credits rather than the core Columbia film-label name, but they remain relevant at the SONY umbrella level. Heavy collision space: British Columbia (Canadian province) accounts for ~15 strings with high counts (especially Province of BC Film Incentive at 88 and Production Services Tax Credit at 75); Columbia University accounts for 5+ strings; Columbia Records / Nippon Columbia / CBS are separate music/broadcasting lineages despite shared Columbia Phonograph ancestry. Bare `Columbia` (count 4) is ambiguous and excluded from members for safety.

## TWENTIETH_CENTURY — "20th Century Studios"

**Retune status:** DROPPED pre-1935 Fox Film Corporation / Fox Film Company / Fox Films (silent/early-talkie era is film-historian territory, not casual mental model). KEPT: all post-1935 20th Century Fox + Studios variants + Fox 2000 Pictures (Fight Club, Cast Away, Walk the Line, Life of Pi were Fox-marketed). Window fix: `20th Century Pictures` widened from (1935, 2020) to (1933, 2020) to cover pre-merger films.

| Member | Start | End | Rationale |
|---|---|---|---|
| Twentieth Century Fox Film Corporation | 1935 | 2020 | Pre-rename core production entity |
| 20th Century Studios | 2020 | — | Post-Disney rename (Jan 2020) |
| Fox 2000 Pictures | 1994 | 2020 | Prestige imprint (Cast Away, Life of Pi, Hidden Figures); shuttered post-acquisition |

**Rename history:** Twentieth Century Fox Film Corporation → 20th Century Studios (January 2020).

**Notes:** Also member of DISNEY since 2019. Excludes Fox Searchlight (separate brand SEARCHLIGHT), Fox Star Studios (India), Blue Sky Studios (animation studio typically credited distinctly).

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| Twentieth Century Fox Film Corporation | `Twentieth Century Fox` | 1358 |
|  | `Twentieth Century-Fox Productions` | 42 |
|  | `Twentieth Century Fox Animation` | 37 |
|  | `20th Century Pictures` | 24 |
|  | `20th Century Fox` | 19 |
|  | `Twentieth Century Productions` | 1 |
|  | `Twentieth Century-Fox Studios, Hollywood` | 1 |
|  | `20th Century Foss` | 1 |
| Fox Film Corporation (pre-1935) | `Fox Film Corporation` | 892 |
|  | `Fox Film Company` | 8 |
|  | `Fox Films` | 2 |
| Fox 2000 Pictures | `Fox 2000 Pictures` | 78 |
| 20th Century Studios | `20th Century Studios` | 56 |
|  | `Twentieth Century Animation` | 11 |

**Collisions / exclusions flagged:**

- `20th Century Fox Television` — TV division — excluded per task instructions.
- `20th Television` — TV division — excluded per task instructions.
- `20th Television Animation` — TV animation division — excluded.
- `Fox Television Studios` — TV division — excluded.
- `Fox Television Network` — TV network — excluded.
- `Fox Television Animation` — TV animation — excluded.
- `Fox Television` — TV — excluded.
- `Fox Television Stations` — TV — excluded.
- `Fox Television Studios Pictures` — TV — excluded.
- `Fox 21 Television Studios` — TV — excluded.
- `Fox Searchlight Pictures` — Separate brand SEARCHLIGHT — excluded per tier list.
- `Fox STAR Studios` — India / Fox Star Studios — excluded per tier list.
- `Fox Star Studios` — India / Fox Star Studios — excluded per tier list.
- `Blue Sky Studios` — Separately credited animation studio — excluded per tier list (belongs to DISNEY brand umbrella only, not TWENTIETH_CENTURY).
- `20th Century Fox Home Entertainment` — Home-video distribution arm; not a theatrical production credit. Excluded per 'foreign distribution arms / distribution-only' convention.
- `20th Century Fox Argentina` — Foreign distribution arm — excluded per selection criteria.
- `20th Century Fox Korea` — Foreign distribution arm — excluded.
- `20th Century Fox Post Production Services` — Services entity, not a production company — excluded.
- `Fox Film Europa, Paris` — Foreign distribution arm — excluded.
- `Twentieth Century Vixen` — Unrelated entity (adult film production) — name collision only.
- `20th Century Vixen` — Unrelated entity — name collision only.
- `Fox International Productions (FIP)` — International co-production arm — excluded per 'foreign distribution arms' convention; curator may reconsider if needed.
- `Fox International Productions` — International arm — excluded.
- `Fox Atomic` — Short-lived (2006-2009) genre sublabel. NOT listed in tier file members; excluding as a conservative default. Curator may promote to member if desired.
- `Fox Animation Studios` — Ambiguous — historically Phoenix-based Don Bluth-era unit (Anastasia, Titan A.E.). Arguably a TCF member, but not listed in tier file; flagged for curator review.
- `Fox Searchlab` — Fox Searchlight incubator — belongs to SEARCHLIGHT brand, not TWENTIETH_CENTURY.
- `Fox West Pictures` — Unclear affiliation / likely unrelated — excluded; curator review.

**Surface-form notes:** Enumeration drawn from /tmp/company_strings.tsv by grepping for 'Twentieth Century', '20th Century', 'Fox 2000', 'Fox Film', 'Fox Pictures', and 'Fox Films'. Members cover three ownership/rename eras: (1) Fox Film Corporation (pre-1935) and its variants, (2) Twentieth Century Fox Film Corporation (1935-2020) and its variants including the in-house Animation unit and the Fox 2000 imprint, and (3) 20th Century Studios (post-Jan 2020 rename). Also includes 20th Century Pictures (1933-1935), Darryl Zanuck's company that merged with Fox Film Corporation in 1935 to form Twentieth Century Fox — explicitly part of the merger forming the core brand. Excluded per task: Fox Searchlight (SEARCHLIGHT brand), Fox Star Studios (India), Blue Sky Studios (separately credited, mapped to DISNEY umbrella only), all TV divisions (20th Century Fox Television, 20th Television, Fox Television Studios, etc.), home-entertainment and post-production-services entities, foreign regional distribution arms (Argentina, Korea, Europa/Paris), and Fox International Productions. Curator-review items: 'Fox Atomic' (Tier-list omitted but was a legit 2006-2009 TCF sublabel), 'Fox Animation Studios' (Phoenix-era Don Bluth unit), and '20th Century Foss' (probable OCR typo of 20th Century Fox). 'Twentieth Century Vixen' / '20th Century Vixen' are unrelated name collisions (1972 animated adult feature and its sequel).

## SEARCHLIGHT — "Searchlight Pictures"

**Retune status:** No member drops. Window fix: `Searchlight Pictures` widened from (2020, None) to (None, None) because IMDB retroactively applies the current name to pre-rename films (Slumdog Millionaire 2008, Black Swan 2010, 12 Years a Slave 2013 were all missing their tag).

| Member | Start | End | Rationale |
|---|---|---|---|
| Fox Searchlight Pictures | 1994 | 2020 | Original label name |
| Searchlight Pictures | 2020 | — | Post-rename |

**Rename history:** Fox Searchlight Pictures → Searchlight Pictures (January 2020).

**Notes:** Also member of DISNEY since 2019.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| Fox Searchlight Pictures | `Fox Searchlight Pictures` | 104 |
| Searchlight Pictures | `Searchlight Pictures` | 50 |

**Collisions / exclusions flagged:**

- `Mukta Searchlight Films` — Indian production company (Mukta Arts joint venture). Unrelated to Fox/Disney Searchlight Pictures.
- `Searchlight Films` — Distinct small production company; not the canonical 'Searchlight Pictures' label. Name similarity only.
- `Searchlight Productions` — Unrelated company sharing the 'Searchlight' name.
- `Creative Searchlight` — Unrelated company; no affiliation with the Searchlight Pictures brand.

**Surface-form notes:** Searchlight is a compact brand with just two canonical strings: 'Fox Searchlight Pictures' (1994-2020) and 'Searchlight Pictures' (2020-) following the January 2020 Disney-era rebrand. Four other TSV entries contain the word 'Searchlight' but are unrelated companies (Indian JV Mukta Searchlight, plus small outfits named Searchlight Films / Searchlight Productions / Creative Searchlight). No regional arms found. Brand is also a member of DISNEY since 2019.

## DREAMWORKS_ANIMATION — "DreamWorks Animation"

**Retune status:** No member drops. Proposed ADDs `DreamWorks Animation SKG` and `PDI/DreamWorks` were validated against IMDB and removed — zero evidence in live data. Live-action DreamWorks (Gladiator, American Beauty, Transformers) remains correctly excluded — a separate entity casual viewers don't conflate.

| Member | Start | End | Rationale |
|---|---|---|---|
| DreamWorks Animation | 2004 | — | Current name since public spin-off |
| DreamWorks Animation SKG | 2004 | 2016 | Formal corporate name 2004-2016 |
| Pacific Data Images | 1994 | 2015 | PDI/DreamWorks co-credit on Shrek, Madagascar, Kung Fu Panda |

**Rename history:** DreamWorks SKG animation division (1994-2004) → DreamWorks Animation SKG (2004) → DreamWorks Animation (post-2016).

**Notes:** Also member of UNIVERSAL since 2016. Excludes DreamWorks Pictures (live-action, separate brand), DreamWorks Records/Interactive/TV.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| DreamWorks Animation | `DreamWorks Animation` | 120 |
| Pacific Data Images | `Pacific Data Images (PDI)` | 24 |

**Collisions / exclusions flagged:**

- `DreamWorks Pictures` — Live-action DreamWorks Pictures (would have belonged to AMBLIN, now deferred). Not an animation string; excluded from DWA.
- `DreamWorks Animation Television` — Television arm — excluded per tier list (TV).
- `DreamWorks Television` — Television arm — excluded per tier list (TV).
- `DreamWorks Studios` — Live-action DreamWorks Studios (post-2008 DreamWorks II / Reliance-backed successor to DreamWorks Pictures). Live-action (AMBLIN-adjacent, now deferred) — excluded from DWA.
- `DreamWorks Classics` — Classic Media legacy catalog brand (Lassie, Rocky & Bullwinkle, Casper reissues) — not an animation production unit. Excluded.
- `DreamWorks Records` — Record label — excluded per tier list.
- `DreamWorks Theatricals` — Live-theater division — not film production. Excluded.
- `Oriental DreamWorks` — Shanghai-based JV between DWA and Chinese partners (2012-2018, later rebranded Pearl Studio). DWA-adjacent but a separately incorporated co-production entity; flagging but not including as a DWA member string.
- `DreamWorks Home Entertainment` — Home-video distribution label — not a production credit. Excluded.
- `DreamWorks Television Animation Studio` — Television animation arm — excluded per tier list (TV).
- `Sucheta DreamWorks Productions` — Indian production company, name-collision only — unrelated to DreamWorks Animation.
- `Sucheta DreamWorks Productions [IND]` — Same Indian company with region tag — unrelated.
- `Kaustav Dreamworks` — Indian production company, name-collision only — unrelated.
- `Shadow Dreamworks` — Name-collision only — unrelated to DreamWorks Animation.
- `East West Dream Works Entertainment` — Two-word 'Dream Works' — unrelated Asian production company, name-collision only.
- `PDI Media` — Name-collision — this 'PDI' is a different entity, not Pacific Data Images the DWA animation arm.

**Surface-form notes:** TSV contains no 'DreamWorks Animation SKG' variant and no 'PDI/DreamWorks' variant — only bare 'DreamWorks Animation' and 'Pacific Data Images (PDI)' appear as legitimate DWA member strings. All other DreamWorks* strings are live-action (AMBLIN-adjacent, deferred from MVP), TV, records, theater, distribution, or unrelated name collisions. Oriental DreamWorks (JV) is flagged rather than included. Only one two-word 'Dream Works' string exists in the TSV and it is an unrelated Asian company.

## ILLUMINATION — "Illumination"

**Retune status:** DROPPED `Mac Guff Ligne` (pre-2011 French VFX house — Illumination's Paris animation unit was spun off from it in 2011 and renamed `Illumination Studios Paris`, which is the relevant surface going forward). KEPT: `Illumination Entertainment` (widened to (2007, None) — IMDB retains this credit even post-2018 rename) + `Illumination Studios Paris`. Proposed ADD bare `Illumination` was validated and removed — zero evidence.

| Member | Start | End | Rationale |
|---|---|---|---|
| Illumination | 2018 | — | Current brand name post-rebrand |
| Illumination Entertainment | 2007 | 2018 | Original name, used through late 2010s |
| Illumination Mac Guff | 2011 | — | Paris-based in-house animation arm |

**Rename history:** Illumination Entertainment → Illumination (~2018).

**Notes:** Also member of UNIVERSAL always (distribution partnership from inception).

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| Illumination Entertainment | `Illumination Entertainment` | 59 |
| Illumination Mac Guff | `Illumination Studios Paris` | 1 |
|  | `Mac Guff Ligne` | 15 |

**Collisions / exclusions flagged:**

- `Illuminations Films` — UK-based production company unrelated to Illumination Entertainment. Note the pluralized 'Illuminations' — different entity entirely.
- `Illumination Films` — Unrelated UK/indie production company. Not the Universal-partnered animation studio.
- `Illuminations` — UK arts and classical-music focused production company (Illuminations Ltd / Illuminations Media), unrelated to Meledandri's animation studio.
- `Illumination Cinema` — Unrelated small production entity — name collision only.
- `RD Illuminations` — Likely lighting services or unrelated production entity. Not affiliated with Illumination Entertainment.
- `Illuminations Television` — UK Illuminations Media TV arm (arts documentaries). Unrelated to the Universal animation studio.
- `Illumination Productions` — Low-count name-collision; not a known Illumination Entertainment credit form.
- `Illumination Pictures` — Unrelated indie entity; Illumination Entertainment does not use 'Pictures' in its credits.
- `Illumination Studios` — Ambiguous one-off; 'Illumination Studios Paris' (the real Illumination Paris arm) is a separate row. This bare form is more likely an unrelated entity or transcription variant; excluded conservatively.
- `Film Illumination` — Word-order variant; not an Illumination Entertainment credit form.
- `Les Illuminations` — French-named unrelated entity (likely named after the Rimbaud poem cycle or Britten composition).

**Surface-form notes:** The flagship brand member is overwhelmingly 'Illumination Entertainment' (count 59), with no standalone 'Illumination' string appearing in the TSV — suggesting that since the 2018 rebrand IMDB production_companies still predominantly retains the legacy 'Illumination Entertainment' credit. 'Illumination Studios Paris' (1) captures the post-2022 Paris-arm rebrand. 'Mac Guff Ligne' (15) is included as the predecessor/parent of Illumination Mac Guff; because Mac Guff Ligne continued as an independent VFX shop, some of those 15 credits may be non-Illumination work and should be verified during member-row curation. All other Illumination-token strings are unrelated (UK Illuminations Films/Illuminations Media ecosystem and assorted one-offs) and are flagged as collisions. No 'Illumination Mac Guff' surface string appears in the TSV directly — if it appears in imdb_data in practice, add it during DB-scan verification.

## MGM — "Metro-Goldwyn-Mayer"

**Retune status:** DROPPED the shorts-era animation units (`Metro-Goldwyn-Mayer Cartoon Studios`, `Metro-Goldwyn-Mayer Animation`, `MGM Animation/Visual Arts` — Tom and Jerry era, not casual theatrical expectation), `MGM-Pathé Communications Co.` (two-year 1990-92 corporate shell), `MGM Producción` (Spanish-language regional arm), and `MGM Family Entertainment` (kids sub-imprint). KEPT: core MGM labels + `Metro-Goldwyn-Mayer British Studios` / `MGM British Studios` (produced 2001: A Space Odyssey and early Bond).

| Member | Start | End | Rationale |
|---|---|---|---|
| Metro-Goldwyn-Mayer | 1924 | — | Core studio credit since 1924 merger |
| Metro-Goldwyn-Mayer Pictures | 1924 | — | Common formal variant on theatrical releases |
| MGM Studios | 1986 | — | Shortened credit variant used post-1980s restructurings |
| Metro-Goldwyn-Mayer Cartoon Studio | 1937 | 1957 | Tom and Jerry, Droopy-era in-house animation |
| Metro-Goldwyn-Mayer British Studios | 1936 | 1970 | UK-based production arm at Borehamwood |

**Notes:** Also member of AMAZON_MGM since 2022. Amazon MGM Studios umbrella credit belongs under AMAZON_MGM, not here. Excludes United Artists (separate brand), Orion (separate brand), MGM Television.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| Metro-Goldwyn-Mayer (MGM) | `Metro-Goldwyn-Mayer (MGM)` | 2265 |
| Metro-Goldwyn-Mayer Cartoon Studios | `Metro-Goldwyn-Mayer Cartoon Studios` | 276 |
| Metro-Goldwyn-Mayer British Studios | `Metro-Goldwyn-Mayer British Studios` | 39 |
| Metro-Goldwyn-Mayer Animation | `Metro-Goldwyn-Mayer Animation` | 7 |
| Metro-Goldwyn-Mayer (MGM) Studios | `Metro-Goldwyn-Mayer (MGM) Studios` | 2 |
| Metro-Goldwyn-Mayer Studios | `Metro-Goldwyn-Mayer Studios` | 1 |
| MGM Animation/Visual Arts | `MGM Animation/Visual Arts` | 38 |
| MGM Family Entertainment | `MGM Family Entertainment` | 5 |
| MGM British Studios | `MGM British Studios` | 1 |
| MGM-Pathé Communications Co. | `MGM-Pathé Communications Co.` | 3 |
| MGM Producción | `MGM Producción` | 1 |

**Collisions / exclusions flagged:**

- `MGM Television` — explicitly excluded per tier list (TV arm).
- `Amazon MGM Studios` — belongs to AMAZON_MGM brand, excluded.
- `MGM Home Entertainment` — home-video distribution arm, not a theatrical production member; excluded as non-producer.
- `MGM/UA Home Entertainment` — home-video JV with UA; excluded (distribution, not production).
- `MGM/UA Television` — TV JV; excluded (TV scope and UA brand separation).
- `MGM International TV Productions` — TV; excluded.
- `MGM Alternative` — TV alt-programming arm; excluded.
- `MGM Domestic Television Distribution` — TV distribution; excluded.
- `MGM Worldwide Television / MGM Worldwide Television Productions` — TV; excluded.
- `MGM Telecommunications` — telecom/licensing shell, not a theatrical producer; excluded.
- `MGM Global Holdings` — corporate holdco, not a producer credit; excluded.
- `MGM Release` — ambiguous release/distribution label with only 1 credit; excluded as distribution rather than production (flagged — could be reconsidered if evidence shows theatrical production credits).
- `Metro Goldwyn Mayer Home Entertainment` — home-video; excluded.
- `MGM Grand` — casino/hospitality brand, unrelated; excluded.
- `MGMM STUDIOS / MGMM Reklamproduktion` — Swedish advertising production firm (MGMM), not MGM; false-positive on substring 'MGM'; excluded.
- `MGMT Entertainment / Bakrr MGMT / HAUG MGMT` — 'MGMT' abbreviation of management, unrelated; excluded.
- `AMGMP` — unclear acronym with 1 credit, no evidence of MGM connection; excluded.

**Surface-form notes:** Members were selected by matching TSV strings to the five canonical MGM production entities in the tier list (Metro-Goldwyn-Mayer, Metro-Goldwyn-Mayer Pictures, MGM Studios, Metro-Goldwyn-Mayer Cartoon Studio, Metro-Goldwyn-Mayer British Studios). The TSV's dominant credit form is the parenthetical acronym 'Metro-Goldwyn-Mayer (MGM)' (2265 credits). No standalone 'Metro-Goldwyn-Mayer' or 'Metro-Goldwyn-Mayer Pictures' string exists in the TSV — the parenthetical form serves both. Studio variants included: 'Metro-Goldwyn-Mayer (MGM) Studios' and 'Metro-Goldwyn-Mayer Studios' both map to MGM Studios (1986-). Cartoon-studio member pluralizes in the TSV as 'Metro-Goldwyn-Mayer Cartoon Studios'; 'Metro-Goldwyn-Mayer Animation' and 'MGM Animation/Visual Arts' are in-house animation successors to the Cartoon Studio, so included under that member's lineage. 'MGM British Studios' is a shortened variant of the Borehamwood entity. 'MGM-Pathé Communications Co.' (1990-1992) was the corporate parent during the Pathé Communications ownership period and produced theatrical credits under the MGM umbrella, so included. 'MGM Family Entertainment' (1990s MGM kids/family label) and 'MGM Producción' (Spanish-language MGM production label) are bona fide MGM production imprints. Excluded everything TV, home-video, holdco, casino, and the Swedish MGMM / 'MGMT' false positives. Per task rules, Amazon MGM Studios, United Artists strings, Orion strings, and MGM Television were excluded regardless of their MGM lineage.

## LIONSGATE — "Lionsgate"

**Retune status:** DROPPED `Mandate Pictures` (Juno, Stranger Than Fiction — retained its own identity post-acquisition) and `Artisan Entertainment` (Blair Witch, Requiem for a Dream — remembered as indie-Artisan, not Lionsgate). KEPT: all Lionsgate naming variants + `Summit Entertainment` (2012-) / `Summit Premiere` (2012-). The 2012 Summit gate correctly excludes pre-acquisition Summit output (Memento, Twilight, Hurt Locker) which casual viewers associate with competitor-era Summit, not Lionsgate.

| Member | Start | End | Rationale |
|---|---|---|---|
| Lionsgate | 1997 | — | Core modern-era banner |
| Lions Gate Films | 1997 | — | Original film label; still credited |
| Summit Entertainment | 2012 | — | Acquired Jan 2012 (Twilight, John Wick) |
| Mandate Pictures | 2007 | — | Acquired 2007 (Juno, Stranger Than Fiction) |
| Artisan Entertainment | 2003 | — | Acquired 2003 (Blair Witch, T2 rights) |

**Notes:** Excludes Starz (spun off 2024, TV/streaming).

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| Lionsgate | `Lionsgate` | 324 |
|  | `Lion's Gate Films` | 33 |
|  | `Lions Gate Films` | 4 |
|  | `Lions Gate Entertainment` | 21 |
|  | `Lions Gate` | 1 |
|  | `Lionsgate Premiere` | 5 |
|  | `Lionsgate Productions` | 3 |
|  | `Lions Gate Studios` | 2 |
| Lions Gate Films | `Lions Gate Films` | 4 |
|  | `Lion's Gate Films` | 33 |
| Summit Entertainment | `Summit Entertainment` | 124 |
|  | `Summit Premiere` | 1 |
| Mandate Pictures | `Mandate Pictures` | 25 |
| Artisan Entertainment | `Artisan Entertainment` | 35 |

**Collisions / exclusions flagged:**

- `'Summit' bare token and many 'Summit ...' entries in TSV are unrelated companies (Summit Films, Summit Pictures, Summit Media, Summit Productions, etc.) and must NOT be mapped to Lionsgate's Summit Entertainment. Only 'Summit Entertainment' and 'Summit Premiere' are brand members.`
- `'Artisan' bare token and entries like 'Artisan Films', 'Artisan Home Entertainment', 'Artisan Productions', 'Artisan Film', 'Artisan Television', 'Artisan Producteur' are ambiguous` — they may refer to the Lionsgate-owned Artisan Entertainment or to unrelated namesakes. Excluded out of caution; only the unambiguous 'Artisan Entertainment' is included.
- `'Roadside Attractions' is excluded from the LIONSGATE member set because Lionsgate only held a minority stake; it is not treated as a Lionsgate brand member.`
- `'Roadside Cinema', 'Roadside Entertainment', 'Roadside Pictures' are not the Howard Cohen / Eric d'Arbeloff Roadside Attractions brand` — excluded.
- `'Mandate Pictures International' (1 occurrence) is a foreign/international distribution arm; excluded per brand-registry rule that foreign distribution arms do not qualify.`
- `TV-only strings excluded per tier list (TV/streaming excluded, consistent with Starz exclusion): 'Lionsgate Television' (26), 'Lions Gate Television' (20), 'Lionsgate Alternative Television' (1).`
- `Home video distribution arms excluded (distribution-only, no production credit): 'Lions Gate Films Home Entertainment' (7), 'Lionsgate Home Entertainment' (7), 'Lions Gate Family Entertainment' (5).`
- `Foreign distribution arms excluded per rule: 'Lions Gate International' (1), 'Lionsgate India' (1).`

**Surface-form notes:** Tier list includes both one-word `Lionsgate` and two-word `Lions Gate` lineage; apostrophe variant `Lion's Gate Films` also present in TSV and included. Starz excluded per tier list (spun off 2024, TV/streaming). Roadside Attractions is intentionally NOT included because the relationship is a minority stake, not full umbrella ownership. All retained surface strings appear verbatim in /tmp/company_strings.tsv. Lions Gate Films surface strings overlap with the parent Lionsgate member because both the one-word and two-word film-label credits can be attributed to either the flagship brand or the legacy label — curation may dedupe at `brand_member_company` insertion.

## A24 — "A24"

**Retune status:** No change. Proposed ADDs `A24 Films` and `A24 Films LLC` were validated against IMDB and removed — zero evidence (IMDB uses bare `A24` on all A24 titles).

| Member | Start | End | Rationale |
|---|---|---|---|
| A24 | 2012 | — | Primary production/distribution credit |

**Notes:** "A24 Films" was the original legal name; dropped from members because zero TSV occurrences exist (IMDB credits everything as bare `A24`). Re-add only if a future ingest surfaces the string.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| A24 | `A24` | 112 |

**Collisions / exclusions flagged:**

- `A24 Television` — TV-only arm of A24 (launched ~2018). Excluded per brand spec which lists only 'A24' as a member, and per registry selection criteria that excludes TV-only divisions. Refers to the same corporate entity but is not in scope for the film-catalog brand.

**Surface-form notes:** Only two A24-token strings exist in the TSV: 'A24' (112) and 'A24 Television' (4). No occurrences of 'A24 Films', 'A24 Distribution', or 'A24 Productions' found -- on IMDB the company is credited as bare 'A24' for both production and distribution. Despite 'A24' being a weak/alphanumeric token, no unrelated-entity collisions were found in the TSV; all matches refer to the NY-based indie distributor/producer.

## NEON — "Neon"

**Retune status:** No change. Proposed ADD `Neon Rated` was validated against IMDB and removed — zero evidence (IMDB uses bare `Neon`).

| Member | Start | End | Rationale |
|---|---|---|---|
| Neon | 2017 | — | Primary credit for Tom Quinn / Tim League distributor |
| Super LTD | 2017 | — | Genre-focused sublabel |

**Notes:** Disambiguation risk — "Neon" is a common word. Unrelated entities (Neon Heart, Neon Rouge) must be excluded during token-index build.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| Neon | `Neon` | 20 |

**Collisions / exclusions flagged:**

- `Neon Productions` — Generic name; not the 2017 Tom Quinn / Tim League distributor. Multiple unrelated production houses have used this name over decades.
- `Neon Heart Productions` — Unrelated production company; explicitly called out in task prompt.
- `Neon Rouge Production` — Unrelated; explicitly called out in task prompt. Likely French/Belgian.
- `Neon Films` — Unrelated; Australian indie production company (Rolf de Heer). Distinct from Neon distributor.
- `Neon Noir` — Unrelated.
- `Neon Sheep Pictures` — Unrelated.
- `Neon Shark Media` — Unrelated.
- `Neonorma` — Unrelated; token-bearing but distinct word.
- `Neon Beam Films` — Unrelated.
- `Neon Bridge Films` — Unrelated.
- `Neon Cinema Films` — Unrelated.
- `Neon Diesel Finishing` — Unrelated; post-production/finishing house.
- `Neon District Pictures` — Unrelated.
- `Neon Harbor Entertainment` — Unrelated.
- `Neon Jane Productions` — Unrelated.
- `Neon Lights Pictures` — Unrelated.
- `Neon Mirage` — Unrelated.
- `NEON Organisation for Culture and Development` — Unrelated; Greek cultural NGO.
- `Neon Park` — Unrelated.
- `Neon Realism` — Unrelated.
- `Neon Reaper Productions` — Unrelated.
- `Neon Rooster Films` — Unrelated.
- `Neon Sine Wave Video` — Unrelated.
- `The Neon Briefcase Motion Picture Company` — Unrelated.
- `We Are Your Neon` — Unrelated; vanity production company name.
- `Sweaty Neon Nightmares` — Unrelated.
- `Neontetra` — Token-bearing substring, not a reference to the Neon brand.
- `Neonate Audio` — Audio/post house, unrelated.
- `Neonergia` — Unrelated; distinct word.
- `Alphaneon Studioz` — Substring match only; unrelated.
- `Geneon Entertainment` — Substring match only; Geneon is the Japanese anime distributor (Pioneer LDC successor), entirely distinct from Neon.
- `Geneon Universal Entertainment` — Substring match only; Japanese anime distributor, unrelated.
- `CINEON Entertainment` — Substring match only; unrelated.
- `Diamond Super Ltd.` — Matched on 'Super Ltd' search but is not Neon's Super LTD sublabel; unrelated entity.

**Surface-form notes:** Only one confirmed real surface string: 'Neon' (count=20). The low count is expected — Neon is primarily a distributor, so it appears mainly as distributor (not in production_companies) on most of its high-profile releases (Parasite, Anora). Production credits exist only on co-produced titles (Pig, Spencer, Together Together, etc.). The Super LTD genre sublabel does not appear as a distinct production-company string in the TSV; it is primarily used as a release/distribution brand rather than a production credit. The single 'Diamond Super Ltd.' match is a false positive (completely unrelated entity). The 'Neon' token collided with 30+ unrelated strings — all flagged above for curator review. No ambiguous-but-possibly-real cases beyond the exact 'Neon' string.

## BLUMHOUSE — "Blumhouse Productions"

**Retune status:** DROPPED `Blumhouse Television` ("Blumhouse movies" query intent is theatrical; TV output dilutes the roster). KEPT: `Blumhouse Productions`, `Blumhouse International`, `Atomic Monster` (gated to 2024- to correctly exclude pre-merger Conjuring/Saw — those remain under the James Wan / Atomic Monster identity).

| Member | Start | End | Rationale |
|---|---|---|---|
| Blumhouse Productions | 2000 | — | Core horror/thriller production company |
| Blumhouse Television | 2008 | — | TV arm with occasional film-adjacent output |
| Atomic Monster | 2024 | — | James Wan's company; merged into Blumhouse 2024 |

**Notes:** Universal is distribution partner, NOT owner — excluded.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| Blumhouse Productions | `Blumhouse Productions` | 153 |
|  | `Blumhouse International` | 1 |
| Blumhouse Television | `Blumhouse Television` | 46 |
| Atomic Monster | `Atomic Monster` | 25 |

**Collisions / exclusions flagged:**

- `Atomic Television` — Name similarity to 'Atomic Monster' but no evidence of Blumhouse/James Wan affiliation; excluded.
- `Blum Hanson Allen Films` — Starts with 'Blum' but unrelated to Jason Blum / Blumhouse; excluded.
- `Blum Israel Productions` — Starts with 'Blum' but unrelated to Blumhouse; excluded.

**Surface-form notes:** Universal Pictures excluded per tier list (distribution partner, not owner). No 'Blumhouse Tilt' string exists in the TSV (Tilt was a low-budget label that shut down in 2014). 'Blumhouse International' included as a brand variant (sales/production arm). No ambiguous Atomic Monster variants found — the single string 'Atomic Monster' cleanly identifies the James Wan company. Several 'Blum*' strings exist (Blum Group, Blum Family Foundation, Blump International, Blumayan, Blummer, Blumovie, etc.) but none are affiliated with Jason Blum or Blumhouse — all excluded.

## STUDIO_GHIBLI — "Studio Ghibli"

**Retune status:** No change. Proposed ADD `Studio Ghibli, Inc.` was validated against IMDB and removed — zero evidence (IMDB uses bare `Studio Ghibli` on all Ghibli films).

| Member | Start | End | Rationale |
|---|---|---|---|
| Studio Ghibli | 1985 | — | Primary credit for Ghibli films from Laputa (1986) onward |

**Notes:** Topcraft (produced Nausicaä 1984 pre-Ghibli) excluded — Nausicaä typically surfaced via director lexical match. Distribution partners (Tokuma Shoten, Nippon TV, Buena Vista, GKIDS) excluded.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| Studio Ghibli | `Studio Ghibli` | 55 |

**Surface-form notes:** Only one distinct string present in the TSV: 'Studio Ghibli' (id 1207:55). No variants found for 'Studio Ghibli K.K.' or the Japanese form 'スタジオジブリ'. Distribution/co-production partners explicitly excluded per tier list (Topcraft, Tokuma Shoten, Nippon TV, Buena Vista, GKIDS) are not included — none appeared in a 'Ghibli' grep anyway, so no collisions to flag from this search.

## NETFLIX — "Netflix"

**Retune status:** DROPPED `A Netflix Original Documentary` (marketing tagline, not a real `production_companies` entry). KEPT: all genuine production entities (`Netflix`, `Netflix Studios`, `Netflix Animation`, `Netflix Worldwide Entertainment`, `Netflix Worldwide Productions`, `Netflix India`) — casual "Netflix movies" intent is inclusive of any Netflix-produced film including regional arms.

| Member | Start | End | Rationale |
|---|---|---|---|
| Netflix | 2015 | — | Primary credit since Beasts of No Nation (2015) |
| Netflix Studios | 2016 | — | In-house production entity |
| Netflix Animation | 2018 | — | Animation studio (Klaus, The Sea Beast) |
| Netflix International Originals | 2016 | — | International originals arm |

**Notes:** Millarworld (acquired 2017) retains its own credit. Partner-produced titles distributed by Netflix (e.g., Plan B films on Netflix) are excluded from this brand. **Streamer disambiguation:** brand_id NETFLIX is emitted only for producer intent, not platform intent — see Streamer Disambiguation below.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| Netflix | `Netflix` | 461 |
| Netflix Studios | `Netflix Studios` | 104 |
| Netflix Animation | `Netflix Animation` | 28 |
| A Netflix Original Documentary | `A Netflix Original Documentary` | 5 |
| Netflix Worldwide Entertainment | `Netflix Worldwide Entertainment` | 2 |
| Netflix Worldwide Productions | `Netflix Worldwide Productions` | 2 |
| Netflix India | `Netflix India` | 1 |

**Surface-form notes:** All seven Netflix* strings in the TSV appear in the production_companies column, so by DB scope they are producer-role credits — streamer-disambiguation rule (platform vs. producer intent) is a query-time concern and does not exclude any DB string here. The tier list explicitly names 'Netflix International Originals' but no such exact string exists in the TSV; 'Netflix India' is the closest concrete international-originals production-credit variant present and is included under that spirit. No Millarworld variants appear in the TSV, so no joint Netflix/Millarworld credit handling was needed. No collisions with other brands detected.

---

# Tier 2 — Should-have, still in MVP (7 brands)

## SONY_PICTURES_ANIMATION — "Sony Pictures Animation"

**Retune status:** No change. Single-identity studio.

| Member | Start | End | Rationale |
|---|---|---|---|
| Sony Pictures Animation | 2002 | — | In-house animation studio |

**Notes:** Also member of SONY always. Sony Pictures Imageworks is primarily a VFX/services house — excluded.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| Sony Pictures Animation | `Sony Pictures Animation` | 54 |

**Collisions / exclusions flagged:**

- `Sony Pictures Imageworks (SPI)` — VFX/services house, not the animation studio; explicitly excluded per brief.
- `Sony Pictures Classics` — Arthouse/specialty distribution label (SONY_PICTURES_CLASSICS deferred from MVP). Not an animation-studio credit.
- `Sony Pictures` — Ambiguous parent/umbrella credit; belongs to SONY parent brand token path, not SONY_PICTURES_ANIMATION.
- `Sony Pictures Entertainment` — Corporate parent umbrella; belongs to SONY, not the animation studio.

**Surface-form notes:** Only one distinct surface string in the TSV unambiguously refers to Sony Pictures Animation: 'Sony Pictures Animation' (count 54). No hyphenated, abbreviated, or legacy-name variants appear in the index. Sony Pictures Imageworks is the primary collision risk (VFX/services arm that is sometimes conflated with SPA) and is explicitly excluded per the brand registry. Other 'Sony Pictures ...' strings belong to sibling Sony entities (Classics, Television, Releasing, Imageworks, Entertainment, Home Entertainment, etc.) and are out of scope for this brand.

## TRISTAR — "TriStar Pictures"

**Retune status:** DROPPED `Tri Star` (space-separated variant — IMDB data-entry artifact with negligible evidence). KEPT: `TriStar Pictures` (1982-), `Tri-Star Pictures` (1982-1991), `TriStar Productions` (2015- revival).

| Member | Start | End | Rationale |
|---|---|---|---|
| TriStar Pictures | 1982 | — | Original and still-used label |
| TriStar Productions | 2015 | — | Relaunch under Sony |
| Tri-Star Pictures | 1982 | 1991 | Hyphenated variant on early credits |

**Rename history:** TriStar Pictures → TriStar Productions (~2015 relaunch).

**Notes:** Also member of SONY since 1989.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| TriStar Pictures | `TriStar Pictures` | 131 |
|  | `Tri Star` | 1 |
| Tri-Star Pictures | `Tri-Star Pictures` | 76 |
| TriStar Productions | `TriStar Productions` | 5 |

**Collisions / exclusions flagged:**

- `TriStar Television` — TV arm; brief excludes TV credits by default (tier list explicitly excludes TV-only divisions).
- `Columbia TriStar Television` — Combined Columbia+TriStar TV credit; TV-only per scope.
- `Columbia TriStar Home Entertainment` — Home video distribution arm, not theatrical production.
- `Columbia TriStar Home Video` — Home video distribution, not theatrical production.
- `Deutsche Columbia TriStar Filmproduktion` — German co-production/distribution arm; foreign distribution excluded by brief.
- `Columbia TriStar Productions Pty. Ltd.` — Australian subsidiary — foreign distribution/production JV, not the core TriStar film label.
- `Columbia TriStar Domestic Television` — TV distribution arm.
- `Columbia TriStar International Television` — International TV distribution arm.
- `Tri Star Productions` — Name collision — this is an unrelated small production company (not Sony's TriStar). Excluded to avoid catalog contamination.
- `Bizarre Tri Star Productions` — Unrelated small production entity; clear collision on token.
- `Columbia TriStar Comercio Internacional (Madeira) Ltda.` — Portuguese distribution/finance entity; foreign distribution, excluded.
- `Columbia TriStar Entertainment` — Corporate/umbrella entity (Columbia TriStar Motion Picture Group era, 1998-2002); per COLUMBIA tier entry line 223, 'Columbia TriStar' combined credits belong under COLUMBIA, not TRISTAR.
- `Columbia TriStar Film Distributors International` — International distribution arm, not theatrical production.
- `Columbia TriStar Television Productions (UK) Ltd.` — UK TV production arm; TV-only.
- `Columbia TriStar Television Pty. Ltd.` — Australian TV arm; TV-only.
- `Gaumont/Columbia TriStar Home Video` — Home video JV; non-theatrical.
- `Tristar Products` — Hard collision — consumer-products/infomercial company unrelated to film. Excluded.

**Surface-form notes:** Three high-confidence members correspond exactly to the three tier-list members (TriStar Pictures, Tri-Star Pictures, TriStar Productions). A fourth low-count 'Tri Star' surface (n=1) is tentatively included as a TriStar Pictures variant but flagged for manual verification. 'Columbia TriStar'-prefixed surfaces are excluded here because they belong to the short-lived Columbia TriStar Motion Picture Group era (1998-2002) and are assigned to COLUMBIA per the tier list (COLUMBIA entry line 223). All TV, home video, and foreign-distribution variants are excluded per brief. Collision risks noted: 'Tri Star Productions' / 'Bizarre Tri Star Productions' / 'Tristar Products' are unrelated entities that share tokens.

## TOUCHSTONE — "Touchstone Pictures"

**Retune status:** No change.

| Member | Start | End | Rationale |
|---|---|---|---|
| Touchstone Pictures | 1986 | 2018 | Primary mature-audience Disney label; effectively dormant after 2018 per Wikidata |
| Touchstone Films | 1984 | 1986 | Original pre-rename name |

**Rename history:** Touchstone Films (1984) → Touchstone Pictures (1986).

**Notes:** Also member of DISNEY (1984-2018). Label dormant since ~2016-2018 but technically active.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| Touchstone Pictures | `Touchstone Pictures` | 230 |
| Touchstone Films | `Touchstone Films` | 7 |

**Collisions / exclusions flagged:**

- `Touchstone` — Bare 'Touchstone' credit is ambiguous — could be a short-form credit for Touchstone Pictures/Films, or an unrelated small entity. Single occurrence makes it low-value and risky to include without disambiguation via per-movie ownership check. Recommend exclude from brand_member rows; revisit only if manual inspection confirms the one movie is Disney-era Touchstone output.
- `Touchstone Pictures México` — Foreign distribution arm — excluded per brand selection criteria ('foreign distribution arms' explicitly excluded in the selection rules). Single occurrence and user intent under TOUCHSTONE is for US theatrical catalog, not Mexican distribution credit.
- `Touchstone Trust` — Unrelated entity collision — 'Trust' naming indicates a holding/financial entity, not a Disney-era production credit. No known Disney sub-brand by this name. Exclude.
- `Touchstone Television` — TV-only division, excluded per brand selection criteria ('TV-only divisions unless brand also has real film output' — TOUCHSTONE brand is scoped to film; Touchstone Television is the TV arm, renamed ABC Studios in 2007). Exclude from brand_member_company.

**Surface-form notes:** TSV grep for 'Touchstone' returned 6 distinct strings. Two are true members (Touchstone Pictures, Touchstone Films); the other four are excluded. 'Touchstone Home Entertainment' (the home-video division to exclude per instructions) does not appear in the TSV, so no explicit filter needed. The 230 vs 7 distribution between Touchstone Pictures and Touchstone Films is consistent with Touchstone Films being a brief 1984-1986 label before the 1986 rename. Also member of DISNEY (1984-2018) per cross-brand membership table. Label dormant since ~2016-2018 but technically active.

## MIRAMAX — "Miramax"

**Retune status:** DROPPED `Dimension Films` (Scream, Spy Kids, Scary Movie — horror/family-genre built its own identity distinct from Miramax's prestige indie brand; casual "Miramax movies" expects Pulp Fiction / Good Will Hunting). KEPT: `Miramax` (1979-). Proposed ADD `Miramax Films` was validated against IMDB and removed — zero evidence (IMDB uses bare `Miramax`).

| Member | Start | End | Rationale |
|---|---|---|---|
| Miramax | 1979 | — | Primary credit across all ownership eras |
| Miramax Films | 1979 | 2010 | Pre-rebrand IMDB credit form used through Disney era |
| Dimension Films | 1992 | 2005 | Genre imprint during Miramax era (pre-TWC move) |

**Rename history:** Miramax Films → Miramax (branding).

**Notes:** Also member of DISNEY (1993-2010) and member of PARAMOUNT (2020-). Post-2005 Dimension belongs to The Weinstein Company — excluded.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| Miramax | `Miramax` | 266 |
| Dimension Films | `Dimension Films` | 118 |

**Collisions / exclusions flagged:**

- `Miramax Family Films` — Ambiguous — 'Miramax Family Films' was a real Miramax sub-label (mid-1990s, e.g., Fairy Tale: A True Story) during the Disney era. Likely a legitimate Miramax member, but flagged because it is not listed explicitly in the tier doc's member table. Curator should decide whether to include.
- `Miramax International` — Ambiguous — likely the international sales/distribution arm of Miramax. Probably related, but not listed in the tier doc. Curator should decide.
- `Miramax Home Entertainment` — Home video/distribution arm of Miramax. Not a production credit in the traditional sense, but may appear as a production_companies entry. Curator should decide whether to treat as a production brand member.
- `Miramax Television` — Miramax TV production arm. Strictly speaking a television label, but tier doc does not exclude TV divisions. Curator should decide.
- `Dimension Films (II)` — IMDB disambiguation suffix indicates a DIFFERENT entity sharing the 'Dimension Films' name — not the Miramax/Weinstein imprint. Exclude.
- `Fourth Dimension Films` — Substring collision; unrelated production company.
- `Other Dimension Films` — Substring collision; unrelated production company.
- `3-Dimension Films` — Substring collision; unrelated (likely a 3D-focused boutique).
- `Epic Dimension Films` — Substring collision; unrelated production company.

**Surface-form notes:** Two confirmed member strings: 'Miramax' (266) and 'Dimension Films' (118). CRITICAL CURATOR CAVEAT: The 'Dimension Films' surface string cannot be split on name alone between the Miramax/Disney era (1992-2005, in scope) and the Weinstein Company era (2005-, out of scope). The 118-count bucket contains both. If the curator needs era-level accuracy, segmentation must happen downstream using movie release year (<=2005 -> Miramax, >2005 -> TWC/out of scope). No 'Miramax Films' surface string exists in the TSV — the historical pre-rebrand credit form appears to have been collapsed into 'Miramax' by TMDB/IMDB data normalization. Ambiguous sub-labels ('Miramax Family Films', 'Miramax International', 'Miramax Home Entertainment', 'Miramax Television') are flagged for curator decision rather than auto-included, because the tier doc's member table lists only 'Miramax' / 'Miramax Films' / 'Dimension Films'. Per tier doc, Miramax is ALSO a member of DISNEY (1993-2010) and PARAMOUNT (2020-); all three brands will share the same 'Miramax' / 'Dimension Films' surface strings — curator dedupes at brand-resolution time.

## UNITED_ARTISTS — "United Artists"

**Retune status:** DROPPED `United Artists Film Corporation` (legal-entity string, not consumer-facing) and `United Artists Europa` (regional distribution arm). Window fix: the two `United Artists` rows (1919-1981 classic + 2024- revival) were wrong — IMDB uses bare `United Artists` on MGM-era UA films too (confirmed against Rocky IV 1985, Rain Man 1988, Valkyrie 2008). Collapsed to single `United Artists` (1919, None) covering all eras; `United Artists Pictures` (1981, 2018) retained as additional surface coverage.

| Member | Start | End | Rationale |
|---|---|---|---|
| United Artists | 1919 | 1981 | Classic UA through Transamerica era |
| United Artists Pictures | 1981 | 2018 | MGM-era credit variant during largely dormant period |
| United Artists Releasing | 2018 | 2023 | MGM + Annapurna JV for distribution |
| United Artists | 2024 | — | Revived under Amazon MGM with Scott Stuber |

**Rename history:** United Artists (1919) → United Artists Releasing (2018) → United Artists revival (2024).

**Notes:** Also member of AMAZON_MGM since 2024 revival.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| United Artists | `United Artists` | 123 |
| United Artists Pictures | `United Artists Pictures` | 1 |
| United Artists Film Corporation | `United Artists Film Corporation` | 1 |
| United Artists Europa | `United Artists Europa` | 1 |

**Collisions / exclusions flagged:**

- `United Artists Television` — TV arm of UA. Brand scope is film-only per selection criteria; TV-only divisions are excluded.
- `United Artists Theaters` — Exhibition (movie theater) chain, not a production or distribution entity. Not part of the film studio production lineage.
- `Madras United Artists Corporation` — Unrelated Indian (Madras/Chennai) regional production company; coincidental token overlap with 'United Artists'.

**Surface-form notes:** Only 'United Artists' (123) has meaningful volume; the other three member surfaces each occur once and reflect historical corporate/regional credit variants of the same studio lineage. Expected member 'United Artists Releasing' (2018-2023 MGM+Annapurna JV) is NOT present in the TSV as a distinct string — its films likely credit 'United Artists' or co-production partners instead. Excluded 'United Artists Television' (TV-only per brand scope), 'United Artists Theaters' (exhibitor, not studio), and 'Madras United Artists Corporation' (unrelated Indian company).

## AMAZON_MGM — "Amazon MGM Studios"

**Retune status:** DROPPED `Amazon Studios Germany` (regional TV/local-language arm), `Metro-Goldwyn-Mayer Animation` / `MGM Animation/Visual Arts` / `MGM Family Entertainment` / `MGM British Studios` / `MGM Producción` (legacy sublabels not in Amazon-era casual viewer mental model), `United Artists Film Corporation` / `United Artists Europa` (mirrors UNITED_ARTISTS drops). KEPT: `Amazon Studios`, `Amazon MGM Studios`, `Metro-Goldwyn-Mayer (MGM)` + two Studios variants (2022-), `United Artists` + `United Artists Pictures` (2024-).

| Member | Start | End | Rationale |
|---|---|---|---|
| Amazon Studios | 2010 | 2023 | Original name through October 2023 rename |
| Amazon MGM Studios | 2023 | — | Current umbrella credit post-rename |
| Metro-Goldwyn-Mayer | 2022 | — | Post-2022 MGM acquisition folds MGM films under Amazon umbrella |
| United Artists | 2024 | — | Post-2024 revival as Amazon MGM sub-label |
| Amazon Content Services | 2010 | — | Legal/production-services entity frequently co-credited |

**Rename history:** Amazon Studios (2010) → Amazon MGM Studios (October 2023).

**Notes:** Pre-2022 MGM and pre-2024 UA catalogs belong to the standalone MGM and UNITED_ARTISTS brands, not here.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| Amazon Studios | `Amazon Studios` | 154 |
| Amazon MGM Studios | `Amazon MGM Studios` | 95 |
| Amazon Studios Germany | `Amazon Studios Germany` | 1 |

**Collisions / exclusions flagged:**

- `Metro-Goldwyn-Mayer (MGM)` — Classic MGM catalog string. Per tier list, pre-2022 MGM belongs to standalone MGM brand. Post-2022 Amazon-era MGM films use this same string; no way to disambiguate from the string alone. Flagged here for curator attention (matches tier list open-curation item #5).
- `Metro-Goldwyn-Mayer (MGM) Studios` — Same disambiguation problem as 'Metro-Goldwyn-Mayer (MGM)'. Rare variant.
- `Metro-Goldwyn-Mayer Studios` — Same ambiguity between MGM standalone brand and post-2022 Amazon-era MGM.
- `MGMM STUDIOS` — All-caps variant; ambiguous — could be MGM-era or post-2022 Amazon-era. Also possible unrelated entity (MGMM is not a canonical MGM spelling). Curator should inspect actual credits before attribution.
- `United Artists` — Pre-2024 UA belongs to standalone UNITED_ARTISTS brand; post-2024 revival is Amazon MGM sub-label. Cannot distinguish eras from surface string alone. Flagged for curator; also applies to 'United Artists Pictures' (1), 'United Artists Film Corporation' (1), 'United Artists Europa' (1), 'United Artists Theaters' (1).
- `Metro-Goldwyn-Mayer British Studios` — UK Borehamwood production arm 1936-1970 — classic MGM era, belongs to standalone MGM brand. Listed here only so curator confirms it is NOT attributed to AMAZON_MGM.
- `MGM British Studios` — Same as 'Metro-Goldwyn-Mayer British Studios' — classic-era MGM UK arm, not Amazon-era.
- `Metro-Goldwyn-Mayer Cartoon Studios` — 1937-1957 animation arm; classic MGM, belongs to standalone MGM.
- `Metro-Goldwyn-Mayer Animation` — Classic-era MGM animation credit; standalone MGM brand.
- `MGM Animation/Visual Arts` — Classic-era MGM animation; standalone MGM.

**Surface-form notes:** Confident direct Amazon-owned producer credits in the IMDB surface strings are only three: 'Amazon Studios', 'Amazon MGM Studios', and the rare 'Amazon Studios Germany'. 'Amazon Content Services' (listed as a member in the tier spec) does NOT appear anywhere in the TSV, so no surface-string row is produced for it. Because the task is Workstream B surface-form enumeration for production credits (producer intent only), the streamer 'Amazon Prime Video' is deliberately excluded per the tier-list streamer-disambiguation rule. The large open curation question (tier list item #5) is the MGM and United Artists strings: the SAME surface strings 'Metro-Goldwyn-Mayer (MGM)', 'United Artists', etc. are used both pre- and post-Amazon-acquisition. Per the tier-list instruction to exclude pre-2022 MGM and pre-2024 UA, and the near-impossibility of distinguishing eras from the string alone, these strings are NOT emitted as AMAZON_MGM members — they are surfaced under collisions_flagged so curation can decide whether to (a) attribute post-acquisition credits via release-year cross-checks against movie metadata, or (b) leave all classic-string MGM/UA credits under the standalone MGM and UNITED_ARTISTS brands and rely on 'Amazon MGM Studios' as the single reliable Amazon-era umbrella credit.

## APPLE_STUDIOS — "Apple Studios"

**Retune status:** No change.

| Member | Start | End | Rationale |
|---|---|---|---|
| Apple Original Films | 2019 | — | Apple's film production and branding label |
| Apple Studios | 2019 | — | In-house studio entity, often co-credited with Apple Original Films |

**Notes:** **Streamer disambiguation:** brand_id APPLE_STUDIOS is emitted only for producer intent, not "on Apple TV+" platform intent.

### Surface forms

| Member | Surface string | Count |
|---|---|---|
| Apple Original Films | `Apple Original Films` | 34 |
| Apple Studios | `Apple Studios` | 18 |

**Collisions / exclusions flagged:**

- `Apple TV+` — Streaming platform name, not a production entity. Per tier-list streamer disambiguation, APPLE_TV+ platform intent is handled by watch_providers, not the APPLE_STUDIOS brand. IMDB credits using this string appear to be imprecise platform-as-producer credits; excluded to keep the brand clean, but flagged because a small number of real Apple-produced films may carry this string instead of 'Apple Studios'/'Apple Original Films'.
- `Apple TV` — Same as above — platform brand, not producer. Exclude from APPLE_STUDIOS.
- `Apple Music` — Apple's music streaming platform, not a film production entity. Exclude.
- `Apple Park Films` — Name references Apple's HQ campus ('Apple Park'), suggesting possible Apple Inc. in-house label, but not listed in tier spec and could equally be an unrelated namesake. Flagged rather than included — needs manual verification before claiming as an APPLE_STUDIOS member.
- `Apple Corps` — Explicit exclusion per prompt: The Beatles' company, unrelated to Apple Inc.
- `Apple Films` — Explicit exclusion per prompt: 1970s UK indie (David Puttnam's production company), unrelated to Apple Inc.
- `Apple Film Productions` — Variant of the 1970s UK indie Apple Films (David Puttnam). Unrelated to Apple Inc. — excluded by the same reasoning as 'Apple Films'.
- `Apple Film Production` — Singular variant of the same 1970s UK indie. Unrelated to Apple Inc. — excluded.

**Surface-form notes:** Apple Studios / Apple Original Films are a young label (2019-), so canonical credits are limited to two clean strings with a combined ~52 credits in the corpus. The main curation risk is 'Apple TV+' (6 credits) and 'Apple TV' (1) appearing as production credits — these are the streaming platform name and are explicitly excluded by the Streamer Disambiguation rule in the tier spec (APPLE_STUDIOS brand_id applies only to producer intent). 'Apple Park Films' (4) looks Apple-Inc.-adjacent given the HQ-campus name but is not in the tier spec and needs human verification. Excluded all Apple Inc.-unrelated namesakes: 'Apple Corps' (Beatles), 'Apple Films'/'Apple Film Productions'/'Apple Film Production' (1970s UK David Puttnam indie), and all unrelated 'X Apple' / 'Apple X' small-indie strings (Green Apple, Big Apple, Red Apple, Poison Apple, Applecreek, Appleseed, etc. — none of these refer to Apple Inc.).

---

# Cross-Brand Membership Summary

Sub-brands that are BOTH their own brand AND a time-bounded member of a parent umbrella. (This produces the `brand_member_company` rows that power umbrella queries.)

| Sub-brand | Member of | Start | End |
|---|---|---|---|
| WALT_DISNEY_ANIMATION | DISNEY | — | — |
| PIXAR | DISNEY | 2006 | — |
| MARVEL_STUDIOS | DISNEY | 2009 | — |
| LUCASFILM | DISNEY | 2012 | — |
| TWENTIETH_CENTURY | DISNEY | 2019 | — |
| SEARCHLIGHT | DISNEY | 2019 | — |
| TOUCHSTONE | DISNEY | 1984 | 2018 |
| MIRAMAX | DISNEY | 1993 | 2010 |
| DC | WARNER_BROS | — | — |
| NEW_LINE_CINEMA | WARNER_BROS | 1996 | — |
| COLUMBIA | SONY | 1989 | — |
| TRISTAR | SONY | 1989 | — |
| SONY_PICTURES_ANIMATION | SONY | 2002 | — |
| FOCUS_FEATURES | UNIVERSAL | 2002 | — |
| ILLUMINATION | UNIVERSAL | 2007 | — |
| DREAMWORKS_ANIMATION | UNIVERSAL | 2016 | — |
| MGM | AMAZON_MGM | 2022 | — |
| UNITED_ARTISTS | AMAZON_MGM | 2024 | — |

---

# Excluded on purpose

These do not get brand registry rows.

### Streamers as platforms (not as producers)

`NETFLIX-as-platform`, `APPLE_TV+`, `PRIME_VIDEO`, `HULU`, `MAX`, `PEACOCK`, `DISNEY+` — "movies on X" handled by the `watch_providers` metadata path, not the studio resolver. The streamer-producer brands NETFLIX, AMAZON_MGM, APPLE_STUDIOS cover the narrower "produced by X" intent only.

### Pure financiers / coproducers with no vanity-card audience

`Syncopy`, `Village Roadshow`, `Regency`, `Participant`, `TSG`, `IAC Films`, `Perfect World`, `Entertainment One`.

### Deep-cult back-catalog brands

`Cannon Films`, `Carolco`, `Troma`, `Hammer`, `Janus Films`, `IFC Films`, `GKIDS`, `RKO`.

### Deferred from MVP — add back if query logs justify

Brand rows evaluated in WS-A/WS-B research but cut from the MVP because their main credit string has a distinctive token (freeform token match covers them losslessly), near-zero DB mass, or a niche enthusiast-only search profile:

- **Tier-2 cuts (7):** `SONY_PICTURES_CLASSICS`, `SCREEN_GEMS`, `HBO_FILMS`, `LAIKA`, `AARDMAN`, `WORKING_TITLE`, `AMBLIN`.
- **Tier-3 cuts (10):** `LEGENDARY`, `ANNAPURNA`, `SKYDANCE`, `BAD_ROBOT`, `PLAN_B`, `TOHO`, `STUDIOCANAL`, `CJ_ENTERTAINMENT`, `ORION`, `CASTLE_ROCK`.

Their `production_company` strings still get ingested and remain reachable via the freeform token-match path; only the closed-enum `BrandEnum` row and the `brand_member_company` umbrella edges are deferred.

---

# Streamer Disambiguation

Netflix, Amazon MGM, and Apple Studios are the only brands where the name also doubles as a streaming platform.

- **Producer intent** ("Netflix originals", "movies made by Netflix", "Apple original films", "Prime Video originals") → brand_id = the relevant brand.
- **Platform intent** ("movies on Netflix", "streaming on Apple TV+", "what's on Prime") → brand_id must be null; the `watch_providers` metadata path handles it.

The `thinking` field in `StudioQuerySpec` is where the LLM explicitly reasons about this distinction before committing.

---

# Open Curation Choices

1. **DC Comics as a pre-2009 production credit.** Some pre-2009 Warner DC films credit `DC Comics` (the publisher) rather than a DC-branded production entity. Confirm via DB scan that enough pre-2009 DC films credit `DC Comics` to justify the member row.
2. **Amazon Studios vs Amazon MGM Studios overlap.** Amazon Studios (pre-2023) is a member of AMAZON_MGM; Amazon MGM Studios (post-2023) is also a member. MGM and United Artists strings should stay year-gated at stamp time rather than resolved from raw string alone.

*Closed by WS-B verification:* Wikidata date validation (all 31 member assertions checked, 2 dates corrected inline — DisneyToon Studios 2003→1988 start, Touchstone Pictures null→2018 end); StudioCanal/Canal+ resolved by dropping STUDIOCANAL from MVP (deferred section); HBO Max Films resolved by dropping HBO_FILMS from MVP.

---

# Summary

- Tier 1: 24 brands — the obvious majors (Disney family, Warner family, Universal, Paramount, Sony, Fox-successor Disney labels, major indies A24/Neon/Blumhouse/Ghibli, Netflix).
- Tier 2: 7 brands — sub-labels and streamer-producers with clear MVP search intent (Sony Pictures Animation, TriStar, Touchstone, Miramax, United Artists, Amazon MGM, Apple Studios).
- Total MVP brands: 31.
- Total `brand_member_company` rows (approximate): ~85 direct-member rows + 18 cross-brand umbrella membership rows.
- 17 additional brands deferred from MVP (see "Excluded on purpose → Deferred from MVP"); their `production_company` strings still ingest and remain reachable via the freeform token-match path.

Workstream B (DB surface-form enumeration) was executed via 48 parallel
per-brand agents against a pre-built index of every distinct IMDB
`production_companies` string with tag counts (~182k distinct strings
across ~362k movies). Each kept brand section above carries a
`### Surface forms` subsection listing verbatim DB strings, counts, and
flagged collisions.

Validation performed on the kept 31:
- **DB evidence floor** — every member's surface strings checked against the index; 3 zero-evidence members dropped (PIXAR/Pixar Canada, WARNER_BROS/Warner Animation Group, A24/A24 Films).
- **Wikidata date validation** — all 31 date-bearing member assertions checked; 29 OK, 2 corrected inline (DisneyToon Studios start 2003→1988; Touchstone Pictures end null→2018).

Next steps:
1. **Curator pass** — resolve the 2 remaining open curation choices and the per-brand flagged collisions before `brand_member_company` seeding.
2. **Schema + ingest** — build the `brand`, `production_company`, and `brand_member_company` tables from this file and stamp `movie.brand_ids` during ingestion.
