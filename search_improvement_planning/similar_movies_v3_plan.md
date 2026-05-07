# Similar Movies — V3 Improvements

Consolidated plan after end-to-end V2 testing ([similar_movies_v2_results.md](similar_movies_v2_results.md)) and a follow-up audit of every V2 grouping/registry. Items below pull from V2 test failures, registry audits, and direct user findings.

**Status note**: Section 1.1 (format registry) and 1.2 (production medium registry) **already implemented** ahead of the rest of the V3 work. See [search_v2/format_registry.py](../search_v2/format_registry.py) and [search_v2/production_medium_registry.py](../search_v2/production_medium_registry.py). Verified Oppenheimer single-anchor: Fat Man and Little Boy now #1 (was buried at #5); top 10 is mostly narrative biopics + Nolan-director adjacents instead of literal Manhattan-Project documentaries.

Sections:
1. [Categorization fixes (audited registries)](#1-categorization-fixes-audited-registries)
2. [Lane-level reworks](#2-lane-level-reworks)
3. [Weaving and multipliers](#3-weaving-and-multipliers)
4. [Quality, fallback, and miscellaneous](#4-quality-fallback-and-miscellaneous)
5. [Implementation order / priority](#5-implementation-order--priority)

---

## 1. Categorization fixes (audited registries)

I audited every grouping introduced by V2 — format, production medium, country/language, studio, and award taxonomy — by sampling the catalog. Format and production medium both have category-shape errors that mirror the DOCUDRAMA bug; country/language has scope gaps; studio is mostly fine; award taxonomy needs minor tuning.

### 1.1 Format registry — remove DOCUDRAMA, TRUE_CRIME, and SKETCH_COMEDY ✅ DONE

**Findings**: I queried the catalog for prestige (≥80 reception) movies tagged with each format-bucket member tag. Three were misclassified as format signals when they are actually content/style tags:

`DOCUDRAMA`-tagged narrative features: Schindler's List, GoodFellas, 12 Years a Slave, Battle of Algiers, Battleship Potemkin, Social Network, Spotlight, Platoon, Irishman, Diving Bell and the Butterfly, Oppenheimer, Zero Dark Thirty, Raging Bull, Pianist, King's Speech, United 93, Bloody Sunday, Straight Story, The Queen, Patton.

`TRUE_CRIME`-tagged narrative features: GoodFellas, Spotlight, Close-Up, Irishman, French Connection, Badlands, In Cold Blood, Dog Day Afternoon, Killers of the Flower Moon, American Hustle, In the Name of the Father, Bonnie and Clyde, Capote, Memories of Murder, Boys Don't Cry, Lilya 4-ever, Fruitvale Station, Serpico.

`SKETCH_COMEDY`-tagged narrative features (found during deeper audit): Monty Python and the Holy Grail, Monty Python's The Meaning of Life, And Now for Something Completely Different, The Sunshine Boys, The Best of Benny Hill — all 70+ minute features tagged with SKETCH_COMEDY because they contain sketch-comedy elements. The previous registry put them in the `tv_format` bucket, separating them from other narrative comedies.

Of 30 prestige DOCUDRAMA samples, **27 had no actual DOCUMENTARY tag**. Of 30 prestige TRUE_CRIME samples, **28 were narrative crime dramas**. Of 13 prestige SKETCH_COMEDY samples, **all 13 were narrative features**.

The deeper audit also verified: SHORT (1 mistag in prestige tier — minor), MOCKUMENTARY (clean), NEWS (clean), CONCERT/STAND_UP (clean), the remaining TV_format tags (clean). DOCUDRAMA / TRUE_CRIME / SKETCH_COMEDY were the only meaningful misclassifications.

**Done**: removed all three from their respective buckets in [search_v2/format_registry.py](../search_v2/format_registry.py). They now feed the themes lane via the keyword pool (the exclusion at `_themes_traits_for_movie` strips only `FORMAT_KEYWORD_IDS_ALL`, which automatically shrunk when they were removed from their groups).

**Verified V3 effect** on Oppenheimer single-anchor: previously 4 of top 5 were literal documentaries; now top 5 is `Fat Man and Little Boy` (1989 narrative), `Before Night Falls` (DOCUDRAMA biopic now correctly bucketed narrative), `Day One` (1989 narrative), `The Birth of a Nation`, `The Dark Knight` (Nolan director_signature). Documentaries are pushed down to #7 and #9.

### 1.2 Production medium registry — remove `ADULT_ANIMATION` and `HOLIDAY_ANIMATION` ✅ DONE

**Findings**: I sampled prestige `ADULT_ANIMATION` films and prestige `HOLIDAY_ANIMATION` films. Both tags are **audience/tone signals, not techniques**:

`ADULT_ANIMATION` co-occurs with the actual technique tag in 17/20 prestige cases: Persepolis (HD), Waltz with Bashir (HD), Flee (HD), Grave of the Fireflies (HD/anime), Mahavatar Narsimha (CG), Sita Sings the Blues (none), Cat City (none).

`HOLIDAY_ANIMATION` is purely thematic (Christmas-themed), co-occurs with HD (Charlie Brown Christmas, Grinch), stop-motion (Nightmare Before Christmas, Rudolph), and CG (Dragons: Gift of the Night Fury) across the sample.

`ANIME` is a stylistic origin signal: 928 films total; 828 (89 %) co-occur with `HAND_DRAWN_ANIMATION`. ANIME is essentially "Japanese hand-drawn animation tradition" — kept as a medium tag because it captures a meaningful viewing-experience distinction.

The previous 8×8 matrix collapsed two orthogonal axes (technique × audience) into one, producing wrong scores like Persepolis vs Sita Sings the Blues at 1.0 (ADULT vs ADULT) when their techniques are incompatible.

**Done**: dropped to a 6×6 technique-only matrix in [search_v2/production_medium_registry.py](../search_v2/production_medium_registry.py). The IDFs reload automatically on next process start; existing values for the removed tags become unused.

**Note**: ADULT_ANIMATION and HOLIDAY_ANIMATION are now naturally available to the themes lane as content tags (they were never in `FORMAT_KEYWORD_IDS_ALL`, so they're already in the themes pool when single-anchor themes is added per §2.3).

### 1.3 Country/language registry — extend to single-anchor with calibrated penalty

**Findings**:
- Current V2 multi-anchor only: see [search_v2/similar_movies.py:1841–1848](../search_v2/similar_movies.py#L1841).
- Verified Barbie single-anchor case: anchor has no country tag → `US_DEFAULT`. Result #1 `Swag (2024)` is `TELUGU`; result #2 `I Am Not an Easy Man` is `FRENCH`. Neither matches anchor's tradition.

User direction: "*not a fully exclusion but enough to have it consistently drop out of the first few spots (or make the bar higher for a movie from a particular country to show results from another one)*"

**Calibration math** for the Barbie case:
- Swag (Telugu) base score ≈ 0.70 (currently #1).
- Next-best on-tradition US film (Poor Things at #5) base score ≈ 0.56.
- For Swag to drop *below* Poor Things consistently, we need `0.70 * penalty < 0.56` → `penalty < 0.80`.
- Current V2 mismatch penalty `0.85` is too soft (0.70 × 0.85 = 0.595 — Swag still wins).
- Proposed `0.75` → 0.70 × 0.75 = 0.525, drops below Poor Things.
- Proposed match boost: `1.05` (light, so a foreign anchor with cross-tradition candidates doesn't get over-amplified). Keeps the "make the bar higher" framing without over-rewarding match.

**Action**:
- Extend the country/language coherence multiplier to single-anchor. Anchor's own `country_set(keyword_ids)` becomes the consensus (instead of cross-anchor consensus). Same multiplier helper, same code path — just enable for `n == 1`.
- Tune the multipliers in [search_v2/similar_movies.py:217–218](../search_v2/similar_movies.py#L217):
  - Match boost: `1.10 → 1.05` (single-anchor and multi-anchor both — over-rewarding match was creating its own noise).
  - Mismatch penalty: `0.85 → 0.75` (single-anchor and multi-anchor both — calibrated to push cross-tradition results out of the top 3 unless they're *significantly* better matches).
- Edge case: anchor with multiple country tags (French-Italian co-production) — keep co-production support unchanged. Match if candidate has any of the anchor's tradition tags.
- The default `US_DEFAULT` bucket lumps US/UK/Canada/Australia together, which is correct: Barbie shouldn't penalize a Christopher Nolan UK production.

Catalog distribution: HINDI=3044, KOREAN=1061, SPANISH=2874, FRENCH=4440, JAPANESE=2878, MANDARIN=1378 out of ~109 K movies. Most of the catalog falls into US_DEFAULT, so the penalty effectively "let cross-tradition films in only when they decisively beat in-tradition adjacents."

### 1.4 Studio registry — mostly fine; verify Lucasfilm era windows

**Findings**: The curated list is reasonable. Era windows are subjective but coherent. The eras for Lucasfilm: 1977–1989, 1999–2012, 2015–present — this misses the gap years 1990–1998 and 2013–2014, which is intentional (no major Lucasfilm output during those gaps). Verified Star Wars 1977 maps to era window 1977–1989; Force Awakens 2015 maps to 2015-present.

**Action** (low priority): no changes needed. If V3 expands the curated list (e.g. add Wes Anderson's American Empirical Pictures, Coen Brothers' Mike Zoss Productions), do it as a one-line addition and rebuild the normalized-string lookup.

### 1.5 Award taxonomy — minor SPECIFICITY_FACTOR tuning

**Findings**: [search_v2/award_taxonomy.py:37](../search_v2/award_taxonomy.py#L37) uses `{0: 1.0, 1: 0.6, 2: 0.3}`. The L2=0.3 means any anchor set sharing only at the L2 group level (e.g., all 3 anchors won *some* PICTURE-bucket award — Best Picture or Best Animated Feature or Best Foreign Language) still produces meaningful cohesion. In the multi-anchor War film trio, anchors share L1/L2 PICTURE awards → moderate specific_award lane firing, ranking well-awarded war films higher (Dunkirk, All Quiet on the Western Front, etc.). That's the intended behavior.

**Concern**: with L2 = 7 broad buckets (ACTING, DIRECTING, WRITING, PICTURE, CRAFT, RAZZIE, FESTIVAL_OR_TRIBUTE), the L2 fallback can over-trigger. A 3-anchor set where each won *any* award at all will repeat at L2. With ratio = 1.0, cohesion = 0.3 × 2.0 = 0.6 — meaningful but not dominant. If we observe over-firing in V3 testing, lower L2 to `0.2`.

**Action** (low priority, observe first): keep V2 values; revisit if V3 testing shows over-firing.

---

## 2. Lane-level reworks

### 2.1 Director lane — manual auteur list only; drop the non-curated tier entirely

User direction: "*could they be dropped entirely? If those popular but not worth being in that upper tier directors should only really match movies in the same franchise, then shouldn't franchise already cover this case?*"

**Yes — agreed.** Walking through the cases:
- Spielberg's *Indy* for an *Indy* anchor: same lineage → franchise lane covers.
- Spielberg's *E.T.* for an *Indy* anchor: different franchise, different style → director isn't the right signal here either (vector adjacency or themes might pick it up if genuinely similar; if not, it shouldn't surface).
- Lucas's *American Graffiti* for *Star Wars*: completely different style → director surfacing this was a bug, not a feature.

So the only cases where director matters BEYOND what other lanes already cover are **truly auteur** directors whose films share an unmistakable identity beyond shared world or shared genre — Tarantino-style sharp dialogue and structural play, Wes Anderson-style visual symmetry, Lynch-style oneiric tone, Miyazaki-style hand-drawn pastoral. For those, director is doing real work; for everyone else, the franchise lane and shape vectors already cover.

**Findings (V2 director lane is broken on multiple fronts)**:

1. **George Lucas case**: Star Wars single-anchor surfaces `American Graffiti (1973)` at #9 via director_signature. American Graffiti is stylistically nothing like Star Wars (1973 coming-of-age drama vs 1977 space opera). The director_strength MV scores Lucas highly because *Star Wars* drags his percentile up, but Lucas's non-Star-Wars filmography doesn't share his "unique style" — he's a Star-Wars-creator more than an auteur.

2. **Frank Darabont case**: The Green Mile single-anchor lists Shawshank (Darabont, same director, same novelist, same prison-drama genre) at only #6 with score 0.392, behind random shape adjacents like *The Whale* and *Last Light*. Darabont's strength MV value is presumably modest (small filmography), so the director boost is small.

3. **David Fincher case**: Fight Club single-anchor shows Fincher's films (The Game #2, Girl with Dragon Tattoo #7) but they're outranked by obscure shape matches (Luster, Revolver). Fincher *is* a high-strength auteur but the +0.10 director-lane delta isn't enough to push his identity-defining films past centroid drift.

4. **Spielberg, Coppola in Godfather case**: The Godfather single-anchor surfaces The Conversation (Coppola) at #10 — fine — but also Lawrence of Arabia (Lean) at #8. Director isn't even involved there; that's the quality lane firing.

The throughline: `director_strength = 0.8*pop + 0.2*reception` measures **director popularity**, not **stylistic coherence across films**. For directors whose filmography is genuinely stylistically uniform (Tarantino, Wes Anderson, Lynch, Miyazaki, Coen Bros), directorship is a near-perfect similarity signal. For prolific generalists (Spielberg, Lucas), it's misleading.

**Action**:
- Maintain a **manual auteur list** of style-coherent directors. Lane fires under EITHER of:
  - **(a)** any anchor's director ∈ auteur list AND candidate shares that director, OR
  - **(b)** multi-anchor only — ≥2 anchors share a director (auteur or not) AND candidate shares that director.
- Stored as a Python frozenset of `lex.lexical_dictionary.norm_str` values keyed in the form actually present in the catalog. Initial composition is **60 entries**, verified against `lex.lexical_dictionary` and `lex.inv_director_postings` (table below). Flat boolean tier; easy to extend.
- **Retire `mv_director_strength`** as a similar-movies lane signal. The auteur list replaces it. The MV may still be useful elsewhere (e.g., the director endpoint in the standard V2 search pipeline) — out of scope to rip out, but stop reading it from `similar_movies.py`.

**Scoring (unified single + multi-anchor)**:

For each director `d` shared between candidate and at least one anchor, with `M_d` = number of anchors that share `d` and `N` = total active anchors:

```
if N == 1:                          contribution_d = 0.20    # auteur required for lane to fire
elif d ∈ auteur_list:               contribution_d = 0.20 + 0.10 * (M_d / N)   # caps at 0.30
else (cohesion-only, M_d ≥ 2):      contribution_d = 0.10 * (M_d / N)          # caps at 0.10
```

Take `max(contribution_d)` across all shared directors as the candidate's director-lane score.

**Multi-director boost — both boosted independently.** When the anchor set splits between multiple curated directors (e.g., 4 anchors evenly split between Wes Anderson and PTA), candidates matching each director get scored separately under their own `M_d`. A WA candidate computes via `M_WA=2, N=4 → 0.25`; a PTA candidate computes via `M_PTA=2, N=4 → 0.25`. Both boost. The lane never asks "which director wins for this anchor set" — it asks "how strong is the director signal for this candidate."

**Floor (multi-anchor only, ignores auteur status)**:
```
if max_d (M_d / N) >= 0.75 AND shape_score >= 0.30:
    combined_score = max(combined_score, 0.35)
```

Near-unanimous director cohesion overrides shape regardless of curation — if 3-of-4 anchors share director X, the user has signaled director matters even if X isn't on the list. The 0.75 threshold activates at M/N values of 2-of-2, 3-of-3, 3-of-4, 4-of-4, 4-of-5, 5-of-5 (the 2-of-3 case at 0.667 stays additive-only).

**Curated auteur list — 60 entries.** Each row is the normalized key (the form actually present in `lex.lexical_dictionary` after `normalize_string`); film count is the count of director-postings in the live catalog at finalization (2026-05-07). The selection bar applied: (i) recognizable style across ≥3 features, (ii) general moviegoer awareness, (iii) other films by the director are likely to "feel the same" to a viewer who liked one — the third criterion is the actual recommender bar; "famous + good craft" alone is not enough.

| Group | Display name | Normalized key (DB form) | Catalog films |
|---|---|---|---:|
| Visual stylists | Wes Anderson | `wes anderson` | 19 |
| Visual stylists | Stanley Kubrick | `stanley kubrick` | 16 |
| Visual stylists | Terrence Malick | `terrence malick` | 11 |
| Visual stylists | David Fincher | `david fincher` | 12 |
| Visual stylists | Tim Burton | `tim burton` | 23 |
| Visual stylists | Guillermo del Toro | `guillermo del toro` | 15 |
| Visual stylists | Yorgos Lanthimos | `yorgos lanthimos` | 11 |
| Visual stylists | Sofia Coppola | `sofia coppola` | 9 |
| Visual stylists | Nicolas Winding Refn | `nicolas winding refn` | 10 |
| Visual stylists | Denis Villeneuve | `denis villeneuve` | 13 |
| Visual stylists | Edgar Wright | `edgar wright` | 9 |
| Visual stylists | Wong Kar-wai | `wong kar-wai` | 15 |
| Visual stylists | Pedro Almodóvar | `pedro almodovar` | 25 |
| Visual stylists | Jean-Pierre Jeunet | `jean-pierre jeunet` | 11 |
| Visual stylists | Baz Luhrmann | `baz luhrmann` | 7 |
| Visual stylists | Zack Snyder | `zack snyder` | 14 |
| Tonal / structural | David Lynch | `david lynch` | 23 |
| Tonal / structural | Christopher Nolan | `christopher nolan` | 14 |
| Tonal / structural | Darren Aronofsky | `darren aronofsky` | 10 |
| Tonal / structural | Charlie Kaufman | `charlie kaufman` | 4 |
| Tonal / structural | Spike Jonze | `spike jonze` | 9 |
| Tonal / structural | M. Night Shyamalan | `m night shyamalan` | 16 |
| Tonal / structural | Ari Aster | `ari aster` | 6 |
| Tonal / structural | Robert Eggers | `robert eggers` | 4 |
| Tonal / structural | Jordan Peele | `jordan peele` | 3 |
| Tonal / structural | Bong Joon-ho | `bong joon ho` | 9 |
| Tonal / structural | Park Chan-wook | `park chan-wook` | 16 |
| Voice-driven | Quentin Tarantino | `quentin tarantino` | 15 |
| Voice-driven | Aaron Sorkin | `aaron sorkin` | 3 |
| Voice-driven | Richard Linklater | `richard linklater` | 24 |
| Voice-driven | Noah Baumbach | `noah baumbach` | 14 |
| Voice-driven | Joel Coen | `joel coen` | 21 |
| Voice-driven | Ethan Coen | `ethan coen` | 23 |
| Career-spanning masters | Martin Scorsese | `martin scorsese` | 41 |
| Career-spanning masters | Paul Thomas Anderson | `paul thomas anderson` | 14 |
| Career-spanning masters | Spike Lee | `spike lee` | 36 |
| Career-spanning masters | David Cronenberg | `david cronenberg` | 27 |
| Career-spanning masters | Michael Mann | `michael mann` | 14 |
| Career-spanning masters | Brian De Palma | `brian de palma` | 30 |
| Career-spanning masters | John Carpenter | `john carpenter` | 21 |
| Career-spanning masters | Sam Raimi | `sam raimi` | 17 |
| Career-spanning masters | Hayao Miyazaki | `hayao miyazaki` | 14 |
| International auteurs | Akira Kurosawa | `akira kurosawa` | 31 |
| International auteurs | Federico Fellini | `federico fellini` | 24 |
| International auteurs | Ingmar Bergman | `ingmar bergman` | 47 |
| International auteurs | Andrei Tarkovsky | `andrei tarkovsky` | 10 |
| International auteurs | Werner Herzog | `werner herzog` | 59 |
| International auteurs | Hirokazu Kore-eda | `hirokazu koreeda` | 16 |
| International auteurs | Lars von Trier | `lars von trier` | 18 |
| International auteurs | Michael Haneke | `michael haneke` | 16 |
| International auteurs | Luca Guadagnino | `luca guadagnino` | 11 |
| International auteurs | Céline Sciamma | `celine sciamma` | 5 |
| International auteurs | Jane Campion | `jane campion` | 13 |
| Modern voices | Damien Chazelle | `damien chazelle` | 6 |
| Modern voices | Barry Jenkins | `barry jenkins` | 4 |
| Modern voices | Greta Gerwig | `greta gerwig` | 4 |
| Modern voices | Steve McQueen | `steve mcqueen` | 6 |
| Modern voices | Alfonso Cuarón | `alfonso cuaron` | 9 |
| Modern voices | Alejandro González Iñárritu | `alejandro g inarritu` | 10 |
| Modern voices | James Wan | `james wan` | 12 |

**Credit-string notes** (cases where the DB form deviates from the most common spelling — verified by direct query):
- `bong joon ho` — Bong Joon-ho's IMDb credit lacks the hyphen; canonical key has no hyphen, space-separated.
- `hirokazu koreeda` — Kore-eda is credited as one word in IMDb (no hyphen, no space).
- `alejandro g inarritu` — credited as "Alejandro G. Iñárritu"; the period collapses to "g", diacritic stripped.

The Coen brothers and any other co-directing pair are stored as **separate keys** (one per person). The lane fires for either credit independently.

**Deliberate exclusions** (so future passes don't re-litigate):
- Generalist craftsmen with no signature: Ridley Scott, Ron Howard, Steven Soderbergh (distinctive *approach* but genre-promiscuous), Clint Eastwood (visually plain by design), James Cameron, Guy Ritchie (one genre lane), Kevin Smith, Robert Rodriguez.
- Sample size < 3 features: Jonathan Glazer, Chloé Zhao — reconsider when filmographies grow.
- Held back pending user call on public-discourse considerations: Woody Allen, Mel Gibson, Roman Polanski. Style signal is real; inclusion is a separate decision.

### 2.2 Franchise lane — tier system based on lineage + subgroup, with spinoff/crossover handling

**Conceptual framing first** (per user discussion): when *should* we surface a same-franchise film, and how do we keep quality high when a franchise spans iconic mainline entries plus a tail of variable-quality spinoffs/shorts/crossovers?

The principle: **proximity within the franchise structure should map directly to candidate strength.** A user asking "movies like Star Wars (1977)" expects:
1. **Strongest**: other mainline entries in the same era/trilogy (Empire Strikes Back, Return of the Jedi).
2. **Strong**: mainline entries from a different era of the same franchise (Phantom Menace, Force Awakens).
3. **Moderate**: spinoffs in the same lineage (Solo: A Star Wars Story).
4. **Weak**: cross-franchise crossovers that include this franchise (rare for SW; common for MCU — e.g., Avengers crossing Iron Man + Cap + Thor lineages).
5. **None**: direct-to-DVD entries with no real shape or quality match.

**Verified data structure** ([similar_movies.py:486–514](../search_v2/similar_movies.py#L486), and queried catalog):
- `lineage_entry_ids`: the canonical franchise. Star Wars 1977 has lineage `[3]`; Empire `[3]`; Solo `[3]`. Avengers crossovers have multiple lineages (Iron Man + Cap + Thor + Hulk).
- `subgroup_entry_ids`: sub-franchise grouping. Star Wars 1977 has subgroup `[1, 2]` (= original trilogy + main feature films). Phantom Menace has `[2, 484]` (= main features + prequel trilogy). Force Awakens `[2, 8909]` (= main features + sequel trilogy). Solo has `[]` (no subgroup → spinoff signal!).
- `shared_universe_entry_ids`: cross-franchise universe (MCU, DCEU). Civil War has universe `[436]`; non-MCU films don't.

This gives us a **structural spinoff detector**: a candidate with the anchor's lineage but **no shared subgroup** is a spinoff. A candidate with **multiple lineages including the anchor's** is a crossover.

**Verified Star Wars confidence issue**:
- Star Wars lineage_id=3: confidence=0.719 (≥0.65 ✓), **consistency=0.571** (<0.6 ✗).
- `1 - clamp(2 * stddev(0.8*pop + 0.2*reception), 0, 1)` knocks Star Wars below the consistency gate because the lineage includes Holiday Special / shorts / weaker spinoffs alongside the iconic entries.
- The MV is correctly scoped to `movie_card`-eligible movies (verified — answers the user's question: no, we did *not* accidentally include all TMDB).

**V3 proposal — replace V2's confidence gating with a 2D role × overlap matrix**:

The structural lookup uses two dimensions: the **role** of each side (mainline / spinoff / crossover) and the **overlap relationship** (same subgroup / same lineage / same universe / disjoint). Each side's role is determined once per movie:

- **mainline**: single lineage AND non-empty `subgroup_entry_ids` (Empire Strikes Back, Iron Man, Toy Story 2)
- **spinoff**: single lineage AND empty `subgroup_entry_ids` (Solo: A Star Wars Story, Rogue One)
- **crossover**: `len(lineage_entry_ids) >= 2` (Avengers, Civil War, Endgame)

Then the (anchor_role, candidate_role, overlap) triple looks up a weight in this table:

| anchor role | candidate role | overlap relationship | weight |
|---|---|---|---:|
| mainline | mainline | same lineage + same subgroup | **1.00** |
| mainline | mainline | same lineage, different subgroup | 0.70 |
| mainline | spinoff | same lineage | 0.50 |
| mainline | crossover | shares ≥1 lineage | 0.40 |
| spinoff | spinoff | **same lineage** | **0.85** |
| spinoff | mainline | same lineage | 0.50 |
| spinoff | crossover | shares lineage | 0.40 |
| crossover | crossover | shares ≥1 lineage | **0.85** |
| crossover | mainline | shares lineage | 0.40 |
| crossover | spinoff | shares lineage | 0.40 |
| any | any | same universe, no lineage overlap | 0.30 |
| any | any | disjoint | 0.00 |

The three boldface rows encode the user's "role consistency is a strong signal" principle: same-role + same-lineage gets a bigger boost than mixed-role + same-lineage. Concrete examples:

- **Toy Story 1 ↔ Toy Story 2** = 1.00 (mainline-mainline, same subgroup).
- **Star Wars 1977 ↔ Phantom Menace** = 0.70 (mainline-mainline, same lineage `[3]` but different subgroup — original trilogy `[1, 2]` vs prequel `[2, 484]`).
- **Star Wars 1977 ↔ Solo** = 0.50 (mainline-spinoff, same lineage).
- **Rogue One ↔ Solo** = 0.85 (spinoff-spinoff, same lineage — the user's "even more strongly rewarded" case).
- **Avengers ↔ Civil War** = 0.85 (crossover-crossover, share lineages — both span Iron Man + Cap + Thor + Hulk).
- **Iron Man ↔ Avengers** = 0.40 (mainline anchor → crossover candidate; relevant but secondary to other Iron Man entries).
- **Civil War ↔ Iron Man** = 0.40 (crossover-mainline, shares lineage).
- **Iron Man ↔ Cap: Civil War** = 0.30 (same MCU universe, no shared lineage). Confirm with data — if Civil War actually carries Iron Man's lineage too, this falls into the 0.40 row instead.

**Tie-breaking rule**: take the **max** of all matching cells. A candidate that satisfies multiple rows (e.g., a mainline-mainline same-subgroup AND same-universe) takes the highest applicable weight (1.00 in that example).

**Drop the `franchise_confidence` gate entirely**. The structural matrix encodes match quality directly — Star Wars's "low consistency" measurement artifact is irrelevant once we score by structure rather than statistical variance over the lineage's IDs.

**Drop the `franchise_confidence` gate entirely**. The structural tier system makes Star Wars's "consistency problem" moot — within the mainline lineage+subgroup space (Tier 1/2), variance is naturally low because we're only comparing major entries.

**Quality safety net**: keep the V2 shape gate (`shape >= 0.35` for franchise candidates to enter the top 5–10). Direct-to-DVD spinoffs that the centroid doesn't pull naturally won't surface even at Tier 3. Optional V3 addition: per-candidate quality gate — `popularity_percentile >= 0.50` OR `reception_score >= 60` — for franchise candidates, to filter the long tail.

**Crossover detection**: simple — `len(lineage_entry_ids) >= 2`. Verified the data: Avengers films have multiple lineages, regular MCU entries have one. May want to inspect the actual data shape for Avengers in the catalog before fully committing to this signal.

This rework also fixes the Barbie problem differently: Barbie's lineage is sparse and direct-to-DVD spinoffs all share the same lineage but no subgroup → Tier 3 (0.50) instead of full strength → they don't dominate. No need for the V2 `franchise_low_confidence` low-multiplier path at all.

### 2.3 Single-anchor needs themes / keyword / concept_tag support

**Findings (user direction)**: "*Barbie's main characteristics aren't being fully represented... female lead (we have a tag for that) and other distinct tags... we don't currently make use of keywords / concept tags / culture tags for the single movie case right?*"

Confirmed by reading [search_v2/similar_movies.py:1437–1439](../search_v2/similar_movies.py#L1437):

```python
# Multi-only lanes stay empty in single-anchor flow.
"themes": {},
"cast": {},
"specific_award": {},
```

Single-anchor flow drops the themes lane entirely. So Barbie's defining concept tags (`FEMALE_LEAD`, `BARBIE_DOLL`, `SATIRE`, `EXISTENTIAL`, etc., whatever the catalog has) provide no signal. The centroid does the work alone.

**User's structural question**: "*Should we break it up into distinct categories? Like keywords that serve as info on themes and stuff, keywords that serve as style or production, keywords that serve as source of inspiration, keywords that serve as culture, etc? Or would it be better to do a generic 'any keywords this movie has gets used with its strength being based on how rare that keyword is' and then tune based on the results? If a keyword is especially rare should it get its own lane?*"

**Recommendation: start with generic IDF, observe, then split categories only if a category visibly underperforms.**

Why generic IDF is the right starting point:
- IDF naturally does what categorization is trying to do: rare = important, common = not. `BARBIE_DOLL` (very rare) carries near 1.0 weight; `DRAMA` (very common) carries near 0. The math handles the cardinality problem we'd otherwise hand-tune via category weights.
- Categorization adds ~5 hand-tuned multipliers (theme×source×style×culture×production) and a maintenance burden. Each multiplier is a knob we have to test against real outputs and decide a value for. With generic IDF there's one knob: the lane weight.
- We can already observe whether IDF is doing the job by inspecting V3 outputs. If, say, source-of-inspiration tags are systematically underrepresented in the lane, we can boost them with a category multiplier *then* — but we should measure first.
- Multi-anchor V2 already ships generic IDF in the themes lane and it's clearly contributing real signal (Eternal Sunshine surfacing for Nolan trio, Killers of the Flower Moon for Best Picture trio).

**On rare keywords getting their own lane**: not initially. IDF already gives a single rare keyword shared between anchor and candidate near-1.0 contribution to themes. Promoting "exceptionally rare → own lane" adds threshold tuning (what counts as exceptionally rare?) and lane-weight allocation overhead. If after V3 testing we observe e.g. "rare-keyword matches are competing too closely with shape-only matches", *then* split out a "rare_signal" lane with its own weight. Premature otherwise.

**Action — single-anchor themes lane**:
- Run themes scoring against the anchor's own trait pool (vs. multi-anchor's "repeated traits across anchors"). Use the same `(kind, trait_id)` namespacing already in `_themes_traits_for_movie`.
- Score: `themes_score(candidate) = sum(idf(t) for t in candidate_traits ∩ anchor_traits) / sum(idf(t) for t in anchor_traits)`. Same formula shape as multi-anchor; the denominator is the IDF mass of the anchor's own traits.
- Lane weight: copy V2 multi-anchor's `0.06` base.
- Active for single-anchor regardless of trait count (no "must repeat ≥2 anchors" gate; that's a multi-anchor requirement only).
- Continue excluding format / medium / country tags from the themes pool (they're handled by their dedicated lanes/multipliers). Once §1.1 / §1.2 are merged, DOCUDRAMA / TRUE_CRIME / SKETCH_COMEDY / ADULT_ANIMATION / HOLIDAY_ANIMATION will be **available** to themes (they're no longer in `FORMAT_KEYWORD_IDS_ALL` or `MEDIUM_TAG_IDS`) — which is the desired behavior. Documentary biopics get to share `DOCUDRAMA` as a content-similarity signal even though they're no longer format-bucketed as documentary.

**Direct effect on Barbie**: `FEMALE_LEAD`, `SATIRE`, `EXISTENTIAL`, `BARBIE_DOLL`, `MUSICAL_NUMBER`, etc. become signal. The Telugu film at #1 likely shares few of those rare tags; Poor Things and other on-theme films share them and rise. Combined with §1.3 country/language penalty, the cross-tradition #1 problem disappears.

**Should single-anchor also get `cast` and `specific_award`?**
- Cast: keep disabled (per user direction below). DiCaprio-in-Titanic isn't Inception; the V2 rationale stands.
- Specific_award: defer to V4. Sharing a Best Picture award between anchor and candidate is real prestige signal, but the middle-bucket quality lane already covers most of the case.

### 2.4 Country/language coherence for single-anchor

Already covered by §1.3. Calls back here for completeness — single-anchor flow gets a country/language multiplier driven by the anchor's own tradition, with the calibrated `1.05` / `0.75` multipliers.

### 2.5 Cast lane — multi-anchor bucket-with-floor (generic N-anchor formula)

**User direction**: "*for the multi movie case I'd say if they share a lead actor that's actually a really big signal we should at least try to weave in, even if the movies have other dominant shapes. Like if I had 'big', 'polar express', and 'toy story' that's 3/3 on Tom Hanks. I should absolutely weave in some Tom hanks movies in there.*" Followed by: "*treat it as a bucket when there's strong cohesion with a reasonable floor, kinda like what we had for director before (but don't let it get too crazy)*", and "*ensure this is for top 3 billing position not just #1*", and "*give this as a generic formula when we have N movies and M actor matches*".

**Generic formula** (works for any number of anchors `N`):

Definitions (per anchor set):
- For each lead actor (anyone in **top-3 billing** in *any* anchor), let `M_a` = count of anchors where actor `a` is in top-3 billing.
- `shared_leads = {a : M_a >= 2}` — actors carrying real cohesion signal (singletons are dropped).
- `max_M = max(M_a for a in shared_leads)` (or `0` if `shared_leads` is empty).
- `ratio = max_M / N`.

Per-candidate cast lane score (additive contribution):
```text
matches    = |candidate.top_3_billing ∩ shared_leads|
cast_score = matches / max(1, len(shared_leads))      # range [0, 1]
```

Lane weight (scales with cohesion strength):
```text
single-anchor:   shared_leads is empty by construction → cast lane silent.
multi-anchor:    lane_weight = 0.05 + 0.10 * ratio    # silent when max_M < 2
                                                      # ratio = 0.5 → 0.10
                                                      # ratio = 0.67 → 0.117
                                                      # ratio = 1.00 → 0.15
```

Bucket floor (the weaving mechanism, multi-anchor only):
```text
if max_M < 2 or ratio < 0.5:
    floor = 0
else:
    floor = 0.25 + 0.20 * ratio
```
Yields: `2/4 → 0.35`, `2/3 → 0.384`, `3/4 → 0.40`, `3/3 → 0.45`, `4/4 → 0.45`, `3/5 → 0.37`.

Floor application:
- Activates per-candidate when `cast_score > 0` (candidate matches at least one shared lead in their own top-3 billing).
- `combined_score = max(combined_score, floor)` — pulls weak-shape candidates up to the floor; doesn't touch already-strong candidates ("don't let it get too crazy").
- **Shape gate**: floor only applies when `shape >= 0.30`. Truly off-shape candidates (e.g. a Tom Hanks indie drama vs. a Big/Polar Express/Toy Story trio that's strongly family-comedy-shaped) don't get pulled in — keeps the floor from rescuing tonally-wrong matches.

Multi-shared-lead case (e.g. Hanks AND Travolta both in 2/3 anchors):
- Floor magnitude uses `max_M` only.
- Per-candidate `cast_score` differentiates within the bucket — a candidate matching both shared leads scores higher additively than one matching just one. That's where multi-actor strength gets captured.
- Optional refinement once observed: bump floor by `+0.03` per additional shared lead beyond the first, capped at `+0.10`. Defer until real cases call for it.

**Single-anchor cast**: stays disabled per user direction. DiCaprio-in-Titanic isn't Inception.

### 2.6 Rare-keyword lane — tiered IDF with combo-driven floor (NEW)

**User direction**: "*each keyword should contribute its own score rather than cramming them all into a single closeness value...this should update lane weaving as well when the rarity (or combination of the rarity) is unique enough. Kinda like it adds a floor similar to multi-anchor actor so the shape still needs to be decent in order to make it work. Maybe low to medium rarity terms are pure additive (low could combine into that single 0.05 weight, moderate could be individual additions) and then high rarity individuals or high rarity combos (or maybe just allow them to stack in some way) changes weaving logic.*"

This is a new lane added in V3 (parallel to themes — it operates on the same trait pool but applies tiered visibility and a floor mechanism for distinctive matches). The themes lane (§2.3) handles common-and-moderate signal in aggregate; the rare-keyword lane handles the high-rarity individual + combo cases that deserve weave-level attention.

**Pool definition** (which traits participate):

Include:
- Concept tags (`CategoryName.CONCEPT_TAG`) — designed to be specific, the highest-value pool.
- `OverallKeyword`s NOT handled by other lanes/registries (see exclusion list).
- TMDB genres (`tmdb_genres`) — included per user direction: "*include genres for now. Worst case they don't contribute as much when they're frequent.*" High-IDF genres (rare combinations) will surface; common genres (DRAMA, COMEDY) will fall into the low tier and be effectively muted.

Exclude (already handled elsewhere — would double-count):
- `FORMAT_KEYWORD_IDS_ALL` (format lane).
- `MEDIUM_TAG_IDS` (medium multiplier).
- Country/language tags from `country_language_registry` (country/language multiplier).
- Source-material tags (source IDF lane).
- Award-related category tags (quality / specific_award).

**Three rarity tiers** (using IDF):

| Tier | IDF range | Behavior |
|---|---|---|
| **Low** | `< 2.5` | Pooled into the existing **themes lane** (§2.3 generic IDF). Combined contribution weighted at `0.05`, capped. *No individual visibility — just adds to themes lane numerator.* |
| **Moderate** | `2.5 ≤ IDF < 4.5` | **Individual additive contributions.** Each shared moderate-tier trait adds `IDF × 0.03` to combined score. Visible per-trait in the rare-keyword lane breakdown (separate from themes lane). |
| **High** | `IDF ≥ 4.5` | **Individual additive contributions** at higher weight: `IDF × 0.05` per shared trait. Plus counts toward floor trigger (below). |

IDF thresholds `2.5 / 4.5` are starting points — sanity-check against the actual `mv_trait_idf` distribution before locking. The thresholds should land roughly at the p70 / p90 IDF percentile (i.e., low = bottom 70 % of trait IDFs, moderate = next 20 %, high = top 10 %).

**Floor mechanism (the weaving update)**:

Triggers if either:
1. **Single super-rare hit**: `max(IDF for shared high-tier traits) >= 5.0`.
2. **Rare combo**: `sum(IDF for shared high+moderate-tier traits) >= 7.0`. So 2-3 strong moderate matches can also trigger, even without a single super-rare hit.

When triggered:
- `floor = 0.40 + 0.05 * (number_of_high_tier_matches)`, capped at `0.55`.
- **Shape gate**: `shape >= 0.30` required (same as cast — no truly off-shape rescues).
- Apply: `combined_score = max(combined_score, floor)`.

**Single vs multi-anchor**:
- Single-anchor: pool = anchor's own keywords + concept tags + genres, after exclusions.
- Multi-anchor: pool = traits shared across ≥2 anchors (cohesion intersection — same as the themes lane multi-anchor logic).

**Where it adds to score** (both flows):
- Low tier: into themes lane (no separate score addition — themes lane is the home).
- Moderate / high tier: each contributes individually to the rare-keyword lane's additive sum, with high-tier weighted more.

**Where it updates weaving** (both flows):
- Floor only — same mechanism as cast multi-anchor §2.5. Sufficient rare matches pull combined score up to the floor regardless of weaker shape (with shape gate). Most rare-keyword signal is pure score; only the high-rarity / strong-combo cases get bucket protection.

**Concrete examples**:

Oppenheimer anchor with candidate sharing `MANHATTAN_PROJECT` (IDF ≈ 6.0, high tier):
- Score: `+0.30` from high-tier individual contribution (6.0 × 0.05).
- Floor: max IDF 6.0 ≥ 5.0 → activates. Magnitude: `0.40 + 0.05 × 1 = 0.45`.
- Effect: candidate at combined 0.32 with shape 0.40 → pulled up to 0.45, lands in top 10. Candidate already at 0.55 → unchanged (already over floor).

Memento anchor with candidate sharing `NON_LINEAR_NARRATIVE` (≈3.5), `UNRELIABLE_NARRATOR` (≈4.0), `AMNESIA` (≈3.8):
- Score: `0.105 + 0.120 + 0.114 = 0.339` total moderate-tier additive.
- Floor: sum 11.3 ≥ 7.0 → activates (combo path, no high-tier hit). Magnitude: `0.40 + 0.05 × 0 = 0.40`.
- Effect: thematically-distinctive matches surface even when shape is mediocre.

**Why this matters beyond pure score**: visibility. When Oppenheimer matches via "Manhattan Project" specifically, the hit shows up labeled in the lane breakdown rather than smearing into a generic themes score. Easier to explain ranks to ourselves and to debug regressions.

---

## 3. Weaving and multipliers

### 3.1 Format weave — harsh downrank for shorts, asymmetric short-anchor handling

**Combines V2 findings F2 + user direction**: "*When we're matching on a single movie that isn't a short or there isn't moderate cohesion for the short format in the multi case we should severely downrank shorts. Like the user clearly isn't looking for shorts so they should be fully locked out of the top 10 but I don't want it to reach #10 and then be a flood of shorts so really we should do a HARSH downrank on them.*"

**Mechanism — multiplier + hard cap, not just a top-5 lock**:
The V2 top-5 lock is a half-fix. It blocks shorts from positions 1–5 but lets them flood positions 6–10 because the underlying combined score still ranks them well. The fix has to act on the score itself, with a structural cap as a safety net.

- **Combined-score multiplier**: when candidate is `format_bucket = short` and the active anchor bucket is *not* `short`, multiply the candidate's combined score by **0.30**. A typical short-with-strong-franchise candidate (Partysaurus Rex against Toy Story at raw 0.34) drops to ~0.10, well below position-10 of the feature pool.
- **Hard cap**: at most 1 short in the entire top 10, regardless of score. This catches pathological cases (catalog-wide Pixar short pile-up surviving the 0.30 multiplier).
- **Multi-anchor "moderate cohesion" definition**: the dominant format bucket = the bucket that ≥50 % of anchors share. If no bucket clears 50 %, treat as "no consensus" → no shorts penalty (drop the constraint, same as V2). If shorts clear 50 %, treat shorts *as* the dominant bucket → no penalty, in fact slight upweight (1.10 multiplier within shorts).
- **Short-anchor case** (single-anchor with anchor bucket = `short`): invert — apply a soft top-1 lock (require at least one short in top 1, then let features fill from 2 onward). Don't fully lock to shorts; the catalog's shorts skew toward the same-franchise tail and the user likely also wants the related features.

**Where this lives in code**:
- The multiplier is applied during the combined-score computation in [`_build_results`](../search_v2/similar_movies.py#L1016) (or wherever `combined_score` is finalized), conditional on `format_bucket(candidate) == "short"` and `anchor_format_bucket != "short"`.
- The hard cap is applied during [`_weave_candidates`](../search_v2/similar_movies.py#L1016) — track a running short-count as we fill the top section + remainder, skip any short past the first.
- Replaces today's `TOP_FORMAT_LOCK = 5` enforcement that only catches half the problem.

This kills all the "Pixar shorts at #9–10 / Animatrix shorts at #7–10 / They Shall Not Grow Old at #8" failures simultaneously.

### 3.2 Medium multiplier — strengthen for live-action ↔ animation crossings only

**Detailed explanation** (per user request):

V2's medium multiplier is `0.85 + 0.15 * medium_score`, range `[0.85, 1.0]`:
- Perfect medium agreement (e.g., live-action vs live-action) → multiplier = `1.00` (no penalty).
- Within-animation cross-technique (e.g., CG vs stop-motion, score 0.50) → multiplier ≈ `0.925` (7.5 % penalty).
- Live-action vs animation (medium_score = 0) → multiplier = `0.85` (15 % penalty).

The 15 % penalty is the issue: animated Batman films (Year One, Long Halloween Pt 2) ride into Dark Knight's top 10 with franchise=1.0 + source=1.0 + format=1.0 + shape ≈ 0.30 contributing roughly +0.30 to combined score. The 0.85 multiplier subtracts ~0.045 — not enough to displace them. Wallace & Gromit (stop-motion) entered Toy Story's top 10 at #4 with the lighter 7.5 % penalty.

The fix has to acknowledge that **live-action vs animation is a categorically different watching experience**, while within-animation crossings (CG vs stop-motion, hand-drawn vs anime) are softer differences. The V2 spec itself frames it that way; the multiplier just needs to follow through.

**Proposal — piecewise multiplier**:

```python
def _medium_multiplier(anchor_tags, candidate_tags):
    if not anchor_tags or not candidate_tags:
        return 1.0
    score = medium_score(anchor_tags, candidate_tags)
    if score == 0.0:
        # No overlap at all → categorical mismatch (live ↔ animation).
        # Drop combined score by 35% so on-brand-but-wrong-medium franchise
        # candidates can't ride strong franchise/source contributions into
        # the top of the list.
        return 0.65
    # Within-category crossings (CG vs stop-motion, anime vs HD, etc.):
    # keep V2 behavior — partial penalty proportional to similarity gap.
    return 0.85 + 0.15 * score
```

Effects:
- LIVE ↔ ANIM (score 0.0): multiplier `0.65` (35 % penalty). Animated Batman films in TDK drop combined contribution from ~0.50 to ~0.33 — likely below shape-only adjacents.
- CG vs STOP_MOTION (score 0.50): multiplier `0.925` (unchanged from V2).
- ANIME vs HD (score 0.85): multiplier `0.978` (unchanged).
- Perfect medium match: `1.00` (unchanged).

**Alternative — hard gate (more aggressive)**: replace the multiplier with a top-10 gate. If anchor has a strong medium signal (e.g., LIVE_ACTION-only or any animation-only), candidates must satisfy `medium_score(anchor, candidate) >= 0.50` to enter the top 10. Implies hard exclusion of pure animation from a live-action anchor's top 10.

Recommendation: start with the piecewise multiplier (lower-risk change). If V3 testing still shows wrong-medium leakage, escalate to the hard gate.

### 3.3 Cross-cultural penalty for single-anchor

Already covered in §1.3 / §2.4. Apply alongside medium / format gates.

---

## 4. Quality, fallback, and miscellaneous

### 4.1 Middle-bucket quality lane — narrow scope

V2 made the middle-bucket quality lane always-on (`0.06` weight). It's surfacing tonally-wrong prestige picks: Lawrence of Arabia / Citizen Kane in The Godfather's top 10 with no real shape adjacency. The lane is doing what it was told (rank by `0.8*pop + 0.2*reception`) but it's adding noise.

**Action**: lower the middle-bucket quality lane weight from `0.06` to `0.03` (closer to the V1 "off" baseline), or keep it active only as a tie-breaker when shape scores are within `0.03` of each other.

### 4.2 Low-cohesion fallback gate

V2 finding: chaotic mixed bag (Toy Story + Godfather + Sharknado) had `mean_pairwise_cosine = 0.276` (clearly chaotic) but `metadata_max_cohesion = 1.69` (Toy Story + Godfather are both `prestige`) prevented the fallback from firing.

**Action**: raise `LOW_COHESION_METADATA_MAX_THRESHOLD` from `1.0` to `1.5`. The current threshold catches most chaotic sets but leaks the prestige-by-2-of-3 case.

### 4.3 Production vector weight — already lowered, may need further attention

V2 single-anchor already drops `production` from V1's 0.65 → 0.30 in the base profile. Titanic still shows ship-disaster matches at 6/10 of the top 10. If V3 testing shows continued ship/setting bias on Titanic-like anchors, reduce further or zero out `production` for single-anchor (keep it for multi-anchor where cohesion does its own filtering).

### 4.4 Source-IDF — confirm all source-material types have meaningful IDF spread

Verified: source-material trait IDFs span 0.198 (most common — likely "novel") to 0.547 (rarest). Even rare matches max out at ~0.55, so the lane is intentionally weak. This is fine for V2's intent (kill the `Players (2012)` Bollywood false positive), but means the lane will rarely surface a candidate by source alone. Acceptable.

### 4.5 Author-level source matching (deferred)

Source author (Stephen King-specific, Tolkien-specific) is not in the DB. Adding it would let the Stephen King horror trio surface Carrie/Pet Sematary/Cujo specifically. Out of scope for V3; tracked as V4.

---

## 5. Implementation order / priority

By impact (high → low), with recommended priority. Items marked ✅ are already shipped.

| # | Change | Section | Impact | Status |
|---|---|---|---|---|
| 1 | Remove DOCUDRAMA + TRUE_CRIME + SKETCH_COMEDY from format buckets | §1.1 | **Huge** — fixes Oppenheimer, Best Picture trio, Monty Python miscategorization | ✅ done |
| 2 | Remove ADULT_ANIMATION + HOLIDAY_ANIMATION from MEDIUM_TAG_IDS | §1.2 | Medium-large — fixes false matrix matches; cleans medium IDF signal | ✅ done |
| 3 | Shorts harsh downrank (0.30 multiplier + max-1 hard cap) for non-short anchors / no-cohesion multi-anchor | §3.1 | Huge — fixes Pixar shorts, Animatrix shorts, LotR documentary | pending |
| 4 | Add themes lane (generic IDF over keywords/concepts/genres) to single-anchor | §2.3 | Large — fixes Barbie identity, Get Out, Mad Max-style anchors | pending |
| 5 | Add country/language multiplier to single-anchor + tune to `1.05`/`0.75` | §1.3, §2.4 | Medium-large — fixes Barbie Telugu #1; tighter cross-tradition bar everywhere | pending |
| 6 | Franchise rework: structural tier system + spinoff/crossover detection | §2.2 | Medium-large — fixes Star Wars silent-franchise, Barbie suppression, Avengers crossover handling | pending |
| 7 | Director: manual auteur list only; drop non-curated tier entirely | §2.1 | Medium — fixes Lucas's American Graffiti, Spielberg over-firing, etc. | pending (auteur list locked 2026-05-07) |
| 8 | Cast lane: generic N-anchor formula — bucket-with-floor on shared top-3 leads (multi-anchor) | §2.5 | Medium — fixes Tom-Hanks-trio scenario | pending |
| 9 | Rare-keyword lane: tiered IDF (low/moderate/high) + floor on high-rarity hits or rare combos (NEW) | §2.6 | Medium — interpretability + distinctive-match weaving for both single and multi | pending |
| 10 | Medium multiplier: piecewise — `0.65` for cross-category, V2 formula within category | §3.2 | Medium — fixes animated Batman in TDK, Wallace & Gromit in Toy Story | pending |
| 11 | Lower middle-bucket quality weight `0.06 → 0.03` | §4.1 | Small — cleans prestige-pick noise | pending |
| 12 | Raise low-cohesion metadata threshold `1.0 → 1.5` | §4.2 | Small — chaotic-mixed-bag handling | pending |
| 13 | Re-verify award SPECIFICITY_FACTOR L2 after above changes | §1.5 | Smallest — observe first | pending |

**Suggested batches** (each batch is testable end-to-end in isolation):

- **Batch A — categorization fixes** (✅ done): items 1, 2. Pure registry edits, low blast radius. Already verified Oppenheimer correctness improvement.
- **Batch B — single-anchor enrichment**: items 4, 5, 7. Add themes + country to single-anchor; auteur-list director rework. Biggest architectural change in V3 but unlocks Barbie-class cases that V2 silently fails. Auteur list is finalized (§2.1) — Batch B is unblocked end-to-end.
- **Batch C — franchise + weaving + cast**: items 3, 6, 8. Tier-system franchise, format harsh-downrank, cast generic-formula bucket-with-floor. Fixes the rank-order issues that the lane-level reworks alone don't.
- **Batch D — distinctive-match interpretability**: item 9 (rare-keyword lane). Adds the new lane with tiered IDF and floor; depends on the themes lane (Batch B item 4) being in place since they share the trait pool.
- **Batch E — multiplier strengthening**: item 10. Asymmetric medium multiplier; small-surface change but high specificity for the live↔anim crossing failures.
- **Batch F — tuning**: items 11, 12, 13. Observe-first calibration once A–E are in.

Each batch should be re-run against the same 20 single-anchor + 12 multi-anchor sets used here, so we can quantify wins/regressions per change.

---

## 6. Open questions / things to verify before V3 implementation

1. **Auteur list — RESOLVED 2026-05-07.** The full 60-entry list is now embedded in §2.1 with normalized keys verified against the live DB. Resolutions on the prior open sub-questions: (a) inclusion criteria — pure stylistic auteurs whose *whole catalog* is likely to feel cohesive to a viewer; franchise-defining genre directors are in only when their broader filmography also coheres (Carpenter ✅ for synth-anamorphic-siege fingerprint; not gating on franchise-only fame); (b) sourcing — hand-picked from parametric knowledge plus a research subagent's auteurism survey, then filtered by the three-criterion bar in §2.1; (c) data-side resolution — Coen brothers stored as **two separate keys** (`joel coen`, `ethan coen`); lane fires for either independently. No Cronenberg ambiguity in the list (only David Cronenberg included).

2. **Crossover detection** (§2.2). Verify on actual catalog: do Avengers films *really* carry multiple lineage_entry_ids, or is the data structured with a single "Avengers lineage" that subsumes Iron Man / Cap / Thor lineages? The franchise tier system depends on this. A 2-line query against `movie_card.lineage_entry_ids` for Avengers films will resolve it.

3. **Cast top-3 ordering** (§2.5). Verify `movie_card`'s cast member IDs are ordered by billing position. If not, the cast lane today is reading "first 3 cast members" without billing-order semantics, and the Tom Hanks weave reservation could miss the Hanks signal because his ID isn't in slot 0–2.

4. **Themes IDF — denominator behavior on tag-rich anchors** (§2.3). Movies with 50+ keyword/concept tags will have a denominator dominated by IDF mass. Verify the score distribution looks reasonable (top candidates should still hit ~0.5+, not collapse to ~0.05 because the anchor has too many tags).

5. **Director auteur list — multi-anchor behavior**. The proposal keeps non-auteur director cohesion firing in multi-anchor when ≥2 anchors share a director. But should we still allow that for *any* director, or also gate on "≥2 anchors share a director AND that director has ≥3 films in the catalog" (to avoid noise from 1-credit directors)?
