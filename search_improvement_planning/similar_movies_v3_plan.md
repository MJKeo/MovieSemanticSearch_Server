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
- Maintain a **manual auteur list** (~30–50 directors) for "truly iconic, style-coherent" directors. These get the full director-lane boost.
  - Initial list (style is consistent across filmography): Quentin Tarantino, Wes Anderson, David Lynch, Stanley Kubrick, Martin Scorsese, Coen Brothers, Hayao Miyazaki, Christopher Nolan, Paul Thomas Anderson, Darren Aronofsky, Jim Jarmusch, Spike Lee, Spike Jonze, Charlie Kaufman, Edgar Wright, Yorgos Lanthimos, Ari Aster, Robert Eggers, Jordan Peele, Greta Gerwig, David Fincher, Sofia Coppola, David Cronenberg, Lars von Trier, Pedro Almodóvar, Luca Guadagnino, Jane Campion, Mike Leigh, Ken Loach, Akira Kurosawa, Andrei Tarkovsky, Krzysztof Kieślowski, Bong Joon-ho, Park Chan-wook, Hirokazu Kore-eda, Takashi Miike, Wong Kar-wai, Hou Hsiao-hsien.
  - Stored as a curated Python list / YAML mapping director term IDs to a tier (initially just "auteur" / not — flat boolean). Easy to extend; no MV needed.
- For directors **not** on the auteur list (single-anchor): **drop director lane entirely**. Their films compete on shape, themes, franchise, etc. The user's reasoning holds: when director matters for non-auteurs, it's via franchise or shape, both of which are already lanes.
- For **multi-anchor only**: when ≥2 anchors share a non-auteur director, treat that as a true repetition signal (the user explicitly picked multiple films by this director). Director cohesion fires regardless of auteur status. So in the multi case the lane stays cohesion-driven exactly as V2.
- **Retire `mv_director_strength`** as a lane signal. The auteur list replaces it. The MV may still be useful elsewhere (e.g., the director endpoint in the standard V2 search pipeline) — out of scope to rip out, but stop reading it from `similar_movies.py`.

### 2.2 Franchise lane — confidence formula breaks for large franchises; consider reworking entirely

**Findings**: investigated Star Wars franchise data:
- Star Wars lineage_id=3 has confidence=0.719 (≥0.65 ✓), **consistency=0.571** (<0.6 ✗).
- Therefore `franchise_high_confidence = False` → low-confidence path → multiplicative-only.
- That's why **no Star Wars film shows the franchise lane in its candidate_sources column** in the single-anchor output — the lane doesn't appear in `lane_scores`, only in the multiplier path.
- Captain America: Brave New World's lineage 447 (MCU): confidence=0.729, consistency=0.600 (exactly at threshold — barely passes).

The consistency formula at [mv_franchise_confidence](../db/init/01_create_postgres_tables.sql) is `1 - clamp(2 * stddev(0.8*pop + 0.2*reception), 0, 1)`. For Star Wars, the stddev across (Original trilogy + Prequels + Sequels + Solo + Holiday Special + animated entries + various spinoffs) is enough to push consistency below 0.6.

**This is wrong for two reasons**:
1. The MV is computed over *all* `movie_card`-eligible movies in the lineage. Big franchises have intrinsically more variance because they include shorts, spinoffs, TV movies, and weaker entries. A franchise being big shouldn't disqualify it.
2. Star Wars is *the* iconic franchise. If our system says it's "low-confidence" we have a definitional problem, not a marginal-tuning problem.

User question: "*Maybe it isn't a lane at all?*"

**Action — proposal**: replace the lane with a simpler, more user-aligned signal:
- **Option A (preferred)**: keep franchise as an **always-on additive lane** with a fixed weight (no confidence gating). Cap exposure with the existing `MAX_TOP_FRANCHISE = 3` rule and a shape gate (e.g., shape ≥ 0.30) so direct-to-DVD spinoffs can't dominate. The exposure caps already do most of the work; the confidence formula was over-engineered.
- **Option B**: drop franchise as an explicit lane. Rely on shape similarity (sequels naturally vector-cluster with the original) plus a separate "same-lineage" hard-gate — e.g., always include the top-3 highest-popularity-percentile non-anchor entries from the anchor's lineage, regardless of shape. This guarantees obvious franchise candidates appear without giving franchise its own scoring weight.
- **Option C (simplest)**: keep V2 architecture but drop the `consistency` requirement. Use `confidence ≥ 0.50` only. Star Wars (0.719) passes, prestige franchises (LotR, Pixar) pass, low-confidence franchises (Barbie kids' films) still fail.

I lean Option C as the smallest patch that fixes Star Wars while preserving the Barbie-kids'-films suppression that V2 nailed.

**Independent verification needed**:
- User asked: "*Did we accidentally look at all movies not just the ones with movie cards when making the franchise DB?*"
- Verified: the MV joins on `movie_card` ([01_create_postgres_tables.sql:mv_franchise_confidence](../db/init/01_create_postgres_tables.sql)) — only quality-passed movies are included. Not the cause of Star Wars's low consistency. The cause is intrinsic Star Wars variance.
- Verified the `_franchise_score_v2()` logic at [search_v2/similar_movies.py:494–514](../search_v2/similar_movies.py#L494): for Star Wars (1977) vs Empire Strikes Back, both have `lineage=[3]` so `a_lin & c_lin = {3}` → score 1.0. The lineage match works correctly; the issue is purely whether the franchise lane is *additive* or *multiplicative-only*.

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

**Action — single-anchor themes lane**:
- Run themes scoring against the anchor's own trait pool (vs. multi-anchor's "repeated traits").
- Score: `themes_score(candidate) = sum(idf(t) for t in candidate_traits ∩ anchor_traits) / sum(idf(t) for t in anchor_traits)`. Identical formula to multi-anchor, just with anchor traits in place of repeated traits.
- IDF weighting handles the cardinality problem automatically: shared `DRAMA` (very common) contributes near zero; shared `BARBIE_DOLL` (very rare) carries real signal.
- Lane weight: copy V2 multi-anchor's `0.06` base.
- This reform is independently helpful for every single-anchor case where the anchor has a distinctive concept-tag profile (Get Out's `RACE_RELATIONS`, Inception's `DREAMS`, Mad Max Fury Road's `POST_APOCALYPTIC` + `CHASE`).

**Should single-anchor also get `cast` and `specific_award`?** The V2 spec rejected single-anchor cast on the principle that "DiCaprio in Titanic isn't like Inception", which is correct. Single-anchor `specific_award` is more debatable — sharing a Best Picture award between anchor and candidate is a real prestige signal — but I'd defer it to V4 to stay focused.

### 2.4 Country/language coherence for single-anchor

Already covered by §1.3. Calls back here for completeness — single-anchor flow gets a country/language multiplier driven by the anchor's own tradition.

---

## 3. Weaving and multipliers

### 3.1 Format weave — extend lock to all 10 slots and exclude shorts/docs entirely when the anchor is `narrative_feature`

**Combines V2 findings F2 + user direction**: "*We should honestly exclude shorts if we're not searching for similar to a short. If we have a short we should heavily boost shorts, and if there is significant overlap in the multi case where they're shorts then boost it there, otherwise harshly penalize or fully exclude.*"

**Action**:
- For single-anchor:
  - If anchor format ∈ {`narrative_feature`, `documentary`, etc.}: **hard-exclude** candidates from a different format bucket from the entire top 10 (not just top 5). `(top + remainder)` should be filtered before slicing.
  - If anchor format = `short`: invert — heavily boost short candidates (e.g., medium-style multiplier of 1.10 for same-bucket, harsh 0.70 for cross-bucket, OR a hard exclusion of non-shorts).
- For multi-anchor:
  - If a single format bucket repeats across ≥2 anchors: hard-exclude other buckets from top 10, same as single-anchor.
  - If anchors disagree on format: drop the constraint entirely (current V2 behavior).
- Update [search_v2/similar_movies.py:_weave_candidates](../search_v2/similar_movies.py#L1016) to apply the format gate to *both* the top section and the deferred remainder, replacing today's 5-slot-only enforcement.

This single change kills all the "Pixar shorts at #9–10 / Animatrix shorts at #7–10 / They Shall Not Grow Old at #8" failures.

### 3.2 Medium multiplier — strengthen for live-action vs animation

**V2 finding F3**: floor `0.85` is too soft. For live-action vs animation, 15 % penalty doesn't push animated Batman / Wallace & Gromit out of TDK / Toy Story top 10.

**Action**:
- Make the multiplier asymmetric: live-action ↔ animation = `0.65–0.70` floor (35 % penalty); within-animation cross-technique = `0.85` floor (current behavior).
- Or apply medium as a **hard gate inside the format top-10 lock**: a candidate must share at least one medium tag at score ≥ 0.50 with the anchor to enter the top 10 when the anchor has a strong medium signal. (Keep the multiplier for finer ranking within the gate.)

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

By impact (high → low), with recommended priority:

| # | Change | Section | Impact |
|---|---|---|---|
| 1 | Remove DOCUDRAMA + TRUE_CRIME from documentary bucket | §1.1 | **Huge** — fixes Oppenheimer, Best Picture trio, frees GoodFellas et al. |
| 2 | Remove ADULT_ANIMATION + HOLIDAY_ANIMATION from MEDIUM_TAG_IDS | §1.2 | Medium-large — fixes false matrix matches; cleans medium IDF signal |
| 3 | Hard-exclude wrong-format candidates from full top 10 (not just top 5) | §3.1 | Huge — fixes Pixar shorts, Animatrix shorts, LotR documentary |
| 4 | Add themes lane (IDF over keywords/concepts/genres) to single-anchor | §2.3 | Large — fixes Barbie, Get Out, Mad Max-style anchors with distinctive concept-tag profiles |
| 5 | Add country/language multiplier to single-anchor | §1.3, §2.4 | Medium-large — fixes Barbie's Telugu #1 |
| 6 | Director lane: manual auteur list, drop pure director_strength | §2.1 | Medium-large — fixes Star Wars's American Graffiti, Lucasfilm's non-SW credits, balances Tarantino/Fincher |
| 7 | Franchise: drop consistency requirement (or rework lane entirely) | §2.2 | Medium — fixes Star Wars franchise silence, doesn't break Barbie suppression |
| 8 | Strengthen medium multiplier asymmetrically (live ↔ anim = 0.65 floor) | §3.2 | Medium — fixes animated Batman in TDK |
| 9 | Lower middle-bucket quality weight 0.06 → 0.03 | §4.1 | Small — cleans prestige-pick noise |
| 10 | Raise low-cohesion metadata threshold 1.0 → 1.5 | §4.2 | Small — chaotic-mixed-bag handling |
| 11 | Re-verify award SPECIFICITY_FACTOR L2 after above changes | §1.5 | Smallest — observe first |

**Suggested batches** (each batch is testable end-to-end in isolation):

- **Batch A (categorization fixes)**: 1, 2 — pure registry edits, low blast radius, immediate visible improvement on Oppenheimer / Best Picture / Toy Story-shorts cases.
- **Batch B (single-anchor enrichment)**: 4, 5, 6 — add themes + country to single-anchor; auteur-list rework. Bigger architectural change but enables a class of cases V2 silently fails.
- **Batch C (weaving + franchise)**: 3, 7, 8 — final polish on rank order and exposure caps.
- **Batch D (tuning)**: 9, 10, 11 — observe-first calibration once A/B/C are in.

Each batch should be re-tested against the same 20 single-anchor + 12 multi-anchor sets used here, so we can quantify wins/regressions per change.
