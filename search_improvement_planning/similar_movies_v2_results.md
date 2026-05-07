# Similar Movies V2 — Test Results & Findings

End-to-end audit of the V2 single-anchor and multi-anchor flows on the same anchor sets used to evaluate V1. Each section below states the V2 output, what an informed user would expect, where V2 agrees and diverges, and the most likely code- or data-level cause of each divergence. The findings section at the end consolidates issues that appear across multiple test cases.

Source data: full V2 outputs are written to [search_v2/similar_movies_v2_batch.md](../search_v2/similar_movies_v2_batch.md) (single-anchor) and [search_v2/similar_movies_v2_multi_anchor.md](../search_v2/similar_movies_v2_multi_anchor.md) (multi-anchor). All scores quoted below come from those runs (limit=10, all 20 single-anchor anchors and all 12 multi-anchor sets).

---

## Cross-cutting findings

These four issues recur across multiple test cases and account for most of the V2 mismatches I observed. They are listed here once and referenced from individual test sections.

### F1 — DOCUDRAMA / TRUE_CRIME tags pull prestige biopics into the `documentary` format bucket

[search_v2/format_registry.py](../search_v2/format_registry.py) lines 48–65 group `DOCUDRAMA` and `TRUE_CRIME` into `DOCUMENTARY_KEYWORD_IDS`. With priority `documentary > narrative_feature`, any narrative biopic that carries one of these tags lands in the `documentary` bucket.

Verified against the catalog (anchors and key candidates):

| Movie | Has tag | format_bucket |
|---|---|---|
| Oppenheimer (2023) `872585` | DOCUDRAMA | documentary |
| Schindler's List (1993) `424` | DOCUDRAMA | documentary |
| 12 Years a Slave (2013) `76203` | DOCUDRAMA | documentary |
| The Pianist (2002) `423` | DOCUDRAMA | documentary |
| Killers of the Flower Moon (2023) `466420` | TRUE_CRIME | documentary |
| The Killing Fields (1984) `625` | DOCUDRAMA | documentary |
| Before Night Falls (2000) `5001` | DOCUDRAMA | documentary |
| The Godfather (1972) `238` | — | narrative_feature |

Consequences:

- **Oppenheimer single-anchor**: anchor itself is bucketed `documentary`. The top-5 format lock then *requires* documentary candidates, so positions 1–4 are filled with literal Manhattan Project documentaries (Einstein and the Bomb, Oppenheimer: The Real Story, The Trials of J. Robert Oppenheimer, plus the docudrama biopic Before Night Falls). Narrative-feature peers (Dunkirk, The Imitation Game, A Beautiful Mind) are pushed to positions 6+ or out of the top 10 entirely.
- **Best Picture multi-anchor trio (Godfather/Schindler/12YS)**: 2 of 3 anchors have DOCUDRAMA, so consensus format = `documentary`. The top-5 lock again forces docudrama biopics. Positions 1–4 (The Pianist, Killing Fields, Killers of the Flower Moon, Elephant Man) are all DOCUDRAMA/TRUE_CRIME. Goodfellas — the obvious Godfather adjacent — never appears.
- **LotR single-anchor**: position 8 `They Shall Not Grow Old` is a Peter Jackson WW1 documentary. It rides into top 10 on `director` (Jackson) + `quality`, despite being a different format. Format lock didn't catch it because it landed in the deferred remainder, not the top section (see F4).

Recommended fix: split DOCUDRAMA and TRUE_CRIME out of the documentary bucket. Treat them as their own bucket (or as part of `narrative_feature`) so a DOCUDRAMA tag on a biopic doesn't make it format-equivalent to a literal documentary about the same subject. Per the V2 spec's own framing, "the experience matches docs more closely than fiction" — but for prestige biopics like Oppenheimer/Schindler/12YS, viewer expectations are clearly narrative-feature, not Ken Burns.

### F2 — Format top-lock applies only to first 5 slots; deferred remainder fills 6–10 with anything

`_weave_candidates` in [search_v2/similar_movies.py:1016](../search_v2/similar_movies.py#L1016) enforces format coherence only for `TOP_FORMAT_LOCK = 5` slots, then relaxes for 6–10. Weak candidates rejected from the top section land in `deferred`; the final return is `(top + remainder)[:limit]`, so any candidate that didn't enter the top section but isn't otherwise filtered shows up in positions 6–10.

Examples:

- **Toy Story single-anchor**: positions 9 and 10 are `Small Fry (2011)` and `Partysaurus Rex (2012)` — both Pixar SHORTS (verified `format_bucket = "short"`). They fail `_can_enter_top_section` because dominant_lane = franchise and shape < 0.35, but the franchise score is high enough that they're the top remainder candidates. Result: feature anchor returns shorts in top 10.
- **Matrix single-anchor**: positions 7–10 are all Animatrix shorts (A Detective Story, Kid's Story, The Animatrix, Matriculated). Same mechanic — franchise saturated at 3 in top section, the Animatrix shorts get deferred and roll into the final list.
- **LotR single-anchor**: position 8 `They Shall Not Grow Old` (a documentary).

Recommended fix: extend the format-bucket gate to all 10 top-section slots, OR exclude `short` (and probably `documentary` when the anchor is a narrative feature) from the deferred-remainder pool entirely. The current 5-slot lock makes positions 6–10 effectively format-blind.

### F3 — Cross-medium 15 % penalty is too soft for live-action vs animation

`_medium_multiplier` produces `0.85 + 0.15 * medium_score` (range `[0.85, 1.0]`). For a live-action anchor vs. animated candidate, `medium_score = 0.0` → multiplier = 0.85 (15% penalty). That's not enough to pull animated entries off the top 10 when other lanes contribute strongly.

Examples:

- **The Dark Knight single-anchor**: positions 7 and 8 are `Batman: Year One (2011)` and `Batman: The Long Halloween, Part Two (2021)` — both animated (verified `ANIMATION` + `ADULT_ANIMATION` tags). Anchor TDK is `LIVE_ACTION`. medium_score = 0.0 → 0.85× multiplier. But these candidates have `franchise=1.0`, `source=1.0`, `format=1.0` (they don't have SHORT/DOCUMENTARY tags so they fall through to narrative_feature), and shape ≈ 0.30. The 15% penalty isn't enough to displace them.
- **Toy Story single-anchor**: position 4 `Wallace & Gromit: Vengeance Most Fowl` (stop-motion). Anchor is computer animation. medium_score(CG, STOP_MOTION) = 0.50 → multiplier ≈ 0.92. Soft enough that any decent shape match still rides into the top.

Recommended fix: lower the floor to ~0.70 for true live-action/animation crossings (medium_score = 0.0), or apply medium as a hard *gate* in the top-5 lock instead of a soft post-additive multiplier.

### F4 — Director-strength MV makes the director lane silent for moderately-known directors

In [search_v2/similar_movies.py:1347–1356](../search_v2/similar_movies.py#L1347), single-anchor director_score = `max(director_strengths.get(t, 0.0) for t in shared)`. Strength values come from `mv_director_strength` (percentile-rank of `0.8*pop + 0.2*reception`). The `director_signature` anchor type only fires when strength ≥ 0.80 (top-tier auteurs).

For directors *between* top-tier and obscure, the lane is awkward:

- A director with strength 0.5 contributes 0.5 to director_score even when they'd otherwise be a defining identity (Frank Darabont for Green Mile / Shawshank, Peyton Reed for Ant-Man, etc.).
- A director with only 1 cataloged film is excluded from the MV entirely → director_score = 0 for any candidate sharing that director, even though sharing the director is a real signal.

Examples:

- **The Green Mile single-anchor**: Shawshank Redemption (same director, Frank Darabont) appears at position 6 with score 0.392. With Darabont's strength likely < 0.80 (no `director_signature` for him in this run — though oddly it's listed as active for Green Mile, suggesting another contributing director), the lane delta is small. The expected #1 candidate is buried beneath weak shape-only matches.
- **Fight Club single-anchor**: David Fincher films (The Game #2, The Girl with the Dragon Tattoo #7) are present but sandwiched between obscure shape matches like `Luster (2010)` #1, `Revolver (2005)` #3, `Freeze Frame (2004)` #4. Fincher's strength is presumably high but the lane only weighs 0.10–0.15 in the additive sum, and the shape centroid for Fight Club lands in arthouse-thriller territory.

Recommended fix: when `director_signature` is active, the boost should be larger than `+0.10` to actually surface the auteur's filmography ahead of obscure shape adjacents. Alternatively, give moderate-strength directors a flat floor (e.g., `max(strength, 0.5)`) so director sharing always meaningfully matters.

---

## Single-anchor results

### Inception (2010) `27205` — active: standard_shape, director_signature

V2 top 10: Tenet, Trance, Paprika, Mulholland Drive, **The Prestige**, Babel, **Memento**, Holy Motors, Thank You Life, Vanilla Sky.

**Strengths**: Tenet at #1 (Nolan + same dream-physics premise) is exactly right. Paprika (anime that inspired Inception) at #3 is a great catch. Director_signature surfaces The Prestige and Memento (anchors-of-the-Nolan-trio in their own right). Vanilla Sky at #10 captures the dream/reality genre.

**Issues**: Babel at #6 is genre-mismatched (multi-narrative drama, not a mind-bender). Holy Motors at #8 is surreal but tonally very different (French art film). `Thank You, Life (1991)` at #9 is an obscure French film and looks like a vector-centroid artifact. None of these are major regressions; the centroid did most of the work and the director lane backfilled cleanly.

### The Matrix (1999) `603` — active: standard_shape, franchise_dominant, director_signature

V2 top 10: Matrix Reloaded, Total Recall, Tenet, Point Break, Matrix Revolutions, Matrix Resurrections, **A Detective Story**, **Kid's Story**, **The Animatrix**, **Matriculated**.

**Issues**: positions 7–10 are all Animatrix shorts. They're franchise-dominant with low shape (the franchise top-3 cap pushes them out of the top section, but they ride back in via remainder — see **F2**). Reasonable substantive candidates that should appear (Equilibrium, Inception, Dark City, Ghost in the Shell) are not surfaced. Point Break at #4 is a stretch — clearly a vector-centroid Wachowski-era action artifact.

### Star Wars (1977) `11` — active: standard_shape, prestige, studio_lineage, director_signature

V2 top 10: Empire Strikes Back, Phantom Menace, Force Awakens, Return of the Jedi, **LotR Fellowship**, Fifth Element, Star Trek (2009), Revenge of the Sith, **American Graffiti**, Rise of Skywalker.

**Strengths**: this set is near-ideal. Original-trilogy + prequel + sequel-trilogy entries dominate. LotR Fellowship at #5 (epic adjacency) is right. American Graffiti at #9 (Lucas's earlier film) is a nice director_signature catch.

### Toy Story (1995) `862` — active: standard_shape, prestige, franchise_dominant, studio_lineage, director_signature

V2 top 10: Toy Story 2, Toy Story 3, Monsters Inc, **Wallace & Gromit: Vengeance Most Fowl**, Toy Story 4, A Bug's Life, Cars, Toy Story of Terror!, **Small Fry**, **Partysaurus Rex**.

**Issues**:
- Positions 9–10 are Pixar shorts (verified `format_bucket = "short"` for both). Format=0 vs anchor's narrative_feature, but they're inserted via deferred remainder — see **F2**.
- Wallace & Gromit at #4 is stop-motion vs Toy Story's CG. medium_score = 0.50 → 0.925× multiplier (soft penalty, see **F3**). It's not wrong, but a more on-medium pick (Inside Out, Coco) would be expected at that slot. Notably Inside Out doesn't appear in the top 10 here at all, which is a regression from V1's behavior on related Pixar anchors.

### Spirited Away (2001) `129` — active: standard_shape, prestige, studio_lineage, director_signature

V2 top 10: Howl's Moving Castle, Ponyo, The Boy and the Heron, Castle in the Sky, My Neighbor Totoro, Princess Mononoke, Kiki's Delivery Service, The Wind Rises, Porco Rosso, The Tale of The Princess Kaguya.

**Strengths**: textbook result. Every entry is Ghibli or Miyazaki-directed. Director_signature + studio_lineage stacking did exactly what they should.

### The Godfather (1972) `238` — active: standard_shape, prestige, franchise_dominant, source_material, director_signature

V2 top 10: Godfather II, **The Brotherhood (1968)**, Road to Perdition, Character (1997), Godfather III, Apocalypse Now, GoodFellas, Lawrence of Arabia, Citizen Kane, **The Conversation**.

**Strengths**: Godfather II/III by franchise. GoodFellas at #7 (mafia adjacency) is great — quality+source carries it. The Conversation at #10 (Coppola, director_signature) is a nice catch.

**Issues**: Lawrence of Arabia at #8 and Citizen Kane at #9 are quality-dominant prestige picks with weak topical match. The middle-bucket-always-on quality lane is firing for any high-reception candidate. This was deliberate in V2 to keep middle anchors from being lane-starved, but it surfaces tonally-wrong prestige films when shape adjacency is thin.

### The Dark Knight (2008) `155` — active: standard_shape, prestige, franchise_dominant, source_material, director_signature

V2 top 10: TDK Rises, Batman (1989), The Batman (2022), **Captain America: Civil War**, Batman Begins, Batman Returns, **Batman: Year One**, **Batman: The Long Halloween Pt 2**, Batman Forever, **Oppenheimer (2023)**.

**Issues**:
- Positions 7–8 are animated Batman films (see **F3**). 15% medium penalty insufficient.
- Civil War at #4 (source = comic book, low shape, no Batman connection) is a stretch even at score 0.302. The source-IDF formula should be downweighting "comic book" → IDF ~0.2-ish since it's common in the catalog. This needs a numerical sanity-check (might be the IDF is computed correctly but the lane is still adding 0.04 weight × full IDF score, enough to push Civil War above shape-pure adjacents).
- Oppenheimer at #10 — director_signature (Nolan), but shape is low and the dominant lane reads "director". Reasonable catch but feels stretched at this rank.

### Get Out (2017) `419430` — active: standard_shape, studio_lineage, director_signature

V2 top 10: Us, Antebellum, Weapons (2025), **Arachnophobia**, Nope, **The Thing with Two Heads**, **Welcome Home Brother Charles**, Barbarian, The Blackening, Tyrel.

**Strengths**: Us at #1 and Nope at #5 are the obvious Peele picks. Antebellum, Barbarian, The Blackening are solid Black-led horror adjacents.

**Issues**: Arachnophobia at #4 is a creature feature with no thematic overlap (vector-centroid artifact). Welcome Home Brother Charles and Thing with Two Heads (both 1970s blaxploitation) are vector matches on Black-led genre fare but not "like Get Out" in the social-thriller sense. The studio_lineage anchor is active but Blumhouse's signal is no longer additive in V2 (multiplier path), so the on-brand-but-low-shape leak is largely fixed compared to V1.

### John Wick (2014) `245891` — active: standard_shape, franchise_dominant, studio_lineage, director_signature

V2 top 10: John Wick 3, John Wick 2, Ninja: Shadow of a Tear, The Protégé, John Wick 4, **Ballerina**, **Wick Is Pain**, Darc, Polar, Rambo: Last Blood.

**Strengths**: every John Wick sequel + Ballerina (Wick-universe spinoff) appear, plus thematically-matched stylized hitman films (Polar, Darc, The Protégé).

**Issues**: `Wick Is Pain (2025)` at #7 — looks like a Wick documentary based on the title. Worth verifying its format bucket; if it's a docu it shouldn't be at #7 for a narrative-feature anchor.

### Oppenheimer (2023) `872585` — active: standard_shape, prestige, source_material, director_signature

V2 top 10: **Einstein and the Bomb (2024)**, **Before Night Falls (2000)**, **Oppenheimer: The Real Story (2023)**, **The Trials of J. Robert Oppenheimer (2008)**, Fat Man and Little Boy, The Dark Knight, Dunkirk, Inception, **The Day After Trinity (1981)**, Memento.

**Major issue (see F1)**: positions 1, 3, 4, 9 are *literal documentaries* about Oppenheimer / atomic-bomb history. Position 2 is a DOCUDRAMA biopic. Why? Oppenheimer the anchor has the `DOCUDRAMA` tag → its `format_bucket` = `documentary`. The top-5 format lock then *requires* documentary-bucket candidates. Verified directly:

```
872585 Oppenheimer            bucket=documentary  format_tags=['DOCUDRAMA']
1168709 Einstein and the Bomb  bucket=documentary  format_tags=['DOCUMENTARY']
1142906 Oppenheimer: The Real Story bucket=documentary  format_tags=['DOCUMENTARY']
401783 Trials of Oppenheimer  bucket=documentary  format_tags=['DOCUMENTARY']
59314  Day After Trinity      bucket=documentary  format_tags=['DOCUMENTARY']
5001   Before Night Falls     bucket=documentary  format_tags=['DOCUDRAMA']
```

Positions 6–10 (Dark Knight, Dunkirk, Inception, Memento) are correctly Nolan-director adjacents but the actual content of "movies like Oppenheimer" — narrative biopics about scientists / historical figures — doesn't appear because the format lock ate the top.

This is the most severe single failure in V2. Fix is **F1**: remove DOCUDRAMA / TRUE_CRIME from the documentary bucket.

### Barbie (2023) `346698` — active: standard_shape, director_signature

V2 top 10: Swag (2024), I Am Not an Easy Man, Patch Town, Bliss (1985), Poor Things, The Dressmaker, Titina, **Tiny Shoulders: Rethinking Barbie**, **Barbie Nation: An Unauthorized Tour**, Christmas on Cherry Lane.

**Issues**: positions 8 and 9 are Barbie documentaries (no `format` lane in their candidate_sources; they're surfacing via shape + quality only). Poor Things at #5 is the only mainstream feminist-satire match. The Barbie franchise lane is correctly suppressed (low-confidence — the multiplier path handles direct-to-DVD Barbie kids' films), so V1's failure mode of Barbie kids' films dominating is fixed. But the centroid lands in odd-feminist-comedy territory and the result is mostly obscure picks. This is more a vector-quality limitation than a V2-logic bug.

### Sharknado (2013) `205321` — active: standard_shape, cult_garbage, franchise_dominant

V2 top 10: every Sharknado sequel + Mega Python vs. Gatoroid + Piranha 3DD + Leprechaun 4 + The Meg + Disaster Movie. Cult_garbage anchor type and franchise lane both firing. Excellent result.

### The Room (2003) `17473` — active: standard_shape, cult_garbage

V2 top 10: Space Mutiny, The Minis, Troll 2, Betrayed, This Is Me…Now, A Complete History of My Sexual Failures, Alex the Ram, Clerk, Birdemic: Shock and Terror, Movie 43.

**Strengths**: Troll 2 and Birdemic — the canonical "so bad it's good" cohort — appear (though Troll 2 should arguably be #1 instead of #3). Space Mutiny is another classic cult-bad pick.

**Issues**: positions 2, 4, 5, 7, 8 are obscure dramas that the centroid finds via low-budget-amateur-acting signal, not true cult adjacents. The `cult_garbage` quality lane is doing real work, but the candidate pool from shape is mostly noise.

### The Green Mile (1999) `497` — active: standard_shape, source_material, director_signature

V2 top 10: The Whale, Last Light, The Last Appeal, Joint Security Area, **Dead Man Walking**, **The Shawshank Redemption**, **The Mist**, The Killing Fields, The Majestic, Buried Alive.

**Issues (see F4)**:
- The two obvious matches — Shawshank (same director Frank Darabont, prison drama) and The Mist (same director, Stephen King) — are at #6 and #7, behind The Whale, Last Light, The Last Appeal, and Joint Security Area.
- Dead Man Walking at #5 is a great catch (death-row drama, source). But it scores 0.556 — nearly twice the surrounding shape-only entries — yet still slots behind random shape matches. That's the additive sum doing what it should; the issue is the shape lane is finding tonally-wrong "drama" adjacents.
- Director_signature is listed as active here (so at least one of Green Mile's director term IDs has strength ≥ 0.80 in the MV — possibly through a multi-credit term), but the surfacing of Shawshank/Mist still feels weaker than it should be.

### Pulp Fiction (1994) `680` — active: standard_shape, prestige, studio_lineage, director_signature

V2 top 10: Reservoir Dogs, Sin City, Thursday, Amores Perros, Kill Bill: The Whole Bloody Affair, Django Unchained, Once Upon a Time in Hollywood, Jackie Brown, Go (1999), Kill Bill Vol. 2.

**Strengths**: every Tarantino feature appears (or its director-influenced peers). Director_signature is doing its job. Sin City (Tarantino guest-directed segment) and Go (Tarantinoesque) are solid.

### Fight Club (1999) `550` — active: standard_shape, studio_lineage, source_material, director_signature

V2 top 10: **Luster (2010)**, The Game, Revolver, Freeze Frame, Magical Girl, Serenity (2019), The Girl with the Dragon Tattoo, By Deception, Open Your Eyes, Fluxx (2025).

**Issues (see F4)**: Fincher films (The Game, Girl with the Dragon Tattoo) are present at #2 and #7, but the top is dominated by obscure shape-only matches (Luster, Revolver, Freeze Frame, Magical Girl). Even with `director_signature` active for Fincher, the +0.10 director-lane delta isn't enough to push his films past the centroid's preferred neighborhood. This is a pattern: V2's director boost is calibrated for "appear in top 5", not "dominate when the anchor's identity is auteur-driven".

### Back to the Future (1985) `105` — active: standard_shape, prestige, franchise_dominant, director_signature

V2 top 10: BTTF Part II, Bill & Ted's Excellent Adventure, Time Rewind, Action Replayy, BTTF Part III, Forrest Gump, Back in Time, Romancing the Stone, Cast Away, Death Becomes Her.

**Strengths**: BTTF II/III by franchise, Bill & Ted (genre adjacent), Romancing the Stone (Zemeckis early film, director). Forrest Gump and Cast Away (Zemeckis dramas) appear via director_signature.

**Issues**: Action Replayy at #4 (Bollywood time-travel comedy) is a vector artifact — but actually a reasonable thematic match for "time travel comedy".

### Interstellar (2014) `157336` — active: standard_shape, director_signature

V2 top 10: Gravity, Ad Astra, The Wandering Earth, Contact, Star Trek: The Motion Picture, Inception, Mission to Mars, The Midnight Sky, The Wandering Earth II, Green Dolphin Street.

**Strengths**: Gravity, Ad Astra, Contact, Wandering Earth — all hard-sci-fi space dramas, exactly right. Inception at #6 (director_signature, Nolan) is good.

**Issues**: `Green Dolphin Street (1947)` at #10 is wildly off-genre (1940s romance). It's surfacing via centroid drift in some space — `production` or `viewer_experience`? Not a critical issue but a clear vector-centroid artifact.

### LotR: Fellowship (2001) `120` — active: standard_shape, prestige, studio_lineage, source_material, director_signature

V2 top 10: Two Towers, Return of the King, Hobbit: Desolation of Smaug, LotR (1978), Hobbit: An Unexpected Journey, Hobbit: Battle of Five Armies, King Kong (2005), **They Shall Not Grow Old (2018)**, Star Wars (1977), The Lovely Bones.

**Issues**: They Shall Not Grow Old at #8 is a Peter Jackson WW1 documentary (verified `bucket = documentary`). It rides into the top via director + quality after the format top-5 lock has expired. This is **F2** in action. Otherwise this set is very strong.

### Titanic (1997) `597` — active: standard_shape, director_signature

V2 top 10: Titanic (1953), Green Dolphin Street, S.O.S. Titanic, One Way Passage, Ghosts of the Abyss, Luxury Liner, Avatar (2009), The Abyss, Aftershock, Aliens.

**Issues**:
- Positions 1–4, 6, 9 are all "ship/ocean disaster" films via the production vector. V2 lowered `production` weight from 0.65 → 0.30 in single-anchor base, but it's still pulling Titanic toward the literal ship-disaster cluster instead of the romantic-epic cluster.
- Positions 5, 7, 8, 10 are Cameron's filmography (director_signature). Avatar / The Abyss are reasonable; Aliens / Ghosts of the Abyss less so for "movies like Titanic".
- Missing: The Notebook, Atonement, Pearl Harbor, Cold Mountain — the actual romantic-epic peers. These would need a lane that matches "doomed romance + period drama" specifically; vectors don't capture it.

This is a known limitation noted in the V2 spec; I'd call it a "lower production weight further or zero it out for romance-tagged anchors" candidate.

---

## Multi-anchor results

### Nolan trio (Inception, Memento, The Prestige) — director-heavy

V2 top 10: Mulholland Drive, Shutter Island, Trance, Buster's Mal Heart, The I Inside, I'm Thinking of Ending Things, Mentally Apart, **Eternal Sunshine of the Spotless Mind**, Extracted, Chaos.

V2 changes vs V1: Eternal Sunshine entered (great catch via themes + format + specific_award lanes). Following (1999) — Nolan's debut — *dropped out* (was V1 #9).

**Findings**:
- Themes lane is firing strongly here (most candidates show themes-lane contribution 0.4–0.8). cohesion = 1.5, normalized weight 0.079.
- Following has director_score = 1.0 but probably very low shape (~0.20–0.25). Shape weight is now 0.658 normalized, so a 0.20 shape contributes 0.13 — not enough to crack top 10 against shape-strong centroid hits.
- Specific_award lane is firing (Mulholland Drive, I'm Thinking, Eternal Sunshine all show specific_award scores). The 3 Nolan films share Oscar nominations across directing/writing/production design, so the L1/L2 award taxonomy is creating consensus where V1 had none.

Tenet does not appear in the top 10 (it's #1 for Inception single-anchor). The Memento+Prestige centroid doesn't favor Tenet because Tenet is 2020-era big-budget action vs the smaller-scale early-Nolan style. This is the centroid behaving correctly, but it does mean missing a candidate that most users would expect.

### Pixar trio (Toy Story, Finding Nemo, Up)

V2 top 10: Toy Story 3, Inside Out, Monsters Inc, Cars, Toy Story 2, Toy Story 4, The Good Dinosaur, Onward, Ratatouille, A Bug's Life. Every entry is Pixar. Studio + themes + specific_award + format all stacking. **No issues.**

### Ghibli / Miyazaki trio

V2 top 10: Ponyo, Howl's Moving Castle, Castle in the Sky, Princess Kaguya, Arrietty, Boy and the Heron, Porco Rosso, **Pokémon 3: The Movie**, The Wind Rises, **Little Nemo: Adventures in Slumberland**.

**Strengths**: every Miyazaki/Ghibli entry; consensus_countries = `{100}` (JAPANESE keyword) is correctly identified.

**Issues**:
- Pokémon 3 at #8 (shape=0.62, themes=0.55) is a centroid artifact — anime cluster bleed. No studio match. It's not catastrophic but ranks above Wind Rises and Princess Kaguya, which is wrong.
- Little Nemo at #10 (1989 anime feature) is similar — anime adjacency without Ghibli identity.

### MCU trio (Iron Man, Avengers, Civil War)

V2 top 10: Avengers: Age of Ultron, Iron Man 2, Captain America: Winter Soldier, **The Rock (1996)**, Iron Man 3, Guardians of the Galaxy, Avengers: Infinity War, Captain America: Brave New World, Spider-Man: Far From Home, Captain America: First Avenger.

**Issues**: The Rock at #4 (Michael Bay 1996) — shape=0.60, themes=0.47, format=1.0, no franchise/source/cast/studio. Surfaces via the action-blockbuster vector cluster. With the US_DEFAULT consensus 1.10× boost, it scores 0.493. This is centroid drift; the Bay-era action cluster overlaps MCU vectors enough that one Bay film bubbles up.

Otherwise strong — every other entry is MCU.

### Stephen King horror (The Shining, IT 2017, Misery)

V2 top 10: 1408, Ghost Story (1981), Caveat, In Our Blood, Afraid of the Dark, The Forbidden Door, Nocturnal Animals, The Woods, Halloween III: Season of the Witch, Sole Survivor.

**Strengths**: 1408 at #1 (literal King adaptation, anchored on shape + themes). Source IDF has done its job — `source_score` is 0.20 max for any candidate (because King's only shared source-type with the anchors is `novel`, which collapses to ~0.20 IDF). This kills the V1 false positives like the Chinese horror "Players" that rode at 1.0 source.

**Issues**: the King-specific catalog is thin in the top 10. Pet Sematary, Carrie, Christine, Cujo — none appear. They share `novel` source but the novel-author-level data isn't tracked, so there's no signal that pulls King-specifically over generic novel-based horror. This is a known limitation noted in the V2 spec ("exact source author matching not currently available"). The themes lane is generic-horror; a specific "Stephen-King-style supernatural horror" trait would need either author-level metadata or a tighter concept tag.

### Best Picture trio (Godfather, Schindler's List, 12 Years a Slave)

V2 top 10: The Pianist, The Killing Fields, Killers of the Flower Moon, The Elephant Man, The Mission, The Shop on Main Street, Citizen Kane, Katyn, Holy Cross, The Grapes of Wrath.

**Major issue (F1)**: consensus format bucket = `documentary` because Schindler's List + 12 Years a Slave both have `DOCUDRAMA` tags. The top-5 lock then favors DOCUDRAMA/TRUE_CRIME films (The Pianist, Killing Fields, Killers of the Flower Moon, Elephant Man, Before Night Falls all have DOCUDRAMA or TRUE_CRIME). That's why The Mission (no DOCUDRAMA tag — `format=0` in the row) drops to #5 despite a strong shape score of 0.84.

**Specific_award lane is doing real work** (Killers of the Flower Moon at #3, The Killing Fields at #2, Pianist at #1 all show high specific_award scores): the 3 anchors all won Best Picture or other top Oscars, so the lane's L0 BEST_PICTURE / L2 PICTURE consensus boosts other Best Picture-tier films. **GoodFellas does not appear** in the top 10, despite being the obvious mafia-prestige-drama adjacent for The Godfather. Why not? It's narrative_feature (no DOCUDRAMA) → fails the consensus format match → format_score = 0 → no contribution from the format lane. With the DOCUDRAMA-driven format lock, GoodFellas is structurally disadvantaged.

This case alone justifies the **F1** fix.

### Cult-garbage trio (Sharknado, The Room, Birdemic)

V2 top 10: Troll 2, Space Mutiny, Plan 9 from Outer Space, Birdemic 2, Sharknado 4, Mega Shark vs. Giant Octopus, Mayday, Sharknado 2, Hard Rock Zombies, Mega Shark vs. Crocosaurus.

Nearly perfect — every entry is cult-bad-cinema or Sharknado/Mega-Shark sequels. Cult_garbage anchor type + themes lane (cult-bad-genre concept tags) carry the result. Only quibble: Mayday (2019) at #7 is a survival thriller that doesn't quite fit, but its themes score is 0.36 so it's a borderline call.

### Tarantino trio (Pulp Fiction, Reservoir Dogs, Kill Bill Vol 1)

V2 top 10: True Romance, Kill Bill 2, Lucky Number Slevin, Caliber 9, Truth or Consequences, Sin City, Inglourious Basterds, Man on Fire, Tazza: The High Rollers, Bullets Blood Cash.

**Issues vs V1**:
- V1 had Hateful Eight at #7 and Jackie Brown at #9. **Both dropped from top 10 in V2.** Hateful Eight has director=1.0 + studio=0.67 + format=1.0 + cast=? but probably modest shape (~0.30); the V2 cohesion-driven shape weight is high (0.635 normalized) so shape-light Tarantino films lose out.
- Caliber 9 (1972 Italian crime) at #4 with shape=1.00 — the centroid is heavily favoring `Caliber 9` because it's the highest-shape candidate. But Caliber 9 is not a Tarantino film and is a vector artifact.
- Man on Fire at #8 (Tony Scott) — themes=0.58 puts it ahead of more obvious Tarantino picks like Jackie Brown.

This trio shows that V2's shape-cohesion scaling — where shape grows with vector cohesion — *can over-amplify shape* relative to director when the director identity is the actual signal the user cares about. Hateful Eight should be top-3 here.

### Heist trio (Ocean's Eleven, Heat, Italian Job)

V2 top 10: Ocean's Thirteen, **Foolproof**, **The Italian Job (1969)**, Fast Five, Lift (2024), Den of Thieves, Takers, The Rise (2012), The Outfit (1973), The Heist (1989).

**V2 win**: V1 had `Players (2012)` (Bollywood novel-based action) at #2, riding the source lane at 1.0. V2 source-IDF reduces "novel" to ~0.20 contribution, and `Players` is gone from the top 10. Every entry is now a real heist film. **F2 win, no issues.**

### LotR trilogy

V2 top 10: Hobbit: Unexpected Journey, Hobbit: Battle of Five Armies, LotR (1978), Hobbit: Desolation of Smaug, **Dark Knight Rises**, Return of the King (1980), Harry Potter Deathly Hallows Pt 2, Gladiator, Star Wars (1977), King Kong (2005).

**Strengths**: shape weight reaches max (raw 1.20) thanks to per-space cohesion ≥0.97 across most spaces. Mean pairwise cosine = 0.927 → shape_raw = 0.6 × (1 + clamp((0.927-0.55)/0.30, -0.4, 1.0)) = 0.6 × 2 = 1.20. The expanded shape-cohesion mechanism is doing exactly what V2 promised.

Hobbit films + LotR (1978) animated + Return of the King (1980) animated all surface correctly via franchise + source.

**Issues**:
- Dark Knight Rises at #5 (shape=0.71, no franchise/source/director): pure centroid adjacency in the "epic action with weight" cluster. Reasonable but felt shoehorned.
- Star Wars at #9 (epic-fantasy adjacency, quality, themes, specific_award): this is also vector + thematic adjacency, and it's a defensible call, but more LotR-genre adjacents (Willow, Conan, Excalibur, Stardust) might be expected. The `themes` lane likely picks up some EPIC/FANTASY tags but the pool of LotR-flavored candidates is small.

### Chaotic mixed bag (Toy Story, Godfather, Sharknado)

V2 top 10: Toy Story 3, Star Wars, Life or Something Like It, Mission to Mars, A Bronx Tale, Star Trek III, Sharknado 4, Phantom of the Opera, The Incredibles, Sword in the Desert.

**Findings**:
- The low-cohesion fallback **did NOT fire**, despite mean_pairwise_cosine = 0.276 (< 0.35 threshold). Why? `metadata_max_cohesion = 1.69` (the prestige bucket repeats across 2 of 3 anchors — Toy Story and Godfather are both `prestige`, Sharknado is `cult_garbage`). The fallback gate requires `metadata_max_cohesion < 1.0`, so prestige consensus by 2/3 anchors keeps us in the normal flow.
- The result is dominated by Toy Story-adjacents (Toy Story 3 at #1) plus Godfather-adjacents (A Bronx Tale at #5) plus Sharknado-adjacents (Sharknado 4 at #7). It's actually a reasonable "mix every anchor" output even without the fallback.
- The Incredibles at #9 (Pixar studio doesn't repeat across anchors so studio lane = 0; surfacing via specific_award cohesion) is interesting — Star Wars + Toy Story share Picture-tier or Animated Feature awards, which might be the L2 PICTURE consensus.

### War film trio (Saving Private Ryan, 1917, Apocalypse Now)

V2 top 10: All Quiet on the Western Front (2022), Dunkirk, All Quiet on the Western Front (1930), The Thin Red Line (1964), Platoon, The Thin Red Line (1998), Fury, Kajaki, Saints and Soldiers, Journey's End. Every entry is a war film, with both All Quiet versions and both Thin Red Line versions appearing. Themes lane firing on WAR / EPIC / SOLDIER concept tags. **No issues.**

---

## Summary of recommended fixes (priority order)

1. **Move DOCUDRAMA and TRUE_CRIME out of the `documentary` format bucket.** This is the single highest-impact change. It would unilaterally fix Oppenheimer single-anchor (currently top 4 are documentaries because anchor itself is `documentary`-bucketed) and the Best Picture trio (consensus = documentary because 2/3 anchors are DOCUDRAMA-tagged). Either give DOCUDRAMA/TRUE_CRIME their own bucket(s) or fold them into `narrative_feature`. [search_v2/format_registry.py:48–65](../search_v2/format_registry.py#L48).

2. **Extend the format lock past TOP_FORMAT_LOCK = 5 slots, OR exclude `short`/`documentary` from the deferred-remainder pool when the anchor is `narrative_feature`.** Fixes Toy Story (Pixar shorts at #9–10), Matrix (Animatrix shorts at #7–10), LotR (They Shall Not Grow Old at #8). [search_v2/similar_movies.py:1016–1072](../search_v2/similar_movies.py#L1016) and the `(top + remainder)[:limit]` line at 1072.

3. **Lower the medium-multiplier floor for live-action vs. animation crossings**, or apply medium as a hard gate inside the format lock rather than a soft post-additive multiplier. Currently `0.85 + 0.15 * medium_score` floors at 15 % penalty — not enough to keep animated Batman films out of TDK's top 10. [search_v2/similar_movies.py:213–214](../search_v2/similar_movies.py#L213).

4. **Increase director-lane influence when `director_signature` is active**, or floor director_score for moderate-strength directors. Current `+0.10` delta is too small to push auteur films past obscure shape adjacents (Fight Club's Fincher films, Green Mile's Darabont films, Tarantino trio's Hateful Eight + Jackie Brown). [search_v2/similar_movies.py:181–189](../search_v2/similar_movies.py#L181).

5. **Verify source-IDF for `comic book`** — Civil War surfacing in Dark Knight at #4 with low shape suggests the comic-book IDF may still be high enough to over-credit unrelated comic adaptations. Worth dumping the actual IDF values for sanity-check.

6. **Consider raising the `metadata_max_cohesion` bar for the low-cohesion fallback** from 1.0 to ~1.5. The prestige-2-of-3 case (Chaotic mixed bag) is borderline — `metadata_max_cohesion = 1.69` keeps it in the normal flow but the result still feels like a fallback would have been cleaner. [search_v2/similar_movies.py:236](../search_v2/similar_movies.py#L236).

7. **Long-term: track source author** (Stephen King-specific signal) and **add a romance-period lane / decay production weight further for romance anchors** (Titanic). Both are noted as out-of-scope in the V2 spec — flagging here for V3.

---

## Test execution notes

- Single-anchor results: `python -m search_v2.run_similar_movies_batch --limit 10 --json-out search_v2/similar_movies_v2_batch.json --markdown-out search_v2/similar_movies_v2_batch.md`
- Multi-anchor results: ran via `/tmp/run_multi_anchor.py` (custom script that drives `run_similar_movies_for_ids` over the 12 anchor sets from V1 and writes [search_v2/similar_movies_v2_multi_anchor.md](../search_v2/similar_movies_v2_multi_anchor.md)).
- All anchor sets match the V1 evaluation files [search_v2/similar_movies_batch_results.md](../search_v2/similar_movies_batch_results.md) and [search_v2/similar_movies_multi_anchor_results.md](../search_v2/similar_movies_multi_anchor_results.md) for direct comparability.
