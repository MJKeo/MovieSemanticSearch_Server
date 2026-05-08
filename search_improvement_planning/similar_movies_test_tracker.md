# Similar Movies Test Tracker

This document tracks test iterations for the similar-movies search
system over time. The goal is to ensure we are always trending in the
right direction — that each calibration / architecture / recall change
either fixes an identified failure mode without introducing new ones,
or surfaces a new class of issue worth addressing in the next pass.

The benchmark is the harness in
[`search_v2/run_similar_movies_batch.py`](../search_v2/run_similar_movies_batch.py)
run against the anchor and cohort set documented in
[`similar_movies_test_set.md`](similar_movies_test_set.md). Diagnostic
output (`base_score`, per-lane `raw→contribution`, multipliers, floor
activations) makes per-row reasoning inspectable without reading lane
code.

## Current failure cases (as of V3, pre-V3.1 verification)

These are the failure modes the most recent diagnostic-enabled smoke
run surfaced. V3.1 has been built but **not yet verified** — the next
test iteration will be the first to evaluate these against a re-run.

1. **Barbie's top 10 misses obvious thematic neighbors.** Lady Bird
   sits at #30 (`director=0.20→0.20` firing but no shape match);
   The Favourite at #28 (themes raw 0.139 → contribution 0.008);
   *I Am Not an Easy Man* depressed by `×0.75` country penalty
   despite themes raw 0.305. Female-empowerment / Gerwig adjacents
   fail to surface even when the lanes that should signal them are
   working as designed.

2. **Themes lane is sub-perceptual at current weight (0.06).** Real
   thematic matches scoring `themes=0.30–0.95` raw contribute
   `0.018–0.057` to combined score — invisible against shape (~0.50
   weight). The lane is finding the right matches; the budget given
   to it can't surface them.

3. **Single-anchor director lane has no floor.** Multi-anchor has a
   0.35 floor at `M_d/N ≥ 0.75 AND shape ≥ 0.40`. Single-anchor only
   adds a flat 0.20, which can't compensate when the anchor's
   embedding doesn't match an auteur sibling (Barbie ↔ Lady Bird
   shape ≈ 0.0). Auteur-curated single-anchor matches deserve floor
   protection too.

4. **Rare-keyword floor never activates.** Even after the V3 →
   recalibrated tier thresholds (which raised lane scores from ~0.020
   to 0.10–0.55), zero floor activations across the entire 21-anchor
   + 14-cohort smoke. The 0.30 shape gate blocks every observed
   high-rare-keyword candidate (Schindler shape=0.030, Dunkirk=0.160).
   Either the gate is too tight or the floor is reserved for an
   un-encountered case.

5. **Themes lane is purely a re-ranker — no candidate generation.**
   A thematically-aligned candidate that isn't in Qdrant's top-K
   (default 500) and isn't pulled by director / franchise / studio /
   source / rare-medium has no path into the pool. Lady Bird vs.
   Barbie illustrates this: the embedding doesn't put it in top 500,
   so themes / director never gets a shot at scoring it.

6. **Aggregate tag co-occurrence under-rewarded.** Themes treats each
   shared trait linearly — matching 5 moderate-IDF tags is the same
   as 5 × matching 1 tag. But random-pair sampling shows shared
   moderate+high IDF p99 = 0.000, meaning matching 3+ moderate tags
   together is qualitatively rare and should carry a bigger signal.

7. **Cross-tradition films suppressed by country penalty when they're
   the right thematic match.** I Am Not an Easy Man (French, gender-
   flip comedy) is the most thematically-aligned non-Barbie film in
   the top 30 but the `×0.75` country penalty pushes it down. The
   penalty does correct work for Telugu / Bollywood mismatches; the
   problem is that themes is too weak to override it when warranted.

A round of testing on the V3.1 changes will run soon to gather the
current state of affairs and start populating iterations below.

## Iteration format

Each test iteration documents one cycle of (hypothesis → change →
observation → next-step). Sections:

- **Hypothesis**: what we expected the change to fix, and why we
  believed it would. State the failure case being targeted and the
  mechanism (lane / multiplier / floor / recall path) that should
  address it.
- **Changes actually made**: the specific code / config / data edits
  shipped in this iteration. Reference file paths, constant names,
  and PR / commit links where applicable. Distinguish "shipped" from
  "decided but deferred."
- **Observations**: what the harness re-run actually showed.
  - Key patterns (e.g., "rare_keyword now fires on 10/10 candidates
    for Best Picture trio").
  - Failure modes resolved (which items from the failure-case list
    above can be crossed off, and the evidence).
  - New failure modes detected (regressions, unexpected
    interactions, surprises that weren't in the hypothesis).
- **Ways to improve going forward**: concrete follow-ups for the next
  iteration. Bias toward small, single-axis changes — the diagnostic
  output is most useful when only one knob moved between iterations.
  If a follow-up needs design discussion (rather than just a knob
  twist), call that out explicitly.

## Iterations

### V3.1 — Calibration + recall expansion (2026-05-07)

#### Hypothesis

Each of the 7 V3.1 changes targeted a specific failure case:

1. **Themes weight 0.06 → 0.12**: should make thematic raw scores
   (0.30–0.95) contribute perceptibly (~0.04–0.11) to combined score,
   surfacing matches like *I Am Not an Easy Man* / *The Favourite*
   for Barbie that currently die at sub-0.020 contributions.
2. **Rare-keyword floor shape gate 0.30 → 0.20**: should let the
   floor fire on borderline-shape candidates with strong distinctive-
   match evidence (Schindler's-style, shape=0.030).
3. **Single-anchor director floor (0.35 mag, 0.20 shape gate)**:
   should pull Lady Bird and The Favourite into Barbie top 10 when
   the auteur (Gerwig) match fires but shape doesn't align.
4. **Moderate-combo bonus on rare-keyword lane**: should reward 3+
   moderate-tag co-occurrence so candidates like Pleasantville
   (Barbie's meta-fiction territory) climb.
5. **Themes-recall candidate fetch (single + multi)**: should put
   thematically-aligned candidates that miss Qdrant top-K into the
   pool via `single_idf >= 0.55 OR combo_sum >= 0.50` SQL gate.
6. **Multi-anchor consensus traits**: cohort variant of (5) with
   IDF-scaled cohesion bar (LOW=1.0, MOD=0.67, HIGH=0.5).
7. **Qdrant `DEFAULT_QDRANT_LIMIT` 500 → 2000**: should catch
   vector-distant matches that even the targeted recall path can't
   help (Lady Bird shape ≈ 0 against Barbie).

#### Changes actually made

All 7 shipped (no deferrals). Code touched:

- [`search_v2/similar_movies.py`](../search_v2/similar_movies.py):
  `BASE_LANE_WEIGHTS["themes"]` 0.06 → 0.12; `DEFAULT_QDRANT_LIMIT`
  500 → 2000; `RARE_KW_FLOOR_SHAPE_GATE` 0.30 → 0.20; new
  `RARE_KW_COMBO_*` constants + combo bonus inside
  `_rare_keyword_score_for_traits`; new `DIRECTOR_FLOOR_SINGLE_*`
  constants + per-flow floor params threaded through
  `_build_results`; new `_multi_anchor_consensus_themes_traits`
  helper with `THEMES_RECALL_COHESION_BAR_*` constants; multi-anchor
  themes IDF fetch moved into the first parallel gather; multi-anchor
  themes weight now reads from `BASE_LANE_WEIGHTS` instead of
  hardcoded 0.06.
- [`db/postgres.py`](../db/postgres.py): new
  `fetch_movie_ids_by_themes_recall` — single SQL aggregate, joins
  `movie_card` array columns to `mv_trait_idf` per kind, GROUP BY +
  HAVING gates on single-trait or all-tier sum IDF.
- Single-anchor and multi-anchor flows wire themes-recall into their
  `candidate_ids` unions.

Doc updates: V3.1 §6/7 + constants matrix + Expected Outcomes
in [`similar_movies.md`](similar_movies.md);
[`similar_movies_test_set.md`](similar_movies_test_set.md) created;
this tracker created.

#### Observations

##### Failure modes resolved

- **Failure case #2 (themes lane sub-perceptual)** — RESOLVED. Themes
  contributions are now visibly material:
  - Schindler's vs Best Picture trio: themes=1.000→0.107 (was 0.057).
  - Pianist vs Best Picture trio: themes=0.726→0.107 (was 0.065).
  - I Am Not an Easy Man vs Barbie: themes=0.305→0.035 (was 0.018).
  - Tenet vs Inception: themes=0.430→0.054 (was 0.029).
- **Failure case #6 (aggregate co-occurrence under-rewarded)** —
  RESOLVED. Combo bonus produces concrete uplifts:
  - Schindler's vs Best Picture trio: rare_keyword=0.449 (was 0.319).
  - Empire Strikes Back vs Star Wars: rare_keyword=0.556 (was 0.556 — already at high tier ceiling pre-V3.1; combo bonus mostly affects mid-tier candidates).
  - Dunkirk vs Oppenheimer: rare_keyword=0.321 (was 0.228).
  - Last Action Hero vs Barbie: rare_keyword=0.206 (was 0.137).
  - Pleasantville vs Barbie: rare_keyword=0.202 (newly visible).

##### New / persistent failure modes

- **Failure case #1 (Barbie misses Gerwig adjacents)** — PARTIALLY
  RESOLVED, partially worsened. *I Am Not an Easy Man* moved from
  beyond top 10 to **#7** (themes weight bump did its job). But
  **Lady Bird is at #39, Little Women at #43, The Favourite at #28**
  — still not in top 10. Root cause exposed by diagnostics: Lady
  Bird's `shape=0.033` is **below the new 0.20 director-floor gate**,
  Little Women's `shape=0.077` ditto. The V3.1 single-anchor director
  floor as designed doesn't help auteur matches whose embedding is
  truly disjoint from the anchor — exactly the case it was supposed
  to address.

- **Failure case #3 (single-anchor director floor)** — DESIGNED FIX
  DOESN'T REACH ITS TARGET. The 0.20 shape gate is still too tight
  for Gerwig-vs-Barbie. Floor never activates anywhere in the harness
  (see next item).

- **Failure case #4 (rare-keyword floor never activates)** —
  PERSISTENT, BUT NOW REDUNDANT. **Zero floor activations across the
  entire 21-anchor + 14-cohort suite** (single OR multi). The
  recalibrated rare-keyword tier scoring + combo bonus now produces
  additive sums that already exceed any floor magnitude whenever the
  shape gate would clear. The floors became architecturally
  redundant: candidates that would have needed a floor (high
  rare-keyword, low shape) still fail the gate; candidates that
  clear the gate don't need the floor. All four V3.1 floor mechanisms
  (rare-keyword, cast, director-multi, director-single) are
  effectively dead code in the observed regime.

- **Failure case #5 (themes recall)** — PARTIALLY RESOLVED, with
  recall pollution. The recall path is firing — diagnostics show
  candidates entering via themes-recall that vector search would have
  missed (e.g., *Pleasantville*, *Stranger Than Fiction*, *Free Guy*
  for Barbie). But the all-tier SUM gate is too permissive: the
  Barbie top 10 now contains:
  - **#8 Man in Outer Space (1962)** — `rare_keyword=0.315 dominant`,
    obscure foreign film surfaced by tag overlap.
  - **#10 You Are a Widow, Sir (1971)** — same pattern, also #18-#33
    has 8+ similarly obscure titles.

  These candidates have weak shape but enough random low/mid-tier
  tag overlap (combined sum ≥ 0.50) to enter the pool, where the
  combo bonus + themes weight then propels them past more-relevant
  candidates whose rare-keyword overlap is smaller. The user's
  pre-flight call ("include all-tier IDFs, see what happens") shows
  its predicted risk: pool explodes with obscure long-tail
  candidates. **First V3.1 issue worth tightening.**

- **Failure case #7 (country penalty)** — UNCHANGED. *I Am Not an
  Easy Man* now at #7 (was #8) — improvement from themes weight
  bump, but the `×0.75` country penalty still drops its 0.661 base
  to 0.496 final. With themes carrying real weight now, the penalty
  is the sole barrier to it cracking the top 5.

##### New patterns / surprises

- **Score inflation is widespread.** Top scores routinely exceed 1.0
  due to combo bonus + themes weight + country boost stacking:
  - Empire Strikes Back vs Star Wars: 1.820 (was 1.672)
  - Mulholland Drive in Nolan trio: 1.422
  - Pianist vs Best Picture trio: 1.319
  - Iron Man 2 in MCU trio: 1.323 (was 0.999)
  - Toy Story 3 in Pixar trio: 1.242 (was 1.212)
  - Antebellum vs Get Out: 1.175

  The "score in [0,1]" semantic is broken. Acceptable for debug, but
  downstream consumers (LLM rerankers, threshold-based gates) need
  to know.

- **Recall pollution from rare-keyword recall path.** The new SQL
  fetcher pulls candidates whose shared anchor traits sum to 0.50+
  in IDF — but many qualifying candidates are obscure foreign films
  whose tag overlap is incidental rather than thematic. Manifests
  most visibly on Barbie (broad keyword pool) and Toy Story (where
  Wallace & Gromit, Boss Baby, Spider-Verse displaced Cars / Inside
  Out / Up that were in V3 top 10).

- **Some V3 wins shifted unexpectedly:**
  - Tom Hanks trio lost The Terminal and Cast Away from top 10
    (V3 had 4 Hanks vehicles, V3.1 has 3 — all Toy Story sequels).
    Score inflation from rare_keyword/themes pushed family-shape
    candidates above Hanks-vehicle-via-cast.
  - Get Out lost *Nope* from top 10 (Peele auteur). Now: Antebellum,
    Us (Peele), Weapons, Last Shift, Barbarian, Dead End, Pandorum,
    Wes Craven's New Nightmare, Candyman, Autopsy of Jane Doe.
    Social-horror shape candidates outranked the Peele sibling.

- **Confirmed wins (no regression, often improvement):**
  - Inception: Tenet, Interstellar, Memento, Mulholland Drive,
    Trance — added 2001: A Space Odyssey and The Matrix.
  - Best Picture trio: clean prestige biopic cluster, no docs.
  - MCU trio: 9/10 are MCU films.
  - Star Wars: 6 same-saga films + 2 Star Trek + Fifth Element +
    Last Starfighter (sci-fi adjacents).
  - Female-led / Gerwig multi-anchor: Frances Ha, 20th Century
    Women, Juno, The Spectacular Now, Little Women (1994) all
    surface — themes recall + weight bump working as designed for
    cohort-level coherence. Notable that the same fixes are LESS
    effective in single-anchor Barbie because there's no consensus
    pool to anchor on.

#### Ways to improve going forward

In rough priority order:

1. **Tighten the themes-recall combo gate.** The all-tier SUM
   threshold is letting in obscure candidates with incidental tag
   overlap. Two single-axis options:
   (a) Filter the SUM to moderate+high tier (idf ≥ 0.30) only —
       aligns with the combo bonus's tier semantics.
   (b) Raise the all-tier SUM threshold from 0.50 to 0.80 or 1.00.

   Recommend (a) — more principled.

2. **Decide whether to keep the V3.1 floors.** Zero activations
   across the entire suite means they're either ornamental or
   reserved for cases the harness doesn't yet exercise. Single-axis
   options:
   (a) Drop the floor mechanisms entirely; lane scoring carries the
       intended behavior already.
   (b) Lower the magnitudes (e.g., 0.35 → 0.45) so they can displace
       additive scores in the observed range.
   (c) Drop the shape gate on the single-anchor director floor —
       a curated auteur match is itself a strong-enough signal.

   Recommend (a) plus (c) for the single-anchor director case
   specifically — Lady Bird at #39 is the load-bearing failure mode.

3. **Address score inflation.** Final scores routinely exceed 1.0,
   stacking country boost (×1.05) + studio multiplier (×1.10) + combo
   bonus + bumped themes onto an already-strong shape. Cleanest fix:
   normalize the final score to [0,1] after all multipliers / floors
   resolve. Alternative: cap at 1.0 (loses precision among top
   matches).

4. **Country penalty softening when themes is strong.** Already
   discussed pre-V3.1, deferred. Now that themes carries weight and
   *I Am Not an Easy Man* sits at #7 with a `×0.75` penalty visible
   in the breakdown, this is testable: if `themes_raw >= 0.30`,
   dampen the cross-tradition penalty (e.g., `×0.90` instead of
   `×0.75`).

5. **"Director-only candidate" pathway.** Lady Bird vs. Barbie has
   `shape=0.033` (essentially noise) and `director=0.20` is the only
   real signal. The combined score of 0.392 is correct given current
   weights but the *intent* of the user anchoring on a Gerwig film is
   "show me other Gerwig films." Consider a special-case pathway:
   when a candidate's only meaningful lane is `director` (curated
   auteur, no other contribution > 0.05), promote them with a higher
   floor — without the shape gate that's currently blocking the case.

6. **Investigate the Tom Hanks / Get Out regressions.** Both lost
   "anchor-canonical" matches (The Terminal, Nope) due to score
   inflation lifting peer-shape adjacents above them. Likely needs
   no architectural fix once score normalization (#3) lands — the
   relative gap is what matters, not the absolute score.

7. **Consider trimming the rare-keyword recall pool by applying a
   quality gate.** "Man in Outer Space (1962)", "A Nice Plate of
   Spinach (1977)" surfaces because the SQL doesn't filter by
   ingestion-quality. A light minimum-popularity / minimum-vote-count
   filter on the recall fetch would prune the long-tail noise without
   affecting any of the known good matches.

Bias for next iteration: **single-axis change**. Recommend starting
with #1 (tighten combo gate) — it's the biggest perceived-quality
hit on Barbie and a clean knob twist. The diagnostic output will
make the deltas inspectable per-row.

### V3.2 — Tighten themes-recall combo gate to moderate+high tier (2026-05-07)

#### Hypothesis

The V3.1 themes-recall SQL gate uses `SUM(idf) >= 0.50 OR MAX(idf) >= 0.55`
where the SUM includes **all-tier** IDFs. Smoke run showed this is
too permissive — Barbie's top 10 picked up obscure foreign films
("Man in Outer Space (1962)" #8, "You Are a Widow, Sir (1971)" #10)
whose shared anchor traits are mostly low-tier (common keywords)
that coincidentally accumulate to >=0.50. The combo bonus's tier
semantics already filter to `idf >= RARE_KW_TIER_LOW_MAX (0.30)`;
the recall gate should align.

**Hypothesis**: filtering the recall SUM to `idf >= 0.30` (moderate
+ high tier) only will:
- Prune the long-tail incidental-overlap pollution (Man in Outer
  Space, You Are a Widow Sir, A Nice Plate of Spinach) from
  Barbie's top 10.
- Preserve genuine multi-moderate-tag matches like Pleasantville,
  Stranger Than Fiction, Free Guy that surfaced legitimately via
  themes recall.
- Leave franchise / studio / director-driven cohorts (Star Wars,
  MCU, Pixar, Best Picture trio, Inception) unchanged — those
  pools weren't dependent on themes recall in the first place.
- Not affect the `MAX(idf) >= 0.55` single-trait gate at all,
  which preserves the "Manhattan Project" rare-keyword path
  (single uniquely-rare trait can still pull a candidate).

The combo bonus already proved that aggregate moderate+ tier
co-occurrence is a meaningful signal; the recall gate should use
the same lens to decide who enters the pool.

#### Changes actually made

Single-axis: aligned the recall SUM gate's tier semantics with the
combo bonus.

- [`db/postgres.py`](../db/postgres.py): added new parameter
  `combo_sum_min_idf: float = 0.30` to
  `fetch_movie_ids_by_themes_recall`. Modified the `HAVING` clause:
  ```sql
  HAVING SUM(idf) FILTER (WHERE idf >= %s) >= %s
      OR MAX(idf) >= %s
  ```
  The new param threads through ahead of `combo_sum_threshold` and
  `single_idf_threshold`. Default of 0.30 matches
  `RARE_KW_TIER_LOW_MAX`. Single-trait MAX gate is untouched.

No call-site changes needed: both
`_run_single_anchor_similarity` and `_run_multi_anchor_similarity`
use the function with default arguments, so the new tier filter
applies automatically.

#### Observations

Hypothesis was **partially correct, partially wrong**. Multi-anchor
cohorts benefit materially; single-anchor Barbie noise persists
because the suspected mechanism (low-tier accumulation) is not what
those candidates were exploiting.

##### Wins / failure modes improved

- **Pixar trio (multi) — MAJOR WIN.** V3.1 had Wallace & Gromit
  (#4), Boss Baby (#9), Spider-Verse (#10) displacing core Pixar
  films. V3.2 top 10 is **Toy Story 3, Toy Story 2, Inside Out,
  Monsters Inc, The Good Dinosaur, Toy Story 4, Onward, Bolt,
  Wreck-It Ralph, Ice Age** — 7/10 Pixar with the rest being
  family-animation (Bolt/Wreck-It Ralph/Ice Age are reasonable
  thematic peers). The non-Pixar pollution was filtered out by the
  consensus pool's tightened gate.
- **Best Picture trio (multi)** preserved: Pianist, Mission,
  Killing Fields, Killers of the Flower Moon, Citizen Kane in top 5.
- **MCU trio (multi)** preserved: 9/10 MCU films, only outlier is
  The Rock at #4 (1996 Bay action, themes-adjacent).
- **Female-led / Gerwig (multi)** preserved+improved: Little Women
  (1994), 20th Century Women, Juno, Frances Ha, Spectacular Now,
  Jojo Rabbit all in top 10. The Fabelmans at #1 — strong
  coming-of-age peer.
- **Inception, Star Wars, Oppenheimer single-anchor** unchanged
  (anchor types not dependent on themes-recall).

##### Hypothesis was WRONG for these cases

- **Barbie single-anchor: Man in Outer Space (#8) and You Are a
  Widow, Sir (#10) still present.** Diagnostic shows
  `rare_keyword=0.315 / 0.316` and `themes=0.732 / 0.735` — these
  candidates share genuinely-moderate-tier traits with Barbie, not
  just low-tier accumulation. They pass the tightened
  `idf >= 0.30` filter because their shared trait set IS at
  moderate+ tier. The hypothesis assumed the noise was low-tier;
  reality is they coincidentally share several moderate-tier traits
  with Barbie's tag-rich anchor (likely satire / surreal /
  fairy-tale type tags). This is recall-quality noise, not
  recall-mechanism noise.
- **Toy Story single-anchor unchanged.** Wallace & Gromit (#4),
  Boss Baby (#9), Spider-Verse (#10) still surface. Their
  `themes=0.573 / 0.973 / 0.755` confirms they have genuine
  moderate+high-tier trait overlap (animation + family + buddy +
  toy/spirit-of-friendship type tags). Same root cause as Barbie:
  these candidates pass any IDF-tier-based gate because they
  legitimately share moderate-tier traits — they're just not
  *Pixar*. Single-anchor lacks the cohesion filter that multi-anchor
  Pixar cohort uses to prune them.

##### Persistent issues unrelated to combo gate

- **Lady Bird (#39), Little Women (#43), The Favourite (#28) for
  Barbie** — same as V3.1. Shape ≈ 0 against Barbie embedding
  blocks them from the director-floor shape gate at 0.20. Combo
  gate change has zero leverage on this — they're not in the
  candidate pool because no recall path catches them.
- **Tom Hanks trio (multi)** still loses Cast Away, The Terminal,
  Captain Phillips, Apollo 13 — same regression as V3.1. Cause is
  score inflation lifting family-animation candidates above
  Hanks-vehicle-via-cast, not recall pollution.
- **Get Out** still missing Nope — Peele auteur. Not a recall
  problem; Nope just isn't catching enough lanes.
- **Score inflation persists**: Empire 1.820, Iron Man 2 1.323,
  Pianist 1.319, Toy Story 3 (Pixar trio) 1.242. Combo gate change
  doesn't address multipliers stacking.

##### New patterns / surprises

- **Single vs multi asymmetry is now stark.** Multi-anchor
  consensus + cohesion-IDF tradeoff prunes thematically-incidental
  candidates well; single-anchor has no such filter and admits
  candidates that share genuinely-moderate-tier traits but aren't
  thematically-aligned. The combo bonus's tier-aligned recall gate
  helps multi but doesn't help single — single needs a different
  signal (popularity / quality / vote-count gate, or
  same-tradition cohesion test).
- **Barbie #5 and #6 are documentaries about Barbie itself**
  ("Tiny Shoulders: Rethinking Barbie", "Barbie Nation: An
  Unauthorized Tour"). They're entering via franchise-dominant lane
  (anchor name → docs about Barbie). The format-bucket boost was
  supposed to suppress documentaries; investigate whether
  franchise-name lookup is bypassing the format check.
- **Pleasantville (#9 in V3.2 single-anchor Barbie)** is the
  legitimate themes-recall win the gate tightening was supposed to
  preserve — `rare_keyword=0.202 themes=0.537`, both moderate-tier.
  Confirms the gate is tight enough to admit thematic peers but
  loose enough to also admit Man-in-Outer-Space-class noise.

#### Was the hypothesis correct?

**Mixed**. The combo gate change **fully cleaned up** multi-anchor
recall pollution (Pixar trio is the cleanest demonstration), but
**did NOT fix** single-anchor Barbie's foreign-film noise. The
original failure-mode framing was incomplete: I assumed the
recall-path noise was low-tier-trait accumulation, but the
single-anchor noise is moderate-tier-trait *coincidence* —
candidates whose shared trait IDFs are genuinely at moderate tier
but aren't thematically aligned. Tier-based gating cannot
distinguish the two.

#### Ways to improve going forward

In rough priority order:

1. **Add a popularity / quality gate to single-anchor themes-recall.**
   Man in Outer Space (1962), You Are a Widow Sir (1971), Adventure
   in Baltimore (1949) — all extremely-low-popularity foreign films.
   A minimum vote-count or stage_5_quality_score filter on the
   single-anchor recall fetch would prune these without touching
   any known good match. Multi-anchor doesn't need it (cohesion
   filter handles it). This is the cleanest next single-axis change.

2. **Direct-fetch auteur-director candidates for single-anchor.**
   Lady Bird / Little Women / The Favourite / Where'd You Go
   Bernadette can't be reached via shape OR themes recall when the
   embedding is distant. A direct "fetch all movies by Greta
   Gerwig (curated auteur)" recall path for single-anchor would
   guarantee Lady Bird and Little Women enter the pool — then the
   single-anchor director floor (already wired but inactive) would
   activate them.

3. **Investigate franchise-dominant lane on Barbie docs.** Tiny
   Shoulders / Barbie Nation are documentaries about the Barbie
   brand, not narrative features. They surface at #5 / #6 via the
   franchise lane. If the format-bucket multiplier is meant to
   suppress documentaries, why isn't it suppressing these? Could be
   that the franchise lane bypasses format scoring or the
   docs-as-franchise-content edge case isn't tagged.

4. **Cap final score at the multiplier-stacking layer.** Score
   inflation persists (Empire 1.820, Iron Man 2 1.323). Either
   normalize at the end or cap each multiplier so the final
   product ≤ 1.0. The semantic of "score in [0,1]" matters for
   downstream LLM rerankers and threshold gates.

5. **Tom Hanks / Get Out regressions** likely need item 4
   addressed first — the relative gap is what matters and may
   resolve once score inflation is bounded.

Bias for next iteration: **single-axis change**. Recommend item #1
(popularity gate on single-anchor themes-recall) — it's the
cleanest fix for the residual Barbie noise and the most isolated
change.

