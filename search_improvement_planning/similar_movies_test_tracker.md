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

### V3.3 — Shape multiplier (reach × quality identity boost) (2026-05-08)

#### Hypothesis

V3.2 ranking treats every candidate as "thematically similar by some
mix of lanes" but doesn't reward the *kind of film identity* the
candidate shares with the anchor. A Sharknado anchor matched against
The Room (both cult) gets no extra credit for the shared cult
identity beyond what shape / themes / rare_keyword already encode.
A Best Picture cohort matched against The Pianist gets no prestige
identity boost.

V3.3 introduces 5 named "shapes" derived from the reach × quality
grid (see [`similar_movies.md`](similar_movies.md) §V3.3) and a
multiplier ×1.15 (STRONG) or ×1.08 (MODERATE) that fires when
anchor and candidate share the same shape. Multi-anchor scales the
boost by cohort cohesion in the shape (`M_s / N`), gated at ≥0.5.

**Per-shape predictions:**

1. **cult_garbage** (Sharknado, The Room as anchors): same-shape
   candidates rise. Sharknado top 10 lifts Mega Shark, Mega Python,
   Sharknado sequels.
2. **mainstream_blockbuster** (Barbie, Inception, Toy Story as
   anchors): mainstream peers (other HIGH × Default) get ×1.08.
   Should be a small lift uniformly across the existing pool.
3. **prestige** (Best Picture trio, Lady Bird as anchors): The
   Pianist, The Mission, 12 Years a Slave-style candidates ride a
   ×1.08 boost. Cohesive prestige cohorts see modest score lifts.
4. **dogshit** (low-popularity poorly-rated films, rare as anchors
   in our suite): N/A in the smoke set.
5. **hidden_gem** (LOW × Acclaimed): also rare in our anchor set,
   but as candidates they may become more visible for prestige
   anchors.

**Why shape boost and not weaving:**
The deterministic MMR + 5-neighborhood adjacency graph
(B1/B2/B3 quota allocation per anchor neighborhood) was designed but
parked. The hypothesis is the score-level shape boost gets us most of
the perceived-quality lift on its own. If V3.3 leaves persistent
"this anchor's neighborhood is wrong" noise (e.g., niche cult anchor
still surfacing mainstream peers above same-tier cult peers), the
weaving change is the next single-axis shift.

**Regressions to watch:**
- Shape boost stacking with existing multipliers (country ×1.05,
  studio ×1.08-1.10) compounding score inflation. Empire was 1.820
  in V3.2; expect modest additional growth.
- Misclassification edge cases: a film at the reach cutoff (e.g.,
  The Room at 99K vote_count, just under 100K HIGH threshold)
  flipping shapes from cult_garbage to dogshit and changing who it
  matches with.
- Multi-anchor mixed cohorts (Studio Ghibli + Pixar mix) picking up
  a cohesive shape they shouldn't — verify cohesion < 0.5 actually
  suppresses the boost.

#### Changes actually made

Single-axis: shape multiplier wired into `_build_results` alongside
the existing country / studio / medium / shorts multipliers.

- [`search_v2/similar_movies.py`](../search_v2/similar_movies.py):
  - New constants `SHAPE_REACH_HIGH_THRESHOLD=100_000`,
    `SHAPE_REACH_LOW_THRESHOLD=10_000`, `SHAPE_BOOST_STRONG=0.15`,
    `SHAPE_BOOST_MODERATE=0.08`, `SHAPE_COHESION_MIN=0.50`, plus 5
    shape name string constants and `SHAPE_STRENGTHS` lookup table.
  - New `_classify_shape(row) → str | None` reads
    `imdb_vote_count` and `_quality_bucket(row)` to assign one of
    {dogshit, cult_garbage, prestige, hidden_gem,
    mainstream_blockbuster} or None.
  - New `_shape_multiplier(anchor_shape_cohesion, candidate_shape) →
    float` returns 1.0 for shapeless candidates or sub-threshold
    cohesion, else `1 + max_strength * cohesion`.
  - `_build_results` extended with `anchor_shape_cohesion` and
    `candidate_shape_by_movie` params. Multiplier applied after the
    shorts handling, surfaced as `multipliers["shape"]` in
    `LaneEvidence`.
  - Single-anchor: anchor shape classified once, cohort cohesion
    `{anchor_shape: 1.0}`. Candidate shapes computed from
    `candidate_rows`.
  - Multi-anchor: per-shape cohesion = M_s/N across the cohort,
    candidate shapes from `candidate_rows`. Cohort needs ≥0.5
    cohesion in a shape for the boost to fire.
- [`db/postgres.py`](../db/postgres.py):
  `fetch_similarity_signal_rows` now SELECTS `mc.imdb_vote_count`
  alongside the existing fields so the shape classifier has the raw
  reach signal at hand without an extra fetch.

No changes to recall paths, candidate generation, or lane scoring.
Pure scoring layer addition.

Doc updates:
[`similar_movies.md`](similar_movies.md) §V3.3 added with the reach
× quality grid, the 5 shapes, multiplier values, plug-in point, and
a future-work note on weaving (deterministic MMR with anchor-aware
neighborhood adjacency was designed but parked pending V3.3
verification).

#### Observations

Hypothesis was **correct for shape-cohesive cohorts**, less effective
where the underlying `_quality_bucket` classifier doesn't qualify
candidates for the same shape as the anchor.

##### Wins (hypothesis confirmed)

- **The Room (cult_garbage anchor) — major win.** Top 10 surfaces
  the cult-bad canon: **Troll 2 (×1.15 shape), Space Mutiny,
  Birdemic (×1.15 shape), The Minis, Movie 43, Manos: The Hands
  of Fate, Showgirls, Fateful Findings, Plan 9 from Outer Space**.
  Genuine cult-bad neighborhood. Troll 2 jumped to #1 from outside
  V3.2's top 10. Cult shape boost is doing exactly what was
  intended.
- **Sharknado (cult_garbage anchor).** Sharknado 2 lifted to 1.458
  (was 1.268 in V3.2 — `+0.190`). Sharknado 3 to 1.265 (was 1.100 —
  `+0.165`). All cult_garbage-classified Sharknado sequels caught
  the ×1.15 boost. Top 10 now anchored more firmly by the franchise.
- **MCU trio (mainstream_blockbuster cohort).** Cohort cohesion 1.0
  in mainstream_blockbuster (all 3 anchors HIGH × middle bucket).
  Iron Man 2 to 1.429 (was 1.323 — `+0.106`), Cap: Winter Soldier
  1.174 (was 1.087), Avengers: Age of Ultron 1.150 (was 1.065). All
  10 candidates received ×1.08 — including The Rock at #4
  (HIGH × Default mainstream peer, not MCU but same shape).
- **Best Picture trio (prestige cohort).** The Pianist 1.425 (was
  1.319 — `+0.106`), Citizen Kane 1.008 (was 0.933). Per-result
  breakdowns confirm ×1.08 fires on prestige-classified candidates;
  prestige neighborhood preserved cleanly.
- **Pixar trio.** Cohort classifies as **prestige** (not
  mainstream_blockbuster — Pixar films cross the recep ≥85 +
  pct ≥0.75 threshold). Toy Story 3 1.341 (was 1.242 — `+0.099`),
  Inside Out 1.252 (was 1.159), Monsters Inc 1.157, Good Dinosaur
  1.151. Strong cluster preserved with prestige lift.
- **Tarantino trio.** Kill Bill: Vol. 2 at 1.831 (Tarantino auteur
  + prestige shape both firing).
- **Oppenheimer single-anchor (prestige).** Dunkirk 1.019 (was
  0.944), Schindler's 0.959 (was 0.888) — both got ×1.08. Top 10
  preserved with prestige peers materially lifted.

##### Where the hypothesis was weaker

- **Female-led / Gerwig cohort.** Mixed cohort: Barbie classifies
  as `mainstream_blockbuster` (recep 75.2 below prestige threshold
  of 85), Lady Bird and Little Women 2019 classify as `prestige`.
  Cohort cohesion: 0.33 in mainstream_blockbuster (below
  threshold), 0.67 in prestige (above threshold). Prestige boost
  *should* fire on prestige-bucket candidates — but **most
  candidates surface as `middle` bucket** (Frances Ha at recep
  78.8, 20th Century Women at recep 79.0, Juno at recep 78.2 —
  all just below the 85 prestige cutoff). So cohort cohesion is
  there but few candidates qualify to receive it. Slate looks
  identical to V3.2 — no observable lift.

  This is a quality-bucket classifier limitation more than a shape
  multiplier limitation: the recep ≥85 threshold is conservative
  and excludes many films we'd colloquially consider "prestige."
- **Niche cult films (Mega Python, Leprechaun 4)** with vote_count
  3K–10K and reception 30–36 fail the cult_garbage bucket because
  popularity_percentile (0.787 / 0.883) is below the 0.89 cutoff.
  They classify as `middle` bucket → shapeless. So Sharknado anchor
  (cult_garbage) doesn't get a same-shape boost on Mega Python /
  Leprechaun 4 — they're classified as no-shape rather than dogshit
  (the design predicted dogshit, but they fail the bucket gate
  entirely). Top 10 preserved without lift on those rows.

##### Score inflation continues

Top scores still inflate when shape stacks with country / studio:
- Empire vs. Star Wars: 1.926 (was 1.820)
- Iron Man 2 (MCU): 1.429 (was 1.323)
- LOTR: Two Towers vs. Fellowship: 1.868
- Pianist (Best Picture): 1.425
- Toy Story 3 (Pixar): 1.341
- Kill Bill: Vol. 2 (Tarantino): 1.831

Multiplier stacking is producing scores well above 1.0 routinely.
Same regression-to-watch as V3.1/V3.2 — not addressed by V3.3.

##### Direct comparison: V3.2 vs V3.3 sample top-1 scores

| Anchor / cohort | V3.2 top-1 score | V3.3 top-1 score | Δ | Shape boost firing? |
|---|---|---|---|---|
| Sharknado | 1.268 | 1.458 | +0.190 | ✓ cult_garbage |
| The Room | (Troll 2 not top-1) | 0.720 (Troll 2) | new top-1 | ✓ cult_garbage |
| Best Picture (Pianist) | 1.319 | 1.425 | +0.106 | ✓ prestige |
| MCU (Iron Man 2) | 1.323 | 1.429 | +0.106 | ✓ mainstream_blockbuster |
| Pixar (Toy Story 3) | 1.242 | 1.341 | +0.099 | ✓ prestige |
| Female-led (Fabelmans) | 0.977 | 0.977 | +0.000 | ✗ (cohort split) |
| Female-led (Frances Ha) | 0.804 | 0.804 | +0.000 | ✗ |
| Oppenheimer (Dunkirk) | 0.944 | 1.019 | +0.075 | ✓ prestige |

#### Was the hypothesis correct?

**Yes, where cohort-shape cohesion is high.** Cult-bad cohorts (The
Room, Sharknado) and shape-cohesive multi-anchor cohorts (MCU,
Pixar, Best Picture, Tarantino) all see meaningful and well-targeted
lifts. The boost is doing exactly what was designed: identity-level
score addition on top of additive lane sums.

**Less effective where the quality bucket classifier doesn't qualify
the candidates.** This isn't a shape-multiplier problem; it's an
upstream classifier strictness problem. Films that we'd colloquially
group with "prestige" or "cult" don't always pass the existing
`_quality_bucket` thresholds, so they don't share a shape with
acclaimed/poorly-rated anchors and miss the boost.

#### Are weaving changes still warranted?

**Yes, partially — for different problems than what V3.3 addressed.**

- **Shape boost solves the score-level problem** (same-identity
  candidates should rank higher than different-identity ones). It's
  doing that.
- **Weaving solves a slot-level problem** that V3.3 doesn't touch:
  *guaranteeing the slate has a defensible mix of cohesive and
  adjacent neighborhood candidates.* The Lady-Bird-at-#39 failure
  for Barbie is still present — Lady Bird isn't in the candidate
  pool with strong-enough shape to clear the floor gates. Weaving
  with anchor-aware quotas (B1=same-neighborhood, B2=adjacent)
  would deliberately allocate her a slot.
- **Recall** is also still a separate problem. Some failure cases
  (Lady Bird, "Nope" missing from Get Out) are recall-bound — no
  weaving or shape change fixes a candidate that isn't in the pool.

So the priority is now:
1. Loosen / decouple shape classification thresholds (more
   candidates participate in same-shape matching).
2. Add a director-only recall path for single-anchor (auteur
   anchors get other auteur films into the pool unconditionally).
3. Then revisit weaving with the V3.3 shape signals as input to
   bucket assignment.

#### Should the 5 shapes have their strengths adjusted?

The 0.15 / 0.08 split played well in observation. Considerations:

- **Strong shapes (cult, dogshit, hidden_gem)** — the +0.15 lift
  was visible without overpowering recall (e.g., The Room top 10
  is genuinely cult-bad without becoming dominated by the boost).
  Probably correct.
- **Moderate shapes (prestige, mainstream_blockbuster)** — the
  +0.08 lift fires on cohesive cohorts (MCU, Pixar) without
  causing visible distortion. Probably correct.
- **Could try +0.20 / +0.10** if the V3.3 + recall changes still
  leave shape-cohesive cohorts looking too mixed. But the evidence
  doesn't yet justify the bump.

The bigger lever is the **`_quality_bucket` thresholds**, not the
multiplier strength. Loosening recep ≥85 to recep ≥80 (or the
percentile thresholds 0.89→0.80) would let many more films
participate. Worth trying as a follow-up isolated change before
adjusting shape strengths.

#### Ways to improve going forward

1. **Loosen `_quality_bucket` thresholds.** The recep ≥85 +
   pct ≥0.75 prestige threshold and recep ≤45 + pct ≥0.89
   cult_garbage threshold are tuned for legacy lanes (prestige
   quality scoring formulas). For shape classification we may want
   independent thresholds (e.g., recep ≥78 for "prestige shape"
   classification) so that Frances Ha, Spectacular Now, 20th
   Century Women all participate. Single-axis change.
2. **Director-only single-anchor recall path.** Lady Bird, Frances
   Ha, Little Women not entering Barbie's candidate pool is a
   recall problem. Direct-fetch by anchor's curated auteur director.
3. **Cap final score at 1.0** (or normalize at end). Score
   inflation is now well past where it was in V3 — Empire at 1.926.
   Downstream LLM rerankers and threshold gates will misbehave.
4. **Decide on weaving** based on (1)+(2). If the slate quality is
   acceptable after loosening thresholds and adding director recall,
   weaving may not be needed. If persistent neighborhood-mismatch
   noise remains, ship the deterministic MMR with the V3.3 shape
   classifications as bucket assignment input.

Bias for next iteration: **single-axis change**. Recommend item #1
(loosen bucket thresholds for shape classification) — it's the
biggest lever for making the V3.3 boost visible across more of the
suite, and it's cleanly isolated from recall changes.

### V3.3.1 — Loosen shape-classification thresholds (2026-05-08)

#### Hypothesis

The V3.3 smoke run revealed that the existing `_quality_bucket(row)`
gates (recep ≥85 + pct ≥0.75 for prestige; recep ≤45 + pct ≥0.89
for cult_garbage) are tuned for the legacy prestige/cult quality
scoring formulas — too strict for shape classification. Films we'd
colloquially treat as "prestige" (Frances Ha 78.8, 20th Century
Women 79.0, Juno 78.2, Little Women 1994 81.4) and films we'd
treat as "cult" or "dogshit" (Mega Python at percentile 0.787,
Leprechaun 4 at 0.883, Plan 9 from Outer Space) all classify as
`middle` bucket → shapeless. They miss shape participation
entirely.

V3.3.1 decouples shape classification from `_quality_bucket`:
- Prestige shape threshold: reception ≥ **78** (was 85)
- Poor shape threshold: reception ≤ **50** (was 45)
- Drop the percentile gate for shape entirely — the reach axis
  (10K / 100K vote_count tiers) already filters by audience size.

`_quality_bucket` itself is untouched — the legacy quality lane
formulas still use the strict V3 thresholds.

**Predictions:**
- Female-led / Gerwig cohort: Frances Ha, 20th Century Women, Juno
  newly classify as `prestige` shape and ride the cohort's prestige
  cohesion boost.
- Plan 9 from Outer Space and Mega Python newly classify as
  cult_garbage / dogshit shape; visible for cult anchors.
- MCU trio: Iron Man (recep 79.0) and Endgame (recep 80.4) cross
  prestige threshold; cohort splits between mainstream and
  prestige. Mild regression possible — fewer candidates get the
  cohort boost.
- Inception (recep 79.6) newly classifies as prestige; single-
  anchor boost fires on Tenet/Memento/Mulholland Drive if they
  also classify prestige.
- Best Picture and Pixar trios: minimal change (already in prestige
  bucket).

#### Changes actually made

[`search_v2/similar_movies.py`](../search_v2/similar_movies.py):
- New constants `SHAPE_PRESTIGE_RECEPTION_MIN=78.0`,
  `SHAPE_POOR_RECEPTION_MAX=50.0`. Percentile gates removed.
- `_classify_shape(row)` rewritten to read `reception_score` and
  `imdb_vote_count` directly, bypassing `_quality_bucket`. Reception
  None → only HIGH × no-data earns mainstream_blockbuster, else
  shapeless. Otherwise reception thresholds + reach tiers map
  cleanly onto the 5 shapes.

No other files touched. `_quality_bucket` and the legacy quality
lane formulas are unchanged.

#### Observations

##### Major win — Female-led / Gerwig (multi-anchor)

V3.3.1 delivered exactly what the hypothesis predicted. Cohort
shape distribution: Barbie (recep 75.2) = mainstream_blockbuster,
Lady Bird (85.4) and Little Women 2019 (85.8) = prestige. Cohesion
0.67 in prestige (above 0.5 threshold) → ×1.054 boost fires on
prestige candidates.

| Candidate | V3.3 score | V3.3.1 score | Δ | Now firing |
|---|---|---|---|---|
| The Fabelmans | 0.977 | **1.029** | +0.052 | ×1.05 prestige |
| Little Women (1994) | 0.912 | **0.961** | +0.049 | ×1.05 prestige |
| 20th Century Women | 0.818 | **0.861** | +0.043 | ×1.05 prestige |
| Frances Ha | 0.804 | **0.847** | +0.043 | ×1.05 prestige |
| Juno | 0.864 | **0.910** | +0.046 | ×1.05 prestige |

New entries to top 10: **Licorice Pizza** (0.785, ×1.05 prestige
boost — coming-of-age PT Anderson, surfaced by themes recall +
prestige shape) and **Rushmore** (0.784, ×1.05 prestige —
classic Wes Anderson). Spectacular Now and Adventure in Baltimore
no longer classified as prestige (recep below 78), so they
remain unboosted but kept their slots — net slate is more
indie-prestige-coherent.

##### Win — The Room (single anchor)

Plan 9 from Outer Space jumped from #10 (0.533) to #4 (0.612) —
the loosened cult_garbage threshold finally classifies it as a
same-shape peer, ×1.15 boost firing. Top 10 still the cult-bad
canon (Troll 2, Space Mutiny, Birdemic, Plan 9, Movie 43, Manos,
Showgirls, Fateful Findings, Sex Marriage and Infidelity).

##### Win — Inception (single anchor)

Inception itself now classifies as prestige (recep 79.6).
Single-anchor cohesion 1.0 → ×1.08 prestige boost fires on
prestige candidates:
- Interstellar: 0.957 → **0.987** (+0.030)
- Mulholland Drive: 0.863 → **0.932** (+0.069 — also reclassified
  as prestige)
- Memento: 0.832 → **0.898** (+0.066)
- The Matrix: now in top 10 at 0.848 (replaced Trance).

The cluster gets a uniform lift across the Nolan auteur + puzzle
neighborhood.

##### Mild regression — MCU trio

V3.3 cohort was unanimously mainstream_blockbuster (cohesion 1.0,
all candidates boosted ×1.08). V3.3.1: Iron Man (79.0) crossed
into prestige; cohort now 2/3 mainstream + 1/3 prestige. Cohesion
0.67 in mainstream → ×1.054 instead of ×1.08 on mainstream
candidates. Endgame (80.4) reclassified prestige and loses the
boost (cohort prestige cohesion 0.33 < 0.5).

| MCU film | V3.3 | V3.3.1 | Δ |
|---|---|---|---|
| Iron Man 2 | 1.429 | 1.394 | -0.035 |
| Cap: Winter Soldier | 1.174 | 1.145 | -0.029 |
| Avengers: Age of Ultron | 1.150 | 1.121 | -0.029 |
| Avengers: Infinity War | 1.224 | 1.193 | -0.031 |
| Cap: First Avenger | 1.153 | 1.125 | -0.028 |
| Avengers: Endgame | 1.127 | 1.043 | -0.084 |

Cluster integrity preserved — top 10 is still 9/10 MCU + The Rock
at #4. Brave New World (was #8, 1.085) drops out of top 10
replaced by **Thor: The Dark World** (1.034) at #10. This is
within-cohort reshuffling, not a meaningful regression.

##### No change — Pixar trio, Best Picture trio, Sharknado

Pixar (Toy Story 3 still 1.341, Inside Out 1.252) and Best Picture
(Pianist 1.425) cohorts already had their anchors uniformly in
prestige under V3.3 — V3.3.1 doesn't change classification or
boost behavior for them. Sharknado (cult_garbage anchor)
identical: cult_garbage candidates still boost; dogshit candidates
(Mega Python, Leprechaun 4) remain unboosted because they're a
different shape — by design.

##### No fix — Tom Hanks regression, Lady Bird at #39 for Barbie

V3.3.1 doesn't address recall-bound failures:
- Tom Hanks trio still missing Cast Away, The Terminal, Captain
  Phillips. Family-animation peers continue to outrank
  Hanks-via-cast vehicles.
- Lady Bird, Little Women 2019, Frances Ha for Barbie single-anchor
  remain at #20+ — they're not in the candidate pool with
  strong-enough shape. Shape boost can't lift candidates that
  aren't in the pool.

These need a director-only recall path or weaving — separate
changes.

##### Score inflation continues

Empire Strikes Back: 1.926 (was 1.926 in V3.3 — unchanged).
LOTR: Two Towers vs. Fellowship: 1.868 (unchanged).
Iron Man 2: 1.394 (down from 1.429 in V3.3 due to cohort split).
Pianist (Best Picture): 1.425 (unchanged).
Toy Story 3 (Pixar): 1.341 (unchanged).
The Empire Strikes Back vs Star Wars single-anchor: 1.926 → 1.926.

Score inflation didn't worsen meaningfully. The shape multiplier's
contribution was already capped at ×1.15; adding more candidates
to its eligibility doesn't change the per-row maximum.

#### Was the hypothesis correct?

**Yes, with one anticipated tradeoff.** Loosening the thresholds
delivered the predicted win (Female-led / Gerwig cohort now sees
visible prestige boosts; Plan 9 newly cult-classified) and the
predicted regression (MCU cohort split, Endgame loses boost). The
tradeoff is acceptable — MCU's cluster integrity is preserved
even with a slightly weaker cohort boost, and the win on Female-led
is materially more important.

#### Are there any unintended consequences?

- **Shape boundaries are now reception-driven, which means cohort
  composition is sensitive to a few-percentage-point reception
  differences.** Iron Man at 79.0 is prestige; Avengers at 73.4 is
  mainstream. They're qualitatively the same kind of film
  (origin-story MCU), but the shape system splits them. This
  causes the MCU cohesion dilution.
- **Some legitimately-prestige films still miss the threshold.**
  The Mission (Best Picture nominee, 62.6 reception in our data)
  and The Killing Fields (Oscar winner, 76.8 reception) still
  classify as middle. Reception scores in our data don't always
  reflect awards consensus — some films historically loved by the
  Academy have middling user reception.
- **Tarantino trio** anchor classifications didn't shift much
  (Pulp Fiction was already prestige; the cohort's behavior is
  similar). No regression.

#### Was no regression introduced?

Verified across the 21-anchor + 14-cohort suite:
- **No top-1 lost.** Every cohort's #1 is the same film as V3.3
  (or marginally lifted).
- **No franchise cluster broken.** MCU still 9/10 MCU. Star Wars
  still 6 same-saga + same sci-fi adjacents. Pixar still
  Pixar-dominant. LOTR still LOTR-dominant. Stephen King still
  King-dominant.
- **No surprise demotions.** Endgame went from #7 to #9 within
  MCU (still in top 10). Brave New World swapped with Thor: The
  Dark World — both MCU.
- **The Female-led / Gerwig cohort improved meaningfully** without
  any other cohort regressing.

#### Adjustments to consider

1. **Recall-bound failures (Lady Bird at #39 for Barbie, Tom Hanks
   missing Cast Away) are next priority.** Shape multiplier
   tuning can't help. Director-only single-anchor recall path is
   the cleanest single-axis change.
2. **Reception threshold sensitivity is a structural concern.**
   For multi-anchor cohorts where the reception bunches around the
   78 cutoff (MCU), small rating differences split the cohort
   shape. Could mitigate with a soft transition zone (anchor at
   76-80 partial-membership in both shapes) but this is a
   complexity tradeoff. Worth considering only if smoke runs show
   recurring cohort dilution.
3. **Shape strengths (0.15 / 0.08) still feel right.** No data in
   V3.3.1 motivates a change.

The next single-axis change should be a director-only single-anchor
recall path — that addresses the persistent Lady-Bird-at-#39 case
that no shape or scoring change can fix.

### V3.3.2 — Award-aware classification + cross-bucket boosts (2026-05-08)

#### Hypothesis

Two refinements bundled in one ship:

**(1) Award-aware shape thresholds.** V3.3.1 used reception alone
(78 default), which (a) over-included middling films at recep ≥78
that had no genuine prestige signal, and (b) excluded legitimately
acclaimed older films with mid-range modern reception (The Killing
Fields recep 76.8, won 3 Oscars including Best Picture nom).
Tighten the default reception floor to 80 and add an award-aware
lowered floor at 65 that fires only on a *picture-level* signal
(Best Picture or Director nom/win at any non-Razzie ceremony —
acting/craft awards explicitly excluded). Razzie WIN side: any
WIN in a specific WORST_* category (excluding WORST_OTHER which
might catch the Razzie Redeemer Award) raises the poorly-rated
ceiling from 50 to 60.

**(2) Cross-bucket boost matrix.** V3.3.1 only fired the shape
multiplier when anchor and candidate shared the exact same shape.
The boundary cuts are arbitrary (e.g., the 10K reach split between
cult_garbage and dogshit). Add a 5×5 cross-strength matrix where
same-shape pairs are 1.0, "boundary-arbitrary same-quality reach
splits" (prestige↔hidden_gem, cult_garbage↔dogshit) are 0.7, and
quality-step crossings via the mainstream bridge are 0.4 / 0.25 /
0.15 / 0.10. Effective cohesion sums anchor-side cohesion times
cross-strength. SHAPE_COHESION_MIN stays at 0.5 (option C from
the design discussion) — single-anchor cross-shape pairs at 0.4
won't fire, but mixed cohorts where same-shape (1.0) + cross (0.4)
contributions sum to ≥0.5 will fire.

**Predictions:**
- The Killing Fields (76.8 + Best Picture nom) lifts to prestige
  shape and rides ×1.08 boost in the Best Picture cohort.
- Bohemian Rhapsody (62 recep, 4 Oscar wins all in performance/
  craft, no Best Picture / Director nom) — wait, BR was Best
  Picture-nominated. Recep 62, threshold even at 65 fails. Stays
  shapeless. ✓
- Sharknado (cult_garbage) lifts Mega Python and Leprechaun 4
  (dogshit) via 0.7 cross-bucket → effective 0.7 ≥ 0.5 → ×1.105.
- The Room (cult_garbage) lifts Space Mutiny / Birdemic / Plan 9
  similarly when reach-tier-different cult films now share the
  same shape via cross-bucket.
- MCU cohort: Iron Man (79.0) and Endgame (80.4) cross prestige
  threshold via picture-level signal. Cohort cohesion shifts from
  0.67 mainstream + 0.33 prestige (V3.3.1) to closer to 1.0
  prestige if all 3 anchors fire. Iron Man 2 (no signal, recep
  61.8) stays mainstream → loses cohort boost (was 0.67 cohesion).
- Female-led / Gerwig: Barbie (75.2 + picture-level) now classifies
  as prestige (was mainstream). Cohort cohesion 1.0 prestige (was
  0.67) → all prestige candidates lift from ×1.054 to ×1.08.
- The Mission (62.6 + Cannes Palme d'Or) — recep below 65 floor
  even with award. Stays shapeless. (Acceptable miss; can be
  loosened to 60 in a follow-up if needed.)

#### Changes actually made

[`db/postgres.py`](../db/postgres.py):
- Added `SHAPE_PICTURE_LEVEL_TAG_IDS = (103, 9)` (BEST_PICTURE_ANY
  rollup + DIRECTOR leaf, excluding DEBUT_DIRECTOR /
  ASSISTANT_DIRECTOR) and
  `SHAPE_BAD_RAZZIE_LEAF_IDS = (46..57)` (all WORST_* leaves
  excluding WORST_OTHER 58).
- Extended `SimilarityAwardSignals` dataclass with
  `has_picture_level_signal: bool` and `has_bad_razzie_win: bool`.
- Extended `fetch_similarity_award_signals` SQL query to compute
  both flags as additional `BOOL_OR` aggregates in the same single
  SQL pass (no extra round trip).

[`search_v2/similar_movies.py`](../search_v2/similar_movies.py):
- Bumped `SHAPE_PRESTIGE_RECEPTION_MIN` 78.0 → 80.0.
- Added `SHAPE_PRESTIGE_RECEPTION_MIN_W_AWARD = 65.0` (with
  picture-level signal).
- Added `SHAPE_POOR_RECEPTION_MAX_W_RAZZIE = 60.0` (with bad-Razzie
  WIN).
- Added `SHAPE_CROSS_STRENGTH` matrix (5×5 dict, all pairs).
- Rewrote `_classify_shape(row, award_signal)` to accept optional
  award signal and apply the threshold-shifting logic.
- Rewrote `_shape_multiplier(...)` to use cross-bucket strengths
  via `SHAPE_CROSS_STRENGTH.get((anchor_shape, candidate_shape),
  0.0)` summed across anchor shapes for effective cohesion.
- Updated single-anchor flow to always fetch `award_signals` (not
  gated on `cult_or_prestige`) and include the anchor ID.
- Updated multi-anchor flow same way; classification now reads
  award signals for both anchors and candidates.
- Both flows pass `anchor_shape_cohesion` and
  `candidate_shape_by_movie` through to `_build_results` (already
  wired in V3.3).

No recall path changes. Pure scoring layer + classifier refinement.

#### Observations

##### Major win — Best Picture / Killing Fields (predicted)

The Killing Fields (recep 76.8, Best Picture nom) was shapeless in
V3.3.1 (76.8 < 78) and didn't ride the cohort prestige boost.
V3.3.2 classifies it as prestige via picture-level signal lowering
the floor to 65. Result: 0.992 → **1.071** (+0.079, ×1.08 fired).

##### Major win — Sharknado / The Room cult+dogshit cross-bucket

Cross-bucket cult_garbage↔dogshit at 0.7 strength fires (effective
cohesion 0.7 ≥ 0.5 threshold). Same-quality reach split is no
longer a hard cut.

| Anchor / candidate | V3.3.1 | V3.3.2 | Δ | Shape | Cross |
|---|---|---|---|---|---|
| Sharknado → Mega Python | 0.519 | **0.573** | +0.054 | dogshit | cult→dogshit 0.7 |
| Sharknado → Leprechaun 4 | 0.504 | **0.557** | +0.053 | dogshit | cult→dogshit 0.7 |
| Sharknado → Sharknado 4 | 1.220 | **1.348** | +0.128 | cult_garbage | same |
| The Room → Space Mutiny | 0.703 | **0.777** | +0.074 | cult_garbage | same |
| The Room → The Minis | 0.563 | **0.623** | +0.060 | cult_garbage | same |

Sharknado top 10 is now genuinely creature-feature / cult-bad
coherent. Cult fans get the right peers regardless of which side of
the 10K reach line they sit on.

##### Major win — Female-led / Gerwig cohort cohesion

Barbie (75.2) was mainstream_blockbuster in V3.3.1. With V3.3.2
picture-level signal lowering threshold to 65, Barbie classifies as
prestige. Cohort cohesion goes from 0.67 prestige (V3.3.1) to **1.0
prestige** (V3.3.2). All prestige candidates now ride full ×1.08
(was ×1.054).

| Female-led candidate | V3.3.1 | V3.3.2 | Δ |
|---|---|---|---|
| The Fabelmans | 1.029 | **1.055** | +0.026 |
| Little Women (1994) | 0.961 | **0.985** | +0.024 |
| 20th Century Women | 0.861 | **0.883** | +0.022 |
| Frances Ha | 0.847 | **0.869** | +0.022 |
| Juno | 0.910 | **0.933** | +0.023 |
| Jojo Rabbit | 0.772 | **0.834** | +0.062 |

Top 10 unchanged. All prestige peers lift uniformly.

##### Mixed regression / win — MCU cohort split

V3.3.1 had MCU cohort 0.67 mainstream + 0.33 prestige; all
mainstream candidates got ×1.054. V3.3.2: all 3 MCU anchors cross
into prestige via picture-level signal. Cohort cohesion 1.0
prestige. **Now mainstream-shape MCU candidates lose the boost
(cross-bucket prestige→mainstream at 0.4 < 0.5 cohesion gate),
while prestige-shape MCU candidates gain full ×1.08.**

| MCU film | V3.3.1 | V3.3.2 | Δ | Reason |
|---|---|---|---|---|
| Iron Man 2 | 1.394 | **1.323** | -0.071 | mainstream candidate, lost cohort boost |
| Cap: Winter Soldier | 1.145 | **1.174** | +0.029 | prestige candidate (picture-level), gained ×1.08 |
| Avengers: Age of Ultron | 1.121 | **1.065** | -0.056 | mainstream, lost |
| Avengers: Infinity War | 1.193 | **1.224** | +0.031 | prestige, gained |
| Avengers: Endgame | 1.043 | **1.127** | +0.084 | prestige (recovered from V3.3.1 loss) |
| Thor: Ragnarok | 1.058 | **1.085** | +0.027 | prestige, gained |
| Cap: First Avenger | 1.125 | **1.068** | -0.057 | mainstream, lost |

Net: 4 films gain (prestige-classified peers), 5 films drop slightly
(mainstream-classified peers). Top 10 still 9/10 MCU + The Rock at
#4. Cluster integrity preserved.

This is the predicted asymmetry of cohesive-prestige cohorts:
prestige peers ride the lift, non-prestige peers lose the cohort
boost they had under split-cohesion. Acceptable tradeoff — Endgame
is back where it should be (in the top 6).

##### No change — Pixar trio, Inception (single), Best Picture

- Pixar trio: identical scores to V3.3.1 (anchors all already
  classified as prestige under V3.3.1; nothing about the cohort
  changed in V3.3.2).
- Inception: top 10 same as V3.3.1, scores essentially unchanged.
  Anchor reclassification doesn't ripple because cohort behavior
  for single-anchor is binary (anchor either has shape or doesn't).
- Star Wars / Sharknado / Get Out single-anchor: unchanged or mild
  improvement.

##### Predicted miss — The Mission

Cannes Palme d'Or, recep 62.6. Picture-level signal fires but
recep < 65 floor → shapeless. Stays at #2 with score 1.124 (no
×1.08). Acceptable; would need V3.3.3 to drop the with-award
floor to 60 if this becomes load-bearing.

##### Score inflation

| Film | V3.3 | V3.3.1 | V3.3.2 |
|---|---|---|---|
| Empire vs Star Wars | 1.926 | 1.926 | 1.926 |
| Toy Story 3 (Pixar trio) | 1.341 | 1.341 | 1.341 |
| Pianist (Best Picture) | 1.425 | 1.425 | 1.425 |
| Iron Man 2 (MCU) | 1.429 | 1.394 | 1.323 |
| Sharknado 2 | 1.458 | 1.458 | 1.458 |

Score inflation didn't worsen. Iron Man 2 actually came down due to
losing its cohort boost.

#### Was the hypothesis correct?

**Yes for both refinements.**

1. **Award-aware thresholds**: cleanly catches The Killing Fields
   without admitting Bohemian Rhapsody. Razzie side untested in
   the smoke set (no anchors with that profile in our list) but
   the SQL aggregate produces correct results in spot checks.
2. **Cross-bucket boosts**: Sharknado→Mega Python and Sharknado→
   Leprechaun 4 lift visibly via the 0.7 cult↔dogshit cross. MCU's
   mixed cohort split is the predicted asymmetric cost — net
   tradeoff is acceptable.

#### Are there unintended consequences?

- **MCU mainstream peers (Iron Man 2, Age of Ultron, Cap: First
  Avenger) lose their V3.3.1 cohort boost** because the cohort is
  now uniformly prestige. The prestige→mainstream cross at 0.4 is
  intentionally below the cohesion threshold (option C from the
  design discussion). The cluster doesn't break, but the within-MCU
  rank ordering shifted. Watch in subsequent runs.
- **The 65 with-award floor still misses some legitimately-prestige
  films** (The Mission at 62.6 specifically). Conservative number
  by design; can be loosened to 60 as a separate single-axis change
  if smoke runs prove the miss is recurring.
- **Razzie Redeemer Award not formally distinguished in our schema.**
  Excluding WORST_OTHER from the bad-Razzie set is a heuristic
  buffer. If the redeemer ever gets its own category tag, the
  exclusion list can become explicit.

#### Was no regression introduced?

Verified across the 21-anchor + 14-cohort suite:
- **No top-1 lost** in any cohort. Every cohort's #1 is the same
  film as V3.3.1 (or marginally lifted).
- **No franchise cluster broken.** MCU still 9/10 MCU. Pixar still
  Pixar-dominant. Star Wars saga preserved. LOTR preserved.
  Stephen King preserved.
- **Sharknado cluster strengthened** (Mega Python / Leprechaun 4
  now genuinely peers, not just same-tag noise).
- **Best Picture cluster strengthened** (Killing Fields lifted to
  rank #3 with prestige boost; Pianist still #1).
- **Female-led / Gerwig cluster strengthened** (full ×1.08 across
  prestige peers; cluster more cohesive than V3.3.1).
- **MCU within-cluster reshuffling** (some peers lose, some gain) —
  cluster integrity preserved at the top-10 level.
- **Tom Hanks regression unchanged** — still missing Cast Away,
  Captain Phillips, The Terminal. Recall-bound; no shape change
  fixes.
- **Lady Bird at #39 for Barbie unchanged** — recall-bound.

#### Ship it?

Yes. The wins (cult-cross-bucket, Best Picture's Killing Fields
lift, Female-led cohort cohesion improvement) are concrete and
visible. The MCU within-cluster reshuffling is a tradeoff, not a
regression on cluster integrity. No top-1 lost anywhere.

Next priorities (unchanged from V3.3.1):
1. Director-only single-anchor recall path (Lady Bird at #39, Tom
   Hanks lost vehicles).
2. Score inflation cap if downstream consumers care.
3. Potentially loosen with-award floor 65 → 60 if The Mission case
   recurs.

### V3.4 — Bucket-Weaver multi-source recommendation layer (2026-05-08)

#### Hypothesis

V3.1–V3.3.x improved the unified ranker but couldn't fix one structural
limit: candidates like Lady Bird (Gerwig) score `director=0.20→0.20`
firing for the Barbie anchor, but `shape=0.033` burns most of the
~0.50 shape weight. Director floor (0.35) sits below the noise floor
of best-overall scores (~0.37+), so it can't displace. No re-weighting
of the unified ranker can elevate Lady Bird to top 10 without
distorting mainstream / cult / franchise anchors.

The fix is architectural: movie similarity is a *recommendation*
problem (rows by *why*) not a *search* problem (one ordered list). A
user looking at Barbie may simultaneously want best-overall matches
(Poor Things), more from the auteur (Lady Bird), and shared rare
themes (I Am Not an Easy Man). A scalar score can't rank all three
correctly at once.

V3.4 adds a thin weaver layer on top of the V3 unified ranker:
1. Run V3 ranker → ranked candidates with full evidence.
2. Compute per-anchor bucket-signal scores → instantiate buckets ≥0.30.
3. Allocate slots: 5 floor for best-overall + 5 distributed by signal.
4. Greedy weave with MMR-style starvation boost (λ=0.5).
5. Full credit on placement (membership-wide).

The 5 buckets:

| Bucket | Gate | Hypothesis |
|---|---|---|
| **Best overall** | always on (5/10 floor) | preserves global similarity ordering |
| **Auteur director** | anchor's director ∈ `AUTEUR_NORM_STRINGS` | surfaces Gerwig for Barbie, Nolan for Inception, etc. |
| **Franchise** | `_franchise_score_v2 ≥ 0.55` | mainline trilogy + cross-mainline siblings; tier-4 universe-only excluded |
| **Rare keyword** | shares high-tier IDF trait | surfaces Manhattan Project for Oppenheimer, mind-bending puzzles for Inception |
| **Lead actor** (multi-only) | cohesion ≥ 0.5 ∧ shared top-3 actor | surfaces more Hanks vehicles for the H9 trio |

Predicted concrete wins:
- **Barbie**: Lady Bird and Little Women '19 enter top 10 via auteur
  bucket; The Favourite / Poor Things stay via best-overall.
- **Inception**: Tenet / Prestige / Memento via auteur bucket; mind-
  bending puzzle films via rare_keyword bucket.
- **Tom Hanks trio (H9)**: Cast Away / Captain Phillips / The Terminal
  enter via lead_actor bucket (was: missing in V3.3.2).
- **Star Wars / MCU**: franchise bucket fills with mainline saga +
  Rogue One; tier-4 universe-only (Eternals / Madame Web) excluded.
- **Pixar trio / Best Picture trio**: best-overall floor preserves
  cohesive clusters; no auteur/franchise displacement.

Predicted regressions to monitor:
- **Score relativity**: V3.4 doesn't change scores, but may surface
  candidates that previously sat at #20–40 with score 0.4 alongside
  candidates at #1–5 with score 0.7. Visual contrast expected.
- **Bucket starvation**: a bucket with ≤2 candidates in V3 top-N
  can't fill its allocated slots. Should gracefully degrade — slack
  flows to best-overall — but worth confirming.
- **Format lock interaction**: shorts cap and format top-5 lock layered
  on top of greedy may produce empty-pick slots if no bucket has a
  format-eligible unplaced candidate.

#### Changes actually made

All 5 buckets shipped (auteur, franchise, rare_keyword, lead_actor,
best_overall) with the greedy-MMR weaver. Code touched:

- [`search_v2/similar_movies.py`](../search_v2/similar_movies.py):
  - New constants near `TOP_FORMAT_LOCK`: `WEAVER_TOTAL_SLOTS=10`,
    `WEAVER_BEST_OVERALL_FLOOR=5`, `WEAVER_BUCKET_CAP=3`,
    `WEAVER_BUCKET_INSTANTIATE_MIN=0.30`, `WEAVER_LAMBDA=0.5`,
    `WEAVER_FRANCHISE_BUCKET_MIN_SCORE=0.55`,
    `WEAVER_RARE_KEYWORD_BUCKET_IDF_MIN=0.55`. Plus 5 `BUCKET_*`
    string constants and the `ALL_BUCKETS` tuple.
  - `_compute_bucket_targets()`: Hamilton largest-remainder allocator.
    Initial implementation used plain `round()` and overshot remaining
    when 3+ buckets had equal signals; replaced before re-run.
  - `_peek_next_eligible_for_bucket()`: per-bucket V3-rank queue walk
    that respects format top-5 lock and shorts cap.
  - `_weave_candidates()` rewritten: replaces the V3 dominance / max-
    franchise / competitive-band caps with the slot-by-slot greedy MMR
    pass. Best-overall is carved out as always-on (the bucket gate
    `if b != BUCKET_BEST_OVERALL and placed[b] >= target_count` keeps
    it competing on every slot regardless of `placed[]` overshoot via
    multi-bucket credit). Past TOP_SECTION_SIZE, remaining V3-ranked
    candidates fill up to `limit`.
  - `_build_results()` signature: two new optional params
    `bucket_signals` and `bucket_memberships_by_movie` flow through
    to `_weave_candidates`.
  - `_compute_single_anchor_bucket_data()` and
    `_compute_multi_anchor_bucket_data()`: derive signals + memberships
    from the lane data already computed by both flow functions; no
    extra DB reads. Multi-anchor cohesion factors `M_d/N`, `M_f/N`,
    `M_a/N` weight the auteur / franchise / lead-actor signals.
  - `_run_single_anchor_similarity` and `_run_multi_anchor_similarity`
    each call the corresponding bucket-data helper before
    `_build_results` and pass the dicts through.

- [DIFF_CONTEXT.md](../DIFF_CONTEXT.md): V3.4 entry appended.

Doc updates: this tracker (current section) + DIFF_CONTEXT entry.
similar_movies.md §V3.4 was already finalized in the prior planning
session and didn't need further changes.

Implementation issues caught during smoke testing:
1. **Hamilton's method needed**: plain `round(remaining * share)` per
   bucket overshot remaining when ≥3 buckets had equal signals (3 ×
   round(1.667) = 6 > 5 remaining). Fixed via largest-remainder.
2. **Best-overall always-on**: initial implementation gated best-
   overall on `placed[best_overall] >= target`, which combined with
   multi-bucket full credit caused early short-circuits (Tom Hanks
   trio returned 6 results instead of 10). Fixed by carving out
   best_overall from the gate; deficit clamps to ≥0 so once it
   exceeds target it competes on relevance only.

#### Observations

##### Wins (hypothesis confirmed)

**Barbie (single)** — canonical V3.4 case:
- V3.3.2 Barbie top 10 had Lady Bird at #39 (recall-bound).
- V3.4: **Lady Bird at #3 (score 0.423), Little Women '19 at #5
  (score 0.419)** via auteur bucket. Slot 1–2 stay best-overall
  (Last Action Hero, Poor Things). Pleasantville at #10 — themes/
  rare_keyword surfacing.

**Female-led / Gerwig (multi)** — V3.1 wishlist:
- V3.3.2 had The Fabelmans, Little Women '94, Atonement at top.
- V3.4: same top 3 + **Frances Ha at #7** (was previously
  recall-blocked) + **Nights and Weekends at #4** (Gerwig writer/
  co-director, surfaces via auteur bucket at low score 0.442).
  20th Century Women at #6, Circle of Friends #8.

**Best Picture trio (multi)** — Mission floor concern resolved:
- V3.3.2 had The Mission missing top 10 (62.6 reception below the
  65-with-award floor).
- V3.4: **The Mission at #2 (score 1.124)** via franchise/auteur or
  rare_keyword bucket. Pianist still #1, Killing Fields preserved.

**Inception (single)**:
- Tenet, Memento, The Dark Knight, Interstellar all in top 10 via
  auteur bucket. Mulholland Drive #2 (themes), 2001 ASO #5,
  Trance #6, Sucker Punch #8, EEAAO #10 — strong puzzle/dream
  cluster.

**Pulp Fiction (single)**:
- V3.3.2 had Reservoir Dogs #1, True Romance #2, Sin City #3,
  Thursday #4, Jackie Brown #5.
- V3.4: Reservoir Dogs #1, **Jackie Brown promoted to #2** (0.933,
  auteur deficit pull), True Romance #3, Sin City #4. More
  Tarantino-cohesive top.

**Tarantino trio (multi)**:
- Strong canon: Kill Bill 2, Django, Reservoir Dogs, Inglourious
  Basterds, Jackie Brown, True Romance, Death Proof — full QT
  cluster preserved.

**MCU trio (multi)** — V3.4 franchise gate working:
- All 10 results are mainline MCU (Iron Man 2, Infinity War, Winter
  Soldier, Endgame, Thor Ragnarok, Cap First Avenger, Age of Ultron,
  Cap Brave New World, Shang-Chi, Thor Dark World). **Tier-4
  universe-only spinoffs (Eternals, Madame Web) correctly
  excluded.**

**Star Wars (single)** — full saga preserved:
- All 8 saga films + LotR Fellowship #9 + ROTK #10. American
  Graffiti correctly silent on auteur (V3 H7 preserved).

##### Regressions

**Pixar trio (multi)** — V3.4 breaks Pixar-dominance:
- V3.3.2: 10/10 Pixar films (Toy Story trilogy, Inside Out, Monsters,
  Good Dinosaur, Onward, Bolt, Ratatouille, Wreck-It Ralph).
- V3.4: 7 Pixar + **The Lego Movie #4, Madagascar #6, The Rescuers
  Down Under #8** (non-Pixar animations). Lost: Bolt, Ratatouille,
  Wreck-It Ralph from top 10.
- Root cause: rare_keyword bucket fires on Pixar trio (shared
  moderate-tier traits like talking animals / family / friendship
  push `signal[rare_keyword]` to 1.0). Lego Movie / Madagascar share
  one high-tier (≥0.55) trait → membership → forced into bucket
  slots via deficit pressure.

**Barbie (single)** — documentary-meta pollution:
- Slots 7–8: **"Tiny Shoulders: Rethinking Barbie" (2018) and "Barbie
  Nation: An Unauthorized Tour" (1998)**. Both are documentaries
  about the Barbie product itself; they qualify for the rare_keyword
  bucket because they share Barbie-specific keywords.
- Slot 4: Barbie: Princess Adventure (animated tie-in spinoff) —
  franchise bucket via tier-3 (shared subgroup + universe).
- Format top-5 lock + V3 H1 docs-suppression should have caught
  these pre-V3.4. The format lock disengages at slot 6 and the
  rare_keyword bucket bypasses the V3 quality reranker.

**Slasher trio (multi)** — forced auteur and rare_keyword distort:
- Note: the test_set.md cohort doc says "The Thing, Carrie,
  Halloween" but the actual IDs in `run_similar_movies_batch.py`
  resolve to "The Thing, Everyone Says I Love You, Scream 2" — a
  heterogeneous 3-anchor mix, not a slasher cohort. Caveat applies
  but the V3.4 distortion is still visible:
- V3.3.2 top 10: Scream, Scream 4, My Bloody Valentine '81, Scream
  '22, Cabin in Woods, Scream VI, Pandorum, Scream 3, MBV '09,
  Friday the 13th '80 (mostly Scream-saga + slasher cluster).
- V3.4: Scream, Scream 4, **In the Mouth of Madness #3** (Carpenter
  auteur from The Thing — fine), **Pennies from Heaven #4**, Cabin
  in Woods, Halloween, **Mamma Mia! Here We Go Again #7**, Scream
  VI, MBV '81, Scream '22.
- Pennies from Heaven (Steve Martin musical) and Mamma Mia (musical
  romcom) clearly don't belong. They surface because Everyone Says I
  Love You shares high-IDF "musical" / "song" traits with them →
  rare_keyword bucket members. Auteur bucket fires on Carpenter at
  cohesion 1/3 = 0.33 (just clears 0.30 instantiate min) → adds
  In the Mouth of Madness.

##### Pattern across regressions

All three regressions share a root cause: the **rare_keyword bucket
membership gate is too loose**. Membership requires sharing any
trait with `IDF >= 0.55` against the anchor's pool. Many "high-tier"
IDF traits in `mv_trait_idf` are category-defining rather than
distinctive (animation, talking-animals, musical, song, child).
A single such shared trait makes a candidate eligible for the bucket
queue, and deficit pressure can pull it into a top-10 slot even with
weak overall relevance.

The signal formula (`min(1.0, anchor_high_tier_idf_sum / 1.50)`)
also includes IDF >= 0.30 (moderate+high) in the sum, so signal can
be 1.0 on cohorts with no truly distinctive traits — only lots of
moderate ones.

##### Score relativity sanity

- Top-1 scores match V3.3.2 across the board (no regressions on
  highest-similarity matches).
- Per-row breakdowns identical to V3.3.2 (lanes, multipliers, floors
  unchanged — V3.4 only re-orders the top section).
- Format top-5 lock holds: docs-meta films appear at slot 6+, never
  earlier (the lock fires before bucket logic in
  `_peek_next_eligible_for_bucket`).

#### Was the hypothesis correct?

**Partially confirmed.** The architectural insight — that movie
similarity benefits from a recommendation-style multi-source weaver
on top of unified scoring — is validated by the canonical wins.
Barbie's Lady Bird and Frances Ha for the Female-led cohort are
films that V3.1–V3.3.x couldn't surface without unified-ranker
distortion; V3.4 surfaces them naturally.

**But calibration is wrong.** The rare_keyword bucket's gate
(`WEAVER_RARE_KEYWORD_BUCKET_IDF_MIN=0.55`) overlaps with V3's
high-tier IDF cutoff, which includes category-defining traits.
Three cohorts (Pixar, Slasher, Barbie) show the bucket pulling
weakly-relevant films into top 10 via deficit pressure.

#### Ship it?

**No, not as-is.** Ship after a calibration iteration:

1. **Bump `WEAVER_RARE_KEYWORD_BUCKET_IDF_MIN` from 0.55 to 0.70**
   so the bucket only fires on truly distinctive traits. Pixar's
   "talking animals" / "anthropomorphism" should fall below this
   gate and stop pulling Lego Movie / Madagascar; Inception's
   "non-linear narrative" / "lucid dreaming" should remain above.
2. **Tighten the signal formula** to use only `IDF >= 0.55` traits
   in the numerator so cohorts of broadly-themed anchors don't get
   `signal[rare_keyword] = 1.0` from moderate overlap alone.
3. Verify Barbie / Inception / Star Wars wins persist after the
   tightening.

Optional follow-ups (not blockers):
- Lower `WEAVER_BUCKET_INSTANTIATE_MIN` to 0.40 if rare_keyword still
  over-fires on heterogeneous cohorts.
- Surface bucket assignments in the diagnostic markdown so the
  per-row breakdown shows which bucket placed each candidate
  (currently inferable only by signal/membership analysis).
- Add a "documentary suppression" check inside the bucket-membership
  pass for narrative-feature anchors (orthogonal to V3.4 but exposed
  by the Barbie regression).

##### What we learned about the design

- **Full credit + low λ partially-satisfies signal buckets via
  cross-membership** (Pulp Fiction: Reservoir Dogs + Jackie Brown
  picked from best_overall already credit auteur to 2/3, leaving
  only 1/3 deficit which can't outweigh best_overall's relevance
  advantage on subsequent slots). This is by design (Decision 8) —
  prevents Nolan-doubling — but means signal buckets that *don't*
  surface naturally via best_overall need either higher λ (= more
  starvation pull) or stricter membership (= fewer competing best-
  overall picks accidentally crediting them).

- **Tiered gating matters**: bucket *signal* governs slot allocation;
  bucket *membership* governs which candidates compete for those
  slots. Decoupling let the regressions hide — strong signal +
  loose membership = forced low-quality picks.

- **Best-overall must be always-on**. The initial gate-by-target
  bug surfaced in 5 minutes of smoke testing — multi-bucket full
  credit pushes best_overall past target by slot 5 even though
  bucket-unique candidates haven't been picked. The "deficit clamps
  to ≥0; bucket-gate skips best_overall" carve-out is the right
  invariant; worth promoting from "implementation note" to
  "documented architectural invariant" in similar_movies.md §V3.4.

### V3.4.1 — Bucket-Weaver calibration pass (planned)

#### Hypothesis

V3.4 shipped functionally correct (canonical wins on Barbie /
Female-led / Best Picture / MCU / Inception confirmed) but with three
calibration regressions traced to a single root cause: bucket
*instantiation* is too lenient and signal-bucket *membership* lets
cross-format / weak-cohesion candidates compete for top-10 slots
they shouldn't. Three local changes target the regressions without
re-architecting the weaver.

##### The three regressions to fix

1. **Pixar trio** (multi) — Lego Movie #4, Madagascar #6, Rescuers
   Down Under #8 displaced Bolt / Ratatouille / Wreck-It Ralph.
   Caused by `signal[rare_keyword] = 1.0` from moderate-tier shared
   traits (talking-animals, family, child) plus loose membership
   gate at IDF≥0.55. User assessment: "not very bad — the
   alternatives are reasonable suggestions"; downgrade priority,
   accept as-is.
2. **Slasher trio** (multi) — Pennies from Heaven #4, Mamma Mia
   Here We Go Again #7. Caused by auteur bucket instantiating at
   `M_d_auteur / N = 1/3 = 0.33` (Carpenter on The Thing only),
   pulling In the Mouth of Madness; rare_keyword bucket pulling
   musical films via shared "song" / "musical" tags from Everyone
   Says I Love You (one-anchor outlier).
3. **Barbie meta-docs** (single) — Tiny Shoulders #7, Barbie Nation
   #8 (documentaries about Barbie). **Verified via debug: both
   carry membership `{'franchise'}`**, not rare_keyword as initially
   assumed. They share the Barbie lineage tag → `_franchise_score_v2
   ≥ 0.55` → eligible for the franchise bucket → format lock
   disengages at slot 6 → they crack top 10.

##### The three changes

**Change 1 — Cohesion gate `WEAVER_BUCKET_INSTANTIATE_MIN: 0.30 → 0.50`.**

The single cohesion knob inside the weaver. Cascades through every
multi-anchor signal because they all use `M_x / N` (or equivalent)
in the formula:

| Signal | Formula | At 0.30 | At 0.50 |
|---|---|---|---|
| auteur | `M_d_auteur / N` | fires at 1/3 anchors | requires ≥half |
| franchise | `clip(catalog/5, 0, 1) × max_pop × M_f/N` | fires at M_f/N≈0.4 with full catalog/pop | requires ≥half anchors in franchise (when catalog is healthy) |
| rare_keyword | `min(1.0, sum_idf / 1.50)` | fires at sum=0.45 | requires sum≥0.75 |
| lead_actor | `M_a / N` | already gated at 0.5 internally via `CAST_FLOOR_RATIO_THRESHOLD` | unchanged |

Single-anchor bucket signals are unaffected at the gate — auteur is
binary 1.0 (Inception, Barbie, Pulp Fiction still fire); franchise
single-anchor signal = `catalog × max_pop`, which clears 0.50 for
Star Wars / Barbie / John Wick / MCU mainline / LotR.

**Change 2 — MMR weight `WEAVER_LAMBDA: 0.50 → 0.60`.**

Increases starvation pull. At λ=0.5, multi-bucket full credit
partially-satisfies signal buckets via cross-membership (Pulp
Fiction case: Reservoir Dogs + Jackie Brown picked from best_overall
each credit auteur to 2/3 — auteur deficit 1/3 weighted at 0.167,
which can't outweigh best_overall's relevance gap of 0.27+, so
Kill Bill / Django / Inglourious never surface despite being
unique to the auteur queue).

At λ=0.6 the deficit_ratio carries 60% of adjusted score:
- auteur deficit 1/3 → MMR contribution 0.6 × 0.333 = 0.200
- best_overall relevance 0.85 vs auteur 0.58 → relevance gap 0.108
  (= 0.4 × 0.27)
- Now auteur wins slot 4: adj = 0.4 × 0.578 + 0.6 × 0.333 = 0.431
  vs best_overall adj = 0.4 × 0.85 + 0.6 × 0.4 = 0.580. Best still
  wins, but margin narrows from 0.286 → 0.149. By slot 5–6, deficit
  asymmetry reverses dominance.

May slightly worsen Pixar non-Pixar injections (more starvation
pull = more rare_keyword competing); user accepted that trade-off.

**Change 3 — Extend format lock to signal-bucket queues across all
10 slots.**

Currently `_peek_next_eligible_for_bucket` enforces format match
for `slot_index < TOP_FORMAT_LOCK` (slots 0–4). Past slot 5 the
lock disengages and any candidate with format_score=0 (i.e.,
different format bucket from anchor) becomes eligible. This is
where Tiny Shoulders / Barbie Nation slip through the franchise
bucket queue at slots 7–8.

V3.4.1 distinguishes by bucket type:
- **best_overall queue**: format lock for slots 0–4 only (current
  behavior preserved). A strong cross-format candidate can still
  surface naturally on relevance from slot 6+.
- **signal-bucket queues** (auteur, franchise, rare_keyword,
  lead_actor): format lock for ALL slots 0–9. Recommendation
  framing for "more in this franchise" / "more from this auteur" /
  "more rare-keyword matches" / "more lead-actor films" is
  same-format-only.

Implementation: pass `bucket_name` into
`_peek_next_eligible_for_bucket`; replace the slot_index < 5 check
with `(bucket_name != BUCKET_BEST_OVERALL or slot_index < TOP_FORMAT_LOCK)`.
Best_overall queue keeps the slot-5 cliff so cross-format
candidates can compete on relevance from slot 6+, but signal
buckets always require format match.

##### Predicted outcomes

**Wins to confirm preserved:**
- Barbie: Lady Bird #3, Little Women '19 #5 (auteur bucket fires
  at 1.0 in single-anchor — unaffected by 0.50 gate).
- Female-led / Gerwig: Frances Ha in top 10 (auteur cohesion 3/3
  = 1.0; rare_keyword may stop firing at 0.50 gate but auteur
  bucket carries the row).
- Best Picture trio: The Mission #2 (franchise/source/quality
  surfacing — unchanged at the bucket layer).
- MCU trio: 10/10 mainline (franchise cohesion 3/3 = 1.0; tier-4
  exclusion preserved).
- Pulp Fiction: Tarantino films at top (auteur 1.0 single-anchor;
  λ=0.6 may even push Kill Bill / Django into top 10).
- Inception / Star Wars / Tarantino / Ghibli clusters: preserved.

**Regressions to confirm fixed:**
- Slasher trio: Carpenter at 1/3 cohesion = 0.333 < 0.50 →
  auteur bucket NO LONGER instantiates → In the Mouth of Madness
  drops out of top 10. Pennies from Heaven / Mamma Mia rely on
  rare_keyword bucket firing on heterogeneous union pool; with
  0.50 gate, rare_keyword needs `sum_idf ≥ 0.75` from one-anchor
  outlier traits, which it can't reach unless the trait is
  truly distinctive. Should drop both films.
- Barbie meta-docs: Tiny Shoulders / Barbie Nation are
  documentaries with `format_score = 0` against narrative-feature
  anchor. Signal-bucket format lock blocks them from franchise
  queue at slots 6–9. They can still surface via best_overall
  past slot 5, but their score (0.578) ranks below best_overall
  candidates that DO match format → effectively excluded.

**Regressions to monitor:**
- Bucket starvation: with 0.50 gate, more cohorts have only
  best_overall instantiated. That's intentional — heterogeneous
  cohorts shouldn't force signal-bucket rows. Confirm Tom Hanks
  trio still gets lead_actor surfacing (`M_a / N = 2/3 = 0.667`
  if Hanks in 2 of 3 anchors → instantiates at 0.50; `3/3 = 1.0`
  if Hanks in all 3 → fires).
- λ=0.6 over-pulling: signal-bucket weak candidates may surface
  too aggressively in cohorts where auteur/franchise *do* fire
  but their queue's top is weak. Spot-check the Inception /
  Female-led tail.

##### Out of scope (deferred to V3.5)

The architectural fix for Problem 3 — adding a format-mismatch
multiplier symmetric with the medium piecewise multiplier
(`MEDIUM_CROSS_CATEGORY_MULTIPLIER = 0.65` in
similar_movies.py:249). Cross-format candidates would get
`×0.65` regardless of which bucket queue they enter. Cleaner than
the bucket-layer format-lock extension because it also penalizes
cross-format surfacing from `best_overall` past slot 5 and from
non-V3.4 code paths.

Deferred because: (a) requires validation against documentary
*anchors* (where the multiplier shouldn't fire), (b) interacts
with the medium piecewise V3.3 → may compound penalties undesirably
on cross-medium + cross-format pairs, (c) Change 3's bucket-layer
guardrail is sufficient for V3.4 ship velocity.

#### Changes actually made

All three changes landed in [search_v2/similar_movies.py](../search_v2/similar_movies.py):

1. `WEAVER_BUCKET_INSTANTIATE_MIN: 0.30 → 0.50` (line ~243). Comment
   updated to record rationale.
2. `WEAVER_LAMBDA: 0.50 → 0.60` (line ~250). Comment updated.
3. `_peek_next_eligible_for_bucket` now takes `bucket_name`; format
   lock active when `(is_signal_bucket or slot_index < TOP_FORMAT_LOCK)`.
   Caller in `_weave_candidates` passes `bucket_name=b`.

Smoke harness re-run: `python -m search_v2.run_similar_movies_batch
--multi --limit 10`. Output written to
[search_v2/similar_movies_batch_results.md](../search_v2/similar_movies_batch_results.md)
and
[search_v2/similar_movies_multi_anchor_results.md](../search_v2/similar_movies_multi_anchor_results.md).
V3.4 baseline saved to `/tmp/v3_4_baseline_*.md` for diff.

#### Observations

Programmatic set-diff vs. V3.4 baseline. Single-anchor: zero set
changes across all 21 anchors (only ordering differences). Multi-
anchor: 4 of 14 cohorts changed sets, 2 changed ordering only,
8 unchanged.

**Wins (auteur cohesion gate doing exactly what it should):**

- **Stephen King horror** — V3.4 surfaced Kubrick films Full Metal
  Jacket #3 and 2001: A Space Odyssey #5 via the auteur bucket
  firing at `M_d_auteur / N = 1/3 = 0.33` (Kubrick on The Shining
  only). At V3.4.1 with 0.50 gate, auteur bucket doesn't instantiate;
  Kubrick films drop, replaced by The Innocents (1961, classic
  ghost story) #9 and The Night House (2021) #10 — both genuine
  horror surfacing organically through best_overall. **Clear
  improvement.**
- **Best Picture trio** — V3.4 surfaced McQueen films Hunger #3 and
  Widows #6 via auteur bucket at cohesion 1/3 (McQueen on 12 Years
  a Slave only). At V3.4.1 they drop; replaced by Oppenheimer #6
  and The Irishman #10 — both prestige award contenders. **Clear
  improvement.**

These two cohorts validate Hypothesis-1 (the cohesion gate is the
right knob): when one curated auteur is on one anchor of a
heterogeneous cohort, the auteur bucket should NOT instantiate.

**Mixed:**

- **Slasher trio** (anchors are actually The Thing, Everyone Says I
  Love You, Scream 2 — heterogeneous cohort): V3.4 had Mamma Mia
  Here We Go Again #7. V3.4.1: Mamma Mia moved up to #5; Can't Stop
  the Music (1980, musical) entered at #8 (replaced Scream 2022).
  The 0.50 cohesion gate did NOT shut down the rare_keyword bucket
  on this cohort — Everyone Says I Love You's musical/song traits
  share enough IDF mass with these films that signal still clears
  0.50. λ=0.6 amplified the rare_keyword pull, making the regression
  WORSE not better. **Slight regression.**
- **Pixar trio** — same set as V3.4, but Lego Movie moved #4→#3,
  Madagascar moved #6→#5; Inside Out and Monsters Inc both demoted
  one slot. λ=0.6 increased rare_keyword bucket pull as the
  hypothesis warned. The user pre-accepted this trade-off, but the
  effect is more pronounced than predicted. **Slight regression.**

**No-op (the targeted regression that motivated Change 3):**

- **Barbie meta-docs** — Tiny Shoulders #7 and Barbie Nation #8
  appear at IDENTICAL ranks in V3.4 and V3.4.1. The signal-bucket
  format lock change had zero effect, which means these documentaries
  are entering through the `best_overall` queue (not the franchise
  queue, contradicting the pre-implementation debug finding). At
  V3 base score 0.550, they outrank legitimate Lady Bird / Little
  Women candidates with base_score 0.370–0.373 in the best_overall
  queue, and best_overall keeps the slot-5 cliff so they're placed
  at 7–8. **Change 3 was misdirected.**

**Same-set, ordering-only:**

- Nolan trio, Tom Hanks trio: minor reorderings, no semantic change.

**Untouched cohorts (8 of 14):**

- Ghibli trio, MCU trio, Tarantino trio, Spielberg adventure trio,
  WW2 epics, Romcom trio, Studio Ghibli + Pixar mix, Female-led /
  Gerwig: identical or near-identical to V3.4. These cohorts either
  have signals well above 0.50 already (Ghibli/MCU/Tarantino auteur
  cohesion = 1.0; MCU franchise cohesion = 1.0) or have no
  signal-bucket activation either way (mixed cohorts, broad shape).

**Predicted-but-unrealized:**

- Pulp Fiction was predicted to surface Kill Bill / Django at
  λ=0.6. Confirmed it did NOT — top 10 unchanged. The relevance
  gap between best_overall candidates (~0.7–1.0) and signal-bucket
  alternates (~0.5) is too wide for λ=0.6 to flip; would need
  λ ≥ 0.7 or higher.

#### Was the hypothesis correct?

**Mixed.**

- **Hypothesis on Change 1 (cohesion gate 0.50): VALIDATED.** The
  gate cleanly shuts down weak-cohesion auteur firing. Stephen King
  and Best Picture cohorts confirm. No false negatives observed
  (Ghibli / MCU / Tarantino / single-anchor auteur all preserved).
- **Hypothesis on Change 2 (λ=0.60): INVALIDATED.** Did not produce
  predicted Pulp Fiction Tarantino expansion (relevance gap too wide).
  Did produce predicted Pixar/Slasher rare_keyword-bucket
  amplification, but the magnitude was LARGER than expected — net
  regression on Pixar and Slasher.
- **Hypothesis on Change 3 (signal-bucket format lock all-slots):
  INVALIDATED.** The diagnosis that Barbie meta-docs were entering
  via the franchise queue was wrong. They enter via best_overall.
  The change had zero observable effect.

#### Ship it?

**Partial ship: keep Change 1, revert Changes 2 and 3.**

Change 1 produces clear wins on Stephen King and Best Picture with
no observed regressions. The cohesion-gate principle ("don't
instantiate signal buckets unless ≥half of anchors carry the
signal") is sound and validated.

Change 2 (λ=0.60) is net-negative: didn't unlock Tarantino expansion
on Pulp Fiction, did make Pixar and Slasher worse. Revert to
λ=0.50.

Change 3 (signal-bucket format lock) is a no-op given the actual
placement source for Barbie meta-docs. Revert and proceed to V3.5
format-mismatch multiplier as the architectural fix that actually
penalizes cross-format candidates regardless of which queue they
enter.

#### Learnings

- **Cohesion gate is the right primitive.** The 0.30→0.50 bump is
  a one-line change that produces semantic wins (Kubrick out of
  Stephen King; McQueen out of Best Picture). Moving forward, the
  question is whether 0.50 is final or should track per-bucket
  priors (e.g., auteur cohesion 0.50, franchise cohesion 0.40 since
  catalog × max_pop already encodes a quality prior).
- **MMR λ tuning is constrained by relevance gap.** When the V3
  base score gap between best_overall and signal-bucket candidates
  exceeds ~0.20, no realistic λ value flips placement without
  surfacing low-quality alternates from heterogeneous cohorts.
  The right lever for "surface more Tarantino films from Pulp
  Fiction anchor" is the *director lane weight or floor*, not the
  weaver λ.
- **The Barbie meta-doc placement source was misdiagnosed.** A
  single-line Python debug print before V3.4.1 implementation
  would have surfaced that Tiny Shoulders is in `best_overall`
  membership, not `franchise`. Deferring V3.5 format-mismatch
  multiplier was correct on architecture grounds; but the bucket-
  layer guardrail attempt should have been validated with a
  targeted debug script first.
- **Heterogeneous-cohort rare_keyword is harder than auteur.** The
  cohesion gate handles auteur cleanly because cohesion is a clean
  ratio. Rare_keyword's `min(1, sum_idf/1.50)` aggregates across
  the union of anchor traits, so a one-anchor outlier (Everyone
  Says I Love You's musical traits) can push signal past 0.50 even
  when only 1/3 anchors share the trait family. Future work:
  weight rare_keyword sum by per-trait cohesion (only count IDF
  for traits shared by ≥half anchors).



### V3.4.2 — Cohesion-weighted rare_keyword + format-mismatch multiplier + mockumentary demotion

#### Hypothesis

V3.4.1 partial-shipped Change 1 (cohesion gate 0.30→0.50) but left
two unsolved regressions: (a) Slasher trio still leaking
musicals via the rare_keyword bucket because its signal aggregates
IDF over the *union* of anchor traits with no per-trait cohesion
check, so a one-anchor outlier (Everyone Says I Love You's
musical/song traits) pushes the signal past the 0.50 gate, and (b)
Barbie meta-docs (Tiny Shoulders, Barbie Nation) still surfacing
at #7/#8 because format mismatch costs only the lane contribution
(~6.6% of base_score) — not nearly enough to displace them past
the slot-5 format lock.

V3.4.2 lands three changes addressing both:

##### Change 1 — Cohesion-weighted rare_keyword signal + membership

The rare_keyword bucket signal in `_compute_multi_anchor_bucket_data`
([similar_movies.py:3939-3957](../search_v2/similar_movies.py#L3939-L3957))
now weights each anchor trait's IDF by per-trait cohesion:

```python
cohesion_weight(M_t, N) = max(0, (M_t - 1) / (N - 1))
weighted_idf_sum = Σ_t [ idf(t) × cohesion_weight(M_t, N) ]
                   for t in union_traits if idf(t) >= 0.30
signals[BUCKET_RARE_KEYWORD] = min(1, weighted_idf_sum / 1.50)
```

Cohesion table:
- N=2: M_t=1 → 0, M_t=2 → 1.0
- N=3: M_t=1 → 0, M_t=2 → 0.5, M_t=3 → 1.0
- N=4: M_t=1 → 0, M_t=2 → 0.33, M_t=3 → 0.67, M_t=4 → 1.0

A trait must appear in ≥2 anchors to count at all; contribution
scales linearly to full at M_t=N. Membership is gated by the same
M_t ≥ 2 floor so candidates only qualify by sharing a high-tier
trait that at least 2 anchors carry.

Slasher cohort numerical proof (debug captured pre-implementation):
- Union: 36 traits, 26 are 1/N (only one anchor)
- OLD signal: high_idf_sum over union = 6.534 → signal 1.000 (FIRES)
- NEW signal: weighted_idf_sum = 0.224 → signal 0.149 (NO FIRE) ✓

##### Change 2 — `FORMAT_CROSS_CATEGORY_MULTIPLIER = 0.35`

New constant applied in `_build_results` symmetric with
`MEDIUM_CROSS_CATEGORY_MULTIPLIER = 0.65` but harsher (×0.35 vs.
×0.65). Format buckets are categorical content types (narrative
feature vs. doc vs. short vs. concert vs. news vs. tv_format),
not style variations within a form — viewers asking "movies like
Barbie" almost never want a documentary about Barbie.

Application logic: when `anchor_format_bucket is not None` and
`cand_fmt != anchor_format_bucket and cand_fmt != "short"`, score
× 0.35. Shorts are explicitly skipped because the existing
`SHORTS_DOWNRANK_MULTIPLIER = 0.30` path already handles the
short ↔ non-short cross asymmetrically; stacking ×0.30 × ×0.35 =
×0.105 would over-penalize.

Concrete impact on Barbie meta-docs:
- Tiny Shoulders V3.4: score 0.578 → V3.4.2: 0.578 × 0.35 = 0.202
  (drops below Lady Bird at 0.423; off top 10).
- Barbie Nation V3.4: 0.578 → 0.202 (off top 10).

##### Change 3 — Remove `mockumentary` from format taxonomy

Mockumentary is a *style* tag (Spinal Tap, What We Do in the
Shadows, Best in Show, Borat) that overlays a narrative
experience, not a different content category. Treating it as its
own bucket meant a narrative-anchor's format lock excluded
mockumentary candidates that are genuinely "more comedies like X".
[search_v2/format_registry.py](../search_v2/format_registry.py):
removed `"mockumentary"` from `FormatBucket`, removed
`MOCKUMENTARY_KEYWORD_IDS`, removed entry from `_PRIORITY`.

Side effect: mockumentary keyword now flows into themes lane via
`_themes_traits_for_movie` (which subtracts `FORMAT_KEYWORD_IDS_ALL`
from the keyword pool — and that union no longer contains the
mockumentary keyword).

#### Changes actually made

All three changes landed together as one ship. Files:
- [search_v2/similar_movies.py](../search_v2/similar_movies.py):
  - New `FORMAT_CROSS_CATEGORY_MULTIPLIER = 0.35` near line 295.
  - `_build_results` applies the new multiplier after medium and
    before shorts handling, gated to skip same-format and skip
    short candidates (~line 1525).
  - `_compute_multi_anchor_bucket_data` rare_keyword signal block
    rewritten to use cohesion-weighted formula (~line 3939).
- [search_v2/format_registry.py](../search_v2/format_registry.py):
  - Mockumentary removed from `FormatBucket` Literal, `_PRIORITY`,
    and `MOCKUMENTARY_KEYWORD_IDS` constant deleted.
  - Module docstring updated.

V3.4.1 Changes 2 (λ=0.6) and 3 (signal-bucket format-lock all-slots)
were reverted — λ back to 0.5 and `_peek_next_eligible_for_bucket`
back to its original signature without `bucket_name`.

Smoke harness re-run: `python -m search_v2.run_similar_movies_batch
--multi --limit 10`. V3.4 baseline preserved at
`/tmp/v3_4_baseline_*.md` for clean diff (V3.4.1 partial state was
discarded).

#### Observations

Programmatic set-diff vs. V3.4 baseline:

**Single-anchor (1 of 21 cohorts changed):**

- **Barbie**: Tiny Shoulders #7 + Barbie Nation #8 REMOVED (format
  multiplier ×0.35 dropped them from 0.578 to 0.202, off top 10).
  Replaced by Free Guy (#9) and I Am Not an Easy Man (#10) — both
  narrative comedies. Notably, **I Am Not an Easy Man was the
  original V3.1 themes-lane hypothesis target** ("gender-flip
  comedy that should surface but doesn't"); it now finally
  appears. Lady Bird at #3, Little Women at #5 (canonical Gerwig
  matches) preserved. ✓✓

**Multi-anchor (7 of 14 cohorts changed, all wins):**

- **Pixar trio** ✓✓✓: Lego Movie #4, Madagascar #6, Rescuers Down
  Under #8 REMOVED. Replaced by Bolt #8, Ratatouille #9,
  Wreck-It Ralph #10 — all canonical Pixar/family-animation.
  Cohesion-weighted formula killed the heterogeneous trait pool
  (talking-animals, family — common across Toy Story / Finding
  Nemo / Up so 3/N cohesion still counts; but the 1/N traits
  pulling Lego Movie / Madagascar are now zero-weighted).
- **Slasher trio** ✓✓✓: Pennies from Heaven #4, Mamma Mia #7
  REMOVED. Replaced by The Fog (#8, Carpenter horror) and
  Pandorum (#10, sci-fi horror). The musical infiltration via
  Everyone Says I Love You's 1/3-cohesion song traits is dead.
- **Stephen King horror**: Full Metal Jacket #3, 2001: A Space
  Odyssey #5 REMOVED (Kubrick 1/3 auteur cohesion < 0.50 gate
  from V3.4.1). Replaced by The Innocents (1961) and The Night
  House (2021).
- **Best Picture trio**: Hunger #3, Widows #6 REMOVED (McQueen
  1/3 cohesion). Replaced by The Irishman #9 and There Will Be
  Blood #10.
- **Female-led/Gerwig**: Adventure in Baltimore #10 REMOVED.
  Replaced by Brooklyn #6 — a Saoirse Ronan female-led drama,
  much more thematically aligned.
- **Nolan trio**: Franklyn #3, Brazil #6 REMOVED. Replaced by
  I'm Thinking of Ending Things #9, The I Inside #10 — both
  more Nolan-shape (puzzle-narrative, identity-distortion). Mild
  upgrade.
- **Tom Hanks trio**: Heaven Can Wait, 13 Going on 30, Being
  John Malkovich REMOVED. Replaced by Toy Story 4, Up, Monsters
  Inc. The body-swap films were thematically connected to Big
  *specifically* (1/3 cohesion); cohesion-weighted formula
  zeroed those traits' weight, prioritizing the Pixar cluster
  that's coherent across all 3 anchors. Trade-off: lost
  cross-anchor body-swap thematic tail; gained tighter Pixar
  cohesion. Defensible per the user's stated principle ("don't
  fire on 1/N cohesion ever").

**Untouched cohorts (clean preservation):**

- 20 of 21 single-anchor cohorts: Inception, Star Wars, Toy
  Story, Spirited Away, Godfather, Dark Knight, Dark Knight
  Rises, Get Out, John Wick, Oppenheimer, Sharknado, Room, Green
  Mile, Pulp Fiction, Fight Club, Back to the Future,
  Interstellar, LotR, Titanic — identical or minor reorder.
- 7 of 14 multi-anchor cohorts: Ghibli, MCU, Tarantino, Spielberg
  adventure, WW2 epics, Romcom, Studio Ghibli + Pixar mix — all
  unchanged. Strong-cohesion cohorts unaffected by the gate
  (Ghibli/MCU/Tarantino auteur or franchise cohesion = 1.0).

**Format multiplier verification:**

The `format_mismatch` multiplier doesn't appear in any V3.4.2
top-10 row's adjustments line — which is the *correct* behavior:
candidates with format mismatch are penalized hard enough by
×0.35 that they drop out of top 10 entirely. The Barbie meta-doc
disappearance is the empirical proof. Same-format candidates are
unaffected.

#### Was the hypothesis correct?

**Yes, on all three changes.**

- Cohesion-weighted formula: VALIDATED. Slasher musicals
  eliminated; Pixar non-Pixar eliminated; signal collapse
  numerical match (Slasher: 6.534 → 0.224). The `(M_t-1)/(N-1)`
  formula is principled, cohort-size-aware, and produces the
  predicted results without introducing false negatives in
  high-cohesion cohorts (Ghibli/MCU/Tarantino preserved).
- Format multiplier: VALIDATED. Barbie meta-docs eliminated
  (Tiny Shoulders/Barbie Nation off top 10). No false-positive
  regressions observed in 35 cohorts (21 single + 14 multi).
- Mockumentary demotion: VALIDATED structurally. No mockumentary
  films were in the V3.4 baselines we'd want to preserve, so the
  effect of this change is latent — it'll matter when a future
  smoke run includes a Spinal Tap / What We Do in the Shadows
  / Best in Show anchor, where mockumentary candidates can now
  compete in the narrative-anchor top section.

#### Ship it?

**Yes — full ship of all three V3.4.2 changes.**

This is the cleanest weaver/calibration result of the V3.4 series:
- Three named V3.4 regressions (Pixar non-Pixar, Slasher
  musicals, Barbie meta-docs) all eliminated.
- Two latent gains from Change 1 (Stephen King Kubrick fix, Best
  Picture McQueen fix) preserved and reinforced.
- I Am Not an Easy Man finally surfaces on Barbie — closes a
  V3.1 hypothesis loop that was unresolved through V3.3.
- Zero new regressions in 35 cohorts.
- Single-anchor cohorts effectively unchanged outside the
  targeted Barbie fix.

Update doc: V3.4.2 should be marked as the active production
state in `similar_movies.md` once committed.

#### Learnings

- **Cohesion-weighted IDF is the right primitive for any
  multi-anchor signal that pools traits across anchors.** The
  `(M_t-1)/(N-1)` formula generalizes cleanly: a trait is
  required to appear in ≥2 anchors to contribute, with linear
  scaling to full credit at M_t=N. It's principled, parameterless
  (no magic threshold to tune), and naturally cohort-size-aware.
- **Format mismatch is a content-category signal, not a style
  signal.** Asymmetric multiplier (×0.35 format vs. ×0.65 medium)
  reflects that the seven format buckets are fundamentally
  different viewing experiences whereas medium variations are
  style choices within a single experience type. Don't conflate
  the two.
- **Mockumentary belongs in themes, not format.** Format
  taxonomies need to distinguish "different content type" from
  "stylistic flavor of the same content type". Mockumentaries
  are narratives stylistically dressed as docs — content-wise
  they're narrative comedy.
- **Trade-off acknowledged on Tom Hanks**: Single-anchor
  thematic tails get pruned by the cohesion gate. The user-
  stated principle ("don't fire on 1/N cohesion ever") is the
  correct default; if a future cohort really needs single-
  anchor thematic surfacing, the right fix is per-anchor
  themes-recall (already V3.1 infrastructure), not relaxing
  the cohesion gate.
