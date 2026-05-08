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




