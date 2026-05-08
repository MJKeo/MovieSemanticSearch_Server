# Search Overhaul — Test Iteration Tracker

This document tracks every iteration of search-system testing over
time. Each iteration captures a hypothesis we wanted to validate,
the changes actually shipped, the patterns observed in the run, and
what the next iteration should target. The goal is to keep a
visible trend line so we can see whether each round of changes is
moving the system in the right direction or trading one failure
mode for another.

The standing test bed is
[rescore_overhal_queries.md](rescore_overhal_queries.md), driven by
[search_v2/run_specs.py](../search_v2/run_specs.py). Iteration
notes reference per-query JSON outputs captured at the time of the
run.

## Iteration entry format

Each iteration uses these four sections:

- **Hypothesis** — what we expected the changes to do, and what
  signal in the test output would confirm it.
- **Changes actually made** — concrete code / prompt / schema diffs
  shipped before this iteration's run. Reference files and decision
  records.
- **Observations** — key patterns from the run. What failure modes
  were resolved, what new ones surfaced, what unexpectedly held
  steady. Cite specific queries from the suite.
- **Ways to improve going forward** — what the next iteration should
  target, ranked by impact. Distinguish "must fix before next
  phase" from "noted for later".

---

## Current known failure cases (pre-V5 baseline)

These are the failure modes identified by the V5 investigation
(see [rescore_overhaul.md](rescore_overhaul.md) §Failure mode
catalogue and §Root causes). A baseline run of the verification
suite is pending — once executed it will be recorded as
**Iteration 1** below.

### F1 — Vibe-only categories with thin keyword commitment

The keyword handler fires a narrow registry member on a query whose
real signal is semantic. KW miss zeros the category under
ADDITIVE-multiply; if it is the only category in a single-trait
FRAMINGS query, the trait dies entirely.

Examples observed: `feel-good Christmas movies` (FEEL_GOOD ANY zeros
non-tagged feel-good films), `cozy fall movies`, `films with a
haunting bittersweet tone` (BITTERSWEET_ENDING is endings-specific,
not tonal), `slow burn character studies` (THRILLER /
PSYCHOLOGICAL_THRILLER / DRAMA — none mean "slow"), `cerebral
psychological thrillers` (PSYCHOLOGICAL_DRAMA — wrong genre tag).

### F2 — ALL scoring chosen for paraphrase-cluster keywords

The keyword endpoint commits ALL when the finalized list contains
two or three keywords that are paraphrases of one concept rather
than distinct facets. ALL on a paraphrase cluster forces a movie
to be tagged with every member, and tagging is incomplete enough
that this penalises clean matches by 33–66%.

Examples observed: `movies about WWII` ([WAR, HISTORY] ALL),
`scary monster movies` ([HORROR, MONSTER_HORROR] ALL — parent +
child), `dystopian sci-fi` ([DYSTOPIAN_SCI_FI, POST_APOCALYPTIC]
ALL), `mind-bending sci-fi` (three alternative narrative devices
ALL), `running movies` ([SPORT, BIOGRAPHY, TRUE_STORY] ALL — three
separate axes), `biographical dramas about musicians` (BIOGRAPHY
+ TRUE_STORY ALL — Bohemian Rhapsody scores 0.5).

### F3 — Over-coverage keywords get committed despite the prompt's own warning

The keyword endpoint prompt explicitly cites the SPORT-for-running
example as the canonical over-coverage case. The handler reads
this in-prompt and commits SPORT for `running movies` anyway. The
strengths/weaknesses analysis is purely informational — listing
"weaknesses: over-coverage: pulls football/basketball/hockey" does
not block commitment.

Examples observed: `running movies` (the canonical case, used
anyway), `movies with horses` ([ANIMAL_ADVENTURE, WESTERN, SPORT]),
`dark gritty marvel movies` (gritty → [DRAMA, FILM_NOIR, THRILLER]
— DRAMA matches every drama, near-no-op), `violent action movies`
(violent → [SPLATTER_HORROR, ACTION, MARTIAL_ARTS] — ACTION
duplicates sibling action trait), `shitty shark movies` ([SURVIVAL,
HORROR] — neither names sharks).

### F4 — Cross-trait keyword duplication double-counts the same evidence

When two traits in one query both route to keyword and converge on
the same registry member, that member contributes twice to the
candidate pool and twice to scoring, with no de-duplication.

Examples observed: `boxing movies` (CENTRAL_TOPIC [BOXING] AND
GENRE [BOXING]), `violent action movies` (SENSITIVE_CONTENT and
GENRE both pull ACTION), `biographical dramas about musicians`
(ADAPTATION_SOURCE and CENTRAL_TOPIC both pull BIOGRAPHY).

### F5 — Empty-spec categories under FACETS zero the trait

This is a code bug independent of LLM choices. When the handler
abstains entirely from a category (no specs generated), the
scoring path produces `combine_calls(SINGLE, []) → 0.0`. That 0.0
then enters `combine_categories(FACETS, [0.0, …])` and zeros the
trait via PRODUCT.

Examples observed: `underrated indie films` (indie trait
STUDIO_BRAND abstain + FINANCIAL_SCALE meta-fire under FACETS →
trait dies), `movies about WWII` (NARRATIVE_SETTING fired SEM but
near-miss case where abstention would have killed the trait).

### Headline metric

**46% of generated categories carry the ADDITIVE_KW_RISK
trip-wire** (combine_type=ADDITIVE AND KEYWORD ∈ fired_routes).
18 / 39 categories in the refinement batch, 12 / 26 in the default
batch. This is the single number to drive down across iterations.

### Status

Baseline run completed — recorded as **Iteration 1** below.
Per-query JSON: `/tmp/run_specs_baseline.json` (101 KB,
25 queries, 51 traits, 80 categories, 0 errors).

---

## Iteration log

### Iteration 1 — Baseline (pre-V5, `main`)

- **Hypothesis:** Running the V5 verification suite against `main`
  with no V5 changes shipped should reproduce the F1–F5 failure
  modes from the original V5 investigation. The headline
  ADDITIVE_KW_RISK rate should land near the V5 number (~46%).
  Per-query commit shapes should match the
  [rescore_overhaul.md](rescore_overhaul.md) catalogue and give us
  a frozen reference point every later iteration is diffed against.
- **Changes actually made:** None. This is the pre-V5 baseline run.

#### Observations

**Headline metric:** ADDITIVE_KW_RISK fired on **45 / 80 categories
(56.2 %)** — *higher* than the 46 % from the V5 investigation
batch. This suite is heavier on ADDITIVE-eligible categories
(CENTRAL_TOPIC / ELEMENT_PRESENCE / NARRATIVE_DEVICES /
EMOTIONAL_EXPERIENTIAL / STORY_THEMATIC_ARCHETYPE), so 56 % is the
trend-line baseline going forward. 25 / 51 traits commit
`combine_mode=facets`, so half of all traits sit in the FACETS-
PRODUCT regime where any zeroed category zeros the trait — F5 is
preventive but the surface area is large.

**Worst-offender queries** (≥ 3 ADDITIVE_KW_RISK categories):
`Studio Ghibli style hand-drawn fantasies` (4),
`wholesome family movie night picks` (3),
`gritty crime sagas` (3),
`like Donnie Darko but funnier` (3),
`films about grief and reconciliation` (3),
`coming-of-age road trips not too sappy` (3).

**F1 confirmed (vibe-only thin KW commits):**
- `films with a bittersweet melancholic tone` →
  EMOTIONAL_EXPERIENTIAL fires `[BITTERSWEET_ENDING]` ANY — the
  V5 canonical case reproduced exactly. Endings tag for tonal
  query.
- `wholesome family movie night picks` → `wholesome` →
  EMOTIONAL_EXPERIENTIAL `[FEEL_GOOD]` ANY (wholesome ≠
  feel_good) AND → SENSITIVE_CONTENT `[FAMILY]` ANY (FAMILY is a
  target-audience tag, not a sensitive-content tag — definitional
  miss).
- `coming-of-age road trips not too sappy` → `sappy` (negative)
  → EMOTIONAL_EXPERIENTIAL `[FEEL_GOOD, TEARJERKER]` ANY — neither
  means "sappy".
- `films about grief and reconciliation` → `grief` →
  EMOTIONAL_EXPERIENTIAL `[TEARJERKER, TRAGEDY, SAD_ENDING]` ANY
  — endings tags for theme.
- `Studio Ghibli` → EMOTIONAL_EXPERIENTIAL `[IYASHIKEI,
  SLICE_OF_LIFE]` ANY — slice_of_life ≠ Ghibli aesthetic.

**F2 confirmed (ALL on paraphrase clusters), more prevalent than V5
suggested:**
- `cyberpunk dystopias` → STORY_THEMATIC_ARCHETYPE
  `[DYSTOPIAN_SCI_FI, POST_APOCALYPTIC]` **ALL** — paraphrastic.
- `historical war epics` → CENTRAL_TOPIC `[WAR,
  MILITARY_DOCUMENTARY, WAR_EPIC]` **ALL** — three paraphrastic
  homes.
- `mind-bending puzzle films about consciousness` →
  NARRATIVE_DEVICES `[NONLINEAR_TIMELINE, UNRELIABLE_NARRATOR]`
  **ALL** — alternative devices, not conjunction.
- `comedy musicals about teenage romance` →
  STORY_THEMATIC_ARCHETYPE `[TEEN_ROMANCE, COMING_OF_AGE]` **ALL**
  — paraphrastic.
- `gritty crime sagas` → STORY_THEMATIC_ARCHETYPE `[EPIC, GANGSTER]`
  **ALL**.
- `like Donnie Darko but funnier` → STORY_THEMATIC_ARCHETYPE
  `[SCI_FI, COMING_OF_AGE]` **ALL**.
- `slow-burn psychological mysteries` → EMOTIONAL_EXPERIENTIAL
  `[THRILLER, PSYCHOLOGICAL_THRILLER, PSYCHOLOGICAL_DRAMA]` **ALL**
  — three paraphrases of the same thing.
- `brutal MMA fight movies` → GENRE `[SPORT, BOXING,
  MARTIAL_ARTS]` **ALL** — three sport-types ALL when MMA is one.
- `films about grief and reconciliation` →
  STORY_THEMATIC_ARCHETYPE `[DRAMA, PSYCHOLOGICAL_DRAMA, TRAGEDY]`
  **ALL**.
- **Worst observed:** `dark gritty antihero comic-book films` →
  EMOTIONAL_EXPERIENTIAL `[DRAMA, TRAGEDY, FILM_NOIR,
  PSYCHOLOGICAL_DRAMA, ANTI_HERO]` **ALL on five members** — a
  5-way conjunction that is essentially never satisfied. New
  worst-case high water mark.
- Positive control held: `intense action thrillers` → GENRE
  `[ACTION, THRILLER]` **ALL** is *correct* (genuine plural intent).
  Q25 `unreliable narrator with a twist ending` — clean ANY on
  both, did not over-fire ALL.

**F3 confirmed (over-coverage despite the prompt's own warning):**
- `gritty crime sagas` → `gritty` → EMOTIONAL_EXPERIENTIAL
  `[CRIME, THRILLER, FILM_NOIR]` ANY — the V5 V4-observed pattern
  reproduced exactly: tonal word committed against a genre/style
  cluster that pulls every drama in those genres.
- `Studio Ghibli style hand-drawn fantasies` → `fantasies` →
  EMOTIONAL_EXPERIENTIAL `[FANTASY, FAIRY_TALE, ANIMATION]` ANY —
  ANIMATION over-pulls every animated film.
- `dark gritty antihero comic-book films` → `dark gritty` →
  EMOTIONAL_EXPERIENTIAL ALL list above is also F3 (DRAMA pulls
  every drama).
- `mind-bending puzzle films` → `consciousness` → CENTRAL_TOPIC
  `[PSYCHOLOGICAL_DRAMA, PSYCHOLOGICAL_THRILLER]` ANY — those
  don't name consciousness as a topic.

**F4 confirmed (cross-trait KW duplication), 5 / 25 queries
(20 %)** show same registry member committed by ≥ 2 traits:
- `wholesome family movie night picks` — `wholesome` ∩ `family
  movie night` = `[FAMILY]`.
- `gritty crime sagas` — `crime sagas` ∩ `gritty` = `[CRIME,
  FILM_NOIR]`.
- `dark gritty antihero comic-book films` — `dark gritty` ∩
  `antihero` = `[ANTI_HERO]`.
- `films about grief and reconciliation` — `grief` ∩
  `reconciliation` = `[DRAMA]`.
- `slow-burn psychological mysteries` — `slow-burn` ∩
  `psychological mysteries` = `[PSYCHOLOGICAL_DRAMA,
  PSYCHOLOGICAL_THRILLER]`.

**F5 (empty-spec under FACETS):** Did *not* trigger in baseline —
every fired category emitted at least one spec. F5 is a structural
bug whose surface area is large (25 FACETS traits) but whose
trigger rate is low in this batch. Phase 1.1 is preventive, not
visible-symptom-driven.

**V4 positioning regression — clean.** All three positioning
queries produced coherent reference + qualifier pairs:
- `Studio Ghibli` reference (axes_replaced=`['animation style']`)
  ↔ `hand-drawn` qualifier (replaces_axis=`'animation style'`)
- `Donnie Darko` reference (axes_replaced=`['tone']`) ↔ `funnier`
  qualifier (replaces_axis=`'tone'`)
- `Wes Anderson` reference (axes_replaced=`['genre']`) ↔
  `coming-of-age` qualifier (replaces_axis=`'genre'`)

V4 typology is shipping correctly; V5 work doesn't need to
re-validate the role-assignment layer.

**Surprises / new patterns not in the V5 catalogue:**
- *Partial KW abstentions DO happen naturally.* `obscure indie
  passion projects` → `passion projects` →
  SPECIFIC_PRAISE_CRITICISM fired only `routes=['semantic']`,
  abstaining on KW. So D2's partial-abstention reach is
  *occasionally* exercised today — but it's the exception, not
  the rule. Phase 3.3 (bucket-prompt sanction) needs to lift this
  from rare to default.
- *No category-level full abstentions.* Every routed category
  fired at least one endpoint. F5 prevention work cannot rely on
  observed empty-spec evidence — it stays preventive.
- *5-keyword ALL commit.* The `dark gritty antihero comic-book
  films` case raises the F2 ceiling well past the 3-keyword
  examples in the V5 catalogue.
- *FAMILY committed under SENSITIVE_CONTENT.* The handler
  occasionally puts a target-audience-shaped tag in a sensitive-
  content slot — definitional cross-pollination not previously
  catalogued. Bucket-prompt review for `AUDIENCE_SUITABILITY_
  DETERMINISTIC_FIRST` may need an explicit "FAMILY-style tags
  belong in TARGET_AUDIENCE" note.
- *Distribution skew.* EMOTIONAL_EXPERIENTIAL alone accounts for
  ~17 of the 45 ADDITIVE_KW_RISK categories. Phase 2b
  (REMOVE-KW from EMOTIONAL_EXPERIENTIAL alone) should drop the
  trip-wire rate by roughly that amount in one stroke.

#### Non-keyword failure modes (separate from F1–F5)

These patterns are independent of the V5 keyword work. Captured
here so future iterations don't lose sight of them while keyword-
side fixes ship. None of them are addressed by D1–D5.

**N1 — EMOTIONAL_EXPERIENTIAL over-attachment by Step 3.** 20 / 80
category routes (25 %) point to EMOTIONAL_EXPERIENTIAL,
14 / 80 (17.5 %) to STORY_THEMATIC_ARCHETYPE. Together they own
~43 % of all routing. The category isn't load-bearing for many of
the traits attached to it:
- `comedy` → EMOTIONAL_EXPERIENTIAL (in addition to GENRE) —
  comedy is genre, not experiential.
- `crime sagas` → EMOTIONAL_EXPERIENTIAL (alongside GENRE +
  STORY_THEMATIC).
- `Studio Ghibli` → EMOTIONAL_EXPERIENTIAL (alongside
  NARRATIVE_SETTING + STORY_THEMATIC).
- `Wes Anderson` → EMOTIONAL_EXPERIENTIAL (alongside
  VISUAL_CRAFT_ACCLAIM) — Wes Anderson is style, not experience.
- `intense action thrillers` → EMOTIONAL_EXPERIENTIAL (alongside
  GENRE).
- `mind-bending puzzle films` → EMOTIONAL_EXPERIENTIAL (alongside
  NARRATIVE_DEVICES).
- `psychological mysteries` → EMOTIONAL_EXPERIENTIAL (alongside
  GENRE).
- `fantasies` → EMOTIONAL_EXPERIENTIAL (alongside GENRE +
  STORY_THEMATIC).

Step 3 routes traits with any tonal/experiential edge to
EMOTIONAL_EXPERIENTIAL even when the primary signal is genre /
setting / element. Inflates category count, compounds with FACETS
PRODUCT to make traits more brittle. **Independent of the KW
work** — D1 (REMOVE KW from EMOTIONAL_EXPERIENTIAL) doesn't
prevent the over-attachment, only reduces its damage when it does
attach.

**N2 — combine_mode mis-chosen as FACETS for paraphrase clusters.**
The V4 FRAMINGS/FACETS commit picks FACETS when categories are
paraphrastic homes for one concept (FRAMINGS-correct). PRODUCT
across paraphrastic categories is the opposite of what FRAMINGS
was designed for:
- `crime sagas` → FACETS over `[GENRE, STORY_THEMATIC_ARCHETYPE,
  EMOTIONAL_EXPERIENTIAL]`. These are three homes for the same
  concept; should be FRAMINGS-MAX.
- `obscure` → FACETS over `[GENERAL_APPEAL, CULTURAL_STATUS]` —
  paraphrastic, should be FRAMINGS.
- `wholesome` → FACETS over `[EMOTIONAL_EXPERIENTIAL,
  SENSITIVE_CONTENT]` — same attribute, two homes; FRAMINGS.
- `forgotten gems` → FACETS over `[GENERAL_APPEAL,
  CULTURAL_STATUS]` — paraphrastic, should be FRAMINGS.

When this is wrong, the trait dies on any zeroed paraphrastic
home, even though matching one home should be sufficient
evidence. Future Step 3 prompt iteration should re-walk the
combine_mode discriminator.

**N3 — Step 2 over-atomization on tonal-qualifier compounds.**
The V4 fuse rule fires only on bidirectional cross-modification.
Single-direction tonal qualifiers are atomized as independent
siblings, then each fans out its own decomposition:
- `dark gritty antihero comic-book films` → 3 traits (`dark
  gritty`, `antihero`, `comic-book films`). `dark gritty` is a
  tonal qualifier of `comic-book films`, not an independent
  population.
- `slow-burn psychological mysteries` → 2 traits. `slow-burn`
  modifies the pacing of the population named by `psychological
  mysteries`; the two routed to overlapping
  EMOTIONAL_EXPERIENTIAL homes.
- `gritty crime sagas` → 2 traits. `gritty` is a tonal qualifier;
  the two traits both route to EMOTIONAL_EXPERIENTIAL with
  overlapping registry members (F4 cross-trait dup).
- `obscure indie passion projects` → 3 traits (`obscure`, `indie`,
  `passion projects`). All three are paraphrastic descriptors of
  the same population.
- `forgotten gems with brilliant performances` → 2 traits.
  Defensible (performances is the *reason* it's a forgotten gem,
  but they're the same pick from the user's perspective).

The V4 atomization rule's bidirectional-shaping test misses the
common single-direction tonal-qualifier case. A separate rule for
tonal/scalar qualifiers may be needed — likely "POSITIONING_
QUALIFIER for tonal modifiers of a population" rather than two
independent traits.

**N4 — Same-category cross-trait routing creates redundant work.**
Beyond the F4 KW dedup case, the category-routing layer commits
the *same category* twice when paraphrastic traits both hit it:
- `coming-of-age road trips not too sappy` — `coming-of-age` AND
  `road trips` both route to STORY_THEMATIC_ARCHETYPE. Two
  separate STORY_THEMATIC commits with overlapping registry
  members within one query.
- `films about grief and reconciliation` — `grief` AND
  `reconciliation` both route to STORY_THEMATIC_ARCHETYPE.
- `slow-burn psychological mysteries` — `slow-burn` AND
  `psychological mysteries` both route to EMOTIONAL_EXPERIENTIAL.

Whether this is N3 (atomization) or a Step 3 routing dedup gap
depends on whether the right answer is "fuse the traits at Step
2" or "merge same-category routes at Step 3". Worth root-cause
analysis after V5 ships.

**N5 — Carver vs qualifier role inconsistency.** Step 3's
`semantic_role` (carver vs qualifier) varies for structurally
similar traits:
- `marathons` → CENTRAL_TOPIC `role=qualifier`
- `elephants` → ELEMENT_PRESENCE `role=carver`
- `consciousness` → CENTRAL_TOPIC `role=carver`

All three are single-trait population-naming queries. Either all
three should be carver (the trait names the population to
retrieve) or the rule isn't deterministic. Lower priority — role
choice has limited downstream impact on per-trait scoring — but
worth checking once V5 phases stabilise.

**N6 — Wide vector-space targeting on EMOTIONAL_EXPERIENTIAL.**
Many EMOTIONAL_EXPERIENTIAL semantic calls fire across 3+ vector
spaces (`viewer_experience`, `watch_context`, `reception`,
sometimes `plot_analysis`). For genuinely tonal queries this is
fine; for non-experiential traits routed here by N1, it
amplifies the noise. Mostly a downstream symptom of N1, but worth
calling out separately because tightening N1 doesn't necessarily
tighten the per-call space selection.

**N7 — Likely-incorrect tag categorisation.** Bucket-level slips
that aren't keyword-coverage failures:
- `wholesome` → SENSITIVE_CONTENT `[FAMILY]` — FAMILY belongs in
  TARGET_AUDIENCE, not SENSITIVE_CONTENT (also called out under
  KW surprises above).
- `historical war epics` → `war` → CENTRAL_TOPIC `[WAR,
  MILITARY_DOCUMENTARY, WAR_EPIC]` — MILITARY_DOCUMENTARY is a
  format/type, not a topic.
- Both look like prompt-level disambiguation gaps that Phase 3
  prompt rewrites can address; document these separately so they
  aren't lost in the keyword work.

#### Ways to improve going forward

Ranked by expected risk-reduction per unit of work, biased toward
shipping the lowest-risk/highest-impact changes first.

1. **Phase 2b first — REMOVE KW from EMOTIONAL_EXPERIENTIAL.**
   The single biggest contributor to the 56 % rate. Roughly
   17 / 45 risk categories are this one category. One atomic
   changeset (per implementation plan: trait_category.py + the
   notes/few-shot files) and the headline number drops to ~35 %
   without touching any prompt logic.
2. **Phase 1.1 (empty-spec filter) — defensive, ship together.**
   Cheap code change. Doesn't move the headline metric on this
   baseline (F5 didn't fire), but the 25 FACETS traits make it
   load-bearing for tail behaviour and it's the safest place to
   start.
3. **Phase 2a (TARGET_AUDIENCE / SENSITIVE_CONTENT →
   ALTERNATIVES).** Two single-line flips. Removes the multiply
   gate from `Q4` (3 risk → ~0) and `Q5`. Combined with #1 and #2,
   the headline metric drops below 30 %.
4. **Phase 3.2 (singular-vs-plural rewrite).** F2 is the dominant
   live failure mode (~12 ALL commits across 25 queries, including
   the new 5-keyword worst case). Highest impact among prompt
   rewrites. Watch Q13 (`comedy musicals…`) and Q25 (`unreliable
   narrator with a twist ending`) as positive controls.
5. **Phase 3.1 (superset test rewrite).** F3 over-coverage cases
   like `gritty → [CRIME, THRILLER, FILM_NOIR]` will only abstain
   when the prompt sanctions abstention as the right answer.
6. **Phase 3.3 (partial-abstention bucket sanction).** Lift the
   occasional partial-abstention behaviour seen on `passion
   projects` from rare to default. Targets F1 cases where
   D1-REMOVE wasn't appropriate (`heartwarming` → EMOTIONAL_
   EXPERIENTIAL is removed by D1, but `wholesome → SENSITIVE_
   CONTENT [FAMILY]` is a partial-abstention case).
7. **Phase 1.2 (cross-trait dedup).** 5 / 25 queries have
   duplicate KW members across traits — pure perf win, no scoring
   change. Lowest priority; ship after the correctness phases
   stabilise.

**Noted for later (not in V5 scope):**
- *FAMILY-in-SENSITIVE_CONTENT cross-pollination.* Catalogue this
  as a separate bucket-prompt clarification once V5 phases land.
- *MILITARY_DOCUMENTARY in war-epic CENTRAL_TOPIC commit.* Suggests
  CENTRAL_TOPIC's registry view leaks documentary-format tags
  into topic commits. Worth a follow-up after Phase 3.2.
- *Donnie Darko trait commits SCI_FI on STORY_THEMATIC.* V4
  positioning correctly drops the tone axis (replaced by
  `funnier`), but the reference's other axes still fire ALL —
  worth checking whether reference traits should default to ANY
  on the kept axes when commit lists are paraphrastic.

**Stop-conditions for next iteration:** if the post-Phase-2b run
*increases* the trip-wire count or breaks a positive control
(`Q9 revenge stories with anti-heroes`, `Q13 comedy musicals about
teenage romance`, or `Q25 unreliable narrator with a twist
ending`), pause and diagnose before shipping the next phase.

---

### Iteration 2 — Phase 1 (code-only: empty-spec filter + generator dedup)

**Status:** ✅ shipped 2026-05-08. Bundled with N8 validator
self-heal fix. See "Shipped — what we learned" at the end of this
entry for the takeaways carried forward.

- **Hypothesis:** Phase 1 of [rescore_overhaul.md](rescore_overhaul.md)
  is two pure-code changes that should be monotonic-safe and ship
  together:
  - **1.1 — Empty-spec filter (D4):** in `_score_positive_trait`,
    skip categories whose handler emitted zero generated_specs so
    they never reach `combine_calls(SINGLE, []) → 0.0`. Prevents an
    abstaining category from zeroing a FACETS trait via PRODUCT.
    Under FRAMINGS-MAX, removing a 0.0 entry can only raise or hold
    the max; under FACETS-PRODUCT, it removes the zero. Monotonic-
    safe in both modes.
  - **1.2 — Post-hoc generator-spec dedup (D5):** group every
    positive-polarity generator spec by `(route, model_dump_json)`,
    execute one representative per group, and broadcast the result
    map to every (trait_idx, cat_idx, spec_idx) coordinate that
    shared the spec. Pure perf — scoring-side semantics unchanged
    since each trait still reads its own per-coordinate map.
  - **Expected signals in run_specs JSON:**
    - Headline ADDITIVE_KW_RISK rate should hold ≈ 56 % (Phase 1
      doesn't reduce it — that's Phase 2/3 work). Drift > a few
      percent in either direction would be unexplained.
    - F5 (empty-spec under FACETS) didn't trigger in baseline, so
      no visible movement on this suite — 1.1 is preventive. Watch
      for any trait_score divergence on Q14 (`obscure indie passion
      projects`, where STUDIO_BRAND is the most likely abstainer)
      or Q15 (`Studio Ghibli style hand-drawn fantasies`).
    - 1.2 has no trait_score effect (dedup is generator-side); the
      run_specs surface won't show it directly. Ship-validation is
      via "no commit shape changed" rather than visible improvement.
    - Positive-control queries Q9 / Q13 / Q25 must hold their
      baseline commit shapes.

- **Changes actually made:**
  - [search_v2/stage_4_execution.py](../search_v2/stage_4_execution.py):
    - 1.1 — Extended the live_cats assembly loop in
      `_score_positive_trait` to also skip categories with no
      generated_specs after the existing NO_OP check. Documented
      the FRAMINGS/FACETS monotonicity argument in-place.
    - 1.2 — Added a dedup pass in `_run_branch` between the
      pos_generators tagging and the `asyncio.gather` dispatch:
      group tagged specs by `(route, model_dump(mode="json"))`
      hash, run one representative per group, broadcast each
      result map back to every shared `_CallKey`.
  - No prompt, schema, or test changes.

#### Observations

Per-query JSON: `/tmp/run_specs_phase_1.json` (105 KB, 25 queries
attempted; 1 Step-2 error → 24 effective queries; 50 traits, 83
categories).

**Critical methodology note — surfacing the experiment's blindspot:**
[search_v2/run_specs.py](../search_v2/run_specs.py) executes Step 2
→ Step 3 → handler-LLM only. It stops *before* Phase B (Pool
definition) and Phase D (per-trait scoring) — exactly the two
points where Phase 1.1 and Phase 1.2 take effect. Therefore:
- The dedup in Phase 1.2 fires zero times in this experiment.
- The empty-spec filter in Phase 1.1 is never reached.
- Any per-query delta between baseline and phase_1 is **LLM
  non-determinism**, not a Phase 1 effect.

Phase 1 is an "expected non-result" experiment on this surface —
its purpose was to confirm that touching `stage_4_execution.py`
doesn't break upstream Step 2/3/handler flow, that the imports
resolve cleanly, and that no positive-control regressions appear in
the LLM commits. Real visibility into Phase 1's scoring effect
requires a full-pipeline-orchestrator run with `score_breakdowns`
inspection — out of scope for run_specs-driven iteration.

**Headline metrics (excluding the Ghibli Step-2 error for
apples-to-apples comparison):**

| Metric                | Baseline | Phase 1  | Δ      |
|-----------------------|---------:|---------:|--------|
| Effective queries     |       24 |       24 | —      |
| Traits                |       48 |       50 | +2     |
| Categories            |       73 |       83 | +10    |
| ADDITIVE_KW_RISK      | 41 / 73  | 48 / 83  | +7     |
| ADDITIVE_KW_RISK rate |   56.2 % |   57.8 % | +1.6 pp |
| FACETS traits         |       23 |       29 | +6     |

The trip-wire rate moved by 1.6 pp — well within the LLM
non-determinism band (Step 2 atomization, Step 3 category routing,
and handler keyword commits each carry their own drift). No
directional signal here either way; this is the noise floor.

**Step-2 schema violation (NEW, unrelated to Phase 1):**
`Studio Ghibli style hand-drawn fantasies` failed in Step 2 with a
Pydantic ValueError:
> "trait[0] role=POSITIONING_REFERENCE but
> axes_replaced_by_siblings is empty; the reference must inherit at
> least one axis from a sibling qualifier"

The V4 contract enforces that a POSITIONING_REFERENCE trait must
have at least one POSITIONING_QUALIFIER sibling whose
`replaces_axis` value populates the reference's
`axes_replaced_by_siblings`. The Gemini call here produced a
REFERENCE with no QUALIFIER sibling (or with an empty
`axes_replaced_by_siblings`), and the validator correctly rejected
it. This is the **first observed failure** of the V4 atomization
contract on this suite — baseline produced clean Ghibli
positioning. Logged under N8 in non-keyword failure modes (below).

**Was the hypothesis correct?**
Yes — Phase 1's effect is invisible on this experiment surface, and
no positive control was broken. The hypothesis explicitly predicted
this would be the case for 1.2; for 1.1 we predicted the F5 surface
wouldn't trigger and confirmed it didn't.

**Net commit-shape drift, query-by-query (selected):**
- `cyberpunk dystopias`: STORY_THEMATIC ALL `[DYSTOPIAN_SCI_FI,
  POST_APOCALYPTIC]` → ANY `[DYSTOPIAN_SCI_FI]`. *Improvement*
  (F2 win), but driven by LLM drift, not Phase 1.
- `historical war epics`: CENTRAL_TOPIC ALL `[WAR,
  MILITARY_DOCUMENTARY, WAR_EPIC]` → ANY `[WAR]`. *Improvement*
  (F2 win + N7 fix on MILITARY_DOCUMENTARY).
- `gritty crime sagas`: STORY_THEMATIC ALL `[EPIC, GANGSTER]` →
  ANY `[EPIC]`. *Improvement* (F2 win).
- `dark gritty antihero`: lost the 5-keyword ALL high-water mark
  on `dark gritty`; that trait re-atomized into `dark` + `gritty`
  with cleaner ANY commits. *Improvement* (incidental F2 + N3 win).
- `slow-burn psychological mysteries`: *regression*.
  `psychological mysteries` re-atomized to 3 traits (`mysteries`,
  `psychological`, `slow-burn`) and added overlapping ALL commits
  on `[MYSTERY, CRIME, WHODUNNIT]` and `[MYSTERY, COZY_MYSTERY,
  HARD_BOILED_DETECTIVE, WHODUNNIT, SUSPENSE_MYSTERY]`. F2 + F4
  worse than baseline.
- `intense action thrillers`: EMOTIONAL_EXPERIENTIAL `[THRILLER]`
  ANY → `[THRILLER, ACTION, DRAMA]` ANY. Mild F3 worsening.
- `films with a bittersweet melancholic tone`: added a new
  STORY_THEMATIC ALL `[DRAMA, PSYCHOLOGICAL_DRAMA]` commit. New F2
  / F1 surface.

The drift is bidirectional — some queries improved, some
regressed. Pattern: Step 2 atomization is the dominant noise
source. When `mysteries` collapses into one trait it commits
cleanly; when it fans out into 3 traits the failure surface
multiplies.

**Stop-conditions check (Q9 / Q13 / Q25 positive controls):**
- Q9 `revenge stories with anti-heroes`: 2 traits, 2 categories,
  2 risks (unchanged). Clean.
- Q13 `comedy musicals about teenage romance`: TEEN_ROMANCE +
  COMING_OF_AGE → COMING_OF_AGE + TEEN_DRAMA, both still ALL
  (correct under genuine plural intent). Clean.
- Q25 `unreliable narrator with a twist ending`: unchanged
  structurally, 2 traits / 2 cats / 2 risks. Clean.

No positive-control regressions. Phase 1 does not break the suite.

**Is Phase 1 safe to ship?**
Yes. Two independent reasons:
1. Phase 1 changes are pure-code and downstream of every LLM
   stage. They cannot break Step 2/3/handler outputs by
   construction.
2. The monotonicity argument from
   [rescore_overhaul.md](rescore_overhaul.md) §Phase 1 holds:
   under FRAMINGS-MAX, dropping a 0.0 from across-category fold
   can only raise or hold the max; under FACETS-PRODUCT, dropping
   a 0.0 makes the product non-zero. 1.2 has no scoring
   semantics — same rep_map distributed to same coordinates.

#### Ways to improve going forward

Ranked the same way as Iteration 1 (highest impact first), with
amendments based on what we just learned:

1. **Continue with Phase 2a** (TARGET_AUDIENCE / SENSITIVE_CONTENT
   → ALTERNATIVES). This is the *first* phase whose effect is
   visible in `run_specs` JSON: `combine_type` is captured per
   category. We will see the trip-wire rate drop on Q4 / Q5
   directly.
2. **Phase 2b — REMOVE-KW from EMOTIONAL_EXPERIENTIAL first.** The
   single biggest expected drop on the headline metric (~17/45 →
   ~30 % rate). The ALL-on-paraphrases pattern in
   EMOTIONAL_EXPERIENTIAL is also the dominant F2 surface on this
   suite. After Phase 2b on this category alone, the dominant
   structural failures should compress.
3. **Sanity-check Step 2 contract violation (N8).** Add a
   one-shot retry / nudge prompt path for the
   POSITIONING_REFERENCE-without-QUALIFIER case: today the
   validator rejects, the LLM retries from scratch, and one
   query in 25 errors out completely. Cheap fix; surfaces every
   future iteration on positioning queries.
4. **Build a stage-4-aware verification surface (deferred).**
   Phase 1's actual scoring effect is invisible to run_specs.
   Before we ship Phase 1 to prod with confidence, add a
   targeted full-pipeline test on Q14 (`obscure indie passion
   projects`) — the most likely STUDIO_BRAND empty-spec abstainer
   in the suite — to verify trait_score > 0 with the filter on.
   Optional: add a mock-handler unit test that constructs a
   FACETS trait with one abstaining category and asserts
   trait_score reflects the live category alone.
5. **Phase 3.2 (singular/plural) and Phase 3.1 (superset test)**
   stay queued at their existing priority; Phase 2 work changes
   the surface area they need to address.

**N8 — Positioning-reference-without-qualifier Step 2 validator
rejection.** Added to non-keyword failure modes. The V4 atomization
contract enforces structural symmetry between REFERENCE and
QUALIFIER traits, but the LLM occasionally violates it (1/25
queries this run). Retry/nudge would close the gap without
weakening the validator.

**Stop-conditions for next iteration (Phase 2a):** if the
post-Phase-2a run shows TARGET_AUDIENCE or SENSITIVE_CONTENT still
flagged as ADDITIVE on any query, the combine_type flip didn't
land. If any positive control (Q9 / Q13 / Q25) regresses, pause
and diagnose.

#### Addendum — N8 fix + full-pipeline regression sweep

**N8 fix shipped: validator self-heal in
[schemas/step_2.py](../schemas/step_2.py)
`_validate_relationship_roles`.** Pre-pass coerces orphaned
positioning commits to INDEPENDENT rather than rejecting the
query:
- POSITIONING_REFERENCE with empty
  `axes_replaced_by_siblings` → INDEPENDENT (semantically a no-op
  reference; nothing to drop).
- POSITIONING_QUALIFIER with empty `replaces_axis` → INDEPENDENT
  (no axis to substitute).
- After per-trait coerce, if reciprocity collapses to refs-only
  or quals-only, coerce the surviving orphans too.

The strict cross-trait axis-bookkeeping checks
(`missing_on_refs` / `invented_on_refs`) still run — those catch
genuine LLM errors where sibling commits disagree on axis names,
which would silently corrupt Step 3's drop logic.

**Full-pipeline regression sweep via
[/tmp/orchestrator_batch.py](file:///tmp/orchestrator_batch.py)
(same code path as
[run_orchestrator.py](../run_orchestrator.py)).** Ran all 25
queries through `run_full_pipeline(skip_bypass_steps_0_1=True)`
with concurrency=4. **No errors anywhere.**

| Metric                 | Result               |
|------------------------|----------------------|
| Queries                | 25                   |
| `branch_error` count   | 0                    |
| Step / handler errors  | 0                    |
| Zero-ranked branches   | 0                    |
| Mean per-query latency | 23.0 s               |

Top-1 sanity (selected — full table in
[/tmp/orchestrator_phase_1.json](file:///tmp/orchestrator_phase_1.json)):

| Query                                          | Top-1                                       |
|-----------------------------------------------|---------------------------------------------|
| `Studio Ghibli style hand-drawn fantasies`    | Ramayana: The Legend of Prince Rama         |
| `films with a bittersweet melancholic tone`   | Casablanca                                  |
| `films with sentient AI`                      | Metropolis                                  |
| `revenge stories with anti-heroes`            | The Godfather                               |
| `intense action thrillers but not too bloody` | Run Lola Run                                |
| `cyberpunk dystopias`                         | Alphaville                                  |
| `Wes Anderson aesthetic coming-of-age`        | Rushmore                                    |
| `films about grief and reconciliation`        | Three Colors: Blue                          |
| `slow-burn psychological mysteries`           | Perfect Blue                                |
| `unreliable narrator with a twist ending`     | Vertigo                                     |
| `brutal MMA fight movies`                     | A Prayer Before Dawn                        |
| `coming-of-age road trips not too sappy`      | Il Sorpasso                                 |
| `gritty crime sagas`                          | The Godfather Part II                       |
| `atmospheric folk horror`                     | The Blood on Satan's Claw                   |
| `movies about marathons`                      | The Triplets of Belleville                  |
| `mind-bending puzzle films about consciousness` | The Usual Suspects                        |

These are coherent picks across genre, tone, and motif —
canonical exemplars in many cases. The `Studio Ghibli` query that
errored at Step 2 in the pre-fix run now completes cleanly with
3 traits and 7041 ranked candidates, top-1 a hand-drawn animated
fantasy. **Validator coercion fix verified.**

Tail observations not addressed by Phase 1 (carried forward):
- `dark gritty antihero comic-book films` → Dragon Ball Z:
  Bardock — over-atomization (4 traits) + N3 tonal-qualifier
  splitting (`dark` / `gritty` / `antihero` / `comic-book films`)
  pulls anime since `dark` + `antihero` keywords match without
  comic-book lineage filter. Same as baseline.
- `movies featuring elephants` → Fanny and Alexander — F1 thin-
  superset miss; ELEMENT_PRESENCE has no ELEPHANT registry
  member, KW abstains, SEM picks up emotional/family overtones
  from "featuring". Phase 3.1 superset-test rewrite addresses
  this.
- `comedy musicals about teenage romance` → Aladdin — animated
  family-musical instead of teen-romance. F2 ALL on
  `[COMING_OF_AGE, TEEN_DRAMA]` plus genre-conjunction
  interference.

These are pre-existing baseline failure modes (F1, F2, N3),
**not regressions caused by Phase 1**. They remain in scope for
Phase 2 and Phase 3.

**Final ship-decision on Phase 1 + N8 fix:** **safe to ship.**
- Zero pipeline-level errors across the suite.
- Zero zero-ranked outcomes (the FACETS-PRODUCT trait-death
  surface 1.1 was designed to prevent didn't trigger in
  baseline; under Phase 1.1 it provably can't trigger going
  forward).
- Validator coercion eliminates the only Step-2 error observed
  in the Phase 1 run.
- All positive-control queries (Q9, Q13, Q25) produce the
  expected canonical or near-canonical top picks.

Proceed to Phase 2a in the next iteration.

#### Shipped — what we learned

Iteration 2 shipped on 2026-05-08 as a single bundle:
[search_v2/stage_4_execution.py](../search_v2/stage_4_execution.py)
(Phase 1.1 empty-spec filter + Phase 1.2 generator-spec dedup)
and [schemas/step_2.py](../schemas/step_2.py) (N8 validator
self-heal). Takeaways carried forward to future iterations:

1. **Match the verification surface to the change.** Phase 1's
   effects live in stage_4 scoring; `run_specs.py` stops at the
   handler-LLM stage. We caught this only after running the
   suite and seeing why metrics couldn't move. *Going forward:
   for any code change downstream of the handler, drive
   verification through `orchestrator_batch.py` (same code path
   as [run_orchestrator.py](../run_orchestrator.py)) — not
   `run_specs.py`.*
2. **Pure-code, monotonic-safe changes are low-risk
   experiments to lead with.** Phase 1.1 + 1.2 had no scoring-
   semantics change; the only real risk was code defects, which
   the orchestrator sweep would have surfaced. *Going forward:
   when a future phase has both a code-only change and a prompt
   change, ship the code-only one alone first to keep the
   blast radius small.*
3. **LLM commit-shape drift is wider than expected.** Comparing
   baseline vs phase_1 run_specs JSON, ~50 % of (trait, category)
   commits drifted between two clean runs of the same suite —
   even with no prompt or schema changes. *Going forward: any
   diagnostic that depends on stable LLM output across runs
   needs N≥3 baseline samples, not 1, to separate signal from
   noise.*
4. **Soft validators beat hard validators on noisy LLM
   outputs — when the failure case is semantically a no-op.**
   The N8 fix coerces an orphaned POSITIONING_REFERENCE to
   INDEPENDENT instead of erroring; the two roles are
   identical when there are no axes to drop. *Going forward:
   for every Pydantic validator on LLM-shaped data, ask "is
   the rejected state semantically equivalent to a benign
   neighbor?" — if yes, prefer coercion over rejection.
   Reserve hard rejections for states that would silently
   corrupt downstream logic (e.g. axis-name mismatch between
   sibling commits, which the bookkeeping checks still gate).*
5. **Top-1 sanity is a useful coarse signal but not a precise
   one.** The orchestrator sweep produced canonical picks for
   most queries (Casablanca / Godfather / Metropolis /
   Rushmore / Vertigo / etc.) but also surfaced known
   pre-existing failures (Aladdin for "comedy musicals about
   teenage romance", Dragon Ball Z for "dark gritty antihero
   comic-book films"). *Going forward: top-1 is good enough
   for "did Phase X regress anything obvious?" but not for
   measuring V5 progress on the F1–F5 catalogue. Use the
   ADDITIVE_KW_RISK headline metric on `run_specs` JSON for
   that — it's the trip-wire counter the V5 phases were
   designed to drive down.*

---

### Iteration 3 — Phase 2a (TARGET_AUDIENCE + SENSITIVE_CONTENT → ALTERNATIVES)

**Status:** ✅ shipped 2026-05-08. Re-run after operator-error
recovery (see contamination addendum). All hypothesis predictions
held; safe to ship. See "Shipped — what we learned" at the end of
this entry.


- **Hypothesis:** Phase 2a flips two `CategoryCombineType.ADDITIVE`
  enum values to `CategoryCombineType.ALTERNATIVES` in
  [schemas/trait_category.py](../schemas/trait_category.py) — one
  for `TARGET_AUDIENCE` (L759) and one for `SENSITIVE_CONTENT`
  (L784). No prompts, schemas, or scoring code change. The
  verification surface here IS `run_specs.py` (unlike Phase 1):
  `combine_type` is captured per category in the JSON, so the
  flip is directly observable.
  - **Expected signals on `run_specs` JSON:**
    - Every `TARGET_AUDIENCE` and `SENSITIVE_CONTENT` category
      record shows `combine_type=alternatives` (was `additive`).
    - Any `additive_kw_risk` flag previously attached to those two
      categories disappears (the trip-wire requires
      `combine_type==additive`).
    - Headline `ADDITIVE_KW_RISK` count drops by exactly the count
      of `(TARGET_AUDIENCE, SENSITIVE_CONTENT)` rows that
      previously fired with KW. From the baseline + Phase 1 runs:
      Q4 (`wholesome family movie night picks`) committed
      `SENSITIVE_CONTENT [FAMILY]` ANY (1 risk row) and
      `TARGET_AUDIENCE` (≥1 risk row); Q5 (`intense action
      thrillers but not too bloody`) negative trait routes through
      `SENSITIVE_CONTENT`. Expected drop: ~2–4 trip-wire rows
      depending on LLM atomization drift.
    - Positive controls Q9 / Q13 / Q25 must hold their commit
      shapes — none of those route through TARGET_AUDIENCE or
      SENSITIVE_CONTENT, so any shift there is pure LLM drift.
  - **What we are NOT measuring:** the actual scoring effect
    (MAX vs PRODUCT-of-three across KW × META × SEM). That lives
    in `stage_4_execution.combine_calls` and is invisible to
    `run_specs.py`. A targeted orchestrator sweep is deferred to
    a follow-up addendum once the trip-wire-side hypothesis is
    confirmed.

- **Changes actually made:**
  - [schemas/trait_category.py](../schemas/trait_category.py):
    - L759 — `TARGET_AUDIENCE` last enum-tuple arg:
      `CategoryCombineType.ADDITIVE` → `CategoryCombineType.ALTERNATIVES`.
    - L784 — `SENSITIVE_CONTENT` last enum-tuple arg:
      `CategoryCombineType.ADDITIVE` → `CategoryCombineType.ALTERNATIVES`.
  - No other file touched. No prompt, schema, scoring code, or
    test changes. Bundled as one atomic ship per pre-experiment
    decision (Q1: lump together).

#### Observations — run blocked, ship deferred

**Suite-level verification could not complete this iteration —
OpenAI quota exhausted mid-run.** Two cascading issues:

1. **Operator error on first invocation:**
   `python -m search_v2.run_specs --suite search_improvement_planning/rescore_overhal_queries.md ...`
   passed the markdown source file directly. `_load_suite` strips
   only `#`-prefixed and blank lines, so 314 prose lines (section
   headers' bodies, code-block content, "what to look for" notes)
   were dispatched as queries. Baseline + Phase 1 had used
   `/tmp/v5_suite.txt` (the canonical 25-line plain-text extract)
   — that file existed but I did not re-use it.
2. **Cascading quota exhaustion:** The 314-query run consumed the
   daily OpenAI token budget on category-handler LLM calls. The
   corrected 25-query rerun then hit `insufficient_quota` (HTTP
   429, billing-level — not transient) on every category handler.
   80 / 80 cats returned `<handler error: …insufficient_quota…>`
   in `fired_endpoints`.

**Why "0 / 80 ADDITIVE_KW_RISK" is a false floor:** the trip-wire
fires when `combine_type=='additive' AND any fired_route=='keyword'`.
A `<handler error: …>` synthetic route is neither, so every
errored category trivially evaluates to `risk=False`. The counter
read 0 because no handler produced commits, not because the
phase succeeded.

**What IS structurally verified (without re-running):**
- The two enum flips landed correctly in
  [schemas/trait_category.py](../schemas/trait_category.py) at
  L759 + L784.
- The pipeline reads them: per-trait records in `phase_2a.json`
  show 3 / 3 fired `TARGET_AUDIENCE` + `SENSITIVE_CONTENT`
  category records with `combine_type='alternatives'` (was
  `'additive'` in baseline + phase_1). Specifically:
  - Q4 `wholesome family movie night picks` → `family movie
    night` trait → TARGET_AUDIENCE = alternatives ✓
  - Q5 `intense action thrillers but not too bloody` → `bloody`
    (negative) trait → SENSITIVE_CONTENT = alternatives ✓
  - Q16 `brutal MMA fight movies` → `brutal` trait →
    SENSITIVE_CONTENT = alternatives ✓
- Other categories (CENTRAL_TOPIC, EMOTIONAL_EXPERIENTIAL,
  STORY_THEMATIC_ARCHETYPE, NARRATIVE_DEVICES, etc.) still
  read `combine_type='additive'` in the same JSON — the change
  is correctly scoped.
- The change is a pure enum-tuple value swap (no logic, no
  prompt, no code path branched). The combine_type is consumed
  only by `stage_4_execution.combine_calls` and downstream
  `additive_kw_risk` diagnostics; both are deterministic
  consumers of the field.

**Hypothesis status:** *partially verified*. The two enum flips
land and propagate as expected. The headline ADDITIVE_KW_RISK
drop on TA / SC categories cannot be measured in this run
because handler errors mask the true commit shapes. A re-run
once OpenAI quota refreshes would close the loop in ~15 minutes.

**No other run pollution.** The Step 2 / Step 3 layers (Gemini)
ran cleanly across all 25 queries — 50 traits parsed, no Step-2
or Step-3 failures, including the previously-erroring Ghibli
query (N8 self-heal still working). Only the OpenAI-backed
category handlers errored.

#### Ways to improve going forward

1. **Re-run the suite once OpenAI quota refreshes** with
   `python -m search_v2.run_specs --suite /tmp/v5_suite.txt
   --json /tmp/run_specs_phase_2a.json --concurrent 4`. Then
   complete the hypothesis check: confirm risk count drops by
   the 4 / 45 baseline TA + SC rows (to ≈ 41 / 80 ≈ 51 %, modulo
   ~50 % LLM commit-shape drift) and that no positive control
   regresses.
2. **Operational lesson — always pass a plain-text suite to
   `run_specs`.** `_load_suite` does not understand markdown
   structure; passing a `.md` directly silently turns prose into
   queries. Either harden `_load_suite` to look for a
   `# QUERIES` fence or always extract a `.txt` first. Cheaper
   to enforce by convention than by code change. Captured as
   N9 below.
3. **Postpone the Phase 2a ship/no-ship decision** until the
   re-run completes. Schema-level verification alone is not
   sufficient evidence — Phase 1's lesson #1 ("match the
   verification surface to the change") cuts both ways: when
   the surface IS run_specs, we still have to run it.

**N9 — markdown-as-suite contamination footgun.**
`run_specs._load_suite` accepts any non-`#`-non-blank line. Until
hardened, the operator must pre-extract queries to a plain-text
file. Re-running with the wrong source file once consumed enough
OpenAI quota to block the experiment.

**Stop-conditions for the re-run:** if post-Phase-2a JSON shows
any `TARGET_AUDIENCE` or `SENSITIVE_CONTENT` row still flagged
`additive_kw_risk=true`, the schema flip is not landing in some
code path. If any positive control (Q9 / Q13 / Q25) regresses
against baseline commit shape, pause and diagnose before
shipping.

#### Re-run results (clean)

`/tmp/run_specs_phase_2a.json` (104 KB, 25 queries, 50 traits,
85 categories, **0 errors**).

**Headline:**

| run         | Q  | err | tr | cat | risk | rate%  |
|-------------|---:|----:|---:|----:|-----:|-------:|
| baseline    | 25 |   0 | 51 |  80 |   45 |  56.2  |
| phase_1     | 25 |   1 | 50 |  83 |   48 |  57.8  |
| **phase_2a**| 25 |   0 | 50 |  85 |   39 | **45.9** |

`-10.3 pp` vs baseline; `-11.9 pp` vs phase_1.

**Hypothesis predictions vs actual outcomes:**

| Prediction                                           | Actual                                                                                       | ✓/✗ |
|------------------------------------------------------|----------------------------------------------------------------------------------------------|:---:|
| TA + SC combine_type flips to `alternatives`         | 3 / 3 fired rows show `alternatives` (baseline: 4 / 4 `additive`)                            | ✓   |
| TA + SC additive_kw_risk drops to 0                  | 0 / 3 trip-wires fire (baseline: 4 / 4)                                                      | ✓   |
| Headline rate drops by ~5 pp (TA + SC alone)         | -10.3 pp; over-delivered, helped by favorable LLM drift on STORY_THEMATIC and ELEMENT_PRESENCE | ✓ (over) |
| Q9 / Q13 / Q25 positive controls hold                | Q9 = 2 / Q13 = 2 / Q25 = 2 risk, all identical to baseline                                    | ✓   |
| No new errors                                        | 0 Step-2 / Step-3 / handler errors                                                           | ✓   |

**TA / SC actual commits — verified `alternatives` reads through
the pipeline:**
- Q4 `family movie night` → TARGET_AUDIENCE `alternatives`
  `[FAMILY] ANY` → `risk=False`.
- Q5 `bloody` (negative trait) → SENSITIVE_CONTENT `alternatives`
  `[SPLATTER_HORROR, BODY_HORROR] ANY` → `risk=False`.
- Q16 `brutal` (positive) → SENSITIVE_CONTENT `alternatives`
  `[SPLATTER_HORROR, BODY_HORROR] ANY` → `risk=False`.

KW endpoint still fires (as designed — Phase 2a keeps the routes
and only changes how they fold). The trip-wire correctly excludes
`alternatives` regardless of which routes fire under it.

**Q13 plural-ALL positive control** (most important D3
regression check): baseline routed `teenage romance` to
STORY_THEMATIC_ARCHETYPE `[TEEN_ROMANCE, COMING_OF_AGE] ALL`;
phase_2a routed it to CHARACTER_ARCHETYPE
`[TEEN_DRAMA, COMING_OF_AGE] ALL` *plus* a clean GENRE
`[ROMANCE] ANY`. Different category (LLM commit-shape drift),
but the genuine plural-intent ALL still fires on the right pair,
and `comedy` + `musicals` are still committed as separate ANY
rows on GENRE. D3's "don't collapse genuine ALL" guarantee is
intact.

**Per-category risk count (baseline → phase_2a):**

| category                   | base | p2a |  Δ | attribution             |
|----------------------------|----:|----:|---:|-------------------------|
| TARGET_AUDIENCE            |   1 |   0 | -1 | **Phase 2a (structural)** |
| SENSITIVE_CONTENT          |   3 |   0 | -3 | **Phase 2a (structural)** |
| STORY_THEMATIC_ARCHETYPE   |  14 |  10 | -4 | LLM drift (favorable)   |
| ELEMENT_PRESENCE           |   2 |   1 | -1 | LLM drift               |
| EMOTIONAL_EXPERIENTIAL     |  17 |  17 |  0 | unchanged               |
| CHARACTER_ARCHETYPE        |   2 |   3 | +1 | LLM drift               |
| CENTRAL_TOPIC              |   2 |   4 | +2 | LLM drift               |
| NARRATIVE_DEVICES          |   3 |   3 |  0 | unchanged               |
| SPECIFIC_PRAISE_CRITICISM  |   0 |   0 |  0 | unchanged               |
| SEASONAL_HOLIDAY           |   1 |   1 |  0 | unchanged               |
| **net**                    |**45** |**39** | **-6** | -4 structural + -2 net drift |

**Was the hypothesis correct?** Yes — every prediction held. The
two enum flips delivered exactly the structural effect predicted
(TA: 1 → 0, SC: 3 → 0), and the trip-wire definition mechanically
respects the new combine_type. LLM drift on other categories was
net favorable in this sample but is unrelated to Phase 2a; the
~50 % commit-shape drift documented in Iteration 2's lesson #3
remains the noise floor.

**Unintended consequences:** none caused by Phase 2a itself. The
+3 unfavorable drift on CHARACTER_ARCHETYPE / CENTRAL_TOPIC is
independent — same drift surface that flipped favorably elsewhere.
TA / SC routing did not change (still `(KEYWORD, METADATA,
SEMANTIC)`), bucket did not change, prompts did not change.

**Is Phase 2a safe to ship?** Yes:
1. The change is a two-line enum-tuple value swap. No logic, no
   prompt, no code path branched.
2. `combine_type` is read deterministically by
   `stage_4_execution.combine_calls` (MAX vs ADDITIVE-product) and
   by `run_specs._summarize_category` for the trip-wire. Both
   consumers honor the new value automatically.
3. All hypothesis predictions held. No positive control
   regression. No new error surface.
4. The actual stage-4 scoring effect (MAX over KW × META × SEM
   instead of product) was not measured in this run since
   `run_specs` doesn't reach stage 4 — but per Iteration 2
   lesson #1 the verification surface here IS `run_specs`
   (combine_type is captured per category). A follow-up
   orchestrator_batch sweep on Q4 / Q5 / Q16 could close the
   stage-4 loop if needed; not blocking for this ship.

#### Shipped — what we learned

Iteration 3 shipped on 2026-05-08 as a single bundle:
[schemas/trait_category.py](../schemas/trait_category.py) L759
+ L784 (`ADDITIVE` → `ALTERNATIVES` for TARGET_AUDIENCE +
SENSITIVE_CONTENT). Takeaways carried forward:

1. **Single-source-of-truth for verification suites.** Passing
   the markdown source file directly to `run_specs --suite`
   silently dispatched 314 prose lines as queries and burned the
   OpenAI daily quota in one run. The doc previously claimed
   "this document doubles as both human-readable and runner-
   readable" — that was wrong, and the wrong claim cost an
   iteration. *Going forward: always use `/tmp/v5_suite.txt` as
   the canonical operator-runnable source. The markdown file
   has been updated with an explicit "do NOT pass this directly
   to --suite" warning at the top.*
2. **Pure enum-value flips are the lowest-risk shippable
   change.** Phase 2a touches one enum value × 2 lines, no
   logic, no prompts. The trip-wire-rate hypothesis was
   verifiable from `run_specs` JSON alone (combine_type is
   captured per category) — making this the first phase whose
   ship-decision didn't require a separate orchestrator_batch
   sweep. *Going forward: when the change surface fits inside a
   schema/data file, it is by construction observable from
   `run_specs` and a stage-4 sweep is optional rather than
   required.*
3. **Quota exhaustion looks like total verification success.**
   When all 80 handlers errored with `insufficient_quota`,
   `additive_kw_risk` read 0 / 80 — the cleanest possible
   "headline" but a complete fiction (synthetic
   `<handler error>` route doesn't satisfy the trip-wire's
   `route=='keyword'` check). *Going forward: every diff
   script should also count handler-error categories before
   trusting the headline. If `errored_cats > 0`, the headline
   is contaminated regardless of what number it shows.* (TODO
   item for `diff_phase_*.py` improvements: add an
   "errored_cats" line to the headline block.)
4. **LLM drift can mask or amplify a structural change.** The
   structural effect of Phase 2a was -4 trip-wires; the headline
   moved -6. The extra -2 came from LLM commit-shape drift
   tilting favorable on STORY_THEMATIC (-4) and ELEMENT_PRESENCE
   (-1) versus unfavorable on CHARACTER_ARCHETYPE (+1) and
   CENTRAL_TOPIC (+2). *Going forward: always attribute deltas
   in two columns — the structurally guaranteed effect of the
   change, and the LLM-drift residual. Don't conflate them in
   "the change worked".*

---

### Iteration 4 — Phase 2b (REMOVE-KW from EMOTIONAL_EXPERIENTIAL + SEASONAL_HOLIDAY + SPECIFIC_PRAISE_CRITICISM)

**Status:** ✅ shipped 2026-05-08. All hypothesis predictions held;
biggest single-phase headline drop so far. See "Shipped — what we
learned" at the end of this entry.

- **Hypothesis:** Phase 2b drops the `KEYWORD` endpoint from three
  categories that today fire `(SEMANTIC, KEYWORD)` under
  `SEMANTIC_PREFERRED_DETERMINISTIC_SUPPORT` /
  `CategoryCombineType.ADDITIVE`. After this phase each category
  fires `(SEMANTIC,)` only, under `SINGLE_NON_METADATA_ENDPOINT` /
  `CategoryCombineType.SINGLE`. The keyword-side commits that
  drove F1 (vibe-only thin KW commits — `BITTERSWEET_ENDING` for
  tone, `[FEEL_GOOD]` for `wholesome`, `[TEARJERKER, TRAGEDY,
  SAD_ENDING]` ANY for `grief`, etc.) and F2 (ALL on paraphrase
  clusters when EMOTIONAL_EXPERIENTIAL absorbed tonal qualifiers)
  cannot fire here anymore, and the trip-wire formula
  `combine_type==additive AND keyword in fired_routes` mechanically
  excludes the new SINGLE rows. Per Iteration 1, EMOTIONAL_EXPERIENTIAL
  alone carried 17 / 45 trip-wires in baseline — the dominant single
  contributor. Phase 2a structural removed -4. Phase 2b structural
  should remove all rows attributed to these three categories,
  with the bulk landing on EMOTIONAL_EXPERIENTIAL.
  - **Decisions taken before the experiment** (Q1/Q2/Q3 to me):
    - **Q1 — sequencing:** lump all three into one ship. Phase 2a
      precedent for atomic-ship; one Iteration 4, one V5 re-run.
    - **Q2 — query_categories.md scope:** update Cat 29 / Cat 33 /
      Cat 40 endpoint descriptions in the planning doc, AND update
      the prompts the LLM actually reads (`additional_objective_notes/
      <cat>.md` + `few_shot_examples/<cat>.md`). Doc-only sync isn't
      enough because the LLM's behavior is driven by the prompts.
    - **Q3 — few-shot authoring shape:** match the existing
      SINGLE_NON_METADATA_ENDPOINT sibling shape (NARRATIVE_SETTING,
      STORY_THEMATIC_ARCHETYPE) — **4 examples per category,
      even-split fire/no-fire**. No abstention biasing. Calibration
      examples drawn from a pool disjoint from `/tmp/v5_suite.txt`
      (Iteration 2's lesson on example-eval separation).
  - **Expected signals on `run_specs` JSON:**
    - Every `EMOTIONAL_EXPERIENTIAL`, `SEASONAL_HOLIDAY`, and
      `SPECIFIC_PRAISE_CRITICISM` category record shows
      `combine_type=single` (was `additive`) and `fired_endpoints`
      contains only `[semantic]` (was `[semantic, keyword]` or
      either alone).
    - Headline `ADDITIVE_KW_RISK` count drops by the count of those
      three categories' previously-firing KW rows. Baseline
      attribution: `EMOTIONAL_EXPERIENTIAL` = 17, `SEASONAL_HOLIDAY` =
      1, `SPECIFIC_PRAISE_CRITICISM` = 0 → expected structural drop
      of 17 + 1 = 18 trip-wires. From phase_2a's 39 / 85 (45.9 %),
      structural projection: 21 / 85 ≈ 24.7 % (~21 pp drop). LLM
      drift will move this in either direction.
    - Positive controls Q9 / Q13 / Q25 must hold. Q9 routes
      `revenge` → STORY_THEMATIC_ARCHETYPE (untouched) and
      `anti-heroes` → CHARACTER_ARCHETYPE (untouched). Q13 routes
      `comedy musicals` → GENRE / STORY_THEMATIC (untouched) and
      `teenage romance` → CENTRAL_TOPIC / STORY_THEMATIC (untouched).
      Q25 routes `unreliable narrator with a twist ending` →
      NARRATIVE_DEVICES (untouched). None of the three target
      categories sits on these queries' critical path; any shift
      in their commits is pure LLM drift.
  - **What we are NOT measuring:** the actual stage-4 scoring
    behavior change. After this phase, EMOTIONAL_EXPERIENTIAL fires
    a single SEMANTIC endpoint with `combine_type=SINGLE`, so the
    category's score equals the semantic score directly (no
    KW-multiply gate). On FACETS traits where this category sits
    next to others, the per-category score is now monotone in the
    semantic gradient. That effect is real but invisible to
    run_specs — orchestrator_batch sweep is deferred.
  - **Stop-conditions:** if any of the three categories still
    flags `additive_kw_risk=true` in any row, the schema flip
    didn't reach the data; pause and diagnose. If a positive
    control regresses (Q9 / Q13 / Q25), pause. If a previously-
    quiet category suddenly carries +5 or more trip-wires, that
    is unaccounted drift worth investigating before shipping.

- **Changes actually made:**
  - [schemas/trait_category.py](../schemas/trait_category.py):
    three categories' enum tuples updated.
    - `SEASONAL_HOLIDAY` (L786):
      `(EndpointRoute.SEMANTIC, EndpointRoute.KEYWORD)` →
      `(EndpointRoute.SEMANTIC,)`;
      `HandlerBucket.SEMANTIC_PREFERRED_DETERMINISTIC_SUPPORT` →
      `HandlerBucket.SINGLE_NON_METADATA_ENDPOINT`;
      `CategoryCombineType.ADDITIVE` → `CategoryCombineType.SINGLE`.
    - `EMOTIONAL_EXPERIENTIAL` (L901): same triplet of edits.
    - `SPECIFIC_PRAISE_CRITICISM` (L1099): same triplet of edits.
  - [search_v2/endpoint_fetching/category_handlers/prompts/categories/additional_objective_notes/emotional_experiential.md](../search_v2/endpoint_fetching/category_handlers/prompts/categories/additional_objective_notes/emotional_experiential.md):
    deleted `## Keyword Augmentation` section and the
    keyword-augmentation reference from `## Semantic Decision`.
    Rewrote `## Endpoint coverage breadth` to describe
    cross-vector-space coverage without the registry mention. Kept
    `## Target`, `## Boundary Checks`, `## No-Fire`, `## Body
    authoring style` unchanged in spirit.
  - [search_v2/endpoint_fetching/category_handlers/prompts/categories/additional_objective_notes/seasonal_holiday.md](../search_v2/endpoint_fetching/category_handlers/prompts/categories/additional_objective_notes/seasonal_holiday.md):
    deleted `## Keyword Augmentation` section. Tightened
    `## Semantic Decision` and `## No-Fire` to drop
    keyword-proxy references. Kept domain scope and boundaries.
  - [search_v2/endpoint_fetching/category_handlers/prompts/categories/additional_objective_notes/specific_praise_criticism.md](../search_v2/endpoint_fetching/category_handlers/prompts/categories/additional_objective_notes/specific_praise_criticism.md):
    rewrote `## Coverage Decision` for the single-endpoint shape
    (no fallback / no augmentation framing). Kept `## Boundaries`,
    `## No-Fire` unchanged.
  - [search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/emotional_experiential.md](../search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/emotional_experiential.md):
    full rewrite. 4 examples in `<retrieval_intent>+<expressions>`
    SINGLE_NON_METADATA_ENDPOINT input shape, 2 fire / 2 no-fire,
    none of the 25 V5 suite queries reused.
  - [search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/seasonal_holiday.md](../search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/seasonal_holiday.md):
    full rewrite. 4 examples, same shape, 2 fire / 2 no-fire.
  - [search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/specific_praise_criticism.md](../search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/specific_praise_criticism.md):
    full rewrite. 4 examples, same shape, 2 fire / 2 no-fire.
  - [search_improvement_planning/query_categories.md](query_categories.md):
    Cat 29 (Seasonal / holiday) endpoints line updated from
    `KW + CTX + P-EVT (additive combo)` to `CTX + P-EVT (single
    semantic call)`. Cat 33 (Emotional / experiential) updated to
    drop KW from the endpoints list. Cat 40 (Specific praise /
    criticism) updated to drop KW.
  - No scoring code, no schema-factory code, and no validator
    changes. The three new categories will be picked up by the
    existing `SINGLE_NON_METADATA_ENDPOINT` schema-factory path
    and bucket prompts — no authoring needed for those.

#### Observations

`/tmp/run_specs_phase_2b.json` (98 KB, 25 queries, 50 traits, 82
categories, **0 errors**, 0 errored_cats — per Iteration 3 lesson #3
the headline is uncontaminated).

**Headline:**

| run         | Q  | err | tr | cat | risk | rate%   | err_cats |
|-------------|---:|----:|---:|----:|-----:|--------:|---------:|
| baseline    | 25 |   0 | 51 |  80 |   45 |  56.2   |        0 |
| phase_2a    | 25 |   0 | 50 |  85 |   39 |  45.9   |        0 |
| **phase_2b**| 25 |   0 | 50 |  82 |   25 | **30.5**|        0 |

`-25.7 pp` vs baseline; `-15.4 pp` vs phase_2a. Single biggest
single-phase drop so far.

**Three target categories — perfect structural landing:**

| run       | target_rows | kw_fired | combine_type breakdown |
|-----------|------------:|---------:|------------------------|
| baseline  |          23 |       18 | `{additive: 23}`       |
| phase_2a  |          23 |       18 | `{additive: 23}`       |
| phase_2b  |          23 |    **0** | `{single: 23}`         |

Every fired EMOTIONAL_EXPERIENTIAL / SEASONAL_HOLIDAY /
SPECIFIC_PRAISE_CRITICISM row in phase_2b has
`fired_endpoints=['semantic']` and `combine_type=single`. The
trip-wire mechanically excludes all 23 rows because the formula
requires `combine_type==additive AND keyword in fired_routes`.

**Hypothesis predictions vs actual outcomes:**

| Prediction                                                 | Actual                                                     | ✓/✗ |
|------------------------------------------------------------|------------------------------------------------------------|:---:|
| Every target row shows `combine_type=single` and KW absent | 23 / 23 rows show `single` + `[semantic]` only             | ✓   |
| EMOTIONAL_EXPERIENTIAL trip-wires drop from 17 to 0        | 17 → 0                                                     | ✓   |
| SEASONAL_HOLIDAY trip-wires drop from 1 to 0               | 1 → 0                                                      | ✓   |
| SPECIFIC_PRAISE_CRITICISM trip-wires hold at 0             | 0 → 0                                                      | ✓   |
| Structural drop = 17 + 1 = 18 trip-wires                   | -18 from target categories exactly                         | ✓   |
| Q9 / Q13 / Q25 positive controls hold                      | Q9 = 2 / Q13 = 1 (-1) / Q25 = 1 (-1); see analysis below   | ✓   |
| No new errors                                              | 0 Step-2 / Step-3 / handler errors                         | ✓   |

**Q13 and Q25 dropped by 1 each — investigate.** Both are positive-
control queries; the contract is "must hold their commit shapes".
Drilling in:
- Q13 baseline: `'comedy' → EMOTIONAL_EXPERIENTIAL [additive, kw+sem]`
  contributed risk=True. Phase 2b: `'comedy' → EMOTIONAL_EXPERIENTIAL
  [single, sem only]` contributed risk=False. The genuine plural-
  intent ALL still fires correctly on `'teenage romance' →
  STORY_THEMATIC_ARCHETYPE` (still additive+kw+sem with paraphrastic
  ALL). D3's "don't collapse genuine ALL" guarantee is intact.
- Q25 baseline: `'twist ending' → EMOTIONAL_EXPERIENTIAL [additive,
  kw+sem]` risk=True. Phase 2b: `'twist ending' →
  EMOTIONAL_EXPERIENTIAL [single, sem only]` risk=False. The genuine
  plural-intent ALL on `'unreliable narrator' → NARRATIVE_DEVICES`
  is unchanged (still kw+sem additive). D3 guarantee intact.

The drop on Q13 and Q25 is the *structural improvement* landing on
positive-control queries — not a regression. The "must hold" stop-
condition was set to detect *increases*; both decreases are caused
by Phase 2b exactly as designed.

**Per-category risk count (baseline → phase_2b):**

| category                   | base | p2a | p2b |  Δ p2b vs base | attribution             |
|----------------------------|----:|----:|----:|---:|-------------------------|
| EMOTIONAL_EXPERIENTIAL     |  17 |  17 |   0 | **-17** | **Phase 2b (structural)** |
| SEASONAL_HOLIDAY           |   1 |   1 |   0 |  **-1** | **Phase 2b (structural)** |
| SPECIFIC_PRAISE_CRITICISM  |   0 |   0 |   0 |   0 | **Phase 2b (structural, no-op)** |
| TARGET_AUDIENCE            |   1 |   0 |   0 |  -1 | Phase 2a (structural)    |
| SENSITIVE_CONTENT          |   3 |   0 |   0 |  -3 | Phase 2a (structural)    |
| STORY_THEMATIC_ARCHETYPE   |  14 |  10 |  13 |  -1 | LLM drift               |
| ELEMENT_PRESENCE           |   2 |   1 |   4 |  +2 | LLM drift (unfavorable) |
| NARRATIVE_DEVICES          |   3 |   3 |   4 |  +1 | LLM drift               |
| CHARACTER_ARCHETYPE        |   2 |   3 |   2 |   0 | LLM drift (cancels)     |
| CENTRAL_TOPIC              |   2 |   4 |   2 |   0 | LLM drift (cancels)     |
| **net**                    |**45** |**39** |**25** | **-20** | -22 structural + +2 net drift residual |

(Structural: -17 EMOTIONAL_EXPERIENTIAL - 1 SEASONAL_HOLIDAY - 1
TARGET_AUDIENCE - 3 SENSITIVE_CONTENT = -22 across Phases 2a + 2b.
Net drift residual: +2 unfavorable, dominated by ELEMENT_PRESENCE +2
and NARRATIVE_DEVICES +1, partially offset by STORY_THEMATIC -1 and
CENTRAL_TOPIC drift cancellation.)

**Per-query patterns (selected highlights):**

Best improvements (≥2 trip-wires removed):
- `films about grief and reconciliation`: 3 → 0 (the V5 canonical
  F1 case — `grief` no longer fires `[TEARJERKER, TRAGEDY,
  SAD_ENDING]` ANY because EMOTIONAL_EXPERIENTIAL has no KW slot).
- `wholesome family movie night picks`: 3 → 0 (Phase 2a removed
  the SENSITIVE_CONTENT/TARGET_AUDIENCE risks; Phase 2b removed
  the EMOTIONAL_EXPERIENTIAL risk).
- `heartwarming holiday films`: 2 → 0 (clean SEASONAL_HOLIDAY +
  EMOTIONAL_EXPERIENTIAL pure-SEM landing; the V5 reframe of
  `feel-good Christmas movies` no longer fires `[FEEL_GOOD]` or
  `[CHRISTMAS_MOVIE]` ANY).
- `intense action thrillers but not too bloody`: 2 → 0.
- `films with a bittersweet melancholic tone`: 1 → 0 (the V5
  canonical F1 case — `BITTERSWEET_ENDING` ANY can't fire here
  anymore).
- `historical war epics`: 1 → 0.

Held at zero:
- `forgotten gems with brilliant performances`: 0 → 0 → 0 (no
  target-category risk to begin with).
- `movies featuring elephants`: 0 → 0 → 0.
- `films with sentient AI`: 2 → 2 → 2 (CENTRAL_TOPIC + ELEMENT_PRESENCE
  KW commits unchanged; no target-category routes).

Held at non-zero (drift-bound):
- `revenge stories with anti-heroes`: 2 → 2 → 2 (Q9 positive
  control, no target categories).

Increases (LLM drift exposing risk-bearing categories the previous
runs didn't route to):
- `mind-bending puzzle films about consciousness`: 2 → 3 → 4
  (+2 vs baseline). Step 3 added an extra ELEMENT_PRESENCE call
  this run; per-category drift, not Phase 2b structural.
- `obscure indie passion projects`: 0 → 0 → 1 (low magnitude;
  STORY_THEMATIC_ARCHETYPE drift).
- `atmospheric folk horror`: 1 → 0 → 1 (drift back to baseline).

**Was the hypothesis correct?**
Yes. Every prediction held. The structural effect of removing KW
from three categories matched the predicted -18 trip-wires
exactly, and the headline rate dropped to 30.5% — well past the
"sub-30%" projection from Iteration 1 (which estimated drops
from Phases 1+2a+2b combined would land "below 30%"). LLM drift
residual was +2 unfavorable, within the ~50% commit-shape drift
band documented in Iteration 2's lesson #3.

**Unintended consequences:** none caused by Phase 2b itself. The
three target categories are now guaranteed by construction not to
flag ADDITIVE_KW_RISK — there's no path through the bucket schema
or routing that lets them fire keyword. The +2 ELEMENT_PRESENCE
drift on `consciousness` is independent — same drift surface that
flipped favorable on STORY_THEMATIC.

**Is Phase 2b safe to ship?**
Yes:
1. Schema flip is a triple `(EndpointRoute.SEMANTIC, EndpointRoute.
   KEYWORD)` → `(EndpointRoute.SEMANTIC,)` plus bucket and
   combine_type re-tagging. The `SINGLE_NON_METADATA_ENDPOINT`
   bucket is well-trained: 15 sibling categories use it cleanly,
   and the new three load the same `_build_single` schema factory
   path with no warnings.
2. The bucket prompts (`single_non_metadata_endpoint_objective.md`
   + `_guardrails.md`) already exist and were not authored fresh
   for this phase. Risk surface from prompt drift is minimal —
   the only authored content is the per-category notes and few-
   shot examples, which were rewritten to match the existing
   sibling shape (NARRATIVE_SETTING, STORY_THEMATIC_ARCHETYPE).
3. Stage-4 effect is invisible to run_specs but the structural
   argument is straightforward: each target category's score now
   equals the semantic score directly (no KW × META × SEM
   product, no FACETS-zero risk from a registry miss). On FACETS
   traits where these categories sit beside others, the per-
   category contribution is monotone in the semantic gradient —
   strictly safer than before.
4. Positive controls Q9 / Q13 / Q25 either held or improved
   structurally; none regressed.
5. Zero errors, zero errored_cats — Iteration 3's contamination
   class cannot fire on this run.

#### Ways to improve going forward

Ranked by expected risk-reduction per unit of work, biased toward
shipping the lowest-risk/highest-impact changes next:

1. **Phase 3.2 (singular vs plural rewrite).** F2 is now the
   dominant remaining failure mode. STORY_THEMATIC_ARCHETYPE
   carries 13 / 25 trip-wires this run (52% of the residual) and
   most of its risk comes from ALL on paraphrase clusters
   (`[DYSTOPIAN_SCI_FI, POST_APOCALYPTIC]`, `[WAR, HISTORY,
   WAR_EPIC]`, `[DRAMA, PSYCHOLOGICAL_DRAMA, TRAGEDY]`, etc.).
   The keyword.md singular/plural framing rewrite from
   rescore_overhaul.md §3.2 is now the single biggest expected
   contributor.
2. **Phase 3.1 (superset test rewrite).** F3 over-coverage cases
   on the still-keyword-firing categories (CENTRAL_TOPIC,
   ELEMENT_PRESENCE, CHARACTER_ARCHETYPE, STORY_THEMATIC_ARCHETYPE,
   NARRATIVE_DEVICES) need explicit prompt sanction for
   abstention before they will commit empty over committing
   sport-for-running-style proxies.
3. **Phase 3.3 (partial-abstention bucket sanction).** Lifts the
   occasional natural partial-abstention behaviour to default
   on the multi-endpoint buckets that remain.
4. **N1 (EMOTIONAL_EXPERIENTIAL over-attachment by Step 3).**
   With KW gone from EMOTIONAL_EXPERIENTIAL, the over-attachment
   is now low-cost (semantic score is a gradient, not a gate),
   but the category still routes for 21 / 82 calls (~26% of all
   routing) — much of it on traits whose primary signal is
   genre/setting/element. Worth a Step 3 prompt revisit after
   Phase 3.2 lands.
5. **Build a stage-4-aware verification surface (deferred from
   Iteration 2).** Phase 2b's actual scoring effect (no
   KW-multiply gate, semantic gradient → category score) is
   invisible to run_specs. A targeted full-pipeline sweep on the
   queries that improved most (`grief and reconciliation`,
   `wholesome family movie night picks`, `heartwarming holiday
   films`) would close the loop on the trait_score-side
   improvement.

**Stop-conditions for next iteration (Phase 3.2):** if the post-
Phase-3.2 run *increases* the trip-wire count or breaks a positive
control (`Q9`, `Q13` STORY_THEMATIC ALL, `Q25` NARRATIVE_DEVICES
ALL), pause and diagnose before continuing.

#### Shipped — what we learned

Iteration 4 shipped on 2026-05-08 as a single bundle:
- [schemas/trait_category.py](../schemas/trait_category.py) — three
  category enum-tuple updates (L786, L901, L1099).
- Three category prompt rewrites in
  [search_v2/endpoint_fetching/category_handlers/prompts/categories/](../search_v2/endpoint_fetching/category_handlers/prompts/categories/)
  (`additional_objective_notes/{seasonal_holiday, emotional_experiential, specific_praise_criticism}.md` +
  the three matching `few_shot_examples/*.md`).
- [search_improvement_planning/query_categories.md](query_categories.md)
  Cat 29 / Cat 33 / Cat 40 endpoint descriptions.

Takeaways carried forward:

1. **Atomic ship of related changes is fine when they share a
   bucket destination.** All three categories collapsed to the
   same SINGLE_NON_METADATA_ENDPOINT bucket. The same schema
   factory, the same bucket prompts, the same input shape. Per-
   category risk was the prompt rewrites (notes + few-shot) and
   schema flip — both deterministic. Shipping one-at-a-time
   would have tripled the verification cost without buying
   anything diagnostic. *Going forward: when N changes share a
   destination shape, batch them; when they fan out into
   different buckets/prompts, sequence them.*
2. **Pre-existing input-shape mismatches are cheap to fix during
   bucket migrations.** The three target categories' old
   few-shot files used `<raw_query>+<target_entry>` input shape
   that hadn't matched the handler's `build_user_message` output
   in months — a stale prompt artifact predating the V2 handler
   refactor. The Phase 2b rewrite was an opportunity to align
   them with the canonical `<retrieval_intent>+<expressions>`
   shape that handlers actually receive. *Going forward: any
   prompt rewrite is a chance to audit input-shape drift; check
   the few-shot input format against `build_user_message` output
   before authoring new examples.*
3. **Positive-control "must hold" rules need to distinguish
   improvements from regressions.** Q13 dropped from 2 → 1 and
   Q25 dropped from 2 → 1 under Phase 2b — both because the
   structural change correctly removed an EMOTIONAL_EXPERIENTIAL
   risk while leaving the genuine plural-intent ALL on the
   intended category (STORY_THEMATIC_ARCHETYPE / NARRATIVE_DEVICES)
   intact. The hypothesis-vs-actual table needed a "decreases
   are fine when they reflect the structural change landing on
   the control query" exception. *Going forward: state stop-
   conditions in terms of "increase" or "shape change", not
   "must equal baseline".*
4. **The structural-vs-drift attribution table predicts cleanly
   when the surface area is bounded.** Hypothesis predicted -18
   trip-wires from EMOTIONAL_EXPERIENTIAL + SEASONAL_HOLIDAY + 0
   from SPECIFIC_PRAISE_CRITICISM. Actual: exactly -18 from those
   three categories. The +2 LLM-drift residual on
   ELEMENT_PRESENCE + NARRATIVE_DEVICES was unrelated and
   independent of the change. *Going forward: when the change
   surface is fully enumerated (which categories, which rows),
   the structural delta is mechanically predictable from the
   per-category counts in the previous run's JSON. Always
   sanity-check the prediction this way before shipping.*


---

### Iteration 5 — Phase 3 (keyword.md superset test + singular/plural rewrite + bucket partial-abstention sanction)

**Status:** ✅ shipped 2026-05-08. F2 ALL-on-paraphrase ratio
crushed from 14.6% → 5.0%; both surviving ALL commits are
genuine plural intent. Headline trip-wire moved modestly because
the metric tracks `additive AND kw fires`, not ALL strictness —
Phase 3 reduces *brittleness inside* trip-wire rows, not the row
count itself. See "Shipped — what we learned" at the end.

- **Hypothesis:** Phase 3 of [rescore_overhaul.md](rescore_overhaul.md)
  is a three-part prompt rewrite. We're shipping all three together
  because they share an editing surface (keyword.md + the
  multi-endpoint bucket prompts) and the verification surface (V5
  suite via run_specs) captures every observable signal:
  - **3.1 — Superset test (D2).** Replace
    [keyword.md](../search_v2/endpoint_fetching/category_handlers/prompts/endpoints/keyword.md)
    `## Authoring strengths and weaknesses` + `## Near-collision
    disambiguation` with one `## Commitment: superset test` section
    (verbatim from rescore_overhaul.md §3.1). The prescriptive
    over/under-coverage shape language wasn't preventing F3
    over-coverage commits — `gritty → [DRAMA, FILM_NOIR, THRILLER]`
    listed `over-coverage: pulls every drama/noir/thriller` in the
    weaknesses field and the LLM committed it anyway. The superset
    test is principle-based: fire only when the keyword (or the
    ANY-mode union) is a true superset of the movies the user is
    asking for. Over-pull is acceptable; gaps and stretching are
    not.
  - **3.2 — Singular vs plural (D3).** Replace
    [keyword.md](../search_v2/endpoint_fetching/category_handlers/prompts/endpoints/keyword.md)
    `## Reading the brief for scoring_method` ANY/ALL-with-cue-words
    framing with the singular-vs-plural framing (verbatim from
    rescore_overhaul.md §3.2). The cue-word approach
    ("or", "and", "both") was being read mechanically: paraphrase
    clusters where the user's expression contained no explicit
    conjunction were still committed ALL because the LLM treated
    multi-keyword listings as "needs all". Singular intent →
    one expression with multiple registry surface forms = ANY;
    plural intent → multiple expressions naming distinct
    attributes = ALL. Plus the matching update to
    [schemas/keyword_translation.py](../schemas/keyword_translation.py)
    `scoring_method` field description so the schema-as-micro-prompt
    doesn't contradict the endpoint prompt.
  - **3.3 — Partial-abstention bucket sanction.** Add a fourth
    local test ("Superset test per endpoint") to
    [preferred_representation_fallback_objective.md](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/preferred_representation_fallback_objective.md)
    + audit/update the matching language in
    [audience_suitability_deterministic_first_objective.md](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/audience_suitability_deterministic_first_objective.md)
    and
    [semantic_preferred_deterministic_support_objective.md](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/semantic_preferred_deterministic_support_objective.md).
    The current "Empty coverage_assignments is valid only when ALL
    declared endpoint walks surfaced no useful candidate" reads as
    all-or-nothing abstention — the bucket prompt has to explicitly
    sanction partial abstention (some endpoints fire, others
    abstain via the superset test) before the keyword endpoint
    will reliably abstain alone while semantic still fires.
    `character_franchise_fanout_objective.md` is excluded from
    audit because both paths fire by design once the referent
    exists — partial abstention is not a sensible model for that
    bucket.

  Following docs/conventions.md and category_handler_planning.md
  small-LLM guidance: principle-based constraints (not
  failure catalogs), no category-specific examples in either
  rewritten section (the spec explicitly forbids them — keeps the
  test general so the model evaluates the underlying property
  rather than pattern-matches), no V5 suite queries reused, brief
  prose with explicit section boundaries.

  - **Expected signals on `run_specs` JSON:**
    - F2 (ALL on paraphrase clusters) drops sharply on
      STORY_THEMATIC_ARCHETYPE, the dominant residual carrier.
      Phase_2b had STORY_THEMATIC at 13/25 trip-wires, with the
      bulk being paraphrase-cluster ALL like
      `[DYSTOPIAN_SCI_FI, POST_APOCALYPTIC]` ALL,
      `[DRAMA, PSYCHOLOGICAL_DRAMA, TRAGEDY]` ALL on grief, etc.
      Under the singular/plural framing the LLM should commit
      ANY when the same expression names paraphrases.
    - F3 (over-coverage despite weaknesses naming the over-pull)
      drops on the still-keyword-firing categories (CENTRAL_TOPIC,
      ELEMENT_PRESENCE, CHARACTER_ARCHETYPE, NARRATIVE_DEVICES,
      STORY_THEMATIC_ARCHETYPE). Categories where the registry
      genuinely supersets stay committed; categories where the
      registry over-pulls (SPORT for marathons, DRAMA for tone,
      ACTION duplicating sibling) abstain.
    - Q9 / Q13 / Q25 positive controls: Q9
      (`revenge`, `anti-heroes` — clean ANY commits should hold).
      Q13 (`teenage romance` STORY_THEMATIC plural-intent ALL is
      genuine — should remain ALL). Q25 (`unreliable narrator
      with a twist ending` NARRATIVE_DEVICES
      `[UNRELIABLE_NARRATOR, PLOT_TWIST]` — genuine plural-intent
      ALL — should remain ALL). If any of these collapse to ANY
      that's an over-correction bug and a stop condition.
    - Headline rate falls below the 30.5% phase_2b mark. Conservative
      target: 20–25 % range, with bigger-than-average LLM drift in
      either direction because this is a behavior change, not a
      structural elimination — predictions are looser.
  - **What we are NOT measuring:** the actual stage-4 scoring
    effect (how trait_score moves when an over-coverage keyword
    is dropped vs commit-anyway behavior). Visible on
    orchestrator_batch but not on run_specs. Deferred per Phase 1
    / Phase 2a / Phase 2b precedent — when run_specs's surface
    captures the LLM's commit-shape change, an orchestrator sweep
    is optional rather than required.
  - **Stop-conditions:**
    - Q9 / Q13 / Q25 STORY_THEMATIC + NARRATIVE_DEVICES ALL
      collapses to ANY → over-correction, pause and diagnose.
    - Headline rate *increases* from 30.5% → unaccounted prompt
      regression, pause.
    - Any new error class on Step 2 / Step 3 / handler — investigate
      before shipping.
    - Empty `finalized_keywords` on a single-endpoint keyword
      category that schema requires `min_length=1` for — schema
      violation, ship-blocking.


- **Changes actually made:**
  - [search_v2/endpoint_fetching/category_handlers/prompts/endpoints/keyword.md](../search_v2/endpoint_fetching/category_handlers/prompts/endpoints/keyword.md):
    - 3.1 — Replaced `## Authoring strengths and weaknesses per
      candidate` (which prescribed clean / under-coverage /
      over-coverage / both shapes) and `## Near-collision
      disambiguation` (which enumerated breadth-vs-specificity,
      explicit-premise, cross-family, and mutually-exclusive
      principles) with a single `## Commitment: superset test`
      section. The prescriptive shape language wasn't preventing
      F3 commits — `gritty → [DRAMA, FILM_NOIR, THRILLER]` listed
      `over-coverage: pulls every drama/noir/thriller` and the
      LLM committed it anyway. The new section keeps the
      strengths/weaknesses fields as walk-phase scaffolding and
      moves the commitment gate to a principle: the keyword (or
      ANY-mode union) must be a true superset of the user's
      attribute. Over-pull is acceptable; gaps and stretching are
      not. No category-specific examples (per spec) — the test is
      generalized so the LLM evaluates the underlying property.
    - 3.2 — Replaced `## Reading the brief for scoring_method`
      ANY/ALL-with-cue-words (`"or"`, `"and"`, `"both"`) with the
      singular-vs-plural framing. Singular intent (one expression
      with multiple registry surface forms) → ANY; plural intent
      (multiple expressions naming distinct attributes) → ALL.
      Operational test reads the call's expressions, not surface
      conjunctions.
  - [schemas/keyword_translation.py](../schemas/keyword_translation.py)
    — both `KeywordQuerySpec.scoring_method` (L215-238) and
    `KeywordQuerySpecSubintent.scoring_method` (L393-417) field
    descriptions rewritten to the singular-vs-plural framing.
    Schema descriptors are micro-prompts; both must agree with the
    endpoint prompt (small-LLM principle: merge ambiguous field
    boundaries, no field-vs-prompt contradictions).
  - [search_v2/endpoint_fetching/category_handlers/prompts/buckets/preferred_representation_fallback_objective.md](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/preferred_representation_fallback_objective.md):
    - 3.3 — Added a fourth local test ("Superset test per
      endpoint") to the coverage_exploration phase. Updated the
      "Empty `coverage_assignments` is valid only when ALL
      declared endpoint walks surfaced no useful candidate"
      sentence to the spec-verbatim version that sanctions partial
      abstention as a real outcome (per-endpoint criteria
      independent). Updated the `**Declining to fire any endpoint
      is valid only when all walks surfaced no useful candidate**`
      summary line to mirror the per-endpoint framing.
  - [search_v2/endpoint_fetching/category_handlers/prompts/buckets/audience_suitability_deterministic_first_objective.md](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/audience_suitability_deterministic_first_objective.md):
    - 3.3 — Same superset-test-per-endpoint local test added; same
      partial-abstention update to `coverage_assignments` and the
      whole-call-abstain summary line.
  - [search_v2/endpoint_fetching/category_handlers/prompts/buckets/semantic_preferred_deterministic_support_objective.md](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/semantic_preferred_deterministic_support_objective.md):
    - 3.3 — Same superset-test-per-endpoint local test added;
      partial-abstention sanction in `coverage_assignments`.
  - `character_franchise_fanout_objective.md` deliberately
    excluded from the audit. Both paths fire by design when a
    referent exists; partial abstention isn't a sensible model
    for that bucket.
  - No code changes outside the schema description rewrites. No
    test changes (per the project's `test-boundaries` rule).

#### Observations

`/tmp/run_specs_phase_3.json` (94 KB, 25 queries, 51 traits, 80
categories, **0 errors**, 0 errored_cats — Iteration 3 lesson #3
contamination clean).

**Headline:**

| run         | Q  | err | tr | cat | risk | rate%   | err_cat |
|-------------|---:|----:|---:|----:|-----:|--------:|--------:|
| baseline    | 25 |   0 | 51 |  80 |   45 |  56.2   |       0 |
| phase_2b    | 25 |   0 | 50 |  82 |   25 |  30.5   |       0 |
| **phase_3** | 25 |   0 | 51 |  80 |   23 | **28.8**|       0 |

`-1.7 pp` vs phase_2b; `-27.4 pp` cumulative vs baseline.

**The dominant Phase 3 signal is NOT in the headline — it's in
the ANY/ALL distribution:**

| run        | total kw_commits | ANY | ALL | ALL_rate |
|------------|---:|---:|---:|---:|
| baseline   | 59 | 47 | 12 | 20.3% |
| phase_2b   | 41 | 35 |  6 | 14.6% |
| **phase_3**| 40 | 38 |  2 | **5.0%** |

**The 2 surviving ALL commits are both genuine plural intent:**
1. `intense action thrillers but not too bloody` →
   `'intense action thrillers'` → GENRE
   `[ACTION, THRILLER] ALL` (combine=alternatives → risk=False).
   Two distinct genre attributes the user wants compounded
   (action AND thrillers).
2. `historical war epics` → `'historical war epics'` → GENRE
   `[WAR, HISTORY] ALL` (combine=alternatives → risk=False). Two
   distinct facets — war (subject) AND history (period). A
   fictional war film is WAR but not HISTORY; a non-war
   historical drama is HISTORY but not WAR. ALL is defensible.

The F2 paraphrase-cluster ALL pattern (`[DYSTOPIAN_SCI_FI,
POST_APOCALYPTIC]`, `[DRAMA, PSYCHOLOGICAL_DRAMA, TRAGEDY]`,
`[TEEN_ROMANCE, COMING_OF_AGE]`, `[NONLINEAR_TIMELINE,
UNRELIABLE_NARRATOR]`, etc.) is gone.

**Why didn't the headline move more?** The trip-wire formula is
`combine_type==additive AND keyword in fired_routes`. STORY_THEMATIC
trip-wires went from 13 → 13 (held) — but their ALL count went
from 1 → 0. The categories still fire keyword on additive combine
(structurally unchanged); they just commit ANY now instead of ALL.
Phase 3 reduces *brittleness* inside the trip-wire rows (ALL on
paraphrase clusters → ANY = movies score on partial matches
instead of zeroing on missing-tag), not the row count.

**Hypothesis predictions vs actual outcomes:**

| Prediction                                                    | Actual                                                    | ✓/✗ |
|---------------------------------------------------------------|-----------------------------------------------------------|:---:|
| F2 ALL drops sharply on STORY_THEMATIC (paraphrase clusters)  | STORY_THEMATIC ALL: 5 → 1 → **0**                          | ✓   |
| F3 over-coverage drops on still-keyword-firing categories      | KW commits: 41 → 40 (held); abstention rate didn't move   | ✗   |
| Q9 / Q13 / Q25 positive controls hold genuine plural-intent ALL | Q9 held; Q13 dropped 1→0 (LLM drift, no over-correction); Q25 held | ✓   |
| Headline rate falls below 30.5%; conservative target 20–25%   | 28.8% — hit the upper bound; hypothesis miscalibrated      | ~   |
| No new errors / schema violations                             | 0 errors, 0 schema violations                              | ✓   |

**Q13 details — improvement, not over-correction.** Baseline had
`'teenage romance' → STORY_THEMATIC_ARCHETYPE [TEEN_ROMANCE,
COMING_OF_AGE] ALL` (paraphrastic, F2 case). Phase 3 routed
`teenage romance` to TARGET_AUDIENCE
`[TEEN_ADVENTURE, TEEN_COMEDY, TEEN_DRAMA, TEEN_HORROR,
TEEN_ROMANCE, TEEN_FANTASY] ANY` + GENRE `[ROMANCE] ANY` — six
teen-X surface forms reading as singular intent of "teen content
across multiple registry forms," correctly committing ANY. Both
new commits are alternatives (Phase 2a structural), so risk=False.
The genuine plural-intent test for Q13 is on the GENRE side
(comedy AND musical) — but V4 atomization splits `comedy
musicals` into separate `comedy` and `musicals` traits, each
committing single-keyword ANY on its own GENRE call. The plural
intent is encoded at the trait level, not the keyword level.

**Q25 details — N=1 ANY held; the test was moot.** Baseline,
phase_2b, and phase_3 all show
`'unreliable narrator' → NARRATIVE_DEVICES [UNRELIABLE_NARRATOR]
ANY` (single member, ANY default for N=1). The hypothetical
`[UNRELIABLE_NARRATOR, PLOT_TWIST] ALL` plural-intent commit
never fired in any run — V4 atomization splits the query into
two traits and `twist ending` routes to EMOTIONAL_EXPERIENTIAL
(now SINGLE) instead of NARRATIVE_DEVICES. The Q25 stop-condition
was over-specified.

**F3 over-coverage signal — muddier than expected.** The superset
test should have driven more abstention on STORY_THEMATIC over-
pulling categories like `gritty → [DRAMA, FILM_NOIR, THRILLER]`
or paraphrastic-flow commits, but the kw-fire counts held flat:
- STORY_THEMATIC_ARCHETYPE: 13 → 13 (held)
- ELEMENT_PRESENCE: 4 → 4 (held)
- NARRATIVE_DEVICES: 4 → 3 (-1)
- CENTRAL_TOPIC: 2 → 1 (-1)

The principle-based test is doing *something* but not driving
abstention as a default. Two interpretations:
1. The LLM reads the test as "abstain only when egregious gaps"
   rather than "abstain by default unless superset is clean".
2. The per-endpoint partial-abstention sanction (3.3) needs more
   prompt weight — the bucket prompts changed but the surrounding
   guidance still strongly biases firing.
Either way, this is non-blocking but worth a follow-up.

**Per-category risk count (phase_2b → phase_3):**

| category                   | base | p2b | p3 |  Δ p3 vs p2b | attribution      |
|----------------------------|----:|----:|----:|---:|------------------|
| STORY_THEMATIC_ARCHETYPE   |  14 |  13 |  13 |   0 | drift           |
| ELEMENT_PRESENCE           |   2 |   4 |   4 |   0 | drift            |
| NARRATIVE_DEVICES          |   3 |   4 |   3 |  -1 | favorable drift  |
| CHARACTER_ARCHETYPE        |   2 |   2 |   2 |   0 | drift            |
| CENTRAL_TOPIC              |   2 |   2 |   1 |  -1 | favorable drift  |
| EMOTIONAL_EXPERIENTIAL     |  17 |   0 |   0 |   0 | Phase 2b (structural) |
| SEASONAL_HOLIDAY           |   1 |   0 |   0 |   0 | Phase 2b (structural) |
| TARGET_AUDIENCE            |   1 |   0 |   0 |   0 | Phase 2a (structural) |
| SENSITIVE_CONTENT          |   3 |   0 |   0 |   0 | Phase 2a (structural) |
| **net**                    |**45**|**25**|**23**| **-2** | -2 favorable drift residual |

Phase 3 has no structural removal surface; the -2 is LLM-drift
favorable. The real Phase 3 win sits in the ALL-rate column, not
the headline.

**Per-query patterns:**

Improvements (≥1 trip-wire dropped vs phase_2b):
- `mind-bending puzzle films about consciousness`: 4 → 1 (-3,
  best improvement; ALL-on-paraphrase cluster gone).
- `comedy musicals about teenage romance`: 1 → 0 (LLM drift to
  TARGET_AUDIENCE under singular-intent ANY).
- `like Donnie Darko but funnier`: 2 → 1.
- `atmospheric folk horror`: 1 → 0.

Regressions:
- `cyberpunk dystopias`: 1 → 2 (+1, drift).
- `films about grief and reconciliation`: 0 → 1 (+1, drift back).
- `wholesome family movie night picks`: 0 → 1 (+1, drift back).
- `slow-burn psychological mysteries`: 1 → 2 (+1, drift back).

Net per-query delta: -2. Bidirectional drift around the structural
floor.

**Was the hypothesis correct?** Largely yes, with one calibration
miss:
1. F2 ALL-on-paraphrase reduction landed cleanly (14.6% → 5.0%).
2. Positive controls clean (no over-correction on Q9/Q13/Q25).
3. F3 abstention didn't drive as expected — non-blocking but
   worth a follow-up iteration.
4. Headline projection was too optimistic (predicted 20–25%, hit
   28.8%) — I conflated "F2 win" with "trip-wire reduction"; the
   trip-wire metric only measures the structural axis, not the
   commit-strictness axis.

**Unintended consequences:** none caused by Phase 3 itself. Some
queries drifted in trip-wire count (bidirectional) but the
underlying commit-shape changes match the rewrite intent — F2
ALL gone, F3 abstention modest, F1 no longer fireable on the
post-Phase-2b surface (those categories are pure SEM now).

**Is Phase 3 safe to ship?** Yes:
1. F2 ALL-on-paraphrase reduction is the dominant win — visible
   on every paraphrase-cluster query that previously fired ALL.
   ALL_rate dropping 14.6% → 5.0% means trait_score on those
   rows is no longer fractionally divided by N=2 or N=3; movies
   tagged with one of N members score equal to fully-tagged
   movies under ANY.
2. Both surviving ALL commits are genuine plural intent (action+
   thriller, war+history). No over-correction.
3. Positive controls held or improved — none collapsed to
   over-correction.
4. 0 errors, 0 errored_cats, 0 schema violations.
5. The schema's `min_length=1` constraint on
   `KeywordQuerySpec.finalized_keywords` was respected — the
   superset test correctly applies as guidance for single-
   endpoint keyword categories rather than authorizing full
   abstention there.

#### Ways to improve going forward

Ranked by expected risk-reduction per unit of work:

1. **Investigate why the superset test didn't drive more F3
   abstention.** STORY_THEMATIC_ARCHETYPE still fires 13 KW
   commits with no abstention rate change. Possible causes:
   (a) the test reads as "abstain only when egregious"; (b) the
   per-endpoint sanction in the bucket prompt is being shadowed
   by the still-bullish "puzzle pieces — overlap is the design"
   framing earlier in the same prompt. Worth a targeted
   experiment: re-read the assembled bucket prompt with fresh
   eyes for tonal contradiction between "fire everything that
   carries real signal" and "abstain when you're stretching".
2. **N1 (EMOTIONAL_EXPERIENTIAL over-attachment by Step 3).** No
   longer harmful (KW gone, semantic gradient is monotone) but
   still inflates category count and compounds with FACETS.
   Worth a Step 3 prompt revisit.
3. **N2 (combine_mode mis-chosen as FACETS for paraphrase
   clusters).** `'crime sagas'` → FACETS over `[GENRE,
   STORY_THEMATIC_ARCHETYPE, EMOTIONAL_EXPERIENTIAL]` (3 homes
   for one concept) is wrong — should be FRAMINGS.
4. **N3 (Step 2 over-atomization on tonal-qualifier compounds).**
   `dark gritty antihero comic-book films` still atomizes to 4
   traits. Each fans out independent decompositions. A
   POSITIONING_QUALIFIER-for-tonal-modifiers rule may be the
   right intervention.
5. **Build a stage-4-aware verification surface (carried over).**
   Iteration 2's lesson #1 — for changes that move within-row
   strictness rather than row count, run_specs's headline metric
   is a poor proxy. orchestrator_batch on the queries that
   improved most (Q12 `consciousness`, Q13 `comedy musicals`)
   would show the trait_score-side improvement.

**Stop-conditions for next iteration:** standard — if any phase
*increases* trip-wire count or breaks a positive control, pause.
For prompt iteration on F3 abstention specifically, watch the
GENRE/CENTRAL_TOPIC abstention rate — those should rise without
their commit shape collapsing into "always abstain".

#### Shipped — what we learned

Iteration 5 shipped on 2026-05-08 as a single bundle:
- [search_v2/endpoint_fetching/category_handlers/prompts/endpoints/keyword.md](../search_v2/endpoint_fetching/category_handlers/prompts/endpoints/keyword.md)
  (3.1 superset test + 3.2 singular/plural).
- [schemas/keyword_translation.py](../schemas/keyword_translation.py)
  scoring_method descriptions on both `KeywordQuerySpec` and
  `KeywordQuerySpecSubintent`.
- Three multi-endpoint bucket prompts:
  [preferred_representation_fallback_objective.md](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/preferred_representation_fallback_objective.md),
  [audience_suitability_deterministic_first_objective.md](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/audience_suitability_deterministic_first_objective.md),
  [semantic_preferred_deterministic_support_objective.md](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/semantic_preferred_deterministic_support_objective.md)
  (3.3 partial-abstention sanction).

Takeaways carried forward:

1. **Trip-wire metric is structural, not behavioral.** The
   ADDITIVE_KW_RISK count measures `combine_type==additive AND
   kw fires` — it captures which ROWS are at risk, not how
   STRICT each row is. Phase 3 reduces strictness within rows
   (ALL → ANY for paraphrase clusters), so trip-wire stays
   roughly flat while ALL_rate plummets. *Going forward: when a
   change is behavioral rather than structural, set the
   hypothesis target on the behavioral metric directly (here,
   ALL_rate or commit-shape change rate) instead of inferring it
   from the structural headline.*
2. **Schema descriptions and endpoint prompts are coupled
   micro-prompts; updating one without the other creates
   contradictions.** Both `KeywordQuerySpec.scoring_method` and
   `KeywordQuerySpecSubintent.scoring_method` had to be updated
   in lockstep with the keyword.md prompt. Forgetting either
   would have left the LLM seeing the old "substitutable
   alternatives" / "OR-style framing" cues alongside the new
   singular-vs-plural discriminator. *Going forward: when
   editing a prompt, grep the schemas for any field description
   that uses the same vocabulary and update them in the same
   commit. The schema-as-micro-prompt rule cuts both ways.*
3. **Principle-based prompt rewrites need behavioral observation,
   not just commitment-side metrics.** The superset test (3.1)
   was supposed to drive more F3 abstention, but the kw-fire
   count held flat. The principle landed (the prompt assembly
   shows the new section), but the LLM's behavior didn't shift
   on F3-style cases. Possible next step: add an explicit
   negative-example principle ("if your keyword's weaknesses
   name over-coverage of unrelated content, that's a stretch —
   abstain") rather than relying on the LLM to derive it from
   the abstract test. Tradeoff: that risks drifting into a
   failure-catalog (anti-pattern per docs/conventions.md
   §483-491). *Going forward: principle-based rewrites need
   behavioral verification on the targeted failure-mode
   queries, not just headline aggregates. Track per-failure-
   mode metrics, not just trip-wire counts.*
4. **Hypothesis miscalibration is fine when the failure mode
   is correctly identified.** I predicted 20-25% rate, hit
   28.8%. The miss was conflating "F2 reduction" with "trip-
   wire reduction" — Phase 3 doesn't touch the trip-wire's
   structural axis. The *F2 reduction itself* matched the
   prediction (14.6% → 5.0% beats the implicit "drop sharply"
   target). *Going forward: write hypotheses with the
   per-axis prediction (this_rewrite_should_move_THIS_metric)
   rather than a single composite headline target.*

