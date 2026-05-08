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

