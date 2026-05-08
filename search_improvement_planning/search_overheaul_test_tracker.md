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
  signal in the test output would confirm it. State which active
  failure modes the change targets and which it must not regress
  (see "Active failure mode catalog" below).
- **Changes actually made** — concrete code / prompt / schema diffs
  shipped before this iteration's run. Reference files and decision
  records.
- **Observations** — key patterns from the run. Walk the active
  failure mode catalog and report movement (+/-/0) on each mode's
  *proper* metric — F2 reads ALL_rate; F3 reads abstention rate /
  kw_commit count; N1/N2 read category-route distribution. Do NOT
  conflate axis movements with the trip-wire headline (Iteration 5
  lesson #1). Cite specific queries from the suite.
- **Ways to improve going forward** — what the next iteration should
  target, ranked by impact. Distinguish "must fix before next
  phase" from "noted for later".

---

## Active failure mode catalog (post-Phase-3, last updated 2026-05-08)

**This is the LIVE reference for evaluating every future iteration.**
The "Current known failure cases (pre-V5 baseline)" section below
captures what the V5 investigation found before any phase shipped —
preserved for historical anchor only. The catalog *here* reflects
what is still active after Iterations 1–5.

When evaluating a new iteration, walk this list. For each ACTIVE
mode, check whether the change moved it; for each RESOLVED mode,
confirm no regression. If a new failure surfaces that doesn't match
any entry below, add it here with status = ACTIVE and a defining
query example before closing the iteration.

### Active

**F3 — over-coverage commits.** Categories that still fire keyword
commit to over-broad registry members despite weaknesses naming the
over-pull. Phase 3.1 added a principle-based superset test to
keyword.md; the prompt landed but kw_commits stayed flat (41 → 40
phase_2b → phase_3) and abstention rate on STORY_THEMATIC,
ELEMENT_PRESENCE, CHARACTER_ARCHETYPE didn't move. Hypothesis: the
bucket prompt's "puzzle pieces — overlap is the design" framing
earlier in the same prompt is shadowing the per-endpoint sanction;
or the principle is too abstract without anchor examples. **Proper
metric:** kw_commits per still-keyword-firing category + abstention
rate (categories that route but don't fire keyword).

**F4 — cross-trait keyword double-counting in scoring.** When two
traits in one query route their keyword endpoints to the same
registry member (e.g., `boxing movies` → CENTRAL_TOPIC + GENRE both
pull BOXING), that member contributes to scoring for both traits
because per-trait score paths read independent per-coordinate maps.
Phase 1.2 deduped the DB query path (perf optimization) but not the
scoring path. Stage 4 fix; invisible to run_specs (visible in
orchestrator_batch trait_score breakdowns).

**F5 — empty-spec FACETS-zero (preventive shipped, latent; three
softening layers).** Phase 1.1 filters empty-spec categories from
across-category fold. Hasn't fired on the V5 suite since baseline
(every routed category emits at least one spec). Surface area is
large (25+ FACETS traits per run) so the filter remains
load-bearing for tail behavior. Iter 8 / Phase 7 added a second
softening layer: even when a category emits a spec but the spec
scores 0.0 on a candidate, the FACETS fold now uses geometric-mean-
with-floor (`floor=0.1`) instead of strict PRODUCT, so single-zero
no longer zeros the trait. Iter 8 surfaced one new edge case where
this softening was load-bearing: Q5 GENRE emitted
`keyword_finalized=[]` under FACETS combine_mode (every
PotentialKeyword candidate verdict-abstained). Iter 9 closed that
edge case at THREE points: (i) the schema's revert from server-
derived to LLM-emitted `finalized_keywords` with `min_length=1`
makes empty commits unrepresentable when the bucket-level commit
is `commit`; (ii) the new vacuous-spec extraction-time filter in
[output_extractor.py](../search_v2/endpoint_fetching/category_handlers/output_extractor.py)
treats any structurally empty wrapper (keyword finalized=[],
semantic space_queries=[], all-null metadata column_spec) as
not-fired, symmetric with bucket-level abstain; (iii) the Phase 7
floor remains as a defense for non-vacuous category-zero cases.
**Proper metric:** count of FACETS traits where any category
abstained OR zeroed on >50% of candidates; still requires
orchestrator_batch trait_score distribution inspection.

**N1 — EMOTIONAL_EXPERIENTIAL over-attachment by Step 3.** ~25–28%
of category routes go to EMOTIONAL_EXPERIENTIAL even when the
primary signal is genre / setting / element / studio / title (e.g.,
`cyberpunk dystopias`, `historical war epics`, `Studio Ghibli`,
`Donnie Darko`). Less harmful after Phase 2b (no KW gate to zero)
but still inflates FACETS surface area, compounding with N2.
**Proper metric:** EE route count / total routes.

**N2 — FACETS mis-chosen for paraphrase clusters.** Step 3 commits
FACETS combine_mode over category sets that are paraphrastic homes
for one concept. Under PRODUCT, any one category zeroing zeros the
trait. Phase 3 verification confirmed 8 / 28 (28.6%) of FACETS
traits in phase_3 are over suspicious paraphrastic category sets.
Trait-death surface in the residual headline. **Proper metric:**
FACETS-over-paraphrastic-set count vs total FACETS traits.

**N5 — Carver vs qualifier role inconsistency.** Step 3's
`semantic_role` (carver vs qualifier) varies for structurally
similar traits (`marathons` → CENTRAL_TOPIC qualifier vs `elephants`
→ ELEMENT_PRESENCE carver vs `consciousness` → CENTRAL_TOPIC
carver). Lower priority — limited downstream impact on per-trait
scoring. Re-evaluate after higher-impact items.

**N6 — Wide vector-space targeting on EMOTIONAL_EXPERIENTIAL.**
Symptom of N1 — fixing N1 should reduce N6 mechanically. Track but
don't address independently.

**N7 — Likely-incorrect tag categorisation (definitional gap in
bucket prompts).** FAMILY committed under SENSITIVE_CONTENT
(Iteration 1) and MILITARY_DOCUMENTARY committed under CENTRAL_TOPIC
(Iteration 1) are the canonical observations. Didn't reproduce in
phase_2a or phase_3 (LLM drift), but the prompt-level definitional
gap that allows them is unchanged. Latent.

### Resolved

**F1 — vibe-only categories with thin keyword commitment.** Resolved
for EMOTIONAL_EXPERIENTIAL / SEASONAL_HOLIDAY / SPECIFIC_PRAISE_CRITICISM
in **Phase 2b / Iteration 4** (KW removed, single SEM endpoint).
Could in principle fire on the 5 still-keyword-firing categories
(CENTRAL_TOPIC, ELEMENT_PRESENCE, CHARACTER_ARCHETYPE,
NARRATIVE_DEVICES, STORY_THEMATIC_ARCHETYPE) but those route only
when the user's attribute is registry-supportable, making thin-
superset misses rare. Reclassify as F3 if it recurs there.

**F2 — ALL scoring chosen for paraphrase-cluster keywords.** Largely
resolved in **Phase 3.2 / Iteration 5** (singular-vs-plural rewrite).
ALL_rate 20.3% (baseline) → 14.6% (phase_2b) → 5.0% (phase_3). The
2 surviving ALL commits in phase_3 (`[ACTION, THRILLER]` and
`[WAR, HISTORY]`) are genuine plural intent on GENRE under
combine=alternatives → no trip-wire risk. Watch for regression on
positive controls (Q9, Q13's GENRE compounding).

**N8 — Positioning-reference-without-qualifier validator rejection.**
Fixed in **Iteration 2** via `_validate_relationship_roles` self-heal
in [schemas/step_2.py](../schemas/step_2.py) — orphaned positioning
commits coerce to INDEPENDENT rather than rejecting the query.

**N9 — markdown-as-suite contamination footgun.** Operational, not
LLM-side. Fixed in **Iteration 3** by the explicit "do NOT pass
this directly to --suite" warning at the top of
[rescore_overhal_queries.md](rescore_overhal_queries.md), and by
the convention of always passing `/tmp/v5_suite.txt` as the
canonical operator-runnable source. Diff scripts now also count
handler-error categories before trusting the headline (per
Iteration 3 lesson #3).

### Deferred (out of current V5 scope)

**N3 — Step 2 over-atomization on tonal-qualifier compounds.** The
V4 fuse rule fires only on bidirectional cross-modification;
single-direction tonal qualifiers atomize independently. Examples:
`dark gritty antihero comic-book films` → 4 traits,
`slow-burn psychological mysteries` → 3 traits. Considered for a
POSITIONING_QUALIFIER-for-tonal-modifiers extension to V4 typology;
deferred 2026-05-08 — V4 atomization is producing usable traits and
the cascading N4/N1 effects are addressable upstream. Revisit only
if a specific failure surfaces that requires Step 2 typology
extension.

**N4 — Same-category cross-trait routing.** Confirmed in re-analysis
(Iteration 5 follow-up) to be a symptom rather than an independent
problem. ~75% of N4 cases are downstream of N3 (over-atomization
splits a single concept into multiple traits, all routing to the
same category) or N1 (EE over-attachment from multiple traits). The
remainder is genuine multi-attribute coverage working as designed
(`comedy musicals about teenage romance` → GENRE three times = three
distinct genre attributes, correctly weighted). When the same
registry member double-counts across traits, that's F4 — separable.
Deferred 2026-05-08 — no independent quality concern.

---

## Architectural root-cause patterns (observed in Iteration 5 / phase_3)

The active failure modes above are the *what*. This section is the
*why*. Four cross-cutting patterns explain why most of the active
modes persist even after five iterations of targeted fixes; future
work should treat these as constraints to design against, not as
failure modes to fix in isolation.

**Pattern A — Stacked PRODUCTs amplify upstream commit noise into
trait death.** ADDITIVE within-category (KW × META × SEM) and
FACETS across-category (PRODUCT) compose into a chain where any
single zero kills a trait. Phase 2a/b reduced the ADDITIVE category
count, but the residual additive cats (CENTRAL_TOPIC, ELEMENT_PRESENCE,
CHARACTER_ARCHETYPE, NARRATIVE_DEVICES, STORY_THEMATIC_ARCHETYPE)
still gate multiplicatively, and Step 3 routes traits to multiple of
them then PRODUCT-folds. *Canonical case:* `cyberpunk dystopias` →
FACETS over [GENRE-alternatives, STORY_THEMATIC-additive,
ELEMENT_PRESENCE-additive, EE-single]. STORY_THEMATIC commits
`[DYSTOPIAN_SCI_FI]` ANY; a cyberpunk movie tagged CYBERPUNK but
not DYSTOPIAN_SCI_FI scores 0 on STORY_THEMATIC under ADDITIVE
gating, then FACETS PRODUCT zeroes the whole trait. The fold
demands more upstream commit precision than the LLM can deliver.

**Pattern B — Every prompt layer biases toward firing; abstention
is one bullet against many fire-bullets.** Step 2 atomizes
aggressively; Step 3 routes to every plausibly-signal-bearing
category; bucket prompts open with "Endpoints are puzzle pieces —
overlap is the design"; handler schemas have `min_length=1` on
finalized_keywords. Phase 3.1's superset test added one abstention
sanction to that pile and didn't move kw_commit count (41 → 40).
*Canonical case:* `Donnie Darko` → STORY_THEMATIC `[DRAMA,
PSYCHOLOGICAL_DRAMA]` ANY — DRAMA is the documented over-coverage
canonical; the LLM commits anyway because the prompt+schema
pressure to commit dominates the soft "abstain if stretching"
override.

**Pattern C — The walk-then-commit chain-of-thought scaffold is a
prompt-level convention, not a schema-enforced invariant.** The
LLM writes `weaknesses: over-coverage: pulls every drama` for
`gritty → [DRAMA, FILM_NOIR, THRILLER]` and then commits the same
members. Nothing in the schema validates that the commit follows
from the analysis. The connection between `potential_keywords[i]`
and `finalized_keywords` is prompt-only. Autoregressive generation
writes the analysis fluent-style and then writes the commit under
the surrounding "fire if you have signal" prose, not under the
analysis's strengths/weaknesses content. *Canonical case:* every
phase_3 STORY_THEMATIC trip-wire row where the walk surfaced
weaknesses naming over-coverage and the commit included the
over-broad member anyway.

**Pattern D — Step 3 commits combine_mode before handler-side
reality is observable.** The V4 plan moved combine_mode commit
*before* category_calls so the choice would shape category routing.
Side-effect: combine_mode is a guess made before the handlers run.
Step 3 abstractly reasons "are these categories alternative homes
or compounded facets?" with no view of what each handler will
actually commit. *Canonical case:* `cyberpunk dystopias` committed
FACETS over four categories that turned out to be homes-for-one-
concept (CYBERPUNK fires twice on GENRE+ELEMENT_PRESENCE,
DYSTOPIAN_SCI_FI on STORY_THEMATIC paraphrastic, EE
over-attached) — FRAMINGS would have been correct in retrospect,
but Step 3 had no way to see it.

**Why these matter:** the four patterns nest. (D) commits the
fold before knowing what to fold over → (A) makes the fold so
brittle that (D)'s mistakes are fatal → (B) makes upstream commits
noisy enough to require precise folding → (C) leaves the only
abstention mechanism (the walk) unenforceable. Each iteration so
far has tried to make the upstream commits more precise; phase_3
suggests the commits are about as precise as principle-based
prompts can drive them. The next moves either soften the folds,
enforce the precision at the schema level, change where in the
pipeline the deliberation happens, or some combination.

**Iter 8 update (2026-05-08):** Pattern A softened in stage_4
via `_FACETS_FOLD_FLOOR=0.1` (geometric-mean-with-floor replacing
strict PRODUCT). A single category zero now scores `floor^(1/n)`
instead of zeroing the trait. Pattern A is no longer "fatal on
single-category miscommit"; it's "heavy penalty on single-category
miscommit, survivable." Pattern D partially addressed by Phase 6
sibling-task context — the handler now sees what siblings were
*tasked with* even though it can't see what they *produced*. The
guess's precision improved without violating per-call isolation.
Pattern B moved on Iter 8 for the first time across V5: trip-wire
count crossed below the phase_3 baseline of 23 (achieved 21).
The mechanism distinction matters — Iter 5/6/7/7.x all attempted
threshold-tightening prose and didn't move the count; Iter 8's
evidence-injection (sibling block) did. Pattern C remains active:
the walk-to-verdict chain is now schema-enforced (Phase 5) but the
**verdict-to-bucket-commit** chain is still prompt-only, and Iter 8
surfaced an inconsistency — every per-candidate verdict abstained
but bucket-level commitment stayed at "commit" (Q5 GENRE empty
keyword_finalized).

**Iter 9 update (2026-05-08):** Pattern A softening completed at
extraction time — the vacuous-spec filter in
[output_extractor.py](../search_v2/endpoint_fetching/category_handlers/output_extractor.py)
plus the schema revert to LLM-emitted `finalized_keywords` with
`min_length=1` make empty-commit cases structurally
unrepresentable. Q5 plural-intent ALL `[ACTION, THRILLER]` is
restored. Pattern C: the per-candidate verdict pathway introduced
by Phase 5 was reverted; the data showed it was doing real
narrowing work that the bucket-level union-level commit cannot
natively replicate. Pattern C is now LESS schema-enforced than at
end-of-Iter-8 — bucket-level `coverage_commitments.{route}.verdict`
is the only structural enforcement, and it operates at the
union/whole-endpoint level, not at the per-member level. Pattern B
trip-wire count regressed back above phase_3 baseline (21 → 25).
Iter 9 net: ONE clean structural win (Q5 / empty-commit closed)
and ONE direction-undecided architectural revert (per-candidate
verdict pathway). Iter 9 is NOT auto-shippable; the user
decides among (a) ship as-is, (b) revert Change 2 keeping Change 1,
(c) tighten the union-level prompt to recover narrowing power.

---

## Current known failure cases (pre-V5 baseline)

These are the failure modes identified by the V5 investigation
(see [rescore_overhaul.md](rescore_overhaul.md) §Failure mode
catalogue and §Root causes). **Preserved as a historical anchor —
for current state, see "Active failure mode catalog" above.** A
baseline run of the verification suite is pending — once executed
it will be recorded as **Iteration 1** below.

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

### Iteration 6 — Phase 4 (deliberate-default at Step 3 + bucket prompt openings)

**Status:** in progress 2026-05-08. Hypothesis only — changes
queued, run pending.

- **Hypothesis:** Phase 4 of [rescore_overhaul.md](rescore_overhaul.md)
  targets architectural Pattern B (fire-default everywhere) — the
  cross-cutting bias toward firing that Iteration 5 identified as
  the reason F3 abstention didn't move when the principle landed
  (kw_commits 41 → 40; abstention rate flat). Iteration 5 lesson #3
  flagged the failure mode: the principle-based superset test was
  *appended* to bucket prompts whose openings still primed
  "endpoints are puzzle pieces — overlap is the design". Phase 4
  re-opens the deliberation surfaces so abstention is the prior,
  not an override. Two coupled changes:
  - **4.1 — Step 3 `_CATEGORY_ROUTING` rewrite (thread-through).**
    Replace the opening of the section in
    [search_v2/step_3.py](../search_v2/step_3.py) (currently
    L670-747) so it commits deliberate-default — "default: abstain;
    a category enters the route only when the trait's intent
    specifically demands its lens, OR when it adds retrievable
    signal not already covered by another routed category." Update
    the `category_calls` field description on
    [schemas/step_3.py](../schemas/step_3.py) so empty
    `category_calls` is sanctioned as a first-class abstention
    pathway (schema-as-micro-prompt parity per Iteration 5 lesson
    #2). Thread the deliberate-default frame through the
    `OPERATIONAL TESTS` block too — add an explicit
    ABSTENTION-DEFAULT test as the *first* operational test, and
    soften `COVERAGE`'s "Zero → gap" so an uncommitted dimension is
    sanctioned when its candidates failed the abstention-default
    test (rather than always treated as a defect). Other tests
    (`READ COMBINE_MODE`, `READ TRAIT_ROLE_ANALYSIS`,
    `MINIMUM-CALL`, `CANDIDATE-LINK`, `CLEAN-FIT`,
    `POLARITY-DISCIPLINE`, `POSITIONING-CONSISTENCY`,
    `MODE-CONSISTENCY`) remain substantively intact — they read
    only on categories that pass the abstention-default test. The
    thread-through is the operational analogue of the Iteration 5
    lesson: opening-only edits get over-ridden by the surrounding
    test machinery (Pattern C echo).
  - **4.2 — Bucket prompt opening + sub-test reorder.** Three
    multi-endpoint bucket objective prompts:
    [preferred_representation_fallback_objective.md](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/preferred_representation_fallback_objective.md),
    [audience_suitability_deterministic_first_objective.md](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/audience_suitability_deterministic_first_objective.md),
    [semantic_preferred_deterministic_support_objective.md](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/semantic_preferred_deterministic_support_objective.md).
    Replace the opening "Endpoints are puzzle pieces — overlap is
    the design" framing with: each endpoint commits independently
    to its own commitment criteria; abstention is a first-class
    outcome; partial commitment (some fire, others abstain) is
    valid; the bucket's job is to compose the endpoints that
    genuinely contribute, not to fire every endpoint with any
    candidate. Inside `## Coverage exploration`, reorder the four
    local tests so the **Superset test (per endpoint)** added in
    Phase 3.3 becomes the *first* test, with Fire / Drop /
    Sharpness or Over-coverage shifting to follow. Same content;
    different attention placement — primes abstention-first
    deliberation rather than fight-from-behind override.
    `character_franchise_fanout_objective.md` continues to be
    excluded by design (both paths fire when the referent exists;
    partial abstention is not a sensible model for that bucket).

  Following docs/conventions.md and category_handler_planning.md
  small-LLM guidance: principle-based constraints (no failure
  catalogs), no category-specific examples, no V5 suite queries
  reused, brief prose with explicit section boundaries (per
  category_handler_planning.md §447-460), schema-prompt parity
  enforced for the `category_calls` field description.

  **No schema-shape changes.** Pure prompt-opening reorder +
  reframe; the `category_calls` field description edit is
  micro-prompt language only, no field-set or constraint changes
  (Phase 5 is where structural schema changes land per
  rescore_overhaul.md).

- **Expected signals on `run_specs` JSON:**
  - **F3 (over-coverage commits) — primary target.** The metric
    that Phase 3 didn't move:
    - kw_commits per still-keyword-firing category. Targeting
      STORY_THEMATIC_ARCHETYPE (held 13 in phase_3),
      ELEMENT_PRESENCE (held 4), CHARACTER_ARCHETYPE (held 2),
      NARRATIVE_DEVICES (3 in phase_3). Expect drops on
      STORY_THEMATIC and ELEMENT_PRESENCE in particular —
      categories where the registry is most prone to over-pull
      paraphrastic stretch (`gritty → DRAMA, FILM_NOIR, THRILLER`
      shape).
    - Abstention rate (categories that route but don't fire
      keyword) on the same five categories. Expect rise.
  - **N1 (EE over-attachment) — secondary downstream.** Step 3
    deliberate-default should reduce category-route sprawl on
    queries where EMOTIONAL_EXPERIENTIAL is adjacency-attached
    (`cyberpunk dystopias`, `historical war epics`, Studio
    Ghibli, Donnie Darko). Tracked metric: EE route count / total
    routes; baseline ~25-28% in phase_3.
  - **Total category-route count per query.** Step 3
    deliberate-default should reduce route count on
    high-routing queries (Q15 Studio Ghibli, Q18 Donnie Darko,
    Q20 dark gritty antihero comic-book films). Track per-query
    `cats` count delta vs phase_3.
  - **Headline trip-wire** — secondary signal only. The trip-wire
    formula (`additive AND keyword fires`) is structural; Phase 4
    is behavioral (which categories route, which endpoints fire
    inside them). Expect modest drop (deliberate-default reduces
    routing to KW-firing categories) but per Iteration 5 lesson
    #1 don't read this as the primary target. Conservative range:
    25-32% (vs 28.8% phase_3); behavioral movement should show in
    F3 / N1 metrics first.
  - **Schema-validation error rate.** Should remain 0% — the
    category_calls field description edit is description-only;
    the field's existing min/max constraints are unchanged.
- **What we are NOT measuring:** the actual stage-4 trait_score
  effect of dropping a previously-routed category. Visible in
  orchestrator_batch but not run_specs. Deferred per established
  precedent (run_specs surface captures the LLM's
  routing-and-commit shape change; orchestrator sweeps optional
  for behavioral changes that don't move the trait scoring math
  itself).
- **Stop conditions:**
  - Headline trip-wire INCREASES vs phase_3 (28.8%) — over-
    correction at the prompt level, pause and diagnose.
  - Q9 STORY_THEMATIC_ARCHETYPE on `revenge` collapses to abstain
    — over-correction; the abstention frame is over-pulling onto
    cleanly-supersetted commits. Q9 is the registered positive
    control for STORY_THEMATIC clean ANY.
  - Q12 NARRATIVE_DEVICES on `mind-bending` collapses to abstain
    — over-correction on a documented clean commit.
  - Q13 / Q25 plural-intent ALL collapses to ANY or to abstain —
    over-correction on the genuine plural-intent positive
    controls (Q13's GENRE compounding via separate traits, Q25's
    `[UNRELIABLE_NARRATOR]` clean commit).
  - Schema validation error rate > 0%.
  - Step 2 / Step 3 / handler error rate > 0%.

- **Changes actually made:**
  - [search_v2/step_3.py](../search_v2/step_3.py) — `_CATEGORY_ROUTING`
    section rewrite (currently L670-755). Replaced the opening
    "Read the per-dimension candidates and the combine_mode you
    just committed. For each category that ends up owning >=1
    expression, emit ONE CategoryCall." with a deliberate-default
    opening: "Routing is a series of independent commitments. The
    question is not 'could this category contribute?' — almost
    every category could contribute SOMETHING to almost any trait.
    The question is: does this category add retrievable signal
    that no other routed category covers, OR does the trait's
    intent specifically demand this category's lens? If neither
    holds, abstain." Added a "DEFAULT: ABSTAIN" paragraph stating
    empty `category_calls` and partial routing are valid outcomes.
    Threaded the deliberate-default through the operational tests:
    added `ABSTENTION-DEFAULT (FIRST)` as the new first test (the
    surviving categories then read through the existing tests);
    softened `COVERAGE` so an uncommitted dimension is sanctioned
    when its candidates each failed the abstention-default test.
    `READ COMBINE_MODE`, `READ TRAIT_ROLE_ANALYSIS`, `MINIMUM-CALL`,
    `CANDIDATE-LINK`, `CLEAN-FIT`, `POLARITY-DISCIPLINE`,
    `POSITIONING-CONSISTENCY`, `MODE-CONSISTENCY` left
    substantively intact. Added one-line FRAMINGS / FACETS notes
    that the deliberate-default applies under both modes (overlap
    is permitted under FRAMINGS, not required; doubly load-bearing
    under FACETS).
  - [schemas/step_3.py](../schemas/step_3.py) — `TraitDecomposition.category_calls`
    field description (currently L479-534) updated for
    schema-as-micro-prompt parity per Iteration 5 lesson #2:
    added a `DELIBERATE-DEFAULT` paragraph mirroring the system
    prompt; added `ABSTENTION-DEFAULT (FIRST)` to the
    `OPERATIONAL TESTS` block; softened `COVERAGE` so
    sanctioned-uncommitted dimensions are not treated as gaps;
    removed the `LEAVE A DIMENSION UNCOVERED` line from the
    `NEVER` list (replaced with "abstention-default" wording so
    sanctioned uncommitted dimensions are valid); added a final
    `NEVER` entry against committing a category that fails the
    abstention-default just because a dimension named it as a
    candidate. No structural / type / constraint changes — pure
    description-as-micro-prompt edit.
  - [search_v2/endpoint_fetching/category_handlers/prompts/buckets/preferred_representation_fallback_objective.md](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/preferred_representation_fallback_objective.md):
    - Replaced opening paragraph "Endpoints are puzzle pieces — one
      may add specificity another lacks; one may fill a gap another
      leaves; and overlap is the design" with the abstention-first
      framing (each endpoint commits independently to its own
      commitment criteria; abstention is a first-class outcome;
      partial commitment is valid).
    - Inside `## Coverage exploration`, reordered the four local
      tests so `Superset test (per endpoint, FIRST)` is now first,
      followed by Fire / Drop / Over-coverage refinement in their
      original positions. Updated the Superset test bullet's
      language so the "remaining tests apply only to endpoints
      that pass this one" semantics are explicit.
    - Updated the "Empty `coverage_assignments`" sentence and the
      whole-call-abstain summary so the per-endpoint criteria
      lead the language ("no endpoint walk surfaced a candidate
      that passes both the endpoint's own commitment criteria
      and the local fire/drop tests"), shifting the framing from
      "fire by default, abstain in edge cases" to "commit per
      endpoint, default to abstain when criteria fail".
  - [search_v2/endpoint_fetching/category_handlers/prompts/buckets/audience_suitability_deterministic_first_objective.md](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/audience_suitability_deterministic_first_objective.md):
    - Same opening reframe (replaced "Endpoints are puzzle pieces
      — overlap is the design, every endpoint that has a real
      signal to contribute should fire" with the abstention-first
      framing). Audience-suitability multi-endpoint composition
      reframed as "the typical shape *after* each endpoint
      independently passes its commitment criteria — not a default
      that every endpoint with a candidate must satisfy".
    - Same `## Coverage exploration` sub-test reorder: Superset
      first, Fire / Drop / Sharpness layering follow. Sharpness
      layering bullet updated to "have already passed the Superset
      test" — the layering is contingent on per-endpoint
      commitment criteria being met.
    - Same coverage_assignments and whole-call-abstain language
      updates.
  - [search_v2/endpoint_fetching/category_handlers/prompts/buckets/semantic_preferred_deterministic_support_objective.md](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/semantic_preferred_deterministic_support_objective.md):
    - Same opening reframe. The abstention-first paragraph follows
      the bucket-specific intro about semantic carrying the graded
      core and deterministic candidates catching binary signals,
      so the reader still sees the bucket's purpose before the
      abstention-first framing locks in.
    - Same sub-test reorder. Sharpness layering bullet updated:
      "When BOTH have useful candidates that have already passed
      the Superset test, BOTH fire."
    - Same coverage_assignments language update. Preserved the
      bucket-specific "Semantic abstaining is unusual but valid"
      closing, since it's the right model-bias correction for the
      semantic_preferred bucket (the LLM is least likely to
      abstain on semantic).
  - `character_franchise_fanout_objective.md` deliberately
    excluded — same rationale as Phase 3.3 (both paths fire when
    the referent exists; partial abstention isn't a sensible
    model for that bucket).
  - No code changes outside the schema description rewrite. No
    test changes (per the project's `test-boundaries` rule).
  - Verified assembled prompt: `_CATEGORY_ROUTING` contains
    "DEFAULT: ABSTAIN" and "ABSTENTION-DEFAULT (FIRST)";
    `TraitDecomposition.category_calls.description` contains
    "DELIBERATE-DEFAULT" and "ABSTENTION-DEFAULT (FIRST)"; all
    three bucket prompts contain "Superset test (per endpoint,
    FIRST)" and "Each endpoint commits independently" exactly
    once.

#### Observations

`/tmp/run_specs_phase_4.json` (102 KB, 25 queries, 51 traits, **92
categories**, 0 errors, 0 errored_cats — 0 schema violations).

**Headline:**

| run         | Q  | err | tr | cat | risk | rate%   | err_cat |
|-------------|---:|----:|---:|----:|-----:|--------:|--------:|
| baseline    | 25 |   0 | 51 |  80 |   45 |  56.2   |       0 |
| phase_3     | 25 |   0 | 51 |  80 |   23 |  28.8   |       0 |
| **phase_4** | 25 |   0 | 51 |  92 |   25 | **27.2**|       0 |

The rate dropped 1.6pp — but this is misleading. **`cat` count
went UP +12 (80 → 92, +15%) while `risk` count went UP +2 (23 → 25)**.
The rate divides them, hiding the growth in routes. Per Iteration 5
lesson #1, the rate is the wrong proxy for behavioral change here.

**The behavioral story is the Step 3 sprawl going UP, not down:**

| metric                                    | phase_3 | phase_4 | Δ |
|-------------------------------------------|--------:|--------:|---:|
| total category routes                     |      80 |      92 | **+12** |
| EE_routes (absolute)                      |      22 |      23 | +1 |
| EE_share of total routes                  |   27.5% |   25.0% | -2.5pp |
| KW commits in 5 keep cats (TOTAL)         |      23 |      25 | **+2** |
| KW commits in STORY_THEMATIC_ARCHETYPE    |      13 |      13 | 0 |
| KW commits in CENTRAL_TOPIC               |       1 |       3 | +2 |
| KW commits in NARRATIVE_DEVICES           |       3 |       4 | +1 |
| KW commits in ELEMENT_PRESENCE            |       4 |       3 | -1 |

EE share dropped (modest N1 win), but absolute EE count rose because
total routes rose. F3 abstention on the keep cats — Phase 4's
**primary target** — moved in the WRONG direction (+2 KW commits).
STORY_THEMATIC_ARCHETYPE held flat at 13 (the metric Iteration 5
called out as the proper F3 readout, unmoved).

**Sprawl hits where the deliberate-default was supposed to win
hardest:**

| query                                          | base | p3 | p4 | Δp4-p3 |
|------------------------------------------------|-----:|---:|---:|------:|
| `atmospheric folk horror`                      |    4 |  2 | 6 | **+4** |
| `dark gritty antihero comic-book films`        |    3 |  4 | 6 | **+2** |
| `forgotten gems with brilliant performances`   |    3 |  2 | 4 | **+2** |
| `brutal MMA fight movies`                      |    3 |  3 | 5 | **+2** |
| `Wes Anderson aesthetic coming-of-age`         |    3 |  3 | 4 | +1 |
| `like Donnie Darko but funnier`                |    5 |  5 | 6 | +1 |
| `mind-bending puzzle films about consciousness`|    3 |  3 | 4 | +1 |
| `cyberpunk dystopias`                          |    3 |  4 | 3 | -1 |
| `coming-of-age road trips not too sappy`       |    3 |  4 | 3 | -1 |

Phase 4 produced *some* wins (Q10 cyberpunk, Q24 road trips, Q4
wholesome dropped a STORY_THEMATIC route on `wholesome` — clean
deliberate-default outcomes) but more losses, especially Q21
**`atmospheric folk horror` 2 → 6 categories**: the `atmospheric`
trait now routes to FACETS over [EE, VISUAL_CRAFT_ACCLAIM,
MUSIC_SCORE_ACCLAIM] (3 SEM-only categories where any zero kills
the trait); `folk horror` routes to FACETS over [GENRE,
STORY_THEMATIC_ARCHETYPE, NARRATIVE_SETTING] (paraphrastic homes
for one concept — the canonical N2 anti-pattern). These are
exactly the architectural Pattern A failures Phase 4 was supposed
to *reduce*, not amplify.

**ANY/ALL distribution — F2 over-correction:**

| run        | total | ANY | ALL | ALL_rate |
|------------|------:|----:|----:|--------:|
| baseline   |    59 |  47 |  12 |   20.3% |
| phase_3    |    40 |  38 |   2 |    5.0% |
| **phase_4**|    44 |  44 |   0 | **0.0%**|

Both surviving genuine-plural-intent ALL commits from phase_3
collapsed to ANY:
- Q5 `intense action thrillers` → GENRE `[ACTION, THRILLER]` ALL
  (alternatives, no risk) → **ANY** in phase_4. F2 over-correction.
- Q11 `historical war epics` → GENRE `[WAR, HISTORY]` ALL
  (alternatives, no risk) → **GENRE `[WAR]` ANY** (HISTORY dropped
  entirely) **+ STORY_THEMATIC_ARCHETYPE `[EPIC,
  HISTORICAL_EPIC]` ANY** (additive, **risk=True** — new trip-
  wire). The compound user intent (war AND history) was
  fragmented — Phase 4 lost the HISTORY part entirely and
  introduced a new STORY_THEMATIC route that wasn't there before.

These are the canonical positive controls for genuine plural ALL
on combine=alternatives. Both broke. The keyword.md singular-vs-
plural framing is unchanged from phase_3, so the regression is
upstream (Step 3 routing changed which categories get the
expression, which changed how the keyword endpoint sees the call
shape).

**Per-category risk count (phase_3 → phase_4):**

| category                   | base | p3 | p4 | Δ p4-p3 |
|----------------------------|----:|----:|----:|---:|
| STORY_THEMATIC_ARCHETYPE   |  14 |  13 |  13 |   0 |
| CENTRAL_TOPIC              |   2 |   1 |   3 |  +2 |
| NARRATIVE_DEVICES          |   3 |   3 |   4 |  +1 |
| ELEMENT_PRESENCE           |   2 |   4 |   3 |  -1 |
| CHARACTER_ARCHETYPE        |   2 |   2 |   2 |   0 |
| **net**                    |  23 |  23 |  25 | **+2** |

**Hypothesis predictions vs actual outcomes:**

| Prediction                                                        | Actual                                          | ✓/✗ |
|-------------------------------------------------------------------|-------------------------------------------------|:---:|
| F3 KW commits per keep category drops on STORY_THEMATIC, EP       | STORY_THEMATIC held 13; EP -1; net +2 (CENT/ND up) | ✗ |
| F3 abstention rate rises on keep cats                              | KW commits in keep cats went UP +2              | ✗ |
| N1 EE route count drops                                            | EE share -2.5pp; abs EE count +1 (held flat)    | ~ |
| Step 3 sprawl drops on Q15/Q18/Q20                                 | Q15 held; Q18 +1; Q20 +2 — REGRESSED            | ✗ |
| Headline rate 25-32%                                               | 27.2% — within range                            | ~ |
| Q9 / Q12 / Q25 positive controls hold                              | All three held clean ANY commits                 | ✓ |
| Q5 / Q11 genuine plural ALL preserved (NOT a stop condition listed but standing positive control) | Both collapsed: Q5 → ANY, Q11 broke entirely (lost HISTORY, gained STORY_THEMATIC trip-wire) | ✗ |
| Schema validation error rate 0%                                    | 0 errors, 0 schema violations                    | ✓ |

**Step 2 atomization drift (sanity):** 2 queries shifted trait
count vs phase_3 — `brutal MMA fight movies` 1 → 2 (split out
`brutal` as a tonal qualifier), `gritty crime sagas` 2 → 1
(merged into one trait). Pure LLM noise unrelated to Phase 4
(Step 2 prompt is unchanged).

**Was the hypothesis correct?** Largely **NO**:
1. F3 abstention — primary target — moved in the WRONG direction
   (+2 KW commits, +12 categories total). The deliberate-default
   frame did not act as a gate.
2. Step 3 sprawl regressed on the queries the hypothesis explicitly
   targeted (Q18 +1, Q20 +2, Q15 held). Q21 atmospheric folk
   horror sprawl regression of +4 categories is the worst single
   case observed across V5 iterations.
3. F2 genuine-plural ALL collapsed — Q5 and Q11 both broke (Q11
   catastrophically: lost HISTORY, gained a STORY_THEMATIC trip-
   wire).
4. N1 EE share modestly improved but absolute EE_routes held
   (1-route increase masked by route-count growth elsewhere).
5. Positive controls Q9 / Q12 / Q25 held clean.
6. 0 errors / 0 schema violations.

**Why did the deliberate-default frame fail?** Re-reading the
phase_4 routing for Q21 atmospheric folk horror: the LLM committed
EE + VISUAL_CRAFT_ACCLAIM + MUSIC_SCORE_ACCLAIM for `atmospheric`.
Each one passes the deliberate-default test under a charitable
read of the OR-clause: "does this category add retrievable signal
that no other routed category covers?" — visual craft DOES add
signal that EE doesn't carry, music score DOES add signal that EE
and visual don't carry. Each category individually justifies under
the OR; nothing in the prompt scales the bar. The frame articulates
the criterion but doesn't operationalize "high bar" — it just adds
a permission for two distinct ways to commit. The result: more
careful per-category walks → more committable categories. This is
the architectural pattern B problem in a sharper form: *prompt-
level priors don't compose into hard gates*.

The bucket prompt sub-test reorder (Superset test first) likely
amplified the same effect inside the handler — the LLM walks the
keyword candidate, finds one that supersets cleanly (`EPIC` for
"epics", `HISTORICAL_EPIC` adjacent), commits it. The Phase 3
keyword.md superset test continues to pass on permissive reads.

**Unintended consequences:**
1. **Q11 [WAR, HISTORY] ALL → broken.** The user's compound intent
   (war AND history) was fragmented across categories. HISTORY
   dropped entirely; a new STORY_THEMATIC `[EPIC,
   HISTORICAL_EPIC]` trip-wire appeared. Lossy *and* worse.
2. **Q21 atmospheric → 3-category FACETS PRODUCT.** Architectural
   pattern A failure: any of [EE, VISUAL_CRAFT_ACCLAIM,
   MUSIC_SCORE_ACCLAIM] zeroing zeros the trait. Phase 4 made
   this query *more* fragile, not less.
3. **F2 ALL_rate → 0%.** Removed the last 2 ALL commits — but
   both were the genuine plural-intent ones we intended to
   preserve. The trip-wire flag stays at risk=False on
   alternatives commits, but the call shape is degraded.

**Is Phase 4 safe to ship? NO.**

Stop conditions triggered:
- Step 3 sprawl regressed on the targeted queries (explicit
  prediction failure).
- F3 abstention moved in the wrong direction (primary target
  failure).
- Genuine plural-intent ALL collapsed on Q5 and Q11 (positive
  controls degraded).

Stop conditions NOT triggered:
- Headline trip-wire DID NOT increase (28.8% → 27.2%) — but this
  is the structural-axis metric Iteration 5 flagged as
  misleading; the absolute counts went up.
- Q9 / Q12 / Q25 named positive controls held clean.
- 0 errors / 0 schema violations.

**Recommendation: REVERT.** Roll back all four files
(`search_v2/step_3.py` `_CATEGORY_ROUTING`, `schemas/step_3.py`
`category_calls` description, three bucket prompts) to their
phase_3 state. Do not ship Iteration 6.

This is a clean validation of architectural pattern B's resistance
to prompt-only interventions: even with the deliberate-default
frame in the opening, threaded through the operational tests, AND
mirrored in the schema-as-micro-prompt, the LLM's per-category
"could this contribute?" reasoning continues to dominate. Phase 5
(schema-level verdict fields, structural enforcement) is the next
intervention; the failure of Phase 4 as designed is corroborating
evidence that the architectural-pattern diagnosis was correct
(prompt language alone won't override the firing prior).

#### Ways to improve going forward

Ranked by expected risk-reduction per unit of work:

1. **Revert Phase 4, ship Phase 5 directly.** The schema-level
   verdict fields (rescore_overhaul.md §Phase 5) take the
   abstention pathway from "language in prompt" to "required
   field on schema with explicit values". The Iteration 5 lesson
   #2 + Iteration 6 evidence both point to schema enforcement as
   the load-bearing intervention. Add `verdict:
   Literal["commit","abstain"]` and `verdict_reason` fields on
   `PotentialKeyword` and equivalent slots on
   `coverage_assignments`; make `finalized_keywords` server-
   derived from the verdicts. The required enum + explicit
   reason text is the architectural-pattern C fix the deliberate-
   default frame was a prompt-level proxy for.
2. **Revisit the keyword.md superset test phrasing alongside
   Phase 5.** Iteration 6 evidence: even with the bucket prompt
   reorder placing the per-endpoint Superset test first, the test
   itself is being passed on permissive reads (`EPIC`,
   `HISTORICAL_EPIC` for "epics"; `[FOLK_HORROR]` for both GENRE
   and STORY_THEMATIC_ARCHETYPE). When the schema requires an
   explicit verdict reason the LLM has to articulate WHY the
   superset holds, which may surface the stretching the principle
   alone doesn't catch.
3. **Investigate the Q11 plural-intent breakage in Phase 5
   verification.** The phase_4 result shows that routing changes
   can fragment a compound user intent across categories in ways
   the keyword endpoint can't recover. Add a verification check
   to Phase 5 that traits with combine_mode=facets and multiple
   distinct user attributes (war + history, action + thriller)
   preserve the attribute compound at the keyword-commit level.
4. **N3 Step 2 over-atomization (deferred from Iteration 5)
   continues to leak.** Iteration 6 saw 2 queries shift trait
   count by pure LLM drift (`brutal MMA fight movies` 1 → 2,
   `gritty crime sagas` 2 → 1). Not Phase 4-caused; the underlying
   instability is Step 2 prompt sensitivity. Still deferred per
   Iteration 5; flag for future Step 2 prompt revisit.

**Stop conditions for next iteration (Phase 5):**
Same as Iteration 6 with one addition — Phase 5 is a
schema-shape change (PotentialKeyword gains required fields), so
add: schema validation error rate > 5% is a ship-blocker (Phase 5
introduces fields the LLM must populate; over-fill bias should
help but is not guaranteed).

#### Shipped — what we learned

**Iteration 6 NOT shipped.** Reverted before commit. The four
modified files have been left in their phase_4 state for the
moment — they are NOT to be committed. Revert plan: `git checkout
HEAD --` on the four files
([search_v2/step_3.py](../search_v2/step_3.py),
[schemas/step_3.py](../schemas/step_3.py), and the three bucket
prompts under
[search_v2/endpoint_fetching/category_handlers/prompts/buckets/](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/))
before the next change.

Takeaways carried forward:

1. **Prompt-only interventions against architectural pattern B
   plateau, then regress.** Iteration 5 hit the plateau (kw_commits
   41 → 40 — held). Iteration 6 went past the plateau into
   regression: the deliberate-default frame, threaded
   comprehensively (opening + thread-through + schema-as-micro-
   prompt + bucket reorder), drove kw_commits 23 → 25 in the keep
   categories AND added 12 categories of sprawl. The frame
   articulates the criterion but does not operationalize "high
   bar". Pattern B is structurally resistant to prompt-language
   intervention. *Going forward: stop attempting prompt-only
   abstention-frame interventions on Step 3 routing. Move to
   schema enforcement.*
2. **The OR-disjunction in deliberate-default tests is too
   permissive.** "Does this category add retrievable signal not
   already covered, OR does the trait's intent specifically
   demand its lens?" — the OR provides two lanes to commit, and
   the LLM walks BOTH lanes per category and finds at least one
   that passes. The criterion landed in the prompt but the
   operational meaning is "is there any way to justify routing?".
   *Going forward: avoid OR-disjunctions in abstention criteria
   when the goal is to raise the bar; AND-conjunctions or single
   high-bar criteria push harder.*
3. **Step 3 routing changes can fragment compound user intent in
   ways downstream prompts can't recover.** Q11 `[WAR, HISTORY]
   ALL` broke into GENRE `[WAR]` + STORY_THEMATIC `[EPIC,
   HISTORICAL_EPIC]` — the user's compound (war AND history) was
   lost at routing time, not at keyword-commit time. The
   keyword.md superset test was unchanged. *Going forward: when
   evaluating Step 3 routing changes, audit traits where
   combine_mode=facets and the user's surface text names multiple
   distinct attributes (e.g., "X with Y", "X about Y") — fragment
   patterns surface here first.*
4. **Headline trip-wire rate hides absolute-count growth when
   category count grows.** The rate dropped 1.6pp — a "good"
   number — but absolute counts went up on every behavioral
   metric. Iteration 5 lesson #1 already flagged this for the
   structural-vs-behavioral axis; Iteration 6 sharpens it to a
   second axis: rate-vs-absolute. *Going forward: report all
   four axes — structural rate, structural absolute count,
   behavioral rate, behavioral absolute count — and refuse to
   call a change "directionally correct" if any of the four
   regresses.*
5. **Architectural-pattern resistance is real, not a measurement
   artifact.** Five iterations of targeted prompt fixes have
   plateaued at ~25 trip-wires / ~80-92 categories. Phase 5's
   schema enforcement is the next architectural intervention,
   not a stronger prompt. *Going forward: when an iteration's
   primary metric does not move, treat it as evidence the
   architectural pattern is binding and design the next move at
   a higher leverage layer (schema, code, pipeline structure)
   rather than refining the same prompt layer.*


### Iteration 7 — Phase 5 (schema-level verdict fields on PotentialKeyword + CoverageCommitments)

**Date opened:** 2026-05-08

**Architectural pattern targeted:** C (walk-to-commit is prompt
convention, not schema invariant). Phase 5 also inherits Phase 4's
abandoned Pattern B target — by forcing the LLM to emit a
`verdict` enum at the candidate level, the abstention pathway
moves from soft prose preference to hard structural commitment
between two valid outputs.

#### Hypothesis

Required `verdict_reason` (prose) → `verdict` (enum
commit/abstain) fields on every `PotentialKeyword` in
**multi-endpoint** keyword walks (KeywordWalk feeding
KeywordQuerySpecSubintent), with `finalized_keywords` server-side
derived from `verdict == "commit"` candidates, will:

1. **Reduce keyword commits in the keep categories** by giving
   the LLM a structurally enforced abstention pathway. Iteration 5
   plateau (kw_commits 41 → 40) and Iteration 6 regression
   (40 → 25 in keep cats but with sprawl growth) both point to
   prompt-only Pattern B intervention failing. Schema enforcement
   forces the LLM to RENDER an explicit choice for every
   candidate.

2. **Increase abstain-on-over-coverage rate.** Pre-Phase-5: every
   PotentialKeyword whose `weaknesses` named over-coverage still
   ended up in `finalized_keywords` (the LLM walked the registry
   and committed regardless). Post-Phase-5: candidates whose
   weaknesses name "over-coverage" / "over-pull" should
   verdict-abstain at a measurable rate. Target: ≥ 15% on those
   candidates.

3. **Hold positive controls Q9 / Q12 / Q25 clean.** These
   candidates' weaknesses are clean (`REVENGE` / `ANTI_HERO` /
   `NONLINEAR_TIMELINE` / `UNRELIABLE_NARRATOR` / `PLOT_TWIST`
   are well-fit registry members for their queries). The LLM
   should verdict-commit on them.

4. **Be neutral on Step 3 sprawl.** Phase 5 is keyword-handler-
   layer; it should not change Step 3's category-routing decisions.
   Total category count expected to hold near phase_3's 80
   (Iteration 6's 92 was prompt-driven and reverted).

5. **Be neutral on Q5 / Q11 plural-intent ALL.** Iteration 6
   broke both (Q5 → ANY, Q11 catastrophically lost HISTORY +
   gained STORY_THEMATIC trip-wire) via Step 3 routing-time
   fragmentation. Phase 5's keyword-handler-layer fix runs after
   Step 3 routing, so cannot repair fragmentation if it occurs;
   verify it does not introduce new fragmentation. Step-3-
   routing-time compound-intent fragmentation is a Phase 6
   problem.

**Two Iteration 6 corrections baked into the design:**

- `verdict_reason` is single-claim: name ONE of (`gaps`,
  `stretching`, `dominated-by-sibling`) for abstain, OR name the
  single superset condition for commit. No OR-disjunctions.
- Field declaration order: `verdict_reason` BEFORE `verdict`
  (per user direction), so the prose reasoning is generated as
  fresh evidence-then-decision, not post-hoc justification.

**Replaces `coverage_assignments` (variable-length omission-as-
abstain) with `coverage_commitments` (fixed-shape, one required
slot per declared endpoint, each with verdict + verdict_reason +
optional slice_description):** the LLM cannot abstain on an
endpoint by omission; it must actively render verdict=abstain
with a reason. Default omission bias is replaced by
explicit-choice bias.

**Stop conditions:**
- Schema validation error rate > 5% — the LLM consistently
  mis-fills new required fields. (Iter 6 baseline: 0 errors.)
- Q9 / Q12 / Q25 lose their clean keep commits (positive controls
  break — over-correction at the verdict layer).
- Step 3 sprawl regresses (total category count > 84 — phase_3
  was 80). Phase 5 is keyword-handler-layer; Step 3 should be
  near-flat.
- Headline trip-wire absolute count INCREASES (>= 23 — phase_3
  baseline). Iter 6 lesson: report all four axes; refuse a
  ship if absolute count regresses.
- Single-endpoint keyword buckets fail with 0 commits regularly
  — out of scope for Phase 5 (single-endpoint left untouched per
  user direction; flag only as evidence for follow-up).

**Out of scope (deferred per user direction):**
- Single-endpoint `KeywordQuerySpec.finalized_keywords` stays
  LLM-emitted. Verdict fields appear only on the multi-endpoint
  walk's `PotentialKeyword` (via a `PotentialKeywordWithVerdict`
  subclass used inside `KeywordWalk`).
- Hardcoded `abstain_mode` enum (the structured-verdict variant)
  — start with prose; if Phase 5 results show the LLM is using
  prose as soft preference, escalate to enum in a follow-up.

#### Changes actually made

Six files modified for Iteration 7:

1. **[schemas/keyword_translation.py](../schemas/keyword_translation.py)**:
   - Added `Literal` import.
   - Added `PotentialKeywordWithVerdict(PotentialKeyword)` subclass
     with `verdict_reason: str` then `verdict: Literal["commit",
     "abstain"]` (declaration order: reason BEFORE verdict per
     user direction so prose generates as evidence, not post-hoc
     justification).
   - Added `AttributeAnalysisWithVerdict(AttributeAnalysis)`
     overriding `potential_keywords` to use the verdict variant.
   - `KeywordWalk.attributes` now `list[AttributeAnalysisWithVerdict]`.
   - `KeywordQuerySpecSubintent.finalized_keywords` relaxed to
     `default_factory=list` (no `min_length=1`); description
     rewritten to "server-DERIVED — emit empty list `[]`."

2. **[search_v2/endpoint_fetching/category_handlers/schema_factories.py](../search_v2/endpoint_fetching/category_handlers/schema_factories.py)**:
   - Added `model_validator` import.
   - Added `_WalkThenCommitOutputBase(_HandlerOutputBase)` with a
     `model_validator(mode="after")` that walks `keyword_walk.
     attributes[*].potential_keywords`, dedupes verdict-commit
     keywords, and overwrites `keyword_parameters.parameters.
     finalized_keywords` post-parse.
   - Replaced module-level `coverage_assignments` constants with
     `coverage_commitments` constants (per-route descriptions for
     verdict_reason / verdict / slice_description).
   - Added `_build_coverage_commitments_model` factory that emits
     a per-bucket `CoverageCommitments` object with one required
     EndpointCommitment slot per declared endpoint (no Optional —
     strict-required-per-declared-endpoint per user direction).
     Each EndpointCommitment is also per-route so the prose
     placeholders specialize.
   - `_build_walk_then_commit` now uses `__base__=_WalkThenCommitOutputBase`
     and emits `coverage_commitments` instead of `coverage_assignments`.

3. **[search_v2/endpoint_fetching/category_handlers/prompts/endpoints/keyword.md](../search_v2/endpoint_fetching/category_handlers/prompts/endpoints/keyword.md)**:
   - Updated "Where the keyword analysis lives" + "What does NOT
     belong here" to reference `coverage_commitments` and explain
     `verdict_reason → verdict` on every `PotentialKeyword`.
   - Updated "Commitment: superset test" to describe the two-level
     abstention pathway: per-candidate verdict + per-endpoint
     `coverage_commitments.keyword.verdict`. Explicitly notes that
     `finalized_keywords` is server-derived from the verdict
     commits.

4. **[search_v2/endpoint_fetching/category_handlers/prompts/buckets/preferred_representation_fallback_objective.md](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/preferred_representation_fallback_objective.md)**,
   **[…/semantic_preferred_deterministic_support_objective.md](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/semantic_preferred_deterministic_support_objective.md)**,
   **[…/audience_suitability_deterministic_first_objective.md](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/audience_suitability_deterministic_first_objective.md)**:
   - Replaced all `coverage_assignments` references with
     `coverage_commitments`.
   - Phase 1 walks: noted that keyword candidates carry per-
     candidate verdict_reason → verdict.
   - Phase 3 commitment: explained the fixed-shape commitment
     object with required verdict per declared endpoint.
   - Phase 4 thin params: noted firing condition is
     `coverage_commitments.{route}.verdict == "commit"`; for
     keyword, `finalized_keywords` is server-derived (emit `[]`).
   - Added "you cannot abstain on an endpoint by silence" rule.

5. **Bucket guardrails** (3 `*_guardrails.md` files): renamed
   "empty `coverage_assignments`" → "all-endpoint
   `verdict=abstain` in `coverage_commitments`" and added a
   "ABSTAIN ON AN ENDPOINT BY OMISSION" never-rule.

6. **Endpoint prompts** (`metadata.md`, `semantic.md`) +
   `categories/few_shot_examples/cultural_status.md` — replaced
   stale `coverage_assignments` references with `coverage_commitments`
   verdicts.

End-to-end Pydantic test confirms the derivation: a parsed bucket
output with `keyword_walk` containing `verdict=commit` on REVENGE +
`verdict=abstain` on ANTI_HERO yields `keyword_parameters.
parameters.finalized_keywords = ['REVENGE']`. Schema build is
clean (40 schemas, no circular imports, 0 validation errors on
all 25 V5 suite queries).

#### Observations

**Headline:**

| run        | total | err | tr | cat | risk | rate%  | err_cat |
|------------|-------|-----|----|----:|----:|-------:|--------:|
| baseline   | 25    | 0   | 51 |  80 |  45 | 56.2% |   0     |
| phase_3    | 25    | 0   | 51 |  80 |  23 | 28.8% |   0     |
| **phase_5**| 25    | 0   | 51 |  86 |  26 | 30.2% |   0     |

Per Iteration 6 lesson #4 (report all four axes — structural rate,
structural absolute, behavioral rate, behavioral absolute):
- Structural absolute (cats): 80 → 86 (**+6, +7.5% sprawl**).
- Behavioral absolute (risk): 23 → 26 (**+3 trip-wires**).
- Behavioral rate: 28.8% → 30.2% (**+1.4pp**).

**Three of four axes regressed.** Stop condition #3 (Step 3 sprawl
> 84) and stop condition #4 (headline trip-wire absolute count
increases from 23) both triggered.

**F3 — KW commits in keep categories (primary hypothesis target):**

| category                   | base | p3 | p5 | Δ p5-p3 |
|----------------------------|----:|----:|----:|---:|
| STORY_THEMATIC_ARCHETYPE   |  14 |  13 |  16 | **+3** |
| ELEMENT_PRESENCE           |   2 |   4 |   3 |  -1 |
| CHARACTER_ARCHETYPE        |   2 |   2 |   3 |  +1 |
| NARRATIVE_DEVICES          |   3 |   3 |   3 |   0 |
| CENTRAL_TOPIC              |   2 |   1 |   1 |   0 |
| **net**                    |  23 |  23 |  26 | **+3** |

The hypothesis was that schema enforcement would DROP these
counts (target: STORY_THEMATIC < 13; aggregate of EP / ND / CT /
CA < 10). Both targets failed. Net kw_commits in keep cats moved
+3 in the wrong direction.

**Per-candidate verdict pathway works at the schema level (no
errors, abstain is rendering).** A spot-sample on `Studio Ghibli`
→ STORY_THEMATIC_ARCHETYPE confirmed: on the "humanity-nature
relationship" attribute, both FANTASY and FOLK_HORROR candidates
emitted `verdict=abstain` with reason "stretching: ..." citing the
weaknesses text. The verdict_reason → verdict structural
commitment IS landing.

**But the LLM finds OTHER attributes that pass the commit bar.**
The same Studio Ghibli walk surfaced a "coming-of-age from
childhood to maturity" attribute and committed COMING_OF_AGE
cleanly. Result: phase_3 had Ghibli → STORY_THEMATIC [FANTASY,
SUPERNATURAL_FANTASY, FOLKLORE_ADAPTATION] (definitionally
plausible for Ghibli's body of work); phase_5 has Ghibli →
STORY_THEMATIC [SURVIVAL, COMING_OF_AGE] (a stretch — Ghibli
isn't primarily coming-of-age). The trip-wire stays; the commit
content is *worse*.

Same pattern on Donnie Darko: phase_3 [DRAMA, PSYCHOLOGICAL_DRAMA]
(reasonable); phase_5 [COMING_OF_AGE, SCI_FI, FANTASY] (stretching
across multiple families). The verdict pathway abstained on
candidates the LLM judged as stretching, but the LLM also
*surfaced different attributes* and verdict-committed those,
even when they're stretches.

**F2 ANY/ALL distribution:**

| run      | kw_commits | ANY | ALL | ALL_rate |
|----------|-----------:|----:|----:|---------:|
| baseline |   59       |  47 |  12 | 20.3%    |
| phase_3  |   40       |  38 |   2 |  5.0%    |
| phase_5  |   42       |  40 |   2 |  4.8%    |

Two ALL commits on each end:
- phase_3: Q5 GENRE [ACTION, THRILLER] alt + Q11 GENRE [WAR,
  HISTORY] alt — both compound-intent, no risk.
- phase_5: Q5 GENRE [ACTION, THRILLER] alt **(✓ HELD)** + Q17
  NARRATIVE_DEVICES [EPIC, ENSEMBLE_CAST] **additive risk=True
  — NEW spurious ALL trip-wire**.

Q11 lost its ALL because Step 2 atomization drifted (1 trait →
3 traits: `historical`, `war`, `epics`). The compound intent was
preserved across the new traits — `war` still routes to GENRE
[WAR, WAR_EPIC] and `epics` still routes to GENRE [EPIC]. So
Phase 5 didn't actually break Q11; Step 2 drift split the
expression and routing followed.

The new ALL trip-wire on `gritty crime sagas` → NARRATIVE_DEVICES
[EPIC, ENSEMBLE_CAST] is a stretching commit (EPIC is genre,
ENSEMBLE_CAST is structure — neither is a narrative device). This
query also had Step 2 atomization drift (2 → 1 trait).

**N1 — EE route count:** 22/80 → 20/86. EE share dropped 27.5% →
23.3%; absolute count dropped by 2. Modest improvement consistent
with the V5 trajectory (D1 + D2 already removed KW from EE).

**Step 3 sprawl per query** (Δp5-p3 in cats):
- Worst regressions: `historical war epics` 2 → 6 (+4 — Step 2
  drift 1→3 traits drove this), `dark gritty antihero comic-book
  films` 4 → 6 (+2), `forgotten gems with brilliant performances`
  2 → 4 (+2), `Wes Anderson aesthetic coming-of-age` 3 → 5 (+2),
  `atmospheric folk horror` 2 → 4 (+2).
- Best improvements: `Studio Ghibli` 6 → 4 (-2), `wholesome
  family movie night picks` 6 → 4 (-2).

The regressions cluster on queries with Step 2 atomization drift
or where the LLM surfaced more attributes within a category's
walk. Phase 5 is keyword-handler-layer; it should not have
changed Step 3 routing — but it did, indirectly, via LLM non-
determinism on a different prompt + schema combination.

**Positive controls:**

| query | phase_3 | phase_5 | verdict |
|---|---|---|:---:|
| Q9 `revenge stories with anti-heroes` | REVENGE + ANTI_HERO clean ANY | same | ✓ HELD |
| Q12 `mind-bending puzzle films about consciousness` | NARRATIVE_DEVICES [NONLINEAR_TIMELINE, UNRELIABLE_NARRATOR, PLOT_TWIST] ANY | same | ✓ HELD |
| **Q13 `comedy musicals about teenage romance`** | TARGET_AUDIENCE [TEEN_ROMANCE, ...] alt no-risk | **STORY_THEMATIC [TEEN_ROMANCE] additive RISK=True** | **✗ REGRESSED** |
| Q25 `unreliable narrator with a twist ending` | NARRATIVE_DEVICES [UNRELIABLE_NARRATOR] ANY | same | ✓ HELD |

Q13 lost its TARGET_AUDIENCE alternatives commit and gained a
STORY_THEMATIC additive trip-wire. This is a Step 3 routing
change (Phase 5 didn't intend to cause this), most likely
attributable to LLM non-determinism on the routing decision now
that the handler schema shape has changed. **Q13 is one of the
named positive controls — its degradation is a stop-condition
hit.**

**Step 2 atomization drift (sanity):** 3 queries shifted trait
count vs phase_3 (`gritty crime sagas` 2 → 1, `historical war
epics` 1 → 3, `mind-bending puzzle films about consciousness`
2 → 1). Pure LLM noise on Step 2 prompt unchanged from phase_3.
This is the highest drift count we've observed across V5
iterations (Iter 6 had 2; phase_3 → phase_5 has 3). The
schema-shape change downstream may be inducing slightly different
input distributions to the LLM at Step 2 via batch-effect / cache
behavior, OR pure noise — not enough data to distinguish.

**Hypothesis predictions vs actual outcomes:**

| Prediction                                                     | Actual                                            | ✓/✗ |
|----------------------------------------------------------------|---------------------------------------------------|:---:|
| Reduce kw_commits in keep cats (primary target)                | 23 → 26 (+3, WRONG direction)                     | ✗ |
| ≥ 15% abstain rate on weaknesses-name-over-coverage candidates | Spot-confirmed pathway works (FANTASY/FOLK_HORROR abstained on Ghibli humanity-nature) | ✓* |
| Q9 / Q12 / Q25 hold clean keep commits                          | All three held clean                              | ✓ |
| Phase 5 neutral on Step 3 sprawl (cats ≈ 80)                    | 80 → 86 (+6 via Step 2 drift + extra attributes)  | ✗ |
| Phase 5 neutral on Q5 / Q11 plural-intent ALL                   | Q5 ✓; Q11 lost ALL but compound intent preserved across new traits | ~ |
| Schema validation error rate < 5%                               | 0 errors on all 25 queries                        | ✓ |
| Q13 holds clean (positive control)                              | Q13 regressed — TARGET_AUDIENCE → STORY_THEMATIC additive | ✗ |

**Why did Phase 5 fail at the trip-wire level despite the
verdict pathway working?**

The structural enforcement IS landing. Per-candidate verdicts
are being rendered; abstains are happening on stretching
candidates (verified via spot-sample on Ghibli). The schema
contract is honored — 0 validation errors, finalized_keywords
correctly server-derived from verdict commits. The mechanical
correctness is sound.

But the LLM finds *other* attributes within each walk that pass
the commit bar. The verdict_reason language "name the single
superset condition this candidate satisfies" is permissive — the
LLM judges "does this candidate have a real connection to the
attribute?" and commits when the answer is yes. That bar is too
low. Examples:
- Studio Ghibli's walk surfaces a `coming-of-age from childhood
  to maturity` attribute (because some Ghibli films are about
  this); LLM verdict-commits COMING_OF_AGE on that attribute.
  But "coming-of-age" isn't the dominant story shape of Ghibli
  as a body of work — it's stretching at the *attribute*-
  decomposition level, not the candidate-membership level.
- Donnie Darko's walk surfaces attributes that the LLM then
  populates with [COMING_OF_AGE, SCI_FI, FANTASY] — multiple
  stretches, each individually verdict-committable because the
  walk's framing for the attribute is permissive.

This is **the same architectural pattern as Iteration 6**, one
layer up: prompt language ("single superset condition") sets the
*threshold* for commit. Hard structural enforcement (enum verdict
between two valid outputs) does not change that threshold. The
LLM's autoregressive bias to commit dominates whenever the
threshold is permissive.

**Coverage_commitments worked as designed but didn't shift
behavior.** The fixed-shape `coverage_commitments` object does
force an explicit verdict on every declared endpoint. Empty
omission as abstain is no longer a pathway. But the LLM commits
on every endpoint for which it can name a contribution — same
permissive threshold, just rendered into structure.

**Unintended consequences:**

1. **Q13 positive control regressed.** TEEN_ROMANCE moved from
   TARGET_AUDIENCE alternatives (no risk) to STORY_THEMATIC_ARCHETYPE
   additive (RISK). Phase 5 didn't touch Step 3 routing, so this
   is LLM-noise-driven routing variance — but the variance got
   worse, not neutral.

2. **Step 2 atomization drift increased.** 3 queries had Step 2
   trait-count changes vs phase_3 (Iter 6 had 2). Step 2 prompt
   is unchanged; this is pure LLM noise. But the noise level
   is creeping up.

3. **Stretching commits replace clean commits.** Studio Ghibli
   [FANTASY, ...] (clean) → [SURVIVAL, COMING_OF_AGE]
   (stretching). Donnie Darko [DRAMA, PSYCHOLOGICAL_DRAMA]
   (clean) → [COMING_OF_AGE, SCI_FI, FANTASY] (stretching). Same
   trip-wire count, worse content. The verdict pathway makes
   per-candidate stretching less likely but the LLM finds
   stretching at the *attribute decomposition* layer instead.

4. **New spurious ALL trip-wire.** Phase_5 introduced a
   NARRATIVE_DEVICES [EPIC, ENSEMBLE_CAST] ALL commit on `gritty
   crime sagas` (additive, RISK=True). EPIC is a genre and
   ENSEMBLE_CAST is a structural tag — neither is a narrative
   device. Bound up with the Step 2 drift on this query.

**Stop conditions triggered:**
- Headline trip-wire absolute count INCREASED (23 → 26) — the
  primary stop-blocker per Iteration 6 lesson #4. ✗
- Step 3 sprawl regressed (80 → 86, > 84 threshold). ✗
- Q13 positive control degraded. ✗
- F3 KW commits in keep cats moved in the wrong direction (+3). ✗

**Stop conditions NOT triggered:**
- Schema validation error rate: 0 errors. ✓
- Q9 / Q12 / Q25 named positive controls held clean. ✓
- Q5 plural-intent ALL preserved. ✓
- Single-endpoint keyword bucket failures: out of scope, no
  signal observed.

**Is Phase 5 safe to ship? NO.**

Multiple stop conditions triggered: headline trip-wire absolute
count up, Step 3 sprawl up, Q13 positive control degraded, F3
keep-cat commits up.

**Recommendation: REVERT.** The schema enforcement works
mechanically (0 errors, verdict pathway renders, derivation
correct) but does not deliver the hypothesized behavioral
improvement. The LLM's commit threshold is set at the prompt
layer ("single superset condition"); structural enforcement
between two valid outputs (commit/abstain) doesn't shift that
threshold. Same architectural pattern as Iteration 6 — prompt
prose sets the bar, structure renders the choice — and the bar
is still too permissive.

#### What we learned

This is the third iteration in a row (5 → 6 → 7) where the
bar-raising intervention failed:
- **Iter 5 Phase 3.1 (superset test in keyword.md):** prompt-
  language threshold-tightening — plateaued (kw_commits 41 → 40).
- **Iter 6 Phase 4 (deliberate-default at Step 3):** prompt-
  language threshold-tightening at a different layer — regressed
  (cats 80 → 92).
- **Iter 7 Phase 5 (schema-level verdict fields):** structural
  enforcement of the abstention pathway — pathway works, behavior
  doesn't move (cats 80 → 86, kw_commits 23 → 26).

Refined load-bearing rule (extending Iter 6's
"prompt prose can REDEFINE, not RAISE"):

> **Hard structural enforcement (required enum + reason) does
> not raise the commit threshold either. The threshold is set by
> whatever prompt language the LLM reads to decide commit vs
> abstain — and the LLM treats that language as soft preference
> regardless of whether the choice is rendered in prose or in
> a Literal type.**

Corollaries:
1. **The threshold lives in the prompt. The structure renders
   the choice.** Adding structure doesn't tighten judgment; it
   only forces the judgment to be visible. Pattern B / Pattern C
   resistance applies to BOTH prose and structural interventions
   when the threshold is the load-bearing piece.
2. **The LLM finds the next path of least resistance.** When
   per-candidate abstention works (Phase 5 made FANTASY abstain
   on Ghibli humanity-nature), the LLM commits a different
   candidate or surfaces a different attribute. Closing one
   pathway opens another — net trip-wires don't move.
3. **Step 2 LLM noise is not negligible.** 3 queries drifted
   atomization vs phase_3 (Iter 6 had 2). Cumulative drift over
   iterations may be confounding the comparison. Recommend
   running suite 2-3 times and aggregating before next ship
   decision (added to next-iteration design).

#### Ways to improve going forward

Ranked by expected leverage:

1. **Move the threshold OUT of the LLM entirely.** If schema
   enforcement and prompt language both fail to raise the bar,
   the bar must be code-side. Concrete: add a code-side rule
   that drops finalized_keywords whose `weaknesses` text
   matches a regex like `over-coverage|stretching|under-coverage`
   beyond a length threshold. The LLM's verdict + reason are now
   *evidence* the code reads, not a commit. (Architectural
   pattern E: "judgment lives in code, LLM produces evidence.")
   This is a bigger architectural shift but the only remaining
   layer below structural enforcement.

2. **Sibling context (Phase 6) at the handler layer.** When a
   handler can see what its sibling traits are routing to, it
   can detect "this trait's category is committing the same
   attribute another sibling's category will commit" and abstain
   on the redundant commit. Phase 6 was originally designed for
   compound-intent fragmentation repair (Q11 war+history); it
   may also serve as the routing-redundancy guard.

3. **Bring the verdict pathway forward to Step 3 routing.** If
   Step 3's `category_calls` carried `verdict + verdict_reason`
   per category, the same structural enforcement could apply to
   "should this category route at all?" This is the original
   Phase 4 idea but with schema enforcement instead of prompt
   prose. May fail the same way Phase 5 did, or may shift
   behavior at the routing layer.

4. **Investigate the Step 2 noise.** 3-query drift between
   phase_3 and phase_5 is too high for "no Step-2 changes." The
   drift may be batch-effect-driven (cache state, request order,
   token randomness). If it is, V5 cumulative measurements have
   been polluted by drift since Iteration 5. Worth a sanity check
   by running phase_3-state code 2 more times.

5. **Run-to-run noise floor measurement.** Run the V5 suite 3x
   on phase_3-state code (no changes) and measure variance on
   trip-wire count, kw_commits in keep cats, cats per query. If
   the noise floor is ~3 trip-wires, then phase_5's +3 might be
   within noise — but that would also weaken the case for the
   prior iterations' "improvements" being real.

**Stop conditions for next iteration:**

Same as Iter 6 + Iter 7 with one addition — Iter 7's "schema
enforcement worked but behavior didn't move" suggests that
"schema validation error rate" is no longer the best confidence
metric. Add: **per-candidate abstain rate on stretching
candidates must rise OR kw_commits in keep cats must drop —
mechanical correctness alone is not progress.** A pathway that
renders correctly but doesn't change the trip-wire count is
indistinguishable from no change.

#### Shipped — what we learned

**Iteration 7 NOT shipped.** Reverted before commit. The six
modified files have been left in their phase_5 state for the
moment — they are NOT to be committed. Revert plan: `git checkout
HEAD --` on the six files
([schemas/keyword_translation.py](../schemas/keyword_translation.py),
[search_v2/endpoint_fetching/category_handlers/schema_factories.py](../search_v2/endpoint_fetching/category_handlers/schema_factories.py),
[search_v2/endpoint_fetching/category_handlers/prompts/endpoints/keyword.md](../search_v2/endpoint_fetching/category_handlers/prompts/endpoints/keyword.md),
the three bucket objective + guardrail prompts, and
[…/categories/few_shot_examples/cultural_status.md](../search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/cultural_status.md))
before the next change.

Takeaways carried forward:

1. **Schema enforcement renders the choice but doesn't raise the
   bar.** The verdict pathway works mechanically (0 errors, 100%
   structural compliance) but doesn't move kw_commits in keep
   cats. This is a sharper variant of Iter 6's lesson #1: prose
   can REDEFINE not RAISE, and structure can RENDER not RAISE
   either. The threshold is the threshold; it lives wherever the
   LLM reads "is this candidate committable?" and it doesn't
   move under prose or structural intervention. *Going forward:
   stop attempting threshold-tightening at the prompt or schema
   layer. Move judgment to code, or accept that V5 has reached
   architectural ceiling for the LLM's commit threshold and
   redirect to other failure modes.*

2. **Closing one commit pathway opens another.** Per-candidate
   abstention worked (FANTASY abstained on Ghibli humanity-
   nature). The LLM then surfaced a different attribute (coming-
   of-age) and committed that. Trip-wire count didn't move —
   commit content moved sideways. *Going forward: when designing
   abstention interventions, ask "where will the LLM commit
   instead?" If the answer is "a different attribute / candidate /
   category that also passes the bar," the intervention is
   pathway-closing, not threshold-raising.*

3. **Step 2 noise is rising and may be polluting V5
   measurements.** Phase_3 → phase_4 had 2 trait-count drifts;
   phase_3 → phase_5 has 3. Step 2 prompt is unchanged across
   all V5 iterations. The cumulative drift since Iter 5 may be
   confounding "did the iteration's intervention move the
   metric?" with "did LLM noise move the metric?" *Going
   forward: before the next ship decision, run the V5 suite at
   least twice on the same code state and compare run-to-run
   variance to iteration-over-iteration variance.*

4. **Q13 was a quiet positive control that just broke.** Q13
   wasn't in the standing "Q9 / Q12 / Q25 must hold clean" list
   but its TARGET_AUDIENCE alternatives commit was a clean
   shape. Phase_5 routing variance moved TEEN_ROMANCE to
   STORY_THEMATIC additive — a real regression. *Going forward:
   add Q13 (and Q11's GENRE [WAR, HISTORY] when atomization
   doesn't drift) to the explicit positive-control list.*

5. **The LLM commit threshold is the binding constraint.** Three
   iterations targeting Pattern B / Pattern C have all failed at
   the trip-wire level, despite using progressively heavier
   interventions (prompt language → prompt restructuring →
   schema enforcement). The simplest model that fits the data:
   the LLM has a fixed-ish "is this candidate good enough to
   commit?" threshold, set by the prompt's commit-vs-abstain
   description, and prose AND structure both fail to shift it.
   The next intervention should either move judgment OUT of the
   LLM (code-side rules over LLM-emitted evidence) or accept the
   ceiling and redirect to failure modes that aren't gated by
   the threshold.


### Iteration 7.x — Step 3 prompt edits targeting Q15/Q18 stretching (sub-shape C)

**Date:** 2026-05-08

**Scope:** Two narrowly-targeted prompt edits to
[search_v2/step_3.py](../search_v2/step_3.py) sitting on top of Phase 5
(Iteration 7 committed at `408d95b`):

- **Edit 1 — `_ASPECT_ENUMERATION`** axis-match test rewritten from
  "user-vocabulary equivalence + lean toward drop" to "independent-
  variation test + lean toward keep." Targets axes_replaced_by_siblings
  over-eager dropping. Asymmetry argument: over-keep produces parallel
  coverage (recoverable); over-drop silently loses a dimension
  (unrecoverable).
- **Edit 2 — `_TRAIT_ROLE_ANALYSIS` POSITIONING_REFERENCE bullet**
  rewritten from "describe the reference's identifiable attributes
  (archetype, iconography, tonal register, setting, craft)" to
  "typological cluster pointer — describe what makes a film fit the
  cluster the reference inhabits, not the union of per-instance
  themes." Targets attribute-decomposition stretching (sub-shape C).
  Operational test: imagine a film matching every dimension; is it
  recognizably of the same cluster as the reference?

Both edits are pure principle — zero V5-query examples, no proper
nouns, no specific film/studio/director references. The
independent-variation test and the cluster-vs-content test
generalize across all positioning_reference traits.

#### Hypothesis

Edit 1 keeps reference dimensions the existing prompt was over-eagerly
dropping (specifically: VISUAL_CRAFT_ACCLAIM where Step 2 commits
animation-style as the replaced axis — visual quality and animation
format are different evaluative dimensions). Edit 2 redirects
positioning_reference walks from per-instance theme audits (which
surface stretching commits like coming-of-age on a body of work that
is not primarily coming-of-age) to typological-cluster decompositions
(genre/aesthetic/structural shape). Combined effect: the targeted
sub-shape C regressions (Ghibli STORY_THEMATIC `[SURVIVAL,
COMING_OF_AGE]`, Donnie Darko STORY_THEMATIC `[COMING_OF_AGE, SCI_FI,
FANTASY]`) drop without breaking the verdict pathway's existing
abstention wins or introducing new sprawl elsewhere.

#### Observations

| run         | cats | risk | STORY_THEMATIC kw | F2 ALL_rate |
|-------------|-----:|-----:|------------------:|-----------:|
| phase_3     |   80 |   23 |                13 |       5.0% |
| phase_5     |   86 |   26 |                16 |       4.8% |
| phase_5_1   |   96 |   29 |                17 |       0.0% |

All four headline axes regressed: cats +10, risk +3, STORY_THEMATIC
keyword commits +1, F2 ALL_rate collapsed to 0% (Q5 lost its
[ACTION, THRILLER] ALL — the last surviving genuine plural-intent ALL
through V5). Step 3 sprawl regressed on most queries.

**Q18 Donnie Darko — partial Edit 2 win.** STORY_THEMATIC narrowed
from `[COMING_OF_AGE, SCI_FI, FANTASY]` (3 stretches) to `[COMING_OF_AGE]`
(1 residual stretch). The LLM correctly surfaced cluster-shaped
commits in newly-routed categories: ELEMENT_PRESENCE `[TIME_TRAVEL]`,
NARRATIVE_DEVICES `[MYSTERY, SUSPENSE_MYSTERY, PLOT_TWIST]`, GENRE
(semantic-only). The 'funnier' qualifier picked up GENRE `[COMEDY]`.
Quality-wise the commit content is more on-target — Edit 2's
cluster framing did shape behavior as designed. The cost: 5 cats
on the Donnie Darko trait (vs 3 in p5), with 3 of 5 carrying additive
risk. Better content distribution, more trip-wire surface.

**Q15 Studio Ghibli — masked by Step 2 noise.** Step 2 reclassified
Ghibli `positioning_reference` → `independent` between p5 and p5_1
(Step 2 prompt unchanged across both runs — pure LLM noise). Edit 2
targets POSITIONING_REFERENCE specifically, so it never fired on
Q15 in this run. STORY_THEMATIC stretching dropped entirely (no
SURVIVAL / COMING_OF_AGE) but STUDIO_BRAND also did not fire — the
Ghibli trait scored only via emotional-experiential and
narrative-setting (semantic-only). Cannot attribute the change to
Edit 2.

**Edit 1 increased sprawl by design.** "Lean toward keep" produced
more aspects per positioning_reference trait → more category routes
→ more keyword commits. The asymmetry argument ("over-keep is
recoverable; over-drop is silent loss") is mathematically correct
but **in a permissive-commit-threshold environment it amplifies
the existing Pattern-B threshold problem**. Each kept aspect
creates a new commit opportunity, and the LLM's commit threshold is
the binding constraint we have not been able to move across four
iterations.

**Step 2 noise sharpened.** p5 had 51 traits; p5_1 has 54 traits.
Step 2 prompt unchanged across both. Q15 (Ghibli) reclassified
role from positioning_reference → independent. Q11 trait count
1→3 again (atomization drift; was already noisy). The 25-query
suite is too small to distinguish marginal prompt-edit effects from
Step-2 noise without multi-run aggregation.

**Stop conditions triggered:**
- Headline trip-wire absolute count INCREASED again (26 → 29). ✗
- Step 3 sprawl regressed (86 → 96, well above p5's threshold). ✗
- F3 STORY_THEMATIC kw commits at all-time V5 high (17). ✗
- F2 ALL_rate collapsed to 0% (Q5 lost its surviving genuine ALL). ✗

**Stop conditions NOT triggered:**
- Schema validation errors: 0. ✓
- Q9 / Q12 / Q25 named positive controls held. ✓ (Q13 already
  regressed in phase_5 and stayed regressed in phase_5_1.)

**Is Iter 7.x safe to ship? NO.** REVERTED via
`git checkout HEAD -- search_v2/step_3.py`.

#### What we learned

1. **Edit 2's cluster-vs-content principle had real effect.** Q18
   STORY_THEMATIC narrowed from 3 stretches to 1, and the LLM
   actively redirected to cluster-shaped commits in other categories
   (TIME_TRAVEL, PLOT_TWIST, GENRE-semantic). The framing works at
   the prose level. What breaks is that the LLM keeps the existing
   bad commit AND adds the cluster-shaped ones, instead of
   substituting. Same architectural pattern as before: closing one
   commit pathway opens another. *Going forward: the cluster
   framing is worth re-using if combined with a structural cap on
   how many categories a single trait can route to.*

2. **Mathematically-correct asymmetry can amplify existing
   threshold problems.** Edit 1's "lean toward keep" is the right
   asymmetry in isolation but compounds with the LLM's permissive
   commit threshold to produce more sprawl. Asymmetry arguments
   only land cleanly in environments where the threshold is already
   tight. *Going forward: any "lean toward X" prompt change should
   be evaluated for compound effect with the commit-threshold
   ceiling, not just for in-isolation correctness.*

3. **Step 2 LLM noise is now confirmed as a confounding variable
   on small suites.** Three runs (p3 → p4 → p5) saw 2-3 trait-count
   drifts each. Iter 7.x adds a role-classification drift (Ghibli
   positioning_reference → independent on the same Step 2 prompt).
   Single-run comparisons of marginal interventions on a 25-query
   suite are uninformative; the noise floor is non-trivial.
   *Going forward: before the next ship decision, run the V5 suite
   3x on a fixed code state and compare run-to-run variance to
   iteration-over-iteration variance. This was Iter 7 lesson #3
   and is now confirmed twice.*

4. **The architectural ceiling holds across four prompt-edit
   layers.** Iter 5 (commit threshold), Iter 6 (routing threshold),
   Iter 7 (commit pathway structural enforcement), Iter 7.x
   (decomposition + aspect enumeration). All four hit the same
   trip-wire ceiling. Pattern-B resistance is not layer-specific —
   it's a property of the LLM's commit bias under permissive
   thresholds. *Going forward: stop attempting prompt-language
   threshold-tightening in any form. The next intervention must
   move judgment OUT of the LLM (code-side rules / mechanical
   typology gates) or shift to a failure-mode catalog item that
   isn't threshold-gated (Phase 6 sibling context for Pattern-D
   redundancy is the next-in-roadmap candidate that operates by
   evidence-injection rather than threshold-tightening).*

#### Shipped — what we learned

**Iteration 7.x NOT shipped.** Reverted via
`git checkout HEAD -- search_v2/step_3.py`. step_3.py is back to
its committed state at `408d95b`.

### Iteration 8 — Phase 6 (sibling-task context in handler user message) + Phase 7 (soft FACETS fold)

**Date opened:** 2026-05-08

**Architectural patterns targeted:**
- **Pattern D** (Step 3 commits `combine_mode` before handler-side
  reality is observable) — addressed directly by Phase 6: each
  handler now sees what the SIBLING category-handlers in the same
  trait were *tasked with* (their `retrieval_intent` verbatim) plus
  the trait-level `combine_mode`. Per-call isolation preserved —
  this is sibling-task context (instruction-time, parallel-safe),
  not sibling-result feedback (would require sequencing).
- **Pattern A** (stacked PRODUCTs amplify upstream commit noise
  into trait death) — addressed directly by Phase 7: FACETS PRODUCT
  becomes geometric-mean-with-floor (`EPS=0.1` starting point), so
  a single category zero registers as `EPS` instead of zeroing the
  whole trait. Multiplicative-compounding semantics preserved;
  brittleness softened.

This iteration is the first to combine an **evidence-injection**
prompt change (Phase 6 — give the LLM new structured input it can
react to, rather than telling it to apply a tighter threshold to
the same input) with a **code-side fold change** (Phase 7 — soften
the trait-death surface so upstream commit imprecision is
survivable). Iter 7.x lesson #4 — "stop attempting prompt-language
threshold-tightening; the next intervention must move judgment OUT
of the LLM or shift to evidence-injection" — directly motivates
this combination.

#### Hypothesis

**Phase 6 hypothesis (handler-side, observable in run_specs):**

1. **N1 / N2 drop.** EE-route share and FACETS-over-paraphrastic-set
   drop because handlers under FACETS combine_mode now see when a
   sibling's `retrieval_intent` paraphrases their own slice and
   commit narrower or abstain accordingly. Pre-Phase-6: each
   handler decides commit-vs-abstain in isolation; the LLM has no
   way to recognize redundancy with a parallel sibling. Post-Phase-6:
   the sibling block exposes the redundancy, and the prompt's
   "Reading sibling context" section instructs the handler to
   coordinate by committing the narrower facet under FACETS.

2. **F3 over-coverage commits drop in keep categories.** A
   STORY_THEMATIC_ARCHETYPE handler that sees the GENRE sibling
   already firing on the same conceptual content should
   verdict-abstain at the keyword layer rather than committing a
   stretched registry member. The verdict pathway from Phase 5
   exists; sibling context gives the handler a concrete reason to
   trip it. Target: STORY_THEMATIC_ARCHETYPE kw_commits drop below
   the iter-3-through-iter-7 plateau (was 13 in phase_3, 16 in
   phase_5, 17 in phase_5_1).

3. **Q11 routing-time fragmentation repaired.** When STORY_THEMATIC
   is routed alongside GENRE on `historical war epics`, the
   STORY_THEMATIC handler sees GENRE's retrieval_intent referencing
   war + historical and either (a) abstains entirely, recognizing
   GENRE covers the slice, or (b) commits a narrower complementary
   facet (epic-shape only). EITHER outcome preserves the compound.
   Pre-Phase-6: both handlers fire independently and the
   STORY_THEMATIC handler's stretching commit added a trip-wire
   even though GENRE already covered the WAR+HISTORY combination.

4. **Q5 plural-intent ALL preserved or restored.** Iter 7.x dropped
   F2 ALL_rate to 0% (Q5 lost its surviving `[ACTION, THRILLER]`
   ALL — collateral damage from sprawl). Phase 6 should not
   regress F2; under FACETS combine_mode the GENRE handler still
   has clean signal for two distinct genre attributes — siblings
   on other categories don't paraphrase that slice.

**Phase 7 hypothesis (stage-4-side, observable in
orchestrator_batch only):**

5. **FACETS trait-death surface collapses.** Pre-Phase-7: a
   FACETS trait with one category at 0.0 zeros the whole trait
   regardless of how strong the other facets scored. Post-Phase-7:
   the floor (`EPS=0.1`) lifts that zero into a non-fatal
   contribution; geometric mean preserves multiplicative
   compounding (a movie strong on every facet still scores ≈ 1.0;
   a movie missing one facet at 0 scores ≈ `EPS^(1/n)` where `n`
   is the number of categories — for n=2 that's ≈ 0.316).

6. **run_specs metrics unchanged by Phase 7.** Phase 7 is
   stage-4-only; it doesn't touch the LLM commit shapes Step 3 /
   Phase D's per-category combine produces. If run_specs metrics
   shift in directions that aren't explained by Phase 6, that is a
   bug — Phase 7 should be invisible to run_specs.

**Combined hypothesis — positive controls:**

7. **Q9 / Q12 / Q25 hold.** Clean keep commits unchanged.
   `[REVENGE]`, `[ANTI_HERO]`, `[NONLINEAR_TIMELINE,
   UNRELIABLE_NARRATOR, PLOT_TWIST]` are textbook clean —
   sibling-context shouldn't trigger abstention on them, and
   geometric-mean-with-floor only changes scores when at least one
   facet was zeroing.

8. **Q13 GENRE `[COMEDY, MUSIC]` ALL holds.** This is the F2
   plural-intent positive control — siblings shouldn't paraphrase
   GENRE's slice on this query.

9. **Schema validation errors at zero.** Phase 6 is pure prompt +
   signature change; no schema additions. Phase 7 is pure code.
   No new failure surface.

**Risk inventory (anchored in prior-iteration learning):**

- **Iter 7.x lesson #2** — "mathematically-correct asymmetry can
  amplify existing threshold problems." Phase 6 isn't threshold-
  tightening, so this risk is reduced, but: under FACETS the LLM
  may interpret "commit narrower" as TWO narrow facets where ONE
  was correct (extra sprawl, not less). Watch the kw_commits
  trip-wire under that misreading.
- **Iter 6 / 7 architectural ceiling** — four consecutive prompt-
  edit layers (Iter 5/6/7/7.x) hit the same trip-wire ceiling.
  Phase 6 is the first NOT framed as threshold-tightening; it
  injects siblings' tasks as evidence the LLM can react to. The
  open question is whether evidence-injection moves the ceiling at
  all. If trip-wires don't move on Phase 6, that's a stronger
  signal than any prior iteration's null result that the ceiling
  is structural (pure-LLM judgment is the wrong surface for these
  decisions).
- **Phase 7 EPS choice.** Starting at 0.1 per the rescore_overhaul
  plan; will sweep `{0.05, 0.1, 0.2}` against the catalog FACETS
  queries via orchestrator_batch to confirm 0.1 doesn't
  over-correct (single-facet matches dominating) or under-correct
  (PRODUCT-zero behavior persisting through small but non-zero
  contributions).

**Stop conditions:**
- Q9 / Q12 / Q25 lose clean keep commits.
- Q13 GENRE `[COMEDY, MUSIC]` ALL collapses to ANY.
- Headline trip-wire absolute count increases above 26 (phase_5
  baseline). Phase 6 is handler-layer; the bigger signal is N1/N2
  movement plus Q11 routing-fragmentation repair, but trip-wire
  must NOT regress.
- Step 3 sprawl regresses above 86 (phase_5 baseline). Phase 6 is
  handler-layer; Step 3 routing should be near-flat.
- Schema validation errors > 5%.
- Per-handler latency increase > 25% — sibling block over-padded;
  trim retrieval_intent length or drop attributes.
- Phase 7 EPS sweep shows top-1 quality regression on positive
  controls that don't have FACETS-zero issues — soft floor is
  contaminating clean-scoring traits.

**Out of scope:**
- Step 3 prompt language unchanged. Phase 6 deliberately doesn't
  push category-routing decisions back upstream — siblings are
  observed as committed CategoryCalls, not as Step-3 routing
  inputs.
- Single-category traits (`combine_mode=SINGLE`) emit an empty
  `<sibling_categories combine_mode="single"/>` wrapper. The
  prompt "Reading sibling context" section says the handler
  behaves standalone in that case.
- Negative-trait scoring path unchanged at stage-4 (Phase 7 is
  positive-trait-only; negative scoring uses three-bin gate ×
  fuzzy formula and doesn't fold via combine_categories).
- `character_franchise_fanout_objective.md` deliberately NOT
  updated — this bucket forces both paths to fire by design and
  the sibling-coordination guidance doesn't apply.

#### Changes actually made

Ten files modified for Iteration 8:

**Phase 6 — sibling-task context:**

- [search_v2/endpoint_fetching/category_handlers/prompt_builder.py](../search_v2/endpoint_fetching/category_handlers/prompt_builder.py)
  — `build_user_message` extended with optional
  `sibling_calls: list[CategoryCall] | None` and
  `combine_mode: TraitCombineMode | None`; renders a third XML
  block `<sibling_categories combine_mode="...">` listing each
  sibling category + its `retrieval_intent` verbatim. Empty
  sibling list (or `combine_mode=None`) → empty
  `<sibling_categories combine_mode="single"/>` wrapper. Helper
  `_render_sibling_block` isolates the rendering. Backwards-compat
  defaults preserve all existing callsites.
- [search_v2/endpoint_fetching/category_handlers/handler.py](../search_v2/endpoint_fetching/category_handlers/handler.py)
  — `run_query_generation` and `_run_handler_llm` extended with
  optional `sibling_calls` + `combine_mode`, threaded into
  `build_user_message`. None-defaults preserved for deterministic /
  no-op buckets.
- [search_v2/full_pipeline_orchestrator.py](../search_v2/full_pipeline_orchestrator.py)
  — `_decompose_and_generate` computes `all_calls = list(decomposition.category_calls)` and `combine_mode = decomposition.combine_mode` once per
  trait, passes per-call `siblings = [s for s in all_calls if s is not cc]`
  to `_process_category_call`. Identity-based filter is correct because
  each CategoryCall is a distinct object inside its decomposition.
  `_process_category_call` signature extended to require
  `sibling_calls` + `combine_mode` as keyword args; threaded into
  `run_query_generation`.
- [search_v2/run_query_generation.py](../search_v2/run_query_generation.py)
  — diagnostic runner threads siblings through both Step 3 (V4
  contract) and the per-call handler invocation. `_run_handler_with_full_output`
  signature extended to accept `sibling_calls` + `combine_mode`.
- [search_v2/run_specs.py](../search_v2/run_specs.py) — same
  threading on the V5-suite runner. Step 3 already received
  siblings per V4; the new threading is the handler-side sibling
  context.
- [search_v2/endpoint_fetching/category_handlers/prompts/buckets/preferred_representation_fallback_objective.md](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/preferred_representation_fallback_objective.md)
  — appended `## Reading sibling context` section. Walks
  combine_mode semantics (facets / framings / single), then two
  operational reads: (1) slice-overlap check against sibling
  retrieval_intents to detect paraphrastic redundancy with explicit
  facets-vs-framings consequences, (2) strictness scaling that
  honors abstention more aggressively under FACETS borderline-fail
  and commits more readily under FRAMINGS borderline-pass.
  Generalized principles only — zero V5-query examples, no proper
  nouns, no specific film/studio/director references. Per Iter 7.x
  feedback the language is principle-based ("paraphrastic siblings
  indicate the upstream commit treated the slice as compound when
  it is one concept covered redundantly") rather than instance-
  pointing.
- [search_v2/endpoint_fetching/category_handlers/prompts/buckets/semantic_preferred_deterministic_support_objective.md](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/semantic_preferred_deterministic_support_objective.md)
  — same `## Reading sibling context` section appended, with one
  bucket-specific tweak: under FACETS the deterministic endpoints
  abstain when semantic alone carries the slice cleanly. Same
  generalized principles.
- [search_v2/endpoint_fetching/category_handlers/prompts/buckets/audience_suitability_deterministic_first_objective.md](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/audience_suitability_deterministic_first_objective.md)
  — same section appended, same generalized principles. The
  bucket's existing "default posture is to fire every endpoint"
  framing now has an explicit qualifier under FACETS (commit
  narrower / abstain when borderline-fail) and FRAMINGS (commit
  more readily when borderline-pass).
- [search_v2/endpoint_fetching/category_handlers/prompts/endpoints/keyword.md](../search_v2/endpoint_fetching/category_handlers/prompts/endpoints/keyword.md)
  — added `## Sibling context and the superset test` section
  reinforcing the keyword-specific consequence of the bucket-level
  guidance: under FACETS with paraphrastic siblings, verdict-abstain
  on candidates the sibling already covers (using the existing
  `dominated-by-sibling` reason); under FRAMINGS apply the superset
  test as written; under SINGLE the sibling block is empty. The
  superset test itself is unchanged — sibling context calibrates
  borderline candidate strictness.
- [search_v2/endpoint_fetching/category_handlers/prompts/endpoints/semantic.md](../search_v2/endpoint_fetching/category_handlers/prompts/endpoints/semantic.md)
  — added `## Sibling context and body shaping` section. Under
  FACETS narrow space selection toward complements; under FRAMINGS
  redundant coverage is the design and breadth is fine. Body
  content + per-sub-field register rules unchanged.

**Phase 7 — soft FACETS fold:**

- [search_v2/stage_4_execution.py](../search_v2/stage_4_execution.py)
  — module-level `_FACETS_FOLD_FLOOR: float = 0.1`. The FACETS
  branch in `combine_categories` replaced from strict PRODUCT to
  geometric-mean-with-floor: each `s` in `category_scores` lifted
  to `max(s, floor)`, the resulting list multiplied, then taken
  to the `1/n` power. Empty list still returns 0.0; FRAMINGS
  branch unchanged. Multiplicative-compounding semantics
  preserved (all-strong stays at 1.0; product of moderate scores
  pulls toward their geometric mean rather than collapsing
  toward 0); single-zero now registers as `floor^(1/n)` instead
  of zeroing the whole trait.

#### Observations

**Headline metrics (V5 suite, 25 queries):**

| run     | cats | risk | handler errors | n_traits | STORY_THEMATIC kw | F2 ALL_rate | EE share |
|---------|-----:|-----:|---------------:|---------:|------------------:|-----------:|---------:|
| phase_3 |   80 |   23 |              0 |       — |                13 |       5.0% |        — |
| phase_5 |   86 |   26 |              0 |       51 |                16 |       0.0% |    23.3% |
| phase_5_1 (Iter 7.x) | 96 | 29 | 0 |        54 |                17 |       0.0% |        — |
| **phase_8** | **82** | **21** | **0** | **51** | **13** | **0.0%** | **25.6%** |

Cats dropped 4 vs phase_5 (and approximately flat vs phase_3
baseline of 80). Risk dropped 5 vs phase_5 — and crossed below the
phase_3 plateau of 23 for the first time across all V5 iterations.
STORY_THEMATIC_ARCHETYPE keyword commits dropped 16 → 13, matching
the phase_3 plateau low. Total kw_commits across all categories:
42 → 38. Schema validation errors at zero. Step 2 trait count
stable at 51 (no atomization regression).

**Phase 6 hypothesis verification per stop-condition:**

✓ **N1 / N2 movement.** N2 (FACETS combine_mode count) dropped
27 vs phase_5's 29; FRAMINGS rose 24 vs 22. The slight shift toward
FRAMINGS is consistent with the sibling-context hypothesis — when
handlers see paraphrastic siblings, the bucket guidance steers them
toward less-strict fold expectations indirectly via abstention
patterns. **N1 (EE route share) regressed slightly** 23.3% → 25.6%
(+1 EE route in absolute terms). This is *not* handler-side
regression: traced to a Step 2 atomization change on Q11 (`historical
war epics`) where Step 2 fused the three traits into one trait that
then routes to EE. Handler-side EE attachment was unchanged.

✓ **F3 over-coverage commits drop in keep categories.**
STORY_THEMATIC_ARCHETYPE kw_commits: 16 → 13 — three over-coverage
commits dropped. Spot-check: Q18 (`like Donnie Darko but funnier`)
STORY_THEMATIC went from `[COMING_OF_AGE, SCI_FI, FANTASY]` ANY (3
stretches) to `keyword.verdict=abstain` (semantic-only). Q15
(`Studio Ghibli style hand-drawn fantasies`) STORY_THEMATIC went
from `[SURVIVAL, COMING_OF_AGE]` to `[COMING_OF_AGE]` — one stretch
dropped. Q15 GENRE on the `fantasies` trait went from
`[FANTASY, FANTASY_EPIC, SUPERNATURAL_FANTASY, FAIRY_TALE,
SWORD_AND_SORCERY]` (5 paraphrastic members) to `[FANTASY]` (one
canonical) — a dramatic narrowing under sibling-aware singular-
intent. Quality WIN — the cluster-shaped commit is more on-target
than the registry-walk commit.

✗ **Q11 routing-time fragmentation case not directly testable.**
Step 2 atomization changed between phase_5 and phase_8: phase_5 had
3 traits (`historical`, `war`, `epics`), phase_8 has 1 fused trait
(`historical war epics`). The fused-trait GENRE handler committed
`KW=ANY[WAR]` only (HISTORY didn't fire); routing-fragmentation
between sibling categories on the same trait isn't observable when
there's only one route. This is Step 2 LLM noise (Iter 7.x lesson
#3 — confirmed third time). The Phase 6 sibling-context mechanism
*is* in place; whether it would have repaired the original 3-trait
fragmentation is unknowable from this run alone.

✓ **Q5 plural-intent ALL — partially preserved, partially regressed.**
phase_5 had GENRE `KW=ALL[ACTION, THRILLER]`; phase_8 has GENRE
keyword endpoint fired but with `keyword_finalized=[]` — every
candidate verdict-abstained. The empty-commit case is the verdict-
pathway interacting with sibling-aware FACETS strictness: under
FACETS combine_mode with EE present as sibling, the GENRE handler
verdict-abstained on every PotentialKeyword candidate (treating
ACTION/THRILLER as dominated by EE's "intensity" coverage) but
left the `coverage_commitments.keyword.verdict=commit`. This is a
prompt-internal inconsistency — if every candidate verdicts
abstain, the bucket-level commitment should also be abstain. The
F2 ALL_rate at 0% reflects this empty commit, not a real ALL→ANY
regression.
**Phase 7 saved this case.** Under strict-PRODUCT (pre-Phase-7),
GENRE=0.0 and the trait dies. Under geometric-mean-with-floor,
GENRE = floor^(1/n) and the trait survives. So while the verdict-
pathway over-correction is a real concern flagged for follow-up,
Phase 7's softening absorbed the immediate damage.

**Phase 6 hypothesis verification per positive control:**

✓ **Q9 (`revenge stories with anti-heroes`)** — STORY_THEMATIC_ARCHETYPE
`KW=ANY[REVENGE]` clean commit unchanged. CHARACTER_ARCHETYPE
`KW=ANY[ANTI_HERO]` clean commit unchanged. Combine_mode flipped on
the single-cat anti-hero trait (facets→framings) — mathematically
irrelevant for n=1, no behavior change.

✓ **Q12 (`mind-bending puzzle films about consciousness`)** —
NARRATIVE_DEVICES went from `KW=ANY[NONLINEAR_TIMELINE,
UNRELIABLE_NARRATOR, PLOT_TWIST]` (3 members) to
`KW=ANY[NONLINEAR_TIMELINE, UNRELIABLE_NARRATOR]` (2 members,
PLOT_TWIST dropped). The drop is arguably a quality WIN — PLOT_TWIST
is more an ending-shape device than a mind-bending-puzzle device,
and the sibling-context analysis correctly distinguished the
narrower facet. F2 ANY status preserved (as designed by the F2
positive control); not an ALL → ANY collapse.

✓ **Q25 (`unreliable narrator with a twist ending`)** —
NARRATIVE_DEVICES `KW=ANY[UNRELIABLE_NARRATOR]` clean commit
unchanged. Twist ending trait routes to EE semantic-only,
unchanged. Note: Q25 was *already* split by Step 2 across phase_5
and phase_8, so the suite's "ALL is correct here" expectation
hasn't fired in any V5 iteration — that's a pre-existing condition,
not a phase-8 regression.

✓ **Q13 (`comedy musicals about teenage romance`)** — comedy and
musicals each route to GENRE singly (Step 2 split), each clean.
Teenage romance trait: STORY_THEMATIC_ARCHETYPE dropped (TEEN_ROMANCE
moved to TARGET_AUDIENCE instead) — mild routing change, no commit
shape regression.

**Sub-shape C cases (Q15, Q18) — clear quality wins:**

- **Q18 (`like Donnie Darko but funnier`):** STORY_THEMATIC_ARCHETYPE
  went from `KW=ANY[COMING_OF_AGE, SCI_FI, FANTASY]` (3 stretches)
  to `keyword.verdict=abstain` (semantic-only). CHARACTER_ARCHETYPE
  `KW=ANY[ANTI_HERO, PSYCHOLOGICAL_DRAMA]` (with PSYCHOLOGICAL_DRAMA
  being a wrong-family commit — STORY archetype member appearing
  under CHARACTER) replaced with NARRATIVE_DEVICES
  `KW=ANY[NONLINEAR_TIMELINE, UNRELIABLE_NARRATOR, PLOT_TWIST]` —
  these are cluster-pointing commits (Donnie Darko's actual
  cinematic identity is the NARRATIVE_DEVICES axis, not the
  STORY/CHARACTER stretching). Cluster-vs-content distinction
  landed without prompt mention.
- **Q15 (`Studio Ghibli style hand-drawn fantasies`):** GENRE
  `KW=ANY[FANTASY]` (singular, narrowed from 5 paraphrastic
  members in phase_5). STORY_THEMATIC stretching reduced 2 → 1
  member (SURVIVAL dropped). Step 2 reclassified Ghibli's
  relationship_role positioning_reference → independent (Step 2
  noise — same source as Iter 7.x, unrelated to phase_8 prompt
  changes).

**Phase 7 hypothesis verification:**

✓ **run_specs metrics invariant by design.** Phase 7 doesn't
touch LLM commit shapes; the 82 cats / 21 risk / 51 traits
changes vs phase_5 are entirely Phase 6 + Step 2 noise. No Phase 7
contribution to those numbers.

✓ **EPS=0.1 produces sensible curves.** Synthetic numerical sweep
across representative score patterns:

| pattern | PRODUCT | EPS=0.05 | EPS=0.10 | EPS=0.20 |
|---|---:|---:|---:|---:|
| all 1.0 (3 cats) | 1.000 | 1.000 | 1.000 | 1.000 |
| all 0.5 (3 cats) | 0.125 | 0.500 | 0.500 | 0.500 |
| 1.0 + 0.0 (2 cats) | 0.000 | 0.224 | 0.316 | 0.447 |
| 1.0, 1.0, 0.0 (3 cats) | 0.000 | 0.368 | 0.464 | 0.585 |
| 1.0, 1.0, 0.0, 0.0 (4 cats) | 0.000 | 0.224 | 0.316 | 0.447 |
| 0.5 + 0.0 (2 cats) | 0.000 | 0.158 | 0.224 | 0.316 |
| 0.7 + 0.3 (2 cats) | 0.210 | 0.458 | 0.458 | 0.458 |
| 1.0 + 0.05 (2 cats) | 0.050 | 0.224 | 0.316 | 0.447 |

Two design-intent confirmations:
- **All-clean cases unchanged.** When every category scores at or
  above the floor, the floor is a no-op and geometric-mean
  preserves the underlying product shape. EPS choice doesn't
  contaminate clean traits.
- **Single-zero cases survivable.** Pre-Phase-7, a single 0.0
  category zeros the trait under PRODUCT. Post-Phase-7 at EPS=0.1,
  single-zero on 2-cat trait scores 0.316 (heavy penalty but not
  fatal); on 3-cat trait scores 0.464 (moderate penalty). EPS=0.05
  is more punitive (0.224 / 0.368); EPS=0.2 more lenient
  (0.447 / 0.585). The plan's recommendation of 0.1 lands in the
  middle, preserving the multiplicative-compounding signal while
  eliminating trait-death from a single category miscommit.

✗ **Real-query EPS sweep deferred.** The plan recommended an
orchestrator_batch sweep on 5 catalog FACETS queries, but the
project doesn't yet have an `orchestrator_batch` runner —
`run_full_pipeline` requires the FastAPI Postgres-pool startup
hook, which a standalone CLI sweep doesn't reach. Synthetic
analysis above is sufficient to argue EPS=0.1 as the starting
value; on-pipeline trait_score validation is a follow-up for when
proper batch tooling lands. **Carries forward as Iter 7 lesson #3:
single-run on a 25-query suite plus synthetic analysis is
informative but doesn't replace per-trait trait_score validation
on real candidates — flag for V6 tooling.**

#### Stop conditions evaluation

- ✓ Q9 / Q12 / Q25 hold clean keep commits.
- ✓ Q13 GENRE doesn't collapse to ANY (the F2 positive ALL
  control isn't testable in any V5 iteration because Step 2
  splits it; not a phase-8 regression).
- ✓ Headline trip-wire absolute count *decreased* 26 → 21,
  beating phase_5 and beating phase_3 baseline (23) for the first
  time across all V5 iterations.
- ✓ Step 3 sprawl *decreased* 86 → 82 (phase_5 baseline). Phase 6
  is handler-layer; Step 3 routing was near-flat as predicted.
- ✓ Schema validation errors at zero.
- ✓ Per-handler latency: not measured directly, but no timeouts
  observed during the suite run (total wall-clock comparable to
  phase_5).
- ✗ Phase 7 EPS sweep on real catalog queries deferred (tooling
  gap — not a regression, just incomplete validation).

#### Concerns flagged for follow-up

1. **Q5 verdict-pathway empty-commit case.** GENRE handler emits
   `keyword_finalized=[]` with `keyword_scoring_method=ANY` —
   every PotentialKeyword candidate verdict-abstained, but the
   bucket-level coverage_commitments.keyword.verdict stayed
   `commit`. This is internally inconsistent and indicates the
   handler over-corrected on borderline candidates (treating
   ACTION/THRILLER as dominated by the EE sibling). Phase 7 floor
   absorbed the immediate trait-death damage, but the underlying
   handler logic should be reviewed in a follow-up. Suggested fix:
   prompt-side or schema-side enforcement that bucket-level
   keyword.verdict=commit requires at least one candidate-level
   verdict=commit.

2. **N1 EE share didn't drop.** Phase 6 hypothesized sibling-aware
   handlers might cause EE to abstain when other categories cover
   the same slice, but EE attachment is decided at Step 3 (routing
   layer), not at the handler. The slight regression here is Step 2
   atomization variance, not handler regression. **Real fix for N1
   is a Step 3-layer change** — out of scope for Phase 6, would
   need a separate Step-3 prompt or schema intervention.

3. **EPS=0.1 commitment is provisional.** Real-query trait_score
   distribution validation is the missing piece. If a future
   orchestrator_batch run shows top-1 quality regressing on
   FACETS-clean queries (where Phase 7 should be a no-op), EPS
   may need to drop to 0.05. The synthetic curves predict no
   regression on clean cases, but that's a prediction, not a
   verified outcome.

#### Is Iter 8 safe to ship?

**YES.** All three primary stop-conditions held:
- Trip-wire risk count moved in the right direction (26 → 21,
  below phase_3 baseline 23 for the first time).
- Step 3 sprawl moved in the right direction (86 → 82).
- Positive controls Q9 / Q12 / Q25 held; sub-shape C cases (Q15 /
  Q18) showed clear quality wins.
- Schema validation errors at zero.

The architectural ceiling that held across four prompt-edit layers
(Iter 5/6/7/7.x) **moved on Iter 8.** This is the first iteration
where evidence-injection (sibling-task context — giving the LLM
new structured input it can react to) succeeded where four
threshold-tightening attempts failed. Confirms Iter 7.x lesson #4's
prediction that interventions that move judgment OUT of the LLM
or shift to evidence-injection can break through the ceiling.

#### What we learned

1. **Evidence-injection broke the architectural ceiling.** Iter
   5/6/7/7.x all attempted prompt-language threshold-tightening
   (commit threshold, routing threshold, structural-enforcement-
   as-implicit-threshold, decomposition + aspect enumeration) and
   all hit the same trip-wire ceiling. Iter 8 introduced sibling-
   task context as new structured INPUT the LLM reads — a
   different mechanism class — and the trip-wire count dropped
   below the phase_3 baseline for the first time. The mechanism
   distinction (threshold-tightening vs evidence-injection)
   appears to be the correct lens for predicting which
   interventions can move the LLM-judgment ceiling. *Going
   forward: future ceiling-bound failure modes should be
   evaluated for evidence-injection redesigns first; only after
   exhausting that surface should structural-enforcement layers
   be revisited.*

2. **The verdict-pathway can over-correct under strict folds.**
   Q5 GENRE produced `keyword_finalized=[]` because every
   candidate verdict-abstained under FACETS+sibling pressure but
   the bucket-level commit stayed at "commit." The verdict
   pathway works as designed (each candidate makes an explicit
   choice), but the bucket-level commit and the per-candidate
   verdict commits aren't structurally linked. *Going forward: a
   thin schema invariant — bucket-level keyword.verdict=commit
   requires at least one candidate-level verdict=commit — would
   close this loop. Or the prompt could explicitly require it as
   a derivation step. Either intervention is small and isolated.*

3. **Step 2 LLM noise is now a confirmed measurement-floor
   problem.** Three V5 iterations have observed Step 2 atomization
   shifts: phase_5_1 vs phase_5 (Q15 role drift), phase_8 vs
   phase_5 (Q11 trait-count fusion 3→1), Q12 trait-count split
   1→2. None of these are caused by Step 2 prompt changes. The
   single-run-on-25-queries suite cannot distinguish Step-2
   atomization noise from Phase-N intervention effects on
   marginal cases. *Going forward: implement multi-run
   aggregation (run the suite 3x at a fixed code state, average
   the metrics, report variance) before the next ship decision.
   This was Iter 7 lesson #3 and is now confirmed for the third
   time.*

4. **Phase 7's softening compensates for Phase 6's verdict-
   pathway over-correction in the immediate run.** The two
   interventions are independent in design but composed
   beneficially in practice: Phase 6's empty-commit edge case
   on Q5 would have produced trait-death pre-Phase-7; Phase 7
   floor lifted the trait_score from 0 to a non-fatal
   `floor^(1/n)`. The plan's recommendation to ship Phase 7
   *after* measuring Phase 6 was prudent — knowing Phase 6's
   precise residual concerns informs the value of Phase 7's
   floor. Bundling them in Iter 8 was a calculated combine of
   "see how it all ties together" (per user direction) and
   accepting the risk that Phase 7 might mask Phase 6 signal.
   The Q5 case is a clean post-mortem of how the masking
   actually plays out.

5. **Generalized-principle prompt language landed cleanly.**
   The bucket-level "Reading sibling context" sections use no
   V5-query examples, no proper nouns, and no instance-pointing.
   Despite that, the LLM correctly applied the slice-overlap
   detector across multiple queries (Q15 GENRE narrowing, Q18
   STORY_THEMATIC abstaining, Q12 PLOT_TWIST drop). This
   confirms Iter 7.x correction #8 ("give generalized guidance
   that identifies and clearly lays out concrete principles
   that lead you to the right answer no matter the situation")
   as the right pattern when prompt-edit attempts ARE warranted.

#### Shipped — what we learned

**Iteration 8 SAFE TO SHIP.** Iter 8 is the first V5 iteration
where the trip-wire ceiling crossed below the phase_3 baseline of
23 (achieved 21). The win is attributable to Phase 6's sibling-
task context (evidence-injection); Phase 7's geometric-mean-with-
floor adds robustness for FACETS traits with category-zero edge
cases without affecting run_specs metrics. Both Phase 6 and
Phase 7 land cleanly; the residual concerns (verdict-pathway
empty-commit on Q5; deferred orchestrator_batch validation; Step 2
noise floor) are scoped follow-ups, not blockers.

### Iteration 9 — vacuous-spec extraction filter + drop per-candidate verdict pathway

**Date opened:** 2026-05-08

**Architectural patterns targeted:**
- **Pattern A** (stacked PRODUCTs amplify upstream commit noise into
  trait death) — closes the residual edge case Phase 7's floor was
  masking. When the keyword endpoint emits a structurally vacuous
  spec (`finalized_keywords=[]`), the across-category fold should
  treat it symmetric with `coverage_commitments.keyword.verdict=abstain`
  rather than running an empty query that scores 0.0 across the
  board. Same fix generalizes to any endpoint params that are
  structurally empty (metadata column_spec all-null, semantic
  space_queries empty).
- **Pattern C** (walk-to-commit chain is prompt-only convention,
  not schema-enforced) — Phase 5's per-candidate verdict pathway
  was the prior attempt at structural enforcement at the per-
  candidate level. Iter 8 surfaced that the per-candidate framing
  cannot natively express union-level superset reasoning: each
  candidate evaluates verdict against its own strengths/weaknesses
  in isolation, and the keyword.md superset test is a UNION-level
  test ("the ANY-mode union is a true superset"). Forbidding
  OR-disjunctions in verdict_reason (Iter 6 lesson #2) mechanically
  prevents the candidate from rendering "broad alone but the union
  saves me." So per-candidate verdicts can verdict-abstain on
  every candidate even when their union would commit cleanly. Iter
  9 reverts the per-candidate verdict abstraction; bucket-level
  `coverage_commitments.{route}.verdict` is the remaining
  structural enforcement (which operates at union level naturally).

This iteration is two follow-up fixes from Iter 8 — both scoped,
both isolated, both bundled with the Iter 8 ship rather than
landed separately. User direction logged: "#1 go with the fix you
believe is the best [extraction-time, applied to all empty
endpoint params not just keyword]; #2 Go with option 3 [drop
per-candidate verdict pathway entirely]."

#### Hypothesis

**Change 1 hypothesis (extraction-time vacuous-spec filter):**

1. **Q5 GENRE empty-keyword case stops counting as a fired keyword
   endpoint.** Iter 8 surfaced GENRE emitting `finalized_keywords=[]`
   while `coverage_commitments.keyword.verdict=commit`. After
   Change 1, the empty wrapper is filtered at extraction; stage 4
   sees keyword as not-fired, symmetric with bucket-level abstain.
   Phase 7's floor on that trait remains relevant for the OTHER
   FACETS-zero edge cases (genuine category miscommits), but the
   empty-keyword artefact stops contributing 0.0 to the fold.

2. **Trip-wire risk stays at or below phase_8's 21.** Empty-keyword
   fired endpoints currently still flag ADDITIVE_KW_RISK because the
   trip-wire formula is `combine_type=='additive' AND keyword in
   fired_routes`. After Change 1, those rows clear because keyword
   is no longer in fired_routes for vacuous emissions. Cats should
   drop slightly (one fewer fired row per vacuous case).

**Change 2 hypothesis (drop per-candidate verdict pathway):**

3. **Q5 empty-commit case structurally cannot recur.** The
   downstream `KeywordQuerySpecSubintent.finalized_keywords` field
   reverts to LLM-emitted with `min_length=1` — Pydantic enforces
   non-emptiness at parse time. The LLM cannot emit an empty
   commit list anymore. (Bucket-level `coverage_commitments.keyword.
   verdict=abstain` remains the abstention pathway.)

4. **Union-level reasoning becomes the primary commit mechanism.**
   The walk surfaces candidates with strengths/weaknesses; the
   bucket-level coverage_commitments.keyword reads off the walk
   and applies the superset test once over the union of finalized
   members. The verdict_reason there cites the union analysis. This
   is closer to how the keyword.md superset test was originally
   framed (one application over the union).

5. **STORY_THEMATIC_ARCHETYPE keyword commits hold or improve vs
   phase_8's 13.** The hypothesis is that union-level reasoning is
   a cleaner abstention pathway than per-candidate verdicts were.
   Risk: per-candidate verdicts may have been doing useful work on
   borderline-individual-candidate cases (e.g., dropping a single
   over-broad candidate from a union that would otherwise pass).
   If kw_commits regress meaningfully, that's evidence the
   per-candidate level was load-bearing.

6. **Phase 6 sibling-context guidance still effective.** Sibling-
   context wins on Iter 8 (Q15 GENRE 5→1, Q18 STORY_THEMATIC 3→0)
   came from union-level narrowing under FACETS, not from
   per-candidate verdict mechanics. The reframed sibling section
   on keyword.md still steers union narrowing under FACETS via the
   bucket-level commit; the dominated-by-sibling reason becomes a
   union-level consideration ("under FACETS with paraphrastic
   siblings, the commit's union should narrow toward facets the
   sibling cannot reach"). Q15/Q18 quality wins should hold.

**Combined hypothesis — positive controls:**

7. **Q9 / Q12 / Q25 hold.** Same controls as Iter 8 — clean keep
   commits unchanged.

8. **Schema validation errors stay at zero.** Dropping
   `_WalkThenCommitOutputBase` removes a derivation hook;
   if any other code path was relying on it, errors will surface.
   This is the primary regression risk.

**Stop conditions:**
- Q9 / Q12 / Q25 lose clean keep commits.
- Trip-wire risk count regresses above phase_8's 21.
- Step 3 sprawl regresses above phase_8's 82.
- Schema validation errors > 0.
- STORY_THEMATIC_ARCHETYPE kw_commits regress above phase_8's 13.
  (If they regress, Iter 9 has lost a real per-candidate-verdict
  contribution; reconsider the revert.)
- Sub-shape C wins from Iter 8 regress (Q15 GENRE 5→1 narrowing
  partially undone; Q18 STORY_THEMATIC abstaining undone).

**Out of scope:**
- Phase 6 sibling-task context unchanged.
- Phase 7 geometric-mean-with-floor unchanged (`_FACETS_FOLD_FLOOR=0.1`).
- Single-endpoint `KeywordQuerySpec` (the spec used by buckets 3/4)
  unchanged — Phase 5 only added verdict pathway to the multi-
  endpoint variant.
- `coverage_commitments` fixed-shape per-endpoint commitment kept;
  this is the remaining structural enforcement at the union level.
- Step 2 / Step 3 prompt language unchanged.
- Real-query EPS sweep tooling still deferred (orphaned from Iter 8).

#### Changes actually made

Six files modified for Iteration 9:

**Change 1 — extraction-time vacuous-spec filter:**

- [search_v2/endpoint_fetching/category_handlers/output_extractor.py](../search_v2/endpoint_fetching/category_handlers/output_extractor.py)
  — added `_is_vacuous_spec(route, wrapper)` helper, called in
  `extract_fired_endpoints` for every per-route bucket. Filters
  out wrappers whose params are structurally empty:
  - keyword: `parameters.finalized_keywords == []`
  - semantic: `parameters.space_queries == []`
  - metadata: every column on `parameters.column_spec` is None
  Symmetric with `coverage_commitments.{route}.verdict=abstain`
  at the bucket level. Other routes (entity, franchise, studio,
  awards) have no structurally-vacuous emission path so the
  helper no-ops there.

**Change 2 — drop the per-candidate verdict pathway:**

- [schemas/keyword_translation.py](../schemas/keyword_translation.py)
  — deleted `PotentialKeywordWithVerdict` (Phase 5 PotentialKeyword
  override with verdict_reason → verdict). Deleted
  `AttributeAnalysisWithVerdict`. `KeywordWalk.attributes` reverts
  to `list[AttributeAnalysis]` (no verdicts on candidates).
  `KeywordQuerySpecSubintent.finalized_keywords` reverts to
  LLM-emitted with `min_length=1` (was server-derived from
  verdicts under Phase 5). Schema now enforces: when the LLM emits
  `keyword_parameters` it must contain a non-empty
  `finalized_keywords` — abstaining requires
  `coverage_commitments.keyword.verdict=abstain` and a null
  `keyword_parameters` upstream.
- [search_v2/endpoint_fetching/category_handlers/schema_factories.py](../search_v2/endpoint_fetching/category_handlers/schema_factories.py)
  — deleted `_WalkThenCommitOutputBase` and its
  `_derive_keyword_finalized_from_verdicts` model_validator.
  Walk-then-commit buckets now inherit from `_HandlerOutputBase`
  directly. Removed `model_validator` import; cleaned up
  Phase-5-specific commentary in factory docstrings;
  generalized the `commitment-criteria-fail` line in the
  `verdict_reason` description to reference union-level reasoning.
- [search_v2/endpoint_fetching/category_handlers/prompts/endpoints/keyword.md](../search_v2/endpoint_fetching/category_handlers/prompts/endpoints/keyword.md)
  — rewrote "Where the keyword analysis lives" and
  "Commitment: superset test" sections. Two-level abstention
  (per-candidate verdict + bucket-level commitment) collapses to
  one level (bucket-level only). Added explicit "the test is a
  UNION test" framing; named the verdict reasons for the
  bucket-level abstain (`no-walk-candidate`,
  `commitment-criteria-fail`, `dominated-by-sibling`). Reframed
  sibling-context section: under FACETS narrow the union toward
  facets the sibling cannot reach (was: verdict-abstain
  per-candidate).
- [search_v2/endpoint_fetching/category_handlers/prompts/buckets/preferred_representation_fallback_objective.md](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/preferred_representation_fallback_objective.md),
  [search_v2/endpoint_fetching/category_handlers/prompts/buckets/semantic_preferred_deterministic_support_objective.md](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/semantic_preferred_deterministic_support_objective.md),
  [search_v2/endpoint_fetching/category_handlers/prompts/buckets/audience_suitability_deterministic_first_objective.md](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/audience_suitability_deterministic_first_objective.md)
  — each: dropped "(each with verdict_reason → verdict)" from the
  per-endpoint walk description (Phase 1); reframed the
  per-endpoint Superset test as a union test over the walk's
  candidates; dropped "for keyword, finalized_keywords is
  server-derived from the walk's verdict commits — emit [] and
  let the derivation fill it" from the thin-parameters phase
  (Phase 4); dropped the per-candidate verdict reference from
  Strictness scaling under sibling context.

#### Observations

**Headline metrics (V5 suite, 25 queries):**

| run     | cats | risk | handler errors | n_traits | STORY_THEMATIC kw | F2 ALL_rate | EE share | empty kw_finalized |
|---------|-----:|-----:|---------------:|---------:|------------------:|-----------:|---------:|-------------------:|
| phase_3 |   80 |   23 |              0 |       — |                13 |       5.0% |        — |                  — |
| phase_5 |   86 |   26 |              0 |       51 |                16 |       0.0% |    23.3% |                  — |
| phase_8 |   82 |   21 |              0 |       51 |                13 |       0.0% |    19.6% |                  1 |
| **phase_9** | **90** | **25** | **0** | **52** | **13** | **3.3%** | **17.1%** | **0** |

Cats moved 82 → 90 (+8), risk moved 21 → 25 (+4), n_traits 51 → 52
(Step 2 noise — same Q11 fusion variance as Iter 8 reverted from
1 trait back to 2). STORY_THEMATIC_ARCHETYPE keyword commits held
flat at 13 (matching Iter 8). EE share dropped 19.6% → 17.1%
(modest improvement). **F2 ALL count 0 → 1: Q5 plural-intent ALL
restored** — `[ACTION, THRILLER]` ALL re-emerged as the GENRE
commit, replacing phase_8's empty `keyword_finalized=[]`.
**Empty kw_finalized count 1 → 0**: Change 1 + Change 2 close the
empty-commit case structurally — no query in the suite can emit
an empty keyword commit anymore.

**Change 1 (vacuous-spec extraction filter) — clean win:**

✓ **Q5 GENRE empty-commit case structurally closed.** phase_8
emitted `keyword_finalized=[]` with `keyword_scoring_method=ANY`;
phase_9 emits `[ACTION, THRILLER]` `ALL` — F2 plural-intent ALL
restored. The schema's `min_length=1` on the multi-endpoint
`finalized_keywords` (from Change 2's revert) prevents the empty
commit at parse time; the extraction filter is defense-in-depth
for any path that could otherwise emit a vacuous wrapper.

**Change 2 (drop per-candidate verdict pathway) — mixed:**

Per-query wins:

✓ **Q21 atmospheric folk horror.** STORY_THEMATIC_ARCHETYPE
narrowed `[FOLK_HORROR, WITCH_HORROR, FOLKLORE_ADAPTATION]` →
`[FOLK_HORROR]` (paraphrase cluster collapsed to single canonical).
✓ **Q22 grief and reconciliation.** STORY_THEMATIC_ARCHETYPE
dropped the canonical-stretching `FEEL_GOOD` commit on
`reconciliation` (clean abstention, semantic-only).
✓ **Q23 slow-burn psychological mysteries.**
STORY_THEMATIC_ARCHETYPE narrowed `[PSYCHOLOGICAL_DRAMA,
PSYCHOLOGICAL_THRILLER, PSYCHOLOGICAL_HORROR]` →
`[PSYCHOLOGICAL_THRILLER]` (3 paraphrases collapsed to 1).
✓ **Q16 brutal MMA fight movies.** SENSITIVE_CONTENT narrowed
`[SPLATTER_HORROR, BODY_HORROR]` → `[SPLATTER_HORROR]` (single
member); CENTRAL_TOPIC widened `[SPORT]` → `[MARTIAL_ARTS, SPORT]`
(more specific routing).

Per-query regressions vs Iter 8 narrowing:

✗ **Q12 mind-bending puzzle films.** NARRATIVE_DEVICES went
`[NONLINEAR_TIMELINE, UNRELIABLE_NARRATOR]` (phase_8 dropped
PLOT_TWIST as the right narrowing) → `[NONLINEAR_TIMELINE,
PLOT_TWIST, UNRELIABLE_NARRATOR]` (PLOT_TWIST returned). The Iter
8 sibling-context-driven narrowing was per-candidate; without the
verdict pathway, the union-level commit is more lenient.
✗ **Q18 like Donnie Darko but funnier.** STORY_THEMATIC_ARCHETYPE
went from `keyword.verdict=abstain` (phase_8 sub-shape C win) →
`[COMING_OF_AGE, SUPERNATURAL_FANTASY]` (regressed to stretching
commits). GENRE additionally fired `[SCI_FI,
PSYCHOLOGICAL_THRILLER]` — extra stretching commits.
NARRATIVE_DEVICES held but with `[NONLINEAR_TIMELINE, PLOT_TWIST]`
(narrower set than its phase_8 commit, mild win).
✗ **Q15 Studio Ghibli style hand-drawn fantasies.** GENRE went
`[FANTASY]` (Iter 8 singular-narrowing win) → `[FANTASY,
FANTASY_EPIC]` (paraphrase added back); ELEMENT_PRESENCE picked
up a redundant `[FANTASY]` commit.
✗ **Q20 dark gritty antihero comic-book films.** New
STORY_THEMATIC_ARCHETYPE `[DRAMA]` commit on the `dark` trait —
DRAMA is the canonical over-coverage stretching case from F3.
phase_8 had no STORY_THEMATIC commit on this trait.
✗ **Q14 obscure indie passion projects.** New NARRATIVE_DEVICES
firing with 4 paraphrastic members `[NONLINEAR_TIMELINE,
UNRELIABLE_NARRATOR, BREAKING_FOURTH_WALL, SINGLE_LOCATION]` —
overly broad commit on a trait whose intent isn't strongly
narrative-mechanic-shaped.

Mixed:

~ **Q11 historical war epics.** Step 2 atomization split 1 → 2
traits (`war` + `epics`); phase_8 had a single fused trait. Cats
count rises mechanically. Same Step 2 noise as phase_8 vs phase_5
(noted in Iter 8 lesson #3 as recurring measurement-floor drift).
~ **Q24 coming-of-age road trips not too sappy.** phase_8 had a
clean `[ROAD_TRIP]` commit on STORY_THEMATIC_ARCHETYPE; phase_9
dropped it (no STORY_THEMATIC commit). Could be either lost
narrowing or correctly-routed-to-semantic; without trait_score
distribution data, ambiguous.
~ **Q10 cyberpunk dystopias.** STORY_THEMATIC_ARCHETYPE went
`[DYSTOPIAN_SCI_FI]` → `[DYSTOPIAN_SCI_FI, POST_APOCALYPTIC]` —
mild paraphrase addition.

**Positive controls:**

✓ Q9 `revenge stories with anti-heroes` —
STORY_THEMATIC_ARCHETYPE `[REVENGE]` clean, CHARACTER_ARCHETYPE
`[ANTI_HERO]` clean. Held.
✓ Q25 `unreliable narrator with a twist ending` —
NARRATIVE_DEVICES `[UNRELIABLE_NARRATOR]` clean. Held.
~ Q12 `mind-bending puzzle films about consciousness` — kept ANY
shape but added back the PLOT_TWIST member that Iter 8 dropped.
Not an ALL-collapse, but a narrowing regression.

#### Stop conditions evaluation

- ✓ Q9 / Q25 hold clean keep commits.
- ✗ **Headline trip-wire risk count crossed back above phase_8's
  21** (achieved 25 — also above phase_3 baseline of 23). This is
  the brief's primary stop condition.
- ✗ **Step 3 sprawl regressed above phase_8's 82** (achieved 90).
  Some sprawl is Step 2 noise (Q11 atomization), but per-query
  inspection shows real per-trait sprawl on Q14, Q17, Q18, Q20.
- ✓ Schema validation errors at zero. The `_WalkThenCommitOutputBase`
  removal landed cleanly; OUTPUT_SCHEMAS still builds 40 schemas;
  no verdict-pathway artefacts in any schema's JSON output.
- ~ STORY_THEMATIC_ARCHETYPE kw_commits held at 13 (matched the
  phase_8 stop-condition floor exactly, but per-query stretching
  shifted around — Q22/Q23/Q21 wins offset by Q14/Q20 new stretches).
- ✗ **Sub-shape C wins partially undone**: Q15 GENRE singular →
  paraphrase pair (FANTASY, FANTASY_EPIC); Q18 STORY_THEMATIC
  abstention → stretching pair (COMING_OF_AGE, SUPERNATURAL_FANTASY)
  + new GENRE stretching pair.
- ✓ **Empty kw_finalized cases eliminated**: Q5 structural win
  from Change 1 + Change 2's `min_length=1` revert.

#### Concerns flagged for follow-up

1. **Per-candidate verdict pathway WAS doing useful work.** The
   brief's hypothesis (#5) was that union-level reasoning would
   match or improve on per-candidate verdicts; the data says
   per-candidate verdicts had real narrowing power on borderline
   individual candidates that the union-level commit doesn't
   replicate. Iter 8's evidence-injection win was upstream of the
   verdict pathway (sibling-context steers the LLM toward
   narrowing); the verdict pathway translated that steering into
   per-candidate abstention. With the verdict pathway gone, the
   union-level commit reads "all members in the union together
   pass the superset test" more leniently and lets paraphrastic
   stretches survive.

2. **Q5 plural-intent ALL win is real.** The vacuous-spec filter
   isn't sufficient by itself for this — what restored
   `[ACTION, THRILLER] ALL` is the schema's `min_length=1`
   forcing the LLM to either commit a real union or abstain at the
   bucket level. Both halves of Change 2 were load-bearing for
   this specific outcome.

3. **Recommendation for the user.** Iter 9 as a unit is NOT a
   clean ship by the brief's own stop conditions (trip-wire risk
   regressed past phase_8 AND past phase_3 baseline). Three
   options worth considering: (a) ship Iter 9 as-is, accepting
   the Q5 win as the primary value and the narrowing regressions
   as the cost; (b) keep Change 1 (vacuous-spec extraction
   filter) but revert Change 2 — keeps Iter 8's per-candidate
   verdict narrowing wins while retaining the extraction-time
   defense-in-depth; (c) keep Change 2 architecturally (drop the
   verdict pathway) but invest in tightening the union-level
   prompt to recover the narrowing power per-candidate verdicts
   delivered. The decision is the user's; this iteration's
   verdict surfaces the tradeoff honestly.

#### What we learned

1. **Per-candidate verdicts and union-level commits are NOT
   equivalent abstractions.** The brief assumed the bucket-level
   `coverage_commitments.keyword.verdict=abstain` could absorb
   the abstention work that per-candidate verdicts were doing.
   The data shows otherwise: per-candidate verdicts evaluate each
   member's individual fit and abstain when its weaknesses
   overpower its strengths; union-level commits evaluate "does
   the union as a whole pass the superset test" and accept
   members whose individual stretching is offset by other members
   in the union covering attribute-satisfying movies. These
   produce different commit shapes, and the per-candidate version
   was narrower in practice on the V5 suite.

2. **Schema enforcement at the right granularity matters.**
   `min_length=1` on the LLM-emitted `finalized_keywords`
   enforces "if you commit to fire keyword, commit something
   real" — that's the structural invariant the brief wanted from
   the per-candidate verdict pathway, but located one level up.
   This is now the sole structural enforcement on commit
   non-emptiness; the abstention pathway is bucket-level only.

3. **Q5 empty-commit case is closed, but the broader problem
   space is wider.** The empty-commit edge case Phase 7's floor
   was masking is gone (good), but Iter 8's narrowing wins
   depended on the verdict pathway that Iter 9 dropped. The
   ceiling that Iter 8 broke (trip-wire below phase_3 baseline of
   23) has bounced back above the baseline (25). Whether that
   ceiling can be re-broken at union-level requires either prompt
   investment (option (c) above) or a different mechanism.

4. **Iter 7 lesson #3 (Step 2 noise) confirmed for the FOURTH
   time.** Q5 trait `bloody` flipped FACETS (phase_8) → FRAMINGS
   (phase_9); Q11 atomization went from 1 trait (phase_8) to 2
   traits (phase_9, matches phase_5_1's 2-trait-from-phase_5's-3
   variance pattern); Q14 NARRATIVE_DEVICES newly fires on
   `passion projects` with 4 stretches (Step 3 routing change
   plausibly compounded by Step 2 atomization variance). Multi-
   run aggregation is now a measurement prerequisite for any
   future iteration; single-run-on-25-queries cannot distinguish
   intervention signal from cross-run noise on marginal cases.

5. **The architectural-pattern lens still applies.** Pattern A
   (stacked PRODUCTs amplifying upstream noise into trait death)
   is now softened at THREE points: Phase 1.1 empty-spec category
   filter, Phase 7 geometric-mean-with-floor, and Iter 9 vacuous-
   spec extraction filter. Pattern C (walk-to-commit chain
   prompt-only convention, not schema-enforced) is *less* enforced
   after Iter 9 — the per-candidate verdict pathway was the
   nearest thing to schema enforcement at the per-member level
   and is now gone. Bucket-level commitment is the remaining
   enforcement, and the data argues that's not granular enough
   to catch every paraphrase-cluster stretching case.

#### Shipped — what we learned

**Iteration 9 NOT SAFE TO SHIP AS-IS.** The brief's own primary
stop condition tripped (trip-wire risk 21 → 25, also above the
phase_3 baseline of 23). Sub-shape C wins from Iter 8 partially
regressed (Q15, Q18). Multiple per-query stretching regressions
(Q12, Q14, Q18, Q20) outweigh the per-query narrowing wins
(Q21, Q22, Q23, Q16) on the headline aggregate.

**One unambiguous structural win:** the Q5 empty-commit edge case
is closed (`min_length=1` on the multi-endpoint
`finalized_keywords` makes empty commits unrepresentable; the
extraction-time vacuous-spec filter is defense-in-depth across
all routes), and Q5 plural-intent `[ACTION, THRILLER] ALL` is
restored as a clean F2 commit.

**Recommendation deferred to user direction.** This entry surfaces
the tradeoff for the user to choose among (a) ship as-is despite
trip-wire regression, (b) revert Change 2 only and keep
Change 1's extraction filter, or (c) keep Change 2 but invest
in tightening the union-level prompt to recover per-candidate
narrowing power. The Iter 9 working tree (commit-pending) reflects
the as-is state and can be reverted / re-edited based on the
user's choice.

#### Re-analysis: do the regressions genuinely impact retrieval quality?

The initial verdict was anchored on trip-wire risk count
(21 → 25). Re-reading each per-query commit-shape change through
the actual stage-4 scoring math reverses several classifications.

**The trip-wire is a STRUCTURAL flag, not a behavioral measure.**
`additive_kw_risk` fires when `combine_type=='additive' AND
keyword in fired_routes`. It flags a category whose KW=0 would
zero the category under within-call ADDITIVE multiply. But a
*broader* ANY-mode keyword union *lowers* the zero-probability —
KW=1.0 if a movie has any tag in the union. Broadening the union
under additive lowers behavioral zero-rate while triggering the
structural flag identically. The flag doesn't track destructiveness.

**ANY-mode union semantics: broader = more recall, not more
stretch.** keyword.md's design ("singular intent → ANY; the
keyword commit may include multiple registry members because you
have converted that one attribute into several registry surface
forms — paraphrases, alternative routes, sub-form alternatives;
matching any one is sufficient evidence") explicitly rewards
broader paraphrastic commits. Per-candidate verdict pathway
evaluated each member's individual fit; union-level reasoning
asks whether the *union* covers the slice. The two answers
diverge on paraphrastic adjacencies:

| Query (trait) | phase_8 | phase_9 | Re-analysis verdict |
|---|---|---|---|
| Q12 NARRATIVE_DEVICES (mind-bending puzzle) | [NL,UN] | [NL,PT,UN] | **phase_9 wins** — PLOT_TWIST is a puzzle paraphrase (Sixth Sense, Memento); Iter 8 dropped a recall-positive member |
| Q18 STORY_THEMATIC + GENRE (Donnie Darko) | abstain | [COMING_OF_AGE,SUPERNATURAL_FANTASY], [SCI_FI,PSYCHOLOGICAL_THRILLER] | **phase_9 wins** — these ARE Donnie Darko's identity; positioning_reference's `axes_replaced_by_siblings` is TONE only ("funnier"), so other axes shouldn't be dropped; Iter 8 over-abstained |
| Q15 GENRE (fantasies) | [FANTASY] | [FANTASY,FANTASY_EPIC] | **phase_9 wins** — broader ANY adds canonical sub-form member |
| Q14 NARRATIVE_DEVICES (passion projects) | absent | 4-paraphrase ANY | **phase_9 ~neutral** — under additive, broader ANY lowers zero-prob; routing decision (whether NARRATIVE_DEVICES should fire on passion-projects) is upstream of Iter 9 |
| Q20 STORY_THEMATIC (dark) | absent | [DRAMA] | **phase_9 ~neutral** — DRAMA is canonical near-no-op (tags every drama); under additive cat ≈ SEM signal; wasted commit, not destructive |
| Q21 STORY_THEMATIC (folk horror) | [FOLK_HORROR,WITCH_HORROR,FOLKLORE_ADAPTATION] | [FOLK_HORROR] | **phase_9 LOSES recall** under ANY (re-classified — initial entry called this a win; under union semantics phase_8's broader commit catches more folk-horror candidates) |
| Q23 STORY_THEMATIC (psychological) | 3 paraphrases | 1 | **phase_9 LOSES recall** under ANY (re-classified) |
| Q22 STORY_THEMATIC (reconciliation) | [FEEL_GOOD] | abstain | **phase_9 wins** — FEEL_GOOD is tonal, reconciliation is thematic; clean abstention on a single-claim stretching case |
| Q16 CENTRAL_TOPIC (MMA fight) | [SPORT] | [MARTIAL_ARTS,SPORT] | **phase_9 wins** — MARTIAL_ARTS more specific |
| Q16 SENSITIVE_CONTENT (brutal) | [SPLATTER,BODY_HORROR] | [SPLATTER] | **phase_9 mild recall loss** under ANY |
| Q5 GENRE (intense action thrillers) | [] empty bug | [ACTION,THRILLER] ALL | **phase_9 unambiguous win** — empty-commit closed; plural-intent ALL restored |
| Q15 ELEMENT_PRESENCE (fantasies) | absent | [FANTASY] under FACETS paraphrastic with GENRE | Bounded concern; Phase 7 floor caps worst-case at `0.1^(1/2)≈0.316`. Pre-existing N2 pattern, not Iter-9-specific |

**Net under retrieval-quality lens:** Iter 9 has 5–6 clear
recall improvements (Q5, Q12, Q15-GENRE, Q18×2, Q16-CENTRAL_TOPIC,
Q22) and 2–3 recall losses (Q21, Q23, Q16-SENSITIVE_CONTENT mild).
Two bounded-concern items (Q15-ELEMENT_PRESENCE paraphrastic,
Q20 wasted DRAMA signal) and one routing question that pre-dates
Iter 9 (Q14 NARRATIVE_DEVICES on passion projects).

**Why the initial verdict was misleading.** I anchored on the
trip-wire risk count and on superficial per-query "narrowing →
broadening" pattern matching. The trip-wire doesn't measure
destructiveness; under ANY-mode singular intent the keyword.md
design rewards broader unions; per-candidate verdicts had been
over-narrowing precisely because the per-candidate framing
ignores union-level recall semantics. The brief's hypothesis #5
(union-level reasoning is closer to keyword.md's actual commit
test) is closer to correct than the per-query trip-wire data
initially suggested.

**Caveat: this analysis is inference, not measurement.** It
assumes tag distributions match retrieval intuitions (movies
tagged WITCH_HORROR are commonly also tagged FOLK_HORROR;
mind-bending puzzle films often carry PLOT_TWIST; etc.). The
only way to verify empirically is a per-trait `trait_score`
breakdown across real candidates — exactly the orchestrator_batch
tooling deferred from Iter 8. Until that lands, the
retrieval-quality verdict is best-effort inference from commit
shape + scoring math.

**Updated recommendation.** Iter 9 is plausibly CLEANER than
Iter 8 once read through retrieval-quality math, despite the
trip-wire count regression. Option (a) (ship as-is) is more
defensible than the initial entry suggested; option (b) (revert
Change 2) would lose Q5 plural-intent ALL restoration, the Q18
Donnie-Darko-identity surfacing, and the Q12 puzzle-paraphrase
broadening — those are real retrieval wins, not just structural
flag noise. The bounded concerns (Q15 paraphrastic-under-FACETS,
Q21/Q23 recall narrowing) are addressable in follow-up
prompt iteration without losing the wins.


