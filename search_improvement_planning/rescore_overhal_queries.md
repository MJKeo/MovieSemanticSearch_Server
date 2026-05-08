# Rescore Overhaul — Verification Query Suite

A 25-query test suite designed to confirm the V5 changes (D1–D5)
are improving behavior at each implementation phase. Run before
every phase, after every phase, and diff the per-query output.

## How to use

Driver: [search_v2/run_specs.py](../search_v2/run_specs.py).

**Do NOT pass this markdown file directly to `--suite`.**
`_load_suite` only strips `#`-prefixed and blank lines, so the
prose between sections (and code-block contents) would be
dispatched as queries — once burned ~314 prose lines through the
pipeline before being caught. Always run from the canonical
plain-text extract `/tmp/v5_suite.txt` (one query per line, no
markdown), or extract the queries fresh from the
`## Query list (one per line)` section at the bottom of this
document.

```bash
# baseline before any V5 work
python -m search_v2.run_specs --suite /tmp/v5_suite.txt \
  --json /tmp/run_specs_baseline.json

# after each phase ships
python -m search_v2.run_specs --suite /tmp/v5_suite.txt \
  --json /tmp/run_specs_phase_<N>.json
```

The "what to look for" notes below describe the diagnostic shape we
want — they are intentionally not pass/fail thresholds. Compare the
per-query JSON output across runs and confirm the directional
changes hold. LLM non-determinism means individual keyword
commitments will drift; what we are validating is the *shape* of
the per-(trait, category) commit.

---

## Section 1 — REMOVE-KW categories (validates D1 / Phase 2b)

Each category in the REMOVE-KW set
(`SEASONAL_HOLIDAY`, `EMOTIONAL_EXPERIENTIAL`,
`SPECIFIC_PRAISE_CRITICISM`) gets a query whose primary trait
routes there. Pre-fix: KW endpoint fires, brittle commit zeros
ADDITIVE category on KW miss. Post-fix: KW removed, single SEM
endpoint under `SEMANTIC_PREFERRED_DETERMINISTIC_SUPPORT` →
`SINGLE_NON_METADATA_ENDPOINT` with combine_type `SINGLE`.

### Q1. `heartwarming holiday films`
- **Hits:** SEASONAL_HOLIDAY (primary), EMOTIONAL_EXPERIENTIAL
- **Pre-fix expectation:** KW fires `[CHRISTMAS_MOVIE]` or
  `[FEEL_GOOD]`, ADDITIVE multiply, KW miss zeros category.
- **Post-fix:** No KEYWORD route in fired endpoints for either
  category; combine_type=SINGLE; bucket=SINGLE_NON_METADATA_ENDPOINT;
  SEM space targets CTX `watch_scenarios` + RCP/VWX prose.
- **Risk flag:** ADDITIVE_KW_RISK should drop to 0 for both
  categories.

### Q2. `films with a bittersweet melancholic tone`
- **Hits:** EMOTIONAL_EXPERIENTIAL
- **Pre-fix:** KW often fires `[BITTERSWEET_ENDING]` ANY — endings-
  specific tag mismatched against tone query.
- **Post-fix:** Pure SEM via VWX `emotional_journey` /
  `tonal_register`; trait_score driven by gradient, not gate.

### Q3. `forgotten gems with brilliant performances`
- **Hits:** CULTURAL_STATUS (regression — no KW already),
  SPECIFIC_PRAISE_CRITICISM
- **Pre-fix:** SPECIFIC_PRAISE_CRITICISM fires KW like `[DRAMA]` or
  `[INDIE]` ANY — over-broad noise.
- **Post-fix:** SPECIFIC_PRAISE_CRITICISM single-endpoint SEM on
  RCP `praised_qualities`; CULTURAL_STATUS unchanged
  (combine_type=ADDITIVE on SEM × META). No regression on
  CULTURAL_STATUS commit shape.

---

## Section 2 — ALTERNATIVES categories (validates D1 / Phase 2a)

`TARGET_AUDIENCE` and `SENSITIVE_CONTENT` keep KW but flip
combine_type from ADDITIVE to ALTERNATIVES so KW miss does not
zero the category when META/SEM carry the signal.

### Q4. `wholesome family movie night picks`
- **Hits:** TARGET_AUDIENCE (primary), maybe VIEWING_OCCASION
- **Pre-fix:** TARGET_AUDIENCE ADDITIVE(KW × META × SEM); a movie
  rated PG with strong CTX `watch_scenarios` for family but lacking
  `FAMILY_MOVIE` keyword tag scores 0.
- **Post-fix:** combine_type=ALTERNATIVES; KW miss is non-fatal;
  META.maturity_rank or CTX `watch_scenarios` either carries the
  category alone via MAX.

### Q5. `intense action thrillers but not too bloody`
- **Hits:** GENRE (action thrillers), SENSITIVE_CONTENT (negative
  trait: too bloody), maybe CHARACTER_ARCHETYPE
- **Pre-fix:** SENSITIVE_CONTENT positive variants fire KW × META ×
  SEM ADDITIVE.
- **Post-fix:** SENSITIVE_CONTENT (when fired positively elsewhere)
  shows combine_type=ALTERNATIVES. The negative trait path is
  unchanged by V5 — sanity-check that no negative-trait regression
  appears.

---

## Section 3 — Superset test for KEEP-ADDITIVE categories (validates D2)

KEEP-with-abstention categories (`CENTRAL_TOPIC`,
`ELEMENT_PRESENCE`, `STORY_THEMATIC_ARCHETYPE`) and KEEP-unchanged
categories (`CHARACTER_ARCHETYPE`, `NARRATIVE_DEVICES`) should
commit KW only when registry coverage forms a true superset.

### Q6. `movies featuring elephants`
- **Hits:** ELEMENT_PRESENCE
- **Pre-fix:** likely commits `[ANIMAL_ADVENTURE]` ANY or worse —
  pulls all animal movies; or commits nothing useful; or commits
  with stretching.
- **Post-fix:** Registry has no ELEPHANT-specific member ⇒ KW
  abstains (gap → fail superset). SEM on P-EVT / P-AN carries the
  motif. Bucket prompt's new partial-abstention sanction is what
  enables this.

### Q7. `movies about marathons`
- **Hits:** CENTRAL_TOPIC, ELEMENT_PRESENCE
- **Pre-fix:** likely commits `[SPORT, BIOGRAPHY, TRUE_STORY]`
  ALL — over-pull + ALL-strict.
- **Post-fix:** SPORT alone is acceptable per superset (over-pull
  is allowed when SEM refines on the SPORT semantic space + plot
  events). Scoring_method should be ANY (singular intent: marathon
  is one attribute). BIOGRAPHY/TRUE_STORY drop unless the user
  explicitly demanded a true story.

### Q8. `films with sentient AI`
- **Hits:** ELEMENT_PRESENCE, CENTRAL_TOPIC
- **Pre-fix:** likely commits `[ROBOT, ARTIFICIAL_INTELLIGENCE]` if
  registry has these.
- **Post-fix:** If registry has both → ANY commit clean superset.
  If only ROBOT exists, superset still passes (ROBOT covers most
  sentient-AI films). Verify ANY scoring per D3.

### Q9. `revenge stories with anti-heroes`
- **Hits:** STORY_THEMATIC_ARCHETYPE (revenge), CHARACTER_ARCHETYPE
  (anti-hero)
- **Pre-fix:** typically commits clean — `[REVENGE]` and
  `[ANTI_HERO]`. Used as a positive-control / regression check.
- **Post-fix:** Clean ADDITIVE(KW × SEM) commit unchanged for both
  categories. CHARACTER_ARCHETYPE is in KEEP-unchanged. If
  trait_score drops or combine shape shifts, that is a regression.

---

## Section 4 — Singular vs plural intent for ANY/ALL (validates D3)

Targets the F2 ALL-on-paraphrase failure. Rewrite of the
ANY/ALL discriminator should make singular-intent expressions
default to ANY even when multiple registry members are committed.

### Q10. `cyberpunk dystopias`
- **Hits:** STORY_THEMATIC_ARCHETYPE / GENRE
- **Pre-fix:** commits `[DYSTOPIAN_SCI_FI, POST_APOCALYPTIC]` ALL —
  paraphrastic adjacent neighbors treated as compound.
- **Post-fix:** scoring_method=ANY (one expression with multiple
  paraphrastic registry members). ALL would require the user to
  have named two distinct attributes.

### Q11. `historical war epics`
- **Hits:** CENTRAL_TOPIC (war), maybe RELEASE_DATE-adjacent or
  GENRE (epics)
- **Pre-fix:** commits `[WAR, HISTORY]` ALL on CENTRAL_TOPIC.
- **Post-fix:** scoring_method=ANY for the WAR/HISTORY commit
  (singular intent: war epic is one attribute named with
  paraphrastic registry homes).

### Q12. `mind-bending puzzle films about consciousness`
- **Hits:** NARRATIVE_DEVICES (mind-bending), CENTRAL_TOPIC
  (consciousness)
- **Pre-fix:** NARRATIVE_DEVICES commits `[NONLINEAR_TIMELINE,
  UNRELIABLE_NARRATOR, PLOT_TWIST]` ALL — alternative devices, not
  conjunction.
- **Post-fix:** NARRATIVE_DEVICES scoring_method=ANY; CENTRAL_TOPIC
  KW abstains for "consciousness" (registry has no member that
  forms a clean superset) and SEM carries the topic.

### Q13. `comedy musicals about teenage romance`
- **Hits:** GENRE (multi-genre conjunction), CENTRAL_TOPIC (romance)
- **Pre-fix expectation:** GENRE commits `[COMEDY, MUSIC]`.
- **Post-fix (plural-intent positive case):** GENRE scoring_method
  may legitimately be ALL — user named two distinct genre attributes
  (comedy AND musical) the user wants compounded. This query is the
  positive control for ALL: confirm we did not over-correct so that
  ALL never fires.

---

## Section 5 — Empty-spec category filter (validates D4 / Phase 1.1)

Triggers categories where the handler is likely to abstain entirely
on at least one fired category, while another category in the same
trait fires. Pre-fix: empty category returns 0.0 → FACETS PRODUCT
zeros trait. Post-fix: empty category filtered, trait scores from
the firing category alone.

### Q14. `obscure indie passion projects`
- **Hits:** CULTURAL_STATUS, FINANCIAL_SCALE, STUDIO_BRAND
- **Pre-fix:** STUDIO_BRAND likely abstains (no specific studio
  named); under FACETS combine_mode this zeros the indie trait
  even when FINANCIAL_SCALE meta-fires cleanly.
- **Post-fix:** STUDIO_BRAND filtered from across-category fold;
  trait_score reflects FINANCIAL_SCALE + CULTURAL_STATUS only.
  Inspect `live_cats` in Phase D — should not contain abstained
  STUDIO_BRAND.

### Q15. `Studio Ghibli style hand-drawn fantasies`
- **Hits:** STUDIO_BRAND (Ghibli), GENRE (fantasy), FORMAT_VISUAL
  (hand-drawn animation)
- **Pre-fix:** if any one category abstains under FACETS the trait
  dies.
- **Post-fix:** even if FORMAT_VISUAL or one of the others
  abstains, trait_score driven by remaining live categories.

---

## Section 6 — Cross-trait keyword spec dedup (validates D5 / Phase 1.2)

When two traits route their KW endpoints to the same registry
member with the same scoring_method, the generator pass should
execute once and feed both traits' scoring paths.

### Q16. `brutal MMA fight movies`
- **Hits:** CENTRAL_TOPIC (MMA / fighting), GENRE (action / sport),
  potentially same registry members on both
- **What to inspect:** generator log / Phase B telemetry should
  show one DB query for the duplicated `(KEYWORD, params)` spec.
  Per-trait scoring should still run independently — the dedup is
  generator-side only.
- **Risk:** If trait_score for either trait diverges from baseline
  by more than rounding, dedup leaked into scoring (bug).

### Q17. `gritty crime sagas`
- **Hits:** trait `gritty` → EMOTIONAL_EXPERIENTIAL (post-D1: no KW
  here so no dedup target on this trait), trait `crime sagas` →
  GENRE / STORY_THEMATIC_ARCHETYPE
- **Purpose:** confirms that after D1 removes KW from
  EMOTIONAL_EXPERIENTIAL, dedup opportunities concentrate on the
  remaining KW-active categories. Also a regression check: the
  `gritty` trait should no longer carry the spurious
  `[DRAMA, FILM_NOIR, THRILLER]` ANY commit observed in V4.

---

## Section 7 — V4 positioning regression

V4 typology (POSITIONING_REFERENCE / POSITIONING_QUALIFIER) must
not regress when V5 ships. These queries exercise the
relationship_role + axes_replaced commit from Step 2 and the
axis-honoring decomposition in Step 3.

### Q18. `like Donnie Darko but funnier`
- **Hits:** Donnie Darko trait (POSITIONING_REFERENCE), funnier
  trait (POSITIONING_QUALIFIER, replaces_axis="tone")
- **Expect:** Donnie Darko trait drops EMOTIONAL_EXPERIENTIAL aspect
  for cerebral/unsettling tone; funnier trait covers tone via comedy
  GENRE / EMOTIONAL_EXPERIENTIAL. trait_role_analysis prose
  explicitly cites the dropped tone axis.

### Q19. `Wes Anderson aesthetic coming-of-age`
- **Hits:** Wes Anderson trait (POSITIONING_REFERENCE, axes_
  replaced=["story_archetype"] or similar), coming-of-age trait
  (POSITIONING_QUALIFIER, replaces_axis="story_archetype")
- **Expect:** Wes trait drops STORY_THEMATIC_ARCHETYPE aspect;
  coming-of-age handles it. No double-up on the archetype axis.

---

## Section 8 — Compound FACETS regression

V4 introduced FACETS (PRODUCT) for compound concepts. These
queries verify that V5 changes (especially the empty-spec filter
in D4) interact correctly with FACETS combine_mode.

### Q20. `dark gritty antihero comic-book films`
- **Hits:** dark/gritty trait (FACETS, multi-aspect), antihero
  trait (CHARACTER_ARCHETYPE, KEEP-unchanged), comic-book trait
  (FRANCHISE_LINEAGE / STUDIO_BRAND, FRAMINGS)
- **Expect:** dark/gritty trait combine_mode=FACETS; under D4 if
  any one category abstains the FACETS trait survives via the
  remaining live categories. Antihero KW commit should remain
  clean.

### Q21. `atmospheric folk horror`
- **Hits:** GENRE (folk horror), EMOTIONAL_EXPERIENTIAL
  (atmospheric, post-D1: no KW)
- **Expect:** GENRE clean ADDITIVE(KW × SEM) — KW likely
  `[FOLK_HORROR]` if registry has it, otherwise abstain. atmospheric
  trait routes only to SEM post-D1; trait_score comes from VWX
  `tonal_register`.

---

## Section 9 — Clean SEM-only abstention (validates D2 abstention reach)

Pure-vibe queries where the handler should abstain on KW for
every category that fires, leaving SEM to do the work. Confirms
the bucket prompt's partial-abstention sanction is reaching the
handler.

### Q22. `films about grief and reconciliation`
- **Hits:** CENTRAL_TOPIC, EMOTIONAL_EXPERIENTIAL
- **Pre-fix:** EMOTIONAL_EXPERIENTIAL fires KW even when registry
  has no canonical grief tag.
- **Post-fix:** Both categories pure SEM. CENTRAL_TOPIC KW abstains
  per superset (no grief-specific registry member). EMOTIONAL_
  EXPERIENTIAL has no KW endpoint at all post-D1. Zero
  ADDITIVE_KW_RISK trip-wires for the query.

### Q23. `slow-burn psychological mysteries`
- **Hits:** STORY_THEMATIC_ARCHETYPE (psychological mystery), GENRE
  (mystery), EMOTIONAL_EXPERIENTIAL (slow-burn)
- **Expect:** GENRE clean KW commit. STORY_THEMATIC_ARCHETYPE may
  abstain on KW if `PSYCHOLOGICAL_DRAMA` doesn't superset the
  query (mystery, not drama). EMOTIONAL_EXPERIENTIAL pure SEM
  post-D1.

---

## Section 10 — Negation and KEEP-ADDITIVE regression

Negative-polarity traits and KEEP-ADDITIVE-unchanged categories
should be unaffected by V5 changes.

### Q24. `coming-of-age road trips not too sappy`
- **Hits:** STORY_THEMATIC_ARCHETYPE (coming-of-age), positive
  trait. Negative trait `not too sappy` → EMOTIONAL_EXPERIENTIAL.
- **Expect:** STORY_THEMATIC_ARCHETYPE positive: KEEP+abstain
  category — clean KW commit if registry has COMING_OF_AGE
  (likely), otherwise abstain. Negative-trait path unchanged by V5
  — combine_mode is positive-trait-only.

### Q25. `unreliable narrator with a twist ending`
- **Hits:** NARRATIVE_DEVICES (KEEP-unchanged)
- **Pre-fix:** clean commit `[UNRELIABLE_NARRATOR, PLOT_TWIST]`
  but possibly ALL.
- **Post-fix:** scoring_method=ALL is *correct* here per D3 —
  user named two distinct attributes that should compound. This
  is the positive control for ALL on a multi-attribute query
  expressed across a single trait. If V5 over-corrected and
  collapsed this to ANY, that is a bug.

---

## Verification rubric per phase

| Phase | What to confirm in JSON output |
|---|---|
| **Phase 1.1 (empty-spec filter)** | Q14, Q15: trait_score > 0 even when one category abstained. `live_cats` excludes empty cats. |
| **Phase 1.2 (generator dedup)** | Q16, Q17: generator log shows single DB query for duplicated specs across traits. trait_scores match baseline within rounding. |
| **Phase 2a (TARGET_AUDIENCE / SENSITIVE_CONTENT → ALTERNATIVES)** | Q4, Q5: combine_type=ALTERNATIVES on those categories. KW-miss-with-strong-META no longer zeros category. |
| **Phase 2b (REMOVE-KW from 3 cats)** | Q1, Q2, Q3, Q22: zero KEYWORD routes fired in SEASONAL_HOLIDAY / EMOTIONAL_EXPERIENTIAL / SPECIFIC_PRAISE_CRITICISM. ADDITIVE_KW_RISK count drops sharply. |
| **Phase 3.1 (superset test in keyword.md)** | Q6, Q7, Q22: KW abstains where no superset exists (Q6 elephants, Q22 grief). Q7 commits SPORT cleanly without ALL. |
| **Phase 3.2 (singular/plural rewrite)** | Q10, Q11, Q12, Q13, Q25: paraphrase clusters score ANY (Q10–12). Genuine plural intents (Q13, Q25) still score ALL. |
| **Phase 3.3 (partial-abstention bucket sanction)** | Q6, Q22: handler abstains on KW endpoint while SEM endpoint continues to fire within the same category — visible as KEYWORD route absent from `fired_endpoints` while SEMANTIC is present. |

After each phase, run the suite and produce a `phase_<N>_diff.md`
listing per-query before/after. The whole-suite ADDITIVE_KW_RISK
percentage should fall monotonically as phases ship. If a phase
*increases* the trip-wire count or breaks a positive-control query
(Q9, Q13, Q25), stop and diagnose before proceeding.

---

## Query list (one per line)

```
heartwarming holiday films
films with a bittersweet melancholic tone
forgotten gems with brilliant performances
wholesome family movie night picks
intense action thrillers but not too bloody
movies featuring elephants
movies about marathons
films with sentient AI
revenge stories with anti-heroes
cyberpunk dystopias
historical war epics
mind-bending puzzle films about consciousness
comedy musicals about teenage romance
obscure indie passion projects
Studio Ghibli style hand-drawn fantasies
brutal MMA fight movies
gritty crime sagas
like Donnie Darko but funnier
Wes Anderson aesthetic coming-of-age
dark gritty antihero comic-book films
atmospheric folk horror
films about grief and reconciliation
slow-burn psychological mysteries
coming-of-age road trips not too sappy
unreliable narrator with a twist ending
```

---

## Notes on what was avoided

The following queries from V5 diagnostics or prompt few-shot
examples were intentionally NOT used here, to keep the suite
disjoint from material the LLM has been pre-conditioned on:

- `running movies` (canonical example in keyword.md) → replaced
  with `movies about marathons` (different sport, similar shape)
- `biographical dramas about musicians` (keyword.md example) →
  replaced via `comedy musicals about teenage romance` for the
  plural-intent control
- `movies with horses` (element_presence few-shot) → replaced
  with `movies featuring elephants` (registry-gap test)
- `zombie movies` (genre few-shot) → not used; the
  CENTRAL_TOPIC abstention case is covered by Q22
- `coming-of-age stories` / `redemption arc` (story_thematic
  few-shot) → reframed as `coming-of-age road trips` (Q24) and
  `revenge stories with anti-heroes` (Q9)
- `cult classic`, `praised for performances`, `criticized as
  plodding` (specific_praise_criticism few-shot) → reframed via
  `forgotten gems with brilliant performances` (Q3)
- `feel-good Christmas movies` (V5 diagnostic) → reframed as
  `heartwarming holiday films`
- `cheap shark movies` (V5 diagnostic) → not duplicated; the
  thin-superset cross-trait shape is covered by Q6 + Q16
- `underrated indie films` (V5 diagnostic) → reframed as
  `obscure indie passion projects`

If LLM commits drift over time and a query in this suite ends up
being added to a few-shot, replace it with a structurally similar
but lexically distinct query.
