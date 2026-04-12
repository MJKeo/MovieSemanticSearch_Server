# Franchise Generator — Test Iteration Report

Report on failure analysis of `franchise_test_results.json` (75 movies × 3
reasoning tiers × 3 samples = 675 candidates). Focus: what the failure
patterns tell us about our definitions and prompt design, not just which
rows were wrong.

## Failure categories

Every candidate that diverged from SOT was bucketed as one of:

- **F** — source-of-truth or input is flawed; model output is defensible
- **H** — harmful to search; model got something materially wrong
- **I** — technically wrong but inconsequential (synonyms, casing, extra
  low-info groups)

## Category F — SOT / input flawed

### Test harness prompt contamination (11 entries)

A harness bug sent the wrong movie's `user_prompt` into the LLM call for
11 of 75 test entries. Models were evaluated against the *expected*
movie but shown a *different* movie's data. Mismatches found:

| id | expected `title_year` | prompt actually contained |
|---|---|---|
| 417859 | Puss in Boots: The Last Wish (2022) | Puss in Boots (2011) |
| 447404 | Detective Pikachu (2019) | Pokémon Detective Pikachu (2019) |
| 18998 | Pride and Prejudice (2005) | Darkman II: The Return of Durant (1995) |
| 17015 | Great Expectations (1998) | Persuasion (1995) |
| 556984 | Emma (2020) | The Trial of the Chicago 7 (2020) |
| 6038 | Scooby-Doo (2002) | Shanghai Knights (2003) |
| 59440 | The Town (2010) | Warrior (2011) |
| 140607 | Snowpiercer (2013) | Star Wars: The Force Awakens (2015) |
| 771 | Home Alone 2: Lost in New York (1992) | Home Alone (1990) |
| 578 | Jaws 2 (1978) | Jaws (1975) |
| 338952 | Pacific Rim: Uprising (2018) | Fantastic Beasts: The Crimes of Grindelwald (2018) |

Consequences for earlier analysis:
- Home Alone 2 / Jaws 2 being called `starter` was **not** a "numbered
  sequel misread" — the model never saw the "2". These are fully
  explained by contamination.
- Pride and Prejudice → Darkman, Scooby-Doo → Shanghai Noon, Snowpiercer
  → Star Wars, Pacific Rim → Wizarding World are all explained the
  same way. All model outputs in those cases are correct for the
  polluted input.

**Status: fixed (user resolved the underlying TMDB-fetch bug).**
Before drawing any further conclusions about model behavior, the
test set must be re-run against the fixed prompts.

### SOT missing real cultural sub-groupings

Models repeatedly surfaced grouping labels that are genuinely
culturally recognized but absent from SOT. These are SOT omissions,
not model errors:

- **Disney live-action remakes** (Lion King 2019, Aladdin 2019, Jungle
  Book 2016) — canonical Disney marketing label
- **Wolverine trilogy** (Logan) — widely-recognized sub-series
- **Raimi trilogy** (Spider-Man 2002) — SOT had "Raimi trilogy"; models
  said "Sam Raimi trilogy" (pure synonym)
- **Ip Man trilogy** (Ip Man) — real sub-grouping
- **Michael Bay era** (Transformers) — widely-used label for T1–T5
- **Minions films** (Minions) — real sub-series within Despicable Me

Either accept these in SOT or add explicit instructions excluding
sub-trilogy groupings.

### Creed — SOT is defensible-but-wrong

Initially bucketed as "harmful" because SOT says `spinoff`. On
reflection the model's `mainline` call is at least as defensible:
- Rocky Balboa is a major co-lead with real screen time, not a cameo
- The plot is a direct narrative continuation of *Rocky Balboa* (2006)
- Spinoff usually implies parent-franchise characters are absent or
  peripheral; that's not true here

Reclassified as **F**. Worth revisiting the SOT label.

## Category H — Genuinely harmful (after removing contamination)

The remaining harmful cases cluster into four definition-level
problems. These are the patterns worth acting on.

### 1. Umbrella-brand attraction

Cases: Godzilla 2014 → "MonsterVerse", Fantastic Beasts → "Wizarding
World", Jungle Book 2016 / Aladdin 2019 → "Disney", Wonder Woman
outlier, Rise of the Planet of the Apes.

The model has no rule for choosing between a narrow lineage name
(Godzilla, Harry Potter) and a broad shared-universe name
(MonsterVerse, Wizarding World, MCU, DCEU). "MonsterVerse" and
"Wizarding World" are real, studio-sanctioned labels — the model is
not hallucinating. But picking them over the lineage name severs
search linkages ("show me Godzilla movies" would miss Godzilla 2014).

**Proposed definition change:** add an explicit priority rule.

> If a film descends from an earlier same-title, same-character, or
> same-source-material lineage, `franchise_name` MUST be that lineage.
> Shared-universe umbrella labels (MCU, DCEU, MonsterVerse, Wizarding
> World, Dark Universe, etc.) are reserved for films with no such
> lineage of their own. A film can belong to an umbrella *as a group*,
> but the franchise name is the narrowest recognizable lineage.

This single rule would fix Godzilla, Fantastic Beasts, Jungle Book
2016, Aladdin 2019, and Wonder Woman's outlier in one pass. It also
makes the "Wizarding World vs Harry Potter" question a schema
decision rather than a case-by-case judgment call.

### 2. `starter` vs `mainline` — no chronological check

Cases: Rurouni Kenshin: Origins (called `mainline`), Sonic the
Hedgehog 2020 (called `mainline`), Super Mario Bros. Movie 2023
(called `mainline`).

The model knows these franchises *as franchises*, so when given any
entry it defaults to `mainline`. The definition probably describes
`starter` passively ("the first film in a series") without forcing a
verification step. The model never asks "is THIS film the earliest?"

**Proposed definition change:** make the chronological check
mandatory and explicit.

> Before assigning any role, identify the earliest theatrical film in
> this franchise by release year. If the current film IS that earliest
> film, the role is `starter`. Do not assign `mainline` without first
> confirming an earlier film exists.

Forcing this check into the reasoning trace should eliminate the
"defaults to mainline" failure mode.

### 3. `reboot` / `remake` — model misses older predecessors

Cases: Scarface 1983 (called `starter`, ignoring the 1932 Howard
Hawks original), Rise of the Planet of the Apes (called `starter`,
ignoring the 1968 franchise).

This is a knowledge-recency failure: the model treats the culturally
dominant version as "the first" because older films aren't salient.
Passive definitions of `remake` and `reboot` don't help because the
model never actively checks whether a predecessor exists.

**Proposed definition change:** require a predecessor search as a
gating step.

> If ANY earlier theatrical film exists with the same title, same
> central character, or adapting the same source material, this film
> cannot be `starter`. It is:
> - `remake` — a new production of the same story with different
>   cast and continuity
> - `reboot` — a restart of a dormant franchise with continuity reset
> - `mainline` — part of a continuous series

Combined with rule 2, this makes both `starter` and `mainline`
conditional on an explicit check rather than a vibe.

### 4. Over-trusting `collection_name` / fabricating franchises for standalones

Two distinct sub-cases:

**(a) Thematic collections misread as franchises.** Oldboy's prompt
carries `collection_name: The Vengeance Trilogy`. Minimal-tier
candidates dutifully wrote `franchise_name: The Vengeance Trilogy,
role: mainline`. But Park Chan-wook's Vengeance Trilogy is a
*thematic* grouping — no shared characters, no continuity, different
protagonists. By our own schema this should be standalone.

**(b) Fabrication on empty collection.** Ready Player One, Little
Women 2019, Romeo + Juliet 1996 all have `collection_name: not
available`, yet minimal-tier candidates still invented franchise
entries for them.

**Proposed definition changes:** two rules, one for each sub-case.

> A TMDB collection is evidence but not proof of a franchise. A true
> franchise requires shared characters, shared continuity, OR
> sequential narrative. Thematic anthologies (e.g., Park Chan-wook's
> Vengeance Trilogy, Kieślowski's Three Colors, Linklater's Before
> trilogy) are NOT franchises — classify these films as standalone
> and return null.

> If you cannot name at least one other specific film in this
> franchise by title, it is not a franchise — return null. Do not
> populate `franchise_name` speculatively.

The second rule is the one that kills the Ready Player One / Little
Women hallucinations directly.

## Category I — Inconsequential failures

These dominate the raw row count (roughly 290 of ~347 failure rows)
but are not signal about model quality. Two sub-patterns:

### Low-info group additions

The vast majority of I-class rows are the model adding `"original
trilogy"`, `"film series"`, `"franchise"`, or `"X film series"` to
`culturally_recognized_groups` when SOT expected `[]`. Examples:
Toy Story 3, Jurassic Park, The Matrix, Saw, Shrek, Mad Max, Kung
Fu Panda 3, Train to Busan, K.G.F, Dhoom 2, Mission: Impossible
Fallout, Hobbs & Shaw, Puss in Boots, Resident Evil, Goldfinger.

These are low-info because they restate the franchise name. Fix
with a prompt instruction: *"Do not list groups that merely restate
the franchise name (e.g. 'X film series', 'X franchise'). Only list
groups with their own culturally recognized identity (e.g. 'The
Dark Knight Trilogy', 'Infinity Saga', 'Disney live-action
remakes')."*

### MCU group casing / number-word inconsistency

Spider-Man: Homecoming, Black Panther, Iron Man, Avengers: Endgame,
Guardians of the Galaxy — all fail on `"phase three"` vs `"phase 3"`
or `"Infinity Saga"` vs `"infinity saga"`. Pure case/number-word
mismatch. Fix with a normalizer at ingest (lowercase, word→digit for
phase numbers), not with prompt changes.

Also worth noting: Spider-Man: Homecoming's SOT has `["phase 3"]`
while Black Panther's SOT has `["phase 3", "infinity saga"]`. That's
a SOT consistency bug, not a model failure — Homecoming is also in
the Infinity Saga.

## Meta-observation: reasoning tier matters

Almost every harmful failure came from `gpt5-mini-minimal`:

- Oldboy → "Vengeance Trilogy" (minimal only)
- Little Women → starter (minimal only)
- Romeo + Juliet → spinoff (minimal only)
- Ready Player One → starter (minimal only)
- Aladdin 2019 → "Disney" / "Walt Disney Pictures" (minimal only)
- Jungle Book 2016 → "Disney" / "Walt Disney Pictures" (minimal only)
- Sonic → mainline (minimal only)
- Wonder Woman → starter (minimal only, one candidate)

Medium and low tiers mostly avoid these. Strong signal that franchise
classification needs reasoning budget for:
- the chronological check (rule 2)
- the predecessor search (rule 3)
- the "can I name another film in this franchise?" check (rule 4b)

Recommendation: **require at least `low` reasoning for this
generator.** Otherwise we'll be playing whack-a-mole against
`minimal`'s shortcuts even with improved definitions.

## Action checklist

Priority-ordered:

1. **Re-run the test set** against the fixed TMDB prompts before
   drawing further model-quality conclusions. 11 of 75 entries were
   contaminated; headline numbers are unreliable until re-run.
2. **Add the four definition rules** above:
   - umbrella-brand priority rule
   - mandatory chronological check before `mainline`
   - mandatory predecessor search before `starter`
   - thematic-anthology exclusion + "name another film" fabrication guard
3. **Add a group-normalization pass at ingest** (lowercase, phase
   word→digit). Resolves the MCU case cluster without prompt changes.
4. **Add a low-info group exclusion instruction** to the prompt ("do
   not list groups that merely restate the franchise name").
5. **Revisit SOT** for: Creed's role, Disney live-action remakes
   subgroup, Wolverine/Raimi/X-Men/Ip Man sub-trilogies, Spider-Man
   Homecoming's missing "infinity saga", and the "Wizarding World vs
   Harry Potter" schema decision.
6. **Require ≥ `low` reasoning** for the franchise generator. Drop
   `minimal` from the tier matrix for this task.
7. **Measure null/non-null pairing consistency.** The `model_validator`
   was removed pre-evaluation. During the re-run, count how often the
   LLM produces mismatched pairings across `franchise_name`,
   `franchise_role`, and `culturally_recognized_groups` (e.g. name
   present but role null, or groups populated with name null). If the
   rate is negligible, leave it removed. If significant, re-add as a
   `validate_and_fix()` fixup rather than a hard validation error.
8. **Check `launches_subgroup` / groups co-occurrence (v2).** By
   definition, `launches_subgroup=True` should always imply that the
   launching subgroup's label appears in `culturally_recognized_groups`
   (and the reverse: if a film launches a subgroup, there must be a
   name for it). Count v2-run cases where the model emits
   `launches_subgroup=True` alongside an empty `culturally_recognized_groups`.
   If this happens non-trivially, tighten the prompt to explicitly
   require the launching label be present in the groups list.

---

# v2 run — per-candidate evaluation (79 movies × 4 models × 3 samples)

Headline numbers from the v2 run, after re-classifying each failing
row as H (harmful), D (debatable / SOT flaw), or I (inconsequential /
normalization):

| Candidate | Correct | H | D | I | Effective |
|---|---|---|---|---|---|
| gpt5-mini-medium | 60% | 5% | 24% | 11% | 71% |
| gpt5-mini-low | 61% | 5% | 24% | 10% | 71% |
| gpt5-mini-minimal | 53% | 11% | 22% | 14% | 67% |
| gpt54-mini-none | 62% | 9% | 21% | 8% | 70% |

Medium and low are statistically indistinguishable — the extra
reasoning budget bought nothing. Minimal has roughly 2× the harmful
rate of medium/low and adds a distinctive top-level-entity-rule
violation cluster (Logan→name='x-men', Batman Begins→name='batman',
X-Men 2000→name='x-men') that other tiers avoid. None has the highest
raw accuracy but its harmful failures cluster on missed predecessors
(Super Mario Bros Movie 3/3, Aladdin 2019 name='disney' 2/3).

## Confirmed failure patterns and causes

1. **`role='spinoff'` over-application to shared-universe pillar films
   and legacy sequels.** Every wrong spinoff trace followed the same
   shortcut: "franchise existed before + character appeared earlier
   + now has own film → spinoff." Wonder Woman (medium 3/3, none 2/3),
   Black Panther (scattered), Creed (every tier 3/3). Root cause: the
   v2 prompt stated "MINOR character" as a spinoff constraint but
   never operationalized "minor," and had no language distinguishing
   planned shared-universe pillar films from genuine side-project
   spinoffs. Models dropped the minor constraint and collapsed
   "prior appearance + own film" to spinoff.

2. **`role=None` on films that belong to a franchise existing in
   non-cinematic media.** Transformers 2007 (every tier 3/3),
   Guardians of the Galaxy (medium/low 3/3), Rurouni Kenshin Origins
   (medium 3/3), Dragon Ball Z: Resurrection F (every tier partial).
   Root cause: the v2 prompt forbade starter for prior-IP films but
   provided no role that fit first-theatrical entries, so
   "missing over wrong" forced null. Models were correctly following
   the prompt; the prompt itself had a gap.

3. **`culturally_recognized_groups` noise (74-87 rows per tier).**
   Dominant sub-patterns:
   - `"original trilogy"` padding on Toy Story 3, Jurassic Park,
     Matrix, Mad Max, Kung Fu Panda 3, Iron Man, etc. — caused by
     the prompt's Raiders reference example teaching models to
     generalize the label to any trilogy.
   - Bare sub-series restatements: `"minions"`, `"creed"`,
     `"guardians of the galaxy"`, `"the avengers"`, `"iron man
     trilogy"`, `"fantastic beasts"`, `"puss in boots"`, `"hobbs
     and shaw"` — the v2 restatement rule only forbade restating
     `franchise_name`, not restating the film's own sub-series.
   - Missing SOT labels that are real: `"skywalker saga"` on all
     Star Wars OT/PT, `"disney live-action remakes"` on Aladdin /
     Jungle Book / Lion King, `"wizarding world"` on Harry Potter,
     `"wolverine trilogy"` on Logan, `"caesar trilogy"` vs `"reboot
     trilogy"` on Rise of the Planet of the Apes.
   - Normalization misses: `"guy ritchie sherlock holmes films"`
     vs `"ritchie sherlock holmes films"` (first-name drop rule not
     applied), `"phase 3"` vs `"phase three"`.

4. **Minimal-tier top-level entity rule violations.** Logan (3/3
   name='x-men'), X-Men 2000 (2/3 name='x-men'), Batman Begins (2/3
   name='batman'), Dark Knight Rises (1/3 name='batman'), Spider-Man
   2002 (all tiers 3/3 name='spider-man'). Root cause: the v2 prompt
   stated the top-level entity rule in a separate section but did
   not re-apply it as a forced step inside the `franchise_name`
   procedure. Higher tiers performed the check anyway; minimal
   skipped it.

5. **Missing-over-wrong bias backfiring on the role field.**
   Guardians of the Galaxy (medium 3/3 role=None), Dragon Ball Z
   (role=None), Rurouni Kenshin on medium (role=None), Iron Man
   (role=starter on minimal). Models correctly followed the "null
   is preferred over weak guess" rule, but SOT expected mainline
   for GoG and starter for Rurouni Kenshin. The prompt was biasing
   the model toward an outcome rather than letting it think
   critically.

---

# v3 changes — prompt + schema rewrite

Full rewrite of `movie_ingestion/metadata_generation/prompts/franchise.py`
and `schemas/metadata.py` `FranchiseOutput` field descriptions. The
rewrite targets the five confirmed patterns above rather than
patching individual failures.

## Structural changes

- **Numbered procedural walkthroughs** replace free-form "Step 1
  Evidence / Step 2 Analysis" bullets on every field. Mirrors
  `concept_tags.py` ENDINGS comparative evaluation, which replaced
  factual-ledger shortcuts with forced step-by-step reasoning.
- **Evidence hierarchy** (direct / concrete inference / parametric)
  stated once up front so every field reasons against the same rubric.
- **Annotated INPUTS section** — each input carries its reasoning role
  (primary / confirming / cross-reference signal) and which fields it
  serves. `top_billed_cast` is called out as the primary signal for
  the spinoff prior-role prominence test.
- **EMPTY-THEN-ADD framing** for `culturally_recognized_groups`
  replaces brainstorm-then-filter. Parallels concept_tags' "starting
  point for every category is EMPTY" pattern.
- **IS NOT clauses** added to every field, derived directly from
  observed v2 failure modes.
- **Positive commitment framing** throughout. "Prefer missing over
  wrong" language is removed. Replaced with "when evidence points
  somewhere, commit; null is reserved for genuine non-fit, not
  uncertainty."
- **Schema field descriptions encode the same procedural
  walkthroughs in compact form**, so the structured-output JSON
  schema itself carries the procedure, not just the system prompt.

## Field-specific changes

### franchise_name
- Top-level entity rule elevated into a forced step *inside* the
  procedure, not just a sidebar. Targets the minimal-tier
  `name='batman'/'x-men'/'spider-man'` failure mode.
- IS NOT clauses: no studios/distributors (Disney, Warner Bros.), no
  sub-lineages (Batman, Spider-Man, X-Men, Wolverine), no single
  source works without a multi-entry brand, no thematic critic-coined
  labels, no plot-similarity inference.

### franchise_role
- Comparative evaluation replaces first-match scan. The procedure
  explicitly instructs: "build the affirmative case for each
  candidate role, commit to the strongest."
- **Spinoff operationalized as three required constraints** that
  must ALL hold:
  - (a) MINOR IN SOURCE — character/element/subplot was minor in
    the source work, not a lead, co-lead, or planned pillar.
  - (b) GOES SOMEWHERE NEW — new protagonist / story spine /
    setting focus.
  - (c) LEAVES THE SOURCE BEHIND — source film's main characters
    and main plot are not the focus here.
  An explicit "Examples where constraints fail" block lists Wonder
  Woman, Black Panther, Creed, Ghostbusters: Afterlife as mainline,
  with the specific constraint each fails called out.
- **NULL ROLE is a first-class answer** with two documented valid
  cases: (i) first theatrical entry in a franchise whose prior
  existence is limited to non-cinematic media (toys, games, TV,
  anime, manga, comics-only continuity), (ii) documentaries.
  "Null is NOT correct when choosing between two plausible roles"
  stated explicitly.
- Shared-universe pillar language added to mainline: "planned
  shared-universe pillar films for headline characters introduced
  in earlier ensemble films" — Wonder Woman 2017, Black Panther
  2018, Captain Marvel 2019, Doctor Strange 2016, Thor 2011.
- Legacy sequel language added: prior protagonist present as a
  major character means mainline, not spinoff (Creed, Tron:
  Legacy, Top Gun: Maverick, Blade Runner 2049, Ghostbusters:
  Afterlife, Halloween 2018).

### is_prequel
- Procedure tightened, but this field was already clean and is
  low-priority for quality improvement.
- IS NOT clauses added: reboots set early (Batman Begins),
  flashback-heavy mainline, first films of a franchise, films set
  in the same time period as an earlier entry.

### launches_subgroup
- Coupled to `culturally_recognized_groups` via an explicit internal
  consistency check: "if true, the launched subgroup's normalized
  label MUST appear in this film's groups output." Directly
  addresses action item #8 from the v2 run.
- Mad Max 1979 given as an explicit FALSE example because "the
  original Mad Max trilogy" is not a culturally-used label despite
  it being first of a trilogy.

### culturally_recognized_groups
- **EMPTY-THEN-ADD framing**: "starting point is an empty list; add
  each label deliberately, one at a time, with a specific reason."
- **Three named evidence tiers** for inclusion: studio/official
  usage (wizarding world, phase three, monsterverse, disney live-
  action remakes), mainstream critical convention (the dark knight
  trilogy, kelvin timeline, daniel craig era), widely-used fan
  terminology (snyderverse, bayverse, caesar trilogy).
- **Raiders → `["original trilogy"]` example removed.** Was a
  booby trap that trained models to generalize "original trilogy"
  to every trilogy. "Original trilogy" is now explicitly scoped:
  valid ONLY for Star Wars, where it's the commonly-used
  distinguishing label (contrasted against prequel and sequel
  trilogies).
- **Sub-series restatement blocked.** Bare sub-series names are
  explicitly forbidden: "minions", "creed", "guardians of the
  galaxy", "the avengers", "puss in boots", "hobbs and shaw",
  "iron man trilogy", "fantastic beasts". Exception carved out for
  sub-series labels with distinct critical identity ("the dark
  knight trilogy", "raimi trilogy").
- First-name-drop normalization rule kept, with the Sherlock
  Holmes failure ("guy ritchie sherlock holmes films" → "ritchie
  sherlock holmes films") added as an explicit example.

## FranchiseRole enum
- **Unchanged.** No roles added or removed. Null covers the new
  non-cinematic-IP-first-theatrical case. The v2 enum (starter,
  mainline, spinoff, reboot, remake, crossover) is the final
  vocabulary.

## SOT updates required before next test run

The v3 prompt makes null a correct answer for cases where v2 SOT
still expects a role. Running the v2 test set against the v3 prompt
without updating SOT will produce false negatives on these cases
because the model will correctly commit to null and SOT will score
it wrong.

| id | title | v2 SOT role | v3 intended role |
|---|---|---|---|
| 1858 | Transformers (2007) | reboot | **null** |
| 127533 | Rurouni Kenshin Part I: Origins (2012) | starter | **null** |
| 447404 | Detective Pikachu (2019) | reboot | **null** |
| 303857 | Dragon Ball Z: Resurrection F (2015) | mainline | **null** if no prior same-continuity film; otherwise keep mainline |
| 36657 | X-Men (2000) | null (name=marvel) | **null** (name=marvel) — already matches |
| 1726 | Iron Man (2008) | null (name=marvel) | **null** (name=marvel) — already matches |
| 1576 | Resident Evil (2002) | null (name=resident evil) | **null** — already matches |
| 9637 | Scooby-Doo (2002) | null (name=scooby-doo) | **null** — already matches |
| 454626 | Sonic the Hedgehog (2020) | null (name=sonic the hedgehog) | **null** — already matches |
| 312221 | Creed (2015) | spinoff | **mainline** (confirmed by v3 definition: Rocky is a major returning co-lead) |

Also review (lower priority):
- SOT missing labels to add: `"skywalker saga"` on Star Wars OT/PT,
  `"disney live-action remakes"` on Aladdin 2019 / Jungle Book 2016 /
  Lion King 2019, `"wizarding world"` on Harry Potter mainline
  entries, `"wolverine trilogy"` on Logan, `"caesar trilogy"` on
  Rise of the Planet of the Apes.
- Spider-Man: Homecoming SOT missing `"infinity saga"` (consistency
  bug — Black Panther has it, Homecoming does not, both are phase
  three).

---

# Next test run — what to verify

After SOT updates, re-run the v2 test set (79 movies × all tiers ×
3 samples) against the v3 prompt. Verification targets, organized by
the failure pattern they test:

## Pattern 1 — spinoff over-application (expected: largely fixed)

- [ ] Wonder Woman (2017) — expect `role='mainline'` across all
  tiers. The v2 run had medium 3/3 spinoff, none 2/3 spinoff/starter.
  If v3 still produces spinoff, the leaves-behind test is being
  interpreted wrong or the planned-pillar disqualifier is being
  skipped.
- [ ] Black Panther (2018) — expect `role='mainline'` across all
  tiers. v2 had scattered single-row spinoff failures.
- [ ] Creed (2015) — expect `role='mainline'` (SOT updated). The
  v2 run had every tier 3/3 spinoff with consistent reasoning
  ("Apollo Creed's son is a legacy element"). If v3 still says
  spinoff, the "legacy sequel with prior protagonist present"
  disqualifier isn't firing.
- [ ] Puss in Boots: The Last Wish — debatable case. v3 might now
  call this mainline (continuation of Puss in Boots 2011) rather
  than spinoff-of-Shrek. Check reasoning and decide whether to
  update SOT.
- [ ] Rogue One, Solo — expect `role='spinoff'` with the
  leaves-behind test passing (Skywalker main plot is not the focus).

Spot-check reasoning traces on at least one passing case and one
failing case for each film above. If the reasoning cites the three
spinoff constraints by name, the procedural walkthrough is working.
If it collapses back to "character appeared before → spinoff," the
prompt isn't being followed and needs more forceful language or
schema-level enforcement.

## Pattern 2 — NULL role for non-cinematic-IP first-theatrical
  (expected: fixed, but only if SOT is updated)

- [ ] Transformers (2007) — expect `role=null` with `franchise_name
  ='Transformers'`. Confirm across all tiers.
- [ ] Rurouni Kenshin: Origins (2012) — expect `role=null`. The v2
  run had medium 3/3 null (correctly per the v2 prompt) but SOT
  expected starter. v3 + updated SOT should now show these as
  correct.
- [ ] Detective Pikachu (2019) — expect `role=null`. Also a
  character-encoding case: confirm `franchise_name='Pokemon'` (or
  'Pokémon' — pick one and update SOT to match).
- [ ] Dragon Ball Z: Resurrection F — expect either `role=null`
  (if treated as first theatrical in the continuity) or mainline
  (if the model recognizes prior DBZ theatrical films). Check
  reasoning to understand which the model is applying.
- [ ] Sonic the Hedgehog (2020), Resident Evil (2002), Scooby-Doo
  (2002), X-Men (2000), Iron Man (2008) — all expected `role=null`
  already in v2 SOT. Confirm v3 maintains this and that medium and
  low no longer produce the old drift.

## Pattern 3 — culturally_recognized_groups noise
  (expected: large reduction)

- [ ] `"original trilogy"` padding — expect drops to zero on Toy
  Story 3, Jurassic Park, Matrix, Mad Max, Kung Fu Panda 3, Iron
  Man, Jaws 2, and all other non-Star-Wars trilogies. Star Wars OT
  entries should still get it. If any non-Star-Wars film still
  emits it, the example scope isn't being respected.
- [ ] Bare sub-series restatements — expect drops to zero on
  Minions, Creed, Guardians of the Galaxy, Endgame, Puss in Boots,
  Hobbs & Shaw, Fantastic Beasts, Iron Man.
- [ ] First-name-drop normalization — expect `"ritchie sherlock
  holmes films"` not `"guy ritchie sherlock holmes films"` on
  Sherlock Holmes 2009. If the rule still isn't applied, promote
  it to an ingest-time normalizer.
- [ ] Empty-then-add discipline — spot-check 10 random non-Star-Wars
  films that should have empty groups lists. The most common
  correct output is `[]`.
- [ ] SOT omission labels — after updating SOT, verify models emit
  `"skywalker saga"` on Star Wars OT/PT, `"disney live-action
  remakes"` on Disney remakes, `"wolverine trilogy"` on Logan.

## Pattern 4 — top-level entity rule violations on minimal

- [ ] Logan — expect `franchise_name='Marvel'` (not 'x-men') across
  all tiers including minimal. If minimal still produces 'x-men',
  the forced procedure step inside `franchise_name` isn't being
  honored on minimal.
- [ ] X-Men (2000) — same expectation.
- [ ] Batman Begins, Dark Knight Rises — expect
  `franchise_name='DC Comics'` (not 'batman') on minimal.
- [ ] Spider-Man (2002) — expect `franchise_name='Marvel'` (not
  'spider-man') on all tiers. This one failed even at medium/low
  in v2 because models were pattern-matching on "Spider-Man is a
  standalone film lineage."
- [ ] Aladdin (2019) — expect `franchise_name='Aladdin'` (not
  'Disney') on none/minimal. None had 2/3 with 'disney' in v2.

If minimal still has this failure cluster after v3, the action is
to drop minimal from the tier matrix per the v2 recommendation. The
prompt rewrite was not designed to fix a reasoning-budget problem.

## Pattern 5 — missing-over-wrong bias

- [ ] Guardians of the Galaxy — expect `role='mainline'`, not null.
  v2 had medium 3/3 null. If v3 still produces null, the commitment
  framing isn't working and we need to revisit the procedure step
  that says "commit to the strongest case."
- [ ] Endgame, other MCU mainline entries — confirm no drift to
  null.

## Internal consistency checks (run as post-processing)

After the test run, compute these counts directly from the results
JSON — they don't need per-movie judgment, just field comparison.

- [ ] **`launches_subgroup` ⇒ label in groups.** For every row
  where `launches_subgroup=true`, confirm the film's
  `culturally_recognized_groups` contains a label for the launched
  subgroup. Count violations. Target: 0.
- [ ] **Null propagation.** For every row where `franchise_name=null`,
  confirm `franchise_role=null`, `culturally_recognized_groups=[]`,
  `is_prequel=false`, `launches_subgroup=false`. Count violations.
  Target: 0 (hard schema constraint).
- [ ] **Role-group coherence.** For every row where
  `franchise_role in {starter, reboot}`, the film should NOT be a
  prequel (`is_prequel=false`). Count violations. Target: 0.
- [ ] **Sub-series restatement check.** Scan every
  `culturally_recognized_groups` entry for bare sub-series names
  (exact matches of: 'minions', 'creed', 'guardians of the galaxy',
  'the avengers', 'puss in boots', 'hobbs and shaw', 'iron man
  trilogy', 'fantastic beasts', 'the dark knight' without
  'trilogy'). Count violations. Target: 0.
- [ ] **"original trilogy" on non-Star-Wars films.** Scan every
  row for `"original trilogy"` in groups AND the film is not a
  Star Wars OT entry. Count violations. Target: 0.

## Per-candidate target metrics

With all of the above fixes landing and SOT updated, the target for
the next run is:

| Candidate | Correct target | H target |
|---|---|---|
| gpt5-mini-medium | ≥ 78% | ≤ 2% |
| gpt5-mini-low | ≥ 78% | ≤ 2% |
| gpt5-mini-minimal | ≥ 70% | ≤ 5% (still limited by top-level entity drift) |
| gpt54-mini-none | ≥ 75% | ≤ 4% |

If medium and low both land above 80% correct with H ≤ 2%, the
recommended production setting is `low` (equivalent quality to
medium at lower cost). If minimal remains above 5% H, drop it from
the tier matrix and do not ship it for this task. If none matches
or exceeds low on correct rate, consider it as the production
setting for cost reasons and fall back to low only on the films
where none's H failures concentrate (shared-universe films,
non-cinematic-IP first theatricals).

---

# v3 run results — summary

Test: 79 movies × 4 candidates × 3 samples = 948 samples against the
v3 prompt, scored with v3 SOT overrides applied.

| Candidate | Correct | H | D | I | Effective |
|---|---|---|---|---|---|
| gpt5-mini-medium | 80.2% | ~4% | ~6% | ~10% | ~86% |
| gpt5-mini-low | 80.2% | ~4% | ~6% | ~10% | ~86% |
| gpt54-mini-low | 78.1% | ~5% | ~7% | ~10% | ~85% |
| gpt5-mini-minimal | 62.4% | ~14% | ~8% | ~15% | ~71% |

Medium and low are indistinguishable — the extra reasoning budget
bought nothing. Raw accuracy jumped ~19 points over v2 on medium/low
(60% → 80%). v2's five targeted failure patterns are largely fixed:
spinoff over-application (Wonder Woman/BP/Creed → mainline),
missing-over-wrong null bias (Guardians → mainline), and
"original trilogy" padding on non-Star-Wars films (zero leakage
found). Top-level entity violations and bare sub-series emission
are partially fixed — lingering on minimal specifically.

**Production recommendation:** drop `minimal` from the tier matrix,
ship `low`. Medium buys nothing over low. Minimal has 2× the H rate,
2× the bare-sub-series rate, and is the only tier producing
null-propagation and launches_subgroup-coupling violations.

## Scoring-methodology correction

`culturally_recognized_groups` over-inclusion is NOT a failure and
should not be scored as one in future runs. Redundant but non-wrong
labels (e.g. "the matrix trilogy" on Matrix, "dragon ball z" on DBZ)
have no negative impact on retrieval — they add a key that nothing
searches for. Only MISSING important labels should count against a
candidate on this field. Rescore future runs accordingly: a row is
wrong on groups only if it (a) emits a factually incorrect label or
(b) omits a label from an SOT-recognized group the film definitely
belongs to.

This reclassification moves most of the bare-sub-series failures
from I into "correct," and brings effective accuracy on medium/low
closer to ~92%.

## Open issues flagged for the role-schema rewrite

The v3 run surfaced three structural problems with the role schema
itself that no amount of prompt-tuning will fix. Capture for the
upcoming redesign.

### 1. `franchise_role` name is load-bearing in a misleading way

Scarface (1983) retelling Scarface (1932) is a clean remake
relationship, but "Scarface" is not a franchise — no brand, no
ongoing lineage, no merchandise, no audience treating it as a
franchise. The v3 prompt's remake definition accepts a single prior
film while the franchise definition requires "multi-entry brand
identity" — these two requirements are mutually inconsistent for
pair-remakes (Scarface, Total Recall 2012, True Grit 2010, Psycho
1998). The model currently resolves the contradiction by returning
`name=null, role=null`, which cascades via the hard null rule.

The underlying issue is naming: `franchise_role` implies the film
must be in a franchise for the role to be meaningful. But "remake"
is a relationship between two films, not a statement about franchise
membership. Same critique applies less sharply to "reboot" (usually
implies a brand with multiple entries, but not always).

Redesign options:
- Split the field: one slot for franchise membership, a separate
  slot for inter-film relationships (remake-of, reboot-of,
  prequel-to, spinoff-of) that can populate independently of
  franchise_name.
- Drop remake from the role vocabulary entirely and treat
  single-pair remakes as standalone films with a pointer to the
  original elsewhere in metadata.
- Keep the current structure but explicitly allow
  `name=null, role="remake"` by relaxing the null-propagation rule.

Prefer the split. Cleanest separation of concerns.

### 2. Null-role case definition is incoherent for Iron Man / X-Men 2000

The v3 prompt documents "first theatrical entry in a non-cinematic-IP
franchise" as a valid null-role case, and lists Iron Man (2008) and
X-Men (2000) in the examples. But Marvel IS a cinematic franchise
by 2008 — Blade (1998), X-Men (2000), Spider-Man (2002), the Fantastic
Four films, and others all predate Iron Man. The example contradicts
the rule. Models that commit to `mainline` are correctly applying the
stated rule (Marvel is cinematic, Iron Man is a shared-universe solo
debut); models that commit to `null` are following the example list.
The prompt can't be followed consistently because it's internally
contradictory.

Root cause: the role vocabulary doesn't have a good slot for "first
solo film in an existing shared cinematic universe for a character
with no prior theatrical appearance as a lead." The current options
are mainline (stretches the definition), starter (blocked by the
top-level brand predating), or null (stretches the non-cinematic-IP
rule). None fit cleanly.

The redesign should either introduce a dedicated role for
shared-universe-solo-debut films or explicitly fold them into mainline
with a clear tiebreaker that doesn't depend on the non-cinematic-IP
carve-out.

### 3. reboot vs remake overlap has no clean tiebreaker

Jungle Book (2016), Cinderella (2015), Mulan (2020), Karate Kid
(2010) are all "same story spine + fresh continuity + prior
theatrical entry exists" — which satisfies BOTH the reboot
definition ("fresh continuity telling a new story") and the remake
definition ("retelling the same core premise"). The v3 prompt's
tiebreaker ("if incoherent without the prior film's plot as
scaffolding → remake") is too subtle for minimal and ambiguous even
for medium/low on films like Jungle Book where the story spine
matches but the continuity is fresh.

The problem is categorical: reboot and remake occupy overlapping
regions of the same concept space. The current definitions treat
them as disjoint when they're not. Redesign options:
- Merge them into a single role ("new version of an existing film")
  and distinguish via a sub-flag (continuity-reset yes/no).
- Tighten remake to require that the prior film is the ONLY prior
  film in the brand (single-pair remakes), and make reboot require
  a brand with ≥2 prior theatrical entries being reset.
- Add an explicit source-type tiebreaker: live-action version of
  an animated classic → remake; live-action version of live-action
  → reboot.

Prefer the ≥2-prior-entry tiebreaker for reboot. Matches intuition
(Batman Begins reboots a brand; Jungle Book 2016 remakes a film).

## Role mutual-exclusivity audit

For the redesign, here's the mutual-exclusivity structure of the
current v3 role vocabulary. The role set is NOT a partition — it's
a set of overlapping predicates with prompt-level tiebreakers.

### Truly mutually exclusive
- `starter` ⊥ `{mainline, reboot, remake, spinoff}` — the latter
  four all require a prior theatrical entry; starter forbids one.
- `starter` ⊥ null-case-(i) — null-case requires prior non-cinematic
  IP to exist; starter requires nothing to exist.
- `starter` ⊥ `crossover` — crossover requires two existing top-level
  franchises.
- `mainline` ⊥ `reboot` — mainline requires prior films' events to
  be canon; reboot requires them to NOT be canon. Direct contradiction.

### Real overlaps that need prompt-level tiebreakers
- `mainline` ↔ `spinoff` — legacy sequels and pillar films that
  follow a side character while bringing the original protagonist
  along satisfy both. v3 resolves via the leaves-behind test
  (constraint c). Tiebreaker is arbitrary — Creed, Puss in Boots:
  The Last Wish, and Fantastic Beasts all live in this overlap.
- `reboot` ↔ `remake` — see issue 3 above. Both match "same
  characters, fresh continuity, prior film exists" for many films.
- `mainline` ↔ `remake` — rare but real. Rehash sequels that
  re-tell the first film's plot (arguably The Force Awakens vs A
  New Hope) technically qualify as both. Prompt handles implicitly
  by not listing rehash sequels as remake examples.
- `spinoff` ↔ `crossover` — a spinoff that pulls in a character
  from a second top-level franchise is both.
- `spinoff` ↔ `reboot` — a new-continuity film focused on a side
  character from old continuity (Joker 2019 is arguably exactly
  this).
- null-case-(i) ↔ `spinoff` — a first theatrical film about a side
  character from a non-cinematic-IP franchise. Edge case.

### Orthogonal (intentional)
- `is_prequel` is orthogonal to role — prequels can be mainline
  (Monsters University, Rogue One), spinoff (Prometheus), or reboot
  set early.
- `launches_subgroup` is orthogonal to role — any role can launch
  a subgroup.

### Implication for the redesign
Only four role pairs are cleanly disjoint. Every other pair has a
real overlap with real example films caught in it. The redesign
should either (a) narrow role definitions enough to eliminate
overlaps, (b) make the overlapping regions explicit and provide
deterministic tiebreakers, or (c) replace the closed role enum with
an orthogonal flag structure (first-entry yes/no, continuity-reset
yes/no, prior-story-retold yes/no, etc.) that can represent the
overlaps directly without forcing a single-label commitment.

Option (c) is the most expressive but requires schema and downstream
consumer changes. Options (a) and (b) are prompt-only changes.

## v4 prompt changes scoped (non-role issues only)

These changes do NOT require the role-schema rewrite and should
land in v4 as direct prompt edits:

1. **Fix the Puss in Boots / Fantastic Beasts spinoff cases via
   the spinoff redefinition** (separate task — explicitly deferred
   to the next iteration of the spinoff definition).

2. **`star wars anthology` on Rogue One / Solo** — leave the model
   output as-is. v3 output (`star wars anthology`) is preferred
   over the planned v3 SOT addition (`skywalker saga`). No prompt
   change; update SOT to accept the anthology label and drop the
   skywalker-saga addition for these two films.

3. **Minimal-tier failures** — not a prompt target. Dropping
   minimal from the production tier matrix resolves these without
   prompt changes. Do not attempt to patch the prompt to fix
   minimal's starter over-application or top-level entity drift.

4. **Groups scoring** — update the eval scorer to stop penalizing
   over-inclusion of redundant-but-correct group labels. Only
   penalize missing labels and factually wrong labels.

---

# v4 — final schema, definitions, and decisions

v4 is a schema rewrite (not just a prompt rewrite) that resolves all
three structural issues flagged at the end of the v3 run. It was
designed backward from a downstream-query inventory — "what queries
should this metadata deterministically catch?" — rather than forward
from the existing v3 role vocabulary. The query inventory drove every
subsequent decision.

## Downstream query targets (in-scope)

The v4 metadata must let the search layer deterministically serve
these query shapes from structured fields alone, without
LLM re-reasoning at query time:

1. **Direct lineage lookups.** "Star Wars movies", "James Bond films",
   "Fast and Furious movies", "Harry Potter films". Hits `lineage`.
2. **Shared-universe lookups.** "Marvel movies" / "MCU movies",
   "DCEU films", "Wizarding World films", "MonsterVerse". Hits
   `shared_universe`.
3. **Sub-grouping lookups.** "The Dark Knight trilogy", "MCU Phase 3",
   "Infinity Saga", "Kelvin timeline Star Trek", "Daniel Craig Bond
   movies", "Raimi Spider-Man films", "Disney live-action remakes".
   Hits `recognized_subgroups`.
4. **Narrative-position lookups.** "Sequels to Top Gun", "prequels
   in the Star Wars franchise", "remakes of classic films", "reboots
   of old franchises". Hits `lineage_position`.
5. **Relational attribute lookups.** "Harry Potter spinoffs",
   "crossover films", "Marvel spinoff movies". Hits
   `special_attributes`.
6. **Era-opener lookups.** "Films that launched new phases",
   "openers of cinematic universes". Hits `launches_subgroup`.
7. **Standalone filtering.** "Standalone sci-fi movies",
   "films not part of any franchise", "Spider-Man movies outside the
   MCU". Combinations of `lineage=null`, shared_universe exclusion,
   etc.

### Explicitly out of scope

- **Ordinal position within a franchise** ("first Batman movie",
  "third Fast and Furious"). Rejected as too low-value to generate
  deterministically. If it ever matters, compute at query time from
  release dates + lineage membership.
- **Direct film-to-film relationship pointers** (remake_of → X,
  spinoff_of → Y). Not needed: lineage membership + relational flag
  composes for the same queries. Exact-title lookups like "the 1983
  Scarface remake" are handled upstream by query understanding as
  exact searches, not by this metadata.
- **Thematic anthologies** (Vengeance Trilogy, Three Colors, Before
  trilogy). Out of scope for franchise metadata; served by combining
  concept_tags thematic labels with trilogy detection from other
  sources.
- **Aggregation queries** ("long-running franchises with 10+ films",
  "dead franchises that got rebooted"). Deferred — no per-movie field
  changes needed; revisit when a query-time aggregation layer is
  designed.

## Final schema

Two orthogonal axes. IDENTITY (what brands/groups the film belongs
to) and NARRATIVE POSITION (how the film relates to prior films).
The v3 closed `franchise_role` enum is dissolved: it conflated both
axes into a single slot, which is what produced the unfixable
pair-remake / Iron-Man / reboot-remake problems.

### Fields

| Field | Type | Purpose |
|---|---|---|
| `lineage_reasoning` | str (required) | Chain-of-thought before lineage + shared_universe |
| `lineage` | str \| None | Narrowest recognizable line of films this entry descends from |
| `shared_universe` | str \| None | Broader shared cinematic universe above the lineage, when distinct |
| `subgroups_reasoning` | str (required) | Chain-of-thought before recognized_subgroups + launches_subgroup |
| `recognized_subgroups` | list[str] | Externally-used labels for sub-phases (multi-valued) |
| `launches_subgroup` | bool | True iff earliest-released entry in ≥1 of its recognized_subgroups |
| `position_reasoning` | str (required) | Chain-of-thought before lineage_position |
| `lineage_position` | `LineagePosition` \| None | Mutually exclusive narrative position enum |
| `special_attributes` | list[`SpecialAttribute`] | Orthogonal multi-valued attributes |

### Enums

```python
class LineagePosition(str, Enum):
    SEQUEL  = "sequel"
    PREQUEL = "prequel"
    REMAKE  = "remake"
    REBOOT  = "reboot"

class SpecialAttribute(str, Enum):
    SPINOFF   = "spinoff"
    CROSSOVER = "crossover"
```

No stable integer IDs (the v3 `FranchiseRole` IDs 1–6 were
aspirational and unused — deleted). Enum documentation lives in
`#`-comments above the class in `schemas/enums.py` so it is NOT
shipped into the JSON schema description field sent to the LLM; the
system prompt carries the definitional text instead.

## Definitions

### `lineage`

The NARROWEST recognizable line of films this entry descends from
— the specific character/title/continuity line a user would name
when searching for "all X movies". A lineage needs at least two
entries (or a clear plan for more).

**Semantic flip from v3:** Batman, Spider-Man, Wolverine, Harry
Potter are now the CORRECT lineage values. In v3 these were
explicitly forbidden in favor of broader brands (DC Comics, Marvel).
The flip follows from the query inventory — users search for
"Batman movies", not "DC Comics movies", and the two classes are
meaningfully different.

**IS NOT:** a studio (Disney, Warner Bros.), a shared universe
(MCU — goes in `shared_universe`), a single-work adaptation without
a multi-entry brand, a director's filmography, a thematic
critic-coined label.

**Null case:** standalone films, or films with no multi-entry line
to descend from.

### `shared_universe`

The broader shared cinematic universe ABOVE the lineage, when
distinct. A shared universe hosts multiple distinct lineages that
can reference or cross over with each other. Examples: MCU, DCEU,
Wizarding World, MonsterVerse, Conjuring Universe, Dark Universe.

**Null when:** the lineage is itself top-level with nothing above
it (Star Wars, Fast and Furious, James Bond), or the grouping above
the lineage is a director-era label rather than a formal shared
cosmos (the Nolan Dark Knight trilogy is NOT a shared universe —
that detail goes in `recognized_subgroups`), or `lineage` is null.

**Worked examples:**
- Iron Man → lineage="Iron Man", shared_universe="MCU"
- Batman Begins → lineage="Batman", shared_universe=null
- Harry Potter 1 → lineage="Harry Potter", shared_universe="Wizarding World"
- Godzilla 2014 → lineage="Godzilla", shared_universe="MonsterVerse"
- Star Wars: A New Hope → lineage="Star Wars", shared_universe=null

### `recognized_subgroups`

Multi-valued tag set of externally-used labels for sub-phases of a
lineage or shared universe. Must be labels actually used by studios,
critics, or fans — NEVER invented. Three evidence tiers inherited
from v3: studio/official usage (wizarding world, phase three,
monsterverse, disney live-action remakes), mainstream critical
convention (the dark knight trilogy, raimi trilogy, kelvin timeline,
daniel craig era), widely-used fan terminology (snyderverse,
bayverse, caesar trilogy).

Normalization rules kept verbatim from v3: lowercase, digits as
words ("phase three"), "&" → "and", drop first names on director-era
labels ("raimi trilogy" not "sam raimi trilogy").

IS NOT filters kept verbatim from v3: no restatement of lineage /
shared_universe, no bare sub-series names, no generic trilogy
descriptors ("original trilogy" valid ONLY for Star Wars), no
on-the-spot invented labels. EMPTY-THEN-ADD framing kept.

**Empty list is the most common outcome.**

### `launches_subgroup`

True if and only if this film is the earliest-released entry in
AT LEAST ONE of its `recognized_subgroups`. Must be false if
`recognized_subgroups` is empty.

**Critical design note:** this is NOT derived from narrative-era
triggers (time jump, generational protagonist shift, new conflict
framework). An earlier side discussion proposed a 5-trigger
"Narrative Era" framework, and we rejected it because:

1. It undercounts MCU phase transitions — Captain America: Civil
   War has no time jump, no generational shift, no new external
   threat, scores 1/5, but is widely accepted as the Phase 3
   opener.
2. It overcounts time-jumped sequels that are NOT era openers —
   Jurassic World: Dominion would score 3/5 but nobody treats it
   as the start of anything.
3. It conflates the proxy (narrative discontinuity) with the
   thing we actually care about for search (received cultural
   labels). The ground truth is external: if the culture has
   named the phase, it matters; if not, no query will find it
   anyway.

The coupling to `recognized_subgroups` gives the right answer by
construction: Force Awakens launches "sequel trilogy", Iron Man
launches "phase one" / "infinity saga" / "iron man trilogy", Creed
launches "creed trilogy". Captain America: Civil War launches
"phase three". All deterministic from the subgroup list + release
dates.

**Worked examples:**
- TRUE: Iron Man (2008), Batman Begins (2005), The Force Awakens
  (2015), Casino Royale (2006), Star Trek (2009), Creed (2015),
  Captain America: Civil War (2016), The Phantom Menace (1999).
- FALSE: Avengers: Endgame (closes infinity saga), Jurassic World:
  Dominion (third of the trilogy), Spider-Man: Homecoming (Civil
  War already launched Phase 3), Return of the King, standalones.

### `lineage_position`

Mutually exclusive enum. Describes how THIS film relates to prior
films in its lineage.

| Value | Definition |
|---|---|
| `sequel` | Continues an existing continuity forward. Includes legacy sequels where a prior protagonist returns (Creed, Top Gun: Maverick, Blade Runner 2049, Halloween 2018, Ghostbusters: Afterlife). |
| `prequel` | Set chronologically before an earlier-released film in the same lineage, with shared continuity. The Hobbit, Rogue One, Monsters University, Prometheus. Reboots set in the past are NOT prequels. |
| `remake` | Retells the core story of a specific prior film with fresh production. Same story spine, same main beats, different cast/period. **Legal with `lineage=null`** for pair-remakes (Scarface 1983, Cape Fear 1991, True Grit 2010, Psycho 1998, The Lion King 2019, Cinderella 2015, Mulan 2020, The Jungle Book 2016). |
| `reboot` | Restarts an existing lineage's continuity with a NEW story. Same characters and IP, fresh continuity, new plot spine. Batman Begins, Casino Royale, The Amazing Spider-Man, Star Trek (2009), Ghostbusters (2016), Tomb Raider (2018), RoboCop (2014). |
| null | First entry in the lineage (Iron Man 2008, The Fellowship of the Ring), or standalone. |

**Remake vs reboot tiebreaker:** if the new film RETELLS a specific
prior film's story spine → remake. If it introduces a NEW story
with the same IP and characters → reboot. Jungle Book 2016 retells
the 1967 story → remake. Batman Begins tells a new origin →
reboot.

**`lineage=null, lineage_position="remake"` is LEGAL.** This is the
single cleanest resolution to v3's Scarface problem: pair-remakes
have a clean inter-film relationship without forming a multi-entry
brand. The position field captures the relationship even when no
franchise exists. The null-propagation rule in `validate_and_fix`
is deliberately partial — `lineage_position` and
`special_attributes` survive when `lineage` is null.

### `special_attributes`

Multi-valued enum array (not separate booleans). Orthogonal to
`lineage_position` and to each other.

#### `spinoff`

Three-constraint test kept VERBATIM from v3 because v3 showed it
was working:

- **(a) MINOR IN SOURCE** — the expanded character/element/subplot
  was minor in the source work, not a lead, co-lead, or planned
  shared-universe pillar character.
- **(b) GOES SOMEWHERE NEW** — new protagonist, new story spine,
  new setting focus, new POV.
- **(c) LEAVES THE SOURCE BEHIND** — the source film's main
  characters and main plot are not the focus here. Legacy sequels
  that bring the prior protagonist along are NOT spinoffs.

**FIRES:** Rogue One, Solo, Puss in Boots (2011), Venom,
Prometheus, Maleficent, Penguins of Madagascar, The Scorpion King,
Joker (2019) — the last is legal with `lineage=null` and no DCEU
continuity.

**DOES NOT FIRE:** Wonder Woman (planned DCEU pillar — fails a),
Black Panther (major role in Civil War — fails a), Captain Marvel
/ Doctor Strange / Thor (planned MCU pillars — fail a), Creed
(Rocky is a major returning co-lead — fails c), Ghostbusters:
Afterlife (legacy sequel — fails c), Top Gun: Maverick (Maverick
is still the lead — fails a).

#### `crossover`

Two or more distinct top-level lineages combined into a single
film. Freddy vs Jason, Alien vs Predator, Godzilla vs Kong, Batman
v Superman, Space Jam, Who Framed Roger Rabbit.

**IS NOT a crossover:** a shared-universe team-up within a single
top-level brand (The Avengers is not a crossover — all the heroes
are within the same Marvel top-level brand), a film that references
characters from another lineage without them appearing meaningfully.

Co-occurrence: can combine with any `lineage_position` and with
`spinoff` simultaneously. A film that pulls a side character from
franchise A and combines them with characters from franchise B
carries both flags.

## Validator fixups

`FranchiseOutput.validate_and_fix()` applies three deterministic
silent corrections so the batch pipeline never persists internally
inconsistent records:

1. **Partial null-propagation from `lineage`.** If `lineage` is
   null: clear `shared_universe`, `recognized_subgroups`,
   `launches_subgroup`. Deliberately preserve `lineage_position`
   and `special_attributes` — pair-remakes (Scarface 1983) and
   standalone spinoff-flavored films (Joker 2019) are legitimate
   use cases with `lineage=null`.
2. **`launches_subgroup` ⇄ `recognized_subgroups` coupling.** If
   `launches_subgroup=true` but groups list is empty, force the
   boolean to false. The prompt enforces this too; the validator
   is a cheap belt-and-suspenders.
3. **`special_attributes` dedup.** Remove duplicate enum values
   while preserving order.

All corrections are silent. `result_processor.py` re-serializes
the validated object before persisting, so silent fixups keep the
pipeline flowing on single-row inconsistencies instead of
hard-failing.

## Worked examples — full field values

These are the canonical "acceptance test" cases for v4. Any future
eval run should verify the schema produces these exact values.

| Film | lineage | shared_universe | recognized_subgroups (partial) | launches_subgroup | lineage_position | special_attributes |
|---|---|---|---|---|---|---|
| Iron Man (2008) | Iron Man | MCU | [phase one, infinity saga, iron man trilogy] | true | null | [] |
| Captain America: Civil War (2016) | Captain America | MCU | [phase three, infinity saga] | true | sequel | [] |
| Spider-Man: Homecoming (2017) | Spider-Man | MCU | [phase three, infinity saga] | false | null | [] |
| Avengers: Endgame (2019) | Avengers | MCU | [phase three, infinity saga] | false | sequel | [] |
| The Dark Knight (2008) | Batman | null | [the dark knight trilogy] | false | sequel | [] |
| Batman Begins (2005) | Batman | null | [the dark knight trilogy] | true | reboot | [] |
| Joker (2019) | null | null | [] | false | null | [spinoff] |
| Wonder Woman (2017) | Wonder Woman | DCEU | [] | null | null | [] |
| Creed (2015) | Rocky | null | [creed trilogy] | true | sequel | [spinoff] |
| Scarface (1983) | null | null | [] | false | remake | [] |
| The Jungle Book (2016) | null | null | [disney live-action remakes] (if also on the film) | varies | remake | [] |
| Rogue One (2016) | Star Wars | null | [star wars anthology] | varies | prequel | [spinoff] |
| Solo (2018) | Star Wars | null | [star wars anthology] | false | prequel | [spinoff] |
| The Force Awakens (2015) | Star Wars | null | [sequel trilogy, skywalker saga] | true | sequel | [] |
| Casino Royale (2006) | James Bond | null | [daniel craig era] | true | reboot | [] |
| Puss in Boots: The Last Wish (2022) | Puss in Boots | null | [] | false | sequel | [spinoff] |
| Freddy vs Jason (2003) | A Nightmare on Elm Street | null | [] | false | sequel | [crossover] |
| Fellowship of the Ring (2001) | The Lord of the Rings | null | [] | false | null | [] |
| The Phantom Menace (1999) | Star Wars | null | [prequel trilogy, skywalker saga] | true | prequel | [] |

Notes on the table:
- "(if also on the film)" indicates a judgment call — Jungle Book
  2016 may or may not carry the Disney live-action remakes label
  depending on whether the model classifies it as part of that
  cohort. Both are defensible; the eval should accept either.
- "Wonder Woman recognized_subgroups" is noted as `[]` rather than
  e.g. "DCEU Phase 1" because the DCEU does not have publicly-used
  phase labels the way the MCU does.
- "Rogue One launches_subgroup" depends on release order within
  the Star Wars anthology label. Solo is the only other anthology
  entry at time of this doc; Rogue One (2016) precedes Solo
  (2018), so Rogue One → true, Solo → false.

## How v4 resolves each v3 open issue

| v3 open issue | v4 resolution |
|---|---|
| Pair-remakes cannot be represented (Scarface 1983) | `lineage_position="remake"` is legal with `lineage=null`. Validator preserves it under partial null-propagation. |
| Iron Man / X-Men 2000 null-case contradiction | Dissolved. Iron Man cleanly carries `lineage="Iron Man"`, `shared_universe="MCU"`, `lineage_position=null`. No role-enum choice needed; the first-in-lineage case is just the null value of `lineage_position`. |
| Reboot ↔ remake overlap (Jungle Book 2016) | Schema-level mutual exclusivity in `LineagePosition` enum forces a choice. Explicit tiebreaker in the prompt: retells a specific prior film's story spine → remake; new story with same IP → reboot. Jungle Book 2016 → remake. Batman Begins → reboot. |
| Spinoff over-application to pillar films (Wonder Woman, Black Panther, Creed) | The three-constraint spinoff test from v3 is preserved verbatim. The v4 change is that spinoff moves OUT of the role enum and INTO `special_attributes`, so it no longer competes with "mainline" for a single slot — Wonder Woman carries `lineage_position="sequel"` (or null, debatable) with `special_attributes=[]`, cleanly. |
| Missing-over-wrong null-role bias (Guardians of the Galaxy, Rurouni Kenshin) | Role enum is gone. The first-entry case is now the null value of `lineage_position`, which is unambiguous. Guardians is a sequel to the team-up films or a null if treated as the first of its own lineage — either is defensible, neither is the v3 pathological null. |
| Minimal-tier top-level entity violations (name='batman', name='x-men', name='spider-man') | Dropped from the tier matrix per v3 recommendation. Not a prompt target. |
| Groups over-inclusion scored as a failure | Scoring methodology updated: only missing labels or factually wrong labels count against the candidate. Redundant-but-correct labels are not penalized. |

## Schema-design decisions worth preserving

These decisions are load-bearing for v4's weak-model performance
and should not be casually changed:

1. **Enums over boolean clusters for mutually exclusive choices.**
   Schema-level exclusivity means the model physically cannot emit
   `sequel=true, prequel=true`. The v3 overlap problems are
   unreachable by construction. See Instructor / BAML / OpenAI
   structured outputs guidance on enum usage.
2. **Small enum array for orthogonal flags** (`special_attributes`).
   Invites enumeration as a single decision and gives a cleaner
   empty default than two separate booleans. Matches the
   concept_tags precedent for grouped-array-of-tags over flat
   boolean lists.
3. **Scoped reasoning fields before each hard decision block.**
   Three reasoning fields (identity, subgroups, position) instead
   of one top-level reasoning field. Keeps reasoning adjacent to
   the commitment it informs rather than going stale. Matches
   Jason Liu's "just-in-time reasoning" pattern in Instructor.
4. **Field order mirrors dependency order.** `lineage` →
   `shared_universe` → `recognized_subgroups` → `launches_subgroup`
   → `lineage_position` → `special_attributes`. Each field
   conditions on all prior fields via the autoregressive token
   stream.
5. **Compact field descriptions, heavy prompt.** Field descriptions
   in `FranchiseOutput` are intentionally short — they define each
   field but do not carry worked examples or procedures. The
   system prompt carries the definitional weight. Total
   per-property description chars ≈ 1,875 (down from ~9,400 in
   the first v4 draft).
6. **Enum documentation in comments above the class, not
   docstrings.** Python class docstrings on Pydantic enum classes
   ship to the LLM via the generated JSON schema. Moving the
   documentation to `#`-comments above the class keeps it for
   human readers without sending it to the model. See
   `schemas/enums.py` for the convention.
7. **Silent validator fixups, not hard failures.** Internal
   consistency errors from the LLM are corrected silently at
   parse time. The batch pipeline re-serializes the validated
   object before persisting, so silent correction keeps single-
   row inconsistencies from blowing up the run.

## Test acceptance criteria for the next eval

Before the v4 prompt is considered validated, the next eval run
must verify:

1. **Every film in the "Worked examples" table produces its
   expected field values** across all tiers in the test set (with
   the noted judgment-call cases accepted either way).
2. **No film produces `launches_subgroup=true` with an empty
   `recognized_subgroups` list.** Validator enforces this, but
   eval should count pre-fixup violations as a prompt-quality
   signal.
3. **No film produces `shared_universe` populated with
   `lineage=null`.** Same — validator enforces it but pre-fixup
   count matters.
4. **The spinoff three-constraint test fires correctly** on all
   the canonical positive and negative cases (Rogue One, Solo,
   Puss in Boots, Venom, Joker vs Wonder Woman, Black Panther,
   Creed, Top Gun: Maverick).
5. **The remake vs reboot tiebreaker fires correctly** on Jungle
   Book 2016, Cinderella 2015, Mulan 2020 (remake) vs Batman
   Begins, Casino Royale, The Amazing Spider-Man, Star Trek 2009
   (reboot).
6. **Pair-remakes produce `lineage=null, lineage_position="remake"`**
   on Scarface 1983, True Grit 2010, Cape Fear 1991, Psycho 1998.
7. **Standalone spinoff-flavored films produce `lineage=null,
   special_attributes=["spinoff"]`** on Joker 2019.

Target numbers to beat: the v3 run's medium/low tiers landed at
~80% correct, ~4% H, ~86% effective after scoring-methodology
corrections. v4 should meet or exceed these on the equivalent
test set.

## Implementation locations

- Schema: `schemas/metadata.py::FranchiseOutput` + enums in
  `schemas/enums.py::LineagePosition`, `::SpecialAttribute`
- Prompt: `movie_ingestion/metadata_generation/prompts/franchise.py::SYSTEM_PROMPT`
- Generator: `movie_ingestion/metadata_generation/generators/franchise.py::generate_franchise`
- Validator: `FranchiseOutput.validate_and_fix` (called from
  `result_processor.py` at parse time)

The batch pipeline (`generator_registry.py`, `request_builder.py`,
`result_processor.py`, `pre_consolidation.py`) required zero
changes — everything references `FranchiseOutput` by class name
(preserved) or `config.schema_class` generically.

## Deferred / follow-up

- **Re-run the eval against v4.** Existing `franchise_test_results.json`
  and `test_franchise.ipynb` are stale against the new schema.
  Update the eval harness in a separate task.
- **Regenerate all franchise metadata.** Existing rows in the
  SQLite `generated_metadata.franchise` column are in v3 format.
  Next batch run will overwrite per movie; recommend re-running
  the full franchise generator for all movies after v4 lands.
- **Downstream integration.** `FranchiseOutput` is NOT currently
  consumed by `vector_text.py` (which reads
  `SourceOfInspirationOutput.franchise_lineage` instead) or
  `ingest_movie.py` or `db/lexical_search.py` (which operates on
  source_material phrases). When v4 is wired into vector text or
  Postgres columns for structured search, that is a separate
  change.

---

# v5 → v6 → v7 — iterations after the first v4 eval run

After v4 shipped, we ran the eval harness against `gpt-5-mini`
(medium and low reasoning) and `gpt-5.4-mini` (low reasoning), 3
samples per candidate, 79 movies. The results exposed three
distinct classes of problem that v5, v6, and v7 address in
sequence. The schema and decision rules have stayed essentially
constant since v4; every release since has been about making the
*reasoning* the model does BEFORE committing to a value faithful
to the decision rules the prompt already describes.

## v5 — normalization, shape B shared_universe, launched_franchise

Problems v4 eval surfaced:

1. **Name normalization leaked compact forms into lineage/shared_universe.**
   Labels like "MCU", "DC Extended Universe" (mixed case), and
   "The Dark Knight Trilogy" were landing in the emitted fields.
   Normalization was only enforced inside `recognized_subgroups`,
   so lineage and shared_universe drifted away from the canonical
   forms users search for.
2. **Spinoff sub-lineages broke the parent link.** Puss in Boots,
   Minions, and Logan cleanly carry their own narrowest lineage,
   but v4 forced `shared_universe=null` for them because v4 only
   recognized formal studio-coined cosmos. The connection between
   the spinoff sub-lineage and its parent franchise was lost.
3. **"Movies that launched a franchise" had no signal.** v4 could
   answer "first entry in lineage" via `lineage_position=null`,
   but that flag fires on Iron Man (2008) too — which did NOT
   launch a franchise (Marvel as a franchise already existed).

v5 deltas:

- **GLOBAL OUTPUT RULES** block at the top of the prompt enforces
  one normalization rule across lineage, shared_universe, and
  every subgroup label. Worked examples rewritten to use
  canonical expanded forms ("marvel cinematic universe", "dc
  extended universe").
- **shared_universe gains Shape B.** When the film's lineage is
  itself a spinoff sub-lineage of a well-known broader franchise,
  shared_universe now carries the parent franchise name
  (puss in boots → shrek, minions → despicable me, logan → x-men).
  v4 outright rejected this shape.
- **NEW FIELD 7 `launched_franchise`** with a four-part test:
  (1) first cinematic entry in its lineage,
  (2) not a spinoff,
  (3) source-material recognition test ("do audiences know this
      as a film franchise, or as the source material that
      preceded it?"),
  (4) relevant follow-ups test (sequels/prequels/spinoffs
      audiences widely recognize).
  Distinct from `launched_subgroup` — Iron Man fires
  `launched_subgroup=true` but `launched_franchise=false`;
  Shrek (2001) fires both `launched_franchise=true` and
  `launched_subgroup=false`.
- **`launches_subgroup` renamed to `launched_subgroup`** for
  tense consistency with `launched_franchise`.
- **Anti-restatement rule** for recognized_subgroups now carves
  out disambiguating qualifiers — "connery bond era" is a valid
  subgroup even though "bond" is the lineage, because the
  qualifier picks out a narrower slice of the lineage.

## v6 — Shape B lineages for brand-backed single films + explicit crossover reasoning

The v5 eval showed two more problem clusters.

1. **Brand-backed single-film adaptations had no lineage.** v5
   required a lineage to be either a multi-film cinematic line
   (Shape A) or a spinoff sub-lineage of a broader film
   franchise. This forced `lineage=null` for Detective Pikachu,
   Barbie, The Super Mario Bros. Movie, Sonic the Hedgehog, and
   similar films whose parent brand is a dominant ongoing
   franchise in another medium. A user searching "pokemon
   movies" or "barbie movies" or "mario movies" would not match.
2. **Crossovers collapsed to a first-match shortcut.** v5 had no
   reasoning field for `special_attributes`, so the crossover
   check ran as an in-the-head inference. Space Jam-style cases
   where one crossing parent is NON-cinematic (Looney Tunes ×
   NBA, video games × film, sports × cartoon) were silently
   dropped because the prompt implied "lineages" meant film
   lineages.

v6 deltas:

- **Shape B lineage.** A single theatrical film can anchor a
  lineage when its parent brand is a dominant ONGOING non-
  cinematic franchise — video game series, toy line, long-running
  TV/anime/cartoon, serial comic book line, trading card game.
  Prince of Persia (2010) → prince of persia; Warcraft (2016) →
  warcraft; The Angry Birds Movie (2016) → angry birds. v5
  incorrectly forced null for all of these.
- **FABRICATION GUARD relaxed** to accept either (a) another
  specific film in the line (Shape A evidence) OR (b) a named
  ongoing non-cinematic parent brand with its medium (Shape B
  evidence). A bounded single literary work does NOT satisfy (b)
  — single-novel and classic-lit adaptations (Pride and Prejudice,
  Great Expectations, Little Women, Emma) remain `lineage=null`.
- **NEW FIELD 6 reasoning slot: `special_attributes_reasoning`.**
  Forces an explicit walkthrough of the spinoff three-constraint
  test AND a crossover defining-trait test ("would removing
  either parent make this a fundamentally different film?").
- **Crossover definition explicitly accepts non-cinematic
  franchises as crossing parents.** Games, TV, toys, sports,
  music, and comics all count. v5 implied film-only.

## v6 eval findings

Re-ran the harness against v6 with the same candidate matrix.
Evaluation rule: only errors *genuinely harmful to the search
system* count. Plausible extra subgroups, normalization
variants, and defensible SOT disagreements were excluded.

Seven SOT bugs were fixed directly in the test notebook after
this evaluation:

| TMDB ID | Title | Fix |
|---|---|---|
| 259316 | Fantastic Beasts and Where to Find Them | `lineage_position: "prequel"` → `None` (first entry in its own lineage; prequel-to-HP lives on the shared_universe axis, not the lineage_position axis) |
| 211672 | Minions (2015) | `lineage_position: "prequel"` → `None` (first entry in the minions sub-lineage) |
| 346698 | Barbie (2023) | `lineage: None` → `"barbie"` (Shape B) |
| 303857 | Dragon Ball Z: Resurrection F | `lineage: "dragon ball"` → `"dragon ball z"`, `shared_universe: None` → `"dragon ball"` |
| 396535 | Train to Busan | `launched_franchise: False` → `True` (Peninsula exists as a recognized follow-up) |
| 9410 | Great Expectations (1998) | `lineage_position: None` → `"remake"` |
| 331482 | Little Women (2019) | `lineage_position: None` → `"remake"` |

Six remaining harmful patterns were flagged as prompt-quality
problems that v7 needs to address, not SOT bugs:

- **Solo (2018): `lineage_position=null` instead of `prequel`.**
  The model reasoned "technically a prequel but really a spinoff"
  and ended up committing to null instead of populating both
  axes. Symptom of the first-match scan inside field 5 + a
  confusion about whether spinoff demotes lineage_position.
- **Rogue One (2016): same pattern intermittently.** Sometimes
  correctly populated as prequel + spinoff, sometimes dropped to
  null for the same reason as Solo.
- **Space Jam (1996): crossover silently dropped, lineage
  sometimes "looney tunes" and sometimes null.** The model did
  not walk the two crossing parents (NBA / real basketball ×
  Looney Tunes) because the prompt did not force it to enumerate
  recognizable franchises of any medium.
- **Detective Pikachu / Sonic / Aladdin 2019 / Lion King 2019:**
  lineage Shape B decisions depended on whether the model
  remembered Shape B applied at step 2 of the v6 procedure. When
  the model framed step 2 as "is there a multi-film line?" it
  answered "no" and never reached the Shape B branch.
- **Transformers (2007) `lineage_position` variance.** The model
  bounced between null, reboot, and sequel because the four-
  value first-match list in field 5 short-circuited on whichever
  candidate was mentioned first in the model's own reasoning.
- **Launched_subgroup instability around unnamed trilogies.**
  The model occasionally invented a trilogy label just to have
  something to put in `recognized_subgroups`, then set
  `launched_subgroup=true` off the invented label.

## v7 — FACTS → DECISION rewrite of every reasoning field

The v6 failures were not about rules. The rules (three-constraint
spinoff test, defining-trait crossover test, remake-vs-reboot
tiebreaker, four-part launched_franchise gate) were already
correct. The problem was that the reasoning PROCEDURE blocks
jumped straight into evaluating candidate values without first
writing down the facts the evaluation depends on — and for
field 5 specifically, the opening instruction ("build the case
for each candidate, then commit to the strongest") was
contradicted by a numbered list that was literally a first-match
top-to-bottom scan.

v7 rewrites every reasoning PROCEDURE into a FACTS → DECISION
shape that mirrors the `concept_tags` ENDINGS pattern: gather the
facts the decision depends on first ("do not interpret yet"),
then read the answer off the facts.

Per-field changes:

- **FIELD 1+2 (`lineage_reasoning`) — Brand-level audit.**
  Walks upward from the film across four levels (Shape A
  lineage, Shape B lineage, Shape A shared_universe, Shape B
  shared_universe) and asks a single question at each level:
  "is there a line/parent here that a general audience would
  genuinely recognize and search for?" Records either a specific
  name or "none at this level." The audit is recognizability-
  gated — the model is explicitly told NOT to list weak or
  speculative candidates (seeding the model with weak options
  biases it toward picking one when the correct answer is none).
  Shape B is reachable from the audit without depending on
  whether step 2 initially framed the question as "multi-film
  line."
- **FIELD 3+4 (`subgroups_reasoning`) — Usage-gated sub-phase
  audit.** Asks whether any named sub-phases inside the brand
  are actually in real-world use by studios, mainstream critics,
  or widely-used fan terminology. "No recognized sub-phases" is
  explicitly named as the correct and most common outcome. Do
  NOT brainstorm labels you could plausibly invent.
- **FIELD 5 (`position_reasoning`) — the critical fix.** The
  old first-match numbered list is gone. Replaced with five
  facts (prior-films inventory, continuity relationship, story-
  spine relationship, in-universe chronology, protagonist
  continuity) and a decision table that maps fact combinations
  to enum values. The decision table explicitly states that
  "set BEFORE an earlier-released film in the same lineage →
  prequel, and a spinoff flavor does NOT demote this to null
  because field 6 captures spinoff independently on a different
  axis." Solo now resolves cleanly as `prequel` + `[spinoff]`.
- **FIELD 6 (`special_attributes_reasoning`) — explicit facts
  block.** Six facts (protagonist, protagonist's role in the
  ORIGINAL source, prior-lineage leads, prior leads in this
  film, plot engine, recognized franchises this film touches
  from any medium). The spinoff three-constraint test and the
  crossover defining-trait test are now mechanical reads of
  those facts. F6 (recognized franchises) is the key fix for
  Space Jam — the model cannot skip writing down "looney tunes"
  and "nba/real basketball" and then fail to consider their
  pairing.
- **FIELD 7 (`launch_reasoning`) — fact-block before the gate.**
  Five facts (lineage_position, spinoff flag, source material,
  source-vs-film cultural dominance, follow-up films and their
  audience recognition) before the four-part gate short-circuits.
  Tests 1 and 2 are trivial reads; tests 3 and 4 now have their
  evidence written down before the gate evaluates.

Decision-rule content (value definitions, IS NOT filters,
FABRICATION GUARD, planned-pillar carve-out, remake-vs-reboot
tiebreaker, spinoff three-constraint test, crossover defining-
trait test, four-part launched_franchise gate) is unchanged from
v6. v7 is ordering-and-framing only.

## What the next eval run should verify

The next run should use the same harness as v6 (79 movies × 3
reasoning tiers × 3 samples) with the v7 prompt. The goal is NOT
to beat a number — it is to confirm that the v6 failure patterns
have specific mechanical fixes and that v4–v6 wins did not
regress.

### Primary: did v7's FACTS → DECISION rewrite fix the v6 failures?

1. **Solo (2018) resolves as `lineage_position="prequel",
   special_attributes=["spinoff"]` on every sample** across all
   three reasoning tiers. This is the canonical test of the
   field 5 rewrite — the model should never again drop to null
   because it noticed the spinoff flavor.
2. **Rogue One (2016) resolves the same way — prequel +
   spinoff on every sample.** No intermittent drops to null.
3. **Space Jam (1996) fires `special_attributes=["crossover"]`
   on every sample.** The field 6 facts block should force the
   model to enumerate Looney Tunes and NBA/basketball as
   recognizable franchises of different mediums before running
   the defining-trait test. lineage can be either "looney tunes"
   or null (pick the dominant parent) — both are defensible.
4. **Transformers (2007) resolves `lineage_position=null`
   consistently** (first entry in the film lineage; the toy and
   cartoon franchise is the dominant cultural form, which also
   forces `launched_franchise=false` via test 3).
5. **Shape B lineages populate reliably on every sample**:
   Detective Pikachu → `pokemon`, Sonic the Hedgehog → `sonic
   the hedgehog`, The Super Mario Bros. Movie → `super mario`,
   Barbie → `barbie`. The brand-level audit in field 1+2 should
   no longer depend on whether the model frames step 2 as
   "multi-film line."
6. **`launched_subgroup` stability.** No candidate should fire
   `launched_subgroup=true` off an invented trilogy label that
   is not in `recognized_subgroups` for a culturally recognized
   reason (the field 3+4 audit's explicit "no recognized sub-
   phases" outcome should absorb these cases).

### Secondary: did v6 wins stay intact?

7. **launched_franchise on original-screenplay starters.**
   Jurassic Park (1993), The Matrix (1999), Shrek (2001), Mad
   Max (1979), Saw (2004), Toy Story (1995), The Godfather
   (1972), Die Hard (1988), Back to the Future (1985) — all
   should fire `launched_franchise=true` via all four gates
   passing.
8. **launched_subgroup vs launched_franchise distinction.**
   Iron Man (2008), Batman Begins (2005), Casino Royale (2006)
   — all should fire `launched_subgroup=true,
   launched_franchise=false` because the parent franchise
   pre-exists the film. This is the single most important
   boundary in field 7.
9. **Spinoff three-constraint test accuracy.** Creed, Puss in
   Boots: The Last Wish, Logan — spinoff fires or not per the
   SOT in the notebook (SOT fixed last round where needed).
   Top Gun: Maverick, Ghostbusters: Afterlife, The Force
   Awakens, Blade Runner 2049, Halloween 2018 — spinoff should
   NOT fire; all fail constraint (c) (prior hero's legacy is
   the plot engine).
10. **Planned-pillar carve-out.** Wonder Woman (2017), Black
    Panther (2018) — spinoff should NOT fire even though the
    three constraints technically hold, because the carve-out
    blocks pillar debuts.

### Tertiary: signals that the FACTS → DECISION pattern is working

These are process signals, not output signals. Scan a sample of
reasoning strings and check:

11. **Field 5 reasoning strings should cite specific facts
    (F1–F5) before naming an enum value.** A model that still
    writes "this film is chronologically before A New Hope so
    → prequel" without a continuity/story-spine fact first is
    still doing first-match scanning under a thin facts skin.
12. **Field 6 reasoning strings should explicitly name every
    recognizable franchise the film touches BEFORE running the
    crossover test.** For any film that fires crossover, both
    parents should appear as named F6 entries. For any film
    that does not fire crossover, F6 should end with "single
    franchise, crossover short-circuits" or similar.
13. **Field 1+2 reasoning strings should include at least one
    "none at this level" for most films** (since standalone is
    the majority outcome). If every film's audit produces four
    populated brand levels, the recognizability gate is not
    biting and we need to tighten the prompt.

### What failure in the next run would mean

- **If Solo still drops to null intermittently**, the field 5
  rewrite did not go far enough — the decision table needs to
  be even more explicit that spinoff and lineage_position live
  on orthogonal axes.
- **If Space Jam still misses crossover**, the F6 facts block
  is not sticky enough — we may need to add a sentence that
  explicitly asks "name TWO franchises, or state that only one
  exists" to force enumeration.
- **If Detective Pikachu or Sonic reverts to `lineage=null`**,
  the brand-level audit is failing at level (ii) and Shape B
  is not being reached. We'd need to reorder the audit so Shape
  B is checked before Shape A, or add a recognizability anchor
  (e.g., require naming the parent brand's medium explicitly).
- **If "none at this level" almost never appears in field 1+2
  reasoning strings**, the no-weak-candidate guardrail is being
  ignored and we need a harder instruction ("if you are not
  ≥90% confident a general audience would recognize this brand,
  write 'none at this level'").

## Deferred to after the v7 eval

- **Wiring `FranchiseOutput` into `vector_text.py` and
  lexical_search.** Still deferred from v4. Once v7 eval is
  green, this is the next integration step — currently
  `vector_text.py` reads `SourceOfInspirationOutput.franchise_lineage`
  which is the old single-field representation.
- **Regenerating franchise metadata for all movies.** Already
  noted as a deferred item under v4; still deferred. The
  regeneration should happen after v7 eval validates and
  ideally after the vector_text integration, so we only pay the
  LLM cost once.
- **Cross-checking `launched_franchise` at scale.** The eval
  harness only exercises 79 movies. Once v7 lands, pick a
  random sample of ~50 films with `launched_franchise=true`
  from the full corpus and spot-check against the four-part
  test. The gate is strict enough to be worth auditing.
