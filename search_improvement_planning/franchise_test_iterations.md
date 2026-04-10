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
