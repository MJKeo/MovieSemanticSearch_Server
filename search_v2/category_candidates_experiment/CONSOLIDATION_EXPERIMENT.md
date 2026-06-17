# Step 3 Consolidation Redesign — Experiment

Tests the Step 3 trait-decomposition redesign that moves the
minimum-viable-set discipline upstream into an explicit consolidation
step, adds a per-candidate `fit` label, gates category eligibility on
inclusion-only framing, and reframes SENSITIVE_CONTENT / MATURITY_RATING.

## Result sets (in `results/`, 5 repeats per query)

| Prefix | Model | Prompt/schema |
|--------|-------|----------------|
| `base` | Gemini 3.5 Flash (thinking minimal, temp 0.15) | category edits only, OLD step-3 prompt/schema |
| `fix_gemini` | Gemini 3.5 Flash (same) | FULL redesign applied |
| `fix_gpt` | gpt-5.4-mini (reasoning low, verbosity low) | FULL redesign applied |

`base` is the reference. `fix_*` differ from `base` only in the step-3
prompt + schema (and model, for `fix_gpt`). Step 2 output is held fixed
across all three (`step_2_results.json`).

## Test set (12 queries, in `queries.py` as `QUERIES`)

- **Diagnostic (5):** the original failure-mode queries (plot-shape
  fragmentation, family-friendly avoidance, serbian audio-language).
- **Edge (4):** `scary horror movie that isn't too gory` (neg-polarity
  content), `japanese horror films` (nationality+genre), `movie about a
  washed-up boxer making a comeback` (single story shape), `clean comedy
  with no swearing` (audience/content inclusion).
- **Regression guards (3):** `movies starring Tom Hanks` (entity),
  `Oscar winning war films from the 1990s` (awards+genre+era),
  `hidden gem thrillers` (multi-facet figurative — over-consolidation guard).

## Rerun

```
python -m search_v2.category_candidates_experiment.run_step_2_batch        # regenerate fixed Step 2 input
python -m search_v2.category_candidates_experiment.run_step_3_batch <prefix>
python -m search_v2.category_candidates_experiment.summarize base fix_gemini fix_gpt
```

`fix_gpt` requires temporarily setting `_PROVIDER/_MODEL/_MODEL_KWARGS`
in `search_v2/step_3.py` to OpenAI / gpt-5.4-mini / {reasoning low,
verbosity low}; restore Gemini after.

## Findings (5-rep distributions)

**Improvements vs base**
- Plot shapes consolidate toward one call: `loser guy gets hot girl`
  SOLO 0/5→3/5 (gemini), `pursues popular girl` SOLO 0/5→5/5 (gpt,
  romance folded into the single call's intent), `washed-up boxer`
  SOLO 1/5→3/5 (both).
- Family-friendly avoidance category: gemini drops SENSITIVE_CONTENT
  entirely (4/5→0/5), using a MATURITY_RATING proxy + TARGET_AUDIENCE.
- `gory` trait no longer framed as "avoid gory content" (was an
  avoidance violation in base; presence-framed in both fix sets).
- `clean comedy` de-fragments (base 5/5 three-facet → fewer facets).
- Over-consolidation guard held: `hidden gem` stayed multi-call (not
  collapsed to SOLO) in 5/5 gemini, 4/5 gpt.
- Pure regression guards stable: Tom Hanks / Oscar / 1990s all 5/5 SOLO
  unchanged across all three.

**Residual / not fixed**
- **Audio-language trap NOT fixed:** `serbian` / `serbian movies` still
  commit AUDIO_LANGUAGE alongside COUNTRY_OF_ORIGIN in both fix sets
  (gemini 5/5; gpt 5/5, 2/5 of which as FACETS — a worse PRODUCT fold).
  Inclusion + granularity gates don't catch it (Serbian audio is a
  valid presence and not an identity entity); the model genuinely
  rationalizes language as a facet. Needs a category-definition lever
  (AUDIO_LANGUAGE boundary: bare-nationality phrasing favors
  origin/tradition over language), not a step-3 reasoning change.
- **gemini regressed `thrillers`** (regression guard) from 5/5 SOLO
  Genre → 5/5 FACETS[Genre, Emotional]: it split a bare genre into
  genre + tone and PRODUCT-folded. gpt kept it 5/5 SOLO Genre. The
  aspect-partition emphasis can over-split atomic genre words for gemini.
- gpt still reaches for SENSITIVE_CONTENT on family-friendly 3/5
  (framed as presence, not avoidance — so not the original bug, but a
  questionable category choice vs gemini's MATURITY_RATING proxy).
- Minor inclusion residue: gemini family-friendly MATURITY_RATING intent
  says "free of mature elements" (absence framing); both models
  occasionally narrate system mechanics ("negative polarity filter",
  "Step 4") in retrieval_intent despite that being stripped from guidance.

**Gemini vs gpt-5.4-mini**
- gpt consolidates more decisively (higher SOLO rates on pursues,
  for-kids, comedy; correct SOLO on thrillers). Closer to the
  minimum-calls goal. Occasionally picks a non-ideal single anchor
  (Character archetype vs Story/thematic for a romance premise) but
  folds the rest into intent, so coverage is retained.
- gemini is more conservative: cleaner category hygiene on
  family-friendly (drops the avoidance category) but under-consolidates
  plot shapes (FACETS on pursues, for-kids) and regressed thrillers.
- Net: for the primary goal (stop over-fragmenting single concepts),
  gpt-5.4-mini at low/low performed better here; gemini was cleaner on
  avoidance-category hygiene. Neither fixed audio-language.

## Round 2: category-definition audit (`audit_gpt`)

Result set `audit_gpt` = gpt-5.4-mini (reasoning low, verbosity low) on the
category-definition audit rewrite (clusters A1/A2/A3/A4 + B1/B2/B3/B6/B9 in
`schemas/trait_category.py`, plus the Step-3 `fit` definition-adherence
criterion). Step 2 held fixed (same `step_2_results.json`). Compare with
`python -m search_v2.category_candidates_experiment.summarize base fix_gpt audit_gpt`.

**Wins vs `fix_gpt`:**
- **Suitability redundancy (A2) — fully fixed.** `family friendly`: was
  3/5 [Sensitive content, Target audience] + 2/5 solo → now **5/5 SOLO
  [Target audience]**. `movies for kids` holds 5/5 solo Target audience.
  No more Maturity/Sensitive piggybacking on an audience trait.
- **Audio-language trap (A4) — fixed for `japanese`.** Was 5/5 with Audio
  language (base) → `audit_gpt` 0/5 Audio (routes Country / Cultural
  tradition, 3/5 SOLO). The explicit-only gate + cross-axis tests work
  when a strong tradition/country alternative exists.
- **Story consolidation (A1/B2) — `washed-up boxer` improved**: 3/5 → **4/5
  SOLO [Story / thematic archetype]**; the Element-motif and
  Character-archetype facet splits are gone (one residual [Central topic,
  Story] split).
- **`clean comedy`** de-fragments further: the Target-audience facet drops
  (was [Emotional, Genre, Target] in base) → 5/5 [Emotional, Genre].

**Mixed:**
- **`loser guy pursues a popular girl`**: `fix_gpt` was 5/5 SOLO but 2/5 on
  the WRONG anchor (solo Character archetype, losing the pursuit shape).
  `audit_gpt` is 2/5 SOLO [Story] + 3/5 FACETS [Character, Story] — Story is
  now the home in 5/5 (mis-anchor eliminated), but the "loser guy" descriptor
  still peels into a Character facet 3/5, so pure-solo rate dropped. Net:
  better coverage/correctness, slightly more fragmentation.

**Not fixed:**
- **`serbian` / `serbian movies` still commit Audio language 5/5.** Step 2
  does NOT pre-tag (`step2_category=None`), so the Audio candidate is
  generated at Step 3 — which DOES see the explicit-only boundary gate, yet
  keeps Audio for `serbian` while dropping it for `japanese`. Likely cause:
  AUDIO_LANGUAGE's `description`/`good_examples` still imply
  "<nationality>-language films" (e.g. "Spanish-language films"), and Step 2's
  trimmed vocab shows only name+description+good_examples (not the boundary),
  so the nationality→language pattern survives; with no strong
  CULTURAL_TRADITION alternative for "serbian," Step 3 falls back to language.
  Next lever: strip the nationality→language implication from AUDIO_LANGUAGE's
  description/good_examples (Step-2-visible), then re-run.

**Regression guards — all held:** `thrillers` 5/5 SOLO Genre, `Tom Hanks`
5/5, `Oscar winning`/`war films`/`from the 1990s` stable, `hidden gem` stays
multi-call (no over-consolidation), `gory`/`swearing` → 5/5 Sensitive content.
Inclusion-only intact: absence-scan hits are mostly the model affirming
presence framing ("the presence of X, not the absence/avoidance").

## Round 3: Step-2 source fix for the audio-language trap (`s2fix_gpt`)

Result set `s2fix_gpt` = gpt-5.4-mini (reasoning low, verbosity low). **Unlike
Rounds 1–2, this round changed Step 2** — so `step_2_results.json` was
regenerated (old copy preserved as `step_2_results.pre_s2fix.json`) and the run
is NOT Step-3-isolated. Compare with
`python -m search_v2.category_candidates_experiment.summarize audit_gpt s2fix_gpt`.

Three levers (the first two are Step-2-visible vocab; the third is Step 2 itself):
- **AUDIO_LANGUAGE de-nationalized** (`schemas/trait_category.py`): description +
  good_examples no longer use the "<nationality>-language films" template
  (removed "Spanish-language films" / "Korean-language"); now framing-forward
  ("subtitled, not dubbed", "spoken in Mandarin", "in the original Tamil").
- **COUNTRY_OF_ORIGIN gained a bare-nationality template** ("Mexican films") so a
  bare "<nationality> movies" query has an origin home at least as clean as the
  one removed from AUDIO_LANGUAGE.
- **Step 2 intent_exploration calibrated** (`search_v2/step_2.py` framing +
  `schemas/step_2.py` `QueryAnalysis.intent_exploration` field): enumerate
  multiple reads only when comparably plausible; new NEVER bullet forbids
  WIDENING ONE READ WITH A LOW-CONFIDENCE 'OR' (the exact serbian failure shape).

**ROOT-CAUSE CORRECTION to Round 2.** Round 2 concluded the serbian Audio commit
was *generated at Step 3* and the lever was the AUDIO examples / a Step-3 reasoning
fix. That was a mis-diagnosis. The real origin is **upstream in Step 2**: its
`intent_exploration` bundled the language read into the trait's `evaluative_intent`
as a fake-single interpretation — "produced in Serbia **OR are in the Serbian
language**", explicitly asserting "no competing interpretations." Step 3 then
faithfully decomposed that into two aspects (country + audio). Controlled proof:
`japanese` Step 2 intent was origin-only ("originate from Japan") → 0 Audio, while
`serbian` carried the OR-language clause → Audio every time. Same Step-3 category
definitions, opposite outcomes, decided entirely by Step 2. The AUDIO vocab template
"Spanish-language films" (structurally == "serbian movies") and the *absence* of any
"<nationality> films" template in COUNTRY_OF_ORIGIN reinforced the asymmetry.

**Result — trap fixed at the source:**
- Step 2 `serbian movies` intent is now "produced in or originating from Serbia"
  (a within-COUNTRY disjunction, no language clause); `family friendly serbian
  films` routes 'serbian' → origin only. `japanese` unchanged (origin/tradition).
- Step 3: `serbian movies` **5/5 SOLO [Country of origin]** (was 3× facets + 2×
  framings [Audio, Country]); `family friendly serbian films` 'serbian' **5/5 SOLO
  [Country of origin]** (was 5× framings [Audio, Country]). The fix also removed the
  spurious 2-call Audio+Country compound.
- **Zero committed Audio-language calls across all 12 queries.** Audio still appears
  as a Step-3 *candidate* on serbian (2/5 runs) but ranked `likely_disregard` — the
  fit machinery correctly demotes the residual.

**Regression guards held:** `japanese` 0 Audio, 4× framings + 1× solo [Country,
Cultural tradition] + horror 5/5 Genre; `Tom Hanks` 5/5 Person credit; `Oscar
winning` 5/5 Award; `war films` 5/5 SOLO Genre (cleaner — audit had 2× framings with
Central topic); `1990s` 5/5 era; `gory`/`swearing` 5/5 Sensitive; `washed-up boxer`
4/5 SOLO Story; `scary horror` 5/5 facets [Emotional, Genre]. `loser guy pursues`
improved (Character-facet peel 3/5 → 1/5).

**Caveat — Step-2 resampling.** Regenerating Step 2 (single sample, temp 0.35)
reshuffled atomization on a few queries: `anime movie` grouped vs `anime`; `for
kids` vs `movies for kids` (both 5/5 Target); and `clean comedy with no swearing`
split out a standalone `clean` trait that routes inconsistently (2× Sensitive, 1×
Target, 1× [Emotional, Target], 1× [General appeal, Target]). These are atomization
variance from resampling, not effects of the category edits — but the scattered
`clean` routing is worth a future look.
