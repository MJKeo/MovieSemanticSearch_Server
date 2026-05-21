# Concept Tags Reference Table

Reference table for all 25 concept tags across 7 categories used by the
`concept_tags` metadata generator. Built from
[schemas/enums.py](schemas/enums.py) (enum definitions + IDs) and
[movie_ingestion/metadata_generation/prompts/concept_tags.py](movie_ingestion/metadata_generation/prompts/concept_tags.py)
(prompt-side inclusion/exclusion rules).

Use this as the ground-truth definition sheet when establishing a
baseline for evaluation. Each row captures the tag's definition, the
positive evidence patterns that should trigger inclusion, and the
negative patterns that look superficially like a match but are NOT this
tag.

## General classification rules

- **Starting point for every category is EMPTY.** Tags are added only
  when input evidence supports them; empty categories are correct and
  expected.
- **"When in doubt, include it"** — applies to all tags EXCEPT
  `FEMALE_LEAD`, which has a stricter default (a false positive is
  worse than a miss).
- **Evidence hierarchy:** (1) direct evidence in inputs > (2) concrete
  inference from a specific plot detail > (3) parametric knowledge at
  95%+ confidence (well-known films only). Genre conventions alone are
  NEVER sufficient.
- **Structural endings vs. emotional endings are independent.**
  `OPEN_ENDING` and `CLIFFHANGER_ENDING` (narrative structure) coexist
  with `HAPPY_ENDING` / `SAD_ENDING` / `BITTERSWEET_ENDING` (emotional
  ending category — exactly one of these must be selected).

---

## Narrative Structure (IDs 1–9)

Structural choices in how the story is told. Multi-label — zero or more
tags may apply.

| Tag (ID) | What it represents | Include WHEN | Do NOT include for |
|---|---|---|---|
| **PLOT_TWIST** (1) | A surprise revelation recontextualizes events the audience already saw. | The audience formed an understanding that the reveal overturns. Direct signals: `information_control` terms, plot_keywords "surprise ending"/"plot twist", plot_summary reveals. | Any surprise or unexpected event; a late-act betrayal the audience could see coming; new information that adds to the story without changing prior understanding. |
| **TWIST_VILLAIN** (2) | The villain's *identity* is the surprise — the audience does not know who the true antagonist is until the reveal. | The villain's *status* as villain was hidden until reveal. Implies PLOT_TWIST (added deterministically). | A known villain who turns out to have additional motivations; a villain suspicious from the start; a known antagonist whose plan is revealed later. |
| **TIME_LOOP** (3) | Characters relive the same time period repeatedly. | `narrative_delivery` terms, plot_keywords "time loop", plot_summary describes repeated days/events. | Time travel (distinct concept); a single repeated scene used as flashback. |
| **NONLINEAR_TIMELINE** (4) | Non-chronological structure is a *defining* feature of how the story is told. | `narrative_delivery` terms, plot_keywords "nonlinear timeline", plot_summary structure shows deliberate non-chronological ordering. | Occasional flashbacks within a chronological main narrative; a single framing device or prologue. |
| **UNRELIABLE_NARRATOR** (5) | The narrator/POV character's account is *revealed* as untrustworthy. | `pov_perspective` terms, plot_summary shows the audience discovering what was shown/told was distorted or fabricated. | A character who lies to *other characters* (that is deception, not unreliable narration); a character who hallucinates UNLESS the film presents hallucinations as reality and then reveals the distortion. |
| **OPEN_ENDING** (6) | Story completes its arc but deliberately leaves its central question ambiguous — audiences debate what happened. | plot_keywords "ambiguous ending"; plot_summary ending where central conflict is intentionally and centrally ambiguous. | Sequel setup (that is CLIFFHANGER_ENDING); unanswered side questions; an ending that is emotionally unsatisfying but narratively clear. If the central conflict is resolved, do not tag. |
| **SINGLE_LOCATION** (7) | Bottle movie — nearly all action in one location; spatial constraint is a defining feature. | plot_summary shows events confined to one place as the film's identity. | A movie set mostly in one building but with significant scenes elsewhere; a haunted-house movie where characters also travel elsewhere. |
| **BREAKING_FOURTH_WALL** (8) | Characters directly address the audience or acknowledge they're in a movie, as a notable deliberate choice. | `additional_narrative_devices` terms, plot_keywords "breaking the fourth wall". | Voiceover narration where the character tells their story without acknowledging the audience; documentary-style interviews; songs commenting on the action unless characters explicitly address viewers. |
| **CLIFFHANGER_ENDING** (9) | Central conflict is unresolved with strong setup for continuation — the story stopped mid-arc. | plot_summary ending shows unresolved main conflict and sequel setup. | A satisfying resolution where the villain survives or a sequel is possible; central conflict resolved even if side threads remain open. Distinct from OPEN_ENDING (story complete but ambiguous). |

---

## Plot Archetypes (IDs 11–14)

The central premise or driving force. Tag applies when the concept IS
the movie, not just an element in the plot.

| Tag (ID) | What it represents | Include WHEN | Do NOT include for |
|---|---|---|---|
| **REVENGE** (11) | Vengeance is the *primary* narrative engine; protagonist's central goal throughout. | plot_keywords "revenge"; `conflict_type` supports it; plot_summary shows revenge as the driving plot. | A rescue mission motivated by anger; a character seeking justice through legal means; a subplot of retaliation within a different main plot. |
| **UNDERDOG** (12) | Protagonist expected to lose due to lack of resources, talent, or status; the story's emotional engine is "can they overcome despite being outmatched?" | `narrative_archetype` terms; plot_keywords "underdog"; plot_summary frames the disadvantage as the central dramatic question. | A protagonist who faces a stronger adversary but is competent enough that the outcome feels uncertain (skilled grifters vs. a mob boss); a franchise where one side is structurally weaker as setting (rebels vs. empire); intellectual disagreement; any power asymmetry that is setting rather than identity. Conflict-type asymmetry alone is NOT enough. |
| **KIDNAPPING** (13) | A kidnapping/abduction IS the central plot — the movie is about the kidnapping event and its direct consequences (rescue, escape, ransom). | plot_keywords "kidnapping"; plot_summary centers the abduction event itself. | Imprisonment as *backstory* motivating a different main plot (e.g., revenge); captives as the premise for a chase/escape story where the chase IS the plot; supernatural capture; a brief capture that is one event among many. |
| **CON_ARTIST** (14) | Protagonist is a con artist/grifter/scammer — the movie is about deception as a craft. | plot_keywords; plot_summary shows deception-driven plot with con artistry as the protagonist's mode. | A character who lies/manipulates for personal revenge or survival; a villain who deceives; a character who runs a single con as part of a larger non-con plot. Distinct from heist/robbery. |

---

## Settings (IDs 21–23)

The setting must be a defining characteristic of the movie, not
incidental.

| Tag (ID) | What it represents | Include WHEN | Do NOT include for |
|---|---|---|---|
| **POST_APOCALYPTIC** (21) | Set after civilization's collapse; society has fallen. | plot_keywords "post apocalypse"; plot_summary establishes a collapsed-society setting. | Dystopian (society intact but oppressive — distinct concept); sci-fi set on other planets or in space; a localized disaster that hasn't toppled civilization. |
| **HAUNTED_LOCATION** (22) | Centered around a haunted house, building, or specific location — the haunting of THAT place IS the story. | plot_keywords "haunted house"; plot_summary establishes a specific haunted location as the story's anchor. | Broader supernatural horror (possession, curses, mobile ghosts); a scary location that isn't supernaturally haunted; concentration camps, prisons, or other places of *historical* suffering. |
| **SMALL_TOWN** (23) | Small-town setting is central to the story's identity and atmosphere — the story feels inseparable from its small-town context. | plot_keywords "small town"; plot_summary makes the small-town setting integral. | A movie set in a rural area that isn't a town; a movie where the small-town setting is incidental and could happen anywhere; a city suburb. |

---

## Characters (IDs 31–33)

| Tag (ID) | What it represents | Include WHEN | Do NOT include for |
|---|---|---|---|
| **FEMALE_LEAD** (31) ⚠️ stricter default | The *single core protagonist* of the story is female. | All three conditions must hold: (1) the movie has a single core protagonist whose arc IS the movie; (2) that protagonist is identifiable in plot_summary AND corroborated by top_billed_cast prominence ranking; (3) that protagonist is female with high confidence. | Any movie without a single dominant protagonist; 2 co-leads of equal weight even if one is female; a prominent female character in an ensemble; a wife/girlfriend/daughter/mother/love interest important but not core; a female POV character whose core arc belongs to someone else; ANY case where slot 1 of top_billed_cast is a man and the plot does not clearly establish a different female protagonist. When in doubt, do NOT tag. |
| **ENSEMBLE_CAST** (32) | 3+ main characters share roughly equal importance, screen time, and storyline focus. The "protagonist" is often the group or an event. | `pov_perspective` terms suggesting multiple POVs; plot_summary shows 3+ characters with independent intertwined arcs of comparable importance. Decision test: removing any one character's storyline would NOT fundamentally collapse the film. | A protagonist with several important supporting characters; parallel plotlines where one character's arc is clearly primary; exactly 2 co-leads of equal weight (ensemble requires 3+); many named characters with one clear lead. A long character list ≠ ensemble — count whose *decisions* drive the plot. |
| **ANTI_HERO** (33) | Protagonist operates outside conventional morality as a defining trait — substantive boundary-crossing (criminal acts, violence, exploitation, vigilantism) as primary mode of operating. | `audience_character_perception` terms; `character_arc_labels`; plot_keywords "anti hero"; plot_summary supports it. | A flawed but fundamentally moral character who does the right thing; a teenager who skips school or breaks minor rules; a character described as "rebellious" without substantive moral transgression. |

---

## Endings (IDs 41–43, plus classification-only NO_CLEAR_CHOICE = -1)

**Exactly one** tag per movie. Captures how the *audience feels* when
the credits roll, NOT a factual ledger of plot outcomes. Independent of
OPEN_ENDING/CLIFFHANGER_ENDING above. Primary signal:
`emotional_observations` reports of audience reactions to the ending.

| Tag (ID) | What it represents | Select WHEN | Do NOT select for |
|---|---|---|---|
| **HAPPY_ENDING** (41) | Audience leaves feeling positive — satisfaction, relief, triumph, or warmth. | `emotional_observations` report positive end-state audience emotion. A hard-won victory is still happy. Surviving horror and defeating the threat is happy when protagonists are safe and the danger is over. Sacrifice along the way doesn't disqualify if final emotion is positive. | Merely surviving a horrific ordeal without positive feeling; a victory that feels hollow or Pyrrhic to the audience. |
| **SAD_ENDING** (42) | Audience leaves feeling predominantly sad — loss, failure, defeat; lasting emotion is grief/devastation/heartbreak. | `emotional_observations` report devastation or heartbreak at the ending. A cliffhanger where heroes have lost and the villain remains at large IS sad — narrative closure isn't required. | A victory achieved at great cost where the audience still feels the victory; an emotionally intense movie with a positive outcome; a tragic journey that ends in redemption or peace. |
| **BITTERSWEET_ENDING** (43) | Audience experiences genuinely mixed emotions — joy and sadness in unresolvable tension. Both achievement and loss are real and substantial. | Protagonist achieves their main goal but suffers a significant, concrete loss; `emotional_observations` describe the ending as holding both emotions in tension. Uncommon but real — not a compromise/fallback. | A happy ending that required sacrifice (audience feels the victory → happy); a sad ending with thematic beauty (audience feels the loss → sad); a movie with both happy and sad moments during runtime but whose ending lands clearly on one side; structural ambiguity (open/ambiguous ending) — narrative uncertainty is NOT emotional ambiguity. |
| **NO_CLEAR_CHOICE** (-1) | Evidence ambiguous, insufficient, or the ending doesn't fit happy/sad/bittersweet. | Extracted observations do not point clearly to one of the above — do not force a classification. Many movies legitimately land here. | Use only as a fallback. Filtered out before storage — never appears in `concept_tag_ids`. |

---

## Experiential (IDs 51–52)

Binary deal-breaker qualities. Primary source for both:
`emotional_observations`.

| Tag (ID) | What it represents | Include WHEN | Do NOT include for |
|---|---|---|---|
| **FEEL_GOOD** (51) | Overall experience is warm and uplifting — viewer leaves feeling positive, hopeful, happy. About emotional *warmth*, not excitement. | `emotional_observations` use words like "uplifting", "heartwarming", "feel-good", "joyful", "life-affirming", "charming", "delightful", "triumph", "empathy", "playful". Tension during the journey doesn't disqualify warmth at the destination. | A pure adrenaline experience (action thrills, horror scares); cathartic satisfaction from violent revenge; guilty-pleasure enjoyment of trashy/gory content. Do NOT infer from genre alone. |
| **TEARJERKER** (52) | The movie makes people cry — audiences *report* that it does. | `emotional_observations` contain explicit reports of crying, tears, "emotionally wrecked", "bring tissues", "sobbed". Bar is high — actual crying must be reported. | A movie described as "moving", "touching", "tugs at heartstrings", or "poignant" WITHOUT reports of actual crying. These indicate emotion but do not meet the tearjerker threshold. |

---

## Content Flags (ID 61)

Avoidance deal-breakers — things users specifically search to AVOID.

| Tag (ID) | What it represents | Include WHEN | Do NOT include for |
|---|---|---|---|
| **ANIMAL_DEATH** (61) | A non-human animal (dog, cat, horse, bird, etc.) dies on screen or as a significant plot point. | plot_keywords "animal death" / "dog dies"; plot_summary describes the death of a real animal on screen or as a significant plot beat. | Human deaths of any kind; violence against humans; the word "animal" appearing in an unrelated context; fantasy/sci-fi creatures clearly not real animals. |

---

## Deterministic post-generation fixups

Applied by `ConceptTagsOutput.apply_deterministic_fixups()` after the
LLM returns:

1. **Deduplication** — duplicate enum values within a tag list are
   removed (preserve first occurrence order). Skips `endings` (single
   value).
2. **TWIST_VILLAIN → PLOT_TWIST implication** — if
   `narrative_structure` contains `TWIST_VILLAIN` but not
   `PLOT_TWIST`, `PLOT_TWIST` is appended. TWIST_VILLAIN is
   definitionally a subset of PLOT_TWIST.

---

## Known recurring failure modes (per prompt's stated motivation)

The prompt's NOT-patterns were derived from these observed
classification failures — these are the categories most worth
inspecting when establishing a baseline:

- **FEMALE_LEAD over-inclusion** (the motivating example) — tagged on
  ensembles, two-handers, prominent-but-not-lead female characters,
  and films where the male top-billed actor is the actual lead.
- **ANTI_HERO** tagged on mild mischief or "rebellious" framing
  without substantive moral transgression.
- **KIDNAPPING** tagged when imprisonment is backstory for a
  different plot (typically revenge).
- **FEEL_GOOD** tagged from excitement/adrenaline rather than warmth.
- **TEARJERKER** tagged from "moving"/"touching" language without
  reports of actual crying.

---

## Evaluation Test Set

23 movies covering positive and hard-negative cases for every tag.
Movie IDs are TMDB IDs (== `movie_card.movie_id`); the importable list
lives in [concept_tags_test_movies.py](concept_tags_test_movies.py).

Each row lists:

- **✓ Expected tags** — tags the generator SHOULD produce. Endings are
  prefixed `ending:` since exactly one is always selected.
- **✗ Watch for over-tagging** — tags the generator might incorrectly
  add because of surface signals (genre, prominent characters, "moving"
  reviews, etc.). These are the explicit trip-ups this movie is in the
  set to catch.

Tags not listed in either column are expected to be absent without any
particular concern about over-tagging.

| # | Movie (year) | TMDB | ✓ Expected tags | ✗ Watch for over-tagging |
|---|---|---|---|---|
| 1 | Kill Bill: Vol. 1 (2003) | 24 | REVENGE, FEMALE_LEAD, ANTI_HERO, NONLINEAR_TIMELINE, CLIFFHANGER_ENDING, ending: NO_CLEAR_CHOICE | KIDNAPPING (assault is backstory), ENSEMBLE_CAST (clear single lead), ANIMAL_DEATH |
| 2 | Fight Club (1999) | 550 | PLOT_TWIST, UNRELIABLE_NARRATOR, ANTI_HERO, ending: NO_CLEAR_CHOICE | BREAKING_FOURTH_WALL (voiceover ≠ audience acknowledgment), FEEL_GOOD, TEARJERKER, HAPPY_ENDING |
| 3 | Frozen (2013) | 109445 | TWIST_VILLAIN (Hans), PLOT_TWIST (deterministic implication), FEEL_GOOD, ending: HAPPY_ENDING | FEMALE_LEAD (Anna + Elsa are co-leads — strict default forbids tagging), TEARJERKER (heartwarming ≠ crying), ANIMAL_DEATH, BITTERSWEET_ENDING |
| 4 | Groundhog Day (1993) | 137 | TIME_LOOP, SMALL_TOWN, FEEL_GOOD, ending: HAPPY_ENDING | NONLINEAR_TIMELINE (each loop runs chronologically), TEARJERKER |
| 5 | Pulp Fiction (1994) | 680 | NONLINEAR_TIMELINE, ENSEMBLE_CAST, ANTI_HERO, ending: NO_CLEAR_CHOICE | FEMALE_LEAD (Mia prominent but ensemble — no single lead), CON_ARTIST (deception everywhere but no protagonist-by-trade), SMALL_TOWN |
| 6 | Deadpool (2016) | 293660 | BREAKING_FOURTH_WALL, ANTI_HERO, REVENGE, ending: HAPPY_ENDING | FEEL_GOOD (gory comedy ≠ warmth), TEARJERKER |
| 7 | Taken (2008) | 8681 | KIDNAPPING, ending: HAPPY_ENDING | REVENGE (rescue mission ≠ vengeance), PLOT_TWIST, ANTI_HERO |
| 8 | The Conjuring (2013) | 138843 | HAUNTED_LOCATION, ANIMAL_DEATH (Sadie the dog dies), ending: HAPPY_ENDING | FEEL_GOOD, TEARJERKER |
| 9 | Get Out (2017) | 419430 | PLOT_TWIST, TWIST_VILLAIN (the Armitage plan), ending: HAPPY_ENDING | HAUNTED_LOCATION (creepy house but social horror, no supernatural haunting), FEEL_GOOD, SMALL_TOWN, SAD_ENDING |
| 10 | 12 Angry Men (1957) | 389 | SINGLE_LOCATION, ENSEMBLE_CAST, ending: HAPPY_ENDING | FEMALE_LEAD (no women in the cast — trivial NO), PLOT_TWIST (deliberation ≠ recontextualizing reveal), ANTI_HERO |
| 11 | The Mist (2007) | 5876 | ending: SAD_ENDING | SINGLE_LOCATION (mostly the store but significant exterior scenes), HAPPY_ENDING, FEEL_GOOD |
| 12 | Catch Me If You Can (2002) | 640 | CON_ARTIST, ending: HAPPY_ENDING | UNRELIABLE_NARRATOR (Frank deceives characters, not the audience), PLOT_TWIST, ANTI_HERO |
| 13 | Mad Max: Fury Road (2015) | 76341 | POST_APOCALYPTIC, ending: HAPPY_ENDING | FEMALE_LEAD (Furiosa is co-lead with Max — title character), KIDNAPPING (captives' chase IS the plot, not the abduction itself), ENSEMBLE_CAST (two co-leads ≠ 3+ ensemble), TEARJERKER |
| 14 | Rocky (1976) | 1366 | UNDERDOG, FEEL_GOOD, ending: BITTERSWEET_ENDING ⚠ borderline (HAPPY also defensible — see notes) | ANTI_HERO (flawed but fundamentally moral) |
| 15 | Star Wars: A New Hope (1977) | 11 | ending: HAPPY_ENDING | UNDERDOG (rebels-vs-empire asymmetry is *setting*, not story identity), POST_APOCALYPTIC (sci-fi in space ≠ collapsed Earth), CLIFFHANGER_ENDING (resolved despite sequel), TWIST_VILLAIN (Vader known as villain from start), KIDNAPPING (Leia rescue is a set piece, not the central plot) |
| 16 | Inception (2010) | 27205 | OPEN_ENDING, ending: HAPPY_ENDING (Cobb returns to his kids — open-ending question is about *reality*, not emotion) | FEMALE_LEAD (Ariadne is prominent female but Cobb is clearly the lead — primary over-tag candidate), TIME_LOOP (dream layers feel cyclical), PLOT_TWIST, TEARJERKER, TWIST_VILLAIN |
| 17 | La La Land (2016) | 313369 | ending: BITTERSWEET_ENDING | FEMALE_LEAD (Mia + Sebastian are co-leads — primary over-tag candidate), ENSEMBLE_CAST (two co-leads ≠ ensemble), FEEL_GOOD (ending lands bittersweet, not warm) |
| 18 | Marley & Me (2008) | 14306 | ANIMAL_DEATH, TEARJERKER, ending: SAD_ENDING | FEEL_GOOD (despite warm moments, takeaway is grief), BITTERSWEET_ENDING (debatable — but audience's lasting emotion is grief, not mixed) |
| 19 | John Wick (2014) | 245891 | REVENGE, ANIMAL_DEATH (the puppy), ANTI_HERO, ending: HAPPY_ENDING | FEEL_GOOD (adrenaline ≠ warmth), TEARJERKER |
| 20 | Paddington 2 (2017) | 346648 | FEEL_GOOD, ending: HAPPY_ENDING | TEARJERKER (heartwarming but not crying-tier), ANIMAL_DEATH (no animals die — Paddington is alive and well), ANTI_HERO |
| 21 | Erin Brockovich (2000) | 462 | FEMALE_LEAD (textbook positive), UNDERDOG, ending: HAPPY_ENDING | ANTI_HERO (brash and rule-breaking but fundamentally moral), SAD_ENDING, TEARJERKER |
| 22 | A Serious Man (2009) | 12573 | OPEN_ENDING, ending: NO_CLEAR_CHOICE (existential dread + cosmic indifference — outside the happy/sad/bittersweet axis) | HAPPY_ENDING, SAD_ENDING, BITTERSWEET_ENDING |
| 23 | The Graduate (1967) | 37247 | ending: NO_CLEAR_CHOICE (the famous fading smiles) | HAPPY_ENDING (the surface trip — "they got together"), BITTERSWEET_ENDING (the over-correction trip), FEMALE_LEAD (Elaine prominent but Ben is the lead), ANTI_HERO |

### Borderline cases — flag for human review during baseline grading

These rows have a defensible second-choice tag. Don't auto-grade the
model wrong if it picks the alternative — record both and decide later
whether the prompt's definitions need sharpening.

- **Rocky — BITTERSWEET vs HAPPY ending.** Rocky loses the fight but
  achieves his stated goal of "going the distance." Per the prompt's
  "a hard-won victory is still happy" rule, HAPPY is defensible. Per
  the standard critical reading and the prompt's BITTERSWEET definition
  ("achieves main goal but suffers a significant, concrete loss"),
  BITTERSWEET is defensible.
- **Marley & Me — SAD vs BITTERSWEET ending.** The dog dies; the family
  lives on with memories. Per the prompt's "lasting emotion is grief"
  rule, SAD is the expected tag. BITTERSWEET defenders would argue the
  warm memories are the "achievement" balancing the loss.
- **Kill Bill: Vol. 1 — NO_CLEAR vs SAD vs BITTERSWEET ending.** Vol. 1
  ends mid-arc on the "B.B. is alive" reveal. The emotion is
  anticipation / setup, which doesn't map cleanly to any named ending.
  NO_CLEAR_CHOICE is the cleanest call but reasonable evaluators might
  pick something else.
- **Inception — PLOT_TWIST yes/no.** The spinning-top ambiguity is
  structural (covered by OPEN_ENDING), not a recontextualizing reveal
  of prior scenes. Expected absence, but if the model tags it,
  flag for definition-sharpening rather than automatic miss.

---

## Tag Definition Reference (Rewrite Baseline)

Per-tag definition sheet built after the baseline evaluation run.
Tightens inclusion criteria, calls out concrete boundary cases from
the baseline failures, names which inputs to consult and how, and
flags **already-generated metadata that is NOT currently fed to
`concept_tags`** but would help.

Columns:

- **Inclusion criteria** — the rule for when to tag.
- **Boundary cases** — gray-zone calls and how to resolve them
  (baseline failures cited inline).
- **Evidence (how to use it)** — which existing inputs to consult and
  what specifically to look for.
- **Missing data (already generated)** — fields that already exist in
  the generated metadata or movie input but aren't currently passed
  to the `concept_tags` prompt.

### Already-generated but NOT currently fed to concept_tags

The following fields are produced by upstream stages and stored, but
the current `build_concept_tags_user_prompt` doesn't pass them:

- **From `MovieInputData`:** `genres`, `directors`, `featured_reviews`
  (summary + text), `audience_reception_attributes` (review themes
  with sentiment), `maturity_rating` + `maturity_reasoning`,
  `parental_guide_items` (category + severity), `collection_name`.
- **From `ReceptionOutput`:** `thematic_observations`,
  `craft_observations`, `source_material_hint`, `reception_summary`,
  `praised_qualities`, `criticized_qualities`.
- **From `PlotAnalysisOutput`:** `genre_signatures`,
  `thematic_concepts` (concept_label), `elevator_pitch`,
  `generalized_plot_overview`, `character_arcs[].reasoning`.
- **From `NarrativeTechniquesOutput`:** the 3 dropped sections —
  `characterization_methods.terms`, `character_arcs.terms`
  (film-language arc labels), `conflict_stakes_design.terms` — plus
  the `evidence_basis` text for every section.
- **From `ViewerExperienceOutput`:** entirely — especially
  `ending_aftertaste`, `emotional_palette`, `tone_self_seriousness`,
  `emotional_volatility`, `disturbance_profile`.
- **From `WatchContextOutput`:** entirely — especially `identity_note`,
  `key_movie_feature_draws`.

### Narrative Structure

| Tag | Inclusion criteria | Boundary cases | Evidence (how to use it) | Missing data (already generated) |
|---|---|---|---|---|
| **PLOT_TWIST** | A late-film reveal that recontextualizes events the audience already witnessed. The audience must have formed a model that the reveal explicitly overturns. | Tragic irony at the ending (The Mist) ≠ twist — nothing prior is recontextualized.<br>Sequel-setup reveals (Kill Bill Vol. 1's "B.B. is alive") = setup, not recontextualizing.<br>Structural ambiguity (Inception's spinning top) = OPEN_ENDING.<br>Character revealing motivation audience already suspected ≠ twist. | **`information_control`** NT terms — primary direct signal.<br>**`plot_keywords`** — "twist ending," "plot twist."<br>**`plot_summary`** — explicit late reveals reframing prior scenes. | **`craft_observations`** (ReceptionOutput) — reviewers often explicitly call out "twist," "rug-pull," "third-act reveal" when discussing narrative craft. Currently invisible to concept_tags. |
| **TWIST_VILLAIN** | The *identity* of the antagonist is hidden from the audience and revealed late. Auto-implies PLOT_TWIST. | Known villain whose deeper motivations are uncovered later ≠ tagged (Vader's parentage is about who he IS).<br>Cooperative ally revealed as antagonist clearly qualifies (Hans, Get Out's Armitages).<br>Tyler Durden in Fight Club is borderline — psychotic alter-ego, not a "villain identity reveal." | **`information_control`** — "false ally," "hidden antagonist," "betrayal reveal."<br>**`plot_summary`** — "is revealed to be," "all along," "secretly engineered." | **`craft_observations`** — "the reveal of [character] as the antagonist" patterns.<br>**`audience_reception_attributes`** — review themes often include "shocking villain reveal" with sentiment. |
| **TIME_LOOP** | Characters relive the same time period repeatedly as central premise. | Time travel ≠ time loop.<br>Single repeated scene as flashback ≠ loop. | **`narrative_delivery`** — "time loop," "reliving."<br>**`plot_keywords`** — "time loop" direct.<br>**`plot_summary`** — "wakes up to the same day." | None — current signals work (Groundhog Day correctly tagged 3/3). |
| **NONLINEAR_TIMELINE** | Non-chronological structure is a *defining identity* of the film. Audience has to reconstruct the timeline from scrambled pieces. | Occasional flashbacks within chronological narrative ≠ nonlinear (Catch Me If You Can over-tag).<br>Prologue/epilogue out of order ≠ nonlinear.<br>Multi-chapter scrambled (Pulp Fiction, Kill Bill Vol. 1, Memento) qualifies.<br>**Kill Bill miss** — 0/3 runs tagged it; upstream NT didn't surface chapter-structure signal. | **`narrative_delivery`** — "nonlinear," "fragmented timeline."<br>**`plot_keywords`** — "nonlinear timeline."<br>**`plot_summary`** — explicit structural language. | **`craft_observations`** — primary place reviewers describe structure ("told in chapters," "the film moves between timelines"). For Kill Bill, this almost certainly contains the missing signal.<br>**`audience_reception_attributes`** — often includes "non-linear narrative" as a tagged theme. |
| **UNRELIABLE_NARRATOR** | The narrator/POV character's account is later revealed to the audience as distorted or fabricated. Trust with the AUDIENCE is broken. | Characters lying to other characters (Catch Me If You Can) ≠ tagged.<br>Hallucinations presented as reality and revealed = qualifies.<br>Inception current 3/3 tag is debatable — Cobb's Mal flashbacks are subjective but audience knows his issues from early on. | **`pov_perspective`** — "unreliable narrator," "subjective POV."<br>**`plot_summary`** — explicit revelation that prior shown material was distorted. | **`craft_observations`** — reviewers explicitly flag unreliable narration as a craft choice.<br>**NT `evidence_basis` text** for `pov_perspective` — the justification sentence usually states whether the narration is reliable. Currently we pass terms only. |
| **OPEN_ENDING** | Story completes its arc but the *central question* is deliberately left ambiguous. | Sequel setup = CLIFFHANGER, not OPEN.<br>Side questions open ≠ tagged if central conflict resolved.<br>Inception, The Graduate, A Serious Man = positives. | **`plot_keywords`** — "ambiguous ending."<br>**`plot_summary`** — final beat avoids resolution.<br>**`emotional_observations`** — reviewers debating meaning. | **`thematic_observations`** — explicitly describes how reviewers interpret central question ("audiences continue debating whether...").<br>**`ending_aftertaste`** (ViewerExperience) — directly captures whether the ending lingers as an open question. |
| **SINGLE_LOCATION** | Nearly all action confined to one physical location. The spatial constraint IS the film's identity. | "Mostly one building but significant exterior scenes" ≠ tagged (The Mist over-tag).<br>12 Angry Men, Phone Booth = positives. | **`plot_summary`** — count distinct locations. | **`identity_note`** (WatchContext) — short classifier like "claustrophobic single-room thriller" would directly flag this.<br>**`key_movie_feature_draws`** — "bottle-movie tension" / "confined setting" often called out as a draw.<br>**`elevator_pitch`** (PlotAnalysis) — log-line phrasing usually names the constrained setting if it's identity-level. |
| **BREAKING_FOURTH_WALL** | Characters directly address the audience or acknowledge the film as a film. Notable, recurring stylistic choice. | Voiceover narration (Fight Club) ≠ tagged.<br>Deadpool's direct camera address = positive. | **`additional_narrative_devices`** — "fourth wall break," "direct address."<br>**`plot_keywords`** — "breaking the fourth wall." | **`craft_observations`** — reviewers often discuss fourth-wall as craft.<br>**`praised_qualities` / `criticized_qualities`** — sometimes lists "fourth wall breaks" as a draw. |
| **CLIFFHANGER_ENDING** | Central conflict unresolved at credits with strong continuation setup. Story stopped mid-arc. | Distinct from OPEN_ENDING (completed arc with ambiguity).<br>Kill Bill Vol. 1 = positive. Star Wars: A New Hope = NOT (Death Star destroyed). | **`plot_summary`** ending — "to be continued" energy.<br>**`plot_keywords`** — "cliffhanger." | **`collection_name`** (MovieInputData) — knowing the movie is part of a named collection ("Kill Bill Collection") flags planned-continuation context.<br>**`ending_aftertaste`** (VE) — captures "unresolved, leaves you wanting more" sensation. |

### Plot Archetypes

| Tag | Inclusion criteria | Boundary cases | Evidence (how to use it) | Missing data (already generated) |
|---|---|---|---|---|
| **REVENGE** | Vengeance is the *primary narrative engine*. Protagonist's central sustained goal IS getting revenge. | **Taken over-tag** — rescue mission ≠ revenge; Bryan's goal is rescue.<br>Revenge subplot inside a larger plot ≠ tagged.<br>John Wick, Kill Bill = positives. | **`plot_keywords`** — "revenge" direct.<br>**`conflict_type`** — supporting signal.<br>**`plot_summary`** — protagonist's stated goal involves "make them pay"? | **`elevator_pitch`** — 6-word log-line; if it doesn't contain a vengeance verb, REVENGE probably isn't the central engine. Would have flagged Taken as "father rescues kidnapped daughter."<br>**`thematic_concepts`** (PlotAnalysis) — likely contains "vengeance," "retribution" as concepts for real revenge movies.<br>**`generalized_plot_overview`** — 1-3 sentence thematic frame. |
| **UNDERDOG** | The protagonist's disadvantage IS the central dramatic question. Identity-level, not setting-level. | **Star Wars over-tag** — rebels-vs-empire is *setting*; Luke is chosen-one.<br>Skilled professionals facing stronger opponents ≠ underdog.<br>Rocky, Erin Brockovich = positives. | **`narrative_archetype`** — "underdog rising."<br>**`plot_keywords`** — "underdog."<br>**`plot_summary`** — protagonist's lack of skill/resources framed as the central engine. | **`conflict_stakes_design.terms`** (the dropped NT section) — directly describes the *design* of the conflict and what's at stake. Would distinguish "outmatched individual vs. world" (UNDERDOG) from "small force vs. larger force as setting" (NOT UNDERDOG).<br>**`thematic_concepts`** — "outmatched protagonist," "against the odds" appear here when central.<br>**`elevator_pitch`** — usually names the disadvantage if it's the identity. |
| **KIDNAPPING** | A kidnapping IS the central plot. The movie is about the abduction event and its direct consequences. | **Mad Max over-tag** — captives in a chase movie; the chase IS the plot.<br>Imprisonment as backstory ≠ tagged.<br>Taken = positive. | **`plot_keywords`** — "kidnapping," "abduction."<br>**`plot_summary`** — kidnapping is inciting incident AND ongoing plot driver. | **`elevator_pitch`** — log-line names the central plot engine; if the abduction isn't in 6 words, it's likely not central.<br>**`generalized_plot_overview`** — describes what the movie is *about* thematically.<br>**`parental_guide_items`** — likely tags "kidnapping" as a content category with severity, surfacing it even when plot_summary is thin. |
| **CON_ARTIST** | Protagonist is a con artist/grifter/scammer as defining occupation. | Single con in non-con plot ≠ tagged.<br>Villain who deceives ≠ tagged.<br>Catch Me If You Can = positive. | **`plot_keywords`** — "con artist," "grifter."<br>**`plot_summary`** — does protagonist make their living through cons? | **`thematic_concepts`** — "deception as craft," "art of the con" appear here when central.<br>**`identity_note`** (WatchContext) — often phrased "stylish con-artist romp" / "grifter thriller" for these films. |

### Settings

| Tag | Inclusion criteria | Boundary cases | Evidence (how to use it) | Missing data (already generated) |
|---|---|---|---|---|
| **POST_APOCALYPTIC** | Set after civilization's collapse. Society has fallen. | Dystopian (intact-but-oppressive) ≠ tagged.<br>Sci-fi in space (Star Wars) ≠ tagged.<br>Localized disaster ≠ tagged. | **`plot_keywords`** — "post apocalypse" direct.<br>**`plot_summary`** — collapsed-civilization framing. | **`genres`** (TMDB list) — explicit "post-apocalyptic" / "dystopia" labels exist as genre tags.<br>**`genre_signatures`** (PlotAnalysis) — already-classified genre phrases; if "post-apocalyptic" is here, tag confidently.<br>**`identity_note`** — typically names it ("brutal post-apocalyptic chase"). |
| **HAUNTED_LOCATION** | Movie centers on a supernaturally haunted location; that place IS the story's anchor. | Mobile ghosts, possessions ≠ tagged.<br>Scary non-supernatural location (Get Out — correct NOT) ≠ tagged.<br>The Conjuring, The Shining = positives. | **`plot_keywords`** — "haunted house."<br>**`plot_summary`** — specific location named as locus of supernatural events. | **`genres`** — "Horror" + "Haunted House" subgenre signals.<br>**`identity_note`** — usually phrases this directly ("classic haunted-house chiller").<br>**`thematic_observations`** — often distinguishes supernatural-location horror from possession/cult horror. |
| **SMALL_TOWN** | Small-town setting is central to film's identity and atmosphere — story feels inseparable from its context. | Rural area not a town ≠ tagged.<br>Suburb ≠ tagged.<br>Town as incidental backdrop ≠ tagged.<br>Groundhog Day (Punxsutawney) = positive. | **`plot_keywords`** — "small town" direct.<br>**`plot_summary`** — explicit naming of small-town setting AND atmospheric significance. | **`key_movie_feature_draws`** (WatchContext) — atmospheric setting often appears here ("eerie small-town atmosphere," "small-town nostalgia") when load-bearing.<br>**`identity_note`** — frequently names the setting genre.<br>**`thematic_observations`** — often discusses small-town themes (community, claustrophobia, secrets) when central. |

### Characters

| Tag | Inclusion criteria | Boundary cases | Evidence (how to use it) | Missing data (already generated) |
|---|---|---|---|---|
| **FEMALE_LEAD** ⚠️ stricter default | All three: (1) single core protagonist, (2) identifiable in plot_summary AND corroborated by top_billed_cast slot 1 or 2, (3) female with high confidence. | 2 co-leads of equal weight ≠ tagged (Frozen, La La Land, Mad Max all correctly NOT).<br>Prominent female in ensemble ≠ tagged.<br>If slot 1 of cast is male AND plot doesn't unambiguously center a different female → do NOT tag. | **`plot_summary`** — primary (whose decisions drive the plot).<br>**`top_billed_cast`** — corroborating; slot 1 prominence significant.<br>**`plot_keywords`** — "female protagonist." | **`elevator_pitch`** — 6-word log-line; if it doesn't name a single female character, drop the tag. Reliable single-protagonist test.<br>**`character_arcs[].reasoning`** (PlotAnalysis) — full sentence describing the arc; would name the protagonist explicitly.<br>**`thematic_observations`** — reviewers often discuss a film as "a [female adjective]-led story" when that's the identity. |
| **ENSEMBLE_CAST** | 3 OR MORE main characters share roughly equal narrative weight. The "protagonist" is often the group or an event. | 2 co-leads ≠ ensemble (Mad Max over-tag).<br>Protagonist with several important supports ≠ ensemble (The Conjuring over-tag).<br>**Decision test:** would removing any one character collapse the film? Ensemble = no.<br>Pulp Fiction, 12 Angry Men = positives. | **`pov_perspective`** — "multiple POVs," "rotating POV."<br>**`plot_summary`** — count protagonists by DECISION-driving role.<br>**`top_billed_cast`** — equal plot prominence across 3-5 supports ensemble. | **`character_arcs[].reasoning`** (PA, plural) — PlotAnalysis emits 0-3 arc objects; count of named-and-developed protagonists is a direct signal.<br>**`characterization_methods.terms`** (dropped NT section) — sometimes contains "ensemble character work" / "rotating POV characterization."<br>**`elevator_pitch`** — naming a group ("five strangers," "the jury") vs. an individual is diagnostic. |
| **ANTI_HERO** | Protagonist's defining mode involves substantive moral transgression. Identity-level, not arc-level. | **MASS FAILURE — 4 false positives, 3/3 each:**<br>• Taken (Bryan — fundamentally moral father)<br>• Groundhog Day (Phil — arc IS redemption, ends moral)<br>• Catch Me If You Can (Frank — charming-rogue framing)<br>• Inception (Cobb — morally grey, family-motivated)<br>**Rule:** if the protagonist's arc moves toward conventional morality (redemption arc), they are NOT anti-hero by film's end. | **`audience_character_perception`** — "morally compromised," "anti-hero."<br>**`character_arc_labels`** (PA) — does the arc keep them outside morality, or end with redemption?<br>**`plot_keywords`** — "anti hero."<br>**`plot_summary`** — primary mode of operating throughout. | **`character_arcs.terms`** (the dropped NT section — distinct from PA arcs) — contains film-language arc labels like "redemption arc," "fall from grace," "moral awakening." A "redemption arc" label here should DISQUALIFY anti-hero — would catch 3 of 4 current false positives.<br>**`thematic_concepts`** (PA) — "morality redeemed," "loyalty over self" are common for false-positive cases.<br>**`character_arcs[].reasoning`** — full sentence on the arc; explicit moral trajectory described. |

### Endings

| Tag | Selection criterion | Boundary cases | Evidence (how to use it) | Missing data (already generated) |
|---|---|---|---|---|
| **HAPPY_ENDING** | Audience leaves feeling positive — satisfaction, relief, triumph, warmth. A hard-won victory IS happy. | **3 current misses defaulted to NO_CLEAR:**<br>• Inception — Cobb returns to kids; open question is reality, NOT emotion.<br>• The Conjuring — family freed, evil contained.<br>• Get Out — Chris escapes.<br>**Rule:** structural ambiguity DOES NOT mean emotional ambiguity.<br>John Wick currently BITTERSWEET — should be HAPPY (puppy died in setup, not ending). | **`emotional_observations`** — primary; how audiences felt LEAVING.<br>**`plot_summary`** — final beat at credits.<br>**`plot_keywords`** — "happy ending." | **`ending_aftertaste`** (ViewerExperience) — purpose-built section: terms describing the AUDIENCE FEELING at the ending. Currently invisible to concept_tags. This single field would resolve most ending failures.<br>**`emotional_volatility`** (VE) — captures the shape of the emotional ride; an emotionally turbulent journey landing on warmth is different from one landing on grief.<br>**`praised_qualities`** — "satisfying conclusion," "uplifting ending" often appear here.<br>**`audience_reception_attributes`** — review themes are sentiment-tagged ({name: "ending", sentiment: "positive"}), giving direct audience signal. |
| **SAD_ENDING** | Audience leaves feeling predominantly sad — grief, devastation, heartbreak. | Victory at great cost still felt as victory ≠ sad.<br>Tragic journey ending in peace ≠ sad.<br>Marley & Me, The Mist = positives. | **`emotional_observations`** — "devastating," "heartbroken," "left me sobbing."<br>**`plot_summary`** — ending state of loss. | **`ending_aftertaste`** (VE) — same field; should directly contain "devastating ending," "tragic close."<br>**`audience_reception_attributes`** — sentiment-tagged themes around the ending.<br>**`featured_reviews`** (raw text) — reviewer reactions to the ending often quoted in summary or text. |
| **BITTERSWEET_ENDING** | Audience genuinely experiences MIXED emotions — joy AND sadness in unresolvable tension. Both achievement AND substantial loss at the ending. | **NOT a fallback option.**<br>• The Graduate over-tag (3/3) — fading-smiles is existential NO_CLEAR.<br>• John Wick over-tag (3/3) — puppy died in setup, revenge at end → HAPPY.<br>Happy ending requiring sacrifice ≠ bittersweet (audience feels victory).<br>La La Land = positive. | **`emotional_observations`** — explicit "mixed feelings," "joy and sadness."<br>**`plot_summary`** — substantial achievement AND substantial loss simultaneously at ending. | **`ending_aftertaste`** (VE) — bittersweet endings often described directly as "bittersweet" / "joy and sorrow in equal measure" in this purpose-built field.<br>**`emotional_volatility`** (VE) — high volatility ending often correlates with bittersweet.<br>**`thematic_concepts`** — sometimes names "sacrifice for love," "achievement at cost." |
| **NO_CLEAR_CHOICE** | Ending genuinely doesn't fit happy/sad/bittersweet — existential, contemplative, unresolved-question. | A Serious Man, The Graduate, Kill Bill Vol. 1 = positives.<br>Do NOT pick just because endings have *some* complexity (currently the default trap). | **`emotional_observations`** — absence of clear positive/negative/mixed signal; presence of "contemplative," "existential." | **`ending_aftertaste`** (VE) — when it contains "ambiguous," "lingering," "contemplative" without clear valence, NO_CLEAR is right.<br>**`tone_self_seriousness`** (VE) — existential/philosophical tone signals correlate with NO_CLEAR endings.<br>**`thematic_observations`** — reviewers describing the ending as "philosophical" / "open to interpretation." |

### Experiential

| Tag | Inclusion criteria | Boundary cases | Evidence (how to use it) | Missing data (already generated) |
|---|---|---|---|---|
| **FEEL_GOOD** | Overall experience is warm and uplifting. About emotional **warmth**, not excitement. | Pure adrenaline (Deadpool — correct NOT) ≠ feel_good.<br>Cathartic violent revenge (John Wick — correct NOT) ≠ feel_good.<br>Tense moments don't disqualify warmth at destination.<br>Paddington 2, Groundhog Day, Frozen = positives. | **`emotional_observations`** — "uplifting," "heartwarming," "feel-good," "joyful," "delightful," "playful."<br>**Do NOT infer from genre.** | **`emotional_palette`** (VE) — direct sectional capture of the emotional tone.<br>**`tone_self_seriousness`** (VE) — distinguishes sincere-warm from adrenaline-cynical.<br>**`identity_note`** (WatchContext) — phrases like "sincere feel-good comedy" / "wholesome family adventure" are diagnostic.<br>**`praised_qualities`** — often contains "feel-good," "uplifting" when reviewers consistently report warmth. |
| **TEARJERKER** | Audiences REPORT crying. Bar is HIGH — explicit tears, sobbing, "needed tissues." | **Paddington 2 over-tag (3/3)** — heartwarming without crying reports.<br>"Moving," "touching," "poignant" without crying ≠ tagged.<br>Marley & Me = textbook positive. | **`emotional_observations`** — only signal that counts: "cried," "tears," "sobbed," "bring tissues." | **`featured_reviews`** (raw review text) — actual reviewer language about crying lives here most authentically; the LLM-condensed `emotional_observations` may strip the explicit "I cried" phrasing.<br>**`audience_reception_attributes`** — review themes often include "tearjerker" / "emotional" with sentiment.<br>**`emotional_volatility`** (VE) — extreme volatility correlates with tearjerker territory. |

### Content Flags

| Tag | Inclusion criteria | Boundary cases | Evidence (how to use it) | Missing data (already generated) |
|---|---|---|---|---|
| **ANIMAL_DEATH** | A non-human animal dies on screen OR as a significant plot point. | Brief incidental animal death without plot consequence ≠ tagged (Get Out's opening-scene deer is borderline).<br>Fantasy creatures clearly not real animals ≠ tagged.<br>John Wick (puppy), The Conjuring (Sadie), Marley & Me = positives. | **`plot_keywords`** — "animal death," "dog dies."<br>**`plot_summary`** — animal death described as a beat, not background. | **`parental_guide_items`** (MovieInputData) — IMDB-scraped {category, severity} entries directly include "Violence Against Animals" / "Animal cruelty" with severity ratings. This is the **single highest-leverage missing input** for this tag — would resolve the Get Out borderline (low severity → incidental) and confirm John Wick / The Conjuring / Marley & Me (high severity → significant).<br>**`maturity_reasoning`** — sometimes lists animal death as a reason for the rating. |

### Cross-cutting observations

A few already-generated fields keep showing up across many tags,
suggesting **single-field additions with compound payoff**:

- **`ending_aftertaste` (ViewerExperience)** — the most leveraged
  single field. Purpose-built to capture audience emotional landing.
  Directly addresses all 4 ending failures (3 HAPPY misses + 2
  BITTERSWEET over-corrections). Would also help OPEN_ENDING and
  CLIFFHANGER_ENDING.
- **`craft_observations` (ReceptionOutput)** — reviewers' explicit
  descriptions of structural/storytelling craft. Directly helps
  NONLINEAR_TIMELINE (the Kill Bill miss), PLOT_TWIST,
  UNRELIABLE_NARRATOR, BREAKING_FOURTH_WALL.
- **NT `character_arcs.terms` (dropped section)** — film-language arc
  labels distinct from PlotAnalysis's thematic arcs. A "redemption
  arc" label here would disqualify ANTI_HERO and catch 3 of 4 current
  false positives (Phil, Frank, Cobb).
- **`elevator_pitch` (PlotAnalysis)** — 6-word log-line is a strong
  protagonist-count + central-plot-engine signal. Helps FEMALE_LEAD,
  ENSEMBLE_CAST, REVENGE vs. KIDNAPPING (Taken), UNDERDOG.
- **`parental_guide_items` (MovieInputData)** — IMDB category-with-
  severity tags directly capture ANIMAL_DEATH and could help
  KIDNAPPING by surfacing "abduction" categories.
- **`identity_note` (WatchContext)** — 2-8 word viewing-appeal
  classifier. Already does much of what concept_tags is trying to do;
  useful cross-check for SMALL_TOWN, POST_APOCALYPTIC,
  HAUNTED_LOCATION, FEEL_GOOD, CON_ARTIST.

The cluster of failures most directly addressable by these
existing-but-unused fields:

1. **Ending failures (5 of them)** → `ending_aftertaste` alone.
2. **ANTI_HERO over-tagging (4 of them)** → NT `character_arcs.terms`
   redemption-arc check.
3. **NONLINEAR_TIMELINE Kill Bill miss** → `craft_observations`.
4. **ANIMAL_DEATH Get Out borderline** → `parental_guide_items`
   severity.
