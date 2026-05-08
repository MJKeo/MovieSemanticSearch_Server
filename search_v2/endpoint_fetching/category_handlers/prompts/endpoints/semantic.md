# Endpoint: Semantic (Vector Spaces)

Authors per-space vector queries against 7 Qdrant embedding spaces attached to every movie. Read the call's brief, decompose it into atomic aspects, decide which of the 7 spaces genuinely carry signal for those aspects, then write structured-label bodies in each chosen space's native vocabulary that read like ingest-side text for a matching movie.

The schema's field descriptions are the source of truth for what each field should contain. This document carries what they can't: what each vector space actually IS and how to author bodies in its ingest-side register.

## Canonical question

"Which embedding space(s) genuinely carry signal for this trait, and what text — in each space's native vocabulary — would a matching movie's ingest-side vector text look like?"

## The 7 vector spaces

Every movie carries 7 named vectors. Each was embedded at ingest from a structured body in that space's native vocabulary. Your job is to author the query-side equivalent for whichever spaces honestly cover the trait.

**plot_events** — What literally happens.
- Purpose: Chronological plot — actions, events, character-arc beats on screen. The "what happens" space.
- Sub-fields: `plot_summary` (dense prose summary of concrete events and characters).
- Boundary: Literal and concrete. Themes / genre signatures / conflict archetypes → plot_analysis. How the story is told (POV, structure, pacing devices) → narrative_techniques.
- Example queries: "a heist that unravels when a team member betrays the crew"; "a lone survivor crosses a frozen continent to find their family"; "the one where strangers trapped in an elevator discover a murderer among them".

**plot_analysis** — What kind of story thematically.
- Purpose: Genre signatures, thematic concepts, conflict archetypes, character-arc patterns. The "what type of story" space.
- Sub-fields: `elevator_pitch` (one-sentence thematic capsule), `plot_overview` (short prose), `genre_signatures`, `conflict_type` (e.g. "man vs self"), `thematic_concepts` (e.g. "grief", "redemption", "moral compromise"), `character_arcs` (e.g. "fall from grace", "coming of age").
- Boundary: Categorical and thematic shape. Specific events ("the one where X happens") → plot_events. Emotional/cognitive watching experience → viewer_experience.
- Example queries: "stories about grief and reconciliation between estranged siblings"; "man-vs-nature survival dramas"; "tragic fall-from-grace character arcs".

**viewer_experience** — What it feels like to watch.
- Purpose: Emotional, sensory, and cognitive experience of sitting through the movie. Tone, tension profile, disturbance level, cognitive load, ending aftertaste. The "what it feels like" space.
- Sub-fields: `emotional_palette`, `tension_adrenaline`, `tone_self_seriousness`, `cognitive_complexity`, `disturbance_profile`, `sensory_load`, `emotional_volatility`, `ending_aftertaste`. Each is a pair: `terms` (what the experience IS) and `negations` (what it deliberately is NOT, populated only when the boundary actually matters).
- Boundary: Subjective and experiential. Story mechanics or themes → plot_analysis / plot_events. Viewing occasion ("date night", "background movie") → watch_context. Craft choices → narrative_techniques.
- Example queries: "something unsettling but not gory, slow-burn tension that stays with you"; "light, breezy, low-stakes, leaves you smiling"; "emotionally devastating but in a cathartic way".

**watch_context** — Why and when to watch.
- Purpose: Viewing occasions, motivations, situational pulls. The "why and when" space.
- Sub-fields: `self_experience_motivations` (what the viewer seeks — "cheer me up", "challenge me"), `external_motivations` (external pulls — "for a first-date impression"), `key_movie_feature_draws` (specific draws — "great soundtrack", "short runtime"), `watch_scenarios` (occasions — "date night", "Sunday afternoon", "background movie").
- Boundary: Viewing situation, not internal content. Moment-to-moment feel → viewer_experience. What the movie is about → plot_analysis.
- Example queries: "something to put on in the background while cooking"; "a good first-date movie that won't alienate anyone"; "comfort rewatch for a rainy Sunday".

**narrative_techniques** — How the story is told.
- Purpose: Storytelling craft — structure, point of view, information control, characterization methods, narrative devices. The "how it is told" space.
- Sub-fields: `narrative_archetype`, `narrative_delivery`, `pov_perspective`, `characterization_methods`, `character_arcs`, `audience_character_perception`, `information_control`, `conflict_stakes_design`, `additional_narrative_devices`. Each holds short terms describing craft choices ("unreliable narrator", "non-linear timeline", "ensemble mosaic", "dramatic irony").
- Boundary: Craft, not content. Thematic content → plot_analysis. Literal events → plot_events. Techniques describe the delivery, not the story.
- Example queries: "told in reverse chronological order"; "found-footage mockumentary style"; "ensemble film where multiple storylines converge".

**production** — How and where physically made.
- Purpose: Filming locations and production techniques. The "how / where it was made" space.
- Sub-fields: `filming_locations` (proper-noun place names — cities, regions, landscapes), `production_techniques` (craft terms — "practical effects", "shot on 16mm", "motion capture", "single-take long shot").
- Boundary: Physical making, not storytelling or narrative craft. Storytelling structure / POV / pacing devices → narrative_techniques. Genre or theme → plot_analysis.
- Example queries: "movies filmed in New Zealand landscapes"; "shot entirely on 16mm film"; "heavy practical creature effects, minimal CGI".

**reception** — What critics and audiences thought, by named aspect.
- Purpose: Specific praised/criticized qualities and overall reception shape. The "what people thought" space.
- Sub-fields: `reception_summary` (short prose), `praised_qualities` (terms for specific praised aspects — "lead performance", "cinematography", "emotional resonance"), `criticized_qualities` (terms for specific criticized aspects — "pacing in the third act", "thin supporting characters").
- Boundary: Nuanced response to specific named aspects. Broad "critically acclaimed" / "fan favorite" without naming an aspect → metadata reception (scalar). Use only when the trait names the aspect being praised or criticized.
- Example queries: "praised for its cinematography and production design"; "widely criticized for a rushed third act"; "cult reception despite a mixed initial critical reaction".

## Body authoring — match the ingest side

Each body is the query-side text that gets embedded and compared cosine-wise to the corresponding ingest-side vector. Match the **vocabulary**, **verbosity**, and **register** the ingest side uses for that space. **The per-sub-field descriptions on each Body class (PlotEventsBody, ViewerExperienceBody, WatchContextBody, NarrativeTechniquesBody, PlotAnalysisBody, ProductionBody, ReceptionBody) carry the density, register, vocabulary, and synonym targets — read them before populating.**

### Per-space register table

| Space | Density per active sub-field | Phrase length | Voice | Negations |
|---|---|---|---|---|
| `plot_events.plot_summary` | One body: 1–3 sentences for events; OR fragments per motif | n/a | Past-tense third-person synopsis prose. Restate only user-grounded events / motifs. NEVER fabricate plot detail. | n/a |
| `plot_analysis` | All 6 fields when grounded | varies | Generic, no proper nouns. **Reuse load-bearing thematic terms across `elevator_pitch` / `plot_overview` / `thematic_concepts` / `character_arcs`** — the ingest side does this on purpose to weight the central concept. | n/a |
| `viewer_experience` | 5–10 terms / section when active | 1–5 words | Search-query vernacular. Slang OK ("tearjerker", "white knuckle", "snoozefest"). First/second-person fragments OK ("kept me guessing", "made me nauseous"). | **1–3 default per active section, naming the section's closest-opposite axis** |
| `watch_context` | 4–8 terms / section when active | 1–6 words | Intent-framed search vernacular ("turn my brain off", "stoned movie", "feel small in the universe"). All four sections fire when the trait grounds them. | n/a |
| `narrative_techniques` | 1–3 terms / section, 4–6 sections active typical | 1–6 words | Canonical craft labels VERBATIM: "Chekhov's gun", "ticking clock deadline", "non-linear timeline", "unreliable narrator", "dramatic irony". Do NOT paraphrase. | n/a |
| `production` | filming_locations 1–3, production_techniques 0–2 | strings / 1–3 words | **Match the user's geographic specificity exactly — do NOT add finer detail than asked.** | n/a |
| `reception` | praised/criticized 3–6 each, summary 1–2 sentences | 1–3 words (terms) | Adjective+noun aspect labels for terms ("evocative score", "convoluted plot"). Evaluative third-person prose for `reception_summary`. | n/a |

### Phrasing rules (term-list spaces — viewer_experience, watch_context, narrative_techniques, reception term lists)

These are the same rules used at ingest time to author the embedded text. Match them so query and document vectors land in the same neighborhood.

1. **Write phrases like search queries, not sentences.** Good: "edge of your seat", "date night movie", "turn my brain off". Bad: "This movie will keep you on the edge of your seat."
2. **Use common, everyday user wording.** Prefer everyday language over academic terms. ("kept me guessing" beats "narratively unpredictable".)
3. **Include redundant near-duplicates on purpose, but TRUE synonyms only.** Synonyms that mean the same thing ("uplifting / inspiring / hopeful"), paraphrases ("kept me guessing / unpredictable"), and slang you actually understand ("tearjerker", "gorefest", "edge of your seat") all qualify. **Adjacent concepts that drift the meaning do NOT** — "haunting" → "eerie / supernatural" is drift, not synonymy ("eerie" implies a different feel; "supernatural" is a content claim, not a feel claim). "Bittersweet" → "tragic / melancholic" drifts too — tragic is stronger than bittersweet, melancholic is adjacent.
4. **No proper nouns, no character names, no plot details** in the term-list spaces.
5. **Use canonical craft terms verbatim in `narrative_techniques`** — "Chekhov's gun", "unreliable narrator", "non-linear timeline", "ticking clock deadline". Established technique names should NOT be paraphrased.

### The substitution test

Before adding a synonym to a term list, run this test:

> **"Could I show this term to the user instead of their original word, and would they say yes, that's the same thing?"**

If yes → safe to include. If no → drop it; it's drift, not synonymy. Drift terms shift the retrieval target away from what the user asked for and hurt cosine match against the films they actually want.

### Negations (viewer_experience only)

**Both `terms` and `negations` point at the SAME retrieval target.** They are complementary phrasings of the same concept, not opposites:

- `terms` = "what films matching this body ARE", no `not`/`no` prefix.
- `negations` = "what films matching this body are NOT", with `not`/`no` prefix.

Both fields cluster on the same side of the embedding. They reinforce each other — `"happy"` and `"not sad"` are the same idea phrased two ways.

Correct pairings (terms and negations point the same direction):

- Feel-good body: `terms = ["happy", "uplifting", "joyful"]` + `negations = ["not sad", "not depressing", "not bleak"]`.
- Gory body (looking for gore-heavy films): `terms = ["gory", "bloody", "graphic violence"]` + `negations = ["not peaceful", "not gentle", "not for kids"]`.
- Non-gory body (looking for restrained slashers): `terms = ["light scares", "tame violence", "restrained"]` + `negations = ["no gore", "not too gory", "not bloody"]`.

Contradictory pairings — DO NOT EMIT:

- `terms = ["gory"]` + `negations = ["not too gory"]` — contradicts itself.
- `terms = ["happy"]` + `negations = ["not happy"]` — contradicts itself.

**Field signature is mechanical.** `terms` never carries `not`/`no` prefix. `negations` always does. If you find yourself writing `"not too X"` inside a `terms` list, move it to `negations` and check that the body's direction matches.

**Default-populate negations.** The ingest side routinely emits 1–3 negations per active section even when no user-side boundary was named. Author 1–3 negations per active section that REINFORCE the direction the terms already point — same retrieval target, opposite-syntactic-form (`not`/`no`) phrasing. Suppress only when the section is barely populated.

**Polarity flips at the trait level, never inside the body.** When the user wants to AVOID gory films, the trait gets `polarity=negative` upstream and the body still searches *affirmatively* for gory films (`terms=["gory", "bloody", ...]`, `negations=["not peaceful", "not for kids", ...]`). The orchestrator inverts the score downstream. The body never inverts.

### Plot events — motifs and specific events (no fabrication)

Two valid shapes for `plot_events.plot_summary`:

- **Specific event query** ("a heist that falls apart due to crew betrayal"): 1–3 dense past-tense sentences restating only what the user named. Generic agents are fine ("a heist crew", "the protagonist"); specific names, settings, side-events, motives, or outcomes you fabricated are not.
- **Motif query** ("clowns as a recurring motif"): short fragments naming the motif in the contexts a real synopsis would mention it. For "clowns": `"the clown. is a clown. and then the clown. encounters a clown. the clown returns."` This retrieves films that contain the motif WITHOUT inventing a plot around it.

The failure mode is fabricating plot detail. "Clowns" → "a clown chases a woman through a carnival as her boyfriend tries to save her" shifts the retrieval target away from real motif occurrences.

### Production specificity

`filming_locations` matches the user's geographic specificity exactly. If the user said "Iceland", emit `["Iceland"]` — do NOT add `"Reykjavik, Iceland"` or specific landmarks. Adding finer detail changes what the user asked for. The ingest side stores raw IMDB filming-location strings (city + country); query-side specificity should be capped at what the user supplied.

### Other authoring rules (apply across all spaces)

- **Populate only sub-fields the aspects genuinely land in.** Empty sub-fields are valid and expected — padding dilutes the query vector.
- **Translate the register only when it differs.** Rewrite user-side phrasing into the space's ingest register when they differ ("it's so dumb but in a fun way" → `tone_self_seriousness.terms` = ["unselfserious", "knowingly silly"]). **Keep user phrasing verbatim when it already matches** — `viewer_experience` and `watch_context` use search-query vernacular, so "tearjerker", "edge of your seat", "date night movie" pass through, not get rewritten into critic-prose.
- **No numerics.** Years, runtimes, ratings, budgets, box-office figures route to metadata.
- **One body per space.** Fold every aspect a space owns into ONE entry's `query.content`; do not split coverage across multiple entries with the same `query.space`.

## Decomposing into aspects

A trait often consolidates multiple qualifiers — tone, pacing, occasion, thematic flavor — into one rich phrase. Aspects are the atoms.

- **Compose interlocking signals.** "Darkly funny" is ONE aspect, not two. Cosine search performs better when composed signals stay composed than when split and re-intersected; "dark" alone and "funny" alone embed to different regions than "darkly funny" together.
- **Split only on genuinely orthogonal axes.** "Tense thriller for a lighthearted Friday night" splits into two aspects (mood + occasion), each landing in a different space.
- **One aspect can implicate multiple spaces.** "Slow-burn" pulls plot_events pacing AND viewer_experience tension. "Atmospheric" pulls viewer_experience sensory_load AND possibly watch_context scenario. "Epic" often pulls plot_analysis (genre shape) + viewer_experience (emotional scale).
- **A space can absorb multiple aspects.** When two distinct aspects both land in the same space, fold both into ONE body for that space — do not emit two `space_queries` entries for the same space. The schema's merge validator catches accidental duplication, but author cleanly.

The blob-handling failure mode — treating a multi-dimensional description as one undifferentiated concept and collapsing it onto a single space — drops real signal. Decompose first into aspects, then route per space.

## Boundaries

If one of these slips through to this endpoint, fold what fits into the closest space and surface the gap honestly — out-of-scope content stays out of the bodies.

- **Canonical concept tags** (genre labels, source-material types, narrative-mechanic enum members like PLOT_TWIST) → keyword endpoint, not semantic.
- **Numeric / scalar metadata** (year ranges, runtime, popularity, broad "critically acclaimed" without naming an aspect) → metadata endpoint. Specific praised/criticized aspects (cinematography, performances, third-act pacing) DO belong in semantic reception.
- **Named entities** (people, characters, franchises, studios, awards) → their dedicated endpoints.

## Positive-presence invariant

Bodies describe what to FIND, regardless of polarity. Negation is committed upstream on the trait that owns this call and applied later by the orchestrator. Never invert, negate, or "undo" an exclusion in the body. If the user's intent was exclusion, upstream already rewrote it as positive-presence framing; produce the body that retrieves movies HAVING the target.

The viewer_experience `negations` sub-fields are NOT polarity — they describe the boundary the trait wants the *match* to respect (e.g. "tense, but not gory"), not whether to include or exclude.

## Authoring `strengths` and `weaknesses` per space candidate

Each `space_candidates` entry carries a vector space plus two short prose fields. Frame both operationally — what does this space's embedding do at retrieval time, given the aspects under consideration? Four shapes recur:

- **clean** — strengths name the aspects the space genuinely owns; weaknesses = "none". Common when the aspect lands cleanly in one space's ingest-side vocabulary.
- **under-coverage** — strengths name the aspects owned; weaknesses lead with `under-coverage:` and name aspects the space's boundary redirects elsewhere (occasion → watch_context; craft → narrative_techniques; scalar reception → metadata; etc.).
- **over-coverage** — strengths name the aspects owned; weaknesses lead with `over-coverage:` and name what the embedding would ALSO match beyond the slice (e.g. plot_analysis on a broad genre query pulls every adjacent thematic shape).
- **both** — strengths name the partial slice owned; weaknesses names BOTH `under-coverage:` (what's missed) and `over-coverage:` (what's over-pulled).

The over-coverage axis is what the commitment phase uses to decide whether to fire a sibling endpoint that refines this one. Surface it honestly even when this space is the best semantic fit.

## Where the semantic analysis lives

In single-endpoint buckets the analysis (`aspects` / `space_candidates`) and the commitment (`role_exploration` / `role` / `space_queries`) live together in one `SemanticParameters`.

In multi-endpoint buckets the analysis is hoisted to a bucket-level `semantic_walk` field that sits BEFORE the `coverage_exploration` / `coverage_commitments` commitment, while the commitment lives in a thin `semantic_parameters` slot AFTER it. `role_exploration` + `role` stay paired with the commitment because they're semantic-internal selectivity, not space-grounded analysis. Refer to the schema descriptors for exact field locations.

## Routing trust and abstention

Upstream routing chose semantic as a candidate for this category, but the walk above the commitment is allowed to surface "no clean fit" when the call's intent doesn't actually land in any of the 7 spaces. In single-endpoint buckets, signal that with `should_run_endpoint=false`. In multi-endpoint buckets, set `coverage_commitments.semantic.verdict = "abstain"` and leave `semantic_parameters` null. Do not coerce out-of-scope intent into a noisy multi-space body. When the walk DOES surface clean coverage, produce the best multi-space query plan the schema allows.

## Sibling context and body shaping

In multi-endpoint buckets the user message carries a `<sibling_categories>` block listing the other categories Step 3 committed for the same trait, plus a `combine_mode` attribute. The bucket-level "Reading sibling context" section spells out the general protocol; the semantic-specific consequence:

- When a sibling's `retrieval_intent` paraphrases the same conceptual slice your call carries, your space-coverage choice should narrow toward the spaces that genuinely complement what the sibling is fetching, not duplicate it. Under `facets` fold, duplicate coverage across categories means the trait's compound is multiplying near-zero contributions on both sides of the same axis when either embedding misses; a focused, complementary semantic body is more robust than a broad one.
- Under `framings` fold, redundant coverage is the design — a clean body in any one of the categories carries the trait. Choose the spaces that best embed the call's intent and let the MAX absorb whatever overlaps.
- Under `single` apply space selection on the call's intent alone.

Body content per space and the per-sub-field register rules are unchanged. Sibling context only affects the breadth of space selection and whether borderline-fit spaces should be included or dropped given the fold rule's tolerance for redundancy.
