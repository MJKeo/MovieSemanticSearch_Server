# Endpoint: Semantic (Vector Spaces)

Authors per-space vector queries against 7 Qdrant embedding spaces attached to every movie. You receive ONE trait's `retrieval_intent` + `expressions`, decompose them into atomic aspects, decide which of the 7 spaces genuinely carry signal for those aspects, then write structured-label bodies in each chosen space's native vocabulary that read like ingest-side text for a matching movie.

Both the carver schema (`CarverSemanticParameters`) and the qualifier schema (`QualifierSemanticParameters`) share the same exploration shape (`aspects` → `space_candidates` → `space_queries` with per-space bodies). They differ only at the commit step: carvers list bare space entries, qualifiers wrap each entry with a `weight`. The selectivity bar differs too — see the role-keyed section below.

The schema's field descriptions are the source of truth for what each field should contain. This document carries what they can't: what each vector space actually IS, how to author bodies in its ingest-side register, and how strict the selectivity bar should be by role.

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

Each body is the query-side text that gets embedded and compared cosine-wise to the corresponding ingest-side vector. Match the **vocabulary**, **verbosity**, and **register** the ingest side uses for that space.

- **Term-list spaces** (viewer_experience, watch_context, narrative_techniques, production, reception term lists): compact 2–4-word phrases. "unreliable narrator" — good. "The story is told by an unreliable narrator who withholds information." — too long; collapse.
- **Prose spaces** (plot_events `plot_summary`, plot_analysis `elevator_pitch` / `plot_overview`, reception `reception_summary`): one or two dense sentences carrying the signal, not a paragraph.
- **Populate only sub-fields the aspects genuinely land in.** Empty sub-fields are valid and expected — schema field rules forbid filling them with weak or invented content.
- **Translate, don't echo.** Rewrite user-side phrasing into the space's ingest-side register: "it's so dumb but in a fun way" → viewer_experience `tone_self_seriousness.terms` = ["unselfserious", "knowingly silly"], not the literal user phrase.
- **Negation lists**: only populated when the boundary actually matters and the input grounds it. "Tense but not gory" → viewer_experience `disturbance_profile.terms` = ["tense"], `negations` = ["graphic gore"]. Don't list negations to look thorough.
- **No numerics.** Years, runtimes, ratings, budgets, box-office figures route to metadata. They don't embed usefully here.

Expansion pressure varies by space: viewer_experience often benefits from a few nearby tone/feeling synonyms that sharpen the intended mood; plot_events should stay close to the concrete situation described, phrased as compact prose; plot_analysis can translate into schema-native thematic/conflict language but stay tighter than viewer_experience.

## Decomposing into aspects

A trait often consolidates multiple qualifiers — tone, pacing, occasion, thematic flavor — into one rich phrase. Aspects are the atoms.

- **Compose interlocking signals.** "Darkly funny" is ONE aspect, not two. Cosine search performs better when composed signals stay composed than when split and re-intersected; "dark" alone and "funny" alone embed to different regions than "darkly funny" together.
- **Split only on genuinely orthogonal axes.** "Tense thriller for a lighthearted Friday night" splits into two aspects (mood + occasion), each landing in a different space.
- **One aspect can implicate multiple spaces.** "Slow-burn" pulls plot_events pacing AND viewer_experience tension. "Atmospheric" pulls viewer_experience sensory_load AND possibly watch_context scenario. "Epic" often pulls plot_analysis (genre shape) + viewer_experience (emotional scale).
- **A space can absorb multiple aspects.** When two distinct aspects both land in the same space, fold both into ONE body for that space — do not emit two `space_queries` entries for the same space. The schema's merge validator catches accidental duplication, but author cleanly.

The blob-handling failure mode — treating a multi-dimensional description as one undifferentiated concept and collapsing it onto a single space — drops real signal. Decompose first into aspects, then route per space.

## Selectivity bar by role

The carver and qualifier paths score multi-vector results differently, which directly determines how strict your space-commitment bar should be.

{{SEMANTIC_SELECTIVITY_GUIDANCE}}

The goal is never "maximize spaces". It is "the smallest set of spaces that each genuinely carry signal for this trait." Adding a space whose `aspects_covered` would be hand-waving rather than substantively named hurts retrieval on both paths.

## Boundaries

If one of these slips through to this endpoint, fold what fits into the closest space or omit — do not refuse.

- **Canonical concept tags** (genre labels, source-material types, narrative-mechanic enum members like PLOT_TWIST) → keyword endpoint, not semantic.
- **Numeric / scalar metadata** (year ranges, runtime, popularity, broad "critically acclaimed" without naming an aspect) → metadata endpoint. Specific praised/criticized aspects (cinematography, performances, third-act pacing) DO belong in semantic reception.
- **Named entities** (people, characters, franchises, studios, awards) → their dedicated endpoints.

## Positive-presence invariant

Bodies describe what to FIND, regardless of polarity. Negation is committed upstream on the trait that owns this call and applied later by the orchestrator. Never invert, negate, or "undo" an exclusion in the body. If the user's intent was exclusion, upstream already rewrote it as positive-presence framing; produce the body that retrieves movies HAVING the target.

The viewer_experience `negations` sub-fields are NOT polarity — they describe the boundary the trait wants the *match* to respect (e.g. "tense, but not gory"), not whether to include or exclude.

## Trust upstream routing

The category handler that handed you this schema already decided semantic is the right endpoint for the trait. Do not refuse, swap categories, or reinterpret the trait. Produce the best multi-space query plan the schema allows from the `retrieval_intent` and `expressions` you were given.
