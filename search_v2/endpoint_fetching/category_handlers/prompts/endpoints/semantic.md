# Endpoint: Semantic (Vector Spaces)

## Purpose

Authors per-space vector queries against 7 Qdrant embedding spaces attached to every movie. The LLM (1) picks one or more of the 7 spaces that genuinely carry signal for the requirement, (2) authors a structured-label **body** in each selected space's native vocabulary that reads like ingest-side text for a matching movie, and (3) identifies a `primary_vector` among the populated spaces for downstream single-space execution paths.

Both filter-mode (dealbreaker-style) and trait-mode (preference-style) findings use the same schema — the two paths read the payload differently (filter uses only the `primary_vector` entry; trait uses every entry with its weight), but the LLM always populates the full payload.

## Canonical question

"Which embedding space(s) best capture this requirement, and what text — in each space's native vocabulary — would a matching movie's ingest text look like?"

## Capabilities

- Similarity search across 7 specialized vector spaces, each targeted at a different dimension of a movie.
- Handles free-form thematic, tonal, experiential, occasion-based, storytelling-craft, production-method, and reception-shape requirements that have no canonical registry member.
- Multi-space queries: a single requirement often lands across multiple spaces when it carries multiple dimensions (e.g. "slow-burn atmospheric melancholy" splits across plot_events pacing, viewer_experience tension/sensory, watch_context occasion).

## Boundaries (what does NOT belong here)

- Canonical concept tags (genre, source material, cultural tradition, narrative mechanics like PLOT_TWIST) → keyword endpoint.
- Structured numeric attributes (dates, runtime, rating, ratings-as-scalars like popularity / reception) → metadata endpoint. Broad "critically acclaimed" without naming the aspect praised → metadata reception, not semantic reception.
- Named entities, franchises, studios, awards → their dedicated endpoints.

## The 7 vector spaces

Each entry shows the space's **purpose**, the **sub-fields** embedded at ingest (what the body must populate), the **boundary** (what would be a misroute), and 2–3 canonical **example queries**.

**plot_events** — What literally happens.
- Purpose: Chronological plot — actions, events, character-arc beats on screen. The "what happens" space.
- Embedded sub-fields: `plot_summary` (dense prose summary of concrete events and characters).
- Boundary: Literal and concrete. Themes / genre signatures / conflict archetypes → plot_analysis. How the story is told → narrative_techniques.
- Example queries: "a heist that unravels when a team member betrays the crew"; "a lone survivor crosses a frozen continent to find their family"; "the one where strangers trapped in an elevator discover a murderer among them".

**plot_analysis** — What type of story thematically.
- Purpose: Genre signatures, thematic concepts, conflict archetypes, character-arc patterns — what kind of story it is. The "what type of story" space.
- Embedded sub-fields: `elevator_pitch` (one-sentence thematic capsule), `plot_overview` (short prose), `genre_signatures` (terms), `conflict_type` (archetype terms, e.g. "man vs self"), `thematic_concepts` (abstract theme terms, e.g. "grief", "redemption", "moral compromise"), `character_arcs` (arc-pattern terms, e.g. "fall from grace", "coming of age").
- Boundary: Categorical and thematic shape. Specific events ("the one where X happens") → plot_events. Emotional/cognitive watching experience → viewer_experience.
- Example queries: "stories about grief and reconciliation between estranged siblings"; "man-vs-nature survival dramas"; "tragic fall-from-grace character arcs".

**viewer_experience** — What it feels like to watch.
- Purpose: Emotional, sensory, and cognitive experience of sitting through the movie. Tone, tension profile, disturbance level, cognitive load, ending aftertaste. The "what it feels like" space.
- Embedded sub-fields: `emotional_palette`, `tension_adrenaline`, `tone_self_seriousness`, `cognitive_complexity`, `disturbance_profile`, `sensory_load`, `emotional_volatility`, `ending_aftertaste`. Each is a pair: `terms` (what the experience IS) and `negations` (what it deliberately is NOT, when a boundary matters).
- Boundary: Subjective and experiential. Story mechanics or themes → plot_analysis / plot_events. Viewing occasion ("date night", "background movie") → watch_context. Craft choices → narrative_techniques.
- Example queries: "something unsettling but not gory, slow-burn tension that stays with you"; "light, breezy, low-stakes, leaves you smiling"; "emotionally devastating but in a cathartic way".

**watch_context** — Why and when to watch.
- Purpose: Viewing occasions, motivations, situational pulls — the contexts in which this movie is the right choice. The "why and when" space.
- Embedded sub-fields: `self_experience_motivations` (what the viewer seeks — "cheer me up", "challenge me"), `external_motivations` (external pulls — "for a first-date impression"), `key_movie_feature_draws` (specific draws — "great soundtrack", "short runtime"), `watch_scenarios` (occasions — "date night", "Sunday afternoon", "background movie").
- Boundary: Viewing situation, not internal content. Moment-to-moment feel → viewer_experience. What the movie is about → plot_analysis.
- Example queries: "something to put on in the background while cooking"; "a good first-date movie that won't alienate anyone"; "comfort rewatch for a rainy Sunday".

**narrative_techniques** — How the story is told.
- Purpose: Storytelling craft — structure, point of view, information control, characterization methods, narrative devices. The "how it is told" space.
- Embedded sub-fields: `narrative_archetype`, `narrative_delivery`, `pov_perspective`, `characterization_methods`, `character_arcs`, `audience_character_perception`, `information_control`, `conflict_stakes_design`, `additional_narrative_devices`. Each holds short terms describing craft choices ("unreliable narrator", "non-linear timeline", "ensemble mosaic", "dramatic irony").
- Boundary: Craft, not content. Thematic content → plot_analysis. Literal events → plot_events. Techniques describe the delivery, not the story.
- Example queries: "told in reverse chronological order"; "found-footage mockumentary style"; "ensemble film where multiple storylines converge".

**production** — How and where physically made.
- Purpose: Filming locations and production techniques — the craft and circumstances of making the film. The "how / where it was made" space.
- Embedded sub-fields: `filming_locations` (proper-noun place names — cities, regions, landscapes), `production_techniques` (craft terms — "practical effects", "shot on 16mm", "motion capture", "single-take long shot").
- Boundary: Physical making, not storytelling or narrative craft. Storytelling structure / POV / pacing devices → narrative_techniques. Genre or theme → plot_analysis.
- Example queries: "movies filmed in New Zealand landscapes"; "shot entirely on 16mm film"; "heavy practical creature effects, minimal CGI".

**reception** — What critics and audiences thought.
- Purpose: Specific praised/criticized qualities and overall reception shape. The "what people thought" space.
- Embedded sub-fields: `reception_summary` (short prose), `praised_qualities` (terms for specific praised aspects — "lead performance", "cinematography", "emotional resonance"), `criticized_qualities` (terms for specific criticized aspects — "pacing in the third act", "thin supporting characters").
- Boundary: Nuanced response to specific named aspects. Broad "critically acclaimed" / "fan favorite" without naming an aspect → metadata reception (scalar), not this space. Use only when the query names the aspect being praised or criticized.
- Example queries: "praised for its cinematography and production design"; "widely criticized for a rushed third act"; "cult reception despite a mixed initial critical reaction".

## Body-authoring principles

Each space's body is the query-side text that will be embedded and compared against ingest-side text for every candidate movie. Same vocabulary, same shape, same style.

- **Populate only the sub-fields where the concept genuinely lands.** Empty lists are valid and expected. Filling a sub-field with weak or invented content dilutes the query vector and hurts matching.
- **Use the space's native vocabulary.** Each space has its own preferred phrasings — themes and conflict types for plot_analysis, emotional-palette adjectives for viewer_experience, motivation phrases for watch_context, craft terms for narrative_techniques, location names and technique terms for production. Write in the register the sub-fields already use on the ingest side.
- **Term lists hold compact phrases, not sentences.** Two-to-four-word items are typical. "unreliable narrator" — good. "The story is told by an unreliable narrator who withholds information from the audience." — too long; collapse to the key phrase.
- **Prose fields stay brief and dense** (plot_summary, elevator_pitch, reception_summary, identity_pitch, identity_overview, plot_overview) — one or two sentences carrying the signal, not a full paragraph.
- **Translate into the target space's native format, not the user's raw wording.** Some spaces want compact prose (especially plot_events); others want short labeled phrases (viewer_experience, watch_context, narrative_techniques, production). Small expansions are good when they restate the same supported signal in the space's native vocabulary. Do not add new story facts, themes, motivations, or evaluative claims that the input does not support.
- **Expansion pressure varies by space.** viewer_experience often benefits from a few nearby feeling/tone terms that sharpen the intended mood. plot_events should stay close to the concrete situation described, phrased as compact prose. plot_analysis can translate into schema-native thematic/conflict language but stay tighter than viewer_experience.
- **No numeric values** — no ratings, years, runtimes, budgets, or box-office figures. Numerics route to metadata/award endpoints and would not embed usefully.
- **Do not restate the user's original phrasing verbatim** when it does not match the space's vocabulary. Translate it into the terms the ingest side would use for a matching movie.

## Decomposing multi-qualifier requirements

A requirement often consolidates multiple qualifiers about the desired viewing experience — tone, pacing, occasion, thematic flavor — into one rich phrase. Before picking spaces, split it into atomic qualifiers.

- **Split on conjunctions and commas.** "funny, dark, and thought-provoking with a cozy date-night vibe" → "funny", "dark", "thought-provoking", "cozy", "date-night vibe".
- **Same-dimension near-synonyms stay as one unit.** "dark and gritty" is one tone qualifier, not two.
- **A single qualifier can implicate multiple spaces.** "slow-burn" is pacing (plot_events) + tension profile (viewer_experience). "atmospheric" is viewer_experience sensory_load + possibly watch_context scenario. "epic" often spans plot_analysis (genre shape) + viewer_experience (emotional scale).
- **A qualifier that maps cleanly to no space is flagged, not force-routed.** Better to leave a qualifier out of every space than to pollute a space with content that does not belong in its vocabulary.

**Blob-handling is the primary failure mode.** Treating a multi-dimensional description as one undifferentiated concept and collapsing it onto a single space (usually plot_analysis or viewer_experience — the two broadest available spaces) drops real signal. Decompose first, then pick spaces per dimension.

## Space selection discipline

- Pick the **smallest set of spaces** that each provide genuinely strong signal. Multiple qualifiers can land in the same space. Do not add a space just because you can weakly justify it — a space carrying barely-there signal dilutes the result.
- The goal is not to maximize the number of spaces. Multiple spaces are right only when the requirement genuinely spans dimensions.
- `primary_vector` is a **retrospective** pick from the populated spaces — the single most effective space for an execution path that requires one. Populate `space_queries` honestly first, then identify the strongest entry among them. Do NOT commit to a single "best" space up front and then feel structurally pressured to keep the list short.

## Weight assignment (for trait-mode findings)

Each populated space carries one of two weights. The distinction is about how much of the user's desired experience that space carries, not how confident you are that the space is correct.

- **central** — The space carries a major part of the intended match: a headline qualifier or multiple load-bearing qualifiers. The user's request would feel fundamentally different if this space were missing.
- **supporting** — The space carries meaningful supporting signal that rounds out the experience but is not load-bearing on its own. The user's request would still be recognizable without it, just less complete.

A single request can have zero, one, or several central spaces. `central` does NOT imply unique — multiple spaces can each be central when each carries a major part of the match. If nothing clearly stands out as load-bearing, all-supporting is acceptable — that is a truthful signal that the preference is broad-and-balanced rather than focused on one dimension. If a space would be below supporting (barely-there signal), do not select it at all. Spreading weight across many marginal spaces dilutes the query.

For filter-mode findings, `weight` is ignored at execution time (only the `primary_vector` entry is used) — populate it honestly anyway; the payload is symmetric across paths.
