# Concept Tags

Binary deal-breaker tags extracted by LLM classification. These are concepts
that users expect as yes/no when searching ("movies with X") and that cannot
be reliably retrieved through vector similarity alone.

Replaces the V1 `production_keywords` generation slot. Stored as
`concept_tag_ids INT[]` on `movie_card` with a GIN index.

---

## Design Principles

1. **Binary, not spectrum.** Every tag answers a yes/no question: "does this
   movie have X?" Concepts where degree matters (how scary, how dark, how
   funny) belong in vector spaces, not here.

2. **Not already covered.** Tags must fill a gap not served by `genre_ids`,
   `overall_keywords` (225-term taxonomy), `source_material_type_ids`,
   `franchise_membership`, `movie_awards`, or other movie_card fields.

3. **Primary search concept.** Users search for the tag as the main filter
   ("revenge movies"), not just a secondary detail. The tag must be something
   where a user would consider a result WRONG if the movie didn't have it.

4. **Classifiable from available inputs.** The LLM must be able to identify
   the concept from plot_keywords, overall_keywords, plot_summary (Wave 1),
   craft_observations (Wave 1), emotional_observations (Wave 1), title +
   year (parametric knowledge), and/or viewer_experience metadata (Wave 2).

---

## Finalized Tag List

### Narrative Structure / Storytelling Devices

Binary structural choices in how the story is told.

| Tag | Description | Example user query | Classification signals |
|-----|-------------|-------------------|----------------------|
| `PLOT_TWIST` | Movie has a significant plot twist or surprise revelation that recontextualizes part or all of the story. Not limited to ending twists — includes mid-story reveals, identity twists, betrayal reveals. | "movies with a plot twist", "twist movies" | plot_keywords ("surprise ending" 285, "plot twist" 47), narrative_techniques.information_control, plot_summary, parametric knowledge |
| `TWIST_VILLAIN` | A character revealed as the villain is a surprise — the villain's identity is a twist. Subset of PLOT_TWIST but distinct search intent. | "movies with a twist villain" | plot_summary (character reveals), plot_keywords, parametric knowledge |
| `TIME_LOOP` | Character(s) relive the same time period repeatedly. Distinct from TIME_TRAVEL (which is in `overall_keywords`). | "time loop movies", "Groundhog Day-type movies" | plot_keywords ("time loop" 97), plot_summary |
| `NONLINEAR_TIMELINE` | Story is told out of chronological order as a deliberate structural choice. Not just "has a flashback" — the non-chronological structure must be a defining feature. | "non-linear movies", "movies told out of order" | plot_keywords ("nonlinear timeline" 223), craft_observations, plot_summary structure |
| `UNRELIABLE_NARRATOR` | The narrator or POV character's account is revealed as untrustworthy. | "unreliable narrator movies" | plot_summary structure (narrative contradictions), craft_observations, parametric knowledge. Note: sparse tagging (25 in plot_keywords) — LLM inference from plot structure is the primary signal. |
| `OPEN_ENDING` | The story deliberately leaves its central question unresolved or ambiguous. Not every loose thread qualifies — the ambiguity must be intentional and central. | "movies with ambiguous endings", "open-ended movies" | plot_keywords ("ambiguous ending" 65), plot_summary ending, parametric knowledge |
| `SINGLE_LOCATION` | Nearly all action takes place in one location (bottle movie). The constraint must be a defining feature, not just "most scenes are in one building." | "one-location movies", "bottle movies" | plot_summary (all events in one place), parametric knowledge |
| `BREAKING_FOURTH_WALL` | Characters directly address the audience or acknowledge they are in a movie. Must be a notable, deliberate choice, not a brief aside. | "movies that break the fourth wall" | plot_keywords (297), craft_observations, parametric knowledge |

### Plot Archetypes / Thematic Patterns

The central premise or driving force of the movie. Tag applies when the
concept IS the movie — not just an element that appears somewhere in the plot.

| Tag | Description | Example user query | Classification signals |
|-----|-------------|-------------------|----------------------|
| `REVENGE` | The central plot is driven by a character seeking vengeance. Revenge must be the primary narrative engine, not a subplot. | "revenge movies" | plot_keywords ("revenge" 2,199), plot_summary |
| `UNDERDOG` | Protagonist is clearly outmatched and overcomes the odds. The power imbalance must be a defining feature of the story. | "underdog movies", "underdog sports movies" | plot_summary (protagonist vs. superior opposition), thematic_observations |
| `KIDNAPPING` | The plot centers on a kidnapping or abduction. Must be central to the story, not just one event among many. | "kidnapping movies", "abduction movies" | plot_keywords ("kidnapping" 842), plot_summary |
| `CON_ARTIST` | Protagonist is a con artist, grifter, or scammer — the movie is about deception and manipulation as a craft. Distinct from "Heist" keyword (which covers theft/robbery). | "con artist movies", "scam movies" | plot_keywords, plot_summary (deception-driven plot), parametric knowledge |

### Setting Concepts

Settings that users search for as the primary filter. Tag applies when the
setting is a defining characteristic of the movie.

| Tag | Description | Example user query | Classification signals |
|-----|-------------|-------------------|----------------------|
| `POST_APOCALYPTIC` | Set after civilization's collapse. Distinct from "Dystopian Sci-Fi" keyword (oppressive society still functioning). Post-apocalyptic = society has fallen; dystopian = society is intact but oppressive. | "post-apocalyptic movies" | plot_keywords ("post apocalypse" 310), plot_summary, overall_keywords context |
| `HAUNTED_LOCATION` | Set in or centered around a haunted house, building, or specific location. Distinct from "Supernatural Horror" keyword (1,400) which is much broader (includes possession, curses, ghosts in any context). | "haunted house movies" | plot_keywords ("haunted house" 288), plot_summary |
| `SMALL_TOWN` | The small-town setting is central to the story's identity and atmosphere — not just incidental. Fargo, Hot Fuzz, Jaws. | "small town movies", "small town horror" | plot_keywords ("small town" 356), plot_summary setting descriptions |

### Character Concepts

| Tag | Description | Example user query | Classification signals |
|-----|-------------|-------------------|----------------------|
| `FEMALE_PROTAGONIST` | The lead character (or co-lead in a two-hander) is female. About the story's protagonist, not just "women appear in the movie." | "movies with a female lead" | plot_keywords ("female protagonist" 2,127), plot_summary character names and roles |
| `ENSEMBLE_CAST` | No single protagonist — multiple characters share roughly equal narrative weight. Pulp Fiction, Love Actually, Magnolia. | "ensemble movies", "ensemble cast" | plot_summary (multiple POV characters), cast structure, parametric knowledge |
| `ANTI_HERO` | Protagonist is morally ambiguous, operates outside conventional morality, or lacks traditional heroic qualities. The moral ambiguity must be a defining character trait. | "anti-hero movies" | plot_keywords ("anti hero" 205), plot_summary (protagonist behavior), parametric knowledge |

### Ending Type

Users frequently search for or specifically avoid movies based on how they
end. These are strong deal-breakers.

| Tag | Description | Example user query | Classification signals |
|-----|-------------|-------------------|----------------------|
| `HAPPY_ENDING` | Things work out for the protagonists. The overall resolution is positive/optimistic. | "movies with a happy ending" | plot_summary ending, parametric knowledge, emotional_observations |
| `SAD_ENDING` | The story ends predominantly sad or negatively for the protagonists — loss, failure, or death. Not just bittersweet. Distinct from "Tragedy" keyword (1,094) which is a genre classification — a movie can have a sad ending without being a Tragedy genre film. | "movies with a sad ending", "movies that end badly" | plot_summary ending, parametric knowledge, emotional_observations |

### Experiential Tags

Binary versions of experiential qualities that users search for as
deal-breakers. These overlap with viewer_experience vector content but serve
a different purpose: deterministic candidate retrieval in Phase 1, while
vectors handle degree/ranking in Phase 2.

The LLM sets these based on viewer_experience metadata and
emotional_observations — it's already making these judgments during
generation, so storing the binary verdict is low marginal cost.

| Tag | Description | Example user query | Classification signals |
|-----|-------------|-------------------|----------------------|
| `FEEL_GOOD` | The overall effect of watching the movie is uplifting and positive. Not just "has some nice moments" — the movie's trajectory and ending leave the viewer feeling good. Must meet a clear threshold, not borderline cases. | "feel-good movies", "uplifting movies" | viewer_experience emotional_palette, ending_aftertaste, emotional_observations. Note: "Feel-Good Romance" keyword (454) covers romance only — this tag covers all genres. |
| `TEARJERKER` | The movie is designed to make you cry — and people report that it does. Not inferred from sad plot events alone; based on audience/reviewer reports of emotional impact. | "tearjerker movies", "movies that will make me cry" | emotional_observations (reviewer reports of crying/emotional devastation), viewer_experience emotional_palette + ending_aftertaste |

### Content Flags

Things users search to AVOID. Extremely strong avoidance deal-breakers.

| Tag | Description | Example user query | Classification signals |
|-----|-------------|-------------------|----------------------|
| `ANIMAL_DEATH` | An animal dies on screen or as a significant plot point. The defining avoidance deal-breaker — DoesTheDogDie.com exists for this. | "does the dog die", filtering out movies where animals die | plot_keywords, plot_summary |

---

## Concepts NOT in This Enum — Routing to Other Systems

These concepts were evaluated as tag candidates but are better handled by
existing or planned V2 systems. This section exists to ensure the V2 query
understanding LLM knows HOW to route these concepts when users search for
them.

### Covered by `overall_keywords` (225-term keyword taxonomy)

When the V2 Phase 0 LLM encounters these concepts, it should emit the
corresponding `overall_keywords` term(s) for keyword-based retrieval via
`keyword_ids`.

| User concept | Route to keyword(s) | Notes |
|-------------|---------------------|-------|
| "conspiracy movies" | `Conspiracy Thriller` (384) | Conspiracy in non-thriller contexts is rare. Also consider `Political Thriller` (321), `Political Drama` (459). |
| "heist movies" | `Heist` (346), `Caper` (192) | CON_ARTIST tag covers the non-heist con/scam space. |
| "survival movies" | `Survival` (319) | |
| "road trip movies" | `Road Trip` (269) | |
| "time travel movies" | `Time Travel` (224) | TIME_LOOP tag covers the distinct time-loop sub-concept. |
| "coming of age movies" | `Coming-of-Age` (1,252) | |
| "superhero movies" | `Superhero` (839) | |
| "film noir" | `Film Noir` (926) | |
| "spy movies" | `Spy` (366) | |
| "disaster movies" | `Disaster` (247) | |
| "whodunit / murder mystery" | `Whodunnit` (693) | |
| "serial killer movies" | `Serial Killer` (281) | |
| "zombie movies" | `Zombie Horror` (333) | Horror-specific; non-horror zombies (Shaun of the Dead) may need vector fallback. |
| "vampire movies" | `Vampire Horror` (239) | Horror-specific; vampire romance/drama may need vector fallback. |
| "found footage movies" | `Found Footage Horror` (309) | Horror-specific; non-horror found footage (~50 movies) uses vector fallback. |
| "Christmas / holiday movies" | `Holiday` (804) + sub-types | |
| "mockumentary" | `Mockumentary` (80) | |
| "gangster movies" | `Gangster` (383) | |
| "true crime" | `True Crime` (832) | |
| "body swap movies" | `Body Swap Comedy` (69) | |
| "prison movies" | `Prison Drama` (180) | Drama-specific; prison movies in other genres use vector fallback or genre + keyword combo. |
| "one-man army / action hero" | `One-Person Army Action` (533) | |
| "martial arts movies" | `Martial Arts` (991) + sub-types | |
| "car chase movies" | `Car Action` (137) | |

### Covered by `source_material_type_ids`

| User concept | Route to enum value(s) |
|-------------|----------------------|
| "based on a true story" | `TRUE_STORY`, `BIOGRAPHY` |
| "book adaptations" | `NOVEL_ADAPTATION`, `SHORT_STORY_ADAPTATION` |
| "comic book movies" | `COMIC_ADAPTATION` |
| "remakes" | `REMAKE` |
| "video game movies" | `VIDEO_GAME_ADAPTATION` |
| "based on a play" | `STAGE_ADAPTATION` |
| "folklore / fairy tale movies" | `FOLKLORE_ADAPTATION` |

### Covered by `genre_ids`

| User concept | Route to genre(s) |
|-------------|-------------------|
| "documentaries" | `DOCUMENTARY` |
| "biographies / biopics" | `BIOGRAPHY` |
| "musicals" | `MUSICAL` |
| "war movies" | `WAR` |
| "westerns" | `WESTERN` |

### Covered by other structured fields

| User concept | Route to field |
|-------------|---------------|
| "award-winning movies" | `movie_awards` table — filter by ceremony + outcome |
| "movies from [country]" | `country_of_origin_ids` + `audio_language_ids` + language/nationality keywords |
| "movies on Netflix" | `watch_offer_keys` |
| "movies from the 80s" | `release_ts` soft constraint |
| "short movies" | `runtime_minutes` |
| "R-rated movies" | `maturity_rank` |
| "big budget movies" | `budget_bucket` |
| "box office hits" | `box_office_bucket` |
| "[franchise] movies" | `franchise_membership` table |
| "[actor/director] movies" | Role-specific posting tables |

### Handled by vector similarity (spectrum concepts)

These are concepts where degree matters. V2 Phase 2 cross-space rescoring
with threshold+flatten handles the binary aspect; raw similarity handles
ranking.

| User concept | Route to vector space(s) | Notes |
|-------------|-------------------------|-------|
| "scary movies" | `viewer_experience` (tension_adrenaline, disturbance_profile) | Degree matters — "scariest ever" is a ranking query. |
| "funny [genre] movies" | `viewer_experience` (tone_self_seriousness) | "Funny horror" needs semantic scoring, not binary tagging. |
| "dark / bleak movies" | `viewer_experience` (emotional_palette, tone_self_seriousness) | Spectrum from "slightly dark" to "nihilistic." |
| "intense / tense movies" | `viewer_experience` (tension_adrenaline) | Degree of intensity is the ranking signal. |
| "mind-bending movies" | `viewer_experience` (cognitive_complexity) + `narrative_techniques` | Degree of disorientation matters. PLOT_TWIST and NONLINEAR_TIMELINE tags cover the structural aspects. |
| "slow burn movies" | `viewer_experience` (tension_adrenaline) | Pacing is fundamentally a spectrum. |
| "disturbing movies" | `viewer_experience` (disturbance_profile) | Wide spectrum from "unsettling" to "unwatchable." |
| "cozy / comforting movies" | `viewer_experience` (emotional_palette) + `watch_context` | Degree matters. |
| "visually stunning movies" | `viewer_experience` (sensory_load) + `narrative_techniques` | Degree of visual quality. |
| "great soundtrack movies" | `watch_context` (key_movie_feature_draws) | Degree and subjectivity. |
| "cheesy movies" | `viewer_experience` (tone_self_seriousness) | Degree of cheesiness is the point. |
| "gritty movies" | `viewer_experience` (tone_self_seriousness) | Spectrum — realism/rawness varies. |
| "date night movies" | `watch_context` (watch_scenarios) | Viewing context, not binary movie attribute. |
| "turn your brain off movies" | `watch_context` (self_experience_motivations) + `viewer_experience` (cognitive_complexity) | Viewing context. |

### Derived at query time (not movie attributes)

| User concept | How to handle |
|-------------|--------------|
| "critically acclaimed" | `reception_score` threshold or `movie_awards` presence |
| "underrated / hidden gems" | High `reception_score` + low `popularity_score` (anti-correlation) |
| "cult classics" | Emerges over time from audience engagement patterns — not classifiable at ingestion |
| "trending / popular right now" | Redis trending set + `popularity_score` |
| "classics / iconic movies" | `reception_score` + `popularity_score` + age (dynamic quality prior in Phase 0) |

### Removed as too broad, fuzzy, or low-frequency

| Concept | Reason for exclusion |
|---------|---------------------|
| Betrayal | Too generic — almost every thriller/crime movie has some betrayal. Would apply to too many films to provide precision. |
| Race against time | Too generic — most action movies have time pressure. Line between "ticking clock movie" and "action movie with a deadline" is subjective. |
| Redemption arc | Too many movies have some redemption element. Boundary between "redemption movie" and "character who improves" is too fuzzy. |
| Rags to riches | Overlaps with UNDERDOG. Incremental value (success stories that aren't underdog-framed) is too small. |
| Escape / breakout | Too broad — escaping prison, escaping captivity, escaping a situation are very different movie types. KIDNAPPING covers escape-from-captor; "Prison Drama" keyword covers prison context. |
| Fish out of water | Fuzzy boundary — at what point is a character "out of water"? Also lower-frequency as a primary search concept. |
| Forbidden love | Overlaps with "Tragic Romance" keyword (375). Fuzzy boundary — most romances have some obstacle. |
| Isolated location | Too broad — cabin, arctic station, island, submarine are very different movies. The concept fragments into sub-types. |
| Anthology | Low search frequency. The total corpus is small (~200-300 films). |
| Found footage (as tag) | "Found Footage Horror" keyword (309) covers the vast majority. Non-horror gap is ~50 movies. |
| Jump scares | Valuable concept but the LLM can't reliably classify from text inputs. Jump scares are visual/auditory — not detectable from plot summaries or keywords. |
| Black and white | Right concept but wrong extraction pipeline. Whether a film is B&W isn't in the text inputs available to the LLM. Needs visual metadata from a different source (IMDB technical specs). |

---

## Relationship: PLOT_TWIST vs. TWIST_VILLAIN

TWIST_VILLAIN is a subset of PLOT_TWIST where the surprise is specifically
about who the villain is. A movie can have PLOT_TWIST without TWIST_VILLAIN
(Fight Club's twist isn't about a villain's identity) and theoretically
TWIST_VILLAIN without PLOT_TWIST being the defining feature (rare in
practice).

For search routing:
- "twist movies" / "movies with a plot twist" → filter by `PLOT_TWIST` tag
- "movies with a twist ending" → filter by `PLOT_TWIST` tag + rank by
  ending-specific signals (narrative_techniques vector, information_control
  terms mentioning reveals near resolution)
- "movies with a twist villain" → filter by `TWIST_VILLAIN` tag

Both tags can coexist on the same movie (The Usual Suspects has both a
plot twist and a twist villain).

---

## Implementation Notes

### Storage

```sql
-- New enum (or static mapping table)
-- ConceptTag values assigned stable integer IDs

-- New movie_card column
ALTER TABLE public.movie_card
    ADD COLUMN IF NOT EXISTS concept_tag_ids INT[] NOT NULL DEFAULT '{}';
CREATE INDEX IF NOT EXISTS idx_movie_card_concept_tags
    ON public.movie_card USING GIN (concept_tag_ids gin__int_ops);
```

### Generation pipeline placement

Replaces the `production_keywords` generation slot. Late Wave 1 — needs
Wave 1 outputs (plot_summary from plot_events, emotional_observations from
reception) but does NOT need Wave 2 viewer_experience. Can generate as soon
as plot_events and reception are complete.

### LLM inputs

Six inputs, chosen to maximize signal density and minimize context bloat.
Each serves a distinct purpose with minimal overlap.

| Input | Tokens (typical) | What it reveals | Tags it primarily serves |
|-------|------------------|-----------------|-------------------------|
| `title_with_year` | ~5-10 | Parametric knowledge tiebreaker for well-known films | All (confirmation) |
| `plot_keywords` | ~30-150 | Direct community tags — highest signal when present | 15+ tags with strong keyword frequencies |
| `plot_summary` (Wave 1) | ~200-800 | Plot events, character arcs, ending, setting | Fallback for all; primary for TWIST_VILLAIN, SINGLE_LOCATION, CON_ARTIST, KIDNAPPING, ANIMAL_DEATH, endings |
| `emotional_observations` (Wave 1) | ~30-80 | Reviewer reports of audience emotional response | FEEL_GOOD, TEARJERKER (primary), HAPPY_ENDING, SAD_ENDING (confirmation) |
| `narrative_techniques` terms (Wave 1†) | ~30-60 | Pre-classified structural/craft labels from 6 sections | PLOT_TWIST, UNRELIABLE_NARRATOR, UNDERDOG, NONLINEAR_TIMELINE, ANTI_HERO, BREAKING_FOURTH_WALL, ENSEMBLE_CAST |
| `plot_analysis` fields (Wave 2) | ~15-35 | Pre-classified arc labels and conflict types | HAPPY/SAD_ENDING (arc direction), REVENGE, UNDERDOG (confirmation) |

**Total input: ~310-1140 tokens.** plot_summary is 50-70% of the budget
and is the only large unstructured input. Everything else is pre-classified
labels or short reviewer observations.

† narrative_techniques is Wave 2 in the current pipeline but its inputs
are Wave 1 outputs. Concept tag generation depends on its term outputs
being available.

**Inputs included — terms only, not full objects:**

From `narrative_techniques` (6 of 9 sections, terms only — no justifications):
- `narrative_archetype` (0-1 terms) → UNDERDOG
- `narrative_delivery` (0-2 terms) → NONLINEAR_TIMELINE, TIME_LOOP
- `pov_perspective` (0-2 terms) → UNRELIABLE_NARRATOR, ENSEMBLE_CAST
- `information_control` (0-2 terms) → PLOT_TWIST, TWIST_VILLAIN
- `audience_character_perception` (0-3 terms) → ANTI_HERO
- `additional_narrative_devices` (0-4 terms) → BREAKING_FOURTH_WALL

From `plot_analysis` (2 of 6 fields):
- `character_arcs[].arc_transformation_label` (0-3 labels) → ending valence, anti-hero
- `conflict_type` (0-2 phrases) → REVENGE, UNDERDOG confirmation

**Inputs explicitly excluded:**

| Input | Why excluded |
|-------|-------------|
| `overall_keywords` | Disambiguation value (post-apocalyptic vs dystopian) already served by plot_summary + plot_keywords together. Marginal edge-case improvement not worth the token cost. |
| `craft_observations` (Wave 1) | Already distilled into narrative_techniques terms. Raw prose adds tokens without new signal. |
| `thematic_observations` (Wave 1) | Already distilled into plot_analysis fields (thematic_concepts, conflict_type, character_arcs). Same redundancy issue as craft_observations. |
| `viewer_experience` (Wave 2) | emotional_observations captures audience response signal needed for FEEL_GOOD/TEARJERKER. Dropping this eliminates the Wave 2 dependency. |

**Coverage analysis by signal source:**

Tags with strong backup when plot_keywords are absent (narrative_techniques
terms provide independent signal): PLOT_TWIST, TIME_LOOP, NONLINEAR_TIMELINE,
REVENGE, ANTI_HERO, BREAKING_FOURTH_WALL, UNDERDOG.

Tags with single-source dependency on plot_keywords (no structured backup —
rely on plot_summary fallback + parametric knowledge): KIDNAPPING,
SMALL_TOWN, POST_APOCALYPTIC, HAUNTED_LOCATION, FEMALE_PROTAGONIST,
ANIMAL_DEATH.

### Output schema design

**Approach: category-level enum arrays with per-tag justification.**

The output is organized by tag category rather than a flat list. Each
category field forces the model to consider that domain independently,
solving the recall problem where a flat array lets the model stop emitting
after 1-2 tags and skip entire categories.

```python
class TagEvidence(BaseModel):
    evidence: str = Field(
        description="1 sentence: what input signal supports this tag"
    )
    tag: ConceptTag  # validated against the category's tag subset

class ConceptTagsOutput(BaseModel):
    narrative_structure: list[TagEvidence]  # PLOT_TWIST, TWIST_VILLAIN, TIME_LOOP,
                                            # NONLINEAR_TIMELINE, UNRELIABLE_NARRATOR,
                                            # OPEN_ENDING, SINGLE_LOCATION,
                                            # BREAKING_FOURTH_WALL
    plot_archetypes: list[TagEvidence]      # REVENGE, UNDERDOG, KIDNAPPING, CON_ARTIST
    settings: list[TagEvidence]             # POST_APOCALYPTIC, HAUNTED_LOCATION,
                                            # SMALL_TOWN
    characters: list[TagEvidence]           # FEMALE_PROTAGONIST, ENSEMBLE_CAST,
                                            # ANTI_HERO
    endings: list[TagEvidence]              # HAPPY_ENDING, SAD_ENDING
    experiential: list[TagEvidence]         # FEEL_GOOD, TEARJERKER
    content_flags: list[TagEvidence]        # ANIMAL_DEATH
```

**Why this structure over alternatives:**

- **Not a boolean grid (27 bool fields).** With ~90% of fields being false
  per movie, boolean grids cause false-negative bias, positional fatigue,
  and autoregressive anchoring toward "false." Well-documented failure mode
  for sparse multi-label classification with LLMs.

- **Not a flat enum array.** A single `list[TagEvidence]` has one start and
  one stop — the model can emit 1-2 tags and stop, silently skipping
  entire categories. Pure recall from 27 options is harder than recognition
  from 1-8 options per category.

- **Category fields act as attention redirects.** Each field transition is a
  mini-prompt that triggers a fresh sweep of a small tag subset. The model
  can't skip settings — it must produce output for that field (even if
  empty). Same mechanism as the 9-section narrative_techniques output.

- **evidence-before-tag ordering.** The `evidence` field precedes `tag` in
  the schema, forcing the model to articulate supporting input signals
  before committing to a classification. This is lightweight chain-of-thought
  that improves accuracy even at minimal reasoning effort. Matches the
  existing pipeline pattern (ThematicConceptWithJustification,
  CharacterArcWithReasoning, TermsWithJustificationSection).

- **Bias direction is favorable.** Array-based outputs tend toward
  over-emission (false positives); boolean grids toward under-emission
  (false negatives). For deal-breaker retrieval, false negatives are silent
  search failures. False positives are caught downstream. We want to err
  toward recall.

### Prompt design notes

- **Organize tag definitions by category** matching the output schema.
  Each category section lists its tags with descriptions and the specific
  input signals to check. This gives the model recognition-level prompting
  (it sees each tag definition) rather than pure recall.

- **Explicit sweep instruction** at the top: "Consider each category
  below. For each, check whether any tags apply based on the input signals
  listed. Empty lists are correct when no tags apply."

- **Tag definitions should be concise.** The description + "check these
  signals" pattern, not multi-paragraph rubrics. The model's job is
  mostly pattern matching (keyword present → tag applies), not nuanced
  reasoning.

- **Model and reasoning effort.** gpt-5-mini, reasoning effort minimal
  or low. The task is multi-label binary classification from pre-structured
  inputs — not a complex generation task. If evaluation shows accuracy
  issues on specific categories, bump reasoning effort before splitting
  into multiple calls.

- **Single LLM call per movie.** Inputs overlap heavily across tags;
  splitting by category would repeat the same context across calls for a
  net increase in total tokens. Only split if evaluation shows the model
  struggling with 27 tags — this would be an empirical finding, not a
  design-time constraint.

### Enum size

27 tags. Designed to be extended — new tags can be added without regenerating
existing movies (new tag defaults to absent/false).

### Overlap with vector spaces

Concept tags and vector spaces serve different purposes for the same movie:
- Tags → Phase 1 deterministic candidate retrieval ("give me ALL revenge movies")
- Vectors → Phase 2 spectrum scoring ("among these, rank by how dark they are")

A movie tagged PLOT_TWIST still has twist-related content in its
narrative_techniques vector. The tag gets the candidate set right; the vector
scores nuance within it. This is complementary, not redundant.

---

## Dependency on Other V2 Work

- Requires Wave 1 outputs: plot_events (for plot_summary), reception (for
  emotional_observations)
- Requires narrative_techniques terms and plot_analysis fields (character
  arc labels, conflict_type) — these are Wave 2 but their inputs are Wave 1
- Does NOT depend on: viewer_experience, keyword audit (complete),
  source material enum (complete), country enum, awards scraping, franchise
  generation
- Can proceed once plot_events + reception + narrative_techniques +
  plot_analysis are available
