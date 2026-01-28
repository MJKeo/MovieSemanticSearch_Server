# Vector Schema Outline

This document outlines the schema for each vector embedding used in the Movie Finder RAG system, detailing exactly which movie traits and metadata attributes are included in the text representation of each vector.

## 1. Dense Anchor Vector
**Method:** `create_dense_anchor_vector_text`
**Purpose:** Captures the full "movie card" identity for broad recall (identity, content, production, cast, reception).

### Attributes Used
*   **Title Information**
    *   `title`: The official title of the movie in English (or transliterated).
    *   `original_title` (if available): The original language title of the movie if different from the primary title.
*   **Plot & Content**
    *   `plot_analysis_metadata` (Partial usage)
        *   `generalized_plot_overview`: A high-level summary of the plot, focusing on the main narrative arc without excessive detail.
    *   `genres` (Subset): A list of standard genre classifications (e.g., Action, Drama, Comedy) associated with the movie.
    *   `overall_keywords`: Broad, high-level keywords describing the movie's general topics and style.
    *   `plot_keywords`: Specific keywords describing particular plot elements, objects, or events in the movie.
*   **Production Details**
    *   `countries_of_origin`: The list of countries where the movie was primarily produced or funded.
    *   `production_companies`: The names of the production studios or companies involved in making the movie.
    *   `filming_locations`: Real-world geographic locations where the filming took place.
    *   `languages` (Primary and additional audio): The spoken languages available in the movie, including the original audio and available dubs.
    *   `release_date` (Converted to decade bucket): The release year converted into a decade-based category (e.g., "80s", "Golden Age of Hollywood 1940s") to capture the era.
    *   `duration` (Converted to bucket): The runtime of the movie categorized into descriptive length buckets (e.g., "short, quick watch", "very long").
    *   `budget` (Converted to era-aware bucket): The estimated production budget classified as "small budget" or "big budget, blockbuster" relative to its release era.
    *   `production_metadata`
        *   `production_keywords`
            *   `terms`: Keywords specific to the production context, style, or technical achievements.
        *   `sources_of_inspiration`
            *   `production_mediums`: The format or medium of the source material (e.g., "Novel", "Stage Play", "Original Screenplay").
            *   `sources_of_inspiration`: Specific titles or works that inspired the movie (e.g., "Based on the book by...").
*   **Cast & Characters**
    *   `directors`: The names of the director(s) of the movie.
    *   `writers`: The names of the screenwriters or story creators.
    *   `producers` (First 4): The names of the primary producers.
    *   `composers`: The names of the music composers for the film's score.
    *   `actors` (Main 5): The names of the top-billed cast members.
    *   `characters` (Main 5 from list) OR `major_characters` (names only): The names of the primary characters in the story.
*   **Themes & Lessons**
    *   `plot_analysis_metadata` (Partial usage)
        *   `core_concept`
            *   `core_concept_label`: A concise label representing the single dominant story concept or premise (e.g., "Hero's Journey", "Whodunit").
        *   `themes_primary`
            *   `theme_label`: High-signal labels summarizing the central themes of the movie (e.g., "Redemption", "Coming of Age").
        *   `lessons_learned`
            *   `lesson_label`: Simple, generic labels for the moral or practical lessons conveyed by the story.
*   **Audience Reception & Vibe**
    *   `viewer_experience_metadata`
        *   `emotional_palette`
            *   `terms`: Search-query-like phrases describing the primary emotions evoked by the movie (e.g., "Heartwarming", "Terrifying").
    *   `watch_context_metadata`
        *   `key_movie_feature_draws`
            *   `terms`: Main selling points or features that attract viewers to this specific movie.
    *   `maturity_rating` & `maturity_reasoning` / `parental_guide_items` (Combined into guidance text): The official age rating (e.g., PG-13, R) and the specific reasons for it (e.g., "Rated R for violence").
    *   `reception_tier` (Derived from `imdb_rating` and `metacritic_rating`): A descriptive label categorizing the critical consensus (e.g., "Universally acclaimed", "Mixed reviews").
    *   `reception_metadata`
        *   `praise_attributes`: Specific aspects of the movie that received critical praise (e.g., "Cinematography", "Performances").
        *   `complaint_attributes`: Specific aspects of the movie that received critical criticism (e.g., "Pacing", "Dialogue").

---

## 2. Plot Events Vector
**Method:** `create_plot_events_vector_text`
**Purpose:** Focuses on chronological plot details, settings, and major characters for specific plot queries.

### Attributes Used
*   **Plot Events Metadata** (`plot_events_metadata`)
    *   `plot_summary` (Detailed, spoiler-containing): A comprehensive, chronological summary of the entire movie plot, including spoilers and ending details.
    *   `setting` (Specific where/when): A short phrase describing the specific time period and location where the story takes place.
    *   `major_characters` (List of objects)
        *   `name`: The name of a major character.
        *   `description`: A brief description of who the character is (e.g., "A retired detective").
        *   `primary_motivations`: A concise statement of what the character wants to achieve and why.
        *   *(Note: `role` is defined in schema but NOT included in vector text)*

---

## 3. Plot Analysis Vector
**Method:** `create_plot_analysis_vector_text`
**Purpose:** Focuses on deep content, themes, and narrative structure for "what happens" and thematic queries.

### Attributes Used
*   **Plot Analysis Metadata** (`plot_analysis_metadata`)
    *   `generalized_plot_overview`: A high-level summary of the plot, focusing on the main narrative arc without excessive detail.
    *   `core_concept`
        *   `core_concept_label`: A concise label representing the single dominant story concept or premise.
        *   `explanation_and_justification`: A one-sentence explanation of why this concept fits the movie.
    *   `genre_signatures` (List of terms): Specific elements, tropes, or stylistic choices typical of the movie's genre.
    *   `conflict_scale`: The scope or magnitude of the conflict in the story (e.g., "Interpersonal", "Global", "Cosmic").
    *   `character_arcs` (List of objects)
        *   `arc_transformation_label`: A generic phrase classifying the type of change a character undergoes (e.g., "Corruption arc").
        *   *(Note: `character_name` and `arc_transformation_description` are defined in schema but NOT included in vector text)*
    *   `themes_primary` (List of objects)
        *   `theme_label`: High-signal labels summarizing the central themes of the movie.
        *   *(Note: `explanation_and_justification` is defined in schema but NOT included in vector text)*
    *   `lessons_learned` (List of objects)
        *   `lesson_label`: Simple, generic labels for the moral or practical lessons conveyed by the story.
        *   *(Note: `explanation_and_justification` is defined in schema but NOT included in vector text)*
*   **Keywords**
    *   `overall_keywords`: Broad, high-level keywords describing the movie's general topics and style.

---

## 4. Narrative Techniques Vector
**Method:** `create_narrative_techniques_vector_text`
**Purpose:** Focuses on storytelling style, perspective, and structural devices.

### Attributes Used
*   **Narrative Techniques Metadata** (`narrative_techniques_metadata`)
    *   `pov_perspective`
        *   `terms`: Phrases describing the narrative point of view (e.g., "First-person narration", "Ensemble cast").
    *   `narrative_delivery`
        *   `terms`: Descriptions of how the story is told structurally (e.g., "Non-linear timeline", "Flashbacks").
    *   `narrative_archetype`
        *   `terms`: Classic story patterns or structures used (e.g., "Hero's Journey", "Tragedy").
    *   `information_control`
        *   `terms`: How information is revealed to the audience (e.g., "Unreliable narrator", "Mystery").
    *   `characterization_methods`
        *   `terms`: Techniques used to establish and develop characters (e.g., "Show don't tell", "Monologue").
    *   `character_arcs` (Generic Terms Section - distinct from Plot Analysis arcs)
        *   `terms`: General terms describing the nature of character development in the film.
    *   `audience_character_perception`
        *   `terms`: How the audience is intended to view or relate to the characters (e.g., "Sympathetic villain").
    *   `conflict_stakes_design`
        *   `terms`: The nature of the risks and conflict driving the story.
    *   `thematic_delivery`
        *   `terms`: How themes are conveyed to the audience (e.g., "Subtext", "Allegory").
    *   `meta_techniques`
        *   `terms`: Self-referential or fourth-wall-breaking elements (e.g., "Mockumentary style").
    *   `additional_plot_devices`
        *   `terms`: Specific narrative tools or devices used (e.g., "MacGuffin", "Chekhov's Gun").
    *   *(Note: `justification` is defined in schema for all sections but NOT included in vector text)*

---

## 5. Viewer Experience Vector
**Method:** `create_viewer_experience_vector_text`
**Purpose:** Focuses on the "feel" of the movie (mood, energy, emotional impact).

### Attributes Used
*   **Viewer Experience Metadata** (`viewer_experience_metadata`)
    *   `emotional_palette`
        *   `terms`: Phrases describing the primary emotions evoked by the movie.
        *   `negations`: "Avoidance" phrases describing emotions the movie explicitly does NOT evoke.
    *   `tension_adrenaline`
        *   `terms`: Descriptions of the level of suspense, excitement, and pacing.
        *   `negations`: Descriptions of tension levels the movie does NOT have.
    *   `tone_self_seriousness`
        *   `terms`: Descriptions of how seriously the movie takes itself (e.g., "Campy", "Gritty realism").
        *   `negations`: Tone descriptions the movie avoids.
    *   `cognitive_complexity`
        *   `terms`: Descriptions of how difficult the movie is to follow or understand.
        *   `negations`: Complexity levels the movie does not fit.
    *   `disturbance_profile` (If applicable)
        *   `terms`: Descriptions of unsettling, scary, or disturbing content.
        *   `negations`: Types of disturbance the movie avoids.
    *   `sensory_load` (If applicable)
        *   `terms`: Descriptions of the visual and auditory intensity of the film.
        *   `negations`: Sensory styles the movie avoids.
    *   `emotional_volatility` (If applicable)
        *   `terms`: Descriptions of how quickly or drastically the emotional tone shifts.
        *   `negations`: Volatility patterns the movie avoids.
    *   `ending_aftertaste`
        *   `terms`: Descriptions of the feeling or mood left with the viewer after the movie ends.
        *   `negations`: Ending feelings the movie avoids.
    *   *(Note: `justification` is defined in schema but NOT included in vector text)*

---

## 6. Watch Context Vector
**Method:** `create_watch_context_vector_text`
**Purpose:** Focuses on use-cases, suitability, and viewing scenarios.

### Attributes Used
*   **Watch Context Metadata** (`watch_context_metadata`)
    *   `self_experience_motivations`
        *   `terms`: Internal reasons or moods for watching this movie (e.g., "Need a good cry").
    *   `external_motivations`
        *   `terms`: Social or external reasons for watching (e.g., "Date night", "Film study").
    *   `key_movie_feature_draws`
        *   `terms`: Main selling points or features that attract viewers to this specific movie.
    *   `watch_scenarios`
        *   `terms`: Ideal physical or social situations for viewing (e.g., "Rainy Sunday", "Party background").
    *   *(Note: `justification` is defined in schema but NOT included in vector text)*

---

## 7. Production Vector
**Method:** `create_production_vector_text`
**Purpose:** Focuses on the "making of" details (who, where, when, inspiration).

### Attributes Used
*   **Production Basics**
    *   `countries_of_origin`: The list of countries where the movie was primarily produced or funded.
    *   `production_companies`: The names of the production studios or companies involved in making the movie.
    *   `filming_locations`: Real-world geographic locations where the filming took place.
    *   `languages` (Primary and additional audio): The spoken languages available in the movie.
    *   `release_date` (Converted to decade bucket): The release year converted into a decade-based category.
    *   `budget` (Converted to era-aware bucket): The estimated production budget classified relative to its release era.
*   **Production Metadata** (`production_metadata`)
    *   `production_keywords`
        *   `terms`: Keywords specific to the production context, style, or technical achievements.
    *   `sources_of_inspiration`
        *   `production_mediums`: The format or medium of the source material.
        *   `sources_of_inspiration`: Specific titles or works that inspired the movie.
*   **Cast & Characters**
    *   `directors`: The names of the director(s).
    *   `writers`: The names of the screenwriters.
    *   `producers` (First 4): The names of the primary producers.
    *   `composers`: The names of the music composers.
    *   `actors` (Main 5): The names of the top-billed cast members.
    *   `characters` (Main 5 from list) OR `major_characters` (names only): The names of the primary characters.

---

## 8. Reception Vector
**Method:** `create_reception_vector_text`
**Purpose:** Focuses on critical acclaim and specific praises/complaints.

### Attributes Used
*   **Reception Tier** (`reception_tier`)
    *   Derived from `imdb_rating` and `metacritic_rating`: A descriptive label categorizing the critical consensus.
*   **Reception Metadata** (`reception_metadata`)
    *   `praise_attributes`: Specific aspects of the movie that received critical praise.
    *   `complaint_attributes`: Specific aspects of the movie that received critical criticism.
    *   *(Note: `justification` is defined in schema but NOT included in vector text)*
