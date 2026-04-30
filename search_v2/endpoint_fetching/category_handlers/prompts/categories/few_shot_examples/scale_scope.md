# Examples

These examples calibrate three clean-fire patterns (epic-scale, intimate-scale, vibe-shaped "feels like X") plus two nearest-neighbor no-fires: plot-shaped "feels like" (Cat 20) and a purely during-viewing cognitive-demand ask (Cat 22).

**Example: clean fire on an epic-scale ask**

```xml
<raw_query>epic war movie</raw_query>
<overall_query_intention_exploration>The user wants a war film with epic scale — sweeping canvas, large scope, multi-front or multi-year ambition. The war-genre atom lives on a sibling fragment; the target here is the scale framing.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>A war film with epic, large-canvas scope — sweeping in time and scale.</captured_meaning>
  <category_name>Scale / scope / holistic vibe</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies whose scope and canvas is epic — large-scale, sweeping ambition.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>epic</query_text>
  <description>An epic, large-canvas scope.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>war movie</query_text>
    <description>War-genre film.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Match films whose identity capsule reads as epic in scope — sweeping canvas, large scale, ambitious in time and setting.",
      "relation_to_endpoint": "Semantic plot_analysis carries the identity-level framing via elevator_pitch — a one-sentence capsule that routinely encodes scope on the ingest side — and genre_signatures carries compact scope-implying labels like 'epic'. Embedding a pitch that reads like the ingest text of an epic-scale film surfaces matches whose scope framing lines up. The war-genre dimension is a separate atom on the sibling fragment and is not emitted here.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "trait",
    "parameters": {
      "qualifier_inventory": "One scope atom: an epic, large-canvas scale — sweeping in time, setting, and ambition. No tonal or thematic content specified beyond the scope framing.",
      "space_queries": [
        {
          "carries_qualifiers": "plot_analysis.elevator_pitch carries identity-level scope framing on the ingest side — a one-sentence capsule that names canvas and ambition. genre_signatures carries 'epic' as a compact label in the same register as ingest-side genre terms.",
          "space": "plot_analysis",
          "weight": "central",
          "content": {
            "elevator_pitch": "a sweeping, large-canvas epic spanning years and multiple fronts, following characters through a story of grand scale and ambition.",
            "plot_overview": null,
            "genre_signatures": ["epic", "sweeping saga", "large-scale"],
            "conflict_type": [],
            "thematic_concepts": [],
            "character_arcs": []
          }
        }
      ],
      "primary_vector": "plot_analysis"
    },
    "polarity": "positive"
  }
}
```

**Example: clean fire on an intimate-scale ask**

```xml
<raw_query>intimate character study</raw_query>
<overall_query_intention_exploration>The user wants a film built at intimate scale — small canvas, close focus on one or a few characters. Both "intimate" (scope) and "character study" (identity framing) point at a small-scale, inward-facing identity.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>A film with intimate, small-scale scope framed as a character study.</captured_meaning>
  <category_name>Scale / scope / holistic vibe</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies whose scope is intimate and whose identity is a character study — small canvas, close focus.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>intimate character study</query_text>
  <description>A small-scale, character-focused film.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Match films whose identity capsule reads as small-scale and character-focused — intimate canvas, close on one or a few characters.",
      "relation_to_endpoint": "Semantic plot_analysis.elevator_pitch carries the one-sentence identity framing that names scope and focus; 'intimate character study' is the exact register an ingest-side pitch would use for a small-canvas, character-driven film. genre_signatures carries 'character study' and 'chamber piece' as compact identity labels.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "trait",
    "parameters": {
      "qualifier_inventory": "Two same-axis atoms collapsed into one scope framing: intimate scale (small canvas) plus character-study identity (close focus on one or a few characters).",
      "space_queries": [
        {
          "carries_qualifiers": "plot_analysis.elevator_pitch is where identity-level scope lives on the ingest side; 'intimate character study' lands as a capsule. genre_signatures carries the compact identity labels in the same register as ingest-side terms.",
          "space": "plot_analysis",
          "weight": "central",
          "content": {
            "elevator_pitch": "a small-scale, intimate character study that stays close to one or a few people, observing their inner lives and relationships within a contained setting.",
            "plot_overview": null,
            "genre_signatures": ["character study", "intimate drama", "chamber piece"],
            "conflict_type": [],
            "thematic_concepts": [],
            "character_arcs": []
          }
        }
      ],
      "primary_vector": "plot_analysis"
    },
    "polarity": "positive"
  }
}
```

**Example: clean fire on a vibe-shaped "feels like X" ask**

```xml
<raw_query>movies that feel like Lost in Translation</raw_query>
<overall_query_intention_exploration>The user wants films whose overall vibe and identity resembles Lost in Translation — a quiet, melancholic, low-energy character piece built around strangers connecting briefly in an unfamiliar place. The reference is vibe-shaped and identity-level, not plot-template-shaped.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films whose holistic vibe and identity resembles Lost in Translation — quiet, melancholic, small-scale connection story.</captured_meaning>
  <category_name>Scale / scope / holistic vibe</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies whose overall vibe matches Lost in Translation — a quiet, low-energy, intimate connection film with a melancholic register.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>feel like Lost in Translation</query_text>
  <description>Films with the same overall vibe and identity as Lost in Translation.</description>
  <modifiers>
    <modifier>
      <original_text>feel like</original_text>
      <effect>binds the following title as a vibe/identity reference rather than a franchise or plot-pattern reference</effect>
      <type>ROLE_MARKER</type>
    </modifier>
  </modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Match films whose identity capsule lines up with Lost in Translation's overall register — quiet, melancholic, small-scale story of a brief connection between strangers in an unfamiliar setting.",
      "relation_to_endpoint": "Semantic plot_analysis.elevator_pitch carries the identity-level framing; the way to target a vibe-shaped reference is to write the capsule the referenced film's ingest-side pitch would carry — translate the reference into the shape of a matching pitch rather than citing the film by name. viewer_experience co-populates for the quiet, melancholic tonal register that is load-bearing on the referenced identity.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "trait",
    "parameters": {
      "qualifier_inventory": "One holistic identity reference decomposed into: small-scale intimate scope; melancholic, quiet tonal register; brief connection between strangers in an unfamiliar place as the identity framing.",
      "space_queries": [
        {
          "carries_qualifiers": "plot_analysis.elevator_pitch carries the identity capsule on the ingest side. Writing the pitch a matching film would carry — intimate connection story, melancholic register, unfamiliar setting — targets the same prose neighborhood. genre_signatures carries the compact identity labels that round out the scope.",
          "space": "plot_analysis",
          "weight": "central",
          "content": {
            "elevator_pitch": "a quiet, melancholic, small-scale story of a brief and unlikely connection between two strangers adrift in an unfamiliar place, carried by mood and observation more than plot.",
            "plot_overview": null,
            "genre_signatures": ["intimate drama", "mood piece", "character study"],
            "conflict_type": [],
            "thematic_concepts": ["loneliness", "fleeting connection", "displacement"],
            "character_arcs": []
          }
        },
        {
          "carries_qualifiers": "viewer_experience carries the during-viewing tonal register that is load-bearing on this identity — quiet, melancholic, low-key emotional palette plus a gentle, non-serious self-register. Supporting rather than central because the primary ask is the identity capsule, and tone rounds it out.",
          "space": "viewer_experience",
          "weight": "supporting",
          "content": {
            "emotional_palette": {
              "terms": ["melancholic", "wistful", "bittersweet"],
              "negations": []
            },
            "tension_adrenaline": {
              "terms": ["low-stakes", "quiet"],
              "negations": []
            },
            "tone_self_seriousness": {
              "terms": [],
              "negations": []
            },
            "cognitive_complexity": {
              "terms": [],
              "negations": []
            },
            "disturbance_profile": {
              "terms": [],
              "negations": []
            },
            "sensory_load": {
              "terms": ["understated", "restrained"],
              "negations": []
            },
            "emotional_volatility": {
              "terms": [],
              "negations": []
            },
            "ending_aftertaste": {
              "terms": [],
              "negations": []
            }
          }
        }
      ],
      "primary_vector": "plot_analysis"
    },
    "polarity": "positive"
  }
}
```

**Example: no-fire on a plot-shaped "feels like" ask (Cat 20 boundary)**

```xml
<raw_query>movies that feel like a heist</raw_query>
<overall_query_intention_exploration>The user wants films that play like a heist — the reference is a plot template (a team plans and executes a score) rather than a holistic vibe. Plot-shaped similarity, not identity-shaped.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films whose holistic vibe reads as a heist.</captured_meaning>
  <category_name>Scale / scope / holistic vibe</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies whose overall vibe feels like a heist film.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>feel like a heist</query_text>
  <description>Films playing like a heist story.</description>
  <modifiers>
    <modifier>
      <original_text>feel like</original_text>
      <effect>binds the following phrase as a similarity reference</effect>
      <type>ROLE_MARKER</type>
    </modifier>
  </modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The user is naming a plot-template similarity — the shape of a heist story — not a scope or identity-level vibe.",
      "relation_to_endpoint": "Semantic plot_analysis carries identity-level scope and holistic vibe framing via elevator_pitch and genre_signatures. 'Heist' is a specific plot template — a team assembling, planning, and executing a score — which lives on plot_events (synopsis prose) or on the keyword channel as a recognized story-pattern label. Routing a plot-template ask through a scope/vibe capsule would embed the wrong axis: the match would pick up films with similar identity mood rather than films that actually run a heist. The correct routes are Cat 20 (plot events) for the event-template shape or Cat 15 (sub-genre) for the named pattern.",
      "coverage_gaps": "Upstream dispatch read 'feel like a heist' as a holistic vibe, but 'heist' is plot-template-shaped, not identity-shaped. The axis is events-and-structure, not scope. This needs to route through Cat 20 or Cat 15."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```

**Example: no-fire on a cognitive-demand ask (Cat 22 boundary)**

```xml
<raw_query>cerebral movies</raw_query>
<overall_query_intention_exploration>The user wants cognitively demanding films — "cerebral" names the during-viewing mental workload, not the movie's scope or identity canvas. No scale framing is present.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films whose holistic vibe reads as cerebral.</captured_meaning>
  <category_name>Scale / scope / holistic vibe</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies whose overall vibe feels cerebral.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>cerebral movies</query_text>
  <description>Cognitively demanding films.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The user is naming during-viewing cognitive demand — how much mental work the film asks of the viewer — not scope, canvas, or identity-level vibe.",
      "relation_to_endpoint": "Semantic plot_analysis is the home for identity-level scope and holistic-vibe framing (scale, canvas, one-sentence identity capsules). 'Cerebral' does not name any of those axes — it names the cognitive load experienced during viewing, which lives on viewer_experience.cognitive_complexity on the ingest side. Forcing 'cerebral' into a scope capsule or genre_signatures list would route a cognitive-demand ask through the wrong vocabulary and dilute the query. Cat 22 (Viewer experience) is the correct category.",
      "coverage_gaps": "Upstream dispatch read 'cerebral' as a holistic vibe, but it is a during-viewing cognitive-demand qualifier with no scale or identity dimension attached. This needs to route through Cat 22."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```
