# Examples

These examples calibrate the three clean-fire patterns (dark-gritty tonal, cerebral cognitive-demand, whimsical-cozy tonal) plus the two nearest-neighbor no-fires: post-viewing resonance (Cat 26) and scale/scope (Cat 27).

**Example: clean fire on a dark-gritty tonal ask**

```xml
<raw_query>dark gritty crime movies</raw_query>
<overall_query_intention_exploration>The user wants crime films whose during-viewing feel is dark and gritty — a tonal aesthetic claim about how the movie feels moment to moment, distinct from the crime-genre aspect which lives in another fragment.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>A dark and gritty during-viewing tonal feel — heavy, unvarnished, unglamorous.</captured_meaning>
  <category_name>Viewer experience / feel / tone</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies whose during-viewing feel is dark and gritty.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>dark gritty</query_text>
  <description>A dark, gritty tonal aesthetic.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>crime movies</query_text>
    <description>Crime-genre films.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Match films whose during-viewing tone reads as dark and gritty — heavy emotional palette and a serious, unglamorous register.",
      "relation_to_endpoint": "Semantic viewer_experience carries emotional_palette for tonal descriptors and tone_self_seriousness for the serious/unglamorous register. 'Dark' and 'gritty' land cleanly in those two sub-fields as near-synonyms of the ingest-side vocabulary; embedding them there surfaces films whose tonal metadata matches the register.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "trait",
    "parameters": {
      "qualifier_inventory": "Two same-dimension tonal qualifiers treated as one unit: a dark, gritty during-viewing feel — heavy palette plus an unglamorous, serious register.",
      "space_queries": [
        {
          "carries_qualifiers": "viewer_experience emotional_palette holds tonal-mood descriptors on the ingest side; 'dark', 'bleak', 'grim' are the native vocabulary for this tonal register. tone_self_seriousness carries the unglamorous, serious-register dimension that 'gritty' lands on.",
          "space": "viewer_experience",
          "weight": "central",
          "content": {
            "emotional_palette": {
              "terms": ["dark", "bleak", "grim"],
              "negations": []
            },
            "tension_adrenaline": {
              "terms": [],
              "negations": []
            },
            "tone_self_seriousness": {
              "terms": ["gritty", "unglamorous", "serious"],
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
              "terms": [],
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
      "primary_vector": "viewer_experience"
    },
    "polarity": "positive"
  }
}
```

**Example: clean fire on a cerebral cognitive-demand ask**

```xml
<raw_query>cerebral sci-fi</raw_query>
<overall_query_intention_exploration>The user wants sci-fi films that demand active thought during viewing — a cognitive-demand claim about the during-viewing experience. The sci-fi genre atom lives on a sibling fragment.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>A cerebral during-viewing experience — the movie asks the viewer to think, track, and interpret.</captured_meaning>
  <category_name>Viewer experience / feel / tone</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies that feel cerebral to watch — cognitively demanding during viewing.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>cerebral</query_text>
  <description>A cerebral, thought-provoking during-viewing feel.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>sci-fi</query_text>
    <description>Science fiction genre.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Match films whose during-viewing experience is cognitively demanding — asks the viewer to track, interpret, and reason rather than coast.",
      "relation_to_endpoint": "Semantic viewer_experience carries cognitive_complexity as a dedicated sub-field for the demand a film places on the viewer mid-watch. 'Cerebral' is a direct-match term in that sub-field's ingest-side vocabulary.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "trait",
    "parameters": {
      "qualifier_inventory": "One atom: a cerebral during-viewing experience — the film demands active thought and interpretation to follow.",
      "space_queries": [
        {
          "carries_qualifiers": "viewer_experience cognitive_complexity is the sub-field that carries mid-watch cognitive demand on the ingest side; 'cerebral', 'thought-provoking', and 'intellectually demanding' are the native terms.",
          "space": "viewer_experience",
          "weight": "central",
          "content": {
            "emotional_palette": {
              "terms": [],
              "negations": []
            },
            "tension_adrenaline": {
              "terms": [],
              "negations": []
            },
            "tone_self_seriousness": {
              "terms": [],
              "negations": []
            },
            "cognitive_complexity": {
              "terms": ["cerebral", "thought-provoking", "intellectually demanding"],
              "negations": []
            },
            "disturbance_profile": {
              "terms": [],
              "negations": []
            },
            "sensory_load": {
              "terms": [],
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
      "primary_vector": "viewer_experience"
    },
    "polarity": "positive"
  }
}
```

**Example: clean fire on a whimsical cozy tonal ask**

```xml
<raw_query>whimsical cozy movies for a rainy day</raw_query>
<overall_query_intention_exploration>The user wants films whose during-viewing feel is whimsical and cozy. The "rainy day" part is a watch occasion handled by another category; the target atom here is purely the tonal aesthetic.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>A whimsical, cozy during-viewing feel — light, warm, gently playful.</captured_meaning>
  <category_name>Viewer experience / feel / tone</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies whose during-viewing feel is whimsical and cozy.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>whimsical cozy</query_text>
  <description>A whimsical, cozy tonal aesthetic.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>for a rainy day</query_text>
    <description>Occasion: rainy-day watching.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Match films whose during-viewing tone reads as whimsical and cozy — warm, light emotional palette plus a playful, low-self-seriousness register.",
      "relation_to_endpoint": "Semantic viewer_experience emotional_palette carries warm/light tonal descriptors; tone_self_seriousness carries the playful, unserious register that 'whimsical' lands on. Both sub-fields hold ingest-side terms close to 'whimsical', 'cozy', 'warm', 'playful'.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "trait",
    "parameters": {
      "qualifier_inventory": "Two same-dimension tonal atoms: a whimsical, cozy during-viewing feel — warm/light palette plus a playful, unserious register.",
      "space_queries": [
        {
          "carries_qualifiers": "viewer_experience emotional_palette holds warm, light tonal descriptors on the ingest side; 'cozy', 'warm', 'gentle' are the native vocabulary. tone_self_seriousness carries the playful, whimsical register that sits on the low-self-seriousness end of that sub-field.",
          "space": "viewer_experience",
          "weight": "central",
          "content": {
            "emotional_palette": {
              "terms": ["cozy", "warm", "gentle"],
              "negations": []
            },
            "tension_adrenaline": {
              "terms": [],
              "negations": []
            },
            "tone_self_seriousness": {
              "terms": ["whimsical", "playful", "light-hearted"],
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
              "terms": [],
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
      "primary_vector": "viewer_experience"
    },
    "polarity": "positive"
  }
}
```

**Example: no-fire on a post-viewing aftertaste ask (Cat 26 boundary)**

```xml
<raw_query>something haunting that stays with you</raw_query>
<overall_query_intention_exploration>The user wants films whose impact lingers after the credits — an aftertaste claim, not a during-viewing feel. "Haunting" and "stays with you" both name what the movie leaves behind, not how it plays moment to moment.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>A during-viewing feel of being haunted — mid-watch mood of lingering disquiet.</captured_meaning>
  <category_name>Viewer experience / feel / tone</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies that feel haunting to watch.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>something haunting that stays with you</query_text>
  <description>Films with a haunting, lingering quality after viewing.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The user is naming what the movie leaves behind after viewing, not how it feels during viewing.",
      "relation_to_endpoint": "Semantic viewer_experience covers the during-viewing experience across tone, tension, cognitive demand, and sensory load. 'Haunting' and 'stays with you' are both aftertaste claims — the residue a film leaves once it's over. That axis lives on viewer_experience.ending_aftertaste on the ingest side, but aftertaste is Cat 26's primary target, not this category's. Forcing these terms into emotional_palette would mis-embed the query as a during-viewing mood claim when the user is naming post-viewing resonance.",
      "coverage_gaps": "Upstream dispatch read 'haunting' as a during-viewing tonal term when the accompanying 'stays with you' makes clear the axis is post-viewing resonance. The correct route is Cat 26 (Post-viewing resonance), not here."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```

**Example: no-fire on a scale/scope ask (Cat 27 boundary)**

```xml
<raw_query>epic sweeping movies</raw_query>
<overall_query_intention_exploration>The user wants films with an epic, sweeping scope. "Epic" and "sweeping" both name scale — how large and ambitious the movie's canvas is — rather than a tonal feel or cognitive demand.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>An epic, sweeping during-viewing feel — grand, large-scale tonal register.</captured_meaning>
  <category_name>Viewer experience / feel / tone</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies that feel epic and sweeping to watch.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>epic sweeping</query_text>
  <description>Films with an epic, sweeping scale.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The user is naming scale and scope — how large and sweeping the movie's canvas is — not a tonal aesthetic or cognitive-demand property.",
      "relation_to_endpoint": "Semantic viewer_experience carries tone, tension, cognitive demand, sensory load, and emotional volatility — dimensions of during-viewing feel. 'Epic' and 'sweeping' are scope descriptors: they name the size of the story's canvas rather than any of those experiential axes. Embedding them into emotional_palette or tone_self_seriousness would route a scope query through the wrong vocabulary. Cat 27 (Scale / scope / holistic vibe) is the correct category for scale framing.",
      "coverage_gaps": "Upstream dispatch read 'epic' as a tonal term, but paired with 'sweeping' the axis is unambiguously scope. The viewer_experience sub-fields have no lever for scale; this needs to route through Cat 27."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```
