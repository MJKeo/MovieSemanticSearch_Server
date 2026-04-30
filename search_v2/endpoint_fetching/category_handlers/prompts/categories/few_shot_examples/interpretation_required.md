# Examples

These examples calibrate the interpretation-driven fallback: decoding fuzzy intent into the single space most likely to carry it on the ingest side, spanning watch_context, plot_analysis, reception, and the empty-intent no-fire.

**Example: oddball comfort framing → watch_context + viewer_experience**

```xml
<raw_query>movies that feel like a warm hug after a hard day</raw_query>
<overall_query_intention_exploration>The user wants films whose appeal is a specific comfort effect — the feeling of being enveloped in warmth after stress. It is not a canonical genre, not a pure during-viewing tonal label, and not a named occasion like "date night". It blends self-experience motivation (recovering from a hard day) with a warm, gentle tonal register.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films whose appeal is a warm, enveloping comfort effect that restores the viewer after a stressful day.</captured_meaning>
  <category_name>Interpretation-required</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies that function as a warm, comforting balm after a stressful day.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies that feel like a warm hug after a hard day</query_text>
  <description>Films that comfort and warm the viewer after a difficult day.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Interpretation-driven: match films whose watch_context talks about comfort / decompression / recovery as the motivation for watching, and whose viewer_experience tone reads as warm, gentle, low-tension — the during-viewing texture of a 'warm hug'.",
      "relation_to_endpoint": "Semantic watch_context.self_experience_motivations on the ingest side carries motivation phrases for why a viewer would put a film on — 'comfort watch', 'decompress', 'unwind' are the native register. viewer_experience.emotional_palette and tone_self_seriousness carry the warm, gentle tonal descriptors a matching film's ingest text would use. Neither space alone fully captures the ask, but together they reconstruct the intent.",
      "coverage_gaps": "Interpretation-driven query — no canonical tag or single-space label names this concept, so the match relies on nearby ingest-side vocabulary in each space rather than a direct term match."
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "trait",
    "parameters": {
      "qualifier_inventory": "Two atoms: (1) a motivation to decompress and be comforted after a stressful day — a watch_context ask; (2) a warm, gentle, enveloping during-viewing tonal register — a viewer_experience ask.",
      "space_queries": [
        {
          "carries_qualifiers": "watch_context self_experience_motivations carries motivation phrases for why a viewer picks a film; 'comfort watch', 'decompress after a hard day', 'unwind' are the native terms for this intent. watch_scenarios adds the situational hook of a low-energy evening recovery.",
          "space": "watch_context",
          "weight": "central",
          "content": {
            "self_experience_motivations": {
              "terms": ["comfort watch", "decompress after a hard day", "unwind", "feel soothed"]
            },
            "external_motivations": {
              "terms": []
            },
            "key_movie_feature_draws": {
              "terms": []
            },
            "watch_scenarios": {
              "terms": ["low-energy evening", "rough day recovery"]
            }
          }
        },
        {
          "carries_qualifiers": "viewer_experience emotional_palette and tone_self_seriousness carry the warm, gentle, low-stakes tonal texture a 'warm hug' maps onto — this is the during-viewing side of the ask, complementary to the motivation captured in watch_context.",
          "space": "viewer_experience",
          "weight": "supporting",
          "content": {
            "emotional_palette": {
              "terms": ["warm", "gentle", "tender", "reassuring"],
              "negations": []
            },
            "tension_adrenaline": {
              "terms": [],
              "negations": ["intense", "high-tension"]
            },
            "tone_self_seriousness": {
              "terms": ["low-key", "unhurried"],
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
      "primary_vector": "watch_context"
    },
    "polarity": "positive"
  }
}
```

**Example: abstract thematic ask → plot_analysis + reception**

```xml
<raw_query>movies that make you think about mortality</raw_query>
<overall_query_intention_exploration>The user wants films whose thematic substance is mortality — the confrontation with death, finitude, or meaning in the face of the end. It is an abstract thematic ask that sits above any specific plot template or genre. A matching film's elevator pitch and thematic concepts would name these ideas directly, and reception prose for such films typically dwells on the philosophical weight of the material.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films whose thematic substance centers on mortality and the confrontation with death.</captured_meaning>
  <category_name>Interpretation-required</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies that engage thematically with mortality.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies that make you think about mortality</query_text>
  <description>Films that prompt reflection on mortality and death.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Interpretation-driven: match films whose plot_analysis names mortality / death / finitude as a core thematic concept, and whose reception prose discusses the philosophical weight of the material.",
      "relation_to_endpoint": "plot_analysis.thematic_concepts is the ingest-side home for abstract theme terms like 'mortality' and 'meaning in the face of death'. reception.reception_summary on films that actually engage with mortality tends to dwell on that reflective quality — 'meditation on death', 'confronts finitude' — giving a second signal path. The primary lever is plot_analysis; reception rounds out the match.",
      "coverage_gaps": "Interpretation-driven query — 'makes you think about X' translates onto thematic vocabulary but there is no direct 'asks the viewer to reflect' sub-field; the match approximates the intent through thematic density and reception tone."
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "trait",
    "parameters": {
      "qualifier_inventory": "One atom, thematic: mortality and the confrontation with death as the film's thematic substance. A secondary atom: the film should be the kind that provokes reflection, which surfaces in reception prose.",
      "space_queries": [
        {
          "carries_qualifiers": "plot_analysis thematic_concepts is the ingest-side sub-field for abstract theme terms; 'mortality', 'death', 'finitude', 'meaning of life' are the native register. elevator_pitch carries the one-sentence identity framing a mortality-themed film would use.",
          "space": "plot_analysis",
          "weight": "central",
          "content": {
            "elevator_pitch": "a reflective meditation on mortality and the meaning a life carries when confronted with its end",
            "plot_overview": null,
            "genre_signatures": [],
            "conflict_type": ["man vs self", "man vs mortality"],
            "thematic_concepts": ["mortality", "death", "finitude", "meaning of life", "legacy"],
            "character_arcs": []
          }
        },
        {
          "carries_qualifiers": "reception.reception_summary on films that engage seriously with mortality tends to describe them as meditations on death or philosophical reflections — a second signal path for 'the kind of film that prompts this reflection'.",
          "space": "reception",
          "weight": "supporting",
          "content": {
            "reception_summary": "praised as a philosophical meditation on mortality that prompts the viewer to reflect on death and the shape of a life",
            "praised_qualities": ["thematic depth", "philosophical weight"],
            "criticized_qualities": []
          }
        }
      ],
      "primary_vector": "plot_analysis"
    },
    "polarity": "positive"
  }
}
```

**Example: personal-audience ask → watch_context with uncertainty flagged**

```xml
<raw_query>movies my grandmother would love</raw_query>
<overall_query_intention_exploration>The user wants films fitting a personal-audience profile — a grandmother, implying gentle content, familiar emotional terrain, and a viewing context of watching together with older family. The ingest side does not carry a "grandmother" field, but watch_context motivations and scenarios commonly describe films as 'good to watch with parents / grandparents' and 'family-friendly across generations'.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films suited to an older-family-member audience — gentle, familiar, watchable across generations.</captured_meaning>
  <category_name>Interpretation-required</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies suitable for watching with a grandmother.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies my grandmother would love</query_text>
  <description>Films a grandmother would enjoy watching.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Interpretation-driven: match films whose watch_context talks about cross-generational viewing and gentle content — the ingest-side proxy for 'a grandmother would love it'.",
      "relation_to_endpoint": "watch_context.watch_scenarios and external_motivations carry phrases like 'watch with grandparents', 'multi-generational family viewing', 'low-conflict content for older relatives' on the ingest side. These are the nearest honest vocabulary for the personal profile the user described. No space carries a direct 'grandmother' signal, so the match relies on this proxy.",
      "coverage_gaps": "Interpretation-driven query — 'grandmother' is a personal stand-in for a cross-generational, gentle-content audience profile. The endpoint cannot verify individual fit; it can only surface films whose watch_context metadata names that viewing profile."
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "trait",
    "parameters": {
      "qualifier_inventory": "One atom: a cross-generational personal-audience fit, translated as 'good for watching with an older relative, gentle content, broadly accessible' — the watch_context register this most plausibly lands on.",
      "space_queries": [
        {
          "carries_qualifiers": "watch_context watch_scenarios carries situational descriptors for who to watch with ('with grandparents', 'family movie night across generations'); external_motivations carries the relational pull of picking a film for an older relative's sensibilities.",
          "space": "watch_context",
          "weight": "central",
          "content": {
            "self_experience_motivations": {
              "terms": []
            },
            "external_motivations": {
              "terms": ["picking for an older relative", "family viewing across generations"]
            },
            "key_movie_feature_draws": {
              "terms": ["gentle", "broadly accessible"]
            },
            "watch_scenarios": {
              "terms": ["watch with grandparents", "multi-generational family night", "older-relative viewing"]
            }
          }
        }
      ],
      "primary_vector": "watch_context"
    },
    "polarity": "positive"
  }
}
```

**Example: reception-shape ask → reception space**

```xml
<raw_query>movies critics still argue about</raw_query>
<overall_query_intention_exploration>The user wants films whose reception itself has a specific shape — an unresolved, contested critical conversation that persists. It is not an award ask, not a metadata reception score, and not "acclaimed". It names a specific reception texture: divisiveness and continued debate. A matching film's reception_summary on the ingest side would describe exactly this pattern.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films whose critical reception is divisive and remains an active debate.</captured_meaning>
  <category_name>Interpretation-required</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies that critics continue to debate.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies critics still argue about</query_text>
  <description>Films with ongoing, divisive critical conversation.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Interpretation-driven: match films whose reception prose names divisiveness and an ongoing critical debate as the reception shape.",
      "relation_to_endpoint": "reception.reception_summary is the ingest-side prose field that would describe a film as 'divisive', 'polarizing', 'the subject of ongoing critical debate', or 'reassessed with each generation'. That vocabulary is the direct-match target for 'critics still argue about'. A scalar reception score (metadata) cannot distinguish a divisive reception from a middling one; the texture lives in the prose.",
      "coverage_gaps": "Interpretation-driven query — relies on reception prose carrying the debate/divisiveness framing. Films that are divisive but whose ingest-side summary does not call it out will be under-surfaced."
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "trait",
    "parameters": {
      "qualifier_inventory": "One atom, reception-shape: the film's critical reception is contested and the debate remains active over time.",
      "space_queries": [
        {
          "carries_qualifiers": "reception reception_summary is the prose field that names reception texture on the ingest side; 'divisive', 'polarizing', 'ongoing critical debate', 'reassessed over time' are the native phrasings for this shape.",
          "space": "reception",
          "weight": "central",
          "content": {
            "reception_summary": "a divisive, polarizing reception with ongoing critical debate that has continued to be reassessed over time",
            "praised_qualities": [],
            "criticized_qualities": []
          }
        }
      ],
      "primary_vector": "reception"
    },
    "polarity": "positive"
  }
}
```

**Example: empty intent → no-fire**

```xml
<raw_query>you know, movies</raw_query>
<overall_query_intention_exploration>The filler phrase 'you know, movies' carries no resolvable intent. Upstream dispatch kept the fragment because it contained a noun, but decoding the underlying ask surfaces nothing — no genre, no mood, no occasion, no theme.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>A conversational filler with no decodable film preference.</captured_meaning>
  <category_name>Interpretation-required</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies (filler — no preference specified).</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>you know, movies</query_text>
  <description>Filler phrase with no specific preference.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "No decodable intent — the phrase is conversational filler with no film preference attached.",
      "relation_to_endpoint": "Every semantic space would require naming a concrete signal the user wants — a theme, tone, motivation, reception shape, or production detail. The input names none; populating any space would mean fabricating vocabulary the user did not imply. No honest query can be constructed.",
      "coverage_gaps": "The fragment has no resolvable intent to interpret. Upstream dispatch should have treated this as no_fit rather than routing to the fallback; at handler time, no-fire is the correct response."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```
