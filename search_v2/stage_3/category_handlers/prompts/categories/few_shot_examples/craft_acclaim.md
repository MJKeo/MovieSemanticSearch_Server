# Examples

These calibrate which spaces to populate per craft axis (reception always, production or narrative_techniques per axis), and the no-fire shape when the phrase names a creator rather than an axis.

**Example: visual craft acclaim — reception + production**

```xml
<raw_query>visually stunning movies</raw_query>
<overall_query_intention_exploration>The user wants films singled out for their visuals — cinematography, visual style, production design — as a praised craft axis. A craft-acclaim ask anchored on the visual dimension.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films acclaimed for their visual craft — cinematography, visual style, and production design.</captured_meaning>
  <category_name>Craft acclaim (visual / music / dialogue)</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies praised for visual craft — cinematography, visual style, production design.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>visually stunning movies</query_text>
  <description>Films acclaimed for their visuals.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify films praised specifically for visual craft — cinematography, visual style, production design as the named axis of acclaim.",
      "relation_to_endpoint": "Reception.praised_qualities carries axis-naming praise tags including 'cinematography', 'visual style', and 'production design' — a direct fit for the acclaim framing. Production.production_techniques carries the visible technique side of visual craft (long takes, practical effects, lighting design) which rounds out the match when the acclaim is for how the film looks.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "trait",
    "parameters": {
      "qualifier_inventory": "Visual acclaim as the headline axis; cinematography and visual style as the praised dimension; production design and visible technique as the rounding-out craft signal.",
      "space_queries": [
        {
          "carries_qualifiers": "reception carries the acclaim framing itself — praised_qualities names the exact axis ('cinematography', 'visual style', 'production design') that ingest-side tags would carry for a matching film.",
          "space": "reception",
          "weight": "central",
          "content": {
            "reception_summary": "praised for visual craft — cinematography, visual style, and production design singled out by critics and audiences.",
            "praised_qualities": ["cinematography", "visual style", "production design", "visual beauty"],
            "criticized_qualities": []
          }
        },
        {
          "carries_qualifiers": "production carries the visible technique side of visual craft — production_techniques labels the concrete methods (lighting, composition, practical effects) that shape the acclaimed look.",
          "space": "production",
          "weight": "supporting",
          "content": {
            "filming_locations": [],
            "production_techniques": ["striking cinematography", "strong visual composition", "distinctive lighting design"]
          }
        }
      ],
      "primary_vector": "reception"
    },
    "polarity": "positive"
  }
}
```

**Example: musical craft acclaim — reception only**

```xml
<raw_query>movies with an iconic score</raw_query>
<overall_query_intention_exploration>The user wants films singled out for their musical score — memorable, widely praised, a headline craft axis. Musical craft as acclaim.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films acclaimed for an iconic musical score.</captured_meaning>
  <category_name>Craft acclaim (visual / music / dialogue)</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies praised for an iconic musical score.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies with an iconic score</query_text>
  <description>Films acclaimed for a memorable musical score.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify films praised specifically for their musical score — music as the named axis of acclaim.",
      "relation_to_endpoint": "Reception.praised_qualities carries axis-naming praise tags including 'musical score', 'soundtrack', and 'memorable theme' — a direct fit for music-craft acclaim. No other vector space carries music-craft vocabulary at ingest, so reception is the only honest target here.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "trait",
    "parameters": {
      "qualifier_inventory": "Musical score as the headline praised axis; iconic, memorable, widely recognized as the shape of the acclaim.",
      "space_queries": [
        {
          "carries_qualifiers": "reception carries both the acclaim framing and the music-craft axis — praised_qualities names 'musical score' / 'iconic theme' / 'memorable soundtrack', the exact ingest-side phrasing a matching film would carry.",
          "space": "reception",
          "weight": "central",
          "content": {
            "reception_summary": "praised for an iconic musical score — music singled out as a headline craft achievement.",
            "praised_qualities": ["musical score", "iconic theme", "memorable soundtrack", "composer's work"],
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

**Example: dialogue craft acclaim — reception + narrative_techniques**

```xml
<raw_query>movies with quotable dialogue</raw_query>
<overall_query_intention_exploration>The user wants films singled out for their dialogue — sharp, memorable, quotable lines as a headline craft axis. Dialogue-craft acclaim.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films acclaimed for sharp, quotable dialogue.</captured_meaning>
  <category_name>Craft acclaim (visual / music / dialogue)</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies praised for quotable, memorable dialogue.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies with quotable dialogue</query_text>
  <description>Films acclaimed for sharp, memorable dialogue.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify films praised specifically for dialogue craft — sharp, memorable, quotable writing as the named axis of acclaim.",
      "relation_to_endpoint": "Reception.praised_qualities carries dialogue-as-praise tags ('dialogue', 'script', 'screenplay'). Narrative_techniques' narrative_delivery and characterization_methods carry the craft-of-writing descriptors that round out dialogue-driven acclaim on the ingest side. Both spaces genuinely apply — reception frames it as praise, narrative_techniques frames it as the craft being praised.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "trait",
    "parameters": {
      "qualifier_inventory": "Dialogue as the headline praised axis; quotable, sharp, memorable lines as the shape of the acclaim; writing-craft as the underlying dimension being celebrated.",
      "space_queries": [
        {
          "carries_qualifiers": "reception carries the acclaim framing — praised_qualities names dialogue and script as the axis being celebrated, matching ingest-side praise tags for films singled out for their writing.",
          "space": "reception",
          "weight": "central",
          "content": {
            "reception_summary": "praised for sharp, quotable dialogue — writing and script singled out as a headline craft achievement.",
            "praised_qualities": ["dialogue", "quotable lines", "sharp writing", "memorable screenplay"],
            "criticized_qualities": []
          }
        },
        {
          "carries_qualifiers": "narrative_techniques carries the craft-of-writing side — narrative_delivery captures dialogue-driven storytelling, characterization_methods captures speech as a character-surfacing method, and additional_narrative_devices carries 'quotable dialogue' as the labeled device.",
          "space": "narrative_techniques",
          "weight": "supporting",
          "content": {
            "narrative_archetype": {"terms": []},
            "narrative_delivery": {"terms": ["dialogue-driven storytelling", "verbal wit"]},
            "pov_perspective": {"terms": []},
            "characterization_methods": {"terms": ["characters defined through speech", "sharp verbal voice"]},
            "character_arcs": {"terms": []},
            "audience_character_perception": {"terms": []},
            "information_control": {"terms": []},
            "conflict_stakes_design": {"terms": []},
            "additional_narrative_devices": {"terms": ["quotable dialogue", "memorable one-liners"]}
          }
        }
      ],
      "primary_vector": "reception"
    },
    "polarity": "positive"
  }
}
```

**Example: named below-the-line creator → no-fire (Cat 29 territory)**

```xml
<raw_query>Roger Deakins movies</raw_query>
<overall_query_intention_exploration>The user names a specific cinematographer — a below-the-line creator — rather than asking about a craft axis in the abstract. The ask is credit-driven, not axis-driven.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films shot by cinematographer Roger Deakins.</captured_meaning>
  <category_name>Craft acclaim (visual / music / dialogue)</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies whose cinematographer is Roger Deakins.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>Roger Deakins movies</query_text>
  <description>Films shot by Roger Deakins.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify films by a specific named below-the-line creator — cinematographer Roger Deakins.",
      "relation_to_endpoint": "The target is a named creator, not a craft axis in the abstract. Reception.praised_qualities carries axis-naming tags ('cinematography') but not person-level attribution, and inventing 'Deakins-style cinematography' in the praise field would fabricate ingest content that matching films do not actually carry. The below-the-line creator endpoint owns this ask; dispatch was wrong.",
      "coverage_gaps": "The ask names a specific creator rather than a craft axis. Firing reception would embed a praise-tag payload against films whose connection to Deakins is by credit, not by axis-specific acclaim tagging."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```

**Example: named director → no-fire (Cat 1 territory)**

```xml
<raw_query>Christopher Nolan films</raw_query>
<overall_query_intention_exploration>The user names a specific director — a credit-indexed role — rather than asking about a craft axis in the abstract. The ask is credit-driven, not axis-driven.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films directed by Christopher Nolan.</captured_meaning>
  <category_name>Craft acclaim (visual / music / dialogue)</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies whose director is Christopher Nolan.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>Christopher Nolan films</query_text>
  <description>Films directed by Christopher Nolan.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify films by a specific named director — Christopher Nolan.",
      "relation_to_endpoint": "The target is a named indexed-role credit, not a craft axis. The credit-and-title endpoint owns this ask. Reception has no person-level attribution in praised_qualities, and populating a 'Nolan-style' phrasing would fabricate ingest content no matching film actually carries. Dispatch to this category was wrong.",
      "coverage_gaps": "The ask names a specific director rather than any craft axis. Firing the semantic endpoint would embed a praise-tag payload that cannot match films by credit."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```
