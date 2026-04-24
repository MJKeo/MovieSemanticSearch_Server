# Examples

These calibrate the name-plus-role reception payload for below-the-line creators, and the no-fire shape when the query either names no creator (Cat 24 territory) or names an indexed-role creator (Cat 1 territory).

**Example: cinematographer by name — reception fire**

```xml
<raw_query>Roger Deakins films</raw_query>
<overall_query_intention_exploration>The user names a specific cinematographer — a below-the-line, non-indexed creator role. The only honest retrieval path is semantic reception, where critics routinely name Deakins alongside the cinematography axis in praise prose.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films shot by cinematographer Roger Deakins.</captured_meaning>
  <category_name>Below-the-line creator</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies whose cinematographer is Roger Deakins.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>Roger Deakins films</query_text>
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
      "aspect_description": "Identify films whose cinematographer is Roger Deakins — a below-the-line creator named by name.",
      "relation_to_endpoint": "Cinematographer is not an indexed role, so semantic reception is the only channel that can carry the name. Reception.praised_qualities and reception_summary routinely name cinematographers alongside the cinematography axis at ingest, which is exactly the pairing a matching film would carry.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "qualifier_inventory": "Roger Deakins as the named creator; cinematographer / director of photography as his below-the-line role; cinematography as the craft axis that will anchor the reception-side mention.",
      "space_queries": [
        {
          "carries_qualifiers": "reception carries name-plus-role attribution in praised_qualities and reception_summary — matching films have ingest-side prose naming Deakins alongside the cinematography axis.",
          "space": "reception",
          "weight": "central",
          "content": {
            "reception_summary": "cinematography by Roger Deakins, widely praised for its lighting, composition, and visual craft.",
            "praised_qualities": ["cinematography by Roger Deakins", "Roger Deakins' cinematography", "cinematography", "visual craft"],
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

**Example: editor by name — reception fire**

```xml
<raw_query>Thelma Schoonmaker-edited movies</raw_query>
<overall_query_intention_exploration>The user names a specific editor — a below-the-line, non-indexed role. Reception prose naming Schoonmaker alongside the editing axis is the only ingest-side surface where this pairing lives.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films edited by Thelma Schoonmaker.</captured_meaning>
  <category_name>Below-the-line creator</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies whose editor is Thelma Schoonmaker.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>Thelma Schoonmaker-edited movies</query_text>
  <description>Films edited by Thelma Schoonmaker.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify films whose editor is Thelma Schoonmaker — a below-the-line creator named by name.",
      "relation_to_endpoint": "Editor is not an indexed role, so semantic reception is the only available channel. Reception prose and praised_qualities at ingest commonly name editors alongside the editing axis, so embedding a name-plus-role pairing is a truthful match target.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "qualifier_inventory": "Thelma Schoonmaker as the named creator; film editor as her below-the-line role; editing as the craft axis that will anchor the reception-side mention.",
      "space_queries": [
        {
          "carries_qualifiers": "reception carries name-plus-role attribution for editors in praised_qualities and reception_summary — a matching film's ingest prose names Schoonmaker alongside the editing axis.",
          "space": "reception",
          "weight": "central",
          "content": {
            "reception_summary": "edited by Thelma Schoonmaker, whose editing shapes the film's pacing and rhythm.",
            "praised_qualities": ["edited by Thelma Schoonmaker", "Thelma Schoonmaker's editing", "editing", "film editing"],
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

**Example: costume designer by name — reception fire**

```xml
<raw_query>Sandy Powell costumes</raw_query>
<overall_query_intention_exploration>The user names a specific costume designer — a below-the-line, non-indexed role. The honest retrieval path is reception prose pairing Powell's name with the costume-design axis.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films with costumes designed by Sandy Powell.</captured_meaning>
  <category_name>Below-the-line creator</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies whose costume designer is Sandy Powell.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>Sandy Powell costumes</query_text>
  <description>Films with costumes designed by Sandy Powell.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify films whose costume designer is Sandy Powell — a below-the-line creator named by name.",
      "relation_to_endpoint": "Costume designer is not an indexed role. Semantic reception is the only channel — matching films carry ingest prose naming Powell alongside the costume-design axis in praised_qualities.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "qualifier_inventory": "Sandy Powell as the named creator; costume designer as her below-the-line role; costume design as the craft axis that anchors the reception-side mention.",
      "space_queries": [
        {
          "carries_qualifiers": "reception carries name-plus-role attribution for costume designers in praised_qualities and reception_summary — a matching film's ingest prose names Powell alongside the costume-design axis.",
          "space": "reception",
          "weight": "central",
          "content": {
            "reception_summary": "costume design by Sandy Powell, singled out for its period craftsmanship and character-defining wardrobe.",
            "praised_qualities": ["costume design by Sandy Powell", "Sandy Powell's costumes", "costume design", "wardrobe"],
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

**Example: axis-only praise with no name → no-fire (Cat 24 territory)**

```xml
<raw_query>beautifully shot films</raw_query>
<overall_query_intention_exploration>The user praises a craft axis — visual / cinematography — without naming a specific creator. The ask is axis-driven, not credit-driven.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films acclaimed for beautiful cinematography.</captured_meaning>
  <category_name>Below-the-line creator</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies praised for beautiful cinematography.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>beautifully shot films</query_text>
  <description>Films acclaimed for beautiful cinematography.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify films acclaimed for visual / cinematography craft — no specific creator named.",
      "relation_to_endpoint": "This category requires a specific below-the-line creator named by name. The query provides a craft axis only ('beautifully shot') with no person attached. Reception prose without a name becomes an axis-only praise payload, which is Craft acclaim's territory, not this one. Firing here would embed a reception vector indistinguishable from a pure axis-praise ask — the name is what makes Cat 29 honest.",
      "coverage_gaps": "The ask names no creator. Without a name to pair with the axis, there is no below-the-line-specific signal to embed."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```

**Example: indexed-role director by name → no-fire (Cat 1 territory)**

```xml
<raw_query>Christopher Nolan films</raw_query>
<overall_query_intention_exploration>The user names a specific director — an indexed role handled by the Credit + title text category and the entity endpoint. Below-the-line is for non-indexed roles only.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films directed by Christopher Nolan.</captured_meaning>
  <category_name>Below-the-line creator</category_name>
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
      "relation_to_endpoint": "Director is an indexed role handled by the entity endpoint via Credit + title text. Below-the-line is the fallthrough for non-indexed creator roles only. Dispatch was wrong — routing a director through semantic reception would miss the deterministic credit match the entity endpoint provides and would embed a name-plus-role payload against a space whose praised_qualities rarely names directors in that form.",
      "coverage_gaps": "Named role is indexed (director), which is Cat 1's territory. No below-the-line role applies."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```
