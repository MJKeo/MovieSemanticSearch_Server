# Examples

These examples calibrate a clean filming-country fire, a clean filming-region fire, and the three main no-fire boundaries: narrative-setting asks (Cat 20), cultural-tradition asks (Cat 12), and country-of-origin asks (Cat 10).

**Example: clean fire on a filming-country ask**

```xml
<raw_query>movies filmed in New Zealand</raw_query>
<overall_query_intention_exploration>The user wants films that were physically shot in New Zealand. Filming geography, not story setting — New Zealand as a shooting location.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films that were physically shot on location in New Zealand.</captured_meaning>
  <category_name>Filming location</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies whose physical production was filmed in New Zealand.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies filmed in New Zealand</query_text>
  <description>Films shot in New Zealand.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Find films whose physical shooting took place in New Zealand.",
      "relation_to_endpoint": "Semantic production space carries filming_locations as a proper-noun sub-field at ingest. Embedding 'New Zealand' into the query body matches movies whose ingest text lists New Zealand as a shooting location — covering Hollywood-funded films shot in NZ whose country_of_origin is US.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "qualifier_inventory": "One atom: New Zealand as the physical shooting location of the film.",
      "space_queries": [
        {
          "carries_qualifiers": "production carries filming_locations on the ingest side as proper-noun place names; 'New Zealand' lands directly in that sub-field.",
          "space": "production",
          "weight": "central",
          "content": {
            "filming_locations": ["New Zealand"],
            "production_techniques": []
          }
        }
      ],
      "primary_vector": "production"
    },
    "polarity": "positive"
  }
}
```

**Example: clean fire on a specific filming region**

```xml
<raw_query>shot on location in Iceland</raw_query>
<overall_query_intention_exploration>The user wants films that used Iceland as a shooting location — the landscape itself, not Icelandic cinema as a tradition.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films that were physically shot on location in Iceland.</captured_meaning>
  <category_name>Filming location</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies whose physical production used Iceland as a shooting location.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>shot on location in Iceland</query_text>
  <description>Films shot on location in Iceland.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Find films whose shooting took place on location in Iceland.",
      "relation_to_endpoint": "Semantic production filming_locations holds the proper-noun place names from ingest-side production data. 'Iceland' embedded there surfaces films that used Icelandic locations regardless of their legal production country.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "qualifier_inventory": "One atom: Iceland as the physical on-location shooting geography of the film.",
      "space_queries": [
        {
          "carries_qualifiers": "production filming_locations holds proper-noun place names; 'Iceland' is the direct term on the ingest side for films that shot there.",
          "space": "production",
          "weight": "central",
          "content": {
            "filming_locations": ["Iceland"],
            "production_techniques": []
          }
        }
      ],
      "primary_vector": "production"
    },
    "polarity": "positive"
  }
}
```

**Example: no-fire on a narrative-setting ask (Cat 20 boundary)**

```xml
<raw_query>movies set in 1940s Berlin</raw_query>
<overall_query_intention_exploration>The user wants films whose story takes place in 1940s Berlin. This is narrative setting — where and when the story happens — not where the camera was.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films whose physical production was conducted in 1940s Berlin.</captured_meaning>
  <category_name>Filming location</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies filmed in 1940s Berlin.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies set in 1940s Berlin</query_text>
  <description>Films with a 1940s Berlin narrative setting.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The user is asking about narrative setting — where and when the story takes place — not where the film was shot.",
      "relation_to_endpoint": "Semantic production filming_locations carries physical shooting geography, which is orthogonal to story setting. Embedding '1940s Berlin' into production would match films whose crew shot there, not films whose story is set there. Narrative place/time setting belongs to plot_events and routes through Cat 20, not here.",
      "coverage_gaps": "Upstream dispatch misread 'set in' as filming geography. The phrase names a narrative setting; Cat 20 (Plot events + narrative setting) is the correct category. Firing production would silently return films shot in modern-day Berlin regardless of their story's era or setting."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```

**Example: no-fire on a cultural-tradition ask (Cat 12 boundary)**

```xml
<raw_query>Japanese cinema</raw_query>
<overall_query_intention_exploration>The user wants films belonging to Japanese cinema as a national cinematic tradition — not films merely shot in Japan. This is a tradition ask.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films that were physically shot in Japan.</captured_meaning>
  <category_name>Filming location</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies filmed in Japan.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>Japanese cinema</query_text>
  <description>Films from the Japanese national cinematic tradition.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The user is asking about a national cinematic tradition, not a shooting location.",
      "relation_to_endpoint": "Semantic production filming_locations holds proper-noun place names a crew shot in. 'Japanese cinema' names a tradition — a body of work defined by cultural and stylistic lineage — not a physical place where shooting happened. Treating this as filming geography would miss Japanese-tradition films shot abroad and surface Hollywood films shot in Tokyo. Cat 12 (Cultural tradition / national cinema) is the correct category.",
      "coverage_gaps": "Upstream dispatch conflated a cinematic tradition with a filming location. The production space has no lever for tradition membership; this needs to route through Cat 12."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```

**Example: no-fire on a country-of-origin ask (Cat 10 boundary)**

```xml
<raw_query>American productions</raw_query>
<overall_query_intention_exploration>The user wants films whose legal / financial production origin is the United States — country-of-origin framing, not a claim about where shooting took place.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films that were physically shot in the United States.</captured_meaning>
  <category_name>Filming location</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies filmed in the United States.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>American productions</query_text>
  <description>Films produced out of the United States.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The user is asking about legal/financial production country, not physical shooting location.",
      "relation_to_endpoint": "Semantic production filming_locations carries where the camera was, which is a different question from which country legally produced the film. Hollywood productions routinely shoot abroad; forcing 'United States' through filming_locations would miss US-produced films shot in Canada, Jordan, or New Zealand and include foreign-produced films shot in America. Country-of-origin is a structured column that lives in Cat 10 (Structured metadata).",
      "coverage_gaps": "Upstream dispatch conflated production country with filming geography. The correct target is the metadata country_of_origin column, not the production space."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```
