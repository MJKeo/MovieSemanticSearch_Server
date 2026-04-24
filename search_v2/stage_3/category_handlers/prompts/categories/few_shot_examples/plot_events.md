# Examples

These examples calibrate a clean plot-event fire, a narrative-time-plus-place fire, a narrative-place-only fire, and two no-fire boundaries: a filming-geography misroute (Cat 13) and a purely thematic ask (Cat 21).

**Example: clean fire on a concrete plot-event description**

```xml
<raw_query>a heist that unravels when one of the crew betrays the others</raw_query>
<overall_query_intention_exploration>The user is describing a specific plot situation — a heist story where the twist is an internal betrayal. This is event-level content, not a thematic abstraction and not a named label.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>A heist movie where the plot turns on one of the crew betraying the others mid-job.</captured_meaning>
  <category_name>Plot events + narrative setting</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies whose plot describes a heist that falls apart due to a crew member betraying the others.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>a heist that unravels when one of the crew betrays the others</query_text>
  <description>A heist plot that falls apart when a crew member betrays the team.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Find films whose plot is a heist that breaks down because of an internal betrayal by a crew member.",
      "relation_to_endpoint": "plot_events carries the raw synopsis prose at ingest — concrete events, characters, and on-screen action. A dense plot-summary body describing a heist unraveling through a crew betrayal matches the same register as a matching movie's ingest-side plot summary.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "qualifier_inventory": "One atom: a heist plot whose turning point is an internal betrayal by a crew member, causing the job to fall apart.",
      "space_queries": [
        {
          "carries_qualifiers": "plot_events is the only space whose ingest body is synopsis prose; the concrete event (a heist derailed by a crew member's betrayal) lands directly there.",
          "space": "plot_events",
          "weight": "central",
          "content": {
            "plot_summary": "a crew pulls off a heist that unravels when one of the members betrays the others mid-job, turning the team against itself and collapsing the score."
          }
        }
      ],
      "primary_vector": "plot_events"
    },
    "polarity": "positive"
  }
}
```

**Example: clean fire on narrative time + place**

```xml
<raw_query>movies set in 1940s Berlin</raw_query>
<overall_query_intention_exploration>The user wants films whose story takes place in 1940s Berlin — narrative time and place. Not films released in the 1940s, not films shot in modern Berlin.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films whose story is set in Berlin during the 1940s.</captured_meaning>
  <category_name>Plot events + narrative setting</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies whose narrative setting is 1940s Berlin.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies set in 1940s Berlin</query_text>
  <description>Films with a 1940s Berlin narrative setting.</description>
  <modifiers>
    <modifier>
      <original_text>set in</original_text>
      <effect>binds the following phrase as the story's setting rather than production geography or release era</effect>
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
      "aspect_description": "Find films whose narrative time is the 1940s and whose narrative place is Berlin.",
      "relation_to_endpoint": "Narrative time and place have no closed-schema column — they surface on the ingest side only inside the plot_events synopsis prose, which routinely names when and where the story takes place. Embedding a short synopsis body that grounds the story in 1940s Berlin matches that ingest shape.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "qualifier_inventory": "Two atoms folded into one setting: narrative time = 1940s; narrative place = Berlin. No event content is specified — just the setting.",
      "space_queries": [
        {
          "carries_qualifiers": "plot_events synopsis prose at ingest names the story's time and place alongside events; a body set in 1940s Berlin lands in the same prose shape.",
          "space": "plot_events",
          "weight": "central",
          "content": {
            "plot_summary": "a story set in 1940s berlin, unfolding in the city during that era."
          }
        }
      ],
      "primary_vector": "plot_events"
    },
    "polarity": "positive"
  }
}
```

**Example: clean fire on narrative place only**

```xml
<raw_query>something that takes place in Tokyo</raw_query>
<overall_query_intention_exploration>The user wants films whose story is set in Tokyo. Narrative place — where the story happens, not where the cameras were.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films whose story is set in Tokyo.</captured_meaning>
  <category_name>Plot events + narrative setting</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies whose narrative takes place in Tokyo.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>something that takes place in Tokyo</query_text>
  <description>Films whose story is set in Tokyo.</description>
  <modifiers>
    <modifier>
      <original_text>takes place in</original_text>
      <effect>binds the following location as the story's setting</effect>
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
      "aspect_description": "Find films whose story is set in Tokyo.",
      "relation_to_endpoint": "plot_events synopsis prose is the only ingest surface that states where a story is set. Embedding a short body grounding the story in Tokyo matches that surface; production.filming_locations is a different question (where the crew shot) and would miss stories set in Tokyo but shot on a soundstage or abroad.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "qualifier_inventory": "One atom: Tokyo as the narrative place of the film. No era and no event content specified.",
      "space_queries": [
        {
          "carries_qualifiers": "plot_events prose at ingest states the city the story unfolds in; 'takes place in Tokyo' lands directly in that register.",
          "space": "plot_events",
          "weight": "central",
          "content": {
            "plot_summary": "a story set in tokyo, with the city as its backdrop."
          }
        }
      ],
      "primary_vector": "plot_events"
    },
    "polarity": "positive"
  }
}
```

**Example: no-fire on a filming-geography ask (Cat 13 boundary)**

```xml
<raw_query>filmed in Tokyo</raw_query>
<overall_query_intention_exploration>The user wants films physically shot in Tokyo — production geography, not narrative setting. The camera was there, not necessarily the story.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films whose story takes place in Tokyo.</captured_meaning>
  <category_name>Plot events + narrative setting</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies whose narrative takes place in Tokyo.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>filmed in Tokyo</query_text>
  <description>Films physically shot in Tokyo.</description>
  <modifiers>
    <modifier>
      <original_text>filmed in</original_text>
      <effect>binds the following place as a shooting location rather than story setting</effect>
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
      "aspect_description": "The phrase names a physical shooting location, not a narrative setting. The ROLE_MARKER 'filmed in' binds Tokyo to production geography.",
      "relation_to_endpoint": "plot_events synopsis prose describes where a story is set, not where the crew shot. Embedding 'Tokyo' into a plot_summary body would match films whose story takes place in Tokyo — which is a different question from whether the camera was there. Filming geography lives in the production space and routes through Cat 13 (Filming location).",
      "coverage_gaps": "Upstream dispatch misread 'filmed in' as narrative setting. The correct category is Cat 13; firing plot_events here would return films set in Tokyo regardless of where they were shot, silently missing Hollywood productions that actually filmed on location in Tokyo."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```

**Example: no-fire on a purely thematic ask (Cat 21 boundary)**

```xml
<raw_query>a story about grief</raw_query>
<overall_query_intention_exploration>The user wants films built around grief as a thematic concept. This is an abstract theme, not a concrete event or a specific setting.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>A film about the experience of grief.</captured_meaning>
  <category_name>Plot events + narrative setting</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies whose plot centers on a character experiencing grief.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>a story about grief</query_text>
  <description>A film thematically centered on grief.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The phrase names an abstract theme (grief) with no concrete event, situation, or narrative setting attached.",
      "relation_to_endpoint": "plot_events carries synopsis prose — concrete actions, characters, where and when the story unfolds. A bare theme like 'grief' has no concrete event to embed and would surface in plot_analysis.thematic_concepts on the ingest side, not in plot_events prose. Forcing a plot_summary body built from the word 'grief' would drift into thematic territory that Cat 21 (Kind of story / thematic archetype) handles directly.",
      "coverage_gaps": "Upstream dispatch conflated a theme with a plot-event description. The correct category is Cat 21. plot_events has no event or setting atom here to ground a synopsis body."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```
