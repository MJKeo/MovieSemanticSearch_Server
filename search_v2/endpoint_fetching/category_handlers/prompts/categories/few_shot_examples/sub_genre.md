# Examples

These calibrate the Keyword-vs-Semantic decision for sub-genre and story-archetype asks: clean keyword wins when a registry member names the sub-form, lower-tier Semantic wins when the registry only carries the parent genre or the pattern is a compound the closed vocabulary cannot absorb, and the no-fire shape when the atom is too vague to target either channel.

**Example: sub-genre with a direct registry member → keyword wins cleanly**

```xml
<raw_query>body horror movies</raw_query>
<overall_query_intention_exploration>The user wants body-horror films — horror where bodily transformation, mutation, or physical degradation is the source of fear. A clean named sub-genre ask.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films in the body-horror sub-genre.</captured_meaning>
  <category_name>Sub-genre + story archetype</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies in the body-horror sub-genre.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>body horror movies</query_text>
  <description>Films in the body-horror sub-genre.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the body-horror sub-genre — horror driven by bodily transformation, mutation, or physical degradation.",
      "endpoint_coverage": "Keyword has a canonical registry member, BODY_HORROR, whose definition names horror movies where bodily transformation, mutilation, or physical degradation is the main source of fear — a direct cover for the sub-genre. Semantic's plot_analysis.genre_signatures could also carry 'body horror' as a label, but that is strictly weaker when a definitionally-matching tag exists.",
      "best_endpoint": "keyword",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "Keyword wins on its own merits — the BODY_HORROR tag's definition covers the sub-genre exactly, so the authoritative posting list delivers the right set. The tier-1 bias and the on-merits choice agree; this is not a close call that the bias had to break.",
  "endpoint_to_run": "keyword",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The query names 'body horror' — a horror sub-genre centered on bodily transformation, mutation, or physical degradation as the source of fear. A named sub-form, not a bare top-level genre.",
      "candidate_shortlist": "BODY_HORROR definition names exactly this sub-genre. HORROR is the parent family — picking it would broaden the set to every horror film, dropping the sub-form signal. No other member names the body-horror pattern specifically.",
      "classification": "body_horror"
    },
    "polarity": "positive"
  }
}
```

**Example: story archetype with a direct registry member → keyword wins cleanly**

```xml
<raw_query>heist movies</raw_query>
<overall_query_intention_exploration>The user wants heist films — crime stories centered on planning and executing a major theft, typically by a skilled crew. A clean named story archetype.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films built around a heist plot.</captured_meaning>
  <category_name>Sub-genre + story archetype</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies built around a heist plot.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>heist movies</query_text>
  <description>Films centered on planning and executing a major theft.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the heist story archetype — a crime plot built around planning and executing a major theft, typically by a skilled team.",
      "endpoint_coverage": "Keyword has a canonical registry member, HEIST, whose definition names crime movies centered on meticulously planning and executing a major theft or robbery carried out by a skilled team — a direct cover for the archetype. Semantic's plot_analysis.conflict_type could carry 'heist' as a conflict archetype, but the tag is authoritative when it covers the pattern definitionally.",
      "best_endpoint": "keyword",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "Keyword wins on merits — the HEIST tag definition matches the requirement exactly. The bias reinforces a choice that would stand on its own even without it.",
  "endpoint_to_run": "keyword",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The query asks for heist films — the premise is the planning and execution of a major theft by a crew. A named story archetype, not a character type and not a broad genre.",
      "candidate_shortlist": "HEIST definition names exactly this archetype. CON_ARTIST is adjacent (also crime-premised) but its definition names grifter/scammer deception as a craft, not a planned theft — it would misclassify the target set. No other archetype or plot member names the heist pattern.",
      "classification": "heist"
    },
    "polarity": "positive"
  }
}
```

**Example: lower-tier-wins — sub-genre whose parent has a tag but the sub-form does not → semantic**

```xml
<raw_query>neo-noir movies</raw_query>
<overall_query_intention_exploration>The user wants neo-noir films — the modern reinterpretation of classic film noir, carrying noir's moral ambiguity, fatalism, and stylized crime-drama shape into a contemporary setting. The registry covers FILM_NOIR but not neo-noir specifically.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films in the neo-noir sub-genre.</captured_meaning>
  <category_name>Sub-genre + story archetype</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies in the neo-noir sub-genre.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>neo-noir movies</query_text>
  <description>Films in the neo-noir sub-genre — modern reinterpretations of classic film noir.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the neo-noir sub-genre — the modern reinterpretation of classic film noir's moral ambiguity, fatalism, and stylized crime-drama shape.",
      "endpoint_coverage": "Keyword has FILM_NOIR but no member whose definition names the neo-noir sub-form specifically. FILM_NOIR is the parent-era label — picking it would return classical 1940s-1950s noirs and miss the modern reinterpretations the user asked for. Semantic's plot_analysis.genre_signatures is the purpose-built sub-field for sub-genre labels the registry does not absorb, and viewer_experience can carry supporting signal for noir's fatalistic tone.",
      "best_endpoint": "semantic",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "The tier-1 bias would pick Keyword, but no registry member's definition names neo-noir specifically — FILM_NOIR is the parent-era label, not a match, and picking it to honor the bias would silently broaden the lookup past the user's intent. Semantic is clearly the better channel on its own merits: genre_signatures exists to carry sub-genre labels the closed vocabulary does not cover. The bias is not a veto when the lower-preference endpoint fits decisively better.",
  "endpoint_to_run": "semantic",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "qualifier_inventory": "neo-noir sub-genre label; modern reinterpretation of classic film noir; fatalistic tone and moral ambiguity as defining signatures.",
      "space_queries": [
        {
          "carries_qualifiers": "plot_analysis carries the neo-noir label under genre_signatures and the fatalistic/moral-ambiguity shape under thematic_concepts — the target sub-fields for an uncanonized sub-genre.",
          "space": "plot_analysis",
          "weight": "central",
          "content": {
            "elevator_pitch": null,
            "plot_overview": null,
            "genre_signatures": ["neo-noir", "modern noir", "stylized crime drama"],
            "conflict_type": [],
            "thematic_concepts": ["moral ambiguity", "fatalism", "corruption"],
            "character_arcs": []
          }
        },
        {
          "carries_qualifiers": "viewer_experience carries the cynical, fatalistic tone that is load-bearing to the neo-noir identity beyond the genre label itself.",
          "space": "viewer_experience",
          "weight": "supporting",
          "content": {
            "emotional_palette": {"terms": ["cynical", "fatalistic", "melancholic"], "negations": []},
            "tension_adrenaline": {"terms": [], "negations": []},
            "tone_self_seriousness": {"terms": ["morally grey", "stylized seriousness"], "negations": []},
            "cognitive_complexity": {"terms": [], "negations": []},
            "disturbance_profile": {"terms": [], "negations": []},
            "sensory_load": {"terms": [], "negations": []},
            "emotional_volatility": {"terms": [], "negations": []},
            "ending_aftertaste": {"terms": [], "negations": []}
          }
        }
      ],
      "primary_vector": "plot_analysis"
    },
    "polarity": "positive"
  }
}
```

**Example: lower-tier-wins — compound pattern no single tag covers → semantic**

```xml
<raw_query>slow-burn revenge thrillers</raw_query>
<overall_query_intention_exploration>The user wants revenge-driven thrillers with a slow-burn pacing profile — the revenge archetype plus a specific tension shape. A compound pattern: REVENGE covers the plot engine, but the slow-burn pacing qualifier is not absorbed by the revenge tag.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Thrillers built around a revenge plot with slow-burn pacing.</captured_meaning>
  <category_name>Sub-genre + story archetype</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Revenge thrillers with slow-burn pacing.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>slow-burn revenge thrillers</query_text>
  <description>Revenge-driven thrillers with a slow-burn tension profile.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the compound pattern: a revenge-driven thriller plot combined with a slow-burn pacing and tension profile.",
      "endpoint_coverage": "Keyword has REVENGE, which covers the plot engine cleanly — but a single member can carry only one dimension of this compound and the registry has no member that combines revenge with slow-burn pacing. Picking REVENGE alone drops the pacing qualifier the user explicitly named, returning high-adrenaline revenge films alongside the slow-burn ones. Semantic can populate the full compound: plot_analysis.conflict_type carries the revenge archetype, viewer_experience.tension_adrenaline carries the slow-burn profile, and plot_events can supplement with the deliberately-paced revenge situation.",
      "best_endpoint": "semantic",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "Keyword covers half the requirement cleanly via REVENGE but cannot absorb the slow-burn pacing qualifier — the one-member-per-call constraint forces it to drop a named dimension. Semantic captures both dimensions in their native sub-fields. The tier-1 bias does not force a half-match when the lower-preference endpoint expresses the full compound. Semantic wins on its own merits.",
  "endpoint_to_run": "semantic",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "qualifier_inventory": "revenge-driven plot engine; thriller genre shape; slow-burn pacing and tension profile.",
      "space_queries": [
        {
          "carries_qualifiers": "plot_analysis carries the revenge archetype as a conflict type and the thriller signature under genre_signatures — the categorical shape of the story.",
          "space": "plot_analysis",
          "weight": "central",
          "content": {
            "elevator_pitch": null,
            "plot_overview": null,
            "genre_signatures": ["revenge thriller", "slow-burn thriller"],
            "conflict_type": ["revenge", "vengeance pursuit"],
            "thematic_concepts": ["vengeance", "retribution"],
            "character_arcs": []
          }
        },
        {
          "carries_qualifiers": "viewer_experience carries the slow-burn tension profile — the load-bearing pacing dimension the revenge tag cannot absorb.",
          "space": "viewer_experience",
          "weight": "central",
          "content": {
            "emotional_palette": {"terms": [], "negations": []},
            "tension_adrenaline": {"terms": ["slow-burn tension", "simmering dread", "mounting suspense"], "negations": ["frenetic action", "high-adrenaline"]},
            "tone_self_seriousness": {"terms": [], "negations": []},
            "cognitive_complexity": {"terms": [], "negations": []},
            "disturbance_profile": {"terms": [], "negations": []},
            "sensory_load": {"terms": [], "negations": []},
            "emotional_volatility": {"terms": [], "negations": []},
            "ending_aftertaste": {"terms": [], "negations": []}
          }
        },
        {
          "carries_qualifiers": "plot_events carries the concrete situational pattern of a deliberately-paced revenge pursuit as the ingest-side plot summary would describe it.",
          "space": "plot_events",
          "weight": "supporting",
          "content": {
            "plot_summary": "a protagonist methodically pursues revenge against those who wronged them, with the tension building gradually rather than through frequent action set pieces"
          }
        }
      ],
      "primary_vector": "plot_analysis"
    },
    "polarity": "positive"
  }
}
```

**Example: too-vague ask → no-fire**

```xml
<raw_query>movies with a cool genre vibe</raw_query>
<overall_query_intention_exploration>The user's phrasing names no specific sub-genre or archetype — "cool genre vibe" is a vague positive gesture without any identifiable story-pattern label.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films with an unspecified appealing genre feel.</captured_meaning>
  <category_name>Sub-genre + story archetype</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies with an appealing genre feel.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies with a cool genre vibe</query_text>
  <description>Films with an appealing but unspecified genre feel.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the named sub-genre or story archetype the user is asking for.",
      "endpoint_coverage": "Neither candidate can fit. Keyword requires a specific registry member whose definition covers the named sub-form — the phrase names no sub-form, so no member is definitionally supported. Semantic's plot_analysis sub-fields require concrete label phrases (a sub-genre name, a conflict archetype, an arc pattern) — 'cool genre vibe' supplies none, and inventing placeholder terms would dilute the query vector rather than target matching movies.",
      "best_endpoint": "None",
      "best_endpoint_gaps": "The requirement names no identifiable sub-form or story-pattern label. Firing either endpoint would fabricate content the input does not support."
    }
  ],
  "performance_vs_bias_analysis": "The tier-1 bias does not force a pick when no sub-form is actually named. Picking any registry member would invent a sub-genre the user did not specify, and populating plot_analysis with placeholder labels would pollute the semantic channel. No-fire is the correct response.",
  "endpoint_to_run": "None",
  "endpoint_parameters": null
}
```
