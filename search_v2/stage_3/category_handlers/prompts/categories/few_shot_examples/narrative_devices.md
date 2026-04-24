# Examples

These calibrate the Keyword-vs-Semantic decision for narrative-device asks: clean keyword wins when a NarrativeStructureTag or ENSEMBLE_CAST definition names the device, lower-tier Semantic wins when the craft ask falls outside the closed vocabulary (dialogue style, pacing-as-feel), and the no-fire shape when the atom is too vague to target either channel.

**Example: canonical structural device named → keyword wins cleanly**

```xml
<raw_query>movies with a great plot twist</raw_query>
<overall_query_intention_exploration>The user wants films built around a major plot twist — a late-story reveal that reframes what came before. A canonical structural device the registry carries as PLOT_TWIST.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films that feature a significant plot twist.</captured_meaning>
  <category_name>Narrative devices + structural form + craft</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies structured around a major plot twist.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies with a great plot twist</query_text>
  <description>Films built around a significant plot twist.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the plot-twist device — a late-story reveal that reframes the preceding narrative.",
      "endpoint_coverage": "Keyword has the PLOT_TWIST registry member, whose definition names exactly this device — a major unexpected reveal that recontextualizes earlier events. Semantic's narrative_techniques.additional_narrative_devices could carry 'plot twist' as a label, but that is strictly weaker when a definitionally-matching tag exists.",
      "best_endpoint": "keyword",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "Keyword wins on its own merits — the PLOT_TWIST tag's definition covers the device exactly, so the authoritative posting list delivers the right set. The tier-1 bias and the on-merits choice agree; the bias did not have to break a close call.",
  "endpoint_to_run": "keyword",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The query names 'plot twist' — a canonical structural device in which a late reveal reframes earlier events. A named narrative mechanic, not a pacing or tone descriptor.",
      "candidate_shortlist": "PLOT_TWIST definition names exactly this device. TWIST_VILLAIN is adjacent but narrower — it names specifically the antagonist-reveal variant, which is not what the user asked for. CLIFFHANGER_ENDING describes an unresolved ending rather than a reframing reveal.",
      "classification": "plot_twist"
    },
    "polarity": "positive"
  }
}
```

**Example: canonical structural form named → keyword wins cleanly**

```xml
<raw_query>movies with an unreliable narrator</raw_query>
<overall_query_intention_exploration>The user wants films told through a narrator whose account cannot be trusted — a canonical POV/information-control device the registry carries as UNRELIABLE_NARRATOR.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films told through an unreliable narrator.</captured_meaning>
  <category_name>Narrative devices + structural form + craft</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies narrated by a character whose account the audience cannot trust.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies with an unreliable narrator</query_text>
  <description>Films whose narrator's account the audience cannot fully trust.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the unreliable-narrator device — a POV choice where the narrator's account is deliberately untrustworthy.",
      "endpoint_coverage": "Keyword has the UNRELIABLE_NARRATOR registry member, whose definition names exactly this POV device. Semantic's narrative_techniques.pov_perspective could also carry 'unreliable narrator' as a term, but the canonical tag is authoritative when it covers the device definitionally.",
      "best_endpoint": "keyword",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "Keyword wins on merits — the UNRELIABLE_NARRATOR tag definition matches the requirement exactly. The bias reinforces a choice that would stand on its own even without it.",
  "endpoint_to_run": "keyword",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The query names 'unreliable narrator' — a canonical POV/information-control device where the narrator's account cannot be trusted at face value. A named craft mechanic.",
      "candidate_shortlist": "UNRELIABLE_NARRATOR definition names exactly this device. NONLINEAR_TIMELINE is adjacent in the information-control family but covers temporal reordering rather than narrator credibility. PLOT_TWIST can overlap when the unreliability produces a late reveal, but the user framed the device itself (narrator trust), not the reveal.",
      "classification": "unreliable_narrator"
    },
    "polarity": "positive"
  }
}
```

**Example: lower-tier-wins — dialogue-style craft with no canonical tag → semantic**

```xml
<raw_query>movies with Sorkin-style dialogue</raw_query>
<overall_query_intention_exploration>The user wants films whose dialogue reads like Aaron Sorkin's — rapid, layered, walk-and-talk, articulate exchanges carrying most of the dramatic load. A craft-of-writing ask with no canonical registry tag.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films whose dialogue carries a Sorkin-style shape — rapid, layered, articulate exchanges driving the drama.</captured_meaning>
  <category_name>Narrative devices + structural form + craft</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies whose dialogue style resembles Aaron Sorkin's — rapid, layered, articulate exchanges.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies with Sorkin-style dialogue</query_text>
  <description>Films with rapid, layered, walk-and-talk dialogue in the Sorkin tradition.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the dialogue-style craft — rapid, layered, articulate exchanges in the Sorkin tradition as the carrier of dramatic weight.",
      "endpoint_coverage": "Keyword has no registry member whose definition names a dialogue style. The NarrativeStructureTag family covers structural devices (plot twist, nonlinear timeline, unreliable narrator, single location, time loop, fourth-wall breaks, endings) and ENSEMBLE_CAST covers cast configuration — none of these touch dialogue craft. Semantic's narrative_techniques.narrative_delivery and characterization_methods are the purpose-built sub-fields for dialogue-driven delivery and writer-level craft descriptors; additional_narrative_devices can carry the 'Sorkin-style' label itself.",
      "best_endpoint": "semantic",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "The tier-1 bias would pick Keyword, but no NarrativeStructureTag or other registry member names dialogue style — the closed vocabulary covers structural devices and cast configuration only. Picking a weakly-adjacent tag to honor the bias would misclassify the set. Semantic is clearly the better channel on its own merits: narrative_techniques exists precisely to carry craft descriptors the closed vocabulary does not absorb. The bias is not a veto when the lower-preference endpoint fits decisively better.",
  "endpoint_to_run": "semantic",
  "endpoint_parameters": {
    "match_mode": "trait",
    "parameters": {
      "qualifier_inventory": "Sorkin-style dialogue as the headline craft signature; rapid, layered, articulate verbal exchanges; dialogue as the primary carrier of drama rather than action or visuals.",
      "space_queries": [
        {
          "carries_qualifiers": "narrative_techniques carries the dialogue-delivery style as a craft choice — narrative_delivery captures dialogue-driven storytelling, characterization_methods captures verbal-articulacy as a character-surfacing method, and additional_narrative_devices carries the 'Sorkin-style' label itself.",
          "space": "narrative_techniques",
          "weight": "central",
          "content": {
            "narrative_archetype": {"terms": []},
            "narrative_delivery": {"terms": ["dialogue-driven storytelling", "walk-and-talk sequences", "rapid-fire dialogue"]},
            "pov_perspective": {"terms": []},
            "characterization_methods": {"terms": ["verbal articulacy", "characters defined through speech"]},
            "character_arcs": {"terms": []},
            "audience_character_perception": {"terms": []},
            "information_control": {"terms": []},
            "conflict_stakes_design": {"terms": ["verbal sparring", "argument as conflict"]},
            "additional_narrative_devices": {"terms": ["sorkin-style dialogue", "overlapping dialogue"]}
          }
        }
      ],
      "primary_vector": "narrative_techniques"
    },
    "polarity": "positive"
  }
}
```

**Example: lower-tier-wins — pacing-as-feel with no canonical tag → semantic on viewer_experience**

```xml
<raw_query>slow-burn movies</raw_query>
<overall_query_intention_exploration>The user wants films with a slow-burn profile — tension that accumulates gradually rather than arriving in bursts. A pacing-as-feel ask with no canonical registry tag for the pacing shape itself.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films whose tension builds gradually — a slow-burn pacing profile.</captured_meaning>
  <category_name>Narrative devices + structural form + craft</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies with a slow-burn pacing profile — tension accumulating gradually rather than in bursts.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>slow-burn movies</query_text>
  <description>Films with slow-burn pacing — gradual accumulation of tension.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the slow-burn pacing profile — tension that accumulates gradually as the viewer experiences the film.",
      "endpoint_coverage": "Keyword has no registry member whose definition names a pacing profile. The NarrativeStructureTag family covers structural devices, not pacing-as-feel. Semantic's viewer_experience.tension_adrenaline is the purpose-built sub-field for the tension shape the viewer feels, with sensory_load carrying the atmospheric texture that often accompanies slow-burn pacing.",
      "best_endpoint": "semantic",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "The tier-1 bias would pick Keyword, but no registry member names a pacing profile — 'slow burn' is experiential, not a structural device. Semantic wins decisively on its own merits: viewer_experience.tension_adrenaline exists to carry the tension-shape the viewer feels, with negations available to distinguish slow-burn from its high-adrenaline opposite. narrative_techniques is a plausible alternative home (pacing as a craft choice), but the 'slow burn' framing is felt rather than structural — viewer_experience is the stronger sub-space here.",
  "endpoint_to_run": "semantic",
  "endpoint_parameters": {
    "match_mode": "trait",
    "parameters": {
      "qualifier_inventory": "slow-burn pacing as the headline quality; tension accumulating gradually rather than in bursts; atmospheric patience as part of the felt experience.",
      "space_queries": [
        {
          "carries_qualifiers": "viewer_experience carries the slow-burn profile as the viewer feels it — tension_adrenaline captures the gradual-accumulation tension shape, and sensory_load captures the patient, atmospheric texture that typically accompanies it.",
          "space": "viewer_experience",
          "weight": "central",
          "content": {
            "emotional_palette": {"terms": [], "negations": []},
            "tension_adrenaline": {"terms": ["slow-burn tension", "gradual accumulation", "simmering dread"], "negations": ["frenetic pacing", "high-adrenaline bursts"]},
            "tone_self_seriousness": {"terms": [], "negations": []},
            "cognitive_complexity": {"terms": [], "negations": []},
            "disturbance_profile": {"terms": [], "negations": []},
            "sensory_load": {"terms": ["atmospheric patience", "deliberate pacing"], "negations": []},
            "emotional_volatility": {"terms": [], "negations": []},
            "ending_aftertaste": {"terms": [], "negations": []}
          }
        }
      ],
      "primary_vector": "viewer_experience"
    },
    "polarity": "positive"
  }
}
```

**Example: too-vague ask → no-fire**

```xml
<raw_query>movies that are well-told</raw_query>
<overall_query_intention_exploration>The user's phrasing names no specific device, structural form, or craft axis — "well-told" is a vague positive gesture at storytelling quality without any identifiable craft target.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films with an unspecified sense of good storytelling.</captured_meaning>
  <category_name>Narrative devices + structural form + craft</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies that feel well-told.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies that are well-told</query_text>
  <description>Films with a general sense of good storytelling craft.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the specific narrative device, structural form, or craft axis the user is asking for.",
      "endpoint_coverage": "Neither candidate can fit. Keyword requires a specific registry member whose definition names the device — the phrase names no device, structural form, or cast configuration. Semantic's narrative_techniques and viewer_experience sub-fields require concrete craft or experiential labels — 'well-told' supplies none, and inventing placeholder terms would dilute the query vector rather than target matching movies. Broad storytelling-quality gestures without a named aspect belong to reception-as-scalar, not this category.",
      "best_endpoint": "None",
      "best_endpoint_gaps": "The requirement names no identifiable craft axis. Firing either endpoint would fabricate content the input does not support."
    }
  ],
  "performance_vs_bias_analysis": "The tier-1 bias does not force a pick when no device or craft axis is actually named. Picking any NarrativeStructureTag would invent a device the user did not specify, and populating narrative_techniques with placeholder terms would pollute the semantic channel. No-fire is the correct response.",
  "endpoint_to_run": "None",
  "endpoint_parameters": null
}
```
