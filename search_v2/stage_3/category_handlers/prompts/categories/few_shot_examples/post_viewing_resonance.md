# Examples

These calibrate the Keyword-vs-Semantic decision for post-viewing resonance: clean keyword wins when the user names a structural ending type the registry covers, lower-tier Semantic wins when the ask is experiential aftertaste no structural tag encodes, and the no-fire shape when the atom is really a during-viewing feel belonging to Cat 22.

**Example: clean structural ending → keyword wins**

```xml
<raw_query>movies with a happy ending</raw_query>
<overall_query_intention_exploration>The user wants films that end on a positive resolution — things work out for the protagonists. A canonical structural ending the registry carries as HAPPY_ENDING.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films that end on a positive, upbeat resolution.</captured_meaning>
  <category_name>Post-viewing resonance</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies with a happy ending.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies with a happy ending</query_text>
  <description>Films whose ending is predominantly positive.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the ending shape — a positive, upbeat resolution where things work out for the protagonists.",
      "endpoint_coverage": "Keyword has the HAPPY_ENDING registry member, whose definition names exactly this structural shape — the overall resolution is positive or optimistic. Semantic's viewer_experience.ending_aftertaste could hold 'uplifting' terms, but that is strictly weaker when a definitional tag covers the structural ending.",
      "best_endpoint": "keyword",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "Keyword wins on its own merits — the HAPPY_ENDING tag definition covers the structural ending exactly, so the authoritative posting list delivers the right set. The tier-1 bias and the on-merits choice agree; the bias did not have to break a close call.",
  "endpoint_to_run": "keyword",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The query names 'happy ending' — a canonical structural ending where the overall resolution is positive for the protagonists. A named ending shape, not an experiential aftertaste descriptor.",
      "candidate_shortlist": "HAPPY_ENDING definition names exactly this shape. BITTERSWEET_ENDING is adjacent but narrower — it mixes positive and negative elements, which contradicts the user's ask for a straightforwardly positive resolution. FEEL_GOOD is a viewer-response framing about the overall film leaving you uplifted, not specifically about the ending's structural shape.",
      "classification": "happy_ending"
    },
    "polarity": "positive"
  }
}
```

**Example: clean structural twist ending → keyword wins**

```xml
<raw_query>films with a twist ending</raw_query>
<overall_query_intention_exploration>The user wants films whose ending pivots on a late reveal that reframes what came before — a canonical structural device the registry carries as PLOT_TWIST.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films whose ending turns on a twist reveal.</captured_meaning>
  <category_name>Post-viewing resonance</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies with a twist ending.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>films with a twist ending</query_text>
  <description>Films whose ending delivers an unexpected twist reveal.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the ending shape — a late-story reveal that reframes the preceding narrative.",
      "endpoint_coverage": "Keyword has the PLOT_TWIST registry member, whose definition names exactly this device — a significant reveal that recontextualizes part or all of the story. No dedicated TWIST_ENDING tag exists in the registry; PLOT_TWIST is the canonical home whether the twist lands mid-story or at the ending. Semantic's viewer_experience.ending_aftertaste could hold 'shocking reveal' terms, but that is weaker than a definitional tag match.",
      "best_endpoint": "keyword",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "Keyword wins on its own merits — the PLOT_TWIST tag definition covers twist-driven endings exactly. The tier-1 bias and the on-merits choice agree; the bias did not have to break a close call.",
  "endpoint_to_run": "keyword",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The query names 'twist ending' — the canonical surprise-reveal device, framed as the ending's defining element. A named structural shape, not an experiential aftertaste.",
      "candidate_shortlist": "PLOT_TWIST definition names exactly this mechanic; the registry has no TWIST_ENDING member, so PLOT_TWIST is the authoritative home for twist-driven endings. TWIST_VILLAIN is adjacent but narrower — it names specifically the antagonist-identity reveal, which the user did not scope to. OPEN_ENDING names ambiguous non-resolution rather than a reframing reveal.",
      "classification": "plot_twist"
    },
    "polarity": "positive"
  }
}
```

**Example: lower-tier-wins — experiential aftertaste with no structural tag → semantic**

```xml
<raw_query>haunting films that stay with you</raw_query>
<overall_query_intention_exploration>The user wants films that leave a lasting residue after the credits — haunting, lingering, the kind you keep thinking about for days. An experiential aftertaste ask with no structural ending type attached.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films that leave a haunting, lingering aftertaste on the viewer after watching.</captured_meaning>
  <category_name>Post-viewing resonance</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies that haunt the viewer and stay with them for days after watching.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>haunting films that stay with you</query_text>
  <description>Films whose post-viewing residue is haunting and long-lingering.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the post-viewing aftertaste — a haunting, lingering residue that persists long after the credits.",
      "endpoint_coverage": "Keyword has no registry member that encodes experiential aftertaste. The ending-tag family (HAPPY_ENDING, SAD_ENDING, BITTERSWEET_ENDING) and structural tags (OPEN_ENDING, CLIFFHANGER_ENDING, PLOT_TWIST) name ending shapes — a haunting film can have any of these shapes, so no structural tag is a real fit. Semantic's viewer_experience.ending_aftertaste is the purpose-built sub-field for the residue the viewer carries away, which is exactly what the user asked for.",
      "best_endpoint": "semantic",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "No structural ending tag covers experiential aftertaste — 'haunting' and 'stays with you' describe the residue a film leaves on the viewer, which is independent of the ending's literal shape. The tier-1 Keyword bias is inapplicable here rather than overridden: there is no definitional tag for the bias to point at. Semantic wins decisively on its own merits, with viewer_experience.ending_aftertaste as the purpose-built sub-field for this dimension.",
  "endpoint_to_run": "semantic",
  "endpoint_parameters": {
    "match_mode": "trait",
    "parameters": {
      "qualifier_inventory": "haunting aftertaste as the headline quality; lingering residue the viewer carries away; the kind of film the viewer keeps thinking about long after watching.",
      "space_queries": [
        {
          "carries_qualifiers": "viewer_experience carries the post-viewing residue directly — ending_aftertaste captures the haunting, lingering quality the viewer is left with, and emotional_palette captures the haunted tone that typically accompanies it.",
          "space": "viewer_experience",
          "weight": "central",
          "content": {
            "emotional_palette": {"terms": ["haunted", "melancholic"], "negations": []},
            "tension_adrenaline": {"terms": [], "negations": []},
            "tone_self_seriousness": {"terms": [], "negations": []},
            "cognitive_complexity": {"terms": [], "negations": []},
            "disturbance_profile": {"terms": [], "negations": []},
            "sensory_load": {"terms": [], "negations": []},
            "emotional_volatility": {"terms": [], "negations": []},
            "ending_aftertaste": {"terms": ["haunting", "lingering", "stays with you", "hard to shake", "keeps working on you for days"], "negations": ["forgettable"]}
          }
        }
      ],
      "primary_vector": "viewer_experience"
    },
    "polarity": "positive"
  }
}
```

**Example: gut-punch ending straddle — experiential framing wins → semantic**

```xml
<raw_query>movies with a gut-punch ending</raw_query>
<overall_query_intention_exploration>The user wants films whose ending lands like a blow — devastating, leaving the viewer reeling. The framing straddles a sad structural shape and the experiential aftertaste, but the emphasis is on the blow the ending delivers, not the structural fact of an unhappy resolution.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films whose ending lands as a devastating gut-punch — the viewer is left reeling.</captured_meaning>
  <category_name>Post-viewing resonance</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies with a gut-punch ending that devastates the viewer.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies with a gut-punch ending</query_text>
  <description>Films whose ending delivers a devastating emotional blow.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the gut-punch aftertaste — an ending that lands as an emotional blow and leaves the viewer reeling, not merely a structural sad resolution.",
      "endpoint_coverage": "Keyword's SAD_ENDING tag is a partial cover — a gut-punch ending is typically unhappy for the protagonists, but the tag only encodes the structural shape. It does not distinguish a quietly melancholy sad ending from a shattering gut-punch, which is the distinction the user actually asked for. Semantic's viewer_experience.ending_aftertaste is the purpose-built sub-field for the blow-lands-on-the-viewer dimension — devastating, shattering, knocks the wind out. That is where the load-bearing signal lives.",
      "best_endpoint": "semantic",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "The tier-1 bias would point at SAD_ENDING, but it only captures half the ask — the structural unhappiness without the experiential blow. 'Gut-punch' is load-bearing on the aftertaste side: the user wants films where the ending devastates the viewer, not merely films where the protagonists lose. Semantic on ending_aftertaste is clearly better on merits; the bias does not force a pick when the lower-tier endpoint fits the actual framing more faithfully.",
  "endpoint_to_run": "semantic",
  "endpoint_parameters": {
    "match_mode": "trait",
    "parameters": {
      "qualifier_inventory": "gut-punch ending as the headline framing; devastating emotional blow at the close; the viewer is left reeling rather than merely saddened.",
      "space_queries": [
        {
          "carries_qualifiers": "viewer_experience carries the gut-punch framing as the viewer experiences it — ending_aftertaste captures the devastating, shattering blow the ending lands, and emotional_palette captures the heavy devastation the aftertaste sits in.",
          "space": "viewer_experience",
          "weight": "central",
          "content": {
            "emotional_palette": {"terms": ["devastating", "heartbreaking"], "negations": []},
            "tension_adrenaline": {"terms": [], "negations": []},
            "tone_self_seriousness": {"terms": [], "negations": []},
            "cognitive_complexity": {"terms": [], "negations": []},
            "disturbance_profile": {"terms": [], "negations": []},
            "sensory_load": {"terms": [], "negations": []},
            "emotional_volatility": {"terms": [], "negations": []},
            "ending_aftertaste": {"terms": ["gut-punch ending", "devastating finale", "knocks the wind out", "leaves you reeling", "emotionally shattering"], "negations": ["tidy resolution", "uplifting closer"]}
          }
        }
      ],
      "primary_vector": "viewer_experience"
    },
    "polarity": "positive"
  }
}
```

**Example: during-viewing feel misrouted here → no-fire (Cat 22's territory)**

```xml
<raw_query>dark cerebral movies</raw_query>
<overall_query_intention_exploration>The user wants films with a dark, intellectually demanding tone during watching — the during-viewing feel, not a post-viewing residue or a specific ending type.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films with a dark tone and cerebral, intellectually demanding texture.</captured_meaning>
  <category_name>Post-viewing resonance</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies that feel dark and cerebral while watching.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>dark cerebral movies</query_text>
  <description>Films with a dark, intellectually demanding during-viewing tone.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the post-viewing resonance signal — a specific ending shape or a concrete aftertaste descriptor.",
      "endpoint_coverage": "Neither candidate in this category's set fits. 'Dark' is a during-viewing emotional-palette tone and 'cerebral' is a during-viewing cognitive-complexity texture — both live on Cat 22 (viewer experience) where tone_self_seriousness, emotional_palette, and cognitive_complexity are the purpose-built sub-fields. Keyword has no ending tag whose definition names 'dark' or 'cerebral'. Semantic's viewer_experience could technically hold those terms, but using ending_aftertaste to encode a during-viewing tone misrepresents the ask — the user did not frame the ending or the post-viewing residue specifically. Firing here would duplicate Cat 22's work with worse fidelity.",
      "best_endpoint": "None",
      "best_endpoint_gaps": "The requirement names no specific ending shape or aftertaste — it names during-viewing tone and cognitive texture, which belong to Cat 22."
    }
  ],
  "performance_vs_bias_analysis": "The tier-1 bias does not force a pick when no structural ending is named and the aftertaste framing is absent. The atom routed here is a during-viewing feel that belongs to Cat 22 — firing Keyword would invent a structural ending the user did not specify, and firing Semantic on ending_aftertaste would pull tone terms out of their native sub-fields. No-fire is the correct response.",
  "endpoint_to_run": "None",
  "endpoint_parameters": null
}
```
