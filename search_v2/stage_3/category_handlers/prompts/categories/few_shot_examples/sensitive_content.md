# Examples

These calibrate the full `match_mode × polarity` 2×2 for sensitive-content asks: KEYWORD for binary content-flag presence/absence, METADATA for rating ceilings as content proxies, SEMANTIC for gradient intensity, and the empty combination when the input names no concrete content axis.

**Example: hard exclusion — "no animal death" fires KEYWORD filter+negative, others hold**

```xml
<raw_query>a good adventure movie but nothing where the dog dies</raw_query>
<overall_query_intention_exploration>The user wants an adventure movie but has one hard content dealbreaker — they do not want to watch a movie in which an animal dies on screen. The content axis is explicit and binary.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The user wants to avoid movies where an animal dies.</captured_meaning>
  <category_name>Sensitive content</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Avoid movies that depict on-screen animal death.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>nothing where the dog dies</query_text>
  <description>A hard exclusion on movies depicting on-screen animal death.</description>
  <modifiers>
    <modifier>
      <original_text>nothing</original_text>
      <effect>negates the following clause — rules the described content out entirely</effect>
      <type>POLARITY_MODIFIER</type>
    </modifier>
  </modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>a good adventure movie</query_text>
    <description>An adventure film.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Hard-exclude movies tagged with on-screen animal death — the user framed this as a binary dealbreaker, not a gradient preference.",
      "endpoint_coverage": "KEYWORD has a registry member (ANIMAL_DEATH) whose definition names exactly this content flag — an animal dies on screen or as a significant plot point. METADATA has no structured column for on-screen animal death; the maturity_rating column would excise far more than the user asked for. SEMANTIC's disturbance_profile could carry the atom as a gradient, but the user framed it as a binary rule-out, and KEYWORD covers the binary case with a clean registry hit."
    }
  ],
  "overall_endpoint_fits": "KEYWORD carries the full requirement via the ANIMAL_DEATH content flag with filter+negative polarity — a binary hard exclusion. METADATA does not fire: no rating ceiling was implied, and excising R-rated films would over-broaden. SEMANTIC does not fire: the ask is binary, not gradient, and duplicating the signal on disturbance_profile would risk softening a hard rule-out.",
  "per_endpoint_breakdown": {
    "keyword": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "filter",
        "parameters": {
          "concept_analysis": "The query names 'the dog dies' under a negation modifier — a binary content flag for on-screen animal death, framed as a dealbreaker.",
          "candidate_shortlist": "ANIMAL_DEATH definition names exactly this flag — an animal dies on screen or as a significant plot point. No other registry member names on-screen animal death specifically; SPLATTER_HORROR targets gore as a genre, not this axis.",
          "classification": "ANIMAL_DEATH"
        },
        "polarity": "negative"
      }
    },
    "metadata": {
      "should_run_endpoint": false,
      "endpoint_parameters": null
    },
    "semantic": {
      "should_run_endpoint": false,
      "endpoint_parameters": null
    }
  }
}
```

**Example: gradient preference — "not too bloody" fires SEMANTIC trait+negative, others hold**

```xml
<raw_query>a horror movie that's not too bloody</raw_query>
<overall_query_intention_exploration>The user wants a horror movie with a softened gore profile — they are not hard-excluding gore entirely, just asking the results to lean away from heavily bloody / graphic titles within the genre.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>A preference for movies that are not heavy on blood and gore.</captured_meaning>
  <category_name>Sensitive content</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies with heavy bloodshed / gore intensity should score lower without being ruled out entirely.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>not too bloody</query_text>
  <description>A gradient preference against heavy bloodshed / gore.</description>
  <modifiers>
    <modifier>
      <original_text>not too</original_text>
      <effect>modulates strength downward — softens rather than rules out entirely</effect>
      <type>POLARITY_MODIFIER</type>
    </modifier>
  </modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>a horror movie</query_text>
    <description>A horror film.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Movies with heavy bloodshed / gore intensity should score lower — the user modulated the ask with 'not too', which is a gradient preference rather than a binary exclusion.",
      "endpoint_coverage": "SEMANTIC's viewer_experience.disturbance_profile is the purpose-built sub-field for content-intensity gradients — it carries the 'how bloody' axis as terms and negations. KEYWORD would need a binary gore flag and only SPLATTER_HORROR comes close (genre-narrow, not an axis). METADATA's maturity_rating would over-broaden — the user named gore specifically, not overall rating."
    }
  ],
  "overall_endpoint_fits": "SEMANTIC carries the full requirement as a trait+negative on viewer_experience.disturbance_profile — it captures the 'too bloody' axis as a gradient where higher intensity scores lower without being ruled out, which matches the framing exactly. KEYWORD does not fire: no binary content-flag member maps to 'gore' as an axis, and SPLATTER_HORROR would rule out every non-splatter horror film rather than soften the blood-intensity preference. METADATA does not fire: the user named a specific content axis, not a rating ceiling.",
  "per_endpoint_breakdown": {
    "keyword": {
      "should_run_endpoint": false,
      "endpoint_parameters": null
    },
    "metadata": {
      "should_run_endpoint": false,
      "endpoint_parameters": null
    },
    "semantic": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "trait",
        "parameters": {
          "qualifier_inventory": "content axis is gore / bloodshed; intensity framing is 'not too' — a gradient dial-down rather than a hard rule-out; no other dimensions named.",
          "space_queries": [
            {
              "carries_qualifiers": "viewer_experience carries the content-intensity axis — disturbance_profile describes gore / bloodshed level. terms name what the disturbance IS (the axis the user is concerned about); negations name the intensity boundary the user is drawing.",
              "space": "viewer_experience",
              "weight": "central",
              "content": {
                "disturbance_profile": {
                  "terms": ["bloodshed", "gore", "graphic violence"],
                  "negations": ["heavy gore", "splatter", "excessive blood"]
                }
              }
            }
          ],
          "primary_vector": "viewer_experience"
        },
        "polarity": "negative"
      }
    }
  }
}
```

**Example: rating ceiling — "family-friendly intensity" fires METADATA filter+negative**

```xml
<raw_query>an action movie with family-friendly intensity</raw_query>
<overall_query_intention_exploration>The user wants an action movie whose overall content intensity stays in family-friendly territory — a rating-ceiling proxy for content that never reaches R-rated violence, language, or sexual content.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>A content-intensity ceiling consistent with family-friendly packaging.</captured_meaning>
  <category_name>Sensitive content</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Restrict to movies whose overall content intensity is family-friendly — exclude R and above.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>family-friendly intensity</query_text>
  <description>A rating-ceiling implication — overall content intensity should stay family-friendly.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>an action movie</query_text>
    <description>An action film.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Restrict to the maturity ceiling 'family-friendly intensity' implies — exclude R and NC-17 as a hard rule-out on overall content intensity.",
      "endpoint_coverage": "METADATA's maturity_rating column carries the rating axis directly via filter+negative polarity. KEYWORD has no axis-agnostic 'family-friendly intensity' flag — FAMILY names the packaging, not the intensity ceiling, and would belong on Cat 17. SEMANTIC's disturbance_profile carries specific content axes, not a global rating ceiling across all axes."
    }
  ],
  "overall_endpoint_fits": "METADATA carries the requirement via filter+negative on maturity_rating at the R threshold — the phrase 'family-friendly intensity' frames the rating as a content-intensity proxy, which is the ceiling case this category's METADATA slot exists for. KEYWORD does not fire: FAMILY is an audience-packaging label (Cat 17's territory), not an intensity axis, and no other registry member names a global intensity ceiling. SEMANTIC does not fire: the user did not name a specific content axis (gore, language, sexual content) — they named a global ceiling that disturbance_profile cannot express without over-specifying an axis the input does not.",
  "per_endpoint_breakdown": {
    "keyword": {
      "should_run_endpoint": false,
      "endpoint_parameters": null
    },
    "metadata": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "filter",
        "parameters": {
          "constraint_phrases": ["family-friendly intensity"],
          "target_attribute": "maturity_rating",
          "value_intent_label": "exclude R and above",
          "maturity_rating": {
            "rating": "r",
            "match_operation": "greater_than_or_equal"
          }
        },
        "polarity": "negative"
      }
    },
    "semantic": {
      "should_run_endpoint": false,
      "endpoint_parameters": null
    }
  }
}
```

**Example: positive inclusion — "where the dog actually dies" fires KEYWORD filter+positive**

```xml
<raw_query>give me a sad movie where the dog actually dies</raw_query>
<overall_query_intention_exploration>The user wants a sad movie whose emotional payoff specifically involves the on-screen death of a dog or other pet. The content flag is named as a positive inclusion, not a dealbreaker.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The user wants a movie in which an animal dies on screen.</captured_meaning>
  <category_name>Sensitive content</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Include movies that depict on-screen animal death as a significant plot element.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>where the dog actually dies</query_text>
  <description>A binary content-flag inclusion — movies whose plot involves the on-screen death of an animal.</description>
  <modifiers>
    <modifier>
      <original_text>actually</original_text>
      <effect>strengthens the content-flag assertion — the user specifically wants this to occur</effect>
      <type>POLARITY_MODIFIER</type>
    </modifier>
  </modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>give me a sad movie</query_text>
    <description>A sad film.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Hard-include movies tagged with on-screen animal death — the user framed this as a required content element, not a gradient preference.",
      "endpoint_coverage": "KEYWORD has a registry member (ANIMAL_DEATH) whose definition names exactly this content flag and can be emitted as filter+positive for the inclusion framing. METADATA has no structured column for animal death. SEMANTIC's disturbance_profile could gesture at the intensity dimension, but the user asked for a binary presence of a specific plot element — KEYWORD delivers that precisely where SEMANTIC would soften it into a similarity score."
    }
  ],
  "overall_endpoint_fits": "KEYWORD carries the full requirement via the ANIMAL_DEATH content flag with filter+positive polarity. METADATA does not fire: no rating axis was named. SEMANTIC does not fire: the ask is binary inclusion of a specific plot element, not a gradient intensity — populating disturbance_profile would dilute the hit on the concrete content flag.",
  "per_endpoint_breakdown": {
    "keyword": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "filter",
        "parameters": {
          "concept_analysis": "The query names 'the dog actually dies' as a required plot element — a binary content flag for on-screen animal death, framed as a positive inclusion.",
          "candidate_shortlist": "ANIMAL_DEATH definition names exactly this flag. TEARJERKER names the emotional effect (Cat 21/22 territory) rather than the specific content axis. ANIMAL_DEATH wins on axis-specificity — the user named the event, not the feeling.",
          "classification": "ANIMAL_DEATH"
        },
        "polarity": "positive"
      }
    },
    "metadata": {
      "should_run_endpoint": false,
      "endpoint_parameters": null
    },
    "semantic": {
      "should_run_endpoint": false,
      "endpoint_parameters": null
    }
  }
}
```

**Example: no-fire — "nothing heavy" names no concrete content axis**

```xml
<raw_query>something fun, nothing heavy</raw_query>
<overall_query_intention_exploration>The user wants a fun / light watch and is signaling away from heaviness — but they named no specific content axis (no violence, gore, nudity, language, or sexual content specifically), and no rating ceiling.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The user does not want a heavy movie.</captured_meaning>
  <category_name>Sensitive content</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Avoid content-heavy movies.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>nothing heavy</query_text>
  <description>A preference away from heaviness, axis unspecified.</description>
  <modifiers>
    <modifier>
      <original_text>nothing</original_text>
      <effect>negates the following clause</effect>
      <type>POLARITY_MODIFIER</type>
    </modifier>
  </modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>something fun</query_text>
    <description>A fun / light film.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the content axis or rating ceiling the user named.",
      "endpoint_coverage": "No endpoint in this category's set fits. KEYWORD has no 'not heavy' content-flag member — 'heavy' is a tonal / emotional register, not a content axis with a registry entry. METADATA cannot infer a specific rating ceiling from 'heavy' alone — R-rated does not equal heavy, and nothing in the phrasing points at a rating boundary. SEMANTIC's disturbance_profile expects a concrete content axis (gore, violence, sexual content, disturbance) — 'heavy' is closer to emotional tone / weight, which is viewer_experience's emotional_palette or tone_self_seriousness — Cat 22's territory, not this category's slot."
    }
  ],
  "overall_endpoint_fits": "The atom routed here is a tonal / emotional 'heaviness' preference, not a specific content axis or rating ceiling. Cat 22 (viewer experience / feel / tone) carries tonal weight via viewer_experience's tone and emotional_palette sub-fields — not this category. No endpoint in this category's set carries a signal the input actually supports; firing any of them would invent content (an axis the user did not name, a ceiling the user did not imply, a disturbance the user did not describe). The empty combination is correct.",
  "per_endpoint_breakdown": {
    "keyword": {
      "should_run_endpoint": false,
      "endpoint_parameters": null
    },
    "metadata": {
      "should_run_endpoint": false,
      "endpoint_parameters": null
    },
    "semantic": {
      "should_run_endpoint": false,
      "endpoint_parameters": null
    }
  }
}
```
