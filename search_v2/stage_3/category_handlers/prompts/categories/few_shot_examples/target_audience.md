# Examples

These calibrate the per-endpoint breakdown for target-audience asks: METADATA fires only as a maturity ceiling with negative polarity, KEYWORD fires only when a registry member names the audience packaging, SEMANTIC fires only on a situational viewing framing, and the empty combination is the correct response when the atom is really a story-arc or content-axis ask from a nearby category.

**Example: some-fire-some-don't — bare "family movie" fires META + KEYWORD, Semantic stays silent**

```xml
<raw_query>a good family movie</raw_query>
<overall_query_intention_exploration>The user wants a film packaged for family viewing — appropriate for kids and adults together. A bare audience-framing ask with no viewing-occasion qualifier.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>A movie packaged as a family film.</captured_meaning>
  <category_name>Target audience</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies packaged for family viewing.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>a good family movie</query_text>
  <description>A film packaged for family viewing.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Restrict to the maturity ceiling a family film implies — exclude R and NC-17.",
      "endpoint_coverage": "METADATA's maturity_rating column carries the rating axis directly. Keyword and Semantic cannot encode a rating ceiling."
    },
    {
      "aspect_description": "Identify the audience framing — a film packaged for family viewing.",
      "endpoint_coverage": "KEYWORD has a registry member (FAMILY) whose definition names exactly this packaging — made to be appropriate and enjoyable for both children and adults together. METADATA cannot express packaging beyond the rating. SEMANTIC's watch_context could carry a generic family scenario but the user named no specific viewing occasion."
    }
  ],
  "overall_endpoint_fits": "METADATA enforces the maturity ceiling implied by 'family' via filter+negative polarity on R and above. KEYWORD contributes the FAMILY packaging tag additively. SEMANTIC does not fire: the user named no viewing occasion or watch-partner, and watch_scenarios exists to carry situational framings, not the packaging label itself — populating it from a bare audience noun would be fabrication.",
  "per_endpoint_breakdown": {
    "keyword": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "filter",
        "parameters": {
          "concept_analysis": "The query names 'family movie' — a film packaged for family viewing, appropriate and enjoyable for both children and adults together.",
          "candidate_shortlist": "FAMILY definition names exactly this packaging. HOLIDAY_FAMILY is narrower (seasonal variant) with no seasonal cue in the input. ADULT_ANIMATION targets the opposite audience. FAMILY wins on its own merits.",
          "classification": "FAMILY"
        },
        "polarity": "positive"
      }
    },
    "metadata": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "filter",
        "parameters": {
          "constraint_phrases": ["family movie"],
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

**Example: situational ask — META + SEMANTIC fire, KEYWORD holds**

```xml
<raw_query>something to watch with my grandparents on Sunday</raw_query>
<overall_query_intention_exploration>The user wants a film appropriate for a cross-generational Sunday viewing with elderly family members. The ask is situational — specific watch partners and an occasion — and implies a gentle maturity ceiling even though no rating is named.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>A movie suitable for watching with grandparents on a Sunday.</captured_meaning>
  <category_name>Target audience</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>A film appropriate for watching with grandparents on a Sunday afternoon.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>something to watch with my grandparents on Sunday</query_text>
  <description>A film appropriate for watching with grandparents on a Sunday.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Restrict to the maturity ceiling cross-generational viewing with grandparents implies — exclude R and NC-17.",
      "endpoint_coverage": "METADATA's maturity_rating column carries the ceiling axis directly via filter+negative polarity."
    },
    {
      "aspect_description": "Capture the viewing occasion — Sunday afternoon watch with grandparents.",
      "endpoint_coverage": "SEMANTIC's watch_context.watch_scenarios is the purpose-built sub-field for occasion + watch-partner framings. Neither KEYWORD nor METADATA can encode a specific viewing occasion — the registry has no 'watch with grandparents' member."
    }
  ],
  "overall_endpoint_fits": "METADATA enforces the maturity ceiling implied by the cross-generational framing. SEMANTIC captures the situational framing through watch_context — specifically watch_scenarios for the Sunday viewing with grandparents and self_experience_motivations for the shared-comfort pull. KEYWORD does not fire: no registry member's definition names 'grandparent viewing' or a grown-but-gentle audience packaging, and picking FAMILY would misroute this to a kid-inclusive packaging the user did not ask for.",
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
          "constraint_phrases": ["watch with my grandparents"],
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
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "filter",
        "parameters": {
          "qualifier_inventory": "viewing occasion is Sunday afternoon; watch partners are grandparents; implied pull is shared, gentle, cross-generational comfort viewing.",
          "space_queries": [
            {
              "carries_qualifiers": "watch_context carries the occasion and watch-partner framing — a Sunday watch with grandparents as the viewing scenario, with the shared comfort motivation the partner framing implies.",
              "space": "watch_context",
              "weight": "central",
              "content": {
                "self_experience_motivations": {"terms": ["shared comfort", "cross-generational viewing"]},
                "external_motivations": {"terms": ["watching with grandparents", "watching with older family"]},
                "key_movie_feature_draws": {"terms": []},
                "watch_scenarios": {"terms": ["Sunday afternoon", "watch with grandparents", "family gathering"]}
              }
            }
          ],
          "primary_vector": "watch_context"
        },
        "polarity": "positive"
      }
    }
  }
}
```

**Example: single-endpoint KEYWORD fire — "for adults only" with no ceiling and no occasion**

```xml
<raw_query>adult animated movies</raw_query>
<overall_query_intention_exploration>The user wants animation packaged for adult viewers — mature themes, humor, or content aimed at grown-up audiences rather than children. A named audience packaging with no viewing occasion and no maturity-ceiling implication.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Animation packaged for an adult audience.</captured_meaning>
  <category_name>Target audience</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Animation packaged for adult viewers.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>adult animated movies</query_text>
  <description>Animation packaged for adult viewers.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the audience framing — animation packaged for adult viewers, not family or kids.",
      "endpoint_coverage": "KEYWORD has a registry member (ADULT_ANIMATION) whose definition names exactly this packaging — animation aimed mainly at adults, with mature themes, humor, violence, or sexual content. METADATA has no 'adult-packaged' attribute — the maturity column encodes ratings, and the framing here does not imply a ceiling (adult animation can be anywhere from PG-13 to NC-17). SEMANTIC's watch_context carries occasion/partner framings, which the user did not name."
    }
  ],
  "overall_endpoint_fits": "KEYWORD carries the full requirement via the ADULT_ANIMATION registry member. METADATA does not fire: the packaging does not imply a single rating ceiling — inventing one would narrow the pool past the user's intent. SEMANTIC does not fire: no viewing occasion or watch-partner was named.",
  "per_endpoint_breakdown": {
    "keyword": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "filter",
        "parameters": {
          "concept_analysis": "The query names 'adult animated movies' — animation packaged for adult viewers. A named audience-packaging ask.",
          "candidate_shortlist": "ADULT_ANIMATION definition names exactly this packaging — animation aimed at adults. FAMILY is the opposite packaging. No other registry member names adult-aimed animation specifically.",
          "classification": "ADULT_ANIMATION"
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

**Example: situational kid-watch — META + SEMANTIC fire, KEYWORD stays silent**

```xml
<raw_query>something to put on with the kids tonight</raw_query>
<overall_query_intention_exploration>The user wants a film appropriate for watching with their children this evening. The ask is situational — watch partners named (the kids), occasion named (tonight) — and implies a firm maturity ceiling.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>A movie appropriate for a tonight watch with the user's children.</captured_meaning>
  <category_name>Target audience</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>A film to watch with the kids this evening.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>something to put on with the kids tonight</query_text>
  <description>A film to watch with children this evening.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Restrict to the maturity ceiling a kid-inclusive watch implies — exclude PG-13 and above for younger children.",
      "endpoint_coverage": "METADATA's maturity_rating column carries the ceiling axis via filter+negative polarity. The kid-inclusive framing implies a tighter ceiling than a generic family ask."
    },
    {
      "aspect_description": "Capture the viewing occasion — tonight, watching with the kids.",
      "endpoint_coverage": "SEMANTIC's watch_context.watch_scenarios is the purpose-built sub-field for watch-partner and occasion framings. The scenario 'watch with kids on a weeknight' lands here."
    },
    {
      "aspect_description": "Identify the audience packaging label, if any matches.",
      "endpoint_coverage": "KEYWORD has FAMILY, but its definition names films made for children AND adults together — a packaging label. The user did not name the packaging; they named a viewing scenario. Firing FAMILY from a situational phrase risks broadening past the specific watch-with-kids framing SEMANTIC captures more precisely."
    }
  ],
  "overall_endpoint_fits": "METADATA enforces the kid-inclusive ceiling via filter+negative polarity. SEMANTIC captures the situational framing through watch_context's watch_scenarios and external_motivations. KEYWORD does not fire: the query names a scenario, not a packaging label — the semantic channel handles the situational ask more precisely, and adding FAMILY would double-count without carrying distinct signal.",
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
          "constraint_phrases": ["with the kids"],
          "target_attribute": "maturity_rating",
          "value_intent_label": "exclude PG-13 and above",
          "maturity_rating": {
            "rating": "pg-13",
            "match_operation": "greater_than_or_equal"
          }
        },
        "polarity": "negative"
      }
    },
    "semantic": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "filter",
        "parameters": {
          "qualifier_inventory": "viewing occasion is tonight / a weeknight evening; watch partners are the user's children; implied pull is a shared parent-with-kids watch.",
          "space_queries": [
            {
              "carries_qualifiers": "watch_context carries the occasion (tonight, weeknight evening) and watch-partner framing (with the kids) as the viewing scenario.",
              "space": "watch_context",
              "weight": "central",
              "content": {
                "self_experience_motivations": {"terms": ["shared family viewing"]},
                "external_motivations": {"terms": ["watching with the kids", "parent and child viewing"]},
                "key_movie_feature_draws": {"terms": []},
                "watch_scenarios": {"terms": ["watch with kids", "weeknight family viewing", "evening with children"]}
              }
            }
          ],
          "primary_vector": "watch_context"
        },
        "polarity": "positive"
      }
    }
  }
}
```

**Example: no-fire — coming-of-age is a story archetype, not audience framing**

```xml
<raw_query>a good coming-of-age story</raw_query>
<overall_query_intention_exploration>The user wants a coming-of-age story — a narrative arc about the protagonist's transition from adolescence into adulthood. This frames the story's arc pattern, not the audience the film is pitched to.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>A film with a coming-of-age story arc.</captured_meaning>
  <category_name>Target audience</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies with a coming-of-age story arc.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>a good coming-of-age story</query_text>
  <description>A film with a coming-of-age story arc.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the audience packaging or viewing occasion the user named.",
      "endpoint_coverage": "None of the three endpoints fits. The phrase names a story-arc pattern (the protagonist's transition from adolescence into adulthood), not an audience packaging, a maturity ceiling, or a viewing occasion. KEYWORD has no audience-packaging member whose definition covers 'coming-of-age' — the concept is a story archetype routed by Cat 21, not an audience label. METADATA has no attribute for story arcs. SEMANTIC's watch_context expects occasion or watch-partner framings, not arc patterns — populating it from a story-arc phrase would fabricate a scenario the input does not support."
    }
  ],
  "overall_endpoint_fits": "The atom routed here is a story archetype — Cat 21's territory — not an audience framing. No endpoint in this category's set carries a signal the input actually supports. Firing any of them would invent content: a packaging label the user did not name, a rating ceiling the arc does not imply, or a viewing occasion the query does not describe. The empty combination is the correct response.",
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
