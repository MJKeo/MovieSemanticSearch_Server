# Examples

These calibrate the Keyword-vs-Semantic decision for format and visual-format asks: clean Keyword wins when a registry member names the form (format or visual-format alike), lower-tier Semantic wins when the requirement names a technique the closed vocabulary does not carry, and the no-fire shape when the atom is too vague to target either channel.

**Example: clean keyword win on a format flag**

```xml
<raw_query>documentaries about the climate crisis</raw_query>
<overall_query_intention_exploration>The user wants nonfiction films — the documentary form — on the subject of the climate crisis. Two atoms: the format (documentary) and the subject (climate crisis). This fragment is the format slice.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films in the documentary format.</captured_meaning>
  <category_name>Format + visual-format specifics</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies in the documentary format.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>documentaries about the climate crisis</query_text>
  <description>Nonfiction films examining the climate crisis.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>about the climate crisis</query_text>
    <description>Subject: the climate crisis — a Cat 6 specific-subject atom handled separately.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the documentary format — nonfiction films built around real people, events, or issues.",
      "endpoint_coverage": "Keyword has the DOCUMENTARY registry member, whose definition names nonfiction films intended to inform, examine, or record reality — a direct cover for the format ask. Semantic's production space could carry documentary-style craft phrases, but the format itself is a near-definitional classification that movies are tagged with at ingest, so the authoritative posting list is decisively stronger here.",
      "best_endpoint": "keyword",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "Keyword wins on its own merits — the DOCUMENTARY tag's definition covers the format exactly, and every matching movie carries the tag at ingest time. Running Semantic in parallel against 'documentary' prose would only invite near-miss dilution. The tier-1 bias and the on-merits choice agree; this is not a close call that the bias had to break.",
  "endpoint_to_run": "keyword",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The atom asks for the documentary format — nonfiction films about real people, events, or issues. The subject qualifier (climate crisis) is a sibling atom and not this category's concern.",
      "candidate_shortlist": "DOCUMENTARY definition names nonfiction filmmaking as a form — a direct match. MOCKUMENTARY is the inverse (fictional content in documentary style) and would miss the set. Specific documentary sub-forms (HISTORY_DOCUMENTARY, NATURE_DOCUMENTARY, etc.) would silently narrow the set past the user's ask — the climate-crisis subject is already carried by the sibling atom, not by picking a narrow sub-tag here.",
      "classification": "documentary"
    },
    "polarity": "positive"
  }
}
```

**Example: clean keyword win on a visual-format flag**

```xml
<raw_query>stop-motion animated films</raw_query>
<overall_query_intention_exploration>The user wants films rendered through stop-motion animation — the physical-frame-by-frame visual-form. A named visual-format classification.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films made using stop-motion animation.</captured_meaning>
  <category_name>Format + visual-format specifics</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies made with stop-motion animation.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>stop-motion animated films</query_text>
  <description>Films rendered through stop-motion animation.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the stop-motion animation visual-form — films made by photographing real objects or puppets frame by frame.",
      "endpoint_coverage": "Keyword has STOP_MOTION_ANIMATION, whose definition names movies made by moving real objects or puppets in small increments and photographing them frame by frame — a direct cover for the visual-form ask. Semantic's production.production_techniques could carry 'stop-motion' as a craft phrase, but the registry tag is the authoritative classification that every stop-motion film carries at ingest.",
      "best_endpoint": "keyword",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "Keyword wins on its own merits — STOP_MOTION_ANIMATION's definition names the visual-form exactly, and the tag is stamped on ingest for every film in this form. The tier-1 bias reinforces a choice that would stand on its own.",
  "endpoint_to_run": "keyword",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The atom names stop-motion animation — a specific physical rendering technique that produces a distinctive visual-form. A named visual-format classification, not a general animation ask.",
      "candidate_shortlist": "STOP_MOTION_ANIMATION definition names the form exactly. ANIMATION is the parent family — picking it would broaden the set to every animated film and drop the stop-motion signal. COMPUTER_ANIMATION and HAND_DRAWN_ANIMATION are sibling techniques that would misclassify the target set.",
      "classification": "stop_motion_animation"
    },
    "polarity": "positive"
  }
}
```

**Example: lower-tier-wins — visual technique with no canonical tag → semantic**

```xml
<raw_query>movies shot in black and white with heavy practical creature effects and no CGI</raw_query>
<overall_query_intention_exploration>The user wants films that combine black-and-white photography with practical creature effects and an explicit absence of CGI. These are craft-level production techniques — the registry has no member for either. Routing to the format/visual category because the ask is entirely about the physical form the film takes.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films shot in black and white using practical creature effects rather than CGI.</captured_meaning>
  <category_name>Format + visual-format specifics</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies shot in black and white with practical creature effects and no CGI.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies shot in black and white with heavy practical creature effects and no CGI</query_text>
  <description>Films combining black-and-white photography with practical creature effects instead of CGI.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the compound visual-form ask: black-and-white photography combined with practical (non-CGI) creature effects as the craft register.",
      "endpoint_coverage": "Keyword has no registry member whose definition names black-and-white photography as a visual-form, and no member whose definition names practical creature effects as a technique. The registry's format/visual-form family covers documentary, animation sub-types, mockumentary, and short-film form — it deliberately excludes photographic-craft and effects-craft specifics. Semantic's production.production_techniques is the purpose-built sub-field for exactly these craft phrases; the ingest-side text for matching films carries 'black and white photography' and 'practical creature effects' in that sub-field.",
      "best_endpoint": "semantic",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "No classification member names either technique — black-and-white photography and practical creature effects are both craft choices the closed registry intentionally does not carry. The tier-1 bias cannot force Keyword to pick a member that does not definitionally fit, and reaching for a weak neighbor (e.g. an animation tag because effects are mentioned) would silently return a mismatched set. Semantic production is the only channel with the right vocabulary; production_techniques embeds these phrases directly. The bias is not a veto when the lower-preference endpoint fits decisively better.",
  "endpoint_to_run": "semantic",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "qualifier_inventory": "black-and-white photography as the visual register; heavy practical creature effects as the effects craft; an explicit absence of CGI as the negated alternative.",
      "space_queries": [
        {
          "carries_qualifiers": "production carries production_techniques on the ingest side as compact craft phrases; 'black and white photography', 'practical creature effects', and 'practical effects, no CGI' all land directly in that sub-field. filming_locations is not implicated.",
          "space": "production",
          "weight": "central",
          "content": {
            "filming_locations": [],
            "production_techniques": ["black and white photography", "practical creature effects", "practical effects, no CGI"]
          }
        }
      ],
      "primary_vector": "production"
    },
    "polarity": "positive"
  }
}
```

**Example: lower-tier-wins — cinematographic craft with no canonical tag → semantic**

```xml
<raw_query>single long take cinematography</raw_query>
<overall_query_intention_exploration>The user wants films built around a single continuous long take — a specific cinematographic craft technique. The registry has no member for long-take filmmaking; this is exactly the technique level that routes to semantic production.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films shot as a single continuous long take.</captured_meaning>
  <category_name>Format + visual-format specifics</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies shot as a single continuous long take.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>single long take cinematography</query_text>
  <description>Films shot as a single continuous long take.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the single-long-take cinematographic craft — films shot as one continuous take rather than conventionally edited.",
      "endpoint_coverage": "Keyword has no registry member whose definition names long-take or single-take filmmaking. The closed vocabulary covers format families and broad visual-form classifications, not cinematographic craft at this granularity. Semantic's production.production_techniques is where ingest-side text carries 'single-take long shot' and related phrases for films built this way; it is the purpose-built sub-field for this requirement.",
      "best_endpoint": "semantic",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "No classification member names the single-long-take technique, so Keyword cannot carry it — the tier-1 bias cannot force a pick that the registry does not definitionally support. Semantic production is the only channel with the right vocabulary: production_techniques embeds craft phrases in the same form the ingest side uses. Clear on-merits win for the lower-preference endpoint.",
  "endpoint_to_run": "semantic",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "qualifier_inventory": "single continuous long take as the core craft technique; long-take cinematography as the broader craft register.",
      "space_queries": [
        {
          "carries_qualifiers": "production carries production_techniques as compact craft phrases on the ingest side; 'single-take long shot' and 'long take cinematography' land directly in that sub-field. filming_locations is not implicated.",
          "space": "production",
          "weight": "central",
          "content": {
            "filming_locations": [],
            "production_techniques": ["single-take long shot", "long take cinematography", "continuous shot"]
          }
        }
      ],
      "primary_vector": "production"
    },
    "polarity": "positive"
  }
}
```

**Example: too-vague visual ask → no-fire**

```xml
<raw_query>movies with a really cool visual style</raw_query>
<overall_query_intention_exploration>The user gestures at an appealing visual style but names no specific format, visual-form, or craft technique. "Cool visual style" is a vague positive without an identifiable form label or craft phrase.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films with an unspecified appealing visual style.</captured_meaning>
  <category_name>Format + visual-format specifics</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies with a cool visual style.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies with a really cool visual style</query_text>
  <description>Films with an appealing but unspecified visual style.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the named format, visual-form, or craft technique the user is asking for.",
      "endpoint_coverage": "Neither candidate can fit. Keyword requires a specific registry member whose definition covers the named form — the atom names no form, so no member is definitionally supported. Semantic's production.production_techniques requires concrete craft phrases ('shot on 16mm', 'practical effects', 'black and white photography') — 'cool visual style' supplies none, and fabricating a placeholder technique would embed noise against the ingest text of unrelated films.",
      "best_endpoint": "None",
      "best_endpoint_gaps": "The atom names no form, visual-form, or craft technique. Firing either endpoint would invent content the input does not support and silently narrow or pollute the result set."
    }
  ],
  "performance_vs_bias_analysis": "The tier-1 bias does not force a pick when no form or technique is actually named. Picking any registry member would invent a form the user did not specify, and populating production_techniques with a placeholder phrase would dilute the query vector. No-fire is the correct response.",
  "endpoint_to_run": "None",
  "endpoint_parameters": null
}
```
