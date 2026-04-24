# Examples

These examples calibrate the Keyword-vs-Metadata mutex for cinema traditions: picking the registry tag when one covers the tradition, falling back to `country_of_origin` when no tag exists, preferring the tag over the lossy country proxy when both look applicable, and no-firing on vague framings neither channel can target.

**Example: canonical tradition tag → keyword wins (Bollywood)**

```xml
<raw_query>Bollywood films</raw_query>
<overall_query_intention_exploration>The user wants films from the Bollywood tradition — the Hindi-language Mumbai film industry as an aesthetic, not a bare country-of-origin constraint.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films belonging to the Bollywood / Hindi cinema tradition.</captured_meaning>
  <category_name>Cultural tradition / national cinema</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies that belong to the Bollywood tradition.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>Bollywood films</query_text>
  <description>Films in the Bollywood tradition.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the Bollywood tradition — the Hindi-language Mumbai film industry as a named cinema aesthetic.",
      "endpoint_coverage": "Keyword has HINDI, whose definition names Bollywood / Hindi cinema directly. Metadata could approximate via country_of_origin=[IN], but that widens the set to every Indian film (Tamil, Telugu, Marathi, Bengali) and loses diaspora productions that still count as Bollywood. The tag is authoritative and more precise.",
      "best_endpoint": "keyword",
      "best_endpoint_gaps": null
    }
  ],
  "endpoint_to_run": "keyword",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The atom names Bollywood as a cinema tradition. The registry has a dedicated member for the Hindi film tradition whose definition explicitly names Bollywood.",
      "candidate_shortlist": "HINDI directly names Bollywood in its definition. Other Indian-language members (TAMIL, BENGALI, MARATHI, PUNJABI) describe separate regional traditions and would miscover. No broader member fits a specific named tradition.",
      "classification": "HINDI"
    },
    "polarity": "positive"
  }
}
```

**Example: canonical tradition tag → keyword wins (Italian neorealism / giallo family)**

```xml
<raw_query>Italian neorealism</raw_query>
<overall_query_intention_exploration>The user wants films in the Italian neorealist tradition — a specific aesthetic movement within Italian cinema. The registry does not carry neorealism as a standalone member but has ITALIAN, whose definition explicitly names Neorealism as a core part of the Italian cinema tradition.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films in the Italian neorealist cinema tradition.</captured_meaning>
  <category_name>Cultural tradition / national cinema</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies belonging to the Italian neorealism tradition.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>Italian neorealism</query_text>
  <description>Films in the Italian neorealist tradition.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the Italian neorealism tradition — an aesthetic movement within Italian cinema.",
      "endpoint_coverage": "Keyword has ITALIAN, whose definition explicitly names Neorealism among the Italian cinema traditions it covers. Metadata country_of_origin=[IT] would also pull Italian productions but would mix in genre films with no neorealist lineage and miss neorealism-influenced co-productions. The tag definition names the tradition; country does not.",
      "best_endpoint": "keyword",
      "best_endpoint_gaps": null
    }
  ],
  "endpoint_to_run": "keyword",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The atom names Italian neorealism — a named movement. The registry ties neorealism to the ITALIAN member's definition, which covers the Italian cinema tradition including Neorealism, giallo, peplum, and commedia all'italiana.",
      "candidate_shortlist": "ITALIAN covers the Italian cinema tradition including Neorealism per its definition. GIALLO and SPAGHETTI_WESTERN name other specific Italian movements and would under-cover; no neorealism-specific member exists. ITALIAN is the tightest registry match.",
      "classification": "ITALIAN"
    },
    "polarity": "positive"
  }
}
```

**Example: tag wins over lossy country proxy (Hong Kong action)**

```xml
<raw_query>Hong Kong action movies</raw_query>
<overall_query_intention_exploration>The user wants films in the Hong Kong action tradition — the Cantonese action cinema aesthetic (martial arts, gun-fu, stylized choreography). Many films in this tradition were co-produced or funded from outside Hong Kong, so the tradition is not reducible to legal origin.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films in the Hong Kong action-cinema tradition.</captured_meaning>
  <category_name>Cultural tradition / national cinema</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies belonging to the Hong Kong action cinema tradition.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>Hong Kong action movies</query_text>
  <description>Films in the Hong Kong action tradition.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the Hong Kong action-cinema tradition as a named aesthetic.",
      "endpoint_coverage": "Keyword has CANTONESE, whose definition names Hong Kong cinema and specifically its pioneering action / martial-arts tradition. Metadata country_of_origin=[HK] would be a lossy proxy — Hollywood-funded or US co-produced Hong Kong action films (a meaningful slice of the tradition) carry US origin and would drop out entirely, while bare HK origin would also surface unrelated HK dramas. The tag is authoritative and catches co-productions the country column misses.",
      "best_endpoint": "keyword",
      "best_endpoint_gaps": null
    }
  ],
  "endpoint_to_run": "keyword",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The atom names Hong Kong action cinema — a specific aesthetic tradition. The CANTONESE member's definition explicitly covers Hong Kong cinema's pioneering action and martial-arts films, which is the tradition the user named.",
      "candidate_shortlist": "CANTONESE names the Hong Kong cinema tradition including its action aesthetic. MANDARIN covers Mainland / Taiwan cinema and is the wrong tradition. No separate Hong-Kong-action-only member exists; CANTONESE is the tightest fit.",
      "classification": "CANTONESE"
    },
    "polarity": "positive"
  }
}
```

**Example: no registry tag → metadata fallback (Senegalese cinema)**

```xml
<raw_query>Senegalese cinema</raw_query>
<overall_query_intention_exploration>The user wants films from the Senegalese film tradition. The classification registry has no member whose definition names Senegalese cinema specifically, so country_of_origin is the best remaining signal despite being a lossy proxy.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films belonging to the Senegalese cinema tradition.</captured_meaning>
  <category_name>Cultural tradition / national cinema</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies belonging to the Senegalese cinema tradition.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>Senegalese cinema</query_text>
  <description>Films in the Senegalese cinema tradition.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the Senegalese cinema tradition.",
      "endpoint_coverage": "Keyword has no registry member whose definition names Senegalese cinema — the closest African-cinema-adjacent members (ARABIC, FRENCH as the Francophone tradition) do not specifically cover Senegal and would either over-pull or miscover. Metadata country_of_origin=[SN] is the best remaining signal: it is a lossy proxy (it drops diaspora-funded Senegalese-tradition films and under-serves co-productions) but it is the only channel that can target the request at all.",
      "best_endpoint": "metadata",
      "best_endpoint_gaps": "Country_of_origin is a legal-paperwork proxy for the tradition — co-productions and diaspora productions with Senegalese aesthetic lineage but non-SN legal origin will not surface."
    }
  ],
  "endpoint_to_run": "metadata",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "constraint_phrases": ["Senegalese cinema"],
      "target_attribute": "country_of_origin",
      "value_intent_label": "Senegal as production origin",
      "country_of_origin": {
        "countries": ["SN"]
      }
    },
    "polarity": "positive"
  }
}
```

**Example: vague tradition framing → no-fire**

```xml
<raw_query>movies that feel foreign</raw_query>
<overall_query_intention_exploration>The user wants films with a general non-domestic sensibility but names no specific tradition or country. The phrasing is a mood / framing qualifier, not a named cinema tradition.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films with a generally non-domestic feel.</captured_meaning>
  <category_name>Cultural tradition / national cinema</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies that feel like they come from outside the domestic cinema tradition.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies that feel foreign</query_text>
  <description>Films with a non-domestic sensibility.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify films with a non-domestic 'foreign' sensibility, with no specific tradition named.",
      "endpoint_coverage": "Keyword would require a named registry member, and the user named none — there is no 'foreign-feeling' tag and picking an arbitrary national tradition would fabricate specificity. Metadata country_of_origin could in principle encode 'not US' as negation, but the atom does not name any country to anchor on; inventing a concrete country list would be guesswork. Neither channel can target this honestly.",
      "best_endpoint": "None",
      "best_endpoint_gaps": "No named tradition and no named country — the requirement is too vague to translate to either endpoint's parameters without fabrication. Upstream dispatch to this category was a poor fit."
    }
  ],
  "endpoint_to_run": "None",
  "endpoint_parameters": null
}
```
