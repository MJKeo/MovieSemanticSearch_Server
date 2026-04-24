# Few-shot examples

These calibrate the two-path choice (registry brand vs. freeform surface forms), the trust-the-executor rule for era-bounded queries, and the no-fire cases for unstructured descriptors and distribution-only framing.

**Example: canonical enum-path fire.**

```xml
<raw_query>A24 horror movies</raw_query>
<overall_query_intention_exploration>User wants horror films produced by A24 — a studio atom plus a top-level-genre atom.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Produced by A24.</captured_meaning>
  <category_name>Studio / brand</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Films with A24 as the production brand.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>A24 horror movies</query_text>
  <description>Horror films made by A24.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Film is produced under the A24 brand.",
      "relation_to_endpoint": "A24 is a curated registry brand; the executor reads the brand posting list keyed by brand_id with its time window applied automatically.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "thinking": "Umbrella-level ask naming a registry brand. A24 has a dedicated ProductionBrand entry — take the brand path with no freeform fallback.",
      "brand": "a24",
      "freeform_names": null
    },
    "polarity": "positive"
  }
}
```

**Example: era-bounded registry fire, trust the executor on rename chains.**

```xml
<raw_query>classic MGM musicals from the golden age</raw_query>
<overall_query_intention_exploration>User wants golden-age MGM musicals — the studio atom points at MGM in its classic era, with separate atoms covering genre and era.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Produced by MGM in its classic / pre-merger era.</captured_meaning>
  <category_name>Studio / brand</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Films with MGM as the production brand.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>classic MGM musicals from the golden age</query_text>
  <description>Golden-age Hollywood musicals produced by MGM.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>musicals</query_text>
    <description>Musical genre.</description>
    <modifiers></modifiers>
  </fragment>
  <fragment>
    <query_text>from the golden age</query_text>
    <description>Release era is Hollywood's golden age.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Film is produced under the MGM brand.",
      "relation_to_endpoint": "MGM is a registry brand. The era-bounded reading (classic / golden-age) is carried by the sibling era atom on the metadata endpoint; here we pick the brand and let the registry's time windows handle rename-chain disambiguation from Amazon MGM.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "thinking": "Umbrella query naming MGM. The 'classic / golden-age' framing is an era shade handled by a sibling atom, not something to encode here. Pick mgm and let the registry time windows keep Amazon-era credits out when the film's release year is pre-merger.",
      "brand": "mgm",
      "freeform_names": null
    },
    "polarity": "positive"
  }
}
```

**Example: freeform-path fire for a long-tail studio not in the registry.**

```xml
<raw_query>Cannon Films action movies</raw_query>
<overall_query_intention_exploration>User wants action films from Cannon Films, the 80s low-budget outfit. Cannon is not in the curated registry.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Produced by Cannon Films.</captured_meaning>
  <category_name>Studio / brand</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Films credited to Cannon Films as a production company.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>Cannon Films action movies</query_text>
  <description>Action movies produced by Cannon Films.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Film is credited to Cannon Films as a production company.",
      "relation_to_endpoint": "Cannon Films is not a registry brand; it must go through the freeform surface-form path, which tokenizes and intersects against the studio token index.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "thinking": "Not a curated registry brand. Cannon is most commonly credited as 'Cannon Films' and 'The Cannon Group'; the Go-Go Pictures rebrand is a distinct outfit. Emit the two dominant surface forms IMDB actually uses.",
      "brand": null,
      "freeform_names": ["Cannon Films", "The Cannon Group"]
    },
    "polarity": "positive"
  }
}
```

**Example: no-fire on unstructured descriptor.**

```xml
<raw_query>indie studio movies with heart</raw_query>
<overall_query_intention_exploration>User wants small-scale independent films with emotional warmth. "Indie studio" is a descriptor, not a named production entity.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Produced by indie studios.</captured_meaning>
  <category_name>Studio / brand</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Films from independent production studios.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>indie studio movies</query_text>
  <description>Independent-studio films.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>with heart</query_text>
    <description>Emotionally warm tone.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Film is from an independent studio.",
      "relation_to_endpoint": "The studio endpoint resolves concrete named entities via the registry or surface-form tokens. 'Indie studio' is a descriptor — no brand to pick, no canonical IMDB credit form to tokenize.",
      "coverage_gaps": "No concrete studio named. Budget-scale or production-country signals would belong to the metadata endpoint, not here."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```

**Example: no-fire on distribution-deal framing (belongs in streaming-platform territory).**

```xml
<raw_query>Netflix originals to watch this weekend</raw_query>
<overall_query_intention_exploration>User wants things available on Netflix right now — the framing is about where to watch, not who produced the film.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Available as a Netflix original.</captured_meaning>
  <category_name>Studio / brand</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Films labeled as Netflix originals.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>Netflix originals</query_text>
  <description>Films branded as Netflix originals.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>to watch this weekend</query_text>
    <description>Availability framing.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Film carries a Netflix-originals label.",
      "relation_to_endpoint": "The Netflix registry brand covers the producer reading of Netflix. Here the full query frames this as an availability / watch-where question, which is a streaming-platform concern on the metadata endpoint, not production-brand attribution.",
      "coverage_gaps": "The user's intent is about availability, not production. Firing the brand path would retrieve only Netflix-produced titles and miss licensed catalog the user expects, while also blurring with the correct streaming-platform signal handled elsewhere."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```
