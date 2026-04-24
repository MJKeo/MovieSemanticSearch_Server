# Examples

These examples calibrate the bare-genre Keyword fire, the qualifier-laden Semantic fire, and the no-fire shape when the requirement slips into a neighbouring category.

**Example: bare canonical genre → Keyword**

```xml
<raw_query>horror movies</raw_query>
<overall_query_intention_exploration>The user wants films in the horror genre. A bare top-level genre label with no qualifier.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films in the horror genre.</captured_meaning>
  <category_name>Top-level genre</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies whose top-level genre is horror.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>horror movies</query_text>
  <description>Horror films.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Find films whose top-level genre is horror.",
      "endpoint_coverage": "Keyword has a dedicated HORROR registry member resolving onto the keyword_ids posting list — the closed-vocabulary tag directly names the top-level genre. Semantic plot_analysis could match on genre_signatures containing 'horror', but without a qualifier there is nothing for the vector space to carry beyond what the tag already encodes.",
      "best_endpoint": "keyword",
      "best_endpoint_gaps": null
    }
  ],
  "endpoint_to_run": "keyword",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The requirement names horror as the top-level genre with no qualifier. The registry carries HORROR as a canonical member for the broad genre bucket.",
      "candidate_shortlist": "HORROR is the bare top-level genre. Narrower members (SLASHER_HORROR, BODY_HORROR, FOLK_HORROR) would require a premise-specific signal the user did not give — picking one on weak evidence would drop every horror film lacking that specific sub-tag.",
      "classification": "HORROR"
    },
    "polarity": "positive"
  }
}
```

**Example: qualifier-laden genre → Semantic**

```xml
<raw_query>dark action movies</raw_query>
<overall_query_intention_exploration>The user wants action films with a dark tonal texture — not the bright or bombastic end of the genre. The qualifier 'dark' is load-bearing and has no canonical tag equivalent.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Action films with a dark tonal texture.</captured_meaning>
  <category_name>Top-level genre</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies whose top-level genre is action and whose tonal texture reads as dark.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>dark action movies</query_text>
  <description>Action films with a dark tonal texture.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Find action films whose genre texture reads as dark rather than bright or bombastic.",
      "endpoint_coverage": "Keyword has ACTION as a bare top-level member but no tag for the 'dark action' compound — picking ACTION would return every action film regardless of tonal texture, losing the qualifier entirely. Semantic plot_analysis carries genre_signatures on the ingest side where 'dark action' would land as a compact descriptor phrase, exactly matching the space's native vocabulary.",
      "best_endpoint": "semantic",
      "best_endpoint_gaps": null
    }
  ],
  "endpoint_to_run": "semantic",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "qualifier_inventory": "Two atoms: action as the top-level genre, and a dark tonal qualifier shaping the kind of action. The compound is what matters — the qualifier is not separable from the genre, which is why a bare ACTION tag would drop the signal.",
      "space_queries": [
        {
          "carries_qualifiers": "plot_analysis carries genre_signatures where 'dark action' lands as an ingest-side descriptor phrase — the space's native vocabulary for qualified genre textures.",
          "space": "plot_analysis",
          "weight": "central",
          "content": {
            "genre_signatures": ["dark action", "grim action thriller"]
          }
        }
      ],
      "primary_vector": "plot_analysis"
    },
    "polarity": "positive"
  }
}
```

**Example: another bare canonical genre → Keyword**

```xml
<raw_query>romcoms</raw_query>
<overall_query_intention_exploration>The user wants romantic comedies — a canonical top-level genre with a dedicated registry member.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Romantic comedy films.</captured_meaning>
  <category_name>Top-level genre</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies whose top-level genre is romantic comedy.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>romcoms</query_text>
  <description>Romantic comedies.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Find romantic comedy films.",
      "endpoint_coverage": "Keyword has ROMANTIC_COMEDY as a dedicated registry member for the romcom bucket — a closed-vocabulary tag resolving directly onto the keyword_ids posting list. Semantic plot_analysis genre_signatures could match 'romantic comedy' prose but offers no precision advantage over the tag for a bare-genre ask.",
      "best_endpoint": "keyword",
      "best_endpoint_gaps": null
    }
  ],
  "endpoint_to_run": "keyword",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The user names romcom — a standard abbreviation for romantic comedy — as a bare top-level genre bucket. No qualifier, no sub-genre specificity.",
      "candidate_shortlist": "ROMANTIC_COMEDY directly covers the romcom bucket. ROMANCE alone would be broader and drop the comedic dimension; COMEDY alone would drop the romantic dimension. The compound member fits both halves of the ask exactly.",
      "classification": "ROMANTIC_COMEDY"
    },
    "polarity": "positive"
  }
}
```

**Example: a different qualifier-laden genre → Semantic**

```xml
<raw_query>quiet drama</raw_query>
<overall_query_intention_exploration>The user wants drama films with a quiet, low-key texture — understated rather than heavy or emotionally charged. The qualifier 'quiet' is load-bearing and does not map to a canonical tag.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Drama films with a quiet, understated texture.</captured_meaning>
  <category_name>Top-level genre</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies whose top-level genre is drama and whose tonal texture reads as quiet and understated.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>quiet drama</query_text>
  <description>Quiet, understated drama films.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Find drama films whose genre texture reads as quiet and understated rather than heavy or emotionally charged.",
      "endpoint_coverage": "Keyword has DRAMA as a bare top-level member and no tag for the 'quiet drama' compound — picking DRAMA would overshoot by pulling every drama regardless of register. Semantic plot_analysis genre_signatures is the natural home for a qualified genre descriptor; 'quiet drama' is exactly the kind of compact phrase that space holds on the ingest side.",
      "best_endpoint": "semantic",
      "best_endpoint_gaps": null
    }
  ],
  "endpoint_to_run": "semantic",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "qualifier_inventory": "Two atoms: drama as the top-level genre, and a 'quiet' tonal qualifier shaping the register. The compound is what matters — the qualifier selects among dramas rather than describing a separable tone.",
      "space_queries": [
        {
          "carries_qualifiers": "plot_analysis carries genre_signatures where 'quiet drama' lands as an ingest-side descriptor phrase — the space's native register for qualified genre textures.",
          "space": "plot_analysis",
          "weight": "central",
          "content": {
            "genre_signatures": ["quiet drama", "understated character drama"]
          }
        }
      ],
      "primary_vector": "plot_analysis"
    },
    "polarity": "positive"
  }
}
```

**Example: named sub-genre is Cat 15, not Cat 11 — no-fire**

```xml
<raw_query>body horror movies</raw_query>
<overall_query_intention_exploration>The user names a specific, recognized sub-genre — body horror — rather than a top-level genre with an open-ended qualifier. Sub-genre territory belongs to Cat 15.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films in the body horror sub-genre.</captured_meaning>
  <category_name>Top-level genre</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies whose sub-genre is body horror.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>body horror movies</query_text>
  <description>Body horror films.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Find films in the body horror sub-genre.",
      "endpoint_coverage": "The requirement names a recognized sub-genre label, not a top-level genre with an open-ended qualifier. Routing onto Keyword HORROR would silently broaden to every horror film and drop the body-horror specificity; routing onto Semantic genre_signatures would author a body for a sub-genre this category does not own. Sub-genre + story archetype (Cat 15) is the correct owner, with its own tiered Keyword/Semantic handling of named sub-genre labels.",
      "best_endpoint": "None",
      "best_endpoint_gaps": "Upstream dispatch landed a sub-genre ask in the top-level-genre handler; both channels here would misrepresent the requirement. No-fire — the sub-genre category owns this."
    }
  ],
  "endpoint_to_run": "None",
  "endpoint_parameters": null
}
```
