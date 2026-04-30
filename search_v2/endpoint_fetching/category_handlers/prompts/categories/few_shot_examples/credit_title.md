# Examples

Each example shows the full XML payload and the exact JSON output the handler should emit. The set calibrates role-marker routing, title-pattern firing, and the two main no-fire boundaries (below-the-line roles → Cat 29, source-material authors → Cat 30).

**Example: named actor with explicit lead-role marker**

```xml
<raw_query>movies starring Denzel Washington in a lead role</raw_query>
<overall_query_intention_exploration>The user wants films where Denzel Washington plays the leading character, not ensemble pieces where he appears briefly. Lead-role prominence is explicit.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films featuring Denzel Washington as the lead actor.</captured_meaning>
  <category_name>Credit + title text</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Films where Denzel Washington is credited as the leading actor.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies starring Denzel Washington in a lead role</query_text>
  <description>Denzel Washington films where he is the lead.</description>
  <modifiers>
    <modifier>
      <original_text>starring</original_text>
      <effect>binds the named person to the actor role</effect>
      <type>ROLE_MARKER</type>
    </modifier>
    <modifier>
      <original_text>in a lead role</original_text>
      <effect>constrains prominence to leading billing</effect>
      <type>ROLE_MARKER</type>
    </modifier>
  </modifiers>
</parent_fragment>
<sibling_fragments>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Denzel Washington must be credited as an actor on the film.",
      "relation_to_endpoint": "Entity resolves this directly against the actor posting table by exact name match.",
      "coverage_gaps": null
    },
    {
      "aspect_description": "He must be in a lead (top-billed) role, not a supporting or minor appearance.",
      "relation_to_endpoint": "Actor-billing prominence scoring supports lead mode.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "entity_type_evidence": "Named real person with ROLE_MARKER 'starring' binding him to acting credits. Person lookup on the actor table.",
      "name_resolution_notes": "Full conventional credited name; no typo to correct.",
      "primary_form": "Denzel Washington",
      "entity_type": "person",
      "person_category": "actor",
      "primary_category": null,
      "alternative_forms_evidence": "No alternate credited forms known; consistently credited under this full name.",
      "alternative_forms": [],
      "prominence_evidence": "Fragment modifier 'in a lead role' explicitly requests leading billing.",
      "prominence_mode": "lead",
      "title_pattern_match_type": null
    },
    "polarity": "positive"
  }
}
```

**Example: director with "directed by" marker**

```xml
<raw_query>anything directed by Bong Joon-ho</raw_query>
<overall_query_intention_exploration>The user wants Bong Joon-ho's directorial filmography — a wide net over his work as director, with no other constraints.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films directed by Bong Joon-ho.</captured_meaning>
  <category_name>Credit + title text</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Films credited with Bong Joon-ho as director.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>anything directed by Bong Joon-ho</query_text>
  <description>Bong Joon-ho's directorial work.</description>
  <modifiers>
    <modifier>
      <original_text>directed by</original_text>
      <effect>binds the named person to the director role</effect>
      <type>ROLE_MARKER</type>
    </modifier>
  </modifiers>
</parent_fragment>
<sibling_fragments>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Bong Joon-ho must be credited as director on the film.",
      "relation_to_endpoint": "Entity resolves this directly against the director posting table by exact name match.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "entity_type_evidence": "Named real person with ROLE_MARKER 'directed by' binding him to the director role. Person lookup on the director table.",
      "name_resolution_notes": "Standard romanization of the director's name; no correction needed.",
      "primary_form": "Bong Joon-ho",
      "entity_type": "person",
      "person_category": "director",
      "primary_category": null,
      "alternative_forms_evidence": "Alternate romanization 'Bong Joon Ho' without the hyphen is plausible, but shared normalization collapses hyphenation variants so no explicit alias needed.",
      "alternative_forms": [],
      "prominence_evidence": null,
      "prominence_mode": null,
      "title_pattern_match_type": null
    },
    "polarity": "positive"
  }
}
```

**Example: title-substring match**

```xml
<raw_query>movies with the word "love" in the title</raw_query>
<overall_query_intention_exploration>The user wants a literal title-text match: any film whose title contains the word "love" somewhere in it. No thematic filter implied.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films whose title contains the substring "love".</captured_meaning>
  <category_name>Credit + title text</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Films whose movie title contains the literal word "love".</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies with the word "love" in the title</query_text>
  <description>Title contains the word "love".</description>
  <modifiers>
    <modifier>
      <original_text>in the title</original_text>
      <effect>binds the literal token to the movie title field</effect>
      <type>ROLE_MARKER</type>
    </modifier>
  </modifiers>
</parent_fragment>
<sibling_fragments>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The movie's title must contain the literal token 'love'.",
      "relation_to_endpoint": "Entity supports literal substring matching against the title field via title_pattern with contains semantics.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "entity_type_evidence": "Requirement names a literal token to match inside the movie title, with ROLE_MARKER 'in the title' binding to the title field. Title pattern lookup.",
      "name_resolution_notes": "Single lowercase token, no quotes or wildcards in the stored form.",
      "primary_form": "love",
      "entity_type": "title_pattern",
      "person_category": null,
      "primary_category": null,
      "alternative_forms_evidence": null,
      "alternative_forms": null,
      "prominence_evidence": null,
      "prominence_mode": null,
      "title_pattern_match_type": "contains"
    },
    "polarity": "positive"
  }
}
```

**Example: no-fire on a below-the-line creator (cinematographer)**

```xml
<raw_query>films shot by Roger Deakins</raw_query>
<overall_query_intention_exploration>The user wants films where Roger Deakins was the cinematographer — a below-the-line craft credit, not one of the indexed posting roles.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films with Roger Deakins as cinematographer.</captured_meaning>
  <category_name>Credit + title text</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Films credited with Roger Deakins in a film-credit role.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>films shot by Roger Deakins</query_text>
  <description>Roger Deakins cinematography credits.</description>
  <modifiers>
    <modifier>
      <original_text>shot by</original_text>
      <effect>binds the named person to the cinematographer role</effect>
      <type>ROLE_MARKER</type>
    </modifier>
  </modifiers>
</parent_fragment>
<sibling_fragments>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Roger Deakins must be credited as cinematographer on the film.",
      "relation_to_endpoint": "Entity's posting tables cover actor/director/writer/producer/composer only — cinematographer is not indexed.",
      "coverage_gaps": "Cinematographer is a below-the-line craft role outside Entity's scope. The correct channel is the Below-the-line creator category, which reaches cinematographer credits through semantic retrieval over reception and identity prose. Firing here against any posting table would either return nothing or return spurious non-cinematographer credits."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```

**Example: no-fire on a source-material author**

```xml
<raw_query>Stephen King adaptations</raw_query>
<overall_query_intention_exploration>The user wants films adapted from Stephen King's novels and stories. Stephen King is the source author, not a film credit.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films adapted from works by Stephen King.</captured_meaning>
  <category_name>Credit + title text</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Films crediting Stephen King in a film-credit role.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>Stephen King adaptations</query_text>
  <description>Films adapted from Stephen King source material.</description>
  <modifiers>
  </modifiers>
</parent_fragment>
<sibling_fragments>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Films must derive from source material written by Stephen King.",
      "relation_to_endpoint": "Entity indexes film credits, not source-material authorship. King occasionally carries a screenwriter or executive-producer credit, but those are incidental and would miss the adaptations where he holds no film credit at all.",
      "coverage_gaps": "The requirement names an origin-work author, not a film credit. That belongs to the Source-material author category, which retrieves via semantic search over plot and reception vectors where the author's name surfaces in prose. A posting-table lookup on 'Stephen King' here would silently miss the majority of his adaptations."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```
