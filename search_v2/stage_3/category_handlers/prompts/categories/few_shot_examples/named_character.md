# Examples

These examples calibrate character-lookup decisions: when the subject-framing triggers `central`, when plain mention triggers `default`, where alternative credited forms pay off, and the two most common no-fire shapes (archetype misroute, real-person biopic).

**Example: iconic franchise-anchor character, subject-framed → `central`**

```xml
<raw_query>Batman movies</raw_query>
<overall_query_intention_exploration>The user wants films whose story centers on Batman. Short, no qualifiers, no sibling framing — a clean request for Batman-as-subject films.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films centered on the character Batman.</captured_meaning>
  <category_name>Named character</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies in which the character Batman is the subject of the film.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>Batman movies</query_text>
  <description>Movies about Batman.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify Batman as the named character the film must feature.",
      "relation_to_endpoint": "Character lookup resolves the credited character name 'Batman' against the character posting table; alternative credited forms cover the Bruce Wayne billing that some films use.",
      "coverage_gaps": null
    },
    {
      "aspect_description": "Batman is framed as the film's subject, not just a presence.",
      "relation_to_endpoint": "Character prominence mode 'central' weights movies where Batman is the story's subject above those with incidental appearances.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "entity_type_evidence": "The phrase 'Batman movies' names the character Batman as the subject of the film — a specific named persona credited across many films.",
      "name_resolution_notes": "Primary credited form in cast lists is 'Batman'.",
      "primary_form": "Batman",
      "entity_type": "character",
      "person_category": null,
      "primary_category": null,
      "alternative_forms_evidence": "Batman is frequently dual-credited with his civilian identity Bruce Wayne; some films also use the long-form 'The Batman' billing.",
      "alternative_forms": ["Bruce Wayne", "The Batman"],
      "prominence_evidence": "Possessive subject framing — 'Batman movies' — treats Batman as the subject of the film, which is the central-prominence signal.",
      "prominence_mode": "central",
      "title_pattern_match_type": null
    },
    "polarity": "positive"
  }
}
```

**Example: any-appearance framing → `default`**

```xml
<raw_query>movies with any Wolverine appearance, even a cameo</raw_query>
<overall_query_intention_exploration>The user wants films where Wolverine appears at all, explicitly including minor / cameo appearances. The framing is inclusive rather than subject-focused.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films in which the character Wolverine appears at all, including cameos.</captured_meaning>
  <category_name>Named character</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies featuring any appearance of the character Wolverine.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies with any Wolverine appearance, even a cameo</query_text>
  <description>Any film in which Wolverine appears, including minor appearances.</description>
  <modifiers>
    <modifier>
      <original_text>even a cameo</original_text>
      <effect>widens the request to include minor appearances rather than narrowing to leads</effect>
      <type>POLARITY_MODIFIER</type>
    </modifier>
  </modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Find films where Wolverine appears on screen in any capacity.",
      "relation_to_endpoint": "Character lookup on the credited character string 'Wolverine' covers any film where that character has a billed role; alternative credited forms pick up the Logan / James Howlett billings used in many X-Men and standalone films.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "entity_type_evidence": "The query names Wolverine — a specific Marvel character credited across many films — and asks about his presence, not his actor.",
      "name_resolution_notes": "The primary credited form in cast lists is 'Wolverine'.",
      "primary_form": "Wolverine",
      "entity_type": "character",
      "person_category": null,
      "primary_category": null,
      "alternative_forms_evidence": "Wolverine is frequently billed under his civilian names Logan and James Howlett, both of which appear in cast lists across the X-Men and standalone films.",
      "alternative_forms": ["Logan", "James Howlett"],
      "prominence_evidence": "The 'even a cameo' modifier explicitly widens the request to minor appearances — the character is not framed as the subject. Default prominence matches any-appearance semantics without biasing toward subject films.",
      "prominence_mode": "default",
      "title_pattern_match_type": null
    },
    "polarity": "positive"
  }
}
```

**Example: common character name across many films → `central` with inclusive aliases**

```xml
<raw_query>James Bond movies</raw_query>
<overall_query_intention_exploration>The user wants James Bond films — the long-running spy franchise built around the character. The phrasing is subject-framed.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films centered on the character James Bond.</captured_meaning>
  <category_name>Named character</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies in which the character James Bond is the subject of the film.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>James Bond movies</query_text>
  <description>Films about James Bond.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify James Bond as the named character the film must center on.",
      "relation_to_endpoint": "Character lookup on 'James Bond' resolves the canonical credited form that appears across the franchise's cast lists.",
      "coverage_gaps": null
    },
    {
      "aspect_description": "The character is framed as the film's subject, not an incidental presence.",
      "relation_to_endpoint": "Character prominence 'central' biases toward films in which Bond is the story's subject, which is the entire franchise's shape.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "entity_type_evidence": "The phrase 'James Bond movies' names the character James Bond as the subject of the film. Bond is a specific named persona credited across decades of films.",
      "name_resolution_notes": "The canonical credited form is 'James Bond'.",
      "primary_form": "James Bond",
      "entity_type": "character",
      "person_category": null,
      "primary_category": null,
      "alternative_forms_evidence": "Bond films vary billing: some credit him as '007', some as 'James Bond 007', and some as 'Bond, James Bond' in stylized billings. All three appear in real cast lists.",
      "alternative_forms": ["007", "James Bond 007"],
      "prominence_evidence": "'James Bond movies' uses the character name as the possessive subject of the film — the central-prominence signal.",
      "prominence_mode": "central",
      "title_pattern_match_type": null
    },
    "polarity": "positive"
  }
}
```

**Example: archetype misroute — no-fire**

```xml
<raw_query>movies with a lovable rogue</raw_query>
<overall_query_intention_exploration>The user wants films featuring a lovable-rogue character type — a charming-outlaw archetype — not any specific named persona.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films featuring a lovable-rogue style character.</captured_meaning>
  <category_name>Named character</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies that feature a character fitting the lovable-rogue persona.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies with a lovable rogue</query_text>
  <description>Films featuring a lovable-rogue style character.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Find films whose cast includes a 'lovable rogue' character.",
      "relation_to_endpoint": "The entity endpoint performs literal string lookup against credited character names. 'Lovable rogue' is a character type pattern, not a credited character string, and no specific named persona is supplied.",
      "coverage_gaps": "This is a character archetype, not a named character. Character archetypes are handled by a different category (keyword + semantic); fabricating a primary_form from the type phrase would produce a zero-match lookup and mask the real signal."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```

**Example: real-person biopic reaches this category by misroute — no-fire**

```xml
<raw_query>the Princess Diana biopic</raw_query>
<overall_query_intention_exploration>The user wants a film depicting the real person Princess Diana — a biographical subject, not a fictional character credited in cast lists.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>A film depicting Princess Diana as its subject.</captured_meaning>
  <category_name>Named character</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>A movie whose subject is the real figure Princess Diana.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>the Princess Diana biopic</query_text>
  <description>A biographical film about Princess Diana.</description>
  <modifiers>
    <modifier>
      <original_text>biopic</original_text>
      <effect>marks the film as a biographical depiction of a real person</effect>
      <type>ROLE_MARKER</type>
    </modifier>
  </modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Find a biographical film whose subject is Princess Diana.",
      "relation_to_endpoint": "The entity endpoint's character lookup resolves credited fictional character strings from cast lists. Princess Diana is a real historical subject depicted in a biopic, not a fictional character credited as a persona — different category (specific subject).",
      "coverage_gaps": "The requirement is a real-person biographical subject, not a named fictional character. Running a character lookup on 'Princess Diana' would mis-target the channel — actresses playing her are credited under her name but the user's intent is the subject matter, which is better served by the subject-presence pathway."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```
