# Examples

These examples calibrate the fire / no-fire decision and the axis-population shape for the Franchise Structure endpoint. Pay attention to how each case either has or lacks a resolvable franchise anchor, and how wrapper `match_mode` / `polarity` pair with the direction-agnostic `parameters`.

**Example: shared-universe umbrella fire**

```xml
<raw_query>good MCU movies</raw_query>
<overall_query_intention_exploration>The user wants well-received films from the Marvel Cinematic Universe. Two requirements: MCU membership and a reception-quality floor.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The film should be part of the Marvel Cinematic Universe.</captured_meaning>
  <category_name>Franchise / universe lineage</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Member of the Marvel Cinematic Universe shared universe.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>MCU movies</query_text>
  <description>Films in the Marvel Cinematic Universe.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>good</query_text>
    <description>The films should be well-received.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Membership in the Marvel Cinematic Universe shared universe.",
      "relation_to_endpoint": "Resolved by the franchise_or_universe_names axis against the franchise token index; covers both lineage and shared_universe match sets.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The phrase 'MCU movies' names the Marvel Cinematic Universe directly. 'MCU' is a common expansion target — emitting both 'marvel cinematic universe' and 'marvel' performs an umbrella sweep across any Marvel-tagged entry. No subgroup, lineage position, structural flag, or launch scope phrase is present. prefer_lineage stays false because this is explicitly an umbrella query.",
      "franchise_or_universe_names": ["marvel cinematic universe", "marvel"],
      "recognized_subgroups": null,
      "lineage_position": null,
      "structural_flags": null,
      "launch_scope": null,
      "prefer_lineage": false
    },
    "polarity": "positive"
  }
}
```

**Example: subgroup-scoped fire (Harry Potter mainline, excluding Fantastic Beasts)**

```xml
<raw_query>the Harry Potter mainline, not the Fantastic Beasts spinoffs</raw_query>
<overall_query_intention_exploration>The user wants the main Harry Potter film series and wants to exclude the Fantastic Beasts spinoff line. Two requirements working together.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films on the main Harry Potter lineage, not the Fantastic Beasts spinoff branch.</captured_meaning>
  <category_name>Franchise / universe lineage</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Member of the main Harry Potter lineage, excluding the Fantastic Beasts spinoff line.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>the Harry Potter mainline, not the Fantastic Beasts spinoffs</query_text>
  <description>The main Harry Potter line, with the Fantastic Beasts spinoffs excluded.</description>
  <modifiers>
    <modifier>
      <original_text>not</original_text>
      <effect>Excludes the Fantastic Beasts spinoff entries from the requested set.</effect>
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
      "aspect_description": "Main Harry Potter lineage membership, with explicit exclusion of the Fantastic Beasts spinoff branch of the Wizarding World universe.",
      "relation_to_endpoint": "Positive side — lineage-focused Harry Potter membership — populates franchise_or_universe_names with prefer_lineage=true to uprank mainline over universe-adjacent entries. The Fantastic Beasts exclusion is a separate structural-flag requirement on the same endpoint, paired with wrapper polarity=negative.",
      "coverage_gaps": "This atom captures the positive mainline side only; the 'not the Fantastic Beasts spinoffs' exclusion is handled as a separate endpoint call with polarity=negative and structural_flags=[spinoff] at the franchise scope."
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The atom names 'Harry Potter' as a specific lineage and explicitly contrasts the mainline against the spinoff branch. One named franchise, not an umbrella sweep, no subgroup, no structural flag on the positive side — the negative side (spinoff exclusion) is a separate atom. Set prefer_lineage=true to uprank the main Harry Potter films over Fantastic Beasts and other universe-adjacent entries on the lineage side.",
      "franchise_or_universe_names": ["harry potter"],
      "recognized_subgroups": null,
      "lineage_position": null,
      "structural_flags": null,
      "launch_scope": null,
      "prefer_lineage": true
    },
    "polarity": "positive"
  }
}
```

**Example: lineage-position fire (original, not the remake)**

```xml
<raw_query>the original Scarface, not the remake</raw_query>
<overall_query_intention_exploration>The user wants the earlier Scarface film in the lineage — the 1932 original, positioned against the 1983 remake.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The earlier Scarface film, positioned as the original within the Scarface lineage rather than the later remake.</captured_meaning>
  <category_name>Franchise / universe lineage</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>The original entry in the Scarface lineage, not the remake entry.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>the original Scarface, not the remake</query_text>
  <description>The original Scarface film, contrasted against its remake.</description>
  <modifiers>
    <modifier>
      <original_text>not</original_text>
      <effect>Excludes the remake entry from the lineage.</effect>
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
      "aspect_description": "Positional anchor on the Scarface lineage: the original entry, not the remake entry.",
      "relation_to_endpoint": "Franchise name populates franchise_or_universe_names. The positive side of the user's ask is the original / mainline Scarface, so prefer_lineage=true upranks the lineage side. The 'not the remake' exclusion is a separate endpoint call with lineage_position=remake and wrapper polarity=negative.",
      "coverage_gaps": "This atom captures the positive lineage anchor; the remake exclusion is a separate handler call with polarity=negative on lineage_position=remake."
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The atom names 'Scarface' — a specific lineage with two entries, the 1932 original and the 1983 remake. The user's positive ask is the original entry. Populate the name axis and set prefer_lineage=true so the mainline (original) entry ranks above universe-adjacent material. The remake exclusion belongs in a separate call.",
      "franchise_or_universe_names": ["scarface"],
      "recognized_subgroups": null,
      "lineage_position": null,
      "structural_flags": null,
      "launch_scope": null,
      "prefer_lineage": true
    },
    "polarity": "positive"
  }
}
```

**Example: no-fire — unanchored "remakes" (Cat 5 owns this)**

```xml
<raw_query>good remakes of 80s movies</raw_query>
<overall_query_intention_exploration>The user wants well-received remakes of films originally released in the 1980s. No specific franchise is named.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The film is a remake of an earlier film.</captured_meaning>
  <category_name>Franchise / universe lineage</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>The film is positioned as a remake of an earlier entry.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>remakes of 80s movies</query_text>
  <description>Films that are remakes of 1980s releases.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>good</query_text>
    <description>Reception floor.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The film should be a remake, with no named source franchise.",
      "relation_to_endpoint": "The franchise endpoint's lineage_position=remake requires a franchise anchor to point at — it classifies a film's slot within a specific named lineage. A bare 'remakes' ask with no named franchise has no anchor to resolve against.",
      "coverage_gaps": "No named franchise, universe, or subgroup appears in the atom, parent fragment, or sibling fragments. This is an origin-medium remake flag ('the film is a remake of some earlier work') rather than a lineage position inside a specific named series — Cat 5 (Adaptation source flag) owns that via the keyword endpoint's REMAKE classification."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```

**Example: no-fire — "novel adaptations" (also Cat 5)**

```xml
<raw_query>great novel adaptations</raw_query>
<overall_query_intention_exploration>The user wants well-regarded films that adapt novels. No specific franchise is named.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The film is an adaptation of a novel.</captured_meaning>
  <category_name>Franchise / universe lineage</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>The film is positioned as a novel adaptation.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>novel adaptations</query_text>
  <description>Films adapted from novels.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>great</query_text>
    <description>Reception-quality framing.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The film should be based on a novel.",
      "relation_to_endpoint": "The franchise endpoint indexes named franchises, subgroups, lineage positions, structural flags, and launch scopes. 'Novel adaptation' is an origin-medium flag independent of any lineage — none of those axes apply.",
      "coverage_gaps": "No franchise, subgroup, or lineage-position signal is present. This belongs to Cat 5 (Adaptation source flag), which routes to the keyword endpoint and tags films by source medium. Dispatch here looks to be a misroute; declining to fire is the correct outcome."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```
