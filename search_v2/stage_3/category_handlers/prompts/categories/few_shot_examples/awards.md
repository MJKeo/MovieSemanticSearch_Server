# Examples

These examples calibrate the fire / no-fire decision, the FLOOR vs. THRESHOLD split, and the axis-population shape for the Award endpoint. Each case shows how the target atom's award vocabulary (or lack thereof) drives the scoring shape, filter axes, and wrapper polarity.

**Example: clean single-ceremony WIN fire**

```xml
<raw_query>Oscar winners from the last five years</raw_query>
<overall_query_intention_exploration>The user wants recent Oscar-winning films — anything that has taken home an Academy Award in the past five years. Two requirements: Oscar-win status and a recent-year range.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The film should have won an Oscar at the Academy Awards.</captured_meaning>
  <category_name>Award records</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>The film has won at least one Oscar at the Academy Awards.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>Oscar winners</query_text>
  <description>Films that have won an Oscar.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>from the last five years</query_text>
    <description>Recency window on release year.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The film has won at least one Oscar at the Academy Awards.",
      "relation_to_endpoint": "Specific prize naming — 'Oscar' — populates award_names. Winner framing populates outcome. Binary presence is the correct shape, so FLOOR at 1. The sibling recency fragment is a release-year filter, not an award-year filter, and is handled by metadata in a separate call.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The atom names 'Oscar' as the specific prize (not the ceremony). Outcome is explicitly 'winner'. No count, category, or year signal on the award side — the 'last five years' framing is a release-year window handled elsewhere. Single-filter, no-count pattern.",
      "scoring_shape_label": "specific filter, no count",
      "scoring_mode": "floor",
      "scoring_mark": 1,
      "ceremonies": null,
      "award_names": ["Oscar"],
      "category_tags": null,
      "outcome": "winner",
      "years": null
    },
    "polarity": "positive"
  }
}
```

**Example: nomination fire (BAFTA-nominated)**

```xml
<raw_query>BAFTA-nominated British dramas</raw_query>
<overall_query_intention_exploration>The user wants British drama films that have received BAFTA nominations. Three requirements: BAFTA nomination, British origin, drama genre.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The film has been nominated at the BAFTA Awards.</captured_meaning>
  <category_name>Award records</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>The film has at least one BAFTA nomination.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>BAFTA-nominated</query_text>
  <description>Films with at least one BAFTA nomination.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>British</query_text>
    <description>British production origin.</description>
    <modifiers></modifiers>
  </fragment>
  <fragment>
    <query_text>dramas</query_text>
    <description>Drama genre.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The film has at least one nomination at the BAFTA Awards ceremony.",
      "relation_to_endpoint": "BAFTA is a tracked ceremony — populates ceremonies. Nominee framing populates outcome. No specific prize or category named, so award_names and category_tags stay null. Binary presence is the correct shape, so FLOOR at 1.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The atom names 'BAFTA' as the ceremony (not a specific prize), with nominee outcome. No prize name, category, count, or year signal. Single-filter, no-count pattern on the ceremony axis.",
      "scoring_shape_label": "specific filter, no count",
      "scoring_mode": "floor",
      "scoring_mark": 1,
      "ceremonies": ["BAFTA Awards"],
      "award_names": null,
      "category_tags": null,
      "outcome": "nominee",
      "years": null
    },
    "polarity": "positive"
  }
}
```

**Example: multi-win superlative (multi-Oscar winner)**

```xml
<raw_query>multi-Oscar-winning epics</raw_query>
<overall_query_intention_exploration>The user wants epic-scale films that have won multiple Oscars. Two requirements: multiple Oscar wins and an epic scope.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The film has won multiple Oscars.</captured_meaning>
  <category_name>Award records</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>The film has won two or more Oscars.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>multi-Oscar-winning</query_text>
  <description>Films with multiple Oscar wins.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>epics</query_text>
    <description>Epic scope / scale.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The film has won two or more Oscars.",
      "relation_to_endpoint": "'Multi-' is an explicit count of at least 2. Specific prize ('Oscar') populates award_names, outcome is winner. The count is a hard floor, not a gradient toward saturation — past two wins, a film is 'multi-Oscar-winning' regardless of how many more it has — so FLOOR at 2.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The atom names 'Oscar' as the prize, with winner outcome and an explicit count floor of 2 implied by 'multi-'. Not a saturation gradient — the user wants a binary pass at count>=2, not a ramp. Explicit-count pattern, FLOOR at 2.",
      "scoring_shape_label": "explicit count: 2",
      "scoring_mode": "floor",
      "scoring_mark": 2,
      "ceremonies": null,
      "award_names": ["Oscar"],
      "category_tags": null,
      "outcome": "winner",
      "years": null
    },
    "polarity": "positive"
  }
}
```

**Example: no-fire on reception-only framing ("critically acclaimed")**

```xml
<raw_query>critically acclaimed 90s thrillers</raw_query>
<overall_query_intention_exploration>The user wants well-reviewed thrillers from the 1990s. Three requirements: critical acclaim, 90s release, thriller genre.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The film should be critically acclaimed.</captured_meaning>
  <category_name>Award records</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>The film has earned critical acclaim.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>critically acclaimed</query_text>
  <description>Films that have earned critical acclaim.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>90s</query_text>
    <description>1990s release window.</description>
    <modifiers></modifiers>
  </fragment>
  <fragment>
    <query_text>thrillers</query_text>
    <description>Thriller genre.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The film should be critically acclaimed.",
      "relation_to_endpoint": "The Award endpoint resolves formal recognition records — ceremony / prize / category / outcome / count. 'Critically acclaimed' is a reception-quality judgment with no award vocabulary attached; none of the endpoint's axes map onto it.",
      "coverage_gaps": "The atom carries no ceremony, prize name, category, outcome, or count signal — only a generic critical-acclaim framing. This is Reception quality + superlative (Cat 25) territory, routed to semantic reception, not formal award records. Dispatch here looks to be a misroute; declining to fire is the correct outcome."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```

**Example: negated framing ("won no major awards")**

```xml
<raw_query>hidden gems that won no major awards</raw_query>
<overall_query_intention_exploration>The user wants overlooked, under-the-radar films — specifically ones that did NOT pick up wins at major award ceremonies. Two requirements: hidden-gem framing and an exclusion of major-award winners.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The film should not have won any major awards.</captured_meaning>
  <category_name>Award records</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>The film has not won at any major award ceremony.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>won no major awards</query_text>
  <description>Films that have no wins at major award ceremonies.</description>
  <modifiers>
    <modifier>
      <original_text>no</original_text>
      <effect>Negates the award-win clause — the user wants to exclude films that have won a major award.</effect>
      <type>POLARITY_MODIFIER</type>
    </modifier>
  </modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>hidden gems</query_text>
    <description>Overlooked / under-the-radar films.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The film has no wins at major award ceremonies. The parameter payload describes the positive concept (any major-ceremony win, FLOOR at 1); wrapper polarity=negative flips it to an exclusion at the dispatch layer.",
      "relation_to_endpoint": "Generic 'major awards' with no specific ceremony / prize named — leave ceremonies null and rely on the endpoint's default of 'all non-Razzie ceremonies,' which is the correct semantic for 'major'. Outcome is winner (the user is excluding winners, not nominees). FLOOR at 1: any single major-ceremony win makes the film fail the user's exclusion test.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The atom describes the concept 'has won at a major award ceremony' — emitted in positive form so the wrapper's polarity field handles the 'no' modifier. 'Major' maps to the default all-non-Razzie-ceremonies set; leave ceremonies null. Winner outcome, no prize name, no category, no count beyond the binary FLOOR at 1.",
      "scoring_shape_label": "specific filter, no count",
      "scoring_mode": "floor",
      "scoring_mark": 1,
      "ceremonies": null,
      "award_names": null,
      "category_tags": null,
      "outcome": "winner",
      "years": null
    },
    "polarity": "negative"
  }
}
```
