# Examples

These examples calibrate the Chronological no-fire discipline: the endpoint's `release_date` sub-object has no ordinal-position mode, so every well-formed ordinal atom no-fires on schema grounds, and every misrouted range / stature phrase no-fires on category grounds. Each case shows how to record the reason in `coverage_gaps` without approximating an ordinal as a range.

**Example: schema-gap no-fire on ordinal ("the newest Scorsese film")**

```xml
<raw_query>the newest Scorsese film</raw_query>
<overall_query_intention_exploration>The user wants the most recently released film directed by Martin Scorsese — a single film picked by release-date position within his filmography. Two requirements: Scorsese as director (handled by a sibling atom) and latest-release position.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The film is the most recent release in the scoped candidate set.</captured_meaning>
  <category_name>Chronological</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>The film holds the latest-release position by release date.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>the newest Scorsese film</query_text>
  <description>The most recently released film directed by Martin Scorsese.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The film holds the latest-release ordinal position within the scoped candidate set (Scorsese's filmography, handled by a sibling atom).",
      "relation_to_endpoint": "The release_date sub-object pins a date window via first_date / second_date / match_operation from {exact, before, after, between} — literal date comparators only. There is no sort-direction field, no top-N field, and no from-end position field. 'The newest' selects a single rank within an ordering, which no range predicate can express.",
      "coverage_gaps": "The atom is a well-formed ordinal-position request, but the endpoint's parameter surface supports only date ranges and point comparators. Approximating 'the newest' as 'after <some date>' would return every film after that cutoff rather than the single latest one, which is the wrong answer. Declining to fire is the faithful response."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```

**Example: schema-gap no-fire on ordinal ("earliest Kubrick")**

```xml
<raw_query>earliest Kubrick</raw_query>
<overall_query_intention_exploration>The user wants Stanley Kubrick's first film by release date. Two requirements: Kubrick as director (sibling atom) and the earliest-release position within his filmography.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The film is the earliest release in the scoped candidate set.</captured_meaning>
  <category_name>Chronological</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>The film holds the earliest-release position by release date.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>earliest Kubrick</query_text>
  <description>Stanley Kubrick's earliest film by release date.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The film holds the earliest-release ordinal position within the scoped candidate set (Kubrick's filmography, handled by a sibling atom).",
      "relation_to_endpoint": "The release_date sub-object expresses literal date comparators only — before / after / between / exact against concrete dates. No sort-and-pick mode. 'The earliest' names a position, not a window, and cannot be compiled into any first_date / second_date / match_operation triple.",
      "coverage_gaps": "Ordinal-position selection on release date sits outside the endpoint's parameter surface — only literal date comparators are expressible. Substituting a 'before <some year>' window would widen the request into a range and lose the single-position semantics. No-fire is correct."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```

**Example: no-fire on ordinal — endpoint shape doesn't hold it ("the latest Marvel movie")**

```xml
<raw_query>the latest Marvel movie</raw_query>
<overall_query_intention_exploration>The user wants the most recent entry in the Marvel franchise — a single film selected by release-date position within Marvel's output. Two requirements: Marvel franchise membership (sibling atom) and latest-release position.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The film is the most recent release in the scoped candidate set.</captured_meaning>
  <category_name>Chronological</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>The film holds the latest-release position by release date.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>the latest Marvel movie</query_text>
  <description>The most recently released film in the Marvel franchise.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The film holds the latest-release ordinal position within the scoped Marvel candidate set (franchise membership handled by a sibling atom).",
      "relation_to_endpoint": "The release_date sub-object takes literal date comparators against first_date / second_date. There is no mode for picking a single position by date. Recency as a window ('recent Marvel films') would be a Cat 10 range; 'the latest' is an ordinal position and has no predicate form here.",
      "coverage_gaps": "The parameter surface covers ranges and point comparators; ordinal rank selection sits outside its shape. Refusing to fabricate an 'after <recent date>' proxy is the correct response — a proxy would silently return every recent Marvel film rather than the single latest one."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```

**Example: category-misroute no-fire on range phrasing ("90s movies")**

```xml
<raw_query>90s movies with a strong female lead</raw_query>
<overall_query_intention_exploration>The user wants 1990s films built around a strong female protagonist. Two requirements: 1990s release window and female-lead framing.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The film was released in the 1990s.</captured_meaning>
  <category_name>Chronological</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>The film was released between 1990 and 1999.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>90s</query_text>
  <description>1990s release window.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>with a strong female lead</query_text>
    <description>Female-protagonist framing.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The film was released within the 1990s decade — a concrete date window, not a position within an ordering.",
      "relation_to_endpoint": "'90s' names a range of dates that a between-dates predicate handles directly. That shape is Structured metadata (Cat 10), which owns release_date as a range. Chronological owns ordinal position — 'the newest 90s movie' would be chronological; '90s movies' is a window.",
      "coverage_gaps": "Dispatch misroute: the atom is a range phrasing and belongs in Cat 10, where the metadata endpoint can express the between-1990-and-1999 predicate cleanly. Firing here would be redundant with Cat 10 at best and confused at worst."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```

**Example: category-misroute no-fire on superlative ("the best Scorsese")**

```xml
<raw_query>the best Scorsese</raw_query>
<overall_query_intention_exploration>The user wants the most acclaimed film in Martin Scorsese's filmography — a single film picked by reception stature, not by release date. Two requirements: Scorsese as director (sibling atom) and top-stature position.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The film is the most acclaimed release in the scoped candidate set.</captured_meaning>
  <category_name>Chronological</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>The film holds the top-stature position among Scorsese's films.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>the best Scorsese</query_text>
  <description>The most acclaimed film in Scorsese's filmography.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The film holds the top-stature position within Scorsese's filmography — a superlative on the acclaim axis, not the chronology axis.",
      "relation_to_endpoint": "Chronological selects by release-date ordering. 'The best' selects by reception / stature, which has no relationship to release date. The release_date sub-object cannot encode an acclaim superlative under any match_operation.",
      "coverage_gaps": "Dispatch misroute: 'best' is a Reception quality + superlative (Cat 25) framing, not a chronological one. 'Latest Scorsese' would be chronological; 'best Scorsese' is stature. Declining to fire is correct; encoding 'best' as any release_date predicate would flatten acclaim into date noise."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```
