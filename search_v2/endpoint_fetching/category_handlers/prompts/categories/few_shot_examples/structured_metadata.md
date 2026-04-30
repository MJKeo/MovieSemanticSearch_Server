# Examples

These examples calibrate the column-selection decision for Structured metadata, the literal-spec shape for the chosen sub-object, and the no-fire discipline against Chronological and Reception quality misroutes. Each case shows how the atomic rewrite drives the `target_attribute` choice and the tightest literal predicate on that one column.

**Example: release-date range fire (90s movies)**

```xml
<raw_query>90s action movies with a strong female lead</raw_query>
<overall_query_intention_exploration>The user wants 1990s action films built around a strong female protagonist. Three requirements: 1990s release window, action genre, female-lead framing.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The film was released in the 1990s.</captured_meaning>
  <category_name>Structured metadata</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>The film was released between 1990 and 1999.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>90s</query_text>
  <description>1990s release window.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>action movies</query_text>
    <description>Action genre.</description>
    <modifiers></modifiers>
  </fragment>
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
      "aspect_description": "The film was released in the 1990s — a named era that resolves to a concrete date window.",
      "relation_to_endpoint": "The release_date column supports a between-dates predicate. '90s' is a range framing, not an ordinal position, so it belongs here rather than Chronological. Tightest literal spec is 1990-01-01 through 1989-12-31 inclusive — match_operation between.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "constraint_phrases": ["90s"],
      "target_attribute": "release_date",
      "value_intent_label": "1990s decade window",
      "release_date": {
        "first_date": "1990-01-01",
        "match_operation": "between",
        "second_date": "1999-12-31"
      },
      "runtime": null,
      "maturity_rating": null,
      "streaming": null,
      "audio_language": null,
      "country_of_origin": null,
      "budget_scale": null,
      "box_office": null,
      "popularity": null,
      "reception": null
    },
    "polarity": "positive"
  }
}
```

**Example: runtime fire (under 2 hours)**

```xml
<raw_query>funny movies under 2 hours for a weeknight</raw_query>
<overall_query_intention_exploration>The user wants short, comedic films for a weeknight viewing. Two requirements: comedy genre and a runtime ceiling of roughly two hours, framed by a weeknight-viewing occasion.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The film is less than two hours long.</captured_meaning>
  <category_name>Structured metadata</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>The film has a runtime less than 120 minutes.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>under 2 hours</query_text>
  <description>Runtime ceiling of two hours.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>funny movies</query_text>
    <description>Comedy genre.</description>
    <modifiers></modifiers>
  </fragment>
  <fragment>
    <query_text>for a weeknight</query_text>
    <description>Weeknight viewing occasion.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The film has a runtime strictly less than 120 minutes.",
      "relation_to_endpoint": "The runtime column supports a less_than predicate in minutes. 'Under 2 hours' is an explicit ceiling; converts directly to less_than 120.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "constraint_phrases": ["under 2 hours"],
      "target_attribute": "runtime",
      "value_intent_label": "runtime ceiling at two hours",
      "release_date": null,
      "runtime": {
        "first_value": 120,
        "match_operation": "less_than",
        "second_value": null
      },
      "maturity_rating": null,
      "streaming": null,
      "audio_language": null,
      "country_of_origin": null,
      "budget_scale": null,
      "box_office": null,
      "popularity": null,
      "reception": null
    },
    "polarity": "positive"
  }
}
```

**Example: streaming platform fire (on Netflix)**

```xml
<raw_query>sci-fi thrillers on Netflix</raw_query>
<overall_query_intention_exploration>The user wants sci-fi thrillers currently available on Netflix. Two requirements: sci-fi thriller genre blend and Netflix availability.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The film is available on Netflix.</captured_meaning>
  <category_name>Structured metadata</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>The film is available to watch on Netflix.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>on Netflix</query_text>
  <description>Availability on the Netflix streaming service.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>sci-fi thrillers</query_text>
    <description>Sci-fi thriller genre blend.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The film is available on Netflix.",
      "relation_to_endpoint": "The streaming column supports a services-list filter. 'On Netflix' names a single tracked service with no access-type qualifier; services=[netflix] with preferred_access_type=null is the tightest spec.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "constraint_phrases": ["on Netflix"],
      "target_attribute": "streaming",
      "value_intent_label": "available on Netflix",
      "release_date": null,
      "runtime": null,
      "maturity_rating": null,
      "streaming": {
        "services": ["netflix"],
        "preferred_access_type": null
      },
      "audio_language": null,
      "country_of_origin": null,
      "budget_scale": null,
      "box_office": null,
      "popularity": null,
      "reception": null
    },
    "polarity": "positive"
  }
}
```

**Example: maturity rating alone fire (PG-13)**

```xml
<raw_query>PG-13 horror movies</raw_query>
<overall_query_intention_exploration>The user wants horror films carrying a PG-13 rating. Two requirements: horror genre and PG-13 as the maturity rating of the film itself.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The film is rated PG-13.</captured_meaning>
  <category_name>Structured metadata</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>The film has a maturity rating of exactly PG-13.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>PG-13</query_text>
  <description>PG-13 maturity rating on the film.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>horror movies</query_text>
    <description>Horror genre.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The film's maturity rating is PG-13, named as a rating rather than used as a content-sensitivity ceiling.",
      "relation_to_endpoint": "The maturity_rating column supports a rating + direction pair. The atom names PG-13 with no direction word ('or lower', 'at least'), so the tightest spec is rating=pg-13 with match_operation=exact. Audience-framing (Cat 17) phrases like 'family-friendly' would flip to a direction; this atom is the rating alone.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "constraint_phrases": ["PG-13"],
      "target_attribute": "maturity_rating",
      "value_intent_label": "exact rating of PG-13",
      "release_date": null,
      "runtime": null,
      "maturity_rating": {
        "rating": "pg-13",
        "match_operation": "exact"
      },
      "streaming": null,
      "audio_language": null,
      "country_of_origin": null,
      "budget_scale": null,
      "box_office": null,
      "popularity": null,
      "reception": null
    },
    "polarity": "positive"
  }
}
```

**Example: no-fire on ordinal chronology ("the newest Scorsese")**

```xml
<raw_query>the newest Scorsese movie</raw_query>
<overall_query_intention_exploration>The user wants the most recently released film directed by Martin Scorsese — a single film selected by release-date position, not a window of films.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The film is the most recent release in Martin Scorsese's filmography.</captured_meaning>
  <category_name>Structured metadata</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>The film is the latest release by release date among Martin Scorsese's films.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>the newest Scorsese movie</query_text>
  <description>The most recently released Scorsese film.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The film holds the latest-release position within Scorsese's filmography.",
      "relation_to_endpoint": "The release_date column supports ranges and comparators against literal dates, not ordinal selection within a candidate set. 'The newest' picks a single position rather than naming a window — no literal date predicate captures it. This is Chronological (Cat 32) territory, which routes ordinal selection through the metadata endpoint under a different handler shape.",
      "coverage_gaps": "The atom is an ordinal-position request, not a date-range request. No first_date / second_date / comparator spec pins 'the latest one' as a predicate. Dispatch looks to be a misroute against the range-vs-position boundary; declining to fire is correct."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```

**Example: no-fire on superlative stature ("best horror of the 80s")**

```xml
<raw_query>best horror of the 80s</raw_query>
<overall_query_intention_exploration>The user wants the highest-stature, most-acclaimed horror films from the 1980s. Two requirements: horror genre scoped to the 1980s, and a superlative 'best' framing on reception / stature.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The film is among the best-regarded horror films of the 1980s.</captured_meaning>
  <category_name>Structured metadata</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>The film has top-tier stature as a horror film from the 1980s.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>best horror of the 80s</query_text>
  <description>Superlative stature — the best-regarded horror films of the 1980s.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The film carries superlative stature as a horror film from the 1980s — a qualitative 'best of' judgment, not a scalar threshold on the reception column.",
      "relation_to_endpoint": "The reception column supports a binary direction (well_received vs. poorly_received) that stands in for scalar reception as an attribute. A 'best of' superlative framed against a scoped cohort (horror, 80s) is a stature / canon-shape judgment that does not reduce to a single direction flag — it needs semantic reception reasoning. The 80s-era window is a separate atom handled by release_date in another call; it is not this atom's payload.",
      "coverage_gaps": "The atom is a standalone-vs-superlative misroute: 'best' is a stature framing, not a numeric reception score. This is Reception quality + superlative (Cat 25) territory on the semantic channel. Declining to fire here is correct — forcing reception=well_received would flatten the superlative into a generic 'well-reviewed' filter and lose the 'best of' meaning."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```
