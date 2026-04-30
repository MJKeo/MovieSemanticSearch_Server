# Examples

These examples calibrate a clean author-adaptation fire on `plot_events`, a household-name case that pulls `reception` as a supporting entry, and two no-fire boundaries: an author-less adaptation-class phrase (Cat 5's territory) and an indexed-credit name (Cat 1's territory).

**Example: clean fire on a named source author**

```xml
<raw_query>Stephen King adaptations</raw_query>
<overall_query_intention_exploration>The user wants films adapted from Stephen King's novels and stories. The adaptation-medium side of this ask (novel / short-story adaptation as a class) is carried by a co-emitted Cat 5 atom; this Cat 30 atom is specifically the "which author" slice — match movies whose synopsis prose names Stephen King as the source author.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films whose source material was written by Stephen King.</captured_meaning>
  <category_name>Source-material author / origin-work creator</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies adapted from works written by Stephen King.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>Stephen King adaptations</query_text>
  <description>Films adapted from Stephen King's source works.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>Stephen King adaptations</query_text>
    <description>Adaptation-class atom (novel / short-story source) co-emitted to Cat 5.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Find films whose ingest-side synopsis prose names Stephen King as the author of the source work the movie adapts.",
      "relation_to_endpoint": "plot_events synopsis prose routinely names the source author when introducing an adaptation ('based on Stephen King's novel'). A short plot_summary body that names Stephen King alongside adaptation framing matches the same register as a matching movie's ingest text. The adaptation-medium side (novel / short-story class) is handled by the co-emitted Cat 5 atom, not by this payload.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "qualifier_inventory": "One atom: Stephen King as the source-material author of the film. The adaptation medium itself (novel / short-story) is a separate Cat 5 slice and is not encoded here.",
      "space_queries": [
        {
          "carries_qualifiers": "plot_events carries synopsis prose where the source author is named when a film is an adaptation; 'based on Stephen King's novel' lands directly in this register.",
          "space": "plot_events",
          "weight": "central",
          "content": {
            "plot_summary": "a film adapted from a work by stephen king, based on his novel or short story, telling a story he originally wrote."
          }
        }
      ],
      "primary_vector": "plot_events"
    },
    "polarity": "positive"
  }
}
```

**Example: household-name author — plot_events primary, reception supporting**

```xml
<raw_query>Jane Austen movies</raw_query>
<overall_query_intention_exploration>The user wants films adapted from Jane Austen's novels. Austen is a household-name author whose name surfaces both in synopsis prose ('based on Jane Austen's novel') and in reception text (reviews routinely frame Austen adaptations as such).</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films adapted from Jane Austen's novels.</captured_meaning>
  <category_name>Source-material author / origin-work creator</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies adapted from works written by Jane Austen.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>Jane Austen movies</query_text>
  <description>Films adapted from Jane Austen's source works.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>Jane Austen movies</query_text>
    <description>Adaptation-class atom (novel source) co-emitted to Cat 5.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Find films whose source work was written by Jane Austen, using ingest-side prose that names her as the adapted author.",
      "relation_to_endpoint": "plot_events synopsis prose names the source author on adaptations; reception prose for Austen adaptations routinely cites her by name when describing the film as a period-adaptation of her novel. Both surfaces carry the author attribution; plot_events is primary and reception rounds it out.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "qualifier_inventory": "One atom: Jane Austen as the source-material author. Adaptation medium (novel) is carried by the co-emitted Cat 5 atom and not encoded here.",
      "space_queries": [
        {
          "carries_qualifiers": "plot_events synopsis prose names the source author when introducing an adaptation; a body naming Jane Austen and her novel lands in that register.",
          "space": "plot_events",
          "weight": "central",
          "content": {
            "plot_summary": "a film adapted from a novel by jane austen, telling a story she originally wrote — a period romance from her body of work."
          }
        },
        {
          "carries_qualifiers": "reception prose for Austen adaptations routinely names her by name when describing the film as an adaptation of her novel.",
          "space": "reception",
          "weight": "supporting",
          "content": {
            "reception_summary": "reviewed as an adaptation of a jane austen novel, with critics comparing the film to its source work."
          }
        }
      ],
      "primary_vector": "plot_events"
    },
    "polarity": "positive"
  }
}
```

**Example: clean fire on a long-tail genre-fiction author**

```xml
<raw_query>Philip K. Dick stories</raw_query>
<overall_query_intention_exploration>The user wants films adapted from Philip K. Dick's stories and novels. A specific source-material author, typically surfaced in plot_events synopsis prose when a film cites him as the source.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films adapted from Philip K. Dick's stories or novels.</captured_meaning>
  <category_name>Source-material author / origin-work creator</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies adapted from works written by Philip K. Dick.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>Philip K. Dick stories</query_text>
  <description>Films adapted from Philip K. Dick's source works.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>Philip K. Dick stories</query_text>
    <description>Adaptation-class atom (short-story / novel source) co-emitted to Cat 5.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Find films whose source story or novel was written by Philip K. Dick, matching on ingest-side synopsis prose that names him as the adapted author.",
      "relation_to_endpoint": "plot_events synopsis prose cites the source author when introducing an adaptation; films adapted from Dick's stories routinely surface his name in their plot summary. The adaptation-class side (short-story / novel source) is handled by the co-emitted Cat 5 atom.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "qualifier_inventory": "One atom: Philip K. Dick as the source-material author of the film. The adaptation medium is a separate Cat 5 slice.",
      "space_queries": [
        {
          "carries_qualifiers": "plot_events synopsis prose names the source author for adaptations; a body citing Philip K. Dick and his story as the source lands in that register.",
          "space": "plot_events",
          "weight": "central",
          "content": {
            "plot_summary": "a film adapted from a story or novel by philip k. dick, based on his work and telling a story he originally wrote."
          }
        }
      ],
      "primary_vector": "plot_events"
    },
    "polarity": "positive"
  }
}
```

**Example: no-fire on an author-less adaptation-class phrase (Cat 5 boundary)**

```xml
<raw_query>novels adapted to film</raw_query>
<overall_query_intention_exploration>The user wants film adaptations of novels as a category — no specific author is named. The adaptation-medium class (novel source) is Cat 5's territory, not Cat 30.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films adapted from novels whose author the user has not specified.</captured_meaning>
  <category_name>Source-material author / origin-work creator</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies adapted from a novel, with the source author unspecified.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>novels adapted to film</query_text>
  <description>Films adapted from novels — adaptation-medium category.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The requirement names an adaptation medium (novel source) without naming any specific source-material author.",
      "relation_to_endpoint": "plot_events synopsis prose is useful for author attribution only when the author is named. Without a specific author name, there is nothing discriminative to embed — a body saying 'adapted from a novel' would drift into matching every novel adaptation indiscriminately, which is exactly what Cat 5's NOVEL_ADAPTATION keyword tag already does with precision.",
      "coverage_gaps": "Upstream dispatch routed an author-less adaptation-class phrase here. No author is named, so there is no ingest-side author attribution to match against. The correct category is Cat 5 (Adaptation source flag)."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```

**Example: no-fire on an indexed-credit name (Cat 1 boundary)**

```xml
<raw_query>Tom Hanks movies</raw_query>
<overall_query_intention_exploration>The user wants films starring Tom Hanks. He is an actor — an indexed film credit — not the author of any source work. This is Cat 1's territory, not Cat 30's.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films whose source material was written by Tom Hanks.</captured_meaning>
  <category_name>Source-material author / origin-work creator</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies adapted from works written by Tom Hanks.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>Tom Hanks movies</query_text>
  <description>Films associated with Tom Hanks.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The named person is an actor — an indexed film credit — not a source-material author.",
      "relation_to_endpoint": "plot_events synopsis prose does not carry actor attribution in the register this category targets (source-work authorship). Embedding Tom Hanks as an adapted author would match nothing, because he is not the source author of the films he stars in; those credits are indexed in Cat 1's actor posting table and would be resolved there with precision.",
      "coverage_gaps": "Upstream dispatch misread an indexed film-credit name as a source-material author. The correct category is Cat 1 (Credit + title text). Firing here would fabricate an author attribution the user did not make."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```
