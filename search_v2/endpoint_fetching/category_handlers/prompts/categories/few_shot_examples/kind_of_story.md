# Examples

These calibrate the Keyword-vs-Semantic decision for thematic / arc asks: the clean Keyword wins when a registry member covers a binary-framed theme, the spectrum escape that bypasses Keyword even when a tag exists, the lower-tier win for uncanonized themes, and the no-fire shape for vague asks.

**Example: binary-framed theme with a registry member → keyword wins**

```xml
<raw_query>coming-of-age stories about self-acceptance</raw_query>
<overall_query_intention_exploration>The user wants films built around the coming-of-age story pattern, with self-acceptance as the thematic thrust. Binary — the coming-of-age shape is a defining feature, not a flavor.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films that are coming-of-age stories about self-acceptance.</captured_meaning>
  <category_name>Kind of story / thematic archetype</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies whose overarching kind of story is coming-of-age centered on self-acceptance.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>coming-of-age stories about self-acceptance</query_text>
  <description>Films built on the coming-of-age pattern, themed around self-acceptance.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the coming-of-age story shape as the overarching kind of story, with self-acceptance as the thematic thrust.",
      "endpoint_coverage": "Keyword has COMING_OF_AGE, whose definition names a young person's emotional growth and transition toward adulthood — the coming-of-age story pattern directly. The self-acceptance flavor narrows within that set but the story-pattern axis is what routes here. Semantic's plot_analysis.character_arcs could also carry 'coming-of-age' as an arc term, but with a direct registry hit the tag is authoritative.",
      "best_endpoint": "keyword",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "Framing is binary — 'coming-of-age stories' names the kind of story as a defining feature, not as a flavor. The spectrum escape does not apply. Keyword wins on its own merits: the COMING_OF_AGE tag's definition mirrors the story-pattern axis of the ask. The tier-1 bias and the on-merits choice agree here.",
  "endpoint_to_run": "keyword",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The query names the coming-of-age story shape — emotional growth and transition toward adulthood — with self-acceptance as the inner theme. The story-pattern axis is the atom this handler is routing.",
      "candidate_shortlist": "COMING_OF_AGE definition covers the story pattern directly. No other keyword family member names a life-stage story arc. Source-material and concept-tag entries do not address story shape.",
      "classification": "coming_of_age"
    },
    "polarity": "positive"
  }
}
```

**Example: binary-framed conflict archetype with a registry member → keyword wins**

```xml
<raw_query>man-vs-nature survival stories</raw_query>
<overall_query_intention_exploration>The user wants films whose overarching story is a survival ordeal against the natural world. Binary framing — the survival-vs-nature shape is the defining story.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films that are survival stories framed as man-vs-nature.</captured_meaning>
  <category_name>Kind of story / thematic archetype</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies whose overarching kind of story is a man-vs-nature survival ordeal.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>man-vs-nature survival stories</query_text>
  <description>Survival films framed as man against the natural world.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the man-vs-nature survival story shape as the overarching kind of story.",
      "endpoint_coverage": "Keyword has SURVIVAL, whose definition names characters struggling to stay alive in extreme life-threatening conditions — the survival story shape directly. The man-vs-nature conflict framing is the most common instantiation of that pattern, so the tag covers the ask. Semantic plot_analysis.conflict_type could also carry 'man vs nature', but the direct registry hit on the story-pattern axis is authoritative.",
      "best_endpoint": "keyword",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "Framing is binary — the user wants survival-against-nature as the defining story, not as a flavor. Spectrum escape does not apply. Keyword wins on merits because the SURVIVAL tag definition covers the story shape, and the tier-1 bias reinforces a choice that would stand on its own.",
  "endpoint_to_run": "keyword",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The query names a survival story shape with man-vs-nature as the conflict axis — life-threatening struggle against the natural world as the overarching kind of story.",
      "candidate_shortlist": "SURVIVAL definition covers the story pattern. Adventure-family members (JUNGLE_ADVENTURE, MOUNTAIN_ADVENTURE, SEA_ADVENTURE) are setting-scoped and narrower than the requirement; the user did not cite a specific terrain. SURVIVAL is the right breadth.",
      "classification": "survival"
    },
    "polarity": "positive"
  }
}
```

**Example: spectrum escape — gradient framing bypasses Keyword**

```xml
<raw_query>movies that are kind of about grief but not too heavy</raw_query>
<overall_query_intention_exploration>The user wants films that carry grief as a thematic thread present in the story but not as the dominant kind of story. The 'kind of about' phrasing signals a gradient, not a binary theme commitment.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films that carry grief as a thematic thread, present in a measured way rather than as the defining story.</captured_meaning>
  <category_name>Kind of story / thematic archetype</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies with grief as a thematic thread, present in gradient rather than as the defining theme.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>kind of about grief</query_text>
  <description>Films that carry grief as a thematic thread, in a measured degree.</description>
  <modifiers>
    <modifier>
      <original_text>kind of</original_text>
      <effect>weakens the theme commitment from defining to partial / flavor-level</effect>
      <type>POLARITY_MODIFIER</type>
    </modifier>
  </modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>but not too heavy</query_text>
    <description>Ruling out emotionally heavy tone.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Capture grief as a thematic thread, present as a graded signal rather than as the defining kind of story.",
      "endpoint_coverage": "Keyword's posting-list semantics are binary: a grief tag, if one existed, would either include only films where grief is front-and-center or broaden to every film tagged with grief, neither of which matches a gradient ask. (The registry also has no grief member, reinforcing the mismatch.) Semantic is purpose-built for gradient matches — plot_analysis.thematic_concepts carries abstract theme terms and similarity to 'grief', 'mourning', 'loss' produces a naturally graded score that ranks partial-presence films appropriately.",
      "best_endpoint": "semantic",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "The 'kind of' modifier makes this a spectrum-framed ask. The spectrum escape bypasses the tier-1 Keyword preference: even if the registry carried a grief member, a binary posting-list membership is the wrong instrument for a gradient intent. Semantic's similarity-scored match against thematic_concepts is the correct channel on merit, and the bias is explicitly ignored because of the framing — not because Keyword lost a close call.",
  "endpoint_to_run": "semantic",
  "endpoint_parameters": {
    "match_mode": "trait",
    "parameters": {
      "qualifier_inventory": "grief as a thematic thread; measured / gradient degree of presence rather than a defining feature.",
      "space_queries": [
        {
          "carries_qualifiers": "plot_analysis carries the grief theme itself — thematic_concepts is the sub-field purpose-built for abstract theme terms, and the gradient framing lands naturally on similarity scoring against those terms.",
          "space": "plot_analysis",
          "weight": "central",
          "content": {
            "elevator_pitch": null,
            "plot_overview": null,
            "genre_signatures": [],
            "conflict_type": [],
            "thematic_concepts": ["grief", "mourning", "loss", "bereavement"],
            "character_arcs": []
          }
        }
      ],
      "primary_vector": "plot_analysis"
    },
    "polarity": "positive"
  }
}
```

**Example: spectrum escape — "leans redemptive" bypasses Keyword even with tag coverage hypothetically present**

```xml
<raw_query>something that leans redemptive, with a character who earns their second chance</raw_query>
<overall_query_intention_exploration>The user wants films that tilt toward a redemption arc — present as a directional lean, not a full commitment. The 'leans' framing is gradient.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films that tilt toward a redemption arc, with a character earning a second chance as the thematic shape.</captured_meaning>
  <category_name>Kind of story / thematic archetype</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies with a redemption-leaning arc where the protagonist earns a second chance.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>leans redemptive, with a character who earns their second chance</query_text>
  <description>Films that tilt toward a redemption arc centered on earning a second chance.</description>
  <modifiers>
    <modifier>
      <original_text>leans</original_text>
      <effect>weakens the theme commitment from defining to directional</effect>
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
      "aspect_description": "Capture the redemption-arc lean — a character earning a second chance as the trajectory — expressed as a directional preference rather than a defining feature.",
      "endpoint_coverage": "Keyword's posting-list shape is binary: even a hypothetical REDEMPTION tag would either hit only films where the arc is the defining story or broaden to any film tagged with it, which is the wrong shape for a 'leans' intent. Semantic's plot_analysis.character_arcs carries arc-pattern terms directly, and similarity against 'redemption arc', 'earned second chance', 'atonement' produces a graded score matching the user's directional framing.",
      "best_endpoint": "semantic",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "'Leans' is a spectrum cue — the user is asking for tilt, not commitment. The spectrum escape bypasses the tier-1 Keyword preference on framing grounds alone. A binary tag membership cannot rank 'somewhat redemptive' films between 'fully redemptive' and 'not redemptive'; Semantic's graded similarity can. The bias is overridden by the framing, not by a close call on vocabulary.",
  "endpoint_to_run": "semantic",
  "endpoint_parameters": {
    "match_mode": "trait",
    "parameters": {
      "qualifier_inventory": "redemption-arc trajectory; earning a second chance as the specific shape; gradient / directional framing rather than a defining commitment.",
      "space_queries": [
        {
          "carries_qualifiers": "plot_analysis.character_arcs is the native home for arc-pattern terms; thematic_concepts holds the surrounding moral-trajectory vocabulary that rounds out the gradient match.",
          "space": "plot_analysis",
          "weight": "central",
          "content": {
            "elevator_pitch": null,
            "plot_overview": null,
            "genre_signatures": [],
            "conflict_type": [],
            "thematic_concepts": ["redemption", "atonement", "second chances", "moral recovery"],
            "character_arcs": ["redemption arc", "fall and rise", "earned second chance"]
          }
        }
      ],
      "primary_vector": "plot_analysis"
    },
    "polarity": "positive"
  }
}
```

**Example: uncanonized theme, binary framing → lower-tier wins**

```xml
<raw_query>stories about forgiveness</raw_query>
<overall_query_intention_exploration>The user wants films whose overarching story is about forgiveness — a defining thematic commitment. Binary framing.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films whose overarching story is about forgiveness.</captured_meaning>
  <category_name>Kind of story / thematic archetype</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies whose overarching kind of story is about forgiveness.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>stories about forgiveness</query_text>
  <description>Films built around forgiveness as the overarching theme.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify forgiveness as the overarching thematic concept of the story.",
      "endpoint_coverage": "Keyword has no registry member whose definition names forgiveness as a theme — the closest adjacencies (coming-of-age, drama-family tags) address different axes and would broaden the result past the intent. Semantic plot_analysis.thematic_concepts is exactly the sub-field built to carry abstract theme terms the closed vocabulary does not absorb; elevator_pitch can carry the capsule framing when forgiveness is the one-sentence hook.",
      "best_endpoint": "semantic",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "Framing is binary — the user wants forgiveness as the defining story, so the spectrum escape does not apply and the decision is a straight tier-1-vs-tier-2 call. The registry simply does not cover this theme, so Keyword cannot produce an authoritative hit. Semantic wins decisively on its own merits: thematic_concepts is purpose-built for exactly this case. The tier-1 bias is not a veto when no canonical member fits.",
  "endpoint_to_run": "semantic",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "qualifier_inventory": "forgiveness as the overarching thematic concept; binary framing — forgiveness is the defining story, not a flavor.",
      "space_queries": [
        {
          "carries_qualifiers": "plot_analysis carries the theme — thematic_concepts holds 'forgiveness' and its neighbors, and elevator_pitch holds the capsule framing when forgiveness is the story's one-sentence hook.",
          "space": "plot_analysis",
          "weight": "central",
          "content": {
            "elevator_pitch": "a story centered on a character coming to grant or seek forgiveness for a deep wrong",
            "plot_overview": null,
            "genre_signatures": [],
            "conflict_type": [],
            "thematic_concepts": ["forgiveness", "reconciliation", "atonement", "letting go of resentment"],
            "character_arcs": []
          }
        }
      ],
      "primary_vector": "plot_analysis"
    },
    "polarity": "positive"
  }
}
```

**Example: too vague to target → no-fire**

```xml
<raw_query>movies with deep themes that make you think</raw_query>
<overall_query_intention_exploration>The user wants films with thematic depth but does not name any specific theme or arc. Too underspecified to land on concrete parameters.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films with unspecified deep themes.</captured_meaning>
  <category_name>Kind of story / thematic archetype</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies whose overarching kind of story has unspecified thematic depth.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies with deep themes that make you think</query_text>
  <description>Films with thematic depth, unspecified.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Capture unspecified thematic depth as the kind-of-story axis.",
      "endpoint_coverage": "Neither candidate can target this cleanly. Keyword needs a specific theme named — 'deep themes' does not point at any registry member's definition; any pick would be arbitrary. Semantic needs a concrete theme or arc term for thematic_concepts / character_arcs to carry real signal — filling those fields with abstractions like 'deep themes' or 'thought-provoking' would pollute the vector without naming any actual concept a matching movie's ingest text would express. 'Thought-provoking' is a viewer-experience / cognitive-load signal, which is a different handler's territory.",
      "best_endpoint": "None",
      "best_endpoint_gaps": "The requirement names no specific theme, arc, or conflict type, so no endpoint can produce a well-formed payload on the kind-of-story axis. The cognitive-load framing belongs to the viewer-experience handler, not here."
    }
  ],
  "performance_vs_bias_analysis": "Neither endpoint fits: the ask is too underspecified on the kind-of-story axis to name a theme or arc concretely. The tier-1 bias does not force a pick when no candidate can populate a real payload. No-fire is the correct response; the thought-provoking framing will be handled by its own category's handler.",
  "endpoint_to_run": "None",
  "endpoint_parameters": null
}
```
