# Examples

These calibrate the Keyword-vs-Semantic decision for character-type asks: when a canonical tag cleanly covers the archetype, when no tag exists and Semantic wins despite the keyword-first bias, and the no-fire shape for arc-framed asks that belong to Cat 21.

**Example: canonical tag covers the archetype → keyword wins cleanly**

```xml
<raw_query>anti-hero protagonist movies</raw_query>
<overall_query_intention_exploration>The user wants films whose protagonist is an anti-hero — morally ambiguous, outside conventional heroic framing. A clean character-type ask with no arc-trajectory framing.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The protagonist fits the anti-hero character type.</captured_meaning>
  <category_name>Character archetype</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies whose protagonist is an anti-hero.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>anti-hero protagonist movies</query_text>
  <description>Films with an anti-hero protagonist.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the anti-hero character type as the protagonist's defining trait.",
      "endpoint_coverage": "Keyword has a canonical registry member, ANTI_HERO, whose definition names morally-ambiguous protagonists outside conventional heroic framing — a direct cover for the archetype. Semantic's narrative_techniques.characterization_methods could also hold 'anti-hero' as a descriptor, but that is strictly weaker when a tag exists.",
      "best_endpoint": "keyword",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "Keyword wins on its own merits here — the ANTI_HERO tag's definition covers the archetype exactly, so the authoritative posting list delivers the right set. The tier-1 bias and the on-merits choice agree; this is not a close call that the bias had to break.",
  "endpoint_to_run": "keyword",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The query names 'anti-hero' — a morally-ambiguous protagonist archetype. Static character type, not an arc trajectory and not a specific named persona.",
      "candidate_shortlist": "ANTI_HERO definition names morally ambiguous protagonists operating outside conventional morality, matching the archetype exactly. No adjacent character-family member names this type specifically.",
      "classification": "anti_hero"
    },
    "polarity": "positive"
  }
}
```

**Example: ensemble-cast archetype → keyword**

```xml
<raw_query>movies with an ensemble cast, no single lead</raw_query>
<overall_query_intention_exploration>The user wants films where multiple characters share roughly equal narrative weight rather than one protagonist carrying the story. A character-configuration archetype.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The film has an ensemble cast with no single protagonist.</captured_meaning>
  <category_name>Character archetype</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies with an ensemble cast rather than a single lead.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies with an ensemble cast, no single lead</query_text>
  <description>Films built around an ensemble rather than one protagonist.</description>
  <modifiers>
    <modifier>
      <original_text>no single lead</original_text>
      <effect>reinforces the ensemble framing by ruling out single-protagonist films</effect>
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
      "aspect_description": "Identify the ensemble-cast configuration as the defining character-type pattern.",
      "endpoint_coverage": "Keyword has ENSEMBLE_CAST, whose definition names films with no single protagonist and multiple characters sharing narrative weight — a direct cover. Semantic narrative_techniques could also carry 'ensemble mosaic' under characterization_methods, but the tag is authoritative.",
      "best_endpoint": "keyword",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "Keyword wins on merits — the ENSEMBLE_CAST tag definition mirrors the requirement precisely. The bias reinforces a choice that would stand on its own even without it.",
  "endpoint_to_run": "keyword",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The query asks for ensemble films — no single protagonist, multiple characters of equal weight. A static cast-configuration archetype, reinforced by the 'no single lead' modifier.",
      "candidate_shortlist": "ENSEMBLE_CAST definition matches the ask directly. FEMALE_LEAD and ANTI_HERO describe protagonist traits and are structurally inapplicable. No other character-family member names the configuration.",
      "classification": "ensemble_cast"
    },
    "polarity": "positive"
  }
}
```

**Example: lower-tier-wins — no canonical tag covers the archetype → semantic**

```xml
<raw_query>movies with a lovable rogue</raw_query>
<overall_query_intention_exploration>The user wants films featuring a lovable-rogue character — a charming-outlaw archetype, morally flexible but sympathetic. Static character type, not a trajectory.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The film features a lovable-rogue character type.</captured_meaning>
  <category_name>Character archetype</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies featuring a lovable-rogue character.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies with a lovable rogue</query_text>
  <description>Films featuring a charming-outlaw type character.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the lovable-rogue archetype — a charming-outlaw character type that is morally flexible but sympathetic.",
      "endpoint_coverage": "Keyword has no registry member whose definition names this archetype. ANTI_HERO is the nearest neighbor but its definition centers on moral ambiguity generally, not on the specific charming-outlaw flavor the user named; picking it would broaden the lookup past the intent. Semantic's narrative_techniques.characterization_methods is exactly the sub-field designed to carry descriptor phrases for character type, and plot_events can carry a supporting signal for the charm-plus-caper situational pattern.",
      "best_endpoint": "semantic",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "The tier-1 bias would pick Keyword, but no CharacterTag member's definition actually names the lovable-rogue archetype — ANTI_HERO is adjacent, not a match. Semantic is clearly the better channel on its own merits: characterization_methods is purpose-built for descriptor phrases that the closed tag vocabulary does not cover. The bias does not force a weak tag when the lower-preference endpoint fits decisively better.",
  "endpoint_to_run": "semantic",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "qualifier_inventory": "lovable-rogue character type; morally-flexible-but-sympathetic descriptor; charming-outlaw situational flavor.",
      "space_queries": [
        {
          "carries_qualifiers": "narrative_techniques carries the archetype as a characterization descriptor — the target sub-field for uncanonized character types.",
          "space": "narrative_techniques",
          "weight": "central",
          "content": {
            "narrative_archetype": {"terms": []},
            "narrative_delivery": {"terms": []},
            "pov_perspective": {"terms": []},
            "characterization_methods": {"terms": ["lovable rogue", "charming outlaw", "morally flexible protagonist", "sympathetic scoundrel"]},
            "character_arcs": {"terms": []},
            "audience_character_perception": {"terms": []},
            "information_control": {"terms": []},
            "conflict_stakes_design": {"terms": []},
            "additional_narrative_devices": {"terms": []}
          }
        },
        {
          "carries_qualifiers": "plot_events carries the charm-plus-caper situational pattern that ingest-side plot summaries describe for lovable-rogue films.",
          "space": "plot_events",
          "weight": "supporting",
          "content": {
            "plot_summary": "a charming outlaw who pulls small-time schemes or heists while winning the audience's sympathy despite operating outside the law"
          }
        }
      ],
      "primary_vector": "narrative_techniques"
    },
    "polarity": "positive"
  }
}
```

**Example: another lower-tier-wins — manic pixie dream girl → semantic**

```xml
<raw_query>films with a manic pixie dream girl</raw_query>
<overall_query_intention_exploration>The user wants films featuring the manic-pixie-dream-girl archetype — a quirky, life-affirming female character whose function is to pull the male protagonist out of his shell. A static character-type pattern.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The film features a manic-pixie-dream-girl character.</captured_meaning>
  <category_name>Character archetype</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies featuring a manic-pixie-dream-girl character.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>films with a manic pixie dream girl</query_text>
  <description>Films featuring the manic-pixie-dream-girl archetype.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the manic-pixie-dream-girl archetype — a quirky life-affirming female character whose narrative function is to awaken the male protagonist.",
      "endpoint_coverage": "Keyword has no registry member covering this specific archetype. FEMALE_LEAD addresses gender framing, not this character type, and would drastically over-broaden the set. Semantic narrative_techniques.characterization_methods carries the descriptor directly, and audience_character_perception holds the 'viewed through the protagonist's gaze' framing that defines the archetype.",
      "best_endpoint": "semantic",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "No CharacterTag member names this archetype — FEMALE_LEAD is a structural adjacency, not a match, and picking it to honor the bias would return every female-led film rather than MPDG films. Semantic wins on its own merits because characterization_methods is the channel built for descriptor-only archetypes. The bias is not a veto when the lower-preference endpoint fits decisively better.",
  "endpoint_to_run": "semantic",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "qualifier_inventory": "manic-pixie-dream-girl character type; quirky life-affirming female figure; narrative function as catalyst for a withdrawn male protagonist.",
      "space_queries": [
        {
          "carries_qualifiers": "narrative_techniques carries the archetype as a characterization descriptor, plus the audience-perception framing that defines the trope (viewed through the male lead's eyes).",
          "space": "narrative_techniques",
          "weight": "central",
          "content": {
            "narrative_archetype": {"terms": []},
            "narrative_delivery": {"terms": []},
            "pov_perspective": {"terms": []},
            "characterization_methods": {"terms": ["manic pixie dream girl", "quirky free-spirited female catalyst", "life-affirming love interest"]},
            "character_arcs": {"terms": []},
            "audience_character_perception": {"terms": ["viewed through male protagonist's gaze", "idealized love interest framing"]},
            "information_control": {"terms": []},
            "conflict_stakes_design": {"terms": []},
            "additional_narrative_devices": {"terms": []}
          }
        }
      ],
      "primary_vector": "narrative_techniques"
    },
    "polarity": "positive"
  }
}
```

**Example: arc-trajectory framing belongs to Cat 21 → no-fire**

```xml
<raw_query>movies with a redemption arc</raw_query>
<overall_query_intention_exploration>The user wants films built around a redemption arc — a character-change trajectory, not a static character type. Sits in Kind-of-story territory.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films structured around a redemption-arc trajectory.</captured_meaning>
  <category_name>Character archetype</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies whose protagonist follows a redemption arc.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies with a redemption arc</query_text>
  <description>Films built around a redemption-arc trajectory.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Find films whose protagonist follows a redemption-arc trajectory.",
      "endpoint_coverage": "Neither candidate is the right channel. Keyword's CharacterTag family covers static types (ANTI_HERO, FEMALE_LEAD, ENSEMBLE_CAST) — 'redemption arc' is a change trajectory, not a type, so no member's definition fits. Semantic's characterization_methods is designed for type descriptors, not arc shapes; arc trajectories belong to plot_analysis.character_arcs, which is Cat 21's fallback space, not this category's. Firing here would either mispick a static-type tag or populate the wrong semantic sub-field.",
      "best_endpoint": "None",
      "best_endpoint_gaps": "The requirement is an arc trajectory, which is Cat 21's (Kind of story) territory, not Cat 7's. The correct handler for this atom is the Kind-of-story handler, whose fallback lives on plot_analysis.character_arcs."
    }
  ],
  "performance_vs_bias_analysis": "The tier-1 bias does not force a pick when neither candidate fits the requirement. This atom was misrouted to Character archetype — it names an arc shape, not a static type — so the correct response is no-fire rather than picking a weak member or polluting a wrong semantic sub-field.",
  "endpoint_to_run": "None",
  "endpoint_parameters": null
}
```
