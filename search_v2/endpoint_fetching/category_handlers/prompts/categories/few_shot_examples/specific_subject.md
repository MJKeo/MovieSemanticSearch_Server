# Examples

These examples calibrate the keyword-first bias for canonical motifs, the lower-tier-wins case when the registry has no coverage, the spectrum-framing flip to Semantic, and the no-fire shape.

**Example: canonical motif with a registry tag → Keyword**

```xml
<raw_query>zombie movies</raw_query>
<overall_query_intention_exploration>The user wants films featuring zombies as a central element. A clean, binary subject-presence ask — no gradient framing, no sibling qualifiers.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films in which zombies are a central element of the story.</captured_meaning>
  <category_name>Specific subject / element / motif</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies that feature zombies as a central subject of the story.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>zombie movies</query_text>
  <description>Films centered on zombies.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Find films where zombies are a central element of the story.",
      "endpoint_coverage": "Keyword has a canonical registry member for zombies (ZOMBIE_HORROR) that resolves directly onto the per-movie tag posting list; Semantic plot_events could also match via literal 'zombie' mentions in plot summaries, but it lacks the closed-vocabulary precision of the tag.",
      "best_endpoint": "keyword",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "The registry carries ZOMBIE_HORROR as a dedicated concept. The framing is binary membership, not graded — so the canonical tag is both more precise and more recall-complete than Semantic plot_events matching on the word. Keyword wins on its own merits here; the keyword-first bias merely reinforces the pick.",
  "endpoint_to_run": "keyword",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The requirement names zombies as the central motif. The registry has a dedicated member for zombie-centered stories under the horror family.",
      "candidate_shortlist": "ZOMBIE_HORROR directly names zombie stories. No broader horror member fits because the user cited the specific motif; no narrower option exists.",
      "classification": "ZOMBIE_HORROR"
    },
    "polarity": "positive"
  }
}
```

**Example: another canonical-tag win — adaptation / subject flag**

```xml
<raw_query>biopics</raw_query>
<overall_query_intention_exploration>The user wants biographical films — dramatizations of real people's lives. A binary category ask, not a specific named person.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films that dramatize the life of a real person.</captured_meaning>
  <category_name>Specific subject / element / motif</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies that depict the life of a real person as their subject.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>biopics</query_text>
  <description>Biographical films about real people.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Find films that dramatize the life of a real person.",
      "endpoint_coverage": "Keyword registry has BIOGRAPHY as a dedicated member for real-person life-dramatizations. Semantic plot_events could match plot summaries that read biographical, but without the tag it would also surface plenty of fictional character studies that merely read like lived experience.",
      "best_endpoint": "keyword",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "The subject framing — biographical films as a category — is exactly what BIOGRAPHY was built for. The framing is binary (is this a biopic or not), not graded, so Keyword beats Semantic on both precision and recall. The bias agrees with the on-merits pick.",
  "endpoint_to_run": "keyword",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The user names the biopic form directly — real-person life dramatization as a category, not a specific subject.",
      "candidate_shortlist": "BIOGRAPHY covers life-dramatizations of a named real person. TRUE_STORY is broader (real events without a single biographical focus) and would over-pull history dramas; BIOGRAPHY is the tighter fit for 'biopic'.",
      "classification": "BIOGRAPHY"
    },
    "polarity": "positive"
  }
}
```

**Example: lower-tier wins — named real subject has no registry member**

```xml
<raw_query>movies about Princess Diana</raw_query>
<overall_query_intention_exploration>The user wants films whose subject is Princess Diana — a specific real historical figure. The registry has a generic BIOGRAPHY tag but no Diana-specific member.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films whose subject is the real figure Princess Diana.</captured_meaning>
  <category_name>Specific subject / element / motif</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies that depict Princess Diana as their subject.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies about Princess Diana</query_text>
  <description>Films whose subject is Princess Diana.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Find films whose subject is the specific real person Princess Diana.",
      "endpoint_coverage": "Keyword has no registry member naming Princess Diana specifically — BIOGRAPHY would over-pull every biopic regardless of subject, losing the whole point of the request. Semantic plot_events carries the concrete who-is-in-the-story text on the ingest side, so a query body naming Princess Diana and the core beats of her public life would surface the right films.",
      "best_endpoint": "semantic",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "Despite the keyword-first bias, no registry member covers this specific subject. Picking BIOGRAPHY would silently match every biopic in the database — a concept-mismatch, not a tight fit. Semantic plot_events cleanly holds the concrete 'who is in this story' signal. This is a lower-tier-wins case: the bias is a tiebreaker, not a veto, and the tiers are not close here.",
  "endpoint_to_run": "semantic",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "qualifier_inventory": "One atom: Princess Diana as the subject of the film — a specific real public figure whose life events (royal marriage, Panorama interview, death in Paris) would appear in plot summaries of films about her.",
      "space_queries": [
        {
          "carries_qualifiers": "plot_events holds the literal plot-summary prose where Princess Diana would be named as the subject along with the concrete biographical beats of her public life.",
          "space": "plot_events",
          "weight": "central",
          "content": {
            "plot_summary": "a biographical drama centered on Princess Diana — her marriage into the British royal family, her public life under intense media scrutiny, her humanitarian work, the breakdown of her marriage to Prince Charles, and the events surrounding her death in Paris."
          }
        }
      ],
      "primary_vector": "plot_events"
    },
    "polarity": "positive"
  }
}
```

**Example: spectrum framing flips to Semantic despite canonical tag existing**

```xml
<raw_query>subtly zombie-themed movies with loose allegorical undertones</raw_query>
<overall_query_intention_exploration>The user wants films that gesture at the zombie motif without being straight zombie movies — allegory, undertone, thematic echo. The gradient framing ('subtly', 'loose allegorical') is load-bearing.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films with subtle or allegorical zombie undertones rather than literal zombie presence.</captured_meaning>
  <category_name>Specific subject / element / motif</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies that evoke zombie themes allegorically or in undertone rather than featuring literal zombies.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>subtly zombie-themed movies with loose allegorical undertones</query_text>
  <description>Films with subtle, allegorical zombie-themed undertones.</description>
  <modifiers>
    <modifier>
      <original_text>subtly</original_text>
      <effect>softens the requirement from literal presence to thematic undertone</effect>
      <type>POLARITY_MODIFIER</type>
    </modifier>
    <modifier>
      <original_text>loose allegorical</original_text>
      <effect>frames the match as thematic echo rather than literal motif presence</effect>
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
      "aspect_description": "Find films that evoke zombie themes in undertone or allegorically — not literal zombie presence.",
      "endpoint_coverage": "Keyword's ZOMBIE_HORROR is a binary membership tag — a movie either has the tag or does not — and would overshoot by returning straight zombie films while missing films that only echo the motif. Semantic plot_events and plot_analysis can capture graded thematic proximity — 'consumerism-as-infection', 'mindless conformity as undead crowd' — which is exactly what 'allegorical undertone' means.",
      "best_endpoint": "semantic",
      "best_endpoint_gaps": null
    }
  ],
  "performance_vs_bias_analysis": "The framing is gradient, not binary. The modifiers 'subtly' and 'loose allegorical' explicitly push away from literal zombie presence, which is what ZOMBIE_HORROR encodes. A canonical tag treats membership as absolute and would mis-score the request. The keyword-first bias does not apply when the requirement is spectrum-shaped — Semantic wins on its own merits.",
  "endpoint_to_run": "semantic",
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "qualifier_inventory": "Two atoms: zombie motif as a thematic referent, and an allegorical-undertone framing that distinguishes echo from literal presence. Both atoms target thematic proximity rather than literal plot content.",
      "space_queries": [
        {
          "carries_qualifiers": "plot_analysis carries genre signatures and thematic concepts where allegorical zombie echoes land — consumerism, conformity, crowd contagion read as zombie-adjacent without literal zombies.",
          "space": "plot_analysis",
          "weight": "central",
          "content": {
            "elevator_pitch": "a film whose themes allegorically evoke the zombie motif — mindless conformity, crowd contagion, or consumerism-as-infection — without literal undead.",
            "genre_signatures": ["allegorical horror", "social satire"],
            "thematic_concepts": ["mindless conformity", "consumerism as infection", "dehumanization", "crowd contagion", "loss of individuality"]
          }
        },
        {
          "carries_qualifiers": "plot_events carries oblique echoes — scenes of crowds moving uniformly, dehumanized characters, contagion-like social spread — that surface in plot summaries even when the film isn't literally about zombies.",
          "space": "plot_events",
          "weight": "supporting",
          "content": {
            "plot_summary": "a story that evokes zombie imagery allegorically — crowds moving in uniform lockstep, characters dehumanized by their circumstances, or a social contagion that spreads like infection — without any literal undead in the plot."
          }
        }
      ],
      "primary_vector": "plot_analysis"
    },
    "polarity": "positive"
  }
}
```

**Example: vague subject ask neither endpoint can satisfy — no-fire**

```xml
<raw_query>movies with meaningful stuff in them</raw_query>
<overall_query_intention_exploration>The user's phrasing is too vague to identify a specific subject or motif. No concrete thing is named.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films containing unspecified meaningful elements.</captured_meaning>
  <category_name>Specific subject / element / motif</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies whose content is meaningful in some unspecified way.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies with meaningful stuff in them</query_text>
  <description>Films with unspecified meaningful content.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify a specific subject, element, or motif the film must contain.",
      "endpoint_coverage": "No subject or motif is actually named. Keyword would have to pick an arbitrary registry member, and Semantic plot_events would need a concrete target to author a body around. Neither endpoint can translate an unspecified 'meaningful stuff' into a matchable query.",
      "best_endpoint": "None",
      "best_endpoint_gaps": "The requirement names no concrete subject, element, or motif — both endpoints need a target concept, and fabricating one would misrepresent the user's ask."
    }
  ],
  "performance_vs_bias_analysis": "Neither endpoint has a fit. The keyword-first bias cannot break a tie between two non-fits — it only resolves close calls when both endpoints could genuinely run. No-fire is the correct outcome.",
  "endpoint_to_run": "None",
  "endpoint_parameters": null
}
```
