# Few-shot examples — Adaptation source flag

These examples calibrate: picking the right SourceMaterialType member from the origin-medium cue, inferring medium from an author slice, and no-firing when the query names a franchise rather than an adaptation source.

**Example: direct true-story cue**

<example>
<raw_query>movies based on a true story</raw_query>
<overall_query_intention_exploration>The user is asking for films dramatizing real events — a true-story origin-medium filter with no other constraints layered on.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The film dramatizes real events.</captured_meaning>
  <category_name>Adaptation source flag</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Origin medium is real events (true story).</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>based on a true story</query_text>
  <description>Asks for films with a real-events origin.</description>
  <modifiers>
    <modifier>
      <original_text>based on</original_text>
      <effect>marks the following noun as the source material the film is adapted from</effect>
      <type>ROLE_MARKER</type>
    </modifier>
  </modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Origin medium flag: the story is drawn from real events rather than fiction.",
      "relation_to_endpoint": "TRUE_STORY is the SourceMaterialType member whose definition names the real-events origin directly.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The atom asks for films adapted from real events. This is a yes/no origin-medium flag, cleanly inside the SourceMaterialType family.",
      "candidate_shortlist": "TRUE_STORY covers real-events origin generally. BIOGRAPHY narrows to one person's named life, which the query does not cite. TRUE_STORY wins on breadth given the absence of a single-subject cue.",
      "classification": "TRUE_STORY"
    },
    "polarity": "positive"
  }
}
```
</example>

**Example: direct novel-adaptation cue**

<example>
<raw_query>good novel adaptations</raw_query>
<overall_query_intention_exploration>The user wants films adapted from novels. Quality framing ("good") is a separate reception atom; the adaptation-source atom is the origin-medium flag itself.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The film is adapted from a novel.</captured_meaning>
  <category_name>Adaptation source flag</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Origin medium is a novel.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>novel adaptations</query_text>
  <description>Asks for films with a novel origin.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>good</query_text>
    <description>Quality qualifier on the film overall.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Origin medium flag: the film adapts a full-length novel.",
      "relation_to_endpoint": "NOVEL_ADAPTATION is the registry member whose definition directly names the novel origin.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The atom asks for films adapted from novels. This is the canonical SourceMaterialType flag; no spectrum framing or author scoping on this slice.",
      "candidate_shortlist": "NOVEL_ADAPTATION cleanly covers the novel origin. SHORT_STORY_ADAPTATION would under-cover since the query says novel. STAGE_ADAPTATION and COMIC_ADAPTATION are different media. NOVEL_ADAPTATION wins unambiguously.",
      "classification": "NOVEL_ADAPTATION"
    },
    "polarity": "positive"
  }
}
```
</example>

**Example: video-game adaptation**

<example>
<raw_query>video game movies</raw_query>
<overall_query_intention_exploration>The user wants films adapted from video games — a pure origin-medium request.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The film is adapted from a video game.</captured_meaning>
  <category_name>Adaptation source flag</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Origin medium is a video game.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>video game movies</query_text>
  <description>Films whose source material is a video game.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Origin medium flag: the film adapts a video game.",
      "relation_to_endpoint": "VIDEO_GAME_ADAPTATION names the video-game origin directly in its definition.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The phrase 'video game movies' is idiomatic for films adapted from video games, not films featuring video games as a plot element. The SourceMaterialType family covers this as a single flag.",
      "candidate_shortlist": "VIDEO_GAME_ADAPTATION is the only member whose definition names video-game origin. No other SourceMaterialType member applies.",
      "classification": "VIDEO_GAME_ADAPTATION"
    },
    "polarity": "positive"
  }
}
```
</example>

**Example: author name, medium inferred (author handled elsewhere)**

<example>
<raw_query>Stephen King adaptations</raw_query>
<overall_query_intention_exploration>The user wants films adapted from Stephen King's work. Step 2 has decomposed this into two atoms: the adaptation-source flag (handled here) and the author identity (handled by Source-material author, Cat 30). This call handles only the medium slice.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The film is adapted from a novel (inferred from Stephen King's primary medium).</captured_meaning>
  <category_name>Adaptation source flag</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Origin medium is a novel.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>Stephen King adaptations</query_text>
  <description>Films adapted from Stephen King's work.</description>
  <modifiers>
    <modifier>
      <original_text>adaptations</original_text>
      <effect>marks the films as adaptations of the named author's work</effect>
      <type>ROLE_MARKER</type>
    </modifier>
  </modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Origin medium flag for Stephen King adaptations — King is primarily a novelist, so the medium slice resolves to novel adaptation.",
      "relation_to_endpoint": "NOVEL_ADAPTATION covers the inferred origin medium. The author identity is a separate atom in Cat 30 and is not the responsibility of this handler.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "filter",
    "parameters": {
      "concept_analysis": "The atom is the adaptation-source slice of a Stephen King query. King's body of work is overwhelmingly novels; a handful of short-story adaptations exist but the canonical fit is novel adaptation. The author atom is handled by Cat 30 and is not duplicated here.",
      "candidate_shortlist": "NOVEL_ADAPTATION fits King's primary medium. SHORT_STORY_ADAPTATION covers a minority of his output and would under-cover on breadth. NOVEL_ADAPTATION wins as the broader, better-supported fit.",
      "classification": "NOVEL_ADAPTATION"
    },
    "polarity": "positive"
  }
}
```
</example>

**Example: no-fire on franchise-only phrasing**

<example>
<raw_query>Marvel movies</raw_query>
<overall_query_intention_exploration>The user wants films in the Marvel franchise / shared universe. The origin-medium framing is not actually present — "Marvel" names the franchise, not an explicit comic-adaptation ask. Cat 4 owns franchise membership.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>The film is part of the Marvel franchise.</captured_meaning>
  <category_name>Adaptation source flag</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Origin medium is a comic (inferred from Marvel being a comics publisher).</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>Marvel movies</query_text>
  <description>Films from the Marvel franchise.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "The atom has been routed here as an adaptation flag, but the surface request names a franchise, not an origin medium.",
      "relation_to_endpoint": "The keyword endpoint can emit COMIC_ADAPTATION, but doing so here would over-fire: many Marvel films qualify as comic adaptations, but so do DC and countless others. The user's intent is franchise membership, which Cat 4 owns. Upstream dispatch to this category is the wrong lane.",
      "coverage_gaps": "Bare-franchise phrasing does not assert an origin-medium requirement. Firing COMIC_ADAPTATION here would introduce non-Marvel comic adaptations and under-serve the actual intent."
    }
  ],
  "should_run_endpoint": false,
  "endpoint_parameters": null
}
```
</example>
