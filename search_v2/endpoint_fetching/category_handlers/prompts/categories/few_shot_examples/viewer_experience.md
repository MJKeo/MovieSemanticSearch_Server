# Examples

These examples calibrate three things on top of the boundary work the prior
versions modeled:
1. Density per active sub-field — 5–10 terms, not 3.
2. True-synonym discipline (substitution test) — what to include vs. what's
   drift.
3. Default-populate negations on every active section, naming the closest-
   opposite axis.

The first example shows a clean fire WITH a drift counter-example so the
model trains on the substitution test directly. The second example shows
how a user-supplied boundary lives in one section while the others still
default-populate. The last two examples are unchanged in intent — they
exercise the no-fire boundaries with Cat 26 (post-viewing aftertaste) and
Cat 27 (scale/scope).

**Example: clean fire — uplifting feel-good comedy (density + true
synonyms + default negations)**

```xml
<raw_query>uplifting feel-good comedy</raw_query>
<overall_query_intention_exploration>The user wants comedies whose during-viewing feel is uplifting and feel-good — a tonal aesthetic claim about how the movie feels moment to moment. The comedy genre lives on a sibling fragment.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>An uplifting, feel-good during-viewing feel — warm, joyful, light.</captured_meaning>
  <category_name>Viewer experience / feel / tone</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies whose during-viewing feel is uplifting and feel-good.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>uplifting feel-good</query_text>
  <description>An uplifting, feel-good tonal aesthetic.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>
    <query_text>comedy</query_text>
    <description>Comedy-genre films.</description>
    <modifiers></modifiers>
  </fragment>
</sibling_fragments>
```

Expected output (compressed to highlight the body):

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Match films whose during-viewing tone reads as uplifting and feel-good — warm emotional palette plus a playful, low-self-seriousness register.",
      "relation_to_endpoint": "viewer_experience emotional_palette holds warm/light tonal descriptors; tone_self_seriousness carries the playful register; both sub-fields land on the same axis the trait names.",
      "coverage_gaps": null
    }
  ],
  "should_run_endpoint": true,
  "endpoint_parameters": {
    "match_mode": "trait",
    "parameters": {
      "qualifier_inventory": "Two same-dimension tonal qualifiers treated as one unit: an uplifting, feel-good during-viewing feel.",
      "space_queries": [
        {
          "carries_qualifiers": "viewer_experience emotional_palette holds the warm/light tonal axis; tone_self_seriousness holds the playful register.",
          "space": "viewer_experience",
          "weight": "central",
          "content": {
            "emotional_palette": {
              "terms": [
                "uplifting",
                "feel-good",
                "heartwarming",
                "joyful",
                "warm",
                "laugh out loud"
              ],
              "negations": [
                "not depressing",
                "not bleak",
                "not heavy"
              ]
            },
            "tension_adrenaline": {
              "terms": [],
              "negations": [
                "not stressful",
                "not anxiety inducing"
              ]
            },
            "tone_self_seriousness": {
              "terms": [
                "earnest",
                "lighthearted",
                "playful"
              ],
              "negations": [
                "not cynical",
                "not mean-spirited"
              ]
            },
            "cognitive_complexity": {"terms": [], "negations": []},
            "disturbance_profile": {"terms": [], "negations": []},
            "sensory_load": {"terms": [], "negations": []},
            "emotional_volatility": {"terms": [], "negations": []},
            "ending_aftertaste": {
              "terms": [
                "satisfying ending",
                "feel-good payoff"
              ],
              "negations": [
                "not a downer ending"
              ]
            }
          }
        }
      ],
      "primary_vector": "viewer_experience"
    },
    "polarity": "positive"
  }
}
```

Notes on this output:
- `emotional_palette.terms` carries 6 phrases — true synonyms of "uplifting/feel-good" plus the search-query phrase "laugh out loud" that real users type. All pass the substitution test.
- **Terms and negations both point the same way.** For a feel-good body, terms are happy-side phrases (`uplifting`, `joyful`) and negations are the same idea expressed with `not`/`no` prefix (`not depressing`, `not bleak`). They reinforce — they are NOT opposites of each other. `"happy"` and `"not sad"` mean the same thing, and putting both in the body weights that concept in the embedded vector.
- `tension_adrenaline.negations` fires even though `tension_adrenaline.terms` is empty — `not stressful` / `not anxiety inducing` reinforce the feel-good direction (a feel-good film IS not stressful) and live in the same cluster as the explicit happy-palette signals.
- Sub-fields with no real signal stay fully empty (`cognitive_complexity`, `disturbance_profile`, `sensory_load`, `emotional_volatility`).

**COUNTER-EXAMPLE — do NOT emit this for the same query (drift):**

```json
"emotional_palette": {
  "terms": ["uplifting", "redemption", "underdog", "inspirational journey"]
}
```

Why this fails the substitution test:
- "redemption" — story shape, not a feel. A film can be uplifting without being a redemption story.
- "underdog" — character archetype, not a feel. Belongs to a different category entirely.
- "inspirational journey" — thematic claim about the story arc, not a tonal palette descriptor.

Each of these would shift the cosine search away from films whose `emotional_palette` actually carries "uplifting / feel-good / warm" and toward films with redemption arcs or underdog stories regardless of feel. **True synonyms only.**

**Example: clean fire with a user-supplied boundary — campy slasher, not too gory**

```xml
<raw_query>campy slasher horror that is not too gory</raw_query>
<atomic_rewrite>A campy, theatrical during-viewing feel that avoids graphic gore.</atomic_rewrite>
```

Expected output (highlights):

```json
"viewer_experience": {
  "emotional_palette": {
    "terms": ["campy", "theatrical", "fun", "tongue-in-cheek"],
    "negations": ["not earnest", "not grim"]
  },
  "tone_self_seriousness": {
    "terms": ["winking self aware", "knowingly silly", "over the top", "cheesy"],
    "negations": ["not serious", "not grimly serious"]
  },
  "tension_adrenaline": {
    "terms": ["slasher tension", "kills and chases"],
    "negations": ["not a slow burn", "not subtle"]
  },
  "disturbance_profile": {
    "terms": ["light scares", "playful violence"],
    "negations": ["not too gory", "no graphic gore", "not disturbing"]
  }
}
```

Notes:
- This is the NON-GORY direction. `disturbance_profile.terms = ["light scares", "playful violence"]` and `negations = ["not too gory", "no graphic gore", "not disturbing"]` BOTH point at the same retrieval target: films that have light slasher fun without graphic gore. They reinforce each other.
- The OPPOSITE direction (looking for gory films, e.g. trait="gory" with polarity=negative so the orchestrator can penalize them) would emit: `terms = ["gory", "bloody", "graphic violence", "splatter"]` + `negations = ["not peaceful", "not for kids", "not gentle"]` — both pointing at gory films, with polarity flipping the score downstream.
- Contradictory pairings to AVOID: `terms=["gory"] + negations=["not too gory"]` (same axis, opposite directions, contradicts itself).
- The other three active sections still default-populate negations even though no boundary was named there.

**Example: no-fire on a post-viewing aftertaste ask (Cat 26 boundary)**

```xml
<raw_query>something haunting that stays with you</raw_query>
<atomic_rewrite>Movies that feel haunting to watch.</atomic_rewrite>
```

Expected: `should_run_endpoint: false`. "Haunting / stays with you" is post-viewing aftertaste (Cat 26's `viewer_experience.ending_aftertaste`), not during-viewing feel. Forcing terms into `emotional_palette` would mis-embed the query as a during-viewing mood claim when the user named post-viewing resonance.

**Example: no-fire on a scale/scope ask (Cat 27 boundary)**

```xml
<raw_query>epic sweeping movies</raw_query>
<atomic_rewrite>Movies that feel epic and sweeping to watch.</atomic_rewrite>
```

Expected: `should_run_endpoint: false`. "Epic / sweeping" names scale, not tonal aesthetic or cognitive demand. Embedding into `emotional_palette` would route a scope query through the wrong vocabulary.
