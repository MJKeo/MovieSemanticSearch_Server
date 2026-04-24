# Examples

These calibrate the per-endpoint breakdown for occasion / self-experience goal / comfort-watch asks: SEMANTIC fires at the sub-space level (pick the surfaces the atom actually touches), KEYWORD fires only when the goal maps cleanly onto TEARJERKER or FEEL_GOOD, and the empty combination is the correct response when the atom is really a genre or plot request from a nearby category.

**Example: all-channels-fire — "make me cry" as the canonical multi-endpoint case**

```xml
<raw_query>a movie that will make me cry</raw_query>
<overall_query_intention_exploration>The user wants a film that reliably produces tears — a self-experience goal where the movie is meant to do something emotional to the viewer. "Make me cry" is a goal framing, not a plot description; it implies both a viewing motivation, a cathartic emotional target, and a reviewer-recognized "tearjerker" quality.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>A film whose goal is to make the viewer cry — a tearjerker watch.</captured_meaning>
  <category_name>Occasion / self-experience goal / comfort-watch</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies that reliably make the viewer cry.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>a movie that will make me cry</query_text>
  <description>A film the viewer wants in order to be moved to tears.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Capture the self-experience motivation — the user wants the movie to make them cry.",
      "endpoint_coverage": "SEMANTIC's watch_context.self_experience_motivations is the purpose-built sub-field for 'make me cry' as a named viewing goal. KEYWORD cannot hold motivation framing."
    },
    {
      "aspect_description": "Capture the emotional target the during-viewing feel needs to deliver — cathartic / melancholic palette.",
      "endpoint_coverage": "SEMANTIC's viewer_experience.emotional_palette carries the feel the movie must produce to fulfill the goal. KEYWORD has no gradient emotional-palette surface."
    },
    {
      "aspect_description": "Capture the reviewer-recognized 'tearjerker' quality.",
      "endpoint_coverage": "SEMANTIC's reception.praised_qualities carries the critic / audience label. KEYWORD also has a registry member (TEARJERKER) whose definition names exactly this — designed to make audiences cry and succeeds at it."
    }
  ],
  "overall_endpoint_fits": "This is the canonical multi-surface case for this category. SEMANTIC fires across watch_context (motivation), viewer_experience (emotional target), and reception (reviewer label) — each sub-space carries distinct, complementary evidence. KEYWORD fires additively via TEARJERKER: a registry member whose definition names the goal precisely, providing a hard-tag anchor alongside the gradient semantic channels.",
  "per_endpoint_breakdown": {
    "semantic": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "trait",
        "parameters": {
          "qualifier_inventory": "self-experience goal is 'make me cry'; implied emotional target is cathartic / melancholic feel; reviewer-label framing is 'tearjerker' / emotionally devastating.",
          "space_queries": [
            {
              "carries_qualifiers": "watch_context carries the named self-experience goal — the user frames the movie as something to make them cry, a motivation for sitting down to watch.",
              "space": "watch_context",
              "weight": "central",
              "content": {
                "self_experience_motivations": {"terms": ["make me cry", "have a good cry", "emotional release"]},
                "external_motivations": {"terms": []},
                "key_movie_feature_draws": {"terms": []},
                "watch_scenarios": {"terms": []}
              }
            },
            {
              "carries_qualifiers": "viewer_experience carries the during-viewing emotional palette the goal requires — cathartic, melancholic, heart-wrenching feel that actually produces tears.",
              "space": "viewer_experience",
              "weight": "central",
              "content": {
                "emotional_palette": {"terms": ["cathartic", "heart-wrenching", "melancholic", "devastating"], "negations": []},
                "tension_adrenaline": {"terms": [], "negations": []},
                "tone_self_seriousness": {"terms": [], "negations": []},
                "cognitive_complexity": {"terms": [], "negations": []},
                "disturbance_profile": {"terms": [], "negations": []},
                "sensory_load": {"terms": [], "negations": []},
                "emotional_volatility": {"terms": [], "negations": []},
                "ending_aftertaste": {"terms": [], "negations": []}
              }
            },
            {
              "carries_qualifiers": "reception carries the reviewer-recognized 'tearjerker' label — the critical / audience pattern of reporting that this film makes viewers cry.",
              "space": "reception",
              "weight": "supporting",
              "content": {
                "reception_summary": null,
                "praised_qualities": ["tearjerker", "emotionally devastating", "profoundly moving"],
                "criticized_qualities": []
              }
            }
          ],
          "primary_vector": "watch_context"
        },
        "polarity": "positive"
      }
    },
    "keyword": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "trait",
        "parameters": {
          "concept_analysis": "The query names 'make me cry' — the user wants a movie designed to produce tears and known to succeed at it. A self-experience goal framing.",
          "candidate_shortlist": "TEARJERKER definition names exactly this — designed to make audiences cry, with audiences reporting that it does. SAD_ENDING is a different axis (how it ends, not whether it makes you cry during). FEEL_GOOD is the opposite experiential tag. TEARJERKER wins on its own merits.",
          "classification": "TEARJERKER"
        },
        "polarity": "positive"
      }
    }
  }
}
```

**Example: narrow-fire — "rainy Sunday background movie" fires watch_context only, everything else silent**

```xml
<raw_query>a rainy Sunday background movie</raw_query>
<overall_query_intention_exploration>The user wants a film to have on during a rainy Sunday while doing something else — a pure occasion framing. No emotional target is named, no reviewer label is invoked, no goal-about-what-the-movie-does-to-me is stated. Just an occasion and a watch mode.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>A film suited to a rainy Sunday, played in the background.</captured_meaning>
  <category_name>Occasion / self-experience goal / comfort-watch</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>A movie to put on in the background on a rainy Sunday.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>a rainy Sunday background movie</query_text>
  <description>A film for a rainy Sunday as background viewing.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Capture the viewing occasion — rainy Sunday, background / ambient watching.",
      "endpoint_coverage": "SEMANTIC's watch_context.watch_scenarios carries occasion framings directly, and key_movie_feature_draws carries background-watch friendliness. KEYWORD has no registry member for 'rainy Sunday' or 'background'. The reviewer-label and emotional-palette surfaces are silent — the atom names neither."
    }
  ],
  "overall_endpoint_fits": "SEMANTIC fires on watch_context alone — the atom is a pure occasion framing with no emotional target and no reviewer label. viewer_experience stays silent because the user named no feel the movie must produce; reception stays silent because no critic / audience label was invoked. KEYWORD stays silent because the registry carries no member whose definition names this occasion. Narrow atoms fire narrowly.",
  "per_endpoint_breakdown": {
    "semantic": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "trait",
        "parameters": {
          "qualifier_inventory": "viewing occasion is a rainy Sunday; watch mode is background / ambient (playing while doing something else).",
          "space_queries": [
            {
              "carries_qualifiers": "watch_context carries both the occasion (rainy Sunday) via watch_scenarios and the background-friendliness via key_movie_feature_draws — the two distinct atoms the user named.",
              "space": "watch_context",
              "weight": "central",
              "content": {
                "self_experience_motivations": {"terms": []},
                "external_motivations": {"terms": []},
                "key_movie_feature_draws": {"terms": ["easy to follow without full attention", "low-stakes", "gentle pacing"]},
                "watch_scenarios": {"terms": ["rainy Sunday", "background movie", "ambient viewing"]}
              }
            }
          ],
          "primary_vector": "watch_context"
        },
        "polarity": "positive"
      }
    },
    "keyword": {
      "should_run_endpoint": false,
      "endpoint_parameters": null
    }
  }
}
```

**Example: feel-good goal — KEYWORD + watch_context + viewer_experience fire, reception silent**

```xml
<raw_query>something to cheer me up after a rough week</raw_query>
<overall_query_intention_exploration>The user wants a film that lifts their mood — a self-experience goal with an implied uplifting emotional target. "Cheer me up" names both the motivation for watching and the feel the movie must produce. The query does not invoke a specific critic-label framing (no "crowd-pleaser" or similar).</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>A film to cheer up the viewer after a rough week.</captured_meaning>
  <category_name>Occasion / self-experience goal / comfort-watch</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>A movie that will cheer the viewer up.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>something to cheer me up after a rough week</query_text>
  <description>A film the viewer wants to lift their mood.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Capture the self-experience motivation — the user wants the movie to cheer them up.",
      "endpoint_coverage": "SEMANTIC's watch_context.self_experience_motivations is the purpose-built surface for goal framings the user applies to themselves."
    },
    {
      "aspect_description": "Capture the emotional target the during-viewing feel must deliver — uplifting, warm, low-stakes palette.",
      "endpoint_coverage": "SEMANTIC's viewer_experience.emotional_palette carries the feel a mood-lifting movie must produce. Reception's praised_qualities could hold a reviewer label like 'crowd-pleaser' but the user invoked no such label — populating it from a bare goal atom would fabricate."
    },
    {
      "aspect_description": "Identify the registry member whose definition names a feel-good watch.",
      "endpoint_coverage": "KEYWORD's FEEL_GOOD definition names exactly this — the overall effect of watching is uplifting and positive. Fires additively with SEMANTIC."
    }
  ],
  "overall_endpoint_fits": "SEMANTIC fires on watch_context (the motivation) and viewer_experience (the emotional target) — the two surfaces the atom genuinely touches. Reception stays silent: no reviewer label was invoked. KEYWORD fires additively via FEEL_GOOD — a registry member whose definition names the self-experience effect.",
  "per_endpoint_breakdown": {
    "semantic": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "trait",
        "parameters": {
          "qualifier_inventory": "self-experience goal is 'cheer me up'; implied emotional target is uplifting / warm / low-stakes feel.",
          "space_queries": [
            {
              "carries_qualifiers": "watch_context carries the named motivation — the user frames the movie as something to lift their mood after a rough week.",
              "space": "watch_context",
              "weight": "central",
              "content": {
                "self_experience_motivations": {"terms": ["cheer me up", "lift my mood", "feel better"]},
                "external_motivations": {"terms": []},
                "key_movie_feature_draws": {"terms": []},
                "watch_scenarios": {"terms": []}
              }
            },
            {
              "carries_qualifiers": "viewer_experience carries the uplifting emotional palette and light tone the goal requires — the during-viewing feel that actually produces the mood lift.",
              "space": "viewer_experience",
              "weight": "central",
              "content": {
                "emotional_palette": {"terms": ["uplifting", "warm", "joyful", "hopeful"], "negations": ["bleak", "downbeat"]},
                "tension_adrenaline": {"terms": [], "negations": []},
                "tone_self_seriousness": {"terms": ["light", "breezy"], "negations": []},
                "cognitive_complexity": {"terms": [], "negations": []},
                "disturbance_profile": {"terms": [], "negations": []},
                "sensory_load": {"terms": [], "negations": []},
                "emotional_volatility": {"terms": [], "negations": []},
                "ending_aftertaste": {"terms": ["leaves you smiling"], "negations": []}
              }
            }
          ],
          "primary_vector": "watch_context"
        },
        "polarity": "positive"
      }
    },
    "keyword": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "trait",
        "parameters": {
          "concept_analysis": "The query names 'cheer me up' — the user wants a movie whose overall effect is uplifting. A feel-good self-experience goal.",
          "candidate_shortlist": "FEEL_GOOD definition names exactly this — the trajectory and ending leave the viewer feeling good. HAPPY_ENDING is a different axis (how the movie ends, not the overall effect). TEARJERKER is the opposite experiential tag. FEEL_GOOD wins on its own merits.",
          "classification": "FEEL_GOOD"
        },
        "polarity": "positive"
      }
    }
  }
}
```

**Example: gateway ask — watch_context + reception fire, KEYWORD silent**

```xml
<raw_query>a good first anime for someone who's never watched any</raw_query>
<overall_query_intention_exploration>The user wants an accessible entry-point into anime — a gateway ask. The framing is about the movie serving as an introduction, not about its genre or plot. This implies both an entry-point motivation (on the watch_context side) and an accessibility reviewer-label quality (on the reception side).</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>An accessible first anime for a newcomer to the medium.</captured_meaning>
  <category_name>Occasion / self-experience goal / comfort-watch</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>An entry-level, accessible anime for a first-time viewer.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>a good first anime for someone who's never watched any</query_text>
  <description>An accessible entry-point anime for a newcomer.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Capture the gateway / entry-point motivation — the film serves as an introduction for a newcomer.",
      "endpoint_coverage": "SEMANTIC's watch_context carries this through self_experience_motivations (first-time viewer intent) and key_movie_feature_draws (accessibility as a draw)."
    },
    {
      "aspect_description": "Capture the reviewer-recognized 'accessible / gateway' quality.",
      "endpoint_coverage": "SEMANTIC's reception.praised_qualities carries the critic / audience language around a film being a good entry-point — 'accessible', 'gateway anime', 'newcomer-friendly'."
    },
    {
      "aspect_description": "Identify whether the registry carries a member for gateway framing.",
      "endpoint_coverage": "KEYWORD has ExperientialTag members (FEEL_GOOD, TEARJERKER) but none whose definition names 'gateway' or 'accessible entry-point'. No fit."
    }
  ],
  "overall_endpoint_fits": "SEMANTIC fires on watch_context (entry-point motivation + accessibility draw) and reception (reviewer-label for accessibility / gateway suitability). viewer_experience stays silent — the user named no emotional target. KEYWORD stays silent — no registry member's definition names this gateway framing, and reaching for a tangential member would silently broaden the result away from the actual ask.",
  "per_endpoint_breakdown": {
    "semantic": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "trait",
        "parameters": {
          "qualifier_inventory": "gateway motivation is 'first anime' for a newcomer; accessibility is the implied key draw; reviewer-label framing is 'accessible / newcomer-friendly anime'.",
          "space_queries": [
            {
              "carries_qualifiers": "watch_context carries both the entry-point motivation via self_experience_motivations and the accessibility-as-draw via key_movie_feature_draws — the two atoms the gateway framing names.",
              "space": "watch_context",
              "weight": "central",
              "content": {
                "self_experience_motivations": {"terms": ["first anime", "gateway into anime", "introduction to the medium"]},
                "external_motivations": {"terms": []},
                "key_movie_feature_draws": {"terms": ["accessible to newcomers", "approachable entry point", "doesn't require prior anime knowledge"]},
                "watch_scenarios": {"terms": []}
              }
            },
            {
              "carries_qualifiers": "reception carries the reviewer-applied 'gateway / accessible anime' label — the pattern of critics and audiences recommending this title to newcomers.",
              "space": "reception",
              "weight": "supporting",
              "content": {
                "reception_summary": null,
                "praised_qualities": ["accessible to newcomers", "gateway anime", "newcomer-friendly"],
                "criticized_qualities": []
              }
            }
          ],
          "primary_vector": "watch_context"
        },
        "polarity": "positive"
      }
    },
    "keyword": {
      "should_run_endpoint": false,
      "endpoint_parameters": null
    }
  }
}
```

**Example: no-fire — atom is really a plot request, not an occasion or goal**

```xml
<raw_query>a heist movie with a big twist at the end</raw_query>
<overall_query_intention_exploration>The user wants a heist film with a twist ending. The framing is about the movie's plot shape and ending mechanic — what the movie IS — not about the occasion for watching or a self-experience goal. Upstream routing landed this atom in Cat 23, but the captured meaning is plot-centric.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>A heist movie with a big twist ending.</captured_meaning>
  <category_name>Occasion / self-experience goal / comfort-watch</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>A heist film with a twist ending.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>a heist movie with a big twist at the end</query_text>
  <description>A heist film with a twist ending.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the viewing occasion, self-experience goal, comfort archetype, or gateway framing the atom names.",
      "endpoint_coverage": "None fits. The atom describes the movie's plot premise (a heist) and a structural ending mechanic (a twist) — it names neither an occasion for watching, nor a goal the movie should produce in the viewer, nor a comfort-watch or gateway framing. SEMANTIC's watch_context expects occasion / motivation framings and would be fabricating if populated from a plot atom. Reception's praised_qualities would need a reviewer-label framing the user did not invoke. KEYWORD's ExperientialTag registry members (FEEL_GOOD, TEARJERKER) do not cover heist-with-twist — PLOT_TWIST and HEIST belong to other categories' handlers."
    }
  ],
  "overall_endpoint_fits": "The atom is a plot / structure request misrouted here — Cat 15 (sub-genre / archetype) handles the heist premise and Cat 16 (narrative devices) or Cat 26 (ending resonance) handles the twist. No endpoint in this category's set carries signal the atom supports. Firing would fabricate: a watch_scenario the user did not name, a reception label not invoked, or a keyword that lives outside this category's ExperientialTag slice. The empty combination is the correct response.",
  "per_endpoint_breakdown": {
    "semantic": {
      "should_run_endpoint": false,
      "endpoint_parameters": null
    },
    "keyword": {
      "should_run_endpoint": false,
      "endpoint_parameters": null
    }
  }
}
```
