# Examples

These calibrate the seasonal proxy-chain rewrite on KEYWORD, the dual-space SEMANTIC split across watch_context and plot_events, the thin-proxy case where KEYWORD stays silent and SEMANTIC carries it alone, and the no-fire shape when the atom is not really seasonal.

**Example: Christmas — both KEYWORD and SEMANTIC fire**

```xml
<raw_query>Christmas movies</raw_query>
<overall_query_intention_exploration>The user wants films that are Christmas movies — the seasonal packaging people reach for around the holidays. Covers both the viewing-occasion angle (Christmas viewing) and the narrative-setting angle (set during Christmas).</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films framed as Christmas movies — seasonal packaging around the holiday.</captured_meaning>
  <category_name>Seasonal / holiday</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies framed as Christmas movies for holiday viewing and Christmas-set stories.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>Christmas movies</query_text>
  <description>Films framed as Christmas movies.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Cover the Christmas packaging the user named — the kind of film a Christmas viewer reaches for.",
      "endpoint_coverage": "KEYWORD has no dedicated Christmas member, but HOLIDAY_FAMILY's definition covers festive-family seasonal packaging directly and is the strongest proxy for a bare 'Christmas movies' ask. The registry cannot encode the full Christmas framing beyond this proxy — it will miss Christmas-set films that don't land on the family axis."
    },
    {
      "aspect_description": "Capture the Christmas viewing occasion — the ritual of watching at Christmas.",
      "endpoint_coverage": "SEMANTIC watch_context.watch_scenarios holds occasion phrases like 'Christmas viewing' and 'holiday movie night' directly in its native vocabulary — exactly the framing the proxy tag cannot encode."
    },
    {
      "aspect_description": "Capture the Christmas-set narrative angle — stories that take place at Christmas.",
      "endpoint_coverage": "SEMANTIC plot_events.plot_summary carries compact prose naming the seasonal setting of the story — films set at Christmas land here on the ingest side. KEYWORD cannot encode a narrative-setting signal."
    }
  ],
  "overall_endpoint_fits": "KEYWORD contributes the strongest available proxy (HOLIDAY_FAMILY — festive-family seasonal packaging) as the closest registry shadow of 'Christmas movies'. SEMANTIC contributes the seasonal viewing occasion via watch_context and the Christmas-set narrative angle via plot_events — the two spaces the ingest side actually carries seasonal signal on. Both endpoints fire because each carries distinct signal: the proxy tag catches the festive-family shape of Christmas movies, watch_context catches the viewing ritual, and plot_events catches the Christmas-set stories the proxy alone would miss.",
  "per_endpoint_breakdown": {
    "keyword": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "filter",
        "parameters": {
          "concept_analysis": "The user names Christmas movies — festive seasonal packaging for holiday viewing. No dedicated Christmas member in the registry; the proxy must come from the members whose definitions shadow the concept.",
          "candidate_shortlist": "HOLIDAY_FAMILY is defined as 'a family-friendly holiday movie focused on festive traditions, family bonds, and seasonal warmth' — the closest registry shadow of the Christmas-movie framing. FAMILY is broader (no seasonal dimension) and would drop the holiday specificity. FEEL_GOOD covers the uplifting affect but not the seasonal packaging. HOLIDAY_FAMILY wins as the one proxy whose definition directly names the seasonal angle.",
          "classification": "HOLIDAY_FAMILY"
        },
        "polarity": "positive"
      }
    },
    "semantic": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "filter",
        "parameters": {
          "qualifier_inventory": "Two seasonal atoms: the Christmas viewing ritual (watch_context territory) and Christmas as a narrative setting (plot_events territory). Both are genuinely present in a bare 'Christmas movies' ask because the phrase covers both angles.",
          "space_queries": [
            {
              "carries_qualifiers": "watch_context carries the Christmas viewing occasion — the ritual of putting on a Christmas movie during the holidays, the seasonal comfort pull of holiday viewing.",
              "space": "watch_context",
              "weight": "central",
              "content": {
                "self_experience_motivations": {"terms": ["holiday comfort", "seasonal warmth"]},
                "external_motivations": {"terms": ["watching at Christmas", "holiday family viewing"]},
                "key_movie_feature_draws": {"terms": []},
                "watch_scenarios": {"terms": ["Christmas viewing", "holiday movie night", "Christmas Eve watch"]}
              }
            },
            {
              "carries_qualifiers": "plot_events carries the Christmas-set narrative angle — stories that take place at Christmas as their setting.",
              "space": "plot_events",
              "weight": "supporting",
              "content": {
                "plot_summary": "a story that takes place at Christmas, with events unfolding over the holiday — Christmas Eve gatherings, snowy settings, seasonal festivities central to the story."
              }
            }
          ],
          "primary_vector": "watch_context"
        },
        "polarity": "positive"
      }
    }
  }
}
```

**Example: Halloween viewing — both fire, proxy leans HORROR**

```xml
<raw_query>something for Halloween viewing</raw_query>
<overall_query_intention_exploration>The user wants a film for Halloween viewing — the seasonal movie-night ritual on or around Halloween. Leans into the occasion angle but also carries the Halloween-set narrative angle for films that take place on the night itself.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>A film for Halloween viewing — the seasonal movie-night ritual.</captured_meaning>
  <category_name>Seasonal / holiday</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies fitting a Halloween viewing occasion — horror-adjacent, seasonal.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>something for Halloween viewing</query_text>
  <description>A Halloween-appropriate movie for seasonal viewing.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Cover the Halloween packaging — the kind of film a Halloween viewer reaches for.",
      "endpoint_coverage": "KEYWORD has no Halloween member. HORROR is the canonical proxy for the Halloween-viewing ritual — the broad genre whose films are what Halloween viewing largely consists of. Narrower sub-horror members (SLASHER_HORROR, SUPERNATURAL_HORROR) would require premise-specific signal the bare 'Halloween viewing' phrasing does not carry."
    },
    {
      "aspect_description": "Capture the Halloween viewing occasion itself — the seasonal movie-night framing.",
      "endpoint_coverage": "SEMANTIC watch_context.watch_scenarios carries 'Halloween movie night' and 'Halloween viewing' as occasion terms directly — the purpose-built space for seasonal viewing rituals."
    },
    {
      "aspect_description": "Capture the Halloween-set narrative angle — films set on Halloween night.",
      "endpoint_coverage": "SEMANTIC plot_events.plot_summary carries narrative-setting prose; 'events unfold on Halloween night' is exactly the kind of seasonal-setting phrase that lands there on the ingest side."
    }
  ],
  "overall_endpoint_fits": "KEYWORD fires HORROR as the broad proxy for the Halloween viewing ritual — no narrower sub-horror member is signalled by the bare phrasing. SEMANTIC fires on both watch_context (Halloween viewing occasion) and plot_events (Halloween-set stories). All three signals are distinct: the proxy tag catches the genre shadow, watch_context catches the ritual, and plot_events catches Halloween-set films the proxy may miss.",
  "per_endpoint_breakdown": {
    "keyword": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "filter",
        "parameters": {
          "concept_analysis": "The user names Halloween viewing. No Halloween registry member exists; the proxy candidates are the horror-family members whose definitions cover the kind of film Halloween viewers reach for.",
          "candidate_shortlist": "HORROR is the broad default — 'a movie intended to evoke fear, dread, or repulsion' — and covers the breadth of Halloween viewing across sub-forms. SLASHER_HORROR needs a stalker/killer premise cue. SUPERNATURAL_HORROR needs ghosts/demons/possession cued. MONSTER_HORROR needs a monster premise. Bare 'Halloween viewing' gives none of those — HORROR wins as the premise-agnostic proxy.",
          "classification": "HORROR"
        },
        "polarity": "positive"
      }
    },
    "semantic": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "filter",
        "parameters": {
          "qualifier_inventory": "Two seasonal atoms: the Halloween viewing occasion itself (watch_context territory) and Halloween-set stories (plot_events territory). The user's phrasing leans into the viewing occasion, but the seasonal packaging carries the setting angle as well.",
          "space_queries": [
            {
              "carries_qualifiers": "watch_context carries the Halloween viewing ritual — the movie-night occasion and the seasonal pull toward spooky viewing.",
              "space": "watch_context",
              "weight": "central",
              "content": {
                "self_experience_motivations": {"terms": ["seasonal spooky mood", "Halloween ritual viewing"]},
                "external_motivations": {"terms": ["watching on Halloween"]},
                "key_movie_feature_draws": {"terms": []},
                "watch_scenarios": {"terms": ["Halloween movie night", "Halloween viewing", "October spooky watch"]}
              }
            },
            {
              "carries_qualifiers": "plot_events carries the Halloween-set narrative angle — films whose story takes place on Halloween night itself.",
              "space": "plot_events",
              "weight": "supporting",
              "content": {
                "plot_summary": "a story that takes place on Halloween night — trick-or-treating, costumes, a jack-o-lantern-lit suburb or small town as the setting, events unfolding over the night of Halloween."
              }
            }
          ],
          "primary_vector": "watch_context"
        },
        "polarity": "positive"
      }
    }
  }
}
```

**Example: Summer blockbuster — both fire, watch_context alone carries SEMANTIC**

```xml
<raw_query>summer blockbuster</raw_query>
<overall_query_intention_exploration>The user wants a summer-blockbuster film — the large-scale, crowd-pleasing spectacle people reach for during summer viewing. The phrasing is an occasion/scale framing, not a narrative-setting one.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>A summer-blockbuster film — the big-spectacle summer viewing pull.</captured_meaning>
  <category_name>Seasonal / holiday</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies framed as summer blockbusters — large-scale spectacle for summer viewing.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>summer blockbuster</query_text>
  <description>A summer-blockbuster film.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Cover the summer-blockbuster packaging — the large-scale spectacle framing.",
      "endpoint_coverage": "KEYWORD has no summer-blockbuster member. ACTION_EPIC — 'an action movie with large-scale spectacle and a sweeping, high-stakes conflict of major consequence' — is the closest registry shadow of the blockbuster-scale framing. ACTION alone is broader and drops the scale dimension; ADVENTURE_EPIC narrows to adventure specifically."
    },
    {
      "aspect_description": "Capture the summer viewing occasion itself — the seasonal blockbuster pull.",
      "endpoint_coverage": "SEMANTIC watch_context.watch_scenarios carries 'summer viewing' and 'blockbuster night out' directly in its native vocabulary. plot_events does not fire — 'summer blockbuster' frames scale and viewing occasion, not a summer-set story."
    }
  ],
  "overall_endpoint_fits": "KEYWORD fires ACTION_EPIC as the closest registry shadow of the blockbuster-scale framing. SEMANTIC fires on watch_context alone — the summer viewing occasion and the crowd-pleasing spectacle pull. plot_events does NOT fire here: the user's phrasing names a scale and occasion, not a summer-set narrative, and populating plot_summary from 'summer blockbuster' would fabricate a setting the input does not support.",
  "per_endpoint_breakdown": {
    "keyword": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "filter",
        "parameters": {
          "concept_analysis": "The user names summer blockbuster. No summer-blockbuster registry member; the proxy must come from large-scale-spectacle members whose definitions name sweeping scope.",
          "candidate_shortlist": "ACTION_EPIC directly names large-scale spectacle and sweeping high-stakes conflict — the closest registry shadow of the blockbuster-scale framing. ACTION is broader and drops the scale dimension. ADVENTURE_EPIC narrows to adventure premise specifically, which the user did not signal. EPIC covers sweeping scope but not the action-spectacle axis that 'blockbuster' carries. ACTION_EPIC wins.",
          "classification": "ACTION_EPIC"
        },
        "polarity": "positive"
      }
    },
    "semantic": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "filter",
        "parameters": {
          "qualifier_inventory": "One dominant atom: the summer-blockbuster viewing occasion — big-screen spectacle, crowd-pleasing scale, summer viewing ritual. No narrative-setting atom is present.",
          "space_queries": [
            {
              "carries_qualifiers": "watch_context carries the summer-blockbuster viewing ritual — the occasion and the crowd-pleasing spectacle pull that defines the seasonal category.",
              "space": "watch_context",
              "weight": "central",
              "content": {
                "self_experience_motivations": {"terms": ["crowd-pleasing spectacle", "big-screen thrills"]},
                "external_motivations": {"terms": ["summer viewing", "blockbuster night out"]},
                "key_movie_feature_draws": {"terms": ["large-scale spectacle", "summer tentpole"]},
                "watch_scenarios": {"terms": ["summer blockbuster viewing", "big-screen summer night"]}
              }
            }
          ],
          "primary_vector": "watch_context"
        },
        "polarity": "positive"
      }
    }
  }
}
```

**Example: Valentine's Day — SEMANTIC only, KEYWORD stays silent**

```xml
<raw_query>Valentine's Day movies</raw_query>
<overall_query_intention_exploration>The user wants a film for Valentine's Day — the seasonal packaging people reach for around the holiday, typically a romantic date-night viewing pick. The seasonal framing is real but no registry member cleanly shadows the Valentine's packaging.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>A film for Valentine's Day viewing — seasonal romantic packaging.</captured_meaning>
  <category_name>Seasonal / holiday</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Movies framed as Valentine's Day picks — seasonal romantic viewing.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>Valentine's Day movies</query_text>
  <description>Films framed as Valentine's Day viewing picks.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Cover the Valentine's Day packaging — the kind of film a Valentine's viewer reaches for.",
      "endpoint_coverage": "KEYWORD has no Valentine's-Day member and no clean proxy. ROMANTIC_COMEDY / FEEL_GOOD_ROMANCE / ROMANCE are romance-genre members but their definitions cover the genre shape, not the seasonal ritual — firing one would narrow the pool to one romance sub-shape rather than representing the Valentine's packaging. No proxy is a clean fit."
    },
    {
      "aspect_description": "Capture the Valentine's Day viewing occasion itself — the seasonal date-night framing.",
      "endpoint_coverage": "SEMANTIC watch_context.watch_scenarios carries 'Valentine's Day date', 'date-night viewing', and similar seasonal-occasion phrases directly. This is where the seasonal signal actually lives for Valentine's Day."
    },
    {
      "aspect_description": "Capture any Valentine's-Day-set narrative angle.",
      "endpoint_coverage": "SEMANTIC plot_events covers narrative-setting prose, but the user's phrasing leans into the occasion rather than a Valentine's-set story. Populating plot_summary from the bare occasion noun would fabricate a setting the input does not support."
    }
  ],
  "overall_endpoint_fits": "KEYWORD does NOT fire: the romance-genre members only shadow the genre, not the seasonal packaging, and the Valentine's-Day ritual has no clean registry proxy. Firing ROMANTIC_COMEDY or ROMANCE as a stand-in would silently narrow the pool to a genre sub-shape instead of the seasonal framing. SEMANTIC carries the full requirement through watch_context — the seasonal viewing occasion is the load-bearing atom here. plot_events stays silent: the phrasing names an occasion, not a Valentine's-set story.",
  "per_endpoint_breakdown": {
    "keyword": {
      "should_run_endpoint": false,
      "endpoint_parameters": null
    },
    "semantic": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "filter",
        "parameters": {
          "qualifier_inventory": "One dominant atom: the Valentine's Day viewing occasion — the seasonal date-night ritual, romantic pull, couples-viewing framing. No narrative-setting atom is named.",
          "space_queries": [
            {
              "carries_qualifiers": "watch_context carries the Valentine's Day viewing ritual — the seasonal date-night occasion, the couples-viewing pull, the romantic-mood seasonal framing that defines the category for this holiday.",
              "space": "watch_context",
              "weight": "central",
              "content": {
                "self_experience_motivations": {"terms": ["romantic mood", "seasonal date-night feeling"]},
                "external_motivations": {"terms": ["Valentine's Day viewing", "watching with a partner"]},
                "key_movie_feature_draws": {"terms": []},
                "watch_scenarios": {"terms": ["Valentine's Day date", "date-night viewing", "couples romantic watch"]}
              }
            }
          ],
          "primary_vector": "watch_context"
        },
        "polarity": "positive"
      }
    }
  }
}
```

**Example: no-fire — "movies about family" is not a seasonal ask**

```xml
<raw_query>movies about family</raw_query>
<overall_query_intention_exploration>The user wants films about family — the thematic territory of family relationships, bonds, conflict. No seasonal or holiday framing is present; the phrase is a thematic-archetype ask, not a seasonal one.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films about family — thematic territory, not seasonal.</captured_meaning>
  <category_name>Seasonal / holiday</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Movies thematically about family.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies about family</query_text>
  <description>Films thematically about family.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the seasonal or holiday framing in the atom.",
      "endpoint_coverage": "No seasonal framing is present. 'Movies about family' names a thematic territory — family relationships, bonds, conflict — not a holiday, a season, or a viewing occasion tied to one. KEYWORD's seasonal proxies here (HOLIDAY_FAMILY, FAMILY) would either fabricate a holiday angle the user did not name or narrow onto a packaging label rather than the thematic ask. SEMANTIC's watch_context expects seasonal viewing occasions, not a bare thematic descriptor; plot_events expects a seasonal narrative setting, which the user did not name. The atom belongs to the thematic-archetype category (family-as-theme), not here."
    }
  ],
  "overall_endpoint_fits": "The atom routed here is a thematic-archetype ask, not a seasonal or holiday framing. Neither KEYWORD nor SEMANTIC in this category's set carries signal the input actually supports — firing the HOLIDAY_FAMILY proxy would invent a seasonal angle, and populating watch_context or plot_events from a bare thematic phrase would fabricate a scenario or setting the input does not describe. The empty combination is the correct response; the thematic-archetype category owns this.",
  "per_endpoint_breakdown": {
    "keyword": {
      "should_run_endpoint": false,
      "endpoint_parameters": null
    },
    "semantic": {
      "should_run_endpoint": false,
      "endpoint_parameters": null
    }
  }
}
```
