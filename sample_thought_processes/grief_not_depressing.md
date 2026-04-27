# Query

> Films that deal with grief without being depressing.

## Step 2 — Query Pre-pass

_elapsed: 3.12s_

#### Step 2 Response
```json
{
  "overall_query_intention_exploration": "The user is seeking movies that explore the heavy thematic subject of grief but specifically requests a tonal balance that avoids a depressing viewer experience. This suggests a search for stories that might be cathartic, hopeful, or life-affirming rather than purely somber.",
  "requirements": [
    {
      "query_text": "grief",
      "description": "The central thematic subject or emotional journey of the movie.",
      "modifiers": [
        {
          "original_text": "deal with",
          "effect": "binds the subject to the narrative focus of the story",
          "type": "role_marker"
        }
      ],
      "coverage_evidence": [
        {
          "captured_meaning": "The story focuses on the emotional process and thematic exploration of loss.",
          "category_name": "Kind of story / thematic archetype",
          "fit_quality": "clean",
          "atomic_rewrite": "about the thematic experience of grief"
        }
      ]
    },
    {
      "query_text": "depressing",
      "description": "The emotional tone or viewer experience the user wants to avoid.",
      "modifiers": [
        {
          "original_text": "without being",
          "effect": "negation — excludes movies with this specific tonal quality",
          "type": "polarity_modifier"
        }
      ],
      "coverage_evidence": [
        {
          "captured_meaning": "The user wants to avoid a viewer experience characterized by a heavy, somber, or overwhelmingly sad tone.",
          "category_name": "Viewer experience / feel / tone",
          "fit_quality": "clean",
          "atomic_rewrite": "without a depressing tone"
        }
      ]
    }
  ]
}
```

## Step 3 — Endpoint Translation + Execution

### CE 1 — Kind of story / thematic archetype
_atomic_rewrite_: about the thematic experience of grief

#### CE 1 payload
```json
{
  "captured_meaning": "The story focuses on the emotional process and thematic exploration of loss.",
  "category_name": "Kind of story / thematic archetype",
  "atomic_rewrite": "about the thematic experience of grief",
  "llm": {
    "elapsed_s": 5.754,
    "output": {
      "requirement_aspects": [
        {
          "aspect_description": "Identify grief as the thematic focus of the story.",
          "endpoint_coverage": "Keyword is the better fit if a registry member named grief existed, but the closed vocabulary does not provide a grief theme tag. Semantic can carry the thematic concept directly in plot_analysis.thematic_concepts, and can also reflect the emotional arc of loss/coping; it is the only endpoint that can express the theme honestly.",
          "best_endpoint": "semantic",
          "best_endpoint_gaps": null
        },
        {
          "aspect_description": "Avoid a depressing viewing experience.",
          "endpoint_coverage": "Keyword has no clean registry member for 'not depressing' in the thematic-archetype axis. Semantic can route this separately through viewer_experience, where negative tone/ending aftertaste terms can represent the requested tonal restraint. This is distinct from the grief theme itself.",
          "best_endpoint": "semantic",
          "best_endpoint_gaps": null
        }
      ],
      "performance_vs_bias_analysis": "This is not a binary thematic-only ask: 'without being depressing' introduces an experiential constraint alongside the grief theme. The spectrum escape does not apply to grief here because the theme is still a defining subject, but the closed Keyword registry lacks a grief member anyway. Semantic wins on merit because it can represent both the theme and the anti-depressing tonal constraint; the tier-1 bias does not override the vocabulary gap.",
      "endpoint_to_run": "semantic",
      "endpoint_parameters": {
        "match_mode": "trait",
        "parameters": {
          "qualifier_inventory": "grief as the central theme; non-depressing / not somber viewer experience.",
          "space_queries": [
            {
              "carries_qualifiers": "plot_analysis carries the grief theme as the story's subject and emotional process of loss/coping.",
              "space": "plot_analysis",
              "weight": "central",
              "content": {
                "elevator_pitch": "a story centered on grief and the process of coping with loss",
                "plot_overview": null,
                "genre_signatures": [],
                "conflict_type": [],
                "thematic_concepts": [
                  "grief",
                  "loss",
                  "mourning",
                  "coping"
                ],
                "character_arcs": []
              }
            },
            {
              "carries_qualifiers": "viewer_experience carries the requested avoidance of a depressing tone and the preference for a more hopeful, cathartic aftertaste.",
              "space": "viewer_experience",
              "weight": "central",
              "content": {
                "emotional_palette": {
                  "terms": [
                    "cathartic",
                    "hopeful",
                    "life-affirming"
                  ],
                  "negations": [
                    "depressing",
                    "hopeless",
                    "bleak"
                  ]
                },
                "tension_adrenaline": {
                  "terms": [],
                  "negations": []
                },
                "tone_self_seriousness": {
                  "terms": [
                    "gentle"
                  ],
                  "negations": [
                    "heavy-handed"
                  ]
                },
                "cognitive_complexity": {
                  "terms": [],
                  "negations": []
                },
                "disturbance_profile": {
                  "terms": [],
                  "negations": []
                },
                "sensory_load": {
                  "terms": [],
                  "negations": []
                },
                "emotional_volatility": {
                  "terms": [
                    "moving"
                  ],
                  "negations": [
                    "emotionally crushing"
                  ]
                },
                "ending_aftertaste": {
                  "terms": [
                    "uplifting"
                  ],
                  "negations": [
                    "downbeat"
                  ]
                }
              }
            }
          ],
          "primary_vector": "plot_analysis"
        },
        "polarity": "positive"
      }
    }
  },
  "endpoints": [
    {
      "endpoint": "semantic",
      "match_mode": "trait",
      "polarity": "positive",
      "wrapper_type": "SemanticEndpointParameters",
      "parameters": {
        "qualifier_inventory": "grief as the central theme; non-depressing / not somber viewer experience.",
        "space_queries": [
          {
            "carries_qualifiers": "plot_analysis carries the grief theme as the story's subject and emotional process of loss/coping.",
            "space": "plot_analysis",
            "weight": "central",
            "content": {
              "elevator_pitch": "a story centered on grief and the process of coping with loss",
              "plot_overview": null,
              "genre_signatures": [],
              "conflict_type": [],
              "thematic_concepts": [
                "grief",
                "loss",
                "mourning",
                "coping"
              ],
              "character_arcs": []
            }
          },
          {
            "carries_qualifiers": "viewer_experience carries the requested avoidance of a depressing tone and the preference for a more hopeful, cathartic aftertaste.",
            "space": "viewer_experience",
            "weight": "central",
            "content": {
              "emotional_palette": {
                "terms": [
                  "cathartic",
                  "hopeful",
                  "life-affirming"
                ],
                "negations": [
                  "depressing",
                  "hopeless",
                  "bleak"
                ]
              },
              "tension_adrenaline": {
                "terms": [],
                "negations": []
              },
              "tone_self_seriousness": {
                "terms": [
                  "gentle"
                ],
                "negations": [
                  "heavy-handed"
                ]
              },
              "cognitive_complexity": {
                "terms": [],
                "negations": []
              },
              "disturbance_profile": {
                "terms": [],
                "negations": []
              },
              "sensory_load": {
                "terms": [],
                "negations": []
              },
              "emotional_volatility": {
                "terms": [
                  "moving"
                ],
                "negations": [
                  "emotionally crushing"
                ]
              },
              "ending_aftertaste": {
                "terms": [
                  "uplifting"
                ],
                "negations": [
                  "downbeat"
                ]
              }
            }
          }
        ],
        "primary_vector": "plot_analysis"
      },
      "exec_elapsed_s": 0.589,
      "result": {
        "total_scored": 3844,
        "top": [
          {
            "movie_id": 659708,
            "score": 0.81195259
          },
          {
            "movie_id": 13668,
            "score": 0.81091595
          },
          {
            "movie_id": 460089,
            "score": 0.7983408
          },
          {
            "movie_id": 345920,
            "score": 0.7955656
          },
          {
            "movie_id": 468225,
            "score": 0.7952070250000001
          },
          {
            "movie_id": 265347,
            "score": 0.791399
          },
          {
            "movie_id": 122698,
            "score": 0.790688525
          },
          {
            "movie_id": 842328,
            "score": 0.7906723
          },
          {
            "movie_id": 444688,
            "score": 0.790558825
          },
          {
            "movie_id": 11447,
            "score": 0.7904215
          }
        ]
      }
    }
  ]
}
```

### CE 2 — Viewer experience / feel / tone
_atomic_rewrite_: without a depressing tone

#### CE 2 payload
```json
{
  "captured_meaning": "The user wants to avoid a viewer experience characterized by a heavy, somber, or overwhelmingly sad tone.",
  "category_name": "Viewer experience / feel / tone",
  "atomic_rewrite": "without a depressing tone",
  "llm": {
    "elapsed_s": 4.459,
    "output": {
      "requirement_aspects": [
        {
          "aspect_description": "Avoid movies whose during-viewing tone feels depressing, heavy, somber, or overwhelmingly sad.",
          "relation_to_endpoint": "Semantic viewer_experience directly models during-viewing tone via emotional_palette and tone_self_seriousness, and it can express the negative boundary with negations. This is a clean fit for excluding a depressing feel.",
          "coverage_gaps": null
        }
      ],
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "trait",
        "parameters": {
          "qualifier_inventory": "Negative tonal ask: avoid a depressing, heavy, somber during-viewing feel. The grief theme is on a sibling fragment, not this atom.",
          "space_queries": [
            {
              "carries_qualifiers": "viewer_experience emotional_palette and tone_self_seriousness carry the during-viewing sad/somber/depressing register being excluded.",
              "space": "viewer_experience",
              "weight": "central",
              "content": {
                "emotional_palette": {
                  "terms": [
                    "depressing",
                    "somber",
                    "heavy"
                  ],
                  "negations": []
                },
                "tension_adrenaline": {
                  "terms": [],
                  "negations": []
                },
                "tone_self_seriousness": {
                  "terms": [
                    "melancholic"
                  ],
                  "negations": []
                },
                "cognitive_complexity": {
                  "terms": [],
                  "negations": []
                },
                "disturbance_profile": {
                  "terms": [],
                  "negations": []
                },
                "sensory_load": {
                  "terms": [],
                  "negations": []
                },
                "emotional_volatility": {
                  "terms": [],
                  "negations": []
                },
                "ending_aftertaste": {
                  "terms": [],
                  "negations": []
                }
              }
            }
          ],
          "primary_vector": "viewer_experience"
        },
        "polarity": "negative"
      }
    }
  },
  "endpoints": [
    {
      "endpoint": "semantic",
      "match_mode": "trait",
      "polarity": "negative",
      "wrapper_type": "SemanticEndpointParameters",
      "parameters": {
        "qualifier_inventory": "Negative tonal ask: avoid a depressing, heavy, somber during-viewing feel. The grief theme is on a sibling fragment, not this atom.",
        "space_queries": [
          {
            "carries_qualifiers": "viewer_experience emotional_palette and tone_self_seriousness carry the during-viewing sad/somber/depressing register being excluded.",
            "space": "viewer_experience",
            "weight": "central",
            "content": {
              "emotional_palette": {
                "terms": [
                  "depressing",
                  "somber",
                  "heavy"
                ],
                "negations": []
              },
              "tension_adrenaline": {
                "terms": [],
                "negations": []
              },
              "tone_self_seriousness": {
                "terms": [
                  "melancholic"
                ],
                "negations": []
              },
              "cognitive_complexity": {
                "terms": [],
                "negations": []
              },
              "disturbance_profile": {
                "terms": [],
                "negations": []
              },
              "sensory_load": {
                "terms": [],
                "negations": []
              },
              "emotional_volatility": {
                "terms": [],
                "negations": []
              },
              "ending_aftertaste": {
                "terms": [],
                "negations": []
              }
            }
          }
        ],
        "primary_vector": "viewer_experience"
      },
      "exec_elapsed_s": 0.385,
      "result": {
        "total_scored": 2000,
        "top": [
          {
            "movie_id": 453362,
            "score": 0.74378395
          },
          {
            "movie_id": 1353829,
            "score": 0.7277775
          },
          {
            "movie_id": 252539,
            "score": 0.7261181
          },
          {
            "movie_id": 221257,
            "score": 0.724329
          },
          {
            "movie_id": 35694,
            "score": 0.7192707
          },
          {
            "movie_id": 82723,
            "score": 0.7150898
          },
          {
            "movie_id": 408544,
            "score": 0.70783997
          },
          {
            "movie_id": 538632,
            "score": 0.7077465
          },
          {
            "movie_id": 29272,
            "score": 0.7077389
          },
          {
            "movie_id": 114171,
            "score": 0.70773315
          }
        ]
      }
    }
  ]
}
```

## Implicit Expectations

_elapsed: 1.64s_

#### Implicit Expectations Response
```json
{
  "query_intent_summary": "The user is looking for films centered on the theme of grief while explicitly requesting a non-depressing tone.",
  "explicit_signals": [
    {
      "query_span": "grief",
      "normalized_description": "thematic focus on the experience of loss",
      "explicit_axis": "neither"
    },
    {
      "query_span": "depressing",
      "normalized_description": "avoidance of a heavy or somber emotional tone",
      "explicit_axis": "neither"
    }
  ],
  "explicit_ordering_axis_analysis": "No explicit ordering axis is present. The query specifies thematic and tonal constraints but does not request a specific ranking by chronology, popularity, or semantic extremeness.",
  "explicitly_addresses_quality": false,
  "explicitly_addresses_notability": false,
  "should_apply_quality_prior": true,
  "should_apply_notability_prior": true
}
```

## Consolidation Buckets (post fan-out)

#### Consolidation summary
```json
{
  "inclusion_unique_ids": 0,
  "downrank_unique_ids": 2000,
  "exclusion_unique_ids": 0,
  "preference_specs_count": 1,
  "used_fallback": "preferences_as_candidates"
}
```

## Final Score Breakdowns (top 100)

#### Top 100 ScoreBreakdowns
```json
[
  {
    "movie_id": 858024,
    "inclusion_sum": 0.789924635,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.23986469850346853,
    "final_score": 1.0297893335034685
  },
  {
    "movie_id": 13668,
    "inclusion_sum": 0.8107233149999999,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.21092293686734592,
    "final_score": 1.021646251867346
  },
  {
    "movie_id": 27585,
    "inclusion_sum": 0.7874460249999999,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.23200074597760692,
    "final_score": 1.019446770977607
  },
  {
    "movie_id": 16619,
    "inclusion_sum": 0.779244415,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2399126027971154,
    "final_score": 1.0191570177971154
  },
  {
    "movie_id": 713776,
    "inclusion_sum": 0.7771339399999999,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.23403011022707101,
    "final_score": 1.011164050227071
  },
  {
    "movie_id": 438634,
    "inclusion_sum": 0.781454085,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.22608285171417952,
    "final_score": 1.0075369367141795
  },
  {
    "movie_id": 4032,
    "inclusion_sum": 0.7820815999999999,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2228121611351754,
    "final_score": 1.0048937611351754
  },
  {
    "movie_id": 641662,
    "inclusion_sum": 0.776119225,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.22753706835897372,
    "final_score": 1.0036562933589737
  },
  {
    "movie_id": 345920,
    "inclusion_sum": 0.7956953,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2060892956625258,
    "final_score": 1.0017845956625258
  },
  {
    "movie_id": 215042,
    "inclusion_sum": 0.776832575,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.22350239056568108,
    "final_score": 1.000334965565681
  },
  {
    "movie_id": 579583,
    "inclusion_sum": 0.771104825,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.22879873593070477,
    "final_score": 0.9999035609307048
  },
  {
    "movie_id": 31052,
    "inclusion_sum": 0.783646575,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.21620929358054414,
    "final_score": 0.9998558685805441
  },
  {
    "movie_id": 468225,
    "inclusion_sum": 0.7952146499999999,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.20430240383534817,
    "final_score": 0.9995170538353481
  },
  {
    "movie_id": 3877,
    "inclusion_sum": 0.773961075,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.22497108653990716,
    "final_score": 0.9989321615399072
  },
  {
    "movie_id": 31005,
    "inclusion_sum": 0.779274,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.21826365927098226,
    "final_score": 0.9975376592709823
  },
  {
    "movie_id": 55347,
    "inclusion_sum": 0.7610187500000001,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2363117304722656,
    "final_score": 0.9973304804722657
  },
  {
    "movie_id": 34653,
    "inclusion_sum": 0.7608089499999999,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.23553414364557163,
    "final_score": 0.9963430936455715
  },
  {
    "movie_id": 896,
    "inclusion_sum": 0.75777245,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2381951772447351,
    "final_score": 0.9959676272447351
  },
  {
    "movie_id": 99,
    "inclusion_sum": 0.7537441149999999,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.24142720104316728,
    "final_score": 0.9951713160431672
  },
  {
    "movie_id": 10511,
    "inclusion_sum": 0.76107407,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.23407093262972228,
    "final_score": 0.9951450026297224
  },
  {
    "movie_id": 78480,
    "inclusion_sum": 0.7604389250000001,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.23462966211972708,
    "final_score": 0.9950685871197271
  },
  {
    "movie_id": 749004,
    "inclusion_sum": 0.755427375,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.23922951893196248,
    "final_score": 0.9946568939319624
  },
  {
    "movie_id": 508883,
    "inclusion_sum": 0.752132425,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.24189065109776586,
    "final_score": 0.9940230760977659
  },
  {
    "movie_id": 313,
    "inclusion_sum": 0.7748355849999999,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2177745789870476,
    "final_score": 0.9926101639870475
  },
  {
    "movie_id": 2355,
    "inclusion_sum": 0.76550485,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2270283982377489,
    "final_score": 0.9925332482377489
  },
  {
    "movie_id": 248212,
    "inclusion_sum": 0.78124142,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.21117343719968287,
    "final_score": 0.9924148571996828
  },
  {
    "movie_id": 6023,
    "inclusion_sum": 0.7766418500000001,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2152611657272096,
    "final_score": 0.9919030157272097
  },
  {
    "movie_id": 157827,
    "inclusion_sum": 0.767758385,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2237788230013356,
    "final_score": 0.9915372080013356
  },
  {
    "movie_id": 758866,
    "inclusion_sum": 0.74946595,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.24204320960824438,
    "final_score": 0.9915091596082444
  },
  {
    "movie_id": 2015,
    "inclusion_sum": 0.760466565,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2308108216833575,
    "final_score": 0.9912773866833575
  },
  {
    "movie_id": 276401,
    "inclusion_sum": 0.772815705,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.21801513016195273,
    "final_score": 0.9908308351619527
  },
  {
    "movie_id": 994108,
    "inclusion_sum": 0.7487745299999999,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.24191319064004724,
    "final_score": 0.9906877206400472
  },
  {
    "movie_id": 1036619,
    "inclusion_sum": 0.7732668,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.21638900550853402,
    "final_score": 0.989655805508534
  },
  {
    "movie_id": 404141,
    "inclusion_sum": 0.763927475,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.22546573949499296,
    "final_score": 0.989393214494993
  },
  {
    "movie_id": 8816,
    "inclusion_sum": 0.752905835,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2353299606905752,
    "final_score": 0.9882357956905752
  },
  {
    "movie_id": 16071,
    "inclusion_sum": 0.7780714,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.20988101624901634,
    "final_score": 0.9879524162490163
  },
  {
    "movie_id": 258230,
    "inclusion_sum": 0.75293637,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.23448656995489375,
    "final_score": 0.9874229399548937
  },
  {
    "movie_id": 47002,
    "inclusion_sum": 0.75526715,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.23160525426405815,
    "final_score": 0.9868724042640582
  },
  {
    "movie_id": 23550,
    "inclusion_sum": 0.76402282,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.22274666466985593,
    "final_score": 0.9867694846698559
  },
  {
    "movie_id": 1214469,
    "inclusion_sum": 0.76069925,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2259517230076452,
    "final_score": 0.9866509730076453
  },
  {
    "movie_id": 376660,
    "inclusion_sum": 0.75159931,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.23502751534029104,
    "final_score": 0.986626825340291
  },
  {
    "movie_id": 64685,
    "inclusion_sum": 0.76838685,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2179097080769557,
    "final_score": 0.9862965580769556
  },
  {
    "movie_id": 419831,
    "inclusion_sum": 0.758667,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2274183964258238,
    "final_score": 0.9860853964258238
  },
  {
    "movie_id": 51857,
    "inclusion_sum": 0.7549152349999999,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.23101529059340492,
    "final_score": 0.9859305255934048
  },
  {
    "movie_id": 47760,
    "inclusion_sum": 0.771089555,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.21377221312221314,
    "final_score": 0.9848617681222132
  },
  {
    "movie_id": 8967,
    "inclusion_sum": 0.746619235,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.23750732949583486,
    "final_score": 0.9841265644958349
  },
  {
    "movie_id": 645484,
    "inclusion_sum": 0.74119853,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.24129555120362742,
    "final_score": 0.9824940812036275
  },
  {
    "movie_id": 239678,
    "inclusion_sum": 0.765877725,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.21612389106212554,
    "final_score": 0.9820016160621254
  },
  {
    "movie_id": 937278,
    "inclusion_sum": 0.7590599,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.22288850072992347,
    "final_score": 0.9819484007299235
  },
  {
    "movie_id": 17170,
    "inclusion_sum": 0.774753585,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.20718334068392627,
    "final_score": 0.9819369256839263
  },
  {
    "movie_id": 785084,
    "inclusion_sum": 0.75405693,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.22786265601195782,
    "final_score": 0.9819195860119578
  },
  {
    "movie_id": 25050,
    "inclusion_sum": 0.7417345049999999,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.23924527959672687,
    "final_score": 0.9809797845967267
  },
  {
    "movie_id": 397601,
    "inclusion_sum": 0.77687835,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.20383622718056577,
    "final_score": 0.9807145771805659
  },
  {
    "movie_id": 866,
    "inclusion_sum": 0.74936963,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.23124192467403634,
    "final_score": 0.9806115546740364
  },
  {
    "movie_id": 11161,
    "inclusion_sum": 0.7624120750000001,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.21788276120787087,
    "final_score": 0.980294836207871
  },
  {
    "movie_id": 842924,
    "inclusion_sum": 0.75064275,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.22918744006557115,
    "final_score": 0.9798301900655711
  },
  {
    "movie_id": 38523,
    "inclusion_sum": 0.77001953,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.20951063461807637,
    "final_score": 0.9795301646180764
  },
  {
    "movie_id": 840430,
    "inclusion_sum": 0.73947047,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.23980402974519588,
    "final_score": 0.9792744997451959
  },
  {
    "movie_id": 222935,
    "inclusion_sum": 0.746657385,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.23248676012801425,
    "final_score": 0.9791441451280143
  },
  {
    "movie_id": 303991,
    "inclusion_sum": 0.7592048499999999,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2199333285694704,
    "final_score": 0.9791381785694704
  },
  {
    "movie_id": 817866,
    "inclusion_sum": 0.75675677,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.22205794215801275,
    "final_score": 0.9788147121580127
  },
  {
    "movie_id": 53113,
    "inclusion_sum": 0.7581253,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2203213361615819,
    "final_score": 0.9784466361615819
  },
  {
    "movie_id": 493899,
    "inclusion_sum": 0.7552576,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.22316072120100802,
    "final_score": 0.978418321201008
  },
  {
    "movie_id": 348678,
    "inclusion_sum": 0.7461939,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.23218319494239004,
    "final_score": 0.97837709494239
  },
  {
    "movie_id": 11354,
    "inclusion_sum": 0.75517367,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.22316968571310733,
    "final_score": 0.9783433557131074
  },
  {
    "movie_id": 621013,
    "inclusion_sum": 0.760896685,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.21730012504661061,
    "final_score": 0.9781968100466105
  },
  {
    "movie_id": 16804,
    "inclusion_sum": 0.746393205,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.23180128604224298,
    "final_score": 0.9781944910422429
  },
  {
    "movie_id": 565310,
    "inclusion_sum": 0.73708535,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2410301281713881,
    "final_score": 0.9781154781713881
  },
  {
    "movie_id": 310121,
    "inclusion_sum": 0.755085,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2226743865814741,
    "final_score": 0.9777593865814741
  },
  {
    "movie_id": 25538,
    "inclusion_sum": 0.73349953,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.24407913657038544,
    "final_score": 0.9775786665703854
  },
  {
    "movie_id": 31007,
    "inclusion_sum": 0.7597598999999999,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.21731053323054159,
    "final_score": 0.9770704332305415
  },
  {
    "movie_id": 44835,
    "inclusion_sum": 0.7600889200000001,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.21659487278734546,
    "final_score": 0.9766837927873455
  },
  {
    "movie_id": 25126,
    "inclusion_sum": 0.747899055,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2285626394725881,
    "final_score": 0.9764616944725881
  },
  {
    "movie_id": 59468,
    "inclusion_sum": 0.74973773,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.22667300843337734,
    "final_score": 0.9764107384333773
  },
  {
    "movie_id": 14857,
    "inclusion_sum": 0.7622614,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.21407426333829144,
    "final_score": 0.9763356633382914
  },
  {
    "movie_id": 23169,
    "inclusion_sum": 0.7603912500000001,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.21590941898847146,
    "final_score": 0.9763006689884716
  },
  {
    "movie_id": 560050,
    "inclusion_sum": 0.7551546149999999,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.22044034607183052,
    "final_score": 0.9755949610718304
  },
  {
    "movie_id": 25643,
    "inclusion_sum": 0.769867895,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2057066402927174,
    "final_score": 0.9755745352927174
  },
  {
    "movie_id": 65057,
    "inclusion_sum": 0.73669435,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2388268959120259,
    "final_score": 0.9755212459120259
  },
  {
    "movie_id": 1587,
    "inclusion_sum": 0.7406435149999999,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2346751346803838,
    "final_score": 0.9753186496803837
  },
  {
    "movie_id": 127370,
    "inclusion_sum": 0.7629404,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.21165862870128752,
    "final_score": 0.9745990287012875
  },
  {
    "movie_id": 100271,
    "inclusion_sum": 0.756728175,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2177355563420669,
    "final_score": 0.9744637313420669
  },
  {
    "movie_id": 11050,
    "inclusion_sum": 0.7387743,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2356062545113381,
    "final_score": 0.9743805545113381
  },
  {
    "movie_id": 308369,
    "inclusion_sum": 0.73941517,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.23482992097791056,
    "final_score": 0.9742450909779106
  },
  {
    "movie_id": 390592,
    "inclusion_sum": 0.787875175,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.18609612997589553,
    "final_score": 0.9739713049758956
  },
  {
    "movie_id": 1245347,
    "inclusion_sum": 0.7444248250000001,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.22935807634317798,
    "final_score": 0.973782901343178
  },
  {
    "movie_id": 8291,
    "inclusion_sum": 0.759385125,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.21399437134958957,
    "final_score": 0.9733794963495895
  },
  {
    "movie_id": 48617,
    "inclusion_sum": 0.749588015,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.22351100984396494,
    "final_score": 0.973099024843965
  },
  {
    "movie_id": 42726,
    "inclusion_sum": 0.744403825,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.22839319731543928,
    "final_score": 0.9727970223154393
  },
  {
    "movie_id": 783675,
    "inclusion_sum": 0.7407674849999999,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.23170750483601302,
    "final_score": 0.9724749898360129
  },
  {
    "movie_id": 1186563,
    "inclusion_sum": 0.7569294,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2138634292558693,
    "final_score": 0.9707928292558693
  },
  {
    "movie_id": 284276,
    "inclusion_sum": 0.75444032,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.216290263015964,
    "final_score": 0.970730583015964
  },
  {
    "movie_id": 1265,
    "inclusion_sum": 0.7372922799999999,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2333356152327427,
    "final_score": 0.9706278952327426
  },
  {
    "movie_id": 745,
    "inclusion_sum": 0.7383337,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.23211526429422136,
    "final_score": 0.9704489642942213
  },
  {
    "movie_id": 254172,
    "inclusion_sum": 0.76175785,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.20864330786163915,
    "final_score": 0.9704011578616392
  },
  {
    "movie_id": 5000,
    "inclusion_sum": 0.737435335,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.23294103440141734,
    "final_score": 0.9703763694014174
  },
  {
    "movie_id": 312804,
    "inclusion_sum": 0.746559135,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.22378122455632868,
    "final_score": 0.9703403595563287
  },
  {
    "movie_id": 722778,
    "inclusion_sum": 0.73695373,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.23338168317709157,
    "final_score": 0.9703354131770916
  },
  {
    "movie_id": 126337,
    "inclusion_sum": 0.76338675,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2062950348677134,
    "final_score": 0.9696817848677134
  },
  {
    "movie_id": 86835,
    "inclusion_sum": 0.75444222,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.21507632303815355,
    "final_score": 0.9695185430381537
  }
]
```

## Step 4 — Summary

### Filters
- (none)

### Traits
- about the thematic experience of grief
- without a depressing tone

_used_fallback: preferences_as_candidates_

### query_intent_summary
The user is looking for films centered on the theme of grief while explicitly requesting a non-depressing tone.

_implicit priors: quality=on  popularity=on_

### Top 100 Results

| # | final | filter | pref | down | impl | title (year) | tmdb_id |
|---|-------|--------|------|------|------|--------------|---------|
| 1 | 1.0298 | 0.7899 | 0.0000 | -0.0000 | 0.2399 | Hamnet (2025) | 858024 |
| 2 | 1.0216 | 0.8107 | 0.0000 | -0.0000 | 0.2109 | Catch and Release (2006) | 13668 |
| 3 | 1.0194 | 0.7874 | 0.0000 | -0.0000 | 0.2320 | Rabbit Hole (2010) | 27585 |
| 4 | 1.0192 | 0.7792 | 0.0000 | -0.0000 | 0.2399 | Ordinary People (1980) | 16619 |
| 5 | 1.0112 | 0.7771 | 0.0000 | -0.0000 | 0.2340 | If Anything Happens I Love You (2020) | 713776 |
| 6 | 1.0075 | 0.7815 | 0.0000 | -0.0000 | 0.2261 | Summer 1993 (2017) | 438634 |
| 7 | 1.0049 | 0.7821 | 0.0000 | -0.0000 | 0.2228 | My Girl (1991) | 4032 |
| 8 | 1.0037 | 0.7761 | 0.0000 | -0.0000 | 0.2275 | Pieces of a Woman (2020) | 641662 |
| 9 | 1.0018 | 0.7957 | 0.0000 | -0.0000 | 0.2061 | Collateral Beauty (2016) | 345920 |
| 10 | 1.0003 | 0.7768 | 0.0000 | -0.0000 | 0.2235 | Metalhead (2013) | 215042 |
| 11 | 0.9999 | 0.7711 | 0.0000 | -0.0000 | 0.2288 | The King of Staten Island (2020) | 579583 |
| 12 | 0.9999 | 0.7836 | 0.0000 | -0.0000 | 0.2162 | The Accidental Tourist (1988) | 31052 |
| 13 | 0.9995 | 0.7952 | 0.0000 | -0.0000 | 0.2043 | The Starling (2021) | 468225 |
| 14 | 0.9989 | 0.7740 | 0.0000 | -0.0000 | 0.2250 | Things We Lost in the Fire (2007) | 3877 |
| 15 | 0.9975 | 0.7793 | 0.0000 | -0.0000 | 0.2183 | Moonlight Mile (2002) | 31005 |
| 16 | 0.9973 | 0.7610 | 0.0000 | -0.0000 | 0.2363 | Beginners (2011) | 55347 |
| 17 | 0.9963 | 0.7608 | 0.0000 | -0.0000 | 0.2355 | A Single Man (2009) | 34653 |
| 18 | 0.9960 | 0.7578 | 0.0000 | -0.0000 | 0.2382 | The World of Apu (1959) | 896 |
| 19 | 0.9952 | 0.7537 | 0.0000 | -0.0000 | 0.2414 | All About My Mother (1999) | 99 |
| 20 | 0.9951 | 0.7611 | 0.0000 | -0.0000 | 0.2341 | In America (2003) | 10511 |
| 21 | 0.9951 | 0.7604 | 0.0000 | -0.0000 | 0.2346 | Monsieur Lazhar (2011) | 78480 |
| 22 | 0.9947 | 0.7554 | 0.0000 | -0.0000 | 0.2392 | Petite Maman (2021) | 749004 |
| 23 | 0.9940 | 0.7521 | 0.0000 | -0.0000 | 0.2419 | The Boy and the Heron (2023) | 508883 |
| 24 | 0.9926 | 0.7748 | 0.0000 | -0.0000 | 0.2178 | Snow Cake (2006) | 313 |
| 25 | 0.9925 | 0.7655 | 0.0000 | -0.0000 | 0.2270 | Reign Over Me (2007) | 2355 |
| 26 | 0.9924 | 0.7812 | 0.0000 | -0.0000 | 0.2112 | Lilting (2014) | 248212 |
| 27 | 0.9919 | 0.7766 | 0.0000 | -0.0000 | 0.2153 | P.S. I Love You (2007) | 6023 |
| 28 | 0.9915 | 0.7678 | 0.0000 | -0.0000 | 0.2238 | Louder Than Bombs (2015) | 157827 |
| 29 | 0.9915 | 0.7495 | 0.0000 | -0.0000 | 0.2420 | Drive My Car (2021) | 758866 |
| 30 | 0.9913 | 0.7605 | 0.0000 | -0.0000 | 0.2308 | Secret Sunshine (2007) | 2015 |
| 31 | 0.9908 | 0.7728 | 0.0000 | -0.0000 | 0.2180 | The Disappearance of Eleanor Rigby: Him (2014) | 276401 |
| 32 | 0.9907 | 0.7488 | 0.0000 | -0.0000 | 0.2419 | All of Us Strangers (2023) | 994108 |
| 33 | 0.9897 | 0.7733 | 0.0000 | -0.0000 | 0.2164 | Good Grief (2023) | 1036619 |
| 34 | 0.9894 | 0.7639 | 0.0000 | -0.0000 | 0.2255 | Nick Cave & The Bad Seeds: One More Time with Feeling (2016) | 404141 |
| 35 | 0.9882 | 0.7529 | 0.0000 | -0.0000 | 0.2353 | My Life as a Dog (1985) | 8816 |
| 36 | 0.9880 | 0.7781 | 0.0000 | -0.0000 | 0.2099 | The Cake Eaters (2007) | 16071 |
| 37 | 0.9874 | 0.7529 | 0.0000 | -0.0000 | 0.2345 | A Monster Calls (2016) | 258230 |
| 38 | 0.9869 | 0.7553 | 0.0000 | -0.0000 | 0.2316 | Love Letter (1995) | 47002 |
| 39 | 0.9868 | 0.7640 | 0.0000 | -0.0000 | 0.2227 | His Secret Life (2001) | 23550 |
| 40 | 0.9867 | 0.7607 | 0.0000 | -0.0000 | 0.2260 | Ghostlight (2024) | 1214469 |
| 41 | 0.9866 | 0.7516 | 0.0000 | -0.0000 | 0.2350 | The Edge of Seventeen (2016) | 376660 |
| 42 | 0.9863 | 0.7684 | 0.0000 | -0.0000 | 0.2179 | Extremely Loud & Incredibly Close (2011) | 64685 |
| 43 | 0.9861 | 0.7587 | 0.0000 | -0.0000 | 0.2274 | I Kill Giants (2017) | 419831 |
| 44 | 0.9859 | 0.7549 | 0.0000 | -0.0000 | 0.2310 | Cria! (1976) | 51857 |
| 45 | 0.9849 | 0.7711 | 0.0000 | -0.0000 | 0.2138 | Restless (2011) | 47760 |
| 46 | 0.9841 | 0.7466 | 0.0000 | -0.0000 | 0.2375 | The Tree of Life (2011) | 8967 |
| 47 | 0.9825 | 0.7412 | 0.0000 | -0.0000 | 0.2413 | Dil Bechara (2020) | 645484 |
| 48 | 0.9820 | 0.7659 | 0.0000 | -0.0000 | 0.2161 | This Is Where I Leave You (2014) | 239678 |
| 49 | 0.9819 | 0.7591 | 0.0000 | -0.0000 | 0.2229 | A Man Called Otto (2022) | 937278 |
| 50 | 0.9819 | 0.7748 | 0.0000 | -0.0000 | 0.2072 | Bright Lights, Big City (1988) | 17170 |
| 51 | 0.9819 | 0.7541 | 0.0000 | -0.0000 | 0.2279 | The Whale (2022) | 785084 |
| 52 | 0.9810 | 0.7417 | 0.0000 | -0.0000 | 0.2392 | Still Walking (2008) | 25050 |
| 53 | 0.9807 | 0.7769 | 0.0000 | -0.0000 | 0.2038 | The Bachelors (2017) | 397601 |
| 54 | 0.9806 | 0.7494 | 0.0000 | -0.0000 | 0.2312 | Finding Neverland (2004) | 866 |
| 55 | 0.9803 | 0.7624 | 0.0000 | -0.0000 | 0.2179 | Grace Is Gone (2007) | 11161 |
| 56 | 0.9798 | 0.7506 | 0.0000 | -0.0000 | 0.2292 | The Life of Chuck (2025) | 842924 |
| 57 | 0.9795 | 0.7700 | 0.0000 | -0.0000 | 0.2095 | Ponette (1996) | 38523 |
| 58 | 0.9793 | 0.7395 | 0.0000 | -0.0000 | 0.2398 | The Holdovers (2023) | 840430 |
| 59 | 0.9791 | 0.7467 | 0.0000 | -0.0000 | 0.2325 | The Fault in Our Stars (2014) | 222935 |
| 60 | 0.9791 | 0.7592 | 0.0000 | -0.0000 | 0.2199 | Demolition (2016) | 303991 |
| 61 | 0.9788 | 0.7568 | 0.0000 | -0.0000 | 0.2221 | Goodbye (2022) | 817866 |
| 62 | 0.9784 | 0.7581 | 0.0000 | -0.0000 | 0.2203 | One True Thing (1998) | 53113 |
| 63 | 0.9784 | 0.7553 | 0.0000 | -0.0000 | 0.2232 | So Long, My Son (2019) | 493899 |
| 64 | 0.9784 | 0.7462 | 0.0000 | -0.0000 | 0.2322 | A Man Called Ove (2015) | 348678 |
| 65 | 0.9783 | 0.7552 | 0.0000 | -0.0000 | 0.2232 | The Upside of Anger (2005) | 11354 |
| 66 | 0.9782 | 0.7609 | 0.0000 | -0.0000 | 0.2173 | Chemical Hearts (2020) | 621013 |
| 67 | 0.9782 | 0.7464 | 0.0000 | -0.0000 | 0.2318 | Departures (2008) | 16804 |
| 68 | 0.9781 | 0.7371 | 0.0000 | -0.0000 | 0.2410 | The Farewell (2019) | 565310 |
| 69 | 0.9778 | 0.7551 | 0.0000 | -0.0000 | 0.2227 | I'll See You in My Dreams (2015) | 310121 |
| 70 | 0.9776 | 0.7335 | 0.0000 | -0.0000 | 0.2441 | Yi Yi (2000) | 25538 |
| 71 | 0.9771 | 0.7598 | 0.0000 | -0.0000 | 0.2173 | Welcome to the Rileys (2010) | 31007 |
| 72 | 0.9767 | 0.7601 | 0.0000 | -0.0000 | 0.2166 | Hesher (2010) | 44835 |
| 73 | 0.9765 | 0.7479 | 0.0000 | -0.0000 | 0.2286 | Six Shooter (2004) | 25126 |
| 74 | 0.9764 | 0.7497 | 0.0000 | -0.0000 | 0.2267 | The Way (2010) | 59468 |
| 75 | 0.9763 | 0.7623 | 0.0000 | -0.0000 | 0.2141 | Personal Effects (2009) | 14857 |
| 76 | 0.9763 | 0.7604 | 0.0000 | -0.0000 | 0.2159 | Remember Me (2010) | 23169 |
| 77 | 0.9756 | 0.7552 | 0.0000 | -0.0000 | 0.2204 | Over the Moon (2020) | 560050 |
| 78 | 0.9756 | 0.7699 | 0.0000 | -0.0000 | 0.2057 | Love Happens (2009) | 25643 |
| 79 | 0.9755 | 0.7367 | 0.0000 | -0.0000 | 0.2388 | The Descendants (2011) | 65057 |
| 80 | 0.9753 | 0.7406 | 0.0000 | -0.0000 | 0.2347 | What's Eating Gilbert Grape (1993) | 1587 |
| 81 | 0.9746 | 0.7629 | 0.0000 | -0.0000 | 0.2117 | Song for Marion (2012) | 127370 |
| 82 | 0.9745 | 0.7567 | 0.0000 | -0.0000 | 0.2177 | A Letter to Momo (2012) | 100271 |
| 83 | 0.9744 | 0.7388 | 0.0000 | -0.0000 | 0.2356 | Terms of Endearment (1983) | 11050 |
| 84 | 0.9742 | 0.7394 | 0.0000 | -0.0000 | 0.2348 | Me and Earl and the Dying Girl (2015) | 308369 |
| 85 | 0.9740 | 0.7879 | 0.0000 | -0.0000 | 0.1861 | Adult Life Skills (2016) | 390592 |
| 86 | 0.9738 | 0.7444 | 0.0000 | -0.0000 | 0.2294 | Twinless (2025) | 1245347 |
| 87 | 0.9734 | 0.7594 | 0.0000 | -0.0000 | 0.2140 | Poetic Justice (1993) | 8291 |
| 88 | 0.9731 | 0.7496 | 0.0000 | -0.0000 | 0.2235 | Father and Daughter (2001) | 48617 |
| 89 | 0.9728 | 0.7444 | 0.0000 | -0.0000 | 0.2284 | A Man and a Woman (1966) | 42726 |
| 90 | 0.9725 | 0.7408 | 0.0000 | -0.0000 | 0.2317 | The First Slam Dunk (2022) | 783675 |
| 91 | 0.9708 | 0.7569 | 0.0000 | -0.0000 | 0.2139 | The Friend (2025) | 1186563 |
| 92 | 0.9707 | 0.7544 | 0.0000 | -0.0000 | 0.2163 | Cake (2014) | 284276 |
| 93 | 0.9706 | 0.7373 | 0.0000 | -0.0000 | 0.2333 | Bridge to Terabithia (2007) | 1265 |
| 94 | 0.9704 | 0.7383 | 0.0000 | -0.0000 | 0.2321 | The Sixth Sense (1999) | 745 |
| 95 | 0.9704 | 0.7618 | 0.0000 | -0.0000 | 0.2086 | Fathers and Daughters (2015) | 254172 |
| 96 | 0.9704 | 0.7374 | 0.0000 | -0.0000 | 0.2329 | Forbidden Games (1952) | 5000 |
| 97 | 0.9703 | 0.7466 | 0.0000 | -0.0000 | 0.2238 | James White (2015) | 312804 |
| 98 | 0.9703 | 0.7370 | 0.0000 | -0.0000 | 0.2334 | The Hand of God (2021) | 722778 |
| 99 | 0.9697 | 0.7634 | 0.0000 | -0.0000 | 0.2063 | Unconditional (2012) | 126337 |
| 100 | 0.9695 | 0.7544 | 0.0000 | -0.0000 | 0.2151 | Knight of Cups (2015) | 86835 |

_total elapsed: 10.13s_
