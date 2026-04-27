# Query

> I want something light and feel-good but with actual substance, around 90 minutes, on a streaming service.

## Step 2 — Query Pre-pass

_elapsed: 4.86s_

#### Step 2 Response
```json
{
  "overall_query_intention_exploration": "The user is looking for a movie with a specific tonal balance, seeking a lighthearted experience that still offers thematic depth or quality writing. They have a strict constraint on runtime (short duration) and a general requirement for availability on digital platforms.",
  "requirements": [
    {
      "query_text": "light",
      "description": "Specifies a tone that is not heavy, dark, or overly serious.",
      "modifiers": [],
      "coverage_evidence": [
        {
          "captured_meaning": "The user wants a movie with a lighthearted and buoyant tonal aesthetic.",
          "category_name": "Viewer experience / feel / tone",
          "fit_quality": "clean",
          "atomic_rewrite": "with a light tone"
        }
      ]
    },
    {
      "query_text": "feel-good",
      "description": "Specifies a movie intended to evoke positive emotions and a sense of well-being.",
      "modifiers": [],
      "coverage_evidence": [
        {
          "captured_meaning": "The movie is intended to cheer the viewer up or provide a positive emotional experience.",
          "category_name": "Occasion / self-experience goal / comfort-watch",
          "fit_quality": "clean",
          "atomic_rewrite": "a feel-good movie to cheer me up"
        }
      ]
    },
    {
      "query_text": "actual substance",
      "description": "Indicates a desire for thematic depth, quality storytelling, or intellectual engagement despite the light tone.",
      "modifiers": [
        {
          "original_text": "but with",
          "effect": "binds a contrasting quality of depth to the preceding light tone",
          "type": "polarity_modifier"
        }
      ],
      "coverage_evidence": [
        {
          "captured_meaning": "The user wants a story with thematic depth or narrative weight.",
          "category_name": "Kind of story / thematic archetype",
          "fit_quality": "partial",
          "atomic_rewrite": "with actual narrative substance"
        },
        {
          "captured_meaning": "The request implies a level of storytelling craft or intellectual engagement.",
          "category_name": "Narrative devices + structural form + craft",
          "fit_quality": "partial",
          "atomic_rewrite": "with substantive storytelling"
        }
      ]
    },
    {
      "query_text": "90 minutes",
      "description": "Specifies a desired runtime for the movie.",
      "modifiers": [
        {
          "original_text": "around",
          "effect": "applies a rough approximation to the numeric constraint",
          "type": "polarity_modifier"
        }
      ],
      "coverage_evidence": [
        {
          "captured_meaning": "The user is filtering for movies with a specific duration of roughly one and a half hours.",
          "category_name": "Structured metadata",
          "fit_quality": "clean",
          "atomic_rewrite": "with a runtime around 90 minutes"
        }
      ]
    },
    {
      "query_text": "streaming service",
      "description": "Specifies the desired distribution or availability platform.",
      "modifiers": [],
      "coverage_evidence": [
        {
          "captured_meaning": "The user is looking for movies available on digital streaming platforms.",
          "category_name": "Structured metadata",
          "fit_quality": "clean",
          "atomic_rewrite": "available on a streaming service"
        }
      ]
    }
  ]
}
```

## Step 3 — Endpoint Translation + Execution

### CE 1 — Viewer experience / feel / tone
_atomic_rewrite_: with a light tone

#### CE 1 payload
```json
{
  "captured_meaning": "The user wants a movie with a lighthearted and buoyant tonal aesthetic.",
  "category_name": "Viewer experience / feel / tone",
  "atomic_rewrite": "with a light tone",
  "llm": {
    "elapsed_s": 4.018,
    "output": {
      "requirement_aspects": [
        {
          "aspect_description": "A light, feel-good during-viewing tone with positive, buoyant vibes.",
          "relation_to_endpoint": "Semantic viewer_experience directly covers tonal aesthetic and emotional palette. 'Light' and 'feel-good' fit emotional_palette and tone_self_seriousness as warm, upbeat, low-weight-feeling descriptors.",
          "coverage_gaps": "None for the tonal part; runtime and streaming availability are outside this endpoint."
        },
        {
          "aspect_description": "A movie with 'actual substance' despite the light tone — some thematic depth or thoughtful engagement.",
          "relation_to_endpoint": "Viewer_experience can carry the 'feel-good but not empty' side only weakly via cognitive_complexity / emotional_palette, but it does not truly encode thematic depth or writing quality as a content attribute.",
          "coverage_gaps": "The substantive/depth aspect is better handled by plot_analysis or reception, not this endpoint."
        }
      ],
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "trait",
        "parameters": {
          "qualifier_inventory": "Lighthearted, feel-good tone; wants some substantive depth alongside that tone.",
          "space_queries": [
            {
              "carries_qualifiers": "Lighthearted, buoyant, positive mood; a feel-good during-viewing experience.",
              "space": "viewer_experience",
              "weight": "central",
              "content": {
                "emotional_palette": {
                  "terms": [
                    "light",
                    "feel-good",
                    "upbeat",
                    "buoyant"
                  ],
                  "negations": []
                },
                "tension_adrenaline": {
                  "terms": [],
                  "negations": []
                },
                "tone_self_seriousness": {
                  "terms": [
                    "not too serious"
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
        "qualifier_inventory": "Lighthearted, feel-good tone; wants some substantive depth alongside that tone.",
        "space_queries": [
          {
            "carries_qualifiers": "Lighthearted, buoyant, positive mood; a feel-good during-viewing experience.",
            "space": "viewer_experience",
            "weight": "central",
            "content": {
              "emotional_palette": {
                "terms": [
                  "light",
                  "feel-good",
                  "upbeat",
                  "buoyant"
                ],
                "negations": []
              },
              "tension_adrenaline": {
                "terms": [],
                "negations": []
              },
              "tone_self_seriousness": {
                "terms": [
                  "not too serious"
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
      "exec_elapsed_s": 1.007,
      "result": {
        "total_scored": 2000,
        "top": [
          {
            "movie_id": 1499180,
            "score": 0.7727337
          },
          {
            "movie_id": 42651,
            "score": 0.76010513
          },
          {
            "movie_id": 122103,
            "score": 0.7514801
          },
          {
            "movie_id": 1266701,
            "score": 0.7478676
          },
          {
            "movie_id": 124829,
            "score": 0.74526215
          },
          {
            "movie_id": 43390,
            "score": 0.7451782
          },
          {
            "movie_id": 455359,
            "score": 0.7425232
          },
          {
            "movie_id": 376459,
            "score": 0.74098015
          },
          {
            "movie_id": 89606,
            "score": 0.7392788
          },
          {
            "movie_id": 1353829,
            "score": 0.737772
          }
        ]
      }
    }
  ]
}
```

### CE 2 — Occasion / self-experience goal / comfort-watch
_atomic_rewrite_: a feel-good movie to cheer me up

#### CE 2 payload
```json
{
  "captured_meaning": "The movie is intended to cheer the viewer up or provide a positive emotional experience.",
  "category_name": "Occasion / self-experience goal / comfort-watch",
  "atomic_rewrite": "a feel-good movie to cheer me up",
  "llm": {
    "elapsed_s": 5.868,
    "output": {
      "requirement_aspects": [
        {
          "aspect_description": "A movie that is feel-good / intended to cheer the viewer up.",
          "endpoint_coverage": "SEMANTIC can capture the viewing motivation in watch_context and the uplifting feel in viewer_experience; KEYWORD can anchor FEEL_GOOD. Reception is only relevant if the query invokes a reviewer label, which it does not. Runtime and streaming are separate metadata concerns."
        },
        {
          "aspect_description": "A light movie with actual substance.",
          "endpoint_coverage": "SEMANTIC can capture the light tone plus the added depth/substance through viewer_experience and plot_analysis; KEYWORD does not cover this tonal/depth balance."
        },
        {
          "aspect_description": "Around 90 minutes, on a streaming service.",
          "endpoint_coverage": "Neither SEMANTIC nor KEYWORD should handle these here; they belong to metadata / availability endpoints, not this category."
        }
      ],
      "overall_endpoint_fits": "SEMANTIC fits the feel-good/light-with-substance part, with watch_context for the comfort-watch goal and viewer_experience for the light/uplifting tone; plot_analysis can add the substance/depth signal if needed. KEYWORD also fits via FEEL_GOOD. Runtime and streaming availability are out of scope for these endpoints.",
      "per_endpoint_breakdown": {
        "semantic": {
          "should_run_endpoint": true,
          "endpoint_parameters": {
            "match_mode": "trait",
            "parameters": {
              "qualifier_inventory": "self-experience goal is feel-good / cheering up; tone is light; secondary request is substance / thematic depth.",
              "space_queries": [
                {
                  "carries_qualifiers": "watch_context captures the intended viewing goal: a movie that feels good and cheers the viewer up.",
                  "space": "watch_context",
                  "weight": "central",
                  "content": {
                    "self_experience_motivations": {
                      "terms": [
                        "feel-good",
                        "cheer me up",
                        "feel better"
                      ]
                    },
                    "external_motivations": {
                      "terms": []
                    },
                    "key_movie_feature_draws": {
                      "terms": []
                    },
                    "watch_scenarios": {
                      "terms": []
                    }
                  }
                },
                {
                  "carries_qualifiers": "viewer_experience captures the light, uplifting, low-weight tone the movie should have.",
                  "space": "viewer_experience",
                  "weight": "central",
                  "content": {
                    "emotional_palette": {
                      "terms": [
                        "uplifting",
                        "warm",
                        "light",
                        "feel-good"
                      ],
                      "negations": [
                        "bleak",
                        "heavy"
                      ]
                    },
                    "tension_adrenaline": {
                      "terms": [],
                      "negations": []
                    },
                    "tone_self_seriousness": {
                      "terms": [
                        "light"
                      ],
                      "negations": [
                        "overly serious"
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
                      "terms": [],
                      "negations": []
                    },
                    "ending_aftertaste": {
                      "terms": [],
                      "negations": []
                    }
                  }
                },
                {
                  "carries_qualifiers": "plot_analysis captures the request for actual substance / thematic depth despite the light tone.",
                  "space": "plot_analysis",
                  "weight": "supporting",
                  "content": {
                    "elevator_pitch": "Light but substantive, with thoughtful writing and thematic depth.",
                    "plot_overview": null,
                    "genre_signatures": [],
                    "conflict_type": [],
                    "thematic_concepts": [
                      "thematic depth",
                      "quality storytelling",
                      "intellectual engagement"
                    ],
                    "character_arcs": []
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
              "concept_analysis": "The query asks for a feel-good movie intended to uplift the viewer.",
              "candidate_shortlist": "FEEL_GOOD directly names the intended overall effect; TEARJERKER is the opposite, and HAPPY_ENDING is a different axis.",
              "classification": "FEEL_GOOD"
            },
            "polarity": "positive"
          }
        }
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
        "qualifier_inventory": "self-experience goal is feel-good / cheering up; tone is light; secondary request is substance / thematic depth.",
        "space_queries": [
          {
            "carries_qualifiers": "watch_context captures the intended viewing goal: a movie that feels good and cheers the viewer up.",
            "space": "watch_context",
            "weight": "central",
            "content": {
              "self_experience_motivations": {
                "terms": [
                  "feel-good",
                  "cheer me up",
                  "feel better"
                ]
              },
              "external_motivations": {
                "terms": []
              },
              "key_movie_feature_draws": {
                "terms": []
              },
              "watch_scenarios": {
                "terms": []
              }
            }
          },
          {
            "carries_qualifiers": "viewer_experience captures the light, uplifting, low-weight tone the movie should have.",
            "space": "viewer_experience",
            "weight": "central",
            "content": {
              "emotional_palette": {
                "terms": [
                  "uplifting",
                  "warm",
                  "light",
                  "feel-good"
                ],
                "negations": [
                  "bleak",
                  "heavy"
                ]
              },
              "tension_adrenaline": {
                "terms": [],
                "negations": []
              },
              "tone_self_seriousness": {
                "terms": [
                  "light"
                ],
                "negations": [
                  "overly serious"
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
                "terms": [],
                "negations": []
              },
              "ending_aftertaste": {
                "terms": [],
                "negations": []
              }
            }
          },
          {
            "carries_qualifiers": "plot_analysis captures the request for actual substance / thematic depth despite the light tone.",
            "space": "plot_analysis",
            "weight": "supporting",
            "content": {
              "elevator_pitch": "Light but substantive, with thoughtful writing and thematic depth.",
              "plot_overview": null,
              "genre_signatures": [],
              "conflict_type": [],
              "thematic_concepts": [
                "thematic depth",
                "quality storytelling",
                "intellectual engagement"
              ],
              "character_arcs": []
            }
          }
        ],
        "primary_vector": "watch_context"
      },
      "exec_elapsed_s": 2.347,
      "result": {
        "total_scored": 5553,
        "top": [
          {
            "movie_id": 86577,
            "score": 0.660101316
          },
          {
            "movie_id": 413543,
            "score": 0.6498703000000001
          },
          {
            "movie_id": 582434,
            "score": 0.6479026720000001
          },
          {
            "movie_id": 1467974,
            "score": 0.647801976
          },
          {
            "movie_id": 436548,
            "score": 0.644365692
          },
          {
            "movie_id": 79897,
            "score": 0.6369529600000001
          },
          {
            "movie_id": 53216,
            "score": 0.636870576
          },
          {
            "movie_id": 372297,
            "score": 0.6352920479999999
          },
          {
            "movie_id": 57280,
            "score": 0.63451385
          },
          {
            "movie_id": 2357,
            "score": 0.632906334
          }
        ]
      }
    },
    {
      "endpoint": "keyword",
      "match_mode": "trait",
      "polarity": "positive",
      "wrapper_type": "KeywordEndpointParameters",
      "parameters": {
        "concept_analysis": "The query asks for a feel-good movie intended to uplift the viewer.",
        "candidate_shortlist": "FEEL_GOOD directly names the intended overall effect; TEARJERKER is the opposite, and HAPPY_ENDING is a different axis.",
        "classification": "FEEL_GOOD"
      },
      "exec_elapsed_s": 0.503,
      "result": {
        "total_scored": 36720,
        "top": [
          {
            "movie_id": 524288,
            "score": 1.0
          },
          {
            "movie_id": 131074,
            "score": 1.0
          },
          {
            "movie_id": 3,
            "score": 1.0
          },
          {
            "movie_id": 131079,
            "score": 1.0
          },
          {
            "movie_id": 8,
            "score": 1.0
          },
          {
            "movie_id": 655369,
            "score": 1.0
          },
          {
            "movie_id": 11,
            "score": 1.0
          },
          {
            "movie_id": 12,
            "score": 1.0
          },
          {
            "movie_id": 13,
            "score": 1.0
          },
          {
            "movie_id": 1310735,
            "score": 1.0
          }
        ]
      }
    }
  ]
}
```

### CE 3 — Kind of story / thematic archetype
_atomic_rewrite_: with actual narrative substance

#### CE 3 payload
```json
{
  "captured_meaning": "The user wants a story with thematic depth or narrative weight.",
  "category_name": "Kind of story / thematic archetype",
  "atomic_rewrite": "with actual narrative substance",
  "llm": {
    "elapsed_s": 4.373,
    "output": {
      "requirement_aspects": [
        {
          "aspect_description": "Capture 'actual substance' as a request for thematic depth or narrative weight alongside the light tone.",
          "endpoint_coverage": "Keyword does not have a clean registry member for 'substance' as an abstract depth/quality ask; forcing a thematic tag would be arbitrary. Semantic can carry the abstract depth signal in plot_analysis.thematic_concepts / character_arcs, but only as a soft descriptive layer rather than a binary story-type gate.",
          "best_endpoint": "semantic",
          "best_endpoint_gaps": "This is not a distinct story archetype; it is a qualitative depth preference. The light / feel-good / runtime / streaming parts belong to other handlers."
        }
      ],
      "performance_vs_bias_analysis": "This is not a binary kind-of-story ask like coming-of-age or survival. 'Actual substance' is a vague depth qualifier, so the spectrum escape applies and bypasses Keyword even though the category is upstream-biased toward it. Semantic is the better fit because it can represent the graded notion of narrative weight without forcing a closed-vocabulary tag.",
      "endpoint_to_run": "semantic",
      "endpoint_parameters": {
        "match_mode": "trait",
        "parameters": {
          "qualifier_inventory": "light tone; feel-good mood; underlying narrative substance / thematic depth; short runtime and streaming availability are outside this atom.",
          "space_queries": [
            {
              "carries_qualifiers": "The query asks for thematic depth / narrative weight beneath a light, feel-good surface.",
              "space": "plot_analysis",
              "weight": "supporting",
              "content": {
                "elevator_pitch": "a light, feel-good story with some thematic depth and narrative weight",
                "plot_overview": null,
                "genre_signatures": [],
                "conflict_type": [],
                "thematic_concepts": [
                  "thematic depth",
                  "narrative weight",
                  "quality storytelling"
                ],
                "character_arcs": []
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
        "qualifier_inventory": "light tone; feel-good mood; underlying narrative substance / thematic depth; short runtime and streaming availability are outside this atom.",
        "space_queries": [
          {
            "carries_qualifiers": "The query asks for thematic depth / narrative weight beneath a light, feel-good surface.",
            "space": "plot_analysis",
            "weight": "supporting",
            "content": {
              "elevator_pitch": "a light, feel-good story with some thematic depth and narrative weight",
              "plot_overview": null,
              "genre_signatures": [],
              "conflict_type": [],
              "thematic_concepts": [
                "thematic depth",
                "narrative weight",
                "quality storytelling"
              ],
              "character_arcs": []
            }
          }
        ],
        "primary_vector": "plot_analysis"
      },
      "exec_elapsed_s": 0.424,
      "result": {
        "total_scored": 2000,
        "top": [
          {
            "movie_id": 79000,
            "score": 0.6373024
          },
          {
            "movie_id": 46889,
            "score": 0.6362839
          },
          {
            "movie_id": 47218,
            "score": 0.6329994
          },
          {
            "movie_id": 4171,
            "score": 0.6327286
          },
          {
            "movie_id": 134168,
            "score": 0.6288681
          },
          {
            "movie_id": 490725,
            "score": 0.62392807
          },
          {
            "movie_id": 7301,
            "score": 0.62364197
          },
          {
            "movie_id": 102916,
            "score": 0.6230202
          },
          {
            "movie_id": 53216,
            "score": 0.6227608
          },
          {
            "movie_id": 601635,
            "score": 0.6222038
          }
        ]
      }
    }
  ]
}
```

### CE 4 — Narrative devices + structural form + craft
_atomic_rewrite_: with substantive storytelling

#### CE 4 payload
```json
{
  "captured_meaning": "The request implies a level of storytelling craft or intellectual engagement.",
  "category_name": "Narrative devices + structural form + craft",
  "atomic_rewrite": "with substantive storytelling",
  "llm": {
    "elapsed_s": 5.453,
    "output": {
      "requirement_aspects": [
        {
          "aspect_description": "Identify the desire for actual substance — light but still thematically or intellectually substantial storytelling.",
          "endpoint_coverage": "Keyword has no registry member for 'substance' or general writing depth. Semantic can capture this through viewer_experience (cognitive_complexity / tone balance) and possibly plot_analysis (thematic_concepts), but the phrasing is broad and not a specific device. Semantic is the better fit for the quality-of-storytelling aspect; Keyword has no clean tag.",
          "best_endpoint": "semantic",
          "best_endpoint_gaps": "The request is still vague: it asks for depth/substance without naming a specific craft device, theme, or narrative mechanic."
        }
      ],
      "performance_vs_bias_analysis": "Semantic wins on its own merits because the ask is a tonal/craft blend, not a canonical structural tag. The tier-1 bias toward Keyword does not apply cleanly here since no registry member names 'substance'; using Keyword would misclassify the requirement. However, this atom is still only a partial fit overall because runtime and streaming availability belong to other categories.",
      "endpoint_to_run": "semantic",
      "endpoint_parameters": {
        "match_mode": "trait",
        "parameters": {
          "qualifier_inventory": "light tone; feel-good mood; substantive storytelling / thematic depth",
          "space_queries": [
            {
              "carries_qualifiers": "Captures the 'actual substance' part as narrative depth and thoughtful writing, not a specific device.",
              "space": "plot_analysis",
              "weight": "supporting",
              "content": {
                "elevator_pitch": "Light, feel-good storytelling with thematic depth and smart writing.",
                "plot_overview": "A movie that feels easygoing and uplifting, but still has meaningful ideas or emotional depth.",
                "genre_signatures": [],
                "conflict_type": [],
                "thematic_concepts": [
                  "thematic depth",
                  "emotional resonance",
                  "thoughtful writing"
                ],
                "character_arcs": []
              }
            },
            {
              "carries_qualifiers": "Captures the 'light' and 'feel-good' watching experience as a positive, breezy tone.",
              "space": "viewer_experience",
              "weight": "central",
              "content": {
                "emotional_palette": {
                  "terms": [
                    "feel-good",
                    "uplifting",
                    "warm"
                  ],
                  "negations": [
                    "dark",
                    "heavy",
                    "bleak"
                  ]
                },
                "tension_adrenaline": {
                  "terms": [
                    "light",
                    "easygoing"
                  ],
                  "negations": [
                    "intense",
                    "tense"
                  ]
                },
                "tone_self_seriousness": {
                  "terms": [
                    "playful",
                    "not self-serious"
                  ],
                  "negations": [
                    "grim",
                    "overly serious"
                  ]
                },
                "cognitive_complexity": {
                  "terms": [
                    "some substance"
                  ],
                  "negations": [
                    "thin",
                    "empty"
                  ]
                },
                "disturbance_profile": {
                  "terms": [],
                  "negations": []
                },
                "sensory_load": {
                  "terms": [
                    "breezy"
                  ],
                  "negations": []
                },
                "emotional_volatility": {
                  "terms": [
                    "gentle"
                  ],
                  "negations": [
                    "melodramatic"
                  ]
                },
                "ending_aftertaste": {
                  "terms": [
                    "leaves you smiling"
                  ],
                  "negations": []
                }
              }
            }
          ],
          "primary_vector": "viewer_experience"
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
        "qualifier_inventory": "light tone; feel-good mood; substantive storytelling / thematic depth",
        "space_queries": [
          {
            "carries_qualifiers": "Captures the 'actual substance' part as narrative depth and thoughtful writing, not a specific device.",
            "space": "plot_analysis",
            "weight": "supporting",
            "content": {
              "elevator_pitch": "Light, feel-good storytelling with thematic depth and smart writing.",
              "plot_overview": "A movie that feels easygoing and uplifting, but still has meaningful ideas or emotional depth.",
              "genre_signatures": [],
              "conflict_type": [],
              "thematic_concepts": [
                "thematic depth",
                "emotional resonance",
                "thoughtful writing"
              ],
              "character_arcs": []
            }
          },
          {
            "carries_qualifiers": "Captures the 'light' and 'feel-good' watching experience as a positive, breezy tone.",
            "space": "viewer_experience",
            "weight": "central",
            "content": {
              "emotional_palette": {
                "terms": [
                  "feel-good",
                  "uplifting",
                  "warm"
                ],
                "negations": [
                  "dark",
                  "heavy",
                  "bleak"
                ]
              },
              "tension_adrenaline": {
                "terms": [
                  "light",
                  "easygoing"
                ],
                "negations": [
                  "intense",
                  "tense"
                ]
              },
              "tone_self_seriousness": {
                "terms": [
                  "playful",
                  "not self-serious"
                ],
                "negations": [
                  "grim",
                  "overly serious"
                ]
              },
              "cognitive_complexity": {
                "terms": [
                  "some substance"
                ],
                "negations": [
                  "thin",
                  "empty"
                ]
              },
              "disturbance_profile": {
                "terms": [],
                "negations": []
              },
              "sensory_load": {
                "terms": [
                  "breezy"
                ],
                "negations": []
              },
              "emotional_volatility": {
                "terms": [
                  "gentle"
                ],
                "negations": [
                  "melodramatic"
                ]
              },
              "ending_aftertaste": {
                "terms": [
                  "leaves you smiling"
                ],
                "negations": []
              }
            }
          }
        ],
        "primary_vector": "viewer_experience"
      },
      "exec_elapsed_s": 1.306,
      "result": {
        "total_scored": 3861,
        "top": [
          {
            "movie_id": 80830,
            "score": 0.8031501799999999
          },
          {
            "movie_id": 1114775,
            "score": 0.8021405333333332
          },
          {
            "movie_id": 1467974,
            "score": 0.8019536199999999
          },
          {
            "movie_id": 710731,
            "score": 0.80131148
          },
          {
            "movie_id": 381755,
            "score": 0.8007889
          },
          {
            "movie_id": 79897,
            "score": 0.7996737066666667
          },
          {
            "movie_id": 53216,
            "score": 0.7994498999999999
          },
          {
            "movie_id": 27862,
            "score": 0.79926428
          },
          {
            "movie_id": 338177,
            "score": 0.7988866166666666
          },
          {
            "movie_id": 136799,
            "score": 0.7987645333333333
          }
        ]
      }
    }
  ]
}
```

### CE 5 — Structured metadata
_atomic_rewrite_: with a runtime around 90 minutes

#### CE 5 payload
```json
{
  "captured_meaning": "The user is filtering for movies with a specific duration of roughly one and a half hours.",
  "category_name": "Structured metadata",
  "atomic_rewrite": "with a runtime around 90 minutes",
  "llm": {
    "elapsed_s": 3.544,
    "output": {
      "requirement_aspects": [
        {
          "aspect_description": "The movie should run around 90 minutes.",
          "relation_to_endpoint": "The runtime column supports a numeric minutes predicate. 'Around 90 minutes' is a rough ceiling/target and can be expressed as a tight runtime window around 90 minutes.",
          "coverage_gaps": null
        }
      ],
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "filter",
        "parameters": {
          "constraint_phrases": [
            "around 90 minutes"
          ],
          "target_attribute": "runtime",
          "value_intent_label": "roughly 90-minute runtime",
          "release_date": null,
          "runtime": {
            "first_value": 90.0,
            "match_operation": "between",
            "second_value": 90.0
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
  },
  "endpoints": [
    {
      "endpoint": "metadata",
      "match_mode": "filter",
      "polarity": "positive",
      "wrapper_type": "MetadataEndpointParameters",
      "parameters": {
        "constraint_phrases": [
          "around 90 minutes"
        ],
        "target_attribute": "runtime",
        "value_intent_label": "roughly 90-minute runtime",
        "release_date": null,
        "runtime": {
          "first_value": 90.0,
          "match_operation": "between",
          "second_value": 90.0
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
      "exec_elapsed_s": 0.79,
      "result": {
        "total_scored": 87553,
        "top": [
          {
            "movie_id": 221432,
            "score": 1.0
          },
          {
            "movie_id": 431,
            "score": 1.0
          },
          {
            "movie_id": 221972,
            "score": 1.0
          },
          {
            "movie_id": 906,
            "score": 1.0
          },
          {
            "movie_id": 222721,
            "score": 1.0
          },
          {
            "movie_id": 1442,
            "score": 1.0
          },
          {
            "movie_id": 224790,
            "score": 1.0
          },
          {
            "movie_id": 223448,
            "score": 1.0
          },
          {
            "movie_id": 1980,
            "score": 1.0
          },
          {
            "movie_id": 1984,
            "score": 1.0
          }
        ]
      }
    }
  ]
}
```

### CE 6 — Structured metadata
_atomic_rewrite_: available on a streaming service

#### CE 6 payload
```json
{
  "captured_meaning": "The user is looking for movies available on digital streaming platforms.",
  "category_name": "Structured metadata",
  "atomic_rewrite": "available on a streaming service",
  "llm": {
    "elapsed_s": 4.999,
    "output": null
  },
  "endpoints": [],
  "note": "handler LLM failed after retry"
}
```

## Implicit Expectations

_elapsed: 1.84s_

#### Implicit Expectations Response
```json
{
  "query_intent_summary": "The user is seeking a lighthearted and emotionally positive movie that maintains thematic depth and narrative quality, constrained by a short runtime and digital availability.",
  "explicit_signals": [
    {
      "query_span": "light",
      "normalized_description": "lighthearted and buoyant tone",
      "explicit_axis": "neither"
    },
    {
      "query_span": "feel-good",
      "normalized_description": "positive emotional experience",
      "explicit_axis": "neither"
    },
    {
      "query_span": "actual substance",
      "normalized_description": "thematic depth and storytelling quality",
      "explicit_axis": "quality"
    },
    {
      "query_span": "90 minutes",
      "normalized_description": "runtime of approximately 90 minutes",
      "explicit_axis": "neither"
    },
    {
      "query_span": "streaming service",
      "normalized_description": "available on digital platforms",
      "explicit_axis": "neither"
    }
  ],
  "explicit_ordering_axis_analysis": "No explicit ordering axis such as trending, chronology, or semantic extremeness is present. The request for 'actual substance' acts as a quality filter but does not define a specific ranking sequence.",
  "explicitly_addresses_quality": true,
  "explicitly_addresses_notability": false,
  "should_apply_quality_prior": false,
  "should_apply_notability_prior": true
}
```

## Consolidation Buckets (post fan-out)

#### Consolidation summary
```json
{
  "inclusion_unique_ids": 87553,
  "downrank_unique_ids": 0,
  "exclusion_unique_ids": 0,
  "preference_specs_count": 5,
  "used_fallback": "none"
}
```

## Final Score Breakdowns (top 100)

#### Top 100 ScoreBreakdowns
```json
[
  {
    "movie_id": 869626,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3535082897733333,
    "implicit_prior_contribution": 0.24383242696319157,
    "final_score": 1.5973407167365248
  },
  {
    "movie_id": 59968,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3492645272386667,
    "implicit_prior_contribution": 0.24595555435154554,
    "final_score": 1.5952200815902122
  },
  {
    "movie_id": 16553,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.34878457962133336,
    "implicit_prior_contribution": 0.24287392518538414,
    "final_score": 1.5916585048067176
  },
  {
    "movie_id": 12105,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3471526471,
    "implicit_prior_contribution": 0.24341724099546302,
    "final_score": 1.5905698880954628
  },
  {
    "movie_id": 255,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.34923203886666665,
    "implicit_prior_contribution": 0.24038786722797734,
    "final_score": 1.5896199060946439
  },
  {
    "movie_id": 10555,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.34276376694933336,
    "implicit_prior_contribution": 0.24674357710261013,
    "final_score": 1.5895073440519434
  },
  {
    "movie_id": 808,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.340202938096,
    "implicit_prior_contribution": 0.2471971406846365,
    "final_score": 1.5874000787806366
  },
  {
    "movie_id": 11017,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3407426280813334,
    "implicit_prior_contribution": 0.24657112640440335,
    "final_score": 1.5873137544857365
  },
  {
    "movie_id": 10806,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.34222277506666665,
    "implicit_prior_contribution": 0.24472329446317248,
    "final_score": 1.5869460695298392
  },
  {
    "movie_id": 9778,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.340426884952,
    "implicit_prior_contribution": 0.24636271344894453,
    "final_score": 1.5867895984009446
  },
  {
    "movie_id": 15919,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3427579777626667,
    "implicit_prior_contribution": 0.24347593496043785,
    "final_score": 1.5862339127231044
  },
  {
    "movie_id": 9928,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.33935333558933334,
    "implicit_prior_contribution": 0.24660668987241138,
    "final_score": 1.5859600254617445
  },
  {
    "movie_id": 5966,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3394059855666667,
    "implicit_prior_contribution": 0.24649225277189574,
    "final_score": 1.5858982383385625
  },
  {
    "movie_id": 40205,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.344988732424,
    "implicit_prior_contribution": 0.24086581136593946,
    "final_score": 1.5858545437899394
  },
  {
    "movie_id": 11260,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.34083419006933335,
    "implicit_prior_contribution": 0.24470269221310562,
    "final_score": 1.585536882282439
  },
  {
    "movie_id": 11132,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.341641726468,
    "implicit_prior_contribution": 0.2438216829172316,
    "final_score": 1.5854634093852318
  },
  {
    "movie_id": 26594,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3454313336786667,
    "implicit_prior_contribution": 0.23931557561260666,
    "final_score": 1.5847469092912734
  },
  {
    "movie_id": 242042,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.34131177007200003,
    "implicit_prior_contribution": 0.24331797663559188,
    "final_score": 1.584629746707592
  },
  {
    "movie_id": 13342,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3378870600466667,
    "implicit_prior_contribution": 0.24631617052719149,
    "final_score": 1.584203230573858
  },
  {
    "movie_id": 7552,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.33694173375333336,
    "implicit_prior_contribution": 0.2465506396651249,
    "final_score": 1.5834923734184583
  },
  {
    "movie_id": 587792,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3362313256773333,
    "implicit_prior_contribution": 0.24674622325000162,
    "final_score": 1.5829775489273348
  },
  {
    "movie_id": 13531,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3372575864466667,
    "implicit_prior_contribution": 0.24554768424781132,
    "final_score": 1.582805270694478
  },
  {
    "movie_id": 1268,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.33622078166133335,
    "implicit_prior_contribution": 0.24644205130632071,
    "final_score": 1.5826628329676542
  },
  {
    "movie_id": 214030,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.33945472371066665,
    "implicit_prior_contribution": 0.24293355344525078,
    "final_score": 1.5823882771559175
  },
  {
    "movie_id": 22794,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3354064140746667,
    "implicit_prior_contribution": 0.2469183456087623,
    "final_score": 1.582324759683429
  },
  {
    "movie_id": 16279,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.34160938914666666,
    "implicit_prior_contribution": 0.2396910145629569,
    "final_score": 1.5813004037096237
  },
  {
    "movie_id": 13768,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.338191318556,
    "implicit_prior_contribution": 0.24302441596625404,
    "final_score": 1.581215734522254
  },
  {
    "movie_id": 89185,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3436812869186667,
    "implicit_prior_contribution": 0.2374783724192679,
    "final_score": 1.5811596593379347
  },
  {
    "movie_id": 13401,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3377520300933333,
    "implicit_prior_contribution": 0.24300950687553635,
    "final_score": 1.5807615369688697
  },
  {
    "movie_id": 1139829,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.33903507366400004,
    "implicit_prior_contribution": 0.24129666492077173,
    "final_score": 1.5803317385847717
  },
  {
    "movie_id": 1648,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.33351523490133334,
    "implicit_prior_contribution": 0.2464846483626027,
    "final_score": 1.579999883263936
  },
  {
    "movie_id": 489929,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.33519488898,
    "implicit_prior_contribution": 0.2440507014891891,
    "final_score": 1.579245590469189
  },
  {
    "movie_id": 324852,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.33257187553333334,
    "implicit_prior_contribution": 0.24662776174686676,
    "final_score": 1.5791996372802002
  },
  {
    "movie_id": 10663,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.332145704828,
    "implicit_prior_contribution": 0.24667267526408151,
    "final_score": 1.5788183800920816
  },
  {
    "movie_id": 4729,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.34451800432,
    "implicit_prior_contribution": 0.23410082626513792,
    "final_score": 1.5786188305851379
  },
  {
    "movie_id": 12921,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.33661006137733335,
    "implicit_prior_contribution": 0.24157257432746757,
    "final_score": 1.5781826357048008
  },
  {
    "movie_id": 144789,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.33531055635466667,
    "implicit_prior_contribution": 0.24280518049948732,
    "final_score": 1.5781157368541538
  },
  {
    "movie_id": 18423,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.35323527333733334,
    "implicit_prior_contribution": 0.22473919997792732,
    "final_score": 1.5779744733152608
  },
  {
    "movie_id": 13704,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.33530986832933335,
    "implicit_prior_contribution": 0.2425901302190595,
    "final_score": 1.5778999985483928
  },
  {
    "movie_id": 543540,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3330331674786667,
    "implicit_prior_contribution": 0.24471974795275225,
    "final_score": 1.577752915431419
  },
  {
    "movie_id": 829051,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.33644392197733336,
    "implicit_prior_contribution": 0.24127588427202093,
    "final_score": 1.5777198062493543
  },
  {
    "movie_id": 186161,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3414299606853333,
    "implicit_prior_contribution": 0.23628294006925576,
    "final_score": 1.5777129007545891
  },
  {
    "movie_id": 13477,
    "inclusion_sum": 0.9833333333333334,
    "downrank_sum": 0.0,
    "preference_contribution": 0.34866923558133334,
    "implicit_prior_contribution": 0.24545857535013652,
    "final_score": 1.5774611442648032
  },
  {
    "movie_id": 499,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.33381011899199997,
    "implicit_prior_contribution": 0.24359865227409427,
    "final_score": 1.5774087712660942
  },
  {
    "movie_id": 31547,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.335635094592,
    "implicit_prior_contribution": 0.2415457040961189,
    "final_score": 1.5771807986881188
  },
  {
    "movie_id": 77459,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3344323901146667,
    "implicit_prior_contribution": 0.24255153863668352,
    "final_score": 1.57698392875135
  },
  {
    "movie_id": 763532,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.34639581799200003,
    "implicit_prior_contribution": 0.2300663694927159,
    "final_score": 1.576462187484716
  },
  {
    "movie_id": 467956,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.33176873828933334,
    "implicit_prior_contribution": 0.24439158432212535,
    "final_score": 1.5761603226114587
  },
  {
    "movie_id": 333667,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3506101543933334,
    "implicit_prior_contribution": 0.2254866827265973,
    "final_score": 1.5760968371199306
  },
  {
    "movie_id": 138222,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.340575720492,
    "implicit_prior_contribution": 0.2354709811822308,
    "final_score": 1.576046701674231
  },
  {
    "movie_id": 12182,
    "inclusion_sum": 0.9833333333333334,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3465899124146667,
    "implicit_prior_contribution": 0.2460141246901472,
    "final_score": 1.5759373704381474
  },
  {
    "movie_id": 230179,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3317293494106667,
    "implicit_prior_contribution": 0.24410230332707147,
    "final_score": 1.5758316527377383
  },
  {
    "movie_id": 9502,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.32857649047066667,
    "implicit_prior_contribution": 0.24714337458507274,
    "final_score": 1.5757198650557394
  },
  {
    "movie_id": 43949,
    "inclusion_sum": 0.9833333333333334,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3461554139306667,
    "implicit_prior_contribution": 0.24614078795435262,
    "final_score": 1.5756295352183527
  },
  {
    "movie_id": 1290287,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3351544609746667,
    "implicit_prior_contribution": 0.24029225604635232,
    "final_score": 1.575446717021019
  },
  {
    "movie_id": 826510,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.340086838868,
    "implicit_prior_contribution": 0.23531806845354392,
    "final_score": 1.575404907321544
  },
  {
    "movie_id": 13785,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.329643806772,
    "implicit_prior_contribution": 0.2456562196763244,
    "final_score": 1.5753000264483243
  },
  {
    "movie_id": 9256,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.33272221981600003,
    "implicit_prior_contribution": 0.2425107579554078,
    "final_score": 1.5752329777714078
  },
  {
    "movie_id": 136799,
    "inclusion_sum": 0.9666666666666667,
    "downrank_sum": 0.0,
    "preference_contribution": 0.36246382050666665,
    "implicit_prior_contribution": 0.24604737055217152,
    "final_score": 1.575177857725505
  },
  {
    "movie_id": 12924,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.33494685280933334,
    "implicit_prior_contribution": 0.24022027193222614,
    "final_score": 1.5751671247415595
  },
  {
    "movie_id": 644090,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.333367181336,
    "implicit_prior_contribution": 0.24164160813216976,
    "final_score": 1.5750087894681697
  },
  {
    "movie_id": 25527,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.33469930814133336,
    "implicit_prior_contribution": 0.24020736385587152,
    "final_score": 1.574906671997205
  },
  {
    "movie_id": 446893,
    "inclusion_sum": 0.9833333333333334,
    "downrank_sum": 0.0,
    "preference_contribution": 0.347834827648,
    "implicit_prior_contribution": 0.24373087770816176,
    "final_score": 1.5748990386894952
  },
  {
    "movie_id": 12606,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.33665607028666666,
    "implicit_prior_contribution": 0.23796867337495586,
    "final_score": 1.5746247436616225
  },
  {
    "movie_id": 16806,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.33383824407733337,
    "implicit_prior_contribution": 0.2407686715817681,
    "final_score": 1.5746069156591014
  },
  {
    "movie_id": 13120,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3421697625546667,
    "implicit_prior_contribution": 0.23222961178567492,
    "final_score": 1.5743993743403415
  },
  {
    "movie_id": 1281,
    "inclusion_sum": 0.9833333333333334,
    "downrank_sum": 0.0,
    "preference_contribution": 0.34478635530266666,
    "implicit_prior_contribution": 0.24623610475368785,
    "final_score": 1.574355793389688
  },
  {
    "movie_id": 15397,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3419338676693333,
    "implicit_prior_contribution": 0.23238758128180587,
    "final_score": 1.5743214489511392
  },
  {
    "movie_id": 423949,
    "inclusion_sum": 0.9833333333333334,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3482944279826667,
    "implicit_prior_contribution": 0.24194020801638869,
    "final_score": 1.5735679693323887
  },
  {
    "movie_id": 830721,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3408306290106667,
    "implicit_prior_contribution": 0.23267751431210393,
    "final_score": 1.5735081433227704
  },
  {
    "movie_id": 9907,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3292463761346667,
    "implicit_prior_contribution": 0.2441925309368915,
    "final_score": 1.5734389070715582
  },
  {
    "movie_id": 337,
    "inclusion_sum": 0.9833333333333334,
    "downrank_sum": 0.0,
    "preference_contribution": 0.35245939433600004,
    "implicit_prior_contribution": 0.23745712886272682,
    "final_score": 1.57324985653206
  },
  {
    "movie_id": 287587,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.33939147830000005,
    "implicit_prior_contribution": 0.23352944970189518,
    "final_score": 1.5729209280018952
  },
  {
    "movie_id": 18147,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.33444665865333334,
    "implicit_prior_contribution": 0.23838762480533035,
    "final_score": 1.5728342834586637
  },
  {
    "movie_id": 17483,
    "inclusion_sum": 0.9833333333333334,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3465631103946667,
    "implicit_prior_contribution": 0.2428224267647853,
    "final_score": 1.5727188704927855
  },
  {
    "movie_id": 8141,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.33130371012666665,
    "implicit_prior_contribution": 0.24139529337470825,
    "final_score": 1.572699003501375
  },
  {
    "movie_id": 177699,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.32869999647333337,
    "implicit_prior_contribution": 0.24386865646341724,
    "final_score": 1.5725686529367506
  },
  {
    "movie_id": 12586,
    "inclusion_sum": 0.9833333333333334,
    "downrank_sum": 0.0,
    "preference_contribution": 0.35242685193333334,
    "implicit_prior_contribution": 0.2366571167431812,
    "final_score": 1.572417302009848
  },
  {
    "movie_id": 334878,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.34303945146799997,
    "implicit_prior_contribution": 0.22906454553531092,
    "final_score": 1.5721039970033108
  },
  {
    "movie_id": 83389,
    "inclusion_sum": 0.9833333333333334,
    "downrank_sum": 0.0,
    "preference_contribution": 0.343499068332,
    "implicit_prior_contribution": 0.24522839671836433,
    "final_score": 1.5720607983836978
  },
  {
    "movie_id": 13499,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.32834640339200005,
    "implicit_prior_contribution": 0.24370565809114078,
    "final_score": 1.5720520614831408
  },
  {
    "movie_id": 11141,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.328993562464,
    "implicit_prior_contribution": 0.242872974778253,
    "final_score": 1.571866537242253
  },
  {
    "movie_id": 582596,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.32666585725200004,
    "implicit_prior_contribution": 0.24490590558096423,
    "final_score": 1.5715717628329642
  },
  {
    "movie_id": 14926,
    "inclusion_sum": 0.9666666666666667,
    "downrank_sum": 0.0,
    "preference_contribution": 0.359800571984,
    "implicit_prior_contribution": 0.24500160495055126,
    "final_score": 1.5714688436012179
  },
  {
    "movie_id": 11025,
    "inclusion_sum": 0.9833333333333334,
    "downrank_sum": 0.0,
    "preference_contribution": 0.344965976824,
    "implicit_prior_contribution": 0.24295332167078207,
    "final_score": 1.5712526318281155
  },
  {
    "movie_id": 653756,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.33574134155333335,
    "implicit_prior_contribution": 0.23545030328030891,
    "final_score": 1.5711916448336423
  },
  {
    "movie_id": 10162,
    "inclusion_sum": 0.9833333333333334,
    "downrank_sum": 0.0,
    "preference_contribution": 0.34372384420266666,
    "implicit_prior_contribution": 0.24341636108630463,
    "final_score": 1.5704735386223048
  },
  {
    "movie_id": 7839,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3290665322186667,
    "implicit_prior_contribution": 0.24131164348494497,
    "final_score": 1.5703781757036117
  },
  {
    "movie_id": 59296,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3357105988226667,
    "implicit_prior_contribution": 0.2346477507847791,
    "final_score": 1.5703583496074458
  },
  {
    "movie_id": 70695,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.33301573144933333,
    "implicit_prior_contribution": 0.23732234237345495,
    "final_score": 1.5703380738227883
  },
  {
    "movie_id": 5559,
    "inclusion_sum": 0.9833333333333334,
    "downrank_sum": 0.0,
    "preference_contribution": 0.34030765449333333,
    "implicit_prior_contribution": 0.24669244692244927,
    "final_score": 1.570333434749116
  },
  {
    "movie_id": 25462,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.33910757321466667,
    "implicit_prior_contribution": 0.23096154325273613,
    "final_score": 1.5700691164674028
  },
  {
    "movie_id": 46689,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.335635893292,
    "implicit_prior_contribution": 0.23429792180064501,
    "final_score": 1.5699338150926452
  },
  {
    "movie_id": 5683,
    "inclusion_sum": 0.9833333333333334,
    "downrank_sum": 0.0,
    "preference_contribution": 0.34116285084000003,
    "implicit_prior_contribution": 0.2453583456927504,
    "final_score": 1.5698545298660838
  },
  {
    "movie_id": 76492,
    "inclusion_sum": 0.9833333333333334,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3395436828186667,
    "implicit_prior_contribution": 0.24696108548876292,
    "final_score": 1.569838101640763
  },
  {
    "movie_id": 235,
    "inclusion_sum": 0.9833333333333334,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3390795144426667,
    "implicit_prior_contribution": 0.24710277530150404,
    "final_score": 1.569515623077504
  },
  {
    "movie_id": 10612,
    "inclusion_sum": 0.9833333333333334,
    "downrank_sum": 0.0,
    "preference_contribution": 0.3446631687146667,
    "implicit_prior_contribution": 0.24146233279436724,
    "final_score": 1.5694588348423673
  },
  {
    "movie_id": 12456,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.334900525988,
    "implicit_prior_contribution": 0.23454058781065487,
    "final_score": 1.5694411137986548
  },
  {
    "movie_id": 10634,
    "inclusion_sum": 0.9833333333333334,
    "downrank_sum": 0.0,
    "preference_contribution": 0.33968899663066665,
    "implicit_prior_contribution": 0.2464033279108323,
    "final_score": 1.5694256578748322
  },
  {
    "movie_id": 5991,
    "inclusion_sum": 1.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.32892605797333335,
    "implicit_prior_contribution": 0.24037008869515164,
    "final_score": 1.569296146668485
  }
]
```

## Step 4 — Summary

### Filters
- with a runtime around 90 minutes

### Traits
- with a light tone
- a feel-good movie to cheer me up
- with actual narrative substance
- with substantive storytelling

_used_fallback: none_

### query_intent_summary
The user is seeking a lighthearted and emotionally positive movie that maintains thematic depth and narrative quality, constrained by a short runtime and digital availability.

_implicit priors: quality=off  popularity=on_

### Top 100 Results

| # | final | filter | pref | down | impl | title (year) | tmdb_id |
|---|-------|--------|------|------|------|--------------|---------|
| 1 | 1.5973 | 1.0000 | 0.3535 | -0.0000 | 0.2438 | Marcel the Shell with Shoes On (2022) | 869626 |
| 2 | 1.5952 | 1.0000 | 0.3493 | -0.0000 | 0.2460 | Our Idiot Brother (2011) | 59968 |
| 3 | 1.5917 | 1.0000 | 0.3488 | -0.0000 | 0.2429 | Little Manhattan (2005) | 16553 |
| 4 | 1.5906 | 1.0000 | 0.3472 | -0.0000 | 0.2434 | Yellow Submarine (1968) | 12105 |
| 5 | 1.5896 | 1.0000 | 0.3492 | -0.0000 | 0.2404 | Stolen Kisses (1968) | 255 |
| 6 | 1.5895 | 1.0000 | 0.3428 | -0.0000 | 0.2467 | Shark Tale (2004) | 10555 |
| 7 | 1.5874 | 1.0000 | 0.3402 | -0.0000 | 0.2472 | Shrek (2001) | 808 |
| 8 | 1.5873 | 1.0000 | 0.3407 | -0.0000 | 0.2466 | Billy Madison (1995) | 11017 |
| 9 | 1.5869 | 1.0000 | 0.3422 | -0.0000 | 0.2447 | In & Out (1997) | 10806 |
| 10 | 1.5868 | 1.0000 | 0.3404 | -0.0000 | 0.2464 | Serendipity (2001) | 9778 |
| 11 | 1.5862 | 1.0000 | 0.3428 | -0.0000 | 0.2435 | Marty (1955) | 15919 |
| 12 | 1.5860 | 1.0000 | 0.3394 | -0.0000 | 0.2466 | Robots (2005) | 9928 |
| 13 | 1.5859 | 1.0000 | 0.3394 | -0.0000 | 0.2465 | Along Came Polly (2004) | 5966 |
| 14 | 1.5859 | 1.0000 | 0.3450 | -0.0000 | 0.2409 | 16 Wishes (2010) | 40205 |
| 15 | 1.5855 | 1.0000 | 0.3408 | -0.0000 | 0.2447 | Meet Dave (2008) | 11260 |
| 16 | 1.5855 | 1.0000 | 0.3416 | -0.0000 | 0.2438 | Confessions of a Teenage Drama Queen (2004) | 11132 |
| 17 | 1.5847 | 1.0000 | 0.3454 | -0.0000 | 0.2393 | The Secret (2006) | 26594 |
| 18 | 1.5846 | 1.0000 | 0.3413 | -0.0000 | 0.2433 | Barefoot (2014) | 242042 |
| 19 | 1.5842 | 1.0000 | 0.3379 | -0.0000 | 0.2463 | Fast Times at Ridgemont High (1982) | 13342 |
| 20 | 1.5835 | 1.0000 | 0.3369 | -0.0000 | 0.2466 | Fun with Dick and Jane (2005) | 7552 |
| 21 | 1.5830 | 1.0000 | 0.3362 | -0.0000 | 0.2467 | Palm Springs (2020) | 587792 |
| 22 | 1.5828 | 1.0000 | 0.3373 | -0.0000 | 0.2455 | Empire Records (1995) | 13531 |
| 23 | 1.5827 | 1.0000 | 0.3362 | -0.0000 | 0.2464 | Mr. Bean's Holiday (2007) | 1268 |
| 24 | 1.5824 | 1.0000 | 0.3395 | -0.0000 | 0.2429 | Fading Gigolo (2013) | 214030 |
| 25 | 1.5823 | 1.0000 | 0.3354 | -0.0000 | 0.2469 | Cloudy with a Chance of Meatballs (2009) | 22794 |
| 26 | 1.5813 | 1.0000 | 0.3416 | -0.0000 | 0.2397 | The Great Buck Howard (2008) | 16279 |
| 27 | 1.5812 | 1.0000 | 0.3382 | -0.0000 | 0.2430 | Tuck Everlasting (2002) | 13768 |
| 28 | 1.5812 | 1.0000 | 0.3437 | -0.0000 | 0.2375 | Radio Rebel (2012) | 89185 |
| 29 | 1.5808 | 1.0000 | 0.3378 | -0.0000 | 0.2430 | The Accidental Husband (2008) | 13401 |
| 30 | 1.5803 | 1.0000 | 0.3390 | -0.0000 | 0.2413 | Orion and the Dark (2024) | 1139829 |
| 31 | 1.5800 | 1.0000 | 0.3335 | -0.0000 | 0.2465 | Bill & Ted's Excellent Adventure (1989) | 1648 |
| 32 | 1.5792 | 1.0000 | 0.3352 | -0.0000 | 0.2441 | Destination Wedding (2018) | 489929 |
| 33 | 1.5792 | 1.0000 | 0.3326 | -0.0000 | 0.2466 | Despicable Me 3 (2017) | 324852 |
| 34 | 1.5788 | 1.0000 | 0.3321 | -0.0000 | 0.2467 | The Waterboy (1998) | 10663 |
| 35 | 1.5786 | 1.0000 | 0.3445 | -0.0000 | 0.2341 | The Gendarme Gets Married (1968) | 4729 |
| 36 | 1.5782 | 1.0000 | 0.3366 | -0.0000 | 0.2416 | Strange Brew (1983) | 12921 |
| 37 | 1.5781 | 1.0000 | 0.3353 | -0.0000 | 0.2428 | I'm So Excited! (2013) | 144789 |
| 38 | 1.5780 | 1.0000 | 0.3532 | -0.0000 | 0.2247 | B.A.P.S (1997) | 18423 |
| 39 | 1.5779 | 1.0000 | 0.3353 | -0.0000 | 0.2426 | License to Drive (1988) | 13704 |
| 40 | 1.5778 | 1.0000 | 0.3330 | -0.0000 | 0.2447 | The Perfect Date (2019) | 543540 |
| 41 | 1.5777 | 1.0000 | 0.3364 | -0.0000 | 0.2413 | About My Father (2023) | 829051 |
| 42 | 1.5777 | 1.0000 | 0.3414 | -0.0000 | 0.2363 | The Gilded Cage (2013) | 186161 |
| 43 | 1.5775 | 0.9833 | 0.3487 | -0.0000 | 0.2455 | When in Rome (2010) | 13477 |
| 44 | 1.5774 | 1.0000 | 0.3338 | -0.0000 | 0.2436 | Cléo from 5 to 7 (1962) | 499 |
| 45 | 1.5772 | 1.0000 | 0.3356 | -0.0000 | 0.2415 | Saban, Son of Saban (1977) | 31547 |
| 46 | 1.5770 | 1.0000 | 0.3344 | -0.0000 | 0.2426 | A Monster in Paris (2011) | 77459 |
| 47 | 1.5765 | 1.0000 | 0.3464 | -0.0000 | 0.2301 | Long Story Short (2021) | 763532 |
| 48 | 1.5762 | 1.0000 | 0.3318 | -0.0000 | 0.2444 | The Professor (2018) | 467956 |
| 49 | 1.5761 | 1.0000 | 0.3506 | -0.0000 | 0.2255 | Rock Dog (2016) | 333667 |
| 50 | 1.5760 | 1.0000 | 0.3406 | -0.0000 | 0.2355 | Best Man Down (2012) | 138222 |
| 51 | 1.5759 | 0.9833 | 0.3466 | -0.0000 | 0.2460 | Nick and Norah's Infinite Playlist (2008) | 12182 |
| 52 | 1.5758 | 1.0000 | 0.3317 | -0.0000 | 0.2441 | Big Game (2014) | 230179 |
| 53 | 1.5757 | 1.0000 | 0.3286 | -0.0000 | 0.2471 | Kung Fu Panda (2008) | 9502 |
| 54 | 1.5756 | 0.9833 | 0.3462 | -0.0000 | 0.2461 | Flipped (2010) | 43949 |
| 55 | 1.5754 | 1.0000 | 0.3352 | -0.0000 | 0.2403 | Hot Frosty (2024) | 1290287 |
| 56 | 1.5754 | 1.0000 | 0.3401 | -0.0000 | 0.2353 | Harold and the Purple Crayon (2024) | 826510 |
| 57 | 1.5753 | 1.0000 | 0.3296 | -0.0000 | 0.2457 | Best in Show (2000) | 13785 |
| 58 | 1.5752 | 1.0000 | 0.3327 | -0.0000 | 0.2425 | Kopps (2003) | 9256 |
| 59 | 1.5752 | 0.9667 | 0.3625 | -0.0000 | 0.2460 | Trolls (2016) | 136799 |
| 60 | 1.5752 | 1.0000 | 0.3349 | -0.0000 | 0.2402 | The Place Promised in Our Early Days (2004) | 12924 |
| 61 | 1.5750 | 1.0000 | 0.3334 | -0.0000 | 0.2416 | Love, Guaranteed (2020) | 644090 |
| 62 | 1.5749 | 1.0000 | 0.3347 | -0.0000 | 0.2402 | The Ron Clark Story (2006) | 25527 |
| 63 | 1.5749 | 0.9833 | 0.3478 | -0.0000 | 0.2437 | Trolls World Tour (2020) | 446893 |
| 64 | 1.5746 | 1.0000 | 0.3367 | -0.0000 | 0.2380 | A Very Brady Sequel (1996) | 12606 |
| 65 | 1.5746 | 1.0000 | 0.3338 | -0.0000 | 0.2408 | Johnny Dangerously (1984) | 16806 |
| 66 | 1.5744 | 1.0000 | 0.3422 | -0.0000 | 0.2322 | In Search of a Midnight Kiss (2007) | 13120 |
| 67 | 1.5744 | 0.9833 | 0.3448 | -0.0000 | 0.2462 | Bean (1997) | 1281 |
| 68 | 1.5743 | 1.0000 | 0.3419 | -0.0000 | 0.2324 | I Am David (2003) | 15397 |
| 69 | 1.5736 | 0.9833 | 0.3483 | -0.0000 | 0.2419 | Unicorn Store (2017) | 423949 |
| 70 | 1.5735 | 1.0000 | 0.3408 | -0.0000 | 0.2327 | The Miracle Club (2023) | 830721 |
| 71 | 1.5734 | 1.0000 | 0.3292 | -0.0000 | 0.2442 | Barnyard (2006) | 9907 |
| 72 | 1.5732 | 0.9833 | 0.3525 | -0.0000 | 0.2375 | Monsieur Ibrahim (2003) | 337 |
| 73 | 1.5729 | 1.0000 | 0.3394 | -0.0000 | 0.2335 | Adult Beginners (2014) | 287587 |
| 74 | 1.5728 | 1.0000 | 0.3344 | -0.0000 | 0.2384 | Unaccompanied Minors (2006) | 18147 |
| 75 | 1.5727 | 0.9833 | 0.3466 | -0.0000 | 0.2428 | Shelter (2007) | 17483 |
| 76 | 1.5727 | 1.0000 | 0.3313 | -0.0000 | 0.2414 | You Kill Me (2007) | 8141 |
| 77 | 1.5726 | 1.0000 | 0.3287 | -0.0000 | 0.2439 | Alan Partridge: Alpha Papa (2013) | 177699 |
| 78 | 1.5724 | 0.9833 | 0.3524 | -0.0000 | 0.2367 | Bella (2006) | 12586 |
| 79 | 1.5721 | 1.0000 | 0.3430 | -0.0000 | 0.2291 | Blind Date (2015) | 334878 |
| 80 | 1.5721 | 0.9833 | 0.3435 | -0.0000 | 0.2452 | From Up on Poppy Hill (2011) | 83389 |
| 81 | 1.5721 | 1.0000 | 0.3283 | -0.0000 | 0.2437 | Yours, Mine & Ours (2005) | 13499 |
| 82 | 1.5719 | 1.0000 | 0.3290 | -0.0000 | 0.2429 | Laws of Attraction (2004) | 11141 |
| 83 | 1.5716 | 1.0000 | 0.3267 | -0.0000 | 0.2449 | The Wrong Missy (2020) | 582596 |
| 84 | 1.5715 | 0.9667 | 0.3598 | -0.0000 | 0.2450 | Uptown Girls (2003) | 14926 |
| 85 | 1.5713 | 0.9833 | 0.3450 | -0.0000 | 0.2430 | New York Minute (2004) | 11025 |
| 86 | 1.5712 | 1.0000 | 0.3357 | -0.0000 | 0.2355 | The Mole Agent (2020) | 653756 |
| 87 | 1.5705 | 0.9833 | 0.3437 | -0.0000 | 0.2434 | Waking Ned (1998) | 10162 |
| 88 | 1.5704 | 1.0000 | 0.3291 | -0.0000 | 0.2413 | Wedding Daze (2006) | 7839 |
| 89 | 1.5704 | 1.0000 | 0.3357 | -0.0000 | 0.2346 | Love, Wedding, Marriage (2011) | 59296 |
| 90 | 1.5703 | 1.0000 | 0.3330 | -0.0000 | 0.2373 | Dirty Girl (2010) | 70695 |
| 91 | 1.5703 | 0.9833 | 0.3403 | -0.0000 | 0.2467 | Bee Movie (2007) | 5559 |
| 92 | 1.5701 | 1.0000 | 0.3391 | -0.0000 | 0.2310 | Two Can Play That Game (2001) | 25462 |
| 93 | 1.5699 | 1.0000 | 0.3356 | -0.0000 | 0.2343 | Waste Land (2010) | 46689 |
| 94 | 1.5699 | 0.9833 | 0.3412 | -0.0000 | 0.2454 | Pee-wee's Big Adventure (1985) | 5683 |
| 95 | 1.5698 | 0.9833 | 0.3395 | -0.0000 | 0.2470 | Hotel Transylvania (2012) | 76492 |
| 96 | 1.5695 | 0.9833 | 0.3391 | -0.0000 | 0.2471 | Stand by Me (1986) | 235 |
| 97 | 1.5695 | 0.9833 | 0.3447 | -0.0000 | 0.2415 | The Englishman Who Went Up a Hill But Came Down a Mountain (1995) | 10612 |
| 98 | 1.5694 | 1.0000 | 0.3349 | -0.0000 | 0.2345 | The Five Obstructions (2003) | 12456 |
| 99 | 1.5694 | 0.9833 | 0.3397 | -0.0000 | 0.2464 | Friday (1995) | 10634 |
| 100 | 1.5693 | 1.0000 | 0.3289 | -0.0000 | 0.2404 | The Last Laugh (1924) | 5991 |

_total elapsed: 20.14s_
