# Query

> What's that movie where the guy can't form new memories and has tattoos all over his body to help him solve his wife's murder?

## Step 2 — Query Pre-pass

_elapsed: 4.56s_

#### Step 2 Response
```json
{
  "overall_query_intention_exploration": "The user is attempting to identify a specific movie based on a detailed plot description involving a protagonist with a unique medical condition and a specific visual trait. The query focuses on narrative mechanics (memory loss), character appearance (tattoos), and the central plot motivation (solving a murder).",
  "requirements": [
    {
      "query_text": "guy can't form new memories",
      "description": "Describes the protagonist's specific cognitive condition which serves as a central narrative device.",
      "modifiers": [],
      "coverage_evidence": [
        {
          "captured_meaning": "The protagonist suffers from a specific type of amnesia or memory impairment that prevents the creation of new memories.",
          "category_name": "Narrative devices + structural form + craft",
          "fit_quality": "partial",
          "atomic_rewrite": "protagonist who cannot form new memories"
        },
        {
          "captured_meaning": "The condition of memory loss is a central thematic element of the story's progression.",
          "category_name": "Kind of story / thematic archetype",
          "fit_quality": "partial",
          "atomic_rewrite": "story centered on memory loss and amnesia"
        }
      ]
    },
    {
      "query_text": "tattoos all over his body",
      "description": "Describes a distinctive physical attribute of the main character used as a tool for the plot.",
      "modifiers": [],
      "coverage_evidence": [
        {
          "captured_meaning": "The character has extensive tattoos covering his body.",
          "category_name": "Specific subject / element / motif",
          "fit_quality": "clean",
          "atomic_rewrite": "character with tattoos all over his body"
        }
      ]
    },
    {
      "query_text": "solve his wife's murder",
      "description": "Identifies the primary narrative goal and inciting incident of the film.",
      "modifiers": [
        {
          "original_text": "to help him",
          "effect": "binds the tattoos as a functional tool for achieving the goal",
          "type": "role_marker"
        }
      ],
      "coverage_evidence": [
        {
          "captured_meaning": "The central plot involves an investigation into the killing of the protagonist's spouse.",
          "category_name": "Plot events + narrative setting",
          "fit_quality": "clean",
          "atomic_rewrite": "solving the murder of the protagonist's wife"
        },
        {
          "captured_meaning": "The story follows a revenge or justice-seeking trajectory typical of a specific story pattern.",
          "category_name": "Sub-genre + story archetype",
          "fit_quality": "clean",
          "atomic_rewrite": "revenge-driven murder mystery"
        }
      ]
    }
  ]
}
```

## Step 3 — Endpoint Translation + Execution

### CE 1 — Narrative devices + structural form + craft
_atomic_rewrite_: protagonist who cannot form new memories

#### CE 1 payload
```json
{
  "captured_meaning": "The protagonist suffers from a specific type of amnesia or memory impairment that prevents the creation of new memories.",
  "category_name": "Narrative devices + structural form + craft",
  "atomic_rewrite": "protagonist who cannot form new memories",
  "llm": {
    "elapsed_s": 7.902,
    "output": {
      "requirement_aspects": [
        {
          "aspect_description": "The protagonist cannot form new memories.",
          "endpoint_coverage": "Keyword can cover this only if the query maps to a canonical tag like UNRELIABLE_NARRATOR or NONLINEAR_TIMELINE, but neither names memory loss. Semantic can capture it in narrative_techniques as an information-control / character-condition device, with plot_events also reflecting the concrete setup and plot goal.",
          "best_endpoint": "semantic",
          "best_endpoint_gaps": "No canonical registry member directly names anterograde amnesia, so Keyword is not a clean fit."
        },
        {
          "aspect_description": "The tattoos function as an on-body memory aid / clue system.",
          "endpoint_coverage": "Keyword has no tag for this specific gimmick. Semantic can capture it in narrative_techniques.additional_narrative_devices and information_control as a plot device and memory-tracking method.",
          "best_endpoint": "semantic",
          "best_endpoint_gaps": "The exact visual gimmick is not canonicalized."
        },
        {
          "aspect_description": "He uses the condition and tattoos to investigate his wife's murder.",
          "endpoint_coverage": "Keyword could loosely suggest REVENGE or MYSTERY, but those miss the specific investigation setup. Semantic can represent the concrete plot premise in plot_events and plot_analysis.",
          "best_endpoint": "semantic",
          "best_endpoint_gaps": "Keyword lacks a direct label for this combined premise."
        }
      ],
      "performance_vs_bias_analysis": "Semantic wins on its own merits. The bias would prefer Keyword only if a registry member clearly named the device, but the query is a specific memory-loss-plus-tattoo investigation premise, and no canonical tag cleanly covers that. The right fit is to use Semantic with multiple spaces rather than force a weak tag.",
      "endpoint_to_run": "semantic",
      "endpoint_parameters": {
        "match_mode": "filter",
        "parameters": {
          "qualifier_inventory": "anterograde amnesia / cannot form new memories; tattoos used as a memory aid; murder investigation for his wife",
          "space_queries": [
            {
              "carries_qualifiers": "Captures the concrete plot setup: a man with no new-memory formation who uses tattoos to pursue his wife's murder.",
              "space": "plot_events",
              "weight": "central",
              "content": {
                "plot_summary": "A man who cannot form new memories tracks clues on his body with tattoos while investigating the murder of his wife."
              }
            },
            {
              "carries_qualifiers": "Captures the story shape: memory impairment, investigation, murder motive, and a revenge/mystery-driven arc.",
              "space": "plot_analysis",
              "weight": "central",
              "content": {
                "elevator_pitch": "A mystery thriller about an amnesiac investigator using tattoos to reconstruct events and solve his wife's murder.",
                "plot_overview": "A protagonist with severe memory impairment relies on tattoos and notes to pursue a murder investigation tied to his wife.",
                "genre_signatures": [
                  "mystery thriller"
                ],
                "conflict_type": [
                  "investigation",
                  "man vs self"
                ],
                "thematic_concepts": [
                  "memory loss",
                  "identity",
                  "revenge"
                ],
                "character_arcs": [
                  "framed around fragmented recollection"
                ]
              }
            },
            {
              "carries_qualifiers": "Captures the memory condition and the tattoo-based clue system as storytelling mechanics.",
              "space": "narrative_techniques",
              "weight": "central",
              "content": {
                "narrative_archetype": {
                  "terms": [
                    "fragmented investigation"
                  ]
                },
                "narrative_delivery": {
                  "terms": [
                    "clue-based progression"
                  ]
                },
                "pov_perspective": {
                  "terms": [
                    "limited memory perspective"
                  ]
                },
                "characterization_methods": {
                  "terms": [
                    "memory notes",
                    "externalized memory aid"
                  ]
                },
                "character_arcs": {
                  "terms": []
                },
                "audience_character_perception": {
                  "terms": [
                    "piecing together clues"
                  ]
                },
                "information_control": {
                  "terms": [
                    "restricted information",
                    "memory gaps"
                  ]
                },
                "conflict_stakes_design": {
                  "terms": [
                    "personal mystery",
                    "wife's murder"
                  ]
                },
                "additional_narrative_devices": {
                  "terms": [
                    "tattoo memory system",
                    "anterograde amnesia"
                  ]
                }
              }
            }
          ],
          "primary_vector": "narrative_techniques"
        },
        "polarity": "positive"
      }
    }
  },
  "endpoints": [
    {
      "endpoint": "semantic",
      "match_mode": "filter",
      "polarity": "positive",
      "wrapper_type": "SemanticEndpointParameters",
      "parameters": {
        "qualifier_inventory": "anterograde amnesia / cannot form new memories; tattoos used as a memory aid; murder investigation for his wife",
        "space_queries": [
          {
            "carries_qualifiers": "Captures the concrete plot setup: a man with no new-memory formation who uses tattoos to pursue his wife's murder.",
            "space": "plot_events",
            "weight": "central",
            "content": {
              "plot_summary": "A man who cannot form new memories tracks clues on his body with tattoos while investigating the murder of his wife."
            }
          },
          {
            "carries_qualifiers": "Captures the story shape: memory impairment, investigation, murder motive, and a revenge/mystery-driven arc.",
            "space": "plot_analysis",
            "weight": "central",
            "content": {
              "elevator_pitch": "A mystery thriller about an amnesiac investigator using tattoos to reconstruct events and solve his wife's murder.",
              "plot_overview": "A protagonist with severe memory impairment relies on tattoos and notes to pursue a murder investigation tied to his wife.",
              "genre_signatures": [
                "mystery thriller"
              ],
              "conflict_type": [
                "investigation",
                "man vs self"
              ],
              "thematic_concepts": [
                "memory loss",
                "identity",
                "revenge"
              ],
              "character_arcs": [
                "framed around fragmented recollection"
              ]
            }
          },
          {
            "carries_qualifiers": "Captures the memory condition and the tattoo-based clue system as storytelling mechanics.",
            "space": "narrative_techniques",
            "weight": "central",
            "content": {
              "narrative_archetype": {
                "terms": [
                  "fragmented investigation"
                ]
              },
              "narrative_delivery": {
                "terms": [
                  "clue-based progression"
                ]
              },
              "pov_perspective": {
                "terms": [
                  "limited memory perspective"
                ]
              },
              "characterization_methods": {
                "terms": [
                  "memory notes",
                  "externalized memory aid"
                ]
              },
              "character_arcs": {
                "terms": []
              },
              "audience_character_perception": {
                "terms": [
                  "piecing together clues"
                ]
              },
              "information_control": {
                "terms": [
                  "restricted information",
                  "memory gaps"
                ]
              },
              "conflict_stakes_design": {
                "terms": [
                  "personal mystery",
                  "wife's murder"
                ]
              },
              "additional_narrative_devices": {
                "terms": [
                  "tattoo memory system",
                  "anterograde amnesia"
                ]
              }
            }
          }
        ],
        "primary_vector": "narrative_techniques"
      },
      "exec_elapsed_s": 0.66,
      "result": {
        "total_scored": 284,
        "top": [
          {
            "movie_id": 449306,
            "score": 1.0
          },
          {
            "movie_id": 660632,
            "score": 1.0
          },
          {
            "movie_id": 227700,
            "score": 1.0
          },
          {
            "movie_id": 207681,
            "score": 1.0
          },
          {
            "movie_id": 93142,
            "score": 1.0
          },
          {
            "movie_id": 744746,
            "score": 1.0
          },
          {
            "movie_id": 190945,
            "score": 1.0
          },
          {
            "movie_id": 464684,
            "score": 1.0
          },
          {
            "movie_id": 189297,
            "score": 1.0
          },
          {
            "movie_id": 1424,
            "score": 1.0
          }
        ]
      }
    }
  ]
}
```

### CE 2 — Kind of story / thematic archetype
_atomic_rewrite_: story centered on memory loss and amnesia

#### CE 2 payload
```json
{
  "captured_meaning": "The condition of memory loss is a central thematic element of the story's progression.",
  "category_name": "Kind of story / thematic archetype",
  "atomic_rewrite": "story centered on memory loss and amnesia",
  "llm": {
    "elapsed_s": 6.301,
    "output": {
      "requirement_aspects": [
        {
          "aspect_description": "Capture the protagonist's inability to form new memories as part of the story's setup.",
          "endpoint_coverage": "Keyword can only help if there were a registry member for this exact story shape; none is available here. Semantic can cover it cleanly in plot_analysis, likely via narrative mechanics like amnesia / memory loss and possibly information control. The tattoos and murder goal belong to other categories, not this one.",
          "best_endpoint": "semantic",
          "best_endpoint_gaps": "This atom is only a partial slice of the full query; it does not capture the tattoos or murder-solving premise."
        }
      ],
      "performance_vs_bias_analysis": "This is not a binary thematic-archetype ask that Keyword can answer. The registry has no direct memory-loss story member, and the phrasing is a plot-device identification, so Semantic wins on merit despite the tier-1 bias. The bias does not apply because there is no authoritative keyword hit.",
      "endpoint_to_run": "semantic",
      "endpoint_parameters": {
        "match_mode": "filter",
        "parameters": {
          "qualifier_inventory": "memory loss / inability to form new memories; investigative murder-solving premise; tattoos used as memory aids.",
          "space_queries": [
            {
              "carries_qualifiers": "The query's memory-loss device and the protagonist relying on it as part of the story mechanics.",
              "space": "plot_analysis",
              "weight": "central",
              "content": {
                "elevator_pitch": "a man with severe memory loss uses notes and body tattoos to reconstruct events and solve a murder",
                "plot_overview": null,
                "genre_signatures": [],
                "conflict_type": [],
                "thematic_concepts": [
                  "amnesia",
                  "memory loss",
                  "identity",
                  "investigation"
                ],
                "character_arcs": []
              }
            },
            {
              "carries_qualifiers": "The tattoos are a distinctive storytelling device that help preserve information across memory gaps.",
              "space": "narrative_techniques",
              "weight": "supporting",
              "content": {
                "narrative_archetype": {
                  "terms": []
                },
                "narrative_delivery": {
                  "terms": [
                    "nonlinear information recovery"
                  ]
                },
                "pov_perspective": {
                  "terms": []
                },
                "characterization_methods": {
                  "terms": [
                    "external memory aids"
                  ]
                },
                "character_arcs": {
                  "terms": []
                },
                "audience_character_perception": {
                  "terms": [
                    "piecemeal revelation"
                  ]
                },
                "information_control": {
                  "terms": [
                    "restricted information",
                    "fragmented memory"
                  ]
                },
                "conflict_stakes_design": {
                  "terms": [
                    "investigative stakes"
                  ]
                },
                "additional_narrative_devices": {
                  "terms": [
                    "tattoos as memory cues"
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
      "match_mode": "filter",
      "polarity": "positive",
      "wrapper_type": "SemanticEndpointParameters",
      "parameters": {
        "qualifier_inventory": "memory loss / inability to form new memories; investigative murder-solving premise; tattoos used as memory aids.",
        "space_queries": [
          {
            "carries_qualifiers": "The query's memory-loss device and the protagonist relying on it as part of the story mechanics.",
            "space": "plot_analysis",
            "weight": "central",
            "content": {
              "elevator_pitch": "a man with severe memory loss uses notes and body tattoos to reconstruct events and solve a murder",
              "plot_overview": null,
              "genre_signatures": [],
              "conflict_type": [],
              "thematic_concepts": [
                "amnesia",
                "memory loss",
                "identity",
                "investigation"
              ],
              "character_arcs": []
            }
          },
          {
            "carries_qualifiers": "The tattoos are a distinctive storytelling device that help preserve information across memory gaps.",
            "space": "narrative_techniques",
            "weight": "supporting",
            "content": {
              "narrative_archetype": {
                "terms": []
              },
              "narrative_delivery": {
                "terms": [
                  "nonlinear information recovery"
                ]
              },
              "pov_perspective": {
                "terms": []
              },
              "characterization_methods": {
                "terms": [
                  "external memory aids"
                ]
              },
              "character_arcs": {
                "terms": []
              },
              "audience_character_perception": {
                "terms": [
                  "piecemeal revelation"
                ]
              },
              "information_control": {
                "terms": [
                  "restricted information",
                  "fragmented memory"
                ]
              },
              "conflict_stakes_design": {
                "terms": [
                  "investigative stakes"
                ]
              },
              "additional_narrative_devices": {
                "terms": [
                  "tattoos as memory cues"
                ]
              }
            }
          }
        ],
        "primary_vector": "plot_analysis"
      },
      "exec_elapsed_s": 0.809,
      "result": {
        "total_scored": 392,
        "top": [
          {
            "movie_id": 77,
            "score": 1.0
          },
          {
            "movie_id": 14070,
            "score": 1.0
          },
          {
            "movie_id": 252296,
            "score": 1.0
          },
          {
            "movie_id": 45258,
            "score": 1.0
          },
          {
            "movie_id": 37195,
            "score": 1.0
          },
          {
            "movie_id": 618019,
            "score": 1.0
          },
          {
            "movie_id": 42061,
            "score": 1.0
          },
          {
            "movie_id": 185248,
            "score": 1.0
          },
          {
            "movie_id": 69636,
            "score": 1.0
          },
          {
            "movie_id": 42825,
            "score": 1.0
          }
        ]
      }
    }
  ]
}
```

### CE 3 — Specific subject / element / motif
_atomic_rewrite_: character with tattoos all over his body

#### CE 3 payload
```json
{
  "captured_meaning": "The character has extensive tattoos covering his body.",
  "category_name": "Specific subject / element / motif",
  "atomic_rewrite": "character with tattoos all over his body",
  "llm": {
    "elapsed_s": 4.076,
    "output": {
      "requirement_aspects": [
        {
          "aspect_description": "Find a movie where the protagonist has tattoos all over his body.",
          "endpoint_coverage": "Keyword has no canonical registry member for 'tattoos all over his body' as a motif. Semantic plot_events can capture the concrete visual detail in a plot summary, but it would also need the rest of the plot to narrow correctly.",
          "best_endpoint": "semantic",
          "best_endpoint_gaps": "Semantic can represent the tattoo detail, but this atom is part of a larger identification query; by itself it does not uniquely identify the film."
        }
      ],
      "performance_vs_bias_analysis": "The registry does not cover tattoos as a standalone subject/motif here, so Keyword has no direct member to fire. Semantic plot_events is the better fit because the distinguishing clue is a concrete story detail about the character’s body and how it functions in the plot. The keyword-first bias does not overcome the missing concept tag.",
      "endpoint_to_run": "semantic",
      "endpoint_parameters": {
        "match_mode": "filter",
        "parameters": {
          "qualifier_inventory": "Distinctive body tattoos; tattoos used as a plot-aiding visual clue; protagonist identification via appearance.",
          "space_queries": [
            {
              "carries_qualifiers": "The tattoos are a concrete visual detail that appears in the story and help identify the character.",
              "space": "plot_events",
              "weight": "central",
              "content": {
                "plot_summary": "a man with tattoos covering his body, using them as clues while trying to solve his wife's murder"
              }
            }
          ],
          "primary_vector": "plot_events"
        },
        "polarity": "positive"
      }
    }
  },
  "endpoints": [
    {
      "endpoint": "semantic",
      "match_mode": "filter",
      "polarity": "positive",
      "wrapper_type": "SemanticEndpointParameters",
      "parameters": {
        "qualifier_inventory": "Distinctive body tattoos; tattoos used as a plot-aiding visual clue; protagonist identification via appearance.",
        "space_queries": [
          {
            "carries_qualifiers": "The tattoos are a concrete visual detail that appears in the story and help identify the character.",
            "space": "plot_events",
            "weight": "central",
            "content": {
              "plot_summary": "a man with tattoos covering his body, using them as clues while trying to solve his wife's murder"
            }
          }
        ],
        "primary_vector": "plot_events"
      },
      "exec_elapsed_s": 0.382,
      "result": {
        "total_scored": 295,
        "top": [
          {
            "movie_id": 109188,
            "score": 1.0
          },
          {
            "movie_id": 9908,
            "score": 1.0
          },
          {
            "movie_id": 44101,
            "score": 1.0
          },
          {
            "movie_id": 182423,
            "score": 1.0
          },
          {
            "movie_id": 61094,
            "score": 1.0
          },
          {
            "movie_id": 351868,
            "score": 1.0
          },
          {
            "movie_id": 59165,
            "score": 1.0
          },
          {
            "movie_id": 105910,
            "score": 1.0
          },
          {
            "movie_id": 108134,
            "score": 1.0
          },
          {
            "movie_id": 56599,
            "score": 1.0
          }
        ]
      }
    }
  ]
}
```

### CE 4 — Plot events + narrative setting
_atomic_rewrite_: solving the murder of the protagonist's wife

#### CE 4 payload
```json
{
  "captured_meaning": "The central plot involves an investigation into the killing of the protagonist's spouse.",
  "category_name": "Plot events + narrative setting",
  "atomic_rewrite": "solving the murder of the protagonist's wife",
  "llm": {
    "elapsed_s": 3.3,
    "output": {
      "requirement_aspects": [
        {
          "aspect_description": "Find the movie where the protagonist is trying to solve his wife's murder.",
          "relation_to_endpoint": "plot_events captures concrete story events and character goals in synopsis form. A plot_summary about a man investigating and trying to solve his wife's murder matches that ingest-side narrative prose directly.",
          "coverage_gaps": null
        }
      ],
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "filter",
        "parameters": {
          "qualifier_inventory": "One plot atom: a man investigates and tries to solve his wife's murder. Sibling context adds amnesia and body tattoos as supporting plot devices.",
          "space_queries": [
            {
              "carries_qualifiers": "This space captures the concrete investigation plot: the protagonist is pursuing the murder of his wife.",
              "space": "plot_events",
              "weight": "central",
              "content": {
                "plot_summary": "a man investigates and tries to solve the murder of his wife, using clues tied to her killing as he tries to uncover what happened."
              }
            }
          ],
          "primary_vector": "plot_events"
        },
        "polarity": "positive"
      }
    }
  },
  "endpoints": [
    {
      "endpoint": "semantic",
      "match_mode": "filter",
      "polarity": "positive",
      "wrapper_type": "SemanticEndpointParameters",
      "parameters": {
        "qualifier_inventory": "One plot atom: a man investigates and tries to solve his wife's murder. Sibling context adds amnesia and body tattoos as supporting plot devices.",
        "space_queries": [
          {
            "carries_qualifiers": "This space captures the concrete investigation plot: the protagonist is pursuing the murder of his wife.",
            "space": "plot_events",
            "weight": "central",
            "content": {
              "plot_summary": "a man investigates and tries to solve the murder of his wife, using clues tied to her killing as he tries to uncover what happened."
            }
          }
        ],
        "primary_vector": "plot_events"
      },
      "exec_elapsed_s": 0.763,
      "result": {
        "total_scored": 401,
        "top": [
          {
            "movie_id": 400995,
            "score": 1.0
          },
          {
            "movie_id": 109188,
            "score": 1.0
          },
          {
            "movie_id": 465934,
            "score": 1.0
          },
          {
            "movie_id": 148044,
            "score": 1.0
          },
          {
            "movie_id": 443849,
            "score": 1.0
          },
          {
            "movie_id": 160254,
            "score": 1.0
          },
          {
            "movie_id": 1243975,
            "score": 1.0
          },
          {
            "movie_id": 56599,
            "score": 1.0
          },
          {
            "movie_id": 1428294,
            "score": 1.0
          },
          {
            "movie_id": 1431043,
            "score": 1.0
          }
        ]
      }
    }
  ]
}
```

### CE 5 — Sub-genre + story archetype
_atomic_rewrite_: revenge-driven murder mystery

#### CE 5 payload
```json
{
  "captured_meaning": "The story follows a revenge or justice-seeking trajectory typical of a specific story pattern.",
  "category_name": "Sub-genre + story archetype",
  "atomic_rewrite": "revenge-driven murder mystery",
  "llm": {
    "elapsed_s": 6.986,
    "output": {
      "requirement_aspects": [
        {
          "aspect_description": "Identify the revenge/justice-seeking story pattern centered on solving the wife's murder.",
          "endpoint_coverage": "Keyword can cover this directly with REVENGE, which names the plot engine of pursuing justice after a personal wrong. Semantic could also cover it via plot_analysis.conflict_type and plot_events, but the requirement is a recognizable story pattern label and a registry member exists for it.",
          "best_endpoint": "keyword",
          "best_endpoint_gaps": "The registry tag captures the revenge axis, but not the specific murder-investigation flavor or memory-loss/tattoo gimmick."
        },
        {
          "aspect_description": "Identify the murder-solving investigative setup used as the story's main premise.",
          "endpoint_coverage": "Keyword has no dedicated 'murder investigation' story-pattern tag here; REVENGE is the closest fit but broadens past the investigative premise. Semantic can express the concrete setup in plot_events and the revenge/justice shape in plot_analysis more faithfully.",
          "best_endpoint": "semantic",
          "best_endpoint_gaps": "It still won't name the exact film-noir-style premise as a single tag; it represents the pattern descriptively instead."
        }
      ],
      "performance_vs_bias_analysis": "This is a close call, but Semantic wins on its own merits because the query is not just 'revenge' — it also depends on a concrete investigative premise plus the memory-loss/tattoo mechanism. A single Keyword tag would collapse that into REVENGE and lose the distinctive setup. The tier-1 bias is not enough to force a broader, less faithful match.",
      "endpoint_to_run": "semantic",
      "endpoint_parameters": {
        "match_mode": "filter",
        "parameters": {
          "qualifier_inventory": "revenge/justice-seeking plot; wife's murder as the central case; anterograde amnesia; tattoos used as memory aids and clues.",
          "space_queries": [
            {
              "carries_qualifiers": "plot_analysis carries the revenge/justice-seeking conflict type and the murder-investigation story shape.",
              "space": "plot_analysis",
              "weight": "central",
              "content": {
                "elevator_pitch": "A man with anterograde amnesia pursues justice for his wife's murder using tattoos and notes to track clues.",
                "plot_overview": null,
                "genre_signatures": [
                  "revenge thriller",
                  "mystery thriller"
                ],
                "conflict_type": [
                  "revenge",
                  "murder investigation"
                ],
                "thematic_concepts": [
                  "justice",
                  "memory",
                  "identity"
                ],
                "character_arcs": [
                  "obsessive pursuit"
                ]
              }
            },
            {
              "carries_qualifiers": "plot_events carries the concrete setup: he cannot form new memories, relies on tattoos, and investigates his wife's murder.",
              "space": "plot_events",
              "weight": "central",
              "content": {
                "plot_summary": "A man who cannot form new memories uses tattoos, notes, and fragments of evidence to investigate his wife's murder and pursue the people responsible."
              }
            },
            {
              "carries_qualifiers": "narrative_techniques carries the memory-limit device and clue-based information control.",
              "space": "narrative_techniques",
              "weight": "supporting",
              "content": {
                "narrative_archetype": {
                  "terms": [
                    "mystery investigation"
                  ]
                },
                "narrative_delivery": {
                  "terms": [
                    "fragmented presentation"
                  ]
                },
                "pov_perspective": {
                  "terms": [
                    "limited perspective"
                  ]
                },
                "characterization_methods": {
                  "terms": [
                    "external memory aids"
                  ]
                },
                "character_arcs": {
                  "terms": []
                },
                "audience_character_perception": {
                  "terms": [
                    "disoriented protagonist"
                  ]
                },
                "information_control": {
                  "terms": [
                    "memory gaps",
                    "reconstructed clues"
                  ]
                },
                "conflict_stakes_design": {
                  "terms": [
                    "personal vendetta"
                  ]
                },
                "additional_narrative_devices": {
                  "terms": [
                    "tattoos as memory aids"
                  ]
                }
              }
            }
          ],
          "primary_vector": "plot_events"
        },
        "polarity": "positive"
      }
    }
  },
  "endpoints": [
    {
      "endpoint": "semantic",
      "match_mode": "filter",
      "polarity": "positive",
      "wrapper_type": "SemanticEndpointParameters",
      "parameters": {
        "qualifier_inventory": "revenge/justice-seeking plot; wife's murder as the central case; anterograde amnesia; tattoos used as memory aids and clues.",
        "space_queries": [
          {
            "carries_qualifiers": "plot_analysis carries the revenge/justice-seeking conflict type and the murder-investigation story shape.",
            "space": "plot_analysis",
            "weight": "central",
            "content": {
              "elevator_pitch": "A man with anterograde amnesia pursues justice for his wife's murder using tattoos and notes to track clues.",
              "plot_overview": null,
              "genre_signatures": [
                "revenge thriller",
                "mystery thriller"
              ],
              "conflict_type": [
                "revenge",
                "murder investigation"
              ],
              "thematic_concepts": [
                "justice",
                "memory",
                "identity"
              ],
              "character_arcs": [
                "obsessive pursuit"
              ]
            }
          },
          {
            "carries_qualifiers": "plot_events carries the concrete setup: he cannot form new memories, relies on tattoos, and investigates his wife's murder.",
            "space": "plot_events",
            "weight": "central",
            "content": {
              "plot_summary": "A man who cannot form new memories uses tattoos, notes, and fragments of evidence to investigate his wife's murder and pursue the people responsible."
            }
          },
          {
            "carries_qualifiers": "narrative_techniques carries the memory-limit device and clue-based information control.",
            "space": "narrative_techniques",
            "weight": "supporting",
            "content": {
              "narrative_archetype": {
                "terms": [
                  "mystery investigation"
                ]
              },
              "narrative_delivery": {
                "terms": [
                  "fragmented presentation"
                ]
              },
              "pov_perspective": {
                "terms": [
                  "limited perspective"
                ]
              },
              "characterization_methods": {
                "terms": [
                  "external memory aids"
                ]
              },
              "character_arcs": {
                "terms": []
              },
              "audience_character_perception": {
                "terms": [
                  "disoriented protagonist"
                ]
              },
              "information_control": {
                "terms": [
                  "memory gaps",
                  "reconstructed clues"
                ]
              },
              "conflict_stakes_design": {
                "terms": [
                  "personal vendetta"
                ]
              },
              "additional_narrative_devices": {
                "terms": [
                  "tattoos as memory aids"
                ]
              }
            }
          }
        ],
        "primary_vector": "plot_events"
      },
      "exec_elapsed_s": 0.899,
      "result": {
        "total_scored": 300,
        "top": [
          {
            "movie_id": 614715,
            "score": 1.0
          },
          {
            "movie_id": 588297,
            "score": 1.0
          },
          {
            "movie_id": 2071,
            "score": 1.0
          },
          {
            "movie_id": 69636,
            "score": 1.0
          },
          {
            "movie_id": 66366,
            "score": 1.0
          },
          {
            "movie_id": 42061,
            "score": 1.0
          },
          {
            "movie_id": 139667,
            "score": 1.0
          },
          {
            "movie_id": 77,
            "score": 1.0
          },
          {
            "movie_id": 552865,
            "score": 1.0
          },
          {
            "movie_id": 252296,
            "score": 1.0
          }
        ]
      }
    }
  ]
}
```

## Implicit Expectations

_elapsed: 2.45s_

#### Implicit Expectations Response
```json
{
  "query_intent_summary": "The user is trying to identify a specific movie based on a detailed plot description involving a protagonist with anterograde amnesia and tattoos used as clues to solve a murder.",
  "explicit_signals": [
    {
      "query_span": "guy can't form new memories",
      "normalized_description": "protagonist with anterograde amnesia",
      "explicit_axis": "neither"
    },
    {
      "query_span": "tattoos all over his body",
      "normalized_description": "character with extensive body tattoos",
      "explicit_axis": "neither"
    },
    {
      "query_span": "solve his wife's murder",
      "normalized_description": "plot involving investigating a spouse's murder",
      "explicit_axis": "neither"
    }
  ],
  "explicit_ordering_axis_analysis": "The query is a specific item retrieval (TIP) request based on plot details; it does not contain any explicit ordering or ranking instructions.",
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
  "inclusion_unique_ids": 1180,
  "downrank_unique_ids": 0,
  "exclusion_unique_ids": 0,
  "preference_specs_count": 0,
  "used_fallback": "none"
}
```

## Final Score Breakdowns (top 100)

#### Top 100 ScoreBreakdowns
```json
[
  {
    "movie_id": 325666,
    "inclusion_sum": 5.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.03345010975428238,
    "final_score": 5.033450109754282
  },
  {
    "movie_id": 109188,
    "inclusion_sum": 5.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.025798514431948392,
    "final_score": 5.025798514431949
  },
  {
    "movie_id": 2071,
    "inclusion_sum": 4.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.21966535481689287,
    "final_score": 4.219665354816893
  },
  {
    "movie_id": 978592,
    "inclusion_sum": 4.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2125585800426072,
    "final_score": 4.212558580042607
  },
  {
    "movie_id": 42061,
    "inclusion_sum": 4.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.03884535034581984,
    "final_score": 4.03884535034582
  },
  {
    "movie_id": 37195,
    "inclusion_sum": 4.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.03277637631404958,
    "final_score": 4.032776376314049
  },
  {
    "movie_id": 238156,
    "inclusion_sum": 4.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.027730037635126877,
    "final_score": 4.027730037635127
  },
  {
    "movie_id": 122263,
    "inclusion_sum": 4.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.025045013984237682,
    "final_score": 4.025045013984237
  },
  {
    "movie_id": 28662,
    "inclusion_sum": 4.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.02392031949779126,
    "final_score": 4.023920319497791
  },
  {
    "movie_id": 854467,
    "inclusion_sum": 4.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.022858323213716766,
    "final_score": 4.0228583232137165
  },
  {
    "movie_id": 252296,
    "inclusion_sum": 4.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.02006658703288625,
    "final_score": 4.020066587032886
  },
  {
    "movie_id": 59165,
    "inclusion_sum": 3.948644970075499,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0657573585555602,
    "final_score": 4.014402328631059
  },
  {
    "movie_id": 296928,
    "inclusion_sum": 3.9050708942762533,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.009176136609316609,
    "final_score": 3.91424703088557
  },
  {
    "movie_id": 400252,
    "inclusion_sum": 3.7935489887558242,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.01790257090868722,
    "final_score": 3.8114515596645115
  },
  {
    "movie_id": 41726,
    "inclusion_sum": 3.6969687144353234,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.07344034297419433,
    "final_score": 3.770409057409518
  },
  {
    "movie_id": 575620,
    "inclusion_sum": 3.600195107309017,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.006739626050107302,
    "final_score": 3.6069347333591244
  },
  {
    "movie_id": 77,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.24229244501560593,
    "final_score": 3.242292445015606
  },
  {
    "movie_id": 69636,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2307525017685759,
    "final_score": 3.230752501768576
  },
  {
    "movie_id": 204922,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.21322753211785567,
    "final_score": 3.213227532117856
  },
  {
    "movie_id": 37696,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.21193999403064193,
    "final_score": 3.211939994030642
  },
  {
    "movie_id": 227700,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.21173795314912525,
    "final_score": 3.2117379531491252
  },
  {
    "movie_id": 215830,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2068200487322294,
    "final_score": 3.2068200487322294
  },
  {
    "movie_id": 26836,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2061132367674693,
    "final_score": 3.2061132367674694
  },
  {
    "movie_id": 2045,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2050935804489727,
    "final_score": 3.2050935804489726
  },
  {
    "movie_id": 340584,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.20330679595324672,
    "final_score": 3.203306795953247
  },
  {
    "movie_id": 1388350,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.19529339408568636,
    "final_score": 3.1952933940856862
  },
  {
    "movie_id": 70009,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.19391213871047597,
    "final_score": 3.193912138710476
  },
  {
    "movie_id": 21946,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.19332171788052704,
    "final_score": 3.193321717880527
  },
  {
    "movie_id": 455957,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.19271524146372798,
    "final_score": 3.192715241463728
  },
  {
    "movie_id": 412000,
    "inclusion_sum": 2.9690663512032343,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.2217936465213994,
    "final_score": 3.1908599977246337
  },
  {
    "movie_id": 18587,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.19076724030436906,
    "final_score": 3.190767240304369
  },
  {
    "movie_id": 46164,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.18984998857300944,
    "final_score": 3.1898499885730094
  },
  {
    "movie_id": 332177,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.17732808753673268,
    "final_score": 3.1773280875367327
  },
  {
    "movie_id": 20357,
    "inclusion_sum": 2.9872616798851808,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.16923172593435507,
    "final_score": 3.156493405819536
  },
  {
    "movie_id": 1243975,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.15249894777828782,
    "final_score": 3.152498947778288
  },
  {
    "movie_id": 449756,
    "inclusion_sum": 2.9487505488671863,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.20168189261961725,
    "final_score": 3.1504324414868035
  },
  {
    "movie_id": 38017,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.14635149805381914,
    "final_score": 3.146351498053819
  },
  {
    "movie_id": 633374,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.11819996420778578,
    "final_score": 3.118199964207786
  },
  {
    "movie_id": 13730,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.11466565332292693,
    "final_score": 3.114665653322927
  },
  {
    "movie_id": 1424,
    "inclusion_sum": 2.940321058826195,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.16217202069195247,
    "final_score": 3.1024930795181476
  },
  {
    "movie_id": 108134,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.08617864060817876,
    "final_score": 3.086178640608179
  },
  {
    "movie_id": 399256,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.07697790470707849,
    "final_score": 3.0769779047070784
  },
  {
    "movie_id": 38273,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.07325427215853667,
    "final_score": 3.0732542721585365
  },
  {
    "movie_id": 56329,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.07246941216025499,
    "final_score": 3.072469412160255
  },
  {
    "movie_id": 534928,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.07111290662696566,
    "final_score": 3.071112906626966
  },
  {
    "movie_id": 449306,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.06603357348779354,
    "final_score": 3.0660335734877937
  },
  {
    "movie_id": 21220,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0651948326327477,
    "final_score": 3.0651948326327476
  },
  {
    "movie_id": 443849,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.04532331671853071,
    "final_score": 3.0453233167185307
  },
  {
    "movie_id": 105910,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.04169635006492175,
    "final_score": 3.041696350064922
  },
  {
    "movie_id": 148044,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.03664089464312227,
    "final_score": 3.0366408946431225
  },
  {
    "movie_id": 56599,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.034076998867910234,
    "final_score": 3.0340769988679104
  },
  {
    "movie_id": 40872,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.03403317704750916,
    "final_score": 3.034033177047509
  },
  {
    "movie_id": 151883,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.033341540040091516,
    "final_score": 3.0333415400400914
  },
  {
    "movie_id": 252865,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0318637643149192,
    "final_score": 3.0318637643149193
  },
  {
    "movie_id": 80520,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.03092675882989397,
    "final_score": 3.030926758829894
  },
  {
    "movie_id": 217439,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.027050231445582044,
    "final_score": 3.027050231445582
  },
  {
    "movie_id": 46401,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.026301717080816633,
    "final_score": 3.0263017170808166
  },
  {
    "movie_id": 505879,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.026018076668162336,
    "final_score": 3.026018076668162
  },
  {
    "movie_id": 410590,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.025277874033877403,
    "final_score": 3.0252778740338773
  },
  {
    "movie_id": 43537,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.025123279252062678,
    "final_score": 3.0251232792520626
  },
  {
    "movie_id": 61094,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.024197175291065142,
    "final_score": 3.0241971752910652
  },
  {
    "movie_id": 1560613,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.02418215071790968,
    "final_score": 3.02418215071791
  },
  {
    "movie_id": 83659,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.023549190834463916,
    "final_score": 3.023549190834464
  },
  {
    "movie_id": 588297,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.023372071530912482,
    "final_score": 3.0233720715309125
  },
  {
    "movie_id": 259928,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.02294703520138216,
    "final_score": 3.022947035201382
  },
  {
    "movie_id": 85204,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.022734767817750336,
    "final_score": 3.0227347678177505
  },
  {
    "movie_id": 72820,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.02264349698080153,
    "final_score": 3.0226434969808014
  },
  {
    "movie_id": 325759,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.022528139280867398,
    "final_score": 3.0225281392808676
  },
  {
    "movie_id": 682545,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.020856473618032726,
    "final_score": 3.0208564736180326
  },
  {
    "movie_id": 400995,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.020044156175419027,
    "final_score": 3.020044156175419
  },
  {
    "movie_id": 541806,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.01920608059860164,
    "final_score": 3.0192060805986016
  },
  {
    "movie_id": 107364,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.018470818273998053,
    "final_score": 3.018470818273998
  },
  {
    "movie_id": 182423,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.018373368793157218,
    "final_score": 3.0183733687931573
  },
  {
    "movie_id": 528348,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.018364867999950702,
    "final_score": 3.0183648679999506
  },
  {
    "movie_id": 464684,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.017915115966173577,
    "final_score": 3.0179151159661735
  },
  {
    "movie_id": 208182,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.01766240611541691,
    "final_score": 3.017662406115417
  },
  {
    "movie_id": 290772,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.01639515317070373,
    "final_score": 3.016395153170704
  },
  {
    "movie_id": 315959,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.01588971725402762,
    "final_score": 3.0158897172540278
  },
  {
    "movie_id": 86699,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.015275974204148439,
    "final_score": 3.0152759742041484
  },
  {
    "movie_id": 78180,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.015037539594609982,
    "final_score": 3.01503753959461
  },
  {
    "movie_id": 110548,
    "inclusion_sum": 2.998456635010394,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.01586795289183471,
    "final_score": 3.014324587902229
  },
  {
    "movie_id": 108173,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.013391567089503485,
    "final_score": 3.0133915670895033
  },
  {
    "movie_id": 310451,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.01250820670675818,
    "final_score": 3.0125082067067583
  },
  {
    "movie_id": 490017,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.011044127018063728,
    "final_score": 3.0110441270180637
  },
  {
    "movie_id": 95206,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.011037558208409416,
    "final_score": 3.0110375582084092
  },
  {
    "movie_id": 597710,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.010503544455417102,
    "final_score": 3.010503544455417
  },
  {
    "movie_id": 1379587,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.009818959317363684,
    "final_score": 3.009818959317364
  },
  {
    "movie_id": 34100,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.009101709111755183,
    "final_score": 3.0091017091117553
  },
  {
    "movie_id": 1273783,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.006869217248851355,
    "final_score": 3.0068692172488514
  },
  {
    "movie_id": 1147322,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0054275822967740864,
    "final_score": 3.005427582296774
  },
  {
    "movie_id": 478992,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.004258933360456693,
    "final_score": 3.0042589333604566
  },
  {
    "movie_id": 1012075,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0018121057303564902,
    "final_score": 3.0018121057303566
  },
  {
    "movie_id": 1012148,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0017384738702681834,
    "final_score": 3.0017384738702684
  },
  {
    "movie_id": 565243,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0016934309457355737,
    "final_score": 3.0016934309457355
  },
  {
    "movie_id": 614715,
    "inclusion_sum": 3.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 6.202649781768995e-05,
    "final_score": 3.0000620264978175
  },
  {
    "movie_id": 416862,
    "inclusion_sum": 2.978045957259166,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.012559077133155943,
    "final_score": 2.990605034392322
  },
  {
    "movie_id": 341,
    "inclusion_sum": 2.8584712955837253,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.1251177528221404,
    "final_score": 2.983589048405866
  },
  {
    "movie_id": 27434,
    "inclusion_sum": 2.873473113065117,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.017329770224927855,
    "final_score": 2.890802883290045
  },
  {
    "movie_id": 322556,
    "inclusion_sum": 2.874048226524583,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.012506101815151074,
    "final_score": 2.886554328339734
  },
  {
    "movie_id": 1428294,
    "inclusion_sum": 2.8432284861080963,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0299874147922995,
    "final_score": 2.8732159009003957
  }
]
```

## Step 4 — Summary

### Filters
- protagonist who cannot form new memories
- story centered on memory loss and amnesia
- character with tattoos all over his body
- solving the murder of the protagonist's wife
- revenge-driven murder mystery

### Traits
- (none)

_used_fallback: none_

### query_intent_summary
The user is trying to identify a specific movie based on a detailed plot description involving a protagonist with anterograde amnesia and tattoos used as clues to solve a murder.

_implicit priors: quality=on  popularity=on_

### Top 100 Results

| # | final | filter | pref | down | impl | title (year) | tmdb_id |
|---|-------|--------|------|------|------|--------------|---------|
| 1 | 5.0335 | 5.0000 | 0.0000 | -0.0000 | 0.0335 | Run a Crooked Mile (1969) | 325666 |
| 2 | 5.0258 | 5.0000 | 0.0000 | -0.0000 | 0.0258 | An Occasional Hell (1996) | 109188 |
| 3 | 4.2197 | 4.0000 | 0.0000 | -0.0000 | 0.2197 | Shattered (1991) | 2071 |
| 4 | 4.2126 | 4.0000 | 0.0000 | -0.0000 | 0.2126 | Sleeping Dogs (2024) | 978592 |
| 5 | 4.0388 | 4.0000 | 0.0000 | -0.0000 | 0.0388 | Street of Chance (1942) | 42061 |
| 6 | 4.0328 | 4.0000 | 0.0000 | -0.0000 | 0.0328 | The Long Wait (1954) | 37195 |
| 7 | 4.0277 | 4.0000 | 0.0000 | -0.0000 | 0.0277 | The Silence (2006) | 238156 |
| 8 | 4.0250 | 4.0000 | 0.0000 | -0.0000 | 0.0250 | Black Widow (1951) | 122263 |
| 9 | 4.0239 | 4.0000 | 0.0000 | -0.0000 | 0.0239 | A Woman's Devotion (1956) | 28662 |
| 10 | 4.0229 | 4.0000 | 0.0000 | -0.0000 | 0.0229 | Indemnity (2022) | 854467 |
| 11 | 4.0201 | 4.0000 | 0.0000 | -0.0000 | 0.0201 | Murder by Night (1989) | 252296 |
| 12 | 4.0144 | 3.9486 | 0.0000 | -0.0000 | 0.0658 | The Tattooed Stranger (1950) | 59165 |
| 13 | 3.9142 | 3.9051 | 0.0000 | -0.0000 | 0.0092 | Blind (2004) | 296928 |
| 14 | 3.8115 | 3.7935 | 0.0000 | -0.0000 | 0.0179 | Evidence of Truth (2016) | 400252 |
| 15 | 3.7704 | 3.6970 | 0.0000 | -0.0000 | 0.0734 | Chasing Ghosts (2005) | 41726 |
| 16 | 3.6069 | 3.6002 | 0.0000 | -0.0000 | 0.0067 | Dark World (2008) | 575620 |
| 17 | 3.2423 | 3.0000 | 0.0000 | -0.0000 | 0.2423 | Memento (2000) | 77 |
| 18 | 3.2308 | 3.0000 | 0.0000 | -0.0000 | 0.2308 | Ghajini (2005) | 69636 |
| 19 | 3.2132 | 3.0000 | 0.0000 | -0.0000 | 0.2132 | Before I Go to Sleep (2014) | 204922 |
| 20 | 3.2119 | 3.0000 | 0.0000 | -0.0000 | 0.2119 | Fear X (2003) | 37696 |
| 21 | 3.2117 | 3.0000 | 0.0000 | -0.0000 | 0.2117 | Anna (2013) | 227700 |
| 22 | 3.2068 | 3.0000 | 0.0000 | -0.0000 | 0.2068 | Open Grave (2013) | 215830 |
| 23 | 3.2061 | 3.0000 | 0.0000 | -0.0000 | 0.2061 | The Fourth Man (2007) | 26836 |
| 24 | 3.2051 | 3.0000 | 0.0000 | -0.0000 | 0.2051 | Unforgettable (1996) | 2045 |
| 25 | 3.2033 | 3.0000 | 0.0000 | -0.0000 | 0.2033 | Lavender (2016) | 340584 |
| 26 | 3.1953 | 3.0000 | 0.0000 | -0.0000 | 0.1953 | Tatsama Tadbhava (2023) | 1388350 |
| 27 | 3.1939 | 3.0000 | 0.0000 | -0.0000 | 0.1939 | The River Murders (2011) | 70009 |
| 28 | 3.1933 | 3.0000 | 0.0000 | -0.0000 | 0.1933 | Counter Investigation (2007) | 21946 |
| 29 | 3.1927 | 3.0000 | 0.0000 | -0.0000 | 0.1927 | Domino (2019) | 455957 |
| 30 | 3.1909 | 2.9691 | 0.0000 | -0.0000 | 0.2218 | Small Town Crime (2018) | 412000 |
| 31 | 3.1908 | 3.0000 | 0.0000 | -0.0000 | 0.1908 | 10 to Midnight (1983) | 18587 |
| 32 | 3.1898 | 3.0000 | 0.0000 | -0.0000 | 0.1898 | Retribution (2007) | 46164 |
| 33 | 3.1773 | 3.0000 | 0.0000 | -0.0000 | 0.1773 | Vendetta (2015) | 332177 |
| 34 | 3.1565 | 2.9873 | 0.0000 | -0.0000 | 0.1692 | Trauma (2004) | 20357 |
| 35 | 3.1525 | 3.0000 | 0.0000 | -0.0000 | 0.1525 | Anand Sreebala (2024) | 1243975 |
| 36 | 3.1504 | 2.9488 | 0.0000 | -0.0000 | 0.2017 | The Postcard Killings (2020) | 449756 |
| 37 | 3.1464 | 3.0000 | 0.0000 | -0.0000 | 0.1464 | Letters from a Killer (1998) | 38017 |
| 38 | 3.1182 | 3.0000 | 0.0000 | -0.0000 | 0.1182 | Remember (2022) | 633374 |
| 39 | 3.1147 | 3.0000 | 0.0000 | -0.0000 | 0.1147 | Black Box (2005) | 13730 |
| 40 | 3.1025 | 2.9403 | 0.0000 | -0.0000 | 0.1622 | November (2004) | 1424 |
| 41 | 3.0862 | 3.0000 | 0.0000 | -0.0000 | 0.0862 | Sketch Artist (1992) | 108134 |
| 42 | 3.0770 | 3.0000 | 0.0000 | -0.0000 | 0.0770 | The Ghoul (2017) | 399256 |
| 43 | 3.0733 | 3.0000 | 0.0000 | -0.0000 | 0.0733 | Calling Dr. Death (1943) | 38273 |
| 44 | 3.0725 | 3.0000 | 0.0000 | -0.0000 | 0.0725 | The Detective (2007) | 56329 |
| 45 | 3.0711 | 3.0000 | 0.0000 | -0.0000 | 0.0711 | Darkness Falls (2020) | 534928 |
| 46 | 3.0660 | 3.0000 | 0.0000 | -0.0000 | 0.0660 | Battle of Memories (2017) | 449306 |
| 47 | 3.0652 | 3.0000 | 0.0000 | -0.0000 | 0.0652 | Frank McKlusky, C.I. (2002) | 21220 |
| 48 | 3.0453 | 3.0000 | 0.0000 | -0.0000 | 0.0453 | Winter Ridge (2018) | 443849 |
| 49 | 3.0417 | 3.0000 | 0.0000 | -0.0000 | 0.0417 | On the Run (1988) | 105910 |
| 50 | 3.0366 | 3.0000 | 0.0000 | -0.0000 | 0.0366 | Romance in a Minor Key (1943) | 148044 |
| 51 | 3.0341 | 3.0000 | 0.0000 | -0.0000 | 0.0341 | The Man Without a Map (1968) | 56599 |
| 52 | 3.0340 | 3.0000 | 0.0000 | -0.0000 | 0.0340 | Hell's Half Acre (1954) | 40872 |
| 53 | 3.0333 | 3.0000 | 0.0000 | -0.0000 | 0.0333 | Jetsam (2007) | 151883 |
| 54 | 3.0319 | 3.0000 | 0.0000 | -0.0000 | 0.0319 | In Broad Daylight (1971) | 252865 |
| 55 | 3.0309 | 3.0000 | 0.0000 | -0.0000 | 0.0309 | Second Nature (2003) | 80520 |
| 56 | 3.0271 | 3.0000 | 0.0000 | -0.0000 | 0.0271 | Diagnosis: Murder (1974) | 217439 |
| 57 | 3.0263 | 3.0000 | 0.0000 | -0.0000 | 0.0263 | Female Jungle (1956) | 46401 |
| 58 | 3.0260 | 3.0000 | 0.0000 | -0.0000 | 0.0260 | Vanjagar Ulagam (2018) | 505879 |
| 59 | 3.0253 | 3.0000 | 0.0000 | -0.0000 | 0.0253 | Hands of a Stranger (1987) | 410590 |
| 60 | 3.0251 | 3.0000 | 0.0000 | -0.0000 | 0.0251 | The Spider (1945) | 43537 |
| 61 | 3.0242 | 3.0000 | 0.0000 | -0.0000 | 0.0242 | Dial 999 (1955) | 61094 |
| 62 | 3.0242 | 3.0000 | 0.0000 | -0.0000 | 0.0242 | The Widow's Payback (2025) | 1560613 |
| 63 | 3.0235 | 3.0000 | 0.0000 | -0.0000 | 0.0235 | The Saint's Return (1953) | 83659 |
| 64 | 3.0234 | 3.0000 | 0.0000 | -0.0000 | 0.0234 | Killed My Wife (2019) | 588297 |
| 65 | 3.0229 | 3.0000 | 0.0000 | -0.0000 | 0.0229 | Dead End (1999) | 259928 |
| 66 | 3.0227 | 3.0000 | 0.0000 | -0.0000 | 0.0227 | American Nightmare (1983) | 85204 |
| 67 | 3.0226 | 3.0000 | 0.0000 | -0.0000 | 0.0226 | Mantrap (1953) | 72820 |
| 68 | 3.0225 | 3.0000 | 0.0000 | -0.0000 | 0.0225 | Jasmine (2015) | 325759 |
| 69 | 3.0209 | 3.0000 | 0.0000 | -0.0000 | 0.0209 | Disrupted (2020) | 682545 |
| 70 | 3.0200 | 3.0000 | 0.0000 | -0.0000 | 0.0200 | Marriage of Lies (2016) | 400995 |
| 71 | 3.0192 | 3.0000 | 0.0000 | -0.0000 | 0.0192 | False Identity (1990) | 541806 |
| 72 | 3.0185 | 3.0000 | 0.0000 | -0.0000 | 0.0185 | Cross Country (1983) | 107364 |
| 73 | 3.0184 | 3.0000 | 0.0000 | -0.0000 | 0.0184 | Monte Carlo Nights (1934) | 182423 |
| 74 | 3.0184 | 3.0000 | 0.0000 | -0.0000 | 0.0184 | The Lady Forgets (1989) | 528348 |
| 75 | 3.0179 | 3.0000 | 0.0000 | -0.0000 | 0.0179 | Off the Rails (2017) | 464684 |
| 76 | 3.0177 | 3.0000 | 0.0000 | -0.0000 | 0.0177 | Asylum (1997) | 208182 |
| 77 | 3.0164 | 3.0000 | 0.0000 | -0.0000 | 0.0164 | Fatal Instinct (2014) | 290772 |
| 78 | 3.0159 | 3.0000 | 0.0000 | -0.0000 | 0.0159 | Murder in My Mind (1997) | 315959 |
| 79 | 3.0153 | 3.0000 | 0.0000 | -0.0000 | 0.0153 | Papertrail (1998) | 86699 |
| 80 | 3.0150 | 3.0000 | 0.0000 | -0.0000 | 0.0150 | Deathmask (1984) | 78180 |
| 81 | 3.0143 | 2.9985 | 0.0000 | -0.0000 | 0.0159 | Deadly Surveillance (1991) | 110548 |
| 82 | 3.0134 | 3.0000 | 0.0000 | -0.0000 | 0.0134 | Killpoint (1984) | 108173 |
| 83 | 3.0125 | 3.0000 | 0.0000 | -0.0000 | 0.0125 | Vampz (2004) | 310451 |
| 84 | 3.0110 | 3.0000 | 0.0000 | -0.0000 | 0.0110 | Your Move (2017) | 490017 |
| 85 | 3.0110 | 3.0000 | 0.0000 | -0.0000 | 0.0110 | Lady in Waiting (1994) | 95206 |
| 86 | 3.0105 | 3.0000 | 0.0000 | -0.0000 | 0.0105 | Shattered Memories (2018) | 597710 |
| 87 | 3.0098 | 3.0000 | 0.0000 | -0.0000 | 0.0098 | Utopia (2024) | 1379587 |
| 88 | 3.0091 | 3.0000 | 0.0000 | -0.0000 | 0.0091 | Zen Noir (2004) | 34100 |
| 89 | 3.0069 | 3.0000 | 0.0000 | -0.0000 | 0.0069 | 6 Hours Away (2024) | 1273783 |
| 90 | 3.0054 | 3.0000 | 0.0000 | -0.0000 | 0.0054 | 12 to Midnight (2024) | 1147322 |
| 91 | 3.0043 | 3.0000 | 0.0000 | -0.0000 | 0.0043 | Final Payback (2001) | 478992 |
| 92 | 3.0018 | 3.0000 | 0.0000 | -0.0000 | 0.0018 | Snowman (2022) | 1012075 |
| 93 | 3.0017 | 3.0000 | 0.0000 | -0.0000 | 0.0017 | Chapel (2025) | 1012148 |
| 94 | 3.0017 | 3.0000 | 0.0000 | -0.0000 | 0.0017 | Prowler (2018) | 565243 |
| 95 | 3.0001 | 3.0000 | 0.0000 | -0.0000 | 0.0001 | 2177: The San Francisco Love Hacker Crimes (2019) | 614715 |
| 96 | 2.9906 | 2.9780 | 0.0000 | -0.0000 | 0.0126 | Stir (1997) | 416862 |
| 97 | 2.9836 | 2.8585 | 0.0000 | -0.0000 | 0.1251 | A Crime (2006) | 341 |
| 98 | 2.8908 | 2.8735 | 0.0000 | -0.0000 | 0.0173 | The Fifth Patient (2007) | 27434 |
| 99 | 2.8866 | 2.8740 | 0.0000 | -0.0000 | 0.0125 | Cop-Out (1991) | 322556 |
| 100 | 2.8732 | 2.8432 | 0.0000 | -0.0000 | 0.0300 | Mrithyunjay (2026) | 1428294 |

_total elapsed: 13.20s_
