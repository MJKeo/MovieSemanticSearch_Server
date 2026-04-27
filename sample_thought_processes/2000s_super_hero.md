# Query

> Best superhero movies from the 2000s, before the MCU took over everything.

## Step 2 — Query Pre-pass

_elapsed: 3.76s_

#### Step 2 Response
```json
{
  "overall_query_intention_exploration": "The user is seeking highly-regarded superhero films released during the 2000-2009 decade. They specifically want to exclude films belonging to the Marvel Cinematic Universe (MCU) franchise, focusing on the era of superhero cinema that preceded its dominance.",
  "requirements": [
    {
      "query_text": "superhero",
      "description": "The primary subject matter or genre of the films.",
      "modifiers": [],
      "coverage_evidence": [
        {
          "captured_meaning": "The user is looking for movies featuring superhero characters or themes.",
          "category_name": "Sub-genre + story archetype",
          "fit_quality": "clean",
          "atomic_rewrite": "superhero movies"
        }
      ]
    },
    {
      "query_text": "Best",
      "description": "A quality judgment seeking the highest-rated or most acclaimed entries.",
      "modifiers": [],
      "coverage_evidence": [
        {
          "captured_meaning": "The user is requesting films with high critical or audience reception.",
          "category_name": "Reception quality + superlative",
          "fit_quality": "clean",
          "atomic_rewrite": "highest-rated or best-received"
        }
      ]
    },
    {
      "query_text": "2000s",
      "description": "The specific release era for the films.",
      "modifiers": [],
      "coverage_evidence": [
        {
          "captured_meaning": "The films must have been released between the years 2000 and 2009.",
          "category_name": "Structured metadata",
          "fit_quality": "clean",
          "atomic_rewrite": "released in the 2000s"
        }
      ]
    },
    {
      "query_text": "MCU",
      "description": "The Marvel Cinematic Universe franchise.",
      "modifiers": [
        {
          "original_text": "before",
          "effect": "temporal exclusion — filters out movies released once this franchise became dominant",
          "type": "polarity_modifier"
        },
        {
          "original_text": "took over everything",
          "effect": "contextual emphasis on excluding the franchise's era and output",
          "type": "polarity_modifier"
        }
      ],
      "coverage_evidence": [
        {
          "captured_meaning": "The user wants to exclude movies that are part of the Marvel Cinematic Universe franchise.",
          "category_name": "Franchise / universe lineage",
          "fit_quality": "clean",
          "atomic_rewrite": "not part of the MCU franchise"
        }
      ]
    }
  ]
}
```

## Step 3 — Endpoint Translation + Execution

### CE 1 — Sub-genre + story archetype
_atomic_rewrite_: superhero movies

#### CE 1 payload
```json
{
  "captured_meaning": "The user is looking for movies featuring superhero characters or themes.",
  "category_name": "Sub-genre + story archetype",
  "atomic_rewrite": "superhero movies",
  "llm": {
    "elapsed_s": 5.108,
    "output": {
      "requirement_aspects": [
        {
          "aspect_description": "Identify superhero movies as the requested story/genre type.",
          "endpoint_coverage": "Keyword can cover this directly via the SUPERHERO registry member. Semantic could represent superhero as a genre signature in plot_analysis, but it is broader and less authoritative when a direct classification exists.",
          "best_endpoint": "keyword",
          "best_endpoint_gaps": null
        },
        {
          "aspect_description": "Exclude MCU-era films from the pool.",
          "endpoint_coverage": "Neither candidate is a good fit for the MCU-franchise exclusion itself. Keyword has no MCU registry member here, and Semantic is not the right place for franchise-specific filtering; that belongs to a franchise/entity endpoint outside this category.",
          "best_endpoint": "semantic",
          "best_endpoint_gaps": "The endpoint set in scope cannot faithfully encode the MCU exclusion."
        }
      ],
      "performance_vs_bias_analysis": "Keyword wins for the superhero sub-genre on its own merits because SUPERHERO is a direct canonical classification. The MCU exclusion cannot be handled by either candidate in scope, so it does not justify routing the whole requirement elsewhere. The bias is not doing any work here; the correct sub-genre target is Keyword, with the franchise constraint left uncovered by this category.",
      "endpoint_to_run": "keyword",
      "endpoint_parameters": {
        "match_mode": "filter",
        "parameters": {
          "concept_analysis": "The query asks for superhero movies — films centered on superhero characters, powers, or superheroic premises. This is the core genre/sub-genre signal.",
          "candidate_shortlist": "SUPERHERO is the direct registry member for this concept. Do not broaden to ACTION or ADVENTURE; those would lose the superhero-specific requirement.",
          "classification": "SUPERHERO"
        },
        "polarity": "positive"
      }
    }
  },
  "endpoints": [
    {
      "endpoint": "keyword",
      "match_mode": "filter",
      "polarity": "positive",
      "wrapper_type": "KeywordEndpointParameters",
      "parameters": {
        "concept_analysis": "The query asks for superhero movies — films centered on superhero characters, powers, or superheroic premises. This is the core genre/sub-genre signal.",
        "candidate_shortlist": "SUPERHERO is the direct registry member for this concept. Do not broaden to ACTION or ADVENTURE; those would lose the superhero-specific requirement.",
        "classification": "SUPERHERO"
      },
      "exec_elapsed_s": 0.023,
      "result": {
        "total_scored": 839,
        "top": [
          {
            "movie_id": 100356,
            "score": 1.0
          },
          {
            "movie_id": 331782,
            "score": 1.0
          },
          {
            "movie_id": 10249,
            "score": 1.0
          },
          {
            "movie_id": 8202,
            "score": 1.0
          },
          {
            "movie_id": 108560,
            "score": 1.0
          },
          {
            "movie_id": 1155089,
            "score": 1.0
          },
          {
            "movie_id": 18450,
            "score": 1.0
          },
          {
            "movie_id": 55326,
            "score": 1.0
          },
          {
            "movie_id": 2080,
            "score": 1.0
          },
          {
            "movie_id": 114721,
            "score": 1.0
          }
        ]
      }
    }
  ]
}
```

### CE 2 — Reception quality + superlative
_atomic_rewrite_: highest-rated or best-received

#### CE 2 payload
```json
{
  "captured_meaning": "The user is requesting films with high critical or audience reception.",
  "category_name": "Reception quality + superlative",
  "atomic_rewrite": "highest-rated or best-received",
  "llm": {
    "elapsed_s": 4.156,
    "output": {
      "requirement_aspects": [
        {
          "aspect_description": "Capture the superlative 'best' framing — the user wants the highest-rated or most acclaimed superhero movies in the slice they named.",
          "endpoint_coverage": "SEMANTIC can encode the acclaim/superlative prose in reception_summary and praised_qualities. METADATA can anchor the broad well_received prior, though it cannot express 'best' itself."
        }
      ],
      "overall_endpoint_fits": "SEMANTIC reception and METADATA reception both fit: the semantic channel carries the 'best / highly regarded' wording, while the metadata scalar adds a broad well_received prior. The superhero genre and 2000s era are handled by other categories, and the MCU exclusion is not a reception-quality signal here.",
      "per_endpoint_breakdown": {
        "semantic": {
          "should_run_endpoint": true,
          "endpoint_parameters": {
            "match_mode": "trait",
            "parameters": {
              "qualifier_inventory": "superlative acclaim; highest-rated / best-received framing within a slice.",
              "space_queries": [
                {
                  "carries_qualifiers": "reception carries the 'best' / highly regarded framing as reception prose, not a literal award or list membership.",
                  "space": "reception",
                  "weight": "central",
                  "content": {
                    "reception_summary": "widely regarded as one of the best-received and most acclaimed films in its slice; a top-tier entry by critical and audience response.",
                    "praised_qualities": [
                      "critical acclaim",
                      "strong audience reception",
                      "top-tier reputation",
                      "canonical standing"
                    ],
                    "criticized_qualities": []
                  }
                }
              ],
              "primary_vector": "reception"
            },
            "polarity": "positive"
          }
        },
        "metadata": {
          "should_run_endpoint": true,
          "endpoint_parameters": {
            "match_mode": "trait",
            "parameters": {
              "constraint_phrases": [
                "best"
              ],
              "target_attribute": "reception",
              "value_intent_label": "broadly well-received",
              "release_date": null,
              "runtime": null,
              "maturity_rating": null,
              "streaming": null,
              "audio_language": null,
              "country_of_origin": null,
              "budget_scale": null,
              "box_office": null,
              "popularity": null,
              "reception": "well_received"
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
        "qualifier_inventory": "superlative acclaim; highest-rated / best-received framing within a slice.",
        "space_queries": [
          {
            "carries_qualifiers": "reception carries the 'best' / highly regarded framing as reception prose, not a literal award or list membership.",
            "space": "reception",
            "weight": "central",
            "content": {
              "reception_summary": "widely regarded as one of the best-received and most acclaimed films in its slice; a top-tier entry by critical and audience response.",
              "praised_qualities": [
                "critical acclaim",
                "strong audience reception",
                "top-tier reputation",
                "canonical standing"
              ],
              "criticized_qualities": []
            }
          }
        ],
        "primary_vector": "reception"
      },
      "exec_elapsed_s": 0.736,
      "result": {
        "total_scored": 2000,
        "top": [
          {
            "movie_id": 196917,
            "score": 0.73500824
          },
          {
            "movie_id": 449632,
            "score": 0.73062897
          },
          {
            "movie_id": 69638,
            "score": 0.7282219
          },
          {
            "movie_id": 76541,
            "score": 0.7280083
          },
          {
            "movie_id": 304337,
            "score": 0.7279091
          },
          {
            "movie_id": 303296,
            "score": 0.723114
          },
          {
            "movie_id": 78955,
            "score": 0.72283936
          },
          {
            "movie_id": 27937,
            "score": 0.7209778
          },
          {
            "movie_id": 66342,
            "score": 0.71951675
          },
          {
            "movie_id": 367397,
            "score": 0.7187042
          }
        ]
      }
    },
    {
      "endpoint": "metadata",
      "match_mode": "trait",
      "polarity": "positive",
      "wrapper_type": "MetadataEndpointParameters",
      "parameters": {
        "constraint_phrases": [
          "best"
        ],
        "target_attribute": "reception",
        "value_intent_label": "broadly well-received",
        "release_date": null,
        "runtime": null,
        "maturity_rating": null,
        "streaming": null,
        "audio_language": null,
        "country_of_origin": null,
        "budget_scale": null,
        "box_office": null,
        "popularity": null,
        "reception": "well_received"
      },
      "exec_elapsed_s": 0.129,
      "result": {
        "total_scored": 5000,
        "top": [
          {
            "movie_id": 1599301,
            "score": 1.0
          },
          {
            "movie_id": 1621842,
            "score": 1.0
          },
          {
            "movie_id": 1290217,
            "score": 1.0
          },
          {
            "movie_id": 238,
            "score": 1.0
          },
          {
            "movie_id": 1490934,
            "score": 1.0
          },
          {
            "movie_id": 1467974,
            "score": 1.0
          },
          {
            "movie_id": 1120317,
            "score": 1.0
          },
          {
            "movie_id": 1552117,
            "score": 1.0
          },
          {
            "movie_id": 1059080,
            "score": 1.0
          },
          {
            "movie_id": 1635529,
            "score": 1.0
          }
        ]
      }
    }
  ]
}
```

### CE 3 — Structured metadata
_atomic_rewrite_: released in the 2000s

#### CE 3 payload
```json
{
  "captured_meaning": "The films must have been released between the years 2000 and 2009.",
  "category_name": "Structured metadata",
  "atomic_rewrite": "released in the 2000s",
  "llm": {
    "elapsed_s": 5.684,
    "output": {
      "requirement_aspects": [
        {
          "aspect_description": "The films must have been released in the 2000s.",
          "relation_to_endpoint": "The release_date column supports a between-dates predicate. '2000s' is a concrete decade window and can be expressed literally as 2000-01-01 through 2009-12-31.",
          "coverage_gaps": null
        }
      ],
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "filter",
        "parameters": {
          "constraint_phrases": [
            "2000s"
          ],
          "target_attribute": "release_date",
          "value_intent_label": "2000s decade window",
          "release_date": {
            "first_date": "2000-01-01",
            "match_operation": "between",
            "second_date": "2009-12-31"
          },
          "runtime": null,
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
          "2000s"
        ],
        "target_attribute": "release_date",
        "value_intent_label": "2000s decade window",
        "release_date": {
          "first_date": "2000-01-01",
          "match_operation": "between",
          "second_date": "2009-12-31"
        },
        "runtime": null,
        "maturity_rating": null,
        "streaming": null,
        "audio_language": null,
        "country_of_origin": null,
        "budget_scale": null,
        "box_office": null,
        "popularity": null,
        "reception": null
      },
      "exec_elapsed_s": 0.262,
      "result": {
        "total_scored": 32467,
        "top": [
          {
            "movie_id": 231,
            "score": 1.0
          },
          {
            "movie_id": 237,
            "score": 1.0
          },
          {
            "movie_id": 292,
            "score": 1.0
          },
          {
            "movie_id": 314,
            "score": 1.0
          },
          {
            "movie_id": 512,
            "score": 1.0
          },
          {
            "movie_id": 421,
            "score": 1.0
          },
          {
            "movie_id": 462,
            "score": 1.0
          },
          {
            "movie_id": 598,
            "score": 1.0
          },
          {
            "movie_id": 786,
            "score": 1.0
          },
          {
            "movie_id": 298,
            "score": 1.0
          }
        ]
      }
    }
  ]
}
```

### CE 4 — Franchise / universe lineage
_atomic_rewrite_: not part of the MCU franchise

#### CE 4 payload
```json
{
  "captured_meaning": "The user wants to exclude movies that are part of the Marvel Cinematic Universe franchise.",
  "category_name": "Franchise / universe lineage",
  "atomic_rewrite": "not part of the MCU franchise",
  "llm": {
    "elapsed_s": 5.894,
    "output": {
      "requirement_aspects": [
        {
          "aspect_description": "Exclude films that are part of the Marvel Cinematic Universe franchise.",
          "relation_to_endpoint": "This is directly expressible via the franchise_or_universe_names axis for the MCU shared universe. The endpoint can identify MCU membership and apply wrapper polarity to exclude it.",
          "coverage_gaps": null
        }
      ],
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "filter",
        "parameters": {
          "concept_analysis": "The atom names the MCU / Marvel Cinematic Universe, and the user wants those films excluded. This is a franchise/universe lineage constraint, not a genre or decade constraint. No subgroup, lineage position, structural flag, or launch-scope axis is needed.",
          "franchise_or_universe_names": [
            "marvel cinematic universe"
          ],
          "recognized_subgroups": null,
          "lineage_position": null,
          "structural_flags": null,
          "launch_scope": null,
          "prefer_lineage": false
        },
        "polarity": "negative"
      }
    }
  },
  "endpoints": [
    {
      "endpoint": "franchise_structure",
      "match_mode": "filter",
      "polarity": "negative",
      "wrapper_type": "FranchiseEndpointParameters",
      "parameters": {
        "concept_analysis": "The atom names the MCU / Marvel Cinematic Universe, and the user wants those films excluded. This is a franchise/universe lineage constraint, not a genre or decade constraint. No subgroup, lineage position, structural flag, or launch-scope axis is needed.",
        "franchise_or_universe_names": [
          "marvel cinematic universe"
        ],
        "recognized_subgroups": null,
        "lineage_position": null,
        "structural_flags": null,
        "launch_scope": null,
        "prefer_lineage": false
      },
      "exec_elapsed_s": 0.068,
      "result": {
        "total_scored": 48,
        "top": [
          {
            "movie_id": 447365,
            "score": 1.0
          },
          {
            "movie_id": 271110,
            "score": 1.0
          },
          {
            "movie_id": 758025,
            "score": 1.0
          },
          {
            "movie_id": 299534,
            "score": 1.0
          },
          {
            "movie_id": 299536,
            "score": 1.0
          },
          {
            "movie_id": 119569,
            "score": 1.0
          },
          {
            "movie_id": 640146,
            "score": 1.0
          },
          {
            "movie_id": 453395,
            "score": 1.0
          },
          {
            "movie_id": 609681,
            "score": 1.0
          },
          {
            "movie_id": 284053,
            "score": 1.0
          }
        ]
      }
    }
  ]
}
```

## Implicit Expectations

_elapsed: 1.84s_

#### Implicit Expectations Response
```json
{
  "query_intent_summary": "The user is looking for high-quality superhero films released during the 2000s, specifically excluding those from the Marvel Cinematic Universe.",
  "explicit_signals": [
    {
      "query_span": "superhero",
      "normalized_description": "superhero genre",
      "explicit_axis": "neither"
    },
    {
      "query_span": "Best",
      "normalized_description": "highest quality and most notable",
      "explicit_axis": "both"
    },
    {
      "query_span": "2000s",
      "normalized_description": "released between 2000 and 2009",
      "explicit_axis": "neither"
    },
    {
      "query_span": "MCU",
      "normalized_description": "excluding the Marvel Cinematic Universe",
      "explicit_axis": "neither"
    }
  ],
  "explicit_ordering_axis_analysis": "No explicit ordering axis such as trending or chronology is requested; the superlative 'Best' already addresses the ranking criteria.",
  "explicitly_addresses_quality": true,
  "explicitly_addresses_notability": true,
  "should_apply_quality_prior": false,
  "should_apply_notability_prior": false
}
```

## Consolidation Buckets (post fan-out)

#### Consolidation summary
```json
{
  "inclusion_unique_ids": 32953,
  "downrank_unique_ids": 0,
  "exclusion_unique_ids": 48,
  "preference_specs_count": 2,
  "used_fallback": "none"
}
```

## Final Score Breakdowns (top 100)

#### Top 100 ScoreBreakdowns
```json
[
  {
    "movie_id": 155,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.35775925325,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.3577592532500002
  },
  {
    "movie_id": 9806,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.34460844949999997,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.3446084495
  },
  {
    "movie_id": 558,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.29877438849999993,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.2987743885
  },
  {
    "movie_id": 187,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.2780241625,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.2780241625
  },
  {
    "movie_id": 1484397,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.269300374,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.269300374
  },
  {
    "movie_id": 213110,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.2684633413,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.2684633413
  },
  {
    "movie_id": 16234,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.2674707875,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.2674707874999998
  },
  {
    "movie_id": 15403,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.2650157895,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.2650157895
  },
  {
    "movie_id": 272,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.26462968174999996,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.26462968175
  },
  {
    "movie_id": 557,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.260812741,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.260812741
  },
  {
    "movie_id": 24660,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.259166488,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.259166488
  },
  {
    "movie_id": 11253,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.255855656,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.255855656
  },
  {
    "movie_id": 24675,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.25484770149999997,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.2548477015
  },
  {
    "movie_id": 24358,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.25459629475,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.25459629475
  },
  {
    "movie_id": 24357,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.2532691665,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.2532691665
  },
  {
    "movie_id": 24960,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.2532542068,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.2532542068
  },
  {
    "movie_id": 34003,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.2530811192,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.2530811192
  },
  {
    "movie_id": 671704,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.25087508775,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.25087508775
  },
  {
    "movie_id": 24959,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.2506382585,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.2506382585
  },
  {
    "movie_id": 24950,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.2500775025,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.2500775025
  },
  {
    "movie_id": 55931,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.24902607275,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.24902607275
  },
  {
    "movie_id": 24362,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.24895597825,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.24895597825
  },
  {
    "movie_id": 24914,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.2488185945,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.2488185945
  },
  {
    "movie_id": 40662,
    "inclusion_sum": 1.943013698630137,
    "downrank_sum": 0.0,
    "preference_contribution": 0.30468641,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.247700108630137
  },
  {
    "movie_id": 36658,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.246394834,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.246394834
  },
  {
    "movie_id": 624479,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.2431345357,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.2431345357
  },
  {
    "movie_id": 41207,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.24117898000000002,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.24117898
  },
  {
    "movie_id": 56828,
    "inclusion_sum": 1.9958904109589042,
    "downrank_sum": 0.0,
    "preference_contribution": 0.24441530749999996,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.240305718458904
  },
  {
    "movie_id": 15359,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.236836453,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.236836453
  },
  {
    "movie_id": 1487,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.23569220499999996,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.235692205
  },
  {
    "movie_id": 200770,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.23554932099999998,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.235549321
  },
  {
    "movie_id": 106583,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.2350554745,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.2350554745
  },
  {
    "movie_id": 752,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.23376755849999997,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.2337675585
  },
  {
    "movie_id": 118717,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.230312936,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.230312936
  },
  {
    "movie_id": 20771,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.22902449325000002,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.22902449325
  },
  {
    "movie_id": 30061,
    "inclusion_sum": 1.9852054794520548,
    "downrank_sum": 0.0,
    "preference_contribution": 0.24340856084999998,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.228614040302055
  },
  {
    "movie_id": 51948,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.22785848679999998,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.2278584867999998
  },
  {
    "movie_id": 115802,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.2276917545,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.2276917545
  },
  {
    "movie_id": 22855,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.226579577,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.226579577
  },
  {
    "movie_id": 17445,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.22297800350000002,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.2229780035
  },
  {
    "movie_id": 27245,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.222103966,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.222103966
  },
  {
    "movie_id": 36657,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.22134745499999992,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.221347455
  },
  {
    "movie_id": 16237,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.217668535,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.217668535
  },
  {
    "movie_id": 13640,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.21580531,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.21580531
  },
  {
    "movie_id": 14011,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.21542997735,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.21542997735
  },
  {
    "movie_id": 23483,
    "inclusion_sum": 1.9767123287671233,
    "downrank_sum": 0.0,
    "preference_contribution": 0.2374503005,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.2141626292671233
  },
  {
    "movie_id": 13053,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.2129634153499999,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.21296341535
  },
  {
    "movie_id": 32740,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.21228474574999998,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.21228474575
  },
  {
    "movie_id": 1452,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.20835837574999994,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.20835837575
  },
  {
    "movie_id": 65584,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.20833224649999998,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.2083322465
  },
  {
    "movie_id": 9741,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.20465892719999995,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.2046589272
  },
  {
    "movie_id": 20077,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.2041127095,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.2041127095
  },
  {
    "movie_id": 14830,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.20057991730000002,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.2005799173
  },
  {
    "movie_id": 8909,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.20015571450000003,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.2001557145
  },
  {
    "movie_id": 14609,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.1997724218,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1997724218
  },
  {
    "movie_id": 13851,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.1977649065,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1977649065
  },
  {
    "movie_id": 20776,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.1971232025,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1971232025
  },
  {
    "movie_id": 604,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.1966751563,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1966751563
  },
  {
    "movie_id": 11459,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.19586022749999993,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1958602275
  },
  {
    "movie_id": 24410,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.19401659024999998,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.19401659025
  },
  {
    "movie_id": 29764,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.19307488620000002,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1930748862
  },
  {
    "movie_id": 21683,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.1917787823,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1917787823
  },
  {
    "movie_id": 13183,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.1907972633,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1907972633
  },
  {
    "movie_id": 14611,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.19053613249999998,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1905361325
  },
  {
    "movie_id": 30675,
    "inclusion_sum": 1.990958904109589,
    "downrank_sum": 0.0,
    "preference_contribution": 0.19920100829999998,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.190159912409589
  },
  {
    "movie_id": 36223,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.18985249429999995,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1898524943
  },
  {
    "movie_id": 59387,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.188900586,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.188900586
  },
  {
    "movie_id": 41988,
    "inclusion_sum": 1.943013698630137,
    "downrank_sum": 0.0,
    "preference_contribution": 0.24516655099999998,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.188180249630137
  },
  {
    "movie_id": 37709,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.183716729,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.183716729
  },
  {
    "movie_id": 13204,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.1831795175,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1831795175
  },
  {
    "movie_id": 45745,
    "inclusion_sum": 1.9252054794520548,
    "downrank_sum": 0.0,
    "preference_contribution": 0.257595058,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1828005374520547
  },
  {
    "movie_id": 16774,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.179645613,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.179645613
  },
  {
    "movie_id": 36668,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.17594731349999998,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1759473135
  },
  {
    "movie_id": 62182,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.1731622809,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1731622809
  },
  {
    "movie_id": 184500,
    "inclusion_sum": 1.9024657534246576,
    "downrank_sum": 0.0,
    "preference_contribution": 0.270507685,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1729734384246577
  },
  {
    "movie_id": 11693,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.17227889665,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.17227889665
  },
  {
    "movie_id": 11662,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.17187496269999997,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1718749627
  },
  {
    "movie_id": 14613,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.1714704677,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1714704677
  },
  {
    "movie_id": 561,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.1706934355,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1706934355
  },
  {
    "movie_id": 29630,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.17024138600000002,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.170241386
  },
  {
    "movie_id": 64202,
    "inclusion_sum": 1.9024657534246576,
    "downrank_sum": 0.0,
    "preference_contribution": 0.26323051124999997,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.165696264674658
  },
  {
    "movie_id": 16187,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.16557406985,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.16557406985
  },
  {
    "movie_id": 559,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.16284873149999998,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1628487315
  },
  {
    "movie_id": 9824,
    "inclusion_sum": 1.9594520547945207,
    "downrank_sum": 0.0,
    "preference_contribution": 0.2003451485,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1597972032945205
  },
  {
    "movie_id": 13964,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.15856008,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.15856008
  },
  {
    "movie_id": 46718,
    "inclusion_sum": 1.9257534246575343,
    "downrank_sum": 0.0,
    "preference_contribution": 0.23013517379999998,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1558885984575342
  },
  {
    "movie_id": 106926,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.1529227525,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1529227525
  },
  {
    "movie_id": 36586,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.1476891752,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1476891752
  },
  {
    "movie_id": 24561,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.147290668,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.147290668
  },
  {
    "movie_id": 45162,
    "inclusion_sum": 1.9257534246575343,
    "downrank_sum": 0.0,
    "preference_contribution": 0.221091969,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1468453936575345
  },
  {
    "movie_id": 8285,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.145545386,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.145545386
  },
  {
    "movie_id": 15993,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.144773783,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1447737829999998
  },
  {
    "movie_id": 23127,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.14449191050000001,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1444919105
  },
  {
    "movie_id": 16577,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.1439474078,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1439474078
  },
  {
    "movie_id": 20693,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.14386964724999998,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.14386964725
  },
  {
    "movie_id": 22123,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.14293317149999998,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1429331715
  },
  {
    "movie_id": 29015,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.14287055929999998,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1428705593
  },
  {
    "movie_id": 43641,
    "inclusion_sum": 1.9123287671232876,
    "downrank_sum": 0.0,
    "preference_contribution": 0.22899122225000001,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1413199893732875
  },
  {
    "movie_id": 22154,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.1410176194,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1410176194
  },
  {
    "movie_id": 605,
    "inclusion_sum": 2.0,
    "downrank_sum": 0.0,
    "preference_contribution": 0.14063779835,
    "implicit_prior_contribution": 0.0,
    "final_score": 2.1406377983500002
  }
]
```

## Step 4 — Summary

### Filters
- superhero movies
- released in the 2000s
- not part of the MCU franchise

### Traits
- highest-rated or best-received

_used_fallback: none_

### query_intent_summary
The user is looking for high-quality superhero films released during the 2000s, specifically excluding those from the Marvel Cinematic Universe.

_implicit priors: quality=off  popularity=off_

### Top 100 Results

| # | final | filter | pref | down | impl | title (year) | tmdb_id |
|---|-------|--------|------|------|------|--------------|---------|
| 1 | 2.3578 | 2.0000 | 0.3578 | -0.0000 | 0.0000 | The Dark Knight (2008) | 155 |
| 2 | 2.3446 | 2.0000 | 0.3446 | -0.0000 | 0.0000 | The Incredibles (2004) | 9806 |
| 3 | 2.2988 | 2.0000 | 0.2988 | -0.0000 | 0.0000 | Spider-Man 2 (2004) | 558 |
| 4 | 2.2780 | 2.0000 | 0.2780 | -0.0000 | 0.0000 | Sin City (2005) | 187 |
| 5 | 2.2693 | 2.0000 | 0.2693 | -0.0000 | 0.0000 | Grayson (2004) | 1484397 |
| 6 | 2.2685 | 2.0000 | 0.2685 | -0.0000 | 0.0000 | The Amazing Screw-On Head (2006) | 213110 |
| 7 | 2.2675 | 2.0000 | 0.2675 | -0.0000 | 0.0000 | Batman Beyond: Return of the Joker (2000) | 16234 |
| 8 | 2.2650 | 2.0000 | 0.2650 | -0.0000 | 0.0000 | Ben 10: Secret of the Omnitrix (2007) | 15403 |
| 9 | 2.2646 | 2.0000 | 0.2646 | -0.0000 | 0.0000 | Batman Begins (2005) | 272 |
| 10 | 2.2608 | 2.0000 | 0.2608 | -0.0000 | 0.0000 | Spider-Man (2002) | 557 |
| 11 | 2.2592 | 2.0000 | 0.2592 | -0.0000 | 0.0000 | A Detective Story (2003) | 24660 |
| 12 | 2.2559 | 2.0000 | 0.2559 | -0.0000 | 0.0000 | Hellboy II: The Golden Army (2008) | 11253 |
| 13 | 2.2548 | 2.0000 | 0.2548 | -0.0000 | 0.0000 | Beyond (2003) | 24675 |
| 14 | 2.2546 | 2.0000 | 0.2546 | -0.0000 | 0.0000 | The Second Renaissance Part I (2003) | 24358 |
| 15 | 2.2533 | 2.0000 | 0.2533 | -0.0000 | 0.0000 | Final Flight of the Osiris (2003) | 24357 |
| 16 | 2.2533 | 2.0000 | 0.2533 | -0.0000 | 0.0000 | World Record (2003) | 24960 |
| 17 | 2.2531 | 2.0000 | 0.2531 | -0.0000 | 0.0000 | Turtles Forever (2009) | 34003 |
| 18 | 2.2509 | 2.0000 | 0.2509 | -0.0000 | 0.0000 | The Crusaders #357: Experiment in Evil! (2008) | 671704 |
| 19 | 2.2506 | 2.0000 | 0.2506 | -0.0000 | 0.0000 | Program (2003) | 24959 |
| 20 | 2.2501 | 2.0000 | 0.2501 | -0.0000 | 0.0000 | Matriculated (2003) | 24950 |
| 21 | 2.2490 | 2.0000 | 0.2490 | -0.0000 | 0.0000 | The Animatrix (2003) | 55931 |
| 22 | 2.2490 | 2.0000 | 0.2490 | -0.0000 | 0.0000 | The Second Renaissance Part II (2003) | 24362 |
| 23 | 2.2488 | 2.0000 | 0.2488 | -0.0000 | 0.0000 | Kid's Story (2003) | 24914 |
| 24 | 2.2477 | 1.9430 | 0.3047 | -0.0000 | 0.0000 | Batman: Under the Red Hood (2010) | 40662 |
| 25 | 2.2464 | 2.0000 | 0.2464 | -0.0000 | 0.0000 | X2 (2003) | 36658 |
| 26 | 2.2431 | 2.0000 | 0.2431 | -0.0000 | 0.0000 | Superman II: The Richard Donner Cut (2006) | 624479 |
| 27 | 2.2412 | 2.0000 | 0.2412 | -0.0000 | 0.0000 | Under the Hood (2009) | 41207 |
| 28 | 2.2403 | 1.9959 | 0.2444 | -0.0000 | 0.0000 | Dexter's Laboratory: Ego Trip (1999) | 56828 |
| 29 | 2.2368 | 2.0000 | 0.2368 | -0.0000 | 0.0000 | Wonder Woman (2009) | 15359 |
| 30 | 2.2357 | 2.0000 | 0.2357 | -0.0000 | 0.0000 | Hellboy (2004) | 1487 |
| 31 | 2.2355 | 2.0000 | 0.2355 | -0.0000 | 0.0000 | Ultraman Cosmos 2: The Blue Planet (2002) | 200770 |
| 32 | 2.2351 | 2.0000 | 0.2351 | -0.0000 | 0.0000 | Chase Me (2003) | 106583 |
| 33 | 2.2338 | 2.0000 | 0.2338 | -0.0000 | 0.0000 | V for Vendetta (2006) | 752 |
| 34 | 2.2303 | 2.0000 | 0.2303 | -0.0000 | 0.0000 | Ultraman Mebius & Ultra Brothers (2006) | 118717 |
| 35 | 2.2290 | 2.0000 | 0.2290 | -0.0000 | 0.0000 | Kim Possible Movie: So the Drama (2005) | 20771 |
| 36 | 2.2286 | 1.9852 | 0.2434 | -0.0000 | 0.0000 | Justice League: Crisis on Two Earths (2010) | 30061 |
| 37 | 2.2279 | 2.0000 | 0.2279 | -0.0000 | 0.0000 | Mega Monster Battle: Ultra Galaxy Legends The Movie (2009) | 51948 |
| 38 | 2.2277 | 2.0000 | 0.2277 | -0.0000 | 0.0000 | The Lobo Paramilitary Christmas Special (2002) | 115802 |
| 39 | 2.2266 | 2.0000 | 0.2266 | -0.0000 | 0.0000 | Superman/Batman: Public Enemies (2009) | 22855 |
| 40 | 2.2230 | 2.0000 | 0.2230 | -0.0000 | 0.0000 | Green Lantern: First Flight (2009) | 17445 |
| 41 | 2.2221 | 2.0000 | 0.2221 | -0.0000 | 0.0000 | Electric Dragon 80000V (2001) | 27245 |
| 42 | 2.2213 | 2.0000 | 0.2213 | -0.0000 | 0.0000 | X-Men (2000) | 36657 |
| 43 | 2.2177 | 2.0000 | 0.2177 | -0.0000 | 0.0000 | Teen Titans: Trouble in Tokyo (2006) | 16237 |
| 44 | 2.2158 | 2.0000 | 0.2158 | -0.0000 | 0.0000 | Superman: Doomsday (2007) | 13640 |
| 45 | 2.2154 | 2.0000 | 0.2154 | -0.0000 | 0.0000 | Justice League: The New Frontier (2008) | 14011 |
| 46 | 2.2142 | 1.9767 | 0.2375 | -0.0000 | 0.0000 | Kick-Ass (2010) | 23483 |
| 47 | 2.2130 | 2.0000 | 0.2130 | -0.0000 | 0.0000 | Bolt (2008) | 13053 |
| 48 | 2.2123 | 2.0000 | 0.2123 | -0.0000 | 0.0000 | Krrish (2006) | 32740 |
| 49 | 2.2084 | 2.0000 | 0.2084 | -0.0000 | 0.0000 | Superman Returns (2006) | 1452 |
| 50 | 2.2083 | 2.0000 | 0.2083 | -0.0000 | 0.0000 | Aquaman (2006) | 65584 |
| 51 | 2.2047 | 2.0000 | 0.2047 | -0.0000 | 0.0000 | Unbreakable (2000) | 9741 |
| 52 | 2.2041 | 2.0000 | 0.2041 | -0.0000 | 0.0000 | The Batman vs. Dracula (2005) | 20077 |
| 53 | 2.2006 | 2.0000 | 0.2006 | -0.0000 | 0.0000 | Doctor Strange (2007) | 14830 |
| 54 | 2.2002 | 2.0000 | 0.2002 | -0.0000 | 0.0000 | Wanted (2008) | 8909 |
| 55 | 2.1998 | 2.0000 | 0.1998 | -0.0000 | 0.0000 | Ultimate Avengers: The Movie (2006) | 14609 |
| 56 | 2.1978 | 2.0000 | 0.1978 | -0.0000 | 0.0000 | Batman: Gotham Knight (2008) | 13851 |
| 57 | 2.1971 | 2.0000 | 0.1971 | -0.0000 | 0.0000 | Return to the Batcave: The Misadventures of Adam and Burt (2003) | 20776 |
| 58 | 2.1967 | 2.0000 | 0.1967 | -0.0000 | 0.0000 | The Matrix Reloaded (2003) | 604 |
| 59 | 2.1959 | 2.0000 | 0.1959 | -0.0000 | 0.0000 | Sky High (2005) | 11459 |
| 60 | 2.1940 | 2.0000 | 0.1940 | -0.0000 | 0.0000 | K-20: The Fiend with Twenty Faces (2008) | 24410 |
| 61 | 2.1931 | 2.0000 | 0.1931 | -0.0000 | 0.0000 | Ultraman: The Next (2004) | 29764 |
| 62 | 2.1918 | 2.0000 | 0.1918 | -0.0000 | 0.0000 | Batman: Mystery of the Batwoman (2003) | 21683 |
| 63 | 2.1908 | 2.0000 | 0.1908 | -0.0000 | 0.0000 | Watchmen (2009) | 13183 |
| 64 | 2.1905 | 2.0000 | 0.1905 | -0.0000 | 0.0000 | Ultimate Avengers 2 (2006) | 14611 |
| 65 | 2.1902 | 1.9910 | 0.1992 | -0.0000 | 0.0000 | Planet Hulk (2010) | 30675 |
| 66 | 2.1899 | 2.0000 | 0.1899 | -0.0000 | 0.0000 | Zebraman (2004) | 36223 |
| 67 | 2.1889 | 2.0000 | 0.1889 | -0.0000 | 0.0000 | The Powerpuff Girls Movie (2002) | 59387 |
| 68 | 2.1882 | 1.9430 | 0.2452 | -0.0000 | 0.0000 | DC Showcase: Jonah Hex (2010) | 41988 |
| 69 | 2.1837 | 2.0000 | 0.1837 | -0.0000 | 0.0000 | Mirageman (2007) | 37709 |
| 70 | 2.1832 | 2.0000 | 0.1832 | -0.0000 | 0.0000 | Hellboy Animated: Blood and Iron (2007) | 13204 |
| 71 | 2.1828 | 1.9252 | 0.2576 | -0.0000 | 0.0000 | Sintel (2010) | 45745 |
| 72 | 2.1796 | 2.0000 | 0.1796 | -0.0000 | 0.0000 | Hellboy Animated: Sword of Storms (2006) | 16774 |
| 73 | 2.1759 | 2.0000 | 0.1759 | -0.0000 | 0.0000 | X-Men: The Last Stand (2006) | 36668 |
| 74 | 2.1732 | 2.0000 | 0.1732 | -0.0000 | 0.0000 | Devilman - Volume 3: Devilman Apocalypse (2000) | 62182 |
| 75 | 2.1730 | 1.9025 | 0.2705 | -0.0000 | 0.0000 | Heat Vision and Jack (1999) | 184500 |
| 76 | 2.1723 | 2.0000 | 0.1723 | -0.0000 | 0.0000 | Big Man Japan (2007) | 11693 |
| 77 | 2.1719 | 2.0000 | 0.1719 | -0.0000 | 0.0000 | Casshern (2004) | 11662 |
| 78 | 2.1715 | 2.0000 | 0.1715 | -0.0000 | 0.0000 | Next Avengers: Heroes of Tomorrow (2008) | 14613 |
| 79 | 2.1707 | 2.0000 | 0.1707 | -0.0000 | 0.0000 | Constantine (2005) | 561 |
| 80 | 2.1702 | 2.0000 | 0.1702 | -0.0000 | 0.0000 | Milarepa (2006) | 29630 |
| 81 | 2.1657 | 1.9025 | 0.2632 | -0.0000 | 0.0000 | Batman Beyond: The Movie (1999) | 64202 |
| 82 | 2.1656 | 2.0000 | 0.1656 | -0.0000 | 0.0000 | Buzz Lightyear of Star Command: The Adventure Begins (2000) | 16187 |
| 83 | 2.1628 | 2.0000 | 0.1628 | -0.0000 | 0.0000 | Spider-Man 3 (2007) | 559 |
| 84 | 2.1598 | 1.9595 | 0.2003 | -0.0000 | 0.0000 | Mystery Men (1999) | 9824 |
| 85 | 2.1586 | 2.0000 | 0.1586 | -0.0000 | 0.0000 | The Machine Girl (2008) | 13964 |
| 86 | 2.1559 | 1.9258 | 0.2301 | -0.0000 | 0.0000 | DC Showcase: Green Arrow (2010) | 46718 |
| 87 | 2.1529 | 2.0000 | 0.1529 | -0.0000 | 0.0000 | Gagamboy (2004) | 106926 |
| 88 | 2.1477 | 2.0000 | 0.1477 | -0.0000 | 0.0000 | Blade II (2002) | 36586 |
| 89 | 2.1473 | 2.0000 | 0.1473 | -0.0000 | 0.0000 | Yatterman (2009) | 24561 |
| 90 | 2.1468 | 1.9258 | 0.2211 | -0.0000 | 0.0000 | Superman/Batman: Apocalypse (2010) | 45162 |
| 91 | 2.1455 | 2.0000 | 0.1455 | -0.0000 | 0.0000 | The Spirit (2008) | 8285 |
| 92 | 2.1448 | 2.0000 | 0.1448 | -0.0000 | 0.0000 | Gen¹³ (2000) | 15993 |
| 93 | 2.1445 | 2.0000 | 0.1445 | -0.0000 | 0.0000 | The Haunted World of El Superbeasto (2009) | 23127 |
| 94 | 2.1439 | 2.0000 | 0.1439 | -0.0000 | 0.0000 | Astro Boy (2009) | 16577 |
| 95 | 2.1439 | 2.0000 | 0.1439 | -0.0000 | 0.0000 | Santos (2008) | 20693 |
| 96 | 2.1429 | 2.0000 | 0.1429 | -0.0000 | 0.0000 | The Chosen One (2007) | 22123 |
| 97 | 2.1429 | 2.0000 | 0.1429 | -0.0000 | 0.0000 | The Specials (2000) | 29015 |
| 98 | 2.1413 | 1.9123 | 0.2290 | -0.0000 | 0.0000 | Superman/Shazam!: The Return of Black Adam (2010) | 43641 |
| 99 | 2.1410 | 2.0000 | 0.1410 | -0.0000 | 0.0000 | Mosaic (2007) | 22154 |
| 100 | 2.1406 | 2.0000 | 0.1406 | -0.0000 | 0.0000 | The Matrix Revolutions (2003) | 605 |

_total elapsed: 10.51s_
