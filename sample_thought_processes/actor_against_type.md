# Query

> What are the best dramatic performances from actors who are mostly known for comedy?

## Step 2 — Query Pre-pass

_elapsed: 3.84s_

#### Step 2 Response
```json
{
  "overall_query_intention_exploration": "The user is looking for movies where actors typically associated with comedic roles deliver high-quality dramatic performances. This request focuses on the contrast between an actor's established persona and a specific performance's genre, emphasizing high reception quality for that specific turn.",
  "requirements": [
    {
      "query_text": "best",
      "description": "A superlative indicating high quality or critical acclaim.",
      "modifiers": [],
      "coverage_evidence": [
        {
          "captured_meaning": "the user is looking for movies with high critical or audience reception",
          "category_name": "Reception quality + superlative",
          "fit_quality": "clean",
          "atomic_rewrite": "best quality or highly acclaimed"
        }
      ]
    },
    {
      "query_text": "dramatic performances",
      "description": "Specifies the genre of the performance and the focus on acting quality.",
      "modifiers": [],
      "coverage_evidence": [
        {
          "captured_meaning": "the movie belongs to the drama genre",
          "category_name": "Top-level genre",
          "fit_quality": "partial",
          "atomic_rewrite": "drama genre"
        },
        {
          "captured_meaning": "the request focuses on the quality of the acting craft",
          "category_name": "Craft acclaim",
          "fit_quality": "partial",
          "atomic_rewrite": "acclaimed acting performances"
        }
      ]
    },
    {
      "query_text": "actors who are mostly known for comedy",
      "description": "Describes a specific type of actor based on their typical career output.",
      "modifiers": [],
      "coverage_evidence": [
        {
          "captured_meaning": "the lead or featured actors are primarily associated with the comedy genre in their broader career",
          "category_name": "Interpretation-required",
          "fit_quality": "clean",
          "atomic_rewrite": "starring actors who are mostly known for comedy"
        }
      ]
    }
  ]
}
```

## Step 3 — Endpoint Translation + Execution

### CE 1 — Reception quality + superlative
_atomic_rewrite_: best quality or highly acclaimed

#### CE 1 payload
```json
{
  "captured_meaning": "the user is looking for movies with high critical or audience reception",
  "category_name": "Reception quality + superlative",
  "atomic_rewrite": "best quality or highly acclaimed",
  "llm": {
    "elapsed_s": 5.188,
    "output": {
      "requirement_aspects": [
        {
          "aspect_description": "Capture the superlative 'best' as a high-regard signal for the works being sought.",
          "endpoint_coverage": "SEMANTIC can encode the acclaim shape in reception_summary/praised_qualities. METADATA can anchor this as broadly well_received."
        },
        {
          "aspect_description": "Keep the focus on dramatic performances by actors mostly known for comedy.",
          "endpoint_coverage": "SEMANTIC can partially reflect the performance-side praise, but the acting/genre contrast itself is better handled by the separate acting/genre categories; METADATA does not capture that contrast."
        }
      ],
      "overall_endpoint_fits": "SEMANTIC reception fits the acclaim framing, and METADATA reception should also fire as a broad well_received prior. The dramatic-performance/comedy-actor contrast is not fully represented here and is mostly outside this category’s scope.",
      "per_endpoint_breakdown": {
        "semantic": {
          "should_run_endpoint": true,
          "endpoint_parameters": {
            "match_mode": "trait",
            "parameters": {
              "qualifier_inventory": "superlative acclaim; performance quality; contrast between comedic persona and dramatic turn",
              "space_queries": [
                {
                  "carries_qualifiers": "captures the 'best' / high-regard framing as reception prose and praise axes.",
                  "space": "reception",
                  "weight": "central",
                  "content": {
                    "reception_summary": "widely regarded as a standout dramatic turn; frequently cited among the best performances in this lane.",
                    "praised_qualities": [
                      "standout performance",
                      "critical acclaim",
                      "dramatic range",
                      "career-redefining turn"
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
        "qualifier_inventory": "superlative acclaim; performance quality; contrast between comedic persona and dramatic turn",
        "space_queries": [
          {
            "carries_qualifiers": "captures the 'best' / high-regard framing as reception prose and praise axes.",
            "space": "reception",
            "weight": "central",
            "content": {
              "reception_summary": "widely regarded as a standout dramatic turn; frequently cited among the best performances in this lane.",
              "praised_qualities": [
                "standout performance",
                "critical acclaim",
                "dramatic range",
                "career-redefining turn"
              ],
              "criticized_qualities": []
            }
          }
        ],
        "primary_vector": "reception"
      },
      "exec_elapsed_s": 0.715,
      "result": {
        "total_scored": 2000,
        "top": [
          {
            "movie_id": 1463175,
            "score": 0.72496796
          },
          {
            "movie_id": 396,
            "score": 0.6964493
          },
          {
            "movie_id": 305237,
            "score": 0.6915474
          },
          {
            "movie_id": 500189,
            "score": 0.6914673
          },
          {
            "movie_id": 3549,
            "score": 0.68606186
          },
          {
            "movie_id": 53229,
            "score": 0.6843605
          },
          {
            "movie_id": 1999,
            "score": 0.6842346
          },
          {
            "movie_id": 1233327,
            "score": 0.68328476
          },
          {
            "movie_id": 71207,
            "score": 0.6818886
          },
          {
            "movie_id": 76541,
            "score": 0.6815758
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
      "exec_elapsed_s": 0.379,
      "result": {
        "total_scored": 5000,
        "top": [
          {
            "movie_id": 1621842,
            "score": 1.0
          },
          {
            "movie_id": 1599301,
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
            "movie_id": 1120317,
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
            "movie_id": 1552117,
            "score": 1.0
          },
          {
            "movie_id": 1059080,
            "score": 1.0
          },
          {
            "movie_id": 928537,
            "score": 1.0
          }
        ]
      }
    }
  ]
}
```

### CE 2 — Top-level genre
_atomic_rewrite_: drama genre

#### CE 2 payload
```json
{
  "captured_meaning": "the movie belongs to the drama genre",
  "category_name": "Top-level genre",
  "atomic_rewrite": "drama genre",
  "llm": {
    "elapsed_s": 3.535,
    "output": {
      "requirement_aspects": [
        {
          "aspect_description": "Find movies with dramatic performances (the acting turn is dramatic, not the film simply being a drama).",
          "endpoint_coverage": "Keyword can only resolve the bare top-level genre DRAMA, which would over-broaden to any drama film and miss the performance-level contrast. Semantic can capture the qualified genre/performative texture in plot_analysis or reception, but the request is really about actor performance quality rather than movie genre.",
          "best_endpoint": "semantic",
          "best_endpoint_gaps": "Even semantic is only an approximate fit here: it can describe dramatic performance quality, but the named actor persona contrast ('mostly known for comedy') belongs to entity/franchise handling, not this endpoint."
        },
        {
          "aspect_description": "Find performances that are considered the best among that kind of role.",
          "endpoint_coverage": "Keyword has no clean registry member for 'best performance' or 'comedian doing drama' as a genre label. Semantic reception could capture praised acting, but this endpoint category is not the right owner for performance-quality comparison.",
          "best_endpoint": "semantic",
          "best_endpoint_gaps": "It can express praise around acting, but not rank across actors or specifically bind the performance to comedians without another endpoint."
        }
      ],
      "endpoint_to_run": "None",
      "endpoint_parameters": null
    }
  },
  "endpoints": [],
  "note": "LLM returned no fired endpoints"
}
```

### CE 3 — Craft acclaim
_atomic_rewrite_: acclaimed acting performances

#### CE 3 payload
```json
{
  "captured_meaning": "the request focuses on the quality of the acting craft",
  "category_name": "Craft acclaim",
  "atomic_rewrite": "acclaimed acting performances",
  "llm": {
    "elapsed_s": 2.951,
    "output": {
      "requirement_aspects": [
        {
          "aspect_description": "Find dramatic acting performances that are especially well-regarded.",
          "relation_to_endpoint": "Reception can capture the 'best' / high-quality acclaim angle for acting as a praised quality, and narrative/production spaces do not fit this ask because it is about performance reception rather than story or filming craft.",
          "coverage_gaps": "The endpoint cannot represent the actor-side constraint — 'actors who are mostly known for comedy' — because that is a credit/persona filter, not an axis of reception praise."
        },
        {
          "aspect_description": "Restrict to performances by actors mostly known for comedy.",
          "relation_to_endpoint": "This endpoint does not cover named actors or career-profile constraints. It can only model praise for the performance, not the performer's comedy background.",
          "coverage_gaps": "The actor-type constraint belongs to a credit/persona endpoint, not craft acclaim."
        }
      ],
      "should_run_endpoint": false,
      "endpoint_parameters": null
    }
  },
  "endpoints": [],
  "note": "LLM returned no fired endpoints"
}
```

### CE 4 — Interpretation-required
_atomic_rewrite_: starring actors who are mostly known for comedy

#### CE 4 payload
```json
{
  "captured_meaning": "the lead or featured actors are primarily associated with the comedy genre in their broader career",
  "category_name": "Interpretation-required",
  "atomic_rewrite": "starring actors who are mostly known for comedy",
  "llm": {
    "elapsed_s": 3.317,
    "output": {
      "requirement_aspects": [
        {
          "aspect_description": "Actors whose broader careers are mostly comedy, but here are wanted in a dramatic turn.",
          "relation_to_endpoint": "This is a named-entity/career-profile ask, which the semantic fallback should not own when a structured creator/performer category exists. Even though interpretation could phrase it as a performer-profile signal, this endpoint cannot reliably encode actor identity or career-wide comedy association.",
          "coverage_gaps": "Out of scope for this endpoint; the core signal is a performer/creator identity attribute, not a movie-internal semantic feature. Cat 31 should not fire for this structured ask."
        },
        {
          "aspect_description": "A dramatic performance that is especially strong or 'best'.",
          "relation_to_endpoint": "The semantic spaces can approximate dramatic tone or reception praise, but they cannot directly rank an individual actor's performance quality or compare 'best' performances across a comedian's filmography.",
          "coverage_gaps": "Performance-level quality is not representable here without inventing reception or plot terms that do not capture the actor-specific comparison."
        }
      ],
      "should_run_endpoint": false,
      "endpoint_parameters": null
    }
  },
  "endpoints": [],
  "note": "LLM returned no fired endpoints"
}
```

## Implicit Expectations

_elapsed: 1.85s_

#### Implicit Expectations Response
```json
{
  "query_intent_summary": "The user is seeking highly-regarded dramatic acting turns from performers who typically work in comedy, explicitly requesting the 'best' examples.",
  "explicit_signals": [
    {
      "query_span": "best",
      "normalized_description": "request for high quality or critical acclaim",
      "explicit_axis": "both"
    },
    {
      "query_span": "dramatic performances",
      "normalized_description": "performances within the drama genre",
      "explicit_axis": "neither"
    },
    {
      "query_span": "actors who are mostly known for comedy",
      "normalized_description": "actors whose primary career association is with comedy",
      "explicit_axis": "neither"
    }
  ],
  "explicit_ordering_axis_analysis": "The query uses the superlative 'best' to define the ordering axis, which is already captured as an explicit quality and notability signal. No other ordering axis like chronology or trending is requested.",
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
  "inclusion_unique_ids": 0,
  "downrank_unique_ids": 0,
  "exclusion_unique_ids": 0,
  "preference_specs_count": 2,
  "used_fallback": "preferences_as_candidates"
}
```

## Final Score Breakdowns (top 100)

#### Top 100 ScoreBreakdowns
```json
[
  {
    "movie_id": 307252,
    "inclusion_sum": 1.6005028000000001,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.6005028000000001
  },
  {
    "movie_id": 496243,
    "inclusion_sum": 1.5865171999999998,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5865171999999998
  },
  {
    "movie_id": 2457,
    "inclusion_sum": 1.5781953400000002,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5781953400000002
  },
  {
    "movie_id": 1438424,
    "inclusion_sum": 1.5722885,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5722885
  },
  {
    "movie_id": 1240009,
    "inclusion_sum": 1.5654724,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5654724
  },
  {
    "movie_id": 702,
    "inclusion_sum": 1.5642534,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5642534
  },
  {
    "movie_id": 210763,
    "inclusion_sum": 1.56374207,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.56374207
  },
  {
    "movie_id": 1245119,
    "inclusion_sum": 1.5598991400000002,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5598991400000002
  },
  {
    "movie_id": 418204,
    "inclusion_sum": 1.5539253,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5539253
  },
  {
    "movie_id": 305237,
    "inclusion_sum": 1.5512321500000001,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5512321500000001
  },
  {
    "movie_id": 11159,
    "inclusion_sum": 1.5505217,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5505217
  },
  {
    "movie_id": 1600694,
    "inclusion_sum": 1.5491844000000001,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5491844000000001
  },
  {
    "movie_id": 1482871,
    "inclusion_sum": 1.5451842999999998,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5451842999999998
  },
  {
    "movie_id": 314365,
    "inclusion_sum": 1.5449285999999998,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5449285999999998
  },
  {
    "movie_id": 11697,
    "inclusion_sum": 1.54388367,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.54388367
  },
  {
    "movie_id": 26910,
    "inclusion_sum": 1.54349823,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.54349823
  },
  {
    "movie_id": 492188,
    "inclusion_sum": 1.54236755,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.54236755
  },
  {
    "movie_id": 957430,
    "inclusion_sum": 1.5420349,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5420349
  },
  {
    "movie_id": 378227,
    "inclusion_sum": 1.5366417,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5366417
  },
  {
    "movie_id": 45269,
    "inclusion_sum": 1.53431564,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.53431564
  },
  {
    "movie_id": 544795,
    "inclusion_sum": 1.5296607999999998,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5296607999999998
  },
  {
    "movie_id": 1493253,
    "inclusion_sum": 1.5285430999999998,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5285430999999998
  },
  {
    "movie_id": 66342,
    "inclusion_sum": 1.5282051,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5282051
  },
  {
    "movie_id": 859987,
    "inclusion_sum": 1.5278763,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5278763
  },
  {
    "movie_id": 66389,
    "inclusion_sum": 1.5272171,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5272171
  },
  {
    "movie_id": 77660,
    "inclusion_sum": 1.5268631,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5268631
  },
  {
    "movie_id": 700394,
    "inclusion_sum": 1.52653046,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.52653046
  },
  {
    "movie_id": 990,
    "inclusion_sum": 1.5253375,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5253375
  },
  {
    "movie_id": 14537,
    "inclusion_sum": 1.5243558000000001,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5243558000000001
  },
  {
    "movie_id": 522986,
    "inclusion_sum": 1.5222587600000002,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5222587600000002
  },
  {
    "movie_id": 66362,
    "inclusion_sum": 1.5210747,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5210747
  },
  {
    "movie_id": 660120,
    "inclusion_sum": 1.52086914,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.52086914
  },
  {
    "movie_id": 11659,
    "inclusion_sum": 1.5176071,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5176071
  },
  {
    "movie_id": 280277,
    "inclusion_sum": 1.5172004700000001,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5172004700000001
  },
  {
    "movie_id": 1390165,
    "inclusion_sum": 1.51661224,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.51661224
  },
  {
    "movie_id": 12498,
    "inclusion_sum": 1.51636276,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.51636276
  },
  {
    "movie_id": 462718,
    "inclusion_sum": 1.515876,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.515876
  },
  {
    "movie_id": 496056,
    "inclusion_sum": 1.5152817,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5152817
  },
  {
    "movie_id": 29845,
    "inclusion_sum": 1.5132089,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5132089
  },
  {
    "movie_id": 38,
    "inclusion_sum": 1.5114893999999999,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5114893999999999
  },
  {
    "movie_id": 1340209,
    "inclusion_sum": 1.5088875,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5088875
  },
  {
    "movie_id": 1999,
    "inclusion_sum": 1.50867905,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.50867905
  },
  {
    "movie_id": 595228,
    "inclusion_sum": 1.5078499,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5078499
  },
  {
    "movie_id": 196917,
    "inclusion_sum": 1.50783234,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.50783234
  },
  {
    "movie_id": 29478,
    "inclusion_sum": 1.5077840999999998,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5077840999999998
  },
  {
    "movie_id": 449632,
    "inclusion_sum": 1.50719604,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.50719604
  },
  {
    "movie_id": 916405,
    "inclusion_sum": 1.50696014,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.50696014
  },
  {
    "movie_id": 11450,
    "inclusion_sum": 1.5064138999999999,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5064138999999999
  },
  {
    "movie_id": 459713,
    "inclusion_sum": 1.50570374,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.50570374
  },
  {
    "movie_id": 21575,
    "inclusion_sum": 1.5050596600000001,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5050596600000001
  },
  {
    "movie_id": 69638,
    "inclusion_sum": 1.5032059,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5032059
  },
  {
    "movie_id": 66351,
    "inclusion_sum": 1.5023422,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5023422
  },
  {
    "movie_id": 18900,
    "inclusion_sum": 1.5018501,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5018501
  },
  {
    "movie_id": 69550,
    "inclusion_sum": 1.50115204,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.50115204
  },
  {
    "movie_id": 461126,
    "inclusion_sum": 1.5010895,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5010895
  },
  {
    "movie_id": 600354,
    "inclusion_sum": 1.5005801399999998,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.5005801399999998
  },
  {
    "movie_id": 26451,
    "inclusion_sum": 1.4991978500000003,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4991978500000003
  },
  {
    "movie_id": 473033,
    "inclusion_sum": 1.49870575,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.49870575
  },
  {
    "movie_id": 32499,
    "inclusion_sum": 1.4979642000000002,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4979642000000002
  },
  {
    "movie_id": 1190085,
    "inclusion_sum": 1.4972618,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4972618
  },
  {
    "movie_id": 575351,
    "inclusion_sum": 1.49695206,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.49695206
  },
  {
    "movie_id": 244049,
    "inclusion_sum": 1.4967735000000002,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4967735000000002
  },
  {
    "movie_id": 311291,
    "inclusion_sum": 1.4949747,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4949747
  },
  {
    "movie_id": 618208,
    "inclusion_sum": 1.4938463,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4938463
  },
  {
    "movie_id": 49032,
    "inclusion_sum": 1.49374084,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.49374084
  },
  {
    "movie_id": 1447109,
    "inclusion_sum": 1.49308853,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.49308853
  },
  {
    "movie_id": 31217,
    "inclusion_sum": 1.4928664999999999,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4928664999999999
  },
  {
    "movie_id": 28571,
    "inclusion_sum": 1.4917709399999999,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4917709399999999
  },
  {
    "movie_id": 29811,
    "inclusion_sum": 1.4910202,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4910202
  },
  {
    "movie_id": 265169,
    "inclusion_sum": 1.4903283699999998,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4903283699999998
  },
  {
    "movie_id": 595671,
    "inclusion_sum": 1.48979156,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.48979156
  },
  {
    "movie_id": 567410,
    "inclusion_sum": 1.48972046,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.48972046
  },
  {
    "movie_id": 51802,
    "inclusion_sum": 1.4896365399999998,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4896365399999998
  },
  {
    "movie_id": 16672,
    "inclusion_sum": 1.4884041,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4884041
  },
  {
    "movie_id": 27937,
    "inclusion_sum": 1.48804245,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.48804245
  },
  {
    "movie_id": 78955,
    "inclusion_sum": 1.4880203,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4880203
  },
  {
    "movie_id": 758866,
    "inclusion_sum": 1.4879809599999998,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4879809599999998
  },
  {
    "movie_id": 426135,
    "inclusion_sum": 1.4875725000000002,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4875725000000002
  },
  {
    "movie_id": 54357,
    "inclusion_sum": 1.4871727,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4871727
  },
  {
    "movie_id": 112999,
    "inclusion_sum": 1.4857765,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4857765
  },
  {
    "movie_id": 4995,
    "inclusion_sum": 1.48568634,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.48568634
  },
  {
    "movie_id": 490078,
    "inclusion_sum": 1.4854912,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4854912
  },
  {
    "movie_id": 16619,
    "inclusion_sum": 1.4847539,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4847539
  },
  {
    "movie_id": 519010,
    "inclusion_sum": 1.4845350599999998,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4845350599999998
  },
  {
    "movie_id": 367580,
    "inclusion_sum": 1.4838539,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4838539
  },
  {
    "movie_id": 412098,
    "inclusion_sum": 1.48293,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.48293
  },
  {
    "movie_id": 576928,
    "inclusion_sum": 1.4828575000000002,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4828575000000002
  },
  {
    "movie_id": 483128,
    "inclusion_sum": 1.4821906999999999,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4821906999999999
  },
  {
    "movie_id": 430273,
    "inclusion_sum": 1.4814743,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4814743
  },
  {
    "movie_id": 806067,
    "inclusion_sum": 1.4810921000000001,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4810921000000001
  },
  {
    "movie_id": 1477887,
    "inclusion_sum": 1.48044815,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.48044815
  },
  {
    "movie_id": 678,
    "inclusion_sum": 1.47956726,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.47956726
  },
  {
    "movie_id": 587030,
    "inclusion_sum": 1.4785202000000002,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4785202000000002
  },
  {
    "movie_id": 91721,
    "inclusion_sum": 1.4778436,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4778436
  },
  {
    "movie_id": 39013,
    "inclusion_sum": 1.4776585400000002,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4776585400000002
  },
  {
    "movie_id": 79897,
    "inclusion_sum": 1.47725525,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.47725525
  },
  {
    "movie_id": 882796,
    "inclusion_sum": 1.4766311600000002,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4766311600000002
  },
  {
    "movie_id": 49980,
    "inclusion_sum": 1.4760129000000002,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4760129000000002
  },
  {
    "movie_id": 77,
    "inclusion_sum": 1.4758839400000001,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.4758839400000001
  },
  {
    "movie_id": 1163194,
    "inclusion_sum": 1.475,
    "downrank_sum": 0.0,
    "preference_contribution": 0.0,
    "implicit_prior_contribution": 0.0,
    "final_score": 1.475
  }
]
```

## Step 4 — Summary

### Filters
- (none)

### Traits
- best quality or highly acclaimed

_used_fallback: preferences_as_candidates_

### query_intent_summary
The user is seeking highly-regarded dramatic acting turns from performers who typically work in comedy, explicitly requesting the 'best' examples.

_implicit priors: quality=off  popularity=off_

### Top 100 Results

| # | final | filter | pref | down | impl | title (year) | tmdb_id |
|---|-------|--------|------|------|------|--------------|---------|
| 1 | 1.6005 | 1.6005 | 0.0000 | -0.0000 | 0.0000 | Thaniyavarthanam (1987) | 307252 |
| 2 | 1.5865 | 1.5865 | 0.0000 | -0.0000 | 0.0000 | Parasite (2019) | 496243 |
| 3 | 1.5782 | 1.5782 | 0.0000 | -0.0000 | 0.0000 | Children of Paradise (1945) | 2457 |
| 4 | 1.5723 | 1.5723 | 0.0000 | -0.0000 | 0.0000 | The Curse of Modigliani (2025) | 1438424 |
| 5 | 1.5655 | 1.5655 | 0.0000 | -0.0000 | 0.0000 | Man Goes On Rant (2024) | 1240009 |
| 6 | 1.5643 | 1.5643 | 0.0000 | -0.0000 | 0.0000 | A Streetcar Named Desire (1951) | 702 |
| 7 | 1.5637 | 1.5637 | 0.0000 | -0.0000 | 0.0000 | A (1998) | 210763 |
| 8 | 1.5599 | 1.5599 | 0.0000 | -0.0000 | 0.0000 | The Yellow Tie (2025) | 1245119 |
| 9 | 1.5539 | 1.5539 | 0.0000 | -0.0000 | 0.0000 | The Unnamed (2016) | 418204 |
| 10 | 1.5512 | 1.5512 | 0.0000 | -0.0000 | 0.0000 | Eradu Kanasu (1974) | 305237 |
| 11 | 1.5505 | 1.5505 | 0.0000 | -0.0000 | 0.0000 | Secrets & Lies (1996) | 11159 |
| 12 | 1.5492 | 1.5492 | 0.0000 | -0.0000 | 0.0000 | Divya Drushti (2025) | 1600694 |
| 13 | 1.5452 | 1.5452 | 0.0000 | -0.0000 | 0.0000 | Stockton to Table Rock (2025) | 1482871 |
| 14 | 1.5449 | 1.5449 | 0.0000 | -0.0000 | 0.0000 | Spotlight (2015) | 314365 |
| 15 | 1.5439 | 1.5439 | 0.0000 | -0.0000 | 0.0000 | The Man Who Shot Liberty Valance (1962) | 11697 |
| 16 | 1.5435 | 1.5435 | 0.0000 | -0.0000 | 0.0000 | Anbe Sivam (2003) | 26910 |
| 17 | 1.5424 | 1.5424 | 0.0000 | -0.0000 | 0.0000 | Marriage Story (2019) | 492188 |
| 18 | 1.5420 | 1.5420 | 0.0000 | -0.0000 | 0.0000 | Making the Day (2021) | 957430 |
| 19 | 1.5366 | 1.5366 | 0.0000 | -0.0000 | 0.0000 | Natsamrat (2016) | 378227 |
| 20 | 1.5343 | 1.5343 | 0.0000 | -0.0000 | 0.0000 | The King's Speech (2010) | 45269 |
| 21 | 1.5297 | 1.5297 | 0.0000 | -0.0000 | 0.0000 | C/o Kancharapalem (2018) | 544795 |
| 22 | 1.5285 | 1.5285 | 0.0000 | -0.0000 | 0.0000 | The Lord of All Future Space & Time (2025) | 1493253 |
| 23 | 1.5282 | 1.5282 | 0.0000 | -0.0000 | 0.0000 | Rudraveena (1988) | 66342 |
| 24 | 1.5279 | 1.5279 | 0.0000 | -0.0000 | 0.0000 | #Home (2021) | 859987 |
| 25 | 1.5272 | 1.5272 | 0.0000 | -0.0000 | 0.0000 | Moondram Pirai (1982) | 66389 |
| 26 | 1.5269 | 1.5269 | 0.0000 | -0.0000 | 0.0000 | The Fifth Seal (1976) | 77660 |
| 27 | 1.5265 | 1.5265 | 0.0000 | -0.0000 | 0.0000 | In The Orchard (2019) | 700394 |
| 28 | 1.5253 | 1.5253 | 0.0000 | -0.0000 | 0.0000 | The Hustler (1961) | 990 |
| 29 | 1.5244 | 1.5244 | 0.0000 | -0.0000 | 0.0000 | Harakiri (1962) | 14537 |
| 30 | 1.5223 | 1.5223 | 0.0000 | -0.0000 | 0.0000 | The Eunuch and the Flute Player (2019) | 522986 |
| 31 | 1.5211 | 1.5211 | 0.0000 | -0.0000 | 0.0000 | Thevar Magan (1992) | 66362 |
| 32 | 1.5209 | 1.5209 | 0.0000 | -0.0000 | 0.0000 | The Worst Person in the World (2021) | 660120 |
| 33 | 1.5176 | 1.5176 | 0.0000 | -0.0000 | 0.0000 | The Best of Youth (2003) | 11659 |
| 34 | 1.5172 | 1.5172 | 0.0000 | -0.0000 | 0.0000 | Sadayam (1992) | 280277 |
| 35 | 1.5166 | 1.5166 | 0.0000 | -0.0000 | 0.0000 | Walampoori: Seven and a Half Dreams (2025) | 1390165 |
| 36 | 1.5164 | 1.5164 | 0.0000 | -0.0000 | 0.0000 | Sling Blade (1996) | 12498 |
| 37 | 1.5159 | 1.5159 | 0.0000 | -0.0000 | 0.0000 | Pariyerum Perumal (2018) | 462718 |
| 38 | 1.5153 | 1.5153 | 0.0000 | -0.0000 | 0.0000 | Juice (2017) | 496056 |
| 39 | 1.5132 | 1.5132 | 0.0000 | -0.0000 | 0.0000 | A Woman Under the Influence (1974) | 29845 |
| 40 | 1.5115 | 1.5115 | 0.0000 | -0.0000 | 0.0000 | Eternal Sunshine of the Spotless Mind (2004) | 38 |
| 41 | 1.5089 | 1.5089 | 0.0000 | -0.0000 | 0.0000 | Versatile (2024) | 1340209 |
| 42 | 1.5087 | 1.5087 | 0.0000 | -0.0000 | 0.0000 | In the Bedroom (2001) | 1999 |
| 43 | 1.5078 | 1.5078 | 0.0000 | -0.0000 | 0.0000 | Soorarai Pottru (2020) | 595228 |
| 44 | 1.5078 | 1.5078 | 0.0000 | -0.0000 | 0.0000 | Tattoo (1991) | 196917 |
| 45 | 1.5078 | 1.5078 | 0.0000 | -0.0000 | 0.0000 | A Raisin in the Sun (1961) | 29478 |
| 46 | 1.5072 | 1.5072 | 0.0000 | -0.0000 | 0.0000 | Summer Rains (1975) | 449632 |
| 47 | 1.5070 | 1.5070 | 0.0000 | -0.0000 | 0.0000 | The Quiet Girl (2022) | 916405 |
| 48 | 1.5064 | 1.5064 | 0.0000 | -0.0000 | 0.0000 | Quiz Show (1994) | 11450 |
| 49 | 1.5057 | 1.5057 | 0.0000 | -0.0000 | 0.0000 | Mahanati (2018) | 459713 |
| 50 | 1.5051 | 1.5051 | 0.0000 | -0.0000 | 0.0000 | A Prophet (2009) | 21575 |
| 51 | 1.5032 | 1.5032 | 0.0000 | -0.0000 | 0.0000 | Pithamagan (2003) | 69638 |
| 52 | 1.5023 | 1.5023 | 0.0000 | -0.0000 | 0.0000 | Kuruthipunal (1995) | 66351 |
| 53 | 1.5019 | 1.5019 | 0.0000 | -0.0000 | 0.0000 | In Cold Blood (1967) | 18900 |
| 54 | 1.5012 | 1.5012 | 0.0000 | -0.0000 | 0.0000 | Kattradhu Thamizh (2007) | 69550 |
| 55 | 1.5011 | 1.5011 | 0.0000 | -0.0000 | 0.0000 | Rangasthalam (2018) | 461126 |
| 56 | 1.5006 | 1.5006 | 0.0000 | -0.0000 | 0.0000 | The Father (2020) | 600354 |
| 57 | 1.4992 | 1.4992 | 0.0000 | -0.0000 | 0.0000 | Investigation of a Citizen Above Suspicion (1970) | 26451 |
| 58 | 1.4987 | 1.4987 | 0.0000 | -0.0000 | 0.0000 | Uncut Gems (2019) | 473033 |
| 59 | 1.4980 | 1.4980 | 0.0000 | -0.0000 | 0.0000 | The Bad and the Beautiful (1952) | 32499 |
| 60 | 1.4973 | 1.4973 | 0.0000 | -0.0000 | 0.0000 | This Is Your Song (2023) | 1190085 |
| 61 | 1.4970 | 1.4970 | 0.0000 | -0.0000 | 0.0000 | Kumbalangi Nights (2019) | 575351 |
| 62 | 1.4968 | 1.4968 | 0.0000 | -0.0000 | 0.0000 | Drishyam (2013) | 244049 |
| 63 | 1.4950 | 1.4950 | 0.0000 | -0.0000 | 0.0000 | 45 Years (2015) | 311291 |
| 64 | 1.4938 | 1.4938 | 0.0000 | -0.0000 | 0.0000 | This Is Not a Burial, It's a Resurrection (2020) | 618208 |
| 65 | 1.4937 | 1.4937 | 0.0000 | -0.0000 | 0.0000 | Iruvar (1997) | 49032 |
| 66 | 1.4931 | 1.4931 | 0.0000 | -0.0000 | 0.0000 | Ghich Pich (2025) | 1447109 |
| 67 | 1.4929 | 1.4929 | 0.0000 | -0.0000 | 0.0000 | The Human Condition I: No Greater Love (1959) | 31217 |
| 68 | 1.4918 | 1.4918 | 0.0000 | -0.0000 | 0.0000 | The Heiress (1949) | 28571 |
| 69 | 1.4910 | 1.4910 | 0.0000 | -0.0000 | 0.0000 | Thalapathi (1991) | 29811 |
| 70 | 1.4903 | 1.4903 | 0.0000 | -0.0000 | 0.0000 | Winter Sleep (2014) | 265169 |
| 71 | 1.4898 | 1.4898 | 0.0000 | -0.0000 | 0.0000 | Never Rarely Sometimes Always (2020) | 595671 |
| 72 | 1.4897 | 1.4897 | 0.0000 | -0.0000 | 0.0000 | System Crasher (2019) | 567410 |
| 73 | 1.4896 | 1.4896 | 0.0000 | -0.0000 | 0.0000 | Of Mice and Men (1939) | 51802 |
| 74 | 1.4884 | 1.4884 | 0.0000 | -0.0000 | 0.0000 | Woman in the Dunes (1964) | 16672 |
| 75 | 1.4880 | 1.4880 | 0.0000 | -0.0000 | 0.0000 | The Very Same Munchhausen (1979) | 27937 |
| 76 | 1.4880 | 1.4880 | 0.0000 | -0.0000 | 0.0000 | Ramanaa (2002) | 78955 |
| 77 | 1.4880 | 1.4880 | 0.0000 | -0.0000 | 0.0000 | Drive My Car (2021) | 758866 |
| 78 | 1.4876 | 1.4876 | 0.0000 | -0.0000 | 0.0000 | Every Brilliant Thing (2016) | 426135 |
| 79 | 1.4872 | 1.4872 | 0.0000 | -0.0000 | 0.0000 | Sadma (1983) | 54357 |
| 80 | 1.4858 | 1.4858 | 0.0000 | -0.0000 | 0.0000 | The Hero (1966) | 112999 |
| 81 | 1.4857 | 1.4857 | 0.0000 | -0.0000 | 0.0000 | Boogie Nights (1997) | 4995 |
| 82 | 1.4855 | 1.4855 | 0.0000 | -0.0000 | 0.0000 | Ee.Ma.Yau. (2018) | 490078 |
| 83 | 1.4848 | 1.4848 | 0.0000 | -0.0000 | 0.0000 | Ordinary People (1980) | 16619 |
| 84 | 1.4845 | 1.4845 | 0.0000 | -0.0000 | 0.0000 | Pain and Glory (2019) | 519010 |
| 85 | 1.4839 | 1.4839 | 0.0000 | -0.0000 | 0.0000 | Beloved (1985) | 367580 |
| 86 | 1.4829 | 1.4829 | 0.0000 | -0.0000 | 0.0000 | Emilie Muller (1993) | 412098 |
| 87 | 1.4829 | 1.4829 | 0.0000 | -0.0000 | 0.0000 | Asuran (2019) | 576928 |
| 88 | 1.4822 | 1.4822 | 0.0000 | -0.0000 | 0.0000 | The Definition of Insanity (2005) | 483128 |
| 89 | 1.4815 | 1.4815 | 0.0000 | -0.0000 | 0.0000 | Rama Rama Re (2016) | 430273 |
| 90 | 1.4811 | 1.4811 | 0.0000 | -0.0000 | 0.0000 | Mandela (2021) | 806067 |
| 91 | 1.4804 | 1.4804 | 0.0000 | -0.0000 | 0.0000 | House of Abraham (2025) | 1477887 |
| 92 | 1.4796 | 1.4796 | 0.0000 | -0.0000 | 0.0000 | Out of the Past (1947) | 678 |
| 93 | 1.4785 | 1.4785 | 0.0000 | -0.0000 | 0.0000 | Kaithi (2019) | 587030 |
| 94 | 1.4778 | 1.4778 | 0.0000 | -0.0000 | 0.0000 | Nothing But a Man (1964) | 91721 |
| 95 | 1.4777 | 1.4777 | 0.0000 | -0.0000 | 0.0000 | Winter's Bone (2010) | 39013 |
| 96 | 1.4773 | 1.4773 | 0.0000 | -0.0000 | 0.0000 | Pranchiyettan & The Saint (2010) | 79897 |
| 97 | 1.4766 | 1.4766 | 0.0000 | -0.0000 | 0.0000 | Pitruroon (2013) | 882796 |
| 98 | 1.4760 | 1.4760 | 0.0000 | -0.0000 | 0.0000 | Ladybird Ladybird (1994) | 49980 |
| 99 | 1.4759 | 1.4759 | 0.0000 | -0.0000 | 0.0000 | Memento (2000) | 77 |
| 100 | 1.4750 | 1.4750 | 0.0000 | -0.0000 | 0.0000 | Lost Ladies (2024) | 1163194 |

_total elapsed: 10.16s_
