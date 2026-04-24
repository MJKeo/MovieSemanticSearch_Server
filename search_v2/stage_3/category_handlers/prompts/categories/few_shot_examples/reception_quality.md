# Examples

These calibrate the per-endpoint breakdown for reception-quality asks: SEMANTIC reception carries the qualitative prose, METADATA reception anchors a numeric prior, they usually compose, "underrated" is the case where prose leads and the scalar softens, and the empty combination is the correct response when the atom really belongs to Cat 8 (awards) or Cat 28 (named list).

**Example: "underrated" — SEMANTIC leads, METADATA stays silent because the direction is ambiguous**

```xml
<raw_query>underrated horror movies</raw_query>
<overall_query_intention_exploration>The user wants horror films whose quality is higher than their popularity or reception score would suggest — underappreciated gems within the genre. The horror slice routes to a separate category; this entry is the underrated-reception framing.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films whose quality exceeds their popular recognition — underrated in reception.</captured_meaning>
  <category_name>Reception quality + superlative</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Films widely considered underrated relative to their quality.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>underrated horror movies</query_text>
  <description>Horror films that are underrated for their quality.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Capture the underrated reception shape — films praised by the minority who noticed them, with reception language describing the quality-versus-recognition gap.",
      "endpoint_coverage": "SEMANTIC's reception space is the purpose-built channel: reception_summary can hold prose about the film being underrated or overlooked, and praised_qualities can name the specific axes the minority highlighted. METADATA's reception column is a single direction (well_received / poorly_received) — neither fits cleanly since 'underrated' literally means the numeric score is lower than the quality deserves."
    },
    {
      "aspect_description": "Anchor the numeric reception baseline.",
      "endpoint_coverage": "METADATA's reception column cannot honestly encode underrated without fighting itself: well_received contradicts the 'lower than deserved' framing, and poorly_received is the opposite of what the user wants. No-fire is the correct read — the Semantic prose carries the signal alone."
    }
  ],
  "overall_endpoint_fits": "SEMANTIC reception carries the full requirement through reception_summary describing the underrated shape and praised_qualities naming the axes the minority praised. METADATA does not fire: 'underrated' is a quality-above-score framing the binary well_received / poorly_received column cannot express without misrepresenting the ask. The horror axis is Cat 11's slice and is not emitted here.",
  "per_endpoint_breakdown": {
    "semantic": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "trait",
        "parameters": {
          "qualifier_inventory": "reception shape is underrated — quality recognized by a minority, broader audience awareness lower than the work deserves; praise axes unspecified but implicit in the framing.",
          "space_queries": [
            {
              "carries_qualifiers": "reception carries the underrated framing directly — reception_summary names the quality-versus-recognition gap, praised_qualities lists the axes the minority who noticed the film consistently highlighted.",
              "space": "reception",
              "weight": "central",
              "content": {
                "reception_summary": "widely considered underrated and overlooked; a cult minority argues the film is better than its broader reception suggests.",
                "praised_qualities": ["underappreciated craft", "overlooked by mainstream audiences", "cult minority recognition"],
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
      "should_run_endpoint": false,
      "endpoint_parameters": null
    }
  }
}
```

**Example: "cult classics" — SEMANTIC primary, METADATA silent because the framing is reception-shape not reception-direction**

```xml
<raw_query>cult classics</raw_query>
<overall_query_intention_exploration>The user wants cult classics — films with a devoted fan base that keep their relevance despite being outside the mainstream canon. A reception-shape framing, not a high-or-low quality direction.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films with cult classic reception.</captured_meaning>
  <category_name>Reception quality + superlative</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Films widely recognized as cult classics.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>cult classics</query_text>
  <description>Films with a cult classic reception shape.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Capture the cult-classic reception shape — devoted fan base, lasting relevance outside the mainstream canon.",
      "endpoint_coverage": "SEMANTIC reception is the right home: reception_summary can hold cult-classic language directly, and praised_qualities can name the axes cult audiences consistently point to (rewatch value, distinctiveness, quotability). METADATA's reception scalar does not encode 'cult' — cult films span the quality spectrum and the axis here is shape, not direction."
    }
  ],
  "overall_endpoint_fits": "SEMANTIC reception carries the full requirement. METADATA does not fire: cult reception is a shape, not a position on the well_received / poorly_received axis — cult films land all over that scale, and pinning a direction would narrow the pool away from half the canonical cult films.",
  "per_endpoint_breakdown": {
    "semantic": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "trait",
        "parameters": {
          "qualifier_inventory": "reception shape is cult-classic — devoted fan base, lasting rewatch relevance, canonical status within a subculture rather than the mainstream.",
          "space_queries": [
            {
              "carries_qualifiers": "reception carries the cult-classic shape directly through reception_summary and praised_qualities that name the axes cult reception emphasizes.",
              "space": "reception",
              "weight": "central",
              "content": {
                "reception_summary": "developed a lasting cult following; widely regarded as a cult classic with devoted fans who champion it outside the mainstream canon.",
                "praised_qualities": ["cult following", "rewatch value", "distinctive voice", "lasting influence on fans"],
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
      "should_run_endpoint": false,
      "endpoint_parameters": null
    }
  }
}
```

**Example: superlative slice of "best horror of the 80s" — SEMANTIC + METADATA both fire**

```xml
<raw_query>best horror of the 80s</raw_query>
<overall_query_intention_exploration>The user wants the top-regarded horror films from the 1980s. The full query decomposes into the horror axis (a separate category's slice), the 1980s era (structured metadata), and the acclaim/superlative framing that reaches this handler.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films widely regarded as the best within their slice.</captured_meaning>
  <category_name>Reception quality + superlative</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Films widely considered among the best in their category.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>best horror of the 80s</query_text>
  <description>Top-regarded horror films from the 1980s.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Capture the 'best of the slice' superlative framing — widely considered top-tier, high-water-mark reception.",
      "endpoint_coverage": "SEMANTIC reception carries the acclaim prose: reception_summary can name the high-water-mark framing, praised_qualities can list the axes top-tier reception emphasizes. METADATA's reception column cannot express the superlative specifically but anchors the numeric baseline that 'best' implies."
    },
    {
      "aspect_description": "Anchor the numeric reception prior that 'best' implies.",
      "endpoint_coverage": "METADATA's reception column carries well_received as an additive numeric lift. 'Best' as a superlative implies the films should sit high on the reception scale broadly — unlike 'underrated' (where direction is ambiguous), 'best' is a clean well_received signal."
    }
  ],
  "overall_endpoint_fits": "SEMANTIC reception carries the qualitative superlative framing through reception_summary and praised_qualities that name high-water-mark acclaim. METADATA reception contributes the numeric well_received prior additively — 'best' cleanly implies broad critical standing. The two compose: prose provides the acclaim shape, scalar lifts the numeric baseline. The horror and 1980s axes are other categories' slices and are not emitted here.",
  "per_endpoint_breakdown": {
    "semantic": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "trait",
        "parameters": {
          "qualifier_inventory": "reception framing is top-tier within a slice — high-water-mark acclaim, widely considered the best of its kind.",
          "space_queries": [
            {
              "carries_qualifiers": "reception carries the 'best-of' acclaim framing — reception_summary names the high-water-mark language and praised_qualities lists axes top-tier reception typically cites.",
              "space": "reception",
              "weight": "central",
              "content": {
                "reception_summary": "widely regarded as one of the defining works of its slice; a high-water-mark for the category and frequently cited among the very best.",
                "praised_qualities": ["defining work of its era", "genre-defining", "widely acclaimed as top-tier", "canonical standing"],
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
          "constraint_phrases": ["best"],
          "target_attribute": "reception",
          "value_intent_label": "broadly well-received",
          "reception": "well_received"
        },
        "polarity": "positive"
      }
    }
  }
}
```

**Example: "stacked A-list cast" — SEMANTIC prose for cast-popularity acclaim, METADATA silent**

```xml
<raw_query>movies with a stacked A-list cast</raw_query>
<overall_query_intention_exploration>The user wants films whose cast is unusually loaded with major stars — the cast itself being the draw. This is cast-popularity acclaim, a reception-adjacent framing the reception prose channel can carry.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films with an unusually star-heavy, top-tier ensemble cast.</captured_meaning>
  <category_name>Reception quality + superlative</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Films with a stacked A-list ensemble cast.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies with a stacked A-list cast</query_text>
  <description>Films whose cast is an A-list ensemble.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Capture the cast-popularity framing — films noted for having an exceptionally star-heavy, A-list ensemble.",
      "endpoint_coverage": "SEMANTIC reception carries this through reception_summary language about the ensemble and praised_qualities naming cast as a draw. This is a reception-adjacent axis — critics and audiences note the cast stature as a specific praised quality. METADATA's reception column scores overall critical-audience reception, not cast recognition; it cannot express cast-popularity specifically."
    }
  ],
  "overall_endpoint_fits": "SEMANTIC reception carries the requirement through reception_summary prose about a loaded ensemble and praised_qualities listing cast stature explicitly. METADATA does not fire: the framing is about cast popularity, not overall reception direction — the films could sit anywhere on the well_received scale and still satisfy the ask.",
  "per_endpoint_breakdown": {
    "semantic": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "trait",
        "parameters": {
          "qualifier_inventory": "reception angle is cast-popularity acclaim — ensemble stature itself as a praised quality, star-studded casts noted by critics and audiences.",
          "space_queries": [
            {
              "carries_qualifiers": "reception carries cast-popularity framing — reception_summary can name the star-studded ensemble as a draw, praised_qualities lists cast stature directly as an axis reception called out.",
              "space": "reception",
              "weight": "central",
              "content": {
                "reception_summary": "widely noted for its stacked A-list ensemble; the cast itself drew significant attention as a defining feature.",
                "praised_qualities": ["stacked A-list cast", "star-studded ensemble", "ensemble performances", "cast as a draw"],
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
      "should_run_endpoint": false,
      "endpoint_parameters": null
    }
  }
}
```

**Example: no-fire on award-anchored framing — Cat 8's territory**

```xml
<raw_query>Oscar-winning movies</raw_query>
<overall_query_intention_exploration>The user wants films that won an Academy Award — a structured-recognition ask anchored to a specific ceremony.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films that won an Academy Award.</captured_meaning>
  <category_name>Reception quality + superlative</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Films that won an Oscar.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>Oscar-winning movies</query_text>
  <description>Films that have won an Academy Award.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the reception-quality axis the user named.",
      "endpoint_coverage": "Neither endpoint in this category's set fits. The framing is anchored to formal award recognition — a structured award-ledger ask that Cat 8's award endpoint handles with per-ceremony precision. SEMANTIC's reception space could loosely encode 'award-winning' as a praised quality, but doing so duplicates the award channel while losing its ceremony- and category-level fidelity. METADATA's reception scalar does not represent awards; well_received is correlated but not equivalent, and firing it here would misrepresent the user's ask as a general-quality preference."
    }
  ],
  "overall_endpoint_fits": "The atom routed here is an award-anchored framing — Cat 8's territory — not a reception-quality judgment. Firing SEMANTIC reception with 'award-winning' praise text would duplicate the award endpoint with weaker precision; firing METADATA reception would silently substitute general quality for a specific recognition record. The empty combination is the correct response.",
  "per_endpoint_breakdown": {
    "semantic": {
      "should_run_endpoint": false,
      "endpoint_parameters": null
    },
    "metadata": {
      "should_run_endpoint": false,
      "endpoint_parameters": null
    }
  }
}
```

**Example: no-fire on named-list membership — Cat 28's territory**

```xml
<raw_query>Criterion Collection picks</raw_query>
<overall_query_intention_exploration>The user wants films in the Criterion Collection — membership in a specific curated named list, not a general reception-quality framing.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films in the Criterion Collection.</captured_meaning>
  <category_name>Reception quality + superlative</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Films included in the Criterion Collection.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>Criterion Collection picks</query_text>
  <description>Films included in the Criterion Collection.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the reception-quality axis the user named.",
      "endpoint_coverage": "Neither endpoint fits. The user named a specific curated list (Criterion), which is Cat 28's territory — that category uses a list-citation Semantic strategy purpose-built for decoding what a named list implies. This category handles 'classic' as generic canonical stature, not membership in a specific named list. SEMANTIC reception would invent list-citation prose without Cat 28's interpretation framing; METADATA reception would collapse the named-list ask into a generic well_received lift and broaden the pool past the specific canon the user asked for."
    }
  ],
  "overall_endpoint_fits": "The atom routed here is a named-list membership ask — Cat 28's territory — not a reception-quality shape. Firing either endpoint would misrepresent list membership as general reception quality and lose the list-specific signal Cat 28's handler is built to carry. The empty combination is the correct response.",
  "per_endpoint_breakdown": {
    "semantic": {
      "should_run_endpoint": false,
      "endpoint_parameters": null
    },
    "metadata": {
      "should_run_endpoint": false,
      "endpoint_parameters": null
    }
  }
}
```
