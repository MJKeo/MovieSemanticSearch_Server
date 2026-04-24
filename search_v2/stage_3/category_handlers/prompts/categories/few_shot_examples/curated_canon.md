# Examples

These calibrate how to decode named-list semantics into Semantic prose and how to weight the METADATA reception lift: firing both with a heavy reception prior when the list broadly skews high-reception (AFI, 1001 Movies), firing Semantic alone or with a lighter reception read when the list aesthetic tolerates lower-scored arthouse picks (Criterion, Sight & Sound), and no-firing when the named "list" does not map to any real canon.

**Example: Criterion Collection — Semantic carries the arthouse-curation aesthetic; METADATA silent because Criterion tolerates lower-scored arthouse picks**

```xml
<raw_query>Criterion Collection films</raw_query>
<overall_query_intention_exploration>The user wants films in the Criterion Collection — membership in a specific curated list associated with arthouse, international, and restored-classics sensibilities.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films included in the Criterion Collection.</captured_meaning>
  <category_name>Curated canon / named list</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Films in the Criterion Collection.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>Criterion Collection films</query_text>
  <description>Films curated by and included in the Criterion Collection.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Decode what Criterion implies and capture it in list-citation prose — curated arthouse, international cinema, auteur-driven, restored classics, craft-first sensibility.",
      "endpoint_coverage": "SEMANTIC reception is the primary channel: reception_summary can name Criterion inclusion and the arthouse-canon aesthetic, and praised_qualities can list the axes Criterion picks are consistently praised for. plot_analysis supports at the identity level — elevator_pitch can frame the film as an auteur-driven or world-cinema masterwork, which is how canonical Criterion picks are described in reviews."
    },
    {
      "aspect_description": "Anchor the reception numeric prior.",
      "endpoint_coverage": "METADATA reception does not fire cleanly. Criterion deliberately includes lower-scored arthouse and international films whose generic reception scalar lands below popular crowd-pleasers — a well_received lift would drop legitimate list members. The list aesthetic is prestige-of-curation, not broad mainstream acclaim."
    }
  ],
  "overall_endpoint_fits": "Criterion Collection implies curated arthouse / international / auteur-driven / restored-classics sensibility — prestige defined by editorial curation, not mainstream reception score. SEMANTIC carries this through reception prose that names Criterion inclusion and the arthouse-canon axes, with plot_analysis supporting via canonical-identity framing. METADATA does not fire: the generic reception scalar would penalize the lower-scored arthouse picks the list is specifically known for, misrepresenting the ask.",
  "per_endpoint_breakdown": {
    "semantic": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "trait",
        "parameters": {
          "qualifier_inventory": "named list is Criterion Collection — curated arthouse, international cinema, auteur-driven, restored classics, craft and cultural-preservation sensibility; canonical stature at the identity level.",
          "space_queries": [
            {
              "carries_qualifiers": "reception carries the Criterion list-citation language directly — reception_summary names the inclusion and the arthouse-canon register, praised_qualities lists the axes Criterion picks are consistently celebrated for.",
              "space": "reception",
              "weight": "central",
              "content": {
                "reception_summary": "included in the Criterion Collection; widely regarded within the arthouse and world-cinema canon, valued for craft, authorial vision, and cultural preservation.",
                "praised_qualities": ["Criterion pick", "arthouse canon", "auteur-driven", "world cinema", "restored classic", "formal craft"],
                "criticized_qualities": []
              }
            },
            {
              "carries_qualifiers": "plot_analysis carries the canonical-stature identity framing — elevator_pitch positions the film as an auteur-driven masterwork of the kind Criterion canonizes.",
              "space": "plot_analysis",
              "weight": "supporting",
              "content": {
                "elevator_pitch": "an auteur-driven work of international or arthouse cinema treated as a canonical masterwork by the curated-canon community.",
                "genre_signatures": ["arthouse", "world cinema", "auteur cinema"],
                "thematic_concepts": ["authorial vision", "formal experimentation", "cultural significance"]
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

**Example: AFI Top 100 — Semantic + METADATA both fire; AFI skews to high-reception American mainstream canon**

```xml
<raw_query>AFI Top 100 movies</raw_query>
<overall_query_intention_exploration>The user wants films on the American Film Institute's Top 100 list — the American mainstream canon of prestige and classic-era films broadly familiar to US audiences.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films on the AFI Top 100 list.</captured_meaning>
  <category_name>Curated canon / named list</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Films included on the AFI Top 100.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>AFI Top 100 movies</query_text>
  <description>Films on the American Film Institute's Top 100 list.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Decode what the AFI Top 100 implies — American mainstream canon, prestige studio-era and landmark films broadly recognized by US audiences.",
      "endpoint_coverage": "SEMANTIC reception carries the AFI-canonization language: reception_summary can cite the list and name the American-mainstream-canon aesthetic, and praised_qualities can list the axes AFI picks are consistently cited for. plot_analysis supports at the identity level via canonical-masterwork framings."
    },
    {
      "aspect_description": "Anchor the numeric reception prior.",
      "endpoint_coverage": "METADATA reception fires as an additive lift. AFI selections skew strongly toward broadly well-regarded mainstream films — the generic well_received signal aligns with the list's shape without fighting it the way it would for arthouse-leaning canons."
    }
  ],
  "overall_endpoint_fits": "AFI Top 100 implies the American mainstream canon — prestige, landmark studio-era and broadly-recognized films whose reception is both high and broadly distributed. SEMANTIC carries the list-citation prose and canonical-stature framing; METADATA adds a well_received lift that matches the list's broadly high-reception shape. Both endpoints compose naturally here, unlike for arthouse-skewed canons where the reception scalar misrepresents the aesthetic.",
  "per_endpoint_breakdown": {
    "semantic": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "trait",
        "parameters": {
          "qualifier_inventory": "named list is AFI Top 100 — American mainstream canon, prestige and landmark films broadly recognized by US audiences, high reception with wide distribution.",
          "space_queries": [
            {
              "carries_qualifiers": "reception carries the AFI-honored list-citation language — reception_summary names the inclusion and the American-mainstream-canon register, praised_qualities lists the axes AFI picks are consistently cited for.",
              "space": "reception",
              "weight": "central",
              "content": {
                "reception_summary": "honored on the American Film Institute's Top 100 list; widely considered a landmark of American cinema and a staple of the mainstream prestige canon.",
                "praised_qualities": ["AFI-honored", "American canon", "landmark film", "widely acclaimed", "culturally significant", "prestige cinema"],
                "criticized_qualities": []
              }
            },
            {
              "carries_qualifiers": "plot_analysis carries the canonical-masterwork identity framing AFI picks commonly carry at the review level.",
              "space": "plot_analysis",
              "weight": "supporting",
              "content": {
                "elevator_pitch": "a landmark work of American cinema routinely cited among the defining films of its era.",
                "genre_signatures": ["American cinema", "prestige drama"],
                "thematic_concepts": ["cultural significance", "landmark work"]
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
          "constraint_phrases": ["AFI Top 100"],
          "target_attribute": "reception",
          "value_intent_label": "broadly well-received mainstream canon",
          "reception": "well_received"
        },
        "polarity": "positive"
      }
    }
  }
}
```

**Example: Sight & Sound greatest films — Semantic carries critic-canon framing; METADATA silent because the list tolerates formalist picks that score lower than popular canons**

```xml
<raw_query>movies from Sight and Sound's greatest films list</raw_query>
<overall_query_intention_exploration>The user wants films on the Sight & Sound greatest-films polls — the critic-canonical list curated by BFI's Sight & Sound magazine, leaning arthouse, international, and formally ambitious.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films on the Sight & Sound greatest-films list.</captured_meaning>
  <category_name>Curated canon / named list</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Films on the Sight and Sound greatest-films poll.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies from Sight and Sound's greatest films list</query_text>
  <description>Films ranked on the Sight and Sound greatest-films poll.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Decode what Sight & Sound implies — critic-canonical, arthouse- and internationally-leaning, formalist and slow-cinema tolerant, less commercial than popular canons.",
      "endpoint_coverage": "SEMANTIC reception is the primary channel: reception_summary can cite the poll and name the critic-canon register, and praised_qualities can list the axes critic-canonical films are consistently cited for (formal ambition, authorial vision, international significance). plot_analysis supports at the identity level for films carried into the critic canon on auteurist and formalist grounds."
    },
    {
      "aspect_description": "Anchor the numeric reception prior.",
      "endpoint_coverage": "METADATA reception does not fire. Sight & Sound's top picks routinely include formalist, slow-cinema, and early-era films whose generic reception scalars can sit below popular mainstream films — a well_received lift would lower the score of legitimate list members. The list's shape is critic-canon prestige, not broad-audience acclaim."
    }
  ],
  "overall_endpoint_fits": "Sight & Sound implies the critic-canon — authorial, formalist, arthouse- and international-leaning cinema with less dependence on mainstream reception scores. SEMANTIC carries this through reception prose that cites the poll and names the critic-canonical axes, with plot_analysis supporting via canonical-identity framing. METADATA does not fire: the generic reception scalar would penalize the formalist and early-cinema picks the list is known for, misrepresenting the ask.",
  "per_endpoint_breakdown": {
    "semantic": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "trait",
        "parameters": {
          "qualifier_inventory": "named list is Sight & Sound greatest-films poll — critic-canonical, arthouse- and internationally-leaning, formally ambitious, tolerant of slow cinema and early-era works less served by popular reception scores.",
          "space_queries": [
            {
              "carries_qualifiers": "reception carries the Sight & Sound list-citation language and critic-canon register — reception_summary names the poll inclusion, praised_qualities lists the axes critic-canonical films are consistently cited for.",
              "space": "reception",
              "weight": "central",
              "content": {
                "reception_summary": "cited on the Sight and Sound greatest-films poll; widely regarded in critic-canonical circles as a defining work of world cinema, valued for formal ambition and authorial vision.",
                "praised_qualities": ["Sight and Sound poll", "critic canon", "formal ambition", "authorial vision", "world cinema", "arthouse prestige"],
                "criticized_qualities": []
              }
            },
            {
              "carries_qualifiers": "plot_analysis carries the canonical-identity framing — elevator_pitch positions the film as an auteur masterwork of the kind the critic canon routinely elevates.",
              "space": "plot_analysis",
              "weight": "supporting",
              "content": {
                "elevator_pitch": "an authorial work of world cinema treated as a defining masterwork by critic-canon consensus.",
                "genre_signatures": ["arthouse", "world cinema", "auteur cinema"],
                "thematic_concepts": ["authorial vision", "formal experimentation"]
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

**Example: "1001 Movies to See Before You Die" — Semantic + METADATA both fire; broadly canonical list with educational-breadth shape**

```xml
<raw_query>films from 1001 Movies You Must See Before You Die</raw_query>
<overall_query_intention_exploration>The user wants films from the "1001 Movies to See Before You Die" editorial canon — a broad educational-breadth list spanning eras and movements, with a generally high-reception shape.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films included in "1001 Movies to See Before You Die".</captured_meaning>
  <category_name>Curated canon / named list</category_name>
  <fit_quality>clean</fit_quality>
  <atomic_rewrite>Films included in the 1001 Movies to See Before You Die canon.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>films from 1001 Movies You Must See Before You Die</query_text>
  <description>Films included in the "1001 Movies to See Before You Die" editorial canon.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Decode what the 1001 Movies canon implies — a broad editorial canon covering eras, movements, and countries with an educational-breadth sensibility and a generally high-reception shape.",
      "endpoint_coverage": "SEMANTIC reception carries the list-citation prose and the broadly-canonical register. plot_analysis supports at the identity level for the canonical-stature framings the list accumulates in reviews."
    },
    {
      "aspect_description": "Anchor the numeric reception prior.",
      "endpoint_coverage": "METADATA reception fires as a light-to-moderate additive lift. The 1001 Movies canon is broader than AFI but still skews to well-regarded films — the well_received lift aligns with the list's overall shape while Semantic prose carries the per-film list-citation signal."
    }
  ],
  "overall_endpoint_fits": "The 1001 Movies canon implies a broad editorial canon covering the history of cinema with an educational-breadth sensibility — less mainstream than AFI, less arthouse-concentrated than Sight & Sound, but broadly high-reception. SEMANTIC carries the list-citation language and canonical-stature framing; METADATA reception adds an additive well_received prior that matches the list's generally high-reception shape.",
  "per_endpoint_breakdown": {
    "semantic": {
      "should_run_endpoint": true,
      "endpoint_parameters": {
        "match_mode": "trait",
        "parameters": {
          "qualifier_inventory": "named list is 1001 Movies to See Before You Die — broad editorial canon spanning eras and movements, educational-breadth sensibility, broadly high-reception.",
          "space_queries": [
            {
              "carries_qualifiers": "reception carries the list-citation language and broadly-canonical register — reception_summary names inclusion in the canon, praised_qualities lists the axes commonly cited for films in the compendium.",
              "space": "reception",
              "weight": "central",
              "content": {
                "reception_summary": "included in the '1001 Movies to See Before You Die' editorial canon; widely regarded as an essential work in the history of cinema.",
                "praised_qualities": ["canonical essential", "must-see film", "educational canon", "widely regarded", "historically significant"],
                "criticized_qualities": []
              }
            },
            {
              "carries_qualifiers": "plot_analysis carries canonical-stature identity framing — elevator_pitch positions the film as an essential must-see of its era or movement.",
              "space": "plot_analysis",
              "weight": "supporting",
              "content": {
                "elevator_pitch": "an essential film frequently included in must-see overviews of cinema history.",
                "genre_signatures": [],
                "thematic_concepts": ["cinematic significance", "canonical essential"]
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
          "constraint_phrases": ["1001 Movies to See Before You Die"],
          "target_attribute": "reception",
          "value_intent_label": "broadly well-received canonical essentials",
          "reception": "well_received"
        },
        "polarity": "positive"
      }
    }
  }
}
```

**Example: unknown / fabricated list — no-fire on both endpoints**

```xml
<raw_query>movies from my favorite critic's top ten</raw_query>
<overall_query_intention_exploration>The user names a "list" that does not map to any real canon the system recognizes — an individual critic's personal top-ten with no identifiable publication or institution behind it. There is no list semantics to decode.</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>Films on an unspecified personal top-ten list.</captured_meaning>
  <category_name>Curated canon / named list</category_name>
  <fit_quality>partial</fit_quality>
  <atomic_rewrite>Films on a personal critic's top-ten list with no named source.</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>movies from my favorite critic's top ten</query_text>
  <description>Films on a personal top-ten list belonging to an unspecified critic.</description>
  <modifiers></modifiers>
</parent_fragment>
<sibling_fragments></sibling_fragments>
```

Expected output:

```json
{
  "requirement_aspects": [
    {
      "aspect_description": "Identify the named list and decode its aesthetic implications.",
      "endpoint_coverage": "Neither endpoint can honestly fire. The user did not name a publication, institution, or recognized canon — only an unspecified personal list. There is no list aesthetic to decode, no citation language to author, and no reception-direction signal strong enough to justify a lift. SEMANTIC reception prose invented from nothing would pollute the result pool; METADATA reception fired as a generic well_received lift would quietly substitute 'broadly well-regarded' for the list the user did not actually name."
    }
  ],
  "overall_endpoint_fits": "The atom routed here names no real curated canon — an unspecified critic's personal top-ten is not a list whose semantics the handler can decode. Authoring Semantic reception prose would be fabrication; firing METADATA reception would substitute a generic quality prior for a list-membership ask the system cannot satisfy. No-fire is the correct response — honesty about the missing list semantics beats inventing them.",
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
