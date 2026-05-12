# Few-shot examples — Story / thematic archetype

These calibrate two things on top of the keyword-vs-semantic routing
decision:

1. When Semantic fires, the body lands EXCLUSIVELY on `plot_analysis`.
   `primary_vector` is always `plot_analysis`. viewer_experience and
   narrative_techniques are NEVER populated for this category.
2. Cross-field repetition of the load-bearing thematic term across
   `elevator_pitch` / `plot_overview` / `thematic_concepts` /
   `character_arcs` is mandatory — the ingest side does this on
   purpose, and the query side must match.

<example>
Input:
```xml
<retrieval_intent>Find films built around a coming-of-age story shape.</retrieval_intent>
<expressions><expression>coming-of-age stories</expression></expressions>
```

Expected: preferred-only when a direct coming-of-age tag exists.
Binary framing makes Keyword appropriate; Semantic does not fire.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films built around a redemption arc.</retrieval_intent>
<expressions><expression>redemption arc</expression></expressions>
```

Expected: Semantic fallback when no canonical "redemption arc" tag
exists. `primary_vector` = `plot_analysis`. Body:

```json
{
  "plot_analysis": {
    "elevator_pitch": "a redemption arc",
    "plot_overview": "A flawed protagonist confronts past wrongs and works toward redemption, repairing relationships and reclaiming moral ground over the course of the story.",
    "genre_signatures": ["redemption drama"],
    "conflict_type": ["man vs self"],
    "thematic_concepts": ["redemption", "atonement", "moral repair"],
    "character_arcs": ["redemption", "moral redemption"]
  }
}
```

Why this works:
- "redemption" appears in five of the six sub-fields — the
  cross-field repetition mirrors the ingest side and weights the
  central concept in the embedded vector.
- `conflict_type` fires because a redemption arc is naturally
  framed as the protagonist against their own past self.
- `genre_signatures` carries one tight subgenre label, not three
  adjacent ones.
- `thematic_concepts` uses true paraphrases of the same idea
  (atonement, moral repair). All pass the substitution test.
- viewer_experience, narrative_techniques, and the other four
  spaces are NOT populated.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films with grief present as a light thematic thread.</retrieval_intent>
<expressions><expression>kind of about grief</expression></expressions>
```

Expected: Semantic fallback even if a `grief` tag exists — "kind
of about" is gradient framing and binary tags do not carry degree.
`primary_vector` = `plot_analysis`. Body:

```json
{
  "plot_analysis": {
    "elevator_pitch": "a story touched by grief",
    "plot_overview": "Against another central concern, characters quietly carry grief, with loss surfacing as a recurring undercurrent rather than the story's main engine.",
    "thematic_concepts": ["grief", "loss", "mourning"]
  }
}
```

Why this works:
- The body LEANS on grief (presence in three fields) without
  asserting it as the central plot driver — matches the "thread
  of" framing.
- `genre_signatures` stays empty: grief is thematic territory, not
  a subgenre signature.
- `character_arcs` and `conflict_type` stay empty: the trait does
  not name a transformation or a "X vs Y" conflict.
- No viewer_experience body. The temptation is to emit
  `emotional_palette.terms = ["grief", "melancholy", "heavy"]` —
  but that retrieves films that FEEL heavy, not films that are
  ABOUT grief. A breezy comedy with a grief subplot matches the
  trait; a grim revenge thriller does not.
</example>

<example>
Input:
```xml
<retrieval_intent>Find man-vs-nature survival stories.</retrieval_intent>
<expressions><expression>man-vs-nature survival</expression></expressions>
```

Expected: Semantic fallback when no canonical man-vs-nature tag
covers the framing. `primary_vector` = `plot_analysis`. Body:

```json
{
  "plot_analysis": {
    "elevator_pitch": "man vs nature survival",
    "plot_overview": "A protagonist pitted against a hostile natural environment fights to survive — the wilderness, the elements, or the wild itself is the antagonist.",
    "genre_signatures": ["survival drama", "wilderness survival"],
    "conflict_type": ["man vs nature"],
    "thematic_concepts": ["survival", "endurance", "human vs wilderness"],
    "character_arcs": ["survival arc"]
  }
}
```

Why this works:
- `conflict_type` fires cleanly — this is the canonical use case.
- `genre_signatures` carries the recognizable subgenre tag pair.
- "survival" repeats across `elevator_pitch` / `plot_overview` /
  `thematic_concepts` / `character_arcs` for cross-field weighting.
- viewer_experience is NOT populated even though survival films
  feel tense — that's a feel claim, not a theme claim.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films whose focal subject is a named historical figure.</retrieval_intent>
<expressions><expression>about JFK</expression></expressions>
```

Expected: no-fire. Concrete focal subject belongs to CENTRAL_TOPIC,
not here.
</example>

<example>
Input:
```xml
<retrieval_intent>Find anti-hero protagonists.</retrieval_intent>
<expressions><expression>anti-hero</expression></expressions>
```

Expected: no-fire. Static character type belongs to
CHARACTER_ARCHETYPE, not here. A redemption arc featuring an
anti-hero would split: the arc fires here, the character type
fires on CHARACTER_ARCHETYPE.
</example>

**COUNTER-EXAMPLE — do NOT emit this for "redemption arc":**

```json
{
  "viewer_experience": {
    "emotional_palette": {
      "terms": ["uplifting", "cathartic", "hopeful", "redemptive"],
      "negations": ["not bleak", "not nihilistic"]
    },
    "tone_self_seriousness": {
      "terms": ["earnest", "sincere"],
      "negations": ["not cynical"]
    }
  }
}
```

Why this fails:
- This retrieves films that FEEL uplifting and earnest — including
  feel-good comedies, sports underdogs, and inspirational romances
  that have no redemption arc at all.
- It misses grim redemption stories (downbeat character studies
  where a flawed protagonist seeks atonement) because they don't
  read as "uplifting" or "earnest" tonally — even though they ARE
  redemption arcs.
- The trait is about the SHAPE of the story, not the FEEL of
  watching it. Route to plot_analysis only.
