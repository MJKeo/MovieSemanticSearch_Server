# Current Search Flaws

Problems discovered through analysis and testing of the current search system.

---

## 1. Additive Scoring Produces Disjunctive Results

The core architectural flaw: every layer of the scoring pipeline uses weighted sums,
which is fundamentally a **soft OR**. A movie that excels at one attribute beats a
movie that is good at multiple attributes.

**Formula at every level:**
```
final = w_V * vector + w_L * lexical + w_M * metadata
vector = SUM(weight_space * normalized_score_space)
```

A weighted sum rewards movies exceptional at ANY ONE thing, not movies good at ALL
things. For "iconic twist ending," a movie scoring 0.9 on narrative_techniques and
0.0 on everything else beats a movie scoring 0.5 on narrative_techniques and 0.5 on
reception, even though the second movie better satisfies the full query.

**Where additive scoring IS correct:** ranking by preferences within a set of
candidates that already satisfy core constraints. The math isn't wrong — it's applied
at the wrong layer.

---

## 2. The "Iconic Twist Ending" Case Study

**Query:** "iconic twist ending" and "popular movies that are known for their twist endings"

**Expected top results:** The Sixth Sense, Fight Club, The Usual Suspects, Psycho,
The Prestige, Se7en, Shutter Island, Gone Girl, Oldboy, Planet of the Apes, The
Others, Primal Fear, Saw, The Game, Identity, Get Out

**Actual top results:** Wild Things, A Perfect Getaway, The Invisible Guest, Dot the
I, Femme Fatale, Malice — all legitimate twist movies, but mid-tier thrillers, not
the iconic ones anyone would name first.

**None of the definitive twist-ending movies appeared in either top-25.**

### What we ruled out

**Metadata quality is NOT the issue.** We compared the generated metadata for Fight
Club (TMDB ID 550), The Sixth Sense (745), Wild Things (617), and A Perfect Getaway
(12403). All four have strong twist-related terms in their `information_control`
section:

| Movie | information_control terms |
|-------|--------------------------|
| Wild Things | "successive plot twists", "post-credits reveal" |
| A Perfect Getaway | "midpoint twist / reversal", "planted-evidence misdirection" |
| Fight Club | "major twist / identity reveal", "late-reveal recontextualization" |
| The Sixth Sense | "plot twist / reversal", "planted-foreshadowing clues" |

Fight Club even has "plot-twist heavy" in its cognitive_complexity terms and The
Sixth Sense has "shocking twist" in ending_aftertaste. The metadata captures twist
endings well for all four movies.

### Actual cause: multi-factor

The issue is a combination of the additive scoring architecture (the dominant factor)
and a secondary embedding density effect:

**Embedding density effect:** Wild Things has the word "twist" appearing ~5 times
across its metadata ("successive plot twists", "twist-driven tension", "twisty",
"dizzying twists", etc.) because twists are its *entire identity*. Fight Club has
"twist" ~2 times because its embedding surface is spread across identity crisis,
consumerism, anarchy, violence, etc. When the query is about twists, a movie whose
embedding is saturated with twist language has higher raw cosine similarity, even
though both movies clearly have twist endings.

**Weight system amplifies this:** The weight calibration (large=3, medium=2, small=1,
not_relevant=0) means narrative_techniques (where "twist ending" lands) dominates the
score. "Iconic" has nowhere meaningful to contribute — reception or watch_context
might get small weight, but that's 1/3 the influence of the twist signal.

---

## 3. Within-Space Normalization Erases Absolute Quality

The exponential decay normalization (vector_scoring.py:442-524) sets the best
candidate in each space to 1.0 and decays everyone else relative to that best:

```
gap(s) = (s_max - s) / (s_max - s_min)
normalized(s) = exp(-k * gap(s))
```

This makes scores **incomparable across spaces**. A 0.7 in narrative_techniques
doesn't mean the same thing as a 0.7 in reception. When summed, you're adding apples
and oranges.

The normalization is working as designed for its original purpose (making scores
within a space relative), but it prevents any cross-space reasoning about whether a
movie is "truly strong" on a particular attribute vs "relatively strong within this
query's particular candidate pool."

---

## 4. Reranking Step Helps But Too Late

Quality reranking (reranking.py:73-100) buckets by rounded final_score (2 decimal
places) and sorts within buckets by reception score. This WOULD surface Fight Club
and The Sixth Sense if they were in the same bucket as mid-tier thrillers — but with
BUCKET_PRECISION=2, a movie at 0.55 and one at 0.53 are in different buckets.

The reranking only helps when movies have essentially identical relevance scores, not
when additive scoring has already created a gap.

---

## 5. Metadata Prompt Anti-Retrieval Pattern

The metadata generation prompts intentionally teach the LLM to avoid the exact
language users search with:

- **viewer_experience cognitive_complexity prompt**: "Phrase in terms of the experience
  of the viewer, not descriptions of the movie (e.g. 'plot twist' is not a good
  phrase)" — so it generates "kept me guessing" instead of "plot twist"
- **narrative_techniques information_control prompt**: requires specific evidence —
  "Genre conventions (e.g., 'mystery' implies twists) are NOT sufficient to populate
  this section"

This creates a retrieval mismatch: embedding text is optimized for nuanced
description rather than discoverability. The subquery generation step is supposed to
bridge this gap by translating user language into metadata language, but it only
partially succeeds.

**Note:** This is less of a problem than initially theorized. The metadata comparison
showed all four movies DO have explicit twist language. The density effect is more
impactful than the vocabulary mismatch.

---

## 6. Channel Weight System Forces Winner-Take-All

The channel-level merge (`final = w_V * vector + w_L * lexical + w_M * metadata`)
has the same additive problem as the vector scoring. For most queries with no lexical
entities and no metadata preferences extracted, the channel weights collapse to
vector-heavy, meaning the metadata channel's quality signals (popularity, trending
status) get minimal influence even when they're conceptually important to the query.

The _correct_channel_weights function (search.py:97-131) applies rule-based fixes but
only for contradictions between channel weights and actual outputs — it doesn't
address the structural issue of additive combination.
