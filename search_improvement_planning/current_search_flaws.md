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

The issue is a combination of the additive scoring architecture, a deeper embedding
format problem, and broken vector space routing:

**Embedding format problem (primary):** Multi-dimensional movies get poorly
represented in the current embedding process. Each vector space's text is embedded
as a flat list of terms. A movie like Wild Things, whose entire identity is "twist
ending," saturates the embedding with twist signal. A movie like The Sixth Sense
spreads its embedding budget across grief, child psychology, a deteriorating marriage,
supernatural horror, AND the twist — individual signals get muted.

**Empirical evidence (from individual vector space testing):**
- The Sixth Sense does not appear in the top 1000 results for "twist ending" in
  narrative_techniques — even when using the *exact wording from its own metadata*.
  It scores 0.5730, which is 82% of the top score. This is a retrieval failure, not
  a scoring failure.
- The gap between density-saturated movies (top 10%) and multi-dimensional movies
  is massive, then flattens out. This isn't because the 15-45% range movies don't
  match — it's because the embedding format loses their signal.

**Theory:** Embedding with structured labels (e.g., "key_movie_feature_draws: twist
ending, plot twist") instead of flat term lists would preserve per-attribute signal
by giving the embedder structural context. The search LLM could generate the same
shape (e.g., NarrativeTechniquesOutput) which gets templated identically before
embedding. This is testable and high-priority.

**Weight system amplifies this:** The weight calibration (large=3, medium=2, small=1,
not_relevant=0) means narrative_techniques (where "twist ending" lands) dominates the
score. "Iconic" has nowhere meaningful to contribute — reception or watch_context
might get small weight, but that's 1/3 the influence of the twist signal.

**Vector space routing is independently broken:** For "iconic twist ending," the
system assigned weights: narrative_techniques=0.3571, reception=0.2381,
plot_analysis=0.1190, watch_context=0 (empty). But watch_context had the most
twist-related content for The Sixth Sense and wasn't queried at all. The subquery
generation system doesn't understand which spaces actually contain relevant signal.

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

**Revised assessment:** The vocabulary mismatch is real but secondary. The metadata
comparison showed all four movies DO have explicit twist language — the anti-retrieval
prompting isn't preventing the right terms from being generated. The deeper problem
is the embedding format: even when the metadata contains "plot twist / reversal,"
the flat-list embedding format dilutes that signal for multi-dimensional movies.
The anti-retrieval pattern and the embedding format problem compound: nuanced phrasing
is harder for the embedder to match AND gets diluted by surrounding terms.

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

---

## 7. Production Vector / Lexical Signal Overlap

When a query mentions a named entity (e.g., "Christopher Nolan"), both the lexical
channel and the production vector space can fire on the same information. The movie
gets double-credited through two channels for a single signal.

The production vector text includes director names, studio names, and other entities
that the lexical search also indexes. When the weighting algorithm treats both channel
scores as independent evidence, it inflates scores for movies that happen to match on
overlapping content — particularly for entity-heavy queries where the lexical channel
is already doing the heavy lifting.

**Possible approaches:**
- Strip lexical-matchable entities from production vector text at ingestion time
- Detect overlap at scoring time and dampen the weaker signal
- Accept the overlap but ensure the weighting algorithm doesn't stack both signals
  (e.g., take the max of the two rather than summing)

**Resolution:** Naturally resolved by the data layer redesign. Production companies
are already in studio lexical postings; countries of origin, source material, franchise
lineage, and other entity-adjacent content are moving to structured fields. After these
removals, the production vector contains only filming locations + production style
keywords — no lexical-matchable entities remain.

---

## 8. Franchise Logic Is Too Broad

"Marvel" as a franchise currently matches too broadly. Querying "Marvel movies" can
match on character names that appear in Marvel films rather than routing to the
studio/brand entity. The franchise concept is actually multi-level:

- **Studio level:** "Marvel Studios films" (production company)
- **Brand level:** "MCU movies" (a specific cinematic universe)
- **Character level:** "Spider-Man movies" (includes Sony Spider-Man, not MCU)

The entity extraction step doesn't distinguish which level the user means. "Marvel"
should route to studio/brand matching, not match any movie with a Marvel character in
the cast. Similarly, "Spider-Man movies" should include all Spider-Man films regardless
of studio, while "MCU movies" should exclude non-MCU Spider-Man films.

**Resolution:** New `movie_franchise_metadata` table with `lineage`,
`shared_universe`, `recognized_subgroups`, `launched_subgroup`,
`lineage_position`, `is_spinoff`, `is_crossover`, and `launched_franchise`.
Both `lineage` and `shared_universe` feed the shared
`lex.inv_franchise_postings` table, replacing the title-token +
character-matching hack. See the current franchise storage plan in
`franchise_metadata_planning.md`.

---

## 9. Implicit Quality Prior Is Too Weak and Non-Adaptive

**Evidence:** "Silly comedies" returned qualitatively different results from "silly
comedies that people actually know about." The explicit version matched the user's
actual intent better — the implicit quality prior either isn't being applied strongly
enough or isn't being applied at all for certain query types.

The current quality reranking (bucket by score, sort by reception within buckets)
only activates when movies have near-identical relevance scores. It can't compensate
when the vector scoring has already created a gap between obscure-but-on-topic movies
and well-known-but-slightly-less-on-topic movies.

This is a distinct problem from the additive scoring flaw (#1). Even if the
architecture were fixed, the system still needs a mechanism to understand how much the
user implicitly values notability/quality and apply that as a dynamic signal rather
than a fixed afterthought.

---

## 10. Metadata Filters Miss Implicit Temporal Signals

**Example:** "Disney animated classics" — should the system trigger a date range
filter? Two failure modes:

- **Too aggressive:** Hard-filtering to pre-2000 misses Tangled, Frozen, Moana
- **Not aggressive enough:** No date signal at all means a 2022 release can top the
  results, which feels wrong for a "classics" query

The word "classic" carries an implicit temporal signal (cultural establishment over
time) that the current metadata extraction doesn't capture. It's not a hard date
range — it's a soft preference for older, more established titles. The query
understanding step needs to translate "classics" into a date *preference* (bias toward
older) rather than either a hard filter or nothing.

This generalizes beyond "classics": words like "iconic," "legendary," "timeless,"
and "essential" all carry similar temporal-establishment implications.

---

## 11. Lexical Matching Lacks Actor Prominence Signal

For a "Tom Hanks movies" query, a Tom Hanks starring role should score higher than a
Tom Hanks 30-second cameo. Currently lexical matching treats actor presence as binary
— the actor is either in the movie or not.

Pure actor list index doesn't tell the whole story because the size of the movie
dictates how many "fluff" characters exist. Being 5th-billed in a small indie with 8
cast members is very different from being 5th-billed in a Marvel ensemble with 40+
credited actors.

**Available signals for prominence:**
- Billing order (already available from TMDB credits)
- Cast list size relative to billing position (normalizes across movie sizes)
- IMDB "known for" designation for that movie-actor pair

**Resolution:** Role-specific person posting tables (inv_actor_postings with
billing_position + cast_size, inv_director_postings, etc.). Prominence score =
`1.0 - (position / cast_size)`. Three query modes: exclude non-major (for "starring"
queries, top min(2-3, 10-15%)), boost by position (default), binary (fallback).
Role boost behavior: when role not stated, boost implied role (e.g., "Spielberg" →
boost director) but still show other roles at lower weight. See data layer decisions
in new_system_brainstorm.md.

---

## 12. Observed Query Failures

Concrete failure cases beyond the "iconic twist ending" case study. These serve as
evaluation benchmarks for the redesigned system.

| Query | Failure Mode | Root Cause |
|-------|-------------|------------|
| "Disney animated classics" | Cross-channel composition fails; recent movies rank too high | Franchise logic (#8), temporal signal (#10), metadata filter calibration |
| "Dark, gritty marvel movies" | Franchise matches wrong entity type | Franchise logic (#8), cross-channel intersection |
| "Critically acclaimed christmas movies" | Kimi errored on lexical parsing; christmas aspect underweighted | Christmas is the deal-breaker but scored as preference; acclaim should be ranking axis, not additive signal |
| "Silly comedies" vs "...that people actually know about" | Vastly different quality of responses; explicit notability phrasing needed | Quality prior too weak (#9) |
| "Psych-thriller lover + partner who likes artistically driven movies" | Can't satisfy partially conflicting audience preferences | Multi-audience intersection; no mechanism for "partially satisfy both" |
| "Dark, gritty marvel movies" (vector test) | Captain America: Winter Soldier missing from vector results entirely | Semantic concepts like "dark and gritty" can't reliably generate candidates — must score within entity-retrieved set (#13) |
| "Funny horror movies" (vector test) | Zero intersection between funny top-1000 and horror top-1000 | Broad tonal concepts produce non-overlapping candidate sets; intersection at retrieval time fails completely (#13) |

---

## 13. Flat-List Embedding Format Loses Per-Attribute Signal

**Root cause underlying flaws #2 and #5.** Each vector space's text is embedded as a
flat concatenation of metadata terms. The embedding model (text-embedding-3-small,
1536 dims) compresses all terms into a single vector, causing signal dilution for
multi-dimensional movies.

**Evidence:**
- The Sixth Sense doesn't appear in the top 1000 for "twist ending" in
  narrative_techniques despite having explicit twist language in its metadata.
  Score: 0.5730 (82% of max). The embedding loses the twist signal among the
  movie's many other attributes.
- Movies in the top 10% of results show massive score gaps over movies in the
  15-45% range, then scores flatten. This pattern is consistent across attribute
  types, suggesting the embedding format systematically favors one-dimensional
  movies.

**Implication for the new system:** The architecture redesign (deal-breaker/preference
split, threshold + flatten) cannot work if relevant movies never enter the candidate
pool. Fixing the embedding format is a prerequisite for the new retrieval architecture,
not an independent improvement.

**Proposed fix:** Embed vector text with structured labels that preserve per-attribute
context (e.g., "information_control: plot twist / reversal, planted-foreshadowing
clues") instead of flat term lists. Generate search queries in the same structured
shape so that query and document embeddings occupy the same semantic structure. This
is testable by comparing retrieval ranks for known-relevant movies under both formats.

---

## 14. Semantic Concepts Cannot Reliably Generate Candidates

**Discovered through cross-channel intersection testing.** Broad tonal/experiential
concepts ("funny," "dark and gritty") fail as candidate generators via vector
retrieval. Zero intersection between "funny" top-1000 and "horror" top-1000. "Dark
and gritty" top-1000 misses Captain America: Winter Soldier entirely.

**This creates a distinction the original planning didn't make:** There are
*user-intent deal-breakers* (the user considers it non-negotiable) and
*retrieval-capable deal-breakers* (the concept can reliably generate candidates via
vector search). These are not the same set.

Some concepts that are user-intent deal-breakers must be handled as post-retrieval
scoring signals because they can't reliably generate candidates. "Funny" in "funny
horror movies" is a deal-breaker from the user's perspective, but "horror" (genre
metadata) must generate the candidates, and "funny" scores/filters within that set.

**Design implication:** Phase 0 needs to classify semantic concepts not just by user
intent (deal-breaker vs preference) but by retrieval reliability. Or alternatively,
the system always generates candidates from the most reliable channel available
(metadata filters, entity lookup, keywords) and applies all semantic deal-breakers
as rescore filters — which is closer to what actually works empirically.

---

## 15. Vector Space Routing Misallocates Signal

The subquery generation system doesn't understand which vector spaces actually contain
relevant signal for a given concept. For "iconic twist ending":
- watch_context had the most twist-related content for The Sixth Sense but got
  zero weight (empty subquery)
- reception got 23.8% weight despite being less twist-specific
- narrative_techniques got 35.7% weight (correct, but insufficient alone)

This is independent of both the scoring architecture and the embedding format. Even
with perfect scoring and perfect embeddings, querying the wrong spaces produces bad
candidates. The current LLM-driven space routing lacks understanding of what content
actually lives in each space.
