# Similar Movies Search

A separate flow for unqualified "like X" queries (e.g. "like Inception"). Does not go through the standard search pipeline — has its own candidate generation and ranking.

## Goal

Given a single anchor movie, return movies that someone would genuinely consider "similar" — capturing both substance (themes, plot) and experience (tone, scale, prestige tier).

## Process Overview

Use a lane-based candidate generator plus final rerank/weave, not one monolithic retrieval score.

1. Load the anchor movie and derive binary anchor types from its stored metadata.
2. Generate candidates from independent similarity lanes.
3. Normalize every lane score to `[0, 1]`.
4. Attach an evidence bundle to each candidate explaining why it was retrieved.
5. Derive lane weights from active anchor types, then normalize the final lane-weight distribution so all weights sum to `1.0`.
6. Compute a combined score from the normalized lane scores and normalized lane weights.
7. Weave the final list using combined score plus lane-variety constraints.

Core principle: people asking for "movies like X" usually mean same experience, but they may also mean same creative fingerprint, same story world, same source tradition, same studio house style, or same taste/status tier. The system should preserve those different meanings long enough to mix them deliberately.

## Candidate Lanes

Each lane can retrieve candidates independently and contributes a normalized `[0, 1]` score to the final evidence bundle.

### 1. Movie Shape (backbone)

Vector similarity across all 8 spaces. The vector blend should use tiers:

- **Tier 1:** `plot_analysis`, `viewer_experience`
- **Tier 2:** `watch_context`, `production`, `reception`, `dense_anchor`
- **Tier 3:** `narrative_techniques`, `plot_events`

Tier 1 should carry the most influence because it best captures what users usually mean by "same kind of movie": thematic shape, premise structure, emotional experience, tone, and audience feel. Tier 3 should be weaker because `plot_events` can be too literal and `narrative_techniques` can be too craft-specific.

Base vector-space weights are the same for single-anchor and multi-anchor search:

```text
plot_analysis:         1.00
viewer_experience:     1.00
watch_context:         0.65
production:            0.65
reception:             0.65
dense_anchor:          0.65
narrative_techniques:  0.35
plot_events:           0.35
```

### 2. Director (strong-moderate lane, not an anchor)

Lexical match on director. Director remains a useful creative-fingerprint lane, but we are **not** deriving a separate `director_signature` anchor type for V1 because identifying when a director is unusually defining requires additional precomputed director-cohesion data.

Should not depend heavily on movie shape (a Nolan film is still a relevant "like Inception" result even if its plot diverges), but director and shape scores **are not mutually exclusive** — a movie that matches on both should rank higher than one that matches on either alone.

### 3. Franchise / Universe / Lineage

Direct franchise / universe match: sequels, prequels, spinoffs, remakes, shared-universe entries, and direct lineage matches. Worth weaving in, but **should not dominate the result list unless the anchor is franchise-dominant and the candidates are still competitive**.

### 4. Studio Lineage

Production-brand / studio-company overlap is useful as a **latent lineage signal**: it can capture shared creative machinery, audience calibration, genre tendency, production mode, or house style that semantic vectors may only partially infer from the movie text.

This is **not** a replacement for semantic shape. The vector spaces already capture much of the observable output of studio influence — tone, premise, pacing, themes, genre, production style. Studio lineage should usually act as a tie-breaker, agreement boost, and limited recall-repair path.

Use only identity-bearing brands or specific company/studio labels whose name meaningfully predicts something about the movie. Broad corporate umbrellas are weak for similarity even when useful for explicit studio search.

Curated company/string-level similarity list:

| Signal | Company / label strings | Date handling | Rationale |
| --- | --- | --- | --- |
| High | `Pixar Animation Studios`, `Pixar` | Split: `1995-2010`, `2011-2019`, `2020-present` | Strong house style, but eras differ: golden original run, sequel/franchise-heavy period, modern/streaming recovery. |
| High | `Walt Disney Animation Studios`, `Walt Disney Feature Animation` | Split: `1989-1999`, `2000-2008`, `2009-2019`, `2020-present` | Renaissance, experimental/post-Renaissance, Revival, and modern eras are meaningfully different. |
| High | `Studio Ghibli` | No V1 split | Extremely coherent studio identity; director/vector lanes can absorb internal differences. |
| High | `DreamWorks Animation`, `Pacific Data Images (PDI)` | Split: `1998-2004`, `2005-2016`, `2017-present` | Early mixed/traditional-CG era, franchise/comedy peak, NBCUniversal era. |
| High | `Illumination Entertainment`, `Illumination Studios Paris` | No V1 split | Highly consistent family-comedy/franchise house style. |
| High | `Marvel Studios` | Split: `2008-2019`, `2021-present`; ignore/weak before `2008` | Infinity Saga vs post-Endgame/Multiverse era. |
| High | `DC Films`, `DC Studios` | Split: `2016-2022`, `2023-present` | DCEU studio era vs Gunn/Safran DC Studios era. |
| High | `Lucasfilm`, `Lucasfilm Animation` | Split: `1977-1989`, `1999-2012`, `2015-present` | Original Indy/Star Wars era, prequel/Clone Wars era, Disney-era Star Wars. |
| High | `A24` | No V1 split | Strong modern indie/auteur/genre-prestige signal. |
| High | `Neon` | No V1 split | Strong arthouse/international/prestige acquisition identity. |
| High | `Blumhouse Productions`, `Blumhouse International` | Split: `2009-2014`, `2015-present` | Early low-budget supernatural/found-footage vs broader social/auteur/franchise horror. |
| High | `Searchlight Pictures`, `Fox Searchlight Pictures` | No V1 split | Specialty/prestige identity is stable across Fox/Disney rename. |
| High | `Focus Features`, `Focus Features International (FFI)` | No V1 split | Stable prestige/specialty label. |
| High | `Good Machine`, `Good Machine Films`, `USA Films` | Use registry windows | Strong Focus-predecessor prestige/indie signal. |
| High | `Miramax` | Split: `1989-2005`, `2006-present` | Weinstein-era indie/prestige signal is strong; post-2005 weaker. |
| High | `New Line Cinema`, `New Line Productions`, `New Line Film`, `New Line Film Productions` | Split: `1984-1994`, `1995-2008`, `2009-present` | Horror/cult period, broad genre/blockbuster period, post-WB integration weaker. |
| Moderate | `Sony Pictures Animation` | Split: `2006-2017`, `2018-present` | Post-`Spider-Verse` era is more distinctive than earlier family-comedy output. |
| Moderate | `Atomic Monster` | `2024-present` in current registry | Useful horror signal, but less proven inside post-merger data. |
| Moderate | `Touchstone Pictures`, `Touchstone Films` | Split: `1984-2002`, `2003-2016` | Adult/non-Disney Disney label was meaningful early; later became less distinct. |
| Moderate | `Screen Gems` | `1999-present` | Useful genre/action-horror signal, not broad Sony. |
| Moderate | `Lionsgate`, `Lion's Gate Films`, `Lions Gate Films`, `Lions Gate Entertainment`, `Lions Gate`, `Lionsgate Premiere`, `Lionsgate Productions`, `Lions Gate Studios` | Split: `2000-2011`, `2012-present` | Early indie/horror/mid-budget, then Summit/YA/franchise-heavy identity. |
| Moderate | `Summit Entertainment`, `Summit Premiere` | `2012-present` per registry | YA/franchise/commercial genre signal under Lionsgate. |
| Moderate | `Fox 2000 Pictures` | `1994-2020` | Literary/adult-commercial drama signal. |
| Moderate | `Paramount Vantage`, `Paramount Classics` | `2006-2013` | Specialty/prestige sublabel, but sparse. |
| Moderate | `Paramount Animation`, `Paramount Animation Studios` | `2011-present` | Some family-animation signal, but less coherent than Pixar/DreamWorks/Illumination. |
| Moderate | `Netflix Animation` | No V1 split | More meaningful than broad Netflix, but still mixed. |
| Moderate | `Apple Original Films`, `Apple Studios` | No V1 split | Moderate prestige/adult-drama signal; catalog still young. |
| Moderate | `Amazon Studios`, `Amazon MGM Studios` | Split: pre-`2022`, `2022-present` | Streamer prestige/commercial mix; MGM integration changes identity. |
| Moderate | `Metro-Goldwyn-Mayer (MGM)`, `Metro-Goldwyn-Mayer (MGM) Studios`, `Metro-Goldwyn-Mayer Studios` | Split: `1924-1959`, `1960-1980`, `1981-present` | Classic MGM is meaningful; modern MGM is mostly franchise/library and weaker. |
| Moderate | `United Artists`, `United Artists Pictures` | Split: `1919-1951`, `1952-1981`, `1982-present` | Stronger as classic/artist-founded and later New Hollywood/adult-film signal; weak after MGM era. |
| Moderate | `Warner Bros. Animation`, `Warner Bros. Feature Animation`, `Warner Bros. Cartoon Studios` | Split classic cartoons separately if shorts matter | Use animation labels only, not broad Warner Bros. |

Do not use these broad company strings for studio similarity, even when they remain valid for explicit studio search:

```text
Walt Disney Pictures
Warner Bros. Pictures / Warner Bros.
Universal Pictures / Universal
Paramount Pictures / Paramount
Sony Pictures / Sony Pictures Entertainment
Columbia Pictures
TriStar Pictures
20th Century Fox / 20th Century Studios
Netflix / Netflix Studios
```

Some broad strings can still matter through other lanes: franchise, source, vectors, or a specific sublabel such as `Screen Gems`, `Fox 2000 Pictures`, or `Sony Pictures Animation`.

Studio identity can be **era-dependent**. For labels whose house style changes over time, apply an era-proximity multiplier rather than treating all same-brand matches equally. Examples: Walt Disney Animation classic / Renaissance / Revival / modern eras; Pixar early-golden vs sequel-heavy/recent eras; Marvel Studios pre-Endgame vs post-Endgame; Miramax's 1990s prestige identity; New Line's genre/indie vs commercial blockbuster phases.

### 5. Source Material

Source-material overlap is its own lane, separate from studio. Use `source_material_v2_metadata.source_material_types` when available, with `source_of_inspiration_metadata` as a fallback for concrete source/adaptation evidence.

This is intentionally not exact-author matching because exact source author is not currently available in the DB. Source lineage should help surface movies that share adaptation tradition or source type — novel, short story, stage play, comic book, folklore/myth/fairy tale, video game, remake, TV adaptation, biography/true-story style signals — even when their literal plot text differs.

### 6. Quality / Reception Bucket

Quality/reception is a scoring lane when the anchor sits at a meaningful taste/status extreme. It is not just a generic quality prior.

The anchor movie is classified into one of three buckets:

- **Bucket 1 — Poorly rated but still loved (cult).** Indicated by opposing reception and popularity stats (low reception score with disproportionate popularity / engagement). Match candidates from the same bucket.
  - Note: this signal also picks up mainstream guilty pleasures (e.g. Transformers) that aren't true cult films. We don't have critic-vs-audience divergence data to disambiguate, so vector matching has to carry that distinction.
- **Bucket 2 — Critically acclaimed.** High reception with popularity as a secondary factor (matters somewhat — separates art-house prestige from mainstream prestige, but reception dominates). Match candidates from the same bucket.
- **Bucket 3 — Middle of the road.** Doesn't match either extreme. Apply a generic quality / notability boost rather than bucket matching, since there's no strong taste signal to preserve.

For a movie like "The Room," "garbage but entertaining" is a defining similarity trait. Cult/so-bad-it's-good anchors should therefore give the quality lane heavy influence, not merely use reception as a tie-breaker.

Bucketing avoids the failure mode where pure vector similarity surfaces tonally-wrong matches (e.g. low-budget indie versions of prestige blockbusters, or prestige cinema in response to "Sharknado").

## Anchor Types

Anchor types are binary, deterministic classifications derived from available movie metadata. Multiple anchor types can be active at once.

### `standard_shape`

Always active.

Impact: movie shape remains the backbone for every anchor. It should be the default source of high-ranking candidates and the main fallback when no special anchor type applies.

### `cult_garbage`

Active when:

- `reception_score <= 45`.
- `public.mv_popularity_percentile.percentile >= 0.89`.

Impact: quality/reception becomes a major lane. The system should actively preserve entertaining badness rather than "correcting" toward conventionally better movies.

### `prestige`

Active when:

- `reception_score >= 85`.
- `public.mv_popularity_percentile.percentile >= 0.75`.

Impact: quality/reception becomes strong, but less dominant than for `cult_garbage`. The system should prefer similarly acclaimed/status-compatible matches inside the semantic neighborhood.

### `franchise_dominant`

Active when:

- The anchor has franchise, shared-universe, subgroup, or direct lineage metadata.
- The related group has multiple catalog entries.
- The relationship is not merely a weak one-off remake edge.

Impact: franchise lane gets more exposure and scoring weight. It should surface obvious direct-world candidates while still preserving non-franchise shape matches.

### `studio_lineage`

Active when:

- The anchor has at least one production brand/company from the curated "meaningful similarity studio/company" list.

Impact: studio lane becomes meaningful. Studio matches can receive real weight and limited exposure, especially when supported by vector agreement and era proximity. If inactive, studio overlap should contribute little or nothing.

### `source_material`

Active when:

- `source_material_v2_metadata.source_material_types` is non-empty.
- Fallback: `source_of_inspiration_metadata` has concrete adaptation/source evidence when V2 source material is missing.

Impact: source-material lane becomes independently important. Source material is more trusted than studio lineage because the mere existence of a source/adaptation pattern often defines user expectations for similarity.

## Lane Weighting

Start with a base lane-weight profile, then apply additive adjustments for every active anchor type. Do **not** cap the lane weights during adjustment. Anchors should be allowed to push lane importance however strongly makes sense. After all anchor adjustments are applied, normalize the final distribution so all lane weights sum to `1.0`.

Finalized single-anchor base profile:

```text
shape:     0.60
director:  0.12
franchise: 0.12
studio:    0.06
source:    0.04
quality:   0.06
```

Finalized single-anchor anchor adjustments:

```text
cult_garbage:
  quality += 0.26
  shape   -= 0.10

prestige:
  quality += 0.16
  shape   -= 0.06

franchise_dominant:
  franchise += 0.18
  shape     -= 0.08

studio_lineage:
  studio += 0.12
  shape  -= 0.04

source_material:
  source += 0.14
  shape  -= 0.04
```

Then normalize:

```text
normalized_weight[lane] = adjusted_weight[lane] / sum(adjusted_weight.values())
```

This keeps the implementation inspectable: a debug payload can show active anchors, raw adjusted weights, normalized weights, lane scores, and final candidate scores.

## Rescoring And Weaving

Each candidate receives an evidence bundle:

```text
movie_id
shape_score
director_score
franchise_score
studio_score
source_score
quality_score
candidate_sources
active_anchor_types
normalized_lane_weights
```

Final combined score:

```text
combined_score = sum(normalized_lane_weight[lane] * lane_score[lane])
```

The combined score answers: "How good is this candidate overall, given the anchor's active similarity profile?"

Weaving answers a different question: "What should the result page feel like?"

Use the combined score to rank candidates within each lane and to decide whether a lane-specific candidate is competitive enough to receive exposure. Weaving should avoid a result list that collapses into one similarity interpretation, while still refusing weak candidates that only match a secondary lane.

Practical weaving rules:

- Mostly follow combined score.
- Prefer candidates with multiple evidence types when scores are close.
- Reserve limited early opportunities for lanes emphasized by active anchors.
- Use `competitive_band = 0.08` when deciding whether a lane-specific candidate is close enough to receive woven exposure.
- Treat the top section as the top 10 results.
- In the top 10, allow at most 4 results dominated by the same lane.
- In the top 10, allow at most 3 franchise results.
- In the top 10, allow at most 1 studio/source result that lacks meaningful shape agreement.
- For `cult_garbage`, allow quality-bucket matches to appear early.
- For `prestige`, prefer high-quality candidates inside close semantic bands.
- For `franchise_dominant`, allow one early direct-lineage candidate, then cap franchise density.
- For `studio_lineage`, require `shape_score >= 0.55` for early exposure.
- For `source_material`, require `shape_score >= 0.45` for early exposure.
- For franchise candidates, require `shape_score >= 0.35` for early exposure.
- For director candidates, require `shape_score >= 0.40` for early exposure.
- Avoid letting any one lane monopolize the first page.

Finalized lane-score details:

```text
quality:
  active only for `cult_garbage` or `prestige` anchors
  inactive for middle-bucket anchors

  cult_score =
    0.5 * low_reception_match
    + 0.5 * engagement_match

  prestige_score =
    0.75 * high_reception_match
    + 0.25 * engagement_or_awards_match

  low_reception_match =
    clamp((50 - reception_score) / 30, 0, 1)

  high_reception_match =
    clamp((reception_score - 75) / 20, 0, 1)

  engagement_match =
    clamp((mv_popularity_percentile - 0.75) / 0.20, 0, 1)

  engagement_or_awards_match =
    max(
      clamp((mv_popularity_percentile - 0.50) / 0.30, 0, 1),
      award_prestige_match
    )

  award_prestige_match =
    1.00 if candidate has major award win
    0.75 if candidate has major award nomination
    0.00 otherwise

  if awards are not available in this flow:
    engagement_or_awards_match =
      clamp((mv_popularity_percentile - 0.50) / 0.30, 0, 1)

studio:
  strong studio match:   1.00
  moderate studio match: 0.65
  weak/broad studio:     0.00

source:
  exact source_material_type match: 1.00
  no exact match:                    0.00

franchise:
  direct lineage / same series: 1.00
  shared universe:             0.85
```

For era-sensitive studios:

```text
era_gap = abs(candidate_release_year - anchor_release_year)
era_proximity = exp(-era_gap / 18)
studio_score *= 0.60 + 0.40 * era_proximity
```

## Multi-Anchor Similarity

For queries like "movies like Inception, Memento, and The Prestige," the system should search for the **shared profile** of the anchor set, not preserve every anchor equally. Assume the provided movies are not totally disjoint. The goal is to find what they have in common and let one-off traits fade.

Do **not** use per-anchor vector similarity after fetching candidates. Drop `anchor_mean_similarity`, `anchor_min_similarity`, and `anchor_coverage` from the design. They are more expensive, harder to reason about, and work against the core retrieval advantage: nearest-neighbor search is cheap when we query one vector, but computing similarity to every anchor for every candidate turns the flow into a heavier rerank system.

### Multi-Anchor Vector Search

For each vector space:

1. Load the anchor vectors for that space.
2. L2-normalize each anchor vector.
3. Compute pairwise cosine similarity among all anchor vectors in that space.
4. Convert the pairwise similarities into a `[0, 1]` cohesion score for that vector space.
5. Average the normalized anchor vectors.
6. L2-normalize the averaged vector.
7. Search Qdrant once for that vector space using the normalized average vector.

The averaged vector is the "jumbo movie" query for that space. It represents the shared center of the anchor set, not the full identity of every input movie.

Vector-space weighting should be driven primarily by measured cohesion, not by the single-anchor prior. The single-anchor tier prior still matters, but mostly as a fallback/stability prior.

Formula:

```text
effective_space_weight =
  (0.75 * cohesion_weight)
  + (0.25 * base_space_weight)
```

V1 cohesion mapping:

```text
cohesion_weight =
  clamp((avg_pairwise_cosine - 0.55) / 0.30, 0.10, 1.00)
```

Then normalize effective weights across all vector spaces:

```text
normalized_space_weight[space] =
  effective_space_weight[space] / sum(effective_space_weight.values())
```

Use the same `base_space_weight` values as single-anchor search.

The practical effect:

- If the anchors strongly agree in `viewer_experience`, that space becomes highly important.
- If the anchors scatter in `plot_events`, that space fades even if it still has a nonzero base prior.
- If every cohesion score is weak, the base prior prevents the vector distribution from becoming arbitrary.

The final multi-anchor shape score is the weighted merge of per-space Qdrant results using `normalized_space_weight`.

### Multi-Anchor Metadata Search

Discrete metadata has no useful "average" equivalent. There is no average director, average franchise, or average source material. For non-vector lanes, search each anchor independently and let repeated candidate matches become the signal.

Metadata lanes:

- `director`
- `franchise` / `shared_universe` / `lineage` (treat lineage and universe as interchangeable for this purpose)
- `studio`
- `source`
- `quality`

For each metadata lane:

1. Extract the lane traits for every anchor movie.
2. Compute lane cohesion from repeated traits across the anchor set.
3. If no trait repeats across at least two anchors, the lane has no multi-anchor boost.
4. Run the lane search independently for each anchor's traits.
5. For each candidate, count how many anchor-specific searches it matched in that lane.
6. Normalize the candidate's lane score by anchor count.

Candidate lane score:

```text
candidate_lane_score =
  matched_anchor_count_for_lane / anchor_count
```

Examples with 3 anchors:

```text
candidate matches director searches from all 3 anchors:
  director_score = 3 / 3 = 1.00

candidate matches director search from 1 anchor:
  director_score = 1 / 3 = 0.33

candidate matches no director searches:
  director_score = 0.00
```

Lane cohesion should be based on repetition among the anchors, not on candidate results. V1 uses a `[0, 2]` logarithmic cohesion multiplier so repeated traits can actively boost a metadata lane before final normalization. This prevents a constant shape lane from becoming more dominant merely because non-shape lanes are only penalized.

```text
repetition_ratio =
  max_trait_anchor_count / anchor_count

metadata_lane_cohesion =
  2 * log1p(9 * repetition_ratio) / log1p(9)
```

Only traits with `max_trait_anchor_count >= 2` count. If no trait repeats, `metadata_lane_cohesion = 0.0`. The logarithmic curve makes strong-but-imperfect agreement close to full agreement: for example, `4/5` anchor agreement should be treated as nearly as meaningful as `5/5`.

Examples with 3 anchors:

```text
all 3 share Christopher Nolan:
  repetition_ratio = 3 / 3 = 1.00
  director cohesion = 2.00

2 of 3 share Pixar:
  repetition_ratio = 2 / 3 = 0.67
  studio cohesion = 1.69

no repeated source material type:
  source cohesion = 0.00
```

The metadata lane's final scoring impact is:

```text
metadata_lane_contribution =
  normalized_lane_weight[lane] * candidate_lane_score
```

Where `normalized_lane_weight[lane]` is derived from metadata cohesion and then normalized with the other multi-anchor lane weights.

### Multi-Anchor Lane Weights

For multi-anchor search, lane weights should come from shared signal strength:

- Vector spaces: `75%` measured vector cohesion, `25%` single-anchor base prior.
- Shape lane: overall `shape` lane raw weight stays at the single-anchor base weight. Vector cohesion changes the internal vector-space mix inside shape, not the overall shape lane weight.
- Metadata lanes: repeated traits only. If a lane has no repeated trait across anchors, it should not receive a consensus boost.
- Quality lane: active only when the anchor set repeatedly lands in the same meaningful bucket (`cult_garbage` or `prestige`). No shared extreme bucket means no quality boost.
- Studio lane: active only when repeated studio/company traits are in the curated meaningful similarity studio/company list.
- Source lane: active when source material types repeat across anchors.
- Franchise lane: active when lineage/universe/franchise traits repeat across anchors.
- Director lane: active when directors repeat across anchors, even though there is no director anchor type.

Raw lane weights:

```text
shape_raw     = 0.60
director_raw  = 0.12 * director_cohesion
franchise_raw = 0.12 * franchise_cohesion
studio_raw    = 0.06 * studio_cohesion
source_raw    = 0.04 * source_cohesion
quality_raw   = 0.06 * quality_cohesion
```

Each metadata-lane cohesion value uses the `[0, 2]` logarithmic multiplier above, with `0.0` when no trait repeats.

After deriving raw lane weights, normalize them to sum to `1.0`, same as single-anchor search.

### Multi-Anchor Decision Rationale

This design intentionally favors shared traits over individual-anchor preservation.

Rejected approach:

```text
retrieve candidates
then compute similarity from each candidate to every anchor
then blend centroid / mean / min / coverage scores
```

Reasons rejected:

- It makes retrieval/rerank heavier than needed.
- It undermines the speed advantage of nearest-neighbor search.
- It can over-preserve individual anchor identities when the user likely wants the overlap.
- It introduces more knobs (`mean`, `min`, `coverage`) without clear need for V1.

Chosen approach:

```text
build consensus vectors for vector spaces
search each consensus vector once
search metadata per anchor
boost metadata candidates by repeated matches
normalize lane weights
score and weave as usual
```

This keeps multi-anchor search explainable: "these movies agree most strongly in these vector spaces and repeat these discrete traits, so candidates matching those shared signals rank higher."

## Explicitly Out of Scope

These were considered and rejected for this flow:

- **Genre cluster** — already implicit in vectors; explicit weighting adds little.
- **Budget / production scale** — partially captured by the production vector and quality bucketing.
- **Lead actor overlap** — too weak (DiCaprio in Titanic is not "like Inception").
- **Same composer / DP** — real signal but too sparse to be worth the complexity.
- **Era / decade, country / language** — captured by vectors.
- **Exact source author matching** — not currently available in the DB; use source-of-inspiration categories/patterns instead.

## Open Questions

- **Cult signal robustness.** Without critic-vs-audience data, how reliably can reception-vs-popularity distinguish true cult films from mainstream guilty pleasures? Current thresholds are `reception_score <= 45` and `mv_popularity_percentile.percentile >= 0.89`, but vector matching is expected to carry the slack.
- **Studio/company curation validation.** Validate the current high/moderate studio list and era windows against catalog examples and search-result quality.
- **Multi-anchor cohesion validation.** Validate the current vector and metadata cohesion mappings against known coherent and incoherent anchor sets.

---

# V2 Planned Changes

V2 is a structural rework triggered by analysis of V1 output across 20 single-anchor + 12 multi-anchor test cases. Failure patterns: source lane false positives (Stephen King, heist), franchise lane swamping tone-mismatched candidates (Barbie, single-anchor LotR), studio lane creating exposure for low-shape on-brand noise (Get Out, Pulp Fiction), middle-bucket anchors getting no quality signal (Inception, Titanic, Oppenheimer), and shape getting crowded out of multi-anchor cohesive sets (LotR trilogy where shape weight dropped to 0.43 despite vector cohesion of ~0.96).

## V2 Vector-Space Weights

Drop the tier groupings. Single-anchor and multi-anchor get separate base profiles.

**Single-anchor base weights:**

```text
plot_analysis:        1.00
viewer_experience:    1.00
watch_context:        0.75
narrative_techniques: 0.55
reception:            0.55
dense_anchor:         0.45
production:           0.30
plot_events:          0.25
```

Rationale: keep `plot_analysis` and `viewer_experience` anchored. Raise `narrative_techniques` (storytelling-style binds Tarantino, Nolan). Lower `production` (it locks Titanic to ship-disasters and Oppenheimer to atomic-subject docs). Keep `plot_events` lowest.

**Multi-anchor base weights (unchanged from V1):**

```text
plot_analysis:        1.00
viewer_experience:    1.00
watch_context:        0.65
production:           0.65
reception:            0.65
dense_anchor:         0.65
narrative_techniques: 0.35
plot_events:          0.35
```

Multi-anchor keeps the original tiered base because cohesion does the heavy lifting and the V1 results on cohesive sets (Pixar, Ghibli, war films, Tarantino) were already strong.

## V2 Shape Lane Scaling (Multi-Anchor)

In V1 the shape lane raw weight was constant at `0.60` while metadata lanes could grow to `0.12 * 2.0 = 0.24` each via the log cohesion multiplier. With four maxed-out metadata lanes, shape's normalized weight could drop from 0.60 to 0.43 even when vector cohesion was near-perfect — a directional asymmetry that crowded out shape exactly when it was most reliable.

V2 fix: shape scales with an expanded cohesion measure that can both boost shape on coherent sets and penalize it on incoherent ones.

```text
# Per-space cohesion_weight stays clamped to [0.10, 1.00] for vector-space mixing
# (negative weights inside the shape mix don't make sense). For shape lane scaling
# we use a separate measure with an expanded lower bound.

mean_pairwise_cosine        = mean(avg_pairwise_cosine across active spaces)
shape_cohesion_signal       = clamp((mean_pairwise_cosine - 0.55) / 0.30, -0.40, 1.00)
shape_raw                   = 0.60 * (1 + shape_cohesion_signal)         # range [0.36, 1.20]
```

Reference points:
- `mean_pairwise_cosine = 0.55` → signal = 0.0, shape_raw = 0.60 (V1 default).
- `mean_pairwise_cosine = 0.85` → signal = 1.0, shape_raw = 1.20 (LotR trilogy ≈ here).
- `mean_pairwise_cosine = 0.43` → signal = -0.40, shape_raw = 0.36 (chaotic mixed bag ≈ here).

Penalizing shape when anchors disagree is desirable: the centroid is in noise and shape's output should be down-weighted before metadata lanes pick up whatever signal still exists. The low-cohesion fallback below catches the worst cases entirely; the expanded range handles the messy middle.

## V2 Source Lane (Inverse-Frequency Weighting)

Source-material-type matching is the most broken lane in V1: any two novel-based movies share `novel` as a source type and get a full 1.0 source score, contributing meaningfully to combined score and surfacing irrelevant matches (Stephen King horror trio returning generic novel-based horror; heist trio returning unrelated thrillers via shared book sources).

V2 replaces binary matching with IDF over source-material-types:

```text
# Computed once at startup from a materialized view of trait frequencies.
idf(trait) = log(N / df(trait)) / log(N)              # normalized to [0, 1]

source_score = max(idf(t) for t in shared_traits)      # max, not sum
```

Common types like `novel` collapse to ~0.20 contribution; rare types (`video_game`, `fairy_tale`, `stage_play`, `comic_book`) stay near 1.0. Use `max` over shared traits so a single rare match dominates and two common-tag overlaps don't add up to a high score.

Lane base weight stays at `0.04`. The `source_material` anchor adjustment drops from `+0.14` to `+0.08`.

## V2 Studio Lane (Multiplicative)

V1 studio lane created exposure for on-brand but low-shape candidates (Oculus and Hush surfacing for Get Out via shared Blumhouse). Even with the `shape_score >= 0.55` floor, once a candidate cleared the floor, the studio additive contribution pushed it above better non-studio shape matches.

V2 makes studio multiplicative on shape-qualifying candidates instead of an independent additive lane:

```text
if studio_match and shape_score >= 0.60:
    combined_score *= 1.0 + 0.10 * studio_score
```

Studio is removed from the additive lane sum entirely. The `studio_lineage` anchor type stays as a flag for debug visibility but stops adjusting lane weights.

## V2 Director Lane (Auteur Prior + Optional Anchor Type)

V1 used binary 1.0 director matching, which over-credited matches on obscure directors and under-credited matches on auteurs whose name carries strong meaning. V2 replaces that with a per-director auteur prior, then introduces an optional `director_signature` anchor type.

**Director strength materialized view:**

```text
For each director_term_id with >= 2 films in the catalog:
  mean_pop_pct  = mean(mv_popularity_percentile across director's films)
  mean_recep    = mean(reception_score / 100 across director's films)
  raw_strength  = 0.8 * mean_pop_pct + 0.2 * mean_recep

director_strength = percentile_rank(raw_strength) across all qualifying directors    # [0, 1]
```

Directors with only one cataloged film are excluded from the percentile-rank pool and never receive a director-lane match (a single film is the anchor itself; there are no other candidates to match through this lane). View refresh follows the daily ingestion cadence already in place — no separate schedule needed.

**Director lane score:**

```text
director_score = director_strength               # if anchor and candidate share director
                                                 # else 0
```

**`director_signature` anchor type (new):**

Active when the anchor's director has `director_strength >= 0.80`.

```text
director_signature:
  director += 0.10
  shape    -= 0.04
```

Multiple anchor types can stack as today. The 0.80 threshold restricts the anchor type to genuine top-tier auteurs (Tarantino, Nolan, Scorsese, Spielberg, Miyazaki, the Coens, Anderson, etc.) rather than any above-median director.

## V2 Franchise Lane (Subgroup Gating + Confidence Prior)

V1 franchise lane treated all lineage edges equally and let subgroup-only matches (e.g., "Original Star Wars Trilogy" subgroup, generic franchise subgroups) ride at full 1.0. Combined with the franchise_dominant boost, this caused the Barbie failure (direct-to-DVD Barbie kids' films dominating top 5) and the LotR-single failure (only Hobbit films, no fantasy adjacency).

**Step 1 — Subgroup gating:**

```text
score = 1.00   if anchor_lineage ∩ candidate_lineage
score = 1.00   if anchor_subgroup ∩ candidate_subgroup
                  AND (anchor_universe ∩ candidate_universe
                       OR anchor_lineage ∩ candidate_lineage)
score = 0.85   if anchor_universe ∩ candidate_universe
score = 0.40   if anchor_subgroup ∩ candidate_subgroup     # subgroup-only fallback
score = 0.00   otherwise
```

**Step 2 — Franchise confidence materialized view** (per lineage_entry_id, refresh nightly):

```text
franchise_confidence  = 0.8 * mean(popularity_percentile)
                      + 0.2 * mean(reception_score / 100)
franchise_consistency = 1 - normalized_stddev(0.8*pop + 0.2*recep)
```

**Step 3 — Confidence-driven behavior:**

```text
if franchise_confidence >= 0.65 and franchise_consistency >= 0.6:
    # High-confidence franchises (LotR, MCU pre-2019, Pixar):
    # additive lane as today, with raised shape gate.
    additive_contribution = lane_weight[franchise] * franchise_score
    shape_gate            = 0.45        # raised from 0.35 in V1

else:
    # Low-confidence franchises (Barbie, mixed-quality IPs):
    # multiplicative on shape, no additive lane contribution.
    if shape_score >= 0.55:
        combined_score *= 1.0 + 0.10 * franchise_score
```

`franchise_dominant` anchor adjustment stays at `franchise +0.18, shape -0.08` for high-confidence franchises only. Low-confidence franchises do not trigger the anchor adjustment.

## V2 Quality Lane (Unified Three-Mode + Awards)

V1 quality lane was inactive for middle-bucket anchors (the largest segment), giving Inception, Interstellar, Pulp Fiction, Get Out etc. no reception-aware tie-breaking. V2 unifies all three modes into a single quality lane with bucket-specific scoring.

**Bucket detection (unchanged from V1):**

- `cult_garbage`: `reception_score <= 45` AND `popularity_percentile >= 0.89`
- `prestige`: `reception_score >= 85` AND `popularity_percentile >= 0.75`
- `middle`: everything else

**Per-bucket score formulas:**

```text
cult_garbage:
  quality_score =
      0.40 * low_reception_match        # clamp((50 - reception)/30, 0, 1)
    + 0.50 * popularity_match           # clamp((pop_pct - 0.75)/0.20, 0, 1)
    + 0.10 * razzie_match               # 1.0 if any Razzie nom/win else 0

prestige:
  quality_score =
      0.80 * high_reception_match       # clamp((reception - 75)/20, 0, 1)
    + 0.20 * (popularity_or_award)
  + 0.20 * non_razzie_award_match    # additive (clamped to <= 1.0)

middle:
  quality_score =
      0.80 * popularity_percentile
    + 0.20 * (reception_score / 100)
```

`non_razzie_award_match`: `1.0` for major non-Razzie win, `0.75` for nomination, `0.0` otherwise. `popularity_or_award`: `max(clamp((pop_pct - 0.50)/0.30, 0, 1), award_match)`.

**Lane weight per bucket:**

```text
base quality lane weight: 0.06
prestige     anchor adjustment: quality +0.16, shape -0.06    # unchanged
cult_garbage anchor adjustment: quality +0.26, shape -0.10    # unchanged
middle       anchor adjustment: none — runs at base 0.06 (tunable down to 0.04 in testing)
```

For middle bucket the lane is now always active as a soft notability prior, without dominating shape.

**Multi-anchor award handling:** if multiple anchors share a specific award ceremony win/nomination, treat the ceremony as a metadata trait so existing metadata-cohesion logic boosts candidates that won that ceremony. Single-anchor uses the per-bucket formula only (no specific-ceremony preference).

## V2 Production Medium (Single-Anchor Only — New Multiplier + Selective Retrieval)

**Scope:** Single-anchor flow only for V2.0. Multi-anchor medium handling is left for the multi-anchor themes work below; in the meantime, multi-anchor candidates are not scored or filtered by medium.

Production medium is a shape-shaping attribute (animated vs live-action affects every other vector dimension). V2 treats it as a multiplier on combined score using a curated similarity table — not pure IDF — because mediums are not equally distant from each other (animation sub-types are closer to each other than to live-action, and the parent ANIMATION tag should match all animated sub-types).

**Complete production-medium tag list** (from `OverallKeyword`):

```text
Pure-medium tags:
  ANIMATION              (#8)    #  6,010 movies — parent / generic animation
  COMPUTER_ANIMATION     (#32)   #    792 movies
  HAND_DRAWN_ANIMATION   (#83)   #  1,591 movies
  STOP_MOTION_ANIMATION  (#185)  #    222 movies
  LIVE_ACTION            (#226)  # ~95% of catalog (auto-stamped complement of ANIMATION)

Animation cross-cuts (imply ANIMATION but mark a sub-style or audience):
  ANIME                  (#9)    #    929 movies — Japanese animation tradition
  ADULT_ANIMATION        (#3)    #    751 movies — animation for adults
  HOLIDAY_ANIMATION      (#92)   #    121 movies — animation, holiday-themed
```

**Anchor and candidate medium sets:**

For each movie, collect every applicable medium tag from the list above. An adult Pixar-style film could have `{ANIMATION, COMPUTER_ANIMATION, ADULT_ANIMATION}`. Coraline would have `{ANIMATION, STOP_MOTION_ANIMATION}`. A live-action drama would have `{LIVE_ACTION}`.

**Medium similarity table** (lookup by `[anchor_tag][candidate_tag]`):

```text
                    LIVE   ANIM   CG     HD     STOP   ANIME  ADULT  HOLIDAY
LIVE_ACTION         1.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
ANIMATION           0.00   1.00   0.90   0.90   0.90   0.90   0.95   0.90
COMPUTER_ANIMATION  0.00   0.90   1.00   0.60   0.50   0.55   0.85   0.85
HAND_DRAWN_ANIM     0.00   0.90   0.60   1.00   0.65   0.85   0.85   0.85
STOP_MOTION_ANIM    0.00   0.90   0.50   0.65   1.00   0.55   0.80   0.80
ANIME               0.00   0.90   0.55   0.85   0.55   1.00   0.80   0.70
ADULT_ANIMATION     0.00   0.95   0.85   0.85   0.80   0.80   1.00   0.55
HOLIDAY_ANIMATION   0.00   0.90   0.85   0.85   0.80   0.70   0.55   1.00
```

Symmetric. ANIMATION (parent) is treated as an "any-animation matches" row: a candidate that's hand-drawn anime still scores 0.90 vs. an ANIMATION-only anchor, because the parent tag carries the "this is animated" signal without specifying technique.

LIVE_ACTION matches only LIVE_ACTION at 1.0 and is 0 against everything animated — animated and live-action are categorically different watching experiences.

**Medium score:**

```text
medium_score = max(
    table[a_tag][c_tag]
    for a_tag in anchor_medium_tags
    for c_tag in candidate_medium_tags
)
```

Taking max across all anchor × candidate pairs lets the parent ANIMATION tag absorb sub-type differences (per user direction: don't over-index on the most specific tag).

**Score multiplier:**

```text
medium_multiplier = 0.85 + 0.15 * medium_score          # range [0.85, 1.00]
combined_score *= medium_multiplier
```

A perfect medium match leaves combined score unchanged; full mismatch (live-action anchor vs. animation candidate) drops it by 15%. Cross-medium-but-related (CG anchor vs. stop-motion candidate, score 0.50) gets a partial penalty (multiplier ≈ 0.925).

**Selective candidate retrieval** (rare-medium recall repair):

If the anchor has at least one medium tag with `medium_idf >= 0.50` (i.e., COMPUTER_ANIMATION, HAND_DRAWN_ANIMATION, STOP_MOTION_ANIMATION, ANIME, ADULT_ANIMATION, HOLIDAY_ANIMATION, or ANIMATION-as-rarest), add candidates sharing that tag as a retrieval lane in addition to scoring. This ensures stop-motion anchors surface other stop-motion films from the catalog even if their vectors don't cluster nearby.

For LIVE_ACTION-only anchors, no candidate-retrieval expansion happens — live action is the catalog default and adding all live-action films as candidates would explode the pool with no signal.

**Medium IDF table** (precomputed once at startup, used for the retrieval gate):

```text
medium_idf(LIVE_ACTION)            ≈ 0.04
medium_idf(ANIMATION)              ≈ 0.27
medium_idf(ANIME)                  ≈ 0.42
medium_idf(ADULT_ANIMATION)        ≈ 0.43
medium_idf(HAND_DRAWN_ANIMATION)   ≈ 0.37
medium_idf(COMPUTER_ANIMATION)     ≈ 0.43
medium_idf(HOLIDAY_ANIMATION)      ≈ 0.55
medium_idf(STOP_MOTION_ANIMATION)  ≈ 0.54
```

Approximate values assuming a ~150K catalog and the per-tag counts above; actual values come from the runtime IDF computation.

## V2 Format Lane (Single-Anchor Only — New Metadata Lane)

**Scope:** Single-anchor flow only for V2.0. Multi-anchor format handling is folded into the multi-anchor themes work below; in the meantime, multi-anchor candidates are not gated or scored by format bucket.

Format (documentary vs short vs narrative feature vs performance) gets its own metadata lane plus weaving rules. Top of the result list stays format-coherent for clear user expectations; tail can mix formats for discovery.

**Format taxonomy:**

```text
documentary:
  DOCUMENTARY                          (#52)   # 7,909
  CRIME_DOCUMENTARY                    (#41)   #   243
  FAITH_AND_SPIRITUALITY_DOCUMENTARY   (#61)   #    86
  FOOD_DOCUMENTARY                     (#72)   #    50
  HISTORY_DOCUMENTARY                  (#90)   #    81
  MILITARY_DOCUMENTARY                 (#116)  #    63
  MUSIC_DOCUMENTARY                    (#122)  #   478
  NATURE_DOCUMENTARY                   (#125)  #   100
  POLITICAL_DOCUMENTARY                (#134)  #    84
  SCIENCE_AND_TECHNOLOGY_DOCUMENTARY   (#158)  #    52
  SPORTS_DOCUMENTARY                   (#179)  #   182
  TRAVEL_DOCUMENTARY                   (#209)  #    66
  DOCUDRAMA                            (#51)   #   703  — dramatized doc
  TRUE_CRIME                           (#210)  #   832  — usually doc-style

mockumentary:
  MOCKUMENTARY                         (#117)  #    80

short:
  SHORT                                (#163)  # 5,646

performance:
  CONCERT                              (#33)   #    86
  STAND_UP                             (#181)  #   123

news:
  NEWS                                 (#126)  #   172

tv_format (sparse — group together for completeness):
  REALITY_TV                           (#147)  #    11
  PARANORMAL_REALITY_TV                (#129)  #     1
  BUSINESS_REALITY_TV                  (#24)   #     1
  GAME_SHOW                            (#76)   #     1
  TALK_SHOW                            (#195)  #     3
  SOAP_OPERA                           (#172)  #     1
  SITCOM                               (#167)  #    11
  COOKING_COMPETITION                  (#36)   #     1
  SKETCH_COMEDY                        (#168)  #    77

narrative_feature (default):
  No format tag from any group above — implicit category covering the bulk of the catalog.
```

A movie is bucketed into the most specific group present in its tags, with priority: `mockumentary > performance > news > tv_format > short > documentary > narrative_feature`. (Mockumentary takes priority over documentary because the conventions are documentary-style but the experience is fictional.)

**Format lane score:**

```text
format_score = 1.0  if anchor_format_bucket == candidate_format_bucket
             = 0.0  otherwise
```

**Lane weights:**

```text
single-anchor base: 0.04
multi-anchor base:  0.04 (subject to existing log cohesion multiplier when anchors share a non-default format)
```

No per-anchor-type adjustment. The lane is always scored; its impact depends on weight.

**Weaving rules** (applied during top-section assembly):

- Top 5 results MUST share the anchor's format bucket.
- Positions 6–10 may include cross-format candidates that survived scoring.
- For multi-anchor: top 5 must share the format bucket repeated by ≥ 2 anchors. If anchors disagree (no repeated bucket), drop the top-5 constraint entirely and let combined score decide.

This handles the Oppenheimer / Barbie failure modes: documentary candidates can still appear in positions 6–10 (so a user discovering "I might also like the Oppenheimer biographical doc" still sees it), but they can't crowd out the actual film's narrative-feature adjacents from the top of the page.

## V2 Themes Lane (Multi-Anchor Only)

A new metadata lane that captures the **shared semantic profile** of an anchor set: keywords, concept tags, and genres that recur across anchors. Vectors blur category boundaries; an explicit lane on these discrete traits gives the multi-anchor flow a way to lock onto specific shared signals (heist, war epic, dark fantasy, slow-burn psychological horror) without depending on the centroid landing in the right neighborhood.

Skipped for single-anchor: the anchor's own traits are already encoded by vectors plus the new medium and format signals; adding themes there would double-count without solving a known failure case.

**Trait sources:**

```text
overall_keywords  (~225 tags from OverallKeyword,
                   minus the 30 country/language tags handled separately
                   by the country/language coherence multiplier below)
concept_tags      (25 binary tags from concept_tags)      — narrow & specific
tmdb_genres       (~20 broad genres on movie_card)        — broad
```

All three sources are concatenated into one trait pool per movie. Combining them into a single lane avoids triple-counting overlapping signals (`HORROR` shows up as both a `tmdb_genre` and an `overall_keyword`); the IDF step below handles cardinality imbalance automatically.

Country/language tags (KOREAN, FRENCH, JAPANESE, HINDI, etc. — the 30 nation/tradition tags in `OverallKeyword`) are pulled out of the themes pool and handled by the dedicated multiplier below, so they get a stronger fixed effect than IDF would naturally assign.

**Anchor-set repeated traits:**

```text
repeated_traits = {
  trait
  for trait in union(anchor_traits)
  if count(anchor_traits where trait in anchor) >= 2
}
```

Only traits appearing in at least 2 anchors qualify. Singletons are dropped — they're per-anchor identity, not shared signal.

**Candidate score:**

```text
shared        = candidate_traits ∩ repeated_traits
score_numer   = sum(idf(t) for t in shared)
score_denom   = sum(idf(t) for t in repeated_traits)
themes_score  = score_numer / score_denom    if score_denom > 0 else 0
```

IDF uses the same `log(N/df) / log(N)` formula as the source lane. Common tags (`DRAMA` at 56,798 movies, `COMEDY` at 33,054, `THRILLER` at 23,577) get near-zero weight; niche tags (`FOLK_HORROR`, `KAIJU`, `CYBERPUNK`, `SAMURAI`) carry the actual signal.

**Lane weight:**

```text
base weight: 0.06    # slightly above source/format because it's broader and integrative

# Existing multi-anchor metadata-cohesion mechanism applies:
themes_cohesion = 2 * log1p(9 * max_trait_repetition_ratio) / log1p(9)
themes_raw      = 0.06 * themes_cohesion
```

When all anchors share many repeated traits → high cohesion → meaningful lane weight. When few or no traits repeat → cohesion → 0 → lane drops out automatically. This gives the "doesn't dominate unless other lanes are also weak" property without a special activation rule.

**Active anchor type:** none. The lane is purely cohesion-driven.

**Effect on the V1 multi-anchor failures:**

- Best Picture trio (Godfather/Schindler/12YS): repeated traits like `DRAMA`, `HISTORY`, `BIOGRAPHY`, `EPIC` combined with prestige-tier concept tags pull mafia/historical-American adjacents (Goodfellas, There Will Be Blood) back up against the Holocaust-pulled centroid.
- Heist trio: repeated `HEIST`, `CRIME`, `THRILLER`, `CAPER` lock in actual heist films vs. adjacent crime thrillers.
- Stephen King trio: repeated `HORROR`, `SUPERNATURAL_HORROR`, `PSYCHOLOGICAL_HORROR` plus relevant concept tags carry the signal that source-material IDF alone can't (since "novel" was the only repeated source type).
- Chaotic mixed bag: no traits repeat → themes_cohesion = 0 → lane drops out → low-cohesion fallback handles the case.

## V2 Country / Language Coherence (Multi-Anchor Only)

A multi-anchor multiplier on combined score that pins results to the same national/linguistic tradition as the anchor set. Vectors capture this weakly; an explicit signal prevents Bollywood matches from surfacing for an American-classics anchor set, or US matches drowning out Korean cinema for a Korean-cinema anchor set.

**Country/language taxonomy:**

The 30 nation/tradition tags from `OverallKeyword`:

```text
ARABIC, BENGALI, CANTONESE, DANISH, DUTCH, FILIPINO, FINNISH, FRENCH,
GERMAN, GREEK, HINDI, ITALIAN, JAPANESE, KANNADA, KOREAN, MALAYALAM,
MANDARIN, MARATHI, NORWEGIAN, PERSIAN, PORTUGUESE, PUNJABI, RUSSIAN,
SPANISH, SWEDISH, TAMIL, TELUGU, THAI, TURKISH, URDU
```

Plus a synthetic `US_DEFAULT` bucket for any movie carrying none of these tags (covers ~75-80% of the catalog: US/UK/Canadian/Australian English-language productions). Treating this as a real bucket (rather than "no signal") is what gives the "boost American films when anchors are American" behavior: candidates with no country tag are explicitly checked against the anchor consensus.

**Per-movie country set:**

```text
country_set(movie) = movie.country_language_tags  if non-empty
                   = {US_DEFAULT}                  otherwise
```

A film carrying multiple country/language tags (e.g., a French-Italian co-production with both `FRENCH` and `ITALIAN`) keeps both — match if anchor and candidate share any.

**Consensus detection:**

```text
country_repetition_count(c) = number of anchors with c in country_set(anchor)
consensus_countries         = {c : country_repetition_count(c) >= 2}
```

If no country/language is shared by ≥ 2 anchors, the multiplier is inactive (anchors are too cosmopolitan to call a tradition).

**Multiplier:**

```text
if consensus_countries:
    if candidate_country_set ∩ consensus_countries:
        combined_score *= 1.10        # candidate matches consensus
    else:
        combined_score *= 0.85        # candidate is outside consensus tradition
```

Three American classics → consensus = `{US_DEFAULT}` → US/UK/Canadian candidates get +10%, Bollywood/Korean/etc. get -15%. Three Korean cinema picks → consensus = `{KOREAN}` → Korean candidates get +10%, US candidates get -15%. Mixed set (one French, one Korean, one US) → no consensus → multiplier inactive.

The multiplier is intentionally smaller than the medium multiplier (0.85 vs. 0.85 floor for both, but country/language tops out at 1.10 instead of 1.00) so it shapes the result list without overwhelming shape-driven adjacency.

## V2 Cast Lane (Multi-Anchor Only)

A small new metadata lane based on top-3-billed cast overlap. Single-anchor cast matching was rightly rejected (DiCaprio in Titanic isn't "like Inception"), but for multi-anchor, repeated cast across 2+ anchors is diagnostic — three Tom Hanks dramas, two Frances McDormand prestige pictures, three Jackie Chan vehicles all carry real shared identity.

**Per-anchor traits:**

```text
top_3_cast_ids = top 3 cast member IDs by billing position from movie_card
```

Top 3 only; deeper billing positions create noise (a third-billed lead is meaningful; a tenth-billed character actor is not). Implementation note: verify `movie_card` has cast member IDs ordered by billing — if not, plumb that ordering during the cast-lane data extraction step.

**Lane scoring (uses the existing repetition-count formula):**

```text
candidate_cast_score = matched_anchor_count_for_lane / anchor_count
```

Same shape as the existing director/franchise/source/quality lanes. A candidate appearing in all 3 anchors' top-3 cast lists scores 1.0; a candidate in 1 of 3 scores 0.33.

**Lane weight:**

```text
base weight: 0.03    # small — gets amplified by cohesion when cast is the strongest signal

cast_cohesion = 2 * log1p(9 * max_cast_repetition_ratio) / log1p(9)
cast_raw      = 0.03 * cast_cohesion
```

The intentionally-small base weight is the user's design: cast should be a small contribution in normal cases (where it overlaps with the obvious shape adjacency) but get amplified to a real signal when the anchor set's strongest cohesion is repeated cast members. The standard log multiplier handles that amplification automatically.

**Active anchor type:** none. Pure cohesion-driven.

## V2 Specific-Award Lane (Multi-Anchor Only)

The existing quality lane buckets anchors as `cult_garbage` / `prestige` / `middle` but can't distinguish "all three won Best Picture" from "all three are well-reviewed crowd-pleasers." A new lane on `movie_awards.category_tag_ids` captures the specific accolade profile shared across anchors — Best Picture cohort vs. Acting cohort vs. festival-jury cohort vs. Razzie cohort.

The lane leans on the three-level taxonomy already in `category_tag_ids`:

```text
L0 (leaves, ids 1..62)        — specific concepts (LEAD_ACTOR, BEST_PICTURE_DRAMA, WORST_DIRECTOR)
L1 (mids, ids 100..199)       — meaningful rollups (LEAD_ACTING covers actor + actress;
                                 BEST_PICTURE_ANY covers all picture genre splits)
L2 (groups, ids 10000..10006) — seven top-level buckets
                                 (ACTING, DIRECTING, WRITING, PICTURE, CRAFT, RAZZIE,
                                  FESTIVAL_OR_TRIBUTE)
```

Every award row already stores leaf + ancestors, so an anchor's full tag pool is just the union of `category_tag_ids` across all its award rows. Three anchors that won Best Actor + Best Actress + Best Supporting Actor share `LEAD_ACTING` (L1) and `ACTING` (L2) but not any L0 leaf — the lane should still recognize that as a meaningful acting-cohort signal, just less strongly than three exact `BEST_PICTURE` matches.

**Per-anchor traits:**

```text
anchor_award_tags(i) = union of category_tag_ids from all award rows for anchor i
                       (wins and nominations lumped together for V2.0;
                        win-vs-nomination split deferred until results show
                        nominee-only matches riding through inappropriately)
```

**Most-specific repeating level:**

```text
counts          = {tag_id : count(anchors containing tag_id)}
repeated_tags   = {tag_id : counts[tag_id] >= 2}

if no repeated_tags:
    lane drops out (cohesion = 0)
else:
    most_specific_level = min(tag.level for tag in repeated_tags)
```

**Lane cohesion (with specificity discount):**

```text
specificity_factor = {0: 1.0, 1: 0.6, 2: 0.3}[most_specific_level]
best_ratio         = max(counts[t] / N for t in repeated_tags
                                          if t.level == most_specific_level)
lane_cohesion      = specificity_factor
                   * 2 * log1p(9 * best_ratio) / log1p(9)
```

Three exact `BEST_PICTURE` wins (L0, ratio 1.0): cohesion = 1.0 × 2.0 = **2.0** (max).
Three different acting awards repeating only at L1 `LEAD_ACTING` (ratio 0.67): cohesion = 0.6 × 1.69 ≈ **1.01** (moderate).
Three disjoint award groups repeating only at L2 (ratio 1.0): cohesion = 0.3 × 2.0 = **0.6** (mild).

**Candidate score (tier-weighted):**

```text
tier_weight(t) = {0: 1.00, 1: 0.50, 2: 0.20}[t.level]

shared      = candidate_award_tags ∩ repeated_tags
numer       = sum(tier_weight(t) for t in shared)
denom       = sum(tier_weight(t) for t in repeated_tags)
award_score = numer / denom        if denom > 0 else 0
```

The denominator includes every repeated tag across all three levels, so a candidate matching only at the L2 group level gets partial credit but is dominated by candidates that match the same L0 leaf as the anchors.

**Lane weight:**

```text
base weight: 0.04
specific_award_raw = 0.04 * lane_cohesion        # uses the lane_cohesion above
```

**Active anchor type:** none. Pure cohesion-driven, parallel to the themes and cast lanes.

**Effect on V1 failure cases:**

- Best Picture trio (Godfather/Schindler/12YS): all three won Oscar Best Picture → L0 `BEST_PICTURE` repeats at 1.0 → strong lane → other Best Picture winners (Goodfellas was nominated, Forrest Gump won, etc.) climb back up against the Holocaust-pulled centroid.
- Razzie trio (Sharknado/The Room/Birdemic): all three nominated/won at Razzies → L0 `WORST_PICTURE` likely repeats; L2 `RAZZIE` definitely repeats → other Razzie titles get a real boost vs. generic cult.
- War film trio: anchors share PICTURE-tier awards but at different L0 leaves → falls back to L1/L2 specificity → mild boost for prestige-war titles, doesn't dominate (which is right — the war genre signal is what should dominate, not the award signal).

When the user provides a chaotic anchor set (Toy Story + Godfather + Sharknado in the V1 batch) the centroid lands in noise and we currently return arbitrary matches.

**Trigger:**

```text
mean_vector_cohesion < 0.35
AND no metadata lane has cohesion >= 1.0
```

**Fallback behavior:**

1. Run independent single-anchor similarity for each anchor with `limit = ceil(final_limit * 1.2 / N)`.
2. Interleave by rank (round-robin), break ties by combined score.
3. Return results without any low-cohesion flag — UI presents them identically.

## V2 Debug Payload Changes

Add per-anchor active-anchor-types to the multi-anchor debug payload. Useful for diagnosing centroid drift in cases like the Best Picture trio (where Schindler's List + 12 Years a Slave dominated the centroid and pushed The Godfather's adjacents off the page).

```text
SimilarMoviesDebug:
  ...existing fields...
  per_anchor_active_anchor_types: dict[int, list[AnchorType]]   # multi-anchor only
```

## V2 Implementation Order

1. **Style soft-signal + Production medium IDF** — cheap, big wins on Oppenheimer / Barbie / Titanic.
2. **Source lane IDF rework** — kills Stephen King / heist false positives.
3. **Franchise restructure** (subgroup gating + confidence prior + multiplicative-when-low-confidence).
4. **Shape-raw scales with vector cohesion** + V2 single-anchor base vector weights.
5. **Unified quality lane** with per-bucket popularity/reception split + middle-bucket activation.
6. **Director auteur prior** + optional `director_signature` anchor type.
7. **Low-cohesion fallback** + debug payload changes.
8. **Genre similarity graph** — deferred until V2 results are evaluated.

---

# V3 Planned Changes

V3 is the next structural rework, motivated by end-to-end V2 testing on the same 20 single-anchor + 12 multi-anchor benchmark (full audit in [similar_movies_v2_results.md](similar_movies_v2_results.md), detailed planning in [similar_movies_v3_plan.md](similar_movies_v3_plan.md)). V2 fixed several V1 problems but introduced or exposed new ones:

- Format buckets contained content tags (`DOCUDRAMA`, `TRUE_CRIME`, `SKETCH_COMEDY`) that misclassified prestige biopics and Monty Python features.
- Production medium matrix contained audience tags (`ADULT_ANIMATION`, `HOLIDAY_ANIMATION`) that produced wrong scores like Persepolis vs. Sita Sings the Blues at 1.0.
- Country / language signal was multi-anchor only — single-anchor Barbie returned a Telugu film at #1.
- Format top-5 lock blocked shorts from positions 1–5 but let them flood positions 6–10.
- Medium multiplier `0.85 + 0.15 * score` was too soft for live↔animation crossings — animated Batman entered Dark Knight's top 10.
- Franchise `franchise_consistency >= 0.6` gate silenced Star Wars (the most franchise-coherent anchor in the benchmark) due to the sequels-and-spinoffs tail.
- Director lane fired off `mv_director_strength` (popularity percentile) — surfaced Lucas's *American Graffiti* for *Star Wars* and Spielberg cross-over films for unrelated anchors.
- Single-anchor flow had no themes / cast / specific-award signal — Barbie's `FEMALE_LEAD`, `SATIRE`, `BARBIE_DOLL` tags contributed nothing.

V3 addresses each through a mix of registry edits, multiplier recalibrations, lane reworks, and one new lane.

## V3 Categorization Fixes (Audited Registries)

### Format registry

Removed from format buckets (audit details in `similar_movies_v3_plan.md` §1.1):
- `DOCUDRAMA` — 27 of 30 prestige samples were narrative biopics (Schindler's List, Spotlight, Oppenheimer, Zero Dark Thirty).
- `TRUE_CRIME` — 28 of 30 prestige samples were narrative crime dramas (GoodFellas, Killers of the Flower Moon).
- `SKETCH_COMEDY` — 13 of 13 prestige samples were narrative features (Monty Python and the Holy Grail, And Now for Something Completely Different).

These tags now feed the themes lane via the keyword pool, so they remain available as content-similarity signals without forcing format misclassification.

### Production medium registry

Removed from `MEDIUM_TAG_IDS` and shrunk the matrix from 8×8 to 6×6 (audit details §1.2):
- `ADULT_ANIMATION` — co-occurs with the actual technique tag in 17/20 prestige cases (Persepolis = HD, Mahavatar Narsimha = CG). Audience signal, not technique.
- `HOLIDAY_ANIMATION` — purely thematic (Christmas), spans HD / stop-motion / CG techniques. Theme signal, not technique.

The remaining matrix covers `LIVE_ACTION`, `ANIMATION` (parent), `COMPUTER_ANIMATION`, `HAND_DRAWN_ANIMATION`, `STOP_MOTION_ANIMATION`, `ANIME` — the actual production techniques.

## V3 Format Weave — Harsh Downrank for Shorts

Replace V2's top-5 format lock (which only blocks shorts from positions 1–5) with a combined-score multiplier plus a structural cap, so shorts cannot flood any part of the top 10 when the user clearly isn't looking for shorts.

```text
# Per-candidate multiplier on combined score
if candidate.format_bucket == "short" and active_anchor_bucket != "short":
    combined_score *= 0.30          # severe — pushes Partysaurus Rex 0.34 → 0.10

# Hard cap during weaving — applied across the full top 10, not just top 5
short_count = 0
for candidate in ranked_candidates:
    if candidate.format_bucket == "short":
        if short_count >= 1:
            skip                      # at most 1 short anywhere in the top 10
        short_count += 1
    add to top_10
```

Multi-anchor "moderate cohesion" definition: the dominant format bucket = the bucket ≥ 50 % of anchors share. If shorts clear 50 %, treat shorts as the dominant bucket → no penalty (in fact slight 1.10× upweight within shorts). If no bucket clears 50 %, drop the constraint entirely (no shorts penalty).

Short-anchor case (single-anchor with anchor bucket = `short`): apply a soft top-1 lock (require at least one short in top 1, then let features fill from 2 onward). The catalog's shorts skew toward franchise-tied tails (Pixar shorts → Pixar features), so a fully strict lock would block legitimate adjacent features.

## V3 Medium Multiplier — Piecewise (Cross-Category vs. Within-Category)

Replace V2's `0.85 + 0.15 * medium_score` (range `[0.85, 1.00]`, too soft for live↔animation crossings) with a piecewise function that distinguishes categorical mismatch from within-category technique differences.

```python
def medium_multiplier(anchor_tags, candidate_tags):
    if not anchor_tags or not candidate_tags:
        return 1.0
    score = medium_score(anchor_tags, candidate_tags)
    if score == 0.0:
        # Live-action vs. animation — categorical mismatch.
        return 0.65          # 35% penalty
    # Within-category crossings (CG vs stop-motion, anime vs HD, etc.):
    return 0.85 + 0.15 * score  # V2 formula preserved
```

Effects:
- LIVE ↔ ANIM (score 0.0): multiplier `0.65`. Animated Batman drops from raw franchise/source contributions of ~0.50 to ~0.33 — likely below shape-only adjacents.
- CG vs STOP_MOTION (score 0.50): multiplier `0.925` (unchanged from V2).
- ANIME vs HD (score 0.85): multiplier `0.978` (unchanged from V2).
- Perfect medium match: `1.00` (unchanged).

## V3 Country / Language — Extend to Single-Anchor and Recalibrate

V2 country/language coherence was multi-anchor only. V3 extends to single-anchor (anchor's own `country_set` becomes the consensus, same code path with `n == 1`) and recalibrates both flows.

Calibrated multipliers (single-anchor and multi-anchor):
```text
match boost:        1.10 → 1.05    # over-rewarding match was creating its own noise
mismatch penalty:   0.85 → 0.75    # too soft — Barbie's Swag (Telugu, 0.70) survived V2's
                                   #   0.85 (= 0.595, still beat Poor Things at 0.56);
                                   #   0.75 → 0.525 drops cleanly below
```

Calibration math (Barbie case): `0.70 * 0.75 = 0.525 < Poor Things 0.56`. Cross-tradition films now consistently exit the top 3 unless they decisively beat in-tradition adjacents — *not* a hard exclusion.

Edge cases (unchanged): co-production support stays multi-tag (any anchor-tag overlap matches); `US_DEFAULT` continues to lump US/UK/Canada/Australia (Nolan's UK productions don't get penalized for an American anchor).

## V3 Franchise Lane — Structural 2D Matrix

Replace V2's `franchise_consistency >= 0.6` gate (which silenced Star Wars due to the sequels-and-spinoffs tail) with a structural 2D lookup that scores by **role** and **lineage overlap** rather than statistical variance over the franchise's IDs.

Role determination (from `movie_card`):
- **mainline**: single `lineage_entry_ids` AND non-empty `subgroup_entry_ids`.
- **spinoff**: single `lineage_entry_ids` AND empty `subgroup_entry_ids`.
- **crossover**: `len(lineage_entry_ids) >= 2`.

Score lookup `(anchor_role, candidate_role, overlap)` → weight, take **max** across applicable cells:

| anchor role | candidate role | overlap relationship | weight |
|---|---|---|---:|
| mainline | mainline | same lineage + same subgroup | **1.00** |
| mainline | mainline | same lineage, different subgroup | 0.70 |
| mainline | spinoff | same lineage | 0.50 |
| mainline | crossover | shares ≥1 lineage | 0.40 |
| spinoff | spinoff | **same lineage** | **0.85** |
| spinoff | mainline | same lineage | 0.50 |
| spinoff | crossover | shares lineage | 0.40 |
| crossover | crossover | shares ≥1 lineage | **0.85** |
| crossover | mainline | shares lineage | 0.40 |
| crossover | spinoff | shares lineage | 0.40 |
| any | any | same universe, no lineage overlap | 0.30 |
| any | any | disjoint | 0.00 |

Bold rows encode the user's role-consistency principle: same-role + same-lineage gets a bigger boost than mixed-role + same-lineage. Rogue One ↔ Solo (spinoff↔spinoff, same Star Wars lineage) = 0.85; Rogue One ↔ Star Wars 1977 (spinoff↔mainline, same lineage) = 0.50. Avengers ↔ Civil War (crossover↔crossover) = 0.85; Iron Man ↔ Avengers (mainline↔crossover) = 0.40.

The V2 `franchise_confidence` gate is removed entirely. The structural matrix encodes match quality directly — Star Wars's "low consistency" measurement artifact becomes irrelevant because we score by structure, not statistical variance.

## V3 Director Lane — Manual Auteur List Only

Drop the `mv_director_strength` percentile-based boost. V2's "popularity percentile" measure rewarded *prolific generalists* (Spielberg surfaced *E.T.* for *Indiana Jones*; Lucas surfaced *American Graffiti* for *Star Wars*) while under-rewarding *style-coherent auteurs* (Darabont ranked Shawshank only #6 for The Green Mile).

V3 director lane fires only when the anchor's director is on a **manual auteur list** of style-coherent directors (Tarantino-style sharp dialogue, Wes Anderson-style symmetry, Lynch-style oneiric tone, Miyazaki-style hand-drawn pastoral). For non-auteur directors, the lane is silent in single-anchor — the franchise lane and shape vectors already cover the cases where director coherence matters.

**Multi-anchor lane firing**: lane fires when (a) any anchor's director is on the auteur list and the candidate shares it, OR (b) ≥2 anchors share a director (auteur or not) and the candidate shares it. Cohesion across anchors is itself sufficient evidence the user wants that director — the curated list is a prior, not a gate.

**Per-director scoring**: for each director `d` shared between candidate and anchors, with `M_d` = anchors that share `d` and `N` = total anchors:

```
N == 1:                         contribution_d = 0.20
auteur, multi-anchor:           contribution_d = 0.20 + 0.10 * (M_d / N)   # caps at 0.30
cohesion-only (M_d ≥ 2):        contribution_d = 0.10 * (M_d / N)          # caps at 0.10
```

Take `max(contribution_d)` across shared directors. **Multiple curated directors boost independently** — e.g., 4 anchors split 2-2 between WA and PTA both produce M=2/N=4 contributions for their respective candidate matches.

**Cohesion floor (multi-anchor only)**: if `max(M_d / N) >= 0.75` and `shape_score >= 0.30`, set `combined_score = max(combined_score, 0.35)`. The floor activates at 2-of-2, 3-of-3, 3-of-4, 4-of-4, 4-of-5, 5-of-5 — near-unanimous director cohesion, regardless of auteur-list membership.

**Auteur list (60 entries, finalized 2026-05-07)**: full table with normalized DB keys and catalog film counts is in `similar_movies_v3_plan.md` §2.1, verified against `lex.lexical_dictionary` and `lex.inv_director_postings`. `mv_director_strength` is retired as a similar-movies signal.

## V3 Single-Anchor Themes Lane

V2 left the themes lane multi-anchor only. V3 enables it for single-anchor too, using the anchor's own trait pool as the basis (vs. multi-anchor's "repeated traits across anchors").

```text
themes_score(candidate) =
  sum(idf(t) for t in candidate_traits ∩ anchor_traits)
  / sum(idf(t) for t in anchor_traits)
```

Trait pool: same as V2 multi-anchor (overall keywords, concept tags, TMDB genres) **minus** format / medium / country / source tags (handled by their dedicated lanes).

Lane weight: copy V2 multi-anchor's `0.06` base. Active for any single-anchor flow regardless of trait count (no "must repeat ≥2" gate — that's a multi-anchor requirement).

Direct effect on Barbie: `FEMALE_LEAD`, `SATIRE`, `EXISTENTIAL`, `BARBIE_DOLL`, `MUSICAL_NUMBER` become signal. Combined with V3 country/language penalty, the cross-tradition Telugu #1 problem disappears.

## V3 Cast Lane — Generic N-Anchor Bucket-with-Floor (Multi-Anchor)

V2 cast lane had base weight `0.03` cohesion-amplified to ~0.06 raw — not enough for Tom Hanks repeated 3/3 across Big + Polar Express + Toy Story to surface relevant Hanks vehicles past stronger shape matches. V3 keeps the additive lane but adds a **bucket floor**, scaled by the strength of the shared signal.

Generic formula for any number of anchors `N`:

```text
# For each lead (top-3 billing in any anchor), count anchors containing them
M_a           = number of anchors with actor a in top-3 billing
shared_leads  = {a : M_a >= 2}                       # singletons dropped
max_M         = max(M_a for a in shared_leads) or 0
ratio         = max_M / N

# Per-candidate score (additive contribution)
matches    = |candidate.top_3_billing ∩ shared_leads|
cast_score = matches / max(1, len(shared_leads))     # range [0, 1]

# Lane weight
if max_M < 2:
    lane_weight = 0
else:
    lane_weight = 0.05 + 0.10 * ratio                # 2/3 → 0.117, 3/3 → 0.15

# Bucket floor (the weave update)
if max_M < 2 or ratio < 0.5:
    floor = 0
else:
    floor = 0.25 + 0.20 * ratio                      # 2/3 → 0.384, 3/3 → 0.45

# Application — only when shape is decent
if cast_score > 0 and shape_score >= 0.30:
    combined_score = max(combined_score, floor)
```

Tom Hanks 3-of-3 case: `max_M=3, N=3, ratio=1.0` → floor `0.45`, lane weight `0.15`. A Hanks vehicle with shape `0.40` and otherwise modest combined score `0.32` gets pulled up to `0.45` — surfaces in top 10. A Hanks vehicle with already-strong combined `0.55` stays at `0.55` (floor doesn't pull down).

Single-anchor cast: stays disabled. By construction `shared_leads` is empty (one anchor cannot repeat itself).

## V3 Rare-Keyword Lane (NEW)

A new lane parallel to themes. Themes handles aggregate signal across the trait pool (low-rarity matches contribute via the IDF sum). Rare-keyword lane handles **distinctive matches** — high-rarity individuals or rare combos that deserve weave-level attention beyond their additive contribution.

**Pool** (same as themes, with explicit exclusions):
- Include: concept tags + overall keywords NOT in dedicated registries + TMDB genres.
- Exclude: format / medium / country / source / award tags (handled by their lanes).

**Three rarity tiers** (using `mv_trait_idf`):

| Tier | IDF range | Behavior |
|---|---|---|
| **Low** | `< 2.5` | Flows into themes lane. No separate lane visibility. |
| **Moderate** | `2.5 ≤ IDF < 4.5` | Each shared trait adds `IDF × 0.03` to combined score. Visible per-trait. |
| **High** | `IDF ≥ 4.5` | Each shared trait adds `IDF × 0.05`. Counts toward floor trigger. |

Threshold values are starting points — calibrate to roughly p70 / p90 of the actual IDF distribution before locking.

**Floor (the weave update)** — triggers if either:

```text
# Single super-rare hit
max(IDF for shared high-tier traits) >= 5.0

# Rare combo
sum(IDF for shared (high|moderate)-tier traits) >= 7.0
```

When triggered:
```text
floor = 0.40 + 0.05 * (number_of_high_tier_matches)        # capped at 0.55
if shape_score >= 0.30:
    combined_score = max(combined_score, floor)
```

**Single vs. multi**:
- Single-anchor: pool = anchor's own qualifying traits.
- Multi-anchor: pool = traits shared across ≥2 anchors (cohesion intersection).

**Concrete examples**:
- Oppenheimer + candidate sharing `MANHATTAN_PROJECT` (IDF ≈ 6.0): `+0.30` to combined, floor `0.45` activates.
- Memento + candidate sharing `NON_LINEAR_NARRATIVE` + `UNRELIABLE_NARRATOR` + `AMNESIA` (sum IDF ≈ 11.3): `+0.339` total moderate-tier additive, combo floor `0.40` activates.

The lane primarily contributes to score (most matches), but the floor protects truly distinctive matches with weaker shape from dropping out of the top 10.

## V3 Hypotheses to Test

Each hypothesis pairs a V3 change with the failure case it should fix and the observable behavior we expect. The benchmark stays the same 20 single-anchor + 12 multi-anchor anchor sets used in V2 testing.

| # | Change | Hypothesis | Observable verification |
|---|---|---|---|
| H1 | Format registry edits | Documentary biopics (Schindler's List, Spotlight, Oppenheimer) appear in narrative-feature top 10 of similar prestige biopic anchors instead of being miscategorized as documentaries | Re-run Best Picture trio + Oppenheimer single-anchor; check format breakdown |
| H2 | Medium registry edits | Persepolis vs. Sita Sings the Blues no longer scores 1.0 (matched only on `ADULT_ANIMATION`); animated films match by *technique* not *audience* | Re-run animation anchors; inspect medium_score breakdown |
| H3 | Shorts harsh downrank (0.30× + max-1 cap) | Toy Story top 10 has zero or one short, not the V2 trio at 8/9/10 | Toy Story single-anchor; count shorts in top 10 |
| H4 | Medium piecewise (0.65 cross-category) | The Dark Knight Rises top 10 has zero animated Batman entries (Year One, Long Halloween) | TDK Rises single-anchor; check for animated Batman in top 10 |
| H5 | Country/language 1.05 / 0.75 single-anchor | Barbie #1 is no longer Telugu (Swag); a US/UK on-tradition match takes #1 | Barbie single-anchor; check country tag of #1 |
| H6 | Franchise structural matrix | Star Wars 1977 surfaces Empire / Phantom Menace / Force Awakens in top 5 (V2 silenced these via the consistency gate) | Star Wars single-anchor; check franchise lane breakdown |
| H7 | Director auteur list (when unblocked) | Star Wars 1977 no longer surfaces American Graffiti; Lucas as a non-auteur becomes director-silent | Star Wars single-anchor; lane breakdown shows zero director contribution |
| H8 | Single-anchor themes lane | Barbie surfaces other satire / female-lead films (Poor Things, Promising Young Woman) in top 10 instead of generic shape adjacents | Barbie single-anchor; check themes lane contributions |
| H9 | Cast bucket floor (multi-anchor) | Tom Hanks 3-of-3 trio (Big + Polar Express + Toy Story) surfaces ≥1 Hanks vehicle in top 10 (Forrest Gump, Cast Away, Captain Phillips) | Run that custom multi-anchor set; check top 10 for Hanks |
| H10 | Rare-keyword lane | Oppenheimer with candidates sharing `MANHATTAN_PROJECT` (Fat Man and Little Boy, Day One) get bucket floor protection — they survive even with weaker shape | Oppenheimer single-anchor; check rare-keyword lane breakdown |
| H11 | Genre inclusion in rare-keyword pool | Common genres (DRAMA, COMEDY) collapse to low tier and contribute negligibly; rare genres (KAIJU, FOLK_HORROR, CYBERPUNK) carry real weight | Inspect IDF distribution of TMDB genres; verify low-tier collapse |

## V3 Success Criteria

V3 is ready to land when the benchmark re-run shows:

1. **No regression on V2 wins.** Pixar / Ghibli / Tarantino / war-film cohesive multi-anchor sets retain their tight clusters; Inception / Get Out single-anchor results don't lose their shape-adjacent matches.
2. **Zero shorts in top 5 for any non-short anchor; ≤1 short anywhere in top 10.** Verified across all 20 single-anchor + 12 multi-anchor cases.
3. **Format coherence in top 5.** For prestige biopic anchors (Oppenheimer, Schindler's List), top 5 contains zero pure documentaries; biopic narrative features dominate.
4. **Country/language coherence.** No cross-tradition film in top 3 of any monolingual anchor (Barbie, Inception, Get Out) unless it decisively beats in-tradition adjacents.
5. **Franchise coverage where it should fire.** Star Wars 1977 surfaces ≥3 same-franchise entries in top 10. Barbie's direct-to-DVD spinoffs do *not* dominate top 5.
6. **Distinctive match interpretability.** Lane breakdown for Oppenheimer #1 explicitly shows the `MANHATTAN_PROJECT` rare-keyword hit; for the Memento case, shows the non-linear-narrative combo.
7. **Single-anchor flow is no longer "shape only".** Themes lane contributes to ≥40 % of single-anchor candidates by lane breakdown sampling.
8. **Cast bucket activates for shared-lead multi-anchor sets.** Tom Hanks trio surfaces ≥1 Hanks vehicle; verified independently with a 4-anchor and 5-anchor variant.
9. **Auteur list (when unblocked) is consistent.** All anchors whose director is on the list see meaningful director-lane contribution; all anchors whose director is *not* on the list see zero director contribution in single-anchor (multi-anchor cohesion path unaffected).
10. **Debug payload completeness.** Every candidate's evidence bundle includes the V3 lane breakdown (rare-keyword tier, cast floor activation, franchise tier hit) so future regressions are inspectable in lane-level detail.

## V3.1 Calibration Adjustments (Post-Smoke)

V3 shipped and survived hypothesis verification, but the 21-anchor + 14-cohort
smoke run with the new diagnostic output (`base_score`, per-lane
`raw→contribution`, multipliers, floor activations) surfaced four
calibration issues that don't change architecture — only thresholds and
weights. All four land together as V3.1.

### Diagnosis

1. **Rare-keyword tier thresholds were sized for raw `log(N/df)` IDF,
   but `mv_trait_idf` normalizes to ~[0, 1]**. Max overall_keyword IDF
   observed is 1.0; only ~5 keywords sit at ≥1.0; only ~18 at ≥0.65.
   The original LOW < 2.5 / MODERATE < 4.5 / FLOOR_HIGH ≥ 5.0 thresholds
   were unreachable — every trait fell into the LOW tier and the lane
   was effectively capped at 0.05 for every candidate. The floor never
   fired across the entire smoke set.

2. **Themes lane weight (0.07 base, ~0.06 effective single-anchor) is
   too small to surface real thematic matches.** Concrete failure case:
   for the Barbie anchor, *I Am Not an Easy Man* (French gender-flip
   comedy) scored `themes=0.305` raw — a clear thematic match — but
   contributed only `0.018` to combined score. *The Favourite* scored
   `themes=0.139` raw → `0.008` contribution. *Lady Bird* (Gerwig auteur
   match) sat at #30. The lane is finding the right matches; the budget
   given to it is sub-perceptual.

3. **Single-anchor director lane has no floor.** Multi-anchor has a
   `0.35` floor at `M_d/N ≥ 0.75 AND shape ≥ 0.40`. Single-anchor only
   adds the flat `0.20` contribution, which can't compensate when the
   anchor's vector embedding doesn't match an auteur sibling
   (Barbie ↔ Lady Bird shape ≈ 0.0). Auteur-curated single-anchor
   matches deserve floor protection too — the user explicitly anchored
   on a film by that director.

4. **Aggregate tag co-occurrence is under-rewarded.** The themes lane
   captures normalized aggregate overlap but treats each shared trait
   linearly. Matching 5 moderate-IDF tags is qualitatively a stronger
   shape signal than matching 2 — random movie-pair sampling shows even
   the p99 of shared moderate-or-high tier IDF sum is **0.000** (324
   pairs, zero crossed 0.5). Real co-occurrence is far rarer than IDF
   independence suggests, and the lane should reward the qualitative
   jump.

### Decisions

**Decision 1 — Recalibrate rare-keyword tiers for [0,1] IDF range** (✅ shipped).

| Constant | V3 plan | V3.1 |
|---|---:|---:|
| `RARE_KW_TIER_LOW_MAX` | 2.5 | **0.30** |
| `RARE_KW_TIER_MODERATE_MAX` | 4.5 | **0.55** |
| `RARE_KW_LOW_COEF` | 0.01 | **0.05** |
| `RARE_KW_MODERATE_COEF` | 0.03 | **0.10** |
| `RARE_KW_HIGH_COEF` | 0.05 | **0.20** |
| `RARE_KW_FLOOR_HIGH_SINGLE` | 5.0 | **0.85** |
| `RARE_KW_FLOOR_COMBO_SUM` | 7.0 | **1.50** |

Verified impact: Star Wars `Empire` rare_keyword 0.020 → 0.556; Oppenheimer
`Schindler's` 0.020 → 0.319; Pixar trio 0.05–0.23 across top 10.

**Decision 2 — Lower rare-keyword floor's shape gate**.
`RARE_KW_FLOOR_SHAPE_GATE` 0.30 → **0.20**. The 0.30 gate blocked every
high-rare-keyword candidate observed in smoke (Schindler shape=0.030,
Dunkirk shape=0.160). Lowering to 0.20 lets the floor activate for
candidates that are genuinely on-tradition shape-wise but don't ride
the strongest vector cluster. Risk: low-shape, single-strong-tag false
positives. Mitigation: the floor's existing combo / single-super-rare
gates remain — a candidate still needs `idf ≥ 0.85 single` OR
`sum_moderate_high ≥ 1.50` to qualify.

**Decision 3 — Bump baseline themes weight 0.07 → 0.12**.
Effective contributions roughly double: Best Picture trio Pianist themes
0.065 → 0.11, Barbie *I Am Not an Easy Man* 0.018 → 0.037, Tenet for
Inception 0.029 → 0.052. Pixar / MCU / Star Wars don't tip rankings
(shape and franchise dominate); thematic-driven anchors (Barbie, Best
Picture, Tarantino, Get Out) recover thematic recall. Risk: spurious
shape-distant matches with one accidental tag overlap. Mitigation: the
themes denominator is `sum-of-anchor-IDFs`, so anchors with rich tag
pools have a higher bar; country/medium/format exclusion already
strips the worst noise.

**Decision 4 — Add single-anchor director floor**.
Mirrors the multi-anchor floor with softer thresholds:
- `DIRECTOR_FLOOR_SINGLE_MAGNITUDE = 0.35`
- `DIRECTOR_FLOOR_SINGLE_SHAPE_GATE = 0.20` (vs. 0.40 in multi —
  auteur-on-auteur is a stronger signal than M_d/N cohesion, so a
  weaker shape gate is appropriate)
- Fires when: anchor has a curated auteur director AND candidate
  shares that director AND `shape ≥ 0.20`.

Concrete impact: Lady Bird vs. Barbie anchor — current score 0.307
(director=0.20 + small contributions). New behavior: shape=0 still
fails the gate, so Lady Bird specifically isn't saved. But *The
Favourite* (currently #28, shape=0.412) and Frances Ha (if its shape
clears 0.20) get pulled into the top 10 by the floor. The intent is
explicit: when the user anchors on a film by a curated director, the
director's other films deserve floor protection unless their shape is
truly disjoint.

**Decision 5 — Add a "moderate combo" bonus to rare-keyword lane**.
Rewards the qualitative jump from "shared 1-2 moderate-rarity tags" to
"shared 3+ moderate-rarity tags." Applied as an additive bonus to the
rare-keyword score (not a separate floor — it stacks with the per-trait
contributions inside the passthrough lane).

Constants (sized from real catalog distribution: 324-pair random
sample showed p99 of shared moderate+high IDF = 0.000, so even 0.50 is
deep in the long tail):

- `RARE_KW_COMBO_THRESHOLD = 0.50` — minimum sum of shared moderate+high
  IDFs before bonus engages.
- `RARE_KW_COMBO_BASE = 0.05` — immediate bonus on crossing threshold
  (signals "this matters categorically, not just incrementally").
- `RARE_KW_COMBO_RATE = 0.05` — additional bonus per unit above
  threshold.
- `RARE_KW_COMBO_CAP = 0.15` — max bonus.

Formula: `bonus = clamp(0.05 + 0.05 * (sum_mod_high - 0.5), 0, 0.15)`
when `sum_mod_high ≥ 0.5`, else 0.

Sanity checks against observed anchor pools:
- Barbie (5 moderate tags, sum 2.01): a candidate matching all 5 →
  bonus ≈ 0.13. A candidate matching 2 of 5 → sum ~0.85 → bonus ≈ 0.07.
- Best Picture trio anchors have sum 1.6-3.4 mod+high IDF; Pianist
  matching most of one anchor's pool → bonus ≈ 0.12-0.15.
- Random pair: bonus = 0 (97%+ of random pairs share zero mod+high).

The bonus is the data-driven answer to the "5 not-individually-rare
tags = a clear shape signature" instinct: it doesn't change the
abstraction (themes still does proportional aggregate overlap;
rare_keyword still does tiered per-trait), but it adds the
qualitative-jump signal that linear-additive can't express.

### Verification harness updates

The 14-cohort multi-anchor harness gained a "Female-led / Gerwig" cohort
(Barbie + Lady Bird + Little Women) to verify the director floor and
themes weight changes against an auteur-saturated set. Two existing
cohort definitions were corrected: "Nolan trio" had ID `49047` (Gravity)
instead of `1124` (The Prestige); "Tarantino trio" had ID `5915` (Into
the Wild) instead of `24` (Kill Bill).

The batch runner output (`search_v2/run_similar_movies_batch.py`) now
prints per-result diagnostics covering base score, per-lane
`raw→contribution`, applied multipliers, and floor activations — making
calibration regressions inspectable without reading lane code.

**Decision 6 — Themes-recall candidate fetch**.
The post-smoke run exposed a recall gap: a thematically-aligned
candidate that misses Qdrant's top-K (current 500) and isn't pulled by
director / franchise / studio / source / quality / rare-medium has no
path into the candidate pool. Themes is purely a re-ranker; it never
adds candidates. Lady Bird vs. Barbie illustrates the problem — shape
embedding ≈ 0, but the auteur (Gerwig) and themes signals would score
it well *if* it ever entered the pool.

The fix: a new candidate-generation lane that fetches movies whose
shared anchor traits qualify by either rarity gate. Single SQL
aggregate joins `movie_card` array columns to `mv_trait_idf` per kind:

```sql
WITH shared AS (
    SELECT m.movie_id, ti.idf
    FROM public.movie_card m, public.mv_trait_idf ti
    WHERE ti.trait_kind = 1
      AND ti.trait_id = ANY(...anchor keyword_ids...)
      AND m.keyword_ids && ARRAY[ti.trait_id]
    UNION ALL
    -- same shape for kind=2 (concept_tag) and kind=3 (tmdb_genre)
)
SELECT movie_id
FROM shared
GROUP BY movie_id
HAVING SUM(idf) >= %s OR MAX(idf) >= %s
```

Two recall paths in one aggregate:

- **Single-trait gate** (`single_idf_threshold = 0.55`): catches the
  "one super-rare keyword uniquely identifies the candidate" case.
- **Combo-sum gate** (`combo_sum_threshold = 0.50`): catches the
  "multiple moderate tags coalesce into a clear signature" case.
  Per user direction the SUM includes **all-tier** IDFs (not filtered
  to moderate+high); the rationale is "even if individual tags aren't
  rare, matching a bunch is rare." Risk monitored — if the pool
  explodes on tag-rich anchors, raise the threshold.

Random-pair sampling (324 random catalog pairs, all kinds) showed p99
shared moderate+high IDF = 0.000 — the long tail is *empty*, so any
candidate clearing 0.50 is genuinely related.

New helper: `fetch_movie_ids_by_themes_recall` in [db/postgres.py](search_v2/../../db/postgres.py),
mirroring the `fetch_movie_ids_by_overall_keywords` pattern. Wired
into both single-anchor and multi-anchor candidate-id unions.

**Multi-anchor variant — consensus traits with cohesion-IDF tradeoff**:
For each (kind, trait_id) appearing in any anchor's themes pool,
compute cohesion = `M_t / N` and require cohesion ≥ a bar that scales
with IDF tier:

| IDF tier | Cohesion bar |
|---|---:|
| `idf < 0.30` (LOW) | 1.00 (all anchors) |
| `0.30 ≤ idf < 0.55` (MOD) | 0.67 (≥ 2/3) |
| `idf ≥ 0.55` (HIGH) | 0.50 (≥ half) |

Rarer traits qualify at lower cohesion because each rare match is
itself a strong signal. Common traits need everyone to carry them
(otherwise they're noise). The consensus pool feeds the same
single-anchor SQL helper — multi-anchor logic collapses to the same
fetch path with a tighter trait set.

**Decision 7 — Qdrant `DEFAULT_QDRANT_LIMIT` 500 → 2000**.
The shape funnel has been the only broad-recall path; bumping it 4×
catches vector-distant matches that the targeted themes-recall fetch
can't help (e.g., Lady Bird vs. Barbie — shares zero high-IDF or
combo-eligible traits with Barbie's pool). Cost: per-anchor latency
expected to grow from ~150ms to ~400-500ms (linear in candidate count
for the per-lane scoring + `_build_results`). Tradeoff is acceptable
for the recall gain.

### Constants matrix (V3.1, single source of truth)

| Constant | Value | Use |
|---|---:|---|
| `BASE_LANE_WEIGHTS["themes"]` | **0.12** | Themes proportional weight (was 0.06) |
| `RARE_KW_FLOOR_SHAPE_GATE` | **0.20** | Rare-keyword floor activation (was 0.30) |
| `DIRECTOR_FLOOR_SINGLE_MAGNITUDE` | 0.35 | Single-anchor director floor magnitude |
| `DIRECTOR_FLOOR_SINGLE_SHAPE_GATE` | 0.20 | Single-anchor director floor shape gate |
| `DIRECTOR_FLOOR_SINGLE_RATIO_THRESHOLD` | 1.00 | Binary single-anchor ratio gate |
| `RARE_KW_COMBO_THRESHOLD` | 0.50 | Combo bonus engagement (sum of mod+high IDFs) |
| `RARE_KW_COMBO_BASE` | 0.05 | Bonus on crossing |
| `RARE_KW_COMBO_RATE` | 0.05 | Per-unit increment |
| `RARE_KW_COMBO_CAP` | 0.15 | Max bonus |
| `THEMES_RECALL_COHESION_BAR_LOW` | 1.00 | Cohesion required for `idf < 0.30` |
| `THEMES_RECALL_COHESION_BAR_MOD` | 0.67 | Cohesion required for `0.30 ≤ idf < 0.55` |
| `THEMES_RECALL_COHESION_BAR_HIGH` | 0.50 | Cohesion required for `idf ≥ 0.55` |
| `DEFAULT_QDRANT_LIMIT` | **2000** | Qdrant top-K (was 500) |

Themes-recall thresholds are function defaults on
`fetch_movie_ids_by_themes_recall` (`single_idf_threshold = 0.55`,
`combo_sum_threshold = 0.50`); not module-level constants.

### Expected outcomes (things to verify in the next re-run)

**Wins to confirm**:
- Lady Bird and The Favourite enter Barbie top 10 (director floor +
  themes recall + themes weight bump combine).
- I Am Not an Easy Man rises (themes weight + combo bonus).
- Frances Ha surfaces if its shape clears 0.20 (Qdrant 2k recall).
- Tenet for Inception preserves #1 spot but with bigger themes
  contribution and unchanged director firing.
- Star Wars / Pixar / MCU / Best Picture / Tarantino top 10s preserve
  their cohesive clusters (no regression on V3 wins).

**Regressions to watch**:
- Floor over-firing from softer 0.20 shape gate.
- Combo bonus producing weak-shape coincidence-match flooding.
- Recall expansion pushing score >1.0 outliers further into noise
  (Empire already at 1.672; expect more candidates with multipliers
  stacking).
- Per-anchor latency degradation beyond ~500ms.
- Themes-recall pool exploding (>10k candidates) on tag-rich anchors —
  first regression to confirm/deny since user opted to include
  all-tier IDFs in the SUM gate.

## V3.3 Shape Multiplier (Reach × Quality Identity Boost)

The V3.2 smoke run made one pattern clear: candidates that share the
*same identity* as the anchor — Sharknado vs. The Room (both cult),
Schindler's vs. The Pianist (both prestige) — should ride a small,
explicit boost on top of the additive lane sum. The existing system
encodes this implicitly via lane scoring but never names it as a
first-class signal. V3.3 adds a `shape` multiplier alongside the
existing country / studio / format multipliers.

### The reach × quality grid

Two orthogonal axes already in the data:

- **Reach** from `imdb_vote_count`, three zones:
  - **HIGH** ≥ 100K (well-known to general audiences)
  - **MID** 10K–100K (genre-aware audiences)
  - **LOW** < 10K (deep-cut, requires effort to find)
- **Quality** from `_quality_bucket(row)` (already computed from
  reception_score + popularity_percentile):
  - **Acclaimed** (`prestige` bucket)
  - **Default** (`middle` bucket)
  - **Poorly rated** (`cult_garbage` bucket)

The 9-cell grid:

|             | **HIGH** (≥100K)                 | **MID** (10K–100K)             | **LOW** (<10K)                |
|---          |---                               |---                             |---                            |
| **Acclaimed** | Prestige cinema                  | Indie/festival darlings         | Deep-cut prestige             |
| **Default** | Mainstream blockbusters          | Mid-tier mainstream             | Forgotten mainstream          |
| **Poorly rated** | Mass cult                  | Niche cult                     | Just bad                      |

### The 5 shapes

Each shape is a coherent identity that absorbs one or more cells:

| Shape | Strength | Cells | Examples |
|---|---|---|---|
| **dogshit** | STRONG | LOW × Poorly rated | Mega Python, most direct-to-streaming horror |
| **cult_garbage** | STRONG | HIGH × Poorly rated, MID × Poorly rated | Sharknado, The Room, Plan 9 |
| **prestige** | MODERATE | HIGH × Acclaimed, MID × Acclaimed | Schindler's List, Lady Bird, 20th Century Women |
| **hidden_gem** | STRONG | LOW × Acclaimed | Foreign festival films, archival rediscoveries |
| **mainstream_blockbuster** | MODERATE | HIGH × Default | Inception, Barbie, Avengers |

Films in shapeless cells (MID × Default, LOW × Default) get no shape
boost — they're "just normal." A film classifies into exactly one
shape (or none).

Strength asymmetry rationale: cult, dogshit, and hidden-gem are
sharply distinctive identities (they say "this kind of movie" — a
shared search context). Prestige and mainstream-blockbuster overlap
heavily with general cinema, so the same-shape signal is weaker.

### Multiplier values

```
SHAPE_BOOST_STRONG   = 0.15    # max ×1.15
SHAPE_BOOST_MODERATE = 0.08    # max ×1.08
```

Comparable to existing multipliers (studio ×1.08–1.10, country boost
×1.05). Multiplicative — applied alongside the existing stack.

### Single-anchor application

If anchor and candidate share the same shape, multiplier =
`1.0 + max_strength`. If shapes differ or either is shapeless,
multiplier = 1.0.

### Multi-anchor application — cohesion-weighted

For each shape, compute cohort cohesion as `M_s / N` (number of
anchors carrying the shape, divided by cohort size). Then for each
candidate:

```
cohesion = anchor_shape_cohesion.get(candidate.shape, 0.0)
if cohesion < 0.5:
    multiplier = 1.0    # cohort doesn't have this shape strongly enough
else:
    multiplier = 1.0 + max_strength * cohesion
```

Examples:
- Pixar trio (Toy Story, Finding Nemo, Up — all mainstream_blockbuster):
  cohesion 1.0 → ×(1 + 0.08·1.0) = ×1.08 for mainstream candidates.
- Best Picture trio (all prestige): cohesion 1.0 → ×1.08 for
  prestige candidates.
- Studio Ghibli + Pixar mix: shape cohesion < 0.5 in any single
  shape → no boost. Right behavior — mixed-tradition cohort doesn't
  earn an identity lift.

### Plug-in point

Same multiplier stack as existing pipeline:

```
final_score = base_score
            × country_multiplier
            × studio_multiplier
            × medium_multiplier
            × shorts_multiplier
            × shape_multiplier         # NEW (V3.3)
```

Surfaced in `LaneEvidence.multipliers["shape"]` for diagnostics.

### Future change: weaving

V3.3 ships only the score-level shape boost. The slot-allocation
(MMR with anchor-aware quotas across B1=same-neighborhood / B2=adjacent
/ B3=spark) was designed but parked — the hypothesis is the shape
multiplier alone gets us most of the perceived-quality lift. If a
post-V3.3 smoke shows persistent neighborhood-mismatch noise (a
prestige anchor surfacing too many mainstream peers, a niche cult
anchor surfacing too many mainstream peers, etc.), revisit weaving as
V3.4. The full design is captured in the test-tracker conversation:

- 5 neighborhoods (Cult, Prestige, HIGH × Default, MID × Default,
  LOW × Default) connected by an adjacency graph (Prestige—HIGH×Default
  edge, Cult—HIGH×Default edge, no Cult—Prestige edge, Default reach
  chain).
- B1/B2/B3 buckets via graph distance from anchor neighborhood.
- Anchor-aware quotas (HIGH × Default 6:3:1, Cult/Prestige 5:4:1,
  LOW × Default 4:5:1).
- Deterministic MMR with `λ ∈ [0.65, 0.85]` driven by result-side
  cohesion (max share of any single cell among top-30 candidates),
  decimal target_quotas, linear filled-ratio penalty.
- Soft within-B1 reach proximity penalty so cult/prestige B1 isn't
  an undifferentiated soup across reach tiers.

### Expected outcomes to verify

**Wins**:
- Sharknado top 10 lifts cult sequels and same-tier cult peers
  (Mega Shark, Mega Python, etc.) without losing The Room as the
  mass-cult bridge.
- Best Picture trio (multi) preserves prestige cluster with a
  modest score boost (×1.08) on prestige candidates.
- Pixar trio (multi) preserves mainstream-blockbuster cluster.
- Schindler's List, Inception, Lady Bird as anchors all see
  identity-aligned candidates rise modestly.

**Regressions to watch**:
- Shape boost compounding with existing multipliers pushing scores
  >1.0 even more. Empire was already 1.820; expect modest further
  inflation.
- A wrong shape classification on a prestige anchor pulling cult
  films closer (or vice versa) — exclusivity check needed.
- Multi-anchor mixed cohorts (Studio Ghibli + Pixar mix) inadvertently
  picking up a cohesive shape — need to confirm cohesion < 0.5 actually
  suppresses the boost.

## V3 Implementation Order

Tracked in detail in [`similar_movies_v3_plan.md`](similar_movies_v3_plan.md) §5 with status flags. Summary:

- **Batch A — categorization fixes** (✅ shipped): registry edits for format and production medium.
- **Batch B — single-anchor enrichment**: themes lane, country/language extension and recalibration, director auteur rework (**blocked on auteur list**).
- **Batch C — franchise + weaving + cast**: franchise structural matrix, format harsh-downrank, cast generic-formula bucket-with-floor.
- **Batch D — distinctive-match interpretability**: rare-keyword lane (depends on Batch B themes lane).
- **Batch E — multiplier strengthening**: medium piecewise multiplier.
- **Batch F — tuning**: middle-bucket quality weight, low-cohesion threshold, award SPECIFICITY_FACTOR observation.

Each batch is testable end-to-end against the same 20 + 12 benchmark, so wins/regressions can be quantified per change.
