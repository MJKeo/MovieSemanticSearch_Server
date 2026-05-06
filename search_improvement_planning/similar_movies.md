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
