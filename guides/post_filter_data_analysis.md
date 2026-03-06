# Post-Hard-Filter Data Analysis Report

**Corpus:** 287,598 movies (surviving the five hard filters)
**Generated from:** `ingestion_data/tmdb_data_analysis.json` (rerun after Stage 3 filtering)

All five hard filter criteria now confirm zero failures across the surviving corpus — structural data quality is clean.

---

## 1. Per-Attribute Findings

### 1.1 Title

| Metric | Count | Rate |
|--------|-------|------|
| Null | 0 | 0% |
| Empty | 0 | 0% |

**Analysis:** Every surviving movie has a valid title. Title was not a differentiation signal even before filtering — this is a non-factor for quality scoring.

---

### 1.2 Release Date

| Metric | Count | Rate |
|--------|-------|------|
| Null (no date) | 667 | 0.23% |
| Future releases | 0 | 0% |
| Anomalous (unparseable) | 0 | 0% |

**Distribution by era:**

| Bucket | Count | Rate |
|--------|-------|------|
| Pre-1930 | 8,537 | 2.97% |
| 1930–1949 | 16,922 | 5.88% |
| 1950–1969 | 24,637 | 8.57% |
| 1970–1999 | 63,565 | 22.10% |
| 2000–2009 | 42,708 | 14.85% |
| 2010–2019 | 81,484 | 28.33% |
| 2020–2024 | 42,864 | 14.91% |
| 2025–present | 6,214 | 2.16% |

**Analysis:** 74% of the corpus released after 2000, with 2010–2019 being the peak decade. Pre-1930 cinema (2.97%) is well-represented despite age. The 667 null-date movies (0.23%) remain in the corpus — their other attributes are valid and excluding them would be too aggressive.

Release era matters for quality interpretation: older films (pre-1970) that survived the vote-count filter are likely canonical classics. Very recent films (2025) have naturally lower vote accumulation and should not be penalized for it.

---

### 1.3 Duration

| Bucket | Count | Rate |
|--------|-------|------|
| null_or_0 | 0 | 0% (all filtered) |
| 1–39 min | 52,425 | 18.23% |
| 40–60 min | 19,077 | 6.63% |
| 61–90 min | 95,651 | 33.25% |
| 91–120 min | 97,830 | 34.02% |
| 121–180 min | 20,930 | 7.28% |
| 181+ min | 1,685 | 0.59% |

**Analysis:** 67.3% of the corpus falls in the canonical feature-film range (61–120 min). 18.2% are short-form content (< 40 min) — these include legitimate award-winning short films but also TV pilots, promotional material, and content that is not a "movie" in any conventional sense. The 0.59% that exceed 3 hours are epics with a distinct character (documentaries, extended cuts, prestige dramas).

Runtime is **not** a direct quality signal for feature-length films (61–180 min), but being in the very short range (< 40 min) is a weak negative proxy — these films are unlikely to be the intended match for a natural-language movie search.

---

### 1.4 Poster URL

| Metric | Count | Rate |
|--------|-------|------|
| Has poster | 273,758 | 95.19% |
| Missing poster | 13,840 | 4.81% |

**Analysis:** 4.81% of movies are missing a poster. From the cross-attribute data, poster coverage is near-perfect for high-engagement movies — at vote_count ≥ 66 (p90), the null rate is effectively 0.003%. The 4.81% missing are concentrated almost entirely in the low vote-count bands (< 15 votes). A missing poster is a strong negative proxy for movie obscurity and TMDB catalogue completeness.

---

### 1.5 Watch Providers (US)

| Bucket | Count | Rate |
|--------|-------|------|
| 0 providers | 192,978 | 67.10% |
| 1–2 providers | 30,017 | 10.44% |
| 3–5 providers | 17,171 | 5.97% |
| 6–10 providers | 28,987 | 10.08% |
| 11+ providers | 18,445 | 6.41% |

| Metric | Count | Rate |
|--------|-------|------|
| Any US provider | 94,620 | 32.90% |
| No US provider | 192,978 | 67.10% |

**Analysis:** Only 32.9% of surviving movies are available on US streaming. The multi-platform segment (6+ providers: 16.5%) represents the broadly distributed mainstream catalogue. US streaming availability is a strong positive signal — distributors only license films with commercial viability. However, the inverse (no US provider) does not mean the film is low-quality; many legitimate international, independent, and classic films have no US streaming presence.

The more useful signal is **provider count as a gradient**, not a binary: 11+ providers (6.4%) = ubiquitous mainstream titles; 6–10 (10.1%) = widely available; 1–5 (16.4%) = limited availability; 0 (67.1%) = not streaming in the US.

---

### 1.6 Vote Count

| Bucket | Count | Rate |
|--------|-------|------|
| 0 | 0 | 0% (all filtered) |
| 1–10 | 197,150 | 68.55% |
| 11–50 | 56,445 | 19.63% |
| 51–100 | 11,925 | 4.15% |
| 101–500 | 14,368 | 4.99% |
| 501–1,000 | 2,984 | 1.04% |
| 1,001–5,000 | 3,732 | 1.30% |
| 5,001–10,000 | 627 | 0.22% |
| 10,001+ | 367 | 0.13% |

**Percentiles:**

| p25 | p50 | p75 | p90 | p95 | p99 |
|-----|-----|-----|-----|-----|-----|
| 2 | 5 | 15 | 66 | 196 | 1,821 |

**Survival curve:**

| vc threshold | Surviving | Rate |
|-------------|-----------|------|
| 10 | 96,187 | 33.4% |
| 50 | 34,409 | 12.0% |
| 100 | 22,228 | 7.7% |
| 500 | 7,721 | 2.7% |
| 1,000 | 4,731 | 1.6% |
| 5,000 | 994 | 0.35% |
| 10,000 | 367 | 0.13% |

**Analysis:** Even after filtering zero-vote movies, the distribution remains dramatically right-skewed. The median is only 5 votes. 68.6% of surviving movies have 10 or fewer votes — meaning they are essentially unknown. A movie at the 90th percentile has just 66 votes; a movie with 1,000 votes is in the top 1.6%.

Vote count is the dominant quality meta-signal. It drives every other quality proxy (streaming coverage, poster presence, completeness, vote_average reliability). The practical threshold for "this movie has some audience validation" is around 50–100 votes; the threshold for "this is a notable film" is around 500+.

---

### 1.7 Popularity

| Bucket | Count | Rate |
|--------|-------|------|
| 0.0 (zero) | 199 | 0.07% |
| 0.01–1.0 | 234,781 | 81.63% |
| 1.01–5.0 | 44,896 | 15.61% |
| 5.01–10.0 | 5,499 | 1.91% |
| 10.01–50.0 | 2,174 | 0.76% |
| 50.01–100.0 | 31 | 0.01% |
| 100.01+ | 18 | 0.006% |

**Percentiles:**

| p25 | p50 | p75 | p90 | p95 | p99 |
|-----|-----|-----|-----|-----|-----|
| 0.17 | 0.34 | 0.72 | 1.85 | 3.27 | 8.95 |

**Vote count cross-tab (split at each attribute's median):**

| Segment | Count | Rate |
|---------|-------|------|
| High vc + High pop | 99,928 | 34.7% |
| High vc + Low pop | 47,006 | 16.3% |
| Low vc + High pop | 43,878 | 15.3% |
| Low vc + Low pop | 96,786 | 33.7% |

**Analysis:** Popularity is TMDB's algorithmic activity score (based on current page views, watchlist additions, etc.) and decays over time. It captures "current attention" rather than historical validation, making it complementary to vote_count.

The 47K movies with high vote_count but low popularity are primarily **older established films** that accumulated votes historically but are no longer trending. The 44K with low vote_count but high popularity are likely **recent releases** still accumulating votes. These two quadrants reveal information that vote_count alone misses. Popularity is a genuine additive signal.

The 199 zero-popularity movies are genuine ghosts in the TMDB system — they have votes but zero discoverability signals.

---

### 1.8 Vote Average

**Reliability by vote_count floor:**

| vc floor | Count | Mean vote_avg | pct good (>6.5) |
|----------|-------|---------------|-----------------|
| ≥ 5 (p50) | 146,934 | 5.80 | 29.2% |
| ≥ 15 (p75) | 74,392 | 6.09 | 35.6% |
| ≥ 66 (p90) | 28,849 | 6.40 | 47.2% |
| ≥ 196 (p95) | 14,412 | 6.57 | 54.3% |
| ≥ 1,821 (p99) | 2,876 | 6.90 | 70.3% |

**Distribution at vc ≥ 66 (p90 floor):**

| Range | Count | Rate |
|-------|-------|------|
| 0 (no rating) | 0 | 0% |
| 0.1–3.0 | 43 | 0.15% |
| 3.1–5.0 | 1,693 | 5.87% |
| 5.1–6.5 | 13,459 | 46.67% |
| 6.6–7.5 | 11,265 | 39.04% |
| 7.6–8.5 | 2,352 | 8.15% |
| 8.6–10.0 | 37 | 0.13% |

**Analysis:** Vote average is the most dangerous signal to use naively because its reliability is entirely dependent on the vote count floor. At the p50 floor (≥ 5 votes), a 6.0 rating could be based on 5 votes and is statistically meaningless. At the p90 floor (≥ 66 votes), the distribution is coherent and meaningful — 47.2% of movies score above 6.5, and the mean of 6.40 reflects genuine audience sentiment.

The 648 movies with vote_average = 0.0 (~0.23% of corpus, surviving because vote_count > 0) are an edge case — they likely have votes but haven't had ratings tallied. They should receive no vote_average bonus.

**Rule:** Only use vote_average as a quality signal when vote_count ≥ 15 (p75). Apply full weight only at vote_count ≥ 66 (p90).

---

### 1.9 Overview Length

| Bucket | Count | Rate |
|--------|-------|------|
| 0 | 0 | 0% (all filtered) |
| 1–20 chars | 363 | 0.13% |
| 21–50 chars | 4,235 | 1.47% |
| 51–100 chars | 26,924 | 9.36% |
| 101–200 chars | 88,064 | 30.62% |
| 201–500 chars | 129,553 | 45.04% |
| 501+ chars | 38,459 | 13.37% |

**Analysis:** 89% of movies have overviews of 101+ characters — a healthy corpus for vector search. The 1.6% with < 50 characters are practically empty: a 20-character overview is barely a sentence and will produce meaningless embeddings. The 501+ group (13.4%) represent richly documented films.

Overview length has a positive correlation with quality: well-documented films attract more descriptive text from TMDB contributors. However, it is partly endogenous — popular films get more edits. It should be used as a weak positive signal for the 501+ segment and a soft negative signal for the < 50 segment.

---

### 1.10 Genre Count

| Metric | Count |
|--------|-------|
| zero_count | 0 |
| negative_count | 0 |

**Analysis:** Every surviving movie has at least one genre. Genre count itself (how many genres a movie has) is not analyzed further in this dataset beyond confirming no zeros remain. A movie tagged with multiple genres (e.g., 3–4) is likely a more mainstream production with broader editorial coverage, but this effect is minor compared to vote-based signals.

---

### 1.11 Boolean Metadata Fields (Completeness)

| Field | True | False | True Rate |
|-------|------|-------|-----------|
| has_production_companies | 223,147 | 64,451 | 77.59% |
| has_cast_and_crew | 262,480 | 25,118 | 91.27% |
| has_production_countries | 257,671 | 29,927 | 89.59% |
| has_keywords | 153,767 | 133,831 | 53.47% |
| has_budget | 28,419 | 259,179 | 9.88% |
| has_revenue | 20,705 | 266,893 | 7.20% |

**Completeness score distribution (0–6, sum of flags above):**

| Score | Count | Rate |
|-------|-------|------|
| 0 | 3,579 | 1.24% |
| 1 | 18,247 | 6.34% |
| 2 | 38,282 | 13.31% |
| 3 | 96,096 | 33.41% |
| 4 | 104,954 | 36.49% |
| 5 | 15,366 | 5.34% |
| 6 | 11,074 | 3.85% |

**Revenue + Budget cross-tab:**

| Status | Count | Rate |
|--------|-------|------|
| Neither | 250,946 | 87.26% |
| Budget only | 15,947 | 5.55% |
| Revenue only | 8,233 | 2.86% |
| Both | 12,472 | 4.34% |

**Analysis:**

- **has_cast_and_crew (91.3%)** and **has_production_countries (89.6%)** are near-universal — their absence (8.7% and 10.4% respectively) is a meaningful negative signal.
- **has_production_companies (77.6%)** — absence (22.4%) often indicates self-distributed or informal productions.
- **has_keywords (53.5%)** — keywords are community-sourced tags. Presence indicates an active audience that has bothered to catalogue the film. Meaningful differentiator at the margins.
- **has_budget / has_revenue (10% / 7%)** — extremely sparse; present only for commercially significant productions. Having **both** (4.34%) is a strong positive signal — these are films important enough to have financial records in TMDB. Does not penalize absence (87% lack both).

The completeness score (0–6) is the cleanest composite proxy for "how well-documented is this film in TMDB." Score 5–6 (9.2% of corpus) = well-documented mainstream productions. Score 0–1 (7.6%) = severely incomplete records.

---

## 2. Broader Cross-Attribute Analysis

### 2.1 Vote Count as the Universal Meta-Signal

The most important finding across this entire dataset is that **vote_count predicts every other quality proxy monotonically**. It is not merely one signal among equals — it is the latent variable that underlies nearly all other observable quality attributes.

| vc Band | US Streaming | Mean Completeness | Null Poster Rate |
|---------|-------------|-------------------|-----------------|
| < 2 (bottom 25%) | 15.2% | 2.77 / 6 | 10.9% |
| 2–5 (p25–p50) | 20.2% | 2.94 / 6 | 7.0% |
| 5–15 (p50–p75) | 31.2% | 3.27 / 6 | 2.1% |
| 15–66 (p75–p90) | 51.1% | 3.71 / 6 | 0.25% |
| ≥ 66 (top 10%) | 80.6% | 4.73 / 6 | 0.003% |

At the p90+ band (vc ≥ 66, just 28,849 movies — 10% of the corpus):
- **80.6%** have US streaming (vs. 15.2% at the bottom)
- **Mean completeness 4.73** (vs. 2.77 at the bottom — 71% higher)
- **Null poster rate near zero** (10.9% at the bottom)
- **Mean vote_average 6.40** and rising with vc

This monotonic relationship across four independent quality proxies provides very strong validation that vote_count is not just a popularity metric but a genuine proxy for "this film has been seen, evaluated, and catalogued by real audiences."

### 2.2 The Two Quadrants of Vote Count vs. Popularity

The vote_count × popularity cross-tab reveals two important edge cases:

1. **High vc + Low pop (47,006 movies, 16.3%):** Films that accumulated votes historically but no longer trend. These are the **established catalogue** — older films, classics, and cult titles that users actively look for. They should not be penalized for low current popularity; their vote history validates them.

2. **Low vc + High pop (43,878 movies, 15.3%):** New releases with strong current momentum but not yet enough votes for statistical reliability. Their vote_average is unreliable, but the high popularity signal is meaningful. These should receive a popularity boost without over-weighting their vote_average.

This asymmetry justifies using **both** signals rather than collapsing them — they capture genuinely different information.

### 2.3 Vote Average Is Unreliable Below the p75 Vote Floor

The mean vote_average climbs monotonically from 5.80 (vc ≥ 5) to 6.90 (vc ≥ 1,821). This is partly because higher-vote movies genuinely are better films (survivor bias is real), but also because a rating based on 5 votes is statistically meaningless — any single reviewer can swing it. The practical rule:

- **vc < 15:** Do not use vote_average as a quality signal at all.
- **15 ≤ vc < 66:** Use vote_average as a weak signal (20–40% weight).
- **vc ≥ 66:** Use vote_average as a strong signal (full weight).

### 2.4 Completeness Score as a Secondary Composite

The completeness score (0–6 boolean flags) is highly correlated with vote_count but is **not redundant** — it captures whether third-party databases and studios actively submitted data to TMDB, which is a distinct signal from audience engagement. A film with moderate vote_count (e.g., 30 votes) but completeness score 5 or 6 (has budget, keywords, cast, production companies, countries) was well-documented by industry sources even if not yet widely seen. This is a useful tiebreaker within the same vote_count band.

### 2.5 Structural Thinness of the Corpus

Despite being post-filter, 68.6% of surviving movies have ≤ 10 votes. The corpus is an enormous long tail. The top 10% by vote_count (≥ 66 votes) contains the movies that are genuinely discoverable and quality-validated. Any quality scoring scheme must weight vote_count heavily enough to surface this top 10% over the bulk of the long tail — otherwise every search result will be dominated by obscure films.

---

## 3. Quality Signal Rankings

### Positive Signals (higher value = stronger indication of quality)

| Signal | Type | Strength | Notes |
|--------|------|----------|-------|
| **vote_count** | Continuous | ★★★★★ Very High | The universal proxy. Every other signal correlates with it. Non-linear — log-scale scoring recommended. |
| **vote_average** (vc ≥ 66) | Continuous | ★★★★☆ High | Highly reliable at p90+ vc floor. Distribution shifts from 5.8 → 6.9 mean as vc rises. |
| **popularity** | Continuous | ★★★☆☆ Moderate | Complementary to vote_count — captures current momentum and new releases that vc hasn't caught up to. Use log-scale. |
| **has_budget AND has_revenue** | Binary (both) | ★★★★☆ High | Only 4.34% of corpus, but these are provably commercially significant productions. Strong positive when present. |
| **completeness_score (5–6)** | Categorical | ★★★☆☆ Moderate | Well-documented films. Mean completeness of 4.73 at p90 vc band. Good tiebreaker. |
| **US streaming provider count** | Ordinal | ★★★☆☆ Moderate | 6+ providers = mainstream, broadly distributed. Strong for that segment (16.5%). Weak/inapplicable for 67%. |
| **has_keywords** | Binary | ★★☆☆☆ Low-Moderate | Community-sourced tags indicate active audience cataloguing. 53% true. Useful only as a tiebreaker. |
| **overview_length (501+ chars)** | Categorical | ★★☆☆☆ Low-Moderate | Richly documented films. 13.4% of corpus. Mild positive for vector search quality. |
| **vote_average** (15 ≤ vc < 66) | Continuous | ★★☆☆☆ Low-Moderate | Statistically weakened by small sample but directionally meaningful. Weight at 25–40% of full. |

---

### Negative Signals (presence = indication of lower quality)

| Signal | Type | Strength | Notes |
|--------|------|----------|-------|
| **vote_average < 5.0** (vc ≥ 66) | Conditional | ★★★★☆ High | At p90 vc floor, only 6% fall below 5.0. Being in this group is a clear negative. |
| **Missing poster** | Binary | ★★★☆☆ Moderate-High | 4.81% overall, but concentrated in low-vc films. At vc ≥ 66, near-zero. Strong negative for obscure films. |
| **completeness_score 0–1** | Categorical | ★★★☆☆ Moderate | Severely undocumented — 7.6% of corpus. Strongly correlated with bottom vc band. |
| **No US streaming provider** | Binary | ★★☆☆☆ Low-Moderate | Majority of corpus (67%), so cannot be used as a strong discriminator. Weak negative only. |
| **overview_length < 50 chars** | Categorical | ★★☆☆☆ Low-Moderate | 1.6% of corpus. Nearly empty overviews will produce poor vector search results. Soft penalty. |
| **duration < 40 min** | Categorical | ★★☆☆☆ Low-Moderate | 18.2% of corpus. Short films are legitimate but unlikely to be the intended match for most movie queries. |
| **Zero popularity** | Binary | ★★☆☆☆ Low-Moderate | 199 movies (0.07%). Tiny but clear — no discoverability signals whatsoever in TMDB. |
| **vote_average < 5.0** (vc < 15) | Conditional | ★☆☆☆☆ Very Low | Statistically unreliable at this vote floor. Do not penalize — rating could be from 1–2 reviewers. |

---

## 4. Scoring Architecture Recommendations

Based on the signal analysis above, a composite quality score should be structured as follows:

### Tier 1: Primary Signals (must drive the score, ~70% weight)
1. **vote_count (log-scaled):** Log transformation recommended — the difference between 1 and 10 votes matters enormously; the difference between 5,000 and 10,000 is marginal. Normalize to [0, 1] over log(vote_count).
2. **vote_average (conditional):** Apply only at vc ≥ 15 as a multiplier or additive component. Full weight at vc ≥ 66. Normalize to [0, 1] over the range [4.0, 9.0].

### Tier 2: Secondary Signals (additive modifiers, ~20% weight)
3. **popularity (log-scaled):** Additive bonus, especially important for new films with low vc but high popularity.
4. **completeness_score:** Normalized 0–6 scale contribution. Useful tiebreaker within the same vc band.
5. **has_budget AND has_revenue:** Binary bonus for the 4.34% of films with financial records.

### Tier 3: Soft Signals (light penalties/bonuses, ~10% weight)
6. **Missing poster:** Soft penalty.
7. **US streaming provider count:** Ordinal bonus (0, low, medium, high, very high tiers).
8. **overview_length < 50:** Very soft penalty for semantic search utility.
9. **duration < 40 min:** Optional soft penalty for films unlikely to match typical user intent.

### What NOT to Score On
- **has_cast_and_crew (91%):** Too common to differentiate. Absence is minor.
- **has_production_countries (90%):** Same issue.
- **vote_average with vc < 15:** Statistically meaningless. Using it will introduce noise.
- **Release year alone:** Not a direct quality signal (pre-1930 classics are genuinely good). Only matters for contextualizing vc accumulation expectations.
