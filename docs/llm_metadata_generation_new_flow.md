# LLM Metadata Generation: Redesigned Flow

Optimal flow for generating LLM metadata, incorporating the findings from the [pipeline report](llm_metadata_generation_report.md) and [efficiency analysis](llm_metadata_generation_efficiency_analysis.md), refined through design review. This document walks through the reasoning behind each decision, then presents the complete new flow.

## Table of Contents

1. [Design Reasoning](#design-reasoning)
2. [New Pipeline Architecture](#new-pipeline-architecture)
3. [Pre-Consolidation Phase](#pre-consolidation-phase)
4. [Wave 1: Foundation Metadata](#wave-1)
5. [Wave 2: Analysis Metadata](#wave-2)
6. [Sparse Movie Skip Conditions](#sparse-movie-skip-conditions)
7. [Estimated Savings Summary](#estimated-savings)
8. [Pending Optimizations](#pending-optimizations)

---

## 1. Design Reasoning <a name="design-reasoning"></a>

### Starting point: what's wrong with the current flow?

The current flow has 8 LLM calls across 2 waves, producing 7 metadata types. The efficiency analysis identified several structural problems:

1. **Review duplication is the #1 cost driver.** `featured_reviews` (500-2500 tokens of raw review text) is serialized and sent to 6 of 8 LLM calls independently. That's ~3000-15000 tokens of duplicated review content per movie. Each call extracts a different slice of insight from the same text, but pays the full input cost.

2. **The "reception triple" appears everywhere.** `reception_summary` + `audience_reception_attributes` + `featured_reviews` — three representations of the same underlying signal (what audiences think) — appear together in 4 calls. That's 3 encodings × 4 calls of redundant audience opinion data.

3. **Wave structure creates an information gap.** Watch context runs in Wave 1 without access to review consolidation or other Wave 1 outputs.

4. **Justification fields cost output tokens with questionable ROI.** 28-32 justification fields generate ~300-670 output tokens per movie that are never embedded or stored. With gpt-5-mini's internal chain-of-thought, these may be redundant.

5. **Several inputs provide near-zero value to specific calls.** `overview` in Wave 2 (superseded by plot_synopsis), `featured_reviews` in source_of_inspiration (reviews rarely discuss production origins), `parental_guide_items` in viewer_experience (verbose, mostly for skipped sections).

6. **Keywords are sent inconsistently.** `plot_keywords` and `overall_keywords` have different signal profiles but are sent without regard to which type is relevant per generation.

### Decision 1: Reception as a dual-purpose call

The most impactful change addresses problems #1 and #2 simultaneously.

**Observation:** Reception already runs in Wave 1 and processes all three reception signals (reviews, summary, attributes). Wave 2 calls already depend on Wave 1 completing (for `plot_summary`). So reception's output is available to Wave 2 at zero additional latency cost.

**Decision:** Add a `review_insights_brief` output field to reception (~150-250 tokens). This field captures the key thematic, emotional, structural, evaluative, and source-material observations from the raw reviews in a single condensed paragraph. Wave 2 calls then consume this brief instead of the raw reviews + reception_summary + audience_reception_attributes.

**Why not a separate pre-consolidation LLM call?** Adding an LLM call to save LLM tokens defeats the purpose. Reception already has all the review data in context — asking it to also produce a brief summary for downstream use is marginal additional work for the same call.

**Contingency:** If empirical testing shows that dual-purpose reception degrades quality (the evaluative summary and descriptive brief cross-contaminate), a separate gpt-5-nano call for review consolidation is the fallback. The brief's schema and downstream consumption would be identical — only the producer changes.

**Source material extraction:** The reception prompt must explicitly instruct the model to include source material observations in the brief (e.g., "reviewers noted this is a faithful adaptation of the novel," "inspired by true events"). This ensures source_of_inspiration gets this signal downstream without needing raw reviews.

**Risk:** If reception fails, Wave 2 calls lose the brief. But they still have plot_synopsis, keywords, and genres — they degrade gracefully rather than catastrophically.

**Savings:** ~3000-5000 input tokens/movie (raw reviews sent to 1 call instead of 6, brief sent to 4-5 calls at ~150-250 tokens each).

### Decision 2: Move watch context to Wave 2, remove all plot info

**Observation:** Watch context answers "WHY and WHEN to watch this movie" — not "WHAT HAPPENS in this movie." Plot detail (whether overview or full synopsis) risks anchoring the model on specific events rather than experiential attributes. A full synopsis might produce "watch this if you want to see a heist gone wrong" instead of the more useful "watch this if you want a high-tension cat-and-mouse thriller."

**Decision:** Move watch context to Wave 2 so it receives `review_insights_brief`, but remove ALL plot information — no `overview`, no `plot_synopsis`. The real value drivers for watch context are:
- `review_insights_brief` — tells you how the movie *feels* and what stands out
- `genres` — strong signal for occasion matching (horror → halloween, romance → date night)
- `overall_keywords` — categorical signals (family-friendly, dark comedy, etc.)
- `maturity_summary` — content advisory context ("don't watch this with kids")

**Latency impact:** Wave 1 shrinks from 3 parallel tasks to 2, finishing slightly faster. Wave 2 grows from 4 to 5 parallel tasks. Net latency change is negligible.

### Decision 3: Selective keyword routing

**Observation:** `plot_keywords` and `overall_keywords` have fundamentally different signal profiles. Plot keywords are community-tagged plot elements — noisy, potentially repetitive across near-duplicate strings, and can accidentally over-emphasize minor plot points. Overall keywords are broader categorical tags about the movie itself. Blindly merging and sending both to every call introduces noise where it doesn't help.

**Decision:** Route keywords selectively per generation based on which type provides value:

| Generation | plot_keywords | overall_keywords | Rationale |
|---|---|---|---|
| Plot Events | yes | no | Plot keywords directly help identify events to cover |
| Plot Analysis | yes | no | Thematic keywords support theme identification |
| Viewer Experience | both | both | Broad signal useful for emotional/sensory assessment |
| Watch Context | no | yes | Categorical tags inform occasions; plot events don't |
| Narrative Techniques | no | yes | Structural tags ("nonlinear timeline", "unreliable narrator") live in overall |
| Production Keywords | both | both | Filtering task — more keywords = more to filter from |
| Source of Inspiration | both | both | Source tags could appear in either list |

For generations receiving both, keywords are merged and deduplicated: `list(dict.fromkeys(plot_keywords + overall_keywords))`.

**Pre-consolidation:** Produce `merged_keywords` (deduped union) once. Generations taking both use the merged list; others use the original list directly.

### Decision 4: Remove justification fields (pending empirical validation)

**Observation:** gpt-5-mini is a reasoning model with internal chain-of-thought (reasoning tokens). The model already "thinks through" its analysis before producing output. Explicit justification fields in the output schema were a technique designed for non-reasoning models where forcing the model to write rationale before answers improved accuracy.

**Decision:** Remove all justification fields from output schemas. This is proposed as a change pending empirical validation — a batch of 50-100 movies should be generated with and without justification fields, and the output quality compared.

**Savings:** ~300-670 output tokens/movie across 28-32 justification fields.

**Risk:** The interaction between explicit output justifications and internal reasoning tokens is not well-studied. Hence: empirical validation required before deploying.

### Decision 5: Conditional maturity consolidation

**Observation:** `maturity_rating`, `maturity_reasoning`, and `parental_guide_items` are three separate inputs serving similar purposes. `parental_guide_items` is verbose (40-80 tokens per movie), primarily informs `disturbance_profile` and `sensory_load` in viewer_experience — both optional sections that get skipped for ~70-80% of movies. However, the severity detail in parental_guide_items IS valuable: there's a meaningful difference between "R for one F-bomb" and "R for prolonged graphic violence."

**Decision:** Conditional consolidation:
- If `maturity_reasoning` exists (≥1 item): consolidate to `"R — violence, strong language, brief nudity"` format. This captures the essential signal compactly.
- If `maturity_reasoning` is absent: fall back to compact `parental_guide_items` formatted as `"severe violence, moderate language, mild nudity"` (severity + category, comma-separated). Preserves severity detail when the primary signal is missing.

Either path produces one `maturity_summary` string sent to viewer_experience and watch_context.

### Decision 6: Title as "Title (Year)" across all generations

**Observation:** The LLM may have parametric knowledge activated by the title alone. Including the release year disambiguates remakes ("The Batman (2022)" vs "Batman (1989)"), grounds temporal context (a 1950s film has different narrative conventions than a 2020s film), and helps the model produce more era-appropriate descriptions.

**Decision:** All 8 LLM calls receive title formatted as `"Title (Year)"` (e.g., "The Matrix (1999)"). This is essentially free (~2-3 extra tokens per call) with significant upside for temporal grounding and disambiguation.

### Decision 7: Add genres to narrative techniques

**Observation:** Narrative techniques currently doesn't receive genres. Genres provide useful context for determining expected narrative patterns ("mystery" implies information control techniques, "documentary" implies specific POV structures).

**Decision:** Add `genres` to narrative techniques input. Cost: trivial (~5-10 tokens). Potential quality improvement for grounding structural analysis.

### Decision 8: Plot events — always attempt, never hallucinate

**Observation:** ~34% of movies (37,677 of 111,899) lack synopsis and plot summaries. The original proposal would have skipped these entirely. However, with the partial pipeline (Decision 9), skipping plot_events doesn't mean losing the movie entirely. And many of these movies still have overviews, keywords, and titles that provide some plot context.

**Decision:** Plot events always attempts — no skip condition based on input sparsity. The prompt must explicitly forbid hallucination: "Only describe what is evident from the provided data. Do not supplement with your own knowledge of this film. If data is limited, produce a shorter summary rather than inventing details."

For sparse movies, this produces thin but accurate plot_summaries. A thin accurate summary is strictly better than a fabricated detailed one — downstream generations can work with less data, but they can't recover from wrong data.

### Decision 9: Partial pipeline on plot_events failure

**Observation:** The current flow treats plot_events failure as a RuntimeError that kills the entire movie. But several generations don't need plot_synopsis at all, or can work without it if review data exists.

**Decision:** If plot_events fails (API error, timeout — not data sparsity, since we always attempt):
- **Still runs:** reception (independent of plot), production (keywords-driven), watch_context (no plot input anyway), viewer_experience (if review_insights_brief exists — reviews carry strong emotional/tonal signal independently)
- **Skips:** plot_analysis (requires plot detail), narrative_techniques (requires plot detail for structural analysis)

The movie gets partial vectors and remains searchable for queries matching its available vectors.

### Decision 10: Source of inspiration — parametric knowledge allowed

**Observation:** Source material facts ("based on a novel," "inspired by true events") are categorical and verifiable. Unlike plot events where hallucination creates cascading false information, a source-of-inspiration claim is either right or wrong, and the model's knowledge of well-known adaptations is generally accurate.

**Decision:** The source_of_inspiration prompt includes: "If you are highly confident about the source material based on your knowledge, include it." Combined with the `"Title (Year)"` format which helps the model disambiguate, this allows the LLM to contribute source facts that may not appear in keywords or reviews.

**Contrast with plot_events:** Plot events explicitly forbids parametric knowledge. Source of inspiration explicitly allows it (with high confidence). The difference is that plot fabrication cascades to 5+ downstream generations, while source material is a leaf-node classification that doesn't cascade.

---

## 2. New Pipeline Architecture <a name="new-pipeline-architecture"></a>

```
BaseMovie
    │
    ▼
Pre-Consolidation (no LLM — pure data processing)
    ├── Keyword routing (selective merge for generations needing both lists)
    ├── Maturity consolidation (conditional: reasoning-based or parental-guide fallback)
    └── Sparse movie input assessment (determine which generations to skip)
    │
    ▼
Wave 1 (2 tasks in parallel)
    ├── Plot Events (always attempts; no-hallucination rule)
    └── Reception (graceful failure OK; produces review_insights_brief for Wave 2)
    │
    │ Outputs fed to Wave 2:
    │   plot_events → plot_summary (as plot_synopsis)
    │   reception → review_insights_brief
    │
    ▼
Wave 2 (up to 5 tasks in parallel, depends on Wave 1)
    ├── Plot Analysis
    ├── Viewer Experience
    ├── Watch Context (NO plot info — review_insights_brief + genres + keywords)
    ├── Narrative Techniques
    └── Production (2 sub-calls: production_keywords + source_of_inspiration)
    │
    ▼
Vector Text Generation → Embedding → Ingestion (unchanged)
```

**If plot_events fails:** Wave 2 runs a partial pipeline — plot_analysis and narrative_techniques are skipped. Watch context, viewer_experience (if review_insights_brief exists), and production still run.

**Current → New comparison:**
| Aspect | Current | New |
|--------|---------|-----|
| Total LLM calls | 8 | 8 (same count, different distribution) |
| Wave 1 tasks | 3 (plot_events, watch_context, reception) | 2 (plot_events, reception) |
| Wave 2 tasks | 4 (plot_analysis, viewer_exp, narr_tech, production) | 5 (+ watch_context) |
| Review tokens sent per movie | ~6000-15000 (raw reviews × 6 calls) | ~2000-3500 (raw reviews × 1 call + brief × 5 calls) |
| Justification output tokens | ~300-670 | 0 (pending validation) |
| Title format | title only | "Title (Year)" |
| Watch context has plot info? | Yes (overview) | No (experiential inputs only) |
| Narrative techniques has genres? | No | Yes |
| Plot events on sparse movies | Skip if <50 words of source text | Always attempt, never hallucinate |
| Plot events failure | RuntimeError — movie dropped | Partial pipeline — 3-4 generations still run |
| Source of inspiration | No review signal, no parametric knowledge | review_insights_brief + parametric knowledge allowed |
| Keyword routing | Inconsistent per-call | Selective routing by signal type |
| Sparse movies waste LLM calls? | Yes | No (skip conditions per generation) |

---

## 3. Pre-Consolidation Phase <a name="pre-consolidation-phase"></a>

Three preprocessing steps run before any LLM calls. All are pure data processing — no LLM cost.

### 3.1 Keyword Routing

**What:** Prepare keyword inputs per generation. Some generations receive only `plot_keywords`, some only `overall_keywords`, and some receive a merged deduplicated list.

**Why:** Plot keywords are community-tagged plot elements — noisy, potentially repetitive, and can over-emphasize minor plot points. Overall keywords are broader categorical tags. Each generation benefits from different keyword signals.

**Input data:**
- `plot_keywords` (list[str]) — IMDB community-tagged plot elements
- `overall_keywords` (list[str]) — broader IMDB keywords about the movie

**Output data:**
- `plot_keywords` (list[str]) — passed directly to: plot_events, plot_analysis
- `overall_keywords` (list[str]) — passed directly to: watch_context, narrative_techniques
- `merged_keywords` (list[str]) — deduplicated union for: viewer_experience, production_keywords, source_of_inspiration

**Implementation:** `merged_keywords = list(dict.fromkeys(plot_keywords + overall_keywords))` — plot_keywords first, then unique overall_keywords appended.

### 3.2 Maturity Consolidation

**What:** Produce a single `maturity_summary` string from available maturity data, with conditional fallback.

**Why:** Three separate maturity fields (`maturity_rating`, `maturity_reasoning`, `parental_guide_items`) carry overlapping signal. A single consolidated string reduces token cost while preserving the essential content advisory signal — including severity detail when the primary signal is missing.

**Input data:**
- `maturity_rating` (str) — G / PG / PG-13 / R / NC-17
- `maturity_reasoning` (list[str]) — why it received that rating (may be empty/absent)
- `parental_guide_items` (list[dict]) — category + severity pairs (fallback only)

**Output data:**
- `maturity_summary` (str) — one of:
  - Primary (maturity_reasoning exists, ≥1 item): `"R — violence, strong language, brief nudity"`
  - Fallback (maturity_reasoning absent): `"R — severe violence, moderate language, mild nudity"` (compact parental_guide_items as "severity category" pairs, comma-separated)

**Dropped from direct LLM input:**
- `maturity_rating` — folded into maturity_summary
- `maturity_reasoning` — folded into maturity_summary
- `parental_guide_items` — either dropped (when reasoning exists) or compacted into fallback format

### 3.3 Sparse Movie Input Assessment

**What:** Evaluate the movie's data completeness and determine which generations to skip.

**Why:** Running an LLM call on near-empty inputs wastes tokens and produces low-quality output that harms search when embedded. Better to skip and let the movie match on its strong vector spaces only.

**Input data:**
- All BaseMovie fields relevant to threshold checks (see [Section 6](#sparse-movie-skip-conditions) for per-generation thresholds)

**Output data:**
- `generations_to_run` (set[str]) — which of the 7 metadata types should be generated for this movie
- `skip_reasons` (dict[str, str]) — for logging: which generations were skipped and why

**Note:** Plot events is always included in `generations_to_run`. It never skips based on input assessment (Decision 8). It may still fail at runtime due to API errors, triggering the partial pipeline (Decision 9).

---

## 4. Wave 1: Foundation Metadata <a name="wave-1"></a>

Two tasks run in parallel. Both are independent — neither depends on the other's output.

### 4.1 Plot Events

**Purpose:** Extract WHAT HAPPENS — concrete events, characters, settings. Produces `plot_summary` which feeds Wave 2 calls that need plot context.

**Failure mode:** If this fails (API error/timeout), the partial pipeline runs — reception, production, watch_context, and viewer_experience (if review_insights_brief exists) still proceed. Plot_analysis and narrative_techniques are skipped.

**No-hallucination rule:** The prompt must explicitly instruct: "Only describe what is evident from the provided data. Do not supplement with your own knowledge of this film. If data is limited, produce a shorter summary rather than inventing details." This is critical for the ~34% of movies lacking synopsis/summaries — they get thin but accurate plot_summaries.

#### Inputs

| Input | Status | Notes |
|-------|--------|-------|
| `title (year)` | **Changed** | Was `title` only. Now formatted as `"Title (Year)"` for temporal grounding and disambiguation. |
| `overview` | Unchanged | Marketing summary; important here because plot_synopses may be absent |
| `plot_summaries` | Unchanged | Shorter user-written IMDB summaries |
| `plot_synopses` | Unchanged | Longest/most detailed plot recounts (primary truth source) |
| `plot_keywords` | Unchanged | Plot-specific keywords only (not overall). Directly relevant to identifying plot events. |

#### Minimum required inputs

**Always attempts.** No skip condition. Even with just title + overview + keywords, the model produces a summary grounded in the provided data. Thin input → thin but accurate output, which is acceptable.

#### Outputs

| Output | Status | Notes |
|--------|--------|-------|
| `plot_summary` | Unchanged | Detailed chronological spoiler-containing summary. **Fed to Wave 2 calls as `plot_synopsis` where applicable.** May be thin for sparse movies — this is by design. |
| `setting` | Unchanged | 10-word where/when phrase. |
| `major_characters` | Unchanged | Essential characters with name, description, role, motivations. |

**No justification fields exist in this schema currently — no change needed.**

---

### 4.2 Reception

**Purpose:** Extract what audiences and critics think about the movie. **Dual purpose:** also produces a `review_insights_brief` that serves as consolidated review input for all Wave 2 calls.

**Failure mode:** Graceful. If reception fails or is skipped, Wave 2 calls run without `review_insights_brief` — they still have plot_synopsis, keywords, and genres. The movie proceeds with `reception_metadata = None`. Viewer_experience additionally requires review_insights_brief if plot_synopsis is also unavailable.

#### Inputs

| Input | Status | Notes |
|-------|--------|-------|
| `title (year)` | **Changed** | Now formatted as `"Title (Year)"`. |
| `reception_summary` | Unchanged | Externally generated summary of audience opinion |
| `audience_reception_attributes` | Unchanged | Key attributes with sentiment labels |
| `featured_reviews` | Unchanged | Up to 5 full review texts — this is now the **only call** that receives raw reviews |

#### Minimum required inputs

At least ONE of: `featured_reviews` (≥1 review), `reception_summary`, or `audience_reception_attributes` (≥2 attributes). Without any reception input, the model would fabricate reception data from parametric knowledge alone.

**If below threshold:** Skip reception entirely. `reception_metadata = None`, `review_insights_brief = None`. Wave 2 calls proceed without reception signal.

#### Outputs

| Output | Status | Notes |
|--------|--------|-------|
| `new_reception_summary` | Unchanged | 2-3 sentence evaluative summary of what viewers thought |
| `praise_attributes` | Unchanged | 0-4 short tag-like phrases for what audiences enjoyed |
| `complaint_attributes` | Unchanged | 0-4 short tag-like phrases for what audiences disliked |
| `review_insights_brief` | **NEW** | ~150-250 tokens. A dense paragraph capturing the key thematic, emotional, structural, evaluative, and **source-material** observations from the raw reviews. Designed to serve as a compact replacement for raw reviews in downstream calls. Not embedded, not stored — purely an intermediate input for Wave 2. |

**Why `review_insights_brief` is distinct from `new_reception_summary`:** The reception summary is evaluative — focused on "was it good/bad and why." The insights brief is descriptive — focused on "what did reviewers observe?" covering themes, emotions, structural elements, and source material mentions. The summary answers "what did people think?" The brief answers "what did people notice?" Wave 2 calls need observations, not evaluations.

**Source material extraction:** The prompt must explicitly instruct the model to include any source material observations from reviews in the brief (e.g., "reviewers described it as a faithful adaptation of the novel," "noted it was inspired by real events"). This enables source_of_inspiration to benefit from review signal without receiving raw reviews.

---

## 5. Wave 2: Analysis Metadata <a name="wave-2"></a>

Up to 5 tasks run in parallel, all dependent on Wave 1 outputs:
- `plot_synopsis` from plot_events (available if plot_events succeeded)
- `review_insights_brief` from reception (may be None if reception failed or was skipped)

### 5.1 Plot Analysis

**Purpose:** Extract WHAT TYPE OF STORY this is — themes, lessons, core concepts. Powers queries like "movie about the cost of revenge" or "redemption story."

**Failure mode:** Graceful. `plot_analysis_metadata = None`.

**Requires plot_events success.** Skipped if plot_events failed.

#### Inputs

| Input | Status | Notes |
|-------|--------|-------|
| `title (year)` | **Changed** | Now formatted as `"Title (Year)"`. |
| `genres` | Unchanged | |
| `plot_synopsis` | Unchanged | From plot_events Wave 1 output |
| `plot_keywords` | Unchanged | Plot-specific keywords only. Thematic keywords support theme identification. |
| `review_insights_brief` | **Changed** | Replaces `reception_summary` + `featured_reviews` (up to 5 raw reviews). The brief captures thematic observations at ~150-250 tokens instead of ~550-2600 tokens. |
| ~~`overview`~~ | **Removed** | Superseded by `plot_synopsis`. The synopsis contains everything the overview has plus far more detail. ~30-80 tokens saved. |
| ~~`reception_summary`~~ | **Removed** | Subsumed by `review_insights_brief`. |
| ~~`featured_reviews`~~ | **Removed** | Subsumed by `review_insights_brief`. Saves ~500-2500 tokens. |

#### Minimum required inputs

`plot_synopsis` (requires plot_events success). Plot analysis can produce reasonable output from plot + genres + keywords alone, even without review data. Always runs when plot_events succeeds.

#### Outputs

| Output | Status | Notes |
|--------|--------|-------|
| `core_concept` | Unchanged | Single dominant story concept (6 words or less). |
| `genre_signatures` | Unchanged | 2-6 search-query-like genre phrases. |
| `conflict_scale` | Unchanged | Scale of consequences. |
| `character_arcs` | Unchanged | 1-3 key character transformations. |
| `themes_primary` | Unchanged | 1-3 core thematic concepts. |
| `lessons_learned` | Unchanged | 0-3 key takeaways. |
| `generalized_plot_overview` | Unchanged | 1-3 sentence thematic overview. |
| ~~justification fields~~ | **Removed** | `explanation_and_justification` on core_concept, themes, lessons, arcs. Pending empirical validation. ~50-150 output tokens saved. |

**Note on `arc_transformation_description`:** This field on `CharacterArc` is not a justification — it's a longer explanation of the arc that may help the model produce a more accurate `arc_transformation_label`. However, it IS never embedded. Recommend keeping it for now but flagging for the same empirical validation as justifications.

---

### 5.2 Viewer Experience

**Purpose:** Extract what it FEELS LIKE to watch the movie — emotional, sensory, cognitive experience. Powers queries like "edge of your seat thriller" or "cozy feel-good movie."

**Failure mode:** Graceful. `viewer_experience_metadata = None`.

**Can run without plot_synopsis** if review_insights_brief exists. Reviews carry strong emotional/tonal signal independently — "I was on the edge of my seat" gives viewer experience plenty to work with.

#### Inputs

| Input | Status | Notes |
|-------|--------|-------|
| `title (year)` | **Changed** | Now formatted as `"Title (Year)"`. |
| `genres` | Unchanged | |
| `plot_synopsis` | Unchanged | From plot_events Wave 1 output. May be absent if plot_events failed — viewer_experience can still run from review data. |
| `merged_keywords` | **Changed** | Was `plot_keywords` + `overall_keywords` as separate inputs. Now receives the deduplicated merged keyword list (both types). Broad signal useful for emotional/sensory assessment. |
| `maturity_summary` | **Changed** | Replaces `maturity_rating` + `maturity_reasoning` + `parental_guide_items` as three separate inputs. Single consolidated string with conditional fallback (see Pre-Consolidation 3.2). |
| `review_insights_brief` | **Changed** | Replaces `reception_summary` + `audience_reception_attributes` + `featured_reviews`. Saves ~550-2650 tokens while preserving emotional observations from reviews. |
| ~~`plot_keywords`~~ | **Removed** | Merged into `merged_keywords`. |
| ~~`overall_keywords`~~ | **Removed** | Merged into `merged_keywords`. |
| ~~`maturity_rating`~~ | **Removed** | Merged into `maturity_summary`. |
| ~~`maturity_reasoning`~~ | **Removed** | Merged into `maturity_summary`. |
| ~~`parental_guide_items`~~ | **Removed** | Conditionally folded into `maturity_summary` fallback or dropped when reasoning exists. Severity detail preserved when needed. |
| ~~`reception_summary`~~ | **Removed** | Subsumed by `review_insights_brief`. |
| ~~`audience_reception_attributes`~~ | **Removed** | Subsumed by `review_insights_brief`. |
| ~~`featured_reviews`~~ | **Removed** | Subsumed by `review_insights_brief`. |

#### Minimum required inputs

`plot_synopsis` OR `review_insights_brief` — at least one must exist. With both absent, the model has only title + genres + keywords + maturity, which is insufficient for movie-specific emotional assessment.

**If below threshold:** Skip viewer experience. The movie's viewer_experience vector is omitted from Qdrant.

#### Outputs

All 8 sections unchanged:

| Output | Status | Notes |
|--------|--------|-------|
| `emotional_palette` | Unchanged | Terms + negations for dominant emotions |
| `tension_adrenaline` | Unchanged | Terms + negations for stress/energy/suspense |
| `tone_self_seriousness` | Unchanged | Terms + negations for earnest vs ironic |
| `cognitive_complexity` | Unchanged | Terms + negations for mental effort |
| `disturbance_profile` | Unchanged | Optional. Terms + negations for unsettling elements |
| `sensory_load` | Unchanged | Optional. Terms + negations for visual/auditory intensity |
| `emotional_volatility` | Unchanged | Optional. Terms + negations for tone changes |
| `ending_aftertaste` | Unchanged | Terms + negations for final emotion |
| ~~justification fields~~ | **Removed** | One per section (8 total). Pending empirical validation. ~80-160 output tokens saved. |

---

### 5.3 Watch Context (MOVED from Wave 1)

**Purpose:** Extract WHY and WHEN someone would choose to watch this movie. Powers queries like "date night movie" or "something to watch high."

**Failure mode:** Graceful. `watch_context_metadata = None`.

**No plot information.** Watch context deliberately receives zero plot info (no overview, no synopsis). It answers "watch this if you want X attributes" — not "watch this if you want these specific events to happen." Plot detail risks anchoring the model on narrative events rather than experiential attributes.

#### Inputs

| Input | Status | Notes |
|-------|--------|-------|
| `title (year)` | **Changed** | Now formatted as `"Title (Year)"`. |
| `genres` | Unchanged | Strong signal for occasion matching: horror → halloween, romance → date night. |
| `overall_keywords` | **Changed** | Was `plot_keywords` + `overall_keywords` as separate inputs. Now receives `overall_keywords` only. Categorical tags ("family-friendly", "cult classic", "dark comedy") inform viewing occasions; plot-specific tags ("murder investigation") don't. |
| `maturity_summary` | **Added** | Content advisory context — "don't watch this with kids" is a valid watch context signal. |
| `review_insights_brief` | **Changed** | Replaces `reception_summary` + `audience_reception_attributes` + `featured_reviews`. The primary value driver for watch context — tells you how the movie *feels* and what stands out. Saves ~550-2650 tokens. |
| ~~`overview`~~ | **Removed** | No plot info in watch context. Experiential inputs (reviews, genres, keywords) are sufficient. |
| ~~`plot_synopsis`~~ | **Not added** | Originally proposed in the first draft. Removed after design review — plot detail anchors on events rather than attributes. |
| ~~`plot_keywords`~~ | **Removed** | Plot-specific keywords ("murder investigation", "time travel") provide limited signal for viewing occasion decisions. |
| ~~`reception_summary`~~ | **Removed** | Subsumed by `review_insights_brief`. |
| ~~`audience_reception_attributes`~~ | **Removed** | Subsumed by `review_insights_brief`. |
| ~~`featured_reviews`~~ | **Removed** | Subsumed by `review_insights_brief`. |

#### Minimum required inputs

`genres` (≥1 entry). Watch scenarios can be reasonably inferred from genres + keywords even without review data (a horror movie is probably a "halloween movie"; a romance is a "date night movie"). Since virtually every movie has at least one genre, watch context almost always runs.

#### Outputs

All 4 sections unchanged:

| Output | Status | Notes |
|--------|--------|-------|
| `self_experience_motivations` | Unchanged | 4-8 self-focused experiential reasons to watch |
| `external_motivations` | Unchanged | 1-4 value-beyond-viewing reasons |
| `key_movie_feature_draws` | Unchanged | 1-4 standout attribute draws |
| `watch_scenarios` | Unchanged | 3-6 real-world occasions and contexts |
| ~~justification fields~~ | **Removed** | One per section (4 total). Pending empirical validation. ~40-80 output tokens saved. |

---

### 5.4 Narrative Techniques

**Purpose:** Extract HOW the story is told — cinematic narrative craft, structure, storytelling mechanics. Powers queries like "movie with an unreliable narrator" or "non-linear timeline."

**Failure mode:** Graceful. `narrative_techniques_metadata = None`.

**Requires plot_events success** with sufficient output. Structural analysis needs plot detail to identify techniques.

#### Inputs

| Input | Status | Notes |
|-------|--------|-------|
| `title (year)` | **Changed** | Now formatted as `"Title (Year)"`. |
| `genres` | **Added** | ~5-10 tokens. Helps ground structural analysis — "mystery" implies information control techniques, "documentary" implies specific POV structures. Essentially free. |
| `plot_synopsis` | Unchanged | From plot_events Wave 1 output |
| `overall_keywords` | **Changed** | Was `plot_keywords` + `overall_keywords` as separate inputs. Now receives `overall_keywords` only. Structural tags ("nonlinear timeline", "unreliable narrator") tend to live in overall keywords. Plot keywords add noise without structural signal. |
| `review_insights_brief` | **Changed** | Replaces `reception_summary` + `featured_reviews`. Saves ~550-2600 tokens. Reviews often reveal structural observations (e.g., "the twist was predictable", "the non-linear storytelling was confusing") — the brief preserves these structural insights. |
| ~~`plot_keywords`~~ | **Removed** | Plot-specific keywords rarely carry structural narrative signal. Overall keywords are the relevant source. |
| ~~`reception_summary`~~ | **Removed** | Subsumed by `review_insights_brief`. |
| ~~`featured_reviews`~~ | **Removed** | Subsumed by `review_insights_brief`. |

#### Minimum required inputs

`plot_synopsis` with meaningful content (>100 words). Structural narrative analysis requires sufficient plot detail to identify techniques. A thin 2-sentence plot_summary doesn't provide enough material to confidently identify POV, temporal structure, or information control patterns.

**If below threshold:** Skip narrative techniques. The movie's narrative_techniques vector is omitted from Qdrant.

#### Outputs

All 11 sections unchanged:

| Output | Status | Notes |
|--------|--------|-------|
| `pov_perspective` | Unchanged | 1-2 phrases |
| `narrative_delivery` | Unchanged | 1-2 phrases |
| `narrative_archetype` | Unchanged | 1 phrase |
| `information_control` | Unchanged | 1-2 phrases |
| `characterization_methods` | Unchanged | 1-3 phrases |
| `character_arcs` | Unchanged | 1-3 phrases |
| `audience_character_perception` | Unchanged | 1-3 phrases |
| `conflict_stakes_design` | Unchanged | 1-2 phrases |
| `thematic_delivery` | Unchanged | 1-2 phrases |
| `meta_techniques` | Unchanged | 0-2 phrases |
| `additional_plot_devices` | Unchanged | Misc phrases |
| ~~justification fields~~ | **Removed** | One per section (11 total). Pending empirical validation. ~100-220 output tokens saved. |

---

### 5.5 Production (2 sub-calls in parallel)

**Purpose:** Extract how the movie was produced in the real world — source material, production medium, production-related keywords. Powers queries like "based on a true story" or "stop motion animation."

**Failure mode:** Graceful. `production_metadata = None`.

#### Sub-call A: Production Keywords

**Purpose:** Filter the keyword list to only production-relevant keywords. The LLM classifies, not generates.

##### Inputs

| Input | Status | Notes |
|-------|--------|-------|
| `title (year)` | **Changed** | Now formatted as `"Title (Year)"`. |
| `merged_keywords` | **Changed** | Was `overall_keywords` only. Now receives the full deduplicated merged keyword list (both plot and overall). Gives the classifier more material — some plot_keywords may have production relevance (e.g., "shot on location"). The model's job is filtering, so extra keywords just mean more to filter, not more noise in the output. |

##### Minimum required inputs

`merged_keywords` with ≥5 entries. Without meaningful keywords to filter, the production keywords call has nothing to classify. Almost always available (only 21 movies lack all keywords).

**If below threshold:** Skip production keywords sub-call. Production metadata uses only source_of_inspiration output.

##### Outputs

| Output | Status | Notes |
|--------|--------|-------|
| `terms` | Unchanged | Filtered subset of input keywords that relate to production |
| ~~`justification`~~ | **Removed** | Pending empirical validation. ~10-20 output tokens saved. |

#### Sub-call B: Source of Inspiration

**Purpose:** Determine what real-world sources the movie is based on and how it was produced visually.

**Parametric knowledge allowed:** The prompt explicitly states: "If you are highly confident about the source material based on your knowledge, include it." This allows the LLM to contribute facts about well-known adaptations that may not appear in keywords or reviews. Combined with `"Title (Year)"` format for disambiguation.

##### Inputs

| Input | Status | Notes |
|-------|--------|-------|
| `title (year)` | **Changed** | Now formatted as `"Title (Year)"`. Particularly valuable here — helps model identify known adaptations and disambiguate remakes. |
| `plot_synopsis` | Unchanged | From plot_events Wave 1 output. May reveal the story follows a known real event. |
| `merged_keywords` | **Changed** | Was `plot_keywords` + `overall_keywords` concatenated. Now receives the pre-deduplicated merged list. Same content, already cleaned. |
| `review_insights_brief` | **Added** | Was raw `featured_reviews` (removed in first draft, now restored via brief). Reviews frequently mention source material: "faithful adaptation of the novel," "inspired by true events." The brief captures these observations at ~150-250 tokens instead of ~500-2000 tokens of raw reviews. |
| ~~`featured_reviews`~~ | **Removed** | Replaced by `review_insights_brief`. The brief preserves source-material observations at a fraction of the token cost. |

##### Minimum required inputs

`title (year)` — always available. With parametric knowledge allowed, the model can determine source material from title + year alone for well-known films. Keywords and review_insights_brief improve accuracy but aren't strictly required. Always runs.

##### Outputs

| Output | Status | Notes |
|--------|--------|-------|
| `sources_of_inspiration` | Unchanged | E.g., "based on a true story", "based on a novel" |
| `production_mediums` | Unchanged | E.g., "live action", "hand-drawn animation" |
| ~~`justification`~~ | **Removed** | Pending empirical validation. ~20-40 output tokens saved. |

---

## 6. Sparse Movie Skip Conditions <a name="sparse-movie-skip-conditions"></a>

Each generation has a minimum data threshold. If a movie's data falls below the threshold, that generation is skipped — no LLM call, no embedding for that vector space. The movie is still searchable via its other vectors.

| Generation | Minimum Required | Skip Impact | Expected Skip Rate |
|-----------|-----------------|-------------|-------------------|
| **Plot Events** | Always attempts (no skip) | If fails (API error): partial pipeline — skip plot_analysis + narrative_techniques | ~0% skip (failures only) |
| **Reception** | ≥1 of: featured_reviews, reception_summary, or audience_reception_attributes (≥2) | No reception vector; no review_insights_brief for Wave 2 | Low-Medium (~5-10%) |
| **Plot Analysis** | plot_synopsis (requires plot_events success) | No plot_analysis vector | ~0% (only on plot_events failure) |
| **Viewer Experience** | plot_synopsis OR review_insights_brief (at least one) | No viewer_experience vector | Very low — only when both plot_events fails AND reception is skipped/fails |
| **Watch Context** | genres ≥1 | No watch_context vector | ~0% — virtually every movie has genres |
| **Narrative Techniques** | plot_synopsis >100 words (requires plot_events success + sufficient output) | No narrative_techniques vector | Low (~3-5%) — depends on plot_events output quality, higher for the ~34% sparse movies |
| **Production Keywords** | merged_keywords ≥5 | Production metadata uses only source_of_inspiration | Very low (<1%) — only 21 movies lack all keywords |
| **Source of Inspiration** | title + year (always available; parametric knowledge allowed) | Always runs | 0% |

### The partial pipeline

If **plot_events fails** (API error — not data sparsity), the following still run:
- **Reception** — independent, doesn't need plot_synopsis
- **Watch context** — no plot input by design
- **Viewer experience** — if review_insights_brief exists (reviews provide independent emotional signal)
- **Production** — keywords-driven; source_of_inspiration uses title + year + parametric knowledge

The following are skipped:
- **Plot analysis** — requires plot detail for thematic analysis
- **Narrative techniques** — requires plot detail for structural analysis

The movie gets 3-5 partial vectors (reception, watch_context, viewer_experience, production_keywords, source_of_inspiration) and is still findable for queries matching those vector spaces.

### The sparse data cascade

For the ~34% of movies lacking synopsis/summaries:
1. Plot events runs with title + overview + keywords → produces a thin but accurate plot_summary (no-hallucination rule)
2. The thin plot_summary cascades to Wave 2 as a shorter-than-usual plot_synopsis
3. Narrative techniques may be skipped if plot_synopsis is <100 words
4. Other Wave 2 calls work with the thin synopsis alongside review_insights_brief, genres, and keywords
5. Output quality is lower but grounded in real data — no fabricated information

---

## 7. Estimated Savings Summary <a name="estimated-savings"></a>

### Per-movie token savings (for a movie with all data present)

| Change | Input Tokens Saved | Output Tokens Saved |
|--------|-------------------|-------------------|
| Review consolidation via `review_insights_brief` | ~3000-5000 | 0 |
| Remove justification fields | 0 | ~300-670 |
| Drop `featured_reviews` from source_of_inspiration (replaced by brief) | ~350-1750 | 0 |
| Drop `overview` from plot_analysis | ~30-80 | 0 |
| Drop `overview` + `plot_synopsis` from watch_context | ~30-680 | 0 |
| Drop `audience_reception_attributes` from watch_context | ~30-60 | 0 |
| Conditional maturity consolidation | ~20-60 | 0 |
| Selective keyword routing (reduced keywords per call) | ~20-100 | 0 |
| **Total** | **~3480-7730** | **~300-670** |

### Additional savings for sparse movies

For movies that skip reception: 1 fewer LLM call + no reception embedding tokens.
For movies that skip narrative_techniques: 1 fewer LLM call + no narrative_techniques embedding tokens.

### Quality improvements (zero cost)

| Change | Quality Impact |
|--------|---------------|
| "Title (Year)" format | Better temporal grounding, remake disambiguation across all generations |
| Watch context receives no plot info | More experiential, less event-anchored watch scenarios |
| Narrative techniques receives `genres` | Better-grounded structural analysis |
| Selective keyword routing | Each generation gets the relevant keyword type, reducing noise |
| Sparse movies skip weak generations | Removes garbage vectors from search index, improving precision |
| Source of inspiration uses parametric knowledge | Captures adaptation facts the data may not contain |
| Review insights brief captures source material | Source_of_inspiration gets review signal without raw review cost |
| Conditional maturity fallback | Preserves severity detail when primary reasoning is absent |
| No-hallucination rule for plot events | Thin but accurate over detailed but fabricated |
| Partial pipeline on failure | Movies retain searchability via non-plot vectors instead of being dropped |

### Changes requiring empirical validation

| Change | What to Test | How |
|--------|-------------|-----|
| Justification field removal | Does output quality degrade without explicit justifications? | Generate metadata for 50-100 movies with/without justification fields; blind-evaluate quality |
| `review_insights_brief` fidelity | Does the brief preserve sufficient signal for downstream calls? | Compare Wave 2 output quality using raw reviews vs. brief for 50 movies |
| Dual-purpose reception quality | Does adding review_insights_brief to reception degrade the evaluative outputs? | Compare reception summary + praise/complaint quality with and without the brief field |
| Sparse movie thresholds | Are the word-count thresholds correct? | Sample movies at various data density levels; evaluate metadata quality to find the quality cliff |

---

## 8. Pending Optimizations <a name="pending-optimizations"></a>

### Synopsis compression (empirical testing required)

`plot_synopsis` can be 300-600 tokens of natural prose. Compressing it would save tokens across every Wave 2 call that receives it. Three variants to test against natural prose on 50 movies:

**Variant A — Caveman speak (LLM-produced):** Prompt instruction: "Write in compressed telegraphic style. Omit articles, prepositions, and filler words. Preserve all character names, actions, and emotional beats." Example: "John discovers portal basement. Travels 1920s, falls love Sarah. Realizes returning destroys future family. Chooses stay, portal closes." Estimated compression: ~40-60% of original token count.

**Variant B — Article/connector stripping (programmatic post-processing):** Remove articles (a, an, the), common prepositions (in, on, at, to, of, for, with, by), and filler phrases. Cheaper (no LLM cost), deterministic, but cruder — may break meaning in edge cases ("the man in the iron mask" → "man iron mask"). Estimated compression: ~20-30%.

**Variant C — Structured synopsis:** Instead of prose, output a structured format: PREMISE (1 sentence), KEY EVENTS (3-5 bullet fragments), RESOLUTION (1 sentence), EMOTIONAL ARC (1 phrase). Naturally compresses because bullets discourage verbose prose. Bonus: sections could be selectively routed if needed. Estimated compression: ~30-50%.

**Evaluation criteria:** Compare downstream metadata quality (Wave 2 outputs) and embedding cosine similarity against natural prose baseline.

### gpt-5-nano review consolidation (contingency)

If dual-purpose reception shows quality degradation (evaluative summary and descriptive brief cross-contaminate), a separate gpt-5-nano call for review consolidation is the fallback. The brief's schema and downstream consumption would be identical — only the producer changes. This adds one LLM call but uses a cheaper model, and separates the evaluative and descriptive tasks.
