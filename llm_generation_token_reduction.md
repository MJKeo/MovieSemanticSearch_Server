# LLM Generation Token Reduction Guide

## Executive Summary

**Current cost:** ~$0.025/movie (28K input tokens + 8K output tokens across 8 LLM calls)
**Target scale:** 100K movies → $2,500 total under current design

This guide identifies concrete strategies to reduce both input and output token usage without meaningful quality loss to the semantic search embeddings. The strategies are organized by estimated impact, with the highest-value changes first.

**Key findings:**
- **Featured reviews** account for ~9,000 input tokens/movie (32% of all input), duplicated across 6 of 8 calls. This is the single largest cost driver and the most compressible.
- **System prompts** consume ~8,000 tokens/movie (29%) and can be compressed ~40% without losing instruction quality.
- **Plot synopsis propagation** across dependent calls adds ~3,600 tokens of repetition.
- **Output schemas force ~1,500+ wasted output tokens** via justification fields, explanation fields, and character metadata fields that are never used in the final embeddings — this is especially costly because output tokens are ~4x the price of input tokens.
- **Reasoning tokens** (internal chain-of-thought in gpt-5-mini) likely account for ~4,000-5,000 of the 8K reported output tokens, and can be reduced by simplifying prompts and lowering reasoning effort where safe.

---

## Table of Contents

1. [Current System Analysis](#1-current-system-analysis)
2. [Input Token Reduction Strategies](#2-input-token-reduction-strategies)
3. [Output Token Reduction Strategies](#3-output-token-reduction-strategies)
4. [Structural Optimizations](#4-structural-optimizations)
5. [Priority Matrix](#5-priority-matrix)

---

## 1. Current System Analysis

### 1.1 LLM Call Flow

The system makes **8 API calls per movie** to `gpt-5-mini`, organized in two parallel waves:

**Wave 1** (3 calls, parallel):
| Call | Reasoning Effort | Primary Inputs |
|------|-----------------|----------------|
| `plot_events` | minimal | title, overview, plot_summaries[], plot_synopses[], plot_keywords |
| `watch_context` | medium | title, genres, overview, plot_keywords, overall_keywords, reception_summary, audience_reception_attributes, featured_reviews[:5] |
| `reception` | low | title, reception_summary, audience_reception_attributes, featured_reviews[:5] |

**Wave 2** (5 calls, parallel — depends on `plot_events` output for `plot_synopsis`):
| Call | Reasoning Effort | Primary Inputs |
|------|-----------------|----------------|
| `plot_analysis` | low | title, genres, overview, plot_synopsis, plot_keywords, reception_summary, featured_reviews[:5] |
| `viewer_experience` | low | title, genres, plot_synopsis, plot_keywords, overall_keywords, maturity_rating, maturity_reasoning, parental_guide_items, reception_summary, audience_reception_attributes, featured_reviews[:5] |
| `narrative_techniques` | medium | title, plot_synopsis, plot_keywords, overall_keywords, reception_summary, featured_reviews[:5] |
| `production_keywords` | low | title, overall_keywords |
| `source_of_inspiration` | low | title, plot_keywords + overall_keywords, plot_synopsis, featured_reviews[:5] |

### 1.2 Input Token Breakdown (estimated per movie)

| Component | Tokens/Call | Calls Used In | Total Tokens | % of Total |
|-----------|-----------|---------------|--------------|------------|
| **System prompts** | varies | 8 | **~8,000** | **29%** |
| **featured_reviews[:5]** | ~1,500 | 6 | **~9,000** | **32%** |
| **plot_synopses (raw)** | ~1,875 | 1 | **~1,875** | **7%** |
| **plot_synopsis (LLM-generated)** | ~900 | 4 | **~3,600** | **13%** |
| **plot_summaries** | ~580 | 1 | **~580** | **2%** |
| **JSON schema overhead** | ~300 | 8 | **~2,400** | **9%** |
| All other fields combined | varies | varies | **~2,500** | **9%** |
| **Total** | | | **~28,000** | **100%** |

### 1.3 Output Token Breakdown (estimated per movie)

| Component | Content Tokens | Reasoning Tokens (est.) | Notes |
|-----------|---------------|------------------------|-------|
| `plot_events` | ~1,100 | ~200 (minimal) | plot_summary is ~900 tokens alone |
| `plot_analysis` | ~600 | ~400 (low) | Includes several unused explanation fields |
| `viewer_experience` | ~720 | ~600 (low) | 8 sections, each with justification |
| `watch_context` | ~260 | ~800 (medium) | 4 sections with justification |
| `narrative_techniques` | ~440 | ~1,000 (medium) | 11 sections, each with justification |
| `production_keywords` | ~50 | ~200 (low) | |
| `source_of_inspiration` | ~60 | ~300 (low) | |
| `reception` | ~140 | ~300 (low) | |
| **Total** | **~3,370** | **~3,800** | **~7,170 total** |

> **Critical insight:** Roughly half of all output tokens are invisible reasoning tokens that don't appear in the final output but still cost 4x the input token rate. Reducing prompt complexity directly reduces reasoning token consumption.

### 1.4 What Reaches the Final Embeddings

By tracing the `__str__()` methods and `create_*_vector_text()` functions in `vectorize.py`, only these output fields actually make it into the embedding text:

| Schema | Fields Used in Embeddings | Fields NOT Used (wasted output tokens) |
|--------|--------------------------|----------------------------------------|
| `PlotEventsMetadata` | plot_summary, setting, major_characters.{name, description, primary_motivations} | major_characters.**role** |
| `PlotAnalysisMetadata` | core_concept.**core_concept_label**, genre_signatures, conflict_scale, character_arcs.**arc_transformation_label**, themes_primary.**theme_label**, lessons_learned.**lesson_label**, generalized_plot_overview | core_concept.**explanation_and_justification**, character_arcs.{**character_name**, **arc_transformation_description**}, themes_primary.**explanation_and_justification**, lessons_learned.**explanation_and_justification** |
| `ViewerExperienceMetadata` | terms[], negations[] (per section) | **justification** (per section, ×8) |
| `WatchContextMetadata` | terms[] (per section) | **justification** (per section, ×4) |
| `NarrativeTechniquesMetadata` | terms[] (per section) | **justification** (per section, ×11) |
| `ProductionMetadata` | production_keywords.terms[], sources_of_inspiration.{sources, mediums} | production_keywords.**justification**, sources_of_inspiration.**justification** |
| `ReceptionMetadata` | new_reception_summary, praise_attributes[], complaint_attributes[] | *(none — all fields used)* |

**Total justification/explanation fields generating wasted tokens: ~27 fields**

---

## 2. Input Token Reduction Strategies

### 2.1 Featured Reviews: Condense Before Sending

**Impact: HIGH (~6,000-7,500 input tokens saved, ~22-27% of total input)**

Featured reviews are the single largest input cost, consuming ~9,000 tokens across 6 calls. The raw review text contains:
- HTML entities (`&#39;`, `<br/>`, `&quot;`) — pure noise
- Personal anecdotes and tangents ("My friends knowing that I'm a huge film buff asked me to...")
- Lengthy plot recaps that duplicate information already in `plot_synopsis`
- Repetitive praise across reviews

**Current state:** Each `IMDBFeaturedReview` has a `summary` (~20-30 tokens) and `text` (~200-400 tokens). The system sends the full `text` of 5 reviews to 6 different calls.

**Recommendations (choose one or combine):**

#### Option A: Send Review Summaries Only to Most Calls (Recommended)
The `summary` field is surprisingly information-dense. For calls that need reviews primarily for sentiment/reception signals (not deep thematic analysis), summaries alone carry ~80% of the useful signal.

- **Full reviews (summary + text):** Send to `plot_analysis` and `narrative_techniques` only — these need review text to identify thematic interpretation and structural observations.
- **Summaries only:** Send to `viewer_experience`, `watch_context`, `reception`, and `source_of_inspiration`.
- **Estimated savings:** ~1,400 tokens × 4 calls = ~5,600 input tokens

#### Option B: Pre-Process Reviews into a Condensed Summary
Before any LLM calls, run a single cheap extraction pass (haiku-tier model or even regex + heuristics) that produces a ~150-200 token "review digest" covering: key praised attributes, key complaints, emotional descriptors, thematic observations. Pass this digest instead of raw reviews.

- **Estimated savings:** ~8,000 input tokens (from ~9,000 to ~1,200)
- **Trade-off:** Adds one extra LLM call (~500 tokens) but saves massively on 6 downstream calls. Net savings ~7,500 tokens.
- **Risk:** The pre-processing step could filter out nuanced observations that a review's full text captures. Test on a sample before committing.

#### Option C: Reduce Review Count from 5 to 3
The top 3 reviews typically capture the primary consensus. Reviews 4-5 are often redundant.

- **Estimated savings:** ~600 tokens × 6 calls = ~3,600 input tokens
- **Trade-off:** Minimal quality loss for well-reviewed movies; riskier for movies where opinions diverge and review 4-5 provide the minority signal.

#### Combined recommendation: Option A + C
Send the full text of 3 reviews to plot_analysis and narrative_techniques. Send only summaries of 3 reviews everywhere else. **Estimated savings: ~7,000 input tokens (25% of total).**

### 2.2 System Prompt Compression

**Impact: MEDIUM (~2,500-3,500 input tokens saved, ~9-13% of total input)**

The system prompts are well-designed but verbose. They repeat patterns, include excessive examples, and use full sentences where terse instructions would suffice. The prompts total ~8,000 tokens across all calls.

**Key offenders:**
- `VIEWER_EXPERIENCE_SYSTEM_PROMPT`: ~2,200 tokens. Contains 8 section descriptions with 4+ example terms each, plus 4+ example negations each. The examples are helpful but could be cut by ~40%.
- `NARRATIVE_TECHNIQUES_SYSTEM_PROMPT`: ~1,700 tokens. 11 sections with examples. Many sections could share a single "style rules" block rather than repeating formatting guidelines.
- `WATCH_CONTEXT_SYSTEM_PROMPT`: ~1,400 tokens. Similar pattern of per-section example bloat.

**Recommendations:**

#### A: Deduplicate Per-Section Instructions
All three multi-section prompts (viewer_experience, watch_context, narrative_techniques) repeat the same formatting rules per section. Move shared rules to the top once and reference them.

**Before (repeated per section):**
```
1) emotional_palette
What to capture:
- The dominant emotions the average viewer feels while watching this movie.
- Must have significant evidence for this emotion in the provided input...
- Include terms users type for emotions.
- Repetition through synonyms is encouraged.
Examples of terms:
- "uplifting and hopeful", "cozy", "laugh out loud", "nostalgic", "emotional rollercoaster"...
Examples of negations:
- "not too sad", "not comforting", "not funny", "not cheesy"
```

**After:**
```
GLOBAL: All terms are 1-5 word search queries. Negations start with "not"/"no". Synonyms encouraged.

1) emotional_palette: dominant viewer emotions. Ex: "cozy", "laugh out loud", "nostalgic"
   Neg ex: "not too sad", "not cheesy"
```

This can compress each section from ~80-100 tokens to ~30-40 tokens. For 23 total sections across the three prompts, this saves ~1,000-1,500 tokens.

#### B: Trim Example Lists
Most sections list 6-12 example terms. The model understands the pattern after 3-4 examples. Cut example lists to 3-4 representative examples per section.

**Estimated savings:** ~500-800 tokens across all prompts.

#### C: Remove Redundant Preamble
Phrases like "You are an expert film analyst whose job is to extract HIGH-SIGNAL representations of..." appear in slight variations across all 8 prompts. This is fine for framing but could be shortened. Similarly, the "INPUTS YOU MAY RECEIVE" blocks repeat near-identical lists — these could be shortened to just the field names without descriptions since the model understands field semantics from the name + data.

**Estimated savings:** ~400-600 tokens.

#### D: Combine CRITICAL phrasing rules
The `VIEWER_EXPERIENCE_SYSTEM_PROMPT` has 8 "CRITICAL phrasing rules" (~200 tokens) that are also conceptually present in `WATCH_CONTEXT_SYSTEM_PROMPT` and `NARRATIVE_TECHNIQUES_SYSTEM_PROMPT`. If prompts were shared (see Section 4), this would be stated once.

**Combined prompt compression estimate: ~2,500-3,500 input tokens saved (30-45% prompt reduction).**

### 2.3 Plot Synopsis Propagation

**Impact: MEDIUM (~1,800-2,700 input tokens saved, ~6-10% of total input)**

The Wave 2 calls receive the LLM-generated `plot_synopsis` (~900 tokens) from `plot_events`. This synopsis feeds into 4 calls: `plot_analysis`, `viewer_experience`, `narrative_techniques`, and `source_of_inspiration`. Three of these 4 calls don't actually need a full chronological plot breakdown:

- **`plot_analysis`:** Needs the full synopsis for thematic analysis. **Keep as-is.**
- **`narrative_techniques`:** Needs structural understanding of how the story unfolds. **Keep as-is** — the synopsis is the primary evidence for identifying techniques.
- **`viewer_experience`:** Needs emotional context, not plot specifics. The `overview` (~60 tokens) + `plot_keywords` + `genres` provide sufficient emotional framing. The synopsis adds marginal value here because viewer experience is about *feelings*, not *events*.
  - **Recommendation:** Replace the ~900 token synopsis with the ~60 token `overview`. **Save ~840 tokens.**
- **`source_of_inspiration`:** Primarily needs to know if it's "based on a true story" or "adapted from a novel." The keywords and overview provide this signal. The full synopsis adds very little.
  - **Recommendation:** Remove synopsis from this call. **Save ~900 tokens.**

**Total estimated savings: ~1,740 tokens.** For a more aggressive approach, also send a truncated synopsis (first 300 tokens, covering setup only) to `viewer_experience` instead of removing it entirely. This hedges quality while still saving ~600 tokens.

### 2.4 Remove overview from Calls That Have plot_synopsis

**Impact: LOW (~120 tokens saved)**

The `overview` is a marketing blurb (~60 tokens) that is a strict subset of information already in `plot_synopsis`. When both are present in the same call, `overview` adds zero signal.

Currently, `plot_analysis` receives both `overview` and `plot_synopsis`. The overview is redundant here.

**Recommendation:** Remove `overview` from `plot_analysis`. The synopsis already covers everything the overview says and more.

### 2.5 Raw Plot Data Optimization for plot_events

**Impact: LOW-MEDIUM (~500-1,000 input tokens saved)**

The `plot_events` call receives ALL raw plot data:
- `plot_summaries[]`: Multiple summaries averaging ~580 total tokens
- `plot_synopses[]`: Usually one long synopsis averaging ~1,875 tokens

These are the raw source materials from which the LLM creates its own `plot_summary`. Two potential optimizations:

#### A: Select the Best Single Synopsis
If multiple synopses exist, the longest/most detailed one contains all the information of shorter ones. Send only the best synopsis rather than all of them.

#### B: Truncate Extremely Long Synopses
Some synopses may exceed 3,000 tokens. Diminishing returns on plot detail set in around 1,500-2,000 tokens. Consider capping at 2,000 tokens and relying on `plot_summaries` to fill gaps.

**Estimated savings: ~500-1,000 tokens depending on input data distribution.**

### 2.6 Summary of Input Savings

| Strategy | Estimated Savings | % of Current Input | Quality Risk |
|----------|------------------|-------------------|--------------|
| Featured reviews (A+C) | ~7,000 tokens | 25% | Low |
| System prompt compression | ~3,000 tokens | 11% | None |
| Plot synopsis propagation | ~1,740 tokens | 6% | Low |
| Raw plot data optimization | ~750 tokens | 3% | None |
| Remove redundant overview | ~120 tokens | <1% | None |
| **Total** | **~12,600 tokens** | **~45%** | |

**Projected input tokens after optimization: ~15,400 (down from ~28,000)**

---

## 3. Output Token Reduction Strategies

> Output tokens are ~4x the cost of input tokens, making each saved output token worth ~4 saved input tokens.

### 3.1 Eliminate Justification Fields from Schemas

**Impact: HIGH (~600-800 content output tokens saved + significant reasoning token reduction)**

There are **27 `justification` / `explanation_and_justification` fields** across all output schemas that are never used in the final embedding text. They exist solely as chain-of-thought scaffolding, but they are expensive scaffolding: each justification averages ~20-30 output tokens, and the model also expends reasoning tokens to formulate them.

**Affected schemas:**
| Schema | Justification Fields | Est. Tokens Wasted |
|--------|---------------------|-------------------|
| `ViewerExperienceMetadata` | 8 sections × `justification` | ~200 |
| `NarrativeTechniquesMetadata` | 11 sections × `justification` | ~275 |
| `WatchContextMetadata` | 4 sections × `justification` | ~100 |
| `ProductionMetadata` | 2 × `justification` | ~50 |
| `PlotAnalysisMetadata` | `core_concept.explanation_and_justification`, up to 3 × `themes.explanation`, up to 3 × `lessons.explanation`, up to 3 × `arcs.arc_transformation_description` | ~200 |
| **Total** | **~28 fields** | **~825 tokens** |

**Recommendation:** Remove all justification/explanation fields from the Pydantic schemas.

**"But doesn't chain-of-thought improve quality?"**

In principle, yes. But gpt-5-mini already has its own internal reasoning (`reasoning_effort` parameter) that serves this purpose. The structured-output justification fields force the model to *write out* a justification in the response, which is redundant with its internal reasoning. You're paying for chain-of-thought twice: once in hidden reasoning tokens, and once in visible justification text.

If you're concerned about quality regression, a cheaper alternative is to add a single sentence to the system prompt like: "Think step-by-step before filling each section, but do not include your reasoning in the output." This encourages the model to reason internally without generating justification tokens.

**Estimated total savings: ~825 content tokens + ~500-1,000 reasoning tokens = ~1,300-1,800 output tokens.**
At ~4x the cost of input tokens, this is equivalent to saving ~5,200-7,200 input tokens in cost impact.

### 3.2 Remove Unused Schema Fields

**Impact: MEDIUM (~200-350 content output tokens saved)**

Several fields in the output schemas are generated but never included in the embedding text (confirmed by `__str__()` methods and `create_*_vector_text()` in `vectorize.py`):

| Field | Schema | Why It's Wasted | Tokens/Movie |
|-------|--------|----------------|--------------|
| `major_characters[].role` | `PlotEventsMetadata` | Explicitly excluded from vector text per schema docs | ~5 × 3-5 chars = ~15-25 |
| `character_arcs[].character_name` | `PlotAnalysisMetadata` | Only `arc_transformation_label` is used in `__str__()` | ~10 × 1-3 arcs = ~10-30 |
| `character_arcs[].arc_transformation_description` | `PlotAnalysisMetadata` | Only `arc_transformation_label` is used | ~30 × 1-3 arcs = ~30-90 |
| `core_concept.explanation_and_justification` | `PlotAnalysisMetadata` | Only `core_concept_label` is used | ~30-40 |
| `themes_primary[].explanation_and_justification` | `PlotAnalysisMetadata` | Only `theme_label` is used | ~30 × 1-3 = ~30-90 |
| `lessons_learned[].explanation_and_justification` | `PlotAnalysisMetadata` | Only `lesson_label` is used | ~30 × 0-3 = ~0-90 |

**Note:** Many of these overlap with the justification fields from Section 3.1. The combined unique savings from 3.1 + 3.2 is approximately **~1,000-1,200 content output tokens**.

**Recommendation:** Flatten nested schemas. For example, replace:
```python
class CharacterArc(BaseModel):
    character_name: str
    arc_transformation_description: str
    arc_transformation_label: str
```

With simply having `character_arcs` be a `List[str]` of arc labels. Same for `themes_primary` and `lessons_learned` — just emit `List[str]` of labels.

For `core_concept`, change from a nested `CoreConcept` object to a plain `str` field.

For `major_characters`, remove the `role` field.

### 3.3 Constrain Output Lengths in Prompts

**Impact: MEDIUM (~300-500 content tokens saved)**

Some output fields are unconstrained and produce longer text than needed for effective embedding:

#### A: `plot_summary` in PlotEventsMetadata
Currently averages ~900 tokens. This is the single largest output field and feeds the Plot Events vector directly. However, embedding models (text-embedding-3-small, 1536 dims) have diminishing returns on input length. Research suggests that text beyond ~500 tokens provides marginal additional embedding quality.

**Recommendation:** Add an explicit word limit to the prompt: "Write the plot summary in 250-350 words maximum." This would constrain output to ~350-450 tokens.
- **Estimated savings:** ~450-550 tokens
- **Quality risk:** Moderate. The full plot summary enables very specific plot-event queries. Consider testing retrieval quality with truncated summaries before committing. If specific character interactions or late-plot details get cut, queries like "movie where the friend destroys the car" might lose recall.
- **Mitigation:** If you truncate, prioritize keeping the beginning (setup) and end (resolution) of the plot, which carry the most distinctive signal.

#### B: `generalized_plot_overview` in PlotAnalysisMetadata
Currently ~150 tokens. This is already reasonably constrained ("1-3 sentences"). No change needed.

#### C: `new_reception_summary` in ReceptionMetadata
Currently ~100 tokens. Already efficient. No change needed.

**Conservative recommendation:** Don't constrain `plot_summary` length unless you validate embedding quality with shorter summaries. The risk-reward is unfavorable here — you save ~500 output tokens but risk losing recall on specific plot queries, which is a core use case.

### 3.4 Reduce Reasoning Effort Where Safe

**Impact: MEDIUM (~800-1,500 reasoning tokens saved)**

Two calls currently use `reasoning_effort="medium"`:
- `watch_context`: Generates watch scenarios and motivations
- `narrative_techniques`: Identifies storytelling devices

**Analysis:**
- `watch_context` generates simple query-like phrases ("date night movie", "lazy sunday watch"). This is more pattern-matching than deep analysis. `reasoning_effort="low"` should produce equivalent quality.
- `narrative_techniques` genuinely benefits from moderate reasoning since identifying structural devices requires multi-step inference from plot details. Keep at "medium" or test "low" carefully.

**Recommendation:** Lower `watch_context` to `reasoning_effort="low"`. Tentatively test `narrative_techniques` at "low" on a sample of 50-100 movies before committing.

**Estimated savings:** ~400-800 reasoning tokens per movie.

### 3.5 Summary of Output Savings

| Strategy | Content Tokens Saved | Reasoning Tokens Saved | Quality Risk |
|----------|---------------------|----------------------|--------------|
| Remove justifications (3.1) | ~825 | ~500-1,000 | Low |
| Remove unused fields (3.2) | ~200-350 | ~100-200 | None |
| Reduce reasoning effort (3.4) | 0 | ~400-800 | Low |
| **Total** | **~1,025-1,175** | **~1,000-2,000** | |

**Projected output tokens after optimization: ~5,000-6,000 (down from ~8,000)**

---

## 4. Structural Optimizations

### 4.1 Consolidate Production Sub-Calls

**Impact: LOW (~500-800 input tokens saved)**

Currently, `generate_production_metadata` spawns two parallel sub-calls:
1. `generate_production_keywords(title, overall_keywords)` — ~500 input tokens
2. `generate_source_of_inspiration(title, plot_synopsis, plot_keywords, overall_keywords, featured_reviews)` — ~2,000 input tokens

These two calls produce small outputs (~50 and ~60 tokens respectively) but pay full system prompt overhead twice and send `title` + `overall_keywords` twice.

**Recommendation:** Merge into a single LLM call with a combined schema (`ProductionMetadata` directly) and a single merged system prompt. This eliminates:
- One duplicate system prompt (~300-400 tokens)
- Duplicate user context fields (~100-200 tokens)

### 4.2 Consider Merging viewer_experience + watch_context

**Impact: HIGH (eliminates one entire LLM call, ~3,000-4,000 input tokens saved)**

These two calls:
- Share most of the same inputs (title, genres, plot data, keywords, reviews, reception)
- Target related but distinct aspects of the viewer's relationship to the movie
- Both output `GenericTermsSection`-style lists of short query phrases

**Current combined cost:** ~7,000 input tokens + ~2,000 output tokens across both calls.

**Merged approach:** A single call with a combined schema producing 12 sections (8 viewer experience + 4 watch context) instead of two calls producing 8 + 4 sections separately.

**Savings:**
- Eliminates duplicate system prompt overhead: ~1,400 tokens (watch_context prompt)
- Eliminates duplicate user data: ~1,500 tokens (shared input fields)
- **Total: ~2,900 input tokens saved**

**Risk:** A single larger call may produce slightly less focused outputs per section. The combined system prompt would need careful structuring to maintain distinct section identities. Test on 50 movies and compare embedding quality before committing.

**Recommendation:** This is worth testing but should be validated. It's a higher-risk optimization that saves significant tokens but could degrade the distinctiveness of the two vector spaces if the model conflates viewer feelings with watch scenarios.

### 4.3 Call Architecture Summary

**Current: 8 calls (6 user-facing + 2 production sub-calls)**
```
Wave 1: plot_events | watch_context | reception
Wave 2: plot_analysis | viewer_experience | narrative_techniques | production_keywords | source_of_inspiration
```

**Proposed: 6 calls (or 5 if viewer_experience + watch_context merge)**
```
Wave 1: plot_events | watch_context (or merged) | reception
Wave 2: plot_analysis | viewer_experience (or merged above) | narrative_techniques | production (consolidated)
```

---

## 5. Priority Matrix

### Tier 1: High Impact, Low Risk (implement first)

| # | Strategy | Input Saved | Output Saved | Risk |
|---|----------|-------------|-------------|------|
| 1 | Remove justification/explanation fields from output schemas | — | ~1,300-1,800 | Low |
| 2 | Send review summaries-only to 4 of 6 calls (keep full text for plot_analysis + narrative_techniques) | ~5,600 | — | Low |
| 3 | Reduce review count from 5 to 3 everywhere | ~3,600 | — | Low |
| 4 | Consolidate production sub-calls | ~500-800 | ~100 | None |
| 5 | Remove unused schema fields (role, character_name, descriptions) | — | ~200-350 | None |

**Combined Tier 1 savings: ~10,000 input tokens + ~1,600-2,300 output tokens**

### Tier 2: Medium Impact, Low-Medium Risk

| # | Strategy | Input Saved | Output Saved | Risk |
|---|----------|-------------|-------------|------|
| 6 | Compress system prompts (deduplicate section instructions, trim examples) | ~2,500-3,500 | — | None |
| 7 | Remove plot_synopsis from viewer_experience and source_of_inspiration inputs | ~1,740 | — | Low |
| 8 | Lower watch_context reasoning effort to "low" | — | ~400-800 | Low |

**Combined Tier 2 savings: ~4,240-5,240 input tokens + ~400-800 output tokens**

### Tier 3: High Impact, Higher Risk (test carefully)

| # | Strategy | Input Saved | Output Saved | Risk |
|---|----------|-------------|-------------|------|
| 9 | Merge viewer_experience + watch_context into one call | ~2,900 | ~200-400 | Medium |
| 10 | Constrain plot_summary length to ~350 words | — | ~450-550 | Medium |
| 11 | Lower narrative_techniques reasoning effort to "low" | — | ~400 | Medium |

### Projected Cost After All Tiers

| Metric | Current | After Tier 1 | After Tier 1+2 | After All Tiers |
|--------|---------|-------------|----------------|-----------------|
| Input tokens | 28,000 | ~18,000 | ~13,500 | ~10,600 |
| Output tokens | 8,000 | ~5,900 | ~5,300 | ~4,200 |
| Est. cost/movie | $0.025 | ~$0.017 | ~$0.014 | ~$0.011 |
| **Reduction** | — | **~32%** | **~44%** | **~56%** |
| **Cost at 100K movies** | **$2,500** | **$1,700** | **$1,400** | **$1,100** |

---

## Appendix: What NOT to Cut

The following inputs/outputs were considered but should be preserved:

1. **Plot synopses for plot_events call:** This is the raw source material that feeds the entire pipeline. Cutting it would cascade quality loss everywhere.

2. **Plot synopsis for plot_analysis and narrative_techniques:** These calls need full plot context to identify themes, arcs, and structural devices. The synopsis is the primary evidence source.

3. **Multiple plot_summaries for plot_events:** Different summaries often capture different aspects of the plot. The model synthesizes these into a better combined summary than any single source provides.

4. **Negation terms in viewer_experience:** These are intentional and valuable — users frequently search with negations ("not too scary", "no jump scares") and these terms enable that retrieval pattern. They are a core design strength.

5. **`new_reception_summary` in reception output:** This is a relatively efficient ~100 token output that powers the reception vector with dense semantic content. Good value per token.

6. **Separate vector spaces:** While consolidating calls saves tokens, the 8-vector architecture exists for a reason — each vector space targets a distinct retrieval lens. Don't merge the *embedding vectors* themselves, only consider merging the *generation calls* where inputs substantially overlap.
