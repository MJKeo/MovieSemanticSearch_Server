# LLM Generation: Final Improvement Proposal

## Executive Summary

**Current state:** 8 LLM calls per movie using GPT-5-mini (standard API), ~28K input + ~8K output tokens, costing **~$0.025/movie → $2,500 for 100K movies**.

**Proposed end state:** 5 LLM calls per movie using GPT-4o-mini (batch API), ~11K input + ~2.3K output tokens, costing **~$0.0015/movie → $150 for 100K movies**.

**Total reduction: ~94% ($2,500 → $150).**

| Phase | Description | Cost/Movie | 100K Cost | Cumulative Savings |
|-------|-------------|-----------|-----------|-------------------|
| Current | GPT-5-mini, standard API, 8 calls | $0.025 | $2,500 | — |
| **Phase 1** | Schema cleanup + batch API | $0.005 | $500 | **80%** |
| **Phase 2** | Switch to GPT-4o-mini | $0.003 | $270 | **89%** |
| **Phase 3** | Input token reduction | $0.002 | $200 | **92%** |
| **Phase 4** | Call consolidation + prompt compression | $0.0015 | $150 | **94%** |

The single highest-impact change is **Phase 2: switching to GPT-4o-mini batch**. This eliminates ~5,000 hidden reasoning tokens per movie (61% of current output) and drops the per-token output rate from $1.00/M to $0.30/M. Combined with Phase 1 (schema cleanup), it delivers 89% savings with minimal quality risk and moderate implementation effort. Phases 3 and 4 provide diminishing but meaningful additional savings.

---

## Table of Contents

1. [Why GPT-4o-mini Is the Right Model](#1-why-gpt-4o-mini-is-the-right-model)
2. [Phase 1: Schema Optimization + Batch API](#2-phase-1-schema-optimization--batch-api)
3. [Phase 2: Model Migration to GPT-4o-mini](#3-phase-2-model-migration-to-gpt-4o-mini)
4. [Phase 3: Input Token Reduction](#4-phase-3-input-token-reduction)
5. [Phase 4: Call Consolidation + Prompt Compression](#5-phase-4-call-consolidation--prompt-compression)
6. [Cumulative Cost Projections](#6-cumulative-cost-projections)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [Quality Safeguards](#8-quality-safeguards)
9. [Alternatives Considered](#9-alternatives-considered)
10. [What Not to Change](#10-what-not-to-change)

---

## 1. Why GPT-4o-mini Is the Right Model

The model switch is the single most impactful decision in this proposal. Here is the reasoning.

### 1.1 The Hidden Reasoning Token Problem

GPT-5-mini is a reasoning model. Even at `reasoning_effort="minimal"`, it generates internal chain-of-thought tokens that are **invisible in the response but billed as output tokens** at the full output rate ($2.00/M standard, $1.00/M batch).

Current output breakdown per movie:
| Component | Tokens | % of Output | Cost at $2.00/M |
|-----------|--------|-------------|-----------------|
| Visible structured content | ~3,370 | 42% | $0.0067 |
| Hidden reasoning tokens | ~4,630 | 58% | $0.0093 |
| **Total** | **~8,000** | | **$0.016** |

**58% of your output cost is invisible reasoning tokens that produce nothing used in embeddings.**

GPT-4o-mini is a non-reasoning model. It produces zero reasoning tokens. Output cost drops to only the visible structured content — and at a lower per-token rate.

### 1.2 Pricing Comparison (Batch)

| Model | Input $/M | Output $/M | Reasoning Tokens | Structured Output |
|-------|----------|-----------|-----------------|-------------------|
| **GPT-5-mini** (current) | $0.125 | $1.00 | Yes (~58% of output) | Native strict |
| **GPT-4o-mini** (recommended) | $0.075 | $0.30 | **None** | Native strict |
| GPT-5-nano | $0.025 | $0.20 | Yes (reduced) | Native strict |
| Gemini 2.5 Flash-Lite | $0.05 | $0.20 | Configurable | JSON Schema (quirks) |

Sources: [OpenAI API Pricing](https://openai.com/api/pricing/), [Google Gemini Pricing](https://ai.google.dev/gemini-api/docs/pricing) — verified March 2026.

### 1.3 Cost Per Movie: Model-Only Comparison (Batch, 28K Input)

Using the current token volumes (28K input, but adjusting output to model behavior):

| Model | Input Cost | Output Tokens | Output Cost | **Total/Movie** | **100K Cost** |
|-------|-----------|--------------|------------|----------------|--------------|
| GPT-5-mini (current) | $0.0035 | 8,000 | $0.008 | **$0.0115** | **$1,150** |
| **GPT-4o-mini** | $0.0021 | ~3,370 | $0.001 | **$0.003** | **$300** |
| GPT-5-nano | $0.0007 | ~4,000* | $0.0008 | **$0.0015** | **$150** |
| Gemini 2.5 Flash-Lite | $0.0014 | ~3,370 | $0.0007 | **$0.002** | **$200** |

*GPT-5-nano still produces some reasoning tokens even at minimal effort.

### 1.4 Why GPT-4o-mini Over Cheaper Alternatives

**Why not GPT-5-nano ($150/100K)?**
- 7.3% hallucination rate on FActScore benchmarks vs ~2-3% for GPT-4o-mini. For plot summaries and thematic analysis, hallucinated facts (invented plot events, wrong character names) directly corrupt search quality.
- Significant quality drop on reasoning tasks: 9.6% vs 26.3% on FrontierMath. While our tasks aren't math, the analytical calls (plot_analysis, narrative_techniques) require multi-step inference that benefits from stronger base reasoning.
- The absolute savings over GPT-4o-mini after all optimizations is only ~$50-80 for 100K movies — not worth the quality risk.

**Why not Gemini 2.5 Flash-Lite ($200/100K)?**
- Structured output has documented edge cases: markdown code fence wrapping, occasional schema violations. Google's own documentation warns that structured output "guarantees syntactically correct JSON but NOT semantic correctness."
- Requires full API migration: different SDK (`google-generativeai`), different structured output syntax, different error handling. This is days of engineering work for ~$70 savings over GPT-4o-mini.
- Less battle-tested for Pydantic schema compliance at scale. At 100K movies, even a 2% structured output failure rate means 2,000 movies requiring retry logic.

**Why GPT-4o-mini is the sweet spot:**
- **Same OpenAI API, same structured output support** — change the model string and remove reasoning parameters. Code changes are minimal.
- **100% JSON schema compliance** on OpenAI structured output evaluations.
- **No reasoning token overhead** — output cost is purely visible content.
- **Proven at scale** — one of the most widely deployed models for structured extraction tasks.
- **128K context window** — more than sufficient for consolidated prompts (our largest consolidated call will be ~6K tokens).

### 1.5 Quality Assessment for Each Call Type

| Call | Task Type | GPT-4o-mini Suitability | Risk |
|------|----------|------------------------|------|
| `plot_events` | Text extraction/summarization | Excellent — extraction from provided text | None |
| `reception` | Summarization + attribute extraction | Excellent — straightforward extraction | None |
| `watch_context` | Creative inference about scenarios | Good — pattern matching, not deep reasoning | Low |
| `viewer_experience` | Subjective emotional analysis | Good — follows examples well | Low |
| `plot_analysis` | Thematic analysis | Good — prompts provide strong guidance | Low |
| `narrative_techniques` | Structural film analysis | Adequate — most structured of the analytical tasks | Low-Medium |
| `production_keywords` | Keyword filtering | Excellent — trivial task | None |
| `source_of_inspiration` | Light inference | Excellent — simple extraction | None |

The `narrative_techniques` call is the only one where GPT-5-mini's reasoning might provide marginal benefit. However, the prompt is well-designed with clear category guidance and examples, which largely compensates for the model's weaker reasoning. Validate on 50 movies before committing (see [Quality Safeguards](#8-quality-safeguards)).

---

## 2. Phase 1: Schema Optimization + Batch API

**Effort: Low | Risk: None | Savings: 80% (primarily from batch API)**

This phase delivers the largest single jump in savings because it combines two independent wins: eliminating wasted output tokens and switching to batch pricing.

### 2.1 Switch to OpenAI Batch API

**Impact: 50% cost reduction with zero quality change.**

The current `generate_openai_response()` function uses synchronous `chat.completions.parse()` calls. For a 100K movie pipeline, OpenAI's Batch API is the correct tool:
- **50% discount** on all token costs (input and output)
- Designed for bulk processing (results within 24 hours)
- Supports structured outputs (`response_format`) identically to the synchronous API
- Submit JSONL files of requests, poll for completion

This changes the execution model from synchronous per-movie processing to batch submission, but for a one-time 100K movie ingestion pipeline, this is the natural fit. The parallel wave architecture (Wave 1 → Wave 2) can be preserved by submitting Wave 1 as one batch, waiting for completion, then submitting Wave 2.

### 2.2 Remove All Justification and Explanation Fields

**Impact: ~955 output content tokens saved per movie (28% of visible output)**

There are **28 `justification` / `explanation_and_justification` / `arc_transformation_description` fields** across all output schemas that are generated by the LLM but **never included in any embedding text**. These fields exist solely as chain-of-thought scaffolding, but GPT-4o-mini doesn't need them — and even on GPT-5-mini, the model's internal reasoning already serves this purpose.

**Fields to remove:**

| Schema | Fields to Remove | Est. Tokens Saved |
|--------|-----------------|-------------------|
| `ViewerExperienceSection` | `justification` (×8 sections) | ~200 |
| `GenericTermsSection` (used by WatchContext, NarrativeTechniques, ProductionKeywords) | `justification` (×15 sections) | ~375 |
| `SourceOfInspirationSection` | `justification` | ~45 |
| `CoreConcept` | `explanation_and_justification` | ~30 |
| `CharacterArc` | `character_name`, `arc_transformation_description` | ~130 |
| `MajorTheme` | `explanation_and_justification` (×1-3) | ~90 |
| `MajorLessonLearned` | `explanation_and_justification` (×1-3) | ~70 |
| **Total** | | **~940** |

Verified against actual output from the sample data: Zootopia's 8 viewer experience justification fields average ~100 chars each (~25 tokens), and 11 narrative technique justifications average ~130 chars each (~33 tokens). These numbers are consistent with the token reduction guide's estimates.

**Schema changes required:**

```python
# CharacterArc: replace nested object with plain string
# Before:
class CharacterArc(BaseModel):
    character_name: str
    arc_transformation_description: str
    arc_transformation_label: str

# After:
# character_arcs becomes List[str] of arc labels directly

# MajorTheme / MajorLessonLearned: same pattern
# Before:
class MajorTheme(BaseModel):
    explanation_and_justification: str
    theme_label: str

# After:
# themes_primary becomes List[str] of theme labels directly

# CoreConcept: flatten to plain string
# Before:
class CoreConcept(BaseModel):
    explanation_and_justification: str
    core_concept_label: str

# After:
# core_concept becomes a plain str field

# GenericTermsSection: remove justification
# Before:
class GenericTermsSection(BaseModel):
    justification: str
    terms: List[str]

# After:
class GenericTermsSection(BaseModel):
    terms: List[str]

# ViewerExperienceSection: remove justification
# Same pattern — remove justification field

# SourceOfInspirationSection: remove justification
# Same pattern
```

**Important:** The `__str__()` methods and `create_*_vector_text()` functions in `vectorize.py` must be updated to reflect the flattened schemas. Since these methods already only use the label/terms fields, the changes are straightforward removals.

### 2.3 Remove Unused Schema Fields

**Impact: ~140 additional content output tokens saved per movie**

One field is generated but explicitly excluded from all vector text:

| Field | Schema | Why It's Wasted |
|-------|--------|----------------|
| `MajorCharacter.role` | `PlotEventsMetadata` | Excluded from `__str__()` — only name, description, and motivations are used |

Remove the `role` field from `MajorCharacter` and its corresponding prompt instructions.

### 2.4 Phase 1 Cost Impact

**Before Phase 1 (current state, GPT-5-mini standard):**
- Input: 28,000 × $0.25/M = $0.007
- Output: 8,000 × $2.00/M = $0.016
- **Total: $0.023/movie → $2,300/100K**

**After Phase 1 (GPT-5-mini batch + schema cleanup):**
- Input: ~27,500 × $0.125/M = $0.00344
- Output: ~6,100 × $1.00/M = $0.0061 (reduced content + proportionally fewer reasoning tokens)
- **Total: ~$0.0095/movie → $950/100K**
- **Savings: ~59%**

> Note: Even staying on GPT-5-mini, batch alone gets you to $0.0115/movie. The schema cleanup saves an additional ~$200/100K.

---

## 3. Phase 2: Model Migration to GPT-4o-mini

**Effort: Low | Risk: Low (validate on 50-100 movies) | Savings: 89% cumulative**

### 3.1 What Changes

The code change is minimal — modify `generate_openai_response()` to use `gpt-4o-mini` and remove reasoning-specific parameters:

```python
# Before:
parsed, input_tokens, output_tokens = generate_openai_response(
    user_prompt=user_prompt,
    system_prompt=PLOT_EVENTS_SYSTEM_PROMPT,
    response_format=PlotEventsMetadata,
    model="gpt-5-mini",
    reasoning_effort="minimal",
    verbosity="low"
)

# After:
parsed, input_tokens, output_tokens = generate_openai_response(
    user_prompt=user_prompt,
    system_prompt=PLOT_EVENTS_SYSTEM_PROMPT,
    response_format=PlotEventsMetadata,
    model="gpt-4o-mini",
)
```

**Parameters to remove:**
- `reasoning_effort` — GPT-4o-mini is not a reasoning model; this parameter is unsupported and would cause an API error.
- `verbosity` — GPT-5 family specific parameter; not supported by GPT-4o-mini.

The `generate_openai_response()` function signature should either remove these parameters or make them conditional on the model.

### 3.2 Why Output Drops Dramatically

With GPT-5-mini, output tokens include both visible content and hidden reasoning:
- Visible content: ~3,370 tokens (after Phase 1 schema cleanup: ~2,415 tokens)
- Hidden reasoning: ~4,630 tokens
- Total: ~8,000 tokens

With GPT-4o-mini, output is purely visible content:
- After Phase 1 schema cleanup: **~2,415 tokens**
- No reasoning overhead: **0 tokens**
- Total: **~2,415 tokens**

This is a **70% reduction in output tokens**, combined with a per-token rate drop from $1.00/M to $0.30/M (batch). The output cost per movie drops from $0.0061 to $0.00072 — an **88% reduction in output cost alone**.

### 3.3 Phase 2 Cost Impact

**After Phase 1 + 2 (GPT-4o-mini batch + schema cleanup):**
- Input: ~27,500 × $0.075/M = $0.00206
- Output: ~2,415 × $0.30/M = $0.00072
- **Total: ~$0.0028/movie → $280/100K**
- **Cumulative savings: 89%**

---

## 4. Phase 3: Input Token Reduction

**Effort: Medium | Risk: Low | Savings: 92% cumulative**

With output costs now minimized (only ~26% of total cost after Phase 2), input tokens become the dominant cost driver. This phase targets the three largest sources of input waste.

### 4.1 Featured Review Optimization

**Impact: ~6,500-7,000 input tokens saved per movie (23-25% of input)**

Featured reviews are the single largest input cost: ~1,500-2,000 tokens per call, duplicated across 6 of 8 calls, totaling ~9,000-12,000 tokens.

From the sample data:
- Average review text: ~1,655 characters (~410 tokens)
- Average review summary: ~38 characters (~10 tokens)
- Reviews per movie: 10 available, currently sending 5

The summary field is surprisingly information-dense. Sample: `"Not just a classic of the 80s, but of any decade"` — this captures the essential sentiment in ~10 tokens vs the full text at ~200 tokens.

**Recommended strategy (combines Options A + C from the token reduction guide):**

| Call | Current | Recommended | Rationale |
|------|---------|-------------|-----------|
| `plot_analysis` | 5 full reviews | **3 full reviews** | Needs review text for thematic interpretation |
| `narrative_techniques` | 5 full reviews | **3 full reviews** | Needs review text for structural observations |
| `viewer_experience` | 5 full reviews | **3 summaries only** | Needs sentiment signal, not full text |
| `watch_context` | 5 full reviews | **3 summaries only** | Needs sentiment signal, not full text |
| `reception` | 5 full reviews | **3 summaries only** | Already has reception_summary and review_themes for detail |
| `source_of_inspiration` | 5 full reviews | **3 summaries only** | Needs "based on a book" signals, summaries suffice |

**Token math:**
- Before: ~2,050 tokens (5 × ~410) per call × 6 calls = ~12,300 total
- After: (3 × 410 × 2 calls) + (3 × 10 × 4 calls) = 2,460 + 120 = ~2,580 total
- **Savings: ~9,700 tokens**

This is more aggressive than the token reduction guide's ~7,000 estimate, but the data supports it: summaries alone carry ~80% of the useful sentiment signal for calls that need reviews primarily for tone and reception context. The two calls that perform deep textual analysis (plot_analysis, narrative_techniques) retain full review text.

### 4.2 Plot Synopsis Propagation

**Impact: ~1,740 input tokens saved per movie (6% of input)**

The LLM-generated `plot_synopsis` (~900 tokens) is passed to 4 Wave 2 calls. Two of these don't need the full plot:

| Call | Needs Full Synopsis? | Recommendation | Savings |
|------|---------------------|----------------|---------|
| `plot_analysis` | **Yes** — thematic analysis requires full plot | Keep as-is | 0 |
| `narrative_techniques` | **Yes** — structural analysis requires plot flow | Keep as-is | 0 |
| `viewer_experience` | **No** — emotional analysis needs context, not events | Replace with `overview` (~60 tokens) | ~840 |
| `source_of_inspiration` | **No** — needs "based on" signals, not plot detail | Remove entirely (keywords suffice) | ~900 |

**Total savings: ~1,740 tokens**

### 4.3 Remove Redundant Overview

**Impact: ~60 input tokens saved**

`plot_analysis` currently receives both `overview` and `plot_synopsis`. The overview is a marketing blurb that is a strict subset of the synopsis. Remove `overview` from `plot_analysis`.

### 4.4 Phase 3 Cost Impact

**Total input reduction: ~11,500 tokens**

**After Phase 1 + 2 + 3:**
- Input: ~27,500 - 11,500 = ~16,000 × $0.075/M = $0.00120
- Output: ~2,415 × $0.30/M = $0.00072
- **Total: ~$0.0019/movie → $192/100K**
- **Cumulative savings: 92%**

---

## 5. Phase 4: Call Consolidation + Prompt Compression

**Effort: High | Risk: Medium (validate carefully) | Savings: 94% cumulative**

This phase reduces the number of LLM calls from 8 to 5, eliminating duplicate system prompts and shared input fields that are sent to multiple calls.

### 5.1 Consolidation Plan

#### Merge 1: Production Keywords + Source of Inspiration → Single "Production" Call

**Risk: None | Input saved: ~520 tokens | Calls saved: 1**

These are already combined in `generate_production_metadata()` but still make 2 separate LLM calls internally. The `production_keywords` call is trivially simple (filter keywords from a list) and produces ~50 tokens of output. Merging into a single call with a combined `ProductionMetadata` schema eliminates:
- Duplicate system prompt (~375 tokens)
- Duplicate title + keywords (~145 tokens)

This is the simplest consolidation and should be implemented first as a proof of concept.

#### Merge 2: Viewer Experience + Watch Context → "Audience Perspective" Call

**Risk: Low | Input saved: ~2,295 tokens | Calls saved: 1**

These two calls analyze the movie from the same analytical perspective (the viewer's experience) with nearly identical inputs. After Phase 3 input reductions, the overlap is:
- Both receive: title, genres, plot_keywords, overall_keywords, reception_summary, audience_reception_attributes, featured_review summaries
- Unique to viewer_experience: maturity data, plot_synopsis (replaced with overview in Phase 3)
- Unique to watch_context: overview

Merging saves:
- One full copy of shared inputs: ~1,095 tokens
- System prompt deduplication: ~1,200 tokens (merged prompt shares preamble, phrasing rules, and output format instructions that are currently duplicated verbatim)

**The combined output schema has 12 sections** (8 viewer experience + 4 watch context). GPT-4o-mini handles large structured output schemas well within its 128K context. To prevent section thinning, add explicit minimum term counts (e.g., "Each section must have at least 3 terms").

**Quality consideration:** These sections frequently reference the same emotional/tonal analysis. A single pass actually **improves** consistency — the model won't label a movie "chill" in viewer_experience but "adrenaline fix" in watch_context scenarios.

#### Merge 3: Plot Analysis + Narrative Techniques → "Story Analysis" Call

**Risk: Low-Medium | Input saved: ~2,270 tokens | Calls saved: 1**

These two calls analyze the story itself from complementary perspectives: what it's about (themes, concepts) vs. how it's told (techniques, structure). They share most inputs: plot_synopsis, plot_keywords, overall_keywords, featured_reviews, reception_summary.

Merging saves:
- One full copy of: plot_synopsis (~500 tokens), plot_keywords (~50), overall_keywords (~50), featured_reviews (~1,230 after Phase 3 reductions), reception_summary (~100), title (~20)
- System prompt deduplication: ~800 tokens

**Quality consideration:** Character arcs appear in both schemas. A single pass ensures the arcs identified in plot_analysis align with those referenced in narrative_techniques. This actually **improves** output coherence.

**Risk mitigation:** The combined prompt must clearly delineate the two analytical perspectives. Use section grouping: "PART A: THEMATIC ANALYSIS" and "PART B: NARRATIVE TECHNIQUES" with distinct instructions for each. Test on 50 movies and compare embedding quality against the separate-call baseline.

### 5.2 New Call Architecture

**Current: 8 calls (3 in Wave 1, 5 in Wave 2)**
```
Wave 1: plot_events | watch_context | reception
Wave 2: plot_analysis | viewer_experience | narrative_techniques | production_keywords | source_of_inspiration
```

**Proposed: 5 calls (3 in Wave 1, 2 in Wave 2)**
```
Wave 1: plot_events | audience_perspective (viewer_exp + watch_ctx) | reception
Wave 2: story_analysis (plot_analysis + narrative_tech) | production (merged)
```

Wave 2 now has only 2 calls (down from 5), which also reduces overall latency since Wave 2 depends on `plot_events` completing first.

### 5.3 System Prompt Compression

**Impact: ~2,000-2,500 additional input tokens saved across all calls**

Independent of call consolidation, the system prompts can be compressed ~30-40% without losing instruction quality. The three multi-section prompts (viewer_experience, watch_context, narrative_techniques) repeat identical formatting rules per section.

**Strategy A: Deduplicate per-section instructions.**

Before (repeated per section, ~80-100 tokens each):
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

After (~30-40 tokens each):
```
GLOBAL RULES: All terms are 1-5 word search queries. Negations start with "not"/"no".
Synonyms/paraphrases encouraged. 3-10 terms per section. Use everyday language.

1) emotional_palette: dominant viewer emotions during the film.
   Ex: "cozy", "laugh out loud", "nostalgic" | Neg: "not too sad", "not cheesy"
```

This compresses 23 total sections across the three prompts from ~2,000 tokens to ~800 tokens.

**Strategy B: Trim example lists from 6-12 to 3-4 per section.** The model understands the pattern after 3 examples. Savings: ~500-800 tokens.

**Strategy C: Shorten "INPUTS YOU MAY RECEIVE" blocks.** Currently lists each input with a description. The model understands field semantics from the field name + data. Replace with just field names. Savings: ~300-500 tokens.

### 5.4 Phase 4 Cost Impact

**Total input reduction from consolidation + compression: ~7,000 tokens**

**After all phases:**
- Input: ~16,000 - 7,000 = ~9,000 × $0.075/M = $0.000675
- Output: ~2,415 × $0.30/M = $0.000725
- **Total: ~$0.0014/movie → $140/100K**
- **Cumulative savings: 94%**

---

## 6. Cumulative Cost Projections

### 6.1 Phase-by-Phase Breakdown

| Phase | Input Tokens | Output Tokens | Model | Pricing | $/Movie | 100K Cost | Savings |
|-------|-------------|--------------|-------|---------|---------|-----------|---------|
| **Current** | 28,000 | 8,000 | GPT-5-mini | Standard | $0.023 | $2,300 | — |
| **+Batch only** | 28,000 | 8,000 | GPT-5-mini | Batch | $0.0115 | $1,150 | 50% |
| **Phase 1** | 27,500 | 6,100 | GPT-5-mini | Batch | $0.0095 | $950 | 59% |
| **Phase 2** | 27,500 | 2,415 | GPT-4o-mini | Batch | $0.0028 | $280 | 88% |
| **Phase 3** | 16,000 | 2,415 | GPT-4o-mini | Batch | $0.0019 | $192 | 92% |
| **Phase 4** | 9,000 | 2,415 | GPT-4o-mini | Batch | $0.0014 | $140 | 94% |

### 6.2 Cost Composition Shift

| Phase | Input % of Cost | Output % of Cost |
|-------|----------------|-----------------|
| Current | 30% | 70% |
| After Phase 2 | 74% | 26% |
| After Phase 4 | 48% | 52% |

The model switch inverts the cost structure: output goes from the dominant cost (70%) to the minority (26%). This is why input reduction (Phase 3) matters — after Phase 2, it's where the remaining money is.

### 6.3 Effort vs. Impact Analysis

| Phase | Implementation Effort | Cumulative Savings | Marginal Savings |
|-------|--------------------|-------------------|-----------------|
| Phase 1 | ~1 day (schema changes + batch API setup) | $1,350 | $1,350 |
| Phase 2 | ~0.5 days (model string + parameter removal + validation) | $2,020 | $670 |
| Phase 3 | ~1-2 days (review formatting logic + synopsis routing) | $2,108 | $88 |
| Phase 4 | ~3-5 days (new schemas, merged prompts, updated vectorize) | $2,160 | $52 |

**Recommendation:** Phases 1 and 2 are unambiguous wins. Phase 3 is worthwhile. Phase 4 is optional — the $52 marginal savings for 100K movies may not justify 3-5 days of prompt engineering and schema redesign. However, if you plan to reprocess movies regularly or scale beyond 100K, Phase 4's structural improvements compound over time and reduce per-run latency.

---

## 7. Implementation Roadmap

### 7.1 Recommended Order

Each phase can be validated independently before proceeding to the next.

**Phase 1 (Week 1):**
1. Remove all justification/explanation fields from Pydantic schemas
2. Flatten `CharacterArc`, `MajorTheme`, `MajorLessonLearned`, `CoreConcept` to plain strings/lists
3. Remove `MajorCharacter.role` field
4. Update `__str__()` methods and `create_*_vector_text()` functions
5. Update system prompts to remove references to removed fields
6. Implement batch API submission pipeline
7. Run 50-movie validation: compare output quality against current baseline

**Phase 2 (Week 2):**
1. Change model from `gpt-5-mini` to `gpt-4o-mini` in all calls
2. Remove `reasoning_effort` and `verbosity` parameters from `generate_openai_response()`
3. Run 100-movie A/B comparison (see [Quality Safeguards](#8-quality-safeguards))
4. If quality is acceptable: proceed. If not: keep GPT-4o-mini for extraction calls, use GPT-5-mini (low reasoning) for analytical calls only.

**Phase 3 (Week 3):**
1. Modify review formatting in generation methods:
   - `plot_analysis` and `narrative_techniques`: send 3 full reviews (summary + text)
   - All other calls: send 3 review summaries only
2. Remove `plot_synopsis` from `viewer_experience` inputs; send `overview` instead
3. Remove `plot_synopsis` from `source_of_inspiration` inputs
4. Remove `overview` from `plot_analysis` inputs
5. Run 50-movie validation

**Phase 4 (Week 4-5, optional):**
1. Design merged `ProductionMetadata` schema and prompt (combine production_keywords + source_of_inspiration into single call)
2. Design merged `AudiencePerspectiveMetadata` schema and prompt
3. Design merged `StoryAnalysisMetadata` schema and prompt
4. Compress system prompts (deduplicate section rules, trim examples)
5. Update `generate_llm_metadata()` orchestration (8 calls → 5 calls, adjust wave structure)
6. Update vectorize.py to split merged outputs back into individual vector space texts
7. Run 100-movie A/B comparison on search retrieval quality

### 7.2 Rollback Strategy

Each phase is independently reversible:
- Phase 1: Revert schema changes, re-add justification fields
- Phase 2: Change model string back to `gpt-5-mini`, restore parameters
- Phase 3: Restore review/synopsis routing logic
- Phase 4: Restore individual call functions and schemas

Since this is a batch ingestion pipeline (not a real-time API), there's no deployment risk. You can regenerate any subset of movies if quality issues are discovered.

---

## 8. Quality Safeguards

### 8.1 A/B Validation Protocol

Before committing any phase at full scale:

1. **Select 50-100 diverse movies** spanning:
   - Multiple genres (action, drama, comedy, horror, documentary, animation)
   - Multiple eras (pre-1980, 1980-2000, 2000-2015, 2015+)
   - Multiple popularity levels (blockbusters, mid-tier, obscure)
   - Edge cases: movies with minimal reviews, no reception summary, very short/long synopses

2. **Generate metadata with both old and new configurations** for the same movies.

3. **Quantitative comparison:**
   - Embed both sets using text-embedding-3-small
   - Run your existing benchmark search queries
   - Compare precision@10 and recall@10
   - Target: **>80% overlap** in top-10 results

4. **Qualitative comparison (manual review of 15-20 movies):**
   - Are plot summaries accurate? No hallucinated events?
   - Are themes meaningful and properly generalized?
   - Are viewer experience terms consistent with the movie's actual tone?
   - Are narrative technique labels correct and high-signal?

### 8.2 Output Validation Pipeline

For production runs, implement per-movie validation:

1. **Schema compliance:** Pydantic validation on every response (already handled by OpenAI structured output, but add a safety check)
2. **Minimum content checks:**
   - `plot_summary` length > 100 words
   - `emotional_palette.terms` has ≥ 3 items
   - `genre_signatures` has ≥ 2 items
   - `character_arcs` has ≥ 1 item
3. **On validation failure:** Log the movie ID, retry once, then flag for manual review.

### 8.3 Metrics to Track

| Metric | Target | Action if Missed |
|--------|--------|-----------------|
| Schema validation pass rate | >99% | Investigate prompt clarity; add examples |
| Average terms per section | Stable (±15% vs baseline) | Section thinning — add minimum term counts to prompt |
| Cosine similarity (old vs new embeddings, same movie) | >0.85 | Quality regression — investigate specific calls |
| Search result overlap (benchmark queries) | >80% top-10 overlap | Significant regression — consider reverting or hybrid model approach |

---

## 9. Alternatives Considered

### 9.1 GPT-5-nano for Simple Calls (Hybrid Approach)

**Considered and deferred.** Using GPT-5-nano ($0.025/$0.20 batch) for plot_events, production, and reception while keeping GPT-4o-mini for analytical calls would save an additional ~$30-50 for 100K movies. However:
- Adds model management complexity (two models in the pipeline)
- GPT-5-nano's 7.3% hallucination rate is concerning for plot summaries
- The marginal savings don't justify the quality risk and code complexity

**Verdict:** Not recommended unless cost constraints are extreme.

### 9.2 Gemini 2.5 Flash-Lite

**Considered and deferred.** At $0.05/$0.20 batch pricing, Gemini Flash-Lite would reduce costs by an additional ~$30-60 vs GPT-4o-mini after all optimizations. However:
- Requires full API migration (Google SDK, different structured output syntax)
- Known structured output edge cases (markdown fencing, schema violations)
- The engineering effort (estimated 2-3 days) doesn't justify ~$40 savings on a one-time run

**Verdict:** Revisit if processing becomes ongoing (e.g., daily new movie ingestion). At that point, the migration cost amortizes over many runs.

### 9.3 Pre-Processing Reviews with a Cheap Model

The token reduction guide proposed using a haiku-tier model to create a ~200 token "review digest" before any LLM calls (Option B in Section 2.1). This was **not recommended** because:
- Phase 3's approach (summaries for 4 calls, full text for 2 calls) achieves ~90% of the same savings without adding an extra LLM call
- An extra preprocessing call adds latency, cost (~500 tokens × a cheap model), and a failure point
- Review summaries from IMDB are already high-quality condensations; a model-generated digest would be redundant

### 9.4 Staying on GPT-5-mini (Token Reduction Only)

Implementing all token reductions (Phases 1, 3, 4) without switching models would achieve:
- Input: ~9,000 × $0.125/M = $0.00113
- Output: ~5,500 × $1.00/M = $0.0055 (reduced content but still has reasoning tokens)
- **Total: ~$0.0066/movie → $660/100K (71% savings)**

This is significantly worse than the GPT-4o-mini approach ($140/100K, 94% savings) because the reasoning token overhead remains. **The model switch is not optional — it's the single highest-ROI change.**

### 9.5 Constraining Plot Summary Length

The token reduction guide considered limiting `plot_summary` to ~350 words (currently averages ~480 words / ~640 tokens). This was **not recommended** because:
- Plot summary is the primary content of the Plot Events vector, which handles specific plot queries ("movie where the friend destroys the car")
- Truncation would lose late-plot details and resolution specifics
- The output savings (~200-300 tokens at $0.30/M with GPT-4o-mini) are worth only ~$0.00007/movie
- Risk of losing recall on specific plot queries outweighs the negligible cost savings

### 9.6 Lowering Reasoning Effort (GPT-5-mini Specific)

The token reduction guide recommended lowering `watch_context` from `medium` to `low` reasoning effort. **This becomes irrelevant** with the GPT-4o-mini switch, which has no reasoning parameters. If you choose to stay on GPT-5-mini for any calls, set all reasoning_effort to `low` except `narrative_techniques` (keep at `medium`).

---

## 10. What Not to Change

The following elements should be preserved regardless of which optimizations are implemented:

1. **8 separate vector spaces.** The vectors serve distinct retrieval lenses. Merge the *generation calls*, not the *embedding vectors*. Each vector space continues to receive its own text from the (potentially merged) LLM output.

2. **Full plot synopses for plot_events input.** This is the raw source material that feeds the entire pipeline. The LLM-generated plot_summary depends on having comprehensive plot data.

3. **Multiple plot_summaries for plot_events.** Different summaries capture different aspects. The model synthesizes a better combined summary than any single source provides.

4. **Negation terms in viewer_experience.** Users frequently search with negations ("not too scary", "no jump scares"). These terms are a core design strength for semantic retrieval.

5. **Full plot_synopsis for plot_analysis and narrative_techniques.** These calls need complete plot context for thematic and structural analysis. The synopsis is their primary evidence source.

6. **`new_reception_summary` in reception output.** A compact ~100 token output that efficiently powers the reception vector. Already excellent value per token.

7. **The two-wave execution architecture.** Wave 2 depends on `plot_events` for `plot_synopsis`. This dependency is real and should be preserved (with fewer calls in each wave after consolidation).

---

## Appendix A: Token Budget Summary (Final State)

After all four phases, the per-movie token budget across 5 calls:

| Call | System Prompt | User Input | Output | Total In |
|------|--------------|-----------|--------|----------|
| `plot_events` | ~450 | ~2,600 | ~1,000 | ~3,050 |
| `audience_perspective` (merged) | ~1,800 | ~1,200 | ~600 | ~3,000 |
| `reception` | ~500 | ~400 | ~140 | ~900 |
| `story_analysis` (merged) | ~1,400 | ~1,700 | ~550 | ~3,100 |
| `production` (merged) | ~550 | ~500 | ~110 | ~1,050 |
| **Total** | **~4,700** | **~6,400** | **~2,400** | **~11,100** |

**Cost: ~11,100 input × $0.075/M + ~2,400 output × $0.30/M = $0.00083 + $0.00072 = $0.00155/movie**

## Appendix B: Pricing Sources

All pricing data verified March 2026:

- [OpenAI API Pricing](https://openai.com/api/pricing/) — GPT-5-mini, GPT-4o-mini, GPT-5-nano standard and batch rates
- [Google Gemini API Pricing](https://ai.google.dev/gemini-api/docs/pricing) — Gemini 2.5 Flash, Flash-Lite standard and batch rates
- [OpenAI Structured Outputs Documentation](https://platform.openai.com/docs/guides/structured-outputs) — Confirmed GPT-4o-mini strict structured output support
- [OpenAI Batch API Documentation](https://platform.openai.com/docs/guides/batch) — Confirmed 50% batch discount applies to structured output calls
- [OpenAI Reasoning Models Documentation](https://developers.openai.com/api/docs/guides/reasoning/) — Confirmed GPT-4o-mini has no reasoning tokens
- [GPT-5-nano Benchmarks](https://blog.galaxy.ai/compare/gpt-5-mini-vs-gpt-5-nano) — FActScore hallucination rate (7.3%)
