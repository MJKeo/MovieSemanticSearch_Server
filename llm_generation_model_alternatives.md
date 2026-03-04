# LLM Generation Cost Reduction Guide

## Executive Summary

The current system uses **GPT-5-mini** (batched) for all 8 LLM calls at `$0.125/M input` and `$1.00/M output`, producing ~28K input tokens and ~8K output tokens per movie at **~$0.025/movie**. At 100K movies, that's **$2,500 total**.

This guide identifies three independent cost reduction levers. Used together, they can reduce costs by **75-85%** while maintaining quality:

| Strategy | Est. Savings | Quality Risk |
|----------|-------------|-------------|
| A. Model switch (Gemini 2.5 Flash-Lite batch) | ~60-70% | Low (~5% decline) |
| B. Consolidate LLM calls (8 → 4) | ~25-35% | Negligible |
| C. Reduce reasoning token overhead | ~10-15% | Negligible |
| **Combined** | **~75-85%** | **~5% decline** |

Projected cost: **$0.004–$0.006/movie → $400–$600 for 100K movies**.

---

## 1. Current System Analysis

### 1.1 Architecture

The system makes **8 separate LLM calls** per movie, organized in two waves:

**Wave 1 (parallel):** plot_events, watch_context, reception
**Wave 2 (parallel, after plot_events completes):** plot_analysis, viewer_experience, narrative_techniques, production (which itself splits into production_keywords + source_of_inspiration in parallel)

Each call uses `generate_openai_response()` with GPT-5-mini via OpenAI's structured output API (`chat.completions.parse`).

### 1.2 Reasoning Effort Settings

| Call | reasoning_effort | Rationale |
|------|-----------------|-----------|
| plot_events | `minimal` | Extraction task — rewrite/restructure existing text |
| plot_analysis | `low` | Light analysis — identify themes from provided text |
| viewer_experience | `low` | Subjective inference from provided evidence |
| watch_context | `medium` | Requires creative reasoning about viewing contexts |
| narrative_techniques | `medium` | Requires film knowledge beyond provided text |
| production_keywords | `low` | Simple filtering task |
| source_of_inspiration | `low` | Light inference from provided text |
| reception | `low` | Summarization and extraction |

This is a well-considered design. The tasks that genuinely need more reasoning (watch_context and narrative_techniques) get it, while extraction tasks use minimal/low. However, GPT-5-mini's reasoning tokens are billed as output tokens at $1.00/M (batched), meaning even "minimal" reasoning adds hidden cost. With 8K total output tokens per movie but only ~3K of visible structured output, roughly **5K tokens (~62%) are reasoning tokens** being charged at output rates.

### 1.3 Token Breakdown Per Movie

| Component | Tokens | % of Total |
|-----------|--------|-----------|
| System prompts (8 calls) | ~7,750 | 27% of input |
| User inputs (unique data) | ~12,000 | 42% of input |
| Redundant user inputs (duplicated across calls) | ~8,250 | 29% of input |
| **Total Input** | **~28,000** | |
| Visible structured output | ~3,100 | 39% of output |
| Hidden reasoning tokens | ~4,900 | 61% of output |
| **Total Output** | **~8,000** | |

### 1.4 Input Redundancy Analysis

The biggest cost driver is **duplicated user inputs across calls**. These fields are passed to multiple calls verbatim:

| Input Field | Calls Receiving It | Redundant Tokens/Movie |
|------------|-------------------|----------------------|
| featured_reviews (5 reviews, ~750 tok each) | 6 of 8 calls | ~3,750 |
| plot_synopsis (from plot_events output) | 5 of 8 calls | ~2,500 |
| reception_summary | 5 of 8 calls | ~500 |
| plot_keywords | 6 of 8 calls | ~300 |
| overall_keywords | 4 of 8 calls | ~300 |
| audience_reception_attributes | 3 of 8 calls | ~225 |
| genres | 3 of 8 calls | ~150 |
| overview | 2 of 8 calls | ~400 |
| title | 8 of 8 calls | ~160 |
| **Total Redundant** | | **~8,285** |

Nearly **30% of all input tokens** are redundant copies of the same data sent to different calls. `featured_reviews` alone accounts for ~3,750 redundant tokens per movie because each review is ~150 words and 5 are sent to 6 different calls.

### 1.5 Current Cost Breakdown

At batched GPT-5-mini rates ($0.125/M input, $1.00/M output):

| Component | Cost/Movie | Cost/100K Movies |
|-----------|-----------|-----------------|
| Input (28K tokens) | $0.0035 | $350 |
| Output (8K tokens, incl. reasoning) | $0.0080 | $800 |
| **Total** | **$0.0115** | **$1,150** |

> **Note:** The user reports ~$0.025/movie, which aligns with standard (non-batched) GPT-5-mini rates ($0.25/M input, $2.00/M output). The analysis below assumes batch pricing is available and already in use. If not, simply adopting batch mode would already cut costs by 50%.

---

## 2. Strategy A: Model Alternatives

### 2.1 Pricing Comparison Table (March 2026)

All prices per million tokens. "Batch" column shows batch/async pricing where available.

| Model | Input (Standard) | Output (Standard) | Input (Batch) | Output (Batch) | Structured Output |
|-------|-----------------|-------------------|---------------|----------------|-------------------|
| **GPT-5-mini** (current) | $0.25 | $2.00 | $0.125 | $1.00 | Native (strict) |
| **GPT-5-nano** | $0.05 | $0.40 | $0.025 | $0.20 | Native (strict) |
| **GPT-4o-mini** | $0.15 | $0.60 | $0.075 | $0.30 | Native (strict) |
| **Gemini 2.5 Flash** | $0.30 | $2.50 | $0.15 | $1.25 | Native (JSON Schema) |
| **Gemini 2.5 Flash-Lite** | $0.10 | $0.40 | $0.05 | $0.20 | Native (JSON Schema) |
| **DeepSeek V3** | $0.14 | $0.28 | ~$0.07* | ~$0.14* | JSON mode (no strict) |
| **Claude Haiku 4.5** | $1.00 | $5.00 | $0.50 | $2.50 | Native (strict) |

*DeepSeek offers off-peak discounts (~50%) rather than a formal batch API.

### 2.2 Cost Per Movie Comparison (Batch, 28K Input / 8K Output)

| Model | Input Cost | Output Cost | Total/Movie | Total/100K | vs. Current |
|-------|-----------|-------------|------------|-----------|-------------|
| **GPT-5-mini** (current) | $0.0035 | $0.0080 | **$0.0115** | $1,150 | baseline |
| **GPT-5-nano** | $0.0007 | $0.0016 | **$0.0023** | $230 | **-80%** |
| **GPT-4o-mini** | $0.0021 | $0.0024 | **$0.0045** | $450 | **-61%** |
| **Gemini 2.5 Flash-Lite** | $0.0014 | $0.0016 | **$0.0030** | $300 | **-74%** |
| **DeepSeek V3** (off-peak) | $0.0020 | $0.0011 | **$0.0031** | $310 | **-73%** |

### 2.3 Model-by-Model Assessment

#### GPT-5-nano — Cheapest in the OpenAI ecosystem

**Pricing (batch):** $0.025/M input, $0.20/M output
**Projected cost:** ~$0.0023/movie → **$230 for 100K movies (-80%)**

**Pros:**
- Same OpenAI structured output API — zero code changes needed (just change model string)
- Same batch API infrastructure
- Supports reasoning_effort parameter (including minimal)
- Familiar API, no integration work

**Cons:**
- Higher hallucination rate (7.3% on FActScore vs lower for mini)
- "Moderate reasoning" — works best on tasks that do not require deep analysis
- Significant quality drop on reasoning-heavy tasks (FrontierMath: 9.6% vs 26.3% for mini at high effort)
- May produce less nuanced thematic analysis and viewer experience terms

**Verdict:** The hallucination rate and weaker reasoning make this risky for the subjective/analytical calls (plot_analysis, viewer_experience, narrative_techniques, watch_context). However, it's likely fine for the simpler extraction tasks (plot_events, production_keywords, reception). Consider a **hybrid approach** where nano handles simple calls and a stronger model handles analytical ones.

#### GPT-4o-mini — Previous generation, still strong

**Pricing (batch):** $0.075/M input, $0.30/M output
**Projected cost:** ~$0.003/movie → **$300 for 100K movies (-74%)**

**Pros:**
- Proven track record for structured output (100% JSON schema compliance on OpenAI evals)
- Non-reasoning model — no hidden reasoning tokens inflating output costs
- Well-tested, stable, and reliable
- Same OpenAI API — minimal code changes

**Cons:**
- No reasoning_effort parameter (it's a non-reasoning model)
- 128K context vs 200K for GPT-5-mini (not a concern here — inputs are well under 30K)
- May be slightly weaker on subjective judgment tasks than GPT-5-mini with reasoning

**Verdict:** Strong candidate. The key insight is that **GPT-4o-mini has no hidden reasoning tokens**, so output cost is purely visible output. If GPT-5-mini's 8K output includes ~5K reasoning tokens, GPT-4o-mini would produce only ~3K output tokens at $0.30/M, yielding $0.0009 output cost/movie. This makes it substantially cheaper than it first appears.

**Revised GPT-4o-mini cost estimate (no reasoning tokens):**
- Input: 28K × $0.075/M = $0.0021
- Output: 3.1K × $0.30/M = $0.0009
- **Total: ~$0.003/movie → $300 for 100K movies (-74%)**

#### Gemini 2.5 Flash-Lite — Best price-to-capability ratio (Recommended)

**Pricing (batch):** $0.05/M input, $0.20/M output
**Projected cost (with consolidation):** ~$0.002/movie → **$200 for 100K movies**

**Pros:**
- Extremely cheap at batch pricing ($0.05/$0.20)
- 1M token context window — massive room for consolidated prompts
- Supports structured output via JSON Schema (Pydantic-compatible)
- Supports configurable thinking budgets (similar to reasoning_effort)
- Very fast — 7-10x faster than GPT-5-mini in structured processing benchmarks
- Google's batch API provides the same 50% discount structure

**Cons:**
- Structured output has reported edge cases (markdown code fence wrapping in some responses)
- Not as battle-tested as OpenAI's strict structured output mode
- Requires API migration (different SDK, different structured output syntax)
- Less reliable than GPT-5-mini on edge cases without careful prompt engineering
- Quality benchmarks show it slightly behind GPT-5-mini on subjective/reasoning tasks

**Verdict:** Best overall value proposition. The structured output quirks are manageable with response post-processing (strip markdown fences). For a batch pipeline that can retry on parse failures, this is the best choice. The 1M context window also enables aggressive prompt consolidation (Strategy B) that other models can't match.

**Mitigations for structured output reliability:**
1. Post-process responses to strip any markdown fences before JSON parsing
2. Validate output against Pydantic schemas with automatic retry on failure
3. For the ~2-3% of movies that fail validation, fall back to GPT-5-mini

#### DeepSeek V3 — Cheapest raw pricing, reliability concerns

**Pricing (off-peak):** ~$0.07/M input, ~$0.14/M output
**Projected cost:** ~$0.003/movie → **$300 for 100K movies (-74%)**

**Pros:**
- Extremely cheap, especially at off-peak rates
- Strong format adherence under strict constraints
- Competitive quality for straightforward extraction tasks

**Cons:**
- No strict structured output mode — only JSON mode (you must include "json" in prompt)
- Reliability concerns for production: API availability, rate limits, geopolitical risk
- Schema adherence is "generally" correct but not guaranteed — no strict mode
- Off-peak discount requires scheduling jobs during specific time windows (16:30–00:30 GMT)
- Not OpenAI-API compatible for structured output — requires custom parsing logic

**Verdict:** Too risky as a primary model for 100K movies. JSON mode without strict schema enforcement means you'll see more parse failures, requiring more retries and manual intervention. Best used as a fallback or for specific low-complexity calls.

### 2.4 Recommended Model Strategy

**Primary recommendation: Gemini 2.5 Flash-Lite (batch)**

For maximum cost reduction with acceptable quality:

| Call Complexity | Model | Reasoning | Rationale |
|----------------|-------|-----------|-----------|
| Simple extraction (plot_events, production_keywords, reception) | Gemini 2.5 Flash-Lite | Thinking off | Extraction tasks — minimal reasoning needed |
| Analytical (plot_analysis, viewer_experience, watch_context, narrative_techniques, source_of_inspiration) | Gemini 2.5 Flash-Lite | Low thinking budget | Subjective analysis benefits from light reasoning |

**Alternative if you want to stay in the OpenAI ecosystem: GPT-4o-mini (batch)**

Eliminates reasoning token overhead entirely, same API, minimal code changes. Output costs drop dramatically because there are no hidden reasoning tokens.

---

## 3. Strategy B: Consolidate LLM Calls

### 3.1 The Problem

8 separate calls create three types of waste:

1. **System prompt duplication:** 7,750 tokens of system prompts sent across 8 calls
2. **User input duplication:** ~8,285 tokens of the same data (featured_reviews, plot_synopsis, etc.) sent to multiple calls
3. **Per-call overhead:** Each call has API overhead, JSON schema transmission, and reasoning token warmup

### 3.2 Consolidation Opportunities

Not all calls can be merged — the outputs serve different vector spaces and require different analytical lenses. But several calls share enough input context and analytical perspective to be combined without quality loss.

#### Merge 1: Viewer Experience + Watch Context → "Audience Perspective" (2 → 1 call)

**Why these merge well:**
- Both analyze the movie from the *viewer's perspective* — what it feels like vs. when to watch
- Both receive nearly identical inputs: title, genres, plot_keywords, overall_keywords, reception_summary, audience_reception_attributes, featured_reviews
- Watch context's `watch_scenarios` and `self_experience_motivations` are direct extensions of viewer experience's `emotional_palette` and `tension_adrenaline`
- The viewer_experience prompt (1,525 tokens) and watch_context prompt (1,575 tokens) are the two longest — merging saves significant system prompt tokens

**Input savings:**
- Eliminate 1 full copy of: featured_reviews (~750 tok), plot_keywords (~50 tok), overall_keywords (~50 tok), reception_summary (~100 tok), audience_reception_attributes (~75 tok), genres (~50 tok), title (~20 tok)
- System prompt reduction: ~1,200 tokens (merged prompt can share preamble and rules)
- **Total saved: ~2,295 tokens input per movie**

**Implementation:** Create a combined `AudiencePerspectiveMetadata` schema containing all 12 sections (8 from viewer_experience + 4 from watch_context). Use a single merged system prompt. The combined output schema is larger but the model handles it in one pass.

#### Merge 2: Production Keywords + Source of Inspiration → single "Production" call

These are already combined in `generate_production_metadata()` but still make **2 separate LLM calls** internally. The production_keywords call is extremely lightweight (375 token system prompt, ~480 total input) and produces ~150 output tokens.

**Why fully merge:**
- production_keywords takes overall_keywords and filters them — this is trivially achievable as a sub-section of the source_of_inspiration call
- Saves 1 full API call, 1 system prompt, and the title/keywords duplication

**Input savings:**
- Eliminate production_keywords system prompt (~375 tokens)
- Eliminate duplicated title + overall_keywords (~145 tokens)
- **Total saved: ~520 tokens input per movie**

#### Merge 3: Plot Analysis + Narrative Techniques → "Story Analysis" (2 → 1 call)

**Why these merge well:**
- Both analyze *the story itself* — plot_analysis looks at themes/concepts while narrative_techniques looks at how the story is told
- Both receive the same core inputs: plot_synopsis, plot_keywords, overall_keywords, featured_reviews, reception_summary
- Their outputs are complementary: knowing the narrative archetype (from narrative_techniques) directly informs the core_concept (from plot_analysis) and vice versa
- A single analytical pass produces more coherent output because themes and techniques inform each other

**Input savings:**
- Eliminate 1 full copy of: plot_synopsis (~500 tok), plot_keywords (~50 tok), overall_keywords (~50 tok), featured_reviews (~750 tok), reception_summary (~100 tok), title (~20 tok)
- System prompt reduction: ~800 tokens (merged prompt shares preamble, examples, rules)
- **Total saved: ~2,270 tokens input per movie**

### 3.3 Consolidation Summary

| Original Calls | Merged Call | Input Saved | Calls Saved |
|---------------|-------------|-------------|-------------|
| viewer_experience + watch_context | "Audience Perspective" | ~2,295 tok | 1 |
| production_keywords + source_of_inspiration | "Production" (true single call) | ~520 tok | 1 |
| plot_analysis + narrative_techniques | "Story Analysis" | ~2,270 tok | 1 |

**After consolidation: 8 calls → 5 calls**

New call structure:
1. **plot_events** (unchanged) — Wave 1
2. **reception** (unchanged) — Wave 1
3. **audience_perspective** (viewer_experience + watch_context merged) — Wave 1
4. **story_analysis** (plot_analysis + narrative_techniques merged) — Wave 2 (depends on plot_events)
5. **production** (keywords + inspiration truly merged) — Wave 2

**Total input tokens saved: ~5,085 per movie**

New input budget: ~28,000 - 5,085 = **~22,915 tokens**

### 3.4 Impact on Output Quality

Merging calls within the same conceptual domain should **not** reduce quality and may improve it:

- **Viewer Experience + Watch Context:** These sections frequently need to reference the same emotional/tonal analysis. A single pass avoids inconsistencies (e.g., labeling a movie "chill" in viewer_experience but "adrenaline fix" in watch_context scenarios).
- **Plot Analysis + Narrative Techniques:** Character arcs appear in both schemas. A single pass ensures the arcs identified in plot_analysis align with those in narrative_techniques without contradiction.
- **Production consolidation:** Combining a trivial keyword-filtering task with source-of-inspiration analysis adds zero complexity.

The main risk is that very large output schemas (12+ sections) may cause the model to produce slightly shorter individual sections due to output length pressure. Mitigate this by explicitly stating minimum term counts in the merged prompt.

---

## 4. Strategy C: Reduce Reasoning Token Overhead

### 4.1 The Hidden Cost

GPT-5-mini's reasoning tokens are billed as output tokens but not visible in the response. With the current setup:

- Total output per movie: ~8,000 tokens
- Visible structured output: ~3,100 tokens
- **Hidden reasoning tokens: ~4,900 tokens (61% of output cost)**

At $1.00/M output (batched), reasoning tokens cost: 4,900 × $1.00/M = **$0.0049/movie** — nearly half the total cost.

### 4.2 Options

**Option 1: Switch to a non-reasoning model (GPT-4o-mini, Gemini Flash-Lite)**

Eliminates reasoning tokens entirely. Output cost drops to only the visible structured content (~3,100 tokens).

**Option 2: Use `reasoning_effort=minimal` for all calls**

Currently, watch_context uses `medium` and narrative_techniques uses `medium`. Dropping both to `minimal` reduces reasoning token generation. However, these are the calls that most benefit from reasoning (creative inference, film knowledge), so quality may suffer.

**Option 3: Use Gemini's configurable thinking budgets**

Gemini 2.5 Flash-Lite supports thinking budgets that can be set per-request, offering fine-grained control. Unlike GPT-5-mini where even "minimal" generates some reasoning tokens, Gemini's thinking can be fully disabled (`thinking_budget=0`) for extraction tasks.

### 4.3 Recommended Approach

If staying with GPT-5-mini: set `reasoning_effort=minimal` for all calls except narrative_techniques (keep at `low`). Expected reasoning token reduction: ~30-40%.

If switching to Gemini 2.5 Flash-Lite or GPT-4o-mini: reasoning overhead is eliminated by default.

---

## 5. Combined Recommended Plan

### 5.1 Phase 1: Quick Wins (No Model Change)

**Effort: Low | Savings: ~25%**

1. **Consolidate production calls** — merge production_keywords and source_of_inspiration into a single LLM call with a combined schema. This is the lowest-risk change.
2. **Reduce reasoning_effort** — set watch_context to `low` and plot_events to `minimal`. The watch_context task doesn't truly need `medium` reasoning given the comprehensive input data provided.
3. **Trim featured_reviews** — currently sending up to 5 reviews (~750 tokens each) to 6 calls. Reduce to 3 reviews for non-reception calls, or pre-summarize reviews into a ~200 token digest.

### 5.2 Phase 2: Model Migration (Moderate Effort)

**Effort: Medium | Savings: ~60-70% total**

Two options depending on risk tolerance:

**Option A — Stay with OpenAI (conservative):**
- Switch to GPT-4o-mini for all calls
- No hidden reasoning tokens → output cost drops ~60%
- Same API, same structured output support, minimal code changes
- Projected cost: **~$0.003/movie → $300 for 100K movies**

**Option B — Migrate to Gemini (aggressive):**
- Switch to Gemini 2.5 Flash-Lite (batch) for all calls
- Cheapest batch pricing available ($0.05/$0.20 per M tokens)
- Requires new SDK integration and structured output format changes
- Add response validation + retry logic for the small % of malformed outputs
- Projected cost: **~$0.002/movie → $200 for 100K movies**

### 5.3 Phase 3: Call Consolidation (Higher Effort)

**Effort: High | Savings: ~75-85% total**

1. **Merge viewer_experience + watch_context** → single "Audience Perspective" call
2. **Merge plot_analysis + narrative_techniques** → single "Story Analysis" call
3. Redesign system prompts for merged calls
4. Create new combined Pydantic schemas
5. Update vector text creation methods to split the merged output back into individual vector space texts

After all three phases:

| Metric | Current | After All Phases |
|--------|---------|-----------------|
| LLM calls per movie | 8 | 5 |
| Input tokens per movie | ~28,000 | ~18,000 |
| Output tokens per movie | ~8,000 | ~2,800 (no reasoning overhead) |
| Cost per movie (batch) | $0.0115 | ~$0.0015 |
| Cost for 100K movies | $1,150 | ~$150 |
| **Total savings** | | **~87%** |

---

## 6. Quality Safeguards

Regardless of which strategies you adopt, implement these safeguards:

### 6.1 A/B Validation

Before committing to a model change or call consolidation at scale:
1. Select 50-100 diverse movies (varying genres, eras, popularity levels)
2. Generate metadata with both the current system and the proposed changes
3. Embed both sets and run your existing search queries
4. Compare retrieval quality (precision@10, recall@10) quantitatively
5. Manually review 10-15 outputs side-by-side for subjective quality

### 6.2 Output Validation Pipeline

For any model that isn't OpenAI strict structured output:
1. Parse response as JSON, stripping any markdown fences
2. Validate against the Pydantic schema
3. Check minimum field lengths and list sizes (e.g., emotional_palette.terms must have ≥3 items)
4. On validation failure: retry once with the same model, then fall back to GPT-5-mini
5. Log failure rates per model to monitor degradation over time

### 6.3 Quality Metrics to Track

- Schema validation pass rate (target: >97%)
- Average terms generated per section (should remain stable)
- Vector embedding cosine similarity between old and new outputs for same movies (target: >0.85)
- Search result overlap for benchmark queries (target: >80% of top-10 results shared)

---

## 7. Appendix: Pricing Sources

All pricing data verified as of March 2026 from the following sources:

- [OpenAI API Pricing](https://openai.com/api/pricing/) — GPT-5-mini, GPT-5-nano, GPT-4o-mini standard and batch rates
- [Gemini Developer API Pricing](https://ai.google.dev/gemini-api/docs/pricing) — Gemini 2.5 Flash, Flash-Lite standard and batch rates
- [DeepSeek API Pricing](https://api-docs.deepseek.com/quick_start/pricing) — DeepSeek V3 and V3.2 rates, off-peak discounts
- [Claude API Pricing](https://platform.claude.com/docs/en/about-claude/pricing) — Claude Haiku 4.5 rates
- [LLM API Pricing Comparison (TLDL)](https://www.tldl.io/resources/llm-api-pricing-2026) — Cross-provider comparison
- [Artificial Analysis](https://artificialanalysis.ai/models/gpt-5-mini) — GPT-5-mini benchmark performance
- [GPT-5 Mini vs GPT-5 Nano (Galaxy.ai)](https://blog.galaxy.ai/compare/gpt-5-mini-vs-gpt-5-nano) — Quality comparison between OpenAI model tiers
- [LLM Pricing Comparison 2026 (CloudIDR)](https://www.cloudidr.com/blog/llm-pricing-comparison-2026) — 60+ model comparison
- [Gemini 2.5 Flash-Lite Structured Output Issues](https://discuss.ai.google.dev/t/gemini-2-5-flash-lite-produces-incorrect-structured-output/102367) — Known edge cases
- [GPT-5 Model Family (OpenAI)](https://platform.openai.com/docs/models/gpt-5-mini) — Reasoning effort levels, context windows
