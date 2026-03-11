# ADR-012: LLM Generation Cost Optimization (Proposals)

**Status:** Proposed (not yet implemented)

## Context

The ingestion-time LLM metadata generation makes 8 API calls per
movie, currently costing ~$0.025/movie (~$2,500 for 100K movies).
Three analysis documents explored reduction strategies.

## Analysis Summary

### Current State

- 8 LLM calls per movie, organized in two parallel waves
- Wave 1: plot_events, watch_context, reception
- Wave 2: plot_analysis, viewer_experience, narrative_techniques,
  production_keywords, source_of_inspiration
- ~28K input tokens + ~8K output tokens per movie
- ~58% of output tokens are hidden reasoning tokens (billed but
  invisible)

### Largest Cost Drivers

1. Featured reviews duplicated across 6 of 8 calls (~32% of input)
2. System prompts (~29% of input)
3. Hidden reasoning tokens (~58% of output cost)
4. Plot synopsis propagation to calls that don't need it (~13%)
5. Justification/explanation fields in output schemas that are
   never used in embeddings (~27 unused fields)

## Proposed Phased Approach

### Phase 1: Schema Cleanup + Batch API (80% savings)
- Remove 28 justification/explanation fields from output schemas
- Flatten CharacterArc, MajorTheme, MajorLessonLearned to plain
  strings
- Remove MajorCharacter.role field
- Switch to OpenAI Batch API (50% discount)

### Phase 2: Model Migration (89% cumulative savings)
- Switch from GPT-5-mini to GPT-4o-mini
- Eliminates hidden reasoning tokens entirely
- Same API, same structured output support, minimal code changes
- Output cost drops ~88%

### Phase 3: Input Token Reduction (92% cumulative)
- Send review summaries (not full text) to 4 of 6 calls
- Reduce review count from 5 to 3
- Remove plot_synopsis from viewer_experience and
  source_of_inspiration inputs

### Phase 4: Call Consolidation (94% cumulative)
- Merge viewer_experience + watch_context → "Audience Perspective"
- Merge plot_analysis + narrative_techniques → "Story Analysis"
- Merge production_keywords + source_of_inspiration → single call
- 8 calls → 5 calls
- Compress system prompts (~30-40%)

### Projected End State

| Metric | Current | After All Phases |
|--------|---------|-----------------|
| LLM calls/movie | 8 | 5 |
| Input tokens/movie | ~28,000 | ~9,000 |
| Output tokens/movie | ~8,000 | ~2,400 |
| Cost/movie (batch) | ~$0.025 | ~$0.0015 |
| Cost/100K movies | ~$2,500 | ~$150 |

## Model Alternatives Analyzed

| Model | Batch $/movie | Pros | Cons |
|-------|--------------|------|------|
| GPT-5-mini (current) | $0.0115 | Best quality | Hidden reasoning tokens |
| **GPT-4o-mini (recommended)** | $0.003 | No reasoning overhead, same API | Slightly weaker on subjective analysis |
| GPT-5-nano | $0.0023 | Cheapest OpenAI | 7.3% hallucination rate |
| Gemini 2.5 Flash-Lite | $0.002 | Cheapest overall | Structured output quirks, API migration |

## What NOT to Change

- 8 separate vector spaces (merge calls, not vectors)
- Full plot synopses for plot_events input
- Negation terms in viewer_experience
- Plot synopsis for plot_analysis and narrative_techniques
- Two-wave execution architecture

## Quality Safeguards

Any change should be validated on 50-100 diverse movies:
- Compare precision@10 and recall@10 (target: >80% overlap)
- Cosine similarity old vs new embeddings (target: >0.85)
- Schema validation pass rate (target: >97%)
- Manual review of 15-20 outputs

## References

- docs/modules/llms.md (ingestion-time LLM generation details)
