# ADR-033 — Plot Events Cost Optimization & Conditional Generation

## Status
Active

## Context

Stage 6 generates 7 types of LLM metadata per movie (~112K movies). The
current plot_events generator (Wave 1) produces a detailed plot_summary
from all available text sources (synopsis, summaries, overview, keywords),
which is then passed as `plot_synopsis` to 4 of 6 Wave 2 generators. This
design has cost inefficiencies:

1. **Synopsis movies (~22,894) re-generate what already exists.** The
   current generator stitches sources into a detailed summary. For movies
   with a rich IMDB synopsis (~1,172 tokens avg), this is largely
   redundant — the synopsis is already a comprehensive plot recount.

2. **All downstream generators receive the same plot text**, regardless
   of whether they need fine detail. Analysis showed:
   - `narrative_techniques`: needs structural detail (POV, timeline,
     twists) — benefits from rich input
   - `plot_analysis` and `viewer_experience`: produce short thematic
     labels — a condensed summary is sufficient and reduces input bloat
   - `source_of_inspiration`: barely uses plot at all — keywords +
     reviews + title are primary signals

3. **Long synopses create edge-case problems.** Synopses range from
   ~171 tokens (P10) to ~14,943 tokens (max). The embedding model
   (`text-embedding-3-small`) has a hard limit of 8,191 tokens and
   quality degrades with longer inputs. Passing 5,000+ token synopses
   to downstream generators wastes money on input that doesn't improve
   short-label outputs.

### Population breakdown (111,898 eligible movies)

| Segment | Count | % | Available text |
|---------|-------|---|----------------|
| Has synopsis | 22,894 | 20.5% | Synopsis (~1,172 tok avg) + overview |
| No synopsis, has summaries | 51,328 | 45.9% | Summaries (~235 tok first 3 combined avg) + overview |
| Overview only | 37,547 | 33.6% | Overview (~53 tok avg) + keywords |
| Nothing | 129 | 0.1% | — |

### Synopsis length tail

| Percentile | Chars | ~Tokens |
|------------|-------|---------|
| P90 | 8,596 | 2,149 |
| P95 | 11,162 | 2,790 |
| P99 | 20,452 | 5,113 |
| Max | 59,774 | 14,943 |

Movies above 8K chars (~2K tokens): 2,752 (12% of synopsis movies).

### Summary combined length tail (non-synopsis movies)

| Percentile | Chars | ~Tokens |
|------------|-------|---------|
| P90 | 1,724 | 431 |
| P99 | 3,215 | 803 |
| Max | 14,710 | 3,677 |

Summaries are well-behaved; only 6 movies exceed 8K chars combined.

## Decision

### 1. Preliminary distillation step: dropped

An initial plan to use gpt-5-nano to condense long synopses before
generation was tested and abandoned. Testing on Star Wars showed 76%
compression (far beyond target), introduced hallucinations (wrong
character relationships, fabricated events), and lost detail on iconic
searchable scenes. Long synopses are handled via the synopsis quality
gate (see section 2b) and truncation/downstream prompt limits instead.

### 2. Redesigned plot_events generation: two branches

The plot_events generator operates in two modes based on whether a
synopsis is present and meets a minimum quality threshold.

#### Branch A — Movie has a synopsis (≥1000 chars, `MIN_SYNOPSIS_CHARS`)

**Inputs:** synopsis, overview, plot_keywords.

**Prompt focus:** Condensation. The synopsis is already a comprehensive
plot recount — the LLM's job is to produce a shorter, summary-like
version. Preserves major plot beats, character arc transitions, and
thematic turning points while dropping scene-level detail.

**System prompt:** `SYSTEM_PROMPT_SYNOPSIS` — frames task as condensation.

**Generates:** `plot_summary`, `setting`, `major_characters`.

#### Synopsis quality gate (`MIN_SYNOPSIS_CHARS = 1000`)

Short synopses (under 1,000 chars) are consistently non-plot text
(blurbs, review fragments, marketing copy). When a synopsis exists
but falls below this threshold, it is demoted: prepended to the
summaries list and routed to Branch B instead. This prevents the
condensation branch (which prohibits model knowledge) from being
forced to fabricate content from an inadequate source.

#### Branch B — Movie does not have a synopsis (or has thin synopsis)

**Inputs:** summaries (first 3), overview, plot_keywords.

**Prompt focus:** Consolidation. Multiple partial sources are merged
into a single organized account. The model is explicitly framed as a
text consolidator, not a narrative generator, to prevent fabrication
from sparse input. Model knowledge is not permitted — if input is
sparse, output is short.

**System prompt:** `SYSTEM_PROMPT_SYNTHESIS` — frames task as
consolidation ("you have no knowledge of any film" fiction to remove
self-assessment problem). Labels input types by what they are NOT good
for. Traceability as internal check: model must verify each detail
appears in input before including it.

**Output length:** 4K token soft cap in prompt + `max_tokens=5000`
hard safety net.

**Generates:** `plot_summary`, `setting`, `major_characters`.

#### Skip condition

Movies with no synopsis, no summaries, and no overview (129 movies)
skip plot_events generation entirely.

#### Generator function interface

`build_plot_events_prompts(movie)` returns `(user_prompt, system_prompt)`
— branching logic is fully contained here. Provider default is
`LLMProvider.OPENAI` with `gpt-5-mini`; Gemini cannot be used as default
because it requires `max_output_tokens` rather than `max_tokens` and
the generic router does not normalize this parameter.

### 3. source_of_inspiration drops plot entirely

Removed `plot_synopsis` parameter from the generator, prompt builder,
`_PREAMBLE`, skip condition check, and all call sites. Saves ~748 tokens
per movie × 111,769 movies = ~83.6M input tokens ($12.54 at gpt-5-mini
rates). Keywords + reviews + title remain sufficient primary signals.

### 4. Downstream routing unchanged

All Wave 2 generators still receive `plot_synopsis` extracted from
`plot_events_output.plot_summary` uniformly. No conditional routing —
all complexity lives in the generation step.

### 5. Schema field descriptions

`PlotEventsOutput` and `MajorCharacter` field descriptions were stripped
to minimal neutral labels (e.g., "Chronological plot summary.",
"Character name."). All behavioral instructions moved into branch-specific
system prompts. This prevents the structured output schema from creating
a competing signal that overrides prompt-level constraints (see ADR-036).

## Alternatives Considered

- **gpt-5-nano distillation preliminary step**: Tested, abandoned. Model
  cut too aggressively (76% compression), introduced hallucinations and
  lost key searchable details. Distillation as a preprocessing step was
  cleaner architecturally but produced poor quality output.
- **Allow knowledge supplementation in synthesis branch**: Briefly added,
  then reverted. Testing showed the model couldn't reliably distinguish
  recall from fabrication; granting permission undermined every subsequent
  guardrail. "No knowledge" fiction proved more effective.
- **Dual summaries (broad + detailed) for synopsis movies**: Considered
  for $8.73 savings potential. Rejected — added complexity to dual output
  fields and conditional downstream routing without sufficient benefit.
- **Branch on synopsis richness rather than presence**: No clean
  richness threshold exists; the `MIN_SYNOPSIS_CHARS` quality gate
  handles the thin-synopsis edge case more directly.

## Consequences

- Significant token savings from synopsis movies not regenerating existing
  content, and from source_of_inspiration dropping plot input.
- Evaluation pipeline accepts `--branch synopsis|synthesis` flag to test
  each branch separately.
- Legacy prompts (`SYSTEM_PROMPT`, `SYSTEM_PROMPT_SHORT`) kept for
  backwards compat with existing eval candidates.
- Unit tests for `source_of_inspiration` and `plot_events` generators
  need updated signatures.
- `MIN_SYNOPSIS_CHARS` threshold must also be applied at embedding time
  when the pipeline uses `plot_synopsis` from `imdb_data` directly
  (tracked in docs/TODO.md).

## References

- ADR-025 (schema design) — schema field descriptions context
- ADR-036 (schema field description minimalism) — why field descriptions
  were moved to prompts
- `movie_ingestion/metadata_generation/generators/plot_events.py`
- `movie_ingestion/metadata_generation/prompts/plot_events.py`
