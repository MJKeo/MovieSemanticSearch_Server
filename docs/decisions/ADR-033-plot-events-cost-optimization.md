# ADR-033 — Plot Events Cost Optimization & Conditional Generation

## Status
Proposed

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

### 1. Preliminary step: Distill long synopses with gpt-5-nano

Before any plot_events generation runs, a one-time preprocessing pass
identifies synopsis movies where the first synopsis exceeds a character
threshold and condenses them using gpt-5-nano. The condensed version
**replaces** the original synopsis in the database (the original is
acceptably lost — text that long is too large to embed or use
effectively anywhere).

**Why a separate preliminary step:**
- Decouples the long-tail edge case from the main generation logic.
  By the time plot_events runs, all synopses are at a reasonable length.
- gpt-5-nano is the cheapest available model ($0.05/M input, $0.20/M
  output) and condensation is a straightforward task well within its
  capabilities.
- Cost for the entire pass: ~$1.54 at the 2K-token threshold (2,752
  movies), or ~$0.48 at the 4K-token threshold (442 movies).

**Implementation details:**

- **Threshold:** Character-based for fast checking (no tokenization
  needed at decision time). Exact value TBD — 12,000 chars (~3K tokens)
  is the recommended starting point based on embedding quality
  considerations and cost.
- **Model:** gpt-5-nano.
- **Prompt guidance:** Condense the synopsis to approximately
  [threshold] tokens while preserving: major plot beats in
  chronological order, character names and key actions, narrative
  structure (non-linear elements, flashback positions, twist/reveal
  mechanics), and resolution. Aggressively cut: dialogue, atmospheric
  description, scene-level transitions, minor character interactions.
- **Database update:** Replace the synopsis value in the `synopses`
  JSON array in the `imdb_data` table. This is a one-way operation —
  the original long text is not preserved.
- **Idempotency:** Movies already under the threshold are skipped.
  Movies already condensed (re-runs) are skipped by checking length.

### 2. Redesigned plot_events generation: two branches

The plot_events generator now operates in two modes based on whether
a synopsis is present.

#### Option A — Movie has a synopsis

**Inputs:** synopsis (now guaranteed to be at a reasonable length after
the preliminary distillation), overview, plot_keywords.

**Prompt focus:** Condensation. The synopsis is already a comprehensive
plot recount — the LLM's job is to produce a shorter, summary-like
version. The prompt instructs the model to abbreviate the synopsis into
a condensed plot_summary, focusing on major plot beats, character arc
transitions, and thematic turning points while dropping scene-level
detail. Length should be proportional to the story's complexity (not a
fixed token target — a 30-minute short film and a 2.5-hour epic have
different amounts of meaningful content).

**Generates:** `plot_summary`, `setting`, `major_characters`.

**Key difference from current:** The current generator treats all inputs
equally and stitches them together. Option A recognizes the synopsis as
the dominant source and focuses the LLM on compression rather than
synthesis.

#### Option B — Movie does not have a synopsis

**Inputs:** summaries (first 3, when available), overview, plot_keywords.

**Prompt focus:** Synthesis. Multiple partial sources need to be unified
into a single coherent plot picture. This is essentially the current
behavior.

**Generates:** `plot_summary`, `setting`, `major_characters`.

**Output length cap:** The structured output schema should include
prompt guidance encouraging the model to keep plot_summary under
approximately 5,000 tokens. This is a soft guardrail — the model
self-regulates rather than being hard-truncated mid-sentence. Combined
with `max_tokens` on the API call as a hard safety net. In practice
this rarely triggers (median non-synopsis input is ~171 tokens), but
prevents bloat from the rare very-long-summary outliers.

#### Skip condition

Movies with no synopsis, no summaries, and no overview (129 movies) skip
plot_events generation entirely. Same as current behavior.

### 3. Downstream routing changes

**`plot_summary` is always the downstream value.** All Wave 2 generators
that currently receive `plot_synopsis` will continue to receive it, and
it always comes from the generated `plot_summary` field of
PlotEventsOutput. No conditional routing — all complexity lives in the
plot_events generation step.

**`source_of_inspiration` drops plot entirely.** This generator currently
receives `plot_synopsis` but barely uses it. Its prompt already says
parametric knowledge + keywords + reviews are the primary signals.
Removing `plot_synopsis` from its input saves ~748 tokens per movie ×
111,769 movies = ~83.6M input tokens ($12.54 at gpt-5-mini rates).
The skip condition check should also be updated to no longer consider
plot_synopsis.

### 4. Embedding

The `plot_summary` field from PlotEventsOutput is always what gets
embedded into `plot_events_vectors`. For synopsis movies, this is the
abbreviated generated summary (not the raw synopsis). For non-synopsis
movies, this is the synthesized summary. This keeps the embedded text
in a good range for `text-embedding-3-small` quality (the preliminary
distillation ensures no synopsis exceeds the threshold, and the
generated summary will always be shorter than the input synopsis).

## Implementation Guide

### Files to modify

| File | Changes |
|------|---------|
| `movie_ingestion/metadata_generation/prompts/plot_events.py` | Add two prompt variants: one for synopsis condensation (Option A), one for synthesis (Option B). Current `SYSTEM_PROMPT_SHORT` can serve as a base for Option B. |
| `movie_ingestion/metadata_generation/generators/plot_events.py` | Branch on `movie.plot_synopses` presence. Option A: pass synopsis + overview + keywords, use condensation prompt. Option B: pass summaries + overview + keywords, use synthesis prompt (current behavior). Select the appropriate system prompt per branch. |
| `movie_ingestion/metadata_generation/generators/source_of_inspiration.py` | Remove `plot_synopsis` parameter from `build_source_of_inspiration_user_prompt()` and `generate_source_of_inspiration()`. |
| `movie_ingestion/metadata_generation/prompts/source_of_inspiration.py` | Remove `plot_synopsis` from the INPUTS section of both prompt variants. |
| `movie_ingestion/metadata_generation/pre_consolidation.py` | Update `_check_source_of_inspiration()` to not consider `plot_synopsis`. Update the call in `assess_skip_conditions()`. |
| New file: `movie_ingestion/metadata_generation/distill_long_synopses.py` | Preliminary distillation script. Queries `imdb_data` for synopses exceeding the character threshold, condenses with gpt-5-nano, updates the `synopses` column in-place. Should be runnable as `python -m movie_ingestion.metadata_generation.distill_long_synopses`. |

### Files that do NOT change

| File | Why unchanged |
|------|---------------|
| `movie_ingestion/metadata_generation/schemas.py` | `PlotEventsOutput` schema stays the same — still has `plot_summary`, `setting`, `major_characters`. |
| `movie_ingestion/metadata_generation/inputs.py` | `MovieInputData` is unchanged. The branching logic reads `movie.plot_synopses` which already exists. |
| `movie_ingestion/metadata_generation/pre_consolidation.py` (Wave 2 flow) | `plot_synopsis` extraction from `plot_events_output.plot_summary` is unchanged. Downstream generators still receive it the same way. Only the `source_of_inspiration` check changes. |
| All other Wave 2 generators | `plot_analysis`, `viewer_experience`, `narrative_techniques` still receive `plot_synopsis` exactly as before. They don't know or care that it was generated differently. |
| `movie_ingestion/metadata_generation/wave1_runner.py` | Storage format unchanged — `PlotEventsOutput` is still serialized to JSON in `wave1_results`. |
| `movie_ingestion/metadata_generation/request_builder.py` | Batch request building is unchanged at the orchestration level — the generator internally picks the right prompt/inputs. |

### Step-by-step implementation order

#### Step 1: Create the distillation script

Create `movie_ingestion/metadata_generation/distill_long_synopses.py`:

1. Connect to `ingestion_data/tracker.db`.
2. Query `imdb_data` for all movies where synopses JSON array's first
   entry exceeds the character threshold (e.g., 8,000 chars).
   Join on `movie_progress` to filter to `imdb_quality_calculated` status.
3. For each movie over the threshold:
   a. Extract the first synopsis from the JSON array.
   b. Call gpt-5-nano with a condensation prompt. The prompt should
      instruct the model to condense to approximately [threshold / 4]
      tokens while preserving: major plot beats in chronological order,
      character names and key actions, narrative structure elements
      (non-linear elements, flashback positions, twist/reveal mechanics),
      and the resolution. Aggressively cut: dialogue, atmospheric
      description, scene transitions, minor character interactions.
   c. Replace the first entry in the synopses JSON array with the
      condensed text.
   d. UPDATE the `synopses` column in `imdb_data`.
4. Use async concurrency with a semaphore (similar to existing patterns
   in the codebase) for throughput.
5. Commit in batches for crash safety.
6. Log progress: count processed, count skipped (already short), count
   failed.

**Run before any plot_events generation.** This is a one-time
preprocessing step.

#### Step 2: Create the Option A prompt (synopsis condensation)

In `movie_ingestion/metadata_generation/prompts/plot_events.py`, add a
new prompt variant for synopsis condensation. Key differences from the
current prompt:

- The preamble should frame the task as **condensation**, not synthesis.
  The synopsis is already comprehensive — the LLM's job is to produce a
  shorter version, not to stitch sources together.
- Field instructions for `plot_summary` should say: abbreviate the
  synopsis into a condensed summary. Preserve major plot beats,
  character arc transitions, thematic turning points, and the
  resolution. Drop scene-level description, dialogue specifics, minor
  subplots, and transitional events. Length should be proportional to
  the story's complexity.
- The inputs list should reflect what Option A receives: synopsis,
  overview, plot_keywords (not summaries).
- `setting` and `major_characters` instructions are unchanged.

Export as `SYSTEM_PROMPT_SYNOPSIS` and `SYSTEM_PROMPT_SYNOPSIS_SHORT`
(following the existing long/short pattern).

#### Step 3: Modify the plot_events generator to branch

In `movie_ingestion/metadata_generation/generators/plot_events.py`:

1. Update `build_plot_events_user_prompt()` to branch on
   `movie.plot_synopses`:
   - **Has synopsis (Option A):** Include first synopsis (collapsed
     newlines, as current), overview, plot_keywords. Do NOT include
     plot_summaries — the synopsis is the dominant source and summaries
     would add redundant tokens.
   - **No synopsis (Option B):** Include plot_summaries (first 3, as
     current), overview, plot_keywords. This is essentially the current
     behavior.

2. Update `generate_plot_events()` to select the appropriate system
   prompt based on the branch:
   - Option A: `SYSTEM_PROMPT_SYNOPSIS_SHORT`
   - Option B: `SYSTEM_PROMPT_SHORT` (current prompt, unchanged)

3. The function signature, return type, and downstream interface are
   all unchanged. Callers don't know about the branching.

#### Step 4: Remove plot from source_of_inspiration

1. In `generators/source_of_inspiration.py`:
   - Remove `plot_synopsis` parameter from
     `build_source_of_inspiration_user_prompt()`.
   - Remove `plot_synopsis` parameter from
     `generate_source_of_inspiration()`.
   - Remove `plot_synopsis=plot_synopsis` from the `build_user_prompt()`
     call inside the builder function.

2. In `prompts/source_of_inspiration.py`:
   - Remove the `plot_synopsis` line from the INPUTS section of both
     `_PREAMBLE` variants.

3. In `pre_consolidation.py`:
   - Update `_check_source_of_inspiration()` signature to remove
     `plot_synopsis` parameter. The function should check only
     `merged_keywords` and `review_insights_brief`.
   - Update the call in `assess_skip_conditions()` (around line 413)
     to not pass `plot_synopsis`.

4. Update any callers that pass `plot_synopsis` to
   `generate_source_of_inspiration()` (check `wave1_runner.py`,
   `request_builder.py`, and evaluation files).

### Key design decisions and rationale

| Decision | Rationale |
|----------|-----------|
| **Preliminary distillation as a separate step** | Decouples edge-case handling from generation logic. Plot_events doesn't need to worry about 15K-token inputs. |
| **Replace synopsis in DB (no preservation)** | Text >8K chars is too long to embed (8,191 token hard limit) or use effectively. Original IMDB JSON files on disk serve as backup if ever needed. |
| **gpt-5-nano for distillation** | Cheapest available model ($0.05/$0.20 per M). Condensation is a straightforward task. Total cost ~$1.54. |
| **Character-based threshold (not token-based)** | Fast check — no tokenization needed. Characters ÷ 4 approximates tokens closely enough for a threshold decision. |
| **Proportional output length, not fixed token target** | A 30-minute short film and a 2.5-hour epic have different amounts of meaningful content. Guidelines on what to keep/cut scale naturally; a fixed target either over-compresses or under-compresses. |
| **Branch on synopsis presence, not richness** | Two clean branches. No need for "rich vs thin" sub-categories — the LLM naturally produces output proportional to input richness. |
| **source_of_inspiration drops plot** | Its own prompt identifies parametric knowledge + keywords + reviews as primary signals. Removing plot saves ~83.6M tokens ($12.54). Occasionally misses a detail but a worthwhile tradeoff. |
| **5K token soft cap on Option B output** | Safety net for rare long-summary outliers (max combined: 14,710 chars). In practice, median input is ~171 tokens so output will be far shorter. Prefer prompt guidance over hard truncation to avoid mid-sentence cutoffs. |
| **No separate broad/detailed summaries** | Considered generating two summary levels (broad for plot_analysis/viewer_experience, detailed for narrative_techniques). The cost savings (~$1-2) didn't justify the added complexity of dual output fields and conditional downstream routing. |
| **Downstream routing unchanged** | All Wave 2 generators still receive `plot_synopsis` extracted from `plot_events_output.plot_summary`. No conditional logic — all complexity is contained in the generation step. |

## Consequences

### Positive
- Significant token savings from synopsis movies not regenerating
  existing content, and from source_of_inspiration dropping plot input
- Long-synopsis edge cases handled cleanly upstream of the pipeline
- Simpler mental model: plot_events has two modes (condense vs
  synthesize), everything downstream is uniform
- Embedding quality improved by keeping all text within a reasonable
  token range

### Negative
- plot_analysis and viewer_experience still receive the full generated
  summary for all movies, which may be more detail than they need. This
  is a deliberate simplicity tradeoff (~$7-9 extra vs generating a
  separate broad summary)
- The preliminary distillation is an extra pipeline step that must run
  before generation
- Replacing synopses in the database is irreversible (mitigated by
  original IMDB JSON files on disk)

### Risks
- gpt-5-nano condensation quality is untested. If it produces poor
  condensations (losing key plot beats, hallucinating), the downstream
  impact is broad since the condensed synopsis feeds everything. Spot-
  check a sample before running on the full population.
- The proportional-length prompt guidance may produce inconsistent
  output lengths. Monitor the distribution of generated plot_summary
  lengths across the first batch and adjust prompt guidance if needed.
