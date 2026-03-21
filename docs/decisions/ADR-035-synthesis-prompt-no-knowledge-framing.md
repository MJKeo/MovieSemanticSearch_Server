# [035] — Synthesis Prompt: "No Knowledge" Framing to Prevent Hallucination

## Status
Active

## Context

The synthesis branch of plot_events generation (Branch B in ADR-033)
produces plot summaries for movies that lack a synopsis, using summaries,
overview, and plot keywords as inputs. This input set is often sparse.

Evaluation of synthesis-branch outputs identified three interacting
root causes for hallucination:

1. **Parametric knowledge permission undermined every guardrail.**
   An explicit "you may supplement from your own knowledge" rule was
   well-intentioned — for sparse input, model knowledge can improve
   embedding quality. However, models cannot reliably distinguish
   accurate recall from confabulation. Granting permission to use
   knowledge made every subsequent anti-fabrication rule ineffective:
   the model could always treat invented content as "supplementing."

2. **Narrative-creation task framing pressured gap-filling.**
   "Synthesize into a unified, coherent plot_summary" is a narrative
   creation instruction. Coherent narratives have structure, arcs, and
   continuity. For sparse input with structural gaps, the model filled
   those gaps to satisfy the task framing — inventing character names
   (Kalakalappu 2), fabricating plot beats (Meteor), and producing
   output disproportionate to input richness.

3. **Anti-fabrication positioned as an exception rather than the
   primary instruction.** When "do not fabricate" appears after a
   positive task framing, models treat fabrication avoidance as a
   secondary constraint that yields under pressure from the primary
   task. Severity in later sections does not override primacy effects.

## Decision

Rewrite `SYSTEM_PROMPT_SYNTHESIS` around a consolidation framing,
with model knowledge removal as the foundational premise.

**"You have no knowledge of any film"**: A deliberate fiction. The
model does have knowledge, but the statement removes the self-assessment
problem entirely. The model no longer needs to judge whether a detail is
accurate recall or fabrication — all details must come from the input.
This is more reliable than requiring accurate meta-cognition about
knowledge reliability.

**Task reframed as consolidation, not narrative creation**: "Consolidate
into a single organized account" rather than "synthesize into a coherent
summary." Reorganizing existing text does not require filling gaps;
creating a narrative does. Proportionality falls out naturally from
consolidation framing — if there's nothing to consolidate, the output
is short.

**Input types labeled by what they are NOT good for**:
- Overview: "often vague — do not treat as plot detail"
- Keywords: "context clues, not plot events"
This prevents the model from treating every input as a source of plot
events to elaborate on.

**Traceability as internal check, not output format**: "Before including
any detail, internally verify it appears in the input." Explicitly
followed by "Do not cite sources in your output" to prevent the model
from externalizing the verification as inline parenthetical citations.

**Anti-fabrication as primary, not exception**: CRITICAL section at the
top of the prompt, before field instructions. Fabrication avoidance is
the primary constraint; consolidation is the task within that constraint.

## Alternatives Considered

1. **Knowledge permission with stronger guardrails**: The previous
   approach. Failed because permission grants create exception-seeking
   behavior — the model applies them whenever convenient. Removing
   permission entirely is more reliable than hedging it.

2. **Keep synthesis framing, tighten with examples**: Providing specific
   failure examples ("do not do X like this: [example]") was considered
   but rejected — it would need constant updates as new failure modes
   emerge and does not address the root task framing issue.

3. **Route all sparse movies to a different generator or skip them**:
   A real option, but would reduce coverage for movies that do have
   some text (summaries, overviews). Consolidation framing achieves
   quality for sparse input without requiring a coverage tradeoff.

## Consequences

- Synthesis-branch output quality for sparse input improves
  significantly — no invented character names, no fabricated plot beats,
  output proportional to input richness.
- For movies with rich summary input, output quality is unchanged —
  consolidation produces the same result as synthesis when input is
  dense.
- The "no knowledge" fiction will occasionally cause the model to
  omit accurate details it could have provided from knowledge. This is
  the correct tradeoff: false omission is less harmful to search quality
  than false inclusion (hallucinated details can corrupt semantic search).
- This pattern is transferable to other generation types where sparse
  input is common and hallucination risk is high.

## References

- ADR-033 (plot events cost optimization) — two-branch design context
- `movie_ingestion/metadata_generation/prompts/plot_events.py` — `SYSTEM_PROMPT_SYNTHESIS`
