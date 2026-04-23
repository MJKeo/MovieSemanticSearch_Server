# Stages 1-2 Performance Improvement

Working analysis of non-deterministic behavior and quality failures in the
current Stage 1 + Stage 2A + Stage 2B pipeline. This document captures
what we observed empirically, evidence-grounded theories for why it's
happening, and the design directions that emerged from the discussion.
Solutions are not prescribed here — they are directions worth testing.

Companion to `steps_1_2_improving.md`, which documented the prior
design iterations that brought us to the current state.

## Methodology

- **Query set**: 18 real feedback queries spanning vibe searches, concrete
  entity/franchise searches, description-based identification, and
  ambiguous evaluative-word queries. See `debug_feedback_queries.py`.
- **Runs per query**: 3 independent LLM calls per stage per query,
  giving us a variance signal per pattern rather than a single-run
  snapshot.
- **Models**: Gemini 3 Flash across every stage, `thinking_budget=0`,
  temperature unset (Gemini 3 default is 1.0).
- **Full data**: `feedback_queries_debug_output.json` (all stages, all
  runs, structured). Compact diff table at `/tmp/variability.txt`.
- **Prompt sizes measured** at call time:
  - Stage 1: ~6,358 tokens of system prompt
  - Stage 2A (primary branch): ~5,249 tokens
  - Stage 2B: ~3,499 tokens

## Observed Failure Patterns

Patterns ranked roughly by frequency and downstream impact. Every
pattern is reproducible across runs; most have a variance component
(they fire in some runs but not others for the same query).

### P1. Same atom routes to a different retrieval family each run
Most frequent pattern. Examples seen:
- "England" → metadata.country_of_origin / keyword.setting / semantic.plot_events
  (three families, three runs)
- "lone female protagonist" → semantic.plot_analysis / keyword.character_type
- "Classic" → metadata.reception / metadata.popularity / both
- "soulful" → semantic.viewer_experience / semantic.plot_analysis
- "Mindless" → three different semantic sub-spaces

### P2. 2A→2B family override
2B silently reroutes the family 2A picked. Stable for real capability
mismatches (`Classic` → always rerouted from metadata to semantic), but
also fires at temperature-driven random points (a 1940s release_date
2A-committed slot getting executed by 2B as semantic/inclusion).

### P3. S2A slot_count varies across runs for the same query
Structural partition instability. Examples: Popcorn 2/3/4 slots across
three runs; Moody neo-noir 3/4/3; Spider-Man 1/2/1; Christmas-happy-cry
2/3/3.

### P4. Duplicate actions emitted in 2B
Same slot emits two or more identical RetrievalAction entries. Fires on
roughly half the queries in at least one run. Strongly correlates with
the `expand:paraphrase_redundancy` verdict.

### P5. 2A fabricates content not in the rewrite
Atoms appear in `inventory` that were never in the rewrite. Near-
universal for abstract/vibe queries. Label varies run-to-run but the
behavior reproduces. Examples: `memorable cinematic experience`,
`positive audience reception`, `family-oriented themes`,
`exceptionally high reception`, `emotionally resonant`.

### P6. S1 "or" hedging in primary_rewrite
Banned by the prompt but fires in 3/3 runs on "Artful dramas" and 2/3
on "hit you in the gut". Hedging connectives fuse two retrieval
targets into one rewrite string where branching would be the correct
response.

### P7. S1 rewrite enrichment vs preservation is coin-flip
Idioms ("popcorn", "gut", "mainline", "mindless") are decomposed in
some runs and preserved in others. No stable policy. Same query,
different rewrite discipline each run.

### P8. S1 alternative_intents drop randomly
"Soul" title alt for Soulful (emitted 2/3), Spider-Man title alt
(2/3), Mainline HP different treatments. The `ambiguity_analysis`
trace commits to emitting an alt, then the structured list sometimes
doesn't include it.

### P9. Transient Pydantic validator failures
The `scope ⊆ inventory` partition-completeness check hard-failed on
"Intimate, slow-moving character studies" in single-run testing and
did not reproduce in 3 fresh runs. Rate is non-zero but ≤1/4 on that
query.

### P10. Non-verbatim atom citations
S2A injects words into cited phrases (e.g., "rolling" appears in the
Indiana Jones scene citation but wasn't in the original user text).
Fidelity drift on phrase copies.

### Cross-cutting observation
Variance is **concentrated on linguistically vague inputs**. Concrete
tokens (actor names, decades, franchise names) route stably across
runs. Abstract tokens ("classic", "popcorn", "soulful", "something to
say") produce a different interpretation each run.

## Root-Cause Theories

Organized by layer. Each theory names the evidence and maps to the
patterns it most likely explains.

### A. Structural / configuration

**A1. Temperature = 1.0, no seed.** Gemini handler in
`generic_methods.py` never sets a temperature; Gemini 3's default is
1.0. Even at temperature 0, Gemini is not bit-deterministic due to
kernel non-determinism, MoE routing, and backend batching.
→ Dominant driver of P1, P2, P7, P8.

**A2. `thinking_budget=0` removes the model's scratchpad.** All three
stages run thinking-disabled. Small non-reasoning models rely on the
prompt's field ordering and free-form trace fields to "simulate" a
scratchpad — which is a probabilistic prior, not a constraint.

**A3. System prompts are well past the "lost-in-the-middle"
degradation threshold.** Research places onset of instruction-
following degradation around ~3,000 tokens for small models, with
U-shaped recall. All three stages exceed this, and the rules that
fail most often (Stage 1's ban-on-"or", Stage 2A's "preserve
evaluative breadth") sit mid-prompt where attention is thinnest.
→ Explains P6 and contributes to P5.

### B. Schema architecture

**B1. Reasoning-first scaffold is a prior, not a constraint.**
`ambiguity_analysis`, `creative_spin_analysis`, `unit_analysis`,
`slot_analysis`, `atom_analysis` are all `constr(min_length=1)` —
i.e., any non-empty string. Structured-CoT helps on average but is
well-documented to be **unfaithful** ("CoT mismatch"): the trace says
one thing, the decision fields emit another.
→ Canonical name for P8 (trace says "emit as alt" but list has 0) and
P2 (2A commits a family in trace, 2B picks another).

**B2. Partition-completeness is the only structural check; everything
else is advisory.** The Pydantic validator enforces `scope ⊆
inventory` and vice versa. No schema constraint protects against
duplicate actions (P4), family drift (P1), or fabricated atoms (P5).
→ Explains P9 (strict set-equality gate fires when verbatim discipline
slips).

**B3. `query_traits` and every trace field are free-form strings.**
The prompt asks for `traits: <trait1>, <trait2>, ...` format but the
schema accepts any string ≥1 char. Format drift across runs is
schema-valid and therefore invisible downstream.

**B4. Verbatim-citation discipline has no enforcement hook.** The
"quote verbatim" rule has no code check that cited phrases actually
appear in the rewrite. Citation fidelity on small models without
external verification is known to be poor.
→ Explains P10 and contributes to P9.

### C. Stage-boundary handoff

**C1. 2B is told to both honor and challenge 2A with no tiebreaker.**
Prompt literally says "neither deference nor rerouting is virtuous —
capability match is the only criterion." At temperature 1.0 the model
resolves this differently each run.
→ Primary driver of P2.

**C2. 2B runs one LLM call per slot in isolation.** No visibility into
sibling slots' committed actions. Spider-Man and Shrek get different
defensive-retrieval decisions because they're independent stochastic
draws — consistency across sibling calls is architecturally impossible
under the current shape.

**C3. 2A's `interpret` verdict invites invention without a max-
expansion anchor.** `interpret → 1+ atoms`, cardinality reveals itself
from the phrase. For abstract queries the phrase under-constrains
cardinality and the model samples freely.
→ Primary mechanism behind P5.

### E. Pipeline-level

**E1. The system trusts the model to do arithmetic on its own text.**
Matching reading-verdict count to `alternative_intents` length,
matching inventory to scope, keeping traits distinct from rewrite
content — these are bookkeeping tasks asked of the model in prose.
Only Pydantic-enforced bookkeeping actually works; the rest is
aspirational.

**E2. Evidence-grounding asymmetry** (described above under cross-
cutting observation). Variance is concentrated on vague inputs,
exactly as faithfulness-hallucination research predicts.

**E3. Branch-dynamic dispatch exists for 2A but not 2B.** 2A assembles
per-branch sections so the model sees only relevant instructions. 2B
keeps every capability bullet in every call — paying the long-prompt
tax on every per-slot call while 2A is not.

## Design Directions Emerging From Discussion

These are directions worth testing, not finalized solutions. Each
entry carries the relevant patterns/theories it addresses and the
tradeoff we identified.

### D1. Lower Gemini temperature (target 0.5)
Google's warning against low temperatures applies mainly to Pro /
reasoning models. Small-model structured extraction conventionally
runs at 0.2–0.5. Won't eliminate variance (Gemini isn't bit-
deterministic at any temperature), but expected to meaningfully
collapse family-routing flips. If spins feel robotic at 0.5, consider
a higher-temp separate call for spin generation only.
Addresses: A1 → P1, P2, P7, P8.

### D2. Selective thinking on Stage 2A only
Enabling thinking pipeline-wide is user-hostile (~3-10x latency per
stage; end-to-end from ~6-9s to 20-40s). Stage 2A is the highest-
leverage stage for faithfulness; Stage 1 (routing + rewrites) and 2B
(per-slot planning) tolerate small-model noise better. Enable thinking
only on 2A if variance doesn't collapse enough after D1.
Addresses: A2, B4 → P5, P9, P10.

### D3. Prompt distillation to < 3k tokens per stage
Realistic target: 30-50% reduction without quality loss. Main levers:
collapse "Your job is / Your job is NOT" duals to one direction, merge
redundant rule restatements, tighten hedging phrases to specific
counts, remove meta-commentary, compress boundary examples to one-line
form.
Addresses: A3 → P6 and general rule adherence.

### D4. Structural schema upgrades on format-drift fields
Small models benefit disproportionately from schema enforcement.
Candidates:
- `query_traits` as `list[constr(min_length=1)]` instead of CSV string
- Reading verdicts as structured `Reading` objects with a `Literal`
  verdict enum, instead of prose
- Similar treatment for spin verdicts and slot_analysis verdicts

Keep reasoning/cohesion fields as prose — freedom is productive there.
Addresses: B1, B3 → P8.

### D5. Retry-with-error-injection on both 2A and 2B
Current 2B retry re-runs the same prompt. Research shows blind retry
re-samples the same failure mode; feedback retry meaningfully
diversifies. Inject the Pydantic ValidationError into the retry user
prompt. Add a retry wrapper to 2A (none exists today).
Addresses: B2 → P9.

### D6. Rephrase 2B's honor-vs-reroute directive
Current "neither is virtuous" gives no default. Better: "Default to
honoring 2A's advisory `retrieval_shape`. Only reroute when you can
name the specific capability the advised route lacks." Makes
rerouting opt-in with explicit burden of proof.
Addresses: C1 → P2.

### D7. 2A does zero consolidation; programmatic semantic merge post-2B
Drop the fuse-vs-split verdict from 2A entirely. Over-split slots are
downstream-safe because:
- Semantic is the only family where merging changes execution (joint
  vector querying). Those get merged in code after 2B.
- Keyword / metadata / entity / studio / awards / franchise atoms
  execute independently either way — merged or split is identical.
- Stage 4's MAX-within / additive-across math stays honest as long as
  the programmatic merger assigns a concept_id that Stage 4 groups by.

Removes the dominant variance source in P3.
Addresses: C3 → P3.

### D8. Per-trait parallel 2A calls
Replace single 2A call with N parallel calls, one per `query_trait`.
Each call: "given ONE trait, decide (a) family, (b) internal split
cardinality, (c) interpret-if-needed." Dramatically shorter per-call
prompt; focus effect keeps attention high; parallel → same wall-clock
latency as one call.

Dependency: requires Stage 1 trait extraction to emit a structured
`list[Trait]` with `raw_phrase` + `normalized_label` fields (D4 for
query_traits is a prerequisite).

Compound-trait edge case ("wartime love", "killer cinematography"):
decision moves upstream to Stage 1 trait extraction, where it's
structured and checkable, rather than hidden in 2A's fuse heuristic.

Addresses: E1, A3 → P1, P3, P5.

### D9. Code-heuristic narrowing of 2B capability bullets
Tension: preserve rerouting autonomy (can't hard-limit families), but
cut prompt length. Middle ground: a code heuristic picks the advised
family + 2 likely alternates based on `retrieval_shape` keywords,
drops the other 5. Preserves most rerouting while cutting ~40% of
prompt length.

Only pursue if D3 alone doesn't get 2B under ~3k tokens.
Addresses: E3, A3.

## Agreed-On Product Tolerance

For this browsing system, Stage 1 producing a reasonable-but-
particular interpretation of a vague query is **acceptable**. The
product already offers alternatives + spins per query and supports
follow-up clarifications. This means the "rewrite enrichment coin-
flip" (P7) is a lower-priority failure than it would be for a
command-and-control interface.

Exception: interpretations that add unsupported filters (e.g.,
inventing "blockbuster films" for "popcorn movies" as a hard
inclusion filter) are still failures because they corrupt retrieval
shape, not just display.

## Open Questions

- What eval signal do we use to confirm a change helps vs just
  reshapes variance? The 3-run-per-query probe is a starting point
  but needs a concrete metric (pattern-hit rate? human-rated quality
  on top-N?).
- If D7 (2A no consolidation) lands, do we still need 2A at all, or
  does it collapse into the per-trait planning step described in D8?
- How does thinking-on + temperature-down interact? Worth measuring
  both jointly and separately before enabling either broadly.
