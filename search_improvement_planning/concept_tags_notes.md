# Concept Tags — Prompt Improvement Notes

Notes from prompt evaluation session (2026-04-09). These inform
implementation when building/revising the concept tags prompt and
surrounding code.

---

## Schema Changes

### Split ConceptTag into per-category enums

Replace the single `ConceptTag` enum with 7 per-category enums
(`NarrativeStructureTag`, `PlotArchetypeTag`, `SettingTag`,
`CharacterTag`, `EndingTag`, `ExperientialTag`, `ContentFlagTag`).
Each `TagEvidence` variant uses its category's enum. Derive a
combined constant (e.g., `ALL_CONCEPT_TAGS`) for codebase consumers
(GIN index population, query routing, etc.).

This makes the JSON schema self-enforcing — the model can't produce
a tag in the wrong category field. Eliminates the `model_validator`
on `ConceptTagsOutput` and reduces prompt cognitive load.

Files affected: `schemas/enums.py`, `schemas/metadata.py`,
`generators/concept_tags.py`, prompt module, downstream consumers.

---

## Post-Generation Fixups

### TWIST_VILLAIN implies PLOT_TWIST

After parsing `ConceptTagsOutput`, if `narrative_structure` contains
TWIST_VILLAIN but not PLOT_TWIST, programmatically insert PLOT_TWIST.
TWIST_VILLAIN is definitionally a subset — handling in code is
deterministic and removes cognitive load from the LLM.

Apply in both the live generator return path and the batch result
processor path.

The prompt should define TWIST_VILLAIN independently (no need to
explain the PLOT_TWIST relationship to the model).

---

## Prompt Changes

### Reframe parametric knowledge as high-confidence fallback

Current framing ("tiebreaker") is ambiguous. Replace with:

- Use only when near-certain (95%+ confidence) based on the film's
  cultural identity (e.g., Groundhog Day = TIME_LOOP)
- This is a fallback for when input data happens to lack evidence
  for something the model knows to be true
- If input evidence contradicts parametric knowledge, trust the
  input — even if the input could technically be wrong
- For keyword-only inputs (see below), be extra strict — movies
  reaching that path are almost certainly on the obscure side

### Remove "2-6 tags total" anchoring

Remove from both preamble and output section. We're not targeting a
count — classification is purely evidence-based. Keep "empty
categories are correct and common" as sufficient calibration.

### Restructure information ordering

Proposed order:
1. Task definition (2 sentences)
2. Inputs section
3. Tag definitions (with "Consider each category" sweep instruction)
4. Evidence discipline (recency advantage — fresher when generating)
5. Output format

Rationale: minimal-reasoning model benefits from evidence rules being
closer to the generation start. Worth A/B testing if evaluation shows
the model being too liberal — "empty by default" first might help.

---

## Programmatic Prompt Adaptation: Keyword-Only Inputs

### Population

~20.7K movies (19% of eligible) qualify via path 3 only: no
plot_events, best_plot_fallback < 250 chars, but plot_keywords >= 3.

These movies have:
- Keywords: typically 6-15, mix of signal-rich and noise
- Short overviews: avg ~170 chars (1-2 sentences of plot setup)
- No Wave 1/2 outputs at all

### Adapted prompt design

Use a separate, shorter system prompt when the keyword-only condition
is detected (no plot_summary, no plot_text, no emotional_observations,
no NT terms, no PA fields).

Key differences from the full prompt:
- Drop input descriptions for absent fields — don't list 5 fields
  as "not available"
- Reframe the task: classify based on keyword matching + overview
  narrative signal
- Keep all 23 tags available — don't exclude tags or frame any as
  "basically impossible." Instead, maintain a uniform evidence
  standard: each tag requires clear evidence from the input or
  high-confidence parametric knowledge. No inferences that are
  "probably true." (e.g., tearjerker requires concrete evidence of
  audience crying, not just a sad topic — since that evidence won't
  be in keywords/overview, it naturally won't get generated)
- Evidence discipline: two primary levels (direct keyword match,
  concrete inference from overview). Parametric knowledge allowed
  but extra strict given the obscurity of this population.

User prompt passes just: `title`, `plot_keywords`, `overview`.

Detection: in `build_concept_tags_user_prompt()`, check whether
plot_summary is None, plot_text fallback is None/short, and all
Wave 1/2 outputs are None. Select `SYSTEM_PROMPT_KEYWORD_ONLY`.

---

## Testing Notes

### FEEL_GOOD and TEARJERKER boundary cases

Specifically test these during evaluation. The strict "audience
response evidence, not inferred from plot" standard is correct for
precision but needs validation that it doesn't cause excessive false
negatives on clear cases. Current prompt text is fine — just needs
empirical verification.

### Mutually exclusive tag co-occurrence

Programmatically check generation results for pairs that shouldn't
co-occur (HAPPY_ENDING + SAD_ENDING being the primary case). Do not
add prompt instructions for this — test first, adjust only if data
shows a real problem.

### Evidence discipline ordering A/B test

If evaluation shows the model being too liberal with tagging,
experiment with placing "empty by default" evidence discipline before
tag definitions instead of after. Current proposal puts it after for
recency advantage but this is a judgment call worth testing.
