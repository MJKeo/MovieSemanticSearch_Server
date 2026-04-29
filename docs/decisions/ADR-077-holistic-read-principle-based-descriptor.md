# [077] — Holistic read: principle-based, relationship-mapping descriptor

## Status
Active

## Context
Step 1 of the new query-understanding pipeline (the holistic read)
produces a single prose field that downstream steps use as the ground
truth for structural relationships in the query. Two design questions
arose:

1. **Descriptor framing**: should the field description frame the task
   as "faithfully restate without expanding" (faithfulness-first) or
   "map the structural relationships between wants" (relationship-first)?
   The prior framing led to bland-restatement outputs where the model
   optimized for "don't expand, don't reduce" and skipped relationship
   mapping entirely.

2. **Example-driven vs. principle-based**: should the descriptor use
   target-shape examples to steer output format, or state principles
   (what to capture, what not to do, what ambiguity means)?
   The query space is too varied to template by example — narrow
   examples cause the model to pattern-match to their shapes and
   distort natural language rendering.

## Decision
**Relationship-mapping is the headline; faithfulness is the constraint.**
The field description enumerates four things to surface:
1. Every want in the user's exact phrasing.
2. Modal and polarity markers with their named effect (not just
   the preserved phrase — "ideally" softens commitment, "not" flips
   polarity).
3. Kept-whole units (comparison anchors, compound concepts, figurative
   phrases), called out explicitly.
4. Relationships between wants in plain prose.

An "AMBIGUITY IS INFORMATION" clause was added: when the user uses a
loose term, the read names the term and its role; it does not enumerate
similarity dimensions, decide which axis a negation targets, or gloss
with system-shaped synonyms. Forbidden forms are enumerated (i.e.,
such as, meaning, parentheticals explaining what the user "really" meant).

The system prompt keeps the full conceptual library (atomicity, modifiers,
carver/qualifier, polarity, salience, category taxonomy) loaded even though
Step 1 does not commit to any of those. This is a pragmatic baseline: the
sections will be reused by later steps and keeping them loaded lets prompt
size be measured before deciding what to defer.

## Alternatives Considered
- **Faithfulness-first framing**: "don't expand, don't reduce." Tested
  and rejected because it caused the model to produce bland restatements
  that skipped relationship mapping.
- **Target-shape examples**: Concrete examples of the intended output
  shape for different query types. Rejected because the query space is
  too varied — the model pattern-matched to example shapes instead of
  applying the underlying principles.
- **Structured slot template**: ("topic anchor + qualifying constraints +
  polarity moves + register"). Rejected as fighting the model's natural
  strength at language generation.

## Consequences
- The modal-effect naming requirement ("ideally" → "softens commitment")
  enables Step 5 to set salience/polarity from the read rather than
  re-parsing the original query.
- "Kept-whole unit" is now a stable vocabulary term that leaks into Step
  1's outputs. Downstream steps understand this term — standardize on it
  across pipeline prompts rather than cleaning it up.
- Anchor-pattern hallucination is a known residual failure ("Inception,
  Interstellar, Tenet" → inferred "Christopher Nolan-style"). This is
  worse than ordinary expansion because it adds entities, not dimensions.
  Mitigated partially by Step 2's loose trust posture; a targeted Step 1
  fix is deferred.
- The prompt size baseline (~44K chars with full taxonomy) is the
  reference point before per-step taxonomy deferral is evaluated.

## References
- schemas/step_2.py (Step 1 holistic-read schema field description)
- search_v2/step_2.py (system prompt)
- search_improvement_planning/v3_step_2_rethinking.md
- ADR-076-five-step-query-understanding-pipeline.md
