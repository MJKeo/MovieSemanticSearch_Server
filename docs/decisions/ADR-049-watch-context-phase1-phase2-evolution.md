# [049] — Watch Context: identity_note, evidence_basis, and Eligibility Tightening

## Status
Active

## Context

Watch context generation went through three evaluation phases (Phase 1: 56 movies,
4 candidates; Phase 2: 40 movies, 9 candidates; Phase 3/Round 3: 50 movies,
9 candidates) before locking. Three design decisions emerged from failure pattern
analysis across these phases.

**Phase 1 finding (justification mechanism):** Candidates with a `justification`
field (explaining why terms were generated) scored lower than no-justification
candidates on every rubric axis. Root cause: justifications acted as post-hoc
rationalization — the model decided on terms first, then wrote reasons, rather
than inventorying evidence before generating. This is the opposite of the
mechanism that made justifications valuable in `viewer_experience`.

**Phase 1 finding (genre-only movies):** All 4 candidates scored 1.6–2.5 on
genre-only movies (movies with genre_signatures but no observation fields).
Output was generic, undifferentiated, and potentially inaccurate without
observation grounding.

**Phase 2 finding (viewing_appeal_summary):** A `viewing_appeal_summary` pre-anchor
(20-30 word sentence generated before sections) improved identity accuracy on
challenging movies (+0.32 composite on challenging_identity bucket) but hurt
rich-input movies (-0.49 on gold_standard). Root cause: the detailed summary was
specific enough to act as a template, suppressing secondary signals and creative
phrasings.

## Decision

**1. Rename `justification` → `evidence_basis` with upstream-constraint framing.**
The field description was changed from "explain why you generated these terms" to
"quote or closely paraphrase specific input phrases" with an explicit instruction
to write "No direct evidence" and leave terms empty when no evidence exists. The
prompt variant was reframed as an evidence inventory that constrains generation,
not a post-generation explanation. This is the production schema
(`WatchContextWithJustificationsOutput` with `evidence_basis`); the base schema
(no justifications) is not deployed.

**2. `viewing_appeal_summary` → `identity_note` (2-8 word classification).** The
full-sentence anchor was replaced with a brief classification (e.g., "sincere
family drama", "so-bad-it's-good camp"). Short enough to prime the model's tone
register without providing a template to expand. Schema: `WatchContextWithIdentityNoteOutput`.
Production prompt: `SYSTEM_PROMPT_WITH_IDENTITY_NOTE`.

**3. Require ≥1 observation field for eligibility.** `_check_watch_context()` now
requires `emotional_observations OR craft_observations OR thematic_observations`
in addition to genre data. Affects ~0.7% of pipeline (776 of 109K movies).

## Alternatives Considered

1. **Keep `justification` but rewrite the prompt instruction**: Already tried in
   prior rounds with different wording. The rationalization failure is structural —
   the field position (after term generation in CoT) determines behavior more than
   the instruction text. Moving it to a pre-generation position and renaming it to
   signal its constraining role is a stronger intervention.

2. **Keep `viewing_appeal_summary` at 20-30 words but position it after sections**:
   Rejected. The template-expansion failure is caused by its existence before section
   generation, not its length alone. Replacing it with a 2-8 word classification
   preserves the identity-priming benefit while removing the template effect.

3. **Allow genre-only movies with a sparse-input prompt variant**: Not pursued.
   Phase 1 showed 99.3% of the pipeline has at least one observation field; the
   eligibility tightening is nearly zero-cost in coverage terms and avoids a
   prompt-branching complexity.

## Consequences

- `TermsWithJustificationSection.justification` renamed to `evidence_basis`.
  Any code referencing the old field name will break.
- `WatchContextWithViewingAppealOutput.viewing_appeal_summary` replaced by
  `WatchContextWithIdentityNoteOutput.identity_note`. Old evaluation pipeline
  callers need updating.
- `_check_watch_context()` signature changed: new optional params
  `emotional_observations`, `craft_observations`, `thematic_observations`.
  Existing unit tests use the old 2-param signature.
- Production generator is locked at module level: OpenAI gpt-5-mini,
  reasoning_effort=minimal, WatchContextWithIdentityNoteOutput schema.

## References

- ADR-045 (Wave 2 finalization pattern)
- `movie_ingestion/metadata_generation/generators/watch_context.py`
- `movie_ingestion/metadata_generation/prompts/watch_context.py`
- `movie_ingestion/metadata_generation/schemas.py`
- `movie_ingestion/metadata_generation/pre_consolidation.py`
- `ingestion_data/watch_context_eval_guide.md`
