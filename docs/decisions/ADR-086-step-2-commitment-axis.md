# [086] — Step 2: 5-level commitment axis replacing role+salience

## Status
Active

## Context
The previous design encoded trait importance via a role enum (primary/secondary/
contextual) combined with a salience float. This two-axis encoding was redundant:
role and salience largely conveyed the same signal and the LLM consistently struggled
to assign them independently. Stage 4 used both values but the combination rules were
ad-hoc. The core need is a single dimension: how strongly does the user require this
trait to be present?

## Decision
Replace role+salience with a single `commitment` field on each trait, a 5-level
ordered enum:
- `required` — trait must be present; absence strongly penalizes.
- `elevated` — trait is important but not a hard requirement.
- `neutral` — trait is a mild preference or default.
- `supporting` — trait provides context but low weight in scoring.
- `diminished` — trait is mentioned but the user is ambivalent or it modifies negatively.

Stage 4 maps commitment levels to scoring weights. The mapping is monotonic:
`required` > `elevated` > `neutral` > `supporting` > `diminished`.

## Alternatives Considered
- **Keep role+salience with clearer LLM instructions**: Tested; the redundancy caused
  the LLM to anchor on one axis and produce arbitrary values for the other.
- **Continuous 0–1 importance score**: More expressive but harder to produce
  consistently from LLM output and harder to explain to a developer debugging a score.
- **Binary important/not-important**: Too coarse; loses the "required vs. preferred"
  distinction that strongly affects result quality.

## Consequences
- Single axis is easier for the LLM to assign consistently.
- Stage 4 weight mapping is a simple lookup table rather than a multi-axis formula.
- 5 levels is a deliberate choice: fewer levels loses the `required` vs. `elevated`
  distinction; more levels exceed what the LLM can reliably distinguish.

## References
- docs/modules/search_v2.md — Step 2 section
- search_v2/step_2.py, search_v2/step_3.py
- ADR-084: V4 trait relationship typology (companion redesign)
