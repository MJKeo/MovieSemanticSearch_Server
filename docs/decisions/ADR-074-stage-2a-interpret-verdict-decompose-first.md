# [074] — Stage 2A: `interpret` verdict and decompose-first reasoning flow

## Status
Active

## Context
Empirical probing of Stage 2A against queries like "Mindless action",
"Tarantino boomers love", "Main character energy", "Indiana Jones runs
from the boulder", and "Popcorn movies" surfaced four prompt-content
failure modes that the schema could not fix:
1. Cross-family fusion: semantic + keyword collapsed into one slot
2. Hallucinated capabilities: invented "demographic popularity metrics"
   against the metadata family
3. Idiom under-expansion: slang passed through as a single literal unit
4. Compound `best_guess` strings: one interpretation mashing multiple
   retrieval families into a single value

## Decision
Rewrite Stage 2A prompt content (no schema changes):

**`interpret` verdict replaces `best_guess`.** `interpret` emits 1+
retrievable atoms, each tagged with exactly one family (literal /
metadata / keyword / semantic). Cardinality reveals itself: one atom for
a single-concept phrase, multiple atoms for a phrase that genuinely spans
families. Compound-string failures are impossible by construction.

**Decompose-first-then-group reasoning flow.** Pass 1 commits per-phrase
verdicts and emits atoms into inventory. Pass 2 operates on atoms and
decides fuse-vs-split. This forces decomposition before grouping.

**New fusion criterion: same-family AND same sub-dimension AND ranking-style.**
Two atoms fuse only when (a) same retrieval family, (b) same
sub-dimension within that family (one metadata attribute, one keyword
category), AND (c) they jointly define a ranking gradient where "more of
both = better match." Cross-family atoms never fuse. Semantic is a
single fusion unit (its spaces can be queried together in one slot).

**Endpoint descriptions rewritten in user-facing capability language.**
Per-attribute bullets for metadata (10 attributes, each with explicit
limits — popularity/reception marked global-only). Per-category bullets
for keyword (closed taxonomy). Per-space bullets for semantic (8
spaces). Each family explicitly names what it CANNOT do.

**Boundary examples illustrate principles, not test queries.** Seven
worked examples cover descriptive-stays-literal, multi-atom interpret
decomposition, cross-family split, same-family ranking fusion,
same-family independent-filter split, fold-into, and evaluative breadth
preservation. None come from probed queries.

## Alternatives Considered
- **Schema change (new field for atoms)**: Would make family-tagging
  structural. Rejected: the schema already handles one-or-many via the
  existing list field; adding a field per atom type would fragment the
  reasoning trace.
- **Per-phrase schema validation**: Pydantic validators on the
  `unit_analysis` free-form field. Rejected per user direction — the
  discipline lives in the free-form trace, not mechanical validators.
- **Single fusion rule (same-family only)**: Observed that metadata has
  10 independent attributes — two metadata atoms targeting different
  attributes (popularity vs. runtime) must not fuse even if both are
  ranking-style. Sub-dimension requirement closes this gap.

## Consequences
- All four prior failure modes resolved on the 6-query probe set.
- Stage 2A system prompt grew to ~21KB (three branch-dynamic variants).
- `stage_2::run_stage_2` remains the broken caller; unit test
  `test_search_v2_stage_2.py` still fails pending the Step 2B rework
  (this ADR covers 2A prompt content only).

## References
- search_improvement_planning/steps_1_2_improving.md
- search_v2/stage_2a.py
- docs/decisions/ADR-075 (creative_alternatives field)
