# [052] — Source of Inspiration: source_material / franchise_lineage Boundary Refinement

## Status
Active

## Context

After a 56-movie evaluation using the ADR-051 schema, boundary errors emerged
on a specific category of films: reboots, reimaginings, spinoffs, and remakes.
The original boundary definition placed franchise_lineage as the field for
"franchise relationships" — a broad concept that some models interpreted as
including reboots (Batman Begins) and reimaginings (21 Jump Street), which share
no linear story continuity with their predecessors but are named franchise
successors.

Two concrete failure modes were observed:
- **21 Jump Street** classified as franchise_lineage ("reimagining") rather
  than source_material — but there is no shared timeline with the TV show.
- **Man of Steel** classified as franchise_lineage ("reboot") rather than
  source_material — but it retells Superman's origin, not continues a timeline.

The root cause: the ADR-051 definitions separated the two fields by named type
("sequel/prequel → lineage; remake/reboot → material") rather than by the
semantic property that makes them different. That semantic property is:
**linear story continuation with shared characters, events, and timeline**.

Additionally, franchise position labels were too coarse. Films can occupy
multiple franchise positions simultaneously (e.g., Creed is both "first in
franchise" for a new sub-franchise AND "sequel" in the Rocky continuity), and
the original closed set of terms (0-2 per field) was too restrictive for
richly-labeled films.

## Decision

**Redefine franchise_lineage as strictly linear story continuation.**
The field is for films that continue a story that already happened in prior
films — same characters, events carried forward, timeline shared. This excludes
reboots, reimaginings, and spinoffs, which *retell* or *branch from* existing
media rather than continuing it.

**Redefine source_material to include all retelling/branching relationships.**
Remakes, reboots, reimaginings, and spinoffs are now source_material entries,
alongside traditional adaptations. Both categories describe what existing media
the film *draws from*, not where it sits temporally.

**Expand franchise position vocabulary and encourage redundancy.** Added
explicit labels: "franchise starter" (intentional), "first in franchise"
(retrospective), trilogy-specific (first in trilogy, second in trilogy,
trilogy finale), "series entry", "series finale". Redundancy is encouraged
for richer semantic coverage (e.g., "sequel, second in trilogy").

**Open vocabulary.** Terms in both fields are guidance, not a closed enum.
Models may phrase naturally as long as the meaning aligns with the field's
semantic definition.

**Raise max terms from 0-2 to 0-3** to accommodate redundant descriptive labels.

**"Later spawned sequels" ≠ automatic lineage.** Retrospective classification
("first in franchise") is valid when confident follow-ups exist; speculation
about potential future franchises is not.

## Alternatives Considered

1. **Keep the closed enum of source types**: Rejected. The evaluation showed
   the semantic distinction (retelling vs. continuation) matters more than the
   label category. An open vocabulary with clear semantic rules is more robust
   than a closed set that models must memorize.

2. **Add "reboot" and "reimagining" as valid franchise_lineage terms**: Rejected.
   This would keep both fields valid for the same films, causing ambiguity rather
   than resolving it. The boundary must be exclusive.

3. **Use a single field for all source relationships**: Rejected. Franchise
   position (sequel in a timeline) and source adaptation (film drew from X) are
   distinct retrieval signals that serve different queries.

## Consequences

- Schema field descriptions updated for both `source_material` and
  `franchise_lineage` in `SourceOfInspirationOutput` and
  `SourceOfInspirationWithReasoningOutput`. Tests checking description strings
  will break.
- Evaluation data scored against the ADR-051 definitions may score differently
  under the new boundary; the rubric's ground truth for boundary-case movies was
  updated in the eval guide accordingly.
- The new eval guide adds H6 (franchise position specificity) and H7 (source
  vs lineage boundary) hypotheses, and two new eval buckets covering these cases.

## References

- ADR-051 (schema redesign — predecessor)
- ADR-045 (Wave 2 finalization pattern)
- `movie_ingestion/metadata_generation/schemas.py`
- `movie_ingestion/metadata_generation/prompts/source_of_inspiration.py`
- `ingestion_data/source_of_inspiration_eval_guide.md`
- `ingestion_data/source_of_inspiration_eval_buckets.json`
