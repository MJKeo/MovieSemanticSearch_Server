# [051] — Source of Inspiration: Schema Redesign — production_mediums Removed, franchise_lineage Added

## Status
Active

## Context

After a 55-movie, 6-candidate evaluation of source_of_inspiration generation,
three design problems emerged in the original schema:

1. **`production_mediums` was unreliable.** The gpt-4o-nano-medium-just candidate
   returned empty lists 25% of the time (14/55 movies). Root cause: "animation"
   and "live action" are derivable deterministically — 100% of Animation-genre
   movies have medium-indicating keywords in IMDB data. LLM generation adds cost
   and failure modes without providing signal the input data already contains.

2. **`sources_of_inspiration` was too broad.** The open-ended prompt admitted
   vague terms like "inspired by historical events" for fiction films with
   realistic themes (e.g., Gladiator). Franchise relationships (sequel/prequel/
   reboot) were also landing here, producing mixed bags rather than clean
   source-type classification.

3. **Franchise position was fragmented.** ~2,060 movies had franchise keywords
   and ~2,888 had `source_material_hint` mentions, with partial overlap and
   inconsistent labeling. Neither input source alone was reliable — keywords miss
   retrospective labels ("franchise starter") and hints are review-extracted so
   coverage is incomplete.

## Decision

**Remove `production_mediums` from LLM generation.** Production medium will be
derived deterministically at embedding time: Animation genre → use specific IMDB
medium keywords; no Animation → "live action". This eliminates the empty-list
abstention failure mode entirely.

**Rename `sources_of_inspiration` → `source_material` and narrow the valid set.**
Closed categories: adaptations (novel, true story, manga, comic, play, stage
musical, video game, real events, autobiography, short story) and
retellings/branches (remake, reboot, reimagining, spinoff). Franchise continuation
(sequel, prequel) moves to the new `franchise_lineage` field. Vague thematic
analogies are forbidden.

**Add `franchise_lineage` field.** LLM serves as consolidation and gap-filling
layer drawing on both merged_keywords and source_material_hint plus parametric
knowledge from title cues (e.g., "Part 2", "Returns"). Valid labels: sequel,
prequel, trilogy position (first/middle/finale), franchise starter, first in
franchise, series entry, series finale.

**Redesign reasoning fields to use non-gating evidence inventory pattern.**
Evidence fields (`source_evidence`, `lineage_evidence`) are placed before their
respective output lists and described as records of what was considered, NOT as
gates. "No direct evidence" does not mandate an empty list when parametric
knowledge provides 95%+ confidence. This addresses the anchoring-to-abstention
failure seen in the prior reasoning variant (ADR-050).

## Alternatives Considered

1. **Keep `production_mediums` but improve the prompt**: The failure mode is
   structural — LLMs abstain when evidence is weak even with good prompts. Since
   the information is fully derivable from existing data, the correct fix is to
   remove the LLM from this path entirely.

2. **Merge `franchise_lineage` into `source_material`**: Rejected. Franchise
   position (sequential story continuation) and source adaptation (what a film
   draws from) are distinct semantic concepts that serve different retrieval queries.
   Keeping them separate allows "based on novel, sequel" to be correctly split
   rather than emitted as a mixed label.

3. **Keep open vocabulary for `source_material`**: Rejected. Evaluation showed
   open vocabulary causes loose inference. A guided closed set with a catch-all
   backstop is the correct design for a leaf-node classifier.

## Consequences

- Schema field names changed: `sources_of_inspiration` → `source_material`,
  `production_mediums` removed, `franchise_lineage` added. All unit tests and
  evaluation data referencing old field names are incompatible.
- `__str__()` output format preserved (comma-joined lowercase terms), but
  input fields changed.
- The existing TODO evaluating a merge of `production_keywords` and
  `source_of_inspiration` is superseded — generators now have clean,
  non-overlapping responsibilities.
- Existing evaluation data from prior runs is incompatible with the new schema;
  a fresh evaluation run is required.

## References

- ADR-050 (evidence-inventory prompt pattern — predecessor)
- ADR-025 (schema design principles)
- ADR-045 (Wave 2 finalization pattern)
- `movie_ingestion/metadata_generation/schemas.py`
- `movie_ingestion/metadata_generation/prompts/source_of_inspiration.py`
- `movie_ingestion/metadata_generation/generators/source_of_inspiration.py`
