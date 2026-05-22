# [093] — concept_tags: upstream label contamination removal

## Status
Active

## Context
The reasoning-field eval (concept_tags_results/reasoning_fields) made
the contamination chain unambiguous: every BITTERSWEET over-tag (7+ movies)
was driven by the upstream `ending_aftertaste` field literally containing
the word "bittersweet". Every ANTI_HERO false positive on Catch Me / Deadpool
and every NONLINEAR_TIMELINE false positive on Deadpool were driven by
`narrative_technique_terms` subsections emitting tag-shaped labels
("antihero maturation arc", "sympathetic antihero", "intercut flashback
structure"). The consumer's per-category reasoning showed it was operating
as a faithful label-transcriber — the schema description's anti-pattern
guidance could not override an upstream signal that literally named the tag.

## Decision
Remove three contaminated inputs from the concept_tags consumer:
1. **`ending_aftertaste` removed** (Tier 1). Replaced as the PRIMARY ending
   signal by a two-stage process: (1) literal closing-scene detection from
   `plot_summary` (celebration beat → HAPPY; loss beat → SAD; quiet beat →
   BITTERSWEET candidate; existential → NO_CLEAR candidate); (2)
   `emotional_observations` filtered for end-state language as support.
2. **NT `character_arcs.terms` and `audience_character_perception.terms`
   removed** (Tier 2). CharacterAssessment now derives ANTI_HERO from
   `plot_summary` raw protagonist behavior + `character_arc_labels` from
   PlotAnalysis (thematic arc transformations) + `conflict_type`. ANTI_HERO
   `selection_criteria` explicitly forbids deriving from upstream-labeled terms.
3. **Remaining NT sections (5 of 9) kept with a vocabulary-overlap warning**
   in the prompt: `narrative_delivery` and `information_control` are tag-adjacent
   but had genuine positive use cases in the eval.

## Alternatives Considered
- **Prompt-side tightening only**: tried across multiple iterations; failed
  because an upstream signal that literally names the tag cannot be overridden
  by prompt instructions. The model was writing correct reasoning against
  its conclusion but still emitting the upstream label as the tag.
- **Regenerate upstream ViewerExperience / NarrativeTechniques generators**:
  rejected by user — regenerating ~100K movies just to change ending_aftertaste
  is expensive and slow. Removal and rederive is faster and cleaner.

## Consequences
- Ending classification now depends entirely on plot_summary closing-scene
  detection and emotional_observations. Risk: if plot_summary ends mid-plot
  or is ambiguous, more endings may fall to NO_CLEAR_CHOICE. The HAPPY-as-
  default base-rate rule mitigates.
- NT `character_arcs.terms` removal may lose some genuine ANTI_HERO positive
  signal for movies where the arc label is correct but plot_summary evidence
  is thin. Accepted as better than the false-positive rate from tag-name leakage.
- `generate_concept_tags` signature no longer carries `ve_output`.
  `load_viewer_experience_output` in `inputs.py` is kept for future consumers.

## References
- `movie_ingestion/metadata_generation/generators/concept_tags.py`
- `movie_ingestion/metadata_generation/prompts/concept_tags.py`
- `movie_ingestion/metadata_generation/inputs.py`
- `schemas/metadata.py` (CharacterAssessment.reasoning description)
- `schemas/enums.py` (ANTI_HERO selection_criteria)
