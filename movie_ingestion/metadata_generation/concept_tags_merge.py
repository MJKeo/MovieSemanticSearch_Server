"""
Majority-vote merge for multiple ConceptTagsOutput runs.

Concept tags are generated multiple independent times per movie (currently
3 runs via gpt-5-mini). The union of these runs improves recall and the
majority-vote consumer of those runs improves precision: a tag joins the
final result iff a strict majority of runs include it.

This module owns the single source of truth for that merge rule. It was
originally inlined in `run_concept_tags_generation.py`; it now lives here
so the eval script and the production backfill
(`movie_ingestion/backfill/backfill_concept_tag_ids.py`) share one
implementation and cannot drift apart.
"""

from __future__ import annotations

from collections import Counter

from schemas.enums import ConceptTagCategory
from schemas.metadata import (
    CharacterAssessment,
    ConceptTagsOutput,
    ContentFlagAssessment,
    EndingAssessment,
    ExperientialAssessment,
    NarrativeStructureAssessment,
    PlotArchetypeAssessment,
    SettingAssessment,
)

# (field_name, AssessmentCls) pairs for the list-typed categories.
# Derived from ConceptTagCategory (the single source of truth for which
# categories exist and which are multi-label vs one-of) — adding a new
# multi-label category to that enum auto-propagates here. Endings is the
# only one-of category and is excluded from this list; majority_merge()
# handles it separately as a mode vote on a single tag value.
_ASSESSMENT_BY_FIELD: dict[str, type] = {
    "narrative_structure": NarrativeStructureAssessment,
    "plot_archetypes":     PlotArchetypeAssessment,
    "settings":            SettingAssessment,
    "characters":          CharacterAssessment,
    "experiential":        ExperientialAssessment,
    "content_flags":       ContentFlagAssessment,
}
LIST_CATEGORIES: list[tuple[str, type]] = [
    (cat.field_name, _ASSESSMENT_BY_FIELD[cat.field_name])
    for cat in ConceptTagCategory
    if cat.cardinality == "multi"
]


def majority_merge(outputs: list[ConceptTagsOutput]) -> ConceptTagsOutput:
    """Merge N ConceptTagsOutputs via majority rules.

    For each list-typed category, a tag joins the merged set iff at least
    ceil(N/2 + epsilon) runs include it — i.e. a strict majority. For N=3
    this is 2-of-3 (matches the user spec: include when 2 or 3 have it,
    exclude when 2 or 3 do not).

    For the single-value endings category, the merged tag is the mode of
    the N votes; on a tie, the first run's vote wins (deterministic).
    """
    if not outputs:
        raise ValueError("majority_merge requires at least one output")

    threshold = (len(outputs) // 2) + 1  # strict majority: 2 of 3, 2 of 2, 3 of 5

    merged_kwargs: dict = {}

    # Helper: concatenate the per-run reasoning strings for a given
    # category into a single audit-trail string. The merged assessment
    # is a programmatic synthesis (majority vote), so its reasoning is
    # the joined reasoning of the underlying runs — useful for debugging
    # disagreements and visible in saved JSON.
    def _join_reasoning(field_name: str) -> str:
        parts = [
            f"[Run {i + 1}] {getattr(out, field_name).reasoning}"
            for i, out in enumerate(outputs)
        ]
        return "\n\n".join(parts)

    # List-typed categories
    for field_name, assessment_cls in LIST_CATEGORIES:
        counter: Counter = Counter()
        for out in outputs:
            for tag in getattr(out, field_name).tags:
                counter[tag] += 1
        majority_tags = [tag for tag, count in counter.items() if count >= threshold]
        # Sort by concept_tag_id for stable deterministic output
        majority_tags.sort(key=lambda t: t.concept_tag_id)
        merged_kwargs[field_name] = assessment_cls(
            reasoning=_join_reasoning(field_name),
            tags=majority_tags,
        )

    # Endings — mode vote, first-run tiebreaker
    ending_counter = Counter(out.endings.tag for out in outputs)
    max_count = max(ending_counter.values())
    top_candidates = [tag for tag, count in ending_counter.items() if count == max_count]
    if len(top_candidates) == 1:
        chosen_ending = top_candidates[0]
    else:
        first_run_ending = outputs[0].endings.tag
        chosen_ending = first_run_ending if first_run_ending in top_candidates else top_candidates[0]
    merged_kwargs["endings"] = EndingAssessment(
        reasoning=_join_reasoning("endings"),
        tag=chosen_ending,
    )

    return ConceptTagsOutput(**merged_kwargs)
