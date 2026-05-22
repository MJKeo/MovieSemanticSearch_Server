"""
Concept Tags generator (Wave 2).

Multi-label binary classification of 25 concept tags across 7 categories.
Each tag answers a yes/no question: "does this movie have X?" Tags enable
deterministic search retrieval ("give me ALL revenge movies") via Postgres
integer array containment queries.

Classification mindset: the LLM evaluates each category independently,
outputs only the tags supported by input evidence. Empty categories are
correct and expected.

Inputs:
    - movie (MovieInputData): title, release_year, plot_keywords, plus
      raw plot sources used as fallback and parental_guide_items
      (category + severity content tags from IMDB)
    - plot_summary: from Wave 1 plot_events (may be None)
    - emotional_observations: from Wave 1 reception (may be None)
    - craft_observations: from Wave 1 reception — reviewer descriptions
      of structural/storytelling craft. Primary signal for
      NONLINEAR_TIMELINE, supports PLOT_TWIST / UNRELIABLE_NARRATOR /
      BREAKING_FOURTH_WALL (may be None)
    - nt_output: NarrativeTechniquesOutput from Wave 2 — 5 sections
      (narrative_archetype, narrative_delivery, pov_perspective,
      information_control, additional_narrative_devices). The
      character_arcs and audience_character_perception sections were
      removed after the three-way eval showed they were the primary
      drivers of ANTI_HERO false positives: their term lists literally
      contain "antihero maturation arc" / "sympathetic antihero" and
      the consumer rubber-stamped those labels (may be None)
    - pa_output: PlotAnalysisOutput from Wave 2. character_arc_labels
      (arc transformation labels like "impostor to reconciled
      contributor") + conflict_type now carry the protagonist-arc
      signal that NT character_arcs previously provided, in a form
      that doesn't pre-classify into a tag (may be None)

When Wave 1/2 outputs are unavailable, the prompt builder uses raw movie
data as fallback (best_plot_fallback for plot, "not available" for absent
fields) so the LLM can classify from whatever signals are available.

Skip condition (enforced by pre_consolidation):
    Eligible when ANY of:
    - plot_summary exists (Wave 1 plot_events succeeded)
    - best_plot_fallback() >= 250 chars
    - plot_keywords >= 3

Provider/model: OpenAI gpt-5-mini, reasoning_effort: minimal, verbosity: low.
"""

from __future__ import annotations

from typing import Tuple

from schemas.enums import MetadataType
from schemas.movie_input import MovieInputData
from schemas.metadata import (
    ConceptTagsOutput,
    PlotAnalysisOutput,
    NarrativeTechniquesOutput,
)
from movie_ingestion.metadata_generation.inputs import (
    build_user_prompt,
    extract_narrative_technique_terms,
)
from movie_ingestion.metadata_generation.prompts.concept_tags import SYSTEM_PROMPT
from movie_ingestion.metadata_generation.errors import (
    MetadataGenerationError,
    MetadataGenerationEmptyResponseError,
)
from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from implementation.llms.vector_metadata_generation_methods import TokenUsage

GENERATION_TYPE = MetadataType.CONCEPT_TAGS

_PROVIDER = LLMProvider.OPENAI
_MODEL = "gpt-5-mini"


def _format_narrative_technique_terms(
    nt_output: NarrativeTechniquesOutput | None,
) -> str:
    """Format narrative technique terms as a compact labeled block.

    Extracts terms from the 5 relevant sections (no justifications):
    narrative_archetype, narrative_delivery, pov_perspective,
    information_control, additional_narrative_devices. Each section is
    one line; empty sections show "not available" so the LLM knows the
    data was checked but absent.

    The character_arcs and audience_character_perception sections were
    removed (see extract_narrative_technique_terms docstring) because
    their terms used tag-shaped vocabulary that the consumer
    rubber-stamped without independent reasoning.

    Returns "not available" if the entire NT output is absent.
    """
    if nt_output is None:
        return "not available"

    terms = extract_narrative_technique_terms(nt_output)
    lines: list[str] = []
    for section_name, section_terms in terms.items():
        if section_terms:
            lines.append(f"{section_name}: {', '.join(section_terms)}")
        else:
            lines.append(f"{section_name}: not available")
    # Leading newline so build_user_prompt renders as:
    #   narrative_technique_terms:
    #   narrative_archetype: underdog rising
    #   narrative_delivery: ...
    # rather than jamming the first section onto the label line.
    return "\n" + "\n".join(lines)


def _format_parental_guide_items(
    movie: MovieInputData,
    max_items: int = 20,
) -> str:
    """Format IMDB parental_guide_items as a compact category/severity list.

    Each item is rendered as 'category (severity)' on its own line.
    Capped at max_items so a long list doesn't dominate the prompt —
    the failure clusters this addresses (ANIMAL_DEATH, KIDNAPPING) key
    on specific categories that surface near the top.

    Returns "not available" when the field is empty.
    """
    items = movie.parental_guide_items or []
    if not items:
        return "not available"

    lines: list[str] = []
    for item in items[:max_items]:
        category = (item.get("category") or "").strip()
        severity = (item.get("severity") or "").strip()
        if not category:
            continue
        if severity:
            lines.append(f"- {category} ({severity})")
        else:
            lines.append(f"- {category}")

    if not lines:
        return "not available"
    # Leading newline so it renders as a labeled bullet block under
    # 'parental_guide_items:' rather than jamming the first line onto
    # the label.
    return "\n" + "\n".join(lines)


def _format_plot_analysis_fields(
    pa_output: PlotAnalysisOutput | None,
) -> tuple[str, str]:
    """Extract character_arc_labels and conflict_type from plot analysis.

    Returns (character_arc_labels_str, conflict_type_str), each as a
    comma-joined string or "not available".
    """
    if pa_output is None:
        return "not available", "not available"

    # character_arcs[].arc_transformation_label
    arc_labels = [
        arc.arc_transformation_label
        for arc in pa_output.character_arcs
        if arc.arc_transformation_label
    ]
    arc_str = ", ".join(arc_labels) if arc_labels else "not available"

    # conflict_type
    conflict_str = (
        ", ".join(pa_output.conflict_type)
        if pa_output.conflict_type
        else "not available"
    )

    return arc_str, conflict_str


def build_concept_tags_user_prompt(
    movie: MovieInputData,
    plot_summary: str | None,
    emotional_observations: str | None,
    nt_output: NarrativeTechniquesOutput | None,
    pa_output: PlotAnalysisOutput | None,
    craft_observations: str | None = None,
) -> str:
    """Build the user prompt for concept tag classification.

    Shared by the production generator and the evaluation pipeline so
    prompt construction logic stays in one place.

    When plot_summary (from Wave 1 plot_events) is available, it's passed
    with the 'plot_summary' label. When unavailable, best_plot_fallback()
    is passed with the 'plot_text' label to signal lower quality tier.

    The post-baseline inputs (craft_observations, parental_guide_items,
    character_arc_labels + conflict_type from pa_output) target specific
    failure clusters. The previously-added `ending_aftertaste` from
    ViewerExperience and the NT `character_arcs` /
    `audience_character_perception` subsections were removed after the
    three-way eval showed they drove the BITTERSWEET / ANTI_HERO
    upstream-label contamination — see extract_narrative_technique_terms
    for the rationale.

    All absent inputs are rendered as "not available" so the LLM sees
    explicit absence rather than silent omission.
    """
    # Resolve plot narrative with quality-tiered label
    plot_label: str
    plot_value: str | None
    if plot_summary:
        plot_label = "plot_summary"
        plot_value = plot_summary
    else:
        fallback = movie.best_plot_fallback()
        if fallback:
            plot_label = "plot_text"
            plot_value = fallback
        else:
            plot_label = "plot_summary"
            plot_value = "not available"

    # Format upstream outputs
    nt_terms_str = _format_narrative_technique_terms(nt_output)
    arc_labels_str, conflict_type_str = _format_plot_analysis_fields(pa_output)
    parental_guide_str = _format_parental_guide_items(movie)

    # Top-5 billed cast gives the LLM a prominence ranking to cross-
    # reference against plot_summary character mentions. Critical for
    # disambiguating lead vs. ensemble vs. supporting roles (used
    # especially by FEMALE_LEAD and ENSEMBLE_CAST classification).
    top_billed_cast = movie.top_billed_cast(n=5) or "not available"

    return build_user_prompt(
        title=movie.title_with_year(),
        plot_keywords=movie.plot_keywords or "not available",
        **{plot_label: plot_value},
        top_billed_cast=top_billed_cast,
        emotional_observations=emotional_observations or "not available",
        craft_observations=craft_observations or "not available",
        narrative_technique_terms=nt_terms_str,
        character_arc_labels=arc_labels_str,
        conflict_type=conflict_type_str,
        parental_guide_items=parental_guide_str,
    )


async def generate_concept_tags(
    movie: MovieInputData,
    plot_summary: str | None = None,
    emotional_observations: str | None = None,
    nt_output: NarrativeTechniquesOutput | None = None,
    pa_output: PlotAnalysisOutput | None = None,
    craft_observations: str | None = None,
    timeout: float | None = None,
) -> Tuple[ConceptTagsOutput, TokenUsage]:
    """Generate concept tags for a single movie.

    Builds the user prompt from the movie's fields plus Wave 1/2 outputs,
    calls the LLM with the binary classification prompt, and returns the
    parsed result alongside token usage.

    Uses gpt-5-mini with reasoning effort 'minimal' and low verbosity.

    Args:
        movie: Raw movie input data from the ingestion pipeline.
        plot_summary: LLM-condensed plot summary from Wave 1 plot_events.
            None if plot_events was skipped or failed.
        emotional_observations: Audience emotional response from Wave 1
            reception extraction zone. None if reception was skipped.
        nt_output: Full NarrativeTechniquesOutput from Wave 2. None if
            narrative_techniques was skipped or not yet generated.
        pa_output: Full PlotAnalysisOutput from Wave 2. None if
            plot_analysis was skipped or not yet generated.
        craft_observations: Reviewer descriptions of structural and
            storytelling craft from Wave 1 reception. None when reception
            was skipped or had no craft content.
        timeout: Optional per-attempt LLM timeout in seconds. Concept
            tags runs gpt-5-mini with reasoning effort 'minimal' over a
            heavy multi-section prompt, so wall time routinely exceeds
            the router default. Callers that fan out many concurrent
            runs (e.g. the eval script) should pass a higher value to
            avoid premature `asyncio.wait_for` cancellation. None falls
            back to the router default.

    Returns:
        Tuple of (ConceptTagsOutput, TokenUsage).

    Raises:
        MetadataGenerationError: If the LLM call raises an exception.
        MetadataGenerationEmptyResponseError: If the LLM returns None.
    """

    user_prompt = build_concept_tags_user_prompt(
        movie,
        plot_summary,
        emotional_observations,
        nt_output,
        pa_output,
        craft_observations=craft_observations,
    )
    title_with_year = movie.title_with_year()

    try:
        parsed, input_tokens, output_tokens = await generate_llm_response_async(
            provider=_PROVIDER,
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            response_format=ConceptTagsOutput,
            model=_MODEL,
            reasoning_effort="minimal",
            verbosity="low",
            timeout=timeout,
        )
    except Exception as e:
        print(f"{GENERATION_TYPE} generation failed for '{title_with_year}': {e}")
        raise MetadataGenerationError(GENERATION_TYPE, title_with_year, e) from e

    if parsed is None:
        print(f"{GENERATION_TYPE} generation returned None for '{title_with_year}'")
        raise MetadataGenerationEmptyResponseError(GENERATION_TYPE, title_with_year)

    # Apply deterministic fixups (e.g., TWIST_VILLAIN → PLOT_TWIST implication)
    parsed.apply_deterministic_fixups()

    return parsed, TokenUsage(input_tokens, output_tokens, _MODEL)
