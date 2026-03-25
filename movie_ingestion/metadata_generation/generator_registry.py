"""
Registry mapping each MetadataType to its generator configuration.

Centralizes the per-type config (schema, eligibility check, prompt builder,
live generator, model settings) so that the batch pipeline (request_builder,
result_processor, run) can operate generically over any metadata type.

Each generator module has its own prompt-building interface — plot_events
returns a (user_prompt, system_prompt) tuple while reception returns just
the user prompt with the system prompt as a separate constant. The registry
normalizes these into a common (user_prompt, system_prompt) tuple contract
via thin adapter wrappers, avoiding changes to the generator modules.

Wave 2 types (plot_analysis, etc.) depend on Wave 1 outputs stored in the
generated_metadata table. Their adapters load those outputs at call time
via load_wave1_outputs() (from inputs.py), keeping the generic pipeline interface
(MovieInputData → result) intact.

To add a new metadata type:
    1. Write its generator in generators/{type}.py
    2. Write its eligibility check in pre_consolidation.py
    3. Add an entry to GENERATOR_REGISTRY below
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic import BaseModel

from movie_ingestion.metadata_generation.inputs import (
    MetadataType,
    MovieInputData,
    load_wave1_outputs,
)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class GeneratorConfig:
    """Everything the batch pipeline needs to handle one metadata type.

    Fields:
        metadata_type: Which type this config is for.
        schema_class: Pydantic output model (e.g. PlotEventsOutput).
        eligibility_checker: (MovieInputData) -> str | None.
            Returns None if eligible, or a skip-reason string.
        prompt_builder: (MovieInputData) -> (user_prompt, system_prompt).
            Normalized to always return a two-tuple.
        live_generator: async (MovieInputData) -> (output, TokenUsage).
            The direct API call function for live/autopilot mode.
        model: OpenAI model name for batch requests.
        model_kwargs: Extra kwargs for batch requests (e.g. reasoning_effort).
    """
    metadata_type: MetadataType
    schema_class: type[BaseModel]
    eligibility_checker: Callable[[MovieInputData], str | None]
    prompt_builder: Callable[[MovieInputData], tuple[str, str]]
    live_generator: Callable[..., Awaitable[Any]]
    model: str
    model_kwargs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompt builder adapters
# ---------------------------------------------------------------------------
# Generator modules have different prompt-building interfaces. These thin
# adapters normalize them into the (user_prompt, system_prompt) tuple
# contract expected by the batch request builder.

def _plot_events_prompt_builder(movie: MovieInputData) -> tuple[str, str]:
    """Adapter for plot_events — already returns (user, system) tuple."""
    from .generators.plot_events import build_plot_events_prompts
    return build_plot_events_prompts(movie)


def _reception_prompt_builder(movie: MovieInputData) -> tuple[str, str]:
    """Adapter for reception — combines user prompt builder + separate system prompt."""
    from .generators.reception import build_reception_user_prompt
    from .prompts.reception import SYSTEM_PROMPT
    return build_reception_user_prompt(movie), SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Wave 2 prompt builder and eligibility adapters
# ---------------------------------------------------------------------------
# Wave 2 generators depend on Wave 1 outputs. These adapters load Wave 1
# data from the DB so the generic pipeline can call them with just
# MovieInputData, matching the same interface as Wave 1 types.

def _plot_analysis_eligibility_checker(movie: MovieInputData) -> str | None:
    """Eligibility checker for plot_analysis — loads Wave 1 outputs from DB.

    Delegates to _check_plot_analysis which implements tiered eligibility:
    movies with Wave 1 plot_synopsis are Tier 1 (always eligible), those
    with sufficient raw plot text are Tier 2/3.
    """
    from .pre_consolidation import _check_plot_analysis

    w1 = load_wave1_outputs(movie.tmdb_id)
    return _check_plot_analysis(w1.plot_summary, w1.thematic_observations, movie)


def _plot_analysis_prompt_builder(movie: MovieInputData) -> tuple[str, str]:
    """Adapter for plot_analysis — loads Wave 1 outputs and builds prompts."""
    from .generators.plot_analysis import build_plot_analysis_user_prompt
    from .prompts.plot_analysis import SYSTEM_PROMPT

    w1 = load_wave1_outputs(movie.tmdb_id)
    return build_plot_analysis_user_prompt(movie, w1.plot_summary, w1.thematic_observations), SYSTEM_PROMPT


async def _plot_analysis_live_generator(movie: MovieInputData):
    """Async adapter for plot_analysis — loads Wave 1 outputs and generates."""
    from .generators.plot_analysis import generate_plot_analysis

    w1 = load_wave1_outputs(movie.tmdb_id)
    return await generate_plot_analysis(movie, w1.plot_summary, w1.thematic_observations)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _build_registry() -> dict[MetadataType, GeneratorConfig]:
    """Build the registry at import time.

    Uses lazy imports inside adapters and here to avoid circular imports
    (generators import from inputs, which is also imported here).
    """
    from .pre_consolidation import check_plot_events, check_reception
    from .generators.plot_events import generate_plot_events
    from .generators.reception import generate_reception
    from .schemas import PlotEventsOutput, ReceptionOutput, PlotAnalysisWithJustificationsOutput

    return {
        MetadataType.PLOT_EVENTS: GeneratorConfig(
            metadata_type=MetadataType.PLOT_EVENTS,
            schema_class=PlotEventsOutput,
            eligibility_checker=check_plot_events,
            prompt_builder=_plot_events_prompt_builder,
            live_generator=generate_plot_events,
            model="gpt-5-mini",
            model_kwargs={"reasoning_effort": "minimal", "verbosity": "low"},
        ),
        MetadataType.RECEPTION: GeneratorConfig(
            metadata_type=MetadataType.RECEPTION,
            schema_class=ReceptionOutput,
            eligibility_checker=check_reception,
            prompt_builder=_reception_prompt_builder,
            live_generator=generate_reception,
            model="gpt-5-mini",
            model_kwargs={"reasoning_effort": "minimal", "verbosity": "low"},
        ),
        MetadataType.PLOT_ANALYSIS: GeneratorConfig(
            metadata_type=MetadataType.PLOT_ANALYSIS,
            schema_class=PlotAnalysisWithJustificationsOutput,
            eligibility_checker=_plot_analysis_eligibility_checker,
            prompt_builder=_plot_analysis_prompt_builder,
            live_generator=_plot_analysis_live_generator,
            model="gpt-5-mini",
            model_kwargs={"reasoning_effort": "minimal", "verbosity": "low"},
        ),
    }


# Module-level registry — populated once on first import.
GENERATOR_REGISTRY: dict[MetadataType, GeneratorConfig] = _build_registry()


def get_config(metadata_type: MetadataType) -> GeneratorConfig:
    """Look up the generator config for a metadata type.

    Raises KeyError with a descriptive message if the type is not yet
    registered (e.g. Wave 2 types that haven't been added yet).
    """
    try:
        return GENERATOR_REGISTRY[metadata_type]
    except KeyError:
        registered = ", ".join(sorted(GENERATOR_REGISTRY.keys()))
        raise KeyError(
            f"No generator registered for '{metadata_type}'. "
            f"Registered types: {registered}"
        ) from None
