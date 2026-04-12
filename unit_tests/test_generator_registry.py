"""
Unit tests for movie_ingestion.metadata_generation.batch_generation.generator_registry.

Covers:
  - GENERATOR_REGISTRY contains expected types
  - get_config returns correct config for plot_events and reception
  - get_config raises KeyError for unregistered types
  - Prompt builder adapters return (user_prompt, system_prompt) tuples
  - Reception prompt adapter uses the correct system prompt constant
"""

import pytest

from movie_ingestion.metadata_generation.inputs import (
    MetadataType,
    MovieInputData,
)
from movie_ingestion.metadata_generation.batch_generation.generator_registry import (
    GENERATOR_REGISTRY,
    GeneratorConfig,
    _production_techniques_prompt_builder,
    get_config,
    _plot_events_prompt_builder,
    _reception_prompt_builder,
)
from schemas.metadata import (
    PlotEventsOutput,
    ReceptionOutput,
    PlotAnalysisOutput,
    ProductionKeywordsOutput,
    ProductionTechniquesOutput,
    ViewerExperienceOutput,
    WatchContextOutput,
    NarrativeTechniquesOutput,
    SourceOfInspirationOutput,
    SourceMaterialV2Output,
    FranchiseOutput,
    ConceptTagsOutput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_movie(**overrides) -> MovieInputData:
    """Build a MovieInputData with sensible defaults."""
    defaults = dict(
        tmdb_id=12345,
        title="Test Movie",
        release_year=2020,
        overview="A great test movie.",
        genres=["Drama"],
        plot_synopses=[],
        plot_summaries=["Summary."],
        plot_keywords=["keyword"],
        reception_summary="Well received.",
        featured_reviews=[{"summary": "Good", "text": "A good film."}],
    )
    defaults.update(overrides)
    return MovieInputData(**defaults)


# ---------------------------------------------------------------------------
# Tests: GENERATOR_REGISTRY
# ---------------------------------------------------------------------------


class TestGeneratorRegistry:
    """Tests for the module-level GENERATOR_REGISTRY."""

    def test_has_exactly_registered_types(self) -> None:
        """GENERATOR_REGISTRY has entries for all currently registered types."""
        assert set(GENERATOR_REGISTRY.keys()) == {
            MetadataType.PLOT_EVENTS,
            MetadataType.RECEPTION,
            MetadataType.PLOT_ANALYSIS,
            MetadataType.PRODUCTION_KEYWORDS,
            MetadataType.PRODUCTION_TECHNIQUES,
            MetadataType.VIEWER_EXPERIENCE,
            MetadataType.WATCH_CONTEXT,
            MetadataType.NARRATIVE_TECHNIQUES,
            MetadataType.SOURCE_OF_INSPIRATION,
            MetadataType.SOURCE_MATERIAL_V2,
            MetadataType.FRANCHISE,
            MetadataType.CONCEPT_TAGS,
        }


# ---------------------------------------------------------------------------
# Tests: get_config
# ---------------------------------------------------------------------------


class TestGetConfig:
    """Tests for get_config lookup."""

    def test_returns_plot_events_config(self) -> None:
        """get_config(PLOT_EVENTS) returns correct schema and model."""
        config = get_config(MetadataType.PLOT_EVENTS)
        assert isinstance(config, GeneratorConfig)
        assert config.schema_class is PlotEventsOutput
        assert config.model == "gpt-5-mini"
        assert config.metadata_type == MetadataType.PLOT_EVENTS

    def test_returns_reception_config(self) -> None:
        """get_config(RECEPTION) returns correct schema and model."""
        config = get_config(MetadataType.RECEPTION)
        assert isinstance(config, GeneratorConfig)
        assert config.schema_class is ReceptionOutput
        assert config.model == "gpt-5-mini"
        assert config.metadata_type == MetadataType.RECEPTION

    def test_returns_plot_analysis_config(self) -> None:
        """get_config(PLOT_ANALYSIS) returns correct schema and model."""
        config = get_config(MetadataType.PLOT_ANALYSIS)
        assert isinstance(config, GeneratorConfig)
        assert config.schema_class is PlotAnalysisOutput
        assert config.model == "gpt-5-mini"

    def test_returns_production_keywords_config(self) -> None:
        """get_config(PRODUCTION_KEYWORDS) returns correct schema and model."""
        config = get_config(MetadataType.PRODUCTION_KEYWORDS)
        assert isinstance(config, GeneratorConfig)
        assert config.schema_class is ProductionKeywordsOutput
        assert config.model == "gpt-5-mini"

    def test_returns_production_techniques_config(self) -> None:
        """get_config(PRODUCTION_TECHNIQUES) returns correct schema and model."""
        config = get_config(MetadataType.PRODUCTION_TECHNIQUES)
        assert isinstance(config, GeneratorConfig)
        assert config.schema_class is ProductionTechniquesOutput
        assert config.model == "gpt-5-mini"

    def test_returns_viewer_experience_config(self) -> None:
        """get_config(VIEWER_EXPERIENCE) returns correct schema and model."""
        config = get_config(MetadataType.VIEWER_EXPERIENCE)
        assert isinstance(config, GeneratorConfig)
        assert config.schema_class is ViewerExperienceOutput
        assert config.model == "gpt-5-mini"

    def test_returns_watch_context_config(self) -> None:
        """get_config(WATCH_CONTEXT) returns correct schema and model."""
        config = get_config(MetadataType.WATCH_CONTEXT)
        assert isinstance(config, GeneratorConfig)
        assert config.schema_class is WatchContextOutput
        assert config.model == "gpt-5-mini"

    def test_returns_narrative_techniques_config(self) -> None:
        """get_config(NARRATIVE_TECHNIQUES) returns correct schema and model."""
        config = get_config(MetadataType.NARRATIVE_TECHNIQUES)
        assert isinstance(config, GeneratorConfig)
        assert config.schema_class is NarrativeTechniquesOutput
        assert config.model == "gpt-5-mini"

    def test_returns_source_of_inspiration_config(self) -> None:
        config = get_config(MetadataType.SOURCE_OF_INSPIRATION)
        assert config.schema_class is SourceOfInspirationOutput

    def test_returns_source_material_v2_config(self) -> None:
        config = get_config(MetadataType.SOURCE_MATERIAL_V2)
        assert config.schema_class is SourceMaterialV2Output

    def test_returns_franchise_config(self) -> None:
        config = get_config(MetadataType.FRANCHISE)
        assert config.schema_class is FranchiseOutput

    def test_returns_concept_tags_config(self) -> None:
        config = get_config(MetadataType.CONCEPT_TAGS)
        assert config.schema_class is ConceptTagsOutput


# ---------------------------------------------------------------------------
# Tests: Prompt builder adapters
# ---------------------------------------------------------------------------


class TestPromptBuilderAdapters:
    """Tests for the prompt builder adapter functions."""

    def test_plot_events_adapter_returns_tuple(self) -> None:
        """_plot_events_prompt_builder returns a (user_prompt, system_prompt) 2-tuple."""
        movie = _make_movie()
        result = _plot_events_prompt_builder(movie)
        assert isinstance(result, tuple)
        assert len(result) == 2
        user_prompt, system_prompt = result
        assert isinstance(user_prompt, str)
        assert isinstance(system_prompt, str)
        assert len(user_prompt) > 0
        assert len(system_prompt) > 0

    def test_reception_adapter_returns_tuple(self) -> None:
        """_reception_prompt_builder returns a (user_prompt, system_prompt) 2-tuple."""
        movie = _make_movie()
        result = _reception_prompt_builder(movie)
        assert isinstance(result, tuple)
        assert len(result) == 2
        user_prompt, system_prompt = result
        assert isinstance(user_prompt, str)
        assert isinstance(system_prompt, str)
        assert len(user_prompt) > 0
        assert len(system_prompt) > 0

    def test_reception_adapter_uses_correct_system_prompt(self) -> None:
        """_reception_prompt_builder's system prompt matches the SYSTEM_PROMPT constant."""
        from movie_ingestion.metadata_generation.prompts.reception import SYSTEM_PROMPT
        movie = _make_movie()
        _, system_prompt = _reception_prompt_builder(movie)
        assert system_prompt == SYSTEM_PROMPT

    def test_production_techniques_adapter_returns_tuple(self) -> None:
        movie = _make_movie(
            plot_keywords=["single-take"],
            overall_keywords=["anthology", "mockumentary", "drama"],
        )
        result = _production_techniques_prompt_builder(movie)
        assert isinstance(result, tuple)
        assert len(result) == 2
        user_prompt, system_prompt = result
        assert "plot_keywords:" in user_prompt
        assert "overall_keywords:" in user_prompt
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0
