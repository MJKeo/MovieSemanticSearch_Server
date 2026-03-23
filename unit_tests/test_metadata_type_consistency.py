"""
Unit tests for MetadataType enum consistency across all 8 generators.

Covers:
  - All generators use MetadataType for GENERATION_TYPE
  - All MetadataType values have corresponding generators
  - MetadataType values match generated_metadata DB column names
"""

import sqlite3

from movie_ingestion.metadata_generation.inputs import MetadataType
from movie_ingestion.tracker import _SCHEMA_SQL


# ---------------------------------------------------------------------------
# Tests: All generators use MetadataType enum for GENERATION_TYPE
# ---------------------------------------------------------------------------


class TestAllGeneratorsUseMetadataType:
    """Tests that all 8 generators use MetadataType for their GENERATION_TYPE constant."""

    def test_all_generators_use_metadata_type_for_generation_type(self) -> None:
        """Import GENERATION_TYPE from each generator and verify it's a MetadataType instance."""
        from movie_ingestion.metadata_generation.generators.plot_events import (
            GENERATION_TYPE as gt_plot_events,
        )
        from movie_ingestion.metadata_generation.generators.reception import (
            GENERATION_TYPE as gt_reception,
        )
        from movie_ingestion.metadata_generation.generators.plot_analysis import (
            GENERATION_TYPE as gt_plot_analysis,
        )
        from movie_ingestion.metadata_generation.generators.viewer_experience import (
            GENERATION_TYPE as gt_viewer_experience,
        )
        from movie_ingestion.metadata_generation.generators.watch_context import (
            GENERATION_TYPE as gt_watch_context,
        )
        from movie_ingestion.metadata_generation.generators.narrative_techniques import (
            GENERATION_TYPE as gt_narrative_techniques,
        )
        from movie_ingestion.metadata_generation.generators.production_keywords import (
            GENERATION_TYPE as gt_production_keywords,
        )
        from movie_ingestion.metadata_generation.generators.source_of_inspiration import (
            GENERATION_TYPE as gt_source_of_inspiration,
        )

        all_types = [
            gt_plot_events,
            gt_reception,
            gt_plot_analysis,
            gt_viewer_experience,
            gt_watch_context,
            gt_narrative_techniques,
            gt_production_keywords,
            gt_source_of_inspiration,
        ]

        for gt in all_types:
            assert isinstance(gt, MetadataType), (
                f"GENERATION_TYPE {gt!r} is not a MetadataType instance"
            )


# ---------------------------------------------------------------------------
# Tests: All MetadataType values have corresponding generators
# ---------------------------------------------------------------------------


class TestAllMetadataTypesHaveGenerators:
    """Tests that each MetadataType enum value has a corresponding generator."""

    def test_all_8_metadata_types_have_generators(self) -> None:
        """Verify each MetadataType enum value has a generator module with a matching GENERATION_TYPE."""
        from movie_ingestion.metadata_generation.generators.plot_events import (
            GENERATION_TYPE as gt_plot_events,
        )
        from movie_ingestion.metadata_generation.generators.reception import (
            GENERATION_TYPE as gt_reception,
        )
        from movie_ingestion.metadata_generation.generators.plot_analysis import (
            GENERATION_TYPE as gt_plot_analysis,
        )
        from movie_ingestion.metadata_generation.generators.viewer_experience import (
            GENERATION_TYPE as gt_viewer_experience,
        )
        from movie_ingestion.metadata_generation.generators.watch_context import (
            GENERATION_TYPE as gt_watch_context,
        )
        from movie_ingestion.metadata_generation.generators.narrative_techniques import (
            GENERATION_TYPE as gt_narrative_techniques,
        )
        from movie_ingestion.metadata_generation.generators.production_keywords import (
            GENERATION_TYPE as gt_production_keywords,
        )
        from movie_ingestion.metadata_generation.generators.source_of_inspiration import (
            GENERATION_TYPE as gt_source_of_inspiration,
        )

        covered_types = {
            gt_plot_events,
            gt_reception,
            gt_plot_analysis,
            gt_viewer_experience,
            gt_watch_context,
            gt_narrative_techniques,
            gt_production_keywords,
            gt_source_of_inspiration,
        }
        all_types = set(MetadataType)

        assert covered_types == all_types, (
            f"Missing generators for: {all_types - covered_types}"
        )


# ---------------------------------------------------------------------------
# Tests: MetadataType values match generated_metadata DB column names
# ---------------------------------------------------------------------------


class TestMetadataTypeMatchesDbColumns:
    """Tests that MetadataType values match columns in the generated_metadata table."""

    def test_metadata_type_values_match_db_columns(self) -> None:
        """Each MetadataType.value matches a column name in generated_metadata table schema."""
        db = sqlite3.connect(":memory:")
        db.executescript(_SCHEMA_SQL)

        cols = db.execute("PRAGMA table_info(generated_metadata)").fetchall()
        col_names = {row[1] for row in cols}
        db.close()

        for mt in MetadataType:
            assert mt.value in col_names, (
                f"MetadataType.{mt.name} value '{mt.value}' has no matching "
                f"column in generated_metadata table"
            )
