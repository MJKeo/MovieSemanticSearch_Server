"""
Unit tests for movie_ingestion.metadata_generation.errors.

Covers:
  - MetadataGenerationError: attributes, message format, inheritance
  - MetadataGenerationEmptyResponseError: attributes, message format, inheritance
"""

from movie_ingestion.metadata_generation.errors import (
    MetadataGenerationError,
    MetadataGenerationEmptyResponseError,
)


# ---------------------------------------------------------------------------
# MetadataGenerationError
# ---------------------------------------------------------------------------

class TestMetadataGenerationError:
    def test_attributes_stored(self):
        cause = ValueError("API timeout")
        err = MetadataGenerationError("plot_events", "The Matrix (1999)", cause)
        assert err.generation_type == "plot_events"
        assert err.title == "The Matrix (1999)"
        assert err.cause is cause

    def test_message_format(self):
        cause = RuntimeError("connection reset")
        err = MetadataGenerationError("reception", "Inception (2010)", cause)
        assert "reception" in str(err)
        assert "Inception (2010)" in str(err)
        assert "connection reset" in str(err)

    def test_inherits_from_exception(self):
        err = MetadataGenerationError("test", "title", ValueError("x"))
        assert isinstance(err, Exception)


# ---------------------------------------------------------------------------
# MetadataGenerationEmptyResponseError
# ---------------------------------------------------------------------------

class TestMetadataGenerationEmptyResponseError:
    def test_attributes_stored(self):
        err = MetadataGenerationEmptyResponseError("plot_analysis", "Tenet (2020)")
        assert err.generation_type == "plot_analysis"
        assert err.title == "Tenet (2020)"

    def test_message_format(self):
        err = MetadataGenerationEmptyResponseError("watch_context", "Dune (2021)")
        assert "watch_context" in str(err)
        assert "Dune (2021)" in str(err)
        assert "None" in str(err)

    def test_inherits_from_exception(self):
        err = MetadataGenerationEmptyResponseError("test", "title")
        assert isinstance(err, Exception)
