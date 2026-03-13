"""
Unit tests for implementation.llms.query_understanding_methods tuple unpacking.

Each async method now unpacks the 3-tuple from generate_kimi_response_async
via `parsed, _, _ = ...` and returns only `parsed`. These tests verify
the unpacking works correctly and only the parsed model is returned.

All LLM calls are mocked — no real API traffic.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from implementation.classes.schemas import (
    ExtractedEntitiesResponse,
    ChannelWeightsResponse,
    VectorCollectionSubqueryData,
    VectorCollectionWeightData,
)
from implementation.llms.query_understanding_methods import (
    extract_lexical_entities_async,
    create_channel_weights_async,
    extract_single_metadata_preference_async,
    create_single_vector_subquery_async,
    create_single_vector_weight_async,
)
from implementation.classes.enums import MetadataPreferenceName, VectorName


# ---------------------------------------------------------------------------
# Shared mock target
# ---------------------------------------------------------------------------

_KIMI_PATCH = "implementation.llms.query_understanding_methods.generate_kimi_response_async"


# ---------------------------------------------------------------------------
# Tests: tuple unpacking returns only parsed model
# ---------------------------------------------------------------------------


class TestQueryUnderstandingUnpacking:
    """Verify each QU method unpacks the 3-tuple and returns only parsed."""

    async def test_extract_lexical_entities_unpacks_tuple_returns_parsed(self) -> None:
        """extract_lexical_entities_async returns only the parsed model, discarding tokens."""
        mock_response = MagicMock(spec=ExtractedEntitiesResponse)

        with patch(_KIMI_PATCH, new_callable=AsyncMock, return_value=(mock_response, 100, 50)):
            result = await extract_lexical_entities_async("action movies with robots")

        assert result is mock_response

    async def test_create_channel_weights_unpacks_tuple_returns_parsed(self) -> None:
        """create_channel_weights_async returns only the parsed model, discarding tokens."""
        mock_response = MagicMock(spec=ChannelWeightsResponse)

        with patch(_KIMI_PATCH, new_callable=AsyncMock, return_value=(mock_response, 200, 80)):
            result = await create_channel_weights_async("romantic comedies from the 90s")

        assert result is mock_response

    async def test_extract_single_metadata_preference_unpacks_tuple_returns_parsed(self) -> None:
        """extract_single_metadata_preference_async returns only the parsed model."""
        mock_response = MagicMock()

        with patch(_KIMI_PATCH, new_callable=AsyncMock, return_value=(mock_response, 150, 60)):
            result = await extract_single_metadata_preference_async(
                "movies from 2020", MetadataPreferenceName.RELEASE_DATE
            )

        assert result is mock_response

    async def test_create_single_vector_subquery_unpacks_tuple_returns_parsed(self) -> None:
        """create_single_vector_subquery_async returns only the parsed model."""
        mock_response = MagicMock(spec=VectorCollectionSubqueryData)

        with patch(_KIMI_PATCH, new_callable=AsyncMock, return_value=(mock_response, 120, 45)):
            result = await create_single_vector_subquery_async(
                "dark thriller with a twist", VectorName.PLOT_EVENTS
            )

        assert result is mock_response

    async def test_create_single_vector_weight_unpacks_tuple_returns_parsed(self) -> None:
        """create_single_vector_weight_async returns only the parsed model."""
        mock_response = MagicMock(spec=VectorCollectionWeightData)

        with patch(_KIMI_PATCH, new_callable=AsyncMock, return_value=(mock_response, 90, 30)):
            result = await create_single_vector_weight_async(
                "feel-good family movie", VectorName.VIEWER_EXPERIENCE
            )

        assert result is mock_response
