from unittest.mock import AsyncMock, patch

import pytest

from db.postgres import (
    NEUTRAL_RERANKER_SEED_POPULARITY_WEIGHT,
    NEUTRAL_RERANKER_SEED_RECEPTION_WEIGHT,
    fetch_neutral_reranker_seed_ids,
)


@pytest.mark.asyncio
async def test_fetch_neutral_reranker_seed_ids_uses_weighted_normalized_formula() -> None:
    mock_execute = AsyncMock(return_value=[(9,), (8,), (7,)])

    with patch("db.postgres._execute_read", new=mock_execute):
        result = await fetch_neutral_reranker_seed_ids(limit=3)

    assert result == [9, 8, 7]
    query, params = mock_execute.await_args.args
    assert "%s * COALESCE" in query
    assert "popularity_score" in query
    assert "reception_score / 100.0" in query
    assert "LIMIT %s" in query
    assert params == (
        NEUTRAL_RERANKER_SEED_POPULARITY_WEIGHT,
        NEUTRAL_RERANKER_SEED_RECEPTION_WEIGHT,
        3,
    )
