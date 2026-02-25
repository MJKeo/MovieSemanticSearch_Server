"""Unit tests for vector-search timing instrumentation."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("qdrant_client")

from db import vector_search
from implementation.classes.enums import RelevanceSize, VectorName
from implementation.classes.schemas import (
    MetadataFilters,
    VectorCollectionSubqueryData,
    VectorCollectionWeightData,
)


class _FakeQdrantClient:
    """Minimal async client stub for query_points used by vector_search tests."""

    async def query_points(self, **kwargs):  # noqa: ANN003
        # Return a stable single candidate so merge/stat paths run.
        return SimpleNamespace(points=[SimpleNamespace(id=101, score=0.91)])


async def _fake_embedding(texts: list[str]) -> list[list[float]]:
    await asyncio.sleep(0.005)
    return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.mark.asyncio
async def test_anchor_job_stats_has_embedding_time_and_null_llm_time(mocker) -> None:
    """Anchor job should carry embedding timing and null LLM timing."""
    mocker.patch("db.vector_search.generate_vector_embedding", new=AsyncMock(side_effect=_fake_embedding))

    async def _all_not_relevant(query: str, collection_name: VectorName) -> VectorCollectionWeightData:  # noqa: ARG001
        await asyncio.sleep(0.003)
        return VectorCollectionWeightData(relevance=RelevanceSize.NOT_RELEVANT, justification="skip")

    async def _all_no_subquery(query: str, collection_name: VectorName) -> VectorCollectionSubqueryData:  # noqa: ARG001
        await asyncio.sleep(0.003)
        return VectorCollectionSubqueryData(justification="skip", relevant_subquery_text=None)

    mocker.patch("db.vector_search.create_single_vector_weight_async", new=AsyncMock(side_effect=_all_not_relevant))
    mocker.patch("db.vector_search.create_single_vector_subquery_async", new=AsyncMock(side_effect=_all_no_subquery))

    result = await vector_search.run_vector_search(
        query="anchor-only query",
        metadata_filters=MetadataFilters(),
        qdrant_client=_FakeQdrantClient(),
    )

    assert result.debug.total_jobs_executed == 1
    anchor_stat = result.debug.per_job_stats[0]
    assert anchor_stat.score_field == "anchor_score_original"
    assert anchor_stat.embedding_time_ms > 0
    assert anchor_stat.llm_generation_time_ms is None


@pytest.mark.asyncio
async def test_original_job_stats_uses_weight_llm_and_shared_embedding_durations(mocker) -> None:
    """Original-query jobs should include weight LLM timing and original embedding timing."""
    mocker.patch("db.vector_search.generate_vector_embedding", new=AsyncMock(side_effect=_fake_embedding))

    async def _weights(query: str, collection_name: VectorName) -> VectorCollectionWeightData:  # noqa: ARG001
        await asyncio.sleep(0.004)
        relevance = RelevanceSize.SMALL if collection_name == VectorName.PLOT_EVENTS else RelevanceSize.NOT_RELEVANT
        return VectorCollectionWeightData(relevance=relevance, justification="timed")

    async def _all_no_subquery(query: str, collection_name: VectorName) -> VectorCollectionSubqueryData:  # noqa: ARG001
        await asyncio.sleep(0.002)
        return VectorCollectionSubqueryData(justification="skip", relevant_subquery_text=None)

    mocker.patch("db.vector_search.create_single_vector_weight_async", new=AsyncMock(side_effect=_weights))
    mocker.patch("db.vector_search.create_single_vector_subquery_async", new=AsyncMock(side_effect=_all_no_subquery))

    result = await vector_search.run_vector_search(
        query="original-query timing",
        metadata_filters=MetadataFilters(),
        qdrant_client=_FakeQdrantClient(),
    )

    original_stat = next(s for s in result.debug.per_job_stats if s.score_field == "plot_events_score_original")
    assert original_stat.embedding_time_ms > 0
    assert original_stat.llm_generation_time_ms is not None
    assert original_stat.llm_generation_time_ms > 0


@pytest.mark.asyncio
async def test_subquery_job_stats_uses_subquery_llm_and_subquery_embedding_durations(mocker) -> None:
    """Subquery jobs should include subquery-LLM timing and subquery-embedding timing."""
    mocker.patch("db.vector_search.generate_vector_embedding", new=AsyncMock(side_effect=_fake_embedding))

    async def _all_not_relevant(query: str, collection_name: VectorName) -> VectorCollectionWeightData:  # noqa: ARG001
        await asyncio.sleep(0.002)
        return VectorCollectionWeightData(relevance=RelevanceSize.NOT_RELEVANT, justification="skip")

    async def _subqueries(query: str, collection_name: VectorName) -> VectorCollectionSubqueryData:  # noqa: ARG001
        await asyncio.sleep(0.004)
        if collection_name == VectorName.PLOT_ANALYSIS:
            return VectorCollectionSubqueryData(justification="timed", relevant_subquery_text="complex moral conflict")
        return VectorCollectionSubqueryData(justification="skip", relevant_subquery_text=None)

    mocker.patch("db.vector_search.create_single_vector_weight_async", new=AsyncMock(side_effect=_all_not_relevant))
    mocker.patch("db.vector_search.create_single_vector_subquery_async", new=AsyncMock(side_effect=_subqueries))

    result = await vector_search.run_vector_search(
        query="subquery timing",
        metadata_filters=MetadataFilters(),
        qdrant_client=_FakeQdrantClient(),
    )

    subquery_stat = next(s for s in result.debug.per_job_stats if s.score_field == "plot_analysis_score_subquery")
    assert subquery_stat.embedding_time_ms > 0
    assert subquery_stat.llm_generation_time_ms is not None
    assert subquery_stat.llm_generation_time_ms > 0


@pytest.mark.asyncio
async def test_skipped_jobs_do_not_emit_stats_rows(mocker) -> None:
    """Only executed searches should appear in per_job_stats."""
    mocker.patch("db.vector_search.generate_vector_embedding", new=AsyncMock(side_effect=_fake_embedding))

    async def _all_not_relevant(query: str, collection_name: VectorName) -> VectorCollectionWeightData:  # noqa: ARG001
        await asyncio.sleep(0.002)
        return VectorCollectionWeightData(relevance=RelevanceSize.NOT_RELEVANT, justification="skip")

    async def _all_no_subquery(query: str, collection_name: VectorName) -> VectorCollectionSubqueryData:  # noqa: ARG001
        await asyncio.sleep(0.002)
        return VectorCollectionSubqueryData(justification="skip", relevant_subquery_text=None)

    mocker.patch("db.vector_search.create_single_vector_weight_async", new=AsyncMock(side_effect=_all_not_relevant))
    mocker.patch("db.vector_search.create_single_vector_subquery_async", new=AsyncMock(side_effect=_all_no_subquery))

    result = await vector_search.run_vector_search(
        query="all skipped except anchor",
        metadata_filters=MetadataFilters(),
        qdrant_client=_FakeQdrantClient(),
    )

    assert result.debug.total_jobs_executed == 1
    assert [s.score_field for s in result.debug.per_job_stats] == ["anchor_score_original"]


@pytest.mark.asyncio
async def test_timing_isolation_not_await_wait_time() -> None:
    """Timed task duration should reflect its own runtime, not delayed await time."""

    async def _work() -> str:
        await asyncio.sleep(0.03)
        return "done"

    task = asyncio.create_task(vector_search._time_awaitable(_work()))
    await asyncio.sleep(0.08)  # Delay awaiting the completed task.
    timed_result = await task

    assert timed_result.value == "done"
    assert 10 <= timed_result.duration_ms <= 80
