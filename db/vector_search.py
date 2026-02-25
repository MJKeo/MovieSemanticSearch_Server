"""
vector_search.py — End-to-end vector search orchestration.

This module owns everything from "I have a user query and metadata filters"
to "here is a dict of movie_id → per-channel cosine similarity scores."

The high-level flow:
  1. Fire off all LLM calls (14 total: 7 weights + 7 subqueries) in parallel.
  2. Simultaneously start embedding the original query text.
  3. As each LLM result resolves, immediately kick off the Qdrant search(es)
     that it unlocks — don't wait for the other collections.
  4. Each Qdrant result is merged into a shared candidates dict as it arrives.
  5. Once all coordinators finish, return the fully-merged candidates + debug info.

Key design decisions:
  - Single-threaded asyncio: we rely on the fact that only one coroutine runs
    at a time and control only yields at `await` points. This makes the shared
    `candidates` dict safe to write to without locks, as long as the write loop
    contains no `await` calls.
  - Each collection gets TWO independent coordinator tasks (one for the
    original-query search, one for the subquery search) so that neither blocks
    the other.
  - Errors in individual coordinators are caught and logged, never crash the
    whole search. If one collection's LLM call fails, the other 6 + anchor
    still produce results.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Generic, Optional, TypeVar

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchAny,
    Range,
    SearchParams,
    QuantizationSearchParams,
)

from implementation.classes.enums import VectorName, RelevanceSize
from implementation.classes.schemas import (
    MetadataFilters,
    VectorCollectionSubqueryData,
    VectorCollectionWeightData,
)
from implementation.llms.generic_methods import (
    generate_vector_embedding,
)
from implementation.llms.query_understanding_methods import (
    create_single_vector_weight_async,
    create_single_vector_subquery_async,
)

# ===========================================================================
# SECTION 1: ENUMS AND CONSTANTS
# ===========================================================================

# The Qdrant collection alias. All queries go to this alias, which points to
# the current physical collection. This lets us
# rebuild the collection behind the scenes without changing any query code.
COLLECTION_ALIAS = os.getenv("QDRANT_COLLECTION_ALIAS", None)

# Qdrant query-time parameters shared across all searches. Module-level constant
# because these values never change between requests.
#   - hnsw_ef=128: higher than default for better recall (our latency budget
#     is dominated by LLM calls, so Qdrant can afford to be thorough)
#   - rescore=True: use original float32 vectors (from disk) to re-rank
#     candidates found via the quantized int8 index (in RAM)
#   - oversampling=2.0: fetch 2x candidates in the quantized stage, then
#     rescore all of them and return the best `limit`. Recovers accuracy
#     lost from int8 quantization.
_QDRANT_SEARCH_PARAMS = SearchParams(
    hnsw_ef=128,
    quantization=QuantizationSearchParams(
        rescore=True,
        oversampling=2.0,
    ),
)


# ===========================================================================
# SECTION 2: DATA MODELS
# ===========================================================================

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class TimedResult(Generic[T]):
    """
    Wrapper for a value plus isolated execution duration of the operation that
    produced it.
    """
    value: T
    duration_ms: float


async def _time_awaitable(awaitable: Awaitable[T]) -> TimedResult[T]:
    """
    Measure the isolated runtime of an awaitable and return both value and
    duration (milliseconds).
    """
    start = time.monotonic()
    value = await awaitable
    duration_ms = round((time.monotonic() - start) * 1000, 2)
    return TimedResult(value=value, duration_ms=duration_ms)


@dataclass(slots=True)
class CandidateVectorScores:
    """
    Holds all vector similarity scores for a single candidate movie.

    One instance per movie_id. Each field stores the cosine similarity score
    returned by Qdrant for a specific (collection, search_type) pair.
    Fields default to 0.0, meaning "this movie did not appear in that search."

    Naming convention: {collection_name}_score_{original|subquery}
    - "original" = searched using the user's raw query embedding
    - "subquery" = searched using the LLM-generated subquery embedding

    Anchor only has an "original" variant (it always uses the raw query).
    """
    anchor_score_original: float = 0.0
    plot_events_score_original: float = 0.0
    plot_events_score_subquery: float = 0.0
    plot_analysis_score_original: float = 0.0
    plot_analysis_score_subquery: float = 0.0
    viewer_experience_score_original: float = 0.0
    viewer_experience_score_subquery: float = 0.0
    watch_context_score_original: float = 0.0
    watch_context_score_subquery: float = 0.0
    narrative_techniques_score_original: float = 0.0
    narrative_techniques_score_subquery: float = 0.0
    production_score_original: float = 0.0
    production_score_subquery: float = 0.0
    reception_score_original: float = 0.0
    reception_score_subquery: float = 0.0


@dataclass(frozen=True, slots=True)
class SearchJob:
    """
    Represents a single Qdrant query to execute.

    This is the internal unit of work produced by a coordinator and consumed
    by _execute_and_merge. It carries everything needed to issue the query
    and write the result to the correct score field.

    Fields:
      - vector_name: which named vector in the Qdrant point to search against
                     (e.g. "anchor", "plot_events", "viewer_experience")
      - score_field: the exact attribute name on CandidateVectorScores to write to
                     (e.g. "plot_events_score_subquery"). This avoids any string
                     formatting at merge time — the coordinator decides upfront
                     where results go.
      - embedding:   the dense vector (list of 1536 floats) to use as the query.
      - limit:       max number of results to request from Qdrant.

    frozen=True because jobs are immutable once created — no reason to modify
    them after construction.
    """
    vector_name: VectorName
    score_field: str
    embedding: list[float]
    limit: int
    embedding_time_ms: float
    llm_generation_time_ms: Optional[float]


@dataclass(frozen=True, slots=True)
class SearchJobStats:
    """
    Diagnostic output for a single executed Qdrant search.
    Collected into the debug payload so we can see which searches ran,
    how many candidates each contributed, and where time was spent.

    `embedding_time_ms` and `llm_generation_time_ms` are isolated operation
    runtimes. `latency_ms` is the Qdrant query + merge time for this job.
    """
    score_field: str
    candidates_returned: int
    embedding_time_ms: float
    llm_generation_time_ms: Optional[float]
    latency_ms: float


@dataclass(slots=True)
class VectorSearchDebug:
    """
    Aggregated debug info for the entire vector search step.

    This is invaluable for:
      - Understanding which collections are actually contributing candidates
      - Identifying slow LLM calls or Qdrant queries in traces
      - Deciding whether to tune limits, HNSW params, or collection relevance
    """
    total_jobs_executed: int = 0
    total_candidates: int = 0
    per_job_stats: list[SearchJobStats] = field(default_factory=list)
    wall_clock_ms: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VectorSearchResult:
    """
    The final return type of run_vector_search().

    candidates: movie_id → CandidateVectorScores mapping. Every movie that
                appeared in at least one Qdrant search has an entry here.
                Score fields that weren't populated remain 0.0.
    debug:      timing, counts, and per-job diagnostics.
    """
    candidates: dict[int, CandidateVectorScores]
    debug: VectorSearchDebug


# ===========================================================================
# SECTION 3: QDRANT FILTER CONSTRUCTION
# ===========================================================================

def build_qdrant_filter(metadata_filters: MetadataFilters) -> Optional[Filter]:
    """
    Translate the application-level MetadataFilters into a Qdrant Filter object.

    This is a pure function with no side effects — easy to unit test.

    The returned Filter uses a `must` clause, meaning ALL conditions must be
    satisfied (AND logic). Each non-None field in MetadataFilters becomes one
    FieldCondition:
      - Range fields (release_ts, runtime_minutes, maturity_rank): use Qdrant's
        Range condition with gte/lte. Only the non-None bound is included, so
        "min_release_ts=X with max_release_ts=None" becomes gte=X with no upper bound.
      - Array match fields (genres, watch_offer_keys): use MatchAny, which means
        "the point's array must contain at least one of these values" (OR within
        the single condition, AND across conditions).

    Returns None if no filters are active, which tells Qdrant "no filtering."
    """
    conditions: list[FieldCondition] = []

    # --- Release date range (unix timestamp in seconds) ---
    # Only build the condition if at least one bound is set.
    if metadata_filters.min_release_ts is not None or metadata_filters.max_release_ts is not None:
        conditions.append(
            FieldCondition(
                key="release_ts",
                range=Range(
                    gte=metadata_filters.min_release_ts,  # None is fine — Qdrant ignores it
                    lte=metadata_filters.max_release_ts,
                ),
            )
        )

    # --- Runtime range (minutes) ---
    if metadata_filters.min_runtime is not None or metadata_filters.max_runtime is not None:
        conditions.append(
            FieldCondition(
                key="runtime_minutes",
                range=Range(
                    gte=metadata_filters.min_runtime,
                    lte=metadata_filters.max_runtime,
                ),
            )
        )

    # --- Maturity rating range (ordinal integer: 0=G, 1=PG, ..., 4=NC-17) ---
    if metadata_filters.min_maturity_rank is not None or metadata_filters.max_maturity_rank is not None:
        conditions.append(
            FieldCondition(
                key="maturity_rank",
                range=Range(
                    gte=metadata_filters.min_maturity_rank,
                    lte=metadata_filters.max_maturity_rank,
                ),
            )
        )

    # --- Genre IDs (match any — "movie must have at least one of these genres") ---
    # genre_ids in the Qdrant payload are integers (TMDB genre IDs).
    # The Genre enum values are strings, so we need to map them. If your Genre enum
    # values are already integers, adjust accordingly. This assumes you have a
    # GENRE_TO_ID mapping somewhere — replace with your actual mapping.
    if metadata_filters.genres is not None and len(metadata_filters.genres) > 0:
        deduped_genre_ids = list(dict.fromkeys(g.genre_id for g in metadata_filters.genres))
        conditions.append(
            FieldCondition(
                key="genre_ids",
                match=MatchAny(any=deduped_genre_ids),
            )
        )

    # --- Watch offer keys (encoded provider_id * 100 + method_code) ---
    if metadata_filters.watch_offer_keys is not None and len(metadata_filters.watch_offer_keys) > 0:
        deduped_watch_offer_keys = list(dict.fromkeys(metadata_filters.watch_offer_keys))
        conditions.append(
            FieldCondition(
                key="watch_offer_keys",
                match=MatchAny(any=deduped_watch_offer_keys),
            )
        )

    # If no conditions were added, return None (no filtering).
    # Qdrant's query API accepts filter=None to mean "search all points."
    if not conditions:
        return None

    # All conditions are ANDed together via the `must` clause.
    return Filter(must=conditions)


# ===========================================================================
# SECTION 4: QDRANT SEARCH EXECUTION + CANDIDATE MERGE
# ===========================================================================

async def _execute_and_merge(
    job: SearchJob,
    qdrant_filter: Optional[Filter],
    qdrant_client: AsyncQdrantClient,
    candidates: dict[int, CandidateVectorScores],
    search_params: SearchParams,
) -> SearchJobStats:
    """
    Execute a single Qdrant vector search and merge results into the shared
    candidates dict.

    This is the workhorse function called by every coordinator. It:
      1. Sends the query to Qdrant (async — yields control here).
      2. Iterates over results and writes scores to the candidates dict
         (sync — no await in the loop, so this is safe for the shared dict
         under asyncio's single-threaded model).
      3. Returns diagnostic stats for the debug payload.

    The candidates dict is mutated in place. For any movie_id that already
    exists (because a different search found it too), we just write to a
    different score_field on the same CandidateVectorScores instance. For
    new movie_ids, we create a fresh CandidateVectorScores (all 0.0) and
    set the one relevant field.

    Important: we request with_payload=False and with_vector=False because
    we only need (id, score) pairs. Payload data is fetched from Postgres
    later during enrichment. This keeps Qdrant responses small and fast.
    """
    start = time.monotonic()

    # --- Issue the Qdrant query ---
    # query_points is the modern Query API method on the async client.
    # `using` specifies which named vector in the point to compare against.
    #
    # NOTE ON PARAMETER NAMES: The qdrant-client Python library has evolved
    # across versions. Verify these parameter names against YOUR installed
    # version (pip show qdrant-client):
    #   - The filter parameter may be called `query_filter` instead of `filter`
    #     in some client versions.
    #   - `with_vectors` (plural) is used in recent versions; older versions
    #     may use `with_vector` (singular).
    #   - `query_points()` is the Query API method (qdrant-client >= 1.7).
    #     Older versions may require `search()` with slightly different params.
    # If you hit AttributeError or TypeError, check the method signature in
    # your installed version's source or docs.
    results = await qdrant_client.query_points(
        collection_name=COLLECTION_ALIAS,
        query=job.embedding,
        using=job.vector_name.value,
        query_filter=qdrant_filter,
        limit=job.limit,
        with_payload=False,
        with_vectors=False,
        search_params=search_params,
    )

    # --- Merge results into the shared candidates dict ---
    # CRITICAL: No `await` inside this loop. This guarantees that no other
    # coroutine can touch `candidates` while we're writing, because asyncio
    # only switches coroutines at await points.
    for point in results.points:
        movie_id: int = point.id
        score: float = point.score

        if movie_id not in candidates:
            # First time we've seen this movie across all searches.
            # Create a fresh scores object — all fields start at 0.0.
            candidates[movie_id] = CandidateVectorScores()

        # Write this search's score to the exact field that the coordinator
        # specified when it created the SearchJob. For example, if this job's
        # score_field is "plot_events_score_subquery", we set that attribute
        # to the cosine similarity score Qdrant returned.
        setattr(candidates[movie_id], job.score_field, score)

    elapsed_ms = (time.monotonic() - start) * 1000

    return SearchJobStats(
        score_field=job.score_field,
        candidates_returned=len(results.points),
        embedding_time_ms=job.embedding_time_ms,
        llm_generation_time_ms=job.llm_generation_time_ms,
        latency_ms=round(elapsed_ms, 2),
    )


# ===========================================================================
# SECTION 5: COORDINATOR TASKS
#
# Each coordinator is a small async function that:
#   1. Awaits its specific LLM dependency (weight or subquery result)
#   2. Decides whether to proceed or short-circuit
#   3. If proceeding, builds a SearchJob and calls _execute_and_merge
#
# There are three types:
#   - Anchor: no LLM dependency, just needs the original query embedding
#   - Original-query: needs the weight result + original query embedding
#   - Subquery: needs the subquery result + its own embedding
# ===========================================================================

async def _coordinate_anchor_search(
    original_embedding_task: asyncio.Task[TimedResult[list[list[float]]]],
    qdrant_filter: Optional[Filter],
    qdrant_client: AsyncQdrantClient,
    candidates: dict[int, CandidateVectorScores],
    search_params: SearchParams,
    limit: int,
) -> SearchJobStats | str | None:
    """
    Coordinator for the anchor vector search.

    The anchor search is special:
      - It ALWAYS runs (no relevance check — anchor is always relevant).
      - It ONLY uses the original query embedding (no LLM-generated subquery).
      - It has no LLM dependency, so it can fire as soon as the original
        query embedding is ready (which is typically very fast — either a
        Redis cache hit or a single OpenAI API call).

    This makes anchor the first search to complete in almost all cases,
    giving us an early baseline set of candidates.
    """
    try:
        # Wait for the original query embedding to be ready.
        # This is the shared embedding task — multiple coordinators await it,
        # but the actual embedding work only happens once.
        embedding_result = await original_embedding_task

        # generate_vector_embedding returns a list of embeddings (one per input text).
        # We only passed one text (the original query), so take index 0.
        original_embedding = embedding_result.value[0]

        job = SearchJob(
            vector_name=VectorName.ANCHOR,
            score_field="anchor_score_original",
            embedding=original_embedding,
            limit=limit,
            embedding_time_ms=embedding_result.duration_ms,
            llm_generation_time_ms=None,
        )

        return await _execute_and_merge(
            job=job,
            qdrant_filter=qdrant_filter,
            qdrant_client=qdrant_client,
            candidates=candidates,
            search_params=search_params,
        )

    except Exception as e:
        # Anchor failure is more concerning than other collections because it's
        # the primary retrieval channel. Return an error string so the orchestrator
        # captures it in debug.errors — the other collections may still provide results.
        return f"Anchor search failed: {e}"


async def _coordinate_original_search(
    collection_name: VectorName,
    weight_task: asyncio.Task[TimedResult[Optional[VectorCollectionWeightData]]],
    original_embedding_task: asyncio.Task[TimedResult[list[list[float]]]],
    qdrant_filter: Optional[Filter],
    qdrant_client: AsyncQdrantClient,
    candidates: dict[int, CandidateVectorScores],
    search_params: SearchParams,
    limit: int,
) -> SearchJobStats | str | None:
    """
    Coordinator for a single collection's ORIGINAL-QUERY search.

    This runs the user's raw query embedding against the collection's named
    vector. It's gated on the weight LLM result:
      - If weight.relevance is NOT_RELEVANT → skip (return None).
      - Otherwise → wait for the original embedding, then search.

    This exists as a SEPARATE task from the subquery coordinator because the
    weight and subquery LLM calls resolve at different times. If the weight
    comes back at t=600ms saying "relevant" but the subquery doesn't finish
    until t=1000ms, we don't want the original-query search to wait for
    the subquery — it can fire at t=600ms using the already-available
    original embedding.
    """
    try:
        # --- Gate 1: Wait for the weight LLM result ---
        weight_timed = await weight_task
        weight_result = weight_timed.value

        # If the LLM call failed entirely (returned None), we can't determine
        # relevance. Treat as not relevant to avoid noisy results.
        if weight_result is None:
            print(
                f"Weight LLM returned None for {collection_name.value}, "
                f"skipping original-query search."
            )
            return None

        # If the collection is not relevant to this query, skip the search
        # entirely. This saves a Qdrant query + HNSW traversal + disk rescore.
        if weight_result.relevance == RelevanceSize.NOT_RELEVANT.value:
            print(
                f"Weight LLM returned not_relevant for {collection_name.value}, "
                f"skipping original-query search."
            )
            return None

        # --- Gate 2: Wait for the original query embedding ---
        # This is the same shared task that anchor and all other original-query
        # coordinators await. The embedding work only runs once; subsequent
        # awaits return the cached result instantly.
        embedding_result = await original_embedding_task
        original_embedding = embedding_result.value[0]

        # Build the search job. The score_field encodes both the collection
        # name and the search type ("original") so the merge step knows
        # exactly where to write the result.
        job = SearchJob(
            vector_name=collection_name,
            score_field=f"{collection_name.value}_score_original",
            embedding=original_embedding,
            limit=limit,
            embedding_time_ms=embedding_result.duration_ms,
            llm_generation_time_ms=weight_timed.duration_ms,
        )

        return await _execute_and_merge(
            job=job,
            qdrant_filter=qdrant_filter,
            qdrant_client=qdrant_client,
            candidates=candidates,
            search_params=search_params,
        )

    except Exception as e:
        return f"Original-query search failed for {collection_name.value}: {e}"


async def _coordinate_subquery_search(
    collection_name: VectorName,
    subquery_task: asyncio.Task[TimedResult[Optional[VectorCollectionSubqueryData]]],
    embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]],
    qdrant_filter: Optional[Filter],
    qdrant_client: AsyncQdrantClient,
    candidates: dict[int, CandidateVectorScores],
    search_params: SearchParams,
    limit: int,
) -> SearchJobStats | str | None:
    """
    Coordinator for a single collection's SUBQUERY search.

    This runs an LLM-generated subquery embedding against the collection's
    named vector. The subquery is a rewritten version of the user's query
    optimized to match the specific content embedded in this vector space
    (e.g. the plot_events subquery focuses on literal plot events, while
    the viewer_experience subquery focuses on emotional descriptors).

    It's gated on the subquery LLM result:
      - If the subquery LLM returned None (failure) → skip.
      - If relevant_subquery_text is None → skip (LLM decided no subquery
        is useful for this collection).
      - Otherwise → embed the subquery text, then search.

    This is independent of the weight result — a collection could have a
    subquery but be weighted "not_relevant" (the original-query coordinator
    handles that case). Conversely, a collection could be "highly relevant"
    but have no useful subquery text.
    """
    try:
        # --- Gate: Wait for the subquery LLM result ---
        subquery_timed = await subquery_task
        subquery_result = subquery_timed.value

        # If the LLM call failed, we have no subquery text to embed.
        if subquery_result is None:
            print(
                f"Subquery LLM returned None for {collection_name.value}, "
                f"skipping subquery search."
            )
            return None

        # If the LLM decided no subquery is useful for this collection,
        # relevant_subquery_text will be None. This is normal and expected —
        # not every collection is relevant to every query.
        if subquery_result.relevant_subquery_text is None:
            print(
                f"Subquery LLM returned relevant_subquery_text as None for {collection_name.value}, "
                f"skipping subquery search."
            )
            return None

        # --- Embed the subquery text ---
        # Unlike the original query embedding (which is shared and started eagerly),
        # each subquery embedding is unique to its collection and can only start
        # after the subquery LLM call resolves. This is the main source of latency
        # in the subquery path — the OpenAI embedding call typically takes 50-200ms.
        #
        # We embed as a single-element list because generate_vector_embedding
        # accepts a list of texts and returns a list of embeddings.
        embedding_result = await _time_awaitable(embed_fn([subquery_result.relevant_subquery_text]))
        subquery_embedding = embedding_result.value[0]

        job = SearchJob(
            vector_name=collection_name,
            score_field=f"{collection_name.value}_score_subquery",
            embedding=subquery_embedding,
            limit=limit,
            embedding_time_ms=embedding_result.duration_ms,
            llm_generation_time_ms=subquery_timed.duration_ms,
        )

        return await _execute_and_merge(
            job=job,
            qdrant_filter=qdrant_filter,
            qdrant_client=qdrant_client,
            candidates=candidates,
            search_params=search_params,
        )

    except Exception as e:
        return f"Subquery search failed for {collection_name.value}: {e}"


# ===========================================================================
# SECTION 6: MAIN ORCHESTRATOR
# ===========================================================================

async def run_vector_search(
    query: str,
    metadata_filters: "MetadataFilters",
    qdrant_client: AsyncQdrantClient,
    original_limit: int = 2000,
    subquery_limit: int = 2000,
    anchor_limit: int = 2000,
) -> VectorSearchResult:
    """
    Main entry point for the vector search step.

    Orchestrates the full pipeline:
      1. Fire all LLM calls and the original query embedding concurrently.
      2. Launch coordinator tasks that stream searches as dependencies resolve.
      3. Collect all results into a merged candidates dict.
      4. Return candidates + debug info.

    Parameters:
      query:            The user's original search query text.
      metadata_filters: Pre-built hard filters (release date, runtime, genres, etc.).
                        Passed in from the caller — this module does NOT own filter
                        construction from the QU output.
      qdrant_client:    An AsyncQdrantClient instance connected to the Qdrant server.
      original_limit:   Max candidates to retrieve per original-query search (default 2000).
      subquery_limit:   Max candidates to retrieve per subquery search (default 2000).
      anchor_limit:     Max candidates to retrieve from the anchor search (default 2000).

    Returns:
      VectorSearchResult containing:
        - candidates: dict[int, CandidateVectorScores] — one entry per movie
          that appeared in any Qdrant search result.
        - debug: timing, per-job stats, error messages.
    """
    wall_clock_start = time.monotonic()

    if COLLECTION_ALIAS is None:
        raise RuntimeError(
            "QDRANT_COLLECTION_ALIAS environment variable is not set. "
            "Cannot execute vector search without a collection target."
        )

    # The shared candidates dict. All coordinators write to this as their Qdrant
    # results come back. Safe under asyncio because writes contain no `await`.
    candidates: dict[int, CandidateVectorScores] = {}

    # The debug container. Coordinators append their stats here after each search.
    # List.append() is atomic in CPython, and we only call it after await points
    # (not during dict writes), so this is also safe.
    debug = VectorSearchDebug()

    # -----------------------------------------------------------------------
    # PHASE A: Build the Qdrant filter (sync, instant)
    # -----------------------------------------------------------------------
    # This is pure computation — no I/O. We build it once and pass the same
    # filter object to every Qdrant query. All searches apply identical hard
    # filters because they all serve the same user request.
    qdrant_filter = build_qdrant_filter(metadata_filters)

    # -----------------------------------------------------------------------
    # PHASE B: Fire all non-dependent async work
    # -----------------------------------------------------------------------

    # --- B1: Embed the original query text ---
    # This has NO LLM dependency — we can start it immediately.
    # The resulting embedding is shared by the anchor search and all 7
    # original-query collection searches. We wrap it in create_task so
    # it starts running in the background while we set up the LLM tasks.
    #
    # generate_vector_embedding accepts a list of texts. We pass a single-
    # element list and will extract index [0] when consuming the result.
    original_embedding_task = asyncio.create_task(
        _time_awaitable(generate_vector_embedding([query])),
        name="embed_original_query",
    )

    # --- B2: Fire all 14 LLM calls (7 weights + 7 subqueries) ---
    # Each collection gets two independent LLM calls. We store the tasks
    # in dicts keyed by VectorName so coordinators can look up
    # their specific dependency.
    weight_tasks: dict[VectorName, asyncio.Task[TimedResult[Optional[VectorCollectionWeightData]]]] = {}
    subquery_tasks: dict[VectorName, asyncio.Task[TimedResult[Optional[VectorCollectionSubqueryData]]]] = {}

    for collection_name in VectorName:
        # Weight LLM call: determines if this collection is relevant to the query.
        # Returns VectorCollectionWeightData with a .relevance field.
        weight_tasks[collection_name] = asyncio.create_task(
            _time_awaitable(create_single_vector_weight_async(query, collection_name)),
            name=f"weight_{collection_name.value}",
        )

        # Subquery LLM call: generates an optimized query string tailored to
        # this collection's embedding space. Returns VectorCollectionSubqueryData
        # with a .relevant_subquery_text field (may be None if the LLM decides
        # no subquery is useful).
        subquery_tasks[collection_name] = asyncio.create_task(
            _time_awaitable(create_single_vector_subquery_async(query, collection_name)),
            name=f"subquery_{collection_name.value}",
        )

    # At this point we have 15 tasks in flight:
    #   1 embedding task + 7 weight tasks + 7 subquery tasks
    # None of them block each other. The event loop will execute them
    # concurrently (and for the LLM calls, the actual HTTP requests are
    # truly concurrent since they're I/O-bound).

    # -----------------------------------------------------------------------
    # PHASE C: Launch coordinator tasks
    # -----------------------------------------------------------------------
    # Each coordinator awaits its specific dependency, then fires a Qdrant
    # search and merges the results. We create all coordinators at once and
    # let asyncio.gather run them concurrently.

    coordinator_tasks: list[asyncio.Task[SearchJobStats | str | None]] = []

    # --- C1: Anchor coordinator ---
    # No LLM dependency — only needs the original embedding.
    # This will be the first search to fire in almost all cases.
    coordinator_tasks.append(
        asyncio.create_task(
            _coordinate_anchor_search(
                original_embedding_task=original_embedding_task,
                qdrant_filter=qdrant_filter,
                qdrant_client=qdrant_client,
                candidates=candidates,
                search_params=_QDRANT_SEARCH_PARAMS,
                limit=anchor_limit,
            ),
            name="coord_anchor_original",
        )
    )

    # --- C2: Per-collection original-query coordinators ---
    # Each waits for its weight task. If relevant, it awaits the shared
    # original embedding (which is likely already resolved by then) and
    # fires a Qdrant search.
    for collection_name in VectorName:
        # ANCHOR IS ALREADY RUNNING IN C1
        if collection_name == VectorName.ANCHOR:
            continue
        
        coordinator_tasks.append(
            asyncio.create_task(
                _coordinate_original_search(
                    collection_name=collection_name,
                    weight_task=weight_tasks[collection_name],
                    original_embedding_task=original_embedding_task,
                    qdrant_filter=qdrant_filter,
                    qdrant_client=qdrant_client,
                    candidates=candidates,
                    search_params=_QDRANT_SEARCH_PARAMS,
                    limit=original_limit,
                ),
                name=f"coord_{collection_name.value}_original",
            )
        )

    # --- C3: Per-collection subquery coordinators ---
    # Each waits for its subquery task. If the subquery text is non-null,
    # it embeds the text (a new OpenAI call) and then fires a Qdrant search.
    # This is fully independent of the weight coordinator for the same
    # collection — neither blocks the other.
    for collection_name in VectorName:
        # ANCHOR IS ALREADY RUNNING IN C1
        if collection_name == VectorName.ANCHOR:
            continue

        coordinator_tasks.append(
            asyncio.create_task(
                _coordinate_subquery_search(
                    collection_name=collection_name,
                    subquery_task=subquery_tasks[collection_name],
                    embed_fn=generate_vector_embedding,
                    qdrant_filter=qdrant_filter,
                    qdrant_client=qdrant_client,
                    candidates=candidates,
                    search_params=_QDRANT_SEARCH_PARAMS,
                    limit=subquery_limit,
                ),
                name=f"coord_{collection_name.value}_subquery",
            )
        )

    # Total coordinator tasks: 1 (anchor) + 7 (original) + 7 (subquery) = 15.
    # Many will short-circuit early (weight=not_relevant, subquery_text=None).

    # Collect every spawned task so we can cancel stragglers if this coroutine
    # is itself cancelled (e.g. by a request timeout). Without this, in-flight
    # LLM and embedding tasks would continue running as orphaned background tasks.
    all_tasks: list[asyncio.Task] = (
        [original_embedding_task]
        + list(weight_tasks.values())
        + list(subquery_tasks.values())
        + coordinator_tasks
    )

    # -----------------------------------------------------------------------
    # PHASE D: Wait for all coordinators to complete
    # -----------------------------------------------------------------------
    # asyncio.gather runs all tasks concurrently and returns when ALL are done.
    # return_exceptions=True means if a coordinator raises an unhandled exception
    # (which shouldn't happen since they all have try/except), it's returned as
    # the result rather than propagated — this prevents one failure from canceling
    # all other tasks.
    #
    # The try/finally guarantees that if this coroutine is cancelled externally,
    # all in-flight tasks are cancelled rather than left running as orphans.
    coordinator_results: list[SearchJobStats | str | None | BaseException]
    try:
        coordinator_results = await asyncio.gather(*coordinator_tasks, return_exceptions=True)
    finally:
        for task in all_tasks:
            if not task.done():
                task.cancel()

    # -----------------------------------------------------------------------
    # PHASE E: Assemble debug info
    # -----------------------------------------------------------------------
    for result in coordinator_results:
        if isinstance(result, BaseException):
            # An unhandled exception slipped through a coordinator's try/except.
            # This should be rare, but we capture it in debug output rather than
            # crashing the entire search.
            debug.errors.append(f"Unhandled coordinator exception: {result}")
        elif isinstance(result, str):
            # A coordinator caught an exception and returned it as an error string.
            # This is the normal error path — the coordinator handled it gracefully
            # and returned a message rather than raising.
            debug.errors.append(result)
        elif result is not None:
            # result is a SearchJobStats from a coordinator that successfully
            # executed a Qdrant search. None means the coordinator short-circuited
            # (irrelevant weight or null subquery) — that's normal, not an error.
            debug.per_job_stats.append(result)

    debug.total_jobs_executed = len(debug.per_job_stats)
    debug.total_candidates = len(candidates)
    debug.wall_clock_ms = round((time.monotonic() - wall_clock_start) * 1000, 2)

    print(
        f"Vector search complete: {debug.total_jobs_executed} jobs, "
        f"{debug.total_candidates} unique candidates, "
        f"{debug.wall_clock_ms}ms wall clock"
    )

    return VectorSearchResult(
        candidates=candidates,
        debug=debug,
    )
