"""
Vector search implementation using Weighted Reciprocal Rank Fusion (RRF).

This module implements a vector-search-only baseline that:
- Embeds the user query once
- Searches three Chroma collections (anchor, content, vibe)
- Fuses the three ranked lists with Weighted RRF
- Breaks ties with average cosine similarity
"""

import os
from pathlib import Path
from typing import TypedDict, Optional
import numpy as np
from dotenv import load_dotenv

from .vectorize import search_similar_vectors, openai_client, normalize_vector
from .enums import VectorCollectionName

# Load environment variables
load_dotenv()

# Default database path: implementation/chroma_db/ relative to this file
_DEFAULT_DB_PATH = Path(__file__).parent / "chroma_db"


# ===== DATA STRUCTURES =====

class RankedResult(TypedDict):
    """Represents a single result from a ranked list."""
    movie_id: str  # TMDB ID as string
    rank: int
    distance: float
    metadata: dict
    document: str


class AxisScoreResults(TypedDict):
    """Represents scoring results for a single collection axis."""
    rank: int
    rrf_term: float
    similarity: float


class CandidateScore(TypedDict):
    """Represents scoring information for a candidate movie."""
    movie_id: str  # TMDB ID as string
    rrf_score: float
    avg_sim: float
    axes: dict[str, AxisScoreResults]
    metadata: dict


# ===== QUERY EMBEDDING =====

def embed_query(query_text: str) -> list[float]:
    """
    Embeds a query text using OpenAI's text-embedding-3-small model.
    
    This function creates a single embedding vector that will be used to search
    all three Chroma collections. The same embedding model is used across all
    collections; only the stored movie text representations differ.
    
    Args:
        query_text: The user's search query as a string
        
    Returns:
        List of floats representing the query embedding vector (1536 dimensions)
        
    Raises:
        ValueError: If query_text is empty or embedding fails
    """
    if not query_text or not query_text.strip():
        raise ValueError("query_text cannot be empty")
    
    # Create embedding using OpenAI's small embedder model
    # text-embedding-3-small produces 1536-dimensional vectors
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query_text.strip()
    )
    embedding = response.data[0].embedding
    return embedding


# ===== COLLECTION SEARCHING =====

def search_collection_and_build_ranks(
    query_vector: list[float],
    collection_name: VectorCollectionName,
    n_results: int,
    db_path: str | Path = _DEFAULT_DB_PATH,
    ids_to_filter_out: Optional[list[str]] = None
) -> dict[str,RankedResult]:
    """
    Searches a Chroma collection and builds ranked results with rank maps.
    
    This function queries a collection with the query vector, retrieves top-K
    results, and builds both a ranked list and a rank map for efficient lookup.
    
    Args:
        query_vector: The query embedding vector
        collection_name: Name of the ChromaDB collection to search
        n_results: Number of top results to retrieve (top-K)
        db_path: Path to the ChromaDB database directory
        ids_to_filter_out: Optional list of TMDB movie IDs (as strings) to exclude from results
        
    Returns:
        Dictionary mapping tmdb_id (as string) to RankedResult
        
    Note:
        All movie IDs are TMDB IDs stored as strings.
    """
    # Search the collection using the existing search function
    results = search_similar_vectors(
        query_vector=query_vector,
        collection_name=collection_name,
        db_path=db_path,
        n_results=n_results,
        ids_to_filter_out=ids_to_filter_out
    )
    
    # Build ranked list and rank map
    ranks_map: dict[str, RankedResult] = {}

    for i, (movie_id, metadata, distance, document) in enumerate(zip(results.ids, results.metadatas, results.distances, results.documents), start=1):        
        ranks_map[movie_id] = {
            "movie_id": movie_id,
            "rank": i,
            "distance": float(distance) if distance is not None else 0.0,
            "metadata": metadata,
            "document": document if document else ""
        }
    
    return ranks_map


# ===== COSINE SIMILARITY CALCULATION =====

def cosine_similarity(vec1: list[float], vec2: list[float], eps: float = 1e-8) -> float:
    """
    Computes cosine similarity between two vectors.
    
    This function normalizes both vectors to unit length before computing cosine
    similarity, ensuring consistent results regardless of input vector magnitudes.
    Cosine similarity ranges from -1 to 1, where 1 indicates identical vectors
    and -1 indicates opposite vectors. For normalized embeddings, this typically
    ranges from 0 to 1.
    
    Args:
        vec1: First vector as list of floats
        vec2: Second vector as list of floats
        eps: Small epsilon to avoid division by zero during normalization
        
    Returns:
        Cosine similarity score as a float
    """
    vec1_normalized = np.array(normalize_vector(vector=vec1, eps=eps))
    vec2_normalized = np.array(normalize_vector(vector=vec2, eps=eps))
    
    # Cosine similarity is dot product of normalized vectors
    dot_product = np.dot(vec1_normalized, vec2_normalized)
    
    return float(dot_product)


def get_movie_embeddings(
    movie_ids: list[str],
    collection_name: VectorCollectionName,
    db_path: str | Path = _DEFAULT_DB_PATH
) -> Optional[list[list[float]]]:
    """
    Retrieves the stored embedding vectors for movies from a collection.
    
    Args:
        movie_ids: List of TMDB movie IDs (as strings) to look up
        collection_name: Name of the ChromaDB collection
        db_path: Path to the ChromaDB database directory
        
    Returns:
        List of embedding vectors (each as a list of floats), one per movie_id, or None if not found.
        The order matches the order of movie_ids provided.
        
    Note:
        Movie IDs are TMDB IDs stored as strings in ChromaDB.
    """
    import chromadb
    from chromadb.config import Settings
    
    db_path = Path(db_path) if isinstance(db_path, str) else db_path
    
    if not db_path.exists():
        print(f"Database not found at path: {db_path}")
        return None
    
    try:
        chroma_client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        collection = chroma_client.get_collection(name=collection_name.value)
        
        # Fetch the specific vectors by IDs
        results = collection.get(ids=list(movie_ids), include=['embeddings'])
        
        if results["ids"] and len(results["embeddings"]) > 0:
            embeddings = results["embeddings"]
            # Convert to list if it's a NumPy array to avoid boolean evaluation issues
            if isinstance(embeddings, np.ndarray):
                return embeddings.tolist()
            return list(embeddings) if embeddings else None
        return None
    except Exception as e:
        print(f"Exception in get_movie_embeddings: {e}")
        return None


def compute_cosine_similarities_for_candidates(
    query_vectors_by_axis: dict[str, list[float]],
    candidate_ids: list[str],
    db_path: str | Path = _DEFAULT_DB_PATH
) -> dict[str, dict[str, float]]:
    """
    Computes cosine similarities for all candidates across all collections.
    
    For each candidate movie, this function retrieves its embeddings from all
    collections and computes cosine similarity with the appropriate query vector.
    Uses strict policy: missing embeddings are treated as 0.0 similarity.
    
    Args:
        query_vectors_by_axis: Dictionary mapping collection names to their query embedding vectors
        candidate_ids: List of TMDB movie IDs (as strings) to compute similarities for
        db_path: Path to the ChromaDB database directory
        
    Returns:
        Dictionary mapping tmdb_id (as string) to a dictionary of similarity scores for each axis (floats, defaults to 0.0)
        
    Note:
        All movie IDs are TMDB IDs stored as strings.
    """
    
    similarities: dict[str, dict[str, float]] = {}

    # Fetch embeddings of every candidate from each collection
    embeddings_by_collection: dict[str, dict[str, list[float]]] = {}
    for collection_name, query_vector in query_vectors_by_axis.items():
        embeddings = get_movie_embeddings(candidate_ids, VectorCollectionName(collection_name), db_path)
        collection_embeddings = {}
        for movie_id, embedding in zip(candidate_ids, embeddings):
            collection_embeddings[movie_id] = embedding
        embeddings_by_collection[collection_name] = collection_embeddings


    # Compute cosine similarities for each candidate
    for movie_id in candidate_ids:
        axes_similarities: dict[str, float] = {}
        running_similarity_sum = 0.0

        # step 1 - convert embeddings per axis to similarities per axis
        for collection_name, query_vector in query_vectors_by_axis.items():
            candidate_embedding_for_axis = embeddings_by_collection[collection_name].get(movie_id)
            similarity = cosine_similarity(query_vector, candidate_embedding_for_axis) if candidate_embedding_for_axis else 0.0
            axes_similarities[collection_name] = similarity
            running_similarity_sum += similarity

        # step 2 - save to overall dict to return
        similarities[movie_id] = axes_similarities
    
    return similarities


# ===== RRF CALCULATION =====

def compute_rrf_terms(
    ranks_by_axis: dict[str, Optional[int]],
    weights_by_axis: dict[str, float],
    rrf_k: float
) -> dict[str, float]:
    """
    Computes Weighted Reciprocal Rank Fusion (RRF) terms for each axis.
    
    RRF uses ranks only (not cosine values) for the fused score. The formula is:
    termX = wX / (rrf_k + rX) if present, else 0
    
    Args:
        ranks_by_axis: Dictionary mapping axis names to their ranks (1-indexed, or None if missing)
        weights_by_axis: Dictionary mapping axis names to their weights
        rrf_k: Rank dampening constant (smaller = more top-heavy)
        
    Returns:
        Dictionary mapping axis names to their RRF terms (float values)
    """
    rrf_terms: dict[str, float] = {}
    
    # Compute RRF term for each axis
    for axis_name, rank in ranks_by_axis.items():
        weight = weights_by_axis.get(axis_name, 0.0)
        rrf_terms[axis_name] = weight / (rrf_k + rank) if rank is not None else 0.0
    
    return rrf_terms


# ===== MAIN SEARCH FUNCTION =====

def fused_vector_search(
    query_text: Optional[str] = None,
    query_movie_id: Optional[str] = None,
    n_candidates_per_axis: int = 50,
    rrf_k: float = 60.0,
    weights: Optional[dict[VectorCollectionName, float]] = None,
    return_top_n: int = 20,
    db_path: str | Path = _DEFAULT_DB_PATH
) -> dict:
    """
    Performs fused vector search across eight Chroma collections using Weighted RRF.
    
    This is the main search function that implements the complete algorithm:
    1. Embeds the query (text) or fetches movie vectors (movie mode)
    2. Searches each collection independently
    3. Builds candidate pool (union)
    4. Computes RRF scores
    5. Computes cosine similarities for tie-breaking
    6. Sorts and returns top-N results
    
    Args:
        query_text: User's search query (required if query_movie_id is None)
        query_movie_id: TMDB movie ID (as string) to use for "more like this" search (required if query_text is None)
        n_candidates_per_axis: Top-K to retrieve from each collection (default 50)
        rrf_k: Rank dampening constant for RRF (default 60.0)
        weights: Dictionary mapping VectorCollectionName enum to weight values (default 1.0 for all collections)
        return_top_n: Number of final results to return (default 20)
        db_path: Path to ChromaDB database directory
        
    Returns:
        Dictionary with keys:
        - 'fused_results': List of CandidateScore dictionaries, sorted by (rrf_score desc, avg_sim desc, movie_id asc). movie_id is TMDB ID as string.
        - 'raw_rank_maps_by_collection': Dictionary mapping collection names to rank maps (dict of tmdb_id to RankedResult)
        - 'raw_similarities_by_movie_id': Dictionary mapping tmdb_id (as string) to dictionary of similarity scores per collection
        
    Raises:
        ValueError: If neither query_text nor query_movie_id is provided, or both are provided
        
    Note:
        All movie IDs in this function refer to TMDB IDs (as strings), not IMDB IDs.
    """
    # Validate input: exactly one of query_text or query_movie_id must be provided
    if query_text is None and query_movie_id is None:
        raise ValueError("Either query_text or query_movie_id must be provided")
    if query_text is not None and query_movie_id is not None:
        raise ValueError("Only one of query_text or query_movie_id should be provided")
    
    # Determine which IDs to filter out (exclude the query movie itself in movie mode)
    ids_to_filter_out = [query_movie_id] if query_movie_id else None

    collection_names = list(VectorCollectionName)
    query_vectors_by_collection = {}
    
    # Step 1: Get query vectors
    if query_movie_id:
        # Movie mode: fetch stored vectors for each collection
        for collection_name in collection_names:
            embedding = get_movie_embeddings([query_movie_id], collection_name, db_path)[0]
            if not embedding:
                raise ValueError(f"Movie {query_movie_id} not found in one or more collections")
            query_vectors_by_collection[collection_name.value] = embedding
    else:
        # Text mode: embed query once and use for all collections
        embedded_text = embed_query(query_text)
        for collection_name in collection_names:
            query_vectors_by_collection[collection_name.value] = embedded_text
    

    # Step 2: Search each collection independently with collection-specific query vectors
    rank_maps_by_collection = {}
    for collection_name in collection_names:
        rank_maps_by_collection[collection_name.value] = search_collection_and_build_ranks(
            query_vectors_by_collection[collection_name.value], collection_name, n_candidates_per_axis, db_path, ids_to_filter_out
        )
    

    # Step 3: Build candidate pool (union)
    candidate_ids = set()
    for rank_map_by_collection in rank_maps_by_collection.values():
        candidate_ids.update(rank_map_by_collection.keys())
    

    # Step 4: Compute RRF scores for all candidates
    candidate_scores: list[CandidateScore] = []
    
    # Get metadata from any collection (they should be consistent)
    metadata_map: dict[str, dict] = {}
    for candidate_id in candidate_ids: 
        if candidate_id not in metadata_map:
            for rank_map_by_collection in rank_maps_by_collection.values():
                if candidate_id in rank_map_by_collection:
                    metadata_map[candidate_id] = rank_map_by_collection[candidate_id]['metadata']

    
    # Convert weights dictionary from VectorCollectionName enum keys to collection name string values
    # Default to 1.0 for any collection not specified in weights
    weights_by_axis: dict[str, float] = {}
    if weights is None:
        weights = {}
    
    for collection_name in collection_names:
        # Use provided weight or default to 1.0
        weight_value = weights.get(collection_name, 1.0)
        weights_by_axis[collection_name.value] = weight_value
    
    
    # Step 4: Compute RRF scores for all candidates
    rrf_terms_by_candidate: dict[str, dict[str, float]] = {}
    ranks_by_candidate: dict[str, dict[str, Optional[int]]] = {}

    print()

    for collection_name in collection_names:
        if collection_name.value in set([VectorCollectionName.DENSE_ANCHOR_VECTORS.value, VectorCollectionName.PLOT_EVENTS_VECTORS.value]):
            print(f"rank maps by collection {collection_name.value}: {rank_maps_by_collection[collection_name.value]}")
    
    print()
    for movie_id in candidate_ids:
        # Build ranks_by_axis dictionary
        ranks_by_axis = {}
        for collection_name in collection_names:
            rank = rank_maps_by_collection[collection_name.value].get(movie_id, {}).get('rank')
            if collection_name.value in set([VectorCollectionName.DENSE_ANCHOR_VECTORS.value, VectorCollectionName.PLOT_EVENTS_VECTORS.value]):
                print(f"rank for {collection_name.value} {movie_id}: {rank}")
            ranks_by_axis[collection_name.value] = rank
        
        ranks_by_candidate[movie_id] = ranks_by_axis
        print(f"ranks by axis for {movie_id}: {ranks_by_axis}")
        
        # Compute RRF terms
        rrf_terms = compute_rrf_terms(ranks_by_axis, weights_by_axis, rrf_k)
        rrf_terms_by_candidate[movie_id] = rrf_terms

        print(f"rrf terms for {movie_id}: {rrf_terms}")
        print()
    

    # Step 5: Compute cosine similarities for tie-breaking
    cosine_sims = compute_cosine_similarities_for_candidates(
        query_vectors_by_collection,
        candidate_ids, db_path
    )

    # Combine ranks, RRF terms, and similarities into AxisScoreResults structure
    for movie_id in candidate_ids:
        axes_results: dict[str, AxisScoreResults] = {}
        
        for collection_name in collection_names:
            rank = ranks_by_candidate[movie_id].get(collection_name.value)
            rrf_term = rrf_terms_by_candidate[movie_id].get(collection_name.value, 0.0)
            similarity = cosine_sims.get(movie_id, {}).get(collection_name.value, 0.0)
            
            axes_results[collection_name.value] = {
                'rank': rank,
                'rrf_term': rrf_term,
                'similarity': similarity
            }
        
        # Compute total RRF score from all axes
        rrf_score = sum(axis_result['rrf_term'] for axis_result in axes_results.values())

        # Compute average similarity across all axes
        similarities = [axis_result['similarity'] for axis_result in axes_results.values()]
        avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
        
        candidate_scores.append({
            'movie_id': movie_id,
            'rrf_score': rrf_score,
            'avg_sim': avg_sim,
            'axes': axes_results,
            'metadata': metadata_map.get(movie_id, {})
        })
    
    
    # Step 6: Sort by (rrf_score desc, avg_sim desc, movie_id asc)
    candidate_scores.sort(
        key=lambda x: (-x['rrf_score'], -x['avg_sim'], x['movie_id'])
    )
    
    # Return top-N results along with raw lists for visualization
    return {
        'fused_results': candidate_scores[:return_top_n],
        'raw_rank_maps_by_collection': rank_maps_by_collection,
        'raw_similarities_by_movie_id': cosine_sims,
    }

