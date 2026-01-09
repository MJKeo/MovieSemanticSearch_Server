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

from vectorize import search_similar_vectors, openai_client, normalize_vector

# Load environment variables
load_dotenv()


# ===== DATA STRUCTURES =====

class RankedResult(TypedDict):
    """Represents a single result from a ranked list."""
    movie_id: str
    rank: int
    similarity: float
    metadata: dict


class CandidateScore(TypedDict):
    """Represents scoring information for a candidate movie."""
    movie_id: str
    rrf_score: float
    avg_sim: float
    rank_anchor: Optional[int]
    rank_content: Optional[int]
    rank_vibe: Optional[int]
    term_anchor: float
    term_content: float
    term_vibe: float
    sim_anchor: float
    sim_content: float
    sim_vibe: float
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
    collection_name: str,
    n_results: int,
    db_path: str | Path = "./chroma_db",
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
        ids_to_filter_out: Optional list of movie IDs to exclude from results
        
    Returns:
        Dictionary mapping movie_id to RankedResult
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

    for i, (movie_id, metadata, distance) in enumerate(zip(results.ids, results.metadatas, results.distances), start=1):        
        ranks_map[movie_id] = {
            "movie_id": movie_id,
            "rank": i,
            "similarity": float(distance) if distance is not None else 0.0,
            "metadata": metadata
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
    collection_name: str,
    db_path: str | Path = "./chroma_db"
) -> Optional[list[float]]:
    """
    Retrieves the stored embedding vector for a movie from a collection.
    
    Args:
        movie_id: The movie ID to look up
        collection_name: Name of the ChromaDB collection
        db_path: Path to the ChromaDB database directory
        
    Returns:
        The embedding vector as a list of floats, or None if not found
    """
    import chromadb
    from chromadb.config import Settings
    
    db_path = Path(db_path) if isinstance(db_path, str) else db_path
    
    if not db_path.exists():
        return None
    
    try:
        chroma_client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        collection = chroma_client.get_collection(name=collection_name)
        
        # Fetch the specific vector by ID
        results = collection.get(ids=movie_ids, include=['embeddings'])
        
        if results["ids"] and len(results["embeddings"]) > 0:
            embeddings = results["embeddings"]
            # Convert to list if it's a NumPy array to avoid boolean evaluation issues
            if isinstance(embeddings, np.ndarray):
                return embeddings.tolist()
            return list(embeddings) if embeddings else None
        return None
    except Exception:
        return None


def compute_cosine_similarities_for_candidates(
    query_vector_anchor: list[float],
    query_vector_content: list[float],
    query_vector_vibe: list[float],
    candidate_ids: list[str],
    db_path: str | Path = "./chroma_db"
) -> dict[str, dict[str, float]]:
    """
    Computes cosine similarities for all candidates across all three collections.
    
    For each candidate movie, this function retrieves its embeddings from all
    three collections and computes cosine similarity with the appropriate query vector.
    Uses strict policy: missing embeddings are treated as 0.0 similarity.
    
    Args:
        query_vector_anchor: The query embedding vector for anchor collection
        query_vector_content: The query embedding vector for content collection
        query_vector_vibe: The query embedding vector for vibe collection
        candidate_ids: Set of movie IDs to compute similarities for
        db_path: Path to the ChromaDB database directory
        
    Returns:
        Dictionary mapping movie_id to a dict with keys:
        - 'sim_anchor': cosine similarity with anchor collection (or 0.0 if missing)
        - 'sim_content': cosine similarity with content collection (or 0.0 if missing)
        - 'sim_vibe': cosine similarity with vibe collection (or 0.0 if missing)
        - 'avg_sim': average of the three similarities
    """
    collection_configs = [
        ('anchor', 'dense_anchor_vectors', query_vector_anchor),
        ('content', 'dense_content_vectors', query_vector_content),
        ('vibe', 'dense_vibe_vectors', query_vector_vibe)
    ]
    
    similarities: dict[str, dict[str, float]] = {}

    # search for all candidate vectors simultaneously rather than one by one
    embeddings = get_movie_embeddings(candidate_ids, collection_configs, db_path)

    if not embeddings:
        return {}

    for movie_id, embedding in zip(candidate_ids, embeddings):
        sims = {}
        
        # Compute average similarity
        sim_anchor = sims.get('sim_anchor', 0.0)
        sim_content = sims.get('sim_content', 0.0)
        sim_vibe = sims.get('sim_vibe', 0.0)
        
        # Strict policy: always average all three (even if some are 0.0)
        avg_sim = (sim_anchor + sim_content + sim_vibe) / 3.0
        
        sims['avg_sim'] = avg_sim
        similarities[movie_id] = sims
    
    return similarities


# ===== RRF CALCULATION =====

def compute_rrf_score(
    rank_anchor: Optional[int],
    rank_content: Optional[int],
    rank_vibe: Optional[int],
    rrf_k: float,
    w_anchor: float,
    w_content: float,
    w_vibe: float
) -> dict[str, float]:
    """
    Computes Weighted Reciprocal Rank Fusion (RRF) score for a candidate movie.
    
    RRF uses ranks only (not cosine values) for the fused score. The formula is:
    termX = wX / (rrf_k + rX) if present, else 0
    rrf_score = termA + termC + termV
    
    Args:
        movie_id: The movie ID (for reference)
        rank_anchor: 1-indexed rank in anchor collection, or None if missing
        rank_content: 1-indexed rank in content collection, or None if missing
        rank_vibe: 1-indexed rank in vibe collection, or None if missing
        rrf_k: Rank dampening constant (smaller = more top-heavy)
        w_anchor: Weight for anchor collection
        w_content: Weight for content collection
        w_vibe: Weight for vibe collection
        
    Returns:
        Dictionary with keys:
        - 'rrf_score': total RRF score
        - 'term_anchor': RRF term for anchor collection
        - 'term_content': RRF term for content collection
        - 'term_vibe': RRF term for vibe collection
    """
    # Compute per-list RRF terms
    term_anchor = w_anchor / (rrf_k + rank_anchor) if rank_anchor is not None else 0.0
    term_content = w_content / (rrf_k + rank_content) if rank_content is not None else 0.0
    term_vibe = w_vibe / (rrf_k + rank_vibe) if rank_vibe is not None else 0.0
    
    # Sum to get total RRF score
    rrf_score = term_anchor + term_content + term_vibe
    
    return {
        'rrf_score': rrf_score,
        'term_anchor': term_anchor,
        'term_content': term_content,
        'term_vibe': term_vibe
    }


# ===== MAIN SEARCH FUNCTION =====

def fused_vector_search(
    query_text: Optional[str] = None,
    query_movie_id: Optional[str] = None,
    n_anchor: int = 50,
    n_content: int = 50,
    n_vibe: int = 50,
    rrf_k: float = 60.0,
    w_anchor: float = 1.0,
    w_content: float = 1.0,
    w_vibe: float = 1.0,
    return_top_n: int = 20,
    db_path: str | Path = "./chroma_db"
) -> dict:
    """
    Performs fused vector search across three Chroma collections using Weighted RRF.
    
    This is the main search function that implements the complete algorithm:
    1. Embeds the query (text) or fetches movie vectors (movie mode)
    2. Searches each collection independently
    3. Builds candidate pool (union)
    4. Computes RRF scores
    5. Computes cosine similarities for tie-breaking
    6. Sorts and returns top-N results
    
    Args:
        query_text: User's search query (required if query_movie_id is None)
        query_movie_id: Movie ID to use for "more like this" search (required if query_text is None)
        n_anchor: Top-K to retrieve from anchor collection
        n_content: Top-K to retrieve from content collection
        n_vibe: Top-K to retrieve from vibe collection
        rrf_k: Rank dampening constant for RRF (default 60.0)
        w_anchor: Weight for anchor collection (default 1.0)
        w_content: Weight for content collection (default 1.0)
        w_vibe: Weight for vibe collection (default 1.0)
        return_top_n: Number of final results to return (default 20)
        db_path: Path to ChromaDB database directory
        
    Returns:
        Dictionary with keys:
        - 'fused_results': List of CandidateScore dictionaries, sorted by (rrf_score desc, avg_sim desc, movie_id asc)
        - 'raw_anchor': List of RankedResult dictionaries from anchor collection
        - 'raw_content': List of RankedResult dictionaries from content collection
        - 'raw_vibe': List of RankedResult dictionaries from vibe collection
        
    Raises:
        ValueError: If neither query_text nor query_movie_id is provided, or both are provided
    """
    # Validate input: exactly one of query_text or query_movie_id must be provided
    if query_text is None and query_movie_id is None:
        raise ValueError("Either query_text or query_movie_id must be provided")
    if query_text is not None and query_movie_id is not None:
        raise ValueError("Only one of query_text or query_movie_id should be provided")
    
    # Determine which IDs to filter out (exclude the query movie itself in movie mode)
    ids_to_filter_out = [query_movie_id] if query_movie_id else None
    
    # Step 1: Get query vectors
    if query_movie_id:
        # Movie mode: fetch stored vectors for each collection
        query_vector_anchor = get_movie_embeddings([query_movie_id], 'dense_anchor_vectors', db_path)[0]
        query_vector_content = get_movie_embeddings([query_movie_id], 'dense_content_vectors', db_path)[0]
        query_vector_vibe = get_movie_embeddings([query_movie_id], 'dense_vibe_vectors', db_path)[0]
        
        if query_vector_anchor is None or query_vector_content is None or query_vector_vibe is None:
            raise ValueError(f"Movie {query_movie_id} not found in one or more collections")
    else:
        # Text mode: embed query once and use for all collections
        query_vector_anchor = query_vector_content = query_vector_vibe = embed_query(query_text)
    
    # Step 2: Search each collection independently with collection-specific query vectors
    ranks_map_anchor = search_collection_and_build_ranks(
        query_vector_anchor, 'dense_anchor_vectors', n_anchor, db_path, ids_to_filter_out
    )
    ranks_map_content = search_collection_and_build_ranks(
        query_vector_content, 'dense_content_vectors', n_content, db_path, ids_to_filter_out
    )
    ranks_map_vibe = search_collection_and_build_ranks(
        query_vector_vibe, 'dense_vibe_vectors', n_vibe, db_path, ids_to_filter_out
    )
    
    # For cosine similarity computation, use the anchor vector as the reference
    # (or average of all three if in movie mode - but anchor is fine for consistency)
    query_vector_for_similarity = query_vector_anchor

    
    # Step 3: Build candidate pool (union)
    candidate_ids = set()
    candidate_ids.update(ranks_map_anchor.keys())
    candidate_ids.update(ranks_map_content.keys())
    candidate_ids.update(ranks_map_vibe.keys())
    
    # Step 4: Compute RRF scores for all candidates
    candidate_scores: list[CandidateScore] = []

    
    # Get metadata from any collection (they should be consistent)
    metadata_map: dict[str, dict] = {}
    for candidate_id in candidate_ids:
        if candidate_id not in metadata_map:
            if candidate_id in ranks_map_anchor:
                metadata_map[candidate_id] = ranks_map_anchor[candidate_id]['metadata']
            elif candidate_id in ranks_map_content:
                metadata_map[candidate_id] = ranks_map_content[candidate_id]['metadata']
            elif candidate_id in ranks_map_vibe:
                metadata_map[candidate_id] = ranks_map_vibe[candidate_id]['metadata']

    
    for movie_id in candidate_ids:
        rank_a = ranks_map_anchor.get(movie_id, {}).get('rank')
        rank_c = ranks_map_content.get(movie_id, {}).get('rank')
        rank_v = ranks_map_vibe.get(movie_id, {}).get('rank')
        
        # Compute RRF score
        rrf_result = compute_rrf_score(
            rank_a, rank_c, rank_v,
            rrf_k, w_anchor, w_content, w_vibe
        )
        
        candidate_scores.append({
            'movie_id': movie_id,
            'rrf_score': rrf_result['rrf_score'],
            'avg_sim': 0.0,  # Will be filled in Step 5
            'rank_anchor': rank_a,
            'rank_content': rank_c,
            'rank_vibe': rank_v,
            'term_anchor': rrf_result['term_anchor'],
            'term_content': rrf_result['term_content'],
            'term_vibe': rrf_result['term_vibe'],
            'sim_anchor': 0.0,  # Will be filled in Step 5
            'sim_content': 0.0,  # Will be filled in Step 5
            'sim_vibe': 0.0,  # Will be filled in Step 5
            'metadata': metadata_map.get(movie_id, {})
        })
    
    # Step 5: Compute cosine similarities for tie-breaking
    cosine_sims = compute_cosine_similarities_for_candidates(
        query_vector_anchor, query_vector_content, query_vector_vibe,
        candidate_ids, db_path
    )

    
    # Update candidate scores with cosine similarities
    for candidate in candidate_scores:
        movie_id = candidate['movie_id']
        if movie_id in cosine_sims:
            sims = cosine_sims[movie_id]
            candidate['sim_anchor'] = sims['sim_anchor']
            candidate['sim_content'] = sims['sim_content']
            candidate['sim_vibe'] = sims['sim_vibe']
            candidate['avg_sim'] = sims['avg_sim']
    
    
    # Step 6: Sort by (rrf_score desc, avg_sim desc, movie_id asc)
    candidate_scores.sort(
        key=lambda x: (-x['rrf_score'], -x['avg_sim'], x['movie_id'])
    )
    
    # Return top-N results along with raw lists for visualization
    return {
        'fused_results': candidate_scores[:return_top_n],
        'raw_anchor': list(ranks_map_anchor.values()),
        'raw_content': list(ranks_map_content.values()),
        'raw_vibe': list(ranks_map_vibe.values())
    }

