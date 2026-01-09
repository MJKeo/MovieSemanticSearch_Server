"""
Vectorization utilities for creating dense vector embeddings from movie data.

This module contains functions to generate text representations that will be
embedded as vectors for semantic search.
"""

import os
from pathlib import Path
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI
from classes import IMDBMovie, ChromaVectorCollection
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client at module level
# Get OpenAI API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY environment variable not set. "
        "Please set it before importing this module."
    )

# Create OpenAI client instance for use throughout the module
openai_client = OpenAI(api_key=api_key)

def normalize_vector(vector: list[float], eps: float = 1e-8) -> list[float]:
    """
    Normalizes a vector to unit length for efficient cosine similarity computation.
    
    Args:
        embedding: Vector as list of floats
        eps: Small epsilon to avoid division by zero
        
    Returns:
        Normalized vector as list of floats
    """
    vec_array = np.array(vector)
    norm = np.linalg.norm(vec_array)
    
    if norm < eps:
        return [0.0] * len(vector)
    
    normalized = vec_array / max(norm, eps)
    return normalized.tolist()

def normalize_embeddings(
    embeddings: dict[str, list[float]],
    eps: float = 1e-8
) -> dict[str, list[float]]:
    """
    Normalizes all embeddings in a dictionary to unit length.
    
    Args:
        embeddings: Dictionary mapping movie_id to embedding vector
        eps: Small epsilon to avoid division by zero
        
    Returns:
        Dictionary mapping movie_id to normalized embedding vector
    """
    return {
        movie_id: normalize_vector(embedding, eps)
        for movie_id, embedding in embeddings.items()
    }


def create_dense_anchor_vector_text(movie: IMDBMovie) -> str:
    """
    Creates the text representation for DenseAnchor vector embedding.
    
    DenseAnchor captures the full "movie card" identity to provide good recall
    for most queries. It includes comprehensive information about the movie's
    identity, content, production, cast, and reception.
    
    Args:
        movie: IMDBMovie instance containing all movie metadata
        
    Returns:
        Formatted text string ready for embedding as DenseAnchor vector
    """
    # Build the text components according to the DenseAnchor template (section 5.1)
    parts = []
    
    # Title string
    parts.append(movie.title_string())
    parts.append("")  # Empty line separator
    
    # Overview (use raw overview, not normalized version)
    parts.append(f"Overview: {movie.overview}")
    parts.append("")
    
    # Genres as comma-separated values
    genres_csv = ", ".join(movie.genres_string()) if movie.genres_string() else ""
    if genres_csv:
        parts.append(f"Genres: {genres_csv}")
    
    # Keywords as comma-separated values
    keywords_csv = ", ".join(movie.overall_keywords) if movie.overall_keywords else ""
    if keywords_csv:
        parts.append(f"Keywords: {keywords_csv}")
    
    parts.append("")
    
    # Release decade bucket
    decade_bucket = movie.release_decade_bucket()
    if decade_bucket:
        parts.append(decade_bucket)
    
    # Duration bucket
    duration_bucket = movie.duration_bucket()
    if duration_bucket:
        parts.append(f"Duration: {duration_bucket}")
    
    # Budget scale for era
    budget_bucket = movie.budget_bucket_for_era()
    if budget_bucket:
        parts.append(budget_bucket)
    
    parts.append("")
    
    # Maturity guidance
    maturity_guidance = movie.maturity_guidance_text()
    if maturity_guidance:
        parts.append(maturity_guidance)
    
    parts.append("")
    
    # Production information
    production_text = movie.production_text()
    if production_text:
        parts.append(production_text)
    
    # Languages information
    languages_text = movie.languages_text()
    if languages_text:
        parts.append(languages_text)
    
    parts.append("")
    
    # Cast and crew information
    cast_text = movie.cast_text()
    if cast_text:
        parts.append(cast_text)
    
    # Characters information
    characters_text = movie.characters_text()
    if characters_text:
        parts.append(characters_text)
    
    parts.append("")
    
    # Reception information
    reception_tier = movie.reception_tier()
    parts.append(f"Reception: {reception_tier}")
    
    # Optional review summary
    reception_summary = movie.reception_summary_text()
    if reception_summary:
        parts.append(reception_summary)

    parts.append("")

    # Watch providers information
    watch_providers_text = movie.watch_providers_text()
    if watch_providers_text:
        parts.append(watch_providers_text)
    
    # Join all parts with newlines
    return "\n".join(parts)


def create_dense_content_vector_text(movie: IMDBMovie) -> str:
    """
    Creates the text representation for DenseContent vector embedding.
    
    DenseContent focuses on "what happens" and major themes without being
    dominated by cast/production information. It emphasizes plot, synopsis,
    keyphrases, genres, and keywords to match content-based queries.
    
    Args:
        movie: IMDBMovie instance containing all movie metadata
        
    Returns:
        Formatted text string ready for embedding as DenseContent vector
    """
    # Build the text components according to the DenseContent template (section 5.2)
    parts = []
    
    # Title string
    parts.append(movie.title_string())
    parts.append("")  # Empty line separator
    
    # Plot synopsis (LLM-derived, may be empty if not yet generated)
    plot_string = movie.plot_string()
    if plot_string:
        parts.append(plot_string)
        parts.append("")
    
    # Genres as comma-separated values
    genres_csv = ", ".join(movie.genres_string()) if movie.genres_string() else ""
    if genres_csv:
        parts.append(f"Genres: {genres_csv}")
    
    # Overall keywords as comma-separated values
    overall_keywords_csv = ", ".join(movie.overall_keywords) if movie.overall_keywords else ""
    if overall_keywords_csv:
        parts.append(f"{overall_keywords_csv}")
    
    # Join all parts with newlines
    return "\n".join(parts)


def create_dense_vibe_vector_text(movie: IMDBMovie) -> str:
    """
    Creates the text representation for DenseVibe vector embedding.
    
    DenseVibe focuses on the viewing experience and suitability for different contexts.
    It matches queries like "cozy date night movies", "edge-of-your-seat thrillers",
    "gross-out horror", "comfort watch", and "background-friendly" by capturing
    how it feels to watch the movie rather than what happens in it.
    
    Args:
        movie: IMDBMovie instance containing all movie metadata (including vibe fields)
        
    Returns:
        Formatted text string ready for embedding as DenseVibe vector
        
    Note:
        If vibe_summary, vibe_keywords, or watch_context_tags are not available
        (None or empty), those sections will be omitted from the output.
    """
    # Build the text components according to the DenseVibe template (section 5.3.4)
    parts = []
    
    # Vibe summary (LLM-generated, may be None if not yet generated)
    if movie.vibe_summary:
        parts.append(f"Vibe summary: {movie.vibe_summary}")
        parts.append("")  # Empty line separator
    
    # Vibe keywords as comma-separated values
    if movie.vibe_keywords:
        vibe_keywords_csv = ", ".join(movie.vibe_keywords)
        parts.append(f"Vibe keywords: {vibe_keywords_csv}")
        parts.append("")
    
    # Watch context tags as comma-separated values
    if movie.watch_context_tags:
        watch_context_tags_csv = ", ".join(movie.watch_context_tags)
        parts.append(f"Watch context: {watch_context_tags_csv}")
        parts.append("")
    
    # Maturity guidance
    maturity_guidance = movie.maturity_guidance_text()
    if maturity_guidance:
        parts.append(f"Maturity guidance: {maturity_guidance}")
    
    # Genres as comma-separated values
    genres_csv = ", ".join(movie.genres_string()) if movie.genres_string() else ""
    if genres_csv:
        parts.append(f"Genres: {genres_csv}")
    
    # Join all parts with newlines
    return "\n".join(parts)


def save_vector_to_chroma(
    vector_id: str,
    embedding: list[float],
    document: str,
    metadata: dict[str, str | int | float],
    collection_name: str,
    db_path: str | Path = "./chroma_db",
    collection_metadata: dict[str, str] | None = None
) -> None:
    """
    Generic function to save a vector embedding to a ChromaDB collection.
    
    This function handles all ChromaDB operations including client initialization,
    collection creation/retrieval, and vector storage. It can be reused for any
    type of vector embedding storage.
    
    Args:
        vector_id: Unique identifier for the vector/document
        embedding: The vector embedding as a list of floats
        document: The original text/document that was embedded
        metadata: Dictionary of metadata to store with the vector
        collection_name: Name of the ChromaDB collection to store vectors in
        db_path: Path to the local ChromaDB database directory
        collection_metadata: Optional metadata dictionary for the collection itself
        
    Raises:
        Exception: If database operations fail
    """
    # Initialize ChromaDB client with persistent storage
    # Convert db_path to Path if it's a string for consistency
    db_path = Path(db_path) if isinstance(db_path, str) else db_path
    db_path.mkdir(parents=True, exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(
        path=str(db_path),
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Get or create the collection with optional metadata
    collection_meta = collection_metadata or {}
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata=collection_meta
    )
    
    # Add the embedding to the collection
    collection.add(
        ids=[vector_id],
        embeddings=[embedding],
        documents=[document],
        metadatas=[metadata]
    )
    print(f"✓ Saved vector '{vector_id}' to collection '{collection_name}'")


def fetch_all_vectors_from_chroma(
    collection_name: str,
    db_path: str | Path = "./chroma_db"
) -> ChromaVectorCollection:
    """
    Generic function to fetch all vectors from a ChromaDB collection.
    
    This function retrieves all vectors stored in a specified collection,
    including their IDs, embeddings, documents, and metadata.
    
    Args:
        collection_name: Name of the ChromaDB collection to fetch vectors from
        db_path: Path to the local ChromaDB database directory
        
    Returns:
        ChromaVectorCollection instance containing:
        - ids: List of vector IDs
        - embeddings: List of embedding vectors (each is a list of floats)
        - documents: List of original text documents
        - metadatas: List of metadata dictionaries
        
    Raises:
        ValueError: If the collection does not exist
        Exception: If database operations fail
    """
    # Initialize ChromaDB client with persistent storage
    # Convert db_path to Path if it's a string for consistency
    db_path = Path(db_path) if isinstance(db_path, str) else db_path
    
    # Check if database directory exists
    if not db_path.exists():
        raise ValueError(f"ChromaDB database not found at path: {db_path}")
    
    chroma_client = chromadb.PersistentClient(
        path=str(db_path),
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Get the collection (will raise error if it doesn't exist)
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        raise ValueError(
            f"Collection '{collection_name}' not found in database at {db_path}. "
            f"Original error: {str(e)}"
        )
    
    # Fetch all vectors from the collection
    # Must explicitly include embeddings, documents, and metadatas
    results = collection.get(include=['embeddings', 'documents', 'metadatas'])
    
    # Return results as ChromaVectorCollection instance
    return ChromaVectorCollection(
        ids=results["ids"],
        embeddings=results["embeddings"],
        documents=results["documents"],
        metadatas=results["metadatas"]
    )


def search_similar_vectors(
    query_vector: list[float],
    collection_name: str,
    db_path: str | Path = "./chroma_db",
    n_results: int = 2,
    ids_to_filter_out: list[str] | None = None
) -> ChromaVectorCollection:
    """
    Performs vector similarity search on a ChromaDB collection.
    
    This function finds the most similar vectors to a query vector using cosine
    similarity. It can optionally exclude certain vector IDs from the results.
    
    Args:
        query_vector: The query vector to search for (list of floats)
        collection_name: Name of the ChromaDB collection to search in
        db_path: Path to the local ChromaDB database directory
        n_results: Number of similar vectors to return. Default is 2.
        ids_to_filter_out: Optional list of vector IDs to exclude from results.
                         If None, no filtering is applied.
        
    Returns:
        ChromaVectorCollection instance containing the top n_results similar vectors:
        - ids: List of vector IDs (most similar first)
        - embeddings: List of embedding vectors
        - documents: List of original text documents
        - metadatas: List of metadata dictionaries
        - distances: List of distance scores (lower is more similar) or None if unavailable
        
    Raises:
        ValueError: If the collection does not exist or query_vector is invalid
        Exception: If database operations fail
    """
    # Validate query vector
    if not query_vector:
        raise ValueError("query_vector cannot be empty")
    
    # Initialize ChromaDB client with persistent storage
    db_path = Path(db_path) if isinstance(db_path, str) else db_path
    
    # Check if database directory exists
    if not db_path.exists():
        raise ValueError(f"ChromaDB database not found at path: {db_path}")
    
    chroma_client = chromadb.PersistentClient(
        path=str(db_path),
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Get the collection (will raise error if it doesn't exist)
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        raise ValueError(
            f"Collection '{collection_name}' not found in database at {db_path}. "
            f"Original error: {str(e)}"
        )
    
    # If we need to filter out IDs, query for more results than needed
    # then filter and return the top n_results
    ids_to_filter = set(ids_to_filter_out) if ids_to_filter_out else set()
    
    # Query for more results if we need to filter (to ensure we get enough after filtering)
    query_n_results = n_results + len(ids_to_filter) if ids_to_filter else n_results
    
    # Perform vector similarity search
    # ChromaDB uses cosine similarity by default
    # Include distances in the query to return similarity scores
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=query_n_results,
        include=['embeddings', 'documents', 'metadatas', 'distances']
    )
    
    # Extract results (query returns a dict with lists of lists, since it supports batch queries)
    result_ids = results["ids"][0]  # First (and only) query result
    result_embeddings = results["embeddings"][0]
    result_documents = results["documents"][0]
    result_metadatas = results["metadatas"][0]
    
    # Get distances if available (ChromaDB should include this when requested)
    result_distances = None
    if "distances" in results and results["distances"] is not None:
        result_distances = results["distances"][0]
    
    # Filter out excluded IDs
    filtered_ids = []
    filtered_embeddings = []
    filtered_documents = []
    filtered_metadatas = []
    filtered_distances = []
    
    for i, vector_id in enumerate(result_ids):
        if vector_id not in ids_to_filter:
            filtered_ids.append(vector_id)
            filtered_embeddings.append(result_embeddings[i])
            filtered_documents.append(result_documents[i])
            filtered_metadatas.append(result_metadatas[i])
            # Include distance if available
            if result_distances is not None:
                filtered_distances.append(result_distances[i])
        
        # Stop once we have enough results
        if len(filtered_ids) >= n_results:
            break
    
    # # If we don't have enough results after filtering, return what we have
    # if len(filtered_ids) < n_results:
    #     print(f"Warning: Only found {len(filtered_ids)} results after filtering (requested {n_results})")
    
    # Return results as ChromaVectorCollection instance
    # Include distances if they were available
    return ChromaVectorCollection(
        ids=filtered_ids[:n_results],
        embeddings=filtered_embeddings[:n_results],
        documents=filtered_documents[:n_results],
        metadatas=filtered_metadatas[:n_results],
        distances=filtered_distances[:n_results] if result_distances is not None else None
    )


def create_and_save_dense_anchor_vector(
    movie: IMDBMovie,
    collection_name: str = "dense_anchor_vectors",
    db_path: str | Path = "./chroma_db"
) -> None:
    """
    Creates a dense anchor vector embedding for a movie and saves it to ChromaDB.
    
    This function:
    1. Generates the text representation using create_dense_anchor_vector_text
    2. Embeds the text using OpenAI's text-embedding-3-small model
    3. Saves the embedding to a local ChromaDB collection
    
    Args:
        movie: IMDBMovie instance containing all movie metadata
        collection_name: Name of the ChromaDB collection to store vectors in
        db_path: Path to the local ChromaDB database directory
        
    Raises:
        Exception: If embedding or database operations fail
        
    Note:
        OpenAI API key must be set in environment variables before importing this module.
        The module will raise ValueError at import time if OPENAI_API_KEY is not found.
    """
    print(f"Processing: {movie.title} (ID: {movie.id})")
    
    # Generate text representation for embedding
    vector_text = create_dense_anchor_vector_text(movie)
    
    # Create embedding using OpenAI's small embedder model
    # text-embedding-3-small produces 1536-dimensional vectors
    print("  Creating embedding...")
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=vector_text
    )
    embedding = response.data[0].embedding
    print(f"  Embedding created: {embedding}")
    
    # Prepare metadata for storage (include movie title and ID for reference)
    metadata = {
        "movie_id": movie.id,
        "tmdb_id": movie.tmdb_id,
        "title": movie.title,
        "release_date": movie.release_date,
        "genres": ", ".join(movie.genres) if movie.genres else "",
    }
    
    # Save the vector to ChromaDB using the generic save function
    save_vector_to_chroma(
        vector_id=movie.id,
        embedding=embedding,
        document=vector_text,
        metadata=metadata,
        collection_name=collection_name,
        db_path=db_path,
        collection_metadata={"description": "Dense anchor vectors for movie semantic search"}
    )


def create_and_save_dense_content_vector(
    movie: IMDBMovie,
    collection_name: str = "dense_content_vectors",
    db_path: str | Path = "./chroma_db"
) -> None:
    """
    Creates a dense content vector embedding for a movie and saves it to ChromaDB.
    
    This function:
    1. Generates the text representation using create_dense_content_vector_text
    2. Embeds the text using OpenAI's text-embedding-3-small model
    3. Saves the embedding to a local ChromaDB collection
    
    DenseContent vectors focus on plot, themes, and content matching rather than
    cast/production information, making them ideal for "what happens" queries.
    
    Args:
        movie: IMDBMovie instance containing all movie metadata
        collection_name: Name of the ChromaDB collection to store vectors in
        db_path: Path to the local ChromaDB database directory
        
    Raises:
        Exception: If embedding or database operations fail
        
    Note:
        OpenAI API key must be set in environment variables before importing this module.
        The module will raise ValueError at import time if OPENAI_API_KEY is not found.
    """
    print(f"Processing: {movie.title} (ID: {movie.id})")
    
    # Generate text representation for embedding
    vector_text = create_dense_content_vector_text(movie)
    
    # Create embedding using OpenAI's small embedder model
    # text-embedding-3-small produces 1536-dimensional vectors
    print("  Creating embedding...")
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=vector_text
    )
    embedding = response.data[0].embedding
    print(f"  Embedding created: {embedding}")
    
    # Prepare metadata for storage (include movie title and ID for reference)
    metadata = {
        "movie_id": movie.id,
        "tmdb_id": movie.tmdb_id,
        "title": movie.title,
        "release_date": movie.release_date,
        "genres": ", ".join(movie.genres) if movie.genres else "",
    }
    
    # Save the vector to ChromaDB using the generic save function
    save_vector_to_chroma(
        vector_id=movie.id,
        embedding=embedding,
        document=vector_text,
        metadata=metadata,
        collection_name=collection_name,
        db_path=db_path,
        collection_metadata={"description": "Dense content vectors for movie semantic search"}
    )


def create_and_save_dense_vibe_vector(
    movie: IMDBMovie,
    collection_name: str = "dense_vibe_vectors",
    db_path: str | Path = "./chroma_db"
) -> None:
    """
    Creates a dense vibe vector embedding for a movie and saves it to ChromaDB.
    
    This function:
    1. Generates the text representation using create_dense_vibe_vector_text
    2. Embeds the text using OpenAI's text-embedding-3-small model
    3. Saves the embedding to a local ChromaDB collection
    
    DenseVibe vectors focus on viewer experience and suitability, enabling semantic
    matching for queries like "cozy date night movies", "edge-of-your-seat thrillers",
    "gross-out horror", "comfort watch", and "background-friendly".
    
    Args:
        movie: IMDBMovie instance containing all movie metadata (including vibe fields)
        collection_name: Name of the ChromaDB collection to store vectors in
        db_path: Path to the local ChromaDB database directory
        
    Raises:
        Exception: If embedding or database operations fail
        
    Note:
        OpenAI API key must be set in environment variables before importing this module.
        The module will raise ValueError at import time if OPENAI_API_KEY is not found.
        If vibe data is not available (vibe_summary, vibe_keywords, watch_context_tags),
        the vector will still be created but may have limited semantic matching capability.
    """
    print(f"Processing: {movie.title} (ID: {movie.id})")
    
    # Generate text representation for embedding
    vector_text = create_dense_vibe_vector_text(movie)
    
    # Create embedding using OpenAI's small embedder model
    # text-embedding-3-small produces 1536-dimensional vectors
    print("  Creating embedding...")
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=vector_text
    )
    embedding = response.data[0].embedding
    print(f"  Embedding created: {embedding}")
    
    # Prepare metadata for storage (include movie title and ID for reference)
    metadata = {
        "movie_id": movie.id,
        "tmdb_id": movie.tmdb_id,
        "title": movie.title,
        "release_date": movie.release_date,
        "genres": ", ".join(movie.genres) if movie.genres else "",
    }
    
    # Save the vector to ChromaDB using the generic save function
    save_vector_to_chroma(
        vector_id=movie.id,
        embedding=embedding,
        document=vector_text,
        metadata=metadata,
        collection_name=collection_name,
        db_path=db_path,
        collection_metadata={"description": "Dense vibe vectors for movie semantic search"}
    )

