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
from .movie import IMDBMovie
from .schemas import ChromaVectorCollection
from .enums import VectorCollectionName
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
    genres_csv = ", ".join(movie.genres_subset()) if movie.genres_subset() else ""
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
    if reception_tier:
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


def create_plot_events_vector_text(movie: IMDBMovie) -> str:
    return str(movie.plot_events_metadata)


def create_plot_analysis_vector_text(movie: IMDBMovie) -> str:
    parts = []

    parts.append(str(movie.plot_analysis_metadata))
    parts.append(", ".join(movie.overall_keywords))

    return "\n".join(parts)


def create_viewer_experience_vector_text(movie: IMDBMovie) -> str:
    """
    Creates the text representation for viewer experience vector embedding.
    
    This function extracts viewer experience descriptors from vibe_metadata,
    focusing on how the movie feels to watch rather than what happens in it.
    
    Args:
        movie: IMDBMovie instance containing vibe_metadata
        
    Returns:
        Formatted text string ready for embedding as viewer experience vector
    """
    # parts = []
    
    # # Extract viewer experience fields from vibe_metadata if available
    # if movie.vibe_metadata:
    #     if movie.vibe_metadata.dominant_mood:
    #         parts.append(f"Dominant mood: {movie.vibe_metadata.dominant_mood}")
    #     if movie.vibe_metadata.movie_energy:
    #         parts.append(f"Movie energy: {movie.vibe_metadata.movie_energy}")
    #     if movie.vibe_metadata.intensity:
    #         parts.append(f"Intensity: {movie.vibe_metadata.intensity}")
    #     if movie.vibe_metadata.romance_humor_sexuality:
    #         parts.append(f"Romance, humor, and sexuality: {movie.vibe_metadata.romance_humor_sexuality}")
    #     if movie.vibe_metadata.final_viewer_impression:
    #         parts.append(f"Final viewer impression: {movie.vibe_metadata.final_viewer_impression}")
    
    # # Include genres as context for viewer experience
    # genres = movie.genres_subset(limit=3)
    # if genres:
    #     genres_csv = ", ".join(genres)
    #     parts.append(f"Genres: {genres_csv}")
    
    # return "\n".join(parts)
    return "Implement this!"


def create_watch_context_vector_text(movie: IMDBMovie) -> str:
    """
    Creates the text representation for watch context vector embedding.
    
    This function extracts viewing context information from vibe_metadata
    and watch providers, focusing on when and where the movie is suitable to watch.
    
    Args:
        movie: IMDBMovie instance containing vibe_metadata and watch_providers
        
    Returns:
        Formatted text string ready for embedding as watch context vector
    """
    # parts = []
    
    # # Extract viewing context from vibe_metadata if available
    # if movie.vibe_metadata and movie.vibe_metadata.viewing_context:
    #     parts.append(f"Viewing context: {movie.vibe_metadata.viewing_context}")
    
    # # Include watch providers information
    # watch_providers_text = movie.watch_providers_text()
    # if watch_providers_text:
    #     parts.append(watch_providers_text)
    
    # # Include duration bucket as it affects watch context
    # duration_bucket = movie.duration_bucket()
    # if duration_bucket:
    #     parts.append(f"Duration: {duration_bucket}")
    
    # return "\n".join(parts)
    return "Implement this!"


def create_production_vector_text(movie: IMDBMovie) -> str:
    """
    Creates the text representation for production vector embedding.
    
    This function extracts production-related information including countries,
    production companies, filming locations, languages, and cast/crew.
    
    Args:
        movie: IMDBMovie instance containing production metadata
        
    Returns:
        Formatted text string ready for embedding as production vector
    """
    return "Implement this!"


def save_vector_to_chroma(
    vector_id: str,
    embedding: list[float],
    document: str,
    metadata: dict[str, str | int | float],
    collection_name: VectorCollectionName,
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
        name=collection_name.value,
        metadata=collection_meta
    )
    
    # Add the embedding to the collection
    collection.upsert(
        ids=[vector_id],
        embeddings=[embedding],
        documents=[document],
        metadatas=[metadata]
    )
    print(f"✓ Saved vector '{vector_id}' to collection '{collection_name.value}'")


def fetch_all_vectors_from_chroma(
    collection_name: VectorCollectionName,
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
        collection = chroma_client.get_collection(name=collection_name.value)
    except Exception as e:
        raise ValueError(
            f"Collection '{collection_name.value}' not found in database at {db_path}. "
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
    collection_name: VectorCollectionName,
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
        collection = chroma_client.get_collection(name=collection_name.value)
    except Exception as e:
        raise ValueError(
            f"Collection '{collection_name.value}' not found in database at {db_path}. "
            f"Original error: {str(e)}"
        )

    ids_to_filter_out = set(ids_to_filter_out) if ids_to_filter_out else set()
    
    # Perform vector similarity search
    # ChromaDB uses cosine similarity by default
    # Include distances in the query to return similarity scores
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results + len(ids_to_filter_out),
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

    if collection_name.value in set([VectorCollectionName.DENSE_ANCHOR_VECTORS.value, VectorCollectionName.PLOT_EVENTS_VECTORS.value]):
        print(f"top {n_results} results for {collection_name.value}")
        for id, distance, document, metadata in zip(result_ids, result_distances, result_documents, result_metadatas):
            print(f"id: {id}, title: {metadata['title']}, distance: {distance}, document: {document}")

    filtered_ids = []
    filtered_embeddings = []
    filtered_documents = []
    filtered_metadatas = []
    filtered_distances = []

    for id, embedding, document, metadata, distance in zip(result_ids, result_embeddings, result_documents, result_metadatas, result_distances):
        if id not in ids_to_filter_out:
            filtered_ids.append(id)
            filtered_embeddings.append(embedding)
            filtered_documents.append(document)
            filtered_metadatas.append(metadata)
            filtered_distances.append(distance)

    
    # Return results as ChromaVectorCollection instance
    # Include distances if they were available
    return ChromaVectorCollection(
        ids=filtered_ids,
        embeddings=filtered_embeddings,
        documents=filtered_documents,
        metadatas=filtered_metadatas,
        distances=filtered_distances
    )

def clear_collections_from_chroma(
    collection_names: list[VectorCollectionName],
    db_path: str | Path = "./chroma_db"
) -> None:
    """
    Deletes all vectors from each of the provided ChromaDB collections.
    
    This function removes all vectors from the specified collections while
    preserving the collection structure itself. If a collection doesn't exist,
    it will be skipped with a warning message.
    
    Args:
        collection_names: List of ChromaDB collection names to clear
        db_path: Path to the local ChromaDB database directory
        
    Raises:
        ValueError: If the database directory doesn't exist
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
    
    # Process each collection name
    for collection_name in collection_names:
        try:
            # Get the collection (will raise error if it doesn't exist)
            collection = chroma_client.get_collection(name=collection_name.value)
            
            # Get all vector IDs from the collection
            # We only need IDs for deletion, so we don't need to fetch embeddings/documents/metadata
            all_results = collection.get()
            all_ids = all_results["ids"]
            
            # Delete all vectors if there are any
            if all_ids:
                collection.delete(ids=all_ids)
                print(f"✓ Cleared {len(all_ids)} vector(s) from collection '{collection_name.value}'")
            else:
                print(f"✓ Collection '{collection_name.value}' is already empty")
                
        except Exception as e:
            # If collection doesn't exist or other error occurs, print warning and continue
            print(f"⚠ Warning: Could not clear collection '{collection_name.value}': {str(e)}")
            continue


def create_and_save_dense_anchor_vector(
    movie: IMDBMovie,
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
        collection_name=VectorCollectionName.DENSE_ANCHOR_VECTORS,
        db_path=db_path,
        collection_metadata={"description": "Dense anchor vectors for movie semantic search"}
    )


def create_and_save_plot_events_vector(
    movie: IMDBMovie,
    db_path: str | Path = "./chroma_db"
) -> None:
    """
    Creates a plot events vector embedding for a movie and saves it to ChromaDB.
    
    This function:
    1. Generates the text representation using create_plot_events_vector_text
    2. Embeds the text using OpenAI's text-embedding-3-small model
    3. Saves the embedding to a local ChromaDB collection
    
    Plot events vectors focus on chronological plot details, settings, and major
    characters, making them ideal for queries about specific plot points or events.
    
    Args:
        movie: IMDBMovie instance containing all movie metadata
        db_path: Path to the local ChromaDB database directory
        
    Raises:
        Exception: If embedding or database operations fail
        
    Note:
        OpenAI API key must be set in environment variables before importing this module.
        The module will raise ValueError at import time if OPENAI_API_KEY is not found.
    """
    print(f"Processing: {movie.title} (ID: {movie.id})")
    
    # Generate text representation for embedding
    vector_text = create_plot_events_vector_text(movie)
    
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
        collection_name=VectorCollectionName.PLOT_EVENTS_VECTORS,
        db_path=db_path,
        collection_metadata={"description": "Plot events vectors for movie semantic search"}
    )


def create_and_save_plot_analysis_vector(
    movie: IMDBMovie,
    db_path: str | Path = "./chroma_db",
) -> None:
    """
    Creates a dense content vector embedding for a movie and saves it to ChromaDB.
    
    This function:
    1. Generates the text representation using create_plot_analysis_vector_text
    2. Embeds the text using OpenAI's text-embedding-3-small model
    3. Saves the embedding to a local ChromaDB collection
    
    DenseContent vectors focus on plot, themes, and content matching rather than
    cast/production information, making them ideal for "what happens" queries.
    
    Args:
        movie: IMDBMovie instance containing all movie metadata
        db_path: Path to the local ChromaDB database directory
        
    Raises:
        Exception: If embedding or database operations fail
        
    Note:
        OpenAI API key must be set in environment variables before importing this module.
        The module will raise ValueError at import time if OPENAI_API_KEY is not found.
    """
    print(f"Processing: {movie.title} (ID: {movie.id})")
    
    # Generate text representation for embedding
    vector_text = create_plot_analysis_vector_text(movie)
    
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
        collection_name=VectorCollectionName.PLOT_ANALYSIS_VECTORS,
        db_path=db_path,
        collection_metadata={"description": "Plot analysis vectors for movie semantic search"}
    )


def create_and_save_viewer_experience_vector(
    movie: IMDBMovie,
    db_path: str | Path = "./chroma_db"
) -> None:
    """
    Creates a viewer experience vector embedding for a movie and saves it to ChromaDB.
    
    This function:
    1. Generates the text representation using create_viewer_experience_vector_text
    2. Embeds the text using OpenAI's text-embedding-3-small model
    3. Saves the embedding to a local ChromaDB collection
    
    Viewer experience vectors focus on how the movie feels to watch, including mood,
    energy, intensity, and emotional impact, making them ideal for queries about
    viewing experience rather than plot content.
    
    Args:
        movie: IMDBMovie instance containing all movie metadata (including vibe_metadata)
        db_path: Path to the local ChromaDB database directory
        
    Raises:
        Exception: If embedding or database operations fail
        
    Note:
        OpenAI API key must be set in environment variables before importing this module.
        The module will raise ValueError at import time if OPENAI_API_KEY is not found.
        If vibe_metadata is not available, the vector will still be created but may
        have limited semantic matching capability.
    """
    print(f"Processing: {movie.title} (ID: {movie.id})")
    
    # Generate text representation for embedding
    vector_text = create_viewer_experience_vector_text(movie)
    
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
        collection_name=VectorCollectionName.VIEWER_EXPERIENCE_VECTORS,
        db_path=db_path,
        collection_metadata={"description": "Viewer experience vectors for movie semantic search"}
    )


def create_and_save_watch_context_vector(
    movie: IMDBMovie,
    db_path: str | Path = "./chroma_db"
) -> None:
    """
    Creates a watch context vector embedding for a movie and saves it to ChromaDB.
    
    This function:
    1. Generates the text representation using create_watch_context_vector_text
    2. Embeds the text using OpenAI's text-embedding-3-small model
    3. Saves the embedding to a local ChromaDB collection
    
    Watch context vectors focus on when and where the movie is suitable to watch,
    including viewing context recommendations, availability, and duration, making
    them ideal for queries about watchability and viewing situations.
    
    Args:
        movie: IMDBMovie instance containing all movie metadata (including vibe_metadata)
        db_path: Path to the local ChromaDB database directory
        
    Raises:
        Exception: If embedding or database operations fail
        
    Note:
        OpenAI API key must be set in environment variables before importing this module.
        The module will raise ValueError at import time if OPENAI_API_KEY is not found.
    """
    print(f"Processing: {movie.title} (ID: {movie.id})")
    
    # Generate text representation for embedding
    vector_text = create_watch_context_vector_text(movie)
    
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
        collection_name=VectorCollectionName.WATCH_CONTEXT_VECTORS,
        db_path=db_path,
        collection_metadata={"description": "Watch context vectors for movie semantic search"}
    )


def create_and_save_production_vector(
    movie: IMDBMovie,
    db_path: str | Path = "./chroma_db"
) -> None:
    """
    Creates a production vector embedding for a movie and saves it to ChromaDB.
    
    This function:
    1. Generates the text representation using create_production_vector_text
    2. Embeds the text using OpenAI's text-embedding-3-small model
    3. Saves the embedding to a local ChromaDB collection
    
    Production vectors focus on production-related information including countries,
    production companies, filming locations, languages, cast, and crew, making
    them ideal for queries about who made the movie and where it was produced.
    
    Args:
        movie: IMDBMovie instance containing all movie metadata
        db_path: Path to the local ChromaDB database directory
        
    Raises:
        Exception: If embedding or database operations fail
        
    Note:
        OpenAI API key must be set in environment variables before importing this module.
        The module will raise ValueError at import time if OPENAI_API_KEY is not found.
    """
    print(f"Processing: {movie.title} (ID: {movie.id})")
    
    # Generate text representation for embedding
    vector_text = create_production_vector_text(movie)
    
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
        collection_name=VectorCollectionName.PRODUCTION_VECTORS,
        db_path=db_path,
        collection_metadata={"description": "Production vectors for movie semantic search"}
    )


