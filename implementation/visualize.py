"""
Visualization utilities for vector embeddings.

This module contains functions to visualize high-dimensional vectors using
dimensionality reduction techniques like t-SNE.
"""

import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from plotly.graph_objects import Figure
import plotly.express as px
from .schemas import ChromaVectorCollection


def visualize_vectors_tsne(
    vector_collection: ChromaVectorCollection,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42
) -> Figure:
    """
    Visualizes a set of vectors using t-SNE dimensionality reduction.
    
    This function takes a ChromaVectorCollection and reduces the high-dimensional
    vectors to a lower dimensional space (typically 2D or 3D) using t-SNE, then
    creates an interactive plotly visualization. Points are colored by the
    first genre of each movie.
    
    Args:
        vector_collection: ChromaVectorCollection instance containing vectors,
                          metadata, and other information from ChromaDB
        n_components: Number of dimensions to reduce to (2 or 3). Default is 2.
        perplexity: t-SNE perplexity parameter (typically between 5 and 50).
                   Lower values focus on local structure, higher values focus
                   on global structure. Default is 30.0.
        random_state: Random seed for reproducibility. Default is 42.
        
    Returns:
        Plotly Figure object that can be displayed using .show() or saved.
        
    Raises:
        ValueError: If vector_collection is empty, vectors have inconsistent
                   dimensions, or n_components is not 2 or 3.
    """
    # Extract embeddings from the collection
    vectors = vector_collection.embeddings
    
    # Validate inputs
    if not vectors:
        raise ValueError("vector_collection.embeddings cannot be empty")
    
    if n_components not in [2, 3]:
        raise ValueError("n_components must be 2 or 3 for visualization")
    
    # Convert vectors to numpy array for t-SNE
    vectors_array = np.array(vectors)
    
    # Validate that all vectors have the same dimensionality
    if len(vectors_array.shape) != 2:
        raise ValueError("embeddings must be a 2D list (list of lists)")
    
    n_vectors, vector_dim = vectors_array.shape
    print(f"Reducing {n_vectors} vectors of dimension {vector_dim} to {n_components}D using t-SNE...")
    
    # Adjust perplexity if needed (must be less than number of samples)
    adjusted_perplexity = min(perplexity, n_vectors - 1)
    if adjusted_perplexity < perplexity:
        print(f"  Adjusted perplexity from {perplexity} to {adjusted_perplexity} (n_vectors={n_vectors})")
    
    # Apply t-SNE dimensionality reduction
    tsne = TSNE(
        n_components=n_components,
        perplexity=adjusted_perplexity,
        random_state=random_state,
        init='pca',  # Use PCA initialization for better results
        learning_rate='auto'
    )
    
    vectors_reduced = tsne.fit_transform(vectors_array)
    print(f"✓ t-SNE reduction complete")
    
    # Extract genres from metadata for coloring
    # Use the first genre of each movie, or "Unknown" if no genres available
    genres = []
    genre_to_num = {}
    genre_num = 0
    
    for metadata in vector_collection.metadatas:
        # Extract genres from metadata (stored as comma-separated string)
        genres_str = metadata.get("genres", "")
        if genres_str:
            # Split comma-separated genres and take the first one
            first_genre = genres_str.split(", ")[0] if ", " in genres_str else genres_str
        else:
            first_genre = "Unknown"
        
        # Map genre to numeric value for coloring
        if first_genre not in genre_to_num:
            genre_to_num[first_genre] = genre_num
            genre_num += 1
        
        genres.append(first_genre)
    
    # Create hover text with movie titles
    hover_text = [
        metadata.get("title", f"ID: {vector_collection.ids[i]}")
        for i, metadata in enumerate(vector_collection.metadatas)
    ]
    
    # Get discrete color palette for genres
    unique_genres = list(genre_to_num.keys())
    n_genres = len(unique_genres)
    # Use plotly's qualitative color palette
    color_palette = px.colors.qualitative.Set3 if n_genres <= 12 else px.colors.qualitative.Dark2
    genre_colors = {genre: color_palette[i % len(color_palette)] for i, genre in enumerate(unique_genres)}
    
    # Create plotly figure
    if n_components == 2:
        # 2D scatter plot
        fig = go.Figure()
        
        # Group by genre and create a trace for each genre
        for genre in unique_genres:
            # Find indices of vectors with this genre
            genre_indices = [i for i, g in enumerate(genres) if g == genre]
            
            if genre_indices:
                fig.add_trace(go.Scatter(
                    x=vectors_reduced[genre_indices, 0],
                    y=vectors_reduced[genre_indices, 1],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=genre_colors[genre]
                    ),
                    text=[hover_text[i] for i in genre_indices],
                    hoverinfo='text',
                    name=genre
                ))
        
        fig.update_layout(
            title=f't-SNE Visualization ({n_vectors} vectors, {vector_dim}D → 2D)',
            xaxis_title='t-SNE Component 1',
            yaxis_title='t-SNE Component 2',
            width=800,
            height=600,
            legend=dict(title="Genre")
        )
        
    else:  # n_components == 3
        # 3D scatter plot
        fig = go.Figure()
        
        # Group by genre and create a trace for each genre
        for genre in unique_genres:
            # Find indices of vectors with this genre
            genre_indices = [i for i, g in enumerate(genres) if g == genre]
            
            if genre_indices:
                fig.add_trace(go.Scatter3d(
                    x=vectors_reduced[genre_indices, 0],
                    y=vectors_reduced[genre_indices, 1],
                    z=vectors_reduced[genre_indices, 2],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=genre_colors[genre]
                    ),
                    text=[hover_text[i] for i in genre_indices],
                    hoverinfo='text',
                    name=genre
                ))
        
        fig.update_layout(
            title=f't-SNE Visualization ({n_vectors} vectors, {vector_dim}D → 3D)',
            scene=dict(
                xaxis_title='t-SNE Component 1',
                yaxis_title='t-SNE Component 2',
                zaxis_title='t-SNE Component 3'
            ),
            width=800,
            height=600,
            legend=dict(title="Genre")
        )
    
    return fig

