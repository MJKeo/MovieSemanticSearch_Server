"""
Gradio interface for the vector search system.

This module provides a Gradio UI for the vector search functionality,
exposing all tunable parameters for rapid testing and tuning.
"""

import json
from pathlib import Path
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import gradio as gr

from search import fused_vector_search


def load_movie_titles(movies_file: str | Path = "../saved_imdb_movies.json") -> dict[str, str]:
    """
    Loads movie titles and IDs from the saved movies JSON file.
    
    Args:
        movies_file: Path to the JSON file containing movie data
        
    Returns:
        Dictionary mapping movie title to movie ID
    """
    movies_path = Path(movies_file)
    if not movies_path.exists():
        # Try relative to implementation directory
        script_dir = Path(__file__).parent
        movies_path = script_dir.parent / "saved_imdb_movies.json"
    
    if not movies_path.exists():
        return {}
    
    with open(movies_path, 'r', encoding='utf-8') as f:
        movies = json.load(f)
    
    # Create title -> id mapping
    title_to_id = {}
    for movie in movies:
        title = movie.get('title', '')
        movie_id = movie.get('id', '')
        if title and movie_id:
            title_to_id[title] = movie_id
    
    return title_to_id


def prepare_table_data_overall(results: list[dict], top_n: int) -> pd.DataFrame:
    """
    Prepares data for overall results table (sorted descending by similarity).
    
    Args:
        results: List of result dictionaries with 'metadata' and 'avg_sim'
        top_n: Number of top results to include
        
    Returns:
        DataFrame with columns: 'movie_title', 'RRF Similarity Score'
    """
    table_data = []
    for result in results[:top_n]:
        metadata = result.get('metadata', {})
        title = metadata.get('title', result.get('movie_id', 'Unknown'))
        score = result.get('rrf_score', 0.0)
        
        table_data.append({
            'movie_title': title,
            'RRF Similarity Score': float(score)
        })
    
    # Sort descending by similarity score
    df = pd.DataFrame(table_data)
    if not df.empty:
        df = df.sort_values('RRF Similarity Score', ascending=False).reset_index(drop=True)
    
    return df


def prepare_table_data_collection(results: list[dict], top_n: int, collection_name: str) -> pd.DataFrame:
    """
    Prepares data for collection-specific results table (sorted ascending by distance/similarity).
    
    Args:
        results: List of result dictionaries with 'metadata' and 'similarity'
        top_n: Number of top results to include
        collection_name: Name of the collection for the column header
        
    Returns:
        DataFrame with columns: 'movie_title', and collection-specific cosine similarity column
    """
    table_data = []
    for result in results[:top_n]:
        metadata = result.get('metadata', {})
        title = metadata.get('title', result.get('movie_id', 'Unknown'))
        # For collection results, similarity is stored as 'similarity'
        # Lower values = more similar (distance), so we sort ascending
        score = result.get('similarity', 0.0)
        
        table_data.append({
            'movie_title': title,
            collection_name: float(score)
        })
    
    # Sort ascending by similarity (lowest distance first)
    df = pd.DataFrame(table_data)
    if not df.empty:
        df = df.sort_values(collection_name, ascending=True).reset_index(drop=True)
    
    return df


def create_bar_plot_figure(results: list[dict], score_key: str, top_n: int, title: str, y_label: str, color: str) -> plt.Figure:
    """
    Creates a matplotlib bar plot figure for gr.Plot visualization.
    
    Handles both fused results (with 'avg_sim') and raw collection results (with 'similarity').
    Results are sorted by similarity score (descending) and ranks are reassigned as sequential integers.
    
    Args:
        results: List of result dictionaries with 'metadata' and similarity/score fields
        top_n: Number of top results to include
        title: Plot title
        y_label: Label for y-axis
        color: Color for the bars (e.g., 'orange', 'red', 'blue', 'green')
        
    Returns:
        matplotlib Figure object, or None if no results
    """
    if not results or top_n <= 0:
        # Return empty figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(title)
        ax.set_xlabel('rank')
        ax.set_ylabel(y_label)
        ax.text(0.5, 0.5, 'No results', ha='center', va='center', transform=ax.transAxes)
        plt.tight_layout()
        return fig
    
    # Extract and prepare data
    plot_data = []
    for i, result in enumerate(results[:top_n], start=1):
        metadata = result.get('metadata', {})
        title_text = metadata.get('title', result.get('movie_id', f'Movie {i}'))

        score = result.get(score_key, 0.0)
        
        # Ensure score is numeric and not NaN
        try:
            score_float = float(score)
            if pd.isna(score_float):
                score_float = 0.0
        except (ValueError, TypeError):
            score_float = 0.0
        
        plot_data.append({
            'rank': i,
            'score': score_float,
            'title': str(title_text)
        })
    
    df = pd.DataFrame(plot_data)
    
    # Extract data for plotting
    ranks = df['rank'].tolist()
    scores = df['score'].tolist()
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(ranks, scores, color=color, alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('rank')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-axis labels if there are many results
    if len(ranks) > 10:
        ax.set_xticks(ranks[::max(1, len(ranks)//10)])  # Show every Nth label
    
    plt.tight_layout()
    return fig


def create_gradio_interface(db_path: str | Path = "./chroma_db"):
    """
    Creates a Gradio interface for the vector search system.
    
    This function sets up a Gradio UI with all tunable parameters exposed as
    controls, allowing rapid testing and tuning of the search system.
    
    Args:
        db_path: Path to ChromaDB database directory
        
    Returns:
        Gradio Interface object
    """
    # Load movie titles for dropdown
    title_to_id = load_movie_titles()
    movie_titles = sorted(title_to_id.keys())
    
    def search_interface(
        query_mode: str,
        manual_query: str,
        selected_movie: str,
        n_anchor: int,
        n_content: int,
        n_vibe: int,
        rrf_k: float,
        w_anchor: float,
        w_content: float,
        w_vibe: float,
        return_top_n: int
    ):
        """
        Gradio interface function that performs search and returns tables and charts.
        
        Args:
            query_mode: "Manual Text" or "Movie Selection"
            manual_query: Text query for manual mode
            selected_movie: Selected movie title for movie mode
            n_anchor: Top-K for anchor collection
            n_content: Top-K for content collection
            n_vibe: Top-K for vibe collection
            rrf_k: RRF rank dampening constant
            w_anchor: Weight for anchor collection
            w_content: Weight for content collection
            w_vibe: Weight for vibe collection
            return_top_n: Number of results to return
            
        Returns:
            Tuple of 8 outputs: (4 tables, 4 charts)
        """
        try:
            # Determine query parameters based on mode
            query_text = None
            query_movie_id = None
            
            if query_mode == "Manual Text":
                if not manual_query or not manual_query.strip():
                    empty_df = pd.DataFrame({'movie_title': [], 'RRF Similarity Score': []})
                    empty_fig = create_bar_plot_figure([], "", 0, "No Results", "score", "gray")
                    return (empty_df, empty_df, empty_df, empty_df, 
                           empty_fig, empty_fig, empty_fig, empty_fig)
                query_text = manual_query.strip()
            else:  # Movie Selection
                if not selected_movie or selected_movie not in title_to_id:
                    empty_df = pd.DataFrame({'movie_title': [], 'RRF Similarity Score': []})
                    empty_fig = create_bar_plot_figure([], "", 0, "No Results", "score", "gray")
                    return (empty_df, empty_df, empty_df, empty_df,
                           empty_fig, empty_fig, empty_fig, empty_fig)
                query_movie_id = title_to_id[selected_movie]
            
            # Perform search
            search_results = fused_vector_search(
                query_text=query_text,
                query_movie_id=query_movie_id,
                n_anchor=n_anchor,
                n_content=n_content,
                n_vibe=n_vibe,
                rrf_k=rrf_k,
                w_anchor=w_anchor,
                w_content=w_content,
                w_vibe=w_vibe,
                return_top_n=return_top_n,
                db_path=db_path
            )
            
            # Extract results
            fused_results = search_results['fused_results']
            raw_anchor = search_results['raw_anchor']
            raw_content = search_results['raw_content']
            raw_vibe = search_results['raw_vibe']

            # Prepare table data
            table_overall = prepare_table_data_overall(fused_results, return_top_n)
            table_anchor = prepare_table_data_collection(raw_anchor, return_top_n, "distance")
            table_content = prepare_table_data_collection(raw_content, return_top_n, "distance")
            table_vibe = prepare_table_data_collection(raw_vibe, return_top_n, "distance")
            
            # Prepare chart data (matplotlib figures for gr.Plot)
            chart_overall = create_bar_plot_figure(fused_results, "rrf_score", return_top_n, "Top Overall Candidates", "RRF similarity score", "orange")
            chart_anchor = create_bar_plot_figure(raw_anchor, "similarity", return_top_n, "Top Anchor Candidates", "distance", "red")
            chart_content = create_bar_plot_figure(raw_content, "similarity", return_top_n, "Top Content Candidates", "distance", "blue")
            chart_vibe = create_bar_plot_figure(raw_vibe, "similarity", return_top_n, "Top Vibe Candidates", "distance", "green")
            
            return (table_overall, table_anchor, table_content, table_vibe,
                   chart_overall, chart_anchor, chart_content, chart_vibe)
            
        except Exception as e:
            # Return empty DataFrames on error
            empty_df = pd.DataFrame({'movie_title': [f'Error: {str(e)}'], 'RRF Similarity Score': [0.0]})
            empty_fig = create_bar_plot_figure([], "", 0, f"Error: {str(e)}", "score", "gray")
            return (empty_df, empty_df, empty_df, empty_df,
                   empty_fig, empty_fig, empty_fig, empty_fig)
    
    # Create Gradio interface
    with gr.Blocks(title="Movie Vector Search") as interface:
        gr.Markdown("# Movie Vector Search with Weighted RRF Fusion")
        gr.Markdown("Search across three vector collections using Reciprocal Rank Fusion.")
        
        # Search button spanning full width at top
        with gr.Row():
            search_button = gr.Button("Search", variant="primary", scale=1)
        
        # Three columns below
        with gr.Row():
            # Left Column: Inputs
            with gr.Column():
                # Query selection
                gr.Markdown("### Query Mode")
                query_mode = gr.Radio(
                    choices=["Manual Text", "Movie Selection"],
                    value="Manual Text",
                    label="Choose between manual text query or selecting a movie for 'more like this' search"
                )
                
                manual_query_input = gr.Textbox(
                    label="Enter a manual query",
                    placeholder="e.g., 'a slow-burn thriller about surveillance and paranoia'",
                    lines=2,
                    visible=True
                )
                
                movie_dropdown = gr.Dropdown(
                    choices=movie_titles,
                    label="Select a movie name",
                    visible=False,
                    interactive=True
                )
                
                # Show/hide inputs based on query mode
                def update_query_inputs(mode):
                    if mode == "Manual Text":
                        return gr.update(visible=True), gr.update(visible=False)
                    else:
                        return gr.update(visible=False), gr.update(visible=True)
                
                query_mode.change(
                    fn=update_query_inputs,
                    inputs=query_mode,
                    outputs=[manual_query_input, movie_dropdown]
                )

                # Return top_n
                gr.Markdown("### Model Settings")
                return_top_n_input = gr.Number(
                    value=20,
                    label="return_top_n",
                    info="Number of top results to return and display. Higher values show more candidates but may include less relevant results."
                )
                n_anchor_input = gr.Number(
                    value=50,
                    label="n_anchor",
                    info="Top-K to retrieve from anchor collection. Higher values increase recall but may add noise."
                )
                n_content_input = gr.Number(
                    value=50,
                    label="n_content",
                    info="Top-K to retrieve from content collection. Higher values increase recall but may add noise."
                )
                n_vibe_input = gr.Number(
                    value=50,
                    label="n_vibe",
                    info="Top-K to retrieve from vibe collection. Higher values increase recall but may add noise."
                )
                rrf_k_input = gr.Number(
                    value=60.0,
                    label="rrf_k",
                    info="Rank dampening constant. Smaller values make top ranks more important; larger values favor consensus across collections."
                )
                w_anchor_input = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.1,
                    label="w_anchor",
                    info="Weight for anchor collection. Higher values increase influence of anchor collection in fusion."
                )
                w_content_input = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.1,
                    label="w_content",
                    info="Weight for content collection. Higher values increase influence of content collection in fusion."
                )
                w_vibe_input = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.1,
                    label="w_vibe",
                    info="Weight for vibe collection. Higher values increase influence of vibe collection in fusion."
                )
            
            # Middle Column: Tables
            with gr.Column():
                gr.Markdown("### Top Overall Candidates")
                table_overall = gr.Dataframe(
                    headers=["movie_title", "score"],
                    label="Movies that are the closest overall match",
                    interactive=False
                )
                
                gr.Markdown("### Top Anchor Candidates")
                table_anchor = gr.Dataframe(
                    headers=["movie_title", "distance (difference)"],
                    label="Movies that best match the whole bundle",
                    interactive=False
                )
                
                gr.Markdown("### Top Content Candidates")
                table_content = gr.Dataframe(
                    headers=["movie_title", "distance (difference)"],
                    label="Movies that best match on content / plot",
                    interactive=False
                )
                
                gr.Markdown("### Top Vibe Candidates")
                table_vibe = gr.Dataframe(
                    headers=["movie_title", "distance (difference)"],
                    label="Movies that best match on vibe / mood",
                    interactive=False
                )
            
            # Right Column: Charts
            with gr.Column():
                gr.Markdown("### Top Overall Candidates")
                chart_overall = gr.Plot(label="Top Overall Candidates", show_label=False)
                
                gr.Markdown("### Top Anchor Candidates")
                chart_anchor = gr.Plot(label="Top Anchor Candidates", show_label=False)
                
                gr.Markdown("### Top Content Candidates")
                chart_content = gr.Plot(label="Top Content Candidates", show_label=False)
                
                gr.Markdown("### Top Vibe Candidates")
                chart_vibe = gr.Plot(label="Top Vibe Candidates", show_label=False)
        
        # Wire up the interface
        search_button.click(
            fn=search_interface,
            inputs=[
                query_mode,
                manual_query_input,
                movie_dropdown,
                n_anchor_input,
                n_content_input,
                n_vibe_input,
                rrf_k_input,
                w_anchor_input,
                w_content_input,
                w_vibe_input,
                return_top_n_input
            ],
            outputs=[
                table_overall, table_anchor, table_content, table_vibe,
                chart_overall, chart_anchor, chart_content, chart_vibe
            ]
        )
    
    return interface


# ===== MAIN ENTRY POINT =====

if __name__ == "__main__":
    # Default to implementation/chroma_db if running from project root
    from pathlib import Path
    
    # Determine db_path based on where script is located
    script_dir = Path(__file__).parent
    default_db_path = script_dir / "chroma_db"
    
    # Create and launch Gradio interface
    interface = create_gradio_interface(db_path=default_db_path)
    interface.launch(share=False, inbrowser=True)
