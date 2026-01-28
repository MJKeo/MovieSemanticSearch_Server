"""
Gradio interface for the vector search system.

This module provides a Gradio UI for the vector search functionality,
exposing all tunable parameters for rapid testing and tuning.
"""

import json
import sys
import traceback
from pathlib import Path
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import gradio as gr

# Add parent directory to path to import from implementation
sys.path.insert(0, str(Path(__file__).parent.parent))
from implementation.search import fused_vector_search
from implementation.enums import VectorCollectionName


# Mapping from VectorCollectionName enum to weight parameter names in fused_vector_search
COLLECTION_TO_WEIGHT_PARAM = {
    VectorCollectionName.DENSE_ANCHOR_VECTORS: "w_anchor",
    VectorCollectionName.PLOT_EVENTS_VECTORS: "w_plot_events",
    VectorCollectionName.PLOT_ANALYSIS_VECTORS: "w_plot_analysis",
    VectorCollectionName.NARRATIVE_TECHNIQUES_VECTORS: "w_narrative_techniques",
    VectorCollectionName.VIEWER_EXPERIENCE_VECTORS: "w_viewer_experience",
    VectorCollectionName.WATCH_CONTEXT_VECTORS: "w_watch_context",
    VectorCollectionName.PRODUCTION_VECTORS: "w_production",
    VectorCollectionName.RECEPTION_VECTORS: "w_reception",
}

# Color palette for collection charts (one unique color per collection)
COLLECTION_COLORS = {
    VectorCollectionName.DENSE_ANCHOR_VECTORS: "#FF6B6B",  # Red
    VectorCollectionName.PLOT_EVENTS_VECTORS: "#4ECDC4",  # Teal
    VectorCollectionName.PLOT_ANALYSIS_VECTORS: "#45B7D1",  # Blue
    VectorCollectionName.NARRATIVE_TECHNIQUES_VECTORS: "#9B59B6",  # Purple
    VectorCollectionName.VIEWER_EXPERIENCE_VECTORS: "#FFA07A",  # Light Salmon
    VectorCollectionName.WATCH_CONTEXT_VECTORS: "#98D8C8",  # Mint
    VectorCollectionName.PRODUCTION_VECTORS: "#F7DC6F",  # Yellow
    VectorCollectionName.RECEPTION_VECTORS: "#E67E22",  # Orange
}

# Color for overall fused results chart
OVERALL_COLOR = "#FF8C42"  # Orange


def format_collection_name(collection_name: VectorCollectionName) -> str:
    """
    Formats a VectorCollectionName enum value into a human-readable display name.
    
    Converts enum values like "dense_anchor_vectors" to "Dense Anchor Vectors"
    or "plot_events_vectors" to "Plot Events Vectors".
    
    Args:
        collection_name: VectorCollectionName enum member
        
    Returns:
        Formatted display name with title case and spaces
    """
    # Convert enum value (e.g., "dense_anchor_vectors") to title case
    name = collection_name.value.replace("_", " ").title()
    return name


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
    
    # Create title -> id mapping (using tmdb_id as string)
    title_to_id = {}
    for movie in movies:
        title = movie.get('title', '')
        tmdb_id = movie.get('tmdb_id')
        if title and tmdb_id is not None:
            title_to_id[title] = str(tmdb_id)
    
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


def prepare_table_data_collection(
    rank_map: dict[str, dict], 
    similarities_by_movie_id: dict[str, dict[str, float]], 
    collection_name: str,
    display_name: str,
    top_n: int
) -> pd.DataFrame:
    """
    Prepares data for collection-specific results table sorted by distance (ascending).
    
    Args:
        rank_map: Dictionary mapping movie_id to RankedResult with 'metadata', 'rank', 'distance', and 'document'
        similarities_by_movie_id: Dictionary mapping movie_id to dict of similarities per collection (unused, kept for compatibility)
        collection_name: Name of the collection (enum value) - unused, kept for compatibility
        display_name: Display name for the distance column header
        top_n: Number of top results to include
        
    Returns:
        DataFrame with columns: 'movie_title', 'distance', and 'document'
    """
    table_data = []
    
    # Convert rank_map to sorted list by distance (ascending, so lowest distance first)
    sorted_results = sorted(
        rank_map.items(),
        key=lambda x: x[1].get('distance', float('inf'))
    )
    
    for movie_id, ranked_result in sorted_results[:top_n]:
        metadata = ranked_result.get('metadata', {})
        title = metadata.get('title', movie_id)
        distance = ranked_result.get('distance', float('inf'))
        document = ranked_result.get('document', '')
        
        table_data.append({
            'movie_title': title,
            'distance': float(distance),
            'document': str(document)
        })
    
    # Sort ascending by distance (lowest distance first)
    df = pd.DataFrame(table_data)
    if not df.empty:
        df = df.sort_values('distance', ascending=True).reset_index(drop=True)
    
    return df


def create_bar_plot_figure_overall(results: list[dict], top_n: int, title: str, y_label: str, color: str) -> plt.Figure:
    """
    Creates a matplotlib bar plot figure for overall fused results (using rrf_score).
    
    Args:
        results: List of CandidateScore dictionaries with 'metadata' and 'rrf_score'
        top_n: Number of top results to include
        title: Plot title
        y_label: Label for y-axis
        color: Color for the bars (e.g., 'orange', 'red', 'blue', 'green')
        
    Returns:
        matplotlib Figure object
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
    ranks = []
    scores = []
    
    for i, result in enumerate(results[:top_n], start=1):
        score = result.get('rrf_score', 0.0)
        
        # Ensure score is numeric and not NaN
        try:
            score_float = float(score)
            if pd.isna(score_float):
                score_float = 0.0
        except (ValueError, TypeError):
            score_float = 0.0
        
        ranks.append(i)
        scores.append(score_float)
    
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


def create_bar_plot_figure_collection(
    rank_map: dict[str, dict],
    similarities_by_movie_id: dict[str, dict[str, float]],
    collection_name: str,
    top_n: int,
    title: str,
    y_label: str,
    color: str
) -> plt.Figure:
    """
    Creates a matplotlib bar plot figure for collection-specific results (using distance).
    
    Args:
        rank_map: Dictionary mapping movie_id to RankedResult with 'metadata', 'rank', and 'distance'
        similarities_by_movie_id: Dictionary mapping movie_id to dict of similarities per collection (unused, kept for compatibility)
        collection_name: Name of the collection (enum value) - unused, kept for compatibility
        top_n: Number of top results to include
        title: Plot title
        y_label: Label for y-axis
        color: Color for the bars (e.g., 'orange', 'red', 'blue', 'green')
        
    Returns:
        matplotlib Figure object
    """
    if not rank_map or top_n <= 0:
        # Return empty figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(title)
        ax.set_xlabel('rank')
        ax.set_ylabel(y_label)
        ax.text(0.5, 0.5, 'No results', ha='center', va='center', transform=ax.transAxes)
        plt.tight_layout()
        return fig
    
    # Convert rank_map to sorted list by distance (ascending, so lowest distance first)
    sorted_results = sorted(
        rank_map.items(),
        key=lambda x: x[1].get('distance', float('inf'))
    )
    
    # Extract and prepare data
    ranks = []
    scores = []
    
    for i, (movie_id, ranked_result) in enumerate(sorted_results[:top_n], start=1):
        # Get distance for this collection (lower is better)
        distance = ranked_result.get('distance', float('inf'))
        
        # Ensure score is numeric and not NaN
        try:
            score_float = float(distance)
            if pd.isna(score_float):
                score_float = float('inf')
        except (ValueError, TypeError):
            score_float = float('inf')
        
        ranks.append(i)
        scores.append(score_float)
    
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
    controls, allowing rapid testing and tuning of the search system. It dynamically
    generates UI components for all collections defined in VectorCollectionName.
    
    Args:
        db_path: Path to ChromaDB database directory
        
    Returns:
        Gradio Interface object
    """
    # Load movie titles for dropdown
    title_to_id = load_movie_titles()
    movie_titles = sorted(title_to_id.keys())
    
    # Get all collections dynamically
    collections = list(VectorCollectionName)
    
    def search_interface(
        query_mode: str,
        manual_query: str,
        selected_movie: str,
        n_candidates_per_axis: int,
        rrf_k: float,
        return_top_n: int,
        *weight_values
    ):
        """
        Gradio interface function that performs search and returns tables and charts.
        
        Args:
            query_mode: "Manual Text" or "Movie Selection"
            manual_query: Text query for manual mode
            selected_movie: Selected movie title for movie mode
            n_candidates_per_axis: Top-K to retrieve from each collection
            rrf_k: RRF rank dampening constant
            return_top_n: Number of results to return
            *weight_values: Variable number of weight values, one per collection (in enum order)
            
        Returns:
            Tuple of outputs: (overall_table, collection_tables..., overall_chart, collection_charts...)
        """
        try:
            # Determine query parameters based on mode
            query_text = None
            query_movie_id = None
            
            if query_mode == "Manual Text":
                if not manual_query or not manual_query.strip():
                    empty_df = pd.DataFrame({'movie_title': [], 'RRF Similarity Score': []})
                    empty_fig = create_bar_plot_figure_overall([], 0, "No Results", "score", OVERALL_COLOR)
                    empty_outputs = [empty_df] + [empty_df] * len(collections) + [empty_fig] + [empty_fig] * len(collections)
                    return tuple(empty_outputs)
                query_text = manual_query.strip()
            else:  # Movie Selection
                if not selected_movie or selected_movie not in title_to_id:
                    empty_df = pd.DataFrame({'movie_title': [], 'RRF Similarity Score': []})
                    empty_fig = create_bar_plot_figure_overall([], 0, "No Results", "score", OVERALL_COLOR)
                    empty_outputs = [empty_df] + [empty_df] * len(collections) + [empty_fig] + [empty_fig] * len(collections)
                    return tuple(empty_outputs)
                query_movie_id = title_to_id[selected_movie]
            
            # Map weight values to VectorCollectionName enum keys
            weights = {}
            for i, collection in enumerate(collections):
                weight_value = weight_values[i] if i < len(weight_values) else 1.0
                weights[collection] = weight_value
            
            # Perform search with weights dictionary
            search_results = fused_vector_search(
                query_text=query_text,
                query_movie_id=query_movie_id,
                n_candidates_per_axis=n_candidates_per_axis,
                rrf_k=rrf_k,
                weights=weights,
                return_top_n=return_top_n,
                db_path=db_path
            )
            
            # Extract results
            fused_results = search_results['fused_results']
            raw_rank_maps_by_collection = search_results['raw_rank_maps_by_collection']
            raw_similarities_by_movie_id = search_results['raw_similarities_by_movie_id']
            
            # Prepare overall table
            table_overall = prepare_table_data_overall(fused_results, return_top_n)
            
            # Prepare collection tables and charts
            collection_tables = []
            collection_charts = []
            
            for collection in collections:
                collection_name = collection.value
                display_name = format_collection_name(collection)
                
                # Get rank map for this collection
                rank_map = raw_rank_maps_by_collection.get(collection_name, {})
                
                # Prepare table
                table = prepare_table_data_collection(
                    rank_map,
                    raw_similarities_by_movie_id,
                    collection_name,
                    display_name,
                    return_top_n
                )
                collection_tables.append(table)
                
                # Prepare chart
                chart = create_bar_plot_figure_collection(
                    rank_map,
                    raw_similarities_by_movie_id,
                    collection_name,
                    return_top_n,
                    f"Top {display_name} Candidates",
                    "Distance",
                    COLLECTION_COLORS[collection]
                )
                collection_charts.append(chart)
            
            # Prepare overall chart
            chart_overall = create_bar_plot_figure_overall(
                fused_results,
                return_top_n,
                "Top Overall Candidates",
                "RRF Score",
                OVERALL_COLOR
            )
            
            # Return outputs: overall_table, collection_tables..., overall_chart, collection_charts...
            return (table_overall, *collection_tables, chart_overall, *collection_charts)
            
        except Exception as e:
            # Print full traceback to console
            print("=" * 80, file=sys.stderr)
            print("ERROR in search_interface:", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            sys.stderr.flush()
            
            # Return empty outputs on error (with error message in UI)
            empty_df = pd.DataFrame({'movie_title': [f'Error: {str(e)}'], 'RRF Similarity Score': [0.0]})
            empty_fig = create_bar_plot_figure_overall([], 0, f"Error: {str(e)}", "score", OVERALL_COLOR)
            empty_outputs = [empty_df] + [empty_df] * len(collections) + [empty_fig] + [empty_fig] * len(collections)
            return tuple(empty_outputs)
    
    # Create Gradio interface
    with gr.Blocks(title="Movie Vector Search") as interface:
        gr.Markdown("# Movie Vector Search with Weighted RRF Fusion")
        gr.Markdown(f"Search across {len(collections)} vector collections using Reciprocal Rank Fusion.")
        
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

                # Model Settings
                gr.Markdown("### Model Settings")
                return_top_n_input = gr.Number(
                    value=5,
                    label="return_top_n",
                    info="Number of top results to return and display. Higher values show more candidates but may include less relevant results."
                )
                n_candidates_per_axis_input = gr.Number(
                    value=5,
                    label="n_candidates_per_axis",
                    info="Top-K to retrieve from each collection. Higher values increase recall but may add noise."
                )
                rrf_k_input = gr.Number(
                    value=60.0,
                    label="rrf_k",
                    info="Rank dampening constant. Smaller values make top ranks more important; larger values favor consensus across collections."
                )
                
                # Dynamically generate weight sliders for each collection
                gr.Markdown("### Collection Weights")
                weight_inputs = []
                for collection in collections:
                    display_name = format_collection_name(collection)
                    weight_param = COLLECTION_TO_WEIGHT_PARAM[collection]
                    weight_input = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=1.0,
                        step=0.1,
                        label=weight_param,
                        info=f"Weight for {display_name} collection. Higher values increase influence in fusion."
                    )
                    weight_inputs.append(weight_input)
            
            # Middle Column: Tables
            with gr.Column():
                gr.Markdown("### Top Overall Candidates")
                table_overall = gr.Dataframe(
                    headers=["movie_title", "RRF Similarity Score"],
                    label="Movies that are the closest overall match",
                    interactive=False
                )
                
                # Dynamically generate tables for each collection
                collection_tables = []
                for collection in collections:
                    display_name = format_collection_name(collection)
                    gr.Markdown(f"### Top {display_name} Candidates")
                    table = gr.Dataframe(
                        headers=["movie_title", "distance", "document"],
                        label=f"Movies that best match on {display_name.lower()}",
                        interactive=False
                    )
                    collection_tables.append(table)
            
            # Right Column: Charts
            with gr.Column():
                gr.Markdown("### Top Overall Candidates")
                chart_overall = gr.Plot(label="Top Overall Candidates", show_label=False)
                
                # Dynamically generate charts for each collection
                collection_charts = []
                for collection in collections:
                    display_name = format_collection_name(collection)
                    gr.Markdown(f"### Top {display_name} Candidates")
                    chart = gr.Plot(label=f"Top {display_name} Candidates", show_label=False)
                    collection_charts.append(chart)
        
        # Wire up the interface
        search_button.click(
            fn=search_interface,
            inputs=[
                query_mode,
                manual_query_input,
                movie_dropdown,
                n_candidates_per_axis_input,
                rrf_k_input,
                return_top_n_input,
                *weight_inputs
            ],
            outputs=[
                table_overall,
                *collection_tables,
                chart_overall,
                *collection_charts
            ]
        )
    
    return interface


# ===== MAIN ENTRY POINT =====

if __name__ == "__main__":
    # Default to implementation/chroma_db if running from evaluations directory
    from pathlib import Path
    
    # Determine db_path based on where script is located
    script_dir = Path(__file__).parent
    default_db_path = script_dir.parent / "implementation" / "chroma_db"
    
    # Create and launch Gradio interface
    interface = create_gradio_interface(db_path=default_db_path)
    interface.launch(share=False, inbrowser=True)
