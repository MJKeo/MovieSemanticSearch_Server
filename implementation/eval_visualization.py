"""
Gradio interface for visualizing evaluation results.

This module provides a web-based interface for running and visualizing
the results of the movie vector embedding evaluation pipeline.
"""

import gradio as gr
import pandas as pd
import json
from pathlib import Path
from typing import Tuple, Optional

from eval import run_evaluation, EvaluationReport


def get_color_for_score(score: float) -> str:
    """
    Returns color code based on score threshold.
    
    Args:
        score: The composite score (mean_hits * mean_mrr * mean_ndcg)
        
    Returns:
        Color string: 'green' if > 0.85, 'yellow' if > 0.7, 'red' otherwise
    """
    if score > 0.85:
        return "green"
    elif score > 0.7:
        return "yellow"
    else:
        return "red"


def get_hex_color_for_score(score: float) -> str:
    """
    Returns hex color code based on score threshold.
    
    Args:
        score: The composite score (mean_hits * mean_mrr * mean_ndcg)
        
    Returns:
        Hex color string: '#6BAA75' if > 0.85, '#FFC145' if > 0.7, '#A3333D' otherwise
    """
    if score > 0.85:
        return "#6BAA75"
    elif score > 0.7:
        return "#FFC145"
    else:
        return "#E63946"


def create_score_display(score: float, axis_name: str) -> str:
    """
    Creates HTML display for the composite score with color coding.
    
    Args:
        score: The composite score to display
        axis_name: Name of the axis (for labeling)
        
    Returns:
        HTML string with formatted score
    """
    color = get_hex_color_for_score(score)
    return f"""
    <div style="text-align: center; padding: 20px;">
        <div style="font-size: 48px; font-weight: bold; color: {color};">
            {score:.4f}
        </div>
        <div style="font-size: 18px; color: #666; margin-top: 10px;">
            {axis_name.upper()}
        </div>
    </div>
    """


def create_bar_chart_data(metrics: dict[str, float], series_name: str) -> pd.DataFrame:
    """
    Creates bar chart data for the three metrics with a Color column based on y-axis values.
    
    Args:
        metrics: Dictionary with 'hits', 'rr', 'ndcg'
        series_name: Name for the Series column (kept for compatibility, not used for coloring)
        
    Returns:
        DataFrame with 'Metric', 'Value', 'Series', and 'Color' columns for BarPlot
    """
    # Get metric values
    values = [
        metrics['hits'],
        metrics['rr'],
        metrics['ndcg']
    ]
    
    # Determine color for each value using get_color_for_score logic
    colors = [get_color_for_score(value) for value in values]
    
    # Prepare data for bar chart as DataFrame with Color column for coloring
    df = pd.DataFrame({
        'Metric': ['Hits@K', 'MRR@K', 'nDCG@K'],
        'Value': values,
        'Series': [series_name, series_name, series_name],  # Kept for compatibility
        'Color': colors  # Color based on y-axis value
    })
    
    return df


def load_movie_id_to_title_map(movies_file: str | Path = None) -> dict[str, str]:
    """
    Loads movie IDs and titles from the saved movies JSON file.
    
    Creates a mapping from movie ID to movie title for all movies
    in the saved_imdb_movies.json file.
    
    Args:
        movies_file: Optional path to the JSON file containing movie data.
                    If not provided, defaults to saved_imdb_movies.json in parent directory.
        
    Returns:
        Dictionary mapping movie ID to movie title
    """
    # Default to saved_imdb_movies.json in parent directory
    if movies_file is None:
        script_dir = Path(__file__).parent
        movies_path = script_dir.parent / "saved_imdb_movies.json"
    else:
        movies_path = Path(movies_file)
    
    # Return empty dict if file doesn't exist
    if not movies_path.exists():
        return {}
    
    # Load movies from JSON file
    with open(movies_path, 'r', encoding='utf-8') as f:
        movies = json.load(f)
    
    # Create id -> title mapping
    id_to_title = {}
    for movie in movies:
        movie_id = movie.get('id', '')
        title = movie.get('title', '')
        # Only add entries with both ID and title
        if movie_id and title:
            id_to_title[movie_id] = title
    
    return id_to_title


def update_per_movie_tables(
    selected_movie_name: str,
    report_state: Optional[EvaluationReport],
    movie_id_to_name_map: dict[str, str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Updates the expected and actual top-k tables and metrics tables for all axes based on selected movie.
    
    Returns tables for all 4 axes in order: overall, anchor, content, vibe.
    Each axis has an expected table, actual table, and metrics table.
    
    Args:
        selected_movie_name: Selected movie title from dropdown
        report_state: The evaluation report stored in state
        movie_id_to_name_map: Dictionary mapping movie_id to movie title
        
    Returns:
        Tuple of 12 DataFrames: (overall_expected, overall_actual, overall_metrics,
                                anchor_expected, anchor_actual, anchor_metrics,
                                content_expected, content_actual, content_metrics,
                                vibe_expected, vibe_actual, vibe_metrics)
    """
    # Create empty tables if no report or no selection
    empty_df = pd.DataFrame({"Rank": [], "Movie Name": []})
    empty_metrics_df = pd.DataFrame({"Metric": [], "Value": []})
    
    if not report_state or not selected_movie_name:
        return (empty_df, empty_df, empty_metrics_df, empty_df, empty_df, empty_metrics_df,
                empty_df, empty_df, empty_metrics_df, empty_df, empty_df, empty_metrics_df)
    
    # Create inverted map from title to ID for lookup
    name_to_id_map = {title: movie_id for movie_id, title in movie_id_to_name_map.items()}
    
    # Convert selected movie name to movie ID
    movie_id = name_to_id_map.get(selected_movie_name)
    if not movie_id:
        return (empty_df, empty_df, empty_metrics_df, empty_df, empty_df, empty_metrics_df,
                empty_df, empty_df, empty_metrics_df, empty_df, empty_df, empty_metrics_df)
    
    # Get the results for this movie using the ID
    movie_results = report_state['per_movie_results'][movie_id]
    
    # Process each axis in order: overall, anchor, content, vibe
    axes = ['overall', 'anchor', 'content', 'vibe']
    result_tables = []
    
    for axis in axes:
        axis_result = movie_results.get(axis, {})
        expected_ids = axis_result.get('expected_top_k', [])
        actual_ids = axis_result.get('actual_top_k', [])
        
        # Create expected table with movie names
        expected_data = []
        for rank, movie_id in enumerate(expected_ids, start=1):
            expected_data.append({
                "Rank": rank,
                "Movie Name": movie_id_to_name_map.get(movie_id, movie_id)
            })
        expected_df = pd.DataFrame(expected_data)
        
        # Create actual table with movie names
        actual_data = []
        for rank, movie_id in enumerate(actual_ids, start=1):
            actual_data.append({
                "Rank": rank,
                "Movie Name": movie_id_to_name_map.get(movie_id, movie_id)
            })
        actual_df = pd.DataFrame(actual_data)
        
        # Create metrics table with RR (MRR), hits, and ndcg as rows
        metrics = axis_result.get('metrics', {})
        metrics_data = [
            {"Metric": "MRR@K", "Value": metrics.get('rr', 0.0)},
            {"Metric": "Hits@K", "Value": metrics.get('hits', 0.0)},
            {"Metric": "nDCG@K", "Value": metrics.get('ndcg', 0.0)}
        ]
        metrics_df = pd.DataFrame(metrics_data)
        
        result_tables.extend([expected_df, actual_df, metrics_df])
    
    return tuple(result_tables)


def run_evaluation_and_update(
    gt_path: str,
    db_path: str,
    n_anchor: float,
    n_content: float,
    n_vibe: float,
    rrf_k: float,
    w_anchor: float,
    w_content: float,
    w_vibe: float
) -> Tuple[str, pd.DataFrame, str, pd.DataFrame, str, pd.DataFrame, str, pd.DataFrame, EvaluationReport]:
    """
    Runs evaluation and returns all visualization components.
    
    This function executes the evaluation pipeline and generates
    visualization components for all axes across 3 columns.
    
    Args:
        gt_path: Path to ground truth JSON file
        db_path: Path to ChromaDB database directory
        n_anchor: Top-K to retrieve from anchor collection
        n_content: Top-K to retrieve from content collection
        n_vibe: Top-K to retrieve from vibe collection
        rrf_k: Rank dampening constant for RRF
        w_anchor: Weight for anchor collection
        w_content: Weight for content collection
        w_vibe: Weight for vibe collection
        
    Returns:
        Tuple of HTML strings and DataFrames for each axis in each column
    """
    # Run evaluation with default paths if not provided
    if not gt_path:
        gt_path = str(Path(__file__).parent.parent / "evaluations" / "most_similar_movies.jsonl")
    if not db_path:
        db_path = str(Path(__file__).parent / "chroma_db")
    
    # Convert to int for n_* parameters
    report = run_evaluation(
        gt_path=gt_path,
        db_path=db_path,
        n_anchor=int(n_anchor),
        n_content=int(n_content),
        n_vibe=int(n_vibe),
        rrf_k=rrf_k,
        w_anchor=w_anchor,
        w_content=w_content,
        w_vibe=w_vibe
    )
    
    # Extract average results
    average_results = report['average_results']
    
    # Process each axis with color mapping
    axes_colors = {
        'overall': 'orange',
        'anchor': 'red',
        'content': 'blue',
        'vibe': 'green'
    }
    
    axes = ['overall', 'anchor', 'content', 'vibe']
    results = []
    
    for axis in axes:
        metrics = average_results[axis]
        # Calculate composite score
        composite_score = (metrics['hits'] + metrics['rr'] + metrics['ndcg']) / 3
        
        # Create score display HTML
        score_html = create_score_display(composite_score, axis)
        
        # Create bar chart with series name for color mapping
        bar_chart = create_bar_chart_data(metrics, axis.capitalize())
        
        # Add to results (we'll duplicate for 3 columns)
        results.extend([score_html, bar_chart])

    # Load mapping from movie ID to movie title
    movie_id_to_name_map = load_movie_id_to_title_map()
    
    # Convert movie IDs to display titles for dropdown
    choices = [movie_id_to_name_map.get(movie_id, movie_id) for movie_id in list(report['per_movie_results'].keys())]
    
    dropdown = gr.Dropdown(
        choices=choices,
        label="Select Movie",
        value=None,
        interactive=True
    )
    
    # Return results for visualization plus the report for state
    return tuple(results) + (report, dropdown)


def create_interface():
    """
    Creates and configures the Gradio interface.
    
    Returns:
        Configured Gradio interface
    """
    # Default paths (relative to implementation/ directory)
    default_gt_path = str(Path(__file__).parent.parent / "evaluations" / "most_similar_movies.jsonl")
    default_db_path = str(Path(__file__).parent / "chroma_db")
    
    # Custom CSS for styling evaluation rows
    custom_css = """
    .eval-row {
        background-color: #f5f5f5 !important;
        border: 1px solid #d0d0d0 !important;
        border-radius: 4px;
        padding: 10px;
        margin: 5px 0;
    }
    """
    
    with gr.Blocks(title="Movie Evaluation Visualization", css=custom_css) as interface:
        gr.Markdown("# Movie Vector Embedding Evaluation")
        gr.Markdown("Run evaluation and visualize results across different vector spaces.")
        
        # Full-width button
        eval_button = gr.Button(
            "Perform Evaluation",
            scale=1,
            size="lg",
            variant="primary"
        )
        
        # Input fields (hidden by default, but can be made visible if needed)
        with gr.Row(visible=False):
            gt_path_input = gr.Textbox(
                label="Ground Truth Path",
                value=default_gt_path
            )
            db_path_input = gr.Textbox(
                label="Database Path",
                value=default_db_path
            )
        
        # Create layout with parameter column on left and 3 result columns
        with gr.Row():
            # Left column: Model Parameters
            with gr.Column(scale=1):
                gr.Markdown("### Model Settings")
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
            
            # Create 3 result columns side by side
            with gr.Column(scale=3):
                with gr.Row():
                    # Column 1
                    with gr.Column():
                        gr.Markdown("### Evaluation Results")
                        # Overall row
                        with gr.Row(elem_classes=["eval-row"]):
                            with gr.Column(scale=1):
                                overall_score_1 = gr.HTML(label="Overall Score")
                            with gr.Column(scale=2):
                                overall_chart_1 = gr.BarPlot(
                                    label="Overall Metrics",
                                    x="Metric",
                                    y="Value",
                                    y_lim=[0, 1.0],
                                    color="Color",
                                    color_map={"green": "#6BAA75", "yellow": "#FFC145", "red": "#E63946"}
                                )
                        # Anchor row
                        with gr.Row(elem_classes=["eval-row"]):
                            with gr.Column(scale=1):
                                anchor_score_1 = gr.HTML(label="Anchor Score")
                            with gr.Column(scale=2):
                                anchor_chart_1 = gr.BarPlot(
                                    label="Anchor Metrics",
                                    x="Metric",
                                    y="Value",
                                    y_lim=[0, 1.0],
                                    color="Color",
                                    color_map={"green": "#6BAA75", "yellow": "#FFC145", "red": "#E63946"}
                                )
                        # Content row
                        with gr.Row(elem_classes=["eval-row"]):
                            with gr.Column(scale=1):
                                content_score_1 = gr.HTML(label="Content Score")
                            with gr.Column(scale=2):
                                content_chart_1 = gr.BarPlot(
                                    label="Content Metrics",
                                    x="Metric",
                                    y="Value",
                                    y_lim=[0, 1.0],
                                    color="Color",
                                    color_map={"green": "#6BAA75", "yellow": "#FFC145", "red": "#E63946"}
                                )
                        # Vibe row
                        with gr.Row(elem_classes=["eval-row"]):
                            with gr.Column(scale=1):
                                vibe_score_1 = gr.HTML(label="Vibe Score")
                            with gr.Column(scale=2):
                                vibe_chart_1 = gr.BarPlot(
                                    label="Vibe Metrics",
                                    x="Metric",
                                    y="Value",
                                    y_lim=[0, 1.0],
                                    color="Color",
                                    color_map={"green": "#6BAA75", "yellow": "#FFC145", "red": "#E63946"}
                                )
        
        # State to store the evaluation report
        report_state = gr.State(value=None)
        
        # Per-movie results section
        gr.Markdown("---")
        gr.Markdown("## Per-Movie Results")
        gr.Markdown("Select a movie to view its expected and actual top-K similar movies for each axis.")

        # Load mapping from movie ID to movie title
        movie_id_to_name_map = load_movie_id_to_title_map()
        
        # Movie selection dropdown
        movie_dropdown = gr.Dropdown(
            choices=[],
            label="Select Movie",
            value=None,
            interactive=True
        )
        
        # Create 4 rows, one for each axis in order: overall, anchor, content, vibe
        axes_labels = [
            ('overall', 'Overall'),
            ('anchor', 'Anchor'),
            ('content', 'Content'),
            ('vibe', 'Vibe')
        ]
        
        # Store table components for each axis
        axis_tables = []
        
        for axis_key, axis_label in axes_labels:
            with gr.Row():
                gr.Markdown(f"### {axis_label}")
            with gr.Row():
                # Expected top-k table
                with gr.Column(scale=1):
                    expected_table = gr.Dataframe(
                        label=f"{axis_label} - Expected Top-K",
                        headers=["Rank", "Movie Name"],
                        interactive=False
                    )
                    axis_tables.append(expected_table)
                
                # Actual top-k table
                with gr.Column(scale=1):
                    actual_table = gr.Dataframe(
                        label=f"{axis_label} - Actual Top-K",
                        headers=["Rank", "Movie Name"],
                        interactive=False
                    )
                    axis_tables.append(actual_table)
                
                # Metrics table: RR (MRR), hits, ndcg for that movie
                with gr.Column(scale=1):
                    metrics_table = gr.Dataframe(
                        label=f"{axis_label} - Metrics",
                        headers=["Metric", "Value"],
                        interactive=False
                    )
                    axis_tables.append(metrics_table)

        
        # Connect button to update function
        eval_button.click(
            fn=run_evaluation_and_update,
            inputs=[
                gt_path_input,
                db_path_input,
                n_anchor_input,
                n_content_input,
                n_vibe_input,
                rrf_k_input,
                w_anchor_input,
                w_content_input,
                w_vibe_input
            ],
            outputs=[
                overall_score_1, overall_chart_1,
                anchor_score_1, anchor_chart_1,
                content_score_1, content_chart_1,
                vibe_score_1, vibe_chart_1,
                report_state,
                movie_dropdown
            ]
        )
        
        # Update tables when movie is selected
        def update_tables_wrapper(selected_name: str, report: Optional[EvaluationReport]):
            """Wrapper function that captures movie_map from closure."""
            # Use movie_id_to_name_map from closure (defined above)
            return update_per_movie_tables(selected_name, report, movie_id_to_name_map)
        
        # Update all tables when movie dropdown changes
        movie_dropdown.change(
            fn=update_tables_wrapper,
            inputs=[movie_dropdown, report_state],
            outputs=axis_tables
        )
    
    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(inbrowser=True)

