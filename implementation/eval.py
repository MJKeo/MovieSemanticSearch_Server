"""
Evaluation pipeline for movie vector embeddings.

This module implements a comprehensive evaluation system for measuring embedding
quality across three vector spaces (DenseAnchor, DenseContent, DenseVibe) and
combined weighted RRF retrieval. It computes Hits@K, MRR@K, and nDCG@K metrics
against ground truth similarity rankings.
"""

import json
import math
from pathlib import Path
from typing import TypedDict
import tqdm

from search import fused_vector_search


# ===== DATA STRUCTURES =====

class GroundTruthEntry(TypedDict):
    """Represents a single ground truth entry from the evaluation dataset."""
    movie_id: str
    five_most_similar_movies_overall_ordered: list[str]  # Key in file: "5_most_similar_movies_overall_ordered"
    five_most_similar_movies_content_ordered: list[str]  # Key in file: "5_most_similar_movies_content_ordered"
    five_most_similar_movies_vibes_ordered: list[str]  # Key in file: "5_most_similar_movies_vibes_ordered"

class MetricsSet(TypedDict):
    """Metrics for a single axis."""
    hits: float
    rr: float
    ndcg: float

class SingleAxisResult(TypedDict):
    """Metrics for a single query."""
    metrics: MetricsSet
    expected_top_k: list[str]
    actual_top_k: list[str]

class PerMovieResult(TypedDict):
    """Metrics for a single movie."""
    anchor: SingleAxisResult
    content: SingleAxisResult
    vibe: SingleAxisResult
    overall: SingleAxisResult

class EvaluationReport(TypedDict):
    """Complete evaluation report."""
    K: int
    per_movie_results: dict[str, PerMovieResult]
    average_results: dict[str, MetricsSet]


# ===== GROUND TRUTH LOADING AND VALIDATION =====

def load_ground_truth(gt_path: str | Path) -> list[GroundTruthEntry]:
    """
    Loads ground truth data from JSON file.
    
    Args:
        gt_path: Path to the ground truth JSON file
        
    Returns:
        List of ground truth entries
        
    Raises:
        ValueError: If file cannot be loaded or parsed
    """
    gt_path = Path(gt_path)
    if not gt_path.exists():
        raise ValueError(f"Ground truth file not found: {gt_path}")
    
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both JSON array and JSONL format
    if isinstance(data, list):
        return data
    else:
        raise ValueError(f"Expected JSON array, got {type(data)}")


def validate_ground_truth(gt_data: list[GroundTruthEntry]) -> None:
    """
    Validates ground truth data according to hard constraints.
    
    Validates:
    - Each list has exactly 5 items
    - Query movie does not appear in any list
    - Each list contains unique IDs (no duplicates)
    - All listed IDs exist in the query set
    - No duplicate movie_id entries
    
    Args:
        gt_data: List of ground truth entries
        
    Raises:
        ValueError: If any validation constraint fails
    """
    # Build set of all movie IDs
    Q = {entry['movie_id'] for entry in gt_data}
    
    # Check for duplicate movie_id entries
    movie_ids = [entry['movie_id'] for entry in gt_data]
    if len(movie_ids) != len(set(movie_ids)):
        duplicates = [mid for mid in movie_ids if movie_ids.count(mid) > 1]
        raise ValueError(f"Duplicate movie_id entries found: {duplicates}")
    
    # Validate each entry
    for entry in gt_data:
        q = entry['movie_id']
        
        # Check all three lists (using actual keys from JSON file)
        for list_name in [
            '5_most_similar_movies_overall_ordered',
            '5_most_similar_movies_content_ordered',
            '5_most_similar_movies_vibes_ordered'
        ]:
            gt_list = entry.get(list_name, [])
            
            # Check length is exactly 5
            if len(gt_list) != 5:
                raise ValueError(
                    f"Movie {q}: {list_name} has {len(gt_list)} items, expected 5"
                )
            
            # Check query movie not in list
            if q in gt_list:
                raise ValueError(
                    f"Movie {q}: Query movie appears in {list_name}"
                )
            
            # Check for duplicates
            if len(gt_list) != len(set(gt_list)):
                raise ValueError(
                    f"Movie {q}: {list_name} contains duplicate IDs"
                )
            
            # Check all IDs exist in Q
            for movie_id in gt_list:
                if movie_id not in Q:
                    raise ValueError(
                        f"Movie {q}: {list_name} contains ID {movie_id} not in query set"
                    )


# ===== METRIC FUNCTIONS =====

def hits_at_k(calculated: list[str], truth: list[str], k: int) -> float:
    """
    Computes Hits@K metric (coverage of ground truth in top-K predictions).
    
    With K=5 and |GT|=5, this equals Precision@5 and Recall@5.
    
    Args:
        calculated: Calculated top-K list (from the code we're verifying)
        truth: Ground truth list (the correct answers)
        k: Cutoff K
        
    Returns:
        Hits@K score (0.0 to 1.0)
    """
    pred_set = set(calculated[:k])
    gt_set = set(truth)
    
    hits = len(pred_set & gt_set)
    return hits / k if k else 0.0


def rr_at_k(calculated: list[str], truth_set: set[str], k: int) -> float:
    """
    Computes Reciprocal Rank@K metric.
    
    Returns 1/r where r is the first position (1-indexed) where a relevant
    item appears, or 0 if no relevant items appear in top-K.
    
    Args:
        calculated: Calculated top-K list (from the code we're verifying)
        truth_set: Set of ground truth movie IDs (the correct answers)
        k: Cutoff K
        
    Returns:
        Reciprocal Rank@K score (0.0 to 1.0)
    """
    for i, movie_id in enumerate(calculated[:k], start=1):
        if movie_id in truth_set:
            return 1.0 / i
    
    return 0.0


def ndcg_at_k(calculated: list[str], truth: list[str], k: int) -> float:
    """
    Computes normalized Discounted Cumulative Gain@K metric.
    
    Uses exponential gain with graded relevance based on position in GT list.
    Position 0 (best) gets relevance 5, position 4 gets relevance 1.
    
    Args:
        calculated: Calculated top-K list (from the code we're verifying)
        truth: Ground truth list (the correct answers, ordered, best to worst)
        k: Cutoff K
        
    Returns:
        nDCG@K score (0.0 to 1.0)
    """
    # Build relevance map: position i -> relevance (5-i)
    rel_map = {truth[i]: len(truth) - i for i in range(len(truth))}
    
    # Compute DCG@K
    dcg = 0.0
    for i, movie_id in enumerate(calculated[:k], start=1):
        rel = rel_map.get(movie_id, 0)
        # Exponential gain: (2^rel - 1) / log2(i+1)
        gain = (2.0 ** rel - 1.0) / math.log2(i + 1)
        dcg += gain
    
    # Compute IDCG@K using ground truth order
    idcg = 0.0
    for i, movie_id in enumerate(truth[:k], start=1):
        rel = rel_map[movie_id]  # Should always exist
        gain = (2.0 ** rel - 1.0) / math.log2(i + 1)
        idcg += gain
    
    # Normalize
    if idcg > 0:
        return dcg / idcg
    else:
        return 0.0


# ===== MAIN EVALUATION PIPELINE =====

def run_evaluation(
    gt_path: str | Path,
    db_path: str | Path = "./chroma_db",
    rrf_k: float = 60.0,
    w_anchor: float = 1.0,
    w_content: float = 1.0,
    w_vibe: float = 1.0,
    n_anchor: int = 50,
    n_content: int = 50,
    n_vibe: int = 50
) -> EvaluationReport:
    """
    Runs the complete evaluation pipeline.
    
    This function:
    1. Loads and validates ground truth
    2. Loads embeddings from all three collections
    3. Normalizes embeddings
    4. Evaluates each vector space
    5. Evaluates combined RRF
    6. Generates comprehensive report
    
    Args:
        gt_path: Path to ground truth JSON file
        db_path: Path to ChromaDB database directory
        rrf_k: Rank dampening constant for RRF
        w_anchor: Weight for anchor collection
        w_content: Weight for content collection
        w_vibe: Weight for vibe collection
        n_anchor: Top-K to retrieve from anchor collection for RRF
        n_content: Top-K to retrieve from content collection for RRF
        n_vibe: Top-K to retrieve from vibe collection for RRF
        
    Returns:
        Complete evaluation report dictionary
    """
    # Hard code the number of results to return to be 5 (evaluation set is not dynamic)
    k = 5
    
    print("Step 1: Loading ground truth...")
    gt_data = load_ground_truth(gt_path)
    print(f"  Loaded {len(gt_data)} ground truth entries")
    
    print("Step 2: Validating ground truth...")
    validate_ground_truth(gt_data)
    print("  Validation passed")

    # For each ground truth entry (each entry represents a single movie and the movies that should be similar)
    # 1. Get the fused results
    # 2. Compute hits, rr, and ndcg for each vector space (+ overall)
    # 3. Save to broader results list
    # 4. Calculate averages for each vector space (+ overall)
    # 5. Return averages and per-movie results

    print("Step 3: Running evaluation...")
    print(f"  Evaluating {len(gt_data)} movies")

    per_movie_results: dict[str, PerMovieResult] = {}
    for expected_results in tqdm.tqdm(gt_data, desc="Evaluating movies"):
        movie_id = expected_results['movie_id']
        # Track the movies that were deemed the correct answers
        expected_similar_anchor_ids = expected_similar_overall_ids = expected_results['5_most_similar_movies_overall_ordered']
        expected_similar_content_ids = expected_results['5_most_similar_movies_content_ordered']
        expected_similar_vibe_ids = expected_results['5_most_similar_movies_vibes_ordered']

        # Fetch our calculated answers
        calculated_results = fused_vector_search(
            query_movie_id=movie_id,
            n_anchor=n_anchor,
            n_content=n_content,
            n_vibe=n_vibe,
            rrf_k=rrf_k,
            w_anchor=w_anchor,
            w_content=w_content,
            w_vibe=w_vibe,
            return_top_n=k,
            db_path=db_path
        )

        # Extract individual axes
        calculated_overall_ids = [result['movie_id'] for result in calculated_results['fused_results']]
        calculated_anchor_ids = [result['movie_id'] for result in calculated_results['raw_anchor']]
        calculated_content_ids = [result['movie_id'] for result in calculated_results['raw_content']]
        calculated_vibe_ids = [result['movie_id'] for result in calculated_results['raw_vibe']]

        # Compare how accurate our calculated answers are
        expected_calculated_pairs = [
            ('anchor', expected_similar_anchor_ids, calculated_anchor_ids),
            ('content', expected_similar_content_ids, calculated_content_ids),
            ('vibe', expected_similar_vibe_ids, calculated_vibe_ids),
            ('overall', expected_similar_overall_ids, calculated_overall_ids)
        ]

        evaluation_results: dict[str, SingleAxisResult] = {}
        for axis, expected_ids, calculated_ids in expected_calculated_pairs:
            # Calculate how far off we were
            hits = hits_at_k(calculated_ids, expected_ids, k)
            rr = rr_at_k(calculated_ids, expected_ids, k)
            ndcg = ndcg_at_k(calculated_ids, expected_ids, k)

            # Save the per-axis results
            evaluation_results[axis] = {
                'metrics': {    
                    'hits': hits,
                    'rr': rr,
                    'ndcg': ndcg
                },
                'expected_top_k': expected_ids[:k],
                'actual_top_k': calculated_ids[:k]
            }
        
        # Store overall (per-movie) results
        per_movie_results[movie_id] = evaluation_results

    print(f"  Evaluation complete for {len(per_movie_results)} movies")

    print("Step 4: Calculating average metrics...")
    
    # Calculate average metrics across all movies for each axis
    average_results: dict[str, MetricsSet] = {}
    for key in ['anchor', 'content', 'vibe', 'overall']:
        # Collect all metrics for this axis across all movies
        hits_values = []
        rr_values = []
        ndcg_values = []
        
        for movie_result in tqdm.tqdm(per_movie_results.values(), desc="Calculating average metrics"):
            if key in movie_result:
                metrics = movie_result[key]['metrics']
                hits_values.append(metrics['hits'])
                rr_values.append(metrics['rr'])
                ndcg_values.append(metrics['ndcg'])
        
        # Calculate averages (handle empty case)
        num_movies = len(per_movie_results.keys())
        if num_movies > 0:
            average_results[key] = {
                'hits': sum(hits_values) / num_movies,
                'rr': sum(rr_values) / num_movies,
                'ndcg': sum(ndcg_values) / num_movies
            }
        else:
            average_results[key] = {
                'hits': 0.0,
                'rr': 0.0,
                'ndcg': 0.0
            }
    
    # Build report
    report: EvaluationReport = {
        'K': k,
        "per_movie_results": per_movie_results,
        "average_results": average_results
    }
    
    return report


def save_report(report: EvaluationReport, output_path: str | Path) -> None:
    """
    Saves evaluation report to JSON file.
    
    Args:
        report: Evaluation report dictionary
        output_path: Path to save the report JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Report saved to {output_path}")


# ===== MAIN ENTRY POINT =====

if __name__ == "__main__":
    # Default configuration
    # Paths relative to implementation/ directory
    GT_PATH = Path(__file__).parent.parent / "evaluations" / "most_similar_movies.jsonl"
    DB_PATH = Path(__file__).parent / "chroma_db"
    OUTPUT_PATH = Path(__file__).parent.parent / "evaluations" / "evaluation_report.json"
    K = 5
    
    # Run evaluation
    report = run_evaluation(
        gt_path=GT_PATH,
        db_path=DB_PATH,
    )
    
    # Save report
    save_report(report, OUTPUT_PATH)
    
    print("\n=== Evaluation Summary ===")
    print(f"DenseAnchor - Hits@{K}: {report['average_results']['anchor']['hits']:.4f}, "
          f"MRR@{K}: {report['average_results']['anchor']['rr']:.4f}, "
          f"nDCG@{K}: {report['average_results']['anchor']['ndcg']:.4f}")
    print(f"DenseContent - Hits@{K}: {report['average_results']['content']['hits']:.4f}, "
          f"MRR@{K}: {report['average_results']['content']['rr']:.4f}, "
          f"nDCG@{K}: {report['average_results']['content']['ndcg']:.4f}")
    print(f"DenseVibe - Hits@{K}: {report['average_results']['vibe']['hits']:.4f}, "
          f"MRR@{K}: {report['average_results']['vibe']['rr']:.4f}, "
          f"nDCG@{K}: {report['average_results']['vibe']['ndcg']:.4f}")
    print(f"Combined RRF - Hits@{K}: {report['average_results']['overall']['hits']:.4f}, "
          f"MRR@{K}: {report['average_results']['overall']['rr']:.4f}, "
          f"nDCG@{K}: {report['average_results']['overall']['ndcg']:.4f}")

