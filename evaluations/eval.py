"""
Evaluation pipeline for movie vector embeddings.

This module implements a comprehensive evaluation system for measuring embedding
quality across three vector spaces (DenseAnchor, DenseContent, DenseVibe) and
combined weighted RRF retrieval. It computes Hits@K, MRR@K, and nDCG@K metrics
against ground truth similarity rankings.
"""

import json
import math
import sys
from pathlib import Path
from typing import TypedDict
import tqdm

# Add parent directory to path to import from implementation
sys.path.insert(0, str(Path(__file__).parent.parent))
from implementation.search import fused_vector_search


# ===== DATA STRUCTURES =====

class MatchEntry(TypedDict):
    """Represents a single match entry with ID and score."""
    id: int
    score: int

class GroundTruthEntry(TypedDict):
    """Represents a single ground truth entry from the evaluation dataset."""
    overall_matches: list[MatchEntry]
    content_matches: list[MatchEntry]
    vibes_matches: list[MatchEntry]

# GroundTruthData is a dictionary mapping query TMDB ID (as string) to GroundTruthEntry
GroundTruthData = dict[str, GroundTruthEntry]

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


# ===== ID CONVERSION HELPERS =====

def load_tmdb_to_imdb_mapping(movies_file: str | Path = None) -> dict[str, str]:
    """
    Loads mapping from TMDB ID to IMDB ID from saved_imdb_movies.json.
    
    Creates a dictionary mapping TMDB ID (as string) to IMDB ID (as string)
    for all movies in the saved_imdb_movies.json file.
    
    Args:
        movies_file: Optional path to the JSON file containing movie data.
                    If not provided, defaults to saved_imdb_movies.json in parent directory.
        
    Returns:
        Dictionary mapping TMDB ID (as string) to IMDB ID (as string)
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
    
    # Create tmdb_id -> imdb_id mapping
    tmdb_to_imdb = {}
    for movie in movies:
        tmdb_id = movie.get('tmdb_id')
        imdb_id = movie.get('id', '')
        # Only add entries with both tmdb_id and imdb_id
        if tmdb_id is not None and imdb_id:
            tmdb_to_imdb[str(tmdb_id)] = imdb_id
    
    return tmdb_to_imdb


def convert_tmdb_to_imdb_id(tmdb_id: str, tmdb_to_imdb_map: dict[str, str]) -> str:
    """
    Converts a TMDB ID to an IMDB ID using the provided mapping.
    
    Args:
        tmdb_id: TMDB ID as a string
        tmdb_to_imdb_map: Dictionary mapping TMDB ID to IMDB ID
        
    Returns:
        IMDB ID as a string
        
    Raises:
        ValueError: If tmdb_id is not found in the mapping
    """
    imdb_id = tmdb_to_imdb_map.get(tmdb_id)
    if imdb_id is None:
        raise ValueError(f"TMDB ID {tmdb_id} not found in mapping. Cannot convert to IMDB ID.")
    return imdb_id


def load_imdb_to_tmdb_mapping(movies_file: str | Path = None) -> dict[str, str]:
    """
    Loads mapping from IMDB ID to TMDB ID from saved_imdb_movies.json.
    
    Creates a dictionary mapping IMDB ID (as string) to TMDB ID (as string)
    for all movies in the saved_imdb_movies.json file.
    
    Args:
        movies_file: Optional path to the JSON file containing movie data.
                    If not provided, defaults to saved_imdb_movies.json in parent directory.
        
    Returns:
        Dictionary mapping IMDB ID (as string) to TMDB ID (as string)
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
    
    # Create imdb_id -> tmdb_id mapping
    imdb_to_tmdb = {}
    for movie in movies:
        tmdb_id = movie.get('tmdb_id')
        imdb_id = movie.get('id', '')
        # Only add entries with both tmdb_id and imdb_id
        if tmdb_id is not None and imdb_id:
            imdb_to_tmdb[imdb_id] = str(tmdb_id)
    
    return imdb_to_tmdb


def convert_imdb_to_tmdb_id(imdb_id: str, imdb_to_tmdb_map: dict[str, str]) -> str | None:
    """
    Converts an IMDB ID to a TMDB ID using the provided mapping.
    
    Args:
        imdb_id: IMDB ID as a string
        imdb_to_tmdb_map: Dictionary mapping IMDB ID to TMDB ID
        
    Returns:
        TMDB ID as a string, or None if conversion fails
    """
    return imdb_to_tmdb_map.get(imdb_id)


# ===== GROUND TRUTH LOADING AND VALIDATION =====

def load_ground_truth(gt_path: str | Path) -> GroundTruthData:
    """
    Loads ground truth data from JSON file.
    
    The JSON file should be a dictionary mapping TMDB IDs (as strings or integers)
    to ground truth entries containing overall_matches, content_matches, and vibes_matches.
    
    Args:
        gt_path: Path to the ground truth JSON file
        
    Returns:
        Dictionary mapping query TMDB ID (as string) to GroundTruthEntry
        
    Raises:
        ValueError: If file cannot be loaded or parsed
    """
    gt_path = Path(gt_path)
    if not gt_path.exists():
        raise ValueError(f"Ground truth file not found: {gt_path}")
    
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle dictionary format (new schema)
    if isinstance(data, dict):
        # Convert all keys to strings to ensure consistency
        # JSON keys may be strings or integers, but we need strings for ChromaDB compatibility
        result: GroundTruthData = {}
        for key, value in data.items():
            # Convert key to string if it's an integer
            str_key = str(key)
            result[str_key] = value
        return result
    else:
        raise ValueError(f"Expected JSON dictionary, got {type(data)}")


def validate_ground_truth(gt_data: GroundTruthData) -> None:
    """
    Validates ground truth data according to hard constraints.
    
    Validates:
    - Each entry has required lists (overall_matches, content_matches, vibes_matches)
    - Each match entry has 'id' (int) and 'score' (int, 1-3)
    - Query movie ID does not appear in its own match lists
    - Each list contains unique IDs (no duplicates)
    - All listed IDs exist in the query set
    
    Args:
        gt_data: Dictionary mapping query TMDB ID to GroundTruthEntry
        
    Raises:
        ValueError: If any validation constraint fails
    """
    # Build set of all query movie IDs
    Q = set(gt_data.keys())
    
    # Validate each entry
    for query_id, entry in gt_data.items():
        # Check that entry has all required lists
        required_lists = ['overall_matches', 'content_matches', 'vibes_matches']
        for list_name in required_lists:
            if list_name not in entry:
                raise ValueError(
                    f"Query {query_id}: Missing required field '{list_name}'"
                )
            
            matches = entry[list_name]
            
            # Validate that matches is a list
            if not isinstance(matches, list):
                raise ValueError(
                    f"Query {query_id}: {list_name} must be a list, got {type(matches)}"
                )
            
            # Validate each match entry
            match_ids = []
            for i, match in enumerate(matches):
                # Check that match is a dictionary
                if not isinstance(match, dict):
                    raise ValueError(
                        f"Query {query_id}: {list_name}[{i}] must be a dict, got {type(match)}"
                    )
                
                # Check for required fields
                if 'id' not in match:
                    raise ValueError(
                        f"Query {query_id}: {list_name}[{i}] missing required field 'id'"
                    )
                if 'score' not in match:
                    raise ValueError(
                        f"Query {query_id}: {list_name}[{i}] missing required field 'score'"
                    )
                
                # Validate types
                match_id = match['id']
                if not isinstance(match_id, int):
                    raise ValueError(
                        f"Query {query_id}: {list_name}[{i}]['id'] must be int, got {type(match_id)}"
                    )
                
                score = match['score']
                
                # Validate score range (1-3)
                if score < 1 or score > 10:
                    raise ValueError(
                        f"Query {query_id}: {list_name}[{i}]['score'] must be between 1 and 10, got {score}"
                    )
                
                # Convert match ID to string for comparison
                match_ids.append(str(match_id))
            
            # Check query movie not in its own match list
            if query_id in match_ids:
                raise ValueError(
                    f"Query {query_id}: Query movie appears in {list_name}"
                )
            
            # Check for duplicate IDs
            if len(match_ids) != len(set(match_ids)):
                duplicates = [mid for mid in match_ids if match_ids.count(mid) > 1]
                raise ValueError(
                    f"Query {query_id}: {list_name} contains duplicate IDs: {set(duplicates)}"
                )
            
            # Check all match IDs exist in query set (optional validation)
            # Note: This assumes all matches should be from the query set
            # If matches can be from outside the query set, remove this check
            for match_id in match_ids:
                if match_id not in Q:
                    raise ValueError(
                        f"Query {query_id}: {list_name} contains ID {match_id} not in query set"
                    )


# ===== METRIC FUNCTIONS =====

def hits_at_k(calculated: list[str], truth: set[str], k: int) -> float:
    """
    Computes Hits@K metric (coverage of ground truth in top-K predictions).
    
    With K=5 and |GT|=5, this equals Precision@5 and Recall@5.
    
    Args:
        calculated: Calculated top-K list (from the code we're verifying)
        truth: Set of ground truth movie IDs (the correct answers)
        k: Cutoff K
        
    Returns:
        Hits@K score (0.0 to 1.0)
    """
    pred_set = set(calculated[:k])
    
    hits = len(pred_set & truth)
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


def ndcg_at_k(calculated: list[str], truth: dict[str, int], k: int) -> float:
    """
    Computes normalized Discounted Cumulative Gain@K metric.
    
    Uses exponential gain with graded relevance based on score values from ground truth.
    Score values are 1-3, where 3 is most relevant.
    
    Args:
        calculated: Calculated top-K list (from the code we're verifying)
        truth: Dictionary mapping movie ID to score (1-3, where 3 is most relevant)
        k: Cutoff K
        
    Returns:
        nDCG@K score (0.0 to 1.0)
    """
    # Compute DCG@K
    dcg = 0.0
    for i, movie_id in enumerate(calculated[:k], start=1):
        # Get score from truth map, default to 0 if not found
        score = truth.get(movie_id, 0)
        # New formula: (2^score - 1) / log2(i+1)
        gain = (2.0 ** score - 1.0) / math.log2(i + 1)
        dcg += gain
    
    # Compute IDCG@K by sorting all scores descending and taking top K
    # Get all scores and sort descending
    all_scores = sorted(truth.values(), reverse=True)
    
    # Calculate ideal DCG using top K scores
    idcg = 0.0
    for i, score in enumerate(all_scores[:k], start=1):
        gain = (2.0 ** score - 1.0) / math.log2(i + 1)
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

    print("Step 2: Loading ID mappings...")
    tmdb_to_imdb_map = load_tmdb_to_imdb_mapping()
    imdb_to_tmdb_map = load_imdb_to_tmdb_mapping()
    print(f"  Loaded {len(tmdb_to_imdb_map)} TMDB->IMDB mappings")
    print(f"  Loaded {len(imdb_to_tmdb_map)} IMDB->TMDB mappings")

    # For each ground truth entry (each entry represents a single movie and the movies that should be similar)
    # 1. Get the fused results
    # 2. Compute hits, rr, and ndcg for each vector space (+ overall)
    # 3. Save to broader results list
    # 4. Calculate averages for each vector space (+ overall)
    # 5. Return averages and per-movie results

    print("Step 3: Running evaluation...")
    print(f"  Evaluating {len(gt_data)} movies")

    per_movie_results: dict[str, PerMovieResult] = {}
    for query_id, entry in tqdm.tqdm(gt_data.items(), desc="Evaluating movies"):
        # Query ID is a TMDB ID (as string) from load_ground_truth
        tmdb_id = query_id
        
        # Convert TMDB ID to IMDB ID for fused_vector_search
        try:
            imdb_id = convert_tmdb_to_imdb_id(tmdb_id, tmdb_to_imdb_map)
        except ValueError as e:
            print(f"  Warning: {e}. Skipping movie {tmdb_id}")
            continue
        
        # Extract matches for each axis and convert IDs to strings
        # Also build truth maps (id -> score) and truth sets (ids) for metrics
        overall_matches = entry['overall_matches']
        content_matches = entry['content_matches']
        vibes_matches = entry['vibes_matches']
        
        # Build truth maps (id -> score) and truth sets (ids) for each axis
        # Convert all IDs from int to string for ChromaDB compatibility
        overall_truth_map = {str(match['id']): match['score'] for match in overall_matches}
        overall_truth_set = set(overall_truth_map.keys())
        
        content_truth_map = {str(match['id']): match['score'] for match in content_matches}
        content_truth_set = set(content_truth_map.keys())
        
        vibes_truth_map = {str(match['id']): match['score'] for match in vibes_matches}
        vibes_truth_set = set(vibes_truth_map.keys())
        
        # For expected_top_k display, sort by score descending to show most relevant first
        # This maintains intuitive ordering for visualization
        overall_expected_ids = sorted(overall_truth_map.keys(), key=lambda x: overall_truth_map[x], reverse=True)
        content_expected_ids = sorted(content_truth_map.keys(), key=lambda x: content_truth_map[x], reverse=True)
        vibes_expected_ids = sorted(vibes_truth_map.keys(), key=lambda x: vibes_truth_map[x], reverse=True)

        # Fetch our calculated answers
        # Note: fused_vector_search expects IMDB ID, so we pass imdb_id
        calculated_results = fused_vector_search(
            query_movie_id=imdb_id,
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
        # Note: calculated_results contain IMDB IDs, but we need TMDB IDs for comparison
        # Convert IMDB IDs to TMDB IDs, filtering out any that don't have a mapping
        def convert_results(results: list) -> list[str]:
            """Helper function to convert IMDB IDs to TMDB IDs."""
            converted = []
            for result in results:
                tmdb_id = convert_imdb_to_tmdb_id(result['movie_id'], imdb_to_tmdb_map)
                if tmdb_id is not None:
                    converted.append(tmdb_id)
            return converted
        
        calculated_overall_ids = convert_results(calculated_results['fused_results'])
        calculated_anchor_ids = convert_results(calculated_results['raw_anchor'])
        calculated_content_ids = convert_results(calculated_results['raw_content'])
        calculated_vibe_ids = convert_results(calculated_results['raw_vibe'])

        # Compare how accurate our calculated answers are
        # Note: anchor uses overall matches (as per original code logic)
        expected_calculated_pairs = [
            ('anchor', overall_truth_set, overall_truth_map, overall_expected_ids, calculated_anchor_ids),
            ('content', content_truth_set, content_truth_map, content_expected_ids, calculated_content_ids),
            ('vibe', vibes_truth_set, vibes_truth_map, vibes_expected_ids, calculated_vibe_ids),
            ('overall', overall_truth_set, overall_truth_map, overall_expected_ids, calculated_overall_ids)
        ]

        evaluation_results: dict[str, SingleAxisResult] = {}
        for axis, truth_set, truth_map, expected_ids, calculated_ids in expected_calculated_pairs:
            # Calculate how far off we were
            hits = hits_at_k(calculated_ids, truth_set, k)
            rr = rr_at_k(calculated_ids, truth_set, k)
            ndcg = ndcg_at_k(calculated_ids, truth_map, k)

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
        # Use TMDB ID as the key to match the ground truth structure
        per_movie_results[tmdb_id] = evaluation_results

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
    # Paths relative to evaluations/ directory
    GT_PATH = Path(__file__).parent / "most_similar_movies_v2.json"
    DB_PATH = Path(__file__).parent.parent / "implementation" / "chroma_db"
    OUTPUT_PATH = Path(__file__).parent / "evaluation_report.json"
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

